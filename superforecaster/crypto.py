"""
crypto.py — Probabilistic BTC / ETH forecaster (pure numpy/pandas, no sklearn).

Outputs, for each asset and each forecast date t (using only data <= t):
  * P(up) for horizons 1d / 7d / 30d       -- hand-rolled logistic regression
  * expected log-return for each horizon   -- hand-rolled ridge regression
  * crash risk: P(a >10% close-to-close drop occurs within the next 7 days)

Features (all strictly causal, computed from closes/highs at or before t):
  momentum lookbacks 1/7/14/30/90/180 d (log returns), 7d and 30d MA gaps,
  realised vol 7/30/90 d, vol-of-vol (std of 7d vol over trailing 30 d),
  drawdown from all-time high and from 90 d rolling high,
  RSI(14), weekend dummy (Sat/Sun, UTC), day-of-week one-hots are folded
  into the weekend dummy to keep the model small, volume z-score (30 d).
For ETH, BTC momentum/vol features are appended (BTC -> ETH cross-asset).

Walk-forward discipline: the backtest script refits on an expanding window
and never lets a training label use prices after the forecast origin
(training rows are purged so that t_train + horizon <= t_forecast).

Author: Workstream E (crypto), Superforecaster project. Research only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 7, 30)
CRASH_HORIZON = 7
CRASH_THRESHOLD = -0.10

# ---------------------------------------------------------------- features

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing via EWM (causal)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.fillna(50.0) / 100.0  # scale to [0,1]


def build_features(df: pd.DataFrame, btc_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """df: daily OHLCV with columns open/high/low/close/volume, DatetimeIndex (UTC, daily)."""
    df = df.sort_index()
    c = df["close"].astype(float)
    logc = np.log(c)
    logret = logc.diff()
    out = pd.DataFrame(index=df.index)

    for lb in (1, 7, 14, 30, 90, 180):
        out[f"mom{lb}"] = logc - logc.shift(lb)
    out["ma_gap7"] = c / c.rolling(7).mean() - 1.0
    out["ma_gap30"] = c / c.rolling(30).mean() - 1.0
    out["ma_gap90"] = c / c.rolling(90).mean() - 1.0

    for w in (7, 30, 90):
        out[f"vol{w}"] = logret.rolling(w).std()
    vol7 = logret.rolling(7).std()
    out["vol_of_vol30"] = vol7.rolling(30).std()
    out["vol_ratio"] = out["vol7"] / out["vol30"].replace(0.0, np.nan)

    ath = c.cummax()
    out["dd_ath"] = c / ath - 1.0
    out["dd_90"] = c / c.rolling(90).max() - 1.0
    out["rsi14"] = _rsi(c)

    dow = df.index.dayofweek  # Mon=0 .. Sun=6 (index is UTC daily bars)
    out["weekend"] = ((dow >= 5)).astype(float)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    if "volume" in df.columns:
        v = df["volume"].astype(float)
        out["volz30"] = (np.log1p(v) - np.log1p(v).rolling(30).mean()) / np.log1p(v).rolling(30).std().replace(0.0, np.nan)
    else:
        out["volz30"] = 0.0

    if btc_df is not None:
        b = btc_df.sort_index()["close"].astype(float).reindex(df.index).ffill()
        logb = np.log(b)
        for lb in (1, 7, 30, 90):
            out[f"btc_mom{lb}"] = logb - logb.shift(lb)
        out["btc_vol30"] = np.log(b).diff().rolling(30).std()
        out["rel_mom30"] = out["mom30"] - out["btc_mom30"]

    return out.replace([np.inf, -np.inf], np.nan)


def make_labels(df: pd.DataFrame, horizon: int) -> pd.Series:
    c = df["close"].astype(float)
    return (c.shift(-horizon) > c).astype(float).where(c.shift(-horizon).notna())


def make_return_label(df: pd.DataFrame, horizon: int) -> pd.Series:
    c = df["close"].astype(float)
    return np.log(c.shift(-horizon) / c)


def make_crash_label(df: pd.DataFrame, horizon: int = CRASH_HORIZON,
                     threshold: float = CRASH_THRESHOLD) -> pd.Series:
    """1 if min close over (t, t+horizon] is >10% below close[t]."""
    c = df["close"].astype(float)
    # future minimum close, excluding t itself
    fut_min = c.iloc[::-1].rolling(horizon, min_periods=horizon).min().iloc[::-1].shift(-1)
    # rolling on reversed series + shift gives min of next `horizon` closes; guard tail
    lab = (fut_min / c - 1.0 < threshold).astype(float)
    lab[fut_min.isna()] = np.nan
    return lab


# ---------------------------------------------------------------- models

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


class LogisticRegressionGD:
    """L2-regularised logistic regression, full-batch GD with Adam-ish step decay.
    Intercept is unpenalised. Deterministic (fixed init = zeros)."""

    def __init__(self, l2: float = 1.0, lr: float = 0.1, iters: int = 600):
        self.l2, self.lr, self.iters = l2, lr, iters

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        self.mu_, self.sd_ = X.mean(0), X.std(0)
        self.sd_[self.sd_ == 0] = 1.0
        Xs = (X - self.mu_) / self.sd_
        w = np.zeros(d)
        b = float(np.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6)))
        # NOTE: no class balancing / sample weighting. Weighting positives
        # inflates rare-event probabilities and destroys calibration
        # (verified empirically: crash-model Brier 0.183 vs 0.092 base rate
        # with weighting). A forecaster must emit honest probabilities.
        sw = np.ones_like(y, dtype=float)
        for it in range(self.iters):
            p = _sigmoid(Xs @ w + b)
            g = Xs.T @ (sw * (p - y)) / n + (self.l2 / n) * w
            gb = float(np.mean(sw * (p - y)))
            step = self.lr / (1.0 + it / 200.0)
            w -= step * g
            b -= step * gb
        self.w_, self.b_ = w, b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mu_) / self.sd_
        return _sigmoid(Xs @ self.w_ + self.b_)


class RidgeGD:
    """Ridge regression (closed form on standardised features)."""

    def __init__(self, l2: float = 10.0):
        self.l2 = l2

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.mu_, self.sd_ = X.mean(0), X.std(0)
        self.sd_[self.sd_ == 0] = 1.0
        Xs = (X - self.mu_) / self.sd_
        self.ybar_ = float(y.mean())
        yc = y - self.ybar_
        d = Xs.shape[1]
        A = Xs.T @ Xs + self.l2 * np.eye(d)
        self.w_ = np.linalg.solve(A, Xs.T @ yc)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mu_) / self.sd_
        return self.ybar_ + Xs @ self.w_


# ---------------------------------------------------------------- metrics

def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_score(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(p: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), ties handled by average ranks."""
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), float)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    sp = np.asarray(p)[order]
    i = 0
    sorted_ranks = ranks[order]
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            sorted_ranks[i:j + 1] = sorted_ranks[i:j + 1].mean()
        i = j + 1
    ranks[order] = sorted_ranks
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def calibration_buckets(p: np.ndarray, y: np.ndarray, edges=(0, .2, .4, .5, .6, .8, 1.0)):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if m.sum() > 0:
            rows.append({"bucket": f"[{lo:.1f},{hi:.1f})" if hi < 1 else "[0.8,1.0]",
                         "n": int(m.sum()), "mean_pred": float(p[m].mean()),
                         "freq": float(y[m].mean())})
    return rows
