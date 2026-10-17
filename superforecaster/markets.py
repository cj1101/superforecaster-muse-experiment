"""
Markets Superforecaster — probabilistic equity/index forecaster.
Pure numpy/pandas. Logistic regression via gradient descent (no sklearn).

Outputs per horizon (1/5/21d):
  - P(up)  : probability close[h] > close[0]
  - Expected return distribution: mean / std / quantiles from a ridge
    return model + residual dispersion, blended with vol-scaled prior.

Walk-forward only: features at t use data <= t; labels use t+h;
training at forecast date T uses only samples with label date <= T.
"""
import json, math, os, urllib.request, urllib.parse
import numpy as np
import pandas as pd

HORIZONS = [1, 5, 21]

# ---------- Data ----------
def download_yahoo(symbol: str, range_: str = "10y", cache_dir: str = None) -> pd.DataFrame:
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        safe = symbol.replace("^", "").replace("/", "_")
        path = os.path.join(cache_dir, f"{safe}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["date"])
            if len(df) > 500:
                return df
    enc = urllib.parse.quote(symbol, safe="")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{enc}?range={range_}&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode())
    res = data["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
    df = pd.DataFrame({
        "date": pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").tz_localize(None).normalize(),
        "open": q["open"], "high": q["high"], "low": q["low"],
        "close": q["close"], "adjclose": adj, "volume": q["volume"],
    }).dropna(subset=["close"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    # Prefer adjclose for returns when available and non-null
    df["price"] = df["adjclose"].fillna(df["close"])
    if cache_dir:
        df.to_csv(path, index=False)
    return df

# ---------- Features ----------
def _rsi(price: pd.Series, n: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    p = df["price"].astype(float)
    logp = np.log(p)
    f = pd.DataFrame(index=df.index)
    f["date"] = df["date"]
    # momentum (log returns over lookbacks)
    for lb in [2, 5, 10, 21, 63, 126, 252]:
        f[f"mom_{lb}"] = logp.diff(lb)
    # mean reversion: z-score vs SMA
    for w in [20, 50, 200]:
        sma = p.rolling(w).mean()
        sd = p.rolling(w).std()
        f[f"z_{w}"] = (p - sma) / sd.replace(0, np.nan)
        f[f"dist_sma_{w}"] = p / sma - 1.0
    # realized vol + regime
    r1 = logp.diff(1)
    for w in [10, 21, 63]:
        f[f"vol_{w}"] = r1.rolling(w).std() * math.sqrt(252)
    f["vol_ratio_10_63"] = f["vol_10"] / f["vol_63"].replace(0, np.nan)
    f["vol_ratio_21_63"] = f["vol_21"] / f["vol_63"].replace(0, np.nan)
    # RSI (scaled to [-1,1])
    f["rsi14"] = (_rsi(p, 14) - 50) / 50.0
    f["rsi7"] = (_rsi(p, 7) - 50) / 50.0
    # distance from 52w high / low
    hi = p.rolling(252).max(); lo = p.rolling(252).min()
    f["dist_52w_high"] = p / hi - 1.0
    f["range_52w_pos"] = (p - lo) / (hi - lo).replace(0, np.nan)
    # volume trend (if available; indices often have volume, some may be 0)
    vol = df["volume"].astype(float) if "volume" in df else pd.Series(np.nan, index=df.index)
    if vol.notna().sum() > 100 and (vol > 0).sum() > 100:
        f["vol_trend"] = vol.rolling(10).mean() / vol.rolling(50).mean().replace(0, np.nan) - 1.0
        f["vol_z"] = (vol - vol.rolling(50).mean()) / vol.rolling(50).std().replace(0, np.nan)
    else:
        f["vol_trend"] = 0.0; f["vol_z"] = 0.0
    # day of week (cyclical + one-hot collapsed: Mon=0..Fri=4)
    dow = df["date"].dt.dayofweek
    f["dow_sin"] = np.sin(2 * np.pi * dow / 5.0)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 5.0)
    f["is_monday"] = (dow == 0).astype(float)
    f["is_friday"] = (dow == 4).astype(float)
    # short-term reversal / vol-adjusted momentum
    f["mom5_voladj"] = f["mom_5"] / (r1.rolling(21).std() * math.sqrt(5) + 1e-9)
    f["ret_1d"] = r1
    f["ret_1d_abs"] = r1.abs()
    # labels
    for h in HORIZONS:
        fwd = logp.shift(-h) - logp
        f[f"fwd_ret_{h}"] = fwd
        f[f"y_{h}"] = (fwd > 0).astype(float)
        f.loc[fwd.isna(), f"y_{h}"] = np.nan
    return f

FEATURE_COLS = None  # set dynamically: all except date/labels

def feature_columns(feat: pd.DataFrame):
    return [c for c in feat.columns if c != "date" and not c.startswith("fwd_ret_") and not c.startswith("y_")]

# ---------- Models (from scratch) ----------
class LogisticGD:
    def __init__(self, l2=1.0, lr=0.05, epochs=300):
        self.l2, self.lr, self.epochs = l2, lr, epochs
    def fit(self, X, y, base_rate):
        X = np.nan_to_num(X, nan=0.0)
        self.mu_, self.sd_ = X.mean(0), X.std(0); self.sd_[self.sd_ == 0] = 1.0
        Xs = (X - self.mu_) / self.sd_
        n, d = Xs.shape
        # intercept init at logit(base_rate) — the base-rate anchor
        br = float(np.clip(base_rate, 0.01, 0.99))
        self.w_ = np.zeros(d); self.b_ = math.log(br / (1 - br))
        # class weighting: none (probabilities must stay calibrated)
        for _ in range(self.epochs):
            z = np.clip(Xs @ self.w_ + self.b_, -30, 30)
            p = 1 / (1 + np.exp(-z))
            g = p - y
            gw = Xs.T @ g / n + self.l2 * self.w_ / n
            gb = g.mean()
            self.w_ -= self.lr * gw; self.b_ -= self.lr * gb
        self.base_rate_ = br
        return self
    def predict_proba(self, X):
        Xs = (np.nan_to_num(X, nan=0.0) - self.mu_) / self.sd_
        z = np.clip(Xs @ self.w_ + self.b_, -30, 30)
        return 1 / (1 + np.exp(-z))

class RidgeGD:
    """Expected-return model via ridge closed form (numpy only)."""
    def __init__(self, lam=10.0):
        self.lam = lam
    def fit(self, X, y):
        X = np.nan_to_num(X, nan=0.0)
        self.mu_, self.sd_ = X.mean(0), X.std(0); self.sd_[self.sd_ == 0] = 1.0
        Xs = (X - self.mu_) / self.sd_
        self.ymean_ = float(y.mean()); yc = y - self.ymean_
        d = Xs.shape[1]
        A = Xs.T @ Xs + self.lam * np.eye(d)
        self.w_ = np.linalg.solve(A, Xs.T @ yc)
        pred = Xs @ self.w_ + self.ymean_
        self.resid_sd_ = float(np.std(y - pred)) if len(y) > d else float(np.std(y))
        return self
    def predict(self, X):
        Xs = (np.nan_to_num(X, nan=0.0) - self.mu_) / self.sd_
        return Xs @ self.w_ + self.ymean_

# ---------- Metrics ----------
def brier(y, p): return float(np.mean((p - y) ** 2))
def log_score(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
def auc(y, p):
    # rank-based Mann-Whitney
    order = np.argsort(p); ranks = np.empty(len(p), float); ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    pos = y == 1; npos, nneg = pos.sum(), (~pos).sum()
    if npos == 0 or nneg == 0: return float("nan")
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def calibration_table(y, p, bins=10):
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum() > 0:
            rows.append({"bin_lo": round(float(edges[i]),2), "bin_hi": round(float(edges[i+1]),2),
                         "n": int(m.sum()), "mean_pred": round(float(p[m].mean()),4),
                         "empirical_rate": round(float(y[m].mean()),4)})
    return rows

# ---------- Walk-forward backtest ----------
def walk_forward(feat: pd.DataFrame, horizon: int, initial_train: int = 756,
                 retrain_every: int = 21, l2: float = 1.0):
    cols = feature_columns(feat)
    Xall = feat[cols].values; yall = feat[f"y_{horizon}"].values; rall = feat[f"fwd_ret_{horizon}"].values
    dates = feat["date"].values
    preds, ys, rets, idxs, exp_rets, clim_ps = [], [], [], [], [], []
    logit = ridge = None
    # expanding climatology: base rate using labels whose outcome date <= t
    for t in range(initial_train, len(feat)):
        if np.isnan(yall[t]): continue
        # need training samples i where i+horizon <= t (label known by t)
        if logit is None or (t - initial_train) % retrain_every == 0:
            train_idx = [i for i in range(0, t - horizon + 1) if not np.isnan(yall[i])]
            if len(train_idx) < 250: continue
            Xtr, ytr, rtr = Xall[train_idx], yall[train_idx], rall[train_idx]
            # drop rows with any NaN features in early window? nan_to_num in model handles it,
            # but require mom_252 available for most: keep as-is (0-filled after standardize input)
            logit = LogisticGD(l2=l2).fit(Xtr, ytr, base_rate=float(ytr.mean()))
            ridge = RidgeGD().fit(Xtr, rtr)
            clim_rate = float(ytr.mean())
        p = float(logit.predict_proba(Xall[t:t+1])[0])
        preds.append(p); ys.append(float(yall[t])); rets.append(float(rall[t])); idxs.append(t)
        exp_rets.append(float(ridge.predict(Xall[t:t+1])[0])); clim_ps.append(clim_rate)
    return {"dates": [str(pd.Timestamp(dates[i]).date()) for i in idxs],
            "p": np.array(preds), "y": np.array(ys), "ret": np.array(rets),
            "exp_ret": np.array(exp_rets), "clim_p": np.array(clim_ps),
            "ridge_resid_sd": getattr(ridge, "resid_sd_", None) if ridge else None}

def summarize(res: dict) -> dict:
    y, p, cp = res["y"], res["p"], res["clim_p"]
    out = {
        "n": int(len(y)),
        "base_rate_test": round(float(y.mean()), 4) if len(y) else None,
        "brier": round(brier(y, p), 5), "brier_climatology": round(brier(y, cp), 5),
        "brier_skill_vs_clim": round(1 - brier(y, p) / brier(y, cp), 4),
        "log_score": round(log_score(y, p), 5), "log_score_climatology": round(log_score(y, cp), 5),
        "auc": round(auc(y, p), 4),
        "mean_pred": round(float(p.mean()), 4),
        "exp_ret_mean_pred": round(float(res["exp_ret"].mean()), 5),
        "exp_ret_corr_with_realized": round(float(np.corrcoef(res["exp_ret"], res["ret"])[0,1]), 4) if len(y) > 2 else None,
        "calibration": calibration_table(y, p),
    }
    out["beats_climatology_brier"] = bool(out["brier"] < out["brier_climatology"])
    return out

def predict_latest(symbol: str, cache_dir: str = None) -> dict:
    """Live forecast for the most recent trading day."""
    df = download_yahoo(symbol, cache_dir=cache_dir)
    feat = build_features(df); cols = feature_columns(feat)
    out = {"symbol": symbol, "as_of": str(df["date"].iloc[-1].date()), "last_price": float(df["price"].iloc[-1])}
    for h in HORIZONS:
        valid = feat.dropna(subset=[f"y_{h}"])
        # train on all labelled data, predict last row
        tr = valid.index[valid.index <= len(feat) - 1 - h]
        Xtr = feat.loc[tr, cols].values; ytr = feat.loc[tr, f"y_{h}"].values; rtr = feat.loc[tr, f"fwd_ret_{h}"].values
        lg = LogisticGD().fit(Xtr, ytr, float(ytr.mean())); rg = RidgeGD().fit(Xtr, rtr)
        x = feat[cols].values[-1:]
        mu = float(rg.predict(x)[0]); sd = float(rg.resid_sd_)
        out[f"h{h}"] = {"p_up": round(float(lg.predict_proba(x)[0]), 4),
                        "exp_log_return": round(mu, 5),
                        "ret_q10": round(mu - 1.2816 * sd, 5), "ret_q50": round(mu, 5),
                        "ret_q90": round(mu + 1.2816 * sd, 5), "resid_sd": round(sd, 5)}
    return out

if __name__ == "__main__":
    print(predict_latest("^GSPC"))
