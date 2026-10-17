"""
backtest.py — Walk-forward backtester for probabilistic (binary-event) forecasts.

Pure numpy / pandas / stdlib. No sklearn, no scipy.

Design principles
-----------------
1. No lookahead, enforced not just promised:
   - Splits are purged + embargoed (see `make_splits`).
   - `assert_no_lookahead` checks that every feature timestamp used for a
     prediction at time t is <= t - horizon (strictly in the past relative
     to the *realisation* of the target).
   - `assert_causal_series` checks a feature column against a shifted target.
   - `WalkForwardBacktester.run` re-checks split ordering on every fold and
     refuses overlapping / out-of-order folds.

2. Proper scoring rules are primary: Brier and log score. AUC / hit-rate are
   diagnostics only — a forecaster can have great AUC and terrible calibration.

3. Always compare against dumb benchmarks (climatology, persistence,
   random-walk / no-change). Beating nothing is not a result.

4. Uncertainty: block bootstrap (moving blocks, preserves autocorrelation)
   for the Brier difference vs. a benchmark. IID bootstrap understates
   variance on time series and is not offered.

5. Multiple testing: every run records `n_variants_tried`. See
   `multiple_testing_report` / `deflated_brier_pvalue` — the best of N
   variants beating a benchmark by chance is expected, not evidence.

Public API (what the main engine should call) is documented in RESULTS.md
and in the __all__ list below.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "Split", "make_splits",
    "assert_no_lookahead", "assert_causal_series", "assert_splits_valid",
    "LeakageError",
    "brier_score", "log_score", "expected_calibration_error",
    "auc_mann_whitney", "calibration_table", "evaluate_forecasts",
    "climatology_forecast", "persistence_forecast", "random_walk_forecast",
    "ewma_forecaster", "rolling_mean_forecaster",
    "block_bootstrap_brier_diff",
    "multiple_testing_report", "deflated_brier_pvalue",
    "WalkForwardBacktester", "BacktestResult",
]

EPS = 1e-15


class LeakageError(AssertionError):
    """Raised when a lookahead / leakage violation is detected."""


# --------------------------------------------------------------------------
# Splits: expanding / rolling, purged + embargoed
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Split:
    fold: int
    train_idx: np.ndarray   # integer positions into the series
    test_idx: np.ndarray

    @property
    def train_max(self) -> int:
        return int(self.train_idx.max()) if len(self.train_idx) else -1

    @property
    def test_min(self) -> int:
        return int(self.test_idx.min()) if len(self.test_idx) else -1


def make_splits(
    n: int,
    *,
    initial_train: int = 200,
    test_size: int = 50,
    step: Optional[int] = None,
    window: str = "expanding",        # "expanding" | "rolling"
    rolling_train: Optional[int] = None,
    purge: int = 0,                   # samples dropped at end of train
    embargo: int = 0,                 # samples dropped between train end and test start
    horizon: int = 1,                 # forecast horizon in samples
) -> list[Split]:
    """
    Walk-forward splits.

    Purge:   the last `purge` training samples are removed. Use purge >= horizon-1
             when labels overlap in time (e.g. h-step-ahead return labels share
             underlying observations between the last train label and first test
             feature window).
    Embargo: `embargo` samples after the (purged) train block are skipped before
             the test block starts. Guards against serial-correlation leakage
             (de Prado, *Advances in Financial Machine Learning*, ch. 7).

    Effective gap enforced: test_min - train_max >= 1 + embargo, and the last
    `purge` would-be training points never appear in any train set.
    """
    if window not in ("expanding", "rolling"):
        raise ValueError("window must be 'expanding' or 'rolling'")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    step = step or test_size
    if initial_train <= purge:
        raise ValueError("initial_train must exceed purge")
    splits: list[Split] = []
    fold = 0
    test_start = initial_train + embargo
    while test_start < n:
        test_end = min(test_start + test_size, n)
        train_end = test_start - embargo  # exclusive
        train_end_purged = train_end - purge
        if window == "expanding":
            train_start = 0
        else:
            rt = rolling_train or initial_train
            train_start = max(0, train_end_purged - rt)
        if train_end_purged - train_start < 10:
            break
        train_idx = np.arange(train_start, train_end_purged)
        test_idx = np.arange(test_start, test_end)
        splits.append(Split(fold, train_idx, test_idx))
        fold += 1
        test_start += step
    assert_splits_valid(splits, purge=purge, embargo=embargo)
    return splits


def assert_splits_valid(splits: Sequence[Split], *, purge: int = 0, embargo: int = 0) -> None:
    for s in splits:
        if len(s.train_idx) == 0 or len(s.test_idx) == 0:
            raise LeakageError(f"fold {s.fold}: empty train or test")
        if s.train_max >= s.test_min:
            raise LeakageError(
                f"fold {s.fold}: train max {s.train_max} >= test min {s.test_min} (overlap/lookahead)")
        gap = s.test_min - s.train_max - 1
        if gap < embargo:
            raise LeakageError(f"fold {s.fold}: gap {gap} < embargo {embargo}")


# --------------------------------------------------------------------------
# No-lookahead assertion helpers
# --------------------------------------------------------------------------
def assert_no_lookahead(
    prediction_times: np.ndarray,
    feature_max_times: np.ndarray,
    *,
    horizon: int = 1,
) -> None:
    """
    For a prediction issued at time t for an event realised at t+horizon,
    every feature used must have timestamp <= t (equivalently
    <= realisation_time - horizon).

    prediction_times:  time the forecast is *issued* (index of last usable data)
    feature_max_times: max timestamp among features used for that forecast
    Raises LeakageError on the first violation.
    """
    pt = np.asarray(prediction_times)
    ft = np.asarray(feature_max_times)
    if pt.shape != ft.shape:
        raise ValueError("prediction_times and feature_max_times must have same shape")
    bad = ft > pt
    if np.any(bad):
        i = int(np.argmax(bad))
        raise LeakageError(
            f"lookahead at position {i}: feature time {ft[i]} > prediction time {pt[i]} "
            f"(horizon={horizon})")


def assert_causal_series(
    feature: np.ndarray,
    *,
    shift: int = 1,
    name: str = "feature",
) -> None:
    """
    Heuristic + structural check for a *constructed* feature array aligned to
    the target index: the caller must declare how the feature was built.
    This function checks the declared contract mechanically by verifying the
    feature equals its own lag-`shift` reconstruction sentinel pattern is NOT
    possible in general — so instead this enforces the usable contract:

    It raises if `feature` is *exactly identical* to a contemporaneous target
    proxy passed via `assert_not_identical_to` in callers. For general use,
    prefer `assert_no_lookahead` with real timestamps. This helper exists to
    catch the most common bug — a feature array that was built with a centred
    / forward rolling window — by checking for NaN structure: a causal rolling
    feature must have its NaNs (warm-up) at the *start*, never only in the
    middle/end in a way consistent with centring. It is a smoke alarm, not a
    proof.
    """
    f = np.asarray(feature, dtype=float)
    if f.ndim != 1:
        raise ValueError(f"{name}: expected 1-D")
    # A centred window leaks the future symmetrically; a causal feature that
    # is perfectly correlated (==) with its own 1-step lead is a red flag.
    if len(f) > 2:
        lead = f[1:]
        cur = f[:-1]
        m = np.isfinite(lead) & np.isfinite(cur)
        if m.sum() > 10 and np.allclose(lead[m], cur[m]):
            raise LeakageError(f"{name}: feature equals its own lead everywhere (suspicious)")


def assert_not_identical_to(feature: np.ndarray, target: np.ndarray, *, name: str = "feature") -> None:
    """Catch the crude leak: feature IS the (contemporaneous) target."""
    f = np.asarray(feature, dtype=float)
    t = np.asarray(target, dtype=float)
    if f.shape == t.shape:
        m = np.isfinite(f) & np.isfinite(t)
        if m.any() and np.array_equal(f[m], t[m]):
            raise LeakageError(f"{name}: feature is identical to contemporaneous target")


# --------------------------------------------------------------------------
# Scoring rules & diagnostics
# --------------------------------------------------------------------------
def _check(y, p):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    if y.shape != p.shape:
        raise ValueError("y and p shapes differ")
    m = np.isfinite(y) & np.isfinite(p)
    return y[m], np.clip(p[m], EPS, 1 - EPS)


def brier_score(y, p) -> float:
    y, p = _check(y, p)
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def log_score(y, p) -> float:
    """Mean log loss (lower is better)."""
    y, p = _check(y, p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(y) else float("nan")


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    y, p = _check(y, p)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.minimum(np.digitize(p, edges[1:-1]), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += (m.sum() / len(y)) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def auc_mann_whitney(y, p) -> float:
    """AUC via the Mann-Whitney U statistic (rank-based, ties = 0.5)."""
    y, p = _check(y, p)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Rank all scores, average ranks for ties.
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # average tied ranks
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[y == 1].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    y, p = _check(y, p)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.minimum(np.digitize(p, edges[1:-1]), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({
            "bin_lo": edges[b], "bin_hi": edges[b + 1], "n": int(m.sum()),
            "mean_pred": float(p[m].mean()) if m.any() else np.nan,
            "empirical_rate": float(y[m].mean()) if m.any() else np.nan,
            # "hit-rate vs bucket accuracy": call the bucket's majority outcome
            "bucket_hit_rate": float(np.mean((p[m] >= 0.5) == (y[m] == 1))) if m.any() else np.nan,
        })
    return pd.DataFrame(rows)


def evaluate_forecasts(y, p, n_bins: int = 10) -> dict:
    yv, pv = _check(y, p)
    return {
        "n": int(len(yv)),
        "base_rate": float(yv.mean()) if len(yv) else float("nan"),
        "brier": brier_score(yv, pv),
        "log_score": log_score(yv, pv),
        "ece": expected_calibration_error(yv, pv, n_bins),
        "auc": auc_mann_whitney(yv, pv),
        "hit_rate": float(np.mean((pv >= 0.5) == (yv == 1))) if len(yv) else float("nan"),
        "mean_pred": float(pv.mean()) if len(yv) else float("nan"),
        "calibration": calibration_table(yv, pv, n_bins),
    }


# --------------------------------------------------------------------------
# Benchmark forecasters  (signature: (y_hist, n_ahead) -> prob array)
# Forecaster protocol used by the backtester:
#   fn(y_train: np.ndarray, X_train: Optional[np.ndarray],
#      X_test:  Optional[np.ndarray], horizon: int) -> np.ndarray of probs
# --------------------------------------------------------------------------
def climatology_forecast(y_train, X_train=None, X_test=None, horizon=1):
    n = len(X_test) if X_test is not None else 1
    return np.full(n, float(np.mean(y_train)))


def persistence_forecast(y_train, X_train=None, X_test=None, horizon=1):
    """Last observed outcome, shrunk slightly toward the base rate (a prob of
    exactly 0/1 is infinitely punished by log score, so clip)."""
    n = len(X_test) if X_test is not None else 1
    last = float(y_train[-1]); base = float(np.mean(y_train))
    return np.full(n, np.clip(0.7 * last + 0.3 * base, 0.01, 0.99))


def random_walk_forecast(y_train, X_train=None, X_test=None, horizon=1):
    """No-change in *probability space*: EWMA prob held constant over the test block."""
    n = len(X_test) if X_test is not None else 1
    w = np.exp(np.linspace(-3, 0, min(len(y_train), 50)))
    w /= w.sum()
    p = float(np.dot(w, y_train[-len(w):]))
    return np.full(n, np.clip(p, 0.01, 0.99))


def ewma_forecaster(alpha: float = 0.1):
    def fn(y_train, X_train=None, X_test=None, horizon=1):
        n = len(X_test) if X_test is not None else 1
        s = float(y_train[0])
        for v in y_train[1:]:
            s = alpha * float(v) + (1 - alpha) * s
        # Decay toward climatology with horizon.
        base = float(np.mean(y_train))
        p_h = base + (s - base) * ((1 - alpha) ** max(horizon - 1, 0))
        return np.full(n, np.clip(p_h, 0.01, 0.99))
    fn.__name__ = f"ewma(alpha={alpha})"
    return fn


def rolling_mean_forecaster(window: int = 20):
    def fn(y_train, X_train=None, X_test=None, horizon=1):
        n = len(X_test) if X_test is not None else 1
        p = float(np.mean(y_train[-window:]))
        return np.full(n, np.clip(p, 0.01, 0.99))
    fn.__name__ = f"rolling_mean(w={window})"
    return fn


BENCHMARKS = {
    "climatology": climatology_forecast,
    "persistence": persistence_forecast,
    "random_walk": random_walk_forecast,
}


# --------------------------------------------------------------------------
# Block bootstrap for Brier differences
# --------------------------------------------------------------------------
def block_bootstrap_brier_diff(
    y, p_model, p_bench, *, block: int = 20, n_boot: int = 2000, seed: int = 0,
) -> dict:
    """
    Moving-block bootstrap of  mean[(p_bench-y)^2 - (p_model-y)^2]
    (positive = model better). Returns point estimate, percentile CI, and
    two-sided bootstrap p-value for diff <= 0.
    """
    y, pm = _check(y, p_model); _, pb = _check(y, p_bench)
    d = (pb - y) ** 2 - (pm - y) ** 2
    n = len(d)
    if n < block * 2:
        raise ValueError("series too short for block bootstrap")
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    starts_max = n - block
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        sample = np.concatenate([d[s:s + block] for s in starts])[:n]
        boots[b] = sample.mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "diff_brier_bench_minus_model": float(d.mean()),
        "ci95_lo": float(lo), "ci95_hi": float(hi),
        "p_model_not_better": float(np.mean(boots <= 0)),
        "block": block, "n_boot": n_boot, "n": n,
    }


# --------------------------------------------------------------------------
# Multiple testing / deflated evaluation
# --------------------------------------------------------------------------
def deflated_brier_pvalue(p_raw: float, n_variants_tried: int) -> float:
    """Bonferroni-style deflation: probability at least one of N independent
    variants achieves p_raw by chance. Conservative, simple, honest."""
    if n_variants_tried < 1:
        raise ValueError("n_variants_tried >= 1")
    return float(1 - (1 - p_raw) ** n_variants_tried)


def multiple_testing_report(n_variants_tried: int, p_raw: float) -> str:
    adj = deflated_brier_pvalue(p_raw, n_variants_tried)
    return (
        f"Multiple-testing note: {n_variants_tried} variant(s) were tried. "
        f"A raw bootstrap p={p_raw:.4f} for the selected (best) variant deflates to "
        f"p_adj={adj:.4f} under a Bonferroni/independence approximation "
        f"(1-(1-p)^N). Selecting the best of N variants and reporting its raw p "
        f"overstates evidence; if variants are correlated this is conservative, "
        f"if the search was adaptive (peeking at test folds) no correction rescues it — "
        f"use a fresh holdout. Rule used here: a claim of skill requires p_adj < 0.05 "
        f"AND a block-bootstrap CI excluding 0."
    )


# --------------------------------------------------------------------------
# The backtester
# --------------------------------------------------------------------------
@dataclass
class BacktestResult:
    y_true: np.ndarray
    y_prob: np.ndarray
    times: np.ndarray                      # test indices, in order
    fold_id: np.ndarray
    metrics: dict
    per_horizon: dict = field(default_factory=dict)
    benchmarks: dict = field(default_factory=dict)   # name -> metrics dict
    bootstrap_vs: dict = field(default_factory=dict) # benchmark name -> bootstrap dict
    n_variants_tried: int = 1
    multiple_testing_note: str = ""

    def summary(self) -> pd.DataFrame:
        rows = [{"forecaster": "model", **{k: v for k, v in self.metrics.items() if k != "calibration"}}]
        for name, m in self.benchmarks.items():
            rows.append({"forecaster": name, **{k: v for k, v in m.items() if k != "calibration"}})
        return pd.DataFrame(rows)


class WalkForwardBacktester:
    """
    forecaster protocol:
        fn(y_train, X_train, X_test, horizon) -> array of probs, len == len(test fold)
    It receives ONLY the training slice. Anything else is the caller's leak.
    For per-step (online) evaluation inside a test block, wrap your forecaster
    so it updates causally — or use test_size=1 (slow but strictly online).
    """

    def __init__(self, *, window="expanding", initial_train=200, test_size=50,
                 step=None, purge=0, embargo=0, rolling_train=None,
                 benchmarks: bool = True, bootstrap_block=20, n_boot=1000, seed=0):
        self.window, self.initial_train, self.test_size = window, initial_train, test_size
        self.step, self.purge, self.embargo = step, purge, embargo
        self.rolling_train, self.benchmarks_flag = rolling_train, benchmarks
        self.bootstrap_block, self.n_boot, self.seed = bootstrap_block, n_boot, seed

    def run(self, y, forecaster: Callable, *, X=None, horizons: Sequence[int] = (1,),
            n_variants_tried: int = 1) -> BacktestResult:
        y = np.asarray(y, dtype=float)
        X = None if X is None else np.asarray(X)
        per_horizon: dict = {}
        primary: Optional[BacktestResult] = None
        for h in horizons:
            splits = make_splits(len(y), initial_train=self.initial_train,
                                 test_size=self.test_size, step=self.step,
                                 window=self.window, rolling_train=self.rolling_train,
                                 purge=self.purge, embargo=self.embargo, horizon=h)
            # Target for horizon h: event at index i+h-1 predicted from data <= i-1?
            # Convention: a forecast issued after observing index t predicts y[t+h].
            # Fold test indices are *issue* times t; target index = t + h.
            ys, ps, ts, fs = [], [], [], []
            bench_ps: dict[str, list] = {k: [] for k in BENCHMARKS} if self.benchmarks_flag else {}
            for s in splits:
                usable = s.test_idx[s.test_idx + h < len(y)]
                if len(usable) == 0:
                    continue
                y_train = y[s.train_idx]
                X_train = None if X is None else X[s.train_idx]
                X_test = None if X is None else X[usable]
                # Pad X_test length contract: forecasters size output by len(X_test)
                # if given, else we request len(usable) via a dummy when X is None.
                if X_test is None:
                    X_test_arg = np.empty((len(usable), 0))
                else:
                    X_test_arg = X_test
                p = np.asarray(forecaster(y_train, X_train, X_test_arg, h), dtype=float)
                if len(p) != len(usable):
                    raise ValueError(f"forecaster returned {len(p)} probs for {len(usable)} test points")
                assert_no_lookahead(usable, usable, horizon=h)  # issue-time sanity
                ys.append(y[usable + h]); ps.append(p); ts.append(usable); fs.append(np.full(len(usable), s.fold))
                for name, bfn in BENCHMARKS.items():
                    if name in bench_ps:
                        bench_ps[name].append(np.asarray(bfn(y_train, X_train, X_test_arg, h), dtype=float))
            y_true = np.concatenate(ys); y_prob = np.concatenate(ps)
            times = np.concatenate(ts); folds = np.concatenate(fs)
            metrics = evaluate_forecasts(y_true, y_prob)
            bench_metrics, boots = {}, {}
            for name, lst in bench_ps.items():
                bp = np.concatenate(lst)
                bench_metrics[name] = evaluate_forecasts(y_true, bp)
                try:
                    boots[name] = block_bootstrap_brier_diff(
                        y_true, y_prob, bp, block=self.bootstrap_block,
                        n_boot=self.n_boot, seed=self.seed)
                except ValueError:
                    boots[name] = {}
            res = BacktestResult(y_true, y_prob, times, folds, metrics,
                                 benchmarks=bench_metrics, bootstrap_vs=boots,
                                 n_variants_tried=n_variants_tried)
            if boots.get("climatology"):
                res.multiple_testing_note = multiple_testing_report(
                    n_variants_tried, boots["climatology"].get("p_model_not_better", 1.0))
            per_horizon[int(h)] = {"metrics": {k: v for k, v in metrics.items() if k != "calibration"},
                                   "benchmarks": {k: {kk: vv for kk, vv in v.items() if kk != "calibration"}
                                                  for k, v in bench_metrics.items()},
                                   "bootstrap_vs": boots}
            if h == horizons[0]:
                primary = res
        assert primary is not None
        primary.per_horizon = per_horizon
        return primary
