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
    """1.0 if close[t+horizon] > close[t], else 0.0 (NaN in the tail).

    Raises ValueError naming ``horizon`` if it is not a positive
    integer (SF-BL-9848 value domain; SF-BL-11240 type domain: bool,
    string and non-integer-float horizons are rejected before any
    shift, instead of True silently building horizon=1 labels or
    '5'/2.5 dying with a bare TypeError).
    """
    _validate_horizon(horizon)
    c = df["close"].astype(float)
    return (c.shift(-horizon) > c).astype(float).where(c.shift(-horizon).notna())


def make_return_label(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Log-return log(close[t+horizon] / close[t]) (NaN in the tail).

    Raises ValueError naming ``horizon`` under the same guard as
    :func:`make_labels` (SF-BL-9848 / SF-BL-11240).
    """
    _validate_horizon(horizon)
    c = df["close"].astype(float)
    return np.log(c.shift(-horizon) / c)


def _validate_horizon(horizon: int) -> None:
    # horizon must be a positive integer (SF-BL-9848 value domain +
    # SF-BL-11240 type domain): horizon=0 would compare each close to
    # itself (degenerate all-zero / log(1)=0 target) and horizon<0 would
    # shift backwards in time (labels from past closes, reversed-time
    # leakage). The TYPE screen runs first: bool is a subclass of int,
    # so horizon=True silently built labels as horizon=1
    # (shift(-True) == shift(-1)), horizon='5' died with a bare
    # TypeError that never named horizon, and horizon=2.5 (or even the
    # integral float 2.0, which has no __index__) died inside pandas
    # with a bare slice-indices TypeError. Strict Integral-only,
    # mirroring calibration.py's _validate_n_bins (SF-BL-9722): only
    # int / np.integer are lawful; bools (incl. np.bool_), strings /
    # bytes, floats (non-integer and integral alike, incl. NaN/inf),
    # None and every other non-integer type raise the same named
    # ValueError. Check before any shift/rolling so no degenerate
    # label is ever constructed. Identical to the fleet copy's helper
    # (SF-BL-9848 / SF-BL-11240); originally defined here for
    # SF-BL-9849's make_crash_label guard, now also wired into this
    # copy's make_labels / make_return_label by SF-BL-11240.
    if isinstance(horizon, (bool, np.bool_)) or not isinstance(
        horizon, (int, np.integer)
    ):
        raise ValueError(
            f"horizon must be a positive integer (>= 1) (SF-BL-11240); "
            f"got horizon={horizon!r}"
        )
    if horizon < 1:
        raise ValueError(
            f"horizon must be a positive integer (>= 1) (SF-BL-11240); "
            f"got horizon={horizon!r}"
        )


def _validate_crash_threshold(threshold: float) -> float:
    # TYPE / FINITENESS screen (SF-BL-11239), run BEFORE the domain
    # check below. Previously threshold was never type-validated at
    # all — the SF-BL-9849 domain comparison below only incidentally
    # caught some bad types: threshold=False was read as 0.0 and
    # returned a label series silently (False lies in (-1, 0]),
    # threshold=Decimal('NaN') died with a bare decimal
    # InvalidOperation that never named threshold, and an array
    # threshold died inside pandas with a lengths-must-match error.
    # The task's live repros on the pre-9849 code were worse still:
    # threshold=nan returned an all-zero series (NaN comparisons are
    # all False — no crash ever labelled), threshold=True returned
    # an all-crash series (read as 1.0 by the comparison), and
    # threshold='0.1' died with a bare pandas 'Invalid comparison'
    # TypeError. A bool / string / None / non-finite threshold is
    # not a threshold — it is evidence of a corrupt caller, so it
    # raises a ValueError naming threshold. Pure validation: a
    # lawful threshold is returned as the identical float value, so
    # finite numeric-threshold label series are bit-identical.
    type_msg = (
        "threshold must be a finite non-bool real number "
        f"(SF-BL-11239); got threshold={threshold!r}"
    )
    if isinstance(threshold, (bool, np.bool_, str, bytes)):
        raise ValueError(type_msg)
    if threshold is None:
        raise ValueError(type_msg)
    if isinstance(threshold, (list, tuple, dict, set, frozenset, pd.Series)):
        raise ValueError(type_msg)
    if isinstance(threshold, np.ndarray) and (
        threshold.ndim > 0 or threshold.dtype == bool
    ):
        raise ValueError(type_msg)
    if isinstance(threshold, complex):
        raise ValueError(type_msg)
    try:
        t = float(threshold)
    except (TypeError, ValueError, OverflowError, ArithmeticError):
        raise ValueError(type_msg) from None
    if not np.isfinite(t):
        raise ValueError(type_msg)
    # threshold must lie in (-1, 0] (SF-BL-9849): a crash threshold is a
    # fractional drop, so threshold <= -1 demands a >100% fall (impossible
    # for positive closes — a silently all-zero target), threshold > 0
    # labels mere non-rises / rises as crashes, and a non-finite threshold
    # (NaN comparisons are all False) silently labels nothing a crash.
    if not (-1.0 < t <= 0.0):
        raise ValueError(
            f"threshold must lie in (-1, 0], got threshold={threshold!r}"
        )
    return t


def make_crash_label(df: pd.DataFrame, horizon: int = CRASH_HORIZON,
                     threshold: float = CRASH_THRESHOLD) -> pd.Series:
    """1 if min close over (t, t+horizon] is >10% below close[t].

    Raises
    ------
    ValueError
        If ``horizon`` < 1 or ``threshold`` lies outside (-1, 0]
        (SF-BL-9849). Both checks run before any rolling operation:
        horizon=0 previously returned an all-NaN series silently (a
        silently empty crash-training set) and horizon<0 died inside
        pandas with a bare ``min_periods must be >= 0`` error that
        never named horizon.
    ValueError
        If ``threshold`` is a bool, string, None or non-finite
        (SF-BL-11239; see :func:`_validate_crash_threshold`). The
        type/finiteness screen runs before the domain check and
        before any rolling operation, so threshold=False can no
        longer slip through as 0.0 and threshold=nan can no longer
        silently label nothing a crash; lawful finite numeric
        thresholds produce bit-identical label series.
    """
    _validate_horizon(horizon)
    threshold = _validate_crash_threshold(threshold)
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

def _has_string(raw) -> bool:
    """True iff ``raw`` contains any string value (SF-BL-11135).

    Covers a bare ``str`` / ``bytes`` scalar (``np.str_`` /
    ``np.bytes_`` subclass them), lists / tuples at any nesting,
    ndarrays with a string dtype (kind ``U`` / ``S``), object-dtype
    ndarrays holding a string leaf, and pandas Series with a string
    / object dtype holding a string leaf. A plain numeric ndarray
    or Series cannot contain strings by construction, so it returns
    False without scanning elementwise.
    """
    if isinstance(raw, (str, bytes)):
        return True
    if isinstance(raw, np.ndarray):
        if raw.dtype.kind in ("U", "S"):
            return True
        if raw.dtype == object:
            return any(_has_string(v) for v in raw.ravel())
        return False
    if isinstance(raw, pd.Series):
        if raw.dtype.kind in ("U", "S"):
            return True
        if raw.dtype == object or str(raw.dtype) in ("string", "str"):
            return any(_has_string(v) for v in raw.tolist())
        return False
    if isinstance(raw, (list, tuple)):
        return any(_has_string(v) for v in raw)
    return False


def _reject_string_forecasts(p) -> None:
    """Raise a named ValueError if forecast panel ``p`` contains any
    string (SF-BL-11135).

    This screen MUST run on the raw input, before any
    ``np.asarray(..., dtype=float)`` conversion or raw sorting:
    float conversion silently parses numeric strings
    (``brier(['0.3', '0.7'], [0, 1])`` returned the clean-control
    0.0900) and sorting a raw string array orders it lexicographically
    (``auc(['0.3', '0.7'], [0, 1])`` returned 1.0), so a JSON/text-
    sourced panel was graded as if it were a validated numeric
    panel. A numeric string is not a number — the panel's type is
    itself evidence of a corrupt feed. Pure validation — it never
    transforms ``p`` — so lawful numeric panels are bit-identical.
    """
    if _has_string(p):
        raise ValueError(
            "p must not contain string values; got string value(s) in p "
            "(a string is not a valid p value) (SF-BL-11135)"
        )


def _has_string_or_object_outcome(raw) -> bool:
    """True iff outcome panel ``raw`` contains a string element, a
    string-dtype element, or a bare ``object()`` element
    (SF-BL-11390).

    Extends :func:`_has_string` (which already covers bare
    ``str`` / ``bytes``, lists / tuples at any nesting, ndarrays
    with a string dtype (kind ``U`` / ``S``), object-dtype
    ndarrays holding a string leaf, and pandas Series with a
    string / object dtype holding a string leaf) to the outcome
    panel's remaining containers: pandas Index / Categorical /
    extension arrays, category-dtype Series (categories scanned
    too), and bare ``object()`` leaves at any nesting. An
    object-dtype panel carrying only numeric leaves is NOT
    flagged — ``np.array([0, 1], dtype=object)`` stays a lawful
    numeric outcome panel, bit-identical.
    """
    if _has_string(raw):
        return True
    if type(raw) is object:
        return True
    if isinstance(raw, np.ndarray):
        if raw.dtype == object:
            return any(_has_string_or_object_outcome(v) for v in raw.ravel())
        return False
    if isinstance(raw, pd.Series):
        if raw.dtype.kind in ("U", "S") or str(raw.dtype) in ("string", "str"):
            return True
        if str(raw.dtype) == "category":
            return (
                _has_string_or_object_outcome(raw.cat.categories)
                or any(_has_string_or_object_outcome(v) for v in raw.tolist())
            )
        if raw.dtype == object:
            return any(_has_string_or_object_outcome(v) for v in raw.tolist())
        return False
    if isinstance(raw, pd.Index):
        if raw.dtype.kind in ("U", "S") or str(raw.dtype) in ("string", "str"):
            return True
        if hasattr(raw, "categories"):
            return (
                _has_string_or_object_outcome(raw.categories)
                or any(_has_string_or_object_outcome(v) for v in raw.tolist())
            )
        if raw.dtype == object:
            return any(_has_string_or_object_outcome(v) for v in raw.tolist())
        return False
    if isinstance(raw, pd.Categorical):
        return (
            any(_has_string_or_object_outcome(v) for v in raw.categories)
            or any(_has_string_or_object_outcome(v) for v in list(raw))
        )
    if isinstance(raw, pd.api.extensions.ExtensionArray):
        if raw.dtype.kind in ("U", "S") or str(raw.dtype) in ("string", "str"):
            return True
        return any(_has_string_or_object_outcome(v) for v in list(raw))
    if isinstance(raw, (list, tuple)):
        return any(_has_string_or_object_outcome(v) for v in raw)
    return False


def _reject_string_outcomes(y) -> None:
    """Raise a named ValueError if outcome panel ``y`` contains any
    string or object-dtype string/object element (SF-BL-11390).

    This screen MUST run on the raw input, before
    ``np.asarray(..., dtype=float)`` conversion in
    :func:`_validate_brier_inputs`: float conversion silently
    parses numeric strings, so ``brier([0.3, 0.7], ['0', '1'])``
    returned 0.09, bit-identical to the numeric-outcome control,
    and the coerced ``(y == 0) | (y == 1)`` binary check never saw
    the panel's type. A numeric string is not an outcome — the
    panel's type is itself evidence of a corrupt feed. Pure
    validation — it never transforms ``y`` — so lawful numeric
    0/1 outcome panels (including object-dtype panels carrying
    only numeric leaves) are bit-identical.
    """
    if _has_string_or_object_outcome(y):
        raise ValueError(
            "y must not contain string values; got string or "
            "object-dtype value(s) in y (a string is not a valid y "
            "value) (SF-BL-11390)"
        )


def _has_bool(raw) -> bool:
    """True iff ``raw`` contains any bool value (SF-BL-11160).

    Covers a bare ``bool`` / ``np.bool_`` scalar, lists / tuples at
    any nesting, ndarrays with a bool dtype (kind ``b``), object-
    dtype ndarrays holding a bool leaf, and pandas Series with a
    bool dtype (kind ``b`` covers both numpy bool and the nullable
    ``boolean`` dtype) or an object dtype holding a bool leaf. A
    plain int / float ndarray or Series cannot contain bools by
    construction, so it returns False without scanning elementwise
    — ints are NOT bools (``1`` / ``0`` panels stay lawful) even
    though ``bool`` subclasses ``int`` in Python.
    """
    if isinstance(raw, (bool, np.bool_)):
        return True
    if isinstance(raw, np.ndarray):
        if raw.dtype.kind == "b":
            return True
        if raw.dtype == object:
            return any(_has_bool(v) for v in raw.ravel())
        return False
    if isinstance(raw, pd.Series):
        if raw.dtype.kind == "b":
            return True
        if raw.dtype == object or str(raw.dtype) in ("boolean", "bool"):
            return any(_has_bool(v) for v in raw.tolist())
        return False
    if isinstance(raw, (list, tuple)):
        return any(_has_bool(v) for v in raw)
    return False


def _reject_bool_forecasts(p) -> None:
    """Raise a named ValueError if forecast panel ``p`` contains any
    bool (SF-BL-11160).

    This screen MUST run on the raw input, before ``np.clip`` or any
    float conversion: clip / ``np.asarray(..., dtype=float)`` silently
    maps ``True`` -> 1.0 and ``False`` -> 0.0, so
    ``log_score([True, False], [1, 0])`` returned -1.0e-12 — a
    near-perfect log-likelihood bit-comparable to scoring the lawful
    endpoint panel ``[1.0, 0.0]`` — grading boolean flags as if they
    were validated endpoint probabilities. A bool is not a
    probability — the panel's type is itself evidence of a corrupt
    feed. Pure validation — it never transforms ``p`` — so lawful
    numeric (float or int) panels are bit-identical.
    """
    if _has_bool(p):
        raise ValueError(
            "p must not contain bool values; got bool value(s) in p "
            "(a bool is not a valid p value) (SF-BL-11160)"
        )


def _validate_brier_inputs(p, y) -> tuple[np.ndarray, np.ndarray]:
    """Validate a brier() panel and return it as flat float arrays.

    SF-BL-10065: brier was a bare ``mean((p - y) ** 2)`` with no screen at
    all — brier([NaN, 0.7], [0, 1]) returned NaN, brier([], []) returned
    NaN, brier([0.5, 0.6], [1]) returned 0.2050 by broadcasting one
    outcome across two forecasts, brier([1.5], [1]) scored an impossible
    forecast 0.25, and brier([0.5], [2]) returned an impossible 2.25.
    These values grade the crypto crash/direction books feeding CWR
    reads, so an invalid panel must fail loudly, never score silently.
    The screen mirrors the calibration.py brier_score standard
    (SF-BL-9484: same-length, binary outcomes, finite forecasts in
    [0, 1]) and adds the empty-panel rejection this task's prereg
    requires (calibration.py's brier_score still returns NaN on an
    empty panel). The check is pure validation — it never transforms
    lawful values — so clean panels score bit-identically.
    SF-BL-11390: the string screen now also runs on the raw outcome
    panel ``y`` (see :func:`_reject_string_outcomes`) before the
    float coercion below could parse ``'0'`` / ``'1'`` into lawful
    outcomes and hide the panel's type from the binary check.
    """
    _reject_string_forecasts(p)  # SF-BL-11135: raw screen, before float conversion reads '0.3' as 0.3
    _reject_string_outcomes(y)  # SF-BL-11390: raw screen, before float conversion reads '0'/'1' as 0/1
    try:
        p = np.asarray(p, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"brier: p and y must be numeric arrays ({exc})") from exc
    if p.shape[0] != y.shape[0]:
        raise ValueError(
            f"brier: length mismatch: len(p)={p.shape[0]} != len(y)={y.shape[0]}; "
            "forecasts and outcomes must pair one-to-one"
        )
    if p.shape[0] == 0:
        raise ValueError(
            "brier: empty panel: at least one forecast/outcome pair is required"
        )
    if not np.all(np.isfinite(p)):
        raise ValueError(
            "brier: forecasts p must be finite probabilities in [0, 1]; "
            "got non-finite value(s)"
        )
    lo, hi = float(p.min()), float(p.max())
    if lo < 0.0 or hi > 1.0:
        raise ValueError(
            f"brier: forecasts p must lie in [0, 1]; got values outside "
            f"that range (min={lo!r}, max={hi!r})"
        )
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("brier: outcomes y must be binary 0/1")
    return p, y


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error between forecast probability and outcome.

    Raises
    ------
    ValueError
        If the panel is empty, if ``p`` and ``y`` have different
        lengths, if any forecast is non-finite or outside [0, 1], or
        if any outcome is not binary 0/1 (SF-BL-10065), or if
        forecast panel ``p`` contains any string value
        (SF-BL-11135), or if outcome panel ``y`` contains any
        string or object-dtype string/object value (SF-BL-11390).
        Validation
        happens before scoring; lawful panels are scored exactly as
        before (``float(mean((p - y) ** 2))``).
    """
    p, y = _validate_brier_inputs(p, y)
    return float(np.mean((p - y) ** 2))


def _validate_log_score_eps(eps) -> float:
    """Validate ``log_score``'s ``eps`` clip bound type (SF-BL-11162).

    Returns the bound as a float. Pure validation — a valid bound is
    returned unchanged in value, so scores for lawful eps are
    bit-identical to the pre-fix computation.

    Raises
    ------
    ValueError
        Unless ``eps`` is a finite non-bool real number. Previously
        ``eps`` was never type-validated: ``eps=True`` was consumed
        by ``np.clip`` as the bounds [True, 0] and returned NaN on a
        clean panel (vs -0.35667 at the default ``eps=1e-12``), and
        ``eps='1e-12'`` / ``eps=None`` died inside the clip
        arithmetic with bare ``TypeError: unsupported operand``
        errors that never named ``eps``. Booleans (including
        ``np.bool_`` and 0-d bool arrays), strings/bytes, None,
        sequences/mappings, pandas Series, complex values and
        non-finite floats (NaN/inf are not real numbers) raise the
        same named ValueError. This is the TYPE screen only: the
        value domain ``0 < eps < 0.5`` is SF-BL-9817's separate
        guard and is not imposed here.
    """
    msg = ("eps must be a finite non-bool real number (SF-BL-11162); "
           f"got eps={eps!r}")
    if isinstance(eps, (bool, np.bool_, str, bytes)):
        raise ValueError(msg)
    if isinstance(eps, (list, tuple, dict, set, frozenset, pd.Series)):
        raise ValueError(msg)
    if isinstance(eps, np.ndarray) and (eps.ndim > 0 or eps.dtype == bool):
        raise ValueError(msg)
    if isinstance(eps, complex):
        raise ValueError(msg)
    try:
        e = float(eps)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(msg) from None
    if not np.isfinite(e):
        raise ValueError(msg)
    return e


def _validate_log_score_inputs(p, y) -> tuple[np.ndarray, np.ndarray]:
    """Validate a log_score() panel and return it as flat float arrays.

    SF-BL-11206: log_score clipped and logged the raw arrays with no
    panel screen at all, unlike brier() in the same file
    (SF-BL-10065) — log_score([1.5], [1]) returned
    -9.999778782803785e-13, a near-perfect log-likelihood for an
    impossible forecast, an empty panel and a NaN forecast both
    returned NaN, a length-mismatched panel broadcast one outcome
    across two forecasts (log_score([0.5, 0.6], [1]) returned
    -0.601986402162968), and a non-binary outcome y=2 was scored as
    lawful. These values grade the crypto crash/direction books
    feeding CWR reads, so an invalid panel must fail loudly, never
    score silently. The screen mirrors _validate_brier_inputs
    (SF-BL-10065, itself the calibration.py brier_score standard
    SF-BL-9484 plus the empty-panel rejection): same-length, empty,
    finite forecasts in [0, 1], binary outcomes. The string screen
    (SF-BL-11135) runs on the raw forecast panel before the float
    conversion below, so the conversion cannot silently parse a
    numeric string ('0.3') into a lawful forecast — the same reason
    brier screens strings before converting. The check is pure
    validation — it never transforms lawful values — so clean
    panels score bit-identically. This is the PANEL screen only:
    the bool-forecast screen is SF-BL-11160 and the eps guards are
    SF-BL-9817 / SF-BL-11162, all separate and preserved.
    """
    _reject_string_forecasts(p)  # SF-BL-11135: raw screen, before float conversion reads '0.3' as 0.3
    try:
        p = np.asarray(p, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"log_score: p and y must be numeric arrays ({exc})") from exc
    if p.shape[0] != y.shape[0]:
        raise ValueError(
            f"log_score: length mismatch: len(p)={p.shape[0]} != len(y)={y.shape[0]}; "
            "forecasts and outcomes must pair one-to-one"
        )
    if p.shape[0] == 0:
        raise ValueError(
            "log_score: empty panel: at least one forecast/outcome pair is required"
        )
    if not np.all(np.isfinite(p)):
        raise ValueError(
            "log_score: forecasts p must be finite probabilities in [0, 1]; "
            "got non-finite value(s)"
        )
    lo, hi = float(p.min()), float(p.max())
    if lo < 0.0 or hi > 1.0:
        raise ValueError(
            f"log_score: forecasts p must lie in [0, 1]; got values outside "
            f"that range (min={lo!r}, max={hi!r})"
        )
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("log_score: outcomes y must be binary 0/1")
    return p, y


def log_score(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Mean log score (higher is better), forecasts clipped to [eps, 1-eps].

    Raises
    ------
    ValueError
        If the panel is empty, if ``p`` and ``y`` have different
        lengths, if any forecast is non-finite or outside [0, 1], or
        if any outcome is not binary 0/1 (SF-BL-11206; see
        :func:`_validate_log_score_inputs`). Validation happens
        BEFORE the internal clip: the clip to [eps, 1-eps] applies
        only to lawful endpoint forecasts 0/1, never to out-of-range
        values such as 1.5, which previously clipped to a
        near-perfect score; lawful panels are scored exactly as
        before.
    ValueError
        If forecast panel ``p`` contains any bool value
        (SF-BL-11160). The screen runs on the raw input before
        clipping could silently map True/False to the lawful
        endpoints 1.0/0.0; lawful numeric panels are scored exactly
        as before.
    ValueError
        If ``eps`` is not a finite non-bool real number
        (SF-BL-11162; see :func:`_validate_log_score_eps`).
        Previously ``eps=True`` returned NaN on a clean panel and
        ``eps='1e-12'`` / ``eps=None`` raised bare TypeErrors that
        never named ``eps``.
    """
    _reject_bool_forecasts(p)  # SF-BL-11160: raw screen, before np.clip maps True->1.0 / False->0.0
    eps = _validate_log_score_eps(eps)  # SF-BL-11162: type screen, before np.clip consumes eps
    p, y = _validate_log_score_inputs(p, y)  # SF-BL-11206: panel screen, before np.clip / np.log score it
    p = np.clip(p, eps, 1 - eps)
    return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(p: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), ties handled by average ranks.

    Raises
    ------
    ValueError
        If forecast panel ``p`` contains any string value
        (SF-BL-11135). The screen runs on the raw input before any
        sorting; lawful numeric panels are scored exactly as before.
    """
    _reject_string_forecasts(p)  # SF-BL-11135: raw screen, before sorting raw strings lexicographically
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
