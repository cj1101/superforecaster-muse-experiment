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
def _panel_type_defect(raw):
    """SF-BL-11433: return 'string' / 'bool' / 'object' if a RAW panel
    contains an element of that type, else None.

    Recurses through lists / tuples and object-dtype ndarrays, and
    screens pandas Series / Index / Categorical the same way. A
    string-DTYPE container (ndarray kind ``U``/``S``, pandas
    StringDtype, string categories) counts even when empty — its
    dtype is the defect, not any single element — as does a
    bool-dtype container (kind ``b``). An object-dtype container
    holding only plain numbers is lawful: its elements are scanned,
    and a leaf counts only if it is a string, a bool, a bare
    ``object()``, or any other non-numeric scalar. A plain numeric
    ndarray / Series cannot contain such elements by construction,
    so it returns None without scanning elementwise.
    """
    if isinstance(raw, (str, bytes)):
        return "string"
    if isinstance(raw, (bool, np.bool_)):
        return "bool"
    if type(raw) is object:
        return "object"
    if isinstance(raw, pd.Categorical):
        defect = _panel_type_defect(raw.categories)
        if defect is not None:
            return defect
        for v in raw:
            defect = _panel_type_defect(v)
            if defect is not None:
                return defect
        return None
    if isinstance(raw, np.ndarray):
        if raw.dtype.kind in "US":
            return "string"
        if raw.dtype.kind == "b":
            return "bool"
        if raw.dtype == object:
            for v in raw.ravel():
                defect = _panel_type_defect(v)
                if defect is not None:
                    return defect
                if not isinstance(v, (list, tuple, np.ndarray,
                                       pd.Series, pd.Index,
                                       pd.Categorical)) and not isinstance(
                        v, (int, float, complex, np.number)):
                    return "object"
        return None
    if isinstance(raw, (list, tuple)):
        for v in raw:
            defect = _panel_type_defect(v)
            if defect is not None:
                return defect
        return None
    if isinstance(raw, (pd.Series, pd.Index)):
        if isinstance(raw.dtype, pd.CategoricalDtype):
            cats = raw.dtype.categories
            if cats is not None:
                defect = _panel_type_defect(cats)
                if defect is not None:
                    return defect
            for v in raw:
                defect = _panel_type_defect(v)
                if defect is not None:
                    return defect
            return None
        if raw.dtype.kind in "US" or isinstance(raw.dtype, pd.StringDtype):
            return "string"
        if raw.dtype.kind == "b" or str(raw.dtype) in ("boolean", "bool"):
            return "bool"
        if raw.dtype == object:
            for v in raw:
                defect = _panel_type_defect(v)
                if defect is not None:
                    return defect
                if not isinstance(v, (list, tuple, np.ndarray)) and \
                        not isinstance(v, (int, float, complex, np.number)):
                    return "object"
        return None
    return None


def _reject_typed_panels(y, p, scorer: str, tag: str = "SF-BL-11433") -> None:
    """SF-BL-11433: raise a named ValueError if the RAW outcomes `y`
    or forecasts `p` contain any string, object or bool element.

    This screen MUST run on the raw inputs, before any
    ``np.asarray(..., dtype=float)`` conversion: float conversion
    silently maps True -> 1.0 and False -> 0.0 (so
    ``brier([1, 0], [True, False])`` returned 0.0, a perfect score
    for boolean flags) and silently parses numeric strings (so a
    string panel was graded bit-identically to its numeric control,
    or — for non-numeric strings on panels that bypass conversion —
    died with a bare UFuncTypeError naming neither panel). A string,
    bool or object element is not a number — it is the signature of
    an unparsed text / flag column reaching the scorers. Pure
    validation — it never transforms `y`/`p` — so lawful float (and
    int) panels reach the unchanged scoring path bit-identically.
    """
    for raw, name, role in ((y, "y", "outcomes"), (p, "p", "forecasts")):
        defect = _panel_type_defect(raw)
        if defect == "string":
            raise ValueError(
                f"{scorer}: {role} {name} must not contain string "
                f"values; got string value(s) in {name} (a string is "
                f"not a valid {name} value) ({tag})")
        if defect == "bool":
            raise ValueError(
                f"{scorer}: {role} {name} must not contain boolean "
                f"values; got bool value(s) in {name} (a bool is not "
                f"a valid {name} value) ({tag})")
        if defect == "object":
            raise ValueError(
                f"{scorer}: {role} {name} must not contain object "
                f"values; got object-dtype / object value(s) in "
                f"{name} (an object is not a valid {name} value) "
                f"({tag})")


def brier(y, p):
    """Mean Brier score: ``mean((p - y) ** 2)`` over finite pairs.

    Raises
    ------
    ValueError
        If ``y`` and ``p`` have different shapes, or if the panel is
        empty — including empty after masking (SF-BL-9886). Previously
        the bare subtraction broadcast a mismatched panel —
        ``brier([1, 0], [0.5])`` returned 0.25, grading two outcomes
        against one broadcast forecast — and ``brier([], [])``
        returned NaN (mean of an empty slice).

    Non-finite pairs (NaN / +/-inf in ``y`` or ``p``) are MASKED, not
    scored and not rejected: the fleet convention of
    ``backtest._check``, which drops every pair that is not finite in
    both arrays before scoring. Only if masking leaves no pair at all
    does this raise (the empty-panel rule above). Validation and
    masking never transform a surviving value, so a clean matched
    panel is bit-identical to the pre-fix computation.

    SF-BL-11433: the RAW panels are type-screened BEFORE the float
    conversion below (see :func:`_reject_typed_panels`): any string,
    object or bool element in ``y`` or ``p`` raises a ValueError
    naming that panel. Previously ``brier([1, 0], [True, False])``
    returned 0.0 — bool flags coerced to perfect endpoint forecasts
    — and string panels were either coerced into lawful scores or
    died with bare UFuncTypeErrors naming neither panel.
    """
    _reject_typed_panels(y, p, "brier")
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    if y.shape != p.shape:
        raise ValueError(
            f"brier: y and p shapes differ: {y.shape} != {p.shape}; "
            f"outcomes and forecasts must be aligned one-to-one "
            f"(SF-BL-9886)")
    if y.size == 0:
        raise ValueError(
            "brier: empty panel — y and p contain no pairs; a mean "
            "Brier score over zero forecasts is undefined (SF-BL-9886)")
    m = np.isfinite(y) & np.isfinite(p)
    if not m.all():
        y, p = y[m], p[m]
        if y.size == 0:
            raise ValueError(
                "brier: no finite pairs — every (y, p) pair is "
                "non-finite in y or p, so masking leaves an empty "
                "panel (SF-BL-9886)")
    return float(np.mean((p - y) ** 2))
def log_score(y, p):
    """Mean log-likelihood of a forecast panel (higher is better, 0 is
    perfect; the negative of the usual log loss — the convention
    ``summarize`` quotes and stored results carry).

    Raises
    ------
    ValueError
        On any panel that cannot be scored lawfully (SF-BL-9887),
        screened BEFORE the clip/mean below: (1) ``y`` and ``p`` with
        different shapes — previously the subtraction-free formula
        broadcast a 1-element ``p`` across every outcome, so
        ``log_score([1, 0], [0.5])`` returned -0.6931, the score of
        the single broadcast forecast, grading a different sample
        than the caller supplied; (2) an empty panel — previously
        NaN, the mean of an empty slice; (3) any non-finite value in
        ``y`` or ``p`` — previously a NaN forecast returned NaN and
        +/-inf was clipped to a lawful endpoint (+inf scored
        -1.0e-06, a near-perfect score for an impossible forecast).
        Rejection, not masking, is the fleet ``log_score``
        convention (calibration SF-BL-9484, agents SF-BL-10031,
        backtest SF-BL-10071): a bare-float result has no channel to
        surface a dropped-row count, and masking would still grade
        a different sample than the caller supplied; (4) any finite
        forecast outside [0, 1] — previously clipped into a lawful
        score (p=1.5 scored -1.0e-06); the clip below now applies
        only to lawful endpoint forecasts 0/1; (5) any finite
        outcome not exactly 0 or 1 — previously ``y=2`` returned
        exactly -ln(2), an impossible outcome scored as a coin
        flip. Validation is pure — it never transforms a value —
        so every lawful panel reaches the unchanged clip + mean
        arithmetic and is bit-identical to the pre-fix computation.

    SF-BL-11433: the RAW panels are type-screened BEFORE the float
    conversion below (see :func:`_reject_typed_panels`): any string,
    object or bool element in ``y`` or ``p`` raises a ValueError
    naming that panel. Previously a bool forecast panel coerced to
    lawful endpoints (``[True, False]`` scored -1.0e-06, near
    perfect) and string panels were either coerced into lawful
    scores or died with bare UFuncTypeErrors naming neither panel.
    """
    _reject_typed_panels(y, p, "log_score")
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    if y.shape != p.shape:
        raise ValueError(
            f"log_score: y and p shapes differ: {y.shape} != {p.shape}; "
            f"outcomes and forecasts must be aligned one-to-one "
            f"(SF-BL-9887)")
    if y.size == 0:
        raise ValueError(
            "log_score: empty panel — y and p contain no pairs; a mean "
            "log score over zero forecasts is undefined (SF-BL-9887)")
    n_nonfinite = int(np.sum(~(np.isfinite(y) & np.isfinite(p))))
    if n_nonfinite:
        raise ValueError(
            f"log_score: {n_nonfinite} of {y.size} pair(s) are "
            f"non-finite in y or p (NaN/inf); refusing to score a "
            f"corrupt panel (SF-BL-9887)")
    lo, hi = float(p.min()), float(p.max())
    if lo < 0.0 or hi > 1.0:
        raise ValueError(
            f"log_score: forecasts p must lie in [0, 1]; got "
            f"min={lo!r}, max={hi!r} (SF-BL-9887)")
    bad_y = y[(y != 0.0) & (y != 1.0)]
    if bad_y.size:
        raise ValueError(
            f"log_score: outcomes y must be binary (0/1); got "
            f"{bad_y.size} non-binary value(s) "
            f"(min={float(bad_y.min())!r}, max={float(bad_y.max())!r}) "
            f"(SF-BL-9887)")
    p = np.clip(p, 1e-6, 1 - 1e-6); return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
def auc(y, p):
    """Rank-based (Mann-Whitney) area under the ROC curve.

    SF-BL-11432: the RAW panels are type-screened BEFORE the ranking
    below (see :func:`_reject_typed_panels`): any string, object or
    bool element in ``y`` or ``p`` raises a ValueError naming that
    panel. Previously the ranking path coerced such panels silently:
    ``auc([1, 0], ['0.9', '0.1'])`` and ``auc([1, 0], [True,
    False])`` both returned 1.0, bit-identical to the numeric
    control — an unparsed text / flag column read as perfect
    discrimination — and a string-outcome panel returned NaN (no
    outcome ever equalled 1). The screen is pure validation — it
    never transforms a value — so lawful float (and int) panels
    reach the unchanged ranking arithmetic bit-identically.
    """
    _reject_typed_panels(y, p, "auc", tag="SF-BL-11432")
    # rank-based Mann-Whitney
    order = np.argsort(p); ranks = np.empty(len(p), float); ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    pos = y == 1; npos, nneg = pos.sum(), (~pos).sum()
    if npos == 0 or nneg == 0: return float("nan")
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def _validate_bins(bins) -> int:
    """Validate a bin count and return it as a plain int (SF-BL-11466).

    ``bins`` shapes the reliability table, but it was never
    type-validated: it flowed straight into
    ``np.linspace(0, 1, bins + 1)``, so ``bins=True`` silently
    collapsed the table to a single bin (one row, n=2, mean_pred 0.5
    on the task repro panel, where ``bins=10`` returned per-bin rows)
    and ``bins='10'`` died with a bare ``TypeError: can only
    concatenate str (not "int") to str`` that never named ``bins``;
    ``bins=2.5`` and ``bins=None`` likewise died with bare TypeErrors
    naming neither the argument nor the expected type. Mirrors
    calibration.py's ``_validate_n_bins`` (SF-BL-9722), adapted to
    this function's ``bins`` parameter name: only a positive integer
    type is lawful.

    Raises
    ------
    ValueError
        If ``bins`` is not a positive integer (>= 1). Booleans
        (including ``np.bool_`` — bool is an int subclass, so
        ``True`` must be screened explicitly, not read as 1), floats
        (including integral floats such as 10.0 and non-finite
        NaN/inf), strings, None and all other non-integer types are
        rejected, as are integers < 1 (the SF-BL-9888 value domain:
        ``bins=0`` / ``bins=-1`` previously returned ``[]``).
        Numpy integers are accepted. Validation is pure — it never
        coerces, rounds or transforms a lawful value (it returns
        ``int(bins)``, identical in value for every lawful count) —
        so integer-bin tables are bit-identical to the pre-fix
        computation.
    """
    if isinstance(bins, (bool, np.bool_)) or not isinstance(
            bins, (int, np.integer)):
        raise ValueError(
            f"bins must be a positive integer (>= 1); "
            f"got bins={bins!r} ({type(bins).__name__}) (SF-BL-11466)")
    bins = int(bins)
    if bins < 1:
        raise ValueError(
            f"bins must be a positive integer (>= 1); got bins={bins} "
            f"(SF-BL-11466)")
    return bins


def calibration_table(y, p, bins=10):
    """Per-bin reliability table (rows for non-empty bins only).

    Raises ValueError naming ``bins`` if it is not a positive
    integer (SF-BL-11466; see :func:`_validate_bins`). Previously
    ``bins=True`` silently returned a single-bin table and
    ``bins='10'`` died with a bare TypeError naming neither
    ``bins`` nor the expected type. Lawful integer-bin tables are
    bit-identical to the pre-fix computation.

    SF-BL-11465: the RAW ``y``/``p`` panels are type-screened BEFORE
    the comparisons / averages in the body (see
    :func:`_reject_typed_panels`): any bool or string-dtype element
    in ``y`` or ``p`` raises a ValueError naming that panel.
    Previously the body compared and averaged the raw arrays with
    no type screen, so a bool forecast panel ``[True, False]`` was
    tabled as endpoint rows (``mean_pred`` 1.0 and 0.0 with matching
    empirical rates), bit-identical in structure to lawful endpoint
    forecasts — a flag-column mixup read as calibration evidence —
    and string panels died with bare UFuncTypeErrors naming neither
    panel. The screen is pure validation — it never transforms a
    value — so lawful numeric panels produce bit-identical tables.
    """
    _reject_typed_panels(y, p, "calibration_table", tag="SF-BL-11465")
    bins = _validate_bins(bins)
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
