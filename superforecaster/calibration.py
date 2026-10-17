"""
calibration.py — Calibration & aggregation for probabilistic forecasters.

Pure numpy / scipy / stdlib. NO sklearn.

Contents
--------
Scoring:      brier_score, log_score, murphy_decomposition
Calibration diagnostics: ece, mce, calibration_bins / calibration_table
Calibrators:  PlattScaler (Newton's method), IsotonicCalibrator (PAVA by hand),
              TemperatureScaler, BetaCalibrator
Pooling:      linear_pool, geometric_pool, log_odds_pool (extremized),
              optimize_pool_weights (minimises log score on validation data)

Conventions
-----------
y : (N,) array of 0/1 outcomes.
p : (N,) array of probabilities in [0,1], or (N, M) matrix for M forecasters.
All public functions clip probabilities to [eps, 1-eps] internally where a
log / logit is taken, so 0/1 forecasts do not produce inf/nan scores.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

EPS = 1e-15
_CLIP = 1e-7  # clipping for logits: keeps logit finite (~ +/-16.1)


def _as_y(y):
    y = np.asarray(y, dtype=float).ravel()
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must be binary 0/1")
    return y


def _as_p(p, eps=_CLIP):
    p = np.asarray(p, dtype=float)
    return np.clip(p, eps, 1 - eps)


# --------------------------------------------------------------------------
# Scoring rules
# --------------------------------------------------------------------------

def brier_score(y, p) -> float:
    """Mean squared error between forecast probability and outcome."""
    y = _as_y(y); p = np.asarray(p, dtype=float).ravel()
    return float(np.mean((p - y) ** 2))


def log_score(y, p, eps=EPS) -> float:
    """Mean negative log-likelihood (lower is better). Aka log loss."""
    y = _as_y(y); p = np.clip(np.asarray(p, dtype=float).ravel(), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# Alias familiar to sklearn users
log_loss = log_score


def calibration_bins(y, p, n_bins: int = 10, strategy: str = "uniform"):
    """
    Bin forecasts and summarise each bin.

    Returns dict of arrays: bin_edges, count, mean_pred, frac_pos, bin_id.
    strategy: 'uniform' (equal-width) or 'quantile' (equal-count).
    Empty bins have count 0 and nan mean_pred / frac_pos.
    """
    y = _as_y(y); p = np.asarray(p, dtype=float).ravel()
    n = len(y)
    if strategy == "quantile":
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
        # if duplicates collapsed the bins, fall back to uniform
        if len(edges) < 3:
            edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    nb = len(edges) - 1
    bin_id = np.clip(np.digitize(p, edges[1:-1], right=False), 0, nb - 1)
    count = np.zeros(nb, dtype=int)
    mean_pred = np.full(nb, np.nan)
    frac_pos = np.full(nb, np.nan)
    for b in range(nb):
        m = bin_id == b
        count[b] = int(m.sum())
        if count[b]:
            mean_pred[b] = float(p[m].mean())
            frac_pos[b] = float(y[m].mean())
    return {"bin_edges": edges, "count": count, "mean_pred": mean_pred,
            "frac_pos": frac_pos, "bin_id": bin_id, "n": n}


# Backwards-friendly name
calibration_table = calibration_bins


def expected_calibration_error(y, p, n_bins: int = 10, strategy: str = "uniform") -> float:
    """ECE: count-weighted mean |frac_pos - mean_pred| over bins."""
    t = calibration_bins(y, p, n_bins=n_bins, strategy=strategy)
    filled = t["count"] > 0
    if not filled.any():
        return 0.0
    w = t["count"][filled] / t["n"]
    return float(np.sum(w * np.abs(t["frac_pos"][filled] - t["mean_pred"][filled])))


def maximum_calibration_error(y, p, n_bins: int = 10, strategy: str = "uniform") -> float:
    """MCE: max |frac_pos - mean_pred| over non-empty bins."""
    t = calibration_bins(y, p, n_bins=n_bins, strategy=strategy)
    filled = t["count"] > 0
    if not filled.any():
        return 0.0
    return float(np.max(np.abs(t["frac_pos"][filled] - t["mean_pred"][filled])))


ece = expected_calibration_error
mce = maximum_calibration_error


def murphy_decomposition(y, p, n_bins: int = 10, strategy: str = "uniform"):
    """
    Murphy (1973) decomposition of the Brier score, using binned forecasts:

        Brier = Reliability - Resolution + Uncertainty

    Reliability: weighted squared gap between bin mean forecast and bin
                 empirical frequency (lower = better calibrated).
    Resolution:  weighted variance of bin frequencies around the base rate
                 (higher = forecast discriminates more; subtracted).
    Uncertainty: base_rate * (1 - base_rate); irreducible given the marginal.

    Note: the decomposition is exact for the *binned* forecast (each case
    replaced by its bin mean). The returned 'brier_binned' is that quantity;
    'brier' is the raw Brier score, and 'residual' = brier - brier_binned
    captures within-bin forecast variance.
    """
    y = _as_y(y); p = np.asarray(p, dtype=float).ravel()
    t = calibration_bins(y, p, n_bins=n_bins, strategy=strategy)
    base = float(y.mean())
    uncertainty = base * (1 - base)
    filled = t["count"] > 0
    w = t["count"][filled] / t["n"]
    reliability = float(np.sum(w * (t["mean_pred"][filled] - t["frac_pos"][filled]) ** 2))
    resolution = float(np.sum(w * (t["frac_pos"][filled] - base) ** 2))
    brier_binned = reliability - resolution + uncertainty
    raw = brier_score(y, p)
    return {"reliability": reliability, "resolution": resolution,
            "uncertainty": uncertainty, "brier_binned": float(brier_binned),
            "brier": raw, "residual": float(raw - brier_binned),
            "base_rate": base}


# --------------------------------------------------------------------------
# Calibrators — common interface: fit(y, p) -> self ; predict(p) -> calibrated p
# Also callable via transform(). Fit on a *validation/calibration* set that
# is disjoint from the set used to produce the raw forecasts, where possible.
# --------------------------------------------------------------------------

class PlattScaler:
    """
    Platt scaling: p_cal = sigmoid(A * s + B), where s = logit(p_raw).

    Fit by Newton's method (IRLS-equivalent for 1-feature logistic
    regression) with a tiny L2 penalty for numerical stability. No scipy
    optimiser is used for the fit itself, per spec.
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-10, l2: float = 1e-6):
        self.max_iter, self.tol, self.l2 = max_iter, tol, l2
        self.A_, self.B_ = 1.0, 0.0
        self.n_iter_ = 0

    def fit(self, y, p):
        y = _as_y(y); s = logit(_as_p(p)).ravel()
        X = np.column_stack([s, np.ones_like(s)])  # params [A, B]
        theta = np.array([1.0, 0.0])
        for it in range(self.max_iter):
            z = X @ theta
            mu = expit(z)
            W = np.clip(mu * (1 - mu), 1e-12, None)
            grad = X.T @ (mu - y) + self.l2 * theta
            H = (X * W[:, None]).T @ X + self.l2 * np.eye(2)
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, grad, rcond=None)[0]
            theta_new = theta - step
            # damped update if the NLL worsens (guards rare overshoot)
            if log_score(y, expit(X @ theta_new)) > log_score(y, expit(z)) + 1e-12:
                theta_new = theta - 0.5 * step
            delta = np.max(np.abs(theta_new - theta))
            theta = theta_new
            self.n_iter_ = it + 1
            if delta < self.tol:
                break
        self.A_, self.B_ = float(theta[0]), float(theta[1])
        return self

    def predict(self, p):
        return expit(self.A_ * logit(_as_p(p)) + self.B_)

    transform = predict


class IsotonicCalibrator:
    """
    Isotonic regression via the Pool Adjacent Violators Algorithm (PAVA),
    implemented by hand. Fits a non-decreasing step function p_raw -> freq.
    Prediction linearly interpolates between fitted knots (standard practice;
    a pure step function is available with interpolation='step').
    """

    def __init__(self, interpolation: str = "linear", out_of_bounds: str = "clip"):
        if interpolation not in ("linear", "step"):
            raise ValueError("interpolation must be 'linear' or 'step'")
        self.interpolation = interpolation
        self.out_of_bounds = out_of_bounds
        self.x_, self.y_ = None, None

    def fit(self, y, p):
        y = _as_y(y); p = np.asarray(p, dtype=float).ravel()
        order = np.argsort(p, kind="mergesort")
        xs, ys = p[order], y[order]
        # Collapse exact ties: PAVA on unique x with counts as weights.
        ux, inv, counts = np.unique(xs, return_inverse=True, return_counts=True)
        sums = np.zeros(len(ux)); np.add.at(sums, inv, ys)
        vals = sums / counts
        wts = counts.astype(float).copy()
        # --- PAVA ---
        # Blocks are maintained as stacks; merge while monotonicity violated.
        block_val = list(vals); block_wt = list(wts); block_end = list(range(len(ux)))
        i = 0
        # Iterative stack-based PAVA
        vals_s, wts_s, ends_s = [], [], []
        for idx in range(len(ux)):
            v, w, e = float(vals[idx]), float(wts[idx]), idx
            while vals_s and vals_s[-1] > v:
                pv, pw = vals_s.pop(), wts_s.pop(); ends_s.pop()
                v = (pw * pv + w * v) / (pw + w)
                w = pw + w
            vals_s.append(v); wts_s.append(w); ends_s.append(e)
        fitted_unique = np.empty(len(ux))
        start = 0
        for v, e in zip(vals_s, ends_s):
            fitted_unique[start:e + 1] = v
            start = e + 1
        self.x_, self.y_ = ux, np.clip(fitted_unique, 0.0, 1.0)
        return self

    def predict(self, p):
        if self.x_ is None:
            raise RuntimeError("IsotonicCalibrator is not fitted")
        p = np.asarray(p, dtype=float)
        if self.interpolation == "linear":
            out = np.interp(p, self.x_, self.y_, left=self.y_[0], right=self.y_[-1])
        else:
            idx = np.clip(np.searchsorted(self.x_, p, side="right") - 1, 0, len(self.x_) - 1)
            out = self.y_[idx]
        return np.clip(out, 0.0, 1.0)

    transform = predict


def _logistic_fit_newton(X, y, l2=1e-6, max_iter=100, tol=1e-10):
    """Shared Newton solver for logistic regression (used by BetaCalibrator)."""
    n, d = X.shape
    theta = np.zeros(d)
    for _ in range(max_iter):
        mu = expit(X @ theta)
        W = np.clip(mu * (1 - mu), 1e-12, None)
        grad = X.T @ (mu - y) + l2 * theta
        H = (X * W[:, None]).T @ X + l2 * np.eye(d)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        new = theta - step
        if np.max(np.abs(new - theta)) < tol:
            theta = new; break
        theta = new
    return theta


class BetaCalibrator:
    """
    Beta calibration (Kull et al., 2017):
        logit(p_cal) = c + a * ln(p) + b * ln(1 - p)
    Fit as logistic regression on features [ln p, ln(1-p), 1] by Newton's
    method. Nests Platt-on-logit and identity as special cases, and can fix
    both over- and under-confidence asymmetrically.
    Params after fit: a_, b_, c_.
    """

    def __init__(self, l2: float = 1e-6):
        self.l2 = l2
        self.a_, self.b_, self.c_ = 1.0, 1.0, 0.0

    @staticmethod
    def _features(p):
        pc = _as_p(p).ravel()
        return np.column_stack([np.log(pc), np.log(1 - pc), np.ones_like(pc)])

    def fit(self, y, p):
        y = _as_y(y); X = self._features(p)
        theta = _logistic_fit_newton(X, y, l2=self.l2)
        self.a_, self.b_, self.c_ = float(theta[0]), float(theta[1]), float(theta[2])
        return self

    def predict(self, p):
        pc = _as_p(p)
        z = self.c_ + self.a_ * np.log(pc) + self.b_ * np.log(1 - pc)
        return expit(z)

    transform = predict


class TemperatureScaler:
    """
    Temperature scaling: p_cal = sigmoid(logit(p_raw) / T).
    T > 1 softens (fixes overconfidence), T < 1 sharpens.
    T is fit by minimising log score with scipy (bounded, T in [0.05, 20]).
    """

    def __init__(self):
        self.T_ = 1.0

    def fit(self, y, p):
        y = _as_y(y); s = logit(_as_p(p)).ravel()

        def nll(logT):
            return log_score(y, expit(s / np.exp(logT[0])))

        res = minimize(nll, x0=np.array([0.0]), method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 500})
        T = float(np.exp(res.x[0]))
        self.T_ = float(np.clip(T, 0.05, 20.0))
        return self

    def predict(self, p):
        return expit(logit(_as_p(p)) / self.T_)

    transform = predict


# Convenience registry / functional helpers
CALIBRATORS = {"platt": PlattScaler, "isotonic": IsotonicCalibrator,
               "temperature": TemperatureScaler, "beta": BetaCalibrator}


def fit_calibrator(name: str, y, p):
    """Fit a calibrator by name and return the fitted object."""
    if name not in CALIBRATORS:
        raise ValueError(f"unknown calibrator {name!r}; choose from {sorted(CALIBRATORS)}")
    return CALIBRATORS[name]().fit(y, p)


def evaluate(y, p, n_bins: int = 10) -> dict:
    """One-call scorecard: Brier, log score, ECE, MCE + Murphy parts."""
    d = murphy_decomposition(y, p, n_bins=n_bins)
    return {"brier": brier_score(y, p), "log_score": log_score(y, p),
            "ece": expected_calibration_error(y, p, n_bins=n_bins),
            "mce": maximum_calibration_error(y, p, n_bins=n_bins),
            "reliability": d["reliability"], "resolution": d["resolution"],
            "uncertainty": d["uncertainty"]}


# --------------------------------------------------------------------------
# Pooling / aggregation of multiple forecasters
# --------------------------------------------------------------------------

def _as_matrix(probs):
    P = np.asarray(probs, dtype=float)
    if P.ndim == 1:
        P = P[:, None]
    if P.ndim != 2:
        raise ValueError("probs must be (N,) or (N, M)")
    return P


def _norm_weights(w, m):
    if w is None:
        return np.full(m, 1.0 / m)
    w = np.asarray(w, dtype=float).ravel()
    if len(w) != m or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative, length M, with positive sum")
    return w / w.sum()


def linear_pool(probs, weights=None):
    """Weighted arithmetic mean of probabilities."""
    P = _as_matrix(probs); w = _norm_weights(weights, P.shape[1])
    return np.clip(P @ w, 0.0, 1.0)


def geometric_pool(probs, weights=None):
    """
    Weighted geometric mean, renormalised for the binary case:
        p ∝ prod p_i^w_i ,  (1-p) ∝ prod (1-p_i)^w_i
    Equivalent to an (un-extremized) log-odds pool.
    """
    P = _as_matrix(probs); w = _norm_weights(weights, P.shape[1])
    Pc = _as_p(P)
    log_p = (np.log(Pc) * w).sum(axis=1)
    log_q = (np.log(1 - Pc) * w).sum(axis=1)
    return expit(log_p - log_q)


def log_odds_pool(probs, weights=None, extremization: float = 1.0):
    """
    Extremized log-odds (logit) pool (Satopää et al.):

        logit(p_pool) = a * sum_i w_i * logit(p_i)

    a = 1 recovers the geometric pool. a > 1 extremizes — pushes the pooled
    forecast away from 0.5 — which corrects the under-confidence that comes
    from forecasters sharing information / each hedging toward the base rate.
    a < 1 de-extremizes (shrinks toward 0.5).
    """
    if extremization <= 0:
        raise ValueError("extremization must be > 0")
    P = _as_matrix(probs); w = _norm_weights(weights, P.shape[1])
    pooled_logit = logit(_as_p(P)) @ w
    return expit(extremization * pooled_logit)


# Alias matching the literature's name
extremized_pool = log_odds_pool


def optimize_pool_weights(y, probs, method: str = "extremized",
                          fit_extremization: bool = True,
                          a_bounds=(0.3, 4.0), seed: int = 0):
    """
    Learn pool weights (and, for method='extremized', the factor a) by
    minimising log score on validation data (y, probs).

    method: 'linear' | 'geometric' | 'extremized'
    Returns dict: weights (M,), extremization (float or None),
                  log_score, brier, pool_fn (callable probs -> pooled p).

    Optimisation: scipy SLSQP over the probability simplex for weights,
    jointly with log(a); multiple restarts (equal weights + each single
    forecaster) to avoid poor local optima in the extremized case.
    """
    y = _as_y(y); P = _as_matrix(probs); n, m = P.shape
    if method not in ("linear", "geometric", "extremized"):
        raise ValueError("method must be 'linear', 'geometric' or 'extremized'")
    use_a = (method == "extremized" and fit_extremization)
    Pc = _as_p(P); L = logit(Pc)

    def unpack(x):
        w = x[:m]
        a = float(np.exp(x[m])) if use_a else (1.0 if method != "linear" else None)
        return w, a

    def objective(x):
        w, a = unpack(x)
        if method == "linear":
            pooled = np.clip(P @ w, EPS, 1 - EPS)
        elif method == "geometric":
            pooled = expit(L @ w)
        else:
            pooled = expit(a * (L @ w))
        return log_score(y, pooled)

    rng = np.random.default_rng(seed)
    starts = [np.full(m, 1.0 / m)]
    for j in range(m):
        e = np.zeros(m); e[j] = 1.0; starts.append(e)
    if m > 1:
        starts.append(rng.dirichlet(np.ones(m)))

    cons = [{"type": "eq", "fun": lambda x: np.sum(x[:m]) - 1.0}]
    bounds = [(0.0, 1.0)] * m + ([(np.log(a_bounds[0]), np.log(a_bounds[1]))] if use_a else [])
    best = None
    for s in starts:
        x0 = np.concatenate([s, [0.0]]) if use_a else s.copy()
        res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
        if best is None or res.fun < best.fun:
            best = res
    w, a = unpack(best.x)
    w = _norm_weights(w, m)
    if method == "linear":
        pooled, a_out = linear_pool(P, w), None
        fn = lambda Q, w=w: linear_pool(Q, w)  # noqa: E731
    elif method == "geometric":
        pooled, a_out = geometric_pool(P, w), None
        fn = lambda Q, w=w: geometric_pool(Q, w)  # noqa: E731
    else:
        a_out = float(a) if use_a else 1.0
        pooled = log_odds_pool(P, w, a_out)
        fn = lambda Q, w=w, a=a_out: log_odds_pool(Q, w, a)  # noqa: E731
    return {"weights": w, "extremization": a_out, "method": method,
            "log_score": log_score(y, pooled), "brier": brier_score(y, pooled),
            "pooled": pooled, "pool_fn": fn, "success": bool(best.success)}


__all__ = [
    "brier_score", "log_score", "log_loss", "murphy_decomposition",
    "expected_calibration_error", "maximum_calibration_error", "ece", "mce",
    "calibration_bins", "calibration_table", "evaluate",
    "PlattScaler", "IsotonicCalibrator", "TemperatureScaler", "BetaCalibrator",
    "CALIBRATORS", "fit_calibrator",
    "linear_pool", "geometric_pool", "log_odds_pool", "extremized_pool",
    "optimize_pool_weights",
]
