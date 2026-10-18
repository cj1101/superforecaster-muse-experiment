# Integrated copy — adopted 2026-10-17 (Wave 3, meta-cdf-integrate) as the
# Metaculus scaffold's SINGLE numeric CDF implementation. Source of truth
# for the original: wave3/meta-forecast-types/cdf_module.py (identical
# logic; metaculus_client.py delegates to this file, do not re-implement
# CDF construction elsewhere in the scaffold).
"""
cdf_module.py — Percentile elicitation + CDF construction for Metaculus
numeric / discrete / date questions, plus CDF scoring utilities.

Wave 3 / M6-M7 worker. numpy + scipy only.

Design contract (must stay consistent with the fleet scaffold at
metaculus_client.py (this directory)):

  * A numeric forecast is a 201-point CDF (discrete: inbound_outcome_count+1
    points) evaluated at evenly spaced *scaled locations* l in [0, 1] across
    [range_min, range_max].
  * When scaling.zero_point is not None the axis is log-scaled:
        deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
        nominal(l)  = range_min + (range_max - range_min)
                      * (deriv_ratio**l - 1) / (deriv_ratio - 1)
    Otherwise nominal(l) = range_min + (range_max - range_min) * l.
  * Forecasters declare percentiles {prob: nominal value}. The CDF at each
    grid nominal is the linear interpolation (constant extrapolation outside
    the declared range) of prob as a function of value, then clipped to
    [0, 1] and made non-decreasing. `percentiles_to_cdf(..., interp="linear")`
    reproduces the scaffold construction exactly (verified in
    test_module.py against the scaffold's own function).
  * interp="pchip" is an optional smoother variant (monotone cubic through
    the declared points, scipy PchipInterpolator). It is NOT the scaffold
    default and is never used for submitted payloads unless a caller opts
    in explicitly; it exists to measure whether interpolation choice matters.

Elicitation helpers (all causal — they consume only a trailing window of
log-returns supplied by the caller and the spot level at the origin):

  * climatology_percentiles  — empirical quantiles of trailing h-day returns.
  * naive_normal_percentiles — Gaussian fit to trailing *daily* returns,
    scaled by sqrt(h) under an iid assumption (the "naive fit" baseline).
  * disciplined_percentiles  — climatology-anchored elicitation:
      1. start from climatology empirical percentiles (robust base rates);
      2. shrink the median toward the no-change (random-walk) median by
         50% — trailing drift is mostly noise at short horizons;
      3. rescale deviations from the median by a volatility-regime factor
         r' = 1 + 0.5 * (sigma_recent / sigma_long - 1), clamped to
         [0.75, 1.5] (recent 30d vol vs the full trailing window);
      4. fatten the outer (5/95) percentiles by an excess-kurtosis factor
         derived from the trailing window, capped at +25% extra spread —
         Gaussian tails are the naive fit's known failure mode;
      5. enforce strict monotonicity in value.

Scoring utilities operate on the 201-point grid itself, so every method is
scored on exactly the object Metaculus would receive:

  * crps_from_cdf      — CRPS via the standard integral of (F - 1{y<=x})^2,
    trapezoid-integrated over the grid nominals (in question units).
  * log_score_from_cdf — log of the implied density at the realised value
    (finite difference of the CDF over the grid cell containing y, with a
    small uniform mixture floor so no method scores -inf; the floor is
    identical for all methods). Higher is better.
  * interval_log_score — log of the CDF probability mass of the grid cell
    containing y (Metaculus-style binned log score up to a constant).
  * cdf_to_bin_probs   — turn a CDF into multiple-choice probabilities over
    caller-supplied value bins (for the MC transfer check).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

CDF_SIZE = 201
DEFAULT_PERCENTILES = (0.05, 0.25, 0.50, 0.75, 0.95)

# ---------------------------------------------------------------------------
# Grid geometry (scaffold-identical)
# ---------------------------------------------------------------------------

def nominal_at_location(location, range_min, range_max, zero_point=None):
    """Scaled location(s) in [0,1] -> nominal value(s). Scaffold formula."""
    loc = np.asarray(location, dtype=float)
    if zero_point is None:
        return range_min + (range_max - range_min) * loc
    deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
    return range_min + (range_max - range_min) * (deriv_ratio ** loc - 1.0) / (deriv_ratio - 1.0)


def grid_nominals(range_min, range_max, zero_point=None, size=CDF_SIZE):
    locs = np.linspace(0.0, 1.0, size)
    return nominal_at_location(locs, float(range_min), float(range_max),
                               None if zero_point is None else float(zero_point))


# ---------------------------------------------------------------------------
# Percentiles -> CDF
# ---------------------------------------------------------------------------

def _validate_percentiles(percentiles):
    items = sorted((float(p), float(v)) for p, v in dict(percentiles).items())
    if len(items) < 2:
        raise ValueError("need at least 2 declared percentiles")
    probs = np.array([p for p, _ in items])
    vals = np.array([v for _, v in items])
    if np.any(np.diff(probs) <= 0):
        raise ValueError("percentile probabilities must be strictly increasing")
    if np.any((probs <= 0) | (probs >= 1)):
        raise ValueError("percentile probabilities must lie in (0, 1)")
    if np.any(np.diff(vals) < 0):
        raise ValueError("percentile values must be non-decreasing")
    return probs, vals


def percentiles_to_cdf(percentiles, range_min, range_max, zero_point=None,
                       size=CDF_SIZE, interp="linear"):
    """Declared percentiles -> monotone `size`-point CDF on the question grid.

    interp="linear" reproduces the scaffold construction exactly.
    interp="pchip" uses a monotone cubic through (value, prob).
    """
    probs, vals = _validate_percentiles(percentiles)
    nominals = grid_nominals(range_min, range_max, zero_point, size)
    if interp == "linear":
        cdf = np.interp(nominals, vals, probs)  # constant extrapolation, as scaffold
    elif interp == "pchip":
        from scipy.interpolate import PchipInterpolator
        # Pchip needs strictly increasing x; collapse duplicate values.
        keep = np.concatenate([[True], np.diff(vals) > 0])
        if keep.sum() < 2:
            cdf = np.interp(nominals, vals, probs)
        else:
            # IMPORTANT: never let the cubic extrapolate. Pchip extrapolation
            # outside the declared range explodes (measured: F=7.3 / -10.3 at
            # the grid edges on a tame test case), and after clip+monotone
            # repair it pins the entire CDF to 1. Evaluate on nominals
            # clipped to the declared range instead, which reproduces the
            # scaffold's constant-extrapolation tails.
            clipped = np.clip(nominals, vals[keep][0], vals[keep][-1])
            cdf = PchipInterpolator(vals[keep], probs[keep],
                                    extrapolate=False)(clipped)
    else:
        raise ValueError(f"unknown interp={interp!r}")
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf = np.maximum.accumulate(cdf)  # enforce monotone CDF
    return cdf


def cdf_at(cdf, nominals, value):
    """Interpolated CDF value F(value) from a grid CDF."""
    return float(np.interp(value, nominals, cdf))


def validate_cdf(cdf, max_step_warn=0.19):
    """Return list of problems (empty = valid), mirroring scaffold checks."""
    cdf = np.asarray(cdf, dtype=float)
    problems = []
    if np.any((cdf < 0) | (cdf > 1)):
        problems.append("CDF values outside [0, 1]")
    if np.any(np.diff(cdf) < -1e-12):
        problems.append("CDF is not non-decreasing")
    steps = np.diff(cdf)
    if len(steps) and steps.max() > max_step_warn:
        problems.append(f"largest CDF step {steps.max():.3f} > {max_step_warn}")
    return problems


# ---------------------------------------------------------------------------
# Elicitation (inputs: trailing log-return arrays only — caller enforces
# causality by slicing data through the forecast origin)
# ---------------------------------------------------------------------------

def _pct_dict(probs, values):
    return {float(p): float(v) for p, v in zip(probs, values)}


def climatology_percentiles(spot, trailing_h_returns, probs=DEFAULT_PERCENTILES):
    """Empirical quantiles of trailing h-day log-returns applied to spot."""
    r = np.asarray(trailing_h_returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 30:
        raise ValueError("climatology needs >= 30 trailing h-day returns")
    qs = np.quantile(r, np.asarray(probs))
    return _pct_dict(probs, spot * np.exp(qs))


def naive_normal_percentiles(spot, trailing_daily_returns, horizon_days,
                             probs=DEFAULT_PERCENTILES):
    """Gaussian fit to trailing daily log-returns, iid-scaled to horizon."""
    d = np.asarray(trailing_daily_returns, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 30:
        raise ValueError("naive fit needs >= 30 trailing daily returns")
    mu_h = d.mean() * horizon_days
    sd_h = d.std(ddof=1) * np.sqrt(horizon_days)
    z = stats.norm.ppf(np.asarray(probs))
    return _pct_dict(probs, spot * np.exp(mu_h + sd_h * z))


def disciplined_percentiles(spot, trailing_h_returns, trailing_daily_returns,
                            recent_daily_returns, probs=DEFAULT_PERCENTILES):
    """Climatology-anchored, regime-scaled, tail-fattened elicitation.

    See module docstring for the four adjustments. All statistics come from
    the trailing windows passed in (causal by construction at the caller).
    """
    probs = np.asarray(probs, dtype=float)
    base = climatology_percentiles(spot, trailing_h_returns, probs)
    logv = np.log(np.array([base[float(p)] for p in probs]) / spot)

    # (2) shrink median toward no-change (log-return 0) by 50%.
    med_idx = int(np.argmin(np.abs(probs - 0.5)))
    median_shrunk = 0.5 * logv[med_idx]
    dev = logv - logv[med_idx]
    logv = median_shrunk + dev + 0.0  # re-centre deviations on shrunk median
    # NOTE: re-centring shifts all quantiles by the median shrinkage, which
    # is intended: drift shrinkage is a location adjustment, not just median.

    # (3) volatility-regime scaling of deviations from the (shrunk) median.
    d_long = np.asarray(trailing_daily_returns, dtype=float)
    d_recent = np.asarray(recent_daily_returns, dtype=float)
    sd_long = d_long[np.isfinite(d_long)].std(ddof=1)
    sd_recent = d_recent[np.isfinite(d_recent)].std(ddof=1)
    if sd_long > 0 and np.isfinite(sd_recent):
        ratio = sd_recent / sd_long
        r_adj = float(np.clip(1.0 + 0.5 * (ratio - 1.0), 0.75, 1.5))
    else:
        r_adj = 1.0
    centre = logv[med_idx]
    logv = centre + (logv - centre) * r_adj

    # (4) fatten outer tails by trailing excess kurtosis (capped).
    r = np.asarray(trailing_h_returns, dtype=float)
    r = r[np.isfinite(r)]
    exk = float(stats.kurtosis(r)) if len(r) >= 30 else 0.0  # Fisher excess
    fat = float(np.clip(1.0 + 0.05 * max(exk, 0.0), 1.0, 1.25))
    outer = (probs <= 0.10) | (probs >= 0.90)
    logv[outer] = centre + (logv[outer] - centre) * fat

    values = spot * np.exp(logv)
    values = np.maximum.accumulate(values)  # (5) monotonicity
    # Guarantee strict increase for interpolation stability.
    for i in range(1, len(values)):
        if values[i] <= values[i - 1]:
            values[i] = values[i - 1] * (1.0 + 1e-9)
    return _pct_dict(probs, values)


# ---------------------------------------------------------------------------
# Scoring (all computed on the submitted grid object itself)
# ---------------------------------------------------------------------------

def crps_from_cdf(cdf, nominals, y):
    """CRPS = int (F(x) - 1{y <= x})^2 dx, trapezoid over the grid."""
    cdf = np.asarray(cdf, dtype=float)
    nominals = np.asarray(nominals, dtype=float)
    integrand = (cdf - (nominals >= y).astype(float)) ** 2
    return float(np.trapz(integrand, nominals))  # np.trapezoid on numpy>=2


def _cell_density(cdf, nominals, y, floor_mass=1e-4):
    """Density at y from CDF finite differences + uniform-mixture floor.

    The floor mixes in floor_mass of a uniform distribution over the grid
    range, identically for every method, so log scores stay finite and the
    comparison is not decided by a single -inf.
    """
    cdf = np.asarray(cdf, dtype=float)
    nominals = np.asarray(nominals, dtype=float)
    span = nominals[-1] - nominals[0]
    cdf_mix = (1.0 - floor_mass) * cdf + floor_mass * (nominals - nominals[0]) / span
    if y <= nominals[0]:
        dens = (cdf_mix[1] - cdf_mix[0]) / (nominals[1] - nominals[0])
        mass = cdf_mix[0] + 0.5 * (cdf_mix[1] - cdf_mix[0])
        return max(dens, 1e-12), mass
    if y >= nominals[-1]:
        dens = (cdf_mix[-1] - cdf_mix[-2]) / (nominals[-1] - nominals[-2])
        mass = (1.0 - cdf_mix[-2]) + cdf_mix[-1] - cdf_mix[-2]
        return max(dens, 1e-12), 1.0 - cdf_mix[-2]
    i = int(np.searchsorted(nominals, y))  # cell [i-1, i]
    width = nominals[i] - nominals[i - 1]
    mass = cdf_mix[i] - cdf_mix[i - 1]
    dens = mass / width if width > 0 else 0.0
    return max(dens, 1e-12), max(mass, 1e-12)


def log_score_from_cdf(cdf, nominals, y):
    """Log density at the realised value (higher is better)."""
    dens, _ = _cell_density(cdf, nominals, y)
    return float(np.log(dens))


def interval_log_score_from_cdf(cdf, nominals, y):
    """Log probability mass of the grid cell containing y (higher better)."""
    _, mass = _cell_density(cdf, nominals, y)
    return float(np.log(mass))


def cdf_to_bin_probs(cdf, nominals, bin_edges):
    """Multiple-choice probabilities for bins (-inf, e1], (e1, e2], ...

    bin_edges: interior edges in nominal units. Returns len(edges)+1 probs
    summing to 1 (renormalised after a small epsilon floor per bin).
    """
    edges = np.asarray(bin_edges, dtype=float)
    bounds = np.concatenate([[nominals[0]], edges, [nominals[-1]]])
    cdf_vals = np.array([cdf_at(cdf, nominals, b) for b in bounds])
    cdf_vals[0], cdf_vals[-1] = 0.0, 1.0
    probs = np.diff(cdf_vals)
    probs = np.clip(probs, 1e-6, None)
    return probs / probs.sum()


def mc_scores(probs, true_bin):
    """(log score, Brier) for a multiple-choice forecast. Higher log better,
    lower Brier better."""
    probs = np.asarray(probs, dtype=float)
    onehot = np.zeros_like(probs)
    onehot[true_bin] = 1.0
    return float(np.log(max(probs[true_bin], 1e-12))), float(np.sum((probs - onehot) ** 2))
