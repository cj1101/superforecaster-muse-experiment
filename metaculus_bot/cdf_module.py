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

import numbers

import numpy as np
from scipy import stats

CDF_SIZE = 201
DEFAULT_PERCENTILES = (0.05, 0.25, 0.50, 0.75, 0.95)

# ---------------------------------------------------------------------------
# Grid geometry (scaffold-identical)
# ---------------------------------------------------------------------------

def _validate_grid_geometry(range_min, range_max, zero_point=None):
    """SF-BL-10108: validate grid geometry BEFORE any grid is built.

    Returns (range_min, range_max, zero_point) as floats. Every caller
    of the grid relies on a strictly increasing grid; pre-fix the range
    and size were forwarded unchecked into np.linspace and the scaling
    formula, so a reversed range silently returned a DESCENDING grid
    (grid_nominals(30, 0) -> [30, 29.85, ..., 0]), equal endpoints a
    constant grid, non-finite endpoints an all-NaN/inf grid, and a
    zero_point inside the range a NaN grid (zero_point == range_min
    raised a bare ZeroDivisionError instead). All of those are invalid
    geometry and now raise a named ValueError; valid geometry is
    returned untouched, so lawful grids stay bit-identical.
    """
    rmin = float(range_min)
    rmax = float(range_max)
    if not (np.isfinite(rmin) and np.isfinite(rmax)):
        raise ValueError(
            "grid range endpoints must be finite; got non-finite grid "
            f"endpoint(s) range_min={range_min!r}, range_max={range_max!r} "
            "(SF-BL-10108)"
        )
    if rmax <= rmin:
        raise ValueError(
            "grid_nominals requires range_max > range_min (a strictly "
            f"increasing grid); got range_max <= range_min with "
            f"range_min={range_min!r}, range_max={range_max!r} "
            "(SF-BL-10108)"
        )
    if zero_point is None:
        return rmin, rmax, None
    zp = float(zero_point)
    if not np.isfinite(zp):
        raise ValueError(
            f"zero_point must be finite; got zero_point={zero_point!r} "
            "(SF-BL-10108)"
        )
    if rmin <= zp <= rmax:
        # deriv_ratio = (rmax - zp) / (rmin - zp) is <= 0 here (or a
        # division by zero at zp == range_min): the log-scaled formula
        # then yields a NaN/degenerate grid, never an increasing one.
        raise ValueError(
            "zero_point must lie outside [range_min, range_max] for a "
            f"log-scaled grid; got zero_point={zero_point!r} inside "
            f"[{range_min!r}, {range_max!r}] (non-positive deriv_ratio) "
            "(SF-BL-10108)"
        )
    return rmin, rmax, zp


def _checked_deriv_ratio(range_min, range_max, zero_point):
    """SF-BL-10268: compute deriv_ratio for the log-scale branch, or
    raise a ValueError naming zero_point.

    Pre-fix nominal_at_location divided by (range_min - zero_point)
    and raised deriv_ratio to loc with no screen, so
    zero_point == range_min died with a bare ZeroDivisionError and an
    interior zero_point (50 on [0, 100], deriv_ratio -1) returned NaN
    nominals. SF-BL-10108's interval screen catches those basic cases
    in this copy, but NOT deriv_ratio itself: a zero_point so large
    that float subtraction rounds deriv_ratio to exactly 1
    (1e20 on [0, 100]) or so extreme that the ratio overflows to
    non-finite (-1e308 on [0, 1e308]) passed that screen and still
    returned NaN nominals, and deriv_ratio == 1 from equal ranges was
    reported without naming zero_point. This screen is on the ratio
    itself: it raises whenever deriv_ratio is non-finite, <= 0, or
    == 1. Lawful ratios are returned as the identical float the
    unscreened division produced, so lawful grids stay bit-identical.
    """
    try:
        zp = float(zero_point)
    except (TypeError, ValueError):
        raise ValueError(
            "nominal_at_location requires a finite zero_point for a "
            f"log-scaled grid; got zero_point={zero_point!r} "
            "(SF-BL-10268; cf. SF-BL-10108)"
        )
    if not np.isfinite(zp):
        raise ValueError(
            "nominal_at_location requires a finite zero_point for a "
            f"log-scaled grid; got non-finite zero_point={zero_point!r} "
            "making deriv_ratio non-finite (SF-BL-10268; cf. SF-BL-10108)"
        )
    denominator = float(range_min) - zp
    if denominator == 0.0:
        raise ValueError(
            "nominal_at_location requires zero_point != range_min "
            "for a log-scaled grid; got zero_point="
            f"{zero_point!r} == range_min={range_min!r}, so "
            "deriv_ratio is undefined (division by zero in "
            "(range_max - zero_point) / (range_min - zero_point)) "
            "(SF-BL-10268; cf. SF-BL-10108)"
        )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        deriv_ratio = (float(range_max) - zp) / denominator
    if (not np.isfinite(deriv_ratio)) or deriv_ratio <= 0.0 \
            or deriv_ratio == 1.0:
        raise ValueError(
            "nominal_at_location requires zero_point to yield a "
            "finite deriv_ratio > 0 and != 1 for a log-scaled grid; "
            f"got zero_point={zero_point!r} with "
            f"range_min={range_min!r}, range_max={range_max!r} "
            f"giving deriv_ratio={deriv_ratio!r} (non-finite, <= 0, "
            "or == 1) (SF-BL-10268; cf. SF-BL-10108)"
        )
    return float(deriv_ratio)


def _checked_location(location):
    """SF-BL-10312: validate the location argument of
    nominal_at_location, or raise a ValueError naming location.

    Pre-fix nominal_at_location converted location with
    np.asarray(location, dtype=float) and used it unscreened, so
    location=NaN returned a NaN nominal, location=+/-inf returned
    an infinite nominal, and an out-of-[0, 1] location such as 2.0
    silently extrapolated off the scaled axis (200.0 on a [0, 100]
    range; -0.5 returned -50.0), poisoning every grid nominal built
    from it with no flag to grid_nominals or percentiles_to_cdf
    callers. Distinct from SF-BL-10268, which screens zero_point /
    deriv_ratio, and SF-BL-10108, which covers grid range/size
    geometry only: this screen is on the location argument itself.
    Every location element must be finite and lie in the closed
    interval [0, 1]. Lawful locations are returned as the identical
    array np.asarray(location, dtype=float) produced, so in-range
    outputs stay bit-identical.
    """
    try:
        loc = np.asarray(location, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            "nominal_at_location requires location to be finite "
            "and lie in [0, 1]; got location="
            f"{location!r} (SF-BL-10312)"
        ) from None
    if not np.all(np.isfinite(loc)):
        raise ValueError(
            "nominal_at_location requires location to be finite "
            "and lie in [0, 1]; got non-finite location value(s) "
            f"{loc[~np.isfinite(loc)]!r} in location={location!r} "
            "(SF-BL-10312)"
        )
    if np.any((loc < 0.0) | (loc > 1.0)):
        raise ValueError(
            "nominal_at_location requires location to lie in "
            "[0, 1]; got out-of-range location value(s) "
            f"{loc[(loc < 0.0) | (loc > 1.0)]!r} in "
            f"location={location!r} (SF-BL-10312)"
        )
    return loc


def nominal_at_location(location, range_min, range_max, zero_point=None):
    """Scaled location(s) in [0,1] -> nominal value(s). Scaffold formula."""
    # SF-BL-10312: screen location (naming location) BEFORE any use
    # -- including the zero_point / geometry screens below -- so a
    # non-finite or out-of-[0, 1] location can never return a NaN /
    # extrapolated nominal. The validated array is used below in
    # place of the previous unscreened np.asarray conversion.
    loc = _checked_location(location)
    # SF-BL-10268: screen deriv_ratio (naming zero_point) BEFORE the
    # SF-BL-10108 geometry validation, so every bad-ratio case --
    # including deriv_ratio == 1 -- is reported as the zero_point
    # problem it is; valid ratios pass through and the 10108 checks
    # still govern range/size geometry exactly as before.
    deriv_ratio = None
    if zero_point is not None:
        deriv_ratio = _checked_deriv_ratio(range_min, range_max,
                                           zero_point)
    range_min, range_max, zero_point = _validate_grid_geometry(
        range_min, range_max, zero_point)
    if zero_point is None:
        return range_min + (range_max - range_min) * loc
    return range_min + (range_max - range_min) * (deriv_ratio ** loc - 1.0) / (deriv_ratio - 1.0)


def grid_nominals(range_min, range_max, zero_point=None, size=CDF_SIZE):
    # SF-BL-10108: reject a degenerate grid size BEFORE np.linspace --
    # pre-fix size=0 silently returned an empty grid and size=1 a
    # single point (size<0 leaked a bare numpy ValueError that never
    # named the grid). A lawful grid needs at least 2 points.
    if size < 2:
        raise ValueError(
            "grid_nominals requires size >= 2 grid points; got "
            f"size={size!r} (empty/degenerate grid) (SF-BL-10108)"
        )
    locs = np.linspace(0.0, 1.0, size)
    return nominal_at_location(locs, float(range_min), float(range_max),
                               None if zero_point is None else float(zero_point))


# ---------------------------------------------------------------------------
# Percentiles -> CDF
# ---------------------------------------------------------------------------

def _validate_percentiles(percentiles):
    # SF-BL-11307: type-validate the percentiles argument's OWN keys
    # and values BEFORE any float() coercion. Pre-fix the first line
    # coerced every key/value with float(), so string percentiles
    # parsed as lawful -- {'0.1': '10', '0.5': '50', '0.9': '90'}
    # returned a CDF bit-identical to the numeric control -- and a
    # bool value was read as 1.0/0.0, so {0.1: True, 0.5: 50,
    # 0.9: 90} silently shifted the CDF (other bool panels were
    # caught only collaterally, mislabelled, by the value-domain
    # checks below). A bool is not a number (fleet convention) and
    # a string key/value is a caller type bug, not a percentile:
    # any bool / np.bool_ / str / bytes / np.str_ key or value --
    # and any other non-numeric key/value (None, containers,
    # complex, non-scalar or bool/string-dtype arrays) -- now
    # raises a ValueError naming percentiles. Distinct from
    # SF-BL-10107 / SF-BL-10425 (non-finite VALUE domain, screened
    # below), SF-BL-11246 (this function's size type) and
    # SF-BL-11247 (its range-endpoint / zero_point types); none of
    # those fixtured the percentiles argument's own type domain.
    # Reject-only: genuinely numeric keys/values (int / float /
    # numpy numeric scalars / Decimal / Fraction / 0-d numeric
    # arrays) fall through to the identical float() conversion
    # below, so numeric-percentile CDFs stay bit-identical.
    try:
        _raw_items_11307 = list(dict(percentiles).items())
    except (TypeError, ValueError):
        raise ValueError(
            "percentiles_to_cdf requires percentiles to be a "
            "mapping of real-number probabilities to real-number "
            f"values; got percentiles={percentiles!r} (SF-BL-11307)"
        ) from None
    for _p_11307, _v_11307 in _raw_items_11307:
        for _role_11307, _el_11307 in (
                ("key", _p_11307), ("value", _v_11307)):
            _bad_11307 = False
            if isinstance(_el_11307, (bool, np.bool_)):
                _bad_11307 = True
            elif isinstance(_el_11307, (str, bytes)):
                _bad_11307 = True
            elif isinstance(_el_11307, (complex, np.complexfloating)):
                _bad_11307 = True
            elif isinstance(_el_11307, np.ndarray):
                if _el_11307.ndim != 0 \
                        or _el_11307.dtype == np.dtype(bool) \
                        or _el_11307.dtype.kind not in "iuf":
                    _bad_11307 = True
            elif not isinstance(_el_11307, numbers.Number):
                _bad_11307 = True
            if _bad_11307:
                raise ValueError(
                    "percentiles_to_cdf requires percentiles keys "
                    "and values to be real numbers (a bool or "
                    "string is not a number); got percentiles "
                    f"{_role_11307} {_el_11307!r} in "
                    f"percentiles={percentiles!r} (SF-BL-11307)"
                )
    items = sorted((float(p), float(v)) for p, v in _raw_items_11307)
    if len(items) < 2:
        raise ValueError("need at least 2 declared percentiles")
    probs = np.array([p for p, _ in items])
    vals = np.array([v for _, v in items])
    # SF-BL-10425: reject a non-finite percentile PROBABILITY (a NaN /
    # +/-inf dict key) BEFORE any comparison. Every comparison against
    # NaN is False, so pre-fix a NaN prob key passed the
    # strictly-increasing diff check and the (0, 1) range check below
    # unscreened in the live (superforecaster_out) copy, which lacked
    # any finiteness screen, and np.interp then propagated it across
    # the grid -- live repro percentiles_to_cdf({NaN: 10, 0.5: 20,
    # 0.9: 30}, 0, 100) returned a 201-point CDF whose entries were
    # all NaN (NaN-headed at the first grid points), and +/-inf prob
    # keys raised only via the (0, 1) message, which never named
    # finiteness. Distinct from SF-BL-10107, whose live repro was a
    # NaN percentile VALUE: this screen is on the probability keys.
    # It runs on the float()-converted probs array, so a NaN key is
    # caught whatever position sorted() placed it in. Reject-only:
    # finite probs fall through untouched, so lawful CDFs stay
    # bit-identical. (In this repo copy the SF-BL-10107 screen below
    # already caught non-finite probs as a side effect; this screen
    # is the dedicated SF-BL-10425 guard and fires first, and the
    # 10107 screen remains as defence in depth.)
    if not np.all(np.isfinite(probs)):
        raise ValueError(
            "percentile probabilities must be finite; got non-finite "
            f"percentile probability key(s) {probs[~np.isfinite(probs)]} "
            "(SF-BL-10425)"
        )
    # SF-BL-10107: reject non-finite percentiles BEFORE any comparison.
    # Every comparison against NaN is False, so NaN previously passed the
    # strictly-increasing, (0, 1) and non-decreasing checks below and
    # np.interp propagated it across the whole grid (all-NaN CDF returned
    # as a normal result); +/-inf values likewise slipped through the
    # value checks. isfinite screens both fields up front.
    if not np.all(np.isfinite(probs)):
        raise ValueError(
            "percentile probabilities must be finite; got non-finite "
            f"percentile probability value(s) {probs[~np.isfinite(probs)]} "
            "(SF-BL-10107)"
        )
    if not np.all(np.isfinite(vals)):
        raise ValueError(
            "percentile values must be finite; got non-finite "
            f"percentile value(s) {vals[~np.isfinite(vals)]} "
            "(SF-BL-10107)"
        )
    if np.any(np.diff(probs) <= 0):
        raise ValueError("percentile probabilities must be strictly increasing")
    if np.any((probs <= 0) | (probs >= 1)):
        raise ValueError("percentile probabilities must lie in (0, 1)")
    if np.any(np.diff(vals) < 0):
        raise ValueError("percentile values must be non-decreasing")
    return probs, vals


def _tied_percentile_groups(vals, probs):
    """SF-BL-10770: group consecutive tied declared values.

    Returns (uniq_vals, p_low, p_high): one entry per DISTINCT declared
    value, where p_low is the first (smallest) declared probability at
    that value and p_high the last (largest). A tie group is a JUMP of
    the declared CDF at a single value -- the forecaster is asserting
    an atom there: F approaches p_low from the left and the
    right-continuous CDF takes F(value) = p_high at the value itself.
    """
    uniq, p_low, p_high = [], [], []
    for v, p in zip(vals, probs):
        if uniq and v == uniq[-1]:
            p_high[-1] = p
        else:
            uniq.append(v)
            p_low.append(p)
            p_high.append(p)
    return (np.asarray(uniq, dtype=float),
            np.asarray(p_low, dtype=float),
            np.asarray(p_high, dtype=float))


def _fc_deriv_10770(d_prev, d_next, h_prev, h_next):
    """Fritsch-Carlson interior derivative (weighted harmonic mean of
    adjacent secants, as in scipy's PchipInterpolator). Secants here
    are always > 0 (declared probabilities are strictly increasing),
    so the sign screen of the general formula never binds."""
    w1 = 2.0 * h_next + h_prev
    w2 = h_next + 2.0 * h_prev
    return (w1 + w2) / (w1 / d_prev + w2 / d_next)


def _cdf_from_tied_percentiles(nominals, vals, probs, interp):
    """SF-BL-10770: CDF on `nominals` from declared percentiles whose
    values contain ties, with EXPLICIT, IDENTICAL tie handling for
    both interp paths.

    Pre-fix the two paths disagreed by up to 0.45-0.50 in sup norm on
    tied declarations: the linear path handed duplicate x-values to
    np.interp, whose behaviour over non-increasing xp is
    implementation-defined (on this numpy it happens to jump to the
    LAST tied probability at the tied value), while the pchip path
    silently collapsed each tie group to its FIRST probability and
    smoothed the jump away over the next segment -- not two smoothings
    of one curve but two different curves.

    The shared construction (the ONLY part that differs between the
    paths is the interpolation BETWEEN distinct values, exactly as in
    the untied case):

      * group ties as in _tied_percentile_groups;
      * at a tied value the CDF is a jump: nominals below the value
        see the segment ending at p_low, a nominal exactly AT the
        value takes p_high, and the segment above the value starts
        from p_high;
      * tails are constant extrapolation from the OUTER edge of the
        end groups (left tail = p_low of the first group, right tail
        = p_high of the last), matching the untied convention;
      * linear interpolates each inter-value segment linearly; pchip
        interpolates each segment with a monotone cubic Hermite whose
        endpoint derivatives are Fritsch-Carlson means of the
        neighbouring segment secants -- but NEVER across a jump: a
        segment endpoint at a tied value uses that segment's own
        secant, because smoothing across a discontinuity is precisely
        the defect being fixed;
      * a single distinct value (all tied) is a pure step from p_low
        (strictly below the value) to p_high (at and above it).

    Only called when a tie is present; untied inputs keep the
    original code paths untouched and stay bit-identical.
    """
    if interp not in ("linear", "pchip"):
        raise ValueError(f"unknown interp={interp!r}")
    uniq, p_low, p_high = _tied_percentile_groups(vals, probs)
    x = np.asarray(nominals, dtype=float)
    cdf = np.empty(x.shape, dtype=float)
    if uniq.size == 1:
        cdf[:] = np.where(x < uniq[0], p_low[0], p_high[0])
        return cdf
    cdf[x < uniq[0]] = p_low[0]
    cdf[x > uniq[-1]] = p_high[-1]
    for i, v in enumerate(uniq):
        cdf[x == v] = p_high[i]
    h = np.diff(uniq)
    # Segment secants: from the jump-top (p_high) of the left value
    # to the jump-bottom (p_low) of the right value. All > 0.
    d = (p_low[1:] - p_high[:-1]) / h
    tied = p_high > p_low
    for s in range(uniq.size - 1):
        mask = (x > uniq[s]) & (x < uniq[s + 1])
        if not np.any(mask):
            continue
        t = (x[mask] - uniq[s]) / h[s]
        y0, y1 = p_high[s], p_low[s + 1]
        if interp == "linear":
            # np.interp on the segment endpoints: identical arithmetic
            # to the untied scaffold path, so a tied panel whose tied
            # value falls between grid points stays bit-identical to
            # the pre-fix linear output wherever the jump does not
            # bind (cf. the SF-BL-10107 dup-values panel).
            cdf[mask] = np.interp(x[mask], [uniq[s], uniq[s + 1]],
                                  [y0, y1])
        else:
            if s == 0 or tied[s]:
                m0 = d[s]
            else:
                m0 = _fc_deriv_10770(d[s - 1], d[s], h[s - 1], h[s])
            if s == d.size - 1 or tied[s + 1]:
                m1 = d[s]
            else:
                m1 = _fc_deriv_10770(d[s], d[s + 1], h[s], h[s + 1])
            t2 = t * t
            t3 = t2 * t
            cdf[mask] = ((2.0 * t3 - 3.0 * t2 + 1.0) * y0
                         + (t3 - 2.0 * t2 + t) * h[s] * m0
                         + (-2.0 * t3 + 3.0 * t2) * y1
                         + (t3 - t2) * h[s] * m1)
    return cdf


def _validate_cdf_size_10311(size):
    """SF-BL-10311: validate percentiles_to_cdf's size argument
    BEFORE any grid is constructed, returning it as a plain int.

    Pre-fix percentiles_to_cdf never validated size itself: it
    forwarded the argument unchecked into grid_nominals /
    np.linspace, so in the live (superforecaster_out) copy size=0
    silently returned an EMPTY CDF [] and size=1 a 1-point CDF
    [0.1] -- a degenerate object downstream scoring
    (crps_from_cdf, _cell_density, which need at least 2 points)
    cannot handle -- True/False were read as 1/0 with the same
    results, negative sizes leaked a bare numpy ValueError that
    never named size, and non-integer sizes ('201', None, 201.5,
    NaN/inf, containers) leaked bare TypeErrors from linspace /
    the comparison, naming neither size nor the defect. In the
    repo copy integer size<2 was caught only collaterally, and
    mislabelled, by grid_nominals' SF-BL-10108 screen; the
    non-integer cases leaked the same bare TypeErrors there.
    This screen is percentiles_to_cdf's own: any size that is
    not an integer >= 2 -- bool / np.bool_, string / bytes, None,
    containers, complex, non-finite or non-integral numbers --
    raises a ValueError naming size, at entry, before grid
    construction. Genuinely integral sizes (int / np.integer,
    integral float / Decimal / Fraction, 0-d integral numeric
    arrays) are normalised to the identical int, so lawful
    outputs -- above all the default 201-point CDF -- stay
    byte-identical.
    """
    def _raise():
        raise ValueError(
            "percentiles_to_cdf requires size to be an integer "
            ">= 2 (the number of CDF grid points; a CDF needs at "
            f"least 2 points); got size={size!r} (SF-BL-10311; "
            "cf. SF-BL-10108)"
        )
    if isinstance(size, (bool, np.bool_)):
        _raise()
    _val = size
    if isinstance(size, np.ndarray):
        if size.ndim != 0 or size.dtype == np.dtype(bool) \
                or size.dtype.kind not in "iuf":
            _raise()
        _val = size.item()
        if isinstance(_val, (bool, np.bool_)):
            _raise()
    if isinstance(_val, numbers.Integral):
        _n = int(_val)
    elif isinstance(_val, numbers.Number) \
            and not isinstance(_val, (complex, np.complexfloating)):
        try:
            _f = float(_val)
        except (TypeError, ValueError, OverflowError):
            _raise()
        if not np.isfinite(_f) or _f != int(_f):
            _raise()
        _n = int(_f)
    else:
        _raise()
    if _n < 2:
        _raise()
    return _n


def percentiles_to_cdf(percentiles, range_min, range_max, zero_point=None,
                       size=CDF_SIZE, interp="linear"):
    """Declared percentiles -> monotone `size`-point CDF on the question grid.

    interp="linear" reproduces the scaffold construction exactly.
    interp="pchip" uses a monotone cubic through (value, prob).
    Tied declared values are jumps, handled explicitly and
    identically for both interp values (SF-BL-10770).
    """
    # SF-BL-10311: validate size at entry, BEFORE grid construction
    # (see _validate_cdf_size_10311); the normalised int is what
    # grid_nominals receives, so lawful sizes are untouched.
    size = _validate_cdf_size_10311(size)
    probs, vals = _validate_percentiles(percentiles)
    # SF-BL-10107 (defence in depth): non-finite grid bounds would also
    # yield a non-finite CDF; finite bounds are unaffected.
    if not (np.isfinite(float(range_min)) and np.isfinite(float(range_max))):
        raise ValueError(
            "range_min and range_max must be finite; got "
            f"range_min={range_min!r}, range_max={range_max!r} (SF-BL-10107)"
        )
    if zero_point is not None and not np.isfinite(float(zero_point)):
        raise ValueError(
            f"zero_point must be finite; got zero_point={zero_point!r} "
            "(SF-BL-10107)"
        )
    nominals = grid_nominals(range_min, range_max, zero_point, size)
    if np.any(np.diff(vals) == 0):
        # SF-BL-10770: tied declared values -- explicit jump handling,
        # identical for both interp paths (see
        # _cdf_from_tied_percentiles). Untied inputs never enter this
        # branch, so their outputs stay bit-identical.
        cdf = _cdf_from_tied_percentiles(nominals, vals, probs, interp)
        cdf = np.clip(cdf, 0.0, 1.0)
        cdf = np.maximum.accumulate(cdf)  # enforce monotone CDF
        return cdf
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


def _checked_cdf_at_grid_arg_11308(arg, name):
    """SF-BL-11308: type-validate one cdf_at grid argument (cdf or
    nominals) BEFORE any float coercion, returning the argument to
    feed the existing conversion / np.interp call.

    Pre-fix cdf_at coerced both grid arguments with
    np.asarray(..., dtype=float) / np.interp, which silently parse
    strings and read bools as 1.0/0.0: live repros at value=50 on
    cdf [0.0, 0.5, 1.0] over nominals [0, 50, 100] (control 0.5)
    gave nominals=['0', '50', '100'] -> 0.5 bit-identical to the
    numeric control, nominals=[False, True, 100] -> 0.7475 (bool
    grid read as [0, 1, 100]) and cdf=[False, True, True] -> 1.0,
    while a string cdf in the scaffold copy died with a bare numpy
    cast TypeError naming nothing (and in this copy was parsed to
    the control 0.5). A bool is not a number (fleet convention)
    and a string element is a caller type bug, not a grid point:
    any bool / np.bool_ / str / bytes element of either array --
    and any other non-numeric element (None, containers, complex,
    non-scalar or bool/string-dtype arrays), a bool/string-dtype
    array itself, or a bare bool/string argument -- now raises a
    ValueError naming that argument (cdf or nominals). Distinct
    from SF-BL-11270 (this function's value TYPE only, still
    queued) and SF-BL-10109 (value/grid finiteness and ordering --
    the value DOMAIN, screened after this in the repo copy);
    neither fixtured the cdf or nominals type domain.
    Reject-only: genuinely numeric elements (int / float / numpy
    numeric scalars / Decimal / Fraction / 0-d numeric arrays)
    fall through and the argument is returned unchanged (a
    one-shot iterable is materialised once so the scan does not
    consume it), so numeric-grid interpolations stay
    bit-identical.
    """
    def _raise(detail):
        raise ValueError(
            f"cdf_at requires {name} to be a sequence of real "
            "numbers (a bool or string is not a number); got "
            f"{detail} in {name}={arg!r} (SF-BL-11308)"
        )
    if isinstance(arg, (bool, np.bool_)):
        _raise("a bool")
    if isinstance(arg, (str, bytes)):
        _raise("a string")
    if isinstance(arg, np.ndarray):
        if arg.dtype == np.dtype(bool) or arg.dtype.kind in "US":
            _raise(f"a {arg.dtype} array")
        if arg.dtype.kind in "iuf":
            return arg
        # Object / complex / datetime / other dtypes: scan the
        # elements (tolist() yields the original scalars).
        elements = arg.ravel().tolist()
    elif isinstance(arg, (list, tuple, range)):
        elements = list(arg)
    elif hasattr(arg, "__iter__"):
        elements = list(arg)
        arg = elements
    else:
        # A scalar / non-iterable argument has no elements to
        # type-screen; the downstream size/finiteness screens (or
        # np.interp) handle it exactly as pre-fix.
        return arg
    for _i_11308, _el_11308 in enumerate(elements):
        _bad_11308 = False
        if isinstance(_el_11308, (bool, np.bool_)):
            _bad_11308 = True
        elif isinstance(_el_11308, (str, bytes)):
            _bad_11308 = True
        elif isinstance(_el_11308, (complex, np.complexfloating)):
            _bad_11308 = True
        elif isinstance(_el_11308, np.ndarray):
            if _el_11308.ndim != 0 \
                    or _el_11308.dtype == np.dtype(bool) \
                    or _el_11308.dtype.kind not in "iuf":
                _bad_11308 = True
        elif not isinstance(_el_11308, numbers.Number):
            _bad_11308 = True
        if _bad_11308:
            _raise(f"element {_el_11308!r} at index {_i_11308}")
    return arg


def cdf_at(cdf, nominals, value):
    """Interpolated CDF value F(value) from a grid CDF."""
    # SF-BL-11308: type-screen cdf and nominals (naming the
    # argument) BEFORE any float coercion -- see
    # _checked_cdf_at_grid_arg_11308.
    cdf = np.asarray(
        _checked_cdf_at_grid_arg_11308(cdf, "cdf"), dtype=float)
    nominals = np.asarray(
        _checked_cdf_at_grid_arg_11308(nominals, "nominals"),
        dtype=float)
    # SF-BL-10109: validate the grid BEFORE interpolating. np.interp
    # assumes xp (nominals) is increasing without checking, so a
    # reversed/unsorted grid silently returned the wrong endpoint value
    # (live repro: cdf_at([0.2, 0.5, 0.9], nominals=[2, 1, 0],
    # value=1.0) -> 0.9), and it propagated non-finite inputs silently
    # (value=NaN -> NaN, +/-inf query values pinned to an endpoint,
    # NaN/+/-inf cdf or nominal entries returning NaN/inf/plausible
    # numbers). cdf_at feeds cdf_to_bin_probs, so the corruption
    # reached multiple-choice probabilities. Every screen below raises
    # a named ValueError; lawful (matched, non-empty, finite, strictly
    # increasing) grids fall through to the identical np.interp call,
    # so ordered-grid results stay bit-identical.
    if cdf.size != nominals.size:
        raise ValueError(
            "cdf_at grid mismatch: cdf and nominals must have the same "
            f"number of grid points (mismatched nominals); got cdf size "
            f"{cdf.size} and nominals size {nominals.size} (SF-BL-10109)"
        )
    if cdf.size == 0:
        raise ValueError(
            "cdf_at requires a non-empty grid; got an empty grid "
            "(cdf and nominals both size 0) (SF-BL-10109)"
        )
    if not np.all(np.isfinite(nominals)):
        raise ValueError(
            "cdf_at nominals must be finite; got non-finite nominal "
            f"value(s) {nominals[~np.isfinite(nominals)]} (SF-BL-10109)"
        )
    if not np.all(np.isfinite(cdf)):
        raise ValueError(
            "cdf_at cdf values must be finite; got non-finite cdf "
            f"value(s) {cdf[~np.isfinite(cdf)]} (SF-BL-10109)"
        )
    value_arr = np.asarray(value, dtype=float)
    if value_arr.size != 1 or not np.all(np.isfinite(value_arr)):
        raise ValueError(
            "cdf_at query value must be finite (a single finite "
            f"number); got non-finite or non-scalar value {value!r} "
            "(SF-BL-10109)"
        )
    if cdf.size >= 2 and np.any(np.diff(nominals) <= 0):
        raise ValueError(
            "cdf_at nominals must be strictly increasing (sorted in "
            "increasing order); got non-increasing/unsorted nominals "
            f"{nominals} (SF-BL-10109)"
        )
    return float(np.interp(float(value_arr), nominals, cdf))


def validate_cdf(cdf, max_step_warn=0.19, expected_size=None):
    """Return list of problems (empty = valid), mirroring scaffold checks.

    SF-BL-10426: expected_size is the caller's known CDF grid size
    (201 for a numeric question; inbound_outcome_count+1 for a
    discrete one). When supplied, any CDF whose length differs
    reports a length-mismatch problem; when None (default), only
    the unconditional length floor (at least 2 points) is enforced,
    so CDFs on other lawful grids are not falsely flagged.
    """
    # SF-BL-11133: type-validate max_step_warn BEFORE any use.
    # Pre-fix the threshold was first touched by
    # steps.max() > max_step_warn with no screen at all: True was
    # read as 1.0, so validate_cdf([0.0, 0.9, 1.0],
    # max_step_warn=True) returned [] -- the 0.9 step the check
    # exists to flag (flagged at the 0.19 default) was certified
    # valid -- and False was read as 0.0 and formatted into the
    # problem string ("... > False"); '0.19' died with a bare
    # numpy UFuncTypeError and None with a bare TypeError, neither
    # naming max_step_warn; an empty / single-point CDF never
    # reached the comparison, so a bad threshold passed silently
    # there too. A bool is not a number (fleet convention, cf.
    # the SF-BL-10471 screen of the cdf argument below), and a
    # string / None / non-numeric threshold is a caller bug, not
    # a CDF defect, so -- unlike the cdf argument, whose defects
    # are RETURNED as problems -- a bad threshold RAISES a
    # ValueError naming max_step_warn, at entry, whatever the CDF.
    # Distinct from SF-BL-10355, which covers the VALUE domain
    # of a numeric threshold (non-finite / negative); this
    # screen is type-only. Genuinely numeric thresholds (int /
    # float / numpy numeric, incl. the 0.19 default) fall
    # through untouched: the original object is kept for the
    # comparison and problem formatting below, so
    # numeric-threshold problem lists stay bit-identical (an
    # int threshold 1 still formats as "1", not "1.0").
    if isinstance(max_step_warn, (bool, np.bool_)):
        raise ValueError(
            "validate_cdf requires max_step_warn to be a real "
            "number (a bool is not a number); got "
            f"max_step_warn={max_step_warn!r} (SF-BL-11133)"
        )
    if isinstance(max_step_warn, (str, bytes)) \
            or max_step_warn is None:
        raise ValueError(
            "validate_cdf requires max_step_warn to be a real "
            f"number; got max_step_warn={max_step_warn!r} "
            "(SF-BL-11133)"
        )
    if isinstance(max_step_warn, np.ndarray) \
            and max_step_warn.ndim > 0:
        raise ValueError(
            "validate_cdf requires max_step_warn to be a real "
            "number (a scalar); got "
            f"max_step_warn={max_step_warn!r} (SF-BL-11133)"
        )
    try:
        float(max_step_warn)
    except (TypeError, ValueError):
        raise ValueError(
            "validate_cdf requires max_step_warn to be a real "
            f"number; got max_step_warn={max_step_warn!r} "
            "(SF-BL-11133)"
        ) from None
    # SF-BL-10426: validate expected_size (when supplied) at entry,
    # alongside the max_step_warn screen above. The expected grid
    # size is a caller-supplied contract, not a CDF defect, so --
    # like a bad max_step_warn -- a bad expected_size RAISES a
    # ValueError naming expected_size, whatever the CDF: a bool is
    # not a size (fleet convention), and a string / None-like /
    # non-integral / non-finite size, or any size < 2 (a CDF grid
    # needs at least 2 points, cf. grid_nominals SF-BL-10108 and
    # percentiles_to_cdf SF-BL-10311), is a caller bug. Genuinely
    # integral sizes (int / np.integer / integral float / Decimal /
    # Fraction / 0-d integral numeric arrays) are normalised to the
    # identical int; None keeps its meaning (grid size unknown, no
    # mismatch check). The length DEFECTS themselves are returned
    # as problems below, per this validator's contract.
    if expected_size is not None:
        _expected_size = None
        _bad_expected = isinstance(expected_size, (bool, np.bool_)) \
            or isinstance(expected_size, (str, bytes))
        if not _bad_expected:
            _cand = expected_size
            if isinstance(_cand, np.ndarray):
                if _cand.ndim != 0 \
                        or _cand.dtype == np.dtype(bool) \
                        or _cand.dtype.kind not in "iuf":
                    _bad_expected = True
                else:
                    _cand = _cand.item()
                    if isinstance(_cand, (bool, np.bool_)):
                        _bad_expected = True
            if not _bad_expected:
                if isinstance(_cand, numbers.Integral):
                    _expected_size = int(_cand)
                elif isinstance(_cand, numbers.Number) \
                        and not isinstance(
                            _cand, (complex, np.complexfloating)):
                    try:
                        _f_expected = float(_cand)
                    except (TypeError, ValueError, OverflowError):
                        _bad_expected = True
                    else:
                        if not np.isfinite(_f_expected) \
                                or _f_expected != int(_f_expected):
                            _bad_expected = True
                        else:
                            _expected_size = int(_f_expected)
                else:
                    _bad_expected = True
        if _bad_expected or _expected_size is None \
                or _expected_size < 2:
            raise ValueError(
                "validate_cdf requires expected_size to be an "
                "integer >= 2 (the expected CDF grid size) or "
                f"None; got expected_size={expected_size!r} "
                "(SF-BL-10426)"
            )
        expected_size = _expected_size
    problems = []
    # SF-BL-10471: screen for bool / non-numeric input BEFORE float
    # coercion. Pre-fix the first line was
    # np.asarray(cdf, dtype=float), which coerces True -> 1.0 and
    # False -> 0.0 before every check, so validate_cdf([True] * 201)
    # returned [] -- an all-bool panel passed range, monotonicity
    # and step checks as the constant CDF 1.0, indistinguishable
    # from a degenerate forecast. A bool is not a number (fleet
    # convention, cf. distribution_validity._is_number and
    # metaculus_client.validate_forecast_payload's CDF branch,
    # SF-BL-10387, a DISTINCT function whose all-bool fix does not
    # cover this scorer-side gate), and non-numeric members (str,
    # None, ...) either coerced silently ("0.5" -> 0.5) or died with
    # a bare ValueError/TypeError from the float conversion although
    # this validator's contract is to RETURN a list of problems.
    # Bool dtype is detected on the raw input -- a numpy bool array
    # by dtype, any other sequence elementwise (a mixed list such as
    # [0.0, True, 1.0] has no bool dtype to inspect once asarray has
    # homogenised it to float, so the scan must run on the original
    # elements, np.bool_ included). Named bool / non-numeric
    # problems are returned on their own: the numeric checks below
    # are meaningless over coerced members. Genuinely numeric input
    # (int/float lists, tuples and arrays, incl. NaN/inf floats)
    # falls through to the identical coercion and checks below, so
    # its problems lists are unchanged.
    # SF-BL-11166: string / object-dtype elements of cdf RAISE a
    # TypeError naming cdf, instead of being validated on coerced
    # values or merely RETURNED as a problems list. A JSON-decoded
    # submission whose CDF serialised as strings ('0.1', '0.5',
    # ...) is malformed input, not a numeric CDF with a defect:
    # pre-fix (post-SF-BL-10471) validate_cdf(['0.1', '0.5',
    # '0.9']) returned the non-numeric problems list (and in the
    # original live repro, before 10471's screen, it returned
    # ['largest CDF step 0.400 > 0.19'] on the values parsed
    # through np.asarray(dtype=float)) instead of raising, so a
    # string-typed panel passed through this gate as data rather
    # than being rejected as a type error. The element scan in
    # the SF-BL-10471 screen below therefore also collects
    # string/bytes elements (str / bytes, incl. np.str_ /
    # np.bytes_ and the elements of string-dtype and
    # object-dtype ndarrays, whose tolist() yields the original
    # strings) and bare object() elements, plus a bare str/bytes
    # cdf itself; any of those raises TypeError naming cdf at
    # this point, before the bool / non-numeric problems are
    # returned. Scope is deliberately narrow and SUPERSEDES
    # SF-BL-10471 only for string/object elements: bool elements
    # and other non-numeric elements (None, dict/list members,
    # ...) keep 10471's returned problems, object-dtype arrays
    # carrying only numeric / bool / None elements keep their
    # prior elementwise treatment, and genuinely numeric
    # (float/int) CDFs fall through untouched, so float-CDF
    # problem lists stay bit-identical. Distinct from
    # SF-BL-10471 (the bool CDF of this same argument, returned
    # as a problem) and SF-BL-11133 (the max_step_warn argument's
    # type, raised as ValueError above).
    _string_indices = []
    _object_indices = []
    _string_container = False
    _bool_indices = []
    _non_numeric_indices = []
    _non_numeric_type = None
    _materialised = None
    if isinstance(cdf, (bool, np.bool_)):
        _bool_indices = [0]
    elif isinstance(cdf, np.ndarray):
        if cdf.dtype == np.dtype(bool):
            _bool_indices = list(range(cdf.size))
        elif cdf.dtype.kind in "biuf":
            pass  # genuinely numeric dtype: no element scan needed
        else:
            # Object / string / datetime / complex arrays: inspect
            # the elements (tolist() yields Python scalars).
            for _i, _v in enumerate(cdf.ravel().tolist()):
                if isinstance(_v, (str, bytes)):
                    _string_indices.append(_i)
                elif type(_v) is object:
                    _object_indices.append(_i)
                elif isinstance(_v, (bool, np.bool_)):
                    _bool_indices.append(_i)
                elif not isinstance(
                        _v, (int, float, np.integer, np.floating)):
                    _non_numeric_indices.append(_i)
    elif isinstance(cdf, (str, bytes)):
        _string_container = True
    elif isinstance(cdf, dict):
        _non_numeric_type = type(cdf).__name__
    elif hasattr(cdf, "__iter__"):
        _materialised = list(cdf)
        for _i, _v in enumerate(_materialised):
            if isinstance(_v, (str, bytes)):
                _string_indices.append(_i)
            elif type(_v) is object:
                _object_indices.append(_i)
            elif isinstance(_v, (bool, np.bool_)):
                _bool_indices.append(_i)
            elif not isinstance(
                    _v, (int, float, np.integer, np.floating)):
                _non_numeric_indices.append(_i)
    elif type(cdf) is object:
        _object_indices = [0]
    elif not isinstance(cdf, (int, float, np.integer, np.floating)):
        _non_numeric_type = type(cdf).__name__
    if _string_container or _string_indices or _object_indices:
        _bad_indices = sorted(_string_indices + _object_indices)
        if _string_container:
            _detail = f"cdf itself is a {type(cdf).__name__}"
        else:
            _detail = ("string or object-dtype element(s) at "
                       f"indices {_bad_indices}")
        raise TypeError(
            "validate_cdf requires cdf to be a sequence of real "
            f"numbers; got {_detail} in cdf "
            "(a string / object element is not a number; "
            "SF-BL-11166)"
        )
    if _bool_indices:
        problems.append(
            "CDF contains boolean values at indices "
            f"{_bool_indices} (a bool is not a number)"
        )
    if _non_numeric_indices:
        problems.append(
            "CDF contains non-numeric values at indices "
            f"{_non_numeric_indices} (a CDF member must be a number)"
        )
    if _non_numeric_type is not None:
        problems.append(
            "CDF contains non-numeric values "
            f"(type {_non_numeric_type}: a CDF must be a sequence "
            "of numbers)"
        )
    if problems:
        return problems
    cdf = np.asarray(
        _materialised if _materialised is not None else cdf,
        dtype=float)
    # SF-BL-10426: enforce CDF length BEFORE the value checks
    # below. Pre-fix every check iterated over the values / diffs
    # of the CDF, so for an empty or single-point CDF there was
    # nothing to iterate: validate_cdf([]) == [] and
    # validate_cdf([0.5]) == [] -- a zero-length forecast passed
    # this module's own validity gate, even though the design
    # contract is a 201-point CDF (discrete: outcome_count + 1
    # points) and downstream consumers (np.diff-based checks,
    # trapezoid scoring, cell densities) need at least 2 points;
    # a bare numeric scalar died instead with a bare numpy
    # ValueError from np.diff ("diff requires input that is at
    # least one dimensional"). A CDF with fewer than 2 points now
    # reports a length problem (its size is cdf.size, so a scalar
    # counts as length 1 and is reported, not raised), and when
    # the caller supplied expected_size (the grid size is known)
    # any other length reports a mismatch problem. Length
    # problems lead the list; the pre-existing checks below still
    # run, so a short CDF's other defects (non-finite, out of
    # range, ...) are still named after the length problem.
    # Scope: this SUPERSEDES the length-0/1 panels pinned by
    # SF-BL-9901 / SF-BL-10471 / SF-BL-11133 (which asserted the
    # [] / [0.5] / [2.0] problems lists were unchanged) for
    # lengths 0 and 1 only; every CDF with length >= 2 and
    # expected_size=None keeps its pre-fix problems list
    # bit-identical, as does a CDF whose length equals a supplied
    # expected_size. Distinct from SF-BL-9901 (non-finite VALUES
    # inside a CDF, screened below) and SF-BL-10269
    # (crps_from_cdf scoring an empty grid 0.0 -- a different
    # function, which raises).
    if cdf.ndim == 0:
        # A bare numeric scalar is a single-point CDF (size 1):
        # reshape it so the diff-based checks below stay
        # well-defined instead of dying in np.diff, as pre-fix.
        cdf = cdf.reshape(1)
    if cdf.size == 0:
        problems.append(
            "CDF length 0 is too short: an empty CDF is not a "
            "valid forecast (a CDF needs at least 2 points)"
        )
    elif cdf.size == 1:
        problems.append(
            "CDF length 1 is too short: a single-point CDF is "
            "not a valid forecast (a CDF needs at least 2 points)"
        )
    if expected_size is not None and cdf.size != expected_size:
        problems.append(
            f"CDF length {cdf.size} does not match expected "
            f"grid size {expected_size}"
        )
    # SF-BL-9901: screen for non-finite values FIRST, before any
    # comparison. Every check below is NaN-blind -- (cdf<0)|(cdf>1),
    # diff < -1e-12 and steps.max() > max_step_warn are all False /
    # NaN-poisoned for a NaN entry -- so pre-fix
    # validate_cdf([0.1, NaN, 0.9]) returned [] and a corrupt pooled
    # CDF was certified valid with zero warnings; +/-inf entries only
    # tripped the other checks incidentally, never naming the actual
    # defect. Any NaN/+inf/-inf entry now reports a non-finite-values
    # problem. Finite CDFs fall through to the identical checks below,
    # so their problems lists are unchanged (kill criterion).
    if not np.all(np.isfinite(cdf)):
        problems.append("CDF contains non-finite values")
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


def _screen_probs_types_11243(probs, caller):
    """SF-BL-11243: type-validate a probs argument BEFORE any float
    coercion, returning it as a float ndarray.

    A bool is not a number (fleet convention, cf. SF-BL-10471 /
    SF-BL-11133 / SF-BL-11134 in this module) and a string is not a
    probability, but both survive naive float coercion: float(True)
    is 1.0 and np.asarray(['0.5'], dtype=float) parses the string.
    The screen therefore inspects the RAW elements -- the container
    itself, an ndarray's dtype, and each element of any other
    sequence -- and raises a ValueError naming probs on any bool or
    string/bytes element (or any other non-numeric element, which
    pre-fix died with a bare TypeError naming nothing). Genuinely
    numeric probs (int / float / numpy numeric scalars, lists,
    tuples and numeric ndarrays) are returned as the identical
    float array np.asarray(probs, dtype=float) would produce, so
    numeric-probs outputs stay bit-identical.
    """
    def _raise(detail):
        raise ValueError(
            f"{caller} requires probs to be a sequence of real "
            f"numbers (a bool or string is not a number); got "
            f"{detail} in probs={probs!r} (SF-BL-11243)"
        )
    if isinstance(probs, (bool, np.bool_, str, bytes)):
        _raise(f"probs itself of type {type(probs).__name__}")
    if isinstance(probs, dict):
        _raise(f"probs itself of type {type(probs).__name__}")
    elements = None
    if isinstance(probs, np.ndarray):
        if probs.dtype == np.dtype(bool) or probs.dtype.kind in "US":
            _raise(f"probs array with dtype {probs.dtype}")
        if probs.dtype.kind in "iuf":
            return np.asarray(probs, dtype=float)
        if probs.dtype.kind != "O":
            _raise(f"probs array with dtype {probs.dtype}")
        elements = probs.ravel().tolist()
    else:
        try:
            elements = list(probs)
        except TypeError:
            _raise(f"non-sequence probs of type "
                   f"{type(probs).__name__}")
    for _i, _v in enumerate(elements):
        if isinstance(_v, (bool, np.bool_)):
            _raise(f"boolean probs element at index {_i}: {_v!r}")
        if isinstance(_v, (str, bytes, np.str_, np.bytes_)):
            _raise(f"string probs element at index {_i}: {_v!r}")
        if isinstance(_v, np.ndarray):
            if _v.ndim != 0 or _v.dtype == np.dtype(bool) \
                    or _v.dtype.kind not in "iuf":
                _raise(f"non-numeric probs element at index {_i}: "
                       f"{_v!r}")
            continue
        if not isinstance(_v, (int, float, np.integer, np.floating,
                               numbers.Number)):
            _raise(f"non-numeric probs element at index {_i}: "
                   f"{_v!r} of type {type(_v).__name__}")
    try:
        return np.asarray(elements, dtype=float)
    except (TypeError, ValueError):
        _raise(f"probs={probs!r} not convertible to float")


def _screen_trailing_h_returns_types_11312(trailing_h_returns, caller):
    """SF-BL-11312: type-validate a trailing_h_returns argument
    BEFORE any float coercion, returning it as a float ndarray.

    A bool is not a number (fleet convention, cf. SF-BL-10471 /
    SF-BL-11133 / SF-BL-11243 in this module) and a string is not a
    log-return, but both survive naive float coercion:
    np.asarray(..., dtype=float) reads True/False as 1.0/0.0 and
    parses numeric strings, so pre-fix a 40-element string return
    panel returned a percentile dict bit-identical to the numeric
    control and an all-True bool panel returned spot * exp(1) =
    271.83 at the 0.95 quantile (vs 103.05 control). The screen
    therefore inspects the RAW returns -- the container itself, an
    ndarray's dtype, and each element of any other sequence -- and
    raises a ValueError naming trailing_h_returns on any bool or
    string/bytes element (or any other non-numeric element, which
    pre-fix either coerced silently or died with a bare error
    naming nothing). Genuinely numeric returns (int / float /
    numpy numeric scalars, Decimal / Fraction, lists, tuples and
    numeric ndarrays, incl. non-finite values, which the caller's
    isfinite filter still drops) are returned as the identical
    float array np.asarray(trailing_h_returns, dtype=float)
    would produce, so numeric-return outputs stay bit-identical.
    """
    def _raise(detail):
        raise ValueError(
            f"{caller} requires trailing_h_returns to be a "
            f"sequence of real numbers (a bool or string is not "
            f"a number); got {detail} in "
            f"trailing_h_returns={trailing_h_returns!r} "
            "(SF-BL-11312)"
        )
    if isinstance(trailing_h_returns, (bool, np.bool_, str, bytes)):
        _raise(f"trailing_h_returns itself of type "
               f"{type(trailing_h_returns).__name__}")
    if isinstance(trailing_h_returns, dict):
        _raise(f"trailing_h_returns itself of type "
               f"{type(trailing_h_returns).__name__}")
    if isinstance(trailing_h_returns, np.ndarray):
        if trailing_h_returns.dtype == np.dtype(bool) \
                or trailing_h_returns.dtype.kind in "US":
            _raise(f"trailing_h_returns array with dtype "
                   f"{trailing_h_returns.dtype}")
        if trailing_h_returns.dtype.kind in "iuf":
            return np.asarray(trailing_h_returns, dtype=float)
        if trailing_h_returns.dtype.kind != "O":
            _raise(f"trailing_h_returns array with dtype "
                   f"{trailing_h_returns.dtype}")
        elements = trailing_h_returns.ravel().tolist()
    elif isinstance(trailing_h_returns, numbers.Number):
        # Numeric scalar: fall through as the identical 0-d float
        # array the unscreened conversion produced, so the
        # caller's >= 30 screen still governs it.
        return np.asarray(trailing_h_returns, dtype=float)
    else:
        try:
            elements = list(trailing_h_returns)
        except TypeError:
            _raise(f"non-sequence trailing_h_returns of type "
                   f"{type(trailing_h_returns).__name__}")
    for _i, _v in enumerate(elements):
        if isinstance(_v, (bool, np.bool_)):
            _raise(f"boolean trailing_h_returns element at index "
                   f"{_i}: {_v!r}")
        if isinstance(_v, (str, bytes, np.str_, np.bytes_)):
            _raise(f"string trailing_h_returns element at index "
                   f"{_i}: {_v!r}")
        if isinstance(_v, np.ndarray):
            if _v.ndim != 0 or _v.dtype == np.dtype(bool) \
                    or _v.dtype.kind not in "iuf":
                _raise(f"non-numeric trailing_h_returns element "
                       f"at index {_i}: {_v!r}")
            continue
        if not isinstance(_v, (int, float, np.integer, np.floating,
                               numbers.Number)):
            _raise(f"non-numeric trailing_h_returns element at "
                   f"index {_i}: {_v!r} of type "
                   f"{type(_v).__name__}")
    try:
        return np.asarray(elements, dtype=float)
    except (TypeError, ValueError):
        _raise(f"trailing_h_returns={trailing_h_returns!r} not "
               f"convertible to float")


def climatology_percentiles(spot, trailing_h_returns, probs=DEFAULT_PERCENTILES):
    """Empirical quantiles of trailing h-day log-returns applied to spot."""
    # SF-BL-11312: type-validate trailing_h_returns AT ENTRY,
    # before any use (and before the probs / spot screens below,
    # so a corrupt return is named even when another argument is
    # also bad). Pre-fix trailing_h_returns was never
    # type-validated: it flowed unchecked into
    # np.asarray(trailing_h_returns, dtype=float), which parses
    # string returns and reads True/False as 1.0/0.0 log-returns,
    # so in the live repros (spot=100) a 40-element string return
    # panel returned a percentile dict bit-identical to the
    # numeric control and a bool panel returned a 0.95 percentile
    # of 271.83 (vs 103.05 control) -- fabricated extreme
    # quantiles consumed downstream as if lawful. Distinct from
    # SF-BL-10266 below (this function's spot value domain) and
    # SF-BL-11243 (this function's probs type) -- a different
    # argument in each case. Every bool or string-dtype
    # trailing_h_returns element now raises a ValueError naming
    # trailing_h_returns; numeric returns fall through as the
    # identical float array, so numeric-return dicts stay
    # bit-identical (kill criterion).
    trailing_arr = _screen_trailing_h_returns_types_11312(
        trailing_h_returns, "climatology_percentiles")
    # SF-BL-11243: type-validate probs AT ENTRY, before any use (and
    # before the spot screen below, so a corrupt prob is named even
    # when another argument is also bad -- the ordering of
    # SF-BL-10474's probs screen in naive_normal_percentiles).
    # Pre-fix probs was never type-validated: it flowed unchecked
    # into np.quantile and _pct_dict, so on 40 trailing returns at
    # spot 100 probs=[True] returned {1.0: 103.032} (the bool read
    # as a 100th-percentile request and stored as dict key 1.0),
    # probs=[True, False] returned keys {1.0, 0.0}, and
    # probs=['0.5'] died with a bare "'<=' not supported between
    # instances of 'float' and 'numpy.str_'" TypeError inside
    # np.quantile's sorting, naming neither probs nor the defect.
    # Distinct from SF-BL-10266 below (this function's spot value)
    # and SF-BL-10474 (naive_normal_percentiles' probs VALUE domain;
    # this screen is type-only, so a numeric prob's value domain is
    # untouched). Every bool or string-dtype probs element now
    # raises a ValueError naming probs; numeric probs fall through
    # as the identical float array, so numeric-probs dicts stay
    # bit-identical (kill criterion).
    probs_arr = _screen_probs_types_11243(
        probs, "climatology_percentiles")
    # SF-BL-11346: type-validate spot BEFORE the SF-BL-10266 value
    # screen below can coerce it (and after the SF-BL-11312 /
    # SF-BL-11243 trailing-returns and probs screens above, so
    # their ordering guarantees are preserved). Pre-fix the value
    # screen's float(spot) read True as 1.0, so spot=True returned
    # a percentile dict centred on 1.0 (live repro, 40 trailing
    # returns: median 0.9974218461954439 vs 99.74218461954439 at
    # the spot=100 control), float('100') / float(b'100') silently
    # coerced a string spot to 100.0 and returned the control
    # dict, and on the unscreened wave3 original spot='100' died
    # with a bare UFuncTypeError naming nothing. A bool is not a
    # number (fleet convention) and a string/bytes/None spot is a
    # corrupt type, not a price: every bool / np.bool_ / str /
    # bytes / None spot, and any other non-numeric spot
    # (containers, arbitrary objects, non-scalar or
    # bool/string-dtype arrays), now raises a ValueError naming
    # spot, dual-tagged with SF-BL-10266 so that screen's cases
    # (None, 'abc', []) keep naming it. Distinct from
    # SF-BL-10266, which fixtures this function's spot VALUE
    # domain (finite, positive), and from SF-BL-11310 /
    # SF-BL-11311 (spot TYPE in disciplined_percentiles /
    # naive_normal_percentiles, different functions); this
    # screen is type-only. A genuinely numeric spot (int / float
    # / numpy numeric scalar / Decimal / Fraction / 0-d numeric
    # array) falls through untouched to the identical float
    # conversion below, so numeric-spot percentile dicts stay
    # bit-identical (kill criterion).
    if isinstance(spot, (bool, np.bool_)):
        raise ValueError(
            "climatology_percentiles requires spot to be a real "
            "number (a bool is not a number); got boolean "
            f"spot={spot!r} (SF-BL-11346; SF-BL-10266)"
        )
    if spot is None or isinstance(spot, (str, bytes)):
        raise ValueError(
            "climatology_percentiles requires spot to be a real "
            "number; got non-numeric "
            f"spot={spot!r} of type {type(spot).__name__} "
            "(SF-BL-11346; SF-BL-10266)"
        )
    if isinstance(spot, np.ndarray):
        if spot.ndim != 0 \
                or spot.dtype == np.dtype(bool) \
                or spot.dtype.kind not in "iufc":
            raise ValueError(
                "climatology_percentiles requires spot to be a "
                "real number (a scalar); got "
                f"spot={spot!r} of type ndarray with dtype "
                f"{spot.dtype} (SF-BL-11346; SF-BL-10266)"
            )
    elif not isinstance(spot, numbers.Number):
        raise ValueError(
            "climatology_percentiles requires spot to be a real "
            "number; got non-numeric "
            f"spot={spot!r} of type {type(spot).__name__} "
            "(SF-BL-11346; SF-BL-10266)"
        )
    # SF-BL-10266: validate spot BEFORE any use. The >=30-return screen
    # below filters the returns, but pre-fix spot was never screened, so
    # spot=NaN returned an all-NaN percentile dict, spot=+/-inf an
    # all-inf dict, spot=0 an all-zero dict, and a negative spot (e.g.
    # -100) returned percentiles in REVERSED order -- all of which
    # percentiles_to_cdf then consumed as if lawful (a non-numeric spot
    # died with a bare TypeError/UFuncTypeError naming nothing). A
    # price/level spot must be finite and positive; every corrupt spot
    # now raises a ValueError naming spot. Finite positive spots fall
    # through to the identical computation, so lawful outputs stay
    # bit-identical.
    try:
        spot_f = float(spot)
    except (TypeError, ValueError):
        raise ValueError(
            "climatology_percentiles requires a finite, positive "
            f"spot; got spot={spot!r} (SF-BL-10266)"
        ) from None
    if not np.isfinite(spot_f) or spot_f <= 0:
        raise ValueError(
            "climatology_percentiles requires a finite, positive "
            f"spot; got spot={spot!r} (SF-BL-10266)"
        )
    r = trailing_arr
    r = r[np.isfinite(r)]
    if len(r) < 30:
        raise ValueError("climatology needs >= 30 trailing h-day returns")
    qs = np.quantile(r, probs_arr)
    return _pct_dict(probs_arr, spot_f * np.exp(qs))


def naive_normal_percentiles(spot, trailing_daily_returns, horizon_days,
                             probs=DEFAULT_PERCENTILES):
    """Gaussian fit to trailing daily log-returns, iid-scaled to horizon."""
    # SF-BL-11244: type-validate probs AT ENTRY, before the
    # SF-BL-10474 value screen below can coerce them. Pre-fix
    # np.asarray(probs, dtype=float) silently coerced bool and
    # numeric-string elements: probs=[True] was read as 1.0 (only
    # caught collaterally, mislabelled as a value, by the (0, 1)
    # screen; in the task's live repro, before that screen, it
    # returned {1.0: inf} via norm.ppf(1.0)) and probs=['0.5'] /
    # [0.5, '0.9'] returned percentile dicts as if lawful. A bool
    # is not a number (fleet convention) and a string element is
    # a corrupt type, not a probability: any bool / np.bool_ /
    # str / bytes / np.str_ / None element -- or probs itself in
    # one of those forms, a bool- or string-dtype array, or an
    # object array holding such an element -- now raises a
    # ValueError naming probs. Distinct from SF-BL-10267, which
    # fixtures horizon_days' VALUE domain in a different
    # signature context, and SF-BL-10393, which screens spot;
    # this screen is type-only and dual-tagged with SF-BL-10474
    # so that screen's cases keep naming it. Genuinely numeric
    # elements (int / float / numpy numeric / Decimal / Fraction
    # / 0-d numeric arrays) fall through untouched to the
    # identical conversion below, so numeric probs/horizon
    # percentile dicts stay bit-identical (kill criterion).
    def _bad_prob_element(_el):
        if isinstance(_el, np.ndarray):
            return _el.dtype == np.dtype(bool) \
                or _el.dtype.kind in "US" \
                or (_el.ndim == 0 and _bad_prob_element(_el.item()))
        return isinstance(_el, (bool, np.bool_, str, bytes)) \
            or _el is None

    _probs_type_bad = False
    if _bad_prob_element(probs):
        _probs_type_bad = True
    elif isinstance(probs, np.ndarray):
        if probs.dtype == np.dtype(bool) or probs.dtype.kind in "US":
            _probs_type_bad = True
        elif probs.dtype.kind == "O":
            _probs_type_bad = any(
                _bad_prob_element(_el) for _el in probs.flat)
        elif probs.dtype.kind not in "iufc":
            _probs_type_bad = True
    elif isinstance(probs, (list, tuple, set, frozenset, range)):
        _probs_type_bad = any(_bad_prob_element(_el) for _el in probs)
    if _probs_type_bad:
        raise ValueError(
            "naive_normal_percentiles requires probs elements to "
            "be real numbers (a bool or string is not a "
            f"probability); got probs={probs!r} "
            "(SF-BL-11244; SF-BL-10474)"
        )
    # SF-BL-10474: validate probs AT ENTRY, before any use (and
    # before the spot/horizon screens below, so a corrupt prob is
    # named even when another argument is also bad). probs flows
    # unchecked into stats.norm.ppf, whose ppf(0) = -inf and
    # ppf(1) = +inf, so spot*exp(mu_h + sd_h*z) mapped an
    # endpoint-inclusive caller grid to {0.0: 0.0, 0.5: ~spot,
    # 1.0: inf} -- a degenerate/infinite percentile dict that only
    # failed later, mislabelled, at CDF construction; negative,
    # > 1 and non-finite probs likewise returned NaN entries as
    # if lawful, and a non-numeric prob died with a bare
    # TypeError naming nothing. Distinct from SF-BL-10267
    # (horizon_days) and SF-BL-10393 (spot), which screen other
    # arguments of this same function. Every prob must be finite
    # and lie in the OPEN interval (0, 1); lawful probs fall
    # through as the identical float array the unscreened
    # np.asarray(probs) produced, so interior-prob outputs stay
    # bit-identical (kill criterion).
    try:
        probs_arr = np.asarray(probs, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            "naive_normal_percentiles requires probs to be finite "
            "and lie in the open interval (0, 1); got "
            f"probs={probs!r} (SF-BL-10474)"
        ) from None
    if probs_arr.ndim != 1 or probs_arr.size == 0:
        raise ValueError(
            "naive_normal_percentiles requires probs to be a "
            "non-empty 1-D sequence of probabilities in the open "
            f"interval (0, 1); got probs={probs!r} (SF-BL-10474)"
        )
    if not np.all(np.isfinite(probs_arr)):
        raise ValueError(
            "naive_normal_percentiles requires finite probs; got "
            "non-finite probs value(s) "
            f"{probs_arr[~np.isfinite(probs_arr)]} in "
            f"probs={probs!r} (SF-BL-10474)"
        )
    if np.any((probs_arr <= 0.0) | (probs_arr >= 1.0)):
        raise ValueError(
            "naive_normal_percentiles requires probs to lie in "
            "the open interval (0, 1); got probs value(s) "
            f"{probs_arr[(probs_arr <= 0.0) | (probs_arr >= 1.0)]} "
            f"outside (0, 1) in probs={probs!r} (SF-BL-10474)"
        )
    # SF-BL-11311: type-validate spot BEFORE the SF-BL-10393 value
    # screen below can coerce it. Pre-fix float(spot) read True as
    # 1.0, so spot=True silently returned the spot=1.0 percentile
    # dict (task live repro, horizon 7: centred on ~1.036 vs
    # ~103.562 at the spot=100 control), float('100') silently
    # coerced a string spot to 100.0, and on the unscreened wave3
    # original spot='100' died with a bare UFuncTypeError naming
    # nothing. A bool is not a number (fleet convention) and a
    # string/bytes/None spot is a corrupt type, not a price: every
    # bool / np.bool_ / str / bytes / None spot, and any other
    # non-numeric spot (containers, arbitrary objects, non-scalar
    # or bool/string-dtype arrays), now raises a ValueError naming
    # spot, dual-tagged with SF-BL-10393 so that screen's cases
    # (None, 'abc', []) keep naming it. Distinct from SF-BL-10393,
    # which fixtures spot's VALUE domain (finite, positive), and
    # SF-BL-11244, which fixtures this function's probs and
    # horizon_days types; this screen is type-only. A genuinely
    # numeric spot (int / float / numpy numeric scalar / Decimal /
    # Fraction / 0-d numeric array) falls through untouched to the
    # identical float conversion below, so numeric-spot percentile
    # dicts stay bit-identical (kill criterion).
    if isinstance(spot, (bool, np.bool_)):
        raise ValueError(
            "naive_normal_percentiles requires spot to be a real "
            "number (a bool is not a number); got boolean "
            f"spot={spot!r} (SF-BL-11311; SF-BL-10393)"
        )
    if spot is None or isinstance(spot, (str, bytes)):
        raise ValueError(
            "naive_normal_percentiles requires spot to be a real "
            "number; got non-numeric "
            f"spot={spot!r} of type {type(spot).__name__} "
            "(SF-BL-11311; SF-BL-10393)"
        )
    if isinstance(spot, np.ndarray):
        if spot.ndim != 0 \
                or spot.dtype == np.dtype(bool) \
                or spot.dtype.kind not in "iufc":
            raise ValueError(
                "naive_normal_percentiles requires spot to be a "
                "real number (a scalar); got "
                f"spot={spot!r} of type ndarray with dtype "
                f"{spot.dtype} (SF-BL-11311; SF-BL-10393)"
            )
    elif not isinstance(spot, numbers.Number):
        raise ValueError(
            "naive_normal_percentiles requires spot to be a real "
            "number; got non-numeric "
            f"spot={spot!r} of type {type(spot).__name__} "
            "(SF-BL-11311; SF-BL-10393)"
        )
    # SF-BL-10393: validate spot BEFORE any fitting (and before the
    # horizon/returns screens below, so a corrupt spot is named even
    # when another argument is also bad). The returns window is
    # screened (finite filter, >=30 obs) and horizon_days is screened
    # (SF-BL-10267), but pre-fix spot was never screened, so
    # spot=NaN returned an all-NaN percentile dict, spot=+/-inf an
    # all-inf dict, spot=0 a degenerate all-zero dict, and a negative
    # spot (e.g. -100) returned percentiles in REVERSED order -- all
    # consumed downstream as if lawful (a non-numeric spot died with
    # a bare TypeError/UFuncTypeError naming nothing). Distinct from
    # SF-BL-10266, which registered the same screen in
    # climatology_percentiles, a different function. A price/level
    # spot must be finite and positive; every corrupt spot now
    # raises a ValueError naming spot. Finite positive spots fall
    # through to the identical computation (spot_f is the identical
    # float), so lawful outputs stay bit-identical.
    try:
        spot_f = float(spot)
    except (TypeError, ValueError):
        raise ValueError(
            "naive_normal_percentiles requires a finite, positive "
            f"spot; got spot={spot!r} (SF-BL-10393)"
        ) from None
    if not np.isfinite(spot_f) or spot_f <= 0:
        raise ValueError(
            "naive_normal_percentiles requires a finite, positive "
            f"spot; got spot={spot!r} (SF-BL-10393)"
        )
    # SF-BL-11244: type-validate horizon_days BEFORE the
    # SF-BL-10267 value screen below can coerce it. Pre-fix
    # float(horizon_days) read True as 1.0, so horizon_days=True
    # silently returned horizon-1 percentiles (task live repro:
    # median 99.879 at spot 100 vs 99.397 for horizon_days=5),
    # and horizon_days='5' was either coerced to 5.0 or, before
    # that conversion existed, died with a bare 'can't multiply
    # sequence by non-int' TypeError naming nothing. A bool is
    # not a number and a string/bytes/None horizon is a corrupt
    # type, not a duration: every bool / np.bool_ / str / bytes /
    # None horizon, and any other non-numeric horizon
    # (containers, arbitrary objects, non-scalar or
    # bool/string-dtype arrays), now raises a ValueError naming
    # horizon_days, dual-tagged with SF-BL-10267 so that
    # screen's cases keep naming it. Distinct from SF-BL-10267,
    # which fixtures horizon's VALUE domain (finite, > 0);
    # this screen is type-only. A genuinely numeric horizon
    # (int / float / numpy numeric scalar / Decimal / Fraction /
    # 0-d numeric array) falls through untouched to the
    # identical float conversion below, so numeric-horizon
    # percentile dicts stay bit-identical (kill criterion).
    if isinstance(horizon_days, (bool, np.bool_)):
        raise ValueError(
            "naive_normal_percentiles requires horizon_days to "
            "be a real number (a bool is not a number); got "
            f"boolean horizon_days={horizon_days!r} "
            "(SF-BL-11244; SF-BL-10267)"
        )
    if horizon_days is None or isinstance(horizon_days, (str, bytes)):
        raise ValueError(
            "naive_normal_percentiles requires horizon_days to "
            "be a real number; got non-numeric "
            f"horizon_days={horizon_days!r} of type "
            f"{type(horizon_days).__name__} "
            "(SF-BL-11244; SF-BL-10267)"
        )
    if isinstance(horizon_days, np.ndarray):
        if horizon_days.ndim != 0 \
                or horizon_days.dtype == np.dtype(bool) \
                or horizon_days.dtype.kind not in "iufc":
            raise ValueError(
                "naive_normal_percentiles requires horizon_days "
                "to be a real number (a scalar); got "
                f"horizon_days={horizon_days!r} of type ndarray "
                f"with dtype {horizon_days.dtype} "
                "(SF-BL-11244; SF-BL-10267)"
            )
    elif not isinstance(horizon_days, numbers.Number):
        raise ValueError(
            "naive_normal_percentiles requires horizon_days to "
            "be a real number; got non-numeric "
            f"horizon_days={horizon_days!r} of type "
            f"{type(horizon_days).__name__} "
            "(SF-BL-11244; SF-BL-10267)"
        )
    # SF-BL-10267: validate horizon_days BEFORE any arithmetic.
    # horizon_days flows unchecked into mean*horizon and
    # std*sqrt(horizon), so a negative horizon NaNs via sqrt, NaN
    # poisons all five percentiles, and inf yields 0.0/NaN mixtures
    # — corrupt elicitations indistinguishable from real ones
    # downstream. Only a finite, strictly positive horizon is lawful.
    try:
        _h = float(horizon_days)
    except (TypeError, ValueError):
        raise ValueError(
            "naive_normal_percentiles requires horizon_days to be "
            f"finite and > 0; got horizon_days={horizon_days!r} "
            "(SF-BL-10267)"
        ) from None
    if not np.isfinite(_h) or _h <= 0:
        raise ValueError(
            "naive_normal_percentiles requires horizon_days to be "
            f"finite and > 0; got horizon_days={horizon_days!r} "
            "(SF-BL-10267)"
        )
    # SF-BL-11313: type-validate trailing_daily_returns BEFORE the
    # np.asarray(..., dtype=float) coercion below can parse it.
    # Pre-fix that coercion silently parsed string returns -- the
    # task's string panel ([str(x) for x in the numeric control],
    # spot=100, horizon 7) returned a percentile dict bit-identical
    # to the numeric control -- and read bools as 1/0 daily returns
    # that explode the Gaussian scale: the task's bool panel
    # ([True, False] * 20, mean 0.5 read as returns) returned a
    # median of 3311.55 (100 * exp(0.5 * 7)) vs 103.56 for the
    # numeric control (mean 0.005). A bool is not a number (fleet
    # convention) and a string return is a caller type bug, not a
    # log-return: any bool / np.bool_ / str / bytes / np.str_
    # element -- in a list / tuple / other sequence at any nesting
    # depth, a bool- or string-dtype array, an object array
    # holding such an element, a 0-d bool/string array element, or
    # trailing_daily_returns itself in one of those forms -- now
    # raises a ValueError naming trailing_daily_returns, as does
    # any other non-numeric element (None, complex, containers,
    # arbitrary objects) or a non-sequence panel (a bare scalar,
    # dict or set). Distinct from SF-BL-11244 (this function's
    # probs / horizon_days types) and SF-BL-10393 (its spot value
    # domain) -- different arguments in each case; no prior item
    # fixtured this argument. Reject-only: genuinely numeric
    # returns (int / float / numpy numeric, incl. Decimal /
    # Fraction elements, nested / 2-D and object-dtype numeric
    # panels, and NaN/inf floats, whose finiteness filtering below
    # is unchanged) fall through untouched to the identical
    # conversion, so numeric-return percentile dicts stay
    # bit-identical (kill criterion). One-shot iterables are
    # materialised once so the scan does not consume them.
    def _bad_return_element_11313(_el):
        if isinstance(_el, (bool, np.bool_)):
            return True
        if isinstance(_el, (str, bytes)):
            return True
        if _el is None:
            return True
        if isinstance(_el, (complex, np.complexfloating)):
            return True
        if isinstance(_el, np.ndarray):
            if _el.dtype == np.dtype(bool) \
                    or _el.dtype.kind in "US":
                return True
            if _el.dtype.kind == "O":
                return any(
                    _bad_return_element_11313(_x)
                    for _x in _el.flat)
            return _el.dtype.kind not in "iuf"
        if isinstance(_el, (list, tuple)):
            return any(
                _bad_return_element_11313(_x) for _x in _el)
        return not isinstance(
            _el, (numbers.Number, np.integer, np.floating))

    _tdr_bad_11313 = False
    _tdr_detail_11313 = None
    _tdr_mat_11313 = None
    if isinstance(trailing_daily_returns, np.ndarray):
        if trailing_daily_returns.dtype == np.dtype(bool) \
                or trailing_daily_returns.dtype.kind in "US" \
                or (trailing_daily_returns.dtype.kind not in "iuf"
                    and trailing_daily_returns.dtype.kind != "O"):
            _tdr_bad_11313 = True
            _tdr_detail_11313 = trailing_daily_returns
        elif trailing_daily_returns.dtype.kind == "O":
            for _el_11313 in trailing_daily_returns.flat:
                if _bad_return_element_11313(_el_11313):
                    _tdr_bad_11313 = True
                    _tdr_detail_11313 = _el_11313
                    break
    elif isinstance(trailing_daily_returns, (list, tuple, range)):
        for _el_11313 in trailing_daily_returns:
            if _bad_return_element_11313(_el_11313):
                _tdr_bad_11313 = True
                _tdr_detail_11313 = _el_11313
                break
    elif isinstance(trailing_daily_returns, (bool, np.bool_)) \
            or trailing_daily_returns is None \
            or isinstance(trailing_daily_returns, (str, bytes)) \
            or isinstance(
                trailing_daily_returns, (dict, set, frozenset)) \
            or isinstance(trailing_daily_returns, numbers.Number) \
            or not hasattr(trailing_daily_returns, "__iter__"):
        _tdr_bad_11313 = True
        _tdr_detail_11313 = trailing_daily_returns
    else:
        _tdr_mat_11313 = list(trailing_daily_returns)
        for _el_11313 in _tdr_mat_11313:
            if _bad_return_element_11313(_el_11313):
                _tdr_bad_11313 = True
                _tdr_detail_11313 = _el_11313
                break
    if _tdr_bad_11313:
        raise ValueError(
            "naive_normal_percentiles requires "
            "trailing_daily_returns to be a sequence of real "
            "numbers (a bool or string is not a number); got "
            "non-numeric trailing_daily_returns value "
            f"{_tdr_detail_11313!r} in "
            f"trailing_daily_returns={trailing_daily_returns!r} "
            "(SF-BL-11313)"
        )
    if _tdr_mat_11313 is not None:
        trailing_daily_returns = _tdr_mat_11313
    d = np.asarray(trailing_daily_returns, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 30:
        raise ValueError("naive fit needs >= 30 trailing daily returns")
    mu_h = d.mean() * _h
    sd_h = d.std(ddof=1) * np.sqrt(_h)
    z = stats.norm.ppf(probs_arr)
    return _pct_dict(probs_arr, spot_f * np.exp(mu_h + sd_h * z))


def _screen_recent_daily_returns_types_11350(recent_daily_returns, caller):
    """SF-BL-11350: type-validate a recent_daily_returns argument
    BEFORE any float coercion, returning it as a float ndarray.

    A bool is not a number (fleet convention, cf. SF-BL-11312 /
    SF-BL-11313 in this module) and a string is not a daily return,
    but both survive naive float coercion: np.asarray(...,
    dtype=float) parses numeric strings and reads True/False as
    1.0/0.0, so pre-fix in disciplined_percentiles a 30-element
    string panel returned the numeric control dict bit-identically
    and a [True]*30 panel (all-ones coerced returns, sd_recent = 0)
    silently rescaled the dict via the volatility-regime factor
    (task live repro, spot=100: 0.95 percentile 102.079 vs 102.880
    control). The screen therefore inspects the RAW returns -- the
    container itself, an ndarray's dtype, and each element of any
    other sequence, recursively for nested panels -- and raises a
    ValueError naming recent_daily_returns on any bool or
    string/bytes element (or any other non-numeric element, or a
    non-sequence panel, which pre-fix either coerced silently or
    died with a bare error naming nothing). Genuinely numeric
    returns (int / float / numpy numeric scalars, Decimal /
    Fraction, lists, tuples, ranges and numeric ndarrays, incl.
    nested / 2-D and object-dtype numeric panels and non-finite
    values, which the caller's isfinite filter still drops) are
    returned as the identical float array
    np.asarray(recent_daily_returns, dtype=float) would produce,
    so numeric-return outputs stay bit-identical. One-shot
    iterables are materialised once so the scan does not consume
    them.
    """
    def _raise(detail):
        raise ValueError(
            f"{caller} requires recent_daily_returns to be a "
            f"sequence of real numbers (a bool or string is not "
            f"a number); got {detail} in "
            f"recent_daily_returns={recent_daily_returns!r} "
            "(SF-BL-11350)"
        )

    def _bad(_el):
        if isinstance(_el, (bool, np.bool_)):
            return True
        if isinstance(_el, (str, bytes, np.str_, np.bytes_)):
            return True
        if _el is None:
            return True
        if isinstance(_el, (complex, np.complexfloating)):
            return True
        if isinstance(_el, np.ndarray):
            if _el.dtype == np.dtype(bool) \
                    or _el.dtype.kind in "US":
                return True
            if _el.dtype.kind == "O":
                return any(_bad(_x) for _x in _el.flat)
            return _el.dtype.kind not in "iuf"
        if isinstance(_el, (list, tuple, range)):
            return any(_bad(_x) for _x in _el)
        if isinstance(_el, (dict, set, frozenset)):
            return True
        return not isinstance(
            _el, (numbers.Number, np.integer, np.floating))

    if isinstance(recent_daily_returns, np.ndarray):
        if recent_daily_returns.dtype == np.dtype(bool) \
                or recent_daily_returns.dtype.kind in "US" \
                or (recent_daily_returns.dtype.kind not in "iuf"
                    and recent_daily_returns.dtype.kind != "O"):
            _raise(f"recent_daily_returns array with dtype "
                   f"{recent_daily_returns.dtype}")
        if recent_daily_returns.dtype.kind in "iuf":
            return np.asarray(recent_daily_returns, dtype=float)
        _mat = None
        _elements = list(recent_daily_returns.flat)
    elif isinstance(recent_daily_returns, (list, tuple, range)):
        _mat = None
        _elements = list(recent_daily_returns)
    elif isinstance(recent_daily_returns, (bool, np.bool_)) \
            or recent_daily_returns is None \
            or isinstance(recent_daily_returns, (str, bytes)) \
            or isinstance(recent_daily_returns,
                          (dict, set, frozenset)) \
            or isinstance(recent_daily_returns, numbers.Number) \
            or not hasattr(recent_daily_returns, "__iter__"):
        _raise(f"non-sequence recent_daily_returns of type "
               f"{type(recent_daily_returns).__name__}")
    else:
        _mat = list(recent_daily_returns)
        _elements = _mat
    for _i, _v in enumerate(_elements):
        if _bad(_v):
            _raise(f"non-numeric recent_daily_returns element at "
                   f"index {_i}: {_v!r}")
    try:
        return np.asarray(
            _mat if _mat is not None else recent_daily_returns,
            dtype=float)
    except (TypeError, ValueError):
        _raise(f"recent_daily_returns={recent_daily_returns!r} "
               f"not convertible to float")


def _screen_trailing_h_returns_types_11384(trailing_h_returns, caller):
    """SF-BL-11384: type-validate a trailing_h_returns argument
    BEFORE any float coercion, returning it as a float ndarray.

    A bool is not a number (fleet convention, cf. SF-BL-11312 /
    SF-BL-11349 / SF-BL-11350 in this module) and a string is not
    an h-day return, but both survive naive float coercion:
    np.asarray(..., dtype=float) parses numeric strings and reads
    True/False as 1.0/0.0, so pre-fix in disciplined_percentiles
    a 60-element string panel returned the numeric control dict
    bit-identically (task live repro, spot=100: 0.95 percentile
    102.888) and a bool panel silently shifted the dict (task
    live repro: 0.95 percentile 164.872). The screen therefore
    inspects the RAW returns -- the container itself, an
    ndarray's dtype, and each element of any other sequence,
    recursively for nested panels -- and raises a ValueError
    naming trailing_h_returns on any bool or string/bytes
    element (or any other non-numeric element, or a non-sequence
    panel, which pre-fix either coerced silently or died with a
    bare error naming nothing). Genuinely numeric returns (int /
    float / numpy numeric scalars, Decimal / Fraction, lists,
    tuples, ranges and numeric ndarrays, incl. nested / 2-D and
    object-dtype numeric panels and non-finite values, which
    the callers' isfinite filters still drop) are returned as
    the identical float array
    np.asarray(trailing_h_returns, dtype=float) would produce,
    so numeric-return outputs stay bit-identical. One-shot
    iterables are materialised once so the scan does not
    consume them.
    """
    def _raise(detail):
        raise ValueError(
            f"{caller} requires trailing_h_returns to be a "
            f"sequence of real numbers (a bool or string is not "
            f"a number); got {detail} in "
            f"trailing_h_returns={trailing_h_returns!r} "
            "(SF-BL-11384)"
        )

    def _bad(_el):
        if isinstance(_el, (bool, np.bool_)):
            return True
        if isinstance(_el, (str, bytes, np.str_, np.bytes_)):
            return True
        if _el is None:
            return True
        if isinstance(_el, (complex, np.complexfloating)):
            return True
        if isinstance(_el, np.ndarray):
            if _el.dtype == np.dtype(bool) \
                    or _el.dtype.kind in "US":
                return True
            if _el.dtype.kind == "O":
                return any(_bad(_x) for _x in _el.flat)
            return _el.dtype.kind not in "iuf"
        if isinstance(_el, (list, tuple, range)):
            return any(_bad(_x) for _x in _el)
        if isinstance(_el, (dict, set, frozenset)):
            return True
        return not isinstance(
            _el, (numbers.Number, np.integer, np.floating))

    if isinstance(trailing_h_returns, np.ndarray):
        if trailing_h_returns.dtype == np.dtype(bool) \
                or trailing_h_returns.dtype.kind in "US" \
                or (trailing_h_returns.dtype.kind not in "iuf"
                    and trailing_h_returns.dtype.kind != "O"):
            _raise(f"trailing_h_returns array with dtype "
                   f"{trailing_h_returns.dtype}")
        if trailing_h_returns.dtype.kind in "iuf":
            return np.asarray(trailing_h_returns, dtype=float)
        _mat = None
        _elements = list(trailing_h_returns.flat)
    elif isinstance(trailing_h_returns, (list, tuple, range)):
        _mat = None
        _elements = list(trailing_h_returns)
    elif isinstance(trailing_h_returns, (bool, np.bool_)) \
            or trailing_h_returns is None \
            or isinstance(trailing_h_returns, (str, bytes)) \
            or isinstance(trailing_h_returns,
                          (dict, set, frozenset)) \
            or isinstance(trailing_h_returns, numbers.Number) \
            or not hasattr(trailing_h_returns, "__iter__"):
        _raise(f"non-sequence trailing_h_returns of type "
               f"{type(trailing_h_returns).__name__}")
    else:
        _mat = list(trailing_h_returns)
        _elements = _mat
    for _i, _v in enumerate(_elements):
        if _bad(_v):
            _raise(f"non-numeric trailing_h_returns element at "
                   f"index {_i}: {_v!r}")
    try:
        return np.asarray(
            _mat if _mat is not None else trailing_h_returns,
            dtype=float)
    except (TypeError, ValueError):
        _raise(f"trailing_h_returns={trailing_h_returns!r} "
               f"not convertible to float")


def disciplined_percentiles(spot, trailing_h_returns, trailing_daily_returns,
                            recent_daily_returns, probs=DEFAULT_PERCENTILES):
    """Climatology-anchored, regime-scaled, tail-fattened elicitation.

    See module docstring for the four adjustments. All statistics come from
    the trailing windows passed in (causal by construction at the caller).
    """
    # SF-BL-11310: type-validate spot AT ENTRY, before any coercion
    # (and before the probs screen below, so a corrupt spot type is
    # named even when probs is also bad; a NaN/inf spot is a numeric
    # TYPE, passes this screen, and is still named by the probs
    # screen / the climatology value screen as before). Pre-fix
    # spot was never type-validated in this function: float(spot)
    # inside climatology_percentiles read True as 1.0, so spot=True
    # returned a percentile dict centred on ~1.0 (live repro:
    # median 1.0008 vs 100.083 at the spot=100 control), and
    # spot='100' died in this function's own `... / spot` division
    # with a bare UFuncTypeError naming nothing. A bool is not a
    # number (fleet convention) and a string spot is a caller type
    # bug, not a price: every bool / np.bool_ / str / bytes /
    # np.str_ / None spot, and any other non-numeric spot
    # (containers, arbitrary objects, non-scalar or
    # bool/string-dtype arrays), now raises a ValueError naming
    # spot, at entry. Distinct from SF-BL-10475 (this function's
    # spot VALUE domain: 0 / negative / NaN / inf, still governed
    # via climatology), SF-BL-11268 (this function's probs type)
    # and SF-BL-10393 (naive_normal_percentiles' spot, a different
    # function). Reject-only: a genuinely numeric spot (int /
    # float / numpy numeric scalar / Decimal / Fraction / 0-d
    # numeric array) falls through untouched -- the original spot
    # object is still used below -- so numeric-spot percentile
    # dicts stay bit-identical (kill criterion). Identical screen
    # in the scaffold copy (superforecaster_out/), which carries
    # the full comment.
    if isinstance(spot, (bool, np.bool_)):
        raise ValueError(
            "disciplined_percentiles requires spot to be a real "
            "number (a bool is not a number); got boolean "
            f"spot={spot!r} (SF-BL-11310)"
        )
    if spot is None or isinstance(spot, (str, bytes)):
        raise ValueError(
            "disciplined_percentiles requires spot to be a real "
            f"number; got non-numeric spot={spot!r} of type "
            f"{type(spot).__name__} (SF-BL-11310)"
        )
    if isinstance(spot, np.ndarray):
        _spot_item_bad = False
        if spot.ndim == 0 and spot.dtype.kind == "O":
            _spot_item = spot.item()
            _spot_item_bad = isinstance(
                _spot_item, (bool, np.bool_, str, bytes)) \
                or _spot_item is None \
                or not isinstance(_spot_item, numbers.Number)
        if spot.ndim != 0 \
                or spot.dtype == np.dtype(bool) \
                or spot.dtype.kind in "US" \
                or _spot_item_bad \
                or (spot.dtype.kind != "O"
                    and spot.dtype.kind not in "iufc"):
            raise ValueError(
                "disciplined_percentiles requires spot to be a "
                "real number (a scalar); got "
                f"spot={spot!r} of type ndarray with dtype "
                f"{spot.dtype} (SF-BL-11310)"
            )
    elif not isinstance(spot, numbers.Number):
        raise ValueError(
            "disciplined_percentiles requires spot to be a real "
            f"number; got non-numeric spot={spot!r} of type "
            f"{type(spot).__name__} (SF-BL-11310)"
        )
    # SF-BL-11350: type-validate recent_daily_returns after the
    # spot screen above and BEFORE the probs screen / any coercion
    # (so a corrupt recent panel is named even when probs is also
    # bad). Pre-fix recent_daily_returns was never type-validated
    # in this function: it fed sd_recent in the volatility-regime
    # factor (step 3 below) via np.asarray(dtype=float) with no
    # dtype screen, so a 30-element string panel returned the
    # control dict bit-identically (task live repro, spot=100:
    # median 99.871, 0.95: 102.880) and a [True]*30 panel -- coerced
    # to all-ones returns with sd_recent = 0, hence r_adj clipped
    # to 0.75 -- silently rescaled the dict (0.95: 102.079 vs
    # 102.880 control). A bool is not a number (fleet convention)
    # and a string return is a caller type bug, not a daily
    # return: every bool or string-dtype recent_daily_returns
    # element now raises a ValueError naming
    # recent_daily_returns. Distinct from SF-BL-11312 /
    # SF-BL-11313 (trailing panels in climatology_percentiles /
    # naive_normal_percentiles) and SF-BL-11310 / SF-BL-11268
    # (this function's spot / probs types) -- this argument is
    # consumed directly in this function's body, not delegated to
    # climatology_percentiles. Reject-only: genuinely numeric
    # recent panels fall through as the identical float array, so
    # numeric-panel percentile dicts stay bit-identical (kill
    # criterion).
    _recent_arr_11350 = _screen_recent_daily_returns_types_11350(
        recent_daily_returns, "disciplined_percentiles")
    # SF-BL-11243: screen probs types via the same helper BEFORE
    # float coercion -- pre-fix this line's np.asarray(dtype=float)
    # coerced bool/string probs to numbers before
    # climatology_percentiles could ever see their original types.
    # Numeric probs return the identical float array, so lawful
    # disciplined outputs stay bit-identical.
    # SF-BL-11268: this screen also closes that task's disciplined
    # repro (probs=['0.05', '0.5', '0.95'] bit-identical to the
    # numeric control; probs=[True, 0.5, 0.9] keyed 1.0) -- the
    # scaffold copy carries the dedicated SF-BL-11268 screen; both
    # copies verified in SF-BL-11268's verify_fix.py.
    probs = _screen_probs_types_11243(probs, "disciplined_percentiles")
    # SF-BL-10316: validate probs ORDER (the VALUE domain of this
    # function's probs), after the type screen above and before any
    # use. Pre-fix the function indexed log-returns by caller probs
    # order, picked med_idx as nearest-to-0.5, then let
    # np.maximum.accumulate plus the strict-increase nudge repair
    # monotonicity in VALUE space -- so an unsorted probs argument
    # silently overwrote out-of-order quantiles: the live repro
    # probs=(0.95, 0.05) returned {0.95: 101.706, 0.05: 101.706},
    # the elicited 5th percentile destroyed onto the 95th with no
    # error, and duplicate probs collapsed to a single dict key.
    # Unlike _validate_percentiles, which enforces strictly
    # increasing probabilities for declared percentiles, nothing
    # here screened order at all. Every non-increasing probs --
    # unsorted, reversed, partially out of order, or holding a
    # duplicate -- now raises a ValueError naming probs, at this
    # point, before climatology_percentiles is called. Reject-only:
    # strictly increasing probs (incl. the DEFAULT_PERCENTILES
    # default and any single-element probs, vacuously increasing)
    # fall through to the identical computation below, so
    # sorted-probs outputs stay bit-identical (done criterion).
    # Identical screen in the scaffold copy
    # (superforecaster_out/), which carries the same comment.
    if probs.size >= 2 and np.any(np.diff(probs) <= 0):
        raise ValueError(
            "disciplined_percentiles requires probs to be "
            "strictly increasing; got non-increasing "
            f"probs={probs!r} (SF-BL-10316)"
        )
    # SF-BL-11384: type-validate trailing_h_returns BEFORE the
    # climatology call below (its first consumption) and before
    # this function's own step-(4) reuse for the trailing
    # excess-kurtosis tail fattening. Pre-fix disciplined_
    # percentiles never type-validated this argument in either
    # place: it flowed into climatology_percentiles and into
    # step (4) via np.asarray(..., dtype=float) with no screen,
    # so a 60-element string panel returned a percentile dict
    # bit-identical to the numeric control (task live repro,
    # spot=100: 0.95 percentile 102.888) and a bool panel read
    # as 1.0/0.0 h-day returns silently shifted the dict (task
    # live repro: 0.95 percentile 164.872). A bool is not a
    # number (fleet convention) and a string return is a caller
    # type bug, not a log-return: every bool or string-dtype
    # trailing_h_returns element now raises a ValueError
    # naming trailing_h_returns, from THIS function (the
    # scaffold copy's climatology_percentiles has no screen at
    # all; the repo copy's SF-BL-11312 screen belongs to
    # climatology_percentiles and never guarded this
    # function's own step-(4) consumption). Distinct from
    # SF-BL-11312 (climatology_percentiles' trailing_h_returns,
    # a different function), SF-BL-11349 / SF-BL-11350 (this
    # function's trailing_daily_returns / recent_daily_
    # returns) and SF-BL-11310 / SF-BL-11268 (this function's
    # spot / probs types). The screen runs at this point --
    # after the SF-BL-11310 spot, SF-BL-11350 recent and
    # probs screens, so those keep naming their own arguments
    # first -- and the screened float array is what both
    # consumers below receive. Reject-only: genuinely numeric
    # panels fall through as the identical float array, so
    # numeric-panel percentile dicts stay bit-identical (kill
    # criterion).
    _trailing_arr_11384 = _screen_trailing_h_returns_types_11384(
        trailing_h_returns, "disciplined_percentiles")
    base = climatology_percentiles(spot, _trailing_arr_11384, probs)
    logv = np.log(np.array([base[float(p)] for p in probs]) / spot)

    # (2) shrink median toward no-change (log-return 0) by 50%.
    med_idx = int(np.argmin(np.abs(probs - 0.5)))
    median_shrunk = 0.5 * logv[med_idx]
    dev = logv - logv[med_idx]
    logv = median_shrunk + dev + 0.0  # re-centre deviations on shrunk median
    # NOTE: re-centring shifts all quantiles by the median shrinkage, which
    # is intended: drift shrinkage is a location adjustment, not just median.

    # SF-BL-11349: type-validate trailing_daily_returns BEFORE the
    # np.asarray(..., dtype=float) coercion below can parse it.
    # Pre-fix that coercion silently parsed string returns -- the
    # task's 60-element string panel returned a percentile dict
    # bit-identical to the numeric control (median 99.871, 0.95:
    # 102.880 on the finder's panel) -- and read bools as 1/0 daily
    # returns that inflated sd_long and silently rescaled every
    # deviation from the median via the regime factor (the task's
    # bool panel shifted the 0.95 percentile to 102.079 vs the
    # 102.880 control). A bool is not a number (fleet convention)
    # and a string return is a caller type bug, not a log-return:
    # any bool / np.bool_ / str / bytes / np.str_ element -- in a
    # list / tuple / other sequence at any nesting depth, a bool-
    # or string-dtype array, an object array holding such an
    # element, a 0-d bool/string array element, or
    # trailing_daily_returns itself in one of those forms -- now
    # raises a ValueError naming trailing_daily_returns, as does
    # any other non-numeric element (None, complex, containers,
    # arbitrary objects) or a non-sequence panel (a bare scalar,
    # dict or set). Distinct from SF-BL-11312 / SF-BL-11313
    # (trailing panels in climatology_percentiles /
    # naive_normal_percentiles, different functions) and
    # SF-BL-11268 / SF-BL-11310 (this function's probs / spot
    # types, different arguments); no prior item fixtured this
    # argument of this function. The screen runs at this point --
    # after the SF-BL-11310 spot and SF-BL-11268 probs screens, so
    # those keep naming their own arguments first -- and
    # immediately before the coercion it guards. Reject-only:
    # genuinely numeric returns (int / float / numpy numeric,
    # incl. Decimal / Fraction elements, nested / 2-D and
    # object-dtype numeric panels, and NaN/inf floats, whose
    # finiteness filtering below is unchanged) fall through
    # untouched to the identical conversion, so numeric-panel
    # percentile dicts stay bit-identical (kill criterion).
    # One-shot iterables are materialised once so the scan does
    # not consume them.
    def _bad_return_element_11349(_el):
        if isinstance(_el, (bool, np.bool_)):
            return True
        if isinstance(_el, (str, bytes)):
            return True
        if _el is None:
            return True
        if isinstance(_el, (complex, np.complexfloating)):
            return True
        if isinstance(_el, np.ndarray):
            if _el.dtype == np.dtype(bool) \
                    or _el.dtype.kind in "US":
                return True
            if _el.dtype.kind == "O":
                return any(
                    _bad_return_element_11349(_x)
                    for _x in _el.flat)
            return _el.dtype.kind not in "iuf"
        if isinstance(_el, (list, tuple)):
            return any(
                _bad_return_element_11349(_x) for _x in _el)
        return not isinstance(
            _el, (numbers.Number, np.integer, np.floating))

    _tdr_bad_11349 = False
    _tdr_detail_11349 = None
    _tdr_mat_11349 = None
    if isinstance(trailing_daily_returns, np.ndarray):
        if trailing_daily_returns.dtype == np.dtype(bool) \
                or trailing_daily_returns.dtype.kind in "US" \
                or (trailing_daily_returns.dtype.kind not in "iuf"
                    and trailing_daily_returns.dtype.kind != "O"):
            _tdr_bad_11349 = True
            _tdr_detail_11349 = trailing_daily_returns
        elif trailing_daily_returns.dtype.kind == "O":
            for _el_11349 in trailing_daily_returns.flat:
                if _bad_return_element_11349(_el_11349):
                    _tdr_bad_11349 = True
                    _tdr_detail_11349 = _el_11349
                    break
    elif isinstance(trailing_daily_returns, (list, tuple, range)):
        for _el_11349 in trailing_daily_returns:
            if _bad_return_element_11349(_el_11349):
                _tdr_bad_11349 = True
                _tdr_detail_11349 = _el_11349
                break
    elif isinstance(trailing_daily_returns, (bool, np.bool_)) \
            or trailing_daily_returns is None \
            or isinstance(trailing_daily_returns, (str, bytes)) \
            or isinstance(
                trailing_daily_returns, (dict, set, frozenset)) \
            or isinstance(trailing_daily_returns, numbers.Number) \
            or not hasattr(trailing_daily_returns, "__iter__"):
        _tdr_bad_11349 = True
        _tdr_detail_11349 = trailing_daily_returns
    else:
        _tdr_mat_11349 = list(trailing_daily_returns)
        for _el_11349 in _tdr_mat_11349:
            if _bad_return_element_11349(_el_11349):
                _tdr_bad_11349 = True
                _tdr_detail_11349 = _el_11349
                break
    if _tdr_bad_11349:
        raise ValueError(
            "disciplined_percentiles requires "
            "trailing_daily_returns to be a sequence of real "
            "numbers (a bool or string is not a number); got "
            "non-numeric trailing_daily_returns value "
            f"{_tdr_detail_11349!r} in "
            f"trailing_daily_returns={trailing_daily_returns!r} "
            "(SF-BL-11349)"
        )
    if _tdr_mat_11349 is not None:
        trailing_daily_returns = _tdr_mat_11349
    # (3) volatility-regime scaling of deviations from the (shrunk) median.
    d_long = np.asarray(trailing_daily_returns, dtype=float)
    d_recent = _recent_arr_11350  # SF-BL-11350: screened above
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
    r = _trailing_arr_11384  # SF-BL-11384: screened above
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
    # SF-BL-11309: type-validate the cdf and nominals arguments AT
    # ENTRY, before any float coercion. Pre-fix the first use of both
    # arguments was np.asarray(..., dtype=float), which coerces with
    # no dtype screen, so on the task panel (cdf [0, 0.5, 1] over
    # nominals [0, 50, 100], y=50, control CRPS 12.5) a string cdf
    # ['0', '0.5', '1'] and string nominals ['0', '50', '100'] each
    # returned 12.5 bit-identical to the numeric control, and a bool
    # cdf [False, True, True] was read as [0.0, 1.0, 1.0] and
    # returned a perfect CRPS 0.0. A bool is not a number (fleet
    # convention, cf. distribution_validity._is_number, the
    # validate_cdf bool screen SF-BL-10471 and the crps_from_cdf
    # outcome screen SF-BL-11134 in this module) and a string grid
    # value is a corrupt type, not a number: every bool or
    # string/bytes element of either array -- a Python or numpy
    # bool/str scalar in a list/tuple/other sequence (at any nesting
    # depth), a bool- or string-dtype ndarray, an object-dtype
    # element, or a bare bool/string cdf/nominals itself -- now
    # raises a ValueError naming the offending argument (cdf or
    # nominals), at entry, before the SF-BL-11134 y screen and every
    # value screen below, so the type error is named even when
    # another argument is also bad. Distinct from SF-BL-11134 (this
    # function's outcome y type), SF-BL-9924 (cdf/nominals VALUE
    # domain: finiteness, monotonicity, [0, 1] range) and
    # SF-BL-10473 (cdf/nominals finiteness) -- none of which fixtures
    # the cdf or nominals TYPE domain. Scope is deliberately narrow:
    # only bool and string/bytes elements are screened here; other
    # non-numeric elements fall through to the existing conversion /
    # finiteness screens. Reject-only: genuinely numeric grids (int /
    # float / numpy numeric lists, tuples and arrays, incl.
    # Decimal/Fraction elements) fall through untouched to the
    # identical np.asarray(..., dtype=float) below, so numeric-grid
    # CRPS values stay bit-identical (kill criterion).
    def _bad_grid_elements_11309(_arr):
        """(bad_indices, bad_values, materialised) for bool/string
        elements of _arr; materialised is the consumed item list when
        _arr was a one-shot iterable, else None."""
        if isinstance(_arr, (bool, np.bool_)):
            return [0], [_arr], None
        if isinstance(_arr, (str, bytes)):
            return [0], [_arr], None
        if isinstance(_arr, np.ndarray):
            if _arr.dtype == np.dtype(bool) \
                    or _arr.dtype.kind in "US":
                return (list(range(_arr.size)),
                        _arr.ravel().tolist(), None)
            if _arr.dtype.kind == "O":
                _idx, _vals = [], []
                for _i, _v in enumerate(_arr.ravel().tolist()):
                    if isinstance(_v, (bool, np.bool_, str, bytes)):
                        _idx.append(_i)
                        _vals.append(_v)
                    elif isinstance(_v, np.ndarray) and (
                            _v.dtype == np.dtype(bool)
                            or _v.dtype.kind in "US"):
                        _idx.append(_i)
                        _vals.append(_v)
                return _idx, _vals, None
            return [], [], None
        if hasattr(_arr, "__iter__"):
            _items = list(_arr)
            _idx, _vals = [], []
            _stack = [( _i, _v) for _i, _v in enumerate(_items)]
            while _stack:
                _i, _v = _stack.pop()
                if isinstance(_v, (bool, np.bool_, str, bytes)):
                    _idx.append(_i)
                    _vals.append(_v)
                elif isinstance(_v, (list, tuple)):
                    _stack.extend((_i, _w) for _w in _v)
                elif isinstance(_v, np.ndarray) and (
                        _v.dtype == np.dtype(bool)
                        or _v.dtype.kind in "US"):
                    _idx.append(_i)
                    _vals.append(_v)
                elif isinstance(_v, np.ndarray) \
                        and _v.dtype.kind == "O":
                    _stack.extend(
                        (_i, _w) for _w in _v.ravel().tolist())
            _pairs = sorted(zip(_idx, _vals), key=lambda _p: _p[0])
            return ([_p[0] for _p in _pairs],
                    [_p[1] for _p in _pairs],
                    _items if not isinstance(
                        _arr, (list, tuple, range)) else None)
        return [], [], None

    _cdf_bad_idx, _cdf_bad_vals, _cdf_mat = \
        _bad_grid_elements_11309(cdf)
    if _cdf_mat is not None:
        cdf = _cdf_mat
    _nom_bad_idx, _nom_bad_vals, _nom_mat = \
        _bad_grid_elements_11309(nominals)
    if _nom_mat is not None:
        nominals = _nom_mat
    if _cdf_bad_idx:
        raise ValueError(
            "crps_from_cdf requires cdf to be a sequence of real "
            "numbers (a bool is not a number and a string is not "
            "a number); got bool or string cdf value(s) "
            f"{_cdf_bad_vals!r} at cdf indices {_cdf_bad_idx} "
            "(SF-BL-11309; cdf/nominals type screening cf. "
            "SF-BL-11134 y type, SF-BL-9924 / SF-BL-10473 value "
            "and finiteness screens)"
        )
    if _nom_bad_idx:
        raise ValueError(
            "crps_from_cdf requires nominals to be a sequence of "
            "real numbers (a bool is not a number and a string is "
            "not a number); got bool or string nominals value(s) "
            f"{_nom_bad_vals!r} at nominals indices "
            f"{_nom_bad_idx} (SF-BL-11309; cdf/nominals type "
            "screening cf. SF-BL-11134 y type, SF-BL-9924 / "
            "SF-BL-10473 value and finiteness screens)"
        )
    # SF-BL-11134: type-validate the outcome y AT ENTRY, before any
    # float coercion or comparison. Pre-fix y was never type-checked:
    # the finiteness screens below call float(y), which a bool passes
    # as 1.0/0.0 and a numeric string passes as its parsed value, so
    # y=True silently returned CRPS 0.875 -- bit-identical to scoring
    # the lawful resolution y=1.0 on the task panel -- and y='2.0'
    # then died inside the (nominals >= y) integrand comparison with
    # a bare UFuncTypeError that never names y (y=None / non-numeric
    # strings were caught only collaterally, mislabelled as a
    # finiteness failure by SF-BL-10473's float() catch). A bool is
    # not a number (fleet convention, cf.
    # distribution_validity._is_number and validate_cdf's bool screen
    # in this module, SF-BL-10471), and a string/bytes/None outcome is
    # a corrupt resolution type, not a value: every bool, string,
    # bytes or None y now raises a ValueError naming y and its type,
    # as does any other non-numeric y (containers, arbitrary objects,
    # non-scalar or bool/string arrays). Distinct from SF-BL-9924 /
    # SF-BL-10473 (the VALUE domain of y: NaN/+/-inf), which still
    # govern numeric y below. Reject-only: a numeric y (int, float,
    # numpy integer/floating scalar, numbers.Number such as Fraction
    # or Decimal, or a 0-d numeric array) falls through untouched to
    # the identical screens and integrand, so numeric-y CRPS values
    # stay bit-identical (kill criterion).
    if isinstance(y, (bool, np.bool_)):
        raise ValueError(
            "crps_from_cdf requires a numeric y (outcome); got "
            f"boolean y={y!r} (a bool is not a number) (SF-BL-11134)"
        )
    if y is None or isinstance(y, (str, bytes)):
        raise ValueError(
            "crps_from_cdf requires a numeric y (outcome); got "
            f"non-numeric y={y!r} of type {type(y).__name__} "
            "(SF-BL-11134)"
        )
    if isinstance(y, np.ndarray):
        if y.ndim != 0 or y.dtype == np.dtype(bool) \
                or y.dtype.kind not in "iufc":
            raise ValueError(
                "crps_from_cdf requires a numeric y (outcome); got "
                f"non-numeric y={y!r} of type ndarray with dtype "
                f"{y.dtype} (SF-BL-11134)"
            )
    elif not isinstance(y, numbers.Number):
        raise ValueError(
            "crps_from_cdf requires a numeric y (outcome); got "
            f"non-numeric y={y!r} of type {type(y).__name__} "
            "(SF-BL-11134)"
        )
    cdf = np.asarray(cdf, dtype=float)
    nominals = np.asarray(nominals, dtype=float)
    # SF-BL-10269: reject empty/degenerate grids BEFORE integrating.
    # A trapezoid over 0 or 1 grid points integrates to exactly 0.0 --
    # the best possible proper score -- so an empty or single-point
    # grid silently scored perfect, and a <2-point side paired with a
    # longer side either broadcast into a finite score (e.g. a 1-point
    # cdf against 2 nominals returned 2.5) or raised a bare broadcast
    # error that never named the grid. A lawful grid needs at least 2
    # grid points on each side; lawful (>=2-point, matched) grids are
    # unaffected and stay bit-identical.
    if cdf.size < 2 or nominals.size < 2:
        raise ValueError(
            "crps_from_cdf requires a non-degenerate grid with at least "
            f"2 grid points; got empty/degenerate grid (fewer than 2 "
            f"grid points) with cdf size {cdf.size} and nominals size "
            f"{nominals.size} (SF-BL-10269)"
        )
    if cdf.size != nominals.size:
        raise ValueError(
            "crps_from_cdf grid mismatch: cdf and nominals must have "
            f"the same number of grid points; got cdf size {cdf.size} "
            f"and nominals size {nominals.size} (SF-BL-10269)"
        )
    # SF-BL-10473: reject ANY non-finite input BEFORE integrating --
    # this task's distinct input class is a non-finite VALUE inside an
    # otherwise lawful-looking grid, above all a NaN-valued CDF:
    # crps_from_cdf([0.0, NaN, 1.0], [0, 1, 2], y=1.0) propagated the
    # NaN through the trapezoid integrand and returned NaN as an
    # ordinary score, so a corrupt numeric forecast was graded as an
    # ordinary NaN CRPS in pooled comparisons (the unscreened frozen
    # copies still return NaN / inf / a finite 0.75 for y=NaN on this
    # panel). SF-BL-9924 (below) registered finiteness screens for
    # this function collaterally with its unsorted-nominals /
    # NaN-outcome classes; this screen registers the finiteness class
    # itself, per argument (cdf, then nominals, then y), so no
    # non-finite input can return a numeric or NaN CRPS even if the
    # screens below are ever reordered. Reject-only: finite inputs
    # fall through to the identical integrand, so clean-grid CRPS
    # values stay bit-identical (kill criterion).
    if not np.all(np.isfinite(cdf)):
        raise ValueError(
            "crps_from_cdf requires finite cdf values; got "
            f"non-finite cdf value(s) {cdf[~np.isfinite(cdf)]} "
            "(SF-BL-10473; finiteness screening cf. SF-BL-9924)"
        )
    if not np.all(np.isfinite(nominals)):
        raise ValueError(
            "crps_from_cdf requires finite nominals; got "
            "non-finite nominals value(s) "
            f"{nominals[~np.isfinite(nominals)]} "
            "(SF-BL-10473; finiteness screening cf. SF-BL-9924)"
        )
    try:
        _y_finite_10473 = float(y)
    except (TypeError, ValueError):
        raise ValueError(
            "crps_from_cdf requires a finite y (outcome); got "
            f"non-finite or non-numeric y={y!r} "
            "(SF-BL-10473; finiteness screening cf. SF-BL-9924)"
        ) from None
    if not np.isfinite(_y_finite_10473):
        raise ValueError(
            "crps_from_cdf requires a finite y (outcome); got "
            f"non-finite y={y!r} "
            "(SF-BL-10473; finiteness screening cf. SF-BL-9924)"
        )
    # SF-BL-9924: validate grid / outcome BEFORE integrating.
    # Pre-fix the trapezoid integrated over the nominals as given
    # with no monotonicity, finiteness or y check, unlike
    # validate_cdf() in this module which at least inspects the CDF
    # itself: unsorted nominals [2, 1, 0] returned CRPS -1.25 for
    # cdf [0, 0.5, 1] at y=1 (task repro panel reported -1.06 on its
    # own panel -- a negative proper score), y=NaN returned a finite
    # 0.75, non-finite cdf/nominals propagated NaN, and a CDF outside
    # [0, 1] scored as if lawful. Every screen below raises a named
    # ValueError; lawful (finite, strictly increasing, CDF in
    # [0, 1], finite y) grids fall through to the identical
    # integrand/trapezoid, so control CRPS values stay bit-identical.
    # Kill check: no production caller passes a deliberately
    # reversed grid -- every production caller builds nominals via
    # grid_nominals(lo, hi) with lo < hi (evaluate.py, run_hc2.py)
    # or validates span > 0 before delegating (oob_budget
    # scaled_crps) -- so raising, not sort-and-flag, is correct.
    try:
        _y_f = float(y)
    except (TypeError, ValueError):
        raise ValueError(
            "crps_from_cdf requires a finite y (outcome); got "
            f"non-finite or non-numeric y={y!r} (SF-BL-9924)"
        ) from None
    if not np.isfinite(_y_f):
        raise ValueError(
            "crps_from_cdf requires a finite y (outcome); got "
            f"non-finite y={y!r} (SF-BL-9924)"
        )
    if not np.all(np.isfinite(nominals)):
        raise ValueError(
            "crps_from_cdf requires finite nominals; got non-finite "
            f"nominals value(s) {nominals[~np.isfinite(nominals)]} "
            "(SF-BL-9924)"
        )
    if not np.all(np.isfinite(cdf)):
        raise ValueError(
            "crps_from_cdf requires finite cdf values; got "
            f"non-finite cdf value(s) {cdf[~np.isfinite(cdf)]} "
            "(SF-BL-9924)"
        )
    if np.any(np.diff(nominals) <= 0):
        raise ValueError(
            "crps_from_cdf requires strictly increasing nominals "
            "(non-increasing nominals are not a valid grid); got "
            f"non-increasing nominals {nominals} (SF-BL-9924)"
        )
    if np.any((cdf < 0.0) | (cdf > 1.0)):
        raise ValueError(
            "crps_from_cdf requires a CDF within [0, 1]; got CDF "
            f"outside [0, 1] with value(s) "
            f"{cdf[(cdf < 0.0) | (cdf > 1.0)]} (SF-BL-9924)"
        )
    integrand = (cdf - (nominals >= y).astype(float)) ** 2
    return float(np.trapz(integrand, nominals))  # np.trapezoid on numpy>=2


def _cell_density(cdf, nominals, y, floor_mass=1e-4):
    """Density at y from CDF finite differences + uniform-mixture floor.

    The floor mixes in floor_mass of a uniform distribution over the grid
    range, identically for every method, so log scores stay finite and the
    comparison is not decided by a single -inf.
    """
    # SF-BL-11316: type-validate the cdf and nominals arguments AT
    # ENTRY, before any float coercion. Pre-fix the first use of both
    # arguments in _cell_density was np.asarray(..., dtype=float),
    # which coerces with no dtype screen, so on the task panel
    # (cdf [0, 0.5, 1] over nominals [0, 50, 100], y=50, control log
    # score -4.605170185988091 and interval score -0.6931471805599453)
    # a string cdf ['0', '0.5', '1'] and string nominals
    # ['0', '50', '100'] each returned -4.605170185988091 bit-identical
    # to the numeric control (interval string-nominals likewise
    # -0.6931471805599453), and a bool cdf [False, True, True] was
    # read as [0.0, 1.0, 1.0] and returned -3.9120730066781877.
    # A bool is not a number (fleet convention, cf.
    # distribution_validity._is_number, the validate_cdf bool screen
    # SF-BL-10471 and the crps_from_cdf grid screen SF-BL-11309 in
    # this module, which fixtures the same class for crps_from_cdf,
    # a different function) and a string grid value is a corrupt
    # type, not a number: every bool or string/bytes element of
    # either array -- a Python or numpy bool/str scalar in a
    # list/tuple/other sequence (at any nesting depth), a bool- or
    # string-dtype ndarray, an object-dtype element, or a bare
    # bool/string cdf/nominals itself -- now raises a ValueError
    # naming the offending argument (cdf or nominals), at entry,
    # before the SF-BL-11242 floor_mass screen, the SF-BL-10110
    # finiteness screens and every other screen, so the type error
    # is named even when another argument is also bad. Distinct
    # from SF-BL-10110 (cdf/nominals finiteness -- value domain),
    # SF-BL-11203 (this path's outcome y type) and SF-BL-11242
    # (this function's floor_mass validation) -- none of which
    # fixtures the cdf or nominals TYPE domain. Scope is
    # deliberately narrow: only bool and string/bytes elements are
    # screened here; other non-numeric elements fall through to
    # the existing conversion / finiteness screens. Reject-only:
    # genuinely numeric grids (int / float / numpy numeric lists,
    # tuples and arrays, incl. Decimal/Fraction elements) fall
    # through untouched to the identical
    # np.asarray(..., dtype=float) below, so numeric-grid scores
    # stay bit-identical (kill criterion).
    def _bad_grid_elements_11316(_arr):
        """(bad_indices, bad_values, materialised) for bool/string
        elements of _arr; materialised is the consumed item list when
        _arr was a one-shot iterable, else None."""
        if isinstance(_arr, (bool, np.bool_)):
            return [0], [_arr], None
        if isinstance(_arr, (str, bytes)):
            return [0], [_arr], None
        if isinstance(_arr, np.ndarray):
            if _arr.dtype == np.dtype(bool) \
                    or _arr.dtype.kind in "US":
                return (list(range(_arr.size)),
                        _arr.ravel().tolist(), None)
            if _arr.dtype.kind == "O":
                _idx, _vals = [], []
                for _i, _v in enumerate(_arr.ravel().tolist()):
                    if isinstance(_v, (bool, np.bool_, str, bytes)):
                        _idx.append(_i)
                        _vals.append(_v)
                    elif isinstance(_v, np.ndarray) and (
                            _v.dtype == np.dtype(bool)
                            or _v.dtype.kind in "US"):
                        _idx.append(_i)
                        _vals.append(_v)
                return _idx, _vals, None
            return [], [], None
        if hasattr(_arr, "__iter__"):
            _items = list(_arr)
            _idx, _vals = [], []
            _stack = [(_i, _v) for _i, _v in enumerate(_items)]
            while _stack:
                _i, _v = _stack.pop()
                if isinstance(_v, (bool, np.bool_, str, bytes)):
                    _idx.append(_i)
                    _vals.append(_v)
                elif isinstance(_v, (list, tuple)):
                    _stack.extend((_i, _w) for _w in _v)
                elif isinstance(_v, np.ndarray) and (
                        _v.dtype == np.dtype(bool)
                        or _v.dtype.kind in "US"):
                    _idx.append(_i)
                    _vals.append(_v)
                elif isinstance(_v, np.ndarray) \
                        and _v.dtype.kind == "O":
                    _stack.extend(
                        (_i, _w) for _w in _v.ravel().tolist())
            _pairs = sorted(zip(_idx, _vals), key=lambda _p: _p[0])
            return ([_p[0] for _p in _pairs],
                    [_p[1] for _p in _pairs],
                    _items if not isinstance(
                        _arr, (list, tuple, range)) else None)
        return [], [], None

    _cdf_bad_idx, _cdf_bad_vals, _cdf_mat = \
        _bad_grid_elements_11316(cdf)
    if _cdf_mat is not None:
        cdf = _cdf_mat
    _nom_bad_idx, _nom_bad_vals, _nom_mat = \
        _bad_grid_elements_11316(nominals)
    if _nom_mat is not None:
        nominals = _nom_mat
    if _cdf_bad_idx:
        raise ValueError(
            "log_score_from_cdf / interval_log_score_from_cdf "
            "require cdf to be a sequence of real numbers (a bool "
            "is not a number and a string is not a number); got "
            f"bool or string cdf value(s) {_cdf_bad_vals!r} at "
            f"cdf indices {_cdf_bad_idx} (SF-BL-11316; "
            "cdf/nominals type screening cf. SF-BL-11309 "
            "crps_from_cdf grid type, SF-BL-10110 finiteness and "
            "SF-BL-11242 floor_mass screens)"
        )
    if _nom_bad_idx:
        raise ValueError(
            "log_score_from_cdf / interval_log_score_from_cdf "
            "require nominals to be a sequence of real numbers "
            "(a bool is not a number and a string is not a "
            "number); got bool or string nominals value(s) "
            f"{_nom_bad_vals!r} at nominals indices "
            f"{_nom_bad_idx} (SF-BL-11316; cdf/nominals type "
            "screening cf. SF-BL-11309 crps_from_cdf grid type, "
            "SF-BL-10110 finiteness and SF-BL-11242 floor_mass "
            "screens)"
        )
    # SF-BL-11242: validate floor_mass AT ENTRY, before any use.
    # Pre-fix floor_mass flowed unscreened into the cdf_mix line
    # below, so floor_mass=nan returned a (nan, nan) density pair,
    # floor_mass=True was read as a 1.0 floor and returned
    # (0.5, 0.5) on the task's control panel -- swamping the true
    # 0.4 cell mass -- floor_mass=-0.5 returned (0.35, 0.35), and
    # floor_mass='1e-4' died with a bare TypeError naming nothing.
    # A bool is not a number (fleet convention, cf. the SF-BL-11133
    # screen of validate_cdf's max_step_warn), a string or other
    # non-numeric floor is a caller bug, and a mixture weight must
    # be finite and non-negative; every corrupt floor_mass now
    # raises a ValueError naming floor_mass. Distinct from
    # SF-BL-11203, which fixtures only this function's outcome y,
    # and from SF-BL-8742, a floor sensitivity audit, not a
    # validation guard. Lawful floors (the 1e-4 default, 0, any
    # finite non-negative int / float / numpy numeric) fall
    # through untouched: the float copy below is used ONLY for
    # these checks and the cdf_mix computation keeps the original
    # object, so lawful floor densities stay bit-identical.
    if isinstance(floor_mass, (bool, np.bool_)):
        raise ValueError(
            "_cell_density requires floor_mass to be a real number "
            "(a bool is not a number); got "
            f"floor_mass={floor_mass!r} (SF-BL-11242)"
        )
    if isinstance(floor_mass, (str, bytes)) or floor_mass is None:
        raise ValueError(
            "_cell_density requires floor_mass to be a real number; "
            f"got floor_mass={floor_mass!r} (SF-BL-11242)"
        )
    if isinstance(floor_mass, np.ndarray) and floor_mass.ndim > 0:
        raise ValueError(
            "_cell_density requires floor_mass to be a real number "
            f"(a scalar); got floor_mass={floor_mass!r} (SF-BL-11242)"
        )
    try:
        _floor_mass_f = float(floor_mass)
    except (TypeError, ValueError):
        raise ValueError(
            "_cell_density requires floor_mass to be a real number; "
            f"got floor_mass={floor_mass!r} (SF-BL-11242)"
        ) from None
    if not np.isfinite(_floor_mass_f):
        raise ValueError(
            "_cell_density requires a finite floor_mass; got "
            f"floor_mass={floor_mass!r} (SF-BL-11242)"
        )
    if _floor_mass_f < 0.0:
        raise ValueError(
            "_cell_density requires a non-negative floor_mass; got "
            f"floor_mass={floor_mass!r} (SF-BL-11242)"
        )
    cdf = np.asarray(cdf, dtype=float)
    nominals = np.asarray(nominals, dtype=float)
    # SF-BL-10110: screen y / cdf / nominals BEFORE any comparison or
    # indexing. Every comparison against NaN is False, so a NaN outcome
    # fell past the y<=first / y>=last guards below into the searchsorted
    # branch, which places NaN one past the last cell and died with a
    # bare 'IndexError: index 3 is out of bounds' naming nothing about
    # the corrupt outcome; +/-inf outcomes silently returned scores,
    # non-finite cdf/nominals propagated NaN/inf scores, and empty or
    # single-point grids died with bare IndexErrors on nominals[-1] /
    # cdf_mix[1]. All of these now raise a named ValueError. Finite,
    # lawful (>=2-point, matched) grids are unaffected and stay
    # bit-identical: the density/mass computation below is untouched.
    if cdf.size < 2 or nominals.size < 2:
        raise ValueError(
            "log_score_from_cdf / interval_log_score_from_cdf require a "
            "non-degenerate grid with at least 2 grid points; got "
            f"empty/degenerate grid (fewer than 2 grid points) with "
            f"cdf size {cdf.size} and nominals size {nominals.size} "
            "(SF-BL-10110)"
        )
    if cdf.size != nominals.size:
        raise ValueError(
            "log_score_from_cdf / interval_log_score_from_cdf grid "
            "mismatch: cdf and nominals must have the same number of "
            f"grid points; got cdf size {cdf.size} and nominals size "
            f"{nominals.size} (SF-BL-10110)"
        )
    try:
        y_finite = bool(np.isfinite(float(y)))
    except (TypeError, ValueError):
        y_finite = False
    if not y_finite:
        raise ValueError(
            "log_score_from_cdf / interval_log_score_from_cdf require "
            f"finite y; got non-finite y={y!r} (SF-BL-10110)"
        )
    if not np.all(np.isfinite(cdf)):
        raise ValueError(
            "log_score_from_cdf / interval_log_score_from_cdf require "
            "finite cdf values; got non-finite cdf value(s) "
            f"{cdf[~np.isfinite(cdf)]} (SF-BL-10110)"
        )
    if not np.all(np.isfinite(nominals)):
        raise ValueError(
            "log_score_from_cdf / interval_log_score_from_cdf require "
            "finite nominals values; got non-finite nominals value(s) "
            f"{nominals[~np.isfinite(nominals)]} (SF-BL-10110)"
        )
    span = nominals[-1] - nominals[0]
    if not span > 0:
        raise ValueError(
            "log_score_from_cdf / interval_log_score_from_cdf require "
            "nominals spanning a positive range; got degenerate grid "
            f"with span {span} from nominals[0]={nominals[0]} to "
            f"nominals[-1]={nominals[-1]} (SF-BL-10110)"
        )
    cdf_mix = (1.0 - floor_mass) * cdf + floor_mass * (nominals - nominals[0]) / span
    # SF-BL-9116: a resolution exactly at a range bound is IN-range
    # (INTEL-metaculus Sec 3.1 / oob_budget PREREG Sec 3: inside is the
    # inclusive [range_min, range_max]; only y < lo / y > hi is
    # out-of-bounds). The bound gets its own explicit branches below
    # returning the edge-cell density/mass, so the tail branches are
    # strictly out-of-range and no scorer's predicates disagree about
    # the bound (oob_budget.metaculus_log_score has always used strict
    # y < lo / y > hi). Computations are unchanged: every branch
    # returns exactly what the pre-fix y <= first / y >= last branches
    # returned for the same y.
    if y == nominals[0]:
        dens = (cdf_mix[1] - cdf_mix[0]) / (nominals[1] - nominals[0])
        mass = cdf_mix[0] + 0.5 * (cdf_mix[1] - cdf_mix[0])
        return max(dens, 1e-12), mass
    if y == nominals[-1]:
        dens = (cdf_mix[-1] - cdf_mix[-2]) / (nominals[-1] - nominals[-2])
        mass = (1.0 - cdf_mix[-2]) + cdf_mix[-1] - cdf_mix[-2]
        return max(dens, 1e-12), 1.0 - cdf_mix[-2]
    if y < nominals[0]:
        dens = (cdf_mix[1] - cdf_mix[0]) / (nominals[1] - nominals[0])
        mass = cdf_mix[0] + 0.5 * (cdf_mix[1] - cdf_mix[0])
        return max(dens, 1e-12), mass
    if y > nominals[-1]:
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
    # SF-BL-11351: type-validate cdf BEFORE any float coercion.
    # Pre-fix cdf was never type-screened at this entry point --
    # in the repo/out copies the first use was
    # np.asarray(cdf, dtype=float), which coerces with no dtype
    # screen, and in the wave3 original cdf flowed straight into
    # the per-bound cdf_at / np.interp evaluations -- so on the
    # task panel (nominals [0, 50, 100], bin_edges [25, 75])
    # cdf=['0', '0.5', '1'] returned [0.25, 0.5, 0.25]
    # bit-identical to the numeric control [0, 0.5, 1] (string
    # cdf elements parse through the per-bound cdf_at
    # evaluations) and cdf=[False, True, True] returned
    # [0.5, 0.5, 1e-06] (bool cdf read as [0, 1, 1], fabricated
    # into a two-bin split by the epsilon floor / renormalisation);
    # on wave3 a string cdf died inside cdf_at with a bare numpy
    # TypeError that never named cdf. A bool is not a number
    # (fleet convention, cf. validate_cdf SF-BL-10471 and
    # crps_from_cdf SF-BL-11134 in this module) and a string cdf
    # element is a caller type bug, not a probability: every bool
    # or string/bytes cdf element -- list/tuple element,
    # np.bool_/np.str_/np.bytes_ scalar, bool- or string-dtype
    # ndarray, object-dtype element, nested 0-d bool/string
    # array element, or a bare bool/string cdf itself -- now
    # raises a ValueError naming cdf, at entry, before the
    # SF-BL-11315 nominals type screen and every other screen,
    # so the type error is named even when another argument is
    # also bad. Distinct from SF-BL-11315 (this function's
    # nominals TYPE only), SF-BL-11272 (its bin_edges TYPE only),
    # SF-BL-11308 (the cdf type at the cdf_at entry point -- a
    # different function; per-entry-point granularity is the
    # established pattern, cf. SF-BL-11275 vs SF-BL-9845) and
    # SF-BL-10190 (cdf finiteness -- value domain), none of
    # which fixtures the cdf type domain at this entry point.
    # Scope is deliberately narrow: only bool and string/bytes
    # elements are screened here; other non-numeric elements
    # fall through to the existing conversion / finiteness
    # screens. Reject-only: genuinely numeric cdf values
    # (int / float / numpy numeric, incl. Decimal/Fraction
    # elements) fall through untouched to the identical
    # conversion below, so numeric-cdf bin probabilities stay
    # bit-identical.
    _cdf_bad_indices = []
    _cdf_bad_values = []
    if isinstance(cdf, (bool, np.bool_)):
        _cdf_bad_indices = [0]
        _cdf_bad_values = [cdf]
    elif isinstance(cdf, (str, bytes)):
        _cdf_bad_indices = [0]
        _cdf_bad_values = [cdf]
    elif isinstance(cdf, np.ndarray):
        if cdf.dtype == np.dtype(bool) \
                or cdf.dtype.kind in "US":
            _cdf_bad_indices = list(range(cdf.size))
            _cdf_bad_values = cdf.ravel().tolist()
        elif cdf.dtype.kind == "O":
            for _i, _v in enumerate(cdf.ravel().tolist()):
                if isinstance(_v, (bool, np.bool_, str, bytes)):
                    _cdf_bad_indices.append(_i)
                    _cdf_bad_values.append(_v)
    elif hasattr(cdf, "__iter__"):
        _cdf_items = list(cdf)
        cdf = _cdf_items
        for _i, _v in enumerate(_cdf_items):
            if isinstance(_v, (bool, np.bool_, str, bytes)):
                _cdf_bad_indices.append(_i)
                _cdf_bad_values.append(_v)
            elif isinstance(_v, np.ndarray) and (
                    _v.dtype == np.dtype(bool)
                    or _v.dtype.kind in "US"):
                _cdf_bad_indices.append(_i)
                _cdf_bad_values.append(_v)
    if _cdf_bad_indices:
        raise ValueError(
            "cdf_to_bin_probs requires cdf to be real "
            "numbers (a bool is not a number and a string is not "
            "a number); got bool or string cdf value(s) "
            f"{_cdf_bad_values!r} at cdf indices "
            f"{_cdf_bad_indices} (SF-BL-11351; cdf type "
            "screening cf. SF-BL-11315 nominals type, "
            "SF-BL-11272 bin_edges type, SF-BL-11308 cdf_at "
            "cdf type and SF-BL-10190 finiteness)"
        )
    # SF-BL-11315: type-validate nominals BEFORE any float coercion.
    # Pre-fix bounds were built directly from nominals with no type
    # screen -- in the repo/out copies the first use was
    # np.asarray(nominals, dtype=float), which coerces with no dtype
    # screen, and in the wave3 original nominals flowed straight into
    # cdf_at / np.interp -- so on the task panel
    # (cdf [0, 0.5, 1], bin_edges [25, 75])
    # nominals=['0', '50', '100'] returned [0.25, 0.5, 0.25]
    # bit-identical to the numeric control [0, 50, 100] (string
    # nominals parse through cdf_at) and nominals=[False, True, 100]
    # returned [0.621, 0.253, 0.126] (bool nominals become 0/1 grid
    # points); a bare string nominal in wave3 died with a bare numpy
    # error that never named nominals. A bool is not a number (fleet
    # convention, cf. validate_cdf SF-BL-10471 and crps_from_cdf
    # SF-BL-11134 in this module) and a string-dtype nominal is a
    # caller type bug, not a grid point: every bool or string/bytes
    # nominal -- list/tuple element, np.bool_/np.str_/np.bytes_
    # scalar, bool- or string-dtype ndarray, object-dtype element,
    # or a bare bool/string nominals itself -- now raises a
    # ValueError naming nominals, at entry, before the SF-BL-11272
    # bin_edges type screen and every other screen, so the type
    # error is named even when another argument is also bad.
    # Distinct from SF-BL-11272 (this function's bin_edges TYPE
    # only), SF-BL-9926 (bin_edges value domain) and SF-BL-10190
    # (cdf finiteness -- value domain); none of those fixtures the
    # nominals type domain. Scope is deliberately narrow: only bool
    # and string/bytes elements are screened here; other
    # non-numeric elements fall through to the existing conversion /
    # finiteness screens. Reject-only: genuinely numeric nominals
    # (int / float / numpy numeric, incl. Decimal/Fraction elements)
    # fall through untouched to the identical conversion below, so
    # numeric-nominal bin probabilities stay bit-identical.
    _nom_bad_indices = []
    _nom_bad_values = []
    if isinstance(nominals, (bool, np.bool_)):
        _nom_bad_indices = [0]
        _nom_bad_values = [nominals]
    elif isinstance(nominals, (str, bytes)):
        _nom_bad_indices = [0]
        _nom_bad_values = [nominals]
    elif isinstance(nominals, np.ndarray):
        if nominals.dtype == np.dtype(bool) \
                or nominals.dtype.kind in "US":
            _nom_bad_indices = list(range(nominals.size))
            _nom_bad_values = nominals.ravel().tolist()
        elif nominals.dtype.kind == "O":
            for _i, _v in enumerate(nominals.ravel().tolist()):
                if isinstance(_v, (bool, np.bool_, str, bytes)):
                    _nom_bad_indices.append(_i)
                    _nom_bad_values.append(_v)
    elif hasattr(nominals, "__iter__"):
        _nom_items = list(nominals)
        nominals = _nom_items
        for _i, _v in enumerate(_nom_items):
            if isinstance(_v, (bool, np.bool_, str, bytes)):
                _nom_bad_indices.append(_i)
                _nom_bad_values.append(_v)
            elif isinstance(_v, np.ndarray) and (
                    _v.dtype == np.dtype(bool)
                    or _v.dtype.kind in "US"):
                _nom_bad_indices.append(_i)
                _nom_bad_values.append(_v)
    if _nom_bad_indices:
        raise ValueError(
            "cdf_to_bin_probs requires nominals to be real "
            "numbers (a bool is not a number and a string is not "
            "a number); got bool or string nominals value(s) "
            f"{_nom_bad_values!r} at nominals indices "
            f"{_nom_bad_indices} (SF-BL-11315; nominals type "
            "screening cf. SF-BL-11272 bin_edges type, "
            "SF-BL-10190 finiteness and SF-BL-9926 value domain)"
        )
    # SF-BL-11272: type-validate bin_edges BEFORE any float coercion.
    # Pre-fix the first use was np.asarray(bin_edges, dtype=float),
    # which coerces with no dtype screen, so on the task panel
    # (cdf [0.1, 0.5, 0.9] over nominals [0, 100, 200])
    # bin_edges=['50', '150'] returned [0.3, 0.4, 0.3] bit-identical
    # to the numeric control [50, 150] and bin_edges=[True] returned
    # [0.104, 0.896] (bool read as an edge at nominal 1.0); a bare
    # string died with a numpy concatenate ValueError that never
    # named bin_edges. A bool is not a number (fleet convention, cf.
    # validate_cdf SF-BL-10471 and crps_from_cdf SF-BL-11134 in this
    # module) and a string-dtype edge is a config-typing bug, not a
    # value: every bool or string/bytes edge -- list/tuple element,
    # np.bool_/np.str_/np.bytes_ scalar, bool- or string-dtype
    # ndarray, object-dtype element, or a bare bool/string bin_edges
    # itself -- now raises a ValueError naming bin_edges, at entry,
    # before the SF-BL-10276 finiteness screen and every other
    # screen, so the type error is named even when another argument
    # is also bad. Distinct from SF-BL-9926 (edge VALUE domain:
    # non-increasing / outside the grid) and SF-BL-10276 (edge
    # finiteness: NaN/+/-inf), neither of which fixtures edge TYPE.
    # Scope is deliberately narrow: only bool and string/bytes
    # elements are screened here; other non-numeric elements fall
    # through to the existing conversion / finiteness screens.
    # Reject-only: genuinely numeric edges (int / float / numpy
    # numeric, incl. Decimal/Fraction elements) fall through
    # untouched to the identical np.asarray(..., dtype=float), so
    # numeric-edge bin probabilities stay bit-identical.
    _be_bad_indices = []
    _be_bad_values = []
    if isinstance(bin_edges, (bool, np.bool_)):
        _be_bad_indices = [0]
        _be_bad_values = [bin_edges]
    elif isinstance(bin_edges, (str, bytes)):
        _be_bad_indices = [0]
        _be_bad_values = [bin_edges]
    elif isinstance(bin_edges, np.ndarray):
        if bin_edges.dtype == np.dtype(bool) \
                or bin_edges.dtype.kind in "US":
            _be_bad_indices = list(range(bin_edges.size))
            _be_bad_values = bin_edges.ravel().tolist()
        elif bin_edges.dtype.kind == "O":
            for _i, _v in enumerate(bin_edges.ravel().tolist()):
                if isinstance(_v, (bool, np.bool_, str, bytes)):
                    _be_bad_indices.append(_i)
                    _be_bad_values.append(_v)
    elif hasattr(bin_edges, "__iter__"):
        _be_items = list(bin_edges)
        bin_edges = _be_items
        for _i, _v in enumerate(_be_items):
            if isinstance(_v, (bool, np.bool_, str, bytes)):
                _be_bad_indices.append(_i)
                _be_bad_values.append(_v)
            elif isinstance(_v, np.ndarray) and (
                    _v.dtype == np.dtype(bool)
                    or _v.dtype.kind in "US"):
                _be_bad_indices.append(_i)
                _be_bad_values.append(_v)
    if _be_bad_indices:
        raise ValueError(
            "cdf_to_bin_probs requires bin_edges to be real "
            "numbers (a bool is not a number and a string is not "
            "a number); got bool or string bin_edges value(s) "
            f"{_be_bad_values!r} at bin_edges indices "
            f"{_be_bad_indices} (SF-BL-11272; bin_edges type "
            "screening cf. SF-BL-10276 finiteness and "
            "SF-BL-9926 value domain)"
        )
    cdf = np.asarray(cdf, dtype=float)
    nominals = np.asarray(nominals, dtype=float)
    edges = np.asarray(bin_edges, dtype=float)
    # SF-BL-10276: screen bin_edges for non-finiteness FIRST, before
    # any other validation or interpolation. This is the non-finite
    # bin_edges input class, distinct from SF-BL-10190 (NaN in the cdf
    # values argument) and SF-BL-9926 (unsorted but finite edges):
    # unscreened, a NaN edge flowed into cdf_at interpolation and
    # cdf_to_bin_probs([0.0, 0.5, 1.0], [0, 1, 2], bin_edges=[NaN])
    # returned [nan, nan] as a normal result (and +/-inf edges
    # silently returned plausible-looking distributions). SF-BL-10190
    # later added an edges screen to the live copies, but it ran
    # AFTER the cdf/grid screens, so a non-finite edge paired with a
    # non-finite CDF or a degenerate grid raised an error that did
    # not name bin_edges, and the wave3 original never got the screen
    # at all. The preregistered criterion is unconditional -- a
    # ValueError naming bin_edges if ANY edge is non-finite -- so the
    # check sits immediately after the asarray conversion. Finite
    # edges fall through untouched, so finite-edge outputs stay
    # bit-identical. Cross-references keep the SF-BL-10190 /
    # SF-BL-10109 suites' message assertions valid.
    if not np.all(np.isfinite(edges)):
        _bad_edges = np.asarray(edges, dtype=float).ravel()
        raise ValueError(
            "cdf_to_bin_probs requires finite bin_edges; got "
            "non-finite bin_edges value(s) "
            f"{_bad_edges[~np.isfinite(_bad_edges)]} "
            "(SF-BL-10276; bin_edges screening cf. SF-BL-10190; "
            "grid validation cf. SF-BL-10109)"
        )
    # SF-BL-10190: screen the CDF (and the grid / edges it is read
    # against) BEFORE any interpolation. Pre-fix nothing was screened:
    # np.interp propagated a NaN CDF entry into every bin value and the
    # epsilon-floor / renormalisation step never detected that the
    # result summed to NaN, not 1 — the task repro
    # cdf_to_bin_probs([0.0, NaN, 1.0], [0, 1, 2], [0.5, 1.5]) returned
    # [NaN, NaN, NaN] — and a non-monotone CDF's negative bin masses
    # were silently clipped away and renormalised into a plausible
    # distribution. Every screen below raises a named ValueError;
    # finite, non-decreasing CDFs on lawful grids fall through to the
    # identical computation, so clean CDFs stay byte-identical.
    if cdf.size < 2 or nominals.size < 2:
        raise ValueError(
            "cdf_to_bin_probs requires a non-degenerate grid with at "
            "least 2 grid points; got empty/degenerate grid (fewer "
            f"than 2 grid points) with cdf size {cdf.size} and "
            f"nominals size {nominals.size} (SF-BL-10190)"
        )
    if cdf.size != nominals.size:
        raise ValueError(
            "cdf_to_bin_probs grid mismatch: cdf and nominals must "
            "have the same number of grid points; got cdf size "
            f"{cdf.size} and nominals size {nominals.size} "
            "(SF-BL-10190; grid validation cf. SF-BL-10109)"
        )
    if not np.all(np.isfinite(cdf)):
        raise ValueError(
            "cdf_to_bin_probs requires a finite CDF; got non-finite "
            f"CDF value(s) {cdf[~np.isfinite(cdf)]} "
            "(SF-BL-10190; grid validation cf. SF-BL-10109)"
        )
    if not np.all(np.isfinite(nominals)):
        raise ValueError(
            "cdf_to_bin_probs requires finite nominals; got "
            "non-finite nominals value(s) "
            f"{nominals[~np.isfinite(nominals)]} "
            "(SF-BL-10190; grid validation cf. SF-BL-10109)"
        )
    if not np.all(np.isfinite(edges)):
        raise ValueError(
            "cdf_to_bin_probs requires finite bin_edges; got "
            "non-finite bin edge value(s) "
            f"{edges[~np.isfinite(edges)]} "
            "(SF-BL-10190; grid validation cf. SF-BL-10109)"
        )
    if np.any(np.diff(nominals) <= 0):
        raise ValueError(
            "cdf_to_bin_probs requires strictly increasing nominals; "
            f"got nominals {nominals} "
            "(SF-BL-10190; grid validation cf. SF-BL-10109)"
        )
    # Monotonicity tolerance mirrors validate_cdf in this module: a
    # decrease larger than 1e-12 is a non-monotone CDF, not float noise.
    drops = np.diff(cdf)
    if np.any(drops < -1e-12):
        i = int(np.argmin(drops))
        raise ValueError(
            "cdf_to_bin_probs requires a non-decreasing (monotone) "
            f"CDF; got CDF decrease of {drops[i]} between grid points "
            f"{i} and {i + 1} (cdf[{i}]={cdf[i]}, cdf[{i + 1}]="
            f"{cdf[i + 1]}) (SF-BL-10190)"
        )
    bounds = np.concatenate([[nominals[0]], edges, [nominals[-1]]])
    cdf_vals = np.array([cdf_at(cdf, nominals, b) for b in bounds])
    cdf_vals[0], cdf_vals[-1] = 0.0, 1.0
    probs = np.diff(cdf_vals)
    probs = np.clip(probs, 1e-6, None)
    probs = probs / probs.sum()
    # SF-BL-10190 post-condition: never hand mc_scores a distribution
    # that is not finite and normalised — assert it on the way out.
    if not np.all(np.isfinite(probs)) or abs(float(probs.sum()) - 1.0) > 1e-9:
        raise ValueError(
            "cdf_to_bin_probs produced bin probabilities that are not "
            f"finite and summing to 1 within 1e-9; got probs {probs} "
            f"with sum {float(probs.sum())} (SF-BL-10190)"
        )
    return probs


def mc_scores(probs, true_bin):
    """(log score, Brier) for a multiple-choice forecast. Higher log better,
    lower Brier better.

    SF-BL-9925: inputs are validated before any scoring — the screens
    below are reject-only, so every valid call is bit-identical to the
    pre-fix computation. ``probs`` must be a non-empty 1-D vector of
    finite, non-negative probabilities summing to 1 within 1e-6 (the
    fleet SUM_TOL for multiple-choice distributions, cf.
    distribution_validity.py); ``true_bin`` must be an integer in
    [0, len(probs)). Any violation raises a ValueError naming
    mc_scores and SF-BL-9925. Pre-fix, true_bin=-1 silently scored
    the LAST bin via negative indexing (indistinguishable from a
    genuine last-bin resolution) and off-simplex probs such as
    [-0.5, 1.5] returned a Brier of 4.5, above the 2.0 maximum for
    any valid probability vector. No caller relies on -1 as a
    last-bin shorthand (audited 2026-10-23, SF-BL-9925 RESULTS.md):
    every live call site computes true_bin in [0, len(probs)).

    SF-BL-10472: distinct input class — a NaN *member* probability.
    On the wave3 source-of-truth copy mc_scores([NaN, 0.5],
    true_bin=0) still returned (nan, nan) live: max(NaN, 1e-12) and
    the Brier sum both propagate NaN, and a NaN sum defeats a
    sum-only screen because every NaN comparison is False, so a
    corrupt multiple-choice panel entered classwise ECE/Brier
    accounting as an ordinary NaN score pair (distinct from the
    SF-BL-9787-line true_bin=-1 and off-simplex [-0.5, 1.5] cases,
    which SF-BL-9925 already screens on the live copies). The live
    copies also had no explicit per-member upper bound: a member
    above 1 within the sum tolerance (e.g. [1.0000004, 0.0], sum
    1.0000004 within 1e-6) returned a positive log score. The
    screens below therefore (a) check finiteness FIRST, before any
    comparison or sum, (b) check every member is in [0, 1]
    explicitly (both bounds, not via the sum), and (c) dual-tag
    every raise SF-BL-10472 + SF-BL-9925 so both suites' message
    assertions stay valid. Reject-only: clean panels unchanged.

    SF-BL-11269: distinct input class -- the *type* of a probs
    member. The screens above validate values only *after*
    ``np.asarray(probs, dtype=float)`` has coerced every member,
    and that coercion is type-blind: a numeric string parses to
    its value, so mc_scores(['0.5', '0.5'], true_bin=1) returned
    (-0.693147, 0.5), bit-identical to the lawful numeric control,
    and a bool is read as 1.0/0.0, so mc_scores([True, False],
    true_bin=0) returned a perfect (0.0, 0.0) for boolean flags.
    SF-BL-9925 fixtures probs VALUES (off-simplex panels) and
    SF-BL-10472 non-finite members; neither fixtures member TYPE,
    and SF-BL-11202 fixtures only true_bin's type. The screen
    below therefore runs BEFORE any float coercion, on the raw
    input: any bool or string/bytes member -- a Python or numpy
    bool/str scalar, an element of a list/tuple/other sequence
    (at any nesting depth), a bool- or string-dtype ndarray, or
    a bare string/bytes probs itself -- raises a ValueError
    naming probs. A bool is not a number (fleet convention, cf.
    validate_cdf SF-BL-10471) and a numeric string is not a
    number either: both are corrupt member types, never parsed.
    The raises are triple-tagged SF-BL-11269 + SF-BL-10472 +
    SF-BL-9925 so every suite's message assertions stay valid
    (the SF-BL-9925 structural suite already expects its tag on
    ['a', 'b']). Reject-only: the screen inspects types only and
    keeps the original probs object for the conversion below,
    so genuinely numeric probs (int/float lists, tuples and
    numeric-dtype arrays, incl. NaN/inf floats, whose value
    screens live above) fall through untouched and their
    (log score, Brier) pairs stay bit-identical.
    """
    # SF-BL-11269: type-validate probs BEFORE any float coercion
    # (see docstring). The float conversion below would otherwise
    # silently parse string members and read bools as 1.0/0.0.
    _tags_11269 = "(SF-BL-11269; SF-BL-10472; SF-BL-9925)"
    if isinstance(probs, (bool, np.bool_)):
        raise ValueError(
            "mc_scores requires probs to be a numeric probability "
            "vector (a bool is not a number); got boolean "
            f"probs={probs!r} {_tags_11269}"
        )
    if isinstance(probs, (str, bytes)):
        raise ValueError(
            "mc_scores requires probs to be a numeric probability "
            "vector; got string probs "
            f"probs={probs!r} (a string is not a number) "
            f"{_tags_11269}"
        )
    if isinstance(probs, np.ndarray):
        if probs.dtype == np.dtype(bool):
            raise ValueError(
                "mc_scores requires probs to be a numeric "
                "probability vector (a bool is not a number); got "
                f"bool-dtype probs array probs={probs!r} "
                f"{_tags_11269}"
            )
        if probs.dtype.kind in "US":
            raise ValueError(
                "mc_scores requires probs to be a numeric "
                "probability vector; got string-dtype probs array "
                f"with dtype {probs.dtype} probs={probs!r} "
                f"(a string is not a number) {_tags_11269}"
            )
        if probs.dtype.kind == "O":
            _stack_11269 = list(probs.ravel().tolist())
            while _stack_11269:
                _v = _stack_11269.pop()
                if isinstance(_v, (bool, np.bool_)):
                    raise ValueError(
                        "mc_scores requires probs to be a numeric "
                        "probability vector (a bool is not a "
                        "number); got boolean probs element "
                        f"{_v!r} in probs={probs!r} {_tags_11269}"
                    )
                if isinstance(_v, (str, bytes)):
                    raise ValueError(
                        "mc_scores requires probs to be a numeric "
                        "probability vector; got string probs "
                        f"element {_v!r} in probs={probs!r} "
                        f"(a string is not a number) {_tags_11269}"
                    )
                if isinstance(_v, (list, tuple)):
                    _stack_11269.extend(_v)
        elif probs.dtype.kind not in "iuf":
            raise ValueError(
                "mc_scores requires probs to be a numeric "
                "probability vector; got probs array with "
                f"non-numeric dtype {probs.dtype} "
                f"probs={probs!r} {_tags_11269}"
            )
    elif hasattr(probs, "__iter__"):
        _items_11269 = list(probs)
        if not isinstance(probs, (list, tuple, range, dict, set,
                                  frozenset)):
            # One-shot iterable (generator/iterator): keep the
            # materialised items for the conversion below, since
            # the scan just consumed the original.
            probs = _items_11269
        _stack_11269 = list(_items_11269)
        while _stack_11269:
            _v = _stack_11269.pop()
            if isinstance(_v, (bool, np.bool_)):
                raise ValueError(
                    "mc_scores requires probs to be a numeric "
                    "probability vector (a bool is not a number); "
                    f"got boolean probs element {_v!r} in "
                    f"probs={probs!r} {_tags_11269}"
                )
            if isinstance(_v, (str, bytes)):
                raise ValueError(
                    "mc_scores requires probs to be a numeric "
                    "probability vector; got string probs element "
                    f"{_v!r} in probs={probs!r} (a string is not "
                    f"a number) {_tags_11269}"
                )
            if isinstance(_v, (list, tuple)):
                _stack_11269.extend(_v)
            elif isinstance(_v, np.ndarray):
                if _v.dtype == np.dtype(bool) \
                        or _v.dtype.kind in "US":
                    raise ValueError(
                        "mc_scores requires probs to be a numeric "
                        "probability vector; got bool/string-dtype "
                        f"probs element {_v!r} in probs={probs!r} "
                        f"{_tags_11269}"
                    )
                if _v.dtype.kind == "O":
                    _stack_11269.extend(_v.ravel().tolist())
    try:
        probs_arr = np.asarray(probs, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "mc_scores requires probs to be a numeric 1-D probability "
            f"vector; got probs {probs!r} (SF-BL-10472; SF-BL-9925)"
        ) from exc
    if probs_arr.ndim != 1 or probs_arr.size == 0:
        raise ValueError(
            "mc_scores requires a non-empty 1-D probability vector; "
            f"got probs with shape {probs_arr.shape} (SF-BL-10472; SF-BL-9925)"
        )
    if not np.all(np.isfinite(probs_arr)):
        raise ValueError(
            "mc_scores requires finite probabilities in [0, 1]; got "
            "non-finite probs value(s) "
            f"{probs_arr[~np.isfinite(probs_arr)]} "
            f"in probs {probs_arr} (SF-BL-10472; SF-BL-9925)"
        )
    if np.any((probs_arr < 0) | (probs_arr > 1)):
        raise ValueError(
            "mc_scores requires probabilities in [0, 1]; got "
            "out-of-[0, 1] probs value(s) "
            f"{probs_arr[(probs_arr < 0) | (probs_arr > 1)]} "
            f"in probs {probs_arr} (SF-BL-10472; SF-BL-9925)"
        )
    total = float(probs_arr.sum())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            "mc_scores requires probs summing to 1 within 1e-6; "
            f"got probs {probs_arr} with sum {total} (SF-BL-10472; SF-BL-9925)"
        )
    if isinstance(true_bin, bool) or not isinstance(
            true_bin, (int, np.integer)):
        raise ValueError(
            "mc_scores requires true_bin to be an integer in "
            f"[0, {probs_arr.size}); got {true_bin!r} (SF-BL-10472; SF-BL-9925)"
        )
    if not 0 <= int(true_bin) < probs_arr.size:
        raise ValueError(
            "mc_scores requires true_bin in "
            f"[0, {probs_arr.size}); got true_bin {true_bin} "
            f"for {probs_arr.size} bins (SF-BL-10472; SF-BL-9925)"
        )
    probs = probs_arr
    true_bin = int(true_bin)
    onehot = np.zeros_like(probs)
    onehot[true_bin] = 1.0
    return float(np.log(max(probs[true_bin], 1e-12))), float(np.sum((probs - onehot) ** 2))
