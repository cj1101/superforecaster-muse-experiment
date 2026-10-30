"""
Superforecaster — production policy engine (integration layer).

This module encodes, as executable policy, everything the v1 research
workstreams actually found. It does not contain a new model. It contains the
rules for *combining* models, priors and agent forecasts so that the system
only ever deviates from a strong prior as far as validated evidence allows.

Findings baked in (all numbers in ../results/):

1. MARKETS / CRYPTO DIRECTION: classic price-feature logistic models did NOT
   beat expanding climatology out-of-sample in any of the 12 equity cells or
   the 6 crypto direction cells (walk-forward, 2019-2026). Therefore a raw
   model probability is never shipped. It is shrunk toward the prior with a
   weight derived from its validated skill, and given zero weight unless
   that skill is positive. The v2 calibration-rescue experiment
   (results/calibration_rescue.json, 20 streams, fully causal) showed fitted
   shrinkage is the robust layer: skill vs climatology stayed inside
   [-0.009, +0.025] on every stream, converting models that lost up to
   -0.10 raw into approximately-climatology forecasts while preserving the
   crash-risk edge. Platt scaling applied on top HURT several real streams
   (NVDA 21d skill -0.051) despite shining on synthetic data — regime drift
   breaks a fixed recalibration map — so Platt is NOT in the production
   path; it may only be used causally-refit, per-stream, behind the same
   skill gate.

2. PREDICTION MARKETS: on 763 resolved Polymarket markets ($26.1B volume),
   no adjustment rule beat the raw market price out-of-sample on both proper
   scores (raw T-7d test Brier 0.09051). Therefore, when a liquid market
   price exists it IS the prior, with anchor weight >= 0.85, and model/agent
   views may only move it within the remaining weight unless they carry a
   validated track record in that domain.

3. CRYPTO CRASH RISK is the one validated model edge (BTC Brier 0.0901 vs
   base rate 0.0922, AUC 0.607; ETH 0.1417 vs 0.1446). Rare-event models are
   never class-balance-weighted (that destroyed calibration in testing:
   Brier 0.183 vs 0.092).

4. AGENT AGGREGATION: in a 1,000-question simulated tournament, a
   track-record-weighted log-odds pool beat the best single agent
   (Brier 0.1850 vs 0.1863; gap widens with n). Fixed extremizing (1.3x)
   HURT in every seed, and a separate calibration study found blind
   extremizing hurts whenever forecasters share information, while fitted
   extremization recovers the theory (a=2.96) when agents hold genuinely
   independent partial information. Therefore: extremization defaults to
   1.00 and may only be *learned* per domain after >=150 resolved questions,
   capped at 1.40, adopted only if it beats 1.00 held-out.

5. CALIBRATION research defaults (synthetic benchmarks): Platt scaling per
   forecaster (beta calibration when n >= ~2000), isotonic NOT default (it
   hurt a well-calibrated forecaster's log score at n=6,000), log score is
   the fitting objective everywhere. Production overrides per finding 1:
   on real drifting streams, fitted shrinkage toward the prior is the
   default transform; recalibration maps must re-earn their place causally.

Usage:
    sf = Superforecaster()
    p = sf.forecast(prior=0.62, model_probs=[0.71, 0.66],
                    model_skills=[0.03, -0.01])   # negative skill -> dropped
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logit

_EPS = 1e-6


def _require_finite(name, value):
    """SF-BL-9510: reject NaN/inf inputs explicitly.

    np.clip leaves NaN untouched, so without this check a NaN prior,
    market price, component probability or shrink weight silently
    pooled to a NaN forecast shipped with a success-style note.
    Raises ValueError carrying the INVALID_INPUT flag; never returns
    a bare NaN to callers.

    SF-BL-11054: reject string-typed numbers BEFORE the float()
    coercion below. float('0.8') succeeds, so shrink('0.8', '0.5',
    '0.5') silently returned 0.65, skill_weight('0.10', '0.25')
    returned 0.6 and log_odds_pool(['0.7', '0.3']) returned 0.5 --
    production numbers manufactured from strings -- while
    forecast(prior='0.5') passed this screen on the coerced value
    and died downstream with a bare UFuncTypeError. A str / bytes /
    bytearray input (np.str_ / np.bytes_ included, as subclasses)
    is a caller type error, not a number: raise the named
    INVALID_INPUT ValueError naming the argument. No other type is
    newly restricted -- bool is SF-BL-10937/SF-BL-10980's dimension,
    and lawful numeric types (int, float, np.float64, np.int64,
    Decimal, ...) keep flowing through float() exactly as before.
    """
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(
            f"INVALID_INPUT: {name}={value!r} is not a finite number "
            f"(string inputs rejected; a numeric type is required, "
            f"got {type(value).__name__})")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"INVALID_INPUT: {name}={value!r} is not a finite number") from None
    if not np.isfinite(v):
        raise ValueError(f"INVALID_INPUT: {name}={value!r} is not finite (NaN/inf rejected)")
    return v


def _require_not_bool(name, value):
    """SF-BL-10921: reject bool inputs to Superforecaster.forecast().

    bool subclasses int, so float(True)==1.0 / float(False)==0.0 pass
    the _require_finite screen and every downstream range check, turning
    a JSON/config flag leaking into a probability field into the most
    extreme lawful value. np.bool_ is the same leak in numpy dress.
    Raises ValueError carrying the INVALID_INPUT flag naming the
    argument; int 1/0 are NOT bools and stay lawful.

    SF-BL-10980: this screen is also applied by skill_weight (both
    Brier arguments), by shrink (p_model / p_prior only) and by the
    standalone log_odds_pool (the probs container and every probs
    member). The remaining surfaces that route through
    _require_finite -- the constructor, skill_weight_from_skill,
    shrink's w argument and log_odds_pool's weights -- are
    SF-BL-10937 / SF-BL-11022's separate dimensions and are
    deliberately NOT screened here or in those functions by this
    task, so the shared _require_finite helper is untouched.

    SF-BL-11022: this screen is now also applied by the standalone
    log_odds_pool to its weights argument (the weights container
    and every weights element), closing the dimension SF-BL-10980
    left open above. The constructor, skill_weight_from_skill and
    shrink's w argument remain SF-BL-10937's separate dimension.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"INVALID_INPUT: {name}={value!r} is not a finite number "
            f"(bool inputs rejected; a numeric type is required, "
            f"got {type(value).__name__})")
    return value


def _require_finite_seq(name, values):
    if values is None:
        return None
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(
            f"INVALID_INPUT: {name}={values!r} is not a finite number "
            f"(string inputs rejected; a sequence of numeric values "
            f"is required, got {type(values).__name__})")
    out = []
    for i, v in enumerate(values):
        out.append(_require_finite(f"{name}[{i}]", v))
    return out


def _clip(p):
    return float(np.clip(p, _EPS, 1 - _EPS))


def shrink(p_model: float, p_prior: float, w: float) -> float:
    """Linear shrink of a model probability toward a prior. w in [0, 1].

    SF-BL-9510: p_model, p_prior and w must all be finite; NaN/inf
    raises ValueError (INVALID_INPUT) instead of returning NaN --
    np.clip(w, 0, 1) leaves a NaN weight untouched.

    SF-BL-10980: p_model and p_prior must not be bools. bool
    subclasses int, so pre-fix float(True)==1.0 / float(False)==0.0
    passed the finite screen: shrink(0.8, True, 0.5) returned 0.9 and
    shrink(True, False, 0.5) returned 0.5 -- a JSON/config flag
    leaking into a probability field became an extreme lawful value.
    A bool p_model / p_prior now raises ValueError (INVALID_INPUT)
    naming the argument, before any float() coercion. Scope is
    p_model / p_prior only: w is SF-BL-10937's separate dimension.
    """
    _require_not_bool("shrink p_model", p_model)
    _require_not_bool("shrink p_prior", p_prior)
    p_model = _require_finite("shrink p_model", p_model)
    p_prior = _require_finite("shrink p_prior", p_prior)
    w = float(np.clip(_require_finite("shrink w", w), 0.0, 1.0))
    return _clip(w * p_model + (1 - w) * p_prior)


def fit_shrink_weight(y, p_model, p_prior, grid=None) -> float:
    """Fit the shrink weight w by minimizing Brier score on PAST forecasts.

    y, p_model, p_prior are arrays of already-resolved forecasts (each
    p_prior must be the prior that was known at that forecast's time).
    Returns w in [0, 1]; 0 means 'the model earned no deviation from prior'.

    SF-BL-8869: the three history arrays must be non-empty, 1-D and of
    exactly equal length, and a supplied grid must be non-empty, 1-D
    and finite. An empty or length-mismatched history raises ValueError
    (INVALID_INPUT) instead of returning a numeric w: previously an
    empty history returned the initial best_w=0.0 standing (mean of an
    empty slice is NaN, which never beats inf), indistinguishable from
    a fitted 'no deviation earned', and a length-1 p_prior broadcast
    silently across the whole history.

    SF-BL-11501: a supplied grid must contain only numeric entries.
    np.asarray(grid, float) coerces bool and numeric-string entries,
    so pre-fix grid=[True, False] was read as weights 1 and 0 and
    silently fitted w=1.0, and grid=['0.5', '1.0'] was silently
    coerced the same way (or died with a bare UFuncTypeError naming
    no argument). Any bool or string grid entry -- including a
    string-like grid container, a bool/string-dtype numpy grid, and
    np.bool_ / np.str_ / np.bytes_ elements -- now raises ValueError
    (INVALID_INPUT) naming grid, before any float() coercion. Scope
    is the grid TYPE domain only: the grid VALUE domain ([0, 1]
    range) is SF-BL-9783's separate dimension, and numeric entries
    flow through untouched, bit-identical.

    SF-BL-11019: the history VALUES must be lawful — every outcome
    in y must be exactly 0 or 1, and every p_model / p_prior entry
    must lie in [0, 1] (boundaries inclusive). Pre-fix the fitter
    screened only finiteness, so fit_shrink_weight([2, 0],
    [0.6, 0.4], [0.5, 0.5]) returned w=1.0 on an impossible outcome,
    y=[0.5, 0] also returned w=1.0, and a p_model containing 1.5
    returned w=0.55. Any unlawful value now raises ValueError
    (INVALID_INPUT) naming the argument, index and value; lawful
    histories are bit-identical.
    """
    # SF-BL-11054: np.asarray(..., float) coerces string arrays just
    # like float() coerces string scalars, so screen the RAW arguments
    # for string-likes / string dtypes before the conversion below.
    for _sname, _raw in (("fit_shrink_weight y", y),
                         ("fit_shrink_weight p_model", p_model),
                         ("fit_shrink_weight p_prior", p_prior)):
        if isinstance(_raw, (str, bytes, bytearray)) \
                or np.asarray(_raw).dtype.kind in ("U", "S"):
            raise ValueError(
                f"INVALID_INPUT: {_sname}={_raw!r} is not a finite "
                f"number array (string inputs rejected; numeric "
                f"values are required)")
    y = np.asarray(y, float); p_model = np.asarray(p_model, float); p_prior = np.asarray(p_prior, float)
    for _name, _arr in (("fit_shrink_weight y", y), ("fit_shrink_weight p_model", p_model), ("fit_shrink_weight p_prior", p_prior)):
        if not np.all(np.isfinite(_arr)):
            raise ValueError(f"INVALID_INPUT: {_name} contains NaN/inf")
    # SF-BL-8869: refuse an empty or length-mismatched history rather
    # than returning a numeric w from no / broadcast data (see the
    # docstring note). Scalars are 0-d and land here too: a scalar is
    # not a history, and a scalar against longer arrays would
    # broadcast. A matched length-1 history (one resolved forecast in
    # each array) is valid and unaffected.
    for _name, _arr in (("fit_shrink_weight y", y), ("fit_shrink_weight p_model", p_model), ("fit_shrink_weight p_prior", p_prior)):
        if _arr.ndim != 1 or _arr.size == 0:
            raise ValueError(
                f"INVALID_INPUT: {_name} history is empty or not a 1-D "
                f"sequence (ndim={_arr.ndim}, size={_arr.size}); "
                f"refusing to return a numeric shrink weight from no "
                f"data (shrink layer inactive, no fitted w)")
    # SF-BL-9794: this equality guard also closes the singleton-y
    # broadcast — fit_shrink_weight([1.0],[0.9,0.1,0.9,0.1],
    # [0.5,0.5,0.5,0.5]) returned w=0.0 on the pristine engine by
    # broadcasting the one outcome over all four pairs; it must raise.
    # SF-BL-10773: the same guard closes the 1-vs-5 variant —
    # fit_shrink_weight([1.0],[0.9,0.8,0.9,0.85,0.95],[0.5]*5)
    # returned w=1.0 (full trust) on the pristine engine by
    # broadcasting the one outcome over all five pairs; it must raise.
    if not (len(y) == len(p_model) == len(p_prior)):
        raise ValueError(
            f"INVALID_INPUT: fit_shrink_weight length mismatch: "
            f"y len={len(y)}, p_model len={len(p_model)}, "
            f"p_prior len={len(p_prior)}; refusing to broadcast or "
            f"truncate a shrink-fit history")
    # SF-BL-11019: value-validity screen — the guards above check only
    # finiteness, shape and length, never whether the VALUES are lawful.
    # An outcome y is a resolution and must be exactly 0 or 1, and
    # p_model / p_prior are probabilities and must lie in [0, 1]
    # (boundaries inclusive). Pre-fix, fit_shrink_weight([2, 0],
    # [0.6, 0.4], [0.5, 0.5]) returned w=1.0 (full trust fitted on an
    # impossible outcome), y=[0.5, 0] also returned w=1.0, and a
    # p_model containing 1.5 returned w=0.55. The screen is
    # reject-only: it never transforms a lawful value, so lawful
    # histories stay bit-identical.
    _bad_y = np.flatnonzero((y != 0.0) & (y != 1.0))
    if _bad_y.size:
        _i = int(_bad_y[0])
        raise ValueError(
            f"INVALID_INPUT: fit_shrink_weight y[{_i}]={y[_i]!r} is "
            f"not a lawful outcome (outcomes must be 0 or 1); "
            f"refusing to fit a shrink weight on a corrupt history")
    for _name, _arr in (("fit_shrink_weight p_model", p_model),
                        ("fit_shrink_weight p_prior", p_prior)):
        _bad_p = np.flatnonzero((_arr < 0.0) | (_arr > 1.0))
        if _bad_p.size:
            _i = int(_bad_p[0])
            raise ValueError(
                f"INVALID_INPUT: {_name}[{_i}]={_arr[_i]!r} is "
                f"outside [0, 1] (probabilities must lie in [0, 1]); "
                f"refusing to fit a shrink weight on a corrupt "
                f"history")
    if grid is None:
        grid = np.linspace(0.0, 1.0, 41)
    else:
        # SF-BL-11501: type-validate the RAW grid BEFORE the float
        # conversion below (see the docstring note). The screen is
        # reject-only: it never transforms a lawful numeric entry,
        # so numeric grids stay bit-identical.
        if isinstance(grid, (str, bytes, bytearray)):
            raise ValueError(
                f"INVALID_INPUT: fit_shrink_weight grid={grid!r} is "
                f"not a finite number array (string inputs rejected; "
                f"a sequence of numeric grid values is required, "
                f"got {type(grid).__name__})")
        if isinstance(grid, (bool, np.bool_)):
            raise ValueError(
                f"INVALID_INPUT: fit_shrink_weight grid={grid!r} is "
                f"not a finite number array (bool inputs rejected; "
                f"a sequence of numeric grid values is required, "
                f"got {type(grid).__name__})")
        _grid_probe = None
        if isinstance(grid, np.ndarray):
            if grid.dtype == bool or grid.dtype.kind in ("U", "S"):
                raise ValueError(
                    f"INVALID_INPUT: fit_shrink_weight grid={grid!r} "
                    f"contains bool/string entries (bool and string "
                    f"grid entries rejected; numeric grid values are "
                    f"required, got dtype {grid.dtype})")
            _grid_probe = [grid.item()] if grid.ndim == 0 \
                else list(np.asarray(grid, dtype=object).ravel())
        elif isinstance(grid, (list, tuple)):
            _grid_probe = list(grid)
        if _grid_probe is not None:
            for _gi, _gv in enumerate(_grid_probe):
                if isinstance(_gv, (bool, np.bool_)):
                    raise ValueError(
                        f"INVALID_INPUT: fit_shrink_weight "
                        f"grid[{_gi}]={_gv!r} is not a finite number "
                        f"(bool grid entries rejected; a numeric type "
                        f"is required, got {type(_gv).__name__})")
                if isinstance(_gv, (str, bytes, bytearray)):
                    raise ValueError(
                        f"INVALID_INPUT: fit_shrink_weight "
                        f"grid[{_gi}]={_gv!r} is not a finite number "
                        f"(string grid entries rejected; a numeric "
                        f"type is required, got "
                        f"{type(_gv).__name__})")
        # SF-BL-8869: an empty / scalar / non-finite grid has no
        # candidate w to fit either -- pre-fix grid=[] returned the
        # same standing 0.0 and a NaN grid point could never win.
        try:
            grid = np.asarray(grid, float)
        except (TypeError, ValueError):
            raise ValueError(
                f"INVALID_INPUT: fit_shrink_weight grid={grid!r} is "
                f"not a finite number array") from None
        if grid.ndim != 1 or grid.size == 0 \
                or not np.all(np.isfinite(grid)):
            raise ValueError(
                f"INVALID_INPUT: fit_shrink_weight grid is empty, "
                f"not 1-D, or contains NaN/inf (ndim={grid.ndim}, "
                f"size={grid.size}); refusing to return a numeric "
                f"shrink weight with no candidate grid")
    best_w, best_b = 0.0, np.inf
    for w in grid:
        b = float(np.mean((y - (w * p_model + (1 - w) * p_prior)) ** 2))
        if b < best_b:
            best_b, best_w = b, float(w)
    return best_w


def skill_weight(brier_model: float, brier_prior: float) -> float:
    """Map validated Brier skill to a pool weight contribution.

    Positive skill only; a model at or below the prior gets weight 0.

    SF-BL-9781: both arguments are Brier scores, which lie in [0, 1]
    by construction (a mean of squared errors of probabilities
    against 0/1 outcomes). _require_finite screens NaN/inf only, so
    pre-fix an impossible negative model Brier was accepted:
    skill_weight(-0.10, 0.25) returned max(0, 1-(-0.4)) = 1.4, a pool
    weight 40% above the 1.0 a perfect model (Brier 0) earns, and a
    prior Brier above the theoretical 1.0 maximum was likewise
    accepted. Both arguments must now be finite AND in [0, 1]; an
    out-of-domain value raises ValueError (INVALID_INPUT) naming the
    argument -- it is never clamped into range. Valid-grid behaviour
    is unchanged, including brier_prior == 0 -> 0.0 below.

    SF-BL-10980: neither argument may be a bool. bool subclasses int,
    so pre-fix float(False)==0.0 passed the finite and domain screens:
    skill_weight(False, 0.5) returned max(0, 1-0/0.5) = 1.0 -- a False
    Brier became a perfect validated skill worth the maximum pool
    weight, above even what the SF-BL-9781 domain guard allows a real
    score to earn legitimately through this path. A bool argument now
    raises ValueError (INVALID_INPUT) naming the argument, before any
    float() coercion. Int 1/0 are NOT bools and stay lawful.
    """
    _require_not_bool("skill_weight brier_model", brier_model)
    _require_not_bool("skill_weight brier_prior", brier_prior)
    brier_model = _require_finite("skill_weight brier_model", brier_model)
    brier_prior = _require_finite("skill_weight brier_prior", brier_prior)
    for _name, _v in (("skill_weight brier_model", brier_model),
                      ("skill_weight brier_prior", brier_prior)):
        if not 0.0 <= _v <= 1.0:
            raise ValueError(
                f"INVALID_INPUT: {_name}={_v!r} is outside the Brier "
                f"domain [0, 1] (a Brier score cannot be negative or "
                f"exceed 1; refusing to derive a pool weight from an "
                f"impossible score)")
    if brier_prior <= 0:
        return 0.0
    return max(0.0, 1.0 - brier_model / brier_prior)


def log_odds_pool(probs, weights=None, extremization: float = 1.0) -> float:
    """Weighted log-odds (geometric) pool with optional extremization.

    Raises ValueError if ``weights`` is given with a length different from
    ``probs`` — a mismatched weight must never silently broadcast / truncate
    away a component (SF-BL-8868).

    Raises ValueError if ``extremization`` is outside (0, 1.40]
    (SF-BL-8900) — see the guard below.

    Raises ValueError if any supplied ``weights`` entry is negative,
    if the supplied weights sum to <= 0 (an all-zero vector), or if
    any entry is NaN/inf (SF-BL-9512) — see the guards below.

    Raises ValueError if any ``probs`` member is a finite number
    outside [0, 1] (SF-BL-10936) — see the guard below.

    Raises ValueError if the ``probs`` container or any ``probs``
    member is a bool (SF-BL-10980) — see the guard below.

    Raises ValueError if the ``weights`` container or any
    ``weights`` entry is a bool (SF-BL-11022) — see the guard below.
    """
    # SF-BL-10980: reject a bool probs container and any bool probs
    # member BEFORE any float() coercion. bool subclasses int, so
    # pre-fix float(True)==1.0 passed the SF-BL-9510 finite screen and
    # the SF-BL-10936 range screen and was clipped to 0.999999:
    # log_odds_pool([True, 0.5]) returned 0.999001 -- a JSON/config
    # flag leaking into the pool became near-certainty. The screen
    # runs on the RAW members, on EVERY supplied element BEFORE the
    # finite/range screens and BEFORE weights/extremization
    # validation, so a bool member cannot be silently rescued by a
    # zero weight. Scope is probs only: weights are screened
    # separately below (SF-BL-11022) and extremization is a separate
    # dimension, unscreened here. Int 1/0 members are NOT bools and
    # stay lawful.
    _require_not_bool("log_odds_pool probs", probs)
    # SF-BL-11054: a string-like probs container is one string input,
    # not a sequence of one-character probabilities -- name it whole
    # instead of iterating its characters into the element screen.
    if isinstance(probs, (str, bytes, bytearray)):
        raise ValueError(
            f"INVALID_INPUT: log_odds_pool probs={probs!r} is not a "
            f"finite number (string inputs rejected; a sequence of "
            f"numeric values is required, got {type(probs).__name__})")
    _probs_raw = list(probs) if probs is not None else None
    if _probs_raw is not None:
        for _i, _v in enumerate(_probs_raw):
            _require_not_bool(f"log_odds_pool probs[{_i}]", _v)
    _probs = _require_finite_seq("log_odds_pool probs", _probs_raw) if _probs_raw is not None else None
    if _probs is None or len(_probs) == 0:
        raise ValueError("INVALID_INPUT: log_odds_pool probs is empty/None; refusing to pool to NaN")
    # SF-BL-10936: reject any finite probs member outside [0, 1].
    # The SF-BL-9510 screen above checks finiteness only, so pre-fix
    # a finite impossible member passed it and was clipped by the
    # np.clip below to 0.999999 / 1e-06: log_odds_pool([1.5, 0.5])
    # returned 0.999001 -- a percent-unit / corrupt component pooled
    # as near-certainty. forecast() guards its own components
    # (SF-BL-10881), but this function is re-exported in
    # superforecaster/__init__.py, so any direct / non-forecast
    # consumer of the primitive bypassed that guard entirely. Raise
    # INVALID_INPUT naming the argument and element, never clip into
    # range. The check runs on EVERY supplied element BEFORE the
    # extremization / weights validation and BEFORE the clip, so a
    # corrupt member cannot be silently rescued by a zero weight
    # ([1.5, 0.5] with weights [0.0, 1.0] must raise, not return
    # 0.5). Endpoints 0.0 and 1.0 are lawful probabilities in [0, 1]
    # and keep their established _clip-to-eps behaviour, mirroring
    # the SF-BL-9890 / SF-BL-10881 guards. Scope is this function's
    # probs only: calibration.py's and agents.py's own log_odds_pool
    # functions are separate, unaffected. forecast()'s internal
    # calls pass only _clip'd values in [_EPS, 1-_EPS], so this
    # guard cannot fire on that path.
    for _i, _v in enumerate(_probs):
        if not 0.0 <= _v <= 1.0:
            raise ValueError(
                f"INVALID_INPUT: log_odds_pool probs[{_i}]={_v!r} "
                f"is outside [0, 1] (a component probability must "
                f"lie in [0, 1]; refusing to clip an out-of-range "
                f"component to near-certainty and pool it)")
    # SF-BL-11054: use the VALIDATED floats from here on -- the raw
    # extremization / weights previously flowed into the arithmetic
    # below, so a string extremization died with a bare TypeError
    # even though it had passed the (coercing) screen.
    _extremization = _require_finite("log_odds_pool extremization", extremization)
    # SF-BL-8900: reject an extremization outside (0, 1.40] here, in the
    # function itself -- not only in Superforecaster.__init__ ([1.00,
    # 1.40]), which this public/exported function bypasses entirely
    # (superforecaster/__init__ re-exports log_odds_pool). Pre-fix any
    # finite factor was accepted: -1.0 inverted a 0.8 pool to 0.2
    # (logit sign flip), 0.0 collapsed it to 0.5, and 2.0 silently
    # exceeded the 1.40 production cap (finding 4). The standalone
    # floor is 0 (exclusive), not the constructor's 1.00: a factor in
    # (0, 1) is legitimate de-extremization (shrink toward 0.5), while
    # <= 0 inverts/collapses and > 1.40 exceeds the cap. Raise, never
    # clip -- clipping would manufacture a pooled number the caller
    # did not ask for. calibration.py's own log_odds_pool (a research
    # fitter, a in (0.3, 4.0)) is a separate function, unaffected.
    if not (0.0 < _extremization <= 1.40):
        raise ValueError(
            f"INVALID_INPUT: log_odds_pool extremization={extremization!r} "
            f"is outside (0, 1.40]; refusing to pool with a factor that "
            f"inverts (<=0), collapses (0) or exceeds the 1.40 "
            f"extremization cap")
    _weights = None
    if weights is not None:
        # SF-BL-11022: reject a bool weights container and any bool
        # weights entry BEFORE any float() coercion. bool subclasses
        # int, so pre-fix _require_finite float()ed True/False to
        # 1.0/0.0 and the vector was renormalised as a 1/0 member
        # mask: log_odds_pool([0.7, 0.3], weights=[True, False])
        # returned exactly 0.7 -- the production pool silently
        # collapsed to its first component and every other component
        # was discarded. A scalar bool container died with a bare
        # TypeError ('bool' object is not iterable) instead of the
        # named INVALID_INPUT. The screen runs on the RAW entries,
        # on EVERY supplied element BEFORE the finiteness, length,
        # negative and all-zero handling below, so a bool weight
        # cannot be silently rescued by its numeric value (a False
        # weight is not a lawful zero/drop) or hidden behind the
        # length-mismatch error. Scope is this function's weights
        # only: extremization is a separate dimension, unscreened
        # here; calibration.py's and agents.py's own log_odds_pool
        # functions are separate, unaffected. Int 1/0 weights are
        # NOT bools and stay lawful. forecast()'s internal calls
        # pass only validated floats > 0 (or [1.0]), so this screen
        # cannot fire on that path.
        _require_not_bool("log_odds_pool weights", weights)
        if isinstance(weights, np.ndarray) and weights.dtype == bool:
            raise ValueError(
                f"INVALID_INPUT: log_odds_pool weights={weights!r} "
                f"is not a finite number (bool inputs rejected; a "
                f"numeric type is required, got bool-dtype ndarray)")
        _weights_raw = list(weights)
        for _i, _v in enumerate(_weights_raw):
            _require_not_bool(f"log_odds_pool weights[{_i}]", _v)
        _weights = _require_finite_seq("log_odds_pool weights", _weights_raw)
    p = np.clip(np.asarray(_probs, float), _EPS, 1 - _EPS)
    if weights is None:
        w = np.full(len(p), 1.0 / len(p))
    else:
        w = np.asarray(_weights, float)
        if len(w) != len(p):
            raise ValueError(
                f"log_odds_pool probs/weights length mismatch: "
                f"{len(p)} probs vs {len(w)} weights; refusing to pool a "
                f"silently truncated/broadcast component set")
        # SF-BL-9512: reject negative weights and a non-positive total
        # instead of renormalising them as-is / silently substituting
        # equal weights. Pre-fix, weights=[2.0, -1.0] on [0.3, 0.7]
        # returned 0.073 (the -1.0 member was SUBTRACTED, not
        # weighted), weights=[-0.5, 1.0] returned 0.927, and any
        # vector with w.sum() <= 0 -- all-negative or all-zero --
        # silently fell back to the equal-weight pool, manufacturing
        # weights the caller never supplied. NaN/inf entries already
        # raise above via the SF-BL-9510 _require_finite_seq screen
        # (a NaN sum made the old `w.sum() > 0` test False, which is
        # how NaN reached that fallback before 9510). Raise, never
        # clip-to-zero and never substitute -- both would manufacture
        # a pooled number the caller did not ask for. This mirrors
        # calibration.py's _norm_weights contract ("weights must be
        # non-negative, length M, with positive sum"). Individual
        # zero weights stay lawful while the total is positive
        # ([0.0, 1.0] drops a member by value, by design, exactly as
        # forecast() drops w <= 0 components); -0.0 counts as zero.
        # forecast()'s internal calls pass only weights filtered
        # > 0 (or [1.0]), so this guard cannot fire on that path.
        if np.any(w < 0):
            raise ValueError(
                f"INVALID_INPUT: log_odds_pool weights={_weights!r} "
                f"contain negative value(s) {w[w < 0].tolist()}; "
                f"weights must be non-negative -- refusing to pool, "
                f"since a negative weight subtracts its member "
                f"instead of weighting it")
        if w.sum() <= 0:
            raise ValueError(
                f"INVALID_INPUT: log_odds_pool weights={_weights!r} "
                f"sum to {float(w.sum())}; weights must have a "
                f"positive sum -- refusing to silently substitute "
                f"equal weights for an all-zero weight vector")
        w = w / w.sum()
    z = float(np.sum(w * logit(p)) * _extremization)
    _out = _clip(expit(z))
    return _require_finite("log_odds_pool result", _out)


class Superforecaster:
    """Production forecaster implementing the v1 evidence-based policy."""

    MARKET_ANCHOR_MIN = 0.85      # finding 2: liquid market price dominates
    EXTREMIZE_DEFAULT = 1.00      # finding 4: never extremize blind
    EXTREMIZE_CAP = 1.40
    EXTREMIZE_MIN_RESOLVED = 150

    def __init__(self, extremization: float = EXTREMIZE_DEFAULT):
        # SF-BL-11054: keep and use the validated float -- the raw
        # argument previously flowed into the range comparison, so a
        # string extremization died with a bare TypeError.
        extremization = _require_finite("extremization", extremization)
        if not (1.0 <= extremization <= self.EXTREMIZE_CAP):
            raise ValueError("extremization must be in [1.00, 1.40] and is "
                             "learned per-domain only after >=150 resolved questions")
        self.extremization = float(extremization)

    def forecast(self, prior: float, model_probs=None, model_skills=None,
                 agent_probs=None, agent_weights=None,
                 market_price: float | None = None) -> dict:
        """Combine a prior, quantitative models and agent forecasts.

        prior        base rate / climatology known at forecast time.
                     Must be a finite probability in [0, 1] (SF-BL-9890:
                     a finite prior outside [0, 1] raises INVALID_INPUT;
                     it is never clipped to the clip bound and shipped).
                     No argument may be a bool (SF-BL-10921: prior,
                     market_price, and every element of model_probs,
                     model_skills, agent_probs and agent_weights are
                     screened before any float() coercion; a bool raises
                     INVALID_INPUT instead of becoming 1.0/0.0).
        model_probs  raw model probabilities; they are shrunk toward the
                     anchor here by validated skill (findings 1/5).
                     Each must be a finite probability in [0, 1]
                     (SF-BL-10881: a finite value outside [0, 1] raises
                     INVALID_INPUT; it is never clipped to the clip
                     bound and pooled as near-certainty).
        agent_probs  each must likewise be a finite probability in
                     [0, 1] (SF-BL-10881, same guard).
        model_skills validated Brier skill of each model vs the prior
                     (1 - brier_model/brier_prior); <= 0 models are dropped.
        agent_probs / agent_weights  independent agent forecasts and their
                     track-record weights (see agents.py).
        market_price if a liquid prediction-market price exists it replaces
                     the prior and is anchored (finding 2). Must be a
                     finite probability in [0, 1] (SF-BL-10981: a finite
                     market_price outside [0, 1] raises INVALID_INPUT;
                     it is never clipped to the clip bound and anchored
                     as near-certainty).

        Length contract (SF-BL-8868): if ``model_skills`` is supplied it
        MUST have exactly ``len(model_probs)`` entries, and if
        ``agent_weights`` is supplied it MUST have exactly
        ``len(agent_probs)`` entries. Any mismatch — in either direction,
        including skills/weights supplied with no probs at all — raises
        ValueError. Previously ``zip`` silently truncated to the shorter
        list, so production pools lost validated components with no
        warning. Omitting skills/weights entirely (None) keeps the
        documented defaults (skill 0.0 / weight 1.0 per component).
        Intentional drops are unaffected: a model with skill <= 0 and an
        agent with weight <= 0 are still dropped *by value*, by design.
        """
        # --- SF-BL-10921: reject bool inputs BEFORE any float()
        # coercion. _require_finite float()s bools to 1.0/0.0, so
        # pre-fix prior=True shipped p=0.999999 under the note 'prior
        # shipped unchanged', market_price=False shipped p=1e-06, and
        # agent_probs=[True] entered the pool as a 0.999999 component.
        # The screen runs on the RAW arguments (the validated floats
        # below have already lost the bool type), on every element
        # BEFORE the by-value drops further down (skill <= 0 /
        # weight <= 0), so a corrupt component cannot be silently
        # rescued by being dropped. Scope is forecast()'s six
        # arguments only (see _require_not_bool). String-like sequence
        # containers are left untouched here so the SF-BL-11054 screen
        # in _require_finite_seq still names them whole.
        _require_not_bool("prior", prior)
        if market_price is not None:
            _require_not_bool("market_price", market_price)
        _seq_args = {"model_probs": model_probs,
                     "model_skills": model_skills,
                     "agent_probs": agent_probs,
                     "agent_weights": agent_weights}
        for _seq_name, _seq_val in _seq_args.items():
            if _seq_val is None \
                    or isinstance(_seq_val, (str, bytes, bytearray)):
                continue
            _require_not_bool(_seq_name, _seq_val)
            try:
                _seq_list = list(_seq_val)
            except TypeError:
                continue  # non-iterable: _require_finite_seq raises as before
            for _i, _v in enumerate(_seq_list):
                _require_not_bool(f"{_seq_name}[{_i}]", _v)
            _seq_args[_seq_name] = _seq_list
        model_probs = _seq_args["model_probs"]
        model_skills = _seq_args["model_skills"]
        agent_probs = _seq_args["agent_probs"]
        agent_weights = _seq_args["agent_weights"]
        # --- SF-BL-9510: reject non-finite inputs before any clipping /
        # pooling. np.clip leaves NaN untouched, so a NaN prior, market
        # price, component prob, skill, agent weight or (via shrink) a
        # NaN shrink weight previously shipped as {'p': nan, ...}.
        # --- SF-BL-11054: keep the VALIDATED floats / float-lists and
        # use them everywhere below. The validated values were
        # previously discarded and the raw arguments re-read, so a
        # string prior passed this (coercing) screen and then died
        # inside _clip with a bare UFuncTypeError, a string model
        # skill was silently coerced into a pooled forecast, and a
        # string agent weight died with a bare TypeError.
        prior = _require_finite("prior", prior)
        # --- SF-BL-9890: reject a finite prior outside [0, 1]. The
        # SF-BL-9510 screen above checks finiteness only, so pre-fix a
        # finite impossible prior (1.5, 2.0, -0.5) passed it, was
        # clipped by _clip below to 0.999999 / 1e-06, and shipped on
        # the no-components path as near-certainty under the note
        # 'prior shipped unchanged' -- the shipped value was neither
        # the caller's prior nor a probability the caller supplied.
        # A prior is a probability (base rate / climatology): raise
        # INVALID_INPUT naming the argument, never clip into range
        # (mirrors the SF-BL-9781 Brier-domain guard in skill_weight).
        # Endpoints 0.0 and 1.0 are lawful and keep their established
        # _clip-to-eps behaviour. Scope is the prior argument only:
        # market_price is SF-BL-10981 and the component probs are
        # SF-BL-10881, both separate tasks.
        if not 0.0 <= prior <= 1.0:
            raise ValueError(
                f"INVALID_INPUT: prior={prior!r} is outside [0, 1] "
                f"(a prior is a probability; refusing to clip an "
                f"impossible prior to near-certainty and ship it as "
                f"unchanged)")
        if market_price is not None:
            market_price = _require_finite("market_price", market_price)
            # --- SF-BL-10981: reject a finite market_price outside
            # [0, 1]. The SF-BL-9510 screen above checks finiteness
            # only, so pre-fix a finite impossible market price (1.5,
            # -0.5) passed it, replaced the prior as the anchor
            # (finding 2, anchor weight >= 0.85), and was clipped by
            # _clip below to 0.999999 / 1e-06: market_price=1.5 with
            # agents [0.70, 0.66] returned anchor 0.999999 and
            # p=0.9522, and market_price=-0.5 shipped p=1e-06 on the
            # no-components path under the note 'prior shipped
            # unchanged'. SF-BL-9890 fixtures only the prior argument,
            # but market_price passes through the same _clip as the
            # dominant anchor -- the highest-leverage single input in
            # the engine. A market price is a probability: raise
            # INVALID_INPUT naming the argument, never clip into
            # range (mirrors the SF-BL-9890 prior guard above).
            # Endpoints 0.0 and 1.0 are lawful and keep their
            # established _clip-to-eps behaviour. Scope is the
            # market_price argument only: prior is SF-BL-9890
            # (guarded above) and the component probs are
            # SF-BL-10881 (guarded below).
            if not 0.0 <= market_price <= 1.0:
                raise ValueError(
                    f"INVALID_INPUT: market_price={market_price!r} "
                    f"is outside [0, 1] (a market price is a "
                    f"probability; refusing to clip an impossible "
                    f"market price to near-certainty and anchor the "
                    f"forecast on it)")
        model_probs = _require_finite_seq("model_probs", model_probs)
        model_skills = _require_finite_seq("model_skills", model_skills)
        agent_probs = _require_finite_seq("agent_probs", agent_probs)
        agent_weights = _require_finite_seq("agent_weights", agent_weights)
        # --- SF-BL-10881: reject any finite model/agent component
        # probability outside [0, 1]. The SF-BL-9510 screen above
        # checks finiteness only, so pre-fix a finite impossible
        # component passed it and was clipped by _clip in the loops
        # below to 0.999999 / 1e-06: agent_probs=[82.0] (a percent-unit
        # forecast) entered the pool as a 0.999999 component -- the
        # strongest component the production pool can receive -- and
        # model_probs=[1.5] with skill 0.5 shipped p=0.75. This is the
        # same failure AgentForecast.__post_init__ was fixed to reject
        # in agents.py (SF-BL-10769) and the same defect class as the
        # SF-BL-9890 prior guard above, unguarded on the component
        # paths. Raise INVALID_INPUT naming the argument and element,
        # never clip into range. The check runs on EVERY supplied
        # element BEFORE the by-value drops below (skill <= 0 /
        # weight <= 0), so a corrupt component cannot be silently
        # rescued by being dropped. Endpoints 0.0 and 1.0 are lawful
        # probabilities in [0, 1] and keep their established
        # _clip-to-eps behaviour, mirroring SF-BL-9890. Scope is the
        # two component sequences only: prior is SF-BL-9890 (guarded
        # above) and market_price is SF-BL-10981 (separate task).
        for _seq_name, _seq in (("model_probs", model_probs),
                                ("agent_probs", agent_probs)):
            if _seq is not None:
                for _i, _v in enumerate(_seq):
                    if not 0.0 <= _v <= 1.0:
                        raise ValueError(
                            f"INVALID_INPUT: {_seq_name}[{_i}]={_v!r} "
                            f"is outside [0, 1] (a component probability "
                            f"must lie in [0, 1]; refusing to clip an "
                            f"out-of-range component to near-certainty "
                            f"and pool it)")

        anchor = _clip(market_price) if market_price is not None else _clip(prior)
        _require_finite("anchor", anchor)
        components, weights = [], []

        # --- SF-BL-8868: fail loudly on any length mismatch, never zip-drop.
        if model_probs is None and model_skills is not None and len(model_skills) > 0:
            raise ValueError(
                f"model_probs/model_skills length mismatch: 0 probs vs "
                f"{len(model_skills)} skills; refusing to silently drop skills")
        if agent_probs is None and agent_weights is not None and len(agent_weights) > 0:
            raise ValueError(
                f"agent_probs/agent_weights length mismatch: 0 probs vs "
                f"{len(agent_weights)} weights; refusing to silently drop weights")
        if model_probs is not None and model_skills is not None \
                and len(model_skills) != len(model_probs):
            raise ValueError(
                f"model_probs/model_skills length mismatch: "
                f"{len(model_probs)} probs vs {len(model_skills)} skills; "
                f"refusing to silently drop "
                f"{abs(len(model_probs) - len(model_skills))} component(s)")
        if agent_probs is not None and agent_weights is not None \
                and len(agent_weights) != len(agent_probs):
            raise ValueError(
                f"agent_probs/agent_weights length mismatch: "
                f"{len(agent_probs)} probs vs {len(agent_weights)} weights; "
                f"refusing to silently drop "
                f"{abs(len(agent_probs) - len(agent_weights))} component(s)")

        if model_probs is not None:
            skills = model_skills if model_skills is not None else [0.0] * len(model_probs)
            for p, s in zip(model_probs, skills, strict=True):
                w = skill_weight_from_skill(s)
                if w > 0:
                    # Finding 1: a model may only move the forecast from the
                    # anchor by its validated skill weight.
                    components.append(shrink(_clip(p), anchor, w)); weights.append(w)
        if agent_probs is not None:
            aw = agent_weights if agent_weights is not None else [1.0] * len(agent_probs)
            for p, w in zip(agent_probs, aw, strict=True):
                if w > 0:
                    components.append(_clip(p)); weights.append(float(w))

        if not components:
            return {"p": _require_finite("forecast p", anchor), "anchor": anchor, "n_components": 0,
                    "note": "no validated component; prior shipped unchanged"}

        pooled = log_odds_pool(components, weights, extremization=1.0)
        # Total deviation budget: with a market anchor, models+agents together
        # may only claim the non-anchor share unless every component agrees.
        if market_price is not None:
            pooled = shrink(pooled, anchor, 1.0 - self.MARKET_ANCHOR_MIN)
        p_final = log_odds_pool([pooled], [1.0], extremization=self.extremization)
        _require_finite("forecast pooled_components", pooled)
        _require_finite("forecast p", p_final)
        return {"p": p_final, "anchor": anchor, "pooled_components": pooled,
                "n_components": len(components),
                "extremization": self.extremization}


def skill_weight_from_skill(skill: float) -> float:
    """A model's validated Brier skill IS its max shrink weight (capped 0.5:
    no single unproven-in-production model may move a forecast more than
    halfway from the prior on its first validated edge)."""
    return float(np.clip(_require_finite("skill", skill), 0.0, 0.5))


if __name__ == "__main__":
    sf = Superforecaster()
    # Demo 1: the NVDA 21d case from results/markets_results.json — the raw
    # model said 0.817 while its validated skill was negative (-0.007).
    print("NVDA 21d, raw model 0.817, validated skill -0.007:")
    print(" ", sf.forecast(prior=0.55, model_probs=[0.817], model_skills=[-0.007]))
    # Demo 2: crypto crash-risk — the one validated edge (skill +0.024 BTC).
    print("BTC crash-risk, model 0.16 vs base rate 0.10, validated skill +0.024:")
    print(" ", sf.forecast(prior=0.10, model_probs=[0.16], model_skills=[0.024]))
    # Demo 3: prediction-market anchored question.
    print("Market-priced question, market 0.62, agents [0.70, 0.66]:")
    print(" ", sf.forecast(prior=0.50, market_price=0.62,
                           agent_probs=[0.70, 0.66], agent_weights=[0.16, 0.14]))
