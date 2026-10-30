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
The headline scoring rules (brier_score, log_score) VALIDATE their inputs:
any forecast outside [0,1] — including nan/inf — raises ValueError instead
of being silently clipped or scored (SF-BL-9484). The internal clip in
log_score applies only to lawful endpoints 0/1 after validation.
"""
from __future__ import annotations

import numbers

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

EPS = 1e-15
_CLIP = 1e-7  # clipping for logits: keeps logit finite (~ +/-16.1)


def _require_finite(x, name):
    """Raise ValueError if any element of ``x`` is non-finite (SF-BL-3596).

    NaN comparisons are all False, so negativity / sum / range checks
    alone silently pass NaN: ``_norm_weights`` accepted a NaN weight
    (``w < 0`` False, ``w.sum() <= 0`` False) and ``_as_p`` clipped
    +/-inf to a lawful endpoint while propagating NaN, even though
    ``_as_y`` rejected NaN outcomes. Every shared input helper now
    screens finiteness explicitly so NaN/Inf can never pool, fit, or
    score asymmetrically. Pure validation — never transforms ``x``.
    """
    if x.size and not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be finite; got non-finite value(s) in {name}")


def _has_bool(raw) -> bool:
    """True iff ``raw`` contains any boolean value (SF-BL-10938).

    Recurses through lists / tuples and object / bool-dtype ndarrays.
    A plain numeric ndarray cannot contain bools by construction, so
    it returns False without scanning elementwise.
    """
    if isinstance(raw, (bool, np.bool_)):
        return True
    if isinstance(raw, np.ndarray):
        if raw.dtype == bool:
            return True
        if raw.dtype == object:
            return any(_has_bool(v) for v in raw.ravel())
        return False
    if isinstance(raw, (list, tuple)):
        return any(_has_bool(v) for v in raw)
    return False


def _reject_bools(raw, name):
    """Raise a named ValueError if ``raw`` contains any bool
    (SF-BL-10938).

    This screen MUST run on the raw input, before any
    ``np.asarray(..., dtype=float)`` conversion: float conversion
    silently maps True -> 1.0 and False -> 0.0, so after conversion a
    boolean flag that leaked into a probability / outcome field is
    indistinguishable from a lawful endpoint and was scored as a
    perfect forecast (``brier_score([1], [True])`` returned 0.0,
    ``log_score([1], [True])`` returned ~1e-15 and
    ``evaluate([1], [True])`` returned a perfect scorecard). A bool
    is not a probability and not a 0/1 outcome — in Python it is an
    int subclass, which is exactly why every fleet bool screen
    (SF-BL-9722, SF-BL-10883, SF-BL-10921) rejects it explicitly.
    Pure validation — it never transforms ``raw`` — so lawful 0/1
    panels (int or float) are bit-identical to the pre-fix
    computation.
    """
    if _has_bool(raw):
        raise ValueError(
            f"{name} must not contain boolean values; got boolean "
            f"value(s) in {name} (a bool is not a valid {name} value) "
            f"(SF-BL-10938)"
        )


def _has_string(raw) -> bool:
    """True iff ``raw`` contains any string value (SF-BL-11052).

    Covers a bare ``str`` / ``bytes`` scalar (``np.str_`` /
    ``np.bytes_`` subclass them), lists / tuples at any nesting,
    ndarrays with a string dtype (kind ``U`` / ``S``), and
    object-dtype ndarrays holding a string leaf. A plain numeric
    ndarray cannot contain strings by construction, so it returns
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
    if isinstance(raw, (list, tuple)):
        return any(_has_string(v) for v in raw)
    return False


def _reject_strings(raw, name):
    """Raise a named ValueError if ``raw`` contains any string
    (SF-BL-11052).

    This screen MUST run on the raw input, before any
    ``np.asarray(..., dtype=float)`` conversion: float conversion
    silently parses numeric strings, so a forecast or outcome that
    arrived as text (a JSON / CSV pipeline bug) was scored as a
    genuine number (``brier_score([1], ['0.5'])`` returned 0.25,
    ``brier_score(['1'], [0.5])`` returned 0.25,
    ``log_score([1], ['0.5'])`` returned 0.6931 and
    ``evaluate([1, 0], ['0.5', '0.5'])`` returned a full
    scorecard). A numeric string is not a number — the panel's
    type is itself evidence of a corrupt feed, and scoring it
    hides that bug. This is the string dimension of the headline
    scorers; the bool dimension is SF-BL-10938
    (:func:`_reject_bools`). Pure validation — it never transforms
    ``raw`` — so lawful numeric panels (int or float) are
    bit-identical to the pre-fix computation.
    """
    if _has_string(raw):
        raise ValueError(
            f"{name} must not contain string values; got string "
            f"value(s) in {name} (a string is not a valid {name} value) "
            f"(SF-BL-11052)"
        )


def _reject_bool_pool_members(raw):
    """Raise a named ValueError if a pool panel contains any bool
    (SF-BL-10975).

    The pooling path never screened this case: :func:`_as_matrix`
    converts the raw panel with ``np.asarray(probs, dtype=float)``
    before any validation, so a boolean flag leaking into a member
    field was coerced True -> 1.0 / False -> 0.0 and became the most
    extreme lawful forecast in the pool (``linear_pool([[True,
    False]])`` returned ``[0.5]``; ``geometric_pool`` /
    ``log_odds_pool`` on ``[[True, 0.5]]`` returned 0.99968). The
    SF-BL-10934 range screen runs on the already-converted matrix,
    where 1.0/0.0 are lawful endpoints, and the SF-BL-10938 bool
    screen covered only the headline scorers — no screen claimed a
    bool *member* in the pools. This screen therefore runs on the
    raw ``probs`` input, inside :func:`_as_matrix` before the float
    conversion, so ``linear_pool``, ``geometric_pool``,
    ``log_odds_pool`` (and the ``extremized_pool`` alias and
    ``optimize_pool_weights``, which share ``_as_matrix``) all reject
    it. Pure validation — it never transforms ``raw`` — so lawful
    float/int panels are bit-identical to the pre-fix computation.
    Bool *weights* are untouched (SF-BL-10883's scope).
    """
    if _has_bool(raw):
        raise ValueError(
            "probs must not contain boolean values; got boolean "
            "value(s) in probs (a bool is not a valid probs member) "
            "(SF-BL-10975)"
        )

def _reject_string_pool_members(raw):
    """Raise a named ValueError if a pool panel contains any string
    (SF-BL-11125).

    The pooling path never screened this case for the ``probs``
    argument: :func:`_as_matrix` converts the raw panel with
    ``np.asarray(probs, dtype=float)`` before any validation, so a
    member forecast serialised as text was parsed as the number it
    spells and became a lawful forecast in the pool
    (``linear_pool([['0.7', '0.3']])``, ``geometric_pool`` and
    ``log_odds_pool`` on the same panel all returned ``[0.5]``,
    bit-identical to the numeric ``[[0.7, 0.3]]`` control). The
    SF-BL-10934 range screen and the SF-BL-3596 finiteness screen
    run on the already-converted matrix, where ``'0.7'`` has
    already become 0.7 — no screen claimed a string *member* in
    the pools. This is the ``probs``/member argument, distinct
    from SF-BL-11090 (string ``weights``, screened in
    :func:`_norm_weights`) and from SF-BL-10975 (bool members,
    screened alongside this one). This screen therefore runs on
    the raw ``probs`` input, inside :func:`_as_matrix` before the
    float conversion, so ``linear_pool``, ``geometric_pool``,
    ``log_odds_pool`` (and the ``extremized_pool`` alias and
    ``optimize_pool_weights``, which share ``_as_matrix``) all
    reject it. Detection is :func:`_has_string` (Python ``str`` /
    ``bytes``, ``np.str_`` / ``np.bytes_`` leaves, string-dtype
    ndarrays of kind ``U`` / ``S``, object-dtype ndarrays and
    lists/tuples at any nesting). Pure validation — it never
    transforms ``raw`` — so lawful float/int panels are
    bit-identical to the pre-fix computation. String *weights*
    are untouched here (SF-BL-11090's scope).
    """
    if _has_string(raw):
        raise ValueError(
            "probs must not contain string values; got string "
            "value(s) in probs (a string is not a valid probs member) "
            "(SF-BL-11125)"
        )



def _reject_bool_predict_input(raw):
    """Raise a named ValueError if a calibrator ``predict`` input
    contains any bool (SF-BL-11051).

    All four calibrator ``predict`` paths converted their input with
    ``np.asarray(p, dtype=float)`` BEFORE any validation, so the
    SF-BL-9719/9720/9751/9752 range screens (via
    :func:`_validate_p_in_range`) only ever saw the already-converted
    1.0 / 0.0 endpoints: ``PlattScaler.predict(True)`` returned
    0.9252, ``BetaCalibrator.predict(False)`` returned 3.5e-23,
    ``TemperatureScaler.predict(True)`` returned 0.9292 and
    ``IsotonicCalibrator.predict(True/False)`` returned 1.0/0.0 on a
    40-row noisy fit — a boolean flag recalibrated as a lawful
    endpoint forecast. A bool is numerically in-range, so it evades
    every existing screen; it differs from the NaN / finite
    out-of-range predict cases in input dimension (type, not value).
    This screen therefore runs on the RAW ``p`` argument, as the
    first statement of every calibrator ``predict`` (and its
    ``transform`` alias), before any float conversion. Detection is
    :func:`_has_bool` (Python ``bool``, ``np.bool_``, bool-dtype and
    object ndarrays, lists/tuples at any nesting). Pure validation
    — it never transforms ``raw`` — so lawful float/int in-range
    predictions are bit-identical to the pre-fix computation.
    """
    if _has_bool(raw):
        raise ValueError(
            "p must not contain boolean values; got boolean "
            "value(s) in p (a bool is not a valid p value) "
            "(SF-BL-11051)"
        )


def _reject_string_predict_input(raw):
    """Raise a named ValueError if a calibrator ``predict`` input
    contains any string (SF-BL-11463).

    All four calibrator ``predict`` paths converted their input with
    ``np.asarray(p, dtype=float)`` / ``_as_p`` BEFORE any validation,
    so the SF-BL-9719/9720/9751/9752 range screens only ever saw the
    already-converted number: fitted ``predict(['0.55'])`` returned
    values bit-identical to ``predict([0.55])`` for PlattScaler,
    BetaCalibrator, TemperatureScaler and IsotonicCalibrator (task
    live repro: 0.9999, 0.9998, 0.9822 and 1.0 respectively; kill
    criterion is TemperatureScaler.predict(['0.55']) returning
    0.9822 instead of raising). A numeric string is not a number —
    its dtype is evidence of a text/JSON/CSV pipeline bug — and it
    differs from SF-BL-11198 (finite out-of-range / NaN values) and
    SF-BL-11051 (bool predict inputs) in input dimension (type, not
    value). Isotonic's SF-BL-9719 frozen screen likewise validated
    range only after the same coercion. This screen therefore runs
    on the RAW ``p`` argument, as an early statement of every
    calibrator ``predict`` (and its ``transform`` alias), before
    any float conversion, before Temperature's intercept-only early
    return, and before Isotonic's unfitted RuntimeError. Detection
    is :func:`_has_string` (Python ``str``/``bytes``, ``np.str_`` /
    ``np.bytes_``, string-dtype ndarrays kind ``U``/``S``, object
    ndarrays and lists/tuples at any nesting). Pure validation —
    it never transforms ``raw`` — so lawful float/int in-range
    predictions are bit-identical to the pre-fix computation.
    Explicitly NOT a causal-Platt quant-path proposal: this is a
    type-validation guard on the calibrator classes only.
    """
    if _has_string(raw):
        raise ValueError(
            "p must not contain string values; got string "
            "value(s) in p (a string is not a valid p value) "
            "(SF-BL-11463)"
        )


def _has_bare_object(raw) -> bool:
    """True iff ``raw`` contains a bare ``object()`` leaf
    (SF-BL-11395).

    A bare ``object()`` has no numeric value at all; in an
    object-dtype panel it is the non-string form of the
    object-dtype element the prereg names. Recurses through
    lists / tuples and object-dtype ndarrays exactly like
    :func:`_has_string`. Numeric object panels (float/int leaves
    in an object-dtype array) contain no bare object and stay
    lawful, per fleet precedent.
    """
    if type(raw) is object:
        return True
    if isinstance(raw, np.ndarray):
        if raw.dtype == object:
            return any(_has_bare_object(v) for v in raw.ravel())
        return False
    if isinstance(raw, (list, tuple)):
        return any(_has_bare_object(v) for v in raw)
    return False


def _reject_string_fit_input(raw, name):
    """Raise a named ValueError if a calibrator ``fit`` input
    contains any string or object-dtype element (SF-BL-11395).

    All four calibrator ``fit`` paths converted ``p`` with
    ``np.asarray(p, dtype=float)`` (directly, or via
    :func:`_as_p`) BEFORE any type check, so a forecast that
    arrived as text was parsed as the number it spells and fitted
    bit-identically to the numeric panel
    (``PlattScaler().fit([1, 0, 1, 0, 1, 0], ['0.9', '0.2',
    '0.8', '0.3', '0.7', '0.4'])`` returned ``A_=17.7628``,
    ``B_=-3.7957``; BetaCalibrator ``a_=17.2495``,
    ``b_=-16.3920``; TemperatureScaler ``T_=0.05``;
    IsotonicCalibrator knots identical). SF-BL-11052 screened
    string panels only for the headline scorers and ``evaluate``
    (and, via the shared :func:`_as_y`, for a non-pandas string
    ``y``), and SF-BL-11024 fixtured only bool ``fit`` panels —
    the ``fit`` string dimension, above all the ``p`` argument
    (``_as_p`` has no string screen), was unclaimed. A numeric
    string is not a number — its type is itself evidence of a
    corrupt JSON/CSV feed, and fitting it hides that bug.
    This screen therefore runs on the RAW ``y`` and ``p``
    arguments, as the first statements of every calibrator
    ``fit`` (hence also ``fit_calibrator``), before any float
    conversion, finiteness/range screen, parameter
    re-validation, or intercept-only early return, so a rejected
    fit leaves the calibrator unfitted. Detection is
    :func:`_has_string` (Python ``str``/``bytes``,
    ``np.str_``/``np.bytes_`` leaves, string-dtype ndarrays
    kind ``U``/``S``, object-dtype ndarrays holding a string
    leaf, lists/tuples at any nesting) plus
    :func:`_has_string_in_pandas_panel` (pandas ``Series`` /
    ``Index`` / ``Categorical`` string panels, which
    ``_has_string`` cannot see) plus :func:`_has_bare_object`
    (bare ``object()`` leaves in any container, including
    object-dtype arrays). Pure validation — it never
    transforms ``raw`` — so lawful panels are bit-identical:
    float/int panels, numpy numeric arrays, tuples, numeric
    pandas panels, and object-dtype arrays carrying only
    numeric leaves all pass untouched.
    """
    has_str = _has_string(raw) or _has_bare_object(raw)
    if not has_str:
        try:
            has_str = _has_string_in_pandas_panel(raw)
        except Exception:  # noqa: BLE001 — helper is defined later in this module; never break lawful panels
            has_str = False
    if has_str:
        raise ValueError(
            f"{name} must not contain string values; got string "
            f"or object-dtype value(s) in {name} (a string is not "
            f"a valid {name} value) (SF-BL-11395)"
        )


def _has_bare_object_in_pandas_panel(raw) -> bool:
    """True iff a pandas panel contains a bare ``object()`` leaf
    (SF-BL-11424).

    Mirror of :func:`_has_string_in_pandas_panel` for the
    bare-object dimension: :func:`_has_bare_object` returns False
    for any non-list/tuple/ndarray container, so a bare
    ``object()`` carried in a pandas ``Series`` / ``Index`` /
    ``Categorical`` would otherwise reach float conversion and
    die with a bare ``TypeError`` naming nothing. Categories are
    scanned as well as observed values. Any inspection error
    returns False, leaving downstream behaviour untouched.
    """
    try:
        if not (type(raw).__module__ or "").startswith("pandas"):
            return False
        if type(raw).__name__ == "DataFrame":
            return False
        cats = getattr(raw, "categories", None)
        dtype = getattr(raw, "dtype", None)
        if cats is None and dtype is not None:
            cats = getattr(dtype, "categories", None)
        if cats is not None and any(_has_bare_object(c) for c in list(cats)):
            return True
        return any(_has_bare_object(v) for v in list(raw))
    except Exception:  # noqa: BLE001 — inspection must never break lawful panels
        return False


def _reject_binned_panel(raw, name):
    """Raise a named ValueError if a binned-calibration panel
    (``y`` or ``p`` for :func:`expected_calibration_error` /
    :func:`maximum_calibration_error`, funnelled through
    :func:`calibration_bins`) contains any string, object-dtype
    or bool element (SF-BL-11424).

    The binned entry points never type-validated their own
    panels: ``calibration_bins`` screened only a non-pandas bool
    ``p`` (SF-BL-10938) and ``_as_y`` screened only a non-pandas
    string/bool ``y`` (SF-BL-11052/10938), while ``p`` was
    converted with ``np.asarray(p, dtype=float)`` unscreened —
    so ``expected_calibration_error([1, 0], ['0.9', '0.1'])``
    returned 0.1 and ``maximum_calibration_error`` on the same
    panel returned 0.1, both bit-identical to the numeric
    controls, a pandas bool forecast panel returned ECE/MCE
    0.0 (perfect calibration) via float coercion, pandas string
    panels returned the numeric controls, and a bare
    ``object()`` leaf died with a bare ``TypeError`` naming
    neither argument. SF-BL-11052 fixtured string panels only
    for ``brier_score`` / ``log_score`` / ``evaluate`` and
    SF-BL-10938 only bool panels for those same headline
    scorers, so the binned ECE/MCE entry points' own panel type
    domain was unfixtured. This screen therefore runs on the
    RAW ``y`` and ``p``, explicitly in
    :func:`expected_calibration_error` and
    :func:`maximum_calibration_error` themselves and again in
    :func:`calibration_bins`, before any float conversion.
    Detection is :func:`_has_bool` +
    :func:`_has_bool_in_pandas_panel` (bool dimension, checked
    first, mirroring ``_as_y``'s screen order) and
    :func:`_has_string` + :func:`_has_string_in_pandas_panel` +
    :func:`_has_bare_object` +
    :func:`_has_bare_object_in_pandas_panel` (string /
    object-dtype dimension: ``str``/``bytes`` leaves,
    string-dtype ndarrays, object-dtype arrays holding a string
    or bare-object leaf, pandas string panels, bare
    ``object()`` leaves). The error messages carry the
    SF-BL-10938 / SF-BL-11052 tags alongside SF-BL-11424 so the
    earlier dimensions' named errors stay recognisable. Pure
    validation — it never transforms ``raw`` — so lawful
    panels are bit-identical: float/int panels, numpy numeric
    arrays, tuples, numeric pandas panels, and object-dtype
    arrays carrying only numeric leaves all pass untouched.
    """
    if _has_bool(raw) or _has_bool_in_pandas_panel(raw):
        raise ValueError(
            f"{name} must not contain boolean values; got boolean "
            f"value(s) in {name} (a bool is not a valid {name} value) "
            f"(SF-BL-10938, SF-BL-11424)"
        )
    if (_has_string(raw) or _has_bare_object(raw)
            or _has_string_in_pandas_panel(raw)
            or _has_bare_object_in_pandas_panel(raw)):
        raise ValueError(
            f"{name} must not contain string values; got string "
            f"or object-dtype value(s) in {name} (a string is not "
            f"a valid {name} value) (SF-BL-11052, SF-BL-11424)"
        )


def _as_y(y):
    _reject_bools(y, "y")  # SF-BL-10938: before float conversion coerces True->1.0
    _reject_strings(y, "y")  # SF-BL-11052: before float conversion parses '1'->1.0
    y = np.asarray(y, dtype=float).ravel()
    _require_finite(y, "y")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must be binary 0/1")
    return y


def _as_p(p, eps=_CLIP):
    p = np.asarray(p, dtype=float)
    _require_finite(p, "p")
    return np.clip(p, eps, 1 - eps)


def _require_same_length(y, p):
    """Raise ValueError if y and p do not have exactly the same length.

    SF-BL-8867: without this check a 1-element y silently broadcasts
    across the whole calibration set in Platt/Beta/Temperature fits
    (numpy ``mu - y`` / ``y * log(p)`` broadcasting) and Isotonic
    silently truncates a longer y via ``y[order]`` indexing.  Every
    public entry point that pairs outcomes with forecasts must call
    this before any computation so a mismatched set can never produce
    fitted parameters or a score.
    """
    ny = len(np.asarray(y).ravel())
    npr = len(np.asarray(p).ravel())
    if ny != npr:
        raise ValueError(
            f"len(y) ({ny}) != len(p) ({npr}): y and p must have the same length"
        )


def _validate_p_in_range(p):
    """Raise ValueError if any forecast lies outside [0, 1].

    SF-BL-9484: brier_score / log_score previously accepted impossible
    forecasts silently — log_score clipped p=1.5 and p=2.0 to 1-eps and
    returned a near-perfect ~1e-15 loss, brier_score scored p=-0.5 with
    no complaint, and nan/inf propagated. A scoring rule must never
    reward (or quietly score) a forecast that is not a probability.
    Non-finite values are rejected too: nan comparisons are all False,
    so an explicit finiteness check is required, not just range checks.
    The check is pure validation — it never transforms p — so in-range
    scores are bit-identical to the pre-fix computation.
    """
    _reject_bools(p, "p")  # SF-BL-10938: catches a raw bool-dtype panel
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        return
    if not np.all(np.isfinite(p)):
        raise ValueError(
            "p must be finite probabilities in [0, 1]; got non-finite value(s)"
        )
    lo, hi = float(p.min()), float(p.max())
    if lo < 0.0 or hi > 1.0:
        raise ValueError(
            f"p must lie in [0, 1]; got values outside that range "
            f"(min={lo!r}, max={hi!r})"
        )


def _validate_fit_p_range(p):
    """Raise ValueError naming ``p`` if any *finite* element of a
    calibrator ``fit`` input lies outside [0, 1] (SF-BL-9904).

    The four calibrator fit paths never screened this case: Platt,
    Beta and Temperature route ``p`` through :func:`_as_p`, whose clip
    silently mapped 1.5 to the 1-eps endpoint and -0.5 to eps, so
    impossible forecasts became extreme-logit training rows (Platt
    fit on a book containing 1.5 / -0.5 returned a clipped-logit
    artefact slope in the millions), and IsotonicCalibrator.fit —
    which bypasses ``_as_p`` — built PAVA knots at x=-0.5 and x=1.5.
    The predict-side screens (SF-BL-9719/9720/9751/9752) and the
    Isotonic fit NaN screen did not claim this fit-side finite
    out-of-range case.

    Non-finite handling is UNCHANGED by design: this helper ignores
    non-finite values entirely, and every fit() runs
    :func:`_require_finite` (SF-BL-3596, directly or via ``_as_p``)
    BEFORE calling it, so NaN / +/-inf — alone or mixed with a finite
    out-of-range value — still raise exactly the pre-existing
    finiteness ValueError. Pure validation — it never transforms
    ``p`` — so fitted parameters for in-range training sets are
    bit-identical to the pre-fix computation.
    """
    arr = np.asarray(p, dtype=float)
    if arr.size == 0:
        return
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return
    lo, hi = float(finite.min()), float(finite.max())
    if lo < 0.0 or hi > 1.0:
        raise ValueError(
            f"p must lie in [0, 1]; got finite value(s) outside that "
            f"range in p (min={lo!r}, max={hi!r}) (SF-BL-9904)"
        )


# --------------------------------------------------------------------------
# Scoring rules
# --------------------------------------------------------------------------

def brier_score(y, p) -> float:
    """Mean squared error between forecast probability and outcome.

    Raises
    ------
    ValueError
        If any element of ``p`` lies outside [0, 1] or is non-finite
        (SF-BL-9484). Forecasts must be probabilities; out-of-range
        values are never silently scored.
    ValueError
        If any element of ``p`` or ``y`` is a boolean (SF-BL-10938;
        see :func:`_reject_bools`). The screen runs on the raw
        inputs, before float conversion coerces True -> 1.0 /
        False -> 0.0 into lawful endpoints.
    ValueError
        If any element of ``p`` or ``y`` is a string (SF-BL-11052;
        see :func:`_reject_strings`). The screen runs on the raw
        inputs, before float conversion parses '0.5' -> 0.5 into a
        lawful forecast.
    """
    _reject_bools(p, "p")  # SF-BL-10938: raw screen, before conversion
    _reject_strings(p, "p")  # SF-BL-11052: raw screen, before conversion
    y = _as_y(y); p = np.asarray(p, dtype=float).ravel()
    _require_same_length(y, p)
    _validate_p_in_range(p)
    return float(np.mean((p - y) ** 2))


def _validate_eps(eps) -> float:
    """Validate ``log_score``'s ``eps`` clip bound (SF-BL-9659 value
    domain; SF-BL-11161 type screen).

    Returns the bound as a float. Pure validation — a valid bound is
    returned unchanged in value, so scores for lawful eps are
    bit-identical to the pre-fix computation.

    Raises
    ------
    ValueError
        Unless ``eps`` is a non-bool real number (SF-BL-11161) that
        is finite with ``0 < eps < 0.5`` (SF-BL-9659). Previously
        ``eps`` was never validated: ``eps=0`` returned NaN on a
        perfect forecast via ``log(0)``, ``eps=-0.1`` was silently
        accepted (the clip to ``[-0.1, 1.1]`` is a no-op), and
        ``eps=0.9`` clipped every forecast to the inverted bounds
        ``[0.9, 0.1]`` and returned 1.204 on ``p=[0.2, 0.8]`` — a
        mis-set eps silently destroyed or fabricated the log score
        that ``optimize_pool_weights`` optimises and ``evaluate``
        reports. ``eps >= 0.5`` is outside the domain by
        construction: the clip interval ``[eps, 1 - eps]`` is empty
        or inverted there. SF-BL-9659 funnelled conversion through
        ``float()``, which fixed the value cases and the named
        repros ``eps=True`` / ``eps='1e-7'`` / ``eps=None``, but it
        still ACCEPTED values that are not real numbers at all
        merely because ``float()`` can convert them —
        ``Decimal('1e-7')``, a 0-d ``np.array(1e-7)``, and any
        object defining only ``__float__`` — and ``np.bool_`` was
        rejected only incidentally, via its converted value. The
        type screen is now explicit: booleans (including
        ``np.bool_``) are rejected outright, and ``eps`` must be an
        instance of ``numbers.Real`` (int, float, ``Fraction``,
        real numpy scalars, real enum members); strings, bytes,
        None, complex, sequences, mappings, ndarrays of any ndim,
        ``Decimal`` and all other non-Real objects raise the same
        documented ValueError naming ``eps``.
    """
    msg = ("eps must be a non-bool real number, finite with "
           "0 < eps < 0.5 (SF-BL-9659, SF-BL-11161); "
           f"got eps={eps!r}")
    if isinstance(eps, (bool, np.bool_)):
        raise ValueError(msg)
    if not isinstance(eps, numbers.Real):
        raise ValueError(msg)
    try:
        e = float(eps)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(msg) from None
    if not np.isfinite(e) or e <= 0.0 or e >= 0.5:
        raise ValueError(msg)
    return e


def log_score(y, p, eps=EPS) -> float:
    """Mean negative log-likelihood (lower is better). Aka log loss.

    Raises
    ------
    ValueError
        If any element of ``p`` lies outside [0, 1] or is non-finite
        (SF-BL-9484). Validation happens BEFORE the internal clip: the
        clip to [eps, 1-eps] now applies only to lawful endpoint
        forecasts 0/1 (so they yield a large finite loss instead of
        inf), never to out-of-range values such as 1.5 or 2.0, which
        previously clipped to a near-perfect score.
    ValueError
        If ``eps`` is not a finite real scalar with
        ``0 < eps < 0.5`` (SF-BL-9659 value domain, SF-BL-11161 type screen; see :func:`_validate_eps`).
        Previously eps=0 returned NaN on a perfect forecast,
        eps=-0.1 was silently accepted, and eps=0.9 returned a
        meaningless score from inverted clip bounds.
    ValueError
        If any element of ``p`` or ``y`` is a string (SF-BL-11052;
        see :func:`_reject_strings`). The screen runs on the raw
        inputs, before float conversion parses '0.5' -> 0.5 into a
        lawful forecast. (The bool dimension is SF-BL-10938.)
    """
    eps = _validate_eps(eps)
    _reject_bools(p, "p")  # SF-BL-10938: raw screen, before conversion
    _reject_strings(p, "p")  # SF-BL-11052: raw screen, before conversion
    y = _as_y(y); p = np.asarray(p, dtype=float).ravel()
    _require_same_length(y, p)
    _validate_p_in_range(p)
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# Alias familiar to sklearn users
log_loss = log_score


def _validate_n_bins(n_bins) -> int:
    """Validate a bin count and return it as a plain int (SF-BL-9722).

    ``n_bins`` shapes every binned field in the scorecard, but it was
    never validated: ``n_bins=0`` (and ``True``/``False``) silently
    returned a scorecard with ECE/MCE/reliability forced to 0, negative
    values died inside numpy with "negative dimensions are not allowed",
    and non-integer values (2.5, NaN, "10", None) died with bare
    TypeErrors that never named the argument. Every binned entry point
    funnels through :func:`calibration_bins`, so validating here — and
    again at the top of :func:`evaluate` — makes all of them raise the
    one named domain error:

    Raises
    ------
    ValueError
        If ``n_bins`` is not a positive integer (>= 1). Booleans,
        floats (including integral floats such as 10.0 and non-finite
        values such as NaN/inf), strings, None and all other
        non-integer types are rejected; numpy integers are accepted.
    """
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)):
        raise ValueError(
            f"n_bins must be a positive integer (>= 1); "
            f"got n_bins={n_bins!r} ({type(n_bins).__name__})"
        )
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError(
            f"n_bins must be a positive integer (>= 1); got n_bins={n_bins}"
        )
    return n_bins


def calibration_bins(y, p, n_bins: int = 10, strategy: str = "uniform"):
    """
    Bin forecasts and summarise each bin.

    Returns dict of arrays: bin_edges, count, mean_pred, frac_pos, bin_id.
    strategy: 'uniform' (equal-width) or 'quantile' (equal-count).
    Empty bins have count 0 and nan mean_pred / frac_pos.

    Raises ValueError naming ``n_bins`` if it is not a positive
    integer (SF-BL-9722; see :func:`_validate_n_bins`).

    Raises
    ------
    ValueError
        If any element of ``p`` lies outside [0, 1] or is non-finite
        (SF-BL-8778). Previously ``p`` was digitized unclipped, so a
        malformed forecast such as 1.2 or -0.1 landed in an edge bin
        and pushed that bin's ``mean_pred`` outside [0, 1], silently
        distorting ECE/MCE — while ``brier_score``/``log_score``
        (SF-BL-9484) raised on the same row. Validation is pure (it
        never transforms ``p``), so in-range bins/ECE/MCE are
        bit-identical to the pre-fix computation. ``ece``, ``mce``,
        ``murphy_decomposition`` and ``evaluate`` inherit the guard
        through this function.
    ValueError
        If ``len(y) != len(p)`` (SF-BL-9822), naming both lengths.
        Pre-fix no length check ran before the per-bin loop, so
        ``bin_id`` (built from ``p``) indexed ``y`` with a boolean
        mask of the wrong length and
        ``calibration_bins([0, 1, 0], [0.5])`` died with a bare
        ``IndexError: boolean index did not match indexed array``
        instead of naming the mismatch. The
        :func:`_require_same_length` check is pure validation before
        any binning, so equal-length bins are bit-identical, and
        ``ece``, ``mce``, ``murphy_decomposition`` and ``evaluate``
        inherit it through this function.
    ValueError
        If any element of ``y`` or ``p`` is a string, an
        object-dtype (string / bare-object) element or a boolean
        (SF-BL-11424; see :func:`_reject_binned_panel`). The
        screens run on the raw inputs, before float conversion
        parses '0.9' -> 0.9 or coerces True -> 1.0.
    """
    n_bins = _validate_n_bins(n_bins)
    _reject_binned_panel(y, "y")  # SF-BL-11424: raw screen, before conversion (subsumes the SF-BL-10938 bool-p screen below for p, kept via the combined tags)
    _reject_binned_panel(p, "p")  # SF-BL-11424
    y = _as_y(y); p = np.asarray(p, dtype=float).ravel()
    _require_same_length(y, p)  # SF-BL-9822: documented length guard
    _validate_p_in_range(p)  # SF-BL-8778 (subsumes the SF-BL-3596 finite check)
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
    """ECE: count-weighted mean |frac_pos - mean_pred| over bins.

    Raises ValueError naming ``len(y)`` and ``len(p)`` if they differ
    (SF-BL-9822), inherited from :func:`calibration_bins`.

    Raises ValueError naming ``y`` / ``p`` if any element of either
    is a string, an object-dtype (string / bare-object) element or
    a boolean (SF-BL-11424; see :func:`_reject_binned_panel`). The
    screen is explicit in this function, on the raw inputs, before
    :func:`calibration_bins`' float conversion could parse '0.9'
    into a lawful forecast or coerce True into the 1.0 endpoint:
    previously ``expected_calibration_error([1, 0], ['0.9',
    '0.1'])`` returned 0.1, bit-identical to the numeric control,
    and a pandas bool panel returned 0.0.
    """
    _reject_binned_panel(y, "y")  # SF-BL-11424: explicit raw screen in ECE itself
    _reject_binned_panel(p, "p")  # SF-BL-11424
    t = calibration_bins(y, p, n_bins=n_bins, strategy=strategy)
    filled = t["count"] > 0
    if not filled.any():
        return 0.0
    w = t["count"][filled] / t["n"]
    return float(np.sum(w * np.abs(t["frac_pos"][filled] - t["mean_pred"][filled])))


def maximum_calibration_error(y, p, n_bins: int = 10, strategy: str = "uniform") -> float:
    """MCE: max |frac_pos - mean_pred| over non-empty bins.

    Raises ValueError naming ``len(y)`` and ``len(p)`` if they differ
    (SF-BL-9822), inherited from :func:`calibration_bins`.

    Raises ValueError naming ``y`` / ``p`` if any element of either
    is a string, an object-dtype (string / bare-object) element or
    a boolean (SF-BL-11424; see :func:`_reject_binned_panel`). The
    screen is explicit in this function, on the raw inputs, before
    :func:`calibration_bins`' float conversion could parse '0.9'
    into a lawful forecast or coerce True into the 1.0 endpoint:
    previously ``maximum_calibration_error([1, 0], ['0.9',
    '0.1'])`` returned 0.1, bit-identical to the numeric control,
    and a pandas bool panel returned 0.0.
    """
    _reject_binned_panel(y, "y")  # SF-BL-11424: explicit raw screen in MCE itself
    _reject_binned_panel(p, "p")  # SF-BL-11424
    t = calibration_bins(y, p, n_bins=n_bins, strategy=strategy)
    filled = t["count"] > 0
    if not filled.any():
        return 0.0
    return float(np.max(np.abs(t["frac_pos"][filled] - t["mean_pred"][filled])))


ece = expected_calibration_error
mce = maximum_calibration_error


def _reject_murphy_panel(raw, name):
    """Raise a named ValueError if a ``murphy_decomposition`` panel
    contains any string, object-dtype or bool element (SF-BL-11425).

    ``murphy_decomposition`` never type-validated its own panels —
    and uniquely among the binned entry points it could not inherit
    a screen either: it converts both panels with
    ``_as_y`` / ``np.asarray(p, dtype=float)`` BEFORE delegating to
    :func:`calibration_bins` and :func:`brier_score`, so every
    downstream screen (including the SF-BL-11424 bins screen) only
    ever sees the already-converted floats. Float conversion parses
    numeric strings as the numbers they spell and coerces bools to
    the lawful 1.0 / 0.0 endpoints, so
    ``murphy_decomposition([1, 0], ['0.9', '0.1'])`` returned brier
    0.01 bit-identical to the numeric control and a ``[True,
    False]`` forecast panel returned Murphy brier 0.0; pandas
    panels evaded even the canonical copy's ``_as_y`` screens, and
    a bare ``object()`` leaf died with a bare ``TypeError`` naming
    neither argument. SF-BL-11052 fixtured string panels only for
    ``brier_score`` / ``log_score`` / ``evaluate`` and no backlog
    item fixtured ``murphy_decomposition``'s own type domain
    (different function/arguments from SF-BL-11384-11395; the
    SF-BL-11424 bins screen cannot reach it, per above). This
    screen is therefore explicit in ``murphy_decomposition``
    itself, runs on the RAW ``y`` then ``p`` as its first
    statements — before the SF-BL-10938 bool-``p`` screen, any
    float conversion, the length guard, or delegation — and uses
    the same detection as :func:`_reject_binned_panel`:
    :func:`_has_bool` + :func:`_has_bool_in_pandas_panel` (bool
    dimension, first) and :func:`_has_string` +
    :func:`_has_string_in_pandas_panel` + :func:`_has_bare_object`
    + :func:`_has_bare_object_in_pandas_panel` (string /
    object-dtype dimension). The messages carry the SF-BL-10938 /
    SF-BL-11052 tags alongside SF-BL-11425, because the murphy bool
    funnel was already pinned to the SF-BL-10938 message by its
    test. Pure validation — it never transforms ``raw`` — so
    lawful panels are bit-identical: float/int panels, numpy
    numeric arrays, tuples, numeric pandas panels, and
    object-dtype arrays carrying only numeric leaves all pass
    untouched.
    """
    if _has_bool(raw) or _has_bool_in_pandas_panel(raw):
        raise ValueError(
            f"{name} must not contain boolean values; got boolean "
            f"value(s) in {name} (a bool is not a valid {name} value) "
            f"(SF-BL-10938, SF-BL-11425)"
        )
    if (_has_string(raw) or _has_bare_object(raw)
            or _has_string_in_pandas_panel(raw)
            or _has_bare_object_in_pandas_panel(raw)):
        raise ValueError(
            f"{name} must not contain string values; got string "
            f"or object-dtype value(s) in {name} (a string is not "
            f"a valid {name} value) (SF-BL-11052, SF-BL-11425)"
        )


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

    Raises ValueError naming ``len(y)`` and ``len(p)`` if they differ
    (SF-BL-9822 for the :func:`calibration_bins` funnel this function
    delegates to; explicit guard below per SF-BL-9558).
    ValueError
        If any element of ``y`` or ``p`` is a string, an object-dtype
        (string / bare-object) element, or a boolean (SF-BL-11425;
        see :func:`_reject_murphy_panel`). The screen is explicit in
        this function and runs on the raw panels, ``y`` first, before
        this function float-converts them — downstream screens in
        :func:`calibration_bins` / :func:`brier_score` could
        otherwise never see the original types.
    """
    _reject_murphy_panel(y, "y")  # SF-BL-11425: OWN raw screen FIRST — murphy float-converts before delegating, so downstream screens can never see the original types
    _reject_murphy_panel(p, "p")  # SF-BL-11425: OWN raw screen FIRST, before float conversion parses '0.9'->0.9 / coerces True->1.0
    _reject_bools(p, "p")  # SF-BL-10938: raw screen, before conversion (kept; SF-BL-11425 screen above fires first)
    y = _as_y(y); p = np.asarray(p, dtype=float).ravel()
    # SF-BL-9558 / SF-BL-9822: explicit length guard — murphy must raise
    # ValueError on len(y) != len(p) itself, not merely inherit an error
    # transitively from calibration_bins/brier_score (pre-fix the
    # singleton case surfaced as an IndexError from the binning path).
    _require_same_length(y, p)
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

# --------------------------------------------------------------------------
# Identifiability guard for the parametric calibrators (SF-BL-9410)
#
# Platt, Beta and Temperature all fit a slope through logit(p). If the
# calibration forecasts are (near-)constant, the logit feature is
# (near-)collinear with the intercept: only the fitted value at the
# single observed logit is identified, and the l2=1e-6 Newton fits split
# it arbitrarily between slope and intercept (Platt on 200 cases all at
# p=0.7 returned A=0.27, B=0.32 and extrapolated predict(0.3)=0.52 from
# a slope no data identified; Beta likewise returned a=-0.08, b=-0.26).
# predict() then applies that invented slope arbitrarily far from the
# calibration point.
#
# Guard: before fitting, compute the population std of the calibration
# set's (clipped) logits. Below _LOGIT_STD_FLOOR the slope is
# unidentifiable — Fisher information for a logistic slope is
# n * Var(s) * E[mu(1-mu)] <= n * Var(s) / 4, so at std 1e-2 even
# n=8000 gives slope SE >= ~2.2, larger than any plausible slope — and
# fit() returns the intercept-only fit instead: the constant base-rate
# prediction, the only fit such data identify, with the explicit flag
# ``intercept_only_ = True`` (and ``logit_std_`` recording the observed
# statistic) on the fitted object. The floor sits >90x below the logit
# std of every identified fit in test_calibration.py (benchmark
# Beta(2,2) ~= 1.14), so identified fits are bit-identical.
# --------------------------------------------------------------------------

_LOGIT_STD_FLOOR = 1e-2


def _logit_std(s) -> float:
    """Population std (ddof=0) of a logit array; NaN for an empty set."""
    s = np.asarray(s, dtype=float).ravel()
    return float(np.std(s)) if s.size else float("nan")


def _is_intercept_only(s) -> bool:
    """True iff a calibration set's logits identify no slope (SF-BL-9410).

    Empty sets return False and fall through to the pre-existing fit
    path unchanged; a single case (std exactly 0) is intercept-only by
    construction — one point cannot identify a slope.
    """
    return bool(np.asarray(s).size) and _logit_std(s) < _LOGIT_STD_FLOOR


def _base_rate_prob(y) -> float:
    """Intercept-only prediction: the empirical base rate.

    Clipped to [_CLIP, 1 - _CLIP] (the module's logit clip) only so
    logit() of it stays finite when every outcome is the same; for any
    mixed calibration set the clip is inactive and the value is the
    exact empirical rate.
    """
    return float(np.clip(float(np.mean(y)), _CLIP, 1 - _CLIP))


def _validate_max_iter(max_iter) -> int:
    """Validate a calibrator's ``max_iter`` iteration budget
    (SF-BL-11097).

    Returns the budget as a plain int. Pure validation — a valid
    budget is returned unchanged in value, so fits for lawful
    ``max_iter`` (including the ``100`` default and numpy integers)
    are bit-identical to the pre-fix computation.

    Raises
    ------
    ValueError
        Unless ``max_iter`` is a positive integer (>= 1). Previously
        neither calibrator constructor type-validated ``max_iter``:
        the Newton loops consume it via ``range(self.max_iter)``, so
        ``PlattScaler(max_iter=True)`` silently ran a single Newton
        iteration (bool is an int subclass: ``range(True)`` is
        ``range(1)``) and returned a truncated fit as if converged,
        ``BetaCalibrator(max_iter=True)`` did the same through the
        shared solver, and ``max_iter='100'`` / ``max_iter=2.5`` /
        ``max_iter=None`` died mid-fit with bare TypeErrors that
        never named the argument. ``max_iter=0`` / negative budgets
        silently returned the unfitted defaults. Booleans, floats
        (including integral floats such as 100.0, by parity with
        :func:`_validate_n_bins`, SF-BL-9722), strings, None,
        sequences and all other non-integer types are rejected;
        numpy integers are accepted.
    """
    msg = ("max_iter must be a positive integer (>= 1) (SF-BL-11097); "
           f"got max_iter={max_iter!r}")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)):
        raise ValueError(msg)
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError(msg)
    return max_iter


def _validate_platt_l2(l2) -> float:
    """Validate ``PlattScaler``'s ``l2`` penalty (SF-BL-9681 value
    domain; SF-BL-11097 type screen).

    Returns the penalty as a float. Pure validation — a valid penalty
    is returned unchanged in value, so fits for lawful ``l2``
    (including the ``1e-6`` default and ``0``) are bit-identical to
    the pre-fix computation.

    Raises
    ------
    ValueError
        Unless ``l2`` is a finite real scalar with ``l2 >= 0``.
        Previously ``l2`` was never validated in this (canonical)
        copy: the Newton gradient / Hessian add ``l2 * theta`` /
        ``l2 * I``, so ``PlattScaler(l2=True)`` fitted with the bool
        read as a penalty of 1.0 — six orders of magnitude above the
        ``1e-6`` default — and returned a silently shrunk slope,
        ``l2='1e-6'`` died mid-fit with a bare UFuncTypeError that
        never named ``l2``, ``l2=NaN`` / ``l2=inf`` poisoned the fit,
        and ``l2=-1.0`` returned an anti-penalised fit (a negative
        penalty rewards large coefficients). On an intercept-only
        (constant-logit) calibration set — which returns before the
        Newton loop ever uses ``l2`` — any invalid ``l2`` fitted
        silently as a normal calibrator. Values that are not real
        scalars at all (bool, string, None, a sequence, or any value
        ``float()`` cannot read as a real scalar) raise the same
        documented ValueError. (The ``superforecaster_out`` mirror
        already carried this guard under SF-BL-9681; this is the
        canonical-copy parity fix, extended to the type dimension
        by SF-BL-11097.)
    """
    msg = ("l2 must be finite with l2 >= 0 (SF-BL-9681, SF-BL-11097); "
           f"got l2={l2!r}")
    if isinstance(l2, bool) or isinstance(l2, (str, bytes)):
        raise ValueError(msg)
    if isinstance(l2, np.ndarray) and l2.ndim > 0:
        raise ValueError(msg)
    try:
        v = float(l2)
    except (TypeError, ValueError):
        raise ValueError(msg) from None
    if not np.isfinite(v) or v < 0.0:
        raise ValueError(msg)
    return v


def _validate_platt_tol(tol) -> float:
    """Validate ``PlattScaler``'s ``tol`` convergence threshold (SF-BL-9683).

    Returns the threshold as a float. Pure validation — a valid
    threshold is returned unchanged in value, so fits for lawful
    ``tol`` (including the ``1e-10`` default) are bit-identical to the
    pre-fix computation.

    Raises
    ------
    ValueError
        Unless ``tol`` is a finite real scalar with ``tol > 0``.
        Previously ``tol`` was never validated: the Newton loop breaks
        only when ``delta < tol``, and ``delta`` is a max absolute
        parameter change (always >= 0), so for ``tol=NaN`` every
        comparison is False and for ``tol <= 0`` the test can never
        fire — ``tol=NaN``, ``tol=0.0`` and ``tol=-1.0`` each burned
        all ``max_iter=100`` iterations and ``fit()`` returned the fit
        as if converged, with ``n_iter_=100`` the only tell. The
        mirror defect: ``tol=+inf`` makes ``delta < tol`` fire on the
        first iteration regardless of fit quality, silently truncating
        the fit. Values that are not real scalars at all (bool,
        string, None, a sequence, or any value ``float()`` cannot read
        as a real scalar) raise the same documented ValueError.
    """
    msg = ("tol must be finite with tol > 0 (SF-BL-9683); "
           f"got tol={tol!r}")
    if isinstance(tol, bool) or isinstance(tol, (str, bytes)):
        raise ValueError(msg)
    if isinstance(tol, np.ndarray) and tol.ndim > 0:
        raise ValueError(msg)
    try:
        v = float(tol)
    except (TypeError, ValueError):
        raise ValueError(msg) from None
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(msg)
    return v


class PlattScaler:
    """
    Platt scaling: p_cal = sigmoid(A * s + B), where s = logit(p_raw).

    Fit by Newton's method (IRLS-equivalent for 1-feature logistic
    regression) with a tiny L2 penalty for numerical stability. No scipy
    optimiser is used for the fit itself, per spec.

    Identifiability guard (SF-BL-9410): if the calibration logits have
    std below ``_LOGIT_STD_FLOOR``, the slope is unidentifiable and
    fit() returns the intercept-only fit A_=0, B_=logit(base rate) with
    ``intercept_only_ = True`` instead of an arbitrarily split slope.

    Raises
    ------
    ValueError
        If ``max_iter`` is a bool or any non-integer value, or is
        < 1 (SF-BL-11097; see :func:`_validate_max_iter`). The check
        runs at construction and again at the top of :meth:`fit` —
        before the intercept-only early return — so a
        post-construction mutation of ``max_iter`` cannot bypass it
        either. Previously ``max_iter=True`` silently truncated the
        Newton fit to a single iteration and ``max_iter='100'``
        died mid-fit with a bare TypeError naming nothing.
    ValueError
        If ``l2`` is a bool, a non-numeric value, NaN, infinite or
        negative (SF-BL-9681, SF-BL-11097; see
        :func:`_validate_platt_l2`). Same construction + fit
        re-validation discipline as ``max_iter``. Previously
        ``l2=True`` fitted with the bool read as a penalty of 1.0
        and ``l2='1e-6'`` died mid-fit with a bare UFuncTypeError.
    ValueError
        If ``tol`` is NaN, infinite or <= 0 (SF-BL-9683; see
        :func:`_validate_platt_tol`). The check runs at construction
        and again at the top of :meth:`fit` — before the
        intercept-only early return — so a post-construction mutation
        of ``tol`` cannot bypass it either.
    ValueError
        If ``fit`` receives a finite forecast outside [0, 1]
        (SF-BL-9904; see :func:`_validate_fit_p_range`). Previously
        fit routed ``p`` through :func:`_as_p` with no range screen,
        so 1.5 / -0.5 became extreme-logit training rows. The screen
        runs after the SF-BL-3596 finiteness check (non-finite
        handling unchanged) and before the intercept-only early
        return. Lawful endpoints 0 and 1 remain lawful.
    ValueError
        If ``fit`` receives any string or object-dtype element in
        ``y`` or ``p`` (SF-BL-11395; see
        :func:`_reject_string_fit_input`). The screen runs on the
        raw inputs, before any float conversion.
    ValueError
        If ``predict`` receives a non-finite forecast or a forecast
        outside [0, 1] (SF-BL-9751; see
        :func:`_validate_p_in_range`, SF-BL-9484). The frozen rule
        is rejection: previously ``predict`` passed its input
        straight to :func:`_as_p`, whose clip silently mapped 1.5
        to the 1-eps endpoint and -0.5 to eps, so a corrupt pooled
        forecast was recalibrated as a lawful endpoint forecast
        instead of being rejected. Lawful endpoints 0 and 1 remain
        lawful and are clipped internally exactly as before.
    ValueError
        If ``predict`` receives any boolean value (SF-BL-11051; see
        :func:`_reject_bool_predict_input`). The screen runs on the
        raw input, before float conversion coerces True -> 1.0 /
        False -> 0.0 into lawful endpoints.
    ValueError
        If ``predict`` receives any string value (SF-BL-11463; see
        :func:`_reject_string_predict_input`). The screen runs on the
        raw input, before float conversion parses '0.55' -> 0.55 into
        a lawful forecast.
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-10, l2: float = 1e-6):
        self.max_iter = _validate_max_iter(max_iter)  # SF-BL-11097
        self.tol = _validate_platt_tol(tol)  # SF-BL-9683
        self.l2 = _validate_platt_l2(l2)  # SF-BL-9681 / SF-BL-11097
        self.A_, self.B_ = 1.0, 0.0
        self.n_iter_ = 0
        self.intercept_only_ = False
        self.logit_std_ = float("nan")

    def fit(self, y, p):
        _reject_string_fit_input(y, "y")  # SF-BL-11395: raw screen FIRST, before float conversion parses '1'->1.0
        _reject_string_fit_input(p, "p")  # SF-BL-11395: raw screen FIRST, before float conversion parses '0.9'->0.9
        # SF-BL-11097: re-validate in case max_iter was mutated after
        # construction; must precede the intercept-only early return.
        self.max_iter = _validate_max_iter(self.max_iter)
        # SF-BL-9681 / SF-BL-11097: same re-validation for l2.
        self.l2 = _validate_platt_l2(self.l2)
        # SF-BL-9683: re-validate in case tol was mutated after
        # construction; must precede the intercept-only early return.
        self.tol = _validate_platt_tol(self.tol)
        y = _as_y(y)
        p_arr = np.asarray(p, dtype=float)
        _require_finite(p_arr, "p")  # SF-BL-3596: unchanged, runs first
        _validate_fit_p_range(p_arr)  # SF-BL-9904: before _as_p clips
        s = logit(_as_p(p_arr)).ravel()
        _require_same_length(y, s)
        self.logit_std_ = _logit_std(s)
        self.intercept_only_ = _is_intercept_only(s)
        if self.intercept_only_:
            self.A_, self.B_ = 0.0, float(logit(_base_rate_prob(y)))
            self.n_iter_ = 0
            return self
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
        _reject_bool_predict_input(p)  # SF-BL-11051: raw screen, before float conversion coerces True->1.0
        _reject_string_predict_input(p)  # SF-BL-11463: raw screen, before float conversion parses '0.55'->0.55
        p_arr = np.asarray(p, dtype=float)
        # SF-BL-9751 (frozen rule): predict inputs must be finite
        # SF-BL-11198: composite parity task -- this same screen is the
        # Platt/Beta/Temperature rejection of finite out-of-range and
        # NaN predict inputs, matching IsotonicCalibrator (SF-BL-9719).
        # probabilities in [0, 1]; anything else raises ValueError.
        # Without this screen _as_p silently clipped corrupt pooled
        # forecasts to the [eps, 1-eps] endpoints and the Platt map
        # recalibrated them as if they were lawful endpoint
        # forecasts (predict(1.5) == predict(1.0),
        # predict(-0.5) == predict(0.0)). Non-finite values already
        # raised via _as_p's finiteness screen (SF-BL-3596); this
        # screen runs first so non-finite and out-of-range inputs
        # raise the one documented SF-BL-9484 range error, as frozen
        # for IsotonicCalibrator.predict by SF-BL-9719 and for
        # BetaCalibrator.predict by SF-BL-9720. The screen runs
        # before A_/B_ are used, so it also applies on an
        # intercept-only fit and on an unfitted calibrator. Inputs
        # inside [0, 1] -- endpoints 0/1 included -- flow through
        # _as_p exactly as before, so lawful predictions are
        # bit-identical.
        _validate_p_in_range(p_arr)
        return expit(self.A_ * logit(_as_p(p_arr)) + self.B_)

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
        """Fit PAVA on a calibration set of (y, p) pairs.

        Raises
        ------
        ValueError
            If the calibration set is empty (SF-BL-9685). Previously
            ``fit([], [])`` returned a fitted object with empty knots
            and the failure surfaced only later, inside ``predict``,
            as a bare ``IndexError`` — far from the bad call and in
            breach of this class's own unfitted ``RuntimeError``
            contract. The guard is pure validation before any PAVA
            work and before ``x_``/``y_`` are assigned, so a rejected
            fit leaves the calibrator unfitted and non-empty fits are
            bit-identical.
        ValueError
            If any finite element of ``p`` lies outside [0, 1]
            (SF-BL-9904; see :func:`_validate_fit_p_range`).
            Previously fit screened only finiteness, so it built
            PAVA knots at x=-0.5 and x=1.5. The screen runs after
            the SF-BL-3596 finiteness check (non-finite handling
            unchanged), after the empty/length guards, and before
            any PAVA work or ``x_``/``y_`` assignment.
        ValueError
            If ``fit`` receives any string or object-dtype element
            in ``y`` or ``p`` (SF-BL-11395; see
            :func:`_reject_string_fit_input`). The screen runs on
            the raw inputs, before any float conversion, so a
            rejected fit leaves the calibrator unfitted.
        """
        _reject_string_fit_input(y, "y")  # SF-BL-11395: raw screen FIRST
        _reject_string_fit_input(p, "p")  # SF-BL-11395: raw screen FIRST
        y = _as_y(y); p = np.asarray(p, dtype=float).ravel()
        _require_finite(p, "p")  # SF-BL-3596: fit bypasses _as_p
        # SF-BL-9685: empty calibration set — before the length check,
        # so a one-sided empty input names the empty set too.
        if y.size == 0 or p.size == 0:
            raise ValueError(
                "IsotonicCalibrator.fit: calibration set must be "
                f"non-empty; got empty calibration set "
                f"(len(y)={y.size}, len(p)={p.size}) — there is "
                "nothing to fit (SF-BL-9685)"
            )
        _require_same_length(y, p)
        _validate_fit_p_range(p)  # SF-BL-9904: after the finite check above
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
        _reject_bool_predict_input(p)  # SF-BL-11051: raw screen first — even on an unfitted calibrator a bool raises the named ValueError, not RuntimeError
        _reject_string_predict_input(p)  # SF-BL-11463: raw screen first — even on an unfitted calibrator a string raises the named ValueError, not RuntimeError
        if self.x_ is None:
            raise RuntimeError("IsotonicCalibrator is not fitted")
        p = np.asarray(p, dtype=float)
        # SF-BL-9719 (frozen rule): predict inputs must be finite
        # probabilities in [0, 1]; anything else raises ValueError.
        # Without this screen np.interp / searchsorted silently mapped
        # corrupt pooled forecasts to plausible endpoint probabilities
        # (predict([inf]) -> y_[-1], step-mode predict([nan]) -> y_[-1],
        # predict([-0.5, 1.5]) flat-clipped to the endpoints). Inputs
        # inside [0, 1] but outside the fitted knot range keep the
        # existing flat endpoint behaviour, so lawful predictions are
        # bit-identical to the pre-fix computation.
        _validate_p_in_range(p)
        if self.interpolation == "linear":
            out = np.interp(p, self.x_, self.y_, left=self.y_[0], right=self.y_[-1])
        else:
            idx = np.clip(np.searchsorted(self.x_, p, side="right") - 1, 0, len(self.x_) - 1)
            out = self.y_[idx]
        return np.clip(out, 0.0, 1.0)

    transform = predict


def _validate_beta_tol(tol) -> float:
    """Validate ``BetaCalibrator``'s ``tol`` convergence threshold
    (SF-BL-9718).

    Returns the threshold as a float. Pure validation — a valid
    threshold is returned unchanged in value, so fits for lawful
    ``tol`` (including the ``1e-10`` default, which equals the Newton
    solver's own default) are bit-identical to the pre-fix
    computation.

    Raises
    ------
    ValueError
        Unless ``tol`` is a finite real scalar with ``tol > 0``.
        Previously the Beta path never validated ``tol`` at all:
        ``BetaCalibrator`` did not even expose the parameter (it
        silently inherited the shared Newton solver's default and
        ignored a ``tol`` attribute assigned before ``fit()``), and
        the solver's break test ``max|step| < tol`` can never fire
        for a NaN threshold (every comparison is False) or a
        non-positive one (the step size is non-negative), so
        ``tol=NaN``, ``tol=0`` and ``tol=-1`` each exhausted the full
        ``max_iter=100`` Newton budget on a 200-row book and returned
        the three-parameter fit as if converged, with no error and
        no convergence flag. Values that are not real scalars at
        all (bool, string, None, a sequence, or any value
        ``float()`` cannot read as a real scalar) raise the same
        documented ValueError.
    """
    msg = ("tol must be finite with tol > 0 (SF-BL-9718); "
           f"got tol={tol!r}")
    if isinstance(tol, bool) or isinstance(tol, (str, bytes)):
        raise ValueError(msg)
    if isinstance(tol, np.ndarray) and tol.ndim > 0:
        raise ValueError(msg)
    try:
        v = float(tol)
    except (TypeError, ValueError):
        raise ValueError(msg) from None
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(msg)
    return v


def _logistic_fit_newton(X, y, l2=1e-6, max_iter=100, tol=1e-10):
    """Shared Newton solver for logistic regression (used by BetaCalibrator).

    Raises ValueError if ``tol`` is not a finite real scalar
    with ``tol > 0`` (SF-BL-9718; see :func:`_validate_beta_tol`).
    The check runs before the Newton loop iterates, so a
    mis-set threshold can never silently exhaust the iteration
    budget and return an unconverged fit as converged.
    Raises ValueError if ``max_iter`` is a bool or any non-integer
    value, or is < 1 (SF-BL-11097; see :func:`_validate_max_iter`).
    Previously ``range(max_iter)`` consumed the budget unchecked,
    so ``max_iter=True`` silently ran a single iteration and
    ``max_iter='100'`` died with a bare TypeError naming nothing.
    """
    max_iter = _validate_max_iter(max_iter)  # SF-BL-11097: before iterating
    tol = _validate_beta_tol(tol)  # SF-BL-9718: before iterating
    if X.shape[0] != len(np.asarray(y).ravel()):
        raise ValueError(
            f"len(y) ({len(np.asarray(y).ravel())}) != X rows ({X.shape[0]}): "
            "y and p must have the same length"
        )
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


def _validate_beta_l2(l2) -> float:
    """Validate ``BetaCalibrator``'s ``l2`` penalty (SF-BL-9682).

    Returns the penalty as a float. Pure validation — a valid penalty
    is returned unchanged in value, so fits for lawful ``l2`` (including
    the ``1e-6`` default and ``0``) are bit-identical to the pre-fix
    computation.

    Raises
    ------
    ValueError
        Unless ``l2`` is a finite real scalar with ``l2 >= 0``.
        Previously ``l2`` was never validated: ``fit`` passed
        ``self.l2`` unchecked into the shared ``_logistic_fit_newton``,
        whose gradient / Hessian add ``l2 * theta`` / ``l2 * I``, so
        ``l2=NaN`` (and ``l2=inf``) fitted to ``a_=b_=c_=NaN`` and
        ``l2=-1.0`` on a repeated 20-row book returned an
        anti-penalised ``a=-11.17, b=+11.17`` — a silently corrupt
        three-parameter recalibration map (a negative penalty rewards
        large coefficients). On an intercept-only (constant-logit)
        calibration set — which returns before the Newton solver ever
        uses ``l2`` — any invalid ``l2`` fitted silently as a normal
        calibrator. Values that are not real scalars at all (bool,
        string, None, a sequence, or any value ``float()`` cannot read
        as a real scalar) raise the same documented ValueError. This
        guard is Beta-specific: the shared solver itself gains no
        central ``l2`` check (its SF-BL-9718 ``tol`` check does not
        screen ``l2``), and PlattScaler (SF-BL-9681, different class,
        different solver path) is untouched.
    """
    msg = ("l2 must be finite with l2 >= 0 (SF-BL-9682); "
           f"got l2={l2!r}")
    if isinstance(l2, bool) or isinstance(l2, (str, bytes)):
        raise ValueError(msg)
    if isinstance(l2, np.ndarray) and l2.ndim > 0:
        raise ValueError(msg)
    try:
        v = float(l2)
    except (TypeError, ValueError):
        raise ValueError(msg) from None
    if not np.isfinite(v) or v < 0.0:
        raise ValueError(msg)
    return v


class BetaCalibrator:
    """
    Beta calibration (Kull et al., 2017):
        logit(p_cal) = c + a * ln(p) + b * ln(1 - p)
    Fit as logistic regression on features [ln p, ln(1-p), 1] by Newton's
    method. Nests Platt-on-logit and identity as special cases, and can fix
    both over- and under-confidence asymmetrically.
    Params after fit: a_, b_, c_.

    Identifiability guard (SF-BL-9410): on a (near-)constant calibration
    set both log features are (near-)constant and collinear with the
    intercept, so fit() returns the intercept-only fit a_=b_=0,
    c_=logit(base rate) with ``intercept_only_ = True``. The guard
    statistic is std(logit p) = std(ln p - ln(1-p)), taken from the
    fitted features themselves.

    Raises
    ------
    ValueError
        If ``max_iter`` is a bool or any non-integer value, or is
        < 1 (SF-BL-11097; see :func:`_validate_max_iter`). The check
        runs at construction and again at the top of :meth:`fit` —
        before the intercept-only early return — and again inside
        the shared Newton solver, so a post-construction mutation
        of ``max_iter`` cannot bypass it either. Previously
        ``max_iter=True`` silently truncated the fit to a single
        Newton iteration and ``max_iter='100'`` died mid-fit with
        a bare TypeError naming nothing. (TASK.md's prereg names
        ``l2`` for this class explicitly and ``max_iter`` for the
        calibrator constructors collectively; the ``max_iter``
        screen is applied here for parity with ``PlattScaler``.)
    ValueError
        If ``tol`` is NaN, infinite or <= 0 (SF-BL-9718; see
        :func:`_validate_beta_tol`). The check runs at
        construction and again at the top of :meth:`fit` —
        before the intercept-only early return — so a
        post-construction mutation of ``tol`` cannot bypass
        it either.
    ValueError
        If ``l2`` is a bool, a non-numeric value, NaN, infinite or
        negative (SF-BL-9682; SF-BL-11097 for the type dimension;
        see :func:`_validate_beta_l2`). The check runs at construction
        and again at the top of :meth:`fit` — before the
        intercept-only early return — so a post-construction mutation
        of ``l2`` cannot bypass it either. Previously (in the
        pre-SF-BL-9682 state the task repro was taken against)
        ``l2=True`` fitted with the bool read as a penalty of 1.0.
    ValueError
        If ``fit`` receives a finite forecast outside [0, 1]
        (SF-BL-9904; see :func:`_validate_fit_p_range`). Previously
        fit routed ``p`` through :func:`_as_p` (via ``_features``)
        with no range screen, so 1.5 / -0.5 were clipped to the
        endpoints and fitted as lawful rows. The screen runs after
        the SF-BL-3596 finiteness check (non-finite handling
        unchanged) and before the intercept-only early return.
        Lawful endpoints 0 and 1 remain lawful.
    ValueError
        If ``fit`` receives any string or object-dtype element in
        ``y`` or ``p`` (SF-BL-11395; see
        :func:`_reject_string_fit_input`). The screen runs on the
        raw inputs, before any float conversion.
    ValueError
        If ``predict`` receives a non-finite forecast or a forecast
        outside [0, 1] (SF-BL-9720; see
        :func:`_validate_p_in_range`, SF-BL-9484). The frozen rule
        is rejection: previously ``predict`` passed its input
        straight to :func:`_as_p`, whose clip silently mapped 1.5
        to the 1-eps endpoint and -0.2 to eps, so a corrupt pooled
        forecast was recalibrated as a lawful endpoint forecast
        instead of being rejected. Lawful endpoints 0 and 1 remain
        lawful and are clipped internally exactly as before.
    ValueError
        If ``predict`` receives any boolean value (SF-BL-11051; see
        :func:`_reject_bool_predict_input`). The screen runs on the
        raw input, before float conversion coerces True -> 1.0 /
        False -> 0.0 into lawful endpoints.
    ValueError
        If ``predict`` receives any string value (SF-BL-11463; see
        :func:`_reject_string_predict_input`). The screen runs on the
        raw input, before float conversion parses '0.55' -> 0.55 into
        a lawful forecast.
    """

    def __init__(self, l2: float = 1e-6, tol: float = 1e-10, max_iter: int = 100):
        self.l2 = _validate_beta_l2(l2)  # SF-BL-9682
        self.tol = _validate_beta_tol(tol)  # SF-BL-9718
        self.max_iter = _validate_max_iter(max_iter)  # SF-BL-11097
        self.a_, self.b_, self.c_ = 1.0, 1.0, 0.0
        self.intercept_only_ = False
        self.logit_std_ = float("nan")

    @staticmethod
    def _features(p):
        pc = _as_p(p).ravel()
        return np.column_stack([np.log(pc), np.log(1 - pc), np.ones_like(pc)])

    def fit(self, y, p):
        _reject_string_fit_input(y, "y")  # SF-BL-11395: raw screen FIRST, before float conversion
        _reject_string_fit_input(p, "p")  # SF-BL-11395: raw screen FIRST, before float conversion
        # SF-BL-11097: re-validate in case max_iter was mutated after
        # construction; must precede the intercept-only early return.
        self.max_iter = _validate_max_iter(self.max_iter)
        # SF-BL-9718: re-validate in case tol was mutated after
        # construction; must precede the intercept-only early return.
        self.tol = _validate_beta_tol(self.tol)
        # SF-BL-9682: same re-validation for l2.
        self.l2 = _validate_beta_l2(self.l2)
        y = _as_y(y)
        p_arr = np.asarray(p, dtype=float)
        _require_finite(p_arr, "p")  # SF-BL-3596: unchanged, runs first
        _validate_fit_p_range(p_arr)  # SF-BL-9904: before _features clips
        X = self._features(p_arr)
        _require_same_length(y, X[:, 0])
        s = X[:, 0] - X[:, 1]  # logit(p), exactly as the fit sees it
        self.logit_std_ = _logit_std(s)
        self.intercept_only_ = _is_intercept_only(s)
        if self.intercept_only_:
            self.a_, self.b_, self.c_ = 0.0, 0.0, float(logit(_base_rate_prob(y)))
            return self
        theta = _logistic_fit_newton(X, y, l2=self.l2, max_iter=self.max_iter, tol=self.tol)
        self.a_, self.b_, self.c_ = float(theta[0]), float(theta[1]), float(theta[2])
        return self

    def predict(self, p):
        _reject_bool_predict_input(p)  # SF-BL-11051: raw screen, before float conversion coerces True->1.0
        _reject_string_predict_input(p)  # SF-BL-11463: raw screen, before float conversion parses '0.55'->0.55
        p_arr = np.asarray(p, dtype=float)
        # SF-BL-9720 (frozen rule): predict inputs must be finite
        # SF-BL-11198: composite parity task -- this same screen is the
        # Platt/Beta/Temperature rejection of finite out-of-range and
        # NaN predict inputs, matching IsotonicCalibrator (SF-BL-9719).
        # probabilities in [0, 1]; anything else raises ValueError.
        # Without this screen _as_p silently clipped corrupt pooled
        # forecasts to the [eps, 1-eps] endpoints and the beta map
        # recalibrated them as if they were lawful endpoint forecasts
        # (predict(1.5) == predict(1.0), predict(-0.2) == predict(0.0),
        # and an intercept-only fit returned its constant for 1.5).
        # Non-finite values already raised via _as_p's finiteness
        # screen (SF-BL-3596); this screen runs first so non-finite
        # and out-of-range inputs raise the one documented
        # SF-BL-9484 range error, as frozen for
        # IsotonicCalibrator.predict by SF-BL-9719. Inputs inside
        # [0, 1] -- endpoints 0/1 included -- flow through _as_p
        # exactly as before, so lawful predictions are bit-identical.
        _validate_p_in_range(p_arr)
        pc = _as_p(p_arr)
        z = self.c_ + self.a_ * np.log(pc) + self.b_ * np.log(1 - pc)
        return expit(z)

    transform = predict


class TemperatureScaler:
    """
    Temperature scaling: p_cal = sigmoid(logit(p_raw) / T).
    T > 1 softens (fixes overconfidence), T < 1 sharpens.
    T is fit by minimising log score with scipy (bounded, T in [0.05, 20]).

    Identifiability guard (SF-BL-9410): with (near-)constant calibration
    logits, a fitted T is pinned by a single point and then applied as a
    through-origin slope to every other logit — a shape no data
    identified. The temperature form has no intercept, so fit() returns
    the intercept-only fit with ``intercept_only_ = True`` and
    ``constant_prob_`` = base rate, and predict() returns that constant
    for every input while the flag is set (T_ is left at the neutral
    1.0 placeholder and is unused in that state).

    Raises
    ------
    ValueError
        If ``fit`` receives a finite forecast outside [0, 1]
        (SF-BL-9904; see :func:`_validate_fit_p_range`). Previously
        fit routed ``p`` through :func:`_as_p` with no range screen,
        so 1.5 / -0.5 became extreme-logit training rows. The screen
        runs after the SF-BL-3596 finiteness check (non-finite
        handling unchanged) and before the intercept-only early
        return. Lawful endpoints 0 and 1 remain lawful.
    ValueError
        If ``fit`` receives any string or object-dtype element in
        ``y`` or ``p`` (SF-BL-11395; see
        :func:`_reject_string_fit_input`). The screen runs on the
        raw inputs — before float conversion and before the
        intercept-only early return.
    ValueError
        If ``predict`` receives a non-finite forecast or a forecast
        outside [0, 1] (SF-BL-9752; see
        :func:`_validate_p_in_range`, SF-BL-9484). The frozen rule
        is rejection: previously ``predict`` passed its input
        straight to :func:`_as_p`, whose clip silently mapped 1.5
        to the 1-eps endpoint and -0.5 to eps, so a corrupt pooled
        forecast was recalibrated as a lawful endpoint forecast
        instead of being rejected (and an intercept-only fit
        returned its constant for 1.5). Lawful endpoints 0 and 1
        remain lawful and are clipped internally exactly as before.
    ValueError
        If ``predict`` receives any boolean value (SF-BL-11051; see
        :func:`_reject_bool_predict_input`). The screen runs on the
        raw input — before float conversion and before the
        intercept-only early return — so an intercept-only fit does
        not return its constant for True/False either.
    ValueError
        If ``predict`` receives any string value (SF-BL-11463; see
        :func:`_reject_string_predict_input`). The screen runs on the
        raw input — before float conversion and before the
        intercept-only early return — so an intercept-only fit does
        not return its constant for '0.55' either.
    """

    def __init__(self):
        self.T_ = 1.0
        self.intercept_only_ = False
        self.logit_std_ = float("nan")
        self.constant_prob_ = 0.5

    def fit(self, y, p):
        _reject_string_fit_input(y, "y")  # SF-BL-11395: raw screen FIRST, before float conversion
        _reject_string_fit_input(p, "p")  # SF-BL-11395: raw screen FIRST, before float conversion
        y = _as_y(y)
        p_arr = np.asarray(p, dtype=float)
        _require_finite(p_arr, "p")  # SF-BL-3596: unchanged, runs first
        _validate_fit_p_range(p_arr)  # SF-BL-9904: before _as_p clips
        s = logit(_as_p(p_arr)).ravel()
        _require_same_length(y, s)
        self.logit_std_ = _logit_std(s)
        self.intercept_only_ = _is_intercept_only(s)
        if self.intercept_only_:
            self.T_ = 1.0
            self.constant_prob_ = _base_rate_prob(y)
            return self

        def nll(logT):
            return log_score(y, expit(s / np.exp(logT[0])))

        res = minimize(nll, x0=np.array([0.0]), method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 500})
        T = float(np.exp(res.x[0]))
        self.T_ = float(np.clip(T, 0.05, 20.0))
        return self

    def predict(self, p):
        _reject_bool_predict_input(p)  # SF-BL-11051: raw screen, before float conversion and before the intercept-only early return
        _reject_string_predict_input(p)  # SF-BL-11463: raw screen, before float conversion and before the intercept-only early return
        p_arr = np.asarray(p, dtype=float)
        # SF-BL-9752 (frozen rule): predict inputs must be finite
        # SF-BL-11198: composite parity task -- this same screen is the
        # Platt/Beta/Temperature rejection of finite out-of-range and
        # NaN predict inputs, matching IsotonicCalibrator (SF-BL-9719).
        # probabilities in [0, 1]; anything else raises ValueError.
        # Without this screen _as_p silently clipped corrupt pooled
        # forecasts to the [eps, 1-eps] endpoints and temperature
        # scaling recalibrated them as if they were lawful endpoint
        # forecasts (predict(1.5) == predict(1.0), predict(-0.5) ==
        # predict(0.0), and an intercept-only fit returned its
        # constant for 1.5). Non-finite values already raised via
        # _as_p's finiteness screen (SF-BL-3596); this screen runs
        # first — and before the intercept-only early return — so
        # non-finite and out-of-range inputs raise the one documented
        # SF-BL-9484 range error, as frozen for
        # IsotonicCalibrator.predict by SF-BL-9719 and for
        # BetaCalibrator.predict by SF-BL-9720. Inputs inside
        # [0, 1] — endpoints 0/1 included — flow through _as_p
        # exactly as before, so lawful predictions are bit-identical.
        _validate_p_in_range(p_arr)
        if getattr(self, "intercept_only_", False):
            # SF-BL-9410: intercept-only fit — the identified constant
            # for every lawful input, whatever logit it carries.
            return np.full(p_arr.shape, self.constant_prob_,
                           dtype=float)
        return expit(logit(_as_p(p_arr)) / self.T_)

    transform = predict


# Convenience registry / functional helpers
CALIBRATORS = {"platt": PlattScaler, "isotonic": IsotonicCalibrator,
               "temperature": TemperatureScaler, "beta": BetaCalibrator}


def _has_string_in_dataframe_panel(raw) -> bool:
    """True iff a pandas ``DataFrame`` panel contains any string or
    bare-object value (SF-BL-11426).

    :func:`_has_string` returns False for any non-list/tuple/ndarray
    container and :func:`_has_string_in_pandas_panel` deliberately
    returns False for a ``DataFrame``, so a text-serialised panel
    carried in a DataFrame slipped past every fit screen:
    ``np.asarray(df, dtype=float)`` reads the frame's values and
    parses the strings they spell. A DataFrame is a lawful panel
    shape for the dispatcher (``np.asarray`` reads its values), so
    this detector adds the DataFrame scan, mirroring
    :func:`_has_string_in_trimmed_pool_panel` (SF-BL-11464): a
    ``string`` / ``U`` / ``S`` column dtype, a category dtype whose
    categories contain a string, or any string / bare-object leaf
    in the frame's values. Numeric DataFrames contain no string
    value, so they return False and stay bit-identical. Any
    inspection error returns False, leaving the pre-existing
    downstream behaviour untouched.
    """
    try:
        if not (type(raw).__module__ or "").startswith("pandas"):
            return False
        if type(raw).__name__ != "DataFrame":
            return False
        dtypes = getattr(raw, "dtypes", None)
        if dtypes is not None:
            for dt in list(dtypes):
                if getattr(dt, "name", None) == "string":
                    return True
                if getattr(dt, "kind", None) in ("U", "S"):
                    return True
                cats = getattr(dt, "categories", None)
                if cats is not None and any(_has_string(c) for c in list(cats)):
                    return True
        values = getattr(raw, "values", None)
        if values is not None and (_has_string(values)
                                   or _has_bare_object(values)):
            return True
        return False
    except Exception:  # noqa: BLE001 — inspection must never break lawful panels
        return False


def _reject_string_dispatcher_input(raw, name):
    """Raise a named ValueError if a ``fit_calibrator`` panel
    contains any string or object-dtype element (SF-BL-11426).

    The registry dispatcher never type-validated its own panels:
    it forwarded ``y`` / ``p`` straight into the calibrator
    classes' ``fit`` (hence into ``_as_y`` / ``_as_p`` coercion),
    so its entry point's type domain was claimed by no screen of
    its own — SF-BL-11395 fixtured string panels only for the four
    classes' direct ``fit()`` methods and SF-BL-11024 only bool
    panels there — and one container form bypassed even those
    inherited screens: ``fit_calibrator('platt', y,
    pd.DataFrame({'p': ['0.9', '0.2', '0.8', '0.3', '0.7',
    '0.4']}))`` returned ``A_=17.762763687307135,
    B_=-3.7957007060548054`` and the ``'beta'`` dispatch returned
    ``a_=17.249465734206364, b_=-16.39197958793144``, both
    bit-identical to the numeric controls, because
    :func:`_has_string_in_pandas_panel` returns False for a
    DataFrame by construction. A numeric string is not a number —
    the panel's type is itself evidence of a corrupt JSON/CSV
    feed, and fitting it hides that bug. This screen therefore
    runs on the RAW ``y`` and ``p`` as the first statements of
    :func:`fit_calibrator`, before the registry lookup and before
    any float conversion, for every registered calibrator name.
    Detection is :func:`_has_string` + :func:`_has_bare_object`
    (lists/tuples at any nesting, string-dtype ndarrays kind
    ``U``/``S``, object-dtype ndarrays holding a string or
    bare-object leaf, bare ``str``/``bytes`` scalars) plus
    :func:`_has_string_in_pandas_panel` (pandas ``Series`` /
    ``Index`` / ``Categorical`` string panels, which
    ``_has_string`` cannot see) plus
    :func:`_has_string_in_dataframe_panel` (pandas ``DataFrame``
    panels). Pure validation — it never transforms ``raw`` — so
    lawful panels are bit-identical: float/int panels, numpy
    numeric arrays, tuples, numeric pandas panels (Series and
    DataFrame), and object-dtype arrays carrying only numeric
    leaves all pass untouched. The raised error cites
    SF-BL-11395 alongside SF-BL-11426 because this entry screen
    layers over that fit()-level screen (the SF-BL-11395 suite
    pins ``fit_calibrator`` string errors to its tag; both guards
    reject the same panels, the dispatcher's first).
    """
    has_str = (_has_string(raw) or _has_bare_object(raw)
               or _has_string_in_dataframe_panel(raw))
    if not has_str:
        try:
            has_str = _has_string_in_pandas_panel(raw)
        except Exception:  # noqa: BLE001 — helper is defined later in this module; never break lawful panels
            has_str = False
    if has_str:
        raise ValueError(
            f"{name} must not contain string values; got string "
            f"or object-dtype value(s) in {name} (a string is not "
            f"a valid {name} value) (SF-BL-11426; cf. SF-BL-11395, "
            f"the fit()-level screen this entry screen layers over)"
        )


def fit_calibrator(name: str, y, p):
    """Fit a calibrator by name and return the fitted object.

    Raises
    ------
    ValueError
        If ``y`` or ``p`` contains any string or object-dtype
        element (SF-BL-11426; see
        :func:`_reject_string_dispatcher_input`). The screen runs
        on the raw panels at the dispatcher's own entry point —
        before the registry lookup and before the chosen
        calibrator's ``fit`` (hence any ``_as_y`` / ``_as_p``
        float conversion) could parse a text-serialised panel as
        the numbers it spells.
    ValueError
        If ``name`` is not a registered calibrator name.
    """
    _reject_string_dispatcher_input(y, "y")  # SF-BL-11426: dispatcher entry screen FIRST, before registry lookup / float conversion parses '1'->1.0
    _reject_string_dispatcher_input(p, "p")  # SF-BL-11426: dispatcher entry screen FIRST, before float conversion parses '0.9'->0.9
    if name not in CALIBRATORS:
        raise ValueError(f"unknown calibrator {name!r}; choose from {sorted(CALIBRATORS)}")
    return CALIBRATORS[name]().fit(y, p)


def _validate_evaluate_panel(y, p):
    """Raise a named ValueError on an empty or all-non-finite panel
    (SF-BL-10742).

    ``evaluate`` previously had no panel-content guard of its own:
    on an empty panel ``murphy_decomposition`` / ``brier_score`` /
    ``log_score`` averaged empty slices to NaN while
    ``expected_calibration_error`` / ``maximum_calibration_error``
    returned 0.0 for "no filled bin", so ``evaluate([], [])``
    returned one scorecard simultaneously claiming perfect
    calibration (ECE 0.0, MCE 0.0, reliability 0.0) and no score at
    all (Brier / log score / uncertainty NaN). A zero-observation
    book has no scorecard — it must fail at the entry point, not
    fabricate a mixed verdict. All-non-finite panels (no row whose
    ``y`` and ``p`` are both finite) are the same defect in the
    limit: there is no finite observation to score, so they raise
    the same named error here instead of relying on whichever
    downstream finiteness screen happened to fire first.

    Pure validation — it never transforms ``y`` or ``p``. A panel
    with at least one fully finite row passes untouched, so
    partially non-finite panels keep raising exactly the
    pre-existing downstream finiteness/range ValueErrors and
    non-empty lawful panels are bit-identical to the pre-fix
    computation. Inputs ``float()`` cannot read defer to the
    downstream validators (the helper returns without raising), so
    their pre-existing errors are preserved too.
    """
    # Emptiness needs no dtype conversion: length equality has
    # already been checked by _require_same_length.
    if np.asarray(y).size == 0:
        raise ValueError(
            "evaluate: panel must be non-empty; got empty panel "
            f"(len(y)={len(np.asarray(y).ravel())}, "
            f"len(p)={len(np.asarray(p).ravel())}) — there is no "
            "scorecard for zero observations (SF-BL-10742)"
        )
    try:
        y_arr = np.asarray(y, dtype=float).ravel()
        p_arr = np.asarray(p, dtype=float).ravel()
    except (TypeError, ValueError):
        return  # defer to downstream validators, as pre-fix
    if y_arr.size and not np.any(np.isfinite(y_arr) & np.isfinite(p_arr)):
        raise ValueError(
            "evaluate: panel must contain at least one finite "
            f"observation; got all-non-finite panel "
            f"(len(y)={y_arr.size}, len(p)={p_arr.size}) — there is "
            "no finite observation to score (SF-BL-10742)"
        )


def evaluate(y, p, n_bins: int = 10) -> dict:
    """One-call scorecard: Brier, log score, ECE, MCE + Murphy parts.

    Raises ValueError naming ``n_bins`` if it is not a positive
    integer (SF-BL-9722; see :func:`_validate_n_bins`).

    Raises ValueError if ``len(y) != len(p)`` (SF-BL-9558; SF-BL-9822
    for the :func:`calibration_bins` funnel): the
    scorecard must fail on a mismatched panel itself, before any
    component scorer is reached.

    Raises ValueError naming the empty / all-non-finite panel if
    the panel is empty or contains no fully finite observation
    (SF-BL-10742; see :func:`_validate_evaluate_panel`). Previously
    ``evaluate([], [])`` returned a fabricated mixed verdict —
    ECE/MCE/reliability 0.0 alongside NaN Brier/log score.

    Raises ValueError naming ``y`` / ``p`` if any element of either
    is a boolean (SF-BL-10938; see :func:`_reject_bools`).
    Previously ``evaluate([1], [True])`` returned a perfect
    scorecard (Brier 0.0, ECE 0.0).

    Raises ValueError naming ``y`` / ``p`` if any element of either
    is a string (SF-BL-11052; see :func:`_reject_strings`).
    Previously ``evaluate([1, 0], ['0.5', '0.5'])`` returned a full
    scorecard (Brier 0.25, log score 0.6931).
    """
    n_bins = _validate_n_bins(n_bins)
    _reject_bools(y, "y")  # SF-BL-10938: raw screen, before any conversion
    _reject_bools(p, "p")  # SF-BL-10938
    _reject_strings(y, "y")  # SF-BL-11052: raw screen, before any conversion
    _reject_strings(p, "p")  # SF-BL-11052
    _require_same_length(y, p)  # SF-BL-9558 / SF-BL-9822: explicit, not inherited
    _validate_evaluate_panel(y, p)  # SF-BL-10742: empty / all-non-finite
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
    _reject_bool_pool_members(probs)  # SF-BL-10975: raw screen, before float conversion coerces True->1.0
    _reject_string_pool_members(probs)  # SF-BL-11125: raw screen, before float conversion parses '0.7'->0.7
    P = np.asarray(probs, dtype=float)
    if P.ndim == 1:
        P = P[:, None]
    if P.ndim != 2:
        raise ValueError("probs must be (N,) or (N, M)")
    _require_finite(P, "probs")  # SF-BL-3596
    return P


def _validate_pool_members_range(P):
    """Raise ValueError naming ``probs`` if any *finite* element of a
    pool panel lies outside [0, 1] (SF-BL-10934).

    The three pool primitives never screened this case:
    ``linear_pool`` clipped the pooled mean, while
    ``geometric_pool`` / ``log_odds_pool`` routed members through
    :func:`_as_p`, whose clip silently mapped 1.5 to the 1-1e-7
    endpoint and 82.0 likewise, so a percent-unit or otherwise
    impossible member became the most extreme lawful forecast in
    the pool (``linear_pool([[1.5, 0.5]])`` returned ``[1.0]``;
    ``geometric_pool([[82.0, 0.5]])`` returned ``0.99968``). The
    SF-BL-3596 finiteness screen, the SF-BL-9762 empty-panel work
    and the SF-BL-10880 screen inside ``optimize_pool_weights`` do
    not claim a finite out-of-range member passed directly to a
    pool primitive.

    Non-finite handling is UNCHANGED by design: this helper ignores
    non-finite values entirely, and every pool runs
    :func:`_require_finite` (SF-BL-3596, via ``_as_matrix``) BEFORE
    calling it, so NaN / +/-inf -- alone or mixed with a finite
    out-of-range value -- still raise exactly the pre-existing
    finiteness ValueError. Pure validation -- it never transforms
    ``P`` -- so lawful panels (endpoints 0/1 included) are
    bit-identical to the pre-fix computation, and empty panels are
    a no-op (their pre-existing behaviour is unchanged).
    """
    arr = np.asarray(P, dtype=float)
    if arr.size == 0:
        return
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return
    lo, hi = float(finite.min()), float(finite.max())
    if lo < 0.0 or hi > 1.0:
        raise ValueError(
            f"probs must lie in [0, 1]; got finite value(s) outside that "
            f"range in probs (min={lo!r}, max={hi!r}) (SF-BL-10934)"
        )




def _has_string(raw) -> bool:
    """True iff ``raw`` contains any string value (SF-BL-11090).

    Recurses through lists / tuples and object-dtype ndarrays, and
    recognises string-dtype ndarrays (dtype kind ``U`` / ``S``)
    directly. A plain numeric or bool ndarray cannot contain strings
    by construction, so it returns False without scanning.
    """
    if isinstance(raw, (str, bytes)):
        return True
    if isinstance(raw, np.ndarray):
        if raw.dtype.kind in "US":
            return True
        if raw.dtype == object:
            return any(_has_string(v) for v in raw.ravel())
        return False
    if isinstance(raw, (list, tuple)):
        return any(_has_string(v) for v in raw)
    return False


def _reject_string_weights(raw):
    """Raise a named ValueError if pool ``weights`` contain any
    string value (SF-BL-11090).

    This screen MUST run on the raw weights, before any
    ``np.asarray(..., dtype=float)`` conversion: float conversion
    silently reads a numeric string as the number it spells, so
    after conversion a weight serialised as text in a pool config
    (``weights=['0.5', '0.5']``) is indistinguishable from a lawful
    numeric weight and ``linear_pool`` / ``geometric_pool`` /
    ``log_odds_pool`` on ``[[0.7, 0.3]]`` all returned 0.5 — a
    silently honoured text config instead of a loud failure at the
    pooling boundary. The SF-BL-10883 bool-weight fixtures and the
    SF-BL-3596 non-finite screen do not claim string weights: the
    finiteness screen runs on the already-converted vector, where
    ``'0.5'`` has already become 0.5. Pure validation — it never
    transforms ``raw`` — so lawful numeric weight vectors (and
    ``None``, and bool weights, whose behaviour is unchanged) are
    bit-identical to the pre-fix computation.
    """
    if _has_string(raw):
        raise ValueError(
            "weights must not contain string values; got string "
            "value(s) in weights (a string is not a valid weights "
            "value) (SF-BL-11090)"
        )


def _norm_weights(w, m):
    if w is None:
        return np.full(m, 1.0 / m)
    _reject_string_weights(w)  # SF-BL-11090: raw screen, before float conversion reads '0.5' as 0.5
    w = np.asarray(w, dtype=float).ravel()
    # SF-BL-3596: finiteness FIRST — NaN passes both checks below
    # (comparisons are False) and an Inf weight normalises to NaN.
    _require_finite(w, "weights")
    if len(w) != m or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative, length M, with positive sum")
    return w / w.sum()


def linear_pool(probs, weights=None):
    """Weighted arithmetic mean of probabilities.

    Raises
    ------
    ValueError
        If any member of ``probs`` is a boolean (SF-BL-10975; see
        :func:`_reject_bool_pool_members`). Previously
        ``_as_matrix`` coerced True/False to 1.0/0.0 and
        ``linear_pool([[True, False]])`` returned ``[0.5]``. The
        screen runs on the raw input inside ``_as_matrix``, before
        any float conversion.
    ValueError
        If any finite member of ``probs`` lies outside [0, 1]
        (SF-BL-10934; see :func:`_validate_pool_members_range`).
        Previously the output clip silently mapped a member of 1.5
        to a pooled 1.0. Non-finite members raise the pre-existing
        SF-BL-3596 finiteness ValueError, which runs first.
    ValueError
    ValueError
        If any member of ``probs`` is a string (SF-BL-11125; see
        :func:`_reject_string_pool_members`). Previously
        ``_as_matrix`` parsed '0.7' as 0.7 and this pool returned
        ``[0.5]`` on ``[['0.7', '0.3']]``, bit-identical to the
        numeric control. The screen runs on the raw input inside
        ``_as_matrix``, before any float conversion.
        If ``weights`` contains any string value (SF-BL-11090; see
        :func:`_reject_string_weights`). Previously
        ``_norm_weights`` coerced ``weights=['0.5', '0.5']`` through
        ``np.asarray(..., dtype=float)`` and this pool returned
        ``[0.5]`` on ``[[0.7, 0.3]]``. The screen runs on the raw
        weights inside ``_norm_weights``, before any float
        conversion.
    """
    P = _as_matrix(probs)
    _validate_pool_members_range(P)  # SF-BL-10934: before any pooling
    w = _norm_weights(weights, P.shape[1])
    return np.clip(P @ w, 0.0, 1.0)


def geometric_pool(probs, weights=None):
    """
    Weighted geometric mean, renormalised for the binary case:
        p ∝ prod p_i^w_i ,  (1-p) ∝ prod (1-p_i)^w_i
    Equivalent to an (un-extremized) log-odds pool.

    Raises
    ------
    ValueError
        If any member of ``probs`` is a boolean (SF-BL-10975; see
        :func:`_reject_bool_pool_members`). Previously
        ``_as_matrix`` coerced True to 1.0 and this pool returned
        0.99968 on ``[[True, 0.5]]`` — the kill repro. The screen
        runs on the raw input inside ``_as_matrix``, before any
        float conversion.
    ValueError
        If any finite member of ``probs`` lies outside [0, 1]
        (SF-BL-10934; see :func:`_validate_pool_members_range`).
        Previously ``_as_p`` clipped a member of 82.0 to the
        1-1e-7 endpoint and the pool returned 0.99968. Non-finite
        members raise the pre-existing SF-BL-3596 finiteness
        ValueError, which runs first.
    ValueError
    ValueError
        If any member of ``probs`` is a string (SF-BL-11125; see
        :func:`_reject_string_pool_members`). Previously
        ``_as_matrix`` parsed '0.7' as 0.7 and this pool returned
        ``[0.5]`` on ``[['0.7', '0.3']]``, bit-identical to the
        numeric control. The screen runs on the raw input inside
        ``_as_matrix``, before any float conversion.
        If ``weights`` contains any string value (SF-BL-11090; see
        :func:`_reject_string_weights`). Previously
        ``_norm_weights`` coerced ``weights=['0.5', '0.5']`` through
        ``np.asarray(..., dtype=float)`` and this pool returned
        ``[0.5]`` on ``[[0.7, 0.3]]``. The screen runs on the raw
        weights inside ``_norm_weights``, before any float
        conversion.
    """
    P = _as_matrix(probs)
    _validate_pool_members_range(P)  # SF-BL-10934: before _as_p clips
    w = _norm_weights(weights, P.shape[1])
    Pc = _as_p(P)
    log_p = (np.log(Pc) * w).sum(axis=1)
    log_q = (np.log(1 - Pc) * w).sum(axis=1)
    return expit(log_p - log_q)


def _validate_extremization(extremization) -> float:
    """Validate ``log_odds_pool``'s extremization factor (SF-BL-9513).

    Returns the factor as a float. Pure validation — a valid factor is
    returned unchanged in value, so pooled outputs for finite factors
    are bit-identical to the pre-fix computation.

    Raises
    ------
    ValueError
        If ``extremization`` is non-finite (NaN or +/-inf), is not
        positive (<= 0), or is not a real scalar at all (bool, string,
        None, a sequence, or any value ``float()`` cannot read as a
        real scalar). The original guard was only ``extremization
        <= 0``: NaN comparisons are all False, so a=NaN pooled to NaN
        silently, and a=inf saturated the pool to a certain 0.0 on a
        [0.7, 0.3] panel (expit of -inf — maximal certainty in the
        wrong tail) and to 1.0 on a [0.7] panel. SF-BL-3596 added a
        bare ``np.isfinite`` screen for numeric factors; this helper
        completes it — one documented ValueError naming the factor
        for every invalid value, raised before any pooling.
        SF-BL-11053 is the type dimension of the same guard (distinct
        from the finiteness dimension above): the type screen is
        explicit, not a side effect of ``float()`` — Python ``bool``,
        ``np.bool_``, 0-d bool arrays and complex factors are
        rejected by isinstance checks, because ``float()`` silently
        reads ``True`` / ``np.bool_(True)`` / ``np.array(True)`` as
        1.0 (a=True pooled ``[[0.7, 0.6]]`` to 0.6517, bit-identical
        to a=1.0, vs 0.7778 at a=2.0) and reads ``np.complex128(2)``
        as 2.0, while a string died with a bare ``TypeError`` from
        the original ``<=`` comparison instead of this named
        ``ValueError``. Round 2 (SF-BL-11053): the screen also
        rejects the remaining bytes-like carriers — ``bytearray``
        and ``memoryview``, which ``float()`` likewise decoded as
        ASCII digits (``bytearray(b"2")`` pooled as a=2.0) — and,
        categorically, any other buffer-protocol carrier, instead
        of enumerating buffer types one by one.
    """
    msg = ("extremization must be finite and > 0 "
           "(SF-BL-9513; type screen SF-BL-11053); "
           f"got extremization={extremization!r}")
    if isinstance(extremization, (bool, np.bool_)) or isinstance(
            extremization, (str, bytes, bytearray, memoryview)):
        raise ValueError(msg)
    if isinstance(extremization, (complex, np.complexfloating)):
        raise ValueError(msg)
    if isinstance(extremization, np.ndarray):
        if extremization.ndim > 0:
            raise ValueError(msg)
        _item = extremization.item()
        if isinstance(_item, (bool, np.bool_, str, bytes, bytearray,
                              memoryview, complex, np.complexfloating)):
            raise ValueError(msg)
        if not isinstance(_item, (int, float, np.floating, np.integer)):
            try:
                memoryview(_item)
            except (TypeError, ValueError, BufferError):
                pass
            else:
                raise ValueError(msg)
    if not isinstance(extremization, (np.floating, np.integer,
                                      np.ndarray)):
        # Any remaining buffer-protocol carrier (stdlib array('B'),
        # mmap, ctypes buffers, ...) is non-numeric: float() would
        # decode its raw bytes as an ASCII number, so reject it
        # categorically instead of enumerating buffer types
        # (SF-BL-11053 round 2). The lawful buffer-supporting
        # scalars (np.floating / np.integer) and 0-d arrays
        # (screened above) are exempt.
        try:
            memoryview(extremization)
        except (TypeError, ValueError, BufferError):
            pass
        else:
            raise ValueError(msg)
    try:
        a = float(extremization)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(msg) from None
    if not np.isfinite(a) or a <= 0:
        raise ValueError(msg)
    return a


def log_odds_pool(probs, weights=None, extremization: float = 1.0):
    """
    Extremized log-odds (logit) pool (Satopää et al.):

        logit(p_pool) = a * sum_i w_i * logit(p_i)

    a = 1 recovers the geometric pool. a > 1 extremizes — pushes the pooled
    forecast away from 0.5 — which corrects the under-confidence that comes
    from forecasters sharing information / each hedging toward the base rate.
    a < 1 de-extremizes (shrinks toward 0.5).

    Raises
    ------
    ValueError
        If ``extremization`` is non-finite or non-positive
        (SF-BL-9513; see :func:`_validate_extremization`). Previously
        a=NaN returned NaN with no error and a=inf collapsed a
        [0.7, 0.3] panel to a certain 0.0.
    ValueError
        If ``extremization`` is a bool or non-numeric value
        (SF-BL-11053, the type dimension; see
        :func:`_validate_extremization`). Previously the only screen
        was ``extremization <= 0``, which a bool silently satisfies
        as 1.0 — a=True pooled [[0.7, 0.6]] to 0.6517, bit-identical
        to a=1.0 (vs 0.7778 at a=2.0) — and a string such as '2'
        died with a bare TypeError from the ``<=`` comparison
        instead of this named ValueError.
    ValueError
        If any member of ``probs`` is a boolean (SF-BL-10975; see
        :func:`_reject_bool_pool_members`). Previously
        ``_as_matrix`` coerced True to 1.0 and this pool returned
        0.99968 on ``[[True, 0.5]]`` — the kill repro. The screen
        runs on the raw input inside ``_as_matrix``, before any
        float conversion; the extremization screen above keeps its
        pre-existing precedence over it.
    ValueError
        If any finite member of ``probs`` lies outside [0, 1]
        (SF-BL-10934; see :func:`_validate_pool_members_range`).
        Previously ``_as_p`` clipped a member of 82.0 to the
        1-1e-7 endpoint and the pool returned 0.99968. The
        extremization screen above keeps its pre-existing
        precedence; non-finite members raise the pre-existing
        SF-BL-3596 finiteness ValueError, which runs first inside
        ``_as_matrix``.
    ValueError
    ValueError
        If any member of ``probs`` is a string (SF-BL-11125; see
        :func:`_reject_string_pool_members`). Previously
        ``_as_matrix`` parsed '0.7' as 0.7 and this pool returned
        ``[0.5]`` on ``[['0.7', '0.3']]``, bit-identical to the
        numeric control. The screen runs on the raw input inside
        ``_as_matrix``, before any float conversion.
        The extremization screen above keeps its pre-existing
        precedence over it.
        If ``weights`` contains any string value (SF-BL-11090; see
        :func:`_reject_string_weights`). Previously
        ``_norm_weights`` coerced ``weights=['0.5', '0.5']`` through
        ``np.asarray(..., dtype=float)`` and this pool returned
        ``[0.5]`` on ``[[0.7, 0.3]]``. The screen runs on the raw
        weights inside ``_norm_weights``, before any float
        conversion; the extremization screen above keeps its
        pre-existing precedence over it.
    """
    a = _validate_extremization(extremization)
    P = _as_matrix(probs)
    _validate_pool_members_range(P)  # SF-BL-10934: before _as_p clips
    w = _norm_weights(weights, P.shape[1])
    pooled_logit = logit(_as_p(P)) @ w
    return expit(a * pooled_logit)


# Alias matching the literature's name
extremized_pool = log_odds_pool


# --------------------------------------------------------------------------
# Minimum-n guard for optimize_pool_weights (SF-BL-9443)
#
# The sibling learners enforce sample floors — a4's learn_extremization
# requires >= 150 resolved (EXTREMIZE_MIN_RESOLVED) and the a4
# calibration gate requires >= 20 — but this research pooling path
# enforced none: on n=2 resolved rows it returned success with weights
# [1.0, ~0] and extremization pinned at the 4.0 a_bounds cap, a fit no
# two rows can identify. Below POOL_MIN_N resolved rows the function
# therefore returns an explicit insufficient-n result (see
# optimize_pool_weights) instead of fitted weights / a. The floor is
# pre-registered at 50: above the a4 gate's 20 (pool weights + a joint
# fit estimates strictly more than a gate statistic), below a4's 150
# (a research fitter may legitimately run on smaller validation panels
# than a production extremization adoption, and the forward-book audit
# for this task found in-tree callers refitting at n=50-150). A caller
# may override it via the ``min_n`` keyword; the default is the guard.
# --------------------------------------------------------------------------

POOL_MIN_N = 50

# --------------------------------------------------------------------------
# Degenerate-panel + small-n extremization cap (SF-BL-10801)
#
# Mirrors engine.py's production pair (Superforecaster.EXTREMIZE_CAP /
# EXTREMIZE_MIN_RESOLVED, SF-BL-8875): a learned extremization above
# 1.40 is a production adoption only after >= 150 resolved questions.
# This research fitter previously ignored both facts: on a
# single-forecaster panel (simplex weight degenerate at [1.0], so no
# pool exists to extremize) it fitted a to the 4.0 a_bounds cap, and
# on any panel below 150 rows it could return a > 1.40. The fix in
# optimize_pool_weights: m == 1 returns a = 1.0 with an explicit
# degenerate_panel flag and no fitted a; below 150 resolved rows the
# fitted a is capped at EXTREMIZE_CAP. Kill-audited (see this task's
# KILL_AUDIT.md): the uncapped single-member fit does NOT beat
# a = 1.0 on held-out log score for a calibrated forecaster.
# --------------------------------------------------------------------------

EXTREMIZE_CAP = 1.40
EXTREMIZE_MIN_RESOLVED = 150


def _validate_a_bounds(a_bounds):
    """Validate ``optimize_pool_weights``' ``a_bounds`` (SF-BL-9616).

    Returns the bounds as a ``(lo, hi)`` pair of plain floats. Pure
    validation — a valid pair is returned unchanged in value, so
    ``np.log`` of the bounds, the SLSQP search interval and every
    valid fit are bit-identical to the pre-fix computation.

    Raises
    ------
    ValueError
        Unless ``a_bounds`` is a pair of finite real scalars with
        ``0 < lo <= hi``. Previously the bounds were never validated:
        they entered the optimiser only as ``np.log(a_bounds[0])`` /
        ``np.log(a_bounds[1])`` SLSQP bounds, so ``(0, 1)`` logged to
        ``-inf``, ``(-1, 2)`` to NaN (with only a RuntimeWarning) and
        ``(NaN, 2)`` to NaN, and each silently returned
        ``success=True`` with the fitted extremization pinned at the
        one surviving bound (1.0, 2.0, 2.0) — a caller typo silently
        changed the pooled extremization. Values that cannot be read
        as a pair of real scalars at all (bool, string, None, wrong
        arity, non-numeric entries) raise the same documented
        ValueError instead of a bare TypeError from inside ``np.log``.
        ``lo == hi`` is lawful (``0 < lo <= hi``): a degenerate
        interval that pins ``a`` to a single value.

        Type completion (SF-BL-11089): the SF-BL-9616 screen tested
        only the Python forms of those classes, so the numpy forms
        still slipped through — ``np.bool_(True)`` is not a ``bool``
        instance and ``float()`` read it as 1.0 (a fitted fit, the
        TASK's bool repro in numpy dress), a 0-d ndarray evaded the
        ``ndim > 0`` check (``np.array(True)`` -> 1.0,
        ``np.array('0.3')`` -> 0.3), a bool-dtype pair unpacks to
        ``np.bool_`` entries, and ``float(np.complex128(0.3+2j))``
        silently discarded the imaginary part where
        ``float(0.3+2j)`` raises. Every bool or non-numeric entry —
        Python or numpy form — now raises the same documented
        ValueError naming ``a_bounds``. Lawful numeric entries
        (ints, floats, numpy real scalars, 0-d real ndarrays,
        ``Decimal`` / ``Fraction``) are unchanged in value.
    """
    msg = ("a_bounds must be finite with 0 < lo <= hi (SF-BL-9616); "
           f"got a_bounds={a_bounds!r}")
    if isinstance(a_bounds, (str, bytes)):
        raise ValueError(msg)
    try:
        lo_raw, hi_raw = a_bounds
    except (TypeError, ValueError):
        raise ValueError(msg) from None
    pair = []
    for v in (lo_raw, hi_raw):
        # SF-BL-11089: numpy-form bool / string / complex entries —
        # see the docstring. The entry screen also covers pair
        # containers of those dtypes, which unpack to np.bool_ /
        # np.str_ / np.complexfloating entries.
        if isinstance(v, (bool, np.bool_)) or isinstance(v, (str, bytes)):
            raise ValueError(msg)
        if isinstance(v, (complex, np.complexfloating)):
            raise ValueError(msg)
        if isinstance(v, np.ndarray):
            if v.ndim > 0:
                raise ValueError(msg)
            if v.dtype == bool or v.dtype.kind in ("U", "S"):
                raise ValueError(msg)
            if np.issubdtype(v.dtype, np.complexfloating):
                raise ValueError(msg)
            if v.dtype == object and isinstance(
                    v.item(), (bool, np.bool_, str, bytes, complex,
                               np.complexfloating)):
                raise ValueError(msg)
        try:
            pair.append(float(v))
        except (TypeError, ValueError):
            raise ValueError(msg) from None
    lo, hi = pair
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0.0 or lo > hi:
        raise ValueError(msg)
    return lo, hi


def _validate_pool_panel_range(P):
    """Raise ValueError naming ``probs`` if any *finite* element of an
    ``optimize_pool_weights`` panel lies outside [0, 1] (SF-BL-10880).

    The function never screened this case: after the SF-BL-9686
    finiteness screen the panel went straight to ``Pc = _as_p(P)``,
    whose clip silently mapped 1.5 to the 1-1e-7 endpoint and -0.5 to
    1e-7, so an impossible forecast became a near-certain training row.
    On the filed repro (y=[1,1,0,0], member column 1.5 on the positive
    rows) the extremized and geometric fits assigned the corrupt member
    weight 1.0 and the extremized fit reported log_score 0.0019 /
    brier ~0.0 — a flawless-looking scorecard with which a corrupt
    member can capture a forward pool. The SF-BL-9592 empty/length
    screens and the SF-BL-9686 finiteness screen do not claim this
    finite out-of-range case.

    Non-finite handling is UNCHANGED by design: this helper ignores
    non-finite values entirely, and ``optimize_pool_weights`` runs
    :func:`_require_finite` (SF-BL-9686) BEFORE calling it, so NaN /
    +/-inf — alone or mixed with a finite out-of-range value — still
    raise exactly the pre-existing finiteness ValueError. Pure
    validation — it never transforms ``P`` — so lawful panels (endpoints
    0/1 included) return bit-identical weights and scores.
    """
    arr = np.asarray(P, dtype=float)
    if arr.size == 0:
        return
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return
    lo, hi = float(finite.min()), float(finite.max())
    if lo < 0.0 or hi > 1.0:
        raise ValueError(
            f"probs must lie in [0, 1]; got finite value(s) outside that "
            f"range in probs (min={lo!r}, max={hi!r}) (SF-BL-10880)"
        )


def _validate_pool_panel_no_bool(probs):
    """Raise ValueError naming ``probs`` if the RAW
    ``optimize_pool_weights`` panel contains any boolean value
    (SF-BL-10976).

    The screen must run on the raw input, before any
    ``np.asarray(..., dtype=float)`` conversion: float conversion
    silently maps True -> 1.0 and False -> 0.0, so after conversion a
    leaked boolean column is indistinguishable from a lawful endpoint
    forecaster. Pre-fix, ``_as_p`` clipped the converted member to the
    1-1e-7 / 1e-7 endpoints, which exactly matched the outcomes on the
    filed repro (first member [True, False, True, False] against
    y=[1, 0, 1, 0]): the fits returned dominant positive weight on the
    bool member with log_score ~1e-10 and Brier ~1e-21 — a flawless-
    looking scorecard with which a leaked outcome column captures a
    fitted forward pool. A bool is not a probability (in Python it is
    an int subclass, which is exactly why it slips past the numeric
    screens: the SF-BL-9686 finiteness screen and the SF-BL-10880
    range screen both see only the converted 1.0/0.0 endpoints).

    Detection recurses through lists / tuples and ndarrays: any leaf
    that is a Python ``bool`` or ``np.bool_``, any ndarray with dtype
    ``bool``, and any such leaf inside an object-dtype ndarray counts.
    A plain numeric ndarray cannot contain bools by construction and
    returns without an elementwise scan. This screen is explicit in
    ``optimize_pool_weights`` itself — not merely inherited from the
    SF-BL-10975 screen inside ``_as_matrix`` (which guards the pool
    primitives) — so a shared-helper change must never silently
    reopen this path (the SF-BL-9686 discipline). Pure validation —
    it never transforms ``probs`` — so lawful panels (int 0/1 and
    float endpoints included; ints are not bools) return
    bit-identical weights and scores.
    """
    def _contains_bool(raw):
        if isinstance(raw, (bool, np.bool_)):
            return True
        if isinstance(raw, np.ndarray):
            if raw.dtype == bool:
                return True
            if raw.dtype == object:
                return any(_contains_bool(v) for v in raw.ravel())
            return False
        if isinstance(raw, (list, tuple)):
            return any(_contains_bool(v) for v in raw)
        return False

    if _contains_bool(probs):
        raise ValueError(
            "probs must not contain boolean values; got boolean "
            "value(s) in probs — a bool is not a probability "
            "(SF-BL-10976)"
        )


def _validate_pool_seed(seed):
    """Validate ``optimize_pool_weights``' ``seed`` (SF-BL-11088).

    Returns the seed unchanged. Pure validation — a lawful seed is
    passed to ``np.random.default_rng`` exactly as the caller gave
    it, so integer-seed fits (weights, extremization, success) are
    bit-identical to the pre-fix computation.

    Raises
    ------
    ValueError
        Unless ``seed`` is a non-negative integer (Python ``int`` or
        ``np.integer``). Previously the seed was never validated: it
        fed ``np.random.default_rng`` for the Dirichlet restart with
        no type screen, so a string seed from a config file
        (``seed='0'``) aborted the fit with a bare SeedSequence
        ``TypeError`` that never named ``seed``, a float seed died
        the same unnamed way, ``seed=None`` / a sequence seed
        silently drew fresh / sequence entropy, and ``seed=True``
        silently optimised with entropy 1 (bool is an int subclass)
        — on the filed 4-row repro returning weights [1.0, 0.0] and
        extremization 4.0 — a different reproducibility contract
        than the caller recorded. A negative integer reached numpy's
        own ValueError, which likewise never named ``seed``. Bools
        (``bool`` / ``np.bool_``), non-integers (string, bytes,
        float, None, sequence, ndarray, SeedSequence / Generator
        objects) and negative integers all raise this one documented
        ValueError naming ``seed`` instead.
    """
    msg = ("seed must be a non-negative integer (SF-BL-11088); "
           f"got seed={seed!r}")
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError(msg)
    if not isinstance(seed, (int, np.integer)):
        raise ValueError(msg)
    if seed < 0:
        raise ValueError(msg)
    return seed


def _validate_fit_extremization(fit_extremization):
    """Validate ``optimize_pool_weights``' ``fit_extremization`` flag
    (SF-BL-11062).

    Pure validation — it returns nothing and never transforms the
    flag, so ``True`` / ``False`` fits are bit-identical to the
    pre-fix computation.

    Raises
    ------
    ValueError
        Unless ``fit_extremization`` is a bool (Python ``bool`` or
        ``np.bool_``). Previously the flag was consumed by truthiness
        alone — ``use_a = (method == "extremized" and
        fit_extremization)`` — with no type check: on the filed
        repro the string ``'no'`` was truthy, so the joint
        extremization fit the caller appeared to decline ran anyway
        and returned ``success=True`` with the fitted extremization
        pinned at the 4.0 ``a_bounds`` cap, bit-for-bit the ``True``
        result; ``'yes'``, ``1`` and ``1.0`` did the same, while
        ``''``, ``0`` and ``None`` silently acted as ``False``. A
        control flag read from a config / CLI layer routinely
        arrives as a string or int, so a declined fit must never be
        re-enabled by truthiness. Int ``0`` / ``1`` raise even though
        ``bool`` subclasses ``int``: ``isinstance(1, bool)`` is
        False, and an int is not a bool.
    """
    if isinstance(fit_extremization, (bool, np.bool_)):
        return
    raise ValueError(
        "fit_extremization must be a bool (True or False) "
        f"(SF-BL-11062); got fit_extremization={fit_extremization!r}"
    )

def _validate_pool_y_no_bool(y):
    """Raise ValueError naming ``y`` if the RAW
    ``optimize_pool_weights`` outcome vector contains any boolean
    value (SF-BL-11016).

    The screen must run on the raw input, before any
    ``np.asarray(..., dtype=float)`` conversion: float conversion
    silently maps True -> 1.0 and False -> 0.0, so after conversion
    a boolean flag is indistinguishable from a lawful 0/1 outcome
    and ``_as_y``'s ``==0/==1`` check passes it. Pre-fix, on the
    filed repro (y=[True, False, True, False] against a near-perfect
    first member) the fits returned weights [1.0, ~0.0] with
    log_score 0.00015 and success=True — a flawless-looking fit to
    boolean flags treated as lawful resolutions. A bool is not a
    0/1 outcome (in Python it is an int subclass, which is exactly
    why it slips past the numeric screens: the finiteness and
    binary checks both see only the converted 1.0/0.0 endpoints).

    Detection is :func:`_has_bool` (Python ``bool`` / ``np.bool_``
    leaves, bool-dtype ndarrays, object-dtype contents, recursing
    through lists / tuples) plus
    :func:`_has_bool_in_pandas_panel` (pandas Series / Index /
    Categorical panels with a bool / boolean dtype, bool categories,
    or bool leaves — containers ``_has_bool`` alone returns False
    for). This screen is explicit in ``optimize_pool_weights``
    itself — not merely inherited from any ``_as_y`` screen (the
    repo copy's SF-BL-10938 ``_as_y`` screen; the out copy's
    ``_as_y`` has no bool screen at all) — so a shared-helper change
    must never silently reopen this path (the SF-BL-9686
    discipline). Pure validation — it never transforms ``y`` — so
    lawful panels (int 0/1 and float endpoints included; ints are
    not bools) return bit-identical weights and scores.
    """
    if _has_bool(y) or _has_bool_in_pandas_panel(y):
        raise ValueError(
            "y must not contain boolean values; got boolean "
            "value(s) in y \u2014 a bool is not a valid outcome "
            "(SF-BL-11016)"
        )


def _pandas_panel_has_bool(raw) -> bool:
    """True iff a pandas panel contains any boolean value
    (SF-BL-11427 backstop).

    :func:`_has_bool_in_pandas_panel` covers Series / Index /
    Categorical but returns False for a DataFrame by design, so a
    bool ``probs`` DataFrame (or a bool ``y`` frame) slipped past
    it. This helper adds the DataFrame case: a bool / boolean
    dtype column, or any bool leaf in the frame's values. Any
    inspection error returns False.
    """
    try:
        if not (type(raw).__module__ or "").startswith("pandas"):
            return False
        if type(raw).__name__ == "DataFrame":
            for dt in list(raw.dtypes):
                if dt == bool or getattr(dt, "name", None) in ("bool", "boolean"):
                    return True
            return _has_bool(raw.values)
        return _has_bool_in_pandas_panel(raw)
    except Exception:  # noqa: BLE001 - inspection must never break lawful panels
        return False


def _pandas_panel_has_string_or_bare(raw) -> bool:
    """True iff a pandas panel contains any string or bare-object
    value (SF-BL-11427).

    :func:`_has_string_in_pandas_panel` covers Series / Index /
    Categorical but returns False for a DataFrame by design, and it
    does not see bare ``object()`` leaves. This helper adds both:
    for a DataFrame, a string dtype column or any string / bare
    leaf in the frame's values; otherwise the shared string scan
    plus a bare-object scan of the panel's values. Any inspection
    error returns False.
    """
    try:
        if not (type(raw).__module__ or "").startswith("pandas"):
            return False
        if type(raw).__name__ == "DataFrame":
            for dt in list(raw.dtypes):
                if getattr(dt, "kind", None) in ("U", "S") or getattr(dt, "name", None) == "string":
                    return True
            return _has_string(raw.values) or _has_bare_object(raw.values)
        if _has_string_in_pandas_panel(raw):
            return True
        return any(_has_bare_object(v) for v in list(raw))
    except Exception:  # noqa: BLE001 - inspection must never break lawful panels
        return False


def _validate_pool_y_panel_types(y):
    """Raise ValueError naming ``y`` if the RAW
    ``optimize_pool_weights`` outcome panel contains any string,
    object-dtype (bare-object) or boolean element (SF-BL-11427).

    This function never had its OWN complete type screen for its
    ``y`` / ``probs`` panels: it relied on screens inherited from
    shared helpers and on the bool-only SF-BL-11016 screen. The
    inherited screens are incomplete and differ between copies --
    the out copy's ``_as_y`` has no string screen at all, so
    ``optimize_pool_weights(['1', '0'], [[0.9, 0.2], [0.8, 0.3]])``
    returned the numeric control's fitted weights
    [0.90003379, 0.09996621] there -- and neither copy's helpers see
    pandas containers (``_has_string`` / ``_has_bool`` return False
    for any non-list/tuple/ndarray container, so pandas string
    ``y`` panels returned the same weights in both copies), while a
    bare ``object()`` element died with a bare ``TypeError`` from
    ``float()`` that never named ``y``. This screen is explicit in
    ``optimize_pool_weights`` itself, runs on the RAW ``y`` after
    the SF-BL-11016 bool screen (whose message is preserved for the
    cases it already catches) and before ``_as_y``'s float
    conversion, the minimum-n early return and any SLSQP work.
    Detection is :func:`_has_bool` / :func:`_has_string` /
    :func:`_has_bare_object` plus the pandas backstops
    :func:`_pandas_panel_has_bool` /
    :func:`_pandas_panel_has_string_or_bare`. Pure validation -- it
    never transforms ``y`` -- so lawful panels (float/int, numeric
    ndarrays, numeric object-dtype arrays, numeric pandas panels)
    return bit-identical weights, extremization and scores.
    """
    if _has_bool(y) or _pandas_panel_has_bool(y):
        raise ValueError(
            "y must not contain boolean values; got boolean "
            "value(s) in y (a bool is not a valid outcome) "
            "(SF-BL-11427)"
        )
    if _has_string(y) or _has_bare_object(y) or _pandas_panel_has_string_or_bare(y):
        raise ValueError(
            "y must not contain string values; got string or "
            "object-dtype value(s) in y (a string/object is not "
            "a valid y value) (SF-BL-11427)"
        )


def _validate_pool_probs_panel_types(probs):
    """Raise ValueError naming ``probs`` if the RAW
    ``optimize_pool_weights`` member panel contains any string,
    object-dtype (bare-object) or boolean element (SF-BL-11427).

    Mirror of :func:`_validate_pool_y_panel_types` for the member
    panel: the SF-BL-10976 bool screen and the ``_as_matrix``
    string screen (SF-BL-11125) cover list / tuple / ndarray
    panels, but neither sees a pandas DataFrame, so a string or
    bool DataFrame panel was parsed / coerced by
    ``np.asarray(..., dtype=float)`` and fitted -- a string
    DataFrame returned the numeric control's weights
    [0.90003379, 0.09996621] and a bool DataFrame returned a
    flawless-looking fit -- and a bare ``object()`` member died
    with a bare ``TypeError`` that never named ``probs``. This
    screen is explicit in ``optimize_pool_weights`` itself, runs
    on the RAW ``probs`` after the SF-BL-10976 bool screen (whose
    message is preserved for the cases it already catches) and
    before ``_as_matrix``'s float conversion, the minimum-n early
    return and any SLSQP work. Pure validation -- it never
    transforms ``probs`` -- so lawful panels are bit-identical.
    """
    if _has_bool(probs) or _pandas_panel_has_bool(probs):
        raise ValueError(
            "probs must not contain boolean values; got boolean "
            "value(s) in probs (a bool is not a valid probs member) "
            "(SF-BL-11427)"
        )
    if _has_string(probs) or _has_bare_object(probs) or _pandas_panel_has_string_or_bare(probs):
        raise ValueError(
            "probs must not contain string values; got string or "
            "object-dtype value(s) in probs (a string/object is "
            "not a valid probs member) (SF-BL-11427)"
        )

def optimize_pool_weights(y, probs, method: str = "extremized",
                          fit_extremization: bool = True,
                          a_bounds=(0.3, 4.0), seed: int = 0,
                          min_n: int = POOL_MIN_N):
    """
    Learn pool weights (and, for method='extremized', the factor a) by
    minimising log score on validation data (y, probs).

    method: 'linear' | 'geometric' | 'extremized'
    Returns dict: weights (M,), extremization (float or None),
                  log_score, brier, pool_fn (callable probs -> pooled p),
                  bound_limited (bool), a_bound ('upper'/'lower'/None).

    Bound disclosure (SF-BL-9443): when extremization is fitted and the
    fitted ``a`` lands on an ``a_bounds`` edge (within
    1e-6 * max(1, |bound|)), the fit is disclosed as bound-limited —
    ``bound_limited`` is True and ``a_bound`` names the edge — because
    such an ``a`` is the cap, not an estimate: the uncapped optimum
    lies outside the searched interval. Non-extremized fits and
    interior fits report ``bound_limited=False, a_bound=None``.

    Minimum-n guard (SF-BL-9443): if the panel has fewer than
    ``min_n`` resolved rows (default ``POOL_MIN_N`` = 50), NO fit is
    attempted and the returned dict is the explicit insufficient-n
    result ``{"weights": None, "extremization": None, "log_score":
    None, "brier": None, "pooled": None, "pool_fn": None,
    "success": False, "status": "insufficient_n",
    "insufficient_n": True, "bound_limited": False, "a_bound": None,
    "n": n, "min_n": min_n, "method": method}`` — no fitted weights,
    no fitted ``a`` and no live ``pool_fn`` that a downstream readout
    could mistake for a completed optimisation.

    Degenerate single-forecaster panel (SF-BL-10801): if the panel
    has exactly one forecaster (``m == 1``), the simplex weight is
    degenerate ([1.0]) and there is no pool to extremize — pooled
    log-odds extremization corrects the shrinkage that averaging
    several members induces, and a single member has none. NO
    extremization is fitted: the result is the member itself
    (weights [1.0]; extremization exactly 1.0 for
    method='extremized', None for 'linear'/'geometric'), with
    ``degenerate_panel=True`` on the returned dict (False for
    ``m >= 2``; the insufficient-n dict carries the flag too, for
    schema consistency). Pre-fix, this path fitted ``a`` to the 4.0
    ``a_bounds`` cap on the filed 4-row repro.

    Small-n extremization cap (SF-BL-10801): when extremization is
    fitted on a non-degenerate panel with fewer than
    ``EXTREMIZE_MIN_RESOLVED`` (150) resolved rows, the fitted ``a``
    never exceeds ``EXTREMIZE_CAP`` (1.40, engine.py's production
    cap): the SLSQP search runs on the effective bounds
    ``(min(a_lo, 1.40), min(a_hi, 1.40))``. The returned dict
    discloses ``small_n_cap`` (1.40 when the cap regime applies,
    else None) and ``small_n_cap_applied`` (True iff the cap
    actually reduced the caller's upper bound), and
    ``bound_limited`` / ``a_bound`` are computed against the
    effective bounds. At ``n >= 150`` the effective bounds equal
    ``a_bounds`` and fits are unchanged.

    Optimisation: scipy SLSQP over the probability simplex for weights,
    jointly with log(a); multiple restarts (equal weights + each single
    forecaster) to avoid poor local optima in the extremized case.

    Raises
    ------
    ValueError
        If ``probs`` is an empty panel (0 rows or 0 forecasters), or if
        ``len(y) != n_rows(probs)`` (SF-BL-9592). Validation happens
        BEFORE any optimisation: without the empty-panel check an
        empty panel returned weights with a nan log score (0-row) or
        raised ZeroDivisionError (0-column); the length check (from
        SF-BL-8867) stops a length-1 ``y`` silently broadcasting across
        the pooled vector and returning fitted weights scored against
        copies of a single outcome.
    ValueError
        If ``probs`` contains NaN or infinite entries (SF-BL-9686).
        The screen is explicit in this function — not merely inherited
        from ``_as_matrix`` / ``_as_p`` — and runs BEFORE any
        optimisation and before the minimum-n early return: without
        it, a corrupt member column silently degraded the SLSQP fit
        to equal weights [0.5, 0.5] with ``log_score`` / ``brier``
        NaN, ``success=False`` buried in the dict, and a live
        ``pool_fn`` that a downstream readout could quote as a
        completed optimisation.
    ValueError
        If ``probs`` contains a finite value outside [0, 1]
        (SF-BL-10880; see :func:`_validate_pool_panel_range`). The
        screen is explicit in this function and runs BEFORE any
        optimisation and before the minimum-n early return: without
        it, ``Pc = _as_p(P)`` clipped 1.5 to 1-1e-7, the corrupt
        member took weight 1.0, and the extremized fit reported
        log_score 0.0019 / brier ~0.0 — a flawless-looking scorecard
        for an impossible forecast. Non-finite entries are unchanged:
        they raise the SF-BL-9686 finiteness ValueError above, because
        that screen runs first and this helper ignores non-finite
        values by design.
    ValueError
        If ``probs`` contains any boolean value (SF-BL-10976; see
        :func:`_validate_pool_panel_no_bool`). The screen is explicit
        in this function, runs on the RAW panel as its first
        statement — before ``_as_matrix``'s float conversion coerces
        True -> 1.0 / False -> 0.0 — and before the minimum-n early
        return: without it, ``Pc = _as_p(P)`` clipped the converted
        bool member to the 1-1e-7 / 1e-7 endpoints, the leaked member
        took dominant weight, and the fits reported log_score
        ~1e-10 / brier ~1e-21 — a flawless-looking scorecard for a
        boolean outcome column. This is this function's own screen,
        not merely the SF-BL-10975 ``_as_matrix`` screen inherited
        transitively.
    ValueError
        If ``y`` contains any boolean value (SF-BL-11016; see
        :func:`_validate_pool_y_no_bool`). The screen is explicit
        in this function, runs on the RAW ``y`` as the statement
        immediately after the SF-BL-10976 ``probs`` bool screen —
        before ``_as_y``'s float conversion coerces True -> 1.0 /
        False -> 0.0 and its ``==0/==1`` check passes the result —
        and before the minimum-n early return: without it, the
        filed bool-y repro returned weights [1.0, ~0.0] with
        log_score 0.00015 and ``success=True``, bit-identical to
        the lawful ``[1, 0, 1, 0]`` control. This is this
        function's own screen, not merely a screen inherited from
        ``_as_y`` (which the out copy does not have at all).
    ValueError
        If ``a_bounds`` is not a pair of finite reals with
        ``0 < lo <= hi`` (SF-BL-9616; see
        :func:`_validate_a_bounds`). Validation is unconditional —
        it runs for every method, even when extremization is not
        fitted, and before the minimum-n early return — because the
        bounds are part of the call's contract and an invalid pair
        must never reach ``np.log`` or the SLSQP bounds.
    ValueError
        If ``fit_extremization`` is not a bool (SF-BL-11062; see
        :func:`_validate_fit_extremization`). The check is explicit
        in this function, is its FIRST statement — before the panel
        screens, the minimum-n early return and any optimisation —
        and is unconditional (every method, even though only
        'extremized' consumes the flag), because the flag is part of
        the call's contract: without it, truthiness silently treated
        the string ``'no'`` as ``True`` and fitted an extremization
        of 4.0 (the ``a_bounds`` cap) the caller appeared to
        decline, while ``''`` / ``0`` / ``None`` silently acted as
        ``False``.
    ValueError
        If ``seed`` is a bool or not a non-negative integer
        (SF-BL-11088; see :func:`_validate_pool_seed`). Validation
        is unconditional — it runs for every method, after the
        ``a_bounds`` validation and before the minimum-n early
        return, and before ``np.random.default_rng`` consumes the
        seed for the Dirichlet restart: without it, ``seed='0'``
        aborted the fit with a bare SeedSequence ``TypeError`` that
        never named ``seed`` and ``seed=True`` silently optimised
        with entropy 1, returning fitted weights under a different
        reproducibility contract than the caller recorded.
    ValueError
        If ``y`` or ``probs`` contains any string, object-dtype
        (bare-object) or boolean element (SF-BL-11427; see
        :func:`_validate_pool_y_panel_types` and
        :func:`_validate_pool_probs_panel_types`). The screens are
        explicit in this function, run on the RAW panels after the
        SF-BL-10976 / SF-BL-11016 bool screens and before
        ``_as_y`` / ``_as_matrix`` float conversion, the minimum-n
        early return and any optimisation: without them, a string
        ``y`` returned the numeric control's fitted weights in the
        out copy, pandas string/bool panels returned them in both
        copies, and a bare ``object()`` element died with a bare
        ``TypeError`` that never named ``y`` / ``probs``.
    """
    # SF-BL-11062: fit_extremization type guard — the FIRST
    # statement, before every panel screen, the min_n early return
    # and any SLSQP work: the flag is a scalar call contract, and
    # truthiness alone silently turned 'no' into a fitted
    # extremization at the a_bounds cap. Pure validation, so lawful
    # True / False calls keep every pre-existing check's precedence
    # and result exactly.
    _validate_fit_extremization(fit_extremization)
    # SF-BL-10976: bool screen on the RAW panel — before _as_y /
    # _as_matrix convert anything: float conversion
    # coerces True -> 1.0 / False -> 0.0, after which a leaked boolean
    # member is indistinguishable from a lawful endpoint forecaster
    # and the finiteness / range screens below cannot see it. Pure
    # validation, so non-bool panels keep every pre-existing check's
    # precedence exactly.
    _validate_pool_panel_no_bool(probs)
    # SF-BL-11016: bool screen on the RAW y — immediately after the
    # SF-BL-10976 probs screen above and before _as_y converts
    # anything: float conversion coerces True -> 1.0 / False -> 0.0,
    # after which a boolean flag is indistinguishable from a lawful
    # 0/1 outcome and _as_y's ==0/==1 check passes it. The screen is
    # explicit in this function, not merely inherited from _as_y.
    # Pure validation, so non-bool outcomes keep every pre-existing
    # check's precedence exactly.
    _validate_pool_y_no_bool(y)
    # SF-BL-11427: explicit string / object-dtype / bool screens on
    # the RAW y and probs panels -- immediately after the bool
    # screens above and before _as_y / _as_matrix convert anything:
    # the inherited helper screens are incomplete and differ
    # between copies (the out copy's _as_y has no string screen, so
    # a string y was fitted there; neither copy's helpers see
    # pandas panels, so pandas string y and pandas string/bool
    # probs DataFrames were fitted in both copies; a bare object()
    # element died with a bare TypeError naming nothing). The
    # screens are explicit in this function, not merely inherited,
    # so a shared-helper change must never silently reopen this
    # path (the SF-BL-9686 discipline). Pure validation, so lawful
    # panels keep every pre-existing check's precedence exactly.
    _validate_pool_probs_panel_types(probs)
    _validate_pool_y_panel_types(y)
    y = _as_y(y); P = _as_matrix(probs); n, m = P.shape
    # SF-BL-9686: explicit finiteness screen in this function itself —
    # not merely the transitive SF-BL-3596 screen inside _as_matrix
    # (the _as_p screen sits after the min_n early return below, and a
    # shared-helper change must never silently reopen this path).
    # Pure validation, before every other check and any optimisation:
    # NaN comparisons are all False, so range/length checks alone pass
    # a NaN member entry, and the SLSQP fit then degraded to equal
    # weights with NaN scores and a live pool_fn.
    _require_finite(P, "probs")
    # SF-BL-10880: range screen — immediately after the finiteness
    # screen above, so it precedes the empty/length/method/a_bounds
    # checks' downstream computation, the min_n early return, and
    # Pc = _as_p(P): a finite 1.5 / -0.5 panel entry raises here and
    # is never clipped to near-certainty and fitted. The helper
    # ignores non-finite values, which already raised just above.
    _validate_pool_panel_range(P)
    if n == 0 or m == 0:
        raise ValueError(
            f"probs must be a non-empty panel (at least one row and one "
            f"forecaster); got probs with shape {P.shape}"
        )
    if len(y) != n:
        raise ValueError(
            f"len(y) ({len(y)}) != len(probs) ({n}): y and probs must have the same length"
        )
    if method not in ("linear", "geometric", "extremized"):
        raise ValueError("method must be 'linear', 'geometric' or 'extremized'")
    # SF-BL-9616: validate a_bounds before the min_n early return and
    # before any np.log / SLSQP consumption — unconditionally, so an
    # invalid pair raises even when extremization is not fitted.
    a_lo, a_hi = _validate_a_bounds(a_bounds)
    # SF-BL-11088: validate seed before the min_n early return and
    # before np.random.default_rng consumes it — unconditionally, so
    # a bool / non-integer seed raises the named ValueError even when
    # extremization is not fitted or the panel is insufficient-n,
    # and never silently becomes a different entropy than recorded.
    _validate_pool_seed(seed)
    if n < min_n:
        # SF-BL-9443: insufficient-n — return the explicit result
        # documented above; never run the optimiser on a panel too
        # small to identify weights (let alone weights + a jointly).
        return {"weights": None, "extremization": None, "method": method,
                "log_score": None, "brier": None, "pooled": None,
                "pool_fn": None, "success": False,
                "status": "insufficient_n", "insufficient_n": True,
                "bound_limited": False, "a_bound": None,
                "degenerate_panel": bool(m == 1),
                "small_n_cap_applied": False, "small_n_cap": None,
                "n": int(n), "min_n": min_n}
    use_a = (method == "extremized" and fit_extremization)
    if m == 1:
        # SF-BL-10801: degenerate single-forecaster panel — the
        # simplex weight is forced to [1.0], so no pool exists to
        # extremize and no a is identifiable as a *pool* parameter.
        # Return the member itself with a = 1.0 and the explicit
        # flag; never run the joint SLSQP fit that pinned a at the
        # 4.0 a_bounds cap on the filed repro.
        w_single = np.ones(1)
        if method == "linear":
            pooled_single, a_single = linear_pool(P, w_single), None
            fn_single = lambda Q, w=w_single: linear_pool(Q, w)  # noqa: E731
        elif method == "geometric":
            pooled_single, a_single = geometric_pool(P, w_single), None
            fn_single = lambda Q, w=w_single: geometric_pool(Q, w)  # noqa: E731
        else:
            a_single = 1.0
            pooled_single = log_odds_pool(P, w_single, a_single)
            fn_single = lambda Q, w=w_single: log_odds_pool(Q, w, 1.0)  # noqa: E731
        return {"weights": w_single, "extremization": a_single,
                "method": method,
                "log_score": log_score(y, pooled_single),
                "brier": brier_score(y, pooled_single),
                "pooled": pooled_single, "pool_fn": fn_single,
                "success": True, "bound_limited": False, "a_bound": None,
                "degenerate_panel": True,
                "small_n_cap_applied": False, "small_n_cap": None}
    # SF-BL-10801: small-n cap — below EXTREMIZE_MIN_RESOLVED resolved
    # rows the fitted a never exceeds EXTREMIZE_CAP (engine.py's
    # production cap). The lower bound is clamped too, so a caller
    # interval lying entirely above the cap degenerates to a pinned
    # a = EXTREMIZE_CAP instead of an empty SLSQP interval.
    a_lo_eff, a_hi_eff = a_lo, a_hi
    small_n_cap_applied, small_n_cap = False, None
    if use_a and n < EXTREMIZE_MIN_RESOLVED:
        small_n_cap = EXTREMIZE_CAP
        a_hi_eff = min(a_hi, EXTREMIZE_CAP)
        a_lo_eff = min(a_lo, EXTREMIZE_CAP)
        small_n_cap_applied = a_hi_eff < a_hi
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
    bounds = [(0.0, 1.0)] * m + ([(np.log(a_lo_eff), np.log(a_hi_eff))] if use_a else [])
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
    # SF-BL-9443: disclose a fit whose extremization landed on an
    # a_bounds edge — that a is the cap, not an estimate.
    bound_limited, a_bound_hit = False, None
    if use_a and a_out is not None:
        lo_b, hi_b = a_lo_eff, a_hi_eff  # SF-BL-9616 validated pair; SF-BL-10801 effective (small-n-capped) bounds
        if abs(a_out - hi_b) <= 1e-6 * max(1.0, abs(hi_b)):
            bound_limited, a_bound_hit = True, "upper"
        elif abs(a_out - lo_b) <= 1e-6 * max(1.0, abs(lo_b)):
            bound_limited, a_bound_hit = True, "lower"
    return {"weights": w, "extremization": a_out, "method": method,
            "log_score": log_score(y, pooled), "brier": brier_score(y, pooled),
            "pooled": pooled, "pool_fn": fn, "success": bool(best.success),
            "bound_limited": bound_limited, "a_bound": a_bound_hit,
            "degenerate_panel": False,
            "small_n_cap_applied": small_n_cap_applied,
            "small_n_cap": small_n_cap}


def _has_string_in_pandas_panel(raw) -> bool:
    """True iff a pandas peer panel contains any string value
    (SF-BL-11124 round 2).

    :func:`_has_string` returns False for any non-list/tuple/ndarray
    container, so numeric strings carried in a pandas ``Series`` /
    ``Index`` / ``Categorical`` slipped past the round-1 peer screen
    and ``brier_score``'s ``np.asarray(p, dtype=float)`` parsed them
    as lawful probabilities — bit-identical to the numeric control.
    Pandas is detected WITHOUT importing it (this module is pure
    numpy / scipy / stdlib), by the type's module (``pandas.core...``)
    plus duck-typed attributes:

    - a ``string`` dtype (``dtype.name == "string"``) holds strings
      by construction, even when the panel is empty;
    - a category dtype whose categories (the container's own
      ``categories``, or ``dtype.categories``) contain any ``str`` /
      ``bytes`` value is a string panel, even if an individual
      observed value happens to be numeric;
    - otherwise the panel's values are scanned elementwise with
      :func:`_has_string`, which catches object-dtype ``Series`` /
      ``Index`` panels holding ``str`` / ``bytes`` leaves (mixed
      panels included) while leaving numeric object panels lawful.

    Numeric pandas panels (float/int ``Series`` / ``Index`` and
    numeric-category panels) contain no string value, so they return
    False here and stay bit-identical. Any inspection error returns
    False, leaving the pre-existing downstream behaviour untouched.
    """
    try:
        if not (type(raw).__module__ or "").startswith("pandas"):
            return False
        if type(raw).__name__ == "DataFrame":
            return False
        dtype = getattr(raw, "dtype", None)
        if dtype is not None and getattr(dtype, "name", None) == "string":
            return True
        cats = getattr(raw, "categories", None)
        if cats is None and dtype is not None:
            cats = getattr(dtype, "categories", None)
        if cats is not None and any(_has_string(c) for c in list(cats)):
            return True
        return any(_has_string(v) for v in list(raw))
    except Exception:  # noqa: BLE001 — inspection must never break lawful panels
        return False


def _has_bool_in_pandas_panel(raw) -> bool:
    """True iff a pandas outcome panel contains any boolean value
    (SF-BL-11461).

    Mirror of :func:`_has_string_in_pandas_panel` for the bool
    dimension: :func:`_has_bool` returns False for any
    non-list/tuple/ndarray container, so boolean flags carried in a
    pandas ``Series`` / ``Index`` / ``Categorical`` slipped past
    every existing screen and ``np.asarray(..., dtype=float)``
    coerced them to lawful 1.0 / 0.0 endpoints. Pandas is detected
    WITHOUT importing it (this module is pure numpy / scipy /
    stdlib), by the type's module (``pandas.core...``) plus
    duck-typed attributes:

    - a ``bool`` / ``boolean`` dtype (``dtype.name``) holds bools
      by construction, even when the panel is empty;
    - a category dtype whose categories (the container's own
      ``categories``, or ``dtype.categories``) contain any bool
      value is a bool panel, even if no observed value is boolean;
    - otherwise the panel's values are scanned elementwise with
      :func:`_has_bool`, which catches object-dtype ``Series`` /
      ``Index`` panels holding bool leaves (mixed panels included)
      while leaving numeric object panels lawful.

    Numeric pandas panels (float/int ``Series`` / ``Index`` and
    numeric-category panels) contain no bool value, so they return
    False here and stay bit-identical. Any inspection error returns
    False, leaving the pre-existing downstream behaviour untouched.
    """
    try:
        if not (type(raw).__module__ or "").startswith("pandas"):
            return False
        if type(raw).__name__ == "DataFrame":
            return False
        dtype = getattr(raw, "dtype", None)
        if dtype is not None and getattr(dtype, "name", None) in ("bool", "boolean"):
            return True
        cats = getattr(raw, "categories", None)
        if cats is None and dtype is not None:
            cats = getattr(dtype, "categories", None)
        if cats is not None and any(_has_bool(c) for c in list(cats)):
            return True
        return any(_has_bool(v) for v in list(raw))
    except Exception:  # noqa: BLE001 — inspection must never break lawful panels
        return False


def _reject_peer_y(raw):
    """Raise a named ValueError if ``peer_score``'s outcome panel
    ``y`` contains any string or boolean value (SF-BL-11461).

    This screen MUST run on the raw ``y`` input, inside
    :func:`peer_score` as its first statement, before the
    SF-BL-11124 ``p_agent`` / ``p_crowd`` screens and before either
    :func:`brier_score` call: ``peer_score`` never type-validated
    its own outcome panel — it differences two ``brier_score``
    calls whose ``_as_y`` path coerces via
    ``np.asarray(y, dtype=float)`` — so a text-serialised outcome
    panel was parsed as the outcomes it spells and a boolean flag
    panel was coerced True -> 1.0 / False -> 0.0, and both returned
    the numeric control (``peer_score(['1','0'], [0.9,0.1],
    [0.5,0.5])`` and the ``[True, False]`` mirror each returned
    0.24). Where a downstream screen existed at all (the headline
    scorers' SF-BL-10938/11052 screens in one copy) it was not
    ``peer_score``'s own, and it missed every pandas panel:
    :func:`_has_bool` / :func:`_has_string` return False for any
    non-list/tuple/ndarray container, so pandas string / bool
    ``y`` panels (``Series`` / ``Index`` / ``Categorical``, object,
    ``string``, ``bool``, ``boolean`` and category dtypes) were
    scored silently in both copies, and a ``string``-dtype Series
    bearing ``pd.NA`` died with a bare ``TypeError`` that never
    named ``y``. A numeric string is not an outcome and a bool is
    not a 0/1 outcome — the panel's type is itself evidence of a
    corrupt feed, and scoring it hides that bug in the fleet's
    difficulty-controlled promotion metric. Detection is
    :func:`_has_bool` + :func:`_has_bool_in_pandas_panel` (bool
    dimension, checked first, mirroring ``_as_y``'s screen order)
    and :func:`_has_string` + :func:`_has_string_in_pandas_panel`
    (string dimension). Pure validation — it never transforms
    ``raw`` — so lawful numeric panels, including numeric pandas
    panels, are bit-identical to the pre-fix computation.
    """
    if _has_bool(raw) or _has_bool_in_pandas_panel(raw):
        raise ValueError(
            "y must not contain boolean values; got boolean "
            "value(s) in y (a bool is not a valid y value) "
            "(SF-BL-11461)"
        )
    if _has_string(raw) or _has_string_in_pandas_panel(raw):
        raise ValueError(
            "y must not contain string values; got string "
            "value(s) in y (a string is not a valid y value) "
            "(SF-BL-11461)"
        )


def _reject_bool_peer_panel(raw, name):
    """Raise a named ValueError if a ``peer_score`` panel contains any
    boolean value (SF-BL-11389).

    This screen MUST run on the raw ``p_agent`` / ``p_crowd`` inputs,
    inside :func:`peer_score` before either :func:`brier_score` call:
    ``peer_score`` differences two ``brier_score`` calls whose
    ``_as_y`` / ``_as_p`` path converts the panel with
    ``np.asarray(p, dtype=float)``, so a boolean flag panel was
    coerced True -> 1.0 / False -> 0.0 and became the most extreme
    lawful forecast — ``peer_score([1, 0], [True, False],
    [0.5, 0.5])`` returned 0.25, bit-identical to the perfect numeric
    agent ``[1, 0]``, and a bool crowd panel returned -0.25. Where a
    downstream bool screen existed at all (the headline scorers'
    SF-BL-10938 screen in one copy) it named the generic ``p``, never
    ``p_agent`` / ``p_crowd`` — ``peer_score``'s own entry point was
    never claimed — and it missed every pandas panel:
    :func:`_has_bool` returns False for any non-list/tuple/ndarray
    container, so pandas bool ``Series`` / ``Index`` /
    ``Categorical`` panels (bool, ``boolean``, object and category
    dtypes) were scored silently in both copies. A bool is not a
    probability — the panel's type is itself evidence of a corrupt
    feed, and scoring it hides that bug in the fleet's
    difficulty-controlled promotion metric. Detection is
    :func:`_has_bool` + :func:`_has_bool_in_pandas_panel`,
    mirroring the SF-BL-11124 string screen alongside which this one
    runs. Pure validation — it never transforms ``raw`` — so lawful
    float/int panels, including numeric pandas panels, are
    bit-identical to the pre-fix computation.
    """
    if _has_bool(raw) or _has_bool_in_pandas_panel(raw):
        raise ValueError(
            f"{name} must not contain boolean values; got boolean "
            f"value(s) in {name} (a bool is not a valid {name} "
            f"value) (SF-BL-11389)"
        )


def _reject_string_peer_panel(raw, name):
    """Raise a named ValueError if a ``peer_score`` panel contains any
    string value (SF-BL-11124).

    This screen MUST run on the raw ``p_agent`` / ``p_crowd`` inputs,
    inside :func:`peer_score` before either :func:`brier_score` call:
    ``brier_score`` converts its panel with
    ``np.asarray(p, dtype=float)``, which silently parses numeric
    strings, so a text-serialised agent or crowd panel (a JSON / CSV
    pipeline bug) was scored as lawful probabilities —
    ``peer_score([1,0,1,0], ['0.7','0.3','0.6','0.4'], <numeric
    crowd>)`` and the string-crowd mirror both returned the numeric
    control value, the agent panel's underlying Brier being the
    clean-control 0.1250. Where a downstream string screen existed at
    all it named the generic ``p``, never the peer panel that was
    actually corrupt. A numeric string is not a number — the panel's
    type is itself evidence of a corrupt feed, and scoring it hides
    that bug in the fleet's difficulty-controlled promotion metric.
    Round 2: the screen also covers pandas containers via
    :func:`_has_string_in_pandas_panel` — round 1 reused
    :func:`_has_string` alone, which returns False for any
    non-list/tuple/ndarray container, so string ``Series`` /
    ``Index`` / ``Categorical`` panels (object, ``string`` and
    category dtypes) were still scored silently. Pure validation —
    it never transforms ``raw`` — so lawful numeric panels,
    including numeric pandas panels, are bit-identical to the
    pre-fix computation.
    """
    if _has_string(raw) or _has_string_in_pandas_panel(raw):
        raise ValueError(
            f"{name} must not contain string values; got string "
            f"value(s) in {name} (a string is not a valid {name} "
            f"value) (SF-BL-11124)"
        )


def peer_score(y, p_agent, p_crowd) -> float:
    """Relative Brier vs a contemporaneous crowd/market forecast.

    peer = BS_crowd - BS_agent. > 0 beats the crowd; > ~0.02 sustained over a
    large question set is 'superforecaster tier' in the ForecastBench /
    Good Judgment literature. Controls for question difficulty, which raw
    Brier does not. Added in v1.1 (see docs/EXTERNAL_ADVICE_REVIEW.md).

    Raises
    ------
    ValueError
        If ``y`` contains any string or boolean value (SF-BL-11461;
        see :func:`_reject_peer_y`). The screen runs on the raw
        outcome panel, first, before any float conversion could
        parse a text-serialised outcome as a lawful 0/1 outcome or
        coerce a boolean flag to a lawful endpoint.
    ValueError
        If ``p_agent`` or ``p_crowd`` contains any string value
        (SF-BL-11124; see :func:`_reject_string_peer_panel`). The
        screen runs on the raw panels, ``p_agent`` first, before any
        float conversion could read a text-serialised forecast as a
        lawful probability.
    ValueError
        If ``p_agent`` or ``p_crowd`` contains any boolean value
        (SF-BL-11389; see :func:`_reject_bool_peer_panel`). The
        screen runs on the raw panels, interleaved per panel with
        the SF-BL-11124 string screen so ``p_agent`` is fully
        screened before ``p_crowd``, before any float conversion
        could coerce True/False to the lawful 1.0/0.0 endpoints.
    """
    _reject_peer_y(y)  # SF-BL-11461: raw screen on y, first, before brier_score's _as_y float conversion parses '1' as 1.0 / coerces True to 1.0
    _reject_bool_peer_panel(p_agent, "p_agent")  # SF-BL-11389: raw screen, before brier_score's float conversion coerces True->1.0 / False->0.0
    _reject_string_peer_panel(p_agent, "p_agent")  # SF-BL-11124: raw screen, before brier_score's float conversion reads '0.7' as 0.7
    _reject_bool_peer_panel(p_crowd, "p_crowd")  # SF-BL-11389
    _reject_string_peer_panel(p_crowd, "p_crowd")  # SF-BL-11124
    return float(brier_score(y, p_crowd) - brier_score(y, p_agent))


def _validate_trim(trim) -> float:
    """Validate ``trimmed_mean_pool``'s ``trim`` fraction (SF-BL-11087).

    Returns the fraction as a float. Pure validation — a valid
    fraction is returned unchanged in value, so pooled outputs for
    lawful ``trim`` are bit-identical to the pre-fix computation.

    Raises
    ------
    ValueError
        Unless ``trim`` is a finite real scalar with
        ``0 <= trim < 0.5``. Previously ``trim`` was never
        type-validated: it flowed straight into
        ``int(np.floor(trim * m))``, so ``trim=True`` set k=m and the
        empty-core fallback silently returned the plain mean (0.4625
        on the titled 8-member panel, vs 0.45 at the lawful
        ``trim=0.25``), ``trim='0.1'`` (the form a JSON-loaded pool
        config actually delivers) died with a bare ufunc TypeError
        that never named the argument, ``trim=NaN`` died with a bare
        ``cannot convert float NaN to integer`` ValueError and
        ``trim=+/-inf`` with a bare OverflowError. A bool is not a
        fraction (in Python it is an int subclass, which is exactly
        why it slipped past the arithmetic), and a string is not a
        fraction either — both raise this named error, as do
        non-finite values and non-scalars (None, sequences,
        multi-element arrays, or any value ``float()`` cannot read
        as a real scalar). Finite values outside [0, 0.5) raise the
        same error: at ``trim >= 0.5`` nothing would be left to
        average, and pre-fix ``trim=-0.1`` silently returned [0.9]
        via a negative-k slice while ``trim=0.5`` / ``0.6`` silently
        fell back to the plain mean (the range dimension is
        SF-BL-8805's neighbouring fixture).
    """
    msg = ("trim must be finite with 0 <= trim < 0.5 (SF-BL-11087); "
           f"got trim={trim!r}")
    if isinstance(trim, (bool, np.bool_)) or isinstance(trim, (str, bytes)):
        raise ValueError(msg)
    if isinstance(trim, np.ndarray) and trim.ndim > 0:
        raise ValueError(msg)
    try:
        t = float(trim)
    except (TypeError, ValueError):
        raise ValueError(msg) from None
    if not np.isfinite(t) or t < 0.0 or t >= 0.5:
        raise ValueError(msg)
    return t


def _has_bool_in_trimmed_pool_panel(raw) -> bool:
    """True iff a ``trimmed_mean_pool`` panel contains any boolean
    value (SF-BL-11020).

    :func:`_has_bool` covers Python ``bool`` / ``np.bool_`` scalars,
    lists / tuples at any nesting, bool-dtype ndarrays and
    object-dtype ndarrays; the peer-panel helper
    :func:`_has_bool_in_pandas_panel` covers pandas ``Series`` /
    ``Index`` / ``Categorical`` panels (``bool`` / ``boolean`` and
    category dtypes). That helper deliberately returns False for a
    ``DataFrame``, but a DataFrame is a lawful 2-D panel shape for
    this pool (``np.asarray`` reads its values), so this detector
    adds the DataFrame scan: a ``bool`` / ``boolean`` column dtype,
    a category dtype whose categories contain a bool, or any bool
    leaf in the frame's values. Numeric pandas panels contain no
    bool value, so they return False and stay bit-identical. Any
    inspection error returns False, leaving the pre-existing
    downstream behaviour untouched.
    """
    if _has_bool(raw) or _has_bool_in_pandas_panel(raw):
        return True
    try:
        if not (type(raw).__module__ or "").startswith("pandas"):
            return False
        if type(raw).__name__ != "DataFrame":
            return False
        dtypes = getattr(raw, "dtypes", None)
        if dtypes is not None:
            for dt in list(dtypes):
                if getattr(dt, "name", None) in ("bool", "boolean"):
                    return True
                if getattr(dt, "kind", None) == "b":
                    return True
                cats = getattr(dt, "categories", None)
                if cats is not None and any(_has_bool(c) for c in list(cats)):
                    return True
        values = getattr(raw, "values", None)
        if values is not None and _has_bool(values):
            return True
        return False
    except Exception:  # noqa: BLE001 — inspection must never break lawful panels
        return False


def _reject_bool_trimmed_mean_pool_members(raw):
    """Raise a named ValueError if a ``trimmed_mean_pool`` panel
    contains any boolean member (SF-BL-11020).

    This screen MUST run on the raw ``probs`` input, as the first
    statement of :func:`trimmed_mean_pool`, before its
    ``np.asarray(probs, float)`` conversion: ``trimmed_mean_pool``
    bypasses :func:`_as_matrix`, so the SF-BL-10975 bool-member
    screen that protects ``linear_pool`` / ``geometric_pool`` /
    ``log_odds_pool`` never saw this pool's panel, and float
    conversion silently coerced a boolean flag leaking into a
    member field to True -> 1.0 / False -> 0.0, the most extreme
    lawful forecasts in the pool — ``trimmed_mean_pool([[True,
    False, True]])`` returned ``[0.6667]``, bit-identical to the
    lawful int control ``[[1, 0, 1]]`` — so a caller's boolean
    mask bug became a pooled probability. Bools sit inside
    [0, 1] numerically, so the SF-BL-3596 finiteness screen (and
    any range screen) passes them; only a type screen on the raw
    input can catch them. A bool is not a probability: the
    panel's type is itself evidence of a corrupt feed. This is
    the ``probs`` / member argument of ``trimmed_mean_pool``
    only, distinct from SF-BL-11464 (string members of this
    pool, screened immediately after this one) and from
    SF-BL-11087 (this pool's ``trim`` fraction, screened in
    :func:`_validate_trim`). Detection is
    :func:`_has_bool_in_trimmed_pool_panel`. Pure validation —
    it never transforms ``raw`` — so lawful float/int panels
    (including numeric pandas panels and the int endpoints
    0 / 1) are bit-identical to the pre-fix computation.
    """
    if _has_bool_in_trimmed_pool_panel(raw):
        raise ValueError(
            "probs must not contain boolean values; got boolean "
            "value(s) in probs (a bool is not a valid probs member) "
            "(SF-BL-11020; cf. SF-BL-10975)"
        )


def _has_string_in_trimmed_pool_panel(raw) -> bool:
    """True iff a ``trimmed_mean_pool`` panel contains any string
    value (SF-BL-11464).

    :func:`_has_string` covers Python ``str`` / ``bytes`` scalars,
    lists / tuples at any nesting, string-dtype ndarrays (kind
    ``U`` / ``S``) and object-dtype ndarrays; the peer-panel helper
    :func:`_has_string_in_pandas_panel` covers pandas ``Series`` /
    ``Index`` / ``Categorical`` panels (object, ``string`` and
    category dtypes). That helper deliberately returns False for a
    ``DataFrame``, but a DataFrame is a lawful 2-D panel shape for
    this pool (``np.asarray`` reads its values), so this detector
    adds the DataFrame scan: a ``string`` / ``U`` / ``S`` column
    dtype, a category dtype whose categories contain a string, or
    any string leaf in the frame's values. Numeric pandas panels
    contain no string value, so they return False and stay
    bit-identical. Any inspection error returns False, leaving the
    pre-existing downstream behaviour untouched.
    """
    if _has_string(raw) or _has_string_in_pandas_panel(raw):
        return True
    try:
        if not (type(raw).__module__ or "").startswith("pandas"):
            return False
        if type(raw).__name__ != "DataFrame":
            return False
        dtypes = getattr(raw, "dtypes", None)
        if dtypes is not None:
            for dt in list(dtypes):
                if getattr(dt, "name", None) == "string":
                    return True
                if getattr(dt, "kind", None) in ("U", "S"):
                    return True
                cats = getattr(dt, "categories", None)
                if cats is not None and any(_has_string(c) for c in list(cats)):
                    return True
        values = getattr(raw, "values", None)
        if values is not None and _has_string(values):
            return True
        return False
    except Exception:  # noqa: BLE001 — inspection must never break lawful panels
        return False


def _reject_string_trimmed_mean_pool_members(raw):
    """Raise a named ValueError if a ``trimmed_mean_pool`` panel
    contains any string member (SF-BL-11464).

    This screen MUST run on the raw ``probs`` input, at the top
    of :func:`trimmed_mean_pool`, before its
    ``np.asarray(probs, float)`` conversion: ``trimmed_mean_pool``
    bypasses :func:`_as_matrix`, so the SF-BL-11125 string-member
    screen that protects ``linear_pool`` / ``geometric_pool`` /
    ``log_odds_pool`` never saw this pool's panel, and float
    conversion silently parsed a member forecast serialised as
    text as the number it spells — ``trimmed_mean_pool([['0.7',
    '0.3']])`` returned ``[0.5]``, bit-identical to the numeric
    ``[[0.7, 0.3]]`` control — so a corrupt CSV/JSON-sourced
    forward panel entered the pooled forecast undetected. A
    numeric string is not a probability: the panel's type is
    itself evidence of a corrupt feed. This is the ``probs`` /
    member argument of ``trimmed_mean_pool`` only, distinct from
    SF-BL-11020 (bool members of this pool) and from SF-BL-11087
    (this pool's ``trim`` fraction, screened in
    :func:`_validate_trim`). Detection is
    :func:`_has_string_in_trimmed_pool_panel`. Pure validation —
    it never transforms ``raw`` — so lawful float/int panels
    (including numeric pandas panels) are bit-identical to the
    pre-fix computation.
    """
    if _has_string_in_trimmed_pool_panel(raw):
        raise ValueError(
            "probs must not contain string values; got string "
            "value(s) in probs (a string is not a valid probs member) "
            "(SF-BL-11464; cf. SF-BL-11125)"
        )


def trimmed_mean_pool(probs, trim: float = 0.1):
    """Trimmed-mean pool: drop the top/bottom `trim` fraction of agent
    probabilities per question, average the rest. Robust to one wild agent;
    validated as an aggregation baseline by Halawi et al. (ForecastBench).
    probs: (n_questions, n_agents) or (n_agents,).

    Raises
    ------
    ValueError
        If any member of ``probs`` is a boolean (SF-BL-11020; see
        :func:`_reject_bool_trimmed_mean_pool_members`).
        Previously this pool bypassed ``_as_matrix`` and coerced
        True/False to 1.0/0.0, returning ``[0.6667]`` on
        ``[[True, False, True]]``, bit-identical to the lawful
        int control ``[[1, 0, 1]]``. The screen runs on the raw
        input, before any float conversion.
    ValueError
        If any member of ``probs`` is a string (SF-BL-11464; see
        :func:`_reject_string_trimmed_mean_pool_members`).
        Previously this pool bypassed ``_as_matrix`` and parsed
        '0.7' as 0.7, returning ``[0.5]`` on ``[['0.7', '0.3']]``,
        bit-identical to the numeric control. The screen runs on
        the raw input, before any float conversion.
    ValueError
        If ``trim`` is a bool, a string, a non-scalar or non-finite,
        or a finite value outside [0, 0.5) (SF-BL-11087; see
        :func:`_validate_trim`). Previously ``trim=True`` silently
        returned the plain mean, ``trim='0.1'`` died with a bare
        TypeError and ``trim=NaN`` with an unnamed ValueError.
    """
    _reject_bool_trimmed_mean_pool_members(probs)  # SF-BL-11020: raw screen, before float conversion coerces True->1.0
    _reject_string_trimmed_mean_pool_members(probs)  # SF-BL-11464: raw screen, before float conversion parses '0.7'->0.7
    trim = _validate_trim(trim)  # SF-BL-11087: before any floor/slice use
    P = np.asarray(probs, float)
    _require_finite(P, "probs")  # SF-BL-3596: bypasses _as_matrix/_as_p
    if P.ndim == 1:
        P = P[None, :]
    m = P.shape[1]
    k = int(np.floor(trim * m))
    Ps = np.sort(P, axis=1)
    core = Ps[:, k:m - k] if m - 2 * k > 0 else Ps
    return core.mean(axis=1)


__all__ = [
    "peer_score", "trimmed_mean_pool",
    "brier_score", "log_score", "log_loss", "murphy_decomposition",
    "expected_calibration_error", "maximum_calibration_error", "ece", "mce",
    "calibration_bins", "calibration_table", "evaluate",
    "PlattScaler", "IsotonicCalibrator", "TemperatureScaler", "BetaCalibrator",
    "CALIBRATORS", "fit_calibrator",
    "linear_pool", "geometric_pool", "log_odds_pool", "extremized_pool",
    "optimize_pool_weights",
]
