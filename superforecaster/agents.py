"""
agents.py — Agent Ensemble / Superforecaster Aggregation + Scoring Layer
=========================================================================
Pure numpy + stdlib. No network, no credentials, no external data.

Implements:
  * Forecast / AgentForecast objects
  * Log-odds (logit) pooling with per-agent weights
  * Extremizing (Satopää et al. 2014)
  * Brier / log-score scoring and track-record weight updating
  * A simulated tournament: synthetic agents with known skill / noise /
    bias profiles forecasting >=500 synthetic questions whose generating
    probabilities are known, comparing aggregation strategies.

Run:
    python3 agents.py
    python3 agents.py --n-questions 2000 --seed 7

Design notes (see DESIGN.md for the full protocol):
  - Independent round-1 forecasts are pooled in log-odds space because
    probabilities are not additive; averaging in logit space respects
    the geometry of evidence (each unit of log-odds = a likelihood ratio).
  - Extremizing corrects the systematic under-confidence of averaged
    forecasts when forecasters share information (they double-count the
    common prior and dilute private signals).
  - Track-record weights are a proper-scoring-rule exponential update:
        w_i  ∝  exp(-lambda * mean_brier_i)   renormalised to sum to 1
    with shrinkage toward equal weights early in a forecaster's history,
    so a lucky streak on 5 questions cannot capture the pool.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-6  # probability clipping bound for logit / log-score safety


# ---------------------------------------------------------------------------
# Core probability utilities
# ---------------------------------------------------------------------------

def clip_p(p: np.ndarray | float) -> np.ndarray | float:
    return np.clip(p, EPS, 1.0 - EPS)


def logit(p: np.ndarray | float) -> np.ndarray | float:
    p = clip_p(np.asarray(p, dtype=float) if not np.isscalar(p) else float(p))
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid (scalar- and array-safe)."""
    if np.isscalar(x):
        x = float(x)
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x)) if x < 700 else 1.0
        ex = math.exp(x) if x > -700 else 0.0
        return ex / (1.0 + ex)
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _reject_bool_probs(probs, caller: str) -> None:
    """Raise ValueError if any probability in ``probs`` is a bool.

    SF-BL-10939: a bool satisfies every existing scorer screen
    numerically — ``float(True) == 1.0`` is finite and lies in
    [0, 1] — so the SF-BL-9784 / SF-BL-10031 / SF-BL-10032
    finiteness and range checks graded a flag that leaked into a
    forecast field as a lawful endpoint forecast:
    ``brier_score([True], [1])`` returned 0.0,
    ``calibration_stats([True, False], [1, 0])`` returned ECE 0.0
    with mean_forecast 0.5, and ``log_score([True], [1])`` returned
    ~1e-06. A bool is not a probability (precedent: SF-BL-10349 in
    metaculus_client.clip_prob). This screen therefore runs on the
    ORIGINAL input, before any ``np.asarray(..., dtype=float)``
    conversion erases the bool: it catches a scalar bool / np.bool_,
    a bool-dtype array, an object-dtype array holding bools, and
    bools nested in lists/tuples (a mixed list such as
    ``[0.5, True]`` promotes to float64 under ``np.asarray``, so
    dtype inspection alone would miss it). Only ``probs`` is
    screened — bool OUTCOMES stay lawful (``brier_score([0.5, 0.6],
    [True, False]) == 0.305``, pinned by SF-BL-10853) — and a
    genuine float/int endpoint panel such as ``[0.0, 1.0]`` or
    ``[1, 0]`` is untouched, so lawful panels are bit-identical.
    Pure validation — it never transforms its inputs.
    """
    def _has_bool(x) -> bool:
        if isinstance(x, (bool, np.bool_)):
            return True
        if isinstance(x, np.ndarray):
            if x.dtype.kind == "b":
                return True
            if x.dtype == object:
                return any(_has_bool(v) for v in x.flat)
            return False
        dtype = getattr(x, "dtype", None)
        if dtype is not None and getattr(dtype, "kind", None) == "b":
            return True
        if isinstance(x, (list, tuple)):
            return any(_has_bool(v) for v in x)
        return False

    if _has_bool(probs):
        shown = repr(probs)
        if len(shown) > 120:
            shown = shown[:117] + "..."
        raise ValueError(
            f"{caller}: probs must be real probabilities, not bool; "
            f"got bool probability value(s) in {shown} — a bool is "
            "not a probability (True/False coerce to the endpoint "
            "forecasts 1.0/0.0) (SF-BL-10939)"
        )


def _reject_string_inputs(probs, outcomes, caller: str) -> None:
    """Raise ValueError naming the argument if a panel holds strings.

    SF-BL-11058: a numeric string satisfies every existing scorer
    screen after conversion, because the scorers convert with
    ``np.asarray(..., dtype=float)``, which silently parses it —
    ``brier_score(['0.5'], [1])`` returned 0.25,
    ``brier_score([0.5, 0.6], ['1', '0'])`` returned 0.305,
    ``log_score(['0.5'], [1])`` returned ln(2), and
    ``calibration_stats(['0.5', '0.6'], [1, 0])`` returned ECE 0.05
    with mean_forecast 0.55. A string is not a number (precedent:
    SF-BL-11054 in engine.py, SF-BL-8800 in this module's
    ``ForecasterRecord.add``), whether it arrives as a forecast or
    as an outcome — a text field leaking into either panel (CSV /
    JSON round-trips make every field text) must not be graded as
    a lawful numeric panel. This screen therefore runs on the
    ORIGINAL inputs, before any ``np.asarray(..., dtype=float)``
    conversion erases the string type, and screens BOTH arguments:
    it catches a scalar str / bytes, a string-dtype (kind ``U`` /
    ``S``) array, a mixed list such as ``[0.5, '0.6']`` (which
    promotes to a string dtype under ``np.asarray``), strings
    nested in lists/tuples, and object-dtype arrays holding
    strings. Unlike the SF-BL-10939 bool screen, string OUTCOMES
    are rejected too — the bool-outcome carve-out does not extend
    to strings, since no lawful caller supplies outcomes as text.
    Pure validation — it never transforms its inputs — so genuine
    numeric panels (float/int arrays, lists and tuples, bool
    outcomes included) are bit-identical.
    """
    def _has_string(x) -> bool:
        if isinstance(x, (str, bytes, bytearray, np.str_, np.bytes_)):
            return True
        if isinstance(x, np.ndarray):
            if x.dtype.kind in ("U", "S"):
                return True
            if x.dtype == object:
                return any(_has_string(v) for v in x.flat)
            return False
        dtype = getattr(x, "dtype", None)
        if dtype is not None and getattr(dtype, "kind", None) in ("U", "S"):
            return True
        if isinstance(x, (list, tuple)):
            return any(_has_string(v) for v in x)
        return False

    for name, panel in (("probs", probs), ("outcomes", outcomes)):
        if _has_string(panel):
            shown = repr(panel)
            if len(shown) > 120:
                shown = shown[:117] + "..."
            raise ValueError(
                f"{caller}: {name} must be real numbers, not string; "
                f"got string value(s) in {shown} — a numeric string "
                "is not a number and must not be silently parsed "
                "into a scored panel (SF-BL-11058)"
            )


def _require_same_length(probs: np.ndarray, outcomes: np.ndarray) -> None:
    """Raise ValueError if probs and outcomes do not pair row-for-row.

    SF-BL-9784: without this check a 1-element outcomes array silently
    broadcasts across the whole forecast panel in ``(probs - outcomes)``
    (``brier_score([0.5, 0.6], [1])`` returned 0.205, grading two forecasts
    against copies of one outcome), an empty panel returned NaN as an
    ordinary float, and a NaN forecast likewise propagated to a NaN
    score. Mirrors calibration.py's ``_require_same_length``
    (SF-BL-8867 family), plus the empty-panel and non-finite screens
    this task's prereg froze. Pure validation — it never transforms its
    inputs — so finite equal-length non-empty scores are bit-identical
    to the pre-fix computation.
    """
    n_probs = len(np.asarray(probs).ravel())
    n_outcomes = len(np.asarray(outcomes).ravel())
    if n_probs != n_outcomes:
        raise ValueError(
            f"len(probs) ({n_probs}) != len(outcomes) ({n_outcomes}): "
            "probs and outcomes must have the same length (SF-BL-9784)"
        )
    if n_probs == 0:
        raise ValueError(
            "brier_score is undefined on an empty panel: "
            "len(probs) == len(outcomes) == 0 (SF-BL-9784)"
        )
    for name, arr in (("probs", probs), ("outcomes", outcomes)):
        if not np.all(np.isfinite(np.asarray(arr, dtype=float))):
            raise ValueError(
                f"{name} must contain only finite values; got "
                "non-finite value(s) (NaN/inf) (SF-BL-9784)"
            )


def _validate_brier_inputs(probs: np.ndarray, outcomes: np.ndarray) -> None:
    """Raise ValueError if a Brier panel cannot be scored lawfully.

    SF-BL-10853: ``_require_same_length`` (SF-BL-9784) screens only
    length, emptiness and finiteness, so an impossible forecast or a
    fractional outcome was silently scored by the tournament/pooling
    path — ``brier_score([1.5, 0.6], [0, 1])`` returned 1.205 (a binary
    Brier above the lawful maximum 1.0) and ``brier_score([0.5, 0.6],
    [0.5, 1])`` returned 0.08, grading a forecast against a half-outcome
    that can never resolve. The sibling ``log_score`` in this module
    already screens both cases (forecast range SF-BL-10031, binary
    outcomes SF-BL-9785), and calibration.py screens both as well
    (``_validate_p_in_range`` / ``_as_y``, SF-BL-9484). This screen runs
    the SF-BL-9784 checks first (preserving their error precedence),
    then the [0, 1] forecast-range screen, then the binary-outcome
    screen — the same order as ``_validate_log_score_inputs``. Pure
    validation — it never transforms its inputs — so lawful panels
    (endpoints 0/1 included, bool outcomes included via the float
    conversion in ``brier_score``) are bit-identical to the pre-fix
    computation.
    """
    _require_same_length(probs, outcomes)
    probs_arr = np.asarray(probs, dtype=float)
    lo, hi = float(probs_arr.min()), float(probs_arr.max())
    if lo < 0.0 or hi > 1.0:
        raise ValueError(
            f"probs must lie in [0, 1]; got values outside that range "
            f"(min={lo!r}, max={hi!r}) (SF-BL-10853)"
        )
    outcomes_arr = np.asarray(outcomes, dtype=float)
    bad_outcomes = outcomes_arr[
        (outcomes_arr != 0.0) & (outcomes_arr != 1.0)
    ]
    if bad_outcomes.size:
        raise ValueError(
            f"outcomes must be exactly 0 or 1 for binary questions; got "
            f"outcome value(s) {bad_outcomes.tolist()!r} — refusing to "
            f"score an impossible outcome (SF-BL-10853)"
        )


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean Brier score for binary forecasts. Lower is better. Range [0, 1].

    Raises ValueError if ``probs`` and ``outcomes`` have different
    lengths, if the panel is empty, if any value is non-finite
    (SF-BL-9784), if any forecast lies outside [0, 1], or if any
    outcome is not exactly 0 or 1 (SF-BL-10853) — validation runs
    before any scoring arithmetic. Raises ValueError if any
    probability is a bool (SF-BL-10939), screened on the original
    input before the float conversion below would erase it; bool
    outcomes remain lawful. Raises ValueError naming probs or
    outcomes if either panel holds a string (SF-BL-11058), likewise
    screened on the original inputs before the float conversion
    below would silently parse a numeric string.
    """
    _reject_bool_probs(probs, "brier_score")
    _reject_string_inputs(probs, outcomes, "brier_score")
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    _validate_brier_inputs(probs, outcomes)
    return float(np.mean((probs - outcomes) ** 2))


def _validate_log_score_inputs(probs: np.ndarray, outcomes: np.ndarray) -> None:
    """Raise ValueError if a log-score panel cannot be scored lawfully.

    SF-BL-10031: without this screen a 1-element outcomes array silently
    broadcasts across the whole forecast panel (``log_score([0.5, 0.6],
    [1])`` returned 0.6020, grading two forecasts against copies of one
    outcome), an empty panel and a NaN forecast both returned NaN as an
    ordinary float, and ``log_score([1.5], [1])`` returned 1.0e-06 — a
    near-perfect loss for an impossible forecast — because ``clip_p``
    clipped 1.5 to ``1 - EPS`` before scoring. The sibling ``brier_score``
    in this module already screens length, emptiness and finiteness
    (SF-BL-9784); this screen mirrors it and adds the [0, 1] forecast-range
    check (precedent: calibration.py ``_validate_p_in_range``, SF-BL-9484),
    run BEFORE ``clip_p`` so the clip applies only to lawful endpoint
    forecasts 0/1. Outcome VALUES are binary-checked as the final screen
    (SF-BL-9785): without it ``log_score([0.5], [2])`` returned exactly
    ln(2) — an impossible outcome scored as a coin-flip loss — and
    ``log_score([0.8], [2])`` returned -1.163, a NEGATIVE log score,
    better than a perfect forecast. Non-finite outcomes are rejected by
    the finiteness loop above, before the binary screen runs. Pure
    validation — it never transforms its inputs — so clean panels are
    bit-identical to the pre-fix computation.
    """
    n_probs = len(np.asarray(probs).ravel())
    n_outcomes = len(np.asarray(outcomes).ravel())
    if n_probs != n_outcomes:
        raise ValueError(
            f"len(probs) ({n_probs}) != len(outcomes) ({n_outcomes}): "
            "probs and outcomes must have the same length (SF-BL-10031)"
        )
    if n_probs == 0:
        raise ValueError(
            "log_score is undefined on an empty panel: "
            "len(probs) == len(outcomes) == 0 (SF-BL-10031)"
        )
    for name, arr in (("probs", probs), ("outcomes", outcomes)):
        if not np.all(np.isfinite(np.asarray(arr, dtype=float))):
            raise ValueError(
                f"{name} must contain only finite values; got "
                "non-finite value(s) (NaN/inf) (SF-BL-10031)"
            )
    probs_arr = np.asarray(probs, dtype=float)
    lo, hi = float(probs_arr.min()), float(probs_arr.max())
    if lo < 0.0 or hi > 1.0:
        raise ValueError(
            f"probs must lie in [0, 1]; got values outside that range "
            f"(min={lo!r}, max={hi!r}) (SF-BL-10031)"
        )
    outcomes_arr = np.asarray(outcomes, dtype=float)
    bad_outcomes = outcomes_arr[
        (outcomes_arr != 0.0) & (outcomes_arr != 1.0)
    ]
    if bad_outcomes.size:
        raise ValueError(
            f"outcomes must be exactly 0 or 1 for binary questions; got "
            f"outcome value(s) {bad_outcomes.tolist()!r} — refusing to "
            f"score an impossible outcome (SF-BL-9785)"
        )


def log_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean log score (negative log-likelihood per question). Lower is better.

    Raises ValueError if ``probs`` and ``outcomes`` have different
    lengths, if the panel is empty, if any value is non-finite, if
    any forecast lies outside [0, 1] (SF-BL-10031), or if any outcome
    is not exactly 0 or 1 (SF-BL-9785) — validation runs before
    ``clip_p`` and before any scoring arithmetic, so the clip applies
    only to lawful endpoint forecasts 0/1. Raises ValueError if any
    probability is a bool (SF-BL-10939), screened on the original
    input before the float conversion below would erase it; bool
    outcomes remain lawful. Raises ValueError naming probs or
    outcomes if either panel holds a string (SF-BL-11058), likewise
    screened on the original inputs before the float conversion
    below would silently parse a numeric string.
    """
    _reject_bool_probs(probs, "log_score")
    _reject_string_inputs(probs, outcomes, "log_score")
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    _validate_log_score_inputs(probs, outcomes)
    probs = clip_p(probs)
    return float(-np.mean(outcomes * np.log(probs) + (1.0 - outcomes) * np.log(1.0 - probs)))


def _require_valid_calibration_panel(probs: np.ndarray, outcomes: np.ndarray) -> None:
    """Raise ValueError if the calibration panel is unusable.

    SF-BL-10032: ``calibration_stats`` binned and averaged with no panel
    validation at all — an empty panel returned ``{'ece': 0.0,
    'mean_forecast': NaN, 'base_rate': NaN}`` (perfect ECE from no data),
    a NaN forecast silently vanished from every bin (ece contribution
    0), a forecast of 1.5 returned ece 0.0 with mean_forecast 1.5, and a
    length mismatch died with a bare boolean-index IndexError. This
    screen runs before any binning and raises a named ValueError on a
    length mismatch, an empty panel, any non-finite value, or any value
    outside [0, 1] in either array — the brier_score (SF-BL-9784)
    standard plus the [0, 1] range screen this task's prereg froze.
    Pure validation — it never transforms its inputs — so clean panels
    are byte-identical to the pre-fix computation.

    SF-BL-10914: the [0, 1] range screen alone still let a fractional
    outcome through — ``calibration_stats([0.5, 0.6], [0.5, 1])``
    returned ECE 0.2 with base_rate 0.75, averaging a half-resolution
    into the bin frequencies, while calibration.py ``_as_y`` and this
    module's fixed ``brier_score`` / ``log_score`` (SF-BL-10853 /
    SF-BL-9785) all require outcomes in {0, 1}. The binary-outcome
    screen below runs last, after the SF-BL-10032 screens (preserving
    their error precedence: a non-finite or out-of-[0, 1] outcome still
    reports SF-BL-10032 first, exactly as the sibling validators run
    their range screens before the binary screen). Forecasts stay
    fractional — only outcomes are binary-checked. Pure validation, so
    binary panels (bool outcomes included, via the float conversion in
    ``calibration_stats``) are byte-identical to the pre-fix computation.
    """
    n_probs = len(np.asarray(probs).ravel())
    n_outcomes = len(np.asarray(outcomes).ravel())
    if n_probs != n_outcomes:
        raise ValueError(
            f"calibration_stats: len(probs) ({n_probs}) != len(outcomes) "
            f"({n_outcomes}): probs and outcomes must have the same "
            "length (SF-BL-10032)"
        )
    if n_probs == 0:
        raise ValueError(
            "calibration_stats is undefined on an empty panel: "
            "len(probs) == len(outcomes) == 0 (SF-BL-10032)"
        )
    for name, arr in (("probs", probs), ("outcomes", outcomes)):
        values = np.asarray(arr, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"calibration_stats: {name} must contain only finite "
                "values; got non-finite value(s) (NaN/inf) (SF-BL-10032)"
            )
        if float(values.min()) < 0.0 or float(values.max()) > 1.0:
            raise ValueError(
                f"calibration_stats: {name} must lie in [0, 1]; got "
                f"min={float(values.min())}, max={float(values.max())} "
                "(SF-BL-10032)"
            )
    outcomes_arr = np.asarray(outcomes, dtype=float)
    bad_outcomes = outcomes_arr[
        (outcomes_arr != 0.0) & (outcomes_arr != 1.0)
    ]
    if bad_outcomes.size:
        raise ValueError(
            f"calibration_stats: outcomes must be exactly 0 or 1 for "
            f"binary questions; got outcome value(s) "
            f"{bad_outcomes.tolist()!r} — refusing to score an "
            f"impossible outcome (SF-BL-10914)"
        )


def calibration_stats(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """Simple ECE-style calibration summary.

    Raises ValueError if ``probs`` and ``outcomes`` have different
    lengths, if the panel is empty, or if any value in either array is
    non-finite or outside [0, 1] (SF-BL-10032), or if any outcome is
    not exactly 0 or 1 (SF-BL-10914) — validation runs before
    any binning. Raises ValueError if any probability is a bool
    (SF-BL-10939), screened on the original input before the float
    conversion below would erase it; bool outcomes remain lawful.
    Raises ValueError naming probs or outcomes if either panel
    holds a string (SF-BL-11058), likewise screened on the original
    inputs before the float conversion below would silently parse
    a numeric string.

    Raises ValueError if ``n_bins`` is not a positive integer
    (SF-BL-9786): ``n_bins`` was never validated, so ``n_bins=0`` made
    ``range(n_bins)`` empty and returned ``ece == 0.0`` — a perfect-
    calibration reading on real forecasts — while negative values
    failed only incidentally inside ``np.linspace`` and non-integers
    raised ``TypeError``. Mirrors ``calibration._validate_n_bins``
    (SF-BL-9722): booleans, non-``(int, np.integer)`` types (including
    integral floats such as ``10.0``) and values < 1 are all rejected
    before any binning, and a lawful ``n_bins`` is never reassigned,
    so valid calls are bit-identical to the pre-fix code.
    """
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)):
        raise ValueError(
            f"calibration_stats: n_bins must be a positive integer "
            f"(>= 1); got n_bins={n_bins!r} ({type(n_bins).__name__}) "
            "(SF-BL-9786)"
        )
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError(
            f"calibration_stats: n_bins must be a positive integer "
            f"(>= 1); got n_bins={n_bins} (SF-BL-9786)"
        )
    _reject_bool_probs(probs, "calibration_stats")
    _reject_string_inputs(probs, outcomes, "calibration_stats")
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    _require_valid_calibration_panel(probs, outcomes)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        mask = (probs >= edges[b]) & (probs < edges[b + 1] if b < n_bins - 1 else probs <= edges[b + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.mean()) * abs(probs[mask].mean() - outcomes[mask].mean())
    return {"ece": float(ece), "mean_forecast": float(probs.mean()), "base_rate": float(outcomes.mean())}


# ---------------------------------------------------------------------------
# Forecast objects
# ---------------------------------------------------------------------------

@dataclass
class AgentForecast:
    """One agent's forecast on one question."""
    agent_id: str
    question_id: str
    probability: float                 # in (0, 1)
    round: int = 1                     # 1 = independent, 2 = post-debate revision
    rationale: str = ""
    strongest_counterevidence: str = ""  # required in debate round (DESIGN.md §4)
    confidence_note: str = ""

    def __post_init__(self) -> None:
        # SF-BL-10769: reject, never clip, a corrupt probability. The old
        # body resolved finite out-of-(0, 1) values with np.clip to the EPS
        # bound, so a percent-unit 82.0 was stored as 0.999999 — the best
        # forecast the pool can receive — and fed aggregate() directly;
        # NaN passed through the clip untouched (SF-BL-9789). Pure
        # validation: a lawful in-range probability is never reassigned,
        # so in-range construction is bit-identical to the pre-fix code.
        p = float(self.probability)
        if not math.isfinite(p):
            raise ValueError(
                f"AgentForecast probability must be finite and in "
                f"(0, 1); got {self.probability!r} (SF-BL-10769)"
            )
        if not (0.0 < p < 1.0):
            raise ValueError(
                f"AgentForecast probability must lie in (0, 1); got "
                f"{self.probability!r} — refusing to clip a finite "
                "out-of-range forecast into near-certainty (SF-BL-10769)"
            )

    @property
    def log_odds(self) -> float:
        return float(logit(self.probability))


@dataclass
class Question:
    """Question specification (see DESIGN.md §1 for the full schema)."""
    question_id: str
    text: str
    resolution_criteria: str
    horizon_days: int
    base_rate: Optional[float] = None        # reference-class frequency, if known
    base_rate_source: Optional[str] = None
    asset_class: str = "general"             # markets | prediction_markets | crypto
    resolves_at: Optional[str] = None        # ISO date string


@dataclass
class ForecasterRecord:
    """Running track record for one agent, used for pool weighting."""
    agent_id: str
    briers: List[float] = field(default_factory=list)
    log_scores: List[float] = field(default_factory=list)

    def add(self, probability: float, outcome: int) -> None:
        # SF-BL-8800: validate BEFORE scoring. The old body clipped the
        # probability but never validated the outcome, so add(0.8, 2)
        # recorded Brier 1.44 and add(0.8, -1) recorded 3.24 — impossible
        # for a binary question (max 1.0) — plus negative log scores;
        # NaN probabilities stored NaN Briers and +inf was clipped to
        # 1-EPS, laundering it into a near-perfect forecast. Those values
        # flow straight into track_record_weights' exp(-12*mean_brier).
        # Pure validation, run before any mutation (a rejected call
        # appends nothing): finite probabilities keep the exact pristine
        # clipping, so every lawful call is bit-identical.
        if isinstance(probability, (str, bytes)):
            raise ValueError(
                f"ForecasterRecord.add probability must be a finite "
                f"number; got {probability!r} (SF-BL-8800)"
            )
        try:
            p_raw = float(probability)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"ForecasterRecord.add probability must be a finite "
                f"number; got {probability!r} (SF-BL-8800)"
            ) from exc
        if not math.isfinite(p_raw):
            raise ValueError(
                f"ForecasterRecord.add probability must be finite; got "
                f"{probability!r} — refusing to clip/score a non-finite "
                f"forecast (SF-BL-8800)"
            )
        # SF-BL-10739: range-screen the FINITE probability before the
        # clip. SF-BL-8800 stopped NaN/inf and bad outcomes, but a finite
        # corrupt value (percent units, e.g. 82.0) still passed and
        # np.clip laundered it to 1-EPS, recording Brier 1e-12 — the
        # best score in the record — and inflating this agent's weight
        # in track_record_weights. The lawful range is the CLOSED
        # interval [0, 1]: endpoints 0.0/1.0 keep the exact pristine
        # clipping, so every lawful call stays bit-identical. Pure
        # validation, before any mutation (a rejected call appends
        # nothing).
        if p_raw < 0.0 or p_raw > 1.0:
            raise ValueError(
                f"ForecasterRecord.add probability must lie in [0, 1]; "
                f"got {probability!r} — refusing to clip a finite "
                f"out-of-range forecast into a near-perfect Brier "
                f"(SF-BL-10739)"
            )
        if isinstance(outcome, (bool, np.bool_)):
            o = int(outcome)
        elif isinstance(outcome, (int, np.integer)):
            o = int(outcome)
            if o not in (0, 1):
                raise ValueError(
                    f"ForecasterRecord.add outcome must be 0 or 1 for a "
                    f"binary question; got {outcome!r} — refusing to "
                    f"record an impossible Brier (SF-BL-8800)"
                )
        elif isinstance(outcome, (float, np.floating)):
            if not math.isfinite(float(outcome)) or float(outcome) not in (0.0, 1.0):
                raise ValueError(
                    f"ForecasterRecord.add outcome must be 0 or 1 for a "
                    f"binary question; got {outcome!r} — refusing to "
                    f"record an impossible Brier (SF-BL-8800)"
                )
            o = int(outcome)
        else:
            raise ValueError(
                f"ForecasterRecord.add outcome must be 0 or 1 for a "
                f"binary question; got {outcome!r} (SF-BL-8800)"
            )
        p = float(np.clip(p_raw, EPS, 1 - EPS))
        self.briers.append((p - o) ** 2)
        self.log_scores.append(-(o * math.log(p) + (1 - o) * math.log(1 - p)))

    @property
    def n(self) -> int:
        return len(self.briers)

    @property
    def mean_brier(self) -> float:
        return float(np.mean(self.briers)) if self.briers else 0.25  # ignorance prior

    @property
    def mean_log_score(self) -> float:
        return float(np.mean(self.log_scores)) if self.log_scores else math.log(2)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _reject_bool_weights(weights, caller: str) -> None:
    """Raise ValueError naming weights if a weight vector holds bools.

    SF-BL-11023: a bool satisfies every existing weights screen after
    conversion, because the weights paths convert with
    ``np.asarray(weights, dtype=float)``, which silently coerces it —
    ``log_odds_pool([0.7, 0.3], weights=[True, False])`` returned
    exactly 0.7, bit-identical to the intentional integer mask
    ``weights=[1, 0]``, silently collapsing the agent pool to its
    first forecaster, and ``mean_pool`` / ``median_pool`` on the same
    weights likewise returned 0.7. The finiteness screen (SF-BL-9788,
    and SF-BL-8799 in ``_normalized_weights``) runs on the
    already-converted vector, where 1.0/0.0 are lawful weights — no
    screen claimed a bool *weight* in this module (precedents:
    SF-BL-10939 bool probs in this module, SF-BL-10883 bool weights
    in calibration.py's ``_norm_weights``). A bool is not a weight:
    ``bool`` is an ``int`` subclass in Python, so the screen must
    test type, not value, and it therefore runs on the ORIGINAL
    weights, before any ``np.asarray(..., dtype=float)`` conversion
    erases the bool: it catches a scalar bool / np.bool_, a
    bool-dtype array, an object-dtype array holding bools, and bools
    nested in lists/tuples (a mixed list such as ``[0.5, True]``
    promotes to float64 under ``np.asarray``, so dtype inspection
    alone would miss it). Pure validation — it never transforms its
    input — so genuine numeric weight vectors (float/int lists,
    tuples, ndarrays, NumPy scalars, including an intentional
    integer 1/0 mask) are bit-identical to the pre-fix computation.
    String weights keep their own screen (SF-BL-11250), which runs
    first at both call sites, preserving its precedence.
    """
    def _has_bool(x) -> bool:
        if isinstance(x, (bool, np.bool_)):
            return True
        if isinstance(x, np.ndarray):
            if x.dtype.kind == "b":
                return True
            if x.dtype == object:
                return any(_has_bool(v) for v in x.flat)
            return False
        dtype = getattr(x, "dtype", None)
        if dtype is not None and getattr(dtype, "kind", None) == "b":
            return True
        if isinstance(x, (list, tuple)):
            return any(_has_bool(v) for v in x)
        return False

    if _has_bool(weights):
        shown = repr(weights)
        if len(shown) > 120:
            shown = shown[:117] + "..."
        raise ValueError(
            f"{caller}: weights must be real numbers, not bool; "
            f"got bool value(s) in weights {shown} — a bool is "
            "not a weight and must not be silently coerced into a "
            "1/0 pool mask (True/False coerce to 1.0/0.0) "
            "(SF-BL-11023)"
        )


def _reject_string_weights(weights, caller: str) -> None:
    """Raise ValueError naming weights if a weight vector holds strings.

    SF-BL-11250: a numeric string satisfies every existing weights
    screen after conversion, because the weights paths convert with
    ``np.asarray(weights, dtype=float)``, which silently parses it —
    ``log_odds_pool([0.7, 0.3], weights=['0.5', '0.5'])`` returned
    0.5, bit-identical to the numeric control ``weights=[0.5, 0.5]``,
    and ``aggregate(strategy='weighted_log_odds', weights=['0.5',
    '0.5'])`` likewise coerced. calibration.py's string-weight guard
    (SF-BL-11090, in that module's ``_norm_weights``) does not reach
    this divergent copy, which is the production tournament /
    aggregate path. A string is not a number (precedents: SF-BL-11090
    in calibration.py, SF-BL-11058 / SF-BL-11093 in this module for
    probs). This screen therefore runs on the ORIGINAL weights,
    before any ``np.asarray(..., dtype=float)`` conversion erases
    the string type: it catches a scalar str / bytes, a string-dtype
    (kind ``U`` / ``S``) array, a mixed list such as ``[0.5, '0.5']``,
    strings nested in lists/tuples, and object-dtype arrays holding
    strings. Pure validation — it never transforms its input — so
    genuine numeric weight vectors (float/int lists, tuples,
    ndarrays, NumPy scalars, and bool weights, whose behaviour is
    SF-BL-11023's distinct scope and is unchanged here) are
    bit-identical to the pre-fix computation.
    """
    def _has_string(x) -> bool:
        if isinstance(x, (str, bytes, bytearray, np.str_, np.bytes_)):
            return True
        if isinstance(x, np.ndarray):
            if x.dtype.kind in ("U", "S"):
                return True
            if x.dtype == object:
                return any(_has_string(v) for v in x.flat)
            return False
        dtype = getattr(x, "dtype", None)
        if dtype is not None and getattr(dtype, "kind", None) in ("U", "S"):
            return True
        if isinstance(x, (list, tuple)):
            return any(_has_string(v) for v in x)
        return False

    if _has_string(weights):
        shown = repr(weights)
        if len(shown) > 120:
            shown = shown[:117] + "..."
        raise ValueError(
            f"{caller}: weights must be real numbers, not string; "
            f"got string value(s) in weights {shown} — a numeric "
            "string is not a number and must not be silently parsed "
            "into a pool weight (SF-BL-11250)"
        )


def _normalized_weights(n: int, weights: Sequence[float], caller: str = "_normalized_weights") -> np.ndarray:
    """Validate + normalise caller-supplied pool weights (SF-BL-8799).

    Raises ValueError on a length mismatch or any non-finite weight —
    never silently discards or NaN-pools them. A non-positive total
    mirrors log_odds_pool's existing fallback to equal weights.
    Returns weights summing to 1.

    Raises ValueError naming weights on any string weight
    (SF-BL-11250): the raw screen runs before the float conversion
    below could silently parse a numeric string (see
    ``_reject_string_weights``), so ``aggregate`` (which validates
    through here for every strategy) and the mean/median pools all
    reject a text-serialised weight vector instead of honouring it.

    Raises ValueError naming weights on any bool weight
    (SF-BL-11023): the raw screen runs before the float conversion
    below could silently coerce True/False to 1.0/0.0 (see
    ``_reject_bool_weights``), so ``aggregate`` and the mean/median
    pools all reject a boolean mask or flag instead of honouring it
    as a 1/0 member mask.
    """
    _reject_string_weights(weights, caller)
    _reject_bool_weights(weights, caller)
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(
            f"weights must have length {n} (one per forecast); got shape "
            f"{w.shape} (SF-BL-8799)"
        )
    if not np.all(np.isfinite(w)):
        raise ValueError(
            "weights must all be finite; got non-finite weight(s) "
            f"{w[~np.isfinite(w)]} (SF-BL-8799)"
        )
    if w.sum() <= 0:
        return np.full(n, 1.0 / n) if n else w.copy()
    return w / w.sum()


def _require_nonempty_pool(n: int, caller: str, arg: str) -> None:
    """Raise ValueError if a pool would be computed from zero forecasts.

    SF-BL-8902: ``aggregate([], strategy='mean')`` returned NaN under a
    RuntimeWarning only, and the other strategies were worse, because
    each empty path fabricates a different plausible-looking number:
    mean / median return NaN (``np.mean`` / ``np.median`` of an empty
    slice), the log-odds strategies return exactly 0.0 (the mean logit
    is NaN and the scalar ``sigmoid`` overflow guard launders NaN to
    0.0), the extremized strategies return 1.58e-08 (extremizing that
    laundered 0.0), ``aggregate([], weights=[])`` returns 0.0 via the
    empty dot product, and ``log_odds_pool([], weights=[])`` returns
    0.5 — a coin flip manufactured from no forecasts at all. An
    all-absent question (every agent abstained / failed) therefore
    propagated a pooled probability downstream as if it were real. The
    sibling ``engine.log_odds_pool`` already raises on empty
    ("refusing to pool to NaN"), and this module's scoring functions
    raise on empty panels (SF-BL-9784 / 10031 / 10032); the production
    dispatch (DESIGN.md §5) now does the same. A call-site census found
    NO production caller of ``aggregate()`` that pre-checks
    non-emptiness (the only in-repo callers are tests), so the kill
    condition is not met and the guard belongs here. Pure validation —
    it never transforms its inputs — so non-empty pools are
    bit-identical to the pre-fix computation.
    """
    if n == 0:
        raise ValueError(
            f"{caller} is undefined on an empty forecast list: "
            f"len({arg}) == 0 — refusing to pool zero forecasts into "
            "a fabricated probability (NaN, 0.0 or 0.5 depending on "
            "the path); an all-absent question must abstain "
            "(SF-BL-8902)"
        )


def _reject_string_pool_members(probs, caller: str) -> None:
    """Raise ValueError naming probs if a mean/median pool panel holds strings.

    SF-BL-11093: a numeric string satisfies every existing mean/median
    pool screen after conversion, because both pools convert with
    ``np.asarray(probs, dtype=float)``, which silently parses it —
    ``mean_pool(['0.7', '0.3'])`` returned 0.5 and
    ``median_pool(['0.7', '0.3', '0.9'])`` returned 0.7, so a
    text-serialised member forecast (CSV / JSON round-trips make
    every field text) was honoured as a lawful probability by the
    simplest pools in the production dispatch helper. The sibling
    finite (SF-BL-9787) and range (SF-BL-10738) screens run on the
    converted array, so they can never see the string type. A string
    is not a number (precedent: SF-BL-11058's ``_reject_string_inputs``
    for this module's scorers, SF-BL-11054 in engine.py). This screen
    therefore runs on the ORIGINAL input, before any
    ``np.asarray(..., dtype=float)`` conversion erases the string
    type: it catches a scalar str / bytes, a string-dtype (kind
    ``U`` / ``S``) array, a mixed list such as ``[0.5, '0.6']``
    (which promotes to a string dtype under ``np.asarray``), strings
    nested in lists/tuples, and object-dtype arrays holding strings.
    A bare string panel (``mean_pool('0.7')``), which pre-fix died
    with a bare ``TypeError: len() of unsized object`` inside the
    converted-array screens, now gets this named ValueError instead.
    Pure validation — it never transforms its input — so genuine
    numeric panels (float/int lists, tuples, ndarrays and NumPy
    scalars, endpoints included) are bit-identical to the pre-fix
    computation. Only ``probs`` is screened: supplied ``weights``
    keep their existing SF-BL-8799 validation, and bool members are
    screened separately by :func:`_reject_bool_pool_members`
    (SF-BL-11388), which runs immediately before this screen.
    """
    def _has_string(x) -> bool:
        if isinstance(x, (str, bytes, bytearray, np.str_, np.bytes_)):
            return True
        if isinstance(x, np.ndarray):
            if x.dtype.kind in ("U", "S"):
                return True
            if x.dtype == object:
                return any(_has_string(v) for v in x.flat)
            return False
        dtype = getattr(x, "dtype", None)
        if dtype is not None and getattr(dtype, "kind", None) in ("U", "S"):
            return True
        if isinstance(x, (list, tuple)):
            return any(_has_string(v) for v in x)
        return False

    if _has_string(probs):
        shown = repr(probs)
        if len(shown) > 120:
            shown = shown[:117] + "..."
        raise ValueError(
            f"{caller}: probs must be real numbers, not string; "
            f"got string value(s) in {shown} — a numeric string "
            "is not a number and must not be silently parsed "
            "into a pooled probability (SF-BL-11093)"
        )


def _reject_bool_pool_members(probs, caller: str) -> None:
    """Raise ValueError naming probs if a mean/median pool panel holds bools.

    SF-BL-11388: a bool satisfies every existing mean/median pool
    screen after conversion, because both pools convert with
    ``np.asarray(probs, dtype=float)``, which silently coerces it —
    ``mean_pool([True, False])`` returned 0.5, bit-identical to the
    lawful ``[1.0, 0.0]`` control, and ``median_pool([True, False,
    True])`` returned 1.0, bit-identical to ``[1.0, 0.0, 1.0]`` — so
    a boolean flag leaking into a member field (a mask, a settled /
    hit flag, a JSON true/false) was honoured as the most extreme
    lawful forecasts in the panel, 1.0 / 0.0, by the simplest pools
    in the production dispatch helper. The sibling finite
    (SF-BL-9787) and range (SF-BL-10738) screens run on the
    converted array and pass bools (True -> 1.0 lies inside
    [0, 1]), so they can never see the bool type; the SF-BL-11093
    string screen fixtures only the string dimension. A bool is
    not a probability (precedents: SF-BL-11020, calibration.py's
    ``trimmed_mean_pool`` bool guard in a different module;
    SF-BL-10939, this module's ``_reject_bool_probs`` for the
    scorers). This screen therefore runs on the ORIGINAL input,
    before any ``np.asarray(..., dtype=float)`` conversion erases
    the bool type: it catches a scalar bool / ``np.bool_``, a
    bool-dtype array, a mixed list such as ``[0.5, True]`` (which
    promotes to float64 under ``np.asarray``, so dtype inspection
    alone would miss it), bools nested in lists/tuples, and
    object-dtype arrays holding bools. It also covers pandas
    panels without importing pandas (detected by the type's
    module): a ``bool`` / ``boolean``-dtype Series / Index holds
    bools by construction, a category dtype whose categories
    contain a bool is a bool panel, an object-dtype Series holding
    bool leaves is scanned elementwise, and a DataFrame is scanned
    by column dtype and values — mirroring calibration.py's
    SF-BL-11020 pandas handling for ``trimmed_mean_pool``. A bare
    bool panel (``mean_pool(True)``), which pre-fix died with a
    bare ``TypeError: len() of unsized object`` inside the
    converted-array screens, now gets this named ValueError
    instead. Pure validation — it never transforms its input —
    so genuine numeric panels (float/int lists, tuples, ndarrays
    and NumPy scalars, endpoints 0 / 1 included: the screen is on
    type, not value) are bit-identical to the pre-fix computation.
    Only ``probs`` is screened: supplied ``weights`` keep their
    existing SF-BL-8799 validation (bool weights already raise
    there), and string members keep their SF-BL-11093 screen,
    which runs immediately after this one.
    """
    def _has_bool(x) -> bool:
        if isinstance(x, (bool, np.bool_)):
            return True
        if isinstance(x, np.ndarray):
            if x.dtype.kind == "b":
                return True
            if x.dtype == object:
                return any(_has_bool(v) for v in x.flat)
            return False
        dtype = getattr(x, "dtype", None)
        if dtype is not None and getattr(dtype, "kind", None) == "b":
            return True
        if isinstance(x, (list, tuple)):
            return any(_has_bool(v) for v in x)
        return False

    def _has_bool_in_pandas_panel(x) -> bool:
        try:
            if not (type(x).__module__ or "").startswith("pandas"):
                return False
            if type(x).__name__ == "DataFrame":
                dtypes = getattr(x, "dtypes", None)
                if dtypes is not None:
                    for dt in list(dtypes):
                        if getattr(dt, "name", None) in ("bool", "boolean"):
                            return True
                        if getattr(dt, "kind", None) == "b":
                            return True
                        cats = getattr(dt, "categories", None)
                        if cats is not None and any(
                                _has_bool(c) for c in list(cats)):
                            return True
                values = getattr(x, "values", None)
                return values is not None and _has_bool(values)
            dtype = getattr(x, "dtype", None)
            if dtype is not None and getattr(dtype, "name", None) in (
                    "bool", "boolean"):
                return True
            cats = getattr(x, "categories", None)
            if cats is None and dtype is not None:
                cats = getattr(dtype, "categories", None)
            if cats is not None and any(_has_bool(c) for c in list(cats)):
                return True
            return any(_has_bool(v) for v in list(x))
        except Exception:  # noqa: BLE001 — inspection must never break lawful panels
            return False

    if _has_bool(probs) or _has_bool_in_pandas_panel(probs):
        shown = repr(probs)
        if len(shown) > 120:
            shown = shown[:117] + "..."
        raise ValueError(
            f"{caller}: probs must be real probabilities, not bool; "
            f"got bool value(s) in {shown} — a bool is not a "
            "probability and must not be silently coerced to the "
            "endpoint forecasts 1.0/0.0 in a pooled probability "
            "(SF-BL-11388)"
        )


def _require_pool_members_in_unit_interval(p: np.ndarray, caller: str) -> None:
    """Raise ValueError if any pool member lies outside [0, 1] (SF-BL-10738).

    Callers must run the finiteness screen first: NaN comparisons are
    all False, so a NaN member is invisible to this range screen and
    is rejected by the finite screen that precedes it. Endpoints 0.0
    and 1.0 are lawful (the preregistered domain is the closed
    interval [0, 1]); only a finite member strictly below 0 or
    strictly above 1 raises. Pure validation — it never transforms
    its input — so panels inside [0, 1] are bit-identical to the
    pre-fix computation.
    """
    flat = p.ravel()
    bad = flat[(flat < 0.0) | (flat > 1.0)]
    if bad.size:
        raise ValueError(
            f"{caller}: probs must lie in [0, 1]; got member(s) "
            f"outside that range {bad} — refusing to pool, since a "
            "finite out-of-range member (e.g. a percent-unit 82.0) "
            "is arithmetically indistinguishable from a lawful "
            "probability to np.mean / np.median: mean_pool([82.0, "
            "0.5]) returned 41.25 and median_pool silently sorted "
            "the corrupt member to the edge of the panel "
            "(SF-BL-10738)"
        )


def _require_valid_mean_median_pool(p: np.ndarray, caller: str) -> None:
    """Raise ValueError if a mean/median pool is empty or corrupt.

    SF-BL-9787: ``mean_pool([])`` / ``median_pool([])`` are already
    guarded on empty by SF-BL-8902, but a non-finite member passed
    straight through the arithmetic: ``mean_pool([0.5, nan])`` and
    ``median_pool([0.5, nan])`` both returned NaN as an ordinary float,
    ``[0.5, inf]`` returned inf, and the weighted median silently
    sorted a NaN member away (``median_pool([0.2, nan, 0.8],
    weights=equal)`` returned 0.8). ``aggregate()`` screens finite
    members at dispatch entry (SF-BL-10233), so the dispatch layer
    never reaches these functions with a corrupt panel, but direct
    callers do — ``run_tournament`` and any public caller — and no
    caller guarantees finiteness for the functions themselves, so the
    guard belongs here. Pure validation — it never transforms its
    input — so non-empty finite pools are bit-identical to the
    pre-fix computation. The empty message carries both tags
    (SF-BL-8902) (SF-BL-9787), precedent log_odds_pool's
    (SF-BL-8902) (SF-BL-10233).

    SF-BL-10738: a FINITE out-of-[0, 1] member passed every screen
    above — the log-odds strategies in the same dispatcher at least
    validate their weights and clip their members, while the scalar
    pools averaged the corrupt value verbatim: ``mean_pool([82.0,
    0.5])`` returned 41.25 (a "probability" forty-one times too
    large), ``median_pool([82.0, 0.5])`` returned 41.25, and
    ``median_pool([0.2, 82.0, 0.8])`` returned 0.8, silently
    absorbing the percent-unit member with no signal at all. A
    call-site census found NO production caller relying on
    unscreened mean/median output (the only in-repo production
    caller, ``run_tournament``, feeds sigmoid + ``clip_p`` outputs
    in [EPS, 1-EPS]; ``aggregate()`` itself has no production
    caller outside tests), so the kill condition is not met and the
    range screen belongs here, after the finiteness screen (NaN is
    invisible to range comparisons). Lawful panels in [0, 1] —
    endpoints included — are bit-identical to the pre-fix
    computation.
    """
    if len(p) == 0:
        raise ValueError(
            f"{caller} is undefined on an empty forecast list: "
            "len(probs) == 0 — refusing to pool zero forecasts into "
            "a fabricated probability (NaN, 0.0 or 0.5 depending on "
            "the path); an all-absent question must abstain "
            "(SF-BL-8902) (SF-BL-9787)"
        )
    if not np.all(np.isfinite(p)):
        flat = p.ravel()
        bad = flat[~np.isfinite(flat)]
        raise ValueError(
            f"{caller}: probs must all be finite; got non-finite "
            f"member(s) {bad} — refusing to pool, since np.mean / "
            "np.median propagate NaN/inf as an ordinary pooled "
            "probability and the weighted median silently sorts a "
            "NaN member away (SF-BL-9787)"
        )
    _require_pool_members_in_unit_interval(p, caller)


def mean_pool(probs: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
    """Arithmetic pool; weighted mean when `weights` is supplied (SF-BL-8799).

    Raises ValueError on an empty input (SF-BL-8902): pre-fix this
    returned NaN (or 0.0 on the weighted path).

    Raises ValueError on any non-finite member (NaN / ±inf)
    (SF-BL-9787): pre-fix ``mean_pool([0.5, nan])`` returned NaN and
    ``mean_pool([0.5, inf])`` returned inf as ordinary floats. The
    screen runs after the array is formed and before any pooling
    arithmetic or weights handling.

    Raises ValueError on any member outside [0, 1] (SF-BL-10738):
    pre-fix ``mean_pool([82.0, 0.5])`` returned 41.25 — a
    percent-unit member averaged verbatim into an impossible pooled
    "probability". The range screen runs after the finiteness
    screen and before any pooling arithmetic or weights handling;
    endpoints 0.0 and 1.0 remain lawful.

    Raises ValueError naming probs on any string member
    (SF-BL-11093): pre-fix ``mean_pool(['0.7', '0.3'])`` returned
    0.5 — a text-serialised member silently parsed by the
    ``np.asarray(..., dtype=float)`` conversion below. The screen
    runs on the original input, before that conversion, and before
    any pooling arithmetic or weights handling.

    Raises ValueError naming probs on any bool member
    (SF-BL-11388): pre-fix ``mean_pool([True, False])`` returned
    0.5, bit-identical to the lawful ``[1.0, 0.0]`` control — a
    boolean flag silently coerced by the ``np.asarray(...,
    dtype=float)`` conversion below to the endpoint forecast 1.0 /
    0.0. The screen runs on the original input, before that
    conversion, and before any pooling arithmetic or weights
    handling; int endpoints 1 / 0 remain lawful.
    """
    _reject_bool_pool_members(probs, "mean_pool")
    _reject_string_pool_members(probs, "mean_pool")
    p = np.asarray(probs, dtype=float)
    _require_valid_mean_median_pool(p, "mean_pool")
    if weights is None:
        return float(np.mean(p))
    return float(np.dot(_normalized_weights(len(p), weights, "mean_pool"), p))


def median_pool(probs: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
    """Median pool; weighted median when `weights` is supplied (SF-BL-8799).

    Weighted median: sort the forecasts, take the first whose cumulative
    normalised weight reaches 0.5; on an exact 0.5 tie with a next value,
    average the two (so equal weights reproduce np.median exactly in the
    even-n tie case). weights=None preserves the original np.median path.

    Raises ValueError on an empty input (SF-BL-8902): pre-fix this
    returned NaN (or died with a bare IndexError on the weighted path).

    Raises ValueError on any non-finite member (NaN / ±inf)
    (SF-BL-9787): pre-fix ``median_pool([0.5, nan])`` returned NaN,
    ``median_pool([0.5, inf])`` returned inf, and the weighted path
    silently sorted a NaN member away and returned a finite median
    of the remaining members. The screen runs after the array is
    formed and before any pooling arithmetic or weights handling.

    Raises ValueError on any member outside [0, 1] (SF-BL-10738):
    pre-fix ``median_pool([82.0, 0.5])`` returned 41.25 and
    ``median_pool([0.2, 82.0, 0.8])`` returned 0.8 — the
    percent-unit member was silently sorted to the edge of the
    panel and absorbed with no signal. The range screen runs after
    the finiteness screen and before any sorting, pooling
    arithmetic or weights handling; endpoints 0.0 and 1.0 remain
    lawful.

    Raises ValueError naming probs on any string member
    (SF-BL-11093): pre-fix ``median_pool(['0.7', '0.3', '0.9'])``
    returned 0.7 — a text-serialised member silently parsed by the
    ``np.asarray(..., dtype=float)`` conversion below and sorted
    into the panel. The screen runs on the original input, before
    that conversion, and before any sorting, pooling arithmetic or
    weights handling.

    Raises ValueError naming probs on any bool member
    (SF-BL-11388): pre-fix ``median_pool([True, False, True])``
    returned 1.0, bit-identical to the lawful ``[1.0, 0.0, 1.0]``
    control — a boolean flag silently coerced by the
    ``np.asarray(..., dtype=float)`` conversion below to the
    endpoint forecast 1.0 / 0.0 and sorted into the panel. The
    screen runs on the original input, before that conversion, and
    before any sorting, pooling arithmetic or weights handling;
    int endpoints 1 / 0 remain lawful.
    """
    _reject_bool_pool_members(probs, "median_pool")
    _reject_string_pool_members(probs, "median_pool")
    p = np.asarray(probs, dtype=float)
    _require_valid_mean_median_pool(p, "median_pool")
    if weights is None:
        return float(np.median(p))
    w = _normalized_weights(len(p), weights, "median_pool")
    order = np.argsort(p, kind="stable")
    ps, ws = p[order], w[order]
    cw = np.cumsum(ws)
    idx = int(np.searchsorted(cw, 0.5, side="left"))
    if idx >= len(ps):
        idx = len(ps) - 1
    if cw[idx] == 0.5 and idx < len(ps) - 1:
        return float((ps[idx] + ps[idx + 1]) / 2.0)
    return float(ps[idx])


def _require_finite_pool_members(probs, caller: str) -> np.ndarray:
    """Raise ValueError if any pool member is non-finite (SF-BL-10233).

    ``log_odds_pool`` clipped its way to a hard 0.0 from nothing:
    ``clip_p`` passes NaN through and clips ±inf to the EPS bounds, the
    logit of a NaN member is NaN, and the scalar ``sigmoid`` overflow
    guard (``x > -700`` is False for NaN) launders that NaN to exactly
    0.0 — a maximally confident No forecast pooled from zero usable
    information or from a corrupt member. ±inf members were likewise
    silently clipped to 1-EPS / EPS and pooled as near-certainty.
    ``aggregate([], strategy='weighted_extremized')``'s 1.58e-08 was
    this same laundered 0.0 after extremizing (via a member mutated to
    NaN after construction, which bypasses AgentForecast's own screen).
    A call-site census found NO production caller of ``aggregate()``
    that guarantees a non-empty finite list (the only in-repo callers
    are tests, and ``aggregate`` itself is the production dispatch), so
    the kill condition is not met and the guard belongs here. Pure
    validation — it never transforms a finite input — so non-empty
    finite panels are bit-identical to the pre-fix computation.
    Returns the float array so callers need not reconvert.
    """
    try:
        p = np.asarray(probs, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{caller}: probs must all be finite numbers; got a "
            f"non-numeric member in {probs!r} — refusing to pool a "
            "corrupt forecast panel (SF-BL-10233)"
        ) from exc
    flat = p.ravel()
    if not np.all(np.isfinite(flat)):
        bad = flat[~np.isfinite(flat)]
        raise ValueError(
            f"{caller}: probs must all be finite; got non-finite "
            f"member(s) {bad} — refusing to pool, since the clip / "
            "logit / sigmoid path launders NaN to exactly 0.0 and "
            "clips ±inf to the EPS bounds, fabricating a confident "
            "forecast from a corrupt member (SF-BL-10233)"
        )
    return p


def _require_log_odds_pool_members_in_unit_interval(p: np.ndarray) -> None:
    """Raise ValueError if any finite pool member lies outside [0, 1]
    (SF-BL-10935).

    Callers must run the finiteness screen first: NaN comparisons are
    all False, so a NaN member is invisible to this range screen and
    is rejected by ``_require_finite_pool_members`` (SF-BL-10233),
    which precedes it. Endpoints 0.0 and 1.0 are lawful (the
    preregistered domain is the closed interval [0, 1]) and still go
    through the existing ``clip_p`` EPS bound exactly as before; only
    a finite member strictly below 0 or strictly above 1 raises.
    Pre-fix, such a member was silently mapped by ``clip_p`` to
    1-EPS / EPS before pooling, so ``log_odds_pool([1.5, 0.5])``
    returned 0.999001 — a percent-unit 82.0 or a 1.5 entered the pool
    as the strongest lawful forecast the pool can receive and
    dominated it, exactly as AgentForecast's rejected clip did
    before SF-BL-10769. The sibling mean/median pools already raise
    on this case (SF-BL-10738). Pure validation — it never
    transforms its input — so panels inside [0, 1] are bit-identical
    to the pre-fix computation.
    """
    flat = p.ravel()
    bad = flat[(flat < 0.0) | (flat > 1.0)]
    if bad.size:
        raise ValueError(
            f"log_odds_pool: probs must lie in [0, 1]; got finite "
            f"member(s) outside that range {bad} — refusing to pool, "
            "since clip_p would clip an out-of-range member (e.g. a "
            "percent-unit 82.0 or a 1.5) to the 1-EPS bound and pool "
            "it as near-certainty: log_odds_pool([1.5, 0.5]) "
            "returned 0.999001 pre-fix (SF-BL-10935)"
        )


def _collect_log_odds_pool_type_defects(x, defects: set, in_object: bool = False,
                                        is_element: bool = False) -> None:
    """Collect 'string' / 'bool' / 'object' defects in a raw pool panel.

    SF-BL-11429 helper for :func:`log_odds_pool`. Mirrors the
    SF-BL-11393 / SF-BL-11433 fleet pattern: the scan runs on the
    ORIGINAL input, before any ``np.asarray(..., dtype=float)``
    conversion erases a member's type. A string carrier is a
    str / bytes / bytearray / memoryview / np.str_ / np.bytes_
    leaf, a string-dtype (kind ``U`` / ``S``) container, or a
    string leaf inside an object-dtype container; a bool defect
    is a bool / np.bool_ leaf or a bool-dtype container; an
    object defect is a bare non-numeric leaf (``object()``,
    ``None``, a dict, ...) carried as a panel element, in
    particular inside an object-dtype container. Object-dtype
    containers holding only plain numbers (int / float / complex
    / NumPy scalars / Decimal / Fraction) carry NO defect and
    stay lawful, exactly as pre-fix (fleet precedent
    SF-BL-11433, and SF-BL-11393 in this module). Pandas panels
    are detected by the type's module without importing pandas:
    bool / string dtypes, categorical categories, and
    object-dtype Series / Index leaves are scanned elementwise.
    """
    from decimal import Decimal
    from fractions import Fraction

    def _is_plain_number(v) -> bool:
        return isinstance(v, (int, float, complex, np.integer,
                              np.floating, np.complexfloating,
                              Decimal, Fraction))

    if isinstance(x, (str, bytes, bytearray, memoryview,
                      np.str_, np.bytes_)):
        defects.add("string")
        return
    if isinstance(x, (bool, np.bool_)):
        defects.add("bool")
        return
    if x is None or type(x) is object:
        defects.add("object")
        return
    if isinstance(x, np.ndarray):
        if x.dtype.kind in ("U", "S"):
            defects.add("string")
            return
        if x.dtype.kind == "b":
            defects.add("bool")
            return
        if x.dtype == object:
            for v in x.flat:
                _collect_log_odds_pool_type_defects(
                    v, defects, in_object=True, is_element=True)
        return
    if isinstance(x, (list, tuple)):
        for v in x:
            _collect_log_odds_pool_type_defects(
                v, defects, in_object=in_object, is_element=True)
        return
    if (type(x).__module__ or "").startswith("pandas"):
        try:
            if type(x).__name__ == "DataFrame":
                dtypes = getattr(x, "dtypes", None)
                if dtypes is not None:
                    for dt in list(dtypes):
                        if getattr(dt, "name", None) in ("bool", "boolean") \
                                or getattr(dt, "kind", None) == "b":
                            defects.add("bool")
                        if getattr(dt, "kind", None) in ("U", "S") \
                                or getattr(dt, "name", None) in ("string", "str"):
                            defects.add("string")
                        cats = getattr(dt, "categories", None)
                        if cats is not None:
                            _collect_log_odds_pool_type_defects(
                                cats, defects, in_object=in_object,
                                is_element=True)
                values = getattr(x, "values", None)
                if values is not None:
                    _collect_log_odds_pool_type_defects(
                        values, defects, in_object=in_object,
                        is_element=is_element)
                return
            dtype = getattr(x, "dtype", None)
            if dtype is not None:
                if getattr(dtype, "name", None) in ("bool", "boolean") \
                        or getattr(dtype, "kind", None) == "b":
                    defects.add("bool")
                    return
                if getattr(dtype, "kind", None) in ("U", "S") \
                        or getattr(dtype, "name", None) in ("string", "str"):
                    defects.add("string")
                    return
                cats = getattr(dtype, "categories", None)
                if cats is not None:
                    _collect_log_odds_pool_type_defects(
                        cats, defects, in_object=in_object,
                        is_element=True)
                if getattr(dtype, "kind", None) == "O":
                    for v in list(x):
                        _collect_log_odds_pool_type_defects(
                            v, defects, in_object=True, is_element=True)
            return
        except Exception:  # noqa: BLE001 — inspection must never break lawful panels
            return
    dtype = getattr(x, "dtype", None)
    if dtype is not None:
        kind = getattr(dtype, "kind", None)
        if kind in ("U", "S"):
            defects.add("string")
            return
        if kind == "b":
            defects.add("bool")
            return
        if kind == "O":
            try:
                for v in list(x):
                    _collect_log_odds_pool_type_defects(
                        v, defects, in_object=True, is_element=True)
            except TypeError:
                pass
            return
        return
    if not _is_plain_number(x) and (in_object or is_element):
        # A panel element that is neither a plain number nor a
        # container handled above (a bare object, a dict, a custom
        # scalar): never a lawful pool member — float() either
        # silently coerced the coercible forms or died on it with
        # a bare TypeError naming neither argument.
        defects.add("object")


def _shown_panel(x) -> str:
    shown = " ".join(repr(x).split())
    if len(shown) > 120:
        shown = shown[:117] + "..."
    return shown


def _reject_log_odds_pool_probs_types(probs) -> None:
    """Raise ValueError naming probs on a string / bool / object member.

    SF-BL-11429: ``log_odds_pool`` never type-validated its
    ``probs`` — the float conversion in
    ``_require_finite_pool_members`` silently parsed numeric
    strings and coerced bools, so
    ``log_odds_pool(['0.7', '0.3'])`` returned 0.5, bit-identical
    to the numeric control ``[0.7, 0.3]``, and
    ``log_odds_pool([True, False])`` returned
    0.49999999999640554, bit-identical to the ``[1.0, 0.0]``
    control — a text-serialised forecast (CSV / JSON round-trips)
    or a boolean flag was honoured as a lawful probability by
    the logit pool. The sibling screens fixtured only other
    dimensions: SF-BL-11023 only bool WEIGHTS for this function,
    SF-BL-10935 only finite out-of-range member values, and
    SF-BL-11090 string weights only in calibration.py's separate
    pools; the mean/median pools' string/bool member screens
    (SF-BL-11093 / SF-BL-11388) do not reach this function. This
    screen therefore runs on the ORIGINAL probs, before the
    empty check (whose ``np.asarray`` can itself die on an
    inhomogeneous text-carrier panel) and before any float
    conversion. Pure validation — it never transforms its
    input — so genuine numeric panels (float/int lists, tuples,
    ndarrays, NumPy scalars, Decimal/Fraction members, and
    object-dtype panels holding only plain numbers) are
    bit-identical to the pre-fix computation.
    """
    defects: set = set()
    _collect_log_odds_pool_type_defects(probs, defects)
    if "string" in defects:
        raise ValueError(
            "log_odds_pool: probs must be real numbers, not string; "
            f"got string or object-dtype value(s) in "
            f"{_shown_panel(probs)} — a numeric string is not a "
            "number and must not be silently parsed into a pooled "
            "probability (SF-BL-11429)"
        )
    if "bool" in defects:
        raise ValueError(
            "log_odds_pool: probs must be real probabilities, not "
            f"bool; got bool value(s) in {_shown_panel(probs)} — "
            "a bool is not a probability and must not be silently "
            "coerced to the endpoint forecasts 1.0/0.0 in a pooled "
            "probability (SF-BL-11429)"
        )
    if "object" in defects:
        raise ValueError(
            "log_odds_pool: probs must be real numbers, not object; "
            f"got object-dtype / object value(s) in "
            f"{_shown_panel(probs)} — a bare object-dtype element "
            "is not a number and must not be pooled as a "
            "probability (SF-BL-11429)"
        )


def _reject_log_odds_pool_weights_types(weights) -> None:
    """Raise ValueError naming weights on a string / bool / object weight.

    SF-BL-11429 (weights side): the task's repro
    ``log_odds_pool([0.7, 0.3], weights=['1', '1'])`` returned
    0.5 pre-fixture. String weights were since fixtured by
    SF-BL-11250 and bool weights by SF-BL-11023, and this screen
    preserves those tags (string first, then bool — the
    precedence of the screens it replaces at this call site)
    while adding this task's tag, so both the prior and the
    SF-BL-11429 fixtures match at the ``log_odds_pool`` call
    site. It additionally covers the forms neither prior screen
    reaches: a memoryview weight (the shared string screen's
    carrier tuple omits it, and the float conversion died on it
    with a bare numpy error) and a bare non-numeric object /
    object-dtype weight element (pre-fix a bare ``TypeError``
    from ``float()``, naming neither weights nor any tag). Runs
    on the ORIGINAL weights, in the weights branch, after the
    probs screens (preserving their precedence) and before the
    float conversion. Object-dtype weight vectors holding only
    plain numbers stay lawful and bit-identical, as do genuine
    numeric weight vectors of every pre-fix form.
    """
    defects: set = set()
    _collect_log_odds_pool_type_defects(weights, defects)
    if "string" in defects:
        raise ValueError(
            "log_odds_pool: weights must be real numbers, not string; "
            f"got string value(s) in weights {_shown_panel(weights)} "
            "— a numeric string is not a number and must not be "
            "silently parsed into a pool weight "
            "(SF-BL-11250) (SF-BL-11429)"
        )
    if "bool" in defects:
        raise ValueError(
            "log_odds_pool: weights must be real numbers, not bool; "
            f"got bool value(s) in weights {_shown_panel(weights)} "
            "— a bool is not a weight and must not be silently "
            "coerced into a 1/0 pool mask (True/False coerce to "
            "1.0/0.0) (SF-BL-11023) (SF-BL-11429)"
        )
    if "object" in defects:
        raise ValueError(
            "log_odds_pool: weights must be real numbers, not object; "
            f"got object-dtype / object value(s) in weights "
            f"{_shown_panel(weights)} — a bare object-dtype element "
            "is not a number and must not be pooled as a weight "
            "(SF-BL-11429)"
        )


def log_odds_pool(probs: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
    """
    Weighted log-odds (geometric-odds) pool.

        logit(p_pool) = sum_i w_i * logit(p_i) / sum_i w_i
        p_pool = sigmoid(logit(p_pool))

    Equal weights when `weights` is None.

    Raises ValueError if any supplied weight is non-finite (NaN / ±inf):
    a NaN weight makes ``w.sum() <= 0`` False, so the equal-weight fallback
    below is bypassed, the weights normalise to NaN, and the scalar sigmoid
    path silently emits 0.0 — a maximally confident wrong-direction pool
    presented as valid (SF-BL-9788). Finite-weight behaviour is unchanged.

    Raises ValueError on an empty input (SF-BL-8902): pre-fix this
    returned 0.0 (the NaN mean logit laundered by ``sigmoid``), or
    exactly 0.5 on the empty-weights path — a fabricated coin flip.

    Raises ValueError on any non-finite member (NaN / ±inf)
    (SF-BL-10233): pre-fix ``log_odds_pool([nan])`` returned exactly
    0.0 and ``log_odds_pool([inf])`` returned 0.999999 (clipped to
    1-EPS) — a confident forecast fabricated from a corrupt member.
    The screen runs before any clipping / logit arithmetic, on the
    raw members; finite members are never transformed by it.

    Raises ValueError on any finite member outside [0, 1]
    (SF-BL-10935): pre-fix ``log_odds_pool([1.5, 0.5])`` returned
    0.999001 and ``log_odds_pool([82.0, 0.5])`` returned the same —
    ``clip_p`` mapped the impossible member to 1-EPS before pooling,
    so it dominated the pool as near-certainty. The range screen
    runs after the finiteness screen and before ``clip_p`` / logit
    arithmetic; endpoints 0.0 and 1.0 remain lawful and are clipped
    exactly as before, so lawful weighted and unweighted pools are
    bit-identical to the pre-fix computation.

    Raises ValueError naming weights on any string weight
    (SF-BL-11250): pre-fix ``log_odds_pool([0.7, 0.3], weights=
    ['0.5', '0.5'])`` returned 0.5, bit-identical to the numeric
    control — the ``np.asarray(weights, dtype=float)`` conversion
    below silently parsed the text-serialised weights. The raw
    screen (``_reject_string_weights``) runs on the original weights
    in the weights branch, before that conversion and after the
    probs screens above (preserving their precedence); bool weights
    keep their pre-fix behaviour (SF-BL-11023's distinct scope).

    Raises ValueError naming weights on any bool weight
    (SF-BL-11023): pre-fix ``log_odds_pool([0.7, 0.3], weights=
    [True, False])`` returned exactly 0.7 — the
    ``np.asarray(weights, dtype=float)`` conversion below silently
    coerced the boolean mask to ``[1.0, 0.0]``, which passes the
    SF-BL-9788 finiteness screen and the positive-sum check, and
    collapsed the pool to its first forecaster. The raw screen
    (``_reject_bool_weights``) runs on the original weights in the
    weights branch, before that conversion, after the string screen
    above (preserving its precedence) and after the probs screens;
    lawful float/int weights — including an intentional integer
    1/0 mask — are bit-identical to the pre-fix computation.

    Raises ValueError naming probs on any string or object-dtype
    element in probs, or any bool element in probs (SF-BL-11429):
    pre-fix ``log_odds_pool(['0.7', '0.3'])`` returned 0.5,
    bit-identical to the numeric control, and
    ``log_odds_pool([True, False])`` returned
    0.49999999999640554, bit-identical to the ``[1.0, 0.0]``
    control — the float conversion inside
    ``_require_finite_pool_members`` parsed the strings and
    coerced the bools before any screen could see their types.
    The raw screen (``_reject_log_odds_pool_probs_types``) runs
    on the original probs at entry, before the empty check and
    before that conversion; object-dtype panels holding only
    plain numbers stay lawful.

    Raises ValueError naming weights on any string or
    object-dtype element in weights (SF-BL-11429, with the
    SF-BL-11250 / SF-BL-11023 tags preserved for the string /
    bool forms at this call site): pre-fixture
    ``log_odds_pool([0.7, 0.3], weights=['1', '1'])`` returned
    0.5, and a bare object-dtype weight died with a bare
    ``TypeError`` naming neither weights nor any tag. The raw
    screen (``_reject_log_odds_pool_weights_types``) runs on the
    original weights in the weights branch, before the float
    conversion and after the probs screens above (preserving
    their precedence), with the SF-BL-11250 / SF-BL-11023
    screens kept as backstops; lawful float/int weights are
    bit-identical to the pre-fix computation.
    """
    _reject_log_odds_pool_probs_types(probs)
    if len(np.asarray(probs).ravel()) == 0:
        raise ValueError(
            "log_odds_pool is undefined on an empty forecast list: "
            "len(probs) == 0 — refusing to pool zero forecasts into "
            "a fabricated probability (NaN, 0.0 or 0.5 depending on "
            "the path); an all-absent question must abstain "
            "(SF-BL-8902) (SF-BL-10233)"
        )
    raw = _require_finite_pool_members(probs, "log_odds_pool")
    _require_log_odds_pool_members_in_unit_interval(raw)
    p = clip_p(raw)
    lo = np.log(p / (1.0 - p))
    if weights is None:
        pooled_lo = float(np.mean(lo))
    else:
        _reject_log_odds_pool_weights_types(weights)
        _reject_string_weights(weights, "log_odds_pool")
        _reject_bool_weights(weights, "log_odds_pool")
        w = np.asarray(weights, dtype=float)
        if not np.all(np.isfinite(w)):
            raise ValueError(
                "log_odds_pool: weights must all be finite; got non-finite "
                f"weight(s) {w[~np.isfinite(w)]} — refusing to pool, since a "
                "non-finite weight silently normalises to NaN and the pool "
                "collapses to 0.0 (SF-BL-9788)"
            )
        if w.sum() <= 0:
            w = np.ones_like(w)
        w = w / w.sum()
        pooled_lo = float(np.dot(w, lo))
    return float(sigmoid(pooled_lo))


EXTREMIZE_PRODUCTION_MIN = 1.00  # engine.py Superforecaster band (SF-BL-8875)
EXTREMIZE_PRODUCTION_MAX = 1.40  # engine.py EXTREMIZE_CAP (SF-BL-8875)


def _validate_extremize_factor(factor: float) -> float:
    """Validate an extremize factor for a forward (production) pool.

    SF-BL-8875: ``extremize`` accepted any factor unchecked —
    ``extremize(0.8, factor=-1.0)`` returned 0.2 (a full inversion of the
    pooled forecast), ``factor=0.0`` returned 0.5 (the pool collapsed to
    the centre), NaN returned 0.0 and +inf returned 1.0, and
    ``aggregate()`` forwarded caller factors straight in. The sibling
    modules already guard this: calibration.py's ``log_odds_pool``
    raises on ``extremization <= 0`` (SF-BL-3596) and engine.py's
    ``Superforecaster`` caps learned factors to [1.00, 1.40]. This helper
    applies both screens, in this order: non-numeric / bool / string
    (which otherwise died with a bare TypeError, or coerced silently),
    non-finite (checked before the range screens because NaN comparisons
    are all False), ``<= 0`` (inversion / collapse), and outside the
    learned-only production band [1.00, 1.40]. Pure validation — it
    never transforms its input — so an in-band factor yields a
    bit-identical result to the pre-fix computation.

    SF-BL-11091: the type screen also covers ``bytearray`` and
    ``memoryview``. ``float()`` text-parses both, so a bytes-buffer
    factor (e.g. from a pooling config) slipped past the
    ``(str, bytes, bool)`` screen and was silently read as its parsed
    value — ``factor=bytearray(b'1.3')`` returned the 1.30 result —
    the same silent-coercion family this screen exists to block for
    ``bytes`` itself.

    SF-BL-11126: the type screen also covers a bool or a string
    carried in a NumPy array. ``float()`` erases both —
    ``factor=np.array(True)`` / ``np.array([True])`` returned the
    factor-1.0 result through ``aggregate`` (extremizing silently
    disabled) and ``factor=np.array('1.3')`` returned the parsed
    1.30 result — the SF-BL-11056 array-bool and SF-BL-11093
    string-dtype families applied to this screen. Bool-carrying
    arrays are caught via ``_is_bool_hyperparameter`` and
    string-carrying arrays (dtype kind ``U`` / ``S``, or an
    object-dtype array holding str/bytes/bytearray/memoryview)
    directly; 0-d / size-1 *numeric* arrays stay lawful and
    bit-identical (SF-BL-11057 precedent).
    """
    if isinstance(factor, (str, bytes, bytearray, memoryview, bool, np.bool_)):
        raise ValueError(
            f"extremize factor must be a finite number in "
            f"[{EXTREMIZE_PRODUCTION_MIN:.2f}, {EXTREMIZE_PRODUCTION_MAX:.2f}]; "
            f"got {factor!r} (SF-BL-8875)"
        )
    if _is_bool_hyperparameter(factor) or (
        isinstance(factor, np.ndarray)
        and (
            factor.dtype.kind in ("U", "S")
            or (
                factor.dtype == object
                and any(
                    isinstance(v, (str, bytes, bytearray, memoryview))
                    for v in factor.flat
                )
            )
        )
    ):
        raise ValueError(
            f"extremize factor must be a finite number in "
            f"[{EXTREMIZE_PRODUCTION_MIN:.2f}, {EXTREMIZE_PRODUCTION_MAX:.2f}]; "
            f"got {factor!r} (SF-BL-8875) (SF-BL-11126)"
        )
    try:
        f = float(factor)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"extremize factor must be a finite number in "
            f"[{EXTREMIZE_PRODUCTION_MIN:.2f}, {EXTREMIZE_PRODUCTION_MAX:.2f}]; "
            f"got {factor!r} (SF-BL-8875)"
        ) from exc
    if not math.isfinite(f):
        raise ValueError(
            f"extremize factor must be finite; got {factor!r} — refusing "
            "to extremize, since a non-finite factor silently collapses "
            "the pool to 0.0/1.0 (SF-BL-8875)"
        )
    if f <= 0.0:
        raise ValueError(
            f"extremize factor must be > 0; got {f!r} — factor 0 "
            "collapses the pooled forecast to the centre and a negative "
            "factor fully inverts it (SF-BL-8875)"
        )
    if not (EXTREMIZE_PRODUCTION_MIN <= f <= EXTREMIZE_PRODUCTION_MAX):
        raise ValueError(
            f"extremize factor {f!r} is outside the learned-only "
            f"production band [{EXTREMIZE_PRODUCTION_MIN:.2f}, "
            f"{EXTREMIZE_PRODUCTION_MAX:.2f}] (engine.py EXTREMIZE_CAP); "
            "refusing to ship a forward pool with an unlearned / "
            "out-of-band factor (SF-BL-8875)"
        )
    return f


def _extremize_core(p: float, factor: float, centre: float = 0.5) -> float:
    """Unchecked extremize arithmetic — RESEARCH / LEARNING USE ONLY.

    Identical arithmetic to the pre-fix ``extremize`` body. Exists so
    ``select_extremize_factor``'s burn-in grid search can *evaluate*
    candidate factors outside the production band in order to learn;
    a factor evaluated here does not ship — passing it to the public
    ``extremize`` / ``aggregate`` re-validates it against
    [1.00, 1.40] (SF-BL-8875). Never call this for a forward pool.
    """
    p = float(np.clip(p, EPS, 1 - EPS))
    c = float(np.clip(centre, EPS, 1 - EPS))
    lo = math.log(p / (1 - p))
    lo_c = math.log(c / (1 - c))
    return float(sigmoid(lo_c + factor * (lo - lo_c)))


def _validate_extremize_endpoint(name: str, value: float) -> float:
    """Validate an extremize endpoint (`p` or `centre`) for a forward pool.

    SF-BL-10191: ``extremize`` validated only its factor (SF-BL-8875).
    ``p`` and ``centre`` flowed unchecked into the clipping logit path:
    ``extremize(nan)`` returned exactly 0.0 — np.clip passes NaN through,
    the logit is NaN, and ``sigmoid``'s ``x > -700`` overflow guard is
    False for NaN, so sigmoid takes its exp-underflow branch and emits
    0.0: a corrupt pool member laundered into a maximally confident
    pooled forecast instead of an abstention. ``centre=nan`` likewise
    returned 0.0, ``centre=+inf`` returned 0.0877 (silently clipped to
    1-EPS and treated as near-certainty), and ``p=±inf`` was silently
    clipped to the EPS bounds. ``aggregate()`` does not rescue this: it
    never supplies `centre`, and the pooled value it does supply is
    finite only because ``log_odds_pool``'s sigmoid launders a NaN pool
    to 0.0 upstream (a mutated-after-construction NaN member pools to
    1.58e-08 through aggregate, not an error). extremize is a public
    boundary, so it screens its own endpoints. Pure validation — it
    never transforms a finite input — so finite-input results are
    bit-identical to the pre-fix computation.

    SF-BL-11091: the type screen also covers ``bytearray`` and
    ``memoryview`` — ``float()`` text-parses both, so
    ``centre=bytearray(b'0.5')`` / ``centre=memoryview(b'0.5')``
    returned the 0.5-centre result silently and
    ``centre=bytearray(b'1.3')`` returned 0.0877 (parsed, then clipped
    to near-certainty — the centre=True failure shape) instead of the
    named ValueError a ``bytes`` centre already raised.

    SF-BL-11092: a *finite* ``centre`` outside [0, 1] passed every
    screen above and was clipped by ``_extremize_core`` to
    [EPS, 1-EPS]: ``extremize(0.8, centre=1.5)`` and
    ``extremize(0.8, centre=82.0)`` both returned 0.0877, bit-identical
    to ``centre=1.0`` — a percent-unit or otherwise corrupt centre
    silently extremized the pooled forecast around near-certainty,
    while the lawful ``centre=0.9`` returned 0.7582. The range screen
    below therefore applies to ``centre`` only (the criterion names
    centre; ``p`` keeps its existing handling), runs after the finite
    screen so NaN / ±inf keep their SF-BL-10191 errors, and leaves the
    endpoints 0.0 / 1.0 (and -0.0) lawful with their pre-fix clipping
    behaviour. Pure validation — an in-range centre is returned
    untransformed, so in-range results are bit-identical.

    SF-BL-11091 (round 2): the type screen also covers a bool or a
    string carried in a NumPy array — the array route the round-1
    screen missed. ``float()`` erases both — ``centre=np.array(True)``
    / ``np.array([True])`` returned 0.0877 (the centre=True kill
    value, the bool read as a near-certain centre),
    ``centre=np.array('0.5')`` returned the parsed 0.5-centre result,
    and ``extremize(np.array(True))`` returned 0.9999999841510476.
    Bool-carrying arrays are caught via ``_is_bool_hyperparameter``
    and string-carrying arrays (dtype kind ``b`` / ``U`` / ``S``, or
    an object-dtype array holding str/bytes/bytearray/memoryview)
    directly — mirroring the SF-BL-11126 screen in
    ``_validate_extremize_factor`` and the in-tree SF-BL-11056/11057
    screens. 0-d / size-1 *numeric* arrays stay lawful and
    bit-identical (SF-BL-11057 precedent).
    """
    if isinstance(value, (str, bytes, bytearray, memoryview, bool, np.bool_)):
        raise ValueError(
            f"extremize {name} must be a finite number; got {value!r} "
            "(SF-BL-10191)"
        )
    if _is_bool_hyperparameter(value) or (
        isinstance(value, np.ndarray)
        and (
            value.dtype.kind in ("b", "U", "S")
            or (
                value.dtype == object
                and any(
                    isinstance(v, (str, bytes, bytearray, memoryview))
                    for v in value.flat
                )
            )
        )
    ):
        raise ValueError(
            f"extremize {name} must be a finite number; got {value!r} "
            "(SF-BL-10191) (SF-BL-11091-R2)"
        )
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"extremize {name} must be a finite number; got {value!r} "
            "(SF-BL-10191)"
        ) from exc
    if not math.isfinite(v):
        raise ValueError(
            f"extremize {name} must be finite; got {value!r} — refusing "
            "to extremize, since a non-finite endpoint is laundered by "
            "the clipping / sigmoid path into a confident 0.0/1.0 "
            "(SF-BL-10191)"
        )
    if name == "centre" and not (0.0 <= v <= 1.0):
        raise ValueError(
            f"extremize centre must be in [0, 1]; got {value!r} — "
            "refusing to extremize, since a finite centre outside "
            "[0, 1] is clipped by the extremize core to [EPS, 1-EPS] "
            "and silently extremizes the pooled forecast around "
            "near-certainty (SF-BL-11092)"
        )
    return v


def extremize(p: float, factor: float = 1.30, centre: float = 0.5) -> float:
    """
    Extremize a pooled probability away from `centre` in log-odds space.

        logit(p_ext) = logit(centre) + factor * (logit(p) - logit(centre))

    factor = 1  -> no change
    factor > 1  -> push away from centre (corrects crowd under-confidence)
    factor < 1  -> pull toward centre

    The canonical GJP finding (Satopää et al. 2014; Mellers et al. 2014)
    is that optimal factors for human crowds are typically ~1.2–2.0
    depending on shared-information overlap; for LLM-agent ensembles with
    genuinely diverse methods a more conservative 1.15–1.45 is the sane
    search range, selected on a burn-in / validation split — never assumed.

    Raises ValueError if `factor` is non-numeric, non-finite, <= 0, or
    outside the learned-only production band [1.00, 1.40] (SF-BL-8875) —
    validation runs before any extremizing arithmetic. Candidate factors
    outside that band may be *evaluated* on a burn-in split via
    ``select_extremize_factor``, but they cannot ship through here.

    Raises ValueError if `p` or `centre` is non-numeric or non-finite
    (SF-BL-10191): pre-fix, extremize(nan) returned exactly 0.0 and
    centre=+inf was silently clipped to near-certainty.

    Raises ValueError naming `factor` / `centre` if either is a bool
    or a non-numeric type — str, bytes, bytearray or memoryview
    included (SF-BL-8875 / SF-BL-10191 / SF-BL-11091): on the pre-guard
    source, factor=True returned the input unchanged (extremizing
    silently disabled), centre=True returned 0.0877, and
    factor='1.3' / centre='0.5' died with a bare TypeError /
    UFuncTypeError instead of the named ValueError.

    Raises ValueError naming `centre` if `centre` is finite but
    outside [0, 1] (SF-BL-11092): pre-fix, centre=1.5 and centre=82.0
    both returned 0.0877 for p=0.8, bit-identical to centre=1.0,
    because the centre was clipped to [EPS, 1-EPS] with no range
    screen. In-range centres (endpoints 0.0 / 1.0 included) are
    bit-identical to the pre-fix computation.
    """
    factor = _validate_extremize_factor(factor)
    p = _validate_extremize_endpoint("p", p)
    centre = _validate_extremize_endpoint("centre", centre)
    return _extremize_core(p, factor, centre)


def _is_bool_hyperparameter(value) -> bool:
    """True iff ``value`` is a bool in any form that ``float()`` erases.

    SF-BL-11056: the SF-BL-9889 / SF-BL-10274 screens reject a scalar
    ``bool`` / ``np.bool_``, but a bool carried in a NumPy array still
    flowed through ``float(...)`` as 1.0 / 0.0 — ``lam=np.array(True)``
    and ``lam=np.array([True])`` silently returned the lam=1 weights
    [0.57598979, 0.42401021] on the Briers 0.01-vs-0.81 panel (vs the
    12.0-default [0.69997291, 0.30002709]), ``shrinkage_n`` in the same
    forms returned the shrinkage_n=1 weights, and
    ``floor=np.array(False)`` silently returned the floor=0 weights.
    This screen therefore also catches a bool-dtype array of any
    shape and an object-dtype array holding a bool (recursively).
    A numeric array (float/int dtype, any shape ``float()`` accepts)
    is NOT a bool and stays lawful, bit-identical to pre-fix.
    """
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "b":
            return True
        if value.dtype == object:
            return any(_is_bool_hyperparameter(v) for v in value.flat)
        return False
    dtype = getattr(value, "dtype", None)
    if dtype is not None and getattr(dtype, "kind", None) == "b":
        return True
    return False


def _validate_track_record_lam(lam: float) -> float:
    """Validate the `lam` argument of ``track_record_weights``.

    SF-BL-9889: ``track_record_weights`` never validated ``lam`` —
    with two 20-record forecasters (mean Briers 0.01 vs 0.81) the
    base ``lam=12`` gives [0.70, 0.30] but ``lam=-12`` returned
    [0.30, 0.70], silently inverting the panel so the Brier-0.81
    forecaster took the larger weight; ``lam=NaN`` and ``lam=+inf``
    silently returned equal weights (the NaN/inf raw sum fails the
    ``> 0`` test and triggers the equal fallback, erasing the track
    record with no signal), ``lam=-inf`` returned all-NaN weights,
    and ``lam=0`` erased every Brier difference. This helper screens,
    in order: non-numeric / bool / string, non-finite (checked before
    the range screen because NaN comparisons are all False), and
    ``<= 0`` — the preregistered SF-BL-9889 domain is finite and
    strictly positive (contrast the a4 namesake SF-BL-9514, whose
    prereg admitted ``lam == 0``). Pure validation — it never
    transforms a valid input — so finite positive ``lam`` yields
    bit-identical weights to pre-fix.
    """
    if _is_bool_hyperparameter(lam):
        raise ValueError(
            f"track_record_weights lam must be finite and > 0; "
            f"got {lam!r} — a bool is not a number: True/False would "
            "coerce to lam=1/0 and silently rewrite the pool "
            "weighting (SF-BL-11056) (SF-BL-9889)"
        )
    if isinstance(lam, (str, bytes, bool, np.bool_)):
        raise ValueError(
            f"track_record_weights lam must be finite and > 0; "
            f"got {lam!r} (SF-BL-9889)"
        )
    try:
        v = float(lam)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"track_record_weights lam must be finite and > 0; "
            f"got {lam!r} (SF-BL-9889)"
        ) from exc
    if not math.isfinite(v):
        raise ValueError(
            f"track_record_weights lam must be finite and > 0; "
            f"got {lam!r} — refusing to compute pool weights, since a "
            "non-finite lam silently collapses the panel to equal "
            "weights (or all-NaN weights for -inf), erasing the track "
            "record with no signal (SF-BL-9889)"
        )
    if v <= 0.0:
        raise ValueError(
            f"track_record_weights lam must be finite and > 0; "
            f"got {lam!r} — lam=0 erases every Brier difference "
            "(equal raw weights) and a negative lam inverts the "
            "ranking, giving the worst forecaster the largest "
            "weight (SF-BL-9889)"
        )
    return v


def _validate_track_record_shrinkage_n(shrinkage_n: float) -> float:
    """Validate the `shrinkage_n` argument of ``track_record_weights``.

    SF-BL-9889: ``track_record_weights`` never validated
    ``shrinkage_n`` — ``shrinkage_n=-5`` returned [0.983, 0.017] for
    the two 20-record forecasters above (mean Briers 0.01 vs 0.81)
    because ``alpha = n / (n + shrinkage_n)`` exceeds 1, and
    ``shrinkage_n=-20`` with ``n=20`` records raised a bare
    ``ZeroDivisionError``; ``shrinkage_n=0`` silently disabled
    shrinkage, ``shrinkage_n=NaN`` returned all-NaN weights, and
    ``shrinkage_n=+inf`` silently zeroed every alpha, discarding
    the track record. This helper screens, in order: non-numeric /
    bool / string, non-finite, and ``<= 0``. Pure validation — it
    never transforms a valid input — so finite positive
    ``shrinkage_n`` yields bit-identical weights to pre-fix. (The
    same screen is independently preregistered for this module by
    SF-BL-10662; this fix satisfies that criterion too.)
    """
    if _is_bool_hyperparameter(shrinkage_n):
        raise ValueError(
            f"track_record_weights shrinkage_n must be finite and "
            f"> 0; got {shrinkage_n!r} — a bool is not a number: "
            "True/False would coerce to shrinkage_n=1/0 and silently "
            "rewrite the pool weighting (SF-BL-11056) (SF-BL-9889)"
        )
    if isinstance(shrinkage_n, (str, bytes, bool, np.bool_)):
        raise ValueError(
            f"track_record_weights shrinkage_n must be finite and "
            f"> 0; got {shrinkage_n!r} (SF-BL-9889)"
        )
    try:
        v = float(shrinkage_n)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"track_record_weights shrinkage_n must be finite and "
            f"> 0; got {shrinkage_n!r} (SF-BL-9889)"
        ) from exc
    if not math.isfinite(v):
        raise ValueError(
            f"track_record_weights shrinkage_n must be finite and "
            f"> 0; got {shrinkage_n!r} — refusing to compute pool "
            "weights, since a non-finite shrinkage_n makes every "
            "alpha NaN (or 0 for +inf, silently discarding the track "
            "record) (SF-BL-9889)"
        )
    if v <= 0.0:
        raise ValueError(
            f"track_record_weights shrinkage_n must be finite and "
            f"> 0; got {shrinkage_n!r} — shrinkage_n = -n divides by "
            "zero, -n < shrinkage_n < 0 puts alpha = n / (n + "
            "shrinkage_n) outside [0, 1] and inverts the trust "
            "ordering, and shrinkage_n = 0 silently disables "
            "shrinkage (SF-BL-9889)"
        )
    return v


def _validate_track_record_floor(floor: float, k: int) -> float:
    """Validate the `floor` argument of ``track_record_weights``.

    SF-BL-10274: ``track_record_weights`` never validated ``floor`` —
    ``floor=NaN`` and ``floor=inf`` both returned ``[nan, nan]`` for
    two 20-record forecasters (mean Briers 0.01 vs 0.81) because
    ``np.maximum(w, floor)`` propagates NaN before renormalisation,
    silently destroying the pool weights the extremized pool consumes.
    ``floor=-inf`` / negative floors were silently ignored and
    ``floor>=1`` silently collapsed the pool to equal weights. This
    helper screens, in order: non-numeric / bool / string, non-finite
    (checked before the range screen because NaN comparisons are all
    False), and outside the valid range. Pure validation — it never
    transforms a valid input — so lawful floors yield bit-identical
    weights to pre-fix.

    SF-BL-9889 tightens the SF-BL-10274 range ``[0, 1)`` to the
    preregistered ``[0, 1/K)`` for a panel of ``K = k`` records: a
    per-member floor of ``1/K`` or more is arithmetically
    unsatisfiable for a weight vector that must sum to 1 (at
    ``floor == 1/K`` every member would have to sit exactly at the
    floor, collapsing the pool to equal weights regardless of the
    track record — pre-fix ``floor=0.99`` at K=2 returned exactly
    [0.5, 0.5]). For the empty panel (``k == 0``) ``1/K`` is
    undefined, so the SF-BL-10274 ``[0, 1)`` screen applies, matching
    the a4 namesake precedent (SF-BL-9514). Distinct from SF-BL-8707
    (floor-then-renormalise arithmetic on lawful floors).
    """
    limit = 1.0 if k <= 0 else 1.0 / k
    bound = "[0, 1)" if k <= 0 else f"[0, 1/K) with K={k} (i.e. [0, {limit}))"
    if _is_bool_hyperparameter(floor):
        raise ValueError(
            f"track_record_weights floor must be finite and in "
            f"{bound}; got {floor!r} — a bool is not a number: "
            "True/False would coerce to floor=1/0 and silently "
            "rewrite the pool weighting (SF-BL-11056) "
            "(SF-BL-10274) (SF-BL-9889)"
        )
    if isinstance(floor, (str, bytes, bool, np.bool_)):
        raise ValueError(
            f"track_record_weights floor must be finite and in "
            f"{bound}; got {floor!r} (SF-BL-10274) (SF-BL-9889)"
        )
    try:
        f = float(floor)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"track_record_weights floor must be finite and in "
            f"{bound}; got {floor!r} (SF-BL-10274) (SF-BL-9889)"
        ) from exc
    if not math.isfinite(f):
        raise ValueError(
            f"track_record_weights floor must be finite and in "
            f"{bound}; got {floor!r} — refusing to compute pool "
            "weights, since a non-finite floor propagates through "
            "np.maximum and renormalisation into all-NaN weights "
            "(SF-BL-10274) (SF-BL-9889)"
        )
    if not (0.0 <= f < limit):
        raise ValueError(
            f"track_record_weights floor must be finite and in "
            f"{bound}; got {floor!r} (SF-BL-10274) (SF-BL-9889)"
        )
    return f


def _validate_track_record_records(records: Sequence[ForecasterRecord]) -> None:
    """Validate every record passed to ``track_record_weights``.

    SF-BL-10919: ``track_record_weights`` trusted the stored records
    verbatim — only the ``lam`` / ``shrinkage_n`` / ``floor``
    parameters carried guards (SF-BL-9889 / 10274 / 10662), never the
    records themselves. A record whose ``briers`` is ``[NaN]`` (a
    stored / restored record that bypassed ``ForecasterRecord.add``)
    makes ``mean_brier`` NaN, the raw ``exp(-lam * mean_brier)`` NaN,
    and the raw-sum check falls through to the equal-weight branch:
    the panel silently returned [0.5, 0.5] with no signal. A record
    with ``mean_brier`` -0.5 — impossible for a lawful record, whose
    Briers are squares in [0, 1] — took the panel's top weight
    (0.516) over a genuine 0.10 record, and a non-finite
    ``mean_log_score`` or a negative ``n`` (reachable on a
    deserialised record whose ``n`` is a stored field rather than
    ``len(briers)``) was likewise pooled without any screen — a
    negative ``n`` was even silently treated as a cold start by the
    ``r.n > 0`` alpha branch. This helper screens, per record, in
    order: the stored ``briers`` elements (finite, in [0, 1]) and
    ``log_scores`` elements (finite) when those lists are present,
    then the derived ``n`` (finite, >= 0), ``mean_brier`` (finite,
    in [0, 1]) and ``mean_log_score`` (finite). Every violation
    raises ``ValueError`` naming the offending field. Pure
    validation — it never transforms a record — so lawful panels
    (including the empty cold-start record, ``n == 0`` with the
    0.25 / log(2) ignorance priors, and the ``mean_brier``
    endpoints 0.0 / 1.0) yield bit-identical weights to pre-fix.
    """
    for i, r in enumerate(records):
        # Stored-element screens (present on a real ForecasterRecord;
        # a duck-typed / restored record may expose only the derived
        # fields, which are screened below either way).
        for attr, field, bounded in (
            ("briers", "mean_brier", True),
            ("log_scores", "mean_log_score", False),
        ):
            vals = getattr(r, attr, None)
            if vals is None:
                continue
            try:
                items = list(vals)
            except TypeError as exc:
                raise ValueError(
                    f"track_record_weights records[{i}] {field} must "
                    f"be finite"
                    + (" and in [0, 1]" if bounded else "")
                    + f"; got {attr}={vals!r} (SF-BL-10919)"
                ) from exc
            for v in items:
                if isinstance(v, (str, bytes, bool, np.bool_)):
                    raise ValueError(
                        f"track_record_weights records[{i}] {field} "
                        f"must be finite"
                        + (" and in [0, 1]" if bounded else "")
                        + f"; got {attr} element {v!r} (SF-BL-10919)"
                    )
                try:
                    fv = float(v)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"track_record_weights records[{i}] {field} "
                        f"must be finite"
                        + (" and in [0, 1]" if bounded else "")
                        + f"; got {attr} element {v!r} (SF-BL-10919)"
                    ) from exc
                if not math.isfinite(fv) or (bounded and not (0.0 <= fv <= 1.0)):
                    raise ValueError(
                        f"track_record_weights records[{i}] {field} "
                        f"must be finite"
                        + (" and in [0, 1]" if bounded else "")
                        + f"; got {attr} element {v!r} — a corrupt "
                        "stored score bypassing ForecasterRecord.add "
                        "(SF-BL-10919)"
                    )
        # Derived-field screens.
        try:
            n_raw = r.n
        except Exception as exc:
            raise ValueError(
                f"track_record_weights records[{i}] n must be "
                f"finite and >= 0; got unreadable n ({exc}) "
                "(SF-BL-10919)"
            ) from exc
        if isinstance(n_raw, (str, bytes, bool, np.bool_)):
            raise ValueError(
                f"track_record_weights records[{i}] n must be "
                f"finite and >= 0; got {n_raw!r} (SF-BL-10919)"
            )
        try:
            n_val = float(n_raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"track_record_weights records[{i}] n must be "
                f"finite and >= 0; got {n_raw!r} (SF-BL-10919)"
            ) from exc
        if not math.isfinite(n_val) or n_val < 0.0:
            raise ValueError(
                f"track_record_weights records[{i}] n must be "
                f"finite and >= 0; got {n_raw!r} — refusing to "
                "compute pool weights, since a negative n is "
                "impossible for a lawful record and was silently "
                "treated as a cold start (alpha 0) pre-fix "
                "(SF-BL-10919)"
            )
        try:
            mb_raw = r.mean_brier
        except Exception as exc:
            raise ValueError(
                f"track_record_weights records[{i}] mean_brier "
                f"must be finite and in [0, 1]; got unreadable "
                f"mean_brier ({exc}) (SF-BL-10919)"
            ) from exc
        if isinstance(mb_raw, (str, bytes, bool, np.bool_)):
            raise ValueError(
                f"track_record_weights records[{i}] mean_brier "
                f"must be finite and in [0, 1]; got {mb_raw!r} "
                "(SF-BL-10919)"
            )
        try:
            mb_val = float(mb_raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"track_record_weights records[{i}] mean_brier "
                f"must be finite and in [0, 1]; got {mb_raw!r} "
                "(SF-BL-10919)"
            ) from exc
        if not math.isfinite(mb_val):
            raise ValueError(
                f"track_record_weights records[{i}] mean_brier "
                f"must be finite and in [0, 1]; got {mb_raw!r} — "
                "refusing to compute pool weights, since a "
                "non-finite mean_brier makes the raw exp weight NaN "
                "and the raw-sum check silently collapses the "
                "panel to equal weights, erasing the track record "
                "with no signal (SF-BL-10919)"
            )
        if not (0.0 <= mb_val <= 1.0):
            raise ValueError(
                f"track_record_weights records[{i}] mean_brier "
                f"must be finite and in [0, 1]; got {mb_raw!r} — "
                "a mean Brier outside [0, 1] is impossible for a "
                "lawful record (Briers are squares in [0, 1]) and "
                "pre-fix took the panel's top weight over genuine "
                "records (SF-BL-10919)"
            )
        try:
            mls_raw = r.mean_log_score
        except Exception as exc:
            raise ValueError(
                f"track_record_weights records[{i}] "
                f"mean_log_score must be finite; got unreadable "
                f"mean_log_score ({exc}) (SF-BL-10919)"
            ) from exc
        if isinstance(mls_raw, (str, bytes, bool, np.bool_)):
            raise ValueError(
                f"track_record_weights records[{i}] "
                f"mean_log_score must be finite; got {mls_raw!r} "
                "(SF-BL-10919)"
            )
        try:
            mls_val = float(mls_raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"track_record_weights records[{i}] "
                f"mean_log_score must be finite; got {mls_raw!r} "
                "(SF-BL-10919)"
            ) from exc
        if not math.isfinite(mls_val):
            raise ValueError(
                f"track_record_weights records[{i}] "
                f"mean_log_score must be finite; got {mls_raw!r} "
                "— refusing to compute pool weights from a record "
                "whose stored scores are corrupt (SF-BL-10919)"
            )


def track_record_weights(
    records: Sequence[ForecasterRecord],
    lam: float = 12.0,
    shrinkage_n: float = 30.0,
    floor: float = 0.02,
) -> np.ndarray:
    """
    Exponential proper-score weights with shrinkage to equal weights.

        raw_i = exp(-lam * mean_brier_i)
        alpha_i = n_i / (n_i + shrinkage_n)          # trust grows with history
        w_i = alpha_i * raw_i + (1 - alpha_i) * (1/K)  then floored + renormalised

    `lam` controls how aggressively past Brier differences translate into
    weight. lam=12 means a 0.05 Brier edge multiplies weight by exp(0.6)≈1.82×
    before shrinkage — material but not winner-take-all. The floor keeps
    every persona in the pool (diversity is itself predictive; dropping the
    contrarian because of a bad quarter destroys the ensemble's value).

    Raises ValueError naming ``lam`` unless ``lam`` is finite and
    > 0, and naming ``shrinkage_n`` unless ``shrinkage_n`` is finite
    and > 0 (SF-BL-9889): pre-fix, lam=-12 inverted the panel (the
    Brier-0.81 forecaster took weight 0.70 over the Brier-0.01
    forecaster's 0.30), lam=NaN silently returned equal weights, and
    shrinkage_n=-5 returned [0.983, 0.017] via an alpha above 1
    (shrinkage_n=-n raised a bare ZeroDivisionError).

    Raises ValueError naming ``floor`` unless ``floor`` is finite and
    in [0, 1/K) for a panel of K records ([0, 1) on the empty panel,
    where 1/K is undefined) (SF-BL-10274 as tightened by SF-BL-9889):
    pre-fix, floor=NaN and floor=inf returned all-NaN weights and a
    floor at or above 1/K silently collapsed the pool toward equal
    weights. All validation runs before any weighting arithmetic
    (including the empty-records return), so no call can ship
    corrupt weights.

    Raises ValueError naming the offending hyperparameter on any
    bool ``lam``, ``shrinkage_n`` or ``floor`` (SF-BL-11056), in
    every form ``float()`` would erase: a scalar bool / ``np.bool_``
    (already screened under SF-BL-9889 / SF-BL-10274), a bool-dtype
    NumPy array of any shape, and an object-dtype array holding a
    bool. Pre-fix those array forms coerced to 1.0 / 0.0 with no
    signal — ``lam=np.array(True)`` returned the lam=1 weights and
    ``floor=np.array(False)`` the floor=0 weights. Numeric
    hyperparameters (including numeric NumPy scalars and arrays)
    are bit-identical to pre-fix.

    Raises ValueError naming ``mean_brier``, ``mean_log_score`` or
    ``n`` on any record whose stored scores are corrupt
    (SF-BL-10919): a non-finite ``mean_brier`` / ``mean_log_score``,
    a ``mean_brier`` outside [0, 1], or a negative ``n``. Pre-fix
    the records themselves were trusted verbatim (only the
    parameters above carried guards): a record with
    ``briers=[NaN]`` silently collapsed the panel to equal weights
    [0.5, 0.5], and a record with ``mean_brier`` -0.5 — impossible
    for a lawful record, reachable via restored records that
    bypass ``ForecasterRecord.add`` — took the panel's top weight
    (0.516) over a genuine 0.10 record. The record screen runs
    after the parameter screens and before any weighting
    arithmetic, so no call can ship corrupt weights.
    """
    k = len(records)
    lam = _validate_track_record_lam(lam)
    shrinkage_n = _validate_track_record_shrinkage_n(shrinkage_n)
    floor = _validate_track_record_floor(floor, k)
    _validate_track_record_records(records)
    if k == 0:
        return np.array([])
    equal = 1.0 / k
    raw = np.array([math.exp(-lam * r.mean_brier) for r in records], dtype=float)
    raw = raw / raw.sum() if raw.sum() > 0 else np.full(k, equal)
    alphas = np.array([r.n / (r.n + shrinkage_n) if r.n > 0 else 0.0 for r in records])
    w = alphas * raw + (1.0 - alphas) * equal
    w = np.maximum(w, floor)
    return w / w.sum()


def _validate_aggregate_extremize_factor(extremize_factor: float) -> float:
    """Validate ``aggregate``'s own ``extremize_factor`` argument.

    SF-BL-11126: SF-BL-8875 pre-validated this argument through
    ``_validate_extremize_factor``, whose messages name ``factor``
    — ``extremize``'s argument — so a bool, string or None
    ``extremize_factor`` on the production DESIGN.md §5 dispatch
    path raised a ValueError that never named the argument the
    caller actually passed (and on the pre-guard source the same
    values returned 0.6517 for forecasts 0.7/0.6 — bit-identical to
    factor 1.0, extremizing silently disabled — or died with bare
    TypeErrors inside ``extremize``). This wrapper delegates to the
    shared validator (identical screens, band and arithmetic, so
    numeric factors stay bit-identical) and re-raises any rejection
    as a ValueError naming ``extremize_factor``, embedding the
    shared validator's message so its SF-BL-8875 detail survives.
    """
    try:
        return _validate_extremize_factor(extremize_factor)
    except ValueError as exc:
        raise ValueError(
            f"aggregate extremize_factor is invalid: {exc} (SF-BL-11126)"
        ) from exc


def aggregate(
    forecasts: Sequence[AgentForecast],
    strategy: str = "weighted_extremized",
    weights: Optional[Sequence[float]] = None,
    extremize_factor: float = 1.30,
) -> float:
    """Dispatch helper used by the production protocol (DESIGN.md §5).

    Every strategy honours `weights` (SF-BL-8799): previously only the
    two weighted_* branches read the argument and the other four
    silently pooled equal-weight, so a caller who believed a pool was
    track-weighted could be running an equal-weight pool. Supplied
    weights are validated (length / finiteness) before dispatch, so a
    malformed weight vector raises instead of being dropped.
    weights=None keeps every strategy's original equal-weight behaviour.

    `extremize_factor` is validated before dispatch for every strategy
    (SF-BL-8875): previously the extremized branches forwarded it
    straight into the unchecked `extremize`, so factor=-1.0 inverted
    the pooled forecast and factor=0.0 collapsed it to 0.5, while the
    other strategies silently dropped it. Like the weights screen
    above, a malformed / out-of-band factor now raises before a pool
    ships instead of being dropped or forwarded unchecked.

    Raises ValueError naming `extremize_factor` on any bool, string
    or None `extremize_factor` (SF-BL-11126): the SF-BL-8875 screen
    delegated to `_validate_extremize_factor`, whose messages name
    `factor` — `extremize`'s argument, not this dispatch argument —
    so the rejection never named `extremize_factor`. Validation now
    runs through `_validate_aggregate_extremize_factor`, which
    re-raises the shared validator's rejection under this argument's
    own name; screens, band and numeric results are unchanged.

    Raises ValueError on an empty `forecasts` list for every strategy
    (SF-BL-8902): previously an all-absent question shipped a fabricated
    pooled probability — NaN for mean / median, exactly 0.0 for the
    log-odds strategies, 1.58e-08 for the extremized ones. The check
    runs after the factor screen and before weights validation /
    dispatch, so no strategy can reach a pool primitive with zero
    forecasts; the primitives guard themselves as well.

    Raises ValueError on any non-finite member probability for every
    strategy (SF-BL-10233): pre-fix a NaN member (e.g. mutated onto an
    AgentForecast after construction, bypassing its screen) pooled to
    exactly 0.0 through the log-odds strategies and to 1.58e-08 through
    the extremized ones, while mean / median returned NaN / ±inf. The
    finite screen runs at dispatch entry — after the empty check and
    before weights validation / dispatch — so every strategy
    propagates the ``log_odds_pool`` contract instead of only the
    log-odds branches; ``log_odds_pool`` also guards itself.

    Raises ValueError on any member probability outside [0, 1] for
    the 'mean' and 'median' strategies (SF-BL-10738): pre-fix a
    finite percent-unit member mutated onto an AgentForecast after
    construction passed the finite screen above and was pooled
    verbatim — ``aggregate(..., strategy='mean')`` returned 41.25
    for members [82.0, 0.5] and the median strategy silently
    absorbed such a member. At SF-BL-10738 time the log-odds
    strategies still clipped their members, so the range screen
    runs only on the mean / median dispatch path — after the
    finite screen and before weights validation / dispatch — and
    ``mean_pool`` / ``median_pool`` guard themselves identically
    for direct callers. Since SF-BL-10935 the log-odds strategies
    raise on a finite out-of-range member as well, via
    ``log_odds_pool``'s own range screen at the branch, so every
    strategy now refuses such a panel. Lawful panels in [0, 1] are
    bit-identical to the pre-fix computation for every strategy.

    Raises ValueError naming `weights` on any string weight for
    every strategy (SF-BL-11250): pre-fix ``aggregate(...,
    strategy='weighted_log_odds', weights=['0.5', '0.5'])`` returned
    0.5, bit-identical to the numeric control — the weights
    validation through ``_normalized_weights`` float-converted the
    vector with no dtype screen, and the log-odds branches' own
    inline conversion did the same. The raw string screen now runs
    inside ``_normalized_weights`` (this dispatch validation, before
    any branch) and inside ``log_odds_pool`` itself, before either
    conversion could parse a numeric string; numeric weight vectors
    are bit-identical for every strategy.

    Raises ValueError naming `weights` on any bool weight for
    every strategy (SF-BL-11023): pre-fix ``aggregate(...,
    strategy='weighted_log_odds', weights=[True, False])`` returned
    the first member's forecast, the boolean mask coerced to
    ``[1.0, 0.0]`` by the same two conversions. The raw bool screen
    now runs inside ``_normalized_weights`` (this dispatch
    validation, before any branch) and inside ``log_odds_pool``
    itself, before either conversion could coerce a bool; numeric
    weight vectors are bit-identical for every strategy.
    """
    extremize_factor = _validate_aggregate_extremize_factor(extremize_factor)
    probs = [f.probability for f in forecasts]
    if len(probs) == 0:
        raise ValueError(
            "aggregate is undefined on an empty forecast list: "
            "len(forecasts) == 0 — refusing to pool zero forecasts "
            "into a fabricated probability (NaN, 0.0 or 0.5 depending "
            "on the path); an all-absent question must abstain "
            "(SF-BL-8902) (SF-BL-10233)"
        )
    _require_finite_pool_members(probs, "aggregate")
    if strategy in ("mean", "median"):
        _require_pool_members_in_unit_interval(
            np.asarray(probs, dtype=float), "aggregate"
        )
    if weights is not None:
        _normalized_weights(len(probs), weights, "aggregate")  # validate; branches re-normalise
    if strategy == "mean":
        return mean_pool(probs, weights)
    if strategy == "median":
        return median_pool(probs, weights)
    if strategy == "log_odds":
        return log_odds_pool(probs, weights)
    if strategy == "extremized":
        return extremize(log_odds_pool(probs, weights), extremize_factor)
    if strategy == "weighted_log_odds":
        return log_odds_pool(probs, weights)
    if strategy == "weighted_extremized":
        return extremize(log_odds_pool(probs, weights), extremize_factor)
    raise ValueError(f"unknown strategy: {strategy}")


def _require_valid_select_panel(pooled_probs: np.ndarray, outcomes: np.ndarray) -> None:
    """Raise ValueError if a select_extremize_factor panel is unusable.

    SF-BL-10192: without this screen a single NaN pooled forecast
    certified a perfect score — ``select_extremize_factor([nan], [0])``
    returned ``(0.8, 0.0)``, the first grid factor with a fabricated
    Brier of 0.0. The mechanism bypasses the SF-BL-9784 finite screen
    inside ``brier_score``: ``_extremize_core`` clips NaN to NaN, its
    logit is NaN, and the scalar ``sigmoid`` guard returns exactly 0.0
    for NaN, so the value reaching ``brier_score`` is finite; ±inf
    pooled forecasts are likewise laundered by the clip to 1-EPS / EPS
    and scored as ordinary near-certain forecasts. A non-finite outcome
    previously raised only incidentally, via ``brier_score`` and under
    that function's name. This screen runs at selector entry, before
    the grid is built or scanned, and raises a named ValueError on a
    length mismatch, an empty panel, or any non-finite value in either
    array. Pure validation — it never transforms its inputs — so clean
    panels are bit-identical to the pre-fix computation.

    SF-BL-11015: the same selector also lacked any [0, 1] range screen
    on ``pooled_probs``, so a *finite* out-of-range pooled forecast was
    certified: ``_extremize_core`` clips it to 1-EPS / EPS before
    ``brier_score`` sees it, hence ``[1.5, 0.5]`` and ``[82.0, 0.5]``
    (outcomes ``[1, 0]``) both returned ``(1.4, 0.125)``. The range
    screen below runs after the finiteness screen and raises a named
    ValueError on any finite pooled value outside [0, 1]; endpoints
    0.0 / 1.0 stay lawful and outcomes are not range-screened here.

    SF-BL-11393: neither panel's TYPE was ever screened at this entry
    point, so a string panel was certified bit-identically to its
    numeric twin — ``select_extremize_factor(['0.7', '0.6'], [1, 0])``
    returned ``(0.9, 0.2248)``, exactly the numeric control. The
    mechanism bypasses the SF-BL-11058 screen inside ``brier_score``:
    each pooled value is extremized through ``float(p)`` first, so the
    array ``brier_score`` receives is already numeric, and for a
    pandas object-dtype outcome Series the downstream screen does not
    see the strings either (that form also returned ``(0.9, 0.2248)``
    pre-fix). The type screen below therefore runs FIRST, on the
    ORIGINAL inputs, before any ``np.asarray(..., dtype=float)``
    conversion or ``float(p)`` coercion erases a string type: any
    string / bytes / bytearray / memoryview element — a scalar, a
    mixed or nested list/tuple element, a string-dtype (kind ``U`` /
    ``S``) array, an object-dtype element holding one, or a pandas
    Series carrying one — and any bare non-numeric object-dtype
    element raises a ValueError naming ``pooled_probs`` or
    ``outcomes`` (precedent: SF-BL-11058's ``_reject_string_inputs``,
    extended to the carriers ``float()`` parses identically, cf. the
    SF-BL-11057 round-2 grid screen). Object-dtype panels holding
    only plain numbers stay lawful (fleet precedent SF-BL-11433),
    and bool outcomes stay lawful (SF-BL-10853/10939 carve-out) —
    pure validation, so float panels are bit-identical.
    """
    # SF-BL-11393 type screen (runs first, on the original inputs).
    def _is_text_carrier(v) -> bool:
        return isinstance(v, (str, bytes, bytearray, memoryview,
                              np.str_, np.bytes_))

    def _is_plain_number(v) -> bool:
        # Decimal is not a numbers.Number, so name it (and Fraction,
        # a Rational) explicitly; both are lawful pre-fix panel values
        # via float() and must stay lawful inside object arrays too.
        from decimal import Decimal
        from fractions import Fraction

        return isinstance(v, (bool, np.bool_, int, float, np.integer,
                              np.floating, Decimal, Fraction))

    def _panel_has_bad_type(x, in_object: bool = False) -> bool:
        if _is_text_carrier(x):
            return True
        if isinstance(x, np.ndarray):
            if x.dtype.kind in ("U", "S"):
                return True
            if x.dtype == object:
                return any(_panel_has_bad_type(v, in_object=True)
                           for v in x.flat)
            return False
        if isinstance(x, (list, tuple)):
            return any(_panel_has_bad_type(v, in_object=in_object)
                       for v in x)
        dtype = getattr(x, "dtype", None)
        if dtype is not None:
            kind = getattr(dtype, "kind", None)
            if kind in ("U", "S"):
                return True
            if kind == "O":
                try:
                    return any(_panel_has_bad_type(v, in_object=True)
                               for v in list(x))
                except TypeError:
                    return False
            return False
        if in_object and not _is_plain_number(x):
            # A bare object-dtype element that is neither a plain
            # number nor a text carrier (e.g. object(), None, a
            # nested container is handled above): not a lawful panel
            # value, and float() would die on it with a bare
            # TypeError downstream.
            return True
        return False

    for _name, _panel in (("pooled_probs", pooled_probs),
                          ("outcomes", outcomes)):
        if _panel_has_bad_type(_panel):
            _shown = " ".join(repr(_panel).split())
            if len(_shown) > 120:
                _shown = _shown[:117] + "..."
            raise ValueError(
                f"select_extremize_factor: {_name} must be real "
                f"numbers, not string; got string or object-dtype "
                f"value(s) in {_shown} — a numeric string is not a "
                "number and must not be silently parsed into a "
                "scored panel (SF-BL-11393)"
            )
    n_probs = len(np.asarray(pooled_probs).ravel())
    n_outcomes = len(np.asarray(outcomes).ravel())
    if n_probs != n_outcomes:
        raise ValueError(
            f"select_extremize_factor: len(pooled_probs) ({n_probs}) != "
            f"len(outcomes) ({n_outcomes}): pooled_probs and outcomes "
            "must have the same length (SF-BL-10192)"
        )
    if n_probs == 0:
        raise ValueError(
            "select_extremize_factor is undefined on an empty panel: "
            "len(pooled_probs) == len(outcomes) == 0 (SF-BL-10192)"
        )
    for name, arr in (("pooled_probs", pooled_probs), ("outcomes", outcomes)):
        if not np.all(np.isfinite(np.asarray(arr, dtype=float))):
            raise ValueError(
                f"select_extremize_factor: {name} must contain only "
                "finite values; got non-finite value(s) (NaN/inf) — "
                "refusing to certify a factor from a corrupt panel "
                "(SF-BL-10192)"
            )
    # SF-BL-11015: finite-but-out-of-range pooled forecasts. The finite
    # screen above cannot see them, and the selector's only downstream
    # screen is inside the scoring path — but ``_extremize_core`` clips
    # every pooled value to [EPS, 1-EPS] before ``brier_score`` ever
    # sees it, so 1.5, 82.0 (e.g. a percent-unit panel) and -0.2 are all
    # silently laundered to near-certainty / near-impossibility:
    # pre-fix ``select_extremize_factor([1.5, 0.5], [1, 0])`` and
    # ``([82.0, 0.5], [1, 0])`` both returned ``(1.4, 0.125)`` — the
    # grid-boundary factor certified from an impossible panel. This
    # screen runs after the finiteness screen (preserving its
    # precedence) and screens ``pooled_probs`` ONLY: the criterion
    # names pooled forecasts, and outcomes keep their existing
    # handling. Endpoints 0.0 / 1.0 (and -0.0 == 0.0) stay lawful.
    # Pure validation — it never transforms its inputs.
    pooled_arr = np.asarray(pooled_probs, dtype=float).ravel()
    if pooled_arr.size:
        lo, hi = float(pooled_arr.min()), float(pooled_arr.max())
        if lo < 0.0 or hi > 1.0:
            bad = pooled_arr[(pooled_arr < 0.0) | (pooled_arr > 1.0)]
            raise ValueError(
                f"select_extremize_factor: pooled_probs must lie in "
                f"[0, 1]; got finite value(s) outside that range "
                f"{bad[:5]!r} (min={lo!r}, max={hi!r}) — refusing to "
                "certify a factor from an impossible pooled forecast, "
                "since extremize clips it to 1-EPS/EPS before "
                "brier_score sees it (SF-BL-11015)"
            )


def _require_valid_select_grid(grid) -> List[float]:
    """Materialise and validate a select_extremize_factor grid.

    SF-BL-9790: without an emptiness screen an empty grid returned the
    unsearched initialiser ``(1.0, inf)`` as if a search had run — the
    loop body never executes, so ``best_f, best_b`` keep their initial
    values and the caller receives a certified-looking pair backed by
    zero evaluations. The SF-BL-8875 finite/>0 screen previously lived
    inside the scan loop, so a bad entry was rejected only after every
    earlier entry had already been scored. This helper materialises the
    grid once (lists, tuples, ndarrays, ranges and generators alike)
    and validates the WHOLE grid — non-empty, every entry finite and
    > 0 — before any candidate is scored, raising a named ValueError
    on the first violation. Pure validation: it returns the candidate
    factors as plain floats and never transforms a lawful grid, so the
    default-grid and good-grid results are bit-identical.

    SF-BL-11057: the grid also never TYPE-validated its entries — each
    entry was coerced with ``float(f)`` unchecked, so a bool entry
    became 1.0/0.0 and a numeric string became its parsed value before
    any screen could see a type, and either could be certified as the
    optimal factor: on pooled [0.7, 0.3] vs outcomes [1, 0],
    ``grid=[True]`` returned ``(1.0, 0.09)`` (the bool True certified)
    and ``grid=['2.0', '0.8']`` returned ``(2.0, 0.0241)``. The type
    screen below runs on the ORIGINAL entries, before ``float()``
    erases their type (the SF-BL-8875 factor-screen pattern): any
    bool / string / bytes entry — including one carried in a NumPy
    array (dtype kind b/U/S, cf. SF-BL-11056) — and any other entry
    ``float()`` cannot convert (previously a bare TypeError), raises
    a ValueError naming the grid. Lawful numeric entries (int, float,
    np.integer, np.floating, and other float()-convertible reals)
    are untouched.

    Round 2 (tester gaps): three carrier forms still bypassed that
    screen, because ``float()`` parses them exactly like the direct
    forms — ``[bytearray(b'2.0'), bytearray(b'0.8')]`` returned the
    TASK kill pair (2.0, 0.0241), ``[memoryview(b'1.3')]`` returned
    (1.3, 0.0622...), and a 0-d object-dtype ndarray entry wrapping a
    bool / string / bytes value unwrapped under ``float()`` (kind
    "O" fell through the "bUS" test), so ``[np.array(True,
    dtype=object)]`` returned (1.0, 0.09). The screen now runs
    through ``_is_bool_or_text_carrier``: it also rejects bytearray
    entries (the SF-BL-11058 ``_reject_string_inputs`` tuple already
    includes bytearray), any memoryview entry (a bytes-backed view
    is the text carrier ``float()`` parses; a numeric-backed view
    is not a scalar and was never convertible, so no lawful
    memoryview entry exists), and object-dtype ndarray entries
    whose element(s) recursively carry a bool / string / bytes /
    bytearray / memoryview value (the SF-BL-11058 object-array
    recursion). An object-dtype entry wrapping a plain number stays
    lawful, exactly as pre-fix.

    SF-BL-11471: the numeric-STRING grid dtype is fixtured separately
    from the bool entries above (the string-vs-bool separation of
    SF-BL-11124 vs SF-BL-11389): at the unguarded repro site,
    ``grid=['0.8', '1.3']`` was silently coerced through ``float(f)``
    and returned ``(0.8, 0.2251)`` where the default numeric grid
    returned ``(0.9, 0.2248)`` on the same pooled panel. The type
    screen above already sees the original string entries before
    ``float()`` erases their type, so that exact grid raises the
    named ValueError here; SF-BL-11471 pins it with its own fixture
    so the string dtype cannot regress independently of the bool
    fixtures.
    """
    def _is_bool_or_text_carrier(entry) -> bool:
        if isinstance(entry, (str, bytes, bytearray, bool, np.bool_,
                              np.str_, np.bytes_)):
            return True
        if isinstance(entry, memoryview):
            return True
        if isinstance(entry, np.ndarray):
            if entry.dtype.kind in "bUS":
                return True
            if entry.dtype == object:
                return any(
                    _is_bool_or_text_carrier(v) for v in entry.flat
                )
            return False
        return False

    try:
        raw_values = list(grid)
    except TypeError as exc:
        raise ValueError(
            "select_extremize_factor: grid must be an iterable of "
            f"numeric candidate factors; got non-iterable grid "
            f"{grid!r} (SF-BL-11057)"
        ) from exc
    values: List[float] = []
    for entry in raw_values:
        if _is_bool_or_text_carrier(entry):
            raise ValueError(
                "select_extremize_factor: grid entries must be "
                f"numeric candidate factors, not bool/string; got "
                f"grid entry {entry!r} of type "
                f"{type(entry).__name__} — refusing to certify a "
                "factor from a grid entry that float() would silently "
                "coerce (SF-BL-11057; numeric-string grid SF-BL-11471)"
            )
        try:
            values.append(float(entry))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "select_extremize_factor: grid entries must be "
                f"numeric candidate factors; got grid entry "
                f"{entry!r} of type {type(entry).__name__}, which is "
                "not convertible to float (SF-BL-11057)"
            ) from exc
    if len(values) == 0:
        raise ValueError(
            "select_extremize_factor is undefined on an empty grid: "
            "grid must contain at least one candidate factor — "
            "refusing to return the unsearched initialiser (1.0, inf) "
            "as if a search had run (SF-BL-9790; grid guard SF-BL-8875)"
        )
    for f in values:
        if not math.isfinite(f) or f <= 0.0:
            raise ValueError(
                f"select_extremize_factor grid values must be finite "
                f"and > 0; got {f!r} (SF-BL-9790; grid guard SF-BL-8875)"
            )
    return values


def select_extremize_factor(
    pooled_probs: np.ndarray, outcomes: np.ndarray, grid: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """Grid-search the extremizing factor minimising Brier on a validation set.

    Burn-in / validation RESEARCH only (SF-BL-8875): candidate factors
    are evaluated with the private unchecked core so the search can
    explore outside the production band [1.00, 1.40] in order to learn —
    the default grid is 0.80–2.00. Grid entries must still be finite
    and > 0 (a non-positive candidate would invert / collapse every
    candidate pool and is rejected, never scored). The returned factor
    does NOT ship by itself: `extremize` / `aggregate` re-validate it,
    so a learned factor outside [1.00, 1.40] is rejected at the
    forward-pool boundary, exactly as engine.py caps learned factors.

    Raises ValueError if `pooled_probs` and `outcomes` have different
    lengths, if the panel is empty, or if any value in either array is
    non-finite (SF-BL-10192) — validation runs before the grid scan, so
    a corrupt panel can never certify a factor with a fabricated score.
    Also raises ValueError if any finite `pooled_probs` value lies
    outside [0, 1] (SF-BL-11015): pre-fix the extremize clip laundered
    1.5 and 82.0 alike to 1-EPS and both panels certified (1.4, 0.125).
    Also raises ValueError if a supplied `grid` is empty or contains
    any non-finite or non-positive entry (SF-BL-9790) — the whole grid
    is validated before any candidate is scored, so an empty grid can
    never return the unsearched initialiser (1.0, inf).
    Also raises ValueError naming the grid if any grid entry is a bool
    or a string (SF-BL-11057): pre-fix each entry was coerced with
    float() unchecked, so grid=[True] certified (1.0, 0.09) and
    grid=['2.0', '0.8'] certified (2.0, 0.0241) on the [0.7, 0.3]/[1, 0]
    panel instead of raising. The screen also covers the carrier
    forms float() parses identically — bytearray and bytes-backed
    memoryview entries, and 0-d object-dtype ndarray entries wrapping
    a bool/string/bytes value.
    Also raises ValueError naming the grid on any numeric-string grid
    entry (SF-BL-11471, fixtured separately from the SF-BL-11057 bool
    entries): pre-fix ``grid=['0.8', '1.3']`` returned (0.8, 0.2251)
    at the unguarded repro site instead of raising, while the default
    numeric grid returned (0.9, 0.2248) on the same panel.
    Also raises ValueError naming `pooled_probs` or `outcomes` if
    either panel holds a string or a non-numeric object-dtype element
    (SF-BL-11393): pre-fix ``(['0.7', '0.6'], [1, 0])`` returned
    (0.9, 0.2248), bit-identical to the numeric control, because each
    pooled value was coerced with float(p) before scoring and the
    SF-BL-11058 screen inside `brier_score` never saw the original
    pooled types. The screen runs on the original inputs at entry;
    object-dtype panels holding only plain numbers stay lawful.
    """
    _require_valid_select_panel(pooled_probs, outcomes)
    if grid is None:
        grid = np.round(np.arange(0.8, 2.05, 0.05), 2)
    grid_values = _require_valid_select_grid(grid)
    best_f, best_b = 1.0, float("inf")
    for f in grid_values:
        ext = np.array([_extremize_core(float(p), f) for p in pooled_probs])
        b = brier_score(ext, outcomes)
        if b < best_b:
            best_b, best_f = b, float(f)
    return best_f, best_b


# ---------------------------------------------------------------------------
# Synthetic tournament
# ---------------------------------------------------------------------------

@dataclass
class SyntheticAgent:
    """
    A synthetic forecaster with known parameters.

    Generative model per question (in log-odds space):
        L          = logit(p_true)
        common     ~ N(0, rho)          shared shock (everyone read the same news)
        eps_i      ~ N(0, sigma_i)      private noise
        logit(p_i) = bias_i + beta_i * L + common + eps_i

    beta_i < 1  : under-reacts to the true signal (shrinks toward 50%)
    beta_i > 1  : over-reacts (overconfident)
    sigma_i     : idiosyncratic noise — lower is more skilled
    bias_i      : systematic directional bias in log-odds
    """
    agent_id: str
    persona: str
    beta: float
    sigma: float
    bias: float = 0.0


# Persona roster mirroring DESIGN.md §2. Parameters are illustrative but
# ordered by the qualitative claims in the design: the Bayesian updater is
# the best single forecaster; the contrarian is weak alone but decorrelated;
# the base-rate anchorer is well calibrated but dull; the news hunter is
# high-variance; the novice is present to prove weighting can suppress noise.
AGENT_ROSTER: List[SyntheticAgent] = [
    SyntheticAgent("bayesian_updater",   "Bayesian updater",      beta=0.98, sigma=0.55, bias=0.00),
    SyntheticAgent("base_rate_anchor",   "Base-rate anchorer",    beta=0.62, sigma=0.35, bias=0.00),
    SyntheticAgent("trend_model_reader", "Trend / model reader",  beta=0.90, sigma=0.75, bias=0.05),
    SyntheticAgent("news_signal_hunter", "News / signal hunter",  beta=1.05, sigma=1.05, bias=0.10),
    SyntheticAgent("contrarian",         "Contrarian / red-team", beta=0.55, sigma=0.90, bias=-0.15),
    SyntheticAgent("overconfident",      "Overconfident pundit",  beta=1.35, sigma=0.85, bias=0.15),
    SyntheticAgent("underconfident",     "Underconfident hedger", beta=0.70, sigma=0.60, bias=0.00),
    SyntheticAgent("noisy_novice",       "Noisy novice",          beta=0.80, sigma=1.60, bias=0.20),
]

COMMON_SHOCK_SD = 0.35   # sd of the shared log-odds shock (shared information)
BURN_IN = 150            # questions used to learn weights + extremize factor
EXTREMIZE_FIXED = 1.30   # a-priori factor evaluated as its own strategy


def generate_questions(rng: np.random.Generator, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw true probabilities from a realistic mixture:
      60% moderate (Beta(2.2, 2.2) — mass in the middle),
      25% skewed   (Beta(1.2, 3.5) or its mirror — likely / unlikely events),
      15% sharp    (near 0.05 or 0.95 — strong-signal questions).
    Returns (p_true, logit_true, outcomes).
    """
    u = rng.random(n)
    p = np.empty(n)
    moderate = u < 0.60
    skewed = (u >= 0.60) & (u < 0.85)
    sharp = u >= 0.85
    p[moderate] = rng.beta(2.2, 2.2, size=int(moderate.sum()))
    if skewed.sum():
        flip = rng.random(int(skewed.sum())) < 0.5
        draws = rng.beta(1.2, 3.5, size=int(skewed.sum()))
        p[skewed] = np.where(flip, draws, 1.0 - draws)
    if sharp.sum():
        p[sharp] = np.where(rng.random(int(sharp.sum())) < 0.5,
                             rng.uniform(0.02, 0.12, size=int(sharp.sum())),
                             rng.uniform(0.88, 0.98, size=int(sharp.sum())))
    p = clip_p(p)
    outcomes = (rng.random(n) < p).astype(int)
    return p, np.log(p / (1.0 - p)), outcomes


def simulate_forecasts(
    rng: np.random.Generator,
    agents: List[SyntheticAgent],
    logit_true: np.ndarray,
) -> np.ndarray:
    """Return matrix of shape (n_questions, n_agents) of forecast probabilities."""
    n = len(logit_true)
    k = len(agents)
    common = rng.normal(0.0, COMMON_SHOCK_SD, size=n)
    mat = np.empty((n, k))
    for j, a in enumerate(agents):
        eps = rng.normal(0.0, a.sigma, size=n)
        lo = a.bias + a.beta * logit_true + common + eps
        mat[:, j] = sigmoid(lo)
    return clip_p(mat)


def run_tournament(n_questions: int = 1000, seed: int = 42, verbose: bool = True) -> Dict:
    if n_questions < 300:
        raise ValueError("tournament needs >=300 questions for a meaningful burn-in split")
    rng = np.random.default_rng(seed)
    agents = AGENT_ROSTER
    k = len(agents)

    p_true, logit_true, outcomes = generate_questions(rng, n_questions)
    forecasts = simulate_forecasts(rng, agents, logit_true)

    # --- Online track-record weighting (no leakage: weights for question t
    #     use only questions < t; extremize factor is fixed a priori for the
    #     'extremized' strategies and *learned on burn-in only* for the
    #     tuned variant reported in the diagnostics). ---
    records = [ForecasterRecord(a.agent_id) for a in agents]
    n_test = n_questions - BURN_IN

    strat_names = ["mean", "median", "log_odds", "extremized",
                   "weighted_log_odds", "weighted_extremized"]
    strat_probs: Dict[str, np.ndarray] = {s: np.empty(n_test) for s in strat_names}
    weights_history = np.empty((n_test, k))

    for t in range(n_questions):
        row = forecasts[t]
        if t >= BURN_IN:
            idx = t - BURN_IN
            w = track_record_weights(records)
            weights_history[idx] = w
            strat_probs["mean"][idx] = mean_pool(row)
            strat_probs["median"][idx] = median_pool(row)
            strat_probs["log_odds"][idx] = log_odds_pool(row)
            strat_probs["extremized"][idx] = extremize(log_odds_pool(row), EXTREMIZE_FIXED)
            strat_probs["weighted_log_odds"][idx] = log_odds_pool(row, w)
            strat_probs["weighted_extremized"][idx] = extremize(log_odds_pool(row, w), EXTREMIZE_FIXED)
        # Update records AFTER forecasting question t (strictly causal).
        for j in range(k):
            records[j].add(float(row[j]), int(outcomes[t]))

    test_outcomes = outcomes[BURN_IN:]
    test_p_true = p_true[BURN_IN:]

    results: Dict[str, Dict[str, float]] = {}
    for s in strat_names:
        results[s] = {
            "brier": brier_score(strat_probs[s], test_outcomes),
            "log_score": log_score(strat_probs[s], test_outcomes),
            "ece": calibration_stats(strat_probs[s], test_outcomes)["ece"],
        }

    # Best single agent (identified on burn-in only, to avoid selection leakage)
    burn_briers = [brier_score(forecasts[:BURN_IN, j], outcomes[:BURN_IN]) for j in range(k)]
    best_idx_burnin = int(np.argmin(burn_briers))
    # Oracle best single agent on the test set (upper bound for any single agent)
    test_briers_all = [brier_score(forecasts[BURN_IN:, j], test_outcomes) for j in range(k)]
    oracle_idx = int(np.argmin(test_briers_all))

    per_agent = {
        agents[j].agent_id: {
            "persona": agents[j].persona,
            "beta": agents[j].beta, "sigma": agents[j].sigma, "bias": agents[j].bias,
            "brier_test": float(test_briers_all[j]),
            "log_score_test": log_score(forecasts[BURN_IN:, j], test_outcomes),
            "final_weight": float(weights_history[-1, j]),
            "mean_weight_test": float(weights_history[:, j].mean()),
        }
        for j in range(k)
    }

    # Oracle forecaster (knows p_true) and a coin-flip baseline for scale.
    oracle_brier = brier_score(test_p_true, test_outcomes)
    baseline_brier = brier_score(np.full(n_test, float(test_outcomes.mean())), test_outcomes)

    # Tuned extremizing factor on burn-in pooled forecasts (diagnostic):
    burn_pooled = np.array([log_odds_pool(forecasts[t]) for t in range(BURN_IN)])
    tuned_factor, tuned_burn_brier = select_extremize_factor(burn_pooled, outcomes[:BURN_IN])
    # Backtest diagnostic only (SF-BL-8875): the tuned factor is evaluated
    # with the unchecked core because a burn-in search may land outside
    # the production band [1.00, 1.40]; it is never a forward pool, and
    # shipping that factor via extremize()/aggregate() would re-validate
    # (and, if out of band, reject) it.
    tuned_test = np.array([_extremize_core(float(p), tuned_factor) for p in
                           np.array([log_odds_pool(forecasts[t]) for t in range(BURN_IN, n_questions)])])
    tuned_test_brier = brier_score(tuned_test, test_outcomes)
    tuned_test_log = log_score(tuned_test, test_outcomes)

    summary = {
        "n_questions": n_questions, "burn_in": BURN_IN, "n_test": n_test, "seed": seed,
        "n_agents": k,
        "strategies": results,
        "per_agent": per_agent,
        "best_single_burnin_id": agents[best_idx_burnin].agent_id,
        "best_single_burnin_brier_test": float(test_briers_all[best_idx_burnin]),
        "best_single_burnin_log_test": log_score(forecasts[BURN_IN:, best_idx_burnin], test_outcomes),
        "oracle_best_single_id": agents[oracle_idx].agent_id,
        "oracle_best_single_brier_test": float(test_briers_all[oracle_idx]),
        "oracle_prob_brier_test": float(oracle_brier),
        "oracle_prob_log_test": log_score(test_p_true, test_outcomes),
        "climatology_brier_test": float(baseline_brier),
        "tuned_extremize_factor_on_burnin": tuned_factor,
        "tuned_extremized_brier_test": float(tuned_test_brier),
        "tuned_extremized_log_test": float(tuned_test_log),
        "fixed_extremize_factor": EXTREMIZE_FIXED,
        "final_weights": {agents[j].agent_id: float(weights_history[-1, j]) for j in range(k)},
    }

    if verbose:
        print(f"=== Simulated tournament: {n_questions} questions "
              f"({BURN_IN} burn-in / {n_test} test), {k} agents, seed={seed} ===")
        print(f"Climatology (always base rate)  Brier {baseline_brier:.4f}")
        print(f"Oracle (knows p_true)           Brier {oracle_brier:.4f}  "
              f"Log {summary['oracle_prob_log_test']:.4f}")
        print(f"Best single (burn-in pick: {agents[best_idx_burnin].agent_id}) "
              f"Brier {test_briers_all[best_idx_burnin]:.4f}  "
              f"Log {summary['best_single_burnin_log_test']:.4f}")
        print(f"Oracle best single ({agents[oracle_idx].agent_id}) "
              f"Brier {test_briers_all[oracle_idx]:.4f}")
        print("\nPer-agent (test set):")
        for j in range(k):
            a = per_agent[agents[j].agent_id]
            print(f"  {agents[j].agent_id:20s} Brier {a['brier_test']:.4f}  "
                  f"Log {a['log_score_test']:.4f}  final w {a['final_weight']:.3f}")
        print("\nAggregation strategies (test set, lower = better):")
        for s in strat_names:
            r = results[s]
            print(f"  {s:20s} Brier {r['brier']:.4f}  Log {r['log_score']:.4f}  ECE {r['ece']:.4f}")
        print(f"  {'tuned_extremized':20s} Brier {tuned_test_brier:.4f}  "
              f"Log {tuned_test_log:.4f}  (factor {tuned_factor:.2f} from burn-in)")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Superforecaster agent-ensemble simulation")
    ap.add_argument("--n-questions", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run_tournament(n_questions=args.n_questions, seed=args.seed)
