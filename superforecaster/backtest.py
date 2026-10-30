"""
backtest.py — Walk-forward backtester for probabilistic (binary-event) forecasts.

Pure numpy / pandas / stdlib. No sklearn, no scipy.

Design principles
-----------------
1. No lookahead, enforced not just promised:
   - Splits are purged + embargoed (see `make_splits`).
   - `assert_no_lookahead` checks that every feature timestamp used for a
     prediction issued at time t is <= t (equivalently <= the target's
     realisation time minus the horizon, since realisation = t + horizon;
     SF-BL-9846 corrected the earlier "<= t - horizon" phrasing, which
     conflated issue time with realisation time), and that every timestamp
     is finite — a NaN or ±inf timestamp raises, it never passes silently.
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
import numbers
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


def _validate_test_size(test_size) -> int:
    """SF-BL-11130: strict test-size screen for make_splits and
    WalkForwardBacktester.

    The backtester constructor stored and forwarded its test_size with
    no type screen (and make_splits screened at most the value, in the
    copies that had SF-BL-9854's ``<= 0`` check at all — a check that
    itself died with a bare TypeError on a string/None size): in live
    repros on a 60-point alternating panel (initial_train=20),
    test_size=True ran a full backtest with the size read as 1
    (metrics n=39, bit-identical to test_size=1), test_size='10' died
    with a bare 'int + str' TypeError inside make_splits that never
    named test_size, test_size=10.0 died with an IndexError in the
    backtester (and returned lawful-looking float folds from
    make_splits directly), test_size=2.5 died with a misleading
    LeakageError after folds were built, and test_size=0/False stalled
    where no ``<= 0`` guard existed. This screen therefore requires an
    integer TYPE (Python int or numpy integer; bool is rejected — it
    is an int subclass and True/False are not test sizes; an integral
    float such as 10.0 is likewise rejected) with value >= 1,
    mirroring _validate_n_bins / _validate_n_variants_tried and the
    make_splits step screen (SF-BL-9845) in this module, and raises a
    ValueError naming test_size for every other value. Validation is
    pure — it returns int(test_size), identical in value for every
    lawful integer — so lawful sizes reach the unchanged split /
    backtest path bit-identically.
    """
    if isinstance(test_size, bool) or not isinstance(
            test_size, (int, np.integer)):
        raise ValueError(
            f"test_size must be a positive integer, got "
            f"test_size={test_size!r} ({type(test_size).__name__}) "
            f"(SF-BL-11130)")
    test_size = int(test_size)
    if test_size <= 0:
        raise ValueError(
            f"test_size must be a positive integer, got "
            f"test_size={test_size!r} (SF-BL-11130)")
    return test_size

def _validate_initial_train(initial_train) -> int:
    """SF-BL-11158: strict initial-train screen for make_splits and
    WalkForwardBacktester.

    make_splits never type-validated initial_train: in a live repro
    make_splits(300, initial_train=True, test_size=20) returned []
    (the bool read as a 1-sample initial train that can never satisfy
    the 10-sample minimum fold), versus 10 folds at the lawful
    initial_train=100 on the same panel; initial_train='100'/None died
    with a bare TypeError from the ``initial_train <= purge``
    comparison that never named the value as an initial_train defect,
    initial_train=100.0 returned lawful-looking float-indexed folds,
    and initial_train=2.5 / 5 / >= n all returned [] silently — an
    empty fold list downstream is indistinguishable from a completed
    backtest with no test data. This screen therefore requires an
    integer TYPE (Python int or numpy integer; bool is rejected — it
    is an int subclass and True/False are not train sizes; an integral
    float such as 100.0 is likewise rejected) with value >= 1,
    mirroring _validate_test_size (SF-BL-11130) / _validate_n_bins /
    the make_splits step screen (SF-BL-9845) in this module, and raises
    a ValueError naming initial_train for every other value.
    Validation is pure — it returns int(initial_train), identical in
    value for every lawful integer — so lawful sizes reach the
    unchanged split / backtest path bit-identically. The companion
    empty-splits guard in make_splits (same tag) covers lawful-type
    values that still form no fold (too small for the 10-sample
    minimum, or no test block remains): make_splits never returns an
    empty list without an explicit error.
    """
    if isinstance(initial_train, bool) or not isinstance(
            initial_train, (int, np.integer)):
        raise ValueError(
            f"initial_train must be a positive integer, got "
            f"initial_train={initial_train!r} "
            f"({type(initial_train).__name__}) (SF-BL-11158)")
    initial_train = int(initial_train)
    if initial_train <= 0:
        raise ValueError(
            f"initial_train must be a positive integer, got "
            f"initial_train={initial_train!r} (SF-BL-11158)")
    return initial_train


def _validate_purge(purge) -> int:
    """SF-BL-11352: strict purge screen for make_splits and
    WalkForwardBacktester.

    make_splits never type-validated purge: in live repros
    (n=300, initial_train=100, test_size=50) purge=True returned
    splits bit-identical to purge=1 (first-fold train max 98,
    against 99 at purge=0), purge=False was silently read as 0,
    purge='5'/None died with a bare ``'<=' not supported between
    instances of 'int' and 'str'`` TypeError from the
    ``initial_train <= purge`` comparison that never named purge,
    and purge=5.0 / 2.5 returned lawful-looking float-affected
    folds. Purge is the leakage guard that drops the last training
    samples, so a silently coerced value silently weakens (or
    strengthens) that guard. This screen therefore requires an
    integer TYPE (Python int or numpy integer; bool is rejected —
    it is an int subclass and True/False are not purge counts; an
    integral float such as 5.0 is likewise rejected, as is any
    non-integer float, string or None) with value >= 0, mirroring
    _validate_test_size (SF-BL-11130), _validate_initial_train
    (SF-BL-11158) and the make_splits step screen (SF-BL-9845) in
    this module, and raises a ValueError naming purge for every
    other value. Validation is pure — it returns int(purge),
    identical in value for every lawful integer — so lawful
    integer-purge split indices are bit-identical.
    """
    if isinstance(purge, bool) or not isinstance(
            purge, (int, np.integer)):
        raise ValueError(
            f"purge must be a non-negative integer, got "
            f"purge={purge!r} ({type(purge).__name__}) "
            f"(SF-BL-11352)")
    purge = int(purge)
    if purge < 0:
        raise ValueError(
            f"purge must be a non-negative integer, got "
            f"purge={purge!r} (SF-BL-11352)")
    return purge


def _validate_embargo(embargo) -> int:
    """SF-BL-11353: strict embargo screen for make_splits,
    assert_splits_valid and WalkForwardBacktester.

    SF-BL-11353 was filed (2026-10-22) against a make_splits that
    never type-validated its own embargo argument: embargo=True
    returned a first fold (train max 99, test min 101, n=300,
    initial_train=100, test_size=50) bit-identical to embargo=1,
    and embargo='5' died with a bare ``unsupported operand type(s)
    for +: 'int' and 'str'`` TypeError that never named embargo.
    SF-BL-11132 (landed 2026-10-23) subsequently added an inline
    copy of this screen to make_splits and to assert_splits_valid,
    which closed the make_splits repro itself; what remained open
    for SF-BL-11353 was (a) no single fixtured screen for
    make_splits' own embargo argument — the check existed only as
    two duplicated inline copies — and (b) WalkForwardBacktester
    still stored embargo unscreened at construction (embargo=True /
    '5' / 2.5 all constructed silently and were only caught, if at
    all, when make_splits ran). This helper is that single screen,
    now used by all three entry points. It requires an integer
    TYPE (Python int or numpy integer; bool is rejected — it is
    an int subclass and True/False are not embargo counts; an
    integral float such as 5.0 is likewise rejected, as is any
    non-integer float, string, None or container), mirroring
    _validate_purge (SF-BL-11352) / _validate_test_size
    (SF-BL-11130) / _validate_initial_train (SF-BL-11158) in this
    module, and raises a ValueError naming embargo for every
    other value. It imposes NO range restriction, exactly as
    SF-BL-11132 preregistered: validation is pure — it returns
    int(embargo), identical in value for every lawful integer,
    including 0 and negative values, which reach the unchanged
    split geometry / gap comparison and keep their pre-fix
    verdicts — so lawful integer-embargo split indices are
    bit-identical. The message keeps the SF-BL-11132 citation
    alongside SF-BL-11353 so both tasks' fixtures match it.
    """
    if isinstance(embargo, bool) or not isinstance(
            embargo, (int, np.integer)):
        raise ValueError(
            f"embargo must be an integer, got "
            f"embargo={embargo!r} ({type(embargo).__name__}) "
            f"(SF-BL-11132, SF-BL-11353)")
    return int(embargo)


def _validate_rolling_train(rolling_train) -> Optional[int]:
    """SF-BL-11385: strict rolling-train screen for make_splits and
    WalkForwardBacktester.

    make_splits never type-validated rolling_train: it was read only
    in ``rt = rolling_train or initial_train`` and arithmetic on it,
    so in live repros (n=300, initial_train=100, test_size=20,
    window='rolling') rolling_train=True collapsed the rolling
    window to 1 and the <10-train break returned zero folds with no
    error naming rolling_train (an empty list silently before the
    SF-BL-11158 empty guard; that guard's misleading initial_train
    message after it), rolling_train='100' died with a bare
    ``TypeError: unsupported operand type(s) for -: 'int' and
    'str'`` that never named rolling_train, rolling_train=100.5
    silently built 10 folds whose folds 1-9 carried float64 train
    indices (19.5, 39.5, ...), rolling_train=100.0 was silently
    accepted the same way, and rolling_train=False / 0 were
    silently read via the ``or`` fallback as "use initial_train".
    Purge (SF-BL-11352), embargo (SF-BL-11353 / SF-BL-11132),
    horizon (SF-BL-11157) and initial_train (SF-BL-11158) types are
    fixtured for this function, but rolling_train was not. This
    screen therefore allows None (the lawful sentinel meaning
    "fall back to initial_train") or an integer TYPE (Python int
    or numpy integer; bool is rejected — it is an int subclass
    and True/False are not window sizes; an integral float such
    as 100.0 is likewise rejected, as is any non-integer float,
    string or container) with value >= 1, mirroring
    _validate_test_size (SF-BL-11130), _validate_initial_train
    (SF-BL-11158), _validate_purge (SF-BL-11352) and the
    make_splits step screen (SF-BL-9845) in this module, and
    raises a ValueError naming rolling_train for every other
    value. Validation is pure — it returns None unchanged or
    int(rolling_train), identical in value for every lawful
    integer — so lawful integer-window splits are bit-identical,
    and the screen runs unconditionally (including when
    window='expanding', where a bad rolling_train was previously
    ignored silently), exactly as the horizon screen runs even
    though horizon does not enter the split geometry.
    """
    if rolling_train is None:
        return None
    if isinstance(rolling_train, bool) or not isinstance(
            rolling_train, (int, np.integer)):
        raise ValueError(
            f"rolling_train must be a positive integer or None, got "
            f"rolling_train={rolling_train!r} "
            f"({type(rolling_train).__name__}) (SF-BL-11385)")
    rolling_train = int(rolling_train)
    if rolling_train <= 0:
        raise ValueError(
            f"rolling_train must be a positive integer or None, got "
            f"rolling_train={rolling_train!r} (SF-BL-11385)")
    return rolling_train


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
    # SF-BL-11158: type-validate initial_train FIRST, before any
    # arithmetic or comparison uses it (test_start = initial_train +
    # embargo and initial_train <= purge below) — see
    # _validate_initial_train. Pure validation, bit-identical for
    # every lawful integer.
    initial_train = _validate_initial_train(initial_train)
    # SF-BL-11352: type-validate purge BEFORE any comparison or
    # arithmetic uses it (initial_train <= purge and
    # train_end - purge below) — see _validate_purge. Pure
    # validation, bit-identical for every lawful integer.
    purge = _validate_purge(purge)
    # SF-BL-11385: type-validate rolling_train BEFORE it is read
    # (`rt = rolling_train or initial_train` in the fold loop) —
    # see _validate_rolling_train. Pure validation, bit-identical
    # for None and every lawful integer.
    rolling_train = _validate_rolling_train(rolling_train)
    # SF-BL-11132 / SF-BL-11353: type-validate embargo BEFORE it
    # enters any arithmetic (test_start = initial_train + embargo
    # below) — see _validate_embargo, the single screen shared with
    # assert_splits_valid and WalkForwardBacktester. Without it,
    # embargo=True was silently read as 1 here, and embargo='5'/None
    # died on that addition with a bare TypeError that never named
    # embargo, before the validator's own screen could ever run.
    # Pure validation, bit-identical for every lawful integer.
    embargo = _validate_embargo(embargo)
    if window not in ("expanding", "rolling"):
        raise ValueError("window must be 'expanding' or 'rolling'")
    # SF-BL-11157: type-validate horizon. Previously the body only ever
    # evaluated `horizon < 1` and never used horizon in the split
    # geometry, so horizon=True and horizon=2.5 both returned the same
    # 10 folds as horizon=1 in live repros
    # (make_splits(300, initial_train=100, test_size=20)) and
    # horizon='2' died with a bare TypeError that never named horizon.
    # The declared horizon is the leakage screen's only record of the
    # label overlap a split set was built for, so it must be an
    # integer TYPE (Python int or numpy integer; bool rejected — it
    # is an int subclass and True/False are not horizons; an integral
    # float such as 2.0 is likewise rejected) with value >= 1,
    # mirroring this module's step (SF-BL-9845), test_size
    # (SF-BL-11130), embargo (SF-BL-11132) and benchmark-horizon
    # (SF-BL-11129) screens, and every other value raises a ValueError
    # naming horizon. Validation is pure — it rebinds int(horizon),
    # identical in value for every lawful integer — so horizon=1 (and
    # every lawful integer horizon) splits are bit-identical.
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise ValueError(
            f"horizon must be a positive integer, got "
            f"horizon={horizon!r} ({type(horizon).__name__}) "
            f"(SF-BL-11157)")
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError(
            f"horizon must be a positive integer, got "
            f"horizon={horizon!r} (SF-BL-11157)")
    # SF-BL-11130: type-validate test_size BEFORE step defaulting and
    # before any fold is built — step defaults to test_size, so a bad
    # size must be screened first (see _validate_test_size). This also
    # supplies the SF-BL-9854 positivity guard this copy lacked
    # (test_size=0 stalled the fold loop via step=0).
    test_size = _validate_test_size(test_size)
    # SF-BL-9845: validate step BEFORE defaulting and before any fold is
    # built. Previously there was no check at all and `step = step or
    # test_size` compounded it: make_splits(500, step=-5) walked
    # test_start backwards and returned 39 folds whose test_idx held
    # 1,950 draws over only 240 unique points (1,710 duplicated), step=0
    # was silently coerced to test_size, a non-integer step made
    # test_start fractional / float-indexed (5.0), died with a bare
    # TypeError ("50"), was silently treated as 1 (True), or silently
    # returned a single fold (NaN / +inf). Step must be an integer TYPE
    # (Python int or numpy integer; bool rejected — it is an int
    # subclass and True/False are not steps; an integral float such as
    # 5.0 is likewise rejected), mirroring _validate_n_bins /
    # _validate_n_variants_tried in this module, and must be > 0. Validation is pure (raises only, never coerces a lawful
    # value), so every lawful integer step is bit-identical.
    if step is not None:
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)):
            raise ValueError(
                f"step must be a positive integer, got step={step!r} "
                f"({type(step).__name__}) (SF-BL-9845)")
        if step <= 0:
            raise ValueError(
                f"step must be > 0, got step={step!r} (SF-BL-9845)")
    step = test_size if step is None else step
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
    # SF-BL-11158: never return an empty list silently — an empty
    # fold list downstream is indistinguishable from a completed
    # backtest with no test data. A lawful-type initial_train can
    # still form no fold (below the 10-sample minimum train after
    # purge, or no test block remaining in n), so that case raises
    # an explicit ValueError naming initial_train here.
    if not splits:
        raise ValueError(
            f"make_splits: no folds could be formed — "
            f"initial_train={initial_train!r} with n={n!r}, "
            f"purge={purge!r}, embargo={embargo!r}, "
            f"test_size={test_size!r}, window={window!r}, "
            f"rolling_train={rolling_train!r} leaves no train block of "
            f">= 10 samples before a test block (SF-BL-11158, SF-BL-11385)")
    assert_splits_valid(splits, embargo=embargo)  # purge enforced above, by construction (SF-BL-9847)
    return splits


def assert_splits_valid(splits: Sequence[Split], *, embargo: int = 0) -> None:
    """Validate split indices: non-empty, strictly increasing within each
    block, train strictly before test, embargo gap, and (where the
    cross-fold checks are present) forward non-overlapping test blocks.

    SF-BL-9847: the former `purge` parameter is REMOVED. Purge is a
    construction property of `make_splits` — it drops the last `purge`
    would-be training points before any Split exists — and it is
    unobservable from split indices alone: a train block ending at k is
    consistent with every declared purge value, so the parameter could
    never be checked here and accepting it issued a false leakage
    certificate to any caller-constructed Split (live repro: a
    contiguous train 0..99 / test 100..149 split passed with purge=50
    declared). A fleet census (2026-10-23, work/SF-BL-9847/census.txt)
    found no Split construction outside `make_splits` in production
    code and no caller of this validator other than `make_splits`
    itself, so per the task's pre-registered kill branch the dead
    parameter is deleted instead of gaining a provenance mechanism
    nobody consumes. Callers still passing purge= now fail loudly
    (TypeError) instead of silently. `make_splits` keeps its own
    `purge` parameter, genuinely enforced by construction.
    SF-BL-9847 also adds the within-block integrity check below:
    train_idx and test_idx must be strictly increasing — duplicates
    (e.g. [0,0,1,2,2,5]) and non-monotone blocks previously passed
    silently. make_splits emits np.arange blocks, so its fixtures are
    unchanged.
    SF-BL-11132: `embargo` is type-validated FIRST, before any fold is
    examined. Previously there was no check at all: bool participated
    in the gap comparison as its int value (True read as a 1-sample
    embargo — a zero-gap fold raised 'gap 0 < embargo True' — and
    False silently passed as 0), a float such as 2.0/2.5 was silently
    compared as a fractional embargo, and a string or None embargo
    died on `gap < embargo` with a bare TypeError that never named
    embargo. Embargo must now be an integer TYPE (Python int or numpy
    integer; bool rejected — it is an int subclass and True/False are
    not embargo counts; an integral float such as 2.0 is likewise
    rejected), mirroring make_splits' step screen (SF-BL-9845) and
    _validate_n_bins in this module; any other type raises ValueError
    naming embargo. Validation is pure — it never coerces a lawful
    value (it rebinds int(embargo), identical in value for every
    lawful integer) and imposes no range restriction, so every lawful
    integer embargo, including 0 and negative values, reaches the
    unchanged gap comparison and its pass / LeakageError verdicts are
    bit-identical.
    """
    # SF-BL-11132 / SF-BL-11353: embargo type screen — runs first,
    # even on an empty splits sequence, so a misconfigured embargo
    # never slips through simply because there were no folds to
    # compare it against. See _validate_embargo (the single screen
    # shared with make_splits and WalkForwardBacktester).
    embargo = _validate_embargo(embargo)
    for s in splits:
        if len(s.train_idx) == 0 or len(s.test_idx) == 0:
            raise LeakageError(f"fold {s.fold}: empty train or test")
        # SF-BL-9847: within-block integrity — indices must be strictly
        # increasing (no duplicates, no non-monotone order) in both the
        # train and the test block.
        for _name, _idx in (("train", s.train_idx), ("test", s.test_idx)):
            _arr = np.asarray(_idx)
            if _arr.size > 1:
                _d = np.diff(_arr)
                _bad = np.nonzero(_d <= 0)[0]
                if _bad.size:
                    _i = int(_bad[0])
                    raise LeakageError(
                        f"fold {s.fold}: {_name} indices not strictly "
                        f"increasing at position {_i + 1} — "
                        f"{_arr[_i]!r} followed by {_arr[_i + 1]!r} "
                        f"(duplicate or non-monotone index, SF-BL-9847)")
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

    SF-BL-9846: every timestamp must be FINITE. Previously the check was
    only `ft > pt`, and every IEEE-754 comparison with NaN is False, so a
    NaN feature time and a NaN prediction time both passed silently — the
    guard certified corrupt timestamps as causal (+inf prediction times
    and -inf feature times failed open the same way). Any non-finite
    (NaN / +inf / -inf) prediction or feature timestamp now raises
    LeakageError. No caller in the fleet encodes a missing timestamp as
    NaN (the only internal caller passes integer split indices; audited
    2026-10-21), so there is no mask protocol — a non-finite timestamp is
    corruption, not data. The validation is pure: it never masks, drops
    or transforms a value, so all-finite behaviour is unchanged.

    `horizon` is retained for backward compatibility (existing callers
    pass it) but is REMOVED from the comparison contract: it does not —
    and must not — shift the threshold. prediction_times is already the
    issue time, so the causal bound is ft <= pt at every horizon;
    enforcing `ft <= pt - horizon` would double-count the horizon and
    reject legitimate predictions whose newest feature is the issue-time
    observation itself (e.g. WalkForwardBacktester.run's issue-time
    sanity call with identical arrays). If supplied, horizon must be a
    finite number >= 1 (as in `make_splits`) or ValueError is raised;
    it is otherwise context for the error message only.

    SF-BL-11245: `horizon` is also TYPE-validated, before the
    SF-BL-9846 finite/>= 1 value screen below. That screen alone never
    type-validated: `math.isfinite(True)` is True and `True >= 1`, so
    `horizon=True` returned None — a config flag certified as a lawful
    horizon in the fleet's leakage screen — and `horizon='1'` died
    inside `math.isfinite` with a bare `TypeError` ("must be real
    number, not str") that never names horizon. Horizon must now be a
    real-number TYPE (numbers.Real — Python int / float, every numpy
    integer / floating scalar, Fraction; bool and np.bool_ are
    rejected explicitly — bool is an int subclass and True/False are
    not horizons) and every other type — str, None, containers,
    complex, Decimal (not a numbers.Real) — raises a ValueError
    naming horizon, mirroring this module's p_raw type screen
    (SF-BL-11094). Validation is pure — it never coerces, parses or
    transforms a value — so every int/float horizon reaches the
    unchanged SF-BL-9846 value screen and its pass / LeakageError /
    ValueError verdicts are bit-identical.
    """
    # SF-BL-11245: type screen FIRST, before math.isfinite can either
    # pass a bool silently or die with a bare TypeError on a
    # non-numeric horizon (see docstring above).
    if isinstance(horizon, (bool, np.bool_)) or not isinstance(
            horizon, numbers.Real):
        raise ValueError(
            f"horizon must be a finite number >= 1, got "
            f"horizon={horizon!r} ({type(horizon).__name__}) "
            f"(SF-BL-11245)")
    if not math.isfinite(horizon) or horizon < 1:
        raise ValueError(
            f"horizon must be a finite number >= 1, got {horizon!r} "
            f"(SF-BL-9846)")
    pt = np.asarray(prediction_times)
    ft = np.asarray(feature_max_times)
    if pt.shape != ft.shape:
        raise ValueError("prediction_times and feature_max_times must have same shape")
    # SF-BL-9846: non-finite timestamps are leakage-guard failures, never
    # silent passes. Checked before the comparison, prediction array
    # first. Finiteness is tested WITHOUT coercing dtype (int timestamps
    # keep their exact legacy values/messages below); datetime64 NaT —
    # whose comparisons also fail open — is screened via isnat.
    # Round 2 (verifier gaps): (1) object-dtype arrays are screened
    # elementwise — round 1 skipped them (`else: continue`), so an
    # object array holding NaN/±inf fell through to `ft > pt` and passed
    # silently, the exact fail-open class this task closes. Each object
    # element that is a real/complex/Decimal-like number is tested for
    # finiteness (datetime64/timedelta64 elements via isnat); genuinely
    # non-numeric elements (str/bytes/None/datetimes) are skipped, so
    # object arrays holding junk keep the legacy comparison path, while
    # object arrays holding NaN/±inf raise. Screening never coerces the
    # stored arrays, so comparison values/messages are unchanged.
    # (2) The reported value is read via a flattened view
    # (`_arr.ravel()[i]`, i = flat argmax index): round 1 indexed
    # `_arr[i]` on axis 0 only, so 0-d scalar and multi-dimensional
    # non-finite inputs raised IndexError instead of LeakageError.
    # Only the non-finite path changes; finite behaviour — including
    # the pristine IndexError quirk for finite multi-dim lookahead —
    # is untouched.
    for _label, _arr in (("prediction", pt), ("feature", ft)):
        if _arr.dtype.kind in "biufc":
            _bad_flat = (~np.isfinite(_arr)).ravel()
        elif _arr.dtype.kind in "Mm":
            _bad_flat = np.isnat(_arr).ravel()
        elif _arr.dtype.kind == "O":
            _bad_flat = np.zeros(_arr.size, dtype=bool)
            for _j, _v in enumerate(_arr.ravel()):
                if _v is None or isinstance(_v, (str, bytes)):
                    continue
                if isinstance(_v, (np.datetime64, np.timedelta64)):
                    if np.isnat(_v):
                        _bad_flat[_j] = True
                    continue
                try:
                    _is_finite = bool(np.isfinite(_v))
                except (TypeError, ValueError):
                    try:
                        _is_finite = math.isfinite(float(_v))
                    except (TypeError, ValueError, OverflowError):
                        continue
                if not _is_finite:
                    _bad_flat[_j] = True
        else:
            continue  # non-numeric dtypes keep the legacy comparison path
        if np.any(_bad_flat):
            i = int(np.argmax(_bad_flat))
            raise LeakageError(
                f"non-finite {_label} timestamp at position {i}: {_arr.ravel()[i]} "
                f"(SF-BL-9846: NaN/inf timestamps cannot be certified causal)")
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

    SF-BL-9884: the equality smoke alarm below runs at the *declared*
    `shift` — the feature is compared against its own lead-`shift`, the
    construction lag the caller declared. Previously the body ignored
    `shift` entirely and always checked the 1-step lead, so a caller
    declaring a longer lag got a check for a different contract than
    the one declared (a period-5 series identical to its own lead-5
    passed with shift=5). `shift` must be a positive integer (>= 1);
    anything else is rejected with ValueError.
    """
    # SF-BL-9884: validate the declared shift before any array work,
    # mirroring this module's integer validators (make_splits step,
    # _validate_n_bins): integer TYPE required (Python int or numpy
    # integer; bool rejected — it is an int subclass and True/False are
    # not lags; integral floats such as 5.0 likewise rejected) and
    # >= 1. Validation is pure (raises only, never coerces).
    if isinstance(shift, bool) or not isinstance(shift, (int, np.integer)):
        raise ValueError(
            f"{name}: shift must be a positive integer (>= 1), got {shift!r} (SF-BL-9884)")
    if shift < 1:
        raise ValueError(
            f"{name}: shift must be a positive integer (>= 1), got {shift!r} (SF-BL-9884)")
    f = np.asarray(feature, dtype=float)
    if f.ndim != 1:
        raise ValueError(f"{name}: expected 1-D")
    # A centred window leaks the future symmetrically; a causal feature that
    # is perfectly correlated (==) with its own lead-`shift` is a red flag.
    # With shift >= len(f) there are no overlapping pairs and the check is
    # vacuous. For shift == 1 the len guard differs from the old len > 2
    # only where <= 10 finite pairs exist, which can never fire the
    # m.sum() > 10 test — so default behaviour is unchanged.
    if len(f) > shift:
        lead = f[shift:]
        cur = f[:-shift]
        m = np.isfinite(lead) & np.isfinite(cur)
        if m.sum() > 10 and np.allclose(lead[m], cur[m]):
            raise LeakageError(f"{name}: feature equals its own lead-{shift} everywhere (suspicious)")


def assert_not_identical_to(feature: np.ndarray, target: np.ndarray, *, name: str = "feature") -> None:
    """Catch the crude leak: feature IS the (contemporaneous) target.

    SF-BL-9885: a shape mismatch raises ValueError. Previously the whole
    comparison was guarded by `if f.shape == t.shape`, so any length
    mismatch — the common case when a target column is sliced
    differently from its feature — disabled this leak screen entirely
    with no warning: feature [1, 2, 3] against target [1, 2] returned
    silently even though the overlapping indices are identical.
    Incomparable inputs are a failed check, not a passed one, mirroring
    this module's other shape screens (`assert_no_lookahead` and
    `_check` both raise ValueError on shape mismatch). Validation is
    pure — it raises only, never truncates, pads, broadcasts or slices
    to the overlap — so every equal-shape input reaches the unchanged
    finite-mask + equality path below and its outcome is bit-identical.
    A caller census (2026-10-23, work/SF-BL-9885/census.txt) found no
    call site that asserts equal shapes before calling (the only calls
    pass the same object twice in tests), so the task's kill branch
    does not apply.
    """
    f = np.asarray(feature, dtype=float)
    t = np.asarray(target, dtype=float)
    if f.shape != t.shape:
        raise ValueError(
            f"{name}: feature and target shapes differ — feature shape "
            f"{f.shape} vs target shape {t.shape}; incomparable inputs "
            f"cannot pass the identical-to-target leak check (SF-BL-9885)")
    m = np.isfinite(f) & np.isfinite(t)
    if m.any() and np.array_equal(f[m], t[m]):
        raise LeakageError(f"{name}: feature is identical to contemporaneous target")


# --------------------------------------------------------------------------
# Scoring rules & diagnostics
# --------------------------------------------------------------------------
def _has_string_value(raw) -> bool:
    """SF-BL-11059: True iff a RAW panel input is string-typed.

    Recurses through lists / tuples and object ndarrays, and screens
    pandas Series / Index the same way. A plain numeric ndarray or
    Series cannot contain strings by construction, so it returns
    False without scanning elementwise; a string-DTYPE container
    (ndarray kind ``U``/``S``, pandas StringDtype) counts even when
    empty — its dtype is the defect, not any single element.

    Round 2 (G1): pandas categorical containers are screened through
    their categories. A ``pd.Categorical``, or a Series / Index with
    a ``CategoricalDtype`` (``dtype='category'``, CategoricalIndex),
    whose categories contain a str/bytes value is string-typed —
    the categories define the panel's dtype, so this counts even
    when the panel is empty — and so is one whose values contain a
    str/bytes. A categorical with numeric categories is NOT
    string-typed and stays lawful.
    """
    if isinstance(raw, (str, bytes)):
        return True
    if isinstance(raw, pd.Categorical):
        return _has_string_value(raw.categories) or any(
            _has_string_value(v) for v in raw)
    if isinstance(raw, np.ndarray):
        if raw.dtype.kind in "US":
            return True
        if raw.dtype == object:
            return any(_has_string_value(v) for v in raw.ravel())
        return False
    if isinstance(raw, (list, tuple)):
        return any(_has_string_value(v) for v in raw)
    if isinstance(raw, (pd.Series, pd.Index)):
        if isinstance(raw.dtype, pd.CategoricalDtype):
            cats = raw.dtype.categories
            if cats is not None and _has_string_value(cats):
                return True
            return any(_has_string_value(v) for v in raw)
        if raw.dtype.kind in "US" or isinstance(raw.dtype, pd.StringDtype):
            return True
        if raw.dtype == object:
            return any(_has_string_value(v) for v in raw)
        return False
    return False


def _reject_string_panels(y, p, scorer: str,
                          citations: str = "(SF-BL-11059, SF-BL-11467)") -> None:
    """SF-BL-11059: raise a named ValueError if the RAW outcomes `y`
    or forecasts `p` are string-typed.

    This screen MUST run on the raw inputs, before any
    ``np.asarray(..., dtype=float)`` conversion (in `_check` or in a
    per-scorer panel validator): float conversion silently parses
    numeric strings, so after conversion a text export that leaked
    into a forecast/outcome field is indistinguishable from a lawful
    number and was graded as one — brier_score([1], ['0.5']) returned
    0.25 and evaluate_forecasts([1, 0], ['0.9', '0.1']) returned a
    full scorecard with Brier 0.01, log score 0.1054 and AUC 1.0 in
    live repros. A numeric string is not a number — in this module it
    is the signature of an unparsed text column reaching the scorers.
    Pure validation — it never transforms `y`/`p` — so lawful numeric
    panels reach the unchanged scoring path bit-identically.

    SF-BL-11467: the named entry points covered by this screen
    now also include expected_calibration_error and
    calibration_table, which previously reached this screen
    only through the `_check` backstop (so a string panel was
    refused under `_check`'s name, never the scorer's own).
    Both now call this screen first, under their own names,
    exactly as brier_score already did; the raised messages
    therefore carry the dual citation (SF-BL-11059,
    SF-BL-11467).

    SF-BL-11391: the named entry points now also include
    auc_mann_whitney, which previously reached this screen only
    through the `_check` backstop (so a string panel was refused
    under `_check`'s name, never the AUC diagnostic's own). It
    now calls this screen first, under its own name, exactly as
    the other scorers do, passing the triple citation
    (SF-BL-11059, SF-BL-11467, SF-BL-11391) via `citations`;
    every other scorer keeps the default citation, so its
    messages are byte-identical.

    SF-BL-11392: the named entry points now also include
    log_score, which previously reached this screen only
    through the `_check` backstop (so a string panel was
    refused under `_check`'s name, never log_score's own —
    and non-numeric strings died unnamed inside
    _validate_log_score_panel's float coercion before the
    backstop was even reached). It now calls this screen
    first, under its own name, passing the citation
    (SF-BL-11059, SF-BL-11392) via `citations`.
    """
    if _has_string_value(y):
        raise ValueError(
            f"{scorer}: outcomes y must not contain string values; "
            f"got string value(s) in y (a numeric string is not a "
            f"valid outcome) {citations}")
    if _has_string_value(p):
        raise ValueError(
            f"{scorer}: forecasts p must not contain string values; "
            f"got string value(s) in p (a numeric string is not a "
            f"valid probability) {citations}")


def _check(y, p):
    """Shared panel screen for the scorers in this module.

    SF-BL-9853: outcomes must be binary. Previously _check masked
    non-finite pairs and clipped forecasts but never validated y, so
    brier_score([2, 0], [0.5, 0.5]) returned 1.25 (a binary Brier is
    bounded by 1) and auc_mann_whitney([2, 0, 1], [0.9, 0.1, 0.8])
    returned 1.0 because the y=2 row belongs to neither the y==1 nor
    the y==0 class and vanished from the U statistic — one corrupt
    outcome column corrupted brier_score, log_score,
    expected_calibration_error and auc_mann_whitney (and, through
    them, calibration_table, evaluate_forecasts and
    block_bootstrap_brier_diff) at once. _check therefore raises a
    named ValueError on any FINITE outcome outside {0, 1}. The screen
    is finite-only, exactly as preregistered: non-finite outcomes
    (NaN / +/-inf in y) are not screened and keep the existing mask
    behaviour below, as do rows whose forecast is non-finite when the
    outcome itself is binary or non-finite. Validation is pure — it
    never masks, drops, clips or transforms a value — so a lawful
    binary panel reaches the unchanged mask + clip path and every
    scorer's result is bit-identical. No caller in the fleet
    intentionally feeds fractional soft labels (audited 2026-10-23,
    SF-BL-9853 caller census); a soft-label need must add an explicit
    soft-label scorer, not feed this one.
    """
    # SF-BL-11059: screen the RAW panels for string dtype before the
    # float conversion below can parse numeric strings into lawful
    # numbers (see _reject_string_panels). The named entry points
    # (brier_score / evaluate_forecasts) screen first with their own
    # names; this is the shared-funnel backstop for every other
    # scorer that reaches _check.
    _reject_string_panels(y, p, "_check")
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    if y.shape != p.shape:
        raise ValueError("y and p shapes differ")
    _finite_y = y[np.isfinite(y)]
    _bad_y = _finite_y[(_finite_y != 0.0) & (_finite_y != 1.0)]
    if _bad_y.size:
        raise ValueError(
            f"outcomes y must be binary (0/1); got {_bad_y.size} "
            f"finite non-binary value(s) (min={float(_bad_y.min())!r}, "
            f"max={float(_bad_y.max())!r}) (SF-BL-9853)")
    m = np.isfinite(y) & np.isfinite(p)
    return y[m], np.clip(p[m], EPS, 1 - EPS)


def _has_bool_probability(raw) -> bool:
    """SF-BL-10983: True iff a RAW forecast input contains any boolean.

    Recurses through lists / tuples and object / bool-dtype ndarrays,
    and screens pandas Series / Index the same way. A plain numeric
    ndarray cannot contain bools by construction, so it returns False
    without scanning elementwise.

    SF-BL-11391: pandas categorical containers are screened through
    their categories, mirroring _has_string_value's SF-BL-11059
    round-2 handling — a ``pd.Categorical``, or a Series / Index
    with a ``CategoricalDtype``, whose categories contain a bool
    is bool-typed (the categories define the panel's dtype, so
    this counts even when the panel is empty), and so is one whose
    values contain a bool. A categorical with numeric categories
    is NOT bool-typed and stays lawful. Before this extension a
    bool categorical panel reached the scorers unscreened:
    auc_mann_whitney([1, 0], pd.Categorical([True, False]))
    returned 1.0 in a live repro.
    """
    if isinstance(raw, (bool, np.bool_)):
        return True
    if isinstance(raw, pd.Categorical):
        return _has_bool_probability(raw.categories) or any(
            _has_bool_probability(v) for v in raw)
    if isinstance(raw, np.ndarray):
        if raw.dtype == bool:
            return True
        if raw.dtype == object:
            return any(_has_bool_probability(v) for v in raw.ravel())
        return False
    if isinstance(raw, (list, tuple)):
        return any(_has_bool_probability(v) for v in raw)
    if isinstance(raw, (pd.Series, pd.Index)):
        if isinstance(raw.dtype, pd.CategoricalDtype):
            cats = raw.dtype.categories
            if cats is not None and _has_bool_probability(cats):
                return True
            return any(_has_bool_probability(v) for v in raw)
        if raw.dtype == bool:
            return True
        if raw.dtype == object:
            return any(_has_bool_probability(v) for v in raw)
        return False
    return False


def _reject_bool_probabilities(p, scorer: str,
                               citations: str = "(SF-BL-10983)") -> None:
    """SF-BL-10983: raise a named ValueError if the RAW forecasts `p`
    contain any boolean value.

    This screen MUST run on the raw input, before any
    ``np.asarray(..., dtype=float)`` conversion (in `_check` or in a
    per-scorer panel validator): float conversion silently maps
    True -> 1.0 and False -> 0.0, so after conversion a boolean flag
    that leaked into a forecast field is indistinguishable from a
    lawful endpoint and was graded as a perfect forecast —
    brier_score([1], [True]) returned 9.98e-31, log_score([1], [True])
    returned 9.99e-16 and expected_calibration_error([1], [True])
    returned 9.99e-16 in live repros. A bool is not a probability —
    in Python it is an int subclass, which is exactly why every fleet
    bool screen rejects it explicitly (mirrors calibration.py's
    SF-BL-10938 screen for its separate implementation). Pure
    validation — it never transforms `p` — so lawful 0/1 panels (int
    or float) reach the unchanged scoring path bit-identically.

    SF-BL-11391: auc_mann_whitney's coverage under this screen is
    re-fixtured under that task (its bool-outcome companion is
    _reject_bool_outcomes below); it passes the dual citation
    (SF-BL-10983, SF-BL-11391) via `citations`, while every other
    scorer keeps the default citation, so its messages are
    byte-identical.
    """
    if _has_bool_probability(p):
        raise ValueError(
            f"{scorer}: forecasts p must not contain boolean values; "
            f"got boolean value(s) in p (a bool is not a valid "
            f"probability) {citations}")


def _reject_bool_outcomes(y, scorer: str) -> None:
    """SF-BL-11391: raise a named ValueError if the RAW outcomes
    `y` contain any boolean value — auc_mann_whitney ONLY.

    This screen MUST run on the raw input, before any
    ``np.asarray(..., dtype=float)`` conversion in `_check`: float
    conversion silently maps True -> 1.0 and False -> 0.0, so after
    conversion a boolean flag that leaked into an outcome field is
    indistinguishable from a lawful binary outcome and was ranked
    as one — auc_mann_whitney([True, False], [0.9, 0.1]) returned
    a perfect 1.0 in a live repro. SF-BL-10983 / SF-BL-11017
    screened bool FORECASTS only, so the AUC diagnostic's outcome
    panel was never type-validated; this task's prereg covers any
    string or bool element in either panel. A bool is not an
    outcome here — in Python it is an int subclass, which is
    exactly why every fleet bool screen rejects it explicitly.
    Pure validation — it never transforms `y` — so lawful 0/1
    panels (int or float) reach the unchanged scoring path
    bit-identically. This helper is deliberately auc-only: the
    other scorers keep their preregistered bool-outcome legacy
    behaviour (e.g. brier_score([True], [0.5]) == 0.25).
    """
    if _has_bool_probability(y):
        raise ValueError(
            f"{scorer}: outcomes y must not contain boolean values; "
            f"got boolean value(s) in y (a bool is not a valid "
            f"outcome) (SF-BL-10983, SF-BL-11391)")


def _reject_nonfinite_rows(y, p, scorer: str) -> None:
    """SF-BL-12282: strict non-finite screen for brier_score,
    expected_calibration_error and calibration_table ONLY.

    _check masks non-finite pairs and clips before those three
    scorers run, so direct calls previously returned survivor-only
    scores with no dropped-row signal: in live repros
    expected_calibration_error([0, 1], [NaN, 0.9]) returned 0.1,
    bit-identical to the surviving row [1] / [0.9] alone,
    brier_score returned the survivor's 0.01, and
    calibration_table's n summed to 1 of 2 rows. SF-BL-10071 fixed
    only log_score (its own _validate_log_score_panel) and
    SF-BL-9295 covers artefact-level n disclosure, neither of
    which screens these three direct scorers. This screen
    therefore validates the RAW inputs before _check runs and
    raises a named ValueError carrying the dropped-row count on
    any row that is non-finite in y or p — exactly the rows
    _check would drop, since a bare float / table has no channel
    to surface the count. It MUST run after the string screen
    (and the bool screen, where present): float conversion
    silently parses numeric strings and maps bools to endpoints,
    so those defects must be named by their own screens first.
    Validation is pure — it never masks, drops, clips or
    transforms a value — so a clean (all-finite) panel reaches
    the unchanged _check + formula path and its result is
    bit-identical; an empty panel (0 dropped rows) keeps its
    pre-fix behaviour.
    """
    y_arr = np.asarray(y, dtype=float); p_arr = np.asarray(p, dtype=float)
    if y_arr.shape != p_arr.shape:
        raise ValueError("y and p shapes differ")
    n_dropped = int(np.sum(~(np.isfinite(y_arr) & np.isfinite(p_arr))))
    if n_dropped:
        raise ValueError(
            f"{scorer}: {n_dropped} of {y_arr.size} row(s) are "
            f"non-finite in y or p and would be silently dropped by "
            f"_check; refusing to score the survivors (SF-BL-12282)")


def _validate_brier_score_panel(y, p):
    """SF-BL-10940: strict panel screen for brier_score ONLY.

    _check (shared with expected_calibration_error /
    calibration_table / auc_mann_whitney, whose own forecast-range
    verdicts are separate tasks — SF-BL-10804 / SF-BL-10982) masks
    non-finite pairs and clips forecasts into [EPS, 1-EPS], so
    brier_score previously converted an impossible p=1.5 into a
    near-perfect 9.98e-31 loss via the clip (live repro:
    brier_score([1], [1.5]) == 9.98402083170343e-31) and scored
    p=-0.5 as 0.999999999999998 via the EPS floor. brier_score
    therefore validates its RAW inputs before _check runs and raises
    a named ValueError on any FINITE forecast outside [0, 1]. The
    screen is finite-only, exactly as preregistered: non-finite rows
    keep their pre-fix _check masking behaviour, and an empty panel
    keeps its pre-fix NaN return. Validation is pure — it never
    masks, drops, clips or transforms a value — so a lawful panel
    (including endpoints 0/1, which still reach the EPS clip)
    reaches the unchanged _check + formula path and its score is
    bit-identical.
    """
    y_arr = np.asarray(y, dtype=float); p_arr = np.asarray(p, dtype=float)
    if y_arr.shape != p_arr.shape:
        raise ValueError("y and p shapes differ")
    _finite_p = p_arr[np.isfinite(p_arr)]
    _bad = _finite_p[(_finite_p < 0.0) | (_finite_p > 1.0)]
    if _bad.size:
        raise ValueError(
            f"brier_score: forecasts p must lie in [0, 1]; got "
            f"{_bad.size} finite value(s) outside [0, 1] "
            f"(min={float(_bad.min())!r}, max={float(_bad.max())!r}) "
            f"(SF-BL-10940)")


def brier_score(y, p) -> float:
    """Mean Brier score (lower is better).

    SF-BL-10940: raises ValueError on any finite forecast outside
    [0, 1] — validation precedes the EPS clip in _check, which now
    applies only to lawful forecasts. Clean panels are scored by
    the unchanged formula and are bit-identical to the pre-fix
    result; empty panels keep their pre-fix behaviour (NaN).
    Non-finite rows no longer keep masked-survivor behaviour —
    see SF-BL-12282 below.

    SF-BL-10983: raises ValueError on any boolean value in the RAW
    forecasts — see _reject_bool_probabilities.

    SF-BL-11059: raises ValueError naming y/p on any string value in
    the RAW outcomes or forecasts — see _reject_string_panels. That
    screen runs first, before the float conversion in the panel
    validator below could parse a numeric string into a lawful
    number.

    SF-BL-11467: this scorer's string screen is re-fixtured
    together with expected_calibration_error and
    calibration_table — all three raise the named ValueError on
    any string-dtype forecast or outcome panel (kill form:
    brier_score(['1', '0'], ['0.9', '0.1']) must raise, never
    return 0.01), with numeric panels bit-identical.

    SF-BL-12282: raises ValueError naming the dropped-row count
    on any non-finite y/p row — see _reject_nonfinite_rows. That
    screen runs after the string/bool screens and before the
    range validator, so a non-finite row is named even when a
    finite row is also out of range. Clean panels bit-identical.
    """
    _reject_string_panels(y, p, "brier_score")
    _reject_bool_probabilities(p, "brier_score")
    _reject_nonfinite_rows(y, p, "brier_score")
    _validate_brier_score_panel(y, p)
    y, p = _check(y, p)
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def _validate_log_score_panel(y, p):
    """SF-BL-10071: strict panel screen for log_score ONLY.

    _check (shared with brier_score / expected_calibration_error /
    calibration_table, whose own verdicts are separate tasks) masks
    non-finite pairs silently and clips forecasts into [EPS, 1-EPS], so
    log_score previously (a) scored only the survivors of a NaN row with
    no dropped-row signal in its bare-float result, (b) converted an
    impossible p=1.5 into a near-perfect 9.99e-16 loss via the clip, and
    (c) returned NaN for an empty panel. log_score therefore validates
    its RAW inputs before _check runs: shape, emptiness, finiteness of
    BOTH arrays (a non-finite pair is exactly a row _check would drop —
    refused here, with the count, since a float result has no channel
    to surface it), and forecast range [0, 1]. Validation is pure — it
    never transforms a value — so a clean panel reaches the unchanged
    _check + formula path and its score is bit-identical.
    """
    y_arr = np.asarray(y, dtype=float); p_arr = np.asarray(p, dtype=float)
    if y_arr.shape != p_arr.shape:
        raise ValueError("y and p shapes differ")
    if y_arr.size == 0:
        raise ValueError(
            "log_score: empty panel — y and p contain no rows; a mean "
            "log loss over zero forecasts is undefined (SF-BL-10071)")
    n_dropped = int(np.sum(~(np.isfinite(y_arr) & np.isfinite(p_arr))))
    if n_dropped:
        raise ValueError(
            f"log_score: {n_dropped} of {y_arr.size} row(s) are "
            f"non-finite in y or p and would be silently dropped by "
            f"_check; refusing to score the survivors (SF-BL-10071)")
    lo, hi = float(p_arr.min()), float(p_arr.max())
    if lo < 0.0 or hi > 1.0:
        raise ValueError(
            f"log_score: forecasts p must lie in [0, 1]; got "
            f"min={lo!r}, max={hi!r} (SF-BL-10071)")


def log_score(y, p) -> float:
    """Mean log loss (lower is better).

    SF-BL-10071: raises ValueError on an empty panel, on any non-finite
    y/p row (rows _check would silently drop), and on any forecast
    outside [0, 1] — validation precedes the EPS clip, which now applies
    only to lawful endpoint forecasts 0/1. Clean panels are scored by
    the unchanged formula and are bit-identical to the pre-fix result.

    SF-BL-10983: raises ValueError on any boolean value in the RAW
    forecasts — see _reject_bool_probabilities. That screen runs
    after the string screen below and before the float conversion
    in the panel validator below could coerce True/False into
    lawful endpoints.

    SF-BL-11392: raises ValueError naming y/p on any string value
    in the RAW outcomes or forecasts — see _reject_string_panels.
    That screen runs first of all, before the bool screen and
    before _validate_log_score_panel's float coercion could parse
    a numeric string into a lawful number (kill form:
    log_score([1, 0], ['0.5', '0.5']) must raise, never return
    0.6931) — previously a direct log_score call was refused only
    by the `_check` backstop under `_check`'s name, and a
    non-numeric string died inside the coercion with a bare
    conversion error that named neither log_score nor the panel.
    Lawful panels are bit-identical.
    """
    _reject_string_panels(y, p, "log_score",
                          citations="(SF-BL-11059, SF-BL-11392)")
    _reject_bool_probabilities(p, "log_score")
    _validate_log_score_panel(y, p)
    y, p = _check(y, p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _validate_n_bins(n_bins) -> int:
    """SF-BL-9791: strict bin-count screen for expected_calibration_error,
    calibration_table and evaluate_forecasts.

    n_bins shapes the shared binning (np.linspace(0, 1, n_bins + 1)) but
    was never validated: n_bins=0 (and n_bins=-1, and False) returned
    ECE 0.0 — perfect calibration — on real forecasts, because the edge
    array collapses and range(n_bins) iterates zero times so the ECE
    accumulator stays at its 0.0 initialiser; True silently meant 1
    bin; and every other misconfiguration (2.5, 10.0, NaN, "10", None)
    died with a bare TypeError / numpy error that never named n_bins.
    backtest.py is the fleet's deflation/harness layer whose ECE feeds
    promotion decisions, so a misconfigured backtest must fail loudly,
    not certify perfect calibration. This screen therefore requires an
    integer TYPE (Python int or numpy integer; bool is rejected — it
    is an int subclass and True/False are not bin counts) with value
    >= 1, mirroring calibration.py's _validate_n_bins (SF-BL-9722) for
    its separate implementation and this module's
    _validate_n_variants_tried (SF-BL-10156). Validation is pure — it
    never coerces, rounds or transforms a lawful value (it returns
    int(n_bins), identical in value for every lawful count) — so valid
    panels, including the n_bins=10 default, reach the unchanged
    binning path and are bit-identical.
    """
    if isinstance(n_bins, bool) or not isinstance(
            n_bins, (int, np.integer)):
        raise ValueError(
            f"n_bins must be a positive integer (>= 1); got "
            f"n_bins={n_bins!r} ({type(n_bins).__name__}) (SF-BL-9791)")
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError(
            f"n_bins must be a positive integer (>= 1); got "
            f"n_bins={n_bins} (SF-BL-9791)")
    return n_bins


def _validate_ece_table_panel(y, p, scorer: str) -> None:
    """SF-BL-10804: strict forecast-range screen for
    expected_calibration_error and calibration_table ONLY.

    _check (shared with brier_score / log_score / auc_mann_whitney,
    whose own forecast-range verdicts are separate tasks —
    SF-BL-10940 / SF-BL-10071 / SF-BL-10982) masks non-finite pairs
    and clips forecasts into [EPS, 1-EPS], so these two scorers
    previously converted an impossible p=1.5 into a near-perfect
    ECE of 9.99e-16 via the clip (live repro:
    expected_calibration_error([1], [1.5]) == 9.992007221626409e-16)
    and binned the clipped value in the calibration table — while
    calibration.py's ece / calibration_bins raise on the same row
    under the SF-BL-9484 range standard (_validate_p_in_range).
    This screen therefore validates the panel before _check runs
    and raises a named ValueError on any FINITE forecast outside
    [0, 1]. The screen is finite-only by design: non-finite rows
    are already refused, with their dropped-row count, by
    _reject_nonfinite_rows (SF-BL-12282), which runs before this
    screen in both scorers, so the raise verdict on non-finite
    forecasts matches calibration.py as well. It MUST run after
    the string screen (and the bool screen, where present):
    float conversion silently parses numeric strings and maps
    bools to endpoints, so those defects are named by their own
    screens first. Validation is pure — it never masks, drops,
    clips or transforms a value — so a lawful panel (including
    endpoints 0/1, which still reach the EPS clip) reaches the
    unchanged _check + binning path and its result is
    bit-identical; an empty panel (0 finite forecasts) keeps its
    pre-fix behaviour.
    """
    y_arr = np.asarray(y, dtype=float); p_arr = np.asarray(p, dtype=float)
    if y_arr.shape != p_arr.shape:
        raise ValueError("y and p shapes differ")
    _finite_p = p_arr[np.isfinite(p_arr)]
    _bad = _finite_p[(_finite_p < 0.0) | (_finite_p > 1.0)]
    if _bad.size:
        raise ValueError(
            f"{scorer}: forecasts p must lie in [0, 1]; got "
            f"{_bad.size} finite value(s) outside [0, 1] "
            f"(min={float(_bad.min())!r}, max={float(_bad.max())!r}) "
            f"(SF-BL-10804)")


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """Expected calibration error (count-weighted mean |rate - pred|).

    SF-BL-9791: raises ValueError unless n_bins is a positive integer
    (>= 1) — see _validate_n_bins. The screen runs first, before
    _check, so an invalid count raises even on an empty panel (which
    otherwise returns NaN). Lawful counts are scored by the unchanged
    binning path and are bit-identical to the pre-fix result.

    SF-BL-10983: raises ValueError on any boolean value in the RAW
    forecasts — see _reject_bool_probabilities. That screen runs
    before _validate_n_bins and before _check's float
    conversion could coerce True/False into lawful endpoints.

    SF-BL-11467: raises ValueError naming y/p on any string value
    in the RAW outcomes or forecasts — see
    _reject_string_panels. That screen runs first of all, before
    the bool screen above: previously this scorer had no string
    screen of its own, so a string panel was refused only by
    the `_check` backstop under `_check`'s name — and
    SF-BL-11467's repro records the underlying defect, string
    panels scored as ECE 0.1 bit-identical to the numeric
    control through `_check`'s float coercion. Lawful panels
    are bit-identical.

    SF-BL-12282: raises ValueError naming the dropped-row count
    on any non-finite y/p row — see _reject_nonfinite_rows
    (kill fixture: ECE([0, 1], [NaN, 0.9]) previously returned
    0.1, the surviving row alone). That screen runs after the
    string/bool screens and before _validate_n_bins, so a
    non-finite row is always named. Clean panels bit-identical.

    SF-BL-10804: raises ValueError on any finite forecast outside
    [0, 1] — see _validate_ece_table_panel. That screen runs after
    _validate_n_bins and before _check, so an impossible forecast
    is refused instead of being clipped into a near-perfect ECE
    (pre-fix ECE([1], [1.5]) == 9.99e-16); its raise verdict now
    matches calibration.py's ece on every vector. Clean panels
    bit-identical.
    """
    _reject_string_panels(y, p, "expected_calibration_error")
    _reject_bool_probabilities(p, "expected_calibration_error")
    _reject_nonfinite_rows(y, p, "expected_calibration_error")
    n_bins = _validate_n_bins(n_bins)
    _validate_ece_table_panel(y, p, "expected_calibration_error")
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
    """AUC via the Mann-Whitney U statistic (rank-based, ties = 0.5).

    SF-BL-11017: raises ValueError on any boolean value in the RAW
    forecasts — see _reject_bool_probabilities. Previously this
    entry point ran _check first, whose float conversion mapped
    True -> 1.0 / False -> 0.0, so auc_mann_whitney([0, 1],
    [False, True]) returned a perfect 1.0 for a boolean flag panel.
    The screen runs first, before _check; validation is pure, so
    lawful panels are bit-identical.

    SF-BL-11391: raises ValueError naming y/p on any STRING value
    in the RAW outcomes or forecasts — see _reject_string_panels —
    and on any boolean value in the RAW OUTCOMES — see
    _reject_bool_outcomes. Previously this entry point had no
    string screen of its own, so a string panel was refused only
    by the `_check` backstop under `_check`'s name
    (auc_mann_whitney([1, 0], ['0.9', '0.1']) is this task's kill
    form: it must raise naming auc_mann_whitney and p, never
    return the numeric control's 1.0), and no screen inspected
    `y` for bools at all, so auc_mann_whitney([True, False],
    [0.9, 0.1]) returned 1.0. The string screen runs first of
    all, then the bool-outcome screen, then the SF-BL-11017
    bool-forecast screen — all before _check's float conversion
    can parse a numeric string or map a bool to an endpoint.
    Validation is pure, so lawful panels are bit-identical. This
    task's broader any-element / either-panel scope supersedes
    SF-BL-11017's p-only scope note for this entry point only.
    """
    _reject_string_panels(y, p, "auc_mann_whitney",
                          citations="(SF-BL-11059, SF-BL-11467, SF-BL-11391)")
    _reject_bool_outcomes(y, "auc_mann_whitney")
    _reject_bool_probabilities(p, "auc_mann_whitney",
                               citations="(SF-BL-10983, SF-BL-11391)")
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
    """Per-bin calibration summary (shares ECE's binning).

    SF-BL-9791: raises ValueError unless n_bins is a positive integer
    (>= 1) — see _validate_n_bins. Previously n_bins=0 returned an
    empty (0, 0) frame with no error. Lawful counts are bit-identical.

    SF-BL-11467: raises ValueError naming y/p on any string value
    in the RAW outcomes or forecasts — see
    _reject_string_panels. That screen runs first, before
    _validate_n_bins: previously this scorer had no panel screen
    of its own, so a string panel was refused only by the
    `_check` backstop under `_check`'s name — and SF-BL-11467's
    repro records it returning a full (10, 6) table through the
    shared `_check` float coercion. Lawful panels are
    bit-identical.

    SF-BL-12282: raises ValueError naming the dropped-row count
    on any non-finite y/p row — see _reject_nonfinite_rows
    (previously the table's n summed to 1 of 2 rows on
    ([0, 1], [NaN, 0.9]), the dropped row invisible). That
    screen runs after the string screen and before
    _validate_n_bins. Clean panels bit-identical.

    SF-BL-10804: raises ValueError on any finite forecast outside
    [0, 1] — see _validate_ece_table_panel — and on any boolean
    forecast, via the _reject_bool_probabilities screen ECE
    already had (SF-BL-10983): calibration.py's calibration_bins
    rejects bool forecasts (SF-BL-10938), so parity requires the
    screen here too; previously True was coerced to the 1-EPS
    endpoint and binned as a lawful forecast. The bool screen
    runs after the string screen and before the non-finite
    screen; the range screen runs after _validate_n_bins and
    before _check. Clean panels bit-identical.

    SF-BL-11017: this entry point is one of the three named in
    that task (with evaluate_forecasts and auc_mann_whitney,
    which gained their screens there); its bool screen above
    already satisfies the task — verified by live repro, no code
    change was needed here.
    """
    _reject_string_panels(y, p, "calibration_table")
    _reject_bool_probabilities(p, "calibration_table")
    _reject_nonfinite_rows(y, p, "calibration_table")
    n_bins = _validate_n_bins(n_bins)
    _validate_ece_table_panel(y, p, "calibration_table")
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


def _validate_evaluate_forecasts_panel(y, p):
    """SF-BL-11018: strict RAW-panel range screen for evaluate_forecasts ONLY.

    evaluate_forecasts previously ran _check FIRST and passed only the
    clipped survivors to brier_score / log_score, so a finite impossible
    forecast was laundered before any strict scorer could see it:
    evaluate_forecasts([1, 0], [1.5, 0.5]) returned Brier 0.125, log
    score 0.3466 and mean_pred 0.75, reporting the clipped panel as if
    1.5 had never existed — defeating log_score's SF-BL-10071 raw-panel
    screen at the fleet's main scorecard entry point. This screen
    therefore validates the RAW inputs before _check runs and raises a
    named ValueError on any FINITE forecast outside [0, 1] (min/max
    computed over finite p only, so a NaN cannot poison the diagnostic).
    Validation is pure — it never masks, drops, clips or transforms a
    value — so a panel with no finite out-of-range forecast reaches the
    unchanged _check + scorer path and its scorecard is bit-identical.
    Non-finite rows are NOT screened here: _check masking / dropped-row
    disclosure for them is SF-BL-9295's scope, and their behaviour is
    unchanged. Per-scorer direct-call screens (SF-BL-10940 / 10804 /
    10982) are likewise untouched.
    """
    y_arr = np.asarray(y, dtype=float); p_arr = np.asarray(p, dtype=float)
    if y_arr.shape != p_arr.shape:
        raise ValueError("y and p shapes differ")
    finite_p = p_arr[np.isfinite(p_arr)]
    if finite_p.size:
        lo, hi = float(finite_p.min()), float(finite_p.max())
        if lo < 0.0 or hi > 1.0:
            raise ValueError(
                f"evaluate_forecasts: forecasts p must lie in [0, 1]; got "
                f"min={lo!r}, max={hi!r} (SF-BL-11018)")


def evaluate_forecasts(y, p, n_bins: int = 10) -> dict:
    """Full scorecard for a forecast panel.

    SF-BL-11018: raises ValueError on any finite forecast outside
    [0, 1] — the RAW panel is screened before _check runs, so an
    impossible forecast can no longer be clipped into a lawful-looking
    scorecard. Lawful panels (and panels whose only defect is
    non-finite rows, SF-BL-9295 scope) are scored by the unchanged
    path and are bit-identical to the pre-fix result.

    SF-BL-9791: raises ValueError unless n_bins is a positive integer
    (>= 1) — validated before the panel range screen, so a bad bin
    count can never be laundered through the scorecard's ECE /
    calibration entries. Lawful counts are bit-identical.

    SF-BL-11059: raises ValueError naming y/p on any string value in
    the RAW outcomes or forecasts — see _reject_string_panels. That
    screen runs first of all, before _check (reached below) could
    parse a numeric string into a lawful number and return a
    scorecard for it.

    SF-BL-11017: raises ValueError on any boolean value in the RAW
    forecasts — see _reject_bool_probabilities. Previously _check
    ran before any bool screen, float-converting True -> 1.0 /
    False -> 0.0, so evaluate_forecasts([1, 0], [True, False])
    returned a perfect scorecard (Brier ~1e-30, AUC 1.0) and the
    strict bool screens in the downstream scorers never saw the
    raw flags. The screen runs right after the string screen and
    before _validate_n_bins / the range screen / _check; it is
    pure, so lawful panels are bit-identical.
    """
    _reject_string_panels(y, p, "evaluate_forecasts")
    _reject_bool_probabilities(p, "evaluate_forecasts")
    n_bins = _validate_n_bins(n_bins)
    _validate_evaluate_forecasts_panel(y, p)
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
def _validate_climatology_history(y_train) -> None:
    """SF-BL-10115: strict history screen for climatology_forecast ONLY.

    climatology_forecast is np.full(n, mean(y_train)) — the denominator
    of every Brier skill score (BENCHMARKS['climatology']) — and
    previously screened nothing: an empty history returned [NaN, NaN,
    NaN], a NaN row returned NaN forecasts, and a non-binary history
    (y=2) returned a forecast of 2.0, outside [0, 1], each shipped as
    an ordinary benchmark forecast. This screen therefore validates
    the RAW y_train before any mean runs and raises a named ValueError
    on (1) an empty history, (2) any non-finite value, and (3) any
    finite value that is not exactly 0 or 1. Validation is pure — it
    never masks, drops, clips or transforms a value — so a lawful
    binary history reaches the unchanged mean path and its forecast
    is bit-identical.
    """
    arr = np.asarray(y_train, dtype=float)
    if arr.size == 0:
        raise ValueError(
            "climatology_forecast: y_train is empty \u2014 no training "
            "history to compute a climatology base rate from "
            "(SF-BL-10115)")
    n_nonfinite = int(np.sum(~np.isfinite(arr)))
    if n_nonfinite:
        raise ValueError(
            f"climatology_forecast: y_train contains {n_nonfinite} of "
            f"{arr.size} non-finite value(s) (NaN/inf); refusing to "
            f"emit a corrupt climatology benchmark (SF-BL-10115)")
    nonbinary = arr[(arr != 0.0) & (arr != 1.0)]
    if nonbinary.size:
        raise ValueError(
            f"climatology_forecast: y_train must be binary (0/1); got "
            f"{nonbinary.size} non-binary value(s) "
            f"(min={float(nonbinary.min())!r}, "
            f"max={float(nonbinary.max())!r}) (SF-BL-10115)")


def _validate_benchmark_horizon(horizon, fn_name: str) -> None:
    """SF-BL-11129: strict horizon screen for the three BENCHMARKS
    (climatology_forecast / persistence_forecast / random_walk_forecast).

    All three bodies size their output from len(X_test) and never read
    `horizon`, yet WalkForwardBacktester.run passes each fold's h to
    them as if it were live — live repros returned bit-identical arrays
    for horizon='2', True, None, -5 and 999, so per-horizon benchmark
    comparisons in per_horizon results were restatements of one number
    even though these benchmarks are the denominator of every Brier
    skill score the backtester reports. The three benchmarks are
    horizon-INVARIANT by construction — each holds one constant
    probability over the test block, and horizon selects the target
    timing in the backtester, not the benchmark value — so the honest
    contract is validation, not removal: `horizon` stays in the shared
    forecaster protocol (it is genuinely live for e.g. ewma_forecaster)
    but must be a positive integer, exactly as this module screens its
    other integer arguments (make_splits step SF-BL-9845,
    _validate_n_bins SF-BL-9791, assert_causal_series shift
    SF-BL-9884): integer TYPE required (Python int or numpy integer;
    bool rejected — it is an int subclass and True/False are not
    horizons; an integral float such as 2.0 is likewise rejected) and
    >= 1. Validation is pure (raises only, never coerces a lawful
    value), so every lawful positive-integer horizon — including the
    horizon=1 default — returns a forecast bit-identical to before.
    """
    if isinstance(horizon, bool) or not isinstance(
            horizon, (int, np.integer)):
        raise ValueError(
            f"{fn_name}: horizon must be a positive integer, got "
            f"horizon={horizon!r} ({type(horizon).__name__}) "
            f"(SF-BL-11129)")
    if horizon < 1:
        raise ValueError(
            f"{fn_name}: horizon must be a positive integer, got "
            f"horizon={horizon!r} (SF-BL-11129)")


def climatology_forecast(y_train, X_train=None, X_test=None, horizon=1):
    """Climatology benchmark: training base rate held constant.

    SF-BL-10115: raises ValueError on an empty, non-finite or
    non-binary y_train — see _validate_climatology_history. Lawful
    binary histories return bit-identical forecasts.

    SF-BL-11129: raises ValueError naming horizon on any horizon that
    is not a positive integer — see _validate_benchmark_horizon. The
    forecast itself is horizon-invariant (a constant base rate over
    the test block); horizon is a validated protocol argument.
    """
    _validate_benchmark_horizon(horizon, "climatology_forecast")
    _validate_climatology_history(y_train)
    n = len(X_test) if X_test is not None else 1
    return np.full(n, float(np.mean(y_train)))


def persistence_forecast(y_train, X_train=None, X_test=None, horizon=1):
    """Last observed outcome, shrunk slightly toward the base rate (a prob of
    exactly 0/1 is infinitely punished by log score, so clip).

    SF-BL-9855: raises ValueError on an empty y_train — see the validation
    below. Valid histories are computed by the unchanged formula and are
    bit-identical to the pre-fix result.

    SF-BL-11129: raises ValueError naming horizon on any horizon that
    is not a positive integer — see _validate_benchmark_horizon. The
    forecast itself is horizon-invariant (the last observation, shrunk
    toward the base rate, held over the test block); horizon is a
    validated protocol argument.
    """
    _validate_benchmark_horizon(horizon, "persistence_forecast")
    # SF-BL-9855: validate the RAW history BEFORE y_train[-1] is indexed.
    # Previously an empty history died with a bare IndexError (list index
    # out of range / index -1, size 0) that never named y_train, while the
    # sibling benchmarks climatology (SF-BL-10115) and random_walk
    # (SF-BL-10116) already raised a named ValueError on the same input —
    # the three BENCHMARKS must fail uniformly, since they anchor every
    # Brier comparison the backtester reports. Validation is pure: it
    # inspects a float view for the check only and never masks, drops,
    # clips or transforms a value, and the computation below still
    # operates on the original y_train argument, so every non-empty
    # history is bit-identical.
    if np.asarray(y_train, dtype=float).size == 0:
        raise ValueError(
            "persistence_forecast: empty y_train — no training history "
            "to take a last observation from (SF-BL-9855)")
    n = len(X_test) if X_test is not None else 1
    last = float(y_train[-1]); base = float(np.mean(y_train))
    return np.full(n, np.clip(0.7 * last + 0.3 * base, 0.01, 0.99))


def random_walk_forecast(y_train, X_train=None, X_test=None, horizon=1):
    """No-change in *probability space*: EWMA prob held constant over the test block.

    SF-BL-10116: raises ValueError on an empty y_train or on any
    non-finite (NaN / +/-inf) value in y_train — see the validation
    below. Valid histories are computed by the unchanged EWMA path and
    are bit-identical to the pre-fix result.

    SF-BL-11129: raises ValueError naming horizon on any horizon that
    is not a positive integer — see _validate_benchmark_horizon. The
    forecast itself is horizon-invariant (an EWMA probability held
    over the test block); horizon is a validated protocol argument.
    """
    _validate_benchmark_horizon(horizon, "random_walk_forecast")
    # SF-BL-10116: validate the RAW history BEFORE the EWMA weights are
    # built. Previously the weights were built from
    # min(len(y_train), 50) points with no empty/finite guard, so an
    # empty history silently became an empty dot product (0.0) clipped
    # up to a maximally confident 0.01 benchmark invented from zero
    # observations, a NaN row propagated NaN forecasts, and +/-inf was
    # clipped into a confident 0.99 / 0.01 — all returned as normal
    # BENCHMARKS entries. Validation is pure: it inspects a float view
    # for the check only and never masks, drops, clips or transforms a
    # value, and the computation below still operates on the original
    # y_train argument, so every valid history is bit-identical.
    _y_arr = np.asarray(y_train, dtype=float)
    if _y_arr.size == 0:
        raise ValueError(
            "random_walk_forecast: empty y_train — no training history "
            "to form an EWMA benchmark from (SF-BL-10116)")
    _n_nonfinite = int(np.sum(~np.isfinite(_y_arr)))
    if _n_nonfinite:
        raise ValueError(
            f"random_walk_forecast: y_train contains {_n_nonfinite} "
            f"non-finite value(s) (NaN/inf); refusing to propagate "
            f"them into a benchmark forecast (SF-BL-10116)")
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
def _validate_block_bootstrap_args(block, n_boot, seed):
    """SF-BL-11096: strict TYPE + value screen for
    block_bootstrap_brier_diff's sizing and seed arguments.

    block / n_boot shape the resampling and seed drives
    np.random.default_rng, but none of the three was ever
    type-validated: SF-BL-9814 fixtured the sizing arguments' value
    domain and SF-BL-11021 the forecast panels' range, yet no item
    fixtured their types, so on a 40-row fixture block=True was
    silently read as block=1 and returned a bootstrap dict (diff
    0.225 on the filed fixture), destroying the block structure the
    registry's bootstrap verdicts depend on; seed=True / seed=False
    likewise silently ran with entropy 1 / 0, and seed=None silently
    drew fresh entropy. Every other mis-typed call died unnamed
    instead: n_boot='10' and block=2.5 with bare TypeErrors that
    never name the argument, seed='0' with a bare SeedSequence
    TypeError, block=0 / block=False with a ZeroDivisionError and
    n_boot=0 with an IndexError. This screen therefore requires an
    integer TYPE for all three arguments (Python int or numpy
    integer; bool / np.bool_ rejected — bool is an int subclass and
    True/False are not sizes or seeds; every float is rejected,
    fractional or integral, mirroring _validate_n_bins /
    _validate_n_variants_tried in this module and calibration.py's
    _validate_pool_seed, SF-BL-11088), block / n_boot >= 1 and
    seed >= 0, and raises a ValueError NAMING the offending argument
    for every other value (strings, None, sequences and NaN / +/-inf
    included). Validation is pure — lawful values are never coerced,
    rounded or transformed (the returned int(...) of an
    already-integer sizing value is identical in value, and seed is
    passed to default_rng exactly as given) — so integer sizing with
    an integer seed returns bit-identical dicts.
    """
    for _name, _value in (("block", block), ("n_boot", n_boot)):
        if isinstance(_value, (bool, np.bool_)) or not isinstance(
                _value, (int, np.integer)):
            raise ValueError(
                f"block_bootstrap_brier_diff: {_name} must be a "
                f"positive integer (>= 1); got {_name}={_value!r} "
                f"({type(_value).__name__}) (SF-BL-11096)")
        if _value < 1:
            raise ValueError(
                f"block_bootstrap_brier_diff: {_name} must be a "
                f"positive integer (>= 1); got {_name}={_value!r} "
                f"(SF-BL-11096)")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)):
        raise ValueError(
            f"block_bootstrap_brier_diff: seed must be a "
            f"non-negative integer; got seed={seed!r} "
            f"({type(seed).__name__}) (SF-BL-11096)")
    if seed < 0:
        raise ValueError(
            f"block_bootstrap_brier_diff: seed must be a "
            f"non-negative integer; got seed={seed!r} (SF-BL-11096)")
    return int(block), int(n_boot)


def _is_object_dtype_panel(raw) -> bool:
    """SF-BL-11430: True iff a RAW panel container is object-dtype.

    An object-dtype ndarray / pandas Series / Index has no numeric
    dtype contract: its elements are arbitrary Python objects that
    ``np.asarray(..., dtype=float)`` coerces elementwise, so a text
    export, a flag column or a boxed-number column is
    indistinguishable from a lawful numeric panel after conversion.
    The dtype itself is the defect, so this returns True even for an
    empty object-dtype panel. Plain lists / tuples have no dtype and
    numeric-dtype containers are lawful; their ELEMENTS are screened
    by _has_string_value / _has_bool_probability.
    """
    if isinstance(raw, np.ndarray):
        return raw.dtype == object
    if isinstance(raw, (pd.Series, pd.Index)):
        return raw.dtype == object
    return False


def _is_nullable_bool_panel(raw) -> bool:
    """SF-BL-11430: True iff a RAW pandas container carries the
    nullable BooleanDtype. _has_bool_probability matches the plain
    bool dtype only, so a ``pd.Series(..., dtype="boolean")`` panel
    slipped past it and float-coerced to its [1, 0] control; the
    nullable dtype is bool-typed by construction, so it counts even
    when the panel is empty or holds pd.NA."""
    if isinstance(raw, (pd.Series, pd.Index)):
        return isinstance(raw.dtype, pd.BooleanDtype)
    return False


def _validate_block_bootstrap_panels(y, p_model, p_bench) -> None:
    """SF-BL-11430: strict TYPE screen for block_bootstrap_brier_diff's
    three panels (y / p_model / p_bench).

    This function — the fleet's deflated Brier-difference readout —
    never type-validated its panels: the SF-BL-11021 range screen and
    the shared _check both run ``np.asarray(..., dtype=float)``,
    which silently parses numeric strings and maps True -> 1.0 /
    False -> 0.0, so in live repros (n=40, n_boot=20, seed=1) string
    p_model and string y panels returned diff 0.205 bit-identical to
    their numeric controls and a bool p_bench panel returned -0.045
    bit-identical to its [1, 0] control. The only string protection
    was the _check backstop, which names the generic "_check" / "p",
    never this function or its p_model / p_bench panels — and
    non-numeric strings in p_model / p_bench died even earlier, with
    a bare unnamed "could not convert string to float" ValueError
    from the range screen's float coercion. SF-BL-11096 fixtured only
    this function's block / n_boot / seed argument types and
    SF-BL-11021 only a finite out-of-range p_model value, so its
    panels' type domain was unfixtured. This screen therefore runs
    on the RAW panels, after the SF-BL-11096 sizing screen and
    before the SF-BL-11021 range screen's float coercion, and raises
    a ValueError NAMING the offending panel (y / p_model / p_bench)
    on any string value (via _has_string_value, SF-BL-11059 —
    including string-dtype and string-categorical containers), any
    boolean value (via _has_bool_probability, SF-BL-10983 —
    including bool-dtype and bool-categorical containers; a bool is
    not an outcome or a probability, it is an int subclass, which is
    exactly why it must be rejected explicitly) or any object-dtype
    panel (via _is_object_dtype_panel above — object dtype is
    refused as a dtype, whatever its elements). Validation is pure
    — it never coerces, parses or transforms a value — so lawful
    float / int panels (ndarrays, lists, tuples, numeric pandas
    Series) reach the unchanged SF-BL-11021 screen + _check +
    bootstrap path and return bit-identical diff / CI / p at fixed
    seed.
    """
    for _panel, _raw, _kind in (
            ("y", y, "outcome"), ("p_model", p_model, "forecast"),
            ("p_bench", p_bench, "forecast")):
        if _has_string_value(_raw):
            raise ValueError(
                f"block_bootstrap_brier_diff: {_panel} must not "
                f"contain string values; got string value(s) in "
                f"{_panel} (a numeric string is not a valid "
                f"{_kind}) (SF-BL-11430)")
        if _has_bool_probability(_raw) or _is_nullable_bool_panel(_raw):
            raise ValueError(
                f"block_bootstrap_brier_diff: {_panel} must not "
                f"contain boolean values; got boolean value(s) in "
                f"{_panel} (a bool is not a valid {_kind}) "
                f"(SF-BL-11430)")
        if _is_object_dtype_panel(_raw):
            raise ValueError(
                f"block_bootstrap_brier_diff: {_panel} must not be "
                f"object-dtype; got object-dtype {_panel} (an "
                f"object-dtype panel has no numeric dtype contract) "
                f"(SF-BL-11430)")


def block_bootstrap_brier_diff(
    y, p_model, p_bench, *, block: int = 20, n_boot: int = 2000, seed: int = 0,
) -> dict:
    """
    Moving-block bootstrap of  mean[(p_bench-y)^2 - (p_model-y)^2]
    (positive = model better). Returns point estimate, percentile CI, and
    two-sided bootstrap p-value for diff <= 0.

    SF-BL-11021: raises ValueError on any FINITE p_model / p_bench value
    outside [0, 1] — see the validation below.

    SF-BL-11096: raises ValueError naming block / n_boot / seed unless
    block and n_boot are positive integers and seed is a non-negative
    integer — see _validate_block_bootstrap_args. That screen runs
    first, before the panel screen below.

    SF-BL-11430: raises ValueError naming y / p_model / p_bench on
    any string or boolean element in, or any object-dtype, panel —
    see _validate_block_bootstrap_panels. That screen runs after the
    sizing screen and before the SF-BL-11021 range screen below, so
    a mis-typed panel is named by this function instead of being
    float-coerced into a lawful-looking bootstrap verdict.
    """
    # SF-BL-11096: type-screen the sizing / seed arguments before any
    # panel or resampling work — a mis-typed block / n_boot / seed
    # must fail loudly and named, never silently resample (block=True
    # ran as block=1) or die inside numpy with an unnamed TypeError.
    block, n_boot = _validate_block_bootstrap_args(block, n_boot, seed)
    # SF-BL-11430: type-screen the RAW panels before any float
    # coercion (the SF-BL-11021 screen below or _check) can parse a
    # numeric string or map a bool to an endpoint — a mis-typed
    # panel must fail loudly under its own name (y / p_model /
    # p_bench), never be certified as a bootstrap diff.
    _validate_block_bootstrap_panels(y, p_model, p_bench)
    # SF-BL-11021: screen the RAW forecast panels BEFORE _check runs.
    # _check clips forecasts into [EPS, 1-EPS], so an impossible panel
    # such as p_model=[1.5]*30 on y=[1]*30 was silently converted into a
    # near-perfect model and certified with diff 0.25, CI [0.25, 0.25]
    # and p_model_not_better 0.0 — the bootstrap certifying a
    # certainly-better model from an impossible forecast. The screen is
    # finite-only, exactly as preregistered: non-finite values keep
    # their pre-fix _check masking behaviour (that class is
    # SF-BL-9294's scope). Validation is pure — it never transforms a
    # value — so lawful panels reach the unchanged _check + bootstrap
    # path and return bit-identical dicts.
    for _panel, _raw in (("p_model", p_model), ("p_bench", p_bench)):
        _arr = np.asarray(_raw, dtype=float)
        _finite = _arr[np.isfinite(_arr)]
        _bad = _finite[(_finite < 0.0) | (_finite > 1.0)]
        if _bad.size:
            raise ValueError(
                f"block_bootstrap_brier_diff: {_panel} forecasts must lie "
                f"in [0, 1]; got {_bad.size} finite value(s) outside "
                f"[0, 1] (min={float(_bad.min())!r}, "
                f"max={float(_bad.max())!r}) (SF-BL-11021)")
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
def _validate_n_variants_tried(n_variants_tried) -> None:
    """SF-BL-10156: strict trial-count screen for deflated_brier_pvalue
    and multiple_testing_report.

    r53 discipline requires every comparison registered with an integer
    trial count, yet the only guard was `n_variants_tried < 1`, which
    fails open: `2.5 < 1` is False (a fractional count passed and was
    used as a fractional exponent), and every IEEE-754 comparison with
    NaN is False, so NaN / +inf counts passed too —
    multiple_testing_report(2.5, 0.05) formatted '2.5 variant(s)' with
    p_adj 0.1204 and multiple_testing_report(NaN, 0.05) formatted a note
    with p_adj nan. This screen therefore requires an integer TYPE
    (Python int or numpy integer; bool is rejected — it is an int
    subclass and True is not a trial count) with value >= 1, and raises
    a named ValueError for every other value, including fractional,
    integral-float (3.0 would format as '3.0 variant(s)'), NaN and
    +/-inf counts. Validation is pure — it never coerces, rounds or
    transforms a value — so lawful integer counts reach the unchanged
    formula / report string and are bit-identical.
    """
    if isinstance(n_variants_tried, bool) or not isinstance(
            n_variants_tried, (int, np.integer)):
        raise ValueError(
            f"n_variants_tried must be a finite integer >= 1, got "
            f"{n_variants_tried!r} (SF-BL-10156)")
    if n_variants_tried < 1:
        raise ValueError(
            f"n_variants_tried must be a finite integer >= 1, got "
            f"{n_variants_tried!r} (SF-BL-10156)")


def _validate_p_raw_type(p_raw) -> None:
    """SF-BL-11094: strict TYPE screen for deflated_brier_pvalue's
    p_raw (inherited by multiple_testing_report).

    SF-BL-9792 fixtures p_raw's finite/range VALUE domain, but a bool
    passes that screen lawfully — math.isfinite(True) is True and
    0.0 <= True <= 1.0 reads it as the endpoint 1.0 — so
    deflated_brier_pvalue(True, N) returned 1.0, the maximum deflated
    p, and False returned 0.0: a bool flag in a registry row silently
    manufactured the strongest possible deflation verdict. A
    non-numeric p_raw failed the other way: p_raw='0.05' died inside
    math.isfinite with a bare TypeError ("must be real number, not
    str") that never names p_raw. This screen therefore requires a
    real-number TYPE (numbers.Real — Python int / float, every numpy
    integer / floating scalar, Fraction; bool and np.bool_ are
    rejected explicitly — bool is an int subclass and True/False are
    not probabilities) and raises a named ValueError for every other
    type, including str, None, containers, complex and Decimal (not a
    numbers.Real; a serialized probability must be converted by its
    producer, never coerced here — cf. SF-BL-11014 for the same
    discipline on alpha_kelly's separate p_raw). Validation is pure
    — it never coerces, parses or transforms a value — so every real
    numeric p_raw reaches the unchanged SF-BL-9792 screen and formula
    bit-identically.
    """
    if isinstance(p_raw, (bool, np.bool_)) or not isinstance(
            p_raw, numbers.Real):
        raise ValueError(
            f"p_raw must be a real number (int / float / numpy "
            f"numeric, bool excluded), got p_raw={p_raw!r} "
            f"({type(p_raw).__name__}) (SF-BL-11094)")


def deflated_brier_pvalue(p_raw: float, n_variants_tried: int) -> float:
    """Bonferroni-style deflation: probability at least one of N independent
    variants achieves p_raw by chance. Conservative, simple, honest.

    SF-BL-10156: n_variants_tried must be a finite integer >= 1 — see
    _validate_n_variants_tried. That screen runs first, preserving the
    legacy precedence (an invalid count raises before the SF-BL-9792
    p_raw screen when both arguments are invalid).

    SF-BL-11094: p_raw must be a real number — see
    _validate_p_raw_type. That screen runs after the count screen and
    before the SF-BL-9792 value screen, so a bool or non-numeric
    p_raw raises a ValueError naming p_raw instead of returning the
    endpoint verdict (True -> 1.0) or dying with a bare TypeError.
    """
    _validate_n_variants_tried(n_variants_tried)
    _validate_p_raw_type(p_raw)
    # SF-BL-9792: p_raw must be a lawful probability. Without this screen an
    # impossible raw p silently manufactured a lawful-looking deflated entry
    # (p_raw=1.5 -> 0.9990 at N=10, p_raw=-0.1 -> a negative "probability",
    # p_raw=NaN -> NaN, p_raw=inf -> -inf). Validation is pure — it never
    # transforms p_raw — so in-range outputs are bit-identical.
    # SF-BL-11094: the finiteness test is OverflowError-safe — a huge
    # int such as 10**400 is a real number (it passes the type screen
    # above) but math.isfinite raises OverflowError converting it;
    # it is non-finite for this purpose and takes the same ValueError.
    try:
        _p_raw_finite = math.isfinite(p_raw)
    except OverflowError:
        _p_raw_finite = False
    if not _p_raw_finite or not 0.0 <= p_raw <= 1.0:
        raise ValueError(
            f"p_raw must be a finite probability in [0, 1], got {p_raw!r} "
            f"(SF-BL-9792)")
    return float(1 - (1 - p_raw) ** n_variants_tried)


def multiple_testing_report(n_variants_tried: int, p_raw: float) -> str:
    """Human-facing multiple-testing registry note.

    SF-BL-10156: raises ValueError unless n_variants_tried is a finite
    integer >= 1 — validated here as the first statement, so the report
    path can never format a deflation note for a fractional or NaN
    count; deflated_brier_pvalue re-validates on the way through. The
    report string for integer counts is bit-identical to pre-fix.

    SF-BL-11094: a bool or non-numeric p_raw raises the named
    ValueError from deflated_brier_pvalue's type screen (see
    _validate_p_raw_type), which runs before this function formats
    p_raw — previously report(10, True) printed a note with raw
    p=1.0000 / p_adj=1.0000 and report(10, '0.05') died with the same
    bare TypeError as the direct call.
    """
    _validate_n_variants_tried(n_variants_tried)
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

    SF-BL-11130: test_size must be a positive integer — __init__ (and
    run, for a mutated attribute) raises ValueError naming test_size on
    any bool, string or non-integer size; see _validate_test_size.

    SF-BL-11158: initial_train must be a positive integer — __init__
    (and run, for a mutated attribute) raises ValueError naming
    initial_train on any bool, string, float or non-positive value;
    see _validate_initial_train.

    SF-BL-11352: purge must be a non-negative integer — __init__
    (and run, for a mutated attribute) raises ValueError naming
    purge on any bool, string, float or negative value;
    see _validate_purge.

    SF-BL-11385: rolling_train must be None or a positive integer —
    __init__ (and run, for a mutated attribute) raises ValueError
    naming rolling_train on any bool, string, float or
    non-positive value; see _validate_rolling_train.

    SF-BL-11353: embargo must be an integer — __init__ (and run,
    for a mutated attribute) raises ValueError naming embargo on
    any bool, string, float or other non-integer value; see
    _validate_embargo (SF-BL-11132, SF-BL-11353).
    """

    def __init__(self, *, window="expanding", initial_train=200, test_size=50,
                 step=None, purge=0, embargo=0, rolling_train=None,
                 benchmarks: bool = True, bootstrap_block=20, n_boot=1000, seed=0):
        # SF-BL-11130: type-validate test_size at construction — the
        # constructor previously stored and forwarded it with no screen,
        # so a bool/string/non-integer size only surfaced (if at all)
        # deep inside make_splits at run time. See _validate_test_size.
        test_size = _validate_test_size(test_size)
        # SF-BL-11158: type-validate initial_train at construction,
        # mirroring the test_size screen above — the constructor
        # previously stored and forwarded it unscreened, so a bool /
        # string / float size only surfaced (if at all) as a silent
        # empty split list deep inside make_splits at run time.
        initial_train = _validate_initial_train(initial_train)
        # SF-BL-11352: type-validate purge at construction, mirroring
        # the test_size / initial_train screens above — the constructor
        # previously stored and forwarded it unscreened, so a bool /
        # string / float purge only surfaced (if at all) inside
        # make_splits at run time, or was silently read as its int value.
        purge = _validate_purge(purge)
        # SF-BL-11385: type-validate rolling_train at construction,
        # mirroring the screens above — the constructor previously
        # stored and forwarded it unscreened, so a bool / string /
        # float window only surfaced (if at all) inside make_splits
        # at run time, or was silently coerced there.
        rolling_train = _validate_rolling_train(rolling_train)
        # SF-BL-11353: type-validate embargo at construction, mirroring
        # the screens above — the constructor previously stored and
        # forwarded it unscreened, so embargo=True / '5' / 2.5 all
        # constructed silently and were only caught (if at all) inside
        # make_splits at run time. See _validate_embargo.
        embargo = _validate_embargo(embargo)
        self.window, self.initial_train, self.test_size = window, initial_train, test_size
        self.step, self.purge, self.embargo = step, purge, embargo
        self.rolling_train, self.benchmarks_flag = rolling_train, benchmarks
        self.bootstrap_block, self.n_boot, self.seed = bootstrap_block, n_boot, seed

    def run(self, y, forecaster: Callable, *, X=None, horizons: Sequence[int] = (1,),
            n_variants_tried: int = 1) -> BacktestResult:
        # SF-BL-11130: re-validate at run — a post-construction mutation
        # of self.test_size to a bool/string/non-integer must raise the
        # same named ValueError here, not a bare error inside make_splits.
        self.test_size = _validate_test_size(self.test_size)
        # SF-BL-11158: re-validate at run — a post-construction
        # mutation of self.initial_train must raise the same named
        # ValueError here, not a silent empty result in make_splits.
        self.initial_train = _validate_initial_train(self.initial_train)
        # SF-BL-11352: re-validate at run — a post-construction
        # mutation of self.purge must raise the same named ValueError
        # here, not a bare error or silent coercion in make_splits.
        self.purge = _validate_purge(self.purge)
        # SF-BL-11385: re-validate at run — a post-construction
        # mutation of self.rolling_train must raise the same named
        # ValueError here, not a bare error or silent coercion in
        # make_splits.
        self.rolling_train = _validate_rolling_train(self.rolling_train)
        # SF-BL-11353: re-validate at run — a post-construction
        # mutation of self.embargo must raise the same named
        # ValueError here, not a bare error or silent coercion in
        # make_splits.
        self.embargo = _validate_embargo(self.embargo)
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
