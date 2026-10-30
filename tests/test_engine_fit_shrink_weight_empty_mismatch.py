"""SF-BL-8869 regression tests: fit_shrink_weight must never return a
numeric w from an empty or length-mismatched (broadcast) history.

TASK criterion: Pass iff empty or length-mismatched history raises or
returns None with the shrink layer inactive, never a numeric w; kill
if 0.0 can still be returned from an empty or broadcast history.
This fix raises ValueError carrying INVALID_INPUT (and 'length
mismatch' for the unequal-length case), matching the module's
SF-BL-9510 / SF-BL-8868 guard conventions.

Pre-fix (verified live 2026-10-22, repro_before.txt):
  fit_shrink_weight([], [], []) -> 0.0 (NaN Brier never beats inf,
    initial best_w stands)
  fit_shrink_weight([0,1,0,1], [0.6,0.7,0.4,0.8], [0.5]) -> 1.0
    (length-1 p_prior broadcasts across the 4-row history)
  fit_shrink_weight(..., grid=[]) -> 0.0 (same standing-best_w defect)

Run: python3 test_engine_fit_shrink_weight_empty_mismatch.py
No pytest dependency: test_* functions + __main__ runner.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.expanduser("~/workspace/superforecaster"), "superforecaster"))

import numpy as np  # noqa: E402

from engine import fit_shrink_weight as fsw  # noqa: E402

Y4, M4, P4 = [0, 1, 0, 1], [0.6, 0.7, 0.4, 0.8], [0.5, 0.5, 0.5, 0.5]


def assert_named_raise(fn, *needles):
    try:
        r = fn()
    except ValueError as e:
        assert type(e) is ValueError, type(e)
        assert "INVALID_INPUT" in str(e), f"missing INVALID_INPUT flag: {e}"
        for n in needles:
            assert n in str(e), f"missing {n!r} in error: {e}"
        return
    raise AssertionError(f"expected ValueError, got return {r!r}")


def test_empty_history_raises():
    assert_named_raise(lambda: fsw([], [], []), "empty")
    assert_named_raise(lambda: fsw(np.array([]), np.array([]), np.array([])), "empty")
    assert_named_raise(lambda: fsw([], M4, P4))
    assert_named_raise(lambda: fsw(Y4, [], P4))
    assert_named_raise(lambda: fsw(Y4, M4, []))


def test_broadcast_length1_raises():
    # The TASK repro and its two siblings.
    assert_named_raise(lambda: fsw(Y4, M4, [0.5]), "length mismatch")
    assert_named_raise(lambda: fsw(Y4, [0.6], P4), "length mismatch")
    assert_named_raise(lambda: fsw([1], M4, P4), "length mismatch")


def test_unequal_lengths_raise_named_not_bare_numpy():
    assert_named_raise(lambda: fsw([0, 1, 0], [0.6, 0.7], [0.5, 0.5, 0.5]),
                       "length mismatch", "y len=3", "p_model len=2")
    assert_named_raise(lambda: fsw(Y4, M4[:3], P4), "length mismatch")
    assert_named_raise(lambda: fsw(Y4[:2], M4, P4), "length mismatch")


def test_scalar_and_degenerate_grid_raise():
    assert_named_raise(lambda: fsw(1, 0.8, 0.5))
    assert_named_raise(lambda: fsw(1, [0.6, 0.7], [0.5, 0.5]))
    assert_named_raise(lambda: fsw([0, 1], [0.2, 0.8], [0.5, 0.5], grid=[]), "grid")
    assert_named_raise(lambda: fsw([0, 1], [0.2, 0.8], [0.5, 0.5], grid=0.5))
    assert_named_raise(lambda: fsw([0, 1], [0.2, 0.8], [0.5, 0.5],
                                   grid=[0.0, float("nan")]))


def test_kill_criterion_no_numeric_w_from_empty_or_broadcast():
    bad_calls = [
        lambda: fsw([], [], []),
        lambda: fsw(Y4, M4, [0.5]),
        lambda: fsw(Y4, [0.6], P4),
        lambda: fsw([1], M4, P4),
        lambda: fsw(Y4, M4, [], ),
    ]
    for fn in bad_calls:
        try:
            r = fn()
        except ValueError:
            continue
        raise AssertionError(f"KILL: numeric w {r!r} returned from bad history")


def test_valid_matched_histories_bit_identical():
    # Values frozen from the pristine engine (repro.py GOOD section).
    assert fsw(Y4, [0.9, 0.9, 0.9, 0.9], [0, 1, 0, 1]) == 0.0  # earned 0.0
    assert fsw([0, 1], [0.0, 1.0], [0.5, 0.5]) == 1.0
    assert fsw(Y4, M4, P4) == 1.0
    assert fsw(np.array(Y4), np.array(M4), np.array(P4)) == 1.0
    assert fsw([0, 1], [0.2, 0.8], [0.5, 0.5], grid=[0.0, 0.5, 1.0]) == 1.0
    # A matched length-1 history is a genuine single observation,
    # NOT a broadcast: it must still fit.
    assert fsw([1], [0.8], [0.5]) == 1.0
    w = fsw([0, 1, 1, 0], [0.3, 0.6, 0.7, 0.4], [0.5, 0.5, 0.5, 0.5])
    assert isinstance(w, float) and 0.0 <= w <= 1.0


def test_prior_guards_preserved():
    # SF-BL-9510: NaN/inf elements still raise INVALID_INPUT.
    assert_named_raise(lambda: fsw([0, 1], [float("nan"), 0.5], [0.5, 0.5]))
    # SF-BL-11054: string arrays still raise INVALID_INPUT.
    assert_named_raise(lambda: fsw(["0", "1"], [0.2, 0.8], [0.5, 0.5]))


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for fn in ALL_TESTS:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(ALL_TESTS)} unit tests passed.")
