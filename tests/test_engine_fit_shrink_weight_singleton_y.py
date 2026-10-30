"""SF-BL-9794 regression tests: fit_shrink_weight must not broadcast a
singleton outcome across the whole history.

TASK criterion: fit_shrink_weight raises ValueError unless
len(y)==len(p_model)==len(p_prior), with equal-length fits
bit-identical; kill if the 1-vs-4 repro still returns a fitted w
instead of raising.

Pre-fix (pristine HEAD engine, repro_before.txt):
  fit_shrink_weight([1.0],[0.9,0.1,0.9,0.1],[0.5,0.5,0.5,0.5]) -> 0.0
    (the single outcome broadcast over all four pairs)

Run: python3 test_engine_fit_shrink_weight_singleton_y.py
No pytest dependency: test_* functions + __main__ runner.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.expanduser("~/workspace/superforecaster"), "superforecaster"))

import numpy as np  # noqa: E402

from engine import fit_shrink_weight as fsw  # noqa: E402

M4, P4 = [0.9, 0.1, 0.9, 0.1], [0.5, 0.5, 0.5, 0.5]


def assert_named_raise(fn, *needles):
    try:
        r = fn()
    except ValueError as e:
        assert type(e) is ValueError, type(e)
        assert "INVALID_INPUT" in str(e), f"missing INVALID_INPUT flag: {e}"
        assert "length mismatch" in str(e), f"missing 'length mismatch': {e}"
        for n in needles:
            assert n in str(e), f"missing {n!r} in error: {e}"
        return
    raise AssertionError(f"expected ValueError, got return {r!r}")


def test_task_repro_singleton_y_raises():
    # The exact TASK.md repro — pre-fix RETURNED w=0.0.
    assert_named_raise(lambda: fsw([1.0], M4, P4),
                       "y len=1", "p_model len=4", "p_prior len=4")
    assert_named_raise(lambda: fsw([0.0], M4, P4), "y len=1")


def test_kill_criterion_no_numeric_w():
    for fn in (lambda: fsw([1.0], M4, P4),
               lambda: fsw([1.0, 0.0, 1.0, 0.0], [0.9], P4),
               lambda: fsw([1.0, 0.0, 1.0, 0.0], M4, [0.5]),
               lambda: fsw([1.0], [0.9], P4)):
        try:
            r = fn()
        except ValueError:
            continue
        raise AssertionError(f"KILL: numeric w {r!r} returned from broadcast history")


def test_all_mismatch_permutations_raise():
    assert_named_raise(lambda: fsw([1.0, 0.0], M4, P4), "y len=2")
    assert_named_raise(lambda: fsw([1.0, 0.0, 1.0, 0.0], M4[:3], P4), "p_model len=3")
    assert_named_raise(lambda: fsw([1.0, 0.0, 1.0, 0.0], M4, P4[:3]), "p_prior len=3")
    assert_named_raise(lambda: fsw([1.0, 0.0, 1.0], [0.9, 0.1], [0.5, 0.5]))
    assert_named_raise(lambda: fsw(np.array([1.0]), np.array(M4), np.array(P4)), "y len=1")


def test_equal_length_fits_bit_identical():
    # Values frozen from the pristine engine (repro.py GOOD section).
    assert fsw([1.0], [0.9], [0.5]) == 1.0          # matched len-1 is lawful
    assert fsw([0.0], [0.2], [0.5]) == 1.0
    assert fsw([1, 0], [0.9, 0.2], [0.5, 0.5]) == 1.0
    assert fsw([1.0, 0.0, 1.0, 0.0], M4, P4) == 1.0  # task arrays, matched
    assert fsw([0, 1, 0, 1], [0.6, 0.7, 0.4, 0.8], [0.4, 0.6, 0.3, 0.7]) == 0.0
    assert fsw([1, 1, 0, 0, 1], [0.8, 0.6, 0.3, 0.45, 0.9], [0.5] * 5) == 1.0
    assert fsw([0, 1], [0.2, 0.8], [0.5, 0.5], grid=[0.0, 0.5, 1.0]) == 1.0
    w = fsw([0, 1, 1, 0], [0.3, 0.6, 0.7, 0.4], [0.5] * 4)
    assert isinstance(w, float) and 0.0 <= w <= 1.0


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for fn in ALL_TESTS:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(ALL_TESTS)} unit tests passed.")
