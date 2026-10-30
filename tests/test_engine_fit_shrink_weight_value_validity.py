"""SF-BL-11019 regression tests: fit_shrink_weight must reject unlawful
history VALUES — outcomes outside {0, 1} and probabilities outside
[0, 1] — instead of fitting a shrink weight on a corrupt history.

TASK criterion: fit_shrink_weight raises INVALID_INPUT on any outcome
not in {0, 1} or any p_model/p_prior outside [0, 1] while lawful
histories return bit-identical weights; kill if y=2 still fits w=1.0
or p_model=1.5 still fits any weight.

Pre-fix (pristine engine, repro_before.txt):
  fit_shrink_weight([2, 0], [0.6, 0.4], [0.5, 0.5])   -> w=1.0
  fit_shrink_weight([0.5, 0], [0.6, 0.4], [0.5, 0.5]) -> w=1.0
  fit_shrink_weight([1, 0], [1.5, 0.4], [0.5, 0.5])   -> w=0.55

Run: python3 test_engine_fit_shrink_weight_value_validity.py
No pytest dependency: test_* functions + __main__ runner.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.expanduser("~/workspace/superforecaster"), "superforecaster"))

import numpy as np  # noqa: E402

from engine import fit_shrink_weight as fsw  # noqa: E402


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


def test_task_repros_raise():
    # The three exact TASK.md live repros — pre-fix RETURNED 1.0/1.0/0.55.
    assert_named_raise(lambda: fsw([2, 0], [0.6, 0.4], [0.5, 0.5]),
                       "y[0]", "2.0", "not a lawful outcome")
    assert_named_raise(lambda: fsw([0.5, 0], [0.6, 0.4], [0.5, 0.5]),
                       "y[0]", "0.5", "not a lawful outcome")
    assert_named_raise(lambda: fsw([1, 0], [1.5, 0.4], [0.5, 0.5]),
                       "p_model[0]", "1.5", "outside [0, 1]")


def test_kill_criterion_no_numeric_w():
    for fn in (lambda: fsw([2, 0], [0.6, 0.4], [0.5, 0.5]),
               lambda: fsw([1, 0], [1.5, 0.4], [0.5, 0.5]),
               lambda: fsw([1, 0], [0.6, 0.4], [1.5, 0.5])):
        try:
            r = fn()
        except ValueError:
            continue
        raise AssertionError(f"KILL: numeric w {r!r} returned from corrupt history")


def test_all_invalid_value_permutations_raise():
    assert_named_raise(lambda: fsw([-1, 0], [0.6, 0.4], [0.5, 0.5]), "y[0]")
    assert_named_raise(lambda: fsw([1, 3], [0.6, 0.4], [0.5, 0.5]), "y[1]", "3.0")
    assert_named_raise(lambda: fsw([0.999, 0], [0.6, 0.4], [0.5, 0.5]), "y[0]")
    assert_named_raise(lambda: fsw([2], [0.6], [0.5]), "y[0]")
    assert_named_raise(lambda: fsw([1, 0], [-0.2, 0.4], [0.5, 0.5]), "p_model[0]")
    assert_named_raise(lambda: fsw([1, 0], [0.6, 1.0000001], [0.5, 0.5]), "p_model[1]")
    assert_named_raise(lambda: fsw([1, 0], [0.6, 0.4], [1.5, 0.5]), "p_prior[0]")
    assert_named_raise(lambda: fsw([1, 0], [0.6, 0.4], [0.5, -0.1]), "p_prior[1]")
    assert_named_raise(lambda: fsw(np.array([2.0, 0.0]), np.array([0.6, 0.4]),
                                   np.array([0.5, 0.5])), "y[0]")
    assert_named_raise(lambda: fsw(np.array([1.0, 0.0]), np.array([1.5, 0.4]),
                                   np.array([0.5, 0.5])), "p_model[0]")


def test_lawful_histories_bit_identical():
    # Values frozen from the pristine engine (repro.py GOOD section).
    assert fsw([1, 0], [1.0, 0.0], [0.0, 1.0]) == 1.0   # boundary probs lawful
    assert fsw([0, 1], [0.0, 1.0], [1.0, 0.0]) == 1.0
    assert fsw([1, 0], [0.9, 0.2], [0.5, 0.5]) == 1.0
    assert fsw([1.0, 0.0, 1.0, 0.0], [0.9, 0.1, 0.9, 0.1], [0.5] * 4) == 1.0
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
