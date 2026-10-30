"""
test_markets_brier_validation.py — SF-BL-9886 regression tests.

Pre-registered done criteria (TASK.md): markets.py brier must raise on
length mismatch and on empty input, and must mask or reject non-finite
pairs under the fleet's stated convention (backtest._check: mask
non-finite pairs, raise on shape mismatch).

Pre-fix reproduction (both markets.py copies): brier(y=[1, 0], p=[0.5])
returned 0.25 by broadcasting two outcomes against one forecast,
brier([], []) returned NaN, and a NaN outcome returned NaN instead of
being masked; clean brier(y=[0, 1], p=[0.3, 0.7]) = 0.09000000000000001.

Note the argument order: markets.brier is (y, p), unlike crypto.brier.

Run:  python3 tests/test_markets_brier_validation.py   (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as the other fleet tests.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "superforecaster"))

from markets import brier  # noqa: E402


def _assert_raises_valueerror(y, p, keyword):
    try:
        brier(np.asarray(y, dtype=float), np.asarray(p, dtype=float))
    except ValueError as e:
        assert "brier" in str(e), f"message does not name brier: {e}"
        assert keyword in str(e), f"message does not name {keyword!r}: {e}"
        return
    raise AssertionError(f"brier(y={y}, p={p}) did not raise ValueError")


def test_brier_rejects_length_mismatch():
    _assert_raises_valueerror([1, 0], [0.5], "shapes differ")   # TASK repro
    _assert_raises_valueerror([1], [0.5, 0.6], "shapes differ")
    _assert_raises_valueerror([[1, 0]], [0.5, 0.5], "shapes differ")


def test_brier_rejects_empty_input():
    _assert_raises_valueerror([], [], "empty")


def test_brier_rejects_all_nonfinite_after_masking():
    _assert_raises_valueerror([np.nan, np.nan], [0.5, 0.5], "finite")
    _assert_raises_valueerror([1, 0], [np.inf, -np.inf], "finite")


def test_brier_masks_nonfinite_pairs():
    # backtest._check convention: drop any pair non-finite in y or p,
    # score the survivors.
    assert brier(np.array([np.nan, 1.0]), np.array([0.5, 0.5])) == 0.25
    assert brier(np.array([1.0, 0.0]), np.array([np.nan, 0.5])) == 0.25
    assert brier(np.array([1.0, 0.0, 1.0]),
                 np.array([0.5, np.inf, 0.9])) == 0.13
    assert brier(np.array([np.inf, 0.0]), np.array([0.5, 0.25])) == 0.0625


def test_brier_clean_panels_unchanged():
    assert brier(np.array([0.0, 1.0]), np.array([0.3, 0.7])) == 0.09000000000000001
    assert brier(np.array([1.0, 0.0]), np.array([0.5, 0.5])) == 0.25
    assert brier(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 0.0
    assert brier(np.array([1.0]), np.array([0.0])) == 1.0


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
