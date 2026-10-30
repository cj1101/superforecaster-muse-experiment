"""
test_crypto_brier_validation.py — SF-BL-10065 regression tests.

Pre-registered done criteria (TASK.md): crypto.py brier must raise a named
ValueError on length mismatch, empty panel, non-finite or out-of-[0,1]
forecasts, or non-binary outcomes before scoring, with clean panels
byte-identical.

Pre-fix reproduction (both crypto.py copies): brier([NaN, 0.7], [0, 1])
returned NaN, brier([], []) returned NaN, brier([0.5, 0.6], [1]) returned
0.2050 by broadcasting, brier([1.5], [1]) returned 0.25, brier([0.5], [2])
returned 2.25; clean brier([0.3, 0.7], [0, 1]) = 0.09000000000000001.

Run:  python3 tests/test_crypto_brier_validation.py   (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as the other fleet tests.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "superforecaster"))

from crypto import brier  # noqa: E402


def _assert_raises_valueerror(p, y, keyword):
    try:
        brier(np.asarray(p, dtype=float), np.asarray(y, dtype=float))
    except ValueError as e:
        assert "brier" in str(e), f"message does not name brier: {e}"
        assert keyword in str(e), f"message does not name {keyword!r}: {e}"
        return
    raise AssertionError(f"brier({p}, {y}) did not raise ValueError")


def test_brier_rejects_nonfinite_forecasts():
    for bad in (np.nan, np.inf, -np.inf):
        _assert_raises_valueerror([bad, 0.7], [0, 1], "finite")


def test_brier_rejects_empty_panel():
    _assert_raises_valueerror([], [], "empty")


def test_brier_rejects_length_mismatch():
    _assert_raises_valueerror([0.5, 0.6], [1], "length")
    _assert_raises_valueerror([0.5], [0, 1], "length")


def test_brier_rejects_out_of_range_forecasts():
    for bad in (1.5, 2.0, -0.5, 1.0000001, -1e-12):
        _assert_raises_valueerror([bad], [1], "[0, 1]")
    _assert_raises_valueerror([0.5, 1.5], [0, 1], "[0, 1]")


def test_brier_rejects_nonbinary_outcomes():
    for bad in (2, 0.5, -1, np.nan, np.inf):
        _assert_raises_valueerror([0.5], [bad], "binary")


def test_brier_clean_panels_unchanged():
    assert brier(np.array([0.3, 0.7]), np.array([0.0, 1.0])) == 0.09000000000000001
    assert brier(np.array([0.0]), np.array([0.0])) == 0.0
    assert brier(np.array([1.0]), np.array([1.0])) == 0.0
    assert brier(np.array([0.0]), np.array([1.0])) == 1.0
    assert brier(np.array([0.25, 0.75, 0.5]), np.array([0.0, 1.0, 1.0])) == 0.125


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
