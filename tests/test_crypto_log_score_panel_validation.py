"""
test_crypto_log_score_panel_validation.py — SF-BL-11206 regression tests.

Pre-registered done criteria (TASK.md): crypto.py log_score must raise a
ValueError naming the defect on an empty panel, a length mismatch, any
non-finite or out-of-[0, 1] forecast, or any non-binary outcome, with
lawful-panel scores bit-identical.
Kill: log_score([1.5], [1]) still returns ~-1e-12, or the mismatched
panel log_score([0.5, 0.6], [1]) still returns -0.6020.

Pre-fix reproduction (both crypto.py copies, numpy form):
log_score([1.5], [1]) returned -9.999778782803785e-13, log_score([], [])
returned NaN, log_score([NaN], [1]) returned NaN,
log_score([0.5, 0.6], [1]) returned -0.601986402162968 by broadcasting,
log_score([0.5], [2]) returned -0.6931471805599453; clean
log_score([0.3, 0.7], [0, 1]) = -0.35667494393873245.
Plain-list panels are also covered: the TASK repros are written as
lists, and pre-fix list/list panels died with a bare TypeError on 1-y.

Run:  python3 tests/test_crypto_log_score_panel_validation.py
No pytest dependency: test_* functions with assert + __main__ runner.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "superforecaster"))

from crypto import log_score  # noqa: E402


def _assert_raises(p, y, keyword):
    for pp, yy in ((np.asarray(p, dtype=float), np.asarray(y, dtype=float)), (p, y)):
        try:
            log_score(pp, yy)
        except ValueError as e:
            assert "log_score" in str(e), f"message does not name log_score: {e}"
            assert keyword in str(e), f"message does not name {keyword!r}: {e}"
            continue
        raise AssertionError(f"log_score({p}, {y}) did not raise ValueError")


def test_log_score_rejects_empty_panel():
    _assert_raises([], [], "empty")


def test_log_score_rejects_length_mismatch():
    _assert_raises([0.5, 0.6], [1], "length")
    _assert_raises([0.5], [0, 1], "length")
    _assert_raises([], [1], "length")


def test_log_score_rejects_nonfinite_forecasts():
    for bad in (np.nan, np.inf, -np.inf):
        _assert_raises([bad, 0.7], [0, 1], "finite")


def test_log_score_rejects_out_of_range_forecasts():
    for bad in (1.5, 2.0, -0.5, 1.0000001, -1e-12):
        _assert_raises([bad], [1], "[0, 1]")
    _assert_raises([0.5, 1.5], [0, 1], "[0, 1]")


def test_log_score_rejects_nonbinary_outcomes():
    for bad in (2, 0.5, -1, np.nan, np.inf):
        _assert_raises([0.5], [bad], "binary")


def test_log_score_clean_panels_unchanged():
    assert log_score(np.array([0.3, 0.7]), np.array([0, 1])) == -0.35667494393873245
    assert log_score(np.array([1.0, 0.0]), np.array([1, 0])) == -9.999778782803785e-13
    assert log_score(np.array([0.5, 0.5]), np.array([0, 1])) == -0.6931471805599453
    assert log_score(np.array([1.0, 0.0]), np.array([1, 0]), eps=1e-6) == -1.000000500029089e-06
    assert log_score([0.3, 0.7], [0, 1]) == -0.35667494393873245


def test_kill_repros_no_longer_score():
    for p, y in (([1.5], [1]), ([0.5, 0.6], [1])):
        try:
            v = log_score(np.asarray(p, dtype=float), np.asarray(y, dtype=float))
        except ValueError:
            continue
        raise AssertionError(f"kill criterion triggered: log_score({p}, {y}) returned {v!r}")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
