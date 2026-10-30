"""
test_crypto_log_score_bool_panel.py — SF-BL-11160 regression tests.

Prereg: crypto.log_score raises ValueError naming the forecast panel
on any bool element, with float panels bit-identical.
Kill: log_score([True, False], [1, 0]) still returns a numeric score
near 0 (pre-fix np form returned -9.999778782803785e-13).

Run: python3 tests/test_crypto_log_score_bool_panel.py (exit 0 iff all pass)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "superforecaster"))
from crypto import log_score  # noqa: E402

def _raises(fn, *args, **kw):
    try:
        fn(*args, **kw)
    except ValueError as e:
        assert "p" in str(e), f"does not name p: {e}"
        assert "bool" in str(e).lower(), f"does not name bool: {e}"
        return
    raise AssertionError(f"{fn.__name__}{args} did not raise ValueError")

Y = np.array([1, 0])
BOOL_PANELS = [
    (np.array([True, False]), Y),
    ([True, False], Y),
    ((True, False), Y),
    ([[True], [False]], Y),
    ([0.3, True], np.array([0, 1])),
    (np.array([0.3, True], dtype=object), np.array([0, 1])),
    ([np.bool_(True), np.bool_(False)], Y),
    ([True], np.array([1])),
    ([False], np.array([0])),
]

def test_log_score_rejects_bool_panels():
    for p, y in BOOL_PANELS: _raises(log_score, p, y)

def test_log_score_rejects_bool_scalars():
    _raises(log_score, True, np.array([1]))
    _raises(log_score, np.bool_(True), np.array([1]))
    _raises(log_score, np.array(True), np.array([1]))

def test_log_score_rejects_pandas_bool_series():
    import pandas as pd
    _raises(log_score, pd.Series([True, False]), Y)
    _raises(log_score, pd.Series([True, False], dtype="boolean"), Y)
    _raises(log_score, pd.Series([0.3, True], dtype=object), np.array([0, 1]))

def test_numeric_panels_bit_identical():
    assert log_score(np.array([0.3, 0.7]), np.array([0, 1])) == -0.35667494393873245
    assert log_score(np.array([1.0, 0.0]), np.array([1, 0])) == -9.999778782803785e-13
    assert log_score(np.array([0.5, 0.5]), np.array([0, 1])) == -0.6931471805599453
    assert log_score(np.array([1.0, 0.0]), np.array([1, 0]), eps=1e-6) == -1.000000500029089e-06

def test_int_panels_are_not_bools():
    # ints subclass-check trap: bool subclasses int, not the reverse —
    # lawful 0/1 int panels must still score exactly as pre-fix.
    assert log_score(np.array([1, 0]), np.array([1, 0])) == -9.999778782803785e-13
    assert log_score([1, 0], np.array([1, 0])) == -9.999778782803785e-13

def test_kill_repro_no_longer_scores():
    try:
        v = log_score(np.array([True, False]), np.array([1, 0]))
    except ValueError:
        return
    raise AssertionError(f"kill criterion triggered: returned {v!r}")

if __name__ == "__main__":
    tests=[(n,f) for n,f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name,fn in tests: fn(); print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
