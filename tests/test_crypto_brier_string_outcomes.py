"""
test_crypto_brier_string_outcomes.py — SF-BL-11390 regression tests.

Prereg: crypto.py brier raises ValueError naming y on any string or
object-dtype outcome element, with numeric 0/1 outcome panels
bit-identical. Kill: brier([0.3, 0.7], ['0', '1']) returns 0.09
instead of raising.

Run: python3 tests/test_crypto_brier_string_outcomes.py (exit 0 iff all pass)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "superforecaster"))
from crypto import brier  # noqa: E402

def _raises_y(fn, *args):
    try:
        fn(*args)
    except ValueError as e:
        assert "y" in str(e), f"does not name y: {e}"
        assert "string" in str(e).lower(), f"does not name string: {e}"
        return
    raise AssertionError(f"{fn.__name__}{args} did not raise ValueError")

STRING_Y = [
    ['0', '1'], ('0', '1'), np.array(['0', '1']),
    np.array([b'0', b'1']), np.array(['0', '1'], dtype=object),
    np.array(['0', 1], dtype=object), [['0'], ['1']], ['0', 1],
    ['yes', 'no'], [object(), 1],
]

def test_brier_rejects_string_outcomes():
    for y in STRING_Y: _raises_y(brier, [0.3, 0.7], y)

def test_brier_rejects_scalar_and_single_string_outcomes():
    _raises_y(brier, [0.5], '1'); _raises_y(brier, [0.5], ['1'])

def test_kill_repro():
    _raises_y(brier, [0.3, 0.7], ['0', '1'])

def test_numeric_outcomes_bit_identical():
    control = 0.09000000000000001
    assert brier([0.3, 0.7], [0, 1]) == control
    assert brier([0.3, 0.7], [0.0, 1.0]) == control
    assert brier([0.3, 0.7], (0, 1)) == control
    assert brier([0.3, 0.7], np.array([0, 1])) == control
    assert brier([0.3, 0.7], np.array([0, 1], dtype=object)) == control
    assert brier([0, 1], [0, 1]) == 0.0
    assert brier([0.0], [1.0]) == 1.0

def test_brier_prior_validation_unchanged():
    for p, y in [([], []), ([0.5, 0.6], [1]), ([1.5], [1]), ([0.5], [2])]:
        try: brier(p, y)
        except ValueError: continue
        raise AssertionError(f"brier({p},{y}) did not raise")

def test_brier_string_forecast_still_rejected():
    try: brier(['0.3', '0.7'], [0, 1])
    except ValueError as e:
        assert "p" in str(e); return
    raise AssertionError("string forecast panel did not raise")

try:
    import pandas as pd
    def test_pandas_string_outcomes():
        for y in [pd.Series(['0', '1']), pd.Series(['0', '1'], dtype="string"),
                  pd.Series(['0', '1'], dtype=object),
                  pd.Series(['0', '1'], dtype="category"),
                  pd.Index(['0', '1']), pd.Categorical(['0', '1'])]:
            _raises_y(brier, [0.3, 0.7], y)
    def test_pandas_numeric_outcomes_unchanged():
        assert brier([0.3, 0.7], pd.Series([0, 1])) == 0.09000000000000001
        assert brier([0.3, 0.7], pd.Series([0, 1], dtype=object)) == 0.09000000000000001
        assert brier([0.3, 0.7], pd.Index([0, 1])) == 0.09000000000000001
except ImportError:
    pass

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests: fn(); print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
