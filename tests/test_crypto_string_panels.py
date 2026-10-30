"""
test_crypto_string_panels.py — SF-BL-11135 regression tests.

Prereg: crypto.py brier and auc raise ValueError naming p on any
string-dtype forecast panel, with numeric panels bit-identical.
Kill: brier(['0.3','0.7'],[0,1]) returns 0.0900 or auc(...) returns 1.0.

Run: python3 tests/test_crypto_string_panels.py (exit 0 iff all pass)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "superforecaster"))
from crypto import brier, auc  # noqa: E402

def _raises(fn, *args):
    try:
        fn(*args)
    except ValueError as e:
        assert "p" in str(e), f"does not name p: {e}"
        assert "string" in str(e).lower(), f"does not name string: {e}"
        return
    raise AssertionError(f"{fn.__name__}{args} did not raise ValueError")

STRING_PANELS = [
    ['0.3','0.7'], ('0.3','0.7'), np.array(['0.3','0.7']),
    np.array([b'0.3',b'0.7']), np.array(['0.3',0.7], dtype=object),
    [['0.3'],['0.7']], [0.3,'0.7'], ['abc','0.7'],
]

def test_brier_rejects_string_panels():
    for p in STRING_PANELS: _raises(brier, p, [0,1])

def test_auc_rejects_string_panels():
    for p in STRING_PANELS: _raises(auc, p, [0,1])

def test_brier_rejects_scalar_and_single_strings():
    _raises(brier, '0.3', [0]); _raises(brier, ['0.5'], [1])

def test_auc_rejects_scalar_and_single_strings():
    _raises(auc, '0.3', [0]); _raises(auc, ['0.5'], [1])

def test_numeric_panels_bit_identical():
    assert brier([0.3,0.7],[0,1]) == 0.09000000000000001
    assert auc([0.3,0.7],[0,1]) == 1.0
    assert brier([0,1],[0,1]) == 0.0
    assert auc([0.5,0.5],[0,1]) == 0.5
    assert auc(np.array([0.1,0.4,0.35,0.8]), np.array([0,0,1,1])) == 0.75
    assert np.isnan(auc([0.3,0.7],[1,1]))

def test_brier_prior_validation_unchanged():
    for p,y in [([],[]), ([0.5,0.6],[1]), ([1.5],[1]), ([0.5],[2])]:
        try: brier(p,y)
        except ValueError: continue
        raise AssertionError(f"brier({p},{y}) did not raise")

if __name__ == "__main__":
    tests=[(n,f) for n,f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name,fn in tests: fn(); print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
