"""
test_crypto_log_score_eps_types.py — SF-BL-11162 regression tests.

Prereg: crypto.log_score raises ValueError naming eps unless eps is a
non-bool real number, with default-eps scores bit-identical.
Kill: eps=True still returns NaN, or eps='1e-12' still raises an
unnamed TypeError.

Run: python3 tests/test_crypto_log_score_eps_types.py (exit 0 iff all pass)
"""
import os, sys
from decimal import Decimal
from fractions import Fraction
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "superforecaster"))
from crypto import log_score  # noqa: E402

P = np.array([0.3, 0.7]); Y = np.array([0, 1])
DEFAULT = -0.35667494393873245

def _prefix(eps):
    """Verbatim pre-fix computation for a lawful eps."""
    p = np.clip(P, eps, 1 - eps)
    return float(np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p)))

def _raises(eps):
    try:
        log_score(P, Y, eps=eps)
    except ValueError as e:
        assert "eps" in str(e), f"does not name eps: {e}"
        return
    except Exception as e:  # noqa: BLE001 - any non-ValueError is the bug
        raise AssertionError(f"eps={eps!r} raised {type(e).__name__}, not ValueError: {e}")
    raise AssertionError(f"eps={eps!r} did not raise ValueError")

BAD_EPS = [
    True, False, np.bool_(True), np.array(True),
    "1e-12", "", "abc", b"1e-12", None,
    [1e-12], (1e-12,), {"eps": 1e-12}, {1e-12},
    1e-12 + 0j, complex(0, 1), object(),
    np.array([1e-12]), np.array(["1e-12"]), pd.Series([1e-12]),
    float("nan"), float("inf"), float("-inf"), np.float64("nan"),
]

def test_bad_eps_types_raise_named_valueerror():
    for eps in BAD_EPS: _raises(eps)

def test_kill_repros():
    _raises(True)        # previously returned NaN
    _raises("1e-12")     # previously bare TypeError
    _raises(None)        # previously bare TypeError

def test_default_and_explicit_default_bit_identical():
    assert log_score(P, Y) == DEFAULT
    assert log_score(P, Y, eps=1e-12) == DEFAULT
    assert log_score([0.3, 0.7], Y) == DEFAULT

def test_valid_real_eps_bit_identical():
    for eps in [1e-12, 1e-9, 1e-6, 1e-4, 0.01, 0.1, 0.25,
                np.float64(1e-9), np.float32(1e-6), np.array(1e-12), 0]:
        assert log_score(P, Y, eps=eps) == _prefix(float(eps)), eps

def test_valid_real_eps_accepted_types():
    # Real-number types beyond float keep their float value.
    assert log_score(P, Y, eps=Decimal("1e-12")) == DEFAULT
    assert log_score(P, Y, eps=Fraction(1, 10**12)) == DEFAULT

def test_sibling_screens_unchanged():
    # SF-BL-11160 bool panel screen still fires (default eps).
    try: log_score([True, False], Y)
    except ValueError as e: assert "p" in str(e)
    else: raise AssertionError("bool panel did not raise")

if __name__ == "__main__":
    tests=[(n,f) for n,f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name,fn in tests: fn(); print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
