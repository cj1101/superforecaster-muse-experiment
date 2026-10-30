"""
test_cdf_module_disciplined_spot_type.py — SF-BL-11310 verification.

Pre-registered done/kill criterion (TASK.md):
  "disciplined_percentiles raises ValueError naming spot on any
   bool or string spot, with numeric-spot percentile dicts
   bit-identical; kill if spot=True still returns the spot=1.0 dict
   or spot='100' still raises a bare UFuncTypeError."

Bug: disciplined_percentiles never type-validated spot. float(spot)
inside climatology_percentiles read True as 1.0 (spot=True returned
a dict centred on ~1.0), and spot='100' died in the function's own
`... / spot` division with a bare UFuncTypeError. Distinct from
SF-BL-10475 (this function's spot VALUE domain: 0/neg/NaN/inf),
SF-BL-11268 (this function's probs type) and SF-BL-10393
(naive_normal_percentiles' spot, a different function).

Fix: reject-only spot type screen at disciplined_percentiles entry
in both live copies; numeric spots fall through untouched, so their
dicts stay bit-identical (expected bits below are the pre-fix
outputs, verified vs a verbatim pre-fix reference in the task's
verify_fix.py, 520/520 across both copies).

Run: python3 tests/test_cdf_module_disciplined_spot_type.py
     or pytest tests/test_cdf_module_disciplined_spot_type.py
"""
import os, struct, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metaculus_bot"))
import numpy as np
import pytest
import cdf_module as cm

RNG = np.random.default_rng(11310)
TH = RNG.normal(0.001, 0.02, 60)
TD = RNG.normal(0.0005, 0.012, 120)
RD = RNG.normal(0.0008, 0.015, 30)
BITS_100 = "67bc7c8689de5740780882f2c69e5840bb8bda2755055940fd587adf2d7459401a687d4b8a095a40"
BITS_250 = "80eb1be82bd66d40978a22af78c66e40aa2ed171aa466f403cef585739d16f4010612e6ff6457040"
BITS_50 = "67bc7c8689de4740780882f2c69e4840bb8bda2755054940fd587adf2d7449401a687d4b8a094a40"

def _bits(d): return "".join(struct.pack("<d", d[p]).hex() for p in sorted(d))
def _call(spot): return cm.disciplined_percentiles(spot, TH, TD, RD)
def _raises_named(spot):
    with pytest.raises(ValueError) as ei:
        _call(spot)
    assert type(ei.value) is ValueError
    assert "spot" in str(ei.value), str(ei.value)

def test_kill_bool_spot_does_not_return_spot1_dict():
    _raises_named(True)

def test_kill_string_spot_not_bare_ufunc():
    _raises_named("100")

def test_all_bad_spot_types_raise_named():
    for s in [False, np.bool_(True), "100.0", "", b"100", np.str_("100"),
              None, {"s": 100}, [100], (100,), object(),
              np.array([100.0, 200.0]), np.array(True), np.array("100"),
              np.array("100", dtype=object), np.array(None, dtype=object)]:
        _raises_named(s)

def test_bad_spot_named_even_with_bad_probs():
    with pytest.raises(ValueError) as ei:
        cm.disciplined_percentiles(True, TH, TD, RD, probs=["0.5"])
    assert "spot" in str(ei.value), str(ei.value)

def test_bad_probs_still_named_with_nan_spot():
    with pytest.raises(ValueError) as ei:
        cm.disciplined_percentiles(float("nan"), TH, TD, RD, probs=["0.5"])
    assert "probs" in str(ei.value), str(ei.value)

def test_spot_value_domain_preserved():
    for s in [0, 0.0, -100, float("nan"), float("inf"), float("-inf")]:
        _raises_named(s)

def test_numeric_spot_bit_identical():
    assert _bits(_call(100.0)) == BITS_100
    assert _bits(_call(100)) == BITS_100
    assert _bits(_call(np.float64(100.0))) == BITS_100
    assert _bits(_call(np.array(100.0))) == BITS_100
    assert _bits(_call(250.0)) == BITS_250
    assert _bits(_call(50.0)) == BITS_50

if __name__ == "__main__":
    test_kill_bool_spot_does_not_return_spot1_dict()
    test_kill_string_spot_not_bare_ufunc()
    test_all_bad_spot_types_raise_named()
    test_bad_spot_named_even_with_bad_probs()
    test_bad_probs_still_named_with_nan_spot()
    test_spot_value_domain_preserved()
    test_numeric_spot_bit_identical()
    print("ALL TESTS PASSED (7 test functions)")
