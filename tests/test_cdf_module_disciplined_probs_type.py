"""
test_cdf_module_disciplined_probs_type.py — SF-BL-11268 verification.

Pre-registered done/kill criterion (TASK.md):
  "disciplined_percentiles raises ValueError naming probs on any
   bool or string-dtype probs element, with numeric probs dicts
   bit-identical; kill if probs=['0.05','0.5','0.95'] still returns
   the control dict or probs=[True, 0.5, 0.9] still returns a dict
   keyed 1.0."

Bug: disciplined_percentiles opened with
np.asarray(probs, dtype=float), which coerces with no dtype screen:
numeric strings parsed silently (bit-identical to the numeric
control) and True was read as 1.0 (dict keyed 1.0). Distinct from
SF-BL-10316 (the VALUE domain — unsorted probs — in this function)
and SF-BL-11243/11244 (probs type in climatology_percentiles /
naive_normal_percentiles, different functions).

Fix: probs is type-screened at disciplined_percentiles entry,
before any coercion. In this repo copy the screen is the shared
SF-BL-11243 helper (landed concurrently, caller-named for this
function; the scaffold copy carries the dedicated SF-BL-11268
screen) — either way a bool/string probs element raises ValueError
naming probs, and numeric probs fall through bit-identical
(expected bit patterns below match the pre-fix outputs, verified
against a verbatim pre-fix reference in SF-BL-11268 verify_fix.py).

Run: python3 tests/test_cdf_module_disciplined_probs_type.py
     or pytest tests/test_cdf_module_disciplined_probs_type.py
"""
import os, struct, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metaculus_bot"))
import numpy as np
import pytest
import cdf_module as cm

RNG = np.random.default_rng(11268)
TH = RNG.normal(0.001, 0.02, 60)
TD = RNG.normal(0.0005, 0.012, 120)
RD = RNG.normal(0.0008, 0.015, 30)
EXPECTED_BITS = {
    "default": "285e22a4c91d584021d643d6db97584048107c5bc70a59408df3cebd89685940da28221154e35940",
    "custom": "19eb6e86d9c857404c81553e9a50584048107c5bc70a5940c7d3563c47a35940255346ecc8ff5940",
    "single": "48107c5bc70a5940",
    "pair": "db94f97614da58406c50b845f4ac5940",
    "near": "41898196a8ae574048107c5bc70a5940746876fe6f0b5a40",
    "unsorted": "fbd0fcb33065594094be8fb330655940",
}
PANELS = {"default": None, "custom": (0.01, 0.1, 0.5, 0.9, 0.99),
          "single": (0.5,), "pair": [0.25, 0.75],
          "near": (1e-9, 0.5, 1 - 1e-9), "unsorted": (0.95, 0.05)}

def _bits(d): return "".join(struct.pack("<d", d[p]).hex() for p in sorted(d))
_DEFAULT = object()
def _call(probs=_DEFAULT):
    if probs is _DEFAULT:
        return cm.disciplined_percentiles(100.0, TH, TD, RD)
    return cm.disciplined_percentiles(100.0, TH, TD, RD, probs=probs)

def _raises_named(probs):
    with pytest.raises(ValueError) as ei:
        _call(probs)
    assert type(ei.value) is ValueError
    assert "probs" in str(ei.value), str(ei.value)

def test_kill_string_probs_do_not_return_control():
    _raises_named(["0.05", "0.5", "0.95"])

def test_kill_bool_probs_do_not_key_1p0():
    _raises_named([True, 0.5, 0.9])

def test_all_bad_types_raise_named():
    for p in [["0.5"], [True], [False, 0.5], [0.05, "0.5", 0.95],
              [0.05, True, 0.95], (True, 0.5), ("0.25", 0.75),
              [np.bool_(True), 0.5], [np.str_("0.5"), 0.9],
              [b"0.5", 0.9], [None, 0.5], [0.5, None],
              np.array(["0.05", "0.5", "0.95"]),
              np.array([True, False, True]),
              np.array(["0.5", 0.5], dtype=object),
              np.array([True, 0.5], dtype=object),
              np.array("0.5"), np.array(True),
              "0.5", b"0.5", True, False, None]:
        _raises_named(p)

def test_bad_probs_named_even_with_bad_spot():
    with pytest.raises(ValueError) as ei:
        cm.disciplined_percentiles(float("nan"), TH, TD, RD, probs=["0.5"])
    assert "probs" in str(ei.value), str(ei.value)

def test_numeric_probs_bit_identical():
    for name, probs in PANELS.items():
        out = _call() if probs is None else _call(probs)
        assert _bits(out) == EXPECTED_BITS[name], name

def test_numeric_containers_bit_identical():
    ref = _call([0.05, 0.5, 0.95])
    for probs in [(0.05, 0.5, 0.95), np.array([0.05, 0.5, 0.95]),
                  np.array([0.05, 0.5, 0.95], dtype=object),
                  [np.float64(0.05), np.float32(0.5), 0.95]]:
        assert _call(probs) == ref

if __name__ == "__main__":
    test_kill_string_probs_do_not_return_control()
    test_kill_bool_probs_do_not_key_1p0()
    test_all_bad_types_raise_named()
    test_bad_probs_named_even_with_bad_spot()
    test_numeric_probs_bit_identical()
    test_numeric_containers_bit_identical()
    print("ALL TESTS PASSED (6 test functions)")
