"""
test_cdf_module_disciplined_trailing_h_type.py — SF-BL-11384 verification.

Pre-registered done/kill criterion (TASK.md):
  "Guard raises ValueError naming trailing_h_returns on any string
   or bool element, with float panels bit-identical; kill if a
   string trailing_h panel still returns the numeric-control dict
   or a bool panel still returns 0.95=164.872 instead of raising."

Bug: disciplined_percentiles never type-validated
trailing_h_returns. It consumes the argument twice — as the
climatology base via climatology_percentiles and directly for
the trailing excess-kurtosis tail fattening (step 4) — with only
np.asarray(dtype=float) coercion, so a 60-element string panel
returned the control dict bit-identically (task live repro,
spot=100: 0.95 percentile 102.888) and a bool panel silently
shifted the dict (task live repro: 0.95 = 164.872). Distinct
from SF-BL-11312 (climatology_percentiles' trailing_h_returns
screen — a different function; the scaffold copy's climatology
has no screen at all and that screen never guarded
disciplined's own step-4 use), SF-BL-11349/SF-BL-11350 (this
function's trailing_daily_returns / recent_daily_returns) and
SF-BL-11310/SF-BL-11268 (this function's spot / probs).

Fix: reject-only trailing_h_returns type screen in
disciplined_percentiles (after the spot / recent / probs
screens, immediately before the climatology call; the screened
float array feeds both consumers) in both live copies; numeric
panels fall through as the identical float array, so their
dicts stay bit-identical (expected bits below are the pre-fix
outputs for seed-11384 panels, verified vs a verbatim pre-fix
reference in the task's verify_fix.py, 24/24 across both copies).

Run: python3 tests/test_cdf_module_disciplined_trailing_h_type.py
     or pytest tests/test_cdf_module_disciplined_trailing_h_type.py
"""
import os, struct, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metaculus_bot"))
import numpy as np
import pytest
import cdf_module as cm

RNG = np.random.default_rng(11384)
TH = RNG.normal(0.001, 0.02, 60)
TD = RNG.normal(0.0005, 0.012, 120)
RD = RNG.normal(0.0008, 0.015, 30)
BITS_DEFAULT = "561b5fb4dc33584002b17f5175a15840ca147d40e306594000957495156f5940482051fd0cec5940"
BITS_FLOAT32 = "73b513b4dc335840277b6c5175a1584082528040e306594013f75195156f594060bc44fd0cec5940"
BITS_CUSTOM = "a33c3c45921a584030d6791fff4c5840ca147d40e306594038b202048bc459402ef128c1de435a40"
CONTROL = {0.05: 96.81034573829433, 0.25: 98.52278554410807,
           0.5: 100.10762035574876, 0.75: 101.73569237122501,
           0.95: 103.68829281733008}
BOOL_095_PREFIX = 236.64487766448758  # seed-11384 [True,False]*30 shift (out copy)

def _bits(d): return "".join(struct.pack("<d", d[p]).hex() for p in sorted(d))
def _call(th, **kw): return cm.disciplined_percentiles(100.0, th, TD, RD, **kw)
def _raises_named(th, **kw):
    with pytest.raises(ValueError) as ei:
        _call(th, **kw)
    assert type(ei.value) is ValueError
    assert "trailing_h_returns" in str(ei.value), str(ei.value)
    assert "SF-BL-11384" in str(ei.value), str(ei.value)

def test_kill_string_panel_does_not_return_control():
    _raises_named([str(x) for x in TH])

def test_kill_bool_panel_does_not_return_shifted_dict():
    _raises_named([True, False] * 30)
    _raises_named([True] * 60)

def test_string_panels_raise_named():
    for p in [[str(x) for x in TH], np.array([str(x) for x in TH]),
              np.array([str(x) for x in TH], dtype=object),
              [b"0.01"] * 60, [np.str_("0.01")] * 60,
              list(TH[:59]) + ["0.01"]]:
        _raises_named(p)

def test_bool_panels_raise_named():
    for p in [[True] * 60, [False] * 60, np.array([True] * 60),
              np.array([True] * 60, dtype=object),
              [np.bool_(True)] * 60, list(TH[:59]) + [True]]:
        _raises_named(p)

def test_other_corrupt_type_panels_raise_named():
    for p in [[None] * 60, [complex(0.01, 0.0)] * 60,
              list(TH[:59]) + [["0.01"]], list(TH[:59]) + [object()],
              "0.01", True, None, {"r": 0.01}, {0.01, 0.02}, 0.01]:
        _raises_named(p)

def test_no_corrupt_type_returns_a_dict_kill_sweep():
    for p in [[str(x) for x in TH], [True, False] * 30]:
        with pytest.raises(ValueError):
            _call(p)
    assert _call(TH) == CONTROL
    assert _call(TH)[0.95] != BOOL_095_PREFIX

def test_spot_screen_still_first():
    with pytest.raises(ValueError) as ei:
        cm.disciplined_percentiles(True, [True] * 60, TD, RD)
    assert "spot" in str(ei.value), str(ei.value)

def test_numeric_bit_identical():
    assert _bits(_call(TH)) == BITS_DEFAULT
    assert _bits(_call(TH.astype(np.float32))) == BITS_FLOAT32
    assert _bits(_call(TH, probs=(0.01, 0.1, 0.5, 0.9, 0.99))) == BITS_CUSTOM

def test_numeric_forms_bit_identical():
    from decimal import Decimal
    from fractions import Fraction
    ref = _call(TH)
    for panel in [list(TH), tuple(TH), np.array(list(TH), dtype=object),
                  [Decimal(str(x)) for x in TH],
                  [Fraction(x).limit_denominator(10 ** 9) for x in TH],
                  TH.reshape(6, 10), iter(list(TH))]:
        assert _call(panel) == ref

def test_prior_screens_preserved():
    with pytest.raises(ValueError) as ei:
        _call(TH, probs=["0.5"])
    assert "probs" in str(ei.value)
    with pytest.raises(ValueError) as ei:
        _call(TH, probs=(0.95, 0.05))
    assert "SF-BL-10316" in str(ei.value)
    with pytest.raises(ValueError) as ei:
        cm.disciplined_percentiles(100.0, TH, TD, [True] * 30)
    assert "recent_daily_returns" in str(ei.value)
    with pytest.raises(ValueError) as ei:
        cm.disciplined_percentiles(100.0, TH, [True] * 60, RD)
    assert "trailing_daily_returns" in str(ei.value)
    # climatology's own SF-BL-11312 screen untouched (repo copy)
    with pytest.raises(ValueError) as ei:
        cm.climatology_percentiles(100.0, [True] * 40)
    assert "trailing_h_returns" in str(ei.value)
    assert "SF-BL-11312" in str(ei.value)

def test_client_adapter_raises_and_matches():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metaculus_bot"))
    import metaculus_client as mc
    with pytest.raises(ValueError) as ei:
        mc.disciplined_percentiles(100.0, [True] * 60, TD, RD)
    assert "trailing_h_returns" in str(ei.value)
    assert mc.disciplined_percentiles(100.0, TH, TD, RD) == CONTROL

if __name__ == "__main__":
    test_kill_string_panel_does_not_return_control()
    test_kill_bool_panel_does_not_return_shifted_dict()
    test_string_panels_raise_named()
    test_bool_panels_raise_named()
    test_other_corrupt_type_panels_raise_named()
    test_no_corrupt_type_returns_a_dict_kill_sweep()
    test_spot_screen_still_first()
    test_numeric_bit_identical()
    test_numeric_forms_bit_identical()
    test_prior_screens_preserved()
    test_client_adapter_raises_and_matches()
    print("ALL TESTS PASSED (11 test functions)")
