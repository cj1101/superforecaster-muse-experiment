"""
test_cdf_module_disciplined_recent_returns_type.py — SF-BL-11350 verification.

Pre-registered done/kill criterion (TASK.md):
  "disciplined_percentiles raises ValueError naming
   recent_daily_returns on any bool or string element, with
   numeric-panel percentile dicts bit-identical; kill if a string
   panel still returns the control dict or a bool panel still
   returns 102.079 at the 0.95 quantile."

Bug: disciplined_percentiles never type-validated
recent_daily_returns. It feeds sd_recent in the volatility-regime
factor via np.asarray(dtype=float) with no dtype screen, so a
30-element string panel returned the control dict bit-identically
(task live repro, spot=100: median 99.871, 0.95: 102.880) and a
[True]*30 panel silently rescaled the dict (0.95: 102.079).
Distinct from SF-BL-11312/SF-BL-11313 (trailing panels in
climatology_percentiles / naive_normal_percentiles) — this
argument is consumed directly in disciplined_percentiles' body,
not delegated to climatology_percentiles.

Fix: reject-only recent_daily_returns type screen in
disciplined_percentiles (after the SF-BL-11310 spot screen, before
the probs screen) in both live copies; numeric panels fall through
as the identical float array, so their dicts stay bit-identical
(expected bits below are the pre-fix outputs for seed-11350
panels, verified vs a verbatim pre-fix reference in the task's
verify_fix.py, 26/26 across both copies).

Run: python3 tests/test_cdf_module_disciplined_recent_returns_type.py
     or pytest tests/test_cdf_module_disciplined_recent_returns_type.py
"""
import os, struct, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metaculus_bot"))
import numpy as np
import pytest
import cdf_module as cm

RNG = np.random.default_rng(11350)
TH = RNG.normal(0.001, 0.02, 60)
TD = RNG.normal(0.0005, 0.012, 120)
RD = RNG.normal(0.0008, 0.015, 30)
BITS_DEFAULT = "3351f1c994fb5740713cc88e369758406b19beb5771159408347787614875940fa1567b12cf85940"
BITS_FLOAT32 = "cc3bd7c994fb5740909dbc8e369758406b19beb57711594075ab837614875940e89e7db12cf85940"
BITS_CUSTOM = "ebeca58012e1574004c95d757e4458406b19beb5771159409dc2ef8ea0c75940ada01c9702355a40"
CONTROL = {0.05: 95.93095635000354, 0.25: 98.36270494036013,
           0.5: 100.27293151440033, 0.75: 102.1106239485362,
           0.95: 103.87772784296968}
BOOL_095_PREFIX = 102.77148870608642  # seed-11350 bool-panel shift

def _bits(d): return "".join(struct.pack("<d", d[p]).hex() for p in sorted(d))
def _call(rd, **kw): return cm.disciplined_percentiles(100.0, TH, TD, rd, **kw)
def _raises_named(rd, **kw):
    with pytest.raises(ValueError) as ei:
        _call(rd, **kw)
    assert type(ei.value) is ValueError
    assert "recent_daily_returns" in str(ei.value), str(ei.value)
    assert "SF-BL-11350" in str(ei.value), str(ei.value)

def test_kill_string_panel_does_not_return_control():
    _raises_named([str(x) for x in RD])

def test_kill_bool_panel_does_not_return_shifted_dict():
    _raises_named([True] * 30)

def test_string_panels_raise_named():
    for p in [[str(x) for x in RD], np.array([str(x) for x in RD]),
              np.array([str(x) for x in RD], dtype=object),
              [b"0.01"] * 30, [np.str_("0.01")] * 30,
              list(RD[:29]) + ["0.01"]]:
        _raises_named(p)

def test_bool_panels_raise_named():
    for p in [[True] * 30, [False] * 30, np.array([True] * 30),
              np.array([True] * 30, dtype=object),
              [np.bool_(True)] * 30, list(RD[:29]) + [True]]:
        _raises_named(p)

def test_other_corrupt_type_panels_raise_named():
    for p in [[None] * 30, [complex(0.01, 0.0)] * 30,
              list(RD[:29]) + [["0.01"]], list(RD[:29]) + [object()],
              "0.01", True, None, {"r": 0.01}, {0.01, 0.02}, 0.01]:
        _raises_named(p)

def test_no_corrupt_type_returns_a_dict_kill_sweep():
    for p in [[str(x) for x in RD], [True] * 30]:
        with pytest.raises(ValueError):
            _call(p)
    # The pre-fix kill values are absent: control is intact.
    assert _call(RD) == CONTROL
    assert _call(RD)[0.95] != BOOL_095_PREFIX

def test_bad_recent_named_even_with_bad_probs():
    _raises_named([True] * 30, probs=["0.5"])

def test_spot_screen_still_first():
    with pytest.raises(ValueError) as ei:
        cm.disciplined_percentiles(True, TH, TD, [True] * 30)
    assert "spot" in str(ei.value), str(ei.value)

def test_numeric_bit_identical():
    assert _bits(_call(RD)) == BITS_DEFAULT
    assert _bits(_call(RD.astype(np.float32))) == BITS_FLOAT32
    assert _bits(_call(RD, probs=(0.01, 0.1, 0.5, 0.9, 0.99))) == BITS_CUSTOM

def test_numeric_forms_bit_identical():
    from decimal import Decimal
    from fractions import Fraction
    ref = _call(RD)
    for panel in [list(RD), tuple(RD), np.array(list(RD), dtype=object),
                  [Decimal(str(x)) for x in RD],
                  [Fraction(x).limit_denominator(10 ** 9) for x in RD],
                  RD.reshape(6, 5), iter(list(RD))]:
        assert _call(panel) == ref

def test_prior_screens_preserved():
    with pytest.raises(ValueError) as ei:
        _call(RD, probs=["0.5"])
    assert "probs" in str(ei.value)
    with pytest.raises(ValueError) as ei:
        _call(RD, probs=(0.95, 0.05))
    assert "SF-BL-10316" in str(ei.value)
    with pytest.raises(ValueError) as ei:
        cm.disciplined_percentiles(100.0, [True] * 40, TD, RD)
    assert "trailing_h_returns" in str(ei.value)

def test_client_adapter_raises_and_matches():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metaculus_bot"))
    import metaculus_client as mc
    with pytest.raises(ValueError) as ei:
        mc.disciplined_percentiles(100.0, TH, TD, [True] * 30)
    assert "recent_daily_returns" in str(ei.value)
    assert mc.disciplined_percentiles(100.0, TH, TD, RD) == CONTROL

if __name__ == "__main__":
    test_kill_string_panel_does_not_return_control()
    test_kill_bool_panel_does_not_return_shifted_dict()
    test_string_panels_raise_named()
    test_bool_panels_raise_named()
    test_other_corrupt_type_panels_raise_named()
    test_no_corrupt_type_returns_a_dict_kill_sweep()
    test_bad_recent_named_even_with_bad_probs()
    test_spot_screen_still_first()
    test_numeric_bit_identical()
    test_numeric_forms_bit_identical()
    test_prior_screens_preserved()
    test_client_adapter_raises_and_matches()
    print("ALL TESTS PASSED (12 test functions)")
