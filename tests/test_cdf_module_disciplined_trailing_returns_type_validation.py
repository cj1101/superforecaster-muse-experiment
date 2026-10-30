"""
test_cdf_module_disciplined_trailing_returns_type_validation.py — SF-BL-11349 verification.

Pre-registered done/kill criterion (TASK.md):
  "disciplined_percentiles raises ValueError naming
   trailing_daily_returns on any bool or string element, with
   numeric-panel percentile dicts bit-identical; kill if a string
   panel still returns the control dict or a bool panel still
   returns 102.079 at the 0.95 quantile."

Bug: cdf_module.disciplined_percentiles consumed
trailing_daily_returns via np.asarray(..., dtype=float) for the
long-window volatility sd_long in the regime factor, with no type
screen, so strings parsed and bools became 1/0 daily returns.
Live repro (spot=100, lawful windows; finder's panel in TASK.md):
the 60-element string panel returned the control dict
bit-identically (median 99.871, 0.95: 102.880) and a bool panel
silently rescaled every deviation (0.95: 102.079). On this file's
fixture (default_rng(11349): TH 60, TD 60, RD 30) the pre-fix
values are control 0.95 = 104.30929537412344 and bool-panel 0.95 =
102.9516547805366 — same defect, panel-independent. Distinct from
SF-BL-11312 / SF-BL-11313 (trailing panels of
climatology_percentiles / naive_normal_percentiles, different
functions) and SF-BL-11268 / SF-BL-11310 (this function's probs /
spot types, different arguments).

Fix (metaculus_bot/cdf_module.py, disciplined_percentiles): a
reject-only type screen immediately before the asarray coercion —
any bool / np.bool_ / str / bytes / np.str_ element (list / tuple /
sequence at any nesting depth, bool- or string-dtype array,
object array holding one, 0-d bool/string array element, or the
panel itself in one of those forms), and any other non-numeric
element or non-sequence panel, raises a ValueError naming
trailing_daily_returns, tagged SF-BL-11349. Genuinely numeric
returns (int / float / numpy numeric, Decimal / Fraction elements,
nested / 2-D / object-dtype numeric panels, NaN/inf floats) fall
through untouched, so numeric-panel dicts are bit-identical.
Reference bytes were captured live from the pre-fix implementation
(work dir verify_fix.py covers all panels on both live copies).

Run:  python3 tests/test_cdf_module_disciplined_trailing_returns_type_validation.py
      or pytest tests/test_cdf_module_disciplined_trailing_returns_type_validation.py
"""
import os
import sys
from decimal import Decimal
from fractions import Fraction

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metaculus_bot"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import cdf_module as cm  # noqa: E402

RNG = np.random.default_rng(11349)
TH = RNG.normal(0.001, 0.02, 60)
TD = RNG.normal(0.0005, 0.012, 60)
RD = RNG.normal(0.0008, 0.015, 30)
STR_PANEL = [str(x) for x in TD]
BOOL_PANEL = [True, False] * 30
DEFAULT_HEX = ("d96aeda4ae065840dcfd9d3831a65840969e8f2a8c085940"
               "607ae6913c7e5940832ad37ecb135a40")


def _hex(d):
    return np.array(list(d.values())).tobytes().hex()


def _raises_named(td, **kw):
    with pytest.raises(ValueError) as excinfo:
        cm.disciplined_percentiles(100.0, TH, td, RD, **kw)
    msg = str(excinfo.value)
    assert "trailing_daily_returns" in msg and "SF-BL-11349" in msg, msg
    assert "disciplined_percentiles" in msg, msg


def test_kill_string_panel_does_not_return_control_dict():
    control = cm.disciplined_percentiles(100.0, TH, TD, RD)
    try:
        out = cm.disciplined_percentiles(100.0, TH, STR_PANEL, RD)
    except ValueError:
        return
    raise AssertionError(f"string panel returned {out!r} "
                         f"(control dict is {control!r})")


def test_kill_bool_panel_does_not_return_shifted_dict():
    try:
        out = cm.disciplined_percentiles(100.0, TH, BOOL_PANEL, RD)
    except ValueError:
        return
    raise AssertionError(f"bool panel returned {out!r} "
                         f"(0.95 {out[0.95]!r}; kill value on the "
                         "finder's panel was 102.079)")


def test_string_panels_raise_named():
    for bad in (STR_PANEL, tuple(STR_PANEL), np.array(STR_PANEL),
                np.array(STR_PANEL, dtype=object),
                [b"0.01"] * 60,
                list(TD[:59]) + ["0.01"],
                list(TD[:59]) + [np.str_("0.01")],
                list(TD[:59]) + [b"0.01"],
                list(TD[:59]) + [np.array("0.01")],
                [list(TD[:30]), [str(x) for x in TD[30:]]],
                "0.01"):
        _raises_named(bad)


def test_bool_panels_raise_named():
    for bad in (BOOL_PANEL, np.array(BOOL_PANEL),
                np.array(BOOL_PANEL, dtype=object),
                np.array([True] * 60),
                list(TD[:59]) + [True],
                list(TD[:59]) + [np.bool_(False)],
                list(TD[:59]) + [np.array(True)],
                [list(TD[:30]), [True] + list(TD[31:])],
                True):
        _raises_named(bad)


def test_other_corrupt_type_panels_raise_named():
    for bad in (None, list(TD[:59]) + [None],
                [complex(x, 0) for x in TD],
                {i: float(v) for i, v in enumerate(TD)}, 0.01):
        _raises_named(bad)


def test_no_corrupt_type_returns_a_dict_kill_sweep():
    for bad in (STR_PANEL, BOOL_PANEL, np.array(STR_PANEL),
                np.array(BOOL_PANEL)):
        with pytest.raises(ValueError):
            cm.disciplined_percentiles(100.0, TH, bad, RD)


def test_numeric_bit_identical():
    assert _hex(cm.disciplined_percentiles(100.0, TH, TD, RD)) == DEFAULT_HEX
    assert _hex(cm.disciplined_percentiles(
        100.0, TH, list(TD), RD)) == DEFAULT_HEX
    assert _hex(cm.disciplined_percentiles(
        100.0, TH, tuple(TD), RD)) == DEFAULT_HEX


def test_numeric_forms_bit_identical():
    for rets in ([Decimal(str(x)) for x in TD],
                 [Fraction(x).limit_denominator(10**9) for x in TD],
                 TD.reshape(2, 30),
                 [list(TD[:30]), list(TD[30:])],
                 np.array(TD.tolist(), dtype=object),
                 (x for x in TD)):
        assert _hex(cm.disciplined_percentiles(
            100.0, TH, rets, RD)) == DEFAULT_HEX, type(rets)


def test_prior_screens_preserved():
    with pytest.raises(ValueError) as e1:
        cm.disciplined_percentiles(True, TH, BOOL_PANEL, RD)
    assert "spot" in str(e1.value) and "SF-BL-11310" in str(e1.value)
    with pytest.raises(ValueError) as e2:
        cm.disciplined_percentiles(100.0, TH, BOOL_PANEL, RD,
                                   probs=["0.5"])
    assert "probs" in str(e2.value)
    with pytest.raises(ValueError) as e3:
        cm.disciplined_percentiles(100.0, TH, TD, RD, probs=(0.95, 0.05))
    assert "probs" in str(e3.value) and "SF-BL-10316" in str(e3.value)


def test_client_adapter_raises_and_matches():
    import metaculus_client as mc
    with pytest.raises(ValueError) as excinfo:
        mc.disciplined_percentiles(100.0, TH, BOOL_PANEL, RD)
    assert "trailing_daily_returns" in str(excinfo.value)
    assert mc.disciplined_percentiles(100.0, TH, TD, RD) == \
        cm.disciplined_percentiles(100.0, TH, TD, RD)


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
