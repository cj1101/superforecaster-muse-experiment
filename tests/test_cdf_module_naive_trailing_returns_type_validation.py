"""
test_cdf_module_naive_trailing_returns_type_validation.py — SF-BL-11313 verification.

Pre-registered done/kill criterion (TASK.md):
  "naive_normal_percentiles raises ValueError naming
   trailing_daily_returns on any bool or string element, with
   numeric-return percentile dicts bit-identical; kill if a string
   panel still returns the control dict or a bool panel still
   returns a 3311.55 median."

Bug: cdf_module.naive_normal_percentiles coerced its trailing
panel with np.asarray(trailing_daily_returns, dtype=float) and no
type screen, so strings parsed and bools became 1/0 daily returns.
Live repro (spot=100, horizon 7, control = [0.01, 0.0] * 20, mean
0.005, median 103.56197087996233): the string panel
[str(x) for x in control] returned a dict bit-identical to the
numeric control and the bool panel [True, False] * 20 (mean 0.5
read as returns) returned median 3311.545195869231 =
100 * exp(0.5 * 7). Distinct from SF-BL-11244 (this function's
probs / horizon_days types) and SF-BL-10393 (its spot value
domain) — different arguments in each case.

Fix (metaculus_bot/cdf_module.py, naive_normal_percentiles): a
reject-only type screen immediately before the asarray coercion —
any bool / np.bool_ / str / bytes / np.str_ element (list / tuple /
sequence at any nesting depth, bool- or string-dtype array,
object array holding one, 0-d bool/string array element, or the
panel itself in one of those forms), and any other non-numeric
element or non-sequence panel, raises a ValueError naming
trailing_daily_returns, tagged SF-BL-11313. Genuinely numeric
returns (int / float / numpy numeric, Decimal / Fraction elements,
nested / 2-D / object-dtype numeric panels, NaN/inf floats) fall
through untouched, so numeric-return dicts are bit-identical.
Reference bytes were captured live from the pre-fix implementation
(work dir verify_fix.py covers all panels on all three copies).

Run:  python3 tests/test_cdf_module_naive_trailing_returns_type_validation.py
      or pytest tests/test_cdf_module_naive_trailing_returns_type_validation.py
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

BASE = np.array([0.01, 0.0] * 20)
STR_PANEL = [str(x) for x in BASE]
BOOL_PANEL = [True, False] * 20
R60 = np.random.default_rng(11313).normal(0.001, 0.02, 60)
TASK_HEX = ("f141ae8c81535940dc32a8eb57a8594086afb554f7e3594"
            "017f8a04a21205a405089091ca5775a40")
RNG_HEX = ("598ee9940f815640410b63871ec257400b612dffc3ab58407b0"
           "15e37639e5940f2fae754e10b5b40")


def _hex(d):
    return np.array(list(d.values())).tobytes().hex()


def _raises_named(rets, spot=100.0, horizon=7):
    with pytest.raises(ValueError) as excinfo:
        cm.naive_normal_percentiles(spot, rets, horizon)
    msg = str(excinfo.value)
    assert "trailing_daily_returns" in msg and "SF-BL-11313" in msg, msg
    assert "naive_normal_percentiles" in msg, msg


def test_kill_string_panel_does_not_return_control_dict():
    control = cm.naive_normal_percentiles(100.0, BASE, 7)
    try:
        out = cm.naive_normal_percentiles(100.0, STR_PANEL, 7)
    except ValueError:
        return
    raise AssertionError(f"string panel returned {out!r} "
                         f"(control dict is {control!r})")


def test_kill_bool_panel_does_not_return_3311_median():
    try:
        out = cm.naive_normal_percentiles(100.0, BOOL_PANEL, 7)
    except ValueError:
        return
    raise AssertionError(f"bool panel returned {out!r} "
                         f"(median {out[0.5]!r}, kill value 3311.55)")


def test_string_panels_raise_named():
    for bad in (STR_PANEL, tuple(STR_PANEL), np.array(STR_PANEL),
                np.array(STR_PANEL, dtype=object),
                [b"0.01", b"0.0"] * 20,
                list(BASE[:39]) + ["0.01"],
                list(BASE[:39]) + [np.str_("0.01")],
                list(BASE[:39]) + [b"0.01"],
                list(BASE[:39]) + [np.array("0.01")],
                [list(BASE[:20]), [str(x) for x in BASE[20:]]],
                "0.01"):
        _raises_named(bad)


def test_bool_panels_raise_named():
    for bad in (BOOL_PANEL, np.array(BOOL_PANEL),
                np.array(BOOL_PANEL, dtype=object),
                np.array([True] * 40),
                list(BASE[:39]) + [True],
                list(BASE[:39]) + [np.bool_(False)],
                list(BASE[:39]) + [np.array(True)],
                [list(BASE[:20]), [True] + list(BASE[21:])],
                True):
        _raises_named(bad)


def test_other_corrupt_type_panels_raise_named():
    for bad in (None, list(BASE[:39]) + [None],
                [complex(x, 0) for x in BASE],
                {i: float(v) for i, v in enumerate(BASE)}, 0.01):
        _raises_named(bad)


def test_no_corrupt_type_returns_a_dict_kill_sweep():
    for bad in (STR_PANEL, BOOL_PANEL, np.array(STR_PANEL),
                np.array(BOOL_PANEL)):
        with pytest.raises(ValueError):
            cm.naive_normal_percentiles(100.0, bad, 7)


def test_numeric_bit_identical():
    assert _hex(cm.naive_normal_percentiles(100.0, BASE, 7)) == TASK_HEX
    assert _hex(cm.naive_normal_percentiles(100.0, R60, 7)) == RNG_HEX
    assert _hex(cm.naive_normal_percentiles(
        100.0, list(R60), 7)) == RNG_HEX
    assert _hex(cm.naive_normal_percentiles(
        100.0, tuple(R60), 7)) == RNG_HEX


def test_numeric_forms_bit_identical():
    for rets in ([Decimal("0.01"), Decimal("0.0")] * 20,
                 [Fraction(1, 100), Fraction(0, 1)] * 20,
                 BASE.reshape(2, 20),
                 [list(BASE[:20]), list(BASE[20:])],
                 np.array(BASE.tolist(), dtype=object),
                 (x for x in BASE)):
        assert _hex(cm.naive_normal_percentiles(
            100.0, rets, 7)) == TASK_HEX, type(rets)


def test_prior_screens_preserved():
    with pytest.raises(ValueError) as e1:
        cm.naive_normal_percentiles(100.0, R60, 5, probs=[True])
    assert "SF-BL-11244" in str(e1.value) and "probs" in str(e1.value)
    with pytest.raises(ValueError) as e2:
        cm.naive_normal_percentiles(100.0, R60, True)
    assert "SF-BL-11244" in str(e2.value)
    with pytest.raises(ValueError) as e3:
        cm.naive_normal_percentiles(float("nan"), R60, 5)
    assert "SF-BL-10393" in str(e3.value)
    with pytest.raises(ValueError) as e4:
        cm.naive_normal_percentiles(100.0, [0.01] * 10, 7)
    assert ">= 30" in str(e4.value)


def test_trailing_type_screened_before_length_screen():
    # A short corrupt panel names its type, not the >= 30 screen.
    _raises_named([True, False])
    _raises_named(["0.01"])


def test_client_adapter_raises_and_matches():
    import metaculus_client as mc
    with pytest.raises(ValueError) as excinfo:
        mc.naive_normal_percentiles(100.0, BOOL_PANEL, 7)
    assert "trailing_daily_returns" in str(excinfo.value)
    assert mc.naive_normal_percentiles(100.0, BASE, 7) == \
        cm.naive_normal_percentiles(100.0, BASE, 7)


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
