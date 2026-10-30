"""
test_cdf_module_climatology_trailing_returns_type_validation.py — SF-BL-11312.

Pre-registered done/kill criteria (TASK.md):
  "climatology_percentiles raises ValueError naming
   trailing_h_returns on any bool or string element, with
   numeric-return percentile dicts bit-identical; kill if a string
   panel still returns the control dict or a bool panel still
   returns 271.83 at the 0.95 quantile."

Bug: climatology_percentiles never type-validated
trailing_h_returns. np.asarray(dtype=float) parses string returns
and reads True/False as 1.0/0.0 log-returns: a 40-element string
panel (default_rng(0), spot 100) returned a dict bit-identical to
the numeric control, and an all-True bool panel returned
spot*exp(1) = 271.828... at every quantile (0.95: 271.83).
SF-BL-10266 covers this function's spot value and SF-BL-11243 its
probs type — a different argument in each case.

Fix (metaculus_bot/cdf_module.py):
_screen_trailing_h_returns_types_11312 inspects the RAW returns
(container, ndarray dtype, each element) at entry, before any
float coercion / probs / spot screen, and raises ValueError
naming trailing_h_returns + SF-BL-11312 on any bool or
string/bytes element (and any other non-numeric element).
Numeric returns are returned as the identical float array, so
numeric dicts stay bit-identical: expected raw bytes below were
captured live from the pre-fix implementation.

Run:  python3 tests/test_cdf_module_climatology_trailing_returns_type_validation.py
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

RNG = np.random.default_rng(0)
RETS40 = RNG.normal(0.0, 0.02, 40)
STR40 = [str(x) for x in RETS40]


def _raises_named(rets, spot=100.0, probs=None):
    with pytest.raises(ValueError) as excinfo:
        if probs is None:
            cm.climatology_percentiles(spot, rets)
        else:
            cm.climatology_percentiles(spot, rets, probs)
    msg = str(excinfo.value)
    assert "trailing_h_returns" in msg and "SF-BL-11312" in msg, msg
    assert "climatology_percentiles" in msg, msg


def test_task_repros_raise_named():
    _raises_named(STR40)                 # pre-fix: control dict, bit-identical
    _raises_named([True] * 40)           # pre-fix: 0.95 -> 271.828...
    _raises_named(np.array(STR40))       # pre-fix: control dict
    _raises_named(np.array([True] * 40))  # pre-fix: 271.828...


def test_bool_elements_raise():
    for bad in ([True] * 40, [False] * 40, list(RETS40[:39]) + [True],
                list(RETS40[:39]) + [np.bool_(False)],
                np.array([True] * 40),
                np.array(list(RETS40[:39]) + [True], dtype=object)):
        _raises_named(bad)


def test_string_elements_raise():
    for bad in (STR40, list(RETS40[:39]) + [str(RETS40[-1])],
                [b"0.01"] * 40, [np.str_("0.01")] * 40,
                np.array(STR40), np.array([b"0.01"] * 40),
                np.array(STR40, dtype=object)):
        _raises_named(bad)


def test_container_and_scalar_returns_raise():
    for bad in (True, False, "0.01", b"0.01", None, {"a": 1}):
        _raises_named(bad)


def test_other_nonnumeric_elements_raise_named():
    for bad in ([None] * 40, list(RETS40[:39]) + [None],
                list(RETS40[:39]) + [[0.01]],
                list(RETS40[:39]) + [object()]):
        _raises_named(bad)


def test_no_bool_or_string_panel_returns_a_dict():
    for bad in (STR40, [True] * 40, np.array(STR40),
                np.array([True] * 40)):
        try:
            out = cm.climatology_percentiles(100.0, bad)
        except ValueError:
            continue
        raise AssertionError(f"trailing_h_returns={bad!r} returned {out!r}")


def test_kill_values_absent():
    # The bool panel must not return 271.83 at 0.95, and the string
    # panel must not return the numeric control dict.
    control = cm.climatology_percentiles(100.0, RETS40)
    assert abs(control[0.95] - 271.8281828459045) > 1.0
    for bad in ([True] * 40, STR40):
        with pytest.raises(ValueError):
            cm.climatology_percentiles(100.0, bad)


def test_returns_screened_before_spot_and_probs_screens():
    _raises_named([True] * 40, spot=float("nan"))
    _raises_named(STR40, probs=[True])


def test_existing_screens_preserved():
    with pytest.raises(ValueError) as excinfo:
        cm.climatology_percentiles(float("nan"), RETS40, [0.5])
    assert "spot" in str(excinfo.value) and "SF-BL-10266" in str(
        excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        cm.climatology_percentiles(100.0, RETS40, [True])
    assert "probs" in str(excinfo.value) and "SF-BL-11243" in str(
        excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        cm.climatology_percentiles(100.0, [0.01] * 29)
    assert ">= 30" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        cm.climatology_percentiles(100.0, 0.01)
    assert ">= 30" in str(excinfo.value)


def test_clean_panels_bit_identical():
    keys_hex = ("9a9999999999a93f000000000000d03f000000000000e03f"
                "000000000000e83f666666666666ee3f")
    vals_hex = ("bf089d445f62584056a21e21c0af58406422ebf37fef5840"
                "16cfaf9c103059405a5bc49684a95940")
    panels = {
        "list_floats": RETS40,
        "tuple": tuple(RETS40),
        "np_array": np.array(RETS40),
        "nan_inf_mixed": list(RETS40) + [float("nan"), float("inf")],
        "decimal": [Decimal(str(x)) for x in RETS40],
        "fraction": [Fraction(x).limit_denominator(10 ** 9)
                     for x in RETS40],
        "zero_d_elements": [np.array(x) for x in RETS40],
    }
    for name, rets in panels.items():
        d = cm.climatology_percentiles(100.0, rets)
        assert np.array(list(d.keys())).tobytes().hex() == keys_hex, name
        assert np.array(list(d.values())).tobytes().hex() == vals_hex, name
    extra = {
        "int_ones": ([1] * 40, keys_hex,
                     "4ae4a73c40fd7040" * 5),
        "int_zeros": ([0] * 40, keys_hex,
                      "0000000000005940" * 5),
        "range_obj": (range(40), keys_hex,
                      "e8d96c37f3f68540c49184e1de2c3a41809af0afe267"
                      "1b426c5c78a4b5b1fc42e648b813f118b143"),
    }
    for name, (rets, khex, vhex) in extra.items():
        d = cm.climatology_percentiles(100.0, rets)
        assert np.array(list(d.keys())).tobytes().hex() == khex, name
        assert np.array(list(d.values())).tobytes().hex() == vhex, name


def test_disciplined_percentiles_inherits_returns_screen():
    daily = RNG.normal(0.0, 0.01, 60)
    recent = RNG.normal(0.0, 0.012, 30)
    for bad in ([True] * 40, STR40):
        with pytest.raises(ValueError) as excinfo:
            cm.disciplined_percentiles(100.0, bad, daily, recent)
        assert "trailing_h_returns" in str(excinfo.value)
    disc = cm.disciplined_percentiles(100.0, RETS40, daily, recent)
    vals = np.array(list(disc.values()))
    assert np.all(np.isfinite(vals)) and np.all(np.diff(vals) > 0)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed.append(fn.__name__)
            print(f"FAIL  {fn.__name__}: {exc!r}")
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
