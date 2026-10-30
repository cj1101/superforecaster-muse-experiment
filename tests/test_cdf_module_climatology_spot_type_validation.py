"""
test_cdf_module_climatology_spot_type_validation.py — SF-BL-11346 verification.

Pre-registered done/kill criteria (TASK.md):
  "climatology_percentiles raises ValueError naming spot on any bool
   or string spot, with numeric-spot percentile dicts bit-identical;
   kill if spot=True still returns the spot=1.0 dict or spot='100'
   still raises a bare UFuncTypeError."

Bug: cdf_module.climatology_percentiles screened spot's VALUE
(SF-BL-10266: finite, positive, via float(spot)) but never its TYPE,
so float(True) read a bool spot as 1.0 and spot=True returned a
percentile dict centred on 1.0 (live repro, 40 trailing returns,
default_rng(0): median 0.9974218461954439 vs 99.74218461954439 at
the spot=100 control); float('100') / float(b'100') silently
coerced a string spot to 100.0 and returned the control dict, and
on the unscreened wave3 original spot='100' died with a bare
UFuncTypeError naming nothing.

Fix (metaculus_bot/cdf_module.py, climatology_percentiles, mirrored
to the scaffold and wave3 copies): a reject-only spot TYPE screen
immediately before the SF-BL-10266 value screen — any bool /
np.bool_ / str / bytes / None spot, and any other non-numeric spot
(containers, arbitrary objects, non-scalar or bool/string-dtype
arrays), raises a ValueError naming spot, dual-tagged
(SF-BL-11346; SF-BL-10266) so the SF-BL-10266 cases (None, 'abc',
[]) keep naming it. Numeric spots fall through untouched to the
identical float conversion, so numeric-spot dicts stay
bit-identical: expected raw bytes below are the SF-BL-10266
pre-fix captures for the same panels.

Run:  python3 tests/test_cdf_module_climatology_spot_type_validation.py
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
CONTROL_MEDIAN = 99.74218461954439
BOOL_MEDIAN_PREFIX = 0.9974218461954439  # pre-fix spot=True median


def _raises_named(spot, rets=RETS40, probs=None):
    with pytest.raises(ValueError) as excinfo:
        if probs is None:
            cm.climatology_percentiles(spot, rets)
        else:
            cm.climatology_percentiles(spot, rets, probs)
    msg = str(excinfo.value)
    assert "spot" in msg and "climatology_percentiles" in msg, msg
    assert "SF-BL-11346" in msg and "SF-BL-10266" in msg, msg


def test_kill_bool_spot_does_not_return_spot1_dict():
    for bad in (True, False, np.bool_(True), np.bool_(False)):
        _raises_named(bad)


def test_kill_string_spot_not_bare_ufunc():
    for bad in ("100", "abc", "", b"100", np.str_("100"),
                np.bytes_(b"100")):
        _raises_named(bad)


def test_all_bad_spot_types_raise_named():
    for bad in (None, [100], (100,), {"s": 100}, {100}, object(),
                np.array([100.0]), np.array(["100"]),
                np.array([True]), np.array(True), np.array("100"),
                np.array(None, dtype=object),
                np.array("100", dtype=object)):
        _raises_named(bad)


def test_no_bad_spot_type_returns_a_dict():
    for bad in (True, False, "100", b"100", None, [100]):
        try:
            out = cm.climatology_percentiles(bad, RETS40)
        except ValueError:
            continue
        raise AssertionError(f"spot={bad!r} returned {out!r} "
                             "instead of raising")


def test_spot_value_domain_preserved():
    # SF-BL-10266: numeric-type but corrupt-value spots still raise
    # naming spot + SF-BL-10266 via the value screen.
    for bad in (float("nan"), float("inf"), -float("inf"),
                np.float64("nan"), 0.0, -0.0, -100.0, -1e-9):
        with pytest.raises(ValueError) as excinfo:
            cm.climatology_percentiles(bad, RETS40)
        msg = str(excinfo.value)
        assert "spot" in msg and "SF-BL-10266" in msg, msg


def test_spot_type_screen_ordering_preserved():
    # SF-BL-11312 trailing screen and SF-BL-11243 probs screen fire
    # before the spot type screen; a bool spot with lawful
    # returns/probs names spot.
    with pytest.raises(ValueError) as excinfo:
        cm.climatology_percentiles(True, [True] * 40)
    assert "trailing_h_returns" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        cm.climatology_percentiles(True, RETS40, [True])
    assert "probs" in str(excinfo.value)
    _raises_named(True, probs=[0.5])
    # >= 30-returns screen unchanged for a lawful spot.
    with pytest.raises(ValueError) as excinfo:
        cm.climatology_percentiles(100.0, [0.01] * 29)
    assert ">= 30" in str(excinfo.value)


def test_numeric_spot_bit_identical():
    rng = np.random.default_rng(10266)
    r_norm = rng.normal(0.0, 0.02, 40)
    panels = {
        "task40_spot100": ((100.0, r_norm, None),
                           "b69c2e165f1d5840a3bf4c1a419c584011e2b589c9dc5840c3f0d13d2c235940d9ae27a281ce5940"),
        "int_spot": ((100, r_norm, None),
                     "b69c2e165f1d5840a3bf4c1a419c584011e2b589c9dc5840c3f0d13d2c235940d9ae27a281ce5940"),
        "spot_small": ((0.01, r_norm, None),
                       "43d917d339c1833f71c197172b29843f9de4f6a1085e843ff1dafb8db197843f1cbd90ce0c24853f"),
        "custom_probs": ((100.0, r_norm, (0.1, 0.5, 0.9)),
                         "0f2c5142f82f584011e2b589c9dc5840b1b6f1ba49985940"),
    }
    for name, ((spot, rets, probs), expected_hex) in panels.items():
        d = (cm.climatology_percentiles(spot, rets) if probs is None
             else cm.climatology_percentiles(spot, rets, probs))
        vals = np.array(list(d.values()))
        assert vals.tobytes().hex() == expected_hex, name
    # Other lawful numeric spot forms all reproduce the control dict.
    control = cm.climatology_percentiles(100.0, RETS40)
    assert control[0.5] == CONTROL_MEDIAN
    assert control[0.5] != BOOL_MEDIAN_PREFIX
    for spot in (np.int64(100), np.float64(100.0), np.float32(100.0),
                 Decimal("100"), Fraction(100, 1), np.array(100.0),
                 np.array(100)):
        d = cm.climatology_percentiles(spot, RETS40)
        assert list(d.keys()) == list(control.keys()), spot
        assert np.array(list(d.values())).tobytes() == \
            np.array(list(control.values())).tobytes(), spot


def test_disciplined_percentiles_inherits_spot_type_screen():
    daily = RNG.normal(0.0, 0.01, 60)
    recent = RNG.normal(0.0, 0.012, 30)
    for bad in (True, "100"):
        with pytest.raises(ValueError) as excinfo:
            cm.disciplined_percentiles(bad, RETS40, daily, recent)
        assert "spot" in str(excinfo.value)
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
