"""
test_cdf_module_naive_spot_type_validation.py — SF-BL-11311 verification.

Pre-registered done/kill criterion (TASK.md):
  "naive_normal_percentiles raises ValueError naming spot on any
   bool or string spot, with numeric-spot percentile dicts
   bit-identical; kill if spot=True still returns the spot=1.0 dict
   or spot='100' still raises a bare UFuncTypeError."

Bug: cdf_module.naive_normal_percentiles value-validated spot
(SF-BL-10393: finite, positive) via float(spot), which silently
coerces bool and numeric-string spots: with lawful trailing returns
at horizon 7, spot=True returned the spot=1.0 percentile dict
(task repro: centred on 1.036 vs 103.562 at the spot=100 control)
and spot='100' was read as 100.0 (on the unscreened wave3 original
it died with a bare UFuncTypeError). Distinct from SF-BL-10393
(spot value domain) and SF-BL-11244 (this function's probs and
horizon_days types).

Fix (metaculus_bot/cdf_module.py, naive_normal_percentiles): a
reject-only type screen immediately before the SF-BL-10393 float
conversion — any bool / np.bool_ / str / bytes / None spot, or any
other non-numeric spot (containers, objects, non-scalar or
bool/string-dtype arrays), raises a ValueError naming spot,
dual-tagged SF-BL-11311 + SF-BL-10393 so the 10393 screen's string
cases (None, 'abc', []) keep naming it. Genuinely numeric spots
(int / float / numpy numeric / Decimal / Fraction / 0-d numeric
arrays) fall through untouched, so numeric-spot dicts are
bit-identical. Reference bytes were captured live from the
pre-fix implementation (seed 11311 panels; work dir verify_fix.py
covers all panels on all three module copies).

Run:  python3 tests/test_cdf_module_naive_spot_type_validation.py
      or pytest tests/test_cdf_module_naive_spot_type_validation.py
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

RETS = np.random.default_rng(11311).normal(0.001, 0.02, 60)
TASK_HEX = ("3ffa58d90f6557405d0fc130d28b58401a76dd9335615940"
            "85216005d83d5a40437b2b8c78885b40")


def _raises_named(spot, rets=RETS, horizon=7):
    with pytest.raises(ValueError) as excinfo:
        cm.naive_normal_percentiles(spot, rets, horizon)
    msg = str(excinfo.value)
    assert "spot" in msg and "SF-BL-11311" in msg, msg
    assert "naive_normal_percentiles" in msg, msg


def test_kill_spot_true_does_not_return_spot1_dict():
    spot1 = cm.naive_normal_percentiles(1.0, RETS, 7)
    try:
        out = cm.naive_normal_percentiles(True, RETS, 7)
    except ValueError:
        return
    raise AssertionError(f"spot=True returned {out!r} "
                         f"(spot=1.0 dict is {spot1!r})")


def test_kill_spot_string_100_raises_named_not_ufunc():
    _raises_named("100")


def test_bool_spots_raise_named():
    for bad in (True, False, np.bool_(True), np.bool_(False),
                np.array(True), np.array([True])):
        _raises_named(bad)


def test_string_spots_raise_named():
    for bad in ("100", "1e2", " 100 ", "", b"100", np.str_("100"),
                np.array("100"), np.array(["100"])):
        _raises_named(bad)


def test_other_corrupt_type_spots_raise_named():
    for bad in (None, [], [100.0], {}, object(),
                np.array([100.0, 200.0])):
        _raises_named(bad)


def test_no_corrupt_type_spot_returns_a_dict_kill_sweep():
    for bad in (True, False, "100", "abc", None, [], np.bool_(True)):
        try:
            out = cm.naive_normal_percentiles(bad, RETS, 7)
        except ValueError:
            continue
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(
                f"spot={bad!r} leaked bare {type(exc).__name__}") from exc
        raise AssertionError(f"spot={bad!r} returned {out!r}")


def test_prior_spot_value_screen_preserved():
    # SF-BL-10393 value domain still raises, still naming SF-BL-10393.
    for bad in (float("nan"), float("inf"), float("-inf"), 0, -100,
                None, "abc", []):
        with pytest.raises(ValueError) as excinfo:
            cm.naive_normal_percentiles(bad, RETS, 7)
        msg = str(excinfo.value)
        assert "spot" in msg and "SF-BL-10393" in msg, msg


def test_spot_type_screened_before_fitting_screens():
    _raises_named(True, rets=[0.01] * 5)
    _raises_named("100", horizon=float("nan"))


def test_numeric_spot_bit_identical():
    d = cm.naive_normal_percentiles(100.0, RETS, 7)
    assert np.array(list(d.values())).tobytes().hex() == TASK_HEX
    assert np.all(np.isfinite(list(d.values())))


def test_numeric_forms_bit_identical():
    for spot in (100, np.float64(100.0), Decimal("100"),
                 Fraction(100, 1), np.array(100.0)):
        d = cm.naive_normal_percentiles(spot, RETS, 7)
        assert np.array(list(d.values())).tobytes().hex() == TASK_HEX, spot


def test_lawful_output_feeds_valid_cdf():
    lawful = cm.naive_normal_percentiles(100.0, RETS, 7)
    cdf = cm.percentiles_to_cdf(lawful, 50.0, 200.0)
    assert cm.validate_cdf(cdf) == []


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
