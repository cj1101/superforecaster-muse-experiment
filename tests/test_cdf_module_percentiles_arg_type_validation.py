"""
test_cdf_module_percentiles_arg_type_validation.py — SF-BL-11307.

Pre-registered done/kill criteria (TASK.md):
  "percentiles_to_cdf raises ValueError naming percentiles on any bool
   or string key or value in the percentiles dict, with
   numeric-percentile CDFs bit-identical; kill if string percentiles
   still return the control CDF or a bool value still shifts it."

Bug: cdf_module._validate_percentiles coerced every key/value with
float(), so {'0.1':'10','0.5':'50','0.9':'90'} returned a CDF
bit-identical to the numeric control {0.1:10, 0.5:50, 0.9:90}
(pre-fix sha256 2d268c053319... on range 0-100, verified live) and
{0.1: True, 0.5: 50, 0.9: 90} silently shifted the CDF (pre-fix
sha256 9e4fc62904eb...; True read as value 1.0). Other bool panels
were caught only collaterally, mislabelled, by value-domain checks.

Fix (metaculus_bot/cdf_module.py, _validate_percentiles): type-screen
the RAW dict keys/values at entry, before any float() coercion —
bool / np.bool_ / str / bytes / np.str_ (and any other non-numeric)
key or value raises ValueError naming percentiles + SF-BL-11307.
Reject-only: numeric keys/values fall through to the identical
float() conversion, so numeric CDFs stay bit-identical (SHA-256 pins
below are pre-fix hashes; the 0-30 / default5 / pchip / tied /
logscale pins are also the SF-BL-10107 panel pins).

Run: python3 tests/test_cdf_module_percentiles_arg_type_validation.py
     or under pytest.
"""
import hashlib
import os
import sys
from decimal import Decimal
from fractions import Fraction

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metaculus_bot"))

import numpy as np  # noqa: E402
import cdf_module as cm  # noqa: E402

PASS = []
CONTROL = {0.1: 10, 0.5: 50, 0.9: 90}
CONTROL_HASH = "2d268c053319c1104c711564333330b611830129e19f60856f8df0a82a61c82c"


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _raises_named(fn):
    try:
        fn()
    except ValueError as e:
        msg = str(e)
        assert "percentiles" in msg and "SF-BL-11307" in msg, msg
        return msg
    raise AssertionError("no ValueError raised")


def test_kill_string_percentiles_do_not_return_control():
    _raises_named(lambda: cm.percentiles_to_cdf(
        {'0.1': '10', '0.5': '50', '0.9': '90'}, 0, 100))


def test_kill_bool_value_does_not_shift_cdf():
    # Pre-fix this returned a shifted CDF (hash 9e4fc629...); now raises.
    _raises_named(lambda: cm.percentiles_to_cdf(
        {0.1: True, 0.5: 50, 0.9: 90}, 0, 100))


def test_string_keys_and_values_each_position():
    for panel in ({'0.1': 10, 0.5: 50, 0.9: 90},
                  {0.1: 10, '0.5': 50, 0.9: 90},
                  {0.1: 10, 0.5: 50, '0.9': 90},
                  {0.1: '10', 0.5: 50, 0.9: 90},
                  {0.1: 10, 0.5: '50', 0.9: 90},
                  {0.1: 10, 0.5: 50, 0.9: '90'},
                  {np.str_('0.1'): 10, 0.5: 50, 0.9: 90},
                  {0.1: np.str_('10'), 0.5: 50, 0.9: 90},
                  {b'0.1': 10, 0.5: 50, 0.9: 90},
                  {0.1: b'10', 0.5: 50, 0.9: 90}):
        _raises_named(lambda p=panel: cm.percentiles_to_cdf(p, 0, 100))


def test_bool_keys_and_values_each_position():
    for panel in ({True: 10, 0.5: 50, 0.9: 90},
                  {False: 10, 0.5: 50, 0.9: 90},
                  {0.1: True, 0.5: 50, 0.9: 90},
                  {0.1: 10, 0.5: True, 0.9: 90},
                  {0.1: 10, 0.5: 50, 0.9: False},
                  {0.1: np.bool_(True), 0.5: 50, 0.9: 90},
                  {np.bool_(True): 10, 0.5: 50, 0.9: 90}):
        _raises_named(lambda p=panel: cm.percentiles_to_cdf(p, 0, 100))


def test_other_nonnumeric_and_direct_and_pchip():
    for panel in ({0.1: None, 0.5: 50, 0.9: 90},
                  {None: 10, 0.5: 50, 0.9: 90},
                  {0.1: [10], 0.5: 50, 0.9: 90}):
        _raises_named(lambda p=panel: cm.percentiles_to_cdf(p, 0, 100))
    _raises_named(lambda: cm._validate_percentiles({'0.5': '10', 0.9: 20}))
    _raises_named(lambda: cm.percentiles_to_cdf(
        {'0.1': '10', '0.5': '50', '0.9': '90'}, 0, 100, interp="pchip"))


def test_no_corrupt_type_returns_a_cdf():
    for panel in ({'0.1': '10', '0.5': '50', '0.9': '90'},
                  {0.1: True, 0.5: 50, 0.9: 90}):
        try:
            cm.percentiles_to_cdf(panel, 0, 100)
        except ValueError:
            continue
        raise AssertionError(f"panel {panel!r} returned a CDF")


def test_numeric_panels_bit_identical():
    pins = [
        (CONTROL, 0, 100, {}, CONTROL_HASH),
        ({0.1: 5, 0.5: 10, 0.9: 20}, 0, 30, {},
         "3d1b186d416e50e2feac16b247071f216de7d0f1adc6bf964145e5ebf10c6b19"),
        ({0.05: 2, 0.25: 7, 0.5: 12, 0.75: 18, 0.95: 27}, 0, 30, {},
         "2c0e96a7f51a37acdffb6b11fef2ae8b319d675672adc05cec46aefbb3d20ff4"),
        ({0.05: 2, 0.25: 7, 0.5: 12, 0.75: 18, 0.95: 27}, 0, 30,
         {"interp": "pchip"},
         "e1bc42b7b13773658c31bd6afe68e142dab1a28c0bece86cb201e98944623dfc"),
        ({0.1: 10, 0.5: 10, 0.9: 20}, 0, 30, {},
         "734c4d725d7015916fadcabb4439f8729314a78f6ebf94b51a56b1d82cfdb00e"),
        ({0.1: 5, 0.5: 10, 0.9: 20}, 1, 100, {"zero_point": 0.0},
         "d2d83907982c8c11f45b845fd767bd39e35d8cec0c602059fcfcd8bd174a1bcd"),
        ({np.float64(0.1): np.float64(10), np.float64(0.5): np.int64(50),
          0.9: 90}, 0, 100, {}, CONTROL_HASH),
        ({Decimal('0.1'): Decimal('10'), Fraction(1, 2): 50, 0.9: 90},
         0, 100, {}, CONTROL_HASH),
    ]
    for panel, lo, hi, kw, want in pins:
        cdf = cm.percentiles_to_cdf(panel, lo, hi, **kw)
        got = hashlib.sha256(np.asarray(cdf).tobytes()).hexdigest()
        assert got == want, f"{got} != {want}"


def test_prior_validations_preserved():
    for panel in ({0.5: 10}, {0.0: 5, 0.5: 10}, {0.5: 10, 1.0: 20},
                  {0.1: 20, 0.5: 10, 0.9: 15},
                  {0.1: float('nan'), 0.5: 10, 0.9: 20}):
        try:
            cm.percentiles_to_cdf(panel, 0, 30)
        except ValueError:
            continue
        raise AssertionError(f"panel {panel!r} did not raise")


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for fn in ALL_TESTS:
        try:
            fn()
            check(fn.__name__, True)
        except AssertionError as e:
            check(fn.__name__, False, str(e))
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
