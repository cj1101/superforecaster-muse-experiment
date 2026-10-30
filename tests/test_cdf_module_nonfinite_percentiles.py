"""
test_cdf_module_nonfinite_percentiles.py — SF-BL-10107 verification tests.

Pre-registered done/kill criteria (TASK.md):
  "Fix raises a named ValueError on any non-finite percentile prob/value
   and valid panels stay bit-identical; kill if any non-finite input
   still yields a CDF or if a finite panel's CDF changes by >0."

Bug: cdf_module._validate_percentiles used diff/range comparisons that
are all False for NaN, so a NaN percentile passed the strictly-
increasing, (0, 1) and non-decreasing checks, and np.interp propagated
it across the grid. Pre-fix live repro (2026-10-21):
  percentiles_to_cdf({0.1: NaN, 0.5: 10, 0.9: 20}, 0, 30)
  -> 201-point CDF, all-NaN, NO error. Val +/-inf and prob NaN likewise
  yielded (corrupt) CDFs; only prob +/-inf raised, via the (0, 1) check
  and without naming finiteness.

Fix (metaculus_bot/cdf_module.py, _validate_percentiles): np.isfinite
screens on probs and vals BEFORE any comparison, raising ValueError
naming the non-finite field and value(s) (SF-BL-10107). A defence-in-
depth finiteness check on range_min/range_max/zero_point was added in
percentiles_to_cdf. Finite panels are untouched: the expected SHA-256
hashes below were computed from the pre-fix (git HEAD) implementation
and the post-fix outputs are byte-identical for every panel.

Run:  python3 tests/test_cdf_module_nonfinite_percentiles.py
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_nan_forecast.py.
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metaculus_bot"))

import numpy as np  # noqa: E402
import cdf_module as cm  # noqa: E402

PASS = []
NAN = float("nan")
INF = float("inf")


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _raises_value_error(fn, *needles):
    try:
        fn()
    except ValueError as e:
        msg = str(e)
        assert all(n in msg for n in needles), f"message {msg!r} missing {needles}"
        return msg
    raise AssertionError("no ValueError raised")


# ---- 1. Task repro + non-finite values, every position --------------------

def test_task_repro_nan_value_raises():
    _raises_value_error(
        lambda: cm.percentiles_to_cdf({0.1: NAN, 0.5: 10, 0.9: 20}, 0, 30),
        "finite", "percentile value")


def test_nan_value_any_position_raises():
    for panel in ({0.1: NAN, 0.5: 10, 0.9: 20},
                  {0.1: 0, 0.5: NAN, 0.9: 20},
                  {0.1: 0, 0.5: 10, 0.9: NAN},
                  {0.1: NAN, 0.5: NAN, 0.9: NAN}):
        _raises_value_error(lambda p=panel: cm.percentiles_to_cdf(p, 0, 30),
                            "finite")


def test_inf_value_any_position_raises():
    for bad in (INF, -INF):
        for panel in ({0.1: bad, 0.5: 10, 0.9: 20},
                      {0.1: 0, 0.5: bad, 0.9: 20},
                      {0.1: 0, 0.5: 10, 0.9: bad}):
            _raises_value_error(
                lambda p=panel: cm.percentiles_to_cdf(p, 0, 30),
                "finite", "percentile value")


# ---- 2. Non-finite probabilities -------------------------------------------

def test_nan_prob_raises():
    _raises_value_error(
        lambda: cm.percentiles_to_cdf({NAN: 0, 0.5: 10, 0.9: 20}, 0, 30),
        "finite", "percentile probabilit")


def test_inf_prob_raises_named_finite():
    # Pre-fix, +/-inf probs raised only via the (0, 1) message, which does
    # not name finiteness; the fix must name it.
    for bad in (INF, -INF):
        _raises_value_error(
            lambda b=bad: cm.percentiles_to_cdf({0.1: 0, 0.5: 10, b: 20}, 0, 30),
            "finite", "percentile probabilit")


def test_validate_percentiles_direct_raises():
    for panel in ({0.1: NAN, 0.5: 10}, {NAN: 5, 0.5: 10},
                  {0.1: INF, 0.5: 10}, {0.1: 5, 0.5: -INF}):
        _raises_value_error(lambda p=panel: cm._validate_percentiles(p),
                            "finite")


def test_nonfinite_raises_for_pchip_too():
    for interp in ("linear", "pchip"):
        _raises_value_error(
            lambda i=interp: cm.percentiles_to_cdf(
                {0.1: NAN, 0.5: 10, 0.9: 20}, 0, 30, interp=i),
            "finite")
        _raises_value_error(
            lambda i=interp: cm.percentiles_to_cdf(
                {0.1: 0, 0.5: 10, 0.9: INF}, 0, 30, interp=i),
            "finite")


def test_nonfinite_grid_bounds_raise():
    valid = {0.1: 5, 0.5: 10, 0.9: 20}
    for lo, hi, zp in ((NAN, 30, None), (0, INF, None), (0, 30, NAN)):
        _raises_value_error(
            lambda lo=lo, hi=hi, zp=zp: cm.percentiles_to_cdf(
                valid, lo, hi, zero_point=zp),
            "finite")


# ---- 3. Valid panels bit-identical (hashes from pre-fix git HEAD) ----------

# (name, panel, range_min, range_max, zero_point, interp, sha256 of CDF bytes)
PANELS = [
    ("task-valid", {0.1: 5, 0.5: 10, 0.9: 20}, 0, 30, None, "linear",
     "3d1b186d416e50e2feac16b247071f216de7d0f1adc6bf964145e5ebf10c6b19"),
    ("default5", {0.05: 2, 0.25: 7, 0.5: 12, 0.75: 18, 0.95: 27},
     0, 30, None, "linear",
     "2c0e96a7f51a37acdffb6b11fef2ae8b319d675672adc05cec46aefbb3d20ff4"),
    ("default5-pchip", {0.05: 2, 0.25: 7, 0.5: 12, 0.75: 18, 0.95: 27},
     0, 30, None, "pchip",
     "e1bc42b7b13773658c31bd6afe68e142dab1a28c0bece86cb201e98944623dfc"),
    ("dup-values", {0.1: 10, 0.5: 10, 0.9: 20}, 0, 30, None, "linear",
     "734c4d725d7015916fadcabb4439f8729314a78f6ebf94b51a56b1d82cfdb00e"),
    # SF-BL-10770 supersedes this one pin: the pre-10770 hash embodied
    # the tied-values bug itself (pchip collapsed the tie at 10 to its
    # first probability 0.1 and smoothed the jump away, diverging from
    # the linear path by ~0.40 on this very panel). Under 10770's
    # explicit, identical jump handling the tied pchip output is the
    # hash below; the linear dup-values pin above is unchanged (the
    # jump does not bind on that grid and 10770 preserves its bits).
    ("dup-values-pchip", {0.1: 10, 0.5: 10, 0.9: 20}, 0, 30, None, "pchip",
     "bfb7dfe11ea9986a15a354ee85bbc12d38589562542436f8faffbc123c6639f5"),
    ("logscale", {0.1: 5, 0.5: 10, 0.9: 20}, 1, 100, 0.0, "linear",
     "d2d83907982c8c11f45b845fd767bd39e35d8cec0c602059fcfcd8bd174a1bcd"),
    ("neg-range", {0.1: -20, 0.5: 0, 0.9: 15}, -30, 30, None, "linear",
     "d58aeeb254e80ff79bbf8988bc31ca9dcf06d6818b9457c9488815923766013b"),
    ("two-point", {0.25: 3, 0.75: 9}, 0, 12, None, "linear",
     "2b2c0a7fcd668d9a4319dc42c98a3027c3900df66b49a2c1dd7afc589851a289"),
]


def test_valid_panels_bit_identical():
    for name, panel, lo, hi, zp, interp, want in PANELS:
        cdf = cm.percentiles_to_cdf(panel, lo, hi, zero_point=zp, interp=interp)
        got = hashlib.sha256(np.asarray(cdf).tobytes()).hexdigest()
        assert got == want, f"{name}: {got} != {want}"
        assert len(cdf) == 201 and np.all(np.isfinite(cdf)), name


# ---- 4. Pre-existing validations unchanged ---------------------------------

def test_existing_validations_still_raise():
    # <2 percentiles, prob at bounds, decreasing values: all still ValueError.
    for panel in ({0.5: 10},
                  {0.0: 5, 0.5: 10},
                  {0.5: 10, 1.0: 20},
                  {0.1: 20, 0.5: 10, 0.9: 15}):
        _raises_value_error(lambda p=panel: cm.percentiles_to_cdf(p, 0, 30))


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
