"""
test_cdf_module_nominal_zero_point_validation.py — SF-BL-10268 verification.

Pre-registered done/kill criterion (TASK.md):
  "Fix raises ValueError naming zero_point whenever it makes
   deriv_ratio non-finite, <=0, or ==1, while lawful log-scale grids
   stay bit-identical; kill if zero_point in {range_min, interior,
   range_max} still crashes bare or returns NaN nominals."

Bug: cdf_module.nominal_at_location divided by (range_min - zero_point)
and raised deriv_ratio to loc with no screen. In the
superforecaster_out copy every kill case failed (bare
ZeroDivisionError at zero_point == range_min, NaN at interior
zero_point, degenerate constant tail at zero_point == range_max).
In this repo copy SF-BL-10108's interval screen already named the
basic cases, but deriv_ratio itself was never validated:
zero_point=1e20 on [0, 100] rounds deriv_ratio to exactly 1.0 and
returned all-NaN nominals, zero_point=-1e308 on [0, 1e308] overflowed
deriv_ratio to inf and returned NaN, and deriv_ratio == 1 from equal
ranges was reported without naming zero_point.

Fix (metaculus_bot/cdf_module.py, mirrored in the
superforecaster_out copy): _checked_deriv_ratio screens the ratio
before any division/power — non-finite zero_point, zero_point ==
range_min (denominator 0), and any deriv_ratio that is non-finite,
<= 0, or == 1 raise a ValueError naming zero_point and SF-BL-10268.
Lawful ratios pass through as the identical float, so lawful grids
stay bit-identical (hashes below recorded pre-fix, SF-BL-10268 repro).

Run:  python3 tests/test_cdf_module_nominal_zero_point_validation.py
No pytest dependency: test_* functions with assert + __main__ runner.
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metaculus_bot"))

import numpy as np  # noqa: E402

import cdf_module as cm  # noqa: E402

FAILURES = []


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAILURES.append(name)


def _raises_named(fn, *needles):
    try:
        fn()
    except ValueError as e:
        msg = str(e)
        assert all(n in msg for n in needles), f"message {msg!r} lacks {needles}"
        return
    except Exception as e:  # noqa: BLE001 - a bare crash is the bug itself
        raise AssertionError(f"bare {type(e).__name__} (not ValueError): {e}")
    raise AssertionError("returned nominals instead of raising (kill criterion)")


# ---- 1. Task kill cases: zp in {range_min, interior, range_max} ------------------

def test_task_kill_cases():
    for zp in (0.0, 25.0, 50.0, 99.9, 100.0):
        _raises_named(lambda zp=zp: cm.nominal_at_location(
            [0.0, 0.5, 1.0], 0.0, 100.0, zp), "zero_point", "SF-BL-10268")
        _raises_named(lambda zp=zp: cm.grid_nominals(0.0, 100.0, zp),
                      "zero_point", "SF-BL-10268")
    # The task's exact live repros:
    _raises_named(lambda: cm.nominal_at_location(0.5, 0, 100, 0),
                  "zero_point")          # was bare ZeroDivisionError
    _raises_named(lambda: cm.nominal_at_location(0.5, 0, 100, 50),
                  "zero_point")          # was NaN


# ---- 2. deriv_ratio == 1 ----------------------------------------------------------

def test_deriv_ratio_equal_one_raises():
    # Huge zp: float subtraction rounds the ratio to exactly 1.0 even
    # though zp lies far outside the range (10108 screen passes it).
    assert (100.0 - 1e20) / (0.0 - 1e20) == 1.0
    _raises_named(lambda: cm.nominal_at_location([0.0, 0.5, 1.0], 0.0, 100.0, 1e20),
                  "zero_point", "deriv_ratio")
    _raises_named(lambda: cm.grid_nominals(0.0, 100.0, 1e20), "zero_point")
    # Equal ranges make the ratio 1 for any zp; error must name zp here.
    _raises_named(lambda: cm.nominal_at_location(0.5, 10.0, 10.0, 0.0),
                  "zero_point", "deriv_ratio")


# ---- 3. deriv_ratio non-finite / <= 0 ----------------------------------------------

def test_deriv_ratio_nonfinite_and_nonpositive_raise():
    for zp, rmin, rmax in ((float("inf"), 0.0, 100.0),
                           (float("-inf"), 0.0, 100.0),
                           (float("nan"), 0.0, 100.0),
                           (-1e308, 0.0, 1e308),   # ratio overflows to inf
                           (100.0, 0.0, 100.0),    # ratio -0.0
                           (50.0, 0.0, 100.0)):    # ratio -1
        _raises_named(lambda zp=zp, a=rmin, b=rmax: cm.nominal_at_location(
            0.5, a, b, zp), "zero_point", "SF-BL-10268")


# ---- 4. Kill-criterion sweep: no bad zp returns nominals ----------------------------

def test_no_bad_zero_point_returns_nominals():
    zps = [0.0, 5.0, 50.0, 100.0, float("nan"), float("inf"),
           float("-inf"), 1e20, -1e20, -1e308]
    ranges = [(0.0, 100.0), (5.0, 50.0), (-50.0, 50.0), (0.0, 1e308)]
    n = 0
    for rmin, rmax in ranges:
        for zp in zps:
            # Skip genuinely lawful combos (finite ratio > 0, != 1):
            denom = rmin - zp
            if denom != 0.0:
                ratio = (rmax - zp) / denom
                if np.isfinite(ratio) and ratio > 0 and ratio != 1.0:
                    continue
            n += 1
            _raises_named(lambda a=rmin, b=rmax, z=zp: cm.nominal_at_location(
                [0.0, 0.5, 1.0], a, b, z), "zero_point")
    assert n >= 20, n


# ---- 5. Lawful log-scale grids bit-identical (pre-fix SHA-256) -----------------------

PANELS = [
    ((1.0, 100.0, 0.0), "8cd3949bfd85ee8e2832952bec39d7e789d658edaae37be14ae61871ab3f2986"),
    ((10.0, 100000.0, 0.0), "c3305e0fcc96503ff7f155b223729a365170da1473839ae3fb8e1e40a3c675e2"),
    ((-1000.0, -10.0, 0.0), "f37d268ceda1155b24f8a929e5179807b27fe7bc6494ca0e6f9d0708ffa47fbe"),
    ((0.0, 100.0, -10.0), "e638067fdf7c4bcc173ed6f525d9b59768fe59f821cbe0a88754efed2dc63a6c"),
    ((0.0, 100.0, 200.0), "85b474c3f66a82179adbb085cc3e73ae96d9f69a906b91a91337d7e630932e09"),
    ((0.5, 2.5, 0.0), "430684286de30e303c314dda0588abe215e21e92f9a3e8ac67c7b9f5f05a6f17"),
    ((0.0, 100.0, None), "dc4d7d4d2de2ff3f09e4051bd3ad5cb63b08d851ad18562521c7581e19379784"),
]


def test_lawful_grids_bit_identical():
    for (rmin, rmax, zp), want in PANELS:
        g = np.asarray(cm.grid_nominals(rmin, rmax, zp), dtype=float)
        got = hashlib.sha256(g.tobytes()).hexdigest()
        assert got == want, f"grid({rmin},{rmax},zp={zp}): {got} != {want}"
        assert np.all(np.isfinite(g)) and np.all(np.diff(g) > 0)


def test_lawful_nominal_values_unchanged():
    locs = np.linspace(0.0, 1.0, 201)
    log = cm.nominal_at_location(locs, 1, 100, 0.0)
    scaffold = 1.0 + 99.0 * (100.0 ** locs - 1.0) / 99.0
    assert np.array_equal(log, scaffold)
    assert float(cm.nominal_at_location(0.5, 0, 30)) == 15.0
    assert float(cm.nominal_at_location(0.5, 0, 100, -10.0)) > 0.0


# ---- 6. Downstream callers -------------------------------------------------------------

def test_downstream_integration():
    _raises_named(lambda: cm.percentiles_to_cdf(
        {0.1: 5, 0.5: 50, 0.9: 95}, 0.0, 100.0, zero_point=50.0), "zero_point")
    cdf = cm.percentiles_to_cdf({0.1: 5, 0.5: 10, 0.9: 20}, 1.0, 100.0,
                                zero_point=0.0)
    assert np.all(np.isfinite(cdf)) and np.all(np.diff(cdf) >= 0)
    import metaculus_client as mc
    _raises_named(lambda: mc._nominal_at_location(0.5, 0.0, 100.0, 50.0),
                  "zero_point")
    assert mc._nominal_at_location(0.5, 0.0, 30.0, None) == 15.0


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for fn in ALL_TESTS:
        try:
            fn()
            check(fn.__name__, True)
        except AssertionError as e:
            check(f"{fn.__name__}: {e}", False)
    print(f"\n{len(ALL_TESTS) - len(FAILURES)}/{len(ALL_TESTS)} tests passed")
    sys.exit(1 if FAILURES else 0)
