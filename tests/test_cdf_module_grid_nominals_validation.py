"""
test_cdf_module_grid_nominals_validation.py — SF-BL-10108 verification.

Pre-registered done/kill criteria (TASK.md):
  "Fix raises a named ValueError for range_max<=range_min, non-finite
   endpoints, or size<2, with all standard 201-point grids
   bit-identical; kill if any invalid geometry still returns a grid
   or a valid grid changes."

Bug: cdf_module.grid_nominals forwarded range_min/range_max/size
unchecked into np.linspace and nominal_at_location's scaling formula.
Live pre-fix repro (task work dir repro.py): grid_nominals(30, 0)
returned a silently DESCENDING 201-point grid [30, 29.85, ..., 0],
grid_nominals(0, 30, size=0) an EMPTY grid, size=1 a single point,
equal endpoints a constant grid, NaN/inf endpoints all-NaN/inf grids,
zero_point inside the range a NaN grid, and zero_point == range_min a
bare ZeroDivisionError — while every interp/scoring caller assumes a
strictly increasing grid.

Fix (metaculus_bot/cdf_module.py): _validate_grid_geometry screens
the geometry before any grid is built — non-finite endpoints,
range_max <= range_min, non-finite zero_point, and zero_point inside
[range_min, range_max] (deriv_ratio <= 0 / division by zero) all raise
a ValueError naming the problem and SF-BL-10108 — and grid_nominals
rejects size < 2 with a named ValueError before np.linspace. Valid
geometry passes through untouched: the expected SHA-256 hashes below
were recorded from the PRE-FIX implementation in the live repro.

Run:  python3 tests/test_cdf_module_grid_nominals_validation.py
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_cdf_module_nonfinite_percentiles.py.
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metaculus_bot"))

import numpy as np  # noqa: E402
import cdf_module as cm  # noqa: E402

NAN, INF = float("nan"), float("inf")
PASS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _raises_named(fn, *needles):
    """ValueError whose message names the geometry problem + task id."""
    try:
        fn()
    except ValueError as e:
        msg = str(e)
        for needle in needles:
            assert needle in msg, f"message missing {needle!r}: {msg!r}"
        assert "SF-BL-10108" in msg, f"message missing task id: {msg!r}"
        return msg
    raise AssertionError("no ValueError raised — an invalid grid may "
                         "have been returned")


# ---- 1. Task repros ----------------------------------------------------------

def test_task_repro_reversed_range_raises():
    _raises_named(lambda: cm.grid_nominals(30, 0), "range_max > range_min")


def test_task_repro_size_zero_raises():
    _raises_named(lambda: cm.grid_nominals(0, 30, size=0), "size")


# ---- 2. range_max <= range_min ------------------------------------------------

def test_reversed_and_equal_ranges_raise():
    for lo, hi in ((30, 0), (10, 10), (0, 0), (-1, -30), (-5, -5),
                   (1e6, 1.0), (0.5, 0.25)):
        _raises_named(lambda lo=lo, hi=hi: cm.grid_nominals(lo, hi),
                      "range_max > range_min")
    # Log-scaled reversed range raises the same named error.
    _raises_named(lambda: cm.grid_nominals(100, 1, zero_point=0.0),
                  "range_max > range_min")


# ---- 3. Non-finite endpoints ---------------------------------------------------

def test_nonfinite_endpoints_raise():
    for lo, hi in ((NAN, 30), (0, NAN), (NAN, NAN), (0, INF), (-INF, 30),
                   (INF, INF), (-INF, INF), (INF, 30)):
        _raises_named(lambda lo=lo, hi=hi: cm.grid_nominals(lo, hi),
                      "finite")


# ---- 4. size < 2 -----------------------------------------------------------------

def test_size_below_two_raises():
    for size in (0, 1, -1, -5, -201):
        _raises_named(lambda s=size: cm.grid_nominals(0, 30, size=s),
                      "size", ">= 2")
    # Also on the log-scale path.
    _raises_named(lambda: cm.grid_nominals(1, 100, zero_point=0.0, size=0),
                  "size")


# ---- 5. zero_point geometry (defence in depth) -----------------------------------

def test_zero_point_invalid_geometry_raises():
    _raises_named(lambda: cm.grid_nominals(0, 30, zero_point=NAN),
                  "zero_point", "finite")
    _raises_named(lambda: cm.grid_nominals(0, 30, zero_point=INF),
                  "zero_point", "finite")
    # Inside the range the deriv_ratio is <= 0: pre-fix an all-NaN grid.
    _raises_named(lambda: cm.grid_nominals(0, 30, zero_point=15.0),
                  "zero_point")
    # == range_min was a bare ZeroDivisionError pre-fix; == range_max a
    # degenerate two-value grid. Both are named ValueErrors now.
    _raises_named(lambda: cm.grid_nominals(0, 30, zero_point=0.0),
                  "zero_point")
    _raises_named(lambda: cm.grid_nominals(0, 30, zero_point=30.0),
                  "zero_point")


# ---- 6. Kill-criterion sweep: no invalid geometry returns a grid -----------------

def test_no_invalid_geometry_returns_a_grid():
    cases = []
    for lo, hi in ((30, 0), (10, 10), (NAN, 30), (0, INF), (-INF, NAN)):
        for size in (201, 2, 1, 0, -3):
            cases.append((lo, hi, None, size))
    for zp in (NAN, INF, 15.0, 0.0, 30.0):
        cases.append((0, 30, zp, 201))
    cases.append((100, 1, 0.0, 201))
    cases.append((1, 100, 0.0, 0))
    for lo, hi, zp, size in cases:
        try:
            g = cm.grid_nominals(lo, hi, zero_point=zp, size=size)
        except ValueError:
            continue
        raise AssertionError(
            f"grid_nominals({lo}, {hi}, zero_point={zp}, size={size}) "
            f"returned a grid of len {len(g)} instead of raising "
            "(kill criterion)")


# ---- 7. Valid grids bit-identical (SHA-256 from the pre-fix repro) ---------------

# (name, args, kwargs, sha256 of grid bytes, pre-fix)
PANELS = [
    ("std 0-30 x201", (0, 30), {},
     "6efb6b7ea9ff2873495ae29061a550d60d3a162fe903125753aa9fb342f8a04f"),
    ("std 0-1000 x201", (0, 1000), {},
     "3eae5611dcd490997032b119d5c6e25d24fb0858d49941df7329a43ca50579c4"),
    ("neg -30-30 x201", (-30, 30), {},
     "bb879fe3576c0195dd0e14db5b544493590569d42aa2a1651739a149f1b8a118"),
    ("log 1-100 zp=0 x201", (1, 100), {"zero_point": 0.0},
     "8cd3949bfd85ee8e2832952bec39d7e789d658edaae37be14ae61871ab3f2986"),
    ("log 10-100000 zp=0 x201", (10, 100000), {"zero_point": 0.0},
     "c3305e0fcc96503ff7f155b223729a365170da1473839ae3fb8e1e40a3c675e2"),
    ("log neg -1000--10 zp=0 x201", (-1000, -10), {"zero_point": 0.0},
     "f37d268ceda1155b24f8a929e5179807b27fe7bc6494ca0e6f9d0708ffa47fbe"),
    ("frac 0.5-2.5 x201", (0.5, 2.5), {},
     "1e446e8d96ba7c6ce36e205a4a24db685a070c1c7511b8b55012ebd2f49ad782"),
    ("small size=2", (0, 30), {"size": 2},
     "c47c6be87d2d68fb467f5770b704c2f3eb939a4cd9a976bcd0279e3b57e1160e"),
    ("small size=3", (0, 30), {"size": 3},
     "8f72f9b1bbc449e16172986557ad281c036c3e7cda890419cbacc7adb129db9e"),
    ("size=11 log", (1, 100), {"zero_point": 0.0, "size": 11},
     "cb6dcfd48228ef3facea362c62f3271d1975d79d751a6ddd15c558e5d69ae46f"),
]


def test_valid_grids_bit_identical():
    for name, args, kwargs, want in PANELS:
        g = np.asarray(cm.grid_nominals(*args, **kwargs), dtype=float)
        got = hashlib.sha256(g.tobytes()).hexdigest()
        assert got == want, f"{name}: {got} != {want} (valid grid changed)"
        assert np.all(np.isfinite(g)), name


def test_valid_grids_strictly_increasing_with_exact_endpoints():
    for args, kwargs in (((0, 30), {}), ((-30, 30), {}),
                         ((1, 100), {"zero_point": 0.0}),
                         ((0, 30), {"size": 2}), ((0, 30), {"size": 7})):
        g = np.asarray(cm.grid_nominals(*args, **kwargs), dtype=float)
        size = kwargs.get("size", cm.CDF_SIZE)
        assert len(g) == size, (args, kwargs, len(g))
        assert g[0] == float(args[0]) and g[-1] == float(args[1])
        assert np.all(np.diff(g) > 0), (args, kwargs)
    # size=2 is lawful (guard is <2): exactly the two endpoints.
    assert cm.grid_nominals(0, 30, size=2).tolist() == [0.0, 30.0]


# ---- 8. nominal_at_location: same guard, values unchanged -------------------------

def test_nominal_at_location_values_unchanged():
    locs = np.linspace(0.0, 1.0, 201)
    lin = cm.nominal_at_location(locs, 0, 30)
    assert np.array_equal(lin, 0.0 + 30.0 * locs)
    # Log scale 1-100 zp=0: independent recomputation of the scaffold
    # formula (deriv_ratio = 100) must match exactly, and the result
    # must sit within float rounding of the closed form 100**l.
    log = cm.nominal_at_location(locs, 1, 100, 0.0)
    scaffold = 1.0 + 99.0 * (100.0 ** locs - 1.0) / 99.0
    assert np.array_equal(log, scaffold)
    assert np.max(np.abs(log - 100.0 ** locs)) < 1e-12
    assert float(cm.nominal_at_location(0.5, 0, 30)) == 15.0


def test_nominal_at_location_invalid_geometry_raises():
    _raises_named(lambda: cm.nominal_at_location(0.5, 30, 0),
                  "range_max > range_min")
    _raises_named(lambda: cm.nominal_at_location(0.5, NAN, 30), "finite")
    _raises_named(lambda: cm.nominal_at_location(0.5, 0, 30, 15.0),
                  "zero_point")


# ---- 9. Downstream callers ---------------------------------------------------------

def test_percentiles_to_cdf_integration():
    # Valid panel CDF unchanged (hash from SF-BL-10107, pre-fix HEAD).
    cdf = cm.percentiles_to_cdf({0.1: 5, 0.5: 10, 0.9: 20}, 0, 30)
    got = hashlib.sha256(np.asarray(cdf).tobytes()).hexdigest()
    assert got == ("3d1b186d416e50e2feac16b247071f216de7d0f1adc6bf964145"
                   "e5ebf10c6b19"), got
    # A reversed range can no longer build a CDF on a descending grid.
    _raises_named(
        lambda: cm.percentiles_to_cdf({0.1: 5, 0.5: 10, 0.9: 20}, 30, 0),
        "range_max > range_min")
    _raises_named(
        lambda: cm.percentiles_to_cdf({0.1: 5, 0.5: 10, 0.9: 20}, 0, 30,
                                      size=1),
        "size")


def test_client_adapter_raises_and_matches():
    import metaculus_client as mc
    assert mc.grid_nominals is cm.grid_nominals
    _raises_named(lambda: mc.grid_nominals(30, 0), "range_max > range_min")
    _raises_named(lambda: mc.grid_nominals(0, 30, size=0), "size")
    g = np.asarray(mc.grid_nominals(0, 30), dtype=float)
    assert hashlib.sha256(g.tobytes()).hexdigest() == PANELS[0][3]
    assert mc._nominal_at_location(0.5, 0.0, 30.0, None) == 15.0
    _raises_named(lambda: mc._nominal_at_location(0.5, 30.0, 0.0, None),
                  "range_max > range_min")


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
