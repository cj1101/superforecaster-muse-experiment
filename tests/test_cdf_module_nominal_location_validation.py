"""
test_cdf_module_nominal_location_validation.py — SF-BL-10312 verification.

Pre-registered done/kill criterion (TASK.md):
  "Fix raises a named ValueError naming location for non-finite or
   out-of-[0,1] locations and leaves in-range outputs unchanged;
   kill if location=NaN or location=2.0 returns a numeric nominal
   instead of raising."

Bug: cdf_module.nominal_at_location converted location with
np.asarray(location, dtype=float) and used it unscreened, so
location=NaN returned NaN, location=+/-inf returned inf, and
location=2.0 on [0, 100] returned 200.0 — silent extrapolation off
the scaled [0, 1] axis (log branch: 300.0 with zero_point=-100).
Distinct from SF-BL-10268 (zero_point / deriv_ratio of the same
function) and SF-BL-10108 (grid range/size geometry).

Fix (metaculus_bot/cdf_module.py, mirrored in the
superforecaster_out copy): _checked_location screens location at
entry, before any other screen — any non-finite element or any
element outside [0, 1] raises a ValueError naming location and
SF-BL-10312. Lawful locations pass through as the identical
np.asarray(location, dtype=float) array, so in-range outputs stay
bit-identical (hashes recorded pre-fix, SF-BL-10312 repro).

Run:  python3 tests/test_cdf_module_nominal_location_validation.py
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
    raise AssertionError("returned a nominal instead of raising (kill criterion)")


# ---- 1. Task kill cases -------------------------------------------------------

def test_task_kill_cases():
    # The task's exact live repros:
    _raises_named(lambda: cm.nominal_at_location(float("nan"), 0, 100),
                  "location", "SF-BL-10312")   # was NaN
    _raises_named(lambda: cm.nominal_at_location(2.0, 0, 100),
                  "location", "SF-BL-10312")   # was 200.0


# ---- 2. Non-finite locations ----------------------------------------------------

def test_nonfinite_locations_raise():
    for bad in (float("nan"), float("inf"), float("-inf")):
        _raises_named(lambda b=bad: cm.nominal_at_location(b, 0, 100), "location")
        _raises_named(lambda b=bad: cm.nominal_at_location([0.0, b, 1.0], 0, 100),
                      "location")
        _raises_named(lambda b=bad: cm.nominal_at_location(b, 0, 100, -100),
                      "location")  # log branch


# ---- 3. Out-of-[0,1] locations ----------------------------------------------------

def test_out_of_range_locations_raise():
    for bad in (2.0, -0.5, 1.0000001, -1e-9, 100.0, -100.0):
        _raises_named(lambda b=bad: cm.nominal_at_location(b, 0, 100), "location")
        _raises_named(lambda b=bad: cm.nominal_at_location([0.5, b], 0, 100),
                      "location")
        _raises_named(lambda b=bad: cm.nominal_at_location(b, 0, 100, -100),
                      "location")  # log branch


# ---- 4. Kill sweep: no bad location returns a nominal -----------------------------

def test_no_bad_location_returns_nominals():
    bads = [float("nan"), float("inf"), float("-inf"), 2.0, -0.5,
            1.5, -1.0, 1.0000001]
    n = 0
    for zp in (None, -100.0, 0.0):
        for b in bads:
            n += 1
            _raises_named(lambda b=b, z=zp: cm.nominal_at_location(b, 0, 100, z),
                          "location")
            _raises_named(lambda b=b, z=zp: cm.nominal_at_location(
                [0.0, b, 1.0], 0, 100, z), "location")
    assert n >= 20, n


# ---- 5. In-range outputs unchanged --------------------------------------------------

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


def test_lawful_nominal_values_unchanged():
    # Endpoints are lawful (closed interval) and exact:
    assert float(cm.nominal_at_location(0.0, 0, 100)) == 0.0
    assert float(cm.nominal_at_location(0.5, 0, 100)) == 50.0
    assert float(cm.nominal_at_location(1.0, 0, 100)) == 100.0
    assert float(cm.nominal_at_location(0.5, 0, 30)) == 15.0
    locs = np.linspace(0.0, 1.0, 201)
    assert np.array_equal(cm.nominal_at_location(locs, 0, 100),
                          0.0 + 100.0 * locs)
    log = cm.nominal_at_location(locs, 1, 100, 0.0)
    scaffold = 1.0 + 99.0 * (100.0 ** locs - 1.0) / 99.0
    assert np.array_equal(log, scaffold)


# ---- 6. Downstream callers -------------------------------------------------------------

def test_downstream_integration():
    cdf = cm.percentiles_to_cdf({0.1: 5, 0.5: 50, 0.9: 95}, 0.0, 100.0)
    assert np.all(np.isfinite(cdf)) and np.all(np.diff(cdf) >= 0)
    cdf = cm.percentiles_to_cdf({0.1: 5, 0.5: 10, 0.9: 20}, 1.0, 100.0,
                                zero_point=0.0)
    assert np.all(np.isfinite(cdf)) and np.all(np.diff(cdf) >= 0)
    import metaculus_client as mc
    assert mc._nominal_at_location(0.5, 0.0, 30.0, None) == 15.0
    _raises_named(lambda: mc._nominal_at_location(float("nan"), 0.0, 100.0, None),
                  "location")
    _raises_named(lambda: mc._nominal_at_location(2.0, 0.0, 100.0, None),
                  "location")


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
