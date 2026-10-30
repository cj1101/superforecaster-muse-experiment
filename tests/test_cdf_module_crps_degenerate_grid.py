"""
test_cdf_module_crps_degenerate_grid.py — SF-BL-10269 verification tests.

Pre-registered done/kill criteria (TASK.md):
  "Fix raises ValueError naming the empty/degenerate grid (fewer than 2
   grid points) before integrating, with lawful-grid CRPS values
   bit-identical; kill if any <2-point grid still returns a finite score."

Bug: cdf_module.crps_from_cdf trapezoid-integrated whatever it was
given with no size screen. np.trapz over 0 or 1 grid points is
identically 0.0 — the best possible proper score — so
crps_from_cdf([], [], y=0.5) and every aligned single-point grid
returned exactly 0.0. A <2-point side paired with a longer side also
leaked finite scores via broadcasting (1-point cdf vs 2 nominals ->
2.5; 2-point cdf vs 1 nominal -> 0.0), and the remaining mismatched
empty cases raised a bare numpy broadcast ValueError that never named
the grid. Live pre-fix repro is in the task work dir (repro.py).

Fix (metaculus_bot/cdf_module.py, crps_from_cdf): before integrating,
if either side has size < 2 raise ValueError naming the
empty/degenerate grid (fewer than 2 grid points, both sizes, task id);
a size mismatch with both sides >=2 also raises a named ValueError
(pre-fix it raised a bare broadcast ValueError — type unchanged).
The integrand / trapezoid line is untouched, so lawful grids are
bit-identical: expected float bit patterns below were recorded from
the pre-fix implementation in the live repro.

Distinct from SF-BL-9924 (unsorted nominals / NaN y on a non-degenerate
grid) — nothing here changes behaviour on >=2-point grids.

Run:  python3 tests/test_cdf_module_crps_degenerate_grid.py
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_cdf_module_nonfinite_percentiles.py.
"""
import os
import struct
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metaculus_bot"))

import numpy as np  # noqa: E402
import cdf_module as cm  # noqa: E402

PASS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _raises_named(fn):
    """ValueError whose message names the empty/degenerate grid."""
    try:
        fn()
    except ValueError as e:
        msg = str(e)
        assert "degenerate" in msg, f"message does not name degenerate grid: {msg!r}"
        assert "grid" in msg, f"message does not name the grid: {msg!r}"
        assert "SF-BL-10269" in msg, f"message missing task id: {msg!r}"
        return msg
    raise AssertionError("no ValueError raised — a finite score may have leaked")


def _bits(v):
    return struct.pack("<d", float(v)).hex()


# ---- 1. Task repro: empty grid ---------------------------------------------

def test_task_repro_empty_grid_raises():
    _raises_named(lambda: cm.crps_from_cdf([], [], y=0.5))
    _raises_named(lambda: cm.crps_from_cdf([], [], y=10.0))
    _raises_named(lambda: cm.crps_from_cdf(np.array([]), np.array([]), y=0.5))


# ---- 2. Single-point grids, every flavour -----------------------------------

def test_single_point_grid_raises():
    for c, n, y in (
        ([0.5], [5.0], 0.5),
        ([0.0], [0.0], 0.5),
        ([1.0], [10.0], 5.0),
        ((0.5,), (5.0,), 0.5),
        (np.array([0.3]), np.array([2.0]), 1.0),
        ([0.25], [-3.0], -3.0),
    ):
        _raises_named(lambda c=c, n=n, y=y: cm.crps_from_cdf(c, n, y))


def test_no_degenerate_grid_returns_finite_score():
    # Kill criterion sweep: no <2-point grid, at any y, may return a
    # finite (or any) score — every call must raise.
    grids = [
        ([], []), ([0.5], [5.0]), ([], [0.0, 10.0]), ([0.0, 1.0], []),
        ([0.5], [0.0, 10.0]), ([0.0, 1.0], [5.0]),
    ]
    for c, n in grids:
        for y in (-1.0, 0.0, 0.5, 5.0, 10.0, 1e6):
            try:
                v = cm.crps_from_cdf(c, n, y)
            except ValueError:
                continue
            raise AssertionError(
                f"crps_from_cdf({c}, {n}, y={y}) returned {v!r} "
                "instead of raising (kill criterion)")


def test_one_sided_degenerate_raises_named_not_broadcast():
    # Pre-fix these either leaked a finite score (2.5 / 0.0) or raised a
    # bare "operands could not be broadcast" ValueError; the fix must
    # name the empty/degenerate grid in every case.
    for c, n in (([], [0.0, 10.0]), ([0.0, 1.0], []),
                 ([0.5], [0.0, 10.0]), ([0.0, 1.0], [5.0])):
        msg = _raises_named(lambda c=c, n=n: cm.crps_from_cdf(c, n, 5.0))
        assert "broadcast" not in msg, msg


def test_scalar_inputs_raise_named():
    # 0-d inputs have size 1 (<2): pre-fix they died inside np.diff with
    # an unrelated message; now they raise the named grid error.
    _raises_named(lambda: cm.crps_from_cdf(0.5, 5.0, 0.5))


def test_mismatched_grid_both_ge2_raises_value_error():
    try:
        cm.crps_from_cdf([0.0, 0.5, 1.0], [0.0, 10.0], 5.0)
    except ValueError as e:
        assert "grid" in str(e), str(e)
        return
    raise AssertionError("mismatched >=2 grid did not raise")


# ---- 3. Lawful grids bit-identical (bit patterns from pre-fix repro) --------

# (name, cdf, nominals, y, pre-fix float bits hex)
LAWFUL = [
    ("two-point step y inside", [0.0, 1.0], [0.0, 10.0], 5.0,
     "0000000000000000"),
    ("two-point y below", [0.2, 0.8], [0.0, 10.0], -1.0,
     "3533333333330b40"),
    ("two-point y above", [0.2, 0.8], [0.0, 10.0], 11.0,
     "3533333333330b40"),
    ("two-point y at edge", [0.0, 1.0], [0.0, 10.0], 0.0,
     "0000000000001440"),
    ("three-point", [0.0, 0.5, 1.0], [0.0, 5.0, 10.0], 5.0,
     "000000000000f43f"),
    ("uniform-ish 5pt", [0.1, 0.3, 0.5, 0.7, 0.9],
     [0.0, 2.5, 5.0, 7.5, 10.0], 4.0, "9a9999999999f13f"),
    ("negative range", [0.0, 0.5, 1.0], [-10.0, 0.0, 10.0], 0.0,
     "0000000000000440"),
]


def test_lawful_grids_bit_identical():
    for name, c, n, y, want_bits in LAWFUL:
        v = cm.crps_from_cdf(c, n, y)
        assert _bits(v) == want_bits, f"{name}: {_bits(v)} != {want_bits}"
        assert np.isfinite(v), name


def test_lawful_two_point_grid_may_legitimately_score_zero():
    # Exactly 2 points is lawful: a perfect step at y scores a genuine
    # 0.0 and must NOT raise — the guard is <2, not <=2.
    assert cm.crps_from_cdf([0.0, 1.0], [0.0, 10.0], 5.0) == 0.0


def test_lawful_201pt_scaffold_grid_bit_identical():
    nominals = cm.grid_nominals(0, 30)
    cdf = cm.percentiles_to_cdf({0.1: 5, 0.5: 10, 0.9: 20}, 0, 30)
    v = cm.crps_from_cdf(cdf, nominals, 15.0)
    assert _bits(v) == "aed3ed91725c0540", _bits(v)  # pre-fix bits
    # Independent recomputation of the same trapezoid integral.
    integrand = (np.asarray(cdf) - (np.asarray(nominals) >= 15.0)) ** 2
    assert _bits(float(np.trapz(integrand, np.asarray(nominals)))) == _bits(v)


def test_client_adapter_raises_and_scores_identically():
    import metaculus_client as mc
    assert mc.crps_from_cdf is cm.crps_from_cdf
    _raises_named(lambda: mc.crps_from_cdf([], [], 0.5))
    _raises_named(lambda: mc.crps_from_cdf([0.5], [5.0], 0.5))
    assert mc.crps_from_cdf([0.0, 0.5, 1.0], [0.0, 5.0, 10.0], 5.0) == 1.25


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
