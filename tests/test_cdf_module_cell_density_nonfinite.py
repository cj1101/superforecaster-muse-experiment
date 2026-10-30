"""
test_cdf_module_cell_density_nonfinite.py — SF-BL-10110 verification.

Pre-registered done/kill criteria (TASK.md):
  "Fix raises a named ValueError on non-finite y, cdf or nominals (and
   empty grids) for both scorers with finite-input scores bit-identical;
   kill if NaN input still raises a bare IndexError or returns a score."

Bug: cdf_module._cell_density (shared by log_score_from_cdf and
interval_log_score_from_cdf) screened nothing. Comparisons against NaN
are False, so y=NaN fell past the y<=first / y>=last guards into the
searchsorted branch, which places NaN one past the last cell:
'IndexError: index 3 is out of bounds for axis 0 with size 3' on a
3-point grid. y=+/-inf silently returned scores, non-finite cdf /
nominals propagated NaN/inf scores (or finite garbage, e.g. -27.63),
and empty / single-point grids died with bare IndexErrors.

Fix (metaculus_bot/cdf_module.py, _cell_density): before any comparison
or indexing — degenerate (<2-point) or mismatched grids, non-finite y,
non-finite cdf values, non-finite nominals values and non-positive
span all raise a ValueError naming the offending input and SF-BL-10110.
The density/mass computation is untouched, so finite lawful grids are
bit-identical: expected float bit patterns below were recorded from
the pre-fix implementation in the live repro (work dir repro.py).

Run:  python3 tests/test_cdf_module_cell_density_nonfinite.py
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_cdf_module_crps_degenerate_grid.py.
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
NAN, INF = float("nan"), float("inf")
NOM = [0.0, 5.0, 10.0]
CDF = [0.0, 0.5, 1.0]
SCORERS = (cm.log_score_from_cdf, cm.interval_log_score_from_cdf)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _raises_named(fn, *needles):
    """ValueError (never IndexError) whose message names the problem."""
    try:
        fn()
    except IndexError as e:
        raise AssertionError(f"bare IndexError leaked (kill criterion): {e}")
    except ValueError as e:
        msg = str(e)
        for n in needles:
            assert n in msg, f"message missing {n!r}: {msg!r}"
        assert "SF-BL-10110" in msg, f"message missing task id: {msg!r}"
        return msg
    raise AssertionError("no ValueError raised — a score may have leaked")


def _bits(v):
    return struct.pack("<d", float(v)).hex()


# ---- 1. Task repro: y=NaN on a 3-point grid, both scorers -------------------

def test_task_repro_nan_y_raises_named_not_indexerror():
    for fn in SCORERS:
        _raises_named(lambda fn=fn: fn(CDF, NOM, NAN), "finite", "y")


def test_nonfinite_y_any_kind_raises():
    for fn in SCORERS:
        for y in (NAN, INF, -INF, np.float64(NAN), np.float64(INF)):
            _raises_named(lambda fn=fn, y=y: fn(CDF, NOM, y), "finite", "y")


# ---- 2. Non-finite cdf / nominals, any position ------------------------------

def test_nonfinite_cdf_raises():
    bad_cdfs = [
        [NAN, 0.5, 1.0], [0.0, NAN, 1.0], [0.0, 0.5, NAN],
        [INF, 0.5, 1.0], [0.0, INF, 1.0], [0.0, 0.5, -INF],
    ]
    for fn in SCORERS:
        for c in bad_cdfs:
            _raises_named(lambda fn=fn, c=c: fn(c, NOM, 5.0), "finite", "cdf")


def test_nonfinite_nominals_raises():
    bad_noms = [
        [NAN, 5.0, 10.0], [0.0, NAN, 10.0], [0.0, 5.0, NAN],
        [-INF, 5.0, 10.0], [0.0, INF, 10.0], [0.0, 5.0, INF],
    ]
    for fn in SCORERS:
        for n in bad_noms:
            _raises_named(lambda fn=fn, n=n: fn(CDF, n, 5.0),
                          "finite", "nominals")


# ---- 3. Empty / degenerate grids ---------------------------------------------

def test_empty_and_single_point_grids_raise_named():
    for fn in SCORERS:
        _raises_named(lambda fn=fn: fn([], [], 5.0),
                      "empty", "degenerate", "grid")
        _raises_named(lambda fn=fn: fn(np.array([]), np.array([]), 0.5),
                      "empty", "degenerate", "grid")
        _raises_named(lambda fn=fn: fn([0.5], [5.0], 5.0),
                      "degenerate", "grid")
        _raises_named(lambda fn=fn: fn([], [0.0, 10.0], 5.0),
                      "degenerate", "grid")
        _raises_named(lambda fn=fn: fn([0.0, 1.0], [], 5.0),
                      "degenerate", "grid")


def test_mismatched_and_zero_span_grids_raise():
    for fn in SCORERS:
        _raises_named(lambda fn=fn: fn([0.0, 0.5, 1.0], [0.0, 10.0], 5.0),
                      "grid", "mismatch")
        _raises_named(lambda fn=fn: fn([0.0, 1.0], [5.0, 5.0], 5.0),
                      "degenerate", "grid")


def test_cell_density_direct_raises():
    # The shared helper itself is screened, not just the wrappers.
    _raises_named(lambda: cm._cell_density(CDF, NOM, NAN), "finite", "y")
    _raises_named(lambda: cm._cell_density([0.0, NAN, 1.0], NOM, 5.0),
                  "finite", "cdf")
    _raises_named(lambda: cm._cell_density([], [], 5.0),
                  "empty", "degenerate", "grid")


# ---- 4. Kill criterion sweep: no bad input returns a score -------------------

def test_no_bad_input_returns_a_score():
    bad = []
    for y in (NAN, INF, -INF):
        bad.append((CDF, NOM, y))
    for c in ([NAN, 0.5, 1.0], [0.0, INF, 1.0]):
        bad.append((c, NOM, 5.0))
    for n in ([0.0, NAN, 10.0], [0.0, 5.0, INF]):
        bad.append((CDF, n, 5.0))
    bad += [([], [], 5.0), ([0.5], [5.0], 5.0)]
    for fn in SCORERS:
        for c, n, y in bad:
            try:
                v = fn(c, n, y)
            except ValueError:
                continue
            except IndexError as e:
                raise AssertionError(
                    f"{fn.__name__}({c}, {n}, y={y}) leaked IndexError: {e}")
            raise AssertionError(
                f"{fn.__name__}({c}, {n}, y={y}) returned {v!r} "
                "instead of raising (kill criterion)")


# ---- 5. Finite inputs bit-identical (bit patterns from pre-fix repro) --------

# (name, cdf, nominals, y, log bits, interval bits)
FINITE = [
    ("3pt y=5", CDF, NOM, 5.0,
     "1555b5bbb16b02c0", "ef39fafe422ee6bf"),
    ("3pt y=0", CDF, NOM, 0.0,
     "1555b5bbb16b02c0", "ef39fafe422ef6bf"),
    ("3pt y=10", CDF, NOM, 10.0,
     "1555b5bbb16b02c0", "ef39fafe422ee6bf"),
    ("3pt y=-1 below", CDF, NOM, -1.0,
     "1555b5bbb16b02c0", "ef39fafe422ef6bf"),
    ("3pt y=11 above", CDF, NOM, 11.0,
     "1555b5bbb16b02c0", "ef39fafe422ee6bf"),
    ("3pt y=2.5", CDF, NOM, 2.5,
     "1555b5bbb16b02c0", "ef39fafe422ee6bf"),
    ("3pt y=7.5", CDF, NOM, 7.5,
     "1555b5bbb16b02c0", "ef39fafe422ee6bf"),
    ("5pt y=4", [0.1, 0.3, 0.5, 0.7, 0.9], [0.0, 2.5, 5.0, 7.5, 10.0], 4.0,
     "e545c81ca43404c0", "8f1b20c127c0f9bf"),
    ("neg range y=0", [0.0, 0.5, 1.0], [-10.0, 0.0, 10.0], 0.0,
     "91e3737b42f707c0", "ef39fafe422ee6bf"),
    ("2pt y=5", [0.0, 1.0], [0.0, 10.0], 5.0,
     "1555b5bbb16b02c0", "0000000000000000"),
]


def test_finite_panels_bit_identical():
    for name, c, n, y, log_bits, int_bits in FINITE:
        lv = cm.log_score_from_cdf(c, n, y)
        iv = cm.interval_log_score_from_cdf(c, n, y)
        assert _bits(lv) == log_bits, f"{name} log: {_bits(lv)} != {log_bits}"
        assert _bits(iv) == int_bits, f"{name} interval: {_bits(iv)} != {int_bits}"
        assert np.isfinite(lv) and np.isfinite(iv), name


def test_finite_201pt_scaffold_grid_bit_identical():
    nominals = cm.grid_nominals(0, 30)
    cdf = cm.percentiles_to_cdf({0.1: 5, 0.5: 10, 0.9: 20}, 0, 30)
    want = {  # (log bits, interval bits) recorded pre-fix
        0.0: ("8948f8791b3929c0", "652693dae46b02c0"),
        15.0: ("5ca1e8b44ac009c0", "b20183f9cb7614c0"),
        30.0: ("9ffff8791b3929c0", "1489ff8ae36b02c0"),
        7.3: ("2887c7cdcf3404c0", "9774f2850eb111c0"),
    }
    for y, (log_bits, int_bits) in want.items():
        assert _bits(cm.log_score_from_cdf(cdf, nominals, y)) == log_bits, y
        assert _bits(cm.interval_log_score_from_cdf(cdf, nominals, y)) == int_bits, y


def test_client_adapter_raises_and_scores_identically():
    import metaculus_client as mc
    assert mc.log_score_from_cdf is cm.log_score_from_cdf
    assert mc.interval_log_score_from_cdf is cm.interval_log_score_from_cdf
    _raises_named(lambda: mc.log_score_from_cdf(CDF, NOM, NAN), "finite", "y")
    _raises_named(lambda: mc.interval_log_score_from_cdf(CDF, NOM, NAN),
                  "finite", "y")
    assert mc.log_score_from_cdf(CDF, NOM, 5.0) == -2.3025850929940455
    assert mc.interval_log_score_from_cdf(CDF, NOM, 5.0) == -0.6931471805599453


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
