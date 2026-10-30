"""
test_cdf_module_cell_density_grid_type_validation.py — SF-BL-11316 verification.

Pre-registered: "log_score_from_cdf and interval_log_score_from_cdf raise
ValueError naming cdf or nominals on any bool or string element of either
array, with numeric-grid scores bit-identical; kill if string cdf/nominals
still return -4.6052 or a bool cdf still returns -3.9121."

Task panel: cdf [0, 0.5, 1], nominals [0, 50, 100], y=50 -> log score
-4.605170185988091, interval score -0.6931471805599453.
Pre-fix (verified live on all three live copies): string cdf
['0', '0.5', '1'] and string nominals ['0', '50', '100'] each returned
-4.605170185988091 bit-identical to the numeric control, and bool cdf
[False, True, True] returned -3.9120730066781877, because
np.asarray(dtype=float) in _cell_density coerces strings and bools before
the SF-BL-10110 finiteness screens, which pass the coerced values.
Closest priors: SF-BL-10110 (cdf/nominals finiteness — value domain),
SF-BL-11203 (this path's outcome y type), SF-BL-11242 (its floor_mass
validation); none fixtured the cdf or nominals type domain.
SF-BL-11309 fixtures the same class for crps_from_cdf, a different function.

Run: python3 tests/test_cdf_module_cell_density_grid_type_validation.py
No pytest dependency: test_* functions with assert + __main__ runner.
"""
import os
import struct
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_repo_bot = os.path.join(_here, "..", "..", "metaculus_bot")
if "CDF_MODULE_DIR" in os.environ:
    _default = os.environ["CDF_MODULE_DIR"]
elif os.path.exists(os.path.join(_repo_bot, "cdf_module.py")):
    _default = _repo_bot
elif os.path.exists(os.path.join(_here, "cdf_module.py")):
    _default = _here  # work-dir copy alongside repro (unused normally)
else:
    _default = os.path.expanduser("~/workspace/superforecaster/metaculus_bot")
sys.path.insert(0, _default)

import numpy as np  # noqa: E402
import cdf_module as cm  # noqa: E402

PASS = []
CDF = [0, 0.5, 1]
NOM = [0, 50, 100]
SCORERS = (cm.log_score_from_cdf, cm.interval_log_score_from_cdf)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _bits(v):
    return struct.pack("<d", float(v)).hex()


def _raises_11316(fn, argname):
    try:
        fn()
    except ValueError as e:
        msg = str(e)
        assert "SF-BL-11316" in msg, f"missing SF-BL-11316 tag: {msg!r}"
        assert argname in msg, f"error does not name {argname}: {msg!r}"
        assert "log_score_from_cdf" in msg, f"error does not name scorer: {msg!r}"
        return msg
    except Exception as e:
        raise AssertionError(
            f"raised {type(e).__name__} instead of the named ValueError: {e}")
    raise AssertionError("no ValueError raised — corrupt grid type returned a score (kill)")


# --- Task repro / kill criteria -------------------------------------------

def test_task_repro_string_cdf_raises_not_minus46052():
    for fn in SCORERS:
        _raises_11316(lambda fn=fn: fn(["0", "0.5", "1"], NOM, 50), "cdf")
    assert cm.log_score_from_cdf(CDF, NOM, 50) == -4.605170185988091


def test_task_repro_string_nominals_raises_not_minus46052():
    for fn in SCORERS:
        _raises_11316(lambda fn=fn: fn(CDF, ["0", "50", "100"], 50), "nominals")
    assert cm.interval_log_score_from_cdf(CDF, NOM, 50) == -0.6931471805599453


def test_task_repro_bool_cdf_raises_not_minus39121():
    for fn in SCORERS:
        _raises_11316(lambda fn=fn: fn([False, True, True], NOM, 50), "cdf")


def test_cell_density_direct_raises():
    _raises_11316(lambda: cm._cell_density(["0", "0.5", "1"], NOM, 50), "cdf")
    _raises_11316(lambda: cm._cell_density(CDF, ["0", "50", "100"], 50), "nominals")
    _raises_11316(lambda: cm._cell_density([False, True, True], NOM, 50), "cdf")
    _raises_11316(lambda: cm._cell_density(CDF, [False, True, True], 50), "nominals")


# --- Every bool/string form, both arguments --------------------------------

BAD_CDFS = [
    ["0", "0.5", "1"], [0, "0.5", 1], [0, 0.5, "1"],
    [False, True, True], [0, True, 1], [0, False, 1],
    [0, np.bool_(True), 1], [0, np.str_("0.5"), 1], [0, b"0.5", 1],
    np.array([False, True, True]), np.array(["0", "0.5", "1"]),
    np.array([0, "0.5", 1], dtype=object), np.array([0, True, 1], dtype=object),
    ("0", "0.5", "1"), (0, True, 1), "0.5", True, [[0, "0.5"], 1],
]
BAD_NOMS = [
    ["0", "50", "100"], [0, "50", 100], [0, np.str_("50"), 100],
    [0, True, 100], [False, 50, 100], [0, np.bool_(True), 100], [0, b"50", 100],
    np.array([False, True, True]), np.array(["0", "50", "100"]),
    np.array([0, "50", 100], dtype=object), ("0", "50", "100"), "50", False,
]


def test_bool_string_cdf_all_forms():
    for bad in BAD_CDFS:
        for fn in SCORERS:
            _raises_11316(lambda bad=bad, fn=fn: fn(bad, NOM, 50), "cdf")


def test_bool_string_nominals_all_forms():
    for bad in BAD_NOMS:
        for fn in SCORERS:
            _raises_11316(lambda bad=bad, fn=fn: fn(CDF, bad, 50), "nominals")


def test_no_corrupt_type_returns_any_score_kill_sweep():
    leaks = []
    for bad in BAD_CDFS:
        for fn in SCORERS:
            try:
                v = fn(bad, NOM, 50)
                leaks.append((fn.__name__, repr(bad), v))
            except ValueError:
                pass
    for bad in BAD_NOMS:
        for fn in SCORERS:
            try:
                v = fn(CDF, bad, 50)
                leaks.append((fn.__name__, repr(bad), v))
            except ValueError:
                pass
    assert not leaks, f"kill sweep leaks: {leaks}"


def test_bool_string_generator_forms():
    # one-shot iterables: a fresh generator per call
    for fn in SCORERS:
        _raises_11316(lambda fn=fn: fn(
            (x for x in ["0", "0.5", "1"]), NOM, 50), "cdf")
        _raises_11316(lambda fn=fn: fn(
            CDF, (x for x in ["0", "50", "100"]), 50), "nominals")


def test_type_error_precedence_over_other_screens():
    # Entry screen: type error named even when y / floor / grid also bad.
    _raises_11316(lambda: cm.log_score_from_cdf(
        ["0", "0.5", "1"], NOM, float("nan")), "cdf")
    _raises_11316(lambda: cm._cell_density(
        ["0", "0.5", "1"], NOM, 50, floor_mass=True), "cdf")
    _raises_11316(lambda: cm.log_score_from_cdf([0.5], ["0"], 50), "nominals")


# --- Numeric grids bit-identical (bit patterns recorded pre-fix) ------------

FINITE = [
    ("task y=50", CDF, NOM, 50, "1555b5bbb16b12c0", "ef39fafe422ee6bf"),
    ("y=0", CDF, NOM, 0, "1555b5bbb16b12c0", "ef39fafe422ef6bf"),
    ("y=100", CDF, NOM, 100, "1555b5bbb16b12c0", "ef39fafe422ee6bf"),
    ("y=-1 below", CDF, NOM, -1, "1555b5bbb16b12c0", "ef39fafe422ef6bf"),
    ("y=101 above", CDF, NOM, 101, "1555b5bbb16b12c0", "ef39fafe422ee6bf"),
    ("y=25", CDF, NOM, 25, "1555b5bbb16b12c0", "ef39fafe422ee6bf"),
    ("y=75", CDF, NOM, 75, "1555b5bbb16b12c0", "ef39fafe422ee6bf"),
    ("3pt y=5", [0.0, 0.5, 1.0], [0.0, 5.0, 10.0], 5.0,
     "1555b5bbb16b02c0", "ef39fafe422ee6bf"),
    ("5pt y=4", [0.1, 0.3, 0.5, 0.7, 0.9], [0.0, 2.5, 5.0, 7.5, 10.0], 4.0,
     "e545c81ca43404c0", "8f1b20c127c0f9bf"),
    ("neg range y=0", [0.0, 0.5, 1.0], [-10.0, 0.0, 10.0], 0.0,
     "91e3737b42f707c0", "ef39fafe422ee6bf"),
    ("2pt y=5", [0.0, 1.0], [0.0, 10.0], 5.0,
     "1555b5bbb16b02c0", "0000000000000000"),
    ("11242 panel y=1", [0.1, 0.5, 0.9], [0, 1, 2], 1.0,
     "2efd45830c52edbf", "2efd45830c52edbf"),
]


def test_numeric_panels_bit_identical():
    for name, c, n, y, lb, ib in FINITE:
        assert _bits(cm.log_score_from_cdf(c, n, y)) == lb, name
        assert _bits(cm.interval_log_score_from_cdf(c, n, y)) == ib, name


def test_numeric_forms_bit_identical():
    from decimal import Decimal
    from fractions import Fraction
    want_log = _bits(-4.605170185988091)
    forms = [
        (tuple(CDF), tuple(NOM)),
        (np.array(CDF), np.array(NOM)),
        (np.array([0, 1, 1]), np.array(NOM)),  # ints are lawful; checked below separately
        ([Decimal("0"), Decimal("0.5"), Decimal("1")],
         [Fraction(0), Fraction(50), Fraction(100)]),
        (np.array([0, 0.5, 1], dtype=object), NOM),
        (CDF, range(0, 101, 50)),
    ]
    # first, third-index-excluded forms must match the control exactly
    for c, n in (forms[0], forms[1], forms[3], forms[4], forms[5]):
        assert _bits(cm.log_score_from_cdf(c, n, 50)) == want_log, (c, n)
    # int grid [0,1,1] legitimately equals its float twin, incl. the
    # pre-fix bool-cdf value -3.9120730066781877 when genuinely numeric
    assert cm.log_score_from_cdf([0, 1, 1], NOM, 50) == \
        cm.log_score_from_cdf([0.0, 1.0, 1.0], NOM, 50) == -3.9120730066781877
    # a numeric generator materialises to the same score
    assert _bits(cm.log_score_from_cdf(
        (x for x in CDF), (x for x in NOM), 50)) == want_log


def test_201pt_scaffold_grid_bit_identical():
    nominals = cm.grid_nominals(0, 30)
    cdf = cm.percentiles_to_cdf({0.1: 5, 0.5: 10, 0.9: 20}, 0, 30)
    want = {
        0.0: ("8948f8791b3929c0", "652693dae46b02c0"),
        15.0: ("5ca1e8b44ac009c0", "b20183f9cb7614c0"),
        30.0: ("9ffff8791b3929c0", "1489ff8ae36b02c0"),
        7.3: ("2887c7cdcf3404c0", "9774f2850eb111c0"),
    }
    for y, (lb, ib) in want.items():
        assert _bits(cm.log_score_from_cdf(cdf, nominals, y)) == lb, y
        assert _bits(cm.interval_log_score_from_cdf(cdf, nominals, y)) == ib, y


# --- Prior screens preserved (repo / integrated copies) ---------------------

def test_prior_screens_preserved():
    def tag_of(fn):
        try:
            fn()
        except ValueError as e:
            return str(e)
        return ""
    # SF-BL-10110 finiteness / degenerate classes still raise with their tag
    assert "SF-BL-10110" in tag_of(
        lambda: cm.log_score_from_cdf(CDF, NOM, float("nan")))
    assert "SF-BL-10110" in tag_of(
        lambda: cm.log_score_from_cdf([0.0, float("nan"), 1.0], NOM, 50))
    assert "SF-BL-10110" in tag_of(
        lambda: cm.log_score_from_cdf(CDF, [0.0, float("nan"), 100.0], 50))
    assert "SF-BL-10110" in tag_of(lambda: cm.log_score_from_cdf([], [], 50))
    # SF-BL-11242 floor_mass class still raises with its tag on a clean grid
    assert "SF-BL-11242" in tag_of(
        lambda: cm._cell_density(CDF, NOM, 50, floor_mass=True))
    # SF-BL-11309 crps grid-type screen untouched
    assert "SF-BL-11309" in tag_of(
        lambda: cm.crps_from_cdf(["0", "0.5", "1"], NOM, 50))


def test_client_adapter_raises_and_scores_identically():
    try:
        import metaculus_client as mc
    except ImportError:
        return  # wave3 copy has no client adapter; repo/out copies do
    assert mc.log_score_from_cdf is cm.log_score_from_cdf
    assert mc.interval_log_score_from_cdf is cm.interval_log_score_from_cdf
    _raises_11316(lambda: mc.log_score_from_cdf(["0", "0.5", "1"], NOM, 50), "cdf")
    _raises_11316(lambda: mc.interval_log_score_from_cdf(CDF, ["0", "50", "100"], 50), "nominals")
    assert mc.log_score_from_cdf(CDF, NOM, 50) == -4.605170185988091
    assert mc.interval_log_score_from_cdf(CDF, NOM, 50) == -0.6931471805599453


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
