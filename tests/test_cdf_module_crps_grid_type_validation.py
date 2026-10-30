"""
test_cdf_module_crps_grid_type_validation.py — SF-BL-11309 verification.

Pre-registered: "crps_from_cdf raises ValueError naming cdf or nominals
on any bool or string element of either array, with numeric-grid CRPS
values bit-identical; kill if string cdf/nominals still return 12.5 or
a bool cdf still returns 0.0."

Task panel: cdf [0, 0.5, 1], nominals [0, 50, 100], y=50 -> 12.5.
Pre-fix (verified live on all three live copies): string cdf
['0', '0.5', '1'] and string nominals ['0', '50', '100'] each returned
12.5 bit-identical to the numeric control, and bool cdf
[False, True, True] returned a perfect 0.0, because
np.asarray(dtype=float) coerces strings and bools before integration.
Closest priors: SF-BL-11134 (this function's outcome y type),
SF-BL-9924 and SF-BL-10473 (cdf/nominals finiteness and grid value
domain); none fixtured the cdf or nominals type domain.

Run: python3 tests/test_cdf_module_crps_grid_type_validation.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metaculus_bot"))
import numpy as np  # noqa: E402
import cdf_module as cm  # noqa: E402
PASS = []
def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}"); PASS.append(bool(cond))
def _raises_11309(fn, argname):
    try:
        fn()
    except ValueError as e:
        msg = str(e)
        assert "SF-BL-11309" in msg, f"missing SF-BL-11309 tag: {msg!r}"
        assert argname in msg, f"error does not name {argname}: {msg!r}"
        assert "crps_from_cdf" in msg, f"error does not name crps_from_cdf: {msg!r}"
        return msg
    except Exception as e:  # any other exception type is a failure, not a pass
        raise AssertionError(f"raised {type(e).__name__} instead of the named ValueError: {e}")
    raise AssertionError("no ValueError raised — corrupt grid type returned a score (kill)")

CDF = [0, 0.5, 1]
NOM = [0, 50, 100]
crps = cm.crps_from_cdf

# --- Task repro / kill criteria ---
_raises_11309(lambda: crps(["0", "0.5", "1"], NOM, 50), "cdf")
check("test_task_repro_string_cdf_raises_not_125", True)
_raises_11309(lambda: crps(CDF, ["0", "50", "100"], 50), "nominals")
check("test_task_repro_string_nominals_raises_not_125", True)
_raises_11309(lambda: crps([False, True, True], NOM, 50), "cdf")
check("test_task_repro_bool_cdf_raises_not_00", True)

# --- Every bool/string form, both arguments ---
BAD_CDFS = [
    ["0", "0.5", "1"], [0, "0.5", 1], [0, 0.5, "1"],
    [False, True, True], [0, True, 1], [0, False, 1],
    [0, np.bool_(True), 1], [0, np.str_("0.5"), 1], [0, b"0.5", 1],
    np.array([False, True, True]), np.array(["0", "0.5", "1"]),
    np.array([0, "0.5", 1], dtype=object), np.array([0, True, 1], dtype=object),
    ("0", "0.5", "1"), (0, True, 1), "0.5", True, [[0, "0.5"], 1],
]
for i, bad in enumerate(BAD_CDFS):
    _raises_11309(lambda b=bad: crps(b, NOM, 50), "cdf")
check("test_bool_string_cdf_all_forms", True, f"{len(BAD_CDFS)} forms")

BAD_NOMS = [
    ["0", "50", "100"], [0, "50", 100], [0, np.str_("50"), 100],
    [0, True, 100], [False, 50, 100], [0, np.bool_(True), 100], [0, b"50", 100],
    np.array([False, True, True]), np.array(["0", "50", "100"]),
    np.array([0, "50", 100], dtype=object), ("0", "50", "100"), "50", False,
]
for i, bad in enumerate(BAD_NOMS):
    _raises_11309(lambda b=bad: crps(CDF, b, 50), "nominals")
check("test_bool_string_nominals_all_forms", True, f"{len(BAD_NOMS)} forms")

# --- Kill sweep: no corrupt form returns any score ---
leaks = 0
for bad in BAD_CDFS:
    try:
        crps(bad, NOM, 50); leaks += 1
    except ValueError:
        pass
for bad in BAD_NOMS:
    try:
        crps(CDF, bad, 50); leaks += 1
    except ValueError:
        pass
check("test_no_corrupt_type_returns_any_score_kill_sweep", leaks == 0, f"leaks={leaks}")

# --- Numeric grids bit-identical ---
def bits(v):
    return float(v).hex()
check("test_numeric_control_bit_identical", bits(crps(CDF, NOM, 50)) == bits(12.5))
check("test_numeric_panels_bit_identical",
      bits(crps([0, 0.25, 0.5, 0.75, 1.0], [0, 1, 2, 3, 4], 2)) == bits(0.375)
      and bits(crps(CDF, NOM, 0)) == bits(37.5)
      and bits(crps(CDF, NOM, 100)) == bits(12.5))
from decimal import Decimal  # noqa: E402
from fractions import Fraction  # noqa: E402
check("test_numeric_forms_bit_identical",
      bits(crps(tuple(CDF), tuple(NOM), 50)) == bits(12.5)
      and bits(crps(np.array(CDF), np.array(NOM), 50)) == bits(12.5)
      and bits(crps([Decimal("0"), Decimal("0.5"), Decimal("1")],
                    [Fraction(0), Fraction(50), Fraction(100)], 50)) == bits(12.5)
      and bits(crps([0, 1, 1], [0, 50, 100], 50)) == bits(crps([0.0, 1.0, 1.0], [0.0, 50.0, 100.0], 50)))
grid_cdf = np.linspace(0.0, 1.0, 201) ** 2
grid_nom = cm.grid_nominals(0.0, 100.0)
check("test_201pt_scaffold_grid_bit_identical",
      all(bits(crps(grid_cdf, grid_nom, yv)) == bits(crps(list(grid_cdf), list(grid_nom), yv))
          for yv in (0, 15, 50, 100)))

# --- Prior screens preserved ---
def _raises_tag(fn, tag):
    try:
        fn()
    except ValueError as e:
        return tag in str(e)
    except Exception:
        return False
    return False
check("test_prior_screens_preserved",
      _raises_tag(lambda: crps(CDF, NOM, True), "SF-BL-11134")
      and _raises_tag(lambda: crps([0.0, float("nan"), 1.0], NOM, 50), "SF-BL-10473")
      and _raises_tag(lambda: crps(CDF, [100, 50, 0], 50), "SF-BL-9924")
      and _raises_tag(lambda: crps([0, 0.5, 1.5], NOM, 50), "SF-BL-9924")
      and _raises_tag(lambda: crps([0.5], NOM, 50), "SF-BL-10269"))

print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
assert all(PASS), "FAILURES PRESENT"
print("ALL TESTS PASSED")
