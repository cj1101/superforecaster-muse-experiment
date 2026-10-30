"""SF-BL-11057 — select_extremize_factor must type-validate its grid.

Pre-fix, _require_valid_select_grid coerced every entry with float(f)
unchecked, so a bool became 1.0/0.0 and a numeric string its parsed
value before any screen saw a type: on pooled [0.7, 0.3] vs outcomes
[1, 0], grid=[True] returned (1.0, 0.09) — the bool True certified as
the optimal factor — and grid=['2.0', '0.8'] returned (2.0, 0.0241).
These tests pin the fixed contract: any bool/string grid entry raises
a ValueError naming the grid and SF-BL-11057, and numeric grids return
bit-identical (factor, brier) pairs.
"""
import numpy as np
import pytest

from superforecaster.agents import select_extremize_factor

POOLED = np.array([0.7, 0.3])
OUTCOMES = np.array([1, 0])


@pytest.mark.parametrize("bad_grid", [
    [True, False], ["2.0", "0.8"], [True], [False], [True, True],
    [1.0, True], [1.0, "1.3"], ["1.0"], ["abc"], [b"1.3"],
    [np.bool_(True)], [np.str_("1.3")], [None], [complex(1.0, 0.0)],
    np.array([True, False]), np.array(["2.0", "0.8"]),
    np.array([1.0, True], dtype=object), "1.3", 1.3,
    # round 2 (VERDICT round-1 gaps): carrier forms float() parses
    # exactly like the direct bool/string forms above
    [bytearray(b"2.0"), bytearray(b"0.8")], [bytearray(b"1.3")],
    [1.0, bytearray(b"1.3")], [memoryview(b"1.3")],
    [memoryview(bytearray(b"1.3"))],
    [np.array(True, dtype=object)], [np.array("1.3", dtype=object)],
    [np.array(b"1.3", dtype=object)], [np.array(["1.3"], dtype=object)],
    # SF-BL-11471: the numeric-string grid dtype, fixtured separately
    # from the bool entries above — the exact TASK kill grid
    ["0.8", "1.3"], ("0.8", "1.3"), np.array(["0.8", "1.3"]),
    np.array(["0.8", "1.3"], dtype=object), ["0.8"], [1.3, "0.8"],
])
def test_select_rejects_bool_or_string_grid(bad_grid):
    with pytest.raises(ValueError, match="SF-BL-11057"):
        select_extremize_factor(POOLED, OUTCOMES, grid=bad_grid)


def test_task_repro_no_certified_bool_or_string():
    with pytest.raises(ValueError, match="grid"):
        select_extremize_factor(POOLED, OUTCOMES, grid=[True, False])  # pre-fix pair: (1.0, 0.09) via [True]
    with pytest.raises(ValueError, match="grid"):
        select_extremize_factor(POOLED, OUTCOMES, grid=["2.0", "0.8"])  # pre-fix: (2.0, 0.02407847800237812)
    with pytest.raises(ValueError):
        select_extremize_factor(POOLED, OUTCOMES, grid=[True])          # pre-fix: (1.0, 0.09)


def test_sf_bl_11471_numeric_string_grid_raises():
    # SF-BL-11471 kill grid: pre-fix (unguarded site) grid=['0.8','1.3']
    # returned (0.8, 0.2251) instead of raising; the numeric twin of the
    # same grid stays lawful and bit-identical.
    with pytest.raises(ValueError, match="grid"):
        select_extremize_factor(POOLED, OUTCOMES, grid=["0.8", "1.3"])
    with pytest.raises(ValueError, match="SF-BL-11471"):
        select_extremize_factor(POOLED, OUTCOMES, grid=["0.8", "1.3"])
    assert select_extremize_factor(POOLED, OUTCOMES, grid=[0.8, 1.3]) == (1.3, 0.06223095924687406)


def test_numeric_grids_unchanged():
    assert select_extremize_factor(POOLED, OUTCOMES, grid=[1.0, 1.3]) == (1.3, 0.06223095924687406)
    assert select_extremize_factor(POOLED, OUTCOMES, grid=[0.8, 1.0, 1.85]) == (1.85, 0.029781416118895683)
    assert select_extremize_factor(POOLED, OUTCOMES, grid=[1, 2]) == (2.0, 0.02407847800237812)
    assert select_extremize_factor(POOLED, OUTCOMES, grid=np.array([0.8, 1.0])) == (1.0, 0.09000000000000001)
    assert select_extremize_factor(POOLED, OUTCOMES) == (2.0, 0.02407847800237812)
    f, _ = select_extremize_factor(POOLED, OUTCOMES, grid=(g for g in [0.9, 1.1]))
    assert f in (0.9, 1.1)
    # round 2: lawful numeric carriers stay lawful / bit-identical
    assert select_extremize_factor(POOLED, OUTCOMES, grid=[np.array(1.3), 1.0]) == (1.3, 0.06223095924687406)
    assert select_extremize_factor(POOLED, OUTCOMES, grid=np.array([[1.0], [1.3]])) == (1.3, 0.06223095924687406)
    assert select_extremize_factor(POOLED, OUTCOMES, grid=[np.array(1.3, dtype=object)]) == (1.3, 0.06223095924687406)
