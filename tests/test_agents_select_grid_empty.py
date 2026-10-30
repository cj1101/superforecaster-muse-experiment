"""SF-BL-9790 — select_extremize_factor must validate its grid up front.

Pre-fix, an empty grid returned the unsearched initialiser (1.0, inf)
as if a search had run, and the SF-BL-8875 finite/>0 screen lived
inside the scan loop so a bad entry was rejected only after earlier
entries had been scored. These tests pin the fixed contract: an empty
grid or any non-finite/non-positive entry raises a named ValueError
before any scoring, and lawful grids (incl. the default) are unchanged.
"""
import numpy as np
import pytest

from superforecaster.agents import select_extremize_factor

POOLED = np.array([0.6, 0.7])
OUTCOMES = np.array([1, 0])
NAN, INF = float("nan"), float("inf")


@pytest.mark.parametrize("bad_grid", [
    [], np.array([]), (), range(0),
    [NAN], [INF], [-INF], [0.0], [-1.0],
    [NAN, 1.0], [1.0, NAN], [1.0, 0.0, 1.3],
])
def test_select_rejects_empty_or_bad_grid(bad_grid):
    with pytest.raises(ValueError, match="SF-BL-9790"):
        select_extremize_factor(POOLED, OUTCOMES, grid=bad_grid)


def test_select_rejects_empty_generator():
    with pytest.raises(ValueError, match="SF-BL-9790"):
        select_extremize_factor(POOLED, OUTCOMES, grid=(g for g in []))


def test_task_repro_no_silent_pairs():
    with pytest.raises(ValueError):
        select_extremize_factor(POOLED, OUTCOMES, grid=[NAN])  # pre-8875: (nan, 0.5)
    with pytest.raises(ValueError):
        select_extremize_factor(POOLED, OUTCOMES, grid=[])     # pre-fix: (1.0, inf)


def test_default_and_good_grids_unchanged():
    assert select_extremize_factor(POOLED, OUTCOMES) == (0.8, 0.3079908284788835)
    assert select_extremize_factor(POOLED, OUTCOMES, grid=[1.0, 1.3]) == (1.0, 0.32499999999999996)
    f, _ = select_extremize_factor(POOLED, OUTCOMES, grid=np.array([0.8, 1.0, 1.85]))
    assert f in (0.8, 1.0, 1.85)
