"""SF-BL-11015 — select_extremize_factor must reject finite out-of-range pooled forecasts.

Pre-fix, [1.5, 0.5]/[1, 0] and [82.0, 0.5]/[1, 0] both returned
(1.4, 0.125): _extremize_core clips pooled values to [EPS, 1-EPS]
before brier_score sees them, so no downstream screen could fire.
"""
import numpy as np
import pytest

from superforecaster.agents import select_extremize_factor


@pytest.mark.parametrize("pooled", [
    [1.5, 0.5], [82.0, 0.5], [-0.2, 0.5], [1.000000001, 0.5],
    [0.5, -1e-09], [50.0, 50.0], [0.1, 0.4, 2.0], [1.5], [-0.5], [2, 0],
])
def test_out_of_range_pooled_raises_named(pooled):
    outcomes = [1] * len(pooled) if len(pooled) == 1 else ([1, 0] if len(pooled) == 2 else [0, 0, 1])
    with pytest.raises(ValueError) as exc:
        select_extremize_factor(np.asarray(pooled, dtype=float),
                                np.asarray(outcomes, dtype=float))
    assert "select_extremize_factor" in str(exc.value)
    assert "SF-BL-11015" in str(exc.value)


def test_kill_repros_do_not_return_tuple():
    for pooled in ([1.5, 0.5], [82.0, 0.5]):
        with pytest.raises(ValueError):
            select_extremize_factor(np.array(pooled), np.array([1, 0]))


@pytest.mark.parametrize("pooled,outcomes,expected", [
    ([0.5, 0.5], [1, 0], (0.8, 0.25)),
    ([0.0, 1.0, 0.0, 1.0], [0, 1, 1, 0], None),
    ([0.42], [1], None),
    ([-0.0, 0.5], [0, 1], None),
])
def test_lawful_panels_unchanged(pooled, outcomes, expected):
    out = select_extremize_factor(np.asarray(pooled, dtype=float),
                                  np.asarray(outcomes, dtype=float))
    if expected is not None:
        assert out == expected
