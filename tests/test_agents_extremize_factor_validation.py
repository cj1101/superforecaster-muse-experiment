"""SF-BL-8875 — extremize()/aggregate() must validate the extremize factor.

Pre-fix, extremize(0.8, factor=-1.0) returned 0.2 (full inversion),
factor=0.0 returned 0.5 (collapse to centre), NaN returned 0.0 and +inf
returned 1.0, and aggregate() forwarded caller factors straight in —
unlike calibration.py (raises on extremization<=0) and engine.py
(learned factors capped to [1.00, 1.40]). These tests pin the fixed
contract: factor<=0, non-finite, non-numeric and out-of-band factors
raise ValueError before a forward pool ships; in-band behaviour is
bit-identical to the pre-fix values.
"""
import numpy as np
import pytest

from superforecaster.agents import AgentForecast, aggregate, extremize, select_extremize_factor

BAD = [-1.0, -0.5, -1e-9, 0.0, 0.5, 0.99, 1.41, 2.0,
       float("nan"), float("inf"), float("-inf"), "1.3", None, True]
STRATS = ["mean", "median", "log_odds", "extremized",
          "weighted_log_odds", "weighted_extremized"]


def _fc(probs):
    return [AgentForecast(f"a{i}", "q", float(p)) for i, p in enumerate(probs)]


@pytest.mark.parametrize("bad", BAD)
def test_extremize_rejects_bad_factor(bad):
    with pytest.raises(ValueError, match="SF-BL-8875"):
        extremize(0.8, factor=bad)


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.parametrize("s", STRATS)
def test_aggregate_rejects_bad_factor_every_strategy(bad, s):
    with pytest.raises(ValueError, match="SF-BL-8875"):
        aggregate(_fc([0.9, 0.5]), strategy=s, extremize_factor=bad)


def test_task_repro_no_inversion_no_collapse():
    with pytest.raises(ValueError):
        extremize(0.8, factor=-1.0)   # pre-fix: 0.2
    with pytest.raises(ValueError):
        extremize(0.8, factor=0.0)    # pre-fix: 0.5
    with pytest.raises(ValueError):
        aggregate(_fc([0.9, 0.5]), strategy="weighted_extremized", extremize_factor=-1.0)


def test_inband_controls_unchanged():
    assert extremize(0.8, factor=1.00) == 0.8
    assert extremize(0.8, factor=1.30) == 0.8584144223701332
    assert extremize(0.8) == 0.8584144223701332
    assert extremize(0.8, factor=1.40) == 0.8744413346659732
    assert extremize(0.8, factor=np.float64(1.3)) == 0.8584144223701332
    assert aggregate(_fc([0.9, 0.5]), strategy="weighted_extremized") == 0.8066200721823593
    assert aggregate(_fc([0.9, 0.5]), strategy="mean", extremize_factor=1.30) == 0.7


def test_select_factor_research_grid_still_learns():
    rng = np.random.default_rng(7)
    pooled = rng.random(120)
    outcomes = (rng.random(120) < pooled).astype(int)
    f, _ = select_extremize_factor(pooled, outcomes, grid=np.array([0.8, 1.0, 1.85]))
    assert f in (0.8, 1.0, 1.85)
    for bad_grid in ([0.0, 1.2], [-1.0, 1.2], [float("nan")]):
        with pytest.raises(ValueError, match="SF-BL-8875"):
            select_extremize_factor(pooled, outcomes, grid=np.array(bad_grid, dtype=float))
