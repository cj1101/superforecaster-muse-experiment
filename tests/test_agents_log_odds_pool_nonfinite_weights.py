"""SF-BL-9788: agents.log_odds_pool must reject non-finite weights.

Bug: log_odds_pool([0.6, 0.7], weights=[nan, 1.0]) returned exactly 0.0.
The NaN weight sum makes ``w.sum() <= 0`` False, bypassing the equal-weight
fallback; weights normalise to NaN and the scalar sigmoid path emits 0.0.
Negative weights were covered by SF-BL-8779; non-finite weights were not.

Done criterion: ValueError on any non-finite weight; finite-weight pools
bit-identical to pre-fix behaviour.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from superforecaster.agents import log_odds_pool  # noqa: E402

BAD = [float("nan"), float("inf"), float("-inf")]


@pytest.mark.parametrize("bad", BAD)
def test_nonfinite_weight_raises(bad):
    with pytest.raises(ValueError):
        log_odds_pool([0.6, 0.7], weights=[bad, 1.0])
    with pytest.raises(ValueError):
        log_odds_pool([0.6, 0.7], weights=[1.0, bad])


def test_all_nonfinite_weights_raise():
    with pytest.raises(ValueError):
        log_odds_pool([0.6, 0.7], weights=[float("nan"), float("nan")])


def test_kill_criterion_nan_repro_no_longer_returns_float():
    try:
        out = log_odds_pool([0.6, 0.7], weights=[math.nan, 1.0])
    except ValueError:
        return
    raise AssertionError(f"NaN-weight repro returned unflagged float {out!r}")


def test_finite_weight_pools_bit_identical():
    # Values captured from the pre-fix implementation (2026-10-21).
    assert log_odds_pool([0.6, 0.7]) == 0.6516685226452116
    assert log_odds_pool([0.6, 0.7], weights=[1.0, 1.0]) == 0.6516685226452116
    assert log_odds_pool([0.6, 0.7], weights=[2.0, 1.0]) == 0.6347716821524939
    assert log_odds_pool([0.6, 0.7], weights=[0.0, 0.0]) == 0.6516685226452116
    assert log_odds_pool([0.1, 0.5, 0.9], weights=[3.0, 1.0, 0.5]) == 0.22781674891972453
    assert log_odds_pool([0.5], weights=[1.0]) == 0.5


def test_aggregate_weighted_strategies_propagate_raise():
    from superforecaster.agents import AgentForecast, aggregate

    fc = [AgentForecast(agent_id="a", question_id="q", probability=0.6),
          AgentForecast(agent_id="b", question_id="q", probability=0.7)]
    with pytest.raises(ValueError):
        aggregate(fc, strategy="weighted_log_odds", weights=[float("nan"), 1.0])
