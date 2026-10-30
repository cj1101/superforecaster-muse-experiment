"""SF-BL-11023 — agents.py log_odds_pool / aggregate must reject bool weights.

Pre-fix, both weights paths float-converted weights with no bool
screen: log_odds_pool([0.7, 0.3], weights=[True, False]) returned
exactly 0.7, bit-identical to the intentional integer mask
weights=[1, 0] — a boolean mask or flag silently collapsed the pool
to its first forecaster (calibration.py's bool-weight scope is
SF-BL-10883; the SF-BL-11250 string screen explicitly left bool
weights' behaviour to this task). These tests pin: ValueError naming
weights on any bool weight vector, numeric pools bit-identical
(including intentional int 1/0 masks), string weights untouched.
"""
import numpy as np
import pytest

from superforecaster.agents import (
    AgentForecast, aggregate, log_odds_pool, mean_pool, median_pool,
)

STRATS = ["mean", "median", "log_odds", "extremized",
          "weighted_log_odds", "weighted_extremized"]

BOOL_WEIGHTS = [
    [True, False],
    [False, True],
    [True, True],
    [False, False],
    (True, False),
    [1.0, True],
    [False, 0.5],
    [np.bool_(True), 0.5],
    [0.5, np.bool_(False)],
    np.array([True, False]),
    np.array([True, False], dtype=object),
    np.array([0.5, True], dtype=object),
    [[True], [False]],
    True,
]


def _fc(probs):
    return [AgentForecast(f"a{i}", "q", float(p)) for i, p in enumerate(probs)]


def test_kill_repro_log_odds_pool():
    with pytest.raises(ValueError, match="weights"):
        log_odds_pool([0.7, 0.3], weights=[True, False])


def test_kill_repro_aggregate():
    with pytest.raises(ValueError, match="weights"):
        aggregate(_fc([0.7, 0.3]), strategy='weighted_log_odds', weights=[True, False])


@pytest.mark.parametrize("w", BOOL_WEIGHTS)
def test_log_odds_pool_bool_weights_raise(w):
    with pytest.raises(ValueError, match="weights.*SF-BL-11023"):
        log_odds_pool([0.7, 0.3], weights=w)


@pytest.mark.parametrize("w", BOOL_WEIGHTS)
@pytest.mark.parametrize("s", STRATS)
def test_aggregate_bool_weights_raise_all_strategies(s, w):
    with pytest.raises(ValueError, match="weights.*SF-BL-11023"):
        aggregate(_fc([0.7, 0.3]), strategy=s, weights=w)


@pytest.mark.parametrize("w", BOOL_WEIGHTS)
def test_mean_median_pool_bool_weights_raise(w):
    with pytest.raises(ValueError, match="weights.*SF-BL-11023"):
        mean_pool([0.7, 0.3], weights=w)
    with pytest.raises(ValueError, match="weights.*SF-BL-11023"):
        median_pool([0.7, 0.3], weights=w)


def test_message_names_weights_and_tag():
    with pytest.raises(ValueError) as ei:
        log_odds_pool([0.7, 0.3], weights=[True, False])
    msg = str(ei.value)
    assert "weights" in msg and "SF-BL-11023" in msg and "bool" in msg


def test_numeric_weights_bit_identical():
    # Intentional integer 1/0 masks stay lawful and exact.
    assert log_odds_pool([0.7, 0.3], weights=[1, 0]) == 0.7
    assert log_odds_pool([0.7, 0.3], weights=[0, 1]) == 0.3
    assert log_odds_pool([0.7, 0.3], weights=np.array([1, 0])) == 0.7
    # Ordinary numeric controls (pre-fix values).
    assert repr(log_odds_pool([0.7, 0.3])) == "0.5"
    assert repr(log_odds_pool([0.7, 0.3], weights=[1.0, 1.0])) == "0.5"
    assert repr(log_odds_pool([0.7, 0.3], weights=[2.0, 1.0])) == "0.5701425119992315"
    assert repr(log_odds_pool([0.7, 0.3], weights=[0.0, 0.0])) == "0.5"
    assert mean_pool([0.7, 0.3], weights=[1, 0]) == 0.7
    assert median_pool([0.7, 0.3], weights=[1, 0]) == 0.7


def test_string_weights_still_raise_their_own_tag():
    with pytest.raises(ValueError, match="weights.*SF-BL-11250"):
        log_odds_pool([0.7, 0.3], weights=['0.5', '0.5'])


def test_nonfinite_weights_still_raise():
    with pytest.raises(ValueError, match="SF-BL-9788"):
        log_odds_pool([0.7, 0.3], weights=[float("nan"), 1.0])
