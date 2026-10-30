"""SF-BL-11250 — agents.py log_odds_pool / aggregate must reject string weights.

Pre-fix, both float-converted weights with no dtype screen:
log_odds_pool([0.7, 0.3], weights=['0.5', '0.5']) returned 0.5,
bit-identical to weights=[0.5, 0.5], and aggregate(strategy=
'weighted_log_odds', weights=['0.5', '0.5']) likewise coerced
(calibration.py's SF-BL-11090 guard does not reach this copy).
These tests pin: ValueError naming weights on any string weight
vector, numeric pools bit-identical, bool weights untouched
(SF-BL-11023's distinct scope).
"""
import numpy as np
import pytest

from superforecaster.agents import (
    AgentForecast, aggregate, log_odds_pool, mean_pool, median_pool,
)

STRATS = ["mean", "median", "log_odds", "extremized",
          "weighted_log_odds", "weighted_extremized"]

STRING_WEIGHTS = [
    ['0.5', '0.5'],
    ('0.5', '0.5'),
    [0.5, '0.5'],
    ['0.5', 0.5],
    '0.5',
    np.array(['0.5', '0.5']),
    np.array([b'0.5', b'0.5']),
    np.array(['0.5', '0.5'], dtype=object),
    np.array([0.5, '0.5'], dtype=object),
    [np.str_('0.5'), 0.5],
    [['0.5'], ['0.5']],
    ['abc', '0.5'],
    ['', ''],
    [b'0.5', b'0.5'],
]


def _fc(probs):
    return [AgentForecast(f"a{i}", "q", float(p)) for i, p in enumerate(probs)]


def test_kill_repro_log_odds_pool():
    with pytest.raises(ValueError, match="weights"):
        log_odds_pool([0.7, 0.3], weights=['0.5', '0.5'])


def test_kill_repro_aggregate():
    with pytest.raises(ValueError, match="weights"):
        aggregate(_fc([0.7, 0.3]), strategy='weighted_log_odds', weights=['0.5', '0.5'])


@pytest.mark.parametrize("w", STRING_WEIGHTS)
def test_log_odds_pool_string_weights_raise(w):
    with pytest.raises(ValueError, match="weights.*SF-BL-11250"):
        log_odds_pool([0.7, 0.3], weights=w)


@pytest.mark.parametrize("w", STRING_WEIGHTS)
@pytest.mark.parametrize("s", STRATS)
def test_aggregate_string_weights_raise_all_strategies(s, w):
    with pytest.raises(ValueError, match="weights.*SF-BL-11250"):
        aggregate(_fc([0.7, 0.3]), strategy=s, weights=w)


@pytest.mark.parametrize("w", STRING_WEIGHTS)
def test_mean_median_pool_string_weights_raise(w):
    with pytest.raises(ValueError, match="weights.*SF-BL-11250"):
        mean_pool([0.7, 0.3], weights=w)
    with pytest.raises(ValueError, match="weights.*SF-BL-11250"):
        median_pool([0.7, 0.3], weights=w)


def test_message_names_weights_and_tag():
    with pytest.raises(ValueError) as ei:
        log_odds_pool([0.7, 0.3], weights=['0.5', '0.5'])
    msg = str(ei.value)
    assert "weights" in msg and "SF-BL-11250" in msg and "string" in msg


def test_numeric_control_bit_identical():
    assert log_odds_pool([0.7, 0.3], weights=[0.5, 0.5]) == 0.5
    assert log_odds_pool([0.7, 0.3], weights=[0.5, 0.5]).hex() == (0.5).hex()
    assert aggregate(_fc([0.7, 0.3]), strategy='weighted_log_odds', weights=[0.5, 0.5]) == 0.5
    # lawful forms: int, tuple, ndarray, object ndarray of pure numbers, None
    ref = log_odds_pool([0.7, 0.3], weights=[1, 2])
    assert log_odds_pool([0.7, 0.3], weights=(1, 2)) == ref
    assert log_odds_pool([0.7, 0.3], weights=np.array([1.0, 2.0])) == ref
    assert log_odds_pool([0.7, 0.3], weights=np.array([1.0, 2.0], dtype=object)) == ref
    assert log_odds_pool([0.7, 0.3]) == 0.5


def test_bool_weights_now_raise_sf_bl_11023():
    # Was test_bool_weights_keep_prefix_behaviour, pinning bool
    # weights as SF-BL-11023's then-unlanded distinct scope.
    # SF-BL-11023 has now landed: bool weights raise their own tag,
    # so the pre-fix pin (== 0.7) is superseded by this one.
    with pytest.raises(ValueError, match="weights.*SF-BL-11023"):
        log_odds_pool([0.7, 0.3], weights=[True, False])
    with pytest.raises(ValueError, match="weights.*SF-BL-11023"):
        mean_pool([0.7, 0.3], weights=[True, False])


def test_existing_weights_errors_unchanged():
    with pytest.raises(ValueError, match="SF-BL-8799"):
        aggregate(_fc([0.7, 0.3]), strategy='mean', weights=[1.0])
    with pytest.raises(ValueError, match="SF-BL-9788"):
        log_odds_pool([0.7, 0.3], weights=[float('nan'), 1.0])
    with pytest.raises(ValueError, match="SF-BL-8799"):
        mean_pool([0.7, 0.3], weights=[float('inf'), 1.0])


def test_probs_screens_keep_precedence_over_string_weights():
    with pytest.raises(ValueError, match="SF-BL-8902"):
        log_odds_pool([], weights=['0.5'])
    with pytest.raises(ValueError, match="SF-BL-10233"):
        log_odds_pool([float('nan'), 0.3], weights=['0.5', '0.5'])
    with pytest.raises(ValueError, match="SF-BL-10935"):
        log_odds_pool([1.5, 0.3], weights=['0.5', '0.5'])
