"""SF-BL-8799 — aggregate() must honour weights for every strategy.

Pre-fix, aggregate() read `weights` only in the weighted_log_odds /
weighted_extremized branches; mean, median, log_odds and extremized
silently pooled equal-weight. These tests pin the fixed contract:
weights honoured everywhere, malformed weights raise, weights=None
unchanged. (Plain asserts; also runnable without pytest via the
verifier in the SF-BL-8799 work dir.)
"""
import pytest

from superforecaster.agents import AgentForecast, aggregate, extremize

STRATS = ["mean", "median", "log_odds", "extremized",
          "weighted_log_odds", "weighted_extremized"]


def _fc(probs):
    return [AgentForecast(f"a{i}", "q", float(p)) for i, p in enumerate(probs)]


def test_task_repro_unit_weight_honoured_everywhere():
    for s in STRATS:
        got = aggregate(_fc([0.9, 0.5]), strategy=s, weights=[1.0, 0.0])
        assert abs(got - aggregate(_fc([0.9, 0.5]), strategy=s)) > 1e-9, s
    assert aggregate(_fc([0.9, 0.5]), strategy="log_odds", weights=[1.0, 0.0]) == pytest.approx(0.9)
    assert aggregate(_fc([0.9, 0.5]), strategy="mean", weights=[1.0, 0.0]) == pytest.approx(0.9)
    assert aggregate(_fc([0.9, 0.5]), strategy="median", weights=[1.0, 0.0]) == pytest.approx(0.9)
    assert aggregate(_fc([0.9, 0.5]), strategy="extremized", weights=[1.0, 0.0]) == pytest.approx(extremize(0.9, 1.30))


@pytest.mark.parametrize("s", STRATS)
def test_equal_weights_match_no_weights(s):
    a = aggregate(_fc([0.2, 0.5, 0.8]), strategy=s, weights=[1.0, 1.0, 1.0])
    b = aggregate(_fc([0.2, 0.5, 0.8]), strategy=s)
    assert a == pytest.approx(b, abs=1e-9)


@pytest.mark.parametrize("s", STRATS)
@pytest.mark.parametrize("bad", [[1.0], [1.0, 0.0, 0.5],
                                 [float("nan"), 1.0], [float("inf"), 1.0]])
def test_malformed_weights_raise(s, bad):
    with pytest.raises(ValueError):
        aggregate(_fc([0.9, 0.5]), strategy=s, weights=bad)
