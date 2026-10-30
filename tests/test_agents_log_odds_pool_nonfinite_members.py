"""SF-BL-10233 — agents.log_odds_pool must reject empty / non-finite panels.

Pre-fix (live repro at task creation, re-verified by the executor):
log_odds_pool([]) fabricated a number on the empty path (0.0 / 0.5
depending on weights; the empty case is now also covered by SF-BL-8902),
log_odds_pool([nan]) returned exactly 0.0, log_odds_pool([inf]) returned
0.999999 (clipped to 1-EPS), and aggregate with a NaN member (mutated
onto an AgentForecast after construction) returned exactly 0.0 for the
log-odds strategies and 1.5848952277057052e-08 for weighted_extremized.
These tests pin the fixed contract: a named ValueError tagged
SF-BL-10233 on an empty panel or any non-finite member, propagated by
aggregate() for every strategy, with non-empty finite panels
bit-identical. Runnable under pytest, or standalone:
python3 test_agents_log_odds_pool_nonfinite_members.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path.home() / "workspace" / "superforecaster"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from superforecaster.agents import (  # noqa: E402
    AgentForecast,
    aggregate,
    log_odds_pool,
)

STRATS = ["mean", "median", "log_odds", "extremized",
          "weighted_log_odds", "weighted_extremized"]
BADS = [float("nan"), float("inf"), float("-inf")]


def _fc(probs):
    return [AgentForecast(f"a{i}", "q", float(p)) for i, p in enumerate(probs)]


def _mutated(p):
    """A forecast whose probability became non-finite after construction."""
    f = AgentForecast("a", "q", 0.5)
    f.probability = p
    return f


# --- log_odds_pool: empty ---------------------------------------------------

def test_log_odds_pool_empty_raises_named():
    with pytest.raises(ValueError, match="SF-BL-10233"):
        log_odds_pool([])


def test_log_odds_pool_empty_with_empty_weights_raises_named():
    with pytest.raises(ValueError, match="SF-BL-10233"):
        log_odds_pool([], weights=[])


def test_log_odds_pool_empty_still_carries_8902_tag():
    with pytest.raises(ValueError, match="SF-BL-8902"):
        log_odds_pool([])


# --- log_odds_pool: non-finite members ---------------------------------------

@pytest.mark.parametrize("bad", BADS)
def test_log_odds_pool_single_nonfinite_raises(bad):
    with pytest.raises(ValueError, match="SF-BL-10233"):
        log_odds_pool([bad])


@pytest.mark.parametrize("bad", BADS)
def test_log_odds_pool_mixed_nonfinite_raises(bad):
    with pytest.raises(ValueError, match="SF-BL-10233"):
        log_odds_pool([0.6, bad, 0.4])


@pytest.mark.parametrize("bad", BADS)
def test_log_odds_pool_nonfinite_with_weights_raises(bad):
    with pytest.raises(ValueError, match="SF-BL-10233"):
        log_odds_pool([0.6, bad], weights=[1.0, 1.0])


def test_log_odds_pool_nonfinite_never_returns_a_number():
    for bad in BADS:
        for panel in ([bad], [0.6, bad]):
            try:
                r = log_odds_pool(panel)
            except ValueError:
                continue
            raise AssertionError(f"{panel} returned {r!r} instead of raising")


def test_kill_criterion_repros_no_longer_return_float():
    # The exact task repros.
    for fn in (lambda: log_odds_pool([float("nan")]),
               lambda: log_odds_pool([float("inf")])):
        with pytest.raises(ValueError, match="SF-BL-10233"):
            fn()


# --- aggregate: propagation for every strategy --------------------------------

@pytest.mark.parametrize("s", STRATS)
def test_aggregate_empty_raises_named(s):
    with pytest.raises(ValueError, match="SF-BL-10233"):
        aggregate([], strategy=s)


@pytest.mark.parametrize("s", STRATS)
@pytest.mark.parametrize("bad", BADS)
def test_aggregate_nonfinite_member_raises_every_strategy(s, bad):
    with pytest.raises(ValueError, match="SF-BL-10233"):
        aggregate([_mutated(bad)], strategy=s)


@pytest.mark.parametrize("s", STRATS)
def test_aggregate_mixed_panel_nonfinite_raises(s):
    with pytest.raises(ValueError, match="SF-BL-10233"):
        aggregate([_mutated(0.5), _mutated(float("nan"))], strategy=s)


@pytest.mark.parametrize("s", STRATS)
def test_aggregate_duck_typed_nonfinite_raises(s):
    class Duck:
        probability = float("nan")
    with pytest.raises(ValueError, match="SF-BL-10233"):
        aggregate([Duck()], strategy=s)


def test_aggregate_weighted_extremized_nan_no_longer_1_58e_08():
    with pytest.raises(ValueError, match="SF-BL-10233"):
        aggregate([_mutated(float("nan"))], strategy="weighted_extremized")


# --- Finite panels bit-identical ------------------------------------------------

def test_finite_pools_exact():
    assert log_odds_pool([0.6, 0.7]) == 0.6516685226452116
    assert log_odds_pool([0.6, 0.7], weights=[1.0, 1.0]) == 0.6516685226452116
    assert log_odds_pool([0.6, 0.7], weights=[2.0, 1.0]) == 0.6347716821524939
    assert log_odds_pool([0.1, 0.5, 0.9], weights=[3.0, 1.0, 0.5]) == 0.22781674891972453
    assert log_odds_pool([0.5]) == 0.5
    assert log_odds_pool([0.42]) == 0.42


def test_finite_aggregate_controls_exact():
    assert aggregate(_fc([0.9, 0.5]), strategy="weighted_extremized") == 0.8066200721823593
    assert aggregate(_fc([0.9, 0.5]), strategy="log_odds") == 0.75
    assert aggregate(_fc([0.42]), strategy="weighted_extremized") == 0.39661117592498224
    assert math.isfinite(aggregate(_fc([0.3, 0.6]), strategy="mean"))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
