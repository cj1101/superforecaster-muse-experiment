"""SF-BL-10738 — aggregate() 'mean'/'median' + mean_pool/median_pool must
refuse finite out-of-[0, 1] members.

Pre-fix (live repro re-verified by the executor on the pristine file):
mean_pool([82.0, 0.5]) returned 41.25, median_pool([82.0, 0.5]) returned
41.25, median_pool([0.2, 82.0, 0.8]) returned 0.8 (the percent-unit
member silently sorted to the edge of the panel and absorbed), and
aggregate() with a member mutated to 82.0 after AgentForecast
construction returned 41.25 for both 'mean' and 'median'. Non-finite
members already raised (SF-BL-9787 at the pools, SF-BL-10233 at
aggregate dispatch); only the FINITE out-of-range case was unscreened,
while the log-odds strategies in the same dispatcher validate their
weights. These tests pin the fixed contract: a named ValueError tagged
SF-BL-10738 on any member outside [0, 1] (endpoints lawful) for
mean_pool / median_pool directly and for aggregate('mean'/'median'),
weighted and unweighted, with lawful panels bit-identical to the
pristine implementation and the log-odds strategies untouched.
Runnable under pytest, or standalone:
python3 test_agents_aggregate_mean_median_range.py
"""
import importlib.util
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
    mean_pool,
    median_pool,
)

POOLS = [mean_pool, median_pool]
OUT_OF_RANGE = [-100.0, -1.0, -0.2, -1e-9, 1.0000001, 1.5, 82.0, 100.0]
PRISTINE = (Path.home() / "workspace" / "superforecaster_out" / "backlog"
            / "work" / "SF-BL-10738" / "pristine" / "agents.py")


def _pristine():
    spec = importlib.util.spec_from_file_location("agents_pristine_10738", PRISTINE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _fc(probs):
    return [AgentForecast(f"a{i}", "q", p) for i, p in enumerate(probs)]


def _mutated(p):
    f = AgentForecast("a", "q", 0.5)
    f.probability = p  # bypasses the SF-BL-10769 construction screen
    return f


# --- Kill repros: the exact TASK cases ----------------------------------------

@pytest.mark.parametrize("fn", POOLS)
def test_task_repro_82_raises_10738(fn):
    with pytest.raises(ValueError, match="SF-BL-10738"):
        fn([82.0, 0.5])


def test_median_silent_absorption_repro_raises():
    # Pre-fix returned 0.8 with no signal.
    with pytest.raises(ValueError, match="SF-BL-10738"):
        median_pool([0.2, 82.0, 0.8])


@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_aggregate_task_repro_mutated_82_raises_10738(strategy):
    with pytest.raises(ValueError, match="SF-BL-10738"):
        aggregate([_mutated(82.0), _mutated(0.5)], strategy=strategy)


# --- Out-of-range grid: every position, weighted + unweighted ------------------

@pytest.mark.parametrize("fn", POOLS)
@pytest.mark.parametrize("bad", OUT_OF_RANGE)
def test_out_of_range_single_and_mixed_raise(fn, bad):
    for panel in ([bad], [bad, 0.4], [0.6, bad, 0.4], [0.6, 0.4, bad]):
        with pytest.raises(ValueError, match="SF-BL-10738"):
            fn(panel)
        with pytest.raises(ValueError, match="SF-BL-10738"):
            fn(panel, weights=[1.0] * len(panel))


@pytest.mark.parametrize("fn", POOLS)
def test_out_of_range_tuple_and_ndarray_raise(fn):
    with pytest.raises(ValueError, match="SF-BL-10738"):
        fn((0.5, 82.0))
    with pytest.raises(ValueError, match="SF-BL-10738"):
        fn(np.array([0.5, -0.25]))


@pytest.mark.parametrize("strategy", ["mean", "median"])
@pytest.mark.parametrize("bad", OUT_OF_RANGE)
def test_aggregate_out_of_range_raises(strategy, bad):
    with pytest.raises(ValueError, match="SF-BL-10738"):
        aggregate([_mutated(0.5), _mutated(bad)], strategy=strategy)
    with pytest.raises(ValueError, match="SF-BL-10738"):
        aggregate([_mutated(0.5), _mutated(bad)], strategy=strategy,
                  weights=[1.0, 1.0])


# --- Non-finite still raises (finite screen runs first) -------------------------

@pytest.mark.parametrize("fn", POOLS)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_still_raises(fn, bad):
    with pytest.raises(ValueError, match="SF-BL-9787"):
        fn([0.5, bad])


@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_aggregate_nonfinite_still_raises_10233(strategy):
    with pytest.raises(ValueError, match="SF-BL-10233"):
        aggregate([_mutated(float("nan"))], strategy=strategy)


# --- Endpoints + lawful panels: exact values -------------------------------------

def test_endpoints_lawful_exact():
    cases = {
        (0.0,): 0.0, (1.0,): 1.0, (0.0, 1.0): 0.5,
        (0.5,): 0.5, (0.2, 0.8): 0.5, (0.9, 0.5): 0.7,
        (0.42, 0.42, 0.42): 0.42,
    }
    for panel, expected in cases.items():
        assert mean_pool(list(panel)) == expected
        assert median_pool(list(panel)) == expected


def test_aggregate_mutated_endpoints_lawful():
    assert aggregate([_mutated(0.0), _mutated(1.0)], strategy="mean") == 0.5
    assert aggregate([_mutated(0.0), _mutated(1.0)], strategy="median") == 0.5


# --- Lawful panels bit-identical vs pristine --------------------------------------

def test_lawful_pools_bit_identical_vs_pristine():
    pristine = _pristine()
    rng = np.random.default_rng(10738)
    panels = [[0.5], [0.2, 0.8], [0.0, 1.0], [0.0], [1.0],
              [1e-9, 0.999999], [0.05, 0.95, 0.5, 0.5],
              [0.42, 0.42, 0.42]]
    for _ in range(300):
        n = int(rng.integers(1, 12))
        panels.append([float(x) for x in rng.random(n)])
        panels.append([float(x) for x in rng.beta(2.0, 2.0, size=n)])
    for panel in panels:
        for weights in (None,
                        [1.0 / len(panel)] * len(panel),
                        [float(i + 1) for i in range(len(panel))],
                        [float(x) for x in rng.random(len(panel))]):
            for name in ("mean_pool", "median_pool"):
                got = getattr(sys.modules["superforecaster.agents"], name)(
                    panel, weights=weights)
                want = getattr(pristine, name)(panel, weights=weights)
                assert got == want and repr(got) == repr(want), (
                    name, panel, weights, got, want)


def test_lawful_aggregate_bit_identical_vs_pristine():
    pristine = _pristine()
    rng = np.random.default_rng(10739)
    panels = [[0.5], [0.2, 0.8], [0.9, 0.5], [0.05, 0.95, 0.5, 0.5]]
    for _ in range(100):
        n = int(rng.integers(1, 8))
        # AgentForecast's own domain is (0, 1); keep panels interior.
        panels.append([min(max(float(x), 1e-6), 1 - 1e-6)
                       for x in rng.random(n)])
    for panel in panels:
        for strategy in ("mean", "median"):
            for weights in (None, [1.0 / len(panel)] * len(panel)):
                got = aggregate(_fc(panel), strategy=strategy, weights=weights)
                want = pristine.aggregate(
                    [pristine.AgentForecast(f"a{i}", "q", p)
                     for i, p in enumerate(panel)],
                    strategy=strategy, weights=weights)
                assert got == want and repr(got) == repr(want), (
                    strategy, panel, weights, got, want)


# --- Neighbours unchanged -----------------------------------------------------------

def test_log_odds_strategies_untouched_by_range_screen():
    # Scope pin: the prereg covers mean/median only. The log-odds
    # strategies keep their existing clip-based contract, bit-identical
    # to pristine even for a (corrupt) finite out-of-range member.
    pristine = _pristine()
    for strategy in ("log_odds", "weighted_log_odds",
                     "extremized", "weighted_extremized"):
        got = aggregate([_mutated(82.0), _mutated(0.5)], strategy=strategy)
        pf = [pristine.AgentForecast("a", "q", 0.5),
              pristine.AgentForecast("b", "q", 0.5)]
        pf[0].probability = 82.0
        want = pristine.aggregate(pf, strategy=strategy)
        assert got == want and repr(got) == repr(want), (strategy, got, want)
    assert log_odds_pool([0.2, 0.8]) == pytest.approx(0.5)


def test_empty_and_unknown_strategy_unchanged():
    for fn in POOLS:
        with pytest.raises(ValueError, match="SF-BL-8902"):
            fn([])
    with pytest.raises(ValueError, match="SF-BL-8902"):
        aggregate([], strategy="mean")
    with pytest.raises(ValueError, match="unknown strategy"):
        aggregate(_fc([0.5]), strategy="no_such_strategy")


def test_run_tournament_lawful_path_unchanged():
    # The only in-repo production caller of mean_pool/median_pool feeds
    # sigmoid+clip_p rows in [EPS, 1-EPS]; its published seed-42/300q
    # mean/median scores must not move (captured pre-fix).
    from superforecaster.agents import run_tournament
    res = run_tournament(n_questions=300, seed=42, verbose=False)
    assert res["strategies"]["mean"]["brier"] == 0.18017171657382994
    assert res["strategies"]["median"]["brier"] == 0.18147790426919883


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
