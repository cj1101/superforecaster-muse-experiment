"""SF-BL-10935 — agents.log_odds_pool must reject finite out-of-range members.

Pre-fix (live repro at task creation, re-verified by the executor):
log_odds_pool([1.5, 0.5]) returned 0.9990009985019828 and
log_odds_pool([82.0, 0.5]) returned the same, because clip_p mapped
the impossible member to 1-EPS before pooling — a percent-unit or
>1 member entered the pool as near-certainty and dominated it. The
sibling screens already covered non-finite weights (SF-BL-9788),
negative weights (SF-BL-8779), empty / non-finite panels
(SF-BL-10233), AgentForecast construction (SF-BL-10769) and the
mean/median range case (SF-BL-10738); no fixture covered a finite
out-of-range member of log_odds_pool itself. These tests pin the
fixed contract: a named ValueError tagged SF-BL-10935 on any finite
member outside [0, 1] (weighted and unweighted), propagated by
aggregate() for the log-odds strategies, with lawful panels —
endpoints 0.0 / 1.0 included — bit-identical to the pristine
pre-fix module. Runnable under pytest, or standalone:
python3 test_agents_log_odds_pool_range_members.py
"""
import hashlib
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
)

_PRISTINE_PATH = (Path.home() / "workspace" / "superforecaster_out" / "backlog"
                  / "work" / "SF-BL-10935" / "pristine" / "agents.py")

BADS = [1.5, 82.0, 2.0, 1.0000001, 100.0, -0.2, -1.0, -1e-9]
LOG_ODDS_STRATS = ["log_odds", "weighted_log_odds",
                   "extremized", "weighted_extremized"]


def _pristine():
    spec = importlib.util.spec_from_file_location(
        "sf_bl_10935_pristine_agents", _PRISTINE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _fc(probs):
    return [AgentForecast(f"a{i}", "q", float(p)) for i, p in enumerate(probs)]


def _mutated(p):
    """A forecast whose probability became out-of-range after construction."""
    f = AgentForecast("a", "q", 0.5)
    f.probability = p
    return f


# --- Kill-criterion repros -----------------------------------------------------

def test_kill_repro_1p5_raises_named():
    with pytest.raises(ValueError, match="SF-BL-10935"):
        log_odds_pool([1.5, 0.5])


def test_kill_repro_82_raises_named():
    with pytest.raises(ValueError, match="SF-BL-10935"):
        log_odds_pool([82.0, 0.5])


def test_kill_repros_never_return_a_float():
    for panel in ([1.5, 0.5], [82.0, 0.5]):
        for weights in (None, [1.0, 1.0], [2.0, 1.0]):
            try:
                r = log_odds_pool(panel, weights=weights)
            except ValueError:
                continue
            raise AssertionError(f"{panel} w={weights} returned {r!r}")


# --- Out-of-range members: positions / weights ---------------------------------

@pytest.mark.parametrize("bad", BADS)
def test_single_out_of_range_raises(bad):
    with pytest.raises(ValueError, match="SF-BL-10935"):
        log_odds_pool([bad])


@pytest.mark.parametrize("bad", BADS)
def test_mixed_out_of_range_raises(bad):
    with pytest.raises(ValueError, match="SF-BL-10935"):
        log_odds_pool([0.6, bad, 0.4])


@pytest.mark.parametrize("bad", BADS)
def test_out_of_range_with_weights_raises(bad):
    with pytest.raises(ValueError, match="SF-BL-10935"):
        log_odds_pool([0.6, bad], weights=[1.0, 1.0])
    with pytest.raises(ValueError, match="SF-BL-10935"):
        log_odds_pool([bad, 0.6], weights=[3.0, 0.5])


def test_error_names_function_and_offending_member():
    with pytest.raises(ValueError) as ei:
        log_odds_pool([0.5, 82.0])
    msg = str(ei.value)
    assert "log_odds_pool" in msg and "82" in msg and "[0, 1]" in msg


# --- Neighbouring screens unchanged ---------------------------------------------

def test_nonfinite_still_raises_10233_not_10935():
    # Finiteness screen runs first: +inf is also > 1 but must keep its
    # SF-BL-10233 identity, and NaN is invisible to range comparisons.
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="SF-BL-10233"):
            log_odds_pool([0.6, bad])


def test_empty_still_raises_8902_10233():
    with pytest.raises(ValueError, match="SF-BL-8902"):
        log_odds_pool([])
    with pytest.raises(ValueError, match="SF-BL-10233"):
        log_odds_pool([], weights=[])


def test_nonfinite_weights_still_raise_9788():
    with pytest.raises(ValueError, match="SF-BL-9788"):
        log_odds_pool([0.6, 0.7], weights=[float("nan"), 1.0])


# --- Lawful panels: endpoints + exact controls -----------------------------------

def test_endpoints_lawful_exact():
    assert log_odds_pool([0.0, 0.5]) == 0.0009990014980028714
    assert log_odds_pool([1.0, 0.5]) == 0.9990009985019828
    assert log_odds_pool([1.0]) == 0.999999
    assert log_odds_pool([0.0]) == 1.0000000000000008e-06
    assert log_odds_pool([0.0, 1.0]) == 0.49999999999640554


def test_finite_pools_exact():
    assert log_odds_pool([0.6, 0.7]) == 0.6516685226452116
    assert log_odds_pool([0.6, 0.7], weights=[1.0, 1.0]) == 0.6516685226452116
    assert log_odds_pool([0.6, 0.7], weights=[2.0, 1.0]) == 0.6347716821524939
    assert log_odds_pool([0.1, 0.5, 0.9], weights=[3.0, 1.0, 0.5]) == 0.22781674891972453
    assert log_odds_pool([0.5]) == 0.5
    assert log_odds_pool([0.42]) == 0.42


# --- aggregate propagation --------------------------------------------------------

@pytest.mark.parametrize("s", LOG_ODDS_STRATS)
@pytest.mark.parametrize("bad", [1.5, 82.0, -0.2])
def test_aggregate_log_odds_strategies_raise_10935(s, bad):
    with pytest.raises(ValueError, match="SF-BL-10935"):
        aggregate([_mutated(bad), _mutated(0.5)], strategy=s)


@pytest.mark.parametrize("s", ["mean", "median"])
def test_aggregate_mean_median_still_raise_10738(s):
    with pytest.raises(ValueError, match="SF-BL-10738"):
        aggregate([_mutated(82.0), _mutated(0.5)], strategy=s)


def test_aggregate_lawful_controls_exact():
    assert aggregate(_fc([0.9, 0.5]), strategy="log_odds") == 0.75
    assert aggregate(_fc([0.9, 0.5]), strategy="weighted_extremized") == 0.8066200721823593
    assert aggregate(_fc([0.42]), strategy="weighted_extremized") == 0.39661117592498224


# --- Bit-identical vs pristine ------------------------------------------------------

def _lawful_cases():
    rng = np.random.default_rng(10935)
    panels = [[0.5], [0.0], [1.0], [0.0, 1.0], [0.0, 0.5], [1.0, 0.5],
              [1e-9, 0.999999], [0.05, 0.95, 0.5, 0.5], [0.42, 0.42, 0.42],
              [0.6, 0.7], [0.1, 0.5, 0.9]]
    for _ in range(300):
        n = int(rng.integers(1, 12))
        panels.append([float(x) for x in rng.random(n)])
        panels.append([float(x) for x in rng.beta(2.0, 2.0, size=n)])
    cases = []
    for panel in panels:
        for weights in (None,
                        [1.0 / len(panel)] * len(panel),
                        [float(i + 1) for i in range(len(panel))],
                        [float(x) for x in rng.random(len(panel))]):
            cases.append((panel, weights))
    return cases


def test_lawful_pools_bit_identical_vs_pristine():
    pristine = _pristine()
    for panel, weights in _lawful_cases():
        got = log_odds_pool(panel, weights=weights)
        want = pristine.log_odds_pool(panel, weights=weights)
        assert got == want and repr(got) == repr(want), (panel, weights, got, want)


def test_lawful_pools_digest_matches_pristine():
    pristine = _pristine()
    def digest(fn):
        h = hashlib.sha256()
        for panel, weights in _lawful_cases():
            h.update(repr(fn(panel, weights=weights)).encode())
            h.update(b"|")
        return h.hexdigest()
    assert digest(log_odds_pool) == digest(pristine.log_odds_pool)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
