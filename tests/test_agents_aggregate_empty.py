"""SF-BL-8902 — aggregate() must never ship a pooled forecast from zero forecasts.

Pre-fix, aggregate([], strategy=s) fabricated a number for every s:
mean/median returned NaN (RuntimeWarning only), log_odds /
weighted_log_odds returned exactly 0.0 (NaN mean logit laundered by
sigmoid), extremized / weighted_extremized returned 1.58e-08, and
aggregate([], weights=[]) returned 0.0; log_odds_pool([], weights=[])
returned exactly 0.5 — a coin flip from no forecasts. These tests pin
the fixed contract (the TASK criterion, parametrized over strategies):
a named ValueError tagged SF-BL-8902 for every strategy, the same guard
in the pool primitives, and bit-identical non-empty behaviour.
Runnable under pytest, or standalone: python3 test_agents_aggregate_empty.py
"""
import importlib.util
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
    mean_pool,
    median_pool,
)

STRATS = ["mean", "median", "log_odds", "extremized",
          "weighted_log_odds", "weighted_extremized"]


def _fc(probs):
    return [AgentForecast(f"a{i}", "q", float(p)) for i, p in enumerate(probs)]


# --- The TASK criterion, parametrized -------------------------------------

@pytest.mark.parametrize("s", STRATS)
def test_aggregate_empty_raises_named_error_every_strategy(s):
    with pytest.raises(ValueError, match="SF-BL-8902"):
        aggregate([], strategy=s)


@pytest.mark.parametrize("s", STRATS)
def test_aggregate_empty_with_empty_weights_raises(s):
    with pytest.raises(ValueError, match="SF-BL-8902"):
        aggregate([], strategy=s, weights=[])


@pytest.mark.parametrize("s", STRATS)
def test_aggregate_empty_tuple_raises(s):
    with pytest.raises(ValueError, match="SF-BL-8902"):
        aggregate((), strategy=s)


def test_aggregate_empty_error_is_named_and_explains():
    with pytest.raises(ValueError) as ei:
        aggregate([], strategy="mean")
    msg = str(ei.value)
    assert "aggregate" in msg and "empty" in msg and "SF-BL-8902" in msg


def test_aggregate_empty_never_returns_a_number():
    # Negative space: no strategy may return NaN / 0.0 / 0.5 / anything.
    for s in STRATS:
        try:
            r = aggregate([], strategy=s)
        except ValueError:
            continue
        raise AssertionError(f"{s} returned {r!r} on empty input instead of raising")


# --- Pool primitives --------------------------------------------------------

@pytest.mark.parametrize("fn", [mean_pool, median_pool, log_odds_pool])
def test_pools_empty_raise(fn):
    with pytest.raises(ValueError, match="SF-BL-8902"):
        fn([])


@pytest.mark.parametrize("fn", [mean_pool, median_pool, log_odds_pool])
def test_pools_empty_with_empty_weights_raise(fn):
    with pytest.raises(ValueError, match="SF-BL-8902"):
        fn([], weights=[])


def test_median_pool_empty_weighted_is_named_not_indexerror():
    # Pre-fix this path died with a bare IndexError.
    with pytest.raises(ValueError, match="SF-BL-8902"):
        median_pool([], weights=[])


# --- Ordering / dispatch integrity ------------------------------------------

def test_empty_check_precedes_unknown_strategy_dispatch():
    with pytest.raises(ValueError, match="SF-BL-8902"):
        aggregate([], strategy="no_such_strategy")


def test_unknown_strategy_nonempty_still_named_unknown():
    with pytest.raises(ValueError, match="unknown strategy"):
        aggregate(_fc([0.5]), strategy="no_such_strategy")


def test_generator_input_still_works_and_empty_generator_raises():
    assert aggregate(iter(_fc([0.9, 0.5])), strategy="mean") == pytest.approx(0.7)
    with pytest.raises(ValueError, match="SF-BL-8902"):
        aggregate(iter([]), strategy="mean")


# --- Non-empty behaviour bit-identical --------------------------------------

CONTROLS = {
    (0.9, 0.5): {"mean": 0.7, "median": 0.7, "log_odds": 0.75,
                 "extremized": 0.8066200721823593,
                 "weighted_log_odds": 0.75,
                 "weighted_extremized": 0.8066200721823593},
    (0.42,): {"mean": 0.42, "median": 0.42, "log_odds": 0.42,
              "extremized": 0.39661117592498224,
              "weighted_log_odds": 0.42,
              "weighted_extremized": 0.39661117592498224},
    (0.05, 0.95, 0.5, 0.5): {"mean": 0.5, "median": 0.5,
                             "log_odds": 0.49999999999999994,
                             "extremized": 0.49999999999999994,
                             "weighted_log_odds": 0.49999999999999994,
                             "weighted_extremized": 0.49999999999999994},
}


@pytest.mark.parametrize("panel", list(CONTROLS))
@pytest.mark.parametrize("s", STRATS)
def test_nonempty_controls_exact(panel, s):
    assert aggregate(_fc(panel), strategy=s) == CONTROLS[panel][s]


def test_single_forecast_pools_unchanged():
    assert mean_pool([0.42]) == 0.42
    assert median_pool([0.42]) == 0.42
    assert log_odds_pool([0.42]) == 0.42


def _load_pristine():
    cand = Path(__file__).parent / "pristine" / "agents.py"
    if not cand.exists():
        return None
    spec = importlib.util.spec_from_file_location("agents_pristine_8902", cand)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_equivalence_sweep_vs_pristine():
    pristine = _load_pristine()
    if pristine is None:
        pytest.skip("pristine snapshot not beside this test file")
    rng = np.random.default_rng(8902)
    for _ in range(300):
        n = int(rng.integers(1, 9))
        panel = np.clip(rng.random(n), 0.02, 0.98)
        fc_new = _fc(panel)
        fc_old = [pristine.AgentForecast(f"a{i}", "q", float(p))
                  for i, p in enumerate(panel)]
        w = rng.random(n) + 0.05
        for s in STRATS:
            for weights in (None, w):
                got = aggregate(fc_new, strategy=s, weights=weights)
                want = pristine.aggregate(fc_old, strategy=s, weights=weights)
                assert got == want and repr(got) == repr(want), (panel, s, weights)
        for fn_new, fn_old in ((mean_pool, pristine.mean_pool),
                               (median_pool, pristine.median_pool),
                               (log_odds_pool, pristine.log_odds_pool)):
            assert fn_new(panel) == fn_old(panel)
            assert fn_new(panel, weights=w) == fn_old(panel, weights=w)
    assert math.isfinite(aggregate(_fc([0.3, 0.6]), strategy="mean"))
