"""SF-BL-9787 — agents.mean_pool / median_pool must refuse empty / non-finite panels.

Pre-fix (live repro re-verified by the executor on the pristine file):
mean_pool([0.5, nan]) and median_pool([0.5, nan]) both returned NaN as
an ordinary float, [0.5, inf] returned inf, and the weighted median
silently sorted a NaN member away (median_pool([0.2, nan, 0.8],
weights=equal) returned 0.8). The empty case already raised via
SF-BL-8902. These tests pin the fixed contract: a named ValueError
tagged SF-BL-9787 on any non-finite member (empty carries both
SF-BL-8902 and SF-BL-9787), with non-empty finite pools bit-identical
to the pristine implementation. Runnable under pytest, or standalone:
python3 test_agents_mean_median_pool_nonfinite.py
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
BADS = [float("nan"), float("inf"), float("-inf")]
PRISTINE = (Path.home() / "workspace" / "superforecaster_out" / "backlog"
            / "work" / "SF-BL-9787" / "pristine" / "agents.py")


def _pristine():
    spec = importlib.util.spec_from_file_location("agents_pristine_9787", PRISTINE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# --- Empty: ValueError, both tags -------------------------------------------

@pytest.mark.parametrize("fn", POOLS)
def test_empty_raises_9787(fn):
    with pytest.raises(ValueError, match="SF-BL-9787"):
        fn([])


@pytest.mark.parametrize("fn", POOLS)
def test_empty_raises_8902_still(fn):
    with pytest.raises(ValueError, match="SF-BL-8902"):
        fn([])


@pytest.mark.parametrize("fn", POOLS)
def test_empty_with_empty_weights_raises(fn):
    with pytest.raises(ValueError, match="SF-BL-9787"):
        fn([], weights=[])


@pytest.mark.parametrize("fn", POOLS)
def test_empty_tuple_and_ndarray_raise(fn):
    with pytest.raises(ValueError, match="SF-BL-9787"):
        fn(())
    with pytest.raises(ValueError, match="SF-BL-9787"):
        fn(np.array([]))


# --- Kill criterion: the exact TASK repros -----------------------------------

@pytest.mark.parametrize("fn", POOLS)
def test_kill_repro_0_5_nan_raises(fn):
    with pytest.raises(ValueError, match="SF-BL-9787"):
        fn([0.5, float("nan")])


@pytest.mark.parametrize("fn", POOLS)
def test_kill_never_returns_nan_for_empty_or_nan_panel(fn):
    for panel in ([], [0.5, float("nan")]):
        try:
            r = fn(panel)
        except ValueError:
            continue
        raise AssertionError(f"{fn.__name__}({panel}) returned {r!r} instead of raising")


# --- Non-finite members -------------------------------------------------------

@pytest.mark.parametrize("fn", POOLS)
@pytest.mark.parametrize("bad", BADS)
def test_single_nonfinite_raises(fn, bad):
    with pytest.raises(ValueError, match="SF-BL-9787"):
        fn([bad])


@pytest.mark.parametrize("fn", POOLS)
@pytest.mark.parametrize("bad", BADS)
def test_mixed_nonfinite_any_position_raises(fn, bad):
    for panel in ([bad, 0.4], [0.6, bad, 0.4], [0.6, 0.4, bad]):
        with pytest.raises(ValueError, match="SF-BL-9787"):
            fn(panel)


@pytest.mark.parametrize("fn", POOLS)
@pytest.mark.parametrize("bad", BADS)
def test_nonfinite_with_weights_raises(fn, bad):
    with pytest.raises(ValueError, match="SF-BL-9787"):
        fn([0.6, bad], weights=[1.0, 1.0])
    with pytest.raises(ValueError, match="SF-BL-9787"):
        fn([0.6, bad], weights=[0.25, 0.75])


@pytest.mark.parametrize("fn", POOLS)
def test_nonfinite_tuple_ndarray_np_scalar_raise(fn):
    with pytest.raises(ValueError, match="SF-BL-9787"):
        fn((0.5, float("nan")))
    with pytest.raises(ValueError, match="SF-BL-9787"):
        fn(np.array([0.5, np.nan]))
    with pytest.raises(ValueError, match="SF-BL-9787"):
        fn([np.float64("inf")])


@pytest.mark.parametrize("fn", POOLS)
def test_weighted_median_nan_silent_drop_repro_raises(fn):
    # Pre-fix weighted median_pool returned 0.8 here, dropping the NaN.
    with pytest.raises(ValueError, match="SF-BL-9787"):
        fn([0.2, float("nan"), 0.8], weights=[1.0 / 3.0] * 3)


@pytest.mark.parametrize("fn", POOLS)
def test_nonfinite_never_returns_a_number(fn):
    for bad in BADS:
        for panel in ([bad], [0.6, bad], [bad, 0.1, 0.9]):
            for w in (None, [1.0] * len(panel)):
                try:
                    r = fn(panel, weights=w)
                except ValueError:
                    continue
                raise AssertionError(f"{fn.__name__}({panel}, weights={w}) returned {r!r}")


@pytest.mark.parametrize("fn", POOLS)
def test_error_message_names_function(fn):
    with pytest.raises(ValueError) as ei:
        fn([0.5, float("nan")])
    assert fn.__name__ in str(ei.value) and "finite" in str(ei.value)


# --- Ordering / weights integrity (disclosed in PREREG) ------------------------

@pytest.mark.parametrize("fn", POOLS)
def test_member_screen_precedes_weights_screen(fn):
    # Corrupt member + corrupt weights: the member screen fires first.
    with pytest.raises(ValueError, match="SF-BL-9787"):
        fn([0.5, float("nan")], weights=[float("nan"), 1.0])


@pytest.mark.parametrize("fn", POOLS)
def test_finite_probs_bad_weights_still_raise_8799(fn):
    with pytest.raises(ValueError, match="SF-BL-8799"):
        fn([0.5, 0.6], weights=[float("nan"), 1.0])
    with pytest.raises(ValueError, match="SF-BL-8799"):
        fn([0.5, 0.6], weights=[1.0])


# --- Finite pools bit-identical ------------------------------------------------

def test_finite_pools_exact():
    cases = {
        (0.5,): 0.5, (0.2, 0.8): 0.5, (0.9, 0.5): 0.7,
        (0.0, 1.0): 0.5, (0.42, 0.42, 0.42): 0.42,
    }
    for panel, expected in cases.items():
        assert mean_pool(list(panel)) == expected
        assert median_pool(list(panel)) == expected


def test_finite_pools_bit_identical_vs_pristine():
    pristine = _pristine()
    rng = np.random.default_rng(9787)
    panels = [[0.5], [0.2, 0.8], [0.0, 1.0], [1e-9, 0.999999],
              [0.05, 0.95, 0.5, 0.5], [0.42, 0.42, 0.42]]
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
                got = getattr(sys.modules["superforecaster.agents"], name)(panel, weights=weights)
                want = getattr(pristine, name)(panel, weights=weights)
                assert got == want and repr(got) == repr(want), (name, panel, weights, got, want)


# --- Neighbours unchanged -------------------------------------------------------

def test_aggregate_mean_median_finite_and_corrupt():
    fc = [AgentForecast("a0", "q", 0.2), AgentForecast("a1", "q", 0.8)]
    assert aggregate(fc, strategy="mean") == 0.5
    assert aggregate(fc, strategy="median") == 0.5
    bad = AgentForecast("a", "q", 0.5)
    bad.probability = float("nan")
    for s in ("mean", "median"):
        with pytest.raises(ValueError):
            aggregate([bad], strategy=s)
    with pytest.raises(ValueError, match="SF-BL-8902"):
        aggregate([], strategy="mean")


def test_log_odds_pool_unchanged():
    with pytest.raises(ValueError, match="SF-BL-10233"):
        log_odds_pool([0.5, float("nan")])
    assert log_odds_pool([0.2, 0.8]) == pytest.approx(0.5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
