"""SF-BL-11093: agents.py mean_pool / median_pool must reject string members.

Bug: both pools coerce members through np.asarray(dtype=float) with
no type screen, so a numeric-string forecast is silently parsed and
pooled as a lawful probability: mean_pool(['0.7', '0.3']) returned
0.5 and median_pool(['0.7', '0.3', '0.9']) returned 0.7 in live
repros. SF-BL-10738 fixtures finite out-of-range members (82.0)
reaching the pools via aggregate, and SF-BL-9787 fixtures empty /
non-finite panels, but no item fixtured string members. Differs from
SF-BL-11058 (the scorers' string panels) in surface: the pool
primitives, reached directly and via aggregate's mean/median
strategies.

Done criterion: guard raises ValueError naming probs on any
string-dtype member panel in mean_pool and median_pool, with numeric
panels bit-identical.
Kill criterion: mean_pool(['0.7', '0.3']) returns 0.5 or
median_pool(['0.7', '0.3', '0.9']) returns 0.7 instead of raising.

Runnable under pytest, or standalone:
python3 test_agents_mean_median_pool_string_members.py
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
PRISTINE = (Path.home() / "workspace" / "superforecaster_out" / "backlog"
            / "work" / "SF-BL-11093" / "pristine" / "agents.py")


def _pristine():
    spec = importlib.util.spec_from_file_location("agents_pristine_11093", PRISTINE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# --- Kill criteria: the exact live repros from the task ---

def test_kill_repro_mean_string_panel_returns_no_half():
    with pytest.raises(ValueError, match=r"probs.*SF-BL-11093"):
        mean_pool(['0.7', '0.3'])


def test_kill_repro_median_string_panel_returns_no_point_seven():
    with pytest.raises(ValueError, match=r"probs.*SF-BL-11093"):
        median_pool(['0.7', '0.3', '0.9'])


@pytest.mark.parametrize("fn,panel", [
    (mean_pool, ['0.7', '0.3']), (median_pool, ['0.7', '0.3', '0.9'])])
def test_kill_never_returns_the_prefix_number(fn, panel):
    try:
        r = fn(panel)
    except ValueError:
        return
    raise AssertionError(f"{fn.__name__}({panel}) returned {r!r} instead of raising")


# --- Every string form, both pools, unweighted and weighted ---

STRING_PANELS = [
    ['0.7', '0.3'],
    ['0.7'],
    [0.7, '0.3'],
    ['0.7', 0.3],
    ('0.7', '0.3'),
    [['0.7', '0.3']],
    np.array(['0.7', '0.3']),
    np.array(['0.7']),
    np.array([b'0.7', b'0.3']),
    np.array(['0.7', '0.3'], dtype=object),
    np.array([0.7, '0.3'], dtype=object),
    [b'0.7', '0.3'],
    [np.str_('0.7'), 0.3],
    '0.7',              # bare scalar panel: pre-fix bare TypeError
    np.str_('0.7'),
    ['abc'],            # non-numeric strings: named raise, not a bare parse error
    [''],
]


def _n(panel):
    if isinstance(panel, (str, bytes, np.str_)):
        return 1
    return len(np.asarray(panel, dtype=object).ravel())


@pytest.mark.parametrize("fn", POOLS)
@pytest.mark.parametrize("bad", STRING_PANELS)
def test_string_panels_raise(fn, bad):
    with pytest.raises(ValueError, match=r"probs"):
        fn(bad)


@pytest.mark.parametrize("fn", POOLS)
@pytest.mark.parametrize("bad", STRING_PANELS)
def test_string_panels_raise_weighted(fn, bad):
    with pytest.raises(ValueError, match=r"probs.*SF-BL-11093"):
        fn(bad, weights=[1.0] * _n(bad))


@pytest.mark.parametrize("fn", POOLS)
def test_message_names_function_probs_and_tag(fn):
    with pytest.raises(ValueError) as ei:
        fn(['0.7', '0.3'])
    msg = str(ei.value)
    assert fn.__name__ in msg and "probs" in msg and "SF-BL-11093" in msg


@pytest.mark.parametrize("fn", POOLS)
def test_string_screen_precedes_weights_screen(fn):
    # String member + corrupt weights: the member screen fires first.
    with pytest.raises(ValueError, match=r"SF-BL-11093"):
        fn(['0.7', '0.3'], weights=[float("nan"), 1.0])


@pytest.mark.parametrize("fn", POOLS)
def test_string_screen_precedes_finite_and_range_screens(fn):
    # A string that would parse out-of-range or non-finite is still a
    # string first: the named 11093 raise, not 10738/9787.
    with pytest.raises(ValueError, match=r"SF-BL-11093"):
        fn(['82.0', '0.5'])
    with pytest.raises(ValueError, match=r"SF-BL-11093"):
        fn(['nan', '0.5'])


# --- Lawful numeric panels bit-identical (captured pre-fix, 2026-10-23) ---

def test_lawful_panels_exact():
    cases = [
        # (panel, weights, mean, median)
        ([0.7, 0.3], None, 0.5, 0.5),
        ([0.7, 0.3, 0.9], None, 0.6333333333333333, 0.7),
        ([0.5], None, 0.5, 0.5),
        ([0.0, 1.0], None, 0.5, 0.5),
        ([0.2, 0.8], None, 0.5, 0.5),
        ([1, 0], None, 0.5, 0.5),
        ((0.25, 0.75), None, 0.5, 0.5),
        ([0.1, 0.9, 0.3, 0.7], None, 0.5, 0.5),
        ([0.42, 0.42, 0.42], None, 0.42, 0.42),
        ([0.7, 0.3], [1.0, 1.0], 0.5, 0.5),
        ([0.7, 0.3], [0.25, 0.75], 0.39999999999999997, 0.3),
        ([0.7, 0.3, 0.9], [1, 2, 3], 0.6666666666666666, 0.8),
        ([0.05, 0.95, 0.5, 0.5], [1, 1, 1, 1], 0.5, 0.5),
    ]
    for panel, w, want_mean, want_median in cases:
        assert mean_pool(panel, weights=w) == want_mean, (panel, w)
        assert median_pool(panel, weights=w) == want_median, (panel, w)
        assert repr(mean_pool(panel, weights=w)) == repr(want_mean)
        assert repr(median_pool(panel, weights=w)) == repr(want_median)


def test_lawful_ndarray_and_numpy_scalar_panels():
    assert mean_pool(np.array([0.7, 0.3])) == 0.5
    assert median_pool(np.array([0.7, 0.3, 0.9])) == 0.7
    assert mean_pool([np.float64(0.7), np.float32(0.3)]) == 0.5000000059604645
    assert median_pool([np.float64(0.7)]) == 0.7
    assert mean_pool(np.array([1, 0])) == 0.5  # int-dtype array stays lawful


def test_bool_members_keep_prefix_behaviour():
    # Out of this task's dimension: bool members pooled to 0.5 pre-fix
    # and the SF-BL-10939 bool screen covers the scorers, not pools.
    assert mean_pool([True, False]) == 0.5
    assert median_pool([True, False]) == 0.5


def test_numeric_pools_bit_identical_vs_pristine():
    pristine = _pristine()
    rng = np.random.default_rng(11093)
    panels = [[0.5], [0.2, 0.8], [0.0, 1.0], [1e-9, 0.999999],
              [0.05, 0.95, 0.5, 0.5], [0.42, 0.42, 0.42], [1, 0]]
    for _ in range(300):
        n = int(rng.integers(1, 12))
        panels.append([float(x) for x in rng.random(n)])
        panels.append([float(x) for x in rng.beta(2.0, 2.0, size=n)])
        panels.append([int(x) for x in rng.integers(0, 2, size=n)])
    for panel in panels:
        for weights in (None,
                        [1.0 / len(panel)] * len(panel),
                        [float(i + 1) for i in range(len(panel))],
                        [float(x) for x in rng.random(len(panel))]):
            for name in ("mean_pool", "median_pool"):
                got = getattr(sys.modules["superforecaster.agents"], name)(panel, weights=weights)
                want = getattr(pristine, name)(panel, weights=weights)
                assert got == want and repr(got) == repr(want), (name, panel, weights, got, want)


# --- Prior guards preserved ---

@pytest.mark.parametrize("fn", POOLS)
def test_prior_guards_still_raise(fn):
    with pytest.raises(ValueError, match="SF-BL-8902"):
        fn([])                                    # empty (SF-BL-8902/9787)
    with pytest.raises(ValueError, match="SF-BL-9787"):
        fn([0.5, float("nan")])                   # non-finite (SF-BL-9787)
    with pytest.raises(ValueError, match="SF-BL-9787"):
        fn([0.5, float("inf")])
    with pytest.raises(ValueError, match="SF-BL-10738"):
        fn([82.0, 0.5])                           # out of range (SF-BL-10738)
    with pytest.raises(ValueError, match="SF-BL-8799"):
        fn([0.5, 0.6], weights=[float("nan"), 1.0])  # bad weights (SF-BL-8799)


# --- Neighbours: aggregate mean/median inherit; log-odds untouched ---

def test_aggregate_mean_median_string_member_mutated_raises():
    fc = [AgentForecast("a0", "q", 0.7), AgentForecast("a1", "q", 0.3)]
    assert aggregate(fc, strategy="mean") == 0.5
    assert aggregate(fc, strategy="median") == 0.5
    bad = AgentForecast("a", "q", 0.5)
    bad.probability = '0.7'  # mutated after construction, bypassing its screen
    for s in ("mean", "median"):
        with pytest.raises(ValueError, match=r"probs.*SF-BL-11093"):
            aggregate([bad, AgentForecast("b", "q", 0.3)], strategy=s)


def test_log_odds_pool_unchanged():
    with pytest.raises(ValueError, match="SF-BL-10233"):
        log_odds_pool([0.5, float("nan")])
    assert log_odds_pool([0.2, 0.8]) == pytest.approx(0.5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
