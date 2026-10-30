"""
test_out_of_range_member.py — SF-BL-10723 verification tests.

Pre-registered done/kill criteria (TASK.md):
  "Fix drops (and logs, like SF-BL-9439) any member whose raw_probability
   is finite but outside [0,1] before clipping/pooling, leaving in-range
   panels bit-identical; kill if a census of in-repo aggregate callers
   shows every producer already guarantees [0,1] members at schema
   level."

Kill condition is NOT met (see CALLER_CENSUS.md): SubAgentForecast is a
plain dataclass with no __post_init__ — it accepts 82.0 / -0.5 at
construction (proven in test_constructor_does_not_validate_range) — its
only in-repo production caller is the benchmark's bare-literal fixture
list, and data-driven fleet producers convert with a bare float() and
no range check. So the fix applies at aggregate_fleet_forecasts: finite
out-of-range members are dropped-and-logged (UserWarning naming
SF-BL-10723, the dropped agent_ids and "pooling n=<kept> of <total>")
BEFORE any clipping, dispersion, branch selection, weighting or
pooling; the pooled result is therefore exactly the clean-pool result
for the remaining members, in every branch. The domain is the CLOSED
interval [0, 1]: endpoints 0.0 / 1.0 are kept, bit-identical to pre-fix.

Pre-fix reproduction (verified live 2026-10-23, before the patch;
see repro.py in this directory):
  log-odds 0.60/0.60/82.0  -> (0.9397383611926594, 0.18620478571245752)
  log-odds 0.60/0.60/-0.5  -> (0.11742244697218977, 0.28048568987066386)
  all out-of-range 82/-0.5 -> (0.5, 0.495)  (no error)
  clean    0.60/0.60       -> (0.6077590814688054, 0.0)
  clean    0.60/0.60/0.60 distinct -> (0.6335270291357783, 0.0)

Run:  python3 test_out_of_range_member.py   (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_nan_forecast.py.
"""
import io
import os
import sys
import warnings
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(
    os.path.expanduser("~/workspace/superforecaster"), "superforecaster"))

import alpha_kelly as ak  # noqa: E402

PASS = []
NAN = float("nan")
INF = float("inf")


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _f(agent, fam, p, weight=1.0, independent=True):
    return ak.SubAgentForecast(agent, fam, p, weight, independent)


def _agg(forecasts):
    """Aggregate with warnings captured; returns (result, [messages])."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = ak.aggregate_fleet_forecasts(forecasts)
    return result, [str(w.message) for w in caught]


def _close(a, b, tol=1e-12):
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def _range_msgs(msgs):
    return [m for m in msgs if "SF-BL-10723" in m]


# ---- 0. Kill-condition probe: the constructor validates nothing ------------

def test_constructor_does_not_validate_range():
    for bad in (82.0, -0.5, 1.5, -0.2, 10**400):
        f = _f("x", "fam", bad)
        assert f.raw_probability == bad


def test_range_helper_classification():
    oor = ak._is_out_of_range_member_probability
    for p in (82.0, -0.5, 1.5, 2.0, -0.2, 1.0000001, -1e-9, 1e308, -1e308,
              10**400, -(10**400), 5, -1):
        assert oor(p) is True, p
    # In-range (endpoints included), non-finite (SF-BL-9439's case), bool
    # and non-numeric values are NOT this task's drop.
    for p in (0.0, 1.0, 0.5, 0.9999999, 1e-9, 0, 1, NAN, INF, -INF,
              True, False, "82", None, [0.5]):
        assert oor(p) is False, p


# ---- 1. Title fixtures, log-odds branch ------------------------------------

def test_title_percent_member_dropped_equals_remaining_pool():
    members = [_f("a1", "fam1", 0.60), _f("a2", "fam2", 0.60),
               _f("a3", "fam3", 82.0)]
    result, msgs = _agg(members)
    # Never the pre-fix (0.9397, 0.1862): exactly the pool of the two
    # remaining members (median branch, 2 distinct families remain).
    assert _close(result, (0.6077590814688054, 0.0)), result
    assert any("n=2 of 3" in m and "a3=82.0" in m for m in _range_msgs(msgs)), msgs


def test_title_negative_member_dropped_equals_remaining_pool():
    members = [_f("a1", "fam1", 0.60), _f("a2", "fam2", 0.60),
               _f("a3", "fam3", -0.5)]
    result, msgs = _agg(members)
    # Never the pre-fix (0.1174, 0.2805).
    assert _close(result, (0.6077590814688054, 0.0)), result
    assert any("n=2 of 3" in m and "a3=-0.5" in m for m in _range_msgs(msgs)), msgs


def test_logodds_panel_stays_logodds_after_drop():
    # >=3 distinct families among the KEPT members: the surviving branch
    # really is the weighted log-odds pool — the title's clean 0.6335.
    members = [_f("a1", "fam1", 0.60), _f("a2", "fam2", 0.60),
               _f("a3", "fam3", 0.60), _f("a4", "fam4", 82.0)]
    result, msgs = _agg(members)
    assert _close(result, (0.6335270291357783, 0.0)), result
    assert any("n=3 of 4" in m for m in _range_msgs(msgs)), msgs


def test_logodds_out_of_range_variants_dropped():
    clean = [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60), _f("a3", "fam3", 0.65)]
    clean_result, _ = _agg(clean)
    assert _close(clean_result, (0.6989517116705906, 0.04082482904638629))
    for bad in (1.5, 2.0, -0.2, 82.0, 1.0000001, -1e-9, 1e308, -1e308, 10**400):
        members = clean + [_f("a4", "fam4", bad)]
        result, msgs = _agg(members)
        assert _close(result, clean_result), (bad, result, clean_result)
        assert any("n=3 of 4" in m and "a4" in m for m in _range_msgs(msgs)), (bad, msgs)


def test_logodds_weighted_out_of_range_dropped_weights_renormalized():
    # The dropped member's weight (99.0) must leave the pool entirely.
    members = [_f("a1", "fam1", 0.70, 1.2), _f("a2", "fam2", 0.60, 1.1),
               _f("a3", "fam3", 0.65, 1.0), _f("a4", "fam4", 82.0, 99.0)]
    result, _ = _agg(members)
    assert _close(result, (0.7009129085159926, 0.04082482904638629)), result


# ---- 2. Median / trimmed branches ------------------------------------------

def test_median_branch_out_of_range_dropped():
    # Pre-fix the clipped 0.995 BECAME the median of this 4-member panel:
    # (0.6611811695087533, 0.18508021909431596). Post-fix: clean 3-pool.
    members = [_f("a1", "fam1", 0.50), _f("a2", "fam1", 0.60),
               _f("a3", "fam1", 0.70), _f("a4", "fam1", 82.0)]
    result, msgs = _agg(members)
    assert _close(result, (0.6077590814688054, 0.08164965809277258)), result
    assert any("n=3 of 4" in m for m in _range_msgs(msgs)), msgs


def test_median_branch_via_shared_context_out_of_range_dropped():
    members = [_f("a1", "fam1", 0.70, 1.0, False), _f("a2", "fam2", 0.60, 1.0, False),
               _f("a3", "fam3", 0.65, 1.0, False), _f("a4", "fam4", -0.5, 1.0, False)]
    result, _ = _agg(members)
    clean, _ = _agg(members[:3])
    assert _close(result, clean), (result, clean)
    assert result[0] < 0.70, result


def test_trimmed_out_of_range_dropped_any_position():
    base = [0.5, 0.6, 0.7, 0.6, 0.5]
    clean = [_f(f"a{i}", "fam1", p) for i, p in enumerate(base)]
    clean_result, _ = _agg(clean)
    assert _close(clean_result, (0.5719288328804474, 0.0748331477354788))
    for bad in (82.0, -0.5):
        for pos in (0, 2, 5):
            members = [_f(f"a{i}", "fam1", p) for i, p in enumerate(base)]
            members.insert(pos, _f("bad", "fam1", bad))
            result, msgs = _agg(members)
            assert _close(result, clean_result), (bad, pos, result, clean_result)
            assert any("n=5 of 6" in m for m in _range_msgs(msgs)), (bad, pos, msgs)


# ---- 3. Boundaries: [0,1] is closed, in-range panels bit-identical ---------

def test_endpoints_zero_and_one_kept_bit_identical():
    # 1.0 / 0.0 are IN [0,1]: kept, no warning, and the result equals
    # the pre-fix value (which coincidentally equals the 82.0 / -0.5
    # pre-fix values, because the internal clip maps all of them to
    # 0.995 / 0.005 — the endpoints are legitimate extreme forecasts).
    members = [_f("a1", "fam1", 0.60), _f("a2", "fam2", 0.60), _f("a3", "fam3", 1.0)]
    result, msgs = _agg(members)
    assert _close(result, (0.9397383611926594, 0.18620478571245752)), result
    assert msgs == [], msgs
    members = [_f("a1", "fam1", 0.60), _f("a2", "fam2", 0.60), _f("a3", "fam3", 0.0)]
    result, msgs = _agg(members)
    assert _close(result, (0.11742244697218977, 0.28048568987066386)), result
    assert msgs == [], msgs


def test_just_inside_range_kept_no_warning():
    for p in (0.9999999, 1e-9, 0.999, 0.001):
        members = [_f("a1", "fam1", 0.60), _f("a2", "fam2", 0.60), _f("a3", "fam3", p)]
        result, msgs = _agg(members)
        clean, _ = _agg([_f("a1", "fam1", 0.60), _f("a2", "fam2", 0.60),
                         _f("a3", "fam3", p)])
        assert _close(result, clean), (p, result, clean)
        assert msgs == [], (p, msgs)


def test_clean_pools_unchanged():
    cases = [
        ([_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60), _f("a3", "fam3", 0.65)],
         (0.6989517116705906, 0.04082482904638629)),
        ([_f("a1", "fam1", 0.70), _f("a2", "fam1", 0.60)],
         (0.6611811695087533, 0.05)),
        ([_f(f"a{i}", "fam1", p) for i, p in enumerate([0.5, 0.6, 0.7, 0.6, 0.5])],
         (0.5719288328804474, 0.0748331477354788)),
        ([_f("a1", "fam1", 0.60)], (0.6077590814688054, 0.0)),
        ([_f("a1", "fam1", 0.60), _f("a2", "fam2", 0.60), _f("a3", "fam3", 0.60)],
         (0.6335270291357783, 0.0)),
    ]
    for members, expected in cases:
        result, msgs = _agg(members)
        assert _close(result, expected), (result, expected)
        assert msgs == [], msgs  # no warning when nothing was dropped


# ---- 4. Drop semantics: composition, all-bad, mutation, single survivor ----

def test_multiple_drops_all_named_n_reported():
    members = [_f("a1", "fam1", 0.70), _f("bad1", "fam2", 82.0),
               _f("a2", "fam2", 0.60), _f("bad2", "fam3", -0.5),
               _f("a3", "fam3", 0.65)]
    result, msgs = _agg(members)
    assert _close(result, (0.6989517116705906, 0.04082482904638629)), result
    assert any("n=3 of 5" in m and "bad1" in m and "bad2" in m
               for m in _range_msgs(msgs)), msgs


def test_mixed_drops_compose_all_three_filters():
    # NaN prob (9439) + out-of-range prob (10723) + NaN weight (9506):
    # three warnings, each reporting the FINAL pooled n=2 of 5, and the
    # pool is exactly the two clean members.
    members = [_f("a1", "fam1", 0.70), _f("badp", "fam2", NAN),
               _f("a2", "fam2", 0.60), _f("badr", "fam3", 82.0),
               _f("badw", "fam3", 0.65, NAN)]
    result, msgs = _agg(members)
    assert _close(result, (0.6611811695087533, 0.05)), result
    assert any("SF-BL-9439" in m and "n=2 of 5" in m and "badp" in m for m in msgs), msgs
    assert any("SF-BL-9506" in m and "n=2 of 5" in m and "badw" in m for m in msgs), msgs
    assert any("n=2 of 5" in m and "badr" in m for m in _range_msgs(msgs)), msgs


def test_member_bad_on_range_and_weight_dropped_once():
    members = [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60),
               _f("bad", "fam3", 82.0, NAN)]
    result, msgs = _agg(members)
    assert _close(result, (0.6611811695087533, 0.05)), result
    assert any("bad" in m for m in _range_msgs(msgs)), msgs
    assert not any("SF-BL-9506" in m for m in msgs), msgs  # dropped once


def test_nonfinite_and_bool_and_string_not_range_drops():
    # +-inf belongs to SF-BL-9439, never to SF-BL-10723; bool / string
    # probabilities likewise stay on the pre-existing 9439 path.
    for bad in (INF, -INF, True, "82"):
        members = [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60),
                   _f("a3", "fam3", bad)]
        result, msgs = _agg(members)
        assert _close(result, (0.6611811695087533, 0.05)), (bad, result)
        assert _range_msgs(msgs) == [], (bad, msgs)
        assert any("SF-BL-9439" in m and "n=2 of 3" in m for m in msgs), (bad, msgs)


def test_all_out_of_range_raises_and_empty_still_raises():
    for members in ([_f("a1", "fam1", 82.0)],
                    [_f("a1", "fam1", 82.0), _f("a2", "fam2", -0.5)],
                    [_f("a1", "fam1", 82.0), _f("a2", "fam2", NAN)],
                    []):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                ak.aggregate_fleet_forecasts(members)
            except ValueError:
                continue
        raise AssertionError(f"no ValueError for {members!r}")


def test_mutated_member_dropped():
    m = _f("a3", "fam3", 0.65)
    m.raw_probability = 82.0  # mutated after construction (percent slip)
    members = [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60), m]
    result, msgs = _agg(members)
    assert _close(result, (0.6611811695087533, 0.05)), result
    assert any("n=2 of 3" in m2 and "a3" in m2 for m2 in _range_msgs(msgs)), msgs


def test_single_survivor_pools_alone_dispersion_zero():
    members = [_f("a1", "fam1", 0.60), _f("a2", "fam2", 82.0)]
    result, msgs = _agg(members)
    assert _close(result, (0.6077590814688054, 0.0)), result
    assert any("n=1 of 2" in m for m in _range_msgs(msgs)), msgs


def test_out_of_range_never_inflates_dispersion():
    # Dispersion is over kept members only: the title fixture reports
    # 0.0 (clean), never the pre-fix 0.1862.
    members = [_f("a1", "fam1", 0.60), _f("a2", "fam2", 0.60), _f("a3", "fam3", 82.0)]
    result, _ = _agg(members)
    assert result[1] == 0.0, result


def test_benchmark_fixture_unchanged():
    buf = io.StringIO()
    with redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    # Recorded from the pre-patch code this run (2026-10-23).
    assert '"north_star_score": -0.1611' in out, out[-2000:]
    assert '"north_star_score": -0.1499' in out, out[-2000:]


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for fn in ALL_TESTS:
        try:
            fn()
            check(fn.__name__, True)
        except AssertionError as e:
            check(fn.__name__, False, str(e))
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
