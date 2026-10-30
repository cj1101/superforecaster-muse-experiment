"""
test_nan_member_forecast.py — SF-BL-9439 verification tests.

Pre-registered done/kill criteria (TASK.md):
  "Reject or drop-and-log non-finite member probabilities before pooling
   (pool of the remaining members only, n reported), fixtured with a NaN
   member in both the log-odds and median branches; kill if
   SubAgentForecast construction is proven to validate finiteness at
   every production call site (then move the fixture to that
   constructor)."

Kill condition is NOT met (see CALLSITE_AUDIT.md): SubAgentForecast is a
plain dataclass with no __post_init__ — it accepts NaN/+inf/-inf at
construction (proven in test_constructor_does_not_validate_finiteness) —
and its only in-repo production call site is the benchmark fixture list
in alpha_kelly.py itself, passing bare finite literals with no finiteness
validation anywhere. So the fix applies at aggregate_fleet_forecasts:
non-finite members are dropped-and-logged (UserWarning naming the dropped
agent_ids and reporting "pooling n=<kept> of <total>") BEFORE any
clipping, dispersion, branch selection, weighting or pooling; the pooled
result is therefore exactly the clean-pool result for the remaining
members, in both branches.

Pre-fix reproduction (verified live 2026-10-21, before the patch;
see repro.py in this directory):
  log-odds 0.70/0.60/NaN -> (0.9500613119274202, 0.16768025127207636)
  median   0.70/0.60/NaN -> (0.7140389069736217, 0.16768025127207636)
  clean    0.70/0.60     -> (0.6611811695087533, 0.05)

Run:  python3 test_nan_member_forecast.py   (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_fee_rate.py.
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


# ---- 0. Kill-condition probe: the constructor validates nothing ------------

def test_constructor_does_not_validate_finiteness():
    # Kill fires only if construction validates finiteness at every call
    # site. It validates nothing: NaN / +-inf construct fine, so the fix
    # must live in aggregate_fleet_forecasts (and must also catch members
    # mutated to NaN after construction — see test_mutated_member_dropped).
    for bad in (NAN, INF, -INF):
        f = _f("x", "fam", bad)
        assert f.raw_probability is bad or f.raw_probability != f.raw_probability


# ---- 1. Prereg fixture, log-odds branch ------------------------------------

def test_prereg_logodds_nan_dropped_equals_clean():
    members = [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60), _f("a3", "fam3", NAN)]
    clean = [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60)]
    result, msgs = _agg(members)
    clean_result, clean_msgs = _agg(clean)
    # Never the pre-fix (0.9501, 0.168): not above every real member.
    assert result[0] < 0.70, result
    assert _close(result, (0.6611811695087533, 0.05)), result
    assert _close(result, clean_result), (result, clean_result)
    assert clean_msgs == [], clean_msgs
    # n reported: pooled n=2 of 3, dropped agent named.
    assert any("SF-BL-9439" in m and "n=2 of 3" in m and "a3" in m for m in msgs), msgs


def test_logodds_nan_among_three_valid_equals_clean_logodds():
    # >=3 distinct families + independent search among the KEPT members:
    # the surviving branch really is the weighted log-odds pool.
    members = [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60),
               _f("a3", "fam3", 0.65), _f("a4", "fam4", NAN)]
    clean = members[:3]
    result, msgs = _agg(members)
    clean_result, _ = _agg(clean)
    assert _close(result, (0.6989517116705906, 0.04082482904638629)), result
    assert _close(result, clean_result), (result, clean_result)
    assert any("n=3 of 4" in m for m in msgs), msgs


def test_logodds_inf_members_dropped():
    for bad in (INF, -INF):
        members = [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60),
                   _f("a3", "fam3", 0.65), _f("a4", "fam4", bad)]
        result, msgs = _agg(members)
        assert _close(result, (0.6989517116705906, 0.04082482904638629)), (bad, result)
        assert any("n=3 of 4" in m and "a4" in m for m in msgs), (bad, msgs)


def test_logodds_weighted_nan_dropped_weights_renormalized():
    # The dropped member's weight must leave the pool entirely: the kept
    # weights renormalize over the kept members only.
    members = [_f("a1", "fam1", 0.70, 1.2), _f("a2", "fam2", 0.60, 1.1),
               _f("a3", "fam3", 0.65, 1.0), _f("a4", "fam4", NAN, 99.0)]
    result, _ = _agg(members)
    assert _close(result, (0.7009129085159926, 0.04082482904638629)), result


# ---- 2. Prereg fixture, median / trimmed branches ---------------------------

def test_prereg_median_nan_dropped_equals_clean():
    members = [_f("a1", "fam1", 0.70), _f("a2", "fam1", 0.60), _f("a3", "fam1", NAN)]
    result, msgs = _agg(members)
    # Never the pre-fix median 0.7140...; exactly the clean median pool.
    assert _close(result, (0.6611811695087533, 0.05)), result
    assert any("n=2 of 3" in m for m in msgs), msgs


def test_median_branch_via_shared_context_nan_dropped():
    # Median branch selected via has_independent_search=False even with
    # >=3 distinct families; a NaN member must still be dropped first.
    members = [_f("a1", "fam1", 0.70, 1.0, False), _f("a2", "fam2", 0.60, 1.0, False),
               _f("a3", "fam3", 0.65, 1.0, False), _f("a4", "fam4", NAN, 1.0, False)]
    result, _ = _agg(members)
    clean, _ = _agg(members[:3])
    assert _close(result, clean), (result, clean)
    assert result[0] < 0.70, result


def test_trimmed_median_nan_dropped_any_position():
    base = [0.5, 0.6, 0.7, 0.6, 0.5]
    clean = [_f(f"a{i}", "fam1", p) for i, p in enumerate(base)]
    clean_result, _ = _agg(clean)
    assert _close(clean_result, (0.5719288328804474, 0.0748331477354788)), clean_result
    for pos in (0, 2, 5):
        members = [_f(f"a{i}", "fam1", p) for i, p in enumerate(base)]
        members.insert(pos, _f("bad", "fam1", NAN))
        result, msgs = _agg(members)
        assert _close(result, clean_result), (pos, result, clean_result)
        assert any("n=5 of 6" in m for m in msgs), (pos, msgs)


def test_median_inf_members_dropped():
    for bad in (INF, -INF):
        members = [_f("a1", "fam1", 0.70), _f("a2", "fam1", 0.60), _f("a3", "fam1", bad)]
        result, _ = _agg(members)
        assert _close(result, (0.6611811695087533, 0.05)), (bad, result)


# ---- 3. Drop semantics: reporting, all-bad, mutation, single survivor ------

def test_multiple_drops_all_named_n_reported():
    members = [_f("a1", "fam1", 0.70), _f("bad1", "fam2", NAN),
               _f("a2", "fam2", 0.60), _f("bad2", "fam3", INF),
               _f("a3", "fam3", 0.65)]
    result, msgs = _agg(members)
    assert _close(result, (0.6989517116705906, 0.04082482904638629)), result
    assert any("n=3 of 5" in m and "bad1" in m and "bad2" in m for m in msgs), msgs


def test_all_nonfinite_raises_and_empty_still_raises():
    for members in ([_f("a1", "fam1", NAN)],
                    [_f("a1", "fam1", NAN), _f("a2", "fam2", INF)],
                    []):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                ak.aggregate_fleet_forecasts(members)
            except ValueError:
                continue
        raise AssertionError(f"no ValueError for {members!r}")


def test_mutated_member_dropped():
    # Aggregate-time filtering, not constructor-only: a member mutated to
    # NaN after construction must be dropped exactly like a born-NaN one.
    m = _f("a3", "fam3", 0.65)
    m.raw_probability = NAN
    members = [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60), m]
    result, msgs = _agg(members)
    assert _close(result, (0.6611811695087533, 0.05)), result
    assert any("n=2 of 3" in m2 and "a3" in m2 for m2 in msgs), msgs


def test_single_survivor_pools_alone_dispersion_zero():
    members = [_f("a1", "fam1", 0.60), _f("a2", "fam2", NAN)]
    result, msgs = _agg(members)
    assert _close(result, (0.6077590814688054, 0.0)), result
    assert any("n=1 of 2" in m for m in msgs), msgs


def test_nonfinite_never_inflates_dispersion():
    # Dispersion is over kept members only: the prereg fixture reports
    # 0.05 (clean), never the pre-fix 0.168 that hid under Gate 2's 0.22.
    members = [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60), _f("a3", "fam3", NAN)]
    result, _ = _agg(members)
    assert abs(result[1] - 0.05) < 1e-12, result


# ---- 4. No regression on clean pools ---------------------------------------

def test_clean_pools_unchanged():
    cases = [
        ([_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60), _f("a3", "fam3", 0.65)],
         (0.6989517116705906, 0.04082482904638629)),
        ([_f("a1", "fam1", 0.70), _f("a2", "fam1", 0.60)],
         (0.6611811695087533, 0.05)),
        ([_f(f"a{i}", "fam1", p) for i, p in enumerate([0.5, 0.6, 0.7, 0.6, 0.5])],
         (0.5719288328804474, 0.0748331477354788)),
        ([_f("a1", "fam1", 0.60)], (0.6077590814688054, 0.0)),
    ]
    for members, expected in cases:
        result, msgs = _agg(members)
        assert _close(result, expected), (result, expected)
        assert msgs == [], msgs  # no warning when nothing was dropped


def test_benchmark_fixture_unchanged():
    buf = io.StringIO()
    with redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    assert '"north_star_score": 0.1059' in out, out[-2000:]
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
