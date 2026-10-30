"""
test_nonfinite_weight.py — SF-BL-9506 verification tests.

Pre-registered done/kill criteria (TASK.md):
  "Falsifiable: aggregate raises or drops-and-logs any member whose
   historical_brier_weight is non-finite and the repro pools the two
   finite members only; kill if every weight producer is proven to emit
   only finite weights with a tested invariant at SubAgentForecast
   construction."

Kill condition is NOT met (see PRODUCER_AUDIT.md): SubAgentForecast is a
plain dataclass with no __post_init__ — it accepts NaN/+-inf weights at
construction — and its only in-repo weight producer is the benchmark
fixture list in alpha_kelly.py itself, passing bare finite literals with
no finiteness invariant anywhere. So the fix applies at
aggregate_fleet_forecasts: members with a non-finite
historical_brier_weight (NaN, +-inf, non-numeric, bool) are
dropped-and-logged (UserWarning "SF-BL-9506: ..." naming the dropped
agent_ids and reporting "pooling n=<kept> of <total>") BEFORE any
clipping, dispersion, branch selection, weighting or pooling, composed
with the SF-BL-9439 probability filter (a member bad on either field is
dropped once). Finite negative weights (SF-BL-9147) and finite zero
weights (SF-BL-8901 zero-sum) are separate preregs and pass untouched.

Pre-fix reproduction (verified live 2026-10-21, before the patch;
see repro.py in this directory):
  probs 0.90/0.30/0.30, one NaN weight (any position)
      -> (0.98, 0.282842712474619)   [the 0.98 cap, dispersion 0.283]
  clean pool of the two finite members (NaN weight on a 0.30 member)
      -> (0.6077590814688054, 0.30000000000000004)
  clean pool of the two finite members (NaN weight on the 0.90 member)
      -> (0.2859610930263783, 0.0)

Run:  python3 test_nonfinite_weight.py   (exit 0 iff all pass)
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


# ---- 0. Kill-condition probe: the constructor validates nothing ------------

def test_constructor_does_not_validate_weight_finiteness():
    for bad in (NAN, INF, -INF):
        f = _f("x", "fam", 0.5, bad)
        assert f.historical_brier_weight is bad or f.historical_brier_weight != f.historical_brier_weight


# ---- 1. Prereg fixture: probs 0.90/0.30/0.30, one NaN weight ---------------

def test_prereg_nan_weight_any_position_equals_clean_two():
    probs = [0.90, 0.30, 0.30]
    expected = {0: (0.2859610930263783, 0.0),
                1: (0.6077590814688054, 0.30000000000000004),
                2: (0.6077590814688054, 0.30000000000000004)}
    for pos in range(3):
        members = [_f(f"a{i+1}", f"fam{i+1}", p, NAN if i == pos else 1.0)
                   for i, p in enumerate(probs)]
        clean = [m for i, m in enumerate(
            [_f(f"a{i+1}", f"fam{i+1}", p) for i, p in enumerate(probs)]) if i != pos]
        result, msgs = _agg(members)
        clean_result, clean_msgs = _agg(clean)
        # Never the pre-fix (0.98, 0.2828...).
        assert result[0] != 0.98 and abs(result[1] - 0.282842712474619) > 1e-9 or pos == 0 and result[1] == 0.0, result
        assert _close(result, expected[pos]), (pos, result)
        assert _close(result, clean_result), (pos, result, clean_result)
        assert clean_msgs == [], clean_msgs
        assert any("SF-BL-9506" in m and "n=2 of 3" in m and f"a{pos+1}" in m
                   for m in msgs), (pos, msgs)


def test_prereg_inf_weights_dropped():
    for bad in (INF, -INF):
        members = [_f("a1", "fam1", 0.90), _f("a2", "fam2", 0.30),
                   _f("a3", "fam3", 0.30, bad)]
        result, msgs = _agg(members)
        assert _close(result, (0.6077590814688054, 0.30000000000000004)), (bad, result)
        assert any("SF-BL-9506" in m and "n=2 of 3" in m and "a3" in m for m in msgs), (bad, msgs)


def test_nonfinite_weight_types_dropped():
    # Non-numeric weights raised a bare TypeError from sum() pre-fix;
    # bool is not a weight under the repo's helper convention.
    for bad in ("high", None, True):
        members = [_f("a1", "fam1", 0.90), _f("a2", "fam2", 0.30),
                   _f("a3", "fam3", 0.30, bad)]
        result, msgs = _agg(members)
        assert _close(result, (0.6077590814688054, 0.30000000000000004)), (bad, result)
        assert any("SF-BL-9506" in m and "a3" in m for m in msgs), (bad, msgs)


# ---- 2. Weighted branch: kept >=3, weights renormalize over kept only ------

def test_logodds_nan_weight_dropped_weights_renormalized():
    members = [_f("a1", "fam1", 0.70, 1.2), _f("a2", "fam2", 0.60, 1.1),
               _f("a3", "fam3", 0.65, 1.0), _f("a4", "fam4", 0.55, NAN)]
    result, msgs = _agg(members)
    clean_result, _ = _agg(members[:3])
    assert _close(result, (0.7009129085159926, 0.04082482904638629)), result
    assert _close(result, clean_result), (result, clean_result)
    assert any("n=3 of 4" in m and "a4" in m for m in msgs), msgs


# ---- 3. Median / trimmed branches: drop applies in every branch ------------

def test_median_branch_nan_weight_dropped():
    members = [_f("a1", "fam1", 0.70, 1.0, False), _f("a2", "fam2", 0.60, 1.0, False),
               _f("a3", "fam3", 0.65, 1.0, False), _f("a4", "fam4", 0.55, NAN, False)]
    result, msgs = _agg(members)
    clean_result, _ = _agg(members[:3])
    assert _close(result, clean_result), (result, clean_result)
    assert _close(result, (0.6611811695087533, 0.04082482904638629)), result
    assert any("n=3 of 4" in m for m in msgs), msgs


def test_trimmed_median_nan_weight_dropped_any_position():
    base = [0.5, 0.6, 0.7, 0.6, 0.5]
    clean = [_f(f"a{i}", "fam1", p) for i, p in enumerate(base)]
    clean_result, _ = _agg(clean)
    assert _close(clean_result, (0.5719288328804474, 0.0748331477354788)), clean_result
    for pos in (0, 2, 5):
        members = [_f(f"a{i}", "fam1", p) for i, p in enumerate(base)]
        members.insert(pos, _f("bad", "fam1", 0.99, NAN))
        result, msgs = _agg(members)
        assert _close(result, clean_result), (pos, result, clean_result)
        assert any("n=5 of 6" in m for m in msgs), (pos, msgs)


# ---- 4. Drop semantics: reporting, all-bad, mutation, composition ----------

def test_multiple_drops_all_named_n_reported():
    members = [_f("a1", "fam1", 0.70), _f("bad1", "fam2", 0.60, NAN),
               _f("a2", "fam2", 0.60), _f("bad2", "fam3", 0.65, INF),
               _f("a3", "fam3", 0.65)]
    result, msgs = _agg(members)
    assert _close(result, (0.6989517116705906, 0.04082482904638629)), result
    assert any("n=3 of 5" in m and "bad1" in m and "bad2" in m for m in msgs), msgs


def test_all_nonfinite_weights_raise_and_empty_still_raises():
    for members in ([_f("a1", "fam1", 0.5, NAN)],
                    [_f("a1", "fam1", 0.90, NAN), _f("a2", "fam2", 0.30, INF),
                     _f("a3", "fam3", 0.30, -INF)],
                    []):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                ak.aggregate_fleet_forecasts(members)
            except ValueError:
                continue
        raise AssertionError(f"no ValueError for {members!r}")


def test_mutated_member_dropped():
    m = _f("a3", "fam3", 0.30)
    m.historical_brier_weight = NAN
    members = [_f("a1", "fam1", 0.90), _f("a2", "fam2", 0.30), m]
    result, msgs = _agg(members)
    assert _close(result, (0.6077590814688054, 0.30000000000000004)), result
    assert any("n=2 of 3" in m2 and "a3" in m2 for m2 in msgs), msgs


def test_single_survivor_pools_alone_dispersion_zero():
    members = [_f("a1", "fam1", 0.60), _f("a2", "fam2", 0.70, NAN)]
    result, msgs = _agg(members)
    assert _close(result, (0.6077590814688054, 0.0)), result
    assert any("n=1 of 2" in m for m in msgs), msgs


def test_mixed_prob_and_weight_drops_compose():
    # SF-BL-9439 and SF-BL-9506 compose: one member dropped for a NaN
    # probability, another for a NaN weight; final n reported by both.
    members = [_f("badp", "fam1", NAN), _f("badw", "fam2", 0.65, NAN),
               _f("a1", "fam3", 0.70), _f("a2", "fam4", 0.60),
               _f("a3", "fam5", 0.65)]
    result, msgs = _agg(members)
    clean_result, _ = _agg(members[2:])
    assert _close(result, (0.6989517116705906, 0.04082482904638629)), result
    assert _close(result, clean_result), (result, clean_result)
    assert any("SF-BL-9439" in m and "badp" in m and "n=3 of 5" in m for m in msgs), msgs
    assert any("SF-BL-9506" in m and "badw" in m and "n=3 of 5" in m for m in msgs), msgs


def test_member_bad_on_both_fields_dropped_once():
    members = [_f("both", "fam1", NAN, NAN), _f("a1", "fam2", 0.70),
               _f("a2", "fam3", 0.60)]
    result, msgs = _agg(members)
    assert _close(result, (0.6611811695087533, 0.05)), result
    assert any("SF-BL-9439" in m and "both" in m for m in msgs), msgs
    assert not any("SF-BL-9506" in m for m in msgs), msgs


# ---- 5. Scope pins: neighbouring preregs untouched ---------------------------

def test_finite_negative_and_zero_weights_pass_filter():
    # Negative finite weight: SF-BL-9147's prereg — no SF-BL-9506 drop,
    # no warning, behaviour unchanged (anti-pooling value preserved).
    members = [_f("a1", "fam1", 0.70, 1.0), _f("a2", "fam2", 0.60, -0.5),
               _f("a3", "fam3", 0.65, 1.0)]
    result, msgs = _agg(members)
    assert _close(result, (0.7571732864368983, 0.04082482904638629)), result
    assert msgs == [], msgs
    # Zero finite weight in the (weight-free) median branch: untouched.
    members = [_f("a1", "fam1", 0.70, 0.0), _f("a2", "fam1", 0.60, 0.0)]
    result, msgs = _agg(members)
    assert _close(result, (0.6611811695087533, 0.05)), result
    assert msgs == [], msgs


# ---- 6. No regression on clean pools ---------------------------------------

def test_clean_pools_unchanged():
    cases = [
        ([_f("a1", "fam1", 0.90), _f("a2", "fam2", 0.30), _f("a3", "fam3", 0.30)],
         (0.5563059058407515, 0.282842712474619)),
        ([_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60), _f("a3", "fam3", 0.65)],
         (0.6989517116705906, 0.04082482904638629)),
        ([_f("a1", "fam1", 0.70), _f("a2", "fam1", 0.60)],
         (0.6611811695087533, 0.05)),
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
