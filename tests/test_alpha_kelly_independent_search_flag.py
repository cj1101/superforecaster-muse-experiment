"""
test_independent_search_flag.py — SF-BL-10848 verification tests.

Pre-registered done/kill criteria (TASK.md):
  "Done when aggregate drops-and-logs (or named-rejects) any member whose
   has_independent_search is not a strict bool before branch selection,
   with strict-bool panels byte-identical; kill if a producer census
   shows the flag is only ever written as a real bool by in-repo
   constructors."

Kill condition is NOT met (see PRODUCER_CENSUS.md): SubAgentForecast is
a plain dataclass with no __post_init__ — it accepts 'False', 1 and None
as the flag at construction (proven in
test_constructor_does_not_validate_flag_type) — its only in-repo
production constructors (the benchmark fixture lists) omit the flag
entirely, and the mutable dataclass plus **dict / JSON-style unpacking
deliver serialized stand-ins ('False' from JSON/CSV text) unchecked.
So the fix applies at aggregate_fleet_forecasts: a member whose
has_independent_search is not a strict bool is dropped-and-logged
(UserWarning naming SF-BL-10848, the dropped agent_id=flag and
"pooling n=<kept> of <total>") BEFORE branch selection, dispersion or
pooling; the pooled result is therefore exactly the clean-pool result
for the remaining members. The flag is never coerced: 'False' does not
become False and 'True' does not become True — the member leaves whole.

Pre-fix reproduction (verified live 2026-10-23, before the patch;
see repro.py in this directory), TASK panel 0.70/0.60/0.50, 3 families:
  third member flag=False (bool) -> (0.6077590814688054, 0.08164965809277258)  [median]
  third member flag='False' (str) -> (0.6373182327645819, 0.08164965809277258)  [extremized log-odds]
Post-fix the 'False' member is dropped, so the panel pools exactly its
remaining members: (0.6611811695087533, 0.05).

Run:  python3 test_independent_search_flag.py   (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_nan_forecast.py.
"""
import io
import json
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

try:
    import numpy as _np
    _NP_BOOL = _np.bool_(False)
except Exception:  # pragma: no cover - numpy is present in this repo
    _NP_BOOL = None


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


CLEAN_PAIR = (0.6611811695087533, 0.05)            # 0.70/0.60 kept pair
TITLE_BOOL_FALSE = (0.6077590814688054, 0.08164965809277258)  # median, kept
TITLE_BOOL_TRUE = (0.6373182327645819, 0.08164965809277258)   # log-odds, kept


# ---- 0. Kill-condition probe: the constructor validates nothing ------------

def test_constructor_does_not_validate_flag_type():
    # Kill fires only if the flag is only ever written as a real bool by
    # in-repo constructors. The constructor validates nothing: string,
    # int and None flags construct fine, so the screen must live in
    # aggregate_fleet_forecasts (and must also catch members mutated to
    # a non-bool flag after construction — see test_mutated_member_dropped).
    for bad in ("False", "True", 1, 0, None):
        f = _f("x", "fam", 0.5, 1.0, bad)
        assert f.has_independent_search is bad or f.has_independent_search == bad


def test_helper_directly():
    assert ak._is_strict_bool_flag(True) is True
    assert ak._is_strict_bool_flag(False) is True
    for bad in ("False", "True", "false", "", 0, 1, 0.0, 1.0, None,
                [], [True], {}, object()):
        assert ak._is_strict_bool_flag(bad) is False, bad
    if _NP_BOOL is not None:
        assert ak._is_strict_bool_flag(_NP_BOOL) is False


# ---- 1. Prereg fixture: the TASK panel --------------------------------------

def test_title_panel_string_false_dropped_equals_clean():
    members = [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60),
               _f("a3", "fam3", 0.50, 1.0, "False")]
    clean = [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60)]
    result, msgs = _agg(members)
    clean_result, clean_msgs = _agg(clean)
    # Exactly the pool of the remaining members...
    assert _close(result, CLEAN_PAIR), result
    assert _close(result, clean_result), (result, clean_result)
    assert clean_msgs == [], clean_msgs
    # ...never either pre-fix value: not the extremized pool the truthy
    # string selected (0.6373) and not the bool-False-kept median pool
    # (0.6078) — the member is dropped, not coerced to False.
    assert not _close(result, TITLE_BOOL_TRUE), result
    assert not _close(result, TITLE_BOOL_FALSE), result
    # n reported: pooled n=2 of 3, dropped agent + flag named.
    assert any("SF-BL-10848" in m and "n=2 of 3" in m and "a3" in m
               and "'False'" in m for m in msgs), msgs
    # Provenance: the drop is counted and the branch flip is visible.
    p = result.provenance
    assert (p.n_input, p.n_kept, p.n_dropped, p.n_flag_dropped) == (3, 2, 1, 1), p
    assert p.branch_flipped is True, p


def test_title_panel_kept_panel_stays_logodds():
    # >=3 distinct families + independent search among the KEPT members:
    # the surviving branch really is the weighted log-odds pool.
    members = [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60),
               _f("a3", "fam3", 0.65), _f("a4", "fam4", 0.50, 1.0, "False")]
    result, msgs = _agg(members)
    assert _close(result, (0.6989517116705906, 0.04082482904638629)), result
    assert any("n=3 of 4" in m and "a4" in m for m in msgs), msgs
    assert result.provenance.n_flag_dropped == 1, result.provenance
    assert result.provenance.branch_flipped is False, result.provenance


# ---- 2. Every non-bool type is dropped, truthy and falsy alike --------------

def test_all_nonbool_variants_dropped_identically():
    variants = ["False", "True", "false", "true", "FALSE", "",
                0, 1, 0.0, 1.0, None, [], [True]]
    if _NP_BOOL is not None:
        variants.append(_NP_BOOL)
    for bad in variants:
        members = [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60),
                   _f("a3", "fam3", 0.50, 1.0, bad)]
        result, msgs = _agg(members)
        assert _close(result, CLEAN_PAIR), (bad, result)
        assert any("SF-BL-10848" in m and "n=2 of 3" in m for m in msgs), (bad, msgs)
        assert result.provenance.n_flag_dropped == 1, (bad, result.provenance)


def test_truthy_nonbools_not_treated_as_independent():
    # Pre-fix 'True' / 1 pooled via extremized log-odds and happened to
    # match the bool-True value; strictness is about type, so they are
    # dropped too — the result is the clean pair, not TITLE_BOOL_TRUE.
    for bad in ("True", 1, 1.0):
        members = [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60),
                   _f("a3", "fam3", 0.50, 1.0, bad)]
        result, _ = _agg(members)
        assert _close(result, CLEAN_PAIR), (bad, result)
        assert not _close(result, TITLE_BOOL_TRUE), (bad, result)


def test_falsy_nonbools_not_coerced_to_false():
    # Pre-fix None / '' / 0 took the median branch by falsiness and
    # returned the bool-False-kept value; dropped, they return the
    # clean pair instead — drop, never coerce.
    for bad in (None, "", 0, 0.0, []):
        members = [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60),
                   _f("a3", "fam3", 0.50, 1.0, bad)]
        result, _ = _agg(members)
        assert _close(result, CLEAN_PAIR), (bad, result)
        assert not _close(result, TITLE_BOOL_FALSE), (bad, result)


# ---- 3. Median / trimmed branches -------------------------------------------

def test_median_branch_via_shared_context_flag_dropped():
    # Median branch selected via has_independent_search=False (bool)
    # even with >=3 distinct families; a string-flag member must still
    # be dropped first.
    members = [_f("a1", "fam1", 0.70, 1.0, False),
               _f("a2", "fam2", 0.60, 1.0, False),
               _f("a3", "fam3", 0.65, 1.0, False),
               _f("a4", "fam4", 0.55, 1.0, "False")]
    result, msgs = _agg(members)
    clean, _ = _agg(members[:3])
    assert _close(result, clean), (result, clean)
    assert any("SF-BL-10848" in m and "n=3 of 4" in m for m in msgs), msgs


def test_trimmed_median_flag_dropped_any_position():
    base = [0.5, 0.6, 0.7, 0.6, 0.5]
    clean = [_f(f"a{i}", "fam1", p) for i, p in enumerate(base)]
    clean_result, _ = _agg(clean)
    assert _close(clean_result, (0.5719288328804474, 0.0748331477354788)), clean_result
    for pos in (0, 2, 5):
        members = [_f(f"a{i}", "fam1", p) for i, p in enumerate(base)]
        members.insert(pos, _f("bad", "fam1", 0.9, 1.0, 1))
        result, msgs = _agg(members)
        assert _close(result, clean_result), (pos, result, clean_result)
        assert any("n=5 of 6" in m and "bad" in m for m in msgs), (pos, msgs)


def test_weighted_flag_dropped_weights_renormalized():
    # The dropped member's weight must leave the pool entirely: the kept
    # weights renormalize over the kept members only.
    members = [_f("a1", "fam1", 0.70, 1.2), _f("a2", "fam2", 0.60, 1.1),
               _f("a3", "fam3", 0.65, 1.0), _f("a4", "fam4", 0.50, 99.0, "False")]
    result, _ = _agg(members)
    assert _close(result, (0.7009129085159926, 0.04082482904638629)), result


# ---- 4. Drop semantics: reporting, composition, all-bad, mutation -----------

def test_multiple_flag_drops_all_named_n_reported():
    members = [_f("a1", "fam1", 0.70), _f("bad1", "fam2", 0.60, 1.0, "False"),
               _f("a2", "fam2", 0.60), _f("bad2", "fam3", 0.65, 1.0, None),
               _f("a3", "fam3", 0.65)]
    result, msgs = _agg(members)
    assert _close(result, (0.6989517116705906, 0.04082482904638629)), result
    assert any("SF-BL-10848" in m and "n=3 of 5" in m and "bad1" in m
               and "bad2" in m for m in msgs), msgs
    assert result.provenance.n_flag_dropped == 2, result.provenance


def test_mixed_drops_compose_each_reason_counted():
    members = [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60),
               _f("badp", "fam3", NAN), _f("badw", "fam4", 0.65, INF),
               _f("badr", "fam5", 82.0), _f("badf", "fam6", 0.55, 1.0, "False")]
    result, msgs = _agg(members)
    assert _close(result, CLEAN_PAIR), result
    p = result.provenance
    assert (p.n_input, p.n_kept, p.n_dropped) == (6, 2, 4), p
    assert (p.n_prob_dropped, p.n_weight_dropped,
            p.n_range_dropped, p.n_flag_dropped) == (1, 1, 1, 1), p
    for tag in ("SF-BL-9439", "SF-BL-9506", "SF-BL-10723", "SF-BL-10848"):
        assert any(tag in m and "n=2 of 6" in m for m in msgs), (tag, msgs)


def test_member_bad_on_prob_and_flag_counted_once_as_prob():
    # The flag screen runs last (documented precedence): a member
    # already dropped for its probability keeps that classification.
    members = [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60),
               _f("a3", "fam3", NAN, 1.0, "False"),
               _f("a4", "fam4", 0.65)]
    result, msgs = _agg(members)
    assert _close(result, (0.6989517116705906, 0.04082482904638629)), result
    p = result.provenance
    assert (p.n_prob_dropped, p.n_flag_dropped, p.n_dropped) == (1, 0, 1), p
    assert not any("SF-BL-10848" in m for m in msgs), msgs


def test_member_bad_on_weight_and_flag_counted_once_as_weight():
    members = [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60),
               _f("a3", "fam3", 0.65), _f("a4", "fam4", 0.55, NAN, "False")]
    result, msgs = _agg(members)
    assert _close(result, (0.6989517116705906, 0.04082482904638629)), result
    p = result.provenance
    assert (p.n_weight_dropped, p.n_flag_dropped, p.n_dropped) == (1, 0, 1), p
    assert not any("SF-BL-10848" in m for m in msgs), msgs


def test_all_nonbool_flags_raise_and_empty_still_raises():
    for members in ([_f("a1", "fam1", 0.60, 1.0, "False")],
                    [_f("a1", "fam1", 0.60, 1.0, "False"),
                     _f("a2", "fam2", 0.65, 1.0, 1)],
                    []):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                ak.aggregate_fleet_forecasts(members)
            except ValueError as e:
                if members:
                    assert "SF-BL-10848" in str(e), e
                continue
        raise AssertionError(f"no ValueError for {members!r}")


def test_mutated_member_dropped():
    # Aggregate-time filtering, not constructor-only: a member mutated
    # to a string flag after construction must be dropped exactly like
    # a born-string one.
    m = _f("a3", "fam3", 0.50)
    m.has_independent_search = "False"
    members = [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60), m]
    result, msgs = _agg(members)
    assert _close(result, CLEAN_PAIR), result
    assert any("n=2 of 3" in m2 and "a3" in m2 for m2 in msgs), msgs


def test_serialized_dict_member_dropped_json_bool_kept():
    # The canonical producer path for this bug: a member rebuilt by
    # **dict unpacking of JSON text, where a bool serializes as the
    # string "False" when the source was CSV/config-style, and as a
    # real bool when the JSON carried a bare false.
    raw = json.loads('{"agent_id": "a3", "model_family": "fam3",'
                     ' "raw_probability": 0.50,'
                     ' "historical_brier_weight": 1.0,'
                     ' "has_independent_search": "False"}')
    members = [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60),
               ak.SubAgentForecast(**raw)]
    result, msgs = _agg(members)
    assert _close(result, CLEAN_PAIR), result
    assert any("SF-BL-10848" in m for m in msgs), msgs
    raw2 = json.loads('{"agent_id": "a3", "model_family": "fam3",'
                      ' "raw_probability": 0.50,'
                      ' "historical_brier_weight": 1.0,'
                      ' "has_independent_search": false}')
    assert raw2["has_independent_search"] is False  # real bool
    members2 = [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60),
                ak.SubAgentForecast(**raw2)]
    result2, msgs2 = _agg(members2)
    assert _close(result2, TITLE_BOOL_FALSE), result2
    assert msgs2 == [], msgs2


def test_single_survivor_after_flag_drop_dispersion_zero():
    members = [_f("a1", "fam1", 0.60), _f("a2", "fam2", 0.65, 1.0, "False")]
    result, msgs = _agg(members)
    assert _close(result, (0.6077590814688054, 0.0)), result
    assert any("n=1 of 2" in m for m in msgs), msgs


def test_provenance_backcompat_positional_construction():
    # Pre-SF-BL-10848 positional constructions (8 fields) still work and
    # read n_flag_dropped as 0.
    p = ak.PanelProvenance(3, 3, 0, 0, 0, 0, 3, 3)
    assert p.n_flag_dropped == 0, p
    assert p.branch_flipped is False, p


def test_flag_drop_threads_into_gate2c():
    # SF-BL-10761 integration: a flag-culled panel carries n_dropped>0,
    # so evaluate_and_size_opportunity SKIPs it at Gate 2C when the
    # provenance is threaded through.
    book = ak.OrderBookSnapshot(
        market_id="FIXTURE-10848", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C10848",
        est_days_to_resolution=10,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100000)],
        fee_rate=0.01, on_chain_verified=True)
    members = [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60),
               _f("a3", "fam3", 0.50, 1.0, "False")]
    r, _ = _agg(members)
    used = {}
    out = ak.evaluate_and_size_opportunity(
        book, r[0], r[1], 100000.0, used,
        panel_n_dropped=r.provenance.n_dropped,
        panel_n_kept=r.provenance.n_kept)
    assert out.action == ak.TradeAction.SKIP, out
    assert out.skip_reason.startswith("CULLED_PANEL_GATE2"), out.skip_reason
    assert out.invested_capital == 0.0 and used == {}, (out, used)


# ---- 5. No regression on strict-bool panels (byte-identical) -----------------

def test_strict_bool_panels_unchanged():
    cases = [
        ([_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60), _f("a3", "fam3", 0.65)],
         (0.6989517116705906, 0.04082482904638629)),
        ([_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60), _f("a3", "fam3", 0.50)],
         TITLE_BOOL_TRUE),
        ([_f("a1", "fam1", 0.70, 1.0, True), _f("a2", "fam2", 0.60, 1.0, True),
          _f("a3", "fam3", 0.50, 1.0, False)],
         TITLE_BOOL_FALSE),
        ([_f("a1", "fam1", 0.70), _f("a2", "fam1", 0.60)],
         CLEAN_PAIR),
        ([_f(f"a{i}", "fam1", p) for i, p in enumerate([0.5, 0.6, 0.7, 0.6, 0.5])],
         (0.5719288328804474, 0.0748331477354788)),
        ([_f("a1", "fam1", 0.60)], (0.6077590814688054, 0.0)),
        # Default-flag members (flag omitted -> True) unchanged.
        ([ak.SubAgentForecast("a1", "fam1", 0.70),
          ak.SubAgentForecast("a2", "fam2", 0.60),
          ak.SubAgentForecast("a3", "fam3", 0.65)],
         (0.6989517116705906, 0.04082482904638629)),
    ]
    for members, expected in cases:
        result, msgs = _agg(members)
        assert _close(result, expected), (result, expected)
        assert tuple(result) == result  # 2-tuple contract intact
        assert msgs == [], msgs  # no warning when nothing was dropped
        assert result.provenance.n_flag_dropped == 0, result.provenance
        assert result.provenance.n_dropped == 0, result.provenance


def test_benchmark_fixture_unchanged():
    buf = io.StringIO()
    with redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
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
