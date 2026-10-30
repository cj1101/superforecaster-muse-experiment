#!/usr/bin/env python3
"""SF-BL-35212 tests: the cluster ledger STORED on the trade must be
validated by settle_or_mark_trade / release_cluster_exposure.

Criterion (TASK.md prereg): success if settle/release rejects a
non-dict stored cluster reference with a named error BEFORE any PnL
is booked, and a stored reference that is not the dict evaluate
charged is flagged instead of silently released into, fixtured at
the list and substituted-dict cases, while genuine evaluate-to-settle
lifecycles stay bit-identical; kill if a caller census shows the
stored reference is never reassigned after evaluate under a tested
invariant (CALLER_CENSUS.md: NOT met).

Fixture (the TASK's numbers): deep book ask 0.50 / bid 0.48, fee
0.01, p=0.60, equity $10,000, cluster 'C1' -- evaluate charges the
entry ledger {'C1': 0.0672}; a win books PnL +$658.42.

Pre-fix behaviour (repro.py): (a) stored=list -> settle booked the
+$658.42 win, then raised a bare AttributeError, entry ledger
stranded at 0.0672, released flag False; (b) stored=fresh dict ->
settle silently released into the substitute ({'C1': 0.0}) and
consumed the flag while the entry ledger kept 0.0672 forever.
"""
import copy
import sys

sys.path.insert(0, "/home/hatch/workspace/superforecaster")
from superforecaster import alpha_kelly as ak  # noqa: E402

CLUSTER = "C1"
PASS = []


def check(name, cond, detail=""):
    PASS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f" | {detail}" if detail else ""))
    assert cond, f"{name} {detail}"


def cfg():
    return ak.FleetRiskConfig(
        budget_sensitivity_veto_enabled=False, max_round_trip_cost=None)


def book(mid="M1", cluster=CLUSTER):
    return ak.OrderBookSnapshot(
        market_id=mid, title="t", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster=cluster, est_days_to_resolution=5.0,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=1_000_000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=1_000_000)],
        fee_rate=0.01)


def evaluate(ledger, mid="M1", p=0.60):
    return ak.evaluate_and_size_opportunity(
        book=book(mid), calibrated_prob_yes=p,
        inter_agent_dispersion=0.05, portfolio_equity=10_000.0,
        cluster_exposure_used=ledger, config=cfg())


def manual_trade(ledger, frac=0.05, invested=500.0, shares=1000.0):
    return ak.TradeExecutionResult(
        market_id="M1", title="t", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster=CLUSTER, action=ak.TradeAction.BUY_YES,
        calibrated_prob_yes=0.80, market_mid_prob=0.49, exec_price=0.50,
        liquidation_bid_price=0.48, kl_divergence_nats=0.1,
        expected_log_growth=0.01, kelly_fraction_raw=0.2,
        allocated_fraction=frac, invested_capital=invested,
        shares_acquired=shares, est_days_to_resolution=5.0,
        is_settled=False, cluster_exposure_used=ledger)


def snap(t):
    return (t.is_settled, t.outcome_yes, t.net_pnl_bid_marked,
            t.cluster_exposure_released)


def raises_named(fn, needle):
    try:
        fn()
    except ValueError as e:
        return needle in str(e) and "SF-BL-35212" in str(e)
    except Exception:
        return False
    return False


# ---- 1. Non-dict stored reference: settle rejects BEFORE booking ---------

def test_list_stored_settle_rejected_before_booking():
    entry = {}
    t = evaluate(entry)
    entry0, before = copy.deepcopy(entry), snap(t)
    t.cluster_exposure_used = ["not-a-dict"]
    check("list_settle_raises_invalid_stored",
          raises_named(lambda: ak.settle_or_mark_trade(
              t, is_settled=True, outcome_yes=1),
              "INVALID_STORED_CLUSTER_LEDGER"))
    check("list_settle_trade_byte_identical", snap(t) == before)
    check("list_settle_entry_ledger_untouched", entry == entry0)
    # Restoring the entry reference reopens the correct path.
    t.cluster_exposure_used = entry
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    check("list_rejection_preserves_genuine_settle",
          entry == {CLUSTER: 0.0} and abs(t.net_pnl_bid_marked - 658.42) < 0.01)


def test_list_stored_direct_release_rejected():
    entry = {}
    t = evaluate(entry)
    t.cluster_exposure_used = ["x"]
    check("list_release_raises_invalid_stored",
          raises_named(lambda: ak.release_cluster_exposure(t),
                       "INVALID_STORED_CLUSTER_LEDGER"))
    check("list_release_nothing_touched",
          entry != {CLUSTER: 0.0} and t.cluster_exposure_released is False)


def test_other_nondict_stored_types_rejected():
    for bad in ("ledger", 123, ("C1",), 0.5):
        entry = {}
        t = evaluate(entry)
        t.cluster_exposure_used = bad
        ok = raises_named(lambda: ak.settle_or_mark_trade(
            t, is_settled=True, outcome_yes=1), "INVALID_STORED_CLUSTER_LEDGER")
        check(f"nondict_{type(bad).__name__}_rejected", ok)


def test_nondict_stored_at_construction_rejected_at_settle():
    t = manual_trade(["not-a-dict"])
    check("construction_list_raises_at_settle",
          raises_named(lambda: ak.settle_or_mark_trade(
              t, is_settled=True, outcome_yes=1),
              "INVALID_STORED_CLUSTER_LEDGER"))
    check("construction_list_pnl_untouched", t.net_pnl_bid_marked == 0.0)


def test_nondict_stored_via_bypass_write_rejected():
    entry = {}
    t = evaluate(entry)
    object.__setattr__(t, "cluster_exposure_used", ["bypass"])
    check("bypass_list_raises_at_settle",
          raises_named(lambda: ak.settle_or_mark_trade(
              t, is_settled=True, outcome_yes=1),
              "INVALID_STORED_CLUSTER_LEDGER"))


# ---- 2. Substituted dict: flagged, never silently released into ----------

def test_substituted_dict_settle_flagged():
    entry = {}
    t = evaluate(entry)
    entry0, before = copy.deepcopy(entry), snap(t)
    substitute = {CLUSTER: 0.10, "OTHER": 0.03}  # a co-tenant's charge
    sub0 = copy.deepcopy(substitute)
    t.cluster_exposure_used = substitute
    check("subst_settle_raises_substituted",
          raises_named(lambda: ak.settle_or_mark_trade(
              t, is_settled=True, outcome_yes=1), "SUBSTITUTED_CLUSTER_LEDGER"))
    check("subst_settle_trade_byte_identical", snap(t) == before)
    check("subst_settle_both_ledgers_untouched",
          entry == entry0 and substitute == sub0)
    t.cluster_exposure_used = entry
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    check("subst_rejection_preserves_genuine_settle",
          entry == {CLUSTER: 0.0} and substitute == sub0)


def test_substituted_dict_direct_release_flagged():
    entry = {}
    t = evaluate(entry)
    t.cluster_exposure_used = {}
    check("subst_release_raises_substituted",
          raises_named(lambda: ak.release_cluster_exposure(t),
                       "SUBSTITUTED_CLUSTER_LEDGER"))
    check("subst_release_flag_not_consumed",
          t.cluster_exposure_released is False)


def test_substituted_dict_dollar_registered_path():
    # p=0.80 takes the full 0.15 cap through the SF-BL-10762 dollar
    # registry; substitution must be flagged there too.
    entry = {}
    t = evaluate(entry, mid="D1", p=0.80)
    assert entry[CLUSTER] > 0.14, entry
    t.cluster_exposure_used = {}
    check("dollar_subst_raises",
          raises_named(lambda: ak.settle_or_mark_trade(
              t, is_settled=True, outcome_yes=1), "SUBSTITUTED_CLUSTER_LEDGER"))
    t.cluster_exposure_used = entry
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    check("dollar_genuine_settle_releases", entry == {CLUSTER: 0.0})


def test_substituted_with_matching_explicit_still_flagged():
    # SF-BL-34868 blind spot: explicit IS the (substituted) stored
    # reference, so its identity check sees no mismatch.
    entry = {}
    t = evaluate(entry)
    substitute = {}
    t.cluster_exposure_used = substitute
    check("subst_plus_matching_explicit_raises",
          raises_named(lambda: ak.settle_or_mark_trade(
              t, is_settled=True, outcome_yes=1,
              cluster_exposure_used=substitute), "SUBSTITUTED_CLUSTER_LEDGER"))
    check("subst_plus_matching_explicit_untouched",
          substitute == {} and entry != {CLUSTER: 0.0})


def test_cleared_stored_with_foreign_explicit_flagged():
    # SF-BL-34868 blind spot: stored cleared to None, foreign explicit.
    entry, foreign = {}, {CLUSTER: 0.10}
    t = evaluate(entry)
    t.cluster_exposure_used = None
    check("cleared_plus_foreign_explicit_raises",
          raises_named(lambda: ak.settle_or_mark_trade(
              t, is_settled=True, outcome_yes=1,
              cluster_exposure_used=foreign), "SUBSTITUTED_CLUSTER_LEDGER"))
    check("cleared_plus_foreign_untouched", foreign == {CLUSTER: 0.10})
    # ...while the entry ledger itself, passed explicitly, releases.
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1,
                            cluster_exposure_used=entry)
    check("cleared_plus_entry_explicit_releases", entry == {CLUSTER: 0.0})


# ---- 3. Ordering and precedence -------------------------------------------

def test_stored_validation_precedes_outcome_validation():
    entry = {}
    t = evaluate(entry)
    t.cluster_exposure_used = ["x"]
    check("list_beats_bad_outcome",
          raises_named(lambda: ak.settle_or_mark_trade(
              t, is_settled=True, outcome_yes=None),
              "INVALID_STORED_CLUSTER_LEDGER"))


def test_34868_precedence_preserved_for_explicit_mismatch():
    entry, foreign = {}, {CLUSTER: 0.10}
    t = evaluate(entry)
    try:
        ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1,
                                cluster_exposure_used=foreign)
        msg = ""
    except ValueError as e:
        msg = str(e)
    check("explicit_mismatch_still_34868",
          "MISMATCHED_CLUSTER_LEDGER" in msg and "SF-BL-34868" in msg, msg[:60])


def test_validation_precedes_release_idempotency():
    entry = {}
    t = evaluate(entry)
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    assert t.cluster_exposure_released is True
    t.cluster_exposure_used = ["x"]
    check("released_trade_list_stored_still_raises",
          raises_named(lambda: ak.release_cluster_exposure(t),
                       "INVALID_STORED_CLUSTER_LEDGER"))


# ---- 4. Preserved surfaces: genuine lifecycles bit-identical --------------

def test_genuine_evaluate_settle_bit_identical():
    entry = {}
    t = evaluate(entry)
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    check("genuine_noarg_releases_and_books",
          entry == {CLUSTER: 0.0} and abs(t.net_pnl_bid_marked - 658.4158) < 1e-3
          and t.cluster_exposure_released is True)
    entry2 = {}
    t2 = evaluate(entry2, mid="G2")
    ak.settle_or_mark_trade(t2, is_settled=True, outcome_yes=0,
                            cluster_exposure_used=entry2)
    check("genuine_explicit_entry_releases",
          entry2 == {CLUSTER: 0.0} and t2.net_pnl_bid_marked < 0)


def test_explicit_arg_path_with_stored_none_no_snapshot():
    # SF-BL-8890 surface: trade constructed with no ledger, settled
    # purely via the explicit argument.
    cluster = {CLUSTER: 0.05}
    t = manual_trade(None)
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1,
                            cluster_exposure_used=cluster)
    check("explicit_only_path_releases", cluster == {CLUSTER: 0.0})


def test_manual_trade_construction_ledger_releases():
    ledger = {CLUSTER: 0.05}
    t = manual_trade(ledger)
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    check("manual_construction_ledger_releases",
          ledger == {CLUSTER: 0.0} and t.net_pnl_bid_marked == 500.0)


def test_post_construction_adoption_releases():
    # settle_atomicity surface: constructed without a ledger, the
    # ledger adopted by assignment afterwards (no entry snapshot).
    ledger = {CLUSTER: 0.05}
    t = manual_trade(None)
    t.cluster_exposure_used = ledger
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    check("adopted_ledger_releases", ledger == {CLUSTER: 0.0})


def test_manual_trade_without_ledger_unchanged():
    t = manual_trade(None)
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    check("no_ledger_pnl_unchanged", t.net_pnl_bid_marked == 500.0)


def test_open_mark_never_reads_stored_reference():
    entry = {}
    t = evaluate(entry)
    used_before = entry[CLUSTER]
    t.cluster_exposure_used = ["x"]
    r = ak.settle_or_mark_trade(t, is_settled=False)
    check("open_mark_with_list_stored_marks_normally",
          r.is_settled is False and entry[CLUSTER] == used_before
          and t.cluster_exposure_released is False)


def test_rebuild_via_dict_preserves_snapshot():
    entry = {}
    t = evaluate(entry)
    rebuilt = ak.TradeExecutionResult(**t.__dict__)
    ak.settle_or_mark_trade(rebuilt, is_settled=True, outcome_yes=1)
    check("rebuilt_genuine_releases", entry == {CLUSTER: 0.0})
    entry2 = {}
    t2 = evaluate(entry2, mid="R2")
    rebuilt2 = ak.TradeExecutionResult(**t2.__dict__)
    rebuilt2.cluster_exposure_used = {}
    check("rebuilt_substituted_raises",
          raises_named(lambda: ak.settle_or_mark_trade(
              rebuilt2, is_settled=True, outcome_yes=1),
              "SUBSTITUTED_CLUSTER_LEDGER"))


def test_entry_snapshot_recorded():
    entry = {}
    t = evaluate(entry)
    check("snapshot_is_entry_dict", t.entry_cluster_ledger is entry)
    m = manual_trade(entry)
    check("manual_snapshot_is_construction_dict",
          m.entry_cluster_ledger is entry)
    n = manual_trade(None)
    check("no_ledger_no_snapshot", n.entry_cluster_ledger is None)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{sum(1 for _, ok in PASS if ok)}/{len(PASS)} checks PASS")
