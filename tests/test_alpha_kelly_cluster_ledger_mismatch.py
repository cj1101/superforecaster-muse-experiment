#!/usr/bin/env python3
"""SF-BL-34868 tests: settlement release must honour the ENTRY ledger.

Criterion (TASK.md prereg): success if a mismatched-ledger settle is
rejected or redirected to the trade's stored entry ledger on fixtures
covering both the legacy-fraction and dollar-registered paths, with
the unrelated ledger byte-identical afterwards; kill if every in-repo
settle caller is shown to pass the entry ledger under a tested
invariant (CALLER_CENSUS.md: NOT met).

The fix REJECTS: settle_or_mark_trade (settled branch, at its head,
before outcome validation or any write) and release_cluster_exposure
(at entry) raise ValueError MISMATCHED_CLUSTER_LEDGER when the trade
stores an entry ledger and the explicit cluster_exposure_used is a
different dict object.

Pre-fix behaviour (repro.py): ledger B holding another position's 0.10
was rewritten to 0.0 on both paths, ledger A kept the settled trade's
0.15 forever, and the trade was flagged released.
"""
import copy
import sys

sys.path.insert(0, "/home/hatch/workspace/superforecaster")
from superforecaster import alpha_kelly as ak  # noqa: E402

CLUSTER = "MM_CLUSTER"
PASS = []


def check(name, cond, detail=""):
    PASS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f" | {detail}" if detail else ""))
    assert cond, f"{name} {detail}"


def cfg():
    return ak.FleetRiskConfig(
        budget_sensitivity_veto_enabled=False, max_round_trip_cost=None)


def book(mid, cluster=CLUSTER):
    return ak.OrderBookSnapshot(
        market_id=mid, title="t", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster=cluster, est_days_to_resolution=5.0,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=1_000_000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=1_000_000)],
        fee_rate=0.01)


def evaluate(equity, ledger, mid, p=0.80, cluster=CLUSTER):
    return ak.evaluate_and_size_opportunity(
        book=book(mid, cluster), calibrated_prob_yes=p,
        inter_agent_dispersion=0.05, portfolio_equity=equity,
        cluster_exposure_used=ledger, config=cfg())


def manual_trade(ledger, frac=0.15, invested=1500.0, shares=3000.0):
    return ak.TradeExecutionResult(
        market_id="M1", title="t", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster=CLUSTER, action=ak.TradeAction.BUY_YES,
        calibrated_prob_yes=0.80, market_mid_prob=0.49, exec_price=0.51,
        liquidation_bid_price=0.48, kl_divergence_nats=0.1,
        expected_log_growth=0.01, kelly_fraction_raw=0.2,
        allocated_fraction=frac, invested_capital=invested,
        shares_acquired=shares, est_days_to_resolution=5.0,
        is_settled=False, cluster_exposure_used=ledger)


def snap(trade):
    return (trade.is_settled, trade.outcome_yes, trade.net_pnl_bid_marked,
            trade.cluster_exposure_released)


def raises_mismatch(fn):
    try:
        fn()
    except ValueError as e:
        return "MISMATCHED_CLUSTER_LEDGER" in str(e) and "SF-BL-34868" in str(e)
    return False


# ---- 1. Legacy-fraction path: mismatched settle rejected ------------------

def test_legacy_mismatched_settle_rejected_ledgers_byte_identical():
    A = {CLUSTER: 0.15, "OTHER": 0.03}
    B = {CLUSTER: 0.10, "OTHER": 0.07}
    A0, B0 = copy.deepcopy(A), copy.deepcopy(B)
    t = manual_trade(A)
    before = snap(t)
    check("legacy_mismatch_raises_named",
          raises_mismatch(lambda: ak.settle_or_mark_trade(
              t, is_settled=True, outcome_yes=1, cluster_exposure_used=B)))
    check("legacy_unrelated_ledger_byte_identical", B == B0 and repr(B) == repr(B0))
    check("legacy_entry_ledger_untouched", A == A0)
    check("legacy_trade_fields_untouched", snap(t) == before
          and t.cluster_exposure_released is False)


def test_legacy_rejection_preserves_correct_release_path():
    A = {CLUSTER: 0.15}
    B = {CLUSTER: 0.10}
    t = manual_trade(A)
    assert raises_mismatch(lambda: ak.settle_or_mark_trade(
        t, is_settled=True, outcome_yes=1, cluster_exposure_used=B))
    # Settle with NO argument releases via the stored reference.
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    check("legacy_correct_settle_releases_entry_ledger", A == {CLUSTER: 0.0})
    check("legacy_correct_settle_leaves_unrelated_ledger", B == {CLUSTER: 0.10})


def test_legacy_direct_release_mismatch_rejected():
    A = {CLUSTER: 0.15}
    B = {CLUSTER: 0.10}
    t = manual_trade(A)
    check("direct_release_mismatch_raises",
          raises_mismatch(lambda: ak.release_cluster_exposure(t, B)))
    check("direct_release_ledgers_untouched",
          A == {CLUSTER: 0.15} and B == {CLUSTER: 0.10}
          and t.cluster_exposure_released is False)
    removed = ak.release_cluster_exposure(t, A)
    check("direct_release_same_ledger_works",
          removed == 0.15 and A == {CLUSTER: 0.0})


def test_legacy_equal_contents_different_object_still_rejected():
    # Identity, not equality: a copy carries the same numbers but is not
    # the dict evaluate charged (and may be the live ledger of another
    # accounting scope that merely looks the same right now).
    A = {CLUSTER: 0.15}
    B = dict(A)
    t = manual_trade(A)
    check("equal_copy_still_mismatch",
          raises_mismatch(lambda: ak.settle_or_mark_trade(
              t, is_settled=True, outcome_yes=1, cluster_exposure_used=B)))
    check("equal_copy_untouched", B == {CLUSTER: 0.15} and A == {CLUSTER: 0.15})


# ---- 2. Dollar-registered path (SF-BL-10762): mismatched settle rejected --

def test_dollar_mismatched_settle_rejected_ledgers_byte_identical():
    A, B = {}, {}
    t1 = evaluate(10_000.0, A, "D1", p=0.80)   # full cap 0.15, dollars registered
    t2 = evaluate(10_000.0, B, "D2", p=0.65)   # partial, dollars registered
    assert A[CLUSTER] == 0.15 and 0.0 < B[CLUSTER] < 0.15, (A, B)
    A0, B0 = copy.deepcopy(A), copy.deepcopy(B)
    before = snap(t1)
    check("dollar_mismatch_raises_named",
          raises_mismatch(lambda: ak.settle_or_mark_trade(
              t1, is_settled=True, outcome_yes=1, cluster_exposure_used=B)))
    check("dollar_unrelated_ledger_byte_identical", B == B0 and repr(B) == repr(B0))
    check("dollar_entry_ledger_untouched", A == A0)
    check("dollar_trade_fields_untouched", snap(t1) == before)
    # Registry integrity: both trades still release their OWN ledgers
    # dollar-exactly after the rejected attempt.
    ak.settle_or_mark_trade(t1, is_settled=True, outcome_yes=1)
    ak.settle_or_mark_trade(t2, is_settled=True, outcome_yes=0)
    check("dollar_correct_settles_release_to_zero",
          A == {CLUSTER: 0.0} and B == {CLUSTER: 0.0})


def test_dollar_direct_release_mismatch_rejected():
    A, B = {}, {}
    t1 = evaluate(10_000.0, A, "R1", p=0.80)
    evaluate(10_000.0, B, "R2", p=0.65)
    B0 = copy.deepcopy(B)
    check("dollar_direct_release_mismatch_raises",
          raises_mismatch(lambda: ak.release_cluster_exposure(t1, B)))
    check("dollar_direct_release_unrelated_untouched", B == B0)
    ak.release_cluster_exposure(t1, A)
    check("dollar_direct_release_own_ledger_works", A == {CLUSTER: 0.0})


# ---- 3. Preserved surfaces (byte-identical behaviour) ---------------------

def test_explicit_same_object_releases_benchmark_pattern():
    A = {}
    t = evaluate(10_000.0, A, "S1")
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1,
                            cluster_exposure_used=A)
    check("explicit_same_object_releases", A == {CLUSTER: 0.0}
          and t.cluster_exposure_released is True)


def test_stored_none_explicit_dict_releases_8890_pattern():
    A = {CLUSTER: 0.15}
    t = manual_trade(None)
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1,
                            cluster_exposure_used=A)
    check("stored_none_explicit_releases", A == {CLUSTER: 0.0})


def test_no_ledger_anywhere_settles_unchanged():
    t = manual_trade(None)
    t = ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    check("no_ledger_settle_pnl_unchanged",
          t.net_pnl_bid_marked == 3000.0 - 1500.0 and t.is_settled is True)


def test_open_mark_with_mismatched_arg_unchanged():
    # Open marks never release and never read the argument.
    A = {CLUSTER: 0.15}
    B = {CLUSTER: 0.10}
    t = manual_trade(A)
    t = ak.settle_or_mark_trade(t, is_settled=False, cluster_exposure_used=B)
    check("open_mark_no_raise_no_release",
          A == {CLUSTER: 0.15} and B == {CLUSTER: 0.10}
          and t.cluster_exposure_released is False
          and abs(t.net_pnl_bid_marked - (3000.0 * 0.48 - 1500.0)) < 1e-9)


def test_mismatch_check_precedes_outcome_validation():
    # A mismatched settle raises the ledger error even when the outcome
    # is ALSO invalid — the identity precondition runs at the head of
    # the settled branch, and nothing is mutated either way.
    A = {CLUSTER: 0.15}
    B = {CLUSTER: 0.10}
    t = manual_trade(A)
    before = snap(t)
    check("mismatch_beats_bad_outcome",
          raises_mismatch(lambda: ak.settle_or_mark_trade(
              t, is_settled=True, outcome_yes=None, cluster_exposure_used=B)))
    check("mismatch_bad_outcome_untouched",
          snap(t) == before and A == {CLUSTER: 0.15} and B == {CLUSTER: 0.10})


def test_released_trade_direct_release_mismatch_still_rejected():
    # The identity check runs before the idempotency guard: naming the
    # wrong ledger is a caller error even when no release remains.
    A = {CLUSTER: 0.15}
    B = {CLUSTER: 0.10}
    t = manual_trade(A)
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    assert A == {CLUSTER: 0.0} and t.cluster_exposure_released is True
    check("released_trade_mismatch_still_raises",
          raises_mismatch(lambda: ak.release_cluster_exposure(t, B)))
    check("released_trade_own_ledger_idempotent_noop",
          ak.release_cluster_exposure(t, A) == 0.0 and B == {CLUSTER: 0.10})


def test_double_settle_same_ledger_releases_once():
    A = {}
    t = evaluate(10_000.0, A, "DS1")
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1,
                            cluster_exposure_used=A)
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1,
                            cluster_exposure_used=A)
    check("double_settle_same_ledger_once", A == {CLUSTER: 0.0})


def test_failed_settlement_bad_outcome_same_ledger_releases_nothing():
    A = {}
    t = evaluate(10_000.0, A, "F1")
    used_before = A[CLUSTER]
    try:
        ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=None,
                                cluster_exposure_used=A)
        raised = False
    except ValueError:
        raised = True
    check("bad_outcome_same_ledger_keeps_exposure",
          raised and A[CLUSTER] == used_before
          and t.cluster_exposure_released is False)


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
    print(f"\n{sum(1 for _, ok in PASS if ok)}/{len(PASS)} checks PASS")
