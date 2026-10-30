#!/usr/bin/env python3
"""SF-BL-8890 tests: cluster_exposure_used must be released at settlement.

Criterion (TASK.md prereg): on a scripted sequence of same-cluster trades
that settle before the next entry, the alpha_kelly path must free the
settled fraction (cap headroom restored within 1e-9 in current-equity
terms) in at least 99% of cases; kill if headroom after settlement still
equals the never-released baseline on more than 1% of sequences.

Pre-fix behaviour (repro.py): round 1 took the full 0.15 cap, settled,
and all later rounds SKIPped CLUSTER_CAP_REACHED -- 0% restored.
"""
import sys

sys.path.insert(0, "/home/hatch/workspace/superforecaster")
from superforecaster import alpha_kelly as ak  # noqa: E402

CLUSTER = "TEST_CLUSTER"
CAP = 0.15
PASS = []


def check(name, cond, detail=""):
    PASS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f" | {detail}" if detail else ""))
    assert cond, f"{name} {detail}"


def cfg():
    # Isolate the cluster-cap lifecycle from the A/B veto gates.
    return ak.FleetRiskConfig(
        budget_sensitivity_veto_enabled=False, max_round_trip_cost=None)


def book(mid="REPRO", cluster=CLUSTER, ask=0.50, bid=0.48):
    return ak.OrderBookSnapshot(
        market_id=mid, title="t", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster=cluster, est_days_to_resolution=5.0,
        yes_asks=[ak.OrderBookLevel(price=ask, size_shares=1_000_000)],
        yes_bids=[ak.OrderBookLevel(price=bid, size_shares=1_000_000)],
        fee_rate=0.01)


def evaluate(equity, cluster_dict, mid="REPRO", p=0.80, cluster=CLUSTER):
    return ak.evaluate_and_size_opportunity(
        book=book(mid, cluster), calibrated_prob_yes=p,
        inter_agent_dispersion=0.05, portfolio_equity=equity,
        cluster_exposure_used=cluster_dict, config=cfg())


# ---- 1. Prereg sequence: settle-before-next-entry, headroom restored ----

def test_prereg_sequence_headroom_restored_100pct():
    cluster, equity, restored = {}, 10_000.0, 0
    rounds = 100
    for i in range(rounds):
        t = evaluate(equity, cluster, mid=f"S{i}")
        assert t.action != ak.TradeAction.SKIP, (i, t.skip_reason)
        t = ak.settle_or_mark_trade(t, is_settled=True,
                                    outcome_yes=1 if i % 2 == 0 else 0)
        equity += t.net_pnl_bid_marked
        used = cluster.get(CLUSTER, 0.0)
        # current-equity terms: full cap dollars available at next entry
        headroom_dollars = (CAP - used) * equity
        if abs(headroom_dollars - CAP * equity) <= 1e-9 * max(1.0, equity) \
                and abs(used) <= 1e-9:
            restored += 1
    check("prereg_sequence_100_rounds_restored_ge99pct",
          restored / rounds >= 0.99, f"restored={restored}/{rounds}")
    check("prereg_sequence_used_exactly_zero", cluster.get(CLUSTER) == 0.0)


def test_prereg_sequence_via_explicit_dict_argument():
    cluster, equity = {}, 10_000.0
    for i in range(10):
        t = evaluate(equity, cluster, mid=f"E{i}")
        t.cluster_exposure_used = None  # force the explicit-arg path only
        t = ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1,
                                    cluster_exposure_used=cluster)
        equity += t.net_pnl_bid_marked
        assert cluster.get(CLUSTER, 0.0) == 0.0, (i, cluster)
    check("explicit_dict_argument_releases", True)


def test_loss_settlement_also_releases():
    cluster = {}
    t = evaluate(10_000.0, cluster)
    assert cluster[CLUSTER] > 0
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=0)
    check("losing_settlement_releases", cluster[CLUSTER] == 0.0)


def test_buy_no_settlement_releases():
    cluster = {}
    t = ak.evaluate_and_size_opportunity(
        book=book("NO1"), calibrated_prob_yes=0.20,
        inter_agent_dispersion=0.05, portfolio_equity=10_000.0,
        cluster_exposure_used=cluster, config=cfg())
    assert t.action == ak.TradeAction.BUY_NO, t.action
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    check("buy_no_settlement_releases", cluster[CLUSTER] == 0.0)


# ---- 2. What must NOT release -------------------------------------------

def test_open_mark_does_not_release():
    cluster = {}
    t = evaluate(10_000.0, cluster)
    used_before = cluster[CLUSTER]
    ak.settle_or_mark_trade(t, is_settled=False)
    check("open_mark_keeps_exposure", cluster[CLUSTER] == used_before)
    # ...and the later settlement still releases exactly once.
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    check("settle_after_mark_releases", cluster[CLUSTER] == 0.0)


def test_double_settle_releases_once_only():
    cluster = {}
    t1 = evaluate(10_000.0, cluster, mid="A")          # takes full cap
    t2 = evaluate(10_000.0, cluster, mid="B")          # cap reached -> SKIP
    assert t2.action == ak.TradeAction.SKIP
    ak.settle_or_mark_trade(t1, is_settled=True, outcome_yes=1)
    ak.settle_or_mark_trade(t1, is_settled=True, outcome_yes=1)  # re-settle
    check("double_settle_no_negative", cluster[CLUSTER] == 0.0)


def test_concurrent_positions_release_only_their_own_fraction():
    cluster = {}
    # Two smaller concurrent opens in the same cluster (lower conviction
    # sizing so both fit under the cap), at DIFFERENT equity bases.
    t1 = evaluate(10_000.0, cluster, mid="C1", p=0.65)
    f1 = t1.allocated_fraction
    assert 0 < f1 < CAP, f1
    t2 = evaluate(20_000.0, cluster, mid="C2", p=0.65)  # equity base doubled
    f2 = t2.allocated_fraction
    assert f2 > 0 and abs(cluster[CLUSTER] - (f1 + f2)) < 1e-12
    ak.settle_or_mark_trade(t1, is_settled=True, outcome_yes=1)
    check("concurrent_settle_one_leaves_other",
          abs(cluster[CLUSTER] - f2) <= 1e-9, f"used={cluster[CLUSTER]} f2={f2}")
    ak.settle_or_mark_trade(t2, is_settled=True, outcome_yes=0)
    check("concurrent_settle_both_returns_to_zero", cluster[CLUSTER] == 0.0)


def test_cap_still_enforced_while_position_open():
    cluster = {}
    t = evaluate(10_000.0, cluster, mid="OPEN1")
    assert t.action != ak.TradeAction.SKIP
    t2 = evaluate(10_000.0, cluster, mid="OPEN2")
    check("cap_enforced_for_open_position",
          t2.action == ak.TradeAction.SKIP
          and "CLUSTER_CAP_REACHED" in (t2.skip_reason or ""))


def test_skip_trade_releases_nothing():
    cluster = {CLUSTER: 0.05}
    t = ak.evaluate_and_size_opportunity(
        book=book("SK"), calibrated_prob_yes=0.52,
        inter_agent_dispersion=0.05, portfolio_equity=10_000.0,
        cluster_exposure_used=cluster, config=cfg())
    assert t.action == ak.TradeAction.SKIP
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1,
                            cluster_exposure_used=cluster)
    check("skip_settle_leaves_dict_untouched", cluster[CLUSTER] == 0.05)


def test_failed_settlement_releases_nothing():
    cluster = {}
    t = evaluate(10_000.0, cluster)
    used_before = cluster[CLUSTER]
    try:
        ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=None)
        raised = False
    except ValueError:
        raised = True
    check("invalid_outcome_raises_and_keeps_exposure",
          raised and cluster[CLUSTER] == used_before)


def test_manual_trade_without_ledger_settles_unchanged():
    # Legacy surface: directly constructed trade, no dict anywhere --
    # PnL identical to pre-fix, no crash, nothing released.
    t = ak.TradeExecutionResult(
        market_id="M", title="t", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="X", action=ak.TradeAction.BUY_YES,
        calibrated_prob_yes=0.8, market_mid_prob=0.49, exec_price=0.50,
        liquidation_bid_price=0.48, kl_divergence_nats=0.0,
        expected_log_growth=0.0, kelly_fraction_raw=0.1,
        allocated_fraction=0.05, invested_capital=500.0,
        shares_acquired=1000.0, est_days_to_resolution=5.0)
    t = ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    check("manual_trade_pnl_unchanged", t.net_pnl_bid_marked == 500.0)


def test_helper_release_cluster_exposure_direct():
    cluster = {}
    t = evaluate(10_000.0, cluster)
    removed = ak.release_cluster_exposure(t, cluster)
    check("helper_releases_full_fraction",
          removed > 0 and cluster[CLUSTER] == 0.0)
    check("helper_second_call_noop", ak.release_cluster_exposure(t, cluster) == 0.0)


def test_other_clusters_untouched():
    cluster = {}
    t = evaluate(10_000.0, cluster, cluster="OTHER", mid="OC")
    # book() cluster param drives factor_cluster; evaluate used OTHER ledger key
    assert cluster.get("OTHER", 0.0) > 0
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    check("release_scoped_to_trade_cluster",
          cluster.get("OTHER") == 0.0 and CLUSTER not in cluster)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{sum(1 for _, ok in PASS if ok)}/{len(PASS)} checks PASS")
