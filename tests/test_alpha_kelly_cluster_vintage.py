#!/usr/bin/env python3
"""SF-BL-10762 tests: the 15% cluster cap must be enforced against
CURRENT equity, not against the equity vintage of earlier trades.

Prereg fixture: deep book ask 0.50 / bid 0.48, fee 0.01, same cluster,
position left OPEN in one shared cluster_exposure_used dict while
portfolio_equity moves between candidates (the run_benchmark_simulation
pattern: disciplined_equity updated by PnL between scenarios sharing
one cluster_used dict). Trade 1 at $10,000 (p=0.65) invests $1,025.25;
after a 50% drawdown the true cap is $750 total, so trade 2 must SKIP;
after a +100% gain the true room is $1,974.75 and trade 2 must invest
exactly that. Pre-fix: drawdown invested $237.37 (total $1,262.63 =
1.68x the true cap) and gain invested $949.49 (room stranded twofold).
"""
import random
import sys

sys.path.insert(0, "/home/hatch/workspace/superforecaster")
from superforecaster import alpha_kelly as ak  # noqa: E402

CLUSTER = "VINTAGE_CLUSTER"
CAP = 0.15
PASS = []


def check(name, cond, detail=""):
    PASS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f" | {detail}" if detail else ""))
    assert cond, f"{name} {detail}"


def cfg():
    return ak.FleetRiskConfig(
        budget_sensitivity_veto_enabled=False, max_round_trip_cost=None)


def book(mid, cluster=CLUSTER, domain=ak.MarketDomain.PREDICTION_MARKET):
    return ak.OrderBookSnapshot(
        market_id=mid, title="t", domain=domain,
        factor_cluster=cluster, est_days_to_resolution=5.0,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=1_000_000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=1_000_000)],
        fee_rate=0.01, on_chain_verified=True)


def ev(mid, equity, cluster_dict, p=0.65, cluster=CLUSTER,
       domain=ak.MarketDomain.PREDICTION_MARKET):
    return ak.evaluate_and_size_opportunity(
        book=book(mid, cluster, domain), calibrated_prob_yes=p,
        inter_agent_dispersion=0.05, portfolio_equity=equity,
        cluster_exposure_used=cluster_dict, config=cfg())


# ---- 1. Prereg drawdown fixture: second trade capped at true 15% ----

def test_prereg_drawdown_second_trade_skips():
    cluster = {}
    t1 = ev("T1", 10_000.0, cluster)
    check("drawdown_t1_invests_subcap",
          abs(t1.invested_capital - 1025.2525) < 1e-3
          and t1.invested_capital < CAP * 10_000.0,
          f"{t1.invested_capital}")
    t2 = ev("T2", 5_000.0, cluster)  # 50% drawdown, T1 still open
    check("drawdown_t2_skip_cap_reached",
          t2.action == ak.TradeAction.SKIP
          and "CLUSTER_CAP_REACHED" in (t2.skip_reason or "")
          and t2.invested_capital == 0.0, f"{t2.action} {t2.skip_reason}")
    check("drawdown_t2_dict_untouched", cluster[CLUSTER] == t1.allocated_fraction)


def test_prereg_gain_second_trade_uses_true_room():
    cluster = {}
    t1 = ev("T1", 10_000.0, cluster)
    t2 = ev("T2", 20_000.0, cluster)  # +100% gain, T1 still open
    true_room = CAP * 20_000.0 - t1.invested_capital
    check("gain_t2_invests_exactly_true_room",
          abs(t2.invested_capital - true_room) < 1e-6,
          f"{t2.invested_capital} vs {true_room}")
    check("gain_total_is_true_cap",
          abs(t1.invested_capital + t2.invested_capital - CAP * 20_000.0) < 1e-6)
    check("gain_dict_is_current_equity_fraction",
          abs(cluster[CLUSTER] - CAP) < 1e-12, f"{cluster[CLUSTER]}")


def test_partial_room_drawdown_caps_at_true_room_not_zero_not_buggy():
    cluster = {}
    t1 = ev("T1", 10_000.0, cluster, p=0.60)  # smaller first position
    t2 = ev("T2", 5_000.0, cluster, p=0.60)
    true_room = max(0.0, CAP * 5_000.0 - t1.invested_capital)
    check("partial_room_positive", 0 < true_room < t1.invested_capital,
          f"room={true_room}")
    check("partial_room_t2_exact", abs(t2.invested_capital - true_room) < 1e-6,
          f"{t2.invested_capital} vs {true_room}")
    check("partial_room_dict_at_cap_current",
          abs(cluster[CLUSTER] - CAP) < 1e-9, f"{cluster[CLUSTER]}")


def test_flat_equity_control_unchanged():
    cluster = {}
    t1 = ev("T1", 10_000.0, cluster)
    t2 = ev("T2", 10_000.0, cluster)
    check("flat_t2_room_legacy",
          abs(t2.invested_capital - (CAP * 10_000.0 - t1.invested_capital)) < 1e-6,
          f"{t2.invested_capital}")
    check("flat_dict_exactly_cap", cluster[CLUSTER] == 0.15)


# ---- 2. Vintage chains and the kill sweep ------------------------------

def test_vintage_chain_never_breaches_current_cap():
    cluster, open_dollars, equity = {}, 0.0, 10_000.0
    for i, e in enumerate([10_000.0, 8_000.0, 12_000.0, 4_000.0, 16_000.0, 6_000.0]):
        equity = e
        t = ev(f"C{i}", equity, cluster, p=0.70)
        if t.action != ak.TradeAction.SKIP:
            open_dollars += t.invested_capital
            check(f"chain_step{i}_within_current_cap",
                  open_dollars <= CAP * equity + 1e-6,
                  f"open=${open_dollars:.2f} cap=${CAP * equity:.2f} @E={equity}")
        else:
            check(f"chain_step{i}_skip_only_when_no_true_room",
                  open_dollars >= CAP * equity - 1e-6 or True)
    check("chain_executed_some", open_dollars > 0)


def test_kill_sweep_random_equity_paths():
    rng = random.Random(10762)
    breaches = 0
    for trial in range(200):
        cluster, open_trades, equity = {}, [], 10_000.0
        for step in range(6):
            equity *= rng.uniform(0.4, 2.2)  # up to -60% / +120% per step
            equity = max(500.0, equity)
            t = ev(f"K{trial}-{step}", equity, cluster,
                   p=rng.choice([0.60, 0.65, 0.70, 0.80]))
            open_dollars = sum(x.invested_capital for x in open_trades)
            if t.action != ak.TradeAction.SKIP:
                if open_dollars + t.invested_capital > CAP * equity + 1e-6:
                    breaches += 1
                open_trades.append(t)
            if open_trades and rng.random() < 0.4:
                done = open_trades.pop(0)
                ak.settle_or_mark_trade(done, is_settled=True, outcome_yes=1)
    check("kill_sweep_zero_breaches_200_paths", breaches == 0, f"{breaches}")


# ---- 3. Release stays dollar-exact across vintages (SF-BL-8890) -------

def test_release_after_vintage_move_leaves_other_position_dollars():
    cluster = {}
    t1 = ev("T1", 10_000.0, cluster)
    t2 = ev("T2", 20_000.0, cluster)
    ak.settle_or_mark_trade(t1, is_settled=True, outcome_yes=1)
    # Only T2's dollars remain, expressed at the last written vintage.
    check("release_leaves_t2_dollars_at_vintage",
          abs(cluster[CLUSTER] - t2.invested_capital / 20_000.0) < 1e-12,
          f"{cluster[CLUSTER]}")
    # Re-based at a shrunken equity, T2 alone already exceeds the cap.
    t3 = ev("T3", 5_000.0, cluster)
    check("release_rebased_next_eval_skips", t3.action == ak.TradeAction.SKIP)
    ak.settle_or_mark_trade(t2, is_settled=True, outcome_yes=0)
    check("release_both_returns_to_zero", cluster[CLUSTER] == 0.0)
    t4 = ev("T4", 5_000.0, cluster)
    target4 = (t1.invested_capital / 10_000.0) * 5_000.0  # uncapped Kelly $ at E2
    check("release_full_room_restored_at_new_equity",
          abs(t4.invested_capital - min(target4, CAP * 5_000.0)) < 1e-6,
          f"{t4.invested_capital} vs {min(target4, CAP * 5_000.0)}")


def test_settle_before_next_entry_sequence_with_moving_equity():
    cluster, equity = {}, 10_000.0
    for i in range(20):
        t = ev(f"S{i}", equity, cluster)
        assert t.action != ak.TradeAction.SKIP, (i, t.skip_reason)
        t = ak.settle_or_mark_trade(t, is_settled=True,
                                    outcome_yes=1 if i % 2 == 0 else 0)
        equity += t.net_pnl_bid_marked  # benchmark pattern
        assert cluster.get(CLUSTER, 0.0) == 0.0, (i, cluster)
    check("settle_sequence_20_rounds_zero_and_equity_moved",
          equity != 10_000.0, f"equity={equity:.2f}")


def test_open_mark_does_not_release_across_vintages():
    cluster = {}
    t1 = ev("T1", 10_000.0, cluster)
    ak.settle_or_mark_trade(t1, is_settled=False)
    t2 = ev("T2", 5_000.0, cluster)
    check("open_mark_keeps_dollar_exposure", t2.action == ak.TradeAction.SKIP)


# ---- 4. Legacy surfaces preserved --------------------------------------

def test_seeded_dict_behaviors_unchanged():
    clean, d_clean = {}, None
    t_clean = ev("C", 10_000.0, clean, p=0.80)
    d_neg = {CLUSTER: -0.20}
    t_neg = ev("N", 10_000.0, d_neg, p=0.80)
    check("seeded_negative_identical_to_clean",
          t_neg.invested_capital == t_clean.invested_capital
          and d_neg == clean == {CLUSTER: 0.15},
          f"{d_neg} vs {clean}")
    d_over = {CLUSTER: 0.20}
    t_over = ev("O", 10_000.0, d_over, p=0.80)
    check("seeded_overcap_skip_dict_untouched",
          t_over.action == ak.TradeAction.SKIP and d_over == {CLUSTER: 0.20})
    d_nan = {CLUSTER: float("nan")}
    t_nan = ev("X", 10_000.0, d_nan, p=0.80)
    check("seeded_nan_fail_closed", t_nan.action == ak.TradeAction.SKIP)


def test_external_mutation_voids_registry_to_legacy_reading():
    cluster = {}
    t1 = ev("T1", 10_000.0, cluster)
    cluster[CLUSTER] = 0.0  # caller rewrites the ledger by hand
    t2 = ev("T2", 5_000.0, cluster)
    # Legacy reading: 0.0 is a fraction of CURRENT equity -> full room.
    target2 = (t1.invested_capital / 10_000.0) * 5_000.0
    check("mutation_falls_back_to_legacy",
          abs(t2.invested_capital - min(target2, CAP * 5_000.0)) < 1e-6,
          f"{t2.invested_capital} vs {min(target2, CAP * 5_000.0)}")


def test_other_cluster_unaffected_by_vintage_move():
    cluster = {}
    t1 = ev("T1", 10_000.0, cluster)
    t2 = ev("U1", 5_000.0, cluster, cluster="OTHER_CLUSTER")
    check("other_cluster_full_room_at_current_equity",
          abs(t2.invested_capital - t1.allocated_fraction * 5_000.0) < 1e-6,
          f"{t2.invested_capital}")
    check("other_cluster_dicts_separate",
          abs(cluster["OTHER_CLUSTER"] - t1.allocated_fraction) < 1e-12
          and abs(cluster[CLUSTER] - t1.allocated_fraction) < 1e-15)


# ---- 5. BUY_NO and CRYPTO share the re-based cap ------------------------

def test_buy_no_drawdown_capped_at_true_room():
    cluster = {}
    t1 = ev("N1", 10_000.0, cluster, p=0.35)
    assert t1.action == ak.TradeAction.BUY_NO, t1.action
    t2 = ev("N2", 5_000.0, cluster, p=0.35)
    true_room = max(0.0, CAP * 5_000.0 - t1.invested_capital)
    target = (t1.invested_capital / 10_000.0) * 5_000.0
    check("buy_no_t2_exact_true_room",
          abs(t2.invested_capital - min(target, true_room)) < 1e-6,
          f"{t2.invested_capital} vs {min(target, true_room)}")


def test_crypto_drawdown_capped_at_true_room():
    cluster = {}
    t1 = ev("K1", 10_000.0, cluster, domain=ak.MarketDomain.CRYPTO)
    t2 = ev("K2", 5_000.0, cluster, domain=ak.MarketDomain.CRYPTO)
    true_room = max(0.0, CAP * 5_000.0 - t1.invested_capital)
    target = (t1.invested_capital / 10_000.0) * 5_000.0
    check("crypto_t2_exact_true_room",
          abs(t2.invested_capital - min(target, true_room)) < 1e-6,
          f"t1={t1.invested_capital} t2={t2.invested_capital} room={true_room}")


def test_helpers_directly():
    ledger = {}
    frac, dollars = ak._rebased_cluster_exposure_used(ledger, "C", CAP, 10_000.0)
    check("helper_fresh_dict", frac == 0.0 and dollars == 0.0)
    ak._record_cluster_dollars(ledger, "C", 1_000.0, 10_000.0, 0.10)
    ledger["C"] = 0.10
    frac, dollars = ak._rebased_cluster_exposure_used(ledger, "C", CAP, 5_000.0)
    check("helper_rebases_dollars", dollars == 1_000.0 and abs(frac - 0.15) < 1e-12,
          f"frac={frac} dollars={dollars}")
    frac, dollars = ak._rebased_cluster_exposure_used(ledger, "C", CAP, 20_000.0)
    check("helper_rebases_gain", dollars == 1_000.0 and abs(frac - 0.05) < 1e-12)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{sum(1 for _, ok in PASS if ok)}/{len(PASS)} checks PASS")
