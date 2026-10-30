#!/usr/bin/env python3
"""SF-BL-34952 tests: renaming factor_cluster between evaluate and
settle must not leak the cluster release.

Prereg: success if settlement releases against the cluster key
recorded at entry (or a renamed key is rejected/flagged) on 100% of
rename fixtures while unrenamed lifecycles stay bit-identical; kill
if a caller census shows no fleet path ever assigns factor_cluster
on an evaluated trade before settlement (CALLER_CENSUS.md: NOT met).

Pre-fix behaviour (repro.py): evaluate charged 'alpha' 0.15 at
$100,000; rename to 'beta'; settle left {'alpha': 0.15, 'beta': 0.0}
with cluster_exposure_released=True — the entry cap room stranded
forever, on both the dollar-registered (SF-BL-10762) and the legacy
fraction (SF-BL-8890) paths.
"""
import sys

sys.path.insert(0, "/home/hatch/workspace/superforecaster")
from superforecaster import alpha_kelly as ak  # noqa: E402

PASS = []


def check(name, cond, detail=""):
    PASS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f" | {detail}" if detail else ""))
    assert cond, f"{name} {detail}"


def cfg():
    return ak.FleetRiskConfig(
        budget_sensitivity_veto_enabled=False, max_round_trip_cost=None)


def book(mid, cluster):
    return ak.OrderBookSnapshot(
        market_id=mid, title="t", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster=cluster, est_days_to_resolution=5.0,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=1_000_000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=1_000_000)],
        fee_rate=0.01)


def evaluate(equity, ledger, mid, cluster, p=0.80):
    return ak.evaluate_and_size_opportunity(
        book=book(mid, cluster), calibrated_prob_yes=p,
        inter_agent_dispersion=0.05, portfolio_equity=equity,
        cluster_exposure_used=ledger, config=cfg())


def manual_trade(ledger, cluster="alpha", frac=0.15, invested=1500.0):
    return ak.TradeExecutionResult(
        market_id="M1", title="t", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster=cluster, action=ak.TradeAction.BUY_YES,
        calibrated_prob_yes=0.80, market_mid_prob=0.49, exec_price=0.51,
        liquidation_bid_price=0.48, kl_divergence_nats=0.1,
        expected_log_growth=0.01, kelly_fraction_raw=0.2,
        allocated_fraction=frac, invested_capital=invested,
        shares_acquired=3000.0, est_days_to_resolution=5.0,
        is_settled=False, cluster_exposure_used=ledger)


# ---- 1. Prereg fixtures: rename between evaluate and settle ---------------

def test_task_fixture_rename_settle_dollar_path():
    ledger = {}
    t = evaluate(100_000.0, ledger, "E1", "alpha")
    assert ledger == {"alpha": 0.15}, ledger  # full cap (TASK fixture)
    t.factor_cluster = "beta"
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    check("task_fixture_entry_key_released", ledger.get("alpha") == 0.0,
          f"ledger={ledger}")
    check("task_fixture_no_stray_key", "beta" not in ledger, f"ledger={ledger}")
    check("task_fixture_flag_set", t.cluster_exposure_released is True)


def test_rename_via_object_setattr_bypass():
    ledger = {}
    t = evaluate(100_000.0, ledger, "E2", "alpha")
    object.__setattr__(t, "factor_cluster", "beta")
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    check("object_setattr_rename_releases_entry",
          ledger == {"alpha": 0.0}, f"ledger={ledger}")


def test_rename_via_dict_write_legacy():
    ledger = {"alpha": 0.15}
    t = manual_trade(ledger)
    t.__dict__["factor_cluster"] = "beta"
    removed = ak.release_cluster_exposure(t)
    check("dict_write_rename_releases_entry",
          ledger == {"alpha": 0.0} and removed == 0.15,
          f"ledger={ledger} removed={removed}")


def test_legacy_direct_release_after_rename():
    ledger = {"alpha": 0.15}
    t = manual_trade(ledger)
    t.factor_cluster = "beta"
    removed = ak.release_cluster_exposure(t)
    check("legacy_rename_removed_full", removed == 0.15, f"removed={removed}")
    check("legacy_rename_ledger_drained", ledger == {"alpha": 0.0},
          f"ledger={ledger}")


def test_rename_to_third_cluster_with_own_exposure():
    # 'beta' already holds another open position's 0.07; the renamed
    # trade's release must drain 'alpha' and leave 'beta' untouched.
    ledger = {"alpha": 0.15, "beta": 0.07}
    t = manual_trade(ledger)
    t.factor_cluster = "beta"
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    check("rename_onto_loaded_key_entry_released",
          ledger.get("alpha") == 0.0, f"ledger={ledger}")
    check("rename_onto_loaded_key_other_untouched",
          ledger.get("beta") == 0.07, f"ledger={ledger}")


def test_cotenant_legacy_rename_releases_only_own():
    ledger = {"alpha": 0.10}
    t1 = manual_trade(ledger, frac=0.05, invested=500.0)
    t2 = manual_trade(ledger, frac=0.05, invested=500.0)
    t2.market_id = "M2"
    t1.factor_cluster = "beta"
    ak.settle_or_mark_trade(t1, is_settled=True, outcome_yes=1)
    check("cotenant_legacy_after_renamed_settle",
          ledger == {"alpha": 0.05}, f"ledger={ledger}")
    ak.settle_or_mark_trade(t2, is_settled=True, outcome_yes=0)
    check("cotenant_legacy_both_settled_drains",
          ledger == {"alpha": 0.0}, f"ledger={ledger}")


def test_cotenant_dollar_rename_releases_only_own():
    ledger = {}
    t1 = evaluate(10_000.0, ledger, "D1", "alpha", p=0.65)
    f1 = t1.allocated_fraction
    t2 = evaluate(10_000.0, ledger, "D2", "alpha", p=0.65)
    assert 0 < f1 and t2.action != ak.TradeAction.SKIP, (f1, t2.skip_reason)
    total = ledger["alpha"]
    t1.factor_cluster = "beta"
    ak.settle_or_mark_trade(t1, is_settled=True, outcome_yes=1)
    check("cotenant_dollar_leaves_other",
          abs(ledger["alpha"] - (total - f1)) <= 1e-9 and "beta" not in ledger,
          f"ledger={ledger} total={total} f1={f1}")
    ak.settle_or_mark_trade(t2, is_settled=True, outcome_yes=1)
    check("cotenant_dollar_drains", ledger == {"alpha": 0.0},
          f"ledger={ledger}")


def test_rename_then_open_mark_then_settle():
    ledger = {}
    t = evaluate(100_000.0, ledger, "E3", "alpha")
    t.factor_cluster = "beta"
    ak.settle_or_mark_trade(t, is_settled=False)  # open mark: no release
    check("renamed_open_mark_keeps_exposure", ledger == {"alpha": 0.15},
          f"ledger={ledger}")
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=0)
    check("renamed_settle_after_mark_releases", ledger == {"alpha": 0.0},
          f"ledger={ledger}")


def test_double_settle_after_rename_releases_once():
    ledger = {}
    t = evaluate(100_000.0, ledger, "E4", "alpha")
    t.factor_cluster = "beta"
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    check("double_settle_after_rename_no_negative_no_stray",
          ledger == {"alpha": 0.0}, f"ledger={ledger}")


def test_explicit_ledger_settle_after_rename():
    # SF-BL-34868 surface preserved: passing the entry ledger itself
    # explicitly still works after a rename.
    ledger = {}
    t = evaluate(100_000.0, ledger, "E5", "alpha")
    t.factor_cluster = "beta"
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1,
                            cluster_exposure_used=ledger)
    check("explicit_ledger_after_rename_releases_entry",
          ledger == {"alpha": 0.0}, f"ledger={ledger}")


def test_mismatched_ledger_after_rename_still_raises_34868():
    ledger_a, ledger_b = {}, {"alpha": 0.10}
    t = evaluate(100_000.0, ledger_a, "E6", "alpha")
    t.factor_cluster = "beta"
    try:
        ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1,
                                cluster_exposure_used=ledger_b)
        raised = ""
    except ValueError as e:
        raised = str(e)
    check("mismatched_ledger_after_rename_raises",
          "MISMATCHED_CLUSTER_LEDGER" in raised and "34868" in raised, raised)
    check("mismatched_ledger_after_rename_ledgers_untouched",
          ledger_a == {"alpha": 0.15} and ledger_b == {"alpha": 0.10},
          f"A={ledger_a} B={ledger_b}")
    # ...and the correct release path survives the rejection.
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    check("post_rejection_settle_releases_entry",
          ledger_a == {"alpha": 0.0}, f"A={ledger_a}")


# ---- 2. Entry-key recording ------------------------------------------------

def test_entry_cluster_recorded():
    ledger = {}
    t = evaluate(100_000.0, ledger, "R1", "alpha")
    check("evaluated_trade_entry_recorded", t.entry_factor_cluster == "alpha")
    m = manual_trade({"alpha": 0.15})
    check("manual_trade_entry_snapshotted", m.entry_factor_cluster == "alpha")
    m.factor_cluster = "beta"
    check("snapshot_survives_rename",
          m.entry_factor_cluster == "alpha" and m.factor_cluster == "beta")


def test_reconstruction_via_dict_preserves_entry():
    ledger = {}
    t = evaluate(100_000.0, ledger, "R2", "alpha")
    t.factor_cluster = "beta"
    rebuilt = ak.TradeExecutionResult(**t.__dict__)
    check("rebuilt_keeps_entry", rebuilt.entry_factor_cluster == "alpha")
    ak.settle_or_mark_trade(rebuilt, is_settled=True, outcome_yes=1)
    check("rebuilt_renamed_settle_releases_entry",
          ledger == {"alpha": 0.0}, f"ledger={ledger}")


# ---- 3. Preserved surfaces: unrenamed lifecycles bit-identical ------------

def test_unrenamed_lifecycle_bit_identical():
    ledger = {}
    t = evaluate(100_000.0, ledger, "K1", "alpha")
    invested, shares = t.invested_capital, t.shares_acquired
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    check("unrenamed_ledger_drained", ledger == {"alpha": 0.0},
          f"ledger={ledger}")
    check("unrenamed_pnl_exact",
          t.net_pnl_bid_marked == shares * 1.0 - invested,
          f"pnl={t.net_pnl_bid_marked}")


def test_rename_to_same_value_is_noop():
    ledger = {}
    t = evaluate(100_000.0, ledger, "K2", "alpha")
    t.factor_cluster = "alpha"
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    check("same_value_rename_lifecycle_unchanged",
          ledger == {"alpha": 0.0}, f"ledger={ledger}")


def test_skip_trade_rename_settle_untouched():
    ledger = {"alpha": 0.05}
    t = evaluate(10_000.0, ledger, "K3", "alpha", p=0.52)
    assert t.action == ak.TradeAction.SKIP, t.skip_reason
    t.factor_cluster = "beta"
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1,
                            cluster_exposure_used=ledger)
    check("skip_rename_settle_leaves_ledger", ledger == {"alpha": 0.05},
          f"ledger={ledger}")


def test_manual_unrenamed_release_unchanged():
    ledger = {"alpha": 0.15}
    t = manual_trade(ledger)
    removed = ak.release_cluster_exposure(t)
    check("manual_unrenamed_removed", removed == 0.15 and ledger == {"alpha": 0.0},
          f"removed={removed} ledger={ledger}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{sum(1 for _, ok in PASS if ok)}/{len(PASS)} checks PASS")
