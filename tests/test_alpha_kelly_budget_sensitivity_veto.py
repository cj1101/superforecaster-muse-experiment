"""SF-BL-3659 — Gate BS: budget-sensitivity veto tests.

Gate: re-sweep each candidate at 0.5x and 2x its Kelly target budget
(full walk, marginal-edge stop disabled) and SKIP when the post-cost
edge sign flips or D_KL per dollar at least halves vs the admitted 1x
execution. Kill criterion (census, separate deliverable): vetoed if
<2% of forward candidates passing at 1x change verdict across budgets;
material at >=10%. Census result: 51.0% primary / 22.5% strict —
NOT killed.
"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster"))
from superforecaster import alpha_kelly as ak

FEE = 0.01


def book(asks, bids, cluster="c1"):
    return ak.OrderBookSnapshot(
        market_id="m1", title="t", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster=cluster, est_days_to_resolution=7.0,
        yes_asks=[ak.OrderBookLevel(p, s) for p, s in asks],
        yes_bids=[ak.OrderBookLevel(p, s) for p, s in bids], fee_rate=FEE)


def ev(b, p, eq=10_000.0, cfg=None, cluster=None):
    cluster = {} if cluster is None else cluster
    r = ak.evaluate_and_size_opportunity(
        b, p, 0.05, eq, cluster, cfg or ak.FleetRiskConfig())
    return r, cluster


def approx(a, b, tol=1e-3):
    return abs(a - b) <= tol


# --- helper-level: exact full-walk arithmetic ---------------------------

def test_helper_deep_book_is_size_stable():
    b = book([(0.50, 100_000)], [(0.48, 100_000)])
    chk = ak.budget_sensitivity_check(b, ak.TradeAction.BUY_YES, 0.80, 1000.0, 0.505)
    assert chk["veto"] is False and chk["triggers"] == []
    for s in (0.5, 2.0):
        assert approx(chk["budgets"][s]["q_exec"], 0.505)
        assert approx(chk["budgets"][s]["edge"], 0.295)
        assert approx(chk["budgets"][s]["kl"], chk["kl_ref"], 1e-9)


def test_helper_dkl_halved_at_2x_exact():
    # rung1 fills the whole 1x budget at 0.505; at 2x the walk reaches
    # the 0.98 rung and q_exec rises to 0.6666, D_KL 0.1868 -> 0.0437.
    b = book([(0.50, 2000), (0.98, 100_000)], [(0.48, 100_000)])
    chk = ak.budget_sensitivity_check(b, ak.TradeAction.BUY_YES, 0.80, 1000.0, 0.505)
    assert chk["veto"] is True
    assert len(chk["triggers"]) == 1 and chk["triggers"][0].startswith("dkl_halved_2.0x")
    assert approx(chk["budgets"][2.0]["q_exec"], 0.66662, 1e-4)
    assert approx(chk["budgets"][2.0]["kl"], 0.04372, 1e-4)
    assert chk["budgets"][2.0]["kl"] <= 0.5 * chk["kl_ref"]
    assert chk["budgets"][0.5]["kl"] == chk["kl_ref"]  # 0.5x unchanged


def test_helper_edge_flip_both_scales_exact():
    # tiny cheap rung (200 sh), then 0.99: even the 0.5x full walk
    # crosses the belief. kl at a flipped budget is 0.0 by definition.
    b = book([(0.50, 200), (0.99, 100_000)], [(0.48, 100_000)])
    chk = ak.budget_sensitivity_check(b, ak.TradeAction.BUY_YES, 0.80, 1500.0, 0.505)
    assert chk["veto"] is True
    names = [t.split(" ")[0] for t in chk["triggers"]]
    assert names == ["edge_flip_0.5x", "edge_flip_2.0x"]
    assert approx(chk["budgets"][0.5]["q_exec"], 0.88332, 1e-4)
    assert approx(chk["budgets"][0.5]["edge"], -0.08332, 1e-4)
    assert chk["budgets"][0.5]["kl"] == 0.0 and chk["budgets"][2.0]["kl"] == 0.0


def test_helper_no_fill_triggers():
    b = book([(0.50, 0)], [(0.48, 100_000)])  # no live ask depth at all
    chk = ak.budget_sensitivity_check(b, ak.TradeAction.BUY_YES, 0.80, 1000.0, 0.505)
    assert chk["veto"] is True
    assert set(t.split(" ")[0] for t in chk["triggers"]) == {"no_fill_0.5x", "no_fill_2.0x"}


# --- evaluate-level: the gate ------------------------------------------

def test_deep_book_trades_unchanged():
    b = book([(0.50, 100_000)], [(0.48, 100_000)])
    r, cl = ev(b, 0.80, eq=100_000.0)
    assert r.action == ak.TradeAction.BUY_YES
    assert approx(r.invested_capital, 15_000.0) and approx(r.exec_price, 0.505)
    assert cl == {"c1": 0.15}


def test_thin_book_vetoed_dkl_halved():
    b = book([(0.50, 2000), (0.98, 100_000)], [(0.48, 100_000)])
    r, cl = ev(b, 0.80)
    assert r.action == ak.TradeAction.SKIP
    assert r.skip_reason.startswith("BUDGET_SENSITIVITY_VETO (dkl_halved_2.0x")
    assert r.invested_capital == 0.0 and r.shares_acquired == 0.0 and cl == {}


def test_tiny_rung_vetoed_edge_flip_at_half_budget():
    # The criterion's headline case: passes the gated 1x sweep ($101
    # fill on the cheap rung) yet flips sign at 0.5x on a full walk.
    b = book([(0.50, 200), (0.99, 100_000)], [(0.48, 100_000)])
    r, cl = ev(b, 0.80)
    assert r.action == ak.TradeAction.SKIP
    assert "edge_flip_0.5x" in r.skip_reason and "edge_flip_2.0x" in r.skip_reason
    assert r.invested_capital == 0.0 and cl == {}


def test_buy_no_mirror_vetoed():
    b = book([(0.52, 100_000)], [(0.50, 200), (0.01, 100_000)])
    r, cl = ev(b, 0.20)
    assert r.action == ak.TradeAction.SKIP
    assert r.skip_reason.startswith("BUDGET_SENSITIVITY_VETO (edge_flip_0.5x")
    assert cl == {}


def test_disabled_config_restores_pre_veto_behaviour():
    cfg = ak.FleetRiskConfig(budget_sensitivity_veto_enabled=False)
    b = book([(0.50, 200), (0.99, 100_000)], [(0.48, 100_000)])
    r, cl = ev(b, 0.80, cfg=cfg)
    assert r.action == ak.TradeAction.BUY_YES
    assert approx(r.invested_capital, 101.0) and approx(r.exec_price, 0.505)
    assert approx(cl["c1"], 0.0101, 1e-6)


def test_veto_fires_regardless_of_gate_d():
    cfg = ak.FleetRiskConfig(max_round_trip_cost=None)
    b = book([(0.50, 200), (0.99, 100_000)], [(0.48, 100_000)])
    r, _ = ev(b, 0.80, cfg=cfg)
    assert r.action == ak.TradeAction.SKIP
    assert r.skip_reason.startswith("BUDGET_SENSITIVITY_VETO")


def test_no_edge_candidate_keeps_earlier_reason():
    b = book([(0.50, 100_000)], [(0.48, 100_000)])
    r, _ = ev(b, 0.52)  # edge 0.015 < hurdle: never reaches Gate BS
    assert r.action == ak.TradeAction.SKIP
    assert r.skip_reason.startswith("INSUFFICIENT_NET_EDGE")


def test_earlier_gate_reasons_keep_precedence():
    # A candidate failing an earlier gate (here the edge screen at a
    # raised hurdle) never reaches Gate BS: its reason is unchanged.
    b = book([(0.50, 100_000)], [(0.48, 100_000)])
    r, _ = ev(b, 0.80, cfg=ak.FleetRiskConfig(min_net_edge=0.40))
    assert r.action == ak.TradeAction.SKIP
    assert r.skip_reason.startswith("INSUFFICIENT_NET_EDGE")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1; print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
