"""SF-BL-8891 — fee-adjusted price parity tests.

Bug: effective_rung_price = price * (1 + fee_rate) had no cap at 1.0, so
near-certainty rungs (ask 0.999 at fee 0.015 -> 1.013985) swept to
q_exec >= 1.0 on every full-walk / naive path, booking executed trades
whose expected_log_wealth_growth was zeroed by its q>=1 guard while
kl_divergence_binary silently clipped q to 1-1e-6.

Fix: (1) parity stop in sweep_order_book's loop — a rung whose
fee-adjusted price is >= 1.0 breaks the sweep before any hurdle test,
for every hurdle including -1.0 full walks and the shrunk-LCB stop;
(2) post-sweep backstop in evaluate_and_size_opportunity — a q_exec
>= 1.0 SKIPs as Q_EXEC_AT_OR_ABOVE_PARITY, zero size, cluster untouched.

Run: python3 -m pytest test_fee_parity.py -q   (or plain python3)
"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster"))
from superforecaster import alpha_kelly as ak


def book(asks, bids, fee=0.015, cluster="T"):
    return ak.OrderBookSnapshot(
        market_id="SF-BL-8891-T", title="t",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster=cluster,
        est_days_to_resolution=7.0,
        yes_asks=[ak.OrderBookLevel(p, s) for p, s in asks],
        yes_bids=[ak.OrderBookLevel(p, s) for p, s in bids],
        fee_rate=fee)


PREREG = dict(asks=[(0.999, 100_000)], bids=[(0.998, 100_000)], fee=0.015)
MIRROR = dict(asks=[(0.002, 100_000)], bids=[(0.001, 100_000)], fee=0.015)


# ---- 1. prereg fixture: no fill at any hurdle, either side --------------
def test_prereg_fixture_no_fill_any_hurdle():
    for hurdle in (0.05, 0.0, -0.5, -1.0):
        assert ak.sweep_order_book(book(**PREREG), ak.TradeAction.BUY_YES,
                                   1_000.0, 0.99, min_net_edge=hurdle) == (0.0, 0.0, 0.0)
        assert ak.sweep_order_book(book(**MIRROR), ak.TradeAction.BUY_NO,
                                   1_000.0, 0.99, min_net_edge=hurdle) == (0.0, 0.0, 0.0)


def test_prereg_arithmetic_pinned():
    assert abs(0.999 * 1.015 - 1.013985) < 1e-12
    inv, sh, q = ak.sweep_order_book(book(**PREREG), ak.TradeAction.BUY_YES,
                                     1_000.0, 0.99, min_net_edge=-1.0)
    assert q < 1.0  # the invariant, whatever the fill


# ---- 2. fee x price grid: q_exec < 1.0 on every fill --------------------
def test_grid_no_q_exec_at_or_above_one():
    for ask in (0.99, 0.995, 0.999):
        for fee in (0.0, 0.005, 0.01, 0.015, 0.0175, 0.07, 0.10):
            b = book([(ask, 100_000)], [(round(ask - 0.001, 6), 100_000)], fee=fee)
            for side in (ak.TradeAction.BUY_YES, ak.TradeAction.BUY_NO):
                for hurdle in (0.05, -1.0):
                    inv, sh, q = ak.sweep_order_book(
                        b, side, 100_000.0, 0.999, min_net_edge=hurdle)
                    assert sh <= 0 or q < 1.0, (ask, fee, side, hurdle, q)
                    if ask * (1 + fee) >= 1.0 and side == ak.TradeAction.BUY_YES:
                        assert (inv, sh, q) == (0.0, 0.0, 0.0), (ask, fee, q)


# ---- 3. boundary: effective price exactly 1.0 blocked, just below fills -
def test_exact_parity_boundary():
    p_exact = 1.0 / 1.01  # p * 1.01 == 1.0 exactly in float
    assert p_exact * 1.01 == 1.0
    b = book([(p_exact, 100_000)], [(0.98, 100_000)], fee=0.01)
    assert ak.sweep_order_book(b, ak.TradeAction.BUY_YES, 1_000.0, 0.999,
                               min_net_edge=-1.0) == (0.0, 0.0, 0.0)
    b = book([(0.99, 100_000)], [(0.98, 100_000)], fee=0.01)  # eff 0.9999
    inv, sh, q = ak.sweep_order_book(b, ak.TradeAction.BUY_YES, 1_000.0, 0.999,
                                     min_net_edge=-1.0)
    assert inv > 0 and abs(q - 0.9999) < 1e-9


# ---- 4. mixed ladder identical to the parity-truncated ladder -----------
def test_mixed_ladder_identical_to_clean():
    cases = [
        ([(0.50, 100), (0.999, 100_000)], [(0.49, 100_000)], ak.TradeAction.BUY_YES),
        ([(0.50, 100)], [(0.49, 100_000)], ak.TradeAction.BUY_YES),  # control row shape
        ([(0.999, 100_000)], [(0.998, 50), (0.001, 100_000)], ak.TradeAction.BUY_NO),
    ]
    mixed = book([(0.50, 100), (0.999, 100_000)], [(0.49, 100_000)])
    clean = book([(0.50, 100)], [(0.49, 100_000)])
    for budget in (100.0, 10_000.0, 100_000.0):
        for hurdle in (0.05, -1.0):
            assert ak.sweep_order_book(mixed, ak.TradeAction.BUY_YES, budget,
                                       0.99, min_net_edge=hurdle) == \
                   ak.sweep_order_book(clean, ak.TradeAction.BUY_YES, budget,
                                       0.99, min_net_edge=hurdle)
    mixed_no = book([(0.999, 100_000)], [(0.998, 50), (0.001, 100_000)])
    clean_no = book([(0.999, 100_000)], [(0.998, 50)])
    for budget in (100.0, 100_000.0):
        assert ak.sweep_order_book(mixed_no, ak.TradeAction.BUY_NO, budget,
                                   0.99, min_net_edge=-1.0) == \
               ak.sweep_order_book(clean_no, ak.TradeAction.BUY_NO, budget,
                                   0.99, min_net_edge=-1.0)


# ---- 5. LCB opt-in stop also respects parity ----------------------------
def test_lcb_sweep_respects_parity():
    b = book(**PREREG)
    assert ak.sweep_order_book(b, ak.TradeAction.BUY_YES, 1_000.0, 0.99,
                               min_net_edge=-1.0, posterior_n_eff=30.0) == (0.0, 0.0, 0.0)


# ---- 6. budget-sensitivity full walks emit only q_exec < 1.0 ------------
def test_budget_sensitivity_budgets_below_parity():
    b = book([(0.90, 100), (0.999, 100_000)], [(0.89, 100_000)])
    chk = ak.budget_sensitivity_check(b, ak.TradeAction.BUY_YES, 0.99, 50_000.0, 0.9135)
    for bd in chk["budgets"].values():
        assert bd["shares"] <= 100.0 + 1e-9  # only the cheap rung can fill
        assert bd["q_exec"] < 1.0


# ---- 7. evaluate: near-certainty book cannot execute at parity ----------
def test_evaluate_prereg_book_skips():
    for cfg in (ak.FleetRiskConfig(),
                ak.FleetRiskConfig(max_round_trip_cost=None,
                                   budget_sensitivity_veto_enabled=False)):
        cluster = {}
        r = ak.evaluate_and_size_opportunity(book(**PREREG), 0.99, 0.05,
                                             100_000.0, cluster, cfg)
        assert r.action == ak.TradeAction.SKIP and r.exec_price == 0.0
        assert cluster == {}


def test_evaluate_backstop_on_stubbed_parity_sweep():
    orig = ak.sweep_order_book
    try:
        ak.sweep_order_book = lambda **kw: (100.0, 100.0, 1.01)
        cluster = {}
        b = book([(0.50, 100_000)], [(0.49, 100_000)], fee=0.01)
        r = ak.evaluate_and_size_opportunity(b, 0.80, 0.05, 100_000.0, cluster)
        assert r.action == ak.TradeAction.SKIP
        assert r.skip_reason.startswith("Q_EXEC_AT_OR_ABOVE_PARITY")
        assert r.invested_capital == 0.0 and r.shares_acquired == 0.0
        assert cluster == {}
    finally:
        ak.sweep_order_book = orig


# ---- 8. controls unchanged ----------------------------------------------
def test_below_parity_control_trade_unchanged():
    b = book([(0.58, 1500), (0.61, 2500)], [(0.56, 1500)], fee=0.005)
    inv, sh, q = ak.sweep_order_book(b, ak.TradeAction.BUY_YES, 2_000.0, 0.80)
    assert inv > 0 and q < 1.0
    r = ak.evaluate_and_size_opportunity(
        book([(0.58, 1500), (0.61, 2500)], [(0.56, 1500)],
             fee=0.005, cluster="C1"), 0.80, 0.05, 100_000.0, {})
    assert r.action == ak.TradeAction.BUY_YES and 0.58 < r.exec_price < 1.0
    assert r.expected_log_growth > 0.0


def test_benchmark_unchanged():
    # Ground truth captured on the CURRENT tree immediately before the
    # SF-BL-8891 patch (scratch copy with only the 8891 hunks reverted):
    # disciplined +0.0547 / naive -0.1499, byte-identical after the
    # patch. (The +0.1059 in the module header dates from an earlier
    # tree state; the drift predates this task and is not its doing —
    # no benchmark book has a fee-adjusted rung anywhere near parity.)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    assert '"north_star_score": 0.0547' in out, out[:2000]
    assert '"north_star_score": -0.1499' in out, out[:2000]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} passed")
