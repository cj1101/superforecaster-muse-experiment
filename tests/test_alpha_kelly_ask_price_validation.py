"""
test_ask_price_validation.py — SF-BL-9542 verification tests.

Pre-registered done/kill criteria (TASK.md):
  "Add rung-price validation (finite, in (0,1]) to the sweep ladder
   construction and best-price properties, or reject the book at Gate 0
   in evaluate_and_size_opportunity: success = the [(-0.10,1000),
   (0.50,1000)] fixture SKIPs or sweeps only the 0.50 rung (200 shares
   at q_exec 0.505) and no finite-but-impossible q_exec reaches a
   TradeExecutionResult; kill if any negative, zero, NaN or >1 ask
   price can alter invested, shares or q_exec."

Domain note: the fleet's agreed contract-price domain is the OPEN
interval (0, 1) (SF-BL-9473) — at exactly 1.0 the fee-adjusted price
exceeds the $1.00 binary payout — so a 1.0 rung is treated as
non-tradeable and skipped, which satisfies the kill criterion in its
strongest form: an invalid rung is IDENTICAL TO ITS ABSENCE in the
best-price properties, mid, the sweep tuples and (for ordered prices)
the evaluate result. NaN is the one evaluate-level exception, by
design: SF-BL-9438 pinned INVALID_EQUITY fail-closed for NaN-poisoned
books, preserved by Gate 0.6, while the sweep itself now also skips
NaN rungs (the SF-BL-9473 carve-out is superseded there).

Pre-fix reproduction (verified live 2026-10-21, after SF-BL-9473;
repro.py in the task work dir): best_yes_ask -0.10 / mid 0.15 /
best_no_bid 1.10 on the prereg fixture; phantom edge inflated
kelly_raw 0.293 -> 0.682 and invested $1,025.25 -> $1,500 (cap) on a
deep valid rung; NaN ask swept to (100.0, nan, nan); a -0.30 bid made
best_no_ask 1.30 and BLOCKED the clean BUY_NO trade.

Run:  python3 test_ask_price_validation.py   (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_derived_no_price.py.
"""
import io
import math
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(
    os.path.expanduser("~/workspace/superforecaster"), "superforecaster"))

import alpha_kelly as ak  # noqa: E402

PASS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def lvl(price, size=1000.0):
    return ak.OrderBookLevel(price=price, size_shares=size)


def book(asks, bids, cluster="C9542", fee=0.01, domain=ak.MarketDomain.PREDICTION_MARKET):
    return ak.OrderBookSnapshot(
        market_id="SF-BL-9542", title="fixture", domain=domain,
        factor_cluster=cluster, est_days_to_resolution=5.0,
        yes_asks=asks, yes_bids=bids, fee_rate=fee)


def fixture_book():
    """Prereg fixture: asks [(-0.10,1000),(0.50,1000)], fee 0.01."""
    return book([lvl(-0.10), lvl(0.50)], [lvl(0.40)])


def clean_book():
    return book([lvl(0.50)], [lvl(0.40)])


BAD_PRICES = [-0.10, -1.0, 0.0, 1.0, 1.50, 2.0,
              float("inf"), float("-inf"), float("nan"), True, False, "0.5", None]
ORDERED_BAD = [p for p in BAD_PRICES if p != p]  # placeholder, replaced below
ORDERED_BAD = [-0.10, -1.0, 0.0, 1.0, 1.50, 2.0,
               float("inf"), float("-inf"), True, False, "0.5", None]


# ---- 1. Prereg fixture --------------------------------------------------------

def test_prereg_fixture_best_prices_are_clean():
    b, c = fixture_book(), clean_book()
    assert b.best_yes_ask == c.best_yes_ask == 0.50, b.best_yes_ask
    assert b.mid_price == c.mid_price == 0.45, b.mid_price
    assert b.best_no_bid == c.best_no_bid == 0.50, b.best_no_bid
    assert b.best_yes_bid == c.best_yes_bid == 0.40
    assert b.best_no_ask == c.best_no_ask == 0.60


def test_prereg_fixture_sweeps_only_the_valid_rung():
    # $100 budget at fee-adjusted 0.505 -> 198.0198 shares at q_exec 0.505
    # (the prereg's "200 shares" is the fee-free count 100/0.50; the
    # fee-adjusted values asserted here are the exact sweep outputs).
    for min_edge in (0.05, -1.0):
        got = ak.sweep_order_book(fixture_book(), ak.TradeAction.BUY_YES,
                                  100.0, 0.90, min_net_edge=min_edge)
        assert got == (100.0, 198.01980198019803, 0.505), (min_edge, got)
        clean = ak.sweep_order_book(clean_book(), ak.TradeAction.BUY_YES,
                                    100.0, 0.90, min_net_edge=min_edge)
        assert got == clean, (got, clean)


def test_prereg_fixture_evaluate_identical_to_clean():
    for p in (0.65, 0.90):
        cl_m, cl_c = {}, {}
        mixed = ak.evaluate_and_size_opportunity(
            fixture_book(), p, 0.05, 10_000.0, cl_m)
        clean = ak.evaluate_and_size_opportunity(
            clean_book(), p, 0.05, 10_000.0, cl_c)
        assert mixed.action == clean.action, (p, mixed.action, clean.action)
        assert mixed.invested_capital == clean.invested_capital
        assert mixed.shares_acquired == clean.shares_acquired
        assert mixed.exec_price == clean.exec_price
        assert mixed.kelly_fraction_raw == clean.kelly_fraction_raw
        assert mixed.market_mid_prob == clean.market_mid_prob == 0.45
        assert mixed.liquidation_bid_price == clean.liquidation_bid_price
        assert cl_m == cl_c


def test_phantom_edge_no_longer_inflates_sizing():
    # Deep valid rung: pre-fix the -0.10 rung inflated kelly_raw to
    # 0.6821 and invested to the $1,500 cluster cap; clean is 0.2929 /
    # $1,025.2525...
    deep_mixed = book([lvl(-0.10), lvl(0.50, 100000.0)], [lvl(0.40)])
    deep_clean = book([lvl(0.50, 100000.0)], [lvl(0.40)])
    cl_m, cl_c = {}, {}
    m = ak.evaluate_and_size_opportunity(deep_mixed, 0.65, 0.05, 10_000.0, cl_m)
    c = ak.evaluate_and_size_opportunity(deep_clean, 0.65, 0.05, 10_000.0, cl_c)
    assert m.invested_capital == c.invested_capital == 1025.2525252525254, m
    assert abs(m.kelly_fraction_raw - 0.2929292929292929) < 1e-12, m
    assert cl_m == cl_c == {"C9542": 0.10252525252525255}, (cl_m, cl_c)


# ---- 2. Kill grid: invalid ask prices are identical to their absence ---------

def test_sweep_yes_identical_to_clean_grid():
    clean = book([lvl(0.50, 2000.0), lvl(0.55, 2000.0)], [lvl(0.40)])
    for bad in BAD_PRICES:
        for pos in (0, 1, 2):
            asks = [lvl(0.50, 2000.0), lvl(0.55, 2000.0)]
            asks.insert(pos, lvl(bad, 700.0))
            mixed = book(asks, [lvl(0.40)])
            for budget in (100.0, 500.0, 5000.0):
                for belief in (0.65, 0.90):
                    for min_edge in (0.05, -1.0):
                        a = ak.sweep_order_book(mixed, ak.TradeAction.BUY_YES,
                                                budget, belief, min_net_edge=min_edge)
                        b = ak.sweep_order_book(clean, ak.TradeAction.BUY_YES,
                                                budget, belief, min_net_edge=min_edge)
                        assert a == b, (bad, pos, budget, belief, min_edge, a, b)
                        assert all(math.isfinite(v) for v in a), a


def test_all_bad_ask_ladder_sweeps_to_zero():
    for bad in BAD_PRICES:
        b = book([lvl(bad)], [lvl(0.40)])
        got = ak.sweep_order_book(b, ak.TradeAction.BUY_YES, 1200.0, 0.90,
                                  min_net_edge=-1.0)
        assert got == (0.0, 0.0, 0.0), (bad, got)


def test_best_prices_identical_to_clean_grid():
    for bad in BAD_PRICES:
        for side in ("ask", "bid"):
            base_asks, base_bids = [lvl(0.50)], [lvl(0.40)]
            if side == "ask":
                mixed = book([lvl(bad), lvl(0.50)], base_bids)
            else:
                mixed = book(base_asks, [lvl(bad), lvl(0.40)])
            clean = book(base_asks, base_bids)
            for prop in ("best_yes_ask", "best_yes_bid", "best_no_ask",
                         "best_no_bid", "mid_price"):
                assert getattr(mixed, prop) == getattr(clean, prop), \
                    (bad, side, prop, getattr(mixed, prop), getattr(clean, prop))


def test_all_invalid_side_falls_back_to_sentinel():
    b = book([lvl(-0.10)], [lvl(1.50)])
    assert b.best_yes_ask == 0.99 and b.best_no_bid == 0.01, b.best_yes_ask
    assert b.best_yes_bid == 0.01 and b.best_no_ask == 0.99, b.best_yes_bid
    assert abs(b.mid_price - 0.50) < 1e-12, b.mid_price


def test_evaluate_ordered_bad_asks_identical_to_clean():
    # Ordered (non-NaN) invalid prices: evaluate must neither fabricate
    # nor deny — the result equals the clean book's, field by field.
    for bad in ORDERED_BAD:
        for pos in (0, 1):
            asks = [lvl(0.50, 5000.0)]
            asks.insert(pos, lvl(bad, 700.0))
            cl_m, cl_c = {}, {}
            m = ak.evaluate_and_size_opportunity(
                book(asks, [lvl(0.40)]), 0.65, 0.05, 10_000.0, cl_m)
            c = ak.evaluate_and_size_opportunity(
                book([lvl(0.50, 5000.0)], [lvl(0.40)]), 0.65, 0.05, 10_000.0, cl_c)
            assert (m.action, m.invested_capital, m.shares_acquired, m.exec_price,
                    m.kelly_fraction_raw, m.market_mid_prob,
                    m.liquidation_bid_price) == \
                   (c.action, c.invested_capital, c.shares_acquired, c.exec_price,
                    c.kelly_fraction_raw, c.market_mid_prob,
                    c.liquidation_bid_price), (bad, pos, m, c)
            assert cl_m == cl_c, (bad, pos, cl_m, cl_c)


def test_no_impossible_q_exec_reaches_a_result():
    # Every non-SKIP result across the bad-price grid carries a finite
    # exec_price equal to the clean book's and a liquidation bid in [0,1].
    for bad in BAD_PRICES:
        b = book([lvl(bad), lvl(0.50, 5000.0)], [lvl(0.40)])
        r = ak.evaluate_and_size_opportunity(b, 0.65, 0.05, 10_000.0, {})
        if r.action != ak.TradeAction.SKIP:
            assert math.isfinite(r.exec_price), (bad, r)
            assert r.exec_price == 0.505, (bad, r)
            assert 0.0 <= r.liquidation_bid_price <= 1.0, (bad, r)


# ---- 3. NaN: sweep skips, evaluate fails closed (SF-BL-9438 preserved) -------

def test_nan_ask_sweep_identical_to_clean():
    for pos in (0, 1):
        asks = [lvl(0.50, 2000.0)]
        asks.insert(pos, lvl(float("nan"), 700.0))
        mixed = book(asks, [lvl(0.40)])
        got = ak.sweep_order_book(mixed, ak.TradeAction.BUY_YES, 500.0, 0.90,
                                  min_net_edge=-1.0)
        clean = ak.sweep_order_book(clean_book(), ak.TradeAction.BUY_YES,
                                    500.0, 0.90, min_net_edge=-1.0)
        # clean_book has a single 1000-share rung; use the mixed-clean twin:
        twin = book([lvl(0.50, 2000.0)], [lvl(0.40)])
        clean = ak.sweep_order_book(twin, ak.TradeAction.BUY_YES, 500.0, 0.90,
                                    min_net_edge=-1.0)
        assert got == clean and all(math.isfinite(v) for v in got), (pos, got)


def test_nan_rung_evaluate_skips_invalid_equity():
    # First-rung and deep-rung NaN, ask side and bid side: SKIP with the
    # SF-BL-9438 INVALID_EQUITY reason, zero size, cluster untouched.
    cases = [
        book([lvl(float("nan")), lvl(0.50)], [lvl(0.40)]),
        book([lvl(0.50), lvl(float("nan"))], [lvl(0.48)], fee=0.0),
        book([lvl(0.50)], [lvl(float("nan")), lvl(0.40)]),
    ]
    for b in cases:
        cl = {}
        r = ak.evaluate_and_size_opportunity(b, 0.80, 0.0, 10_000.0, cl)
        assert r.action == ak.TradeAction.SKIP, r
        assert r.skip_reason and r.skip_reason.startswith("INVALID_EQUITY"), r.skip_reason
        assert r.invested_capital == 0.0 and r.shares_acquired == 0.0
        assert r.exec_price == 0.0 and r.kelly_fraction_raw == 0.0
        assert cl == {}, cl


def test_zero_size_nan_rung_does_not_trip_gate():
    # SF-BL-9398: a zero-size rung is not live — it must not poison
    # evaluate (or the best prices) even at a NaN price.
    b = book([lvl(0.50, 5000.0), lvl(float("nan"), 0.0)], [lvl(0.40)])
    r = ak.evaluate_and_size_opportunity(b, 0.65, 0.05, 10_000.0, {})
    assert r.action == ak.TradeAction.BUY_YES, r
    assert r.exec_price == 0.505, r


# ---- 4. Bid side: derived NO prices can no longer deny/fabricate --------------

def _unlock(bids):
    return ak.OrderBookSnapshot(
        market_id="CRYPTO-UNLOCK-HOLD-SUPPORT",
        title="Token ABC holds above $2.00 through 12% cliff unlock week",
        domain=ak.MarketDomain.CRYPTO, factor_cluster="TOKEN_UNLOCK_SHORT",
        est_days_to_resolution=5.0,
        yes_asks=[ak.OrderBookLevel(price=0.68, size_shares=1200)],
        yes_bids=bids, fee_rate=0.01, on_chain_verified=True)


def test_bad_bid_front_no_longer_denies_clean_buy_no():
    clean_bids = [lvl(0.65, 900.0), lvl(0.60, 1500.0)]
    for bad in ORDERED_BAD:
        cl_m, cl_c = {}, {}
        m = ak.evaluate_and_size_opportunity(
            _unlock([lvl(bad)] + clean_bids), 0.28, 0.05, 10_000.0, cl_m)
        c = ak.evaluate_and_size_opportunity(
            _unlock(clean_bids), 0.28, 0.05, 10_000.0, cl_c)
        assert m.action == c.action == ak.TradeAction.BUY_NO, (bad, m.action)
        assert m.invested_capital == c.invested_capital == 924.15, (bad, m)
        assert m.shares_acquired == c.shares_acquired == 2400.0
        assert m.exec_price == c.exec_price == 0.3850625
        assert cl_m == cl_c, (bad, cl_m, cl_c)


def test_liquidation_bid_from_evaluate_always_in_bounds():
    # best_yes_bid / best_no_bid feed liquidation_bid_price directly:
    # with bad rungs present the result's mark must stay within [0, 1]
    # (pre-fix the prereg fixture produced best_no_bid 1.10).
    r = ak.evaluate_and_size_opportunity(fixture_book(), 0.65, 0.05, 10_000.0, {})
    assert r.action == ak.TradeAction.BUY_YES and r.liquidation_bid_price == 0.40, r
    marked = ak.settle_or_mark_trade(r, is_settled=False)
    assert math.isfinite(marked.net_pnl_bid_marked), marked


# ---- 5. Valid books unchanged ---------------------------------------------------

def test_helper_domain():
    for good in (0.5, 0.001, 0.999, 1e-9):
        assert ak._is_tradeable_price(good), good
        assert ak._is_valid_contract_price(good), good
    for bad in BAD_PRICES:
        assert not ak._is_tradeable_price(bad), bad


def test_near_boundary_valid_prices_still_trade():
    b = book([lvl(0.999, 1000.0)], [lvl(0.40)], fee=0.0)
    assert b.best_yes_ask == 0.999
    inv, sh, q = ak.sweep_order_book(b, ak.TradeAction.BUY_YES, 100.0, 0.9999,
                                     min_net_edge=-1.0)
    assert inv > 0 and abs(q - 0.999) < 1e-9, (inv, sh, q)
    b = book([lvl(0.001, 1000.0)], [lvl(0.40)], fee=0.0)
    assert b.best_yes_ask == 0.001


def test_benchmark_simulation_unchanged():
    buf = io.StringIO()
    with redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    assert '"north_star_score": 0.1059' in out, out[:2000]
    assert '"north_star_score": -0.1499' in out, out[:2000]


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    for name, fn in fns:
        try:
            fn()
            check(name, True)
        except Exception as e:  # noqa: BLE001
            check(name, False, f"-> {type(e).__name__}: {e}")
    print(f"\n{sum(PASS)}/{len(PASS)} passed")
    sys.exit(0 if all(PASS) else 1)
