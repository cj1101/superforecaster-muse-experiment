"""
test_zero_size_levels.py — SF-BL-9398 verification tests.

Pre-registered done/kill criteria (TASK.md):
  "best-price properties and the evaluate edge screen must ignore
   zero/negative-size levels on both sides including the derived NO
   prices, and a book whose only levels are zero-size must SKIP as
   empty; kill if any fixture's screened edge uses a price backed by
   zero shares."

Pre-fix reproduction (verified live 2026-10-21, before the patch;
see repro.py in this directory):
  asks [(0.40, 0), (0.62, 100000)], bid 0.60 live:
    best_yes_ask=0.40, mid=0.50, best_no_bid=0.60 (want 0.62 / 0.61 / 0.38);
    evaluate(p=0.65) passed the screen on the phantom 0.40 edge (0.246)
    and only SKIPs later at ORDER_BOOK_DEPTH_EXHAUSTED.
  bids [(0.88, 0), (0.50, 100000)], ask 0.62 live:
    best_yes_bid=0.88, best_no_ask=0.12, mid=0.75 (want 0.50 / 0.50 / 0.56).
  sweep with a 0-share rung at 0.90 in front of a real 0.50 rung:
    (0.0, 0.0, 0.0) — the phantom rung blocked a fillable trade.

Strongest form of the kill check used throughout: a book with phantom
zero/negative rungs must behave IDENTICALLY (properties, action,
skip_reason, exec price, sizes) to the same book with those rungs
removed, for every fixture and a grid of probabilities.

Run:  python3 test_zero_size_levels.py   (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_fee_rate.py.
"""
import io
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


def _book(asks, bids, fee=0.01):
    return ak.OrderBookSnapshot(
        market_id="FIXTURE", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C9398", est_days_to_resolution=5,
        yes_asks=[ak.OrderBookLevel(price=p, size_shares=s) for p, s in asks],
        yes_bids=[ak.OrderBookLevel(price=p, size_shares=s) for p, s in bids],
        fee_rate=fee)


def _eval(book, p):
    return ak.evaluate_and_size_opportunity(book, p, 0.0, 10000, {})


def _same_result(a, b):
    return (a.action == b.action and a.skip_reason == b.skip_reason
            and a.exec_price == b.exec_price
            and a.invested_capital == b.invested_capital
            and a.shares_acquired == b.shares_acquired
            and a.market_mid_prob == b.market_mid_prob
            and a.liquidation_bid_price == b.liquidation_bid_price
            and a.kelly_fraction_raw == b.kelly_fraction_raw
            and a.allocated_fraction == b.allocated_fraction)


# ---- 1. Best-price properties ignore zero/negative levels -------------------

def test_best_yes_ask_ignores_zero_and_negative():
    b = _book([(0.40, 0), (0.62, 100000)], [(0.60, 100000)])
    assert b.best_yes_ask == 0.62, b.best_yes_ask
    b = _book([(0.40, -5), (0.30, 0), (0.62, 100000)], [(0.60, 100000)])
    assert b.best_yes_ask == 0.62, b.best_yes_ask


def test_best_yes_bid_ignores_zero_and_negative():
    b = _book([(0.62, 100000)], [(0.88, 0), (0.50, 100000)])
    assert b.best_yes_bid == 0.50, b.best_yes_bid
    b = _book([(0.62, 100000)], [(0.88, -1), (0.90, 0), (0.50, 100000)])
    assert b.best_yes_bid == 0.50, b.best_yes_bid


def test_derived_no_prices_and_mid_use_live_levels_only():
    b = _book([(0.40, 0), (0.62, 100000)], [(0.60, 100000)])
    assert b.best_no_bid == 0.38, b.best_no_bid          # 1 - 0.62, not 1 - 0.40
    assert abs(b.mid_price - 0.61) < 1e-12, b.mid_price  # not anchored at 0.50/0.39
    b = _book([(0.62, 100000)], [(0.88, 0), (0.50, 100000)])
    assert b.best_no_ask == 0.50, b.best_no_ask          # 1 - 0.50, not phantom 0.12
    assert abs(b.mid_price - 0.56) < 1e-12, b.mid_price


def test_all_zero_side_falls_back_to_sentinel():
    b = _book([(0.40, 0), (0.62, -5)], [(0.50, 100000)])
    assert b.best_yes_ask == 0.99, b.best_yes_ask
    assert b.best_no_bid == 0.01, b.best_no_bid
    b = _book([(0.62, 100000)], [(0.88, 0)])
    assert b.best_yes_bid == 0.01, b.best_yes_bid
    assert b.best_no_ask == 0.99, b.best_no_ask


def test_properties_identical_to_clean_book():
    phantom = _book([(0.40, 0), (0.62, 100000)], [(0.88, 0), (0.50, 100000)])
    clean = _book([(0.62, 100000)], [(0.50, 100000)])
    for prop in ("best_yes_ask", "best_yes_bid", "best_no_ask", "best_no_bid", "mid_price"):
        assert getattr(phantom, prop) == getattr(clean, prop), prop


# ---- 2. Edge screen never uses a zero-backed price (kill criterion) ---------

def test_prereg_ask_fixture_skips_at_screen_on_real_price():
    # TASK fixture: p=0.65 vs first REAL ask 0.62 (eff 0.6262) -> edge 0.024.
    r = _eval(_book([(0.40, 0), (0.62, 100000)], [(0.60, 100000)]), 0.65)
    assert r.action == ak.TradeAction.SKIP, r
    assert r.skip_reason == "INSUFFICIENT_NET_EDGE (max_edge=0.024 < 0.050)", r.skip_reason
    assert r.invested_capital == 0.0 and r.shares_acquired == 0.0
    assert r.kelly_fraction_raw == 0.0 and r.allocated_fraction == 0.0  # EC-09


def test_prereg_bid_fixture_no_phantom_no_ask():
    # p=0.50: phantom best_no_ask 0.12 would screen edge_no 0.379 and trade;
    # the real derived NO ask is 0.50 (eff 0.505) -> edge_no -0.005 -> SKIP.
    r = _eval(_book([(0.62, 100000)], [(0.88, 0), (0.50, 100000)]), 0.50)
    assert r.action == ak.TradeAction.SKIP, r
    assert r.skip_reason.startswith("INSUFFICIENT_NET_EDGE"), r.skip_reason
    assert r.invested_capital == 0.0


def test_evaluate_identical_to_clean_book_on_grid():
    phantom = [(0.40, 0), (0.62, 100000)], [(0.88, -2), (0.50, 100000)]
    clean = [(0.62, 100000)], [(0.50, 100000)]
    for p in (0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95):
        rp = _eval(_book(*phantom), p)
        rc = _eval(_book(*clean), p)
        assert _same_result(rp, rc), (p, rp, rc)


def test_real_trade_still_executes_at_real_price():
    # A genuine edge against the first LIVE ask still trades, at that price.
    r = _eval(_book([(0.40, 0), (0.50, 100000)], [(0.48, 100000)], fee=0.0), 0.70)
    assert r.action == ak.TradeAction.BUY_YES, r
    assert abs(r.exec_price - 0.50) < 1e-9, r.exec_price
    # And a genuine NO edge against the first LIVE bid still trades.
    r = _eval(_book([(0.52, 100000)], [(0.88, 0), (0.50, 100000)], fee=0.0), 0.30)
    assert r.action == ak.TradeAction.BUY_NO, r
    assert abs(r.exec_price - 0.50) < 1e-9, r.exec_price


# ---- 3. All-zero-size book SKIPs as empty ------------------------------------

def test_all_zero_book_skips_as_empty():
    for p in (0.05, 0.50, 0.95):
        r = _eval(_book([(0.40, 0), (0.62, -5)], [(0.88, 0)]), p)
        assert r.action == ak.TradeAction.SKIP, (p, r)
        assert r.skip_reason.startswith("EMPTY_BOOK"), (p, r.skip_reason)
        assert r.invested_capital == 0.0 and r.shares_acquired == 0.0
        assert r.exec_price == 0.0 and r.kelly_fraction_raw == 0.0
        assert r.allocated_fraction == 0.0


def test_empty_and_negative_only_books_skip_as_empty():
    for asks, bids in (([], []), ([(0.40, -1)], [(0.88, -1)]), ([], [(0.50, 0)])):
        r = _eval(_book(asks, bids), 0.65)
        assert r.action == ak.TradeAction.SKIP and r.skip_reason.startswith("EMPTY_BOOK"), r


def test_one_live_side_still_tradable_other_side_not_phantom():
    # SUPERSEDED by SF-BL-10760 (Gate 0.7 ONE_SIDED_BOOK): SF-BL-9398's
    # "one live side remains tradable" rule let the live side's trade
    # execute with the dead side's sentinel as its liquidation bid
    # (asks-only BUY_YES $15,000 marked at -$14,703). A book with a live
    # ladder on only one side now SKIPs as ONE_SIDED_BOOK for every p —
    # entry and exit each require a live ladder on both sides. The
    # phantom-price half of the original test stands: no edge is ever
    # manufactured from the dead side (it cannot trade at all now).
    r = _eval(_book([(0.40, 0)], [(0.50, 100000)], fee=0.0), 0.30)
    assert r.action == ak.TradeAction.SKIP and r.skip_reason.startswith("ONE_SIDED_BOOK"), r
    r = _eval(_book([(0.40, 0)], [(0.50, 100000)], fee=0.0), 0.95)
    assert r.action == ak.TradeAction.SKIP and r.skip_reason.startswith("ONE_SIDED_BOOK"), r
    r = _eval(_book([(0.50, 100000)], [(0.88, 0)], fee=0.0), 0.70)
    assert r.action == ak.TradeAction.SKIP and r.skip_reason.startswith("ONE_SIDED_BOOK"), r


# ---- 4. sweep_order_book ------------------------------------------------------

def test_sweep_zero_rung_does_not_block_real_fill():
    b = _book([(0.90, 0), (0.50, 10000)], [(0.48, 10000)], fee=0.01)
    inv, sh, q = ak.sweep_order_book(b, ak.TradeAction.BUY_YES, 500.0, 0.70)
    assert inv == 500.0 and abs(q - 0.505) < 1e-9 and sh > 0, (inv, sh, q)


def test_sweep_zero_rung_creates_no_fill_and_matches_clean():
    phantom = _book([(0.40, 0), (0.50, 1000), (0.55, 0), (0.60, 1000)],
                    [(0.45, 0), (0.48, 1000)], fee=0.0)
    clean = _book([(0.50, 1000), (0.60, 1000)], [(0.48, 1000)], fee=0.0)
    for side, belief in ((ak.TradeAction.BUY_YES, 0.80), (ak.TradeAction.BUY_NO, 0.80)):
        assert ak.sweep_order_book(phantom, side, 800.0, belief) == \
            ak.sweep_order_book(clean, side, 800.0, belief), side
    only_zero = _book([(0.40, 0)], [(0.60, 0)], fee=0.0)
    assert ak.sweep_order_book(only_zero, ak.TradeAction.BUY_YES, 500.0, 0.90) == (0.0, 0.0, 0.0)
    assert ak.sweep_order_book(only_zero, ak.TradeAction.BUY_NO, 500.0, 0.90) == (0.0, 0.0, 0.0)


# ---- 5. No regression on live-only books --------------------------------------

def test_trade_control_unchanged():
    b = _book([(0.50, 1000)], [(0.48, 1000)], fee=0.0)
    r = _eval(b, 0.70)
    assert r.action == ak.TradeAction.BUY_YES, r
    assert abs(r.kelly_fraction_raw - 0.40) < 1e-9, r
    assert r.allocated_fraction == 0.05 and r.invested_capital == 500.0, r


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
