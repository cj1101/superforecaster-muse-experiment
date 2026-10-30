"""
test_bid_vwap_mark.py — SF-BL-3660 verification tests.

Done criterion (TASK.md prereg): "Restated MDD on the 8-market fixture
plus forward books must move by >=0.5pp or >=5% of open positions must
mark >=1c/share lower under bid-VWAP; killed if bid-VWAP and top-bid
marks agree within 0.2c/share on >=95% of positions."

These tests pin the ENGINE change (the measurement itself is
measure_marks.py / RESULTS.md in this dir):
  settle_or_mark_trade(..., exit_book=book) marks an OPEN position at
  the bid-ladder VWAP liquidation mark for exactly shares_acquired —
  proceeds from sweeping the bid ladder (BUY_YES: yes_bids; BUY_NO:
  derived 1 - yes_ask ladder), unfilled remainder at $0 — instead of
  shares * single top bid. Without exit_book the legacy mark is
  byte-identical; the settled branch never reads the ladder.

Run:  python3 test_bid_vwap_mark.py   (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as the repo's other alpha_kelly test files.
"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster/superforecaster"))
import alpha_kelly as ak  # noqa: E402

PASS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def _book(bids, asks, fee=0.0):
    return ak.OrderBookSnapshot(
        market_id="T-3660", title="fixture", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="T", est_days_to_resolution=5.0,
        yes_asks=[ak.OrderBookLevel(price=p, size_shares=s) for p, s in asks],
        yes_bids=[ak.OrderBookLevel(price=p, size_shares=s) for p, s in bids],
        fee_rate=fee)


def _trade(action, bid, invested, shares):
    return ak.TradeExecutionResult(
        market_id="T-3660", title="fixture", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="T", action=action, calibrated_prob_yes=0.7,
        market_mid_prob=0.6, exec_price=0.5, liquidation_bid_price=bid,
        kl_divergence_nats=0.0, expected_log_growth=0.0, kelly_fraction_raw=0.0,
        allocated_fraction=0.0, invested_capital=invested, shares_acquired=shares,
        est_days_to_resolution=5.0)


def test_multi_rung_vwap():
    # bids 100 @ 0.60 then 1000 @ 0.50; 200 shares -> proceeds 60 + 50 = 110
    book = _book(bids=[(0.60, 100), (0.50, 1000)], asks=[(0.62, 1000)])
    mark, vwap, fill, proceeds = ak.bid_ladder_liquidation_mark(book, ak.TradeAction.BUY_YES, 200.0)
    check("multi-rung proceeds", approx(proceeds, 110.0), f"proceeds={proceeds}")
    check("multi-rung filled vwap", approx(vwap, 0.55), f"vwap={vwap}")
    check("multi-rung fill", approx(fill, 1.0), f"fill={fill}")
    check("multi-rung mark", approx(mark, 0.55), f"mark={mark}")
    t = ak.settle_or_mark_trade(_trade(ak.TradeAction.BUY_YES, 0.60, 100.0, 200.0),
                                is_settled=False, exit_book=book)
    check("multi-rung pnl = proceeds - invested", approx(t.net_pnl_bid_marked, 10.0),
          f"pnl={t.net_pnl_bid_marked}")
    check("multi-rung stored mark replaced", approx(t.liquidation_bid_price, 0.55))
    check("multi-rung audit fields", approx(t.liquidation_mark_vwap, 0.55)
          and approx(t.liquidation_fill_fraction, 1.0))
    check("multi-rung mark >=1c below top bid", (0.60 - t.liquidation_bid_price) >= 0.01)


def test_depth_exhaustion_lowers_mark():
    # only 100 shares bid at 0.60; position is 200 -> proceeds 60, mark 0.30
    book = _book(bids=[(0.60, 100)], asks=[(0.62, 1000)])
    mark, vwap, fill, proceeds = ak.bid_ladder_liquidation_mark(book, ak.TradeAction.BUY_YES, 200.0)
    check("exhaustion proceeds", approx(proceeds, 60.0), f"proceeds={proceeds}")
    check("exhaustion fill 0.5", approx(fill, 0.5), f"fill={fill}")
    check("exhaustion filled vwap still top bid", approx(vwap, 0.60))
    check("exhaustion effective mark 0.30", approx(mark, 0.30), f"mark={mark}")
    t = ak.settle_or_mark_trade(_trade(ak.TradeAction.BUY_YES, 0.60, 100.0, 200.0),
                                is_settled=False, exit_book=book)
    check("exhaustion pnl", approx(t.net_pnl_bid_marked, -40.0), f"pnl={t.net_pnl_bid_marked}")


def test_buy_no_derived_ladder():
    # BUY_NO exits on 1 - yes_ask: asks 500 @ 0.30 (-> NO bid 0.70), 500 @ 0.45 (-> 0.55)
    # 600 NO shares -> proceeds 500*0.70 + 100*0.55 = 405, mark 0.675
    book = _book(bids=[(0.28, 1000)], asks=[(0.30, 500), (0.45, 500)])
    t = ak.settle_or_mark_trade(_trade(ak.TradeAction.BUY_NO, 0.70, 300.0, 600.0),
                                is_settled=False, exit_book=book)
    check("buy_no mark", approx(t.liquidation_bid_price, 0.675), f"mark={t.liquidation_bid_price}")
    check("buy_no pnl", approx(t.net_pnl_bid_marked, 105.0), f"pnl={t.net_pnl_bid_marked}")


def test_no_exit_depth_marks_zero():
    book = _book(bids=[], asks=[(0.62, 1000)])
    t = ak.settle_or_mark_trade(_trade(ak.TradeAction.BUY_YES, 0.60, 100.0, 200.0),
                                is_settled=False, exit_book=book)
    check("no depth mark 0", approx(t.liquidation_bid_price, 0.0))
    check("no depth pnl = -invested", approx(t.net_pnl_bid_marked, -100.0))
    check("no depth fill 0", approx(t.liquidation_fill_fraction, 0.0))


def test_invalid_and_dead_levels_skipped():
    # zero-size top rung and invalid-price rung must not set the walked mark
    book = _book(bids=[(0.90, 0), (1.50, 500), (0.60, 100), (0.50, 1000)],
                 asks=[(0.62, 1000)])
    mark, vwap, fill, proceeds = ak.bid_ladder_liquidation_mark(book, ak.TradeAction.BUY_YES, 200.0)
    check("dead/invalid rungs skipped", approx(proceeds, 110.0) and approx(mark, 0.55),
          f"proceeds={proceeds} mark={mark}")


def test_legacy_path_unchanged_without_book():
    t = ak.settle_or_mark_trade(_trade(ak.TradeAction.BUY_YES, 0.60, 100.0, 200.0),
                                is_settled=False)
    check("legacy pnl = shares*topbid - invested", approx(t.net_pnl_bid_marked, 20.0),
          f"pnl={t.net_pnl_bid_marked}")
    check("legacy stored bid untouched", approx(t.liquidation_bid_price, 0.60))
    check("legacy audit fields stay None",
          t.liquidation_mark_vwap is None and t.liquidation_fill_fraction is None)


def test_settled_path_ignores_ladder():
    book = _book(bids=[(0.60, 10)], asks=[(0.62, 1000)])  # almost no exit depth
    won = ak.settle_or_mark_trade(_trade(ak.TradeAction.BUY_YES, 0.60, 100.0, 200.0),
                                  is_settled=True, outcome_yes=1, exit_book=book)
    lost = ak.settle_or_mark_trade(_trade(ak.TradeAction.BUY_YES, 0.60, 100.0, 200.0),
                                   is_settled=True, outcome_yes=0, exit_book=book)
    check("settled win pays $1/share regardless of depth", approx(won.net_pnl_bid_marked, 100.0),
          f"pnl={won.net_pnl_bid_marked}")
    check("settled loss pays $0 regardless of depth", approx(lost.net_pnl_bid_marked, -100.0))
    check("settled audit fields stay None", won.liquidation_mark_vwap is None)


def test_skip_and_zero_unchanged():
    book = _book(bids=[(0.60, 100)], asks=[(0.62, 1000)])
    t = _trade(ak.TradeAction.SKIP, 0.60, 0.0, 0.0)
    t = ak.settle_or_mark_trade(t, is_settled=False, exit_book=book)
    check("skip pnl 0", approx(t.net_pnl_bid_marked, 0.0))


def test_small_position_within_top_rung_matches_top_bid():
    # position fully inside the top rung: VWAP mark == top bid (no phantom change)
    book = _book(bids=[(0.60, 1000), (0.50, 1000)], asks=[(0.62, 1000)])
    t = ak.settle_or_mark_trade(_trade(ak.TradeAction.BUY_YES, 0.60, 100.0, 200.0),
                                is_settled=False, exit_book=book)
    check("within top rung mark == top bid", approx(t.liquidation_bid_price, 0.60))
    check("within top rung fill 1", approx(t.liquidation_fill_fraction, 1.0))


def test_mark_never_exceeds_top_bid_or_one():
    book = _book(bids=[(0.99, 50), (0.98, 50)], asks=[(0.995, 100)])
    mark, vwap, fill, proceeds = ak.bid_ladder_liquidation_mark(book, ak.TradeAction.BUY_YES, 500.0)
    check("mark in [0, top bid]", 0.0 <= mark <= 0.99, f"mark={mark}")
    check("mark validates in settle", ak.settle_or_mark_trade(
        _trade(ak.TradeAction.BUY_YES, 0.99, 100.0, 500.0),
        is_settled=False, exit_book=book).net_pnl_bid_marked <= 500.0 - 100.0 + 1e-9)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
    sys.exit(0 if all(PASS) and PASS else 1)
