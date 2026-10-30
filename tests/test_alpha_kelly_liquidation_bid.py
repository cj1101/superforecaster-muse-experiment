"""
test_alpha_kelly_liquidation_bid.py — SF-BL-9436 verification tests.

Pre-registered done criterion: "Validate liquidation_bid_price in [0,1]
(raise or clamp with a logged flag) in settle_or_mark_trade, fixtured at
-0.30, 1.50 and the 0/1 boundaries with settled-path behaviour unchanged;
kill if liquidation bids are proven to only ever come from
OrderBookSnapshot best-bid properties bounded to (0,1) (then assert the
bound at that property instead)."

Pre-fix reproduction (verified live 2026-10-21, before the patch):
  fixture trade: 1000 shares, $500 invested, exec 0.50, BUY_YES, open:
    bid= 1.50 -> PnL +1000.0  (liquidation value $1,500 = 2x binary max)
    bid=-0.30 -> PnL  -800.0  (loss exceeds the $500 stake)
    bid= 0.00 -> PnL  -500.0  (boundary, correct)
    bid= 1.00 -> PnL  +500.0  (boundary, correct)
    bid= 0.50 -> PnL     0.0  (correct)

Kill condition is NOT met (see PRODUCER_AUDIT.md): OrderBookLevel has no
price validation; best_yes_bid returns the raw level price (a -0.30 level
yields -0.30) and best_no_bid = 1 - best_yes_ask yields -0.50 for a 1.50
ask. TradeExecutionResult is also constructed directly (benchmark naive
fleet) and is mutable after construction. So the fix, not the kill,
applies — validation in settle_or_mark_trade (raise, mirroring the
SF-BL-9363 fee_rate pattern and this function's settled-path ValueError
for an invalid outcome_yes).

Run:  python3 test_alpha_kelly_liquidation_bid.py   (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_fee_rate.py.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "..", "..", "superforecaster", "superforecaster"))
# Fallback for repo-tests/ placement:
if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "test_alpha_kelly_liquidation_bid.py")):
    pass
try:
    import alpha_kelly as ak  # noqa: E402
except ImportError:
    sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster/superforecaster"))
    import alpha_kelly as ak  # noqa: E402

PASS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _trade(bid, action=None, invested=500.0, shares=1000.0):
    """Prereg fixture: $500 stake, 1000 shares at exec 0.50."""
    return ak.TradeExecutionResult(
        market_id="FIXTURE", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C9436",
        action=action or ak.TradeAction.BUY_YES,
        calibrated_prob_yes=0.60, market_mid_prob=0.50, exec_price=0.50,
        liquidation_bid_price=bid, kl_divergence_nats=0.0,
        expected_log_growth=0.0, kelly_fraction_raw=0.10,
        allocated_fraction=0.05, invested_capital=invested,
        shares_acquired=shares, est_days_to_resolution=10.0)


def _raises_value_error(bid, **kw):
    try:
        ak.settle_or_mark_trade(_trade(bid, **kw), is_settled=False)
    except ValueError:
        return True
    return False


# ---- 1. Prereg fixtures: out-of-bound bids now raise ------------------------

def test_bid_150_raises_not_plus_1000():
    assert _raises_value_error(1.50), "bid 1.50 was accepted (pre-fix PnL +1000)"


def test_bid_minus_030_raises_not_minus_800():
    assert _raises_value_error(-0.30), "bid -0.30 was accepted (pre-fix PnL -800)"


def test_all_out_of_bound_bids_raise():
    for bad in (-1.0, -0.30, -0.001, -1e-9, 1.000001, 1.50, 2.0, 100.0):
        assert _raises_value_error(bad), f"bid={bad} accepted"


def test_nonfinite_and_nonnumeric_bids_raise():
    for bad in (float("nan"), float("inf"), float("-inf"), "0.50", None, True, False):
        assert _raises_value_error(bad), f"bid={bad!r} accepted"


def test_mutated_bid_also_raises():
    # A trade built validly then mutated (the bypass the fee fix also guards).
    t = _trade(0.50)
    t.liquidation_bid_price = 1.50
    try:
        ak.settle_or_mark_trade(t, is_settled=False)
    except ValueError:
        return
    raise AssertionError("mutated bid 1.50 accepted")


def test_error_message_names_bound_and_task():
    try:
        ak.settle_or_mark_trade(_trade(1.50), is_settled=False)
    except ValueError as e:
        assert "SF-BL-9436" in str(e) and "1.5" in str(e), str(e)
        return
    raise AssertionError("no ValueError")


# ---- 2. Boundaries and in-bound bids: exact pre-fix values ------------------

def test_boundary_bid_0_exact():
    t = ak.settle_or_mark_trade(_trade(0.0), is_settled=False)
    assert t.net_pnl_bid_marked == -500.0, t.net_pnl_bid_marked  # lose full stake, no more


def test_boundary_bid_1_exact():
    t = ak.settle_or_mark_trade(_trade(1.0), is_settled=False)
    assert t.net_pnl_bid_marked == 500.0, t.net_pnl_bid_marked  # shares*1 - stake = max gain


def test_inbound_bids_unchanged():
    for bid, pnl in ((0.5, 0.0), (0.25, -250.0), (0.75, 250.0), (0.01, -490.0), (0.99, 490.0)):
        t = ak.settle_or_mark_trade(_trade(bid), is_settled=False)
        assert t.net_pnl_bid_marked == pnl, (bid, t.net_pnl_bid_marked)
    # int bids are valid reals:
    assert ak.settle_or_mark_trade(_trade(0), is_settled=False).net_pnl_bid_marked == -500.0
    assert ak.settle_or_mark_trade(_trade(1), is_settled=False).net_pnl_bid_marked == 500.0


def test_buy_no_open_mark_uses_same_bound():
    assert _raises_value_error(1.50, action=ak.TradeAction.BUY_NO)
    assert _raises_value_error(-0.30, action=ak.TradeAction.BUY_NO)
    t = ak.settle_or_mark_trade(_trade(0.40, action=ak.TradeAction.BUY_NO), is_settled=False)
    assert t.net_pnl_bid_marked == -100.0, t.net_pnl_bid_marked


def test_pnl_never_exceeds_stake_bounds_for_valid_bids():
    # For every valid bid, PnL in [-invested, shares - invested].
    for i in range(0, 101):
        bid = i / 100.0
        t = ak.settle_or_mark_trade(_trade(bid), is_settled=False)
        assert -500.0 <= t.net_pnl_bid_marked <= 500.0, (bid, t.net_pnl_bid_marked)


# ---- 3. Settled path behaviour unchanged ------------------------------------

def test_settled_win_loss_unchanged():
    win = ak.settle_or_mark_trade(_trade(0.50), is_settled=True, outcome_yes=1)
    assert win.net_pnl_bid_marked == 500.0, win.net_pnl_bid_marked
    loss = ak.settle_or_mark_trade(_trade(0.50), is_settled=True, outcome_yes=0)
    assert loss.net_pnl_bid_marked == -500.0, loss.net_pnl_bid_marked
    no_win = ak.settle_or_mark_trade(_trade(0.50, action=ak.TradeAction.BUY_NO),
                                     is_settled=True, outcome_yes=0)
    assert no_win.net_pnl_bid_marked == 500.0


def test_settled_path_ignores_bid_entirely():
    # Settled PnL is shares*$1 or $0; the bid is never read on this path,
    # so even a stale out-of-bound bid must not change (or break) it —
    # "settled-path behaviour unchanged" per the prereg.
    for bid in (1.50, -0.30, float("nan")):
        t = ak.settle_or_mark_trade(_trade(bid), is_settled=True, outcome_yes=1)
        assert t.net_pnl_bid_marked == 500.0, (bid, t.net_pnl_bid_marked)


def test_settled_still_requires_valid_outcome():
    try:
        ak.settle_or_mark_trade(_trade(0.50), is_settled=True, outcome_yes=None)
    except ValueError:
        return
    raise AssertionError("settled without outcome_yes accepted")


# ---- 4. SKIP / zero-invested early return unchanged --------------------------

def test_skip_and_zero_invested_unchanged():
    skip = _trade(1.50, action=ak.TradeAction.SKIP, invested=0.0, shares=0.0)
    t = ak.settle_or_mark_trade(skip, is_settled=False)
    assert t.net_pnl_bid_marked == 0.0
    zero = _trade(1.50, invested=0.0, shares=0.0)
    t = ak.settle_or_mark_trade(zero, is_settled=False)
    assert t.net_pnl_bid_marked == 0.0


# ---- 5. Kill-condition evidence: source properties are NOT bounded ----------

def test_best_bid_properties_are_unbounded_so_kill_fails():
    # SF-BL-9542 SUPERSESSION (2026-10-21): when this kill-evidence test
    # was written, the best-price properties were raw passthrough (a -0.30
    # bid level yielded best_yes_bid -0.30). SF-BL-9542's prereg required
    # rung-price validation in those very properties, so they now skip
    # invalid-price rungs and fall back to the sentinels. The SF-BL-9436
    # kill still fails — settle/mark validation remains necessary —
    # because TradeExecutionResult is constructed and mutated directly
    # outside evaluate (see the tests above), not because the book
    # properties pass raw prices through any more.
    book = ak.OrderBookSnapshot(
        market_id="KILLCHECK", title="k", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C9436", est_days_to_resolution=5,
        yes_asks=[ak.OrderBookLevel(price=1.50, size_shares=100)],
        yes_bids=[ak.OrderBookLevel(price=-0.30, size_shares=100)], fee_rate=0.01)
    assert book.best_yes_bid == 0.01, book.best_yes_bid   # sentinel, not -0.30
    assert book.best_yes_ask == 0.99, book.best_yes_ask   # sentinel, not 1.50
    assert book.best_no_bid == 0.01, book.best_no_bid      # 1 - 0.99, not -0.50


def test_constants_exported():
    assert ak.LIQUIDATION_BID_MIN == 0.0 and ak.LIQUIDATION_BID_MAX == 1.0
    assert ak._liquidation_bid_in_bounds(0.0) and ak._liquidation_bid_in_bounds(1.0)
    assert not ak._liquidation_bid_in_bounds(1.50) and not ak._liquidation_bid_in_bounds(-0.30)
    assert math.isfinite(ak.LIQUIDATION_BID_MAX)


# ---- 6. End-to-end: N_star inputs can no longer be fabricated ---------------

def test_north_star_not_polluted_by_valid_marks():
    trades = [ak.settle_or_mark_trade(_trade(0.50), is_settled=False),
              ak.settle_or_mark_trade(_trade(1.0), is_settled=False)]
    sc = ak.evaluate_fleet_north_star("fixture", trades, [(0.6, 1), (0.6, 0)])
    assert sc.total_net_pnl == 500.0, sc
    assert sc.cwr_bid_marked == 0.5, sc


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
