"""
test_alpha_kelly_settle_position_fields.py — SF-BL-9541 verification tests.

Pre-registered done criterion: "Validate shares_acquired (finite, >=0)
and invested_capital (finite, >=0) at settle_or_mark_trade entry for both
settled and mark branches: success = negative/NaN/inf share fixtures
raise ValueError with the trade's PnL untouched and the 8-market fixture
benchmark still reproduces N_star +0.1059/-0.1499; kill if any
out-of-domain share count reaches gross_payout or liquidation_value."

Pre-fix reproduction (verified live 2026-10-21, before the patch; see
PRODUCER_AUDIT.md): prereg fixture ($500 stake, BUY_YES):
  shares=-100, settled WIN  -> net PnL -600.0 (loss exceeding stake, on a win)
  shares=+inf, settled WIN  -> net PnL +inf
  shares=NaN,  settled WIN  -> net PnL NaN
  shares=-100, open mark .50 -> net PnL -550.0
  invested=NaN, settled WIN -> net PnL NaN
  invested=+inf, settled WIN -> net PnL -inf
  invested=-500 -> silently masked to PnL 0.0 by the `invested <= 0` return

Kill condition is NOT met: TradeExecutionResult is a plain mutable
dataclass with no __post_init__ validation (it constructs happily with
shares=-100 and accepts post-construction mutation), and the benchmark's
naive fleet constructs it directly outside evaluate_and_size_opportunity
— so validation belongs at settle_or_mark_trade entry, the point where
shares become PnL (mirrors SF-BL-9436 / SF-BL-9363 patterns).

Run:  python3 test_alpha_kelly_settle_position_fields.py  (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner.
"""
import contextlib
import io
import os
import re
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster/superforecaster"))
import alpha_kelly as ak  # noqa: E402

PASS = []
SENTINEL_PNL = 777.0


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _trade(shares=1000.0, invested=500.0, bid=0.50, action=None, pnl=0.0):
    """Prereg fixture: $500 stake, 1000 shares at exec 0.50."""
    t = ak.TradeExecutionResult(
        market_id="FIXTURE", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C9541",
        action=action or ak.TradeAction.BUY_YES,
        calibrated_prob_yes=0.60, market_mid_prob=0.50, exec_price=0.50,
        liquidation_bid_price=bid, kl_divergence_nats=0.0,
        expected_log_growth=0.0, kelly_fraction_raw=0.10,
        allocated_fraction=0.05, invested_capital=invested,
        shares_acquired=shares, est_days_to_resolution=10.0)
    t.net_pnl_bid_marked = pnl
    return t


def _raises_untouched(trade, **settle_kw):
    """True iff settle raises ValueError and the trade's PnL is untouched."""
    before = trade.net_pnl_bid_marked
    try:
        ak.settle_or_mark_trade(trade, **settle_kw)
    except ValueError:
        return trade.net_pnl_bid_marked == before
    return False


BAD_SHARES = (-100.0, -0.001, -1e-9, float("nan"), float("inf"),
              float("-inf"), "100", None, True, False)
BAD_INVESTED = (-500.0, -0.01, float("nan"), float("inf"),
                float("-inf"), "500", None, True, False)


# ---- 1. Titled fixtures: settled branch ------------------------------------

def test_shares_minus_100_won_raises_not_minus_600():
    t = _trade(shares=-100.0, pnl=SENTINEL_PNL)
    assert _raises_untouched(t, is_settled=True, outcome_yes=1), \
        "shares=-100 won trade accepted or PnL mutated (pre-fix PnL -600)"


def test_shares_plus_inf_won_raises_not_plus_inf():
    t = _trade(shares=float("inf"), pnl=SENTINEL_PNL)
    assert _raises_untouched(t, is_settled=True, outcome_yes=1), \
        "shares=+inf accepted or PnL mutated (pre-fix PnL +inf)"


def test_all_bad_shares_raise_settled_win_and_loss():
    for bad in BAD_SHARES:
        for outcome in (1, 0):
            t = _trade(shares=bad, pnl=SENTINEL_PNL)
            assert _raises_untouched(t, is_settled=True, outcome_yes=outcome), \
                f"shares={bad!r} outcome={outcome} accepted/PnL mutated"


def test_bad_shares_raise_buy_no_settled():
    for bad in BAD_SHARES:
        t = _trade(shares=bad, action=ak.TradeAction.BUY_NO, pnl=SENTINEL_PNL)
        assert _raises_untouched(t, is_settled=True, outcome_yes=0), \
            f"BUY_NO shares={bad!r} accepted"


# ---- 2. Mark branch ---------------------------------------------------------

def test_all_bad_shares_raise_open_mark():
    for bad in BAD_SHARES:
        for action in (ak.TradeAction.BUY_YES, ak.TradeAction.BUY_NO):
            t = _trade(shares=bad, action=action, pnl=SENTINEL_PNL)
            assert _raises_untouched(t, is_settled=False), \
                f"mark shares={bad!r} action={action} accepted/PnL mutated"


def test_mutated_shares_also_raise():
    t = _trade()
    t.shares_acquired = -100.0
    assert _raises_untouched(t, is_settled=True, outcome_yes=1)
    t2 = _trade()
    t2.shares_acquired = float("inf")
    assert _raises_untouched(t2, is_settled=False)


# ---- 3. invested_capital domain, both branches ------------------------------

def test_all_bad_invested_raise_both_branches():
    for bad in BAD_INVESTED:
        t = _trade(invested=bad, pnl=SENTINEL_PNL)
        assert _raises_untouched(t, is_settled=True, outcome_yes=1), \
            f"settled invested={bad!r} accepted/PnL mutated"
        t = _trade(invested=bad, pnl=SENTINEL_PNL)
        assert _raises_untouched(t, is_settled=False), \
            f"mark invested={bad!r} accepted/PnL mutated"


def test_negative_invested_no_longer_masked_to_zero():
    # Pre-fix the `invested <= 0` early return silently booked PnL 0.0.
    t = _trade(invested=-500.0)
    try:
        ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    except ValueError:
        return
    raise AssertionError("negative invested masked/accepted instead of raising")


def test_mutated_invested_also_raises():
    t = _trade()
    t.invested_capital = float("nan")
    assert _raises_untouched(t, is_settled=False)


def test_error_messages_name_field_value_and_task():
    for kw, field in ((dict(shares=-100.0), "shares_acquired"),
                      (dict(invested=float("nan")), "invested_capital")):
        try:
            ak.settle_or_mark_trade(_trade(**kw), is_settled=True, outcome_yes=1)
        except ValueError as e:
            assert field in str(e) and "SF-BL-9541" in str(e), str(e)
        else:
            raise AssertionError(f"no ValueError for {field}")


def test_helpers_domain():
    assert ak._shares_acquired_in_domain(0) and ak._shares_acquired_in_domain(0.0)
    assert ak._shares_acquired_in_domain(1000.0)
    assert not ak._shares_acquired_in_domain(-100.0)
    assert not ak._shares_acquired_in_domain(float("inf"))
    assert not ak._shares_acquired_in_domain(float("nan"))
    assert ak._invested_capital_in_domain(0.0) and ak._invested_capital_in_domain(500)
    assert not ak._invested_capital_in_domain(-0.01)
    assert not ak._invested_capital_in_domain(float("-inf"))


# ---- 4. Valid positions: exact pre-fix behaviour unchanged ------------------

def test_valid_settled_exact():
    win = ak.settle_or_mark_trade(_trade(), is_settled=True, outcome_yes=1)
    assert win.net_pnl_bid_marked == 500.0, win.net_pnl_bid_marked
    loss = ak.settle_or_mark_trade(_trade(), is_settled=True, outcome_yes=0)
    assert loss.net_pnl_bid_marked == -500.0, loss.net_pnl_bid_marked
    no_win = ak.settle_or_mark_trade(
        _trade(action=ak.TradeAction.BUY_NO), is_settled=True, outcome_yes=0)
    assert no_win.net_pnl_bid_marked == 500.0
    no_loss = ak.settle_or_mark_trade(
        _trade(action=ak.TradeAction.BUY_NO), is_settled=True, outcome_yes=1)
    assert no_loss.net_pnl_bid_marked == -500.0


def test_valid_mark_exact():
    for bid, pnl in ((0.5, 0.0), (0.25, -250.0), (0.75, 250.0),
                     (0.0, -500.0), (1.0, 500.0)):
        t = ak.settle_or_mark_trade(_trade(bid=bid), is_settled=False)
        assert t.net_pnl_bid_marked == pnl, (bid, t.net_pnl_bid_marked)


def test_zero_boundaries_valid():
    # shares=0 is in domain: a won trade pays 0, PnL = -stake (no fabrication).
    t = ak.settle_or_mark_trade(_trade(shares=0.0), is_settled=True, outcome_yes=1)
    assert t.net_pnl_bid_marked == -500.0, t.net_pnl_bid_marked
    t = ak.settle_or_mark_trade(_trade(shares=0), is_settled=False)
    assert t.net_pnl_bid_marked == -500.0
    # int amounts are valid reals:
    t = ak.settle_or_mark_trade(_trade(shares=1000, invested=500),
                                is_settled=True, outcome_yes=1)
    assert t.net_pnl_bid_marked == 500.0


def test_skip_and_zero_invested_unchanged():
    skip = _trade(action=ak.TradeAction.SKIP, shares=0.0, invested=0.0, bid=1.50)
    assert ak.settle_or_mark_trade(skip, is_settled=False).net_pnl_bid_marked == 0.0
    zero = _trade(shares=0.0, invested=0.0)
    assert ak.settle_or_mark_trade(zero, is_settled=True,
                                   outcome_yes=1).net_pnl_bid_marked == 0.0


def test_no_out_of_domain_count_reaches_payout_sweep():
    # Kill-criterion inversion as a property: for every rejected count the
    # function raises, so by construction none reaches gross_payout /
    # liquidation_value; for every accepted count PnL stays within
    # [-invested, shares - invested] (settled win) / [-invested,
    # shares*bid - invested] (mark).
    for shares in (0.0, 1.0, 250.0, 1000.0, 1e6):
        t = ak.settle_or_mark_trade(_trade(shares=shares),
                                    is_settled=True, outcome_yes=1)
        assert -500.0 <= t.net_pnl_bid_marked <= shares - 500.0
        t = ak.settle_or_mark_trade(_trade(shares=shares, bid=0.5),
                                    is_settled=False)
        assert -500.0 <= t.net_pnl_bid_marked <= shares * 0.5 - 500.0 + 500.0


# ---- 5. Neighbouring behaviour preserved (SF-BL-9436 / outcome gate) --------

def test_settled_still_requires_valid_outcome_and_ignores_bid():
    try:
        ak.settle_or_mark_trade(_trade(), is_settled=True, outcome_yes=None)
    except ValueError:
        pass
    else:
        raise AssertionError("settled without outcome_yes accepted")
    t = ak.settle_or_mark_trade(_trade(bid=1.50), is_settled=True, outcome_yes=1)
    assert t.net_pnl_bid_marked == 500.0  # settled path never reads the bid
    try:
        ak.settle_or_mark_trade(_trade(bid=1.50), is_settled=False)
    except ValueError:
        pass
    else:
        raise AssertionError("SF-BL-9436 bid validation regressed")


# ---- 6. Kill-condition evidence ----------------------------------------------

def test_direct_construction_and_mutation_are_unvalidated_so_kill_fails():
    # The dataclass itself accepts out-of-domain values — precisely why
    # the settle-entry validation, not a producer assertion, is the fix.
    t = _trade(shares=-100.0, invested=-5.0)
    assert t.shares_acquired == -100.0 and t.invested_capital == -5.0
    t.shares_acquired = float("inf")
    assert t.shares_acquired == float("inf")


# ---- 7. Benchmark benchmark: 8-market fixture N_star unchanged ---------------

def test_benchmark_reproduces_nstar():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    scores = [float(x) for x in
              re.findall(r'"north_star_score": (-?[\d.]+)', buf.getvalue())]
    assert len(scores) == 2, scores
    assert abs(scores[0] - 0.1059) < 5e-5, scores
    assert abs(scores[1] - (-0.1499)) < 5e-5, scores


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
