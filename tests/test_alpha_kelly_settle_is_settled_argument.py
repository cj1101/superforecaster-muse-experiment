"""
test_alpha_kelly_settle_is_settled_argument.py — SF-BL-10822 verification.

Pre-registered done criterion: "settle_or_mark_trade must raise a named
ValueError for any is_settled that is not a bool (fixtured at 'False',
'True', 0/1-as-int excluded or explicitly accepted per prereg, and
None), leaving net_pnl_bid_marked untouched, with genuine True/False
calls bit-identical; kill if a caller census shows is_settled is only
ever passed as a bool literal."

Kill NOT met (CALLER_CENSUS.md in the task work dir): the sole in-repo
production caller passes a variable unpacked from unvalidated scenario
dicts, not a bool literal; no schema/deserialisation layer exists; the
annotation is unenforced.

Prereg decision: strict bool only — int 1/0 are EXCLUDED (raise), not
accepted aliases (module discipline: SF-BL-10793/11536/9477).

Fixture (prereg): executed BUY_YES, $1,378.79 invested, 2730.28 shares,
liquidation bid 0.48, outcome_yes=1:
  genuine True  -> +1351.49  (2730.28*1.00 - 1378.79)
  genuine False -> -68.2556  (2730.28*0.48 - 1378.79)

History: SF-BL-11536's stored-field guard already raised on most paths
at assignment, but branch selection ran on truthiness first, so
'False' with no outcome raised SF-BL-9477's outcome error. This suite
pins the ENTRY gate (SF-BL-10822): the is_settled error wins over
outcome/bid/position errors, on SKIP/zero trades and exit-book marks
too, with the full trade snapshot untouched; the SF-BL-10971 trade-type
error still wins over it (that check dereferences nothing and is its
precondition).

Run: python3 test_alpha_kelly_settle_is_settled_argument.py
No pytest dependency.
"""
import contextlib
import io
import os
import re
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster"))
from superforecaster import alpha_kelly as ak  # noqa: E402

BAD_VALUES = ("False", "True", "false", "true", "0", "1", "",
              1, 0, 1.0, 0.0, -1, None, [], [False], {}, [True])


def _trade(action=ak.TradeAction.BUY_YES, invested=1378.79, shares=2730.28,
           bid=0.48, pnl=0.0, is_settled=False):
    return ak.TradeExecutionResult(
        market_id="M10822", title="Fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C1",
        action=action, calibrated_prob_yes=0.6, market_mid_prob=0.5,
        exec_price=0.505, liquidation_bid_price=bid,
        kl_divergence_nats=0.1, expected_log_growth=0.01,
        kelly_fraction_raw=0.1, allocated_fraction=0.05,
        invested_capital=invested, shares_acquired=shares,
        est_days_to_resolution=30.0, is_settled=is_settled,
        outcome_yes=None, net_pnl_bid_marked=pnl)


def _snap(t):
    return (t.is_settled, t.outcome_yes, t.net_pnl_bid_marked,
            t.liquidation_bid_price, t.liquidation_mark_vwap,
            t.liquidation_fill_fraction, t.liquidation_bid_as_of,
            t.liquidation_bid_source, t.cluster_exposure_released)


def _named_10822(fn):
    try:
        fn()
    except ValueError as e:
        return "INVALID_IS_SETTLED" in str(e) and "SF-BL-10822" in str(e)
    except Exception:  # noqa: BLE001
        return False
    return False


# ---- 1. Prereg fixture -------------------------------------------------------

def test_prereg_genuine_controls_exact():
    win = ak.settle_or_mark_trade(_trade(), is_settled=True, outcome_yes=1)
    assert win.is_settled is True and abs(win.net_pnl_bid_marked - 1351.49) < 1e-9, win.net_pnl_bid_marked
    mark = ak.settle_or_mark_trade(_trade(), is_settled=False)
    assert mark.is_settled is False and abs(mark.net_pnl_bid_marked - (-68.2556)) < 1e-9, mark.net_pnl_bid_marked


def test_prereg_string_false_with_outcome_raises_10822_untouched():
    t = _trade(pnl=-999.0)
    before = _snap(t)
    assert _named_10822(lambda: ak.settle_or_mark_trade(t, is_settled="False", outcome_yes=1))
    assert _snap(t) == before and t.net_pnl_bid_marked == -999.0


def test_prereg_string_false_without_outcome_raises_10822_not_9477():
    # The residual gap SF-BL-11536 documented: pre-10822 this raised
    # "Settled trade requires outcome_yes in {0, 1}." (SF-BL-9477).
    t = _trade(pnl=-999.0)
    before = _snap(t)
    assert _named_10822(lambda: ak.settle_or_mark_trade(t, is_settled="False"))
    assert _snap(t) == before


# ---- 2. Extended bad space ---------------------------------------------------

def test_extended_bad_space_raises_10822_untouched():
    for bad in BAD_VALUES:
        for kw in (dict(outcome_yes=1), dict()):
            t = _trade(pnl=-999.0)
            before = _snap(t)
            assert _named_10822(lambda bad=bad, kw=kw, t=t:
                                ak.settle_or_mark_trade(t, is_settled=bad, **kw)), (bad, kw)
            assert _snap(t) == before, (bad, kw)


def test_int_1_and_0_explicitly_excluded():
    # Prereg decision, documented in CALLER_CENSUS.md: not aliases.
    for bad in (1, 0, 1.0, 0.0, -1):
        t = _trade()
        assert _named_10822(lambda bad=bad, t=t:
                            ak.settle_or_mark_trade(t, is_settled=bad, outcome_yes=1)), bad


# ---- 3. Ordering: is_settled gate wins over branch validations ---------------

def test_bad_flag_wins_over_bad_outcome_and_bad_bid():
    t = _trade(bid=1.50)  # bid would raise SF-BL-9436 on the open branch
    assert _named_10822(lambda: ak.settle_or_mark_trade(t, is_settled=0))
    t2 = _trade()
    assert _named_10822(lambda: ak.settle_or_mark_trade(t2, is_settled="True", outcome_yes=None))
    t3 = _trade()
    assert _named_10822(lambda: ak.settle_or_mark_trade(t3, is_settled=None, outcome_yes=0.5))


def test_bad_flag_wins_over_bad_position_fields():
    # Entry ordering chosen this task: argument-type gate runs before
    # the SF-BL-9541 field-value gate (it dereferences nothing).
    t = _trade()
    t.shares_acquired = -100.0
    assert _named_10822(lambda: ak.settle_or_mark_trade(t, is_settled="False", outcome_yes=1))
    assert t.net_pnl_bid_marked == 0.0


def test_bad_trade_argument_still_wins_over_bad_flag():
    # SF-BL-10971 is the precondition: INVALID_TRADE, not 10822.
    try:
        ak.settle_or_mark_trade(None, is_settled="False", outcome_yes=1)
        raise AssertionError("no raise")
    except ValueError as e:
        assert "INVALID_TRADE" in str(e) and "SF-BL-10971" in str(e), e


def test_genuine_flag_bad_outcome_and_bid_keep_their_errors():
    t = _trade()
    try:
        ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=None)
        raise AssertionError("no raise")
    except ValueError as e:
        assert str(e) == "Settled trade requires outcome_yes in {0, 1}.", e
    t2 = _trade(bid=1.50)
    try:
        ak.settle_or_mark_trade(t2, is_settled=False)
        raise AssertionError("no raise")
    except ValueError as e:
        assert "SF-BL-9436" in str(e), e


# ---- 4. SKIP / zero-invested and exit-book paths ------------------------------

def test_skip_and_zero_trades_bad_flag_raises_genuine_unchanged():
    skip = _trade(action=ak.TradeAction.SKIP, invested=0.0, shares=0.0)
    assert _named_10822(lambda: ak.settle_or_mark_trade(skip, is_settled="False"))
    assert skip.net_pnl_bid_marked == 0.0
    zero = _trade(invested=0.0)
    assert _named_10822(lambda: ak.settle_or_mark_trade(zero, is_settled=None))
    # Genuine: early return preserved, incl. tolerated bool outcome (10724 pin).
    r = ak.settle_or_mark_trade(_trade(action=ak.TradeAction.SKIP, invested=0.0, shares=0.0),
                                is_settled=True, outcome_yes=True)
    assert r.net_pnl_bid_marked == 0.0 and r.is_settled is True


def _book():
    return ak.OrderBookSnapshot(
        market_id="B10822", title="fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C1",
        est_days_to_resolution=5.0,
        yes_asks=[ak.OrderBookLevel(price=0.62, size_shares=100000)],
        yes_bids=[ak.OrderBookLevel(price=0.60, size_shares=100000)],
        fee_rate=0.0)


def test_exit_book_bad_flag_raises_untouched_genuine_mark_works():
    t = _trade(invested=100.0, shares=200.0, bid=0.60, pnl=-999.0)
    before = _snap(t)
    assert _named_10822(lambda: ak.settle_or_mark_trade(t, is_settled="False", exit_book=_book()))
    assert _snap(t) == before
    ok = ak.settle_or_mark_trade(_trade(invested=100.0, shares=200.0, bid=0.60),
                                 is_settled=False, exit_book=_book())
    assert ok.is_settled is False and abs(ok.net_pnl_bid_marked - 20.0) < 1e-9, ok.net_pnl_bid_marked


def test_bad_flag_settlement_releases_no_cluster_exposure():
    ledger = {"C1": 0.05}
    t = _trade()
    t.cluster_exposure_used = ledger
    assert _named_10822(lambda: ak.settle_or_mark_trade(
        t, is_settled=1, outcome_yes=1, cluster_exposure_used=ledger))
    assert ledger == {"C1": 0.05} and t.cluster_exposure_released is False


# ---- 5. Genuine calls bit-identical ------------------------------------------

def test_genuine_buy_no_and_lifecycle():
    no_win = ak.settle_or_mark_trade(
        _trade(action=ak.TradeAction.BUY_NO), is_settled=True, outcome_yes=0)
    assert abs(no_win.net_pnl_bid_marked - 1351.49) < 1e-9
    t = ak.settle_or_mark_trade(_trade(), is_settled=False)
    assert abs(t.net_pnl_bid_marked - (-68.2556)) < 1e-9
    t = ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    assert abs(t.net_pnl_bid_marked - 1351.49) < 1e-9
    # A failed bad-flag call does not poison the next valid call.
    assert _named_10822(lambda: ak.settle_or_mark_trade(t, is_settled="False"))
    t = ak.settle_or_mark_trade(t, is_settled=False)
    assert abs(t.net_pnl_bid_marked - (-68.2556)) < 1e-9


def test_benchmark_reproduces():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    scores = [float(x) for x in re.findall(r'"north_star_score": (-?[\d.]+)', buf.getvalue())]
    assert len(scores) == 2, scores
    assert abs(scores[0] - (-0.1611)) < 5e-5 and abs(scores[1] - (-0.1499)) < 5e-5, scores


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in fns:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failures}/{len(fns)} PASS")
    sys.exit(1 if failures else 0)
