"""
test_alpha_kelly_settle_atomicity.py — SF-BL-10724 verification tests.

Pre-registered done criterion: "Validation must complete before any
field of the trade is written, so a raising settle/mark leaves
is_settled, outcome_yes and net_pnl_bid_marked byte-identical to their
pre-call values on both branches; kill if a caller census shows no
caller ever re-settles or re-marks a trade object (single-pass use
only)."

Kill condition is NOT met (CALLER_CENSUS.md in the task work dir):
re-settlement / re-marking of one object is exercised, documented
lifecycle — SF-BL-8890's release is idempotent "however often it is
re-settled", the cluster-release suite marks a trade open then settles
the SAME object and settles another twice.

Pre-fix reproduction (repro.py / repro_prefix.log in the task work
dir; 800 shares / $500 invested fixture):
  A. settled win (+$300) re-marked at bid 1.50 -> raised SF-BL-9436
     ValueError, but returned is_settled=False / outcome_yes=None
     with the +$300 settled PnL intact.
  B. open trade (mark PnL -$100) settled with outcome_yes=None ->
     raised SF-BL-9477 ValueError, but returned is_settled=True with
     the stale -$100 mark PnL intact.
Post-fix (repro_postfix.log): every raising call leaves the triple
(is_settled, outcome_yes, net_pnl_bid_marked) byte-identical.

Run:  python3 test_alpha_kelly_settle_atomicity.py  (exit 0 iff all pass)
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


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _trade(action=None, shares=800.0, invested=500.0, bid=0.50):
    """Fixture: 800 shares, $500 invested (settled win = +$300.00,
    open mark at bid 0.50 = -$100.00, at bid 0.80 = +$140.00)."""
    return ak.TradeExecutionResult(
        market_id="FIXTURE-10724", title="SF-BL-10724 fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C10724",
        action=action or ak.TradeAction.BUY_YES,
        calibrated_prob_yes=0.60, market_mid_prob=0.50, exec_price=0.50,
        liquidation_bid_price=bid, kl_divergence_nats=0.0,
        expected_log_growth=0.0, kelly_fraction_raw=0.10,
        allocated_fraction=0.05, invested_capital=invested,
        shares_acquired=shares, est_days_to_resolution=10.0)


def _settled(**kw):
    t = _trade(**kw)
    return ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)


def _opened(**kw):
    t = _trade(**kw)
    return ak.settle_or_mark_trade(t, is_settled=False)


def _triple(t):
    return (t.is_settled, t.outcome_yes, t.net_pnl_bid_marked)


def _raises_untouched(t, **settle_kw):
    """True iff the call raises (any exception) and the prereg triple
    is byte-identical afterwards: same values AND same types."""
    before = _triple(t)
    try:
        ak.settle_or_mark_trade(t, **settle_kw)
    except Exception:  # noqa: BLE001 - atomicity is exception-agnostic
        after = _triple(t)
        return after == before and [type(x) for x in after] == [type(x) for x in before]
    return False


# ---- 1. Titled hazards ------------------------------------------------------

def test_titled_failed_remark_of_settled_trade_untouched():
    t = _settled()
    assert _triple(t) == (True, 1, 300.0), _triple(t)
    t.liquidation_bid_price = 1.50
    assert _raises_untouched(t, is_settled=False), _triple(t)
    assert _triple(t) == (True, 1, 300.0), _triple(t)


def test_titled_failed_settle_of_open_trade_untouched():
    t = _opened()
    assert _triple(t) == (False, None, -100.0), _triple(t)
    assert _raises_untouched(t, is_settled=True, outcome_yes=None), _triple(t)
    assert _triple(t) == (False, None, -100.0), _triple(t)


# ---- 2. Raising grids on both branches, from both pre-states ----------------

BAD_OUTCOMES = (None, True, False, 0.5, 2, -1, 1.0, 0.0, "1", "yes")
BAD_BIDS = (1.50, -0.30, float("nan"), float("inf"), float("-inf"),
            "0.50", None, True, 1.000001, -1e-9)


def test_bad_outcomes_raise_untouched_from_open_and_settled():
    for outcome in BAD_OUTCOMES:
        for make in (_opened, _settled):
            t = make()
            assert _raises_untouched(t, is_settled=True, outcome_yes=outcome), \
                (outcome, make.__name__, _triple(t))
        for action in (ak.TradeAction.BUY_YES, ak.TradeAction.BUY_NO):
            t = make(action=action)
            assert _raises_untouched(t, is_settled=True, outcome_yes=outcome)


def test_bad_bids_raise_untouched_from_open_and_settled():
    for bid in BAD_BIDS:
        for make in (_opened, _settled):
            t = make()
            t.liquidation_bid_price = bid
            before_bid = t.liquidation_bid_price
            assert _raises_untouched(t, is_settled=False), \
                (bid, make.__name__, _triple(t))
            assert t.liquidation_bid_price is before_bid or \
                t.liquidation_bid_price == before_bid


def test_bad_position_fields_raise_untouched_triple():
    # SF-BL-9541's gate already ran before mutation; SF-BL-10724 pins
    # the FULL triple (pre-fix it left PnL untouched but this suite
    # verifies is_settled/outcome_yes are not flipped either way).
    for mutate in (dict(shares_acquired=-100.0), dict(shares_acquired=float("inf")),
                   dict(invested_capital=float("nan")), dict(invested_capital=-5.0)):
        for kw in (dict(is_settled=True, outcome_yes=1), dict(is_settled=False)):
            t = _settled()
            for k, v in mutate.items():
                setattr(t, k, v)
            assert _raises_untouched(t, **kw), (mutate, kw, _triple(t))


# ---- 3. Exit-book (SF-BL-3660) path atomicity --------------------------------

def _book(bids, asks):
    return ak.OrderBookSnapshot(
        market_id="B-10724", title="fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C10724",
        est_days_to_resolution=5.0,
        yes_asks=[ak.OrderBookLevel(price=p, size_shares=s) for p, s in asks],
        yes_bids=[ak.OrderBookLevel(price=p, size_shares=s) for p, s in bids],
        fee_rate=0.0)


def test_exit_book_successful_mark_unchanged():
    book = _book(bids=[(0.60, 100), (0.50, 1000)], asks=[(0.62, 1000)])
    t = _trade(shares=200.0, invested=100.0, bid=0.60)
    t = ak.settle_or_mark_trade(t, is_settled=False, exit_book=book)
    # proceeds 110 - invested 100 (VWAP arithmetic carries float dust,
    # as in the SF-BL-3660 suite's approx comparisons)
    assert abs(t.net_pnl_bid_marked - 10.0) <= 1e-9, t.net_pnl_bid_marked
    assert abs(t.liquidation_bid_price - 0.55) <= 1e-9
    assert abs(t.liquidation_mark_vwap - 0.55) <= 1e-9
    assert t.liquidation_fill_fraction == 1.0 and t.is_settled is False


def test_exit_book_malformed_book_raises_untouched_all_mark_fields():
    book = _book(bids=[(0.60, 1000)], asks=[(0.62, 1000)])
    book.yes_bids = None  # mutated book: the ladder walk raises TypeError
    t = _settled()
    fields_before = (t.liquidation_bid_price, t.liquidation_mark_vwap,
                     t.liquidation_fill_fraction)
    assert _raises_untouched(t, is_settled=False, exit_book=book), _triple(t)
    assert (t.liquidation_bid_price, t.liquidation_mark_vwap,
            t.liquidation_fill_fraction) == fields_before


def test_exit_book_with_invalid_stored_bid_raises_untouched():
    # Stored bid is validated before the walk, as pre-fix.
    book = _book(bids=[(0.60, 1000)], asks=[(0.62, 1000)])
    t = _settled()
    t.liquidation_bid_price = 1.50
    assert _raises_untouched(t, is_settled=False, exit_book=book), _triple(t)


# ---- 4. Successful calls byte-identical --------------------------------------

def test_successful_settles_and_marks_exact():
    win = ak.settle_or_mark_trade(_trade(), is_settled=True, outcome_yes=1)
    assert _triple(win) == (True, 1, 300.0)
    loss = ak.settle_or_mark_trade(_trade(), is_settled=True, outcome_yes=0)
    assert _triple(loss) == (True, 0, -500.0)
    no_win = ak.settle_or_mark_trade(_trade(action=ak.TradeAction.BUY_NO),
                                     is_settled=True, outcome_yes=0)
    assert _triple(no_win) == (True, 0, 300.0)
    mark = ak.settle_or_mark_trade(_trade(bid=0.80), is_settled=False)
    assert _triple(mark) == (False, None, 140.0)
    mark0 = ak.settle_or_mark_trade(_trade(bid=0.0), is_settled=False)
    assert _triple(mark0) == (False, None, -500.0)
    mark1 = ak.settle_or_mark_trade(_trade(bid=1.0), is_settled=False)
    assert _triple(mark1) == (False, None, 300.0)


def test_lifecycle_mark_then_settle_and_double_settle_still_work():
    t = _opened()
    assert _triple(t) == (False, None, -100.0)
    t = ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    assert _triple(t) == (True, 1, 300.0)
    t = ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)  # re-settle
    assert _triple(t) == (True, 1, 300.0)
    # ...and a VALID re-mark of the settled trade still flips it open
    # with a freshly computed mark PnL (the legitimate re-mark use):
    t.liquidation_bid_price = 0.80
    t = ak.settle_or_mark_trade(t, is_settled=False)
    assert _triple(t) == (False, None, 140.0), _triple(t)


def test_failed_attempt_does_not_poison_the_next_valid_call():
    t = _opened()
    assert _raises_untouched(t, is_settled=True, outcome_yes=None)
    t = ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=0)
    assert _triple(t) == (True, 0, -500.0)
    t.liquidation_bid_price = 1.50
    assert _raises_untouched(t, is_settled=False)
    t.liquidation_bid_price = 0.80
    t = ak.settle_or_mark_trade(t, is_settled=False)
    assert _triple(t) == (False, None, 140.0)


# ---- 5. Preserved orderings: SKIP / zero-invested, cluster release ----------

def test_skip_and_zero_invested_ordering_preserved():
    # Pinned by SF-BL-9477/9436/9541: the early return precedes outcome
    # and bid validation — these calls do NOT raise, so by design they
    # write is_settled/outcome_yes and book PnL 0.0.
    skip = _trade(action=ak.TradeAction.SKIP)
    r = ak.settle_or_mark_trade(skip, is_settled=True, outcome_yes=True)
    assert r.net_pnl_bid_marked == 0.0 and r.is_settled is True
    zero = _trade(invested=0.0)
    r2 = ak.settle_or_mark_trade(zero, is_settled=True, outcome_yes=False)
    assert r2.net_pnl_bid_marked == 0.0
    zero2 = _trade(invested=0.0, bid=1.50)
    r3 = ak.settle_or_mark_trade(zero2, is_settled=False)
    assert r3.net_pnl_bid_marked == 0.0


def test_failed_settle_releases_nothing_then_success_releases_once():
    ledger = {"C10724": 0.05}
    t = _trade()
    t.cluster_exposure_used = ledger
    assert _raises_untouched(t, is_settled=True, outcome_yes=None)
    assert ledger == {"C10724": 0.05} and t.cluster_exposure_released is False
    t = ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    assert ledger == {"C10724": 0.0} and t.cluster_exposure_released is True
    assert _triple(t) == (True, 1, 300.0)


def test_failed_remark_keeps_settled_scorecard_accounting():
    # The downstream harm in the rationale: evaluate_fleet_north_star
    # partitions on is_settled and counts wins from the PnL. After a
    # failed re-mark the settled win must still score as one.
    t = _settled()
    t.liquidation_bid_price = 1.50
    assert _raises_untouched(t, is_settled=False)
    sc = ak.evaluate_fleet_north_star(
        fleet_name="SF-BL-10724", trades=[t],
        all_resolved_forecasts=[(0.60, 1)], starting_capital=10000.0)
    assert sc.settlement_win_rate_pct == 100.0, sc.settlement_win_rate_pct
    assert sc.total_net_pnl == 300.0, sc.total_net_pnl


# ---- 6. Benchmark: immediate pre-image scores reproduced ----------------------

def test_benchmark_reproduces_prefix_scores():
    # Measured on the IMMEDIATE pre-image (the tree minus only this
    # task's change) this run: [-0.1611, -0.1499] — identical post-fix
    # (benchmark trades are single-pass successful settles). The older
    # +0.1059 / 0.0547 disciplined pins from SF-BL-9541/9477 drifted
    # through OTHER fleet landings already present in the pre-image.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    scores = [float(x) for x in
              re.findall(r'"north_star_score": (-?[\d.]+)', buf.getvalue())]
    assert len(scores) == 2, scores
    assert abs(scores[0] - (-0.1611)) < 5e-5, scores
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
