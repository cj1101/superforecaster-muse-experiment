"""
test_alpha_kelly_negative_invested_scorer.py — SF-BL-11304 verification.

Pre-registered done criterion: "evaluate_fleet_north_star must raise a
named ValueError on any non-SKIP trade with invested_capital <= 0 that
carries nonzero PnL (or account it explicitly), so
executed+skipped == total_opportunities holds on every fixture and the
negative-invested repro raises instead of vanishing, with all-positive
books bit-identical; kill if a producer census proves
invested_capital <= 0 cannot reach the scorer (construction-time
validation on TradeExecutionResult plus mutation-proofing)."

Pre-fix reproduction (verified live 2026-10-23, repro.py in the task
work dir; two BUY_YES trades, resolved [(0.6,1),(0.6,0)]):
  GOOD invested 100 / PnL +10, BAD invested -100 / PnL +50
  -> RETURNED total_opportunities 2, executed_trades 1,
     skipped_opportunities 0, total_net_pnl 10.0 (identity broken,
     +$50 silently excluded). Zero-invested/PnL+50 vanished identically.

Kill condition is NOT met (see PRODUCER_AUDIT.md in the task work dir):
TradeExecutionResult validates only `action` at construction/assignment
(SF-BL-10823) — direct construction with invested_capital=-100 succeeds,
the dataclass is mutable (post-construction and post-settlement mutation
and object.__setattr__ all reach the scorer), and settle_or_mark_trade,
while rejecting negative invested at its own entry (SF-BL-9541), never
writes the field and is not the sole writer. So the fix applies at the
scorer's consumption gate: _validate_scored_trade_invested raises
ValueError (NON_POSITIVE_INVESTED_CAPITAL, SF-BL-11304) for any non-SKIP
trade with finite invested_capital <= 0.
Ordering: after SF-BL-9475 finiteness (non-finite invested keeps its
NON_FINITE_TRADE_FIELD identity) and after SF-BL-9364 lockups (a trade
tripping both keeps its NON_POSITIVE_EST_DAYS identity — pinned by the
SF-BL-9364 suite's invested=0.0/est_days=-5 fixture), before the
executed/skipped partition. SKIP trades exempt.

ROUND-2 SCOPE (tester Gap 1): round 1 raised only when the <= 0 trade
carried NONZERO PnL, so a <= 0 trade with PnL exactly 0.0 still
returned and still vanished from the count (tester probes:
[invested=-100, pnl=0.0] alone -> total 1, executed 0, skipped 0;
[invested=0.0, pnl=0.0] alone -> same; [good 100/+10, -100/0.0] ->
total 2, executed 1, skipped 0). The prereg's raise condition is
literally nonzero-PnL, but its stated consequence — "so
executed+skipped == total_opportunities holds on EVERY fixture" — is
the binding requirement, and the narrow scope falsifies it on that
returning class. Round 2 therefore raises on ALL non-SKIP
invested_capital <= 0 regardless of PnL (a superset of the prereg's
literal cases; all-positive books unaffected). The alternative
(account the trade as skipped) was rejected: skipped_opportunities
counts genuine SKIP decisions, and a BUY_YES/BUY_NO record is not one.

Run:  python3 test_alpha_kelly_negative_invested_scorer.py
No pytest dependency: test_* functions with assert + __main__ runner.
"""
import io
import contextlib
import json
import math
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster/superforecaster"))
import alpha_kelly as ak  # noqa: E402

PASS = []
RESOLVED = [(0.6, 1), (0.6, 0)]
NAN = float("nan")
INF = float("inf")


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _trade(**over):
    base = dict(
        market_id="FIXTURE-11304", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C11304",
        action=ak.TradeAction.BUY_YES,
        calibrated_prob_yes=0.60, market_mid_prob=0.50, exec_price=0.50,
        liquidation_bid_price=0.50, kl_divergence_nats=0.0,
        expected_log_growth=0.0, kelly_fraction_raw=0.10,
        allocated_fraction=0.05, invested_capital=100.0,
        shares_acquired=1000.0, est_days_to_resolution=10.0,
        is_settled=True, outcome_yes=1, net_pnl_bid_marked=10.0)
    base.update(over)
    return ak.TradeExecutionResult(**base)


def _raises(trades, resolved=RESOLVED):
    try:
        ak.evaluate_fleet_north_star("fixture", trades, resolved)
    except ValueError as e:
        return str(e)
    return None


# ---- 1. Prereg headline case -------------------------------------------------

def test_negative_invested_repro_raises_not_vanishes():
    good = _trade(market_id="GOOD", invested_capital=100.0, net_pnl_bid_marked=10.0)
    bad = _trade(market_id="BAD-NEG", invested_capital=-100.0, net_pnl_bid_marked=50.0)
    msg = _raises([good, bad])
    assert msg is not None, "negative-invested book RETURNED a scorecard (pre-fix: total 2, executed 1, skipped 0, PnL 10.0)"
    assert "SF-BL-11304" in msg and "NON_POSITIVE_INVESTED_CAPITAL" in msg, msg
    assert "BAD-NEG" in msg and "-100" in msg and "50" in msg, msg
    assert _raises([bad, good]) is not None  # either order


def test_zero_invested_nonzero_pnl_raises():
    for pnl in (50.0, -50.0, 0.01, -0.01):
        msg = _raises([_trade(invested_capital=0.0, net_pnl_bid_marked=pnl)])
        assert msg is not None and "SF-BL-11304" in msg, (pnl, msg)


def test_negative_invested_any_nonzero_pnl_raises_both_sides():
    for invested in (-100.0, -100, -0.5, -1e-9):
        for pnl in (50.0, -50.0):
            for action in (ak.TradeAction.BUY_YES, ak.TradeAction.BUY_NO):
                msg = _raises([_trade(invested_capital=invested,
                                      net_pnl_bid_marked=pnl, action=action)])
                assert msg is not None and "SF-BL-11304" in msg, (invested, pnl, action, msg)


def test_mutated_and_bypassed_trades_raise():
    t = _trade()
    t.invested_capital = -100.0
    t.net_pnl_bid_marked = 50.0
    assert _raises([t]) is not None
    t2 = _trade()
    object.__setattr__(t2, "invested_capital", -100.0)
    assert _raises([t2]) is not None
    t3 = _trade()
    ak.settle_or_mark_trade(t3, is_settled=True, outcome_yes=1)
    t3.invested_capital = -100.0  # post-settlement mutation; PnL +500 intact
    assert _raises([t3]) is not None


# ---- 2. Scope: what does NOT raise --------------------------------------------

def test_zero_pnl_nonpositive_invested_raises():
    # Round-2 scope (tester Gap 1): PnL exactly 0.0 does NOT exempt a
    # non-SKIP <= 0 trade — pre-round-2 it RETURNED and vanished from
    # the count. The three tester probes, verbatim:
    msg = _raises([_trade(invested_capital=-100.0, net_pnl_bid_marked=0.0)])
    assert msg is not None and "SF-BL-11304" in msg, msg
    assert "NON_POSITIVE_INVESTED_CAPITAL" in msg, msg
    msg = _raises([_trade(invested_capital=0.0, net_pnl_bid_marked=0.0)])
    assert msg is not None and "SF-BL-11304" in msg, msg
    good = _trade(market_id="GOOD", invested_capital=100.0, net_pnl_bid_marked=10.0)
    bad0 = _trade(market_id="BAD-ZERO-PNL", invested_capital=-100.0,
                  net_pnl_bid_marked=0.0)
    msg = _raises([good, bad0])
    assert msg is not None and "BAD-ZERO-PNL" in msg, msg
    assert _raises([bad0, good]) is not None  # either order
    # Both actions, int zero, and -0.0 PnL behave identically.
    for invested in (0, -100, -0.5):
        for pnl in (0.0, -0.0, 0):
            for action in (ak.TradeAction.BUY_YES, ak.TradeAction.BUY_NO):
                msg = _raises([_trade(invested_capital=invested,
                                      net_pnl_bid_marked=pnl, action=action)])
                assert msg is not None and "SF-BL-11304" in msg, (invested, pnl, action, msg)


def test_skip_trades_exempt():
    # SKIP records are audit data and contribute nothing (SF-BL-9475/9364 scope).
    skip = _trade(action=ak.TradeAction.SKIP, invested_capital=-100.0,
                  net_pnl_bid_marked=50.0, skip_reason="FIXTURE")
    sc = ak.evaluate_fleet_north_star("fixture", [skip], RESOLVED)
    assert sc.executed_trades == 0 and sc.skipped_opportunities == 1, sc
    assert sc.north_star_score == 0.0 and sc.total_net_pnl == 0.0, sc
    skip0 = _trade(action=ak.TradeAction.SKIP, invested_capital=0.0,
                   net_pnl_bid_marked=0.0, skip_reason="FIXTURE")
    sc2 = ak.evaluate_fleet_north_star("fixture", [skip0], RESOLVED)
    assert sc2.skipped_opportunities == 1, sc2


def test_nonfinite_invested_keeps_sf_bl_9475_identity():
    for bad in (NAN, INF, -INF, "100", None, True):
        msg = _raises([_trade(invested_capital=bad, net_pnl_bid_marked=50.0)])
        assert msg is not None and "SF-BL-9475" in msg, (bad, msg)
        assert "NON_FINITE_TRADE_FIELD" in msg, msg


def test_lockup_violation_keeps_sf_bl_9364_identity():
    # A trade tripping both gates keeps the earlier gate's identity
    # (mirrors SF-BL-9364's own invested=0.0 / est_days=-5 fixture).
    msg = _raises([_trade(invested_capital=0.0, net_pnl_bid_marked=100.0,
                          est_days_to_resolution=-5.0)])
    assert msg is not None and "SF-BL-9364" in msg, msg
    # Round-2 re-check: the broadened gate must not steal the lockup
    # identity for the zero-PnL variant either — SF-BL-9364 runs first.
    msg = _raises([_trade(invested_capital=0.0, net_pnl_bid_marked=0.0,
                          est_days_to_resolution=-5.0)])
    assert msg is not None and "SF-BL-9364" in msg, msg
    assert "SF-BL-11304" not in msg, msg
    msg = _raises([_trade(invested_capital=-100.0, net_pnl_bid_marked=0.0,
                          est_days_to_resolution=-5.0)])
    assert msg is not None and "SF-BL-9364" in msg, msg


def test_helper_scope():
    ak._validate_scored_trade_invested([_trade()])  # positive: no raise
    ak._validate_scored_trade_invested(
        [_trade(action=ak.TradeAction.SKIP, invested_capital=-1.0,
                net_pnl_bid_marked=5.0)])  # SKIP: no raise
    ak._validate_scored_trade_invested(
        [_trade(action=ak.TradeAction.SKIP, invested_capital=0.0,
                net_pnl_bid_marked=0.0)])  # SKIP zero/zero: no raise
    # Non-SKIP <= 0 raises with nonzero AND with exactly-zero PnL.
    for pnl in (5.0, 0.0):
        try:
            ak._validate_scored_trade_invested(
                [_trade(invested_capital=-1.0, net_pnl_bid_marked=pnl)])
        except ValueError as e:
            assert "SF-BL-11304" in str(e)
            continue
        raise AssertionError(f"helper accepted -1.0 invested with PnL {pnl}")


# ---- 3. Count identity + all-positive books bit-identical ----------------------

def test_count_identity_on_returning_books():
    books = [
        [_trade(market_id="A")],
        [_trade(market_id="A"), _trade(market_id="B", action=ak.TradeAction.BUY_NO)],
        [_trade(market_id="A"),
         _trade(market_id="S", action=ak.TradeAction.SKIP, invested_capital=0.0,
                net_pnl_bid_marked=0.0, skip_reason="FIXTURE")],
        [_trade(market_id="S", action=ak.TradeAction.SKIP, invested_capital=0.0,
                net_pnl_bid_marked=0.0, skip_reason="FIXTURE")],
        [],
    ]
    for book in books:
        sc = ak.evaluate_fleet_north_star("fixture", book, RESOLVED)
        assert sc.executed_trades + sc.skipped_opportunities == sc.total_opportunities, sc
        assert sc.total_opportunities == len(book), sc


def test_count_identity_universal_grid():
    # The prereg identity, checked as a universal: for EVERY fixture in
    # the grid the scorer either raises the named SF-BL-11304 error
    # (exactly when the book contains a non-SKIP invested <= 0 trade)
    # or returns a scorecard satisfying executed+skipped == total.
    # Round 1 failed this on the zero-PnL / non-positive class; the
    # grid includes that class explicitly (invested <= 0 x pnl == 0.0).
    singles = []
    for invested in (-100.0, -0.5, 0.0, 100.0):
        for pnl in (0.0, 10.0, -10.0):
            for action in (ak.TradeAction.BUY_YES, ak.TradeAction.BUY_NO,
                           ak.TradeAction.SKIP):
                kw = dict(invested_capital=invested, net_pnl_bid_marked=pnl,
                          action=action)
                if action == ak.TradeAction.SKIP:
                    kw["skip_reason"] = "FIXTURE"
                singles.append(_trade(**kw))
    good = _trade(market_id="GOOD")
    books = [[t] for t in singles]
    books += [[good, t] for t in singles]
    books += [[t, good] for t in singles]
    for book in books:
        has_bad = any(t.action != ak.TradeAction.SKIP
                      and t.invested_capital <= 0 for t in book)
        try:
            sc = ak.evaluate_fleet_north_star("fixture", book, RESOLVED)
        except ValueError as e:
            assert has_bad and "SF-BL-11304" in str(e), (book, e)
            continue
        assert not has_bad, book
        assert sc.executed_trades + sc.skipped_opportunities == sc.total_opportunities, sc
        assert sc.total_opportunities == len(book), sc


def test_control_fixture_exact():
    sc = ak.evaluate_fleet_north_star(
        "fixture", [_trade(invested_capital=500.0, net_pnl_bid_marked=100.0)], RESOLVED)
    assert sc.north_star_score == 0.4445, sc
    assert sc.cwr_bid_marked == 0.2 and sc.executed_trades == 1, sc
    assert sc.total_net_pnl == 100.0, sc


def test_positive_books_bit_identical_variants():
    # Positive invested with zero / negative PnL, int fields, tiny/huge.
    sc = ak.evaluate_fleet_north_star(
        "fixture",
        [_trade(market_id="Z", net_pnl_bid_marked=0.0),
         _trade(market_id="N", net_pnl_bid_marked=-50.0, invested_capital=500)],
        RESOLVED)
    assert math.isfinite(sc.north_star_score), sc
    assert sc.executed_trades == 2 and sc.total_net_pnl == -50.0, sc


def test_benchmark_scorecards_bit_identical():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    cards = []
    decoder = json.JSONDecoder()
    idx = 0
    while True:
        start = out.find('{\n  "fleet_name"', idx)
        if start < 0:
            break
        obj, end = decoder.raw_decode(out[start:])
        cards.append(obj)
        idx = start + end
    assert len(cards) == 2, f"expected 2 scorecards, got {len(cards)}"
    # Values on this tree at this run; the binding no-change evidence is
    # the full-output diff benchmark_prefix.txt vs benchmark_postfix.txt
    # in the task work dir: EMPTY.
    assert cards[0]["north_star_score"] == -0.1611, cards[0]
    assert cards[1]["north_star_score"] == -0.1499, cards[1]


def test_d3_guard_still_fires_for_positive_trades():
    msg = _raises([_trade()], resolved=[])
    assert msg is not None and "NO_ELIGIBLE_ROWS" in msg, msg


# ---- 4. Kill-condition evidence -------------------------------------------------

def test_kill_condition_fails_no_construction_or_mutation_guard():
    # Direct construction with negative invested succeeds (only action
    # is validated on the dataclass) — so <=0 CAN reach the scorer.
    t = _trade(invested_capital=-100.0, net_pnl_bid_marked=50.0)
    assert t.invested_capital == -100.0
    t2 = _trade()
    t2.invested_capital = -100.0  # plain mutation, no guard
    assert t2.invested_capital == -100.0
    # ...and the scorer gate, not a producer invariant, is what stops it.
    assert _raises([t]) is not None and _raises([t2]) is not None


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
