"""
test_alpha_kelly_north_star_finiteness.py — SF-BL-9475 verification tests.

Pre-registered done criterion: "Pass if any non-finite invested_capital,
net_pnl_bid_marked or est_days_to_resolution on an executed trade raises
an explicit error and the benchmark scorecards stay bit-identical; kill
if settle_or_mark_trade is proven to be the sole writer of those fields
and to guarantee finiteness."

Pre-fix reproduction (verified live 2026-10-21, before the patch; fixture:
one BUY_YES trade, $500 invested, 1000 shares, exec 0.50, PnL +100,
est_days 10, resolved [(0.6,1),(0.6,0)]; honest control N_star 0.4445):
  net_pnl=NaN   -> RETURNED north_star=nan ending=nan cwr=nan mdd_pct=0.0
  net_pnl=+inf  -> RETURNED north_star=inf ending=inf cwr=inf mdd_pct=0.0
  net_pnl=-inf  -> RETURNED north_star=-inf ending=-inf cwr=-inf mdd_pct=inf
  invested=NaN  -> RETURNED north_star=0.0 (trade silently dropped by >0)
  invested=+inf -> RETURNED cwr=0.0, avg_lockup=1.0 despite +100 PnL
  est_days=NaN  -> RETURNED north_star=1.4057 (D_lock silently clamped 1.0)
  est_days=+inf -> RETURNED avg_lockup=inf, north_star=0.0
  est_days=-inf -> RETURNED north_star=1.4057

Kill condition is NOT met (see PRODUCER_AUDIT.md): TradeExecutionResult
is a plain mutable dataclass with no __post_init__ validation, built
directly by the benchmark naive fleet and callers/tests;
settle_or_mark_trade computes PnL from unvalidated shares_acquired /
invested_capital (shares=NaN on a settled win yields PnL NaN) and never
writes est_days_to_resolution at all. So the fix applies: a finiteness
gate at the top of evaluate_fleet_north_star over every non-SKIP trade,
raising ValueError naming the market, field, value and SF-BL-9475.

Run:  python3 test_alpha_kelly_north_star_finiteness.py  (exit 0 iff pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_liquidation_bid.py.
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
NAN = float("nan")
INF = float("inf")
RESOLVED = [(0.6, 1), (0.6, 0)]
FIELDS = ("invested_capital", "net_pnl_bid_marked", "est_days_to_resolution")
BAD = (NAN, INF, -INF, "500", None, True)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _trade(**over):
    base = dict(
        market_id="FIXTURE-9475", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C9475",
        action=ak.TradeAction.BUY_YES,
        calibrated_prob_yes=0.60, market_mid_prob=0.50, exec_price=0.50,
        liquidation_bid_price=0.50, kl_divergence_nats=0.0,
        expected_log_growth=0.0, kelly_fraction_raw=0.10,
        allocated_fraction=0.05, invested_capital=500.0,
        shares_acquired=1000.0, est_days_to_resolution=10.0,
        is_settled=True, outcome_yes=1, net_pnl_bid_marked=100.0)
    base.update(over)
    return ak.TradeExecutionResult(**base)


def _raises(trades, resolved=RESOLVED):
    try:
        ak.evaluate_fleet_north_star("fixture", trades, resolved)
    except ValueError as e:
        return str(e)
    return None


# ---- 1. Prereg headline case -------------------------------------------------

def test_nan_pnl_raises_not_nan_scorecard():
    msg = _raises([_trade(net_pnl_bid_marked=NAN)])
    assert msg is not None, "NaN PnL returned a scorecard (pre-fix: N_star NaN, MDD 0.0)"


# ---- 2. Every field x every non-finite value raises ---------------------------

def test_all_fields_all_nonfinite_raise():
    for field_name in FIELDS:
        for bad in BAD:
            msg = _raises([_trade(**{field_name: bad})])
            assert msg is not None, f"{field_name}={bad!r} accepted"
            assert "SF-BL-9475" in msg and field_name in msg, msg


def test_error_message_names_market_field_value():
    msg = _raises([_trade(net_pnl_bid_marked=NAN)])
    assert "FIXTURE-9475" in msg and "net_pnl_bid_marked" in msg and "nan" in msg, msg


def test_nan_invested_raises_not_silently_dropped():
    # Pre-fix the invested>0 filter silently dropped this trade and
    # returned N_star 0.0 with executed_trades=0.
    assert _raises([_trade(invested_capital=NAN)]) is not None


def test_nonfinite_among_finite_trades_raises():
    good = _trade(market_id="GOOD")
    bad = _trade(market_id="BAD", est_days_to_resolution=INF)
    assert _raises([good, bad]) is not None
    assert _raises([bad, good]) is not None


def test_mutated_trade_also_raises():
    t = _trade()
    t.net_pnl_bid_marked = NAN
    assert _raises([t]) is not None
    t2 = _trade()
    t2.est_days_to_resolution = -INF
    assert _raises([t2]) is not None


def test_buy_no_nonfinite_raises_too():
    t = _trade(action=ak.TradeAction.BUY_NO, net_pnl_bid_marked=INF)
    assert _raises([t]) is not None


# ---- 3. Finite behaviour unchanged -------------------------------------------

def test_control_fixture_exact():
    sc = ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED)
    assert sc.north_star_score == 0.4445, sc
    assert sc.ending_account_value_bid_marked == 10100.0, sc
    assert sc.cwr_bid_marked == 0.2, sc
    assert sc.max_drawdown_pct == 0.0 and sc.avg_lockup_days == 10.0, sc
    assert sc.executed_trades == 1 and sc.total_net_pnl == 100.0, sc


def test_finite_edge_values_unchanged():
    # Zero PnL, negative PnL, int fields, tiny/huge finite magnitudes.
    sc = ak.evaluate_fleet_north_star(
        "fixture",
        [_trade(net_pnl_bid_marked=0.0),
         _trade(market_id="NEG", net_pnl_bid_marked=-500.0, invested_capital=500,
                est_days_to_resolution=10)],
        RESOLVED)
    assert math.isfinite(sc.north_star_score), sc
    big = ak.evaluate_fleet_north_star(
        "fixture", [_trade(invested_capital=1e12, net_pnl_bid_marked=-1e12,
                            est_days_to_resolution=3650.0)], RESOLVED)
    assert math.isfinite(big.north_star_score), big


def test_skip_trades_exempt_and_unchanged():
    # Prereg scope is executed trades; SKIP trades feed no sum and their
    # behaviour (counts only) is untouched, finite or not.
    skip = _trade(action=ak.TradeAction.SKIP, invested_capital=0.0,
                  net_pnl_bid_marked=0.0, skip_reason="FIXTURE")
    sc = ak.evaluate_fleet_north_star("fixture", [skip], RESOLVED)
    assert sc.executed_trades == 0 and sc.skipped_opportunities == 1, sc
    assert sc.north_star_score == 0.0, sc
    skip_nan = _trade(action=ak.TradeAction.SKIP, invested_capital=0.0,
                      net_pnl_bid_marked=NAN, skip_reason="FIXTURE")
    sc2 = ak.evaluate_fleet_north_star("fixture", [skip_nan], RESOLVED)
    assert sc2.north_star_score == 0.0 and sc2.executed_trades == 0, sc2


def test_empty_and_zero_executed_books_unchanged():
    sc = ak.evaluate_fleet_north_star("fixture", [], RESOLVED)
    assert sc.north_star_score == 0.0 and sc.executed_trades == 0, sc
    sc = ak.evaluate_fleet_north_star("fixture", [], [])
    assert sc.north_star_score == 0.0, sc


def test_d3_guard_still_fires_for_finite_trades():
    msg = _raises([_trade()], resolved=[])
    assert msg is not None and "NO_ELIGIBLE_ROWS" in msg, msg


# ---- 4. Benchmark scorecards bit-identical ------------------------------------

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
    assert cards[0]["north_star_score"] == 0.1059, cards[0]
    assert cards[1]["north_star_score"] == -0.1499, cards[1]
    assert cards[0]["ending_account_value_bid_marked"] == 12778.38, cards[0]
    assert cards[1]["ending_account_value_bid_marked"] == 9326.36, cards[1]
    assert cards[0]["cwr_bid_marked"] == 0.5414, cards[0]
    assert cards[1]["cwr_bid_marked"] == -0.0784, cards[1]


# ---- 5. Kill-condition evidence ------------------------------------------------

def test_settle_does_not_guarantee_finiteness_so_kill_fails():
    # UPDATED by SF-BL-9541 (2026-10-21): settle_or_mark_trade now DOES
    # validate shares_acquired / invested_capital at entry, so the
    # original evidence route below (shares=NaN settles to PnL NaN) now
    # raises at settle instead. The kill STILL fails, via the routes
    # SF-BL-9541 does not close: settle never writes
    # est_days_to_resolution, and net_pnl_bid_marked can be written
    # directly at construction or mutated AFTER settlement — so the
    # scorer-side finiteness gate remains necessary.
    t = _trade(net_pnl_bid_marked=0.0)
    t.shares_acquired = NAN
    try:
        ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    except ValueError:
        pass  # SF-BL-9541: rejected at settle entry, PnL untouched
    else:
        raise AssertionError("SF-BL-9541 regression: NaN shares settled")
    assert t.net_pnl_bid_marked == 0.0, t.net_pnl_bid_marked
    # Directly constructed (never settled) non-finite PnL: scorer rejects.
    assert _raises([_trade(net_pnl_bid_marked=NAN)]) is not None
    # Post-settlement mutation to non-finite PnL: scorer rejects.
    t2 = _trade()
    ak.settle_or_mark_trade(t2, is_settled=True, outcome_yes=1)
    t2.net_pnl_bid_marked = NAN
    assert _raises([t2]) is not None


def test_helper_scope():
    assert ak._SCORED_TRADE_FIELDS == FIELDS
    ak._validate_scored_trade_fields([_trade()])  # finite: no raise
    try:
        ak._validate_scored_trade_fields([_trade(invested_capital=INF)])
    except ValueError:
        return
    raise AssertionError("helper accepted +inf invested_capital")


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
