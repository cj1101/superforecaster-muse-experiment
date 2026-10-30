"""
test_alpha_kelly_levered_book.py — SF-BL-34866 verification tests.

Pre-registered done criterion: "Success if levered books (summed
invested above starting capital) are rejected with a named error or
reported with an explicit leverage flag and a drawdown bounded to
[0, 100%] on a fixture grid including the 160% repro; kill if a fleet
consumer is shown to pass only capital-bounded books under a tested
invariant."

This task takes the FLAG branch (the prereg's second disjunct):
  * NorthStarScorecard gains is_levered / leverage_ratio /
    executed_invested_capital (defaulted fields, SF-BL-10767
    precedent), computed by _assess_book_leverage over the executed
    set (action != SKIP and invested_capital > 0) against
    starting_capital, BEFORE the SF-BL-10767 stale-mark partition;
  * the equity walk's drawdown goes through _bounded_drawdown, so
    max_drawdown_pct lies in [0, 100] and drawdown_penalty in
    [0, 1.7] on every returned scorecard.

Why not the rejection branch (documented in CALLER_CENSUS.md and in
the alpha_kelly.py SF-BL-34866 header): summed invested above
starting capital is not a corrupt value — sequential turnover sums
above it in normal operation, and two ratified prereg suites fixture
such books as legitimate scoreable edges (SF-BL-9437: a $500 trade
against starting_capital $0.50; SF-BL-9475: invested $1e12 against
the $10,000 default, asserting a finite N_star). Both fixtures are
re-scored here with their original values, now flagged. The kill
condition is NOT met either: no producer guarantees a
capital-bounded book (census in the task work dir).

Pre-fix reproduction (verified live 2026-10-27, repro.py in the task
work dir; two settled BUY_YES trades, $8,000 invested / -$8,000 PnL
each, resolved [(0.6,1),(0.6,0)], honest unlevered control one
$500 / +$100 trade -> N_star 0.4445):
  repro -> RETURNED ending -6000.0, MDD 160.0%, penalty 2.9,
           N_star -5.1226, CWR -1.0, no leverage flag.
Post-fix the same book returns ending -6000.0 (PnL arithmetic is not
floored), MDD 100.0%, penalty 1.7, N_star -3.9226, is_levered True,
leverage_ratio 1.6, executed_invested_capital 16000.0 — numerically
the same score as the exact-cap full-loss book, as it should be:
identical CWR, identical bounded penalty.

Benchmark: pre/post diff (benchmark_prefix.txt /
benchmark_postfix.txt in the task work dir) is EXACTLY the three new
fields on each scorecard — disciplined is_levered False / ratio
0.082913 / invested 829.12, naive False / 0.858892 / 8588.92 — with
every pre-existing numeric bit-identical (disciplined N_star
-0.1611 / ending 9170.88 / MDD 8.29; naive -0.1499 / 9326.36 /
MDD 21.76 / penalty 0.1352).

Run:  python3 test_alpha_kelly_levered_book.py  (exit 0 iff pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_starting_capital.py.
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
CAP = 10000.0


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _trade(mid="FIXTURE-34866", invested=500.0, pnl=100.0, **over):
    base = dict(
        market_id=mid, title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C34866",
        action=ak.TradeAction.BUY_YES,
        calibrated_prob_yes=0.60, market_mid_prob=0.50, exec_price=0.50,
        liquidation_bid_price=0.50, kl_divergence_nats=0.0,
        expected_log_growth=0.0, kelly_fraction_raw=0.10,
        allocated_fraction=0.05, invested_capital=invested,
        shares_acquired=1000.0, est_days_to_resolution=10.0,
        is_settled=True, outcome_yes=1, net_pnl_bid_marked=pnl)
    base.update(over)
    return ak.TradeExecutionResult(**base)


def _book(specs):
    return [_trade(mid=f"T{i}", invested=inv, pnl=pnl)
            for i, (inv, pnl) in enumerate(specs)]


REPRO = [(8000.0, -8000.0), (8000.0, -8000.0)]


def _score(trades, cap=CAP, resolved=RESOLVED, **kw):
    return ak.evaluate_fleet_north_star("fixture", trades, resolved,
                                        starting_capital=cap, **kw)


def _bypass_fix():
    """Return (restore_fn) after setting both fix helpers to pre-fix."""
    orig_assess = ak._assess_book_leverage
    orig_dd = ak._bounded_drawdown
    ak._assess_book_leverage = lambda t, c: (0.0, 0.0, False)
    ak._bounded_drawdown = lambda peak, eq: (
        (peak - eq) / peak if peak > 0 else 0.0)

    def restore():
        ak._assess_book_leverage = orig_assess
        ak._bounded_drawdown = orig_dd
    return restore


# ---- 1. Prereg 160% repro ----------------------------------------------------

def test_prereg_repro_flagged_and_bounded():
    sc = _score(_book(REPRO))
    assert sc.is_levered is True, sc
    assert sc.leverage_ratio == 1.6, sc
    assert sc.executed_invested_capital == 16000.0, sc
    assert sc.max_drawdown_pct == 100.0, sc          # bounded, was 160.0
    assert sc.drawdown_penalty == 1.7, sc            # bounded, was 2.9
    assert sc.north_star_score == -3.9226, sc        # was -5.1226
    assert sc.ending_account_value_bid_marked == -6000.0, sc  # not floored
    assert sc.cwr_bid_marked == -1.0, sc


def test_bypass_reproduces_160pct_unflagged_exactly():
    restore = _bypass_fix()
    try:
        sc = _score(_book(REPRO))
    finally:
        restore()
    assert sc.ending_account_value_bid_marked == -6000.0, sc
    assert sc.max_drawdown_pct == 160.0, sc
    assert sc.drawdown_penalty == 2.9, sc
    assert sc.north_star_score == -5.1226, sc
    assert sc.is_levered is False and sc.leverage_ratio == 0.0, sc
    # fix restored
    assert _score(_book(REPRO)).max_drawdown_pct == 100.0


# ---- 2. Fixture grid: flags and the [0, 100] bound ----------------------------

GRID = [
    # (specs, cap, is_levered, ratio, ending, mdd_pct, penalty)
    ([(500.0, 100.0)], CAP, False, 0.05, 10100.0, 0.0, 0.0),          # control
    ([(5000.0, -5000.0), (5000.0, -5000.0)], CAP, False, 1.0, 0.0, 100.0, 1.7),
    ([(5000.0, -100.0), (5000.01, -100.0)], CAP, True, 1.000001, 9800.0, 2.0, 0.0),
    ([(15000.0, 5000.0)], CAP, True, 1.5, 15000.0, 0.0, 0.0),         # levered WIN
    ([(15000.0, 0.0)], CAP, True, 1.5, 10000.0, 0.0, 0.0),            # levered flat
    ([(10001.0, -10001.0)], CAP, True, 1.0001, -1.0, 100.0, 1.7),
    ([(4000.0, -4000.0), (4000.0, -4000.0), (4000.0, 100.0)], CAP, True, 1.2, 2100.0, 80.0, 1.3),
    ([(2000.0, -500.0)] * 6, CAP, True, 1.2, 7000.0, 30.0, 0.3),
    ([(8000.0, -8000.0)], CAP, False, 0.8, 2000.0, 80.0, 1.3),
    ([(3000.0, -3000.0), (3000.0, 1500.0), (3000.0, -1500.0)], CAP, False, 0.9, 7000.0, 30.0, 0.3),
    ([(2500.0, -1250.0)] * 4, CAP, False, 1.0, 5000.0, 50.0, 0.7),
    (REPRO, 20000.0, False, 0.8, 4000.0, 80.0, 1.3),                  # same book, bigger base
    ([], CAP, False, 0.0, 10000.0, 0.0, 0.0),                         # empty book
]


def test_grid_flags_ratios_and_bounded_drawdown():
    for specs, cap, lev, ratio, ending, mdd, penalty in GRID:
        sc = _score(_book(specs), cap=cap)
        assert sc.is_levered is lev, (specs, sc)
        assert sc.leverage_ratio == ratio, (specs, sc)
        assert sc.executed_invested_capital == sum(i for i, _ in specs), (specs, sc)
        assert sc.ending_account_value_bid_marked == ending, (specs, sc)
        assert sc.max_drawdown_pct == mdd, (specs, sc)
        assert sc.drawdown_penalty == penalty, (specs, sc)
        assert 0.0 <= sc.max_drawdown_pct <= 100.0, (specs, sc)
        assert 0.0 <= sc.drawdown_penalty <= 1.7, (specs, sc)


def test_exact_cap_is_the_unlevered_boundary():
    at = _score(_book([(5000.0, -5000.0), (5000.0, -5000.0)]))
    assert at.is_levered is False and at.leverage_ratio == 1.0, at
    over = _score(_book([(5000.0, -100.0), (5000.01, -100.0)]))
    assert over.is_levered is True and over.leverage_ratio == 1.000001, over
    # The flag is the ONLY scorecard difference the one-cent step makes
    # to the risk readout: both books keep their own PnL arithmetic.
    assert over.max_drawdown_pct == 2.0 and at.max_drawdown_pct == 100.0


def test_deep_insolvency_still_bounded():
    # Equity crossing far below zero (levered, loss = stake per trade,
    # many trades): raw walk would read MDD in the thousands of %.
    specs = [(9000.0, -9000.0)] * 5   # 45000 deployed on 10000
    sc = _score(_book(specs))
    assert sc.is_levered is True and sc.leverage_ratio == 4.5, sc
    assert sc.max_drawdown_pct == 100.0 and sc.drawdown_penalty == 1.7, sc
    assert sc.ending_account_value_bid_marked == -35000.0, sc
    assert math.isfinite(sc.north_star_score), sc


# ---- 3. Ratified neighbour fixtures: scoreable, flagged -----------------------

def test_ratified_neighbour_fixtures_score_with_original_values():
    # SF-BL-9437 suite fixture: $500 / +$100 trade vs $0.50 capital.
    tiny = _score([_trade()], cap=0.5)
    assert tiny.ending_account_value_bid_marked == 100.5, tiny
    assert tiny.portfolio_return_pct == 20000.0, tiny
    assert tiny.is_levered is True and tiny.leverage_ratio == 1000.0, tiny
    # SF-BL-9475 suite fixture: $1e12 invested / -$1e12 PnL vs $10,000.
    big = _score([_trade(invested=1e12, pnl=-1e12,
                         est_days_to_resolution=3650.0)])
    assert math.isfinite(big.north_star_score), big
    assert big.is_levered is True and big.leverage_ratio == 100000000.0, big
    assert big.max_drawdown_pct == 100.0, big   # raw walk: ~1e10 %


# ---- 4. Scope: SKIP records, stale partition, precedence ----------------------

def test_skip_trades_deploy_no_capital():
    skip = _trade(mid="S", invested=20000.0, pnl=0.0,
                  action=ak.TradeAction.SKIP)
    sc = _score([skip, _trade()])
    assert sc.is_levered is False and sc.executed_invested_capital == 500.0, sc
    assert sc.north_star_score == 0.4445, sc  # identical to control alone
    sc2 = _score([skip])
    assert sc2.is_levered is False and sc2.leverage_ratio == 0.0, sc2
    assert sc2.executed_trades == 0 and sc2.north_star_score == 0.0, sc2


def test_stale_mark_exclusion_does_not_unlever_the_book():
    # Two UNSETTLED levered trades with no bid provenance: when
    # scored_at is given, SF-BL-10767 EXCLUDES both as stale from the
    # sums — but leverage is assessed on the submitted executed book
    # BEFORE that partition, so the flag still names the regime.
    open_trades = [
        _trade(mid="O1", invested=8000.0, pnl=-8000.0, is_settled=False,
               outcome_yes=None, liquidation_bid_as_of=None),
        _trade(mid="O2", invested=8000.0, pnl=-8000.0, is_settled=False,
               outcome_yes=None, liquidation_bid_as_of=None),
    ]
    sc = _score(open_trades, scored_at=2_000_000_000.0)
    assert sc.stale_mark_count == 2, sc
    assert sc.is_levered is True and sc.leverage_ratio == 1.6, sc
    assert sc.executed_invested_capital == 16000.0, sc
    assert 0.0 <= sc.max_drawdown_pct <= 100.0, sc


def test_neighbour_gates_keep_their_error_identities():
    # Corrupt books still raise the neighbours' named errors; the
    # leverage assessment introduces no new rejection.
    def raises(trades, cap=CAP):
        try:
            _score(trades, cap=cap)
        except ValueError as e:
            return str(e)
        return None
    msg = raises(_book(REPRO), cap=float("nan"))
    assert msg is not None and "INVALID_STARTING_CAPITAL" in msg, msg
    bad = _book(REPRO)
    bad[0].__dict__["invested_capital"] = float("nan")
    msg = raises(bad)
    assert msg is not None and "NON_FINITE_TRADE_FIELD" in msg, msg
    msg = raises(_book([(8000.0, -8000.0), (-100.0, 0.0)]))
    assert msg is not None and "NON_POSITIVE_INVESTED_CAPITAL" in msg, msg


def test_mixed_actions_and_containers():
    buy_no = _trade(mid="N", invested=8000.0, pnl=-8000.0,
                    action=ak.TradeAction.BUY_NO)
    sc = _score([buy_no, _trade(mid="Y", invested=8000.0, pnl=-8000.0)])
    assert sc.is_levered is True and sc.executed_invested_capital == 16000.0, sc
    sc = _score(tuple(_book(REPRO)))   # tuple container (SF-BL-10968)
    assert sc.is_levered is True and sc.max_drawdown_pct == 100.0, sc
    sc = _score(_book([(8000, -8000), (8000, -8000)]), cap=10000)  # ints
    assert sc.is_levered is True and sc.leverage_ratio == 1.6, sc


# ---- 5. No-op for books whose raw drawdown stayed in [0, 100%] ----------------

def test_in_range_books_bit_identical_except_new_fields():
    in_range = [(specs, cap) for specs, cap, *_ in GRID
                if max(0.0, -sum(p for _, p in specs)) <= cap]
    live = [_score(_book(specs), cap=cap) for specs, cap in in_range]
    restore = _bypass_fix()
    try:
        bypassed = [_score(_book(specs), cap=cap) for specs, cap in in_range]
    finally:
        restore()
    for (specs, cap), sc_live, sc_bypassed in zip(in_range, live, bypassed):
        a, b = asdict(sc_live), asdict(sc_bypassed)
        for f in ("is_levered", "leverage_ratio", "executed_invested_capital"):
            a.pop(f), b.pop(f)
        assert a == b, specs


def test_bounded_drawdown_helper():
    f = ak._bounded_drawdown
    assert f(10000.0, 10000.0) == 0.0
    assert f(10000.0, 2000.0) == 0.8
    assert f(10000.0, 0.0) == 1.0
    assert f(10000.0, -6000.0) == 1.0     # the repro step, bounded
    assert f(0.0, -5.0) == 0.0            # legacy peak>0 guard preserved
    assert f(11000.0, 10500.0) == (11000.0 - 10500.0) / 11000.0


def test_assess_book_leverage_helper():
    total, ratio, lev = ak._assess_book_leverage(_book(REPRO), CAP)
    assert (total, ratio, lev) == (16000.0, 1.6, True)
    total, ratio, lev = ak._assess_book_leverage(_book([(500.0, 100.0)]), CAP)
    assert (total, ratio, lev) == (500.0, 0.05, False)
    total, ratio, lev = ak._assess_book_leverage([], CAP)
    assert (total, ratio, lev) == (0, 0.0, False)
    # inert for a non-numeric capital (unreachable live: SF-BL-9437
    # gates first) — a bypassed-capital repro keeps its historic error
    total, ratio, lev = ak._assess_book_leverage(_book(REPRO), "10000")
    assert lev is False and ratio == 0.0


def test_scorecard_defaults_keep_direct_construction_working():
    sc = ak.NorthStarScorecard(
        fleet_name="x", starting_capital=10000.0,
        ending_account_value_bid_marked=10000.0, total_net_pnl=0.0,
        portfolio_return_pct=0.0, cwr_bid_marked=0.0, avg_lockup_days=30.0,
        velocity_multiplier=1.0, annualized_yield_apy_pct=0.0,
        ece_10bin=0.0, calibration_multiplier=1.0, brier_score=0.0,
        max_drawdown_pct=0.0, drawdown_penalty=0.0, sharpe_ratio_per_trade=0.0,
        total_opportunities=0, executed_trades=0, skipped_opportunities=0,
        settlement_win_rate_pct=0.0, north_star_score=0.0)
    assert sc.is_levered is False and sc.leverage_ratio == 0.0, sc
    assert sc.executed_invested_capital == 0.0, sc


def test_benchmark_delta_is_exactly_the_new_fields():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    for needle in ('"north_star_score": -0.1611', '"ending_account_value_bid_marked": 9170.88',
                   '"max_drawdown_pct": 8.29', '"north_star_score": -0.1499',
                   '"ending_account_value_bid_marked": 9326.36', '"max_drawdown_pct": 21.76',
                   '"drawdown_penalty": 0.1352',
                   '"is_levered": false', '"leverage_ratio": 0.082913',
                   '"executed_invested_capital": 829.12', '"leverage_ratio": 0.858892',
                   '"executed_invested_capital": 8588.92'):
        assert needle in out, needle


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    for name, fn in fns:
        try:
            fn()
            check(name, True)
        except Exception as e:  # noqa: BLE001
            check(name, False, f"{type(e).__name__}: {e}")
    print(f"\n{sum(PASS)}/{len(PASS)} PASS")
    sys.exit(0 if all(PASS) and PASS else 1)
