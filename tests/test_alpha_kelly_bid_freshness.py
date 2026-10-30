"""
test_alpha_kelly_bid_freshness.py — SF-BL-10767 verification tests.

Pre-registered done criteria (TASK.md): "Add a bid as-of timestamp/source
to the mark path and pre-register a staleness rule (marks older than the
book's snapshot cadence are excluded or flagged in the scorecard); kill
if every production mark caller already refreshes the bid within one
cadence of scoring."

Kill condition is NOT met (see CALLER_CENSUS.md): the only production
mark caller (run_benchmark_simulation) passes the SAME entry book as
exit_book in the same iteration, the legacy no-exit-book settle path
refreshes nothing, and pre-fix neither OrderBookSnapshot nor
TradeExecutionResult could carry an observation time at all.

Pre-fix reproduction (verified live, repro_prefix.log): an open trade
whose entry bid 0.62 was observed 7 days before scoring booked
+$120.00 and scored CWR 0.24 / N_star 0.308, indistinguishable from a
fresh mark; the scorecard had no freshness fields and
evaluate_fleet_north_star had no scored_at parameter.

The fix (alpha_kelly.py, SF-BL-10767 block):
  * OrderBookSnapshot.snapshot_ts (epoch seconds, None = unknown),
    validated finite >= 0 at construction when present.
  * TradeExecutionResult.liquidation_bid_as_of / liquidation_bid_source,
    stamped by evaluate ("entry_book") and re-stamped by settle's
    exit-book path ("exit_book"); legacy marks and settlements leave
    the stamps untouched.
  * Cadence pre-registered at MARK_STALENESS_CADENCE_SECONDS = 2700.0
    (45 min, the longest documented 15-45 min trading cycle).
  * evaluate_fleet_north_star(..., scored_at=...): unsettled executed
    trades whose mark is unprovenanced or older than one cadence are
    excluded from all sums and flagged (stale_mark_count /
    stale_mark_invested); unprovenanced open marks are always counted
    (unprovenanced_mark_count). scored_at=None is byte-identical to
    pre-fix scoring. Settled trades are never excluded.

Run:  python3 test_alpha_kelly_bid_freshness.py   (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_liquidation_bid.py.
"""
import io
import contextlib
import math
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_HERE, "..", "..", "superforecaster"),
              os.path.expanduser(
                  "~/workspace/superforecaster/superforecaster")):
    if os.path.exists(os.path.join(_cand, "alpha_kelly.py")):
        sys.path.insert(0, _cand)
        break
import alpha_kelly as ak  # noqa: E402

PASS = []

T0 = 1_000_000.0
CAD = ak.MARK_STALENESS_CADENCE_SECONDS
RESOLVED = [(0.6, 1), (0.6, 0)]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(ts=None, bid=0.62, ask=0.64):
    return ak.OrderBookSnapshot(
        market_id="M1", title="fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C1",
        est_days_to_resolution=30.0,
        yes_asks=[ak.OrderBookLevel(ask, 100000.0)],
        yes_bids=[ak.OrderBookLevel(bid, 100000.0)],
        fee_rate=0.01, snapshot_ts=ts)


def _trade(bid=0.62, as_of="unset", source="unset", settled=False,
           invested=500.0, shares=1000.0):
    kw = {}
    if as_of != "unset":
        kw["liquidation_bid_as_of"] = as_of
    if source != "unset":
        kw["liquidation_bid_source"] = source
    t = ak.TradeExecutionResult(
        market_id="M1", title="fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C1",
        action=ak.TradeAction.BUY_YES, calibrated_prob_yes=0.7,
        market_mid_prob=0.63, exec_price=0.50, liquidation_bid_price=bid,
        kl_divergence_nats=0.1, expected_log_growth=0.01,
        kelly_fraction_raw=0.1, allocated_fraction=0.05,
        invested_capital=invested, shares_acquired=shares,
        est_days_to_resolution=30.0, **kw)
    t.is_settled = settled
    return t


def _open_marked(bid=0.62, **kw):
    """An open trade marked (legacy path) at the stored bid."""
    return ak.settle_or_mark_trade(_trade(bid=bid, **kw), is_settled=False)


# ---------------------------------------------------------------- books

def test_cadence_constant_preregistered():
    assert CAD == 2700.0, CAD


def test_book_snapshot_ts_default_none_and_valid_accepted():
    assert _book().snapshot_ts is None
    assert _book(ts=T0).snapshot_ts == T0
    assert _book(ts=0).snapshot_ts == 0


def test_book_snapshot_ts_invalid_rejected():
    for bad in (math.nan, math.inf, -math.inf, -1.0, True, "now"):
        try:
            _book(ts=bad)
        except ValueError:
            continue
        raise AssertionError(f"snapshot_ts={bad!r} accepted")


# ------------------------------------------------------- evaluate stamp

def test_evaluate_stamps_entry_provenance():
    book = _book(ts=T0, bid=0.60, ask=0.62)
    t = ak.evaluate_and_size_opportunity(
        book=book, calibrated_prob_yes=0.80, inter_agent_dispersion=0.05,
        portfolio_equity=10_000.0, cluster_exposure_used={})
    assert t.action == ak.TradeAction.BUY_YES, t.skip_reason
    assert t.liquidation_bid_as_of == T0, t.liquidation_bid_as_of
    assert t.liquidation_bid_source == ak.BID_SOURCE_ENTRY_BOOK


def test_evaluate_stamps_none_as_of_for_timestampless_book():
    book = _book(ts=None, bid=0.60, ask=0.62)
    t = ak.evaluate_and_size_opportunity(
        book=book, calibrated_prob_yes=0.80, inter_agent_dispersion=0.05,
        portfolio_equity=10_000.0, cluster_exposure_used={})
    assert t.action == ak.TradeAction.BUY_YES, t.skip_reason
    assert t.liquidation_bid_as_of is None
    assert t.liquidation_bid_source == ak.BID_SOURCE_ENTRY_BOOK


# ---------------------------------------------------------- settle path

def test_settle_legacy_mark_preserves_entry_stamp():
    t = _open_marked(as_of=T0, source=ak.BID_SOURCE_ENTRY_BOOK)
    assert t.net_pnl_bid_marked == 120.0, t.net_pnl_bid_marked
    assert t.liquidation_bid_as_of == T0
    assert t.liquidation_bid_source == ak.BID_SOURCE_ENTRY_BOOK


def test_settle_exit_book_restamps_provenance():
    t = _trade(as_of=T0, source=ak.BID_SOURCE_ENTRY_BOOK)
    exit_book = _book(ts=T0 + 900.0, bid=0.55, ask=0.57)
    t = ak.settle_or_mark_trade(t, is_settled=False, exit_book=exit_book)
    assert t.liquidation_bid_price == 0.55, t.liquidation_bid_price
    assert t.net_pnl_bid_marked == 50.0, t.net_pnl_bid_marked
    assert t.liquidation_bid_as_of == T0 + 900.0
    assert t.liquidation_bid_source == ak.BID_SOURCE_EXIT_BOOK


def test_settle_exit_book_without_ts_restamps_source_only():
    t = _trade(as_of=T0, source=ak.BID_SOURCE_ENTRY_BOOK)
    t = ak.settle_or_mark_trade(
        t, is_settled=False, exit_book=_book(ts=None, bid=0.55, ask=0.57))
    assert t.liquidation_bid_source == ak.BID_SOURCE_EXIT_BOOK
    assert t.liquidation_bid_as_of is None


def test_settle_settled_branch_leaves_stamps_untouched():
    t = _trade(as_of=T0, source=ak.BID_SOURCE_ENTRY_BOOK)
    t = ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    assert t.net_pnl_bid_marked == 500.0
    assert t.liquidation_bid_as_of == T0
    assert t.liquidation_bid_source == ak.BID_SOURCE_ENTRY_BOOK


# ------------------------------------------------------------- scoring

def test_scoring_legacy_mode_identical_and_flags_unprovenanced():
    t = _open_marked()  # no stamp at all: the pre-fix record shape
    sc = ak.evaluate_fleet_north_star("fixture", [t], RESOLVED)
    assert sc.total_net_pnl == 120.0, sc
    assert sc.cwr_bid_marked == 0.24, sc
    assert sc.north_star_score == 0.308, sc
    assert sc.stale_mark_count == 0, sc
    assert sc.stale_mark_invested == 0.0, sc
    assert sc.unprovenanced_mark_count == 1, sc


def test_scoring_fresh_mark_included():
    t = _open_marked(as_of=T0, source=ak.BID_SOURCE_ENTRY_BOOK)
    sc = ak.evaluate_fleet_north_star(
        "fixture", [t], RESOLVED, scored_at=T0 + 600.0)
    assert sc.total_net_pnl == 120.0, sc
    assert sc.cwr_bid_marked == 0.24, sc
    assert sc.north_star_score == 0.308, sc
    assert sc.stale_mark_count == 0, sc
    assert sc.unprovenanced_mark_count == 0, sc


def test_scoring_boundary_age_exactly_one_cadence_included():
    t = _open_marked(as_of=T0, source=ak.BID_SOURCE_ENTRY_BOOK)
    sc = ak.evaluate_fleet_north_star(
        "fixture", [t], RESOLVED, scored_at=T0 + CAD)
    assert sc.total_net_pnl == 120.0, sc
    assert sc.stale_mark_count == 0, sc


def test_scoring_stale_mark_excluded():
    for age in (CAD + 1.0, 7 * 24 * 3600.0):
        t = _open_marked(as_of=T0, source=ak.BID_SOURCE_ENTRY_BOOK)
        sc = ak.evaluate_fleet_north_star(
            "fixture", [t], RESOLVED, scored_at=T0 + age)
        assert sc.total_net_pnl == 0.0, (age, sc)
        assert sc.cwr_bid_marked == 0.0, (age, sc)
        assert sc.north_star_score == 0.0, (age, sc)
        assert sc.stale_mark_count == 1, (age, sc)
        assert sc.stale_mark_invested == 500.0, (age, sc)
        assert sc.unprovenanced_mark_count == 0, (age, sc)


def test_scoring_unprovenanced_mark_excluded_when_scored_at():
    t = _open_marked()  # as_of None
    sc = ak.evaluate_fleet_north_star(
        "fixture", [t], RESOLVED, scored_at=T0)
    assert sc.total_net_pnl == 0.0, sc
    assert sc.stale_mark_count == 1, sc
    assert sc.unprovenanced_mark_count == 1, sc


def test_scoring_nonfinite_as_of_reads_unprovenanced():
    t = _open_marked(as_of=T0, source=ak.BID_SOURCE_ENTRY_BOOK)
    t.liquidation_bid_as_of = math.nan  # mutated after construction
    sc = ak.evaluate_fleet_north_star(
        "fixture", [t], RESOLVED, scored_at=T0)
    assert sc.unprovenanced_mark_count == 1, sc
    assert sc.stale_mark_count == 1, sc
    assert sc.total_net_pnl == 0.0, sc


def test_scoring_settled_trade_never_excluded():
    t = _trade(bid=0.50)  # no provenance at all
    t = ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    assert t.net_pnl_bid_marked == 500.0
    sc = ak.evaluate_fleet_north_star(
        "fixture", [t], RESOLVED, scored_at=T0 + 365 * 24 * 3600.0)
    assert sc.total_net_pnl == 500.0, sc
    assert sc.stale_mark_count == 0, sc
    assert sc.unprovenanced_mark_count == 0, sc
    assert sc.settlement_win_rate_pct == 100.0, sc


def test_scoring_mixed_book_excludes_only_stale():
    settled = ak.settle_or_mark_trade(
        _trade(bid=0.50), is_settled=True, outcome_yes=1)      # +500
    fresh = _open_marked(as_of=T0, source=ak.BID_SOURCE_ENTRY_BOOK)
    stale = _open_marked(as_of=T0 - 10 * CAD,
                         source=ak.BID_SOURCE_ENTRY_BOOK)
    sc = ak.evaluate_fleet_north_star(
        "fixture", [settled, fresh, stale], RESOLVED, scored_at=T0 + 60.0)
    assert sc.total_net_pnl == 620.0, sc
    assert sc.stale_mark_count == 1, sc
    assert sc.stale_mark_invested == 500.0, sc
    assert sc.unprovenanced_mark_count == 0, sc


def test_scoring_stale_mark_removed_from_mdd_walk():
    fresh = _open_marked(bid=0.62, as_of=T0,
                         source=ak.BID_SOURCE_ENTRY_BOOK)     # +120
    stale = _open_marked(bid=0.10, as_of=T0 - 10 * CAD,
                         source=ak.BID_SOURCE_ENTRY_BOOK)     # -400
    legacy = ak.evaluate_fleet_north_star("fixture", [fresh, stale],
                                          RESOLVED)
    assert legacy.max_drawdown_pct == 3.95, legacy
    scored = ak.evaluate_fleet_north_star(
        "fixture", [fresh, stale], RESOLVED, scored_at=T0)
    assert scored.max_drawdown_pct == 0.0, scored
    assert scored.total_net_pnl == 120.0, scored


def test_scoring_skip_trades_not_counted():
    skip = _trade()
    skip.action = ak.TradeAction.SKIP
    skip.invested_capital = 0.0
    fresh = _open_marked(as_of=T0, source=ak.BID_SOURCE_ENTRY_BOOK)
    sc = ak.evaluate_fleet_north_star(
        "fixture", [skip, fresh], RESOLVED, scored_at=T0)
    assert sc.unprovenanced_mark_count == 0, sc
    assert sc.stale_mark_count == 0, sc


def test_scoring_invalid_scored_at_raises():
    t = _open_marked(as_of=T0, source=ak.BID_SOURCE_ENTRY_BOOK)
    for bad in (math.nan, math.inf, -math.inf, True, "now"):
        try:
            ak.evaluate_fleet_north_star("fixture", [t], RESOLVED,
                                         scored_at=bad)
        except ValueError:
            continue
        raise AssertionError(f"scored_at={bad!r} accepted")


def test_benchmark_reproduces_prefix_scores():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    scores = re.findall(r'"north_star_score": ([-\d.]+)', buf.getvalue())
    assert scores == ["-0.1611", "-0.1499"], scores


ALL_TESTS = [v for k, v in sorted(globals().items())
             if k.startswith("test_")]

if __name__ == "__main__":
    for fn in ALL_TESTS:
        try:
            fn()
            check(fn.__name__, True)
        except AssertionError as e:
            check(fn.__name__, False, str(e))
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
