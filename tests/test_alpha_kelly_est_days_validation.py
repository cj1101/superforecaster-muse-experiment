"""
test_alpha_kelly_est_days_validation.py — SF-BL-9364 verification tests.

Pre-registered done criterion: "snapshots/trades with
est_days_to_resolution <= 0 are rejected at construction or scoring
with an explicit error and the mixed fixture scores identically to
the clean book after the bad row is quarantined; kill if every
est_days producer is proven to derive it from a resolution timestamp
strictly after the evaluation timestamp."

Pre-fix reproduction (verified live, repro.py / reproduce_prefix.txt;
fixture $500 invested / +$100 PnL per trade, resolved [(0.6,1),(0.6,0)]):
  clean +40/+40  -> avg_lockup 40.0, velocity 3.021, N_star 0.2223
  mixed -30/+40  -> avg_lockup 5.0, velocity 8.544, N_star 0.6286 (2.83x)
  single -30 / 0 -> avg_lockup 1.0 (max-floor mask), velocity 19.105
  OrderBookSnapshot(est_days=-30 / 0) constructed with NO error

Kill condition is NOT met (see PRODUCER_AUDIT.md): no producer derives
the field from timestamps at all — benchmark fixtures and test helpers
pass bare literals, OrderBookSnapshot is mutable, TradeExecutionResult
has no __post_init__ and settle_or_mark_trade never writes the field.
So the fix applies at three surfaces, mirroring SF-BL-9363's fee_rate
defence in depth:
  * OrderBookSnapshot.__post_init__ raises ValueError (SF-BL-9364);
  * evaluate_and_size_opportunity Gate 0.1 SKIPs a mutated/unpickled
    book INVALID_EST_DAYS_TO_RESOLUTION, zero size, cluster untouched;
  * evaluate_fleet_north_star raises ValueError
    (NON_POSITIVE_EST_DAYS_TO_RESOLUTION, SF-BL-9364) for any non-SKIP
    trade with finite est_days <= 0, after SF-BL-9475's finiteness gate
    (non-finite values keep their SF-BL-9475 error identity).
Valid domain: finite real strictly > 0 (bool / non-numeric excluded);
no upper bound is preregistered, so none is imposed.

Run:  python3 test_alpha_kelly_est_days_validation.py  (exit 0 iff pass)
No pytest dependency: test_* functions with assert + __main__ runner.
"""
import io
import contextlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster/superforecaster"))
import alpha_kelly as ak  # noqa: E402

PASS = []
NAN = float("nan")
INF = float("inf")
RESOLVED = [(0.6, 1), (0.6, 0)]
BAD_LOCKUPS = (-30.0, -1, -0.5, 0, 0.0)
NONFINITE = (NAN, INF, -INF, "40", None, True, False)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(**over):
    base = dict(
        market_id="BOOK-9364", title="fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C9364",
        est_days_to_resolution=40.0,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000.0)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100000.0)],
        fee_rate=0.01)
    base.update(over)
    return ak.OrderBookSnapshot(**base)


def _trade(**over):
    base = dict(
        market_id="TRADE-9364", title="fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C9364",
        action=ak.TradeAction.BUY_YES, calibrated_prob_yes=0.60,
        market_mid_prob=0.50, exec_price=0.50, liquidation_bid_price=0.50,
        kl_divergence_nats=0.0, expected_log_growth=0.0,
        kelly_fraction_raw=0.10, allocated_fraction=0.05,
        invested_capital=500.0, shares_acquired=1000.0,
        est_days_to_resolution=40.0, is_settled=True, outcome_yes=1,
        net_pnl_bid_marked=100.0)
    base.update(over)
    return ak.TradeExecutionResult(**base)


def _score_raises(trades, resolved=RESOLVED):
    try:
        ak.evaluate_fleet_north_star("fixture", trades, resolved)
    except ValueError as e:
        return str(e)
    return None


def _construct_raises(**over):
    try:
        _book(**over)
    except ValueError as e:
        return str(e)
    return None


# ---- 1. Prereg headline: the mixed fixture ----------------------------------

def test_mixed_negative_book_raises_not_inflated():
    # Pre-fix this RETURNED avg_lockup 5.0 / velocity 8.544 / N_star 0.6286.
    msg = _score_raises([_trade(market_id="BAD", est_days_to_resolution=-30.0),
                         _trade(market_id="GOOD")])
    assert msg is not None, "mixed -30/+40 book returned a scorecard"
    assert "SF-BL-9364" in msg and "est_days_to_resolution" in msg, msg
    assert "BAD" in msg and "-30" in msg, msg


def test_quarantined_mixed_scores_identically_to_clean():
    clean = ak.evaluate_fleet_north_star(
        "fixture", [_trade(market_id="A"), _trade(market_id="B")], RESOLVED)
    assert clean.avg_lockup_days == 40.0 and clean.velocity_multiplier == 3.021, clean
    quarantined = ak.evaluate_fleet_north_star(
        "fixture", [_trade(market_id="B")], RESOLVED)
    # Bad row removed: every lockup-derived field equals the clean book.
    assert quarantined.avg_lockup_days == clean.avg_lockup_days, quarantined
    assert quarantined.velocity_multiplier == clean.velocity_multiplier, quarantined
    assert quarantined.annualized_yield_apy_pct == clean.annualized_yield_apy_pct, quarantined
    assert quarantined.north_star_score == clean.north_star_score, quarantined
    assert quarantined.cwr_bid_marked == clean.cwr_bid_marked, quarantined


def test_single_negative_and_zero_raise_not_floor_masked():
    # Pre-fix both RETURNED avg_lockup 1.0 via max(1.0, avg_lockup).
    for bad in BAD_LOCKUPS:
        msg = _score_raises([_trade(est_days_to_resolution=bad)])
        assert msg is not None and "SF-BL-9364" in msg, (bad, msg)


# ---- 2. Scoring gate scope ---------------------------------------------------

def test_all_nonpositive_values_raise_for_both_sides():
    for bad in BAD_LOCKUPS:
        for action in (ak.TradeAction.BUY_YES, ak.TradeAction.BUY_NO):
            msg = _score_raises([_trade(action=action, est_days_to_resolution=bad)])
            assert msg is not None, (bad, action)


def test_nonpositive_among_finite_trades_raises_either_order():
    good = _trade(market_id="GOOD")
    bad = _trade(market_id="BAD", est_days_to_resolution=-30.0)
    assert _score_raises([good, bad]) is not None
    assert _score_raises([bad, good]) is not None


def test_mutated_trade_also_raises():
    t = _trade()
    t.est_days_to_resolution = -30.0
    assert _score_raises([t]) is not None
    t2 = _trade()
    t2.est_days_to_resolution = 0.0
    assert _score_raises([t2]) is not None


def test_nonpositive_raises_before_invested_filter():
    # A non-SKIP trade with invested 0 is validated before the
    # invested>0 executed filter can drop it silently (SF-BL-9475 order).
    msg = _score_raises([_trade(invested_capital=0.0, est_days_to_resolution=-5.0)])
    assert msg is not None and "SF-BL-9364" in msg, msg


def test_nonfinite_keeps_sf_bl_9475_identity():
    for bad in (NAN, INF, -INF):
        msg = _score_raises([_trade(est_days_to_resolution=bad)])
        assert msg is not None and "SF-BL-9475" in msg, (bad, msg)
        assert "NON_FINITE_TRADE_FIELD" in msg, msg


def test_skip_trades_exempt_and_unchanged():
    skip = _trade(action=ak.TradeAction.SKIP, invested_capital=0.0,
                  net_pnl_bid_marked=0.0, est_days_to_resolution=-30.0,
                  skip_reason="INVALID_EST_DAYS_TO_RESOLUTION")
    sc = ak.evaluate_fleet_north_star("fixture", [skip], RESOLVED)
    assert sc.executed_trades == 0 and sc.skipped_opportunities == 1, sc
    assert sc.north_star_score == 0.0, sc


def test_helper_scope():
    assert ak._est_days_in_domain(40.0) and ak._est_days_in_domain(1)
    assert ak._est_days_in_domain(0.001) and ak._est_days_in_domain(3650.0)
    for bad in BAD_LOCKUPS + NONFINITE:
        assert not ak._est_days_in_domain(bad), bad
    ak._validate_scored_trade_lockups([_trade()])  # positive: no raise
    try:
        ak._validate_scored_trade_lockups([_trade(est_days_to_resolution=0.0)])
    except ValueError:
        return
    raise AssertionError("helper accepted 0.0 lockup")


# ---- 3. Construction gate ------------------------------------------------------

def test_snapshot_nonpositive_raises():
    for bad in BAD_LOCKUPS:
        msg = _construct_raises(est_days_to_resolution=bad)
        assert msg is not None and "SF-BL-9364" in msg, (bad, msg)


def test_snapshot_nonfinite_and_nonnumeric_raise():
    for bad in NONFINITE:
        msg = _construct_raises(est_days_to_resolution=bad)
        assert msg is not None and "SF-BL-9364" in msg, (bad, msg)


def test_snapshot_valid_lockups_construct():
    for good in (0.001, 1, 2.0, 40.0, 3650.0):
        book = _book(est_days_to_resolution=good)
        assert book.est_days_to_resolution == good


def test_snapshot_error_names_value():
    msg = _construct_raises(est_days_to_resolution=-30.0)
    assert "-30" in msg and "est_days_to_resolution" in msg, msg


# ---- 4. Evaluate gate (mutated / unpickled books) ------------------------------

def test_evaluate_mutated_book_skips_invalid_est_days():
    book = _book()
    book.est_days_to_resolution = -30.0  # bypass __post_init__
    cluster = {}
    res = ak.evaluate_and_size_opportunity(book, 0.80, 0.05, 100000.0, cluster)
    assert res.action == ak.TradeAction.SKIP, res
    assert res.skip_reason is not None, res
    assert "INVALID_EST_DAYS_TO_RESOLUTION" in res.skip_reason, res.skip_reason
    assert "SF-BL-9364" in res.skip_reason, res.skip_reason
    assert res.invested_capital == 0.0 and res.shares_acquired == 0.0, res
    assert res.allocated_fraction == 0.0 and res.kelly_fraction_raw == 0.0, res
    assert cluster == {}, cluster


def test_evaluate_mutated_zero_and_nan_skip():
    for bad in (0.0, NAN):
        book = _book()
        book.est_days_to_resolution = bad
        res = ak.evaluate_and_size_opportunity(book, 0.80, 0.05, 100000.0, {})
        assert res.action == ak.TradeAction.SKIP, (bad, res)
        assert "INVALID_EST_DAYS_TO_RESOLUTION" in (res.skip_reason or ""), res


def test_evaluate_gate01_skip_record_scores_clean():
    # The Gate 0.1 SKIP record carries the offending value for audit but
    # is a SKIP, so scoring it (alone or beside real trades) never pools it.
    book = _book()
    book.est_days_to_resolution = -30.0
    res = ak.evaluate_and_size_opportunity(book, 0.80, 0.05, 100000.0, {})
    sc = ak.evaluate_fleet_north_star("fixture", [res, _trade(market_id="B")], RESOLVED)
    assert sc.avg_lockup_days == 40.0 and sc.velocity_multiplier == 3.021, sc
    assert sc.skipped_opportunities == 1 and sc.executed_trades == 1, sc


def test_evaluate_valid_book_unchanged():
    book = _book()
    cluster = {}
    res = ak.evaluate_and_size_opportunity(book, 0.80, 0.05, 100000.0, cluster)
    assert res.action == ak.TradeAction.BUY_YES, res
    assert res.est_days_to_resolution == 40.0, res
    assert res.invested_capital > 0, res


# ---- 5. Valid behaviour unchanged ----------------------------------------------

def test_control_fixture_exact():
    sc = ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED)
    assert sc.avg_lockup_days == 40.0 and sc.velocity_multiplier == 3.021, sc
    assert sc.cwr_bid_marked == 0.2 and sc.executed_trades == 1, sc


def test_positive_edge_values_unchanged():
    # Tiny positive lock, int lock, huge finite lock (no upper bound
    # preregistered), capital weighting across positive locks.
    tiny = ak.evaluate_fleet_north_star(
        "fixture", [_trade(est_days_to_resolution=0.5)], RESOLVED)
    assert tiny.avg_lockup_days == 1.0, tiny  # pre-existing max(1.0, .) floor
    assert math.isfinite(tiny.north_star_score), tiny
    big = ak.evaluate_fleet_north_star(
        "fixture", [_trade(est_days_to_resolution=3650)], RESOLVED)
    assert big.avg_lockup_days == 3650.0, big
    mixed = ak.evaluate_fleet_north_star(
        "fixture", [_trade(market_id="A", est_days_to_resolution=10.0),
                    _trade(market_id="B", est_days_to_resolution=40.0)], RESOLVED)
    assert mixed.avg_lockup_days == 25.0, mixed


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
    # Current-tree values at this task's run. NOTE: the disciplined card
    # reads 0.0547 here, not the 0.1059 pinned by SF-BL-9475's older
    # suite — other fleet tasks landed between that suite and this one
    # (SF-BL-9475's own benchmark check is likewise 14/15 on this tree).
    # The binding no-change evidence for THIS task is the full-output
    # diff benchmark_prefix.txt vs benchmark_postfix.txt: EMPTY.
    assert cards[0]["north_star_score"] == 0.0547, cards[0]
    assert cards[1]["north_star_score"] == -0.1499, cards[1]
    assert cards[0]["ending_account_value_bid_marked"] == 11123.65, cards[0]
    assert cards[1]["ending_account_value_bid_marked"] == 9326.36, cards[1]
    assert cards[0]["cwr_bid_marked"] == 0.284, cards[0]
    assert cards[1]["cwr_bid_marked"] == -0.0784, cards[1]


# ---- 6. Kill-condition evidence --------------------------------------------------

def test_kill_condition_fails_no_timestamp_producer():
    # No producer derives est_days from a resolution timestamp: the
    # snapshot takes a bare float (proven live — it accepted -30.0
    # pre-fix, and still accepts any positive literal with no timestamp
    # argument anywhere in its signature), the dataclass is mutable,
    # and settle_or_mark_trade never writes the field. So the gates
    # above, not the kill, apply.
    import inspect
    params = inspect.signature(ak.OrderBookSnapshot).parameters
    assert "est_days_to_resolution" in params, params
    assert not any("time" in p or "date" in p or "ts" in p for p in params), params
    t = _trade()
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    assert t.est_days_to_resolution == 40.0, t  # settle left it untouched


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
