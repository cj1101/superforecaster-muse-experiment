"""
test_alpha_kelly_est_days_type.py — SF-BL-11534 verification tests.

Pre-registered done criterion: "OrderBookSnapshot construction or
evaluate entry must reject any bool or non-numeric
est_days_to_resolution with a named type error before any sizing,
fixtured at est_days='30' and est_days=True, with numeric-lock books
and scorecards bit-identical; kill if a string-lock book still
produces an executed TradeExecutionResult or the only rejection
remains the scorer's NON_FINITE mislabel."

Pre-fix (repro_prefix.txt in the task work dir): construction of
'30'/True raised only SF-BL-9364's VALUE-domain error, evaluate
Gate 0.1 only "is not a finite number > 0 (SF-BL-9364)", and a direct
trade scored only SF-BL-9475's NON_FINITE_TRADE_FIELD mislabel.

Post-fix, the TYPE class (bool or non-numeric; int/float only, never
coerced) is named INVALID_EST_DAYS_TYPE / SF-BL-11534 at three
surfaces: construction (raise), evaluate entry (Gate ET, immediately
after the book-argument gate, SKIP zero size / cluster untouched,
umbrella INVALID_EST_DAYS_TO_RESOLUTION token retained), and the
scorer (raise, before SF-BL-9475; SKIP records exempt). Numeric
locks — NaN/inf (SF-BL-9475 identity), <= 0 (SF-BL-9364 identity),
huge int — keep their pre-fix handling bit-identically.

Run:  python3 test_alpha_kelly_est_days_type.py  (exit 0 iff pass)
No pytest dependency: test_* functions with assert + __main__ runner.
"""
import io
import contextlib
import json
import math
import os
import sys
from decimal import Decimal

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster/superforecaster"))
import alpha_kelly as ak  # noqa: E402

PASS = []
NAN = float("nan")
INF = float("inf")
RESOLVED = [(0.6, 1), (0.6, 0)]
BAD_TYPES = ("30", "40", "", "foo", True, False, None, [], {}, (30,), Decimal("30"), 30j)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(**over):
    base = dict(
        market_id="BOOK-11534", title="fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C11534",
        est_days_to_resolution=40.0,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000.0)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100000.0)],
        fee_rate=0.01)
    base.update(over)
    return ak.OrderBookSnapshot(**base)


def _trade(**over):
    base = dict(
        market_id="TRADE-11534", title="fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C11534",
        action=ak.TradeAction.BUY_YES, calibrated_prob_yes=0.60,
        market_mid_prob=0.50, exec_price=0.50, liquidation_bid_price=0.50,
        kl_divergence_nats=0.0, expected_log_growth=0.0,
        kelly_fraction_raw=0.10, allocated_fraction=0.05,
        invested_capital=500.0, shares_acquired=1000.0,
        est_days_to_resolution=40.0, is_settled=True, outcome_yes=1,
        net_pnl_bid_marked=100.0)
    base.update(over)
    return ak.TradeExecutionResult(**base)


def _construct_raises(v):
    try:
        _book(est_days_to_resolution=v)
    except ValueError as e:
        return str(e)
    return None


def _score_raises(trades):
    try:
        ak.evaluate_fleet_north_star("fixture", trades, RESOLVED)
    except ValueError as e:
        return str(e)
    return None


def _mutated_eval(v):
    book = _book()
    book.est_days_to_resolution = v  # bypass __post_init__
    cluster = {}
    res = ak.evaluate_and_size_opportunity(book, 0.80, 0.05, 100000.0, cluster)
    return res, cluster


# ---- 1. Construction: TASK fixtures ---------------------------------------

def test_task_fixtures_raise_named_type_error():
    for v in ("30", True):
        msg = _construct_raises(v)
        assert msg is not None, v
        assert "INVALID_EST_DAYS_TYPE" in msg, msg
        assert "SF-BL-11534" in msg, msg
        assert repr(v) in msg and type(v).__name__ in msg, msg
        assert "SF-BL-9364" in msg, msg  # SF-BL-9364 token retained


def test_full_type_grid_raises():
    for v in BAD_TYPES:
        msg = _construct_raises(v)
        assert msg is not None and "INVALID_EST_DAYS_TYPE" in msg, (v, msg)
        assert "SF-BL-11534" in msg, (v, msg)


def test_no_coercion():
    # '30' is never parsed and True never read as a 1-day lock: both
    # raise; the producer remediation float('30') trades cleanly.
    assert _construct_raises("30") is not None
    book = _book(est_days_to_resolution=float("30"))
    assert book.est_days_to_resolution == 30.0


def test_helper_scope():
    assert ak._est_days_is_numeric_type(30) and ak._est_days_is_numeric_type(30.0)
    assert ak._est_days_is_numeric_type(NAN) and ak._est_days_is_numeric_type(INF)
    assert ak._est_days_is_numeric_type(10 ** 400)  # numeric type, downstream owns it
    for v in BAD_TYPES:
        assert not ak._est_days_is_numeric_type(v), v


def test_numeric_domain_errors_unchanged():
    # Numeric bad values keep SF-BL-9364's domain message exactly,
    # with no 11534 type token.
    for v in (-30.0, 0, 0.0, NAN, INF, -INF):
        msg = _construct_raises(v)
        assert msg is not None and "SF-BL-9364" in msg, (v, msg)
        assert "INVALID_EST_DAYS_TYPE" not in msg, (v, msg)
        assert "outside pre-registered domain" in msg, (v, msg)


# ---- 2. Evaluate entry ------------------------------------------------------

def test_evaluate_task_fixtures_skip_type_named():
    for v in ("30", True):
        res, cluster = _mutated_eval(v)
        assert res.action == ak.TradeAction.SKIP, res
        assert "INVALID_EST_DAYS_TYPE" in (res.skip_reason or ""), res.skip_reason
        assert "INVALID_EST_DAYS_TO_RESOLUTION" in (res.skip_reason or ""), res.skip_reason
        assert "SF-BL-11534" in (res.skip_reason or ""), res.skip_reason
        assert res.invested_capital == 0.0 and res.shares_acquired == 0.0, res
        assert res.allocated_fraction == 0.0 and res.kelly_fraction_raw == 0.0, res
        assert res.est_days_to_resolution == v or res.est_days_to_resolution is v, res
        assert cluster == {}, cluster


def test_evaluate_type_grid_skips():
    for v in BAD_TYPES:
        res, cluster = _mutated_eval(v)
        assert res.action == ak.TradeAction.SKIP, (v, res)
        assert "INVALID_EST_DAYS_TYPE" in (res.skip_reason or ""), (v, res.skip_reason)
        assert cluster == {}, (v, cluster)


def test_evaluate_type_gate_precedes_other_gates():
    # A type-bad lock is named even when an earlier-ordered legacy gate
    # would also have fired (invalid posterior / invalid fee): Gate ET
    # sits at evaluate entry, before Gate E0 / 0.0 / 0.
    book = _book()
    book.est_days_to_resolution = "30"
    res = ak.evaluate_and_size_opportunity(book, NAN, 0.05, 100000.0, {})
    assert "INVALID_EST_DAYS_TYPE" in (res.skip_reason or ""), res.skip_reason
    book2 = _book()
    book2.est_days_to_resolution = True
    book2.fee_rate = -0.20
    res2 = ak.evaluate_and_size_opportunity(book2, 0.80, 0.05, 100000.0, {})
    assert "INVALID_EST_DAYS_TYPE" in (res2.skip_reason or ""), res2.skip_reason


def test_evaluate_numeric_gates_unchanged():
    # Numeric bad locks keep their existing gate/reason exactly.
    for v, token in ((-30.0, "SF-BL-9364"), (0.0, "SF-BL-9364"),
                     (NAN, "SF-BL-9364")):
        res, cluster = _mutated_eval(v)
        assert res.action == ak.TradeAction.SKIP, (v, res)
        assert "INVALID_EST_DAYS_TO_RESOLUTION" in (res.skip_reason or ""), res
        assert "INVALID_EST_DAYS_TYPE" not in (res.skip_reason or ""), res.skip_reason
        assert cluster == {}, cluster


def test_type_gate_skip_record_scores_clean():
    # The SKIP record carries the offending value for audit and is a
    # SKIP, so scoring it beside a real trade never pools it and never
    # trips the scorer's type gate.
    res, _ = _mutated_eval("30")
    sc = ak.evaluate_fleet_north_star("fixture", [res, _trade(market_id="B")], RESOLVED)
    assert sc.avg_lockup_days == 40.0 and sc.velocity_multiplier == 3.021, sc
    assert sc.skipped_opportunities == 1 and sc.executed_trades == 1, sc


def test_no_executed_trade_from_string_lock_kill():
    # Kill criterion: a string-lock book must never produce an
    # executed TradeExecutionResult, at any posterior/equity.
    for p in (0.02, 0.5, 0.8, 0.98):
        for equity in (1000.0, 100000.0):
            res, cluster = _mutated_eval("30")
            assert res.action == ak.TradeAction.SKIP, (p, equity, res)
            assert cluster == {}, cluster


# ---- 3. Scorer ----------------------------------------------------------------

def test_scorer_task_fixtures_raise_type_not_nonfinite():
    for v in ("30", True):
        t = _trade()
        t.est_days_to_resolution = v
        msg = _score_raises([t])
        assert msg is not None, v
        assert "INVALID_EST_DAYS_TYPE" in msg, msg
        assert "SF-BL-11534" in msg, msg
        assert not msg.startswith("NON_FINITE_TRADE_FIELD"), msg
        assert "TRADE-11534" in msg, msg


def test_scorer_type_grid_raises():
    for v in BAD_TYPES:
        t = _trade()
        t.est_days_to_resolution = v
        msg = _score_raises([t])
        assert msg is not None and "INVALID_EST_DAYS_TYPE" in msg, (v, msg)


def test_scorer_numeric_identities_unchanged():
    for v in (NAN, INF, -INF):
        msg = _score_raises([_trade(est_days_to_resolution=v)])
        assert msg is not None and "NON_FINITE_TRADE_FIELD" in msg, (v, msg)
        assert "SF-BL-9475" in msg, (v, msg)
    for v in (-30.0, 0.0):
        msg = _score_raises([_trade(est_days_to_resolution=v)])
        assert msg is not None and "SF-BL-9364" in msg, (v, msg)
        assert "INVALID_EST_DAYS_TYPE" not in msg, (v, msg)


def test_scorer_skip_trades_exempt():
    skip = _trade(action=ak.TradeAction.SKIP, invested_capital=0.0,
                  net_pnl_bid_marked=0.0, skip_reason="x")
    skip.est_days_to_resolution = "30"
    sc = ak.evaluate_fleet_north_star("fixture", [skip], RESOLVED)
    assert sc.executed_trades == 0 and sc.skipped_opportunities == 1, sc
    assert sc.north_star_score == 0.0, sc


def test_scorer_type_gate_order_vs_other_bad_fields():
    # Type gate runs before 9475: a trade bad in both lock type and
    # invested finiteness is named for the lock type first.
    t = _trade(invested_capital=NAN)
    t.est_days_to_resolution = "30"
    msg = _score_raises([t])
    assert msg is not None and "INVALID_EST_DAYS_TYPE" in msg, msg


# ---- 4. Numeric behaviour bit-identical ---------------------------------------

def test_clean_book_evaluates_identically():
    cluster = {}
    res = ak.evaluate_and_size_opportunity(_book(), 0.80, 0.05, 100000.0, cluster)
    assert res.action == ak.TradeAction.BUY_YES, res
    assert res.invested_capital == 15000.0, res
    assert res.shares_acquired == 29702.970297029704, res
    assert res.exec_price == 0.505, res
    assert res.est_days_to_resolution == 40.0, res
    assert cluster == {"C11534": 0.15}, cluster


def test_numeric_lock_grid_constructs():
    for v in (0.001, 1, 2.0, 30, 40.0, 3650.0):
        assert _book(est_days_to_resolution=v).est_days_to_resolution == v


def test_control_scorecard_exact():
    sc = ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED)
    assert sc.avg_lockup_days == 40.0 and sc.velocity_multiplier == 3.021, sc
    assert sc.cwr_bid_marked == 0.2 and sc.executed_trades == 1, sc
    assert sc.north_star_score == 0.2223, sc


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
    # Values pinned from repro_prefix.txt on this tree (other fleet
    # tasks have moved the disciplined card since SF-BL-9364's suite);
    # the binding no-change evidence is the repro pre/post diff, which
    # touches only the type-error sections.
    assert cards[0]["north_star_score"] == -0.1611, cards[0]
    assert cards[1]["north_star_score"] == -0.1499, cards[1]
    assert cards[0]["ending_account_value_bid_marked"] == 9170.88, cards[0]
    assert cards[1]["ending_account_value_bid_marked"] == 9326.36, cards[1]


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for fn in ALL_TESTS:
        try:
            fn()
            check(fn.__name__, True)
        except Exception as e:  # AssertionError, or AttributeError pre-fix
            check(fn.__name__, False, f"{type(e).__name__}: {e}")
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
