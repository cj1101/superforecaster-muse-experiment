"""
test_alpha_kelly_settle_trade_type_validation.py — SF-BL-10971 tests.

Pre-registered done criterion: "settle_or_mark_trade must raise a named
ValueError at entry for a non-TradeExecutionResult trade, fixtured at
None and a dict, with genuine-trade settlements and marks bit-identical
including the SF-BL-9541 fixtures; kill if a caller census shows trade
only ever arrives as an evaluate_and_size_opportunity result under a
tested invariant."

Pre-fix reproduction (verified live 2026-10-23, before the patch; see
CALLER_CENSUS.md / repro_prefix.txt): on both branches —
  trade=None  -> AttributeError: 'NoneType' object has no attribute
                 'shares_acquired'  (at the SF-BL-9541 entry validation)
  trade=dict  -> AttributeError: 'dict' object has no attribute
                 'shares_acquired'
  trade='trade' / 123 / [] -> identical bare AttributeError per type
  duck-typed SimpleNamespace fake -> marked/settled as if genuine; a
                 partial fake booked net PnL +500.0 and flipped
                 is_settled BEFORE crashing in release_cluster_exposure

Kill condition is NOT met (CALLER_CENSUS.md): the benchmark's naive
fleet passes directly constructed TradeExecutionResult records (not
evaluate results), fleet/analysis code rebuilds records from dicts,
the annotation is unenforced, and no isinstance or tested invariant
guards the entry point — so the type is validated at settle entry,
before any attribute of the argument is dereferenced.

Run:  python3 test_alpha_kelly_settle_trade_type_validation.py
No pytest dependency: test_* functions with assert + __main__ runner.
"""
import contextlib
import io
import os
import re
import sys
import types

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster/superforecaster"))
import alpha_kelly as ak  # noqa: E402

PASS = []
SENTINEL_PNL = 777.0


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _trade(shares=1000.0, invested=500.0, bid=0.50, action=None, pnl=0.0):
    """SF-BL-9541 prereg fixture: $500 stake, 1000 shares at exec 0.50."""
    t = ak.TradeExecutionResult(
        market_id="FIXTURE", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C10971",
        action=action or ak.TradeAction.BUY_YES,
        calibrated_prob_yes=0.60, market_mid_prob=0.50, exec_price=0.50,
        liquidation_bid_price=bid, kl_divergence_nats=0.0,
        expected_log_growth=0.0, kelly_fraction_raw=0.10,
        allocated_fraction=0.05, invested_capital=invested,
        shares_acquired=shares, est_days_to_resolution=10.0)
    t.net_pnl_bid_marked = pnl
    return t


def _raises_named(trade, **settle_kw):
    """(True, msg) iff settle raises the named SF-BL-10971 ValueError."""
    try:
        ak.settle_or_mark_trade(trade, **settle_kw)
    except ValueError as e:
        msg = str(e)
        return ("INVALID_TRADE" in msg and "SF-BL-10971" in msg
                and "TradeExecutionResult" in msg), msg
    except Exception as e:  # noqa: BLE001 - a bare crash is a failure
        return False, f"wrong exception {type(e).__name__}: {e}"
    return False, "no exception raised"


DICT_FIXTURE = {"shares_acquired": 1000.0, "invested_capital": 500.0,
                "action": "BUY_YES", "liquidation_bid_price": 0.50}
NON_TRADES = (None, DICT_FIXTURE, {}, "trade", "", 123, 0, 1.5, True,
              [], ["x"], (1, 2), object())


def _fake(partial=False):
    ns = types.SimpleNamespace(
        shares_acquired=1000.0, invested_capital=500.0,
        action=ak.TradeAction.BUY_YES, liquidation_bid_price=0.50,
        is_settled=False, outcome_yes=None, net_pnl_bid_marked=SENTINEL_PNL)
    if not partial:
        ns.factor_cluster = "C10971"
        ns.allocated_fraction = 0.05
        ns.cluster_exposure_used = None
        ns.cluster_exposure_released = False
    return ns


# ---- 1. Prereg fixtures: None and a dict, both branches ---------------------

def test_none_raises_named_valueerror_both_branches():
    for kw in (dict(is_settled=True, outcome_yes=1),
               dict(is_settled=True, outcome_yes=0),
               dict(is_settled=False)):
        ok, msg = _raises_named(None, **kw)
        assert ok, (kw, msg)
        assert "NoneType" in msg and "None" in msg, msg


def test_dict_raises_named_valueerror_both_branches():
    for fixture in (DICT_FIXTURE, {}, {"market_id": "X"}):
        for kw in (dict(is_settled=True, outcome_yes=1),
                   dict(is_settled=False)):
            ok, msg = _raises_named(fixture, **kw)
            assert ok, (fixture, kw, msg)
            assert "dict" in msg, msg


def test_extended_non_trade_space_raises_both_branches():
    for bad in NON_TRADES:
        for kw in (dict(is_settled=True, outcome_yes=1),
                   dict(is_settled=False)):
            ok, msg = _raises_named(bad, **kw)
            assert ok, (bad, kw, msg)


def test_no_bare_attributeerror_anywhere():
    for bad in NON_TRADES + (_fake(), _fake(partial=True)):
        try:
            ak.settle_or_mark_trade(bad, is_settled=True, outcome_yes=1)
        except AttributeError as e:
            raise AssertionError(f"bare AttributeError for {bad!r}: {e}")
        except ValueError:
            pass
        else:
            raise AssertionError(f"no raise for {bad!r}")


# ---- 2. Duck-typed fakes: rejected at entry, never mutated ------------------

def test_fake_rejected_and_untouched_both_branches():
    for kw in (dict(is_settled=True, outcome_yes=1), dict(is_settled=False)):
        fake = _fake()
        ok, _ = _raises_named(fake, **kw)
        assert ok
        # Entry raise precedes ANY read/write of consequence: the fake's
        # mutable state is byte-identical (pre-fix it booked +500 PnL).
        assert fake.net_pnl_bid_marked == SENTINEL_PNL
        assert fake.is_settled is False and fake.outcome_yes is None
        assert fake.cluster_exposure_released is False


def test_partial_fake_no_longer_books_pnl_before_crashing():
    fake = _fake(partial=True)
    ok, _ = _raises_named(fake, is_settled=True, outcome_yes=1)
    assert ok and fake.net_pnl_bid_marked == SENTINEL_PNL
    assert fake.is_settled is False


# ---- 3. Gate ordering: type check precedes every field validation -----------

def test_type_check_precedes_position_field_validation():
    # A dict carrying OUT-OF-DOMAIN field values must raise the
    # SF-BL-10971 type error, not the SF-BL-9541 field error and not a
    # bare AttributeError — the type gate runs before any dereference.
    bad_dict = {"shares_acquired": -100.0, "invested_capital": float("nan")}
    try:
        ak.settle_or_mark_trade(bad_dict, is_settled=True, outcome_yes=1)
    except ValueError as e:
        # The raised error is the 10971 TYPE error (INVALID_TRADE), not
        # the 9541 FIELD error (which opens "shares_acquired -100.0
        # outside pre-registered domain ... (SF-BL-9541)"). The 10971
        # message mentions 9541 only descriptively, as the site of the
        # pre-fix dereference.
        assert "INVALID_TRADE" in str(e) and "SF-BL-10971" in str(e), str(e)
        assert "outside pre-registered domain" not in str(e), str(e)
    else:
        raise AssertionError("bad dict accepted")


def test_helper_domain():
    assert ak._is_trade_execution_result(_trade())
    assert not ak._is_trade_execution_result(None)
    assert not ak._is_trade_execution_result(DICT_FIXTURE)
    assert not ak._is_trade_execution_result(_fake())
    ak._validate_trade_argument(_trade())  # genuine: no raise


def test_subclass_of_genuine_trade_passes():
    class SubTrade(ak.TradeExecutionResult):
        pass
    t = _trade()
    sub = SubTrade(**{**t.__dict__})
    out = ak.settle_or_mark_trade(sub, is_settled=True, outcome_yes=1)
    assert out.net_pnl_bid_marked == 500.0


# ---- 4. Genuine trades bit-identical, incl. SF-BL-9541 fixtures --------------

def test_genuine_settled_exact():
    win = ak.settle_or_mark_trade(_trade(), is_settled=True, outcome_yes=1)
    assert win.net_pnl_bid_marked == 500.0, win.net_pnl_bid_marked
    loss = ak.settle_or_mark_trade(_trade(), is_settled=True, outcome_yes=0)
    assert loss.net_pnl_bid_marked == -500.0
    no_win = ak.settle_or_mark_trade(
        _trade(action=ak.TradeAction.BUY_NO), is_settled=True, outcome_yes=0)
    assert no_win.net_pnl_bid_marked == 500.0
    no_loss = ak.settle_or_mark_trade(
        _trade(action=ak.TradeAction.BUY_NO), is_settled=True, outcome_yes=1)
    assert no_loss.net_pnl_bid_marked == -500.0
    assert isinstance(win, ak.TradeExecutionResult) and win is not None


def test_genuine_mark_exact():
    for bid, pnl in ((0.5, 0.0), (0.25, -250.0), (0.75, 250.0),
                     (0.0, -500.0), (1.0, 500.0)):
        t = ak.settle_or_mark_trade(_trade(bid=bid), is_settled=False)
        assert t.net_pnl_bid_marked == pnl, (bid, t.net_pnl_bid_marked)


def test_sf_bl_9541_fixtures_still_raise_untouched():
    # The 9541 domain gate is unchanged for genuine records: bad shares /
    # invested raise ITS ValueError, PnL sentinel untouched.
    for kw in (dict(shares=-100.0), dict(shares=float("inf")),
               dict(invested=float("nan")), dict(invested=-500.0)):
        t = _trade(pnl=SENTINEL_PNL, **kw)
        try:
            ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
        except ValueError as e:
            assert "SF-BL-9541" in str(e), str(e)
            assert t.net_pnl_bid_marked == SENTINEL_PNL
        else:
            raise AssertionError(f"9541 fixture {kw} accepted")


def test_skip_and_zero_invested_unchanged():
    skip = _trade(action=ak.TradeAction.SKIP, shares=0.0, invested=0.0, bid=1.50)
    assert ak.settle_or_mark_trade(skip, is_settled=False).net_pnl_bid_marked == 0.0
    zero = _trade(shares=0.0, invested=0.0)
    assert ak.settle_or_mark_trade(zero, is_settled=True,
                                   outcome_yes=1).net_pnl_bid_marked == 0.0


def test_evaluate_produced_trades_pass_gate():
    # The one producer path the kill criterion names: an evaluate result
    # (SKIP or executed) settles/marks exactly as before.
    book = ak.OrderBookSnapshot(
        market_id="GATE", title="gate fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C10971",
        est_days_to_resolution=10.0,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100000)],
        fee_rate=0.0)
    used = {}
    t = ak.evaluate_and_size_opportunity(
        book=book, calibrated_prob_yes=0.65, inter_agent_dispersion=0.05,
        portfolio_equity=10000.0, cluster_exposure_used=used)
    out = ak.settle_or_mark_trade(t, is_settled=False)
    assert isinstance(out, ak.TradeExecutionResult)


# ---- 5. Kill-criterion evidence ----------------------------------------------

def test_kill_fails_naive_fleet_constructs_directly():
    # run_benchmark_simulation's naive fleet builds TradeExecutionResult
    # records directly (not via evaluate) and settles them — proof at
    # runtime that "trade only ever arrives as an evaluate result" is
    # false, so no producer invariant can substitute for the entry gate.
    src = io.open(os.path.expanduser(
        "~/workspace/superforecaster/superforecaster/alpha_kelly.py"),
        encoding="utf-8").read()
    bench = src[src.index("def run_benchmark_simulation"):]
    assert bench.count("TradeExecutionResult(") >= 2
    assert bench.count("settle_or_mark_trade(") >= 3


# ---- 6. Benchmark: full output bit-identical to pre-fix baseline -------------

def test_benchmark_output_unchanged():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    scores = [float(x) for x in
              re.findall(r'"north_star_score": (-?[\d.]+)', out)]
    # Current-tree baseline captured pre-fix this cycle (the disciplined
    # figure drifted from older tasks' +0.1059 via intervening fleet
    # work; the invariant pinned here is pre/post identity, verified by
    # full-output diff in repro_postfix.txt / DONE-CLAIM).
    assert scores == [-0.1611, -0.1499], scores
    baseline_path = "/tmp/sf10971/benchmark_prefix.txt"
    if os.path.exists(baseline_path):
        baseline = io.open(baseline_path, encoding="utf-8").read()
        assert out == baseline, "benchmark output differs from pre-fix"


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
