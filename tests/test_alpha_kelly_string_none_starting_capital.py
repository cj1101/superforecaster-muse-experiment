"""SF-BL-11006 verification tests — string / None starting_capital.

Pre-registered done criterion (TASK.md): "evaluate_fleet_north_star
must raise a named ValueError for a starting_capital that is not a
finite real > 0 (bool excluded), fixtured at '10000' and None, with
the clean-capital scorecard bit-identical; kill if a caller census
shows every caller passes a validated numeric literal capital."

Kill condition NOT met (CALLER_CENSUS.md): callers pass variables
(benchmark starting_cap; EQUITY / START / start / cap /
START_CAPITAL), dict values (fx["starting_capital"], b["start"]) and
float(bankroll) — no validated-numeric-literal invariant anywhere,
and no type validation upstream: a capital arriving from a JSON /
config / fixture source as the string '10000', or a missing key
yielding None, reaches the scorer unconverted.

State found by this task: the mandated gate ALREADY exists — landed
by SF-BL-9437 (`_validate_starting_capital`, first statement of
`evaluate_fleet_north_star`, named error INVALID_STARTING_CAPITAL,
reusing `_is_finite_number` so bool / non-numeric / NaN / +-inf all
reject). SF-BL-9437's prereg fixtures were 0 / -10000 / NaN (the
string/None modes appear only in its repro table); SF-BL-10800
fixtured bool and SF-BL-10850 fixtured NaN on losing books. This
suite adds what none of them fixtured as prereg: the non-numeric
TYPE dimension — '10000' and None as headline fixtures on the exact
task fixture, the full non-numeric grid, the exact bare-TypeError
reproduction via a gate-only bypass, and a byte-identity proof that
the gate is a strict no-op for valid capitals. No repo code change
was needed or made by this task.

Fixture (exact task rationale numbers, identified this run):
  one executed trade, invested $500, PnL +$100, est_days 10,
  resolved [(0.8, 1)]
    clean 10000.0 -> N_star 0.1635, CWR 0.2, ECE 0.2, velocity
                     6.042, ending 10100.0, return +1.0%, MDD 0.0%
    '10000' pre-fix (bypassed) -> TypeError: can only concatenate
                     str (not "float") to str
    None pre-fix (bypassed)    -> TypeError: unsupported operand
                     type(s) for +=: 'NoneType' and 'float'

Run:  python3 test_alpha_kelly_string_none_starting_capital.py
(exit 0 iff all pass; no pytest dependency, fleet convention.)
"""
import dataclasses
import inspect
import io
import contextlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster/superforecaster"))
import alpha_kelly as ak  # noqa: E402

PASS = []
RESOLVED = [(0.8, 1)]
# Headline prereg fixtures for THIS task: the non-numeric types.
STRING_CAP = '10000'
NONE_CAP = None
# Full non-numeric grid: strings, None, containers, bool (explicitly
# excluded by the prereg domain), and a bare object.
NON_NUMERIC_CAPS = ('10000', '10000.0', '', 'nan', '0', None,
                    [], {}, True, False)
# Value-dimension companions (same unified named error; fixtured by
# SF-BL-9437 / SF-BL-10850, re-checked here only for domain unity).
BAD_VALUE_CAPS = (float("nan"), float("inf"), float("-inf"),
                  0, 0.0, -10000, -10000.0)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _trade(**over):
    base = dict(
        market_id="FIXTURE-11006", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C11006",
        action=ak.TradeAction.BUY_YES,
        calibrated_prob_yes=0.80, market_mid_prob=0.50, exec_price=0.50,
        liquidation_bid_price=0.50, kl_divergence_nats=0.0,
        expected_log_growth=0.0, kelly_fraction_raw=0.10,
        allocated_fraction=0.05, invested_capital=500.0,
        shares_acquired=1000.0, est_days_to_resolution=10.0,
        is_settled=True, outcome_yes=1, net_pnl_bid_marked=100.0)
    base.update(over)
    return ak.TradeExecutionResult(**base)


def _raises(cap, trades=None, resolved=RESOLVED):
    """The ValueError message for `cap`, or None if no ValueError.

    A TypeError (the pre-fix mode) is NOT caught here: it propagates
    and fails the calling test, which is exactly the distinction
    this task exists to pin.
    """
    if trades is None:
        trades = [_trade()]
    try:
        ak.evaluate_fleet_north_star("fixture", trades, resolved,
                                     starting_capital=cap)
    except ValueError as e:
        return str(e)
    return None


def _json(sc):
    return json.dumps(dataclasses.asdict(sc), sort_keys=True)


# ---- 1. Prereg fixture, clean capital (exact task numbers) ------------------

def test_clean_fixture_exact():
    sc = ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED,
                                      starting_capital=10000.0)
    assert sc.north_star_score == 0.1635, sc
    assert sc.ending_account_value_bid_marked == 10100.0, sc
    assert sc.portfolio_return_pct == 1.0, sc
    assert sc.cwr_bid_marked == 0.2 and sc.ece_10bin == 0.2, sc
    assert sc.velocity_multiplier == 6.042, sc
    assert sc.max_drawdown_pct == 0.0 and sc.drawdown_penalty == 0.0, sc
    assert sc.starting_capital == 10000.0 and sc.executed_trades == 1, sc
    assert sc.total_net_pnl == 100.0, sc


# ---- 2. Headline prereg fixtures: '10000' and None --------------------------

def test_string_capital_raises_named_valueerror_not_typeerror():
    msg = _raises(STRING_CAP)
    assert msg is not None, "cap='10000' returned a scorecard"
    assert "INVALID_STARTING_CAPITAL" in msg, msg
    assert "'10000'" in msg, msg  # error names the offending value
    # The exception must be a ValueError, never the bare TypeError:
    # _raises catches only ValueError, so a TypeError would already
    # have propagated; assert the type explicitly for the record.
    try:
        ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED,
                                     starting_capital=STRING_CAP)
    except TypeError as e:  # pragma: no cover - pre-fix mode
        raise AssertionError(f"bare TypeError leaked: {e}")
    except ValueError:
        pass


def test_none_capital_raises_named_valueerror_not_typeerror():
    msg = _raises(NONE_CAP)
    assert msg is not None, "cap=None returned a scorecard"
    assert "INVALID_STARTING_CAPITAL" in msg, msg
    assert "None" in msg, msg  # error names the offending value
    try:
        ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED,
                                     starting_capital=NONE_CAP)
    except TypeError as e:  # pragma: no cover - pre-fix mode
        raise AssertionError(f"bare TypeError leaked: {e}")
    except ValueError:
        pass


def test_gate_bypass_reproduces_bare_typeerrors_exactly():
    # Mechanism proof: bypass ONLY the entry gate call and the live
    # (otherwise current) code returns the exact bare TypeErrors
    # from the task rationale. Restores the gate in `finally`.
    orig = ak._validate_starting_capital
    ak._validate_starting_capital = lambda v: None  # noqa: E731
    try:
        clean = ak.evaluate_fleet_north_star(
            "fixture", [_trade()], RESOLVED, starting_capital=10000.0)
        assert clean.north_star_score == 0.1635, clean
        try:
            ak.evaluate_fleet_north_star(
                "fixture", [_trade()], RESOLVED, starting_capital='10000')
            raise AssertionError("bypassed '10000' did not raise")
        except TypeError as e:
            assert "can only concatenate str" in str(e), e
        try:
            ak.evaluate_fleet_north_star(
                "fixture", [_trade()], RESOLVED, starting_capital=None)
            raise AssertionError("bypassed None did not raise")
        except TypeError as e:
            assert "+=" in str(e) and "NoneType" in str(e), e
    finally:
        ak._validate_starting_capital = orig
    assert _raises('10000') is not None and _raises(None) is not None


# ---- 3. Full non-numeric / bad-value domain ---------------------------------

def test_all_non_numeric_capitals_raise_named_error():
    for cap in NON_NUMERIC_CAPS:
        msg = _raises(cap)
        assert msg is not None, f"starting_capital={cap!r} accepted"
        assert "INVALID_STARTING_CAPITAL" in msg, msg
        assert repr(cap) in msg, msg


def test_bad_value_capitals_share_the_same_named_error():
    for cap in BAD_VALUE_CAPS:
        msg = _raises(cap)
        assert msg is not None, f"starting_capital={cap!r} accepted"
        assert "INVALID_STARTING_CAPITAL" in msg, msg


def test_string_none_gate_fires_for_empty_and_skip_only_books():
    for cap in (STRING_CAP, NONE_CAP):
        assert _raises(cap, trades=[], resolved=[]) is not None
        skip = _trade(action=ak.TradeAction.SKIP, invested_capital=0.0,
                      net_pnl_bid_marked=0.0, skip_reason="FIXTURE")
        assert _raises(cap, trades=[skip]) is not None


def test_string_none_positional_capital_also_validated():
    for cap in (STRING_CAP, NONE_CAP):
        try:
            ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED, cap)
        except ValueError as e:
            assert "INVALID_STARTING_CAPITAL" in str(e), e
            continue
        raise AssertionError(f"positional starting_capital={cap!r} accepted")


def test_capital_gate_precedes_other_gates():
    # Bad capital + a bad trades container: the capital error must
    # win — the gate is the first statement of the scorer, so a
    # string/None capital can never reach any downstream walk.
    for cap in (STRING_CAP, NONE_CAP):
        try:
            ak.evaluate_fleet_north_star("fixture", None, RESOLVED,
                                         starting_capital=cap)
        except ValueError as e:
            assert "INVALID_STARTING_CAPITAL" in str(e), e
            continue
        raise AssertionError(f"capital gate did not precede container gate for {cap!r}")


def test_helper_scope():
    for bad in (STRING_CAP, NONE_CAP, True, False, '0', [], float("nan")):
        assert not ak._is_valid_starting_capital(bad), bad
        assert not ak._is_finite_number(bad), bad
    assert ak._is_valid_starting_capital(10000.0)
    assert ak._is_valid_starting_capital(10000)
    assert ak._is_valid_starting_capital(1)


# ---- 4. Clean-capital scorecard bit-identical --------------------------------

def test_valid_capital_gate_is_noop_byte_identical():
    orig = ak._validate_starting_capital
    live = {}
    for cap in (10000.0, 5000.0, 100_000.0):
        live[cap] = _json(ak.evaluate_fleet_north_star(
            "fixture", [_trade()], RESOLVED, starting_capital=cap))
    ak._validate_starting_capital = lambda v: None  # noqa: E731
    try:
        for cap, expected in live.items():
            got = _json(ak.evaluate_fleet_north_star(
                "fixture", [_trade()], RESOLVED, starting_capital=cap))
            assert got == expected, f"valid cap {cap!r}: gate changed the scorecard"
    finally:
        ak._validate_starting_capital = orig
    sc_int = ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED,
                                          starting_capital=10000)
    sc_float = ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED,
                                            starting_capital=10000.0)
    assert dataclasses.asdict(sc_int) == dataclasses.asdict(sc_float)
    assert sc_float.north_star_score == 0.1635


def test_valid_capitals_scale_and_no_upper_bound():
    sc = ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED,
                                      starting_capital=5000.0)
    assert sc.portfolio_return_pct == 2.0, sc
    assert sc.ending_account_value_bid_marked == 5100.0, sc
    assert sc.north_star_score == 0.1635, sc  # MDD 0: N_star cap-invariant here
    big = ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED,
                                       starting_capital=1e12)
    assert math.isfinite(big.north_star_score), big


def test_default_is_still_10000():
    sig = inspect.signature(ak.evaluate_fleet_north_star)
    assert sig.parameters["starting_capital"].default == 10_000.0, sig
    sc = ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED)
    assert sc.north_star_score == 0.1635, sc
    assert sc.ending_account_value_bid_marked == 10100.0, sc


def test_benchmark_deterministic_and_current_goldens():
    outs = []
    for _ in range(2):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ak.run_benchmark_simulation()
        outs.append(buf.getvalue())
    assert outs[0] == outs[1], "benchmark not deterministic across runs"
    cards = []
    decoder = json.JSONDecoder()
    idx = 0
    while True:
        start = outs[0].find('{\n  "fleet_name"', idx)
        if start < 0:
            break
        obj, end = decoder.raw_decode(outs[0][start:])
        cards.append(obj)
        idx = start + end
    assert len(cards) == 2, cards
    assert cards[0]["north_star_score"] == -0.1611, cards[0]
    assert cards[0]["ending_account_value_bid_marked"] == 9170.88, cards[0]
    assert cards[0]["max_drawdown_pct"] == 8.29, cards[0]
    assert cards[1]["north_star_score"] == -0.1499, cards[1]
    assert cards[1]["ending_account_value_bid_marked"] == 9326.36, cards[1]
    assert cards[1]["max_drawdown_pct"] == 21.76, cards[1]
    assert all(c["starting_capital"] == 10000.0 for c in cards), cards


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
