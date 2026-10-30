"""SF-BL-10850 verification tests — NaN starting_capital / masked MDD.

Pre-registered done criterion (TASK.md): "Done when
evaluate_fleet_north_star raises a named ValueError for any
starting_capital that is not a finite real > 0 (bool excluded),
including a >15%-loss fixture proving the NaN-masked drawdown penalty
can no longer differ from the clean-capital score, with valid-capital
scorecards byte-identical; kill if a caller census shows capital is
only ever a validated positive literal."

Kill condition NOT met (CALLER_CENSUS.md): callers pass computed /
loaded capitals — float(bankroll) (wave2/e8 northstar_tracker),
EQUITY / START / start / cap variables, 100_000 / 100000 literals —
with no validated-positive-literal invariant anywhere.

State found by this task: the mandated gate ALREADY exists — landed
by SF-BL-9437 (`_validate_starting_capital`, first statement of
`evaluate_fleet_north_star`, named error INVALID_STARTING_CAPITAL,
reusing `_is_finite_number` so bool / non-numeric / NaN / +-inf all
reject). SF-BL-9437 fixtured NaN only on a +$100 WINNING trade
(MDD 0.0% either way). This suite adds what 9437 did not fixture:
the prereg LOSING trade and the >15%-loss penalty fixture, the
bad-domain grid on a losing book, and a byte-identity proof that
the gate is a strict no-op for valid capitals (live vs gate-bypassed
scorecard JSON identical; bypassed NaN reproduces the masked-MDD
bug exactly, proving the gate is the only thing standing between
the corrupt input and the fabricated scorecard). No repo code
change was needed or made by this task.

Fixtures:
  prereg: invested $500, PnL -$400, resolved [(0.8, 0)], est_days 10
    clean 10000 -> MDD 4.0, ending 9600.0, ret -4.0, CWR -0.8,
                   ECE 0.8, penalty 0.0, N_star -0.0016
    NaN pre-fix (bypassed) -> RETURNED MDD 0.0, ending NaN, ret NaN
  >15%: invested $2000, PnL -$2000, same panel
    clean 10000 -> MDD 20.0, penalty 0.1, ending 8000.0, ret -20.0,
                   N_star -0.102
    NaN pre-fix (bypassed) -> RETURNED MDD 0.0, penalty 0.0,
                   N_star -0.002 (the full 0.10 penalty masked)

Run:  python3 test_alpha_kelly_nan_starting_capital_mdd.py
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
NAN = float("nan")
INF = float("inf")
RESOLVED = [(0.8, 0)]
BAD_CAPS = (NAN, INF, -INF, 0, 0.0, -0.0, -10000, -10000.0, -1,
            True, False, "10000", "nan", None)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _trade(**over):
    base = dict(
        market_id="FIXTURE-10850", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C10850",
        action=ak.TradeAction.BUY_YES,
        calibrated_prob_yes=0.80, market_mid_prob=0.50, exec_price=0.50,
        liquidation_bid_price=0.50, kl_divergence_nats=0.0,
        expected_log_growth=0.0, kelly_fraction_raw=0.10,
        allocated_fraction=0.05, invested_capital=500.0,
        shares_acquired=1000.0, est_days_to_resolution=10.0,
        is_settled=True, outcome_yes=0, net_pnl_bid_marked=-400.0)
    base.update(over)
    return ak.TradeExecutionResult(**base)


def _big():
    return _trade(invested_capital=2000.0, net_pnl_bid_marked=-2000.0)


def _raises(cap, trades=None, resolved=RESOLVED):
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

def test_prereg_clean_fixture_exact():
    sc = ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED,
                                      starting_capital=10000)
    assert sc.max_drawdown_pct == 4.0, sc
    assert sc.ending_account_value_bid_marked == 9600.0, sc
    assert sc.portfolio_return_pct == -4.0, sc
    assert sc.cwr_bid_marked == -0.8 and sc.ece_10bin == 0.8, sc
    assert sc.drawdown_penalty == 0.0 and sc.north_star_score == -0.0016, sc
    assert sc.starting_capital == 10000 and sc.executed_trades == 1, sc


def test_prereg_nan_raises_named_error_not_masked_scorecard():
    msg = _raises(NAN)
    assert msg is not None, "cap=NaN returned a scorecard (pre-fix: MDD 0.0%, ending NaN)"
    assert "INVALID_STARTING_CAPITAL" in msg, msg
    assert "nan" in msg, msg  # error names the offending value


# ---- 2. >15%-loss fixture: the drawdown PENALTY can no longer differ --------

def test_big_loss_clean_fixture_exact():
    sc = ak.evaluate_fleet_north_star("fixture", [_big()], RESOLVED,
                                      starting_capital=10000.0)
    assert sc.max_drawdown_pct == 20.0, sc
    assert sc.drawdown_penalty == 0.1, sc  # 2.0 * (0.20 - 0.15)
    assert sc.ending_account_value_bid_marked == 8000.0, sc
    assert sc.portfolio_return_pct == -20.0, sc
    assert sc.north_star_score == -0.102, sc


def test_big_loss_nan_raises_so_penalty_cannot_differ():
    # There is no NaN scorecard at all, so no NaN penalty (masked 0.0
    # pre-fix) can differ from the clean penalty 0.1.
    msg = _raises(NAN, trades=[_big()])
    assert msg is not None and "INVALID_STARTING_CAPITAL" in msg, msg


def test_gate_bypass_reproduces_masked_penalty_exactly():
    # Mechanism proof: bypass ONLY the entry gate call and the live
    # (otherwise current) code returns the corrupt scorecards from
    # the task rationale. Restores the gate in `finally`.
    orig = ak._validate_starting_capital
    ak._validate_starting_capital = lambda v: None  # noqa: E731
    try:
        sc = ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED,
                                          starting_capital=NAN)
        assert sc.max_drawdown_pct == 0.0, sc  # clean reads 4.0
        assert math.isnan(sc.ending_account_value_bid_marked), sc
        assert math.isnan(sc.portfolio_return_pct), sc
        big = ak.evaluate_fleet_north_star("fixture", [_big()], RESOLVED,
                                           starting_capital=NAN)
        assert big.max_drawdown_pct == 0.0 and big.drawdown_penalty == 0.0, big
        assert big.north_star_score == -0.002, big  # clean: -0.102
        clean = ak.evaluate_fleet_north_star("fixture", [_big()], RESOLVED,
                                             starting_capital=10000.0)
        assert clean.drawdown_penalty - big.drawdown_penalty == 0.1, (clean, big)
    finally:
        ak._validate_starting_capital = orig
    assert _raises(NAN) is not None  # gate restored


# ---- 3. Full bad domain on the LOSING book ----------------------------------

def test_all_bad_capitals_raise_named_error_on_losing_book():
    for cap in BAD_CAPS:
        msg = _raises(cap)
        assert msg is not None, f"starting_capital={cap!r} accepted on losing book"
        assert "INVALID_STARTING_CAPITAL" in msg, msg
        assert repr(cap) in msg, msg
        msg_big = _raises(cap, trades=[_big()])
        assert msg_big is not None, f"starting_capital={cap!r} accepted on >15% book"


def test_nan_gate_fires_for_empty_and_skip_only_books():
    assert _raises(NAN, trades=[], resolved=[]) is not None
    skip = _trade(action=ak.TradeAction.SKIP, invested_capital=0.0,
                  net_pnl_bid_marked=0.0, skip_reason="FIXTURE")
    assert _raises(NAN, trades=[skip]) is not None


def test_nan_positional_capital_also_validated():
    try:
        ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED, NAN)
    except ValueError as e:
        assert "INVALID_STARTING_CAPITAL" in str(e), e
        return
    raise AssertionError("positional starting_capital=NaN accepted")


def test_helper_scope():
    assert not ak._is_valid_starting_capital(NAN)
    assert not ak._is_valid_starting_capital(True)
    assert ak._is_valid_starting_capital(10000.0)
    assert ak._is_valid_starting_capital(1)


# ---- 4. Valid-capital scorecards byte-identical ------------------------------

def test_valid_capital_gate_is_noop_byte_identical():
    # Live (gated) vs gate-bypassed JSON for valid capitals must be
    # byte-identical — the gate changes nothing for any valid book,
    # at the prereg capital, int capital, and two other capitals.
    orig = ak._validate_starting_capital
    live = {}
    for cap in (10000.0, 5000.0, 100_000.0):
        live[cap] = (
            _json(ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED,
                                               starting_capital=cap)),
            _json(ak.evaluate_fleet_north_star("fixture", [_big()], RESOLVED,
                                               starting_capital=cap)))
    ak._validate_starting_capital = lambda v: None  # noqa: E731
    try:
        for cap, expected in live.items():
            got = (
                _json(ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED,
                                                   starting_capital=cap)),
                _json(ak.evaluate_fleet_north_star("fixture", [_big()], RESOLVED,
                                                   starting_capital=cap)))
            assert got == expected, f"valid cap {cap!r}: gate changed the scorecard"
    finally:
        ak._validate_starting_capital = orig
    # int capital scores identically to float capital (dict equality:
    # 10000 == 10000.0; their JSON spellings differ only in the
    # starting_capital field's int/float rendering).
    sc_int = ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED,
                                          starting_capital=10000)
    sc_float = ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED,
                                            starting_capital=10000.0)
    assert dataclasses.asdict(sc_int) == dataclasses.asdict(sc_float)


def test_valid_capitals_scale_and_no_upper_bound():
    sc = ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED,
                                      starting_capital=5000.0)
    assert sc.max_drawdown_pct == 8.0 and sc.portfolio_return_pct == -8.0, sc
    assert sc.ending_account_value_bid_marked == 4600.0, sc
    big = ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED,
                                       starting_capital=1e12)
    assert math.isfinite(big.north_star_score), big


def test_default_is_still_10000():
    sig = inspect.signature(ak.evaluate_fleet_north_star)
    assert sig.parameters["starting_capital"].default == 10_000.0, sig
    sc = ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED)
    assert sc.max_drawdown_pct == 4.0 and sc.ending_account_value_bid_marked == 9600.0, sc


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
    # Current goldens at this task's run (valid capital 10000; the
    # disciplined card has moved since SF-BL-9437 due to intervening
    # unrelated gate tasks — this task changes no code, so identity
    # vs the gate-bypass run above is the byte-identity evidence).
    assert cards[0]["north_star_score"] == -0.1611, cards[0]
    assert cards[0]["ending_account_value_bid_marked"] == 9170.88, cards[0]
    assert cards[0]["max_drawdown_pct"] == 8.29, cards[0]
    assert cards[1]["north_star_score"] == -0.1499, cards[1]
    assert cards[1]["ending_account_value_bid_marked"] == 9326.36, cards[1]
    assert cards[1]["max_drawdown_pct"] == 21.76, cards[1]
    assert cards[1]["drawdown_penalty"] == 0.1352, cards[1]
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
