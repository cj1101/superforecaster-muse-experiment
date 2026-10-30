"""
test_alpha_kelly_starting_capital.py — SF-BL-9437 verification tests.

Pre-registered done criterion: "Require starting_capital > 0 and finite
at scorecard entry with a named error, fixtured at 0, -10000 and NaN
while the 8-market benchmark scorecard reproduces exactly; kill if all
callers are shown to pass the literal 10000 default (then pin the
default as a tested constant)."

Kill condition is NOT met (see CALLER_CENSUS.md in the task work dir):
fleet callers pass 100_000.0 / 100000.0 literals and EQUITY / cap /
start / dict-value variables; the benchmark passes a variable; repo
tests omit the argument. No caller invariant exists, so the fix
applies: _validate_starting_capital at the entry of
evaluate_fleet_north_star, raising ValueError
INVALID_STARTING_CAPITAL naming the value and SF-BL-9437, reusing
_is_finite_number (SF-BL-9438) so bool / non-numeric also reject.

Pre-fix reproduction (verified live 2026-10-23, before the patch;
fixture: one BUY_YES trade, $500 invested, PnL +100, est_days 10,
resolved [(0.6,1),(0.6,0)]; honest control N_star 0.4445):
  cap=0       -> ZeroDivisionError: float division by zero
  cap=-10000  -> RETURNED ret%=-1.0 (sign flipped) mdd%=0.0 ending=-9900
  cap=-10000, PnL -500 -> RETURNED ret%=+5.0 mdd%=0.0 (insolvent book
                 certified drawdown-free by the peak>0 guard)
  cap=NaN     -> RETURNED ret%=nan ending=nan mdd%=0.0
  cap=+-inf   -> RETURNED ending=+-inf
  cap='10000'/None -> bare TypeError; cap=True -> RETURNED ret%=10000.0

Benchmark golden is the CURRENT pre-fix output (post all prior tasks):
disciplined N_star 0.0547 / ending 11123.65 / CWR 0.2840, naive
N_star -0.1499 / ending 9326.36 / CWR -0.0784 — full stdout diffed
bit-identical in the task work dir (benchmark_prefix/postfix.txt).

Run:  python3 test_alpha_kelly_starting_capital.py  (exit 0 iff pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_north_star_finiteness.py.
"""
import io
import contextlib
import inspect
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
BAD_CAPS = (0, 0.0, -0.0, -10000, -10000.0, -1, NAN, INF, -INF,
            "10000", "0", None, True, False)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _trade(**over):
    base = dict(
        market_id="FIXTURE-9437", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C9437",
        action=ak.TradeAction.BUY_YES,
        calibrated_prob_yes=0.60, market_mid_prob=0.50, exec_price=0.50,
        liquidation_bid_price=0.50, kl_divergence_nats=0.0,
        expected_log_growth=0.0, kelly_fraction_raw=0.10,
        allocated_fraction=0.05, invested_capital=500.0,
        shares_acquired=1000.0, est_days_to_resolution=10.0,
        is_settled=True, outcome_yes=1, net_pnl_bid_marked=100.0)
    base.update(over)
    return ak.TradeExecutionResult(**base)


def _raises(cap, trades=None, resolved=RESOLVED):
    if trades is None:
        trades = [_trade()]
    try:
        ak.evaluate_fleet_north_star("fixture", trades, resolved,
                                     starting_capital=cap)
    except ValueError as e:
        return str(e)
    return None


# ---- 1. Prereg headline fixtures: 0, -10000, NaN ----------------------------

def test_zero_capital_raises_named_error_not_zerodivision():
    msg = _raises(0)
    assert msg is not None, "cap=0 did not raise ValueError (pre-fix: ZeroDivisionError)"
    assert "INVALID_STARTING_CAPITAL" in msg and "SF-BL-9437" in msg, msg
    assert _raises(0.0) is not None


def test_negative_capital_raises_not_zero_drawdown_scorecard():
    msg = _raises(-10000)
    assert msg is not None, "cap=-10000 returned a scorecard (pre-fix: MDD 0.0%)"
    losing = _raises(-10000.0, trades=[_trade(net_pnl_bid_marked=-500.0)])
    assert losing is not None, "cap=-10000 losing book returned (pre-fix: ret +5.0%, MDD 0.0%)"


def test_nan_capital_raises_not_nan_scorecard():
    msg = _raises(NAN)
    assert msg is not None, "cap=NaN returned a scorecard (pre-fix: ret/ending NaN)"


# ---- 2. Full bad-domain grid -------------------------------------------------

def test_all_bad_capitals_raise_named_error():
    for cap in BAD_CAPS:
        msg = _raises(cap)
        assert msg is not None, f"starting_capital={cap!r} accepted"
        assert "INVALID_STARTING_CAPITAL" in msg and "SF-BL-9437" in msg, msg
        assert repr(cap) in msg, msg


def test_error_message_names_value():
    msg = _raises(-10000)
    assert "-10000" in msg, msg
    msg = _raises(NAN)
    assert "nan" in msg, msg


def test_gate_fires_for_empty_and_skip_only_books():
    # Entry gate: unconditional — pre-fix cap=0 crashed the return
    # division even with no executed trades.
    assert _raises(0, trades=[], resolved=[]) is not None
    assert _raises(-10000, trades=[], resolved=[]) is not None
    skip = _trade(action=ak.TradeAction.SKIP, invested_capital=0.0,
                  net_pnl_bid_marked=0.0, skip_reason="FIXTURE")
    assert _raises(NAN, trades=[skip]) is not None


def test_gate_precedes_other_gates():
    # Entry means entry: a bad capital is named even when the book also
    # carries a bad trade field (SF-BL-9475) or an empty panel (D3).
    msg = _raises(0, trades=[_trade(net_pnl_bid_marked=NAN)])
    assert msg is not None and "INVALID_STARTING_CAPITAL" in msg, msg
    msg = _raises(-1, resolved=[])
    assert msg is not None and "INVALID_STARTING_CAPITAL" in msg, msg


def test_positional_capital_also_validated():
    try:
        ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED, 0)
    except ValueError as e:
        assert "INVALID_STARTING_CAPITAL" in str(e), e
        return
    raise AssertionError("positional starting_capital=0 accepted")


# ---- 3. Valid behaviour unchanged ---------------------------------------------

def test_control_fixture_exact():
    sc = ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED)
    assert sc.north_star_score == 0.4445, sc
    assert sc.ending_account_value_bid_marked == 10100.0, sc
    assert sc.portfolio_return_pct == 1.0 and sc.max_drawdown_pct == 0.0, sc
    assert sc.cwr_bid_marked == 0.2 and sc.executed_trades == 1, sc
    assert sc.starting_capital == 10000.0, sc


def test_default_is_still_10000():
    sig = inspect.signature(ak.evaluate_fleet_north_star)
    assert sig.parameters["starting_capital"].default == 10_000.0, sig


def test_positive_capitals_unchanged_and_scale():
    # Tiny, int, and huge positive capitals: no upper bound preregistered.
    tiny = ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED,
                                        starting_capital=0.5)
    assert tiny.portfolio_return_pct == 20000.0, tiny
    assert tiny.ending_account_value_bid_marked == 100.5, tiny
    one = ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED,
                                       starting_capital=1)
    assert one.portfolio_return_pct == 10000.0 and one.starting_capital == 1, one
    big = ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED,
                                       starting_capital=1e12)
    assert math.isfinite(big.north_star_score) and big.portfolio_return_pct == 0.0, big
    # Return scales exactly with capital; N_star / MDD do not depend on
    # the (valid) capital level for this fixture.
    alt = ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED,
                                       starting_capital=100_000.0)
    assert alt.portfolio_return_pct == 0.1 and alt.north_star_score == 0.4445, alt
    assert alt.ending_account_value_bid_marked == 100100.0, alt


def test_valid_capital_drawdown_still_computed():
    # A real drawdown on a valid capital still registers (the fix did
    # not touch the peak>0 walk for positive peaks).
    trades = [_trade(net_pnl_bid_marked=1000.0),
              _trade(market_id="B", net_pnl_bid_marked=-500.0)]
    sc = ak.evaluate_fleet_north_star("fixture", trades, RESOLVED,
                                      starting_capital=10000.0)
    assert sc.max_drawdown_pct == 4.55, sc  # (11000-10500)/11000


def test_empty_book_valid_capital_unchanged():
    sc = ak.evaluate_fleet_north_star("fixture", [], RESOLVED)
    assert sc.north_star_score == 0.0 and sc.executed_trades == 0, sc
    sc = ak.evaluate_fleet_north_star("fixture", [], [], starting_capital=500.0)
    assert sc.north_star_score == 0.0 and sc.starting_capital == 500.0, sc


def test_helper_scope():
    assert ak._is_valid_starting_capital(10000.0)
    assert ak._is_valid_starting_capital(1)
    for bad in BAD_CAPS:
        assert not ak._is_valid_starting_capital(bad), bad
    ak._validate_starting_capital(10000.0)  # no raise


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
    assert cards[0]["north_star_score"] == 0.0547, cards[0]
    assert cards[1]["north_star_score"] == -0.1499, cards[1]
    assert cards[0]["ending_account_value_bid_marked"] == 11123.65, cards[0]
    assert cards[1]["ending_account_value_bid_marked"] == 9326.36, cards[1]
    assert cards[0]["cwr_bid_marked"] == 0.284, cards[0]
    assert cards[1]["cwr_bid_marked"] == -0.0784, cards[1]
    assert cards[0]["starting_capital"] == 10000.0, cards[0]
    assert cards[0]["max_drawdown_pct"] == 6.94, cards[0]
    assert cards[1]["max_drawdown_pct"] == 21.76, cards[1]


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
