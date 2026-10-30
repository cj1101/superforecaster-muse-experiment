"""
test_alpha_kelly_trades_container.py — SF-BL-10968 verification tests.

Pre-registered done criterion: "evaluate_fleet_north_star must raise a
named ValueError when trades is None/non-iterable or any entry is not
a TradeExecutionResult, fixtured at None and a dict entry, with the
8-market benchmark scorecards bit-identical; kill if a caller census
shows trades only ever arrive as constructed TradeExecutionResult
lists under a tested invariant."

Pre-fix reproduction (verified live 2026-10-23, before the patch;
fixture: one BUY_YES trade, $500 invested, 1000 shares, exec 0.50,
PnL +100, est_days 10, resolved [(0.6,1),(0.6,0)]; honest control
N_star 0.4445):
  trades=None      -> bare TypeError: 'NoneType' object is not iterable
  trades=[dict]    -> bare AttributeError: 'dict' object has no
                      attribute 'action' (JSON-style stored record)
  trades=123/5.0/bare TradeExecutionResult -> bare TypeError
  [None]/['BUY_YES']/[123]/mixed -> bare AttributeError

Kill condition is NOT met (see CALLER_CENSUS.md): the only in-repo
production caller passes plain list variables with no check, the
List[TradeExecutionResult] annotation is unenforced, stored records
round-trip through JSON as plain dicts with no reconstruction layer,
direct callers pass bare literals, and the lists are mutable. So the
fix applies: a container/entry gate at the top of
evaluate_fleet_north_star (_validate_trades_container, reusing the
SF-BL-10971 isinstance helper), raising ValueError naming
INVALID_TRADES_CONTAINER / INVALID_TRADE_ENTRY and SF-BL-10968,
before the SF-BL-10823 action gate and SF-BL-9475 finiteness gate can
dereference any entry. Genuine list/tuple books pass bit-identically.

Run:  python3 test_alpha_kelly_trades_container.py  (exit 0 iff pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_north_star_finiteness.py.
"""
import io
import contextlib
import json
import os
import sys
from dataclasses import asdict
from types import SimpleNamespace

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster/superforecaster"))
import alpha_kelly as ak  # noqa: E402

PASS = []
RESOLVED = [(0.6, 1), (0.6, 0)]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _trade(**over):
    base = dict(
        market_id="FIXTURE-10968", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C10968",
        action=ak.TradeAction.BUY_YES,
        calibrated_prob_yes=0.60, market_mid_prob=0.50, exec_price=0.50,
        liquidation_bid_price=0.50, kl_divergence_nats=0.0,
        expected_log_growth=0.0, kelly_fraction_raw=0.10,
        allocated_fraction=0.05, invested_capital=500.0,
        shares_acquired=1000.0, est_days_to_resolution=10.0,
        is_settled=True, outcome_yes=1, net_pnl_bid_marked=100.0)
    base.update(over)
    return ak.TradeExecutionResult(**base)


def _named_value_error(trades, resolved=RESOLVED, **kw):
    """ValueError message, or None; fail hard on a bare Type/AttributeError."""
    try:
        ak.evaluate_fleet_north_star("fixture", trades, resolved, **kw)
    except ValueError as e:
        return str(e)
    except (TypeError, AttributeError) as e:
        raise AssertionError(f"bare {type(e).__name__} leaked, not named ValueError: {e}")
    return None


DICT_ENTRY = {"market_id": "FIXTURE-10968", "action": "BUY_YES",
              "invested_capital": 500.0, "net_pnl_bid_marked": 100.0,
              "est_days_to_resolution": 10.0}


# ---- 1. Prereg headline fixtures ---------------------------------------------

def test_none_trades_raises_named_not_typeerror():
    msg = _named_value_error(None)
    assert msg is not None, "trades=None returned a scorecard / bare crash"
    assert "INVALID_TRADES_CONTAINER" in msg and "SF-BL-10968" in msg, msg
    assert "None" in msg, msg


def test_dict_entry_raises_named_not_attributeerror():
    msg = _named_value_error([dict(DICT_ENTRY)])
    assert msg is not None, "dict entry returned a scorecard / bare crash"
    assert "INVALID_TRADE_ENTRY" in msg and "SF-BL-10968" in msg, msg
    assert "dict" in msg and "TradeExecutionResult" in msg, msg
    assert "[0]" in msg, msg


def test_control_fixture_exact_04445():
    sc = ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED)
    assert sc.north_star_score == 0.4445, sc
    assert sc.ending_account_value_bid_marked == 10100.0, sc
    assert sc.cwr_bid_marked == 0.2, sc
    assert sc.executed_trades == 1 and sc.total_opportunities == 1, sc


# ---- 2. Container space -------------------------------------------------------

def test_noniterable_containers_raise_named():
    for bad in (123, 5.0, True, object(), _trade()):
        msg = _named_value_error(bad)
        assert msg is not None and "INVALID_TRADES_CONTAINER" in msg \
            and "SF-BL-10968" in msg, (bad, msg)


def test_string_bytes_dict_containers_raise_named():
    for bad in ("not-a-trade-list", b"bytes", {"market_id": "X"}, b""):
        msg = _named_value_error(bad)
        assert msg is not None and "INVALID_TRADES_CONTAINER" in msg \
            and "SF-BL-10968" in msg, (bad, msg)


def test_container_gate_fires_before_d3_and_panel_gates():
    # Even with an empty resolved panel (D3 territory), a malformed
    # container/entry is named for what it is, not NO_ELIGIBLE_ROWS.
    msg = _named_value_error(None, resolved=[])
    assert msg is not None and "INVALID_TRADES_CONTAINER" in msg, msg
    msg = _named_value_error([dict(DICT_ENTRY)], resolved=[])
    assert msg is not None and "INVALID_TRADE_ENTRY" in msg, msg


def test_starting_capital_gate_still_precedes_trade_gates():
    # Documented order (unchanged): SF-BL-9437 capital gate at entry,
    # before every trade gate including this one.
    msg = _named_value_error(None, starting_capital=0)
    assert msg is not None and "INVALID_STARTING_CAPITAL" in msg, msg


# ---- 3. Entry space -----------------------------------------------------------

def test_non_trade_entries_raise_named():
    for entry in (None, "BUY_YES", 123, 0.5, True, ["x"], (1, 2), object()):
        msg = _named_value_error([entry])
        assert msg is not None and "INVALID_TRADE_ENTRY" in msg \
            and "SF-BL-10968" in msg, (entry, msg)


def test_bad_entry_at_any_index_raises_with_index():
    msg = _named_value_error([_trade(), _trade(market_id="B"), dict(DICT_ENTRY)])
    assert msg is not None and "[2]" in msg and "dict" in msg, msg
    msg = _named_value_error([None, _trade()])
    assert msg is not None and "[0]" in msg, msg
    msg = _named_value_error([_trade(), None])
    assert msg is not None and "[1]" in msg, msg


def test_json_roundtrip_dict_entry_raises_named():
    # The stored-record shape the rationale names: asdict + JSON.
    stored = json.loads(json.dumps(asdict(_trade()), default=str))
    assert isinstance(stored, dict)
    msg = _named_value_error([stored])
    assert msg is not None and "INVALID_TRADE_ENTRY" in msg \
        and "SF-BL-10968" in msg, msg


def test_duck_typed_fake_entry_raises_named():
    # A fake carrying every scored attribute must not pass on duck
    # typing (cf. SF-BL-10971): only a genuine TradeExecutionResult.
    fake = SimpleNamespace(**{k: getattr(_trade(), k) for k in (
        "market_id", "action", "invested_capital", "net_pnl_bid_marked",
        "est_days_to_resolution", "is_settled")})
    msg = _named_value_error([fake])
    assert msg is not None and "INVALID_TRADE_ENTRY" in msg, msg


def test_bad_entry_raises_even_in_skip_only_position():
    # The gate runs over EVERY entry, before the executed/SKIP
    # partition — a dict cannot hide behind a SKIP-shaped payload.
    msg = _named_value_error([_trade(action=ak.TradeAction.SKIP,
                                     invested_capital=0.0,
                                     net_pnl_bid_marked=0.0),
                              dict(DICT_ENTRY)])
    assert msg is not None and "INVALID_TRADE_ENTRY" in msg, msg


# ---- 4. Genuine books bit-identical -------------------------------------------

def test_genuine_containers_bit_identical():
    by_list = ak.evaluate_fleet_north_star("fixture", [_trade()], RESOLVED)
    by_tuple = ak.evaluate_fleet_north_star("fixture", (_trade(),), RESOLVED)
    assert asdict(by_list) == asdict(by_tuple), (by_list, by_tuple)
    # A one-shot generator materialises to the same book (pre-fix the
    # first validator exhausted it and the book scored as empty).
    by_gen = ak.evaluate_fleet_north_star(
        "fixture", (t for t in [_trade()]), RESOLVED)
    assert asdict(by_gen) == asdict(by_list), (by_gen, by_list)


def test_empty_and_skip_only_books_unchanged():
    sc = ak.evaluate_fleet_north_star("fixture", [], RESOLVED)
    assert sc.north_star_score == 0.0 and sc.executed_trades == 0, sc
    sc = ak.evaluate_fleet_north_star("fixture", [], [])
    assert sc.north_star_score == 0.0, sc
    skip = _trade(action=ak.TradeAction.SKIP, invested_capital=0.0,
                  net_pnl_bid_marked=0.0, skip_reason="FIXTURE")
    sc = ak.evaluate_fleet_north_star("fixture", [skip], RESOLVED)
    assert sc.executed_trades == 0 and sc.skipped_opportunities == 1, sc
    assert sc.north_star_score == 0.0, sc


def test_downstream_gates_unchanged_for_genuine_entries():
    # Container gate passes genuine records through to the existing
    # gates with their own error identities intact.
    msg = _named_value_error([_trade()], resolved=[])
    assert msg is not None and "NO_ELIGIBLE_ROWS" in msg, msg
    msg = _named_value_error([_trade(net_pnl_bid_marked=float("nan"))])
    assert msg is not None and "NON_FINITE_TRADE_FIELD" in msg \
        and "SF-BL-9475" in msg, msg
    # Bad action via bypass construction still hits SF-BL-10823.
    t = _trade()
    object.__setattr__(t, "action", "buy_yes")
    msg = _named_value_error([t])
    assert msg is not None and "INVALID_TRADE_ACTION" in msg, msg


def test_subclass_entry_accepted():
    class SubTrade(ak.TradeExecutionResult):
        pass
    base = _trade()
    sub = SubTrade(**{f: getattr(base, f) for f in base.__dataclass_fields__
                      if f not in ("cluster_exposure_used",
                                   "cluster_exposure_released")})
    sc = ak.evaluate_fleet_north_star("fixture", [sub], RESOLVED)
    assert sc.north_star_score == 0.4445, sc


# ---- 5. Benchmark scorecards bit-identical ------------------------------------

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
    # Current-tree baseline captured pre-fix this run
    # (benchmark_prefix.txt; full-output pre/post diff is empty — the
    # identity, not the drifted absolute, is the invariant).
    assert cards[0]["north_star_score"] == -0.1611, cards[0]
    assert cards[1]["north_star_score"] == -0.1499, cards[1]
    assert cards[0]["ending_account_value_bid_marked"] == 9170.88, cards[0]
    assert cards[1]["ending_account_value_bid_marked"] == 9326.36, cards[1]
    assert cards[0]["total_opportunities"] == 8, cards[0]
    assert cards[1]["total_opportunities"] == 8, cards[1]


# ---- 6. Helper scope / kill evidence -------------------------------------------

def test_helper_scope():
    out = ak._validate_trades_container([_trade()])
    assert isinstance(out, list) and len(out) == 1, out
    assert ak._validate_trades_container([]) == []
    for bad, token in ((None, "INVALID_TRADES_CONTAINER"),
                       (123, "INVALID_TRADES_CONTAINER"),
                       ("x", "INVALID_TRADES_CONTAINER"),
                       ([None], "INVALID_TRADE_ENTRY"),
                       ([dict(DICT_ENTRY)], "INVALID_TRADE_ENTRY")):
        try:
            ak._validate_trades_container(bad)
        except ValueError as e:
            assert token in str(e) and "SF-BL-10968" in str(e), str(e)
        else:
            raise AssertionError(f"helper accepted {bad!r}")


def test_kill_fails_stored_records_materialise_as_dicts():
    # Kill evidence in miniature: no invariant reconstructs a stored
    # record — the JSON shape IS a dict, and pre-fix it crashed bare.
    stored = json.loads(json.dumps(asdict(_trade()), default=str))
    assert type(stored) is dict and not isinstance(
        stored, ak.TradeExecutionResult)
    assert _named_value_error([stored]) is not None


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
