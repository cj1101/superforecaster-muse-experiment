"""
test_alpha_kelly_resolved_forecasts_container.py — SF-BL-11049 verification.

Pre-registered done criterion: "evaluate_fleet_north_star raises a named
ValueError for a non-iterable or malformed-entry all_resolved_forecasts,
fixtured at 123, 'abc' and a dict entry, with valid resolved sets
bit-identical and the empty-set D3 rejection unchanged; kill if a caller
census proves the resolved set is only ever built by validated
settlement paths."

Pre-fix reproduction (verified live 2026-10-26, before the patch;
fixture: one BUY_YES trade, $500 invested, PnL +$490, est_days 10):
  control [(0.8, 1)]  -> RETURNED N_star 0.8013 (ece 0.2, brier 0.04)
  forecasts=123       -> bare TypeError: 'int' object is not iterable
                         (enumerate() in _validate_resolved_forecast_panel)
  forecasts='abc'     -> mis-reported as entry 'a' (container iterated
                         char by char); pre-SF-BL-11535 a bare unpack
                         ValueError inside compute_empirical_ece_and_brier
  [{'p': 0.6, 'o': 1}] -> mis-reported as forecast='p' (dict unpacks to
                         keys); [{0.8: 'x', 1: 'y'}] SILENTLY SCORED
                         0.8013, bit-identical to the genuine panel
  forecasts=0, no trades -> silently RETURNED N_star 0.0
  generator of valid pairs -> bare TypeError: no len()

Kill condition NOT met (see CALLER_CENSUS.md): the only in-repo
production caller appends bare (cal_p, outcome_yes) literals to a plain
mutable list; settle_or_mark_trade never writes the panel; the
List[Tuple[float, int]] annotation is unenforced; no schema layer
exists. The fix is a container gate at the scorer boundary
(_validate_resolved_forecasts_container, SF-BL-11049) materialising the
panel before SF-BL-11535's element gate, plus an explicit dict/str/
bytes entry rejection in that gate — genuine panels bit-identical,
None/empty still owned by the D3 guard (SF-BL-8898).

Run:  python3 test_alpha_kelly_resolved_forecasts_container.py
No pytest dependency: test_* functions with assert + __main__ runner.
"""
import io
import contextlib
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster/superforecaster"))
import alpha_kelly as ak  # noqa: E402

PASS = []
CONTROL = [(0.8, 1)]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _trade(**over):
    base = dict(
        market_id="FIXTURE-11049", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C11049",
        action=ak.TradeAction.BUY_YES,
        calibrated_prob_yes=0.80, market_mid_prob=0.50, exec_price=0.50,
        liquidation_bid_price=0.50, kl_divergence_nats=0.0,
        expected_log_growth=0.0, kelly_fraction_raw=0.10,
        allocated_fraction=0.05, invested_capital=500.0,
        shares_acquired=1000.0, est_days_to_resolution=10.0,
        is_settled=True, outcome_yes=1, net_pnl_bid_marked=490.0)
    base.update(over)
    return ak.TradeExecutionResult(**base)


def _named(panel, trades=None, needle="SF-BL-11049"):
    """Return the ValueError message, or None; fail hard on bare TypeError."""
    try:
        ak.evaluate_fleet_north_star(
            "fixture", [_trade()] if trades is None else trades, panel)
    except ValueError as e:
        msg = str(e)
        assert needle in msg, msg
        return msg
    except TypeError as e:
        raise AssertionError(f"bare TypeError leaked, not named ValueError: {e}")
    return None


# ---- 1. Prereg headline fixtures ---------------------------------------------

def test_control_fixture_exact_08013():
    sc = ak.evaluate_fleet_north_star("fixture", [_trade()], CONTROL)
    assert sc.north_star_score == 0.8013, sc
    assert sc.ece_10bin == 0.2 and sc.brier_score == 0.04, sc
    assert sc.cwr_bid_marked == 0.98 and sc.avg_lockup_days == 10.0, sc


def test_noniterable_123_raises_named_container():
    msg = _named(123)
    assert msg is not None, "forecasts=123 returned a scorecard"
    assert "INVALID_RESOLVED_FORECASTS_CONTAINER" in msg, msg
    assert "123" in msg and "int" in msg, msg


def test_string_container_abc_raises_named_container():
    msg = _named("abc")
    assert msg is not None, "forecasts='abc' returned a scorecard"
    assert "INVALID_RESOLVED_FORECASTS_CONTAINER" in msg, msg


def test_dict_entry_raises_named_not_misreported():
    msg = _named([{"p": 0.6, "o": 1}])
    assert msg is not None, "dict entry returned a scorecard"
    assert "INVALID_RESOLVED_FORECAST_ELEMENT" in msg, msg
    assert "[0]" in msg and "dict" in msg, msg
    # Must NOT be mis-reported as forecast='p' (the key-unpack artefact)
    assert "forecast='p'" not in msg, msg


# ---- 2. Extended container space ----------------------------------------------

def test_other_noniterable_containers_raise_named():
    for panel in (5.0, True, False, 0, object()):
        msg = _named(panel)
        assert msg is not None and "INVALID_RESOLVED_FORECASTS_CONTAINER" in msg, (panel, msg)


def test_string_bytes_dict_containers_raise_named():
    for panel in ("", "abc", b"abc", {"p": 0.6, "o": 1}, {}):
        msg = _named(panel)
        assert msg is not None and "INVALID_RESOLVED_FORECASTS_CONTAINER" in msg, (panel, msg)


def test_container_gate_fires_with_no_executed_trades():
    # Pre-fix forecasts=0 with no trades silently returned N_star 0.0.
    msg = _named(123, trades=[])
    assert msg is not None and "INVALID_RESOLVED_FORECASTS_CONTAINER" in msg, msg
    msg = _named(0, trades=[])
    assert msg is not None and "INVALID_RESOLVED_FORECASTS_CONTAINER" in msg, msg
    skip = _trade(action=ak.TradeAction.SKIP, invested_capital=0.0,
                  net_pnl_bid_marked=0.0, skip_reason="FIXTURE")
    msg = _named("abc", trades=[skip])
    assert msg is not None and "INVALID_RESOLVED_FORECASTS_CONTAINER" in msg, msg


def test_container_gate_fires_before_d3():
    # A malformed container + executed trades names the container,
    # not NO_ELIGIBLE_ROWS (the D3 guard owns only None/empty).
    msg = _named(123)
    assert "NO_ELIGIBLE_ROWS" not in msg, msg


# ---- 3. Malformed entries -------------------------------------------------------

def test_dict_entry_with_numeric_keys_no_longer_scores():
    # Pre-fix [{0.8: 'x', 1: 'y'}] scored 0.8013 — bit-identical to the
    # genuine [(0.8, 1)] panel — because unpacking yields the keys.
    msg = _named([{0.8: "x", 1: "y"}])
    assert msg is not None and "INVALID_RESOLVED_FORECAST_ELEMENT" in msg, msg
    msg = _named([{0.6: 0, 1: 0}])
    assert msg is not None and "INVALID_RESOLVED_FORECAST_ELEMENT" in msg, msg


def test_dict_entry_at_any_index_raises_with_index():
    msg = _named([(0.8, 1), {"p": 0.6, "o": 1}])
    assert msg is not None and "[1]" in msg, msg


def test_string_and_bytes_entries_raise_named():
    for panel in (["ab"], [b"ab"], ["a"]):
        msg = _named(panel)
        assert msg is not None and "INVALID_RESOLVED_FORECAST_ELEMENT" in msg, (panel, msg)
        assert "SF-BL-11535" in msg, msg  # 11535 substring compatibility


def test_other_malformed_entries_keep_11535_identity():
    for panel in ([123], [None], [(0.8,)], [(0.8, 1, 0)]):
        msg = _named(panel, needle="SF-BL-11535")
        assert msg is not None and "INVALID_RESOLVED_FORECAST_ELEMENT" in msg, (panel, msg)


# ---- 4. Valid panels bit-identical ----------------------------------------------

def test_genuine_containers_bit_identical():
    ref = ak.evaluate_fleet_north_star("fixture", [_trade()], CONTROL)
    for panel in (((0.8, 1),), {(0.8, 1)}, [[0.8, 1]]):
        sc = ak.evaluate_fleet_north_star("fixture", [_trade()], panel)
        assert asdict(sc) == asdict(ref), (panel, sc, ref)


def test_generator_panel_materialises_and_scores_identically():
    ref = ak.evaluate_fleet_north_star("fixture", [_trade()], CONTROL)
    sc = ak.evaluate_fleet_north_star(
        "fixture", [_trade()], (x for x in [(0.8, 1)]))
    assert asdict(sc) == asdict(ref), (sc, ref)
    # A generator is consumed exactly once, by the container gate.
    msg = _named((x for x in [{"p": 0.6}]))
    assert msg is not None and "INVALID_RESOLVED_FORECAST_ELEMENT" in msg, msg


def test_numeric_panels_unchanged():
    for panel, expected in (
            ([(0.6, 1), (0.6, 0)], None),
            ([(1, 1)], None),
            ([(0.0, 0)], None)):
        sc = ak.evaluate_fleet_north_star("fixture", [_trade()], panel)
        assert isinstance(sc.north_star_score, float), sc
    # The SF-BL-11535 control on its own fixture still scores exactly.
    ece, brier = ak.compute_empirical_ece_and_brier([(0.8, 1)], num_bins=10)
    assert round(ece, 4) == 0.2 and round(brier, 4) == 0.04, (ece, brier)


# ---- 5. Empty-set D3 rejection unchanged -----------------------------------------

def test_empty_and_none_d3_unchanged():
    msg = _named([], needle="NO_ELIGIBLE_ROWS")
    assert msg is not None and "NO_ELIGIBLE_ROWS" in msg, msg
    msg = _named(None, needle="NO_ELIGIBLE_ROWS")
    assert msg is not None and "NO_ELIGIBLE_ROWS" in msg, msg
    msg = _named((), needle="NO_ELIGIBLE_ROWS")
    assert msg is not None and "NO_ELIGIBLE_ROWS" in msg, msg
    # Zero-trade books with an empty panel still score the definitional 0.0.
    sc = ak.evaluate_fleet_north_star("fixture", [], [])
    assert sc.north_star_score == 0.0 and sc.executed_trades == 0, sc
    sc = ak.evaluate_fleet_north_star("fixture", [], None)
    assert sc.north_star_score == 0.0, sc


def test_downstream_gates_unchanged_for_genuine_panels():
    # SF-BL-11535 element identity preserved
    msg = _named([(True, 1)], needle="SF-BL-11535")
    assert msg is not None and "INVALID_RESOLVED_FORECAST_ELEMENT" in msg, msg
    msg = _named([("0.7", 1)], needle="SF-BL-11535")
    assert msg is not None and "INVALID_RESOLVED_FORECAST_ELEMENT" in msg, msg
    # SF-BL-9139 range gate still fires inside compute for genuine pairs
    try:
        ak.evaluate_fleet_north_star("fixture", [_trade()], [(1.5, 1)])
        raise AssertionError("out-of-range forecast scored")
    except ValueError:
        pass


# ---- 6. Benchmark + helper scope / kill evidence ----------------------------------

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
    # Current-tree baseline captured pre-fix this run (benchmark_prefix.txt)
    assert cards[0]["north_star_score"] == -0.1611, cards[0]
    assert cards[1]["north_star_score"] == -0.1499, cards[1]
    assert cards[0]["ending_account_value_bid_marked"] == 9170.88, cards[0]
    assert cards[1]["ending_account_value_bid_marked"] == 9326.36, cards[1]


def test_helper_scope():
    assert ak._validate_resolved_forecasts_container(None) is None
    assert ak._validate_resolved_forecasts_container([(0.8, 1)]) == [(0.8, 1)]
    assert ak._validate_resolved_forecasts_container(((0.8, 1),)) == [(0.8, 1)]
    assert ak._validate_resolved_forecasts_container([]) == []
    for bad in (123, "abc", b"abc", {"p": 0.6}):
        try:
            ak._validate_resolved_forecasts_container(bad)
        except ValueError as e:
            assert "INVALID_RESOLVED_FORECASTS_CONTAINER" in str(e)
            assert "SF-BL-11049" in str(e)
            continue
        raise AssertionError(f"helper accepted container {bad!r}")


def test_kill_fails_panel_built_from_bare_literals():
    # Kill evidence: the benchmark builds its panel by appending bare
    # literals to a plain list — no validated settlement path stands
    # between construction and scoring (CALLER_CENSUS.md). A panel
    # mutated after construction reaches the scorer unchecked, which
    # is exactly why the gate lives at the scorer boundary.
    panel = [(0.8, 1)]
    panel.append({"p": 0.6, "o": 1})  # post-construction mutation
    msg = _named(panel)
    assert msg is not None and "[1]" in msg, msg


def test_direct_compute_surface_untouched_scope():
    # SF-BL-11049 fixes the evaluate_fleet_north_star boundary only;
    # the direct compute_empirical_ece_and_brier container claim is
    # SF-BL-10912's separate scope and its numeric behaviour is kept.
    assert ak.compute_empirical_ece_and_brier([]) == (0.0, 0.0)
    ece, brier = ak.compute_empirical_ece_and_brier([(0.8, 1)])
    assert round(ece, 4) == 0.2 and round(brier, 4) == 0.04, (ece, brier)


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
