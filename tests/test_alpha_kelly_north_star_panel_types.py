"""
test_alpha_kelly_north_star_panel_types.py — SF-BL-11535 verification tests.

Pre-registered done criterion: "evaluate_fleet_north_star must raise a
named ValueError on any bool or non-numeric forecast or outcome in
all_resolved_forecasts, fixtured at [(True, 1)], [('0.7', 1)] and
[(0.7, '1')] against the 0.1129 control, with numeric panels
bit-identical; kill if [(True, 1)] still scores 2.2672 or a string
element still raises a bare TypeError."

Pre-fix reproduction (verified live 2026-10-22, before the patch;
fixture: one BUY_YES trade, $500 invested, 1000 shares, exec 0.50,
PnL +325, est_days 30):
  control [(0.7, 1)]   -> RETURNED N_star 0.1129 (ece 0.3, brier 0.09)
  [(True, 1)]          -> RETURNED N_star 2.2672 (bool read as certainty)
  [(0.7, True)]        -> RETURNED N_star 0.1129 (bool outcome read as 1)
  [('0.7', 1)]         -> bare TypeError (Brier subtraction)
  [(0.7, '1')]         -> bare TypeError (Brier subtraction)
  [(None, 1)]/(0.7,None) -> bare TypeError

Kill condition NOT met (see CALLER_CENSUS.md): no validated producer
stands between an external panel and the scorer — the benchmark panel
is built from bare literals, the List[Tuple[float, int]] annotation is
unenforced, panels are plain mutable lists, and no schema layer exists
in the repo. The fix is a panel-element gate at the top of
evaluate_fleet_north_star (_validate_resolved_forecast_panel, reusing
_is_finite_number: finite real, bool excluded), so bool / string /
None / non-finite elements raise a named ValueError while every
numeric panel (int elements included) passes bit-identically.

Run:  python3 test_alpha_kelly_north_star_panel_types.py  (exit 0 iff pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_north_star_finiteness.py.
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
NAN = float("nan")
INF = float("inf")
CONTROL = [(0.7, 1)]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _trade(**over):
    base = dict(
        market_id="FIXTURE-11535", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C11535",
        action=ak.TradeAction.BUY_YES,
        calibrated_prob_yes=0.70, market_mid_prob=0.50, exec_price=0.50,
        liquidation_bid_price=0.50, kl_divergence_nats=0.0,
        expected_log_growth=0.0, kelly_fraction_raw=0.10,
        allocated_fraction=0.05, invested_capital=500.0,
        shares_acquired=1000.0, est_days_to_resolution=30.0,
        is_settled=True, outcome_yes=1, net_pnl_bid_marked=325.0)
    base.update(over)
    return ak.TradeExecutionResult(**base)


def _named_value_error(panel, trades=None):
    """Return the ValueError message, or None; fail hard on a bare TypeError."""
    try:
        ak.evaluate_fleet_north_star(
            "fixture", [_trade()] if trades is None else trades, panel)
    except ValueError as e:
        return str(e)
    except TypeError as e:
        raise AssertionError(f"bare TypeError leaked, not named ValueError: {e}")
    return None


# ---- 1. Prereg headline fixtures ---------------------------------------------

def test_bool_forecast_raises_named_not_22672():
    msg = _named_value_error([(True, 1)])
    assert msg is not None, "[(True, 1)] returned a scorecard (pre-fix N_star 2.2672)"
    assert "INVALID_RESOLVED_FORECAST_ELEMENT" in msg and "SF-BL-11535" in msg, msg
    assert "forecast" in msg and "True" in msg, msg


def test_string_forecast_raises_named_not_bare_typeerror():
    msg = _named_value_error([("0.7", 1)])
    assert msg is not None, "[('0.7', 1)] returned a scorecard"
    assert "INVALID_RESOLVED_FORECAST_ELEMENT" in msg and "SF-BL-11535" in msg, msg


def test_string_outcome_raises_named_not_bare_typeerror():
    msg = _named_value_error([(0.7, "1")])
    assert msg is not None, "[(0.7, '1')] returned a scorecard"
    assert "INVALID_RESOLVED_FORECAST_ELEMENT" in msg and "outcome" in msg, msg


def test_control_fixture_exact_01129():
    sc = ak.evaluate_fleet_north_star("fixture", [_trade()], CONTROL)
    assert sc.north_star_score == 0.1129, sc
    assert sc.ece_10bin == 0.3 and sc.brier_score == 0.09, sc
    assert sc.cwr_bid_marked == 0.65 and sc.avg_lockup_days == 30.0, sc
    assert sc.executed_trades == 1 and sc.total_net_pnl == 325.0, sc


# ---- 2. Bool / non-numeric in either position, any position in panel ---------

def test_bool_outcome_and_false_variants_raise():
    for panel in ([(0.7, True)], [(0.7, False)], [(False, 0)], [(True, True)]):
        msg = _named_value_error(panel)
        assert msg is not None and "SF-BL-11535" in msg, (panel, msg)


def test_none_and_other_nonnumeric_raise():
    for panel in ([(None, 1)], [(0.7, None)], [([0.7], 1)], [(0.7, {"o": 1})],
                  [("1", 0)], [(0.7, "0")]):
        msg = _named_value_error(panel)
        assert msg is not None and "SF-BL-11535" in msg, (panel, msg)


def test_bad_element_in_later_position_raises_with_index():
    msg = _named_value_error([(0.7, 1), (0.5, 0), ("0.6", 1)])
    assert msg is not None and "[2]" in msg and "forecast" in msg, msg
    msg = _named_value_error([(0.7, 1), (0.5, "0")])
    assert msg is not None and "[1]" in msg and "outcome" in msg, msg


def test_nonfinite_elements_raise_named():
    # Via _is_finite_number (SF-BL-9438): pre-fix NaN forecast raised an
    # unnamed ValueError from int() binning, +inf an OverflowError, and a
    # NaN outcome silently returned NaN ECE/Brier in the scorecard.
    for panel in ([(NAN, 1)], [(INF, 1)], [(-INF, 0)], [(0.7, NAN)], [(0.7, INF)]):
        msg = _named_value_error(panel)
        assert msg is not None and "SF-BL-11535" in msg, (panel, msg)


def test_malformed_entry_raises_named_not_bare():
    for panel in ([(0.7,)], [0.7], ["ab"], [(0.7, 1, 0)]):
        msg = _named_value_error(panel)
        assert msg is not None and "SF-BL-11535" in msg, (panel, msg)


def test_bad_panel_raises_even_with_no_executed_trades():
    # The prereg is unconditional: any bool/non-numeric element in the
    # panel, not only panels attached to executed trades.
    msg = _named_value_error([(True, 1)], trades=[])
    assert msg is not None and "SF-BL-11535" in msg, msg
    skip = _trade(action=ak.TradeAction.SKIP, invested_capital=0.0,
                  net_pnl_bid_marked=0.0, skip_reason="FIXTURE")
    msg = _named_value_error([("0.7", 1)], trades=[skip])
    assert msg is not None and "SF-BL-11535" in msg, msg


def test_mutated_panel_also_raises():
    panel = [(0.7, 1)]
    panel.append((True, 0))  # post-construction mutation, no producer invariant
    msg = _named_value_error(panel)
    assert msg is not None and "[1]" in msg, msg


# ---- 3. Numeric panels bit-identical ------------------------------------------

def test_numeric_panels_bit_identical():
    # Expected values captured pre-fix (reproduce_prefix.txt + live runs).
    cases = {
        ((0.7, 1),): 0.1129,
        ((0.7, 0),): 0.0021,
        ((1, 1),): 2.2672,   # int forecast 1 is genuine numeric certainty
        ((0.7, 1.0),): 0.1129,  # float outcome
        ((0, 1), (1, 0), (0.5, 1), (0.25, 0)): None,  # checked via asdict below
    }
    for panel_tuple, expected in cases.items():
        panel = [tuple(p) for p in panel_tuple]
        sc = ak.evaluate_fleet_north_star("fixture", [_trade()], panel)
        if expected is not None:
            assert sc.north_star_score == expected, (panel, sc)
    # Full-scorecard identity for a mixed numeric panel, recomputed from
    # the ECE/Brier primitives the pre-fix code used (same float ops).
    panel = [(0, 1), (1, 0), (0.5, 1), (0.25, 0)]
    sc = ak.evaluate_fleet_north_star("fixture", [_trade()], panel)
    ece, brier = ak.compute_empirical_ece_and_brier(panel, num_bins=10)
    assert sc.ece_10bin == round(ece, 4) and sc.brier_score == round(brier, 4), sc
    import math
    assert sc.north_star_score == round(
        0.65 * math.sqrt(365.0 / 30.0) * math.exp(-10.0 * ece), 4), sc


def test_zero_and_boundary_numeric_panels_unchanged():
    for panel in ([(0.0, 0)], [(1.0, 1)], [(0.5, 0), (0.5, 1)]):
        sc = ak.evaluate_fleet_north_star("fixture", [_trade()], panel)
        assert isinstance(sc.north_star_score, float), sc


# ---- 4. Empty / D3 semantics unchanged -----------------------------------------

def test_empty_panel_d3_guard_unchanged():
    msg = _named_value_error([])
    assert msg is not None and "NO_ELIGIBLE_ROWS" in msg, msg
    msg = _named_value_error(None)
    assert msg is not None and "NO_ELIGIBLE_ROWS" in msg, msg
    sc = ak.evaluate_fleet_north_star("fixture", [], [])
    assert sc.north_star_score == 0.0 and sc.executed_trades == 0, sc


def test_skip_only_book_with_numeric_panel_unchanged():
    skip = _trade(action=ak.TradeAction.SKIP, invested_capital=0.0,
                  net_pnl_bid_marked=0.0, skip_reason="FIXTURE")
    sc = ak.evaluate_fleet_north_star("fixture", [skip], CONTROL)
    assert sc.executed_trades == 0 and sc.skipped_opportunities == 1, sc
    assert sc.north_star_score == 0.0, sc


# ---- 5. Benchmark scorecards bit-identical --------------------------------------

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
    # Values captured pre-fix this run (benchmark_prefix.txt).
    assert cards[0]["north_star_score"] == 0.0547, cards[0]
    assert cards[1]["north_star_score"] == -0.1499, cards[1]
    assert cards[0]["ending_account_value_bid_marked"] == 11123.65, cards[0]
    assert cards[1]["ending_account_value_bid_marked"] == 9326.36, cards[1]
    assert cards[0]["cwr_bid_marked"] == 0.284, cards[0]
    assert cards[1]["cwr_bid_marked"] == -0.0784, cards[1]


# ---- 6. Helper scope / kill evidence ---------------------------------------------

def test_helper_scope():
    ak._validate_resolved_forecast_panel([(0.7, 1), (1, 0), (0.0, 1.0)])  # no raise
    ak._validate_resolved_forecast_panel([])   # empty: no-op (D3 owns it)
    ak._validate_resolved_forecast_panel(None)  # None: no-op (D3 owns it)
    try:
        ak._validate_resolved_forecast_panel([(0.7, "1")])
    except ValueError as e:
        assert "SF-BL-11535" in str(e)
        return
    raise AssertionError("helper accepted a string outcome")


def test_direct_compute_surface_untouched_scope():
    # Scope guard: SF-BL-11535 fixes the evaluate_fleet_north_star
    # boundary only. The direct compute_empirical_ece_and_brier element
    # surface is SF-BL-11048's separate prereg; numeric direct calls
    # must keep working exactly as before.
    ece, brier = ak.compute_empirical_ece_and_brier([(0.7, 1)])
    assert round(ece, 4) == 0.3 and round(brier, 4) == 0.09, (ece, brier)
    assert ak.compute_empirical_ece_and_brier([]) == (0.0, 0.0)


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
