"""
test_alpha_kelly_ece_range_validation.py — SF-BL-9139 verification tests.

Pre-registered done criterion: "Conformance fixture feeding p in
{-0.2, -0.05, 1.2} and o in {2} must raise or be excluded with a logged
count before any fleet ECE is quoted; success if >=1 production forecast
path can emit such a row, kill if a caller census proves every input is
range-validated upstream."

Pre-fix reproduction (verified live 2026-10-22, before the patch;
reproduce_hazard.py in the SF-BL-9139 work dir):
  [(-0.2, 1)]  -> RETURNED ECE 1.2000 Brier 1.4400 (idx -2 -> bin 8)
  [(-0.05, 0)] -> RETURNED ECE 0.0500 Brier 0.0025 (int truncates to bin 0)
  [(-0.10, 0)] -> RETURNED (idx -1 -> bins[-1], the TOP bin 9)
  [(1.2, 1)]   -> RETURNED ECE 0.2000 Brier 0.0400 (min() clamps to bin 9)
  [(0.7, 2)]   -> RETURNED ECE 1.3000 Brier 1.6900 (avg_o = 2)
  combined prereg panel -> RETURNED ECE 0.6100 Brier 0.6525
  via evaluate_fleet_north_star (one BUY_YES trade, $500 invested,
  PnL +$325, est_days 30): [(1.2, 1)] scored N_star 0.3068 vs the
  0.1129 control [(0.7, 1)] — the malformed row IMPROVED the score.

Kill condition NOT met (CALLER_CENSUS.md): the sole in-repo caller
evaluate_fleet_north_star validates element types/finiteness
(SF-BL-11535) but not ranges; the benchmark's outcomes are bare
literals; panels are plain mutable lists fed by fleet wave scripts
with no range validation and no schema layer. The fix is a range /
domain gate at the top of compute_empirical_ece_and_brier
(_validate_ece_panel_ranges, reusing _is_finite_number), so any
forecast outside [0, 1] or outcome not in {0, 1} raises a named
ValueError before any Brier sum or binning, while every in-range
panel — boundaries included — passes bit-identically.

Run:  python3 test_alpha_kelly_ece_range_validation.py  (exit 0 iff pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_north_star_panel_types.py.
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
        market_id="FIXTURE-9139", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C9139",
        action=ak.TradeAction.BUY_YES,
        calibrated_prob_yes=0.70, market_mid_prob=0.50, exec_price=0.50,
        liquidation_bid_price=0.50, kl_divergence_nats=0.0,
        expected_log_growth=0.0, kelly_fraction_raw=0.10,
        allocated_fraction=0.05, invested_capital=500.0,
        shares_acquired=1000.0, est_days_to_resolution=30.0,
        is_settled=True, outcome_yes=1, net_pnl_bid_marked=325.0)
    base.update(over)
    return ak.TradeExecutionResult(**base)


def _direct_error(panel, num_bins=10):
    """ValueError message from compute directly, or None; bare TypeError fails hard."""
    try:
        ak.compute_empirical_ece_and_brier(panel, num_bins=num_bins)
    except ValueError as e:
        return str(e)
    except TypeError as e:
        raise AssertionError(f"bare TypeError leaked, not named ValueError: {e}")
    return None


def _fleet_error(panel):
    try:
        ak.evaluate_fleet_north_star("fixture", [_trade()], panel)
    except ValueError as e:
        return str(e)
    except TypeError as e:
        raise AssertionError(f"bare TypeError leaked, not named ValueError: {e}")
    return None


# ---- 1. Prereg headline fixtures, direct surface -----------------------------

def test_prereg_negative_forecasts_raise_named():
    for p in (-0.2, -0.05):
        msg = _direct_error([(p, 1)])
        assert msg is not None, f"p={p} returned a score (pre-fix silent mis-bin)"
        assert "OUT_OF_RANGE_FORECAST" in msg and "SF-BL-9139" in msg, msg
        assert "forecast" in msg and repr(p) in msg, msg


def test_prereg_over_one_forecast_raises_named():
    msg = _direct_error([(1.2, 1)])
    assert msg is not None, "p=1.2 returned a score (pre-fix clamped to top bin)"
    assert "OUT_OF_RANGE_FORECAST" in msg and "SF-BL-9139" in msg, msg


def test_prereg_outcome_two_raises_named():
    msg = _direct_error([(0.7, 2)])
    assert msg is not None, "o=2 returned a score (pre-fix ECE 1.3)"
    assert "INVALID_OUTCOME" in msg and "SF-BL-9139" in msg, msg
    assert "outcome" in msg, msg


def test_prereg_combined_panel_raises():
    msg = _direct_error([(-0.2, 1), (-0.05, 0), (1.2, 1), (0.7, 2), (0.7, 1)])
    assert msg is not None and "SF-BL-9139" in msg, msg


def test_wrap_boundary_forecasts_raise():
    # p=-0.10 / -0.15 pre-fix wrapped via bins[-1] into the TOP bin.
    for p in (-0.10, -0.15, -1.0, -100.0, 1.0000001, 2.0, 100.0):
        msg = _direct_error([(p, 0)])
        assert msg is not None and "OUT_OF_RANGE_FORECAST" in msg, (p, msg)


def test_nonbinary_outcomes_raise():
    for o in (2, -1, 0.5, 1.5, 100):
        msg = _direct_error([(0.7, o)])
        assert msg is not None and "INVALID_OUTCOME" in msg, (o, msg)


def test_bad_row_in_later_position_raises_with_index():
    msg = _direct_error([(0.7, 1), (0.3, 0), (1.2, 1)])
    assert msg is not None and "[2]" in msg, msg
    msg = _direct_error([(0.7, 1), (0.3, 2)])
    assert msg is not None and "[1]" in msg and "INVALID_OUTCOME" in msg, msg


def test_nonfinite_and_nonnumeric_raise_named_not_bare():
    # Same gate via _is_finite_number: previously NaN died unnamed inside
    # int(), +inf raised OverflowError, strings died with bare TypeError.
    for panel in ([(NAN, 1)], [(INF, 1)], [(-INF, 0)], [(0.7, NAN)],
                  [("0.7", 1)], [(0.7, "1")], [(None, 1)], [(0.7, None)],
                  [(True, 1)], [(0.7, True)]):
        msg = _direct_error(panel)
        assert msg is not None and "SF-BL-9139" in msg, (panel, msg)


def test_malformed_entry_raises_named_not_bare():
    for panel in ([(0.7,)], [(0.7, 1, 0.5)], [0.7], [None]):
        msg = _direct_error(panel)
        assert msg is not None and "SF-BL-9139" in msg, (panel, msg)


def test_custom_num_bins_still_gated():
    msg = _direct_error([(1.2, 1)], num_bins=5)
    assert msg is not None and "OUT_OF_RANGE_FORECAST" in msg, msg
    ece, brier = ak.compute_empirical_ece_and_brier([(0.7, 1)], num_bins=5)
    assert abs(ece - 0.3) < 1e-12 and abs(brier - 0.09) < 1e-12, (ece, brier)


# ---- 2. Fleet surface: the gate fires before any fleet ECE is quoted ---------

def test_fleet_bad_panels_raise_named():
    for panel, tag in ([(1.2, 1)], "OUT_OF_RANGE_FORECAST"), \
                      ([(-0.2, 1)], "OUT_OF_RANGE_FORECAST"), \
                      ([(-0.05, 0)], "OUT_OF_RANGE_FORECAST"), \
                      ([(0.7, 2)], "INVALID_OUTCOME"):
        msg = _fleet_error(panel)
        assert msg is not None and tag in msg and "SF-BL-9139" in msg, (panel, msg)


def test_fleet_control_fixture_exact_01129():
    sc = ak.evaluate_fleet_north_star("fixture", [_trade()], CONTROL)
    assert sc.north_star_score == 0.1129, sc
    assert sc.ece_10bin == 0.3 and sc.brier_score == 0.09, sc
    assert sc.cwr_bid_marked == 0.65 and sc.avg_lockup_days == 30.0, sc


# ---- 3. In-range panels bit-identical ----------------------------------------

def test_valid_panels_bit_identical():
    # Values captured pre-fix (reproduce_prefix.txt + hand computation).
    cases = [
        ([(0.7, 1)], (0.3, 0.09)),
        ([(0.0, 0), (1.0, 1)], (0.0, 0.0)),
        ([(0.0, 0)], (0.0, 0.0)),
        ([(1.0, 1)], (0.0, 0.0)),
        ([(1, 1)], (0.0, 0.0)),          # int forecast, annotated type
        ([(0.7, 1.0)], (0.3, 0.09)),     # float outcome equal to 1
        ([(0.7, 0.0)], (0.7, 0.49)),     # float outcome equal to 0
        ([(0.5, 0), (0.5, 1)], (0.0, 0.25)),
    ]
    for panel, (want_ece, want_brier) in cases:
        ece, brier = ak.compute_empirical_ece_and_brier(panel)
        assert abs(ece - want_ece) < 1e-12 and abs(brier - want_brier) < 1e-12, \
            (panel, ece, brier)


def test_mixed_panel_recomputed_from_primitives():
    panel = [(0.15, 1), (0.25, 0), (0.85, 1), (0.95, 0), (0.45, 1)]
    ece, brier = ak.compute_empirical_ece_and_brier(panel)
    want_brier = sum((p - o) ** 2 for p, o in panel) / len(panel)
    bins = {}
    for p, o in panel:
        bins.setdefault(min(9, int(p * 10)), []).append((p, o))
    want_ece = sum(len(b) / len(panel) *
                   abs(sum(x[1] for x in b) / len(b) - sum(x[0] for x in b) / len(b))
                   for b in bins.values())
    assert abs(ece - want_ece) < 1e-12 and abs(brier - want_brier) < 1e-12, (ece, brier)


def test_empty_panel_convention_unchanged():
    assert ak.compute_empirical_ece_and_brier([]) == (0.0, 0.0)
    # D3 guard at the fleet surface is untouched.
    msg = _fleet_error([])
    assert msg is not None and "NO_ELIGIBLE_ROWS" in msg, msg
    sc = ak.evaluate_fleet_north_star("fixture", [], [])
    assert sc.north_star_score == 0.0, sc


def test_benchmark_scorecards_bit_identical():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    scores = [json.loads(chunk) for chunk in
              out.split("=" * 86)[2].split("-" * 86) if '"fleet_name"' in chunk]
    by_name = {s["fleet_name"]: s for s in scores}
    d = by_name["Alpha-Kelly Disciplined Fleet"]
    n = by_name["Naive Brier-Only Baseline Fleet"]
    assert d["north_star_score"] == 0.0547, d
    assert n["north_star_score"] == -0.1499, n


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        try:
            fn()
            check(fn.__name__, True)
        except AssertionError as e:
            check(fn.__name__, False, str(e))
    print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
    sys.exit(0 if all(PASS) else 1)
