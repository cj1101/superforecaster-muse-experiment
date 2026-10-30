"""
test_alpha_kelly_ece_nonfinite_forecast.py — SF-BL-9820 verification tests.

Pre-registered done criterion: "compute_empirical_ece_and_brier must
raise a named ValueError identifying the non-finite forecast (or return
its documented undefined result) before binning, with finite panels
byte-identical; kill if evaluate_fleet_north_star is shown to
pre-screen every resolved forecast for finiteness before this call."

Task repro (pre-guard algorithm, reproduce_hazard.py):
  [(nan, 1), (0.5, 0)] -> bare ValueError 'cannot convert float NaN
                          to integer' inside int(p * num_bins)
  [(inf, 1), (0.5, 0)] -> OverflowError 'cannot convert float
                          infinity to integer'

Honest history: SF-BL-9139's range gate (landed after this task was
filed) already rejected plain float NaN/+-inf as OUT_OF_RANGE_FORECAST.
Residual gaps closed by SF-BL-9820 (verified live pre-fix,
reproduce_prefix.txt):
  (1) no dedicated non-finite identity — NaN/inf were reported with a
      range label explaining wrap/clamp they never do; now they raise
      NON_FINITE_FORECAST (SF-BL-9820, citing SF-BL-9139) before binning;
  (2) a forecast beyond float range (10**400) crashed the validator
      itself with a bare OverflowError from math.isfinite, on the
      direct surface AND through the fleet's SF-BL-11535 pre-screen;
      _is_finite_number is now OverflowError-safe and the ECE
      validator tri-states a huge int as finite-but-out-of-range, so
      both surfaces raise named ValueErrors.

Kill condition NOT met (CALLER_CENSUS.md): the fleet screen covers
only scorer-routed panels (direct callers have no screen) and itself
leaked the bare OverflowError in gap (2), so it does not pre-screen
EVERY resolved forecast for finiteness.

Run:  python3 test_alpha_kelly_ece_nonfinite_forecast.py  (exit 0 iff pass)
No pytest dependency: test_* functions with assert + __main__ runner.
"""
import io
import contextlib
import json
import os
import sys

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
        market_id="FIXTURE-9820", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C9820",
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
    """ValueError message from compute directly, or None.

    Any non-ValueError exception (bare OverflowError / TypeError) is a
    hard failure: that leak is exactly the hazard under test.
    """
    try:
        ak.compute_empirical_ece_and_brier(panel, num_bins=num_bins)
    except ValueError as e:
        return str(e)
    except BaseException as e:  # noqa: BLE001
        raise AssertionError(
            f"bare {type(e).__name__} leaked, not named ValueError: {e}")
    return None


def _fleet_error(panel):
    try:
        ak.evaluate_fleet_north_star("fixture", [_trade()], panel)
    except ValueError as e:
        return str(e)
    except BaseException as e:  # noqa: BLE001
        raise AssertionError(
            f"bare {type(e).__name__} leaked, not named ValueError: {e}")
    return None


def _preguard(panel, num_bins=10):
    """Original binning body, no validation (hazard documentation)."""
    m = len(panel)
    brier = sum((p - o) ** 2 for p, o in panel) / m
    bins = [[] for _ in range(num_bins)]
    for p, o in panel:
        idx = min(num_bins - 1, int(p * num_bins))
        bins[idx].append((p, o))
    ece = 0.0
    for b in bins:
        if not b:
            continue
        ece += (len(b) / m) * abs(
            sum(x[1] for x in b) / len(b) - sum(x[0] for x in b) / len(b))
    return ece, brier


# ---- 0. The task's quoted hazard is real on the pre-guard algorithm ---------

def test_preguard_reproduces_quoted_bare_errors():
    try:
        _preguard([(NAN, 1), (0.5, 0)])
        raise SystemExit("preguard NaN panel did not raise")
    except ValueError as e:
        assert "cannot convert float NaN to integer" in str(e), e
    try:
        _preguard([(INF, 1), (0.5, 0)])
        raise SystemExit("preguard +inf panel did not raise")
    except OverflowError as e:
        assert "cannot convert float infinity to integer" in str(e), e


# ---- 1. Task repro panels: dedicated NON_FINITE_FORECAST identity -----------

def test_task_nan_panel_raises_nonfinite_named():
    msg = _direct_error([(NAN, 1), (0.5, 0)])
    assert msg is not None, "NaN panel returned a score"
    assert "NON_FINITE_FORECAST" in msg and "SF-BL-9820" in msg, msg
    assert "SF-BL-9139" in msg, msg  # neighbouring suite compatibility
    assert "forecast=nan" in msg and "[0]" in msg, msg


def test_task_inf_panel_raises_nonfinite_not_overflow():
    # _direct_error hard-fails on OverflowError: reaching here at all
    # proves +inf no longer leaks the bare OverflowError.
    msg = _direct_error([(INF, 1), (0.5, 0)])
    assert msg is not None and "NON_FINITE_FORECAST" in msg, msg
    assert "forecast=inf" in msg, msg


def test_neg_inf_raises_nonfinite_named():
    msg = _direct_error([(-INF, 1), (0.5, 0)])
    assert msg is not None and "NON_FINITE_FORECAST" in msg, msg
    assert "forecast=-inf" in msg, msg


def test_nonfinite_in_later_row_names_index():
    msg = _direct_error([(0.5, 0), (0.7, 1), (NAN, 1)])
    assert msg is not None and "NON_FINITE_FORECAST" in msg, msg
    assert "[2]" in msg, msg
    msg = _direct_error([(0.5, 0), (INF, 1)])
    assert msg is not None and "[1]" in msg, msg


def test_numpy_float64_nonfinite_raises_nonfinite():
    np = __import__("numpy")
    for v in (np.float64(NAN), np.float64(INF), np.float64(-INF)):
        msg = _direct_error([(v, 1), (0.5, 0)])
        assert msg is not None and "NON_FINITE_FORECAST" in msg, (v, msg)


def test_nonfinite_gated_for_custom_num_bins():
    for nb in (1, 5, 20):
        msg = _direct_error([(NAN, 1)], num_bins=nb)
        assert msg is not None and "NON_FINITE_FORECAST" in msg, (nb, msg)
        msg = _direct_error([(INF, 1)], num_bins=nb)
        assert msg is not None and "NON_FINITE_FORECAST" in msg, (nb, msg)


# ---- 2. Residual gap (2): magnitude beyond float range ----------------------

def test_huge_int_forecast_raises_named_not_overflow():
    for p in (10 ** 400, -(10 ** 400), 10 ** 1000):
        msg = _direct_error([(p, 1)])
        assert msg is not None and "OUT_OF_RANGE_FORECAST" in msg, (p, msg)
        assert "SF-BL-9139" in msg, msg


def test_fleet_huge_int_raises_named_not_overflow():
    # Pre-fix the fleet's own SF-BL-11535 screen leaked OverflowError.
    msg = _fleet_error([(10 ** 400, 1)])
    assert msg is not None and "INVALID_RESOLVED_FORECAST_ELEMENT" in msg, msg


def test_fleet_nonfinite_panels_raise_named():
    for panel in ([(NAN, 1), (0.5, 0)], [(INF, 1), (0.5, 0)],
                  [(-INF, 1), (0.5, 0)]):
        msg = _fleet_error(panel)
        assert msg is not None and "SF-BL-11535" in msg, (panel, msg)


# ---- 3. Neighbouring gates unchanged ----------------------------------------

def test_finite_out_of_range_still_out_of_range_9139():
    for p in (-0.2, -0.05, 1.2, 2.0):
        msg = _direct_error([(p, 1)])
        assert msg is not None and "OUT_OF_RANGE_FORECAST" in msg, (p, msg)
        assert "NON_FINITE_FORECAST" not in msg, (p, msg)
        assert "SF-BL-9139" in msg, msg


def test_nonnumeric_forecast_still_named_out_of_range():
    for panel in ([("0.7", 1)], [(None, 1)], [(True, 1)]):
        msg = _direct_error(panel)
        assert msg is not None and "OUT_OF_RANGE_FORECAST" in msg, (panel, msg)


def test_bad_outcome_still_invalid_outcome():
    msg = _direct_error([(0.7, 2)])
    assert msg is not None and "INVALID_OUTCOME" in msg, msg
    msg = _direct_error([(0.7, NAN)])
    assert msg is not None and "INVALID_OUTCOME" in msg, msg


# ---- 4. Finite panels byte-identical ----------------------------------------

def test_finite_panels_byte_identical():
    cases = [
        ([(0.7, 1)], (0.30000000000000004, 0.09000000000000002)),
        ([(0.5, 0), (0.7, 1)], (0.4, 0.17)),
        ([(0.0, 0), (1.0, 1)], (0.0, 0.0)),
        ([(1, 1)], (0.0, 0.0)),
        ([(0.7, 1.0)], (0.30000000000000004, 0.09000000000000002)),
        ([(0.5, 0), (0.5, 1)], (0.0, 0.25)),
    ]
    for panel, want in cases:
        got = ak.compute_empirical_ece_and_brier(panel)
        assert got == want, (panel, got, want)  # exact, not approx


def test_mixed_panel_recomputed_from_primitives():
    panel = [(0.15, 1), (0.25, 0), (0.85, 1), (0.95, 0), (0.45, 1)]
    ece, brier = ak.compute_empirical_ece_and_brier(panel)
    want_brier = sum((p - o) ** 2 for p, o in panel) / len(panel)
    bins = {}
    for p, o in panel:
        bins.setdefault(min(9, int(p * 10)), []).append((p, o))
    want_ece = sum(len(b) / len(panel) *
                   abs(sum(x[1] for x in b) / len(b)
                       - sum(x[0] for x in b) / len(b))
                   for b in bins.values())
    assert ece == want_ece and brier == want_brier, (ece, brier)


def test_empty_panel_convention_unchanged():
    assert ak.compute_empirical_ece_and_brier([]) == (0.0, 0.0)


def test_fleet_control_fixture_exact_01129():
    sc = ak.evaluate_fleet_north_star("fixture", [_trade()], CONTROL)
    assert sc.north_star_score == 0.1129, sc
    assert sc.ece_10bin == 0.3 and sc.brier_score == 0.09, sc


def test_benchmark_scorecards_bit_identical():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    scores = [json.loads(chunk) for chunk in
              out.split("=" * 86)[2].split("-" * 86) if '"fleet_name"' in chunk]
    by_name = {s["fleet_name"]: s for s in scores}
    # Values captured on the current tree immediately BEFORE this fix
    # (benchmark_prefix.txt) and verified bit-identical after it
    # (benchmark_postfix.txt, full-stdout diff empty). The 0.0547 in
    # the older SF-BL-9139/9711 suites is stale on the current tree —
    # concurrent fleet executors changed other alpha_kelly surfaces
    # after those suites were written; -0.1611 is the pre-fix value
    # for THIS run, so equality here isolates this fix's effect.
    assert by_name["Alpha-Kelly Disciplined Fleet"]["north_star_score"] == -0.1611
    assert by_name["Naive Brier-Only Baseline Fleet"]["north_star_score"] == -0.1499


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
