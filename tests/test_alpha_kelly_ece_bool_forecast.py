"""
test_alpha_kelly_ece_bool_forecast.py — SF-BL-11010 verification tests.

Pre-registered done criterion: "compute_empirical_ece_and_brier must
raise a named ValueError for any forecast p that is a bool or
non-numeric, fixtured at (True, 1) and (False, 0), with genuine
numeric panels bit-identical; kill if a caller census proves every
resolved forecast reaching this function is a validated float under a
tested invariant."

Kill condition NOT met (CALLER_CENSUS.md in the SF-BL-11010 work dir):
the only in-repo production caller is evaluate_fleet_north_star, whose
SF-BL-11535 screen covers scorer-routed panels only; direct callers
pass bare literals, the annotation is unenforced, panels are plain
mutable lists, and no schema layer stands between a JSON producer's
true/false and this function.

History: at filing the panel [(True, 1), (0.5, 0)] returned
(ECE 0.25, Brier 0.125) — bit-identical to [(1.0, 1), (0.5, 0)], a
bool read as certainty (repro.py Part A in the work dir re-derives it
from the pre-guard algorithm). SF-BL-9139's gate landed afterwards and
rejected plain bool/non-numeric forecasts, but only as
OUT_OF_RANGE_FORECAST with truncation/wrap/clamp text — none of which
a bool does (True == 1 / False == 0 are IN [0, 1]) — and no
SF-BL-11010 identity. Residual gaps closed by SF-BL-11010 itself
(verified live pre-fix, repro_prefix.txt in the work dir):
  (1) dedicated identity — bool / non-numeric forecasts now raise
      BOOL_OR_NON_NUMERIC_FORECAST (SF-BL-11010), naming value and
      type, carrying the OUT_OF_RANGE_FORECAST umbrella token and
      the SF-BL-9139 citation the neighbouring suites pin;
  (2) the isinstance(p, (int, float)) gate falsely rejected genuine
      numpy numeric forecasts (np.float32(0.5), np.int64(1)) and did
      not identify np.bool_ as a bool — the type domain now mirrors
      SF-BL-9711's outcome domain (numbers.Real, bool/np.bool_
      excluded explicitly and by type name).

Run:  python3 test_alpha_kelly_ece_bool_forecast.py  (exit 0 iff pass)
No pytest dependency: test_* functions with assert + __main__ runner.
"""
import io
import contextlib
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster/superforecaster"))
import alpha_kelly as ak  # noqa: E402

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

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
        market_id="FIXTURE-11010", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C11010",
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

    Any non-ValueError exception is a hard failure: a bare TypeError /
    OverflowError leak is exactly the unnamed-crash hazard class.
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

def test_preguard_reproduces_quoted_bool_as_certainty():
    assert _preguard([(True, 1), (0.5, 0)]) == (0.25, 0.125)
    assert _preguard([(True, 1), (0.5, 0)]) == _preguard([(1.0, 1), (0.5, 0)])
    assert _preguard([(False, 0), (0.5, 0)]) == _preguard([(0.0, 0), (0.5, 0)])


# ---- 1. Prereg fixtures: bool forecasts raise the dedicated named error -----

def test_prereg_true_forecast_rejected_bool_named():
    msg = _direct_error([(True, 1), (0.5, 0)])
    assert msg is not None, "bool panel returned a score (pre-guard: ECE 0.25)"
    assert "BOOL_OR_NON_NUMERIC_FORECAST" in msg and "SF-BL-11010" in msg, msg
    assert "forecast=True" in msg and "bool" in msg and "[0]" in msg, msg


def test_prereg_false_forecast_rejected_bool_named():
    msg = _direct_error([(False, 0)])
    assert msg is not None, "False forecast returned a score"
    assert "BOOL_OR_NON_NUMERIC_FORECAST" in msg and "SF-BL-11010" in msg, msg
    assert "forecast=False" in msg, msg


def test_bool_forecast_in_later_row_rejected_with_index():
    msg = _direct_error([(0.5, 0), (0.7, 1), (True, 1)])
    assert msg is not None and "BOOL_OR_NON_NUMERIC_FORECAST" in msg, msg
    assert "[2]" in msg, msg
    msg = _direct_error([(0.5, 0), (False, 0)])
    assert msg is not None and "[1]" in msg, msg


def test_bool_rejected_for_custom_num_bins():
    for nb in (1, 5, 20):
        msg = _direct_error([(True, 1)], num_bins=nb)
        assert msg is not None and "BOOL_OR_NON_NUMERIC_FORECAST" in msg, (nb, msg)


# ---- 2. Non-numeric forecasts: same dedicated named error, never bare -------

def test_nonnumeric_forecasts_rejected_bool_or_nonnumeric_named():
    for p in ("0.5", "1", None, [0.5], (0.5,), {"p": 0.5}, 1 + 0j, object()):
        msg = _direct_error([(p, 0)])
        assert msg is not None and "BOOL_OR_NON_NUMERIC_FORECAST" in msg, (p, msg)
        assert "SF-BL-11010" in msg and type(p).__name__ in msg, (p, msg)


def test_neighbouring_suites_tokens_still_present_for_type_branch():
    # SF-BL-9139 suite pins "SF-BL-9139" and SF-BL-9820 suite pins
    # "OUT_OF_RANGE_FORECAST" on the [(True, 1)] / non-numeric panels.
    for panel in ([(True, 1)], [("0.7", 1)], [(None, 1)]):
        msg = _direct_error(panel)
        assert msg is not None and "SF-BL-9139" in msg, (panel, msg)
        assert "OUT_OF_RANGE_FORECAST" in msg, (panel, msg)


# ---- 3. Residual gap (2): numpy clothing -------------------------------------

def test_numpy_bool_forecasts_rejected_as_bool():
    if not HAS_NUMPY:
        return
    for p in (np.bool_(True), np.bool_(False)):
        msg = _direct_error([(p, 1)])
        assert msg is not None and "BOOL_OR_NON_NUMERIC_FORECAST" in msg, (p, msg)
        assert "bool_" in msg, msg


def test_numpy_numeric_forecasts_score_identical_to_twins():
    if not HAS_NUMPY:
        return
    # Exactly-representable values: float32 0.5, float64 0.7 (subclass),
    # int64/int32 1 and 0 must score == their Python-typed twins.
    want = ak.compute_empirical_ece_and_brier([(0.5, 0), (1, 1), (0, 0)])
    for panel in ([(np.float32(0.5), 0), (np.int64(1), 1), (np.int64(0), 0)],
                  [(np.float64(0.5), 0), (np.int32(1), 1), (np.int32(0), 0)]):
        got = ak.compute_empirical_ece_and_brier(panel)
        assert got == want, (panel, got, want)
    # numpy non-finite keeps the SF-BL-9820 identity, numpy out-of-range
    # keeps SF-BL-9139's, numpy bool-outcome keeps SF-BL-9711's.
    msg = _direct_error([(np.float64(NAN), 1)])
    assert msg is not None and "NON_FINITE_FORECAST" in msg, msg
    msg = _direct_error([(np.float64(1.2), 1)])
    assert msg is not None and "OUT_OF_RANGE_FORECAST" in msg, msg
    assert "BOOL_OR_NON_NUMERIC_FORECAST" not in msg, msg


# ---- 4. Neighbouring gates unchanged -----------------------------------------

def test_finite_out_of_range_still_out_of_range_9139():
    for p in (-0.2, -0.05, 1.2, 2.0, 10 ** 400):
        msg = _direct_error([(p, 1)])
        assert msg is not None and "OUT_OF_RANGE_FORECAST" in msg, (p, msg)
        assert "BOOL_OR_NON_NUMERIC_FORECAST" not in msg, (p, msg)
        assert "SF-BL-9139" in msg, msg


def test_nonfinite_still_non_finite_9820():
    for p in (NAN, INF, -INF):
        msg = _direct_error([(p, 1)])
        assert msg is not None and "NON_FINITE_FORECAST" in msg, (p, msg)
        assert "BOOL_OR_NON_NUMERIC_FORECAST" not in msg, (p, msg)


def test_bad_outcome_still_invalid_outcome():
    msg = _direct_error([(0.7, 2)])
    assert msg is not None and "INVALID_OUTCOME" in msg, msg
    msg = _direct_error([(0.7, True)])
    assert msg is not None and "INVALID_OUTCOME" in msg, msg


def test_malformed_entry_still_named_not_bare():
    for panel in ([(0.7,)], [(0.7, 1, 0.5)], [0.7], [None]):
        msg = _direct_error(panel)
        assert msg is not None and "SF-BL-9139" in msg, (panel, msg)


# ---- 5. Fleet surface ----------------------------------------------------------

def test_fleet_bool_panel_raises_named():
    # The scorer's SF-BL-11535 screen fires first at that boundary; either
    # named identity is a correct rejection — a score is not.
    msg = _fleet_error([(True, 1), (0.5, 0)])
    assert msg is not None and ("INVALID_RESOLVED_FORECAST_ELEMENT" in msg
                                or "BOOL_OR_NON_NUMERIC_FORECAST" in msg), msg


def test_fleet_control_fixture_exact_01129():
    sc = ak.evaluate_fleet_north_star("fixture", [_trade()], CONTROL)
    assert sc.north_star_score == 0.1129, sc
    assert sc.ece_10bin == 0.3 and sc.brier_score == 0.09, sc


# ---- 6. Genuine numeric panels bit-identical -----------------------------------

def test_numeric_panels_bit_identical():
    cases = [
        ([(0.7, 1)], (0.30000000000000004, 0.09000000000000002)),
        ([(0.5, 0), (0.5, 1)], (0.0, 0.25)),
        ([(0.0, 0), (1.0, 1)], (0.0, 0.0)),
        ([(1, 1)], (0.0, 0.0)),          # int forecast 1: genuine certainty
        ([(0, 0)], (0.0, 0.0)),          # int forecast 0: genuine impossibility
        ([(0.7, 1.0)], (0.30000000000000004, 0.09000000000000002)),
        ([(0.5, 0), (0.7, 1)], (0.4, 0.17)),
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


def test_benchmark_scorecards_bit_identical():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    scores = [json.loads(chunk) for chunk in
              out.split("=" * 86)[2].split("-" * 86) if '"fleet_name"' in chunk]
    by_name = {s["fleet_name"]: s for s in scores}
    # Values captured on this tree immediately BEFORE this fix
    # (benchmark_prefix.txt in the work dir): disciplined -0.1611 is the
    # current-tree pre-fix value (the 0.0547 in older suites is stale —
    # concurrent fleet executors changed other surfaces after they were
    # written); equality here isolates this fix's effect.
    assert by_name["Alpha-Kelly Disciplined Fleet"]["north_star_score"] == -0.1611
    assert by_name["Naive Brier-Only Baseline Fleet"]["north_star_score"] == -0.1499


def test_no_dead_line_touched():
    import inspect
    sig = inspect.signature(ak.compute_empirical_ece_and_brier)
    assert list(sig.parameters) == ["predictions_and_outcomes", "num_bins"], sig


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
