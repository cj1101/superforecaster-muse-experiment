"""
test_alpha_kelly_ece_nonnumeric_forecast.py — SF-BL-11048 verification.

Pre-registered done criterion: "compute_empirical_ece_and_brier raises
a named ValueError for a non-numeric forecast inside a well-shaped
pair, fixtured at ('0.6', 1) and (None, 0), with valid panels
bit-identical; kill if a caller census proves forecasts only ever
reach this function through scorer-validated resolved sets."

Kill NOT met (CALLER_CENSUS.md in the SF-BL-11048 work dir): the only
in-repo production caller is evaluate_fleet_north_star (its SF-BL-11535
screen is at the scorer boundary, not this function's); direct
callers pass bare literals, the annotation is unenforced, panels are
plain mutable lists, and no schema layer exists.

History: pre-guard, the Brier sum ran first, so [('0.6', 1)] and
[(None, 0)] raised a bare TypeError from (p - o) ** 2 while
[(0.6, 1)] returned (0.4, 0.16000000000000003) (repro.py Part A).
SF-BL-11010's branch later rejected them under its bool-framed
BOOL_OR_NON_NUMERIC_FORECAST identity with no SF-BL-11048 identity.
This task gives non-bool non-numeric forecasts their own named error,
NON_NUMERIC_FORECAST (SF-BL-11048), carrying the umbrella tokens /
citations the neighbouring suites pin; the bool branch is unchanged.

Run: python3 test_alpha_kelly_ece_nonnumeric_forecast.py (exit 0 iff pass)
"""
import contextlib
import io
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

NAN = float("nan")
INF = float("inf")
CONTROL = [(0.7, 1)]


def _trade(**over):
    base = dict(
        market_id="FIXTURE-11048", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C11048",
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
    """ValueError message, or None; any other exception hard-fails."""
    try:
        ak.compute_empirical_ece_and_brier(panel, num_bins=num_bins)
    except ValueError as e:
        return str(e)
    except BaseException as e:  # noqa: BLE001
        raise AssertionError(f"bare {type(e).__name__} leaked: {e}")
    return None


def _fleet_error(panel):
    try:
        ak.evaluate_fleet_north_star("fixture", [_trade()], panel)
    except ValueError as e:
        return str(e)
    except BaseException as e:  # noqa: BLE001
        raise AssertionError(f"bare {type(e).__name__} leaked: {e}")
    return None


def _preguard(panel, num_bins=10):
    m = len(panel)
    brier = sum((p - o) ** 2 for p, o in panel) / m
    bins = [[] for _ in range(num_bins)]
    for p, o in panel:
        bins[min(num_bins - 1, int(p * num_bins))].append((p, o))
    ece = 0.0
    for b in bins:
        if b:
            ece += (len(b) / m) * abs(
                sum(x[1] for x in b) / len(b) - sum(x[0] for x in b) / len(b))
    return ece, brier


# ---- 0. Filing hazard real on the pre-guard algorithm -----------------------

def test_preguard_reproduces_bare_typeerror():
    for panel in ([("0.6", 1)], [(None, 0)]):
        try:
            _preguard(panel)
            raise AssertionError(f"preguard scored {panel!r}")
        except TypeError:
            pass
    assert _preguard([(0.6, 1)]) == (0.4, 0.16000000000000003)


# ---- 1. Prereg fixtures ------------------------------------------------------

def test_prereg_string_forecast_rejected_nonnumeric_named():
    msg = _direct_error([("0.6", 1)])
    assert msg is not None, "string panel returned a score"
    assert "NON_NUMERIC_FORECAST" in msg and "SF-BL-11048" in msg, msg
    assert "forecast='0.6'" in msg and "str" in msg and "[0]" in msg, msg


def test_prereg_none_forecast_rejected_nonnumeric_named():
    msg = _direct_error([(None, 0)])
    assert msg is not None, "None panel returned a score"
    assert "NON_NUMERIC_FORECAST" in msg and "SF-BL-11048" in msg, msg
    assert "forecast=None" in msg and "NoneType" in msg and "[0]" in msg, msg


def test_prereg_control_valid_panel_exact():
    assert ak.compute_empirical_ece_and_brier([(0.6, 1)]) == \
        (0.4, 0.16000000000000003)


# ---- 2. Wider non-numeric dimension ------------------------------------------

def test_other_nonnumeric_forecasts_rejected_nonnumeric_named():
    for p in ("1", "", "abc", None, [0.6], (0.6,), {"p": 0.6}, 1 + 0j, object()):
        msg = _direct_error([(p, 0)])
        assert msg is not None and "NON_NUMERIC_FORECAST" in msg, (p, msg)
        assert "SF-BL-11048" in msg and type(p).__name__ in msg, (p, msg)


def test_nonnumeric_in_later_row_rejected_with_index():
    msg = _direct_error([(0.5, 0), (0.7, 1), ("0.6", 1)])
    assert msg is not None and "NON_NUMERIC_FORECAST" in msg and "[2]" in msg, msg
    msg = _direct_error([(0.5, 0), (None, 0)])
    assert msg is not None and "NON_NUMERIC_FORECAST" in msg and "[1]" in msg, msg


def test_nonnumeric_rejected_for_custom_num_bins():
    for nb in (1, 5, 20):
        msg = _direct_error([("0.6", 1)], num_bins=nb)
        assert msg is not None and "NON_NUMERIC_FORECAST" in msg, (nb, msg)


def test_neighbouring_tokens_still_present_on_nonnumeric():
    # SF-BL-11010 pins BOOL_OR_NON_NUMERIC_FORECAST + SF-BL-11010 + type
    # name; SF-BL-9139 / SF-BL-9820 suites pin SF-BL-9139 +
    # OUT_OF_RANGE_FORECAST on string / None panels.
    for panel in ([("0.7", 1)], [(None, 1)]):
        msg = _direct_error(panel)
        assert "BOOL_OR_NON_NUMERIC_FORECAST" in msg, msg
        assert "SF-BL-11010" in msg and "SF-BL-9139" in msg, msg
        assert "OUT_OF_RANGE_FORECAST" in msg, msg


def test_numpy_string_forecast_rejected_nonnumeric_named():
    if not HAS_NUMPY:
        return
    msg = _direct_error([(np.str_("0.6"), 1)])
    assert msg is not None and "NON_NUMERIC_FORECAST" in msg, msg


# ---- 3. Neighbouring gates unchanged -----------------------------------------

def test_bool_forecast_still_bool_named_11010_not_11048_primary():
    for panel, frag in (([(True, 1)], "forecast=True"), ([(False, 0)], "forecast=False")):
        msg = _direct_error(panel)
        assert msg is not None and msg.startswith("BOOL_OR_NON_NUMERIC_FORECAST"), msg
        assert "SF-BL-11010" in msg and frag in msg, msg


def test_finite_out_of_range_still_out_of_range_9139():
    for p in (-0.2, -0.05, 1.2, 2.0, 10 ** 400):
        msg = _direct_error([(p, 1)])
        assert msg is not None and "OUT_OF_RANGE_FORECAST" in msg, (p, msg)
        assert "NON_NUMERIC_FORECAST" not in msg, (p, msg)
        assert "BOOL_OR_NON_NUMERIC_FORECAST" not in msg, (p, msg)


def test_nonfinite_still_non_finite_9820():
    for p in (NAN, INF, -INF):
        msg = _direct_error([(p, 1)])
        assert msg is not None and "NON_FINITE_FORECAST" in msg, (p, msg)
        assert "NON_NUMERIC_FORECAST" not in msg, (p, msg)


def test_bad_outcome_still_invalid_outcome():
    for panel in ([(0.7, 2)], [(0.7, True)], [(0.7, "1")], [(0.7, None)]):
        msg = _direct_error(panel)
        assert msg is not None and "INVALID_OUTCOME" in msg, (panel, msg)


def test_malformed_entry_still_named_not_bare():
    for panel in ([(0.7,)], [(0.7, 1, 0.5)], [0.7], [None], ["0.6"]):
        msg = _direct_error(panel)
        assert msg is not None and "SF-BL-9139" in msg, (panel, msg)


def test_numpy_numeric_forecasts_score_identical_to_twins():
    if not HAS_NUMPY:
        return
    want = ak.compute_empirical_ece_and_brier([(0.5, 0), (1, 1), (0, 0)])
    got = ak.compute_empirical_ece_and_brier(
        [(np.float32(0.5), 0), (np.int64(1), 1), (np.int64(0), 0)])
    assert got == want, (got, want)


# ---- 4. Fleet surface ----------------------------------------------------------

def test_fleet_nonnumeric_panel_raises_named():
    msg = _fleet_error([("0.6", 1)])
    assert msg is not None and ("INVALID_RESOLVED_FORECAST_ELEMENT" in msg
                                or "NON_NUMERIC_FORECAST" in msg), msg
    msg = _fleet_error([(None, 0)])
    assert msg is not None and ("INVALID_RESOLVED_FORECAST_ELEMENT" in msg
                                or "NON_NUMERIC_FORECAST" in msg), msg


def test_fleet_control_fixture_exact_01129():
    sc = ak.evaluate_fleet_north_star("fixture", [_trade()], CONTROL)
    assert sc.north_star_score == 0.1129, sc
    assert sc.ece_10bin == 0.3 and sc.brier_score == 0.09, sc


# ---- 5. Valid panels bit-identical ---------------------------------------------

def test_valid_panels_bit_identical():
    cases = [
        ([(0.6, 1)], (0.4, 0.16000000000000003)),
        ([(0.7, 1)], (0.30000000000000004, 0.09000000000000002)),
        ([(0.5, 0), (0.5, 1)], (0.0, 0.25)),
        ([(0.0, 0), (1.0, 1)], (0.0, 0.0)),
        ([(1, 1)], (0.0, 0.0)),
        ([(0, 0)], (0.0, 0.0)),
        ([(0.5, 0), (0.7, 1)], (0.4, 0.17)),
    ]
    for panel, want in cases:
        got = ak.compute_empirical_ece_and_brier(panel)
        assert got == want, (panel, got, want)


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
    scores = [json.loads(c) for c in
              out.split("=" * 86)[2].split("-" * 86) if '"fleet_name"' in c]
    by_name = {s["fleet_name"]: s for s in scores}
    # Captured pre-fix on this tree (benchmark_prefix.txt in work dir).
    assert by_name["Alpha-Kelly Disciplined Fleet"]["north_star_score"] == -0.1611
    assert by_name["Naive Brier-Only Baseline Fleet"]["north_star_score"] == -0.1499


def test_no_dead_line_touched():
    import inspect
    sig = inspect.signature(ak.compute_empirical_ece_and_brier)
    assert list(sig.parameters) == ["predictions_and_outcomes", "num_bins"], sig


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
            passed += 1
        except BaseException as e:  # noqa: BLE001
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"{passed}/{passed + failed} passed")
    sys.exit(1 if failed else 0)
