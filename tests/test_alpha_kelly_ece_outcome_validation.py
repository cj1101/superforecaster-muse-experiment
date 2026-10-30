"""
test_alpha_kelly_ece_outcome_validation.py — SF-BL-9711 verification tests.

Pre-registered done criterion: "Guard raises ValueError unless every
outcome is exactly 0 or 1 (bool excluded) and finite; success = the
outcome-2, outcome--1 and NaN-outcome repros are all rejected and a
clean book's (ece, brier) is byte-identical; kill if a producer census
proves every outcome reaching this function is built only by validated
settlement paths with a tested binary invariant."

Kill condition NOT met (PRODUCER_AUDIT.md in the SF-BL-9711 work dir):
the benchmark panel is bare dict literals appended independently of
settle_or_mark_trade; no validated settlement path builds the panel.

History: at filing (2026-10-21T19:29Z) the function had NO outcome
check — the task fixtures [(0.9, 2), (0.1, 0)] returned (ECE 0.60,
Brier 0.61), [(0.9, -1), (0.1, 0)] returned (ECE 1.00, Brier 1.81) and
a NaN-outcome panel returned (NaN, NaN), all impossible for binary
resolutions (repro.py in the work dir re-derives those values from the
pre-guard algorithm). SF-BL-9139's gate landed afterwards and rejects
every plain-typed bad outcome. The residual gap closed by SF-BL-9711
itself: the gate tested isinstance(o, (int, float)), so numpy-typed
outcomes that ARE exactly 0 or 1 (np.int64(1), np.int32(0),
np.float32(1.0)) were falsely rejected — a clean book from a
numpy-native producer errored instead of scoring byte-identically.
The outcome check now runs through _outcome_in_ece_domain
(numbers.Real, bool/np.bool_ excluded, finite, value in {0, 1}).

Run:  python3 test_alpha_kelly_ece_outcome_validation.py  (exit 0 iff pass)
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
        market_id="FIXTURE-9711", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C9711",
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


# ---- 1. The task's own repro fixtures ---------------------------------------

def test_task_repro_outcome_two_rejected():
    # Pre-guard this panel returned (ECE 0.60, Brier 0.61) — task title values.
    msg = _direct_error([(0.9, 2), (0.1, 0)])
    assert msg is not None, "outcome-2 repro returned a score"
    assert "INVALID_OUTCOME" in msg and "SF-BL-9711" in msg, msg


def test_task_repro_outcome_minus_one_rejected():
    # Pre-guard this panel returned (ECE 1.00, Brier 1.81) — task title values.
    msg = _direct_error([(0.9, -1), (0.1, 0)])
    assert msg is not None, "outcome--1 repro returned a score"
    assert "INVALID_OUTCOME" in msg and "SF-BL-9711" in msg, msg


def test_task_repro_nan_outcome_rejected_not_nan_result():
    # Pre-guard a NaN outcome returned (NaN, NaN) as an ordinary result.
    for panel in ([(0.9, NAN), (0.1, 0)], [(0.6, NAN)]):
        msg = _direct_error(panel)
        assert msg is not None and "INVALID_OUTCOME" in msg, (panel, msg)


def test_nonbinary_and_nonfinite_outcomes_rejected():
    for o in (2, -1, 3, 100, -100, 0.5, 1.5, -0.5, 0.999, INF, -INF):
        msg = _direct_error([(0.7, o)])
        assert msg is not None and "INVALID_OUTCOME" in msg, (o, msg)


def test_bool_outcomes_rejected():
    # bool excluded even though True == 1 / False == 0.
    for o in (True, False):
        msg = _direct_error([(0.7, o)])
        assert msg is not None and "INVALID_OUTCOME" in msg, (o, msg)


def test_nonnumeric_outcomes_rejected_named():
    for o in ("1", "0", None, [1], (1,), 1 + 0j):
        msg = _direct_error([(0.7, o)])
        assert msg is not None and "INVALID_OUTCOME" in msg, (o, msg)


def test_bad_outcome_in_later_row_rejected_with_index():
    msg = _direct_error([(0.7, 1), (0.3, 0), (0.5, 2)])
    assert msg is not None and "[2]" in msg and "INVALID_OUTCOME" in msg, msg
    msg = _direct_error([(0.7, 1), (0.3, NAN)])
    assert msg is not None and "[1]" in msg, msg


def test_custom_num_bins_still_gated():
    msg = _direct_error([(0.7, 2)], num_bins=5)
    assert msg is not None and "INVALID_OUTCOME" in msg, msg


# ---- 2. The residual gap SF-BL-9711 closes: numpy-typed valid outcomes ------

def test_numpy_valid_outcomes_score_byte_identical():
    if not HAS_NUMPY:
        return
    want = ak.compute_empirical_ece_and_brier([(0.7, 1), (0.3, 0)])
    for one, zero in ((np.int64(1), np.int64(0)), (np.int32(1), np.int32(0)),
                      (np.float64(1.0), np.float64(0.0)),
                      (np.float32(1.0), np.float32(0.0))):
        got = ak.compute_empirical_ece_and_brier([(0.7, one), (0.3, zero)])
        assert got == want, (type(one), got, want)


def test_numpy_bad_outcomes_rejected():
    if not HAS_NUMPY:
        return
    for o in (np.int64(2), np.int64(-1), np.float64(0.5), np.float64(NAN),
              np.float32(INF), np.bool_(True), np.bool_(False)):
        msg = _direct_error([(0.7, o)])
        assert msg is not None and "INVALID_OUTCOME" in msg, (o, msg)


# ---- 3. Fleet surface: no corrupt outcome reaches a quoted scorecard --------

def test_fleet_bad_outcome_panels_raise():
    for panel in ([(0.9, 2), (0.1, 0)], [(0.9, -1), (0.1, 0)],
                  [(0.9, NAN), (0.1, 0)], [(0.7, True)]):
        msg = _fleet_error(panel)
        assert msg is not None and ("INVALID_OUTCOME" in msg
                                    or "INVALID_RESOLVED_FORECAST_ELEMENT" in msg), (panel, msg)


def test_fleet_control_fixture_exact_01129():
    sc = ak.evaluate_fleet_north_star("fixture", [_trade()], CONTROL)
    assert sc.north_star_score == 0.1129, sc
    assert sc.ece_10bin == 0.3 and sc.brier_score == 0.09, sc


# ---- 4. Clean books byte-identical ------------------------------------------

def test_valid_outcome_forms_accepted_identical():
    want = ak.compute_empirical_ece_and_brier([(0.7, 1), (0.3, 0)])
    for panel in ([(0.7, 1.0), (0.3, 0.0)], [(0.7, 1), (0.3, 0)]):
        assert ak.compute_empirical_ece_and_brier(panel) == want, panel


def test_clean_book_byte_identical():
    clean = [(0.15, 1), (0.25, 0), (0.85, 1), (0.95, 0), (0.45, 1), (0.0, 0), (1.0, 1)]
    ece, brier = ak.compute_empirical_ece_and_brier(clean)
    assert (ece, brier) == (0.3928571428571428, 0.2875), (ece, brier)
    # Recomputed from primitives (the pre-guard algorithm on valid rows).
    m = len(clean)
    want_brier = sum((p - o) ** 2 for p, o in clean) / m
    bins = {}
    for p, o in clean:
        bins.setdefault(min(9, int(p * 10)), []).append((p, o))
    want_ece = sum(len(b) / m * abs(sum(x[1] for x in b) / len(b)
                                    - sum(x[0] for x in b) / len(b))
                   for b in bins.values())
    # 1e-12 tolerance here only: dict-order summation differs in the last
    # ulp from the function's bin-order summation; the pinned literal
    # above is the exact byte-identity check.
    assert abs(ece - want_ece) < 1e-12 and brier == want_brier, (ece, brier)


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
    assert by_name["Alpha-Kelly Disciplined Fleet"]["north_star_score"] == 0.0547
    assert by_name["Naive Brier-Only Baseline Fleet"]["north_star_score"] == -0.1499


def test_no_dead_line_touched():
    # Gate respected: guard only rejects/scores; signatures unchanged.
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
