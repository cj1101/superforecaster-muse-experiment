"""SF-BL-10939: agents.py scorers must reject bool probabilities.

Bug: brier_score / log_score / calibration_stats validated only
finiteness and the [0, 1] range (SF-BL-9784 / SF-BL-10031 /
SF-BL-10032), which a bool satisfies numerically (float(True)==1.0),
so a flag leaking into a forecast field was graded as an endpoint
forecast: brier_score([True], [1]) returned 0.0,
calibration_stats([True, False], [1, 0]) returned ECE 0.0 with
mean_forecast 0.5, log_score([True], [1]) returned ~1e-06.
SF-BL-10884 fixtured only calibration_stats' n_bins bool and
SF-BL-10914 only its fractional outcomes — neither fixtured a bool
probability in these three scorers.

Done criterion: named ValueError on any bool probability in all
three scorers; lawful panels bit-identical (bool OUTCOMES stay
lawful — SF-BL-10853 pins brier_score([0.5, 0.6], [True, False])
== 0.305 — and float/int endpoint panels [0.0, 1.0] / [1, 0] stay
scored).
Kill criterion: [True] still returns Brier 0.0, or the [True, False]
panel still returns ECE 0.0.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from superforecaster.agents import (  # noqa: E402
    brier_score,
    calibration_stats,
    log_score,
)

SCORERS_2 = [brier_score, log_score]


def _call(fn, probs, outcomes):
    return fn(probs, outcomes)


# --- kill criteria: the exact live repros from the task ---

def test_kill_repro_brier_true_returns_no_zero():
    with pytest.raises(ValueError, match=r"SF-BL-10939"):
        brier_score([True], [1])


def test_kill_repro_calibration_bool_panel_returns_no_ece_zero():
    with pytest.raises(ValueError, match=r"SF-BL-10939"):
        calibration_stats([True, False], [1, 0])


def test_kill_repro_log_score_true():
    with pytest.raises(ValueError, match=r"SF-BL-10939"):
        log_score([True], [1])


# --- every bool form, every scorer ---

BOOL_PANELS = [
    [True],
    [False],
    [True, False],
    [0.5, True],
    [True, 0.5],
    [0.5, False],
    (True, False),
    [[True, False], [0.5, 0.5]],
    np.array([True, False]),
    np.array([True]),
    np.array([0.5, True], dtype=object),
    np.array([True, False], dtype=object),
    np.bool_(True),
    True,
]


@pytest.mark.parametrize("fn", SCORERS_2)
@pytest.mark.parametrize("bad", BOOL_PANELS)
def test_bool_probability_raises(fn, bad):
    n = len(np.asarray(bad, dtype=object).ravel()) if not isinstance(bad, (bool, np.bool_)) else 1
    outcomes = [1] * n if n == 1 else ([1, 0] * ((n // 2) + 1))[:n]
    with pytest.raises(ValueError, match=r"bool"):
        _call(fn, bad, outcomes)


@pytest.mark.parametrize("bad", BOOL_PANELS)
def test_bool_probability_raises_calibration(bad):
    n = len(np.asarray(bad, dtype=object).ravel()) if not isinstance(bad, (bool, np.bool_)) else 1
    outcomes = [1] * n if n == 1 else ([1, 0] * ((n // 2) + 1))[:n]
    with pytest.raises(ValueError, match=r"bool"):
        calibration_stats(bad, outcomes)


def test_bool_message_is_named():
    for fn in (brier_score, log_score):
        with pytest.raises(ValueError, match=r"SF-BL-10939"):
            fn([True], [1])
    with pytest.raises(ValueError, match=r"SF-BL-10939"):
        calibration_stats([True], [1])


# --- lawful panels bit-identical to pre-fix outputs (captured 2026-10-23) ---

def test_lawful_brier_bit_identical():
    assert brier_score([0.5, 0.6], [0, 1]) == 0.20500000000000002
    assert brier_score([0.0, 1.0], [0, 1]) == 0.0
    assert brier_score([1, 0], [1, 0]) == 0.0
    assert brier_score([0.2, 0.8], [1, 0]) == 0.6400000000000001
    assert brier_score(np.array([0.5, 0.6]), [0, 1]) == 0.20500000000000002
    assert brier_score(np.array([1.0, 0.0]), [1, 0]) == 0.0


def test_lawful_log_bit_identical():
    assert log_score([0.5, 0.6], [0, 1]) == 0.601986402162968
    assert log_score([0.0, 1.0], [0, 1]) == 1.000000500029089e-06
    assert log_score([1, 0], [1, 0]) == 1.000000500029089e-06
    assert log_score([0.2, 0.8], [1, 0]) == 1.6094379124341005


def test_lawful_calibration_bit_identical():
    assert calibration_stats([0.5, 0.6], [0, 1]) == {
        "ece": 0.050000000000000044, "mean_forecast": 0.55, "base_rate": 0.5}
    # Float endpoints are lawful: this ECE 0.0 is a real score, not the bug.
    assert calibration_stats([0.0, 1.0], [0, 1]) == {
        "ece": 0.0, "mean_forecast": 0.5, "base_rate": 0.5}
    assert calibration_stats([0.1, 0.9, 0.3, 0.7], [0, 1, 0, 1]) == {
        "ece": 0.2, "mean_forecast": 0.5, "base_rate": 0.5}
    assert calibration_stats([0.5, 0.6], [0, 1], n_bins=5)["ece"] == 0.050000000000000044


# --- bool OUTCOMES stay lawful (probs-only screen) ---

def test_bool_outcomes_stay_lawful():
    assert brier_score([0.5, 0.6], [True, False]) == 0.305
    assert log_score([0.5, 0.6], [True, False]) == 0.8047189562170501
    assert calibration_stats([0.5, 0.6], [True, False]) == {
        "ece": 0.050000000000000044, "mean_forecast": 0.55, "base_rate": 0.5}
    assert brier_score([0.5, 0.6], np.array([True, False])) == 0.305


# --- prior guards preserved ---

def test_prior_guards_still_raise():
    for fn in SCORERS_2:
        with pytest.raises(ValueError):
            fn([0.5, 0.6], [1])                 # length mismatch
        with pytest.raises(ValueError):
            fn([], [])                           # empty panel
        with pytest.raises(ValueError):
            fn([float("nan"), 0.5], [0, 1])      # non-finite forecast
        with pytest.raises(ValueError):
            fn([1.5], [1])                       # out of range
    with pytest.raises(ValueError):
        calibration_stats([0.5, 0.6], [1])
    with pytest.raises(ValueError):
        calibration_stats([], [])
    with pytest.raises(ValueError):
        calibration_stats([0.5], [1], n_bins=True)  # SF-BL-9786 bool n_bins
