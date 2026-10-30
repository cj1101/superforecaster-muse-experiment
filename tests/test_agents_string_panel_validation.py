"""SF-BL-11058: agents.py scorers must reject string panels.

Bug: brier_score / log_score / calibration_stats convert with
np.asarray(dtype=float), which silently parses numeric strings in
either argument: brier_score(['0.5'], [1]) returned 0.25,
brier_score([0.5, 0.6], ['1', '0']) returned 0.305,
log_score(['0.5'], [1]) returned ln(2), and
calibration_stats(['0.5', '0.6'], [1, 0]) returned ECE 0.05 with
mean_forecast 0.55. Differs from SF-BL-10939 (bool probabilities)
and SF-BL-10853 (out-of-range / fractional) in input dimension:
string type.

Done criterion: named ValueError (naming probs or outcomes) on any
string-dtype panel in all three scorers; numeric panels bit-identical.
Kill criterion: brier_score(['0.5'], [1]) or
calibration_stats(['0.5', '0.6'], [1, 0]) returns a score instead
of raising.
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


# --- kill criteria: the exact live repros from the task ---

def test_kill_repro_brier_string_probs_returns_no_quarter():
    with pytest.raises(ValueError, match=r"probs.*SF-BL-11058"):
        brier_score(['0.5'], [1])


def test_kill_repro_brier_string_outcomes():
    with pytest.raises(ValueError, match=r"outcomes.*SF-BL-11058"):
        brier_score([0.5, 0.6], ['1', '0'])


def test_kill_repro_calibration_string_probs_returns_no_ece():
    with pytest.raises(ValueError, match=r"probs.*SF-BL-11058"):
        calibration_stats(['0.5', '0.6'], [1, 0])


def test_kill_repro_calibration_string_outcomes():
    with pytest.raises(ValueError, match=r"outcomes.*SF-BL-11058"):
        calibration_stats([0.5, 0.6], ['1', '0'])


def test_kill_repro_log_string_probs():
    with pytest.raises(ValueError, match=r"probs.*SF-BL-11058"):
        log_score(['0.5'], [1])


def test_kill_repro_log_string_outcomes():
    with pytest.raises(ValueError, match=r"outcomes.*SF-BL-11058"):
        log_score([0.5, 0.6], ['1', '0'])


# --- every string form, every scorer, both arguments ---

STRING_PROB_PANELS = [
    ['0.5'],
    ['0.5', '0.6'],
    [0.5, '0.6'],
    ['0.6', 0.5],
    ('0.5', '0.6'),
    [['0.5', '0.6']],
    np.array(['0.5', '0.6']),
    np.array(['0.5']),
    np.array([b'0.5', b'0.6']),
    np.array(['0.5', '0.6'], dtype=object),
    np.array([0.5, '0.6'], dtype=object),
    [b'0.5'],
    '0.5',
    np.str_('0.5'),
    ['abc'],            # non-numeric strings: named raise, not a bare parse error
    [''],
]


def _n(panel):
    if isinstance(panel, (str, bytes, np.str_)):
        return 1
    return len(np.asarray(panel, dtype=object).ravel())


def _outcomes(n):
    return ([1, 0] * ((n // 2) + 1))[:n]


@pytest.mark.parametrize("fn", SCORERS_2)
@pytest.mark.parametrize("bad", STRING_PROB_PANELS)
def test_string_probs_raise(fn, bad):
    with pytest.raises(ValueError, match=r"probs"):
        fn(bad, _outcomes(_n(bad)))


@pytest.mark.parametrize("bad", STRING_PROB_PANELS)
def test_string_probs_raise_calibration(bad):
    with pytest.raises(ValueError, match=r"probs"):
        calibration_stats(bad, _outcomes(_n(bad)))


STRING_OUTCOME_PANELS = [
    ['1'],
    ['1', '0'],
    [1, '0'],
    ('1', '0'),
    np.array(['1', '0']),
    np.array(['1', '0'], dtype=object),
    [b'1', b'0'],
    '1',
]


@pytest.mark.parametrize("fn", SCORERS_2)
@pytest.mark.parametrize("bad", STRING_OUTCOME_PANELS)
def test_string_outcomes_raise(fn, bad):
    probs = [0.5, 0.6][: _n(bad)] if _n(bad) > 1 else [0.5]
    with pytest.raises(ValueError, match=r"outcomes"):
        fn(probs, bad)


@pytest.mark.parametrize("bad", STRING_OUTCOME_PANELS)
def test_string_outcomes_raise_calibration(bad):
    probs = [0.5, 0.6][: _n(bad)] if _n(bad) > 1 else [0.5]
    with pytest.raises(ValueError, match=r"outcomes"):
        calibration_stats(probs, bad)


def test_message_names_argument_and_tag():
    for fn in SCORERS_2:
        with pytest.raises(ValueError, match=r"SF-BL-11058"):
            fn(['0.5'], [1])
        with pytest.raises(ValueError, match=r"SF-BL-11058"):
            fn([0.5], ['1'])
    with pytest.raises(ValueError, match=r"SF-BL-11058"):
        calibration_stats(['0.5'], [1])
    with pytest.raises(ValueError, match=r"SF-BL-11058"):
        calibration_stats([0.5], ['1'])


# --- lawful numeric panels bit-identical (captured pre-fix, 2026-10-23) ---

def test_lawful_brier_bit_identical():
    assert brier_score([0.5, 0.6], [0, 1]) == 0.20500000000000002
    assert brier_score([0.0, 1.0], [0, 1]) == 0.0
    assert brier_score([1, 0], [1, 0]) == 0.0
    assert brier_score([0.2, 0.8], [1, 0]) == 0.6400000000000001
    assert brier_score(np.array([0.5, 0.6]), [0, 1]) == 0.20500000000000002
    assert brier_score((0.5, 0.6), (0, 1)) == 0.20500000000000002


def test_lawful_log_bit_identical():
    assert log_score([0.5, 0.6], [0, 1]) == 0.601986402162968
    assert log_score([0.0, 1.0], [0, 1]) == 1.000000500029089e-06
    assert log_score([1, 0], [1, 0]) == 1.000000500029089e-06
    assert log_score([0.2, 0.8], [1, 0]) == 1.6094379124341005


def test_lawful_calibration_bit_identical():
    assert calibration_stats([0.5, 0.6], [0, 1]) == {
        "ece": 0.050000000000000044, "mean_forecast": 0.55, "base_rate": 0.5}
    assert calibration_stats([0.0, 1.0], [0, 1]) == {
        "ece": 0.0, "mean_forecast": 0.5, "base_rate": 0.5}
    assert calibration_stats([0.1, 0.9, 0.3, 0.7], [0, 1, 0, 1]) == {
        "ece": 0.2, "mean_forecast": 0.5, "base_rate": 0.5}
    assert calibration_stats([0.5, 0.6], [0, 1], n_bins=5)["ece"] == 0.050000000000000044


def test_bool_outcomes_stay_lawful():
    # SF-BL-10939/10853 carve-out is untouched: bool outcomes are not strings.
    assert brier_score([0.5, 0.6], [True, False]) == 0.305
    assert log_score([0.5, 0.6], [True, False]) == 0.8047189562170501
    assert calibration_stats([0.5, 0.6], [True, False]) == {
        "ece": 0.050000000000000044, "mean_forecast": 0.55, "base_rate": 0.5}


# --- prior guards preserved ---

def test_prior_guards_still_raise():
    for fn in SCORERS_2:
        with pytest.raises(ValueError, match=r"SF-BL-10939"):
            fn([True], [1])                      # bool probs (SF-BL-10939)
        with pytest.raises(ValueError):
            fn([0.5, 0.6], [1])                  # length mismatch
        with pytest.raises(ValueError):
            fn([], [])                           # empty panel
        with pytest.raises(ValueError):
            fn([float("nan"), 0.5], [0, 1])      # non-finite forecast
        with pytest.raises(ValueError):
            fn([1.5], [1])                       # out of range
        with pytest.raises(ValueError):
            fn([0.5], [2])                       # non-binary outcome
    with pytest.raises(ValueError):
        calibration_stats([0.5, 0.6], [1])
    with pytest.raises(ValueError):
        calibration_stats([], [])
    with pytest.raises(ValueError):
        calibration_stats([0.5], [1], n_bins=True)  # SF-BL-9786 bool n_bins
