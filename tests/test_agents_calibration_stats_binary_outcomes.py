"""SF-BL-10914 — agents.calibration_stats must reject fractional outcomes.

Pre-fix, calibration_stats([0.5, 0.6], [0.5, 1]) returned
{'ece': 0.2, 'mean_forecast': 0.55, 'base_rate': 0.75}: the SF-BL-10032
panel screen checked outcomes only for finiteness and [0, 1] range,
never for {0, 1} membership, so a half-resolution was averaged into
base_rate and the bin frequencies — while calibration._as_y and the
fixed agents.brier_score / log_score (SF-BL-10853 / SF-BL-9785) all
require binary outcomes. These tests pin the fixed contract: a named
ValueError (SF-BL-10914) on any outcome not in {0, 1}, fractional
FORECASTS still lawful, and binary panels byte-identical to pre-fix.
"""
import numpy as np
import pytest

from superforecaster.agents import calibration_stats

FRACTIONAL = [0.5, 0.25, 0.75, 0.1, 0.999999, 1e-9, -0.0 + 0.5]


def test_task_repro_kill_criterion():
    # Kill criterion: must NOT still return an ECE dict.
    with pytest.raises(ValueError, match="SF-BL-10914"):
        calibration_stats([0.5, 0.6], [0.5, 1])


@pytest.mark.parametrize("bad", FRACTIONAL)
def test_rejects_fractional_outcome(bad):
    with pytest.raises(ValueError, match="SF-BL-10914"):
        calibration_stats([0.5, 0.6], [bad, 1])


def test_rejects_fractional_outcome_anywhere_in_panel():
    with pytest.raises(ValueError, match="SF-BL-10914"):
        calibration_stats([0.2, 0.4, 0.6, 0.8], [0, 1, 0.5, 0])


def test_error_names_the_offending_values():
    with pytest.raises(ValueError, match=r"\[0\.5\]"):
        calibration_stats([0.5, 0.6], [0.5, 1])


def test_fractional_forecasts_stay_lawful():
    # Only outcomes are binary-checked; probs remain probabilities.
    out = calibration_stats([0.5, 0.6], [0, 1])
    assert out == {"ece": 0.050000000000000044,
                   "mean_forecast": 0.55, "base_rate": 0.5}


def test_binary_panels_byte_identical_to_prefix():
    # Values captured from the pre-fix implementation.
    cases = [
        (([0.5, 0.6], [0, 1], 10),
         {"ece": 0.050000000000000044, "mean_forecast": 0.55, "base_rate": 0.5}),
        (([0.1, 0.9, 0.3, 0.7], [1, 1, 0, 0], 10),
         {"ece": 0.5, "mean_forecast": 0.5, "base_rate": 0.5}),
        (([0.0, 1.0], [0, 1], 10),
         {"ece": 0.0, "mean_forecast": 0.5, "base_rate": 0.5}),
        (([0.5, 0.5], [True, False], 10),
         {"ece": 0.0, "mean_forecast": 0.5, "base_rate": 0.5}),
        (([0.5, 0.6], [0, 1], 5),
         {"ece": 0.050000000000000044, "mean_forecast": 0.55, "base_rate": 0.5}),
    ]
    for (p, o, nb), expected in cases:
        assert calibration_stats(p, o, n_bins=nb) == expected


def test_bool_and_numpy_outcomes_accepted():
    assert calibration_stats([0.5, 0.6], [True, False])["base_rate"] == 0.5
    assert calibration_stats(np.array([0.5, 0.6]),
                             np.array([0, 1]))["base_rate"] == 0.5
    assert calibration_stats(np.array([0.5, 0.6]),
                             np.array([0.0, 1.0]))["base_rate"] == 0.5


def test_negative_zero_outcome_is_zero():
    # -0.0 == 0.0, so it is a lawful binary outcome (sibling behaviour).
    assert calibration_stats([0.5], [-0.0])["base_rate"] == 0.0


def test_existing_screens_keep_precedence():
    # SF-BL-10032 fires before SF-BL-10914 for out-of-range / non-finite.
    with pytest.raises(ValueError, match="SF-BL-10032"):
        calibration_stats([0.5, 0.6], [2, 1])
    with pytest.raises(ValueError, match="SF-BL-10032"):
        calibration_stats([0.5, 0.6], [float("nan"), 1])
    with pytest.raises(ValueError, match="SF-BL-10032"):
        calibration_stats([0.5], [1, 0])
    with pytest.raises(ValueError, match="SF-BL-10032"):
        calibration_stats([], [])
