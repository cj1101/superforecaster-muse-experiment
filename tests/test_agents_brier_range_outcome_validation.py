"""SF-BL-10853: agents.brier_score must reject out-of-range forecasts
and non-binary outcomes.

Bug: brier_score's SF-BL-9784 guard checked only length, emptiness and
finiteness, so brier_score([1.5, 0.6], [0, 1]) returned 1.205 (above the
lawful binary-Brier maximum 1.0) and brier_score([0.5, 0.6], [0.5, 1])
returned 0.08 — a fractional outcome silently scored. The sibling
log_score in the same module already raises on both (SF-BL-10031 /
SF-BL-9785), as does calibration.py (SF-BL-9484).

Done criterion: named ValueError on any p outside [0, 1] or any outcome
not in {0, 1}; lawful panels bit-identical to pre-fix outputs.
Kill criterion: p=1.5 or outcome=0.5 still returns a float.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from superforecaster.agents import brier_score  # noqa: E402


# --- kill criteria: the two live repros from the task ---

def test_kill_repro_out_of_range_forecast():
    with pytest.raises(ValueError, match=r"SF-BL-10853"):
        brier_score([1.5, 0.6], [0, 1])


def test_kill_repro_fractional_outcome():
    with pytest.raises(ValueError, match=r"SF-BL-10853"):
        brier_score([0.5, 0.6], [0.5, 1])


# --- forecast range screen ---

@pytest.mark.parametrize("bad_p", [-0.1, -1.0, 1.0000001, 1.5, 2.0, 82.0])
def test_out_of_range_forecast_raises(bad_p):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier_score([0.5, bad_p], [0, 1])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier_score([bad_p], [1])


# --- binary outcome screen ---

@pytest.mark.parametrize("bad_y", [0.5, 0.999, -1, 2, -0.5, 1.5])
def test_non_binary_outcome_raises(bad_y):
    with pytest.raises(ValueError, match=r"exactly 0 or 1"):
        brier_score([0.5, 0.6], [0, bad_y])
    with pytest.raises(ValueError, match=r"exactly 0 or 1"):
        brier_score([0.5], [bad_y])


# --- prior SF-BL-9784 guards preserved ---

def test_prior_guards_still_raise():
    with pytest.raises(ValueError):
        brier_score([0.5, 0.6], [1])            # length mismatch
    with pytest.raises(ValueError):
        brier_score([], [])                      # empty panel
    with pytest.raises(ValueError):
        brier_score([float("nan"), 0.5], [0, 1])  # non-finite forecast
    with pytest.raises(ValueError):
        brier_score([0.5, 0.6], [0, float("inf")])  # non-finite outcome


def test_range_screen_runs_after_finiteness():
    # NaN is invisible to range comparisons; the SF-BL-9784 finiteness
    # error must win, exactly as in _validate_log_score_inputs.
    with pytest.raises(ValueError, match=r"SF-BL-9784"):
        brier_score([float("nan")], [1])


# --- lawful panels bit-identical to pre-fix outputs (captured 2026-10-23) ---

LAWFUL = [
    (([0.5, 0.6], [0, 1]), 0.20500000000000002),
    (([0.0, 1.0], [0, 1]), 0.0),
    (([0.0, 1.0], [1, 0]), 1.0),
    (([0.2, 0.8], [1, 0]), 0.6400000000000001),
    (([0.5], [1]), 0.25),
    (([0.5], [0]), 0.25),
    (([1.0], [1]), 0.0),
    (([0.0], [0]), 0.0),
    (([0.1, 0.9, 0.3, 0.7], [0, 1, 0, 1]), 0.05),
    (([0.25, 0.25], [0, 0]), 0.0625),
]


@pytest.mark.parametrize("panel,expected", LAWFUL)
def test_lawful_panels_bit_identical(panel, expected):
    assert brier_score(*panel) == expected


def test_bool_and_numpy_outcomes_bit_identical():
    assert brier_score([0.5, 0.6], [True, False]) == 0.305
    assert brier_score(np.array([0.5, 0.6]), np.array([0, 1])) == 0.20500000000000002


def test_negative_zero_outcome_is_zero():
    # -0.0 == 0.0 in IEEE-754; every sibling validator (log_score
    # SF-BL-9785, calibration._as_y) treats it as the lawful outcome 0.
    assert brier_score([0.5], [-0.0]) == 0.25
