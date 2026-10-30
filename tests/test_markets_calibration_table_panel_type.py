"""SF-BL-11465 — markets.py calibration_table panel type validation.

Done/kill criteria (prereg): calibration_table raises ValueError naming
the panel on any bool or string-dtype forecast or outcome panel, with
numeric tables bit-identical; kill if a [True, False] forecast panel
still produces endpoint mean_pred rows instead of raising.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (
    os.path.join(_HERE, "..", "..", "..", "markets"),          # backlog work dir
    os.path.join(_HERE, "..", "superforecaster"),              # repo tests/ dir
    os.path.join(_HERE, "..", "..", "superforecaster_out", "markets"),
):
    _cand = os.path.normpath(_cand)
    if os.path.isfile(os.path.join(_cand, "markets.py")):
        sys.path.insert(0, _cand)
        break
from markets import calibration_table, brier, log_score


def _assert_named_value_error(y, p, panel):
    with pytest.raises(ValueError) as ei:
        calibration_table(y, p)
    msg = str(ei.value)
    assert f"in {panel}" in msg, msg
    assert "calibration_table" in msg, msg


# ---- kill criterion -------------------------------------------------
def test_bool_forecast_panel_no_longer_tables_endpoint_rows():
    # Pre-fix this returned [{'mean_pred': 0.0, ...}, {'mean_pred': 1.0, ...}]
    _assert_named_value_error(np.array([1.0, 0.0]), np.array([True, False]), "p")


# ---- bool panels, both roles, many containers ------------------------
BOOL_P_FORECASTS = [
    np.array([True, False]),
    pd.Series([True, False]),
    pd.Series([True, False], dtype="boolean"),
    [True, False],
    (True, False),
    np.array([True, False], dtype=object),
    np.array([np.bool_(True), np.bool_(False)], dtype=object),
]
BOOL_Y_OUTCOMES = [
    np.array([True, False]),
    pd.Series([True, False]),
    pd.Series([True, False], dtype="boolean"),
    [True, False],
    np.array([True, False], dtype=object),
]


def test_bool_forecast_panels_rejected():
    y = np.array([1.0, 0.0])
    for p in BOOL_P_FORECASTS:
        _assert_named_value_error(y, p, "p")


def test_bool_outcome_panels_rejected():
    p = np.array([0.9, 0.1])
    for y in BOOL_Y_OUTCOMES:
        _assert_named_value_error(y, p, "y")


# ---- string-dtype panels, both roles ---------------------------------
STRING_P_FORECASTS = [
    np.array(["0.9", "0.1"]),
    np.array([b"0.9", b"0.1"]),
    pd.Series(["0.9", "0.1"]),
    pd.Series(["0.9", "0.1"], dtype="string"),
    pd.Index(["0.9", "0.1"]),
    ["0.9", "0.1"],
    np.array(["0.9", "0.1"], dtype=object),
    np.array(["abc", "def"]),  # non-numeric strings: named ValueError, not UFuncTypeError
    np.array([], dtype="<U1"),  # empty string-dtype: the dtype is the defect
    pd.Categorical(["0.9", "0.1"]),
]
STRING_Y_OUTCOMES = [
    np.array(["1", "0"]),
    pd.Series(["1", "0"]),
    pd.Series(["1", "0"], dtype="string"),
    ["1", "0"],
    np.array(["1", "0"], dtype=object),
]


def test_string_forecast_panels_rejected():
    y = np.array([1.0, 0.0])
    for p in STRING_P_FORECASTS:
        _assert_named_value_error(y, p, "p")


def test_string_outcome_panels_rejected():
    p = np.array([0.9, 0.1])
    for y in STRING_Y_OUTCOMES:
        _assert_named_value_error(y, p, "y")


def test_object_garbage_panels_rejected():
    # Shared screen (SF-BL-11433 helper) also rejects bare-object leaves.
    _assert_named_value_error(np.array([1.0, 0.0]),
                              np.array([object(), object()], dtype=object), "p")


# ---- lawful numeric panels: bit-identical tables ----------------------
# Expected values captured from the pre-fix implementation.
def test_numeric_tables_bit_identical():
    assert calibration_table(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == [
        {"bin_lo": 0.0, "bin_hi": 0.1, "n": 1, "mean_pred": 0.0, "empirical_rate": 0.0},
        {"bin_lo": 0.9, "bin_hi": 1.0, "n": 1, "mean_pred": 1.0, "empirical_rate": 1.0},
    ]
    assert calibration_table(np.array([1.0, 0.0, 1.0, 0.0]),
                             np.array([0.9, 0.1, 0.8, 0.2])) == [
        {"bin_lo": 0.1, "bin_hi": 0.2, "n": 1, "mean_pred": 0.1, "empirical_rate": 0.0},
        {"bin_lo": 0.2, "bin_hi": 0.3, "n": 1, "mean_pred": 0.2, "empirical_rate": 0.0},
        {"bin_lo": 0.8, "bin_hi": 0.9, "n": 1, "mean_pred": 0.8, "empirical_rate": 1.0},
        {"bin_lo": 0.9, "bin_hi": 1.0, "n": 1, "mean_pred": 0.9, "empirical_rate": 1.0},
    ]
    assert calibration_table(np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
                             np.array([0.05, 0.15, 0.45, 0.55, 0.85, 0.95]), bins=5) == [
        {"bin_lo": 0.0, "bin_hi": 0.2, "n": 2, "mean_pred": 0.1, "empirical_rate": 0.5},
        {"bin_lo": 0.4, "bin_hi": 0.6, "n": 2, "mean_pred": 0.5, "empirical_rate": 0.5},
        {"bin_lo": 0.8, "bin_hi": 1.0, "n": 2, "mean_pred": 0.9, "empirical_rate": 0.5},
    ]
    # int panels, float Series, and object-dtype panels holding only
    # plain numbers stay lawful (fleet precedent, SF-BL-11433).
    assert calibration_table(np.array([1, 0, 1, 1, 0]), np.array([0, 1, 0, 1, 0]))[0]["n"] == 3
    assert calibration_table(pd.Series([1.0, 0.0, 1.0, 0.0]),
                             pd.Series([0.25, 0.35, 0.75, 0.65]))[0]["mean_pred"] == 0.25
    assert calibration_table(np.array([1.0, 0.0], dtype=object),
                             np.array([0.5, 0.5], dtype=object)) == [
        {"bin_lo": 0.5, "bin_hi": 0.6, "n": 2, "mean_pred": 0.5, "empirical_rate": 0.5},
    ]


# ---- sibling screens unchanged ----------------------------------------
def test_sibling_screens_unchanged():
    # The tag parameter defaults to SF-BL-11433: brier/log_score messages
    # and lawful values are bit-identical to before this task.
    with pytest.raises(ValueError) as ei:
        brier([1, 0], [True, False])
    assert "(SF-BL-11433)" in str(ei.value)
    assert brier(np.array([1.0, 0.0]), np.array([0.9, 0.1])) == 0.009999999999999998
    assert log_score(np.array([1.0, 0.0]), np.array([0.9, 0.1])) == -0.10536051565782628
