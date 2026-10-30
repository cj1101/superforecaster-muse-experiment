"""Repo tests for SF-BL-11125: string member forecasts must raise.

linear_pool / geometric_pool / log_odds_pool on [['0.7','0.3']]
previously all returned [0.5] because _as_matrix coerced string
members through np.asarray(dtype=float) with no type screen.
Post-fix every string member panel raises a ValueError naming
probs (SF-BL-11125); numeric panels are unchanged.
Plain asserts, pytest-compatible, no pytest dependency.
"""
import numpy as np
import pytest

from superforecaster.calibration import (
    extremized_pool, geometric_pool, linear_pool, log_odds_pool,
    optimize_pool_weights,
)

POOLS = (linear_pool, geometric_pool, log_odds_pool)


def test_titled_string_panel_raises_naming_probs():
    for pool in POOLS:
        with pytest.raises(ValueError, match="probs"):
            pool([["0.7", "0.3"]])


def test_string_panel_message_cites_task():
    for pool in POOLS:
        with pytest.raises(ValueError, match="SF-BL-11125"):
            pool([["0.7", "0.3"]])


def test_string_forms_raise():
    forms = [
        [["0.7", "0.3"]], [[0.7, "0.3"]], [["0.7", 0.3]], ["0.7", "0.3"],
        "0.7", np.array([["0.7", "0.3"]]), np.array([[b"0.7", b"0.3"]]),
        np.array([["0.7", "0.3"]], dtype=object), [["a", "b"]],
        [[np.str_("0.7"), np.str_("0.3")]], [[0.7, 0.3], ["0.2", 0.8]],
    ]
    for panel in forms:
        for pool in POOLS:
            with pytest.raises(ValueError, match="probs"):
                pool(panel)


def test_alias_and_optimize_raise():
    with pytest.raises(ValueError, match="probs"):
        extremized_pool([["0.7", "0.3"]])
    with pytest.raises(ValueError, match="probs"):
        optimize_pool_weights([0, 1] * 30, [["0.7", "0.3"]] * 60)


def test_numeric_panels_unchanged():
    for pool in POOLS:
        assert list(pool([[0.7, 0.3]])) == [0.5]
    assert list(linear_pool([[1, 0]])) == [0.5]
    assert list(linear_pool(np.array([[0.7, 0.3]], dtype=object))) == [0.5]
    panel = np.array([[0.9, 0.7], [0.2, 0.4], [0.5, 0.5]])
    assert np.allclose(linear_pool(panel), [0.8, 0.3, 0.5])
    assert np.allclose(geometric_pool(panel), log_odds_pool(panel, extremization=1.0))


def test_bool_and_weights_guards_unchanged():
    with pytest.raises(ValueError, match="SF-BL-10975"):
        linear_pool([[True, False]])
    with pytest.raises(ValueError, match="SF-BL-11090"):
        linear_pool([[0.7, 0.3]], weights=["0.5", "0.5"])
