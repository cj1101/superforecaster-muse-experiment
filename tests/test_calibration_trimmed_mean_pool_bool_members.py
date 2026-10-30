"""SF-BL-11020: trimmed_mean_pool must reject bool member panels.

Pre-fix, trimmed_mean_pool bypassed _as_matrix and coerced members
through np.asarray(probs, float) with no type screen, so
trimmed_mean_pool([[True, False, True]]) returned [0.6667],
bit-identical to the lawful int control [[1, 0, 1]]. Post-fix every
bool member panel, at any trim level, raises a ValueError naming
probs (SF-BL-11020); lawful numeric panels (int endpoints included)
are unchanged.
Plain asserts, pytest-compatible, no pytest dependency beyond raises.
"""
import numpy as np
import pytest

from superforecaster.calibration import trimmed_mean_pool

TRIMS = (0, 0.1, 0.25, 0.4, 0.49)


def test_titled_bool_panel_raises_at_any_trim():
    for trim in TRIMS:
        with pytest.raises(ValueError, match="probs"):
            trimmed_mean_pool([[True, False, True]], trim=trim)


def test_bool_panel_message_cites_task():
    with pytest.raises(ValueError, match=r"probs.*SF-BL-11020"):
        trimmed_mean_pool([[True, False, True]])


def test_bool_forms_raise():
    forms = [
        [[True, False]], [True, False, True], [[0.7, True]], [[True, 0.5]],
        True, [[np.bool_(True), 0.5]],
        np.array([[True, False]]), np.array([True, False]),
        np.array([[True, 0.5]], dtype=object),
        [[0.7, 0.3], [0.2, False]],
    ]
    for panel in forms:
        with pytest.raises(ValueError, match="probs"):
            trimmed_mean_pool(panel)


def test_pandas_bool_panels_raise():
    pd = pytest.importorskip("pandas")
    for panel in (pd.Series([True, False]),
                  pd.Series([True, pd.NA], dtype="boolean"),
                  pd.DataFrame([[True, False, True]]),
                  pd.DataFrame({"a": [True, False], "b": [0.7, 0.3]})):
        with pytest.raises(ValueError, match="probs"):
            trimmed_mean_pool(panel)


def test_numeric_panels_unchanged():
    assert list(trimmed_mean_pool([[0.7, 0.3]])) == [0.5]
    # int endpoints are lawful: the screen is on type, not value
    assert list(trimmed_mean_pool([[1, 0, 1]])) == [2 / 3]
    assert list(trimmed_mean_pool(np.array([[0.7, 0.3]], dtype=object))) == [0.5]
    panel = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]
    assert float(trimmed_mean_pool(panel, trim=0.25)[0]) == pytest.approx(0.45)
    assert float(trimmed_mean_pool(panel, trim=0.1)[0]) == pytest.approx(0.4625)


def test_neighbouring_guards_unchanged():
    with pytest.raises(ValueError, match="SF-BL-11464"):
        trimmed_mean_pool([["0.7", "0.3"]])
    with pytest.raises(ValueError, match="SF-BL-11087"):
        trimmed_mean_pool([[0.7, 0.3]], trim=True)
