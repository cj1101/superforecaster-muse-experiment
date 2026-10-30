"""SF-BL-11464: trimmed_mean_pool must reject string member panels.

Pre-fix, trimmed_mean_pool bypassed _as_matrix and coerced members
through np.asarray(probs, float) with no type screen, so
trimmed_mean_pool([['0.7','0.3']]) returned [0.5], bit-identical to
the numeric control. Post-fix every string member panel, at any
trim level, raises a ValueError naming probs (SF-BL-11464); lawful
numeric panels are unchanged.
Plain asserts, pytest-compatible, no pytest dependency beyond raises.
"""
import numpy as np
import pytest

from superforecaster.calibration import trimmed_mean_pool

TRIMS = (0, 0.1, 0.25, 0.4, 0.49)


def test_titled_string_panel_raises_at_any_trim():
    for trim in TRIMS:
        with pytest.raises(ValueError, match="probs"):
            trimmed_mean_pool([["0.7", "0.3"]], trim=trim)


def test_string_panel_message_cites_task():
    with pytest.raises(ValueError, match=r"probs.*SF-BL-11464"):
        trimmed_mean_pool([["0.7", "0.3"]])


def test_string_forms_raise():
    forms = [
        [["0.7", "0.3"]], [[0.7, "0.3"]], [["0.7", 0.3]], ["0.7", "0.3"],
        "0.7", np.array([["0.7", "0.3"]]), np.array([[b"0.7", b"0.3"]]),
        np.array([["0.7", "0.3"]], dtype=object), [["a", "b"]],
        [[np.str_("0.7"), np.str_("0.3")]], [[0.7, 0.3], ["0.2", 0.8]],
    ]
    for panel in forms:
        with pytest.raises(ValueError, match="probs"):
            trimmed_mean_pool(panel)


def test_pandas_string_panels_raise():
    pd = pytest.importorskip("pandas")
    for panel in (pd.Series(["0.7", "0.3"]),
                  pd.Series(["0.7", "0.3"], dtype="string"),
                  pd.DataFrame([["0.7", "0.3"]])):
        with pytest.raises(ValueError, match="probs"):
            trimmed_mean_pool(panel)


def test_numeric_panels_unchanged():
    assert list(trimmed_mean_pool([[0.7, 0.3]])) == [0.5]
    assert list(trimmed_mean_pool([[1, 0]])) == [0.5]
    assert list(trimmed_mean_pool(np.array([[0.7, 0.3]], dtype=object))) == [0.5]
    panel = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]
    assert float(trimmed_mean_pool(panel, trim=0.25)[0]) == pytest.approx(0.45)
    assert float(trimmed_mean_pool(panel, trim=0.1)[0]) == pytest.approx(0.4625)


def test_trim_guard_unchanged():
    with pytest.raises(ValueError, match="SF-BL-11087"):
        trimmed_mean_pool([[0.7, 0.3]], trim=True)
