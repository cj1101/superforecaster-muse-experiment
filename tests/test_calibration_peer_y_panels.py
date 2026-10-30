"""SF-BL-11461: peer_score must reject string/bool outcome panels y.

Pre-fix, peer_score(['1','0'], [0.9,0.1], [0.5,0.5]) and the
[True, False] mirror returned 0.24, bit-identical to the numeric
control, in the out copy (whose _as_y has no type screen); in the
repo copy plain forms raised only transitively via brier_score's
SF-BL-10938/11052 screens and every pandas y panel still returned
0.24. Post-fix peer_score screens the RAW y first (SF-BL-11461) and
raises ValueError naming y; numeric panels are bit-identical.
Pandas Series/Index/Categorical y panels (object, string, bool,
boolean, category dtypes) are screened too; numeric pandas panels
stay lawful.
"""
import numpy as np
import pytest

from superforecaster.calibration import peer_score

AGENT = [0.9, 0.1]
CROWD = [0.5, 0.5]


def test_numeric_control_value():
    assert peer_score([1, 0], AGENT, CROWD) == pytest.approx(0.24)


def test_task_string_y_raises_naming_y():
    with pytest.raises(ValueError, match="y must not contain string.*SF-BL-11461"):
        peer_score(["1", "0"], AGENT, CROWD)


def test_task_bool_y_raises_naming_y():
    with pytest.raises(ValueError, match="y must not contain boolean.*SF-BL-11461"):
        peer_score([True, False], AGENT, CROWD)


@pytest.mark.parametrize("panel", [
    ("1", "0"),
    np.array(["1", "0"]),
    np.array([b"1", b"0"]),
    np.array(["1", "0"], dtype=object),
    ["1", 0],
    [["1"], ["0"]],
    ["yes", "no"],
    "1",
])
def test_string_y_forms_raise(panel):
    with pytest.raises(ValueError, match="y must not contain string"):
        peer_score(panel, AGENT, CROWD)


@pytest.mark.parametrize("panel", [
    (True, False),
    np.array([True, False]),
    np.array([True, False], dtype=object),
    [True, 0],
    True,
])
def test_bool_y_forms_raise(panel):
    with pytest.raises(ValueError, match="y must not contain boolean"):
        peer_score(panel, AGENT, CROWD)


def test_corrupt_y_named_before_forecast_panels():
    with pytest.raises(ValueError, match="y must not contain string"):
        peer_score(["1", "0"], ["0.9", "0.1"], CROWD)
    with pytest.raises(ValueError, match="y must not contain boolean"):
        peer_score([True, False], AGENT, [1.5, 0.5])


def test_numeric_y_forms_unchanged():
    control = peer_score([1, 0], AGENT, CROWD)
    for panel in [(1, 0), [1.0, 0.0], np.array([1, 0]),
                  np.array([1, 0], dtype=object)]:
        assert peer_score(panel, AGENT, CROWD) == control


try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:  # pragma: no cover
    pd = None
    HAS_PANDAS = False

requires_pandas = pytest.mark.skipif(not HAS_PANDAS,
                                     reason="pandas not installed")


@requires_pandas
def test_pandas_string_y_forms_raise():
    forms = [
        pd.Series(["1", "0"]),
        pd.Series(["1", "0"], dtype="string"),
        pd.Series(["1", "0"], dtype="category"),
        pd.Index(["1", "0"]),
        pd.Categorical(["1", "0"]),
        pd.CategoricalIndex(["1", "0"]),
        pd.array(["1", "0"], dtype="string"),
        pd.Series(["1", pd.NA], dtype="string"),
        pd.Series([], dtype="string"),
        pd.Categorical([1, 0], categories=[1, 0, "x"]),
    ]
    for panel in forms:
        with pytest.raises(ValueError, match="y must not contain string.*SF-BL-11461"):
            peer_score(panel, AGENT, CROWD)


@requires_pandas
def test_pandas_bool_y_forms_raise():
    forms = [
        pd.Series([True, False]),
        pd.Series([True, False], dtype="boolean"),
        pd.Series([True, False], dtype=object),
        pd.Index([True, False]),
        pd.Categorical([True, False]),
        pd.CategoricalIndex([True, False]),
        pd.array([True, False], dtype="boolean"),
        pd.Series([True, pd.NA], dtype="boolean"),
        pd.Series([], dtype=bool),
        pd.Categorical([2, 3], categories=[2, 3, True]),
    ]
    for panel in forms:
        with pytest.raises(ValueError, match="y must not contain boolean.*SF-BL-11461"):
            peer_score(panel, AGENT, CROWD)


@requires_pandas
def test_pandas_numeric_y_unchanged():
    control = peer_score([1, 0], AGENT, CROWD)
    for panel in [pd.Series([1, 0]), pd.Series([1.0, 0.0]),
                  pd.Series([1, 0], dtype="Int64"),
                  pd.Index([1, 0]), pd.Categorical([1, 0])]:
        assert peer_score(panel, AGENT, CROWD) == control
