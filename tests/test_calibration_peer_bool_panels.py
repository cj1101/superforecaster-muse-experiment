"""SF-BL-11389: peer_score must reject bool p_agent/p_crowd panels.

Pre-fix, peer_score([1, 0], [True, False], [0.5, 0.5]) returned 0.25 in
the out copy (bit-identical to the perfect numeric agent) and a bool
crowd panel returned -0.25; in the repo copy plain bool lists raised
only the downstream SF-BL-10938 error naming generic 'p', and pandas
bool panels returned 0.25 silently in both copies. Post-fix peer_score
screens the RAW panels and raises ValueError naming p_agent / p_crowd
(SF-BL-11389); numeric panels are bit-identical.
"""
import numpy as np
import pytest

from superforecaster.calibration import peer_score

Y = [1, 0, 1, 0]
NUM_AGENT = [0.7, 0.3, 0.6, 0.4]
NUM_CROWD = [0.6, 0.4, 0.5, 0.5]


def test_numeric_control_value():
    assert peer_score(Y, NUM_AGENT, NUM_CROWD) == pytest.approx(0.08)
    assert peer_score([1, 0], [1, 0], [0.5, 0.5]) == 0.25
    assert peer_score([1, 0], [0.5, 0.5], [1, 0]) == -0.25


def test_kill_repro_bool_agent_raises_naming_p_agent():
    with pytest.raises(ValueError, match="p_agent.*SF-BL-11389"):
        peer_score([1, 0], [True, False], [0.5, 0.5])


def test_bool_crowd_raises_naming_p_crowd():
    with pytest.raises(ValueError, match="p_crowd.*SF-BL-11389"):
        peer_score([1, 0], [0.5, 0.5], [True, False])


@pytest.mark.parametrize("panel", [
    (True, False),
    [True, 0.5],
    [0.7, False],
    np.array([True, False]),
    np.array([True, False], dtype=object),
    np.bool_(True),
    True,
    [np.bool_(True), np.bool_(False)],
])
def test_bool_agent_forms_raise(panel):
    with pytest.raises(ValueError, match="p_agent"):
        peer_score([1, 0], panel, [0.5, 0.5])


@pytest.mark.parametrize("panel", [
    [True, False],
    np.array([True, False]),
    [0.5, True],
    np.array([False, True], dtype=object),
    (False, True),
])
def test_bool_crowd_forms_raise(panel):
    with pytest.raises(ValueError, match="p_crowd"):
        peer_score([1, 0], [0.5, 0.5], panel)


def test_both_bool_names_p_agent_first():
    with pytest.raises(ValueError, match="p_agent"):
        peer_score([1, 0], [True, False], [True, False])


def test_string_agent_still_takes_precedence_over_bool_crowd():
    with pytest.raises(ValueError, match="p_agent.*SF-BL-11124"):
        peer_score(Y, ["0.7", "0.3", "0.6", "0.4"], [True, False, True, False])


def test_numeric_forms_unchanged():
    assert peer_score(Y, (0.7, 0.3, 0.6, 0.4), NUM_CROWD) == \
        peer_score(Y, NUM_AGENT, NUM_CROWD)
    assert peer_score(Y, [1, 0, 1, 0], NUM_CROWD) == \
        peer_score(Y, np.array([1.0, 0.0, 1.0, 0.0]), NUM_CROWD)


try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:  # pragma: no cover
    pd = None
    HAS_PANDAS = False

requires_pandas = pytest.mark.skipif(not HAS_PANDAS,
                                     reason="pandas not installed")


@requires_pandas
def test_pandas_bool_agent_forms_raise():
    forms = [
        pd.Series([True, False]),
        pd.Series([True, False], dtype="boolean"),
        pd.Series([True, False], dtype=object),
        pd.Series([0.7, True], dtype=object),
        pd.Index([True, False]),
        pd.Categorical([True, False]),
        pd.CategoricalIndex([True, False]),
        pd.Series([], dtype=bool),
        pd.Categorical([0.5, 0.5], categories=[0.5, True]),
    ]
    for panel in forms:
        with pytest.raises(ValueError, match="p_agent.*SF-BL-11389"):
            peer_score([1, 0], panel, [0.5, 0.5])


@requires_pandas
def test_pandas_bool_crowd_forms_raise():
    forms = [
        pd.Series([True, False]),
        pd.Series([True, False], dtype="boolean"),
        pd.Series([True, False], dtype=object),
        pd.Index([True, False]),
        pd.Categorical([True, False]),
    ]
    for panel in forms:
        with pytest.raises(ValueError, match="p_crowd.*SF-BL-11389"):
            peer_score([1, 0], [0.5, 0.5], panel)


@requires_pandas
def test_pandas_numeric_panels_unchanged():
    pairs = [
        (pd.Series(NUM_AGENT), NUM_AGENT),
        (pd.Series([1, 0, 1, 0]), [1, 0, 1, 0]),
        (pd.Series(NUM_AGENT, dtype=object), NUM_AGENT),
        (pd.Series(NUM_AGENT, dtype="category"), NUM_AGENT),
        (pd.Index(NUM_AGENT), NUM_AGENT),
        (pd.Categorical(NUM_AGENT), NUM_AGENT),
    ]
    for panel, equiv in pairs:
        assert peer_score(Y, panel, NUM_CROWD) == \
            peer_score(Y, equiv, NUM_CROWD)
    for panel, equiv in [(pd.Series(NUM_CROWD), NUM_CROWD),
                         (pd.Categorical(NUM_CROWD), NUM_CROWD)]:
        assert peer_score(Y, NUM_AGENT, panel) == \
            peer_score(Y, NUM_AGENT, equiv)
