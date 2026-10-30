"""SF-BL-11124: peer_score must reject string p_agent/p_crowd panels.

Pre-fix, peer_score([1,0,1,0], ['0.7','0.3','0.6','0.4'], numeric crowd)
silently scored the text panel as lawful probabilities (the underlying
brier_score coerces via np.asarray(dtype=float)); where a downstream
screen existed it named generic 'p', never the peer panel. Post-fix
peer_score raises ValueError naming p_agent / p_crowd (SF-BL-11124),
and numeric panels are bit-identical. Round 2: pandas Series/Index/
Categorical string panels (object, string, category dtypes) are
screened too; numeric pandas panels stay lawful.
"""
import numpy as np
import pytest

from superforecaster.calibration import peer_score

Y = [1, 0, 1, 0]
NUM_AGENT = [0.7, 0.3, 0.6, 0.4]
NUM_CROWD = [0.6, 0.4, 0.5, 0.5]
STR_AGENT = ["0.7", "0.3", "0.6", "0.4"]
STR_CROWD = ["0.6", "0.4", "0.5", "0.5"]


def test_numeric_control_value():
    assert peer_score(Y, NUM_AGENT, NUM_CROWD) == pytest.approx(0.08)


def test_string_agent_raises_naming_p_agent():
    with pytest.raises(ValueError, match="p_agent.*SF-BL-11124"):
        peer_score(Y, STR_AGENT, NUM_CROWD)


def test_string_crowd_raises_naming_p_crowd():
    with pytest.raises(ValueError, match="p_crowd.*SF-BL-11124"):
        peer_score(Y, NUM_AGENT, STR_CROWD)


@pytest.mark.parametrize("panel", [
    ("0.7", "0.3", "0.6", "0.4"),
    [0.7, "0.3", 0.6, 0.4],
    np.array(["0.7", "0.3", "0.6", "0.4"]),
    np.array([b"0.7", b"0.3", b"0.6", b"0.4"]),
    np.array([0.7, "0.3", 0.6, 0.4], dtype=object),
    ["high", "low", "high", "low"],
    "0.7",
])
def test_string_agent_forms_raise(panel):
    with pytest.raises(ValueError, match="p_agent"):
        peer_score(Y, panel, NUM_CROWD)


@pytest.mark.parametrize("panel", [
    STR_CROWD,
    np.array(STR_CROWD),
    [0.6, 0.4, "0.5", 0.5],
])
def test_string_crowd_forms_raise(panel):
    with pytest.raises(ValueError, match="p_crowd"):
        peer_score(Y, NUM_AGENT, panel)


def test_both_string_names_p_agent_first():
    with pytest.raises(ValueError, match="p_agent"):
        peer_score(Y, STR_AGENT, STR_CROWD)


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
def test_pandas_string_agent_forms_raise():
    forms = [
        pd.Series(STR_AGENT),
        pd.Series(STR_AGENT, dtype="string"),
        pd.Series(STR_AGENT, dtype="category"),
        pd.Index(STR_AGENT),
        pd.Index(STR_AGENT, dtype="string"),
        pd.Categorical(STR_AGENT),
        pd.CategoricalIndex(STR_AGENT),
        pd.Series([b"0.7", b"0.3", b"0.6", b"0.4"], dtype=object),
        pd.Series([0.7, "0.3", 0.6, 0.4], dtype=object),
    ]
    for panel in forms:
        with pytest.raises(ValueError, match="p_agent.*SF-BL-11124"):
            peer_score(Y, panel, NUM_CROWD)


@requires_pandas
def test_pandas_string_crowd_forms_raise():
    forms = [
        pd.Series(STR_CROWD),
        pd.Series(STR_CROWD, dtype="string"),
        pd.Series(STR_CROWD, dtype="category"),
        pd.Index(STR_CROWD),
        pd.Index(STR_CROWD, dtype="string"),
        pd.Categorical(STR_CROWD),
        pd.CategoricalIndex(STR_CROWD),
    ]
    for panel in forms:
        with pytest.raises(ValueError, match="p_crowd.*SF-BL-11124"):
            peer_score(Y, NUM_AGENT, panel)


@requires_pandas
def test_pandas_both_string_names_p_agent_first():
    with pytest.raises(ValueError, match="p_agent"):
        peer_score(Y, pd.Series(STR_AGENT), pd.Series(STR_CROWD))


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
    # crowd position with value-matched crowd equivalents
    crowd_pairs = [
        (pd.Series(NUM_CROWD), NUM_CROWD),
        (pd.Series(NUM_CROWD, dtype="category"), NUM_CROWD),
        (pd.Index(NUM_CROWD), NUM_CROWD),
        (pd.Categorical(NUM_CROWD), NUM_CROWD),
    ]
    for panel, equiv in crowd_pairs:
        assert peer_score(Y, NUM_AGENT, panel) == \
            peer_score(Y, NUM_AGENT, equiv)
