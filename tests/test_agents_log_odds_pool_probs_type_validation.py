"""SF-BL-11429 — agents.py log_odds_pool must type-validate probs and weights.

Pre-fix, log_odds_pool never type-validated its probs (or, for the
object form, its weights) against numeric strings:
log_odds_pool(['0.7', '0.3']) returned 0.5, bit-identical to the
numeric control, log_odds_pool([0.7, 0.3], weights=['1', '1'])
returned 0.5, and log_odds_pool([True, False]) returned
0.49999999999640554, bit-identical to the [1.0, 0.0] control — float
coercion in the logit pool silently parsed/coerced every form.
SF-BL-11023 fixtures only bool WEIGHTS for this function,
SF-BL-10935 only finite out-of-range member values, and
SF-BL-11090 string weights only in calibration.py's separate pools.

Done criterion: the guard raises ValueError naming probs/weights on
any string or object-dtype element in either argument, or any bool
element in probs, with float panels bit-identical.
Kill criterion: string probs or weights still return 0.5, or the
bool panel still returns 0.49999999999640554, instead of raising.
"""
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from superforecaster.agents import aggregate, AgentForecast, log_odds_pool  # noqa: E402


def _obj(values):
    return np.array(values, dtype=object)


def _bare_obj_arr(v):
    a = np.empty(2, dtype=object)
    a[0] = v
    a[1] = 0.3
    return a


STRING_PROBS = [
    ['0.7', '0.3'],
    ('0.7', '0.3'),
    [0.7, '0.3'],
    ['0.3', 0.7],
    [['0.7', '0.3']],
    np.array(['0.7', '0.3']),
    np.array([b'0.7', b'0.3']),
    _obj(['0.7', '0.3']),
    _obj([0.7, '0.3']),
    [b'0.7', b'0.3'],
    [bytearray(b'0.7'), 0.3],
    [memoryview(b'0.7'), 0.3],
    [np.str_('0.7'), 0.3],
    '0.7',
    np.str_('0.7'),
    ['abc', '0.3'],
    ['', '0.3'],
]

BOOL_PROBS = [
    [True, False],
    [False, True],
    [True, True],
    (True, False),
    [1.0, True],
    [False, 0.5],
    [np.bool_(True), 0.5],
    np.array([True, False]),
    np.array([True, False], dtype=object),
    _obj([0.5, True]),
    [[True], [False]],
    True,
    np.bool_(True),
]

OBJECT_PROBS = [
    _bare_obj_arr(object()),
    _bare_obj_arr(None),
    _obj([None, 0.3]),
    [object(), 0.3],
    _bare_obj_arr(bytearray(b'0.7')),   # text carrier in object array
    _bare_obj_arr(memoryview(b'0.7')),
]

STRING_WEIGHTS = [
    ['1', '1'],
    ['0.5', '0.5'],
    ('0.5', '0.5'),
    [0.5, '0.5'],
    np.array(['0.5', '0.5']),
    np.array(['0.5', '0.5'], dtype=object),
    _obj([0.5, '0.5']),
    [b'0.5', b'0.5'],
    '0.5',
]

OBJECT_WEIGHTS = [
    _obj([object(), 1.0]),
    _obj([None, 1.0]),
    [object(), 1.0],
    [memoryview(b'1'), 1.0],
]


# --- kill criteria: the exact live repros from the task ---

def test_kill_repro_string_probs_raises_not_0_5():
    with pytest.raises(ValueError, match=r"probs.*SF-BL-11429"):
        log_odds_pool(['0.7', '0.3'])


def test_kill_repro_string_weights_raises_not_0_5():
    with pytest.raises(ValueError, match=r"weights.*SF-BL-11429"):
        log_odds_pool([0.7, 0.3], weights=['1', '1'])


def test_kill_repro_bool_probs_raises_not_control_value():
    with pytest.raises(ValueError, match=r"probs.*SF-BL-11429"):
        log_odds_pool([True, False])


# --- every bad form, both arguments ---

@pytest.mark.parametrize("bad", STRING_PROBS)
def test_string_probs_raise(bad):
    with pytest.raises(ValueError, match=r"probs.*SF-BL-11429"):
        log_odds_pool(bad)


@pytest.mark.parametrize("bad", BOOL_PROBS)
def test_bool_probs_raise(bad):
    with pytest.raises(ValueError, match=r"probs.*SF-BL-11429"):
        log_odds_pool(bad)


@pytest.mark.parametrize("bad", OBJECT_PROBS)
def test_object_probs_raise(bad):
    with pytest.raises(ValueError, match=r"probs.*SF-BL-11429"):
        log_odds_pool(bad)


@pytest.mark.parametrize("bad", STRING_WEIGHTS)
def test_string_weights_raise_11429_and_preserve_11250(bad):
    with pytest.raises(ValueError, match=r"weights.*SF-BL-11429"):
        log_odds_pool([0.7, 0.3], weights=bad)
    with pytest.raises(ValueError, match=r"weights.*SF-BL-11250"):
        log_odds_pool([0.7, 0.3], weights=bad)


@pytest.mark.parametrize("bad", OBJECT_WEIGHTS)
def test_object_weights_raise_named_valueerror_not_typeerror(bad):
    with pytest.raises(ValueError, match=r"weights.*SF-BL-11429"):
        log_odds_pool([0.7, 0.3], weights=bad)


def test_bool_weights_raise_11429_and_preserve_11023():
    with pytest.raises(ValueError, match=r"weights.*SF-BL-11429"):
        log_odds_pool([0.7, 0.3], weights=[True, False])
    with pytest.raises(ValueError, match=r"weights.*SF-BL-11023"):
        log_odds_pool([0.7, 0.3], weights=[True, False])
    with pytest.raises(ValueError, match=r"weights.*SF-BL-11023"):
        log_odds_pool([0.7, 0.3], weights=np.array([True, False], dtype=object))


def test_message_contents():
    with pytest.raises(ValueError) as ei:
        log_odds_pool(['0.7', '0.3'])
    msg = str(ei.value)
    assert "probs" in msg and "string" in msg and "SF-BL-11429" in msg
    with pytest.raises(ValueError) as ei:
        log_odds_pool([True, False])
    msg = str(ei.value)
    assert "probs" in msg and "bool" in msg and "SF-BL-11429" in msg
    with pytest.raises(ValueError) as ei:
        log_odds_pool([0.7, 0.3], weights=_obj([object(), 1.0]))
    msg = str(ei.value)
    assert "weights" in msg and "object" in msg and "SF-BL-11429" in msg


def test_pandas_panels():
    pd = pytest.importorskip("pandas")
    with pytest.raises(ValueError, match=r"probs.*SF-BL-11429"):
        log_odds_pool(pd.Series(['0.7', '0.3']))
    with pytest.raises(ValueError, match=r"probs.*SF-BL-11429"):
        log_odds_pool(pd.Series([True, False]))
    with pytest.raises(ValueError, match=r"probs.*SF-BL-11429"):
        log_odds_pool(pd.Series(['0.7', '0.3'], dtype=object))
    with pytest.raises(ValueError, match=r"weights.*SF-BL-11429"):
        log_odds_pool([0.7, 0.3], weights=pd.Series(['1', '1']))
    assert log_odds_pool(pd.Series([0.7, 0.3])) == 0.5
    assert log_odds_pool([0.7, 0.3], weights=pd.Series([1.0, 1.0])) == 0.5


def test_first_bad_argument_is_probs():
    with pytest.raises(ValueError, match=r"probs"):
        log_odds_pool(['0.7', '0.3'], weights=['1', '1'])


# --- lawful panels bit-identical ---

def test_float_panels_bit_identical():
    assert log_odds_pool([0.7, 0.3]) == 0.5
    assert log_odds_pool([0.7, 0.3]).hex() == (0.5).hex()
    assert log_odds_pool([1.0, 0.0]) == 0.49999999999640554
    assert log_odds_pool([0.7, 0.3], weights=[1.0, 1.0]) == 0.5
    assert repr(log_odds_pool([0.7, 0.3], weights=[2.0, 1.0])) == "0.5701425119992315"
    assert log_odds_pool([0.7, 0.3], weights=[1, 0]) == 0.7   # int mask lawful
    assert log_odds_pool([1, 0]) == 0.49999999999640554       # int endpoints lawful
    assert log_odds_pool((0.7, 0.3)) == 0.5
    assert log_odds_pool(np.array([0.7, 0.3])) == 0.5
    assert log_odds_pool([0.5]) == 0.5


def test_object_dtype_numeric_panels_stay_lawful():
    # Fleet precedent (SF-BL-11433 / SF-BL-11393): object-dtype
    # panels holding only plain numbers are lawful.
    assert log_odds_pool(_obj([0.7, 0.3])) == 0.5
    assert log_odds_pool([0.7, 0.3], weights=_obj([1.0, 1.0])) == 0.5
    assert log_odds_pool(_obj([Decimal('0.7'), Decimal('0.3')])) == 0.5
    assert log_odds_pool([Fraction(7, 10), Fraction(3, 10)]) == log_odds_pool([0.7, 0.3])
    assert log_odds_pool([0.7, 0.3], weights=_obj([1, 2])) == \
        log_odds_pool([0.7, 0.3], weights=[1, 2])


# --- prior guards preserved ---

def test_prior_guards_still_raise():
    with pytest.raises(ValueError, match=r"SF-BL-8902"):
        log_odds_pool([])
    with pytest.raises(ValueError, match=r"SF-BL-8902"):
        log_odds_pool([], weights=['0.5'])
    with pytest.raises(ValueError, match=r"SF-BL-10233"):
        log_odds_pool([float('nan'), 0.3])
    with pytest.raises(ValueError, match=r"SF-BL-10233"):
        log_odds_pool([float('nan'), 0.3], weights=['0.5', '0.5'])
    with pytest.raises(ValueError, match=r"SF-BL-10935"):
        log_odds_pool([1.5, 0.5])
    with pytest.raises(ValueError, match=r"SF-BL-10935"):
        log_odds_pool([1.5, 0.3], weights=['0.5', '0.5'])
    with pytest.raises(ValueError, match=r"SF-BL-9788"):
        log_odds_pool([0.7, 0.3], weights=[float('nan'), 1.0])


def test_aggregate_path_unchanged_for_lawful_and_string_weights():
    fc = [AgentForecast(f"a{i}", "q", p) for i, p in enumerate([0.7, 0.3])]
    assert aggregate(fc, strategy='weighted_log_odds', weights=[0.5, 0.5]) == 0.5
    assert aggregate(fc, strategy='log_odds') == 0.5
    with pytest.raises(ValueError, match=r"weights.*SF-BL-11250"):
        aggregate(fc, strategy='weighted_log_odds', weights=['0.5', '0.5'])
