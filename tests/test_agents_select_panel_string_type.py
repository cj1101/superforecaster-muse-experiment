"""SF-BL-11393 — select_extremize_factor must type-validate its panels.

Pre-fix, select_extremize_factor extremized each pooled value through
float(p) and passed outcomes to brier_score, so neither panel's type
was screened at this entry point: string pooled forecasts
['0.7', '0.6'] returned (0.9, 0.2248) bit-identical to the numeric
control, and a pandas object-dtype string outcome panel returned the
same pair (the SF-BL-11058 screen inside brier_score only ever sees
the already-extremized numeric array for pooled_probs, and does not
see inside a pandas object Series for outcomes). SF-BL-11057 fixtures
only this function's grid argument and SF-BL-11058 fixtures the
direct brier_score/calibration_stats entry points.

Done criterion: the guard raises ValueError naming pooled_probs or
outcomes on any string or object-dtype element, with float panels
returning bit-identical (factor, brier) pairs.
Kill criterion: select_extremize_factor(['0.7', '0.6'], [1, 0])
still returns (0.9, 0.2248) instead of raising.
"""
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from superforecaster.agents import select_extremize_factor  # noqa: E402

CONTROL = (0.9, 0.2247763556297586)  # ([0.7, 0.6], [1, 0]), pre-fix capture


def _obj(values):
    return np.array(values, dtype=object)


def _bare_obj_arr(v):
    a = np.empty(2, dtype=object)
    a[0] = v
    a[1] = 0.6
    return a


STRING_POOLED = [
    ['0.7', '0.6'],
    [0.7, '0.6'],
    ['0.6', 0.7],
    ('0.7', '0.6'),
    [['0.7', '0.6']],
    np.array(['0.7', '0.6']),
    np.array([b'0.7', b'0.6']),
    _obj(['0.7', '0.6']),
    _obj([0.7, '0.6']),
    [b'0.7', b'0.6'],
    [bytearray(b'0.7'), 0.6],
    [memoryview(b'0.7'), 0.6],
    _bare_obj_arr(bytearray(b'0.7')),
    _bare_obj_arr(memoryview(b'0.7')),
    np.str_('0.7'),
    '0.7',
    ['abc', '0.6'],   # non-numeric strings: named raise, not a bare parse error
    ['', '0.6'],
    _bare_obj_arr(object()),  # bare non-numeric object-dtype element
]

STRING_OUTCOMES = [
    ['1', '0'],
    [1, '0'],
    ('1', '0'),
    np.array(['1', '0']),
    np.array([b'1', b'0']),
    _obj(['1', '0']),
    _obj([1, '0']),
    [b'1', b'0'],
    '1',
]


def _n(panel):
    if isinstance(panel, (str, bytes, np.str_)):
        return 1
    return len(np.asarray(panel, dtype=object).ravel())


def _pooled(n):
    return ([0.7, 0.6] * ((n // 2) + 1))[:n]


def _outcomes(n):
    return ([1, 0] * ((n // 2) + 1))[:n]


# --- kill criterion: the exact live repro from the task ---

def test_kill_repro_string_pooled_raises():
    with pytest.raises(ValueError, match=r"pooled_probs.*SF-BL-11393"):
        select_extremize_factor(['0.7', '0.6'], [1, 0])


def test_string_outcomes_raise_at_this_entry_point():
    # Pre-fix this surfaced (when it surfaced at all) as a brier_score
    # error naming SF-BL-11058; the guard at THIS entry point must
    # name outcomes and SF-BL-11393.
    with pytest.raises(ValueError, match=r"outcomes.*SF-BL-11393"):
        select_extremize_factor([0.7, 0.6], ['1', '0'])


# --- every string form, both panels ---

@pytest.mark.parametrize("bad", STRING_POOLED)
def test_string_pooled_raises(bad):
    with pytest.raises(ValueError, match=r"pooled_probs"):
        select_extremize_factor(bad, _outcomes(_n(bad)))


@pytest.mark.parametrize("bad", STRING_OUTCOMES)
def test_string_outcomes_raise(bad):
    with pytest.raises(ValueError, match=r"outcomes"):
        select_extremize_factor(_pooled(_n(bad)), bad)


def test_pandas_string_panels_raise():
    pd = pytest.importorskip("pandas")
    with pytest.raises(ValueError, match=r"pooled_probs.*SF-BL-11393"):
        select_extremize_factor(pd.Series(['0.7', '0.6']), [1, 0])
    with pytest.raises(ValueError, match=r"outcomes.*SF-BL-11393"):
        # pre-fix this form RETURNED the control pair: the downstream
        # brier_score screen does not see inside a pandas object Series
        select_extremize_factor([0.7, 0.6], pd.Series(['1', '0'], dtype=object))
    assert select_extremize_factor(pd.Series([0.7, 0.6]), [1, 0]) == CONTROL


def test_first_bad_panel_is_the_one_named():
    with pytest.raises(ValueError, match=r"pooled_probs"):
        select_extremize_factor(['0.7', '0.6'], ['1', '0'])


# --- lawful panels bit-identical ---

def test_lawful_panels_bit_identical():
    assert select_extremize_factor([0.7, 0.6], [1, 0]) == CONTROL
    assert select_extremize_factor(np.array([0.7, 0.6]), np.array([1, 0])) == CONTROL
    assert select_extremize_factor((0.7, 0.6), (1, 0)) == CONTROL
    assert select_extremize_factor([0.7, 0.3], [1, 0]) == (2.0, 0.02407847800237812)
    assert select_extremize_factor([0.0, 1.0], [0, 1]) == (2.0, 1.000090904537999e-24)
    assert select_extremize_factor([1, 0], [1, 0], grid=[1.0, 1.3]) == (1.3, 2.5118928979660866e-16)


def test_object_dtype_numeric_panels_stay_lawful():
    # Fleet precedent (SF-BL-11433): object-dtype panels holding only
    # plain numbers are lawful; only string / non-numeric elements raise.
    assert select_extremize_factor(_obj([0.7, 0.6]), [1, 0]) == CONTROL
    assert select_extremize_factor([0.7, 0.6], _obj([1, 0])) == CONTROL
    assert select_extremize_factor(_obj([Decimal('0.7'), Decimal('0.6')]), [1, 0]) == CONTROL
    assert select_extremize_factor([Fraction(7, 10), Fraction(3, 5)], [1, 0]) == CONTROL


def test_bool_outcomes_stay_lawful():
    # SF-BL-10853/10939 carve-out untouched: bool outcomes are not strings.
    assert select_extremize_factor([0.7, 0.6], [True, False]) == CONTROL
    assert select_extremize_factor([0.7, 0.6], _obj([True, False])) == CONTROL


# --- prior guards preserved ---

def test_prior_guards_still_raise():
    with pytest.raises(ValueError, match=r"SF-BL-10192"):
        select_extremize_factor([float('nan'), 0.5], [0, 1])   # non-finite pooled
    with pytest.raises(ValueError, match=r"SF-BL-10192"):
        select_extremize_factor([0.5, 0.6], [1])               # length mismatch
    with pytest.raises(ValueError, match=r"SF-BL-10192"):
        select_extremize_factor([], [])                        # empty panel
    with pytest.raises(ValueError, match=r"SF-BL-11015"):
        select_extremize_factor([1.5, 0.5], [1, 0])            # out-of-range pooled
    with pytest.raises(ValueError, match=r"SF-BL-11057"):
        select_extremize_factor([0.7, 0.6], [1, 0], grid=['0.8', '1.3'])  # string grid
    with pytest.raises(ValueError):
        select_extremize_factor([None, 0.6], [1, 0])           # None pooled (finite screen)
