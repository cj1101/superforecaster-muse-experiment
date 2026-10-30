"""SF-BL-11091 — extremize must type-validate factor and centre.

On the pre-guard source, extremize(0.8, factor=True) returned 0.8
exactly (extremizing silently disabled, vs 0.8584 at the 1.30 default),
factor='1.3' died with a bare TypeError, centre='0.5' with a bare
UFuncTypeError, and centre=True returned 0.0877 with the bool read as
a near-certain centre. SF-BL-8875 (factor) and SF-BL-10191 (p/centre)
added (str, bytes, bool, np.bool_) screens; SF-BL-11091 closes the
residual hole in the same family: bytearray and memoryview slipped
past both screens because float() text-parses them, so
factor=bytearray(b'1.3') and centre=memoryview(b'0.5') returned the
parsed values silently. These tests pin the full contract: any bool
or non-numeric factor/centre raises a ValueError naming the argument,
and numeric outputs are bit-identical to the unchecked core.
"""
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from superforecaster.agents import (
    AgentForecast,
    _extremize_core,
    aggregate,
    extremize,
)

BAD = [
    True, False, np.bool_(True), np.bool_(False),
    "1.3", "0.5", b"1.3", b"0.5",
    bytearray(b"1.3"), bytearray(b"0.5"),
    memoryview(b"1.3"), memoryview(b"0.5"),
    None, [1.3], (1.3,), {"x": 1.3}, {1.3}, complex(1.3, 0.0),
]


@pytest.mark.parametrize("bad", BAD)
def test_factor_type_rejected(bad):
    with pytest.raises(ValueError, match="factor"):
        extremize(0.8, factor=bad)
    with pytest.raises(ValueError, match="factor"):
        extremize(0.8, bad)  # positional


@pytest.mark.parametrize("bad", BAD)
def test_centre_type_rejected(bad):
    with pytest.raises(ValueError, match="centre"):
        extremize(0.8, centre=bad)
    with pytest.raises(ValueError, match="centre"):
        extremize(0.8, 1.3, bad)  # positional


def test_task_kill_cases():
    # Pre-guard: factor=True -> 0.8, centre=True -> 0.0877 (floats),
    # factor='1.3' -> bare TypeError, centre='0.5' -> UFuncTypeError.
    for kwargs, needle in [
        ({"factor": True}, "factor"), ({"centre": True}, "centre"),
        ({"factor": "1.3"}, "factor"), ({"centre": "0.5"}, "centre"),
        ({"factor": bytearray(b"1.3")}, "factor"),
        ({"centre": memoryview(b"0.5")}, "centre"),
    ]:
        with pytest.raises(ValueError, match=needle):
            extremize(0.8, **kwargs)
    assert extremize(0.8) == 0.8584144223701332


def test_aggregate_factor_type_rejected():
    fc = [AgentForecast(agent_id="a", question_id="q", probability=0.8),
          AgentForecast(agent_id="b", question_id="q", probability=0.7)]
    for bad in (True, "1.3", bytearray(b"1.3"), memoryview(b"1.3"), None):
        with pytest.raises(ValueError, match="factor"):
            aggregate(fc, strategy="weighted_extremized", extremize_factor=bad)


# --- Round 2 (SF-BL-11091-R2): bool/str/bytes carried inside np.ndarray.
# The round-1 tester showed the scalar screens did not see inside arrays:
# centre=np.array(True) returned 0.0877 (the centre=True kill value) and
# centre=np.array('0.5') returned the parsed result. The factor half was
# landed by SF-BL-11126; the endpoint (p/centre) half is SF-BL-11091-R2.

def _obj_arr(v):
    a = np.empty(1, dtype=object)
    a[0] = v
    return a


BAD_FACTOR_ARRAYS = [
    np.array(True), np.array([True]), np.array([True], dtype=object),
    np.array("1.3"), np.array(["1.3"]), np.array(b"1.3"), np.array([b"1.3"]),
    np.array(["1.3"], dtype=object), _obj_arr(bytearray(b"1.3")),
    _obj_arr(memoryview(b"1.3")),
]

BAD_CENTRE_ARRAYS = [
    np.array(True), np.array([True]), np.array([True], dtype=object),
    np.array("0.5"), np.array(["0.5"]), np.array(b"0.5"), np.array([b"0.5"]),
    np.array(["0.5"], dtype=object), _obj_arr(bytearray(b"0.5")),
    _obj_arr(memoryview(b"0.5")), _obj_arr(b"0.5"),
]


@pytest.mark.parametrize("bad", BAD_FACTOR_ARRAYS)
def test_factor_array_carrier_rejected(bad):
    with pytest.raises(ValueError, match="factor"):
        extremize(0.8, factor=bad)
    with pytest.raises(ValueError, match="factor"):
        extremize(0.8, bad)  # positional


@pytest.mark.parametrize("bad", BAD_CENTRE_ARRAYS)
def test_centre_array_carrier_rejected(bad):
    with pytest.raises(ValueError, match="centre"):
        extremize(0.8, centre=bad)
    with pytest.raises(ValueError, match="centre"):
        extremize(0.8, 1.3, bad)  # positional


def test_p_array_carrier_rejected():
    for bad in (np.array(True), np.array([True]), np.array("0.8"),
                np.array([b"0.8"]), _obj_arr("0.8")):
        with pytest.raises(ValueError, match="p"):
            extremize(bad)


def test_aggregate_factor_array_carrier_rejected():
    fc = [AgentForecast(agent_id="a", question_id="q", probability=0.8),
          AgentForecast(agent_id="b", question_id="q", probability=0.7)]
    for bad in (np.array([True]), np.array("1.3")):
        with pytest.raises(ValueError, match="factor"):
            aggregate(fc, strategy="weighted_extremized", extremize_factor=bad)


def test_numeric_arrays_accepted_bit_identical():
    # Numeric-dtype arrays stay lawful (SF-BL-11057 precedent).
    for p in (0.8, np.array(0.8), np.array([0.8])):
        for f in (np.array(1.3), np.array([1.3]), np.array(1)):
            for c in (np.array(0.5), np.array([0.5]),
                      np.array([0.5], dtype=object)):
                got = extremize(p, factor=f, centre=c)
                want = _extremize_core(float(p), float(f), float(c))
                assert float.hex(got) == float.hex(want), (p, f, c)
    assert extremize(0.8, factor=np.array(1.3)) == 0.8584144223701332
    assert extremize(0.8, centre=np.array(0.5)) == 0.8584144223701332


def test_numeric_outputs_bit_identical():
    for p in (0.01, 0.1, 0.3, 0.5, 0.8, 0.99):
        for f in (1.00, 1.15, 1.30, 1.40, 1, np.float64(1.3),
                  Decimal("1.3"), Fraction(13, 10)):
            for c in (0.3, 0.5, 0.7, np.float64(0.5), Decimal("0.5")):
                got = extremize(p, factor=f, centre=c)
                want = _extremize_core(float(p), float(f), float(c))
                assert float.hex(got) == float.hex(want), (p, f, c)
