"""SF-BL-11053: log_odds_pool extremization TYPE validation.

The type dimension, distinct from SF-BL-9513/SF-BL-10313
(finiteness): a bool or non-numeric extremization must raise a
ValueError naming extremization — never pool silently as 1.0
(a=True pooled [[0.7, 0.6]] to 0.6517, bit-identical to a=1.0)
and never die with a bare TypeError (a='2' from the old <= guard).
Finite numeric factors are unchanged.
"""
import array as _array_mod
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from superforecaster import calibration as cal  # noqa: E402

PANEL = [[0.7, 0.6]]
REJECT = [
    True, False, np.bool_(True), np.bool_(False), np.array(True),
    np.array([True]), "2", "1.5", "", b"2", bytearray(b"2"),
    memoryview(b"2"), _array_mod.array("B", [50]), np.str_("2"), None,
    [2.0], (2.0,), {"a": 1}, {2.0}, np.array([2.0]), np.array([[2.0]]),
    2 + 0j, np.complex128(2), object(),
    float("nan"), float("inf"), float("-inf"), 0.0, -1.0,
    np.float64(float("nan")), Decimal("NaN"), Decimal("Infinity"),
]
LAWFUL = [1, 2, 0.05, 0.3, 0.5, 1.0, 1.25, 2.0, 4.0, 10.0,
          np.int64(2), np.float32(0.5), np.float64(2.0),
          Decimal("2"), Decimal("0.5"), Fraction(2, 1),
          np.array(2.0), np.array(1)]


def _raises_named(a):
    try:
        cal.log_odds_pool(PANEL, extremization=a)
    except ValueError as e:
        assert type(e) is ValueError
        assert "extremization" in str(e), str(e)
        assert "SF-BL-11053" in str(e), str(e)
        return True
    return False


def test_task_kill_legs():
    assert _raises_named(True)   # was: pooled to 0.6517 == a=1.0
    assert _raises_named("2")    # was: bare TypeError from <=


def test_reject_set():
    for a in REJECT:
        assert _raises_named(a), f"a={a!r} did not raise named ValueError"


def test_lawful_values_pinned():
    assert abs(float(cal.log_odds_pool(PANEL, extremization=1.0)[0]) - 0.65166852) < 1e-8
    assert abs(float(cal.log_odds_pool(PANEL, extremization=2.0)[0]) - 0.77777778) < 1e-8
    for a in LAWFUL:
        v = cal.log_odds_pool(PANEL, extremization=a)
        assert np.all(np.isfinite(v)), f"a={a!r} -> {v}"


def test_alias_guarded_and_documented():
    assert cal.extremized_pool is cal.log_odds_pool
    assert "SF-BL-11053" in cal.log_odds_pool.__doc__
    assert "SF-BL-11053" in cal._validate_extremization.__doc__
