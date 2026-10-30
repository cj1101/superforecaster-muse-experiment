"""
test_calibration_calibrator_fit_string.py — SF-BL-11395 verification.

Prereg: Guard raises ValueError naming y/p on any string or
object-dtype element in all four calibrators' fit(), with float-panel
fitted parameters bit-identical; kill if
PlattScaler().fit([1,0,1,0,1,0], ['0.9','0.2','0.8','0.3','0.7','0.4'])
still returns A_=17.7628 instead of raising.

Fix: _reject_string_fit_input screens RAW y and p via _has_string +
_has_string_in_pandas_panel + _has_bare_object as the first
statements of every fit() (hence fit_calibrator), before float
conversion / intercept-only return.
Lawful bit-identical: SHA-256 over lawful fitted params (see
repro.py in SF-BL-11395 work dir) is
a23f7083c0c351b99db7aaa984f1a05fd611b538a22e161b5f2dd77842898068
both pre- and post-fix.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from superforecaster.calibration import (
    BetaCalibrator, IsotonicCalibrator, PlattScaler, TemperatureScaler,
    fit_calibrator,
)

PASS = []
CALS = [("Platt", PlattScaler), ("Beta", BetaCalibrator),
        ("Temperature", TemperatureScaler), ("Isotonic", IsotonicCalibrator)]
Y_NUM = [1, 0, 1, 0, 1, 0]
P_NUM = [0.9, 0.2, 0.8, 0.3, 0.7, 0.4]
P_STR = ['0.9', '0.2', '0.8', '0.3', '0.7', '0.4']
Y_STR = ['1', '0', '1', '0', '1', '0']

def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))

def _named_error(fn, name, *args):
    try:
        fn(*args)
    except ValueError as e:
        msg = str(e)
        assert "string" in msg and f"in {name}" in msg and "SF-BL-11395" in msg, msg
        return e
    return None

def test_kill_and_titled_raise():
    # Kill criterion explicit
    assert _named_error(PlattScaler().fit, "p", Y_NUM, P_STR) is not None
    for cal_name, cls in CALS:
        assert _named_error(cls().fit, "p", Y_NUM, P_STR) is not None, cal_name
        assert _named_error(cls().fit, "y", Y_STR, P_NUM) is not None, cal_name
        assert _named_error(cls().fit, "y", Y_STR, P_STR) is not None, cal_name

STRING_P = [
    P_STR, tuple(P_STR), '0.9', b'0.9', np.str_('0.9'),
    [0.9, '0.2', 0.8, 0.3, 0.7, 0.4], ['0.9', 0.2, 0.8, 0.3, 0.7, 0.4],
    [0.9, ['0.2'], 0.8, 0.3, 0.7, 0.4],
    np.array(P_STR), np.array(P_STR, dtype=object),
    np.array([0.9, '0.2', 0.8, 0.3, 0.7, 0.4], dtype=object),
    np.array([b'0.9', b'0.2', b'0.8', b'0.3', b'0.7', b'0.4']),
    np.array([P_STR]), ['abc'] * 6, [''] * 6,
    [object(), 0.2, 0.8, 0.3, 0.7, 0.4],
    np.array([object(), 0.2, 0.8, 0.3, 0.7, 0.4], dtype=object),
]
STRING_Y = [
    Y_STR, tuple(Y_STR), '1', np.array(Y_STR), np.array(Y_STR, dtype=object),
    [1, '0', 1, 0, 1, 0], ['1', 0, 1, 0, 1, 0],
    np.array([b'1', b'0', b'1', b'0', b'1', b'0']),
    [object(), 0, 1, 0, 1, 0],
]

def test_string_forms_raise():
    for cal_name, cls in CALS:
        for v in STRING_P:
            assert _named_error(cls().fit, "p", Y_NUM, v) is not None, (cal_name, repr(v))
        for v in STRING_Y:
            assert _named_error(cls().fit, "y", v, P_NUM) is not None, (cal_name, repr(v))

def test_pandas_string_panels_raise():
    try:
        import pandas as pd
    except Exception:
        return  # pandas optional; core forms covered elsewhere
    for cal_name, cls in CALS:
        assert _named_error(cls().fit, "p", Y_NUM, pd.Series(P_STR)) is not None, cal_name
        assert _named_error(cls().fit, "y", pd.Series(Y_STR), P_NUM) is not None, cal_name
        assert _named_error(cls().fit, "p", Y_NUM, pd.Index(P_STR)) is not None, cal_name
        # numeric pandas panels stay lawful
        m = cls().fit(pd.Series(Y_NUM), pd.Series(P_NUM))
        assert m is not None

def test_intercept_only_and_unfitted_state():
    # Constant-string p would be intercept-only if coerced; must raise first
    for cal_name, cls in CALS:
        assert _named_error(cls().fit, "p", Y_NUM, ['0.7'] * 6) is not None, cal_name
        assert _named_error(cls().fit, "y", Y_STR, [0.7] * 6) is not None, cal_name
    # Rejected Isotonic fit leaves calibrator unfitted
    iso = IsotonicCalibrator()
    assert _named_error(iso.fit, "p", Y_NUM, P_STR) is not None
    assert iso.x_ is None and iso.y_ is None
    iso_step = IsotonicCalibrator(interpolation="step")
    assert _named_error(iso_step.fit, "p", Y_NUM, P_STR) is not None
    assert iso_step.x_ is None
    # fit_calibrator inherits
    for nm in ("platt", "beta", "temperature", "isotonic"):
        assert _named_error(fit_calibrator, "p", nm, Y_NUM, P_STR) is not None, nm
        assert _named_error(fit_calibrator, "y", nm, Y_STR, P_NUM) is not None, nm

def test_lawful_bit_identical():
    platt = PlattScaler().fit(Y_NUM, P_NUM)
    assert platt.A_ == 17.762763687307135 and platt.B_ == -3.7957007060548054
    beta = BetaCalibrator().fit(Y_NUM, P_NUM)
    assert beta.a_ == 17.249465734206364 and beta.b_ == -16.39197958793144
    assert beta.c_ == -3.020478689885648
    temp = TemperatureScaler().fit(Y_NUM, P_NUM)
    assert temp.T_ == 0.05
    iso = IsotonicCalibrator().fit(Y_NUM, P_NUM)
    assert np.array_equal(iso.x_, np.array([0.2, 0.3, 0.4, 0.7, 0.8, 0.9]))
    assert np.array_equal(iso.y_, np.array([0., 0., 0., 1., 1., 1.]))
    # Lawful carriers give identical Platt params
    for y, p in [(tuple(Y_NUM), tuple(P_NUM)),
                 (np.array(Y_NUM), np.array(P_NUM)),
                 (np.array(Y_NUM, dtype=object), np.array(P_NUM, dtype=object)),
                 ([1, 0, 1, 0, 1, 0], [0.9, 0.2, 0.8, 0.3, 0.7, 0.4])]:
        m = PlattScaler().fit(y, p)
        assert m.A_ == platt.A_ and m.B_ == platt.B_, (y, p)
    # Int endpoints lawful
    m = PlattScaler().fit([1, 0, 1, 0], [1, 0, 1, 0])
    assert m.A_ == 1.1091951154466313

def test_preexisting_fit_screens_unchanged():
    def raises(fn, *a):
        try: fn(*a)
        except ValueError: return True
        return False
    for cal_name, cls in CALS:
        assert raises(cls().fit, Y_NUM, [1.5, 0.2, 0.8, 0.3, 0.7, 0.4]), cal_name  # SF-BL-9904 range
        assert raises(cls().fit, Y_NUM, [float("nan")] * 6), cal_name  # finiteness
        assert raises(cls().fit, [True, False, True, False, True, False], P_NUM), cal_name  # bool y via _as_y
        assert raises(cls().fit, Y_NUM, P_NUM + [0.5]), cal_name  # length mismatch (or string-free mismatch path)

ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
if __name__ == "__main__":
    for fn in ALL_TESTS:
        try: fn(); check(fn.__name__, True)
        except AssertionError as e: check(fn.__name__, False, str(e))
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
