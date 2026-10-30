"""
test_calibration_fit_calibrator_string.py — SF-BL-11426 verification.

Prereg: Guard raises ValueError naming y/p on any string or
object-dtype element at fit_calibrator entry for every registered
calibrator name, with float-panel fitted parameters bit-identical;
kill if fit_calibrator('platt', string panel) still returns
A_=17.7628 or 'beta' still returns a_=17.2495 instead of raising.

Fix: _reject_string_dispatcher_input screens RAW y and p via
_has_string + _has_bare_object + _has_string_in_pandas_panel +
_has_string_in_dataframe_panel as the first statements of
fit_calibrator, before the registry lookup and any float conversion.
Pre-fix, plain string lists raised only via the downstream
SF-BL-11395 fit() screens, and pandas DataFrame string panels
returned the kill values outright (A_=17.762763687307135 /
a_=17.249465734206364) in both live copies.
Lawful bit-identical: SHA-256 over lawful fitted params +
predictions via fit_calibrator (see repro.py in SF-BL-11426 work
dir) is 6f59f410856738aa68031adba5e20588fbbb4e3f5c48d0d872e0a2d0a54e14fa
both pre- and post-fix, in both live copies.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from superforecaster.calibration import CALIBRATORS, fit_calibrator

PASS = []
NAMES = sorted(CALIBRATORS)
Y_NUM = [1, 0, 1, 0, 1, 0]
P_NUM = [0.9, 0.2, 0.8, 0.3, 0.7, 0.4]
P_STR = ['0.9', '0.2', '0.8', '0.3', '0.7', '0.4']
Y_STR = ['1', '0', '1', '0', '1', '0']

def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))

def _named_error(name_arg, fn, *args):
    try:
        fn(*args)
    except ValueError as e:
        msg = str(e)
        assert "string" in msg and f"in {name_arg}" in msg \
            and "SF-BL-11426" in msg, msg
        return e
    return None

def test_kill_criteria():
    # fit_calibrator('platt', string panel) must NOT return A_=17.7628
    assert _named_error("p", fit_calibrator, "platt", Y_NUM, P_STR) is not None
    # fit_calibrator('beta', string panel) must NOT return a_=17.2495
    assert _named_error("p", fit_calibrator, "beta", Y_NUM, P_STR) is not None
    for nm in NAMES:
        assert _named_error("p", fit_calibrator, nm, Y_NUM, P_STR) is not None, nm
        assert _named_error("y", fit_calibrator, nm, Y_STR, P_NUM) is not None, nm
        assert _named_error("y", fit_calibrator, nm, Y_STR, P_STR) is not None, nm

STRING_P = [
    P_STR, tuple(P_STR), '0.9', b'0.9', np.str_('0.9'),
    [0.9, '0.2', 0.8, 0.3, 0.7, 0.4], ['0.9', 0.2, 0.8, 0.3, 0.7, 0.4],
    [[s] for s in P_STR],
    np.array(P_STR), np.array([s.encode() for s in P_STR]),
    np.array(P_STR, dtype=object),
    np.array([0.9, '0.2', 0.8, 0.3, 0.7, 0.4], dtype=object),
    ['abc'] * 6, [''] * 6,
    [object(), 0.2, 0.8, 0.3, 0.7, 0.4],
    np.array([object(), 0.2, 0.8, 0.3, 0.7, 0.4], dtype=object),
]
STRING_Y = [
    Y_STR, tuple(Y_STR), '1', np.array(Y_STR), np.array(Y_STR, dtype=object),
    [1, '0', 1, 0, 1, 0], ['1', 0, 1, 0, 1, 0],
    np.array([s.encode() for s in Y_STR]),
    [object(), 0, 1, 0, 1, 0],
]

def test_string_forms_raise_for_every_name():
    for nm in NAMES:
        for v in STRING_P:
            assert _named_error("p", fit_calibrator, nm, Y_NUM, v) is not None, \
                (nm, repr(v))
        for v in STRING_Y:
            assert _named_error("y", fit_calibrator, nm, v, P_NUM) is not None, \
                (nm, repr(v))

def test_pandas_panels_raise_for_every_name():
    try:
        import pandas as pd
    except Exception:
        return  # pandas optional; core forms covered elsewhere
    str_p = [pd.Series(P_STR), pd.Series(P_STR, dtype="string"),
             pd.Index(P_STR), pd.Categorical(P_STR),
             pd.DataFrame({"v": P_STR}),
             pd.DataFrame({"v": pd.Series(P_STR, dtype="string")}),
             pd.DataFrame({"v": pd.Series([P_STR[0]] + P_NUM[1:],
                                          dtype=object)})]
    str_y = [pd.Series(Y_STR), pd.Series(Y_STR, dtype="string"),
             pd.Index(Y_STR), pd.Categorical(Y_STR),
             pd.DataFrame({"v": Y_STR}),
             pd.DataFrame({"v": pd.Series(Y_STR, dtype="string")})]
    for nm in NAMES:
        for v in str_p:
            assert _named_error("p", fit_calibrator, nm, Y_NUM, v) is not None, \
                (nm, type(v))
        for v in str_y:
            assert _named_error("y", fit_calibrator, nm, v, P_NUM) is not None, \
                (nm, type(v))
        # numeric pandas panels (Series AND DataFrame) stay lawful
        m = fit_calibrator(nm, pd.Series(Y_NUM), pd.Series(P_NUM))
        assert m is not None
        m = fit_calibrator(nm, pd.DataFrame({"v": Y_NUM}),
                           pd.DataFrame({"v": P_NUM}))
        assert m is not None

def test_entry_point_precedence():
    # The dispatcher screen runs at entry: a string panel raises the
    # SF-BL-11426 error even when the calibrator name is unknown
    # (before the registry lookup).
    assert _named_error("p", fit_calibrator, "nope", Y_NUM, P_STR) is not None
    # Unknown name with lawful panels keeps the pre-existing error.
    try:
        fit_calibrator("nope", Y_NUM, P_NUM)
        raise AssertionError("unknown name did not raise")
    except ValueError as e:
        assert "unknown calibrator" in str(e), str(e)
    # Constant-string p would be intercept-only if coerced; must raise
    for nm in NAMES:
        assert _named_error("p", fit_calibrator, nm, Y_NUM,
                            ['0.7'] * 6) is not None, nm

def test_lawful_bit_identical():
    platt = fit_calibrator("platt", Y_NUM, P_NUM)
    assert platt.A_ == 17.762763687307135 and platt.B_ == -3.7957007060548054
    beta = fit_calibrator("beta", Y_NUM, P_NUM)
    assert beta.a_ == 17.249465734206364 and beta.b_ == -16.39197958793144
    assert beta.c_ == -3.020478689885648
    temp = fit_calibrator("temperature", Y_NUM, P_NUM)
    assert temp.T_ == 0.05
    iso = fit_calibrator("isotonic", Y_NUM, P_NUM)
    assert np.array_equal(iso.x_, np.array([0.2, 0.3, 0.4, 0.7, 0.8, 0.9]))
    assert np.array_equal(iso.y_, np.array([0., 0., 0., 1., 1., 1.]))
    # Lawful carriers give identical Platt params through the dispatcher
    for y, p in [(tuple(Y_NUM), tuple(P_NUM)),
                 (np.array(Y_NUM), np.array(P_NUM)),
                 (np.array(Y_NUM, dtype=object),
                  np.array(P_NUM, dtype=object)),
                 ([1, 0, 1, 0, 1, 0], [0.9, 0.2, 0.8, 0.3, 0.7, 0.4])]:
        m = fit_calibrator("platt", y, p)
        assert m.A_ == platt.A_ and m.B_ == platt.B_, (y, p)
    # Int endpoints lawful
    m = fit_calibrator("platt", [1, 0, 1, 0], [1, 0, 1, 0])
    assert m.A_ == 1.1091951154466313

def test_preexisting_screens_unchanged():
    def raises(*a):
        try: fit_calibrator(*a)
        except ValueError: return True
        return False
    for nm in NAMES:
        assert raises(nm, Y_NUM, [1.5, 0.2, 0.8, 0.3, 0.7, 0.4]), nm  # SF-BL-9904
        assert raises(nm, Y_NUM, [float("nan")] * 6), nm  # finiteness
        assert raises(nm, [True, False, True, False, True, False],
                      P_NUM), nm  # bool y downstream screens
        assert raises(nm, Y_NUM, P_NUM + [0.5]), nm  # length mismatch

ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
if __name__ == "__main__":
    for fn in ALL_TESTS:
        try: fn(); check(fn.__name__, True)
        except AssertionError as e: check(fn.__name__, False, str(e))
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
