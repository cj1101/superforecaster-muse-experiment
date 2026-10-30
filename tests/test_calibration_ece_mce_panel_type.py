"""
test_calibration_ece_mce_panel_type.py — SF-BL-11424 verification.

Prereg: Guard raises ValueError naming y/p on any string,
object-dtype or bool element in expected_calibration_error and
maximum_calibration_error, with float panels bit-identical; kill if
string forecasts still return ECE/MCE 0.1 or the bool panel still
returns 0.0 instead of raising.

Fix: _reject_binned_panel screens RAW y and p via _has_bool +
_has_bool_in_pandas_panel (bool dimension) and _has_string +
_has_string_in_pandas_panel + _has_bare_object +
_has_bare_object_in_pandas_panel (string / object-dtype
dimension), explicitly in expected_calibration_error and
maximum_calibration_error and again in calibration_bins, before
float conversion. Error messages carry the SF-BL-10938 /
SF-BL-11052 tags alongside SF-BL-11424.
Lawful bit-identical: SHA-256 over lawful ECE/MCE outputs (see
repro.py in SF-BL-11424 work dir) is
5a64bc81df7954d8eba9406e69f71bfc3081412e4db23c14c609c90936f3b625
both pre- and post-fix, in both live copies of calibration.py.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from superforecaster.calibration import (
    calibration_bins, ece, expected_calibration_error, maximum_calibration_error, mce,
)

PASS = []
Y_NUM = [1, 0]
P_NUM = [0.9, 0.1]
P_STR = ['0.9', '0.1']
Y_STR = ['1', '0']
FNS = [("ECE", expected_calibration_error), ("MCE", maximum_calibration_error),
       ("ece-alias", ece), ("mce-alias", mce)]

def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))

def _typed_error(fn, name, kind, *args, **kw):
    try:
        fn(*args, **kw)
    except ValueError as e:
        msg = str(e)
        assert f"in {name}" in msg and "SF-BL-11424" in msg, msg
        assert (kind in msg), msg
        return e
    return None

def test_kill_criteria():
    # String forecasts must not return ECE/MCE 0.1; bool panel must not return 0.0
    for fn_name, fn in FNS:
        assert _typed_error(fn, "p", "string", Y_NUM, P_STR) is not None, fn_name
        assert _typed_error(fn, "p", "boolean", Y_NUM, [True, False]) is not None, fn_name

STRING_P = [
    P_STR, tuple(P_STR), '0.9', b'0.9', np.str_('0.9'),
    [0.9, '0.1'], ['0.9', 0.1], [0.9, ['0.1']],
    np.array(P_STR), np.array(P_STR, dtype=object),
    np.array([0.9, '0.1'], dtype=object),
    np.array([b'0.9', b'0.1']), ['abc', 'def'], ['', ''],
    [object(), 0.1], np.array([object(), 0.1], dtype=object),
]
STRING_Y = [
    Y_STR, tuple(Y_STR), '1', np.array(Y_STR), np.array(Y_STR, dtype=object),
    [1, '0'], ['1', 0], np.array([b'1', b'0']), [object(), 0],
]
BOOL_P = [
    [True, False], [True], (True, False), True, np.bool_(True),
    np.array([True, False]), np.array([0.9, True], dtype=object),
    [0.9, True],
]
BOOL_Y = [
    [True, False], [False, True], (True, False), np.array([True, False]),
    np.array([1, True], dtype=object),
]

def test_all_forms_raise():
    for fn_name, fn in FNS:
        for v in STRING_P:
            assert _typed_error(fn, "p", "string", Y_NUM, v) is not None, (fn_name, repr(v))
        for v in STRING_Y:
            assert _typed_error(fn, "y", "string", v, P_NUM) is not None, (fn_name, repr(v))
        for v in BOOL_P:
            n = len(np.asarray(v, dtype=object).ravel()) if not isinstance(v, (bool, np.bool_)) else 1
            assert _typed_error(fn, "p", "boolean", [1] * n, v) is not None, (fn_name, repr(v))
        for v in BOOL_Y:
            assert _typed_error(fn, "y", "boolean", v, P_NUM) is not None, (fn_name, repr(v))
    # calibration_bins funnel carries the same screen
    assert _typed_error(calibration_bins, "p", "string", Y_NUM, P_STR) is not None
    assert _typed_error(calibration_bins, "p", "boolean", Y_NUM, [True, False]) is not None
    assert _typed_error(calibration_bins, "y", "string", Y_STR, P_NUM) is not None

def test_pandas_panels_raise():
    try:
        import pandas as pd
    except Exception:
        return
    for fn_name, fn in FNS:
        assert _typed_error(fn, "p", "string", Y_NUM, pd.Series(P_STR)) is not None, fn_name
        assert _typed_error(fn, "y", "string", pd.Series(Y_STR), P_NUM) is not None, fn_name
        assert _typed_error(fn, "p", "string", Y_NUM, pd.Index(P_STR)) is not None, fn_name
        assert _typed_error(fn, "p", "string", Y_NUM, pd.Series(P_STR, dtype="category")) is not None, fn_name
        assert _typed_error(fn, "p", "boolean", Y_NUM, pd.Series([True, False])) is not None, fn_name
        assert _typed_error(fn, "y", "boolean", pd.Series([True, False]), P_NUM) is not None, fn_name
        assert _typed_error(fn, "p", "boolean", Y_NUM, pd.Series([True, False], dtype="category")) is not None, fn_name
        # numeric pandas panels stay lawful
        assert fn(pd.Series(Y_NUM), pd.Series(P_NUM)) == fn(Y_NUM, P_NUM), fn_name

def test_lawful_bit_identical():
    for fn_name, fn in (("ECE", expected_calibration_error), ("MCE", maximum_calibration_error)):
        assert fn(Y_NUM, P_NUM) == (0.09999999999999999 if fn_name == "ECE" else 0.1)
    assert expected_calibration_error([1, 0, 1, 0], [1, 0, 1, 0]) == 0.0
    assert maximum_calibration_error([1, 0, 1, 0], [1, 0, 1, 0]) == 0.0
    assert expected_calibration_error([], []) == 0.0
    assert maximum_calibration_error([], []) == 0.0
    rng = np.random.default_rng(0)
    big_y = (rng.random(40) > 0.5).astype(int).tolist(); big_p = rng.random(40).tolist()
    assert expected_calibration_error(big_y, big_p) == 0.20940497908131356
    assert maximum_calibration_error(big_y, big_p) == 0.42866056873222236
    assert expected_calibration_error(big_y, big_p, strategy="quantile") == 0.1900813297648726
    assert expected_calibration_error(big_y, big_p, n_bins=3) == 0.14089891012987985
    # Lawful carriers identical to the plain-list control
    for y, p in [(tuple(Y_NUM), tuple(P_NUM)),
                 (np.array(Y_NUM), np.array(P_NUM)),
                 (np.array(Y_NUM, dtype=float), np.array(P_NUM)),
                 (np.array(Y_NUM, dtype=object), np.array(P_NUM, dtype=object))]:
        assert expected_calibration_error(y, p) == 0.09999999999999999, (y, p)
        assert maximum_calibration_error(y, p) == 0.1, (y, p)

def test_preexisting_screens_unchanged():
    def raises(fn, *a, **kw):
        try: fn(*a, **kw)
        except ValueError: return True
        return False
    for fn_name, fn in FNS:
        assert raises(fn, Y_NUM, [1.5, 0.1]), fn_name          # SF-BL-8778 range
        assert raises(fn, Y_NUM, [float("nan"), 0.1]), fn_name  # finiteness
        assert raises(fn, Y_NUM, [0.5]), fn_name                # SF-BL-9822 length
        assert raises(fn, Y_NUM, P_NUM, n_bins=0), fn_name      # SF-BL-9722 n_bins
        assert raises(fn, [1, 2], P_NUM), fn_name               # binary y

ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
if __name__ == "__main__":
    for fn in ALL_TESTS:
        try: fn(); check(fn.__name__, True)
        except AssertionError as e: check(fn.__name__, False, str(e))
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
