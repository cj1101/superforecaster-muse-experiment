"""
test_calibration_calibrator_predict_string.py — SF-BL-11463 verification.

Prereg: all four calibrators' predict raise ValueError naming the input
on any string-dtype panel, with numeric in-range predictions
bit-identical; kill if TemperatureScaler.predict(['0.55']) still
returns 0.9822 instead of raising.

Fix: _reject_string_predict_input screens RAW p via _has_string as an
early statement of every predict() (and transform alias), before float
conversion, before Temperature intercept-only return, and before
Isotonic's unfitted RuntimeError.
Lawful bit-identical: SHA-256 over lawful outputs (deterministic
40-row seed-0 fit + intercept-only fits, see repro.py in SF-BL-11463
work dir) is 77bc6943326859e0129a31e59f5b79e3c4dedbc8d26e03fbf6bb8ecfcdee4c0f
both pre- and post-fix.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from superforecaster.calibration import BetaCalibrator, IsotonicCalibrator, PlattScaler, TemperatureScaler

PASS = []
CALS = [("Platt", PlattScaler), ("Beta", BetaCalibrator), ("Temperature", TemperatureScaler), ("Isotonic", IsotonicCalibrator)]

def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))

def fit_all(seed=0):
    rng = np.random.default_rng(seed)
    p = rng.random(40)
    y = (rng.random(40) < p).astype(int)
    return [(name, cls().fit(y, p)) for name, cls in CALS]

def _string_error(fn, *args, **kw):
    try:
        fn(*args, **kw)
    except ValueError as e:
        msg = str(e)
        if "string" in msg and "in p" in msg and "SF-BL-11463" in msg:
            return e
        raise AssertionError(f"ValueError not the named string error: {msg}")
    return None

def test_titled_and_kill_repros_raise():
    for name, m in fit_all():
        for v in (['0.55'], '0.55', ['0.5'], '0.5'):
            assert _string_error(m.predict, v) is not None, (name, v)
            assert _string_error(m.transform, v) is not None, (name, v)
    # Explicit kill criterion: Temperature must raise, not return 0.9822 (or any float)
    _, temp = fit_all()[2]
    assert _string_error(temp.predict, ['0.55']) is not None

STRING_INPUTS = [
    '0.55', b'0.55', np.str_('0.55'), ['0.55'], ['0.5'], [0.5, '0.5'], ['0.5', 0.5],
    ('0.55',), [0.2, ['0.3']], np.array(['0.55']), np.array(['0.1', '0.9']),
    np.array([b'0.55']), np.array(['0.5', 0.5], dtype=object),
    np.array([0.5, '0.5'], dtype=object), np.array([['0.55']]),
]

def test_string_forms_raise():
    for name, m in fit_all():
        for v in STRING_INPUTS:
            assert _string_error(m.predict, v) is not None, (name, repr(v))

def test_intercept_only_unfitted_and_step_raise():
    y_const = np.array([0, 1] * 20); p_const = np.full(40, 0.7)
    for name, cls in CALS:
        m = cls().fit(y_const, p_const)
        assert _string_error(m.predict, ['0.55']) is not None, (name, "intercept-only")
        fresh = cls()
        assert _string_error(fresh.predict, ['0.55']) is not None, (name, "unfitted")
    try:
        IsotonicCalibrator().predict(0.5); raise SystemExit("unfitted Isotonic did not raise RuntimeError")
    except RuntimeError:
        pass
    rng = np.random.default_rng(0); p = rng.random(40); y = (rng.random(40) < p).astype(int)
    step = IsotonicCalibrator(interpolation="step").fit(y, p)
    assert _string_error(step.predict, ['0.55']) is not None
    assert _string_error(step.predict, [0.5, '0.5']) is not None

EXPECTED = {"Platt": 0.5338205671880949, "Beta": 0.6072379903667995, "Temperature": 0.5, "Isotonic": 0.5}

def test_lawful_bit_identical():
    for name, m in fit_all():
        assert float(np.asarray(m.predict(0.5))) == EXPECTED[name], name
        assert float(np.asarray(m.predict([0.55])).ravel()[0]) == float(np.asarray(m.predict(0.55)).ravel()[0])
        assert float(np.asarray(m.predict(0.0))) == float(np.asarray(m.predict(0)))
        assert float(np.asarray(m.predict(1.0))) == float(np.asarray(m.predict(1)))
        arr = m.predict(np.array([0.0, 0.13, 0.5, 0.87, 1.0])); lst = m.predict([0.0, 0.13, 0.5, 0.87, 1.0])
        assert np.array_equal(np.asarray(arr), np.asarray(lst)), name
        assert np.array_equal(np.asarray(m.predict([0, 1])), np.asarray(m.predict([0.0, 1.0]))), name

def test_preexisting_errors_unchanged():
    def raises(fn, *a):
        try: fn(*a)
        except ValueError: return True
        return False
    for name, m in fit_all():
        assert raises(m.predict, 1.5), name
        assert raises(m.predict, -0.5), name
        assert raises(m.predict, float("nan")), name
        assert raises(m.predict, float("inf")), name
        assert raises(m.predict, [0.5, 1.5]), name
        assert raises(m.predict, True), name  # SF-BL-11051 bool screen unchanged
        assert raises(m.predict, [True]), name

ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
if __name__ == "__main__":
    for fn in ALL_TESTS:
        try: fn(); check(fn.__name__, True)
        except AssertionError as e: check(fn.__name__, False, str(e))
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
