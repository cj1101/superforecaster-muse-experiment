"""
test_calibration_calibrator_predict_bool.py — SF-BL-11051 verification.

Pre-registered done/kill criteria (TASK.md):
  "Guard raises ValueError naming the input on any bool predict()
   argument for all four calibrators, with float in-range predictions
   bit-identical; kill if any of Platt/Beta/Temperature/Isotonic
   predict(True) or predict(False) returns a float instead of raising."

Bug: all four predict() paths converted their input with
np.asarray(p, dtype=float) before validation, so True/False passed
the SF-BL-9719/9720/9751/9752 range screens as lawful 1/0 endpoints:
  PlattScaler.predict(True)       -> 0.9252 (task repro; seed-dependent)
  BetaCalibrator.predict(False)   -> 3.5e-23
  TemperatureScaler.predict(True) -> 0.9292
  IsotonicCalibrator.predict(True/False) -> 1.0 / 0.0
on a 40-row noisy fit.

Fix (superforecaster/calibration.py): _reject_bool_predict_input
screens the RAW p via _has_bool as the first statement of every
calibrator predict() (and transform alias) — before float conversion,
before Temperature's intercept-only early return, and before
Isotonic's unfitted RuntimeError — raising
"p must not contain boolean values; got boolean value(s) in p
(a bool is not a valid p value) (SF-BL-11051)".
Lawful predictions are bit-identical: SHA-256 over lawful outputs
(all four calibrators, deterministic 40-row fit + intercept-only
fits) is 964d14a823cbf7ea5f3c60c569a8ece7e85a7c8177ce7893de5815c996b83d97
both pre- and post-fix (repro.py / lawful_baseline_prefix.json in
the SF-BL-11051 backlog work dir).

Run:  python3 tests/test_calibration_calibrator_predict_bool.py
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from superforecaster.calibration import (  # noqa: E402
    BetaCalibrator, IsotonicCalibrator, PlattScaler, TemperatureScaler)

PASS = []
CALS = [("Platt", PlattScaler), ("Beta", BetaCalibrator),
        ("Temperature", TemperatureScaler), ("Isotonic", IsotonicCalibrator)]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))


def fit_all(seed=0):
    rng = np.random.default_rng(seed)
    p = rng.random(40)
    y = (rng.random(40) < p).astype(int)
    return [(name, cls().fit(y, p)) for name, cls in CALS]


def _bool_error(fn, *args, **kw):
    """ValueError naming p + 'boolean' + SF-BL-11051, or None."""
    try:
        fn(*args, **kw)
    except ValueError as e:
        msg = str(e)
        if "boolean" in msg and "in p" in msg and "SF-BL-11051" in msg:
            return e
        raise AssertionError(f"ValueError not the named bool error: {msg}")
    return None


# ---- 1. Titled / kill repros: predict(True/False) raises, never returns ---

def test_titled_repros_raise():
    for name, m in fit_all():
        for v in (True, False):
            assert _bool_error(m.predict, v) is not None, (name, v)
            assert _bool_error(m.transform, v) is not None, (name, v)


# ---- 2. Every bool container form ------------------------------------------

BOOL_INPUTS = [
    True, False, np.bool_(True), np.bool_(False),
    [True], [False], [0.5, True], [False, 0.1], (0.5, True),
    [0.2, [True]], np.array([True, False]), np.array([True]),
    np.array([0.9, True], dtype=object), np.array([[True, False]]),
    np.array([0.5, False], dtype=object),
]


def test_bool_forms_raise():
    for name, m in fit_all():
        for v in BOOL_INPUTS:
            assert _bool_error(m.predict, v) is not None, (name, repr(v))


# ---- 3. Intercept-only and unfitted calibrators also raise ------------------

def test_intercept_only_and_unfitted_raise():
    y_const = np.array([0, 1] * 20)
    p_const = np.full(40, 0.7)
    for name, cls in CALS:
        m = cls().fit(y_const, p_const)
        for v in (True, False):
            assert _bool_error(m.predict, v) is not None, (name, "intercept-only", v)
        fresh = cls()
        for v in (True, False):
            assert _bool_error(fresh.predict, v) is not None, (name, "unfitted", v)
    # Lawful input on an unfitted Isotonic still raises RuntimeError.
    try:
        IsotonicCalibrator().predict(0.5)
        raise SystemExit("unfitted Isotonic did not raise RuntimeError")
    except RuntimeError:
        pass
    # Step-mode Isotonic also raises on bool.
    rng = np.random.default_rng(0)
    p = rng.random(40); y = (rng.random(40) < p).astype(int)
    step = IsotonicCalibrator(interpolation="step").fit(y, p)
    assert _bool_error(step.predict, True) is not None
    assert _bool_error(step.predict, [0.5, False]) is not None


# ---- 4. Lawful in-range predictions bit-identical ---------------------------
# Exact values captured pre-fix (repro_prefix.log / repro.py lawful_outputs).

EXPECTED = {
    "Platt": 0.5338205671880949,
    "Beta": 0.6072379903667995,
    "Temperature": 0.5,
    "Isotonic": 0.5,
}


def test_lawful_bit_identical():
    for name, m in fit_all():
        assert float(np.asarray(m.predict(0.5))) == EXPECTED[name], name
        # float endpoints, int endpoints (ints are NOT bools), arrays
        assert float(np.asarray(m.predict(0.0))) == float(np.asarray(m.predict(0)))
        assert float(np.asarray(m.predict(1.0))) == float(np.asarray(m.predict(1)))
        arr = m.predict(np.array([0.0, 0.13, 0.5, 0.87, 1.0]))
        lst = m.predict([0.0, 0.13, 0.5, 0.87, 1.0])
        assert np.array_equal(np.asarray(arr), np.asarray(lst)), name
        ints = m.predict([0, 1])
        floats = m.predict([0.0, 1.0])
        assert np.array_equal(np.asarray(ints), np.asarray(floats)), name


# ---- 5. Pre-existing predict errors unchanged; no dead line touched ---------

def test_preexisting_errors_unchanged():
    def raises(fn, *a):
        try:
            fn(*a)
        except ValueError:
            return True
        return False
    for name, m in fit_all():
        assert raises(m.predict, 1.5), name          # range SF-BL-9719/20/51/52
        assert raises(m.predict, -0.5), name         # range
        assert raises(m.predict, float("nan")), name  # finiteness
        assert raises(m.predict, float("inf")), name  # finiteness
        assert raises(m.predict, [0.5, 1.5]), name   # range in array


ALL_TESTS = [v for k, v in sorted(globals().items())
             if k.startswith("test_")]

if __name__ == "__main__":
    for fn in ALL_TESTS:
        try:
            fn()
            check(fn.__name__, True)
        except AssertionError as e:
            check(fn.__name__, False, str(e))
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
