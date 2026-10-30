"""
test_backtest_bool_probability.py — SF-BL-10983 verification.

Pre-registered done/kill criteria (TASK.md):
  "Pass if backtest brier_score, log_score and
   expected_calibration_error raise a named ValueError on any bool
   probability while lawful 0/1 panels stay bit-identical; kill if
   [True] still returns a Brier, log score or ECE below 1e-9 from
   any of the three."

Bug: _check and the per-scorer panel validators convert p via
np.asarray(..., dtype=float) before scoring, mapping True -> 1.0 /
False -> 0.0, and the EPS clip then grades a leaked boolean flag as
a perfect endpoint forecast:
  brier_score([1], [True]) == 9.98402083170343e-31
  log_score([1], [True])   == 9.992007221626415e-16
  expected_calibration_error([1], [True]) == 9.992007221626409e-16
(repro_before_*.txt in the task work dir.)

Fix (backtest.py, both live copies): _reject_bool_probabilities /
_has_bool_probability screen the RAW p at the top of brier_score,
log_score and expected_calibration_error, before any float
conversion, raising
"<scorer>: forecasts p must not contain boolean values; got boolean
value(s) in p (a bool is not a valid probability) (SF-BL-10983)".
Lawful panels are pure-validation bit-identical: the pre/post
SHA-256 over lawful panels + legacy error fingerprints is
b1e818aa51c60db6a0921ac1d03ae06f612e460f636bfead86010c4401e6a65f
both ways (lawful_baseline.py in the task work dir).

Run:  BT_PATH=<path-to-backtest.py> python3 test_backtest_bool_probability.py
      (default: the canonical package module)
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_bt_path = os.environ.get("BT_PATH")
if _bt_path:
    _spec = importlib.util.spec_from_file_location("bt_under_test", _bt_path)
    bt = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = bt
    _spec.loader.exec_module(bt)
else:
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from superforecaster import backtest as bt  # noqa: E402

SCORERS = [
    ("brier_score", bt.brier_score),
    ("log_score", bt.log_score),
    ("expected_calibration_error", bt.expected_calibration_error),
]
PASS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))


def bool_error(scorer_name, fn, *args, **kw):
    """The named SF-BL-10983 ValueError, or None; any other
    ValueError message is a failure, not a pass."""
    try:
        fn(*args, **kw)
    except ValueError as e:
        msg = str(e)
        assert "boolean" in msg and "SF-BL-10983" in msg \
            and scorer_name in msg, f"unnamed ValueError: {msg}"
        return e
    return None


# ---- A. Titled repros raise in all three scorers ---------------------------

def test_titled_repros():
    for name, fn in SCORERS:
        assert bool_error(name, fn, [1], [True]) is not None, name
        assert bool_error(name, fn, [0], [False]) is not None, name
        err = bool_error(name, fn, [1], [True])
        assert "in p" in str(err), str(err)


# ---- B. Every bool container form ------------------------------------------

BOOL_PS = [
    ([True], [1]), ([False], [0]), ([0.9, True], [1, 0]),
    ([False, 0.1], [0, 1]), ((0.5, True), [1, 0]),
    ([[True, False]], [[1, 0]]), ([[0.5], [True]], [[1], [0]]),
    (True, [1]), (np.bool_(True), [1]), (np.bool_(False), [0]),
    (np.array([True, False]), [1, 0]), (np.array([True]), [1]),
    (np.array(True), [1]),
    (np.array([0.9, True], dtype=object), [1, 0]),
    (np.array([[True, False]]), [[1, 0]]),
    (np.array([np.array([True])], dtype=object), [1]),
    (pd.Series([True, False]), [1, 0]),
    (pd.Series([0.9, True], dtype=object), [1, 0]),
    (pd.Index([True, False]), [1, 0]),
]


def test_bool_probability_forms():
    for name, fn in SCORERS:
        for p, y in BOOL_PS:
            assert bool_error(name, fn, y, p) is not None, (name, repr(p))


# ---- C. Kill criterion: [True] returns no score < 1e-9 ----------------------

def test_kill_criterion():
    for name, fn in SCORERS:
        try:
            v = fn([1], [True])
        except ValueError:
            continue
        raise AssertionError(f"{name} returned {v!r} for [True] (kill)")


# ---- D. Lawful 0/1 panels bit-identical (values pinned pre-fix) -------------

def test_lawful_panels_bit_identical():
    assert bt.brier_score([1, 0], [1, 0]) == 9.992010415851716e-31
    assert bt.brier_score([1, 0], [1.0, 0.0]) == 9.992010415851716e-31
    assert bt.brier_score([1], [1.0]) == 9.98402083170343e-31
    assert bt.brier_score([1, 0], [0.9, 0.1]) == 0.009999999999999998
    assert bt.log_score([1, 0], [0.9, 0.1]) == 0.10536051565782628
    assert bt.expected_calibration_error([1, 0], [0.9, 0.1]) \
        == 0.09999999999999999
    assert bt.expected_calibration_error([1, 0], [1, 0]) \
        == 9.996003610813205e-16
    assert bt.brier_score(np.array([1, 0]), np.array([1, 0])) \
        == 9.992010415851716e-31
    rng = np.random.default_rng(10983)
    for n in (1, 2, 10, 57):
        y = (rng.random(n) > 0.5).astype(int)
        p = rng.random(n)
        assert bt.brier_score(y, p) == float(np.mean((p - y) ** 2))
        pc = np.clip(p, 1e-15, 1 - 1e-15)
        assert bt.log_score(y, p) == float(
            -np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))


# ---- E. Pre-existing errors unchanged ---------------------------------------

def test_preexisting_errors_unchanged():
    def raises(fn, *a, **k):
        try:
            fn(*a, **k)
        except ValueError:
            return True
        return False
    assert raises(bt.brier_score, [1], [1.5])            # SF-BL-10940
    assert raises(bt.brier_score, [1], [-0.5])           # SF-BL-10940
    assert raises(bt.log_score, [1], [1.5])              # SF-BL-10071
    assert raises(bt.log_score, [], [])                  # SF-BL-10071
    assert raises(bt.log_score, [1], [float("nan")])     # SF-BL-10071
    assert raises(bt.expected_calibration_error, [1], [0.5], 0)     # 9791
    assert raises(bt.expected_calibration_error, [1], [0.5], True)  # 9791
    assert raises(bt.brier_score, [2], [0.5])            # SF-BL-9853 y
    assert raises(bt.brier_score, [1, 0], [0.5])         # shape mismatch
    # legacy non-finite / empty behaviour preserved
    assert bt.brier_score([1, 0], [float("nan"), 0.5]) == 0.25
    assert np.isnan(bt.expected_calibration_error([], []))


# ---- F. Scope containment: siblings untouched, y scope per prereg -----------

def test_scope_containment():
    assert bt.auc_mann_whitney([1, 0, 1], [0.9, 0.1, 0.8]) == 1.0
    ev = bt.evaluate_forecasts([1, 0], [0.9, 0.1])
    assert ev["brier"] == 0.009999999999999998
    assert ev["log_score"] == 0.10536051565782628
    assert ev["ece"] == 0.09999999999999999
    # _check itself still clips (shared helper untouched)
    yc, pc = bt._check([1], [1.5])
    assert pc[0] == 1 - 1e-15
    # Prereg scope is bool *probabilities* only: a bool outcome keeps
    # its legacy behaviour (True coerces to the 1 outcome).
    assert bt.brier_score([True], [0.5]) == 0.25


ALL_TESTS = [v for k, v in sorted(globals().items())
             if k.startswith("test_")]

if __name__ == "__main__":
    print(f"module under test: {bt.__file__}")
    for fn in ALL_TESTS:
        try:
            fn()
            check(fn.__name__, True)
        except AssertionError as e:
            check(fn.__name__, False, str(e))
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
