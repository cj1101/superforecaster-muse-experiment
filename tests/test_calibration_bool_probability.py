"""
test_calibration_bool_probability.py — SF-BL-10938 verification.

Pre-registered done/kill criteria (TASK.md):
  "Pass if brier_score, log_score and evaluate raise a named
   ValueError on any bool probability (and bool outcome) while
   lawful 0/1 panels stay bit-identical; kill if [True] still
   scores 0.0 Brier or a perfect evaluate scorecard."

Bug: _validate_p_in_range and _as_y converted bools to 1.0/0.0 via
np.asarray(..., dtype=float) before checking, so a boolean flag in a
probability field scored as a perfect endpoint forecast:
  brier_score([1], [True]) == 0.0
  log_score([1], [True])   == ~1e-15 (9.992007221626415e-16)
  evaluate([1], [True])    == perfect scorecard (Brier 0.0, ECE 0.0)
(repro_prefix.py / repro_prefix.log in the task work dir.)

Fix (superforecaster/calibration.py): _reject_bools / _has_bool run
on the RAW inputs before any float conversion — at the top of
brier_score, log_score, calibration_bins, murphy_decomposition and
evaluate for p, and inside _as_y for y — raising
"p|y must not contain boolean values; got boolean value(s) in p|y
(a bool is not a valid p|y value) (SF-BL-10938)".
_validate_p_in_range also screens its argument so a raw bool-dtype
panel can never pass it. Lawful panels are pure-validation
bit-identical: the pre/post SHA-256 over 7 lawful panels (brier,
log, evaluate, murphy) is abc8573748774ca899cb8ff61a43235bee50a3
both ways (lawful_baseline_prefix.json in the task work dir).

Run:  python3 tests/test_calibration_bool_probability.py
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from superforecaster.calibration import (  # noqa: E402
    brier_score, calibration_bins, ece, evaluate, log_score, mce,
    murphy_decomposition)

PASS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))


def _bool_error(fn, *args, **kw):
    """ValueError naming the argument and 'boolean', or None."""
    try:
        fn(*args, **kw)
    except ValueError as e:
        msg = str(e)
        if "boolean" in msg and "SF-BL-10938" in msg:
            return e
        raise AssertionError(f"ValueError not the named bool error: {msg}")
    return None


# ---- 1. Titled repros: bool probability raises in all headline scorers ---

def test_titled_repros_raise():
    for fn, kw in ((brier_score, {}), (log_score, {}), (evaluate, {})):
        assert _bool_error(fn, [1], [True], **kw) is not None, fn.__name__
        assert _bool_error(fn, [0], [False], **kw) is not None, fn.__name__
        err = _bool_error(fn, [1], [True], **kw)
        assert "in p" in str(err), str(err)


def test_bool_outcome_raises():
    for fn in (brier_score, log_score, evaluate):
        assert _bool_error(fn, [True], [1.0]) is not None, fn.__name__
        assert _bool_error(fn, [False], [0.0]) is not None, fn.__name__
        err = _bool_error(fn, [True], [0.5])
        assert "in y" in str(err), str(err)


# ---- 2. Every bool container form ------------------------------------------

BOOL_PS = [
    [True], [False], [0.9, True], [False, 0.1], (0.5, True), True,
    np.bool_(True), np.array([True, False]), np.array([True]),
    np.array([0.9, True], dtype=object),
    np.array([[True, False]]),
]
BOOL_YS = [
    [True], [False], [1, True], (False, 0), np.bool_(False),
    np.array([True, False]), np.array([0, True], dtype=object),
]


def test_bool_probability_forms():
    for p in BOOL_PS:
        y = [1] * len(np.asarray(p, dtype=object).ravel()) \
            if not isinstance(p, (bool, np.bool_)) else [1]
        if isinstance(p, np.ndarray) and p.ndim == 2:
            y = [1, 0]
        assert _bool_error(brier_score, y, p) is not None, repr(p)
        assert _bool_error(log_score, y, p) is not None, repr(p)
        assert _bool_error(evaluate, y, p) is not None, repr(p)


def test_bool_outcome_forms():
    for y in BOOL_YS:
        n = len(np.asarray(y, dtype=object).ravel()) \
            if not isinstance(y, (bool, np.bool_)) else 1
        p = [0.5] * n
        assert _bool_error(brier_score, y, p) is not None, repr(y)
        assert _bool_error(log_score, y, p) is not None, repr(y)
        assert _bool_error(evaluate, y, p) is not None, repr(y)


# ---- 3. The evaluate funnel inherits the guard ------------------------------

def test_funnel_functions_raise():
    for fn in (calibration_bins, ece, mce, murphy_decomposition):
        assert _bool_error(fn, [1], [True]) is not None, fn.__name__
        assert _bool_error(fn, [True], [0.5]) is not None, fn.__name__


# ---- 4. Lawful 0/1 panels stay bit-identical --------------------------------

def test_lawful_panels_bit_identical():
    # Exact values measured pre-fix (repro_prefix.log / baseline json).
    assert brier_score([1, 0], [1, 0]) == 0.0
    assert brier_score([1, 0], [1.0, 0.0]) == 0.0
    assert log_score([1, 0], [0.9, 0.1]) == 0.10536051565782628
    assert brier_score([1, 0], [0.9, 0.1]) == 0.009999999999999998
    ev = evaluate([1, 0], [0.9, 0.1])
    assert ev == {"brier": 0.009999999999999998,
                  "log_score": 0.10536051565782628,
                  "ece": 0.09999999999999999, "mce": 0.1,
                  "reliability": 0.009999999999999998,
                  "resolution": 0.25, "uncertainty": 0.25}, ev
    # int endpoints, float endpoints, numpy int panels, mixed panels
    assert brier_score(np.array([1, 0]), np.array([1, 0])) == 0.0
    assert evaluate([0, 1], [0, 1])["brier"] == 0.0
    rng = np.random.default_rng(10938)
    for n in (1, 2, 10, 57):
        y = (rng.random(n) > 0.5).astype(int)
        p = rng.random(n)
        ref = float(np.mean((p - y) ** 2))
        assert brier_score(y, p) == ref
        pc = np.clip(p, 1e-15, 1 - 1e-15)
        ref_log = float(-np.mean(y * np.log(pc)
                                 + (1 - y) * np.log(1 - pc)))
        assert log_score(y, p) == ref_log


# ---- 5. Pre-existing errors unchanged; no dead line touched -----------------

def test_preexisting_errors_unchanged():
    def raises(fn, *a, **k):
        try:
            fn(*a, **k)
        except ValueError:
            return True
        return False
    assert raises(brier_score, [1], [1.5])          # range (SF-BL-9484)
    assert raises(log_score, [1], [float("nan")])   # finiteness
    assert raises(log_score, [1], [0.5], eps=0.9)   # eps (SF-BL-9659)
    assert raises(evaluate, [1], [0.5], n_bins=True)  # n_bins bool (9722)
    assert raises(brier_score, [2], [0.5])          # non-binary y
    assert raises(brier_score, [1, 0], [0.5])       # length mismatch
    assert raises(evaluate, [], [])                 # empty (SF-BL-10742)


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
