"""SF-BL-9513: calibration.log_odds_pool extremization guard.

Pass iff non-finite or non-positive extremization raises a documented
ValueError and finite-a pooled outputs on a [0.02, 0.98] grid are
bit-identical (to BOTH the committed pristine module and the
pre-9513 working-tree module). Kill iff a=inf returns exactly 0.0/1.0
or a=nan returns NaN with no error.

Standalone: python3 test_calibration_log_odds_pool_extremization.py
Also pytest-collectable.
"""
import importlib.util
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.expanduser("~/workspace/superforecaster/superforecaster/calibration.py")
PRISTINE = os.path.join(HERE, "pristine", "calibration.py")
PRE9513 = "/tmp/cal_before9513.py"
if not os.path.exists(PRE9513):
    PRE9513 = os.path.join(HERE, "pre9513_calibration.py")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


live = load(REPO, "cal_live")
# Reference modules live in the work dir; when this file is copied into
# the repo's tests/ they are absent and the bit-identity legs skip
# (the work-dir run is the evidence of record).
pristine = load(PRISTINE, "cal_pristine") if os.path.exists(PRISTINE) else None
pre9513 = load(PRE9513, "cal_pre9513") if os.path.exists(PRE9513) else None

PASS = FAIL = 0


def check(name, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        print(f"PASS {name}")
    except AssertionError as e:
        FAIL += 1
        print(f"FAIL {name}: {e}")


def bits(arr):
    return [struct.pack("<d", float(x)) for x in np.asarray(arr, dtype=float).ravel()]


PANELS = [
    np.array([[0.7, 0.3]]),
    np.array([[0.7]]),
    np.array([[0.8, 0.8]]),
    np.array([[0.6, 0.7]]),
    np.array([[0.1, 0.5, 0.9]]),
    np.array([[0.02, 0.98]]),
]
WEIGHTS = [None, None, None, [2.0, 1.0], [3.0, 1.0, 0.5], None]

REJECT = [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, -1e-9,
          -0.5, -1.0, -2.0, -100.0, np.float64("nan"), np.float64("inf"),
          "1.5", "nan", None, True, False, [1.0], (1.0,), np.array([1.0, 2.0])]


def must_raise(a):
    for P, w in zip(PANELS, WEIGHTS):
        try:
            v = live.log_odds_pool(P, weights=w, extremization=a)
        except ValueError as e:
            assert type(e) is ValueError, f"wrong type {type(e)}"
            msg = str(e)
            assert "extremization" in msg, f"arg not named: {msg}"
            assert "finite" in msg, f"finiteness not documented in msg: {msg}"
            assert "SF-BL-9513" in msg, f"task not named: {msg}"
            continue
        raise AssertionError(f"a={a!r} probs={P.tolist()} RETURNED {v}")


def test_task_repros_raise():
    must_raise(float("nan"))
    must_raise(float("inf"))


def test_reject_grid():
    for a in REJECT:
        must_raise(a)


def test_kill_criterion():
    # a=inf must not return exactly 0.0 or 1.0; a=nan must not return NaN.
    for P in PANELS:
        for a in (float("inf"), float("nan")):
            try:
                v = live.log_odds_pool(P, extremization=a)
            except ValueError:
                continue
            raise AssertionError(f"KILL: a={a} returned {v} on {P.tolist()}")


def test_alias_also_guarded():
    assert live.extremized_pool is live.log_odds_pool
    try:
        live.extremized_pool([[0.7, 0.3]], extremization=float("inf"))
    except ValueError:
        return
    raise AssertionError("extremized_pool alias did not raise on a=inf")


def test_finite_grid_bit_identical():
    grid = np.round(np.arange(0.02, 0.981, 0.02), 10)
    factors = [0.05, 0.3, 0.5, 1.0, 1.25, 2.0, 4.0, 10.0, 1, 2]
    panels = []
    for p in grid:
        panels += [np.array([[p, 1.0 - p]]), np.array([[p]]),
                   np.array([[p, 0.5, 1.0 - p]])]
    for a in factors:
        for P in panels:
            got = live.log_odds_pool(P, extremization=a)
            if pristine is not None:
                assert bits(got) == bits(pristine.log_odds_pool(P, extremization=a)), \
                    (a, P, got)
            if pre9513 is not None:
                assert bits(got) == bits(pre9513.log_odds_pool(P, extremization=a)), \
                    (a, P, got)
            # finite (expit may saturate to exactly 0/1 at a=10 extremes,
            # identically in pristine — bit-identity above is the bar)
            assert np.all(np.isfinite(got))


def test_known_values():
    assert bits(live.log_odds_pool([[0.8, 0.8]], extremization=1.0)) == bits([0.8])
    assert bits(live.log_odds_pool([[0.8, 0.8]], extremization=2.0)) == \
        bits([0.9411764705882353])
    # a=1 recovers the geometric pool (allclose — the two use different
    # but equivalent float formulations, cf. repo test_calibration.py);
    # int factors accepted and bit-identical to pristine
    P = np.array([[0.6, 0.7], [0.2, 0.9]])
    assert np.allclose(live.log_odds_pool(P, extremization=1),
                       live.geometric_pool(P))
    if pristine is not None:
        assert bits(live.log_odds_pool(P, extremization=1)) == \
            bits(pristine.log_odds_pool(P, extremization=1))


def test_docstring_documents_raise():
    doc = live.log_odds_pool.__doc__ + live._validate_extremization.__doc__
    assert "Raises" in doc and "ValueError" in doc and "non-finite" in doc


def test_optimize_pool_weights_still_fits():
    rng = np.random.default_rng(0)
    P = rng.uniform(0.05, 0.95, size=(80, 3))
    y = (rng.uniform(size=80) < P.mean(axis=1)).astype(float)
    res = live.optimize_pool_weights(y, P, method="extremized")
    assert res["success"] is not None and res["weights"] is not None, res
    assert res["extremization"] is not None and np.isfinite(res["extremization"])
    # pool_fn built on log_odds_pool must run with the fitted a
    out = res["pool_fn"](P)
    assert np.all(np.isfinite(out))


if __name__ == "__main__":
    check("task repros: a=nan / a=inf on [[0.7,0.3]] raise documented ValueError",
          test_task_repros_raise)
    check("reject grid: non-finite / non-positive / non-scalar all raise named ValueError",
          test_reject_grid)
    check("kill criterion: a=inf never returns 0.0/1.0, a=nan never returns NaN",
          test_kill_criterion)
    check("alias extremized_pool guarded identically", test_alias_also_guarded)
    check("finite-a [0.02,0.98] grid bit-identical vs pristine AND pre-9513 tree",
          test_finite_grid_bit_identical)
    check("known values + a=1 == geometric pool + int factor", test_known_values)
    check("docstring documents the ValueError", test_docstring_documents_raise)
    check("optimize_pool_weights (internal caller) still fits and pools",
          test_optimize_pool_weights_still_fits)
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
