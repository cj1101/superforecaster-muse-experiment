"""
test_calibration_pool_bool_members.py — SF-BL-10975 verification.

Pre-registered done/kill criteria (TASK.md):
  "Pass if linear_pool, geometric_pool and log_odds_pool raise a named
   ValueError on any bool member while lawful float/int panels stay
   bit-identical; kill if [[True, 0.5]] still returns 0.99968 or any
   float from any of the three pools."

Bug: _as_matrix converted the raw panel with
np.asarray(probs, dtype=float) before any screen, coercing True->1.0 /
False->0.0, so a boolean flag in a member field became the most
extreme lawful forecast in the pool:
  linear_pool([[True, False]])  == [0.5]
  geometric_pool([[True, 0.5]]) == [0.9996838721866521]
  log_odds_pool([[True, 0.5]])  == [0.9996838721866521]

Fix (superforecaster/calibration.py): _reject_bool_pool_members
screens the RAW probs via _has_bool at the top of _as_matrix, before
the float conversion, raising
"probs must not contain boolean values; got boolean value(s) in probs
(a bool is not a valid probs member) (SF-BL-10975)".
Full cross-copy bit-identity evidence (live vs pristine snapshots,
22/22 checks) lives in the SF-BL-10975 backlog work dir.

Run:  python3 tests/test_calibration_pool_bool_members.py
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
from scipy.special import expit, logit  # noqa: E402

from superforecaster.calibration import (  # noqa: E402
    extremized_pool, geometric_pool, linear_pool, log_odds_pool,
    optimize_pool_weights)

PASS = []
POOLS = (linear_pool, geometric_pool, log_odds_pool, extremized_pool)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))


def _bool_error(fn, *args, **kw):
    try:
        fn(*args, **kw)
    except ValueError as e:
        msg = str(e)
        if "boolean" in msg and "probs" in msg and "SF-BL-10975" in msg:
            return e
        raise AssertionError(f"ValueError not the named bool error: {msg}")
    return None


def test_titled_repros_raise():
    for fn in POOLS:
        assert _bool_error(fn, [[True, False]]) is not None, fn.__name__
        # kill repro: must not return 0.99968 or any float
        assert _bool_error(fn, [[True, 0.5]]) is not None, fn.__name__


def test_bool_forms_raise():
    panels = [
        [[True, False]], [[0.5, False]], [True, 0.5], (True, 0.5),
        [[0.2, 0.8], [True, 0.1]], np.array([[True, False]]),
        np.array([[True, 0.5]], dtype=object),
        np.array([0.5, True], dtype=object), True,
    ]
    for panel in panels:
        for fn in POOLS:
            assert _bool_error(fn, panel) is not None, (fn.__name__, panel)
    assert _bool_error(linear_pool, [[True, 0.5]], weights=[0.5, 0.5]) is not None
    for a in (0.5, 2.0):
        assert _bool_error(log_odds_pool, [[True, 0.5]], extremization=a) is not None
    rng = np.random.default_rng(7)
    panel = rng.random((60, 3)).tolist(); panel[5][1] = True
    y = (rng.random(60) > 0.5).astype(int)
    # optimize_pool_weights shares _as_matrix (SF-BL-10975 screen); the
    # parallel SF-BL-10976 explicit screen there may cite its own id —
    # either named bool ValueError satisfies this check.
    try:
        optimize_pool_weights(y, panel)
        raise SystemExit("optimize_pool_weights did not raise on bool panel")
    except ValueError as e:
        assert "boolean" in str(e) and "probs" in str(e), str(e)


def test_lawful_panels_unchanged():
    # Independent recomputation of the three pool formulas.
    P = np.array([[0.2, 0.8], [0.6, 0.4], [0.0, 1.0], [1, 0]])
    w = np.array([0.5, 0.5])
    assert np.array_equal(linear_pool(P), np.clip(P @ w, 0.0, 1.0))
    Pc = np.clip(P, 1e-7, 1 - 1e-7)
    geo = expit((np.log(Pc) * w).sum(axis=1) - (np.log(1 - Pc) * w).sum(axis=1))
    assert np.array_equal(geometric_pool(P), geo)
    assert np.array_equal(log_odds_pool(P), expit(logit(Pc) @ w))
    assert np.array_equal(log_odds_pool(P, extremization=2.0),
                          expit(2.0 * (logit(Pc) @ w)))
    # int endpoints remain lawful; exact spot values
    assert linear_pool([[0, 1], [1, 0]]).tolist() == [0.5, 0.5]
    assert linear_pool([[1, 0]]).tolist() == [0.5]


def test_preexisting_errors_unchanged():
    def raises(fn, *a, **k):
        try:
            fn(*a, **k)
        except ValueError:
            return True
        return False
    for fn in (linear_pool, geometric_pool, log_odds_pool):
        assert raises(fn, [[1.5, 0.5]])            # range (SF-BL-10934)
        assert raises(fn, [[82.0, 0.5]])           # range (SF-BL-10934)
        assert raises(fn, [[float("nan"), 0.5]])   # finiteness (SF-BL-3596)
        assert raises(fn, [[0.2, 0.8]], weights=[-0.5, 1.5]) if fn is linear_pool else True
    # extremization keeps precedence over the bool screen
    try:
        log_odds_pool([[True, 0.5]], extremization=0)
        raise SystemExit("extremization=0 did not raise")
    except ValueError as e:
        assert "extremization" in str(e) and "SF-BL-10975" not in str(e)
    # bool weights are NOT screened (SF-BL-10883 scope): unchanged behaviour
    assert linear_pool([[0.2, 0.8]], weights=[True, False]).tolist() == [0.2]


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
