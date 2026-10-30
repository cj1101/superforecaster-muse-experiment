"""SF-BL-9782: engine.log_odds_pool must not extremize via negative weights.

TASK repro (live, pre-fix): log_odds_pool([0.90, 0.10], weights=[2.0, -1.0])
returned 0.9986 -- above the highest member (0.90) with extremization=1.0.
A sign-flipped weight silently became an anti-pool.

Done criterion: ValueError on any negative or non-finite weight and on a
zero-sum weight vector (no silent equal-weight fallback); equal/positive
-weight fixtures byte-identical. This task is the [0.90, 0.10] extremize
framing of the defect SF-BL-9512 fixed for [0.3, 0.7]; this suite locks
the 9782 repro, the within-range property for valid pools, and the
9782-specific golden panel (goldens cross-checked bit-for-bit against an
independent pure-math replica of the valid-weight path -- arithmetically
identical pre- and post-fix -- so byte-identity is proven, not assumed).

Run standalone: python3 test_engine_log_odds_pool_9782_extremize.py
(also pytest-collectable). Exit 0 iff all checks pass.
"""
import math
import os
import struct
import sys

sys.path.insert(0, os.path.join(
    os.path.expanduser("~/workspace/superforecaster")))

from superforecaster.engine import Superforecaster, log_odds_pool  # noqa: E402

NAN, INF = float("nan"), float("inf")
PASS = []
EPS = 1e-6


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))


def bits(x):
    return struct.pack("<d", float(x))


def analytic_valid_pool(probs, weights, extremization=1.0):
    """Independent replica of the VALID-weight path (pure math)."""
    p = [min(max(x, EPS), 1 - EPS) for x in probs]
    if weights is None:
        w = [1.0 / len(p)] * len(p)
    else:
        s = sum(weights)
        w = [x / s for x in weights]
    z = sum(wi * math.log(pi / (1.0 - pi)) for wi, pi in zip(w, p))
    z *= extremization
    return 1.0 / (1.0 + math.exp(-z))


def must_raise(probs, weights, **kw):
    try:
        out = log_odds_pool(probs, weights=weights, **kw)
    except ValueError as e:
        assert type(e) is ValueError, f"wrong type {type(e)}"
        assert "INVALID_INPUT" in str(e), f"no INVALID_INPUT flag: {e}"
        assert "weights" in str(e), f"weights not named: {e}"
        return str(e)
    raise AssertionError(
        f"log_odds_pool({probs}, weights={weights}) RETURNED {out!r}")


def test_task_verbatim_repro_raises():
    # The exact call from TASK.md / the task title.
    must_raise([0.90, 0.10], [2.0, -1.0])
    must_raise([0.90, 0.10], [2, -1])  # int form


def test_negative_weights_raise_9782_panel():
    for w in ([-1.0, 1.0], [1.0, -1.0], [-0.5, 2.0], [-1e-12, 1.0],
              [-1.0, -1.0], [2.0, -1.0]):
        must_raise([0.90, 0.10], list(w))
    must_raise([0.90, 0.10, 0.50], [2.0, -1.0, 0.5])
    must_raise([0.90], [-1.0])


def test_nonfinite_weights_raise():
    for bad in (NAN, INF, -INF):
        must_raise([0.90, 0.10], [bad, 1.0])
        must_raise([0.90, 0.10], [1.0, bad])
    must_raise([0.90, 0.10], [NAN, NAN])


def test_zero_sum_raises_no_silent_fallback():
    # Pre-fix these silently returned the equal-weight pool (0.5 here).
    must_raise([0.90, 0.10], [0.0, 0.0])
    must_raise([0.90, 0.10], [-0.0, -0.0])
    must_raise([0.90], [0.0])
    must_raise([0.90, 0.10, 0.50], [0.0, 0.0, 0.0])
    # A zero-sum vector built from mixed signs is negative-bearing too.
    must_raise([0.90, 0.10], [1.0, -1.0])


def test_valid_pools_byte_identical():
    # Goldens proven bit-equal to the independent analytic replica of
    # the valid path (see module docstring) -- not self-captured only.
    cases = [
        (([0.90, 0.10], None), 0.5000000000000001),
        (([0.90, 0.10], [1.0, 1.0]), 0.5000000000000001),
        (([0.90, 0.10], [2.0, 1.0]), 0.675333511212968),
        (([0.90, 0.10], [1.0, 2.0]), 0.32466648878703214),
        (([0.90, 0.10], [3.0, 1.0]), 0.75),
        (([0.90, 0.10], [0.0, 1.0]), 0.10000000000000002),
        (([0.90, 0.10], [1.0, 0.0]), 0.8999999999999999),
        (([0.90, 0.10], [-0.0, 1.0]), 0.10000000000000002),
        (([0.90, 0.10, 0.50], [1.0, 1.0, 1.0]), 0.5),
        (([0.90, 0.10, 0.50], [2.0, 0.5, 1.0]), 0.7194363970790442),
        (([0.90], [1.0]), 0.8999999999999999),
        (([0.02, 0.98], [1.0, 1.0]), 0.4999999999999999),
    ]
    for (probs, w), expected in cases:
        out = log_odds_pool(probs, weights=w)
        assert bits(out) == bits(expected), (probs, w, out, expected)
        assert bits(out) == bits(analytic_valid_pool(probs, w)), \
            (probs, w, out, analytic_valid_pool(probs, w))
        assert math.isfinite(out) and 0.0 < out < 1.0


def test_valid_pool_stays_within_member_range():
    # The 9782 framing: with valid weights and extremization=1.0 a
    # log-odds pool is a weighted average of logits, so the result can
    # never exceed the highest (or dip below the lowest) member.
    panels = [
        ([0.90, 0.10], [1.0, 1.0]), ([0.90, 0.10], [2.0, 1.0]),
        ([0.90, 0.10], [1.0, 2.0]), ([0.90, 0.10], [0.0, 1.0]),
        ([0.20, 0.40, 0.80], [3.0, 1.0, 0.5]),
        ([0.02, 0.98], [1.0, 1.0]), ([0.65, 0.65], [1.0, 3.0]),
    ]
    for probs, w in panels:
        out = log_odds_pool(probs, weights=w)
        assert min(probs) - 1e-12 <= out <= max(probs) + 1e-12, \
            (probs, w, out)


def test_extremization_interplay():
    # Invalid weights raise at any lawful extremization; valid weights
    # at the cap still pool (and MAY exceed members -- that is the
    # caller's explicit, capped choice, unlike the silent anti-pool).
    must_raise([0.90, 0.10], [2.0, -1.0], extremization=1.40)
    must_raise([0.90, 0.10], [0.0, 0.0], extremization=0.5)
    out = log_odds_pool([0.80, 0.80], weights=[1.0, 1.0],
                        extremization=1.40)
    assert bits(out) == bits(0.8744413346659732), out


def test_forecast_path_unchanged():
    sf = Superforecaster()
    # forecast() drops w<=0 agents by value before pooling; unaffected.
    r = sf.forecast(prior=0.5, agent_probs=[0.7], agent_weights=[-1.0])
    assert r["n_components"] == 0 and r["p"] == 0.5, r
    d3 = sf.forecast(prior=0.50, market_price=0.62,
                     agent_probs=[0.70, 0.66], agent_weights=[0.16, 0.14])
    assert d3 == {"p": 0.629249601776721, "anchor": 0.62,
                  "pooled_components": 0.629249601776721,
                  "n_components": 2, "extremization": 1.0}, d3


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for fn in ALL_TESTS:
        try:
            fn()
            check(fn.__name__, True)
        except AssertionError as e:
            check(fn.__name__, False, str(e))
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
