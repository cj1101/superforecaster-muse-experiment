"""SF-BL-9512: engine.log_odds_pool must reject invalid weights.

Bug (verified live 2026-10-23 against pristine sha256 bc0d5d5b...819):
log_odds_pool validated only the weights length (SF-BL-8868), then ran
``w = w / w.sum() if w.sum() > 0 else np.full(len(p), 1.0/len(p))``:
  log_odds_pool([0.3, 0.7], weights=[2.0, -1.0])  -> 0.072972972972973
      (the -1.0 member was SUBTRACTED, not weighted)
  log_odds_pool([0.3, 0.7], weights=[-0.5, 1.0])  -> 0.927027027027027
  log_odds_pool([0.3, 0.7], weights=[-1.0, -1.0]) -> 0.5 (silent equal fallback)
  log_odds_pool([0.3, 0.7], weights=[0.0, 0.0])   -> 0.5 (silent equal fallback)
NaN/inf weights already raised INVALID_INPUT via the SF-BL-9510
_require_finite_seq screen; this suite locks that in too.

Done criterion: negative, NaN or infinite weights raise a documented
ValueError (no silent equal-weight substitution) and equal/valid-weight
pools on benchmark panels are bit-identical. Kill: weights=[2,-1] or
[nan,1] still return a pooled float.

Run standalone: python3 test_engine_log_odds_pool_weights_validation.py
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


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))


def bits(x):
    return struct.pack("<d", float(x))


def must_raise(probs, weights, **kw):
    """Return the ValueError message; fail loudly if a float comes back."""
    try:
        out = log_odds_pool(probs, weights=weights, **kw)
    except ValueError as e:
        assert type(e) is ValueError, f"wrong type {type(e)}"
        assert "INVALID_INPUT" in str(e), f"no INVALID_INPUT flag: {e}"
        assert "weights" in str(e), f"weights not named: {e}"
        return str(e)
    raise AssertionError(
        f"log_odds_pool({probs}, weights={weights}) RETURNED {out!r}")


def test_task_kill_repros():
    must_raise([0.3, 0.7], [2.0, -1.0])
    must_raise([0.3, 0.7], [NAN, 1.0])
    must_raise([0.3, 0.7], [2, -1])  # int form, TASK verbatim


def test_negative_weights_raise():
    for w in ([-1.0, 1.0], [1.0, -1.0], [-0.5, 1.0], [-1e-12, 1.0],
              [-100.0, 1.0], [-1.0, -1.0], [-2.0, -3.0]):
        must_raise([0.3, 0.7], list(w))
    must_raise([0.1, 0.5, 0.9], [3.0, -1.0, 0.5])
    must_raise([0.1, 0.5, 0.9], [3.0, 1.0, -0.5])
    must_raise([0.5], [-1.0])


def test_nonfinite_weights_raise():
    for bad in (NAN, INF, -INF):
        must_raise([0.3, 0.7], [bad, 1.0])
        must_raise([0.3, 0.7], [1.0, bad])
    must_raise([0.3, 0.7], [NAN, NAN])


def test_all_zero_weights_raise_no_silent_substitution():
    must_raise([0.3, 0.7], [0.0, 0.0])
    must_raise([0.5], [0.0])
    must_raise([0.3, 0.7], [-0.0, -0.0])
    must_raise([0.1, 0.5, 0.9], [0.0, 0.0, 0.0])


def test_valid_pools_bit_identical():
    # Values captured from the pristine implementation (2026-10-23).
    cases = [
        (([0.3, 0.7], None), 0.5),
        (([0.3, 0.7], [1.0, 1.0]), 0.5),
        (([0.3, 0.7], [2.0, 1.0]), 0.42985748800076845),
        (([0.3, 0.7], [0.0, 1.0]), 0.7),
        (([0.3, 0.7], [1.0, 0.0]), 0.30000000000000004),
        (([0.3, 0.7], [-0.0, 1.0]), 0.7),
        (([0.6, 0.7], None), 0.6516685226452116),
        (([0.6, 0.7], [1.0, 1.0]), 0.6516685226452116),
        (([0.6, 0.7], [2.0, 1.0]), 0.6347716821524939),
        (([0.1, 0.5, 0.9], [3.0, 1.0, 0.5]), 0.22781674891972453),
        (([0.1, 0.5, 0.9], None), 0.5),
        (([0.02, 0.98], [1.0, 1.0]), 0.4999999999999999),
        (([0.5], [1.0]), 0.5),
    ]
    for (probs, w), expected in cases:
        out = log_odds_pool(probs, weights=w)
        assert bits(out) == bits(expected), (probs, w, out, expected)
        assert math.isfinite(out) and 0.0 < out < 1.0


def test_extremization_interplay():
    # Valid weights still pool under an allowed extremization;
    # invalid weights raise regardless of the factor.
    assert bits(log_odds_pool([0.8, 0.8], weights=[1.0, 1.0],
                              extremization=1.40)) == bits(0.8744413346659732)
    must_raise([0.8, 0.8], [2.0, -1.0], extremization=1.40)
    must_raise([0.8, 0.8], [0.0, 0.0], extremization=0.5)


def test_preserved_screens():
    # SF-BL-8868 length mismatch and SF-BL-11054 string weights unchanged.
    try:
        log_odds_pool([0.3, 0.7], weights=[1.0])
        raise SystemExit("length mismatch no longer raises")
    except ValueError as e:
        assert "length mismatch" in str(e), str(e)
    try:
        log_odds_pool([0.3, 0.7], weights=["1.0", "1.0"])
        raise SystemExit("string weights no longer raise")
    except ValueError as e:
        assert "INVALID_INPUT" in str(e), str(e)


def test_forecast_path_unchanged():
    sf = Superforecaster()
    # Intentional drops by value are by design and unaffected: a
    # negative agent weight is dropped in forecast(), never pooled.
    r = sf.forecast(prior=0.5, agent_probs=[0.7], agent_weights=[-1.0])
    assert r["n_components"] == 0 and r["p"] == 0.5, r
    r = sf.forecast(prior=0.5, agent_probs=[0.7, 0.6],
                    agent_weights=[0.0, 1.0])
    assert r["n_components"] == 1, r
    # The three __main__ demos stay bit-identical (full dict equality).
    d1 = sf.forecast(prior=0.55, model_probs=[0.817], model_skills=[-0.007])
    assert d1 == {"p": 0.55, "anchor": 0.55, "n_components": 0,
                  "note": "no validated component; prior shipped unchanged"}, d1
    d2 = sf.forecast(prior=0.10, model_probs=[0.16], model_skills=[0.024])
    assert d2 == {"p": 0.10144000000000002, "anchor": 0.1,
                  "pooled_components": 0.10144000000000002,
                  "n_components": 1, "extremization": 1.0}, d2
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
