"""SF-BL-11022 regression tests: engine log_odds_pool must reject
bool weights with INVALID_INPUT instead of float()-coercing them
to a 1/0 member mask.

Pre-fix repro (pristine engine):
  log_odds_pool([0.7, 0.3], weights=[True, False]) -> 0.7 exactly
  (pool silently collapsed to its first component)

Scope: log_odds_pool's weights container + entries ONLY. Bool probs
members are SF-BL-10980; extremization is a separate dimension
(preserved). Int 1/0 weights are NOT bools and stay lawful.

Run: python3 -m pytest tests/test_engine_log_odds_pool_bool_weights.py -q
(or plain python3 — it is self-checking).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from decimal import Decimal
from superforecaster.engine import Superforecaster, log_odds_pool

SF = Superforecaster()


def assert_invalid(fn, *names):
    try:
        out = fn()
    except ValueError as e:
        assert "INVALID_INPUT" in str(e), f"error lacks INVALID_INPUT: {e}"
        assert "bool" in str(e), f"error does not mention bool: {e}"
        for n in names:
            assert n in str(e), f"error does not name {n!r}: {e}"
        return
    raise AssertionError(f"expected INVALID_INPUT ValueError, got return {out!r}")


def test_task_kill_repro_rejected():
    assert_invalid(lambda: log_odds_pool([0.7, 0.3], weights=[True, False]), "weights[0]")


def test_any_bool_weight_rejected():
    for w in ([False, True], [1.0, True], [True, 1.0], [True], [False],
              [np.bool_(True), 1.0], (True, False), [1.0, False, 1.0]):
        probs = [0.5] if len(w) == 1 else ([0.1, 0.5, 0.9] if len(w) == 3 else [0.7, 0.3])
        assert_invalid(lambda p=probs, w=w: log_odds_pool(p, weights=w), "weights")


def test_bool_container_rejected():
    assert_invalid(lambda: log_odds_pool([0.7, 0.3], weights=True), "weights")
    assert_invalid(lambda: log_odds_pool([0.7, 0.3], weights=False), "weights")
    assert_invalid(lambda: log_odds_pool([0.7, 0.3], weights=np.bool_(True)), "weights")
    assert_invalid(lambda: log_odds_pool([0.7, 0.3], weights=np.array([True, False])), "weights")
    assert_invalid(lambda: log_odds_pool([0.7, 0.3], weights=np.array(True)), "weights")


def test_rejection_not_rescued():
    # A False weight is not a lawful zero/drop, and a bool vector
    # raises the bool error even when its length also mismatches.
    assert_invalid(lambda: log_odds_pool([0.7, 0.3], weights=[False, 1.0]), "weights[0]")
    assert_invalid(lambda: log_odds_pool([0.7, 0.3], weights=[True]), "weights[0]")


def test_lawful_weighted_pools_bit_identical():
    assert log_odds_pool([0.7, 0.3]) == 0.5
    assert log_odds_pool([0.3, 0.7], weights=[1.0, 1.0]) == 0.5
    assert log_odds_pool([0.3, 0.7], weights=[2.0, 1.0]) == 0.42985748800076845
    assert log_odds_pool([0.7, 0.3], weights=[2.0, 1.0]) == 0.5701425119992315
    assert log_odds_pool([0.3, 0.7], weights=[0.0, 1.0]) == 0.7
    assert log_odds_pool([0.3, 0.7], weights=[1.0, 0.0]) == 0.30000000000000004
    assert log_odds_pool([0.3, 0.7], weights=[-0.0, 1.0]) == 0.7
    assert log_odds_pool([0.7, 0.3], weights=[1, 0]) == 0.7        # int weights lawful
    assert log_odds_pool([0.7, 0.3], weights=[0, 1]) == 0.30000000000000004
    assert log_odds_pool([0.7, 0.3], weights=[2, 1]) == 0.5701425119992315
    assert log_odds_pool([0.7, 0.3], weights=[np.float64(2.0), np.float64(1.0)]) == 0.5701425119992315
    assert log_odds_pool([0.7, 0.3], weights=[Decimal("2.0"), Decimal("1.0")]) == 0.5701425119992315
    assert log_odds_pool([0.1, 0.5, 0.9], weights=[3.0, 1.0, 0.5]) == 0.22781674891972453
    assert log_odds_pool([0.5], weights=[1.0]) == 0.5
    assert log_odds_pool([0.7, 0.3], weights=np.array([2.0, 1.0])) == 0.5701425119992315


def test_prior_guards_preserved():
    for fn in (lambda: log_odds_pool([0.3, 0.7], weights=[2.0, -1.0]),
               lambda: log_odds_pool([0.3, 0.7], weights=[float("nan"), 1.0]),
               lambda: log_odds_pool([0.3, 0.7], weights=[float("inf"), 1.0]),
               lambda: log_odds_pool([0.3, 0.7], weights=[0.0, 0.0]),
               lambda: log_odds_pool([0.3, 0.7], weights=["0.5", "0.5"]),
               lambda: log_odds_pool([0.3, 0.7], weights=[1.0]),
               lambda: log_odds_pool([True, 0.5]),
               lambda: log_odds_pool([1.5, 0.5])):
        try:
            out = fn()
        except ValueError as e:
            assert "INVALID_INPUT" in str(e) or "length mismatch" in str(e), str(e)
            continue
        raise AssertionError(f"prior guard lost: got return {out!r}")


def test_out_of_scope_preserved():
    # extremization bool dimension is a separate task, preserved.
    assert log_odds_pool([0.7, 0.3], extremization=True) == 0.5


def test_demos_bit_identical():
    assert SF.forecast(prior=0.55, model_probs=[0.817], model_skills=[-0.007]) == \
        {"p": 0.55, "anchor": 0.55, "n_components": 0,
         "note": "no validated component; prior shipped unchanged"}
    assert SF.forecast(prior=0.10, model_probs=[0.16], model_skills=[0.024])["p"] == 0.10144000000000002
    assert SF.forecast(prior=0.50, market_price=0.62, agent_probs=[0.70, 0.66],
                       agent_weights=[0.16, 0.14])["p"] == 0.629249601776721


if __name__ == "__main__":
    for _name, _fn in sorted(dict(globals()).items()):
        if _name.startswith("test_"):
            _fn(); print(f"PASS {_name}")
    print("ALL PASS")
