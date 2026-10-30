"""SF-BL-10980 regression tests: engine skill_weight, shrink
(p_model/p_prior) and log_odds_pool (probs members) must reject bool
inputs with INVALID_INPUT instead of float()-coercing them to the
most extreme lawful value (1.0/0.0).

Pre-fix repro (pristine engine):
  skill_weight(False, 0.5) -> 1.0 (maximum pool weight)
  shrink(0.8, True, 0.5) -> 0.9; shrink(True, False, 0.5) -> 0.5
  log_odds_pool([True, 0.5]) -> 0.9990009985019828 (TASK's 0.999001)

Scope: skill_weight's two Brier arguments, shrink's p_model/p_prior,
and log_odds_pool's probs container + members ONLY. shrink's w,
skill_weight_from_skill and the constructor are SF-BL-10937;
log_odds_pool's weights are SF-BL-11022; forecast() is SF-BL-10921
(all separate, preserved). Int 1/0 are NOT bools and stay lawful.

Run: python3 -m pytest tests/test_engine_bool_skill_shrink_pool.py -q
(or plain python3 — it is self-checking).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from decimal import Decimal
from superforecaster.engine import Superforecaster, skill_weight, shrink, log_odds_pool

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


def test_task_fixtures_rejected():
    assert_invalid(lambda: skill_weight(False, 0.5), "brier_model")
    assert_invalid(lambda: log_odds_pool([True, 0.5]), "probs[0]")


def test_skill_weight_any_bool_rejected():
    for bad_args in [(True, 0.5), (0.5, True), (0.5, False), (True, True),
                     (np.bool_(True), 0.5), (0.5, np.bool_(False))]:
        assert_invalid(lambda a=bad_args: skill_weight(*a), "skill_weight")


def test_shrink_bool_model_or_prior_rejected():
    for bad_args in [(0.8, True, 0.5), (True, False, 0.5), (True, 0.5, 0.5),
                     (0.8, False, 0.5), (np.bool_(False), 0.5, 0.5),
                     (0.5, np.bool_(True), 0.5)]:
        assert_invalid(lambda a=bad_args: shrink(*a), "shrink")


def test_log_odds_pool_bool_member_rejected():
    assert_invalid(lambda: log_odds_pool([False, 0.5]), "probs[0]")
    assert_invalid(lambda: log_odds_pool([0.5, True]), "probs[1]")
    assert_invalid(lambda: log_odds_pool([True]), "probs[0]")
    assert_invalid(lambda: log_odds_pool([np.bool_(True), 0.5]), "probs[0]")
    assert_invalid(lambda: log_odds_pool((False, 0.5)), "probs[0]")
    assert_invalid(lambda: log_odds_pool(True), "probs")


def test_rejection_not_rescued_by_zero_weight():
    # Bool screening precedes weights handling: a bool member with
    # zero weight still raises (pre-fix it returned 0.5).
    assert_invalid(lambda: log_odds_pool([True, 0.5], weights=[0.0, 1.0]), "probs[0]")


def test_lawful_calls_unchanged():
    assert skill_weight(0.10, 0.25) == 0.6
    assert skill_weight(0, 0.5) == 1.0          # int 0 is lawful, not a bool
    assert skill_weight(1, 1) == 0.0
    assert skill_weight(np.float64(0.1), np.float64(0.25)) == 0.6
    assert skill_weight(Decimal("0.1"), Decimal("0.25")) == 0.6
    assert skill_weight(0.25, 0.0) == 0.0       # prior-zero guard intact
    assert shrink(0.8, 0.5, 0.5) == 0.65
    assert shrink(1, 0, 1) == 0.999999          # int endpoints lawful
    assert log_odds_pool([0.7, 0.3]) == 0.5
    assert log_odds_pool([1, 0]) == 0.49999999999640554  # int members lawful
    assert log_odds_pool([0.0, 1.0]) == 0.49999999999640554
    assert log_odds_pool([0.7, 0.3], weights=[2.0, 1.0]) == 0.5701425119992315


def test_prior_guards_preserved():
    # SF-BL-11054 strings, SF-BL-9781 Brier domain, SF-BL-9510
    # nonfinite, SF-BL-10936 probs range all still raise INVALID_INPUT.
    for fn in (lambda: skill_weight("0.10", 0.25),
               lambda: skill_weight(-0.1, 0.25),
               lambda: skill_weight(float("nan"), 0.25),
               lambda: shrink("0.8", 0.5, 0.5),
               lambda: log_odds_pool(["0.7", "0.3"]),
               lambda: log_odds_pool([1.5, 0.5]),
               lambda: log_odds_pool("0.7")):
        try:
            out = fn()
        except ValueError as e:
            assert "INVALID_INPUT" in str(e), str(e)
            continue
        raise AssertionError(f"prior guard lost: got return {out!r}")


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
