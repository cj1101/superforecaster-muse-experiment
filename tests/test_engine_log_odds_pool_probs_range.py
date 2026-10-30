"""SF-BL-10936 regression tests: engine.log_odds_pool must reject any
finite probs member outside [0, 1] with INVALID_INPUT instead of
clipping it to the clip bound (0.999999 / 1e-06) and pooling it as
near-certainty.

Pre-fix repro (pristine engine):
  log_odds_pool([1.5, 0.5]) -> 0.9990009985019828 (TASK's 0.999001)

Scope: engine.log_odds_pool's probs only. forecast()'s components are
SF-BL-10881, prior is SF-BL-9890, NaN/inf members are SF-BL-9510,
string members are SF-BL-11054, weights are SF-BL-9512, extremization
is SF-BL-8900 (all separate, already covered). Endpoints 0.0 / 1.0 are
inside [0, 1] and keep their established _clip-to-eps behaviour.
calibration.py's and agents.py's own log_odds_pool functions are
separate functions, unaffected.

Run: python3 -m pytest tests/test_engine_log_odds_pool_probs_range.py -q
(or plain python3 — it is self-checking).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from superforecaster.engine import Superforecaster, log_odds_pool

SF = Superforecaster()


def assert_invalid(fn):
    try:
        out = fn()
    except ValueError as e:
        assert "INVALID_INPUT" in str(e), f"error lacks INVALID_INPUT: {e}"
        assert "probs" in str(e), f"error does not name probs: {e}"
        return
    raise AssertionError(f"expected INVALID_INPUT ValueError, got return {out!r}")


def test_task_fixture_rejected():
    # The exact repro from the task filing must raise, never return a float.
    assert_invalid(lambda: log_odds_pool([1.5, 0.5]))


def test_just_outside_and_extreme_rejected():
    for bad in (1.000001, -1e-9, 100.0, -100.0, 2.0, -0.5, 82.0):
        assert_invalid(lambda b=bad: log_odds_pool([b, 0.5]))
        assert_invalid(lambda b=bad: log_odds_pool([b]))


def test_bad_element_anywhere_rejected():
    assert_invalid(lambda: log_odds_pool([0.7, 82.0]))
    assert_invalid(lambda: log_odds_pool([0.7, 0.6, -0.2]))


def test_rejection_not_rescued_by_zero_weight_or_extremization():
    # Range validation precedes weights/extremization handling: a
    # corrupt member with zero weight still raises (pre-fix it
    # returned 0.5), and a lawful extremization does not rescue it.
    assert_invalid(lambda: log_odds_pool([1.5, 0.5], weights=[0.0, 1.0]))
    assert_invalid(lambda: log_odds_pool([1.5, 0.5], weights=[1.0, 1.0]))
    assert_invalid(lambda: log_odds_pool([1.5, 0.5], extremization=1.3))


def test_boundaries_and_lawful_pools_unchanged():
    # Endpoints are inside [0, 1] and keep established behaviour;
    # lawful pools keep their exact pristine values.
    assert log_odds_pool([0.0]) == 1.0000000000000008e-06
    assert log_odds_pool([1.0]) == 0.999999
    assert log_odds_pool([0.0, 1.0]) == 0.49999999999640554
    assert log_odds_pool([0.7, 0.3]) == 0.5
    assert log_odds_pool([0.5]) == 0.5
    assert log_odds_pool([0.7, 0.3], weights=[2.0, 1.0]) == 0.5701425119992315
    assert log_odds_pool([0.7, 0.3], weights=[0.0, 1.0]) == 0.30000000000000004
    assert log_odds_pool([0.8, 0.6], extremization=1.3) == 0.7621754669326366
    assert log_odds_pool([0.8, 0.6], extremization=0.5) == 0.6101493074464512


def test_prior_guards_preserved():
    # NaN/inf still raise via the SF-BL-9510 finite screen, negative
    # weights via SF-BL-9512, bad extremization via SF-BL-8900, and a
    # string member via SF-BL-11054 — all INVALID_INPUT ValueErrors.
    for fn in (lambda: log_odds_pool([float("nan"), 0.5]),
               lambda: log_odds_pool([float("inf"), 0.5]),
               lambda: log_odds_pool([0.3, 0.7], weights=[2.0, -1.0]),
               lambda: log_odds_pool([0.3, 0.7], weights=[0.0, 0.0]),
               lambda: log_odds_pool([0.3, 0.7], extremization=2.0),
               lambda: log_odds_pool(["0.7", 0.3]),
               lambda: log_odds_pool([])):
        try:
            out = fn()
        except ValueError as e:
            assert "INVALID_INPUT" in str(e), f"error lacks INVALID_INPUT: {e}"
            continue
        raise AssertionError(f"expected INVALID_INPUT ValueError, got return {out!r}")


def test_demos_bit_identical():
    d1 = SF.forecast(prior=0.55, model_probs=[0.817], model_skills=[-0.007])
    assert d1 == {"p": 0.55, "anchor": 0.55, "n_components": 0,
                  "note": "no validated component; prior shipped unchanged"}, d1
    d2 = SF.forecast(prior=0.10, model_probs=[0.16], model_skills=[0.024])
    assert d2["p"] == 0.10144000000000002, d2
    d3 = SF.forecast(prior=0.50, market_price=0.62,
                     agent_probs=[0.70, 0.66], agent_weights=[0.16, 0.14])
    assert d3["p"] == 0.629249601776721, d3


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} PASS")
