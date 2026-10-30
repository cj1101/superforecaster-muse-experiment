"""SF-BL-9890 regression tests: forecast() must reject a finite prior
outside [0, 1] with INVALID_INPUT instead of clipping it to the clip
bound (0.999999 / 1e-06) and shipping it as near-certainty under the
note 'prior shipped unchanged'.

Scope: the prior argument only. market_price range is SF-BL-10981
and model/agent component ranges are SF-BL-10881 (separate tasks);
the NaN/inf prior arm is SF-BL-9510 (already covered by
test_engine_nonfinite.py).

Run: python3 -m pytest tests/test_engine_prior_range.py -q
(or plain python3 — it is self-checking).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from superforecaster.engine import Superforecaster

SF = Superforecaster()


def assert_invalid(fn):
    try:
        out = fn()
    except ValueError as e:
        assert "INVALID_INPUT" in str(e), f"error lacks INVALID_INPUT: {e}"
        assert "prior" in str(e), f"error does not name the prior argument: {e}"
        return
    raise AssertionError(f"expected INVALID_INPUT ValueError, got return {out!r}")


def test_task_fixtures_rejected():
    # The exact repros from the task filing.
    assert_invalid(lambda: SF.forecast(prior=1.5))
    assert_invalid(lambda: SF.forecast(prior=2.0))
    assert_invalid(lambda: SF.forecast(prior=-0.5))


def test_just_outside_boundaries_rejected():
    assert_invalid(lambda: SF.forecast(prior=1.000001))
    assert_invalid(lambda: SF.forecast(prior=-1e-9))
    assert_invalid(lambda: SF.forecast(prior=100.0))
    assert_invalid(lambda: SF.forecast(prior=-100.0))


def test_rejected_on_every_path():
    # Rejection precedes pooling/anchoring: components or a valid
    # market_price must not rescue an impossible prior.
    assert_invalid(lambda: SF.forecast(prior=1.5, agent_probs=[0.7, 0.6]))
    assert_invalid(lambda: SF.forecast(prior=-0.5, model_probs=[0.7],
                                       model_skills=[0.1]))
    assert_invalid(lambda: SF.forecast(prior=1.5, market_price=0.62))


def test_boundaries_and_in_range_unchanged():
    # Endpoints are inside [0, 1] and keep their established behaviour.
    r0 = SF.forecast(prior=0.0)
    assert r0["p"] == 1e-06 and r0["anchor"] == 1e-06
    r1 = SF.forecast(prior=1.0)
    assert r1["p"] == 0.999999 and r1["anchor"] == 0.999999
    assert SF.forecast(prior=0.5) == {"p": 0.5, "anchor": 0.5, "n_components": 0,
                                      "note": "no validated component; prior shipped unchanged"}
    # A valid prior just inside the bound is NOT confused with the
    # clipped value of an out-of-range one.
    assert SF.forecast(prior=0.999999)["anchor"] == 0.999999


def test_demos_bit_identical():
    d1 = SF.forecast(prior=0.55, model_probs=[0.817], model_skills=[-0.007])
    assert d1 == {"p": 0.55, "anchor": 0.55, "n_components": 0,
                  "note": "no validated component; prior shipped unchanged"}, d1
    d2 = SF.forecast(prior=0.10, model_probs=[0.16], model_skills=[0.024])
    assert d2["p"] == 0.10144000000000002, d2
    d3 = SF.forecast(prior=0.50, market_price=0.62,
                     agent_probs=[0.70, 0.66], agent_weights=[0.16, 0.14])
    assert d3["p"] == 0.629249601776721, d3


def test_kill_criterion_not_met():
    # Kill would require no out-of-range prior call to ship a clipped
    # near-certainty; every fixture above raises instead, and the one
    # production caller (cli.py _skill_gate) passes a stored base rate
    # with no upstream range validation — so the kill does not apply.
    for p in (1.5, 2.0, -0.5):
        assert_invalid(lambda p=p: SF.forecast(prior=p))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} PASS")
