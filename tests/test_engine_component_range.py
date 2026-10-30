"""SF-BL-10881 regression tests: forecast() must reject any finite
model_probs / agent_probs component outside [0, 1] with INVALID_INPUT
instead of clipping it to the clip bound (0.999999 / 1e-06) and pooling
it as near-certainty.

Pre-fix repros (pristine engine):
  forecast(prior=0.5, agent_probs=[82.0]) -> p=0.999999, n_components=1
  forecast(prior=0.5, model_probs=[1.5], model_skills=[0.5]) -> p=0.75

Scope: the two component sequences only. prior range is SF-BL-9890,
market_price range is SF-BL-10981, NaN/inf components are SF-BL-9510
(all separate, already covered). Endpoints 0.0 / 1.0 are inside [0, 1]
and keep their established _clip-to-eps behaviour.

Run: python3 -m pytest tests/test_engine_component_range.py -q
(or plain python3 — it is self-checking).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from superforecaster.engine import Superforecaster

SF = Superforecaster()


def assert_invalid(fn, name):
    try:
        out = fn()
    except ValueError as e:
        assert "INVALID_INPUT" in str(e), f"error lacks INVALID_INPUT: {e}"
        assert name in str(e), f"error does not name {name}: {e}"
        return
    raise AssertionError(f"expected INVALID_INPUT ValueError, got return {out!r}")


def test_task_fixtures_rejected():
    # The exact repros from the task filing.
    assert_invalid(lambda: SF.forecast(prior=0.5, agent_probs=[82.0]), "agent_probs")
    assert_invalid(lambda: SF.forecast(prior=0.5, model_probs=[1.5],
                                       model_skills=[0.5]), "model_probs")


def test_just_outside_and_extreme_rejected():
    for bad in (1.000001, -1e-9, 100.0, -100.0, 2.0, -0.5):
        assert_invalid(lambda b=bad: SF.forecast(prior=0.5, agent_probs=[b]), "agent_probs")
        assert_invalid(lambda b=bad: SF.forecast(prior=0.5, model_probs=[b],
                                                 model_skills=[0.5]), "model_probs")


def test_bad_element_anywhere_rejected():
    assert_invalid(lambda: SF.forecast(prior=0.5, agent_probs=[0.7, 82.0]), "agent_probs")
    assert_invalid(lambda: SF.forecast(prior=0.5, model_probs=[0.7, 1.5],
                                       model_skills=[0.3, 0.5]), "model_probs")


def test_rejection_not_rescued_by_drop_or_anchor():
    # Validation precedes the by-value drops and the market anchor:
    # a component that would be dropped (skill <= 0 / weight <= 0 /
    # default skill 0.0) still raises, and a valid market_price does
    # not rescue a corrupt agent component.
    assert_invalid(lambda: SF.forecast(prior=0.5, model_probs=[1.5]), "model_probs")
    assert_invalid(lambda: SF.forecast(prior=0.5, model_probs=[1.5],
                                       model_skills=[-0.007]), "model_probs")
    assert_invalid(lambda: SF.forecast(prior=0.5, agent_probs=[82.0],
                                       agent_weights=[0.0]), "agent_probs")
    assert_invalid(lambda: SF.forecast(prior=0.5, market_price=0.62,
                                       agent_probs=[82.0]), "agent_probs")


def test_boundaries_and_in_range_unchanged():
    # Endpoints are inside [0, 1] and keep established behaviour.
    assert SF.forecast(prior=0.5, agent_probs=[0.0])["p"] == 1.0000000000000008e-06
    assert SF.forecast(prior=0.5, agent_probs=[1.0])["p"] == 0.999999
    assert SF.forecast(prior=0.5, model_probs=[0.0], model_skills=[0.5])["p"] == 0.2500005
    assert SF.forecast(prior=0.5, model_probs=[1.0], model_skills=[0.5])["p"] == 0.7499994999999998
    assert SF.forecast(prior=0.5, agent_probs=[0.7, 0.6])["n_components"] == 2
    assert SF.forecast(prior=0.5, model_probs=[0.7, 0.6])["n_components"] == 0


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
    # Kill would require every production caller to construct
    # components only through the validating AgentForecast path. The
    # sole production caller, cli.py _skill_gate, passes a raw float
    # from markets_results.json straight into model_probs, and no
    # production caller passes agent_probs at all — so the kill does
    # not apply and the guard above is load-bearing.
    for kw in ({"agent_probs": [82.0]},
               {"model_probs": [1.5], "model_skills": [0.5]}):
        assert_invalid(lambda kw=kw: SF.forecast(prior=0.5, **kw),
                       "agent_probs" if "agent_probs" in kw else "model_probs")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} PASS")
