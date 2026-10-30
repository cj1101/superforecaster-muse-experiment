"""SF-BL-10981 regression tests: engine Superforecaster.forecast must
reject any finite market_price outside [0, 1] with INVALID_INPUT
instead of clipping it to the clip bound (0.999999 / 1e-06) and using
it as the dominant anchor (finding 2, anchor weight >= 0.85).

Pre-fix repro (pristine engine):
  forecast(prior=0.50, market_price=1.5, agent_probs=[0.70, 0.66])
    -> {'p': 0.9520488874243722, 'anchor': 0.999999, ...}  (TASK: p 0.9522)
  forecast(prior=0.50, market_price=-0.5)
    -> {'p': 1e-06, 'note': 'no validated component; prior shipped unchanged'}

Scope: the market_price argument only. prior is SF-BL-9890,
model/agent components are SF-BL-10881, NaN/inf market_price is
SF-BL-9510, bool is SF-BL-10921, string is SF-BL-11054 (all separate,
already covered). Endpoints 0.0 / 1.0 are inside [0, 1] and keep
their established _clip-to-eps behaviour.

Run: python3 -m pytest tests/test_engine_market_price_range.py -q
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
        assert "market_price" in str(e), f"error does not name market_price: {e}"
        return
    raise AssertionError(f"expected INVALID_INPUT ValueError, got return {out!r}")


def test_task_fixtures_rejected():
    # The exact repros from the task filing must raise, never return.
    # Kill check: 1.5 must NOT return anchor 0.999999; -0.5 must NOT ship p 1e-06.
    assert_invalid(lambda: SF.forecast(prior=0.50, market_price=1.5, agent_probs=[0.70, 0.66]))
    assert_invalid(lambda: SF.forecast(prior=0.50, market_price=-0.5))


def test_just_outside_and_extreme_rejected():
    for bad in (1.000001, -1e-9, 100.0, -100.0, 2.0, -0.5, 82.0):
        assert_invalid(lambda b=bad: SF.forecast(prior=0.5, market_price=b))
        assert_invalid(lambda b=bad: SF.forecast(prior=0.5, market_price=b, agent_probs=[0.70, 0.66]))


def test_rejection_on_every_path():
    # No-components, agent, and model paths all raise; a valid prior
    # or valid components do not rescue a corrupt market price.
    assert_invalid(lambda: SF.forecast(prior=0.5, market_price=1.5))
    assert_invalid(lambda: SF.forecast(prior=0.5, market_price=1.5, model_probs=[0.16], model_skills=[0.024]))
    assert_invalid(lambda: SF.forecast(prior=0.5, market_price=-0.5, agent_probs=[0.70, 0.66], agent_weights=[0.16, 0.14]))


def test_boundaries_and_in_range_unchanged():
    # Endpoints are inside [0, 1] and keep established _clip behaviour,
    # identical to pristine values (see repro_before.txt controls).
    r0 = SF.forecast(prior=0.50, market_price=0.0)
    assert r0 == {"p": 1e-06, "anchor": 1e-06, "n_components": 0,
                  "note": "no validated component; prior shipped unchanged"}, r0
    r1 = SF.forecast(prior=0.50, market_price=1.0)
    assert r1 == {"p": 0.999999, "anchor": 0.999999, "n_components": 0,
                  "note": "no validated component; prior shipped unchanged"}, r1
    ra = SF.forecast(prior=0.50, market_price=0.0, agent_probs=[0.70, 0.66])
    assert ra["p"] == 0.10205058742437227 and ra["anchor"] == 1e-06, ra
    rb = SF.forecast(prior=0.50, market_price=1.0, agent_probs=[0.70, 0.66])
    assert rb["p"] == 0.9520488874243722 and rb["anchor"] == 0.999999, rb
    # market_price=None keeps the prior path untouched.
    assert SF.forecast(prior=0.5) == {"p": 0.5, "anchor": 0.5, "n_components": 0,
                                      "note": "no validated component; prior shipped unchanged"}


def test_prior_guards_preserved():
    # Prior range (9890), component range (10881), nonfinite (9510),
    # bool (10921) and string (11054) screens still raise INVALID_INPUT.
    for fn in (lambda: SF.forecast(prior=1.5),
               lambda: SF.forecast(prior=0.5, agent_probs=[82.0]),
               lambda: SF.forecast(prior=0.5, market_price=float("nan")),
               lambda: SF.forecast(prior=0.5, market_price=float("inf")),
               lambda: SF.forecast(prior=0.5, market_price=True),
               lambda: SF.forecast(prior=0.5, market_price="0.62")):
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
