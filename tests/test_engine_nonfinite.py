"""SF-BL-9510 regression tests: engine must never ship a NaN forecast.

TASK prereg: Pass if forecast raises or returns an explicit INVALID_INPUT
flag (never a bare NaN p) for NaN/inf prior, market_price, component probs
or shrink weight, and the three __main__ demo forecasts stay bit-identical;
kill if any NaN-input call still returns a dict whose 'p' is NaN with a
success-style note.

This fix raises ValueError carrying "INVALID_INPUT" for every non-finite
input (prior, market_price, model_probs, model_skills, agent_probs,
agent_weights), for shrink() args (incl. w), and for log_odds_pool /
skill_weight / fit_shrink_weight / skill_weight_from_skill inputs.

Pre-fix (verified live 2026-10-21 before patch):
  forecast(prior=nan) -> {'p': nan, 'note': 'no validated component; prior shipped unchanged'}
  forecast(prior=0.5, agent_probs=[nan, 0.7]) -> n_components=2, p=nan
  forecast(prior=0.5, model_probs=[nan], model_skills=[0.1]) -> p=nan
  forecast(prior=inf) -> p=0.999999 (silently clipped)
  shrink(0.8, 0.5, w=nan) -> nan

Run: python3 test_engine_nonfinite.py  (exit 0 iff all pass)
No pytest dependency: test_* functions + __main__ runner, fleet convention.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.expanduser("~/workspace/superforecaster"), "superforecaster"))

from engine import (  # noqa: E402
    Superforecaster, fit_shrink_weight, log_odds_pool, shrink,
    skill_weight, skill_weight_from_skill,
)

SF = Superforecaster()
NAN, INF = float("nan"), float("inf")
PASS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))


def assert_invalid(fn):
    try:
        r = fn()
    except ValueError as e:
        assert "INVALID_INPUT" in str(e), f"missing INVALID_INPUT flag: {e}"
        return
    # Returning a dict is only acceptable with an explicit flag AND no NaN p.
    assert isinstance(r, dict) and r.get("INVALID_INPUT"), \
        f"expected ValueError/INVALID_INPUT, got return {r!r}"
    assert not (isinstance(r.get("p"), float) and math.isnan(r["p"])), r


def test_forecast_prior_nonfinite():
    for bad in (NAN, INF, -INF):
        assert_invalid(lambda b=bad: SF.forecast(prior=b))


def test_forecast_market_price_nonfinite():
    for bad in (NAN, INF, -INF):
        assert_invalid(lambda b=bad: SF.forecast(prior=0.5, market_price=b))


def test_forecast_component_probs_nonfinite():
    for bad in (NAN, INF, -INF):
        assert_invalid(lambda b=bad: SF.forecast(prior=0.5, agent_probs=[b, 0.7]))
        assert_invalid(lambda b=bad: SF.forecast(prior=0.5, agent_probs=[0.7, b]))
        assert_invalid(lambda b=bad: SF.forecast(
            prior=0.5, model_probs=[b], model_skills=[0.1]))
        assert_invalid(lambda b=bad: SF.forecast(
            prior=0.5, market_price=0.62, agent_probs=[b]))


def test_forecast_skills_and_weights_nonfinite():
    for bad in (NAN, INF, -INF):
        assert_invalid(lambda b=bad: SF.forecast(
            prior=0.5, model_probs=[0.7], model_skills=[b]))
        assert_invalid(lambda b=bad: SF.forecast(
            prior=0.5, agent_probs=[0.7], agent_weights=[b]))


def test_shrink_nonfinite():
    for bad in (NAN, INF, -INF):
        assert_invalid(lambda b=bad: shrink(0.8, 0.5, w=b))
        assert_invalid(lambda b=bad: shrink(b, 0.5, w=0.5))
        assert_invalid(lambda b=bad: shrink(0.8, b, w=0.5))
    assert shrink(0.8, 0.5, w=0.5) == 0.65


def test_helpers_nonfinite():
    for bad in (NAN, INF):
        assert_invalid(lambda b=bad: log_odds_pool([b, 0.7]))
        assert_invalid(lambda b=bad: log_odds_pool([0.6, 0.7], weights=[b, 1.0]))
        assert_invalid(lambda b=bad: log_odds_pool([0.6, 0.7], extremization=b))
        assert_invalid(lambda b=bad: skill_weight(b, 0.10))
        assert_invalid(lambda b=bad: skill_weight(0.09, b))
        assert_invalid(lambda b=bad: skill_weight_from_skill(b))
        assert_invalid(lambda b=bad: fit_shrink_weight([0, 1], [b, 0.5], [0.5, 0.5]))
        assert_invalid(lambda b=bad: Superforecaster(extremization=b))


def test_no_nan_p_ever_returned():
    # Kill criterion probe: sweep finite-valid calls; p is always finite.
    for prior in (0.01, 0.5, 0.99):
        r = SF.forecast(prior=prior, agent_probs=[0.7, 0.6],
                        model_probs=[0.65], model_skills=[0.05])
        assert math.isfinite(r["p"]) and 0.0 < r["p"] < 1.0, r


def test_demos_bit_identical():
    d1 = SF.forecast(prior=0.55, model_probs=[0.817], model_skills=[-0.007])
    assert d1 == {"p": 0.55, "anchor": 0.55, "n_components": 0,
                  "note": "no validated component; prior shipped unchanged"}, d1
    d2 = SF.forecast(prior=0.10, model_probs=[0.16], model_skills=[0.024])
    assert d2 == {"p": 0.10144000000000002, "anchor": 0.1,
                  "pooled_components": 0.10144000000000002,
                  "n_components": 1, "extremization": 1.0}, d2
    d3 = SF.forecast(prior=0.50, market_price=0.62,
                     agent_probs=[0.70, 0.66], agent_weights=[0.16, 0.14])
    assert d3 == {"p": 0.629249601776721, "anchor": 0.62,
                  "pooled_components": 0.629249601776721,
                  "n_components": 2, "extremization": 1.0}, d3


def test_valid_calls_unchanged():
    # SF-BL-8868 behaviours preserved.
    assert SF.forecast(prior=0.5, agent_probs=[0.7, 0.6])["n_components"] == 2
    assert SF.forecast(prior=0.5, model_probs=[0.7, 0.6])["n_components"] == 0
    try:
        SF.forecast(prior=0.5, model_probs=[0.7, 0.6], model_skills=[0.1])
        raise SystemExit("length mismatch no longer raises")
    except ValueError as e:
        assert "length mismatch" in str(e)
    assert log_odds_pool([0.6, 0.7]) == 0.6516685226452116


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
