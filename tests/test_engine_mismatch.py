"""SF-BL-8868 regression tests: Superforecaster.forecast must never
silently truncate mismatched component lists.

Pre-fix, zip(model_probs, model_skills) / zip(agent_probs, agent_weights)
dropped every component past the shorter list (3 probs + 1 skill pooled
as n_components=1). Now any length mismatch raises ValueError; matched
and defaulted (None) calls are unchanged.

Run:  python3 tests/test_engine_mismatch.py     (or: python3 -m pytest tests/test_engine_mismatch.py)
No pytest dependency: every check is a test_* function using assert, and
__main__ runs them all.
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'superforecaster'))

import numpy as np  # noqa: E402

from engine import Superforecaster, log_odds_pool  # noqa: E402

SF = Superforecaster()

MISMATCH_KWARGS = [
    dict(model_probs=[0.7, 0.6, 0.65], model_skills=[0.1]),      # task repro
    dict(agent_probs=[0.7, 0.6], agent_weights=[1.0]),           # task repro
    dict(model_probs=[0.7], model_skills=[0.1, 0.2, 0.3]),       # reverse
    dict(agent_probs=[0.7], agent_weights=[1.0, 2.0]),           # reverse
    dict(model_probs=[0.7, 0.6], model_skills=[]),
    dict(agent_probs=[], agent_weights=[1.0]),
    dict(model_skills=[0.1]),                                    # orphan
    dict(agent_weights=[1.0]),                                   # orphan
]


def _assert_mismatch(fn):
    try:
        fn()
    except ValueError as e:
        assert "length mismatch" in str(e), f"unclear error: {e}"
        return
    raise AssertionError("expected ValueError on length mismatch, none raised")


def test_forecast_mismatch_raises():
    for kw in MISMATCH_KWARGS:
        _assert_mismatch(lambda kw=kw: SF.forecast(prior=0.5, **kw))


def test_forecast_mismatch_raises_with_market_anchor():
    _assert_mismatch(lambda: SF.forecast(
        prior=0.5, market_price=0.62,
        agent_probs=[0.7, 0.66, 0.6], agent_weights=[0.16, 0.14]))


def test_forecast_mismatch_raises_numpy():
    _assert_mismatch(lambda: SF.forecast(
        prior=0.5, model_probs=np.array([0.7, 0.6]),
        model_skills=np.array([0.1])))


def test_engine_log_odds_pool_mismatch_raises():
    _assert_mismatch(lambda: log_odds_pool([0.7, 0.6, 0.5], [1.0]))
    _assert_mismatch(lambda: log_odds_pool([0.7], [1.0, 2.0]))


def test_matched_and_default_calls_unchanged():
    # Negative skill is dropped BY VALUE (by design), not by truncation.
    r = SF.forecast(prior=0.55, model_probs=[0.817], model_skills=[-0.007])
    assert r["n_components"] == 0 and r["p"] == 0.55, r
    r = SF.forecast(prior=0.50, market_price=0.62, agent_probs=[0.70, 0.66],
                    agent_weights=[0.16, 0.14])
    assert r["n_components"] == 2, r
    # Omitted skills/weights keep documented defaults.
    assert SF.forecast(prior=0.5, agent_probs=[0.7, 0.6])["n_components"] == 2
    assert SF.forecast(prior=0.5, model_probs=[0.7, 0.6])["n_components"] == 0
    assert SF.forecast(prior=0.5, model_probs=[], model_skills=[])["n_components"] == 0


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for fn in ALL_TESTS:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(ALL_TESTS)} unit tests passed.")
