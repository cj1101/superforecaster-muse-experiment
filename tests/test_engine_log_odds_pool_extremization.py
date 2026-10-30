"""SF-BL-8900: engine.log_odds_pool must reject extremization outside (0, 1.40].

Pre-fix: log_odds_pool([0.8,0.8], extremization=-1.0) returned 0.2 (inverted),
extremization=0.0 returned 0.5 (collapsed), extremization=2.0 returned 0.941
(over the 1.40 cap) — while Superforecaster.__init__ guarded [1.00, 1.40]
only for the class path and the function is re-exported publicly.
Standalone suite: python3 test_engine_log_odds_pool_extremization.py
"""
import math, os, struct, sys

sys.path.insert(0, os.path.join(
    os.path.expanduser("~/workspace/superforecaster"), "superforecaster"))
from engine import log_odds_pool, Superforecaster  # noqa: E402

PASS = FAIL = 0

def check(name, fn):
    global PASS, FAIL
    try:
        fn(); PASS += 1; print(f"PASS {name}")
    except AssertionError as e:
        FAIL += 1; print(f"FAIL {name}: {e}")

def bits(x):
    return struct.pack("<d", float(x))

def must_raise(f, probs_list):
    for probs, w in probs_list:
        try:
            v = log_odds_pool(probs, weights=w, extremization=f)
        except ValueError as e:
            assert type(e) is ValueError, f"wrong type {type(e)}"
            assert "INVALID_INPUT" in str(e), f"no INVALID_INPUT flag: {e}"
            assert "extremization" in str(e), f"arg not named: {e}"
            continue
        raise AssertionError(f"extremization={f} probs={probs} RETURNED {v}")

PROBS = [([0.8, 0.8], None), ([0.6, 0.7], None), ([0.6, 0.7], [2.0, 1.0]),
         ([0.1, 0.5, 0.9], [3.0, 1.0, 0.5]), ([0.5], None), ([0.02, 0.98], None)]
REJECT = [-100.0, -10.0, -2.0, -1.0, -0.5, -1e-9, 0.0,
          1.4000001, 1.41, 1.5, 2.0, 3.0, 10.0, 100.0, 1e6]
ALLOW = [1e-9, 0.1, 0.25, 0.5, 0.99, 1.0, 1.2, 1.3999999, 1.40]

def t_task_repros():
    must_raise(-1.0, [([0.8, 0.8], None)])
    must_raise(0.0, [([0.8, 0.8], None)])

def t_fuzz_reject():
    for f in REJECT:
        must_raise(f, PROBS)

def t_fuzz_allow_unchanged():
    # Known pristine values (computed pre-fix; arithmetic untouched in range)
    assert bits(log_odds_pool([0.8, 0.8], extremization=0.5)) == bits(0.6666666666666666)
    assert bits(log_odds_pool([0.8, 0.8], extremization=1.0)) == bits(0.8)
    assert bits(log_odds_pool([0.8, 0.8], extremization=1.40)) == bits(0.8744413346659732)
    assert bits(log_odds_pool([0.6, 0.7])) == bits(0.6516685226452116)
    for f in ALLOW:
        for probs, w in PROBS:
            v = log_odds_pool(probs, weights=w, extremization=f)
            assert 0.0 < v < 1.0 and math.isfinite(v), (f, probs, v)

def t_boundary():
    assert log_odds_pool([0.8, 0.8], extremization=1.40) > 0.8  # cap inclusive
    must_raise(1.40 + 1e-12, PROBS)
    must_raise(-0.0, PROBS)  # -0.0 is 0: collapses, must reject

def t_nonfinite_and_string_still_named():
    for f in (float("nan"), float("inf"), float("-inf"), "1.2"):
        try:
            log_odds_pool([0.6, 0.7], extremization=f)
        except ValueError as e:
            assert "INVALID_INPUT" in str(e), str(e); continue
        raise AssertionError(f"{f!r} did not raise")

def t_constructor_and_forecast_unchanged():
    for f, ok in ((1.0, True), (1.40, True), (0.5, False), (1.41, False), (-1.0, False)):
        try:
            Superforecaster(extremization=f); assert ok, f"ctor({f}) should raise"
        except ValueError:
            assert not ok, f"ctor({f}) should pass"
    sf = Superforecaster(extremization=1.40)
    out = sf.forecast(prior=0.50, market_price=0.62, agent_probs=[0.70, 0.66],
                      agent_weights=[0.16, 0.14])
    assert out["n_components"] == 2 and 0.0 < out["p"] < 1.0, out

if __name__ == "__main__":
    check("task repros -1.0/0.0 raise", t_task_repros)
    check("fuzz grid: out-of-(0,1.40] raises INVALID_INPUT naming extremization", t_fuzz_reject)
    check("fuzz grid: allowed factors return, pristine values bit-identical", t_fuzz_allow_unchanged)
    check("boundary: 1.40 inclusive, 1.40+eps and -0.0 rejected", t_boundary)
    check("NaN/inf/string extremization still raises named INVALID_INPUT", t_nonfinite_and_string_still_named)
    check("constructor guard + forecast() unchanged", t_constructor_and_forecast_unchanged)
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
