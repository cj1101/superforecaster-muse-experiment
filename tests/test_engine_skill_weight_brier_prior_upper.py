"""SF-BL-11055 suite: skill_weight brier_prior upper-range guard.

Tests generated from TASK.md done/kill criteria:
- skill_weight(0.10, 1.5) must raise (not return 0.9333) naming brier_prior
- skill_weight(0.10, 2.0) must raise (not return 0.95) naming brier_prior
- any Brier arg outside [0,1] raises naming that arg
- in-range weights bit-identical to pristine formula
Run: python3 -m pytest test_engine_skill_weight_brier_prior_upper.py -q
or standalone: python3 test_engine_skill_weight_brier_prior_upper.py
"""
import random, struct, sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "workspace/superforecaster"))
from superforecaster.engine import skill_weight

def bits(x): return struct.pack(">d", float(x))
def formula(m, pr): return 0.0 if pr <= 0 else max(0.0, 1.0 - m / pr)

def expect_raise(args, name):
    try:
        v = skill_weight(*args)
    except ValueError as e:
        assert name in str(e), f"{args}: message does not name {name}: {e}"
        assert "INVALID_INPUT" in str(e), f"{args}: missing INVALID_INPUT: {e}"
        return
    raise AssertionError(f"{args}: returned {v!r} instead of raising ValueError naming {name}")

def test_task_headline_repros_raise():
    expect_raise((0.10, 1.5), "brier_prior")
    expect_raise((0.10, 2.0), "brier_prior")
    # kill values must NOT be returned
    for args in [(0.10, 1.5), (0.10, 2.0)]:
        try:
            v = skill_weight(*args); assert False, f"kill: {args} returned {v}"
        except ValueError:
            pass

def test_brier_prior_upper_battery():
    for pr in [1 + 1e-12, 1.0000001, 1.1, 1.5, 2.0, 10.0, 100.0]:
        expect_raise((0.10, pr), "brier_prior")
        expect_raise((0.0, pr), "brier_prior")
        expect_raise((1.0, pr), "brier_prior")

def test_brier_prior_lower_and_model_symmetry():
    for pr in [-1e-12, -0.1, -0.2, -1.0]:
        expect_raise((0.10, pr), "brier_prior")
    for m in [-0.10, -1e-12, 1.0000001, 1.2, 1.5, 2.0]:
        expect_raise((m, 0.25), "brier_model")

def test_valid_grid_bit_identical():
    grid = [0.0, 0.01, 0.09, 0.10, 0.25, 0.5, 0.90, 1.0]
    for m in grid:
        for pr in grid:
            assert bits(skill_weight(m, pr)) == bits(formula(m, pr)), (m, pr)
    rng = random.Random(11055)
    for _ in range(500):
        m, pr = rng.random(), rng.random()
        assert bits(skill_weight(m, pr)) == bits(formula(m, pr)), (m, pr)
    assert skill_weight(0.0, 0.25) == 1.0
    assert skill_weight(0.25, 0.25) == 0.0
    assert skill_weight(0.10, 1.0) == 0.9
    assert skill_weight(0.10, 0.0) == 0.0
    assert skill_weight(1.0, 1.0) == 0.0

def test_nonfinite_still_raise():
    for bad in [float("nan"), float("inf"), float("-inf")]:
        expect_raise((bad, 0.25), "brier_model")
        expect_raise((0.10, bad), "brier_prior")

if __name__ == "__main__":
    for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("test_")}.items()):
        fn(); print(f"PASS {name}")
    print("5/5 PASS")
