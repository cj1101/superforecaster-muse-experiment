"""SF-BL-9781 regression tests: skill_weight Brier domain [0, 1].

TASK prereg: Done when skill_weight raises ValueError unless both
Brier arguments are finite in [0, 1] and returns prior values on the
valid grid; kill if every skill_weight caller is shown to pass only
scorer-produced Brier values with an upstream [0, 1] assertion and
test. (Kill audited, NOT triggered: no production caller of
skill_weight exists in the package -- only tests call it; forecast()
uses skill_weight_from_skill.)

Pre-fix (verified live 2026-10-23 against pristine snapshot):
  skill_weight(-0.10, 0.25) -> 1.4   (weight above the 1.0 maximum)
  skill_weight(0.10, 1.5)   -> 0.9333...
  skill_weight(1.2, 0.25)   -> 0.0   (accepted silently)

Run: python3 test_engine_skill_weight_brier_domain.py  (exit 0 iff all pass)
No pytest dependency: test_* functions + __main__ runner, fleet convention.
"""
import os
import struct
import sys

sys.path.insert(0, os.path.join(
    os.path.expanduser("~/workspace/superforecaster"), "superforecaster"))

from engine import skill_weight  # noqa: E402

PASS = []
NAN, INF = float("nan"), float("inf")


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))


def assert_invalid(fn):
    try:
        r = fn()
    except ValueError as e:
        assert "INVALID_INPUT" in str(e), f"missing INVALID_INPUT flag: {e}"
        return
    raise AssertionError(f"expected ValueError/INVALID_INPUT, got return {r!r}")


def test_task_headline_repro_raises():
    assert_invalid(lambda: skill_weight(-0.10, 0.25))


def test_out_of_domain_raises():
    for args in [(-0.10, 0.25), (0.10, 1.5), (0.10, -0.2), (1.2, 0.25),
                 (-1e-12, 0.25), (0.10, 1.0 + 1e-12), (-0.5, -0.5),
                 (2.0, 2.0), (-1.0, 0.5), (0.5, 100.0)]:
        assert_invalid(lambda a=args: skill_weight(*a))


def test_nonfinite_and_string_still_raise():
    for bad in (NAN, INF, -INF, "0.10"):
        assert_invalid(lambda b=bad: skill_weight(b, 0.25))
        assert_invalid(lambda b=bad: skill_weight(0.10, b))


def test_valid_grid_bit_identical():
    grid = [0.0, 0.01, 0.09, 0.10, 0.25, 0.5, 0.90, 1.0]
    for m in grid:
        for p in grid:
            exp = 0.0 if p <= 0 else max(0.0, 1.0 - m / p)
            got = skill_weight(m, p)
            assert struct.pack("=d", got) == struct.pack("=d", exp), (m, p, got, exp)


def test_boundaries_and_prior_values():
    assert skill_weight(0.0, 0.25) == 1.0     # perfect model earns the max
    assert skill_weight(0.25, 0.25) == 0.0    # at prior -> no weight
    assert skill_weight(0.09, 0.25) == 0.64   # pre-existing valid value
    assert skill_weight(0.0, 0.0) == 0.0      # prior==0 branch preserved
    assert skill_weight(0.5, 0.0) == 0.0
    assert skill_weight(1.0, 1.0) == 0.0
    assert skill_weight(-0.0, 0.25) == 1.0    # -0.0 is in [0, 1]
    for m in (0.0, 0.25, 0.5, 1.0):
        for p in (0.01, 0.25, 0.5, 1.0):
            assert 0.0 <= skill_weight(m, p) <= 1.0  # weight never exceeds 1 now


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
