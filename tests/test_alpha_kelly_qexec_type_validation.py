#!/usr/bin/env python3
"""SF-BL-11081 tests: expected_log_wealth_growth must raise a named
ValueError for a bool or non-numeric q_exec, fixtured at '0.5', None
and True (p_side=0.70, fraction=0.30), with finite in-range growth
bit-identical.

Kill adjudication under test: the rejection already existed at task
start via SF-BL-9676 (the landed fix for the duplicate SF-BL-10911
surface); this task's patch makes the q_exec TYPE dimension explicit
(message names SF-BL-11081 AND SF-BL-9676). Non-finite q_exec VALUES
keep the SF-BL-9676-only message; p_side type (SF-BL-11043) and
fraction type (SF-BL-11082) behaviour is pinned unchanged.
"""
import importlib.util
import math
import random
import sys

sys.path.insert(0, "/home/hatch/workspace/superforecaster")
from superforecaster import alpha_kelly as ak  # noqa: E402

TASKSTART = ("/home/hatch/workspace/superforecaster_out/backlog/work/"
             "SF-BL-11081/alpha_kelly_taskstart.py")
spec = importlib.util.spec_from_file_location("ak_taskstart", TASKSTART)
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)

PASS = []
NAN, INF = float("nan"), float("inf")


def check(name, cond, detail=""):
    PASS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f" | {detail}" if detail else ""))
    assert cond, f"{name} {detail}"


def closed_form(p, q, f):
    if f <= 0.0 or q <= 0.0 or q >= 1.0:
        return 0.0
    return p * math.log(1.0 + f * (1.0 - q) / q) + (1.0 - p) * math.log(max(1e-9, 1.0 - f))


def raises(p, q, f, *tags):
    try:
        ak.expected_log_wealth_growth(p, q, f)
    except ValueError as e:
        return all(t in str(e) for t in tags)
    except Exception:  # noqa: BLE001
        return False
    return False


def main():
    g = ak.expected_log_wealth_growth

    # 1. TASK.md fixtures raise ValueError naming SF-BL-11081 (+ 9676)
    for q in ("0.5", None, True):
        check(f"TASK q_exec={q!r} raises 11081+9676",
              raises(0.70, q, 0.30, "SF-BL-11081", "SF-BL-9676", "q_exec"))

    # 2. Wider bool / non-numeric matrix on q_exec
    for q in (False, "x", [0.5], (0.5,), {"a": 1}, b"0.5", complex(0.5, 0)):
        check(f"q_exec={q!r} raises 11081", raises(0.70, q, 0.30, "SF-BL-11081"))
    try:
        import decimal
        check("q_exec=Decimal raises 11081",
              raises(0.70, decimal.Decimal("0.5"), 0.30, "SF-BL-11081"))
    except ImportError:
        pass

    # 3. Never a bare TypeError, never a scored/flagged return
    for q in ("0.5", None, True):
        try:
            g(0.70, q, 0.30)
            check(f"q_exec={q!r} no return", False)
        except ValueError:
            check(f"q_exec={q!r} ValueError not TypeError", True)
        except TypeError:
            check(f"q_exec={q!r} ValueError not TypeError", False)

    # 4. No masking: bad-type q_exec raises even when fraction would
    #    make the old guard early-return the benign 0.0
    for q, f in (("0.5", 0.0), (None, -0.1), (True, 0.0), ("0.5", 1.5)):
        check(f"no-mask q={q!r} f={f} raises", raises(0.70, q, f, "SF-BL-11081"))

    # 5. Kill evidence: the TASK-START snapshot already rejected all
    #    three fixtures with a named ValueError (SF-BL-9676)
    for q in ("0.5", None, True):
        try:
            old.expected_log_wealth_growth(0.70, q, 0.30)
            check(f"taskstart q={q!r} already raised", False)
        except ValueError as e:
            check(f"taskstart q={q!r} already raised", "SF-BL-9676" in str(e))
        except Exception:  # noqa: BLE001
            check(f"taskstart q={q!r} already raised", False)

    # 6. Non-finite q_exec VALUES keep the SF-BL-9676 finiteness path
    for q in (NAN, INF, -INF, 10 ** 400):
        check(f"q_exec={q!r} raises 9676 finiteness",
              raises(0.70, q, 0.30, "SF-BL-9676", "finite"))

    # 7. Finite fixtures: closed form to 1e-12 + bit-identity vs the
    #    task-start snapshot over grid and random draws
    worst, n = 0.0, 0
    for p in (0.0, 0.02, 0.1, 0.5, 0.7, 0.98, 1.0):
        for q in (0.05, 0.1, 0.5, 0.9, 0.99):
            for f in (0.0, 0.01, 0.3, 0.9, 0.99):
                worst = max(worst, abs(g(p, q, f) - closed_form(p, q, f)))
                n += 1
    check(f"finite grid n={n} within 1e-12", worst <= 1e-12, f"max {worst:.3e}")
    check("clean (0.70,0.50,0.30) exact",
          g(0.70, 0.50, 0.30) == 0.07665250194562398)
    rng = random.Random(11081)
    mism = 0
    for _ in range(20000):
        p = rng.uniform(-0.5, 1.5) if rng.random() < 0.1 else rng.random()
        if not (0.0 <= p <= 1.0):
            continue  # out-of-range p_side raises SF-BL-9474 identically; skip value compare
        q = rng.choice([rng.random(), rng.uniform(-0.5, 1.5), 0, 1, 2])
        f = rng.choice([rng.random(), rng.uniform(-0.5, 2.0), 0])
        try:
            a = g(p, q, f)
        except ValueError:
            a = "raise"
        try:
            b = old.expected_log_wealth_growth(p, q, f)
        except ValueError:
            b = "raise"
        if a != b:
            mism += 1
    check("20000 random finite draws bit-identical to taskstart", mism == 0,
          f"mismatches={mism}")
    for args in ((0.70, 0.0, 0.30), (0.70, 1.0, 0.30), (0.70, 1.5, 0.30),
                 (0.70, -0.5, 0.30), (0.70, 0.50, 0.0), (0.70, 0.50, -0.1)):
        check(f"finite guard {args} == 0.0", g(*args) == 0.0)

    # 8. Scope pins: neighbouring surfaces unchanged
    check("fraction type still raises (9676/11082 surface)",
          raises(0.70, 0.50, "0.3", "SF-BL-9676"))
    check("fraction True still raises", raises(0.70, 0.50, True, "SF-BL-9676"))
    # SF-BL-11043 has since landed: bool/non-numeric p_side now raises
    # its named ValueError (these pins superseded by that landing).
    check("p_side '0.7' now raises SF-BL-11043",
          raises("0.7", 0.50, 0.30, "SF-BL-11043"))
    check("p_side True now raises SF-BL-11043",
          raises(True, 0.50, 0.30, "SF-BL-11043"))
    check("p_side None now raises SF-BL-11043",
          raises(None, 0.50, 0.30, "SF-BL-11043"))
    try:
        import numpy as np
        check("np.float64 q_exec accepted",
              g(0.70, np.float64(0.5), 0.30) == 0.07665250194562398)
        check("np.float64 NaN q_exec raises 9676",
              raises(0.70, np.float64("nan"), 0.30, "SF-BL-9676"))
    except ImportError:
        pass

    print(f"\n{sum(1 for _, ok in PASS if ok)}/{len(PASS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
