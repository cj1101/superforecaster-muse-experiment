#!/usr/bin/env python3
"""SF-BL-11043 tests: expected_log_wealth_growth must raise a named
ValueError for a bool or non-numeric p_side, fixtured at '0.7', None
and True (q_exec=0.50, fraction=0.30), with finite in-domain growth
bit-identical.

Pre-fix (task-start snapshot, verified in §5): p_side='0.7'/None
crashed with a bare TypeError in the growth sum, p_side=True returned
0.26236426446749106 — the growth of the certain belief p=1.0, against
0.07665250194562398 for the clean p_side=0.70 fixture — p_side=False
returned the p=0.0 growth, and a bad type was masked as 0.0 whenever
the guard early-returned. Scope: TYPE only — non-finite p_side keeps
SF-BL-11008, finite out-of-range keeps SF-BL-9474, a huge int keeps
the pre-fix OverflowError, and q_exec/fraction keep SF-BL-9676/11081.
"""
import importlib.util
import math
import random
import sys

sys.path.insert(0, "/home/hatch/workspace/superforecaster")
from superforecaster import alpha_kelly as ak  # noqa: E402

TASKSTART = ("/home/hatch/workspace/superforecaster_out/backlog/work/"
             "SF-BL-11043/alpha_kelly_taskstart.py")
spec = importlib.util.spec_from_file_location("ak_taskstart_11043", TASKSTART)
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

    # 1. TASK.md fixtures raise ValueError naming SF-BL-11043 + p_side + type
    for p, tname in (("0.7", "str"), (None, "NoneType"), (True, "bool")):
        check(f"TASK p_side={p!r} raises 11043",
              raises(p, 0.50, 0.30, "SF-BL-11043", "p_side", tname))

    # 2. Wider bool / non-numeric matrix on p_side
    for p in (False, "x", [0.7], (0.7,), {"a": 1}, b"0.7", complex(0.7, 0)):
        check(f"p_side={p!r} raises 11043", raises(p, 0.50, 0.30, "SF-BL-11043"))
    try:
        import decimal
        check("p_side=Decimal raises 11043",
              raises(decimal.Decimal("0.7"), 0.50, 0.30, "SF-BL-11043"))
    except ImportError:
        pass

    # 3. Never a bare TypeError, never a scored/flagged return
    for p in ("0.7", None, True, False):
        try:
            g(p, 0.50, 0.30)
            check(f"p_side={p!r} no return", False)
        except ValueError:
            check(f"p_side={p!r} ValueError not TypeError", True)
        except TypeError:
            check(f"p_side={p!r} ValueError not TypeError", False)

    # 4. No masking: bad-type p_side raises even when the guard would
    #    early-return the benign 0.0 (f<=0, q<=0, q>=1)
    for p, q, f in (("0.7", 0.50, 0.0), (None, 1.0, 0.30), (True, 0.50, 0.0),
                    ("0.7", 0.0, 0.30), (None, 0.50, -0.1)):
        check(f"no-mask p={p!r} q={q} f={f} raises", raises(p, q, f, "SF-BL-11043"))

    # 5. Bug evidence: the TASK-START snapshot exhibits the filing
    #    behaviour verbatim (this task's kill was NOT met — contrast
    #    SF-BL-11081, whose snapshot already raised)
    og = old.expected_log_wealth_growth
    check("taskstart True == certain belief 0.2624",
          og(True, 0.50, 0.30) == og(1.0, 0.50, 0.30) == 0.26236426446749106)
    check("taskstart False == p=0.0 growth",
          og(False, 0.50, 0.30) == og(0.0, 0.50, 0.30) == -0.35667494393873245)
    for p in ("0.7", None):
        try:
            og(p, 0.50, 0.30)
            check(f"taskstart p={p!r} bare TypeError", False)
        except TypeError:
            check(f"taskstart p={p!r} bare TypeError", True)
        except Exception:  # noqa: BLE001
            check(f"taskstart p={p!r} bare TypeError", False)
    check("taskstart '0.7' masked at f=0.0", og("0.7", 0.50, 0.0) == 0.0)

    # 6. Precedence: bad-type p_side raises SF-BL-11043 even when
    #    q_exec / fraction are also invalid (p_side validated first)
    check("precedence p='0.7' q=NaN raises 11043", raises("0.7", NAN, 0.30, "SF-BL-11043"))
    check("precedence p=None f='0.3' raises 11043", raises(None, 0.50, "0.3", "SF-BL-11043"))
    check("precedence p=True q='0.5' raises 11043", raises(True, "0.5", 0.30, "SF-BL-11043"))

    # 7. Neighbouring axes unchanged (type scope only)
    for p in (NAN, INF, -INF):
        check(f"p_side={p!r} still raises 11008 not 11043",
              raises(p, 0.50, 0.30, "SF-BL-11008") and not raises(p, 0.50, 0.30, "SF-BL-11043", "not a real number"))
    for p in (2.0, -0.5, 1.5):
        check(f"p_side={p} still raises 9474", raises(p, 0.50, 0.30, "SF-BL-9474"))
    try:
        g(10 ** 400, 0.50, 0.30)
        check("huge int still OverflowError", False)
    except OverflowError:
        check("huge int still OverflowError", True)
    except Exception:  # noqa: BLE001
        check("huge int still OverflowError", False)
    check("q_exec type still raises 11081/9676",
          raises(0.70, "0.5", 0.30, "SF-BL-11081", "SF-BL-9676"))
    check("fraction type still raises 9676", raises(0.70, 0.50, "0.3", "SF-BL-9676"))
    check("fraction True still raises 9676", raises(0.70, 0.50, True, "SF-BL-9676"))

    # 8. Finite fixtures: closed form to 1e-12 + bit-identity vs the
    #    task-start snapshot over grid and random draws
    worst, n = 0.0, 0
    for p in (0.0, 0.01, 0.02, 0.1, 0.25, 0.5, 0.62, 0.7, 0.8, 0.98, 0.99, 1.0):
        for q in (0.05, 0.10, 0.40, 0.50, 0.55, 0.90, 0.99):
            for f in (0.0, 0.01, 0.0583, 0.10, 0.30, 0.50, 0.90):
                worst = max(worst, abs(g(p, q, f) - closed_form(p, q, f)))
                n += 1
    check(f"finite grid n={n} within 1e-12", worst <= 1e-12, f"max {worst:.3e}")
    check("clean (0.70,0.50,0.30) exact", g(0.70, 0.50, 0.30) == 0.07665250194562398)
    check("int p=1 exact", g(1, 0.50, 0.30) == 0.26236426446749106)
    check("int p=0 exact", g(0, 0.50, 0.30) == -0.35667494393873245)
    check("wave2 (0.62,0.55,0.0)==0", g(0.62, 0.55, 0.0) == 0.0)
    check("wave2 (0.62,0.55,0.0583)>0", g(0.62, 0.55, 0.0583) > 0.0)
    rng = random.Random(11043)
    mism = 0
    draws = 0
    for _ in range(20000):
        p = rng.random()
        q = rng.choice([rng.random(), rng.uniform(-0.5, 1.5), 0, 1, 2])
        f = rng.choice([rng.random(), rng.uniform(-0.5, 2.0), 0])
        draws += 1
        try:
            a = g(p, q, f)
        except ValueError:
            a = "raise"
        try:
            b = og(p, q, f)
        except ValueError:
            b = "raise"
        if a != b:
            mism += 1
    check(f"{draws} random finite draws bit-identical to taskstart", mism == 0,
          f"mismatches={mism}")
    for args in ((0.70, 0.0, 0.30), (0.70, 1.0, 0.30), (0.70, 0.50, 0.0),
                 (0.70, 0.50, -0.1)):
        check(f"finite guard {args} == 0.0", g(*args) == 0.0)

    # 9. Numpy discipline: float64 passes (float subclass), bool_ raises
    try:
        import numpy as np
        check("np.float64 p_side accepted",
              g(np.float64(0.70), 0.50, 0.30) == 0.07665250194562398)
        check("np.float64 NaN p_side raises 11008",
              raises(np.float64("nan"), 0.50, 0.30, "SF-BL-11008"))
        check("np.bool_ p_side raises 11043",
              raises(np.bool_(True), 0.50, 0.30, "SF-BL-11043"))
    except ImportError:
        pass

    print(f"\n{sum(1 for _, ok in PASS if ok)}/{len(PASS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
