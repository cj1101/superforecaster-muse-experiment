#!/usr/bin/env python3
"""SF-BL-9474 tests: expected_log_wealth_growth must reject a finite
p_side outside [0, 1] instead of scoring an impossible belief as growth.

Prereg fixtures (TASK.md): p_side in {-0.50, 1.50, 2.0} raises or flags
invalid; growth on p_side in [0, 1] matches closed-form Meister Eq. III
to 1e-12. TASK.md numbers reproduce at q_exec=0.5: p_side=2.0, f=0.30
pre-fix returned +0.8814034728737146; p_side=1.50, f=0.10 pre-fix
returned +0.19564552753540054.
"""
import math
import sys

sys.path.insert(0, "/home/hatch/workspace/superforecaster")
from superforecaster import alpha_kelly as ak  # noqa: E402

PASS = []


def check(name, cond, detail=""):
    PASS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f" | {detail}" if detail else ""))
    assert cond, f"{name} {detail}"


def closed_form(p, q, f):
    """Independent Meister Eq. III (mirrors the function's own guard)."""
    if f <= 0.0 or q <= 0.0 or q >= 1.0:
        return 0.0
    return p * math.log(1.0 + f * (1.0 - q) / q) + (1.0 - p) * math.log(max(1e-9, 1.0 - f))


def raises_value_error(p, q, f):
    try:
        ak.expected_log_wealth_growth(p, q, f)
    except ValueError as e:
        return "SF-BL-9474" in str(e)
    except Exception:  # noqa: BLE001
        return False
    return False


def main():
    g = ak.expected_log_wealth_growth

    # 1. Prereg fixtures raise ValueError (both TASK.md fraction values)
    for p in (-0.50, 1.50, 2.0):
        for f in (0.10, 0.30):
            check(f"prereg p={p} f={f} raises", raises_value_error(p, 0.5, f))

    # 2. Cross-q out-of-range raises; integer beliefs too
    for p, q, f in ((2.0, 0.40, 0.30), (-0.50, 0.55, 0.20), (2, 0.5, 0.3), (-1, 0.5, 0.3)):
        check(f"cross p={p} q={q} raises", raises_value_error(p, q, f))

    # 3. Sweep: every finite out-of-range value raises (just outside counts)
    sweep = [-100.0, -2.0, -1.0, -0.5, -0.01, -1e-9, 1.0 + 1e-9, 1.01, 1.5, 2.0, 10.0, 100.0]
    check("sweep all raise", all(raises_value_error(p, 0.5, 0.30) for p in sweep),
          f"{len(sweep)} values")

    # 4. No masking: invalid belief raises even when the old guard would
    #    early-return 0.0 (f<=0, q<=0, q>=1)
    for p, q, f in ((2.0, 0.5, 0.0), (2.0, 1.0, 0.30), (-0.50, 0.0, 0.30), (1.5, 0.5, -0.1)):
        check(f"no-mask p={p} q={q} f={f} raises", raises_value_error(p, q, f))

    # 5. In-range grid matches closed form to 1e-12 (expected bit-close)
    worst, n = 0.0, 0
    for p in (0.0, 0.01, 0.02, 0.1, 0.25, 0.5, 0.62, 0.7, 0.8, 0.98, 0.99, 1.0):
        for q in (0.05, 0.10, 0.40, 0.50, 0.55, 0.90, 0.99):
            for f in (0.0, 0.01, 0.0583, 0.10, 0.30, 0.50, 0.90):
                worst = max(worst, abs(g(p, q, f) - closed_form(p, q, f)))
                n += 1
    check(f"in-range grid n={n} within 1e-12", worst <= 1e-12, f"max diff {worst:.3e}")

    # 6. Endpoints are valid beliefs, exact values
    check("p=0.0 exact", g(0.0, 0.5, 0.30) == closed_form(0.0, 0.5, 0.30))
    check("p=1.0 exact", g(1.0, 0.5, 0.30) == closed_form(1.0, 0.5, 0.30))

    # 7. Known pre-fix in-range values unchanged
    check("clean (0.70,0.50,0.30)", g(0.70, 0.50, 0.30) == 0.07665250194562398,
          repr(g(0.70, 0.50, 0.30)))
    check("wave2 (0.62,0.55,0.0)==0", g(0.62, 0.55, 0.0) == 0.0)
    check("wave2 (0.62,0.55,0.0583)>0", g(0.62, 0.55, 0.0583) > 0.0)

    # 8. Deliberate scope (separate preregs, behaviour preserved):
    #    non-finite p_side is SF-BL-11008's axis; bool/non-numeric is
    #    SF-BL-11043's axis.
    # SF-BL-11008 has since landed: non-finite p_side now raises its
    # named ValueError (these pins superseded by that landing).
    for _lbl, _p in (("NaN", float("nan")), ("+inf", float("inf")), ("-inf", float("-inf"))):
        try:
            g(_p, 0.5, 0.30)
            check(f"scope: {_lbl} now raises SF-BL-11008", False)
        except ValueError as e:
            check(f"scope: {_lbl} now raises SF-BL-11008", "SF-BL-11008" in str(e))
    # SF-BL-11043 has since landed: bool/non-numeric p_side now raises
    # its named ValueError (these pins superseded by that landing).
    for _lbl, _p in (("True", True), ("'0.7'", "0.7"), ("None", None)):
        try:
            g(_p, 0.5, 0.30)
            check(f"scope: {_lbl} now raises SF-BL-11043", False)
        except ValueError as e:
            check(f"scope: {_lbl} now raises SF-BL-11043", "SF-BL-11043" in str(e))
        except Exception:  # noqa: BLE001
            check(f"scope: {_lbl} now raises SF-BL-11043", False)

    # 9. fraction / q_exec guard untouched for valid beliefs
    check("guard f=0 -> 0.0", g(0.7, 0.5, 0.0) == 0.0)
    check("guard q=1 -> 0.0", g(0.7, 1.0, 0.3) == 0.0)
    check("guard q=0 -> 0.0", g(0.7, 0.0, 0.3) == 0.0)

    print(f"\n{sum(1 for _, ok in PASS if ok)}/{len(PASS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
