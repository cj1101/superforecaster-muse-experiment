#!/usr/bin/env python3
"""SF-BL-9676 tests: expected_log_wealth_growth must reject a non-finite
or non-numeric q_exec or fraction instead of failing open.

Prereg fixtures (TASK.md): q_exec=NaN returned NaN, fraction=NaN
returned NaN and fraction=+inf returned +inf expected log growth in
live repros (clean (0.70, 0.50, 0.30) = 0.07665250194562398). Done
when those operands raise ValueError (or a fail-closed sentinel no
caller can read as growth) with all finite fixtures unchanged.
Scope: p_side axes are separate preregs (SF-BL-9474 finite range,
SF-BL-11008 non-finite, SF-BL-11043 bool/non-numeric) and finite
fraction >= 1 is SF-BL-9399's axis — all pinned unchanged below.
"""
import math
import sys

sys.path.insert(0, "/home/hatch/workspace/superforecaster")
from superforecaster import alpha_kelly as ak  # noqa: E402

PASS = []
NAN, INF = float("nan"), float("inf")


def check(name, cond, detail=""):
    PASS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f" | {detail}" if detail else ""))
    assert cond, f"{name} {detail}"


def closed_form(p, q, f):
    """Independent Meister Eq. III (mirrors the function's own guard)."""
    if f <= 0.0 or q <= 0.0 or q >= 1.0:
        return 0.0
    return p * math.log(1.0 + f * (1.0 - q) / q) + (1.0 - p) * math.log(max(1e-9, 1.0 - f))


def raises_9676(p, q, f, arg_word=None):
    try:
        ak.expected_log_wealth_growth(p, q, f)
    except ValueError as e:
        msg = str(e)
        return "SF-BL-9676" in msg and (arg_word is None or arg_word in msg)
    except Exception:  # noqa: BLE001
        return False
    return False


def main():
    g = ak.expected_log_wealth_growth

    # 1. TASK.md fixtures raise ValueError (SF-BL-9676)
    check("TASK q_exec=NaN raises", raises_9676(0.70, NAN, 0.30, "q_exec"))
    check("TASK fraction=NaN raises", raises_9676(0.70, 0.50, NAN, "fraction"))
    check("TASK fraction=+inf raises", raises_9676(0.70, 0.50, INF, "fraction"))

    # 2. Full non-finite sweep: every NaN / +/-inf on either axis raises
    for q in (NAN, INF, -INF):
        check(f"q_exec={q} raises", raises_9676(0.70, q, 0.30, "q_exec"))
    for f in (NAN, INF, -INF):
        check(f"fraction={f} raises", raises_9676(0.70, 0.50, f, "fraction"))
    check("both NaN raises", raises_9676(0.70, NAN, NAN))
    check("q=+inf f=+inf raises", raises_9676(0.70, INF, INF))
    check("cross p: (0.0, NaN, 0.1) raises", raises_9676(0.0, NAN, 0.10))
    check("cross p: (1.0, 0.5, +inf) raises", raises_9676(1.0, 0.50, INF))
    check("cross q/f: (0.62, NaN, 0.0583) raises", raises_9676(0.62, NAN, 0.0583))

    # 3. No masking: a non-finite operand raises even when the other
    #    operand would make the old guard early-return 0.0
    for p, q, f in ((0.70, NAN, 0.0), (0.70, NAN, -0.1), (0.70, 0.0, NAN),
                    (0.70, 1.0, NAN), (0.70, 1.5, INF), (0.70, INF, 0.0),
                    (0.70, -INF, 0.30), (0.70, 0.50, -INF)):
        check(f"no-mask p={p} q={q} f={f} raises", raises_9676(p, q, f))

    # 4. Non-numeric / bool operands raise the same ValueError (never a
    #    bare TypeError, never a scored value) — pre-fix '0.5'/None
    #    raised TypeError and fraction=True scored as f=1.0
    for bad in ("0.5", "x", None, True, False, [0.5], (0.5,)):
        check(f"q_exec={bad!r} raises ValueError", raises_9676(0.70, bad, 0.30, "q_exec"))
        check(f"fraction={bad!r} raises ValueError", raises_9676(0.70, 0.50, bad, "fraction"))

    # 5. Finite fixtures unchanged: grid matches closed form to 1e-12
    worst, n = 0.0, 0
    for p in (0.0, 0.01, 0.02, 0.1, 0.25, 0.5, 0.62, 0.7, 0.8, 0.98, 0.99, 1.0):
        for q in (0.05, 0.10, 0.40, 0.50, 0.55, 0.90, 0.99):
            for f in (0.0, 0.01, 0.0583, 0.10, 0.30, 0.50, 0.90, 0.99):
                worst = max(worst, abs(g(p, q, f) - closed_form(p, q, f)))
                n += 1
    check(f"finite grid n={n} within 1e-12", worst <= 1e-12, f"max diff {worst:.3e}")

    # 6. Known finite values bit-unchanged (pre-fix repro values)
    check("clean (0.70,0.50,0.30)", g(0.70, 0.50, 0.30) == 0.07665250194562398,
          repr(g(0.70, 0.50, 0.30)))
    check("wave2 (0.62,0.55,0.0)==0", g(0.62, 0.55, 0.0) == 0.0)
    check("wave2 (0.62,0.55,0.0583)>0", g(0.62, 0.55, 0.0583) > 0.0)
    check("wave2 (0.62,0.55,0.0583) closed form",
          g(0.62, 0.55, 0.0583) == closed_form(0.62, 0.55, 0.0583))

    # 7. Finite guard semantics preserved (0.0 sentinel, not a raise)
    for p, q, f in ((0.70, 0.0, 0.30), (0.70, 1.0, 0.30), (0.70, 1.5, 0.30),
                    (0.70, -0.5, 0.30), (0.70, 0.50, 0.0), (0.70, 0.50, -0.1),
                    (0.70, 0.50, 0), (0.0, 0.0, 0.30), (1, 1, 0.30)):
        check(f"finite guard ({p},{q},{f}) == 0.0", g(p, q, f) == 0.0)

    # 8. Finite fraction >= 1 unchanged (SF-BL-9399 axis, deliberately
    #    untouched): exact pre-fix clamped values
    check("f=1.5 pre-fix value", g(0.70, 0.50, 1.5) == -5.575576238772015,
          repr(g(0.70, 0.50, 1.5)))
    check("f=2.0 pre-fix value", g(0.70, 0.50, 2.0) == -5.447951149016247,
          repr(g(0.70, 0.50, 2.0)))
    check("f=1 (int) closed form", g(0.70, 0.50, 1) == closed_form(0.70, 0.50, 1))

    # 9. Deliberate scope: p_side axes behave exactly as pre-fix
    # SF-BL-11008 has since landed: non-finite p_side now raises its
    # named ValueError (these pins superseded by that landing).
    for _lbl, _p in (("NaN", NAN), ("+inf", INF), ("-inf", -INF)):
        try:
            g(_p, 0.5, 0.30)
            check(f"scope: p {_lbl} now raises SF-BL-11008", False)
        except ValueError as e:
            check(f"scope: p {_lbl} now raises SF-BL-11008", "SF-BL-11008" in str(e))
    # SF-BL-11043 has since landed: bool/non-numeric p_side now raises
    # its named ValueError (these pins superseded by that landing).
    for _lbl, _p in (("True", True), ("'0.7'", "0.7"), ("None", None)):
        try:
            g(_p, 0.5, 0.30)
            check(f"scope: p {_lbl} now raises SF-BL-11043", False)
        except ValueError as e:
            check(f"scope: p {_lbl} now raises SF-BL-11043", "SF-BL-11043" in str(e))
        except Exception:  # noqa: BLE001
            check(f"scope: p {_lbl} now raises SF-BL-11043", False)
    try:
        g(2.0, 0.5, 0.30)
        check("scope: p=2.0 still raises SF-BL-9474", False)
    except ValueError as e:
        check("scope: p=2.0 still raises SF-BL-9474", "SF-BL-9474" in str(e))

    print(f"\n{sum(1 for _, ok in PASS if ok)}/{len(PASS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
