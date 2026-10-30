#!/usr/bin/env python3
"""SF-BL-11008 tests: expected_log_wealth_growth must reject a
non-finite p_side instead of returning NaN / infinite growth.

Prereg fixtures (TASK.md): p_side=NaN returned NaN, p_side=+inf
returned +inf and p_side=-inf returned -inf in live repros (clean
(0.70, 0.50, 0.30) = 0.07665250194562398). Done when those beliefs
raise a named ValueError, with finite in-[0,1] growth matching
Meister Eq. III to 1e-12. Scope: finite out-of-range p_side is
SF-BL-9474 (already landed), q_exec/fraction finiteness is
SF-BL-9676, and bool/non-numeric p_side is SF-BL-11043 — all pinned
unchanged below.
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


def raises_11008(p, q, f):
    try:
        ak.expected_log_wealth_growth(p, q, f)
    except ValueError as e:
        return "SF-BL-11008" in str(e) and "p_side" in str(e)
    except Exception:  # noqa: BLE001
        return False
    return False


def main():
    g = ak.expected_log_wealth_growth

    # 1. TASK.md fixtures raise ValueError (SF-BL-11008)
    for label, p in (("NaN", NAN), ("+inf", INF), ("-inf", -INF)):
        check(f"TASK p_side={label} raises", raises_11008(p, 0.50, 0.30))

    # 2. Cross-q / cross-f non-finite sweep
    for p in (NAN, INF, -INF):
        for q, f in ((0.40, 0.10), (0.55, 0.0583), (0.90, 0.50), (0.05, 0.90)):
            check(f"sweep p={p} q={q} f={f} raises", raises_11008(p, q, f))

    # 3. No masking: a non-finite belief raises even when the old guard
    #    would early-return 0.0 (f<=0, q<=0, q>=1) — pre-fix all of
    #    these returned a benign 0.0
    for p, q, f in ((NAN, 0.50, 0.0), (INF, 1.0, 0.30), (-INF, 0.50, 0.0),
                    (NAN, 0.0, 0.30), (INF, 0.50, -0.1), (-INF, 1.5, 0.30)):
        check(f"no-mask p={p} q={q} f={f} raises", raises_11008(p, q, f))

    # 4. Precedence: non-finite p_side raises SF-BL-11008 even when
    #    q_exec / fraction are also invalid (p_side validated first)
    check("precedence p=NaN q=NaN raises 11008", raises_11008(NAN, NAN, 0.30))
    check("precedence p=+inf f=NaN raises 11008", raises_11008(INF, 0.50, NAN))

    # 5. Finite in-[0,1] grid matches closed form to 1e-12
    worst, n = 0.0, 0
    for p in (0.0, 0.01, 0.02, 0.1, 0.25, 0.5, 0.62, 0.7, 0.8, 0.98, 0.99, 1.0):
        for q in (0.05, 0.10, 0.40, 0.50, 0.55, 0.90, 0.99):
            for f in (0.0, 0.01, 0.0583, 0.10, 0.30, 0.50, 0.90):
                worst = max(worst, abs(g(p, q, f) - closed_form(p, q, f)))
                n += 1
    check(f"in-range grid n={n} within 1e-12", worst <= 1e-12, f"max diff {worst:.3e}")

    # 6. Known finite values bit-unchanged
    check("clean (0.70,0.50,0.30)", g(0.70, 0.50, 0.30) == 0.07665250194562398,
          repr(g(0.70, 0.50, 0.30)))
    check("wave2 (0.62,0.55,0.0)==0", g(0.62, 0.55, 0.0) == 0.0)
    check("wave2 (0.62,0.55,0.0583)>0", g(0.62, 0.55, 0.0583) > 0.0)
    check("p=0.0 exact", g(0.0, 0.5, 0.30) == closed_form(0.0, 0.5, 0.30))
    check("p=1.0 exact", g(1.0, 0.5, 0.30) == closed_form(1.0, 0.5, 0.30))
    check("int p=1 closed form", g(1, 0.5, 0.30) == closed_form(1, 0.5, 0.30))
    check("int p=0 closed form", g(0, 0.5, 0.30) == closed_form(0, 0.5, 0.30))

    # 7. Finite guard semantics preserved (0.0 sentinel, not a raise)
    check("guard f=0 -> 0.0", g(0.7, 0.5, 0.0) == 0.0)
    check("guard q=1 -> 0.0", g(0.7, 1.0, 0.3) == 0.0)
    check("guard q=0 -> 0.0", g(0.7, 0.0, 0.3) == 0.0)

    # 8. Deliberate scope — neighbouring prereg surfaces unchanged:
    #    finite out-of-range still raises SF-BL-9474 (not 11008)
    try:
        g(2.0, 0.5, 0.30)
        check("scope: p=2.0 still raises SF-BL-9474", False)
    except ValueError as e:
        check("scope: p=2.0 still raises SF-BL-9474",
              "SF-BL-9474" in str(e) and "SF-BL-11008" not in str(e))
    #    q_exec / fraction non-finite still raises SF-BL-9676
    try:
        g(0.70, NAN, 0.30)
        check("scope: q=NaN still raises SF-BL-9676", False)
    except ValueError as e:
        check("scope: q=NaN still raises SF-BL-9676", "SF-BL-9676" in str(e))
    #    bool / non-numeric p_side is SF-BL-11043's axis.
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
    #    huge int keeps pre-fix OverflowError (finite, unrepresentable)
    try:
        g(10 ** 400, 0.5, 0.30)
        check("scope: 10**400 still OverflowError", False)
    except OverflowError:
        check("scope: 10**400 still OverflowError", True)
    except Exception:  # noqa: BLE001
        check("scope: 10**400 still OverflowError", False)

    print(f"\n{sum(1 for _, ok in PASS if ok)}/{len(PASS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
