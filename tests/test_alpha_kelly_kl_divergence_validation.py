#!/usr/bin/env python3
"""SF-BL-9507 tests: kl_divergence_binary must reject any p or q that is
not a finite probability in [0, 1] instead of fabricating KL nats.

Prereg fixtures (TASK.md): (NaN, 0.50) returned 0.693, (0.50, NaN)
returned 6.215, (1.50, 0.50) returned 0.693 and (True, 0.50) returned
0.693 pre-fix; post-fix all four raise ValueError and return no nats.
In-domain values (boundaries 0/1 included) must be bit-identical to
pre-fix and match an independent reference D_KL to 1e-12.

Run: python3 -m pytest test_kl_divergence_validation.py -q  (or plain python3)
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


def raises_ve(p, q, arg=None):
    """True iff kl(p, q) raises ValueError naming SF-BL-9507 (and `arg`)."""
    try:
        ak.kl_divergence_binary(p, q)
    except ValueError as e:
        msg = str(e)
        return "SF-BL-9507" in msg and (arg is None or f" {arg} " in msg)
    except Exception:  # noqa: BLE001
        return False
    return False


def reference_kl(p, q):
    """Independent D_KL with the function's documented interior clip."""
    pc = max(1e-6, min(1.0 - 1e-6, p))
    qc = max(1e-6, min(1.0 - 1e-6, q))
    return pc * math.log(pc / qc) + (1.0 - pc) * math.log((1.0 - pc) / (1.0 - qc))


def main():
    kl = ak.kl_divergence_binary

    # 1. TASK.md four repros raise ValueError, no nats returned
    check("TASK (NaN, 0.50) raises p", raises_ve(float("nan"), 0.50, "p"))
    check("TASK (0.50, NaN) raises q", raises_ve(0.50, float("nan"), "q"))
    check("TASK (1.50, 0.50) raises p", raises_ve(1.50, 0.50, "p"))
    check("TASK (True, 0.50) raises p", raises_ve(True, 0.50, "p"))

    # 2. Non-finite sweep, both arguments
    for bad in (float("nan"), float("inf"), float("-inf")):
        check(f"non-finite p={bad!r} raises", raises_ve(bad, 0.5, "p"))
        check(f"non-finite q={bad!r} raises", raises_ve(0.5, bad, "q"))

    # 3. Bool sweep, both arguments (bool is an int subclass — must not pass)
    for bad in (True, False):
        check(f"bool p={bad!r} raises", raises_ve(bad, 0.5, "p"))
        check(f"bool q={bad!r} raises", raises_ve(0.5, bad, "q"))

    # 4. Finite out-of-[0,1] sweep, both arguments (just outside counts)
    sweep = [-100.0, -2.0, -1.0, -0.5, -0.01, -1e-9, 1.0 + 1e-9, 1.01, 1.5, 2.0, 10.0, 100.0, 2, -1]
    check("out-of-range p sweep all raise",
          all(raises_ve(v, 0.5, "p") for v in sweep), f"{len(sweep)} values")
    check("out-of-range q sweep all raise",
          all(raises_ve(0.5, v, "q") for v in sweep), f"{len(sweep)} values")

    # 5. Non-numeric raises ValueError (not the pre-fix bare TypeError)
    for bad in ("0.5", None, [0.5], object()):
        check(f"non-numeric p={type(bad).__name__} raises ValueError", raises_ve(bad, 0.5, "p"))
        check(f"non-numeric q={type(bad).__name__} raises ValueError", raises_ve(0.5, bad, "q"))

    # 6. p validated before q
    check("(NaN, NaN) names p first", raises_ve(float("nan"), float("nan"), "p"))
    check("(2.0, NaN) names p first", raises_ve(2.0, float("nan"), "p"))
    check("(0.5, 2.0) names q", raises_ve(0.5, 2.0, "q"))

    # 7. In-domain grid matches independent reference to 1e-12
    worst, n = 0.0, 0
    grid = [0.0, 0.001, 0.02, 0.1, 0.25, 0.5, 0.62, 0.8, 0.98, 0.999, 1.0]
    for p in grid:
        for q in grid:
            worst = max(worst, abs(kl(p, q) - reference_kl(p, q)))
            n += 1
    check(f"reference grid n={n} within 1e-12", worst <= 1e-12, f"max diff {worst:.3e}")

    # 8. Pinned pre-fix in-domain values (bit-identical)
    check("(0.5, 0.5) == 0.0", kl(0.5, 0.5) == 0.0)
    check("(0.8, 0.5) pinned", kl(0.8, 0.5) == 0.1927447570217575, repr(kl(0.8, 0.5)))
    check("(0.02, 0.98) pinned", kl(0.02, 0.98) == 3.7361474861862005, repr(kl(0.02, 0.98)))
    check("(0, 0.5) int boundary pinned", kl(0, 0.5) == 0.6931323650498874)
    check("(1, 0.5) int boundary pinned", kl(1, 0.5) == 0.693132365049887)
    check("(0.5, 0) int boundary pinned", kl(0.5, 0) == 6.2146085984224415)
    check("(0.5, 1.0) boundary pinned", kl(0.5, 1.0) == 6.214608598408064)
    check("KL >= 0 on grid", all(kl(p, q) >= 0.0 for p in grid for q in grid))
    check("KL(p, p) == 0", all(kl(v, v) == 0.0 for v in (0.02, 0.5, 0.98)))

    # 9. numpy float64 is a Python float subclass — accepted, identical
    import numpy as np
    check("np.float64 accepted", kl(np.float64(0.8), np.float64(0.5)) == kl(0.8, 0.5))
    check("np.bool_ rejected", raises_ve(np.bool_(True), 0.5, "p"))

    # 10. Caller paths unchanged on valid inputs
    book = ak.OrderBookSnapshot(
        market_id="SF-BL-9507-T", title="t",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C1",
        est_days_to_resolution=7.0,
        yes_asks=[ak.OrderBookLevel(0.58, 1500), ak.OrderBookLevel(0.61, 2500)],
        yes_bids=[ak.OrderBookLevel(0.56, 1500)], fee_rate=0.005)
    r = ak.evaluate_and_size_opportunity(book, 0.80, 0.05, 100_000.0, {})
    check("evaluate executes on valid book", r.action == ak.TradeAction.BUY_YES, str(r.action))
    check("evaluate kl matches direct call",
          r.kl_divergence_nats == kl(0.80, r.exec_price), repr(r.kl_divergence_nats))
    chk = ak.budget_sensitivity_check(book, ak.TradeAction.BUY_YES, 0.80, 1000.0, 0.505)
    check("budget_sensitivity kl_ref matches direct call",
          chk["kl_ref"] == kl(0.80, 0.505), repr(chk["kl_ref"]))
    try:
        ak.budget_sensitivity_check(book, ak.TradeAction.BUY_YES, float("nan"), 1000.0, 0.505)
        check("budget_sensitivity NaN belief raises", False)
    except ValueError:
        check("budget_sensitivity NaN belief raises", True)

    print(f"\n{sum(1 for _, ok in PASS if ok)}/{len(PASS)} checks passed")
    return 0


# pytest entry points (thin wrappers over the grouped checks above)
def test_task_repros_raise():
    assert raises_ve(float("nan"), 0.50, "p")
    assert raises_ve(0.50, float("nan"), "q")
    assert raises_ve(1.50, 0.50, "p")
    assert raises_ve(True, 0.50, "p")


def test_invalid_classes_raise():
    for bad in (float("nan"), float("inf"), float("-inf"), True, False, -1e-9, 1.0 + 1e-9, "0.5", None):
        assert raises_ve(bad, 0.5), bad
        assert raises_ve(0.5, bad), bad


def test_valid_values_pinned():
    assert ak.kl_divergence_binary(0.5, 0.5) == 0.0
    assert ak.kl_divergence_binary(0.8, 0.5) == 0.1927447570217575
    assert ak.kl_divergence_binary(0, 0.5) == 0.6931323650498874
    assert ak.kl_divergence_binary(0.5, 1.0) == 6.214608598408064


if __name__ == "__main__":
    sys.exit(main())
