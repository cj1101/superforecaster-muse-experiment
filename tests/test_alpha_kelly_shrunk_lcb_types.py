"""SF-BL-12181 tests — shrunk_lcb type-validates p, prior, n_eff, z, shrink_k.

Criteria (TASK.md prereg): shrunk_lcb raises ValueError naming the argument
on any string or bool p, prior, z, shrink_k or n_eff, with numeric calls
bit-identical; kill if p=True still returns the p=1.0 bound or z='1.0'
still coerces silently.

Run: python3 test_shrunk_lcb_types.py (exit 0 iff all pass).
"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster/superforecaster"))
import alpha_kelly as ak  # noqa: E402

PASS = []
CLEAN = dict(p=0.65, prior=0.49, n_eff=30.0, z=1.0, shrink_k=25.0)
CLEAN_VAL = 0.5106627463635252  # (30*.65+25*.49)/55 - 1.0*sqrt(.57727*.42273/55)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))


def raises_named(arg, **kwargs):
    try:
        ak.shrunk_lcb(**kwargs)
    except ValueError as e:
        return f"'{arg}'" in str(e) and "SF-BL-12181" in str(e), str(e)
    except Exception as e:  # noqa: BLE001
        return False, f"wrong exception {type(e).__name__}: {e}"
    return False, "no exception (silent coercion)"


def test_clean_and_numeric_bit_identical_fixtures():
    check("clean value", ak.shrunk_lcb(**CLEAN) == CLEAN_VAL, repr(ak.shrunk_lcb(**CLEAN)))
    check("z=0 shrunk mean", abs(ak.shrunk_lcb(0.65, 0.49, 30.0, 0.0) - 0.57727) < 1e-4)
    check("huge n_eff approaches point", ak.shrunk_lcb(0.65, 0.49, 1e7, 1.0) > 0.649)
    check("int inputs accepted", ak.shrunk_lcb(1, 0, 30, 1, 25) == ak.shrunk_lcb(1.0, 0.0, 30.0, 1.0, 25.0))
    check("defaults accepted", ak.shrunk_lcb(0.65, 0.49, 30.0) == CLEAN_VAL)
    check("boundaries 0/1", ak.shrunk_lcb(0.0, 0.0, 30.0) == 0.0 and 0.0 <= ak.shrunk_lcb(1.0, 1.0, 30.0) <= 1.0)


def test_bool_and_string_each_argument():
    bads = [True, False, "0.65", "1.0", "30", ""]
    for arg in ("p", "prior", "n_eff", "z", "shrink_k"):
        for bad in bads:
            ok, detail = raises_named(arg, **dict(CLEAN, **{arg: bad}))
            check(f"{arg}={bad!r} raises ValueError naming '{arg}'", ok, "" if ok else detail)


def test_kill_fixtures():
    ok, _ = raises_named("p", **dict(CLEAN, p=True))
    check("KILL: p=True does NOT return p=1.0 bound", ok)
    ok, _ = raises_named("z", **dict(CLEAN, z="1.0"))
    check("KILL: z='1.0' does NOT coerce silently", ok)


def test_other_non_numeric_types():
    for arg, bad in (("p", None), ("prior", None), ("n_eff", [30.0]),
                     ("z", {"z": 1}), ("shrink_k", None)):
        ok, detail = raises_named(arg, **dict(CLEAN, **{arg: bad}))
        check(f"{arg}={bad!r} raises named ValueError (not bare TypeError)", ok, "" if ok else detail)


def test_validation_order_is_signature_order():
    try:
        ak.shrunk_lcb("0.65", "0.49", "30", "1.0", "25")
        check("order p first", False, "no raise")
    except ValueError as e:
        check("order p first", "'p'" in str(e) and "'prior'" not in str(e), str(e)[:60])


def test_sweep_lcb_path_unchanged():
    book = ak.OrderBookSnapshot(
        market_id="FIXTURE", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C12181",
        est_days_to_resolution=5,
        yes_asks=[ak.OrderBookLevel(price=q, size_shares=800) for q in (0.50, 0.60, 0.70, 0.80)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=10_000)], fee_rate=0.0)
    out = ak.sweep_order_book(book, ak.TradeAction.BUY_YES, 1000.0, 0.65, 0.05,
                              posterior_n_eff=100.0, lcb_z=1.0)
    check("sweep LCB fixture (SF-BL-3840)", out == (400.0, 800.0, 0.50), repr(out))
    out = ak.sweep_order_book(book, ak.TradeAction.BUY_YES, 1000.0, 0.65, 0.05)
    check("sweep default point fixture", out == (880.0, 1600.0, 0.55), repr(out))
    # sweep's own fail-closed handling of bad LCB inputs is unchanged
    out = ak.sweep_order_book(book, ak.TradeAction.BUY_YES, 1000.0, 0.65, 0.05,
                              posterior_n_eff="30")
    check("sweep bad n_eff still fails closed (0,0,0)", out == (0.0, 0.0, 0.0), repr(out))


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
