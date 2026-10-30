"""SF-BL-11532 conformance suite: rule_logit_shrink b TYPE screen
(predmarkets).

Sections:
  A  TASK fixtures raise named ValueError (b='0.5', b=None, b=True)
     in both live copies; the b=0.5 control still returns
     [0.75, 0.25] on [0.9, 0.1].
  B  Wider type probes raise: False, np.bool_, 'foo', bytes, None,
     complex, Decimal, list/tuple/set/dict, ndarrays (0-d, 1-elem,
     bool), bare object.
  C  Pristine contrast: b='0.5' was a bare UFuncTypeError, b=None a
     bare TypeError, b=True silently == b=1.0 (unshrunk), b=False
     silently == b=0.
  D  Numeric b bit-identical patched vs pristine: curated forms
     (float, int, np.float64/32, np.int64) x panels, the full
     fit_rules logit_shrink grid, 2000 randomized draws per copy.
  E  Scope discipline: the VALUE domain is untouched — NaN / inf /
     negative / zero / >1 b behave bit-identically to pristine;
     Fraction is the one documented deviation (pristine crashed in
     _sigmoid; as a lawful numbers.Real it is now accepted via
     float() and equals the float control); the SF-BL-11498 price
     type screen still fires with a lawful b.
  F  Integration: apply_rule on a numeric frame is bit-identical;
     apply_rule with a string/bool b raises; fit_rules on a clean
     panel selects identical params patched vs pristine.
Run: python3 test_predmarkets_logit_shrink_b_types.py (exit 0 iff all pass)
"""
import importlib.util
import struct
import sys
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd

HERE = "/home/hatch/workspace/superforecaster_out/backlog/work/SF-BL-11532"
PRISTINE = f"{HERE}/pristine/predmarkets.py"
COPY_A = "/home/hatch/workspace/superforecaster_out/predmarkets/predmarkets.py"
COPY_B = "/home/hatch/workspace/superforecaster/superforecaster/predmarkets.py"

PASS = 0
FAIL = 0
FAILURES = []


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append(label)
        print(f"FAIL: {label}")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bits(arr):
    return [struct.pack("<d", float(x)) for x in np.atleast_1d(np.asarray(arr, dtype=float))]


def outcome(fn, *args):
    """('ok', bits) or ('err', ExceptionTypeName)."""
    try:
        return ("ok", bits(fn(*args)))
    except Exception as e:  # noqa: BLE001
        return ("err", type(e).__name__)


def outcome_nanaware(fn, *args):
    o = outcome(fn, *args)
    if o[0] == "err":
        return o
    return ("ok", tuple("NaN" if np.isnan(struct.unpack("<d", x)[0])
                         else x for x in o[1]))


def raises_named(fn, *args):
    """ValueError message if fn raises the named screen, else marker/None."""
    try:
        fn(*args)
    except ValueError as e:
        msg = str(e)
        return msg if ("b" in msg and "SF-BL-11532" in msg) else f"BADMSG:{msg}"
    except Exception as e:  # noqa: BLE001
        return f"WRONG-TYPE:{type(e).__name__}:{e}"
    return None


P = [0.9, 0.1]

pristine = load("pm_pristine_11532", PRISTINE)
copies = [load("pm_a_11532", COPY_A), load("pm_b_11532", COPY_B)]

for ci, mod in enumerate(copies):
    tag = f"copy{ci}"
    # ---- A: TASK fixtures ----
    for label, bad in (("b='0.5'", "0.5"), ("b=None", None), ("b=True", True)):
        msg = raises_named(mod.rule_logit_shrink, P, None, None, {"b": bad})
        check(f"A {tag} {label} raises named ValueError",
              msg is not None and not msg.startswith(("WRONG", "BADMSG")))
    out = mod.rule_logit_shrink(P, None, None, {"b": 0.5})
    check(f"A {tag} control b=0.5 returns [0.75, 0.25]",
          abs(float(out[0]) - 0.75) < 1e-15
          and abs(float(out[1]) - 0.25) < 1e-15)
    check(f"A {tag} kill: b=True no longer returns unshrunk [0.9, 0.1]",
          raises_named(mod.rule_logit_shrink, P, None, None, {"b": True}) is not None)
    check(f"A {tag} kill: b='0.5' no longer raises UFuncTypeError",
          (raises_named(mod.rule_logit_shrink, P, None, None, {"b": "0.5"}) or "")
          .startswith("rule_logit_shrink b must be"))

    # ---- B: wider probes ----
    bad_values = {
        "False": False, "np.bool_ True": np.bool_(True),
        "np.bool_ False": np.bool_(False), "string 'foo'": "foo",
        "string '1'": "1", "bytes": b"0.5", "None": None,
        "complex": 0.5 + 0j, "Decimal": Decimal("0.5"),
        "list": [0.5], "tuple": (0.5,), "set": {0.5},
        "dict": {"v": 0.5}, "0-d ndarray": np.array(0.5),
        "1-elem ndarray": np.array([0.5]), "bool ndarray": np.array([True]),
        "bare object": object(),
    }
    for label, bad in bad_values.items():
        msg = raises_named(mod.rule_logit_shrink, P, None, None, {"b": bad})
        check(f"B {tag} b={label} raises named ValueError",
              msg is not None and not msg.startswith(("WRONG", "BADMSG")))

    # ---- C: pristine contrast ----
    try:
        pristine.rule_logit_shrink(P, None, None, {"b": "0.5"})
        c1 = "no-raise"
    except Exception as e:  # noqa: BLE001
        c1 = type(e).__name__
    check(f"C {tag} pristine b='0.5' was UFuncTypeError", c1 == "UFuncTypeError")
    try:
        pristine.rule_logit_shrink(P, None, None, {"b": None})
        c2 = "no-raise"
    except Exception as e:  # noqa: BLE001
        c2 = type(e).__name__
    check(f"C {tag} pristine b=None was TypeError", c2 == "TypeError")
    check(f"C {tag} pristine b=True silently == b=1.0 (unshrunk)",
          outcome(pristine.rule_logit_shrink, P, None, None, {"b": True})
          == outcome(pristine.rule_logit_shrink, P, None, None, {"b": 1.0}))
    check(f"C {tag} pristine b=False silently == b=0",
          outcome(pristine.rule_logit_shrink, P, None, None, {"b": False})
          == outcome(pristine.rule_logit_shrink, P, None, None, {"b": 0}))

    # ---- D: numeric bit-identical ----
    numeric_forms = [0.5, 1.0, 1, 0, -1, 2, np.float64(0.85), np.float32(0.85),
                     np.int64(1), np.int32(0), 0.6, -0.1, 1.3]
    panels = [[0.9, 0.1], [0.5], [0.001, 0.999, 0.3, 0.7],
              [0.25, 0.5, 0.75]]
    for form in numeric_forms:
        for p in panels:
            check(f"D {tag} b={form!r} panel{len(p)} bit-identical",
                  outcome(mod.rule_logit_shrink, p, None, None, {"b": form})
                  == outcome(pristine.rule_logit_shrink, p, None, None, {"b": form}))
    for b in np.arange(0.60, 1.301, 0.02):  # full fit_rules grid
        check(f"D {tag} fit grid b={float(b):.4f} bit-identical",
              outcome(mod.rule_logit_shrink, P, None, None, {"b": float(b)})
              == outcome(pristine.rule_logit_shrink, P, None, None, {"b": float(b)}))
    rng = np.random.default_rng(11532)
    for i in range(2000):
        p = rng.uniform(0.01, 0.99, size=4).tolist()
        b = float(rng.uniform(-0.5, 2.0))
        check(f"D {tag} random {i} bit-identical",
              outcome(mod.rule_logit_shrink, p, None, None, {"b": b})
              == outcome(pristine.rule_logit_shrink, p, None, None, {"b": b}))

    # ---- E: value domain untouched; documented Fraction deviation;
    #         price screen intact ----
    for bad in (float("nan"), float("inf"), float("-inf"), -0.5, 0, 2.0):
        check(f"E {tag} value-domain b={bad} matches pristine",
              outcome_nanaware(mod.rule_logit_shrink, P, None, None, {"b": bad})
              == outcome_nanaware(pristine.rule_logit_shrink, P, None, None, {"b": bad}))
    check(f"E {tag} pristine Fraction(1,2) crashed (documented deviation)",
          outcome(pristine.rule_logit_shrink, P, None, None, {"b": Fraction(1, 2)})[0] == "err")
    check(f"E {tag} patched Fraction(1,2) == float control [0.75, 0.25]",
          outcome(mod.rule_logit_shrink, P, None, None, {"b": Fraction(1, 2)})
          == outcome(mod.rule_logit_shrink, P, None, None, {"b": 0.5}))
    check(f"E {tag} patched Fraction(1,3) == float(1/3) control",
          outcome(mod.rule_logit_shrink, P, None, None, {"b": Fraction(1, 3)})
          == outcome(mod.rule_logit_shrink, P, None, None, {"b": float(Fraction(1, 3))}))
    msg = None
    try:
        mod.rule_logit_shrink(["0.9", "0.1"], None, None, {"b": 0.5})
    except ValueError as e:
        msg = str(e)
    check(f"E {tag} string price panel still raises price ValueError",
          msg is not None and "price" in msg)

    # ---- F: integration ----
    df = pd.DataFrame({"price": [0.9, 0.1], "days_to_expiry": [7.0, 7.0],
                       "log10_volume": [5.5, 4.5]})
    check(f"F {tag} apply_rule logit_shrink numeric bit-identical",
          outcome(mod.apply_rule, "logit_shrink", df, {"b": 0.5}, 5.0, 1.0)
          == outcome(pristine.apply_rule, "logit_shrink", df, {"b": 0.5}, 5.0, 1.0))
    msg = raises_named(mod.apply_rule, "logit_shrink", df, {"b": "0.5"}, 5.0, 1.0)
    check(f"F {tag} apply_rule string b raises named ValueError",
          msg is not None and not msg.startswith(("WRONG", "BADMSG")))
    msg = raises_named(mod.apply_rule, "logit_shrink", df, {"b": True}, 5.0, 1.0)
    check(f"F {tag} apply_rule bool b raises named ValueError",
          msg is not None and not msg.startswith(("WRONG", "BADMSG")))

# fit_rules selects identical params patched vs pristine (clean panel)
rng = np.random.default_rng(7)
n = 240
panel = pd.DataFrame({
    "price": rng.uniform(0.35, 0.65, size=n),
    "outcome": (rng.uniform(size=n) < 0.5).astype(int),
    "days_to_expiry": rng.integers(1, 30, size=n).astype(float),
    "log10_volume": rng.uniform(4.0, 7.0, size=n),
})
panel["outcome"] = (panel["price"] + rng.normal(0, 0.15, size=n) > 0.5).astype(int)
for ci, mod in enumerate(copies):
    fitted_new = mod.fit_rules(panel)
    fitted_old = pristine.fit_rules(panel)
    check(f"F copy{ci} fit_rules selects identical params",
          fitted_new == fitted_old)

print(f"\n{PASS} passed, {FAIL} failed")
if FAILURES:
    print("FAILURES:", FAILURES[:20])
sys.exit(1 if FAIL else 0)
