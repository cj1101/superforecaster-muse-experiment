"""SF-BL-11500 conformance suite: zlogvol TYPE screen in
rule_logit_time_vol (predmarkets), via _validated_zlogvol_array.

Sections:
  A  TASK fixtures raise named ValueError (z=['0.5','-0.5'], z=[True, False])
     via rule_logit_time_vol and _validated_zlogvol_array, in both live
     copies; numeric control returns the TASK values [0.9397, 0.1614].
  B  Wider type probes raise named ValueError: string/bool scalars,
     np string/bool arrays, mixed [0.5,'-0.5'], object-dtype panels
     (incl. numeric-valued), string/bool Series, None, bytes, complex,
     Decimal object array, datetime array.
  C  Pristine contrast: string panel was bit-identical to numeric,
     bool panel was bit-identical to [1, 0], None returned all-NaN,
     complex died with a bare TypeError (not a named ValueError).
  D  Numeric z bit-identical patched vs pristine: curated forms
     (float/int lists, tuple, float64/float32/int32/int64 ndarrays,
     numeric Series, numeric scalar, empty panel) x panels, the full
     fit_rules logit_time_vol grid, 2000 randomized panels.
  E  Scope discipline: the VALUE domain is untouched — inf / -inf /
     NaN numeric z behave bit-identically (NaN-aware) to pristine
     (SF-BL-11376's prereg, not taken here); sibling screens still
     fire (b2='0.5' SF-BL-11496, days string SF-BL-11499, price
     string SF-BL-11498); other rules that ignore zlogvol are
     unaffected by a string z (rule_logit_time, rule_raw).
  F  Integration: apply_rule on a numeric frame is bit-identical;
     fit_rules on a clean panel selects identical params.
Run: python3 test_zlogvol_panel_type_screen.py (exit 0 iff all pass)
"""
import importlib.util
import struct
import sys
from decimal import Decimal

import numpy as np
import pandas as pd

HERE = "/home/hatch/workspace/superforecaster_out/backlog/work/SF-BL-11500"
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
    """('ok', bits) or ('err', ExceptionTypeName). NaN bits are kept raw;
    use outcome_nan for panels that may contain NaN."""
    try:
        return ("ok", bits(fn(*args)))
    except Exception as e:  # noqa: BLE001
        return ("err", type(e).__name__)


def outcome_nan(fn, *args):
    try:
        vals = [float(x) for x in np.atleast_1d(np.asarray(fn(*args), dtype=float))]
        return ("ok", [("nan" if np.isnan(v) else struct.pack("<d", v)) for v in vals])
    except Exception as e:  # noqa: BLE001
        return ("err", type(e).__name__)


def raises_named(fn, *args):
    """ValueError message if fn raises the named screen, else marker/None."""
    try:
        fn(*args)
    except ValueError as e:
        msg = str(e)
        return msg if ("zlogvol" in msg and "SF-BL-11500" in msg) else f"BADMSG:{msg}"
    except Exception as e:  # noqa: BLE001
        return f"WRONG-TYPE:{type(e).__name__}:{e}"
    return None


P = [0.9, 0.1]
DAYS = [7, 7]
Z = [0.5, -0.5]
GOOD = {"b0": 1.0, "b1": 0.0, "b2": 0.5}

pristine = load("pm_pristine_11500", PRISTINE)
copies = [load("pm_a_11500", COPY_A), load("pm_b_11500", COPY_B)]

for ci, mod in enumerate(copies):
    tag = f"copy{ci}"
    # ---- A: TASK fixtures ----
    for label, z in (("string ['0.5','-0.5']", ["0.5", "-0.5"]),
                     ("bool [True, False]", [True, False])):
        msg = raises_named(mod.rule_logit_time_vol, P, DAYS, z, GOOD)
        check(f"A {tag} {label} raises named ValueError",
              msg is not None and not msg.startswith(("WRONG", "BADMSG")))
        msg = raises_named(mod._validated_zlogvol_array, z)
        check(f"A {tag} helper {label} raises named ValueError",
              msg is not None and not msg.startswith(("WRONG", "BADMSG")))
    out = mod.rule_logit_time_vol(P, DAYS, Z, GOOD)
    check(f"A {tag} numeric control is [0.9397, 0.1614]",
          abs(float(out[0]) - 0.9397171187267773) < 1e-15
          and abs(float(out[1]) - 0.16139047779640894) < 1e-15)
    out = mod.rule_logit_time_vol(P, DAYS, [1, 0], GOOD)
    check(f"A {tag} numeric [1,0] control is [0.9643, 0.1]",
          abs(float(out[0]) - 0.9642857142857143) < 1e-15
          and abs(float(out[1]) - 0.10000000000000002) < 1e-15)

    # ---- B: wider probes ----
    bad_panels = {
        "string scalar '0.5'": "0.5", "bool scalar True": True,
        "np string array": np.array(["0.5", "-0.5"]),
        "np bool array": np.array([True, False]),
        "mixed [0.5, '-0.5']": [0.5, "-0.5"],
        "object array numeric": np.array([0.5, -0.5], dtype=object),
        "object list via Decimal": [Decimal("0.5"), Decimal("-0.5")],
        "string Series": pd.Series(["0.5", "-0.5"]),
        "bool Series": pd.Series([True, False]),
        "None": None, "bytes panel": [b"0.5", b"-0.5"],
        "complex panel": [0.5 + 0j, -0.5 + 0j],
        "np complex array": np.array([0.5 + 0j, -0.5 + 0j]),
        "datetime array": np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[D]"),
        "bool scalar False": False,
        "single string in list": ["0.5", -0.5],
    }
    for label, z in bad_panels.items():
        msg = raises_named(mod.rule_logit_time_vol, P, DAYS, z, GOOD)
        check(f"B {tag} z={label} raises named ValueError",
              msg is not None and not msg.startswith(("WRONG", "BADMSG")))

    # ---- C: pristine contrast ----
    check(f"C {tag} pristine string == numeric bit-identical",
          outcome(pristine.rule_logit_time_vol, P, DAYS, ["0.5", "-0.5"], GOOD)
          == outcome(pristine.rule_logit_time_vol, P, DAYS, Z, GOOD))
    check(f"C {tag} pristine bool == [1,0] bit-identical",
          outcome(pristine.rule_logit_time_vol, P, DAYS, [True, False], GOOD)
          == outcome(pristine.rule_logit_time_vol, P, DAYS, [1, 0], GOOD))
    o = outcome_nan(pristine.rule_logit_time_vol, P, DAYS, None, GOOD)
    check(f"C {tag} pristine None silently returned all-NaN",
          o[0] == "ok" and all(v == "nan" for v in o[1]))
    msg = raises_named(pristine.rule_logit_time_vol, P, DAYS, [0.5 + 0j, -0.5 + 0j], GOOD)
    check(f"C {tag} pristine complex was not a named ValueError",
          msg is None or msg.startswith("WRONG"))

    # ---- D: numeric bit-identical ----
    numeric_forms = [
        ("float list", [0.5, -0.5]), ("int list", [1, 0]),
        ("tuple", (0.5, -0.5)), ("float64 array", np.array([0.5, -0.5])),
        ("float32 array", np.array([0.5, -0.5], dtype=np.float32)),
        ("int64 array", np.array([1, 0], dtype=np.int64)),
        ("int32 array", np.array([1, 0], dtype=np.int32)),
        ("numeric Series", pd.Series([0.5, -0.5])),
        ("scalar 0.5", 0.5), ("np.float64 scalar", np.float64(0.5)),
        ("np.int64 scalar", np.int64(1)), ("empty list", []),
    ]
    for label, z in numeric_forms:
        p_use = P if np.size(z) != 0 else []
        check(f"D {tag} z form {label} bit-identical",
              outcome(mod.rule_logit_time_vol, p_use, DAYS if np.size(z) else [], z, GOOD)
              == outcome(pristine.rule_logit_time_vol, p_use, DAYS if np.size(z) else [], z, GOOD))
    panels = [([0.9, 0.1], [7, 7], [0.5, -0.5]),
              ([0.5], [1], [0.0]),
              ([0.001, 0.999, 0.3, 0.7], [30, 14, 3, 1], [1.0, -1.0, 0.25, -0.25])]
    for (p, d, z) in panels:
        check(f"D {tag} panel{len(p)} bit-identical",
              outcome(mod.rule_logit_time_vol, p, d, z, GOOD)
              == outcome(pristine.rule_logit_time_vol, p, d, z, GOOD))
    for b0 in np.arange(0.70, 1.16, 0.05):
        for b1 in np.arange(-0.10, 0.151, 0.05):
            for b2 in np.arange(-0.10, 0.151, 0.05):
                prm = {"b0": float(b0), "b1": float(b1), "b2": float(b2)}
                check(f"D {tag} fit grid {prm} bit-identical",
                      outcome(mod.rule_logit_time_vol, P, DAYS, Z, prm)
                      == outcome(pristine.rule_logit_time_vol, P, DAYS, Z, prm))
    rng = np.random.default_rng(11500)
    for i in range(2000):
        p = rng.uniform(0.01, 0.99, size=4).tolist()
        d = rng.integers(0, 60, size=4).tolist()
        z = rng.normal(size=4).tolist()
        prm = {"b0": float(rng.uniform(0.4, 1.4)), "b1": float(rng.uniform(-0.3, 0.4)),
               "b2": float(rng.uniform(-0.3, 0.4))}
        check(f"D {tag} random {i} bit-identical",
              outcome(mod.rule_logit_time_vol, p, d, z, prm)
              == outcome(pristine.rule_logit_time_vol, p, d, z, prm))

    # ---- E: value domain untouched; sibling screens intact ----
    for label, z in (("inf", [float("inf"), 0.0]), ("-inf", [float("-inf"), 0.0]),
                     ("NaN", [float("nan"), 0.0]), ("large 1e6", [1e6, -1e6]),
                     ("negative", [-2.5, 3.5])):
        check(f"E {tag} value-domain z={label} matches pristine (NaN-aware)",
              outcome_nan(mod.rule_logit_time_vol, P, DAYS, z, GOOD)
              == outcome_nan(pristine.rule_logit_time_vol, P, DAYS, z, GOOD))
    # b2 screen (SF-BL-11496) still fires and names b2, not zlogvol
    try:
        mod.rule_logit_time_vol(P, DAYS, Z, {**GOOD, "b2": "0.5"})
        msg_b2 = None
    except ValueError as e:
        msg_b2 = str(e)
    check(f"E {tag} b2='0.5' still raises SF-BL-11496 ValueError",
          msg_b2 is not None and "b2" in msg_b2 and "SF-BL-11496" in msg_b2)
    # days screen (SF-BL-11499) still fires through this rule
    try:
        mod.rule_logit_time_vol(P, ["10", "20"], Z, GOOD)
        msg_d = None
    except ValueError as e:
        msg_d = str(e)
    check(f"E {tag} string days still raises days ValueError",
          msg_d is not None and "days" in msg_d)
    # price screen (SF-BL-11498) still fires through this rule
    try:
        mod.rule_logit_time_vol(["0.9", "0.1"], DAYS, Z, GOOD)
        msg_p = None
    except ValueError as e:
        msg_p = str(e)
    check(f"E {tag} string prices still raise price ValueError",
          msg_p is not None and "price" in msg_p)
    # rules that ignore zlogvol are unaffected by a string z panel
    check(f"E {tag} rule_logit_time ignores string z, bit-identical",
          outcome(mod.rule_logit_time, P, DAYS, ["0.5", "-0.5"], {"b0": 1.0, "b1": 0.1})
          == outcome(pristine.rule_logit_time, P, DAYS, ["0.5", "-0.5"], {"b0": 1.0, "b1": 0.1}))
    check(f"E {tag} rule_raw ignores string z, bit-identical",
          outcome(mod.rule_raw, P, DAYS, ["0.5", "-0.5"], {})
          == outcome(pristine.rule_raw, P, DAYS, ["0.5", "-0.5"], {}))

    # ---- F: integration ----
    df = pd.DataFrame({"price": [0.9, 0.1], "days_to_expiry": [7.0, 7.0],
                       "log10_volume": [5.5, 4.5]})
    check(f"F {tag} apply_rule logit_time_vol numeric bit-identical",
          outcome(mod.apply_rule, "logit_time_vol", df, GOOD, 5.0, 1.0)
          == outcome(pristine.apply_rule, "logit_time_vol", df, GOOD, 5.0, 1.0))

rng = np.random.default_rng(7)
n = 240
panel = pd.DataFrame({
    "price": rng.uniform(0.35, 0.65, size=n),
    "outcome": (rng.uniform(size=n) < 0.5).astype(int),
    "days_to_expiry": rng.integers(1, 30, size=n).astype(float),
    "log10_volume": rng.uniform(4.0, 7.0, size=n),
})
panel["outcome"] = (rng.uniform(size=n) < panel["price"]).astype(int)
for ci, mod in enumerate(copies):
    new = mod.fit_rules(panel)
    old = pristine.fit_rules(panel)
    check(f"F copy{ci} fit_rules params identical", new[0] == old[0])
    check(f"F copy{ci} fit_rules z stats identical", new[1:] == old[1:])

print(f"\n{PASS} passed, {FAIL} failed")
if FAILURES:
    print("failures:", FAILURES[:20])
sys.exit(1 if FAIL else 0)
