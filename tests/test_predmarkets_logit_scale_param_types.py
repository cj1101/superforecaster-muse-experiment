"""SF-BL-11496 conformance suite: b0/b1/b2 TYPE screen in _scale_time
and rule_logit_time_vol (predmarkets).

Sections:
  A  TASK fixtures raise named ValueError (b0='1', b1='0.1', b2='0.5',
     b0=True, b2=True) via _scale_time, rule_logit_time and
     rule_logit_time_vol, in both live copies.
  B  Wider type probes raise for every parameter: False, np.bool_,
     'foo', bytes, None, complex, Decimal, list/tuple/set/dict,
     ndarrays (0-d, 1-element, bool), bare object.
  C  Error is a ValueError whose message names the parameter and the
     task id; never a UFuncTypeError/TypeError.
  D  Numeric parameters bit-identical patched vs pristine: curated
     forms (float, int, np.float64/32, np.int64, Fraction) x panels,
     the full fit_rules grids, 2000 randomized parameter draws.
  E  Scope discipline: the VALUE domain is untouched — NaN / inf /
     negative b0/b1/b2 behave bit-identically to pristine (a later
     value screen's prereg, not taken here); the SF-BL-11499 days
     type screen still fires.
  F  Integration: apply_rule on a numeric frame is bit-identical;
     apply_rule with a string b0 raises; fit_rules on a clean panel
     selects identical params patched vs pristine.
Run: python3 test_predmarkets_logit_scale_param_types.py (exit 0 iff all pass)
"""
import importlib.util
import struct
import sys
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd

HERE = "/home/hatch/workspace/superforecaster_out/backlog/work/SF-BL-11496"
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


def raises_named(fn, param, *args):
    """ValueError message if fn raises the named screen, else None."""
    try:
        fn(*args)
    except ValueError as e:
        msg = str(e)
        return msg if (param in msg and "SF-BL-11496" in msg) else f"BADMSG:{msg}"
    except Exception as e:  # noqa: BLE001
        return f"WRONG-TYPE:{type(e).__name__}:{e}"
    return None


P = [0.9, 0.1]
DAYS = [7, 7]
Z = [0.5, -0.5]
GOOD_TIME = {"b0": 1.0, "b1": 0.1}
GOOD_VOL = {"b0": 1.0, "b1": 0.0, "b2": 0.5}

pristine = load("pm_pristine_11496", PRISTINE)
copies = [load("pm_a_11496", COPY_A), load("pm_b_11496", COPY_B)]

for ci, mod in enumerate(copies):
    tag = f"copy{ci}"
    # ---- A: TASK fixtures ----
    fixtures = [
        ("b0='1'", "b0", lambda m, prm: m._scale_time(prm, DAYS), {**GOOD_TIME, "b0": "1"}),
        ("b1='0.1'", "b1", lambda m, prm: m._scale_time(prm, DAYS), {**GOOD_TIME, "b1": "0.1"}),
        ("b0=True", "b0", lambda m, prm: m._scale_time(prm, DAYS), {**GOOD_TIME, "b0": True}),
        ("b0='1' via rule_logit_time", "b0",
         lambda m, prm: m.rule_logit_time(P, DAYS, Z, prm), {**GOOD_TIME, "b0": "1"}),
        ("b1='0.1' via rule_logit_time", "b1",
         lambda m, prm: m.rule_logit_time(P, DAYS, Z, prm), {**GOOD_TIME, "b1": "0.1"}),
        ("b0=True via rule_logit_time", "b0",
         lambda m, prm: m.rule_logit_time(P, DAYS, Z, prm), {**GOOD_TIME, "b0": True}),
        ("b2='0.5'", "b2", lambda m, prm: m.rule_logit_time_vol(P, DAYS, Z, prm),
         {**GOOD_VOL, "b2": "0.5"}),
        ("b2=True", "b2", lambda m, prm: m.rule_logit_time_vol(P, DAYS, Z, prm),
         {**GOOD_VOL, "b2": True}),
        ("b0='1' via rule_logit_time_vol", "b0",
         lambda m, prm: m.rule_logit_time_vol(P, DAYS, Z, prm), {**GOOD_VOL, "b0": "1"}),
        ("b1='0.1' via rule_logit_time_vol", "b1",
         lambda m, prm: m.rule_logit_time_vol(P, DAYS, Z, prm), {**GOOD_VOL, "b1": "0.1"}),
    ]
    for label, param, fn, prm in fixtures:
        msg = raises_named(lambda prm=prm, fn=fn: fn(mod, prm), param)
        check(f"A {tag} {label} raises named ValueError",
              msg is not None and not msg.startswith(("WRONG", "BADMSG")))

    # TASK's numeric controls still return the TASK values on patched
    out = mod.rule_logit_time_vol(P, DAYS, Z, {**GOOD_VOL, "b2": 1.0})
    check(f"A {tag} lawful b2=1.0 control is [0.9643, 0.25]",
          abs(float(out[0]) - 0.9642857142857143) < 1e-15
          and abs(float(out[1]) - 0.25) < 1e-15)

    # ---- B: wider probes, every parameter ----
    bad_values = {
        "False": False, "np.bool_": np.bool_(True), "string 'foo'": "foo",
        "bytes": b"1", "None": None, "complex": 1 + 0j,
        "Decimal": Decimal("1.0"), "list": [1.0], "tuple": (1.0,),
        "set": {1.0}, "dict": {"v": 1.0}, "0-d ndarray": np.array(1.0),
        "1-elem ndarray": np.array([1.0]), "bool ndarray": np.array([True]),
        "bare object": object(),
    }
    for pname in ("b0", "b1"):
        for label, bad in bad_values.items():
            prm = {**GOOD_TIME, pname: bad}
            msg = raises_named(mod._scale_time, pname, prm, DAYS)
            check(f"B {tag} _scale_time {pname}={label} raises",
                  msg is not None and not msg.startswith(("WRONG", "BADMSG")))
    for label, bad in bad_values.items():
        prm = {**GOOD_VOL, "b2": bad}
        msg = raises_named(mod.rule_logit_time_vol, "b2", P, DAYS, Z, prm)
        check(f"B {tag} rule_logit_time_vol b2={label} raises",
              msg is not None and not msg.startswith(("WRONG", "BADMSG")))

    # ---- C: pristine contrast — fixtures were NOT named ValueErrors ----
    msg = raises_named(pristine._scale_time, "b0", {**GOOD_TIME, "b0": "1"}, DAYS)
    check(f"C {tag} pristine b0='1' was not a named ValueError", msg is None or msg.startswith("WRONG"))
    check(f"C {tag} pristine b0=True silently == b0=1.0",
          outcome(pristine._scale_time, {**GOOD_TIME, "b0": True}, DAYS)
          == outcome(pristine._scale_time, GOOD_TIME, DAYS))
    check(f"C {tag} pristine b2=True silently == b2=1.0",
          outcome(pristine.rule_logit_time_vol, P, DAYS, Z, {**GOOD_VOL, "b2": True})
          == outcome(pristine.rule_logit_time_vol, P, DAYS, Z, {**GOOD_VOL, "b2": 1.0}))

    # ---- D: numeric bit-identical ----
    numeric_forms = [1.0, 1, 0, -1, 2, np.float64(0.85), np.float32(0.85),
                     np.int64(1), Fraction(17, 20), 0.6, -0.1]
    panels = [([0.9, 0.1], [7, 7], [0.5, -0.5]),
              ([0.5], [1], [0.0]),
              ([0.001, 0.999, 0.3, 0.7], [30, 14, 3, 1], [1.0, -1.0, 0.25, -0.25])]
    for form in numeric_forms:
        for (p, d, z) in panels:
            for base, fn_name in ((GOOD_TIME, "rule_logit_time"),):
                prm = {**base, "b0": form}
                check(f"D {tag} {fn_name} b0={form!r} panel{len(p)} bit-identical",
                      outcome(mod.rule_logit_time, p, d, z, prm)
                      == outcome(pristine.rule_logit_time, p, d, z, prm))
                prm = {**base, "b1": form}
                check(f"D {tag} {fn_name} b1={form!r} panel{len(p)} bit-identical",
                      outcome(mod.rule_logit_time, p, d, z, prm)
                      == outcome(pristine.rule_logit_time, p, d, z, prm))
            prm = {**GOOD_VOL, "b2": form}
            check(f"D {tag} rule_logit_time_vol b2={form!r} panel{len(p)} bit-identical",
                  outcome(mod.rule_logit_time_vol, p, d, z, prm)
                  == outcome(pristine.rule_logit_time_vol, p, d, z, prm))
    # full fit grids
    for b0 in np.arange(0.60, 1.21, 0.05):
        for b1 in np.arange(-0.10, 0.201, 0.025):
            prm = {"b0": float(b0), "b1": float(b1)}
            check(f"D {tag} fit grid logit_time {prm} bit-identical",
                  outcome(mod.rule_logit_time, P, DAYS, Z, prm)
                  == outcome(pristine.rule_logit_time, P, DAYS, Z, prm))
    for b0 in np.arange(0.70, 1.16, 0.05):
        for b1 in np.arange(-0.10, 0.151, 0.05):
            for b2 in np.arange(-0.10, 0.151, 0.05):
                prm = {"b0": float(b0), "b1": float(b1), "b2": float(b2)}
                check(f"D {tag} fit grid logit_time_vol {prm} bit-identical",
                      outcome(mod.rule_logit_time_vol, P, DAYS, Z, prm)
                      == outcome(pristine.rule_logit_time_vol, P, DAYS, Z, prm))
    # randomized draws
    rng = np.random.default_rng(11496)
    for i in range(2000):
        p = rng.uniform(0.01, 0.99, size=4).tolist()
        d = rng.integers(0, 60, size=4).tolist()
        z = rng.normal(size=4).tolist()
        prm_t = {"b0": float(rng.uniform(0.4, 1.4)), "b1": float(rng.uniform(-0.3, 0.4))}
        prm_v = {**prm_t, "b2": float(rng.uniform(-0.3, 0.4))}
        check(f"D {tag} random {i} rule_logit_time bit-identical",
              outcome(mod.rule_logit_time, p, d, z, prm_t)
              == outcome(pristine.rule_logit_time, p, d, z, prm_t))
        check(f"D {tag} random {i} rule_logit_time_vol bit-identical",
              outcome(mod.rule_logit_time_vol, p, d, z, prm_v)
              == outcome(pristine.rule_logit_time_vol, p, d, z, prm_v))
        check(f"D {tag} random {i} _scale_time bit-identical",
              outcome(mod._scale_time, prm_t, d)
              == outcome(pristine._scale_time, prm_t, d))

    # ---- E: value domain untouched; days screen intact ----
    for pname, bad in (("b0", float("nan")), ("b0", float("inf")), ("b1", -5.0),
                       ("b1", float("nan")), ("b2", float("inf")), ("b2", -2.5)):
        base = GOOD_VOL if pname == "b2" else GOOD_TIME
        prm = {**base, pname: bad}
        fn = mod.rule_logit_time_vol if base is GOOD_VOL else mod.rule_logit_time
        pfn = pristine.rule_logit_time_vol if base is GOOD_VOL else pristine.rule_logit_time
        o_new, o_old = outcome(fn, P, DAYS, Z, prm), outcome(pfn, P, DAYS, Z, prm)
        same = (o_new[0] == o_old[0]
                and (o_new[0] == "err" or all(
                    (x == y) or (np.isnan(struct.unpack('<d', x)[0])
                                  and np.isnan(struct.unpack('<d', y)[0]))
                    for x, y in zip(o_new[1], o_old[1]))))
        check(f"E {tag} value-domain {pname}={bad} matches pristine", same)
    msg = None
    try:
        mod.rule_logit_time(P, ["10", "20"], Z, GOOD_TIME)
    except ValueError as e:
        msg = str(e)
    check(f"E {tag} days string panel still raises days ValueError",
          msg is not None and "days" in msg)

    # ---- F: integration ----
    df = pd.DataFrame({"price": [0.9, 0.1], "days_to_expiry": [7.0, 7.0],
                       "log10_volume": [5.5, 4.5]})
    check(f"F {tag} apply_rule logit_time_vol numeric bit-identical",
          outcome(mod.apply_rule, "logit_time_vol", df, GOOD_VOL, 5.0, 1.0)
          == outcome(pristine.apply_rule, "logit_time_vol", df, GOOD_VOL, 5.0, 1.0))
    check(f"F {tag} apply_rule logit_time numeric bit-identical",
          outcome(mod.apply_rule, "logit_time", df, GOOD_TIME, 5.0, 1.0)
          == outcome(pristine.apply_rule, "logit_time", df, GOOD_TIME, 5.0, 1.0))
    msg = raises_named(mod.apply_rule, "b0", "logit_time_vol", df,
                       {**GOOD_VOL, "b0": "1"}, 5.0, 1.0)
    check(f"F {tag} apply_rule string b0 raises named ValueError",
          msg is not None and not msg.startswith(("WRONG", "BADMSG")))
    msg = raises_named(mod.apply_rule, "b2", "logit_time_vol", df,
                       {**GOOD_VOL, "b2": True}, 5.0, 1.0)
    check(f"F {tag} apply_rule bool b2 raises named ValueError",
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
# outcomes correlated with price so the grid search is meaningful but lawful
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
