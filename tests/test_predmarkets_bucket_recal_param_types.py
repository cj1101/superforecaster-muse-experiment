"""SF-BL-11531 conformance suite: rule_bucket_recal fitted edges/values TYPE screen.

Run: python3 test_predmarkets_bucket_recal_param_types.py
Tests both live copies against this dir's pristine baseline:
  A TASK fixtures raise named ValueError (values=['0.2','0.8'],
    values=[False,True], edges=['0.0','0.5']) vs the [0.8,0.2] control
  B wider probes raise for BOTH params (bool/string in list, tuple,
    ndarray, object array, Series, scalar, bytes, mixed/coerced lists;
    None, complex, Decimal and other non-numeric)
  C error message names the parameter, the type class, SF-BL-11531
  D numeric fits bit-identical to pristine (curated + 2000 randomized,
    dtype included)
  E value domain untouched (SF-BL-11418 scope): out-of-range /
    non-finite / unsorted numeric fits behave bit-identically
  F integration: apply_rule + fit_rules (bucket entry) clean paths
    bit-identical; apply_rule with corrupt fit raises
  G pristine contrast + patch scope (negative space)
"""
import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LIVE = [
    Path.home() / "workspace/superforecaster_out/predmarkets/predmarkets.py",
    Path.home() / "workspace/superforecaster/superforecaster/predmarkets.py",
]


def load(path, tag):
    spec = importlib.util.spec_from_file_location(f"pm_{tag}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


pris = load(HERE / "pristine" / "predmarkets.py", "pristine")
PASS, FAIL = [], []


def check(label, cond):
    (PASS if cond else FAIL).append(label)
    print(f"  {'PASS' if cond else 'FAIL'} {label}")


P = [0.9, 0.1]
EDGES = [0.0, 0.5]
VALUES = [0.2, 0.8]
CONTROL = {"edges": EDGES, "values": VALUES}


def raises(fn, *needles):
    """Return the ValueError message if fn raises ValueError containing
    all needles, else None."""
    try:
        fn()
    except ValueError as e:
        msg = str(e)
        return msg if all(n in msg for n in needles) else f"MISSING-NEEDLE:{msg}"
    except Exception as e:  # noqa: BLE001
        return f"WRONG-TYPE:{type(e).__name__}:{e}"
    return None


def ok_raises(fn, *needles):
    r = raises(fn, *needles)
    return r is not None and not r.startswith(("MISSING", "WRONG"))


BOOL_PROBES = [
    ("list[bool]", [False, True]),
    ("list[bool] edges-form", [True, False]),
    ("np bool array", np.array([False, True])),
    ("object array of bool", np.array([False, True], dtype=object)),
    ("bool Series", pd.Series([False, True])),
    ("bool scalar", True),
    ("mixed float+bool list", None),  # param-specific below
    ("np.bool_ list", [np.bool_(False), np.bool_(True)]),
]
STR_PROBES = [
    ("list[str]", ["0.2", "0.8"]),
    ("np str array", np.array(["0.2", "0.8"])),
    ("object array of str", np.array(["0.2", "0.8"], dtype=object)),
    ("bytes list", [b"0.2", b"0.8"]),
    ("string Series", pd.Series(["0.2", "0.8"])),
    ("str scalar", "0.5"),
    ("unparseable strings", ["foo", "bar"]),
]
OTHER_PROBES = [
    ("None list", [None, 0.8]),
    ("None scalar", None),
    ("complex array", np.array([0.2 + 0j, 0.8 + 0j])),
    ("complex list", [0.2 + 0j, 0.8 + 0j]),
    ("object numeric array", np.array([0.2, 0.8], dtype=object)),
    ("Decimal list", [Decimal("0.2"), Decimal("0.8")]),
    ("dict", {"a": 0.2}),
]


for li, live_path in enumerate(LIVE):
    pm = load(live_path, f"live{li}")
    tag = f"copy{li}"
    print(f"== {tag}: {live_path}")
    fn = pm.rule_bucket_recal

    print(" §A TASK fixtures")
    out = np.asarray(fn(P, None, None, CONTROL))
    check(f"{tag} A numeric control returns [0.8, 0.2] float64",
          out.tolist() == [0.8, 0.2] and out.dtype == np.float64)
    check(f"{tag} A values=['0.2','0.8'] raises naming values/string",
          ok_raises(lambda: fn(P, None, None,
                               {"edges": EDGES, "values": ["0.2", "0.8"]}),
                    "values", "string"))
    check(f"{tag} A values=[False,True] raises naming values/bool",
          ok_raises(lambda: fn(P, None, None,
                               {"edges": EDGES, "values": [False, True]}),
                    "values", "bool"))
    check(f"{tag} A edges=['0.0','0.5'] raises naming edges/string",
          ok_raises(lambda: fn(P, None, None,
                               {"edges": ["0.0", "0.5"], "values": VALUES}),
                    "edges", "string"))
    check(f"{tag} A edges=[False,True] raises naming edges/bool",
          ok_raises(lambda: fn(P, None, None,
                               {"edges": [False, True], "values": VALUES}),
                    "edges", "bool"))
    check(f"{tag} A string edges + string values raises",
          ok_raises(lambda: fn(P, None, None,
                               {"edges": ["0.0", "0.5"],
                                "values": ["0.2", "0.8"]}), "edges"))

    print(" §B wider probes (both params)")
    for pname, good, mixed in (
            ("values", VALUES, [0.2, True]),
            ("edges", EDGES, [0.0, True])):
        base = dict(CONTROL)
        for plabel, probe in BOOL_PROBES:
            if probe is None:
                probe = mixed
                plabel = "mixed float+bool list"
            prm = dict(base, **{pname: probe})
            check(f"{tag} B {pname} {plabel} raises naming {pname}/bool",
                  ok_raises(lambda prm=prm: fn(P, None, None, prm),
                            pname, "bool"))
        for plabel, probe in STR_PROBES:
            if pname == "values":
                prb = probe
            else:
                prb = (["0.0", "0.5"] if isinstance(probe, list)
                       and probe and isinstance(probe[0], str)
                       else probe)
                if isinstance(probe, np.ndarray) and probe.dtype.kind in "US":
                    prb = np.array(["0.0", "0.5"])
                if isinstance(probe, pd.Series) and probe.dtype == object:
                    prb = pd.Series(["0.0", "0.5"])
                if isinstance(probe, list) and probe and isinstance(probe[0], bytes):
                    prb = [b"0.0", b"0.5"]
            prm = dict(base, **{pname: prb})
            check(f"{tag} B {pname} {plabel} raises naming {pname}/string",
                  ok_raises(lambda prm=prm: fn(P, None, None, prm),
                            pname, "string"))
        mixed_str = [0.2, "0.8"] if pname == "values" else [0.0, "0.5"]
        prm = dict(base, **{pname: mixed_str})
        check(f"{tag} B {pname} mixed float+str list raises naming string",
              ok_raises(lambda prm=prm: fn(P, None, None, prm),
                        pname, "string"))
        for plabel, probe in OTHER_PROBES:
            prm = dict(base, **{pname: probe})
            check(f"{tag} B {pname} {plabel} raises ValueError naming {pname}",
                  ok_raises(lambda prm=prm: fn(P, None, None, prm), pname))

    print(" §C message content")
    msg = raises(lambda: fn(P, None, None,
                            {"edges": EDGES, "values": ["0.2", "0.8"]}),
                 "SF-BL-11531")
    check(f"{tag} C string-values message cites SF-BL-11531",
          msg is not None and not msg.startswith(("MISSING", "WRONG")))
    msg = raises(lambda: fn(P, None, None,
                            {"edges": [False, True], "values": VALUES}),
                 "SF-BL-11531")
    check(f"{tag} C bool-edges message cites SF-BL-11531",
          msg is not None and not msg.startswith(("MISSING", "WRONG")))

    print(" §D numeric fits bit-identical to pristine")
    curated = [
        ("float lists", [0.0, 0.5], [0.2, 0.8]),
        ("int lists", [0, 1], [0, 1]),
        ("tuple edges/values", (0.0, 0.25, 0.5, 0.75, 1.0),
         (0.1, 0.35, 0.6, 0.9)),
        ("float64 arrays", np.array([0.0, 0.5]), np.array([0.2, 0.8])),
        ("float32 arrays", np.array([0.0, 0.5], dtype=np.float32),
         np.array([0.2, 0.8], dtype=np.float32)),
        ("int64 arrays", np.array([0, 1]), np.array([0, 1])),
        ("uint8 arrays", np.array([0, 1], dtype=np.uint8),
         np.array([0, 1], dtype=np.uint8)),
        ("Series", pd.Series([0.0, 0.5]), pd.Series([0.2, 0.8])),
        ("decile fit", [float(e) for e in np.linspace(0, 1, 11)],
         [0.05 + 0.09 * i for i in range(10)]),
        ("single bucket", [0.0], [0.42]),
        ("empty values w/ clip", [0.0, 0.5], [0.3]),
        ("np float scalars list", [np.float64(0.0), np.float64(0.5)],
         [np.float64(0.2), np.float64(0.8)]),
        ("np int scalars list", [np.int64(0), np.int64(1)],
         [np.int64(0), np.int64(1)]),
    ]
    panels = [P, [0.0, 1.0], [0.5], [0.25, 0.75, 0.99, 0.01],
              np.array([0.3, 0.6])]
    for clabel, e, v in curated:
        for panel in panels:
            prm = {"edges": e, "values": v}
            try:
                a = np.asarray(pm.rule_bucket_recal(panel, None, None, prm))
                b = np.asarray(pris.rule_bucket_recal(panel, None, None, prm))
                same = a.tobytes() == b.tobytes() and a.dtype == b.dtype
            except Exception as ex:  # noqa: BLE001
                same = False
                print(f"    ({clabel} raised {type(ex).__name__}: {ex})")
            check(f"{tag} D {clabel} panel n={len(panel)} bit-identical", same)
    rng = np.random.default_rng(11531)
    mism = 0
    for _ in range(2000):
        nb = int(rng.integers(1, 12))
        if rng.random() < 0.5:
            e = np.sort(rng.random(nb + 1))
        else:
            e = rng.random(nb + 1)  # unsorted: pure validation must not care
        v = rng.random(nb)
        panel = rng.random(int(rng.integers(1, 20)))
        prm = {"edges": e.tolist() if rng.random() < 0.5 else e,
               "values": v.tolist() if rng.random() < 0.5 else v}
        a = np.asarray(pm.rule_bucket_recal(panel, None, None, prm))
        b = np.asarray(pris.rule_bucket_recal(panel, None, None, prm))
        if a.tobytes() != b.tobytes() or a.dtype != b.dtype:
            mism += 1
    check(f"{tag} D 2000 randomized fits bit-identical (mismatches={mism})",
          mism == 0)

    print(" §E value domain untouched (SF-BL-11418 scope NOT taken)")
    for elabel, prm in [
            ("values out of range", {"edges": EDGES, "values": [0.2, 1.5]}),
            ("values negative", {"edges": EDGES, "values": [-0.3, 0.2]}),
            ("unsorted edges", {"edges": [0.5, 0.0], "values": VALUES}),
            ("NaN value", {"edges": EDGES, "values": [0.2, np.nan]}),
            ("inf value", {"edges": EDGES, "values": [0.2, np.inf]}),
            ("NaN edge", {"edges": [0.0, np.nan], "values": VALUES})]:
        a = np.asarray(pm.rule_bucket_recal(P, None, None, prm))
        b = np.asarray(pris.rule_bucket_recal(P, None, None, prm))
        check(f"{tag} E {elabel} identical to pristine",
              a.tobytes() == b.tobytes() and a.dtype == b.dtype)

    print(" §F integration")
    df = pd.DataFrame({
        "price": [0.9, 0.1, 0.6, 0.3],
        "days_to_expiry": [7.0, 7.0, 3.0, 14.0],
        "log10_volume": [5.0, 5.5, 4.5, 6.0],
        "outcome": [1, 0, 1, 0],
    })
    a = np.asarray(pm.apply_rule("bucket_recal", df, CONTROL, 5.0, 1.0))
    b = np.asarray(pris.apply_rule("bucket_recal", df, CONTROL, 5.0, 1.0))
    check(f"{tag} F apply_rule clean frame bit-identical",
          a.tobytes() == b.tobytes())
    check(f"{tag} F apply_rule string values raises",
          ok_raises(lambda: pm.apply_rule(
              "bucket_recal", df, {"edges": EDGES, "values": ["0.2", "0.8"]},
              5.0, 1.0), "values", "string"))
    check(f"{tag} F apply_rule string edges raises",
          ok_raises(lambda: pm.apply_rule(
              "bucket_recal", df, {"edges": ["0.0", "0.5"], "values": VALUES},
              5.0, 1.0), "edges", "string"))
    rng = np.random.default_rng(99)
    n = 300
    p_fit = rng.uniform(0.2, 0.8, n)
    fit = pd.DataFrame({
        "price": p_fit,
        "days_to_expiry": rng.choice([1.0, 3.0, 7.0, 14.0, 30.0], n),
        "log10_volume": rng.normal(5.0, 1.0, n),
        "outcome": (rng.random(n) < p_fit).astype(int),
    })
    fa, _, _ = pm.fit_rules(fit.copy())
    fb, _, _ = pris.fit_rules(fit.copy())
    ba, bb = fa["bucket_recal"], fb["bucket_recal"]
    check(f"{tag} F fit_rules bucket_recal entry identical",
          ba["edges"] == bb["edges"] and ba["values"] == bb["values"])
    a = np.asarray(pm.rule_bucket_recal(p_fit, None, None, ba))
    b = np.asarray(pris.rule_bucket_recal(p_fit, None, None, bb))
    check(f"{tag} F fitted decile params bit-identical through rule",
          a.tobytes() == b.tobytes())

print(" §G pristine contrast + patch scope")
for plabel, prm, want in [
        ("string values", {"edges": EDGES, "values": ["0.2", "0.8"]},
         ["0.8", "0.2"]),
        ("bool values", {"edges": EDGES, "values": [False, True]},
         [True, False]),
        ("string edges", {"edges": ["0.0", "0.5"], "values": VALUES},
         [0.8, 0.2])]:
    out = np.asarray(pris.rule_bucket_recal(P, None, None, prm))
    check(f"G pristine {plabel} returned {want} (bug is real)",
          out.tolist() == want)
patch_text = (HERE / "fix.patch").read_text()
check("G fix.patch cites SF-BL-11531", "SF-BL-11531" in patch_text)
check("G fix.patch adds the bucket helper",
      "_validate_bucket_recal_param" in patch_text)
check("G fix.patch touches no scoring/IO function",
      all(s not in patch_text for s in
          ("def brier", "def ece", "def log_score", "def clean_markets",
           "def fetch_", "def build_snapshots", "def calibration_table",
           "def fit_rules")))

print(f"\nTOTAL: {len(PASS)} PASS, {len(FAIL)} FAIL")
if FAIL:
    print("FAILURES:")
    for f in FAIL:
        print(f"  - {f}")
    sys.exit(1)
