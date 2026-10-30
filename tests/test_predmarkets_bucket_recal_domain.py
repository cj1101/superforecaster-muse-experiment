"""SF-BL-11418 conformance suite: rule_bucket_recal fitted edges/values
VALUE-domain screen (monotonic edges; values in [0, 1]; finite).

Run: python3 test_predmarkets_bucket_recal_domain.py
Tests both live copies against this dir's pristine baseline:
  A TASK fixtures raise named ValueError (values=[0.2, 1.5, -0.3];
    edges=[0.8, 0.2]) vs the [0.8, 0.2] clean control
  B wider domain probes raise for both params (non-finite anywhere,
    out-of-range selected or unselected, decreasing edges at any
    position, int/np-scalar/tuple/ndarray/Series forms, empty values)
  C error messages name the parameter, the offence and SF-BL-11418,
    and are ValueError (never a bare IndexError/TypeError)
  D lawful fits bit-identical to pristine (curated forms + 2000
    randomized fits per copy, dtype included; non-decreasing edges
    with equal neighbours, boundary values 0.0/1.0, sentinel edges
    outside [0, 1] all stay lawful)
  E composition with SF-BL-11531: the TYPE screen still fires first
    (string/bool fits raise the 11531 ValueError), lawful typed fits
    pass both screens
  F integration: apply_rule + fit_rules bucket fit clean paths
    bit-identical; apply_rule with a corrupt fit raises
  G pristine contrast + patch scope (apply_fix on pristine ==
    patched/; live bucket sections == patched bucket sections)
"""
import importlib.util
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


P = np.array([0.9, 0.1])
CONTROL = {"edges": [0.0, 0.5], "values": [0.2, 0.8]}


def call(mod, params, p=P):
    return mod.rule_bucket_recal(p, None, None, params)


def raises(fn, *needles):
    """ValueError message containing all needles, else None."""
    try:
        fn()
    except ValueError as e:
        msg = str(e)
        return msg if all(n in msg for n in needles) else None
    except Exception:  # noqa: BLE001 - wrong exception type is a failure
        return None
    return None


def same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.tobytes() == b.tobytes()


for li, live_path in enumerate(LIVE):
    live = load(live_path, f"live{li}")
    tag = f"[copy{li}]"
    print(f"== {live_path} ==")

    # ---- A: TASK fixtures -------------------------------------------
    check(f"A1 {tag} clean control returns [0.8, 0.2]",
          same_bits(call(live, CONTROL), np.array([0.8, 0.2])))
    check(f"A2 {tag} REPRO1 values=[0.2,1.5,-0.3] raises naming values/[0, 1]/11418",
          raises(lambda: call(live, {"edges": [0.0, 0.5, 0.75],
                                     "values": [0.2, 1.5, -0.3]}),
                 "values", "[0, 1]", "SF-BL-11418") is not None)
    check(f"A3 {tag} REPRO1 variant edges=[0.0,0.5] raises",
          raises(lambda: call(live, {"edges": [0.0, 0.5],
                                     "values": [0.2, 1.5, -0.3]}),
                 "values", "SF-BL-11418") is not None)
    check(f"A4 {tag} REPRO2 edges=[0.8,0.2] raises naming edges/monotonic/11418",
          raises(lambda: call(live, {"edges": [0.8, 0.2],
                                     "values": [0.2, 0.8]}),
                 "edges", "monotonic", "SF-BL-11418") is not None)
    check(f"A5 {tag} pristine REPRO1 emitted an impossible value (contrast)",
          float(np.max(call(pris, {"edges": [0.0, 0.5],
                                   "values": [0.2, 1.5, -0.3]}))) == 1.5)
    check(f"A6 {tag} pristine REPRO2 bucketed silently (contrast)",
          same_bits(call(pris, {"edges": [0.8, 0.2], "values": [0.2, 0.8]}),
                    np.array([0.8, 0.2])))

    # ---- B: wider domain probes --------------------------------------
    bad_values = [
        [0.2, float("nan")], [float("nan"), 0.8], [0.2, float("inf")],
        [0.2, float("-inf")], [1.5, 0.2], [-0.3, 0.2], [0.2, 1.0000001],
        [0.2, -1e-12], [2, 0], [0, -1], [0.5, 0.5, 7.0],
    ]
    for bv in bad_values:
        edges = [0.0, 0.5] if len(bv) == 2 else [0.0, 0.4, 0.7]
        check(f"B-v {tag} values={bv} raises",
              raises(lambda bv=bv, edges=edges: call(
                  live, {"edges": edges, "values": bv}),
                  "values", "SF-BL-11418") is not None)
    check(f"B-v-unselected {tag} out-of-range value in unselected bucket raises",
          raises(lambda: call(live, {"edges": [0.0, 0.5], "values": [0.2, 1.5]},
                              p=np.array([0.1, 0.2])),
                 "values", "SF-BL-11418") is not None)
    bad_edges = [
        [0.8, 0.2], [0.0, 0.9, 0.4], [0.5, 0.5, 0.4], [1.0, 0.0],
        [0.0, float("nan")], [float("nan"), 0.5], [0.0, float("inf")],
        [float("-inf"), 0.5, 0.4],
    ]
    for be in bad_edges:
        vals = [0.2] * max(len(be) - 1, 1) + [0.8]
        check(f"B-e {tag} edges={be} raises",
              raises(lambda be=be, vals=vals: call(
                  live, {"edges": be, "values": vals}),
                  "edges", "SF-BL-11418") is not None)
    forms = [
        ("tuple values", {"edges": (0.0, 0.5), "values": (0.2, 1.5)}),
        ("ndarray values", {"edges": [0.0, 0.5],
                            "values": np.array([0.2, np.nan])}),
        ("float32 values", {"edges": [0.0, 0.5],
                            "values": np.array([0.2, 1.5], dtype=np.float32)}),
        ("int values", {"edges": [0, 1], "values": np.array([0, 2])}),
        ("Series values", {"edges": [0.0, 0.5],
                           "values": pd.Series([0.2, -0.1])}),
        ("np scalar value", {"edges": [0.0, 0.5],
                             "values": [0.2, np.float64(1.5)]}),
        ("tuple edges", {"edges": (0.8, 0.2), "values": [0.2, 0.8]}),
        ("ndarray edges", {"edges": np.array([0.0, np.inf]),
                           "values": [0.2, 0.8]}),
        ("Series edges", {"edges": pd.Series([0.9, 0.1]),
                          "values": [0.2, 0.8]}),
    ]
    for label, params in forms:
        check(f"B-f {tag} {label} raises",
              raises(lambda params=params: call(live, params),
                     "SF-BL-11418") is not None)
    check(f"B-empty {tag} empty values raises ValueError naming values",
          raises(lambda: call(live, {"edges": [0.0, 0.5], "values": []}),
                 "values", "non-empty", "SF-BL-11418") is not None)

    # ---- C: message quality ------------------------------------------
    msg = raises(lambda: call(live, {"edges": [0.0, 0.5],
                                     "values": [0.2, 1.5, -0.3]}),
                 "values", "1.5", "-0.3", "[0, 1]", "SF-BL-11418")
    check(f"C1 {tag} range message lists offending values", msg is not None)
    msg = raises(lambda: call(live, {"edges": [0.8, 0.2],
                                     "values": [0.2, 0.8]}),
                 "edges", "0.8", "0.2", "monotonic", "SF-BL-11418")
    check(f"C2 {tag} monotonic message lists offending pair", msg is not None)
    msg = raises(lambda: call(live, {"edges": [0.0, 0.5],
                                     "values": [0.2, float("nan")]}),
                 "values", "finite", "SF-BL-11418")
    check(f"C3 {tag} non-finite message names finiteness", msg is not None)

    # ---- D: lawful fits bit-identical ---------------------------------
    lawful = [
        {"edges": [0.0, 0.5], "values": [0.2, 0.8]},
        {"edges": [0.0, 0.5], "values": [0.0, 1.0]},          # boundaries
        {"edges": [0.0, 0.5, 0.5], "values": [0.2, 0.5, 0.8]},  # equal edges
        {"edges": [-0.5, 0.5, 2.0], "values": [0.2, 0.5, 0.8]},  # sentinels
        {"edges": [0.0], "values": [0.42]},                    # single bucket
        {"edges": [], "values": [0.42]},                      # empty edges
        {"edges": (0.0, 0.25, 0.75, 1.0), "values": (0.1, 0.5, 0.9)},
        {"edges": np.array([0, 1]), "values": np.array([0, 1])},
        {"edges": np.array([0.0, 0.5], dtype=np.float32),
         "values": np.array([0.2, 0.8], dtype=np.float32)},
        {"edges": np.array([0, 5, 10], dtype=np.int64),
         "values": np.array([1, 0], dtype=np.uint8)},
        {"edges": pd.Series([0.0, 0.5]), "values": pd.Series([0.2, 0.8])},
        {"edges": [float(e) for e in np.linspace(0, 1, 11)],
         "values": [0.05 + 0.09 * i for i in range(10)]},      # decile-like
    ]
    panels = [P, np.array([0.0, 0.5, 1.0]), np.array([0.4999, 0.5001]),
              np.linspace(0, 1, 41)]
    ok = True
    for params in lawful:
        for panel in panels:
            if not same_bits(call(live, params, p=panel),
                             call(pris, params, p=panel)):
                ok = False
    check(f"D1 {tag} curated lawful fits bit-identical ({len(lawful)}x{len(panels)})", ok)

    rng = np.random.default_rng(11418)
    ok = True
    for _ in range(2000):
        n_v = int(rng.integers(1, 13))
        n_e = int(rng.integers(0, 13))
        edges = np.sort(rng.uniform(-0.2, 1.2, size=n_e))
        if n_e >= 3 and rng.random() < 0.3:
            edges[1] = edges[0]  # equal neighbours stay lawful
        vals = rng.uniform(0.0, 1.0, size=n_v)
        if rng.random() < 0.2:
            vals[0] = rng.choice([0.0, 1.0])
        panel = rng.uniform(0.0, 1.0, size=int(rng.integers(1, 25)))
        params = {"edges": edges, "values": vals}
        if not same_bits(call(live, params, p=panel),
                         call(pris, params, p=panel)):
            ok = False
            break
    check(f"D2 {tag} 2000 randomized lawful fits bit-identical", ok)

    # ---- E: composition with the SF-BL-11531 type screen --------------
    check(f"E1 {tag} string values still raise the 11531 type ValueError",
          raises(lambda: call(live, {"edges": [0.0, 0.5],
                                     "values": ["0.2", "1.5"]}),
                 "values", "SF-BL-11531") is not None)
    check(f"E2 {tag} bool values still raise the 11531 type ValueError",
          raises(lambda: call(live, {"edges": [0.0, 0.5],
                                     "values": [False, True]}),
                 "values", "SF-BL-11531") is not None)
    check(f"E3 {tag} string edges still raise the 11531 type ValueError",
          raises(lambda: call(live, {"edges": ["0.0", "0.5"],
                                     "values": [0.2, 0.8]}),
                 "edges", "SF-BL-11531") is not None)

    # ---- F: integration ------------------------------------------------
    df = pd.DataFrame({
        "price": [0.9, 0.1, 0.6, 0.3],
        "days_to_expiry": [7, 7, 14, 3],
        "log10_volume": [5.0, 6.0, 5.5, 4.5],
        "outcome": [1, 0, 1, 0],
    })
    check(f"F1 {tag} apply_rule clean fit bit-identical",
          same_bits(live.apply_rule("bucket_recal", df, CONTROL, 5.0, 1.0),
                    pris.apply_rule("bucket_recal", df, CONTROL, 5.0, 1.0)))
    check(f"F2 {tag} apply_rule corrupt fit raises 11418",
          raises(lambda: live.apply_rule(
              "bucket_recal", df, {"edges": [0.8, 0.2], "values": [0.2, 0.8]},
              5.0, 1.0), "edges", "SF-BL-11418") is not None)
    rng2 = np.random.default_rng(7)
    fit = pd.DataFrame({
        "price": rng2.uniform(0.02, 0.98, size=120),
        "days_to_expiry": rng2.integers(1, 30, size=120),
        "log10_volume": rng2.uniform(4.0, 7.0, size=120),
        "outcome": rng2.integers(0, 2, size=120),
    })
    fitted_live, zm, zs = live.fit_rules(fit)
    fitted_pris, _, _ = pris.fit_rules(fit)
    check(f"F3 {tag} fit_rules bucket params identical live vs pristine",
          fitted_live["bucket_recal"] == fitted_pris["bucket_recal"])
    check(f"F4 {tag} fit_rules bucket fit applies bit-identically",
          same_bits(call(live, fitted_live["bucket_recal"],
                         p=fit["price"].to_numpy()),
                    call(pris, fitted_pris["bucket_recal"],
                         p=fit["price"].to_numpy())))

# ---- G: patch scope ----------------------------------------------------
import subprocess  # noqa: E402
import sys  # noqa: E402

import tempfile  # noqa: E402

for name in ("predmarkets.py", "predmarkets_repo.py"):
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / name
        tmp.write_text((HERE / "pristine" / name).read_text())
        out = subprocess.run(
            [sys.executable, str(HERE / "apply_fix.py"), str(tmp)],
            capture_output=True, text=True)
        rebuilt = tmp.read_text()
    check(f"G0 {name} applicator on pristine copy reproduces patched/",
          "applied" in out.stdout
          and rebuilt == (HERE / "patched" / name).read_text())
    check(f"G0b {name} pristine snapshot untouched by applicator check",
          "_validate_bucket_recal_domain"
          not in (HERE / "pristine" / name).read_text())


def section(text):
    i = text.index("def _validate_bucket_recal_param")
    j = text.index("RULE_FUNCS", i)
    return text[i:j]


for live_path, patched_name in zip(LIVE, ("predmarkets.py", "predmarkets_repo.py")):
    check(f"G1 live bucket section == patched ({patched_name})",
          section(Path(live_path).read_text())
          == section((HERE / "patched" / patched_name).read_text()))
    check(f"G2 patched differs from pristine only by the domain screen ({patched_name})",
          "_validate_bucket_recal_domain" in (HERE / "patched" / patched_name).read_text()
          and "_validate_bucket_recal_domain" not in (HERE / "pristine" / patched_name).read_text())

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILURES:", *FAIL, sep="\n  ")
    raise SystemExit(1)
