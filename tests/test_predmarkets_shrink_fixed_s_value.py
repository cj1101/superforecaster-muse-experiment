"""SF-BL-11375 conformance suite: rule_shrink_fixed VALUE domain of s.

For both shipped copies of predmarkets.py:
  A. TASK fixtures s=2.0, -1.0, NaN (+ inf/-inf and other out-of-domain
     reals, incl. numpy/Fraction/int forms) raise a named ValueError
     mentioning s, rule_shrink_fixed and SF-BL-11375 — directly, via
     RULE_FUNCS dispatch and via apply_rule.
  B. Control s=0.925 -> [0.87, 0.13] bit-identical (struct-pack) to the
     verbatim pre-fix formula; all lawful s in [0, 1] (grid 0.70-1.00,
     boundaries 0/1, int/np/Fraction forms) bit-identical on curated,
     extreme and 500-point seeded panels.
  C. Property: for every lawful s and lawful panel, outputs lie in
     [0, 1] and ordering is never inverted.
  D. SF-BL-11495 type screen preserved ('0.5', None, True, False raise
     its ValueError).
  E. fit_rules smoke completes with the clipped grid and returns an
     in-[0, 1] s identical to a pristine-formula grid search clipped
     to [0, 1]; apply_rule integration with shipped fitted s=0.925.
  F. Kill criterion: no fixture returns a probability outside [0, 1]
     or an inverted forecast without an error (asserted in A/C).
Run: python3 test_predmarkets_shrink_fixed_s_value.py
"""
import struct
import sys
from fractions import Fraction

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/hatch/workspace/superforecaster_out/predmarkets")
sys.path.insert(0, "/home/hatch/workspace/superforecaster")

import importlib.util


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


COPIES = [
    load("pm_out", "/home/hatch/workspace/superforecaster_out/predmarkets/predmarkets.py"),
    load("pm_repo", "/home/hatch/workspace/superforecaster/superforecaster/predmarkets.py"),
]

PASS = 0
FAIL = 0


def check(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {label}")


def bits(arr):
    return [struct.pack("<d", float(x)) for x in np.asarray(arr, dtype=float)]


def prefix_formula(p, s):
    """Verbatim pre-fix computation (pristine rule body)."""
    return 0.5 + float(s) * (np.asarray(p, dtype=float) - 0.5)


BAD_S = [2.0, -1.0, float("nan"), float("inf"), float("-inf"),
         1.15, 1.025, 1.0000001, -0.0000001, -0.5, 100.0, -100.0,
         np.float64("nan"), np.float64(2.0), np.float64(-1.0),
         np.float32(2.0), Fraction(3, 2), Fraction(-1, 2),
         Fraction(10**30, 1), 2, -1]

GOOD_S = [0.925, 0.0, 1.0, 0.5, 0.7, 0.9999999, 1e-12,
          0, 1, np.float64(0.925), np.float32(0.5), np.int64(1),
          Fraction(1, 2), Fraction(37, 40), Fraction(0, 1), Fraction(1, 1)]

PANELS = [[0.9, 0.1], [0.001, 0.999], [0.0, 1.0], [0.5], [0.25, 0.75, 0.5, 0.05]]

for ci, m in enumerate(COPIES):
    tag = f"copy{ci}"
    # --- A: out-of-domain s raises the named SF-BL-11375 ValueError ---
    for s in BAD_S:
        for route in ("direct", "dispatch"):
            try:
                fn = m.rule_shrink_fixed if route == "direct" else m.RULE_FUNCS["shrink_fixed"]
                fn([0.9, 0.1], None, None, {"s": s})
                check(False, f"{tag} A {route} s={s!r} did not raise")
            except ValueError as e:
                msg = str(e)
                check("rule_shrink_fixed" in msg and "s" in msg and "SF-BL-11375" in msg,
                      f"{tag} A {route} s={s!r} unnamed error: {msg}")
            except Exception as e:  # noqa: BLE001
                check(False, f"{tag} A {route} s={s!r} wrong type {type(e).__name__}: {e}")
    # A2: via apply_rule (params deserialised path)
    df = pd.DataFrame({"price": [0.9, 0.1], "days_to_expiry": [7, 7],
                       "log10_volume": [5.0, 5.0]})
    for s in (2.0, -1.0, float("nan")):
        try:
            m.apply_rule("shrink_fixed", df, {"s": s}, 5.0, 1.0)
            check(False, f"{tag} A2 apply_rule s={s!r} did not raise")
        except ValueError as e:
            check("SF-BL-11375" in str(e), f"{tag} A2 apply_rule s={s!r}: {e}")
    # --- B: lawful s bit-identical to the verbatim pre-fix formula ---
    rng = np.random.default_rng(11375)
    big = rng.random(500)
    for s in GOOD_S:
        for p in PANELS + [big]:
            got = m.rule_shrink_fixed(p, None, None, {"s": s})
            check(bits(got) == bits(prefix_formula(p, s)),
                  f"{tag} B s={s!r} panel[:2]={list(np.asarray(p)[:2])} not bit-identical")
    for s in np.clip(np.arange(0.70, 1.151, 0.025), 0.0, 1.0):  # clipped fitted grid
        got = m.rule_shrink_fixed([0.9, 0.1], None, None, {"s": float(s)})
        check(bits(got) == bits(prefix_formula([0.9, 0.1], float(s))),
              f"{tag} B grid s={float(s)} not bit-identical")
    # B2: the TASK control, exactly
    got = m.rule_shrink_fixed([0.9, 0.1], None, None, {"s": 0.925})
    check(bits(got) == bits([0.87, 0.13]) or
          bits(got) == bits(prefix_formula([0.9, 0.1], 0.925)),
          f"{tag} B2 control mismatch: {list(got)}")
    check(abs(float(got[0]) - 0.87) < 1e-12 and abs(float(got[1]) - 0.13) < 1e-12,
          f"{tag} B2 control values: {list(got)}")
    # --- C: lawful outputs always in [0, 1], never inverted ---
    for s in GOOD_S + list(np.clip(np.arange(0.0, 1.025, 0.025), 0.0, 1.0)):
        out = m.rule_shrink_fixed(big, None, None, {"s": float(s)})
        check(np.all(out >= 0.0) and np.all(out <= 1.0),
              f"{tag} C s={float(s)} escaped [0, 1]")
        order_in = np.argsort(big, kind="stable")
        check(np.all(np.diff(out[order_in]) >= 0.0),
              f"{tag} C s={float(s)} inverted ordering")
    # --- D: SF-BL-11495 type screen preserved ---
    for s in ("0.5", None, True, False, [0.5], complex(0.5, 0.0)):
        try:
            m.rule_shrink_fixed([0.9, 0.1], None, None, {"s": s})
            check(False, f"{tag} D s={s!r} did not raise")
        except ValueError as e:
            check("SF-BL-11495" in str(e), f"{tag} D s={s!r} wrong msg: {e}")
    # --- E: fit_rules + apply_rule integration ---
    n = 240
    frng = np.random.default_rng(7)
    prices = np.clip(frng.normal(0.45, 0.25, n), 0.01, 0.99)
    frame = pd.DataFrame({
        "price": prices,
        "outcome": (frng.random(n) < prices).astype(float),
        "days_to_expiry": frng.integers(1, 30, n).astype(float),
        "log10_volume": frng.normal(5.0, 1.0, n),
    })
    fitted, _zm, _zs = m.fit_rules(frame)
    fs = fitted["shrink_fixed"]["s"]
    check(0.0 <= fs <= 1.0, f"{tag} E fitted s out of [0, 1]: {fs}")
    # independent clipped-grid search with the verbatim pre-fix formula
    y = frame["outcome"].to_numpy()
    best = (None, 1e18)
    for s in np.clip(np.arange(0.70, 1.151, 0.025), 0.0, 1.0):
        v = float(np.mean((prefix_formula(prices, float(s)) - y) ** 2))
        if v < best[1]:
            best = (float(s), v)
    check(fs == best[0], f"{tag} E fitted s {fs} != clipped-grid optimum {best[0]}")
    out = m.apply_rule("shrink_fixed", df, {"s": 0.925}, 5.0, 1.0)
    check(bits(out) == bits(prefix_formula([0.9, 0.1], 0.925)),
          f"{tag} E apply_rule control not bit-identical: {list(out)}")

print(f"{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
