"""
test_cdf_module_nan_prob_key.py — SF-BL-10425 verification tests.

Pre-registered done/kill criteria (TASK.md):
  "Fix rejects any non-finite percentile probability with a named
   ValueError and a regression test passes; kill if NaN-prob input
   still yields any CDF output."

Bug: cdf_module.percentiles_to_cdf({NaN: 10, 0.5: 20, 0.9: 30}, 0, 100)
returned a 201-point CDF whose entries were all NaN (NaN-headed at
the first grid points), because _validate_percentiles' diff and
(0, 1) range comparisons are all False for a NaN prob, so the NaN
probability key passed unscreened and np.interp propagated it.
Distinct from SF-BL-10107, whose live repro was a NaN percentile
VALUE. The bug was live in the superforecaster_out copy of
cdf_module.py, which lacked any finiteness screen; the repo copy
was already covered as a side effect of SF-BL-10107's screen.

Fix (metaculus_bot/cdf_module.py, _validate_percentiles, BOTH the
repo copy and the live superforecaster_out copy): a dedicated
np.isfinite screen on the probability keys BEFORE any comparison,
raising ValueError naming the non-finite percentile probability
key(s) (SF-BL-10425). The live copy additionally received the
SF-BL-10107 non-finite-VALUES screen verbatim (it lacked both).
Finite panels are untouched: the expected SHA-256 hashes below are
the pre-fix hashes (identical in both copies pre-fix) and post-fix
outputs must match them exactly.

Run:  python3 tests/test_cdf_module_nan_prob_key.py
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_cdf_module_nonfinite_percentiles.py.
"""
import hashlib
import importlib.util
import os
import sys

import numpy as np

PASS = []
NAN = float("nan")
INF = float("inf")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WS = os.path.dirname(_ROOT)
_MODULE_PATHS = [
    ("repo", os.path.join(_ROOT, "metaculus_bot", "cdf_module.py")),
    ("live-out", os.path.join(
        _WS, "superforecaster_out", "metaculus_bot", "cdf_module.py")),
]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(
        f"cdf_module_{name.replace('-', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MODULES = [(name, _load(name, path)) for name, path in _MODULE_PATHS]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _raises_named(fn, *needles):
    try:
        fn()
    except ValueError as e:
        msg = str(e)
        assert all(n in msg for n in needles), \
            f"message {msg!r} missing {needles}"
        return msg
    raise AssertionError("no ValueError raised (a CDF was yielded)")


# ---- 1. Task repro + non-finite probability keys, every position ---------

def test_task_repro_nan_prob_key_raises():
    for label, cm in MODULES:
        msg = _raises_named(
            lambda cm=cm: cm.percentiles_to_cdf(
                {NAN: 10, 0.5: 20, 0.9: 30}, 0, 100),
            "finite", "percentile probabilit", "SF-BL-10425")
        assert "lie in (0, 1)" not in msg, (label, msg)


def test_nan_prob_any_position_raises():
    for label, cm in MODULES:
        for panel in ({NAN: 10, 0.5: 20, 0.9: 30},
                      {0.1: 5, NAN: 20, 0.9: 30},
                      {0.1: 5, 0.5: 20, NAN: 30},
                      {NAN: 10, NAN: 20, 0.9: 30}):
            _raises_named(
                lambda p=panel, cm=cm: cm.percentiles_to_cdf(p, 0, 100),
                "finite", "percentile probabilit")


def test_nan_prob_key_types_raise():
    for label, cm in MODULES:
        for bad_key in (NAN, np.nan, np.float64("nan"), np.float32("nan")):
            _raises_named(
                lambda k=bad_key, cm=cm: cm.percentiles_to_cdf(
                    {k: 10, 0.5: 20, 0.9: 30}, 0, 100),
                "finite", "percentile probabilit")


def test_inf_prob_raises_named_finite():
    # Pre-fix (live copy), +/-inf probs raised only via the (0, 1)
    # message, which does not name finiteness; the fix must name it.
    for label, cm in MODULES:
        for bad in (INF, -INF):
            msg = _raises_named(
                lambda b=bad, cm=cm: cm.percentiles_to_cdf(
                    {b: 10, 0.5: 20, 0.9: 30}, 0, 100),
                "finite", "percentile probabilit", "SF-BL-10425")
            assert "lie in (0, 1)" not in msg, (label, msg)


def test_validate_percentiles_direct_raises():
    for label, cm in MODULES:
        for panel in ({NAN: 10, 0.5: 20}, {0.1: 5, NAN: 20},
                      {INF: 10, 0.5: 20}, {-INF: 10, 0.5: 20}):
            _raises_named(
                lambda p=panel, cm=cm: cm._validate_percentiles(p),
                "finite", "percentile probabilit")


def test_nan_prob_raises_for_pchip_too():
    for label, cm in MODULES:
        for interp in ("linear", "pchip"):
            _raises_named(
                lambda i=interp, cm=cm: cm.percentiles_to_cdf(
                    {NAN: 10, 0.5: 20, 0.9: 30}, 0, 100, interp=i),
                "finite", "percentile probabilit")


def test_kill_criterion_no_cdf_yielded():
    # Kill if NaN-prob input still yields ANY CDF output: every call
    # below must raise; _raises_named fails if a value is returned.
    for label, cm in MODULES:
        for panel in ({NAN: 10, 0.5: 20, 0.9: 30},
                      {0.1: 5, NAN: 20, 0.9: 30},
                      {NAN: 0, 0.5: 10, 0.9: 20}):
            for lo, hi in ((0, 100), (0, 30)):
                _raises_named(
                    lambda p=panel, lo=lo, hi=hi, cm=cm:
                    cm.percentiles_to_cdf(p, lo, hi),
                    "finite")


def test_nan_value_still_raises_10107_guard():
    # Same screen region: a NaN percentile VALUE (SF-BL-10107 scope)
    # must also raise a named-finite ValueError in both copies.
    for label, cm in MODULES:
        _raises_named(
            lambda cm=cm: cm.percentiles_to_cdf(
                {0.1: NAN, 0.5: 20, 0.9: 30}, 0, 100),
            "finite", "percentile value")


# ---- 2. Valid panels bit-identical (pre-fix hashes, both copies) ---------

# (name, panel, range_min, range_max, zero_point, interp, sha256 of CDF bytes)
PANELS = [
    ("valid-0-100", {0.1: 10, 0.5: 20, 0.9: 30}, 0, 100, None, "linear",
     "414897c7e0cae43d951cca255834dc0abc2c7c476e5dd3d68ad6014a52de1f7c"),
    ("valid-0-100-pchip", {0.1: 10, 0.5: 20, 0.9: 30}, 0, 100, None, "pchip",
     "c0426c9b778d67b614d7ef592de18fcb08bbf7878c60d4e17be3efee51d4d261"),
    ("task-valid-0-30", {0.1: 5, 0.5: 10, 0.9: 20}, 0, 30, None, "linear",
     "3d1b186d416e50e2feac16b247071f216de7d0f1adc6bf964145e5ebf10c6b19"),
    ("default5", {0.05: 2, 0.25: 7, 0.5: 12, 0.75: 18, 0.95: 27},
     0, 30, None, "linear",
     "2c0e96a7f51a37acdffb6b11fef2ae8b319d675672adc05cec46aefbb3d20ff4"),
    ("dup-values", {0.1: 10, 0.5: 10, 0.9: 20}, 0, 30, None, "linear",
     "734c4d725d7015916fadcabb4439f8729314a78f6ebf94b51a56b1d82cfdb00e"),
    ("logscale", {0.1: 5, 0.5: 10, 0.9: 20}, 1, 100, 0.0, "linear",
     "d2d83907982c8c11f45b845fd767bd39e35d8cec0c602059fcfcd8bd174a1bcd"),
    ("two-point", {0.25: 3, 0.75: 9}, 0, 12, None, "linear",
     "2b2c0a7fcd668d9a4319dc42c98a3027c3900df66b49a2c1dd7afc589851a289"),
]


def test_valid_panels_bit_identical():
    for label, cm in MODULES:
        for name, panel, lo, hi, zp, interp, want in PANELS:
            cdf = cm.percentiles_to_cdf(
                panel, lo, hi, zero_point=zp, interp=interp)
            got = hashlib.sha256(np.asarray(cdf).tobytes()).hexdigest()
            assert got == want, f"{label}/{name}: {got} != {want}"
            assert len(cdf) == 201 and np.all(np.isfinite(cdf)), \
                f"{label}/{name}"


# ---- 3. Pre-existing validations unchanged --------------------------------

def test_existing_validations_still_raise():
    for label, cm in MODULES:
        for panel in ({0.5: 10},
                      {0.0: 5, 0.5: 10},
                      {0.5: 10, 1.0: 20},
                      {0.1: 20, 0.5: 10, 0.9: 15}):
            _raises_named(
                lambda p=panel, cm=cm: cm.percentiles_to_cdf(p, 0, 30))


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for fn in ALL_TESTS:
        try:
            fn()
            check(fn.__name__, True)
        except AssertionError as e:
            check(fn.__name__, False, str(e))
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
