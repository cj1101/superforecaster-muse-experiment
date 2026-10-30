"""
test_cdf_module_climatology_spot_validation.py — SF-BL-10266 verification.

Pre-registered done/kill criteria (TASK.md):
  "Fix raises ValueError naming spot for any non-finite or <=0 spot
   while finite positive-spot outputs stay bit-identical; kill if any
   corrupt-spot call still returns a percentile dict."

Bug: cdf_module.climatology_percentiles screened the trailing returns
(non-finite filtered, >= 30 required) but never screened spot, so with
40 finite trailing returns spot=NaN returned an all-NaN percentile
dict, spot=+inf an all-inf dict, and spot=-100 returned percentiles in
reversed order — all then consumed by percentiles_to_cdf.

Fix (metaculus_bot/cdf_module.py, climatology_percentiles): spot is
converted to float and screened FIRST — non-finite, <= 0, or
non-numeric spot raises a ValueError naming spot and SF-BL-10266. The
quantile computation is untouched, so finite positive-spot outputs are
bit-identical: expected raw bytes below were captured live from the
pre-fix implementation (work dir verify_fix.py covers all 11 panels
on both module copies; this file pins the canonical copy).

Run:  python3 tests/test_cdf_module_climatology_spot_validation.py
No pytest dependency beyond raises: test_* functions with assert +
__main__ runner, same convention as
tests/test_cdf_module_bin_probs_validation.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metaculus_bot"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import cdf_module as cm  # noqa: E402

NAN, INF = float("nan"), float("inf")
RNG = np.random.default_rng(0)
RETS40 = RNG.normal(0.0, 0.02, 40)


def _raises_named(spot, rets=RETS40):
    with pytest.raises(ValueError) as excinfo:
        cm.climatology_percentiles(spot, rets)
    msg = str(excinfo.value)
    assert "spot" in msg and "SF-BL-10266" in msg, msg
    assert "climatology_percentiles" in msg, msg


def test_task_repros_raise_named():
    _raises_named(NAN)      # pre-fix: all-NaN dict
    _raises_named(INF)      # pre-fix: all-inf dict
    _raises_named(-100.0)   # pre-fix: reversed percentiles


def test_nonfinite_spots_raise():
    for bad in (NAN, INF, -INF, np.float64("nan"), np.float64("inf")):
        _raises_named(bad)


def test_nonpositive_spots_raise():
    for bad in (0.0, -0.0, -1.0, -100.0, -1e-9, -1e300):
        _raises_named(bad)


def test_nonnumeric_spots_raise_named_not_bare_typeerror():
    for bad in (None, "abc", []):
        _raises_named(bad)


def test_no_corrupt_spot_returns_a_dict():
    for bad in (NAN, INF, -INF, 0.0, -100.0, None, "abc"):
        try:
            out = cm.climatology_percentiles(bad, RETS40)
        except ValueError:
            continue
        raise AssertionError(f"spot={bad!r} returned {out!r} "
                             "instead of raising")


def test_spot_screened_before_returns_screen():
    # Bad spot + too-short window: the error must name spot.
    _raises_named(NAN, rets=[0.01] * 5)


def test_returns_screen_preserved_for_lawful_spot():
    with pytest.raises(ValueError) as excinfo:
        cm.climatology_percentiles(100.0, [0.01] * 29)
    assert ">= 30" in str(excinfo.value)


def test_clean_panels_bit_identical():
    rng = np.random.default_rng(10266)
    r_norm = rng.normal(0.0, 0.02, 40)
    r_norm100 = rng.normal(0.001, 0.05, 100)
    mixed = np.concatenate([r_norm, [np.nan, np.inf, -np.inf]])
    panels = {
        "task40_spot100": ((100.0, r_norm, None),
                           "b69c2e165f1d5840a3bf4c1a419c584011e2b589c9dc5840c3f0d13d2c235940d9ae27a281ce5940"),
        "spot_big": ((12345.678, r_norm100, None),
                     "614e19a5902fc640623f49e76840c7401058a964d407c8400fe6409c93b3c840c35d6c62ac22ca40"),
        "spot_small": ((0.01, r_norm, None),
                       "43d917d339c1833f71c197172b29843f9de4f6a1085e843ff1dafb8db197843f1cbd90ce0c24853f"),
        "exact30": ((250.0, np.linspace(-0.04, 0.06, 30), None),
                    "b9957ae4d72c6e4041329740e5c86e40b40e00be66906f40b48df1947a2e7040f0bbe49629827040"),
        "custom_probs": ((100.0, r_norm, (0.1, 0.5, 0.9)),
                         "0f2c5142f82f584011e2b589c9dc5840b1b6f1ba49985940"),
        "returns_with_nonfinite": ((100.0, mixed, None),
                                   "b69c2e165f1d5840a3bf4c1a419c584011e2b589c9dc5840c3f0d13d2c235940d9ae27a281ce5940"),
        "int_spot": ((100, r_norm, None),
                     "b69c2e165f1d5840a3bf4c1a419c584011e2b589c9dc5840c3f0d13d2c235940d9ae27a281ce5940"),
    }
    for name, ((spot, rets, probs), expected_vals_hex) in panels.items():
        d = (cm.climatology_percentiles(spot, rets) if probs is None
             else cm.climatology_percentiles(spot, rets, probs))
        vals = np.array(list(d.values()))
        assert vals.tobytes().hex() == expected_vals_hex, name
        assert np.all(np.isfinite(vals)) and np.all(np.diff(vals) > 0), name


def test_disciplined_percentiles_inherits_spot_screen():
    daily = RNG.normal(0.0, 0.01, 60)
    recent = RNG.normal(0.0, 0.012, 30)
    for bad in (NAN, INF, -5.0, 0.0):
        with pytest.raises(ValueError) as excinfo:
            cm.disciplined_percentiles(bad, RETS40, daily, recent)
        assert "spot" in str(excinfo.value)
    disc = cm.disciplined_percentiles(100.0, RETS40, daily, recent)
    vals = np.array(list(disc.values()))
    assert np.all(np.isfinite(vals)) and np.all(np.diff(vals) > 0)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed.append(fn.__name__)
            print(f"FAIL  {fn.__name__}: {exc!r}")
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
