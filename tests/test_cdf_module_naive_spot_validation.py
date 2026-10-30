"""
test_cdf_module_naive_spot_validation.py — SF-BL-10393 verification.

Pre-registered done/kill criterion (TASK.md):
  "Fix raises a named ValueError for non-finite or non-positive spot
   before fitting, regression-tested for NaN/inf/0/-100; kill if a
   corrupt spot still yields percentile output or clean-spot outputs
   change by more than 1e-12."

Bug: cdf_module.naive_normal_percentiles screened its returns window
(finite filter, >= 30 obs) and horizon_days (SF-BL-10267) but never
spot, so with 60 finite daily returns and horizon 7 spot=NaN returned
an all-NaN percentile dict and spot=0 a degenerate all-zero dict
(also confirmed live: +/-inf all-inf dicts, -100 reversed
percentiles, None/'abc' bare TypeError/UFuncTypeError). Distinct
from climatology_percentiles' spot screen (SF-BL-10266) in a
different function.

Fix (metaculus_bot/cdf_module.py, naive_normal_percentiles): spot is
converted to float and screened FIRST — non-finite, <= 0, or
non-numeric spot raises a ValueError naming spot and SF-BL-10393.
The Gaussian computation is untouched (spot_f is the identical
float), so clean-spot outputs are bit-identical, which implies the
<=1e-12 kill bound. Expected bytes below were captured live from
the pre-fix implementation (work dir verify_fix.py covers all 13
panels on both module copies; this file pins the canonical copy).

Run:  python3 tests/test_cdf_module_naive_spot_validation.py
      or pytest tests/test_cdf_module_naive_spot_validation.py
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
RETS60 = np.random.default_rng(42).normal(0.0, 0.02, 60)


def _raises_named(spot, rets=RETS60, horizon=7):
    with pytest.raises(ValueError) as excinfo:
        cm.naive_normal_percentiles(spot, rets, horizon)
    msg = str(excinfo.value)
    assert "spot" in msg and "SF-BL-10393" in msg, msg
    assert "naive_normal_percentiles" in msg, msg


def test_task_regression_set_raises_named():
    _raises_named(NAN)      # pre-fix: all-NaN dict (task repro)
    _raises_named(INF)      # pre-fix: all-inf dict
    _raises_named(0)        # pre-fix: degenerate all-zero dict (task repro)
    _raises_named(-100)     # pre-fix: reversed percentiles


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
    for bad in (NAN, INF, -INF, 0, 0.0, -100, None, "abc"):
        try:
            out = cm.naive_normal_percentiles(bad, RETS60, 7)
        except ValueError:
            continue
        raise AssertionError(f"spot={bad!r} returned {out!r} "
                             "instead of raising")


def test_spot_screened_before_fitting_screens():
    # Bad spot + too-short window / bad horizon: error must name spot.
    _raises_named(NAN, rets=[0.01] * 5)
    _raises_named(0.0, horizon=NAN)


def test_existing_screens_preserved_for_lawful_spot():
    with pytest.raises(ValueError) as excinfo:
        cm.naive_normal_percentiles(100.0, [0.01] * 29, 7)
    assert ">= 30" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        cm.naive_normal_percentiles(100.0, RETS60, NAN)
    assert "horizon_days" in str(excinfo.value)
    assert "SF-BL-10267" in str(excinfo.value)


def test_clean_panels_bit_identical():
    rng = np.random.default_rng(10393)
    r60 = rng.normal(0.001, 0.02, 60)
    r100 = rng.normal(0.001, 0.05, 100)
    mixed = np.concatenate([r60, [np.nan, np.inf, -np.inf]])
    panels = {
        "task60_spot100_h7": ((100.0, r60, 7, None),
                              "eb57ed7b9c3157408bb32961513e58404fef8b75390059407e88e07630c859400fc94283e1f25a40"),
        "spot_small_h7": ((0.01, r60, 7, None),
                          "77f48b551700833f83a5833237dc833f6597bd59107b843f913aa205e01e853f4ae88b309013863f"),
        "spot_big_h5": ((12345.678, r100, 5, None),
                        "8e10134b45e6c340458a592eb619c640fae359faaac5c7403ca1fcc8fe91c94075090e4a0066cc40"),
        "exact30_h7": ((250.0, np.linspace(-0.04, 0.06, 30), 7, None),
                       "7e93e144405e6d402a557307a1bf6f4034adaa6008c27040736d5ffdd7b0714052ca5ab8eb1f7340"),
        "custom_probs_h7": ((100.0, r60, 7, (0.1, 0.5, 0.9)),
                            "75d23300d39457404fef8b75390059408502711880815a40"),
        "returns_with_nonfinite_h7": ((100.0, mixed, 7, None),
                                      "eb57ed7b9c3157408bb32961513e58404fef8b75390059407e88e07630c859400fc94283e1f25a40"),
        "int_spot_h7": ((100, r60, 7, None),
                        "eb57ed7b9c3157408bb32961513e58404fef8b75390059407e88e07630c859400fc94283e1f25a40"),
    }
    for name, ((spot, rets, h, probs), expected_vals_hex) in panels.items():
        d = (cm.naive_normal_percentiles(spot, rets, h) if probs is None
             else cm.naive_normal_percentiles(spot, rets, h, probs))
        vals = np.array(list(d.values()))
        assert vals.tobytes().hex() == expected_vals_hex, name
        assert np.all(np.isfinite(vals)) and np.all(np.diff(vals) > 0), name


def test_clean_outputs_within_1e12_of_prefix():
    # Explicit kill-bound check on the task panel (bit-identical above
    # already implies it; this asserts the registered bound directly).
    rng = np.random.default_rng(10393)
    r60 = rng.normal(0.001, 0.02, 60)
    d = cm.naive_normal_percentiles(100.0, r60, 7)
    ref = np.frombuffer(bytes.fromhex(
        "eb57ed7b9c3157408bb32961513e58404fef8b7539005940"
        "7e88e07630c859400fc94283e1f25a40"), dtype=np.float64)
    assert float(np.max(np.abs(np.array(list(d.values())) - ref))) <= 1e-12


def test_lawful_output_feeds_valid_cdf():
    lawful = cm.naive_normal_percentiles(100.0, RETS60, 7)
    cdf = cm.percentiles_to_cdf(lawful, 50.0, 200.0)
    assert cm.validate_cdf(cdf) == []


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
