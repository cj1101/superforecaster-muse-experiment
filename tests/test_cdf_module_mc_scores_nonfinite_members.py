"""
test_cdf_module_mc_scores_nonfinite_members.py — SF-BL-10472 verification.

Preregistered criterion (TASK.md): "Fix raises a named ValueError on any
non-finite or out-of-[0,1] probability (and non-int true_bin),
regression-tested for NaN/inf members; kill if a NaN member still returns
a score pair or clean-panel scores change."

Bug: cdf_module.mc_scores([NaN, 0.5], true_bin=0) returned (nan, nan) on
the wave3 source-of-truth copy (max(NaN, 1e-12) and the Brier sum both
propagate NaN; a NaN sum defeats a sum-only screen); the live copies
(SF-BL-9925) screened finiteness but had no explicit per-member >1 check,
so [1.0000004, 0.0] (sum within the 1e-6 tolerance) returned a positive
log score. Distinct from SF-BL-9787-line true_bin=-1 / [-0.5, 1.5] cases.

Fix: finiteness screen first, explicit per-member [0, 1] screen, all
raises dual-tagged SF-BL-10472 + SF-BL-9925, applied identically to all
three live copies. Clean panels are bit-identical (hex values below
captured pre-fix; full 31-panel baseline in the task work dir).

Run: python3 tests/test_cdf_module_mc_scores_nonfinite_members.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metaculus_bot"))
import numpy as np
import pytest
import cdf_module as cm

NAN, INF = float("nan"), float("inf")

def _raises_named(probs, true_bin):
    with pytest.raises(ValueError) as excinfo:
        cm.mc_scores(probs, true_bin)
    msg = str(excinfo.value)
    assert "mc_scores" in msg and "SF-BL-10472" in msg and "SF-BL-9925" in msg, msg

def test_task_repro_nan_member_raises_named():
    _raises_named([NAN, 0.5], 0)

def test_nonfinite_member_any_kind_or_position_raises():
    for bad in ([NAN, 0.5], [0.5, NAN], [NAN], [0.2, NAN, 0.8],
                [INF, 0.0], [0.5, INF], [-INF, 1.0], [0.5, -INF, 0.5]):
        _raises_named(bad, 0)

def test_nonfinite_numpy_and_tuple_forms_raise():
    _raises_named(np.array([NAN, 0.5]), 0)
    _raises_named((INF, 0.0), 0)

def test_out_of_unit_interval_raises():
    for bad in ([1.0000004, 0.0], [1.5, 0.5], [2.0, 0.0], [-0.5, 1.5], [-1e-9, 0.5, 0.5]):
        _raises_named(bad, 0)

def test_nonint_true_bin_raises():
    for tb in (1.5, 1.0, "0", None, True, False):
        _raises_named([0.5, 0.5], tb)

def test_no_nan_member_returns_a_score_pair():
    try:
        cm.mc_scores([NAN, 0.5], true_bin=0)
    except ValueError:
        return
    raise AssertionError("kill: NaN member returned a score pair")

def test_clean_panels_bit_identical():
    panels = [
        ([0.2, 0.8], 1, "-0x1.c8ff7c79a9a20p-3", "0x1.47ae147ae147ap-4"),
        ([0.2, 0.8], 0, "-0x1.9c041f7ed8d33p+0", "0x1.47ae147ae147cp+0"),
        ([0.25, 0.25, 0.25, 0.25], 2, "-0x1.62e42fefa39efp+0", "0x1.8000000000000p-1"),
        ([0.0, 1.0], 0, "-0x1.ba18a998fffa0p+4", "0x1.0000000000000p+1"),
        ([1.0], 0, "0x0.0p+0", "0x0.0p+0"),
    ]
    for probs, tb, log_hex, brier_hex in panels:
        ls, br = cm.mc_scores(probs, tb)
        assert float(ls).hex() == log_hex and float(br).hex() == brier_hex

def test_sum_within_tolerance_and_np_integer_still_score():
    ls, _ = cm.mc_scores([0.5, 0.5000005], 0)
    assert ls == float(np.log(0.5))
    ls, _ = cm.mc_scores([0.2, 0.8], np.int64(1))
    assert float(ls).hex() == "-0x1.c8ff7c79a9a20p-3"

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} passed")
