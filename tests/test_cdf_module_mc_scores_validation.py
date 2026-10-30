"""
test_cdf_module_mc_scores_validation.py — SF-BL-9925 verification.

Pre-registered done/kill criteria (TASK.md):
  "Fix passes when mc_scores() raises a named ValueError for true_bin
   outside [0, len(probs)), negative/non-finite probs, or probs not
   summing to 1 within tolerance, with valid controls bit-identical;
   kill if a caller relies on -1 as a documented last-bin shorthand
   (then convert it explicitly at that call site)."

Bug: metaculus_bot/cdf_module.py mc_scores() validated nothing.
Negative indexing made mc_scores([0.2, 0.8], -1) return exactly the
genuine last-bin score (log -0.2231, Brier 0.08), so a corrupt -1
label was indistinguishable from a real last-bin resolution; and
mc_scores([-0.5, 1.5], 0) returned Brier 4.5, above the 2.0 maximum
for any valid probability vector. Kill criterion checked and NOT met:
no caller uses -1 — every live call site computes true_bin as
int(np.sum(y > edges)) in [0, len(probs)) (wave3 evaluate.py,
SF-BL-2531 restate_wave3.py), an enumerate index (SF-BL-3337), or a
bounded rng draw (SF-BL-4666); metaculus_client.py only re-exports.

Fix (cdf_module.py, mc_scores): reject-only screens before the
unchanged scoring computation — non-numeric / non-1-D / empty probs,
non-finite probs, negative probs, |sum - 1| > 1e-6 (fleet SUM_TOL for
multiple-choice distributions, cf. distribution_validity.py), and a
non-integer or out-of-range true_bin each raise a ValueError naming
mc_scores and SF-BL-9925. Valid controls are bit-identical: expected
float.hex() values below were captured live from the pre-fix
implementation (full 33-panel baseline in the task work dir's
baseline_prefix.json / verify_fix.py).

Run:  python3 tests/test_cdf_module_mc_scores_validation.py
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


def _raises_named(probs, true_bin):
    with pytest.raises(ValueError) as excinfo:
        cm.mc_scores(probs, true_bin)
    msg = str(excinfo.value)
    assert "mc_scores" in msg and "SF-BL-9925" in msg, msg


def test_task_repro_negative_true_bin_raises_named():
    _raises_named([0.2, 0.8], -1)


def test_task_repro_off_simplex_probs_raise_named():
    _raises_named([-0.5, 1.5], 0)


def test_true_bin_out_of_range_raises():
    for tb in (-2, -1, 2, 99):
        _raises_named([0.2, 0.8], tb)
    _raises_named([1.0], 1)
    _raises_named([1.0], -1)
    _raises_named([0.25] * 4, -1)


def test_true_bin_non_integer_raises():
    for tb in (1.0, 0.5, "1", None, True, False):
        _raises_named([0.2, 0.8], tb)


def test_negative_or_nonfinite_probs_raise():
    for bad in ([-0.5, 1.5], [0.5, -0.1, 0.6], [-1e-9, 0.5, 0.5],
                [NAN, 1.0], [0.5, NAN, 0.5], [INF, 0.0], [0.5, INF]):
        _raises_named(bad, 0)


def test_probs_not_summing_to_one_raise():
    for bad in ([0.5, 0.6], [0.2, 0.2], [2.0], [0.0, 0.0],
                [0.5, 0.500002], [0.5, 0.499998], [1 / 3, 1 / 3]):
        _raises_named(bad, 0)


def test_structural_bad_probs_raise():
    for bad, tb in (([], 0), (0.5, 0), ([[0.5, 0.5]], 0),
                    (["a", "b"], 0), (None, 0)):
        _raises_named(bad, tb)


def test_sum_within_tolerance_still_scores():
    ls, br = cm.mc_scores([0.5, 0.5000005], 0)  # dev 5e-7 < 1e-6
    assert ls == float(np.log(0.5))


def test_valid_controls_bit_identical():
    # (probs, true_bin, log.hex(), brier.hex()) captured pre-fix.
    panels = [
        ([0.2, 0.8], 1, "-0x1.c8ff7c79a9a20p-3", "0x1.47ae147ae147ap-4"),
        ([0.2, 0.8], 0, "-0x1.9c041f7ed8d33p+0", "0x1.47ae147ae147cp+0"),
        ([0.25, 0.25, 0.25, 0.25], 2, "-0x1.62e42fefa39efp+0", "0x1.8000000000000p-1"),
        ([0.0, 1.0], 0, "-0x1.ba18a998fffa0p+4", "0x1.0000000000000p+1"),
        ([1.0], 0, "0x0.0p+0", "0x0.0p+0"),
    ]
    for probs, tb, log_hex, brier_hex in panels:
        ls, br = cm.mc_scores(probs, tb)
        assert float(ls).hex() == log_hex, (probs, tb, ls)
        assert float(br).hex() == brier_hex, (probs, tb, br)


def test_numpy_integer_true_bin_accepted():
    ls, br = cm.mc_scores([0.2, 0.8], np.int64(1))
    assert float(ls).hex() == "-0x1.c8ff7c79a9a20p-3"


def test_valid_brier_within_bounds_random():
    rng = np.random.default_rng(9925)
    for _ in range(200):
        k = int(rng.integers(2, 10))
        p = rng.dirichlet(np.ones(k))
        for t in range(k):
            _, br = cm.mc_scores(p, t)
            assert 0.0 <= br <= 2.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} passed")
