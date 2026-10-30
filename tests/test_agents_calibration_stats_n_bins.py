"""SF-BL-9786 — agents.calibration_stats must validate n_bins.

Pre-fix, calibration_stats([0.5, 0.6], [1, 0], n_bins=0) returned
{'ece': 0.0, 'mean_forecast': 0.55, 'base_rate': 0.5}: the bin loop
never executed, so a zero-bin ECE of exactly 0.0 read as perfect
calibration in any dashboard or gate consuming the result. Negative
n_bins failed only incidentally inside np.linspace and non-integer
n_bins raised TypeError. These tests pin the fixed contract — ValueError
for n_bins < 1 or non-integer n_bins, raised before looping — and that
valid calls (incl. the default n_bins=10) are bit-identical to the
pre-fix values. Mirrors calibration._validate_n_bins (SF-BL-9722).
"""
import numpy as np
import pytest

from superforecaster.agents import calibration_stats

P, O = [0.5, 0.6], [1, 0]
BAD = [0, -1, -3, -100, 2.5, 10.0, 0.5, "10", None, True, False,
       float("nan"), float("inf")]


@pytest.mark.parametrize("bad", BAD)
def test_rejects_bad_n_bins(bad):
    with pytest.raises(ValueError, match="SF-BL-9786"):
        calibration_stats(P, O, n_bins=bad)


def test_task_repro_kill_criterion():
    # Kill criterion: n_bins=0 must NOT still return ece 0.0.
    with pytest.raises(ValueError):
        calibration_stats([0.5, 0.6], [1, 0], n_bins=0)


def test_valid_outputs_bit_identical():
    assert calibration_stats(P, O, n_bins=10) == {
        "ece": 0.050000000000000044, "mean_forecast": 0.55, "base_rate": 0.5}
    assert calibration_stats(P, O) == calibration_stats(P, O, n_bins=10)
    assert calibration_stats(P, O, n_bins=np.int64(10)) == \
        calibration_stats(P, O, n_bins=10)
    rng = np.random.default_rng(42)
    p = rng.random(500)
    o = (rng.random(500) < p).astype(float)
    expected = {1: 0.020318662124609033, 2: 0.020318662124609033,
                5: 0.02078486256369779, 10: 0.030530131421808784,
                20: 0.06820173124738259}
    for nb, ece in expected.items():
        assert calibration_stats(p, o, n_bins=nb)["ece"] == ece


def test_validation_precedes_binning_and_panel_use():
    # n_bins is validated first: even an otherwise-invalid panel must
    # surface the n_bins ValueError, proving the check runs before looping.
    with pytest.raises(ValueError, match="SF-BL-9786"):
        calibration_stats([], [], n_bins=0)
