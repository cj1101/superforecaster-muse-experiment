"""SF-BL-10191 — extremize() must reject non-finite p and centre.

Pre-fix, extremize(float("nan")) returned exactly 0.0: np.clip passes
NaN through, the logit is NaN, and sigmoid's `x > -700` overflow guard
is False for NaN, so sigmoid emits its underflow-branch 0.0 — a corrupt
pool member laundered into a maximally confident forecast. centre=nan
also returned 0.0, centre=+inf returned 0.0877 (silently clipped to
1-EPS, i.e. near-certainty), and p=±inf was silently clipped to the EPS
bounds. Factor non-finiteness was already screened by SF-BL-8875; these
tests pin p/centre to the same named-ValueError contract and pin
finite-input behaviour as bit-identical to the pre-fix values.
"""
import numpy as np
import pytest

from superforecaster.agents import extremize

NONFINITE = [float("nan"), float("inf"), float("-inf")]
NONNUMERIC = ["0.8", None, True]


@pytest.mark.parametrize("bad", NONFINITE + NONNUMERIC)
def test_extremize_rejects_nonfinite_p(bad):
    with pytest.raises(ValueError, match="SF-BL-10191"):
        extremize(bad)


@pytest.mark.parametrize("bad", NONFINITE + NONNUMERIC)
def test_extremize_rejects_nonfinite_centre(bad):
    with pytest.raises(ValueError, match="SF-BL-10191"):
        extremize(0.8, centre=bad)


@pytest.mark.parametrize("bad", NONFINITE)
def test_extremize_still_rejects_nonfinite_factor(bad):
    # SF-BL-8875 screen, retained: extremize(0.8, factor=inf) pre-dated
    # this task returning exactly 1.0 in the unvalidated snapshot.
    with pytest.raises(ValueError, match="SF-BL-8875"):
        extremize(0.8, factor=bad)


def test_task_repros_now_raise():
    with pytest.raises(ValueError):
        extremize(float("nan"))            # pre-fix: exactly 0.0
    with pytest.raises(ValueError):
        extremize(0.8, centre=float("nan"))  # pre-fix: exactly 0.0
    with pytest.raises(ValueError):
        extremize(0.8, centre=float("inf"))  # pre-fix: 0.08766614510558131
    with pytest.raises(ValueError):
        extremize(float("inf"))            # pre-fix: 0.9999999841510476


def test_finite_controls_unchanged():
    assert extremize(0.8) == 0.8584144223701332
    assert extremize(0.8, factor=1.00) == 0.8
    assert extremize(0.8, factor=1.40) == 0.8744413346659732
    assert extremize(0.5) == 0.5
    assert extremize(np.float64(0.8)) == 0.8584144223701332
    # Finite endpoints keep their pre-fix clipping behaviour.
    assert extremize(0.0) == extremize(1e-6)
    assert extremize(1.0) == extremize(1 - 1e-6)
    assert extremize(0.8, centre=0.3) == 0.8865898949467702
