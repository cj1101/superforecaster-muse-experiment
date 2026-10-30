"""SF-BL-11092 — extremize() must reject finite out-of-range centres.

Pre-fix, extremize(0.8, centre=1.5) and extremize(0.8, centre=82.0) both
returned 0.08766614510558131, bit-identical to centre=1.0:
_validate_extremize_endpoint screened type (SF-BL-10191/11091) and
finiteness but had no range screen, so _extremize_core clipped any
finite centre to [EPS, 1-EPS] — a percent-unit or otherwise corrupt
centre silently extremized the pooled forecast around near-certainty,
while the lawful centre=0.9 returned 0.7582328207948962. These tests
pin the centre-only [0, 1] screen: any finite centre outside [0, 1]
raises a ValueError naming centre, and in-range centres (endpoints
included) are bit-identical to the pre-fix computation.
"""
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from superforecaster.agents import extremize

OUT_OF_RANGE = [1.5, 82.0, -0.2, 1.000000001, -1e-09, 2, 50.0, 1e6,
                np.float64(1.5), Decimal("1.5"), Fraction(3, 2)]


@pytest.mark.parametrize("bad", OUT_OF_RANGE)
def test_extremize_rejects_out_of_range_centre(bad):
    with pytest.raises(ValueError, match="centre") as exc:
        extremize(0.8, centre=bad)
    assert "SF-BL-11092" in str(exc.value)


def test_out_of_range_centre_positional():
    with pytest.raises(ValueError, match="centre"):
        extremize(0.8, 1.3, 1.5)


def test_task_kill_cases():
    # Pre-fix both returned 0.08766614510558131 (the 0.0877 of the TASK).
    with pytest.raises(ValueError, match="SF-BL-11092"):
        extremize(0.8, centre=1.5)
    with pytest.raises(ValueError, match="SF-BL-11092"):
        extremize(0.8, centre=82.0)


def test_in_range_centres_unchanged():
    assert extremize(0.8, centre=0.5) == 0.8584144223701332
    assert extremize(0.8, centre=0.9) == 0.7582328207948962
    assert extremize(0.8, centre=0.3) == 0.8865898949467702
    assert extremize(0.8) == 0.8584144223701332
    assert extremize(0.8, centre=np.float64(0.9)) == 0.7582328207948962
    # Endpoints stay lawful with their pre-fix clipping behaviour.
    assert extremize(0.8, centre=1.0) == 0.08766614510558131
    assert extremize(0.8, centre=0.0) == 0.9973927160344346
    assert extremize(0.8, centre=-0.0) == 0.9973927160344346


def test_prior_screens_keep_precedence():
    # Non-finite centres still raise the SF-BL-10191 finite error.
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="SF-BL-10191"):
            extremize(0.8, centre=bad)
    # Bool / non-numeric centres still raise naming centre.
    for bad in (True, "0.5", bytearray(b"0.5"), None):
        with pytest.raises(ValueError, match="centre"):
            extremize(0.8, centre=bad)
