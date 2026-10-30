"""SF-BL-11090: string pool weights must raise, never silently pool.

_weights=['0.5', '0.5']_ on [[0.7, 0.3]] returned 0.5 from all three
pools pre-fix because _norm_weights coerced strings through
np.asarray(dtype=float). Now _reject_string_weights screens the raw
weights inside _norm_weights before any float conversion.
"""
import numpy as np
import pytest

from superforecaster.calibration import (
    geometric_pool,
    linear_pool,
    log_odds_pool,
)

POOLS = [linear_pool, geometric_pool, log_odds_pool]
PANEL = [[0.7, 0.3]]


@pytest.mark.parametrize("pool", POOLS)
@pytest.mark.parametrize("weights", [
    ["0.5", "0.5"], ("0.5", "0.5"), [0.5, "0.5"], "0.5",
    np.array(["0.5", "0.5"]), [b"0.5", b"0.5"],
    np.array(["0.5", "0.5"], dtype=object), ["a", "b"],
])
def test_string_weights_raise_naming_weights(pool, weights):
    with pytest.raises(ValueError, match="weights"):
        pool(PANEL, weights=weights)


@pytest.mark.parametrize("pool", POOLS)
def test_numeric_weights_unchanged(pool):
    assert pool(PANEL, weights=[0.5, 0.5]) == pytest.approx([0.5])
    assert pool(PANEL, weights=[1, 0]) == pytest.approx([0.7])
    assert pool(PANEL, weights=None) == pytest.approx([0.5])
