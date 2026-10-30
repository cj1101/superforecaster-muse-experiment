"""SF-BL-11126 — aggregate() must type-validate extremize_factor under its own name.

On the pre-guard source, aggregate(forecasts 0.7/0.6, strategy=
'extremized', extremize_factor=True) returned 0.6517 — bit-identical
to factor 1.0, extremizing silently disabled (vs 0.6930 at the 1.30
default) — and '1.3'/None died with bare TypeErrors inside extremize.
SF-BL-8875 added a pre-dispatch screen, but it delegates to
_validate_extremize_factor, whose messages name `factor` (extremize's
argument), never `extremize_factor`. SF-BL-11126 wraps that screen for
aggregate so every rejection names `extremize_factor`, and closes the
array-carried hole in the shared screen: np.array(True) returned the
factor-1.0 result and np.array('1.3') the parsed 1.30 result.
Numeric factors (incl. 0-d / size-1 numeric arrays) stay bit-identical.
"""
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from superforecaster.agents import AgentForecast, aggregate, extremize

BAD = [
    True, False, np.bool_(True),
    "1.3", b"1.3", bytearray(b"1.3"), memoryview(b"1.3"),
    None, [1.3], (1.3,), {"x": 1.3}, {1.3}, complex(1.3, 0.0),
    np.array(True), np.array([True]), np.array("1.3"), np.array(["1.3"]),
    np.array([True], dtype=object), np.array("1.3", dtype=object),
]
STRATS = ["extremized", "weighted_extremized"]


def _fc(probs):
    return [AgentForecast(f"a{i}", "q", float(p)) for i, p in enumerate(probs)]


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.parametrize("s", STRATS)
def test_aggregate_rejects_bad_type_naming_extremize_factor(bad, s):
    with pytest.raises(ValueError, match="extremize_factor"):
        aggregate(_fc([0.7, 0.6]), strategy=s, extremize_factor=bad)
    with pytest.raises(ValueError, match="extremize_factor"):
        aggregate(_fc([0.7, 0.6]), s, None, bad)  # positional


def test_task_kill_cases():
    for kwargs_bad in (True, "1.3", None):
        with pytest.raises(ValueError, match="extremize_factor"):
            aggregate(_fc([0.7, 0.6]), strategy="extremized",
                      extremize_factor=kwargs_bad)  # pre-guard True: 0.6517
    assert aggregate(_fc([0.7, 0.6]), strategy="extremized",
                     extremize_factor=1.0) == 0.6516685226452116
    assert aggregate(_fc([0.7, 0.6]), strategy="extremized") == 0.6930241845603738


def test_other_rejections_also_named():
    for bad in (-1.0, 0.0, 1.41, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="extremize_factor"):
            aggregate(_fc([0.7, 0.6]), strategy="weighted_extremized",
                      extremize_factor=bad)


def test_numeric_factors_unchanged():
    for f in (1.00, 1.15, 1.30, 1.40, 1, np.float64(1.3),
              Decimal("1.3"), Fraction(13, 10),
              np.array(1.3), np.array([1.3]), np.array([1.3], dtype=object)):
        assert aggregate(_fc([0.7, 0.6]), strategy="extremized",
                         extremize_factor=f) == \
            aggregate(_fc([0.7, 0.6]), strategy="extremized",
                      extremize_factor=float(f))
    # extremize()'s own contract still names `factor`
    with pytest.raises(ValueError, match="factor"):
        extremize(0.8, factor=np.array(True))
    assert extremize(0.8) == 0.8584144223701332
