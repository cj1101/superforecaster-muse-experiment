"""SF-BL-11239: make_crash_label threshold type/finiteness validation.

Prereg: make_crash_label raises ValueError naming threshold on any
bool, string, None or non-finite threshold, with finite
numeric-threshold label series bit-identical; kill if threshold=True
still returns an all-crash series or threshold=nan an all-zero series.

Distinct from SF-BL-9849, which fixtures this function's horizon and
the threshold *domain* (-1, 0]; the domain guard is preserved here.
"""
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from superforecaster.crypto import make_crash_label


def _panel():
    closes = [100.0] * 50
    closes[20] = 85.0  # a >10% drop inside the 7d window for rows 14..20
    return pd.DataFrame(
        {"close": closes},
        index=pd.date_range("2024-01-01", periods=50, freq="D"),
    )


def _original_labels(df, horizon=7, threshold=-0.10):
    """Pre-fix computation, verbatim, for bit-identity checks."""
    c = df["close"].astype(float)
    fut_min = c.iloc[::-1].rolling(horizon, min_periods=horizon).min().iloc[::-1].shift(-1)
    lab = (fut_min / c - 1.0 < threshold).astype(float)
    lab[fut_min.isna()] = np.nan
    return lab


BAD_THRESHOLDS = [
    True, False, np.bool_(True), np.bool_(False),
    np.array(True), np.array(False),
    "0.1", "-0.1", b"-0.1", None, pd.NA,
    float("nan"), float("inf"), float("-inf"),
    np.nan, np.float64("nan"), Decimal("NaN"), Decimal("Infinity"),
    [-0.1], (-0.1,), np.array([-0.1]), -0.1 + 0j,
]


@pytest.mark.parametrize("bad", BAD_THRESHOLDS)
def test_bad_threshold_raises_valueerror_naming_threshold(bad):
    with pytest.raises(ValueError, match="threshold"):
        make_crash_label(_panel(), threshold=bad)


def test_kill_criteria_explicit():
    # threshold=True must NOT return an all-crash series; nan must NOT
    # return an all-zero series — both must raise.
    with pytest.raises(ValueError, match="threshold"):
        make_crash_label(_panel(), threshold=True)
    with pytest.raises(ValueError, match="threshold"):
        make_crash_label(_panel(), threshold=float("nan"))


@pytest.mark.parametrize("thr", [-0.10, -0.05, -0.2, -0.5, 0, 0.0, -1e-9,
                                 np.float64(-0.10), np.array(-0.10),
                                 Decimal("-0.1")])
def test_lawful_threshold_bit_identical(thr):
    df = _panel()
    got = make_crash_label(df, threshold=thr)
    want = _original_labels(df, threshold=float(thr))
    pd.testing.assert_series_equal(got, want, check_exact=True)


def test_default_threshold_bit_identical():
    df = _panel()
    pd.testing.assert_series_equal(
        make_crash_label(df), _original_labels(df), check_exact=True)


def test_domain_guard_preserved_9849():
    for thr in (-1.0, -1.5, 0.1, 1.0):
        with pytest.raises(ValueError, match="threshold"):
            make_crash_label(_panel(), threshold=thr)
