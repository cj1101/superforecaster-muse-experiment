"""SF-BL-11198: Platt/Beta/Temperature predict must reject finite
out-of-range and NaN inputs with ValueError naming p, matching
IsotonicCalibrator (SF-BL-9719), instead of clipping them via _as_p
into lawful outputs. In-range predictions bit-identical.

Scope: finite out-of-range + NaN/inf value dimension only.
Bool predict inputs are SF-BL-11051 (already screened separately).
"""
import hashlib

import numpy as np
import pytest
from scipy.special import expit, logit

from superforecaster.calibration import (
    BetaCalibrator,
    IsotonicCalibrator,
    PlattScaler,
    TemperatureScaler,
    _as_p,
)

BAD = [[1.5], [-0.5], [0.5, 1.5], [1.0 + 1e-9], [-1e-9], [2.0],
       [float("nan")], [float("inf")], [float("-inf")], [0.5, float("nan")]]
BAD_SCALARS = [1.5, -0.5, float("nan")]
LAWFUL = np.array([0.0, 1.0, 0.5, 0.1, 0.9, 1e-9, 1 - 1e-9, 0.25, 0.75, 0.05, 0.95])
CLASSES = [PlattScaler, BetaCalibrator, TemperatureScaler, IsotonicCalibrator]


@pytest.fixture(scope="module")
def fit_data():
    rng = np.random.default_rng(0)
    p = np.clip(rng.beta(2, 2, size=200), 0.01, 0.99)
    y = (rng.random(200) < p).astype(int)
    return y, p


def _fitted(cls, y, p):
    return cls().fit(y, p)


@pytest.mark.parametrize("cls", CLASSES)
def test_predict_rejects_out_of_range_and_nan(cls, fit_data):
    y, p = fit_data
    m = _fitted(cls, y, p)
    for panel in BAD + BAD_SCALARS:
        with pytest.raises(ValueError, match="p"):
            m.predict(panel)
        with pytest.raises(ValueError, match="p"):
            m.transform(panel)


@pytest.mark.parametrize("cls", CLASSES)
def test_kill_panels_do_not_return(cls, fit_data):
    y, p = fit_data
    m = _fitted(cls, y, p)
    # Kill: predict([1.5]) must not return 1.0; predict([NaN]) must not return NaN
    with pytest.raises(ValueError):
        m.predict([1.5])
    with pytest.raises(ValueError):
        m.predict([float("nan")])


@pytest.mark.parametrize("cls", [PlattScaler, BetaCalibrator, TemperatureScaler])
def test_rejects_unfitted_and_intercept_only(cls, fit_data):
    with pytest.raises(ValueError):
        cls().predict([1.5])
    with pytest.raises(ValueError):
        cls().predict([float("nan")])
    y_const = np.array([0, 1] * 100)
    p_const = np.full(200, 0.7)
    m = cls().fit(y_const, p_const)
    assert m.intercept_only_
    with pytest.raises(ValueError):
        m.predict([1.5])
    with pytest.raises(ValueError):
        m.predict([-0.5])
    with pytest.raises(ValueError):
        m.predict([float("nan")])


def test_isotonic_step_mode_rejects(fit_data):
    y, p = fit_data
    m = IsotonicCalibrator(interpolation="step").fit(y, p)
    with pytest.raises(ValueError):
        m.predict([1.5])
    with pytest.raises(ValueError):
        m.predict([float("nan")])


def test_lawful_bit_identical_to_clip_formula(fit_data):
    y, p = fit_data
    # Platt
    m = PlattScaler().fit(y, p)
    assert m.predict(LAWFUL.copy()).tobytes() == expit(m.A_ * logit(_as_p(LAWFUL.copy())) + m.B_).tobytes()
    # Beta
    m = BetaCalibrator().fit(y, p)
    pc = _as_p(LAWFUL.copy())
    assert m.predict(LAWFUL.copy()).tobytes() == expit(m.c_ + m.a_ * np.log(pc) + m.b_ * np.log(1 - pc)).tobytes()
    # Temperature
    m = TemperatureScaler().fit(y, p)
    assert m.predict(LAWFUL.copy()).tobytes() == expit(logit(_as_p(LAWFUL.copy())) / m.T_).tobytes()
    # Endpoints remain lawful
    for cls in CLASSES:
        out = _fitted(cls, y, p).predict([0.0, 1.0])
        assert np.all(np.isfinite(out)) and np.all((out >= 0) & (out <= 1))


def test_error_names_offending_value(fit_data):
    y, p = fit_data
    for cls in CLASSES:
        m = _fitted(cls, y, p)
        with pytest.raises(ValueError, match=r"1\.5"):
            m.predict([1.5])
        with pytest.raises(ValueError, match="non-finite|finite"):
            m.predict([float("nan")])
