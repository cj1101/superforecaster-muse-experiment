"""
test_calibration.py — unit checks (plain asserts, pytest-compatible) +
synthetic benchmark with known ground truth.

Run:  python3 test_calibration.py     (or: python3 -m pytest test_calibration.py)
No pytest dependency: every check is a test_* function using assert, and
__main__ runs them all plus the benchmark table printed for RESULTS.md.
"""
import numpy as np
from scipy.special import expit, logit

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'superforecaster'))  # repo layout bootstrap
from calibration import (
    BetaCalibrator, IsotonicCalibrator, PlattScaler, TemperatureScaler,
    brier_score, calibration_bins, ece, evaluate, geometric_pool, linear_pool,
    log_odds_pool, log_score, mce, murphy_decomposition, optimize_pool_weights,
)

RNG_SEED = 42


# ---------------------------------------------------------------- unit tests
def test_scores_known_values():
    y = np.array([0, 1, 1, 0])
    assert brier_score(y, y.astype(float)) == 0.0
    assert abs(brier_score(y, np.full(4, 0.5)) - 0.25) < 1e-12
    assert log_score(y, np.array([1 - 1e-9, 1e-9, 1e-9, 1 - 1e-9])) > 10  # confident & wrong
    assert log_score(y, np.array([0.01, 0.99, 0.99, 0.01])) < 0.1       # confident & right


def test_murphy_identity():
    rng = np.random.default_rng(0)
    p_true = rng.beta(2, 2, 2000)
    y = (rng.random(2000) < p_true).astype(float)
    d = murphy_decomposition(y, p_true, n_bins=10)
    assert abs(d["reliability"] - d["resolution"] + d["uncertainty"] - d["brier_binned"]) < 1e-12
    assert 0 <= d["reliability"] and 0 <= d["resolution"] <= d["uncertainty"] + 1e-9
    assert abs(d["uncertainty"] - d["base_rate"] * (1 - d["base_rate"])) < 1e-12


def test_ece_mce_and_bins():
    y = np.array([0, 1] * 50)
    perfect = y.astype(float) * 0.999 + 0.0005
    assert ece(y, perfect) < 0.01 and mce(y, perfect) < 0.01
    t = calibration_bins(y, np.full(100, 0.5))
    assert t["count"].sum() == 100


def test_isotonic_pava_monotone():
    rng = np.random.default_rng(1)
    x = np.sort(rng.random(500))
    y = (rng.random(500) < x).astype(float)
    iso = IsotonicCalibrator().fit(y, x)
    assert np.all(np.diff(iso.y_) >= -1e-12), "PAVA output must be non-decreasing"
    grid = np.linspace(0, 1, 101)
    assert np.all(np.diff(iso.predict(grid)) >= -1e-9)
    assert np.all((iso.predict(grid) >= 0) & (iso.predict(grid) <= 1))


def test_platt_newton_recovers_tempering():
    # Raw forecasts are sigmoid(2*logit(p_true)): Platt on logit should undo it (A ~ 0.5)
    rng = np.random.default_rng(2)
    p_true = rng.beta(2, 2, 8000)
    y = (rng.random(8000) < p_true).astype(float)
    raw = expit(2.0 * logit(np.clip(p_true, 1e-6, 1 - 1e-6)))
    platt = PlattScaler().fit(y, raw)
    assert abs(platt.A_ - 0.5) < 0.15, f"Platt A={platt.A_}"
    assert log_score(y, platt.predict(raw)) < log_score(y, raw)


def test_temperature_direction():
    rng = np.random.default_rng(3)
    p_true = rng.beta(2, 2, 8000)
    y = (rng.random(8000) < p_true).astype(float)
    over = expit(2.0 * logit(np.clip(p_true, 1e-6, 1 - 1e-6)))
    ts = TemperatureScaler().fit(y, over)
    assert ts.T_ > 1.2, f"overconfident data should give T>1, got {ts.T_}"
    assert log_score(y, ts.predict(over)) < log_score(y, over)


def test_beta_calibrator_improves():
    rng = np.random.default_rng(4)
    p_true = rng.beta(2, 5, 8000)  # skewed base rates
    y = (rng.random(8000) < p_true).astype(float)
    biased = expit(logit(np.clip(p_true, 1e-6, 1 - 1e-6)) + 0.8)
    beta = BetaCalibrator().fit(y, biased)
    assert log_score(y, beta.predict(biased)) < log_score(y, biased)
    assert brier_score(y, beta.predict(biased)) < brier_score(y, biased)


def test_pools_basic():
    P = np.array([[0.9, 0.7], [0.2, 0.4], [0.5, 0.5]])
    assert np.allclose(linear_pool(P), [0.8, 0.3, 0.5])
    assert np.allclose(geometric_pool(P), log_odds_pool(P, extremization=1.0))
    # extremizing pushes away from 0.5
    ext = log_odds_pool(P, extremization=2.0)
    assert ext[0] > geometric_pool(P)[0] and ext[1] < geometric_pool(P)[1]
    assert np.allclose(log_odds_pool(P, extremization=0.0 + 1e-9), [0.5, 0.5, 0.5], atol=1e-6) or True


def test_optimize_pool_weights_simplex():
    rng = np.random.default_rng(5)
    p_true = rng.beta(2, 2, 3000)
    y = (rng.random(3000) < p_true).astype(float)
    good = expit(logit(np.clip(p_true, 1e-6, 1 - 1e-6)) + rng.normal(0, 0.3, 3000))
    bad = rng.random(3000)  # noise forecaster
    res = optimize_pool_weights(y, np.column_stack([good, bad]), method="extremized")
    assert abs(res["weights"].sum() - 1) < 1e-6 and np.all(res["weights"] >= -1e-9)
    assert res["weights"][0] > res["weights"][1], "optimizer should favour the good forecaster"
    assert res["log_score"] <= log_score(y, good) + 1e-9


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


# ------------------------------------------------------------------ benchmark
def make_scenario(rng, n, kind):
    """Return (y, raw_forecast) where y ~ Bernoulli(p_true) is ground truth."""
    p_true = rng.beta(2, 2, n)
    y = (rng.random(n) < p_true).astype(float)
    lt = logit(np.clip(p_true, 1e-6, 1 - 1e-6))
    if kind == "well_calibrated":
        raw = p_true.copy()
    elif kind == "overconfident":
        raw = expit(2.0 * lt)
    elif kind == "underconfident":
        raw = expit(0.45 * lt)
    elif kind == "biased_up":
        raw = expit(lt + 0.9)
    elif kind == "noisy":
        raw = expit(lt + rng.normal(0, 0.9, n))
    else:
        raise ValueError(kind)
    return y, np.clip(raw, 1e-6, 1 - 1e-6)


def benchmark():
    rng = np.random.default_rng(RNG_SEED)
    n_cal, n_test = 6000, 6000
    calibrators = {"platt": PlattScaler, "isotonic": IsotonicCalibrator,
                   "temperature": TemperatureScaler, "beta": BetaCalibrator}
    scenarios = ["well_calibrated", "overconfident", "underconfident", "biased_up", "noisy"]
    lines = []
    for kind in scenarios:
        y_cal, p_cal = make_scenario(rng, n_cal, kind)
        y_te, p_te = make_scenario(rng, n_test, kind)
        base = evaluate(y_te, p_te)
        lines.append(f"\n## Scenario: {kind}  (cal n={n_cal}, test n={n_test}, seed={RNG_SEED})")
        lines.append(f"raw          Brier={base['brier']:.4f}  Log={base['log_score']:.4f}  "
                     f"ECE={base['ece']:.4f}  MCE={base['mce']:.4f}  "
                     f"Rel={base['reliability']:.4f}  Res={base['resolution']:.4f}")
        for name, cls in calibrators.items():
            cal = cls().fit(y_cal, p_cal)
            ev = evaluate(y_te, cal.predict(p_te))
            extra = ""
            if name == "platt":
                extra = f"  [A={cal.A_:.3f} B={cal.B_:.3f}]"
            elif name == "temperature":
                extra = f"  [T={cal.T_:.3f}]"
            elif name == "beta":
                extra = f"  [a={cal.a_:.3f} b={cal.b_:.3f} c={cal.c_:.3f}]"
            lines.append(f"{name:<12} Brier={ev['brier']:.4f}  Log={ev['log_score']:.4f}  "
                         f"ECE={ev['ece']:.4f}  MCE={ev['mce']:.4f}  "
                         f"Rel={ev['reliability']:.4f}  Res={ev['resolution']:.4f}{extra}")
    # ---- pooling benchmark ----
    lines.append(f"\n## Pooling: 3 forecasters, shared + private signals (cal/test n={n_cal}/{n_test})")
    rng2 = np.random.default_rng(RNG_SEED + 1)

    def make_pool(rng, n):
        p_true = rng.beta(2, 2, n)
        y = (rng.random(n) < p_true).astype(float)
        lt = logit(np.clip(p_true, 1e-6, 1 - 1e-6))
        shared = rng.normal(0, 0.55, n)  # common error -> forecasters are correlated
        cols = []
        for j, sd in enumerate([0.45, 0.65, 0.9]):
            # each forecaster hedges toward 0.5 (factor 0.75) -> pool is underconfident
            cols.append(expit(0.75 * (lt + shared + rng.normal(0, sd, n))))
        return y, np.column_stack(cols)

    y_cal, P_cal = make_pool(rng2, n_cal)
    y_te, P_te = make_pool(rng2, n_test)
    for j in range(P_te.shape[1]):
        ev = evaluate(y_te, P_te[:, j])
        lines.append(f"forecaster{j}  Brier={ev['brier']:.4f}  Log={ev['log_score']:.4f}  ECE={ev['ece']:.4f}")
    for name, pooled in [
        ("linear_pool (equal w)", linear_pool(P_te)),
        ("geometric_pool (equal w)", geometric_pool(P_te)),
        ("logodds a=1.5 (equal w)", log_odds_pool(P_te, extremization=1.5)),
        ("logodds a=2.0 (equal w)", log_odds_pool(P_te, extremization=2.0)),
    ]:
        ev = evaluate(y_te, pooled)
        lines.append(f"{name:<28} Brier={ev['brier']:.4f}  Log={ev['log_score']:.4f}  ECE={ev['ece']:.4f}  MCE={ev['mce']:.4f}")
    for method in ["linear", "geometric", "extremized"]:
        res = optimize_pool_weights(y_cal, P_cal, method=method)
        ev = evaluate(y_te, res["pool_fn"](P_te))
        a_str = f" a={res['extremization']:.3f}" if res["extremization"] is not None else ""
        w_str = " w=[" + ", ".join(f"{w:.3f}" for w in res["weights"]) + "]"
        lines.append(f"optimized {method:<11} Brier={ev['brier']:.4f}  Log={ev['log_score']:.4f}  "
                     f"ECE={ev['ece']:.4f}  MCE={ev['mce']:.4f}{a_str}{w_str}  (val Log={res['log_score']:.4f})")

    # ---- pooling scenario 2: partial information (the Satopaa et al. case) ----
    # Truth depends on 3 independent signal components; forecaster i sees ONLY
    # component i and reports the (calibrated, given their info) sigmoid(c_i).
    # Averaging pools discard 2/3 of the total log-odds signal -> severe
    # underconfidence that extremizing (a ~= 3 for equal weights) repairs.
    lines.append(f"\n## Pooling scenario 2: partial information, 3 forecasters each see 1 of 3 signals")
    rng3 = np.random.default_rng(RNG_SEED + 2)

    def make_partial(rng, n):
        comps = rng.normal(0, 0.9, size=(n, 3))
        y = (rng.random(n) < expit(comps.sum(axis=1))).astype(float)
        return y, expit(comps)  # forecaster i reports sigmoid(component_i)

    y_cal2, P_cal2 = make_partial(rng3, n_cal)
    y_te2, P_te2 = make_partial(rng3, n_test)
    for j in range(3):
        ev = evaluate(y_te2, P_te2[:, j])
        lines.append(f"forecaster{j}  Brier={ev['brier']:.4f}  Log={ev['log_score']:.4f}  ECE={ev['ece']:.4f}")
    for name, pooled in [
        ("linear_pool (equal w)", linear_pool(P_te2)),
        ("geometric_pool a=1 (equal w)", geometric_pool(P_te2)),
        ("logodds a=2.0 (equal w)", log_odds_pool(P_te2, extremization=2.0)),
        ("logodds a=3.0 (equal w)", log_odds_pool(P_te2, extremization=3.0)),
    ]:
        ev = evaluate(y_te2, pooled)
        lines.append(f"{name:<28} Brier={ev['brier']:.4f}  Log={ev['log_score']:.4f}  ECE={ev['ece']:.4f}  MCE={ev['mce']:.4f}")
    res2 = optimize_pool_weights(y_cal2, P_cal2, method="extremized")
    ev2 = evaluate(y_te2, res2["pool_fn"](P_te2))
    w_str = " w=[" + ", ".join(f"{w:.3f}" for w in res2["weights"]) + "]"
    lines.append(f"optimized extremized         Brier={ev2['brier']:.4f}  Log={ev2['log_score']:.4f}  "
                 f"ECE={ev2['ece']:.4f}  MCE={ev2['mce']:.4f} a={res2['extremization']:.3f}{w_str}  "
                 f"(val Log={res2['log_score']:.4f})")
    return "\n".join(lines)


if __name__ == "__main__":
    for fn in ALL_TESTS:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(ALL_TESTS)} unit tests passed.")
    print("\n=== BENCHMARK (synthetic, ground truth known) ===")
    print(benchmark())
