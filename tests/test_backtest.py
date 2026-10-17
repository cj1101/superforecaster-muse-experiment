"""
test_backtest.py — runnable synthetic tests for backtest.py

Run:  python3 test_backtest.py
Prints real numbers (also captured into RESULTS.md by the author — numbers
in RESULTS.md are copied from an actual run of this file, not invented).

Tests
-----
1. Unit metrics: known Brier / AUC / ECE cases.
2. Signal series: latent AR(1) (phi=0.92) + alternating regime shifts.
   The observable signal x_t = latent_t + small noise is known at forecast
   issue time t (think: an order-flow / momentum indicator). The target is
   y_{t+h} ~ Bernoulli(sigmoid(latent_{t+h})). A causal momentum forecaster
   p = sigmoid(x_t * phi^h) SHOULD beat climatology decisively.
   (A y-only EWMA forecaster is also run for honesty: binary outcomes are
   so noisy that it only roughly matches climatology — see RESULTS.md.)
3. Pure noise: y iid Bernoulli(0.3), x iid noise independent of y -> the
   same forecaster must NOT credibly beat climatology (bootstrap CI
   includes 0 and/or deflated p >= 0.05).
4. Leakage: (a) assert_no_lookahead catches a future feature timestamp,
   (b) assert_not_identical_to catches feature == contemporaneous target,
   (c) a deliberately leaky forecaster (peeks at y[t+h]) visibly inflates
   Brier/AUC vs its causal twin.
5. Splits: purge/embargo gaps hold on every fold; rolling == fixed train size.
"""
import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'superforecaster'))  # repo layout bootstrap
from backtest import (
    LeakageError, WalkForwardBacktester, assert_no_lookahead,
    assert_not_identical_to, auc_mann_whitney, brier_score,
    deflated_brier_pvalue, evaluate_forecasts, ewma_forecaster, make_splits,
)

PASS = []
PHI = 0.92


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def signal_momentum_forecaster(y_train, X_train=None, X_test=None, horizon=1):
    """Causal momentum: p(y_{t+h}=1) = sigmoid(x_t * phi^h).
    Uses ONLY X at issue time t (the X_test row) — no future data."""
    x = np.asarray(X_test, dtype=float)[:, 0]
    return np.clip(sigmoid(x * (PHI ** horizon)), 0.01, 0.99)


# ---- 1. unit metrics ------------------------------------------------------
def test_units():
    print("\n== 1. Unit metrics ==")
    check("brier perfect", abs(brier_score([0, 1], [0.0, 1.0]) - 0.0) < 1e-12)
    check("brier coinflip", abs(brier_score([0, 1], [0.5, 0.5]) - 0.25) < 1e-12)
    check("auc perfect", abs(auc_mann_whitney([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) - 1.0) < 1e-12)
    check("auc reversed", abs(auc_mann_whitney([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) - 0.0) < 1e-12)
    check("auc chance-ish", abs(auc_mann_whitney([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) - 0.5) < 1e-12)


# ---- generators ------------------------------------------------------------
def make_signal_series(n=1500, seed=7):
    """Latent AR(1) (phi=0.92) around a mean that regime-shifts every 250
    steps, alternating +1.1 / -1.1 (deterministic alternation = known
    structure, balanced climatology). x_t = latent_t + N(0, 0.25) is the
    observable signal at time t; y_t ~ Bernoulli(sigmoid(latent_t))."""
    rng = np.random.default_rng(seed)
    latent = np.zeros(n); regime = np.zeros(n)
    for i, start in enumerate(range(0, n, 250)):
        regime[start:start + 250] = 1.1 if i % 2 == 0 else -1.1
    eps = rng.normal(0, 0.55, n)
    for t in range(1, n):
        latent[t] = regime[t] + PHI * (latent[t - 1] - regime[t - 1]) + eps[t]
    x = latent + rng.normal(0, 0.25, n)
    y = (rng.random(n) < sigmoid(latent)).astype(float)
    return y, x.reshape(-1, 1)


def make_noise_series(n=1500, seed=11):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(float)
    x = rng.normal(0, 1, n).reshape(-1, 1)  # independent of y
    return y, x


# ---- 2/3. backtests ---------------------------------------------------------
def run_case(name, y, X, forecaster, label="model"):
    bt = WalkForwardBacktester(initial_train=300, test_size=50, step=50,
                               purge=5, embargo=5, n_boot=1000, seed=0)
    res = bt.run(y, forecaster, X=X, horizons=(1, 5), n_variants_tried=3)
    m, clim = res.metrics, res.benchmarks["climatology"]
    boot = res.bootstrap_vs["climatology"]
    print(f"\n== {name} ==")
    print(f"n_test={m['n']} base_rate={m['base_rate']:.4f}")
    print(f"{label}: Brier={m['brier']:.4f} log={m['log_score']:.4f} "
          f"ECE={m['ece']:.4f} AUC={m['auc']:.4f} hit={m['hit_rate']:.4f}")
    print(f"climatology Brier={clim['brier']:.4f} log={clim['log_score']:.4f} AUC={clim['auc']:.4f}")
    for bname, bm in res.benchmarks.items():
        print(f"  bench {bname:12s} Brier={bm['brier']:.4f}")
    print(f"Brier diff (clim - model) = {boot['diff_brier_bench_minus_model']:.4f} "
          f"95% CI [{boot['ci95_lo']:.4f}, {boot['ci95_hi']:.4f}] p(not better)={boot['p_model_not_better']:.4f}")
    print(f"per-horizon Brier model: { {h: round(v['metrics']['brier'],4) for h,v in res.per_horizon.items()} } "
          f"clim: { {h: round(v['benchmarks']['climatology']['brier'],4) for h,v in res.per_horizon.items()} }")
    print(res.multiple_testing_note)
    return res, boot


# ---- 4. leakage --------------------------------------------------------------
def test_leakage(y, X):
    print("\n== 4. Leakage tests ==")
    caught_a = False
    try:
        assert_no_lookahead(np.array([10, 20, 30]), np.array([10, 20, 31]), horizon=1)
    except LeakageError as e:
        caught_a = True; print(f"  (a) caught: {e}")
    check("leakage-a future timestamp caught", caught_a)

    caught_b = False
    try:
        assert_not_identical_to(y, y, name="leaky_feature")
    except LeakageError as e:
        caught_b = True; print(f"  (b) caught: {e}")
    check("leakage-b identical feature caught", caught_b)

    # (c) Leaky "forecaster": p = actual future outcome (0.99/0.01), scored on
    # exactly the test indices the causal run used. This is NOT caught by the
    # timestamp assertions (it never declares timestamps) — it is the inflation
    # you get when a leak bypasses the guards, which is why both layers exist
    # and why suspiciously perfect Brier is itself a red flag.
    bt = WalkForwardBacktester(initial_train=300, test_size=50, step=50,
                               purge=5, embargo=5, n_boot=200, seed=0)
    causal = bt.run(y, signal_momentum_forecaster, X=X, horizons=(1,))
    y_true = causal.y_true
    leaky_metrics = evaluate_forecasts(y_true, np.where(y_true == 1, 0.99, 0.01))
    print(f"  (c) causal Brier={causal.metrics['brier']:.4f} AUC={causal.metrics['auc']:.4f} | "
          f"LEAKY Brier={leaky_metrics['brier']:.4f} AUC={leaky_metrics['auc']:.4f}")
    check("leakage-c leaky visibly inflates",
          leaky_metrics["brier"] < 0.05 and causal.metrics["brier"] > 0.10,
          f"(leaky {leaky_metrics['brier']:.4f} vs causal {causal.metrics['brier']:.4f})")
    print("  VERDICT: (a)/(b) are CAUGHT by assertion; (c) is caught only by")
    print("  recognising impossible performance — guards + scepticism, both.")


# ---- 5. splits ---------------------------------------------------------------
def test_splits():
    print("\n== 5. Split integrity ==")
    sp = make_splits(1000, initial_train=200, test_size=50, purge=10, embargo=7)
    ok = all((s.test_min - s.train_max - 1) >= 7 for s in sp)
    check("purge/embargo gap >= embargo on all folds", ok, f"({len(sp)} folds)")
    spr = make_splits(1000, initial_train=200, test_size=50, window="rolling",
                      rolling_train=150, purge=0, embargo=0)
    sizes = {len(s.train_idx) for s in spr}
    check("rolling train size constant", sizes == {150}, f"(sizes={sizes})")
    spe = make_splits(1000, initial_train=200, test_size=50, window="expanding")
    check("expanding train grows", len(spe[-1].train_idx) > len(spe[0].train_idx))


if __name__ == "__main__":
    test_units()
    test_splits()

    y_sig, X_sig = make_signal_series()
    res_sig, boot_sig = run_case(
        "2. SIGNAL (AR(1)+regimes) — signal-momentum should beat climatology",
        y_sig, X_sig, signal_momentum_forecaster)
    check("signal: model Brier < climatology", res_sig.metrics["brier"] < res_sig.benchmarks["climatology"]["brier"])
    check("signal: bootstrap CI excludes 0 (favouring model)", boot_sig["ci95_lo"] > 0,
          f"CI=[{boot_sig['ci95_lo']:.4f},{boot_sig['ci95_hi']:.4f}]")

    # Honesty check: y-only EWMA on the same series (no signal feature).
    res_ew, _ = run_case("2b. SIGNAL, y-only EWMA (no X) — expected: weak",
                         y_sig, None, ewma_forecaster(0.05), label="ewma_yonly")
    print(f"  -> EWMA Brier {res_ew.metrics['brier']:.4f} vs climatology "
          f"{res_ew.benchmarks['climatology']['brier']:.4f}: binary outcomes alone "
          f"carry little extractable signal at this noise level.")

    y_noise, X_noise = make_noise_series()
    res_n, boot_n = run_case(
        "3. NOISE (iid Bern 0.3, X independent) — must NOT credibly beat climatology",
        y_noise, X_noise, signal_momentum_forecaster)
    p_adj = deflated_brier_pvalue(max(boot_n["p_model_not_better"], 1e-9), 3)
    sig_claim = (boot_n["ci95_lo"] > 0) and (p_adj < 0.05)
    check("noise: no credible skill claim (CI incl. 0 or deflated p>=0.05)", not sig_claim,
          f"CI=[{boot_n['ci95_lo']:.4f},{boot_n['ci95_hi']:.4f}] p_adj={p_adj:.4f}")

    test_leakage(y_sig, X_sig)

    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
