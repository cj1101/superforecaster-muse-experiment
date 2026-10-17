# Workstream B — Backtesting Engine: RESULTS

Files: `backtest.py`, `test_backtest.py` (this file is the third deliverable).
Stack: numpy 1.26.4 / pandas 2.1.4 / stdlib only. Python 3.12.3.
**All numbers below are copied from an actual run of `python3 test_backtest.py`
on 2026-10-17: 14/14 checks passed, exit code 0.** Nothing is invented.

## Design decisions

1. **Binary probabilistic forecasts are the unit of evaluation.** `y ∈ {0,1}`,
   `p ∈ [0,1]`. Proper scoring rules are primary — **Brier** and **log score** —
   because they reward honest, calibrated probabilities. AUC and hit-rate are
   diagnostics only (a model can rank well and still be badly calibrated).
2. **Walk-forward only.** Expanding or rolling windows via `make_splits`;
   no shuffled K-fold anywhere (shuffling a time series *is* leakage).
3. **Forecast convention:** test indices are *issue times* `t`; a forecast
   issued at `t` with horizon `h` is scored against `y[t+h]`, using only
   data with index ≤ t. Forecasters receive only the training slice plus the
   X rows at issue times — the harness never hands them future y.
4. **Block-constant folds:** within a test block the forecaster sees the
   training slice, not outcomes inside the block. For strictly online
   evaluation use `test_size=1` (slower, exact). This is a deliberate,
   documented trade-off, not a hidden leak.
5. **Benchmarks are mandatory:** every run also scores climatology
   (train base rate), persistence (last outcome shrunk to base rate), and
   random-walk/no-change (EWMA probability held constant). A model that
   doesn't beat these has no result.
6. **Uncertainty via moving-block bootstrap** (block=20 default) on the
   per-observation Brier difference vs a benchmark — IID bootstrap would
   understate variance on autocorrelated series and is not offered.
7. **Multiple testing is recorded, not hand-waved:** callers pass
   `n_variants_tried`; the result carries a deflated p-value
   `1-(1-p)^N` (Bonferroni/independence approximation) and a plain-language
   note. Skill claim rule: deflated p < 0.05 **and** bootstrap CI excludes 0.

## Leakage-guard design (three layers)

| Layer | Mechanism | What it catches |
|---|---|---|
| Splits | `purge` drops the tail of each train block (overlapping-label leakage); `embargo` inserts a gap before each test block (serial-correlation leakage, de Prado ch. 7). `assert_splits_valid` runs on every split set and inside every backtest: train max < test min, gap ≥ embargo. | Train/test overlap, insufficient separation |
| Timestamps | `assert_no_lookahead(prediction_times, feature_max_times, horizon=)` raises `LeakageError` if any feature timestamp exceeds its prediction/issue time. `assert_not_identical_to(feature, target)` catches the crude `feature == y` leak. `assert_causal_series` is a documented smoke alarm for centred-window features, not a proof. | Future features, contemporaneous-target features |
| Scepticism | Benchmarks + bootstrap + the leakage inflation test (below). A Brier near 0 on a genuinely uncertain event is treated as evidence *of a leak*, not of genius. | Leaks that never declare timestamps and therefore bypass layers 1–2 |

**Honest limitation:** a forecaster that smuggles future data through a
closure (test 4c) cannot be caught by assertion — it never presents a
timestamp to check. The guards catch declared-feature leaks; undeclared
smuggling is caught only by impossible performance + code review. Both
layers are necessary; neither is sufficient alone.

## Real results (synthetic, known structure)

Common setup: n=1500, expanding window, initial train 300, test 50/step 50,
purge 5, embargo 5, horizons (1, 5), block bootstrap 1000 resamples (block 20),
`n_variants_tried=3`, n_test = 1,194 at horizon 1.

### Signal series — latent AR(1) φ=0.92, regimes alternating ±1.1 every 250 steps
Observable signal `x_t = latent_t + N(0, 0.25)` known at issue time;
`y_t ~ Bernoulli(sigmoid(latent_t))`. Forecaster: `p = sigmoid(x_t · φ^h)`.

| Forecaster | Brier ↓ | Log score ↓ | ECE ↓ | AUC | Hit rate |
|---|---|---|---|---|---|
| **Signal-momentum (model)** | **0.1712** | **0.5171** | **0.0175** | **0.8202** | 0.7513 |
| Climatology (benchmark) | 0.2518 | 0.6972 | — | 0.4531 | — |
| Persistence | 0.3317 | — | — | — | — |
| Random-walk / no-change | 0.2390 | — | — | — | — |

- Base rate 0.4456. Brier diff (climatology − model) = **+0.0807**,
  block-bootstrap 95% CI **[+0.0626, +0.1022]**, p(model not better) = 0.0000,
  deflated p (N=3) = 0.0000 → **credible skill, as designed.**
- Per-horizon Brier: model h=1 **0.1712**, h=5 **0.2026** vs climatology
  0.2518 / 0.2524 — skill decays with horizon, as it should for φ=0.92.

### Honesty check — y-only EWMA (α=0.05, no signal feature), same series
Brier **0.2373** vs climatology 0.2518; diff +0.0146, CI **[−0.0077, +0.0358]**,
raw p = 0.1190, deflated p = 0.3162 → **no credible skill claim.** Binary
outcomes alone are too noisy at this level; the engine correctly refuses to
certify it. (An earlier α=0.1, 50-step-block variant was *worse* than
climatology, Brier 0.2657 vs 0.2373 — stale block forecasts + EWMA overshoot.
That failure is why the signal-feature design above is the headline test.)

### Noise series — y iid Bernoulli(0.3), x iid noise independent of y
Same signal-momentum forecaster. Base rate 0.3132.

| | Brier | Log score | ECE | AUC | Hit rate |
|---|---|---|---|---|---|
| Model | 0.2939 | 0.8048 | 0.2279 | 0.4975 | 0.4841 |
| Climatology | **0.2155** | **0.6226** | — | 0.4641 | — |

Brier diff = **−0.0784**, CI [−0.0947, −0.0627], p(not better) = 1.0000 →
model is significantly *worse* than climatology and **no skill is claimed**,
exactly the required behaviour on pure noise.

### Leakage test — which leaks are caught, which inflate
- **(a) CAUGHT by assertion:** `assert_no_lookahead` with a feature timestamp
  31 > prediction time 30 raised
  `LeakageError: lookahead at position 2: feature time 31 > prediction time 30`.
- **(b) CAUGHT by assertion:** feature identical to the contemporaneous target
  raised `LeakageError: leaky_feature: feature is identical to contemporaneous target`.
- **(c) NOT caught by assertion — visibly inflates:** a "forecaster" returning
  the actual future outcomes (p = 0.99/0.01 on y_true) scored on the same test
  indices as the causal run: **leaky Brier 0.0001, AUC 1.0000** vs causal
  **Brier 0.1712, AUC 0.8202**. This is the inflation signature to hunt for in
  any future result that looks too good.

### Split integrity
16 folds at purge=10/embargo=7 all satisfy gap ≥ embargo; rolling window train
size constant at 150; expanding window train grows monotonically. Unit checks
for Brier (perfect = 0, coin-flip = 0.25) and AUC (perfect = 1, reversed = 0,
constant = 0.5) all pass.

## Exact API the main engine should call

```python
from backtest import (
    WalkForwardBacktester, BacktestResult, Split, make_splits,
    evaluate_forecasts, brier_score, log_score, expected_calibration_error,
    auc_mann_whitney, calibration_table,
    block_bootstrap_brier_diff, deflated_brier_pvalue, multiple_testing_report,
    assert_no_lookahead, assert_not_identical_to, assert_splits_valid, LeakageError,
    climatology_forecast, persistence_forecast, random_walk_forecast,
    ewma_forecaster, rolling_mean_forecaster,
)

# Forecaster protocol (any callable with this shape works):
#   fn(y_train: np.ndarray, X_train: np.ndarray | None,
#      X_test: np.ndarray, horizon: int) -> np.ndarray  # probs, len == len(X_test rows)
# X_test rows are the issue-time feature rows; return one prob per row.

bt = WalkForwardBacktester(
    window="expanding",          # or "rolling" (+ rolling_train)
    initial_train=300, test_size=50, step=None,   # step defaults to test_size
    purge=5, embargo=5,                          # >= horizon-1 purge recommended
    benchmarks=True, bootstrap_block=20, n_boot=1000, seed=0,
)
result: BacktestResult = bt.run(
    y,                  # np.ndarray of 0/1, length n
    forecaster,         # callable per protocol above
    X=None,             # optional (n, k) feature matrix, row t = features known at t
    horizons=(1, 5),    # scored separately; result.metrics = horizons[0]
    n_variants_tried=1, # TOTAL variants tried in this search, for deflation
)
result.metrics          # dict: n, base_rate, brier, log_score, ece, auc, hit_rate, mean_pred, calibration (DataFrame)
result.benchmarks       # {"climatology": metrics, "persistence": metrics, "random_walk": metrics}
result.bootstrap_vs     # {"climatology": {diff_brier_bench_minus_model, ci95_lo, ci95_hi, p_model_not_better, block, n_boot, n}, ...}
result.per_horizon      # {h: {"metrics": ..., "benchmarks": ..., "bootstrap_vs": ...}}
result.y_true, result.y_prob, result.times, result.fold_id
result.multiple_testing_note   # str
result.summary()        # DataFrame: model + benchmarks side by side

# Standalone pieces:
splits: list[Split] = make_splits(n, initial_train=200, test_size=50, step=None,
                                  window="expanding", rolling_train=None,
                                  purge=0, embargo=0, horizon=1)
metrics: dict = evaluate_forecasts(y_true, y_prob, n_bins=10)
boot: dict = block_bootstrap_brier_diff(y, p_model, p_bench, block=20, n_boot=2000, seed=0)
p_adj: float = deflated_brier_pvalue(p_raw, n_variants_tried)
assert_no_lookahead(prediction_times, feature_max_times, horizon=1)  # raises LeakageError
```

## Recommended defaults for the main engine (markets first)
- purge ≥ horizon−1, embargo ≥ 5 bars (more for overlapping return labels).
- Always report climatology + random-walk alongside any model, per horizon.
- Never certify a variant without `n_variants_tried` set truthfully and the
  deflated-p / CI rule above applied.
