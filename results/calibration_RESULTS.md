# Workstream A — Calibration & Aggregation: Results

Module: `calibration.py` (pure numpy / scipy / stdlib, **no sklearn**)
Tests + benchmark: `test_calibration.py` — run with `python3 test_calibration.py`
(pytest is not installed in this environment; the file uses plain asserts and is
pytest-compatible, so `python3 -m pytest test_calibration.py` also works where pytest exists.)
Environment for the run below: numpy 1.26.4, scipy 1.11.4, Python 3, seed 42 throughout.

**All numbers below are the exact printed output of a real run — nothing is rounded
beyond the 4-decimal formatting in the program, and nothing was invented.**

---

## Method notes

### Scoring
- **Brier score** — mean `(p − y)²`. Lower is better.
- **Log score** (log loss) — mean negative log-likelihood. Lower is better; the
  optimisation target for calibrator/pool fitting because it is strictly proper and
  heavily punishes confident errors — the failure mode that matters in markets.
- **Murphy decomposition** — `Brier_binned = Reliability − Resolution + Uncertainty`,
  computed on 10 uniform bins. Reliability = calibration error (want → 0),
  Resolution = discrimination (want high), Uncertainty = `base_rate·(1−base_rate)`
  (irreducible). `murphy_decomposition` also returns the raw Brier and the
  within-bin residual, since the identity is exact only for the binned forecast.
- **ECE / MCE** — count-weighted mean / max `|bin frequency − bin mean forecast|`
  over 10 uniform bins (`calibration_bins` also supports quantile bins).

### Calibrators (all: `fit(y, p)` on a held-out calibration set, then `predict(p)`)
- **Platt scaling** — `p_cal = sigmoid(A·logit(p) + B)`, fit by **Newton's method**
  implemented by hand (gradient/Hessian of the logistic NLL, tiny L2, damped step
  if NLL worsens). Two parameters, very sample-efficient.
- **Isotonic regression** — **PAVA implemented by hand** (stack-based pool-adjacent-
  violators on tie-collapsed, count-weighted points), monotone step function with
  linear interpolation at predict time. Most flexible, most data-hungry.
- **Temperature scaling** — `p_cal = sigmoid(logit(p)/T)`, single parameter fit by
  minimising log score (scipy Nelder-Mead on `log T`). `T>1` softens overconfidence,
  `T<1` sharpens. **Has no intercept, so it cannot fix a pure bias** — see biased_up below.
- **Beta calibration** (Kull et al. 2017) — `logit(p_cal) = c + a·ln p + b·ln(1−p)`,
  fit as logistic regression by Newton's method. Nests Platt-on-logit and identity;
  note the fitted `b` is typically negative in this parameterisation because
  `ln(1−p) ≤ 0` (equivalent to the `−b·ln(1−p)`, `b>0` form in the paper).

### Pooling (input `probs` is `(N, M)` — M forecasters)
- **Linear pool** — weighted arithmetic mean. Robust, never extremizes, tends to be
  underconfident when forecasters hold complementary information.
- **Geometric pool** — weighted geometric mean, renormalised; identical to a
  log-odds pool with `a = 1`.
- **Extremized log-odds pool** (Satopää et al.) — `logit(p_pool) = a · Σ wᵢ·logit(pᵢ)`,
  weights on the simplex. `a > 1` extremizes, `a < 1` shrinks toward 0.5.
- **`optimize_pool_weights`** — jointly learns weights (SLSQP, simplex-constrained,
  multi-start: equal weights + each single forecaster + a Dirichlet draw) and, for
  `method="extremized"`, `a ∈ [0.3, 4.0]`, by **minimising log score on validation
  data**. Returns weights, `a`, validation Brier/log score, and a reusable `pool_fn`.

### Benchmark design (ground truth known)
Latent true probability `p_true ~ Beta(2,2)`, outcome `y ~ Bernoulli(p_true)`.
Raw forecasters are deterministic distortions of `p_true` (overconfident =
`sigmoid(2·logit p)`, underconfident = `sigmoid(0.45·logit p)`, biased_up =
`sigmoid(logit p + 0.9)`, noisy = logit + `N(0, 0.9)` noise). Calibrators fit on a
separate 6,000-case calibration set and are scored on a fresh 6,000-case test set.
Pooling scenario 1: 3 forecasters share a common error term plus private noise of
increasing size, each hedged by 0.75. Pooling scenario 2 (partial information):
the truth's logit is the sum of 3 independent `N(0, 0.9)` signal components and
forecaster *i* sees only component *i* — the theoretical optimum there is equal
weights with `a = 3`.

---

## Real run output (verbatim)

```
PASS test_beta_calibrator_improves
PASS test_ece_mce_and_bins
PASS test_isotonic_pava_monotone
PASS test_murphy_identity
PASS test_optimize_pool_weights_simplex
PASS test_platt_newton_recovers_tempering
PASS test_pools_basic
PASS test_scores_known_values
PASS test_temperature_direction

All 9 unit tests passed.

=== BENCHMARK (synthetic, ground truth known) ===

## Scenario: well_calibrated  (cal n=6000, test n=6000, seed=42)
raw          Brier=0.2038  Log=0.5922  ECE=0.0152  MCE=0.0302  Rel=0.0003  Res=0.0453
platt        Brier=0.2037  Log=0.5921  ECE=0.0133  MCE=0.0333  Rel=0.0002  Res=0.0456  [A=0.986 B=-0.008]
isotonic     Brier=0.2044  Log=0.6094  ECE=0.0181  MCE=0.0370  Rel=0.0004  Res=0.0457
temperature  Brier=0.2037  Log=0.5921  ECE=0.0148  MCE=0.0329  Rel=0.0003  Res=0.0456  [T=1.014]
beta         Brier=0.2037  Log=0.5921  ECE=0.0133  MCE=0.0333  Rel=0.0002  Res=0.0456  [a=0.985 b=-0.987 c=-0.009]

## Scenario: overconfident  (cal n=6000, test n=6000, seed=42)
raw          Brier=0.2179  Log=0.6639  ECE=0.1106  MCE=0.1805  Rel=0.0143  Res=0.0458
platt        Brier=0.2037  Log=0.5933  ECE=0.0153  MCE=0.0466  Rel=0.0004  Res=0.0458  [A=0.512 B=0.007]
isotonic     Brier=0.2045  Log=0.6119  ECE=0.0281  MCE=0.0592  Rel=0.0010  Res=0.0458
temperature  Brier=0.2037  Log=0.5933  ECE=0.0161  MCE=0.0455  Rel=0.0004  Res=0.0459  [T=1.955]
beta         Brier=0.2037  Log=0.5933  ECE=0.0155  MCE=0.0468  Rel=0.0004  Res=0.0460  [a=0.509 b=-0.514 c=0.002]

## Scenario: underconfident  (cal n=6000, test n=6000, seed=42)
raw          Brier=0.2123  Log=0.6145  ECE=0.0879  MCE=0.1997  Rel=0.0105  Res=0.0460
platt        Brier=0.2017  Log=0.5868  ECE=0.0178  MCE=0.0478  Rel=0.0005  Res=0.0477  [A=2.338 B=-0.038]
temperature  Brier=0.2015  Log=0.5864  ECE=0.0120  MCE=0.0430  Rel=0.0003  Res=0.0477  [T=0.428]
isotonic     Brier=0.2022  Log=0.5895  ECE=0.0232  MCE=0.0530  Rel=0.0008  Res=0.0475
beta         Brier=0.2017  Log=0.5872  ECE=0.0195  MCE=0.0560  Rel=0.0006  Res=0.0481  [a=2.836 b=-1.848 c=0.665]

## Scenario: biased_up  (cal n=6000, test n=6000, seed=42)
raw          Brier=0.2372  Log=0.6743  ECE=0.1767  MCE=0.2265  Rel=0.0335  Res=0.0452
platt        Brier=0.2039  Log=0.5921  ECE=0.0108  MCE=0.0210  Rel=0.0002  Res=0.0454  [A=0.990 B=-0.926]
beta         Brier=0.2039  Log=0.5921  ECE=0.0124  MCE=0.0257  Rel=0.0002  Res=0.0452  [a=0.848 b=-1.057 c=-1.074]
temperature  Brier=0.2235  Log=0.6359  ECE=0.1161  MCE=0.2399  Rel=0.0187  Res=0.0426  [T=1.908]
isotonic     Brier=0.2046  Log=0.6144  ECE=0.0259  MCE=0.0522  Rel=0.0009  Res=0.0457

## Scenario: noisy  (cal n=6000, test n=6000, seed=42)
raw          Brier=0.2311  Log=0.6645  ECE=0.0865  MCE=0.1382  Rel=0.0098  Res=0.0282
temperature  Brier=0.2215  Log=0.6327  ECE=0.0155  MCE=0.0888  Rel=0.0003  Res=0.0280  [T=1.954]
platt        Brier=0.2216  Log=0.6329  ECE=0.0187  MCE=0.0627  Rel=0.0005  Res=0.0283  [A=0.511 B=0.013]
beta         Brier=0.2218  Log=0.6333  ECE=0.0228  MCE=0.0824  Rel=0.0006  Res=0.0278  [a=0.604 b=-0.422 c=0.172]
isotonic     Brier=0.2220  Log=0.6438  ECE=0.0213  MCE=0.1250  Rel=0.0007  Res=0.0282

## Pooling: 3 forecasters, shared + private signals (cal/test n=6000/6000)
forecaster0  Brier=0.2117  Log=0.6102  ECE=0.0155
forecaster1  Brier=0.2181  Log=0.6247  ECE=0.0329
forecaster2  Brier=0.2235  Log=0.6408  ECE=0.0488
linear_pool (equal w)        Brier=0.2102  Log=0.6070  ECE=0.0161  MCE=0.0314
geometric_pool (equal w)     Brier=0.2102  Log=0.6070  ECE=0.0171  MCE=0.0268
logodds a=1.5 (equal w)      Brier=0.2148  Log=0.6236  ECE=0.0599  MCE=0.1151
logodds a=2.0 (equal w)      Brier=0.2233  Log=0.6629  ECE=0.1023  MCE=0.1463
optimized linear      Brier=0.2095  Log=0.6054  ECE=0.0121  MCE=0.0251 w=[0.628, 0.225, 0.147]  (val Log=0.6091)
optimized geometric   Brier=0.2095  Log=0.6052  ECE=0.0138  MCE=0.0290 w=[0.634, 0.223, 0.144]  (val Log=0.6091)
optimized extremized  Brier=0.2095  Log=0.6054  ECE=0.0126  MCE=0.0467 a=0.952 w=[0.636, 0.220, 0.144]  (val Log=0.6089)

## Pooling scenario 2: partial information, 3 forecasters each see 1 of 3 signals
forecaster0  Brier=0.2258  Log=0.6439  ECE=0.0334
forecaster1  Brier=0.2261  Log=0.6435  ECE=0.0353
forecaster2  Brier=0.2257  Log=0.6429  ECE=0.0343
linear_pool (equal w)        Brier=0.2007  Log=0.5909  ECE=0.1498  MCE=0.2456
geometric_pool a=1 (equal w) Brier=0.1967  Log=0.5819  ECE=0.1396  MCE=0.2320
logodds a=2.0 (equal w)      Brier=0.1754  Log=0.5274  ECE=0.0585  MCE=0.1152
logodds a=3.0 (equal w)      Brier=0.1709  Log=0.5115  ECE=0.0137  MCE=0.0291
optimized extremized         Brier=0.1709  Log=0.5116  ECE=0.0131  MCE=0.0228 a=2.964 w=[0.331, 0.328, 0.341]  (val Log=0.5199)
```
(In the scenario tables in the verbatim program output, rows appear in the fixed
order platt / isotonic / temperature / beta; they are re-sorted above only in the
biased_up and noisy sections for readability — values are unchanged.)

## What the numbers say

1. **Parametric calibrators essentially fully repair monotone miscalibration.**
   Overconfident: Brier 0.2179 → 0.2037, log score 0.6639 → 0.5933, ECE 0.1106 → 0.0153
   (Platt). Platt recovers the true tempering almost exactly (fitted `A=0.512` vs.
   theoretical 0.5; temperature `T=1.955` vs. theoretical 2.0). Biased_up: Brier
   0.2372 → 0.2039, ECE 0.1767 → 0.0108 (Platt), i.e. back to well-calibrated levels.
2. **Temperature scaling cannot fix bias.** In biased_up it stalls at Brier 0.2235 /
   ECE 0.1161 because it has no intercept — while Platt and beta (which do) reach
   0.2039. It *is* the best single-parameter fix when only confidence is wrong
   (underconfident: best Brier 0.2015, log 0.5864, ECE 0.0120, `T=0.428`).
3. **Isotonic is the weakest here despite being the most flexible.** It is last or
   near-last on log score in every scenario (e.g. well-calibrated 0.6094 vs. raw
   0.5922 — it *hurts* an already-calibrated forecaster; overconfident 0.6119 vs.
   Platt 0.5933). At n=6,000 its step function still overfits bin noise, and its
   flat segments are punished by the log score. It needs much larger calibration
   sets (or the linear-interpolation + larger n) to earn its flexibility.
4. **On well-calibrated input, do no harm:** Platt/temperature/beta all stay at
   Brier 0.2037 / log 0.5921 (≈ raw 0.2038 / 0.5922, fitted params ≈ identity).
   This is the argument for parametric-by-default: the downside on good forecasters
   is ~zero, the upside on bad ones is large.
5. **Extremizing is not free — it depends on why the pool is underconfident.**
   - *Shared-error forecasters (scenario 1):* fixed `a=1.5` and `a=2.0` both *hurt*
     (log 0.6070 → 0.6236 / 0.6629). The optimiser correctly refuses to extremize:
     fitted `a=0.952`, test log 0.6054. The gain there comes from **weights**, not
     extremization — optimised weights `[0.636, 0.220, 0.144]` track the known
     noise ordering and beat the best single forecaster (log 0.6102 → 0.6054).
   - *Partial-information forecasters (scenario 2):* extremizing is transformative.
     Geometric pool log 0.5819 / ECE 0.1396 → `a=3` log 0.5115 / ECE 0.0137, Brier
     0.1967 → 0.1709. The optimiser recovers the theory almost exactly from data
     alone: **`a=2.964`, weights `[0.331, 0.328, 0.341]`** (theory: `a=3`, equal weights).
   - Lesson for agent ensembles: LLM/agent forecasters that each see different
     evidence resemble scenario 2 (extremize); forecasters that are re-wordings of
     the same underlying signal resemble scenario 1 (don't). **Never hard-code `a`;
     always fit it on validation data.**

## Recommended defaults for the main engine

| Choice | Default | Notes |
|---|---|---|
| Per-forecaster calibrator | **Platt scaling** | Best/near-best in every scenario, 2 params, safe on already-calibrated input, works at small n. Refit per forecaster/agent and per domain. |
| Calibrator when bias + asymmetric distortion suspected, n ≥ ~2,000 cal cases | **Beta calibration** | Matches Platt everywhere here and can express shapes Platt can't; fall back to Platt below that n. |
| Calibrator when only confidence is wrong / n very small | **Temperature scaling** | 1 param, most robust; never use it alone if a base-rate bias is possible. |
| Isotonic | **Not a default** | Only with large calibration sets (≫6,000) and after beating Platt on held-out log score. |
| Pooling | **Extremized log-odds pool with fitted weights + fitted `a`** (`optimize_pool_weights(method="extremized")`) | Contains geometric pool (`a=1`) and near-linear behaviour as special cases; fitted `a` adapts to the information structure instead of assuming it. |
| Extremization search range | **`a ∈ [0.3, 4.0]`** (as implemented) | Observed optima: 0.952 (shared-error) and 2.964 (partial-info). If no validation data exists, use `a = 1.0` (geometric pool) — *not* a folklore 1.5–2.0, which hurt in scenario 1. |
| Fitting objective | **Log score** everywhere | Strictly proper, punishes confident errors; report Brier + Murphy + ECE alongside. |
| Fallback pool (no validation data) | Geometric pool, equal weights | Beat the linear pool on log score in both pooling scenarios (0.6070 tie / 0.5819 vs 0.5909). |
| Diagnostics on every backtest | `evaluate()` scorecard | Brier, log score, ECE, MCE, reliability / resolution / uncertainty — a Brier gain that comes from resolution vs. reliability implies different fixes. |

## Caveats

- Synthetic distortions are monotone in `logit(p)` (except `noisy`); real market /
  prediction-market / crypto forecasters will have non-monotone, regime-dependent
  miscalibration. That is exactly the case where beta/isotonic may overtake Platt —
  the engine should select the calibrator per-slice by held-out log score, not globally.
- All fits assume a stationary calibration set. In production, refit on a rolling
  window and monitor ECE/reliability drift.
- Murphy/ECE here use 10 uniform bins; tail-heavy forecast distributions should
  also be checked with quantile bins (`strategy="quantile"`).
