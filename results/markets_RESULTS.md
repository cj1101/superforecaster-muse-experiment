# Markets Forecaster — Walk-Forward Backtest Results

Research only — no trading. All numbers below come from an actual run of `run_markets_backtest.py` on real Yahoo Finance data; see `results.json` for the machine-readable version.

## Method
- Data source: Yahoo Finance chart API v8, daily, range=10y, adjclose used for price/returns.
- Split scheme: Walk-forward expanding window. Initial train 756 trading days (~3y). Retrain every 21 days. At forecast date t, training uses only samples i with i+horizon <= t (label fully known by t). Features at t use data <= t only.
- Model: Logistic regression via full-batch gradient descent (L2=1.0, lr=0.05, 300 epochs), intercept initialized at logit(expanding base rate) as anchor. Standardization from train window only. Plus ridge (lam=10) expected-return model.
- Features: mom 2/5/10/21/63/126/252d log returns; z-score vs SMA20/50/200 + dist to SMA; realized vol 10/21/63d + vol ratios; RSI7/14; dist from 52w high + 52w range position; volume trend/z (when available); day-of-week sin/cos + Mon/Fri flags; vol-adjusted 5d momentum; 1d return + abs return.
- Benchmark (climatology): Expanding base rate of up-moves computed only from training labels known at each forecast date.

## Headline results

| Symbol | Horizon | n | Brier | Clim Brier | Brier skill | Log score | Clim log score | AUC | Beats clim? |
|---|---|---|---|---|---|---|---|---|---|
Data: **^GSPC** 2016-10-17 → 2026-10-16 (2514 daily rows)
| ^GSPC | 1d | 1757 | 0.25176 | 0.24824 | -0.0142 | -0.69714 | -0.68962 | 0.4856 | NO |
| ^GSPC | 5d | 1753 | 0.25056 | 0.23999 | -0.0441 | -0.69669 | -0.67309 | 0.4702 | NO |
| ^GSPC | 21d | 1737 | 0.24365 | 0.21719 | -0.1218 | -0.69236 | -0.62685 | 0.4531 | NO |
Data: **QQQ** 2016-10-17 → 2026-10-16 (2514 daily rows)
| QQQ | 1d | 1757 | 0.25002 | 0.24632 | -0.015 | -0.69366 | -0.68577 | 0.5045 | NO |
| QQQ | 5d | 1753 | 0.24752 | 0.24105 | -0.0269 | -0.69065 | -0.67522 | 0.4996 | NO |
| QQQ | 21d | 1737 | 0.2435 | 0.22615 | -0.0767 | -0.69548 | -0.64594 | 0.4724 | NO |
Data: **NVDA** 2016-10-17 → 2026-10-16 (2514 daily rows)
| NVDA | 1d | 1757 | 0.24902 | 0.24843 | -0.0024 | -0.69128 | -0.69001 | 0.5209 | NO |
| NVDA | 5d | 1753 | 0.24601 | 0.24262 | -0.014 | -0.68565 | -0.67836 | 0.5042 | NO |
| NVDA | 21d | 1737 | 0.22902 | 0.2275 | -0.0067 | -0.65231 | -0.64792 | 0.5383 | NO |
Data: **TSM** 2016-10-17 → 2026-10-16 (2514 daily rows)
| TSM | 1d | 1757 | 0.25227 | 0.25006 | -0.0088 | -0.69796 | -0.69328 | 0.5097 | NO |
| TSM | 5d | 1753 | 0.25905 | 0.24783 | -0.0453 | -0.71263 | -0.68883 | 0.4576 | NO |
| TSM | 21d | 1737 | 0.26318 | 0.23877 | -0.1022 | -0.73125 | -0.67093 | 0.4469 | NO |

Brier skill > 0 means the model beats climatology; negative means it loses to the base rate — reported honestly where it occurs.

## ^GSPC — calibration

### 1d horizon (n=1757, test base rate=0.5441, mean pred=0.5516, exp-ret corr=-0.0013)

| Bin | n | Mean pred | Empirical rate |
|---|---|---|---|
| 0.2–0.3 | 3 | 0.2536 | 0.6667 |
| 0.3–0.4 | 20 | 0.3716 | 0.4 |
| 0.4–0.5 | 232 | 0.4699 | 0.5345 |
| 0.5–0.6 | 1280 | 0.5545 | 0.5422 |
| 0.6–0.7 | 205 | 0.6289 | 0.5902 |
| 0.7–0.8 | 11 | 0.7282 | 0.2727 |
| 0.8–0.9 | 4 | 0.8518 | 0.5 |
| 0.9–1.0 | 2 | 0.9179 | 1.0 |

### 5d horizon (n=1753, test base rate=0.6041, mean pred=0.6133, exp-ret corr=-0.1087)

| Bin | n | Mean pred | Empirical rate |
|---|---|---|---|
| 0.1–0.2 | 4 | 0.1468 | 0.75 |
| 0.2–0.3 | 14 | 0.2711 | 0.5 |
| 0.3–0.4 | 43 | 0.3599 | 0.7907 |
| 0.4–0.5 | 91 | 0.4578 | 0.6374 |
| 0.5–0.6 | 443 | 0.5691 | 0.6163 |
| 0.6–0.7 | 983 | 0.6429 | 0.587 |
| 0.7–0.8 | 168 | 0.7353 | 0.6071 |
| 0.8–0.9 | 5 | 0.834 | 0.8 |
| 0.9–1.0 | 2 | 0.9357 | 0.5 |

### 21d horizon (n=1737, test base rate=0.688, mean pred=0.666, exp-ret corr=0.095)

| Bin | n | Mean pred | Empirical rate |
|---|---|---|---|
| 0.0–0.1 | 7 | 0.0795 | 0.8571 |
| 0.1–0.2 | 21 | 0.1572 | 0.7619 |
| 0.2–0.3 | 28 | 0.243 | 0.6786 |
| 0.3–0.4 | 31 | 0.3523 | 0.7097 |
| 0.4–0.5 | 83 | 0.4606 | 0.7831 |
| 0.5–0.6 | 213 | 0.5558 | 0.7934 |
| 0.6–0.7 | 544 | 0.6587 | 0.6618 |
| 0.7–0.8 | 630 | 0.7426 | 0.681 |
| 0.8–0.9 | 156 | 0.8356 | 0.5897 |
| 0.9–1.0 | 24 | 0.9209 | 0.7083 |

## QQQ — calibration

### 1d horizon (n=1757, test base rate=0.5623, mean pred=0.5628, exp-ret corr=0.0304)

| Bin | n | Mean pred | Empirical rate |
|---|---|---|---|
| 0.1–0.2 | 2 | 0.1539 | 0.5 |
| 0.2–0.3 | 10 | 0.2559 | 0.4 |
| 0.3–0.4 | 39 | 0.3661 | 0.4359 |
| 0.4–0.5 | 216 | 0.4648 | 0.588 |
| 0.5–0.6 | 982 | 0.5562 | 0.5642 |
| 0.6–0.7 | 466 | 0.6314 | 0.5579 |
| 0.7–0.8 | 38 | 0.7268 | 0.5526 |
| 0.8–0.9 | 4 | 0.8301 | 1.0 |

### 5d horizon (n=1753, test base rate=0.599, mean pred=0.6047, exp-ret corr=-0.0595)

| Bin | n | Mean pred | Empirical rate |
|---|---|---|---|
| 0.1–0.2 | 2 | 0.1506 | 0.5 |
| 0.2–0.3 | 8 | 0.2672 | 0.625 |
| 0.3–0.4 | 25 | 0.3579 | 0.64 |
| 0.4–0.5 | 93 | 0.4669 | 0.5054 |
| 0.5–0.6 | 646 | 0.5638 | 0.5991 |
| 0.6–0.7 | 843 | 0.638 | 0.6287 |
| 0.7–0.8 | 109 | 0.7395 | 0.4587 |
| 0.8–0.9 | 26 | 0.8322 | 0.5 |
| 0.9–1.0 | 1 | 0.9037 | 1.0 |

### 21d horizon (n=1737, test base rate=0.6661, mean pred=0.6523, exp-ret corr=0.0096)

| Bin | n | Mean pred | Empirical rate |
|---|---|---|---|
| 0.1–0.2 | 7 | 0.1691 | 1.0 |
| 0.2–0.3 | 23 | 0.2532 | 0.6522 |
| 0.3–0.4 | 56 | 0.3565 | 0.5357 |
| 0.4–0.5 | 89 | 0.4637 | 0.5618 |
| 0.5–0.6 | 303 | 0.56 | 0.7294 |
| 0.6–0.7 | 639 | 0.6512 | 0.6964 |
| 0.7–0.8 | 435 | 0.7366 | 0.6506 |
| 0.8–0.9 | 147 | 0.8389 | 0.6395 |
| 0.9–1.0 | 38 | 0.927 | 0.3158 |

## NVDA — calibration

### 1d horizon (n=1757, test base rate=0.5413, mean pred=0.5425, exp-ret corr=0.0419)

| Bin | n | Mean pred | Empirical rate |
|---|---|---|---|
| 0.2–0.3 | 2 | 0.2543 | 1.0 |
| 0.3–0.4 | 24 | 0.3747 | 0.25 |
| 0.4–0.5 | 359 | 0.4718 | 0.5237 |
| 0.5–0.6 | 1104 | 0.5469 | 0.5471 |
| 0.6–0.7 | 253 | 0.63 | 0.5573 |
| 0.7–0.8 | 14 | 0.7228 | 0.6429 |
| 0.8–0.9 | 1 | 0.8828 | 1.0 |

### 5d horizon (n=1753, test base rate=0.5898, mean pred=0.5873, exp-ret corr=0.0029)

| Bin | n | Mean pred | Empirical rate |
|---|---|---|---|
| 0.3–0.4 | 6 | 0.3701 | 0.8333 |
| 0.4–0.5 | 145 | 0.4712 | 0.5655 |
| 0.5–0.6 | 852 | 0.5582 | 0.5986 |
| 0.6–0.7 | 682 | 0.6362 | 0.5806 |
| 0.7–0.8 | 67 | 0.727 | 0.597 |
| 0.8–0.9 | 1 | 0.8036 | 1.0 |

### 21d horizon (n=1737, test base rate=0.658, mean pred=0.6557, exp-ret corr=-0.0208)

| Bin | n | Mean pred | Empirical rate |
|---|---|---|---|
| 0.3–0.4 | 3 | 0.3884 | 1.0 |
| 0.4–0.5 | 66 | 0.4645 | 0.6515 |
| 0.5–0.6 | 351 | 0.5605 | 0.6439 |
| 0.6–0.7 | 793 | 0.652 | 0.633 |
| 0.7–0.8 | 443 | 0.7364 | 0.7336 |
| 0.8–0.9 | 81 | 0.828 | 0.5432 |

## TSM — calibration

### 1d horizon (n=1757, test base rate=0.5162, mean pred=0.5136, exp-ret corr=0.0537)

| Bin | n | Mean pred | Empirical rate |
|---|---|---|---|
| 0.0–0.1 | 1 | 0.0923 | 0.0 |
| 0.1–0.2 | 2 | 0.1666 | 0.5 |
| 0.2–0.3 | 17 | 0.2699 | 0.4118 |
| 0.3–0.4 | 87 | 0.3723 | 0.3793 |
| 0.4–0.5 | 531 | 0.4636 | 0.5292 |
| 0.5–0.6 | 975 | 0.5408 | 0.52 |
| 0.6–0.7 | 135 | 0.6296 | 0.5333 |
| 0.7–0.8 | 9 | 0.7304 | 0.6667 |

### 5d horizon (n=1753, test base rate=0.5545, mean pred=0.5511, exp-ret corr=-0.1142)

| Bin | n | Mean pred | Empirical rate |
|---|---|---|---|
| 0.1–0.2 | 1 | 0.1939 | 1.0 |
| 0.2–0.3 | 12 | 0.2549 | 0.75 |
| 0.3–0.4 | 38 | 0.3589 | 0.7632 |
| 0.4–0.5 | 302 | 0.4677 | 0.6026 |
| 0.5–0.6 | 1024 | 0.5528 | 0.5479 |
| 0.6–0.7 | 322 | 0.6289 | 0.5155 |
| 0.7–0.8 | 53 | 0.7288 | 0.4528 |
| 0.8–0.9 | 1 | 0.8107 | 0.0 |

### 21d horizon (n=1737, test base rate=0.62, mean pred=0.5967, exp-ret corr=-0.194)

| Bin | n | Mean pred | Empirical rate |
|---|---|---|---|
| 0.1–0.2 | 8 | 0.1494 | 0.75 |
| 0.2–0.3 | 33 | 0.2447 | 0.697 |
| 0.3–0.4 | 80 | 0.3593 | 0.625 |
| 0.4–0.5 | 185 | 0.4614 | 0.6054 |
| 0.5–0.6 | 480 | 0.5574 | 0.675 |
| 0.6–0.7 | 681 | 0.6438 | 0.6388 |
| 0.7–0.8 | 196 | 0.7391 | 0.5714 |
| 0.8–0.9 | 73 | 0.8411 | 0.2055 |
| 0.9–1.0 | 1 | 0.9053 | 0.0 |

## Latest forecasts (model trained on all labelled data, predicting from last close)

### ^GSPC — as of 2026-10-16, last price 7659.43994140625
- 1d: P(up)=0.599, exp log-return=0.00111, q10=-0.01315, q90=0.01537
- 5d: P(up)=0.5787, exp log-return=-0.00054, q10=-0.0294, q90=0.02832
- 21d: P(up)=0.698, exp log-return=0.01266, q10=-0.04347, q90=0.06879

### QQQ — as of 2026-10-16, last price 709.7999877929688
- 1d: P(up)=0.6614, exp log-return=0.00124, q10=-0.01671, q90=0.0192
- 5d: P(up)=0.5912, exp log-return=0.00369, q10=-0.03247, q90=0.03984
- 21d: P(up)=0.6989, exp log-return=0.02004, q10=-0.04878, q90=0.08887

### NVDA — as of 2026-10-16, last price 214.1300048828125
- 1d: P(up)=0.6263, exp log-return=0.00258, q10=-0.03724, q90=0.04239
- 5d: P(up)=0.6803, exp log-return=0.01757, q10=-0.066, q90=0.10115
- 21d: P(up)=0.8171, exp log-return=0.09449, q10=-0.06848, q90=0.25746

### TSM — as of 2026-10-16, last price 412.6099853515625
- 1d: P(up)=0.5958, exp log-return=0.00349, q10=-0.02427, q90=0.03125
- 5d: P(up)=0.5341, exp log-return=0.00041, q10=-0.05606, q90=0.05688
- 21d: P(up)=0.6117, exp log-return=0.02395, q10=-0.09001, q90=0.13792

## Honest interpretation

- Equity direction at 1–21d is close to a coin flip with a positive drift; climatology (always predicting the expanding base rate, ~53–58% up depending on symbol/horizon) is a hard benchmark to beat on Brier/log score.
- Any horizon/symbol with negative Brier skill above genuinely underperforms the base rate out-of-sample — that is a real finding, not a bug: it means those features add variance without signal at that horizon.
- AUC modestly above 0.5 (where present) indicates weak ranking signal even when calibration vs climatology is a wash.
- Overlapping horizons (5d/21d forecasts on consecutive days) mean effective independent sample size is far smaller than n; do not over-read small skill differences.
