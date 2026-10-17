# Crypto Forecaster — Walk-Forward Backtest Results (Workstream E)

**Data source:** Coinbase Exchange public candles API (daily), paginated 290d chunks; Kraken cross-checked (~720 candles only), Binance geo-blocked/not used

**Split scheme:** walk-forward expanding window, min train 1095 days, refit every 30 days, horizon-purged training labels

**Models:** hand-rolled L2 logistic regression (P(up), crash risk) and ridge regression (expected log-return), numpy/pandas only, no sklearn. Features: momentum 1/7/14/30/90/180d, MA gaps, realised vol 7/30/90d, vol-of-vol, drawdown from ATH & 90d high, RSI(14), weekend + day-of-week, volume z-score; ETH adds BTC momentum/vol (cross-asset).

> Honest-reading note: daily/weekly crypto direction is close to a coin flip. A Brier *worse* than climatology or AUC near 0.50 is reported as-is, not hidden.

## BTC

Data: 2015-07-20..2026-10-17 (4108 daily candles), 21 features.

| Horizon | n | Eval window | Base rate (up) | Brier | Brier climatology | Brier skill | Log score | Log score clim. | AUC |
|---|---|---|---|---|---|---|---|---|---|
| 1d | 2832 | 2019-01-15..2026-10-16 | 0.510 | 0.2508 | 0.2507 | -0.000 | -0.6948 | -0.6945 | 0.529 |
| 7d | 2826 | 2019-01-15..2026-10-10 | 0.531 | 0.2537 | 0.2501 | -0.014 | -0.7007 | -0.6934 | 0.494 |
| 30d | 2803 | 2019-01-15..2026-09-17 | 0.570 | 0.2813 | 0.2479 | -0.135 | -0.7641 | -0.6891 | 0.431 |

Expected-return (ridge, log-return) vs climatology (train mean return):

| Horizon | n | MAE | MAE clim. | MAE skill | Corr(pred, actual) | Direction acc. |
|---|---|---|---|---|---|---|
| 1d | 2832 | 0.0218 | 0.0215 | -0.016 | 0.068 | 0.522 |
| 7d | 2826 | 0.0633 | 0.0603 | -0.049 | 0.060 | 0.519 |
| 30d | 2803 | 0.1654 | 0.1429 | -0.157 | -0.073 | 0.484 |

### Crash risk — P(min close in next 7d <= 0.90 * close[t])

Eval 2019-01-15..2026-10-10, n=2826, realised crash frequency (base rate)=0.100, mean predicted=0.127

- Brier = **0.0901** vs base-rate climatology **0.0922** (skill 0.024)
- Log score = -0.3227 vs climatology -0.3348
- AUC = 0.607

Calibration buckets, BTC P(up 1d):

| Bucket | n | Mean pred | Observed freq |
|---|---|---|---|
| [0.2,0.4) | 6 | 0.375 | 0.500 |
| [0.4,0.5) | 663 | 0.473 | 0.490 |
| [0.5,0.6) | 1766 | 0.545 | 0.505 |
| [0.6,0.8) | 396 | 0.630 | 0.566 |
| [0.8,1.0] | 1 | 0.905 | 1.000 |

Calibration buckets, BTC P(up 7d):

| Bucket | n | Mean pred | Observed freq |
|---|---|---|---|
| [0.2,0.4) | 13 | 0.388 | 0.538 |
| [0.4,0.5) | 467 | 0.469 | 0.548 |
| [0.5,0.6) | 1681 | 0.551 | 0.519 |
| [0.6,0.8) | 660 | 0.634 | 0.548 |
| [0.8,1.0] | 5 | 0.822 | 1.000 |

Calibration buckets, BTC P(up 30d):

| Bucket | n | Mean pred | Observed freq |
|---|---|---|---|
| [0.0,0.2) | 71 | 0.154 | 0.944 |
| [0.2,0.4) | 161 | 0.322 | 0.882 |
| [0.4,0.5) | 153 | 0.463 | 0.595 |
| [0.5,0.6) | 953 | 0.556 | 0.551 |
| [0.6,0.8) | 1424 | 0.664 | 0.516 |
| [0.8,1.0] | 41 | 0.920 | 0.902 |

Calibration buckets, crash 7d:

| Bucket | n | Mean pred | Observed freq |
|---|---|---|---|
| [0.0,0.2) | 2512 | 0.110 | 0.091 |
| [0.2,0.4) | 306 | 0.260 | 0.176 |
| [0.4,0.5) | 8 | 0.418 | 0.000 |

## ETH

Data: 2016-05-18..2026-10-17 (3803 daily candles), 27 features.

| Horizon | n | Eval window | Base rate (up) | Brier | Brier climatology | Brier skill | Log score | Log score clim. | AUC |
|---|---|---|---|---|---|---|---|---|---|
| 1d | 2527 | 2019-11-16..2026-10-16 | 0.516 | 0.2499 | 0.2499 | -0.000 | -0.6932 | -0.6929 | 0.541 |
| 7d | 2521 | 2019-11-16..2026-10-10 | 0.523 | 0.2507 | 0.2501 | -0.002 | -0.6948 | -0.6934 | 0.530 |
| 30d | 2498 | 2019-11-16..2026-09-17 | 0.547 | 0.2577 | 0.2509 | -0.027 | -0.7084 | -0.6950 | 0.508 |

Expected-return (ridge, log-return) vs climatology (train mean return):

| Horizon | n | MAE | MAE clim. | MAE skill | Corr(pred, actual) | Direction acc. |
|---|---|---|---|---|---|---|
| 1d | 2527 | 0.0292 | 0.0286 | -0.020 | 0.071 | 0.509 |
| 7d | 2521 | 0.0843 | 0.0796 | -0.059 | 0.060 | 0.488 |
| 30d | 2498 | 0.2274 | 0.1898 | -0.199 | 0.012 | 0.461 |

### Crash risk — P(min close in next 7d <= 0.90 * close[t])

Eval 2019-11-16..2026-10-10, n=2521, realised crash frequency (base rate)=0.170, mean predicted=0.216

- Brier = **0.1417** vs base-rate climatology **0.1446** (skill 0.020)
- Log score = -0.4568 vs climatology -0.4663
- AUC = 0.577

Calibration buckets, ETH P(up 1d):

| Bucket | n | Mean pred | Observed freq |
|---|---|---|---|
| [0.2,0.4) | 77 | 0.376 | 0.468 |
| [0.4,0.5) | 1190 | 0.464 | 0.483 |
| [0.5,0.6) | 1067 | 0.538 | 0.547 |
| [0.6,0.8) | 190 | 0.642 | 0.563 |
| [0.8,1.0] | 3 | 0.879 | 1.000 |

Calibration buckets, ETH P(up 7d):

| Bucket | n | Mean pred | Observed freq |
|---|---|---|---|
| [0.2,0.4) | 80 | 0.383 | 0.512 |
| [0.4,0.5) | 1120 | 0.462 | 0.499 |
| [0.5,0.6) | 1147 | 0.539 | 0.543 |
| [0.6,0.8) | 170 | 0.656 | 0.553 |
| [0.8,1.0] | 4 | 0.813 | 0.500 |

Calibration buckets, ETH P(up 30d):

| Bucket | n | Mean pred | Observed freq |
|---|---|---|---|
| [0.2,0.4) | 349 | 0.352 | 0.665 |
| [0.4,0.5) | 875 | 0.452 | 0.489 |
| [0.5,0.6) | 754 | 0.549 | 0.491 |
| [0.6,0.8) | 475 | 0.659 | 0.625 |
| [0.8,1.0] | 45 | 0.905 | 0.889 |

Calibration buckets, crash 7d:

| Bucket | n | Mean pred | Observed freq |
|---|---|---|---|
| [0.0,0.2) | 1165 | 0.162 | 0.155 |
| [0.2,0.4) | 1321 | 0.258 | 0.182 |
| [0.4,0.5) | 35 | 0.427 | 0.200 |

## Verdict (generated from the numbers above)

- BTC P(up 1d): Brier skill -0.000 — statistically indistinguishable from climatology (AUC 0.529).
- BTC P(up 7d): Brier skill -0.014 — does NOT beat climatology — ship climatology, not this model (AUC 0.494).
- BTC P(up 30d): Brier skill -0.135 — does NOT beat climatology — ship climatology, not this model (AUC 0.431).
- BTC crash-risk 7d: Brier skill 0.024 vs base rate, AUC 0.607 — the only model family here with a consistent (modest) edge; vol/drawdown features carry real crash signal.
- ETH P(up 1d): Brier skill -0.000 — statistically indistinguishable from climatology (AUC 0.541).
- ETH P(up 7d): Brier skill -0.002 — statistically indistinguishable from climatology (AUC 0.530).
- ETH P(up 30d): Brier skill -0.027 — does NOT beat climatology — ship climatology, not this model (AUC 0.508).
- ETH crash-risk 7d: Brier skill 0.020 vs base rate, AUC 0.577 — the only model family here with a consistent (modest) edge; vol/drawdown features carry real crash signal.
- Expected-return ridge models have negative MAE skill at every asset/horizon: point return forecasts are worse than predicting the training-mean return. Use them, if at all, only for sign at 1d.

## Caveats

- Overlapping horizons (7d/30d labels on daily origins) make consecutive errors correlated; effective sample size is far below n, so small Brier/AUC edges are not significant.
- No transaction costs, no trading — research/forecast-quality measurement only.
- Survivorship/regime caveat: sample spans multiple bull/bear regimes; a single expanding window averages over them.
