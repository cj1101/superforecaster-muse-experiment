# Workstream D — Prediction Markets: calibration & adjustment rules

Generated 2026-10-17T19:21:38.516891+00:00 from live public APIs. Research only — no betting.

## Headline

- Clean resolved binary markets analysed: **763** (2024-01-10 23:47:24+00:00 → 2026-10-06 02:22:39+00:00), total lifetime volume $26,124,431,610.
- Forecast samples (market × horizon snapshots): **3407**.
- Split by closing date: 457 fit markets / 306 test markets (cutoff 2025-07-23 02:37:59+00:00).
- **Primary (T-7d price): raw market test Brier = 0.09051, log score = 0.28805.** Best rule selected on fit (logit_shrink): test Brier = 0.09095, log score = 0.28857. Beats raw out-of-sample: **NO**.
- Pooled horizons: raw test Brier = 0.07125; best-by-fit rule (logit_time) test Brier = 0.07177. Beats raw: **NO**.

## Price proxy (exactly what was used)

CLOB daily history: last print at/before close-horizon (freshness window horizon+14d); horizons 30/14/7/3/1d; headline = T-7d. NOT Gamma outcomePrices/lastTradePrice (those are the resolution / ~0-1 at close).

## Scores

### primary_T7  (fit n=414, test n=287, test base rate=0.223)

| rule | params (fit only) | fit Brier | test Brier | test log score | test ECE |
|---|---|---|---|---|---|
| raw | — | 0.08991 | 0.09051 | 0.28805 | 0.07775 |
| shrink_fixed | s=0.9250000000000002 | 0.08889 | 0.09023 | 0.29493 | 0.07164 |
| logit_shrink | b=0.6 | 0.08444 | 0.09095 | 0.28857 | 0.07704 |
| logit_time | b0=0.6, b1=-2.7755575615628914e-17 | 0.08444 | 0.09095 | 0.28857 | 0.07704 |
| logit_time_vol | b0=0.7, b1=-0.05, b2=0.0 | 0.08444 | 0.091 | 0.28896 | 0.07464 |
| bucket_recal | decile recal, shrink n/(n+25) | 0.08464 | 0.0909 | 0.28261 | 0.02613 |

Best rule by FIT Brier: **logit_shrink** (beats raw on test: NO)

Calibration on test (market vs best rule):

| price bucket | n | mean market price | empirical freq | mean best-rule price |
|---|---|---|---|---|
| 0-.05 | 167 | 0.0076 | 0.024 | 0.0429 |
| .05-.1 | 18 | 0.0702 | 0.2778 | 0.1743 |
| .1-.2 | 18 | 0.1458 | 0.1667 | 0.2558 |
| .2-.3 | 12 | 0.2314 | 0.5833 | 0.3269 |
| .3-.4 | 11 | 0.3438 | 0.1818 | 0.404 |
| .4-.5 | 18 | 0.4432 | 0.2778 | 0.4658 |
| .5-.6 | 12 | 0.5519 | 0.8333 | 0.5313 |
| .6-.7 | 9 | 0.6568 | 0.6667 | 0.5963 |
| .7-.8 | 8 | 0.7606 | 1.0 | 0.6673 |
| .8-.9 | 6 | 0.8639 | 1.0 | 0.753 |
| .9-.95 | 2 | 0.92 | 1.0 | 0.8125 |
| .95-1 | 6 | 0.9778 | 1.0 | 0.914 |

### pooled_all_horizons  (fit n=2033, test n=1374, test base rate=0.2147)

| rule | params (fit only) | fit Brier | test Brier | test log score | test ECE |
|---|---|---|---|---|---|
| raw | — | 0.07276 | 0.07125 | 0.22507 | 0.04206 |
| shrink_fixed | s=0.9500000000000002 | 0.07251 | 0.07137 | 0.2356 | 0.04776 |
| logit_shrink | b=0.7000000000000001 | 0.07071 | 0.07189 | 0.2306 | 0.04503 |
| logit_time | b0=0.8000000000000002, b1=-0.05000000000000002 | 0.0706 | 0.07177 | 0.2303 | 0.04028 |
| logit_time_vol | b0=0.8, b1=-0.05, b2=0.0 | 0.0706 | 0.07177 | 0.2303 | 0.04028 |
| bucket_recal | decile recal, shrink n/(n+25) | 0.07136 | 0.07522 | 0.23184 | 0.02727 |

Best rule by FIT Brier: **logit_time** (beats raw on test: NO)

Calibration on test (market vs best rule):

| price bucket | n | mean market price | empirical freq | mean best-rule price |
|---|---|---|---|---|
| 0-.05 | 816 | 0.0075 | 0.011 | 0.0276 |
| .05-.1 | 85 | 0.0704 | 0.1647 | 0.142 |
| .1-.2 | 84 | 0.1439 | 0.2619 | 0.2279 |
| .2-.3 | 53 | 0.2414 | 0.3585 | 0.3166 |
| .3-.4 | 41 | 0.3515 | 0.2683 | 0.3963 |
| .4-.5 | 58 | 0.445 | 0.2759 | 0.4613 |
| .5-.6 | 48 | 0.5583 | 0.6667 | 0.541 |
| .6-.7 | 45 | 0.6513 | 0.7111 | 0.6093 |
| .7-.8 | 31 | 0.7479 | 0.9032 | 0.679 |
| .8-.9 | 29 | 0.8513 | 0.9655 | 0.774 |
| .9-.95 | 16 | 0.9208 | 1.0 | 0.8568 |
| .95-1 | 68 | 0.9876 | 1.0 | 0.9646 |

Raw market Brier by horizon (test, pooled): T-1d: 0.04344 (n=304), T-3d: 0.05419 (n=300), T-7d: 0.09051 (n=287), T-14d: 0.08419 (n=258), T-30d: 0.09217 (n=225)

## Kalshi

status=finalized -> bad_request; status=settled works but the feed is dominated by multivariate combo markets and last_price is null, so Kalshi is not scored. Polymarket is the scored source.

## Caveats

- Liquidity at close is ~0 for resolved markets; lifetime volume is the trust/liquidity proxy.
- Sample is the top-25-by-volume binary markets per month: it over-represents politics/crypto/sports event markets that attracted volume, and under-represents thin markets (survivorship/selection).
- Daily CLOB fidelity: the snapshot print can be up to 14 days older than its nominal horizon for illiquid stretches (freshness window); horizons are nominal, not exact.
- Snapshots of the same market at different horizons are correlated; pooled analysis counts them as samples, primary T-7 analysis does not.
- Markets within an event (e.g. mutually exclusive Fed-decision buckets) are correlated; no event-level clustering adjustment is applied.
- Kalshi not scored: settled feed dominated by multivariate combos with null last_price (see kalshi_probe).

## Files

- `predmarkets.py` — fetch + dataset + rules + scoring (numpy/pandas only).
- `results.json` — machine-readable version of everything above.
- `data/` — cached raw Gamma JSON per month, per-token CLOB histories, `markets.csv`, `snapshots.csv`.

## Interpretation (added after the run; numbers above are untouched)

- **The headline finding is negative and that is the result:** every
  shrinkage rule that looked good on the fit period (logit shrink
  improved fit Brier 0.0899 → 0.0844 at T-7) failed to beat the raw
  market on the held-out test period. The fit period's apparent
  under-confidence (fitted logit temperature b = 0.6–0.8, i.e. prices
  not extreme enough) did not persist into the test period — a regime
  shift, not a stable favourite–longshot bias you can harvest with a
  fixed rule.
- Two partial exceptions, neither a clean win: at T-7, `shrink_fixed`
  (s = 0.925) nominally edges raw on Brier (0.09023 vs 0.09051, a
  0.0003 gap well within noise) while *losing* on log score
  (0.29493 vs 0.28805); and `bucket_recal` improves T-7 log score
  (0.28261 vs 0.28805) and ECE (0.026 vs 0.078) while losing slightly
  on Brier (0.09090 vs 0.09051). No rule wins on both proper scores.
- Volume adds nothing here: the grid search set the volume coefficient
  b2 = 0 in both analyses, i.e. among already-high-volume markets,
  more volume did not mean prices needed less shrinkage.
- Accuracy improves sharply as expiry approaches (raw test Brier by
  horizon: T-30d 0.0922, T-14d 0.0842, T-7d 0.0905, T-3d 0.0542,
  T-1d 0.0434) — most of what looks like "miscalibration" at long
  horizons is unresolved uncertainty, not bias.
- Test-period calibration is noisy in the middle buckets (n = 6–18 per
  bucket at T-7) but directionally mixed: low-priced Yes outcomes hit
  *more* often than priced (5–10¢ bucket: 27.8% vs 7.0% mean price)
  while 40–50¢ outcomes hit *less* often (27.8% vs 44.3%) — there is
  no single monotone bias for a simple rule to exploit.
- Practical implication for the superforecaster ensemble: use raw
  Polymarket prices as-is for high-volume binary markets; do not apply
  a global shrinkage. Any future edge is more likely to come from
  category- or horizon-conditional models fitted on recent data only,
  re-fitted frequently, than from a static calibration map.
