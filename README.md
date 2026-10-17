# Superforecaster

# ⭐ NORTH STAR — $\mathcal{N}_{star}$ (Alpha-Kelly Composite): **0.00**

## What this number is — and why it is the only score that matters here

Most "AI forecasting" leaderboards grade a *quiz*: how close were your
probabilities to what happened (Brier score)? This project grades a **trader**:
**would following this system's advice, at realistic prices, with real order
books, have compounded money?** $\mathcal{N}_{star}$ is that grade:

$$\mathcal{N}_{star} = \mathrm{CWR}_{bid} \times \sqrt{\frac{365}{\bar{D}_{lock}}} \times e^{-10\cdot\widehat{ECE}} - 2.0\cdot\max(0,\; MDD-0.15)$$

- **CWR_bid** — profit per dollar actually invested, with every open position
  valued at its *liquidation bid* (what you could sell for right now), not the
  flattering mid-price.
- **Velocity** — a 3% edge that locks capital for 9 months is worth less than
  one that resolves in a week; the score annualizes for lockup time.
- **Calibration multiplier** — a lucky gambler is indistinguishable from a
  skilled forecaster until you check calibration; ECE of 0.15 slashes the
  score by ~73%, so uncalibrated confidence cannot game it.
- **Drawdown penalty** — blow past a 15% peak-to-trough loss and the score
  pays for it twice over. Solvency discipline is part of the grade.

Trades are allowed only when a calibrated posterior $p$ beats the
**execution-adjusted** market price $q_{exec}$ (the volume-weighted price of
actually filling your size through the order book, plus fees) by ≥ 0.05 —
otherwise the system must **SKIP**. Sizing is fractional Kelly
(λ ∈ [0.25, 0.50]), capped at 15% of the account per correlated event cluster,
with crypto bets halved. In theory (Cover; Meister 2024), the most wealth can
compound per trade is exactly the information edge $D_{KL}(p \parallel q_{exec})$
— this benchmark is that theory, enforced with real books and real costs.
Mission: *compound net liquidation wealth only where our calibrated posterior
holds a verifiable information edge over the execution-adjusted market price.*
Engine: `superforecaster/alpha_kelly.py` · Interface: `./sf` CLI (`docs/AGENT_INTERFACE.md`)
· Full spec: `docs/NORTH_STAR.md`.

## The world field — live records (checked 2026-10-17; every figure dated)

$\mathcal{N}_{star}$ is a composite of the hardest public benchmarks in AI
forecasting/trading (papers in `literature/`; full sourced survey in
`docs/WORLD_FIELD.md`). Two kinds of champions defend this field, and the
published record is sobering for both: **most frontier AI systems lose money
live, and the biggest real profits come from speed and structure, not from
better opinions.**

**The trading arenas (LLM agents, real or realistic books):**

| Arena | Status | Best verified record | The lesson |
| --- | --- | --- | --- |
| **Prophet Arena** (UChicago/USC/Meta) | **LIVE** board, 4,480 resolved events, scored thru Oct 15 2026 | #1 Claude Fable 5.1: Brier ≈ 0.052, ECE 0.021, edge over market **+0.061**; #3 Gemini 3.1 Pro posts the best edge (**+0.072**) and return (104%) | The accuracy bar: edge +0.06–0.07 over the market at ECE ≈ 0.02 |
| **Prediction Arena** (Harvard/Arcada) | **PAUSED** — final standings, data thru Jul 7 2026 ($10K start) | Gemini 3.1 Pro **+74.3%**, GLM 5 +69.4%, Claude Opus 4.6 +40.7% (top 3 were paper trading); bottom: Gemini 3 Pro −31.0% | Even the "live money" arena's winners were paper; the real-money tail is brutal |
| **PolyBench** (38,666 Polymarket markets, real CLOB) | No live board; repo frozen Mar 2026 = latest official | Only **2 of 7** models positive: MiMo-V2-Flash **+17.6% CWR**, Gemini-3-Flash +6.2%; leader's alpha collapsed as size grew toward $1,000 | Slippage is the silent tax; size is a strategy variable |
| **LiveTradeBench** (UIUC) | Live board unreachable (Oct 17 2026); latest official = paper window Aug–Oct 2025 | GPT-4.1 best equities (+6.25% cum.); Qwen3-235B best Polymarket (+57.6%) — with a **72.1% max drawdown** | Raw return without drawdown discipline is a liability, and $\mathcal{N}_{star}$ prices it |

**The accuracy champions (humans & platforms):**

| Champion | Record | Note |
| --- | --- | --- |
| **Good Judgment Superforecasters** (humans) | FOMC decisions since 2023: Brier **0.031 vs CME FedWatch 0.058** — 46% more accurate than the futures market, 4th straight year; historically ~30% more accurate than intelligence analysts with classified access | The calibration gold standard — and they beat a *market*, which is exactly our game |
| **ForecastBench** (live, updated Oct 17 2026) | Tournament board: Google DeepMind systems 1–3 (Brier Index 69.0/68.9/68.7); human Superforecaster median tied 3rd (68.7); baseline board: Superforecaster median **#1** (67.7) ahead of every raw LLM (best 61.9). "LLM–superforecaster parity achieved June 2026" | Agent systems now match elite humans on accuracy; nobody has yet matched them *and* the market, net of costs |
| **Metaculus** community + AI bot tournaments | **Unverified** — site bot-walled on 2026-10-17; we do not quote numbers we could not read | Tracked by the intel division |

**The algorithmic establishment (non-LLM):**

| System class | Record | Note |
| --- | --- | --- |
| **Agentic time-series systems** (GIFT-Eval, live HF board) | LG EXAONE-Forecast-Agent #1 (MASE 0.610 / CRPS 0.419); best pure foundation models trail (TimesFM-3 0.667, Chronos-2 0.698) | Agents around models beat models alone — architecture is the edge |
| **Polymarket bots** (the real money) | 14 of the 20 most profitable wallets are automated (Finance Magnates, Mar 2026); arbitrage bots extracted ≈ $40M in a year; one wallet $300 → $400K+ in a month on ultra-short crypto contracts — on **speed, not foresight**; hedge funds entering Kalshi via prime brokers | The profit bar is structural: latency, arbitrage, market-making |
| **Systematic humans** | "Trump whale" Théo ≈ **$85M** (WSJ) via systematic polling-model buying; M6 competition's investment category won by a fully automated AutoTS pipeline | Discipline + a model + size, again — not punditry |

**The defending champion is the market price itself.** Across our own 763
resolved Polymarket markets, no adjustment rule beat the raw price
out-of-sample (peer score 0.0000). Beating it net of spread, slippage, and
fees — with calibration intact — is the whole game, and it is what
$\mathcal{N}_{star}$ scores.

## Where this fleet stands (2026-10-17, all numbers from real runs)

- **$\mathcal{N}_{star}$ = 0.00** — the fleet has found **no validated net edge
  yet**, so it takes ~no trades and preserves capital. In a field where the
  published norm is −16% to −31% of capital, a verified zero with an honest
  measurement stack is a real position — and the score to beat.
- First real-data $\mathcal{N}_{star}$ backtest (287 Polymarket test markets,
  real cost model, live-book-validated): trading the market against itself
  scores exactly 0.000 (no manufactured profit); our Wave-1 shrink rule is
  *structurally untradeable* under this benchmark (max deviation 0.0375 < the
  0.05 hurdle); the one apparently-positive rule (+1.19) is a variance-driven
  longshot harvest (19.3% win rate, top-3 trades = 108% of profit, 58.8%
  drawdown) — reported as an artifact, not an edge.
- Calibration scoreboard (now a gate, not the goal): production Brier
  **0.2285** vs climatology 0.2290 over 37,589 forecasts; the fleet's one
  validated quant edge is crypto crash-risk (BTC Brier 0.0901 vs 0.0922,
  AUC 0.607).

## The road to #1

200 workstreams run 24/7 against a single gap: a posterior that clears
**+0.05 post-cost edge with calibration intact**. The hunt is ordered the way
the arenas say edge actually lives: (1) discovery across thousands of
neglected contracts instead of famous markets, (2) order-book-native signals
(depth, slippage, cross-venue dislocations), (3) crypto signals gated on
on-chain data, (4) equity catalyst/event trades sized like prediction
contracts, (5) calibration engineering that protects the ECE multiplier.
Every claim this README makes is reproduced from code and data in this repo —
when $\mathcal{N}_{star}$ moves, it will move here first.

## Calibration scoreboard — HEADLINE BRIER (secondary: gates + ECE feed) — Production: **0.2285** · Benchmark: 0.2290 · Raw models: 0.2336
*Pooled over 37,589 scored forecasts (n-weighted). Production = the forecasts this
system would actually ship: skill-gated, shrinkage-disciplined models for markets &
crypto; the market price itself for prediction markets. Lower is better.
Objective: **market prediction dominance** — drive the Markets skill below zero-gap
and keep it there. Updated by the fleet as new backtests land.*

| Domain | Production Brier | Benchmark Brier | Skill | n |
|---|---|---|---|---|
| Markets (priority 1) | 0.2438 | 0.2432 (climatology) | **−0.0024 ← the gap to close** | 17,964 |
| Prediction markets (priority 2) | 0.0905 | 0.1733 (climatology) | peer vs market: 0.0000 | 287 |
| Crypto (priority 3) | 0.2163 | 0.2167 (climatology) | +0.0021 | 19,338 |


An evidence-first forecasting system built by a fleet of 200 AI agent workstreams
running continuously, in the
priority order **(1) markets, (2) prediction markets, (3) crypto** — with a
calibration/aggregation core and a Tetlock-style multi-agent protocol on top.

**Design philosophy:** a forecast only earns the right to move away from a
strong prior (base rate, or a liquid market price) by the exact amount of its
*validated, out-of-sample* skill. Everything below is measured, walk-forward,
and reported whether it flatters the models or not. Several headline models
**lose** to simple benchmarks; the system's job is to know that, quantify it,
and never ship the losing version.

> Research only. This system produces probabilities, not trades or bets.
> No number in this README is simulated or illustrative — each comes from an
> actual run whose script, data and full output live in this repo.

---

## Scoreboard (all out-of-sample, walk-forward)

### 1 · Markets — equity direction, 2016-10-17 → 2026-10-16 (Yahoo, daily)
Expanding walk-forward (train ≥756d, retrain every 21d, labels horizon-purged),
n ≈ 1,750 forecasts per cell. Full tables: `results/markets_RESULTS.md`.

| Symbol | Horizon | Model Brier | Climatology Brier | Skill | AUC |
|---|---|---|---|---|---|
| ^GSPC | 1d / 5d / 21d | .2518 / .2506 / .2437 | .2482 / .2400 / .2172 | −.014 / −.044 / −.122 | .486 / .470 / .453 |
| QQQ | 1d / 5d / 21d | .2500 / .2475 / .2435 | .2463 / .2411 / .2262 | −.015 / −.027 / −.077 | .505 / .500 / .472 |
| NVDA | 1d / 5d / 21d | .2490 / .2460 / .2290 | .2484 / .2426 / .2275 | −.002 / −.014 / −.007 | .521 / .504 / **.538** |
| TSM | 1d / 5d / 21d | .2523 / .2591 / .2632 | .2501 / .2478 / .2388 | −.009 / −.045 / −.102 | .510 / .458 / .447 |

**Verdict:** classic price features (momentum, mean reversion, RSI, vol
regimes) do **not** beat the base rate for index direction, and are
anti-predictive at 21 days. NVDA is the lone symbol with consistently >0.5
AUC — a weak ranking signal, not yet a probability edge.

### 2 · Prediction markets — Polymarket calibration, 763 resolved markets
Jan 2024 → Oct 2026, $26.1B combined volume; fit/test split by close date
(457 / 306 markets). Forecast = CLOB price 7 days before close.
Full study: `results/predmarkets_RESULTS.md`.

| Forecaster (test, T−7d) | Brier | Log score | ECE |
|---|---|---|---|
| **Raw market price** | **0.09051** | **0.28805** | 0.0778 |
| Best adjustment rule fitted on fit-split (logit shrink, b=0.6) | 0.09095 | 0.28857 | — |

**Verdict:** the market is very good (Brier falls to 0.0434 at T−1d) and
**no adjustment rule — favourite/longshot shrink, time/liquidity/logit
variants — beat it out-of-sample on both proper scores.** A fit-period
under-confidence pattern did not persist into the test period. Production
rule: a liquid market price is the prior, anchored at ≥85% weight.

### 3 · Crypto — BTC & ETH, Coinbase daily (BTC from 2015, ETH from 2016)
Strict walk-forward (≥1,095d training, labels horizon-purged), n ≈ 2,800 /
2,500 per horizon. Full tables: `results/crypto_RESULTS.md`.

| Model | Brier | Benchmark Brier | Skill | AUC |
|---|---|---|---|---|
| BTC P(up 1d / 7d / 30d) | .2508 / .2537 / .2813 | .2507 / .2501 / .2479 | ≈0 / − / − | .529 / .494 / **.431** |
| ETH P(up 1d / 7d / 30d) | .2499 / .2507 / .2577 | .2499 / .2501 / .2509 | ≈0 / − / − | .541 / .530 / .508 |
| **BTC P(crash >10% in 7d)** | **0.0901** | 0.0922 (base rate 10.0%) | **+0.024** | **0.607** |
| **ETH P(crash >10% in 7d)** | **0.1417** | 0.1446 (base rate 17.0%) | **+0.020** | **0.577** |

**Verdict:** direction is a coin flip (30-day BTC momentum is
anti-predictive, AUC 0.431), expected-return regression is worse than the
mean — but **crash risk is real signal**: volatility/drawdown features
identify danger regimes even when they can't identify direction.

### Calibration layer (synthetic benchmarks with known ground truth)
`results/calibration_RESULTS.md` — 9/9 tests pass.

- Overconfident forecaster: Brier 0.2179 → **0.2037**, ECE 0.1106 → **0.0153** (Platt; recovered the true distortion almost exactly)
- Biased forecaster: Brier 0.2372 → **0.2039**, ECE 0.1767 → **0.0108**
- Pooling agents with *independent partial information*: extremized log-odds pool log score 0.5819 → **0.5116**, Brier → **0.1709** (fitted extremization a = 2.96, theory ≈ 3)
- Pooling agents with *shared* information: blind extremizing **hurts** (log 0.6070 → 0.6629 at a=2.0); the optimizer correctly fits a ≈ 0.95

### Calibration rescue (v2, real streams — the production transform)
Causal Platt / fitted-shrinkage applied to all 20 model streams
(`results/calibration_rescue_RESULTS.md`):

- **Fitted shrinkage toward the prior is the robust layer**: skill vs
  climatology ∈ [−0.009, +0.025] on *every* stream — losing models are
  converted into ≈climatology instead of shipped raw, and the crash edge
  survives (BTC +0.0247).
- Causal Platt on real streams is **unreliable** (NVDA 21d skill −0.051):
  regime drift breaks fixed recalibration maps. Not in the production path.

### Agent ensemble (simulated tournament, 1,000 questions, known truth)
`results/agents_RESULTS.md`, protocol in `docs/AGENT_PROTOCOL_DESIGN.md`.

| Aggregation | Test Brier | Log score |
|---|---|---|
| **Track-record-weighted log-odds pool** | **0.1850** | **0.5507** |
| Best single agent | 0.1863 | 0.5574 |
| Equal-weight log-odds / mean / median | 0.1867 / 0.1870 / 0.1875 | — |
| Fixed 1.3× extremized pool | 0.1899 | 0.5698 |
| Oracle (knows true p) / climatology | 0.1760 / 0.2500 | — |

**Verdict:** weighting agents by their resolved track record beats the best
individual and every naive pool; fixed extremizing hurts in every seed, so
extremization ships at 1.00 and may only be *learned* per domain after ≥150
resolved questions.

### Backtesting engine validation
`results/backtest_RESULTS.md` — 14/14 checks. On synthetic data with real
signal the engine certifies skill (Brier 0.1712 vs 0.2518, bootstrap CI
excludes 0); on pure noise it refuses to (p = 1.0); declared lookahead leaks
are caught by assertion, and a smuggled leak is exposed by impossible
performance (Brier 0.0001 / AUC 1.0000).

---

## How the production engine decides (`superforecaster/engine.py`)

1. **Anchor.** Prior = base rate, or the liquid market price when one exists.
2. **Skill gate.** A model contributes only if its walk-forward Brier skill
   vs the anchor is positive — and only up to that skill (capped at 0.5).
   *Example: the raw NVDA 21-day model currently says P(up) = 0.817
   (2026-10-16 close); its validated skill is −0.007, so the engine ships
   the prior instead of the 0.817.*
3. **Shrink, don't recalibrate blindly.** Model outputs are shrunk toward
   the anchor by fitted weights — the one transform that never did real
   damage across 20 live streams.
4. **Pool agents by track record** (weighted log-odds, extremization 1.00
   until learned per-domain).
5. **Crash risk is a first-class output**, because it's the one place the
   models demonstrably know something direction models don't.

## Repository layout

```
superforecaster/   calibration.py · backtest.py · markets.py · predmarkets.py
                   crypto.py · agents.py · engine.py (production policy)
experiments/       run_markets_backtest.py · run_crypto_backtest.py · calibrate_oos.py
tests/             9 calibration tests + 14 backtest checks (all passing)
results/           every number above, machine-readable + written up
data/              price histories, out-of-sample prediction CSVs, market snapshots
docs/              agent protocol design (personas, debate, aggregation, cadence)
```

## Run it

```bash
pip install -r requirements.txt          # numpy / pandas / scipy only
python3 tests/test_calibration.py        # 9 checks
python3 tests/test_backtest.py           # 14 checks
python3 superforecaster/engine.py        # production-policy demo
python3 experiments/calibrate_oos.py     # v2 rescue experiment (uses cached data)
python3 experiments/run_markets_backtest.py   # full re-run (fetches live Yahoo data)
python3 experiments/run_crypto_backtest.py    # full re-run (fetches live Coinbase data)
```

## Data sources

Yahoo Finance chart API (equities) · Coinbase Exchange API (crypto, longest
clean history; Kraken cross-checked, Binance geo-blocked) · Polymarket
Gamma + CLOB APIs (resolved-market study; Kalshi probed — its settled feed
carried no usable prices, so it is documented but not scored) · Stooq is
bot-blocked and not used.

## Version history

- **v0.1** — calibration & scoring core (Platt / isotonic / beta, Murphy decomposition, pooling optimizer)
- **v0.2** — leakage-proof walk-forward backtesting engine
- **v0.3** — markets model + full 10-year backtest (honest negative result)
- **v0.4** — prediction-market calibration study (market wins; anchor rule adopted)
- **v0.5** — crypto models (direction fails; crash-risk edge found)
- **v0.6** — agent ensemble protocol + simulated tournament (weighted pool wins)
- **v1.0** — production policy engine + causal calibration-rescue experiment (shrinkage adopted, Platt demoted), this README
- **v1.1** — external advice review (`docs/EXTERNAL_ADVICE_REVIEW.md`): peer score + trimmed-mean pool adopted and measured (all adjustment rules score ≈0 peer vs the market — the crowd stands); supervisor search-to-reconcile, anchorer isolation, contamination filters and scenario-mixture outputs adopted into the protocol; blanket √3 extremization **rejected** against our measurements

## Limitations, stated plainly

Daily-bar price features may simply not contain the information needed to
beat base rates in these domains; the sample windows include the unusual
2020–2026 regimes; prediction-market sampling is top-volume (selection bias
toward politics/crypto/sports); agent-tournament results are simulated
(synthetic agents with a shared shock), not live LLM forecasts. The next
frontiers, in order: richer market features (options-implied, rates,
positioning), live agent forecasting on real questions with scored track
records, and prediction-market *cross-venue* disagreement signals.
