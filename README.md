# Superforecaster

> **Maintenance rule (CJ, 2026-10-17): every commit must fully update this README.**
> A stale section is a defect. If a claim below cannot be reproduced from the code
> and data as committed, the claim — not the code — is what gets fixed in that commit.

# ⭐ NORTH STAR — $\mathcal{N}_{star}$ (Alpha-Kelly Composite): **0.0000**

**Production $\mathcal{N}_{star}$ = 0.0000 — VERIFIED under EC-13 corrected quoting
(Wave 3, 2026-10-17):** every ps1/r8 headline is restated on per-book eligible-set
ECE (see `results/ec13_requote_RESULTS.md`); the maximum eligible $\mathcal{N}_{star}$
over all 50 cells is 0.0000, attained only by zero-trade cells (best trading cell
−0.0532). Canonical citations: ps1 S2a **−0.0949** / S3 **−0.3759** / S4 **−1.7736**;
r8 R1 S2a **−0.2070** / S3 **−0.3681** / S4 **−1.1710**. Traded-subset headlines are
retired to diagnostics. **Canonical scorer (designated v2.0):**
`superforecaster/alpha_kelly.py` (`evaluate_fleet_north_star`, as written) —
capital-weighted lock, $D_{lock} = \sum_k I_k D_k / \sum_k I_k$, with the 1-day lock
floor. The harness copy (`superforecaster_out/nstar/nstar.py`, trade-weighted) is the
deviant scorer; it is not repo code, and the repo contains no trade-weighted
$\mathcal{N}_{star}$ surface, so no repo formula changed in v2.0. Cross-scorer
comparisons are valid only at uniform lock until the harness is migrated (one-expression
stake-weighted change + 1-day floor, per the Wave-3 velocity-canonical migration note).

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
| **Metaculus / FutureEval** AI benchmark + bot tournaments (live-read 2026-10-18) | Pros **34.22** > Community **25.21** > best AI (Claude Fable 5 High) **16.10** peer score; Spring 2026 bot tourney: 173 bots, top GreeneiBot2 18.90; Pros-vs-bots gap collapsed −20.03 → −1.25 (n.s.) | Humans still lead on Metaculus's own benchmark; Fall 2026 bot tournament LIVE ($50K, to Jan 6 2027) — entry is a CJ-account action |

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

## Metaculus bot — open source

The fleet's tournament bot **kodaMuseSuperforecaster** (entered in the Fall 2026
FutureEval Bot Tournament under the account RiverineCairn) lives in
[`metaculus_bot/`](metaculus_bot/) and is **MIT-licensed, open source** (CJ's
decision, 2026-10-17) — API client, price-blind intake, payload builders for
binary / multiple-choice / numeric / group questions, and an offline self-test.
It reads its API token from the `METACULUS_API_TOKEN` environment variable only;
no credentials are in the tree. Public repo:
https://github.com/cj1101/superforecaster-muse-experiment/tree/main/metaculus_bot

## Where this fleet stands (2026-10-17, all numbers from real runs)

- **$\mathcal{N}_{star}$ = 0.0000 — certified, not an absence of trying.** Across
  220 slots, ~200 workstreams, and **2,967 counted statistical comparisons (r53
  deflation registry), the fleet holds ZERO validated tradeable claims.** The only
  Wave-2 result surviving global Bonferroni is r6 — a *negative* (T2 recalibration
  on direction streams is significantly harmful). The fleet takes ~no trades and
  preserves capital. In a field where the published norm is −16% to −31% of
  capital, a verified zero with an honest measurement stack is a real position —
  and the score to beat.
- **Wave 2's final stack test (w20) did not beat production v1:** the frozen v2
  stack scored pooled Brier 0.22808 vs v1's 0.22846, Δ +0.00038, 95% CI
  [−0.00013, +0.00090] — the CI includes 0, so **production v1 STANDS** and no
  "production v2" ships on this evidence. The reproduction gate passed exactly
  (0.22846 / 0.22904 / 0.23355), so the null is a property of the stack, not of
  the measurement.
- First real-data $\mathcal{N}_{star}$ backtest (287 Polymarket test markets,
  real cost model, live-book-validated): trading the market against itself
  scores exactly 0.000 (no manufactured profit); our Wave-1 shrink rule is
  *structurally untradeable* under this benchmark (max deviation 0.0375 < the
  0.05 hurdle); the one apparently-positive rule (+1.19) is a variance-driven
  longshot harvest (19.3% win rate, top-3 trades = 108% of profit, 58.8%
  drawdown) — reported as an artifact, not an edge.
- Calibration scoreboard (now a gate, not the goal): production Brier
  **0.22846** vs benchmark 0.22904, raw models 0.23355, over 37,589 forecasts
  (as-shipped production policy; corrected-policy research figure **0.22814**,
  r12 — dual-labelled per governance decision 2026-10-17, never a silent swap;
  relabelling the headline remains an M10/PM decision); the fleet's one
  validated quant edge is crypto crash-risk (BTC Brier 0.0901 vs 0.0922,
  AUC 0.607 — quoted with all three denominators in the Wave-2 closures below).
- **ECE governance (r12):** production pooled ECE-10U-eligible = **0.0240** is
  **telemetry only**. Every candidate book gates on its **own rows'**
  ECE-10U-eligible, with the r12 bias floor attached (never subtracted). Quoting
  the pooled constant as a book's ECE would have flipped 6 S2a cells positive
  under EC-13 re-quoting — it must never gate a book.

## Wave-2 closures (2026-10-17 synthesis; full record in the fleet ledger)

Wave 2 mapped where edges are **not**, and closed those lines:

- **Markets direction is signal absence, not an estimator problem** (r26/r28/
  r30/r45/r56): every estimator fails on the same rows; the 21-day equity
  direction-feature research line is **retired**. The −0.0024 markets gap is a
  data problem (no causal feature family in 40+ attempts carries signal), not
  a tuning problem. Production keeps fitted shrinkage + climatology discipline.
- **T2-on-direction harm rule (r6)** — the fleet's only global-Bonferroni
  survivor: temperature (T2) recalibration on direction streams is
  significantly harmful (+0.00973, CI excl. 0). Fleet rule: never apply
  T2/temperature transforms to direction streams (the production transform is
  fitted shrinkage; see the engine policy below). The r18 transform table v2
  is **not** re-routed into production — w20's pooled test did not promote it.
- **Crash family — real but narrower than Wave 1 believed** (c4/m34/r35/r45/
  r47): c4 v2 improves on Wave 1 (BTC 0.08925, ETH 0.13996) and w20's crash
  domain is the stack's only positive domain (+0.00244 [+0.00048, +0.00434]),
  but the gain is ETH-carried and shrinks against stronger denominators —
  x1's v1-vintage lead +0.00249 is +0.00112 vs causal-primary and +0.00021 vs
  pure decay. Any crash claim must carry all three denominators (v1 /
  causal-primary / decay). The crash model stays a **research emitter**;
  promotion is governed solely by r35's frozen pre-registration (below).
- **Extremization / pooling — closed:** learned-only, a = 1.00 default, cap
  1.40 (r15: a = 1.00 in 324/324 blocks). Engine enforces the [1.00, 1.40]
  range with the ≥150-resolved-questions learning rule stated in code.
- **Prediction markets — the price is the forecast:** raw price beat every
  adjustment on every fresh frame (peer 0.0000 at T−7d, n=287). Ladder
  underdispersion is **dead in all three families** (weather, crypto, econ).
  Maker strategies are **closed** (markout tax exceeds any edge at every
  horizon). The p9 resolution-risk auditor survives only as a **REJECT veto,
  never a PASS-select** (r31/r37). Carried into Stage 4 only: p9-as-REJECT-veto,
  r32's size curves as clip caps, M4's unified cost model as the q_exec basis.
- **Sports moneylines (r22/r58) — CLOSED-DEAD:** the disjoint confirmation
  scored the generation lead's primary cell at −0.047 [−0.112, +0.012] (n=198)
  against the generation window's +0.073; all three pre-committed kill
  conditions fired (winner's curse). No forward ask-side sports design will
  be built.
- **EC-14 execution gates — implemented (v1.8):** Gate D (conviction:
  p_side ≥ 0.60 or p_side/q_side ≥ 1.35) and Gate A (crypto requires on-chain
  verification) are implemented on the `sf edge` CLI path, with λ default
  0.35 (crypto effective 0.175 via the 0.5 CryptoBench dampener). Verified
  2026-10-17: `sf edge --p 0.55 --q-exec 0.45` → SKIP, LR 1.222 (Gate D);
  `sf edge --p 0.58 --q-exec 0.42` → TRADE, LR 1.381; crypto without
  `--onchain-verified` → SKIP (Gate A), with it → TRADE. Repo test suite
  green at this commit (calibration suite + 14/14 backtest checks).
- **EC-09 SKIP-Kelly fix — landed (v2.1):** the `sf edge` SKIP payload
  previously returned a nonzero `kelly_fraction_of_account` (e.g. 0.0636 on
  the Gate-D SKIP above) because the fraction was computed before the gate
  decisions and never reset — a SKIP that appears to size a position is a
  footgun for any consumer of the JSON contract. Fixed: **every SKIP
  payload now carries `kelly_fraction_of_account` == 0.0 and
  `expected_log_growth_at_kelly_size` == 0.0** (the sibling sizing field,
  audited in the same pass; `side` was already null on SKIP, and the KL
  fields stay populated — they are information diagnostics, not sizing).
  The same class of leak in `alpha_kelly.py` — the CLUSTER_CAP_REACHED and
  ORDER_BOOK_DEPTH_EXHAUSTED SKIP returns carrying the pre-cap/pre-sweep
  `kelly_fraction_raw` while every other SKIP branch zeroed it — is fixed
  the same way (their `allocated_fraction` / `invested_capital` were
  already 0). **No TRADE-path number changed** (re-verified at this commit:
  0.58/0.42 → TRADE kelly 0.0966, growth 0.02995; crypto verified →
  0.0483 / 0.01665; 0.80/0.50 → capped 0.15 / 0.07931), EC-14 gates and
  λ = 0.35 are untouched, and the alpha_kelly fixture benchmark is
  unchanged (disciplined +0.1059 vs naive −0.1499). Regression coverage:
  `tests/test_cli_edge.py` — 16/16 checks asserting kelly == 0.0 on every
  SKIP branch of both surfaces (insufficient edge, Gate D long/short,
  Gate A, dispersion, Gate 3, cluster cap, depth exhausted) and the exact
  pre-fix values on TRADE branches; full suite green at this commit
  (calibration suite + 14/14 backtest checks + 16/16 edge checks).

## Live forward programmes (the only remaining paths to a validated positive $\mathcal{N}_{star}$)

- **r35 — frozen crash confirmation:** dual block-30/60 bootstrap, both CIs
  must exclude 0; honest power is a 2030s readout (D1 at 4,000 origins
  ≈ 2032, D2 at 20,000 ≈ 2054), with DEAD and INCONCLUSIVE-PERMANENT bounds
  pre-committed — silence is also an answer.
- **FWD-1 — as of 2026-10-17: preregistration frozen; preconditions 4/8 PASS
  with closures in flight.** Preregistration combined_sha256
  `3c4448f0706035d612e5f14638c6c9400db6f12b1f700aa58f3323f22dc3cdf9` (frozen,
  hash-anchored). Enrollment remains prohibited until all 8 day-1
  preconditions pass. Cadence reality: Stage A reaches n=250 in ≈15+ weeks
  under the frozen gates. No integration change until a Stage-A book meets
  the F15 promote bar — and even then a positive paper $\mathcal{N}_{star}$
  is a LEAD, not validation.
- **r29 — forward accrual:** triple-barrier/path-risk (family C: pooled H2
  skill +0.0401, deflated CI [+0.0055, +0.0774]) and vol-scaled sign accrue
  fresh post-2026-10-16 origins under r29's frozen protocol on the collector
  substrate; both remain LEADs (family-pool claims only) until accrual
  re-tests them.
- **Substrate:** the consolidated forward collector (S1–S9 in one file:
  Deribit, tracked set, OI, crash emitter, resolutions, NWS archive, CBOE
  options) is accruing on schedule — first `scheduled-r14-hourly` manifest
  line 2026-10-18T00:53:51Z. CJ's **harvest-and-archive policy** is in force
  fleet-wide: never fetch-and-discard; blocked sources fall back to the
  newest archive snapshot, disclosed in results.

## The road to #1

The discovery surface is exhausted (2,967 comparisons, zero validated
tradeable claims); what remains is forward evidence. The fleet's work now
runs on the three pre-registered forward programmes above (r35, FWD-1, r29
accrual) against a single gap: a posterior that clears **+0.05 post-cost
edge with calibration intact**, scored on its own book's eligible-set ECE
under the canonical scorer. Every claim this README makes is reproduced
from code and data in this repo — when $\mathcal{N}_{star}$ moves, it will
move here first.

## Calibration scoreboard — HEADLINE BRIER (secondary: gates + ECE feed) — Production: **0.22846** · Benchmark: 0.22904 · Raw models: 0.23355 · Corrected-policy (research): 0.22814
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
6. **Execution gates (`sf edge`, EC-14, v1.8).** A trade must clear the
   |p − q_exec| ≥ 0.05 hurdle, **Gate D** (conviction: p_side ≥ 0.60 or
   p_side/q_side ≥ 1.35), and — for crypto — **Gate A** (on-chain
   verification required). Sizing is fractional Kelly at λ = 0.35 by
   default (crypto effective 0.175 via the 0.5 dampener), capped at 15%
   of the account per correlated cluster. **A SKIP sizes zero (EC-09,
   v2.1): every SKIP payload carries `kelly_fraction_of_account` == 0.0
   and zero expected log growth at Kelly size.** Candidate books are scored on
   their **own eligible-set ECE** (EC-13); the pooled ECE 0.0240 is
   telemetry only and never gates a book.

## Repository layout

```
superforecaster/   calibration.py · backtest.py · markets.py · predmarkets.py
                   crypto.py · agents.py · engine.py (production policy)
                   alpha_kelly.py (canonical N_star scorer) · cli.py (./sf)
experiments/       run_markets_backtest.py · run_crypto_backtest.py · calibrate_oos.py
tests/             calibration tests + 14 backtest checks + 16 CLI edge /
                   EC-09 SKIP-Kelly regression checks (all passing)
results/           every number above, machine-readable + written up;
                   ec13_requote_RESULTS.md + ec13_requote_grid.csv +
                   ec13_requote_results.json = EC-13 corrected N_star tables
data/              price histories, out-of-sample prediction CSVs, market snapshots
docs/              agent protocol design (personas, debate, aggregation, cadence);
                   NORTH_STAR.md (spec) · WORLD_FIELD.md (verified field survey)
```

Fleet artefacts outside this repo (referenced by the Wave-2 synthesis):
`superforecaster_out/wave2/SYNTHESIS.md` (Wave-2 synthesis + integration
checklist) · `superforecaster_out/wave2/LEDGER.md` (207-row workstream ledger)
· `superforecaster_out/wave2/r53/` (deflation registry, 2,967 comparisons) ·
`superforecaster_out/wave2/fwd1/manifest.json` (FWD-1 freeze manifest,
combined_sha256 `3c4448f0…c3cdf9`) · `superforecaster_out/wave3/ec13-requote/`
(EC-13 source) · `superforecaster_out/wave3/velocity-canonical/` (canonical
scorer designation + migration note).

## Run it

```bash
pip install -r requirements.txt          # numpy / pandas / scipy only
python3 tests/test_calibration.py        # 9 checks
python3 tests/test_backtest.py           # 14 checks
python3 tests/test_cli_edge.py           # 16 checks (EC-14 gates + EC-09 SKIP-Kelly)
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
- **v1.2** — headline production Brier 0.22846 pooled over 37,589 forecasts; objective restated as market prediction dominance
- **v1.3** — NORTH STAR CHANGE: fleet objective is now $\mathcal{N}_{star}$ (Alpha-Kelly Composite); Brier demoted to gate/diagnostic
- **v1.4** — agent CLI (`sf`: headline/forecast/edge/pool/score/schema, JSON contract) + alpha_kelly $\mathcal{N}_{star}$ harness adopted as canonical engine (fixture benchmark verified reproducible; synthetic, not evidence)
- **v1.5** — north star internalized: `docs/NORTH_STAR.md` (adopted spec + our adaptations) + `docs/ARGON_BRIEF_3_NSTAR.md` archived; 7 benchmark papers added to `literature/`
- **v1.6** — headline contextualized: what $\mathcal{N}_{star}$ means + the world field + fleet standing and road to #1; fleet playbook adopted (`docs/FLEET_NORTH_STAR_PROMPT.md`)
- **v1.7** — world field = LIVE records (browser-verified 2026-10-17); `docs/WORLD_FIELD.md` full survey; Metaculus marked unverified (bot-walled) at that time
- **v1.8** — EC-14 (CLI leg): Gate D + Gate A implemented in `sf edge` (r51 repro cases pass: 0.55/0.45 SKIP LR 1.222, 0.58/0.42 TRADE LR 1.381), λ default 0.35; README dual-labelled Brier (production 0.22846 / corrected-policy 0.22814)
- **v1.9** — Metaculus/FutureEval VERIFIED at source (live browser read 2026-10-18): Pros 34.22 > Community 25.21 > best AI 16.10 peer; Spring 2026 bots 173, Pros-vs-bots gap −20.03 → −1.25 n.s.; Fall 2026 bot tournament LIVE; `docs/WORLD_FIELD.md` + README updated
- **v2.0** — Wave-2 synthesis + Wave-3 verified outputs integrated: production $\mathcal{N}_{star}$ **0.0000 VERIFIED under EC-13 corrected (eligible-set ECE) quoting** (canonical citations ps1 S2a −0.0949 / S3 −0.3759 / S4 −1.7736; r8 R1 S2a −0.2070 / S3 −0.3681 / S4 −1.1710; corrected tables vendored in `results/ec13_requote_*`); **canonical scorer designated** — capital-weighted `alpha_kelly.py` as written (repo holds no trade-weighted $\mathcal{N}_{star}$ surface; the deviant harness copy is not repo code, no formula change); w20 final stack test did not beat production v1 (v1 stands); Wave-2 closures documented (markets signal absence, T2-on-direction harm rule r6, extremization/pooling closures, ladders dead in all families, maker closed, p9 = REJECT veto only, r58 sports CLOSED-DEAD); forward programmes documented (r35 frozen crash test, FWD-1 frozen preregistration with preconditions 4/8 PASS as of 2026-10-17, r29 accrual, consolidated S1–S9 collector + harvest-and-archive policy); EC-14 gates stated as implemented in v1.8 and re-verified at this commit; Metaculus section kept VERIFIED (v1.9); README maintenance rule carried in the header
- **v2.1** — EC-09 SKIP-Kelly fix: `sf edge` computed the Kelly fraction before the EC-14 gate decisions and emitted it unchanged on SKIP, so every SKIP branch (insufficient net edge, Gate D conviction, Gate A crypto on-chain) returned a nonzero `kelly_fraction_of_account` (0.0636 on the Gate-D repro) plus nonzero expected log growth at that size. Fix: SKIP now forces the fraction to 0.0 in `cli.py` after the gates decide, and the two `alpha_kelly.py` SKIP returns that leaked the pre-cap/pre-sweep `kelly_fraction_raw` (cluster cap, depth exhausted) now return 0.0 like every other SKIP branch. No TRADE-path number, gate, or λ changed (TRADE controls re-verified byte-identical; fixture benchmark unchanged at +0.1059 / −0.1499); $\mathcal{N}_{star}$ headline unchanged at 0.0000. New regression suite `tests/test_cli_edge.py` (16/16) covers every SKIP branch of both surfaces plus TRADE controls; full suite green (calibration + 14/14 backtest + 16/16 edge); `docs/AGENT_INTERFACE.md` contract updated to state SKIP sizes zero

## Limitations, stated plainly

Daily-bar price features may simply not contain the information needed to
beat base rates in these domains; the sample windows include the unusual
2020–2026 regimes; prediction-market sampling is top-volume (selection bias
toward politics/crypto/sports); agent-tournament results are simulated
(synthetic agents with a shared shock), not live LLM forecasts. After
Wave 2, the honest frontier is no longer another backtest: it is the three
pre-registered forward programmes (r35, FWD-1, r29 accrual) accruing fresh
evidence on the collector substrate — readouts are slow by design (r35's
honest horizon is the 2030s), and no backtest result in this repo should
be read as a shortcut around them.
