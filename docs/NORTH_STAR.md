# NORTH STAR — internalized spec (adopted by CJ, 2026-10-17)

Source: "Gemini 4 Argon" brief #3 (full text: docs/ARGON_BRIEF_3_NSTAR.md),
adopted by CJ with the instruction "treat it as advice, but do adopt the new
north star benchmark." The fleet objective is therefore:

## Mission
> "Your purpose is not to have an opinion on every event; your purpose is to
> compound net liquidation wealth by finding, sizing, and executing only those
> opportunities where our calibrated posterior probability p holds a verifiable
> information edge over the execution-adjusted market price q_exec after
> spreads, slippage, fees, and capital lockup cost."

## The score (rolling 30-day window, maximize)
N_star = CWR_bid × sqrt(365 / mean_lock_days) × exp(−10 · ECE) − 2.0 · max(0, MDD − 0.15)

- CWR_bid: Σ bid-marked trade profits / Σ capital invested (positions valued at
  liquidation bid, never mid).
- Velocity multiplier rewards short capital lockup.
- Calibration multiplier: ECE (10-bin) — ECE 0.02→0.15 cuts the score ~73%.
- Drawdown penalty only beyond 15% MDD.

## Trade-level rules
- Edge per trade (theory): D_KL(p || q_exec); max asymptotic growth = the KL.
- q_exec: VWAP to fill the target size through the ask book + fees.
- SKIP unless |p − q_exec| ≥ 0.05 after all costs. SKIP is the default.
- Sizing: fractional Kelly f = λ·(p − q_exec)/(1 − q_exec), λ ∈ [0.25, 0.50]
  (harness default 0.35); hard cap 15% of account per correlated cluster
  (factor level, not contract level); crypto bets × 0.5 dampener.
- Hold fundamental edge to settlement unless thesis-invalidating news.
- Scale size inversely with book slippage; never sweep past the rung where
  marginal ask eats the minimum edge.
- Separate Discovery agents (scan thousands of contracts) from Pricing &
  Execution agents (curated lists are where LLMs lost −22.6% avg in trials).

## Benchmarks the brief anchors to (all verified real on arXiv, 2026-10-17;
## PDFs in Drive Literature + repo literature/)
- Prophet Arena 2510.17638 — Brier/ECE/CRRA triad; worse-Brier-can-win proof (App. B.4).
- Prediction Arena 2604.07355 — live Kalshi/Polymarket, bid-marked account value, 15% cluster cap.
- PolyBench 2604.14199 — 38,666 Polymarket markets, CLOB sweep sim, CWR, SKIP at conviction < 0.60.
- LiveTradeBench 2511.03628 — equities + Polymarket POMDP, CR/Sharpe/MDD.
- CryptoBench 2512.00417 (+ CryptoTrade 2407.09546) — on-chain retrieval gates before prediction.
- FinTradeBench 2603.19225 (+ TradingAgents) — analyst-role debate fleet for equities.
- Meister 2024, 2412.14144 — Kelly criterion for prediction markets (KL growth theory).

## OUR adaptations (where our measurements override the brief — same policy as briefs #1/#2)
1. **Production N_star today = 0.00.** Our validated edge vs the market is
   peer ≈ 0.0000 (763 resolved Polymarket markets), so honest fleet behavior
   is ~100% SKIP with capital preserved. The brief's fixture result
   (disciplined +0.1059) is a synthetic self-test of Argon's harness, adopted
   as machinery — NOT our score and never quoted as performance.
2. **Posterior cap:** engine outputs stay capped [0.01, 0.99] (adopted brief
   #2); the harness clips trade-evaluation posteriors to [0.02, 0.98].
3. **Platt:** the harness ships post-hoc Platt for agent/LLM outputs only.
   Our causal experiments ban Platt on quant model streams (drifting streams
   were hurt, skill −0.051); quant posteriors use fitted shrinkage.
4. **ECE quirk noted:** in the fixture run both fleets share ECE 0.3736, so
   the calibration multiplier (0.024×) crushes both equally — the fixture
   demonstrates the gates, not calibrated forecasting. Our ECE comes from
   real scored streams (headline feed) once the paper portfolio trades.
5. **Execution is paper-only.** CJ places any real trades himself. The
   harness's live adapters (Polymarket CLOB, Kalshi, Coinbase) are read-only
   data feeds for q_exec and marking.
6. **Interface:** all fleet agents consume the system through the `./sf` CLI
   (docs/AGENT_INTERFACE.md; `./sf schema` is the machine-readable contract).
   The canonical N_star engine is superforecaster/alpha_kelly.py; the real-data
   backtest harness lives in experiments/backtest_nstar.py (fleet-built).

## Current standing (2026-10-17)
- N_star (production): **0.00** — no validated net edge yet; capital preserved.
  Frontier-LLM reference point from the cited trials: −16% to −31% capital
  when trading without an edge. Zero, honestly held, is the bar to beat.
- Calibration scoreboard (secondary): Production Brier 0.22846 vs benchmark
  0.22904, n=37,589. Markets skill −0.0024 (gap #1), crypto +0.0021,
  prediction markets peer 0.0000.
