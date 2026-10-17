# Argon Brief #3 — the N_star north star (relayed by CJ, 2026-10-17)

> Relayed advice. CJ's ruling: "Treat it as advice, but do adopt the new north
> star benchmark." The N_star objective below is therefore ADOPTED as the
> fleet's objective; the surrounding claims were verified where checkable (all
> seven arXiv citations are real papers with matching titles, verified
> 2026-10-17) and the rest is treated as advice. Our internalized spec, with
> the points where our own measurements override the brief, is in
> docs/NORTH_STAR.md. The accompanying harness (alpha_kelly_harness.py) was
> uploaded by CJ, verified runnable, and adopted as superforecaster/alpha_kelly.py.

---

To make money across equities, prediction markets, and crypto, your agent
fleet must stop optimizing for standalone forecasting accuracy (like raw Brier
score) and instead optimize for **market-relative, slippage-adjusted
log-wealth growth**.

As proven mathematically in Prophet Arena (Appendix B.4)
(https://arxiv.org/html/2510.17638v1), an agent with a *worse* Brier score can
generate *higher* financial returns than a better-calibrated agent if its
error lies on the profitable side of the market-implied probability q.
Real-money trials in Prediction Arena (March 2026)
(https://arxiv.org/html/2604.07355v1) and PolyBench (April 2026)
(https://arxiv.org/html/2604.14199v1) confirm that frontier models routinely
lose −16.0% to −30.8% of their capital when they treat trading as a static
quiz rather than pricing order-book depth, bid-ask spreads, and capital
velocity.

## Existing benchmarks to anchor the fleet

| Benchmark | Domains | Core metrics | Mechanism to adopt |
| --- | --- | --- | --- |
| Prediction Arena (2604.07355) | Live Kalshi & Polymarket | Mark-to-market account value at liquidation **bid**, total PnL, settlement win rate, max drawdown | 15% per-market cost-basis concentration limit; curated (Kalshi) vs open discovery (Polymarket); rolling "learning section" of recent wins/losses every 15–45 min |
| PolyBench (2604.14199) | 38,666 Polymarket binary markets w/ CLOB | Confidence-Weighted Return (CWR), APY = r_i × 365/D, Sharpe μ_r/σ_r, Temporal Resolution Error | Sweep the live CLOB ask book with confidence-scaled lots (B_k = c_k × L); explicit SKIP when conviction < 0.60 or EV negative |
| Prophet Arena (2510.17638) | 1,367+ live multi-horizon events (72,136 markets) | 3D triad: Brier, ECE, CRRA utility-maximizing market return U_γ(w) | Halved time horizons t_k = τ − 2^−k·Δ_0; CRRA closed-form sizing vs market-implied odds q_ij |
| LiveTradeBench (2511.03628) | US equities (15 stocks + cash) & Polymarket | Cumulative return, Sharpe, MDD, win rate, volatility | POMDP; agents output a simplex portfolio allocation vector a_t (Σa = 1, incl. cash) each step |
| CryptoBench (2512.00417) & CryptoTrade (2407.09546) | On-chain, DeFi, DEX, derivatives | Avg success rate across 4 quadrants (Simple/Complex × Retrieval/Prediction) ±5% tolerance; reflective PnL/Sharpe | Separate raw on-chain retrieval (whale/KOL flows, funding, OI, unlocks, liquidation heatmaps) from 24h/7d predictive synthesis |
| FinTradeBench (2603.19225) & TradingAgents | Equities, SEC filings, macro | OOS risk-adjusted return, Sharpe, fundamental-to-technical synthesis accuracy | Specialized analyst roles (Fundamental, Sentiment, Technical/Microstructure, Risk, Execution) debate before committing capital |

## The custom north star: Alpha-Kelly Composite Benchmark (N_star)

Fleet mission statement: *"Your purpose is not to have an opinion on every
event; your purpose is to compound net liquidation wealth by finding, sizing,
and executing only those opportunities where our calibrated posterior
probability p holds a verifiable information edge over the execution-adjusted
market price q_exec after spreads, slippage, fees, and capital lockup cost."*

### Trade level: execution-adjusted log-wealth gain
- Calibrated posterior p_k ∈ [0.02, 0.98].
- q_exec,k: VWAP to fill the target allocation across the CLOB ask depth + fees.
- Information edge: D_KL(p_k || q_exec,k) = p·log(p/q) + (1−p)·log((1−p)/(1−q)) —
  the theoretical maximum asymptotic compounding rate (Cover; Meister 2024).
- Sizing: full Kelly f* = (p − q_exec)/(1 − q_exec) is fragile (estimation
  error ε degrades wealth linearly, over-betting penalizes quadratically) →
  mandate quarter-to-half Kelly λ ∈ [0.25, 0.50], or Prophet Arena CRRA with
  γ ∈ [0.5, 0.8], hard-capped at 15% of account per correlated event cluster.

### Fleet level: N_star (rolling 30-day window)
N_star = (Σ π_bid,k / Σ I_k) × sqrt(365 / mean_lock_days) × exp(−10 · ECE) − 2.0 · max(0, MDD − 0.15)

- Bid-marked CWR (Prediction Arena accounting: value at liquidation bid).
- Capital velocity multiplier (PolyBench APY / temporal resolution).
- Calibration multiplier exp(−10·ECE), 10 probability bins (Prophet Arena).
- Hard drawdown penalty beyond 15% MDD (LiveTradeBench / Prediction Arena).

### Six operational directives
1. Default to SKIP unless |p − q_exec| ≥ 0.05 net of spread + slippage.
   (Prediction Arena: the top-returning model, +6.02% in 3 days on Polymarket,
   executed zero trades on curated Kalshi markets.)
2. Separate Market Discovery agents from Pricing & Execution agents
   (−22.6% avg when forced onto 29 efficient Kalshi series vs −1.1% avg with
   discovery over thousands of Polymarket contracts).
3. Crypto: require CryptoBench on-chain retrieval checks (funding, OI, whale
   flows, liquidations) + 0.5× Kelly dampener (LLMs hold rigid 80–90%
   confidence from text news alone → catastrophic returns, PolyBench §5.2).
4. Scale size inversely with slippage: query top-5 CLOB depth; cap order size
   where marginal ask crosses the minimum edge (PolyBench Fig. 3: +17.6% CWR
   at $10 lots collapses to losses at $1,000 lots).
5. Hold fundamental edge to settlement unless thesis-invalidating news
   (early exiters — 51–83% of trades — averaged negative PnL; Prediction
   Arena §8.4).
6. Cap correlated exposure at the factor level: the #1 model in Prediction
   Arena Event 1 lost −8.99% of portfolio in five hours when correlated
   positions resolved adversely together (§8.7).

### Harness verification results (Argon's fixture suite, reproduced by us exactly)
8-market synthetic suite, both fleets sharing Brier 0.2643:
- Disciplined: N_star **+0.1059**, ending $12,778.38 (+27.78%), CWR_bid +54.14%,
  MDD 6.09%, 5/8 traded, 80% win rate.
- Naive: N_star **−0.1499**, ending $9,326.36 (−6.74%), CWR_bid −7.84%,
  MDD 21.76%, 8/8 traded, 50% win rate.
SKIPs fired on: insufficient net edge (KALSHI-CPI-CORE-0.3), failed on-chain
gate (CRYPTO-ALT-PUMP-48H), unresolved agent dispersion (POLY-GEOPOL-AMBIGUOUS);
the illiquid trap was capped at $24.24 invested.

NOTE (ours): the suite is synthetic and hand-priced — it validates the
machinery, not real-world performance. Also noted: ECE 0.3736 is identical for
both fleets (shared forecasts), so the calibration multiplier crushes both;
the divergence comes from the gates, sizing, and drawdown penalty.
