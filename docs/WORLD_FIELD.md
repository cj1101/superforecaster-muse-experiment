# WORLD FIELD — competitive landscape survey (verified 2026-10-17 via live sites)

Maintained by the competitive-intelligence division. Every figure below was
read from the live source on the date shown; paper-frozen figures are marked.
Rule: we never quote a number we could not read at its source.

## LLM trading arenas
- **Prophet Arena** — LIVE (prophetarena.co/leaderboard/forecast), 4,480
  resolved events, last scored Oct 15 2026, boards rescored every 3 days.
  #1 Claude Fable 5.1 (1−Brier 0.9484, edge +0.0606, ECE 0.0207, avg return
  95.98%); #2 GLM-5.3 (0.9479, +0.0621, ECE 0.0131); #3 Gemini 3.1 Pro (0.9474,
  best edge +0.0716, return 104.16%, ECE 0.0156); #4 Gemini 3.8 Flash (0.9471);
  #5 Claude Opus 5.5 (0.9467). No movement Oct 16→17.
- **Prediction Arena** — PAUSED (predictionarena.ai banner, checked Oct 17
  2026; last activity Jul 7 2026). Final all-time, $10K start: Gemini 3.1 Pro*
  +74.26% ($17,425.70), GLM 5* +69.42%, Claude Opus 4.6* +40.73%, GPT 5.4*
  +1.22% (* = paper trading), GLM 4.7 −15.40%, Grok 4.20 −20.04%, Claude Opus
  4.5 −26.62%, GPT 5.2 −26.78%, Grok 4.1 Fast −30.75%, Gemini 3 Pro −30.97%.
  Paper-era highlights (arXiv 2604.07355): grok-4-20 +15.5% over 17 days then
  −8.99% in 5h on correlated resolutions; forced-trade average −22.6%.
- **PolyBench** — NO live board. Official repo (now github.com/NostalgiaJohn/
  PolyBench) last commit Mar 10 2026; results identical to the paper (markets
  sampled Feb 6–12 2026, $10 lots): MiMo-V2-Flash CWR +17.6% (APY 4067%,
  Sharpe 0.02), Gemini-3-Flash +6.2%; Grok-4.1-Fast −12.1%, DeepSeek-V3.2
  −11.4%, Trinity-Large −9.2%, GPT-OSS-120B −17.8%, MiniMax-M2.5 −25.2%.
  All seven models' mean confidence 0.81–0.87 — overconfidence is the norm.
- **LiveTradeBench** — live board trade-bench.live UNREACHABLE Oct 17 2026
  (timeouts ×2). Latest official = paper Table 2 (live window Aug 18–Oct 24
  2025, 21 LLMs, 50 trading days). Equities: GPT-4.1 CR 6.25% / Sharpe 2.64 /
  WR 65.3%; GPT-o3 6.04 / 2.57. Polymarket: Qwen3-235B-Thinking CR 57.62% /
  Sharpe 1.81 / **MDD 72.10%**; GPT-o3 54.84 / 3.68 / MDD 60.99; Kimi-K2 Sharpe
  5.26 with MDD 54.74. Paper's own note: Sharpe correlation between the two
  markets ≈ 0 — skill does not transfer across venues.

## Accuracy champions
- **Good Judgment Superforecasters** (goodjudgment.com, read Oct 17 2026):
  FOMC Brier since Jan 2023 (N=29): SFs **0.031** vs CME FedWatch **0.058**
  (46% more accurate; 4th consecutive year beating the futures market).
  Historical: ~30% more accurate than intelligence analysts with classified
  access (IARPA tournament, 2011–2015; 500 questions, 1M+ forecasts).
- **ForecastBench** (forecastbench.org/leaderboards, "last updated
  2026-10-17"): Tournament board (380 entries, Brier Index 0–100): Google
  DeepMind "fire hedgehog" 69.0, "ceramic-kettle" 68.9, "blue croc" 68.7,
  human Superforecaster median tied-3rd 68.7, FutureSearch bull_v4 68.5.
  Baseline board (130 entries): Superforecaster median #1 at 67.7, public
  median 62.6, best raw LLM Kimi-K3 61.9; Random Uniform 41.9 (128/130).
  Site notes LLM–superforecaster parity achieved June 6, 2026; human
  comparison groups last surveyed July 2024.
- **Metaculus** — UNVERIFIED: metaculus.com is behind a Cloudflare Turnstile
  challenge that our browser could not pass (Oct 17 2026). Community
  Prediction accuracy and AI-bot tournament standings are tracked as an open
  intel task; no figures quoted until read at source.

## Algorithmic establishment
- **GIFT-Eval** (Hugging Face, Salesforce; read Oct 17 2026, no explicit
  update date on page; 24 datasets, 144K series): #1 EXAONE-Forecast-Agent
  (LG AI Research, agentic) MASE 0.610 / CRPS 0.419. TimesFM-3 (Google,
  zero-shot) 0.667 / 0.456; Toto-2.0-FnF (Datadog, agentic) 0.676 / 0.463;
  STRIDE+Chronos-2 0.674 / 0.454; MoiraiAgent 0.689 / 0.476; Chronos-2 (AWS)
  0.698 / 0.485. Agentic systems occupy the top rows.
- **Polymarket systematic traders** (Finance Magnates, Mar 16 2026): "many of
  the most profitable wallets are automated trading systems"; 14/20 most
  profitable traders are bots (per Stacy Muur); arbitrage ≈ $40M extracted in
  a year from structural inefficiencies; wallet 0x8dxd ≈ $300 → $400K+ in a
  month on ultra-short crypto contracts, winning on reaction speed.
  (Mar 12 2026): hedge funds entering Kalshi via prime brokers Clear Street
  and Marex; Tradeweb distributing Kalshi data institutionally.
- **Théo ("Trump whale")** — WSJ Nov 13 2024: ≈ $85M profit on Polymarket
  2024 election markets via systematic purchases driven by his own polling
  (neighbor-effect) analysis; accounts Fredi9999, Theo4, PrincessCaro, Michie.
- **M6 Financial Forecasting Competition** (2022–2023, Makridakis): overall
  winner Filip Staněk; Investment Decisions category won by Colin Catlin with
  a fully automated AutoTS pipeline (his own write-up, Apr 3 2023);
  Forecasting (RPS) winner Dan Weitzenfeld.
- **Named firms**: no prediction-market firm with a verified public
  algorithmic track record was found — the public record is wallets and
  phenomena, not firms. Intel division keeps this as a standing watch item.

## What this means for us (intel synthesis, 2026-10-17)
1. Accuracy frontier (Prophet Arena/ForecastBench): edge +0.06–0.07 over
   market at ECE ~0.02, Brier ≈ 0.05 on shared events — the bar for our
   Stage-2/3 pipeline on prediction questions.
2. Profit frontier is structural: speed, arbitrage, market-making, and
   cross-venue dislocations — a Stage-1/4 opportunity class our fleet
   underweights today; intel hypotheses queued accordingly.
3. Discipline frontier: everyone who posts big raw returns (LiveTradeBench
   +57.6%) carries 55–72% drawdowns; N_star's penalty makes our honest 0.00
   competitive with most of the published field and our risk rules are the
   differentiator once any real edge is found.
4. Human bar: superforecasters beat the CME futures market by 46% on FOMC —
   event/macro questions with process discipline are winnable; our
   price-blind forecaster stage should be benchmarked on FOMC-style
   questions against FedWatch as a public proxy market.
