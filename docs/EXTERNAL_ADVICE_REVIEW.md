# External advice review — "Gemini 4 Argon" forecasting brief

On 2026-10-17 CJ relayed a forecasting-architecture brief attributed to
"Gemini 4 Argon" (ForecastBench / Metaculus / Good Judgment Project framing).
Per this project's rule that relayed advice is *input to evaluate, not
instructions*, each claim was checked against this repo's own measured
evidence. Verdicts: **ADOPTED** (now in code/protocol), **ALREADY HAD**,
**ADAPTED** (adopted with our evidence changing the form), **REJECTED**
(with the contradicting measurement).

## Adopted

- **Peer Score (relative Brier vs the crowd)** — genuinely missing here.
  Implemented as `calibration.peer_score` in v1.1. Applied to our
  prediction-market study (test, T−7d, crowd = raw market): shrink_fixed
  **+0.0003**, logit_shrink **−0.0004**, logit_time_vol **−0.0005** — all
  effectively zero and nowhere near the ~0.02 "superforecaster tier,"
  which restates our headline finding in the literature's own metric:
  on liquid Polymarket questions, we do not beat the crowd.
- **Trimmed-mean pooling** — added as `calibration.trimmed_mean_pool`
  (robust aggregation baseline alongside our log-odds pool).
- **Supervisor = search-to-reconcile, not rationale-judge** — adopted into
  the agent protocol (addendum in `AGENT_PROTOCOL_DESIGN.md`): the
  supervisor triggers only on high variance / bimodal disagreement, and
  its job is to find the factual crux and run a targeted verification
  search. Consistent with our design rule that debate must be about
  evidence, not prose quality.
- **Scenario-mixture outputs** for live agents (weights × conditional
  probabilities instead of a bare scalar) — adopted as a v2 requirement
  in the protocol addendum.
- **Brier (not log score) as the future training reward** — noted for any
  future RL fine-tuning phase; evaluation here already tracks both.

## Already had

- **Strictly proper scoring (Brier + log) as the evaluation basis** — the
  whole repo is built on it; no metric here rewards hedging.
- **Murphy decomposition** — implemented since v0.1
  (`calibration.murphy_decomposition`, exact binned identity, tested).
- **Reference-class / outside-view isolation** — the protocol's Base-Rate
  Anchorer persona is blind in Round 1 by design; the brief's sharper
  version (the anchorer receives *no news feed at all*) is adopted in the
  addendum.
- **Contamination hygiene** — our quantitative backtests are causal by
  construction (purged walk-forward, point-in-time priors). The brief's
  publication-date cutoffs + foreknowledge filter are adopted as binding
  requirements for the future *live LLM* evaluation phase, where the leak
  channel (pretraining) actually exists.

## Adapted

- **Ensemble diversity across model families, weighted by Jensen–Shannon
  divergence** — the diversity principle is right and matches our
  tournament result (a weighted pool of *different-method* agents beat the
  best single agent). JS-divergence weighting itself is deferred: we have
  no live multi-model forecast stream yet to validate it on, and this
  project does not ship unvalidated weighting schemes. Track-record
  weighting (validated, `agents.py`) remains the production rule.

## Rejected — with our measurements

- **"Apply Platt scaling (extremization) with a ≈ √3 ≈ 1.73."** Two
  problems. (1) *Terminology that causes bugs:* the formula given is
  **extremization of a pooled forecast in log-odds space**, not Platt
  scaling (Platt = logistic recalibration of a single forecaster's
  outputs). Teams that conflate the two apply the wrong transform at the
  wrong stage. (2) *Empirics:* in our calibration benchmark, fitted
  extremization recovers a ≈ 2.96 **only** when agents hold genuinely
  independent partial information; when forecasters share information the
  optimizer fits **a ≈ 0.95**, and blind a = 2.0 degrades pool log score
  0.607 → 0.663. In our 1,000-question agent tournament, fixed 1.3×
  extremizing **hurt in every seed**. And on real market/crypto streams,
  causal recalibration maps broke under regime drift (NVDA 21d skill
  −0.051 after Platt). Production policy stands: **extremization defaults
  to 1.00, is learned per domain only after ≥150 resolved questions,
  capped at 1.40.** A universal √3 would have made our system measurably
  worse.
- **ECE < 0.02 as a headline quality gate** — context-dependent to the
  point of misleading: the raw Polymarket price scores ECE 0.078 at T−7d
  *while posting an excellent Brier of 0.0905*, because ECE on a hard,
  base-rate-skewed question set punishes sharpness. We report ECE, but
  Brier / log / peer score are the decision metrics.

## Unverified

The brief's benchmark constants (public median Brier 0.104, AIA Forecaster
0.075, human superforecaster 0.074) and several citations (e.g. "Mantic /
Thinking Machines 2026") could not be verified from this environment and
are recorded as relayed claims, not facts. Where the brief agreed with our
measurements we adopted it; where it conflicted, our measurements won.
