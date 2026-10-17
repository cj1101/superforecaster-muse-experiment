"""
Superforecaster — production policy engine (integration layer).

This module encodes, as executable policy, everything the v1 research
workstreams actually found. It does not contain a new model. It contains the
rules for *combining* models, priors and agent forecasts so that the system
only ever deviates from a strong prior as far as validated evidence allows.

Findings baked in (all numbers in ../results/):

1. MARKETS / CRYPTO DIRECTION: classic price-feature logistic models did NOT
   beat expanding climatology out-of-sample in any of the 12 equity cells or
   the 6 crypto direction cells (walk-forward, 2019-2026). Therefore a raw
   model probability is never shipped. It is shrunk toward the prior with a
   weight derived from its validated skill, and given zero weight unless
   that skill is positive. The v2 calibration-rescue experiment
   (results/calibration_rescue.json, 20 streams, fully causal) showed fitted
   shrinkage is the robust layer: skill vs climatology stayed inside
   [-0.009, +0.025] on every stream, converting models that lost up to
   -0.10 raw into approximately-climatology forecasts while preserving the
   crash-risk edge. Platt scaling applied on top HURT several real streams
   (NVDA 21d skill -0.051) despite shining on synthetic data — regime drift
   breaks a fixed recalibration map — so Platt is NOT in the production
   path; it may only be used causally-refit, per-stream, behind the same
   skill gate.

2. PREDICTION MARKETS: on 763 resolved Polymarket markets ($26.1B volume),
   no adjustment rule beat the raw market price out-of-sample on both proper
   scores (raw T-7d test Brier 0.09051). Therefore, when a liquid market
   price exists it IS the prior, with anchor weight >= 0.85, and model/agent
   views may only move it within the remaining weight unless they carry a
   validated track record in that domain.

3. CRYPTO CRASH RISK is the one validated model edge (BTC Brier 0.0901 vs
   base rate 0.0922, AUC 0.607; ETH 0.1417 vs 0.1446). Rare-event models are
   never class-balance-weighted (that destroyed calibration in testing:
   Brier 0.183 vs 0.092).

4. AGENT AGGREGATION: in a 1,000-question simulated tournament, a
   track-record-weighted log-odds pool beat the best single agent
   (Brier 0.1850 vs 0.1863; gap widens with n). Fixed extremizing (1.3x)
   HURT in every seed, and a separate calibration study found blind
   extremizing hurts whenever forecasters share information, while fitted
   extremization recovers the theory (a=2.96) when agents hold genuinely
   independent partial information. Therefore: extremization defaults to
   1.00 and may only be *learned* per domain after >=150 resolved questions,
   capped at 1.40, adopted only if it beats 1.00 held-out.

5. CALIBRATION research defaults (synthetic benchmarks): Platt scaling per
   forecaster (beta calibration when n >= ~2000), isotonic NOT default (it
   hurt a well-calibrated forecaster's log score at n=6,000), log score is
   the fitting objective everywhere. Production overrides per finding 1:
   on real drifting streams, fitted shrinkage toward the prior is the
   default transform; recalibration maps must re-earn their place causally.

Usage:
    sf = Superforecaster()
    p = sf.forecast(prior=0.62, model_probs=[0.71, 0.66],
                    model_skills=[0.03, -0.01])   # negative skill -> dropped
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logit

_EPS = 1e-6


def _clip(p):
    return float(np.clip(p, _EPS, 1 - _EPS))


def shrink(p_model: float, p_prior: float, w: float) -> float:
    """Linear shrink of a model probability toward a prior. w in [0, 1]."""
    w = float(np.clip(w, 0.0, 1.0))
    return _clip(w * p_model + (1 - w) * p_prior)


def fit_shrink_weight(y, p_model, p_prior, grid=None) -> float:
    """Fit the shrink weight w by minimizing Brier score on PAST forecasts.

    y, p_model, p_prior are arrays of already-resolved forecasts (each
    p_prior must be the prior that was known at that forecast's time).
    Returns w in [0, 1]; 0 means 'the model earned no deviation from prior'.
    """
    y = np.asarray(y, float); p_model = np.asarray(p_model, float); p_prior = np.asarray(p_prior, float)
    if grid is None:
        grid = np.linspace(0.0, 1.0, 41)
    best_w, best_b = 0.0, np.inf
    for w in grid:
        b = float(np.mean((y - (w * p_model + (1 - w) * p_prior)) ** 2))
        if b < best_b:
            best_b, best_w = b, float(w)
    return best_w


def skill_weight(brier_model: float, brier_prior: float) -> float:
    """Map validated Brier skill to a pool weight contribution.

    Positive skill only; a model at or below the prior gets weight 0.
    """
    if brier_prior <= 0:
        return 0.0
    return max(0.0, 1.0 - brier_model / brier_prior)


def log_odds_pool(probs, weights=None, extremization: float = 1.0) -> float:
    """Weighted log-odds (geometric) pool with optional extremization."""
    p = np.clip(np.asarray(probs, float), _EPS, 1 - _EPS)
    if weights is None:
        w = np.full(len(p), 1.0 / len(p))
    else:
        w = np.asarray(weights, float)
        w = w / w.sum() if w.sum() > 0 else np.full(len(p), 1.0 / len(p))
    z = float(np.sum(w * logit(p)) * extremization)
    return _clip(expit(z))


class Superforecaster:
    """Production forecaster implementing the v1 evidence-based policy."""

    MARKET_ANCHOR_MIN = 0.85      # finding 2: liquid market price dominates
    EXTREMIZE_DEFAULT = 1.00      # finding 4: never extremize blind
    EXTREMIZE_CAP = 1.40
    EXTREMIZE_MIN_RESOLVED = 150

    def __init__(self, extremization: float = EXTREMIZE_DEFAULT):
        if not (1.0 <= extremization <= self.EXTREMIZE_CAP):
            raise ValueError("extremization must be in [1.00, 1.40] and is "
                             "learned per-domain only after >=150 resolved questions")
        self.extremization = float(extremization)

    def forecast(self, prior: float, model_probs=None, model_skills=None,
                 agent_probs=None, agent_weights=None,
                 market_price: float | None = None) -> dict:
        """Combine a prior, quantitative models and agent forecasts.

        prior        base rate / climatology known at forecast time.
        model_probs  raw model probabilities; they are shrunk toward the
                     anchor here by validated skill (findings 1/5).
        model_skills validated Brier skill of each model vs the prior
                     (1 - brier_model/brier_prior); <= 0 models are dropped.
        agent_probs / agent_weights  independent agent forecasts and their
                     track-record weights (see agents.py).
        market_price if a liquid prediction-market price exists it replaces
                     the prior and is anchored (finding 2).
        """
        anchor = _clip(market_price) if market_price is not None else _clip(prior)
        components, weights = [], []

        if model_probs is not None:
            skills = model_skills if model_skills is not None else [0.0] * len(model_probs)
            for p, s in zip(model_probs, skills):
                w = skill_weight_from_skill(s)
                if w > 0:
                    # Finding 1: a model may only move the forecast from the
                    # anchor by its validated skill weight.
                    components.append(shrink(_clip(p), anchor, w)); weights.append(w)
        if agent_probs is not None:
            aw = agent_weights if agent_weights is not None else [1.0] * len(agent_probs)
            for p, w in zip(agent_probs, aw):
                if w > 0:
                    components.append(_clip(p)); weights.append(float(w))

        if not components:
            return {"p": anchor, "anchor": anchor, "n_components": 0,
                    "note": "no validated component; prior shipped unchanged"}

        pooled = log_odds_pool(components, weights, extremization=1.0)
        # Total deviation budget: with a market anchor, models+agents together
        # may only claim the non-anchor share unless every component agrees.
        if market_price is not None:
            pooled = shrink(pooled, anchor, 1.0 - self.MARKET_ANCHOR_MIN)
        p_final = log_odds_pool([pooled], [1.0], extremization=self.extremization)
        return {"p": p_final, "anchor": anchor, "pooled_components": pooled,
                "n_components": len(components),
                "extremization": self.extremization}


def skill_weight_from_skill(skill: float) -> float:
    """A model's validated Brier skill IS its max shrink weight (capped 0.5:
    no single unproven-in-production model may move a forecast more than
    halfway from the prior on its first validated edge)."""
    return float(np.clip(skill, 0.0, 0.5))


if __name__ == "__main__":
    sf = Superforecaster()
    # Demo 1: the NVDA 21d case from results/markets_results.json — the raw
    # model said 0.817 while its validated skill was negative (-0.007).
    print("NVDA 21d, raw model 0.817, validated skill -0.007:")
    print(" ", sf.forecast(prior=0.55, model_probs=[0.817], model_skills=[-0.007]))
    # Demo 2: crypto crash-risk — the one validated edge (skill +0.024 BTC).
    print("BTC crash-risk, model 0.16 vs base rate 0.10, validated skill +0.024:")
    print(" ", sf.forecast(prior=0.10, model_probs=[0.16], model_skills=[0.024]))
    # Demo 3: prediction-market anchored question.
    print("Market-priced question, market 0.62, agents [0.70, 0.66]:")
    print(" ", sf.forecast(prior=0.50, market_price=0.62,
                           agent_probs=[0.70, 0.66], agent_weights=[0.16, 0.14]))
