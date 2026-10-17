"""
agents.py — Agent Ensemble / Superforecaster Aggregation + Scoring Layer
=========================================================================
Pure numpy + stdlib. No network, no credentials, no external data.

Implements:
  * Forecast / AgentForecast objects
  * Log-odds (logit) pooling with per-agent weights
  * Extremizing (Satopää et al. 2014)
  * Brier / log-score scoring and track-record weight updating
  * A simulated tournament: synthetic agents with known skill / noise /
    bias profiles forecasting >=500 synthetic questions whose generating
    probabilities are known, comparing aggregation strategies.

Run:
    python3 agents.py
    python3 agents.py --n-questions 2000 --seed 7

Design notes (see DESIGN.md for the full protocol):
  - Independent round-1 forecasts are pooled in log-odds space because
    probabilities are not additive; averaging in logit space respects
    the geometry of evidence (each unit of log-odds = a likelihood ratio).
  - Extremizing corrects the systematic under-confidence of averaged
    forecasts when forecasters share information (they double-count the
    common prior and dilute private signals).
  - Track-record weights are a proper-scoring-rule exponential update:
        w_i  ∝  exp(-lambda * mean_brier_i)   renormalised to sum to 1
    with shrinkage toward equal weights early in a forecaster's history,
    so a lucky streak on 5 questions cannot capture the pool.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-6  # probability clipping bound for logit / log-score safety


# ---------------------------------------------------------------------------
# Core probability utilities
# ---------------------------------------------------------------------------

def clip_p(p: np.ndarray | float) -> np.ndarray | float:
    return np.clip(p, EPS, 1.0 - EPS)


def logit(p: np.ndarray | float) -> np.ndarray | float:
    p = clip_p(np.asarray(p, dtype=float) if not np.isscalar(p) else float(p))
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid (scalar- and array-safe)."""
    if np.isscalar(x):
        x = float(x)
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x)) if x < 700 else 1.0
        ex = math.exp(x) if x > -700 else 0.0
        return ex / (1.0 + ex)
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean Brier score for binary forecasts. Lower is better. Range [0, 1]."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    return float(np.mean((probs - outcomes) ** 2))


def log_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean log score (negative log-likelihood per question). Lower is better."""
    probs = clip_p(np.asarray(probs, dtype=float))
    outcomes = np.asarray(outcomes, dtype=float)
    return float(-np.mean(outcomes * np.log(probs) + (1.0 - outcomes) * np.log(1.0 - probs)))


def calibration_stats(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """Simple ECE-style calibration summary."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        mask = (probs >= edges[b]) & (probs < edges[b + 1] if b < n_bins - 1 else probs <= edges[b + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.mean()) * abs(probs[mask].mean() - outcomes[mask].mean())
    return {"ece": float(ece), "mean_forecast": float(probs.mean()), "base_rate": float(outcomes.mean())}


# ---------------------------------------------------------------------------
# Forecast objects
# ---------------------------------------------------------------------------

@dataclass
class AgentForecast:
    """One agent's forecast on one question."""
    agent_id: str
    question_id: str
    probability: float                 # in (0, 1)
    round: int = 1                     # 1 = independent, 2 = post-debate revision
    rationale: str = ""
    strongest_counterevidence: str = ""  # required in debate round (DESIGN.md §4)
    confidence_note: str = ""

    def __post_init__(self) -> None:
        if not (0.0 < self.probability < 1.0):
            self.probability = float(np.clip(self.probability, EPS, 1 - EPS))

    @property
    def log_odds(self) -> float:
        return float(logit(self.probability))


@dataclass
class Question:
    """Question specification (see DESIGN.md §1 for the full schema)."""
    question_id: str
    text: str
    resolution_criteria: str
    horizon_days: int
    base_rate: Optional[float] = None        # reference-class frequency, if known
    base_rate_source: Optional[str] = None
    asset_class: str = "general"             # markets | prediction_markets | crypto
    resolves_at: Optional[str] = None        # ISO date string


@dataclass
class ForecasterRecord:
    """Running track record for one agent, used for pool weighting."""
    agent_id: str
    briers: List[float] = field(default_factory=list)
    log_scores: List[float] = field(default_factory=list)

    def add(self, probability: float, outcome: int) -> None:
        p = float(np.clip(probability, EPS, 1 - EPS))
        self.briers.append((p - outcome) ** 2)
        self.log_scores.append(-(outcome * math.log(p) + (1 - outcome) * math.log(1 - p)))

    @property
    def n(self) -> int:
        return len(self.briers)

    @property
    def mean_brier(self) -> float:
        return float(np.mean(self.briers)) if self.briers else 0.25  # ignorance prior

    @property
    def mean_log_score(self) -> float:
        return float(np.mean(self.log_scores)) if self.log_scores else math.log(2)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def mean_pool(probs: Sequence[float]) -> float:
    return float(np.mean(np.asarray(probs, dtype=float)))


def median_pool(probs: Sequence[float]) -> float:
    return float(np.median(np.asarray(probs, dtype=float)))


def log_odds_pool(probs: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
    """
    Weighted log-odds (geometric-odds) pool.

        logit(p_pool) = sum_i w_i * logit(p_i) / sum_i w_i
        p_pool = sigmoid(logit(p_pool))

    Equal weights when `weights` is None.
    """
    p = clip_p(np.asarray(probs, dtype=float))
    lo = np.log(p / (1.0 - p))
    if weights is None:
        pooled_lo = float(np.mean(lo))
    else:
        w = np.asarray(weights, dtype=float)
        if w.sum() <= 0:
            w = np.ones_like(w)
        w = w / w.sum()
        pooled_lo = float(np.dot(w, lo))
    return float(sigmoid(pooled_lo))


def extremize(p: float, factor: float = 1.30, centre: float = 0.5) -> float:
    """
    Extremize a pooled probability away from `centre` in log-odds space.

        logit(p_ext) = logit(centre) + factor * (logit(p) - logit(centre))

    factor = 1  -> no change
    factor > 1  -> push away from centre (corrects crowd under-confidence)
    factor < 1  -> pull toward centre

    The canonical GJP finding (Satopää et al. 2014; Mellers et al. 2014)
    is that optimal factors for human crowds are typically ~1.2–2.0
    depending on shared-information overlap; for LLM-agent ensembles with
    genuinely diverse methods a more conservative 1.15–1.45 is the sane
    search range, selected on a burn-in / validation split — never assumed.
    """
    p = float(np.clip(p, EPS, 1 - EPS))
    c = float(np.clip(centre, EPS, 1 - EPS))
    lo = math.log(p / (1 - p))
    lo_c = math.log(c / (1 - c))
    return float(sigmoid(lo_c + factor * (lo - lo_c)))


def track_record_weights(
    records: Sequence[ForecasterRecord],
    lam: float = 12.0,
    shrinkage_n: float = 30.0,
    floor: float = 0.02,
) -> np.ndarray:
    """
    Exponential proper-score weights with shrinkage to equal weights.

        raw_i = exp(-lam * mean_brier_i)
        alpha_i = n_i / (n_i + shrinkage_n)          # trust grows with history
        w_i = alpha_i * raw_i + (1 - alpha_i) * (1/K)  then floored + renormalised

    `lam` controls how aggressively past Brier differences translate into
    weight. lam=12 means a 0.05 Brier edge multiplies weight by exp(0.6)≈1.82×
    before shrinkage — material but not winner-take-all. The floor keeps
    every persona in the pool (diversity is itself predictive; dropping the
    contrarian because of a bad quarter destroys the ensemble's value).
    """
    k = len(records)
    if k == 0:
        return np.array([])
    equal = 1.0 / k
    raw = np.array([math.exp(-lam * r.mean_brier) for r in records], dtype=float)
    raw = raw / raw.sum() if raw.sum() > 0 else np.full(k, equal)
    alphas = np.array([r.n / (r.n + shrinkage_n) if r.n > 0 else 0.0 for r in records])
    w = alphas * raw + (1.0 - alphas) * equal
    w = np.maximum(w, floor)
    return w / w.sum()


def aggregate(
    forecasts: Sequence[AgentForecast],
    strategy: str = "weighted_extremized",
    weights: Optional[Sequence[float]] = None,
    extremize_factor: float = 1.30,
) -> float:
    """Dispatch helper used by the production protocol (DESIGN.md §5)."""
    probs = [f.probability for f in forecasts]
    if strategy == "mean":
        return mean_pool(probs)
    if strategy == "median":
        return median_pool(probs)
    if strategy == "log_odds":
        return log_odds_pool(probs)
    if strategy == "extremized":
        return extremize(log_odds_pool(probs), extremize_factor)
    if strategy == "weighted_log_odds":
        return log_odds_pool(probs, weights)
    if strategy == "weighted_extremized":
        return extremize(log_odds_pool(probs, weights), extremize_factor)
    raise ValueError(f"unknown strategy: {strategy}")


def select_extremize_factor(
    pooled_probs: np.ndarray, outcomes: np.ndarray, grid: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """Grid-search the extremizing factor minimising Brier on a validation set."""
    if grid is None:
        grid = np.round(np.arange(0.8, 2.05, 0.05), 2)
    best_f, best_b = 1.0, float("inf")
    for f in grid:
        ext = np.array([extremize(float(p), float(f)) for p in pooled_probs])
        b = brier_score(ext, outcomes)
        if b < best_b:
            best_b, best_f = b, float(f)
    return best_f, best_b


# ---------------------------------------------------------------------------
# Synthetic tournament
# ---------------------------------------------------------------------------

@dataclass
class SyntheticAgent:
    """
    A synthetic forecaster with known parameters.

    Generative model per question (in log-odds space):
        L          = logit(p_true)
        common     ~ N(0, rho)          shared shock (everyone read the same news)
        eps_i      ~ N(0, sigma_i)      private noise
        logit(p_i) = bias_i + beta_i * L + common + eps_i

    beta_i < 1  : under-reacts to the true signal (shrinks toward 50%)
    beta_i > 1  : over-reacts (overconfident)
    sigma_i     : idiosyncratic noise — lower is more skilled
    bias_i      : systematic directional bias in log-odds
    """
    agent_id: str
    persona: str
    beta: float
    sigma: float
    bias: float = 0.0


# Persona roster mirroring DESIGN.md §2. Parameters are illustrative but
# ordered by the qualitative claims in the design: the Bayesian updater is
# the best single forecaster; the contrarian is weak alone but decorrelated;
# the base-rate anchorer is well calibrated but dull; the news hunter is
# high-variance; the novice is present to prove weighting can suppress noise.
AGENT_ROSTER: List[SyntheticAgent] = [
    SyntheticAgent("bayesian_updater",   "Bayesian updater",      beta=0.98, sigma=0.55, bias=0.00),
    SyntheticAgent("base_rate_anchor",   "Base-rate anchorer",    beta=0.62, sigma=0.35, bias=0.00),
    SyntheticAgent("trend_model_reader", "Trend / model reader",  beta=0.90, sigma=0.75, bias=0.05),
    SyntheticAgent("news_signal_hunter", "News / signal hunter",  beta=1.05, sigma=1.05, bias=0.10),
    SyntheticAgent("contrarian",         "Contrarian / red-team", beta=0.55, sigma=0.90, bias=-0.15),
    SyntheticAgent("overconfident",      "Overconfident pundit",  beta=1.35, sigma=0.85, bias=0.15),
    SyntheticAgent("underconfident",     "Underconfident hedger", beta=0.70, sigma=0.60, bias=0.00),
    SyntheticAgent("noisy_novice",       "Noisy novice",          beta=0.80, sigma=1.60, bias=0.20),
]

COMMON_SHOCK_SD = 0.35   # sd of the shared log-odds shock (shared information)
BURN_IN = 150            # questions used to learn weights + extremize factor
EXTREMIZE_FIXED = 1.30   # a-priori factor evaluated as its own strategy


def generate_questions(rng: np.random.Generator, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw true probabilities from a realistic mixture:
      60% moderate (Beta(2.2, 2.2) — mass in the middle),
      25% skewed   (Beta(1.2, 3.5) or its mirror — likely / unlikely events),
      15% sharp    (near 0.05 or 0.95 — strong-signal questions).
    Returns (p_true, logit_true, outcomes).
    """
    u = rng.random(n)
    p = np.empty(n)
    moderate = u < 0.60
    skewed = (u >= 0.60) & (u < 0.85)
    sharp = u >= 0.85
    p[moderate] = rng.beta(2.2, 2.2, size=int(moderate.sum()))
    if skewed.sum():
        flip = rng.random(int(skewed.sum())) < 0.5
        draws = rng.beta(1.2, 3.5, size=int(skewed.sum()))
        p[skewed] = np.where(flip, draws, 1.0 - draws)
    if sharp.sum():
        p[sharp] = np.where(rng.random(int(sharp.sum())) < 0.5,
                             rng.uniform(0.02, 0.12, size=int(sharp.sum())),
                             rng.uniform(0.88, 0.98, size=int(sharp.sum())))
    p = clip_p(p)
    outcomes = (rng.random(n) < p).astype(int)
    return p, np.log(p / (1.0 - p)), outcomes


def simulate_forecasts(
    rng: np.random.Generator,
    agents: List[SyntheticAgent],
    logit_true: np.ndarray,
) -> np.ndarray:
    """Return matrix of shape (n_questions, n_agents) of forecast probabilities."""
    n = len(logit_true)
    k = len(agents)
    common = rng.normal(0.0, COMMON_SHOCK_SD, size=n)
    mat = np.empty((n, k))
    for j, a in enumerate(agents):
        eps = rng.normal(0.0, a.sigma, size=n)
        lo = a.bias + a.beta * logit_true + common + eps
        mat[:, j] = sigmoid(lo)
    return clip_p(mat)


def run_tournament(n_questions: int = 1000, seed: int = 42, verbose: bool = True) -> Dict:
    if n_questions < 300:
        raise ValueError("tournament needs >=300 questions for a meaningful burn-in split")
    rng = np.random.default_rng(seed)
    agents = AGENT_ROSTER
    k = len(agents)

    p_true, logit_true, outcomes = generate_questions(rng, n_questions)
    forecasts = simulate_forecasts(rng, agents, logit_true)

    # --- Online track-record weighting (no leakage: weights for question t
    #     use only questions < t; extremize factor is fixed a priori for the
    #     'extremized' strategies and *learned on burn-in only* for the
    #     tuned variant reported in the diagnostics). ---
    records = [ForecasterRecord(a.agent_id) for a in agents]
    n_test = n_questions - BURN_IN

    strat_names = ["mean", "median", "log_odds", "extremized",
                   "weighted_log_odds", "weighted_extremized"]
    strat_probs: Dict[str, np.ndarray] = {s: np.empty(n_test) for s in strat_names}
    weights_history = np.empty((n_test, k))

    for t in range(n_questions):
        row = forecasts[t]
        if t >= BURN_IN:
            idx = t - BURN_IN
            w = track_record_weights(records)
            weights_history[idx] = w
            strat_probs["mean"][idx] = mean_pool(row)
            strat_probs["median"][idx] = median_pool(row)
            strat_probs["log_odds"][idx] = log_odds_pool(row)
            strat_probs["extremized"][idx] = extremize(log_odds_pool(row), EXTREMIZE_FIXED)
            strat_probs["weighted_log_odds"][idx] = log_odds_pool(row, w)
            strat_probs["weighted_extremized"][idx] = extremize(log_odds_pool(row, w), EXTREMIZE_FIXED)
        # Update records AFTER forecasting question t (strictly causal).
        for j in range(k):
            records[j].add(float(row[j]), int(outcomes[t]))

    test_outcomes = outcomes[BURN_IN:]
    test_p_true = p_true[BURN_IN:]

    results: Dict[str, Dict[str, float]] = {}
    for s in strat_names:
        results[s] = {
            "brier": brier_score(strat_probs[s], test_outcomes),
            "log_score": log_score(strat_probs[s], test_outcomes),
            "ece": calibration_stats(strat_probs[s], test_outcomes)["ece"],
        }

    # Best single agent (identified on burn-in only, to avoid selection leakage)
    burn_briers = [brier_score(forecasts[:BURN_IN, j], outcomes[:BURN_IN]) for j in range(k)]
    best_idx_burnin = int(np.argmin(burn_briers))
    # Oracle best single agent on the test set (upper bound for any single agent)
    test_briers_all = [brier_score(forecasts[BURN_IN:, j], test_outcomes) for j in range(k)]
    oracle_idx = int(np.argmin(test_briers_all))

    per_agent = {
        agents[j].agent_id: {
            "persona": agents[j].persona,
            "beta": agents[j].beta, "sigma": agents[j].sigma, "bias": agents[j].bias,
            "brier_test": float(test_briers_all[j]),
            "log_score_test": log_score(forecasts[BURN_IN:, j], test_outcomes),
            "final_weight": float(weights_history[-1, j]),
            "mean_weight_test": float(weights_history[:, j].mean()),
        }
        for j in range(k)
    }

    # Oracle forecaster (knows p_true) and a coin-flip baseline for scale.
    oracle_brier = brier_score(test_p_true, test_outcomes)
    baseline_brier = brier_score(np.full(n_test, float(test_outcomes.mean())), test_outcomes)

    # Tuned extremizing factor on burn-in pooled forecasts (diagnostic):
    burn_pooled = np.array([log_odds_pool(forecasts[t]) for t in range(BURN_IN)])
    tuned_factor, tuned_burn_brier = select_extremize_factor(burn_pooled, outcomes[:BURN_IN])
    tuned_test = np.array([extremize(float(p), tuned_factor) for p in
                           np.array([log_odds_pool(forecasts[t]) for t in range(BURN_IN, n_questions)])])
    tuned_test_brier = brier_score(tuned_test, test_outcomes)
    tuned_test_log = log_score(tuned_test, test_outcomes)

    summary = {
        "n_questions": n_questions, "burn_in": BURN_IN, "n_test": n_test, "seed": seed,
        "n_agents": k,
        "strategies": results,
        "per_agent": per_agent,
        "best_single_burnin_id": agents[best_idx_burnin].agent_id,
        "best_single_burnin_brier_test": float(test_briers_all[best_idx_burnin]),
        "best_single_burnin_log_test": log_score(forecasts[BURN_IN:, best_idx_burnin], test_outcomes),
        "oracle_best_single_id": agents[oracle_idx].agent_id,
        "oracle_best_single_brier_test": float(test_briers_all[oracle_idx]),
        "oracle_prob_brier_test": float(oracle_brier),
        "oracle_prob_log_test": log_score(test_p_true, test_outcomes),
        "climatology_brier_test": float(baseline_brier),
        "tuned_extremize_factor_on_burnin": tuned_factor,
        "tuned_extremized_brier_test": float(tuned_test_brier),
        "tuned_extremized_log_test": float(tuned_test_log),
        "fixed_extremize_factor": EXTREMIZE_FIXED,
        "final_weights": {agents[j].agent_id: float(weights_history[-1, j]) for j in range(k)},
    }

    if verbose:
        print(f"=== Simulated tournament: {n_questions} questions "
              f"({BURN_IN} burn-in / {n_test} test), {k} agents, seed={seed} ===")
        print(f"Climatology (always base rate)  Brier {baseline_brier:.4f}")
        print(f"Oracle (knows p_true)           Brier {oracle_brier:.4f}  "
              f"Log {summary['oracle_prob_log_test']:.4f}")
        print(f"Best single (burn-in pick: {agents[best_idx_burnin].agent_id}) "
              f"Brier {test_briers_all[best_idx_burnin]:.4f}  "
              f"Log {summary['best_single_burnin_log_test']:.4f}")
        print(f"Oracle best single ({agents[oracle_idx].agent_id}) "
              f"Brier {test_briers_all[oracle_idx]:.4f}")
        print("\nPer-agent (test set):")
        for j in range(k):
            a = per_agent[agents[j].agent_id]
            print(f"  {agents[j].agent_id:20s} Brier {a['brier_test']:.4f}  "
                  f"Log {a['log_score_test']:.4f}  final w {a['final_weight']:.3f}")
        print("\nAggregation strategies (test set, lower = better):")
        for s in strat_names:
            r = results[s]
            print(f"  {s:20s} Brier {r['brier']:.4f}  Log {r['log_score']:.4f}  ECE {r['ece']:.4f}")
        print(f"  {'tuned_extremized':20s} Brier {tuned_test_brier:.4f}  "
              f"Log {tuned_test_log:.4f}  (factor {tuned_factor:.2f} from burn-in)")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Superforecaster agent-ensemble simulation")
    ap.add_argument("--n-questions", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run_tournament(n_questions=args.n_questions, seed=args.seed)
