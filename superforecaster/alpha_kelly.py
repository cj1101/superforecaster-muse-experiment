# alpha_kelly.py — Alpha-Kelly N_star scoring & order-book harness.
# Contributed via CJ (source: "Gemini 4 Argon" brief, 2026-10-17), adopted as the
# fleet's canonical N_star engine after verification: it runs clean and its
# built-in 8-market fixture benchmark reproduces exactly (disciplined fleet
# N_star +0.1059 vs naive -0.1499). IMPORTANT: that fixture suite is synthetic
# and hand-priced — it is a self-test of the machinery, NOT performance
# evidence. Production N_star is computed only on real data (see
# experiments/backtest_nstar.py and the live paper portfolio).
# Reconciliations with this repo's policy: engine output cap stays [0.01, 0.99]
# (docs/EXTERNAL_ADVICE_REVIEW.md); this harness clips trade-evaluation
# posteriors to [0.02, 0.98] per the N_star spec. Paper simulation only.

#!/usr/bin/env python3
"""
Alpha-Kelly North Star Benchmark & CLOB Execution Harness (alpha_kelly_harness.py)
==================================================================================

An executable evaluation, calibration, order-book simulation, and portfolio-sizing
engine for a multi-agent superforecasting & trading fleet operating across:
  1. Prediction Markets (Polymarket, Kalshi)
  2. Equities & Macro Catalysts
  3. Cryptocurrency Markets (Spot, Perps, DeFi / Token Unlocks)

Grounded in empirical findings and mathematical formulations from:
  - Prediction Arena (Zhang et al., 2026): https://arxiv.org/html/2604.07355v1
    (Bid-price liquidation mark-to-market, 15% per-cluster concentration cap, hold-to-settlement discipline)
  - PolyBench (Cheng et al., 2026): https://arxiv.org/html/2604.14199v1
    (Central Limit Order Book [CLOB] ask-sweep simulation, Confidence-Weighted Return [CWR], SKIP gate, APY)
  - Prophet Arena (Yang et al., 2025/2026): https://arxiv.org/html/2510.17638v1
    (3D evaluation: Brier Loss, 10-bin Expected Calibration Error [ECE], and CRRA Utility Market Return)
  - LiveTradeBench (Yu et al., 2025): https://arxiv.org/html/2511.03628v1
    (Cross-market POMDP portfolio allocation, Sharpe ratio, Maximum Drawdown [MDD])
  - CryptoBench (Guo et al., 2026): https://arxiv.org/html/2512.00417v5
    (Mandatory on-chain & derivatives verification gate to counter LLM overconfidence in crypto)
  - Kelly Criterion in Prediction Markets (Meister, 2024): https://arxiv.org/html/2412.14144v1
    (Kullback-Leibler divergence D_KL(p || q_exec) as asymptotic log-wealth growth rate)
  - Good Judgment Project / Logit Aggregation (Satopaa et al., 2014):
    https://www.researchgate.net/publication/260030454_Combining_multiple_probability_predictions_using_a_simple_logit_model
  - AIA Forecaster (Alur et al., 2025): https://arxiv.org/abs/2511.07678
    (Supervisor reconciliation and post-hoc Platt scaling)
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ============================================================================
# 1. DOMAIN DEFINITIONS & DATA STRUCTURES
# ============================================================================

class MarketDomain(str, Enum):
    PREDICTION_MARKET = "prediction_market"  # Polymarket, Kalshi
    EQUITIES_MACRO = "equities_macro"        # Earnings, FOMC, CPI, Equities
    CRYPTO = "crypto"                        # Spot/Perps, Token Unlocks, DeFi


class TradeAction(str, Enum):
    BUY_YES = "BUY_YES"  # Long outcome / Long asset target
    BUY_NO = "BUY_NO"    # Short outcome / Inverse contract
    SKIP = "SKIP"        # Explicit pass when net edge or checks fail


@dataclass
class OrderBookLevel:
    """Single price-depth rung in a Central Limit Order Book (CLOB)."""
    price: float       # Contract price in (0, 1) or normalized payoff price
    size_shares: float # Available contract shares at this price level


@dataclass
class OrderBookSnapshot:
    """
    Point-in-time Central Limit Order Book (CLOB) state (PolyBench / Prediction Arena).
    All binary or synthetic target contracts are normalized to pay $1.00 at settlement.
    """
    market_id: str
    title: str
    domain: MarketDomain
    factor_cluster: str               # e.g., "US_CPI_OCT", "BTC_HALVING_FLOW", "NE_WEATHER"
    est_days_to_resolution: float     # Expected capital lockup duration D_lock (days)
    yes_asks: List[OrderBookLevel]    # Sorted ascending by price (best ask first)
    yes_bids: List[OrderBookLevel]    # Sorted descending by price (best bid first)
    fee_rate: float = 0.01            # Exchange + gas fee rate (e.g., 1.0%)
    on_chain_verified: bool = True    # Required for MarketDomain.CRYPTO (CryptoBench gate)

    @property
    def best_yes_ask(self) -> float:
        return self.yes_asks[0].price if self.yes_asks else 0.99

    @property
    def best_yes_bid(self) -> float:
        return self.yes_bids[0].price if self.yes_bids else 0.01

    @property
    def best_no_ask(self) -> float:
        # Buying NO is equivalent to selling YES at the best YES bid: cost = 1 - best_yes_bid
        return round(1.0 - self.best_yes_bid, 6)

    @property
    def best_no_bid(self) -> float:
        # Selling NO (liquidation value of NO) is 1 - best_yes_ask
        return round(1.0 - self.best_yes_ask, 6)

    @property
    def mid_price(self) -> float:
        return 0.5 * (self.best_yes_ask + self.best_yes_bid)


@dataclass
class SubAgentForecast:
    """Individual model forecast within the multi-agent ensemble."""
    agent_id: str
    model_family: str          # e.g., "gemini-3-pro", "gpt-5", "claude-4.5-sonnet", "grok-4"
    raw_probability: float     # Raw probability in (0, 1)
    historical_brier_weight: float = 1.0
    has_independent_search: bool = True


@dataclass
class TradeExecutionResult:
    """Record of an evaluated opportunity and its simulated CLOB execution."""
    market_id: str
    title: str
    domain: MarketDomain
    factor_cluster: str
    action: TradeAction
    calibrated_prob_yes: float
    market_mid_prob: float
    exec_price: float                # Volume-weighted execution price q_exec (including fees)
    liquidation_bid_price: float     # Immediate mark-to-market bid price on chosen side
    kl_divergence_nats: float        # Information edge D_KL(p || q_exec)
    expected_log_growth: float       # Expected log-wealth increment Delta G
    kelly_fraction_raw: float        # Uncapped full Kelly fraction
    allocated_fraction: float        # Final portfolio fraction after Kelly/cluster/slippage caps
    invested_capital: float          # Actual dollars I_k swept into CLOB
    shares_acquired: float           # Total contracts S_k acquired
    est_days_to_resolution: float
    skip_reason: Optional[str] = None
    # Settlement / Mark-to-Market fields
    is_settled: bool = True
    outcome_yes: Optional[int] = None  # 1 if YES resolved, 0 if NO resolved
    net_pnl_bid_marked: float = 0.0    # Realized PnL (if settled) or bid-marked unrealized PnL


# ============================================================================
# 2. PROBABILITY AGGREGATION, PLATT SCALING & CAPPING
# ============================================================================

def clip_probability(p: float, min_p: float = 0.02, max_p: float = 0.98) -> float:
    """
    Hard probability cap (Metaculus Fall 2025 Survey r = +0.48, p = 0.005).
    Prevents a single hallucination, unit error, or resolution misread from wiping out log-wealth.
    """
    return max(min_p, min(max_p, p))


def logit(p: float) -> float:
    p_safe = max(1e-6, min(1.0 - 1e-6, p))
    return math.log(p_safe / (1.0 - p_safe))


def inv_logit(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def platt_recalibrate(
    p_raw: float,
    alpha: float = 1.12,
    beta: float = 0.0,
    min_p: float = 0.02,
    max_p: float = 0.98,
) -> float:
    """
    Post-hoc logistic recalibration (Platt scaling):
        logit(p_cal) = alpha * logit(p_raw) + beta
    Validated in Metaculus Q1+Q2 calibration studies (p < 0.001) and Bridgewater AIA Forecaster.
    """
    recalibrated = inv_logit(alpha * logit(p_raw) + beta)
    return clip_probability(recalibrated, min_p=min_p, max_p=max_p)


def aggregate_fleet_forecasts(
    forecasts: List[SubAgentForecast],
    extremizing_a: float = 1.25,
    platt_alpha: float = 1.08,
    platt_beta: float = 0.0,
    min_p: float = 0.02,
    max_p: float = 0.98,
) -> Tuple[float, float]:
    """
    Combines sub-agent forecasts based on information diversity (Satopaa et al., 2014; 2016):
      - If agents ran independent retrieval across distinct model families, applies weighted
        log-odds pooling with extremizing factor `a`.
      - If agents share the same context window, uses the trimmed mean / median (Halawi et al., 2024).
    Returns:
      (calibrated_probability, inter_agent_std_dev)
    """
    if not forecasts:
        raise ValueError("Forecast list cannot be empty.")

    raw_probs = [clip_probability(f.raw_probability, 0.005, 0.995) for f in forecasts]
    dispersion = statistics.pstdev(raw_probs) if len(raw_probs) > 1 else 0.0

    distinct_families = len({f.model_family for f in forecasts})
    independent_search = all(f.has_independent_search for f in forecasts)

    if distinct_families >= 3 and independent_search:
        # Weighted log-odds pooling with extremizing (Satopaa et al., 2014)
        total_w = sum(f.historical_brier_weight for f in forecasts)
        weighted_logit = sum(
            (f.historical_brier_weight / total_w) * logit(p)
            for f, p in zip(forecasts, raw_probs)
        )
        pooled_prob = inv_logit(extremizing_a * weighted_logit)
    else:
        # Median / trimmed mean for shared-context runs (Halawi et al., 2024; nostreambot)
        if len(raw_probs) >= 5:
            sorted_p = sorted(raw_probs)[1:-1]
            pooled_prob = statistics.mean(sorted_p)
        else:
            pooled_prob = statistics.median(raw_probs)

    calibrated = platt_recalibrate(
        pooled_prob, alpha=platt_alpha, beta=platt_beta, min_p=min_p, max_p=max_p
    )
    return calibrated, dispersion


# ============================================================================
# 3. CENTRAL LIMIT ORDER BOOK (CLOB) SWEEP & SLIPPAGE SIMULATOR
# ============================================================================

def sweep_order_book(
    book: OrderBookSnapshot,
    side: TradeAction,
    target_budget: float,
    belief_prob_side: float,
    min_net_edge: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Simulates sweeping the Central Limit Order Book (PolyBench Section 4.2) while
    stopping before any order-book rung where marginal execution price + fees
    erodes the edge below `min_net_edge`.

    Returns:
      (invested_capital_including_fees, shares_acquired, effective_execution_price_per_share)
    """
    if target_budget <= 0 or side == TradeAction.SKIP:
        return 0.0, 0.0, 0.0

    # Construct effective ask ladder for the chosen side
    if side == TradeAction.BUY_YES:
        ladder = book.yes_asks
    elif side == TradeAction.BUY_NO:
        # Buying NO sweeps the YES bids in reverse (price of NO = 1.0 - yes_bid.price)
        ladder = [
            OrderBookLevel(price=round(1.0 - lvl.price, 6), size_shares=lvl.size_shares)
            for lvl in book.yes_bids
        ]
    else:
        return 0.0, 0.0, 0.0

    remaining_budget = target_budget
    total_cost_with_fees = 0.0
    total_shares = 0.0

    for rung in ladder:
        if remaining_budget <= 1e-6:
            break
        effective_rung_price = rung.price * (1.0 + book.fee_rate)
        # Stop sweeping if this rung's marginal price destroys our minimum edge hurdle
        if (belief_prob_side - effective_rung_price) < min_net_edge:
            break

        max_rung_cost = rung.size_shares * effective_rung_price
        spend_on_rung = min(remaining_budget, max_rung_cost)
        shares_on_rung = spend_on_rung / effective_rung_price

        total_cost_with_fees += spend_on_rung
        total_shares += shares_on_rung
        remaining_budget -= spend_on_rung

    if total_shares <= 0:
        return 0.0, 0.0, 0.0

    q_exec = total_cost_with_fees / total_shares
    return total_cost_with_fees, total_shares, q_exec


# ============================================================================
# 4. KELLY / CRRA SIZING & INFORMATION-THEORETIC EDGE ENGINE
# ============================================================================

def kl_divergence_binary(p: float, q: float) -> float:
    """
    Kullback-Leibler divergence D_KL(p || q) in nats (Meister, 2024).
    Measures the theoretical asymptotic compound growth rate of wealth.
    """
    p_c = max(1e-6, min(1.0 - 1e-6, p))
    q_c = max(1e-6, min(1.0 - 1e-6, q))
    return p_c * math.log(p_c / q_c) + (1.0 - p_c) * math.log((1.0 - p_c) / (1.0 - q_c))


def expected_log_wealth_growth(p_side: float, q_exec: float, fraction: float) -> float:
    """
    Expected single-trade log-wealth increment Delta G (Meister, 2024 Eq. III):
      U(p_side, q_exec, f) = p_side * log(1 + f * (1 - q_exec)/q_exec) + (1 - p_side) * log(1 - f)
    """
    if fraction <= 0.0 or q_exec <= 0.0 or q_exec >= 1.0:
        return 0.0
    win_mult = (1.0 - q_exec) / q_exec
    return p_side * math.log(1.0 + fraction * win_mult) + (1.0 - p_side) * math.log(max(1e-9, 1.0 - fraction))


@dataclass
class FleetRiskConfig:
    """Risk and execution governance parameters for the agent fleet."""
    min_net_edge: float = 0.05               # Minimum |p - q_exec| hurdle rate after fees/slippage
    min_confidence: float = 0.60             # PolyBench minimum directional conviction c >= 0.60
    kelly_multiplier: float = 0.35           # Fractional Kelly multiplier (0.25 to 0.50)
    crypto_kelly_dampener: float = 0.50      # Extra 0.5x dampener for Crypto (PolyBench Section 5.2)
    max_cluster_allocation: float = 0.15     # Prediction Arena 15% cap per correlated event cluster
    max_inter_agent_dispersion: float = 0.22 # Trigger supervisor research if sub-agents diverge > 0.22
    require_on_chain_for_crypto: bool = True # CryptoBench gate


def evaluate_and_size_opportunity(
    book: OrderBookSnapshot,
    calibrated_prob_yes: float,
    inter_agent_dispersion: float,
    portfolio_equity: float,
    cluster_exposure_used: Dict[str, float],
    config: FleetRiskConfig = FleetRiskConfig(),
) -> TradeExecutionResult:
    """
    Executes the complete decision gate for a single market opportunity:
      1. Checks CryptoBench on-chain verification gate and inter-agent disagreement gate.
      2. Determines optimal trade direction (BUY_YES, BUY_NO, or SKIP).
      3. Computes Fractional Kelly allocation & enforces the 15% correlated cluster cap.
      4. Sweeps the live CLOB ask ladder, dynamically stopping where slippage erodes net edge.
    """
    # Gate 1: CryptoBench on-chain & derivatives data verification gate
    if (
        book.domain == MarketDomain.CRYPTO
        and config.require_on_chain_for_crypto
        and not book.on_chain_verified
    ):
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=book.mid_price,
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason="FAILED_CRYPTOBENCH_ONCHAIN_GATE",
        )

    # Gate 2: Supervisor reconciliation gate on high inter-agent disagreement
    if inter_agent_dispersion > config.max_inter_agent_dispersion:
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=book.mid_price,
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=f"UNRESOLVED_AGENT_DISPERSION ({inter_agent_dispersion:.2f} > {config.max_inter_agent_dispersion:.2f})",
        )

    # Determine candidate side & top-of-book fee-adjusted ask price
    yes_top_ask_eff = book.best_yes_ask * (1.0 + book.fee_rate)
    no_top_ask_eff = book.best_no_ask * (1.0 + book.fee_rate)

    edge_yes = calibrated_prob_yes - yes_top_ask_eff
    edge_no = (1.0 - calibrated_prob_yes) - no_top_ask_eff

    if edge_yes >= edge_no and edge_yes >= config.min_net_edge:
        chosen_side = TradeAction.BUY_YES
        p_side = calibrated_prob_yes
        q_top = yes_top_ask_eff
        liq_bid = book.best_yes_bid
    elif edge_no > edge_yes and edge_no >= config.min_net_edge:
        chosen_side = TradeAction.BUY_NO
        p_side = 1.0 - calibrated_prob_yes
        q_top = no_top_ask_eff
        liq_bid = book.best_no_bid
    else:
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=book.mid_price,
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=f"INSUFFICIENT_NET_EDGE (max_edge={max(edge_yes, edge_no):.3f} < {config.min_net_edge:.3f})",
        )

    # Gate 3: PolyBench minimum directional conviction c >= 0.60 (or positive Kelly edge on asymmetric odds)
    if p_side < config.min_confidence and (p_side / q_top) < 1.35:
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=book.mid_price,
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=f"LOW_CONVICTION (p_side={p_side:.2f} < {config.min_confidence:.2f})",
        )

    # Step 4: Compute full Kelly fraction f* = (p_side - q_top) / (1 - q_top)
    kelly_raw = max(0.0, (p_side - q_top) / max(1e-6, 1.0 - q_top))
    domain_mult = (
        config.kelly_multiplier * config.crypto_kelly_dampener
        if book.domain == MarketDomain.CRYPTO
        else config.kelly_multiplier
    )
    target_fraction = kelly_raw * domain_mult

    # Enforce 15% correlated factor-cluster concentration cap (Prediction Arena Section 5.4.1)
    used_in_cluster = cluster_exposure_used.get(book.factor_cluster, 0.0)
    remaining_cluster_room = max(0.0, config.max_cluster_allocation - used_in_cluster)
    capped_fraction = min(target_fraction, remaining_cluster_room)

    if capped_fraction <= 0.001:
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=book.mid_price,
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=kelly_raw,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=f"CLUSTER_CAP_REACHED ({book.factor_cluster})",
        )

    # Step 5: Sweep CLOB ask book up to target dollar budget while preserving min_net_edge
    target_budget = portfolio_equity * capped_fraction
    invested_capital, shares_acquired, q_exec = sweep_order_book(
        book=book,
        side=chosen_side,
        target_budget=target_budget,
        belief_prob_side=p_side,
        min_net_edge=config.min_net_edge,
    )

    if invested_capital <= 0.0 or shares_acquired <= 0.0:
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=book.mid_price,
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=kelly_raw,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason="ORDER_BOOK_DEPTH_EXHAUSTED_AT_REQUIRED_EDGE",
        )

    actual_fraction = invested_capital / portfolio_equity
    cluster_exposure_used[book.factor_cluster] = used_in_cluster + actual_fraction

    kl_nats = kl_divergence_binary(p_side, q_exec)
    exp_log_g = expected_log_wealth_growth(p_side, q_exec, actual_fraction)

    return TradeExecutionResult(
        market_id=book.market_id,
        title=book.title,
        domain=book.domain,
        factor_cluster=book.factor_cluster,
        action=chosen_side,
        calibrated_prob_yes=calibrated_prob_yes,
        market_mid_prob=book.mid_price,
        exec_price=q_exec,
        liquidation_bid_price=liq_bid,
        kl_divergence_nats=kl_nats,
        expected_log_growth=exp_log_g,
        kelly_fraction_raw=kelly_raw,
        allocated_fraction=actual_fraction,
        invested_capital=invested_capital,
        shares_acquired=shares_acquired,
        est_days_to_resolution=book.est_days_to_resolution,
        skip_reason=None,
    )


# ============================================================================
# 5. SETTLEMENT, BID-MARKED VALUATION & NORTH STAR SCORING ENGINE (N_star)
# ============================================================================

def settle_or_mark_trade(
    trade: TradeExecutionResult,
    is_settled: bool,
    outcome_yes: Optional[int] = None,
) -> TradeExecutionResult:
    """
    Computes net PnL for a trade:
      - If settled (outcome_yes in {0, 1}): winning shares pay $1.00, losing shares pay $0.00.
      - If still open (is_settled=False): values shares strictly at current liquidation BID price
        (Prediction Arena Section 5.3.1), immediately accounting for bid-ask spread cost.
    """
    trade.is_settled = is_settled
    trade.outcome_yes = outcome_yes

    if trade.action == TradeAction.SKIP or trade.invested_capital <= 0.0:
        trade.net_pnl_bid_marked = 0.0
        return trade

    if is_settled:
        if outcome_yes not in (0, 1):
            raise ValueError("Settled trade requires outcome_yes in {0, 1}.")
        won = (
            (trade.action == TradeAction.BUY_YES and outcome_yes == 1)
            or (trade.action == TradeAction.BUY_NO and outcome_yes == 0)
        )
        gross_payout = trade.shares_acquired * 1.0 if won else 0.0
        trade.net_pnl_bid_marked = gross_payout - trade.invested_capital
    else:
        # Conservative mark-to-market using immediate liquidation bid price
        liquidation_value = trade.shares_acquired * trade.liquidation_bid_price
        trade.net_pnl_bid_marked = liquidation_value - trade.invested_capital

    return trade


def compute_empirical_ece_and_brier(
    predictions_and_outcomes: List[Tuple[float, int]],
    num_bins: int = 10,
) -> Tuple[float, float]:
    """
    Computes 10-bin Expected Calibration Error (ECE) and Brier Score
    (Prophet Arena Section 3.1.1 & Appendix B.3).
    """
    if not predictions_and_outcomes:
        return 0.0, 0.0

    m = len(predictions_and_outcomes)
    brier = sum((p - o) ** 2 for p, o in predictions_and_outcomes) / m

    bins: List[List[Tuple[float, int]]] = [[] for _ in range(num_bins)]
    for p, o in predictions_and_outcomes:
        idx = min(num_bins - 1, int(p * num_bins))
        bins[idx].append((p, o))

    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_p = sum(x[0] for x in b) / len(b)
        avg_o = sum(x[1] for x in b) / len(b)
        ece += (len(b) / m) * abs(avg_o - avg_p)

    return ece, brier


@dataclass
class NorthStarScorecard:
    """Complete fleet evaluation metrics centered on N_star."""
    fleet_name: str
    starting_capital: float
    ending_account_value_bid_marked: float
    total_net_pnl: float
    portfolio_return_pct: float
    cwr_bid_marked: float            # Sum(PnL_bid) / Sum(Invested_Capital)
    avg_lockup_days: float           # Capital-weighted average resolution horizon D_lock
    velocity_multiplier: float       # sqrt(365 / D_lock)
    annualized_yield_apy_pct: float
    ece_10bin: float                 # Expected Calibration Error
    calibration_multiplier: float    # exp(-10 * ECE)
    brier_score: float               # Diagnostic absolute Brier score
    max_drawdown_pct: float          # Peak-to-trough portfolio drawdown
    drawdown_penalty: float          # 2.0 * max(0, MDD - 0.15)
    sharpe_ratio_per_trade: float
    total_opportunities: int
    executed_trades: int
    skipped_opportunities: int
    settlement_win_rate_pct: float
    north_star_score: float          # Composite N_star metric to maximize


def evaluate_fleet_north_star(
    fleet_name: str,
    trades: List[TradeExecutionResult],
    all_resolved_forecasts: List[Tuple[float, int]],
    starting_capital: float = 10_000.0,
) -> NorthStarScorecard:
    """
    Computes the composite Alpha-Kelly North Star Score (N_star):
      N_star = CWR_bid * sqrt(365 / D_lock) * exp(-10 * ECE) - 2.0 * max(0, MDD - 0.15)
    """
    executed = [t for t in trades if t.action != TradeAction.SKIP and t.invested_capital > 0]
    skipped = [t for t in trades if t.action == TradeAction.SKIP]

    total_invested = sum(t.invested_capital for t in executed)
    total_pnl = sum(t.net_pnl_bid_marked for t in executed)
    cwr_bid = (total_pnl / total_invested) if total_invested > 0 else 0.0

    # Sequential equity curve and Max Drawdown (MDD)
    equity = starting_capital
    peak_equity = starting_capital
    max_dd = 0.0
    trade_returns: List[float] = []

    for t in executed:
        equity += t.net_pnl_bid_marked
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        max_dd = max(max_dd, dd)
        trade_returns.append(t.net_pnl_bid_marked / t.invested_capital)

    # Capital-weighted lockup duration D_lock and Velocity Multiplier
    if total_invested > 0:
        avg_lockup = sum(t.invested_capital * t.est_days_to_resolution for t in executed) / total_invested
    else:
        avg_lockup = 30.0
    avg_lockup = max(1.0, avg_lockup)
    velocity_mult = math.sqrt(365.0 / avg_lockup)
    apy_pct = cwr_bid * (365.0 / avg_lockup) * 100.0

    # Calibration Error (ECE) and Brier Score
    ece, brier = compute_empirical_ece_and_brier(all_resolved_forecasts, num_bins=10)
    cal_mult = math.exp(-10.0 * ece)

    # Drawdown Penalty beyond 15% threshold
    dd_penalty = 2.0 * max(0.0, max_dd - 0.15)

    # Composite North Star Score (N_star)
    north_star = (cwr_bid * velocity_mult * cal_mult) - dd_penalty

    # Settlement Win Rate & Per-Trade Sharpe Ratio
    settled_trades = [t for t in executed if t.is_settled]
    wins = sum(1 for t in settled_trades if t.net_pnl_bid_marked > 0)
    win_rate_pct = (wins / len(settled_trades) * 100.0) if settled_trades else 0.0

    if len(trade_returns) >= 2:
        mean_r = statistics.mean(trade_returns)
        std_r = statistics.stdev(trade_returns)
        sharpe = (mean_r / std_r) if std_r > 1e-9 else 0.0
    else:
        sharpe = 0.0

    return NorthStarScorecard(
        fleet_name=fleet_name,
        starting_capital=starting_capital,
        ending_account_value_bid_marked=round(equity, 2),
        total_net_pnl=round(total_pnl, 2),
        portfolio_return_pct=round((total_pnl / starting_capital) * 100.0, 2),
        cwr_bid_marked=round(cwr_bid, 4),
        avg_lockup_days=round(avg_lockup, 2),
        velocity_multiplier=round(velocity_mult, 3),
        annualized_yield_apy_pct=round(apy_pct, 2),
        ece_10bin=round(ece, 4),
        calibration_multiplier=round(cal_mult, 4),
        brier_score=round(brier, 4),
        max_drawdown_pct=round(max_dd * 100.0, 2),
        drawdown_penalty=round(dd_penalty, 4),
        sharpe_ratio_per_trade=round(sharpe, 3),
        total_opportunities=len(trades),
        executed_trades=len(executed),
        skipped_opportunities=len(skipped),
        settlement_win_rate_pct=round(win_rate_pct, 2),
        north_star_score=round(north_star, 4),
    )


# ============================================================================
# 6. LIVE LEARNING FEEDBACK GENERATOR (PREDICTION ARENA SECTION 5.2.2)
# ============================================================================

def build_cycle_learning_prompt_block(
    scorecard: NorthStarScorecard,
    recent_trades: List[TradeExecutionResult],
) -> str:
    """
    Generates the dynamic 'Learning Section' injected into the fleet's context window
    at the start of each 15-45 minute trading cycle (Prediction Arena Section 5.2.2).
    """
    executed = [t for t in recent_trades if t.action != TradeAction.SKIP]
    biggest_win = max(executed, key=lambda t: t.net_pnl_bid_marked, default=None)
    biggest_loss = min(executed, key=lambda t: t.net_pnl_bid_marked, default=None)

    lines = [
        "=== FLEET NORTH STAR TELEMETRY & ROLLING CYCLE FEEDBACK ===",
        f"Current North Star Score (N_star): {scorecard.north_star_score:+.4f}",
        f"Account Value (Bid-Marked): ${scorecard.ending_account_value_bid_marked:,.2f} "
        f"({scorecard.portfolio_return_pct:+.2f}% | Max Drawdown: {scorecard.max_drawdown_pct:.2f}%)",
        f"Confidence-Weighted Return (CWR_bid): {scorecard.cwr_bid_marked * 100:+.2f}% | "
        f"Avg Capital Lockup: {scorecard.avg_lockup_days:.1f}d (Velocity Mult: {scorecard.velocity_multiplier:.2f}x)",
        f"Calibration Error (ECE): {scorecard.ece_10bin:.4f} (Trust Mult: {scorecard.calibration_multiplier:.2f}x) | "
        f"Brier Score: {scorecard.brier_score:.4f}",
        f"Selectivity: {scorecard.executed_trades} executed / {scorecard.total_opportunities} scanned "
        f"({scorecard.skipped_opportunities} skipped)",
    ]
    if biggest_win and biggest_win.net_pnl_bid_marked > 0:
        lines.append(
            f"Best Recent Trade: {biggest_win.market_id} ({biggest_win.action.value}) -> "
            f"+${biggest_win.net_pnl_bid_marked:,.2f} (q_exec={biggest_win.exec_price:.2f}, "
            f"p_yes={biggest_win.calibrated_prob_yes:.2f})"
        )
    if biggest_loss and biggest_loss.net_pnl_bid_marked < 0:
        lines.append(
            f"Worst Recent Trade: {biggest_loss.market_id} ({biggest_loss.action.value}) -> "
            f"-${abs(biggest_loss.net_pnl_bid_marked):,.2f} (q_exec={biggest_loss.exec_price:.2f}, "
            f"p_yes={biggest_loss.calibrated_prob_yes:.2f})"
        )
    return "\n".join(lines)


# ============================================================================
# 7. DEMONSTRATION & VALIDATION SIMULATION (POLYMARKET, KALSHI, EQUITIES, CRYPTO)
# ============================================================================

def run_benchmark_simulation() -> None:
    """
    Simulates a realistic 8-event multi-market slate across Prediction Markets,
    Equities/Macro, and Crypto, comparing:
      1. Alpha-Kelly Disciplined Fleet (optimizing N_star with CLOB sweep & gates)
      2. Naive Brier-Optimized Fleet (trading every market at flat 12% size without CLOB/Crypto gates)
    """
    sample_scenarios = [
        # 1. Polymarket Macro/Tech catalyst with deep liquidity and strong positive edge
        {
            "book": OrderBookSnapshot(
                market_id="POLY-FED-CUT-25BPS",
                title="Fed cuts rates by 25bps at upcoming FOMC meeting",
                domain=MarketDomain.PREDICTION_MARKET,
                factor_cluster="US_RATES_FOMC",
                est_days_to_resolution=6.0,
                yes_asks=[
                    OrderBookLevel(price=0.62, size_shares=1200),
                    OrderBookLevel(price=0.65, size_shares=2000),
                    OrderBookLevel(price=0.71, size_shares=3000),
                ],
                yes_bids=[OrderBookLevel(price=0.60, size_shares=1500)],
                fee_rate=0.005,
            ),
            "sub_agents": [
                SubAgentForecast("agent-1", "gemini-3-pro", 0.78, 1.2),
                SubAgentForecast("agent-2", "gpt-5", 0.81, 1.1),
                SubAgentForecast("agent-3", "claude-4.5-sonnet", 0.76, 1.0),
                SubAgentForecast("agent-4", "grok-4", 0.79, 1.1),
            ],
            "outcome_yes": 1,
            "is_settled": True,
        },
        # 2. Kalshi Curated Market where market is already efficient (tiny 2% edge -> SKIP)
        {
            "book": OrderBookSnapshot(
                market_id="KALSHI-CPI-CORE-0.3",
                title="US Core CPI MoM > 0.3% this month",
                domain=MarketDomain.EQUITIES_MACRO,
                factor_cluster="US_RATES_FOMC",
                est_days_to_resolution=10.0,
                yes_asks=[
                    OrderBookLevel(price=0.53, size_shares=800),
                    OrderBookLevel(price=0.56, size_shares=1500),
                ],
                yes_bids=[OrderBookLevel(price=0.50, size_shares=900)],
                fee_rate=0.015,
            ),
            "sub_agents": [
                SubAgentForecast("agent-1", "gemini-3-pro", 0.55, 1.2),
                SubAgentForecast("agent-2", "gpt-5", 0.54, 1.1),
                SubAgentForecast("agent-3", "claude-4.5-sonnet", 0.56, 1.0),
                SubAgentForecast("agent-4", "grok-4", 0.53, 1.1),
            ],
            "outcome_yes": 0,
            "is_settled": True,
        },
        # 3. Crypto market WITHOUT verified on-chain data (text-only hype -> CryptoBench gate blocks it)
        {
            "book": OrderBookSnapshot(
                market_id="CRYPTO-ALT-PUMP-48H",
                title="Token XYZ breaks $5.00 within 48 hours on social narrative",
                domain=MarketDomain.CRYPTO,
                factor_cluster="ALTCOIN_BETA",
                est_days_to_resolution=2.0,
                yes_asks=[
                    OrderBookLevel(price=0.48, size_shares=500),
                    OrderBookLevel(price=0.58, size_shares=1500),
                ],
                yes_bids=[OrderBookLevel(price=0.42, size_shares=500)],
                fee_rate=0.015,
                on_chain_verified=False,  # Fails CryptoBench gate!
            ),
            "sub_agents": [
                SubAgentForecast("agent-1", "gemini-3-pro", 0.72, 1.2),
                SubAgentForecast("agent-2", "gpt-5", 0.75, 1.1),
                SubAgentForecast("agent-3", "claude-4.5-sonnet", 0.70, 1.0),
                SubAgentForecast("agent-4", "grok-4", 0.74, 1.1),
            ],
            "outcome_yes": 0,
            "is_settled": True,
        },
        # 4. Crypto Token Unlock with VERIFIED on-chain whale distribution + perp funding skew (BUY_NO wins)
        {
            "book": OrderBookSnapshot(
                market_id="CRYPTO-UNLOCK-HOLD-SUPPORT",
                title="Token ABC holds above $2.00 through 12% cliff unlock week",
                domain=MarketDomain.CRYPTO,
                factor_cluster="TOKEN_UNLOCK_SHORT",
                est_days_to_resolution=5.0,
                yes_asks=[OrderBookLevel(price=0.68, size_shares=1200)],
                yes_bids=[
                    OrderBookLevel(price=0.65, size_shares=900),   # Buying NO costs 1 - 0.65 = $0.35
                    OrderBookLevel(price=0.60, size_shares=1500),  # Next rung costs $0.40
                ],
                fee_rate=0.01,
                on_chain_verified=True,
            ),
            "sub_agents": [
                SubAgentForecast("agent-1", "gemini-3-pro", 0.28, 1.2),
                SubAgentForecast("agent-2", "gpt-5", 0.25, 1.1),
                SubAgentForecast("agent-3", "claude-4.5-sonnet", 0.30, 1.0),
                SubAgentForecast("agent-4", "grok-4", 0.27, 1.1),
            ],
            "outcome_yes": 0,  # Token dumps below $2.00 -> BUY_NO pays $1.00!
            "is_settled": True,
        },
        # 5. Thin Order Book Trap (PolyBench Fig. 3): 1st rung has 50 shares at $0.40, then jumps to $0.75!
        {
            "book": OrderBookSnapshot(
                market_id="POLY-ILLIQUID-TRAP",
                title="Niche regulatory bill passes committee this week",
                domain=MarketDomain.PREDICTION_MARKET,
                factor_cluster="REGULATORY_NICHE",
                est_days_to_resolution=7.0,
                yes_asks=[
                    OrderBookLevel(price=0.40, size_shares=60),    # Only $24 of liquidity at $0.40!
                    OrderBookLevel(price=0.76, size_shares=3000),  # Huge slippage cliff!
                ],
                yes_bids=[OrderBookLevel(price=0.35, size_shares=500)],
                fee_rate=0.01,
            ),
            "sub_agents": [
                SubAgentForecast("agent-1", "gemini-3-pro", 0.66, 1.2),
                SubAgentForecast("agent-2", "gpt-5", 0.68, 1.1),
                SubAgentForecast("agent-3", "claude-4.5-sonnet", 0.65, 1.0),
                SubAgentForecast("agent-4", "grok-4", 0.67, 1.1),
            ],
            "outcome_yes": 1,
            "is_settled": True,
        },
        # 6. Equities Earnings Catalyst with strong fundamental + options-implied edge
        {
            "book": OrderBookSnapshot(
                market_id="EQ-SEMI-BEAT-GUIDE",
                title="Semiconductor leader beats Q3 datacenter revenue consensus by >4%",
                domain=MarketDomain.EQUITIES_MACRO,
                factor_cluster="AI_CAPEX_EARNINGS",
                est_days_to_resolution=4.0,
                yes_asks=[
                    OrderBookLevel(price=0.58, size_shares=1500),
                    OrderBookLevel(price=0.61, size_shares=2500),
                ],
                yes_bids=[OrderBookLevel(price=0.56, size_shares=1500)],
                fee_rate=0.005,
            ),
            "sub_agents": [
                SubAgentForecast("agent-1", "gemini-3-pro", 0.77, 1.2),
                SubAgentForecast("agent-2", "gpt-5", 0.79, 1.1),
                SubAgentForecast("agent-3", "claude-4.5-sonnet", 0.75, 1.0),
                SubAgentForecast("agent-4", "grok-4", 0.80, 1.1),
            ],
            "outcome_yes": 1,
            "is_settled": True,
        },
        # 7. High Inter-Agent Dispersion (agents disagree wildly -> Supervisor blocks premature trade)
        {
            "book": OrderBookSnapshot(
                market_id="POLY-GEOPOL-AMBIGUOUS",
                title="Bilateral trade communique signed before Friday midnight UTC",
                domain=MarketDomain.PREDICTION_MARKET,
                factor_cluster="GEOPOL_TRADE",
                est_days_to_resolution=5.0,
                yes_asks=[OrderBookLevel(price=0.45, size_shares=2000)],
                yes_bids=[OrderBookLevel(price=0.42, size_shares=2000)],
                fee_rate=0.01,
            ),
            "sub_agents": [
                SubAgentForecast("agent-1", "gemini-3-pro", 0.88, 1.2),
                SubAgentForecast("agent-2", "gpt-5", 0.22, 1.1),
                SubAgentForecast("agent-3", "claude-4.5-sonnet", 0.82, 1.0),
                SubAgentForecast("agent-4", "grok-4", 0.30, 1.1),
            ],
            "outcome_yes": 0,
            "is_settled": True,
        },
        # 8. Well-calibrated loss on a genuine 75% favorite (demonstrates controlled risk management)
        {
            "book": OrderBookSnapshot(
                market_id="EQ-BIOTECH-PDUFA",
                title="FDA approves Phase 3 rare-disease therapy on PDUFA date",
                domain=MarketDomain.EQUITIES_MACRO,
                factor_cluster="BIOTECH_FDA",
                est_days_to_resolution=8.0,
                yes_asks=[OrderBookLevel(price=0.55, size_shares=1500)],
                yes_bids=[OrderBookLevel(price=0.53, size_shares=1500)],
                fee_rate=0.005,
            ),
            "sub_agents": [
                SubAgentForecast("agent-1", "gemini-3-pro", 0.72, 1.2),
                SubAgentForecast("agent-2", "gpt-5", 0.74, 1.1),
                SubAgentForecast("agent-3", "claude-4.5-sonnet", 0.71, 1.0),
                SubAgentForecast("agent-4", "grok-4", 0.73, 1.1),
            ],
            "outcome_yes": 0,
            "is_settled": True,
        },
    ]

    starting_cap = 10_000.0
    disciplined_equity = starting_cap
    cluster_used: Dict[str, float] = {}
    disciplined_trades: List[TradeExecutionResult] = []
    naive_trades: List[TradeExecutionResult] = []
    resolved_forecasts: List[Tuple[float, int]] = []

    for item in sample_scenarios:
        book: OrderBookSnapshot = item["book"]
        sub_agents: List[SubAgentForecast] = item["sub_agents"]
        outcome_yes: int = item["outcome_yes"]
        is_settled: bool = item["is_settled"]

        cal_p, dispersion = aggregate_fleet_forecasts(sub_agents)
        resolved_forecasts.append((cal_p, outcome_yes))

        # 1. Alpha-Kelly Disciplined Execution
        d_trade = evaluate_and_size_opportunity(
            book=book,
            calibrated_prob_yes=cal_p,
            inter_agent_dispersion=dispersion,
            portfolio_equity=disciplined_equity,
            cluster_exposure_used=cluster_used,
        )
        d_trade = settle_or_mark_trade(d_trade, is_settled=is_settled, outcome_yes=outcome_yes)
        disciplined_equity += d_trade.net_pnl_bid_marked
        disciplined_trades.append(d_trade)

        # 2. Naive Brier-Optimized Baseline (bets flat $1,200 whenever p != mid_price, ignoring gates & slippage)
        naive_side = TradeAction.BUY_YES if cal_p >= book.mid_price else TradeAction.BUY_NO
        naive_p_side = cal_p if naive_side == TradeAction.BUY_YES else (1.0 - cal_p)
        # Naive bot sweeps book with min_net_edge = -1.0 (blind to order-book cliffs!)
        n_inv, n_shares, n_q_exec = sweep_order_book(
            book=book, side=naive_side, target_budget=1_200.0, belief_prob_side=naive_p_side, min_net_edge=-1.0
        )
        n_trade = TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=naive_side,
            calibrated_prob_yes=cal_p,
            market_mid_prob=book.mid_price,
            exec_price=n_q_exec,
            liquidation_bid_price=book.best_yes_bid if naive_side == TradeAction.BUY_YES else book.best_no_bid,
            kl_divergence_nats=kl_divergence_binary(naive_p_side, n_q_exec) if n_q_exec > 0 else 0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.12,
            allocated_fraction=n_inv / starting_cap,
            invested_capital=n_inv,
            shares_acquired=n_shares,
            est_days_to_resolution=book.est_days_to_resolution,
        )
        n_trade = settle_or_mark_trade(n_trade, is_settled=is_settled, outcome_yes=outcome_yes)
        naive_trades.append(n_trade)

    disciplined_scorecard = evaluate_fleet_north_star(
        fleet_name="Alpha-Kelly Disciplined Fleet",
        trades=disciplined_trades,
        all_resolved_forecasts=resolved_forecasts,
        starting_capital=starting_cap,
    )
    naive_scorecard = evaluate_fleet_north_star(
        fleet_name="Naive Brier-Only Baseline Fleet",
        trades=naive_trades,
        all_resolved_forecasts=resolved_forecasts,
        starting_capital=starting_cap,
    )

    print("=" * 86)
    print("ALPHA-KELLY NORTH STAR BENCHMARK HARNESS — SIMULATION VERIFICATION")
    print("=" * 86)
    for sc in (disciplined_scorecard, naive_scorecard):
        print(json.dumps(asdict(sc), indent=2))
        print("-" * 86)

    print("\nTRADE-BY-TRADE EXECUTION LOG (ALPHA-KELLY DISCIPLINED FLEET):")
    for t in disciplined_trades:
        if t.action == TradeAction.SKIP:
            print(f"  [SKIP]    {t.market_id:<28} | p_yes={t.calibrated_prob_yes:.2f} | Reason: {t.skip_reason}")
        else:
            print(
                f"  [{t.action.value:<7}] {t.market_id:<28} | p_yes={t.calibrated_prob_yes:.2f} | "
                f"q_exec={t.exec_price:.3f} | Invested=${t.invested_capital:>7.2f} | "
                f"PnL=${t.net_pnl_bid_marked:>+8.2f}"
            )

    print("\n" + build_cycle_learning_prompt_block(disciplined_scorecard, disciplined_trades))


if __name__ == "__main__":
    run_benchmark_simulation()