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
import numbers
import statistics
import warnings
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


# --- SF-BL-11492: OrderBookLevel field types are never validated -------
# OrderBookLevel was a plain dataclass with no __post_init__ and no
# assignment guard, so its price / size_shares annotations were
# unenforced: a level carrying a STRING price or size constructed
# unchecked, and every consumer then silently filtered it as
# non-live — _is_live_level (SF-BL-9398) returns False for a bool /
# non-numeric size_shares and _is_tradeable_price (SF-BL-9542)
# returns False for a non-numeric price — so a book quoted in JSON
# strings (the CLOB wire format: Polymarket quotes price/size as
# strings, cf. Gamma outcomePrices) evaluated as ONE_SIDED_BOOK
# with best_yes_ask at the 0.99 sentinel instead of failing as the
# type error it is. Verified live pre-fix (repro.py in the task
# work dir): OrderBookLevel(price='0.50', size_shares=100000),
# size_shares='100000' and size_shares=True all constructed, and
# each book SKIPped ONE_SIDED_BOOK at best_yes_ask 0.99.
# SF-BL-10826 covers ladder container/entry shape (None ladder,
# dict entry), not the level fields' types. Kill-criterion audit
# (PRODUCER_AUDIT.md in the task work dir): NOT met — 263 in-repo
# construction sites pass bare literals/variables with no
# validated-float step (the single float(...) call site is a
# test's deliberate NaN fixture), the only converting fleet
# scripts convert partially (r63: price only) and are unpinned,
# no in-repo deserialiser constructs snapshots from CLOB JSON at
# all, and the dataclasses are mutable (the repo's own tests
# assign level fields after construction). The fields' TYPES are
# therefore validated here: at OrderBookLevel construction
# (__post_init__), on assignment (__setattr__, the SF-BL-10823
# pattern), and again over both ladders in
# OrderBookSnapshot.__post_init__ (which catches a level that
# bypassed its own construction via object.__setattr__ /
# __dict__ / unpickling before being placed in a fresh snapshot).
# SCOPE — type only: the predicate is isinstance (int/float,
# bool excluded, the module's standing type discipline). Every
# NUMERIC value keeps its pre-fix behaviour bit-identically,
# including price 0.0 / 1.0 / 1.50 / -0.30 / NaN / +-inf and size
# 0 / negative / NaN, whose filtering identical-to-clean is the
# landed SF-BL-9398/9542/9473 contract; the check never calls
# math.isfinite, so a huge int (10**400) passes as the numeric
# type it is and keeps its SF-BL-9820 downstream handling. The
# validation never coerces: '0.50' is never parsed and True is
# never read as 1 — a string-quoting producer must convert with
# float() before construction, and hears so from the error.

def _level_field_is_numeric(value: object) -> bool:
    """True iff a level field is a real number type (SF-BL-11492):
    an int or float, bool excluded. Type ONLY — finiteness and
    range are the SF-BL-9398/9542 value surfaces, untouched here."""
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _validate_order_book_level_field(field_name: str, value: object) -> None:
    """Raise ValueError unless one OrderBookLevel field is numeric
    in type (SF-BL-11492)."""
    if not _level_field_is_numeric(value):
        raise ValueError(
            f"OrderBookLevel {field_name} {value!r} "
            f"(type {type(value).__name__}) is not a real number "
            "(SF-BL-11492): price and size_shares must be int/float "
            "(bool excluded) — a bool or non-numeric level field "
            "(e.g. the JSON string '0.50' a CLOB producer quotes) "
            "was silently filtered as a non-live rung pre-fix, so "
            "a string-quoted book read as ONE_SIDED_BOOK at the "
            "0.99 sentinel instead of failing as a type error; "
            "convert serialized values with float() before "
            "construction — values are never coerced here")


def _validate_order_book_level(price: object, size_shares: object) -> None:
    """Raise ValueError unless both level fields are numeric in
    type (SF-BL-11492); price is checked before size_shares."""
    _validate_order_book_level_field("price", price)
    _validate_order_book_level_field("size_shares", size_shares)


@dataclass
class OrderBookLevel:
    """Single price-depth rung in a Central Limit Order Book (CLOB).

    Field TYPES are enforced (SF-BL-11492): price and size_shares
    must be int/float (bool excluded) at construction and on
    assignment, else a named ValueError. Field VALUES are not
    policed here — numeric prices/sizes outside the tradeable
    domains are filtered identical-to-clean downstream
    (SF-BL-9398/9542).
    """
    price: float       # Contract price in (0, 1) or normalized payoff price
    size_shares: float # Available contract shares at this price level

    # SF-BL-11492: __setattr__ intercepts every assignment (the
    # dataclass __init__ assigns through it, so construction is
    # covered too) and __post_init__ re-checks the constructed
    # values. Writes that bypass both (object.__setattr__,
    # __dict__, unpickling) are caught when the level is placed
    # in a freshly constructed OrderBookSnapshot, whose
    # __post_init__ re-validates both ladders' level fields.
    def __setattr__(self, name: str, value: object) -> None:
        if name in ("price", "size_shares"):
            _validate_order_book_level_field(name, value)
        super().__setattr__(name, value)

    def __post_init__(self) -> None:
        _validate_order_book_level(self.price, self.size_shares)


# --- SF-BL-9363: fee_rate bound (pre-registered [0, 0.10]) ------------------
# fee_rate enters q_exec multiplicatively (price * (1 + fee_rate)) in both
# evaluate_and_size_opportunity and sweep_order_book, so an unvalidated
# negative fee silently LOWERS the execution price and manufactures gated
# edge: the prereg fixture (p=0.65, best ask 0.62) SKIPs at fee_rate 0.01
# (edge 0.024 < 0.05) but BUY_YES-sized $1,069.44 at q_exec 0.496 with
# fee_rate -0.20. No producer in the repo sources fee_rate from a validated
# venue schedule (benchmark fixtures, test helpers and the r63 scan all pass
# bare literals), so the value itself is validated here. The bound is
# [0, 0.10]: every in-repo fee (0.0-0.0175 in code, venue templates up to
# ~0.07 per SF-BL-0169) sits inside it, and anything outside it is a data
# error, not a price signal.
FEE_RATE_MIN = 0.0
FEE_RATE_MAX = 0.10


def _fee_rate_in_bounds(fee_rate: object) -> bool:
    """True iff fee_rate is a finite real number in [FEE_RATE_MIN, FEE_RATE_MAX]."""
    if isinstance(fee_rate, bool) or not isinstance(fee_rate, (int, float)):
        return False
    return math.isfinite(fee_rate) and FEE_RATE_MIN <= fee_rate <= FEE_RATE_MAX


# --- SF-BL-9438: non-finite portfolio_equity executes a NaN trade ---------
# portfolio_equity enters sizing directly (target_budget = equity *
# capped_fraction, actual_fraction = invested / equity, and the result is
# written into cluster_exposure_used), yet nothing validated it. With
# equity=NaN every comparison downstream is False in the dangerous
# direction: target_budget is NaN, sweep_order_book returned
# (NaN, NaN, NaN), and the caller's `invested <= 0` exhaustion check is
# False for NaN — so the opportunity EXECUTED as BUY_YES with invested,
# allocated and shares all NaN and wrote NaN into the cluster dict, where
# every later cap check on that cluster inherits it. With equity=+inf the
# sweep consumed the whole ladder while actual_fraction came out 0.0, so
# a full-book trade was invisible to the cluster cap. The valid domain is
# a positive finite real (a non-positive equity is the same caller capital
# error, previously mislabelled ORDER_BOOK_DEPTH_EXHAUSTED — SF-BL-9401).
# Kill-criterion audit (PRODUCER_AUDIT.md in the task work dir): there is
# NO single finite-only upstream producer — the benchmark derives equity
# iteratively from net_pnl_bid_marked sums, the r63 scan and the test
# helpers pass bare literals, and live/paper callers compute equity from
# account/PnL values with no finiteness invariant — so the fix belongs at
# this consumption gate, not at a producer assertion.

def _is_finite_number(value: object) -> bool:
    """True iff value is a finite real number (bool excluded)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        # SF-BL-9820: math.isfinite converts an int to float, so an int
        # beyond float range (|v| > ~1.8e308, e.g. 10**400) raises a
        # bare OverflowError here instead of returning a verdict —
        # leaking an unnamed crash out of every gate that reuses this
        # helper (the SF-BL-11535 fleet panel screen included). Such a
        # value is not a usable finite float for this module: False.
        return False


def _is_valid_portfolio_equity(equity: object) -> bool:
    """True iff portfolio_equity is a finite real number strictly > 0."""
    return _is_finite_number(equity) and equity > 0  # type: ignore[operator]


# --- SF-BL-9364: est_days_to_resolution <= 0 manufactures velocity -------
# est_days_to_resolution is the capital lockup D_lock: evaluate_fleet_
# north_star capital-weights it into avg_lockup and derives the velocity
# multiplier sqrt(365 / avg_lockup) from it, so its valid domain is a
# finite real strictly > 0. Pre-fix nothing validated its sign or range
# anywhere: OrderBookSnapshot.__post_init__ checked fee_rate
# (SF-BL-9363), on_chain_verified (SF-BL-10793) and ladder order
# (SF-BL-9051) but never this field, and the scorer's SF-BL-9475 gate
# rejected only NON-finite values. Verified live pre-fix (repro.py in
# the task work dir; $500 invested / +$100 PnL per trade, resolved
# [(0.6, 1), (0.6, 0)]): a -30d trade pooled with a +40d trade reported
# avg_lockup 5.0d and velocity 8.544x vs 3.021x for the clean +40/+40
# book — a 2.83x N_star velocity inflation from an impossible negative
# lock — and a single -30d (or 0d) book was masked by the
# max(1.0, avg_lockup) floor as a 1.0d lock, the maximum-velocity case
# (19.105x, N_star 1.4057 vs the honest 0.2223). Kill-criterion audit
# (PRODUCER_AUDIT.md in the task work dir): NOT met — no producer in the
# repo derives this field from a resolution timestamp at all; the
# benchmark fixtures and test helpers pass bare literals, the dataclass
# is mutable (mutation / unpickling bypass __post_init__), and
# TradeExecutionResult — which copies the field from the book — has no
# __post_init__ and is built / mutated directly, while
# settle_or_mark_trade never writes the field. The value itself is
# therefore validated at construction (raise, as for fee_rate), at
# evaluate entry for mutated/unpickled books (SKIP
# INVALID_EST_DAYS_TO_RESOLUTION), and at scoring in
# evaluate_fleet_north_star (raise, alongside SF-BL-9475). No upper
# bound is preregistered, so none is imposed.

def _est_days_in_domain(value: object) -> bool:
    """True iff est_days_to_resolution is a finite real number > 0."""
    return _is_finite_number(value) and value > 0  # type: ignore[operator]


def _validate_est_days_to_resolution(value: object) -> None:
    """Raise ValueError unless est_days_to_resolution is in domain."""
    # SF-BL-11534: the TYPE class is checked first and raises its own
    # named error (below); numeric values fall through to the
    # SF-BL-9364 domain message byte-identically.
    _validate_est_days_type(value)
    if not _est_days_in_domain(value):
        raise ValueError(
            f"est_days_to_resolution {value!r} outside pre-registered "
            "domain (0, finite] (SF-BL-9364): a non-positive or "
            "non-finite capital lockup is impossible — a negative lock "
            "pooled into the capital-weighted avg_lockup manufactures "
            "velocity (prereg fixture: -30d/+40d scored velocity "
            "8.544x vs 3.021x clean) and a single bad book is masked "
            "by the max(1.0, avg_lockup) floor as a 1.0d lock")


# --- SF-BL-11534: est_days_to_resolution TYPE is never named ----------
# SF-BL-9364 validated the lock's VALUE domain (finite, > 0) but lumped
# every failure into that one domain error, so a bool or non-numeric
# lock — the shape a JSON book feed actually delivers ('30'), or a
# bool producer flag (True, which Python would read as a 1-day lock) —
# was mislabelled as a non-positive / non-finite VALUE at construction,
# as "not a finite number > 0" at evaluate Gate 0.1, and, for a
# directly constructed / mutated TradeExecutionResult, only at scoring
# as SF-BL-9475's NON_FINITE_TRADE_FIELD — the mislabel the task's
# kill criterion names. Verified live pre-fix (repro.py in the task
# work dir). Kill-criterion audit (PRODUCER_AUDIT.md in the task work
# dir): NOT met — 108 in-repo OrderBookSnapshot sites pass bare
# literals, no in-repo deserialiser constructs snapshots from JSON at
# all, and both dataclasses are mutable (TradeExecutionResult
# validates only `action`, SF-BL-10823 — deliberately: evaluate's SKIP
# records carry the offending lock verbatim for audit, so the trade
# record itself must stay constructible with a bad lock and its
# consumption gate is the scorer). The TYPE is therefore named at
# three surfaces: construction (raise INVALID_EST_DAYS_TYPE, via
# _validate_est_days_to_resolution above), evaluate entry (a type gate
# immediately after Gate EB, SKIP INVALID_EST_DAYS_TYPE, before any
# other gate or sizing), and scoring (_validate_scored_trade_est_days_
# types, before SF-BL-9475, so a string/bool lock on an executed trade
# raises INVALID_EST_DAYS_TYPE instead of the NON_FINITE mislabel).
# TYPE ONLY: the predicate is the module's standing discipline —
# int/float, bool excluded, never coerced ('30' is never parsed, True
# is never read as 1) — and every numeric value, including NaN / +-inf
# / <= 0 / a huge int, keeps its pre-fix SF-BL-9364 / SF-BL-9475
# handling bit-identically.

def _est_days_is_numeric_type(value: object) -> bool:
    """True iff est_days is a real number TYPE (SF-BL-11534): an int
    or float, bool excluded. Type ONLY — finiteness and sign are the
    SF-BL-9364 value domain, untouched here."""
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _validate_est_days_type(value: object) -> None:
    """Raise ValueError unless est_days_to_resolution is numeric in
    type (SF-BL-11534)."""
    if not _est_days_is_numeric_type(value):
        raise ValueError(
            f"INVALID_EST_DAYS_TYPE: est_days_to_resolution {value!r} "
            f"(type {type(value).__name__}) is not a real number "
            "(SF-BL-11534): the capital lockup must be an int/float "
            "(bool excluded) — a bool or non-numeric lock (e.g. the "
            "JSON string '30' a book feed quotes, or True read as a "
            "1-day lock) is a TYPE error, not the non-positive / "
            "non-finite VALUE error of SF-BL-9364; convert serialized "
            "values with float() before construction — values are "
            "never coerced here")


# --- SF-BL-9470: finite out-of-range posteriors execute full-size --------
# calibrated_prob_yes is a probability, so its valid domain is the finite
# interval [0, 1]. Pre-fix it entered edge_yes/edge_no, Gate 3 and Kelly
# sizing with NO range check: p=1.50 cleared the hurdle and sized BUY_YES
# to the book (source fixture: $1,232.20; reproduced this run on the
# standard fixture at the full $1,500 cluster cap) and p=-0.20 did the
# same on the NO side (source fixture: $424.20 BUY_NO; $1,500 at the cap
# here), while +inf / -inf executed identically. NaN (SF-BL-9440) SKIPped
# only incidentally as INSUFFICIENT_NET_EDGE, not as an invalid input.
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): the
# [0.02, 0.98] clip inside aggregate_fleet_forecasts is NOT total for
# this argument — aggregate is only the benchmark's producer, the value
# is a plain float with no provenance, direct in-repo callers pass bare
# literals, and TradeExecutionResult objects are constructed/mutated
# outside that path — so the value itself is validated here, at the
# consumption gate, before any edge, gate or sizing computation.
POSTERIOR_MIN = 0.0
POSTERIOR_MAX = 1.0


def _posterior_in_bounds(p: object) -> bool:
    """True iff p is a finite real number in [POSTERIOR_MIN, POSTERIOR_MAX]."""
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        return False
    return math.isfinite(p) and POSTERIOR_MIN <= p <= POSTERIOR_MAX


# --- SF-BL-9440: a non-finite posterior is an INVALID_FORECAST -----------
# Filed (2026-10-21T16:52Z, before SF-BL-9470) on the NaN case alone:
# edge_yes/edge_no came out NaN, both >= comparisons failed, and the else
# branch formatted the NaN into a routine INSUFFICIENT_NET_EDGE skip
# ('max_edge=nan < 0.050'), so skip-reason telemetry — the fleet's
# selectivity accounting — silently absorbed an invalid forecast as
# market efficiency. The prereg asks for an explicit non-finite-posterior
# skip named INVALID_FORECAST, before edge computation, fixtured at NaN
# and +/-inf. SF-BL-9470's Gate 0.0 (added later the same day, AGREEd by
# its tester) already gave NaN/+inf/-inf an explicit INVALID_POSTERIOR
# reason before the edge screen; this task completes the prereg WITHOUT
# renaming that pinned token: Gate 0.0 is split, the non-finite /
# non-numeric case is checked FIRST via the helper below, and its SKIP
# reason carries both tokens — primary INVALID_POSTERIOR (the token
# SF-BL-9470's agreed tests pin with startswith, and the umbrella class
# for every invalid posterior) qualified as INVALID_FORECAST, the
# SF-BL-9440 class name — so telemetry can count invalid forecasts
# distinctly from finite out-of-range posteriors and from genuine
# INSUFFICIENT_NET_EDGE markets. Finite out-of-range posteriors keep
# SF-BL-9470's exact reason byte-identical. Kill-criterion audit
# (PRODUCER_AUDIT.md in the task work dir): NOT met — since SF-BL-9439 /
# SF-BL-9506, aggregate_fleet_forecasts IS NaN-proof for its own return
# (non-finite members dropped-and-logged; an all-non-finite panel raises
# ValueError; the return is Platt-clipped to [0.02, 0.98]), but it is
# NOT the sole posterior source for this argument: evaluate takes a
# plain float with no provenance, the only in-module production caller
# is the benchmark, direct in-repo callers (the test suites) pass bare
# literals, and out-of-tree fleet scripts do the same — so the explicit
# skip belongs here, at this function's entry, exactly as the prereg's
# non-kill branch prescribes (its regression test is
# tests/test_alpha_kelly_invalid_forecast.py).

def _posterior_not_finite(p: object) -> bool:
    """True iff p is not a finite real number: NaN, +/-inf, a bool, or a
    non-numeric value (SF-BL-9440). A finite out-of-range posterior is
    NOT covered here — that is SF-BL-9470's INVALID_POSTERIOR case."""
    return not _is_finite_number(p)


# --- SF-BL-9329: the validated posterior is clipped to the spec band ----
# The NORTH_STAR spec band for a calibrated posterior is [0.02, 0.98]
# (nstar.py: P_LO, P_HI = 0.02, 0.98 — "calibrated posterior lives in
# [0.02, 0.98]"), and every other sizing surface honours it: cli.py
# cmd_edge clips p with np.clip(a.p, 0.02, 0.98) and backtest_nstar.py
# clips to P_LO/P_HI. evaluate_and_size_opportunity was the ONLY sizing
# surface that did not: it fed calibrated_prob_yes raw into the edge
# test, the Gate 3 ratio and the Kelly fraction, so a direct caller
# passing 0.999 or 1.0 sized on a posterior the spec forbids (verified
# live pre-fix, repro.py in the task work dir: uncapped fixture,
# ask 0.50 / bid 0.48, fee 0.0, equity 10000 — p=0.999 invested $3,493.00
# and p=1.0 $3,500.00 vs the clipped equivalent p=0.98 at $3,360.00;
# p=0.01 invested $3,427.08 vs p=0.02 at $3,354.17; at the default
# cluster cap the divergence showed in kelly_fraction_raw 0.998 / 1.0
# vs 0.96 and in the stored calibrated_prob_yes). The aggregate-fed
# benchmark path was in-band by construction (aggregate's Platt step
# clips to [0.02, 0.98]), which is why the fixture benchmark never saw
# it. Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT
# met — aggregate is only the benchmark's producer; direct in-repo
# callers (the test suite) and out-of-tree fleet scripts (wave3 r63
# scan) pass bare float literals with no provenance, so the clip
# belongs here, at the consumption surface. ORDERING vs SF-BL-9470:
# Gate 0.0 validates [0, 1] FIRST and out-of-domain posteriors still
# SKIP INVALID_POSTERIOR — the clip never rescues them (this task's
# -0.05 fixture is governed by SF-BL-9470: clipping it to 0.02 would
# resurrect exactly the out-of-domain trades that prereg forbids, and
# would likewise turn 1.50 into a 0.98 trade). The clip therefore
# applies to the VALIDATED posterior, immediately after Gate 0.0:
# in-band values pass byte-identical (aggregate output included, so
# the benchmark is unchanged), and every downstream consumer — edge
# screen, Gate 3, Kelly, the sweep belief, KL / log-growth — plus the
# stored calibrated_prob_yes on the result read the clipped value, so
# a call at 0.999 / 1.0 returns a result identical in every field to
# a call at 0.98 (likewise 0.0 / 0.01 vs 0.02). The Gate E0 / Gate 0.0
# SKIP records above the clip intentionally keep the caller's raw
# value: they report the invalid input itself.
POSTERIOR_CLIP_MIN = 0.02
POSTERIOR_CLIP_MAX = 0.98


def _validate_fee_rate(fee_rate: object) -> None:
    """Raise ValueError unless fee_rate is in the pre-registered bound."""
    if not _fee_rate_in_bounds(fee_rate):
        raise ValueError(
            f"fee_rate {fee_rate!r} outside pre-registered bound "
            f"[{FEE_RATE_MIN}, {FEE_RATE_MAX}] (SF-BL-9363): a negative or "
            "outsized fee would manufacture or erase gated edge via "
            "price * (1 + fee_rate)")


# --- SF-BL-10793: on_chain_verified must be a strict bool -----------------
# Gate 1 (CryptoBench) read `not book.on_chain_verified` with no type
# check, and OrderBookSnapshot.__post_init__ validated only fee_rate
# (SF-BL-9363), so Python truthiness decided the gate: any non-empty
# string from a JSON/CSV producer is truthy, and on the prereg fixture
# (CRYPTO, ask 0.50 / bid 0.48, p=0.80, fee 0.01) the string 'False'
# traded BUY_YES $1,042.93 instead of SKIPping
# FAILED_CRYPTOBENCH_ONCHAIN_GATE, exactly as int 1, float 0.5 and the
# strings 'no' / '0' / 'True' did; falsy non-bools (0, '', None, [])
# SKIPped only incidentally, with the genuine-unverified reason,
# indistinguishable from a real bool False. Verified live pre-fix.
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT
# met — no in-repo producer sources on_chain_verified from a
# schema-validated source: the benchmark fixtures and test helpers pass
# bare bool literals, no code constructs an OrderBookSnapshot from
# JSON/CSV at all (so no validated deserialisation stands between an
# external producer's string and the gate), OrderBookSnapshot is a
# mutable dataclass (post-construction mutation and unpickling bypass
# __post_init__, and the repo's own tests mutate this field), and the
# field annotation `bool` is not enforced at runtime. The value itself
# is therefore validated at construction (raise, as for fee_rate) and
# re-validated at the consumption gate in
# evaluate_and_size_opportunity (SKIP INVALID_ON_CHAIN_VERIFIED),
# which covers mutated/unpickled books. Strict bool means
# isinstance(v, bool): int 1/0 are NOT bools (bool subclasses int, not
# the reverse), so they are invalid, as are 'True'/'False' strings,
# floats and None.

def _is_strict_bool(value: object) -> bool:
    """True iff value is a Python bool (int 1/0 do not qualify)."""
    return isinstance(value, bool)


def _validate_on_chain_verified(value: object) -> None:
    """Raise ValueError unless on_chain_verified is a strict bool."""
    if not _is_strict_bool(value):
        raise ValueError(
            f"on_chain_verified {value!r} is not a bool "
            f"(type {type(value).__name__}) (SF-BL-10793): the "
            "CryptoBench gate requires a strict bool; a truthy string "
            "such as 'False' from a JSON/CSV producer must never arm it")


# --- SF-BL-9436: liquidation_bid_price bound (pre-registered [0, 1]) --------
# A binary contract pays $1.00 at settlement and $0.00 otherwise, so any
# liquidation (mark-to-market) bid for an open position must lie in
# [0, 1]: shares * bid is the position's liquidation value. The unsettled
# branch of settle_or_mark_trade multiplied shares by liquidation_bid_price
# with no check, so bid 1.50 booked +$1,000 PnL on a $500 stake (1000
# shares: value $1,500, twice the binary maximum payout) and bid -0.30
# booked a -$800 loss exceeding the stake (value -$300) — both feeding
# evaluate_fleet_north_star's CWR and MDD directly. The kill condition
# (bids only ever come from OrderBookSnapshot best-bid properties bounded
# to (0,1)) is NOT met: OrderBookLevel carries no price validation and
# best_yes_bid / best_no_bid return the raw level price (verified: a
# -0.30 level yields best_yes_bid -0.30 and best_no_bid -0.50), and
# TradeExecutionResult is also constructed directly and mutated outside
# evaluate_and_size_opportunity. The value itself is therefore validated
# in settle_or_mark_trade, the point where it becomes PnL.
LIQUIDATION_BID_MIN = 0.0
LIQUIDATION_BID_MAX = 1.0


def _liquidation_bid_in_bounds(bid: object) -> bool:
    """True iff bid is a finite real number in [LIQUIDATION_BID_MIN, LIQUIDATION_BID_MAX]."""
    if isinstance(bid, bool) or not isinstance(bid, (int, float)):
        return False
    return math.isfinite(bid) and LIQUIDATION_BID_MIN <= bid <= LIQUIDATION_BID_MAX


def _validate_liquidation_bid(bid: object) -> None:
    """Raise ValueError unless bid is in the pre-registered bound [0, 1]."""
    if not _liquidation_bid_in_bounds(bid):
        raise ValueError(
            f"liquidation_bid_price {bid!r} outside pre-registered bound "
            f"[{LIQUIDATION_BID_MIN}, {LIQUIDATION_BID_MAX}] (SF-BL-9436): "
            "a binary contract's liquidation value cannot exceed $1.00 per "
            "share or fall below $0.00; an out-of-bound bid would fabricate "
            "PnL beyond the stake / maximum payout and corrupt CWR and MDD")


# --- SF-BL-10767: liquidation-bid marks carry no freshness provenance ---
# evaluate_and_size_opportunity stamps liquidation_bid_price from the
# entry book's best bid (via _declared_liquidation_bid), and the
# unsettled branch of settle_or_mark_trade re-marks at whatever value
# sits in that field — SF-BL-9436 validates only that the bid is in
# [0, 1]. Pre-fix neither OrderBookSnapshot nor TradeExecutionResult
# carried any as-of time or source for the bid, and
# evaluate_fleet_north_star summed net_pnl_bid_marked into CWR_bid and
# the MDD walk with no scoring-time input, so a position never
# re-quoted contributed a current-looking bid mark indefinitely
# (repro.py in the task work dir: an entry bid observed 7 days before
# scoring booked +$120.00 and scored CWR 0.24 / N_star 0.308,
# indistinguishable from a fresh mark).
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT
# met — the only production mark caller, run_benchmark_simulation,
# passes exit_book=book, the SAME entry book object from the same
# fixture iteration (a same-instant re-read, not a re-quote; and all
# 8 fixtures are settled, so the bid never feeds their PnL), the
# legacy no-exit-book settle path refreshes nothing by construction,
# and with no timestamp field anywhere "refreshed within one cadence"
# was not even expressible.
# PRE-REGISTERED RULE (this task): provenance + staleness.
#   * OrderBookSnapshot.snapshot_ts records when a book was observed
#     (epoch seconds; None = the producer did not say). evaluate stamps
#     it onto the trade as liquidation_bid_as_of with source
#     "entry_book"; settle's exit-book path re-stamps both fields from
#     the exit book (source "exit_book"); the legacy no-exit-book mark
#     and the settled branch leave the stamps untouched, so a stored
#     bid keeps its entry provenance and its age stays visible.
#   * The book's snapshot cadence: no cadence constant existed in the
#     repo; books are re-snapshotted per trading cycle, whose only
#     documented length is the 15-45 minute cycle (Prediction Arena
#     S5.2.2 — see build_cycle_learning_prompt_block). The cadence is
#     therefore pre-registered as the LONGEST documented cycle, 2700s:
#     a mark older than one full cadence at scoring has missed at
#     least one re-snapshot opportunity and is STALE by definition.
#   * At scoring, evaluate_fleet_north_star takes scored_at (epoch
#     seconds, optional). An UNSETTLED executed trade whose bid as-of
#     is missing/invalid, or whose age at scored_at exceeds one
#     cadence, is EXCLUDED from every sum (CWR, equity/MDD walk,
#     lockup, returns) and counted on the scorecard
#     (stale_mark_count / stale_mark_invested); settled trades are
#     never excluded (their PnL pays $1/$0 and never read the bid).
#     Unprovenanced open marks are counted in
#     unprovenanced_mark_count whether or not scored_at is given.
#     scored_at=None (every legacy caller) excludes nothing and is
#     byte-identical to pre-fix scoring apart from the new count
#     fields. Provenance failure degrades to exclusion/flagging,
#     never to a raise: the honest response to an untrusted mark is
#     to not score it, and a mutated book/trade carrying a non-finite
#     stamp reads as unprovenanced rather than crashing the scorer.
MARK_STALENESS_CADENCE_SECONDS = 2700.0  # 45 min: longest documented cycle
BID_SOURCE_ENTRY_BOOK = "entry_book"
BID_SOURCE_EXIT_BOOK = "exit_book"


def _snapshot_ts(book: object) -> Optional[float]:
    """The book's observation time as a finite float, else None.

    Reads OrderBookSnapshot.snapshot_ts defensively (SF-BL-10767): a
    missing attribute, a bool, a non-numeric value or a non-finite
    number (a mutated/unpickled book bypassing __post_init__) all read
    as None — provenance unknown — never as a usable time.
    """
    value = getattr(book, "snapshot_ts", None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _validate_snapshot_ts(value: object) -> None:
    """Raise ValueError unless snapshot_ts is None or finite >= 0."""
    if value is None:
        return
    if not _is_finite_number(value) or value < 0:  # type: ignore[operator]
        raise ValueError(
            f"snapshot_ts {value!r} is not a finite epoch-second "
            "timestamp >= 0 (SF-BL-10767): a book's observation time "
            "must be a real, finite time or None (unknown); a NaN, "
            "infinite, negative or non-numeric stamp would fabricate "
            "or destroy the freshness provenance of every mark taken "
            "from the book")


def _bid_as_of(trade: TradeExecutionResult) -> Optional[float]:
    """The trade's bid observation time as a finite float, else None."""
    value = getattr(trade, "liquidation_bid_as_of", None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _mark_is_stale(trade: TradeExecutionResult, scored_at: float) -> bool:
    """True iff the trade's bid mark is stale at `scored_at` (SF-BL-10767).

    Pre-registered rule: an unsettled trade's mark is stale when its
    as-of is missing/invalid (unprovenanced) or its age
    scored_at - as_of exceeds one snapshot cadence
    (MARK_STALENESS_CADENCE_SECONDS). An age of exactly one cadence is
    NOT stale (the mark was refreshed at the last cadence boundary);
    a negative age (as-of after scored_at) is not stale either — the
    mark is newer than the scoring instant, not older.
    """
    as_of = _bid_as_of(trade)
    if as_of is None:
        return True
    return (scored_at - as_of) > MARK_STALENESS_CADENCE_SECONDS


# --- SF-BL-9541: shares_acquired / invested_capital domain at settle -----
# settle_or_mark_trade turns shares_acquired directly into PnL in BOTH
# branches (settled: gross_payout = shares * 1.0; open: liquidation_value
# = shares * bid) and subtracts invested_capital, yet pre-fix neither
# field was validated there beyond the `invested_capital <= 0` early
# return — which is False for NaN and silently MASKS a negative stake as
# a zero-PnL trade. TradeExecutionResult is a plain mutable dataclass
# with no __post_init__ validation, constructed directly (the benchmark's
# naive fleet, callers, tests) and mutated after construction, so no
# producer invariant bounds these fields (the kill condition — an
# out-of-domain count can never reach this function — is therefore NOT
# met, exactly as for the SF-BL-9436 bid surface). Verified live pre-fix
# on the prereg fixture ($500 stake, BUY_YES): shares=-100 on a WON trade
# booked net PnL -$600 (a loss exceeding the stake on a win),
# shares=+inf booked +inf PnL, shares=NaN booked NaN PnL, invested=NaN
# booked NaN PnL and invested=+inf booked -inf PnL — all feeding
# evaluate_fleet_north_star's CWR and MDD. The valid domain for both
# fields is a finite real >= 0 (zero included: SKIP / zero-size trades
# carry 0.0/0.0 and keep their existing zero-PnL early return). The check
# lives at settle_or_mark_trade ENTRY — before any field of the trade is
# mutated, before the SKIP/zero-invested early return, and before either
# branch — so a rejected trade's net_pnl_bid_marked is left untouched
# and no out-of-domain count can reach gross_payout or liquidation_value.

def _shares_acquired_in_domain(shares: object) -> bool:
    """True iff shares is a finite real number >= 0 (bool excluded)."""
    return _is_finite_number(shares) and shares >= 0  # type: ignore[operator]


def _invested_capital_in_domain(invested: object) -> bool:
    """True iff invested is a finite real number >= 0 (bool excluded)."""
    return _is_finite_number(invested) and invested >= 0  # type: ignore[operator]


def _validate_settle_position_fields(trade: TradeExecutionResult) -> None:
    """Raise ValueError unless the trade's position fields are in domain.

    Validates shares_acquired first, then invested_capital (SF-BL-9541);
    the error names the offending field, its value and the task id.
    """
    if not _shares_acquired_in_domain(trade.shares_acquired):
        raise ValueError(
            f"shares_acquired {trade.shares_acquired!r} outside "
            "pre-registered domain [0, finite] (SF-BL-9541): a negative, "
            "NaN or infinite share count would fabricate settled payout "
            "or liquidation value (e.g. shares=-100 on a won trade books "
            "a loss exceeding the stake) and corrupt CWR and MDD")
    if not _invested_capital_in_domain(trade.invested_capital):
        raise ValueError(
            f"invested_capital {trade.invested_capital!r} outside "
            "pre-registered domain [0, finite] (SF-BL-9541): a negative, "
            "NaN or infinite stake cannot be subtracted into an honest "
            "net PnL (NaN/inf stakes booked NaN/-inf PnL pre-fix, and a "
            "negative stake was silently masked as zero PnL)")


# --- SF-BL-9477: bool outcomes pass the {0, 1} membership test -----------
# settle_or_mark_trade's settled branch guarded the outcome with
# `outcome_yes not in (0, 1)` -- an EQUALITY membership test. Python bool
# is an int subclass (True == 1, False == 0), so a bool outcome settled
# silently as a genuine int outcome while 0.5 correctly raised.
# Verified live pre-fix (repro_prefix.py in the task work dir): on the
# titled fixture (BUY_YES, 1267.80 shares, $500.00 invested)
# outcome_yes=True booked net PnL +$767.80 -- bit-identical to int 1 --
# and False booked -$500.00, bit-identical to int 0; floats 1.0 / 0.0
# and numpy bool_ passed the same way. The fleet's bool policy is
# enforced everywhere else this hazard class was found -- this module's
# own validators reject bool ahead of the numeric check (fee_rate,
# liquidation bid, ladder sizes), agents.py rejects (bool, np.bool_)
# outcomes outright, and nstar._finite_float rejects bools for exactly
# this reason -- so the outcome is validated here, at the point where
# it becomes PnL, against a STRICT int domain: the type must be
# Integral (Python int or a numpy integer; bool and np.bool_ excluded)
# and the value must be 0 or 1. Floats 1.0 / 0.0 raise with the same
# ValueError as 0.5: the parameter is annotated Optional[int], and a
# float outcome is a producer type error, not a settlement.
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT
# met -- the only in-repo production caller is run_benchmark_simulation,
# fed by bare dict literals under an unenforced `outcome_yes: int`
# annotation; no schema-validated resolution pipeline stands between a
# producer and this function, and a resolved YES/NO flag is most
# naturally a Python bool at any future call site.

def _outcome_yes_in_domain(outcome: object) -> bool:
    """True iff outcome is a strict integer 0 or 1 (bool excluded)."""
    if isinstance(outcome, bool):
        return False
    return isinstance(outcome, numbers.Integral) and outcome in (0, 1)


def _validate_outcome_yes(outcome: object) -> None:
    """Raise ValueError unless outcome_yes is a strict int in {0, 1}.

    The message is the pre-existing settled-branch message, so a bool
    (or float / string / None) outcome raises the SAME ValueError that
    0.5 raised pre-fix (SF-BL-9477 pre-registered done criterion).
    """
    if not _outcome_yes_in_domain(outcome):
        raise ValueError("Settled trade requires outcome_yes in {0, 1}.")


# --- SF-BL-9398: zero/negative-size levels are not prices ------------------
# A ladder level with size_shares <= 0 cannot fill a single share, so it
# must never set a best price, anchor mid, derive a NO price, pass the
# evaluate edge screen, or block / be swept in sweep_order_book. Pre-fix,
# best_yes_ask / best_yes_bid were yes_asks[0].price / yes_bids[0].price
# with no size check: asks [(0.40, 0), (0.62, 100000)] made 0.40 the best
# ask (mid 0.50, screened edge 0.246 at p=0.65 vs the real 0.0238 at the
# first fillable ask 0.62), and bids [(0.88, 0), (0.50, 100000)] derived a
# phantom best_no_ask of 0.12 (= 1 - 0.88). The fix is at the source:
# every best-price property, the derived NO prices, mid, the edge screen
# (which reads those properties) and the sweep ladder all consider only
# live levels (size_shares > 0, finite). A side with no live level keeps
# its pre-existing empty-list sentinel (ask 0.99 / bid 0.01), which makes
# that side's screened edge negative by construction; a book with no live
# level on either side SKIPs as empty in evaluate (Gate 0.5).
# SUPERSESSION (SF-BL-10760): the sentinel making the DEAD side's entry
# edge negative never protected the LIVE side's trade — edge_yes reads
# only the ask, so an asks-only book's BUY_YES executed with the 0.01
# bid sentinel as its liquidation price. One-sided books are therefore
# no longer tradable at all: evaluate Gate 0.7 SKIPs them as
# ONE_SIDED_BOOK (see the SF-BL-10760 block at that gate).

# --- SF-BL-11046: non-OrderBookLevel ladder entries crash the sweep ---
# _is_live_level dereferenced level.size_shares unconditionally, so a
# ladder entry that is not an OrderBookLevel at all — a dict entry
# ({'price': 0.50, 'size_shares': 1000}, the JSON-decoded CLOB shape) or
# a None entry (a missing rung) — raised a bare AttributeError on
# size_shares inside the live-level filter, aborting sweep_order_book
# (and, through the same predicate, OrderBookSnapshot construction via
# _validate_ladder_order and evaluate's ladder gates) instead of being
# skipped like every other unreadable rung. Verified live pre-fix
# (repro.py in the task work dir): both prereg fixtures raised
# AttributeError from _is_live_level while the clean book swept
# (1000.0, 1800.18..., 0.5555). Distinct from SF-BL-10964 (the book
# argument's own type in this same function — a genuine
# OrderBookSnapshot carrying corrupt ENTRIES passes that check) and
# from SF-BL-10826 (ladder container/entry validation scoped to
# evaluate_and_size_opportunity's gates, which direct sweep callers
# such as the naive baseline never pass through). Kill-criterion audit
# (CALLER_CENSUS.md in the task work dir): NOT met — the constructor
# deliberately skips non-OrderBookLevel entries
# (_validate_snapshot_level_types defers entry shape to SF-BL-10826),
# the dataclasses are mutable so assignment / unpickling bypasses
# __post_init__ regardless, direct sweep callers pass books straight
# through, and no test pins an OrderBookLevel-only ladder invariant.
# The fix follows the fleet's identical-to-clean rule for unreadable
# rungs (SF-BL-9398/9542): a non-OrderBookLevel entry is not a level,
# so it is NOT live — _is_live_level returns False for it before any
# dereference, every consumer of the predicate (best prices, mid,
# ladder order, fallback sides, the sweep ladders, the exit walk)
# skips it exactly as if the rung were absent, and sweep_order_book
# logs the skipped count for the ladder it walks (see the SF-BL-11046
# note there). Genuine OrderBookLevel entries are untouched: the
# isinstance check is True for them (subclasses included) and the
# size test below runs byte-identically.

def _is_live_level(level: OrderBookLevel) -> bool:
    """True iff a ladder level can fill: finite size_shares strictly > 0.

    SF-BL-11046: an entry that is not an OrderBookLevel (a dict, None,
    or any other decoded/unconstructed value) is not a level and is
    never live — False is returned before size_shares is read, so the
    entry is skipped identical-to-clean instead of raising the bare
    AttributeError the pre-fix dereference produced."""
    if not isinstance(level, OrderBookLevel):
        return False
    size = level.size_shares
    if isinstance(size, bool) or not isinstance(size, (int, float)):
        return False
    return math.isfinite(size) and size > 0


def _live_levels(levels: List[OrderBookLevel]) -> List[OrderBookLevel]:
    """The levels of `levels` that can fill, in original (sorted) order."""
    return [lvl for lvl in levels if _is_live_level(lvl)]


def _first_live_price(levels: List[OrderBookLevel], default: float) -> float:
    """Price of the first live level, or `default` when none is live."""
    for lvl in levels:
        if _is_live_level(lvl):
            return lvl.price
    return default


# --- SF-BL-9542: rung PRICE validity for best prices & the YES ladder ----
# SF-BL-9398 filtered rung SIZE everywhere and SF-BL-9473 filtered rung
# PRICE in the sweep ladders, but the best-price properties above the
# sweep still returned the raw first live price: asks [(-0.10, 1000),
# (0.50, 1000)] at fee 0.01 gave best_yes_ask -0.10, mid 0.15 and
# best_no_bid 1.10, and evaluate's edge screen read that phantom ask, so
# the negative rung manufactured edge and inflated Kelly sizing (on a
# deep valid rung: kelly_raw 0.682 / invested $1,500 at the cluster cap
# vs the clean book's 0.293 / $1,025.25) even though the sweep itself
# filled only the 0.50 rung. Symmetrically a -0.30 YES bid derived
# best_no_ask 1.30 and BLOCKED the clean BUY_NO trade behind it, and a
# NaN ask poisoned mid and (via the SF-BL-9473 carve-out) the sweep's
# return tuple. The fix is the fleet's identical-to-clean rule: a rung
# whose price is not a finite real in the open interval (0, 1) — the
# SF-BL-9473 contract-price domain; at exactly 1.0 the fee-adjusted
# price exceeds the $1.00 binary payout, so a 1.0 rung is likewise
# non-tradeable — is not a price at all and is skipped by the best-price
# properties exactly as if the rung were absent, so no invalid price
# can alter best prices, mid, the derived NO prices, the edge screen,
# liquidation bids, invested, shares or q_exec. A side with no
# tradeable live level keeps its SF-BL-9398 sentinel (ask 0.99 / bid
# 0.01), preserving the empty-side semantics.

def _is_tradeable_price(price: object) -> bool:
    """True iff price is a finite real number in the open interval (0, 1).

    Same domain as _is_valid_contract_price (SF-BL-9473, sweep ladders):
    finite (bool / non-numeric excluded) and strictly inside the binary
    contract's $0-$1 payout bounds.
    """
    return _is_finite_number(price) and 0.0 < price < 1.0  # type: ignore[operator]


def _first_tradeable_price(levels: List[OrderBookLevel], default: float) -> float:
    """Price of the first live level carrying a tradeable price (SF-BL-9542),
    or `default` when no live level is tradeably priced."""
    for lvl in levels:
        if _is_live_level(lvl) and _is_tradeable_price(lvl.price):
            return lvl.price
    return default


def _has_tradeable_ladder(levels: List[OrderBookLevel]) -> bool:
    """True iff the ladder can fill or liquidate at a real price (SF-BL-10760):
    at least one LIVE level (SF-BL-9398) carrying a TRADEABLE price
    (SF-BL-9542). This is exactly the population the entry sweep and the
    size-walked exit walk can use, so a ladder failing it provides no
    entry and no exit."""
    return any(
        _is_live_level(lvl) and _is_tradeable_price(lvl.price) for lvl in levels
    )


# --- SF-BL-3549: declared no-mark rule for sentinel-derived prices -------
# The best-price properties keep their SF-BL-9398/9542 sentinel contract
# (ask 0.99 / bid 0.01 when a side has no live tradeable level): the
# sentinels exist so the ENTRY edge screen reads a dead side's entry
# edge as negative, and prior tasks pin them at the property level.
# A sentinel is not a price, however, and SF-BL-10760's Gate 0.7 closed
# only the evaluate entry path for one-sided books. Verified live
# pre-fix (repro.py in the task work dir), the defaults were still
# consumed as real prices on engine records:
#   * Gate 0.5's EMPTY_BOOK record carried market_mid_prob 0.50 — the
#     phantom mid of this task's title, 0.5 * (0.99 + 0.01);
#   * every evaluate gate ordered before Gate 0.5/0.7 (config,
#     posterior, fee, dispersion, equity, NaN-rung, on-chain) stamped
#     book.mid_price on its SKIP record, so an ask-only book tripping
#     INVALID_POSTERIOR carried mid 0.255 and a bid-only book 0.745;
#   * a book live but invalid-priced on BOTH sides passed Gate 0.5 and
#     Gate 0.7 (XOR False) and SKIPped INSUFFICIENT_NET_EDGE at mid 0.50;
#   * the naive baseline in run_benchmark_simulation bypasses evaluate
#     entirely: it chose its side from mid_price and stored the raw
#     best-bid property as liquidation_bid_price, marking an open YES
#     position on an ask-only book at the 0.01 sentinel (-$1,176.24 on
#     a $1,200 flat stake in the repro).
# DECLARED RULE (the rule this task's prereg requires in place of the
# defaults): where a side has no live, tradeably-priced ladder there is
# NO observable price on that side. On any engine record, the declared
# mid is the real mid_price only when BOTH ladders are observable,
# otherwise 0.0 ("no mid"); the declared liquidation bid for a side is
# its real best bid only when that side's exit ladder is observable,
# otherwise 0.0 ("no exit — mark at zero"). 0.0 is the convention the
# engine already uses for exactly this meaning: Gate 0.7's SKIP record
# (SF-BL-10760) and size_walked_exit_bid's no-depth return
# (SF-BL-7432). Two-sided tradeable books are byte-identical under the
# helpers below. Kill-criterion scan (scan_books.py / SCAN.md in the
# task work dir): 426 of 1,131 archived forward S2 books are one-sided
# — the defaults are heavily reached, NOT unreachable, so the kill
# does not trigger and the rule is load-bearing.

def _declared_mid_prob(book: OrderBookSnapshot) -> float:
    """The mid that may appear on an engine record (SF-BL-3549 declared
    rule): book.mid_price when both ladders carry a live, tradeably-
    priced level, else 0.0 — never a sentinel-derived 0.50 / 0.255 /
    0.745 phantom mid."""
    if _has_tradeable_ladder(book.yes_asks) and _has_tradeable_ladder(book.yes_bids):
        return book.mid_price
    return 0.0


def _declared_liquidation_bid(book: OrderBookSnapshot, side: TradeAction) -> float:
    """The liquidation bid that may be stored for `side` (SF-BL-3549
    declared rule): BUY_YES exits on yes_bids, BUY_NO on the derived NO
    bid (1 - best YES ask); the real best bid is returned only when
    that exit ladder is live and tradeably priced, else 0.0 ("no exit
    observable — mark at zero"), never the 0.01 sentinel or the
    1 - 0.99 derived sentinel."""
    if side == TradeAction.BUY_YES:
        return book.best_yes_bid if _has_tradeable_ladder(book.yes_bids) else 0.0
    if side == TradeAction.BUY_NO:
        return book.best_no_bid if _has_tradeable_ladder(book.yes_asks) else 0.0
    return 0.0


# --- SF-BL-6060: empty-ladder fallback prices fabricate a book ---------
# The best-price properties keep their SF-BL-9398/9542 sentinel contract
# (ask 0.99 / bid 0.01 when a side has no live, tradeably-priced level):
# the sentinels are pinned at the property level by prior tasks. A
# sentinel is not a measured touch, so no evaluation path may consume a
# fallback-backed price as if it were real: every path must either
# REJECT the fallback book with a named reason or use a declared rule
# (SF-BL-3549) that never reads the sentinel. The census for this task
# (CENSUS.md in the task work dir) walked every in-module consumer of
# best_yes_ask / best_yes_bid / best_no_ask / best_no_bid / mid_price:
#   * evaluate records — REJECT / declared: Gate 0.5 EMPTY_BOOK, Gate
#     0.7 ONE_SIDED_BOOK (SF-BL-10760), every market_mid_prob via
#     _declared_mid_prob and every liquidation bid via
#     _declared_liquidation_bid (SF-BL-3549), naive baseline
#     NO_OBSERVABLE_PRICE (SF-BL-3549), settle's exit-book walk
#     (live+tradeable only, 0.0 on no depth).
#   * THREE paths still consumed a fallback with no rejection and no
#     flag, verified live pre-fix (repro.py in the task work dir):
#     (1) a book live on both sides but tradeable on NEITHER (e.g.
#         asks [(-0.10, 1000)], bids [(1.50, 1000)], or live-invalid
#         asks with raw-empty bids) passed Gate 0.5 (live levels
#         exist) and Gate 0.7 (the XOR is False when BOTH sides lack
#         a tradeable ladder) and reached evaluate's edge screen,
#         which screened edge_yes / edge_no against the 0.99 / 0.01
#         sentinels and SKIPped INSUFFICIENT_NET_EDGE
#         (max_edge=-0.200) — a fallback-screened verdict mislabelled
#         as a real market with insufficient edge;
#     (2) sweep_order_book's SF-BL-3840 LCB stop defaulted
#         prior_prob_side to book.mid_price — on an ask-only
#         multi-rung book the 0.255 sentinel-derived mid shrank the
#         stop belief so the sweep filled $505.00 where the real
#         two-sided prior (0.49) filled $5,000.00 on the same ladder;
#     (3) _is_crossed_book compared the raw properties, so an
#         ask-only book with best ask 0.005 read CROSSED against the
#         0.01 bid sentinel — a crossing fabricated from a fallback
#         (its sole in-module caller sits behind Gates 0.5/0.7, but
#         the predicate itself was fallback-unsafe for any caller).
# Fixes, all at the consumption points, properties untouched:
#   * _fallback_price_sides(book) names the sides whose best price is
#     fallback-backed (no live tradeable ladder) — the single
#     definition every fix below and any future fallback flag use.
#   * Evaluate Gate 0.7b NO_TRADEABLE_PRICE (after Gate 0.7, before
#     Gate 0.8): a book tradeable on NEITHER side SKIPs with that
#     named reason, zero sizes, declared mid 0.0 — the edge screen
#     never reads a sentinel again.
#   * sweep_order_book's LCB default prior fails closed with
#     (0.0, 0.0, 0.0) — the SF-BL-3840 invalid-input convention — when
#     the book has any fallback side; an EXPLICIT prior_prob_side is
#     the caller's own value, not a fallback, and is untouched.
#   * _is_crossed_book requires both ladders tradeable before
#     comparing, so a fallback side can neither fabricate nor mask a
#     crossing; two-sided tradeable books are bit-identical.
# Historical rate (scan_corpora.py / scan_results.json in the task
# work dir): 23,441 of 69,931 deduped Polymarket CLOB chunk books
# across all archived captures 2026-10-18..2026-10-24 (33.52%) are
# fallback-exposed (23,004 ask-only, 427 bid-only, 10 untradeable on
# both sides), and SF-BL-10760's pre-fix replay executed 34.65% of its
# gate-passing candidates on one-sided books — the rate is NOT zero,
# so the kill criterion does NOT trigger.

def _fallback_price_sides(book: OrderBookSnapshot) -> Tuple[str, ...]:
    """The sides whose best price is fallback-backed (SF-BL-6060):
    "yes_asks" when that ladder has no live, tradeably-priced level
    (best_yes_ask is the 0.99 sentinel and best_no_bid its derived
    0.01), "yes_bids" likewise for the 0.01 / derived-0.99 pair. An
    empty tuple means both best prices are measured touches and
    mid_price is a real mid."""
    sides = []
    if not _has_tradeable_ladder(book.yes_asks):
        sides.append("yes_asks")
    if not _has_tradeable_ladder(book.yes_bids):
        sides.append("yes_bids")
    return tuple(sides)


# --- SF-BL-9365: crossed / locked books are not tradable -----------------
# A live CLOB never rests crossed: best ask <= best bid means the two
# ladders cannot coexist (a stale snapshot, a crossed feed, or corrupt
# data), yet pre-fix nothing in the gate path compared them. Verified
# live pre-fix (repro.py in the task work dir): asks 0.40 / bids 0.60,
# fee 0.01, p=0.62 EXECUTED BUY_YES $1,268.46 at q_exec 0.404 — buying
# YES at 0.404 while the same book bids 0.60 for it — with mid_price
# reporting an innocent 0.50 and best_no_ask deriving to 0.40
# (= 1 - 0.60) from the crossed bid, so the mirror side was corrupted
# too (p=0.30 sized BUY_NO $1,500.00 at the full cluster cap); a locked
# book (ask == bid == 0.50) traded $813.13 the same way. The crossed
# relation also breaks every downstream identity the engine relies on:
# entry price below exit bid inverts the round-trip cost, and
# best_no_ask == best_yes_ask / best_no_bid == best_yes_bid collapse
# the YES/NO mirror. Kill-criterion audit (CONSTRUCTOR_CENSUS.md in
# the task work dir): NOT met — 45 in-repo OrderBookSnapshot
# construction sites pass bare literals; __post_init__ validates
# fee_rate, the verification flag and WITHIN-ladder sortedness only
# (a crossed book is sorted within each ladder, so it constructs
# silently); the dataclass is mutable, so post-construction mutation
# and unpickling bypass construction regardless; and no in-repo
# deserialiser bounds the ask/bid relation. The guard therefore
# belongs at the consumption gate (evaluate Gate 0.8), exactly as for
# SF-BL-9438 / SF-BL-9470 / SF-BL-10760. Construction stays permissive
# on purpose: a crossed snapshot is still recordable market data — it
# is simply not tradable. The comparison reads the best-price
# PROPERTIES (SF-BL-9398 live + SF-BL-9542 tradeable-price filtered),
# so a zero-size or invalid-price rung can neither fabricate a
# crossing (dead 0.40 ask in front of a live 0.62 ask vs a 0.60 bid is
# NOT crossed) nor mask one; and the sentinel fallbacks (ask 0.99 /
# bid 0.01) make empty and one-sided books read uncrossed here —
# those keep their own Gate 0.5 / Gate 0.7 reasons, which precede
# Gate 0.8. Equality (a locked book) is crossed per the prereg's
# `best_yes_ask <= best_yes_bid`: at zero spread the mirror identities
# above already collapse.

def _is_crossed_book(book: OrderBookSnapshot) -> bool:
    """True iff the book is crossed or locked (SF-BL-9365): its best
    YES ask — the first LIVE, tradeably-priced ask (SF-BL-9398/9542) —
    is at or below its best YES bid. SF-BL-6060: both ladders must be
    tradeable before the comparison — a side with no tradeable ladder
    reads its sentinel (ask 0.99 / bid 0.01), which can fabricate a
    crossing (an ask-only book at 0.005 compared against the 0.01 bid
    sentinel) as well as mask one, so a fallback-backed book is never
    crossed BY THIS PREDICATE; evaluate rejects such books earlier
    (Gate 0.5 / 0.7 / 0.7b) with their own named reasons."""
    if _fallback_price_sides(book):
        return False
    return book.best_yes_ask <= book.best_yes_bid


# --- SF-BL-9051: ladder sortedness is validated, not assumed ------------
# OrderBookSnapshot's docstring asserts yes_asks arrive sorted ascending
# and yes_bids descending, but pre-fix nothing enforced it, and every
# order-sensitive consumer trusts presentation order: the best-price
# properties return the FIRST live tradeable rung (not the extremal one),
# evaluate's edge screen reads those properties, and sweep_order_book
# walks the ladder in given order — its SF-BL-8891 parity stop and its
# marginal-edge stop are `break`s whose completeness argument is exactly
# "both ladders are in ascending price order". Verified live pre-fix
# (repro.py in the task work dir): asks [(0.62, 100k), (0.50, 100k)] /
# bids [(0.48, 100k), (0.40, 100k)] at p=0.80 reported best_yes_ask 0.62
# (clean: 0.50), swept BUY_YES at q_exec 0.6262 (clean: 0.505) and flipped
# evaluate from BUY_YES $15,000 to an incidental Gate-D SKIP; bids
# [(0.40, 100k), (0.48, 100k)] stored liquidation_bid_price 0.40 (clean:
# 0.48) on a $15,000 BUY_YES and swept BUY_NO at q_exec 0.606 (clean:
# 0.5252). Sortedness is a relation between rungs, so unlike an invalid
# price/size rung (SF-BL-9398/9542: skipped, identical-to-clean) it
# cannot be filtered rung-by-rung — silently re-sorting would rewrite
# producer data — so the book is REJECTED: ValueError at construction,
# SKIP UNSORTED_LADDER (zero size, cluster untouched) at evaluate entry
# for mutated/unpickled books, and ValueError in sweep_order_book,
# mirroring SF-BL-9363's fee_rate defence in depth. Kill-criterion audit
# (CALLER_CENSUS.md in the task work dir): NOT met — no in-repo producer
# validates sortedness (OrderBookLevel is a plain dataclass with no
# __post_init__; the 42 OrderBookSnapshot construction sites are the
# benchmark fixtures and test helpers passing bare literals; no
# CLOB/JSON deserialiser in the repo constructs snapshots), and the
# dataclass is mutable, so a constructor check alone cannot be the only
# line. The check orders only the live, tradeably-priced subsequence
# (see _ladder_order_violation): a rung no consumer can read — zero-size
# or invalid-price, SF-BL-9398/9542 — cannot mis-order anything, and
# the SF-BL-9542/9473 grids pin invalid rungs at arbitrary positions.
# Equal consecutive prices are allowed (non-decreasing asks /
# non-increasing bids). The other
# SF-BL-9051 categories were closed by their own preregs and are NOT
# re-litigated here: fee_rate at construction in [0, 0.10]
# (SF-BL-9363, a strict subset of this task's [0, 1)), invalid-price and
# zero/negative-size rungs skipped identical-to-clean (SF-BL-9398 /
# SF-BL-9542 / SF-BL-9473), with a purely-invalid book SKIPping zero-size
# in evaluate (EMPTY_BOOK / sentinel edge screen). size_walked_exit_bid
# re-sorts its ladder explicitly (SF-BL-7432) and is order-immune by
# construction, so it needs no check.

def _ladder_order_violation(
    yes_asks: List[OrderBookLevel], yes_bids: List[OrderBookLevel]
) -> Optional[str]:
    """Name of the first mis-ordered ladder (SF-BL-9051), or None.

    Sortedness is checked over the LIVE, TRADEABLY-PRICED subsequence
    (SF-BL-9398 live + SF-BL-9542 price domain) — the exact population
    every order-sensitive consumer reads. A zero-size or invalid-price
    rung is skipped by the best-price properties and the sweep wherever
    it sits, so its presentation position can alter no outcome and does
    not trip this check (the SF-BL-9542/9473 identity-to-clean grids,
    which place invalid rungs at arbitrary positions, depend on this).
    yes_asks must be non-decreasing and yes_bids non-increasing over
    that subsequence.
    """
    ask_prices = [lvl.price for lvl in yes_asks
                  if _is_live_level(lvl) and _is_tradeable_price(lvl.price)]
    if any(a > b for a, b in zip(ask_prices, ask_prices[1:])):
        return "yes_asks"
    bid_prices = [lvl.price for lvl in yes_bids
                  if _is_live_level(lvl) and _is_tradeable_price(lvl.price)]
    if any(a < b for a, b in zip(bid_prices, bid_prices[1:])):
        return "yes_bids"
    return None


def _validate_ladder_order(
    yes_asks: List[OrderBookLevel], yes_bids: List[OrderBookLevel]
) -> None:
    """Raise ValueError unless both ladders are sorted (SF-BL-9051)."""
    violation = _ladder_order_violation(yes_asks, yes_bids)
    if violation is not None:
        raise ValueError(
            f"unsorted ladder {violation} (SF-BL-9051): yes_asks must "
            "be sorted ascending and yes_bids descending by price; an "
            "unsorted ladder would make the first rung, not the best "
            "price, drive the edge screen, the sweep and the "
            "liquidation mark")


# --- SF-BL-10795: book.domain's type is never validated ----------------
# Gate 1 (CryptoBench) compares book.domain == MarketDomain.CRYPTO
# (value 'crypto'), and the Step-4 domain multiplier reads the same
# comparison, yet pre-fix no constructor or gate check required a
# MarketDomain member: the `domain: MarketDomain` annotation was
# unenforced and OrderBookSnapshot.__post_init__ validated fee_rate
# (SF-BL-9363), on_chain_verified (SF-BL-10793), ladder order,
# est_days (SF-BL-9364) and snapshot_ts — never domain. MarketDomain
# is a str Enum, so equality cannot police the type: the exact-value
# plain string 'crypto' equals CRYPTO and arms the gate only by
# accident, while 'CRYPTO', 'Crypto' and 'crypto ' (trailing space)
# compare unequal and disarm BOTH Gate 1 and the crypto dampener.
# Verified live pre-fix (repro.py in the task work dir): on the
# prereg fixture (ask 0.50 / bid 0.48, p=0.80, fee 0.01, equity
# $10,000) OrderBookSnapshot(domain='CRYPTO', on_chain_verified=
# False) EXECUTED BUY_YES at the full $1,500.00 cluster cap instead
# of SKIPping FAILED_CRYPTOBENCH_ONCHAIN_GATE — the genuine verified
# CRYPTO book trades $1,042.93, so the string book got the gate off
# AND undampened sizing; 'Crypto' / 'crypto ' / '' / None / 123
# traded identically, by direct construction and by mutation alike.
# Kill-criterion audit (PRODUCER_AUDIT.md in the task work dir): NOT
# met — 111 in-repo construction sites pass enum literals or bare
# pass-through variables (domain=domain / book.domain) with no
# validated step, NO production deserialiser constructs snapshots
# from JSON/CSV at all (the single MarketDomain(d["domain"])
# conversion is a test remediation for a JSON book whose domain is
# a plain string), and the dataclass is mutable (mutation /
# unpickling bypass __post_init__; the repo's own tests assign
# .domain post-construction). The value is therefore validated at
# construction (raise, as for fee_rate) and re-validated at the
# consumption gate in evaluate_and_size_opportunity (Gate ED, SKIP
# INVALID_DOMAIN, before Gate 1), which covers mutated/unpickled
# books. Membership means isinstance(v, MarketDomain) — the
# SF-BL-10823 discipline for TradeAction: a plain string is never
# coerced or canonicalised here ('CRYPTO' is never folded to
# CRYPTO, 'crypto' is never promoted to a member); a producer
# holding a serialized label must convert with MarketDomain(...)
# before construction and hears so from the error. The check is
# unconditional (not scoped to CRYPTO): a book whose domain is not
# a member has no honest gate, dampener or telemetry identity on
# any path. Genuine members pass every layer bit-identically.
# TradeExecutionResult.domain is deliberately NOT validated: every
# evaluate SKIP record copies book.domain verbatim for audit (as
# for est_days, SF-BL-11534), so the record must stay constructible
# carrying the offending value.

def _is_valid_market_domain(value: object) -> bool:
    """True iff value is a genuine MarketDomain member (SF-BL-10795).

    isinstance, not equality: MarketDomain subclasses str, so the
    plain string 'crypto' compares equal to MarketDomain.CRYPTO
    yet is not a member and must not arm Gate 1 or the dampener.
    """
    return isinstance(value, MarketDomain)


def _validate_market_domain(value: object) -> None:
    """Raise ValueError unless domain is a MarketDomain member."""
    if not _is_valid_market_domain(value):
        raise ValueError(
            f"INVALID_DOMAIN: domain {value!r} "
            f"(type {type(value).__name__}) is not a MarketDomain "
            "member (SF-BL-10795): Gate 1 compares book.domain == "
            "MarketDomain.CRYPTO (value 'crypto'), so a plain-string "
            "domain such as 'CRYPTO', 'Crypto' or 'crypto ' from a "
            "JSON/CSV/env producer never arms the CryptoBench gate "
            "(or the crypto Kelly dampener) and an unverified book "
            "trades instead of SKIPping "
            "FAILED_CRYPTOBENCH_ONCHAIN_GATE, while the exact-value "
            "string 'crypto' arms it only by str-Enum equality "
            "accident; convert serialized labels with "
            "MarketDomain(...) before construction — values are "
            "never coerced or canonicalised here")


def _validate_snapshot_level_types(
    yes_asks: object, yes_bids: object
) -> None:
    """Raise ValueError if any OrderBookLevel entry in either
    ladder carries a bool / non-numeric price or size_shares
    (SF-BL-11492). Runs in OrderBookSnapshot.__post_init__ so a
    level that bypassed its own construction (object.__setattr__
    / __dict__ / unpickling) is still rejected when placed in a
    freshly constructed snapshot. Non-OrderBookLevel entries and
    non-list ladders are skipped untouched: container/entry shape
    is SF-BL-10826's surface, not this task's."""
    for ladder in (yes_asks, yes_bids):
        if not isinstance(ladder, (list, tuple)):
            continue
        for entry in ladder:
            if isinstance(entry, OrderBookLevel):
                _validate_order_book_level(entry.price, entry.size_shares)


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
    fee_rate: float = 0.01            # Exchange + gas fee rate (e.g., 1.0%); must be in [0, 0.10]
    on_chain_verified: bool = True    # Required for MarketDomain.CRYPTO (CryptoBench gate)
    # SF-BL-10767: when this snapshot was observed, in epoch seconds —
    # the freshness provenance every mark taken from the book inherits
    # (stamped onto TradeExecutionResult.liquidation_bid_as_of). None
    # (the default, and every legacy producer) means the observation
    # time is unknown: marks from such a book are unprovenanced and
    # are flagged / excluded at scoring per the SF-BL-10767 rule.
    snapshot_ts: Optional[float] = None

    def __post_init__(self) -> None:
        # SF-BL-11492: reject a bool / non-numeric level field at
        # construction, before any best-price property can filter
        # the rung silently as non-live. First, so a type-invalid
        # level is named as such even if another field is also bad.
        # OrderBookLevel's own __setattr__/__post_init__ already
        # reject normally-constructed levels; this pass catches
        # levels that bypassed their own construction.
        _validate_snapshot_level_types(self.yes_asks, self.yes_bids)
        # SF-BL-10795: reject a non-MarketDomain domain at construction,
        # before equality with MarketDomain.CRYPTO can disarm (or
        # accidentally arm) Gate 1. The field's type contract is
        # unconditional (as for fee_rate); evaluate re-validates at
        # Gate ED for mutated/unpickled books, which bypass this.
        _validate_market_domain(self.domain)
        # SF-BL-9363: reject an out-of-bound fee at construction, before it
        # can reach any q_exec computation.
        _validate_fee_rate(self.fee_rate)
        # SF-BL-10793: reject a non-bool verification flag at construction,
        # before truthiness can arm Gate 1. The field's type contract is
        # domain-independent (as for fee_rate); evaluate re-validates the
        # CRYPTO gate path for mutated/unpickled books, which bypass this.
        _validate_on_chain_verified(self.on_chain_verified)
        # SF-BL-9051: reject an unsorted ladder at construction, before
        # its first rung can be read as the best price by any gate.
        _validate_ladder_order(self.yes_asks, self.yes_bids)
        # SF-BL-9364: reject a non-positive / non-finite lockup at
        # construction, before it can be copied into a trade record or
        # pooled into avg_lockup; evaluate re-validates at Gate 0.1
        # for mutated/unpickled books, which bypass this.
        _validate_est_days_to_resolution(self.est_days_to_resolution)
        # SF-BL-10767: reject a non-finite / negative observation time
        # at construction, before it can be stamped onto a trade as
        # freshness provenance. None (unknown) is valid and stays the
        # default; scoring reads mutated/unpickled books defensively
        # via _snapshot_ts, which treats an invalid stamp as unknown.
        _validate_snapshot_ts(self.snapshot_ts)

    @property
    def best_yes_ask(self) -> float:
        # SF-BL-9398 + SF-BL-9542: first LIVE ask carrying a tradeable
        # price (finite, in (0, 1)); zero/negative-size levels and
        # invalid-price levels are skipped, and the 0.99 sentinel now
        # also covers an all-zero-size or all-invalid-price ask ladder.
        return _first_tradeable_price(self.yes_asks, 0.99)

    @property
    def best_yes_bid(self) -> float:
        # SF-BL-9398 + SF-BL-9542: first LIVE bid carrying a tradeable
        # price; sentinel 0.01 as for best_yes_ask. The derived NO prices
        # below inherit the bound: 1 - (a price in (0, 1) or a sentinel)
        # always lands in (0, 1) / on the opposite sentinel.
        return _first_tradeable_price(self.yes_bids, 0.01)

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


# --- SF-BL-10823: TradeExecutionResult.action is never validated -------
# TradeExecutionResult was a plain mutable dataclass with NO
# __post_init__ and no assignment guard, so its `action` field accepted
# any object. settle_or_mark_trade's won test
# (action == BUY_YES / BUY_NO) then matched NEITHER side for a
# non-canonical action and booked gross payout 0.0, while
# evaluate_fleet_north_star's executed test (action != SKIP) counted
# the same record as executed with its PnL in CWR / MDD. Verified live
# pre-fix (repro.py in the task work dir; $500 invested / 1000 shares,
# YES resolves): action='buy_yes', 'MAYBE' and None each settled the
# YES win as a full -$500.00 loss (genuine BUY_YES win: +$500.00) and
# each scored executed=1, CWR -1.0, N_star -2.2226. TradeAction is a
# str Enum, so equality cannot police the domain — 'BUY_YES' ==
# TradeAction.BUY_YES is True for a plain string (a plain 'SKIP'
# string with invested > 0 would even partition as skipped) — only an
# isinstance check distinguishes a genuine member. Kill-criterion
# audit (PRODUCER_AUDIT.md in the task work dir): NOT met — the naive
# fleet in run_benchmark_simulation and the test suites construct
# TradeExecutionResult directly outside evaluate_and_size_opportunity,
# the dataclass is mutable (post-construction assignment and
# unpickling / __dict__ writes bypass construction), a JSON round-trip
# degrades action to a plain string with no validated path back, and
# the TradeAction annotation is unenforced. The value is therefore
# validated at construction AND assignment on the dataclass itself,
# and re-validated at both consumption gates — settle_or_mark_trade
# entry and evaluate_fleet_north_star entry — so a record that
# bypassed construction (object.__setattr__ / __dict__ / unpickle)
# still cannot settle or be scored as executed. Genuine BUY_YES /
# BUY_NO / SKIP members pass every layer bit-identically.

def _is_valid_trade_action(value: object) -> bool:
    """True iff value is a genuine TradeAction member (SF-BL-10823).

    isinstance, not equality: TradeAction subclasses str, so the plain
    string 'BUY_YES' compares equal to TradeAction.BUY_YES yet is not
    a member and must not settle or score as one.
    """
    return isinstance(value, TradeAction)


def _validate_trade_action(value: object) -> None:
    """Raise ValueError unless action is a TradeAction member."""
    if not _is_valid_trade_action(value):
        raise ValueError(
            f"INVALID_TRADE_ACTION: action {value!r} "
            f"(type {type(value).__name__}) is not a TradeAction "
            "member (SF-BL-10823): settle_or_mark_trade's won test and "
            "evaluate_fleet_north_star's executed test are equality "
            "tests, so a non-canonical action such as 'buy_yes', "
            "'MAYBE' or None matches neither BUY_YES nor BUY_NO at "
            "settlement (booking a YES win as a full loss) yet still "
            "counts as an executed trade in CWR/MDD; only a genuine "
            "TradeAction.BUY_YES / BUY_NO / SKIP member may settle "
            "or be scored")


def _validate_trade_actions(trades: List[TradeExecutionResult]) -> None:
    """Raise ValueError on any trade whose action is not a TradeAction
    member (SF-BL-10823 scorer gate). Runs over EVERY trade, before the
    executed/skipped partition: a non-member is by definition not SKIP,
    so the SF-BL-9475-style 'skip SKIP trades' scoping cannot apply —
    an unvalidated action is exactly what the partition must never see.
    """
    for t in trades:
        if not _is_valid_trade_action(t.action):
            raise ValueError(
                f"INVALID_TRADE_ACTION: evaluate_fleet_north_star "
                f"trade {t.market_id!r} has action={t.action!r} "
                f"(type {type(t.action).__name__}), which is not a "
                "TradeAction member (SF-BL-10823): the scorer's "
                "executed test (action != SKIP) would count this "
                "record as executed and sum its PnL into CWR/MDD "
                "even though no canonical side can have won it")


# --- SF-BL-34997: a trade's action flipped between evaluate and settle ---
# SF-BL-10823 validated that `action` is a genuine TradeAction member,
# at construction, on assignment, and at the settle/scorer gates — but
# a member-to-member FLIP passes every one of those layers, because the
# member is genuine. Yet exec_price, shares_acquired and the stored
# liquidation_bid_price are all built from the side chosen at entry:
# the sweep walks that side's ask ladder and the stored bid is that
# side's bid (_declared_liquidation_bid). Verified live pre-fix
# (repro.py in the task work dir; standard book ask 0.50 / bid 0.48,
# fee 0.01, p=0.65, equity $10,000): evaluate built a BUY_YES position
# ($1,025.25 invested, 2,030.20 shares at q_exec 0.505, stored YES bid
# 0.48); assigning action=BUY_NO — a genuine member, so the SF-BL-10823
# assignment guard passes — and settling at outcome_yes=0 then booked
# +$1,004.95 as a NO win on YES-priced shares (the honest YES result is
# -$1,025.25), an open legacy re-mark priced the position at the stored
# YES bid 0.48 while claiming BUY_NO (whose bid is 0.50), an exit-book
# re-mark walked the NO ladder and rewrote the stored bid to 0.50 for
# a YES position, and flipping to SKIP silently erased the position
# through settle's SKIP early return. The mirror flip (BUY_NO ->
# BUY_YES, settle at outcome_yes=1) booked +$1,356.05 identically.
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT
# met — no tested invariant pins action after evaluate; the agreed
# SF-BL-10823 suite itself pins member-to-member reassignment as
# ACCEPTED behaviour (test_assignment_accepts_genuine_members assigns
# BUY_NO then SKIP to a BUY_YES record), and the SF-BL-10767 suite
# reassigns action before scoring, so `action` cannot simply be made
# immutable in __setattr__ without breaking agreed contracts.
# The record therefore pins its ENTRY side separately: the new
# TradeExecutionResult.entry_action field is set from `action` at
# construction (__post_init__), may not be changed to a different
# member through the normal assignment path (__setattr__), and is
# excluded from repr/equality (compare=False) so genuine records stay
# bit-identical. Consistency is then re-validated at both consumption
# gates — settle_or_mark_trade entry (unscoped: before the SKIP /
# zero-invested early return, whose tests a flipped record would
# otherwise pass) and evaluate_fleet_north_star entry (scoped to
# executed, i.e. current-action non-SKIP, records — the only records
# whose side the scorer reads, the SF-BL-11536 scoping precedent; a
# BUY_YES record reassigned to SKIP for scoring, as the SF-BL-10767
# suite does, partitions as skipped and its PnL is never summed).
# A record with entry_action=None (a bypass-constructed record that
# never carried the field — object.__new__ fixtures, legacy pickles)
# is unpinned: no entry side was ever recorded for it, so there is
# nothing to mismatch against and it settles/scores exactly as
# pre-fix. Writes that bypass the dataclass guard
# (object.__setattr__ / __dict__ on `action` alone) leave the pin
# intact and are caught at the gates; a caller that rewrites BOTH
# fields through bypass writes has re-authored the record, which no
# in-record check can distinguish from a genuine record of that side.

def _is_valid_entry_action(value: object) -> bool:
    """True iff value is None (unpinned) or a genuine TradeAction
    member (SF-BL-34997)."""
    return value is None or isinstance(value, TradeAction)


def _validate_entry_action(value: object) -> None:
    """Raise ValueError unless entry_action is None or a TradeAction
    member (SF-BL-34997)."""
    if not _is_valid_entry_action(value):
        raise ValueError(
            f"INVALID_ENTRY_ACTION: entry_action {value!r} "
            f"(type {type(value).__name__}) is neither None nor a "
            "TradeAction member (SF-BL-34997): the entry-side pin "
            "records the side a position's exec_price / shares / "
            "stored bid were built from, so it must be a genuine "
            "TradeAction.BUY_YES / BUY_NO / SKIP member (or None "
            "before the pin is set at construction)")


def _action_mismatch_error(trade: TradeExecutionResult) -> ValueError:
    """The named ValueError for a flipped trade (SF-BL-34997)."""
    entry = getattr(trade, "entry_action", None)
    return ValueError(
        f"MISMATCHED_TRADE_ACTION: trade {trade.market_id!r} has "
        f"action={trade.action!r} but its entry side, pinned at "
        f"construction, is entry_action={entry!r} (SF-BL-34997): "
        "exec_price, shares_acquired and liquidation_bid_price were "
        "built from the entry side's sweep and bid, so settling or "
        "marking under a different side pays the wrong side — the "
        "prereg fixture (BUY_YES entry: $1,025.25 invested, 2,030.20 "
        "shares, stored YES bid 0.48) flipped to BUY_NO and settled "
        "at outcome_yes=0 booked +$1,004.95 as a NO win the position "
        "never held; a flipped trade must be rejected, never settled "
        "or marked under its new action")


def _validate_trade_action_consistency(trade: TradeExecutionResult) -> None:
    """Raise ValueError if the trade's current action no longer
    matches the entry side pinned at construction (SF-BL-34997).

    No-op when the record is unpinned (entry_action missing or
    None — a bypass-constructed / legacy record that never recorded
    an entry side). A bypass-written non-member pin is itself a
    mismatch and raises.
    """
    entry = getattr(trade, "entry_action", None)
    if entry is None:
        return
    if not _is_valid_trade_action(entry) or entry != trade.action:
        raise _action_mismatch_error(trade)


def _validate_scored_trade_action_consistencies(
    trades: List[TradeExecutionResult],
) -> None:
    """Raise ValueError on any executed trade whose action no longer
    matches its pinned entry side (SF-BL-34997 scorer gate).

    Scope mirrors SF-BL-11536: non-SKIP (by CURRENT action) trades
    only — a record whose current action is SKIP enters neither the
    executed sums nor the settled/open partitions, so its side is
    never read by the scorer (the SF-BL-10767 suite reassigns a
    BUY_YES record to SKIP before scoring it as a skip). Runs after
    the SF-BL-10823 action gate, so every current action here is a
    genuine member.
    """
    for t in trades:
        if t.action == TradeAction.SKIP:
            continue
        entry = getattr(t, "entry_action", None)
        if entry is None:
            continue
        if not _is_valid_trade_action(entry) or entry != t.action:
            raise _action_mismatch_error(t)


# --- SF-BL-11536: TradeExecutionResult.is_settled is never validated ---
# TradeExecutionResult carried NO validation of its `is_settled` field:
# the dataclass's __setattr__ / __post_init__ policed only `action`
# (SF-BL-10823), and evaluate_fleet_north_star reads the field by bare
# truthiness in two places — the settled partition
# (settled_trades = [t for t in scored if t.is_settled], whose length
# is the settlement win-rate denominator and whose members' PnL sign
# is the numerator) and the SF-BL-10767 open partition
# (_open_executed = [t for t in executed if not t.is_settled], which
# drives the stale/unprovenanced mark counts). Verified live pre-fix
# (repro.py in the task work dir; two-trade book: settled winner
# +$100, open trade with a -$100 mark): the open trade constructed
# directly with is_settled='False' — the string a JSON/CSV producer
# emits — is TRUTHY, so it entered the settled denominator and halved
# the settlement win rate to 50.0 against 100.0 for the identical book
# with is_settled=False; 'True', 1 and [False] counted as settled the
# same way, while falsy non-bools (0, '', None, []) read as open —
# silently the right partition for the wrong reason, still not the
# bool the field declares. SF-BL-10822 fixtures the is_settled
# ARGUMENT of settle_or_mark_trade, but a result constructed or
# mutated directly (the benchmark's naive fleet does exactly this)
# bypasses that path entirely: settle never even reads the stored
# field, it overwrites it from its parameter. Kill-criterion audit
# (CALLER_CENSUS.md in the task work dir): NOT met — is_settled is
# NOT only ever written by settle_or_mark_trade: the dataclass
# default (True) and explicit construction kwargs write it (the test
# suites and the wave3 recompute construct records with is_settled=
# directly), the repo's own tests assign it after construction
# (bid_freshness / float_outcome suites), fleet analysis scripts
# assign it, and mutation / unpickling bypass construction in any
# case. The value is therefore validated at construction AND
# assignment on the dataclass itself (strict bool, via
# _is_strict_bool — int 1/0 are NOT bools, per SF-BL-10793), and
# re-validated at the scorer's consumption gate
# (_validate_trade_is_settleds) so a record that bypassed
# construction (object.__setattr__ / __dict__ / unpickling) still
# cannot enter the settled denominator. Scorer-gate scope is the
# prereg's own: EXECUTED (non-SKIP) trades — the only records whose
# is_settled the scorer ever reads (SKIP records enter neither the
# settled nor the open partition; cf. the SF-BL-9475/9364/11304
# scoping). The dataclass gate is unscoped: no record of any action
# may carry a non-bool is_settled. Genuine True/False books pass
# every layer bit-identically. Side effect on the write path:
# settle_or_mark_trade stores its parameter through the dataclass
# __setattr__, and is_settled is the FIRST field written on every
# settle/mark path, so a non-bool argument now raises this named
# error at that assignment with every other field untouched
# (SF-BL-10724 ordering preserved); the parameter's own entry gate
# remains SF-BL-10822's surface.

def _validate_is_settled(value: object) -> None:
    """Raise ValueError unless is_settled is a strict bool
    (SF-BL-11536)."""
    if not _is_strict_bool(value):
        raise ValueError(
            f"INVALID_IS_SETTLED: is_settled {value!r} "
            f"(type {type(value).__name__}) is not a bool "
            "(SF-BL-11536): evaluate_fleet_north_star partitions "
            "executed trades by the truthiness of this field, so a "
            "non-bool such as the string 'False' from a JSON/CSV "
            "producer is truthy and counts an open trade as settled "
            "(prereg fixture: the string-marked open trade halved "
            "the settlement win rate to 50.0 against 100.0 for the "
            "identical bool book), while falsy non-bools (0, '', "
            "None) masquerade as open; only a genuine bool True / "
            "False may mark a trade settled or open")


def _validate_trade_is_settleds(trades: List[TradeExecutionResult]) -> None:
    """Raise ValueError on any executed trade whose is_settled is not
    a strict bool (SF-BL-11536 scorer gate).

    Scope mirrors SF-BL-9475: non-SKIP trades only — a SKIP record's
    is_settled is read by neither scorer partition, so it cannot
    enter the settled denominator this gate protects. Runs before
    the executed/skipped partition and before the SF-BL-10767 open
    partition can read the field.
    """
    for t in trades:
        if t.action == TradeAction.SKIP:
            continue
        if not _is_strict_bool(t.is_settled):
            raise ValueError(
                f"INVALID_IS_SETTLED: evaluate_fleet_north_star "
                f"trade {t.market_id!r} has is_settled={t.is_settled!r} "
                f"(type {type(t.is_settled).__name__}), which is not "
                "a bool (SF-BL-11536): the scorer's settled partition "
                "(t.is_settled) and open partition "
                "(not t.is_settled) are truthiness tests, so a "
                "truthy non-bool such as 'False' would enter the "
                "settlement win-rate denominator as a settled trade "
                "and a falsy non-bool would masquerade as open")


# --- SF-BL-35035: cluster_exposure_released has no assignment guard ---
# TradeExecutionResult.__setattr__ / __post_init__ policed only
# `action` (SF-BL-10823) and `is_settled` (SF-BL-11536), so the
# SF-BL-8890 idempotence flag for the cluster-cap release was freely
# rewritable. Verified live pre-fix (repro.py in the task work dir):
# with ledger {'C1': 0.20} (0.10 from the fixture trade plus 0.10
# from a co-tenant), release_cluster_exposure returned 0.10 and left
# 0.10, then after the unguarded assignment
# trade.cluster_exposure_released=False a second call returned
# another 0.10 and left {'C1': 0.0} — the co-tenant's live exposure
# erased, freeing a full cluster slot the cap never granted;
# conversely assigning the string 'yes' was accepted (truthy) and
# blocked release entirely (0.0 returned, ledger still charged),
# while falsy non-bools (0, '', None) masqueraded as unreleased.
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT
# met — the flag is NOT only ever written by
# release_cluster_exposure under a tested invariant: the dataclass
# default and explicit construction kwargs write it, the dataclass
# is mutable with no guard (any caller may assign it, and mutation /
# object.__setattr__ / __dict__ / unpickling bypass construction in
# any case), and no test pinned its type or its latch semantics.
# The flag is therefore guarded at three layers:
#   (1) DATACLASS: strict-bool validation at construction AND
#       assignment (named INVALID_CLUSTER_EXPOSURE_RELEASED
#       ValueError on any non-bool), plus a MONOTONIC LATCH: once
#       True, a normal assignment of False raises the named
#       CLUSTER_EXPOSURE_RELEASED_RESET ValueError and leaves the
#       stored True intact — a released trade has no legitimate
#       un-released state to return to.
#   (2) RELEASE CONSUMPTION GATE: release_cluster_exposure
#       re-validates the flag's type at entry (so a record that
#       bypassed the dataclass via object.__setattr__ / __dict__ /
#       unpickling with a non-bool flag raises the named error
#       before any ledger is touched), and settle_or_mark_trade
#       pre-validates the same condition at the head of its settled
#       branch, before any field is written (SF-BL-10724 ordering).
#   (3) RELEASE LATCH (shadow): a successful release also sets a
#       private latch via object.__setattr__ (_sf35035_release_
#       latch, deliberately NOT a dataclass field and NOT set at
#       construction, so a fresh record's __dict__ is unchanged).
#       release_cluster_exposure checks the latch before the public
#       flag, so even a bypass reset of the public flag
#       (object.__setattr__ / __dict__ write of False) cannot
#       produce a second subtraction: the call no-ops at 0.0 and
#       the public flag is restored to True. Genuine first releases
#       are bit-identical: the latch is written only after the
#       SF-BL-8890 / SF-BL-10762 subtraction has succeeded, exactly
#       where the public flag was already set True pre-fix.

def _validate_cluster_exposure_released(value: object) -> None:
    """Raise ValueError unless cluster_exposure_released is a
    strict bool (SF-BL-35035)."""
    if not _is_strict_bool(value):
        raise ValueError(
            f"INVALID_CLUSTER_EXPOSURE_RELEASED: "
            f"cluster_exposure_released {value!r} "
            f"(type {type(value).__name__}) is not a bool "
            "(SF-BL-35035): release_cluster_exposure reads this "
            "flag by truthiness as its idempotence guard, so a "
            "non-bool such as the string 'yes' from a JSON/CSV "
            "producer is truthy and blocks a release that never "
            "happened (prereg fixture: 0.0 returned, ledger still "
            "charged), while falsy non-bools (0, '', None) "
            "masquerade as unreleased; only a genuine bool True / "
            "False may mark a trade's cluster exposure released")


def _validate_cluster_exposure_released_reset(
        existing: object, latch: object) -> None:
    """Raise ValueError on a True -> False reset of the release
    flag (SF-BL-35035 monotonic latch)."""
    raise ValueError(
        "CLUSTER_EXPOSURE_RELEASED_RESET: "
        "cluster_exposure_released cannot be reset from True to "
        "False (SF-BL-35035): the flag is the idempotence latch "
        "for the SF-BL-8890 cluster-cap release — resetting it "
        "after a successful release lets a second "
        "release_cluster_exposure call subtract the trade's "
        "fraction again and erase a co-tenant's live exposure "
        "(prereg fixture: ledger {'C1': 0.20} fell to 0.0 instead "
        "of staying at the co-tenant's 0.10); a released trade "
        "stays released")


def _cluster_release_latch(trade: object) -> bool:
    """True iff this trade's release latch was set by a successful
    release (SF-BL-35035 shadow latch; bypass-reset defence)."""
    try:
        return object.__getattribute__(
            trade, "_sf35035_release_latch") is True
    except AttributeError:
        return False


def _set_cluster_release_latch(trade: object) -> None:
    """Set the SF-BL-35035 shadow release latch, bypassing the
    dataclass __setattr__ (it is bookkeeping, not a field)."""
    object.__setattr__(trade, "_sf35035_release_latch", True)


def _validate_trade_cluster_exposure_released(
        trade: TradeExecutionResult) -> None:
    """Raise ValueError if a trade's stored release flag is not a
    strict bool (SF-BL-35035 consumption gate for records that
    bypassed construction/assignment)."""
    value = getattr(trade, "cluster_exposure_released", None)
    if not _is_strict_bool(value):
        raise ValueError(
            f"INVALID_CLUSTER_EXPOSURE_RELEASED: trade "
            f"{getattr(trade, 'market_id', None)!r} has "
            f"cluster_exposure_released={value!r} "
            f"(type {type(value).__name__}), which is not a bool "
            "(SF-BL-35035): release_cluster_exposure reads this "
            "flag by truthiness as its idempotence guard, so a "
            "truthy non-bool such as 'yes' would block a release "
            "that never happened and a falsy non-bool would "
            "masquerade as unreleased")


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
    kelly_fraction_raw: float        # Uncapped, UNSCALED full Kelly f* = (p_side - q_top) / (1 - q_top) — DIAGNOSTIC ONLY (SF-BL-9327): never size off this field; it exceeds the sized fraction by 1/lambda (2.86x at lambda 0.35) until the cluster cap binds. Size off kelly_fraction_sized (pre-sweep target) or allocated_fraction (post-sweep realised).
    allocated_fraction: float        # Final portfolio fraction after Kelly/cluster/slippage caps
    invested_capital: float          # Actual dollars I_k swept into CLOB
    shares_acquired: float           # Total contracts S_k acquired
    est_days_to_resolution: float
    skip_reason: Optional[str] = None
    # Settlement / Mark-to-Market fields
    is_settled: bool = True
    outcome_yes: Optional[int] = None  # 1 if YES resolved, 0 if NO resolved
    net_pnl_bid_marked: float = 0.0    # Realized PnL (if settled) or bid-marked unrealized PnL
    # SF-BL-3660: audit fields for the bid-ladder VWAP mark. Set only when
    # settle_or_mark_trade is given an exit_book: the filled-portion bid
    # VWAP and the fraction of shares_acquired the bid ladder could absorb.
    # None on every legacy (single top-bid) mark and on settled trades.
    liquidation_mark_vwap: Optional[float] = None
    liquidation_fill_fraction: Optional[float] = None
    # SF-BL-10767: freshness provenance of liquidation_bid_price — when
    # the bid now stored in that field was observed (epoch seconds) and
    # which book it came from (BID_SOURCE_ENTRY_BOOK at entry, re-stamped
    # BID_SOURCE_EXIT_BOOK by an exit-book re-mark in
    # settle_or_mark_trade). None/None (the default: every directly
    # constructed or legacy record) means the bid's provenance is
    # unknown; evaluate_fleet_north_star counts such open marks in
    # unprovenanced_mark_count and, when scored_at is given, excludes
    # them as stale. The settled branch never reads the bid, so settled
    # trades are exempt from the staleness rule entirely.
    liquidation_bid_as_of: Optional[float] = None
    liquidation_bid_source: Optional[str] = None
    # SF-BL-9327: dual-report of the SIZED Kelly fraction — the
    # lambda-scaled, crypto-dampened, cluster-headroom-capped target
    # fraction the Step-5 sweep was budgeted at (evaluate's
    # capped_fraction). This is the same quantity nstar.kelly_fraction()
    # returns and the CLI reports as kelly_fraction_of_account for
    # identical (p, q, lambda, cluster) inputs, so a consumer can size
    # off any surface and get the same number. 0.0 on every SKIP
    # (EC-09, like kelly_fraction_raw) and on manually constructed /
    # naive-baseline trades, which do not Kelly-size. It is the
    # PRE-sweep target: allocated_fraction remains the post-sweep
    # realised fraction and is <= this value when book depth binds.
    kelly_fraction_sized: float = 0.0
    # SF-BL-8890: cluster-cap lifecycle bookkeeping. evaluate_and_size_
    # opportunity stores a reference to the caller's cluster_exposure_used
    # dict on every executed trade, so settle_or_mark_trade can release
    # exactly the allocated_fraction it added when the trade settles —
    # even when the caller uses the legacy settle signature with no dict
    # argument. cluster_exposure_released makes that release idempotent:
    # a trade frees its fraction at most once, however often it is
    # re-settled. Both fields are bookkeeping only (excluded from repr /
    # equality); manually constructed trades carry None / False and
    # settle exactly as before unless a dict is passed explicitly.
    cluster_exposure_used: Optional[Dict[str, float]] = field(
        default=None, repr=False, compare=False)
    cluster_exposure_released: bool = field(
        default=False, repr=False, compare=False)
    # SF-BL-34952: the cluster key evaluate charged at entry, snap-
    # shotted at construction (evaluate passes it explicitly on the
    # executed return; __post_init__ snapshots factor_cluster when it
    # is left None). The settlement release resolves its ledger key
    # through this snapshot (see _entry_factor_cluster), never through
    # the mutable factor_cluster field: renaming factor_cluster
    # between evaluate and settle cannot redirect the release to a
    # different key (pre-fix the release wrote <new key>=0.0, consumed
    # the released flag, and stranded the entry key's charge forever).
    # Bookkeeping only (excluded from repr / equality), like the two
    # SF-BL-8890 fields above.
    entry_factor_cluster: Optional[str] = field(
        default=None, repr=False, compare=False)
    # SF-BL-35212: the cluster ledger DICT evaluate charged at entry,
    # snapshotted at construction (evaluate passes it explicitly on
    # the executed return; __post_init__ snapshots
    # cluster_exposure_used when it is left None and the stored
    # reference is a dict). The settlement gates validate the mutable
    # cluster_exposure_used field against this snapshot (see
    # _validate_stored_cluster_ledger): replacing the stored reference
    # with a non-dict crashes the release after PnL is booked, and
    # replacing it with a different dict silently redirects the
    # release — both are rejected by name before any PnL is booked.
    # Bookkeeping only (excluded from repr / equality), like the
    # SF-BL-8890 fields above. None when no dict was stored at
    # construction (manually constructed / SKIP records, or a ledger
    # adopted by later assignment): such records carry no entry
    # snapshot and their stored reference is taken at face value
    # (type check still applies), exactly as pre-fix.
    entry_cluster_ledger: Optional[Dict[str, float]] = field(
        default=None, repr=False, compare=False)
    # SF-BL-34997: the ENTRY side — the side this record's
    # exec_price / shares_acquired / liquidation_bid_price were built
    # from — snapshotted at construction (__post_init__ pins it from
    # `action` when it is left None, exactly the SF-BL-34952 pattern
    # for entry_factor_cluster). Once pinned it cannot be reassigned
    # to a different member through the normal assignment path
    # (__setattr__), and settle_or_mark_trade / the scorer reject a
    # record whose current `action` no longer matches it (see the
    # SF-BL-34997 block above this class). None only transiently
    # (before __post_init__) or on bypass-constructed records that
    # never carried the field; such unpinned records settle/score
    # exactly as pre-fix. Bookkeeping only (excluded from repr /
    # equality), like the SF-BL-8890 / SF-BL-34952 fields above.
    entry_action: Optional[TradeAction] = field(
        default=None, repr=False, compare=False)

    # SF-BL-10823: the action field's domain is enforced on the record
    # itself — __setattr__ intercepts every assignment (the dataclass
    # __init__ assigns through it, so construction is covered too) and
    # __post_init__ re-checks the constructed value. Writes that bypass
    # both (object.__setattr__, direct __dict__ writes, unpickling)
    # are caught at the settle/scorer entry gates instead.
    # SF-BL-11536: the is_settled field's strict-bool domain is
    # enforced the same way (construction + assignment here, scorer
    # entry gate for bypass-constructed records).
    # SF-BL-35035: the cluster_exposure_released field's strict-bool
    # domain is enforced the same way, PLUS its monotonic latch:
    # once True (or once the shadow release latch is set), assigning
    # False raises the named reset error and leaves True intact.
    # Bypass-constructed records are caught at the release/settle
    # consumption gates (see the SF-BL-35035 block above).
    # SF-BL-34997: the entry_action pin is enforced the same way —
    # its type (None or a genuine member) on every assignment, and
    # its immutability once pinned: reassigning entry_action to a
    # DIFFERENT member through the normal path raises ENTRY_ACTION_
    # PINNED, so the pin cannot be re-authored to match a flipped
    # action. `action` itself remains freely reassignable to other
    # genuine members (the SF-BL-10823 agreed contract); a flip is
    # caught against this pin at the settle/scorer gates.
    def __setattr__(self, name: str, value: object) -> None:
        if name == "action":
            _validate_trade_action(value)
        if name == "is_settled":
            _validate_is_settled(value)
        if name == "entry_action":
            _validate_entry_action(value)
            _existing_entry = self.__dict__.get("entry_action", None)
            if _existing_entry is not None and value != _existing_entry:
                raise ValueError(
                    f"ENTRY_ACTION_PINNED: entry_action is already "
                    f"pinned to {_existing_entry!r} and cannot be "
                    f"reassigned to {value!r} (SF-BL-34997): the pin "
                    "records the side the position's exec_price / "
                    "shares / stored bid were built from at entry; "
                    "reassigning it would re-author the record to "
                    "match a flipped action and defeat the settle / "
                    "scorer mismatch gates")
        if name == "cluster_exposure_released":
            _validate_cluster_exposure_released(value)
            if value is False:
                try:
                    _existing = object.__getattribute__(
                        self, "cluster_exposure_released")
                except AttributeError:
                    _existing = None
                if _existing is True or _cluster_release_latch(self):
                    _validate_cluster_exposure_released_reset(
                        _existing, True)
        super().__setattr__(name, value)

    def __post_init__(self) -> None:
        _validate_trade_action(self.action)
        _validate_is_settled(self.is_settled)
        _validate_cluster_exposure_released(
            self.cluster_exposure_released)
        # SF-BL-34952: snapshot the entry cluster key. An explicit
        # entry_factor_cluster (evaluate's executed return) is kept;
        # otherwise the construction-time factor_cluster IS the entry
        # key (manually constructed trades included). Written via
        # object.__setattr__ — it is bookkeeping, set once here.
        if self.entry_factor_cluster is None:
            object.__setattr__(
                self, "entry_factor_cluster", self.factor_cluster)
        # SF-BL-35212: snapshot the entry cluster ledger. An explicit
        # entry_cluster_ledger (evaluate's executed return) is kept;
        # otherwise a dict stored at construction IS the entry ledger
        # (manually constructed trades included). A non-dict stored
        # reference snapshots nothing — the settle/release gates
        # reject it by name. Written via object.__setattr__ — it is
        # bookkeeping, set once here, and later assignments to the
        # mutable cluster_exposure_used field never rewrite it.
        if self.entry_cluster_ledger is None and isinstance(
                self.cluster_exposure_used, dict):
            object.__setattr__(
                self, "entry_cluster_ledger", self.cluster_exposure_used)
        # SF-BL-34997: pin the entry side. An explicit entry_action
        # (evaluate's executed return may pass it) is kept when it
        # agrees with the constructed action; a record constructed
        # already-flipped against its pin is rejected here. Otherwise
        # the construction-time action IS the entry side (manually
        # constructed trades included). Written via
        # object.__setattr__ — it is bookkeeping, set once here.
        if self.entry_action is None:
            object.__setattr__(self, "entry_action", self.action)
        elif self.entry_action != self.action:
            raise _action_mismatch_error(self)


# ============================================================================
# 2. PROBABILITY AGGREGATION, PLATT SCALING & CAPPING
# ============================================================================

# --- SF-BL-9675: clip_probability / logit launder non-finite input ---
# clip_probability was max(min_p, min(max_p, p)) and logit clamped via
# max(1e-6, min(1.0 - 1e-6, p)) — min/max chains whose comparisons with
# NaN are all False, so a non-finite probability fell through to the
# upper clip: verified live pre-fix (repro.py in the task work dir):
# clip_probability(NaN) -> 0.98 and (+inf) -> 0.98 (-inf -> 0.02;
# with custom bounds NaN -> the custom max, 0.9 / 0.995 fixtures), and
# logit(NaN) -> 13.815509557935018 (the TASK's 13.8155), logit(-inf)
# -> -13.815509557963773 — near-certainty manufactured from corrupt
# input by the shared helpers beneath platt_recalibrate and
# aggregate_fleet_forecasts, for any caller that reaches them without
# a finiteness pre-filter. Kill-criterion census (CALLER_CENSUS.md in
# the task work dir): NOT met — aggregate's member clips are shielded
# by SF-BL-9439's filter and evaluate's posterior clip by its Gate 0.0
# validation, and platt's internal logit call by SF-BL-9579's p_raw
# entry check, but both helpers are public and direct callers (in-repo
# tests passing bare literals, out-of-tree fleet scripts) reach them
# with no validation at all. The values are therefore validated at
# the helpers themselves, at entry. Pre-registered domain (PREREG.md
# in the task work dir): p any FINITE real — only non-finiteness is
# invalid under this prereg; finite out-of-range values keep their
# pre-fix behaviour bit-identical (clip: -0.2 -> 0.02, 1.5 -> 0.98;
# logit: -0.2 / 1.5 clamp to the 1e-6 endpoints exactly as 0.0 / 1.0
# do — logit's finite-range domain is SF-BL-9883's separate surface,
# and rejecting it here would break SF-BL-9579's pinned platt
# fixtures). The clip bounds min_p / max_p are deliberately untouched
# (SF-BL-9713 owns bound values; SF-BL-11122 / SF-BL-11491 bound
# types; SF-BL-9713's rationale scopes SF-BL-9675 to "the non-finite
# probability argument only"). The checks reuse _is_finite_number
# (bool / non-numeric excluded), so a bool or non-numeric p raises
# this named ValueError instead of a bare TypeError ('0.5' / None
# pre-fix) or a silent True -> 0.98 / 13.8155 — the overlap with
# SF-BL-11079's clip p-type fixtures is deliberate and triggers that
# task's stated kill criterion, as SF-BL-9508's check did for
# SF-BL-10966. Pure validation — it never transforms its input — so
# every finite p clips / logits bit-identically to pre-fix, and
# platt_recalibrate / aggregate_fleet_forecasts are unchanged on
# their (already validated / filtered) internal paths.

def _validate_clip_probability_p(p: object) -> None:
    """Raise ValueError unless p is a finite real number (SF-BL-9675)."""
    if not _is_finite_number(p):
        raise ValueError(
            f"p {p!r} is not finite (SF-BL-9675): the probability fed "
            "to clip_probability must be a finite real — a non-finite "
            "p is laundered by the min/max clip chain into a clip cap "
            "(NaN and +inf -> 0.98 at the defaults, -inf -> 0.02), "
            "manufacturing near-certainty from no information for "
            "every caller beneath logit, platt_recalibrate and "
            "aggregate_fleet_forecasts")


# --- SF-BL-9883: logit converts a finite out-of-range probability ---
# SF-BL-9675 validated logit's p for finiteness but deliberately left
# its finite range to this task: the clamp max(1e-6, min(1.0 - 1e-6,
# p)) maps every finite p outside [0, 1] onto an endpoint, so a
# corrupt or unit-confused belief entered log-odds pooling and Platt
# recalibration as near-certainty — verified live pre-fix (repro.py
# in the task work dir): logit(2.0) -> 13.815509557935018 and
# logit(-1.0) -> -13.815509557963773 (the TASK's 13.8155 / -13.8155),
# logit(1.5) / logit(82.0) -> 13.815509557935018, logit(-0.2) ->
# -13.815509557963773. Kill-criterion census (CALLER_CENSUS.md in the
# task work dir): NOT met — aggregate's weighted branch logits clips
# in [0.005, 0.995] (in-range by construction), but platt_recalibrate
# pre-validates p_raw finite-only (SF-BL-9579) and forwards finite
# out-of-range values straight into logit (platt(-0.2) -> 0.02,
# (1.5) / (82.0) -> 0.98 pre-fix), and logit is public: direct
# callers (in-repo tests passing bare literals, out-of-tree fleet
# scripts) reach it with no validation at all. The value is
# therefore validated at logit itself, at entry. Pre-registered
# domain (PREREG.md in the task work dir): p a finite real in the
# CLOSED interval [0, 1] — endpoints 0.0 / 1.0 (and ints 0 / 1) stay
# valid and keep their clamped values bit-identically (the TASK:
# "endpoints may clip"); finite p strictly outside [0, 1] raises a
# ValueError naming SF-BL-9883, checked AFTER the SF-BL-9675
# finiteness check so non-finite / bool / non-numeric p keeps its
# existing SF-BL-9675 rejection and precedence (bool never reaches
# the range check: True == 1.0 must not slip through as in-range).
# Pure validation — it never transforms its input — so every
# in-domain p logits bit-identically to pre-fix. Inherited effect,
# registered pre-fix in PREREG.md: platt_recalibrate's internal
# logit(p_raw) call now rejects finite out-of-range p_raw (naming
# SF-BL-9883 via logit) instead of clamping it to the 0.02 / 0.98
# clips — the direction of SF-BL-10904's prereg, but that task's
# own entry check (naming SF-BL-10904 and p_raw, before the
# parameter checks) remains its work and is NOT landed here.
# aggregate_fleet_forecasts is unaffected by construction (clips in
# [0.005, 0.995]; pooled inv_logit output in (0, 1)).

def _validate_logit_p(p: object) -> None:
    """Raise ValueError unless p is a finite real number (SF-BL-9675)
    in the closed interval [0, 1] (SF-BL-9883)."""
    if not _is_finite_number(p):
        raise ValueError(
            f"p {p!r} is not finite (SF-BL-9675): the probability fed "
            "to logit must be a finite real — a non-finite p is "
            "laundered by the logit clamp chain into maximum-certainty "
            "log-odds (NaN and +inf -> 13.815509557935018, -inf -> "
            "-13.815509557963773), manufacturing near-certainty from "
            "no information")
    if not (0.0 <= p <= 1.0):  # type: ignore[operator]
        raise ValueError(
            f"p {p!r} is outside [0, 1] (SF-BL-9883): the probability "
            "fed to logit must be a finite real in [0, 1] — a finite "
            "out-of-range p is clamped by the logit chain into "
            "maximum-certainty log-odds (2.0 and 1.5 -> "
            "13.815509557935018, -1.0 and -0.2 -> "
            "-13.815509557963773), so a corrupt or percent-unit "
            "belief (82.0) enters log-odds pooling and Platt "
            "recalibration as near-certainty instead of being "
            "rejected; endpoints 0.0 / 1.0 remain valid and clip")


# --- SF-BL-9713: clip_probability never validates its own clip bounds ---
# SF-BL-9675 validated clip_probability's probability argument but
# deliberately left its bounds to this task: clip_probability was
# max(min_p, min(max_p, p)) with no check on min_p / max_p themselves,
# and the max/min chain assumes min_p < max_p and both finite. Verified
# live pre-fix (repro.py in the task work dir): inverted bounds
# clip_probability(0.5, 0.90, 0.10) returned 0.90 — min(0.10, p) is
# <= 0.10 for every p, so max(0.90, ...) pins EVERY probability to
# 0.90; clip_probability(0.5, NaN, 0.98) returned NaN, (0.5, 0.02, NaN)
# returned 0.02 (a NaN max silently pins to the floor), +inf min
# returned inf, equal bounds (0.5, 0.5) collapsed every forecast to a
# constant, and out-of-simplex bounds (min -0.5, max 1.5) were accepted
# silently. Both downstream surfaces forward the bounds as caller
# parameters, so the corruption propagated to every pooled and
# recalibrated forecast the trade path consumes:
# platt_recalibrate(0.60, min_p=0.90, max_p=0.10) returned 0.90 and
# aggregate_fleet_forecasts on a 0.70/0.60/0.65 three-family panel
# pooled to (NaN, 0.04082482904638629) at min_p=NaN (the TASK's
# (NaN, 0.041) fixture) and to (0.90, ...) at inverted bounds.
# Kill-criterion census (CALLER_CENSUS.md in the task work dir): NOT
# met — the bounds are NOT pinned or removed: clip_probability,
# platt_recalibrate and aggregate_fleet_forecasts all expose
# min_p / max_p as public caller parameters and forward them, in-repo
# tests already pass non-default literal bounds (0.1/0.9, 0.005/0.995),
# and an in-repo census cannot bind out-of-tree fleet callers or
# config-supplied kwargs (SF-BL-9476/9508 precedent). The values
# themselves are therefore validated at every surface that exposes
# them, per PREREG.md in the task work dir: finite reals with
# 0 <= min_p < max_p <= 1, checked at clip_probability entry (after
# the SF-BL-9675 p check, which keeps precedence), and at
# platt_recalibrate / aggregate_fleet_forecasts entry in the
# SF-BL-9508 entry-validation pattern, so aggregate inherits the
# rejection on every branch and before member processing. The check
# reuses _is_finite_number (bool / non-numeric excluded), so a bool or
# non-numeric bound raises this named ValueError instead of a bare
# TypeError ('0.02' / None pre-fix) or a silent True -> inf-bound /
# False coercion — the overlap with the bound-type surfaces
# SF-BL-11122 / SF-BL-11491 is deliberate, as SF-BL-9508's was with
# SF-BL-10966. Pure validation — it never transforms its inputs — so
# every in-domain bound pair (defaults 0.02/0.98, member 0.005/0.995,
# boundary 0.0/1.0, int 0/1 included) clips bit-identically to pre-fix.

def _clip_bounds_in_domain(min_p: object, max_p: object) -> bool:
    """True iff the clip bounds are finite reals with
    0 <= min_p < max_p <= 1 (SF-BL-9713)."""
    if not _is_finite_number(min_p) or not _is_finite_number(max_p):
        return False
    return 0.0 <= min_p < max_p <= 1.0  # type: ignore[operator]


def _validate_clip_bounds(
    min_p: object,
    max_p: object,
    min_name: str = "min_p",
    max_name: str = "max_p",
) -> None:
    """Raise ValueError unless the clip bounds are in the SF-BL-9713
    domain: finite reals with 0 <= min_p < max_p <= 1.

    `min_name` / `max_name` are the parameter names at the calling
    surface (all three surfaces name them "min_p"/"max_p" today), so
    the error names the argument the caller actually supplied. The
    offending bound is checked in order: min_p finiteness, max_p
    finiteness, then the joint ordering/range.
    """
    if not _is_finite_number(min_p):
        raise ValueError(
            f"{min_name} {min_p!r} is not finite (SF-BL-9713): the "
            "clip_probability lower bound must be a finite real with "
            "0 <= min_p < max_p <= 1 — a non-finite min_p is returned "
            "by (NaN) or pins (via the max/min chain) every clipped, "
            "recalibrated and pooled probability (aggregate fixture: "
            "0.70/0.60/0.65 panel pooled to (NaN, 0.041) at "
            "min_p=NaN)")
    if not _is_finite_number(max_p):
        raise ValueError(
            f"{max_name} {max_p!r} is not finite (SF-BL-9713): the "
            "clip_probability upper bound must be a finite real with "
            "0 <= min_p < max_p <= 1 — a non-finite max_p pins every "
            "clipped, recalibrated and pooled probability to a cap or "
            "the floor (NaN max returned the 0.02 floor for every p "
            "pre-fix)")
    if not (0.0 <= min_p < max_p <= 1.0):  # type: ignore[operator]
        raise ValueError(
            f"clip bounds ({min_name}={min_p!r}, {max_name}={max_p!r}) "
            "outside pre-registered domain 0 <= min_p < max_p <= 1 "
            "(SF-BL-9713): inverted bounds pin every probability to "
            "min_p (fixture: bounds 0.90/0.10 returned 0.90 for every "
            "p), equal bounds collapse every forecast to a constant, "
            "and a bound outside [0, 1] is not a probability")


def clip_probability(p: float, min_p: float = 0.02, max_p: float = 0.98) -> float:
    """
    Hard probability cap (Metaculus Fall 2025 Survey r = +0.48, p = 0.005).
    Prevents a single hallucination, unit error, or resolution misread from wiping out log-wealth.
    Raises ValueError if p is not a finite real number (SF-BL-9675),
    checked before any clipping; finite p clips exactly as before.
    Raises ValueError unless the bounds are finite reals with
    0 <= min_p < max_p <= 1 (SF-BL-9713), checked after p and before
    any clipping; in-domain bounds clip exactly as before.
    """
    _validate_clip_probability_p(p)
    _validate_clip_bounds(min_p, max_p)
    return max(min_p, min(max_p, p))


def logit(p: float) -> float:
    """Log-odds of p, clamped to [1e-6, 1 - 1e-6] before the log.

    Raises ValueError if p is not a finite real number (SF-BL-9675)
    or is a finite real outside [0, 1] (SF-BL-9883), checked before
    the clamp; every in-domain p (endpoints 0.0 / 1.0 included)
    returns exactly its pre-fix clamped value.
    """
    _validate_logit_p(p)
    p_safe = max(1e-6, min(1.0 - 1e-6, p))
    return math.log(p_safe / (1.0 - p_safe))


# --- SF-BL-9919: inv_logit saturates to exact 0.0 / 1.0 and passes NaN --
# inv_logit was the raw stable sigmoid with no validation and no output
# clip, so its float result left the open interval (0, 1) at both tails
# and laundered non-finite input: verified live pre-fix (repro.py in
# the task work dir): inv_logit(1000.0) -> exactly 1.0 and
# inv_logit(-1000.0) -> exactly 0.0 (positive-side saturation begins at
# x = 37, where 1/(1+exp(-x)) rounds to 1.0; negative-side exact zero
# begins at x = -746, where exp(x) underflows), and inv_logit(NaN) ->
# NaN (the x >= 0 comparison is False for NaN, so NaN flowed through
# the else branch), inv_logit(+inf) -> 1.0, inv_logit(-inf) -> 0.0.
# aggregate_fleet_forecasts ships inv_logit(extremizing_a *
# weighted_logit) as its pooled probability and platt_recalibrate
# ships inv_logit(alpha * logit(p_raw) + beta) into clip_probability,
# so an exact 0.0 / 1.0 violates the module's own open-interval
# probability convention (clip_probability bounds calibrated
# probabilities to [0.02, 0.98]; logit clamps its input away from the
# endpoints) and a NaN logit propagates as a NaN belief instead of a
# rejection. Kill-criterion audit (CALLER_CENSUS.md in the task work
# dir): NOT met — no tested invariant shields the helper: extremizing_a
# (SF-BL-9476) and the Platt alpha (SF-BL-9508) are each validated
# finite with NO preregistered upper bound, so a * weighted_logit =
# 52.93 on an extreme (0.995) panel pooled to exactly 1.0 pre-fix at
# a = 10, and alpha * logit(p_raw) overflows to +inf from all-finite
# validated inputs (alpha = 1e308, p_raw = 0.99), which pre-fix
# inv_logit returned as 1.0 into the 0.98 clip; direct out-of-tree
# callers reach the public helper with bare floats and no validation
# at all. The value is therefore validated and clipped here, at the
# helper. Pre-registered design (PREREG.md in the task work dir): (1)
# non-finite input raises — the check reuses _is_finite_number (bool /
# non-numeric excluded), so a bool or non-numeric argument raises this
# named ValueError instead of a bare TypeError ('1.0' / None pre-fix,
# from the x >= 0 comparison) or a silent True -> 0.7310585786300049;
# the overlap with SF-BL-11042's type fixtures is deliberate and
# triggers that task's stated kill criterion, as SF-BL-9675's check
# did for SF-BL-11079. (2) The result is clipped into the OPEN interval
# (0, 1) at the nextafter bounds below — NOT into the [0.02, 0.98]
# band: that band is clip_probability's calibrated-probability policy
# and already applies downstream (platt clips this function's result;
# aggregate's return passes through platt), while embedding it here
# would destroy the logit/inv_logit round-trip for every |x| > 3.89
# (inv_logit(10.0) = 0.9999546021312976 -> 0.98) and rewrite platt's
# pre-clip value for legitimate extreme logits without changing any
# final clipped output. The defect is the EXACT boundary, so the clip
# binds only there: no float lies strictly between the nextafter
# bounds and 0 / 1, so every finite non-saturating output — the whole
# interior grid, ordinary aggregate panels (whose member clip bounds
# |a * weighted_logit| <= 6.62 at the default a) and the benchmark —
# stays bit-identical to pre-fix. Ordinary panels' bit-identity is
# the fleet's required regression property, not the prereg's kill:
# on a panel that actually saturates (0.995 panel, a = 10) pooled_prob
# is NOT bit-identical pre/post (exactly 1.0 -> 0.9999999999999999).

INV_LOGIT_MIN = math.nextafter(0.0, 1.0)  # 5e-324, smallest float > 0
INV_LOGIT_MAX = math.nextafter(1.0, 0.0)  # 0.9999999999999999


def _validate_inv_logit_x(x: object) -> None:
    """Raise ValueError unless x is a finite real number (SF-BL-9919)."""
    if not _is_finite_number(x):
        raise ValueError(
            f"x {x!r} is not finite (SF-BL-9919): the logit fed to "
            "inv_logit must be a finite real — a non-finite logit "
            "passed through the sigmoid branches as an exact boundary "
            "probability or NaN (+inf -> 1.0, -inf -> 0.0, NaN -> NaN), "
            "so a corrupt or overflowed logit sum shipped as a pooled "
            "calibrated probability instead of being rejected")


def inv_logit(x: float) -> float:
    """Sigmoid (inverse logit), result clipped into the open interval
    (0, 1) at the nextafter bounds (SF-BL-9919).

    Raises ValueError if x is not a finite real number, checked before
    any computation; every finite non-saturating x returns exactly its
    pre-fix sigmoid value, and a saturating x returns
    INV_LOGIT_MIN / INV_LOGIT_MAX instead of exactly 0.0 / 1.0.
    """
    _validate_inv_logit_x(x)
    if x >= 0:
        z = math.exp(-x)
        result = 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        result = z / (1.0 + z)
    return max(INV_LOGIT_MIN, min(INV_LOGIT_MAX, result))


# --- SF-BL-9508: Platt parameters are unvalidated ------------------------
# aggregate_fleet_forecasts passed its caller-supplied platt_alpha /
# platt_beta straight into platt_recalibrate, and platt_recalibrate
# applied its alpha / beta with no validity check: the recalibration
# inv_logit(alpha * logit(p_raw) + beta) feeds clip_probability, whose
# min/max comparisons with NaN are all False, so a NaN parameter was
# laundered into the 0.98 cap — a max-confidence forecast manufactured
# from no information. Verified live pre-fix (repro.py in the task work
# dir): aggregate on a uniform 0.70 three-family panel returned
# (0.98, 0.0) at platt_alpha=NaN and at platt_beta=NaN/+inf, (0.02, 0.0)
# at platt_alpha=-inf / platt_beta=-inf, collapsed to (0.5, 0.0) at
# platt_alpha=0.0 (every input maps to sigmoid(beta); the panel is
# discarded) and rank-inverted at negative alpha (platt_alpha=-1.08 ->
# 0.2416, below every member); platt_recalibrate(0.60) showed the same
# grid directly (clean 0.6116; alpha/beta NaN or +inf -> 0.98).
# SF-BL-9476 validated only the extremizing_a argument of the same
# aggregate function — the Platt pair is a separate input surface fed
# by caller configuration. Kill-criterion audit (PRODUCER_AUDIT.md in
# the task work dir): NOT met — no caller in the repo passes either
# parameter at all (the benchmark uses the literal defaults 1.08/0.0),
# the repo's fitted PlattScaler (calibration.py) is wired nowhere near
# this surface (its sole consumer calls PlattScaler.predict directly),
# and even it guarantees no invariant of the required form: its
# SF-BL-9410 intercept-only fit returns A_ = 0.0 by design and no test
# pins fitted A_ / B_ as finite-and-positive for downstream use. The
# values themselves are therefore validated at both consumption gates:
# aggregate_fleet_forecasts entry (SF-BL-9476-style, before member
# processing and branch selection) and platt_recalibrate entry (the
# TASK's done criterion names both surfaces). Pre-registered domains
# (PREREG.md in the task work dir): alpha a finite real strictly > 0
# (1 = identity, >1 sharpens, 0<alpha<1 tempers; 0 collapses the pool
# to a constant, negatives rank-invert; NO upper bound is
# preregistered, so none is imposed — SF-BL-9364/9476 precedent);
# beta any finite real (an unconstrained logit intercept — negative,
# zero and positive fitted intercepts are all legitimate; only
# non-finiteness pins or launders the output). The checks reuse
# _is_finite_number (bool / non-numeric excluded), so a bool or
# non-numeric parameter raises this named ValueError instead of a bare
# TypeError ('1.08' / None pre-fix) or a silent True == 1.0 identity
# rescale — the overlap with SF-BL-10966's fixtures is deliberate and
# triggers that task's stated kill criterion. Pure validation — it
# never transforms its inputs — so every in-domain pair, defaults
# included, recalibrates bit-identically to pre-fix. Scope: p_raw and
# min_p / max_p validity are separate prereg surfaces (SF-BL-9579 /
# 10904 / 11014 / 11084) and are deliberately untouched here.

def _is_valid_platt_alpha(alpha: object) -> bool:
    """True iff a Platt alpha is a finite real number strictly > 0."""
    return _is_finite_number(alpha) and alpha > 0  # type: ignore[operator]


def _is_valid_platt_beta(beta: object) -> bool:
    """True iff a Platt beta is a finite real number (any sign)."""
    return _is_finite_number(beta)


def _validate_platt_params(
    alpha: object,
    beta: object,
    alpha_name: str = "alpha",
    beta_name: str = "beta",
) -> None:
    """Raise ValueError unless the Platt pair is in the SF-BL-9508 domains.

    `alpha_name` / `beta_name` are the parameter names at the calling
    surface ("alpha"/"beta" for platt_recalibrate,
    "platt_alpha"/"platt_beta" for aggregate_fleet_forecasts), so the
    error names the argument the caller actually supplied. Alpha is
    checked first.
    """
    if not _is_valid_platt_alpha(alpha):
        raise ValueError(
            f"{alpha_name} {alpha!r} outside pre-registered domain "
            "(0, finite] (SF-BL-9508): the Platt slope must be a finite "
            "real strictly > 0 — a non-finite slope is laundered by "
            "the clip into the 0.98 cap (uniform 0.70 panel fixture: "
            "platt_alpha=NaN pooled to (0.98, 0.0)), zero collapses "
            "every forecast to the constant sigmoid(beta), and a "
            "negative slope rank-inverts the recalibration")
    if not _is_valid_platt_beta(beta):
        raise ValueError(
            f"{beta_name} {beta!r} is not finite (SF-BL-9508): the "
            "Platt intercept must be a finite real (any sign) — a "
            "non-finite intercept pins every recalibration to a clip "
            "cap (+inf -> 0.98, -inf -> 0.02) or launders NaN into "
            "the 0.98 cap, whatever the pooled probability was")


# --- SF-BL-9579: non-finite p_raw is laundered into near-certainty -----
# SF-BL-9508 validated platt_recalibrate's alpha / beta but deliberately
# left p_raw to this task (its PREREG: "p_raw validity is other tasks'
# surface (SF-BL-9579 ...)"). Pre-fix p_raw entered logit() with no
# check: logit's clamp (max(1e-6, min(1.0 - 1e-6, p))) and
# clip_probability's min/max chain both compare against NaN as False,
# so a non-finite input passed straight through to a clip cap —
# verified live pre-fix (repro.py in the task work dir):
# platt_recalibrate(NaN) -> 0.98 and (+inf) -> 0.98 (a max-confidence
# forecast manufactured from no information), (-inf) -> 0.02, and the
# TASK title's aggregate case pooled a 0.70/0.60/0.65 panel to 0.98
# back when its platt_alpha=NaN went unvalidated (that parameter half
# is now closed by SF-BL-9508 at both surfaces). A mis-fitted or
# corrupted pooled probability thus silently extremized instead of
# raising, corrupting the ECE the calibration multiplier grades.
# Kill-criterion audit (PRODUCER_AUDIT.md in the task work dir): NOT
# met — alpha/beta come from no fitted calibrator at all (bare caller
# configuration, per SF-BL-9508's audit), and p_raw reaches this
# public function from direct out-of-tree callers passing bare floats
# with no validated producer invariant; its sole in-repo caller,
# aggregate_fleet_forecasts, passes a pooled value that is finite
# only as an emergent property of SF-BL-9439/9506's member filters,
# with no test pinning it as this function's input contract. The
# value itself is therefore validated here, at entry, BEFORE the
# SF-BL-9508 parameter check, so p_raw is named even when a parameter
# is also bad. Pre-registered domain (PREREG.md in the task work
# dir): p_raw any FINITE real — only non-finiteness is invalid under
# this prereg; finite out-of-range values are SF-BL-10904's separate
# surface and keep their pre-fix logit-clamp behaviour bit-identical
# (0.0 -> 0.02, 1.0 -> 0.98, -0.2 -> 0.02, 1.5 -> 0.98 at defaults).
# The check reuses _is_finite_number (bool / non-numeric excluded),
# so a bool or non-numeric p_raw raises this named ValueError instead
# of a bare TypeError ('0.6' / None pre-fix) or a silent True -> 0.98
# — the overlap with SF-BL-11014's p_raw type fixtures is deliberate,
# as SF-BL-9508's was with SF-BL-10966. Pure validation — it never
# transforms its input — so every finite p_raw recalibrates
# bit-identically to pre-fix, and aggregate_fleet_forecasts inherits
# the rejection through its internal platt_recalibrate call.

# --- SF-BL-10904: finite out-of-range p_raw has no platt entry check --
# SF-BL-9579 validated platt_recalibrate's p_raw for FINITENESS only
# and deliberately left its finite range to this task (its PREREG:
# "Finite values outside [0, 1] are a separate prereg surface
# (SF-BL-10904)"). At the TASK's filing the logit clamp laundered a
# finite out-of-range p_raw into a clip cap — verified live at
# filing: platt_recalibrate(82.0) -> 0.98 (a percent-unit belief
# emerging as a calibrated 0.98), (1.50) -> 0.98 (the max clip),
# (-0.20) -> 0.02 (the min clip), flowing through
# aggregate_fleet_forecasts into the pooled probability the fleet
# trades and is ECE-scored on (clean platt_recalibrate(0.60) ->
# 0.6116). SF-BL-9883 has since landed logit's own finite-range
# rejection, so on the current tree those fixtures raise — but
# naming SF-BL-9883 and `p`, via the internal logit(p_raw) call,
# AFTER the SF-BL-9508 parameter and SF-BL-9713 bounds checks
# (verified live pre-fix, repro.py in the task work dir:
# platt(82.0, alpha=NaN) named alpha / SF-BL-9508 first;
# platt(-0.20, min_p=0.90, max_p=0.10) named the bounds /
# SF-BL-9713 first). SF-BL-9883's PREREG registered pre-fix that
# "that task's own entry check (naming SF-BL-10904 and p_raw,
# before the parameter checks) remains its work and is NOT landed
# here" — this is that check. Kill-criterion census
# (CALLER_CENSUS.md in the task work dir): NOT met — aggregate's
# internal call passes a range-validated pooled_prob by
# construction (inv_logit in (0, 1) / member clips in
# [0.005, 0.995]), but platt_recalibrate is public: in-repo tests
# call it with bare literals and out-of-tree fleet scripts pass
# bare floats, with no range-validated producer under a tested
# invariant in between. The value is therefore validated here, at
# entry. Pre-registered domain (PREREG.md in the task work dir):
# p_raw a finite real in the CLOSED interval [0, 1] — endpoints
# 0.0 / 1.0 (and ints 0 / 1) stay valid and recalibrate
# bit-identically (0.0 -> 0.02, 1.0 -> 0.98 at defaults); finite
# p_raw strictly outside [0, 1] raises a ValueError naming
# SF-BL-10904 and p_raw, checked INSIDE _validate_platt_p_raw
# AFTER the SF-BL-9579 finiteness check (so non-finite / bool /
# non-numeric p_raw keeps its existing SF-BL-9579 rejection and
# precedence — True == 1.0 never reaches the range check as an
# in-range endpoint) and BEFORE the SF-BL-9508 parameter and
# SF-BL-9713 bounds checks, so p_raw is named even when a
# parameter or bound is also bad. logit's own SF-BL-9883 check is
# untouched and still governs direct logit callers. Pure
# validation — it never transforms its input — so every in-domain
# p_raw recalibrates bit-identically to pre-fix, and
# aggregate_fleet_forecasts is unaffected by construction.

# --- SF-BL-11014: p_raw TYPE is never validated as its own dimension --
# At this task's filing (2026-10-22) platt_recalibrate validated the
# TYPE of its raw probability nowhere: p_raw='0.6' raised a bare
# TypeError ('<' not supported between 'str' and 'float') inside the
# logit chain, and p_raw=True flowed through logit(True)=13.82 to
# return 0.98, the max clip, as if a bool were certainty (both
# verified live at filing; clean 0.6 -> 0.6116). SF-BL-9579 has since
# landed the finiteness check above, whose _is_finite_number
# predicate excludes bool / non-numeric values by type, so on the
# tree before this fix the TASK fixtures already raised — but named
# SF-BL-9579 and framed a string as "not finite" (verified live
# pre-fix, repro.py in the task work dir: platt('0.6') and
# platt(True) both raised the SF-BL-9579 finiteness ValueError).
# '0.6' is not a non-finite number; it is not a number at all, and
# True is not a probability that happens to be non-finite — the
# TYPE dimension is this task's distinct surface (distinct from
# SF-BL-9579's non-finite numerics, SF-BL-10904's finite
# out-of-range values, and SF-BL-10966's platt_alpha / platt_beta
# parameter types), exactly as SF-BL-10904's entry check remained
# its work after SF-BL-9883 landed an inherited raise with the
# wrong name. Kill-criterion census (CALLER_CENSUS.md in the task
# work dir): NOT met — aggregate_fleet_forecasts' internal call
# passes a float pooled_prob by construction, but platt_recalibrate
# is public, in-repo tests call it with bare literals, serialized
# producers (JSON / CSV / config rows, where a probability's
# canonical form IS the string '0.6') reach it with no
# type-validated producer under a tested invariant, so p_raw does
# NOT only ever arrive type-validated from aggregate. Fix, per
# PREREG.md in the task work dir: the TYPE of p_raw is validated
# here, as the FIRST step of _validate_platt_p_raw — bool or
# not-an-int/float raises a ValueError naming SF-BL-11014 and
# p_raw, before the SF-BL-9579 finiteness step, the SF-BL-10904
# range step, the SF-BL-9508 parameter check and the SF-BL-9713
# bounds check, so p_raw's type is named even when a parameter or
# bound is also bad. The type message ALSO names SF-BL-9579
# (dual naming, registered in the PREREG): landed sibling suites
# pin these exact fixtures by substring on SF-BL-9579, and both
# names stay true — this check IS the type half of that
# finiteness predicate, promoted to its own named step. Pure
# validation — it never coerces ('0.6' is never parsed, True is
# never read as 1.0) — so every numeric p_raw (int / float,
# endpoints and huge ints included) flows to the unchanged
# finiteness / range steps and recalibrates bit-identically, and
# aggregate_fleet_forecasts is unaffected by construction.

def _validate_platt_p_raw(p_raw: object) -> None:
    """Raise ValueError unless p_raw is a real number (int / float,
    bool excluded) (SF-BL-11014) that is finite (SF-BL-9579) and in
    the closed interval [0, 1] (SF-BL-10904)."""
    if isinstance(p_raw, bool) or not isinstance(p_raw, (int, float)):
        raise ValueError(
            f"p_raw {p_raw!r} (type {type(p_raw).__name__}) is not "
            "a real number (SF-BL-11014; type rejection first "
            "recorded under SF-BL-9579): the raw probability fed "
            "to Platt recalibration must be a finite real number "
            "(int or float, bool excluded) — a non-numeric p_raw "
            "crashed the logit chain with a bare TypeError ('0.6' "
            "at filing) and a bool p_raw was read as a number "
            "(True == 1 flowed through logit to the 0.98 max clip "
            "as if it were certainty), manufacturing or crashing "
            "on a serialized probability instead of rejecting it "
            "at this entry and corrupting ECE; a serialized "
            "probability must be converted to a number by its "
            "producer, never coerced here")
    if not _is_finite_number(p_raw):
        raise ValueError(
            f"p_raw {p_raw!r} is not finite (SF-BL-9579): the raw "
            "probability fed to Platt recalibration must be a finite "
            "real — a non-finite p_raw is laundered by the logit / "
            "clip min-max chains into a clip cap (NaN and +inf -> "
            "0.98, -inf -> 0.02), manufacturing near-certainty from "
            "no information and corrupting ECE")
    if not (0.0 <= p_raw <= 1.0):  # type: ignore[operator]
        raise ValueError(
            f"p_raw {p_raw!r} is outside [0, 1] (SF-BL-10904): the "
            "raw probability fed to Platt recalibration must be a "
            "finite real in [0, 1] — a finite out-of-range p_raw "
            "was clamped by the logit chain into a clip cap (82.0 "
            "and 1.50 -> 0.98, -0.20 -> 0.02), so a corrupt or "
            "percent-unit belief (82.0 for 82%) shipped as a "
            "calibrated near-certainty into the pooled probability "
            "the fleet trades and is ECE-scored on instead of "
            "being rejected at this entry; endpoints 0.0 / 1.0 "
            "remain valid and recalibrate as before")


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
    Raises ValueError if p_raw is not a real number — bool or
    non-numeric type (SF-BL-11014) — or is not a finite real
    (SF-BL-9579) or is a finite real outside [0, 1] (SF-BL-10904), or if
    alpha is not a finite real > 0 or beta is not a finite real
    (SF-BL-9508), checked before any computation. Raises ValueError
    unless min_p / max_p are finite reals with 0 <= min_p < max_p <= 1
    (SF-BL-9713), also checked before any computation.
    """
    _validate_platt_p_raw(p_raw)
    _validate_platt_params(alpha, beta)
    _validate_clip_bounds(min_p, max_p)
    recalibrated = inv_logit(alpha * logit(p_raw) + beta)
    return clip_probability(recalibrated, min_p=min_p, max_p=max_p)


# --- SF-BL-9439: non-finite member probabilities are dropped before pooling --
# clip_probability / logit use min/max chains whose comparisons with NaN are
# all False, so a NaN raw_probability fell through to the 1-1e-6 clip and
# entered the weighted log-odds pool as near-certainty: members 0.70/0.60/NaN
# (3 distinct families) pooled to (0.9501, dispersion 0.168) — above every
# real member, with dispersion still under the Gate 2 threshold (0.22). The
# median branch was poisoned too (NaN clips to 0.995 and becomes the median /
# corrupts the trimmed sort), +-inf likewise, and an all-NaN list returned
# the 0.98 max clip. SubAgentForecast is a plain dataclass with NO finiteness
# validation at construction (kill condition checked: it accepts NaN/inf,
# and its only in-repo call site passes bare literals), so the filter lives
# here, at aggregation time — which also catches members mutated to NaN
# after construction. Dropped members are logged via a UserWarning naming
# their agent_ids and reporting the pooled n ("pooling n=<kept> of
# <total>"); dispersion, branch selection (distinct families / independent
# search) and weights are all computed over the kept members only.

def _is_finite_member_probability(p: object) -> bool:
    """True iff p is a finite real probability value (bool/non-numeric: no)."""
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        return False
    return math.isfinite(p)


# --- SF-BL-9506: non-finite member weights are dropped before pooling -----
# The weighted log-odds branch divides each historical_brier_weight by
# total_w, so a single NaN weight made total_w and the whole weighted
# logit sum NaN; inv_logit / platt / clip then saturated that NaN at the
# 0.98 cap: members 0.90/0.30/0.30 (3 distinct families) with one NaN
# weight returned (0.98, dispersion 0.283) instead of pooling the finite
# members. +-inf weights did the same (inf/inf is NaN; -inf likewise),
# an all-NaN-weight list returned the 0.98 cap, and a non-numeric weight
# raised a bare TypeError from the sum. SubAgentForecast validates
# nothing at construction (kill condition checked in PRODUCER_AUDIT.md in
# the SF-BL-9506 work dir: no __post_init__, and the only in-repo weight
# producer is the benchmark fixture list passing bare literals — no
# tested finite-only invariant exists), so the filter lives here, at
# aggregation time, alongside SF-BL-9439 — which also catches members
# mutated to a non-finite weight after construction. The drop applies in
# every branch, not only the weighted one: a member whose weight
# metadata is corrupt is dropped whole, so dispersion, branch selection
# and pooling all use the kept members only. Finite negative weights
# (SF-BL-9147) and a zero total over kept weights (SF-BL-8901) are
# separate preregs and pass this finiteness filter untouched.

def _is_finite_member_weight(w: object) -> bool:
    """True iff w is a finite real weight value (bool/non-numeric: no)."""
    if isinstance(w, bool) or not isinstance(w, (int, float)):
        return False
    return math.isfinite(w)


# --- SF-BL-10723: finite out-of-range member probabilities are dropped ---
# SF-BL-9439 / SF-BL-9506 screen member probabilities and weights for
# FINITENESS only, so a finite raw_probability outside [0, 1] passed both
# filters and reached the raw_probs line, whose
# clip_probability(f, 0.005, 0.995) silently clipped it to a near-certain
# member: a percent-unit member at 82.0 became 0.995 and pooled a
# 0.60/0.60 panel (3 distinct families) to (0.9397, dispersion 0.1862)
# instead of the pool of the remaining members, and a member at -0.5
# became 0.005 and pooled the same panel to (0.1174, 0.2805) (both
# verified live pre-fix; repro.py in the task work dir). The median and
# trimmed branches were corrupted too (a +82.0 member became the median
# of a 4-member shared panel; a -0.5 member dragged the trimmed pool and
# pushed dispersion to 0.2249), an all-out-of-range panel returned the
# nonsensical (0.5, 0.495), and the dragged pooled posterior is exactly
# what evaluate_and_size_opportunity sizes on. The valid domain is the
# CLOSED interval [0, 1]: the endpoints 0.0 / 1.0 are legitimate extreme
# probabilities, kept bit-identically (the internal [0.005, 0.995] clip
# handles them as before). Kill-criterion audit (CALLER_CENSUS.md in
# the task work dir): NOT met — no producer guarantees [0, 1] at schema
# level: SubAgentForecast is a plain dataclass with no __post_init__
# (it accepts 82.0 at construction), its only in-repo production caller
# is the benchmark's bare-literal fixture list, data-driven fleet
# producers convert with a bare float() and no range check, and the
# dataclass is mutable, so the drop lives here, at aggregation time,
# alongside SF-BL-9439 / SF-BL-9506 — which also catches members mutated
# out of range after construction. Dropped members are logged via a
# UserWarning naming SF-BL-10723, their agent_ids and the pooled n.

def _is_out_of_range_member_probability(p: object) -> bool:
    """True iff p is a finite real probability outside [0, 1] (SF-BL-10723).

    NaN / +-inf return False — they are SF-BL-9439's non-finite case,
    not this one; bool / non-numeric values likewise return False (they
    fail SF-BL-9439's finiteness screen). The comparison itself is
    exact for ints, so an int beyond float range (e.g. 10**400, whose
    math.isfinite would raise OverflowError) is classified here, before
    the finiteness screen can see it.
    """
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        return False
    if isinstance(p, float) and not math.isfinite(p):
        return False
    return p < 0.0 or p > 1.0


# --- SF-BL-10848: non-bool has_independent_search flips the pool branch ---
# aggregate_fleet_forecasts selected its pooling branch with
# all(f.has_independent_search for f in forecasts) — a bare-truthiness
# read — so the branch was decided by Python truthiness, not by the
# flag's declared bool value. Any non-empty string is truthy, so a
# member whose flag arrived SERIALIZED as the string 'False' (the
# canonical form of a bool in JSON text / CSV / config rows, where
# bool("False") is True) satisfied all(...) and pooled via extremized
# log-odds, while the same member carrying the real bool False pooled
# via the shared-context median. Verified live pre-fix on the TASK's
# three-family panel 0.70/0.60/0.50 (repro.py in the task work dir):
# flag False (bool) -> (0.6077590814688054, 0.08164965809277258)
# [median branch]; flag 'False' (str) -> (0.6373182327645819, same
# dispersion) [extremized log-odds] — a serialized flag silently
# selected the Satopaa extremizing regime the data explicitly
# disclaims. The same bare read misreads every other non-bool: 1 / 1.0
# / 'True' pool as independent, 0 / 0.0 / None / '' pool as shared —
# branch selection by accident of type, not by declared value.
# Kill-criterion census (PRODUCER_CENSUS.md in the task work dir): NOT
# met — the flag is NOT only ever written as a real bool by in-repo
# constructors. SubAgentForecast is a plain dataclass with no
# __post_init__ (it accepts 'False', 1 and None at construction,
# probed live); its only in-repo production constructors are the
# benchmark fixture lists, which omit the flag entirely (default
# True) and enforce nothing; the dataclass is mutable and
# **dict / JSON-style unpacking delivers whatever the serialized
# source carried, so the strict-bool screen lives here, at
# aggregation time, alongside SF-BL-9439 / SF-BL-9506 / SF-BL-10723 —
# which also catches members mutated to a non-bool flag after
# construction. Fix, per the prereg's first option: a member whose
# has_independent_search is not a strict bool is DROPPED-AND-LOGGED
# (never coerced: 'False' must not become False, nor 'True' become
# True — a corrupt flag means the member's independence is unknown,
# so the member leaves the pool whole, exactly like a member whose
# probability or weight metadata is corrupt). The screen runs BEFORE
# branch selection, dispersion and pooling; every strict-bool panel
# pools bit-identically to pre-fix.

def _is_strict_bool_flag(v: object) -> bool:
    """True iff v is a strict bool (True / False) (SF-BL-10848).

    isinstance(v, bool) is the whole test: strings ('False', 'True',
    ''), ints / floats (0, 1, 0.0, 1.0), None and containers all fail
    it, as does numpy's bool_ (not a Python bool instance), so only a
    real bool keeps its member in the pool.
    """
    return isinstance(v, bool)


# --- SF-BL-8901: zero total weight crashes the weighted log-odds pool ----
# The weighted branch divided each historical_brier_weight by total_w
# with no zero check, so a panel of >=3 distinct families with
# independent search and every weight 0.0 raised ZeroDivisionError —
# exactly the state of a fresh panel whose members have no track record
# yet (weight is a track-record weight; 0.0 is its legitimate "none yet"
# value). Kill-criterion census (CENSUS.md in the task work dir): NOT
# met — historical_brier_weight is floored above 0 NOWHERE:
# SubAgentForecast is a plain dataclass with no __post_init__ (it
# accepts 0.0 at construction, verified live), its only in-repo producer
# is the benchmark fixture list passing bare literals with no clamp or
# tested invariant, the dataclass is mutable (post-construction mutation
# bypasses any hypothetical constructor floor), and SF-BL-9506's own
# tests pin 0.0 as a reachable finite weight deferred to this prereg.
# Fix, per the prereg's first option: when total_w == 0.0 there is no
# track-record information to weight by, so the pool falls back to the
# EQUAL-WEIGHT log-odds result (identical to every weight being 1.0),
# logged via a UserWarning naming SF-BL-8901 and the pooled n. This
# covers both all-zero weights and finite weights cancelling to exactly
# 0.0 (SF-BL-9147's negative weights with a non-zero total divide
# exactly as before). Non-finite weights never reach here — SF-BL-9506
# drops them above — so total_w is a sum of finite weights only. The
# median / trimmed branches never read weights and are untouched.


# --- SF-BL-9476: extremizing_a is unvalidated ---------------------------
# aggregate_fleet_forecasts multiplies the pooled logit by the
# caller-supplied extremizing_a with, pre-fix, no finiteness or sign
# check, so the factor's failure modes fed Platt recalibration and the
# trade gates directly (all verified live pre-fix; repro.py in the task
# work dir, uniform 0.70 panel, 3 distinct families, independent):
# a=NaN pooled to the 0.98 cap (NaN * logit is NaN; inv_logit / Platt /
# clip saturate NaN at max_p), a=+inf likewise 0.98, a=-inf the 0.02
# floor, a=0.0 collapsed the pool to (0.5, 0.0), and a=-2.0 rank-inverted
# the panel to 0.13821881130243546 (vs 0.7583861891129468 at the 1.25
# default). On the median / trimmed branch a bad factor was silently
# DROPPED — the branch never reads it — so the same corrupt config
# pooled without complaint. Kill-criterion census (CENSUS.md in the
# task work dir): NOT met — the only in-repo production caller (the
# benchmark) and every test caller omit the argument today, but the
# parameter is public with no default-only enforcement, so an in-repo
# census cannot bind out-of-tree fleet callers or config-supplied
# kwargs; the value itself is validated here, at the consumption gate.
# Pre-registered band (PREREG.md in the task work dir): a finite real
# strictly > 0 — a=1 is the identity, a>1 extremizes (Satopaa et al.),
# 0<a<1 is legitimate de-extremization / shrinkage (cf. engine.py
# log_odds_pool's (0, 1.40] domain and calibration.py's SF-BL-3596
# finite-and->0 screen). NO upper bound is imposed: none is
# preregistered, and the in-file precedent (SF-BL-9364) imposes none
# where none is preregistered; the [1.00, 1.40] learned-only band in
# agents.py / engine.py governs factors those modules learn and ship,
# not this explicit caller parameter. The check reuses
# _is_finite_number (bool / non-numeric excluded), so a bool or
# non-numeric factor raises this named ValueError instead of a bare
# TypeError ('1.25', None pre-fix) or a silent True == 1.0 identity
# pool — the overlap with SF-BL-10965's fixtures is deliberate and
# triggers that task's stated kill criterion. Validation runs at
# function ENTRY, before branch selection (agents.py aggregate /
# SF-BL-8875 precedent: a malformed factor raises before a pool ships
# instead of being dropped by a branch that never reads it). Pure
# validation — it never transforms its input — so every in-band factor,
# the 1.25 default included, pools bit-identically to pre-fix.

def _is_valid_extremizing_a(a: object) -> bool:
    """True iff extremizing_a is a finite real number strictly > 0."""
    return _is_finite_number(a) and a > 0  # type: ignore[operator]


def _validate_extremizing_a(a: object) -> None:
    """Raise ValueError unless extremizing_a is in the SF-BL-9476 band."""
    if not _is_valid_extremizing_a(a):
        raise ValueError(
            f"extremizing_a {a!r} outside pre-registered domain "
            "(0, finite] (SF-BL-9476): a non-finite factor saturates "
            "the pooled logit at the 0.98/0.02 clip (NaN panel fixture: "
            "uniform 0.70 pooled to 0.98), zero collapses the pool to "
            "0.5, and a negative factor rank-inverts the panel "
            "(a=-2.0 fixture: uniform 0.70 pooled to 0.138)")


# --- SF-BL-10761: Gate 2 dispersion is a survivorship statistic -----------
# aggregate_fleet_forecasts drops non-finite / out-of-range members
# (SF-BL-9439 prob, SF-BL-9506 weight, SF-BL-10723 range) and returns
# pstdev over the KEPT members only — 0.0 by construction for a single
# survivor — yet pre-fix the drops surfaced only as warnings and the
# return was the bare (calibrated, dispersion) 2-tuple, so Gate 2 in
# evaluate_and_size_opportunity gated on a panel that might no longer
# exist, and a drop flipping distinct_families below 3 (weighted
# log-odds branch -> median branch) was likewise invisible. Fix: the
# aggregate return now carries a PanelProvenance (counts below), and
# evaluate gains Gate 2C, which SKIPs whenever supplied provenance has
# n_dropped > 0 or n_kept < 3 (see PREREG.md in the task work dir;
# fire-rate restatement in forward_panel_restatement.md there).
#
# --- SF-BL-10849: a duplicated agent_id is not an independent member -----
# aggregate_fleet_forecasts filtered only finiteness / range / weight
# (SF-BL-9439/9506/10723) and never identity, so submitting one agent
# twice counted it as two independent members: the TASK panel
# [a1/famA 0.90, a2/famB 0.50, a3/famC 0.50] pooled (0.7288, 0.1886),
# but the same panel with a1's identical forecast submitted twice
# pooled (0.8150, 0.2000) — the duplicate doubles a1's log-odds
# weight, inflates n and shifts dispersion toward the Gate 2
# threshold (verified live pre-fix, repro.py in the task work dir).
# Caller census (CALLER_CENSUS.md there; kill criterion NOT met): the
# only in-repo production caller passes a plain list literal, no
# panel anywhere is built from a set keyed by agent_id, the public
# API takes a List, and SubAgentForecast is a plain mutable dataclass
# with no __post_init__ — so the fix belongs here, at the consumption
# point. Fix: dedupe by agent_id BEFORE every other filter, on raw
# input order — first occurrence kept, later copies dropped-and-logged
# exactly like the SF-BL-9439 filter (UserWarning naming SF-BL-10849,
# the duplicated agent_ids and the final pooled n). Dedupe on raw
# order means a duplicate of a member whose first copy is itself
# dropped by a later filter does NOT resurrect that agent. Duplicate
# drops are counted in PanelProvenance (n_dropped includes them, new
# n_duplicate_dropped field), so Gate 2C (SF-BL-10761) treats a
# duplicated panel as culled, like any other drop. Unique-agent
# panels are byte-identical and warn-free.


@dataclass(frozen=True)
class PanelProvenance:
    """Panel composition behind one aggregate_fleet_forecasts result
    (SF-BL-10761): how many members entered, how many were kept, how
    many were dropped per reason, and the distinct-family counts on
    both sides of the drops (a drop can flip the pooling branch)."""
    n_input: int
    n_kept: int
    n_dropped: int
    n_prob_dropped: int      # SF-BL-9439 non-finite probability
    n_weight_dropped: int    # SF-BL-9506 non-finite weight
    n_range_dropped: int     # SF-BL-10723 out-of-range probability
    distinct_families_input: int
    distinct_families_kept: int
    # SF-BL-10848 non-bool has_independent_search; trailing with a
    # default so pre-SF-BL-10848 positional constructions (8 fields)
    # keep working unchanged.
    n_flag_dropped: int = 0
    n_duplicate_dropped: int = 0  # SF-BL-10849 duplicate agent_id (default keeps pre-10849 constructions valid)

    @property
    def branch_flipped(self) -> bool:
        """True iff the drops moved the panel across the >=3-family
        weighted log-odds pooling threshold."""
        return (self.distinct_families_input >= 3
                and self.distinct_families_kept < 3)


class FleetAggregation(tuple):
    """Return of aggregate_fleet_forecasts (SF-BL-10761).

    A 2-tuple (calibrated_probability, dispersion) — byte-identical to
    the pre-SF-BL-10761 return for every panel, so every historical
    caller (2-unpack, indexing, tuple equality) is untouched — carrying
    the panel provenance as attributes (the os.stat_result pattern):
    .calibrated_probability, .dispersion and .provenance (a
    PanelProvenance). The dropped-member count is therefore threaded
    through the aggregate return itself, not through a side channel,
    without breaking the 2-tuple contract SF-BL-9439/9506 pinned."""
    # tuple subclass: instances carry a __dict__ for the attributes.

    def __new__(cls, calibrated_probability: float, dispersion: float,
                provenance: PanelProvenance):
        self = super().__new__(
            cls, (calibrated_probability, dispersion))
        self.calibrated_probability = calibrated_probability
        self.dispersion = dispersion
        self.provenance = provenance
        return self

    def __repr__(self) -> str:
        return (f"FleetAggregation(calibrated_probability="
                f"{self.calibrated_probability!r}, dispersion="
                f"{self.dispersion!r}, provenance={self.provenance!r})")


def _is_valid_panel_count(value: object) -> bool:
    """True iff value is a non-negative int count (bool excluded)."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def aggregate_fleet_forecasts(
    forecasts: List[SubAgentForecast],
    extremizing_a: float = 1.25,
    platt_alpha: float = 1.08,
    platt_beta: float = 0.0,
    min_p: float = 0.02,
    max_p: float = 0.98,
) -> FleetAggregation:
    """
    Combines sub-agent forecasts based on information diversity (Satopaa et al., 2014; 2016):
      - If agents ran independent retrieval across distinct model families, applies weighted
        log-odds pooling with extremizing factor `a`.
      - If agents share the same context window, uses the trimmed mean / median (Halawi et al., 2024).
    Members after the first with a repeated agent_id are dropped-and-logged
    before pooling (SF-BL-10849, first occurrence kept), as are members whose
    raw_probability is non-finite (NaN, +-inf, non-numeric; SF-BL-9439), members whose
    historical_brier_weight is non-finite (SF-BL-9506), members whose
    raw_probability is finite but outside [0, 1] (SF-BL-10723) and members
    whose has_independent_search is not a strict bool (SF-BL-10848 — a
    serialized 'False' string is truthy and would otherwise flip the pool
    from the median branch to extremized log-odds); the pool, dispersion
    and branch selection use the remaining members only, and each warning
    issued on a drop reports the final pooled n. Raises ValueError if no
    member with an in-range finite probability, finite weight and
    strict-bool independence flag remains. If the kept
    members' weights total exactly 0.0 in the weighted log-odds branch,
    the pool falls back to equal-weight log-odds with a UserWarning
    (SF-BL-8901) instead of dividing by zero. Raises ValueError if
    extremizing_a is not a finite real number strictly > 0 (SF-BL-9476),
    checked at entry on every branch. Raises ValueError if platt_alpha
    is not a finite real number strictly > 0 or platt_beta is not a
    finite real number (SF-BL-9508), also checked at entry on every
    branch, before member processing. Raises ValueError unless
    min_p / max_p are finite reals with 0 <= min_p < max_p <= 1
    (SF-BL-9713), also checked at entry on every branch, before member
    processing — the rejection is additionally inherited through the
    internal platt_recalibrate / clip_probability calls.
    Returns:
      FleetAggregation(calibrated_probability, inter_agent_std_dev,
      provenance) — the dropped/kept panel counts are threaded through
      the return as a PanelProvenance (SF-BL-10761), so Gate 2 in
      evaluate_and_size_opportunity can gate on the panel that actually
      produced the dispersion instead of the survivorship scalar alone.
    """
    _validate_extremizing_a(extremizing_a)
    _validate_platt_params(platt_alpha, platt_beta, "platt_alpha", "platt_beta")
    _validate_clip_bounds(min_p, max_p)
    if not forecasts:
        raise ValueError("Forecast list cannot be empty.")

    # SF-BL-10849: dedupe by agent_id FIRST, on raw input order — first
    # occurrence kept, later copies are duplicate drops. Identity is
    # compared by equality against the seen list (panels are small; the
    # list scan also tolerates an unhashable agent_id a set would crash
    # on). Dedupe precedes every validity filter, so a duplicate never
    # resurrects an agent whose first copy is itself dropped below.
    _n_input_original = len(forecasts)
    _seen_agent_ids = []
    duplicate_dropped = []
    deduped = []
    for _f in forecasts:
        if any(_f.agent_id == _seen for _seen in _seen_agent_ids):
            duplicate_dropped.append(_f)
        else:
            _seen_agent_ids.append(_f.agent_id)
            deduped.append(_f)
    # SF-BL-9439 + SF-BL-9506 + SF-BL-10723 + SF-BL-10848: drop-and-log
    # invalid members BEFORE any clipping, dispersion, branch selection,
    # weighting or pooling. A member bad on any field is dropped once
    # (probability first: out-of-range before non-finite, so an int
    # beyond float range is classified by SF-BL-10723's exact comparison
    # instead of reaching the finiteness screen's float conversion; the
    # SF-BL-10848 strict-bool flag screen runs last, so a member already
    # dropped for its probability or weight keeps that classification);
    # all warnings report the FINAL pooled n over the original list.
    range_dropped = [f for f in deduped
                     if _is_out_of_range_member_probability(f.raw_probability)]
    range_kept = [f for f in deduped
                  if not _is_out_of_range_member_probability(f.raw_probability)]
    prob_dropped = [f for f in range_kept
                    if not _is_finite_member_probability(f.raw_probability)]
    prob_kept = [f for f in range_kept
                 if _is_finite_member_probability(f.raw_probability)]
    weight_dropped = [f for f in prob_kept
                      if not _is_finite_member_weight(f.historical_brier_weight)]
    weight_kept = [f for f in prob_kept
                   if _is_finite_member_weight(f.historical_brier_weight)]
    flag_dropped = [f for f in weight_kept
                    if not _is_strict_bool_flag(f.has_independent_search)]
    kept = [f for f in weight_kept
            if _is_strict_bool_flag(f.has_independent_search)]
    if duplicate_dropped:
        warnings.warn(
            "SF-BL-10849: dropped "
            f"{len(duplicate_dropped)} duplicate member forecast(s) "
            f"[{', '.join(f'{f.agent_id}={f.raw_probability!r}' for f in duplicate_dropped)}]; "
            f"pooling n={len(kept)} of {len(forecasts)}",
            UserWarning, stacklevel=2)
    if prob_dropped:
        warnings.warn(
            "SF-BL-9439: dropped "
            f"{len(prob_dropped)} non-finite member forecast(s) "
            f"[{', '.join(f'{f.agent_id}={f.raw_probability!r}' for f in prob_dropped)}]; "
            f"pooling n={len(kept)} of {len(forecasts)}",
            UserWarning, stacklevel=2)
    if weight_dropped:
        warnings.warn(
            "SF-BL-9506: dropped "
            f"{len(weight_dropped)} non-finite-weight member forecast(s) "
            f"[{', '.join(f'{f.agent_id}={f.historical_brier_weight!r}' for f in weight_dropped)}]; "
            f"pooling n={len(kept)} of {len(forecasts)}",
            UserWarning, stacklevel=2)
    if range_dropped:
        warnings.warn(
            "SF-BL-10723: dropped "
            f"{len(range_dropped)} out-of-range member forecast(s) "
            f"[{', '.join(f'{f.agent_id}={f.raw_probability!r}' for f in range_dropped)}]; "
            f"pooling n={len(kept)} of {len(forecasts)}",
            UserWarning, stacklevel=2)
    if flag_dropped:
        warnings.warn(
            "SF-BL-10848: dropped "
            f"{len(flag_dropped)} non-bool-flag member forecast(s) "
            f"[{', '.join(f'{f.agent_id}={f.has_independent_search!r}' for f in flag_dropped)}]; "
            f"pooling n={len(kept)} of {len(forecasts)}",
            UserWarning, stacklevel=2)
    if not kept:
        raise ValueError(
            "Forecast list has no member with an in-range finite "
            "probability, finite weight and strict-bool independence "
            "flag (all "
            f"{len(forecasts)} dropped: "
            f"{len(duplicate_dropped)} duplicate agent_id, SF-BL-10849; "
            f"{len(range_dropped)} out-of-range probability, SF-BL-10723; "
            f"{len(prob_dropped)} probability, SF-BL-9439; "
            f"{len(weight_dropped)} weight, SF-BL-9506; "
            f"{len(flag_dropped)} non-bool independence flag, SF-BL-10848).")
    forecasts = kept

    raw_probs = [clip_probability(f.raw_probability, 0.005, 0.995) for f in forecasts]
    dispersion = statistics.pstdev(raw_probs) if len(raw_probs) > 1 else 0.0

    distinct_families = len({f.model_family for f in forecasts})
    # SF-BL-10848: every kept member's flag is a strict bool (the flag
    # screen above dropped the rest), so this all(...) now reads the
    # declared bool values — never the truthiness of a serialized
    # stand-in such as the string 'False'.
    independent_search = all(f.has_independent_search for f in forecasts)

    if distinct_families >= 3 and independent_search:
        # Weighted log-odds pooling with extremizing (Satopaa et al., 2014)
        total_w = sum(f.historical_brier_weight for f in forecasts)
        if total_w == 0.0:
            # SF-BL-8901: no track-record information to weight by (fresh
            # panel, every weight 0.0, or finite weights cancelling to
            # 0.0) — fall back to equal-weight log-odds rather than
            # dividing by zero.
            warnings.warn(
                "SF-BL-8901: total historical_brier_weight is 0.0 over "
                f"{len(forecasts)} independent member(s) "
                f"[{', '.join(f.agent_id for f in forecasts)}]; "
                "falling back to equal-weight log-odds pooling",
                UserWarning, stacklevel=2)
            weighted_logit = (
                sum(logit(p) for p in raw_probs) / len(raw_probs))
        else:
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
    # SF-BL-10761: thread the panel provenance through the return. The
    # counts describe the ORIGINAL list (before `forecasts = kept`):
    # n_dropped is every drop reason combined, and the family counts
    # record whether the drops flipped the pooling branch.
    _provenance = PanelProvenance(
        n_input=_n_input_original,
        n_kept=len(kept),
        n_dropped=len(prob_dropped) + len(weight_dropped)
        + len(range_dropped) + len(duplicate_dropped)
        + len(flag_dropped),
        n_prob_dropped=len(prob_dropped),
        n_weight_dropped=len(weight_dropped),
        n_range_dropped=len(range_dropped),
        distinct_families_input=len(
            {f.model_family for f in
             duplicate_dropped + range_dropped + prob_dropped
             + weight_dropped + flag_dropped + kept}),
        distinct_families_kept=distinct_families,
        n_duplicate_dropped=len(duplicate_dropped),
        n_flag_dropped=len(flag_dropped),
    )
    return FleetAggregation(
        calibrated_probability=calibrated,
        dispersion=dispersion,
        provenance=_provenance,
    )


# ============================================================================
# 3. CENTRAL LIMIT ORDER BOOK (CLOB) SWEEP & SLIPPAGE SIMULATOR
# ============================================================================

# --- SF-BL-9473: derived NO prices are validated in the sweep --------------
# The BUY_NO ladder is derived as round(1.0 - yes_bid.price, 6) with, pre-fix,
# no range re-check, so any YES bid outside (0, 1) priced NO outside (0, 1):
# a bid of 1.50 (1,000 shares, fee 0.01) swept to invested_capital -505.00
# and q_exec -0.505 — the negative rung "spend" refunded budget mid-sweep
# and invented shares, corrupting even a mixed ladder's valid fills (the
# benchmark unlock ladder behind a 1.50 rung executed through evaluate at
# 419.15 / 3,400 shares / q_exec 0.1233 instead of its clean 924.15 /
# 2,400 / 0.3850625). The full derived-price failure set, all verified
# live pre-fix: bid 1.0 -> ZeroDivisionError (derived price 0.0); bid 0.0
# -> q_exec 1.01; bid -0.30 -> q_exec 1.313; bid 2.0 -> invested -1010.00;
# bid NaN -> NaN shares/q_exec on a finite spend; bid +inf -> invested
# -inf. Reachability (CALLER_CENSUS.md in the task work dir; kill
# criterion NOT met): OrderBookLevel carries no price validation and
# OrderBookSnapshot.__post_init__ checks only fee_rate, so both direct
# construction and the post-construction mutation / unpickle path the
# SF-BL-9363 fee fix guards against in this same function deliver bad
# bids to the sweep; SF-BL-9051 covers constructor-time validation only.
# Callers include evaluate_and_size_opportunity (whose edge screen reads
# the equally derived best_no_ask, so a bad bid selects BUY_NO on a
# phantom edge) and the naive baseline in run_benchmark_simulation, which
# calls the sweep directly at min_net_edge=-1.0 and books its return
# unchecked. The fix is at ladder construction, beside the SF-BL-9398
# live-level filter: a rung whose price is not a finite real in the open
# interval (0, 1) is SKIPPED — for BUY_NO the derived price is checked,
# for BUY_YES the raw ask (the same binary-contract domain: a $1-payout
# contract cannot trade at or beyond its bounds on either side; a NaN
# raw ask was originally left to SF-BL-9438's INVALID_EQUITY mechanism,
# a carve-out SF-BL-9542 superseded — see the ladder construction
# below). Skipping
# (not raising) matches the sweep's (0.0, 0.0, 0.0) no-fill contract:
# evaluate maps an all-invalid ladder to its depth-exhausted SKIP with
# zero size and no cluster write, a mixed ladder still fills its valid
# rungs at their true prices, and no caller needs a new exception path.
# A valid price whose fee-adjusted rung price exceeds 1.0 is the distinct
# SF-BL-8891 fee question and is deliberately untouched here (now closed
# by SF-BL-8891 itself: see its block at the sweep loop below — this
# SF-BL-9473 filter still governs the RAW / DERIVED price domain only).

def _is_valid_contract_price(price: object) -> bool:
    """True iff price is a finite real number in the open interval (0, 1)."""
    return _is_finite_number(price) and 0.0 < price < 1.0  # type: ignore[operator]


def _is_nan_price(price: object) -> bool:
    """True iff price is the float NaN (SF-BL-9438's domain, see sweep)."""
    return isinstance(price, float) and math.isnan(price)


# --- SF-BL-3840: calibration-aware (shrunk-LCB) sweep stop ----------------
# The point stop tests each rung against belief_prob_side as a point value,
# ignoring posterior uncertainty that NORTH_STAR's fitted-shrinkage policy
# already quantifies for quant streams; deep rungs are exactly where a
# point-estimate stop overpays. The opt-in bounded stop below shrinks the
# belief toward the side's market-mid prior with the fleet's predmarkets
# pseudo-count convention (w = n/(n+25)) and subtracts a normal-approx
# standard error, yielding a lower confidence bound p_stop; the rung test
# then uses p_stop instead of the point belief. It is OPT-IN
# (posterior_n_eff=None default): every existing caller keeps byte-
# identical point-p behaviour unless it explicitly supplies an effective
# sample size. Full pre-registration: backlog work dir SF-BL-3840/PREREG.md
# (primary config n_eff=30, z=1.0, K=25 — judged there on replayed forward
# sweeps at matched capital; sensitivity rows cannot rescue the primary).
LCB_SHRINK_K = 25.0


# --- SF-BL-12181: shrunk_lcb never type-validated its arguments -------
# shrunk_lcb assumes its inputs are "already validated", but that is only
# true on the sweep_order_book path (which screens its own operands).
# The function is public, and a direct caller reaching it with a
# corrupt-typed belief was silently coerced instead of named — verified
# live pre-fix (repro.py in the task work dir): p=True returned exactly
# the p=1.0 bound (0.7112802265126448 on the repro fixture), prior=True
# likewise, n_eff='30' / shrink_k='25' / z='1.0' and n_eff=True /
# shrink_k=False / z=True all returned the clean bound via the body's
# float() coercions of n_eff / shrink_k / z, and p='0.65' / prior='0.49'
# crashed with a bare TypeError naming no argument (raw arithmetic, no
# float() to coerce them). Kill-criterion audit (CALLER_CENSUS.md in
# the task work dir): NOT met — the sweep's guards stand in front of
# the sweep path only; no caller sanitises direct calls. The fix is a
# TYPE screen at shrunk_lcb's own entry, in signature order: any bool
# or non-(int/float) value for p, prior, n_eff, z or shrink_k raises a
# ValueError naming that argument. Scope is deliberately type-only per
# this task's prereg: finiteness / range are separate surfaces (the
# sweep enforces its own on its path), so every int/float call —
# NaN/inf and plain ints included — flows into the unchanged body
# arithmetic bit-identically.

def _validate_shrunk_lcb_types(
    p: object, prior: object, n_eff: object, z: object, shrink_k: object
) -> None:
    """Raise ValueError naming the argument unless every shrunk_lcb
    argument is a real number (int/float, bool excluded) (SF-BL-12181).
    """
    for name, value in (
        ("p", p), ("prior", prior), ("n_eff", n_eff),
        ("z", z), ("shrink_k", shrink_k),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"shrunk_lcb argument '{name}'={value!r} (type "
                f"{type(value).__name__}) is not a real number — "
                "INVALID_SHRUNK_LCB_ARGUMENT (SF-BL-12181): the "
                "shrunk lower confidence bound consumed by the fleet's "
                "sizing gates must not silently coerce a corrupt-typed "
                "input (pre-fix: p=True returned the p=1.0 bound and "
                "z='1.0' coerced through float() to the clean bound; "
                "a string p/prior crashed with a bare TypeError)")


def shrunk_lcb(
    p: float,
    prior: float,
    n_eff: float,
    z: float = 1.0,
    shrink_k: float = LCB_SHRINK_K,
) -> float:
    """Shrunk lower confidence bound of a side belief (SF-BL-3840).

    p_shrunk = (n_eff * p + shrink_k * prior) / (n_eff + shrink_k)
    se       = sqrt(p_shrunk * (1 - p_shrunk) / (n_eff + shrink_k))
    returns clip(p_shrunk - z * se, 0.0, 1.0). Argument TYPES are
    validated at entry (SF-BL-12181: a bool or non-numeric p, prior,
    n_eff, z or shrink_k raises a ValueError naming the argument);
    finiteness/range remain the caller's responsibility (on the
    sweep_order_book path, its guards).
    """
    _validate_shrunk_lcb_types(p, prior, n_eff, z, shrink_k)
    n = float(n_eff)
    k = float(shrink_k)
    p_shrunk = (n * p + k * prior) / (n + k)
    se = math.sqrt(max(0.0, p_shrunk * (1.0 - p_shrunk)) / (n + k))
    return max(0.0, min(1.0, p_shrunk - float(z) * se))


# --- SF-BL-10502: percent-unit belief sweeps the full budget --------------
# belief_prob_side is a probability, so its valid domain is the finite
# interval [0, 1]. Pre-fix the sweep checked only finiteness (SF-BL-9471,
# which returned (0, 0, 0) on non-finite operands) and imposed NO range
# check: the rung stop compares each rung's fee-adjusted price against
# belief_prob_side, so a finite percent-unit belief clears every rung —
# verified live pre-fix (repro.py in the task work dir): on a book with
# a single 0.90 ask at fee_rate 0, belief_prob_side=82.0 invested the
# full $100 budget (111.11 shares at VWAP 0.90), while the clean belief
# 0.6 correctly stopped at (0, 0, 0) on the same book. A percent-unit
# belief disables the marginal-edge stop that is the sweep's only
# execution guard (the SF-BL-8891 parity stop bounds only rungs at/above
# $1.00; a 0.90 rung is legitimately tradeable for a real belief in
# (0.95, 1]). Finite out-of-range beliefs in the other direction
# (-0.2) stopped only incidentally. The fix raises a named ValueError
# BEFORE walking, per this task's prereg — superseding SF-BL-9471's
# return-(0, 0, 0) convention for these two operands: a silently
# zeroed sweep is indistinguishable from a genuine no-edge stop in
# caller telemetry, while a percent-unit belief is a producer unit
# error that must surface. Caller census (CALLER_CENSUS.md in the task
# work dir): no sweep caller sanitises belief_prob_side to [0, 1] —
# evaluate passes its Gate-0.0-validated/clipped p_side (safe by
# construction), but the naive baseline and budget_sensitivity_check
# call the sweep directly with bare values and book its return
# unchecked, so the check belongs here, at the consumption point.
# min_net_edge is validated on the same call: non-finite hurdles raise
# (SF-BL-9471's class), and the sweep-level hurdle domain is
# [-1.0, 1.0] — a hurdle is an edge difference p - q whose attainable
# magnitude on [0, 1] beliefs/prices is 1, and the documented -1.0
# full-walk override (naive baseline, budget_sensitivity_check, the
# SF-BL-9471 finite grid at -1.0 / -0.10) sits at its lower bound, so
# every in-repo finite hurdle is in domain and bit-identical.
# FleetRiskConfig's stricter >= 0 config domain (SF-BL-9397) is a
# separate surface and is untouched. Valid beliefs (boundaries 0 and
# 1 included, int accepted) and in-domain hurdles fill bit-identically.

BELIEF_PROB_SIDE_MIN = 0.0
BELIEF_PROB_SIDE_MAX = 1.0
SWEEP_MIN_NET_EDGE_MIN = -1.0
SWEEP_MIN_NET_EDGE_MAX = 1.0


def _belief_prob_side_in_domain(value: object) -> bool:
    """True iff belief_prob_side is a finite real in [0, 1] (SF-BL-10502)."""
    return _is_finite_number(value) and BELIEF_PROB_SIDE_MIN <= value <= BELIEF_PROB_SIDE_MAX  # type: ignore[operator]


def _sweep_min_net_edge_in_domain(value: object) -> bool:
    """True iff the sweep hurdle is a finite real in [-1.0, 1.0] (SF-BL-10502)."""
    return _is_finite_number(value) and SWEEP_MIN_NET_EDGE_MIN <= value <= SWEEP_MIN_NET_EDGE_MAX  # type: ignore[operator]


def _validate_sweep_belief_and_edge(
    belief_prob_side: object, min_net_edge: object
) -> None:
    """Raise ValueError unless the sweep stop operands are in domain.

    belief_prob_side is validated before min_net_edge (SF-BL-10502).
    Bool / non-numeric values raise this same ValueError, not a bare
    TypeError from the stop comparison.
    """
    if not _belief_prob_side_in_domain(belief_prob_side):
        raise ValueError(
            f"belief_prob_side {belief_prob_side!r} is not a finite "
            "probability in [0, 1] — INVALID_BELIEF_PROB_SIDE "
            "(SF-BL-10502): the marginal-edge stop compares each rung "
            "against this belief, so a percent-unit belief (e.g. 82.0 "
            "for 82%) clears every rung and sweeps the full budget "
            "(pre-fix: $100 at a single 0.90 ask filled 111.11 shares "
            "where the clean belief 0.6 stopped at no fill); refusing "
            "to walk the book on a belief that is not a probability")
    if not _sweep_min_net_edge_in_domain(min_net_edge):
        raise ValueError(
            f"min_net_edge {min_net_edge!r} is not a finite real in "
            "[-1.0, 1.0] — INVALID_MIN_NET_EDGE (SF-BL-10502): the "
            "marginal-edge stop cannot be evaluated against a "
            "non-finite or out-of-range hurdle (an edge difference "
            "p - q attains at most magnitude 1; the documented -1.0 "
            "full-walk override is the lower bound)")


# --- SF-BL-10964: a non-OrderBookSnapshot book crashes on fee_rate -----
# sweep_order_book dereferenced book.fee_rate as its first act (the
# SF-BL-9363 re-validation), so a book that is not an OrderBookSnapshot
# at all crashed with a bare AttributeError naming an implementation
# detail instead of the argument: verified live pre-fix (repro.py in
# the task work dir) book=None raised AttributeError "'NoneType' object
# has no attribute 'fee_rate'" and book=dict (empty, or carrying a
# fee_rate key) raised the same for 'dict', as did str / int / list and
# an OrderBookLevel passed as the book. The function's other invalid
# inputs fail closed or raise named errors (non-finite budget returns
# (0.0, 0.0, 0.0), SF-BL-9438; invalid belief/hurdle raise
# INVALID_BELIEF_PROB_SIDE / INVALID_MIN_NET_EDGE, SF-BL-10502; invalid
# fee / unsorted ladder on a real book raise ValueError, SF-BL-9363 /
# 9051) — a missing or JSON-decoded book is an ordinary producer error
# (a lookup miss returns None; a decoded snapshot is a dict) and must
# surface the same way, as a named ValueError, before any dereference.
# The check is a strict isinstance, not duck typing: pre-fix an
# attribute-compatible fake (fee_rate / yes_asks / yes_bids / mid_price
# attributes, not an OrderBookSnapshot) swept successfully
# (1000.0, 1980.198, 0.505) in repro.py, so attribute presence was an
# implicit, untested contract — and such a fake bypasses every
# __post_init__ validation (fee bound, ladder order, est-days,
# on-chain flag) the real dataclass enforces. Subclasses pass
# (isinstance semantics), exactly as a genuine book does.
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT
# met — no caller sanitises the book argument and no tested invariant
# stands in front of it: evaluate_and_size_opportunity and
# budget_sensitivity_check pass their own book arguments straight
# through, the benchmark naive baseline passes item["book"] from
# scenario dict literals and books the return unchecked, direct
# test/audit callers pass bare variables, and the
# `book: OrderBookSnapshot` annotation is unenforced at runtime.
# SF-BL-10910 fixtures the sweep's SIDE argument only; this is the
# BOOK argument of the same function.

def _is_order_book_snapshot(book: object) -> bool:
    """True iff book is an OrderBookSnapshot (subclasses included)."""
    return isinstance(book, OrderBookSnapshot)


def _validate_sweep_book(book: object) -> None:
    """Raise ValueError unless book is an OrderBookSnapshot (SF-BL-10964).

    Runs before ANY dereference of the book, so a None / dict / other
    non-book value raises this named ValueError, never the bare
    AttributeError the pre-fix `book.fee_rate` read produced.
    """
    if not _is_order_book_snapshot(book):
        raise ValueError(
            f"book {book!r} (type {type(book).__name__}) is not an "
            "OrderBookSnapshot — INVALID_BOOK (SF-BL-10964): "
            "sweep_order_book walks a constructed book's ladders and "
            "fee_rate; a missing book (None, e.g. a lookup miss) or a "
            "decoded/unconstructed value (e.g. a dict) has no ladders "
            "to sweep and must not crash on an internal attribute "
            "read (pre-fix: AttributeError on book.fee_rate) or fill "
            "via duck-typed attributes that bypassed OrderBookSnapshot "
            "construction validation")


# --- SF-BL-11045: sweep scalar TYPE errors read as genuine no-fill -----
# sweep_order_book's three scalar arguments are the budget and the two
# marginal-edge stop operands. _is_finite_number rejects bool /
# non-numeric values, so the SF-BL-9438/9471 fail-closed guards lumped
# TYPE errors together with numeric non-finiteness: verified live
# pre-fix (repro.py in the task work dir) on a book that fills $1,000
# clean — (1000.0, 1980.198, 0.505) — target_budget='1000',
# target_budget=True, belief_prob_side='0.8' and min_net_edge=None
# each returned the plain (0.0, 0.0, 0.0) no-fill tuple, indistinguishable
# from genuine depth exhaustion, and direct callers such as the naive
# benchmark baseline book that tuple unchecked as a real no-fill.
# Distinct from SF-BL-9438/9471 (finiteness of finite-typed values) and
# SF-BL-10910 (side canonicality): this is the TYPE dimension.
# Disposition per argument at this run's tree:
#   * belief_prob_side / min_net_edge — ALREADY raise a named
#     ValueError for bool / non-numeric values via SF-BL-10502's
#     _validate_sweep_belief_and_edge (INVALID_BELIEF_PROB_SIDE /
#     INVALID_MIN_NET_EDGE), which landed after this task was filed;
#     repro.py confirms '0.8' / True / None raise there. Untouched.
#   * target_budget — the remaining gap: its only guard was the
#     SF-BL-9438 finiteness return, so a bool or non-numeric budget
#     still read as no-fill. The TYPE is now validated here, before
#     that return: a bool or non-numeric budget raises a named
#     ValueError (INVALID_TARGET_BUDGET_TYPE). TYPE ONLY — the
#     predicate is the module's standing discipline (int/float, bool
#     excluded, never coerced: '1000' is never parsed and True is
#     never read as a $1 budget). Every NUMERIC budget keeps its
#     pre-fix behaviour bit-identically: finite budgets sweep as
#     before (int budgets included), zero/negative budgets return
#     (0.0, 0.0, 0.0) at the existing early return, and non-finite
#     numeric budgets (NaN / +-inf / a huge int beyond float range)
#     keep SF-BL-9438's (0.0, 0.0, 0.0) return — the pinned contract
#     of tests/test_alpha_kelly_invalid_equity.py and the SF-BL-10502 /
#     SF-BL-10964 suites.
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT
# met — no caller passes only validated floats under a tested
# invariant: evaluate_and_size_opportunity derives its budget from
# its own (separately gated) equity, but budget_sensitivity_check
# forwards a bare target_budget argument, the naive baseline and the
# test suites pass bare literals/variables, the
# `target_budget: float` annotation is unenforced at runtime, and a
# JSON/CLOB producer naturally quotes a budget as a string.

def _target_budget_is_numeric_type(value: object) -> bool:
    """True iff target_budget is a real number TYPE (SF-BL-11045):
    an int or float, bool excluded. Type ONLY — finiteness is the
    SF-BL-9438 value surface, untouched here."""
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _validate_target_budget_type(value: object) -> None:
    """Raise ValueError unless target_budget is numeric in type
    (SF-BL-11045)."""
    if not _target_budget_is_numeric_type(value):
        raise ValueError(
            f"target_budget {value!r} (type {type(value).__name__}) "
            "is not a real number — INVALID_TARGET_BUDGET_TYPE "
            "(SF-BL-11045): the sweep budget must be an int/float "
            "(bool excluded) — a bool or non-numeric budget (e.g. "
            "the JSON string '1000' a producer quotes, or True read "
            "as a $1 budget) is a TYPE error, not the non-finite "
            "VALUE case of SF-BL-9438; pre-fix it returned the plain "
            "(0.0, 0.0, 0.0) no-fill tuple, indistinguishable from "
            "genuine depth exhaustion on a book that fills clean; "
            "convert serialized values with float() before calling "
            "— values are never coerced here")


def sweep_order_book(
    book: OrderBookSnapshot,
    side: TradeAction,
    target_budget: float,
    belief_prob_side: float,
    min_net_edge: float = 0.05,
    posterior_n_eff: Optional[float] = None,
    lcb_z: float = 1.0,
    prior_prob_side: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Simulates sweeping the Central Limit Order Book (PolyBench Section 4.2) while
    stopping before any order-book rung where marginal execution price + fees
    erodes the edge below `min_net_edge`.

    The sweep also stops before any rung whose fee-adjusted price is at
    or above $1.00 (SF-BL-8891 parity stop, in the loop below): such a
    rung pays at least the contract's maximum settlement value, so no
    hurdle — including a -1.0 full walk — may fill it, and the returned
    q_exec is therefore always < 1.0 on a fill (0.0 on no fill).

    A non-finite or out-of-domain `belief_prob_side` (finite real in
    [0, 1]) or `min_net_edge` (finite real in [-1.0, 1.0], the -1.0
    full-walk override included) raises ValueError before any rung is
    walked (SF-BL-10502, superseding SF-BL-9471's return-(0, 0, 0) for
    these operands): the marginal-edge stop cannot be evaluated against
    a non-finite operand, and a finite percent-unit belief would clear
    every rung and sweep the full budget.

    SF-BL-3840: when `posterior_n_eff` is supplied (a finite number > 0),
    the stop tests rungs against the shrunk lower confidence bound of the
    belief (see shrunk_lcb) instead of the point belief. `prior_prob_side`
    defaults to the side's market mid (mid for BUY_YES, 1 - mid for
    BUY_NO). Invalid LCB inputs (non-finite n_eff / z / prior, n_eff <= 0,
    z < 0, prior outside [0, 1]) fail closed with (0, 0, 0). With
    `posterior_n_eff=None` (the default) behaviour is unchanged.
    SF-BL-6060: when `prior_prob_side` is defaulted from the book, a
    book with any fallback side (no live tradeable ladder on a side —
    its mid averages a sentinel) fails closed with (0, 0, 0) instead
    of shrinking toward a fallback-derived prior.

    A `book` that is not an OrderBookSnapshot raises ValueError naming
    INVALID_BOOK before any dereference (SF-BL-10964): pre-fix book=None
    and book=dict crashed with a bare AttributeError on book.fee_rate.
    This validation precedes every other guard — including the SKIP /
    zero-budget (0.0, 0.0, 0.0) returns — because there is no book to
    fail closed on; a genuine book with side SKIP keeps that contract.

    Ladder ENTRIES that are not OrderBookLevel at all (a dict, None —
    SF-BL-11046) are skipped identical-to-clean with a logged count
    (one UserWarning naming SF-BL-11046): pre-fix they raised a bare
    AttributeError on size_shares inside _is_live_level and aborted
    the sweep; now a mixed ladder fills exactly as the clean ladder
    without the corrupt entries.

    A bool or non-numeric `target_budget` raises ValueError naming
    INVALID_TARGET_BUDGET_TYPE (SF-BL-11045): pre-fix it returned the
    plain (0.0, 0.0, 0.0) no-fill tuple via the SF-BL-9438 finiteness
    guard, indistinguishable from genuine depth exhaustion. Numeric
    budgets are untouched: non-finite values (NaN / +-inf) keep
    SF-BL-9438's (0.0, 0.0, 0.0) return, and bool / non-numeric
    `belief_prob_side` / `min_net_edge` keep SF-BL-10502's named
    INVALID_BELIEF_PROB_SIDE / INVALID_MIN_NET_EDGE raises.

    Returns:
      (invested_capital_including_fees, shares_acquired, effective_execution_price_per_share)
    """
    # SF-BL-10964: validate the book's type FIRST, before any dereference
    # — the fee re-validation below reads book.fee_rate, which is exactly
    # where a None/dict book crashed pre-fix. Raises ValueError
    # (INVALID_BOOK) on a non-OrderBookSnapshot book.
    _validate_sweep_book(book)
    # SF-BL-9363: re-validate here, not only in OrderBookSnapshot.__post_init__
    # — a book mutated (or unpickled) after construction must never sweep at
    # an out-of-bound fee. Raises ValueError on an invalid fee.
    _validate_fee_rate(book.fee_rate)
    # SF-BL-9051: re-validate ladder order here too — the sweep walks the
    # ladder in presentation order and its parity/edge `break`s assume
    # sorted order, so a book mutated (or unpickled) after construction
    # must never sweep unsorted. Raises ValueError, as for the fee.
    _validate_ladder_order(book.yes_asks, book.yes_bids)
    # SF-BL-11045: validate the budget's TYPE before the finiteness
    # return below — a bool or non-numeric budget raises a named
    # ValueError (INVALID_TARGET_BUDGET_TYPE) instead of reading as a
    # genuine no-fill. Book (SF-BL-10964), fee (SF-BL-9363) and
    # ladder-order (SF-BL-9051) raises above keep their precedence.
    # Numeric budgets fall through bit-identically.
    _validate_target_budget_type(target_budget)
    # SF-BL-9438: a non-finite target_budget (only reachable from non-finite
    # equity upstream, or a direct caller error) fails closed with no fill
    # instead of returning a (NaN, NaN, NaN) tuple for callers to misread —
    # `NaN <= 0` is False, so the old early return below never fired on NaN.
    if not _is_finite_number(target_budget):
        return 0.0, 0.0, 0.0
    # SF-BL-9471 (history): a non-finite belief_prob_side or min_net_edge
    # returned (0, 0, 0) here — the marginal-edge stop in the loop below,
    # `(belief_prob_side - effective_rung_price) < min_net_edge`, is False
    # whenever either operand is NaN, so a non-finite belief or hurdle
    # silently disabled the PolyBench slippage stop and the sweep consumed
    # every rung the budget reached.
    # SF-BL-10502 (superseding): the stop operands are now VALIDATED here,
    # before walking, and raise a named ValueError — belief_prob_side must
    # be a finite probability in [0, 1] and min_net_edge a finite real in
    # [-1.0, 1.0] (see the SF-BL-10502 block above sweep_order_book). A
    # finite percent-unit belief (82.0) passed the old finiteness check and
    # cleared every rung; a return of (0, 0, 0) for the non-finite class
    # also hid a producer unit/type error as a genuine no-edge stop. Fee
    # (SF-BL-9363) and ladder-order (SF-BL-9051) raises above keep their
    # precedence, and the SF-BL-9438 non-finite-budget return above is
    # unchanged. Valid beliefs / hurdles — int 1 / 0 and the naive fleet's
    # -1.0 full-walk override included — are untouched.
    _validate_sweep_belief_and_edge(belief_prob_side, min_net_edge)
    # SF-BL-3840: resolve the stop belief. Default (posterior_n_eff=None)
    # is the point belief, byte-identical to pre-SF-BL-3840 behaviour.
    # Opt-in: validate the LCB inputs and fail closed on any invalid one,
    # then test rungs against the shrunk lower confidence bound.
    stop_belief = belief_prob_side
    if posterior_n_eff is not None:
        if not _is_finite_number(posterior_n_eff) or posterior_n_eff <= 0:  # type: ignore[operator]
            return 0.0, 0.0, 0.0
        if not _is_finite_number(lcb_z) or lcb_z < 0:  # type: ignore[operator]
            return 0.0, 0.0, 0.0
        if prior_prob_side is None:
            # SF-BL-6060: the default prior is the book's mid — but on
            # a book with any fallback side that mid averages a real
            # price with the 0.99 / 0.01 sentinel (0.255 on an ask-only
            # book, 0.50 on an empty one), so the shrunk stop belief
            # would be pulled toward a price that was never a measured
            # touch (pre-fix the ask-only fixture filled $505.00 where
            # the real prior filled $5,000.00 on the same ladder).
            # Fail closed with (0, 0, 0), the SF-BL-3840 convention
            # for unusable LCB inputs: no fallback-derived prior may
            # size a sweep. An explicit prior_prob_side is the
            # caller's own value, not a fallback, and is untouched.
            if _fallback_price_sides(book):
                return 0.0, 0.0, 0.0
            prior_prob_side = (
                book.mid_price if side == TradeAction.BUY_YES
                else 1.0 - book.mid_price
            )
        if not _posterior_in_bounds(prior_prob_side):
            return 0.0, 0.0, 0.0
        stop_belief = shrunk_lcb(
            belief_prob_side, prior_prob_side, posterior_n_eff, lcb_z)  # type: ignore[arg-type]
    if target_budget <= 0 or side == TradeAction.SKIP:
        return 0.0, 0.0, 0.0

    # Construct effective ask ladder for the chosen side.
    # SF-BL-9398: live levels only — a zero/negative-size rung must neither
    # be swept (it fills nothing) nor trigger the marginal-edge break below
    # (pre-fix, a 0-share rung at 0.90 in front of a real 0.50 rung made the
    # sweep return (0, 0, 0) and block a fillable trade).
    if side == TradeAction.BUY_YES:
        # SF-BL-9473: live levels with an in-domain price only — a raw ask
        # carrying an ordered price outside (0, 1) (including +-inf and
        # non-numeric values) is the same $1-payout domain violation as a
        # bad derived NO price and its rung is skipped.
        # SF-BL-9542 SUPERSESSION: the NaN-ask carve-out that SF-BL-9473
        # left here (a NaN ask passed through so the sweep returned its
        # NaN tuple for SF-BL-9438's evaluate guard) is removed — a NaN
        # ask in front of a valid rung returned (budget, NaN, NaN) to
        # direct sweep callers, who book the tuple unchecked, so under
        # SF-BL-9542's kill criterion (no negative, zero, NaN or >1 ask
        # price may alter invested, shares or q_exec) the NaN rung is
        # skipped in the ladder like every other invalid price, making
        # the sweep identical to the clean book. SF-BL-9438's pinned
        # evaluate outcome is preserved one level up: evaluate's Gate 0.6
        # (SF-BL-9542) fails a book carrying any live NaN-priced rung
        # closed as INVALID_EQUITY before sizing, cluster untouched, and
        # the post-sweep non-finite guard remains as a backstop.
        ladder = [
            lvl for lvl in _live_levels(book.yes_asks)
            if _is_valid_contract_price(lvl.price)
        ]
    elif side == TradeAction.BUY_NO:
        # Buying NO sweeps the YES bids in reverse (price of NO = 1.0 - yes_bid.price)
        # SF-BL-9473: the derived price is re-checked against (0, 1) here —
        # a YES bid at/above 1.0, at/below 0, or non-finite derives a NO
        # price that is negative, zero (division by zero in the sweep),
        # at/above the $1 maximum payout, or NaN, and its rung is skipped.
        ladder = []
        for lvl in _live_levels(book.yes_bids):
            if not _is_finite_number(lvl.price):
                continue
            derived_price = round(1.0 - lvl.price, 6)  # type: ignore[operator]
            if _is_valid_contract_price(derived_price):
                ladder.append(
                    OrderBookLevel(price=derived_price, size_shares=lvl.size_shares)
                )
    else:
        return 0.0, 0.0, 0.0

    # SF-BL-11046: log the non-OrderBookLevel entries skipped from the
    # ladder just walked (the prereg's "skips ... with a logged count"
    # branch). _is_live_level now filters them identical-to-clean, so
    # the fill above is exactly the clean ladder's; this warning — the
    # module's drop-and-log convention (cf. SF-BL-9439) — is the only
    # trace, naming the count and the offending types so a producer
    # feeding decoded/unconstructed rungs is visible instead of
    # silently absorbed. Only the SWEPT ladder is counted (asks for
    # BUY_YES, bids for BUY_NO): the other ladder's entries are not
    # this sweep's to skip. A ladder of genuine levels emits nothing,
    # so valid books are byte- and warning-identical.
    _swept_source = book.yes_asks if side == TradeAction.BUY_YES else book.yes_bids
    if isinstance(_swept_source, (list, tuple)):
        _skipped_entries = [
            e for e in _swept_source if not isinstance(e, OrderBookLevel)
        ]
        if _skipped_entries:
            warnings.warn(
                "SF-BL-11046: skipped "
                f"{len(_skipped_entries)} non-OrderBookLevel ladder "
                f"entr{'y' if len(_skipped_entries) == 1 else 'ies'} "
                f"in sweep_order_book "
                f"[types: {', '.join(type(e).__name__ for e in _skipped_entries)}]; "
                "filling from the remaining OrderBookLevel rungs "
                "identical to the clean ladder",
                UserWarning, stacklevel=2)

    remaining_budget = target_budget
    total_cost_with_fees = 0.0
    total_shares = 0.0

    for rung in ladder:
        if remaining_budget <= 1e-6:
            break
        effective_rung_price = rung.price * (1.0 + book.fee_rate)
        # SF-BL-9472: zero-price division guard. The prereg crash lived
        # at the division below: a rung at price 0.0 gives
        # effective_rung_price 0.0, max_rung_cost 0.0 and spend_on_rung
        # 0.0, then shares_on_rung = spend / effective_rung_price
        # raised ZeroDivisionError, killing the whole evaluation
        # cycle. The SF-BL-9473/9542 ladder filter above is the first
        # line of defence (it keeps every rung priced outside (0, 1),
        # YES raw and derived-NO alike, out of this ladder), but the
        # division site itself carried no guard — unlike the nstar.py
        # counterpart vwap_for_dollars, which guards its walk with
        # `take_value / price if price > 0 else 0.0`. This guard is
        # that same check at this sweep's division: a rung whose
        # fee-adjusted price is not a finite number strictly > 0 is
        # skipped (`continue`, not `break` — the prereg requires the
        # remaining rungs to sweep, and a rung that reaches this loop
        # despite the filter sits at an arbitrary position), so the
        # sweep returns the remaining rungs' fill or the standard
        # (0, 0, 0) no-fill, never an exception. Kill-criterion audit
        # (PRODUCER_CENSUS.md in the task work dir): NOT met —
        # OrderBookLevel carries no price validation, a 0.0 rung
        # constructs silently, and the mutable dataclasses admit
        # list-replacement and in-place price mutation (and
        # unpickling) after construction, so zero-price rungs DO
        # reach this function; verified live in repro.py, where
        # neutralising the ladder filter reproduces the prereg
        # ZeroDivisionError verbatim pre-patch. On every ladder the
        # filter admits, rung.price is in (0, 1) and book.fee_rate
        # was re-validated in [0, 0.10] at sweep entry, so
        # effective_rung_price is finite and in (0, 1.1): this guard
        # never triggers there and valid books are byte-identical.
        if not _is_finite_number(effective_rung_price) or effective_rung_price <= 0.0:
            continue
        # SF-BL-8891: parity stop. A binary contract pays at most $1.00,
        # so a rung whose FEE-ADJUSTED price is at/above $1.00 can never
        # be a rational fill at any hurdle: pre-fix the multiplicative
        # fee pushed near-certainty rungs over parity (ask 0.999 at fee
        # 0.015 -> 1.013985) and every caller whose hurdle did not
        # incidentally stop there — the naive baseline's and
        # budget_sensitivity_check's full walks (min_net_edge=-1.0) and
        # any direct sweep caller — filled at q_exec >= 1.0, booked an
        # executed trade whose expected_log_wealth_growth was zeroed by
        # that function's q>=1 guard, while kl_divergence_binary
        # silently clipped q to 1-1e-6 and reported a phantom edge.
        # Verified pre-fix (repro.py / replay.py in the task work dir):
        # the prereg fixture sweeps (1000.0, 986.21, 1.013985) on both
        # sides; replaying the archived 2026-10-21T10Z forward snapshot
        # (397 two-sided books, 232 with mid outside [0.02, 0.98] —
        # the counts in the task rationale) produced 6,711 executed
        # fills with q_exec > 1.0 (every one growth-zeroed) and 7,815
        # parity-rung consumptions across the fixture grid + forward
        # books. The disciplined evaluate path was protected only
        # incidentally (its 0.05 edge stop reads the same top rung
        # first; 5,904 pre-fix evaluate executions, 0 at q_exec>=1) —
        # the kill criterion ("any engine path can still emit
        # q_exec>=1.0") is therefore NOT met and the fix belongs here,
        # in the shared sweep loop, before the hurdle stop, so it holds
        # for every hurdle including -1.0 full walks and the shrunk-LCB
        # stop. `break` (not skip) is complete: both ladders are in
        # ascending price order (YES asks ascending; NO derived from
        # YES bids descending), so every later rung's fee-adjusted
        # price is >= this one's. Rungs whose fee-adjusted price is
        # strictly below 1.0 are untouched, and evaluate adds a
        # post-sweep q_exec>=1 backstop (SF-BL-8891) in case a future
        # path ever bypasses this loop.
        if effective_rung_price >= 1.0:
            break
        # Stop sweeping if this rung's marginal price destroys our minimum edge hurdle.
        # SF-BL-3840: stop_belief is the point belief by default, or its
        # shrunk lower confidence bound when the caller opted in.
        if (stop_belief - effective_rung_price) < min_net_edge:
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

# --- SF-BL-9507: kl_divergence_binary fabricates KL on invalid inputs --
# Both arguments were passed straight to max/min clipping with no
# validation. Every comparison against NaN is False, so the clip chains
# collapse NaN to a boundary value, and finite out-of-range values and
# bools clip to the same boundaries: verified live pre-fix (repro.py in
# the task work dir) (NaN, 0.50) returned 0.693132365049887 nats,
# (0.50, NaN) returned 6.214608598408064, (1.50, 0.50) returned
# 0.693132365049887, (True, 0.50) returned 0.693132365049887 and
# (NaN, NaN) returned exactly 0.0 — a plausible-looking information
# edge (or a fabricated "no edge") for beliefs/prices that do not
# exist, and the value is written to
# TradeExecutionResult.kl_divergence_nats on the evaluate path.
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT
# met — evaluate's executed path IS validated (Gate 0.0 + sweep
# finiteness/parity backstops), but cli.py cmd_edge derives both
# arguments via np.clip, which PRESERVES NaN (demonstrated:
# np.clip(NaN, 0.02, 0.98) == NaN), budget_sensitivity_check takes
# bare unvalidated parameters, and the function is a public
# module-level diagnostic callable with bare literals — so the check
# belongs here, at the shared consumption surface, exactly as for the
# sibling nstar.kl_edge guard (SF-BL-9469, a distinct function in a
# distinct module). The valid domain is a finite real in [0, 1] (bool
# excluded, the module's _is_finite_number discipline); the boundaries
# 0 and 1 stay valid and keep flowing through the unchanged interior
# clip below, so every in-domain result is bit-identical to pre-fix.

def _kl_probability_in_domain(value: object) -> bool:
    """True iff value is a finite real number in [0, 1] (bool excluded)."""
    return _is_finite_number(value) and 0.0 <= value <= 1.0  # type: ignore[operator]


def _validate_kl_divergence_args(p: object, q: object) -> None:
    """Raise ValueError unless p and q are probabilities in [0, 1].

    p is validated before q (mirroring nstar.kl_edge, SF-BL-9469).
    Non-numeric values raise this same ValueError, not the bare
    TypeError the pre-fix min/max comparison produced.
    """
    if not _kl_probability_in_domain(p):
        raise ValueError(
            f"kl_divergence_binary p {p!r} is not a finite probability "
            "in [0, 1] (SF-BL-9507): D_KL is defined only for real "
            "probabilities — a non-finite, bool or out-of-range belief "
            "has no information edge (pre-fix (NaN, 0.50) returned "
            "0.693 nats via the clip); refusing to fabricate nats")
    if not _kl_probability_in_domain(q):
        raise ValueError(
            f"kl_divergence_binary q {q!r} is not a finite probability "
            "in [0, 1] (SF-BL-9507): D_KL is defined only for real "
            "probabilities — a non-finite, bool or out-of-range price "
            "has no information edge (pre-fix (0.50, NaN) returned "
            "6.215 nats via the clip); refusing to fabricate nats")


def kl_divergence_binary(p: float, q: float) -> float:
    """
    Kullback-Leibler divergence D_KL(p || q) in nats (Meister, 2024).
    Measures the theoretical asymptotic compound growth rate of wealth.

    SF-BL-9507: `p` and `q` must be finite probabilities in [0, 1]
    (bool excluded); anything else raises ValueError before any
    clipping, so no nats value is ever returned for an input that is
    not a probability. Valid inputs — boundaries 0 and 1 included —
    flow through the unchanged [1e-6, 1-1e-6] interior clip, which
    exists only to keep log(0) finite at the boundaries.
    """
    _validate_kl_divergence_args(p, q)
    p_c = max(1e-6, min(1.0 - 1e-6, p))
    q_c = max(1e-6, min(1.0 - 1e-6, q))
    return p_c * math.log(p_c / q_c) + (1.0 - p_c) * math.log((1.0 - p_c) / (1.0 - q_c))


# --- SF-BL-11008: non-finite beliefs return NaN / infinite growth -----
# expected_log_wealth_growth's p_side validators checked only the
# finite axes — SF-BL-9474's range check is gated on a finite value and
# SF-BL-9676 validates only q_exec / fraction — so a non-finite belief
# flowed into Meister Eq. III unvalidated: verified live pre-fix
# (repro.py in the task work dir, clean (0.70, 0.50, 0.30) =
# 0.07665250194562398) p_side=NaN returned NaN, p_side=+inf returned
# +inf and p_side=-inf returned -inf expected log growth — an
# infinite growth figure a sizing caller can read as unbounded edge.
# A non-finite belief was also MASKED as a benign 0.0 whenever the
# fraction/q_exec guard early-returned (NaN at f=0.0, +inf at
# q_exec=1.0 pre-fix -> 0.0). Kill-criterion audit (CALLER_CENSUS.md
# in the task work dir): NOT met — p_side does NOT only ever arrive
# finiteness-validated from evaluate_and_size_opportunity's posterior
# gate (Gate 0.0, SF-BL-9470/9440, which does validate that one
# path): cli.py cmd_edge derives p_side from its --p argument via its
# own np.clip, which PRESERVES NaN (verified live: nan -> nan), and
# test/fleet code calls this public function with bare literals — so
# the check belongs here, at the shared consumption surface, FIRST in
# the function, before the SF-BL-9474 range check and the guard, so
# no early return can mask it. Scope: ONLY the non-finite float axis
# — detection is float-exact (isinstance float and not isfinite;
# np.float64 subclasses float, so numpy beliefs are covered), so
# bool / non-numeric p_side (SF-BL-11043's surface: True still scores
# as p=1.0, '0.7' still raises the bare TypeError) and huge-int
# p_side (still the pre-fix OverflowError) behave exactly as pre-fix,
# and the finite range axis stays SF-BL-9474's.

def _p_side_is_nonfinite(p_side: object) -> bool:
    """True iff p_side is a non-finite float (NaN or +/-inf)."""
    return isinstance(p_side, float) and not math.isfinite(p_side)


def _validate_p_side_finite(p_side: object) -> None:
    """Raise ValueError if p_side is a non-finite float (SF-BL-11008)."""
    if _p_side_is_nonfinite(p_side):
        raise ValueError(
            f"p_side {p_side!r} is not finite (SF-BL-11008): p_side is "
            "a probability — a NaN or infinite belief has no expected "
            "log growth (pre-fix p_side=NaN returned NaN growth, "
            "p_side=+inf returned +inf and p_side=-inf returned -inf "
            "at q_exec=0.5, f=0.30); refusing to score it instead of "
            "reporting NaN or unbounded growth")


# --- SF-BL-9474: finite out-of-range beliefs score as positive growth ---
# expected_log_wealth_growth's guard checked only fraction and q_exec,
# so p_side — a probability, domain [0, 1] — flowed into Meister Eq. III
# unvalidated: verified live pre-fix (repro.py in the task work dir,
# q_exec=0.5) p_side=2.0 at f=0.30 returned +0.8814034728737146 and
# p_side=1.50 at f=0.10 returned +0.19564552753540054 — positive expected
# growth for beliefs that cannot exist, corrupting any caller that
# sanity-checks or optimises sizing on this function. An invalid belief
# was also MASKED as a benign 0.0 whenever the fraction/q_exec guard
# early-returned (p_side=2.0, f=0.0 pre-fix -> 0.0). Kill-criterion
# audit (CALLER_CENSUS.md in the task work dir): NOT met — p_side does
# NOT only ever arrive pre-clipped from evaluate_and_size_opportunity:
# cli.py cmd_edge calls this function directly off its own np.clip, and
# test/fleet code calls it with bare literals; it is a public
# module-level function with no stated pre-clip precondition, so the
# check belongs here, at the shared consumption surface, BEFORE the
# fraction/q_exec guard so no early return can mask it. Scope: ONLY
# the finite range axis is validated here — non-finite p_side
# (SF-BL-11008) and non-numeric/bool p_side (SF-BL-11043) are separate
# prereg surfaces and behave exactly as pre-fix (the helper is gated on
# a finite real value), and the fraction>1 axis is SF-BL-9399's.

def _p_side_finite_out_of_range(p_side: object) -> bool:
    """True iff p_side is a finite real number outside [0, 1] (SF-BL-9474)."""
    return _is_finite_number(p_side) and not (0.0 <= p_side <= 1.0)  # type: ignore[operator]


def _validate_p_side_range(p_side: object) -> None:
    """Raise ValueError if p_side is a finite real outside [0, 1]."""
    if _p_side_finite_out_of_range(p_side):
        raise ValueError(
            f"p_side {p_side!r} outside [0, 1] (SF-BL-9474): p_side is "
            "a probability — an impossible belief has no expected log "
            "growth (pre-fix p_side=2.0 at q_exec=0.5, f=0.30 returned "
            "+0.881 nats); refusing to score it instead of reporting "
            "fabricated growth")


# --- SF-BL-9676: non-finite q_exec / fraction fail open ---------------
# expected_log_wealth_growth's only guard was the comparison chain
# `fraction <= 0.0 or q_exec <= 0.0 or q_exec >= 1.0`, and every
# comparison against NaN is False, so a NaN operand flowed into
# Meister Eq. III unvalidated: verified live pre-fix (repro.py in the
# task work dir, clean (0.70, 0.50, 0.30) = 0.07665250194562398)
# q_exec=NaN returned NaN, fraction=NaN returned NaN and fraction=+inf
# returned +inf expected log growth — an unbounded growth figure for
# a stake that does not exist. The +/-inf operands in the other
# direction were MASKED as the benign 0.0 sentinel (fraction=-inf via
# fraction<=0, q_exec=+inf via q_exec>=1, q_exec=-inf via q_exec<=0),
# and non-numeric operands ('0.5', None) crashed with a bare TypeError
# from the guard comparison, while bool operands masqueraded as
# numbers (fraction=True scored as f=1.0). Kill-criterion audit
# (CALLER_CENSUS.md in the task work dir): NOT met — the evaluate path
# is guarded upstream for its own sweep/equity inputs (SF-BL-9438),
# but cli.py cmd_edge derives f_side from its --lam argument with NO
# finiteness validation (verified live pre-fix: `sf edge --p 0.80
# --q-exec 0.50 --lam nan` exited 0 as TRADE publishing
# expected_log_growth_at_kelly_size NaN) and derives q_side via
# np.clip, which PRESERVES NaN, and direct test/fleet callers pass
# bare literals to this public function — so the check belongs here,
# at the shared consumption surface, BEFORE the guard so no early
# return can mask a non-finite operand. Scope: ONLY the q_exec and
# fraction arguments (their non-numeric claim included, per
# SF-BL-11043's scoping note) — non-finite p_side is SF-BL-11008's
# surface and bool/non-numeric p_side is SF-BL-11043's, both behaving
# exactly as pre-fix, and finite fraction >= 1 remains SF-BL-9399's
# axis, so every finite fixture is bit-identical. The check reuses
# _is_finite_number (bool / non-numeric excluded), the module's
# established discipline (SF-BL-9507 / 9675 pattern).

# --- SF-BL-11081: q_exec TYPE dimension, already rejected by 9676 -----
# Filed as the q_exec analogue of SF-BL-11043 (p_side type): the claim
# was that q_exec='0.5'/None crash with a bare TypeError at the
# q_exec<=0 guard and q_exec=True is read as price 1.0, returning the
# flagged 0.0 for a corrupt reason. Kill-criterion audit (KILL_AUDIT.md
# in the task work dir): MET in substance — SF-BL-10911, named in this
# task's kill clause, is a later duplicate registration of
# SF-BL-9676's surface (per SF-BL-9676's DONE-CLAIM), and SF-BL-9676's
# landed _validate_q_exec_fraction_finite already rejects every bool /
# non-numeric q_exec via _is_finite_number BEFORE the guard, a fact
# its AGREEd VERDICT verified for exactly '0.5', None and True and
# this task's task-start repro re-confirmed live (all raise the
# SF-BL-9676 ValueError; the filing repro does not reproduce). No new
# validation logic is therefore added: the check below only makes the
# q_exec type dimension explicit — its own named step and an error
# naming both task ids, so the repo-pinned SF-BL-9676 suite (which
# asserts the 9676 tag for these inputs) stays green. The population
# rejected, the exception type, the ordering (after the p_side
# validators, before the guard) and every finite result are unchanged.
# Scope: q_exec ONLY — fraction's type dimension is SF-BL-11082's and
# p_side's is SF-BL-11043's; non-finite q_exec VALUES (NaN, +/-inf,
# oversized int) keep the SF-BL-9676 finiteness message exactly.

def _q_exec_is_bool_or_non_numeric(q_exec: object) -> bool:
    """True iff q_exec is a bool or not an int/float at all (SF-BL-11081).

    The type axis only: a finite or non-finite int/float (np.float64
    included, as a float subclass) is NOT covered here — finiteness
    is SF-BL-9676's check below.
    """
    return isinstance(q_exec, bool) or not isinstance(q_exec, (int, float))


def _validate_q_exec_type(q_exec: object) -> None:
    """Raise ValueError if q_exec is a bool or non-numeric (SF-BL-11081)."""
    if _q_exec_is_bool_or_non_numeric(q_exec):
        raise ValueError(
            f"q_exec {q_exec!r} is not a real number "
            f"(type {type(q_exec).__name__}) (SF-BL-11081; type "
            "dimension of the SF-BL-9676 q_exec validation): an "
            "execution price that is a bool or non-numeric has no "
            "type the binary payoff can be ordered against — pre-"
            "SF-BL-9676 q_exec='0.5'/None crashed with a bare TypeError "
            "at the q_exec<=0 guard and q_exec=True was read as price "
            "1.0, returning the flagged 0.0 for a corrupt reason; "
            "refusing to score it")


def _validate_q_exec_fraction_finite(q_exec: object, fraction: object) -> None:
    """Raise ValueError unless q_exec and fraction are finite reals.

    q_exec is validated before fraction (signature order). A finite
    q_exec outside (0, 1) and a finite fraction <= 0 are NOT errors
    here — they keep the guard's 0.0 sentinel below; only operands
    that cannot be ordered against the binary payoff at all
    (non-finite, bool, non-numeric) raise (SF-BL-9676).
    """
    _validate_q_exec_type(q_exec)
    if not _is_finite_number(q_exec):
        raise ValueError(
            f"q_exec {q_exec!r} is not a finite real number "
            "(SF-BL-9676): an execution price that is NaN, infinite, "
            "a bool or non-numeric cannot be ordered against the "
            "binary payoff — pre-fix q_exec=NaN returned NaN expected "
            "log growth through the all-False guard and q_exec=+/-inf "
            "was masked as 0.0; refusing to score it")
    if not _is_finite_number(fraction):
        raise ValueError(
            f"fraction {fraction!r} is not a finite real number "
            "(SF-BL-9676): a stake fraction that is NaN, infinite, a "
            "bool or non-numeric has no expected log growth — pre-fix "
            "fraction=NaN returned NaN and fraction=+inf returned "
            "+inf growth; refusing to score it")


# --- SF-BL-11043: p_side TYPE is never validated -----------------------
# expected_log_wealth_growth's p_side validators checked only the
# VALUE axes — SF-BL-11008's finiteness check is float-exact
# (isinstance float and not isfinite) and SF-BL-9474's range check is
# gated on a finite real — so a bool or non-numeric belief passed both
# and flowed into Meister Eq. III's growth sum unvalidated in type:
# verified live pre-fix (repro.py in the task work dir, clean
# (0.70, 0.50, 0.30) = 0.07665250194562398) p_side='0.7' and
# p_side=None crashed with a bare TypeError in the growth sum
# ("can't multiply sequence by non-int of type 'float'" /
# "unsupported operand type(s) for *: 'NoneType' and 'float'"),
# p_side=True RETURNED 0.26236426446749106 — bit-identical to the
# certain belief p=1.0, against 0.0767 for the clean 0.70 fixture —
# p_side=False returned the p=0.0 growth -0.35667494393873245, and
# p_side=(0.7+0j) returned a complex "growth". A bad-type belief was
# also MASKED as a benign 0.0 whenever the fraction/q_exec guard
# early-returned ('0.7' at f=0.0, None at q_exec=1.0 pre-fix -> 0.0).
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT
# met — p_side does NOT only ever arrive as a validated float from
# evaluate_and_size_opportunity (whose Gate 0.0 does type-validate
# that one path): cli.py cmd_edge calls this function off its own
# independent construction, the repo's own suites and fleet tests
# call it with bare literals — True / '0.7' / None included, as
# deliberate scope pins — and the function is a public module-level
# surface whose `p_side: float` annotation is unenforced, so the
# check belongs here, at the shared consumption surface, FIRST in
# the function, before the SF-BL-11008 finiteness check, the
# SF-BL-9474 range check and the guard, so no early return can mask
# it. Scope: TYPE only — the predicate is the module's standing
# discipline (int/float, bool excluded, the SF-BL-11081 q_exec
# pattern; np.float64 passes as a float subclass), so every numeric
# p_side keeps its pre-fix handling bit-identically: NaN / +/-inf
# still raise SF-BL-11008, finite out-of-range still raises
# SF-BL-9474, and a huge int (10**400) still raises the pre-fix
# OverflowError downstream. The validation never coerces: '0.7' is
# never parsed and True is never read as 1.

def _p_side_is_bool_or_non_numeric(p_side: object) -> bool:
    """True iff p_side is a bool or not an int/float at all (SF-BL-11043).

    The type axis only: a finite or non-finite int/float (np.float64
    included, as a float subclass) is NOT covered here — finiteness
    is SF-BL-11008's check and finite range is SF-BL-9474's.
    """
    return isinstance(p_side, bool) or not isinstance(p_side, (int, float))


def _validate_p_side_type(p_side: object) -> None:
    """Raise ValueError if p_side is a bool or non-numeric (SF-BL-11043)."""
    if _p_side_is_bool_or_non_numeric(p_side):
        raise ValueError(
            f"p_side {p_side!r} is not a real number "
            f"(type {type(p_side).__name__}) (SF-BL-11043): a "
            "belief that is a bool or non-numeric has no expected "
            "log growth — pre-fix p_side='0.7'/None crashed with a "
            "bare TypeError in the growth sum, p_side=True was read "
            "as p=1.0 and returned 0.26236426446749106 (the growth "
            "of a certain belief, against 0.07665250194562398 for "
            "the clean p_side=0.70 fixture) and p_side=False as "
            "p=0.0; refusing to score it — convert serialized "
            "values with float() before calling, values are never "
            "coerced here")


def expected_log_wealth_growth(p_side: float, q_exec: float, fraction: float) -> float:
    """
    Expected single-trade log-wealth increment Delta G (Meister, 2024 Eq. III):
      U(p_side, q_exec, f) = p_side * log(1 + f * (1 - q_exec)/q_exec) + (1 - p_side) * log(1 - f)

    SF-BL-9676: `q_exec` and `fraction` must be finite real numbers
    (bool excluded); a non-finite or non-numeric value for either
    raises ValueError before the guard below, so no caller can read
    NaN / infinite growth — or a masked 0.0 — off an operand that
    cannot be ordered against the binary payoff. Finite out-of-range
    operands keep the guard's 0.0 sentinel unchanged.

    SF-BL-11081: `q_exec`'s TYPE dimension is validated explicitly by
    `_validate_q_exec_type` (called first inside the SF-BL-9676
    validator above): a bool or non-numeric q_exec raises ValueError
    naming SF-BL-11081 and SF-BL-9676 — never the bare TypeError or
    the True-as-1.0 masked 0.0 the filing repro described (already
    rejected by SF-BL-9676's landed fix; see the SF-BL-11081 block).

    SF-BL-11008: `p_side` must not be a non-finite float; NaN or
    +/-inf raises ValueError first, before the range check and the
    guard, so no caller can read NaN / infinite growth — or a masked
    0.0 — off a belief that is not a probability. Finite beliefs are
    untouched (range is SF-BL-9474's check, next).

    SF-BL-11043: `p_side`'s TYPE dimension is validated explicitly by
    `_validate_p_side_type`, called FIRST below, before the
    SF-BL-11008 finiteness and SF-BL-9474 range validators: a bool or
    non-numeric p_side raises ValueError naming SF-BL-11043 — never
    the bare TypeError from the growth sum, and never the True-as-1.0
    (0.2624) or False-as-0.0 scored values the filing repro described
    (see the SF-BL-11043 block above). Numeric p_side is untouched.
    """
    _validate_p_side_type(p_side)
    _validate_p_side_finite(p_side)
    _validate_p_side_range(p_side)
    _validate_q_exec_fraction_finite(q_exec, fraction)
    if fraction <= 0.0 or q_exec <= 0.0 or q_exec >= 1.0:
        return 0.0
    win_mult = (1.0 - q_exec) / q_exec
    return p_side * math.log(1.0 + fraction * win_mult) + (1.0 - p_side) * math.log(max(1e-9, 1.0 - fraction))


# --- SF-BL-10845: require_on_chain_for_crypto must be a strict bool -------
# Gate 1 (CryptoBench) and Gate 0.9 (SF-BL-10793) both read
# `config.require_on_chain_for_crypto` by bare truthiness, and pre-fix
# FleetRiskConfig had NO __post_init__ at all, so Python truthiness
# decided whether the crypto verification gate existed: on the prereg
# fixture (CRYPTO book, on_chain_verified=False, ask 0.50 / bid 0.48,
# p=0.80, fee 0.01, equity $100,000) the falsy non-bools '' and 0 (and
# 0.0 / None / []) traded BUY_YES $10,429.29 — Gate 1 silently OFF,
# exactly as a strict bool False trades by design — while the string
# 'False' (and 'True' / 'no' / '0' / 1 / 0.5) SKIPped
# FAILED_CRYPTOBENCH_ONCHAIN_GATE, the gate ARMED by a value that reads
# as off to any env-var / JSON / CSV config producer. Worse, Gate 0.9's
# strict-bool check on the BOOK flag is itself conditioned on this
# flag, so require='' disabled that validation too (book flag 'False'
# + require='' traded $10,429.29 pre-fix, verified live). Kill-criterion
# audit (CALLER_CENSUS.md in the task work dir): NOT met — the flag is
# NOT "only ever set by validated in-repo literals": the in-repo
# setters are bare unenforced literals (class default True; one test
# literal False), there is no schema-validation layer anywhere in the
# repo, FleetRiskConfig is a mutable dataclass (post-construction
# mutation and unpickling bypass any constructor check), and evaluate's
# default config is a single shared mutable instance. The value itself
# is therefore validated at construction (FleetRiskConfig.__post_init__
# raises ValueError, as for fee_rate / on_chain_verified) and
# re-validated at evaluate ENTRY in evaluate_and_size_opportunity
# (SKIP INVALID_CONFIG), which covers mutated/unpickled configs and
# runs before Gate 0.9/Gate 1 can read the flag. Strict bool means
# isinstance(v, bool) (_is_strict_bool, SF-BL-10793): int 1/0 do NOT
# qualify. Scope: ONLY this field is validated here; the config's other
# fields are separate prereg surfaces and are deliberately untouched.

def _validate_require_on_chain_for_crypto(value: object) -> None:
    """Raise ValueError unless require_on_chain_for_crypto is a strict bool."""
    if not _is_strict_bool(value):
        raise ValueError(
            f"require_on_chain_for_crypto {value!r} is not a bool "
            f"(type {type(value).__name__}) (SF-BL-10845): invalid "
            "config — the CryptoBench gate switch requires a strict "
            "bool; a truthy string such as 'False' from an env/JSON/CSV "
            "producer must never arm it and a falsy non-bool must never "
            "silently disable it")


# --- SF-BL-10844: max_inter_agent_dispersion threshold domain ------------
# Gate 2 in evaluate_and_size_opportunity compares the caller-supplied
# dispersion against this config threshold with a bare ``>``, yet pre-fix
# the threshold itself was validated NOWHERE: FleetRiskConfig had no
# __post_init__ at all, the dataclass is mutable, and the float
# annotation is unenforced. Verified live pre-fix on the prereg fixture
# (p=0.80, ask 0.50 / bid 0.48, fee 0.01, equity $100,000,
# dispersion=0.50, which SKIPs UNRESOLVED_AGENT_DISPERSION at the clean
# 0.22 threshold): threshold NaN and +inf both traded BUY_YES $15,000
# (``dispersion > NaN`` / ``> +inf`` is never True, so the supervisor
# gate fails OPEN), and threshold '0.22' raised a bare TypeError from
# the comparison instead of a named SKIP; -inf / negative thresholds
# SKIPped only incidentally with the unresolved-dispersion reason, and
# bool True masqueraded as threshold 1.0. Kill-criterion audit
# (CALLER_CENSUS.md in the task work dir): NOT met — no in-repo
# construction passes a threshold at all (every caller uses the shared
# default instance or sets unrelated fields), and nothing validates the
# value; mutation / unpickling bypass construction regardless. The
# valid domain is exactly the prereg domain: a finite real >= 0 (bool
# excluded, as everywhere in this file) — TIGHTENED to [0, 1] by
# SF-BL-9397 (registered earlier; see the helper below and the
# SF-BL-9397 block). Defence in depth, mirroring
# SF-BL-9363's fee_rate: construction raises ValueError via
# __post_init__, and evaluate re-validates at Gate 2 (SKIP
# INVALID_MAX_INTER_AGENT_DISPERSION, zero size) so a mutated /
# unpickled config — or a mutated shared default instance — can never
# reach the bare comparison. Only this field is validated on this
# prereg surface; the config's other fields are separate prereg
# surfaces (SF-BL-10845 validates its own field the same way).

def _is_valid_max_inter_agent_dispersion(threshold: object) -> bool:
    """True iff threshold is a finite real number in [0, 1] (bool excluded).

    SF-BL-10844 registered the domain as >= 0; SF-BL-9397 (registered
    2026-10-21, BEFORE SF-BL-10844) mandates [0, 1] — dispersion is the
    pstdev of probabilities in [0, 1], so a threshold above 1 can never
    fire Gate 2 and silently disables the supervisor gate. The combined
    domain is the intersection, [0, 1] (see CALLER_CENSUS.md in the
    SF-BL-9397 task work dir).
    """
    return (
        _is_finite_number(threshold)
        and 0 <= threshold <= 1  # type: ignore[operator]
    )


def _validate_max_inter_agent_dispersion(threshold: object) -> None:
    """Raise ValueError unless threshold is a finite real in [0, 1]."""
    if not _is_valid_max_inter_agent_dispersion(threshold):
        raise ValueError(
            f"max_inter_agent_dispersion {threshold!r} is not a finite "
            "real >= 0 and <= 1 (outside [0, 1]) (SF-BL-10844, tightened "
            "to [0, 1] by SF-BL-9397): Gate 2 compares dispersion against "
            "this threshold, so a NaN/+inf threshold fails the "
            "supervisor gate open, a threshold above 1 disables it, and "
            "a non-numeric threshold crashes the comparison with a bare "
            "TypeError")


# --- SF-BL-3839: lock-scaled execution hurdle ---------------------------
# FleetRiskConfig.min_net_edge was a flat hurdle whether capital locked
# for 3 days or 300, while N_star multiplies the fleet score by
# sqrt(365 / mean_lock_days) (docs/NORTH_STAR.md) — the gate admitted
# long-lock trades the velocity term then punishes, and the hurdle itself
# was never lock-adjusted. The lock-scaled hurdle (PREREG.md in the task
# work dir, registered before any replay) requires every admitted trade
# to clear the same VELOCITY-ADJUSTED edge: edge * sqrt(365 / D) >=
# base * sqrt(365 / D_REF), i.e.
#     hurdle(D) = base * sqrt(max(D, D_REF) / D_REF),  D_REF = 7.0 days
# sqrt (not linear) because N_star pays sqrt(365/L), not 365/L
# (SF-BL-1684). D_REF = 7.0d is the fleet's canonical entry horizon (the
# N_star backtest evaluates at T-7d; the SF-BL-3659 S2 census uses
# est_days 7.0; SF-BL-0444 flags 7.0d as the harness design constant) —
# it is not fitted to any panel. The flat base is a FLOOR: for
# D <= D_REF the hurdle is exactly the base, so the standing
# NORTH_STAR.md rule "SKIP unless |p - q_exec| >= 0.05 after all costs"
# is never loosened; only long-lock fills must clear more (D=30d ->
# 0.1035, D=90d -> 0.1793, D=365d -> 0.3610 at base 0.05).
# Corrupt-input fallback: a non-finite / non-numeric lock or reference,
# a non-positive reference, or a non-positive base returns the base
# hurdle unchanged (fail to flat — never loosen or tighten on corrupt
# input); the base hurdle's own validity is a separate prereg surface
# (SF-BL-9471 makes the sweep fail closed on a non-finite hurdle).
# Wiring: FleetRiskConfig.lock_scaled_hurdle_enabled (default False,
# Gate BS-style A/B flag) switches evaluate_and_size_opportunity's
# top-of-book screen and Step-5 sweep stop to this per-candidate
# hurdle; with the flag off, behaviour is byte-identical.

#: Reference lock (days) at which the lock-scaled hurdle equals the flat
#: base hurdle (SF-BL-3839).
LOCK_HURDLE_REF_DAYS = 7.0


def lock_scaled_min_net_edge(
    min_net_edge: float,
    est_days_to_resolution: float,
    ref_days: float = LOCK_HURDLE_REF_DAYS,
) -> float:
    """Lock-scaled post-cost edge hurdle for one candidate (SF-BL-3839).

    Returns ``min_net_edge * sqrt(max(D, ref_days) / ref_days)`` where
    ``D = est_days_to_resolution`` — the flat base at or below the
    reference lock, rising with the square root of lock beyond it.
    Returns the base unchanged on any corrupt input (see block comment).
    """
    if not _is_finite_number(min_net_edge) or min_net_edge <= 0:  # type: ignore[operator]
        return min_net_edge
    if not _is_finite_number(est_days_to_resolution):
        return min_net_edge
    if not _is_finite_number(ref_days) or ref_days <= 0:  # type: ignore[operator]
        return min_net_edge
    days = float(est_days_to_resolution)  # type: ignore[arg-type]
    if days <= float(ref_days):  # type: ignore[arg-type]
        return float(min_net_edge)
    return float(min_net_edge) * math.sqrt(days / float(ref_days))  # type: ignore[arg-type]


# --- SF-BL-10619: lock-scaled Gate D round-trip cap ---------------------
# FleetRiskConfig.max_round_trip_cost (SF-BL-7432 Gate D) was a flat
# 0.05/share veto whether capital locked for 0.25 days or 440, while
# N_star multiplies the fleet score by sqrt(365 / mean_lock_days) — a
# short-lock book recycles capital ~365/D times per year and pays the
# round-trip cost at every recycle, while a long-lock trade amortises
# the same cost over its whole hold. The lock-scaled cap (PREREG.md in
# the task work dir, registered before any replay) holds the
# VELOCITY-ADJUSTED round-trip drag constant in N_star's own sqrt
# space (SF-BL-1684 / SF-BL-3839 anchor): rt * sqrt(365 / D) =
# base * sqrt(365 / D_REF), i.e.
#     cap(D) = base * sqrt(D / D_REF),  D_REF = 7.0 days
# At base 0.05: D=0.25d -> 0.00945, D=2d -> 0.0267, D=7d -> 0.0500,
# D=30d -> 0.1035, D=200d -> 0.2673, D=440d -> 0.3964. No floor and no
# ceiling: a floor at the flat base for D < D_REF would make the
# short-lock half of the contrast identical to flat by construction.
# Corrupt-input fallback: a non-finite / non-numeric lock, reference or
# base, a non-positive reference, a non-positive base, or a
# non-positive lock returns the flat base cap unchanged (fail to flat —
# never loosen or tighten on corrupt input; the SF-BL-3839 precedent).
# A None base keeps Gate D disabled in both arms. Wiring:
# FleetRiskConfig.round_trip_lock_scaled_enabled (default False,
# Gate BS-style A/B flag) switches Gate D in
# evaluate_and_size_opportunity to this per-candidate cap; with the
# flag off, behaviour — including the SKIP reason — is byte-identical.

#: Reference lock (days) at which the lock-scaled Gate D cap equals the
#: flat base cap (SF-BL-10619).
ROUND_TRIP_LOCK_REF_DAYS = 7.0


def lock_scaled_max_round_trip_cost(
    max_round_trip_cost: Optional[float],
    est_days_to_resolution: float,
    ref_days: float = ROUND_TRIP_LOCK_REF_DAYS,
) -> Optional[float]:
    """Lock-scaled Gate D round-trip cap for one candidate (SF-BL-10619).

    Returns ``max_round_trip_cost * sqrt(D / ref_days)`` where
    ``D = est_days_to_resolution`` — the flat base at the reference
    lock, tighter below it and looser above it. Returns the base
    unchanged (None included) on any corrupt input (see block comment).
    """
    if max_round_trip_cost is None:
        return None
    if not _is_finite_number(max_round_trip_cost) or max_round_trip_cost <= 0:  # type: ignore[operator]
        return max_round_trip_cost
    if not _is_finite_number(est_days_to_resolution):
        return max_round_trip_cost
    if not _is_finite_number(ref_days) or ref_days <= 0:  # type: ignore[operator]
        return max_round_trip_cost
    days = float(est_days_to_resolution)  # type: ignore[arg-type]
    if days <= 0:
        return max_round_trip_cost
    return float(max_round_trip_cost) * math.sqrt(days / float(ref_days))  # type: ignore[arg-type]


# --- SF-BL-10726: edge-keyed Gate D round-trip veto ----------------------
# FleetRiskConfig.max_round_trip_cost (SF-BL-7432 Gate D) is a flat
# 0.05/share veto that never reads the trade's own cleared entry edge:
# it vetoes a trade with entry edge 0.20 and rt_cost 0.06 (net
# round-trip edge +0.14) and admits a trade with edge 0.051 and rt_cost
# 0.049 (net +0.002). SF-BL-10619 keyed the cap to lock duration; this
# task keys the veto to the entry edge itself (PREREG.md in the task
# work dir, registered before any replay): the edge-keyed Gate D admits
# iff the round trip leaves strictly positive net edge,
#     ADMIT iff rt_cost < entry_edge (= p_side - q_exec),
#     VETO  iff rt_cost >= entry_edge,
# i.e. the effective per-candidate cap IS the candidate's own entry
# edge, and rt_cost == entry_edge (net zero) vetoes. Corrupt-input
# fallback: a non-finite / non-numeric rt_cost or entry_edge makes the
# predicate return False (no edge-keyed veto) and the caller falls back
# to the flat comparison — fail to flat, the SF-BL-3839 / SF-BL-10619
# precedent; via evaluate both quantities are finite by construction.
# Wiring: FleetRiskConfig.round_trip_edge_keyed_enabled (default False,
# Gate BS-style A/B flag) switches Gate D in
# evaluate_and_size_opportunity to this predicate; when enabled it
# REPLACES the flat / SF-BL-10619 lock-scaled cap for that candidate
# (alternative cap rules, not composed). With the flag off, behaviour —
# including the SKIP reason — is byte-identical. A None
# max_round_trip_cost keeps Gate D disabled in both arms.

def edge_keyed_gate_d_veto(rt_cost: object, entry_edge: object) -> bool:
    """Edge-keyed Gate D veto for one candidate (SF-BL-10726).

    True (veto) iff ``rt_cost >= entry_edge`` — the round trip consumes
    the trade's whole cleared entry edge, leaving no positive net
    round-trip edge. False on any corrupt input (non-finite /
    non-numeric rt_cost or entry_edge): the predicate never fires on
    corrupt data; the caller falls back to the flat cap comparison.
    """
    if not _is_finite_number(rt_cost) or not _is_finite_number(entry_edge):
        return False
    return float(rt_cost) >= float(entry_edge)  # type: ignore[arg-type]


# --- SF-BL-11194: relative (price-level) Gate D veto ---------------------
# Gate D's absolute cap is mathematically vacuous on tail books:
# rt_cost = q_exec - q_exit with q_exit >= 0 (SF-BL-9844 form), so
# rt_cost <= q_exec always, and any entry at q_exec <= cap (flat 0.05)
# can never be vetoed however wide the relative exit spread — verified
# live on the title fixture (ask 0.030 / bid 0.001 / fee 0.01, p=0.30:
# BUY_YES at q_exec 0.0303, rt_cost 0.0293, a 96.7% relative exit loss
# passing Gate D). SF-BL-10619's lock-scaled cap only moves the vacuity
# boundary to q_exec <= cap(D); SF-BL-10726's edge-keyed veto fires on a
# tail trade only when entry_edge <= rt_cost <= q_exec (p_side <=
# 2*q_exec). The relative prong (PREREG.md in the task work dir,
# registered before any census) vetoes on the SHARE of entry price the
# round trip consumes:
#     VETO iff rt_cost / q_exec > max_share   (default share 0.50,
#     the task's own tail threshold)
# composed as an ADDITIONAL Gate D veto (OR with the flat / lock-scaled
# / edge-keyed verdict), i.e. the effective cap is
# min(base_cap, max_share * q_exec): above q_exec = base_cap / max_share
# (0.10 at defaults) the absolute cap binds first and behaviour is
# unchanged by construction. Boundary: share == max_share ADMITS
# (strict >, mirroring Gate D's strict rt_cost > cap). Corrupt-input
# fallback: a non-finite / non-numeric rt_cost or q_exec, q_exec <= 0,
# or a share outside (0, 1] returns False (no relative veto) — fail to
# flat, the SF-BL-3839 / 10619 / 10726 precedent; via evaluate both
# quantities are finite by construction. Census warrant (PREREG rule):
# 86.48% of flat-admitted forward candidates on the SF-BL-10619 Level-1
# corpus sit in the tail (q_exec <= 0.05, share >= 0.50), vs the 2%
# pre-registered bar; replication corpus (SF-BL-7432 S2) 47.16%.
# Wiring: FleetRiskConfig.round_trip_relative_enabled (default False,
# Gate BS-style A/B flag) ORs this veto into Gate D in
# evaluate_and_size_opportunity; a None max_round_trip_cost still
# disables Gate D entirely, relative prong included. With the flag off,
# behaviour — including the SKIP reason — is byte-identical.

#: Pre-registered relative Gate D share cap (SF-BL-11194): the round
#: trip may consume at most this share of the entry price.
RELATIVE_ROUND_TRIP_MAX_SHARE = 0.50


def relative_round_trip_veto(
    rt_cost: object,
    q_exec: object,
    max_share: object = RELATIVE_ROUND_TRIP_MAX_SHARE,
) -> bool:
    """Relative Gate D veto for one candidate (SF-BL-11194).

    True (veto) iff ``rt_cost / q_exec > max_share`` — an immediate
    liquidation loses more than the pre-registered share of the entry
    price. False on any corrupt input (non-finite / non-numeric
    rt_cost, q_exec or max_share; q_exec <= 0; max_share outside
    (0, 1]): the predicate never fires on corrupt data and the base
    Gate D verdict stands.
    """
    if not _is_finite_number(rt_cost) or not _is_finite_number(q_exec):
        return False
    if not _is_finite_number(max_share):
        return False
    q = float(q_exec)  # type: ignore[arg-type]
    share_cap = float(max_share)  # type: ignore[arg-type]
    if q <= 0.0 or share_cap <= 0.0 or share_cap > 1.0:
        return False
    return float(rt_cost) / q > share_cap  # type: ignore[arg-type]


# --- SF-BL-9639: dispersion-tapered Kelly sizing -------------------------
# Gate 2 compared inter_agent_dispersion against
# FleetRiskConfig.max_inter_agent_dispersion (0.22) with a bare ``>``:
# dispersion 0.21 sized at FULL fractional Kelly and 0.23 SKIPped
# entirely (UNRESOLVED_AGENT_DISPERSION), although the config comment
# describes the trigger as supervisor research, not a sizing
# discontinuity. The tapered alternative (PREREG.md in the task work
# dir, registered before any replay; the formula is the task's own)
# scales the Step-4 Kelly multiplier continuously:
#     taper_factor(d, T) = min(1, max(0, 1 - d / T)),  T = threshold
# so d=0 -> 1.00, d=0.055 -> 0.75, d=0.11 -> 0.50, d=0.165 -> 0.25,
# d=0.198 -> 0.10 and d >= T -> 0.00 against the default T=0.22. The
# outer clip to [0, 1] keeps a finite negative dispersion (which the
# binary gate passes at full size — SF-BL-9395's pinned grid) from
# AMPLIFYING lambda above the configured fractional Kelly. Boundary:
# at d == T the binary gate still trades (strict ``>``) while the
# taper sizes 0 — inherent in the task's formula, registered in the
# prereg. Corrupt-input fallback: a non-finite / non-numeric / bool
# dispersion or threshold, or a threshold outside [0, 1], returns 1.0
# (fail to full size — never invent a taper on corrupt input, the
# SF-BL-3839 fail-to-flat precedent); via evaluate the branch is
# unreachable, because Gate 0.75 (SF-BL-9395) SKIPs non-finite
# dispersion and Gate 1.9 (SF-BL-10844) SKIPs an invalid threshold
# before Gate 2 in both arms. T == 0 is the degenerate limit: factor
# 1.0 iff d == 0, else 0.0.
# Wiring: FleetRiskConfig.dispersion_taper_enabled (default False,
# Gate BS-style A/B flag) replaces Gate 2's binary SKIP with the
# taper — a zero factor SKIPs at Gate 2's position with
# DISPERSION_TAPER_ZERO, a positive factor scales Step 4's target
# fraction — and with the flag off, behaviour is byte-identical.

def dispersion_taper_factor(dispersion: object, threshold: object) -> float:
    """Continuous Gate 2 taper factor for one candidate (SF-BL-9639).

    Returns ``min(1, max(0, 1 - dispersion / threshold))`` — 1.0 at
    zero dispersion, 0.0 at or beyond the threshold. Returns 1.0 on
    any corrupt input (see block comment); ``threshold == 0`` yields
    1.0 iff ``dispersion == 0``, else 0.0.
    """
    if not _is_finite_number(dispersion) or not _is_finite_number(threshold):
        return 1.0
    thr = float(threshold)  # type: ignore[arg-type]
    if thr < 0.0 or thr > 1.0:
        return 1.0
    d = float(dispersion)  # type: ignore[arg-type]
    if thr == 0.0:
        return 1.0 if d == 0.0 else 0.0
    return min(1.0, max(0.0, 1.0 - d / thr))


# --- SF-BL-9397: FleetRiskConfig range bands (kelly / cluster / hurdle) --
# NORTH_STAR.md mandates the sizing bands this config carries: fractional
# Kelly lambda in [0.25, 0.50] ("Sizing: fractional Kelly f = lambda *
# (p - q_exec)/(1 - q_exec), lambda in [0.25, 0.50]"), a hard 15% cap per
# correlated cluster, and "SKIP unless |p - q_exec| >= 0.05 after all
# costs" — a hurdle that is meaningless below 0. Pre-fix, FleetRiskConfig
# validated NONE of kelly_multiplier / max_cluster_allocation /
# min_net_edge (SF-BL-10844/10845 covered only the dispersion threshold
# and the on-chain switch), and evaluate_and_size_opportunity trusted
# every field: the wild-config repro (repro.py in the task work dir;
# p=0.60, ask 0.55 / bid 0.53, fee 0.01, equity $100,000) at
# kelly_multiplier=2.0, max_cluster_allocation=0.90, min_net_edge=-0.10
# EXECUTED BUY_YES at allocated_fraction 0.2002 / $20,022.50 invested
# where the default config SKIPs INSUFFICIENT_NET_EDGE (edge 0.0445 <
# 0.05), and min_net_edge='0.05' by mutation crashed evaluate with a
# bare TypeError at the edge comparison. Kill-criterion audit
# (CALLER_CENSUS.md in the task work dir): NOT met — no schema layer or
# producer invariant bounds these fields (bare literals everywhere, a
# mutable dataclass, a shared mutable default instance), and the wild
# config changed allocation versus the default through the normal
# construction path. The values are therefore validated at construction
# (FleetRiskConfig.__post_init__ raises ValueError) and re-validated in
# evaluate at Gate 2.5 (SKIP INVALID_CONFIG, zero size, cluster dict
# untouched), immediately after Gate 2 and before the first read of any
# of the three fields (the edge screen / lock-scaled hurdle, Step 4
# Kelly and cluster sizing, the Step-5 sweep stop) — the defence-in-
# depth pattern of SF-BL-9363 / 10844 / 10845, covering mutated /
# unpickled / shared-default configs. Domains (prereg-exact):
# kelly_multiplier a finite real in [0.25, 0.50]; max_cluster_allocation
# a finite real in (0, 0.15] (0 exclusive: a zero cap is a config error,
# not a sizing instruction); min_net_edge a finite real >= 0 (the
# prereg registers only "< 0"; finiteness / type are included so a NaN
# hurdle cannot report a misleading INSUFFICIENT_NET_EDGE and a string
# cannot crash the comparison — the SF-BL-10876 type/finiteness surface
# for this same field; no upper bound is preregistered, so none is
# imposed). Bool is excluded everywhere (the file-wide convention: a
# bool is not a band value). This task's fourth field,
# max_inter_agent_dispersion [0, 1], is validated by the SF-BL-10844
# helpers above as tightened in that block, at construction and at its
# own Gate 1.9. Out of scope (separate prereg surfaces, untouched):
# min_confidence (SF-BL-10820), crypto_kelly_dampener (SF-BL-10821),
# max_round_trip_cost (SF-BL-10798), and sweep_order_book's own
# min_net_edge argument incl. its documented -1.0 full-walk override
# (SF-BL-9471).

#: Mandated fractional-Kelly band (NORTH_STAR.md, SF-BL-9397).
KELLY_MULTIPLIER_MIN = 0.25
KELLY_MULTIPLIER_MAX = 0.50
#: Mandated per-cluster allocation cap (Prediction Arena, SF-BL-9397).
MAX_CLUSTER_ALLOCATION_MAX = 0.15


def _is_valid_kelly_multiplier(value: object) -> bool:
    """True iff kelly_multiplier is a finite real in [0.25, 0.50]."""
    return (
        _is_finite_number(value)
        and KELLY_MULTIPLIER_MIN <= value <= KELLY_MULTIPLIER_MAX  # type: ignore[operator]
    )


def _validate_kelly_multiplier(value: object) -> None:
    """Raise ValueError unless kelly_multiplier is in [0.25, 0.50]."""
    if not _is_valid_kelly_multiplier(value):
        raise ValueError(
            f"kelly_multiplier {value!r} outside mandated band "
            f"[{KELLY_MULTIPLIER_MIN}, {KELLY_MULTIPLIER_MAX}] "
            "(SF-BL-9397): NORTH_STAR.md mandates fractional Kelly "
            "lambda in [0.25, 0.50]; a multiplier outside the band "
            "silently overrides the mandated sizing (the wild-config "
            "repro's 2.0 allocated 20.02% of equity to one trade)")


def _is_valid_max_cluster_allocation(value: object) -> bool:
    """True iff max_cluster_allocation is a finite real in (0, 0.15]."""
    return (
        _is_finite_number(value)
        and 0.0 < value <= MAX_CLUSTER_ALLOCATION_MAX  # type: ignore[operator]
    )


def _validate_max_cluster_allocation(value: object) -> None:
    """Raise ValueError unless max_cluster_allocation is in (0, 0.15]."""
    if not _is_valid_max_cluster_allocation(value):
        raise ValueError(
            f"max_cluster_allocation {value!r} outside mandated band "
            f"(0, {MAX_CLUSTER_ALLOCATION_MAX}] (SF-BL-9397): "
            "NORTH_STAR.md mandates a hard 15% cap per correlated "
            "cluster; a larger cap silently overrides it and a "
            "zero/negative cap is a config error, not a sizing "
            "instruction")


def _is_valid_min_net_edge(value: object) -> bool:
    """True iff min_net_edge is a finite real >= 0 (bool excluded)."""
    return _is_finite_number(value) and value >= 0  # type: ignore[operator]


def _validate_min_net_edge(value: object) -> None:
    """Raise ValueError unless min_net_edge is a finite real >= 0."""
    if not _is_valid_min_net_edge(value):
        raise ValueError(
            f"min_net_edge {value!r} is not a finite real >= 0 "
            "(SF-BL-9397): the post-cost edge hurdle cannot be "
            "negative — a negative hurdle admits trades the mandated "
            "0.05 screen rejects (the wild-config repro's -0.10 "
            "admitted the fixture's 0.0445 edge) — and a non-finite "
            "or non-numeric hurdle corrupts or crashes the edge "
            "comparison")


def _fleet_risk_config_range_violation(
    config: object,
) -> Optional[Tuple[str, object, str]]:
    """First SF-BL-9397 range violation in `config`, or None.

    Returns (field, value, mandated band) for the first of
    kelly_multiplier / max_cluster_allocation / min_net_edge (checked
    in that order) whose value — read via getattr, so a missing
    attribute / config=None reads as None — is outside its mandated
    band. max_inter_agent_dispersion is NOT checked here: Gate 1.9
    (SF-BL-10844, tightened above) owns that field's evaluate path.
    """
    kelly = getattr(config, "kelly_multiplier", None)
    if not _is_valid_kelly_multiplier(kelly):
        return ("kelly_multiplier", kelly,
                f"[{KELLY_MULTIPLIER_MIN}, {KELLY_MULTIPLIER_MAX}]")
    cluster_cap = getattr(config, "max_cluster_allocation", None)
    if not _is_valid_max_cluster_allocation(cluster_cap):
        return ("max_cluster_allocation", cluster_cap,
                f"(0, {MAX_CLUSTER_ALLOCATION_MAX}]")
    hurdle = getattr(config, "min_net_edge", None)
    if not _is_valid_min_net_edge(hurdle):
        return ("min_net_edge", hurdle, "finite real >= 0")
    return None


# --- SF-BL-10798: max_round_trip_cost threshold domain (Gate D cap) -----
# Gate D (SF-BL-7432) vetoes on a bare ``rt_cost > max_round_trip_cost``
# comparison, yet pre-fix the threshold itself was validated NOWHERE:
# FleetRiskConfig.__post_init__ covered the SF-BL-10844 / 10845 / 9397
# fields only, and the SF-BL-9397 block above lists this field
# explicitly as an out-of-scope separate prereg surface (its own) —
# the prereg kill condition ("FleetRiskConfig gains field validation
# covering this field under SF-BL-9397's landed fix") is therefore NOT
# met (CALLER_CENSUS.md in the task work dir). Verified live pre-fix
# (repro.py in the task work dir; wide-spread fixture ask 0.50 /
# bid 0.10, p=0.80, fee 0.01, equity $10,000, rt_cost 0.406/share):
# max_round_trip_cost=NaN traded BUY_YES $1,500 — the full 15% cluster
# cap — because ``rt_cost > NaN`` is always False, so Gate D fails
# OPEN; +inf traded identically; -inf / -1.0 SKIPped only incidentally
# as an ultra-tight Gate D cap (vetoing even a tight rt_cost 0.020
# book); the string '0.05' raised a bare TypeError at the Gate D
# comparison; and bool True masqueraded as threshold 1.0. SF-BL-9811
# validates Gate D's q_exec ARGUMENT inside round_trip_cost, not this
# threshold, and SF-BL-10619's lock_scaled_max_round_trip_cost returns
# a non-finite / negative base unchanged (fail to flat), so neither
# covers this surface. The valid domain is exactly the prereg domain:
# None (Gate D disabled, the documented A/B value) or a finite real
# >= 0 (bool excluded, as everywhere in this file; no upper bound is
# preregistered, so none is imposed — a large finite cap is a
# legitimate near-disabled cap, e.g. the pinned 0.99 fixtures).
# Defence in depth, mirroring SF-BL-9397: construction raises a named
# ValueError (FleetRiskConfig.__post_init__), and evaluate re-validates
# at Gate 2.6, immediately after Gate 2.5 and before any read of the
# field (its first read is Gate D), SKIPping INVALID_CONFIG at zero
# size (EC-09), cluster dict untouched, so mutated / unpickled /
# shared-default configs can never reach the bare comparison. Finite
# thresholds and None pass both layers untouched, so their behaviour —
# including the SF-BL-10619 lock-scaled and SF-BL-10726 edge-keyed
# Gate D arms, which read this same field — is bit-identical.
# AMENDED by SF-BL-10877 (below): the domain's upper bound is now 1.

# --- SF-BL-10877: max_round_trip_cost upper bound [0, 1] ----------------
# Filed on the type half of this field (bool True read as the 1.00
# veto threshold; the string '0.05' crashing Gate D with a bare
# TypeError), with prereg domain None or a finite real in [0, 1],
# bool excluded, fixtured at True, False and '0.05'. Kill-criterion
# audit (KILL_AUDIT.md in the task work dir): PARTIALLY met — the
# SF-BL-10798 fix above already TYPE-checks the field (its
# _is_finite_number delegation excludes bool and rejects strings at
# construction and at Gate 2.6, so all three of this task's fixtures
# were already rejected on the inherited tree), but it RANGE-checks
# to [0, inf), not to this prereg's [0, 1]: its block comment imposes
# no upper bound and its test suite pins 2.0 and 1e6 as VALID.
# Verified live on the inherited tree (repro_prefix.txt in the task
# work dir): thresholds 1.0000001 / 1.5 / 2.0 / 1e6 were accepted and
# traded the wide prereg book — at equity $100,000 a 1.5 threshold
# invested the full $15,000 cluster cap that the 0.05 default vetoes,
# the exact figure in this task's title. A cap above 1 is not a
# looser cap, it is a disabled gate in disguise: rt_cost is a
# per-share cost on a $1 binary, so no real round trip can exceed it
# and the veto can never fire — the same fail-open class as
# True == 1.00, one step up. The documented disable is None. This
# task — the later prereg, filed after SF-BL-10798's and enumerating
# [0, 1] explicitly — therefore narrows the shared validator's
# domain to [0, 1]; both enforcement layers (FleetRiskConfig
# construction and Gate 2.6) inherit the bound from the one helper,
# and the messages keep the SF-BL-10798 token its suites assert.
# In-domain behaviour is bit-identical (162-case snapshot in the
# task work dir): None, the 0.05 default, the strict-> boundary at
# 0.405/0.406, 0.99, and int 1 / float 1.0 (the boundary is
# inclusive; only the BOOLS True/False at those values are invalid,
# by type). The bound applies to the FIELD (the base cap) only: the
# SF-BL-10619 lock-scaled EFFECTIVE cap, base * sqrt(lock / 7d), may
# legitimately exceed 1 for an in-domain base on a long lock and is
# not re-validated. Known supersession: SF-BL-10798's pins of 2.0 /
# 1e6 as valid encode its own prereg (all finite thresholds
# bit-identical; no upper bound registered there) and now fail by
# design — documented in KILL_AUDIT.md and DONE-CLAIM.md.
MAX_ROUND_TRIP_COST_MAX = 1.0


def _is_valid_max_round_trip_cost(value: object) -> bool:
    """True iff max_round_trip_cost is None (Gate D disabled) or a
    finite real number in [0, MAX_ROUND_TRIP_COST_MAX] (bool
    excluded) (SF-BL-10798, upper bound SF-BL-10877)."""
    if value is None:
        return True
    return (_is_finite_number(value)
            and 0 <= value <= MAX_ROUND_TRIP_COST_MAX)  # type: ignore[operator]


def _validate_max_round_trip_cost(value: object) -> None:
    """Raise ValueError unless max_round_trip_cost is None or a finite
    real in [0, 1] (SF-BL-10798 / SF-BL-10877)."""
    if not _is_valid_max_round_trip_cost(value):
        raise ValueError(
            f"max_round_trip_cost {value!r} is not None or a finite "
            "real in [0, 1] (SF-BL-10798 / SF-BL-10877): Gate D "
            "vetoes on a bare `rt_cost > max_round_trip_cost` "
            "comparison, so a NaN threshold fails the round-trip "
            "veto open (the prereg fixture's 0.406/share cost traded "
            "$1,500 at the clean 0.05 cap), a +inf threshold "
            "disables the veto silently, a negative threshold "
            "masquerades as an ultra-tight cap, a bool masquerades "
            "as 1.00/0.00, and a finite threshold above 1 can never "
            "veto a per-share cost on a $1 binary — a disabled gate "
            "in disguise (use None to disable Gate D) — instead of "
            "being rejected as a config error")


# --- SF-BL-10878: crypto_kelly_dampener accepts bool / string types -----
# Step 4 of evaluate_and_size_opportunity multiplies the fractional
# Kelly lambda by config.crypto_kelly_dampener on the CRYPTO sizing
# path (domain_mult = kelly_multiplier * crypto_kelly_dampener) with,
# pre-fix, NO type check anywhere: FleetRiskConfig had no validator
# for this field and construction accepted every value silently.
# Python's numeric coercion then decided the trade (all verified live
# pre-fix, repro.py in the task work dir; verified crypto book ask
# 0.60 / bid 0.58, fee 0.01, p=0.70, equity $100,000, mandated
# dampener 0.50 sizing $4,175.13):
#   * True == 1.0 silently REMOVED the PolyBench x0.5 dampener and
#     sized BUY_YES $8,350.25 — exactly double the mandated stake;
#   * False == 0.0 zeroed the target fraction and SKIPped behind the
#     misleading CLUSTER_CAP_REACHED reason (no cluster room was used);
#   * '0.5' and None aborted sizing with bare TypeErrors
#     ("can't multiply sequence by non-int of type 'float'" /
#     "unsupported operand type(s) for *: 'float' and 'NoneType'").
# SF-BL-10821's prereg covers this field's NUMERIC RANGE failures
# (2.0, -1.0, 0.0, NaN) only. Kill-criterion audit (KILL_AUDIT.md in
# the task work dir): NOT met — SF-BL-10821 has NOT landed (its work
# dir holds only TASK.md; no DONE-CLAIM / VERDICT / patch), this file
# carried no dampener validator of any kind pre-fix, and no test in
# the repo type-checks the field (the only test references pass the
# valid literal 0.50) — so there is no landed fix that type-checks
# the field for this task's kill to rest on. The value is therefore
# validated here. The domain is a single predicate covering both
# preregs: a finite real in (0, 1] (SF-BL-10821's range), with type
# enforced by _is_finite_number (bool / non-numeric excluded — the
# file-wide convention), so True is rejected DESPITE equalling the
# in-range 1.0 and '0.5' / None are rejected instead of crashing.
# Defence in depth, mirroring SF-BL-9397 / SF-BL-10798: construction
# raises a named ValueError (FleetRiskConfig.__post_init__), and
# evaluate re-validates at Gate 2.7 (SKIP INVALID_CONFIG /
# INVALID_CRYPTO_KELLY_DAMPENER, zero size, cluster dict untouched),
# immediately after Gate 2.6 and before the field's first read in
# Step 4 — covering mutated / unpickled / shared-default configs.
# SCOPING: Gate 2.7 fires only on the CRYPTO path, exactly as
# SF-BL-10793's Gate 0.9 scopes to where its flag is read — the
# dampener is read ONLY for CRYPTO books (Step 4's domain_mult), so
# a non-crypto book is bit-identical for EVERY dampener value,
# valid or not (verified pre-fix: a PREDICTION_MARKET book sized
# $8,350.25 identically at dampeners 0.50 / True / False / '0.5' /
# None, never reading the field and never crashing). Construction
# validation is unconditional, as for every other FleetRiskConfig
# field: the config object itself is malformed, whatever book it
# later meets. In-domain values (0.50 mandated; 1.0 / int 1 included)
# pass both layers untouched, so valid behaviour is bit-identical.

def _is_valid_crypto_kelly_dampener(value: object) -> bool:
    """True iff crypto_kelly_dampener is a finite real in (0, 1]
    (bool excluded) (SF-BL-10878 type surface; SF-BL-10821 range)."""
    return _is_finite_number(value) and 0.0 < value <= 1.0  # type: ignore[operator]


def _validate_crypto_kelly_dampener(value: object) -> None:
    """Raise ValueError unless crypto_kelly_dampener is a finite
    real in (0, 1] (SF-BL-10878 / SF-BL-10821)."""
    if not _is_valid_crypto_kelly_dampener(value):
        raise ValueError(
            f"crypto_kelly_dampener {value!r} (type "
            f"{type(value).__name__}) is not a finite real in "
            "(0, 1] (SF-BL-10878 / SF-BL-10821): the PolyBench "
            "crypto dampener multiplies the fractional Kelly "
            "lambda on the CRYPTO sizing path, so a bool True "
            "silently removed the mandated x0.5 dampener and "
            "doubled the stake (prereg fixture: $8,350.25 vs "
            "$4,175.13 at 0.50), a bool False zeroed the stake "
            "behind a misleading CLUSTER_CAP_REACHED reason, a "
            "string / None dampener crashed sizing with a bare "
            "TypeError, and an out-of-range dampener inverted or "
            "mislabelled sizing (SF-BL-10821 fixture, equity "
            "$10,000: 2.0 sized at the full $1,500 cluster cap vs "
            "$417.51 at the mandated 0.50, -1.0 SKIPped behind "
            "CLUSTER_CAP_REACHED and NaN behind "
            "ORDER_BOOK_DEPTH_EXHAUSTED)")


# --- SF-BL-10821: crypto_kelly_dampener range (0, 1] -------------------
# This task's prereg is the NUMERIC RANGE half of the dampener surface
# whose TYPE half is SF-BL-10878 above: a dampener outside (0, 1] must
# be rejected or SKIP with a named invalid-dampener reason before
# sizing, fixtured at 2.0, -1.0, 0.0 and NaN, with dampener 0.50
# (mandated) and non-crypto books bit-identical. The single shared
# predicate _is_valid_crypto_kelly_dampener (finite real in (0, 1],
# bool excluded) already enforces exactly this domain at both layers
# (construction raise + Gate 2.7 SKIP INVALID_CRYPTO_KELLY_DAMPENER),
# so no second validator is added; this task verifies the range
# fixtures on its own prereg fixture (verified crypto book ask 0.60 /
# bid 0.58, fee 0.01, p=0.70, equity $10,000 — repro.py in the task
# work dir: unbound arithmetic sizes 2.0 at the $1,500 cluster cap vs
# $417.51 at 0.50, and in-domain 1.0 at exactly double the mandated
# stake, $835.03) and adds its citations to the two rejection
# messages so the range prereg is named on its own surface.
# Kill-criterion audit (PRODUCER_CENSUS.md in the task work dir): NOT
# met — in-repo call sites pass only the literal 0.50, but that is an
# observation, not an invariant: FleetRiskConfig is a mutable
# dataclass (mutation / unpickling / the shared mutable default
# instance all bypass construction), evaluate accepts any config
# from any caller with no provenance, and the normal construction
# path itself accepted 2.0 pre-fix (the task's verified live repros).


# --- SF-BL-10972: min_confidence accepts a bool as a Gate 3 threshold ---
# Gate 3 in evaluate_and_size_opportunity reads
# `p_side < config.min_confidence` (and formats the threshold with
# `:.2f` in its LOW_CONVICTION reason) with, pre-fix, NO validation
# anywhere: FleetRiskConfig.__post_init__ validated the dispersion and
# on-chain fields and the SF-BL-9397 / 10798 / 10878 fields, but never
# min_confidence — the SF-BL-9397 block lists it explicitly as
# SF-BL-10820's separate prereg surface. Python bool is an int
# subclass, so a JSON/env bool coerced silently (all verified live
# pre-fix, repro.py in the task work dir; prereg fixture ask 0.50 /
# bid 0.48, fee 0.01, equity $100,000, p=0.57, p_side/q_top 1.13):
#   * the default 0.60 config SKIPs LOW_CONVICTION;
#   * post-construction min_confidence=False (read as 0.0 at Gate 3)
#     EXECUTED BUY_YES $4,595.96 — a low-conviction candidate the
#     PolyBench conviction gate exists to reject;
#   * min_confidence=True was read as 1.00, silently raising the bar
#     (SKIP LOW_CONVICTION reported as `< 1.00`).
# SF-BL-10820's prereg covers this field's NaN / out-of-range / string
# failures (NaN and -1.0 traded the same $4,595.96 pre-fix; '0.6'
# crashed Gate 3 with a bare TypeError — repro_prepatch.txt), but it
# has NOT landed (its work dir holds only TASK.md), so no landed fix
# covers either surface. Kill-criterion audit (CALLER_CENSUS.md in
# the task work dir): NOT met — the kill requires min_confidence to
# be only ever set by VALIDATED numeric literals and never mutated
# or unpickled, but there was no validator at all, FleetRiskConfig is
# a mutable dataclass whose own test suites set this field on raw
# `object.__new__` instances, unpickling bypasses __post_init__, and
# evaluate's default config is a shared mutable instance. The value
# is therefore validated here, with a single predicate covering both
# preregs: a finite real in [0, 1] (a probability threshold; the
# 0.60 default and both endpoints included), type enforced by
# _is_finite_number (bool / non-numeric excluded — the file-wide
# convention), so False is rejected DESPITE equalling the in-range
# 0.0 and True DESPITE equalling the in-range 1.0. Defence in depth,
# mirroring SF-BL-9397 / SF-BL-10798 / SF-BL-10878: construction
# raises a named ValueError (FleetRiskConfig.__post_init__), and
# evaluate re-validates at Gate 2.8 (SKIP INVALID_CONFIG /
# INVALID_MIN_CONFIDENCE, zero size, cluster dict untouched),
# immediately after Gate 2.7 and before the edge screen and Gate 3 —
# the field's first reads — covering mutated / unpickled /
# shared-default configs. The gate is unconditional (not domain
# scoped): Gate 3 reads the field for every domain. Genuine
# thresholds in [0, 1] (int 0 / 1 included) pass both layers
# untouched, so their behaviour — LOW_CONVICTION skips, the
# ratio >= 1.35 alternative arm, and every trade — is byte-identical.

MIN_CONFIDENCE_MIN = 0.0
MIN_CONFIDENCE_MAX = 1.0


def _is_valid_min_confidence(value: object) -> bool:
    """True iff min_confidence is a finite real in [0, 1]
    (bool excluded) (SF-BL-10972 bool surface; SF-BL-10820
    NaN / range / string surface)."""
    return (_is_finite_number(value)
            and MIN_CONFIDENCE_MIN <= value <= MIN_CONFIDENCE_MAX)  # type: ignore[operator]


def _validate_min_confidence(value: object) -> None:
    """Raise ValueError unless min_confidence is a finite real
    in [0, 1] (SF-BL-10972 / SF-BL-10820)."""
    if not _is_valid_min_confidence(value):
        raise ValueError(
            f"min_confidence {value!r} (type "
            f"{type(value).__name__}) is not a finite real in "
            "[0, 1] (SF-BL-10972 / SF-BL-10820): Gate 3 compares "
            "p_side against this PolyBench conviction threshold "
            "with a bare `<`, so a bool False was read as 0.0 and "
            "traded a low-conviction candidate the default 0.60 "
            "config SKIPs as LOW_CONVICTION (prereg fixture: "
            "BUY_YES $4,595.96), a bool True was read as 1.00 and "
            "silently raised the bar, a NaN threshold never fired "
            "the gate, and a string threshold crashed Gate 3 with "
            "a bare TypeError")


@dataclass
class FleetRiskConfig:
    """Risk and execution governance parameters for the agent fleet."""
    min_net_edge: float = 0.05               # Minimum |p - q_exec| hurdle rate after fees/slippage
    lock_scaled_hurdle_enabled: bool = False # SF-BL-3839: scale min_net_edge by sqrt(lock / 7d) per candidate (floor at the flat base); False = flat hurdle (A/B)
    lock_hurdle_ref_days: float = 7.0        # SF-BL-3839: reference lock D_REF for the lock-scaled hurdle
    dispersion_taper_enabled: bool = False # SF-BL-9639: replace Gate 2's binary dispersion cliff with a continuous taper of lambda (factor max(0, 1 - dispersion / max_inter_agent_dispersion)); False = binary gate (A/B)
    min_confidence: float = 0.60             # PolyBench minimum directional conviction c >= 0.60
    kelly_multiplier: float = 0.35           # Fractional Kelly multiplier (0.25 to 0.50)
    crypto_kelly_dampener: float = 0.50      # Extra 0.5x dampener for Crypto (PolyBench Section 5.2)
    max_cluster_allocation: float = 0.15     # Prediction Arena 15% cap per correlated event cluster
    max_inter_agent_dispersion: float = 0.22 # Trigger supervisor research if sub-agents diverge > 0.22
    require_on_chain_for_crypto: bool = True # CryptoBench gate
    max_round_trip_cost: Optional[float] = 0.05 # SF-BL-7432 Gate D: veto if q_exec - size-walked exit bid (net of exit fee) exceeds this; None disables; domain None or finite real in [0, 1] (SF-BL-10798 / SF-BL-10877)
    round_trip_lock_scaled_enabled: bool = False # SF-BL-10619: scale the Gate D cap by sqrt(lock / 7d) per candidate (lock_scaled_max_round_trip_cost); False = flat cap (A/B)
    round_trip_lock_ref_days: float = 7.0        # SF-BL-10619: reference lock D_REF for the lock-scaled Gate D cap
    round_trip_edge_keyed_enabled: bool = False  # SF-BL-10726: Gate D vetoes iff rt_cost >= the candidate's own entry edge (p_side - q_exec) instead of the flat/lock-scaled cap (edge_keyed_gate_d_veto); False = flat cap (A/B)
    round_trip_relative_enabled: bool = False  # SF-BL-11194: Gate D additionally vetoes iff rt_cost / q_exec > round_trip_relative_max_share (relative_round_trip_veto) — the absolute cap is vacuous for q_exec <= cap; False = absolute cap only (A/B)
    round_trip_relative_max_share: float = 0.50  # SF-BL-11194: pre-registered share of entry price a round trip may consume before the relative Gate D prong vetoes
    budget_sensitivity_veto_enabled: bool = True # SF-BL-3659 Gate BS: veto if the post-cost edge flips sign or D_KL per dollar halves when the Kelly target budget is re-swept at 0.5x / 2x; False disables (A/B use)

    def __post_init__(self) -> None:
        # Merged construction validation (SF-BL-10844 + SF-BL-10845,
        # landed concurrently — a single __post_init__ must call both,
        # since a second definition would shadow the first):
        # SF-BL-10844: validate ONLY the Gate 2 threshold here (see its
        # block comment above); evaluate re-validates it at Gate 2 for
        # configs mutated / unpickled after construction.
        _validate_max_inter_agent_dispersion(self.max_inter_agent_dispersion)
        # SF-BL-10845: reject a non-bool gate switch at construction,
        # before truthiness can arm or disarm Gate 1. Only this field is
        # validated (its own prereg surface); evaluate re-validates it
        # at entry for mutated/unpickled configs, which bypass this.
        _validate_require_on_chain_for_crypto(self.require_on_chain_for_crypto)
        # SF-BL-9397: validate the three sizing-band fields here (see
        # the SF-BL-9397 block above the validators); evaluate
        # re-validates them at Gate 2.5 for configs mutated / unpickled
        # after construction.
        _validate_kelly_multiplier(self.kelly_multiplier)
        _validate_max_cluster_allocation(self.max_cluster_allocation)
        _validate_min_net_edge(self.min_net_edge)
        # SF-BL-10798 / SF-BL-10877: validate the Gate D round-trip
        # cap here (see the SF-BL-10798 / 10877 blocks above the
        # validators; domain None or a finite real in [0, 1]).
        # Evaluate re-validates it at Gate 2.6 for configs mutated /
        # unpickled after construction. Validated last so the
        # SF-BL-9397 fields keep their first-violation order.
        _validate_max_round_trip_cost(self.max_round_trip_cost)
        # SF-BL-10878: validate the crypto dampener here (see the
        # SF-BL-10878 block above the validators); evaluate
        # re-validates it at Gate 2.7 on the CRYPTO sizing path for
        # configs mutated / unpickled after construction. Validated
        # last so every earlier field keeps its first-violation order
        # at construction.
        _validate_crypto_kelly_dampener(self.crypto_kelly_dampener)
        # SF-BL-10972 / SF-BL-10820: validate the Gate 3 conviction
        # threshold here (see the SF-BL-10972 block above the
        # validators; domain a finite real in [0, 1], bool excluded);
        # evaluate re-validates it at Gate 2.8 for configs mutated /
        # unpickled after construction. Validated last so every
        # earlier field keeps its first-violation order.
        _validate_min_confidence(self.min_confidence)


# --- SF-BL-7432: Gate D — size-walked exit bid & round-trip cost ---------
# evaluate_and_size_opportunity historically gated only on the ENTRY sweep:
# a candidate passed when p_side - q_exec >= min_net_edge, even though
# NORTH_STAR.md marks every position at its liquidation bid (CWR_bid) and
# the nstar live probe shows entry price and bid diverging sharply on thin
# books. A trade whose immediate liquidation at the SIZE-WALKED bid (not
# the top-of-book best bid stored in liquidation_bid_price) already loses
# more than the minimum edge books a negative bid-marked return the moment
# it is taken. Gate D therefore walks the exit ladder for exactly the
# acquired share count:
#   BUY_YES exits on the YES bid ladder (descending);
#   BUY_NO  exits on the derived NO bid ladder, 1 - yes_ask (ascending ask).
# Only live levels carrying a tradeable price count (SF-BL-9398 / SF-BL-9542
# filters, same as the entry sweep). The VWAP covers the filled portion
# only — the fleet convention in nstar.py's vwap_for_dollars, which reports
# fill_fraction separately — so when bid depth cannot absorb the position
# the measured cost is a LOWER bound on the true round-trip cost (stated
# in the task PREREG). No exit depth at all yields exit VWAP 0.0.
# (SF-BL-9844 supersedes that lower-bound convention FOR GATE D PRICING:
# round_trip_cost now scales the filled VWAP by fill_fraction, valuing
# the unfilled remainder at 0 — see its block comment. This function's
# (filled VWAP, fill) return pair is unchanged.)
# The net exit price charges the exit fee symmetrically to the entry fee
# already inside q_exec: q_exit = bid_vwap * (1 - fee_rate), and the
# round-trip cost per share is rt_cost = q_exec - q_exit.

# --- SF-BL-11003: size_walked_exit_bid validates its book argument -----
# size_walked_exit_bid is Gate D's exit-walk entry point and is directly
# callable (round_trip_cost and the SF-BL-7432 gate tests call it
# directly), yet pre-fix it dereferenced book.yes_bids / book.yes_asks
# with NO type check, so a non-OrderBookSnapshot book aborted the walk
# with a bare AttributeError instead of a named failure. Verified live
# pre-fix (repro.py in the task work dir; clean book bids 0.48x1000 /
# asks 0.50x1000 returns (0.48, 1.0) for 100 shares BUY_YES):
# book=None raised AttributeError 'NoneType' object has no attribute
# 'yes_bids', and book=dict (the shape forward snapshot JSON
# materialises as) and book='book' raised the same bare AttributeError
# on yes_bids / yes_asks. Distinct function from SF-BL-10963
# (evaluate_and_size_opportunity's book) and SF-BL-10964
# (sweep_order_book's book): neither reaches this entry point.
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT
# met — the two in-module callers (round_trip_cost, and
# bid_ladder_liquidation_mark via settle_or_mark_trade) pass books
# they were themselves handed with no isinstance invariant, repo
# tests and the SF-BL-7432 gate test call the function directly with
# bare values, and OrderBookSnapshot's annotation is unenforced at
# runtime, so no tested invariant guarantees a constructed snapshot.
# The fix therefore validates the book's TYPE here, at the
# consumption point, BEFORE any dereference and before the shares /
# side early returns — a named ValueError, matching the
# SF-BL-9882 / SF-BL-9811 / SF-BL-10466 raises in round_trip_cost:
# direct callers fail loudly and identifiably instead of crashing on
# an attribute name, and the raise precedes even a SKIP-side /
# invalid-shares no-op call so an invalid book can never be masked
# as a (0.0, 0.0) no-depth result. isinstance, not duck-typing: a
# dict carrying yes_bids keys, or an object with a yes_bids
# attribute, is not a snapshot whose ladders were construction-
# validated (fee bound, ladder order, lockup domain), and accepting
# it would walk unvalidated rungs. Genuine snapshots (subclasses
# included) pass bit-identically; the shares / side (0.0, 0.0)
# contracts for a VALID book are unchanged, so
# bid_ladder_liquidation_mark's SF-BL-3660 no-exit convention and
# round_trip_cost's Gate D pricing are untouched on valid books.
# Scope note: round_trip_cost's OWN book dereference
# (_validate_fee_rate(book.fee_rate), before it calls this function)
# is SF-BL-11004's separate scope and is deliberately untouched here.

def _is_exit_book_snapshot(value: object) -> bool:
    """True iff value is an OrderBookSnapshot instance (SF-BL-11003).

    Named for this exit-walk gate (not the generic
    _is_order_book_snapshot the sibling sweep / round-trip gates
    define) so the three book-type gates never shadow each other.
    """
    return isinstance(value, OrderBookSnapshot)


def _validate_exit_book(book: object) -> None:
    """Raise ValueError unless book is an OrderBookSnapshot."""
    if not _is_exit_book_snapshot(book):
        raise ValueError(
            f"INVALID_BOOK: book {book!r} "
            f"(type {type(book).__name__}) is not an "
            "OrderBookSnapshot (SF-BL-11003): size_walked_exit_bid "
            "walks book.yes_bids / book.yes_asks, so a non-snapshot "
            "book (None, a dict from a JSON-loaded snapshot, a "
            "string) pre-fix raised a bare AttributeError on "
            "yes_bids instead of failing with a named error; "
            "refusing to walk an exit ladder that was never "
            "construction-validated")


# --- SF-BL-11005: size_walked_exit_bid validates the TYPE of shares ---
# shares is a count of contracts, so before any walk it must at least
# be a finite real number. Pre-fix this function folded EVERY invalid
# count into its (0.0, 0.0) no-fill return: verified live pre-fix
# (repro.py in the task work dir; ask 0.50 / bid 0.48 book) shares=
# '100', True and None all returned (0.0, 0.0), indistinguishable from
# genuine no-depth — a caller's type error was read as an unexitable
# book, and any direct consumer pricing off the walk fabricated a
# full-price cost from it (the pattern SF-BL-10466 fixed inside
# round_trip_cost). Kill-criterion audit (CALLER_CENSUS.md in the
# task work dir): PARTIALLY met — SF-BL-10466's landed fix DOES
# type-check shares (via _is_finite_number) as well as range-check
# it, but only inside round_trip_cost; it explicitly left this
# primitive unchanged, so the type dimension remained open here.
# The fix raises a named ValueError when shares is not a finite real
# (bool, string, None, NaN, +/-inf, or an int beyond float range —
# _is_finite_number is OverflowError-safe). SCOPE: this is the
# TYPE/finiteness gate only. A finite shares <= 0 keeps the
# documented (0.0, 0.0) no-position return — that value dimension is
# SF-BL-10466's, which deliberately preserved the primitive's
# contract (its agreed test pins size_walked(..., 0.0) == (0.0,
# 0.0)); positivity is enforced where a cost is fabricated from the
# walk (round_trip_cost's SF-BL-10466 validator, which runs before
# this walk and keeps precedence) and bid_ladder_liquidation_mark
# pre-guards with the same _is_finite_number check, so no in-module
# caller can reach this raise and both of their contracts are
# byte-identical. Ordering: after the SF-BL-11003 book check, whose
# precedence over the shares returns is pinned by its own task.
# Positive numeric share counts are bit-identical.

def _validate_exit_walk_shares(shares: object) -> None:
    """Raise ValueError unless shares is a finite real (SF-BL-11005)."""
    if not _is_finite_number(shares):
        raise ValueError(
            f"shares {shares!r} is not a finite real number "
            f"(type {type(shares).__name__}) (SF-BL-11005): a share "
            "count that is not a finite real number (bool excluded) "
            "is a caller type error, not a position — pre-fix "
            "shares='100', shares=True and shares=None all returned "
            "the (0.0, 0.0) no-fill result, indistinguishable from "
            "genuine no-depth, so a position that was never walked "
            "could be priced as unexitable; refusing to walk a "
            "non-numeric share count")


# --- SF-BL-11047: size_walked_exit_bid validates its exit ladder -----
# The function's own guards covered the book's TYPE (SF-BL-11003) and
# the shares TYPE (SF-BL-11005), but the ladder it actually walks was
# read with no shape check at all: _live_levels iterates the ladder
# and _is_live_level dereferences lvl.size_shares unconditionally.
# Verified live pre-fix (repro.py in the task work dir; clean book
# bids 0.48x1000 / asks 0.50x1000 returns (0.48, 1.0) for 100 shares
# BUY_YES): a book mutated post-construction to yes_bids=None raised
# a bare TypeError ('NoneType' object is not iterable), yes_bids=42
# the same TypeError ('int' object is not iterable), and a dict bid
# entry raised a bare AttributeError ('dict' object has no attribute
# 'size_shares') — likewise for yes_asks on the BUY_NO side, whose
# exit ladder is derived from yes_asks. A malformed ladder cannot be
# CONSTRUCTED (OrderBookSnapshot.__post_init__ iterates the ladders
# and crashes on these shapes itself), so these books arrive via
# post-construction mutation / unpickling — the bypass every sibling
# gate documents — or from a direct caller handing this function a
# snapshot-shaped object whose ladders were never validated.
# Distinct from SF-BL-11003 (the book argument's own type) and from
# SF-BL-10826 (ladder validation scoped to evaluate, whose gates
# direct size_walked_exit_bid / round_trip_cost callers bypass, as
# SF-BL-9882's census established for this call family).
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT
# met — the two in-module callers (round_trip_cost,
# bid_ladder_liquidation_mark) forward books with no ladder
# invariant, repo tests call the function directly and assign
# ladders post-construction (one suite already assigns
# book.yes_bids = None into this call family), and the snapshot
# dataclass is mutable, so no tested invariant guarantees
# constructor-validated ladders. The fix therefore validates the
# SELECTED exit ladder's shape here, at the consumption point: the
# container must be a list/tuple and every entry an OrderBookLevel
# (subclasses pass), else a named ValueError — the prereg's
# named-ValueError branch, matching SF-BL-11003 / SF-BL-11005 for
# this same function. (The prereg's alternative, (0.0, 0.0) as
# no-depth, was NOT chosen: silently pricing a malformed ladder as
# unexitable would fabricate a full round-trip cost in Gate D,
# exactly the failure SF-BL-11005 removed for shares.)
# Ordering / scope, deliberately pinned: validation runs AFTER the
# SF-BL-11003 book check and the SF-BL-11005 shares-type check, and
# AFTER the shares<=0 no-position return — a call with no position
# performs no walk, so its documented (0.0, 0.0) contract is
# unchanged even on a malformed book — and only the ladder the side
# actually walks is validated (BUY_YES -> yes_bids, BUY_NO ->
# yes_asks): a SKIP / other side returns (0.0, 0.0) without reading
# any ladder, unchanged. Within a walked side, the WHOLE ladder is
# scanned before any fill, because pre-fix _live_levels iterated
# every entry regardless of fill depth, so a dict entry beyond the
# filled rungs crashed too — no malformed entry may be hidden by a
# shallow position. Entry FIELD types are not re-checked here: a
# genuine OrderBookLevel whose fields bypassed their own guards is
# filtered identical-to-clean by _is_live_level /
# _is_tradeable_price (SF-BL-9398/9542, field types SF-BL-11492),
# exactly as pre-fix. Well-formed ladders (list/tuple of
# OrderBookLevel) are bit-identical.

def _validate_exit_ladder(ladder: object, ladder_name: str) -> None:
    """Raise ValueError unless ladder is a list/tuple of
    OrderBookLevel entries (SF-BL-11047)."""
    if not isinstance(ladder, (list, tuple)):
        _repr = repr(ladder)
        if len(_repr) > 200:
            _repr = _repr[:200] + "...<truncated>"
        raise ValueError(
            f"INVALID_EXIT_LADDER: {ladder_name} {_repr} "
            f"(type {type(ladder).__name__}) is not a list/tuple "
            "of OrderBookLevel (SF-BL-11047): size_walked_exit_bid "
            "walks this ladder as Gate D's exit depth, so a None / "
            "non-iterable ladder (a mutated or unpickled snapshot, "
            "or a JSON-materialised book) pre-fix raised a bare "
            "TypeError ('NoneType' object is not iterable) instead "
            "of failing with a named error; refusing to walk an "
            "exit ladder that was never construction-validated")
    for idx, entry in enumerate(ladder):
        if not isinstance(entry, OrderBookLevel):
            _repr = repr(entry)
            if len(_repr) > 200:
                _repr = _repr[:200] + "...<truncated>"
            raise ValueError(
                f"INVALID_EXIT_LADDER: {ladder_name}[{idx}] {_repr} "
                f"(type {type(entry).__name__}) is not an "
                "OrderBookLevel (SF-BL-11047): size_walked_exit_bid "
                "reads price / size_shares off every entry of its "
                "exit ladder, so a non-level entry (e.g. the dict a "
                "JSON snapshot materialises as) pre-fix raised a "
                "bare AttributeError on size_shares instead of "
                "failing with a named error; refusing to walk an "
                "exit ladder containing an unvalidated entry")


def size_walked_exit_bid(
    book: OrderBookSnapshot,
    side: TradeAction,
    shares: float,
) -> Tuple[float, float]:
    """VWAP bid obtained by immediately liquidating `shares` on `side`.

    Returns (bid_vwap_over_filled_portion, fill_fraction). (0.0, 0.0) when
    there is no live, tradeably-priced exit depth or `shares` <= 0.

    Raises ValueError if `book` is not an OrderBookSnapshot
    (SF-BL-11003), before any ladder is read — a non-snapshot book
    must fail with a named error, never a bare AttributeError on
    yes_bids / yes_asks and never a masked (0.0, 0.0) no-depth return.
    Raises ValueError if `shares` is not a finite real number
    (SF-BL-11005: bool / non-numeric / NaN / infinite counts are a
    caller type error, never a no-depth result).
    Raises ValueError if the exit ladder selected by `side`
    (yes_bids for BUY_YES, yes_asks for BUY_NO) is not a list/tuple
    of OrderBookLevel entries (SF-BL-11047: a None / non-iterable
    ladder or a non-level entry is a caller type error, never a
    bare TypeError / AttributeError and never a masked (0.0, 0.0)
    no-depth result). The validation runs after the shares<=0
    no-position return, so a no-position call keeps (0.0, 0.0)
    even on a malformed book, and a non-walked side never reads a
    ladder.
    """
    # SF-BL-11003: validate the book's type first — see the block
    # comment above. Precedes the shares / side early returns so no
    # invalid book is masked as a no-depth (0.0, 0.0) result.
    _validate_exit_book(book)
    # SF-BL-11005: type/finiteness gate for shares, after the book
    # check (SF-BL-11003 precedence) and before any ladder read — see
    # the SF-BL-11005 block comment above _validate_exit_walk_shares.
    # Finite <= 0 keeps the (0.0, 0.0) no-position return below
    # (SF-BL-10466 scope line: the value dimension is enforced by
    # round_trip_cost's own validator, not by this primitive).
    _validate_exit_walk_shares(shares)
    if shares <= 0:  # type: ignore[operator]
        return 0.0, 0.0
    if side == TradeAction.BUY_YES:
        # SF-BL-11047: validate the exit ladder's shape before any
        # entry is read — see the block comment above
        # _validate_exit_ladder. Runs after the shares<=0 return
        # above, so no-position calls keep their (0.0, 0.0) contract.
        _validate_exit_ladder(book.yes_bids, "yes_bids")
        ladder = [
            lvl for lvl in _live_levels(book.yes_bids)
            if _is_tradeable_price(lvl.price)
        ]
        ladder.sort(key=lambda lvl: lvl.price, reverse=True)
    elif side == TradeAction.BUY_NO:
        # SF-BL-11047: the BUY_NO exit ladder is derived from
        # yes_asks, so that ladder is the one validated here.
        _validate_exit_ladder(book.yes_asks, "yes_asks")
        ladder = []
        for lvl in _live_levels(book.yes_asks):
            if not _is_finite_number(lvl.price):
                continue
            derived_bid = round(1.0 - lvl.price, 6)  # type: ignore[operator]
            if _is_valid_contract_price(derived_bid):
                ladder.append(
                    OrderBookLevel(price=derived_bid, size_shares=lvl.size_shares)
                )
        ladder.sort(key=lambda lvl: lvl.price, reverse=True)
    else:
        return 0.0, 0.0

    remaining = float(shares)  # type: ignore[arg-type]
    filled = 0.0
    proceeds = 0.0
    for lvl in ladder:
        if remaining <= 1e-12:
            break
        take = min(remaining, lvl.size_shares)
        proceeds += take * lvl.price
        filled += take
        remaining -= take
    if filled <= 0:
        return 0.0, 0.0
    return proceeds / filled, filled / float(shares)  # type: ignore[arg-type]


# --- SF-BL-9882: round_trip_cost re-validates book.fee_rate ---------------
# The exit side charges the fee as q_exit = bid_vwap * (1 - fee_rate),
# the mirror of the entry side's (1 + fee_rate) — but pre-fix this
# function read book.fee_rate with NO range/finiteness check, unlike
# sweep_order_book (SF-BL-9363 re-validates there) and unlike
# evaluate_and_size_opportunity's Gate 0. Verified live pre-fix on the
# task fixture (ask 0.60 / walked bid 0.58, q_exec 0.606, clean fee 0.01
# -> rt_cost 0.0318): a post-construction fee_rate of -0.50 returned
# rt_cost -0.264 with q_exit 0.870 ABOVE the 0.58 walked bid (a negative
# fee pays the seller a premium to exit, so Gate D's
# `rt_cost > max_round_trip_cost` is False and the veto never fires);
# fee_rate 2.0 returned q_exit -0.58 (a negative exit price); fee_rate
# NaN returned rt_cost NaN, and `NaN > max` is likewise False, so the
# veto fails open. Kill-criterion audit (CALLER_CENSUS.md in the task
# work dir): NOT met — OrderBookSnapshot is a mutable (non-frozen)
# dataclass, so __post_init__ is bypassed by post-construction mutation
# and by unpickling, and round_trip_cost is directly callable (the
# SF-BL-7432 gate test and the SF-BL-7432 S2 evaluator call it
# directly), bypassing evaluate's Gate 0 entirely; the only in-evaluate
# caller reaches it after Gate 0, but the function itself performed no
# re-validation, exactly the gap SF-BL-9363 closed in the sweep. The
# fix therefore re-validates at this consumption point, before the
# exit walk. Bound: the pre-registered SF-BL-9363 bound
# [FEE_RATE_MIN, FEE_RATE_MAX] = [0, 0.10] via _validate_fee_rate —
# a strict subset of this task's prereg domain [0, 1), so every fee
# outside [0, 1) is rejected, and fees in (0.10, 1) are rejected too
# for consistency with the constructor, Gate 0 and the sweep (every
# in-repo fee is <= 0.0175; venue templates <= ~0.07). Rejection is a
# ValueError (the prereg's "reject" option), matching sweep_order_book:
# direct callers fail loudly instead of reading a fail-open cost, and
# the evaluate path is unaffected because Gate 0 SKIPs an invalid fee
# before Gate D is ever reached. Post-fix invariant for any accepted
# fee: bid_vwap lies in [0, 1] (size_walked_exit_bid only walks
# tradeably-priced live levels and returns 0.0 with no depth), so
# q_exit = bid_vwap * (1 - fee_rate) lies in [0, bid_vwap] — it can
# never exceed the walked bid — and rt_cost = q_exec - q_exit is
# finite and never negative-by-construction from the fee whenever
# q_exec itself is finite (q_exec validation is SF-BL-9811's separate
# scope and is deliberately untouched here).

# --- SF-BL-9844: Gate D values the unfilled exit remainder at 0 --------
# Pre-fix, round_trip_cost priced the exit as
# q_exit = bid_vwap * (1 - fee_rate), where bid_vwap is the VWAP over
# the FILLED portion only (size_walked_exit_bid) — the fill_fraction it
# also returned was bound by evaluate_and_size_opportunity to
# _exit_fill and never read, so Gate D measured a lower bound on the
# round-trip cost while treating it as the cost itself. Verified live
# pre-fix (repro.py in the task work dir): asks 0.60x10000 /
# bids 0.58x100, fee_rate 0.0 — liquidating 1000 shares returns
# (0.58, fill 0.10) and rt_cost 0.020, below the 0.05 Gate D cap, so
# evaluate sized BUY_YES 2500 shares (exit fill 0.04); valuing the
# unfilled 900 shares at 0 gives exit VWAP 0.058 and rt_cost 0.542.
# Kill-criterion audit (CENSUS.md in the task work dir): NOT met — in
# the SF-BL-7432 forward-book census (candidates.jsonl, 2637
# entry-passing candidates over real forward books), 372 (14.11%) have
# exit fill < 1 and 323 of them were NOT vetoed (sized) under the
# pre-fix cost, median fill 0.328 — fill<1 exits are routinely sized,
# so the lower-bound convention is not documented-and-kept; the cost
# is fixed. The fix mirrors bid_ladder_liquidation_mark (SF-BL-3660):
# the effective per-share exit price is
#   q_exit = bid_vwap * fill_fraction * (1 - fee_rate),
# i.e. filled proceeds net of the exit fee, spread over ALL acquired
# shares, with the unfilled remainder contributing 0 — the fee is
# charged only on proceeds actually received. size_walked_exit_bid
# itself is unchanged (its filled-portion VWAP + fill pair is the
# correct primitive; SF-BL-3660 already combines them this way).
# Full-fill exits (fill_fraction == 1.0) are bit-identical to pre-fix,
# as is the no-depth case (bid_vwap 0.0, fill 0.0 -> q_exit 0.0).

# --- SF-BL-9811: round_trip_cost validates its q_exec argument ----------
# q_exec is an execution price for a binary contract, so its valid
# domain is (0, 1]: a price must be positive and cannot exceed the
# $1.00 maximum payout (1.0 itself is admissible — the preregistered
# domain is open at 0, closed at 1). Pre-fix this function used
# q_exec in `q_exec - q_exit` with NO check at all, while its only
# gating caller (Gate D in evaluate_and_size_opportunity) vetoes on a
# bare `rt_cost > max_round_trip_cost` comparison. Verified live
# pre-fix on the task fixture (ask 0.60 / bid 0.58, fee 0.01, shares
# 1000; clean q_exec=0.60 -> rt_cost 0.0258, q_exit 0.5742, fill 1.0):
# q_exec=NaN returned rt_cost NaN — `NaN > max` is False, so the
# SF-BL-7432 round-trip veto fails open and Gate D is passed by a NaN
# cost; q_exec=-0.20 returned rt_cost -0.7742, and q_exec=0.0 / -inf
# likewise returned negative / -inf costs no veto can fire on;
# q_exec=1.5 and +inf returned costs for prices beyond the binary
# maximum payout. Kill-criterion audit (CALLER_CENSUS.md in the task
# work dir): NOT met — the in-evaluate Gate D call is shielded only
# incidentally, by composing the SF-BL-9438 non-finite guard, the
# SF-BL-8891 parity backstop and the depth check, none of which
# validates q_exec for this function; every direct caller passes an
# unsanitised float (repo tests pass bare literals 0.505 / 0.606; the
# SF-BL-7432 and SF-BL-7785 S2 evaluators guard only
# `inv <= 0 or shares <= 0`, comparisons that are False for NaN, so a
# NaN swept q_exec flows straight in), and round_trip_cost is
# directly callable. The fix therefore validates q_exec here, at the
# consumption point, BEFORE the exit walk — a named ValueError,
# matching the SF-BL-9882 fee validation in this same function.
# Bool / non-numeric q_exec is rejected via _is_finite_number (the
# module's established discipline, SF-BL-9676): True masquerades as
# the in-domain price 1.0 under a bare range check, and a string /
# None cannot be ordered against the payoff at all. Post-fix
# invariant: for any accepted q_exec, q_exit lies in [0, bid_vwap],
# a subset of [0, 1] (SF-BL-9882 fee bound + SF-BL-9844 fill scaling:
# the walk only takes tradeably-priced live levels, fee in
# [0, 0.10], fill_fraction in [0, 1]), so rt_cost = q_exec - q_exit
# is always finite — Gate D can never again be passed by a NaN (or
# -inf) cost produced here. The evaluate path is unaffected: its
# q_exec already reaches Gate D finite and in (0, 1), so the raise
# is never reached via evaluate.

def _q_exec_in_round_trip_domain(q_exec: object) -> bool:
    """True iff q_exec is a finite real number in (0, 1] (SF-BL-9811)."""
    return _is_finite_number(q_exec) and 0.0 < q_exec <= 1.0  # type: ignore[operator]


# --- SF-BL-11080: round_trip_cost validates the TYPE of its q_exec ----
# Filed 2026-10-22 on the TYPE dimension alone: at filing, '0.606' and
# None crashed with a bare TypeError at the `q_exec - q_exit`
# subtraction and True silently returned rt_cost 0.4258 (True == 1.0,
# the maximum payout) against 0.0318 for the clean 0.606 fixture at
# 100 shares — the exact values reproduced from the pre-SF-BL-9811
# baseline in this task's historical_repro.py / repro_prefix.txt.
# SF-BL-9811 landed 2026-10-23 (after this task was filed) and its
# _validate_round_trip_q_exec already rejects bool / non-numeric
# q_exec via _is_finite_number, so the substantive done criterion is
# met in the live tree at this task's own fixture (100 shares, not
# SF-BL-9811's 1000). Kill-criterion audit (KILL_AUDIT.md in the task
# work dir): the kill as literally written — SF-BL-9882's landed fee
# fix also validating q_exec's type — is NOT met: SF-BL-9882's
# _validate_fee_rate validates only book.fee_rate. The type coverage
# comes from the separate SF-BL-9811 fix, not from SF-BL-9882. This
# task's remaining scope is therefore verification at its own fixture
# plus traceability: the validator's error now also cites the
# SF-BL-11080 type dimension, retaining every substring SF-BL-9811's
# agreed tests pin ("q_exec", "SF-BL-9811", "(0, 1]"). No logic
# change: finite-q_exec results are bit-identical.

def _validate_round_trip_q_exec(q_exec: object) -> None:
    """Raise ValueError unless q_exec is in the domain (0, 1]."""
    if not _q_exec_in_round_trip_domain(q_exec):
        raise ValueError(
            f"q_exec {q_exec!r} outside pre-registered domain (0, 1] "
            "(SF-BL-9811; non-numeric-type dimension SF-BL-11080): "
            "an execution price must be a finite real "
            "number, strictly positive and at most the $1.00 binary "
            "maximum payout — pre-fix q_exec=NaN returned a NaN "
            "round-trip cost that Gate D's bare "
            "`rt_cost > max_round_trip_cost` comparison can never "
            "veto on (NaN > max is False), and q_exec=-0.20 returned "
            "rt_cost -0.7742; refusing to price the exit walk off an "
            "impossible entry price")


# --- SF-BL-10466: round_trip_cost validates its shares argument ---------
# shares is the size of the position being liquidated, so its valid
# domain is a finite real strictly > 0: a round-trip cost is defined
# only for a position that exists. Pre-fix this function never
# validated shares at all: it passed the value straight to
# size_walked_exit_bid, whose documented no-fill return for a
# non-finite or <= 0 count is (0.0, 0.0), and converted that into
# rt_cost = q_exec - 0.0 — a fabricated full-price cost. Verified live
# pre-fix (repro.py in the task work dir; frozen fixture ask 0.60 /
# bid 0.58, fee 0.01, q_exec=0.65): shares=NaN, shares=-100 and
# shares=0 ALL returned (0.65, 0.0, 0.0), as did +/-inf, -1e-9, True,
# '1000', None and 10**400 — Gate D then compared that invented 0.65
# against its veto threshold as if a real position cost that much to
# exit, when in fact there is no position to exit at all. This is
# distinct from SF-BL-9811 (the q_exec argument) and SF-BL-9882
# (fee_rate) in the same function, which are deliberately untouched.
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT
# met — the in-evaluate Gate D call is shielded only incidentally by
# evaluate's earlier finiteness + `invested <= 0 or shares <= 0`
# depth checks, neither of which is a validation for this function;
# the SF-BL-7432 and SF-BL-7785 S2 evaluators guard only
# `inv <= 0 or shares <= 0` (both False for NaN, so a NaN swept share
# count flows straight in), the SF-BL-3583 forward evaluator passes
# base/res shares with no shares check at all, repo tests pass bare
# literals, and round_trip_cost is directly callable. The fix
# therefore validates shares here, at the consumption point, BEFORE
# the exit walk — a named ValueError, matching the SF-BL-9811 /
# SF-BL-9882 validations in this same function. Bool / non-numeric
# shares are rejected via _is_finite_number (the module's established
# discipline): True masquerades as a 1-share position under a bare
# range check, a string / None cannot be ordered against the ladder,
# and a huge int (10**400) raises the named ValueError rather than
# leaking an OverflowError. size_walked_exit_bid itself is NOT
# changed: its (0.0, 0.0) no-fill contract is deliberate and
# load-bearing for bid_ladder_liquidation_mark (SF-BL-3660, "no
# exit — mark at zero"); the fabrication happens in this function,
# which turns a no-position walk into a Gate D cost. ORDERING: fee
# (SF-BL-9882) first, then q_exec (SF-BL-9811), then shares — the two
# earlier validators are pinned by their agreed tests on no-op calls
# carrying shares=0.0 (a mutated-fee SKIP-side call and a NaN-q_exec
# SKIP-side call must still raise THEIR named errors), and all three
# raises precede any pricing, so the ordering costs nothing here.
# The evaluate path is unaffected: it never reaches Gate D with a
# non-finite or <= 0 share count, so the raise is never reached via
# evaluate, and every positive-share result is bit-identical.

def _shares_in_round_trip_domain(shares: object) -> bool:
    """True iff shares is a finite real number > 0 (SF-BL-10466)."""
    return _is_finite_number(shares) and shares > 0  # type: ignore[operator]


def _validate_round_trip_shares(shares: object) -> None:
    """Raise ValueError unless shares is in the domain (0, finite]."""
    if not _shares_in_round_trip_domain(shares):
        raise ValueError(
            f"shares {shares!r} outside pre-registered domain "
            "(0, finite] (SF-BL-10466; non-numeric-type dimension "
            "SF-BL-11005): shares must be a finite real "
            "number > 0 — a non-finite or <= 0 share count is a "
            "position that does not exist, and pre-fix shares=NaN, "
            "shares=-100 and shares=0 all returned rt_cost 0.65 at "
            "q_exec=0.65, a fabricated full-price round-trip cost "
            "manufactured from size_walked_exit_bid's (0.0, 0.0) "
            "no-fill result that Gate D then compared against its "
            "veto threshold; refusing to price an exit walk for a "
            "position that does not exist")


# --- SF-BL-11004: round_trip_cost validates the TYPE of its book ------
# Every other argument of this function is validated at this
# consumption point — fee_rate (SF-BL-9882), q_exec (SF-BL-9811),
# shares (SF-BL-10466) — but the book itself was dereferenced
# unguarded: the function's first statement was
# `_validate_fee_rate(book.fee_rate)`, so a book that is not an
# OrderBookSnapshot crashed with a bare AttributeError before any
# validation, fee check or exit pricing could run. Verified live
# pre-fix (repro.py in the task work dir; clean fixture ask 0.50 /
# bid 0.48, fee 0.01, shares 1000, q_exec 0.505 -> (0.0298, 0.4752,
# 1.0)): book=None raised AttributeError "'NoneType' object has no
# attribute 'fee_rate'" and book={} raised AttributeError "'dict'
# object has no attribute 'fee_rate'", both at the SF-BL-9882 line;
# the same bare AttributeError fired for a str / int / list book and
# for an OrderBookLevel passed as the book, and even on a SKIP-side /
# zero-share no-op call. A dict is the realistic producer shape here:
# the S2 / forward evaluators materialise archived books from
# json.load-ed rows, and a missed construction step (or a missing
# snapshot lookup returning None) hands this Gate D cost function
# exactly those values. Distinct from SF-BL-10963 (evaluate's book
# argument) and SF-BL-10964 (the sweep's book argument): this is
# round_trip_cost's own book argument, reached by direct callers who
# never pass through evaluate or the sweep, and those two functions
# are deliberately untouched here. Kill-criterion audit
# (CALLER_CENSUS.md in the task work dir): NOT met on either limb —
# round_trip_cost has multiple direct callers that bypass evaluate
# entirely (repo tests; the SF-BL-7432 S2, SF-BL-7785, SF-BL-3583
# and SF-BL-9232 evaluators), and evaluate itself performs NO
# book-type validation (its entry gates dereference book.market_id /
# book.fee_rate / book.yes_asks unguarded — the SF-BL-10963 crash),
# so there is no "already-validated book path" for the Gate D call
# to inherit. The fix therefore validates the book's type here, as
# the function's FIRST statement — before the fee check (which is
# itself a book dereference and cannot run first) and before the
# exit walk — raising a named ValueError, matching the SF-BL-9882 /
# SF-BL-9811 / SF-BL-10466 validations in this same function. The
# check is a strict isinstance against OrderBookSnapshot: a subclass
# instance IS a snapshot and computes bit-identically, while a
# duck-typed impostor (e.g. a SimpleNamespace carrying fee_rate /
# yes_asks / yes_bids attributes — which pre-fix silently COMPUTED
# via attribute access) is rejected, because it carries none of the
# snapshot's construction invariants (__post_init__ fee / flag /
# ladder-order validation, the live + tradeable-price filtering of
# the best-price properties, snapshot_ts provenance) that this
# function's pricing relies on. ORDERING: book first, then fee
# (SF-BL-9882), then q_exec (SF-BL-9811), then shares (SF-BL-10466)
# — signature order, and the only order in which the fee validator
# can run at all; no agreed sibling test pins a fee / q_exec /
# shares error for a non-snapshot book (pre-fix no such error could
# be raised — the call crashed), so the new precedence costs
# nothing, and for every genuine OrderBookSnapshot all four
# validations run in their previous relative order with every
# result bit-identical. The evaluate path is unaffected: it only
# ever passes its own book argument through, and a genuine snapshot
# passes this check untouched.

def _is_order_book_snapshot(book: object) -> bool:
    """True iff book is an OrderBookSnapshot (SF-BL-11004).

    isinstance, so a subclass instance qualifies; a duck-typed
    impostor carrying the same attribute names does not — it was
    never construction-validated as a snapshot."""
    return isinstance(book, OrderBookSnapshot)


def _validate_round_trip_book(book: object) -> None:
    """Raise ValueError unless book is an OrderBookSnapshot."""
    if not _is_order_book_snapshot(book):
        raise ValueError(
            f"book {book!r} is not an OrderBookSnapshot "
            f"(type {type(book).__name__}) (SF-BL-11004): "
            "round_trip_cost prices the exit walk off a constructed "
            "OrderBookSnapshot — pre-fix book=None and book={} (the "
            "JSON-loaded shape) raised a bare AttributeError on "
            "'fee_rate' at the SF-BL-9882 re-validation line, before "
            "any fee check or exit pricing; refusing to price an "
            "exit walk off a value that is not a book")


def round_trip_cost(
    book: OrderBookSnapshot,
    side: TradeAction,
    shares: float,
    q_exec: float,
) -> Tuple[float, float, float]:
    """(rt_cost, q_exit_net, fill_fraction) for an immediate liquidation.

    rt_cost = q_exec - q_exit_net, with q_exit_net the size-walked exit
    bid (see size_walked_exit_bid) charged the book's fee_rate and
    scaled by the exit fill fraction: shares the bid ladder cannot
    absorb are valued at 0 (SF-BL-9844), so q_exit_net is the net
    proceeds of an immediate liquidation per share acquired, not per
    share filled. At fill_fraction 1.0 this is exactly the pre-SF-BL-9844
    value.

    Raises ValueError if book is not an OrderBookSnapshot
    (SF-BL-11004), before the fee check or the exit walk — a None or
    dict book (the JSON-loaded shape) must fail with a named error,
    not a bare AttributeError on fee_rate. Raises ValueError if
    book.fee_rate is outside the pre-registered
    bound [0, 0.10] (SF-BL-9882, reusing SF-BL-9363's validator) — a
    book mutated or unpickled after construction must never be priced
    at an unvalidated exit fee. Raises ValueError if q_exec is outside
    the pre-registered domain (0, 1] or non-finite (SF-BL-9811),
    before the exit walk — Gate D must never be passable by a NaN
    cost manufactured from an impossible entry price. Raises
    ValueError if shares is non-finite or <= 0 (SF-BL-10466), before
    the exit walk — a position that does not exist has no round-trip
    cost, and its (0.0, 0.0) no-fill walk must never be fabricated
    into a full-price rt_cost = q_exec.
    """
    # SF-BL-11004: validate the book's TYPE here, FIRST — before the
    # fee check below, which dereferences book.fee_rate and is itself
    # the pre-fix crash site for a non-snapshot book. See the
    # SF-BL-11004 block comment above.
    _validate_round_trip_book(book)
    # SF-BL-9882: re-validate here, not only in OrderBookSnapshot
    # .__post_init__ / evaluate Gate 0 — see the block comment above.
    _validate_fee_rate(book.fee_rate)
    # SF-BL-9811: validate q_exec here, before the exit walk — see the
    # SF-BL-9811 block comment above. Fee first (signature order: the
    # book precedes q_exec); both raises precede any pricing.
    _validate_round_trip_q_exec(q_exec)
    # SF-BL-10466: validate shares here, before the exit walk — see
    # the SF-BL-10466 block comment above. After fee/q_exec so their
    # pinned no-op-call precedence is preserved; still before pricing.
    _validate_round_trip_shares(shares)
    bid_vwap, fill_fraction = size_walked_exit_bid(book, side, shares)
    # SF-BL-9844: unfilled remainder valued at 0 — see block above.
    q_exit = bid_vwap * fill_fraction * (1.0 - book.fee_rate)
    return q_exec - q_exit, q_exit, fill_fraction


def _dispersion_is_finite(dispersion: object) -> bool:
    """True iff dispersion is a finite real number (SF-BL-9395).

    Gate 2 compares dispersion with ``>``; NaN never satisfies ``>`` and
    -inf never exceeds any finite threshold, so a bare comparison lets
    non-finite dispersion silently bypass the supervisor gate and reach
    sizing. A bool is not a dispersion, and non-numeric values cannot be
    ordered against the threshold at all, so both count as invalid.
    """
    if isinstance(dispersion, bool) or not isinstance(dispersion, (int, float)):
        return False
    return math.isfinite(dispersion)


# --- SF-BL-9714: a negative finite dispersion is not perfect agreement --
# SF-BL-9395's Gate 0.75 checked only finiteness, and Gate 2 is a bare
# ``>`` against max_inter_agent_dispersion, so a NEGATIVE finite
# dispersion passed both gates as maximal agreement: verified live
# pre-fix (repro.py in the task work dir) on the standard fixture
# (p=0.80, ask 0.50 / bid 0.48, fee 0.01, equity $100,000),
# dispersion=-0.5 and dispersion=-100 both sized BUY_YES at the full
# $15,000 cluster cap, byte-identically to dispersion=0.10. A
# dispersion is a pstdev (aggregate_fleet_forecasts returns
# statistics.pstdev, or 0.0 for a single member), so a negative value
# is mathematically impossible from an honest producer — it is a
# corrupt caller-supplied supervisor input, not low disagreement.
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT
# met — aggregate is only the benchmark's producer; evaluate takes a
# plain float with no provenance, direct in-repo callers (the test
# suites) and out-of-tree fleet scripts (wave2 r63 scan) pass bare
# literals, and no test asserts a non-negativity invariant on the
# aggregate return — so the value itself is validated here, at Gate
# 0.75, exactly as for the non-finite case. Finite dispersion >= 0
# (including -0.0, which is not < 0) passes untouched and stays
# byte-identical; the standalone dispersion_taper_factor helper
# (SF-BL-9639) is unchanged — via evaluate its negative-dispersion
# clip is now unreachable, because Gate 0.75 SKIPs first in both the
# binary and taper arms.

def _dispersion_in_domain(dispersion: object) -> bool:
    """True iff dispersion is in its valid domain (SF-BL-9714): a finite
    real number (SF-BL-9395, bool / non-numeric excluded) that is >= 0.
    -0.0 qualifies: it compares equal to 0.0 and is not < 0."""
    return _dispersion_is_finite(dispersion) and dispersion >= 0  # type: ignore[operator]


# --- SF-BL-3659: Gate BS — budget-sensitivity veto -----------------------
# evaluate_and_size_opportunity historically gated once at the top of
# book (q_top) and then swept a SINGLE target_budget (Step 5): the
# marginal-edge stop in sweep_order_book trims every rung whose price
# breaks the hurdle, so the admitted q_exec is the VWAP of a curated
# fill at exactly one size. But q_exec is a budget-dependent curve —
# nstar.vwap_for_dollars walks a book the same way and the live probe
# (superforecaster_out/nstar/live_books.json) shows $100 filling at the
# best ask while $1,000 walks on thin books — and no code path ever
# checked whether the edge survives a change of size. A candidate whose
# book can absorb its Kelly budget only because the sweep stopped early
# has a size-fragile edge: deploy half or double the budget as a full
# walk and the price walks through the belief.
#
# Gate BS therefore RE-SWEEPS every candidate that passed the 1x gated
# sweep at 0.5x and 2x its Kelly target_budget and SKIPs it when, at
# either re-swept budget:
#   (a) the post-cost edge SIGN FLIPS — edge_s = p_side - q_exec_s <= 0
#       against the admitted 1x edge (which is >= min_net_edge by
#       construction of the gated sweep); or
#   (b) D_KL PER DOLLAR AT LEAST HALVES — kl_s <= 0.5 * kl_1x, where
#       kl_B = D_KL(p_side || q_exec_B) is, per Meister (2024), the
#       asymptotic log-wealth growth rate per unit stake (per dollar
#       deployed); a re-swept budget whose execution price has crossed
#       the belief (edge_s <= 0) carries kl_s = 0.0 by definition —
#       there is no positive information edge in a losing price, and
#       the raw divergence would perversely RISE again as q walks past
#       p; or
#   (c) the re-sweep cannot fill at all (no live depth at that budget
#       on the chosen side).
# The re-sweeps are FULL WALKS (min_net_edge=-1.0, the same override
# the naive baseline uses to disable the marginal stop), NOT gated
# sweeps: a gated re-sweep censors exactly the walk being measured —
# its q_exec can never flip sign, because the sweep breaks before the
# first bad rung — so it would make the veto decorative by
# construction. The reference (1x) values are the candidate's ACTUAL
# admitted execution: q_exec / edge / D_KL from the gated Step-5 sweep.
# Placement: after the depth-exhausted check (a candidate that filled
# nothing at 1x keeps that reason) and before Gate D and the cluster
# write — a vetoed candidate is a SKIP with EC-09 zero sizes and the
# cluster dict untouched, exactly like every other gate.
# config.budget_sensitivity_veto_enabled=False disables the gate
# (A/B use, mirroring Gate D's None).
#
# Materiality (census in the task work dir, CENSUS.md): of 1,771
# forward-S2 candidates passing the pre-veto gate at 1x, 904 (51.0%)
# trigger Gate BS — 539 flip sign at 0.5x, 644 flip at 2x, 107 / 260
# halve D_KL at 0.5x / 2x (overlapping counts) — and even under the
# stricter between-budgets reading (edge / D_KL at 0.5x compared
# directly with 2x, no 1x reference) 399 (22.5%) trigger. Both
# readings are far above the prereg's 10%
# materiality threshold and the 2% kill threshold, so the veto is
# load-bearing, NOT decorative, and the kill does NOT trigger.

# --- SF-BL-8892: NO-side fee is charged on the mirrored YES leg ---------
# evaluate_and_size_opportunity's top-of-book screen priced the NO side
# as no_top_ask_eff = best_no_ask * (1 + fee_rate), with best_no_ask the
# complement 1 - best_yes_bid: the fee was charged on the COMPLEMENT
# price. Buying NO is selling YES at the bid, so the leg actually traded
# is the YES leg at best_yes_bid — its proceeds net of fee are
# bid * (1 - fee_rate), and the mirror-consistent NO effective ask is
# 1 - bid * (1 - fee_rate). The old formula's error is
# fee_rate * (1 - 2 * bid): it UNDERCHARGES NO by 0.0030 at bid 0.65 and
# OVERCHARGES by 0.0040 at bid 0.30 (fee 1%, the prereg points), flips
# sign at bid 0.50, and reaches 0.0135 at bid 0.95 / fee 0.015 — so
# NO-side q_top, Gate 3's p/q ratio and kelly_raw were systematically
# mispriced on every NO candidate. The YES side is untouched: buying
# YES trades the YES leg at the ask, so ask * (1 + fee_rate) is already
# the fee on the traded leg. The correction uses best_yes_bid directly
# (not the round(..., 6) complement property), so the corrected price
# matches 1 - bid * (1 - fee) exactly, not merely to 1e-6. Scope: this
# is the SCREEN price (q_top and everything derived from it in
# evaluate). sweep_order_book's per-rung fee convention is a separate
# surface — its BUY_NO ground truth (unlock book 924.15 / 2400 /
# 0.3850625) and the benchmark N_star are pinned by prior tasks, and
# the neighbouring sweep-fee question (fee-adjusted rung prices above
# 1.0) is SF-BL-8891's prereg, not this one. Forward-books replay
# (replay_forward.py in the task work dir): of 452 borderline NO
# candidates (pre-fix edge within 0.005 of the 0.05 hurdle) on the 705
# two-sided archived S2 books, 218 (48.2%) change screen verdict under
# the correction at fee 0.01 — far above the 5% done threshold; the
# kill (flips < 1% AND deltas < 0.0005 everywhere) does not trigger:
# max |delta| is 0.00998 and 698 of 705 books carry |delta| >= 0.0005.

def _mirror_no_effective_ask(best_yes_bid: float, fee_rate: float) -> float:
    """Mirror-consistent NO effective (fee-adjusted) ask (SF-BL-8892):
    1 - best_yes_bid * (1 - fee_rate). Buying NO is selling YES at the
    bid; the fee is charged on the YES leg's proceeds, not on the
    complement price."""
    return 1.0 - best_yes_bid * (1.0 - fee_rate)


#: Budget scales Gate BS re-sweeps, relative to the Kelly target budget.
BUDGET_SENSITIVITY_SCALES = (0.5, 2.0)


def budget_sensitivity_check(
    book: OrderBookSnapshot,
    side: TradeAction,
    belief_prob_side: float,
    target_budget: float,
    q_exec_ref: float,
) -> dict:
    """Full-walk budget-sensitivity check for one admitted candidate.

    Re-sweeps `book` on `side` at each scale in
    BUDGET_SENSITIVITY_SCALES x `target_budget` with the marginal-edge
    stop disabled (min_net_edge=-1.0), and compares each budget's
    post-cost edge and per-dollar D_KL against the admitted 1x
    execution at `q_exec_ref`. Returns a dict with the reference
    values, a per-scale breakdown (invested / shares / q_exec / edge /
    kl), the list of trigger strings (empty when the candidate is
    size-stable), and ``veto`` = bool(triggers). Pure: no cluster or
    book state is touched.
    """
    edge_ref = belief_prob_side - q_exec_ref
    kl_ref = kl_divergence_binary(belief_prob_side, q_exec_ref)
    budgets: Dict[float, dict] = {}
    triggers: List[str] = []
    for scale in BUDGET_SENSITIVITY_SCALES:
        invested_s, shares_s, q_s = sweep_order_book(
            book=book,
            side=side,
            target_budget=target_budget * scale,
            belief_prob_side=belief_prob_side,
            min_net_edge=-1.0,
        )
        if shares_s <= 0.0 or invested_s <= 0.0:
            budgets[scale] = {
                "invested": invested_s, "shares": shares_s,
                "q_exec": 0.0, "edge": 0.0, "kl": 0.0,
            }
            triggers.append(f"no_fill_{scale}x")
            continue
        edge_s = belief_prob_side - q_s
        # No positive information edge survives a price at/above the
        # belief: kl_s is 0.0 there, not the raw (rising) divergence.
        kl_s = kl_divergence_binary(belief_prob_side, q_s) if edge_s > 0.0 else 0.0
        budgets[scale] = {
            "invested": invested_s, "shares": shares_s,
            "q_exec": q_s, "edge": edge_s, "kl": kl_s,
        }
        if edge_ref > 0.0 and edge_s <= 0.0:
            triggers.append(
                f"edge_flip_{scale}x (edge_1x={edge_ref:.3f}, "
                f"edge_{scale}x={edge_s:.3f})"
            )
        elif kl_ref > 0.0 and kl_s <= 0.5 * kl_ref:
            triggers.append(
                f"dkl_halved_{scale}x (dkl_1x={kl_ref:.4f}, "
                f"dkl_{scale}x={kl_s:.4f})"
            )
    return {
        "veto": bool(triggers),
        "triggers": triggers,
        "edge_ref": edge_ref,
        "kl_ref": kl_ref,
        "q_exec_ref": q_exec_ref,
        "budgets": budgets,
    }


# --- SF-BL-9396: caller-supplied cluster exposure breaches the cap -----
# Step 4 of evaluate_and_size_opportunity read the trade cluster's used
# fraction straight from the caller's mutable cluster_exposure_used dict
# and computed remaining room as cap - used with NO validation, so a
# negative used INFLATED the room past the cap: used=-0.20 returned
# room 0.35 and the prereg fixture (p=0.80 on a 0.50 ask) allocated
# 0.2086 of equity against the 0.15 max_cluster_allocation (verified
# live pre-fix, repro.py in the task work dir); used=-inf breached
# identically, a non-numeric used raised a bare TypeError from the
# subtraction, and the ledger write (used + actual_fraction) propagated
# the corrupt base back into the caller's dict (used=-1.0 left the dict
# at -0.7914 after a full-size trade). NaN used SKIPped only
# incidentally (max(0.0, cap - NaN) is 0.0 by Python's max semantics),
# not by any validated rule. Kill-criterion audit (CALLER_CENSUS.md in
# the task work dir): NOT met — cluster_exposure_used is a plain
# Dict[str, float] argument with no producer invariant: the benchmark
# builds its dict locally from evaluate's own writes, tests and fleet
# callers pass bare literals / externally-sourced dicts, and the dict
# is mutable between calls, so no upstream guarantee bounds the value.
# The value is therefore sanitised here, at the consumption point:
# a finite real (bool excluded) is CLAMPED to [0, cap] — a negative
# used sizes exactly as if it were 0, a >cap used exactly as if it
# were cap (room 0, SKIP CLUSTER_CAP_REACHED) — and a non-finite
# (NaN, +-inf), bool or non-numeric used FAILS CLOSED, sized as if it
# were cap, because no in-range equivalent exists for it and inventing
# headroom for a corrupt ledger entry is exactly the breach above.
# In-range finite values pass through unchanged, so valid behaviour is
# byte-identical. SKIP paths still never write the dict; an executed
# trade writes sanitised_used + actual_fraction, which lies in
# [0, cap] by construction (actual <= capped <= cap - sanitised), so
# the stored ledger is back in range after the trade and the SF-BL-8890
# settlement release (which subtracts exactly allocated_fraction)
# stays exact against the stored value.

def _sanitized_cluster_exposure_used(raw: object, cap: float) -> float:
    """The cluster exposure Step 4 may size against (SF-BL-9396).

    Finite reals clamp to [0, cap]; non-finite / bool / non-numeric
    values fail closed at `cap` (no remaining room).
    """
    if not _is_finite_number(raw):
        return float(cap)
    return min(max(float(raw), 0.0), float(cap))  # type: ignore[arg-type]


# --- SF-BL-10762: cluster cap mixes equity vintages ---------------------
# Pre-fix the cluster ledger stored FRACTIONS of a moving base:
# evaluate wrote cluster_exposure_used[cluster] = used +
# invested / portfolio_equity with the equity of ITS OWN trade time,
# and the next candidate computed room as cap - stored_sum while
# sizing against CURRENT equity. Callers update equity between
# candidates sharing one dict (run_benchmark_simulation:
# `disciplined_equity += net_pnl_bid_marked` per scenario; the paper /
# forward path marks equity while positions stay open), so the stored
# sum was a fraction of no single equity. Verified live pre-fix
# (repro.py in the task work dir; deep book ask 0.50 / bid 0.48,
# fee 0.01, p=0.65, position left OPEN): trade 1 at equity $10,000
# invested $1,025.25 (stored fraction 0.102525); after a 50% drawdown
# to $5,000 the second same-cluster trade invested $237.37 more —
# $1,262.63 total against a true 15%-of-current cap of $750.00
# (1.68x breach; the stored fraction understated live exposure
# twofold) — and after a +100% gain to $20,000 it invested $949.49
# against a true room of $1,974.75 (headroom stranded twofold).
# SF-BL-8890's exact-addend release bounded the drift to concurrently
# open positions and explicitly left this case open ("fraction-basis
# drift among concurrently open positions ... would be a breaking
# API change" — BASIS_AUDIT.md there); nstar.py's Account keys
# cluster_stakes in DOLLARS for exactly this reason.
# Kill-criterion audit (KILL_AUDIT.md in the task work dir): NOT met —
# equity paths DO move >10% between same-cluster trades: the agreed
# SF-BL-8890 suite itself evaluates two same-cluster opens at $10,000
# then $20,000 sharing one dict (+100%), and in the benchmark a single
# intervening executed trade moves disciplined_equity by ~15% (a
# full-cap win at ask 0.50 pays ~2x stake), so the >10% band is
# crossed in one step whenever the loop executes.
# THE FIX (the prereg's re-base variant — the dict's public fraction
# API is unchanged, so seeded-dict callers and the SF-BL-9396/8890
# surfaces stay byte-identical at constant equity): the engine keeps
# a DOLLAR ledger alongside the caller's dict, keyed by the dict's
# identity. Every evaluate Step-4 read re-bases the stored exposure
# to current equity (fraction_used = dollars_at_risk /
# portfolio_equity) before computing room, and every executed write
# stores dollars_new / current_equity — the fraction AT THE CURRENT
# VINTAGE — while recording dollars_new as the truth. Settlement
# release (SF-BL-8890) subtracts the trade's exact invested dollars
# from the dollar ledger and re-derives the dict fraction at its
# last-written vintage. A dict value the engine did not write (a
# caller-seeded fraction, or an externally mutated entry — detected
# by snapshot mismatch) keeps the legacy SF-BL-9396 interpretation:
# a sanitised fraction of CURRENT equity, which is the only meaning
# a bare fraction can carry. The registry holds a strong reference
# to each ledger dict, so an id can never be silently reused by a
# different dict while its entry lives; the registry is pruned
# oldest-first past _CLUSTER_DOLLAR_LEDGER_MAX entries, and a pruned
# or never-registered dict simply falls back to the legacy reading.
# A caller that COPIES the dict between calls likewise falls back —
# the benchmark / forward pattern reuses one dict, as the prereg
# states. SUPERSESSION: SF-BL-8890's concurrent-positions test pins
# the dict at f1 + f2 (fractions of $10,000 and $20,000 summed);
# post-fix the dict holds the true current-equity fraction
# (dollars_total / latest equity), and that one assertion is
# superseded by this prereg — its release assertions (settle one
# leaves the other's exposure, settle both returns to 0.0) still
# hold, in dollar terms.

_CLUSTER_DOLLAR_LEDGER: Dict[int, Dict[str, object]] = {}
_CLUSTER_DOLLAR_LEDGER_MAX = 512


def _cluster_dollar_entry(
    ledger: Dict[str, float], cluster: str
) -> Optional[Dict[str, float]]:
    """The registered dollar entry for (ledger, cluster), or None.

    An entry is authoritative only while the ledger object is the
    exact dict the engine wrote to and the dict's current value for
    the cluster still matches the fraction snapshot the engine last
    wrote — any external write to that key voids the entry and the
    caller falls back to the legacy fraction interpretation.
    """
    entry = _CLUSTER_DOLLAR_LEDGER.get(id(ledger))
    if entry is None or entry.get("ref") is not ledger:
        return None
    clusters = entry.get("clusters")
    if not isinstance(clusters, dict):
        return None
    centry = clusters.get(cluster)
    if not isinstance(centry, dict):
        return None
    raw = ledger.get(cluster, 0.0)
    if not _is_finite_number(raw):
        return None
    if not math.isclose(float(raw), centry["fraction"],  # type: ignore[arg-type]
                        rel_tol=1e-9, abs_tol=1e-12):
        return None
    return centry


def _record_cluster_dollars(
    ledger: Dict[str, float], cluster: str, dollars: float,
    equity: float, fraction_written: float,
) -> None:
    """Record `dollars` at risk for (ledger, cluster) (SF-BL-10762),
    written at vintage `equity` as dict fraction `fraction_written`."""
    key = id(ledger)
    entry = _CLUSTER_DOLLAR_LEDGER.get(key)
    if entry is None or entry.get("ref") is not ledger:
        if len(_CLUSTER_DOLLAR_LEDGER) >= _CLUSTER_DOLLAR_LEDGER_MAX:
            _CLUSTER_DOLLAR_LEDGER.pop(
                next(iter(_CLUSTER_DOLLAR_LEDGER)), None)
        entry = {"ref": ledger, "clusters": {}}
        _CLUSTER_DOLLAR_LEDGER[key] = entry
    clusters = entry["clusters"]
    assert isinstance(clusters, dict)
    clusters[cluster] = {
        "dollars": float(dollars),
        "equity": float(equity),
        "fraction": float(fraction_written),
    }


def _rebased_cluster_exposure_used(
    ledger: Dict[str, float], cluster: str, cap: float,
    portfolio_equity: float,
) -> Tuple[float, Optional[float]]:
    """Step 4's cluster exposure, re-based to current equity (SF-BL-10762).

    Returns (used_fraction, dollars_used): used_fraction is the
    exposure as a fraction of CURRENT portfolio_equity, clamped to
    [0, cap] for the room computation exactly as SF-BL-9396 clamps;
    dollars_used is the true dollars at risk behind it, or None when
    portfolio_equity is not a valid equity (Step 4 is unreachable
    then — Gate 0.5 SKIPs first — so the legacy fraction write is
    kept as a defensive fallback). With a registered dollar entry
    the dollars are authoritative and the fraction is re-based;
    without one, the dict's raw value is a caller-seeded fraction of
    current equity and goes through the SF-BL-9396 sanitiser, so
    every seeded-dict behaviour is unchanged.
    """
    raw = ledger.get(cluster, 0.0)
    if not _is_valid_portfolio_equity(portfolio_equity):
        return _sanitized_cluster_exposure_used(raw, cap), None
    centry = _cluster_dollar_entry(ledger, cluster)
    if centry is not None:
        dollars = float(centry["dollars"])
        fraction = dollars / float(portfolio_equity)  # type: ignore[arg-type]
        return min(max(fraction, 0.0), float(cap)), dollars
    fraction = _sanitized_cluster_exposure_used(raw, cap)
    return fraction, fraction * float(portfolio_equity)  # type: ignore[arg-type]


# --- SF-BL-34952: renaming factor_cluster leaks the cluster release --
# release_cluster_exposure and its dollar helper resolved the ledger
# KEY from the trade's CURRENT factor_cluster, but that field is a
# plain mutable dataclass attribute (__setattr__ guards only action,
# is_settled and cluster_exposure_released) and evaluate charges the
# key the BOOK carried at entry. Verified live pre-fix (repro.py in
# the task work dir): evaluate at $100,000 charged cluster 'alpha'
# the full 0.15 cap; assigning trade.factor_cluster='beta' and
# settling released under 'beta' — the ledger became
# {'alpha': 0.15, 'beta': 0.0} and cluster_exposure_released was set,
# so the alpha cap room is never restored and the consumed flag makes
# any corrective re-release a no-op; the legacy-fraction path leaked
# identically (removed 0.0 against a seeded {'alpha': 0.15} ledger).
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT
# met — no in-repo path renames a genuine evaluated trade (the sole
# `.factor_cluster =` hit is on a SimpleNamespace fake in the
# SF-BL-10971 suite), but the field is unguarded public state with no
# invariant or test pinning it, records are rebuilt via
# TradeExecutionResult(**pre.__dict__) and mutated / unpickled out of
# tree, and both release entry points are public, so no tested
# invariant stands between a rename and this release (SF-BL-34868
# precedent). The entry key is therefore RECORDED on the record
# (entry_factor_cluster, snapshotted in __post_init__; evaluate
# passes it explicitly) and the release resolves its key through
# that snapshot at BOTH consumption points — the SF-BL-10762 dollar
# lookup/write below and the SF-BL-8890 legacy subtraction in
# release_cluster_exposure — so a rename by plain assignment,
# object.__setattr__, __dict__ write or unpickling cannot redirect
# the release, and no stray key is ever written. The visible
# factor_cluster field stays assignable (display / audit surfaces
# are untouched) and every unrenamed lifecycle is bit-identical:
# the snapshot equals factor_cluster unless the field was renamed.
# SF-BL-34868 owns the ledger DICT's identity; this task owns the
# KEY inside that dict. SF-BL-34995 / SF-BL-35172 own the released
# AMOUNT read from mutable fields; this task owns the released KEY.

def _entry_factor_cluster(trade: TradeExecutionResult) -> str:
    """The cluster key this trade's exposure was charged under at
    entry (SF-BL-34952).

    The entry_factor_cluster snapshot when one is recorded (a
    non-empty string); otherwise the trade's current factor_cluster
    (records that predate the snapshot — an old pickle whose
    __post_init__ never ran — keep their pre-fix resolution).
    """
    entry = getattr(trade, "entry_factor_cluster", None)
    if isinstance(entry, str) and entry:
        return entry
    return trade.factor_cluster


def _release_cluster_dollars(
    trade: TradeExecutionResult, ledger: Dict[str, float]
) -> Optional[float]:
    """Dollar-exact settlement release (SF-BL-10762 + SF-BL-8890).

    When (ledger, entry cluster) has a registered dollar
    entry, subtract the trade's exact invested_capital dollars —
    the dollars evaluate added — and re-derive the dict fraction at
    the vintage it was last written at (settlement receives no
    current equity; the next evaluate re-bases from dollars anyway).
    Returns the fraction removed, measured at that vintage, or None
    when no registered entry governs this ledger (the caller then
    applies SF-BL-8890's legacy fraction subtraction).

    SF-BL-34952: the cluster key is the ENTRY key
    (_entry_factor_cluster), not the trade's current factor_cluster —
    a post-entry rename cannot redirect the dollar release.
    """
    _cluster = _entry_factor_cluster(trade)
    centry = _cluster_dollar_entry(ledger, _cluster)
    if centry is None:
        return None
    stake = trade.invested_capital
    if not _is_finite_number(stake) or stake <= 0:  # type: ignore[operator]
        return None
    dollars_before = float(centry["dollars"])
    removed = min(dollars_before, float(stake))  # type: ignore[arg-type]
    remaining = dollars_before - removed
    if remaining < 1e-7:
        removed = dollars_before
        remaining = 0.0
    vintage = float(centry["equity"])
    new_fraction = remaining / vintage if vintage > 0 else 0.0
    ledger[_cluster] = new_fraction
    centry["dollars"] = remaining
    centry["fraction"] = new_fraction
    return removed / vintage if vintage > 0 else 0.0


# --- SF-BL-10963: the book argument itself is never type-validated -----
# evaluate_and_size_opportunity assumed `book` is a constructed
# OrderBookSnapshot, but the annotation is unenforced and every gate
# dereferences the object: Gate 0 reads book.fee_rate unconditionally,
# and every SKIP return copies book.market_id / title / domain /
# factor_cluster / est_days_to_resolution (Gate E0's config SKIP
# included), while _declared_mid_prob reads the ladders. Verified live
# pre-fix (repro.py in the task work dir; standard fixture p=0.80,
# ask 0.50 / bid 0.48, fee 0.01): book=None and book=dict — the shape a
# forward snapshot JSON file materialises as before any
# OrderBookSnapshot construction — both aborted the cycle with a bare
# AttributeError ('NoneType' / 'dict' object has no attribute
# 'fee_rate') before any gate could SKIP, as did str / int books.
# SF-BL-10826 fixtures the ladder fields INSIDE a constructed book and
# SF-BL-10907 the config argument; neither fixtures the book argument
# itself. Kill-criterion audit (CALLER_CENSUS.md in the task work dir):
# NOT met — the sole in-repo production caller is the benchmark passing
# literal constructions, no deserialiser materialises snapshots from
# JSON under any test, and no test pins "evaluate is only called with a
# constructed snapshot", so out-of-tree / JSON-fed callers reach this
# public function with no invariant in between.
# PRE-REGISTERED RESOLUTION (PREREG.md in the task work dir): the
# prereg's ValueError branch. A zero-size SKIP is a TradeExecutionResult
# whose identity fields are copied FROM the book; for book=None there
# is no honest value for any of them and a dict's domain is a plain
# string, so fabricating a placeholder SKIP record would write a
# corrupt trade record into the caller's list — the named-ValueError-at-
# entry branch is the honest one, matching this module's constructor
# validators. The check is isinstance(book, OrderBookSnapshot):
# subclasses pass; duck-typed lookalikes do not (the dataclass's
# __post_init__ invariants are part of the contract). It runs as the
# FIRST statement of evaluate, before Gate E0 — Gate E0's own SKIP
# construction dereferences book.*, so any later placement would still
# crash on a bad book paired with a bad config — which also means no
# gate runs and the cluster dict is untouched by construction. Pure
# validation: a constructed snapshot passes untouched, so every
# constructed-book fixture and the benchmark stay bit-identical. The
# value repr in the message is length-capped: a JSON book can carry a
# full ladder and its repr must not flood the caller's logs; the type
# name is always given in full.

def _validate_book_argument(book: object) -> None:
    """Raise ValueError unless book is an OrderBookSnapshot (SF-BL-10963)."""
    if not isinstance(book, OrderBookSnapshot):
        _repr = repr(book)
        if len(_repr) > 200:
            _repr = _repr[:200] + "...<truncated>"
        raise ValueError(
            f"INVALID_BOOK: book {_repr} (type "
            f"{type(book).__name__}) is not an OrderBookSnapshot "
            "(SF-BL-10963): evaluate_and_size_opportunity gates on a "
            "constructed OrderBookSnapshot — a None book or a "
            "JSON-loaded dict (the shape a forward snapshot "
            "materialises as) must be materialised into an "
            "OrderBookSnapshot before evaluation; no gate ran and no "
            "size was taken")


def evaluate_and_size_opportunity(
    book: OrderBookSnapshot,
    calibrated_prob_yes: float,
    inter_agent_dispersion: float,
    portfolio_equity: float,
    cluster_exposure_used: Dict[str, float],
    config: FleetRiskConfig = FleetRiskConfig(),
    panel_n_dropped: Optional[int] = None,
    panel_n_kept: Optional[int] = None,
) -> TradeExecutionResult:
    """
    Executes the complete decision gate for a single market opportunity:
      1. Checks CryptoBench on-chain verification gate and inter-agent disagreement gate.
      2. Determines optimal trade direction (BUY_YES, BUY_NO, or SKIP).
      3. Computes Fractional Kelly allocation & enforces the 15% correlated cluster cap.
      4. Sweeps the live CLOB ask ladder, dynamically stopping where slippage erodes net edge.
    """
    # Gate EB (SF-BL-10963): book-argument type validity, at evaluate
    # ENTRY — the first statement, before Gate E0, because Gate E0's
    # SKIP construction itself dereferences book.*. A book that is not
    # an OrderBookSnapshot (None, a JSON-loaded dict, any other type)
    # raises the named ValueError from _validate_book_argument instead
    # of crashing on book.fee_rate / book.market_id inside some gate;
    # no gate runs, no size is taken, the cluster dict is untouched.
    # A constructed snapshot passes untouched (bit-identical).
    _validate_book_argument(book)

    # Gate ET (SF-BL-11534): est_days_to_resolution TYPE validity, at
    # evaluate ENTRY — immediately after Gate EB, before Gate E0 and
    # every other gate, because every later gate's SKIP record copies
    # book.est_days_to_resolution verbatim and Gate 0.1's domain check
    # would lump a bool / non-numeric lock into its VALUE error. A
    # mutated/unpickled book (which bypasses __post_init__'s named
    # type raise) with a bool or non-numeric lock SKIPs here: zero
    # size (EC-09), cluster dict untouched, the offending value
    # carried verbatim on the record for audit, and the reason
    # carrying both the umbrella INVALID_EST_DAYS_TO_RESOLUTION token
    # (pinned by SF-BL-9364) and the named type class
    # INVALID_EST_DAYS_TYPE (SF-BL-11534). Numeric locks — valid,
    # NaN/inf and <= 0 alike — pass this gate untouched and keep
    # their existing gate order and reasons exactly (Gate 0.1 owns
    # the numeric domain).
    if not _est_days_is_numeric_type(book.est_days_to_resolution):
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=_declared_mid_prob(book),
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=(
                f"INVALID_EST_DAYS_TO_RESOLUTION "
                f"(INVALID_EST_DAYS_TYPE: "
                f"est_days_to_resolution={book.est_days_to_resolution!r} "
                f"(type {type(book.est_days_to_resolution).__name__}) "
                "is not a real number — a bool or non-numeric capital "
                "lockup is a type error, not a non-finite value "
                "(SF-BL-11534))"
            ),
        )

    # Gate ED (SF-BL-10795): book.domain type validity, at evaluate
    # ENTRY — after Gate EB/ET, before Gate E0 and every other gate,
    # because Gate 1 and the Step-4 domain multiplier both READ
    # book.domain by equality with MarketDomain.CRYPTO: a domain
    # that is not a genuine MarketDomain member (a mutated/unpickled
    # book bypasses __post_init__'s raise) must SKIP here rather
    # than let equality decide — 'CRYPTO' / 'Crypto' / 'crypto '
    # disarm Gate 1 and the dampener pre-fix and trade an unverified
    # book, and 'crypto' arms them only by str-Enum accident. SKIP
    # (zero size, per EC-09, cluster dict untouched) with the named
    # INVALID_DOMAIN reason, distinct from
    # FAILED_CRYPTOBENCH_ONCHAIN_GATE, which is reserved for a
    # genuine CRYPTO member with a strict bool False flag. The
    # check is unconditional (not scoped to CRYPTO or to the gate
    # switch): a non-member domain has no honest identity on any
    # path. The offending value is carried verbatim on the SKIP
    # record's domain for audit (TradeExecutionResult.domain is
    # deliberately unvalidated, as for est_days). Genuine members
    # pass untouched, so valid-book behaviour is byte-identical.
    if not _is_valid_market_domain(book.domain):
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=_declared_mid_prob(book),
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=(
                f"INVALID_DOMAIN (domain={book.domain!r} "
                f"(type {type(book.domain).__name__}) is not a "
                "MarketDomain member; Gate 1 compares domain == "
                "MarketDomain.CRYPTO (value 'crypto'), so a "
                "plain-string domain never arms the CryptoBench "
                "gate (SF-BL-10795))"
            ),
        )

    # Gate E0 (SF-BL-10845): config validity, at evaluate ENTRY — before
    # every other gate, because Gate 0.9 and Gate 1 both READ
    # config.require_on_chain_for_crypto: a non-bool switch must be
    # rejected before its truthiness can arm or disarm the CryptoBench
    # gate ('' / 0 silently disabled Gate 1 pre-fix; 'False' armed it).
    # Construction already rejects a non-bool switch
    # (FleetRiskConfig.__post_init__), but a config mutated/unpickled
    # after construction bypasses that; the entry check therefore
    # re-validates and SKIPs (zero size, per EC-09, cluster dict
    # untouched) with an explicit invalid-config reason, distinct from
    # FAILED_CRYPTOBENCH_ONCHAIN_GATE, which is reserved for a strict
    # bool True switch meeting a strict bool False book flag. The check
    # is unconditional (not scoped to CRYPTO): the config object itself
    # is malformed, whatever the current book is, and a strict bool
    # True/False passes untouched, so valid-config behaviour is
    # byte-identical. A missing attribute / config=None reads as None
    # here and is likewise rejected as invalid config.
    _require_flag = getattr(config, "require_on_chain_for_crypto", None)
    if not _is_strict_bool(_require_flag):
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=_declared_mid_prob(book),
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=(
                f"INVALID_CONFIG (invalid config: "
                f"require_on_chain_for_crypto={_require_flag!r} is not a "
                f"bool (type {type(_require_flag).__name__}); "
                "CryptoBench gate switch requires a strict bool "
                "(SF-BL-10845))"
            ),
        )

    # Gate 0.0 (SF-BL-9470): calibrated posterior validity, FIRST of the
    # book/input gates (after the SF-BL-10845 config-entry gate above) —
    # before any other gate, the edge screen, Gate 3 or Kelly sizing can see it.
    # A posterior outside [0, 1], or non-finite / non-numeric, is a data
    # error, not a signal: SKIP, zero size (EC-09), cluster dict untouched,
    # with an explicit INVALID_POSTERIOR reason. In-range posteriors (including the
    # endpoints 0.0 and 1.0, and every aggregate output in [0.02, 0.98])
    # pass this gate untouched; the endpoints are then conformance-
    # clipped to the [0.02, 0.98] spec band at Step 0.05 (SF-BL-9329).
    #
    # Gate 0.0a (SF-BL-9440): the NON-FINITE / non-numeric case is split
    # out FIRST, before the finite-range check below and long before the
    # edge computation (edge_yes/edge_no are computed near the end of
    # this function; pre-fix a NaN posterior reached them and fell into
    # the INSUFFICIENT_NET_EDGE else branch as 'max_edge=nan'). Its
    # reason keeps the SF-BL-9470 primary token INVALID_POSTERIOR —
    # pinned by that task's agreed tests — and names this task's class
    # INVALID_FORECAST inside it, so skip-reason telemetry can count
    # invalid forecasts distinctly. Zero size (EC-09), cluster untouched,
    # exactly as for Gate 0.0.
    if _posterior_not_finite(calibrated_prob_yes):
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=_declared_mid_prob(book),
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=(
                f"INVALID_POSTERIOR (INVALID_FORECAST: "
                f"calibrated_prob_yes={calibrated_prob_yes!r} is not a "
                "finite real number — a non-finite posterior is an "
                "invalid forecast, not a no-edge market (SF-BL-9440))"
            ),
        )

    if not _posterior_in_bounds(calibrated_prob_yes):
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=_declared_mid_prob(book),
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=(
                f"INVALID_POSTERIOR (calibrated_prob_yes={calibrated_prob_yes!r} "
                f"is not a finite probability in [{POSTERIOR_MIN}, {POSTERIOR_MAX}])"
            ),
        )

    # Step 0.05 (SF-BL-9329): spec-band conformance clip. The posterior
    # is valid ([0, 1], Gate 0.0 above) but the NORTH_STAR spec band is
    # [0.02, 0.98]; clip it here — before the fee gate, the edge screen,
    # Gate 3 and all Kelly sizing — so this surface sizes exactly as
    # cli.py cmd_edge and backtest_nstar.py do on theirs. Rebinding the
    # parameter makes every later SKIP / trade record carry the clipped
    # value in calibrated_prob_yes, so an out-of-band call's result is
    # identical in every field to its clipped-equivalent call. In-band
    # values (including all aggregate_fleet_forecasts output) are
    # untouched. See the SF-BL-9329 block at POSTERIOR_CLIP_MIN/MAX.
    calibrated_prob_yes = clip_probability(
        calibrated_prob_yes, POSTERIOR_CLIP_MIN, POSTERIOR_CLIP_MAX)

    # Gate 0 (SF-BL-9363): fee_rate validity. Construction already rejects an
    # out-of-bound fee (__post_init__), but a book mutated/unpickled after
    # construction bypasses that; the gate therefore re-checks and SKIPs
    # (zero size, per EC-09) rather than ever pricing at the invalid fee.
    if not _fee_rate_in_bounds(book.fee_rate):
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=_declared_mid_prob(book),
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=(
                f"INVALID_FEE_RATE (fee_rate={book.fee_rate!r} outside "
                f"[{FEE_RATE_MIN}, {FEE_RATE_MAX}])"
            ),
        )

    # Gate 0.1 (SF-BL-9364): est_days_to_resolution validity. Construction
    # already rejects a non-positive / non-finite lockup (__post_init__),
    # but a book mutated/unpickled after construction bypasses that; the
    # value is copied into every trade record this function returns and
    # pooled into avg_lockup at scoring, where a negative lock
    # manufactures velocity and a zero/negative single book is masked by
    # the max(1.0, avg_lockup) floor. SKIP here (zero size, per EC-09,
    # cluster dict untouched) with an explicit invalid-lockup reason.
    # Finite positive values pass untouched, so valid-book behaviour is
    # byte-identical.
    if not _est_days_in_domain(book.est_days_to_resolution):
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=_declared_mid_prob(book),
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=(
                f"INVALID_EST_DAYS_TO_RESOLUTION "
                f"(est_days_to_resolution={book.est_days_to_resolution!r} "
                "is not a finite number > 0 (SF-BL-9364))"
            ),
        )

    # Gate 0.45 (SF-BL-9051): ladder sortedness. Construction already
    # rejects an unsorted ladder (__post_init__), but a book mutated or
    # unpickled after construction bypasses that; the best-price
    # properties, the edge screen and the sweep all trust presentation
    # order, so a mis-ordered ladder must SKIP here — before the edge
    # screen, Gate 3 or any sizing can see it — with zero size (EC-09),
    # cluster dict untouched, and market_mid_prob 0.0: a mid averaged
    # from a mis-ordered ladder's first rungs is not observable (the
    # SF-BL-3549 declared-rule convention, as for Gate 0.7).
    _order_violation = _ladder_order_violation(book.yes_asks, book.yes_bids)
    if _order_violation is not None:
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=0.0,
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=(
                f"UNSORTED_LADDER ({_order_violation} not sorted: "
                "yes_asks must be ascending and yes_bids descending "
                "by price (SF-BL-9051))"
            ),
        )

    # Gate 0.5 (SF-BL-9398): empty book. A book whose only levels are
    # zero/negative-size (or which has no levels at all) has no fillable
    # price on either side and must SKIP as empty — never screen an edge
    # against a sentinel or a phantom level. Zero sizes, per EC-09.
    if not _live_levels(book.yes_asks) and not _live_levels(book.yes_bids):
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=_declared_mid_prob(book),
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason="EMPTY_BOOK (no live levels: all zero/negative size)",
        )

    # Gate 0.75 (SF-BL-9395): inter-agent dispersion finiteness. Dispersion
    # is caller-supplied (aggregate_fleet_forecasts or an external caller);
    # Gate 2 below is a bare ``>`` comparison, which NaN never satisfies
    # and -inf never exceeds — pre-fix, dispersion=nan and dispersion=-inf
    # on the prereg fixture (p=0.80, ask 0.50/bid 0.48, fee 0.01) traded
    # BUY_YES at the full $15,000 cluster cap while dispersion=0.50 SKIPped
    # as UNRESOLVED_AGENT_DISPERSION, and +inf SKIPped only incidentally
    # with that unresolved-dispersion reason rather than an invalid-input
    # reason. Non-finite dispersion is a data error, not low disagreement,
    # so it SKIPs here (zero size, per EC-09) with an explicit invalid-input
    # reason, before any other gate or the sizing step can see it. Finite
    # dispersion passes this check untouched, so its behaviour — Gate 2
    # included — is byte-identical.
    #
    # Gate 0.75 extension (SF-BL-9714): a negative FINITE dispersion is
    # rejected at this same gate, with this same INVALID reason token.
    # It is a pstdev and cannot be < 0, so a negative value is corrupt
    # caller-supplied supervisor input (see the SF-BL-9714 block at
    # _dispersion_in_domain); pre-fix it passed the bare ``>`` at Gate 2
    # as perfect agreement and sized at the full cluster cap. The
    # non-finite reason string below is byte-identical to SF-BL-9395;
    # the negative branch names the sign violation explicitly. Finite
    # dispersion >= 0 passes untouched, so its behaviour — Gate 2
    # included — remains byte-identical.
    if not _dispersion_in_domain(inter_agent_dispersion):
        if not _dispersion_is_finite(inter_agent_dispersion):
            _disp_skip_reason = (
                f"INVALID_INTER_AGENT_DISPERSION "
                f"(inter_agent_dispersion={inter_agent_dispersion!r} is not finite)"
            )
        else:
            _disp_skip_reason = (
                f"INVALID_INTER_AGENT_DISPERSION "
                f"(inter_agent_dispersion={inter_agent_dispersion!r} is negative: "
                "dispersion is a pstdev and cannot be < 0)"
            )
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=_declared_mid_prob(book),
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=_disp_skip_reason,
        )

    # Gate 0.5 (SF-BL-9438): portfolio_equity validity, before any sizing.
    # A non-finite (or non-positive) equity must fail closed here: left
    # through, NaN equity makes target_budget and the whole sweep NaN and
    # the invested<=0 exhaustion check is False for NaN, so the trade
    # executed and poisoned cluster_exposure_used with NaN; +inf equity
    # swept the full ladder at allocated_fraction 0.0. SKIP, zero size
    # (EC-09), cluster dict untouched.
    if not _is_valid_portfolio_equity(portfolio_equity):
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=_declared_mid_prob(book),
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=(
                f"INVALID_EQUITY (portfolio_equity={portfolio_equity!r} "
                "is not a positive finite number)"
            ),
        )

    # Gate 0.6 (SF-BL-9542, preserving SF-BL-9438): a live rung priced NaN
    # on either ladder is book corruption, not a price. The sweep now
    # skips NaN rungs (SF-BL-9542 superseded the SF-BL-9473 carve-out
    # there), but evaluate is the gated decision path and SF-BL-9438
    # pinned its outcome for a NaN-poisoned book — SKIP INVALID_EQUITY,
    # zero size, cluster untouched, exactly as its post-sweep guard
    # produced — so the book fails closed here, before the edge screen
    # or sizing can see it. Ordered out-of-domain prices (negative,
    # zero, >1, +-inf) do NOT trip this gate: they are skipped by the
    # best-price properties and the sweep ladders, so a mixed book
    # trades identically to the clean book with those rungs removed.
    # Zero-size NaN rungs are not live (SF-BL-9398) and do not trip it.
    if any(
        _is_nan_price(lvl.price)
        for lvl in _live_levels(book.yes_asks) + _live_levels(book.yes_bids)
    ):
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=_declared_mid_prob(book),
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=(
                "INVALID_EQUITY (non-finite rung price: a live order-book "
                "rung is priced NaN (SF-BL-9542/9438))"
            ),
        )

    # Gate 0.9 (SF-BL-10793): on_chain_verified type validity, immediately
    # before Gate 1. Construction already rejects a non-bool flag
    # (__post_init__), but a book mutated/unpickled after construction
    # bypasses that; Gate 1's bare `not` would then let truthiness decide
    # — a truthy string ('False', 'no', '0') traded pre-fix, and a falsy
    # non-bool (0, '', None) SKIPped with the genuine-unverified reason.
    # A non-bool verification flag on a gated CRYPTO book is a data
    # error, not a verification result: SKIP here (zero size, per
    # EC-09, cluster dict untouched) with an explicit invalid-verification
    # reason, distinct from FAILED_CRYPTOBENCH_ONCHAIN_GATE, which is
    # reserved for a strict bool False. Scoped exactly as Gate 1 is
    # (CRYPTO + require_on_chain_for_crypto — itself a strict bool here
    # since SF-BL-10845's Gate E0, so this condition cannot be silently
    # disabled by a falsy non-bool switch): where the gate does not
    # apply the flag is never read, and strict bool True/False pass this
    # check untouched, so their behaviour is byte-identical. Ordered
    # after the earlier input-validity gates (posterior, fee, dispersion,
    # equity, NaN rung), which keep their precedence.
    if (
        book.domain == MarketDomain.CRYPTO
        and config.require_on_chain_for_crypto
        and not _is_strict_bool(book.on_chain_verified)
    ):
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=_declared_mid_prob(book),
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=(
                f"INVALID_ON_CHAIN_VERIFIED (invalid verification: "
                f"on_chain_verified={book.on_chain_verified!r} is not a "
                f"bool (type {type(book.on_chain_verified).__name__}); "
                "CryptoBench gate requires a strict bool (SF-BL-10793))"
            ),
        )

    # Gate 0.7 (SF-BL-10760): one-sided book. Gate 0.5 SKIPs only when
    # BOTH ladders lack live levels, so pre-fix a book with a live ladder
    # on ONE side only passed every gate: an asks-only book (ask 0.50,
    # no bids) screened edge_yes = p - ask_eff — the missing bid ladder
    # never enters edge_yes — and with Gate D disabled
    # (max_round_trip_cost=None) EXECUTED BUY_YES $15,000 (equity
    # $100,000, p=0.80, verified live) storing liquidation_bid_price =
    # 0.01, the empty-bid SENTINEL, so settle_or_mark_trade marked the
    # position at -$14,702.97 on the $15,000 stake; the bids-only mirror
    # executed BUY_NO identically (liq bid = 1 - 0.99 sentinel ask).
    # With Gate D enabled the same book was vetoed only incidentally
    # (walked exit 0.0) — protection depended on an optional A/B gate.
    # Every trade needs BOTH ladders: BUY_YES enters on yes_asks and
    # exits on yes_bids, BUY_NO enters on yes_bids and exits on
    # yes_asks, so a book lacking a live, tradeably-priced ladder on
    # either side lacks the chosen side's ladder or its exit ladder for
    # whichever side could be chosen, and must SKIP regardless of Gate
    # D configuration. "Ladder" here means live + tradeable price
    # (_has_tradeable_ladder) — the exact population the entry sweep
    # and the exit walk use — so zero-size-only and invalid-price-only
    # exit ladders are one-sided too. Placement is after Gate 0.6 so a
    # NaN-poisoned book keeps its pinned INVALID_EQUITY reason; the
    # both-sides-dead case is Gate 0.5's EMPTY_BOOK above (both XOR
    # operands False there, so this gate never fires for it). The SKIP
    # record carries market_mid_prob 0.0, NOT book.mid_price: on a
    # one-sided book the mid averages a real price with a sentinel
    # (0.255 for the prereg fixture) and is sentinel-derived, which the
    # prereg forbids on SKIP records. Zero sizes, per EC-09; cluster
    # dict untouched. Kill-criterion census (CENSUS in the task work
    # dir): of 1,131 forward S2 books (2026-10-20T00 + 2026-10-21T00),
    # 426 are one-sided, and pre-fix 1,278 of 3,688 gate-passing
    # candidates (34.7%, Gate D disabled) came from one-sided books —
    # far above the 1% kill threshold, so the kill does NOT trigger.
    _has_ask_ladder = _has_tradeable_ladder(book.yes_asks)
    _has_bid_ladder = _has_tradeable_ladder(book.yes_bids)
    if _has_ask_ladder != _has_bid_ladder:
        _missing_side = "yes_bids" if _has_ask_ladder else "yes_asks"
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=0.0,
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=(
                f"ONE_SIDED_BOOK (no live tradeable {_missing_side} ladder: "
                "entry and exit each require a live ladder on both sides)"
            ),
        )

    # Gate 0.7b (SF-BL-6060): no tradeable price on either side. Gate
    # 0.5 SKIPs only when BOTH ladders lack LIVE levels and Gate 0.7
    # only when EXACTLY ONE side lacks a tradeable ladder, so pre-fix
    # a book live on both sides but tradeable on NEITHER (every rung
    # priced outside (0, 1), e.g. asks [(-0.10, 1000)] / bids
    # [(1.50, 1000)]) — and the mirror shape, live-invalid asks with
    # raw-empty bids — passed both gates and reached the edge screen
    # below, which screened edge_yes / edge_no against the 0.99 /
    # 0.01 sentinel fallbacks and SKIPped INSUFFICIENT_NET_EDGE
    # (max_edge=-0.200 on the fixture): a fallback-screened verdict
    # mislabelled as a real market whose edge was merely insufficient,
    # with no flag that no measured touch existed on either side.
    # Such a book has NO observable price at all: SKIP here with the
    # named NO_TRADEABLE_PRICE reason, zero sizes (EC-09), cluster
    # dict untouched, and the declared mid 0.0 (SF-BL-3549). Placement
    # is immediately after Gate 0.7 (whose XOR it complements: 0.7 is
    # exactly-one-side, 0.7b is neither-side) and before Gate 0.8, so
    # the crossed check and every later gate only ever see books with
    # two measured touches. Two-sided tradeable books pass untouched.
    if not _has_ask_ladder and not _has_bid_ladder:
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=0.0,
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=(
                "NO_TRADEABLE_PRICE (no live tradeable ladder on either "
                "side: best_yes_ask / best_yes_bid are the 0.99 / 0.01 "
                "empty-ladder fallbacks, not measured touches "
                "(SF-BL-6060))"
            ),
        )

    # Gate 0.8 (SF-BL-9365): crossed / locked book. Both ladders are
    # live and tradeably priced here (Gate 0.5 / Gate 0.7 above have
    # disposed of the empty and one-sided cases, whose sentinel
    # fallbacks read uncrossed anyway), yet the best ask sits at or
    # below the best bid: the two ladders cannot coexist in a resting
    # book, so there is no coherent top of book to screen edge, size
    # Kelly, or walk an exit against — pre-fix the prereg fixture
    # (asks 0.40 / bids 0.60, fee 0.01, p=0.62) EXECUTED BUY_YES
    # $1,268.46 at q_exec 0.404 with mid 0.50 and best_no_ask 0.40
    # derived from the crossed bid, and a locked book traded too.
    # SKIP here — before Gate 1, the edge screen and any sizing —
    # with zero size (EC-09), cluster dict untouched, and
    # market_mid_prob 0.0, NOT book.mid_price: on a crossed book the
    # mid averages two prices that cannot coexist (the prereg
    # fixture's 0.50), so no mid is observable (the SF-BL-3549
    # declared-rule convention, as for Gates 0.45 / 0.7). The check
    # reads the best-price properties via _is_crossed_book, so dead
    # or invalid-price rungs neither fabricate nor mask a crossing,
    # and books mutated/unpickled after construction are caught here
    # exactly as at the other consumption gates. See the SF-BL-9365
    # block above _is_crossed_book.
    if _is_crossed_book(book):
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=0.0,
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=(
                f"CROSSED_BOOK (best_yes_ask={book.best_yes_ask} <= "
                f"best_yes_bid={book.best_yes_bid}: non-positive spread; "
                "crossed or locked books are not tradable (SF-BL-9365))"
            ),
        )

    # Gate 1: CryptoBench on-chain & derivatives data verification gate.
    # SF-BL-10793: on_chain_verified is a strict bool here — Gate 0.9
    # above has already SKIPped every non-bool on this gated path, and
    # books reaching this check via construction were validated in
    # __post_init__ — so `not` below reads a real verification result.
    # SF-BL-10845: config.require_on_chain_for_crypto is likewise a
    # strict bool here — Gate E0 at evaluate entry has already SKIPped
    # every non-bool switch (and construction validated it) — so the
    # `and` below reads a real gate-switch value, never truthiness.
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
            market_mid_prob=_declared_mid_prob(book),
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

    # Gate 1.9 (SF-BL-10844): Gate 2 threshold validity, immediately
    # before Gate 2. Construction already rejects an invalid threshold
    # (FleetRiskConfig.__post_init__), but a config mutated/unpickled
    # after construction — or the shared mutable default instance,
    # mutated — bypasses that; the gate therefore re-checks and SKIPs
    # (zero size, per EC-09, cluster dict untouched) rather than ever
    # comparing against it. Pre-fix, threshold NaN / +inf on the prereg
    # fixture traded BUY_YES $15,000 (the comparison is never True, so
    # Gate 2 fails open) and threshold '0.22' raised a bare TypeError.
    # A valid threshold (finite real in [0, 1] per SF-BL-10844 as
    # tightened by SF-BL-9397; 0 and int values included)
    # passes this check untouched, so its Gate 2 behaviour — including
    # the strict-> boundary — is byte-identical. Ordered after the
    # earlier input-validity gates and Gate 1, which keep precedence.
    _gate2_threshold = getattr(config, "max_inter_agent_dispersion", None)
    if not _is_valid_max_inter_agent_dispersion(_gate2_threshold):
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=_declared_mid_prob(book),
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=(
                f"INVALID_MAX_INTER_AGENT_DISPERSION "
                f"(max_inter_agent_dispersion={_gate2_threshold!r} "
                f"is not a finite real >= 0 and <= 1 (outside [0, 1]))"
            ),
        )

    # Gate 2C (SF-BL-10761): culled / thin panel gate, immediately
    # before Gate 2 (after Gate 1.9; the SF-BL-9639 taper below reads
    # the same dispersion, so a culled panel SKIPs here regardless of
    # taper mode). The dispersion Gate 2 compares is a survivorship
    # statistic — pstdev over the members aggregate_fleet_forecasts
    # KEPT, 0.0 by construction for a single survivor — so whenever the
    # caller supplies the panel provenance threaded through the
    # aggregate return, a panel that was culled (panel_n_dropped > 0)
    # or is thin (panel_n_kept < 3) SKIPs instead of trading on
    # apparent agreement. Malformed provenance — exactly one count
    # supplied, a non-int / bool / negative count, or two counts
    # summing to zero — SKIPs as INVALID_PANEL_PROVENANCE rather than
    # being trusted or crashing. Both counts None (a caller holding
    # only a scalar dispersion from outside the aggregator) passes
    # untouched: provenance cannot be fabricated for a scalar, and
    # legacy behaviour is byte-identical. Certified-clean provenance
    # (0 drops, >= 3 kept) also passes untouched, so Gate 2 and the
    # taper keep their exact prior behaviour on clean panels. Zero
    # size (EC-09), cluster dict untouched, as for every gate here.
    if (panel_n_dropped is None) != (panel_n_kept is None) or (
            panel_n_dropped is not None
            and not (_is_valid_panel_count(panel_n_dropped)
                     and _is_valid_panel_count(panel_n_kept)
                     and panel_n_dropped + panel_n_kept >= 1)):
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=_declared_mid_prob(book),
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=(
                f"INVALID_PANEL_PROVENANCE (panel_n_dropped="
                f"{panel_n_dropped!r}, panel_n_kept={panel_n_kept!r}: "
                "panel provenance must be two non-negative int counts "
                "(bool excluded) summing to >= 1, or both None "
                "(SF-BL-10761))"
            ),
        )
    if panel_n_dropped is not None and (panel_n_dropped > 0 or panel_n_kept < 3):
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=_declared_mid_prob(book),
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=(
                f"CULLED_PANEL_GATE2 (n_dropped={panel_n_dropped}, "
                f"n_kept={panel_n_kept}: dispersion "
                f"{inter_agent_dispersion:.2f} is a survivorship "
                "statistic over kept members only; Gate 2 requires an "
                "unculled panel of >= 3 (SF-BL-10761))"
            ),
        )

    # SF-BL-9639: dispersion taper factor for THIS candidate. With
    # dispersion_taper_enabled off (default) the factor is exactly 1.0
    # and Gate 2 below is the binary cliff, byte-identical to pre-task;
    # with it on, Gate 2's terminal SKIP is replaced by the continuous
    # taper: a zero factor SKIPs here, at Gate 2's own position and
    # precedence, and a positive factor scales Step 4's target
    # fraction via _taper_factor. Both inputs are already validated by
    # this point (Gate 0.75 dispersion finiteness, Gate 1.9 threshold
    # validity), so the helper's corrupt-input fallback is unreachable
    # on this path.
    _taper_enabled = getattr(config, "dispersion_taper_enabled", False)
    _taper_factor = (
        dispersion_taper_factor(
            inter_agent_dispersion, config.max_inter_agent_dispersion)
        if _taper_enabled
        else 1.0
    )
    if _taper_enabled and _taper_factor == 0.0:
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=_declared_mid_prob(book),
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=(
                f"DISPERSION_TAPER_ZERO (dispersion={inter_agent_dispersion:.2f} "
                f">= threshold={config.max_inter_agent_dispersion:.2f}; taper "
                "factor 0.00 under SF-BL-9639 dispersion taper)"
            ),
        )

    # Gate 2: Supervisor reconciliation gate on high inter-agent disagreement
    if not _taper_enabled and inter_agent_dispersion > config.max_inter_agent_dispersion:
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=_declared_mid_prob(book),
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

    # Gate 2.5 (SF-BL-9397): FleetRiskConfig sizing-band validity,
    # immediately after Gate 2 and before the first read of any of the
    # three fields — the SF-BL-3839 effective hurdle below reads
    # config.min_net_edge, Step 4 reads config.kelly_multiplier and
    # config.max_cluster_allocation, and the Step-5 sweep stop reads
    # the effective hurdle. Construction already rejects an out-of-band
    # value (FleetRiskConfig.__post_init__), but a config mutated /
    # unpickled after construction — or the shared mutable default
    # instance, mutated — bypasses that; the gate therefore re-checks
    # and SKIPs (zero size, per EC-09, cluster dict untouched) rather
    # than ever sizing at a wild band. Pre-fix the wild config
    # (2.0 / 0.90 / -0.10) executed the prereg fixture at 20.02% of
    # equity and a string hurdle crashed the edge comparison with a
    # bare TypeError. In-band values pass this check untouched, so
    # their behaviour — edge screen, sizing, sweep — is byte-identical,
    # and every earlier gate (E0, 0.0-0.9, 1, 1.9, 2) keeps its
    # precedence. max_inter_agent_dispersion is validated at Gate 1.9
    # above (SF-BL-10844 as tightened to [0, 1] by this task), not here.
    _range_violation = _fleet_risk_config_range_violation(config)
    if _range_violation is not None:
        _rv_field, _rv_value, _rv_band = _range_violation
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=_declared_mid_prob(book),
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=(
                f"INVALID_CONFIG (invalid FleetRiskConfig: "
                f"{_rv_field}={_rv_value!r} outside mandated band "
                f"{_rv_band} (SF-BL-9397))"
            ),
        )

    # Gate 2.6 (SF-BL-10798 / SF-BL-10877): max_round_trip_cost
    # validity, immediately after Gate 2.5 and before the field's
    # first read (Gate D, where the veto is a bare
    # `rt_cost > max_round_trip_cost` comparison).
    # Construction already rejects an out-of-domain threshold
    # (FleetRiskConfig.__post_init__), but a config mutated / unpickled
    # after construction — or the shared mutable default instance,
    # mutated — bypasses that; the gate therefore re-checks and SKIPs
    # (zero size, per EC-09, cluster dict untouched) rather than ever
    # comparing against it. Pre-fix a mutated NaN threshold traded the
    # prereg fixture BUY_YES $1,500 (Gate D fails open: `rt_cost > NaN`
    # is always False) and a string threshold crashed Gate D with a
    # bare TypeError. SF-BL-10877 narrowed the domain to [0, 1]: a
    # mutated threshold above 1 traded identically pre-narrowing (a
    # cap no per-share cost on a $1 binary can exceed is a disabled
    # gate in disguise). None (Gate D disabled) and every finite real
    # in [0, 1] pass this check untouched, so their behaviour — flat,
    # lock-scaled and edge-keyed Gate D included — is byte-identical,
    # and every earlier gate keeps its precedence. A missing attribute
    # reads as None via getattr (Gate D disabled), as in Gate 2.5's
    # helper.
    _rt_threshold = getattr(config, "max_round_trip_cost", None)
    if not _is_valid_max_round_trip_cost(_rt_threshold):
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=_declared_mid_prob(book),
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=(
                f"INVALID_CONFIG (INVALID_MAX_ROUND_TRIP_COST: "
                f"max_round_trip_cost={_rt_threshold!r} is not None "
                f"or a finite real in [0, 1] "
                f"(SF-BL-10798 / SF-BL-10877))"
            ),
        )

    # Gate 2.7 (SF-BL-10878 type / SF-BL-10821 range):
    # crypto_kelly_dampener validity, immediately after Gate 2.6 and before the field's
    # first read (Step 4's domain_mult, where it multiplies the
    # fractional Kelly lambda). Construction already rejects an
    # invalid dampener (FleetRiskConfig.__post_init__), but a config
    # mutated / unpickled after construction — or the shared mutable
    # default instance, mutated — bypasses that; the gate therefore
    # re-checks and SKIPs (zero size, per EC-09, cluster dict
    # untouched) rather than ever sizing at it. Pre-fix a mutated
    # True dampener sized the prereg fixture BUY_YES $8,350.25 —
    # exactly double the $4,175.13 mandated-dampener stake, the bool
    # silently removing the PolyBench x0.5 dampener — a mutated False
    # zeroed the stake behind a misleading CLUSTER_CAP_REACHED
    # reason, and '0.5' / None crashed Step 4 with bare TypeErrors.
    # Scoped to CRYPTO exactly as Gate 0.9 scopes to where its flag
    # is read: the dampener is read ONLY on the CRYPTO sizing path,
    # so non-crypto books pass this gate untouched for EVERY
    # dampener value and stay bit-identical. In-domain values (0.50
    # mandated; finite reals in (0, 1], int 1 included) pass
    # untouched, and every earlier gate keeps its precedence. A
    # missing attribute reads as None via getattr and is rejected,
    # as in Gate 2.5's helper.
    if book.domain == MarketDomain.CRYPTO:
        _dampener = getattr(config, "crypto_kelly_dampener", None)
        if not _is_valid_crypto_kelly_dampener(_dampener):
            return TradeExecutionResult(
                market_id=book.market_id,
                title=book.title,
                domain=book.domain,
                factor_cluster=book.factor_cluster,
                action=TradeAction.SKIP,
                calibrated_prob_yes=calibrated_prob_yes,
                market_mid_prob=_declared_mid_prob(book),
                exec_price=0.0,
                liquidation_bid_price=0.0,
                kl_divergence_nats=0.0,
                expected_log_growth=0.0,
                kelly_fraction_raw=0.0,
                allocated_fraction=0.0,
                invested_capital=0.0,
                shares_acquired=0.0,
                est_days_to_resolution=book.est_days_to_resolution,
                skip_reason=(
                    f"INVALID_CONFIG (INVALID_CRYPTO_KELLY_DAMPENER: "
                    f"crypto_kelly_dampener={_dampener!r} (type "
                    f"{type(_dampener).__name__}) is not a finite "
                    f"real in (0, 1] (SF-BL-10878 / SF-BL-10821))"
                ),
            )

    # Gate 2.8 (SF-BL-10972 / SF-BL-10820): min_confidence validity,
    # immediately after Gate 2.7 and before the edge screen and
    # Gate 3 — the field's first reads are Gate 3's bare
    # `p_side < config.min_confidence` comparison and its `:.2f`
    # formatting in the LOW_CONVICTION reason. Construction already
    # rejects an invalid threshold (FleetRiskConfig.__post_init__),
    # but a config mutated / unpickled after construction — or the
    # shared mutable default instance, mutated — bypasses that; the
    # gate therefore re-checks and SKIPs (zero size, per EC-09,
    # cluster dict untouched) rather than ever comparing against
    # it. Pre-fix a mutated False traded the prereg fixture BUY_YES
    # $4,595.96 (read as 0.0, Gate 3 never fired where the default
    # 0.60 SKIPs LOW_CONVICTION), a mutated True silently raised
    # the bar to 1.00, a NaN threshold failed the gate open, and
    # '0.6' crashed Gate 3 with a bare TypeError. Unconditional (not
    # domain scoped): Gate 3 reads the field for every domain.
    # Genuine thresholds in [0, 1] (0.60 default; int 0 / 1
    # included) pass this check untouched, so their behaviour —
    # edge screen, Gate 3 including its ratio >= 1.35 alternative
    # arm, and sizing — is byte-identical, and every earlier gate
    # keeps its precedence. A missing attribute reads as None via
    # getattr and is rejected, as in Gate 2.5's helper.
    _min_conf = getattr(config, "min_confidence", None)
    if not _is_valid_min_confidence(_min_conf):
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=_declared_mid_prob(book),
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=(
                f"INVALID_CONFIG (INVALID_MIN_CONFIDENCE: "
                f"min_confidence={_min_conf!r} (type "
                f"{type(_min_conf).__name__}) is not a finite "
                f"real in [0, 1] (SF-BL-10972 / SF-BL-10820))"
            ),
        )

    # SF-BL-3839: effective post-cost edge hurdle for THIS candidate.
    # With lock_scaled_hurdle_enabled off (default) this is exactly
    # config.min_net_edge, byte-identical to pre-task behaviour; with it
    # on, the hurdle rises with the book's est_days_to_resolution via
    # lock_scaled_min_net_edge (flat base as floor, sqrt(lock / D_REF)
    # beyond D_REF). The same effective hurdle governs the screen below
    # and the Step-5 sweep's marginal stop, so a long-lock fill must
    # clear it at both points, not only at the top of book.
    _eff_min_net_edge = (
        lock_scaled_min_net_edge(
            config.min_net_edge,
            book.est_days_to_resolution,
            getattr(config, "lock_hurdle_ref_days", LOCK_HURDLE_REF_DAYS),
        )
        if getattr(config, "lock_scaled_hurdle_enabled", False)
        else config.min_net_edge
    )

    # Determine candidate side & top-of-book fee-adjusted ask price
    yes_top_ask_eff = book.best_yes_ask * (1.0 + book.fee_rate)
    # SF-BL-8892: the NO effective ask is mirror-consistent — the fee is
    # charged on the mirrored YES leg (bid * (1 - fee)), not on the
    # complement price (best_no_ask * (1 + fee)); see the SF-BL-8892
    # block above _mirror_no_effective_ask.
    no_top_ask_eff = _mirror_no_effective_ask(book.best_yes_bid, book.fee_rate)

    edge_yes = calibrated_prob_yes - yes_top_ask_eff
    edge_no = (1.0 - calibrated_prob_yes) - no_top_ask_eff

    if edge_yes >= edge_no and edge_yes >= _eff_min_net_edge:
        chosen_side = TradeAction.BUY_YES
        p_side = calibrated_prob_yes
        q_top = yes_top_ask_eff
        # SF-BL-3549: declared liquidation bid — past Gate 0.7 both
        # ladders are tradeable, so this equals the raw property; the
        # helper keeps the sentinel out by construction, not by gate
        # ordering alone.
        liq_bid = _declared_liquidation_bid(book, TradeAction.BUY_YES)
    elif edge_no > edge_yes and edge_no >= _eff_min_net_edge:
        chosen_side = TradeAction.BUY_NO
        p_side = 1.0 - calibrated_prob_yes
        q_top = no_top_ask_eff
        liq_bid = _declared_liquidation_bid(book, TradeAction.BUY_NO)
    else:
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=_declared_mid_prob(book),
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=f"INSUFFICIENT_NET_EDGE (max_edge={max(edge_yes, edge_no):.3f} < {_eff_min_net_edge:.3f})",
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
            market_mid_prob=_declared_mid_prob(book),
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
    # SF-BL-9639: with the taper enabled, the effective lambda is the
    # domain multiplier scaled by _taper_factor; with it off the factor
    # is exactly 1.0 and this product is byte-identical to pre-task.
    target_fraction = kelly_raw * domain_mult * _taper_factor

    # Enforce 15% correlated factor-cluster concentration cap (Prediction Arena Section 5.4.1)
    # SF-BL-9396: sanitise the caller-supplied used fraction to [0, cap]
    # (non-finite / non-numeric fails closed at cap) before computing
    # room — a negative used must never inflate room past the cap.
    # SF-BL-10762: the sanitised fraction is now produced by the
    # re-base helper — exposure the engine itself wrote is tracked in
    # dollars and re-based to CURRENT portfolio_equity before the
    # room computation (a stored fraction of a stale equity vintage
    # understates live exposure after a drawdown and overstates it
    # after gains); a caller-seeded fraction keeps the SF-BL-9396
    # sanitised-fraction-of-current-equity reading byte-identically.
    # _cluster_dollars_used is the dollar base of the ledger write
    # below.
    used_in_cluster, _cluster_dollars_used = _rebased_cluster_exposure_used(
        cluster_exposure_used, book.factor_cluster,
        config.max_cluster_allocation, portfolio_equity)
    remaining_cluster_room = max(0.0, config.max_cluster_allocation - used_in_cluster)
    capped_fraction = min(target_fraction, remaining_cluster_room)

    if capped_fraction <= 0.001:
        # EC-09: SKIP sizes zero — kelly_fraction_raw is 0.0 here, as on every
        # other SKIP branch above (it previously leaked the pre-cap kelly_raw).
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=_declared_mid_prob(book),
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
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
        min_net_edge=_eff_min_net_edge,
    )

    # SF-BL-9438: non-finite sweep returns fail closed before execution.
    # A book can still make the sweep return non-finite values with a
    # finite budget (e.g. a deep rung with a NaN price: spend stays finite
    # while shares and q_exec come back NaN), and every `<= 0` check below
    # is False for NaN — without this guard such an opportunity executed
    # with NaN shares and wrote a finite-looking fraction into the cluster
    # dict on top of NaN share accounting. SKIP, zero size, no cluster write.
    if not all(
        _is_finite_number(v)
        for v in (invested_capital, shares_acquired, q_exec)
    ):
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=_declared_mid_prob(book),
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=(
                "INVALID_EQUITY (non-finite sweep return: "
                f"invested={invested_capital!r}, shares={shares_acquired!r}, "
                f"q_exec={q_exec!r})"
            ),
        )

    # SF-BL-8891: parity backstop. The sweep's in-loop parity stop
    # above (effective_rung_price >= 1.0 breaks before ANY hurdle) makes
    # this unreachable through the sweep as written — but a binary
    # contract cannot execute at or above its $1.00 maximum payout, so
    # a q_exec >= 1.0 reaching this point (a future bypass, or a VWAP
    # rounding to exactly 1.0 on prices within 1e-12 of parity) fails
    # closed here rather than executing a trade whose
    # expected_log_wealth_growth would be zeroed by its q>=1 guard
    # while kl_divergence_binary clips the price to 1-1e-6 and reports
    # a phantom edge. SKIP, zero size, cluster dict untouched (EC-09).
    if q_exec >= 1.0:
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=_declared_mid_prob(book),
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason=(
                f"Q_EXEC_AT_OR_ABOVE_PARITY (q_exec={q_exec:.6f} >= 1.0: "
                "fee-adjusted execution price at/above the $1.00 binary "
                "maximum payout — no positive log-wealth growth exists "
                "at this price (SF-BL-8891))"
            ),
        )

    if invested_capital <= 0.0 or shares_acquired <= 0.0:
        # EC-09: SKIP sizes zero — kelly_fraction_raw is 0.0 here, as on every
        # other SKIP branch above (it previously leaked the pre-sweep kelly_raw).
        return TradeExecutionResult(
            market_id=book.market_id,
            title=book.title,
            domain=book.domain,
            factor_cluster=book.factor_cluster,
            action=TradeAction.SKIP,
            calibrated_prob_yes=calibrated_prob_yes,
            market_mid_prob=_declared_mid_prob(book),
            exec_price=0.0,
            liquidation_bid_price=0.0,
            kl_divergence_nats=0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.0,
            allocated_fraction=0.0,
            invested_capital=0.0,
            shares_acquired=0.0,
            est_days_to_resolution=book.est_days_to_resolution,
            skip_reason="ORDER_BOOK_DEPTH_EXHAUSTED_AT_REQUIRED_EDGE",
        )

    # Gate BS (SF-BL-3659): budget-sensitivity veto. The candidate has
    # passed the gated sweep at its single Kelly target_budget; re-sweep
    # it as a full walk at 0.5x and 2x that budget and SKIP when the
    # post-cost edge sign flips or D_KL per dollar at least halves at
    # either size (see the SF-BL-3659 block above budget_sensitivity_
    # check). Zero sizes (EC-09), cluster dict untouched — this gate
    # sits before Gate D and before the cluster write below.
    if getattr(config, "budget_sensitivity_veto_enabled", True):
        _bs_check = budget_sensitivity_check(
            book, chosen_side, p_side, target_budget, q_exec
        )
        if _bs_check["veto"]:
            return TradeExecutionResult(
                market_id=book.market_id,
                title=book.title,
                domain=book.domain,
                factor_cluster=book.factor_cluster,
                action=TradeAction.SKIP,
                calibrated_prob_yes=calibrated_prob_yes,
                market_mid_prob=_declared_mid_prob(book),
                exec_price=0.0,
                liquidation_bid_price=0.0,
                kl_divergence_nats=0.0,
                expected_log_growth=0.0,
                kelly_fraction_raw=0.0,
                allocated_fraction=0.0,
                invested_capital=0.0,
                shares_acquired=0.0,
                est_days_to_resolution=book.est_days_to_resolution,
                skip_reason=(
                    "BUDGET_SENSITIVITY_VETO ("
                    + "; ".join(_bs_check["triggers"])
                    + ")"
                ),
            )

    # Gate D (SF-BL-7432): round-trip gate. The entry sweep above proved
    # p_side - q_exec >= min_net_edge at entry cost alone, but the position
    # is marked at its liquidation bid (NORTH_STAR CWR_bid): if an
    # immediate exit at the size-walked bid already costs more than
    # the effective cap per share (config.max_round_trip_cost, or its
    # SF-BL-10619 lock-scaled per-candidate value when that A/B flag is
    # on), the paper edge does not
    # survive the round trip and the trade is vetoed — SKIP, zero size
    # (EC-09), cluster dict untouched (this gate sits before the cluster
    # write below). max_round_trip_cost=None disables the gate (A/B use).
    if config.max_round_trip_cost is not None:
        rt_cost, q_exit_walked, _exit_fill = round_trip_cost(
            book, chosen_side, shares_acquired, q_exec
        )
        # SF-BL-10619: effective Gate D cap — the flat field, or the
        # per-candidate lock-scaled cap when the A/B flag is on. With
        # the flag off _rt_cap IS config.max_round_trip_cost, so the
        # comparison and the SKIP reason below are byte-identical.
        _rt_cap = config.max_round_trip_cost
        if getattr(config, "round_trip_lock_scaled_enabled", False):
            _rt_cap = lock_scaled_max_round_trip_cost(
                config.max_round_trip_cost,
                book.est_days_to_resolution,
                getattr(config, "round_trip_lock_ref_days",
                        ROUND_TRIP_LOCK_REF_DAYS),
            )
        # SF-BL-10726: edge-keyed Gate D — when the A/B flag is on, the
        # veto keys to the candidate's own entry edge instead of the
        # flat / lock-scaled cap: veto iff rt_cost >= p_side - q_exec
        # (no positive net round-trip edge remains). The predicate
        # fails to flat on corrupt input (unreachable via evaluate,
        # where both quantities are finite). With the flag off this
        # branch is inert and the comparison + SKIP reason below are
        # byte-identical.
        _edge_keyed = bool(
            getattr(config, "round_trip_edge_keyed_enabled", False))
        _entry_edge = p_side - q_exec
        _gate_d_veto = rt_cost > _rt_cap
        if _edge_keyed:
            if (_is_finite_number(rt_cost)
                    and _is_finite_number(_entry_edge)):
                _gate_d_veto = edge_keyed_gate_d_veto(rt_cost, _entry_edge)
        # SF-BL-11194: relative Gate D prong — when the A/B flag is on,
        # Gate D additionally vetoes when the round trip consumes more
        # than the pre-registered share of the entry price
        # (rt_cost / q_exec > max_share), the region where the absolute
        # cap is vacuous (rt_cost <= q_exec <= cap). The predicate
        # fails to flat on corrupt input (unreachable via evaluate).
        # With the flag off this branch is inert and the comparison +
        # SKIP reason below are byte-identical.
        _relative_veto = False
        if bool(getattr(config, "round_trip_relative_enabled", False)):
            _relative_veto = relative_round_trip_veto(
                rt_cost, q_exec,
                getattr(config, "round_trip_relative_max_share",
                        RELATIVE_ROUND_TRIP_MAX_SHARE))
            _gate_d_veto = _gate_d_veto or _relative_veto
        if _gate_d_veto:
            return TradeExecutionResult(
                market_id=book.market_id,
                title=book.title,
                domain=book.domain,
                factor_cluster=book.factor_cluster,
                action=TradeAction.SKIP,
                calibrated_prob_yes=calibrated_prob_yes,
                market_mid_prob=_declared_mid_prob(book),
                exec_price=0.0,
                liquidation_bid_price=0.0,
                kl_divergence_nats=0.0,
                expected_log_growth=0.0,
                kelly_fraction_raw=0.0,
                allocated_fraction=0.0,
                invested_capital=0.0,
                shares_acquired=0.0,
                est_days_to_resolution=book.est_days_to_resolution,
                skip_reason=(
                    f"ROUND_TRIP_GATE_D_RELATIVE (rt_cost={rt_cost:.3f}; "
                    f"share={rt_cost / q_exec:.3f} > "
                    f"{float(getattr(config, 'round_trip_relative_max_share', RELATIVE_ROUND_TRIP_MAX_SHARE)):.3f}; "
                    f"q_exec={q_exec:.3f}, "
                    f"q_exit_walked={q_exit_walked:.3f})"
                    if _relative_veto else
                    f"ROUND_TRIP_GATE_D_EDGE_KEYED (rt_cost={rt_cost:.3f} >= "
                    f"entry_edge={_entry_edge:.3f}; q_exec={q_exec:.3f}, "
                    f"q_exit_walked={q_exit_walked:.3f})"
                    if _edge_keyed else
                    f"ROUND_TRIP_GATE_D (rt_cost={rt_cost:.3f} > "
                    f"{_rt_cap:.3f}; q_exec={q_exec:.3f}, "
                    f"q_exit_walked={q_exit_walked:.3f})"
                ),
            )

    actual_fraction = invested_capital / portfolio_equity
    # SF-BL-8890: this increment is half of a lifecycle — the matching
    # decrement is release_cluster_exposure, called from
    # settle_or_mark_trade when this trade settles (the returned trade
    # carries a reference to this dict so the release finds it).
    # SF-BL-10762: the increment is accumulated in DOLLARS (the true
    # amount at risk, vintage-free) and the dict stores the total as
    # a fraction of the CURRENT equity — identical to the legacy
    # used + actual_fraction write whenever equity has not moved, and
    # the true 15%-of-current-equity exposure when it has. The dollar
    # total is registered so the next call can re-base it.
    if _cluster_dollars_used is not None:
        _new_cluster_dollars = _cluster_dollars_used + invested_capital
        cluster_exposure_used[book.factor_cluster] = (
            _new_cluster_dollars / portfolio_equity)
        _record_cluster_dollars(
            cluster_exposure_used, book.factor_cluster,
            _new_cluster_dollars, portfolio_equity,
            cluster_exposure_used[book.factor_cluster])
    else:
        cluster_exposure_used[book.factor_cluster] = (
            used_in_cluster + actual_fraction)

    kl_nats = kl_divergence_binary(p_side, q_exec)
    exp_log_g = expected_log_wealth_growth(p_side, q_exec, actual_fraction)

    return TradeExecutionResult(
        market_id=book.market_id,
        title=book.title,
        domain=book.domain,
        factor_cluster=book.factor_cluster,
        action=chosen_side,
        calibrated_prob_yes=calibrated_prob_yes,
        market_mid_prob=_declared_mid_prob(book),
        exec_price=q_exec,
        liquidation_bid_price=liq_bid,
        kl_divergence_nats=kl_nats,
        expected_log_growth=exp_log_g,
        kelly_fraction_raw=kelly_raw,
        kelly_fraction_sized=capped_fraction,
        allocated_fraction=actual_fraction,
        invested_capital=invested_capital,
        shares_acquired=shares_acquired,
        est_days_to_resolution=book.est_days_to_resolution,
        skip_reason=None,
        # SF-BL-8890: remember which cluster ledger this trade's
        # allocated_fraction was added to, so settlement can release it.
        cluster_exposure_used=cluster_exposure_used,
        # SF-BL-34952: remember the cluster KEY that ledger was
        # charged under, so settlement releases that key even if the
        # record's factor_cluster is renamed in between.
        entry_factor_cluster=book.factor_cluster,
        # SF-BL-35212: remember the ledger DICT itself, so settlement
        # can tell the dict evaluate charged from any reference later
        # substituted into the mutable cluster_exposure_used field.
        entry_cluster_ledger=cluster_exposure_used,
        # SF-BL-10767: stamp the entry bid's freshness provenance —
        # liq_bid was read from THIS book, so the mark's as-of is the
        # book's observation time (None when the producer did not
        # timestamp the snapshot: the mark is then honestly
        # unprovenanced and flagged/excluded at scoring).
        liquidation_bid_as_of=_snapshot_ts(book),
        liquidation_bid_source=BID_SOURCE_ENTRY_BOOK,
    )


# ============================================================================
# 5. SETTLEMENT, BID-MARKED VALUATION & NORTH STAR SCORING ENGINE (N_star)
# ============================================================================

# --- SF-BL-3660: bid-ladder VWAP liquidation marks -----------------------
# Pre-fix, the unsettled branch of settle_or_mark_trade valued an open
# position as shares_acquired * liquidation_bid_price — the single
# entry-time TOP bid — regardless of position size or bid depth, while
# the entry side walked the full ask ladder in sweep_order_book and
# Gate D (SF-BL-7432) walked the exit ladder only to VETO new trades,
# never to mark open ones. Exit capacity was therefore invisible to
# marks and to the MDD walked from them in evaluate_fleet_north_star:
# a 2,352-share position on a 1,500-share top bid rung marked as if
# every share could exit at that rung's price.
# The fix walks the bid side scaled to shares_acquired, reusing
# size_walked_exit_bid (SF-BL-7432: live, tradeably-priced levels only;
# BUY_YES exits on yes_bids, BUY_NO on the derived NO bid ladder
# 1 - yes_ask). The per-share liquidation mark is the position VWAP
#     mark = proceeds / shares_acquired
#          = (filled-portion bid VWAP) * fill_fraction,
# where proceeds is what an immediate liquidation actually raises:
# when the ladder cannot absorb the whole position, the unfilled
# remainder contributes $0 of proceeds — it cannot be exited at any
# observed bid — so depth shortfall lowers the mark, exactly the exit
# risk the top-bid mark hid. This differs deliberately from Gate D's
# filled-portion VWAP (a per-filled-share price used as a lower bound
# on round-trip cost); both components (filled VWAP, fill fraction)
# are returned here and recorded on the trade so no information is lost.
# A ladder with no live tradeable depth yields mark 0.0 ("no exit —
# mark at zero", the SF-BL-3549 declared convention). The settled
# branch never reads the ladder: a settled binary pays $1/$0 whatever
# the book looked like, so settled behaviour is byte-identical.

def bid_ladder_liquidation_mark(
    book: OrderBookSnapshot,
    side: TradeAction,
    shares: float,
) -> Tuple[float, float, float, float]:
    """Bid-ladder VWAP liquidation mark for liquidating `shares` on `side`.

    Returns (mark_per_share, filled_vwap, fill_fraction, proceeds):
    proceeds is the cash an immediate liquidation of the whole position
    raises by sweeping the bid ladder; mark_per_share = proceeds /
    shares (unfilled remainder valued at $0). (0.0, 0.0, 0.0, 0.0) when
    shares is non-finite / <= 0, the side is not BUY_YES / BUY_NO, or
    there is no live, tradeably-priced exit depth.
    """
    if not _is_finite_number(shares) or shares <= 0:  # type: ignore[operator]
        return 0.0, 0.0, 0.0, 0.0
    filled_vwap, fill_fraction = size_walked_exit_bid(book, side, shares)
    proceeds = filled_vwap * fill_fraction * float(shares)  # type: ignore[arg-type]
    mark = proceeds / float(shares)  # type: ignore[arg-type]
    return mark, filled_vwap, fill_fraction, proceeds


# --- SF-BL-8890: cluster exposure is never released at settlement -------
# evaluate_and_size_opportunity only ever INCREMENTED
# cluster_exposure_used[cluster] (by invested / entry equity after a
# successful sweep) and settle_or_mark_trade never decremented it, so a
# long-lived caller's 15% per-cluster cap permanently exhausted after the
# first settled trade in each cluster: on the prereg sequence (deep book,
# p=0.80, equity $10,000, settle-before-next-entry) round 1 took the full
# 0.15 fraction, settled, and all 19 later rounds SKIPped
# CLUSTER_CAP_REACHED — headroom restored in 0% of rounds (repro.py in
# the task work dir). The stored fractions also mixed equity bases: each
# addend was taken against the equity of its own entry, so the dict sum
# was not a fraction of any single equity. nstar.py's Account.settle
# proves the intended lifecycle — it decrements cluster_stakes by the
# settled position's stake. The fix mirrors it in this module's fraction
# units: settlement releases EXACTLY the allocated_fraction the trade
# added (removing that addend restores the pre-trade sum exactly,
# whatever equity did in between, and once every open trade in a cluster
# has settled the dict returns to 0.0 — full cap headroom in
# current-equity terms). Release is settled-only (an open mark keeps the
# exposure — the position is still at risk), idempotent per trade
# (cluster_exposure_released), clamped at 0.0, and dust below 1e-12 is
# snapped to 0.0 (well under the prereg's 1e-9 tolerance). The ledger is
# found via settle's explicit cluster_exposure_used argument when given,
# else via the reference evaluate stored on the trade — so legacy
# settle(trade, True, outcome) calls release too. SKIP / zero-invested
# trades added nothing and release nothing; a settle that raises (bad
# outcome, out-of-domain position fields) releases nothing.

# --- SF-BL-34868: settlement release honours any ledger it is handed --
# release_cluster_exposure resolved its ledger as "the explicit
# `cluster_exposure_used` when given, else the dict evaluate stored on
# the trade" and never checked the two were the same dict, although the
# trade itself stores the entry reference (SF-BL-8890). A settle handed
# a DIFFERENT dict therefore released into the wrong ledger. Verified
# live pre-fix (repro.py in the task work dir): a trade entered against
# ledger A (cluster fraction 0.15) settled with
# cluster_exposure_used=ledger B, which held another open position's
# 0.10 — B was rewritten to 0.0 (phantom cap headroom for that cluster),
# the trade was flagged cluster_exposure_released, and A still held
# 0.15 with no release path left. The dollar-registered path
# (SF-BL-10762) corrupted identically: B's registered dollars were
# decremented by the settled trade's invested dollars (B at 0.10252…
# rewritten to 0.0) and its fraction re-derived from the corrupted
# total. Kill-criterion audit (CALLER_CENSUS.md in the task work dir):
# NOT met — the sole in-repo production caller passes the entry ledger
# by local discipline only (one mutable local shared by evaluate and
# settle; no invariant or test pins the identity), the SF-BL-8890 suite
# itself exercises the explicit-argument surface with the stored
# reference nulled, release_cluster_exposure is a public entry point,
# and out-of-tree callers are unbounded, so no tested invariant stands
# between a mismatched dict and this release. The identity is
# therefore validated HERE, at both consumption points: when the trade
# carries a stored entry ledger AND an explicit ledger is given AND
# the two are not the same object, release_cluster_exposure raises the
# named ValueError below before any ledger, registry entry, or trade
# field is touched, and settle_or_mark_trade pre-validates the same
# condition at the head of its settled branch — before the outcome
# validation and before any field of the trade is written — so a
# rejected settle leaves the trade byte-identical (SF-BL-10724
# ordering) and both ledgers byte-identical, with the correct release
# path (settle with no argument, or with the entry ledger itself)
# still available afterwards. Preserved surfaces, byte-identical:
# explicit IS the stored dict (the benchmark pattern); stored is None
# and an explicit dict is given (the SF-BL-8890 explicit-argument
# path — the explicit dict is the only ledger the trade knows);
# explicit is None (release via the stored reference); neither (the
# legacy no-ledger settle). Open marks never release and never read
# the argument, so they are unchecked and unchanged. The check runs
# before the idempotency guard: a mismatched argument is a caller
# error even when no release remains for the trade.

def _validate_cluster_ledger_match(
    trade: TradeExecutionResult,
    cluster_exposure_used: Optional[Dict[str, float]],
) -> None:
    """Raise ValueError on a mismatched settlement ledger (SF-BL-34868).

    No-op unless the trade stores an entry ledger (SF-BL-8890) and an
    explicit ledger is given that is not that same dict object.
    """
    stored = trade.cluster_exposure_used
    if (cluster_exposure_used is not None and stored is not None
            and cluster_exposure_used is not stored):
        raise ValueError(
            f"MISMATCHED_CLUSTER_LEDGER: cluster_exposure_used "
            f"(id {id(cluster_exposure_used)}) is not the cluster "
            f"ledger evaluate stored on the trade (id {id(stored)}) "
            f"for cluster {_entry_factor_cluster(trade)!r} (SF-BL-34868): "
            "settlement must release the exact fraction evaluate "
            "added, from the exact dict it added it to — releasing "
            "into a different dict erases an unrelated position's "
            "exposure there (phantom cap headroom) and leaves the "
            "entry ledger permanently charged, the trade already "
            "flagged cluster_exposure_released; pass the entry "
            "ledger itself, or no argument to release via the "
            "reference stored on the trade")


# --- SF-BL-35212: the stored cluster ledger is never validated --------
# settle_or_mark_trade / release_cluster_exposure resolve the legacy
# release ledger (explicit argument None) as "the dict evaluate stored
# on the trade" (SF-BL-8890) — and neither consumption point validated
# that stored reference itself. TradeExecutionResult.cluster_
# exposure_used is a plain public field, settable at construction and
# by assignment with no type or identity check (the record's
# __setattr__ guards action / is_settled / cluster_exposure_released
# only). Verified live pre-fix (repro.py in the task work dir; deep
# book ask 0.50 / bid 0.48, fee 0.01, p=0.60, equity $10,000, cluster
# 'C1' — evaluate charges the entry ledger {'C1': 0.0672}, a win is
# worth PnL +$658.42):
#   (a) stored reference replaced by a LIST: the legacy settle booked
#       the win (is_settled=True, net_pnl_bid_marked +$658.42) and
#       only then raised a bare AttributeError ('list' object has no
#       attribute 'get') from inside release_cluster_exposure,
#       leaving the entry ledger charged {'C1': 0.0672} and
#       cluster_exposure_released=False — every retry re-crashes and
#       the cap charge is stranded behind a booked settlement;
#   (b) stored reference replaced by a FRESH DICT: settle silently
#       released into the substitute (which became {'C1': 0.0}) and
#       consumed the released flag while the entry ledger kept
#       {'C1': 0.0672} forever — phantom headroom in the substitute,
#       a permanently charged entry ledger, no error anywhere.
# Closest neighbours read in full: SF-BL-35167 pre-validates only the
# explicit cluster_exposure_used ARGUMENT's type, and SF-BL-34868
# raises only when an explicit argument differs from the stored
# reference — the legacy path (argument None) reads the stored
# reference itself, which neither item validates; SF-BL-34996 owns
# release's own argument types at its boundary, SF-BL-34952 the
# cluster KEY inside the dict. Kill-criterion audit (CALLER_CENSUS.md
# in the task work dir): NOT met — the stored reference IS reassigned
# after evaluate in exercised test surfaces (set to None to force the
# explicit-argument path; set to a ledger after direct construction),
# records are rebuilt / mutated / unpickled out of tree, and no
# invariant pins it, so an assignment-time lock would break pinned
# surfaces: the validation lives HERE, at both consumption points,
# mirroring SF-BL-34952's snapshot idiom. The dict evaluate charged
# is recorded on the record itself (entry_cluster_ledger: evaluate
# passes it explicitly; __post_init__ snapshots a dict stored at
# construction). _validate_stored_cluster_ledger then raises, BEFORE
# anything is touched:
#   * INVALID_STORED_CLUSTER_LEDGER when the stored reference is not
#     None and not a dict (the list case, and str / int / other
#     non-dict substitutions alike);
#   * SUBSTITUTED_CLUSTER_LEDGER when an entry snapshot exists and
#     the stored reference is a different dict object than the one
#     evaluate charged, or the effective release ledger (explicit
#     argument when given, else the stored reference) is a different
#     dict than the entry ledger — the stored-None + foreign-explicit
#     combination SF-BL-34868 cannot see (its check needs a stored
#     reference to differ from).
# Placement: release_cluster_exposure runs it at entry, immediately
# after the SF-BL-34868 check and before the SF-BL-35035 latch /
# idempotency guards (a corrupt or substituted stored reference is a
# caller error even when no release remains); settle_or_mark_trade
# pre-validates it at the head of its settled branch, before the
# outcome validation and before any field of the trade is written,
# so a rejected settle books NO PnL and leaves the trade and every
# ledger byte-identical, with the correct path (restore the entry
# reference, or settle with the entry ledger itself) still available.
# Preserved surfaces, byte-identical: genuine evaluate-to-settle
# lifecycles (stored IS the entry snapshot); stored=None at
# construction with an explicit dict at settle (no snapshot exists —
# the SF-BL-8890 explicit-argument path, taken at face value);
# manually constructed trades whose construction-time dict is their
# entry ledger (snapshot == stored); post-construction adoption by
# assignment (no snapshot — face value, type check still applies);
# records predating the snapshot (old pickles: getattr fallback, no
# snapshot, face value); open marks, which never release and never
# read the stored reference, unchecked and unchanged; SKIP /
# zero-invested settles, whose early return precedes the settled
# branch and attempts no release.

def _entry_cluster_ledger(
    trade: TradeExecutionResult,
) -> Optional[Dict[str, float]]:
    """The dict evaluate charged at entry, or None (SF-BL-35212).

    The entry_cluster_ledger snapshot when one is recorded (a dict);
    otherwise None — records that predate the snapshot (an old pickle
    whose __post_init__ never ran) and records constructed without a
    dict ledger carry no entry snapshot and are taken at face value.
    """
    entry = getattr(trade, "entry_cluster_ledger", None)
    if isinstance(entry, dict):
        return entry
    return None


def _validate_stored_cluster_ledger(
    trade: TradeExecutionResult,
    cluster_exposure_used: Optional[Dict[str, float]],
) -> None:
    """Raise ValueError on a corrupt / substituted stored ledger.

    Validates the cluster ledger STORED on the trade (SF-BL-35212):
    its type (None or dict) and, when an entry snapshot exists, its
    identity against the dict evaluate charged. The explicit
    argument is read only as the effective release ledger when a
    snapshot exists; its own type/identity against the stored
    reference is SF-BL-35167 / SF-BL-34868's surface.
    """
    stored = trade.cluster_exposure_used
    if stored is not None and not isinstance(stored, dict):
        raise ValueError(
            f"INVALID_STORED_CLUSTER_LEDGER: cluster_exposure_used "
            f"stored on the trade {stored!r} (type "
            f"{type(stored).__name__}) is not a dict "
            f"(SF-BL-35212): settle_or_mark_trade / "
            "release_cluster_exposure resolve the legacy release "
            "ledger through this stored reference, so a non-dict "
            "substitution (a list, a string, a number) reaches the "
            "release's ledger .get() — pre-fix the settle booked "
            "its PnL first and only then crashed with a bare "
            "AttributeError, stranding the entry ledger's charge "
            "behind a booked settlement; restore the dict evaluate "
            "stored on the trade (or None for a trade that never "
            "carried one) before settling")
    entry = _entry_cluster_ledger(trade)
    if entry is None:
        return
    if stored is not None and stored is not entry:
        raise ValueError(
            f"SUBSTITUTED_CLUSTER_LEDGER: cluster_exposure_used "
            f"stored on the trade (id {id(stored)}) is not the "
            f"cluster ledger evaluate charged for cluster "
            f"{trade.factor_cluster!r} (id {id(entry)}) "
            f"(SF-BL-35212): the stored reference was replaced "
            "after entry — releasing into the substitute would "
            "erase exposure it never held and leave the entry "
            "ledger permanently charged, the trade already flagged "
            "cluster_exposure_released; restore the entry ledger "
            "reference stored at entry before settling")
    effective = (cluster_exposure_used if cluster_exposure_used is not None
                 else stored)
    if effective is not None and effective is not entry:
        raise ValueError(
            f"SUBSTITUTED_CLUSTER_LEDGER: settlement ledger "
            f"(id {id(effective)}) is not the cluster ledger "
            f"evaluate charged for cluster {trade.factor_cluster!r} "
            f"(id {id(entry)}) (SF-BL-35212): the stored reference "
            "was cleared or replaced after entry and settlement "
            "would release into a dict evaluate never charged, "
            "leaving the entry ledger permanently charged; pass "
            "the entry ledger itself, or restore the reference "
            "stored on the trade at entry")


def release_cluster_exposure(
    trade: TradeExecutionResult,
    cluster_exposure_used: Optional[Dict[str, float]] = None,
) -> float:
    """Release a settled trade's cluster-cap fraction (SF-BL-8890).

    Subtracts the trade's allocated_fraction — the exact addend
    evaluate_and_size_opportunity wrote — from the cluster ledger
    (the explicit `cluster_exposure_used` when given, else the dict
    evaluate stored on the trade), clamped at 0.0. Returns the amount
    of fraction actually removed. No-ops (returning 0.0) for SKIP
    trades, non-positive / non-finite fractions, an already-released
    trade, or when no ledger is available. Pure bookkeeping: the
    trade's PnL fields are never touched.

    SF-BL-34868: when the trade stores an entry ledger and an explicit
    ledger is given that is NOT that same dict, raises the named
    MISMATCHED_CLUSTER_LEDGER ValueError before anything is touched —
    the explicit argument may repeat the entry ledger, never replace
    it (see the SF-BL-34868 block above this function).

    SF-BL-35035: the trade's cluster_exposure_released flag is
    strict-bool validated at entry (records that bypassed the
    dataclass guard raise the named INVALID_CLUSTER_EXPOSURE_
    RELEASED ValueError before anything is touched), and the
    shadow release latch is checked before the public flag: a
    trade whose release already succeeded no-ops at 0.0 even if
    its public flag was bypass-reset to False, and the public
    flag is restored to True (see the SF-BL-35035 block at the
    dataclass).

    SF-BL-34952: the ledger KEY released is the cluster recorded at
    entry (_entry_factor_cluster — the entry_factor_cluster snapshot
    taken at construction), not the trade's current factor_cluster:
    renaming factor_cluster between evaluate and settle cannot
    redirect the release to a new key, write a stray key into the
    ledger, or strand the entry key's charge (see the SF-BL-34952
    block above _release_cluster_dollars).

    SF-BL-35212: the ledger reference STORED on the trade is itself
    validated at entry — a non-dict stored reference raises the
    named INVALID_STORED_CLUSTER_LEDGER ValueError, and a stored
    (or effective) dict that is not the entry_cluster_ledger
    snapshot raises SUBSTITUTED_CLUSTER_LEDGER, before anything is
    touched (see the SF-BL-35212 block above this function).
    """
    # SF-BL-35035: flag-type validation first of all for genuine
    # records, before the ledger-identity check and before any
    # ledger or dollar-registry write — a bypass-constructed
    # non-bool flag must never be read by truthiness here. (A
    # non-TradeExecutionResult argument keeps its pre-existing
    # handling below; SF-BL-34996 owns that surface.)
    if isinstance(trade, TradeExecutionResult):
        _validate_trade_cluster_exposure_released(trade)
    # SF-BL-34868: ledger-identity validation next, before the
    # idempotency / SKIP / fraction guards and before any ledger or
    # dollar-registry write — a mismatched explicit ledger is a caller
    # error even when no release would remain for this trade.
    _validate_cluster_ledger_match(trade, cluster_exposure_used)
    # SF-BL-35212: stored-ledger validation immediately after, still
    # before the latch / idempotency / SKIP / fraction guards — a
    # non-dict or substituted stored reference is a caller error even
    # when no release would remain for this trade. Genuine records
    # only, like the SF-BL-35035 check above (a non-trade argument
    # keeps its pre-existing handling; SF-BL-34996 owns that surface).
    if isinstance(trade, TradeExecutionResult):
        _validate_stored_cluster_ledger(trade, cluster_exposure_used)
    # SF-BL-35035: shadow latch — a successful release already
    # happened for this trade, whatever the public flag now says
    # (a bypass reset cannot re-arm a second subtraction).
    if _cluster_release_latch(trade):
        if trade.cluster_exposure_released is not True:
            object.__setattr__(trade, "cluster_exposure_released", True)
        return 0.0
    if trade.cluster_exposure_released:
        # A record claiming released without a latch (constructed
        # with True, or a pre-fix pickle): honour the claim and
        # arm the latch so a later bypass reset cannot re-arm it.
        _set_cluster_release_latch(trade)
        return 0.0
    if trade.action == TradeAction.SKIP:
        return 0.0
    frac = trade.allocated_fraction
    if not _is_finite_number(frac) or frac <= 0.0:  # type: ignore[operator]
        return 0.0
    ledger = (cluster_exposure_used if cluster_exposure_used is not None
              else trade.cluster_exposure_used)
    if ledger is None:
        return 0.0
    # SF-BL-10762: when this ledger's cluster exposure is registered
    # in dollars, release the trade's exact invested dollars and
    # re-derive the stored fraction at its written vintage — the
    # legacy fraction subtraction below would remove an addend
    # denominated at the trade's entry equity from a sum the engine
    # has since re-based. Unregistered ledgers keep the SF-BL-8890
    # subtraction byte-identically.
    _dollars_removed = _release_cluster_dollars(trade, ledger)
    if _dollars_removed is not None:
        trade.cluster_exposure_released = True
        _set_cluster_release_latch(trade)
        return _dollars_removed
    # SF-BL-34952: resolve the key ONCE from the entry snapshot —
    # the legacy subtraction below and the dollar helper above must
    # agree on the key evaluate charged, whatever factor_cluster
    # says now.
    _cluster = _entry_factor_cluster(trade)
    used = ledger.get(_cluster, 0.0)
    if not _is_finite_number(used):
        return 0.0
    remaining = float(used) - float(frac)  # type: ignore[arg-type]
    if remaining < 1e-12:
        remaining = 0.0
    ledger[_cluster] = remaining
    trade.cluster_exposure_released = True
    _set_cluster_release_latch(trade)
    return float(used) - remaining  # type: ignore[arg-type]


# --- SF-BL-10724: settle/mark mutated settlement state before validating --
# settle_or_mark_trade wrote `trade.is_settled = is_settled` and
# `trade.outcome_yes = outcome_yes` at ENTRY, before the settled branch
# validated the outcome (SF-BL-9477) and before the open branch validated
# the liquidation bid (SF-BL-9436) — SF-BL-9541's position-field check
# was the only validation that ran before the writes. Every raising
# path therefore left the trade in a FLIPPED state carrying the previous
# branch's net_pnl_bid_marked. Verified live pre-fix (repro.py in the
# task work dir; 800 shares / $500 invested fixture):
#   * a settled trade at +$300.00 (win) re-marked at bid 1.50 raised the
#     SF-BL-9436 ValueError but came back is_settled=False /
#     outcome_yes=None with the +$300 settled PnL intact;
#   * an open trade marked at -$100.00 settled with outcome_yes=None
#     raised the SF-BL-9477 ValueError but came back is_settled=True
#     with the stale -$100 mark PnL intact (bool True did the same,
#     also overwriting outcome_yes with True).
# evaluate_fleet_north_star partitions on exactly those fields
# (settled_trades = [t for t in executed if t.is_settled]; wins counted
# from net_pnl_bid_marked), so a flipped record moves a trade into or
# out of the settled win-rate accounting with the wrong branch's PnL.
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT
# met — re-settlement / re-marking of one object is exercised,
# documented lifecycle (SF-BL-8890's release is idempotent "however
# often it is re-settled"; the cluster-release suite marks a trade
# open and then settles the SAME object, and settles another twice),
# so a raising second call is reachable, not theoretical. The fix is
# ordering only: on each branch, ALL validation — and, for the
# exit-book mark, the bid-ladder computation and its re-validation —
# completes BEFORE any field of the trade is written, and the new PnL
# is computed into a local and assigned together with is_settled /
# outcome_yes. A raising settle/mark therefore leaves is_settled,
# outcome_yes and net_pnl_bid_marked byte-identical to their pre-call
# values (and, on the exit-book path, liquidation_bid_price /
# liquidation_mark_vwap / liquidation_fill_fraction untouched too).
# Preserved orderings, pinned by earlier tasks: SF-BL-9541's
# position-field validation still runs FIRST, before the SKIP /
# zero-invested early return; that early return still precedes the
# outcome / bid validations (a SKIP or zero-invested trade returns
# PnL 0.0 without raising on an invalid outcome or bid — SF-BL-9477 /
# SF-BL-9436 / SF-BL-9541 suites), it simply writes its three fields
# with no validation outstanding, so no raising path can follow it.
# Successful calls are byte-identical: same PnL arithmetic, same
# stored fields, cluster release still only on a successful settlement.

# --- SF-BL-10971: the trade argument itself is never type-checked -------
# settle_or_mark_trade's entry validation (SF-BL-9541) dereferences
# trade.shares_acquired unguarded, so a non-TradeExecutionResult trade
# crashes bare instead of raising a named error. Verified live pre-fix
# (repro.py in the task work dir): trade=None — the failed
# evaluate-lookup case in a settlement batch — raises
# AttributeError: 'NoneType' object has no attribute 'shares_acquired',
# aborting the whole batch with no task id and no field named; a dict
# (a record that never made it through TradeExecutionResult(**row)
# reconstruction), a string, an int and a list crash identically on
# their own type names. Worse, a duck-typed fake carrying the right
# attributes passes every field validation: on the mark branch it is
# marked and returned as if genuine, and on the settled branch it
# books its PnL and flips is_settled BEFORE crashing deeper inside
# release_cluster_exposure on a missing bookkeeping attribute — a
# bare crash AFTER mutation, the SF-BL-10724 hazard in reverse.
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT
# met — trade does NOT only ever arrive as an
# evaluate_and_size_opportunity result: run_benchmark_simulation's
# naive fleet passes directly constructed records, fleet scripts and
# backlog analyses rebuild records from dicts
# (TradeExecutionResult(**pre.__dict__)), the `trade` annotation is
# unenforced, and no isinstance / tested invariant anywhere stands
# between a failed lookup (None), an unreconstructed dict, or a fake
# and this entry point. The type is therefore validated HERE, at
# entry, BEFORE _validate_settle_position_fields dereferences the
# argument: SF-BL-9541 remains first among the FIELD validations, but
# the type check is its precondition. isinstance, so a genuine
# TradeExecutionResult (subclasses included) passes bit-identically
# and every other object raises the named ValueError below with no
# field of any object read or written.

def _is_trade_execution_result(value: object) -> bool:
    """True iff value is a genuine TradeExecutionResult (SF-BL-10971)."""
    return isinstance(value, TradeExecutionResult)


def _validate_trade_argument(trade: object) -> None:
    """Raise ValueError unless trade is a TradeExecutionResult.

    Runs at settle_or_mark_trade entry, before any attribute of the
    argument is dereferenced (SF-BL-10971).
    """
    if not _is_trade_execution_result(trade):
        raise ValueError(
            f"INVALID_TRADE: trade {trade!r} "
            f"(type {type(trade).__name__}) is not a "
            "TradeExecutionResult (SF-BL-10971): settle_or_mark_trade "
            "dereferences trade.shares_acquired at its SF-BL-9541 "
            "entry validation, so a non-trade argument — None from a "
            "failed evaluate lookup in a settlement batch, an "
            "unreconstructed dict record, or a duck-typed fake — "
            "crashes with a bare AttributeError (or, for a fake "
            "carrying the right attributes, books PnL as if genuine) "
            "instead of raising this named error at entry, before "
            "any field is read or written")


# --- SF-BL-10822: settle_or_mark_trade settles on a truthy string -----
# settle_or_mark_trade consumed its is_settled PARAMETER by bare
# truthiness ('if is_settled:') with no type check at entry, so a
# JSON/CSV-sourced string 'False' took the settled branch and paid
# shares x $1.00 instead of marking at liquidation_bid_price.
# Verified live at filing (and re-verified this run, repro.py /
# repro_prefix.txt in the task work dir): on the prereg fixture
# (executed BUY_YES, $1,378.79 invested, 2730.28 shares, bid 0.48,
# outcome_yes=1) is_settled='False' booked a settled +$1,351.49 win
# instead of the open bid-mark of -$68.26 that is_settled=False
# returns on the same trade. SF-BL-9477 covers the bool outcome_yes
# and SF-BL-10724 covers mutation-before-validation; neither covers
# the is_settled flag's own type. SF-BL-11536 (landed after filing)
# validated the STORED field on TradeExecutionResult and, as a side
# effect, made most non-bool arguments raise INVALID_IS_SETTLED at
# the assignment write — but branch selection still ran on truthiness
# BEFORE that write, so 'False'/'True' with no (or an invalid)
# outcome raised SF-BL-9477's outcome error instead of an is_settled
# error, a falsy non-bool with an out-of-bound bid raised
# SF-BL-9436's bid error instead, and every such raise carried the
# scorer-partition rationale of SF-BL-11536, not this parameter's.
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT
# met — the sole in-repo production caller passes a VARIABLE unpacked
# from unvalidated scenario dicts (not a bool literal), no schema /
# deserialisation layer stands between an external producer and this
# parameter, the `bool` annotation is unenforced, and out-of-tree
# callers are unbounded. The parameter is therefore validated HERE,
# at entry: after the SF-BL-10971 trade-type check (its precondition,
# which dereferences nothing on the parameter) and BEFORE the
# SF-BL-9541 field validations, the SKIP / zero-invested early
# return, and any branch selection — so no truthiness test ever reads
# an unvalidated flag, the raise names the parameter itself, and a
# raising call leaves every field of the trade (net_pnl_bid_marked
# included) untouched (SF-BL-10724 ordering preserved and extended).
# Domain: strict bool only, via _is_strict_bool (SF-BL-10793 /
# SF-BL-11536 discipline). Prereg decision on 0/1-as-int: EXCLUDED —
# int 1/0 (and float 1.0/0.0) raise, they are not accepted aliases
# (bool subclasses int, not the reverse). Genuine True/False calls
# are bit-identical.

def _validate_is_settled_argument(value: object) -> None:
    """Raise ValueError unless the settle_or_mark_trade is_settled
    argument is a strict bool (SF-BL-10822)."""
    if not _is_strict_bool(value):
        raise ValueError(
            f"INVALID_IS_SETTLED: is_settled {value!r} "
            f"(type {type(value).__name__}) is not a bool "
            "(SF-BL-10822; strict-bool domain shared with "
            "SF-BL-11536): settle_or_mark_trade's is_settled "
            "parameter must be a genuine bool True / False — its "
            "branch test is a truthiness test, so a non-bool such "
            "as the string 'False' from a JSON/CSV producer is "
            "truthy and would settle an open trade at shares x "
            "$1.00 (prereg fixture: +$1,351.49 settled win) instead "
            "of marking it at the liquidation bid (-$68.26), while "
            "falsy non-bools (0, '', None) would masquerade as an "
            "open mark; int 1/0 are not bools and are excluded, "
            "never accepted as aliases — convert serialized "
            "values to a real bool before calling")


def settle_or_mark_trade(
    trade: TradeExecutionResult,
    is_settled: bool,
    outcome_yes: Optional[int] = None,
    exit_book: Optional[OrderBookSnapshot] = None,
    cluster_exposure_used: Optional[Dict[str, float]] = None,
) -> TradeExecutionResult:
    """
    Computes net PnL for a trade:
      - If settled (outcome_yes in {0, 1}): winning shares pay $1.00, losing shares pay $0.00.
      - If still open (is_settled=False): values shares strictly at current liquidation BID price
        (Prediction Arena Section 5.3.1), immediately accounting for bid-ask spread cost.
        When `exit_book` is supplied (SF-BL-3660), the mark is the
        bid-ladder VWAP liquidation mark for exactly shares_acquired
        (see bid_ladder_liquidation_mark): trade.liquidation_bid_price
        is replaced by that per-share mark and the filled-portion VWAP
        / fill fraction are recorded in trade.liquidation_mark_vwap /
        trade.liquidation_fill_fraction. Without `exit_book` the legacy
        single top-bid mark (shares * liquidation_bid_price) applies
        unchanged — a bare trade record carries no ladder to walk.

    Raises ValueError at entry, before any attribute of the argument
    is dereferenced, if trade is not a TradeExecutionResult
    (SF-BL-10971) — None, a dict, or any other non-trade object raises
    this named error instead of a bare AttributeError on
    shares_acquired at the SF-BL-9541 check below.

    Raises ValueError at entry, before any mutation, if shares_acquired
    or invested_capital is outside its finite >= 0 domain (SF-BL-9541) —
    in that case the trade's net_pnl_bid_marked is left untouched.

    Raises the same kind of ValueError at entry if the trade's action
    is not a genuine TradeAction member (SF-BL-10823): a non-canonical
    action such as 'buy_yes', 'MAYBE' or None can neither settle nor
    be marked, on either branch.

    Raises the same ValueError for a settled trade whose outcome_yes is
    not a strict int in {0, 1} (SF-BL-9477): bools True / False raise
    exactly as 0.5 does, instead of settling as outcome 1 / 0.

    SF-BL-8890: on a successful SETTLEMENT (is_settled=True) the trade's
    cluster-cap fraction is released via release_cluster_exposure —
    from `cluster_exposure_used` when given, else from the ledger
    evaluate stored on the trade. Open marks (is_settled=False) never
    release: the position is still at risk. A settlement that raises
    releases nothing.

    SF-BL-34868: the given `cluster_exposure_used` may only REPEAT the
    ledger evaluate stored on the trade, never replace it: when the
    trade stores an entry ledger and the argument is a different dict,
    the settled branch raises the named MISMATCHED_CLUSTER_LEDGER
    ValueError at its head — before the outcome validation and before
    any field is written — leaving the trade and both ledgers
    byte-identical (releasing into the wrong dict would erase an
    unrelated position's exposure there and strand the entry ledger's
    charge permanently). A trade with no stored ledger (manually
    constructed, or a SKIP record) accepts an explicit dict exactly as
    before, and open marks never read the argument.

    SF-BL-34952: the release frees the cluster key recorded at entry
    (entry_factor_cluster), not the record's current factor_cluster —
    renaming factor_cluster between evaluate and settle releases the
    entry key and writes no stray key (see release_cluster_exposure).

    SF-BL-10724: validation completes before any field of the trade is
    written, on both branches — a raising settle/mark leaves
    is_settled, outcome_yes and net_pnl_bid_marked byte-identical to
    their pre-call values (see the SF-BL-10724 block above this
    function). Pre-fix the is_settled/outcome_yes assignments ran at
    entry, so a failed re-mark flipped a settled trade open (and a
    failed settle flipped an open trade settled) while the previous
    branch's PnL survived into the fleet scorecard.

    SF-BL-10767: bid freshness provenance. An exit-book re-mark
    re-stamps trade.liquidation_bid_as_of / liquidation_bid_source
    from the exit book (source "exit_book"), written together with
    the other mark fields AFTER the ladder walk and its re-validation
    (SF-BL-10724 ordering preserved — _snapshot_ts cannot raise, it
    reads an invalid stamp as None). Without `exit_book` the stored
    bid is re-used with its stamps untouched, so the mark keeps its
    entry provenance and its age at scoring stays visible; the
    settled branch never touches the stamps (a settled PnL pays
    $1/$0 and does not read the bid).

    SF-BL-11536: the stored is_settled field is strict-bool validated
    on the record itself; this function stores its `is_settled`
    argument through that assignment guard, and is_settled is the
    first field written on every path below, so a non-bool argument
    raises the named INVALID_IS_SETTLED ValueError at the assignment
    with every other field untouched.

    SF-BL-35035: the trade's stored cluster_exposure_released flag
    is strict-bool validated on the record itself (construction /
    assignment / monotonic latch) and pre-validated at the head of
    the settled branch above, before any field is written, so a
    bypass-constructed non-bool flag raises the named
    INVALID_CLUSTER_EXPOSURE_RELEASED ValueError with the trade
    and the ledger byte-identical; a post-release reset (normal or
    bypass) can never produce a second release (see the
    SF-BL-35035 block at the dataclass).

    SF-BL-35212: the cluster ledger STORED on the trade is validated
    at the head of the settled branch (see _validate_stored_
    cluster_ledger and the SF-BL-35212 block above
    release_cluster_exposure): a non-dict stored reference raises
    INVALID_STORED_CLUSTER_LEDGER and a stored/effective dict that
    is not the entry_cluster_ledger snapshot raises
    SUBSTITUTED_CLUSTER_LEDGER — both before the outcome validation
    and before any field is written, so a rejected settle books no
    PnL (pre-fix the list case booked +$658.42 and only then crashed
    inside the release, and the substituted-dict case silently
    released into the wrong dict). Open marks never release and
    never read the stored reference, so they are unchecked.

    SF-BL-10822: the `is_settled` PARAMETER is strict-bool validated
    at entry (see _validate_is_settled_argument and the SF-BL-10822
    block above this function) — after the SF-BL-10971 trade-type
    check and before the SF-BL-9541 field validations, the SKIP /
    zero-invested early return and any branch selection. Any
    non-bool (the strings 'False'/'True', int 1/0, float 1.0/0.0,
    None, containers) raises the named INVALID_IS_SETTLED ValueError
    naming SF-BL-10822 (and the shared SF-BL-11536 domain), leaving
    net_pnl_bid_marked and every other field untouched; genuine
    True/False calls are bit-identical. This supersedes the layered
    ordering SF-BL-11536 documented for 'False' with no outcome:
    that call now raises the is_settled error, not SF-BL-9477's
    outcome error, because the flag is validated before the branch
    that would validate the outcome is ever selected.

    SF-BL-34997: the trade's action must still match the entry side
    pinned at construction (entry_action — see the SF-BL-34997 block
    above TradeExecutionResult). The consistency check runs at
    entry, after the SF-BL-10823 action-member check and before the
    SKIP / zero-invested early return: a record flipped to another
    genuine member since entry (BUY_YES -> BUY_NO, BUY_NO ->
    BUY_YES, either -> SKIP, SKIP -> an executed side) raises the
    named MISMATCHED_TRADE_ACTION ValueError with every field
    untouched, instead of paying out as the opposite side
    (prereg fixture: +$1,004.95 booked on a YES position that lost
    $1,025.25), marking YES-built shares at the NO bid, or being
    silently erased by the SKIP early return. Unpinned records
    (entry_action None) and unflipped records are bit-identical.
    """
    # SF-BL-10971: validate the trade argument's TYPE first of all, at
    # entry — before any attribute of it is dereferenced. The SF-BL-9541
    # position-field check below reads trade.shares_acquired, so a
    # non-TradeExecutionResult argument (None / dict / fake) would crash
    # there with a bare AttributeError; this gate raises the named
    # ValueError instead, with nothing read or written.
    _validate_trade_argument(trade)

    # SF-BL-10822: validate the is_settled ARGUMENT's strict-bool type
    # next, still at entry — before the position-field validations
    # below, before the SKIP / zero-invested early return, and before
    # the `if is_settled:` branch test can read the flag by truthiness.
    # A raise here leaves every field of the trade untouched.
    _validate_is_settled_argument(is_settled)

    # SF-BL-9541: validate the position fields FIRST among the field
    # validations, at entry — before is_settled/outcome_yes are written,
    # before the SKIP / zero-invested early return below, and before
    # either the settled or the mark branch can turn shares into
    # gross_payout / liquidation_value.
    _validate_settle_position_fields(trade)

    # SF-BL-10823: validate the action immediately after the position
    # fields, still at entry — before the SKIP / zero-invested early
    # return (whose `action == SKIP` test a plain 'SKIP' string would
    # pass by str-Enum equality) and before either branch can read the
    # action. This is the consumption gate for records that bypassed
    # the dataclass's own construction/assignment validation
    # (object.__setattr__ / __dict__ / unpickling): no such record can
    # settle or be marked, and a raise here leaves every field of the
    # trade untouched (SF-BL-10724 ordering preserved).
    _validate_trade_action(trade.action)

    # SF-BL-34997: validate the action's CONSISTENCY with the entry
    # side pinned at construction, immediately after the SF-BL-10823
    # member check and still before the SKIP / zero-invested early
    # return — a BUY_YES record flipped to BUY_NO would otherwise pay
    # out as the opposite side in the settled branch (prereg fixture:
    # +$1,004.95 on a position that lost $1,025.25), a record flipped
    # to SKIP would be silently erased by the early return below, and
    # a flipped open mark would price YES-built shares at the NO bid
    # (or walk the NO ladder on the exit-book path). Unscoped: settle
    # reads the side of every record it touches. A raise here leaves
    # every field of the trade untouched (SF-BL-10724 ordering
    # preserved); unpinned records (entry_action None — bypass-
    # constructed / legacy) pass exactly as pre-fix.
    _validate_trade_action_consistency(trade)

    if trade.action == TradeAction.SKIP or trade.invested_capital <= 0.0:
        # No branch validation is outstanding for a SKIP / zero-invested
        # trade (SF-BL-9477/9436 ordering, preserved): this return is
        # not a raising path, so writing the three fields here cannot
        # strand a flipped state.
        trade.is_settled = is_settled
        trade.outcome_yes = outcome_yes
        trade.net_pnl_bid_marked = 0.0
        return trade

    if is_settled:
        # SF-BL-34868: ledger-identity pre-validation — when the trade
        # stores an entry ledger and the explicit cluster_exposure_used
        # is a DIFFERENT dict, raise the named MISMATCHED_CLUSTER_LEDGER
        # ValueError here, before the outcome validation below and
        # before any field of the trade is written, so the rejected
        # settle leaves the trade and both ledgers byte-identical and
        # the correct release path remains available (release_cluster_
        # exposure re-validates the same condition at its own entry).
        _validate_cluster_ledger_match(trade, cluster_exposure_used)
        # SF-BL-35035: release-flag pre-validation — a record that
        # bypassed the dataclass guard with a non-bool
        # cluster_exposure_released raises the named
        # INVALID_CLUSTER_EXPOSURE_RELEASED ValueError here, before
        # any field is written, instead of booking the settlement
        # and only then having its release silently blocked (or
        # re-armed) by the flag's truthiness inside
        # release_cluster_exposure.
        _validate_trade_cluster_exposure_released(trade)
        # SF-BL-35212: stored-ledger pre-validation — when the trade's
        # stored cluster reference is not a dict, or is not the dict
        # evaluate charged (entry_cluster_ledger snapshot), raise the
        # named INVALID_STORED_CLUSTER_LEDGER / SUBSTITUTED_CLUSTER_
        # LEDGER ValueError here, before the outcome validation below
        # and before any field of the trade is written, so the
        # rejected settle books no PnL and leaves the trade and every
        # ledger byte-identical (release_cluster_exposure re-validates
        # the same conditions at its own entry).
        _validate_stored_cluster_ledger(trade, cluster_exposure_used)
        # SF-BL-9477: strict-int domain check — the pre-fix equality
        # membership test `outcome_yes not in (0, 1)` let True / False
        # (and 1.0 / 0.0, np.bool_) settle as genuine outcomes.
        # SF-BL-10724: this validation now runs BEFORE is_settled /
        # outcome_yes are written, and the PnL is computed into a local
        # first — a raise here leaves the trade byte-identical.
        _validate_outcome_yes(outcome_yes)
        won = (
            (trade.action == TradeAction.BUY_YES and outcome_yes == 1)
            or (trade.action == TradeAction.BUY_NO and outcome_yes == 0)
        )
        gross_payout = trade.shares_acquired * 1.0 if won else 0.0
        new_pnl = gross_payout - trade.invested_capital
        trade.is_settled = is_settled
        trade.outcome_yes = outcome_yes
        trade.net_pnl_bid_marked = new_pnl
        # SF-BL-8890: settlement closes the position, so its cluster-cap
        # fraction is freed for the next entry (mirrors nstar.py
        # Account.settle decrementing cluster_stakes). Placed after the
        # PnL is booked and after the outcome validation above, so a
        # settlement that raised has released nothing.
        release_cluster_exposure(trade, cluster_exposure_used)
    else:
        # SF-BL-9436: validate the bid before it becomes PnL. The settled
        # branch above never reads liquidation_bid_price and is unchanged;
        # only this unsettled mark is bounded. Raises ValueError on any
        # bid outside [0, 1] (including NaN/inf/non-numeric), so no
        # out-of-bound mark can reach CWR / MDD in evaluate_fleet_north_star.
        # SF-BL-10724: this validation — and, on the exit-book path, the
        # ladder walk and the walked mark's re-validation — now all run
        # BEFORE is_settled / outcome_yes (or any mark field) are
        # written, so a raise here leaves the trade byte-identical.
        _validate_liquidation_bid(trade.liquidation_bid_price)
        if exit_book is not None:
            # SF-BL-3660: bid-side sweep scaled to shares_acquired. The
            # walked mark is in [0, 1] by construction (filled VWAP over
            # tradeably-priced live levels, times a fill fraction in
            # [0, 1]) and is re-validated before it becomes PnL, like
            # the legacy bid above. The stored liquidation_bid_price
            # becomes the mark actually used, so downstream readers
            # (CWR / MDD in evaluate_fleet_north_star) see exit depth.
            mark, filled_vwap, fill_fraction, proceeds = (
                bid_ladder_liquidation_mark(
                    exit_book, trade.action, trade.shares_acquired)
            )
            _validate_liquidation_bid(mark)
            # SF-BL-10767: the walked mark was observed in the exit
            # book, so its provenance is that book's snapshot time —
            # computed here, before any write (SF-BL-10724 ordering);
            # _snapshot_ts cannot raise (an invalid stamp reads None).
            new_bid_as_of = _snapshot_ts(exit_book)
            trade.is_settled = is_settled
            trade.outcome_yes = outcome_yes
            trade.liquidation_bid_price = mark
            trade.liquidation_mark_vwap = filled_vwap
            trade.liquidation_fill_fraction = fill_fraction
            trade.liquidation_bid_as_of = new_bid_as_of
            trade.liquidation_bid_source = BID_SOURCE_EXIT_BOOK
            liquidation_value = proceeds
        else:
            # Conservative mark-to-market using immediate liquidation bid price
            liquidation_value = trade.shares_acquired * trade.liquidation_bid_price
            trade.is_settled = is_settled
            trade.outcome_yes = outcome_yes
        trade.net_pnl_bid_marked = liquidation_value - trade.invested_capital

    return trade


# --- SF-BL-9139: out-of-range forecasts / outcomes silently mis-binned --
# compute_empirical_ece_and_brier binned with
#     idx = min(num_bins - 1, int(p * num_bins))
# and NO range check on p or o, so a malformed row corrupted the ECE that
# evaluate_fleet_north_star exponentiates into its calibration
# multiplier instead of raising (all verified live pre-fix,
# reproduce_hazard.py in the task work dir):
#   p=-0.20 -> int(-2.0) = -2, bins[-2] is bin 8 ([0.8, 0.9)):
#              returned ECE 1.2000, Brier 1.4400 for the row (-0.2, 1)
#   p=-0.10 / -0.15 -> int() = -1, bins[-1] is the TOP bin 9
#   p=-0.05 -> int(-0.5) = 0 (truncation toward zero): bin 0 with a
#              negative avg_p, returned ECE 0.0500
#   p= 1.20 -> min() clamps idx to bin 9, returned ECE 0.2000; through
#              evaluate_fleet_north_star the prereg fixture scored
#              N_star 0.3068 vs the 0.1129 control [(0.7, 1)] -- a
#              malformed row IMPROVED the quoted fleet score ~2.7x
#   o= 2    -> avg_o = 2 in bin 7, returned ECE 1.3000, Brier 1.6900
#   combined prereg panel returned ECE 0.6100, Brier 0.6525, no error
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT met.
# The only in-repo caller, evaluate_fleet_north_star, validates panel
# element TYPES / finiteness (SF-BL-11535) but not ranges -- finite
# -0.2 / 1.2 / o=2 all pass that gate and reached this function live.
# The benchmark producer clips its forecasts to [0.02, 0.98] via
# aggregate_fleet_forecasts, but its outcomes are bare unchecked
# literals, panels are plain mutable lists built by callers (fleet
# wave2/wave3 scripts feed caller-supplied panels straight in), the
# List[Tuple[float, int]] annotation is unenforced, and no schema
# layer exists. The domain is therefore enforced here, at the
# computation gate, before any Brier sum or binning: a forecast must
# be a finite real in [0, 1] and an outcome a finite real equal to
# 0 or 1 (both via _is_finite_number, SF-BL-9438, so bool / string /
# None / NaN / +-inf elements also raise this named error instead of
# a bare TypeError from the Brier subtraction or an unnamed error
# from int()). The empty-input (0.0, 0.0) convention is unchanged:
# validation runs after the empty check and is a no-op on it, and
# every in-range panel (boundaries 0.0 / 1.0 included) is untouched.

# --- SF-BL-9711: the outcome argument was never validated on its own --
# Filed against compute_empirical_ece_and_brier's OUTCOME argument
# specifically: SF-BL-9139 screens the forecast argument's range and
# SF-BL-9578 its num_bins argument; this task's prereg owns the outcome
# domain — ValueError unless every outcome is exactly 0 or 1 (bool
# excluded) and finite. Verified live at filing (repro.py in the task
# work dir; the pre-guard algorithm reproduced alongside): the panel
# [(0.9, 2), (0.1, 0)] returned (ECE 0.60, Brier 0.61), [(0.9, -1),
# (0.1, 0)] returned (ECE 1.00, Brier 1.81) and a NaN-outcome panel
# returned (NaN, NaN) as an ordinary result — ECE and Brier for binary
# resolutions both lie in [0, 1], so every one of those values is
# impossible, yet evaluate_fleet_north_star fed them straight into
# ece_10bin, the Brier diagnostic and the exp(-10 * ECE) calibration
# multiplier. SF-BL-9139's gate (landed after this task was filed)
# already rejects every PLAIN-typed bad outcome via _is_finite_number.
# The residual gap this task closes, verified live pre-fix in
# repro.py: _is_finite_number tests isinstance(v, (int, float)), so a
# numpy-typed outcome that IS exactly 0 or 1 — np.int64(1), np.int32(0),
# np.float32(1.0) — was REJECTED as INVALID_OUTCOME (np.float64 passes
# only because it subclasses float). The fleet is numpy-native
# (agents.py panels, predmarkets.py int64 outcome Series), so a clean
# book from such a producer errored instead of scoring byte-identical
# to its Python-typed twin — a false rejection of a valid binary
# resolution is the same class of falsified calibration grade as a
# false acceptance, in the opposite direction. The helper below
# therefore defines the outcome domain over numbers.Real (Python int /
# float and every numpy integer / floating scalar), with bool and
# np.bool_ excluded explicitly and by type name, so the exclusion does
# not depend on a numpy version's ABC registration (True == 1 would
# otherwise pass the membership test — the SF-BL-9477 hazard), and
# finiteness via math.isfinite. Kill-criterion audit
# (PRODUCER_AUDIT.md in the task work dir): NOT met — no validated
# settlement path builds this panel (the benchmark appends bare dict
# literals to resolved_forecasts independently of settle_or_mark_trade,
# which never writes the panel), so the guard stays here, at the
# consumption gate. Plain-type behaviour is unchanged: every outcome
# SF-BL-9139 rejected still raises the same INVALID_OUTCOME ValueError,
# and every panel it accepted returns byte-identical (ece, brier).

def _outcome_in_ece_domain(o: object) -> bool:
    """True iff o is exactly 0 or 1, finite, bool excluded (SF-BL-9711).

    Accepts any real numeric type — Python int/float and numpy integer /
    floating scalars (numbers.Real) — so a validated binary outcome
    keeps its value across producer type clothing. Rejects bool and
    np.bool_ (type name check covers numpy's bool scalar without
    importing numpy here), non-numeric values, NaN / +-inf, and every
    finite real other than 0 or 1 (2, -1, 0.5, ...).
    """
    if isinstance(o, bool) or type(o).__name__ == "bool_":
        return False
    if not isinstance(o, numbers.Real):
        return False
    try:
        if not math.isfinite(o):  # type: ignore[arg-type]
            return False
    except (TypeError, OverflowError):
        return False
    return o in (0, 1)


# --- SF-BL-9820: a non-finite forecast crashed binning unnamed -------
# Filed against compute_empirical_ece_and_brier's FORECAST argument
# specifically: SF-BL-9711 covers the outcome argument, SF-BL-9578
# num_bins and SF-BL-9139 finite out-of-range forecasts; none owned the
# non-finite forecast. Verified with the pre-guard algorithm
# (reproduce_hazard.py in the task work dir): [(NaN, 1), (0.5, 0)]
# raised a bare ValueError 'cannot convert float NaN to integer' and
# [(+inf, 1), (0.5, 0)] an OverflowError 'cannot convert float
# infinity to integer', both inside int(p * num_bins) — the corrupt
# forecast took down evaluate_fleet_north_star's ECE/Brier readout
# instead of being rejected by name. Honest history: by the time this
# task ran, SF-BL-9139's gate (landed after filing) already rejected
# PLAIN float NaN / +-inf, as OUT_OF_RANGE_FORECAST — a range label
# whose text explains truncation/wrap/clamp, none of which is what a
# non-finite forecast does (it crashed int(), it never wrapped). Two
# residual gaps remained live, both verified pre-fix:
#   (1) No dedicated non-finite identity: the prereg requires a named
#       ValueError IDENTIFYING the forecast as non-finite. The branch
#       below now raises NON_FINITE_FORECAST (SF-BL-9820) for a
#       NaN / +-inf forecast, before any Brier sum or binning, on the
#       direct surface; finite out-of-range forecasts keep the exact
#       SF-BL-9139 OUT_OF_RANGE_FORECAST error and every finite panel
#       returns byte-identical (ece, brier).
#   (2) A forecast magnitude beyond float range (10**400) crashed the
#       VALIDATOR itself: math.isfinite inside _is_finite_number
#       raised a bare OverflowError 'int too large to convert to
#       float' on BOTH surfaces — directly, and through the fleet's
#       SF-BL-11535 pre-screen, which therefore does not in fact
#       screen every forecast. _is_finite_number is now
#       OverflowError-safe (False for unrepresentable magnitudes),
#       and this validator tri-states a huge int as finite-but-out-
#       of-range (exact int/float comparison needs no conversion),
#       so it raises the named OUT_OF_RANGE_FORECAST error.
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT
# met — evaluate_fleet_north_star's SF-BL-11535 screen covers only
# panels routed through the scorer (direct callers of this function
# have no screen at all) and the screen itself leaked the bare
# OverflowError in gap (2), so it does not pre-screen EVERY resolved
# forecast for finiteness. The guard therefore stays here, at the
# computation gate.

# --- SF-BL-11010: a bool forecast scores as a certain one --------------
# Filed against compute_empirical_ece_and_brier's FORECAST element TYPE
# specifically: SF-BL-9820 owns non-finite forecasts, SF-BL-9139 finite
# out-of-range forecasts, and SF-BL-9711 validates only the OUTCOME
# element (bool excluded there, not here). At filing the pair (True, 1)
# was binned and scored exactly as p=1.0 — the panel [(True, 1),
# (0.5, 0)] returned (ECE 0.25, Brier 0.125), bit-identical to
# [(1.0, 1), (0.5, 0)] (repro.py Part A in the task work dir re-derives
# both from the pre-guard algorithm) — so a bool leaking from a JSON
# producer (JSON true/false deserialise to Python bool) became a
# max-confidence forecast in the fleet's ECE term. Honest history: by
# the time this task ran, SF-BL-9139's gate already rejected PLAIN
# bool / non-numeric forecasts, but only as OUT_OF_RANGE_FORECAST with
# text explaining truncation / wrap / clamp — none of which a bool
# does (True == 1 and False == 0 are both numerically IN [0, 1]; the
# defect is the TYPE, not the range) — and with no SF-BL-11010
# identity. Two residual gaps remained live, both verified pre-fix
# (repro_prefix.txt in the task work dir):
#   (1) No dedicated bool/non-numeric identity: the prereg requires a
#       named ValueError for any forecast p that is a bool or
#       non-numeric. The type branch below now raises
#       BOOL_OR_NON_NUMERIC_FORECAST (SF-BL-11010) — carrying the
#       OUT_OF_RANGE_FORECAST umbrella token and the SF-BL-9139
#       citation the neighbouring suites pin — naming the value AND
#       its type, before any Brier sum or binning. Finite out-of-range
#       forecasts keep the exact SF-BL-9139 error, non-finite ones
#       the exact SF-BL-9820 error, and every genuine numeric panel
#       returns byte-identical (ece, brier).
#   (2) The gate's isinstance(p, (int, float)) test falsely rejected
#       genuine NUMERIC forecasts in numpy clothing — np.float32(0.5)
#       and np.int64(1) raised OUT_OF_RANGE_FORECAST — while np.bool_
#       was not identified as a bool at all. The forecast type domain
#       therefore mirrors the SF-BL-9711 outcome domain exactly:
#       numbers.Real (Python int/float and every numpy integer /
#       floating scalar), with bool and np.bool_ excluded explicitly
#       and by type name, so a validated numeric forecast keeps its
#       value across producer type clothing and a bool is rejected as
#       a bool in any clothing. (Decimal is not a numbers.Real and
#       stays rejected, as pre-fix; complex is not Real either.)
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT
# met — the only in-repo production caller is evaluate_fleet_north_star,
# whose SF-BL-11535 screen covers only scorer-routed panels; direct
# callers pass bare literals with no screen, the annotation is
# unenforced, panels are plain mutable lists, and no schema layer
# stands between a JSON producer's true/false and this function. The
# type itself is therefore validated here, at the computation gate.

# --- SF-BL-11048: a non-numeric forecast inside a well-shaped pair ----
# Filed against compute_empirical_ece_and_brier's FORECAST element TYPE
# specifically, on the non-numeric (non-bool) dimension: SF-BL-10912
# owns container / entry shape, SF-BL-11010 owns bool forecasts, and
# SF-BL-9820 owns non-finite forecasts. Verified at filing with the
# pre-guard algorithm (repro.py Part A in the task work dir): the Brier
# sum (p - o) ** 2 runs before any binning, so [('0.6', 1)] and
# [(None, 0)] — correctly shaped 2-tuples — raised a bare TypeError
# 'unsupported operand type(s) for -' from that subtraction, while the
# clean [(0.6, 1)] returned (0.4, 0.16). Honest history: by this run
# SF-BL-11010's type branch (landed after filing) already rejected
# plain string / None forecasts, but only under the BOOL_OR_NON_
# NUMERIC_FORECAST identity framed around a bool leaking from a JSON
# producer, with no SF-BL-11048 identity and no distinction between a
# bool (which IS numeric-valued) and a value with no numeric value at
# all — a string that merely LOOKS numeric ('0.6') is not a forecast,
# and None is the absence of one. The residual gap closed here:
# non-bool non-numeric forecasts now raise the dedicated
# NON_NUMERIC_FORECAST ValueError (SF-BL-11048), naming the value, its
# type and the row index, before any Brier sum or binning. The message
# carries the BOOL_OR_NON_NUMERIC_FORECAST and OUT_OF_RANGE_FORECAST
# umbrella tokens and the SF-BL-11010 / SF-BL-9139 citations the
# neighbouring suites pin, so those suites are unchanged; bool
# forecasts keep the exact SF-BL-11010 error, finite out-of-range
# forecasts keep SF-BL-9139's, non-finite ones SF-BL-9820's, and every
# valid panel returns bit-identical (ece, brier).
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT
# met — the only in-repo production caller, evaluate_fleet_north_star,
# routes through its SF-BL-11535 scorer screen, but direct callers of
# this function have no screen at all, the List[Tuple[float, int]]
# annotation is unenforced, panels are plain mutable lists built from
# bare literals, and no schema layer guarantees forecasts arrive only
# through scorer-validated resolved sets. The type is therefore
# validated here, at the computation gate.

def _forecast_is_bool(p: object) -> bool:
    """True iff p is a bool in any clothing (incl. np.bool_)."""
    return isinstance(p, bool) or type(p).__name__ == "bool_"


def _forecast_is_bool_or_non_numeric(p: object) -> bool:
    """True iff p is a bool (incl. np.bool_) or not a real number.

    The forecast type domain (SF-BL-11010) mirrors the SF-BL-9711
    outcome domain: any real numeric type — Python int/float and numpy
    integer / floating scalars (numbers.Real) — is numeric; bool and
    np.bool_ are excluded explicitly and by type name, so the
    exclusion does not depend on a numpy version's ABC registration
    (True == 1 would otherwise pass every numeric test, the
    SF-BL-9477 hazard class).
    """
    if isinstance(p, bool) or type(p).__name__ == "bool_":
        return True
    return not isinstance(p, numbers.Real)


def _validate_ece_panel_ranges(
    predictions_and_outcomes: List[Tuple[float, int]],
) -> None:
    """Raise ValueError on any out-of-domain (forecast, outcome) row.

    Forecasts must lie in [0, 1]; outcomes must be exactly 0 or 1
    (SF-BL-9139; outcome domain per SF-BL-9711 via
    _outcome_in_ece_domain). A bool forecast (bool / np.bool_ in any
    clothing) raises the dedicated BOOL_OR_NON_NUMERIC_FORECAST
    ValueError (SF-BL-11010, via _forecast_is_bool_or_non_numeric)
    before the finiteness and range branches; a non-bool non-numeric
    forecast (string, None, complex, list, dict, ...) raises the
    dedicated NON_NUMERIC_FORECAST ValueError (SF-BL-11048) at the
    same point. A non-finite forecast (NaN / +-inf) raises the
    dedicated NON_FINITE_FORECAST ValueError (SF-BL-9820) before
    any binning. An entry that is not a (forecast, outcome) pair also
    raises the named ValueError rather than leaking a bare unpack error.
    """
    for idx, entry in enumerate(predictions_and_outcomes):
        try:
            p, o = entry
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"OUT_OF_RANGE_FORECAST: compute_empirical_ece_and_brier "
                f"predictions_and_outcomes[{idx}]={entry!r} is not a "
                "(forecast, outcome) pair (SF-BL-9139): a malformed row "
                "must be rejected by name, not mis-binned or crashed on"
            ) from exc
        if _forecast_is_bool_or_non_numeric(p):
            if not _forecast_is_bool(p):
                # SF-BL-11048: the NON-NUMERIC branch — a forecast with
                # no numeric value at all (a string that merely looks
                # numeric, None, complex, a container, ...) inside a
                # correctly shaped pair. Pre-guard it died with a bare
                # TypeError from the Brier subtraction (p - o) ** 2,
                # which runs before any binning. The message carries
                # the BOOL_OR_NON_NUMERIC_FORECAST and
                # OUT_OF_RANGE_FORECAST umbrella tokens and the
                # SF-BL-11010 / SF-BL-9139 citations the neighbouring
                # suites pin.
                raise ValueError(
                    f"NON_NUMERIC_FORECAST: "
                    f"compute_empirical_ece_and_brier "
                    f"predictions_and_outcomes[{idx}] has forecast={p!r} "
                    f"(type {type(p).__name__}), which is a non-numeric "
                    "value, not a real-number forecast (SF-BL-11048; "
                    "BOOL_OR_NON_NUMERIC_FORECAST / OUT_OF_RANGE_FORECAST "
                    "type gate from SF-BL-11010 / SF-BL-9139): a string "
                    "that merely looks numeric ('0.6') is not a forecast "
                    "and None is the absence of one — neither may be "
                    "coerced or scored, and pre-guard both died with a "
                    "bare TypeError from the Brier subtraction "
                    "(p - o) ** 2 inside compute_empirical_ece_and_brier, "
                    "crashing the ECE/Brier readout that "
                    "evaluate_fleet_north_star exponentiates instead of "
                    "being rejected by name")
            # SF-BL-11010: the BOOL branch — a bool forecast gets its
            # own named error, distinct from the finite out-of-range
            # branch below: True == 1 / False == 0 are numerically IN
            # [0, 1], so a bool never wraps or clamps — pre-guard it
            # was silently read as certainty (p=1.0) or impossibility
            # (p=0.0). The message carries the OUT_OF_RANGE_FORECAST
            # umbrella token and the SF-BL-9139 citation the
            # neighbouring suites pin.
            raise ValueError(
                f"BOOL_OR_NON_NUMERIC_FORECAST: "
                f"compute_empirical_ece_and_brier "
                f"predictions_and_outcomes[{idx}] has forecast={p!r} "
                f"(type {type(p).__name__}), which is a bool or "
                "non-numeric value, not a real-number forecast "
                "(SF-BL-11010; OUT_OF_RANGE_FORECAST type gate from "
                "SF-BL-9139): bool is an int subclass (True == 1, "
                "False == 0), so a bool leaking from a JSON producer "
                "would be binned and scored exactly as a certain "
                "p=1.0 / impossible p=0.0 forecast, and a non-numeric "
                "value would die unnamed in the Brier subtraction — "
                "either way the ECE that evaluate_fleet_north_star "
                "exponentiates is corrupted or the readout crashes")
        # SF-BL-9820: non-finite forecasts get their own named error,
        # distinct from the finite out-of-range branch below — NaN /
        # +-inf never wrap or clamp, they crash int(p * num_bins)
        # (bare ValueError for NaN, OverflowError for +-inf). A huge
        # int makes math.isfinite raise OverflowError on conversion;
        # it is mathematically finite, so treat it as finite here and
        # let the exact range comparison below reject it by name.
        try:
            _p_finite = math.isfinite(p)
        except OverflowError:
            _p_finite = True
        if not _p_finite:
            raise ValueError(
                f"NON_FINITE_FORECAST: compute_empirical_ece_and_brier "
                f"predictions_and_outcomes[{idx}] has forecast={p!r}, "
                "which is not finite (SF-BL-9820; finiteness gate "
                "from SF-BL-9139): a NaN forecast raised a bare "
                "ValueError 'cannot convert float NaN to integer' "
                "and an infinite forecast an OverflowError inside "
                "int(p * num_bins) binning, taking down "
                "evaluate_fleet_north_star's ECE/Brier readout "
                "instead of rejecting the corrupt forecast by name")
        if not 0.0 <= p <= 1.0:  # type: ignore[operator]
            raise ValueError(
                f"OUT_OF_RANGE_FORECAST: compute_empirical_ece_and_brier "
                f"predictions_and_outcomes[{idx}] has forecast={p!r}, "
                "which is not a finite real number in [0, 1] "
                "(SF-BL-9139): int(p * num_bins) truncation plus "
                "negative list indexing would silently place it in the "
                "wrong bin (p <= -0.10 wraps to the top bins, p > 1 "
                "clamps to the top bin) and corrupt the ECE that "
                "evaluate_fleet_north_star exponentiates")
        if not _outcome_in_ece_domain(o):
            raise ValueError(
                f"INVALID_OUTCOME: compute_empirical_ece_and_brier "
                f"predictions_and_outcomes[{idx}] has outcome={o!r}, "
                "which is not 0 or 1 (SF-BL-9139 / SF-BL-9711): a "
                "non-binary outcome would be averaged into a bin as "
                "if it were a realised frequency and corrupt both ECE "
                "and Brier")


def compute_empirical_ece_and_brier(
    predictions_and_outcomes: List[Tuple[float, int]],
    num_bins: int = 10,
) -> Tuple[float, float]:
    """
    Computes 10-bin Expected Calibration Error (ECE) and Brier Score
    (Prophet Arena Section 3.1.1 & Appendix B.3).

    Raises ValueError if any forecast lies outside [0, 1] or any
    outcome is not 0 or 1 (SF-BL-9139 range gate, applied before any
    binning or Brier summation; the outcome check is SF-BL-9711's:
    exactly 0 or 1, bool excluded, finite, in any real numeric type).
    A non-finite forecast (NaN / +-inf) raises the dedicated
    NON_FINITE_FORECAST ValueError (SF-BL-9820), also before binning.
    A bool forecast raises the dedicated
    BOOL_OR_NON_NUMERIC_FORECAST ValueError (SF-BL-11010), also
    before binning; a non-bool non-numeric forecast (string, None,
    complex, ...) raises the dedicated NON_NUMERIC_FORECAST
    ValueError (SF-BL-11048), also before binning; genuine numeric
    forecasts in any real numeric type (Python int/float, numpy
    integer/floating scalars) score identically to their
    Python-typed twins.
    """
    if not predictions_and_outcomes:
        return 0.0, 0.0

    # SF-BL-9139: range/domain gate FIRST, before the Brier sum and the
    # truncating bin index below can silently absorb an out-of-range row.
    _validate_ece_panel_ranges(predictions_and_outcomes)

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
    # SF-BL-10767: bid-mark freshness accounting. stale_mark_count /
    # stale_mark_invested are the unsettled executed trades EXCLUDED
    # from every sum above because their bid mark was stale or
    # unprovenanced at scored_at (both 0 whenever scored_at is None —
    # legacy scoring excludes nothing). unprovenanced_mark_count is
    # the unsettled executed trades whose bid carries no valid as-of,
    # counted whether or not scored_at was given (a subset of the
    # stale set when it was). Defaults keep direct constructions of
    # the scorecard working unchanged.
    stale_mark_count: int = 0
    stale_mark_invested: float = 0.0
    unprovenanced_mark_count: int = 0
    # SF-BL-34866: leverage accounting. executed_invested_capital is
    # the summed invested_capital of the executed set (action != SKIP
    # and invested > 0); leverage_ratio is that sum divided by
    # starting_capital; is_levered is True iff the ratio is strictly
    # above 1 (the book deployed more capital than its capital base).
    # Defaults (False / 0.0 / 0.0) keep direct constructions of the
    # scorecard working unchanged, as with the SF-BL-10767 fields.
    is_levered: bool = False
    leverage_ratio: float = 0.0
    executed_invested_capital: float = 0.0
    # SF-BL-35210: settled-PnL exact-payout flag. settled_pnl_mismatch_
    # count is the number of settled executed trades (valid outcome)
    # whose net_pnl_bid_marked differs from the exact settlement
    # payout (shares - invested on a win, -invested on a loss) by more
    # than SETTLED_PNL_EPSILON (half a cent); settled_pnl_mismatch_
    # invested is their summed invested_capital. Flag-only (prereg
    # "raises or flags"; flag branch per SF-BL-34866 precedent): the
    # trades still score, so every pre-existing field is bit-identical
    # and settle-produced books report 0 / 0.0. Defaults keep direct
    # constructions of the scorecard working unchanged.
    settled_pnl_mismatch_count: int = 0
    settled_pnl_mismatch_invested: float = 0.0


# --- SF-BL-10968: trades container / entry types never checked ---------
# SF-BL-9475 / SF-BL-10825 validate the field VALUES of
# TradeExecutionResult entries, and SF-BL-10823 validates each entry's
# action, but the trades container and the entry types themselves were
# never checked, so a malformed argument aborted N_star scoring with a
# bare crash before any validation could name the problem. Verified
# live pre-fix (repro.py in the task work dir; honest control: one
# BUY_YES trade, $500 invested, PnL +$100, resolved [(0.6, 1), (0.6, 0)]
# -> N_star 0.4445):
#   trades=None        -> TypeError: 'NoneType' object is not iterable
#                         (from _validate_trade_actions' `for t in trades`)
#   trades=[{...}]     -> AttributeError: 'dict' object has no attribute
#                         'action' — a JSON-style dict entry, the shape
#                         stored trade records materialise as
#                         (asdict + JSON round-trip), is the prereg case
#   trades=123 / 5.0 / a bare TradeExecutionResult -> bare TypeError
#   [None] / ['BUY_YES'] / [123] / mixed [valid, dict] -> bare
#                         AttributeError on the entry's type name
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT
# met — no validated producer guarantees the container: the only
# in-repo production caller (run_benchmark_simulation) passes plain
# list variables built by .append() with no check, the
# List[TradeExecutionResult] annotation is unenforced, direct
# callers/tests pass bare literals, stored records round-trip through
# JSON as plain dicts with no reconstruction layer before scoring,
# and the lists are mutable (post-construction appends unchecked).
# The container and entries are therefore validated HERE, at the
# scorer's consumption gate, as the FIRST trade gate — before
# SF-BL-10823's action gate or SF-BL-9475's finiteness gate can
# dereference any entry. isinstance via _is_trade_execution_result
# (SF-BL-10971), so genuine records (subclasses included) pass; the
# validated container is materialised to a list so every downstream
# pass (partition, sums, len) reads the same entries — list/tuple
# books are bit-identical, and a one-shot iterable can no longer be
# silently exhausted by the first validator and scored as empty.

def _validate_trades_container(
    trades: object,
) -> List[TradeExecutionResult]:
    """Return `trades` materialised as a list, or raise ValueError.

    Raises ValueError unless trades is an iterable container (not
    None, not a string/bytes/dict, not a bare TradeExecutionResult)
    whose every entry is a TradeExecutionResult (SF-BL-10968).
    Container failures are named INVALID_TRADES_CONTAINER and entry
    failures INVALID_TRADE_ENTRY, in the scorer's named
    NON_FINITE_TRADE_FIELD style (SF-BL-9475).
    """
    if trades is None:
        raise ValueError(
            "INVALID_TRADES_CONTAINER: evaluate_fleet_north_star "
            "trades=None is not an iterable container of "
            "TradeExecutionResult (SF-BL-10968): a missing trade book "
            "must be rejected by name, not crash with a bare TypeError "
            "('NoneType' object is not iterable) inside the first "
            "trade validator")
    if isinstance(trades, (str, bytes, dict)) or isinstance(
            trades, TradeExecutionResult):
        raise ValueError(
            f"INVALID_TRADES_CONTAINER: evaluate_fleet_north_star "
            f"trades={trades!r} (type {type(trades).__name__}) is not "
            "an iterable container of TradeExecutionResult "
            "(SF-BL-10968): a string, bytes, dict or bare "
            "TradeExecutionResult is a single record or raw payload, "
            "not a trade book — wrap genuine TradeExecutionResult "
            "records in a list")
    try:
        materialised = list(trades)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(
            f"INVALID_TRADES_CONTAINER: evaluate_fleet_north_star "
            f"trades={trades!r} (type {type(trades).__name__}) is not "
            "iterable (SF-BL-10968): the trade book must be an "
            "iterable container of TradeExecutionResult, not a bare "
            "value that crashes the first validator with a TypeError"
        ) from exc
    for idx, entry in enumerate(materialised):
        if not _is_trade_execution_result(entry):
            raise ValueError(
                f"INVALID_TRADE_ENTRY: evaluate_fleet_north_star "
                f"trades[{idx}]={entry!r} "
                f"(type {type(entry).__name__}) is not a "
                "TradeExecutionResult (SF-BL-10968): the scorer's "
                "validators dereference entry.action and the scored "
                "fields, so a non-trade entry — a JSON-style dict "
                "record that was never reconstructed via "
                "TradeExecutionResult(**row), None, or a bare value — "
                "crashes with a bare AttributeError instead of this "
                "named rejection")
    return materialised


# --- SF-BL-9475: non-finite trade fields return a NaN scorecard as valid --
# evaluate_fleet_north_star summed net_pnl_bid_marked, walked the equity
# curve and weighted est_days_to_resolution with NO finiteness check on any
# trade field, so one corrupt executed trade was rounded into a returned
# NorthStarScorecard instead of being rejected like the D3 empty-set case:
#   net_pnl NaN   -> north_star / ending value / CWR all NaN, MDD 0.0%
#   net_pnl +-inf -> north_star / ending value / CWR all +-inf, MDD 0.0%
#   invested NaN  -> `invested > 0` is False for NaN, so the trade silently
#                    VANISHED from the executed set (N_star 0.0, no trace)
#   invested +inf -> CWR 0.0 and avg_lockup clamped to 1.0 despite real PnL
#   est_days NaN  -> max(1.0, NaN) is 1.0, so D_lock silently became 1 day
#                    and the velocity multiplier fabricated N_star 1.4057
#                    on a fixture whose honest score is 0.4445
#   est_days +inf -> avg_lockup_days inf in the returned scorecard
# Kill-criterion audit (PRODUCER_AUDIT.md in the task work dir): the kill
# is NOT met. settle_or_mark_trade is not the sole writer of these fields
# and does not guarantee finiteness: TradeExecutionResult is a plain
# mutable dataclass with no __post_init__ validation, constructed directly
# by the benchmark's naive fleet and by callers/tests, and mutable after
# construction; settle itself computes PnL from unvalidated
# shares_acquired / invested_capital (verified: shares NaN on a settled
# win yields net_pnl_bid_marked NaN); and settle never writes
# est_days_to_resolution at all (it is copied from the unvalidated
# OrderBookSnapshot field at construction). The check therefore belongs
# here, at the scorer's consumption gate, mirroring SF-BL-8806 in nstar.py.
# Scope is exactly the prereg: the three named fields, on executed trades
# (action != SKIP -- a non-SKIP trade purports to have executed, so it is
# validated even before the invested>0 filter, which is precisely the
# filter that used to hide a NaN-invested trade). SKIP trades contribute
# nothing to any sum and keep their pre-existing behaviour untouched, as
# do all-finite books: the benchmark scorecards stay bit-identical.

_SCORED_TRADE_FIELDS = (
    "invested_capital",
    "net_pnl_bid_marked",
    "est_days_to_resolution",
)


def _validate_scored_trade_fields(trades: List[TradeExecutionResult]) -> None:
    """Raise ValueError on any non-finite scored field of an executed trade.

    Executed means action != TradeAction.SKIP. Reuses _is_finite_number
    (SF-BL-9438), so bool and non-numeric values are also rejected: they
    cannot be ordered or summed into an honest scorecard either.
    """
    for t in trades:
        if t.action == TradeAction.SKIP:
            continue
        for fname in _SCORED_TRADE_FIELDS:
            value = getattr(t, fname)
            if not _is_finite_number(value):
                raise ValueError(
                    f"NON_FINITE_TRADE_FIELD: evaluate_fleet_north_star "
                    f"trade {t.market_id!r} has {fname}={value!r}, which is "
                    "not finite (SF-BL-9475): a non-finite invested_capital, "
                    "net_pnl_bid_marked or est_days_to_resolution would be "
                    "summed, equity-walked or lockup-weighted into a "
                    "scorecard that reports corruption as a valid N_star")


def _validate_scored_trade_est_days_types(
    trades: List[TradeExecutionResult],
) -> None:
    """Raise ValueError on any executed trade whose est_days is a bool
    or non-numeric value (SF-BL-11534).

    Runs BEFORE _validate_scored_trade_fields (SF-BL-9475), which
    otherwise rejects a string/bool lock as NON_FINITE_TRADE_FIELD —
    the mislabel this task's kill criterion names. Scope mirrors
    SF-BL-9475 exactly: every non-SKIP trade (SKIP records — in
    particular evaluate's type-gate / Gate 0.1 records — deliberately
    carry the offending value for audit and contribute nothing to any
    sum). Numeric values, finite or not, pass untouched and keep
    their SF-BL-9475 / SF-BL-9364 error identities downstream.
    """
    for t in trades:
        if t.action == TradeAction.SKIP:
            continue
        value = t.est_days_to_resolution
        if not _est_days_is_numeric_type(value):
            raise ValueError(
                f"INVALID_EST_DAYS_TYPE: evaluate_fleet_north_star "
                f"trade {t.market_id!r} has "
                f"est_days_to_resolution={value!r} "
                f"(type {type(value).__name__}), which is not a real "
                "number (SF-BL-11534): a bool or non-numeric capital "
                "lockup is a TYPE error — it must not be lockup-"
                "weighted into avg_lockup, and it is not the "
                "non-finite VALUE defect of SF-BL-9475's "
                "NON_FINITE_TRADE_FIELD")


def _validate_scored_trade_lockups(trades: List[TradeExecutionResult]) -> None:
    """Raise ValueError on any executed trade with est_days <= 0 (SF-BL-9364).

    Companion to _validate_scored_trade_fields (SF-BL-9475), which owns
    finiteness and runs first — so a value reaching this check is known
    finite, and a non-finite lockup keeps its SF-BL-9475 error identity.
    Scope mirrors SF-BL-9475 exactly: every non-SKIP trade (SKIP trades
    contribute nothing to any sum; in particular an evaluate Gate 0.1
    SKIP record deliberately carries the offending value for audit and
    must not trip this gate). A finite lockup of zero or fewer days is
    an impossible holding period: pooled, it drags the capital-weighted
    avg_lockup down and manufactures velocity (prereg fixture: -30d
    pooled with +40d scored velocity 8.544x vs 3.021x clean); alone, the
    max(1.0, avg_lockup) floor masks it as a 1.0d lock.
    """
    for t in trades:
        if t.action == TradeAction.SKIP:
            continue
        value = t.est_days_to_resolution
        if _is_finite_number(value) and not value > 0:  # type: ignore[operator]
            raise ValueError(
                f"NON_POSITIVE_EST_DAYS_TO_RESOLUTION: "
                f"evaluate_fleet_north_star trade {t.market_id!r} has "
                f"est_days_to_resolution={value!r}, which is not > 0 "
                "(SF-BL-9364): a non-positive capital lockup would be "
                "lockup-weighted into avg_lockup and manufacture the "
                "velocity multiplier sqrt(365 / avg_lockup)")


# --- SF-BL-11304: negative/zero invested silently vanishes --------------
# evaluate_fleet_north_star partitioned with
#     executed = [t for t in trades if t.action != SKIP and t.invested_capital > 0]
#     skipped  = [t for t in trades if t.action == SKIP]
# so a non-SKIP trade with invested_capital <= 0 was counted as NEITHER:
# total_opportunities still included it (len(trades)), but its PnL never
# reached any sum. SF-BL-9475's gate validates only FINITENESS of
# invested_capital — a negative value is finite and passes — and
# SF-BL-9473 covers negative invested PRODUCED by sweep_order_book's
# BUY_NO pricing, not the scorer's consumption gate, where directly
# constructed records arrive. Verified live pre-fix (repro.py in the
# task work dir; two BUY_YES trades, resolved [(0.6, 1), (0.6, 0)]):
# invested 100 / PnL +10 pooled with invested -100 / PnL +50 returned
# total_opportunities 2, executed_trades 1, skipped_opportunities 0,
# total_net_pnl 10.0 — the count identity executed + skipped == total
# broken and the +$50 PnL silently excluded; the zero-invested variant
# (invested 0 / PnL +50) vanished identically. Kill-criterion audit
# (PRODUCER_AUDIT.md in the task work dir): NOT met — TradeExecutionResult
# has NO construction-time validation of invested_capital (its
# __post_init__ / __setattr__ validate only action, SF-BL-10823) and is
# a plain mutable dataclass (post-construction and post-settlement
# mutation, object.__setattr__ / unpickling all deliver <= 0 values to
# the scorer); settle_or_mark_trade rejects negative invested at its
# own entry (SF-BL-9541) but never writes the field and is not the sole
# writer. The value is therefore validated here, at the scorer's
# consumption gate. Scope (ROUND 2, broadened per the round-1 tester
# VERDICT): a non-SKIP trade with invested_capital <= 0 raises
# REGARDLESS of PnL. Round 1 raised only when the trade carried nonzero
# PnL (the prereg's literal raise condition), but the prereg states the
# raise's purpose as a consequence — "so executed+skipped ==
# total_opportunities holds on EVERY fixture" — and a <= 0 trade with
# PnL exactly 0.0 still vanishes from the count: verified live by the
# round-1 tester ([invested=-100, pnl=0.0] alone returned total 1,
# executed 0, skipped 0; [invested=0.0, pnl=0.0] alone the same;
# [good 100/+10, invested=-100 pnl=0.0] returned total 2, executed 1,
# skipped 0). No PnL is lost in that class, but the identity the prereg
# promises is, so the narrower reading contradicts the prereg's own
# "every fixture" wording and is NOT adopted: the identity is the
# binding requirement, and raising on every non-SKIP invested <= 0 is
# the smallest scope that satisfies both the literal nonzero-PnL cases
# and the identity (the alternative — counting the trade as skipped —
# would falsify the SKIP audit count, whose records are genuine
# evaluate decisions). SKIP trades (evaluate's zero-size records)
# remain exempt, as for SF-BL-9475 / SF-BL-9364: they ARE counted, as
# skipped, so they never broke the identity. Ordering: this gate runs
# AFTER SF-BL-9475 (a non-finite invested keeps its NON_FINITE identity —
# NaN also fails `<= 0`, so it could never reach here honestly) and
# AFTER SF-BL-9364 (a trade tripping both keeps its lockup identity —
# pinned by SF-BL-9364's invested=0.0 / est_days=-5 fixture), and before
# the executed/skipped partition below. All-positive books are a no-op:
# bit-identical.

def _validate_scored_trade_invested(trades: List[TradeExecutionResult]) -> None:
    """Raise ValueError on any non-SKIP trade with invested_capital <= 0
    (SF-BL-11304, round-2 scope: regardless of PnL).

    Companion to _validate_scored_trade_fields (SF-BL-9475), which owns
    finiteness and runs first — so a value reaching this check is known
    finite (bool / non-numeric invested is rejected there, via
    _is_finite_number), and the PnL reported alongside it is likewise
    known finite and non-bool. Runs after _validate_scored_trade_lockups
    (SF-BL-9364) so a trade violating both keeps its SF-BL-9364 error
    identity.
    """
    for t in trades:
        if t.action == TradeAction.SKIP:
            continue
        invested = t.invested_capital
        pnl = t.net_pnl_bid_marked
        if _is_finite_number(invested) and invested <= 0:  # type: ignore[operator]
            raise ValueError(
                f"NON_POSITIVE_INVESTED_CAPITAL: "
                f"evaluate_fleet_north_star trade {t.market_id!r} has "
                f"invested_capital={invested!r} <= 0 with "
                f"net_pnl_bid_marked={pnl!r} (SF-BL-11304): a "
                "non-SKIP trade that deployed no (or negative) capital "
                "is not an executed trade — the scorer's invested>0 "
                "executed filter would drop it and the SKIP-only "
                "skipped count would not count it, so the trade would "
                "vanish from the scorecard (executed + skipped != "
                "total_opportunities) and any PnL it carries would be "
                "silently excluded from CWR / MDD")


# --- SF-BL-34956: PnL outside the position's payout envelope scores ----
# evaluate_fleet_north_star summed net_pnl_bid_marked into CWR_bid, the
# equity/MDD walk and trade_returns with only the SF-BL-9475 finiteness
# screen, so a single executed trade could carry a PnL no binary
# position of its size can produce. Verified live pre-fix (repro.py in
# the task work dir; one settled BUY_YES trade, shares_acquired=2000,
# invested_capital=1000, est_days 10, resolved [(0.6, 1), (0.6, 0)]):
# PnL -50000 RETURNED CWR -50.0, ending value -$40,000 on a $10,000
# account and N_star -112.8276, and the +50000 twin RETURNED
# N_star +111.1276 — yet the position's feasible payout envelope is
# [-1000, +1000]: at settlement shares pay $1 or $0 (gross in
# {0, shares}), and an open mark is shares * bid with the bid validated
# into [0, 1] (SF-BL-9436; the SF-BL-3660 exit-book proceeds are a
# filled VWAP in (0, 1) times a fill fraction in [0, 1] times shares,
# likewise inside [0, shares]), so every legitimate settle/mark lands
# in [-invested, shares - invested] for BOTH sides, settled and open.
# Distinct from SF-BL-34866, which bounds the AGGREGATE (a set of
# individually feasible trades whose summed invested exceeds capital);
# no item bounded a single trade's PnL against its own envelope, and
# the scorer never validated shares_acquired at all (it is absent from
# SF-BL-9475's _SCORED_TRADE_FIELDS), so the envelope's upper bound was
# not even computable from validated fields. Kill-criterion audit
# (PRODUCER_AUDIT.md in the task work dir): NOT met — in-module only
# settle_or_mark_trade ASSIGNS net_pnl_bid_marked, but the field is a
# plain dataclass field accepted as a constructor keyword and mutated
# freely afterwards (TradeExecutionResult's __setattr__ / __post_init__
# guard only action, SF-BL-10823, and is_settled, SF-BL-11536), direct
# construction with arbitrary PnL is the norm in the test suites and
# fleet recompute scripts, and no provenance field records whether
# settle ever ran — the value is therefore validated here, at the
# scorer's consumption gate.
# PRE-REGISTERED EPSILON: a trade passes iff
# -invested - eps <= PnL <= (shares - invested) + eps with
# eps = PNL_ENVELOPE_EPSILON_ABS + PNL_ENVELOPE_EPSILON_REL *
# max(1.0, |shares|, |invested|) — pure float dust from the
# shares * bid - invested arithmetic (proportionally ~1e-9 of the
# position scale); any economically meaningful excess raises.
# Scope mirrors SF-BL-9475 / SF-BL-11304: executed (non-SKIP) trades
# only — SKIP records contribute no PnL to any sum. Ordering: after
# SF-BL-9475 (which owns finiteness of PnL/invested), SF-BL-9364 and
# SF-BL-11304 (which establishes invested > 0), and before the
# SF-BL-34866 leverage assessment and every sum. The gate also
# validates shares_acquired itself (finite real >= 0, the SF-BL-9541
# domain): a non-finite, negative or non-numeric share count makes
# the envelope undefined and raises the same named error.
# In-envelope books are a pure no-op: bit-identical scorecards.

PNL_ENVELOPE_EPSILON_ABS = 1e-9
PNL_ENVELOPE_EPSILON_REL = 1e-9


def _pnl_envelope_epsilon(shares: float, invested: float) -> float:
    """Pre-registered float-dust tolerance for the SF-BL-34956 envelope."""
    return PNL_ENVELOPE_EPSILON_ABS + PNL_ENVELOPE_EPSILON_REL * max(
        1.0, abs(shares), abs(invested))


def _validate_scored_trade_pnl_envelopes(
    trades: List[TradeExecutionResult],
) -> None:
    """Raise ValueError on any executed trade whose PnL lies outside
    its feasible payout envelope (SF-BL-34956).

    The envelope of a binary position is
    [-invested_capital, shares_acquired - invested_capital]: shares
    pay $1 or $0 at settlement and mark at shares * bid with bid in
    [0, 1] while open, so no legitimate settle/mark can produce a PnL
    outside it. Runs after _validate_scored_trade_invested
    (SF-BL-11304), so invested_capital and net_pnl_bid_marked reaching
    this check are known finite (SF-BL-9475) and invested is positive;
    shares_acquired has no earlier scorer gate, so its own domain
    (finite real >= 0, SF-BL-9541) is validated here first — an
    out-of-domain share count makes the envelope undefined.
    """
    for t in trades:
        if t.action == TradeAction.SKIP:
            continue
        shares = t.shares_acquired
        invested = t.invested_capital
        pnl = t.net_pnl_bid_marked
        if not _shares_acquired_in_domain(shares):
            raise ValueError(
                f"PNL_OUTSIDE_PAYOUT_ENVELOPE: "
                f"evaluate_fleet_north_star trade {t.market_id!r} has "
                f"shares_acquired={shares!r}, which is not a finite "
                "real number >= 0 (SF-BL-34956): the feasible payout "
                "envelope [-invested_capital, shares_acquired - "
                "invested_capital] is undefined for a negative, NaN, "
                "infinite or non-numeric share count, so the trade's "
                f"net_pnl_bid_marked={pnl!r} cannot be verified "
                "against its position and must not be summed into "
                "CWR / MDD")
        lo = -invested  # type: ignore[operator]
        hi = shares - invested  # type: ignore[operator]
        eps = _pnl_envelope_epsilon(float(shares), float(invested))  # type: ignore[arg-type]
        if pnl < lo - eps or pnl > hi + eps:  # type: ignore[operator]
            raise ValueError(
                f"PNL_OUTSIDE_PAYOUT_ENVELOPE: "
                f"evaluate_fleet_north_star trade {t.market_id!r} has "
                f"net_pnl_bid_marked={pnl!r} outside its feasible "
                f"payout envelope [{lo!r}, {hi!r}] "
                f"(shares_acquired={shares!r}, "
                f"invested_capital={invested!r}, epsilon={eps!r}) "
                "(SF-BL-34956): a binary position pays $1 or $0 per "
                "share at settlement and marks at shares * bid with "
                "bid in [0, 1] while open, so no legitimate "
                "settle/mark of this position can produce a PnL "
                "beyond [-invested, shares - invested]; an "
                "out-of-envelope PnL would be summed into CWR / MDD "
                "as a fabricated return (prereg fixture: PnL -50000 "
                "on 2000 shares / $1000 invested scored CWR -50.0)")


# --- SF-BL-35210: settled PnL never checked against its exact payout ---
# evaluate_fleet_north_star summed net_pnl_bid_marked into CWR_bid, the
# equity/MDD walk and the settlement win rate with only the SF-BL-9475
# finiteness screen and (SF-BL-34956) a feasible-RANGE screen, so a
# settled trade could carry any in-range PnL although settlement fully
# determines the figure: a binary position pays $1 per winning share
# and $0 per losing share, so a settled trade's PnL is EXACTLY
# shares_acquired - invested_capital on a win and EXACTLY
# -invested_capital on a loss — the very figures settle_or_mark_trade's
# settled branch books (gross_payout - invested_capital). Verified live
# pre-fix (repro.py in the task work dir; shares_acquired=1000,
# invested_capital=500, outcome_yes=1, resolved [(0.6, 1), (0.6, 0)]):
# a BUY_YES win carrying net_pnl_bid_marked +100 RETURNED
# total_net_pnl +100.0 / CWR 0.20 / N_star 0.4445 although the only
# possible value is +500, and the BUY_NO twin (outcome_yes=1, a NO
# loss) carrying -100 RETURNED -100.0 although the only possible value
# is -500. Both fixtures lie INSIDE SF-BL-34956's envelope
# [-500, +500] and pass its screen — a range check cannot catch a
# wrong-but-feasible figure; only the exact payout identity can.
# Kill-criterion audit (PRODUCER_AUDIT.md in the task work dir): NOT
# met — settle_or_mark_trade is the only in-module ASSIGNER of the
# field, but TradeExecutionResult accepts net_pnl_bid_marked as an
# unvalidated constructor keyword (its __setattr__ / __post_init__
# guard only action, SF-BL-10823, and is_settled, SF-BL-11536), the
# field is mutable after construction (and after settlement, so a
# settled record's shares/PnL can be edited apart), direct
# construction is the norm in test suites and fleet recompute scripts,
# and no provenance field records whether settle ever ran. The check
# therefore belongs here, at the scorer's consumption gate.
# PRE-REGISTERED EPSILON: SETTLED_PNL_EPSILON = 0.005 dollars (half a
# cent, the maximum error of a cent-rounded booked PnL); a settled
# trade is flagged iff |net_pnl_bid_marked - expected| > epsilon.
# settle_or_mark_trade computes the expected figure in float exactly
# as this check does, so settle-produced books differ by 0.0, flag
# 0 / 0.0, and score bit-identically.
# RAISE-OR-FLAG (prereg amendment, task work dir PREREG.md): the
# prereg's success text is disjunctive ("raises or flags"). The raise
# branch was implemented first; its regression sweep broke at least
# six ratified suites outright (SF-BL-9475 / 9437 / 11304 / 10968 /
# 11535 / 9364 — each pins a settled record carrying a non-payout PnL
# as the CONTROL fixture for an unrelated gate) plus SF-BL-34866
# (6/15) and SF-BL-9477 (12/13). Per the SF-BL-34866 precedent, which
# took its prereg's flag/report branch over rejection on exactly this
# evidence, this check FLAGS: the scorecard gains
# settled_pnl_mismatch_count / settled_pnl_mismatch_invested
# (defaulted pure additions, SF-BL-10767 precedent), computed over
# the executed set before the SF-BL-10767 stale partition (settled
# trades are never stale-excluded, so the sets coincide). Every
# pre-existing numeric field of every book is bit-identical.
# Scope: executed (non-SKIP) trades with is_settled True AND
# outcome_yes a strict int in {0, 1} (the SF-BL-9477 domain, via
# _outcome_yes_in_domain) — only those are classifiable as a win or a
# loss. Open trades are exempt: their PnL is a bid mark, bounded by
# SF-BL-34956's envelope but not determined by settlement. Settled
# records without a valid outcome cannot be classified and are outside
# this check's determinable scope. SKIP records contribute nothing to
# any sum and are exempt, as for every neighbouring scorer gate.
# Ordering: immediately after SF-BL-34956 (which follows SF-BL-9475
# finiteness and SF-BL-11304 invested > 0 and validates shares_acquired
# itself), so every value read here — PnL, invested, shares — is known
# finite with invested > 0 and shares >= 0, and a trade tripping an
# earlier gate keeps that gate's error identity.

SETTLED_PNL_EPSILON = 0.005


def _assess_settled_pnl_mismatches(
    trades: List[TradeExecutionResult],
) -> Tuple[int, float]:
    """Return (count, invested) of settled executed trades whose PnL
    is not their exact settlement payout (SF-BL-35210 flag branch).

    For a settled trade the payout is fully determined: the winning
    side's shares pay $1.00 and the losing side's pay $0.00, so the
    expected net PnL is shares_acquired - invested_capital on a win
    and -invested_capital on a loss — exactly what
    settle_or_mark_trade's settled branch books. A booked
    net_pnl_bid_marked further than SETTLED_PNL_EPSILON (half a cent,
    the pre-registered rounding epsilon) from that figure is not a
    rounding artefact; it is a fabricated return being summed into
    CWR / MDD and counted in the settlement win rate, and it is
    flagged on the scorecard (settled_pnl_mismatch_count /
    settled_pnl_mismatch_invested) rather than rejected — see the
    RAISE-OR-FLAG note above. Runs after
    _validate_scored_trade_pnl_envelopes (SF-BL-34956), so the fields
    read here are known finite, invested_capital > 0 and
    shares_acquired >= 0. Scope: non-SKIP trades with is_settled True
    and outcome_yes a strict int in {0, 1}; open trades (bid marks,
    not payouts) and unclassifiable settled records (no valid outcome)
    are not counted.
    """
    count = 0
    invested_sum = 0.0
    for t in trades:
        if t.action == TradeAction.SKIP:
            continue
        if not t.is_settled:
            continue
        if not _outcome_yes_in_domain(t.outcome_yes):
            continue
        won = (
            (t.action == TradeAction.BUY_YES and t.outcome_yes == 1)
            or (t.action == TradeAction.BUY_NO and t.outcome_yes == 0)
        )
        expected = (
            t.shares_acquired - t.invested_capital if won
            else -t.invested_capital
        )
        if abs(t.net_pnl_bid_marked - expected) > SETTLED_PNL_EPSILON:  # type: ignore[operator]
            count += 1
            invested_sum += t.invested_capital
    return count, invested_sum


# --- SF-BL-34866: levered trade sets score MDD > 100%, unflagged -------
# evaluate_fleet_north_star walked the equity curve from
# starting_capital with NO comparison of the book's total deployed
# capital against that capital base, so a book whose summed
# invested_capital exceeds starting_capital (a levered book, by the
# prereg's definition) was scored as if the capital existed. Verified
# live pre-fix (repro.py in the task work dir; two settled BUY_YES
# trades, $8,000 invested each, PnL -$8,000 each — a full loss of
# stake per trade — resolved [(0.6, 1), (0.6, 0)], cap $10,000):
#   RETURNED ending_account_value_bid_marked -$6,000.00,
#   max_drawdown_pct 160.0 and drawdown_penalty 2.9 (N_star -5.1226):
#   the walk runs equity 10000 -> 2000 -> -6000, peak stays 10000, and
#   (peak - equity) / peak exceeds 1.0 once equity crosses zero — a
#   drawdown above 100% is impossible for an unlevered book (its
#   equity cannot fall below zero when no trade loses more than its
#   stake), so the >100% figure certifies leverage that the scorecard
#   nowhere flags. CWR_bid also reads -1.0 against capital that was
#   never there.
# Distinct from the neighbours: SF-BL-10850 / SF-BL-9437 validate the
# starting_capital SCALAR (finite real > 0) — a levered book's scalar
# is perfectly valid; SF-BL-34830 instruments aggregate exposure on
# the evaluate_and_size_opportunity side and SF-BL-6687 an at-risk cap
# inside nstar.py's Account — neither validates the scorecard's own
# capital base at scoring time.
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT
# met — no consumer is shown to pass only capital-bounded books under
# a tested invariant. TradeExecutionResult carries no capital field
# and no construction-time relation to any capital base;
# settle_or_mark_trade never receives starting_capital; the only
# in-repo production caller (run_benchmark_simulation) builds its
# naive baseline by flat $1,200 sweeps that bypass
# evaluate_and_size_opportunity entirely (its current 8 fixtures sum
# under the cap only incidentally — a ninth would not), and direct
# callers/tests construct books with bare literals.
# Prereg branch taken: "reported with an explicit leverage flag and
# a drawdown bounded to [0, 100%]" — NOT the rejection branch, on
# evidence (CALLER_CENSUS.md): a summed invested above starting
# capital is not a corrupt value like the non-finite / non-positive
# fields the neighbouring gates reject. Sequential turnover sums
# above the starting capital in normal operation (capital is
# recycled as trades settle), and two ratified prereg suites fixture
# such books as legitimate scoreable edges: SF-BL-9437's suite scores
# a $500 trade against starting_capital $0.50 (1000x summed leverage,
# a winning book) and SF-BL-9475's suite scores a $1e12 invested /
# -$1e12 PnL trade against the $10,000 default, asserting only a
# finite N_star. Rejection would break both ratified fixtures and
# criminalise turnover; the defect the prereg names is the UNFLAGGED
# >100% drawdown, and that is what is closed here:
#   * the scorecard gains is_levered / leverage_ratio /
#     executed_invested_capital (defaulted fields, SF-BL-10767
#     precedent), computed over the EXECUTED set — exactly the
#     scorer's partition (action != SKIP and invested_capital > 0),
#     assessed BEFORE the SF-BL-10767 stale-mark partition (leverage
#     is a property of the submitted book against its capital base;
#     a stale-mark scoring exclusion does not release capital);
#   * the equity walk's drawdown is bounded to [0, 1] via
#     _bounded_drawdown, so max_drawdown_pct lies in [0, 100] and
#     drawdown_penalty in [0, 1.7] on every returned scorecard.
# Scope / no-op guarantee: the assessment runs after SF-BL-9475 /
# SF-BL-11304, so every summed invested is known finite and positive;
# SKIP records deployed no capital and contribute nothing, as for
# SF-BL-9475 / SF-BL-9364 / SF-BL-11304. Equality is unlevered
# (sum == starting_capital: a full loss ends equity at exactly 0,
# MDD exactly 100%). Every pre-existing numeric field of a book
# whose raw drawdown never exceeded 100% is bit-identical — the
# bound binds only where the pre-fix walk crossed zero, and the new
# fields are pure additions. The ending account value is deliberately
# NOT floored: it is the arithmetic sum of the book's PnL against its
# capital, and flooring it would falsify total_net_pnl / return; the
# leverage flag is what now names the regime that produced it.

def _assess_book_leverage(
    trades: List[TradeExecutionResult],
    starting_capital: float,
) -> Tuple[float, float, bool]:
    """Return (executed_invested, leverage_ratio, is_levered).

    The summed invested_capital of the executed set (action !=
    TradeAction.SKIP and invested_capital > 0 — the scorer's own
    partition) against starting_capital, which SF-BL-9437 has
    already validated as a finite real > 0 (SF-BL-34866). Runs after
    _validate_scored_trade_invested (SF-BL-11304), so every summed
    value is known finite and positive. is_levered is a STRICT
    excess: a sum exactly equal to the capital base is unlevered.
    """
    total_invested = sum(
        t.invested_capital for t in trades
        if t.action != TradeAction.SKIP and t.invested_capital > 0)
    if not _is_finite_number(starting_capital):
        # Unreachable in live scoring (SF-BL-9437's gate runs first);
        # stay inert so a bypassed-capital repro keeps its historic
        # downstream error instead of gaining a new one here.
        return total_invested, 0.0, False
    ratio = total_invested / starting_capital  # type: ignore[operator]
    return total_invested, ratio, total_invested > starting_capital


def _bounded_drawdown(peak_equity: float, equity: float) -> float:
    """Peak-to-current drawdown as a fraction, bounded to [0, 1].

    The pre-fix walk computed (peak - equity) / peak raw, which
    exceeds 1.0 once equity crosses zero (SF-BL-34866: the 160%
    repro). A drawdown is the fraction of peak equity lost and
    cannot exceed 100%; the bound binds only on books whose equity
    crossed zero and is the identity everywhere else.
    """
    if peak_equity <= 0:
        return 0.0
    return min(1.0, max(0.0, (peak_equity - equity) / peak_equity))


# --- SF-BL-11049: resolved-forecast CONTAINER shape never validated ----
# evaluate_fleet_north_star's D3 guard tests only the truthiness of
# all_resolved_forecasts, so any truthy malformed container passed it
# and crashed inside compute_empirical_ece_and_brier (pre-11535) or —
# after SF-BL-11535 added an element gate that enumerates the panel —
# inside that gate's own enumerate(), still with a bare error.
# Verified live pre-fix (repro.py in the task work dir; honest
# control: one BUY_YES trade, $500 invested, PnL +$490, est_days 10,
# resolved [(0.8, 1)] -> N_star 0.8013):
#   forecasts=123 / 5.0 / True / object()
#                      -> TypeError: '<type>' object is not iterable
#                         (bare, from enumerate() in
#                         _validate_resolved_forecast_panel)
#   forecasts='abc'    -> the string was iterated character by
#                         character and mis-reported as a malformed
#                         ENTRY ('a' is not a pair), never as the
#                         malformed CONTAINER it is; pre-11535 it died
#                         with a bare unpack ValueError in
#                         compute_empirical_ece_and_brier
#   forecasts=[{'p': 0.6, 'o': 1}] (dict entry)
#                      -> unpacking a dict yields its KEYS, so the
#                         entry was mis-reported as forecast='p'; and
#                         a dict entry whose two keys are numeric —
#                         [{0.8: 'x', 1: 'y'}] — PASSED the element
#                         gate and scored N_star 0.8013, bit-identical
#                         to the genuine [(0.8, 1)] panel: a malformed
#                         entry silently fabricating a scorecard
#   forecasts=0 with no executed trades
#                      -> falsy, so every guard no-oped and the call
#                         silently RETURNED N_star 0.0
#   forecasts=<generator of valid pairs>
#                      -> TypeError: object of type 'generator' has
#                         no len() inside compute_empirical_ece_and_brier
# Distinct from SF-BL-10968 (the TRADES container of this same
# scorer), SF-BL-8898 (the empty-set D3 case — its NO_ELIGIBLE_ROWS
# rejection is unchanged here: None and empty containers still pass
# straight to it) and SF-BL-10912 (the container claim scoped to
# compute_empirical_ece_and_brier called directly — that direct
# surface is deliberately untouched).
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT
# met — the resolved set is NOT only ever built by validated
# settlement paths: the sole in-repo production caller
# (run_benchmark_simulation) appends bare (cal_p, outcome_yes)
# literals to a plain mutable list, settle_or_mark_trade never writes
# the panel, the List[Tuple[float, int]] annotation is unenforced,
# and no schema/deserialisation layer stands before scoring. The
# container is therefore validated HERE, at the scorer's consumption
# gate (the SF-BL-10968 placement for trades), immediately before
# SF-BL-11535's element gate: materialised to a list, so the D3
# guard, the element gate and the ECE/Brier computation all read the
# same entries — genuine list/tuple/set panels are bit-identical,
# and a one-shot generator can no longer die on len() or be silently
# exhausted by the first validator.

def _validate_resolved_forecasts_container(
    all_resolved_forecasts: object,
) -> Optional[List[Tuple[float, int]]]:
    """Return the panel materialised as a list (None stays None).

    Raises ValueError unless all_resolved_forecasts is None (the D3
    guard, SF-BL-8898, owns None/empty semantics) or an iterable
    container that is not a string/bytes/dict (SF-BL-11049).
    Container failures are named INVALID_RESOLVED_FORECASTS_CONTAINER,
    in the scorer's named INVALID_TRADES_CONTAINER style
    (SF-BL-10968).
    """
    if all_resolved_forecasts is None:
        return None
    if isinstance(all_resolved_forecasts, (str, bytes, dict)):
        raise ValueError(
            f"INVALID_RESOLVED_FORECASTS_CONTAINER: "
            f"evaluate_fleet_north_star "
            f"all_resolved_forecasts={all_resolved_forecasts!r} "
            f"(type {type(all_resolved_forecasts).__name__}) is not "
            "an iterable container of (forecast, outcome) pairs "
            "(SF-BL-11049): a string, bytes or dict is a raw payload "
            "or a single record, not a resolved panel — its "
            "characters/keys would otherwise be iterated as panel "
            "entries and crash the ECE/Brier computation with a "
            "bare TypeError/ValueError, or be mis-reported as a "
            "malformed entry")
    try:
        materialised = list(all_resolved_forecasts)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(
            f"INVALID_RESOLVED_FORECASTS_CONTAINER: "
            f"evaluate_fleet_north_star "
            f"all_resolved_forecasts={all_resolved_forecasts!r} "
            f"(type {type(all_resolved_forecasts).__name__}) is not "
            "iterable (SF-BL-11049): the resolved panel must be an "
            "iterable container of (forecast, outcome) pairs, not a "
            "bare value (e.g. 123) that passes the truthiness-only "
            "D3 guard and crashes the panel validator with a bare "
            "TypeError ('int' object is not iterable)"
        ) from exc
    return materialised


# --- SF-BL-11535: resolved-forecast panel element types never validated --
# evaluate_fleet_north_star handed all_resolved_forecasts straight to
# compute_empirical_ece_and_brier with NO element type check at its own
# boundary, so the panel's element types decided the scorecard:
#   [(True, 1)]    -> bool is an int subclass, so the forecast was read
#                     as certainty (p=1.0): ECE 0.0, Brier 0.0, and N_star
#                     2.2672 on the prereg fixture (one BUY_YES trade,
#                     $500 invested, PnL +$325, D_lock 30d) against
#                     0.1129 for the numeric control [(0.7, 1)] — a
#                     twenty-fold fabrication from a single bool.
#   [(0.7, True)]  -> a bool outcome was likewise read as 1 and scored
#                     identically to the numeric control, silently.
#   [('0.7', 1)] / [(0.7, '1')] / None elements -> bare TypeError from
#                     the (p - o) ** 2 Brier subtraction inside
#                     compute_empirical_ece_and_brier, crashing the
#                     scorecard with no named rejection.
#   NaN / +-inf elements -> bare ValueError / OverflowError from the
#                     binning int() conversion, or (NaN outcome) a NaN
#                     ECE/Brier silently summed into the scorecard.
# SF-BL-11049 fixtures only this panel's CONTAINER shape and
# SF-BL-11048 the element types of compute_empirical_ece_and_brier
# called DIRECTLY; neither validates the elements at this scorer's
# boundary, which is the surface fixed here (the direct-function
# surface remains SF-BL-11048's and is deliberately untouched).
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT
# met — no validated producer stands between an external panel and
# this function. The benchmark builds its panel from
# aggregate_fleet_forecasts floats and bare int outcome literals, the
# annotation List[Tuple[float, int]] is unenforced at runtime, the
# panel is a plain list that callers/tests construct and mutate
# directly with bare literals, and no schema/deserialisation layer
# exists in the repo. The elements themselves are therefore validated
# here, at the consumption gate, reusing _is_finite_number
# (SF-BL-9438): a finite real number, bool excluded — so int forecasts
# / outcomes (the annotated outcome type) and every numeric panel pass
# bit-identically, while bool, string, None and non-finite elements
# raise the named ValueError below.

def _validate_resolved_forecast_panel(
    all_resolved_forecasts: List[Tuple[float, int]],
) -> None:
    """Raise ValueError on any bool / non-numeric / non-finite element.

    Validates every (forecast, outcome) pair in the resolved panel
    (SF-BL-11535). An empty or None panel is a no-op: the D3 guard
    (SF-BL-8898) owns the empty-set semantics and its NO_ELIGIBLE_ROWS
    rejection is unchanged. An entry that is not a (forecast, outcome)
    pair at all also raises the named ValueError rather than leaking a
    bare unpack TypeError/ValueError from the ECE/Brier computation.
    """
    if not all_resolved_forecasts:
        return
    for idx, entry in enumerate(all_resolved_forecasts):
        # SF-BL-11049: a dict / string / bytes ENTRY is not a
        # (forecast, outcome) pair and must never be unpacked:
        # unpacking a dict yields its keys (so {'p': 0.6, 'o': 1} was
        # mis-reported as forecast='p', and a dict with two numeric
        # keys silently SCORED as that pair), and unpacking a string
        # yields its characters. Reject it by name here; genuine
        # tuple/list/array pairs flow to the unpack below unchanged.
        if isinstance(entry, (str, bytes, dict)):
            raise ValueError(
                f"INVALID_RESOLVED_FORECAST_ELEMENT: "
                f"evaluate_fleet_north_star all_resolved_forecasts[{idx}]"
                f"={entry!r} (type {type(entry).__name__}) is not a "
                "(forecast, outcome) pair (SF-BL-11049; entry-shape "
                "leg of the SF-BL-11535 panel gate): a dict entry "
                "unpacks to its keys and a string/bytes entry to its "
                "characters/bytes, so it must be rejected by name, "
                "not crash, mis-report, or silently score inside the "
                "ECE/Brier computation")
        try:
            forecast, outcome = entry
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"INVALID_RESOLVED_FORECAST_ELEMENT: "
                f"evaluate_fleet_north_star all_resolved_forecasts[{idx}]"
                f"={entry!r} is not a (forecast, outcome) pair "
                "(SF-BL-11535): a malformed panel entry must be rejected "
                "by name, not crash or corrupt the ECE/Brier computation"
            ) from exc
        for role, value in (("forecast", forecast), ("outcome", outcome)):
            if not _is_finite_number(value):
                raise ValueError(
                    f"INVALID_RESOLVED_FORECAST_ELEMENT: "
                    f"evaluate_fleet_north_star "
                    f"all_resolved_forecasts[{idx}] has {role}={value!r} "
                    f"(type {type(value).__name__}), which is not a "
                    "finite real number with bool excluded "
                    "(SF-BL-11535): a bool forecast is read as certainty "
                    "and fabricates N_star (prereg fixture: [(True, 1)] "
                    "scored 2.2672 vs the 0.1129 numeric control), and a "
                    "string/None element dies with a bare TypeError "
                    "inside compute_empirical_ece_and_brier")


# --- SF-BL-9437: starting_capital <= 0 / non-finite corrupts the scorecard --
# starting_capital is the scorecard's denominator and equity-walk seed:
# portfolio_return_pct divides total_pnl by it, and the MDD walk seeds
# equity = peak_equity = starting_capital and guards each drawdown with
# `peak_equity > 0`. Pre-fix nothing validated it, so (verified live
# pre-fix, repro.py in the task work dir; one BUY_YES trade, $500
# invested, PnL +$100, resolved [(0.6, 1), (0.6, 0)], honest control
# N_star 0.4445 / return +1.0% / MDD 0.0%):
#   cap=0        -> ZeroDivisionError: float division by zero — a
#                   misconfigured scorecard kills the cycle outright
#   cap=-10000   -> RETURNED return -1.0% (sign flipped on a +$100 book)
#                   and MDD 0.0%; on a -$500 book return +5.0% and MDD
#                   0.0% — peak starts negative, the peak>0 guard yields
#                   dd 0.0 forever, so an insolvent book is certified
#                   drawdown-free
#   cap=NaN      -> RETURNED return NaN, ending value NaN, MDD 0.0%
#   cap=+-inf    -> RETURNED ending value +-inf
#   cap='10000' / None -> bare TypeError from the equity walk
#   cap=True     -> RETURNED return 10000.0% (bool read as $1 of capital)
# The valid domain is a finite real strictly > 0 (bool / non-numeric
# excluded, via _is_finite_number, SF-BL-9438): zero capital has no
# return denominator and no equity curve, and negative capital is a
# caller configuration error, not an insolvent fleet to be scored.
# Kill-criterion audit (CALLER_CENSUS.md in the task work dir): NOT
# met — callers do NOT all pass the literal 10000 default. The
# benchmark passes a variable (starting_cap = 10_000.0), the repo test
# suites omit the argument, and fleet callers pass other literals and
# variables (100_000.0 / 100000.0 in SF-BL-3565 / SF-BL-3624 audits,
# EQUITY / cap / start variables in SF-BL-3836 / SF-BL-3586 /
# SF-BL-10667 and the wave3 velocity-canonical recompute) — the value
# arrives from caller configuration with no finiteness/positivity
# invariant anywhere, so it is validated here, at scorecard entry.
# No upper bound is preregistered, so none is imposed.

def _is_valid_starting_capital(value: object) -> bool:
    """True iff starting_capital is a finite real number strictly > 0."""
    return _is_finite_number(value) and value > 0  # type: ignore[operator]


def _validate_starting_capital(starting_capital: object) -> None:
    """Raise ValueError unless starting_capital is in domain (SF-BL-9437)."""
    if not _is_valid_starting_capital(starting_capital):
        raise ValueError(
            f"INVALID_STARTING_CAPITAL: evaluate_fleet_north_star "
            f"starting_capital={starting_capital!r} is not a finite "
            "real number > 0 (SF-BL-9437): portfolio_return_pct divides "
            "by starting_capital (0 raises ZeroDivisionError) and the "
            "drawdown walk seeds peak_equity from it (a negative seed "
            "is guarded to MDD 0.0 forever, certifying an insolvent "
            "book as drawdown-free)")


def _validate_scored_at(scored_at: object) -> None:
    """Raise ValueError unless scored_at is None or finite (SF-BL-10767).

    scored_at is the scoring instant the SF-BL-10767 staleness rule
    ages bid marks against; None means legacy scoring (no exclusion).
    A non-finite or non-numeric instant cannot age anything honestly
    (every age comparison against NaN is False, which would read
    every mark as fresh), so it is rejected by name, mirroring the
    SF-BL-9437 starting-capital gate. No sign bound is imposed: the
    rule only differences scored_at against each mark's as-of.
    """
    if scored_at is None:
        return
    if not _is_finite_number(scored_at):
        raise ValueError(
            f"INVALID_SCORED_AT: evaluate_fleet_north_star "
            f"scored_at={scored_at!r} is not a finite real number "
            "(SF-BL-10767): mark ages are computed against scored_at, "
            "and a NaN/inf/non-numeric instant would silently read "
            "every stale bid mark as fresh")


def evaluate_fleet_north_star(
    fleet_name: str,
    trades: List[TradeExecutionResult],
    all_resolved_forecasts: List[Tuple[float, int]],
    starting_capital: float = 10_000.0,
    scored_at: Optional[float] = None,
) -> NorthStarScorecard:
    """
    Computes the composite Alpha-Kelly North Star Score (N_star):
      N_star = CWR_bid * sqrt(365 / D_lock) * exp(-10 * ECE) - 2.0 * max(0, MDD - 0.15)

    Raises ValueError if starting_capital is not a finite real number
    > 0 (SF-BL-9437), if trades is None / non-iterable / a string, dict
    or bare record, or any entry is not a TradeExecutionResult
    (SF-BL-10968), if any trade's action is not a genuine TradeAction
    member (SF-BL-10823), if any executed (non-SKIP) trade's is_settled
    is not a strict bool (SF-BL-11536), if any executed (non-SKIP) trade carries a non-finite
    invested_capital, net_pnl_bid_marked or est_days_to_resolution
    (SF-BL-9475) or a finite est_days_to_resolution <= 0 (SF-BL-9364),
    if any executed (non-SKIP) trade carries invested_capital <= 0
    (SF-BL-11304, any net_pnl_bid_marked including exactly 0.0),
    if any executed (non-SKIP) trade carries a shares_acquired that
    is not a finite real >= 0 or a net_pnl_bid_marked outside its
    feasible payout envelope [-invested_capital,
    shares_acquired - invested_capital] beyond the pre-registered
    epsilon (SF-BL-34956),
    if all_resolved_forecasts is a non-iterable, string, bytes or
    dict container, or any entry is a dict/string/bytes value instead
    of a (forecast, outcome) pair (SF-BL-11049), if any forecast or
    outcome element of all_resolved_forecasts is a bool, non-numeric
    or non-finite value (SF-BL-11535), if any forecast in
    all_resolved_forecasts lies
    outside [0, 1] or any outcome is not 0 or 1 (SF-BL-9139 range gate
    inside compute_empirical_ece_and_brier), or if executed trades are
    scored against an empty resolved forecast set (D3 guard, SF-BL-8898).

    SF-BL-10767: bid-mark freshness. When `scored_at` (epoch seconds)
    is given, every UNSETTLED executed trade whose liquidation-bid
    mark is stale at that instant — its liquidation_bid_as_of is
    missing/invalid, or scored_at - as_of exceeds one snapshot
    cadence (MARK_STALENESS_CADENCE_SECONDS = 2700s, the longest
    documented 15-45 min trading cycle) — is EXCLUDED from all sums
    (CWR, equity/MDD walk, lockup, returns) and reported in the
    scorecard's stale_mark_count / stale_mark_invested; settled
    trades are never excluded. Unprovenanced open marks are counted
    in unprovenanced_mark_count whether or not scored_at is given.
    scored_at=None (the default; every legacy caller) excludes
    nothing and scores bit-identically to pre-fix. Raises ValueError
    if scored_at is a bool, non-numeric or non-finite value.

    SF-BL-34866: leverage reporting. The scorecard's is_levered /
    leverage_ratio / executed_invested_capital report the executed
    book's summed invested_capital against starting_capital
    (is_levered iff the sum strictly exceeds the capital base,
    assessed before the SF-BL-10767 stale-mark partition), and the
    equity walk's drawdown is bounded to [0, 1], so
    max_drawdown_pct always lies in [0, 100] — pre-fix a levered
    book whose equity crossed zero reported MDD above 100%
    (prereg repro: 160.0%, penalty 2.9) with no leverage flag.

    SF-BL-35210: settled-PnL exact-payout flag. The scorecard's
    settled_pnl_mismatch_count / settled_pnl_mismatch_invested report
    the settled executed trades (valid outcome_yes) whose
    net_pnl_bid_marked differs from the exact settlement payout —
    shares_acquired - invested_capital on a win, -invested_capital on
    a loss — by more than SETTLED_PNL_EPSILON (half a cent). Flag-only
    (the prereg's "raises or flags"; flag branch per the SF-BL-34866
    precedent): the trades still score, every pre-existing field is
    bit-identical, and settle-produced books report 0 / 0.0.
    """
    # SF-BL-9437: starting-capital gate at scorecard ENTRY, before any
    # trade gate, sum or equity walk — the value denominators and seeds
    # every figure below, and it must reject even for empty / SKIP-only
    # books (pre-fix a zero capital crashed the return division there
    # too). No-op for every positive finite capital.
    _validate_starting_capital(starting_capital)

    # SF-BL-10767: scored_at gate, with the starting-capital gate at
    # entry — a corrupt scoring instant must reject even for empty /
    # SKIP-only books, before any age can be computed against it.
    # No-op for scored_at=None (legacy scoring) and every finite value.
    _validate_scored_at(scored_at)

    # SF-BL-10968: container/entry gate FIRST among the trade gates,
    # before any validator can iterate or dereference the book — a
    # None / non-iterable container or a non-TradeExecutionResult
    # entry (e.g. a JSON dict record) crashed bare with TypeError /
    # AttributeError pre-fix. Materialises the book to a list so every
    # downstream pass reads the same entries. No-op (bit-identical)
    # for genuine list/tuple books of genuine records.
    trades = _validate_trades_container(trades)

    # SF-BL-10823: action gate next among the trade gates, before the
    # executed/skipped partition below can read any action — the
    # partition's `action != SKIP` test would count a non-TradeAction
    # action as executed (and a plain 'SKIP' string as skipped, by
    # str-Enum equality). Runs over every trade and is a no-op for
    # genuine BUY_YES / BUY_NO / SKIP books.
    _validate_trade_actions(trades)

    # SF-BL-11536: is_settled type gate next among the trade gates,
    # before the executed/skipped partition and the SF-BL-10767 open
    # partition below can read the field by truthiness — a truthy
    # non-bool ('False') would enter the settlement win-rate
    # denominator as settled. Scoped to executed (non-SKIP) trades,
    # the only records whose is_settled the scorer reads (cf.
    # SF-BL-9475 scoping); a no-op for genuine bool books.
    _validate_trade_is_settleds(trades)

    # SF-BL-34997: entry-side consistency gate, after the SF-BL-10823
    # action gate — an executed record whose current action no longer
    # matches the side pinned at construction (entry_action) must not
    # be scored: its stored PnL was (or would be) booked under the
    # wrong side. Scoped to executed (current-action non-SKIP)
    # trades, the SF-BL-11536 precedent — a record reassigned TO
    # SKIP partitions as skipped and its PnL is never summed (the
    # SF-BL-10767 suite does exactly this), while a record flipped
    # FROM its entry side TO another executed side raises the named
    # MISMATCHED_TRADE_ACTION ValueError. Unpinned records pass.
    _validate_scored_trade_action_consistencies(trades)

    # SF-BL-11534: est_days TYPE gate, before the SF-BL-9475 finiteness
    # gate below — a bool / non-numeric lock on an executed trade is a
    # type error (INVALID_EST_DAYS_TYPE), not SF-BL-9475's NON_FINITE
    # value defect. No-op for numeric locks (finite or not) and for
    # SKIP records carrying an offending lock for audit.
    _validate_scored_trade_est_days_types(trades)

    # SF-BL-9475: finiteness gate next among the trade gates, before the
    # invested>0 filter below can hide a NaN-invested trade and before
    # any sum can absorb a non-finite PnL or lockup. No-op for
    # all-finite books.
    _validate_scored_trade_fields(trades)

    # SF-BL-9364: lockup sign gate, immediately after the finiteness gate
    # (which owns non-finite values and their error identity) and before
    # the invested>0 filter / lockup weighting below can pool a finite
    # non-positive lockup into avg_lockup. No-op for all-positive books.
    _validate_scored_trade_lockups(trades)

    # SF-BL-11304: invested-sign gate, immediately after the lockup gate
    # (which owns a trade tripping both) and before the invested>0
    # executed filter below can silently drop a finite non-positive
    # invested trade — counted as neither executed nor skipped, and any
    # PnL it carries excluded from every sum. Round-2 scope: raises on
    # every non-SKIP invested <= 0, including PnL exactly 0.0, so no
    # returning scorecard can break executed+skipped == total.
    # No-op for all-positive books.
    _validate_scored_trade_invested(trades)

    # SF-BL-34956: payout-envelope gate, immediately after the
    # invested-sign gate (which establishes every executed invested is
    # finite and positive, with PnL finite per SF-BL-9475) and before
    # the leverage assessment / executed partition below can sum an
    # impossible PnL into CWR / MDD. Validates shares_acquired (no
    # earlier scorer gate covers it) as part of the envelope check.
    # No-op for in-envelope books: bit-identical.
    _validate_scored_trade_pnl_envelopes(trades)

    # SF-BL-35210: exact settled-payout flag, immediately after the
    # SF-BL-34956 envelope gate (which establishes every executed PnL /
    # invested is finite, invested > 0, and shares_acquired is a finite
    # real >= 0) and before the leverage assessment / executed
    # partition below. The envelope is only a RANGE: a settled trade's
    # PnL is exactly shares - invested on a win or -invested on a loss,
    # and any in-range figure further than the pre-registered half-cent
    # epsilon from that payout is counted in the scorecard's
    # settled_pnl_mismatch_count / settled_pnl_mismatch_invested
    # (flag branch of the prereg's "raises or flags" — see the
    # SF-BL-35210 block above _assess_settled_pnl_mismatches). Open
    # trades and settled records without a valid outcome are not
    # counted; settle-produced books differ by 0.0, flag 0 / 0.0, and
    # every pre-existing field scores bit-identically.
    settled_pnl_mismatch_count, settled_pnl_mismatch_invested = \
        _assess_settled_pnl_mismatches(trades)

    # SF-BL-34866: leverage assessment, immediately after the
    # SF-BL-35210 exact-payout gate (which follows the SF-BL-34956
    # envelope gate and the invested-sign gate, establishing every
    # summed invested is finite and positive) and
    # before the executed partition / equity
    # walk below. The submitted book's summed executed invested is
    # compared with its capital base and reported on the scorecard
    # (is_levered / leverage_ratio / executed_invested_capital), and
    # the walk below bounds its drawdown to [0, 100%] — pre-fix a
    # levered book crossed zero and reported MDD 160.0% unflagged.
    # Assessed on the executed book BEFORE the SF-BL-10767 stale-mark
    # partition: a scoring exclusion does not release capital. Every
    # pre-existing numeric field of a book whose raw drawdown stayed
    # within [0, 100%] is bit-identical.
    executed_invested_capital, leverage_ratio, is_levered = \
        _assess_book_leverage(trades, starting_capital)

    # SF-BL-11049: panel-CONTAINER gate, immediately before the
    # SF-BL-11535 element gate — a non-iterable container (123) died
    # with a bare TypeError inside that gate's enumerate(), a string
    # container ('abc') was iterated character by character, and a
    # one-shot generator died on len() in the ECE computation.
    # Materialises the panel to a list so the element gate, the D3
    # guard and compute_empirical_ece_and_brier all read the same
    # entries. None passes through (the D3 guard owns it) and every
    # genuine list/tuple/set panel is bit-identical.
    all_resolved_forecasts = _validate_resolved_forecasts_container(
        all_resolved_forecasts)  # type: ignore[assignment]

    # SF-BL-11535: panel-element type gate, before the D3 guard / ECE
    # computation below can read a bool as certainty or die on a string
    # element with a bare TypeError. No-op for empty/None panels (the
    # D3 guard owns those) and for every numeric panel, which passes
    # bit-identically.
    _validate_resolved_forecast_panel(all_resolved_forecasts)

    executed = [t for t in trades if t.action != TradeAction.SKIP and t.invested_capital > 0]
    skipped = [t for t in trades if t.action == TradeAction.SKIP]

    # --- D3 guard (mirrors nstar.py score_account, ECE_SPEC S5.2/D3) -----
    # An empty resolved forecast set with executed trades must never reach
    # compute_empirical_ece_and_brier: its empty-input convention returns
    # (0.0, 0.0), which below would turn into calibration_multiplier
    # exp(0) = 1.0 and a positive N_star for an ungraded book -- the silent
    # pass the spec forbids (SF-BL-8898). Zero-executed-trade books are
    # exempt (N_star is 0.0 by definition there, D4 analogue) and every
    # non-empty call is untouched: this guard is a no-op unless the
    # resolved set is empty/None while executed trades exist.
    if executed and not all_resolved_forecasts:
        raise ValueError(
            "NO_ELIGIBLE_ROWS: evaluate_fleet_north_star called with "
            "executed trades but an empty resolved forecast set; ECE is "
            "undefined, refusing to return calibration_multiplier 1.0 "
            "(ECE_SPEC S5.2/D3)")

    # SF-BL-10767: freshness partition of the executed set. Only
    # UNSETTLED trades are mark-dependent (a settled trade's PnL pays
    # $1/$0 and never read the bid), so only they can be stale.
    # unprovenanced_mark_count is computed unconditionally (flag-only
    # visibility for legacy callers); the exclusion itself applies
    # only when scored_at is given — with scored_at=None `scored` is
    # `executed` and every sum below is bit-identical to pre-fix.
    _open_executed = [t for t in executed if not t.is_settled]
    unprovenanced_mark_count = sum(
        1 for t in _open_executed if _bid_as_of(t) is None)
    if scored_at is not None:
        _stale = [t for t in _open_executed
                  if _mark_is_stale(t, float(scored_at))]
        _stale_ids = {id(t) for t in _stale}
        scored = [t for t in executed if id(t) not in _stale_ids]
        stale_mark_count = len(_stale)
        stale_mark_invested = sum(t.invested_capital for t in _stale)
    else:
        scored = executed
        stale_mark_count = 0
        stale_mark_invested = 0.0

    total_invested = sum(t.invested_capital for t in scored)
    total_pnl = sum(t.net_pnl_bid_marked for t in scored)
    cwr_bid = (total_pnl / total_invested) if total_invested > 0 else 0.0

    # Sequential equity curve and Max Drawdown (MDD)
    equity = starting_capital
    peak_equity = starting_capital
    max_dd = 0.0
    trade_returns: List[float] = []

    for t in scored:
        equity += t.net_pnl_bid_marked
        peak_equity = max(peak_equity, equity)
        # SF-BL-34866: drawdown bounded to [0, 1] — the raw fraction
        # exceeds 1.0 once equity crosses zero (the 160% repro).
        dd = _bounded_drawdown(peak_equity, equity)
        max_dd = max(max_dd, dd)
        trade_returns.append(t.net_pnl_bid_marked / t.invested_capital)

    # Capital-weighted lockup duration D_lock and Velocity Multiplier
    if total_invested > 0:
        avg_lockup = sum(t.invested_capital * t.est_days_to_resolution for t in scored) / total_invested
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
    # (settled trades are never stale-excluded, so `scored` and
    # `executed` yield the same settled set — SF-BL-10767)
    settled_trades = [t for t in scored if t.is_settled]
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
        stale_mark_count=stale_mark_count,
        stale_mark_invested=round(stale_mark_invested, 2),
        unprovenanced_mark_count=unprovenanced_mark_count,
        is_levered=is_levered,
        leverage_ratio=round(leverage_ratio, 6),
        executed_invested_capital=round(executed_invested_capital, 2),
        settled_pnl_mismatch_count=settled_pnl_mismatch_count,
        settled_pnl_mismatch_invested=round(
            settled_pnl_mismatch_invested, 2),
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

        _agg = aggregate_fleet_forecasts(sub_agents)
        cal_p, dispersion = _agg[0], _agg[1]
        resolved_forecasts.append((cal_p, outcome_yes))

        # 1. Alpha-Kelly Disciplined Execution
        # SF-BL-10761: thread the aggregate's panel provenance into
        # Gate 2C. The benchmark panels are clean (0 drops, 4 kept),
        # so Gate 2C passes them untouched and output is unchanged.
        d_trade = evaluate_and_size_opportunity(
            book=book,
            calibrated_prob_yes=cal_p,
            inter_agent_dispersion=dispersion,
            portfolio_equity=disciplined_equity,
            cluster_exposure_used=cluster_used,
            panel_n_dropped=_agg.provenance.n_dropped,
            panel_n_kept=_agg.provenance.n_kept,
        )
        # SF-BL-3660: pass the book so an OPEN mark walks the bid ladder;
        # the benchmark fixtures are settled, so settled output is unchanged.
        # SF-BL-8890: pass the cluster ledger so settlement releases the
        # trade's cap fraction (evaluate also stored the reference on the
        # trade; passing it explicitly documents the lifecycle here).
        d_trade = settle_or_mark_trade(d_trade, is_settled=is_settled, outcome_yes=outcome_yes, exit_book=book, cluster_exposure_used=cluster_used)
        disciplined_equity += d_trade.net_pnl_bid_marked
        disciplined_trades.append(d_trade)

        # 2. Naive Brier-Optimized Baseline (bets flat $1,200 whenever p != mid_price, ignoring gates & slippage)
        # SF-BL-3549 declared rule: the naive baseline still ignores
        # gates and slippage on a book it can price, but it may not
        # manufacture a price — on a book lacking a live, tradeably-
        # priced ladder on either side there is no observable mid to
        # pick a side from and no observable liquidation bid to mark
        # at, so the naive record is a zero-size SKIP carrying the
        # declared mid / liquidation bid of 0.0, never the 0.50 /
        # 0.255 / 0.745 phantom mid or the 0.01 sentinel mark. All
        # benchmark fixture books are two-sided, so the benchmark
        # output is unchanged.
        if not (_has_tradeable_ladder(book.yes_asks)
                and _has_tradeable_ladder(book.yes_bids)):
            n_trade = TradeExecutionResult(
                market_id=book.market_id,
                title=book.title,
                domain=book.domain,
                factor_cluster=book.factor_cluster,
                action=TradeAction.SKIP,
                calibrated_prob_yes=cal_p,
                market_mid_prob=0.0,
                exec_price=0.0,
                liquidation_bid_price=0.0,
                kl_divergence_nats=0.0,
                expected_log_growth=0.0,
                kelly_fraction_raw=0.0,
                allocated_fraction=0.0,
                invested_capital=0.0,
                shares_acquired=0.0,
                est_days_to_resolution=book.est_days_to_resolution,
                skip_reason=(
                    "NO_OBSERVABLE_PRICE (SF-BL-3549 declared rule: book "
                    "lacks a live tradeable ladder on one or both sides; "
                    "no mid or liquidation bid is fabricated from the "
                    "0.01/0.99 sentinels)"
                ),
            )
            n_trade = settle_or_mark_trade(n_trade, is_settled=is_settled, outcome_yes=outcome_yes, exit_book=book)
            naive_trades.append(n_trade)
            continue
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
            market_mid_prob=_declared_mid_prob(book),
            exec_price=n_q_exec,
            liquidation_bid_price=_declared_liquidation_bid(book, naive_side),
            # SF-BL-10767: this record's bid was read from the entry
            # book directly (the baseline bypasses evaluate), so it
            # carries the same entry provenance evaluate would stamp.
            liquidation_bid_as_of=_snapshot_ts(book),
            liquidation_bid_source=BID_SOURCE_ENTRY_BOOK,
            kl_divergence_nats=kl_divergence_binary(naive_p_side, n_q_exec) if n_q_exec > 0 else 0.0,
            expected_log_growth=0.0,
            kelly_fraction_raw=0.12,
            allocated_fraction=n_inv / starting_cap,
            invested_capital=n_inv,
            shares_acquired=n_shares,
            est_days_to_resolution=book.est_days_to_resolution,
        )
        n_trade = settle_or_mark_trade(n_trade, is_settled=is_settled, outcome_yes=outcome_yes, exit_book=book)
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