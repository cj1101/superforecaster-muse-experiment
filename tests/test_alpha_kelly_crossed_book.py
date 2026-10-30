"""SF-BL-9365 — crossed/locked books must SKIP as CROSSED_BOOK.

Prereg: evaluate_and_size_opportunity SKIPs with an explicit crossed-book
reason whenever best_yes_ask <= best_yes_bid and the repro fixture
produces zero invested capital; kill if a census of every
OrderBookSnapshot constructor proves crossed inputs are impossible at
construction (CONSTRUCTOR_CENSUS.md: NOT met — 45 literal sites, no
cross-ladder check in __post_init__, mutable dataclass).

No-pytest runner, repo convention. Run: python3 test_crossed_book.py
"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster"))
from superforecaster import alpha_kelly as ak

PASSED = []


def check(name, cond, detail=""):
    assert cond, f"FAIL {name}: {detail}"
    PASSED.append(name)
    print(f"  ok {name}")


def _book(asks, bids, fee=0.01, domain=ak.MarketDomain.PREDICTION_MARKET):
    return ak.OrderBookSnapshot(
        market_id="T-9365", title="crossed", domain=domain,
        factor_cluster="C9365", est_days_to_resolution=7.0,
        yes_asks=[ak.OrderBookLevel(price=p, size_shares=s) for p, s in asks],
        yes_bids=[ak.OrderBookLevel(price=p, size_shares=s) for p, s in bids],
        fee_rate=fee)


def _eval(book, p, equity=10000.0):
    cluster = {}
    r = ak.evaluate_and_size_opportunity(book, p, 0.05, equity, cluster)
    return r, cluster


def _is_clean_crossed_skip(r, cluster):
    return (
        r.action == ak.TradeAction.SKIP
        and r.skip_reason is not None and r.skip_reason.startswith("CROSSED_BOOK")
        and r.invested_capital == 0.0 and r.shares_acquired == 0.0
        and r.exec_price == 0.0 and r.liquidation_bid_price == 0.0
        and r.kelly_fraction_raw == 0.0 and r.allocated_fraction == 0.0
        and cluster == {}
    )


# ---- 1. Prereg fixture: asks 0.40 / bids 0.60, fee 0.01, p=0.62 ---------------
# Pre-fix: BUY_YES invested 1268.46 at q_exec 0.404, mid 0.50, no_ask 0.40.
r, cl = _eval(_book([(0.40, 100000)], [(0.60, 100000)]), 0.62)
check("prereg fixture SKIPs CROSSED_BOOK with zero invested",
      _is_clean_crossed_skip(r, cl), r)
check("prereg reason names both best prices",
      "0.4" in r.skip_reason and "0.6" in r.skip_reason, r.skip_reason)
check("prereg SKIP mid is 0.0, NOT the crossed mid 0.50",
      r.market_mid_prob == 0.0, r.market_mid_prob)

# ---- 2. Crossed mirror side: p=0.30 favoured BUY_NO pre-fix ($1,500) ----------
r, cl = _eval(_book([(0.40, 100000)], [(0.60, 100000)]), 0.30)
check("crossed BUY_NO side SKIPs CROSSED_BOOK", _is_clean_crossed_skip(r, cl), r)

# ---- 3. Locked book: ask == bid (prereg <= includes equality) ------------------
r, cl = _eval(_book([(0.50, 100000)], [(0.50, 100000)]), 0.62)
check("locked book (ask == bid) SKIPs CROSSED_BOOK", _is_clean_crossed_skip(r, cl), r)
r, cl = _eval(_book([(0.50, 100000)], [(0.50, 100000)]), 0.30)
check("locked book SKIPs CROSSED_BOOK on NO side too", _is_clean_crossed_skip(r, cl), r)

# ---- 4. Crossed/locked across the whole posterior grid --------------------------
for asks, bids in ([(0.40, 100000)], [(0.60, 100000)]), ([(0.50, 100000)], [(0.50, 100000)]):
    for p in (0.05, 0.20, 0.50, 0.62, 0.80, 0.95):
        r, cl = _eval(_book(asks, bids), p)
        assert _is_clean_crossed_skip(r, cl), (asks, bids, p, r)
check("crossed/locked SKIP CROSSED_BOOK on full p-grid", True)

# ---- 5. Deep ladders: crossing is a top-of-book relation ------------------------
r, cl = _eval(_book([(0.40, 100000), (0.70, 100000)],
                    [(0.60, 100000), (0.30, 100000)]), 0.62)
check("multi-rung crossed top SKIPs CROSSED_BOOK", _is_clean_crossed_skip(r, cl), r)
# A 1-cent inversion is still crossed.
r, cl = _eval(_book([(0.50, 100000)], [(0.51, 100000)]), 0.80)
check("1-cent crossed book SKIPs CROSSED_BOOK", _is_clean_crossed_skip(r, cl), r)

# ---- 6. Mutated-crossed book: consumption gate catches the bypass ---------------
b = _book([(0.62, 100000)], [(0.58, 100000)])
b.yes_bids = [ak.OrderBookLevel(price=0.70, size_shares=100000)]
r, cl = _eval(b, 0.80)
check("post-construction mutation to crossed SKIPs CROSSED_BOOK",
      _is_clean_crossed_skip(r, cl), r)

# ---- 7. Dead / invalid rungs neither fabricate nor mask a crossing --------------
# Dead (zero-size) 0.40 ask in front of live 0.62 ask vs 0.60 bid: NOT crossed.
r, cl = _eval(_book([(0.40, 0), (0.62, 100000)], [(0.60, 100000)]), 0.80)
check("dead ask rung does not fabricate crossing (trades BUY_YES $1,500)",
      r.action == ak.TradeAction.BUY_YES and r.invested_capital == 1500.0, r)
# Dead 0.88 bid in front of live 0.30 bid vs 0.40 ask: NOT crossed (Gate D skip).
r, cl = _eval(_book([(0.40, 100000)], [(0.88, 0), (0.30, 100000)]), 0.62)
check("dead bid rung does not fabricate crossing (keeps ROUND_TRIP_GATE_D)",
      r.action == ak.TradeAction.SKIP and r.skip_reason.startswith("ROUND_TRIP_GATE_D"), r)
# Invalid-price ask rung in front of live 0.62 ask: NOT crossed.
r, cl = _eval(_book([(-0.10, 100000), (0.62, 100000)], [(0.60, 100000)]), 0.80)
check("invalid-price ask rung does not fabricate crossing (trades BUY_YES)",
      r.action == ak.TradeAction.BUY_YES, r)
# ...but a dead rung must not MASK a real crossing at the live top either.
r, cl = _eval(_book([(0.62, 0), (0.40, 100000)], [(0.60, 100000)]), 0.62)
check("dead rung does not mask a live crossed top", _is_clean_crossed_skip(r, cl), r)

# ---- 8. Crossed CRYPTO book: book-integrity gate precedes Gate 1 ----------------
b = _book([(0.40, 100000)], [(0.60, 100000)], domain=ak.MarketDomain.CRYPTO)
b.on_chain_verified = False
r, cl = _eval(b, 0.62)
check("crossed CRYPTO book SKIPs CROSSED_BOOK (before CryptoBench gate)",
      _is_clean_crossed_skip(r, cl), r)

# ---- 9. SKIP marks at zero PnL (no phantom position) -----------------------------
r, cl = _eval(_book([(0.40, 100000)], [(0.60, 100000)]), 0.62)
marked = ak.settle_or_mark_trade(r, is_settled=False)
check("crossed SKIP marks at 0 PnL", marked.net_pnl_bid_marked == 0.0, marked)

# ---- 10. Uncrossed controls are byte-identical -----------------------------------
# 1-cent spread still trades, same numbers as pre-fix.
r, cl = _eval(_book([(0.51, 100000)], [(0.50, 100000)]), 0.80)
check("1-cent spread control still executes BUY_YES $1,500 at liq 0.50 / mid 0.505",
      r.action == ak.TradeAction.BUY_YES and r.invested_capital == 1500.0
      and r.liquidation_bid_price == 0.50 and r.market_mid_prob == 0.505, r)
# Two-sided no-edge book keeps its old reason.
r, cl = _eval(_book([(0.50, 100000)], [(0.48, 100000)]), 0.51)
check("two-sided no-edge keeps INSUFFICIENT_NET_EDGE",
      r.action == ak.TradeAction.SKIP
      and r.skip_reason.startswith("INSUFFICIENT_NET_EDGE"), r)
# One-sided keeps ONE_SIDED_BOOK (Gate 0.7 precedes 0.8; sentinels uncrossed).
r, cl = _eval(_book([(0.50, 100000)], []), 0.80)
check("one-sided keeps ONE_SIDED_BOOK", r.skip_reason.startswith("ONE_SIDED_BOOK"), r)
# Empty keeps EMPTY_BOOK.
r, cl = _eval(_book([(0.40, 0)], [(0.88, 0)]), 0.65)
check("all-dead book keeps EMPTY_BOOK", r.skip_reason.startswith("EMPTY_BOOK"), r)
# NaN-rung book keeps INVALID_EQUITY (Gate 0.6 precedes 0.8).
r, cl = _eval(_book([(float("nan"), 100000), (0.40, 100000)], [(0.60, 100000)]), 0.62)
check("NaN-rung crossed book keeps INVALID_EQUITY",
      r.skip_reason.startswith("INVALID_EQUITY"), r)
# Unsorted keeps UNSORTED_LADDER (Gate 0.45 precedes 0.8).
b = _book([(0.50, 100000)], [(0.48, 100000)])
b.yes_asks = [ak.OrderBookLevel(0.62, 100000), ak.OrderBookLevel(0.40, 100000)]
r, cl = _eval(b, 0.80)
check("unsorted book keeps UNSORTED_LADDER", r.skip_reason.startswith("UNSORTED_LADDER"), r)

print(f"\n{len(PASSED)} checks passed — SF-BL-9365 crossed-book gate green.")
