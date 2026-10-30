"""SF-BL-10760 — one-sided books must SKIP as ONE_SIDED_BOOK.

Prereg: a book lacking a live ladder on the chosen side (or its exit
side) must SKIP with a named ONE_SIDED_BOOK reason regardless of Gate D
configuration, and SKIP records must not carry a sentinel-derived mid.
Kill (census, separate deliverable): <1% of gate-passing forward-S2
candidates one-sided — measured 34.7%, NOT triggered (census_s2.py).

No-pytest runner, repo convention. Run: python3 test_one_sided_book.py
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


def _book(asks, bids, fee=0.01):
    return ak.OrderBookSnapshot(
        market_id="T-10760", title="one-sided", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C10760", est_days_to_resolution=7.0,
        yes_asks=[ak.OrderBookLevel(price=p, size_shares=s) for p, s in asks],
        yes_bids=[ak.OrderBookLevel(price=p, size_shares=s) for p, s in bids],
        fee_rate=fee)


def _eval(book, p, gate_d=None):
    cfg = ak.FleetRiskConfig(max_round_trip_cost=gate_d)
    cluster = {}
    r = ak.evaluate_and_size_opportunity(book, p, 0.05, 100_000.0, cluster, cfg)
    return r, cluster


def _is_clean_one_sided_skip(r, cluster):
    return (
        r.action == ak.TradeAction.SKIP
        and r.skip_reason is not None and r.skip_reason.startswith("ONE_SIDED_BOOK")
        and r.invested_capital == 0.0 and r.shares_acquired == 0.0
        and r.exec_price == 0.0 and r.liquidation_bid_price == 0.0
        and r.kelly_fraction_raw == 0.0 and r.allocated_fraction == 0.0
        and cluster == {}
    )


# ---- 1. Prereg fixture: asks-only, Gate D disabled ---------------------------
r, cl = _eval(_book([(0.50, 100000)], []), 0.80, gate_d=None)
check("prereg asks-only GateD-off SKIPs ONE_SIDED_BOOK", _is_clean_one_sided_skip(r, cl), r)
check("prereg asks-only mid is NOT sentinel-derived 0.255", r.market_mid_prob == 0.0, r.market_mid_prob)
check("prereg reason names missing side", "yes_bids" in r.skip_reason, r.skip_reason)

# ---- 2. Same book, Gate D enabled: reason is ONE_SIDED, not Gate D -----------
r, cl = _eval(_book([(0.50, 100000)], []), 0.80, gate_d=0.05)
check("asks-only GateD-on SKIPs ONE_SIDED_BOOK (not ROUND_TRIP_GATE_D)",
      _is_clean_one_sided_skip(r, cl), r)

# ---- 3. Bids-only mirror ------------------------------------------------------
r, cl = _eval(_book([], [(0.50, 100000)]), 0.20, gate_d=None)
check("bids-only GateD-off SKIPs ONE_SIDED_BOOK", _is_clean_one_sided_skip(r, cl), r)
check("bids-only reason names missing side", "yes_asks" in r.skip_reason, r.skip_reason)
check("bids-only mid is NOT sentinel-derived 0.745", r.market_mid_prob == 0.0, r.market_mid_prob)
r, cl = _eval(_book([], [(0.50, 100000)]), 0.20, gate_d=0.05)
check("bids-only GateD-on SKIPs ONE_SIDED_BOOK", _is_clean_one_sided_skip(r, cl), r)

# ---- 4. One-sided across the whole posterior grid, both Gate D configs -------
for gate_d in (None, 0.05):
    for p in (0.05, 0.20, 0.50, 0.80, 0.95):
        r, cl = _eval(_book([(0.50, 100000)], []), p, gate_d=gate_d)
        assert _is_clean_one_sided_skip(r, cl), (gate_d, p, r)
        r, cl = _eval(_book([], [(0.50, 100000)]), p, gate_d=gate_d)
        assert _is_clean_one_sided_skip(r, cl), (gate_d, p, r)
check("one-sided SKIPs ONE_SIDED_BOOK on p-grid x GateD {off,on}", True)

# ---- 5. Dead exit/entry ladders count as missing ------------------------------
# Exit side present only as zero-size rungs (SF-BL-9398 dead levels).
r, cl = _eval(_book([(0.50, 100000)], [(0.48, 0)]), 0.80, gate_d=None)
check("zero-size-only bids = one-sided", _is_clean_one_sided_skip(r, cl), r)
# Exit side present only at invalid prices (SF-BL-9542 non-tradeable).
for bad_bid in (-0.30, 1.50, 0.0, 1.0):
    r, cl = _eval(_book([(0.50, 100000)], [(bad_bid, 100000)]), 0.80, gate_d=None)
    assert _is_clean_one_sided_skip(r, cl), (bad_bid, r)
check("invalid-price-only bids = one-sided", True)
for bad_ask in (-0.10, 2.0, 0.0, 1.0):
    r, cl = _eval(_book([(bad_ask, 100000)], [(0.48, 100000)]), 0.20, gate_d=None)
    assert _is_clean_one_sided_skip(r, cl), (bad_ask, r)
check("invalid-price-only asks = one-sided", True)
# A dead rung in FRONT of a live rung on the exit side is NOT one-sided.
r, cl = _eval(_book([(0.50, 100000)], [(0.88, 0), (0.48, 100000)]), 0.80, gate_d=None)
check("live exit behind dead rung still trades", r.action == ak.TradeAction.BUY_YES
      and r.liquidation_bid_price == 0.48, r)

# ---- 6. No sentinel liquidation / no -$14,703 mark can arise ------------------
r, cl = _eval(_book([(0.50, 100000)], []), 0.80, gate_d=None)
marked = ak.settle_or_mark_trade(r, is_settled=False)
check("one-sided SKIP marks at 0 PnL (no sentinel mark)", marked.net_pnl_bid_marked == 0.0, marked)

# ---- 7. Two-sided books unaffected ---------------------------------------------
r, cl = _eval(_book([(0.50, 100000)], [(0.48, 100000)]), 0.80, gate_d=None)
check("two-sided control still executes BUY_YES $15,000 at real bid",
      r.action == ak.TradeAction.BUY_YES and r.invested_capital == 15000.0
      and r.liquidation_bid_price == 0.48 and r.market_mid_prob == 0.49, r)
r, cl = _eval(_book([(0.50, 1000)], [(0.48, 1000)], fee=0.0), 0.70, gate_d=None)
check("small two-sided control unchanged (Kelly 0.40 / $500)",
      r.action == ak.TradeAction.BUY_YES and abs(r.kelly_fraction_raw - 0.40) < 1e-9
      and r.invested_capital == 500.0, r)
# Two-sided but no edge still SKIPs for the OLD reason, not ONE_SIDED.
r, cl = _eval(_book([(0.50, 100000)], [(0.48, 100000)]), 0.51, gate_d=None)
check("two-sided no-edge keeps INSUFFICIENT_NET_EDGE", r.action == ak.TradeAction.SKIP
      and r.skip_reason.startswith("INSUFFICIENT_NET_EDGE"), r)
# Both sides dead remains EMPTY_BOOK (Gate 0.5 precedes Gate 0.7).
r, cl = _eval(_book([(0.40, 0)], [(0.88, 0)]), 0.65, gate_d=None)
check("all-dead book remains EMPTY_BOOK", r.skip_reason.startswith("EMPTY_BOOK"), r)
# NaN-poisoned book keeps its pinned INVALID_EQUITY reason (Gate 0.6 precedes 0.7).
r, cl = _eval(_book([(0.50, 1000)], [(float("nan"), 1000)]), 0.80, gate_d=None)
check("NaN-rung book keeps INVALID_EQUITY", r.skip_reason.startswith("INVALID_EQUITY"), r)

print(f"\n{len(PASSED)} checks passed")
