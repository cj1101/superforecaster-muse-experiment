#!/usr/bin/env python3
"""SF-BL-11189 regression tests: the exact prereg fixture and its variants.

Fixture: ask 0.55 / bid 0.60, fee 0.01, p=0.80 — q_exec would be 0.5555,
liquidation bid 0.60 above entry, Gate D round-trip cost -0.0385.
Asserts the canonical engine's decision gate SKIPs it (and locked /
mirror / mutated variants) with zero invested and cluster untouched,
that the hazard arithmetic is as claimed, and that a clean 1-cent-spread
control at the same ask/p is NOT blocked by the crossed gate.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "workspace/superforecaster"))
from superforecaster.alpha_kelly import (  # noqa: E402
    MarketDomain, OrderBookLevel, OrderBookSnapshot, TradeAction,
    evaluate_and_size_opportunity, round_trip_cost,
)

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"PASS {name}")
    else:
        FAIL += 1; print(f"FAIL {name}")


def book(asks, bids, fee=0.01):
    return OrderBookSnapshot(
        market_id="SF-BL-11189-fixture", title="fixture",
        domain=MarketDomain.PREDICTION_MARKET, factor_cluster="TEST",
        est_days_to_resolution=30,
        yes_asks=[OrderBookLevel(p, s) for p, s in asks],
        yes_bids=[OrderBookLevel(p, s) for p, s in bids], fee_rate=fee)


def ev(b, p):
    cl = {}
    return evaluate_and_size_opportunity(b, p, 0.0, 10000.0, cl), cl


# 1. Hazard arithmetic (the TASK.md claim, recomputed)
check("q_exec = 0.55*1.01 = 0.5555", abs(0.55 * 1.01 - 0.5555) < 1e-9)
rt, qexit, fill = round_trip_cost(book([(0.55, 100000)], [(0.60, 100000)]),
                                  TradeAction.BUY_YES, 1000.0, 0.5555)
check(f"Gate D round-trip cost negative (got {rt:.4f})", rt < 0 and fill == 1.0)
check("exit net 0.5940 > q_exec 0.5555", abs(qexit - 0.5940) < 1e-9 and qexit > 0.5555)

# 2. Exact fixture SKIPs CROSSED_BOOK, zero invested, cluster untouched
r, cl = ev(book([(0.55, 100000)], [(0.60, 100000)]), 0.80)
check("fixture SKIPs", r.action == TradeAction.SKIP)
check("fixture reason is CROSSED_BOOK", (r.skip_reason or "").startswith("CROSSED_BOOK"))
check("fixture invested 0 / shares 0", r.invested_capital == 0.0 and r.shares_acquired == 0.0)
check("fixture cluster untouched", cl == {})
check("fixture stores no phantom mark (liq 0.0, mid 0.0)",
      r.liquidation_bid_price == 0.0 and r.market_mid_prob == 0.0)

# 3. Locked variant (bid == ask) at the fixture prices
r, cl = ev(book([(0.55, 100000)], [(0.55, 100000)]), 0.80)
check("locked 0.55/0.55 SKIPs CROSSED_BOOK",
      r.action == TradeAction.SKIP and (r.skip_reason or "").startswith("CROSSED_BOOK")
      and r.invested_capital == 0.0)

# 4. Mirror side: same crossed book at p=0.20 must not BUY_NO off 1-bid
r, cl = ev(book([(0.55, 100000)], [(0.60, 100000)]), 0.20)
check("mirror p=0.20 SKIPs CROSSED_BOOK, no BUY_NO",
      r.action == TradeAction.SKIP and (r.skip_reason or "").startswith("CROSSED_BOOK"))

# 5. Full posterior grid on the fixture book: never invests
ok = True
for p in (0.05, 0.20, 0.50, 0.80, 0.95):
    r, cl = ev(book([(0.55, 100000)], [(0.60, 100000)]), p)
    ok = ok and r.invested_capital == 0.0 and r.action == TradeAction.SKIP and cl == {}
check("p-grid {0.05..0.95} all SKIP zero-invested", ok)

# 6. Mutated book: constructed clean, bids mutated crossed post-construction
b = book([(0.55, 100000)], [(0.54, 100000)])
b.yes_bids = [OrderBookLevel(0.60, 100000)]
r, cl = ev(b, 0.80)
check("mutated-crossed book SKIPs CROSSED_BOOK",
      r.action == TradeAction.SKIP and (r.skip_reason or "").startswith("CROSSED_BOOK"))

# 7. Dead rung cannot fabricate a crossing at these prices:
#    zero-size 0.55 ask in front of live 0.62 ask vs 0.60 bid is NOT crossed
r, cl = ev(book([(0.55, 0), (0.62, 100000)], [(0.60, 100000)]), 0.80)
check("dead ask rung does not fabricate CROSSED_BOOK",
      not (r.skip_reason or "").startswith("CROSSED_BOOK"))

# 8. Clean control: 0.55 ask / 0.54 bid at p=0.80 trades (gate does not over-block)
r, cl = ev(book([(0.55, 100000)], [(0.54, 100000)]), 0.80)
check("clean control BUY_YES", r.action == TradeAction.BUY_YES and r.invested_capital > 0)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
