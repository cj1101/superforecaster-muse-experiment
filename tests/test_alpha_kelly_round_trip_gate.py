"""SF-BL-7432 — Gate D (round-trip) tests for alpha_kelly.py.

Fixtures are hand-computed. Run: python3 test_round_trip_gate.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster"))
from superforecaster import alpha_kelly as ak

PASS = FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")

def book(mid_ask, ask_sizes, bids, fee=0.01, asks=None):
    return ak.OrderBookSnapshot(
        market_id="T-7432", title="fixture", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C7432", est_days_to_resolution=7.0,
        yes_asks=asks if asks is not None else
                 [ak.OrderBookLevel(price=p, size_shares=s) for p, s in ask_sizes],
        yes_bids=[ak.OrderBookLevel(price=p, size_shares=s) for p, s in bids],
        fee_rate=fee)

# Tight deep book: ask 0.50 x100k, bid 0.49 x100k. q_exec=0.505,
# exit VWAP 0.49, q_exit=0.4851, rt=0.0199 <= 0.05 -> trades.
tight = book(0.50, [(0.50, 100000)], [(0.49, 100000)])
# Wide thin book: ask 0.50 x100k, bids walk 0.44 x100 then 0.30.
# Position shares >> 100, so exit VWAP ~0.30x, rt >> 0.05 -> veto.
wide = book(0.50, [(0.50, 100000)], [(0.44, 100), (0.30, 100000)])

# 1. size_walked_exit_bid hand-checks
v, f = ak.size_walked_exit_bid(tight, ak.TradeAction.BUY_YES, 1000.0)
check("walk tight VWAP", abs(v - 0.49) < 1e-9 and abs(f - 1.0) < 1e-9, f"{v} {f}")
v, f = ak.size_walked_exit_bid(wide, ak.TradeAction.BUY_YES, 1100.0)
exp = (100 * 0.44 + 1000 * 0.30) / 1100
check("walk wide VWAP", abs(v - exp) < 1e-9 and abs(f - 1.0) < 1e-9, f"{v} vs {exp}")
v, f = ak.size_walked_exit_bid(wide, ak.TradeAction.BUY_YES, 200000.0)
check("walk insufficient depth fill_fraction",
      abs(f - 100100 / 200000) < 1e-9 and v > 0, f"{v} {f}")
v, f = ak.size_walked_exit_bid(book(0.5, [(0.5, 100)], []), ak.TradeAction.BUY_YES, 10.0)
check("walk no bids -> (0,0)", v == 0.0 and f == 0.0, f"{v} {f}")
# BUY_NO exit derives from asks: asks 0.50 -> NO bid 0.50
v, f = ak.size_walked_exit_bid(tight, ak.TradeAction.BUY_NO, 500.0)
check("walk BUY_NO derived bid", abs(v - 0.50) < 1e-9, f"{v}")

# 2. round_trip_cost hand-check: q_exec 0.505 on tight
rt, qx, ff = ak.round_trip_cost(tight, ak.TradeAction.BUY_YES, 1000.0, 0.505)
check("rt_cost tight", abs(rt - (0.505 - 0.49 * 0.99)) < 1e-9, f"{rt}")

# 3. evaluate: tight book trades, cluster written
cl = {}
r = ak.evaluate_and_size_opportunity(tight, 0.80, 0.05, 10000.0, cl)
check("tight book BUY_YES", r.action == ak.TradeAction.BUY_YES, f"{r.action} {r.skip_reason}")
check("tight cluster written", cl.get("C7432", 0) > 0, f"{cl}")

# 4. evaluate: wide book vetoed by Gate D, zero size, cluster untouched
cl = {}
r = ak.evaluate_and_size_opportunity(wide, 0.80, 0.05, 10000.0, cl)
check("wide book SKIP", r.action == ak.TradeAction.SKIP, f"{r.action}")
check("wide skip reason Gate D", bool(r.skip_reason) and r.skip_reason.startswith("ROUND_TRIP_GATE_D"),
      f"{r.skip_reason}")
check("wide zero size (EC-09)", r.invested_capital == 0.0 and r.shares_acquired == 0.0
      and r.allocated_fraction == 0.0 and r.kelly_fraction_raw == 0.0, f"{r}")
check("wide cluster untouched", cl == {}, f"{cl}")

# 5. Gate disabled (None) -> same wide book trades (A/B control)
cfg = ak.FleetRiskConfig(max_round_trip_cost=None)
cl = {}
r = ak.evaluate_and_size_opportunity(wide, 0.80, 0.05, 10000.0, cl, config=cfg)
check("gate disabled trades", r.action == ak.TradeAction.BUY_YES, f"{r.action} {r.skip_reason}")

# 6. Threshold raised above rt -> trades
cfg = ak.FleetRiskConfig(max_round_trip_cost=0.99)
r = ak.evaluate_and_size_opportunity(wide, 0.80, 0.05, 10000.0, {}, config=cfg)
check("gate threshold 0.99 trades", r.action == ak.TradeAction.BUY_YES, f"{r.action}")

# 7. BUY_NO veto: p low, YES asks walk down steeply for NO exit.
# YES asks: 0.68 x100k (best), bids 0.65 x900 then 0.60 x100k -> entry NO
# q_exec ~0.385 (fleet fixture); NO exit walks asks: only derived bid
# 1-0.68=0.32 -> rt = 0.385 - 0.3168 = 0.068 > 0.05 -> veto.
no_book = ak.OrderBookSnapshot(
    market_id="T-7432-NO", title="fixture no", domain=ak.MarketDomain.PREDICTION_MARKET,
    factor_cluster="C7432NO", est_days_to_resolution=7.0,
    yes_asks=[ak.OrderBookLevel(price=0.68, size_shares=100000)],
    yes_bids=[ak.OrderBookLevel(price=0.65, size_shares=900),
              ak.OrderBookLevel(price=0.60, size_shares=100000)],
    fee_rate=0.01)
cl = {}
r = ak.evaluate_and_size_opportunity(no_book, 0.27, 0.05, 10000.0, cl)
check("BUY_NO wide SKIP Gate D", r.action == ak.TradeAction.SKIP
      and bool(r.skip_reason) and r.skip_reason.startswith("ROUND_TRIP_GATE_D"),
      f"{r.action} {r.skip_reason}")
check("BUY_NO cluster untouched", cl == {}, f"{cl}")
r2 = ak.evaluate_and_size_opportunity(no_book, 0.27, 0.05, 10000.0, {},
                                      config=ak.FleetRiskConfig(max_round_trip_cost=None))
check("BUY_NO gate-off trades", r2.action == ak.TradeAction.BUY_NO, f"{r2.action} {r2.skip_reason}")

# 8. Zero-size / invalid-price exit rungs are skipped (identical-to-clean)
dirty = book(0.50, [(0.50, 100000)], [(0.49, 100000), (0.01, 0), (float("inf"), 5000)])
v1, _ = ak.size_walked_exit_bid(dirty, ak.TradeAction.BUY_YES, 1000.0)
v2, _ = ak.size_walked_exit_bid(tight, ak.TradeAction.BUY_YES, 1000.0)
check("dirty exit ladder == clean", v1 == v2, f"{v1} vs {v2}")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
