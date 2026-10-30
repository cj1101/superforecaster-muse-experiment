"""SF-BL-10619 — lock-scaled Gate D cap tests for alpha_kelly.py.

Fixtures are hand-computed. Run: python3 test_lock_scaled_gate_d.py
Formula (PREREG.md): cap(D) = base * sqrt(D / 7.0); corrupt input -> base.
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


H = ak.lock_scaled_max_round_trip_cost

# 1. Exact prereg arithmetic at base 0.05
check("cap D=7 == flat", abs(H(0.05, 7.0) - 0.05) < 1e-12, f"{H(0.05, 7.0)}")
check("cap D=0.25", abs(H(0.05, 0.25) - 0.05 * math.sqrt(0.25 / 7.0)) < 1e-12,
      f"{H(0.05, 0.25)}")
check("cap D=2", abs(H(0.05, 2.0) - 0.026726124191242442) < 1e-12, f"{H(0.05, 2.0)}")
check("cap D=30", abs(H(0.05, 30.0) - 0.10350983390135314) < 1e-12, f"{H(0.05, 30.0)}")
check("cap D=440", abs(H(0.05, 440.0) - 0.39641248358604597) < 1e-12, f"{H(0.05, 440.0)}")
check("cap scales with base", abs(H(0.10, 28.0) - 0.20) < 1e-12, f"{H(0.10, 28.0)}")
check("cap custom ref", abs(H(0.05, 14.0, ref_days=14.0) - 0.05) < 1e-12,
      f"{H(0.05, 14.0, ref_days=14.0)}")

# 2. Monotonicity in D
caps = [H(0.05, d) for d in (0.25, 1.0, 7.0, 30.0, 90.0, 365.0)]
check("cap monotonic in D", all(a < b for a, b in zip(caps, caps[1:])), f"{caps}")

# 3. Corrupt-input fallback: fail to flat base, never loosen/tighten
check("None base -> None (gate disabled)", H(None, 30.0) is None)
check("NaN lock -> base", H(0.05, float("nan")) == 0.05)
check("inf lock -> base", H(0.05, float("inf")) == 0.05)
check("string lock -> base", H(0.05, "30") == 0.05)
check("bool lock -> base", H(0.05, True) == 0.05)
check("zero lock -> base", H(0.05, 0.0) == 0.05)
check("negative lock -> base", H(0.05, -5.0) == 0.05)
check("zero ref -> base", H(0.05, 30.0, ref_days=0.0) == 0.05)
check("NaN ref -> base", H(0.05, 30.0, ref_days=float("nan")) == 0.05)
check("negative base -> base unchanged", H(-0.05, 30.0) == -0.05)
check("NaN base -> NaN unchanged", math.isnan(H(float("nan"), 30.0)))

# 4. Evaluate-level contrast. Single-rung deep books (Gate BS full walks are
# price-invariant on one rung, so Gate D is the only arm difference).
# ask 0.50, fee 0.01 -> q_exec 0.505. rt = 0.505 - 0.99 * bid.
def make_book(bid, days, cluster):
    return ak.OrderBookSnapshot(
        market_id="T-10619", title="fixture", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster=cluster, est_days_to_resolution=days,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000)],
        yes_bids=[ak.OrderBookLevel(price=bid, size_shares=100000)],
        fee_rate=0.01)

FLAT = ak.FleetRiskConfig(round_trip_lock_scaled_enabled=False)
SCALED = ak.FleetRiskConfig(round_trip_lock_scaled_enabled=True)

# bid 0.47 -> rt = 0.0397: flat admits at every lock; scaled vetoes at 0.25d
b = make_book(0.47, 0.25, "C1")
r_flat = ak.evaluate_and_size_opportunity(b, 0.80, 0.05, 10000.0, {}, config=FLAT)
check("rt=0.0397 D=0.25 FLAT trades", r_flat.action == ak.TradeAction.BUY_YES,
      f"{r_flat.action} {r_flat.skip_reason}")
cl = {}
r_sc = ak.evaluate_and_size_opportunity(b, 0.80, 0.05, 10000.0, cl, config=SCALED)
check("rt=0.0397 D=0.25 SCALED vetoes Gate D",
      r_sc.action == ak.TradeAction.SKIP and bool(r_sc.skip_reason)
      and r_sc.skip_reason.startswith("ROUND_TRIP_GATE_D")
      and "> 0.009;" in r_sc.skip_reason, f"{r_sc.action} {r_sc.skip_reason}")
check("scaled veto zero size + cluster untouched",
      r_sc.invested_capital == 0.0 and cl == {}, f"{r_sc.invested_capital} {cl}")

# Same book at D=7: cap == flat, both trade identically
b7 = make_book(0.47, 7.0, "C1")
rf = ak.evaluate_and_size_opportunity(b7, 0.80, 0.05, 10000.0, {}, config=FLAT)
rs = ak.evaluate_and_size_opportunity(b7, 0.80, 0.05, 10000.0, {}, config=SCALED)
check("D=7 arms identical (trade)",
      rf.action == rs.action == ak.TradeAction.BUY_YES
      and rf.invested_capital == rs.invested_capital
      and rf.shares_acquired == rs.shares_acquired,
      f"{rf.action}/{rs.action} {rf.invested_capital}/{rs.invested_capital}")

# bid 0.40 -> rt = 0.109: flat vetoes at every lock.
b40_7 = make_book(0.40, 7.0, "C2")
rf = ak.evaluate_and_size_opportunity(b40_7, 0.80, 0.05, 10000.0, {}, config=FLAT)
rs = ak.evaluate_and_size_opportunity(b40_7, 0.80, 0.05, 10000.0, {}, config=SCALED)
check("rt=0.109 D=7 both veto, reason byte-identical",
      rf.action == rs.action == ak.TradeAction.SKIP
      and rf.skip_reason == rs.skip_reason,
      f"{rf.skip_reason} vs {rs.skip_reason}")

# At D=30 cap 0.1035 < 0.109: scaled still vetoes, reason shows scaled cap
b40_30 = make_book(0.40, 30.0, "C2")
rs = ak.evaluate_and_size_opportunity(b40_30, 0.80, 0.05, 10000.0, {}, config=SCALED)
check("rt=0.109 D=30 SCALED vetoes at 0.104",
      rs.action == ak.TradeAction.SKIP and bool(rs.skip_reason)
      and "> 0.104;" in rs.skip_reason, f"{rs.skip_reason}")

# At D=90 cap 0.1793 > 0.109: scaled admits what flat vetoes
b40_90 = make_book(0.40, 90.0, "C2")
rf = ak.evaluate_and_size_opportunity(b40_90, 0.80, 0.05, 10000.0, {}, config=FLAT)
rs = ak.evaluate_and_size_opportunity(b40_90, 0.80, 0.05, 10000.0, {}, config=SCALED)
check("rt=0.109 D=90 FLAT vetoes / SCALED trades",
      rf.action == ak.TradeAction.SKIP and rs.action == ak.TradeAction.BUY_YES,
      f"{rf.action} {rf.skip_reason} / {rs.action} {rs.skip_reason}")

# 5. Flag off is the default and matches the pre-task default config exactly
b = make_book(0.47, 0.25, "C3")
r_def = ak.evaluate_and_size_opportunity(b, 0.80, 0.05, 10000.0, {})
r_off = ak.evaluate_and_size_opportunity(b, 0.80, 0.05, 10000.0, {}, config=FLAT)
check("default config == flag-off config",
      r_def.action == r_off.action and r_def.skip_reason == r_off.skip_reason
      and r_def.invested_capital == r_off.invested_capital,
      f"{r_def.action} vs {r_off.action}")

# 6. Gate disabled (None) honoured with the scaled flag on
cfg_none = ak.FleetRiskConfig(max_round_trip_cost=None,
                              round_trip_lock_scaled_enabled=True)
r = ak.evaluate_and_size_opportunity(make_book(0.40, 0.25, "C4"), 0.80, 0.05,
                                     10000.0, {}, config=cfg_none)
check("None cap + scaled flag trades (gate disabled)",
      r.action == ak.TradeAction.BUY_YES, f"{r.action} {r.skip_reason}")

# 7. BUY_NO contrast (7432 fixture: rt = 0.068): flat vetoes at D=90 too,
# scaled cap 0.179 admits.
no_book = ak.OrderBookSnapshot(
    market_id="T-10619-NO", title="fixture no", domain=ak.MarketDomain.PREDICTION_MARKET,
    factor_cluster="C5", est_days_to_resolution=90.0,
    yes_asks=[ak.OrderBookLevel(price=0.68, size_shares=100000)],
    yes_bids=[ak.OrderBookLevel(price=0.65, size_shares=900),
              ak.OrderBookLevel(price=0.60, size_shares=100000)],
    fee_rate=0.01)
rf = ak.evaluate_and_size_opportunity(no_book, 0.27, 0.05, 10000.0, {}, config=FLAT)
rs = ak.evaluate_and_size_opportunity(no_book, 0.27, 0.05, 10000.0, {}, config=SCALED)
check("BUY_NO rt=0.068 D=90 FLAT vetoes Gate D",
      rf.action == ak.TradeAction.SKIP and bool(rf.skip_reason)
      and rf.skip_reason.startswith("ROUND_TRIP_GATE_D"), f"{rf.skip_reason}")
check("BUY_NO rt=0.068 D=90 SCALED trades (or later gate)",
      rs.action == ak.TradeAction.BUY_NO
      or (rs.skip_reason is not None and "ROUND_TRIP_GATE_D" not in rs.skip_reason),
      f"{rs.action} {rs.skip_reason}")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
