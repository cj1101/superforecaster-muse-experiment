"""
test_exit_fill_fraction.py — SF-BL-9844 verification tests.

Pre-registered done criterion: "Gate D must value the unfilled
remainder at 0 (or veto below a pre-registered fill floor) so the
fixture SKIPs on round-trip cost, while every full-fill fixture's
decision stays bit-identical; kill if a census of forward books shows
fill<1 exits are never sized (then document the lower-bound convention
instead)."

Fix under test: round_trip_cost in superforecaster/alpha_kelly.py now
computes q_exit = bid_vwap * fill_fraction * (1 - fee_rate) — filled
proceeds net of the exit fee spread over ALL acquired shares, the
unfilled remainder valued at 0 (mirrors bid_ladder_liquidation_mark,
SF-BL-3660). Kill NOT met (see CENSUS.md): 372/2637 forward-census
candidates have exit fill < 1 and 323 of them were sized pre-fix.

Pre-fix reproduction (verified live 2026-10-23, before the patch),
task fixture asks 0.60x10000 / bids 0.58x100, fee 0.0, 1000 shares:
  size_walked_exit_bid -> (0.58, fill 0.10)
  round_trip_cost(q_exec=0.60) -> rt_cost 0.020 (PASSES the 0.05 cap)
  evaluate(p=0.90, equity 10000) -> BUY_YES 2500 shares, exit fill 0.04
Post-fix: rt_cost 0.542, q_exit 0.058; evaluate SKIPs ROUND_TRIP_GATE_D.

Run:  python3 test_exit_fill_fraction.py   (exit 0 iff all pass)
No pytest dependency: plain checks + __main__ runner, same convention
as tests/test_alpha_kelly_round_trip_gate.py.
"""
import math
import os
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster"))
from superforecaster import alpha_kelly as ak  # noqa: E402

PASS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(asks, bids, fee=0.0, cluster="C9844", mid="T-9844"):
    return ak.OrderBookSnapshot(
        market_id=mid, title="fixture", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster=cluster, est_days_to_resolution=7.0,
        yes_asks=[ak.OrderBookLevel(price=p, size_shares=s) for p, s in asks],
        yes_bids=[ak.OrderBookLevel(price=p, size_shares=s) for p, s in bids],
        fee_rate=fee)


TASK_BOOK = _book([(0.60, 10000)], [(0.58, 100)], fee=0.0)

# ---- 1. Task fixture: direct functions ------------------------------------
v, f = ak.size_walked_exit_bid(TASK_BOOK, ak.TradeAction.BUY_YES, 1000.0)
check("walk unchanged: filled VWAP 0.58, fill 0.10",
      abs(v - 0.58) < 1e-12 and abs(f - 0.10) < 1e-12, f"v={v} f={f}")
rt, qx, ff = ak.round_trip_cost(TASK_BOOK, ak.TradeAction.BUY_YES, 1000.0, 0.60)
check("task fixture rt_cost == 0.542", abs(rt - 0.542) < 1e-9, f"rt={rt}")
check("task fixture q_exit == 0.058 (unfilled at 0)", abs(qx - 0.058) < 1e-9, f"q_exit={qx}")
check("task fixture fill returned 0.10", abs(ff - 0.10) < 1e-12, f"fill={ff}")
check("task fixture rt exceeds Gate D cap 0.05", rt > 0.05, f"rt={rt}")

# ---- 2. Partial fill with fee: fee charged only on filled proceeds --------
b = _book([(0.60, 100000)], [(0.58, 100)], fee=0.01)
rt, qx, ff = ak.round_trip_cost(b, ak.TradeAction.BUY_YES, 1000.0, 0.606)
exp_qx = 0.58 * 0.10 * 0.99
check("fee on filled proceeds only: q_exit == 0.58*0.10*0.99",
      abs(qx - exp_qx) < 1e-9 and abs(rt - (0.606 - exp_qx)) < 1e-9,
      f"q_exit={qx} rt={rt}")

# ---- 3. Multi-rung partial fill -------------------------------------------
b = _book([(0.55, 100000)], [(0.50, 300), (0.40, 200)], fee=0.0)
v, f = ak.size_walked_exit_bid(b, ak.TradeAction.BUY_YES, 1000.0)
check("multi-rung filled VWAP 0.46, fill 0.5",
      abs(v - 0.46) < 1e-9 and abs(f - 0.5) < 1e-12, f"v={v} f={f}")
rt, qx, ff = ak.round_trip_cost(b, ak.TradeAction.BUY_YES, 1000.0, 0.55)
check("multi-rung q_exit == 0.46*0.5 == 0.23, rt == 0.32",
      abs(qx - 0.23) < 1e-9 and abs(rt - 0.32) < 1e-9, f"q_exit={qx} rt={rt}")

# ---- 4. BUY_NO partial fill (derived NO bid ladder from YES asks) ---------
b = _book([(0.40, 100)], [(0.35, 100000)], fee=0.0)
rt, qx, ff = ak.round_trip_cost(b, ak.TradeAction.BUY_NO, 1000.0, 0.65)
# Derived NO bids: 1 - 0.40 = 0.60 x100 only -> fill 0.10, q_exit 0.06.
check("BUY_NO partial: q_exit == 0.60*0.10 == 0.06, rt == 0.59",
      abs(qx - 0.06) < 1e-9 and abs(rt - 0.59) < 1e-9 and abs(ff - 0.10) < 1e-12,
      f"q_exit={qx} rt={rt} fill={ff}")

# ---- 5. Full-fill fixtures are bit-identical to the pre-fix formula -------
def prefix_rt(book, side, shares, q_exec):
    vwap, fill = ak.size_walked_exit_bid(book, side, shares)
    return q_exec - vwap * (1.0 - book.fee_rate), vwap * (1.0 - book.fee_rate), fill

full_cases = [
    (_book([(0.50, 100000)], [(0.49, 100000)], fee=0.01), ak.TradeAction.BUY_YES, 1000.0, 0.505),
    (_book([(0.60, 100000)], [(0.58, 100000)], fee=0.01), ak.TradeAction.BUY_YES, 1000.0, 0.606),
    (_book([(0.60, 100000)], [(0.58, 100000)], fee=0.0), ak.TradeAction.BUY_YES, 500.0, 0.60),
    (_book([(0.60, 100000)], [(0.58, 100000)], fee=0.10), ak.TradeAction.BUY_YES, 1000.0, 0.66),
    (_book([(0.68, 100000)], [(0.65, 100000)], fee=0.01), ak.TradeAction.BUY_NO, 1000.0, 0.35),
    (_book([(0.50, 100), (0.52, 100000)], [(0.49, 50), (0.48, 100000)], fee=0.005),
     ak.TradeAction.BUY_YES, 1100.0, 0.52),
]
for i, (bk, side, shares, qe) in enumerate(full_cases):
    got = ak.round_trip_cost(bk, side, shares, qe)
    exp = prefix_rt(bk, side, shares, qe)
    check(f"full-fill case {i} bit-identical (rt, q_exit, fill)",
          got == exp and got[2] == 1.0, f"got={got} exp={exp}")

# ---- 6. No exit depth: unchanged contract (q_exit 0, rt == q_exec) --------
b = _book([(0.60, 1000)], [], fee=0.01)
rt, qx, ff = ak.round_trip_cost(b, ak.TradeAction.BUY_YES, 1000.0, 0.606)
check("no exit depth -> (q_exec, 0, 0)", rt == 0.606 and qx == 0.0 and ff == 0.0,
      f"rt={rt} q_exit={qx} fill={ff}")

# ---- 7. Invariant sweep over partial fills --------------------------------
ok = True
for bid_depth in [0, 1, 10, 100, 333, 999, 1000, 5000]:
    bk = _book([(0.60, 100000)], [(0.58, bid_depth)], fee=0.01)
    rt, qx, ff = ak.round_trip_cost(bk, ak.TradeAction.BUY_YES, 1000.0, 0.606)
    vwap, fill = ak.size_walked_exit_bid(bk, ak.TradeAction.BUY_YES, 1000.0)
    ok = ok and abs(qx - vwap * fill * 0.99) < 1e-12 and math.isfinite(rt) \
        and 0.0 <= qx <= vwap and abs(ff - fill) < 1e-15
check("invariant: q_exit == vwap*fill*(1-fee), in [0, vwap], rt finite", ok)

# ---- 8. Evaluate: task fixture now SKIPs at Gate D -------------------------
cfg = ak.FleetRiskConfig(budget_sensitivity_veto_enabled=False)
cl = {}
r = ak.evaluate_and_size_opportunity(TASK_BOOK, 0.90, 0.05, 10000.0, cl, config=cfg)
check("evaluate task fixture SKIP ROUND_TRIP_GATE_D",
      r.action == ak.TradeAction.SKIP and bool(r.skip_reason)
      and r.skip_reason.startswith("ROUND_TRIP_GATE_D"),
      f"{r.action} {r.skip_reason}")
check("evaluate task fixture zero size (EC-09), cluster untouched",
      r.invested_capital == 0.0 and r.shares_acquired == 0.0
      and r.allocated_fraction == 0.0 and cl == {}, f"{r} cl={cl}")
check("evaluate skip reason reports honest rt 0.577",
      "rt_cost=0.577" in (r.skip_reason or ""), f"{r.skip_reason}")
# Controls: gate disabled trades; cap raised above honest rt trades.
cl = {}
r = ak.evaluate_and_size_opportunity(
    TASK_BOOK, 0.90, 0.05, 10000.0, cl,
    config=ak.FleetRiskConfig(max_round_trip_cost=None,
                              budget_sensitivity_veto_enabled=False))
check("control: Gate D disabled -> BUY_YES (pre-fix sized position)",
      r.action == ak.TradeAction.BUY_YES and r.shares_acquired == 2500.0,
      f"{r.action} {r.shares_acquired} {r.skip_reason}")
r = ak.evaluate_and_size_opportunity(
    TASK_BOOK, 0.90, 0.05, 10000.0, {},
    config=ak.FleetRiskConfig(max_round_trip_cost=0.99,
                              budget_sensitivity_veto_enabled=False))
check("control: cap 0.99 > honest rt 0.577 -> BUY_YES",
      r.action == ak.TradeAction.BUY_YES, f"{r.action} {r.skip_reason}")

# ---- 9. Evaluate: partial fill can still trade when honest rt <= cap ------
# Not a hard fill floor: ask 0.10 x100k, bid 0.099 x100, cap 0.15 ->
# sized shares >> 100 (fill << 1) but honest rt ~0.099 <= 0.15 trades.
b = _book([(0.10, 100000)], [(0.099, 100)], fee=0.0, cluster="C9844P", mid="T-9844-P")
cl = {}
r = ak.evaluate_and_size_opportunity(
    b, 0.50, 0.05, 10000.0, cl,
    config=ak.FleetRiskConfig(max_round_trip_cost=0.15,
                              budget_sensitivity_veto_enabled=False))
if r.action == ak.TradeAction.BUY_YES:
    _, fill_r = ak.size_walked_exit_bid(b, ak.TradeAction.BUY_YES, r.shares_acquired)
else:
    fill_r = 1.0
check("partial fill trades when honest rt <= cap (no fill floor)",
      r.action == ak.TradeAction.BUY_YES and fill_r < 0.5,
      f"{r.action} shares={r.shares_acquired} fill={fill_r} {r.skip_reason}")

# ---- 10. Evaluate: full-fill fixtures' decisions unchanged -----------------
tight = _book([(0.50, 100000)], [(0.49, 100000)], fee=0.01, cluster="C7432", mid="T-7432")
cl = {}
r = ak.evaluate_and_size_opportunity(tight, 0.80, 0.05, 10000.0, cl)
check("full-fill tight book still BUY_YES, cluster written",
      r.action == ak.TradeAction.BUY_YES and cl.get("C7432", 0) > 0,
      f"{r.action} {r.skip_reason} cl={cl}")
wide = _book([(0.50, 100000)], [(0.44, 100), (0.30, 100000)], fee=0.01,
             cluster="C7432", mid="T-7432")
cl = {}
r = ak.evaluate_and_size_opportunity(wide, 0.80, 0.05, 10000.0, cl)
check("full-fill wide book still SKIP Gate D, cluster untouched",
      r.action == ak.TradeAction.SKIP and bool(r.skip_reason)
      and r.skip_reason.startswith("ROUND_TRIP_GATE_D") and cl == {},
      f"{r.action} {r.skip_reason} cl={cl}")

print(f"\n{sum(PASS)}/{len(PASS)} passed")
sys.exit(0 if all(PASS) else 1)
