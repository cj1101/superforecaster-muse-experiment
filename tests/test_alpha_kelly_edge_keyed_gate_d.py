#!/usr/bin/env python3
"""SF-BL-10726 — edge-keyed Gate D tests for alpha_kelly.py.

The edge-keyed veto (PREREG.md): ADMIT iff rt_cost < entry_edge
(= p_side - q_exec); VETO iff rt_cost >= entry_edge. Fixtures are
hand-computed. Run: python3 test_edge_keyed_gate_d.py
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


CFG_FLAT = ak.FleetRiskConfig(round_trip_edge_keyed_enabled=False)
CFG_EDGE = ak.FleetRiskConfig(round_trip_edge_keyed_enabled=True)


def deep_book(ask, bid, fee=0.01, size=100000.0):
    return ak.OrderBookSnapshot(
        market_id="T-10726", title="fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C10726", est_days_to_resolution=7.0,
        yes_asks=[ak.OrderBookLevel(price=ask, size_shares=size)],
        yes_bids=[ak.OrderBookLevel(price=bid, size_shares=size)],
        fee_rate=fee)


def run(book, p, cfg):
    cl = {}
    r = ak.evaluate_and_size_opportunity(book, p, 0.05, 10000.0, cl, config=cfg)
    return r, cl


# ---------------------------------------------------------------------------
# 1. Helper: exact title-fixture arithmetic (PREREG)
# ---------------------------------------------------------------------------
V = ak.edge_keyed_gate_d_veto
check("helper: edge .20 rt .06 admits (flat vetoes)", V(0.06, 0.20) is False)
check("helper: edge .051 rt .049 admits (net +.002)", V(0.049, 0.051) is False)
check("helper: edge .06 rt .055 admits (flat vetoes)", V(0.055, 0.06) is False)
check("helper: edge .06 rt .065 vetoes (both veto)", V(0.065, 0.06) is True)
check("helper: boundary rt == edge vetoes (net zero)", V(0.05, 0.05) is True)
check("helper: rt just below edge admits", V(0.049999, 0.05) is False)
check("helper: rt 0 admits any positive edge", V(0.0, 0.05) is False)
check("helper: negative edge vetoes any rt >= edge", V(0.0, -0.01) is True)

# 2. Helper: corrupt input never fires the predicate (fail to flat)
for bad in (float("nan"), float("inf"), float("-inf"), None, "0.03", True):
    check(f"helper: corrupt rt {bad!r} -> False", V(bad, 0.20) is False)
    check(f"helper: corrupt edge {bad!r} -> False", V(0.20, bad) is False)

# 3. Config: flag exists and defaults off (byte-identical production default)
check("config default flag off",
      ak.FleetRiskConfig().round_trip_edge_keyed_enabled is False)

# ---------------------------------------------------------------------------
# 4. Title fixture A via production evaluate: edge 0.20, rt in (0.05, edge)
#    ask 0.50 -> q_exec 0.505; p 0.705 -> edge 0.200; bid 0.43 ->
#    q_exit 0.43*0.99 = 0.4257, rt = 0.0793. Flat VETOES, edge ADMITS.
# ---------------------------------------------------------------------------
bookA = deep_book(0.50, 0.43)
rt, qx, ff = ak.round_trip_cost(bookA, ak.TradeAction.BUY_YES, 1000.0, 0.505)
check("fixture A rt hand-check", abs(rt - (0.505 - 0.43 * 0.99)) < 1e-9
      and 0.05 < rt < 0.20, f"rt={rt}")
r, cl = run(bookA, 0.705, CFG_FLAT)
check("fixture A flat SKIP Gate D", r.action == ak.TradeAction.SKIP
      and bool(r.skip_reason) and r.skip_reason.startswith("ROUND_TRIP_GATE_D"),
      f"{r.action} {r.skip_reason}")
check("fixture A flat zero size + cluster untouched",
      r.invested_capital == 0.0 and cl == {}, f"{r.invested_capital} {cl}")
r, cl = run(bookA, 0.705, CFG_EDGE)
check("fixture A edge-keyed BUY_YES", r.action == ak.TradeAction.BUY_YES,
      f"{r.action} {r.skip_reason}")
check("fixture A edge-keyed cluster written", cl.get("C10726", 0) > 0, f"{cl}")

# ---------------------------------------------------------------------------
# 5. Title fixture B via production evaluate: edge 0.051, rt ~0.049
#    ask 0.55 -> q_exec 0.5555; p 0.6065 -> edge 0.051; bid 0.512 ->
#    q_exit 0.50688, rt = 0.04862 (< 0.05 and < edge). BOTH admit.
# ---------------------------------------------------------------------------
bookB = deep_book(0.55, 0.512)
rt, qx, ff = ak.round_trip_cost(bookB, ak.TradeAction.BUY_YES, 1000.0, 0.5555)
check("fixture B rt hand-check", abs(rt - (0.5555 - 0.512 * 0.99)) < 1e-9
      and rt < 0.05, f"rt={rt}")
for cfg, nm in ((CFG_FLAT, "flat"), (CFG_EDGE, "edge")):
    r, _ = run(bookB, 0.6065, cfg)
    check(f"fixture B {nm} admits (net +0.0024)",
          r.action == ak.TradeAction.BUY_YES, f"{r.action} {r.skip_reason}")

# ---------------------------------------------------------------------------
# 6. Both veto: edge 0.12, rt above edge. ask 0.50 -> q_exec 0.505;
#    p 0.625 -> edge 0.120; bid 0.36 -> rt = 0.505 - 0.3564 = 0.1486.
# ---------------------------------------------------------------------------
bookC = deep_book(0.50, 0.36)
r, cl = run(bookC, 0.625, CFG_EDGE)
check("both-veto edge-keyed SKIP", r.action == ak.TradeAction.SKIP,
      f"{r.action} {r.skip_reason}")
check("both-veto reason is EDGE_KEYED variant",
      bool(r.skip_reason) and r.skip_reason.startswith(
          "ROUND_TRIP_GATE_D_EDGE_KEYED"), f"{r.skip_reason}")
check("both-veto reason still Gate-D prefixed (telemetry)",
      bool(r.skip_reason) and r.skip_reason.startswith("ROUND_TRIP_GATE_D"),
      f"{r.skip_reason}")
check("both-veto zero size + cluster untouched",
      r.invested_capital == 0.0 and r.shares_acquired == 0.0 and cl == {},
      f"{r} {cl}")
r, _ = run(bookC, 0.625, CFG_FLAT)
check("both-veto flat SKIP with flat reason byte-form",
      r.action == ak.TradeAction.SKIP and bool(r.skip_reason)
      and r.skip_reason.startswith("ROUND_TRIP_GATE_D (rt_cost="),
      f"{r.action} {r.skip_reason}")

# 7. Flag-off byte-identity on a tight trading book (no Gate D involvement)
tight = deep_book(0.50, 0.49)
r0, _ = run(tight, 0.80, ak.FleetRiskConfig())
r1, _ = run(tight, 0.80, CFG_FLAT)
check("flag-off identical to default config",
      (r0.action, r0.invested_capital, r0.shares_acquired, r0.exec_price,
       r0.skip_reason) == (r1.action, r1.invested_capital, r1.shares_acquired,
                           r1.exec_price, r1.skip_reason),
      f"{r0} vs {r1}")
r2, _ = run(tight, 0.80, CFG_EDGE)
check("tight book trades under edge-keyed too (rt 0.0199 < edge)",
      r2.action == ak.TradeAction.BUY_YES, f"{r2.action} {r2.skip_reason}")

# 8. None cap disables Gate D even with the edge-keyed flag on
cfg_none = ak.FleetRiskConfig(max_round_trip_cost=None,
                              round_trip_edge_keyed_enabled=True)
r, _ = run(bookC, 0.625, cfg_none)
check("None cap + flag on -> Gate D disabled, trades",
      r.action == ak.TradeAction.BUY_YES, f"{r.action} {r.skip_reason}")

# 9. Precedence: edge-keyed replaces the lock-scaled cap when both flags on.
#    Fixture A at D=7d: lock-scaled cap == flat 0.05 (would veto);
#    edge-keyed admits -> edge-keyed governs.
cfg_both = ak.FleetRiskConfig(round_trip_edge_keyed_enabled=True,
                              round_trip_lock_scaled_enabled=True)
r, _ = run(bookA, 0.705, cfg_both)
check("edge-keyed takes precedence over lock-scaled cap",
      r.action == ak.TradeAction.BUY_YES, f"{r.action} {r.skip_reason}")

# 10. BUY_NO side: NO edge keys the veto the same way.
#     YES ask 0.50 / bid 0.43 book, p_yes low: p_no = 0.705 -> p_yes 0.295.
#     NO entry q from best_no_ask = 1 - 0.43 = 0.57 -> q_exec 0.5757,
#     edge_no = 0.705 - 0.5757 = 0.1293; NO exit walks YES asks:
#     derived bid 1 - 0.50 = 0.50 -> q_exit 0.495, rt = 0.0807 in
#     (0.05, edge). Flat VETOES, edge ADMITS as BUY_NO.
r, _ = run(bookA, 0.295, CFG_FLAT)
check("BUY_NO fixture flat SKIP Gate D", r.action == ak.TradeAction.SKIP
      and bool(r.skip_reason) and r.skip_reason.startswith("ROUND_TRIP_GATE_D"),
      f"{r.action} {r.skip_reason}")
r, _ = run(bookA, 0.295, CFG_EDGE)
check("BUY_NO fixture edge-keyed BUY_NO", r.action == ak.TradeAction.BUY_NO,
      f"{r.action} {r.skip_reason}")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
