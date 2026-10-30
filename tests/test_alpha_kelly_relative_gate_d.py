#!/usr/bin/env python3
"""SF-BL-11194 tests: relative Gate D veto (rt_cost / q_exec > share).

Covers: helper arithmetic incl. the title fixture and the strict->
boundary; corrupt-input fail-to-flat; production evaluate contrasts
(flag off admits the 96.7%-loss tail fixture byte-identically, flag on
SKIPs it ROUND_TRIP_GATE_D_RELATIVE); non-tail behaviour unchanged
(relative prong inert above q_exec = cap/share); None cap disables the
relative prong too; composition with the flat veto.
"""
import math
import sys

sys.path.insert(0, "/home/hatch/workspace/superforecaster")
from superforecaster.alpha_kelly import (  # noqa: E402
    FleetRiskConfig, MarketDomain, OrderBookLevel, OrderBookSnapshot,
    TradeAction, evaluate_and_size_opportunity, relative_round_trip_veto,
)

PASS = 0


def check(name, cond):
    global PASS
    assert cond, f"FAIL: {name}"
    PASS += 1
    print(f"  ok {name}")


def book(ask, bid, fee=0.01, days=7.0, size=100000.0):
    return OrderBookSnapshot(
        market_id="T", title="t", domain=MarketDomain.PREDICTION_MARKET,
        factor_cluster="T", est_days_to_resolution=days,
        yes_asks=[OrderBookLevel(price=ask, size_shares=size)],
        yes_bids=[OrderBookLevel(price=bid, size_shares=size)],
        fee_rate=fee)


def run(b, p, cfg):
    return evaluate_and_size_opportunity(
        calibrated_prob_yes=p, book=b, portfolio_equity=10000.0,
        cluster_exposure_used={}, inter_agent_dispersion=0.05, config=cfg)


print("helper arithmetic")
check("title fixture share 0.967 vetoes", relative_round_trip_veto(0.0293, 0.0303) is True)
check("share 0.33 admits", relative_round_trip_veto(0.010, 0.0303) is False)
check("share exactly 0.50 admits (strict >)", relative_round_trip_veto(0.015, 0.030) is False)
check("share 0.5001 vetoes", relative_round_trip_veto(0.015001, 0.030) is True)
check("custom share 0.25: 0.33 vetoes", relative_round_trip_veto(0.010, 0.0303, 0.25) is True)
check("custom share 0.75: 0.60 admits", relative_round_trip_veto(0.018, 0.030, 0.75) is False)
check("zero rt admits", relative_round_trip_veto(0.0, 0.03) is False)
check("mid-price wide rt: 0.30/0.50=0.60 vetoes at helper level",
      relative_round_trip_veto(0.30, 0.50) is True)

print("corrupt input fails to flat (no veto)")
for bad_rt, bad_q, bad_s in [
    (float("nan"), 0.03, 0.5), (float("inf"), 0.03, 0.5), ("0.02", 0.03, 0.5),
    (None, 0.03, 0.5), (True, 0.03, 0.5),
    (0.02, float("nan"), 0.5), (0.02, 0.0, 0.5), (0.02, -0.03, 0.5),
    (0.02, "0.03", 0.5), (0.02, None, 0.5),
    (0.02, 0.03, float("nan")), (0.02, 0.03, 0.0), (0.02, 0.03, -0.5),
    (0.02, 0.03, 1.5), (0.02, 0.03, "0.5"), (0.02, 0.03, True),
]:
    check(f"no veto rt={bad_rt!r} q={bad_q!r} s={bad_s!r}",
          relative_round_trip_veto(bad_rt, bad_q, bad_s) is False)
check("share 1.0 in domain: share 0.967 admits",
      relative_round_trip_veto(0.0293, 0.0303, 1.0) is False)

print("production evaluate: title tail fixture")
tail = book(0.030, 0.001)
off = run(tail, 0.30, FleetRiskConfig())
check("flag OFF: tail fixture BUY_YES (vacuity)", off.action == TradeAction.BUY_YES)
check("flag OFF: exec 0.0303", abs(off.exec_price - 0.0303) < 1e-9)
on = run(tail, 0.30, FleetRiskConfig(round_trip_relative_enabled=True))
check("flag ON: tail fixture SKIP", on.action == TradeAction.SKIP)
check("flag ON: reason RELATIVE", (on.skip_reason or "").startswith("ROUND_TRIP_GATE_D_RELATIVE"))
check("flag ON: zero size (EC-09)", on.invested_capital == 0.0 and on.shares_acquired == 0.0)

print("production evaluate: non-tail unchanged")
tight_tail = book(0.030, 0.028)  # rt ~0.0017, share ~0.057
r_off = run(tight_tail, 0.30, FleetRiskConfig())
r_on = run(tight_tail, 0.30, FleetRiskConfig(round_trip_relative_enabled=True))
check("tight tail admitted both arms", r_off.action == r_on.action == TradeAction.BUY_YES)
check("tight tail identical invested", r_off.invested_capital == r_on.invested_capital)
mid = book(0.50, 0.48)  # q_exec 0.505, rt 0.0298 <= flat cap; share 0.059
m_off = run(mid, 0.80, FleetRiskConfig())
m_on = run(mid, 0.80, FleetRiskConfig(round_trip_relative_enabled=True))
check("mid book admitted both arms, identical",
      m_off.action == m_on.action == TradeAction.BUY_YES
      and m_off.invested_capital == m_on.invested_capital)
wide = book(0.50, 0.10)  # rt 0.406 > flat cap: flat vetoes in both arms
w_off = run(wide, 0.80, FleetRiskConfig())
w_on = run(wide, 0.80, FleetRiskConfig(round_trip_relative_enabled=True))
check("wide book flat-vetoed both arms",
      w_off.action == w_on.action == TradeAction.SKIP)
check("wide book keeps FLAT reason even flag-on (relative share 0.80 also fires? rt/q=0.803>0.5 -> relative reason takes precedence)",
      (w_on.skip_reason or "").startswith("ROUND_TRIP_GATE_D_RELATIVE")
      and (w_off.skip_reason or "").startswith("ROUND_TRIP_GATE_D ("))

print("None cap disables Gate D entirely, relative prong included")
n_on = run(tail, 0.30, FleetRiskConfig(max_round_trip_cost=None,
                                       round_trip_relative_enabled=True))
check("None cap + relative on: tail BUY_YES", n_on.action == TradeAction.BUY_YES)

print("q in (cap, cap/share): relative prong binds where flat admits")
shoulder = book(0.0792, 0.030)  # q_exec ~0.08, walked exit ~0.0297, rt ~0.0503>0.05? tune below
# direct construction: ask 0.07/bid 0.028 -> q_exec 0.0707, q_exit 0.0277, rt 0.0430, share 0.608
shoulder = book(0.07, 0.028)
s_off = run(shoulder, 0.30, FleetRiskConfig())
s_on = run(shoulder, 0.30, FleetRiskConfig(round_trip_relative_enabled=True))
check("shoulder (q 0.0707, rt 0.043, share 0.61): flat admits",
      s_off.action == TradeAction.BUY_YES)
check("shoulder: relative vetoes", s_on.action == TradeAction.SKIP
      and (s_on.skip_reason or "").startswith("ROUND_TRIP_GATE_D_RELATIVE"))

print(f"\nALL {PASS} CHECKS PASSED")
