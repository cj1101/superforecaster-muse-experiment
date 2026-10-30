#!/usr/bin/env python3
"""SF-BL-9396 tests: cluster_exposure_used values must be rejected/clamped
so allocated_fraction never exceeds the 15% cluster cap.

Prereg fixture: deep book ask 0.50 / bid 0.48, fee 0.01, equity $10,000,
p=0.80, cluster C1 -> uncapped target 0.2086, clean (used=0) allocation
is exactly the 0.15 cap.
"""
import math
import sys

sys.path.insert(0, "/home/hatch/workspace/superforecaster")
from superforecaster import alpha_kelly as ak  # noqa: E402

PASS = []

def check(name, cond, detail=""):
    PASS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f" | {detail}" if detail else ""))
    assert cond, f"{name} {detail}"

def book(cluster="C1", domain=ak.MarketDomain.PREDICTION_MARKET):
    return ak.OrderBookSnapshot(
        market_id="T", title="t", domain=domain,
        factor_cluster=cluster, est_days_to_resolution=5.0,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=1_000_000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=1_000_000)],
        fee_rate=0.01)

def run(used, p=0.80, cluster="C1", config=None, domain=ak.MarketDomain.PREDICTION_MARKET):
    d = {cluster: used}
    r = ak.evaluate_and_size_opportunity(
        book=book(cluster, domain), calibrated_prob_yes=p,
        inter_agent_dispersion=0.05, portfolio_equity=10_000.0,
        cluster_exposure_used=d, **({"config": config} if config else {}))
    return r, d

def fields(r):
    return (r.action, r.allocated_fraction, r.invested_capital,
            r.shares_acquired, r.kelly_fraction_sized, r.skip_reason)

# ---- 1. Prereg fixture -------------------------------------------------
clean, clean_d = run(0.0)
check("clean_used0_allocates_exactly_cap", clean.allocated_fraction == 0.15
      and clean.invested_capital == 1500.0, f"{clean.allocated_fraction}")
r, d = run(-0.20)
check("prereg_neg020_never_exceeds_cap", r.allocated_fraction <= 0.15, f"{r.allocated_fraction}")
check("prereg_neg020_identical_to_clean", fields(r) == fields(clean) and d == clean_d,
      f"{fields(r)} vs {fields(clean)}; dict {d}")

# ---- 2. Negative finite values clamp to 0 ------------------------------
for bad in (-0.01, -0.05, -0.20, -1.0, -100.0):
    r, d = run(bad)
    check(f"negative_{bad}_identical_to_clean", fields(r) == fields(clean) and d == {"C1": 0.15},
          f"{fields(r)} dict={d}")

# ---- 3. >cap finite values clamp to cap: SKIP, dict untouched ----------
for bad in (0.150001, 0.20, 1.0, 100.0):
    r, d = run(bad)
    check(f"overcap_{bad}_skip_cap_reached",
          r.action == ak.TradeAction.SKIP and r.allocated_fraction == 0.0
          and r.skip_reason.startswith("CLUSTER_CAP_REACHED") and d == {"C1": bad},
          f"{fields(r)} dict={d}")

# ---- 4. Non-finite / non-numeric fail closed at cap --------------------
for name, bad in [("nan", float("nan")), ("pos_inf", float("inf")),
                  ("neg_inf", float("-inf")), ("str", "0.05"),
                  ("none", None), ("bool_true", True), ("bool_false", False)]:
    r, d = run(bad)
    ok_dict = (math.isnan(d["C1"]) if name == "nan" else d["C1"] == bad)
    check(f"failclosed_{name}_skip_zero_dict_untouched",
          r.action == ak.TradeAction.SKIP and r.allocated_fraction == 0.0
          and r.invested_capital == 0.0 and r.skip_reason.startswith("CLUSTER_CAP_REACHED")
          and ok_dict, f"{fields(r)}")

# ---- 5. In-range values unchanged --------------------------------------
for used, exp_alloc in [(0.0, 0.15), (0.05, 0.10), (0.10, 0.05), (0.149, 0.001)]:
    r, d = run(used)
    check(f"inrange_{used}_alloc_{exp_alloc}",
          abs(r.allocated_fraction - exp_alloc) < 1e-9 and abs(d["C1"] - 0.15) < 1e-9,
          f"alloc={r.allocated_fraction} dict={d}")
r, d = run(0.15)
check("inrange_cap_skip", r.action == ak.TradeAction.SKIP and d == {"C1": 0.15})

# ---- 6. Sub-cap Kelly: negative base does not corrupt the ledger ------
# p=0.65 -> target ~0.1025 (< cap): allocation identical, but post-fix the
# stored ledger is sanitised_used + actual, in [0, cap] (pre-fix: -0.0975).
rc, dc = run(0.0, p=0.65)
r, d = run(-0.20, p=0.65)
check("subcap_negative_alloc_identical", r.allocated_fraction == rc.allocated_fraction)
check("subcap_negative_ledger_sanitised", d == dc and 0.0 <= d["C1"] <= 0.15, f"{d} vs {dc}")

# ---- 7. BUY_NO and CRYPTO sides share the same Step-4 cap --------------
r, d = run(-0.20, p=0.20)
check("buy_no_negative_clamped", r.action == ak.TradeAction.BUY_NO
      and r.allocated_fraction <= 0.15, f"{fields(r)}")
r, d = run(-0.20, domain=ak.MarketDomain.CRYPTO)
check("crypto_negative_clamped", r.allocated_fraction <= 0.15, f"{fields(r)}")

# ---- 8. Custom cap honoured --------------------------------------------
cfg = ak.FleetRiskConfig(max_cluster_allocation=0.10)
r, d = run(-0.20, config=cfg)
check("custom_cap_010_negative", r.allocated_fraction == 0.10, f"{r.allocated_fraction}")
r, d = run(0.20, config=cfg)
check("custom_cap_010_overcap_skip", r.action == ak.TradeAction.SKIP)

# ---- 9. Helper unit checks ---------------------------------------------
h = ak._sanitized_cluster_exposure_used
check("helper_units", h(-0.2, 0.15) == 0.0 and h(0.05, 0.15) == 0.05
      and h(0.2, 0.15) == 0.15 and h(float("nan"), 0.15) == 0.15
      and h(float("-inf"), 0.15) == 0.15 and h("x", 0.15) == 0.15
      and h(True, 0.15) == 0.15 and h(0, 0.15) == 0.0)

# ---- 10. Settlement release stays exact after a clamped trade ----------
r, d = run(-0.20)
ak.settle_or_mark_trade(r, is_settled=True, outcome_yes=1)
check("release_after_clamp_returns_to_zero", d == {"C1": 0.0}, f"{d}")

# ---- 11. Kill criterion sweep: no fixture allocates > cap --------------
import itertools
breach = []
for used in [x / 100 for x in range(-200, 201, 7)] + [float("nan"), float("inf"), float("-inf")]:
    rr, _ = run(used)
    if rr.allocated_fraction > 0.15 + 1e-12:
        breach.append((used, rr.allocated_fraction))
check("kill_sweep_no_alloc_over_cap", not breach, f"{breach}")

print(f"\n{sum(1 for _, ok in PASS if ok)}/{len(PASS)} passed")
