#!/usr/bin/env python3
"""SF-BL-3839 — tests for the lock-scaled execution hurdle.

Covers: helper arithmetic (floor at/below D_REF, sqrt rise beyond it,
exact prereg values), corrupt-input fallback to flat, evaluate wiring
(screen + sweep use the effective hurdle when the flag is on), and
byte-identical behaviour with the flag off (the default).

Run: python3 test_lock_scaled_hurdle.py   (also pytest-compatible)
"""
import math
import os
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster"))
from superforecaster import alpha_kelly as ak

BASE = 0.05
CFG_OFF = ak.FleetRiskConfig(lock_scaled_hurdle_enabled=False)
CFG_ON = ak.FleetRiskConfig(lock_scaled_hurdle_enabled=True)
# Gate D / Gate BS off in the wiring fixtures so the hurdle is the only
# gate that can differ (their defaults are exercised by the replay).
CFG_ON_ND = ak.FleetRiskConfig(lock_scaled_hurdle_enabled=True,
                               max_round_trip_cost=None,
                               budget_sensitivity_veto_enabled=False)
CFG_OFF_ND = ak.FleetRiskConfig(lock_scaled_hurdle_enabled=False,
                                max_round_trip_cost=None,
                                budget_sensitivity_veto_enabled=False)
EQUITY = 100_000.0


def book(days, ask=0.50, bid=0.48, ask_size=100_000, bid_size=100_000):
    return ak.OrderBookSnapshot(
        market_id="T-3839", title="fixture", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="t3839", est_days_to_resolution=days,
        yes_asks=[ak.OrderBookLevel(price=ask, size_shares=ask_size)],
        yes_bids=[ak.OrderBookLevel(price=bid, size_shares=bid_size)],
        fee_rate=0.01)


def ev(b, p, cfg):
    return ak.evaluate_and_size_opportunity(b, p, 0.05, EQUITY, {}, cfg)


# ---- helper arithmetic -----------------------------------------------------
def test_helper_floor_at_and_below_ref():
    for d in (0.25, 1.0, 3.0, 6.999, 7.0):
        assert ak.lock_scaled_min_net_edge(BASE, d) == BASE, d


def test_helper_sqrt_rise_exact_prereg_values():
    cases = {30.0: 0.103510, 90.0: 0.179286, 365.0: 0.361050, 440.0: 0.396412}
    for d, want in cases.items():
        got = ak.lock_scaled_min_net_edge(BASE, d)
        assert abs(got - BASE * math.sqrt(d / 7.0)) < 1e-12, (d, got)
        assert abs(got - want) < 1e-5, (d, got, want)


def test_helper_monotone_and_custom_ref_and_base():
    hs = [ak.lock_scaled_min_net_edge(BASE, d) for d in (7, 14, 28, 56, 112)]
    assert hs == sorted(hs) and hs[0] == BASE and hs[-1] > hs[0]
    assert ak.lock_scaled_min_net_edge(BASE, 14.0, ref_days=14.0) == BASE
    assert abs(ak.lock_scaled_min_net_edge(0.03, 28.0) - 0.06) < 1e-12


def test_helper_corrupt_inputs_fail_to_flat():
    for bad in (float("nan"), float("inf"), float("-inf"), None, "300", True):
        assert ak.lock_scaled_min_net_edge(BASE, bad) == BASE, bad
    for bad_ref in (0.0, -7.0, float("nan"), float("inf"), None):
        assert ak.lock_scaled_min_net_edge(BASE, 300.0, ref_days=bad_ref) == BASE, bad_ref
    # Non-positive base is returned unchanged (scaling it is meaningless).
    assert ak.lock_scaled_min_net_edge(0.0, 300.0) == 0.0
    assert ak.lock_scaled_min_net_edge(-1.0, 300.0) == -1.0


# ---- evaluate wiring -------------------------------------------------------
def test_long_lock_candidate_passes_flat_fails_lock():
    # p_side - q_top = 0.80 - 0.505 = 0.295: clears flat 0.05 and the
    # 30d hurdle (0.1035) but not the 365d hurdle (0.3610).
    b365 = book(365.0)
    r_flat = ev(b365, 0.80, CFG_OFF_ND)
    r_lock = ev(b365, 0.80, CFG_ON_ND)
    assert r_flat.action == ak.TradeAction.BUY_YES, r_flat.skip_reason
    assert r_lock.action == ak.TradeAction.SKIP, r_lock
    assert r_lock.skip_reason.startswith("INSUFFICIENT_NET_EDGE"), r_lock.skip_reason
    assert "< 0.361" in r_lock.skip_reason, r_lock.skip_reason
    assert r_lock.invested_capital == 0.0 and r_lock.shares_acquired == 0.0


def test_lock_between_flat_and_scaled_still_trades():
    # Same candidate at 30d: hurdle 0.1035 < edge 0.295 -> trades under LOCK.
    r_lock = ev(book(30.0), 0.80, CFG_ON_ND)
    assert r_lock.action == ak.TradeAction.BUY_YES, r_lock.skip_reason


def test_short_lock_unchanged_by_flag():
    for d in (0.25, 3.0, 7.0):
        a = ev(book(d), 0.80, CFG_OFF_ND)
        b = ev(book(d), 0.80, CFG_ON_ND)
        assert (a.action, a.invested_capital, a.shares_acquired, a.exec_price) == \
               (b.action, b.invested_capital, b.shares_acquired, b.exec_price), d


def test_no_side_mirror():
    # BUY_NO candidate at 365d: p_yes=0.20 -> p_no=0.80, same geometry.
    r_flat = ev(book(365.0), 0.20, CFG_OFF_ND)
    r_lock = ev(book(365.0), 0.20, CFG_ON_ND)
    assert r_flat.action == ak.TradeAction.BUY_NO, r_flat.skip_reason
    assert r_lock.action == ak.TradeAction.SKIP, r_lock.skip_reason


def test_sweep_stop_uses_scaled_hurdle():
    # Two-rung ask ladder: cheap rung then a rung whose marginal price
    # leaves edge 0.15 — above the flat 0.05 stop, below the 365d scaled
    # stop (0.361) applied to the FIRST rung too... instead use D=28d
    # (hurdle 0.10): first rung edge 0.295 fills; second rung price 0.65
    # -> eff 0.6565, marginal edge 0.1435 >= 0.10 fills under both. For a
    # clean sweep contrast use D=112d (hurdle 0.20): the second rung's
    # marginal edge 0.1435 < 0.20 stops the LOCK sweep at rung 1, while
    # FLAT sweeps both rungs.
    b = ak.OrderBookSnapshot(
        market_id="T-3839-SWEEP", title="fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="t3839",
        est_days_to_resolution=112.0,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100),
                  ak.OrderBookLevel(price=0.65, size_shares=100_000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100_000)],
        fee_rate=0.01)
    r_flat = ev(b, 0.80, CFG_OFF_ND)
    r_lock = ev(b, 0.80, CFG_ON_ND)
    assert r_flat.action == ak.TradeAction.BUY_YES and r_lock.action == ak.TradeAction.BUY_YES
    assert r_lock.invested_capital < r_flat.invested_capital, (r_lock.invested_capital, r_flat.invested_capital)
    assert abs(r_lock.exec_price - 0.505) < 1e-9, r_lock.exec_price  # rung 1 only


def test_flag_off_config_default_and_benchmark_surface():
    cfg = ak.FleetRiskConfig()
    assert cfg.lock_scaled_hurdle_enabled is False and cfg.lock_hurdle_ref_days == 7.0
    assert CFG_ON.min_net_edge == CFG_OFF.min_net_edge == 0.05


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} PASS")
