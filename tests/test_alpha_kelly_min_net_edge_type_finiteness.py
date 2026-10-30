"""
test_alpha_kelly_min_net_edge_type_finiteness.py — SF-BL-10876 verification.

Prereg: "evaluate must SKIP with a named invalid-min-net-edge reason
(zero size, cluster dict untouched) for min_net_edge of NaN, +inf, -inf,
'0.05', None and True before the edge screen, with finite thresholds in
[0, 1) bit-identical; kill if FleetRiskConfig gains field validation
covering this field's type and finiteness under SF-BL-9397's landed fix."

Kill condition MET (see KILL_AUDIT.md in the SF-BL-10876 work dir):
SF-BL-9397's landed fix validates min_net_edge's type and finiteness at
construction (_validate_min_net_edge in FleetRiskConfig.__post_init__)
and at evaluate Gate 2.5 (_fleet_risk_config_range_violation). This
suite pins the SF-BL-10876 fixture behaviour against the CURRENT build
so the coverage cannot silently regress: standard fixture p=0.80,
ask 0.50 / bid 0.48, fee 0.01, equity $100,000 (edge 0.295).

Run: python3 test_alpha_kelly_min_net_edge_type_finiteness.py
No pytest dependency: test_* functions with assert + __main__ runner.
"""
import os
import pickle
import sys

_CANDIDATES = [
    os.path.expanduser("~/workspace/superforecaster/superforecaster"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "superforecaster"),
]
for _c in _CANDIDATES:
    if os.path.isfile(os.path.join(_c, "alpha_kelly.py")):
        sys.path.insert(0, _c)
        break

import alpha_kelly as ak  # noqa: E402

PASS = []
INVALID = [float("nan"), float("inf"), float("-inf"), "0.05", None, True, False]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book():
    return ak.OrderBookSnapshot(
        market_id="FIXTURE-10876", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C10876", est_days_to_resolution=10,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=1000000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=1000000)],
        fee_rate=0.01, on_chain_verified=True)


def _mutated(v):
    cfg = ak.FleetRiskConfig()
    cfg.min_net_edge = v
    return cfg


def _eval(cfg, p=0.80, used=None):
    return ak.evaluate_and_size_opportunity(
        _book(), p, 0.0, 100000.0, used if used is not None else {}, config=cfg)


def _assert_zero_skip(r):
    assert r.action == ak.TradeAction.SKIP, r
    assert r.invested_capital == 0.0 and r.shares_acquired == 0.0, r
    assert r.exec_price == 0.0 and r.liquidation_bid_price == 0.0, r
    assert r.kelly_fraction_raw == 0.0 and r.allocated_fraction == 0.0, r
    assert r.kelly_fraction_sized == 0.0, r
    assert r.kl_divergence_nats == 0.0 and r.expected_log_growth == 0.0, r


def _assert_named_invalid_reason(r, v):
    assert r.skip_reason is not None, r
    assert r.skip_reason.startswith("INVALID_CONFIG"), r.skip_reason
    assert "min_net_edge" in r.skip_reason, r.skip_reason
    assert repr(v) in r.skip_reason, (v, r.skip_reason)
    assert "SF-BL-9397" in r.skip_reason, r.skip_reason
    # The pre-fix mislabels must never reappear for an invalid hurdle.
    assert "INSUFFICIENT_NET_EDGE" not in r.skip_reason, r.skip_reason
    assert "ORDER_BOOK_DEPTH_EXHAUSTED" not in r.skip_reason, r.skip_reason


# ---- 1. Kill evidence: helper + construction cover type & finiteness ------

def test_helper_covers_type_and_finiteness():
    for bad in INVALID + [[], {}, -0.10, -1e-9, 10**400]:
        assert ak._is_valid_min_net_edge(bad) is False, bad
    for good in [0, 0.0, 0.05, 0.5, 0.99]:
        assert ak._is_valid_min_net_edge(good) is True, good


def test_construction_rejects_all_prereg_values():
    for bad in INVALID:
        try:
            ak.FleetRiskConfig(min_net_edge=bad)
        except ValueError as e:
            assert "min_net_edge" in str(e), (bad, e)
            assert "SF-BL-9397" in str(e), (bad, e)
        else:
            raise AssertionError(f"no ValueError for min_net_edge {bad!r}")


# ---- 2. Evaluate: mutated configs SKIP before the edge screen -------------

def test_mutated_invalid_skips_named_zero_cluster_untouched():
    for bad in INVALID:
        used = {}
        r = _eval(_mutated(bad), used=used)  # must not raise TypeError
        _assert_zero_skip(r)
        _assert_named_invalid_reason(r, bad)
        assert used == {}, (bad, used)


def test_mutated_invalid_skips_on_buy_no_side():
    for bad in INVALID:
        used = {}
        r = _eval(_mutated(bad), p=0.20, used=used)
        _assert_zero_skip(r)
        _assert_named_invalid_reason(r, bad)
        assert used == {}, (bad, used)


def test_unpickled_invalid_skips():
    cfg = pickle.loads(pickle.dumps(ak.FleetRiskConfig()))
    cfg.min_net_edge = float("nan")
    used = {}
    r = _eval(cfg, used=used)
    _assert_zero_skip(r)
    _assert_named_invalid_reason(r, float("nan"))
    assert used == {}, used


def test_lock_scaled_arm_still_gated_before_hurdle_read():
    for bad in [float("nan"), "0.05"]:
        cfg = _mutated(bad)
        cfg.lock_scaled_hurdle_enabled = True
        used = {}
        r = _eval(cfg, used=used)
        _assert_zero_skip(r)
        _assert_named_invalid_reason(r, bad)
        assert used == {}, used


# ---- 3. Finite thresholds in [0, 1): bit-identical snapshots ---------------
# Recorded from the current (SF-BL-9397-landed) build AND matching the
# SF-BL-9397 pre-patch valid battery for the shared fixture numbers.

_TRADE = dict(action="BUY_YES", invested=15000.0, shares=29702.970297029704,
              exec_price=0.505, kelly_raw=0.595959595959596, alloc=0.15)


def test_finite_in_range_trade_cases_identical():
    for v in [0, 0.0, 0.05, 0.10, 0.29, 0.295]:
        used = {}
        r = _eval(ak.FleetRiskConfig(min_net_edge=v), used=used)
        assert r.action.value == _TRADE["action"], (v, r)
        assert r.invested_capital == _TRADE["invested"], (v, r)
        assert r.shares_acquired == _TRADE["shares"], (v, r)
        assert r.exec_price == _TRADE["exec_price"], (v, r)
        assert r.kelly_fraction_raw == _TRADE["kelly_raw"], (v, r)
        assert r.allocated_fraction == _TRADE["alloc"], (v, r)
        assert r.skip_reason is None, (v, r)
        assert used == {"C10876": 0.15}, (v, used)


def test_finite_in_range_skip_cases_identical():
    cases = [(0.30, "INSUFFICIENT_NET_EDGE (max_edge=0.295 < 0.300)"),
             (0.50, "INSUFFICIENT_NET_EDGE (max_edge=0.295 < 0.500)"),
             (0.99, "INSUFFICIENT_NET_EDGE (max_edge=0.295 < 0.990)")]
    for v, reason in cases:
        used = {}
        r = _eval(ak.FleetRiskConfig(min_net_edge=v), used=used)
        _assert_zero_skip(r)
        assert r.skip_reason == reason, (v, r.skip_reason)
        assert used == {}, (v, used)


def test_default_matches_explicit_005():
    a = _eval(ak.FleetRiskConfig())
    b = _eval(ak.FleetRiskConfig(min_net_edge=0.05))
    assert (a.action, a.invested_capital, a.shares_acquired, a.exec_price,
            a.allocated_fraction, a.skip_reason) == \
           (b.action, b.invested_capital, b.shares_acquired, b.exec_price,
            b.allocated_fraction, b.skip_reason), (a, b)


def test_sweep_argument_surface_untouched():
    # sweep_order_book's own -1.0 full-walk override (SF-BL-9471) is a
    # separate surface and must keep working.
    inv, shares, q = ak.sweep_order_book(
        _book(), ak.TradeAction.BUY_YES, 1000.0, 0.80, min_net_edge=-1.0)
    assert inv > 0 and shares > 0 and 0 < q < 1, (inv, shares, q)


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for fn in ALL_TESTS:
        try:
            fn()
            check(fn.__name__, True)
        except AssertionError as e:
            check(fn.__name__, False, str(e))
        except Exception as e:  # noqa: BLE001
            check(fn.__name__, False, f"{type(e).__name__}: {e}")
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
