"""
test_fleet_risk_config_validation.py — SF-BL-9397 verification tests.

Pre-registered done/kill criterion: "constructing or using a
FleetRiskConfig with kelly_multiplier outside [0.25, 0.50],
max_cluster_allocation outside (0, 0.15], min_net_edge < 0 or
max_inter_agent_dispersion outside [0, 1] must raise or clamp to the
mandated bands before any sizing; kill if a wild config changes any
allocation versus the default config on the fixture book."
Kill NOT met — see CALLER_CENSUS.md (no producer invariant bounds the
fields; the wild config DID change allocation pre-fix, repro.py).

Pre-fix reproduction (verified live, repro_prepatch.txt): title fixture
(p=0.60, ask 0.55 / bid 0.53, fee 0.01, equity $100,000) — default config
SKIPs INSUFFICIENT_NET_EDGE (0.044 < 0.050); wild config (2.0 / 0.90 /
-0.10) BUY_YES allocated 0.20022497187851465 / $20,022.497187851466;
construction accepted every wild value; min_net_edge='0.05' by mutation
crashed evaluate with a bare TypeError.

Fix (defence in depth, mirroring SF-BL-9363 / 10844 / 10845):
(a) FleetRiskConfig.__post_init__ raises ValueError naming the field,
    value, band and SF-BL-9397 for any out-of-band kelly_multiplier /
    max_cluster_allocation / min_net_edge (finite-real domains, bool
    excluded; min_net_edge has no preregistered upper bound, so none
    is imposed beyond >= 0 finite).
(b) New Gate 2.5 in evaluate_and_size_opportunity, after Gate 2 and
    before the first read of any of the three fields, SKIPs a
    mutated/unpickled config INVALID_CONFIG at zero size (EC-09),
    cluster dict untouched — never a TypeError, never a trade.
(c) max_inter_agent_dispersion tightened from SF-BL-10844's >= 0 to
    this task's earlier-registered [0, 1] (construction ValueError +
    Gate 1.9 SKIP INVALID_MAX_INTER_AGENT_DISPERSION).

Run:  python3 test_fleet_risk_config_validation.py  (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner.
"""
import contextlib
import io
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

INVALID_KELLY = [2.0, 0.10, 0.0, 0.24, 0.51, 1, -0.35, float("nan"),
                 float("inf"), float("-inf"), "0.35", None, True, False,
                 [], {}]
VALID_KELLY = [0.25, 0.50, 0.35, 0.30, 0.4999, 0.2501]
INVALID_CLUSTER = [0.90, 0.16, 0.150001, 1.0, 0.0, -0.15, float("nan"),
                   float("inf"), "0.15", None, True, False, []]
VALID_CLUSTER = [0.15, 0.10, 0.01, 0.001, 0.05]
INVALID_EDGE = [-0.10, -1.0, -1e-9, float("nan"), float("inf"),
                float("-inf"), "0.05", None, True, False, [], {}]
VALID_EDGE = [0.05, 0.0, 0, 0.03, 0.40, 1.0, 2.0]  # no upper bound preregistered
INVALID_DISP = [1.50, 2, 1e308, 1.000001, -0.10, float("nan"),
                float("inf"), float("-inf"), "0.22", None, True, False, []]
VALID_DISP = [0.22, 0, 0.0, 1, 1.0, 0.5, 0.10]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(ask=0.50, bid=0.48, cluster="C9397",
          domain=ak.MarketDomain.PREDICTION_MARKET):
    return ak.OrderBookSnapshot(
        market_id="FIXTURE-9397", title="prereg fixture", domain=domain,
        factor_cluster=cluster, est_days_to_resolution=10,
        yes_asks=[ak.OrderBookLevel(price=ask, size_shares=1000000)],
        yes_bids=[ak.OrderBookLevel(price=bid, size_shares=1000000)],
        fee_rate=0.01, on_chain_verified=True)


def _title_book():
    return _book(ask=0.55, bid=0.53)


def _mutated(**kw):
    cfg = ak.FleetRiskConfig()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _eval(cfg, p=0.80, disp=0.0, book=None, used=None):
    return ak.evaluate_and_size_opportunity(
        book if book is not None else _book(), p, disp, 100000.0,
        used if used is not None else {}, config=cfg)


def _assert_zero_skip(r):
    assert r.action == ak.TradeAction.SKIP, r
    assert r.invested_capital == 0.0 and r.shares_acquired == 0.0, r
    assert r.exec_price == 0.0 and r.liquidation_bid_price == 0.0, r
    assert r.kelly_fraction_raw == 0.0 and r.allocated_fraction == 0.0, r
    assert r.kelly_fraction_sized == 0.0, r
    assert r.kl_divergence_nats == 0.0 and r.expected_log_growth == 0.0, r


# ---- 1. Construction: named ValueError for invalid, valid accepted ---------

def test_construction_rejects_invalid_kelly():
    for bad in INVALID_KELLY:
        try:
            ak.FleetRiskConfig(kelly_multiplier=bad)
        except ValueError as e:
            assert "kelly_multiplier" in str(e), (bad, e)
            assert "SF-BL-9397" in str(e), (bad, e)
        else:
            raise AssertionError(f"no ValueError for kelly {bad!r}")


def test_construction_rejects_invalid_cluster():
    for bad in INVALID_CLUSTER:
        try:
            ak.FleetRiskConfig(max_cluster_allocation=bad)
        except ValueError as e:
            assert "max_cluster_allocation" in str(e), (bad, e)
            assert "SF-BL-9397" in str(e), (bad, e)
        else:
            raise AssertionError(f"no ValueError for cluster {bad!r}")


def test_construction_rejects_invalid_min_net_edge():
    for bad in INVALID_EDGE:
        try:
            ak.FleetRiskConfig(min_net_edge=bad)
        except ValueError as e:
            assert "min_net_edge" in str(e), (bad, e)
            assert "SF-BL-9397" in str(e), (bad, e)
        else:
            raise AssertionError(f"no ValueError for min_net_edge {bad!r}")


def test_construction_rejects_invalid_dispersion_tightened():
    for bad in INVALID_DISP:
        try:
            ak.FleetRiskConfig(max_inter_agent_dispersion=bad)
        except ValueError as e:
            assert "max_inter_agent_dispersion" in str(e), (bad, e)
            assert "SF-BL-9397" in str(e), (bad, e)
        else:
            raise AssertionError(f"no ValueError for dispersion {bad!r}")


def test_construction_rejects_wild_title_config():
    try:
        ak.FleetRiskConfig(kelly_multiplier=2.0, max_cluster_allocation=0.90,
                           min_net_edge=-0.10)
    except ValueError as e:
        assert "kelly_multiplier" in str(e), e  # first field checked
        return
    raise AssertionError("wild title config accepted at construction")


def test_construction_accepts_valid_and_defaults():
    for good in VALID_KELLY:
        assert ak.FleetRiskConfig(kelly_multiplier=good).kelly_multiplier == good
    for good in VALID_CLUSTER:
        assert ak.FleetRiskConfig(max_cluster_allocation=good).max_cluster_allocation == good
    for good in VALID_EDGE:
        assert ak.FleetRiskConfig(min_net_edge=good).min_net_edge == good
    for good in VALID_DISP:
        assert ak.FleetRiskConfig(max_inter_agent_dispersion=good).max_inter_agent_dispersion == good
    d = ak.FleetRiskConfig()
    assert (d.kelly_multiplier, d.max_cluster_allocation, d.min_net_edge,
            d.max_inter_agent_dispersion) == (0.35, 0.15, 0.05, 0.22)


def test_construction_other_fields_unaffected():
    # Out-of-scope fields keep their own (separate-prereg) behaviour.
    assert ak.FleetRiskConfig(max_round_trip_cost=None).max_round_trip_cost is None
    assert ak.FleetRiskConfig(require_on_chain_for_crypto=False) is not None
    assert ak.FleetRiskConfig(min_confidence=0.0).min_confidence == 0.0
    assert ak.FleetRiskConfig(crypto_kelly_dampener=0.5).crypto_kelly_dampener == 0.5


def test_helpers_directly():
    for good in VALID_KELLY:
        assert ak._is_valid_kelly_multiplier(good) is True, good
    for bad in INVALID_KELLY:
        assert ak._is_valid_kelly_multiplier(bad) is False, bad
    for good in VALID_CLUSTER:
        assert ak._is_valid_max_cluster_allocation(good) is True, good
    for bad in INVALID_CLUSTER:
        assert ak._is_valid_max_cluster_allocation(bad) is False, bad
    for good in VALID_EDGE:
        assert ak._is_valid_min_net_edge(good) is True, good
    for bad in INVALID_EDGE:
        assert ak._is_valid_min_net_edge(bad) is False, bad
    for good in VALID_DISP:
        assert ak._is_valid_max_inter_agent_dispersion(good) is True, good
    for bad in INVALID_DISP:
        assert ak._is_valid_max_inter_agent_dispersion(bad) is False, bad
    assert ak._fleet_risk_config_range_violation(ak.FleetRiskConfig()) is None
    v = ak._fleet_risk_config_range_violation(_mutated(max_cluster_allocation=0.9))
    assert v[0] == "max_cluster_allocation", v


# ---- 2. Gate 2.5 / Gate 1.9: mutated invalid configs SKIP, never trade -----

def test_mutated_invalid_fields_skip_invalid_config():
    cases = ([("kelly_multiplier", b) for b in INVALID_KELLY]
             + [("max_cluster_allocation", b) for b in INVALID_CLUSTER]
             + [("min_net_edge", b) for b in INVALID_EDGE])
    for field_name, bad in cases:
        used = {}
        r = _eval(_mutated(**{field_name: bad}), used=used)
        _assert_zero_skip(r)
        assert r.skip_reason is not None, (field_name, bad, r)
        assert r.skip_reason.startswith("INVALID_CONFIG"), (field_name, bad, r.skip_reason)
        assert field_name in r.skip_reason, (field_name, bad, r.skip_reason)
        assert "SF-BL-9397" in r.skip_reason, (field_name, bad, r.skip_reason)
        assert used == {}, (field_name, bad, used)  # cluster untouched


def test_wild_title_config_never_allocates():
    # The headline bug, both fixtures: zero allocation, not 20.02%.
    used = {}
    r = _eval(_mutated(kelly_multiplier=2.0, max_cluster_allocation=0.90,
                       min_net_edge=-0.10), p=0.60, book=_title_book(), used=used)
    _assert_zero_skip(r)
    assert r.skip_reason.startswith("INVALID_CONFIG"), r.skip_reason
    assert used == {}, used
    used2 = {}
    r2 = _eval(_mutated(kelly_multiplier=2.0, max_cluster_allocation=0.90,
                        min_net_edge=-0.10), used=used2)
    _assert_zero_skip(r2)
    assert used2 == {}, used2


def test_string_min_net_edge_no_longer_raises_typeerror():
    r = _eval(_mutated(min_net_edge="0.05"))
    _assert_zero_skip(r)
    assert "'0.05'" in r.skip_reason, r.skip_reason


def test_invalid_config_skips_on_buy_no_side_too():
    for field_name, bad in (("kelly_multiplier", 2.0),
                            ("max_cluster_allocation", 0.90),
                            ("min_net_edge", -0.10)):
        r = _eval(_mutated(**{field_name: bad}), p=0.20)
        _assert_zero_skip(r)
        assert r.skip_reason.startswith("INVALID_CONFIG"), r.skip_reason


def test_mutated_dispersion_above_one_skips_gate19():
    for bad in (1.50, 2, float("inf")):
        used = {}
        r = _eval(_mutated(max_inter_agent_dispersion=bad), disp=0.50, used=used)
        _assert_zero_skip(r)
        assert r.skip_reason.startswith("INVALID_MAX_INTER_AGENT_DISPERSION"), r.skip_reason
        assert "[0, 1]" in r.skip_reason, r.skip_reason
        assert used == {}, used


def test_unpickled_and_raw_configs_skip():
    cfg = pickle.loads(pickle.dumps(ak.FleetRiskConfig()))
    cfg.kelly_multiplier = 2.0  # post-load mutation
    r = _eval(cfg)
    _assert_zero_skip(r)
    assert r.skip_reason.startswith("INVALID_CONFIG"), r.skip_reason
    raw = object.__new__(ak.FleetRiskConfig)
    raw.min_net_edge, raw.min_confidence = 0.05, 0.60
    raw.kelly_multiplier, raw.crypto_kelly_dampener = 0.35, 0.50
    raw.max_cluster_allocation = 0.90
    raw.max_inter_agent_dispersion = 0.22
    raw.require_on_chain_for_crypto, raw.max_round_trip_cost = True, 0.05
    raw.lock_scaled_hurdle_enabled = False
    raw.budget_sensitivity_veto_enabled = True
    r2 = _eval(raw)
    _assert_zero_skip(r2)
    assert "max_cluster_allocation" in r2.skip_reason, r2.skip_reason


def test_earlier_gates_keep_precedence():
    # Invalid posterior (Gate 0.0) still precedes the config-range gate.
    r = ak.evaluate_and_size_opportunity(
        _book(), float("nan"), 0.0, 100000.0, {}, config=_mutated(kelly_multiplier=2.0))
    _assert_zero_skip(r)
    assert r.skip_reason.startswith("INVALID_POSTERIOR"), r.skip_reason
    # Gate 1 (crypto) still precedes Gate 2.5 on an unverified crypto book.
    b = _book(domain=ak.MarketDomain.CRYPTO)
    b.on_chain_verified = False
    r2 = ak.evaluate_and_size_opportunity(
        b, 0.80, 0.0, 100000.0, {}, config=_mutated(kelly_multiplier=2.0))
    _assert_zero_skip(r2)
    assert r2.skip_reason == "FAILED_CRYPTOBENCH_ONCHAIN_GATE", r2.skip_reason
    # Gate 2 still precedes Gate 2.5: valid-range config, high dispersion.
    r3 = _eval(ak.FleetRiskConfig(), disp=0.50)
    assert r3.skip_reason == "UNRESOLVED_AGENT_DISPERSION (0.50 > 0.22)", r3.skip_reason


def test_sweep_argument_surface_untouched():
    # sweep_order_book's own min_net_edge=-1.0 full-walk override is a
    # separate surface (SF-BL-9471) and must still work.
    inv, shares, q = ak.sweep_order_book(
        _book(), ak.TradeAction.BUY_YES, 1000.0, 0.80, min_net_edge=-1.0)
    assert inv > 0 and shares > 0 and 0 < q < 1, (inv, shares, q)


# ---- 3. Valid configs: byte-identical to pre-fix snapshots -----------------
# Snapshots recorded from the PRE-PATCH file (alpha_kelly_prepatch.py)
# on this suite's fixtures (ask 0.50 / bid 0.48, size 1e6, fee 0.01,
# equity $100,000) — see repro/snapshot notes in DONE-CLAIM.md.

_VALID_CASES = [
    # label, kwargs, p, disp, action, invested, shares, exec, kelly_raw, sized, alloc, skip
    ("default", {}, 0.80, 0.0, "BUY_YES", 15000.0, 29702.970297029704, 0.505,
     0.595959595959596, 0.15, 0.15, None),
    ("kelly0.25+cluster0.10", {"kelly_multiplier": 0.25, "max_cluster_allocation": 0.10},
     0.80, 0.0, "BUY_YES", 10000.0, 19801.980198019803, 0.505,
     0.595959595959596, 0.1, 0.1, None),
    ("kelly0.50", {"kelly_multiplier": 0.50}, 0.80, 0.0, "BUY_YES", 15000.0,
     29702.970297029704, 0.505, 0.595959595959596, 0.15, 0.15, None),
    ("kelly0.25", {"kelly_multiplier": 0.25}, 0.80, 0.0, "BUY_YES",
     14898.989898989901, 29502.950295029506, 0.505, 0.595959595959596,
     0.148989898989899, 0.148989898989899, None),
    ("minedge0.40", {"min_net_edge": 0.40}, 0.80, 0.0, "SKIP", 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, "INSUFFICIENT_NET_EDGE (max_edge=0.295 < 0.400)"),
    ("minedge0.0+kelly0.25", {"min_net_edge": 0.0, "kelly_multiplier": 0.25},
     0.80, 0.0, "BUY_YES", 14898.989898989901, 29502.950295029506, 0.505,
     0.595959595959596, 0.148989898989899, 0.148989898989899, None),
    ("minedge0.03", {"min_net_edge": 0.03}, 0.80, 0.0, "BUY_YES", 15000.0,
     29702.970297029704, 0.505, 0.595959595959596, 0.15, 0.15, None),
    ("disp-thr1.0", {"max_inter_agent_dispersion": 1.0}, 0.80, 0.50,
     "BUY_YES", 15000.0, 29702.970297029704, 0.505, 0.595959595959596,
     0.15, 0.15, None),
    ("disp-thr0.0", {"max_inter_agent_dispersion": 0.0}, 0.80, 0.0,
     "BUY_YES", 15000.0, 29702.970297029704, 0.505, 0.595959595959596,
     0.15, 0.15, None),
    ("buy-no", {}, 0.20, 0.0, "BUY_NO", 15000.0, 28560.54836252856, 0.5252,
     0.5791245791245792, 0.15, 0.15, None),
]


def test_valid_cases_byte_identical():
    for (label, kw, p, disp, action, inv, shares, q, kelly, sized,
         alloc, reason) in _VALID_CASES:
        r = _eval(ak.FleetRiskConfig(**kw), p=p, disp=disp)
        assert r.action.value == action, (label, r)
        assert r.invested_capital == inv, (label, r.invested_capital)
        assert r.shares_acquired == shares, (label, r.shares_acquired)
        assert r.exec_price == q, (label, r.exec_price)
        assert r.kelly_fraction_raw == kelly, (label, r.kelly_fraction_raw)
        assert r.kelly_fraction_sized == sized, (label, r.kelly_fraction_sized)
        assert r.allocated_fraction == alloc, (label, r.allocated_fraction)
        assert r.skip_reason == reason, (label, r.skip_reason)


def test_title_fixture_default_unchanged():
    r = _eval(ak.FleetRiskConfig(), p=0.60, book=_title_book())
    _assert_zero_skip(r)
    assert r.skip_reason == "INSUFFICIENT_NET_EDGE (max_edge=0.044 < 0.050)", r.skip_reason


def test_constructed_valid_matches_mutated():
    a = _eval(ak.FleetRiskConfig(kelly_multiplier=0.25, max_cluster_allocation=0.10))
    b = _eval(_mutated(kelly_multiplier=0.25, max_cluster_allocation=0.10))
    assert (a.action, a.invested_capital, a.allocated_fraction) == \
           (b.action, b.invested_capital, b.allocated_fraction), (a, b)


def test_default_config_path_unchanged():
    r = ak.evaluate_and_size_opportunity(_book(), 0.80, 0.0, 100000.0, {})
    assert r.action == ak.TradeAction.BUY_YES and r.invested_capital == 15000.0, r


def test_valid_trade_still_consumes_cluster_room():
    used = {}
    r = _eval(ak.FleetRiskConfig(), used=used)
    assert r.action == ak.TradeAction.BUY_YES, r
    assert used == {"C9397": 0.15}, used


# ---- 4. No regression on the built-in benchmark -----------------------------

def test_benchmark_nstar_unchanged():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    assert '"north_star_score": 0.0547' in out, out[:400]
    assert '"north_star_score": -0.1499' in out, out[:400]


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
