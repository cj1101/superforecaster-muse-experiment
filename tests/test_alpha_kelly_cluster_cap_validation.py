"""
test_cluster_cap_validation.py — SF-BL-10847 verification tests.

Prereg done/kill criteria (TASK.md): "Done when evaluate rejects any
max_cluster_allocation that is not a finite real in (0, 0.15] (bool
excluded) with a named invalid-config reason and the clean-cap
fixtures are byte-identical; kill if a caller census shows every cap
is a validated in-band literal."
Kill NOT met — see CALLER_CENSUS.md (no schema/producer invariant
bounds the field; the dataclass is mutable, unpickling bypasses
construction, evaluate's default config is one shared mutable
instance, and one in-repo test literal is the out-of-band 1.0).

Pre-fix reproduction (repro.py / repro_prefix.txt, run against the
pristine pre-SF-BL-9397 module, which predates the fix for this field):
prereg fixture p=0.80, ask 0.50 / bid 0.48, fee 0.01, equity $100,000,
empty cluster dict — clean 0.15 invested exactly $15,000;
max_cluster_allocation=True invested $20,858.58585858586 (20.86% of
equity in one cluster; BUY_NO side $20,269.36); '0.15' raised a bare
TypeError in the cap subtraction; NaN SKIPped CLUSTER_CAP_REACHED on
a fresh cluster. Construction accepted every value verbatim.

Fix (already landed by sibling SF-BL-9397, verified here for this
task's field and fixture — no new production change in SF-BL-10847):
(a) FleetRiskConfig.__post_init__ raises ValueError naming
    max_cluster_allocation, the value and the (0, 0.15] band via
    _validate_max_cluster_allocation / _is_valid_max_cluster_allocation
    (finite int/float only, bool excluded — the module's
    _is_finite_number convention, so Decimal/Fraction are rejected
    exactly as everywhere else in this file).
(b) Gate 2.5 in evaluate_and_size_opportunity (after Gate 2, before
    the first read of the field at Step 4) SKIPs a mutated/unpickled
    config INVALID_CONFIG naming max_cluster_allocation at zero size
    (EC-09), cluster dict untouched — never a TypeError, never a
    CLUSTER_CAP_REACHED mislabel, never a trade.

Run:  python3 test_cluster_cap_validation.py  (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner.
"""
import contextlib
import io
import math
import os
import pickle
import sys
from decimal import Decimal
from fractions import Fraction

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

INVALID_CAPS = [True, False, "0.15", "0.10", "", float("nan"),
                float("inf"), float("-inf"), 0, 0.0, -0.15, -1.0,
                0.150001, 0.16, 0.50, 0.90, 1.0, 1, 2, None, [], {},
                [0.15], Decimal("0.15"), Fraction(3, 20), 10 ** 400]
VALID_CAPS = [0.15, 0.10, 0.05, 0.01, 0.001, 0.149999, 0.000001]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(cluster="C10847", domain=ak.MarketDomain.PREDICTION_MARKET):
    return ak.OrderBookSnapshot(
        market_id="FIXTURE-10847", title="prereg fixture", domain=domain,
        factor_cluster=cluster, est_days_to_resolution=10,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=1000000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=1000000)],
        fee_rate=0.01, on_chain_verified=True)


def _mutated(cap):
    cfg = ak.FleetRiskConfig()
    cfg.max_cluster_allocation = cap
    return cfg


def _eval(cfg, p=0.80, disp=0.0, equity=100000.0, book=None, used=None):
    return ak.evaluate_and_size_opportunity(
        book if book is not None else _book(), p, disp, equity,
        used if used is not None else {}, config=cfg)


def _assert_zero_skip(r):
    assert r.action == ak.TradeAction.SKIP, r
    assert r.invested_capital == 0.0 and r.shares_acquired == 0.0, r
    assert r.exec_price == 0.0 and r.liquidation_bid_price == 0.0, r
    assert r.kelly_fraction_raw == 0.0 and r.allocated_fraction == 0.0, r
    assert r.kelly_fraction_sized == 0.0, r
    assert r.kl_divergence_nats == 0.0 and r.expected_log_growth == 0.0, r


def _assert_invalid_config(r, used_before=None, used_after=None):
    _assert_zero_skip(r)
    assert r.skip_reason is not None, r
    assert r.skip_reason.startswith("INVALID_CONFIG"), r.skip_reason
    assert "max_cluster_allocation" in r.skip_reason, r.skip_reason
    assert "(0, 0.15]" in r.skip_reason, r.skip_reason
    assert "CLUSTER_CAP_REACHED" not in r.skip_reason, r.skip_reason
    if used_before is not None:
        assert used_after == used_before, (used_before, used_after)


# ---- 1. Construction: named ValueError for invalid, valid accepted ---------

def test_construction_rejects_invalid_caps():
    for bad in INVALID_CAPS:
        try:
            ak.FleetRiskConfig(max_cluster_allocation=bad)
        except ValueError as e:
            assert "max_cluster_allocation" in str(e), (bad, e)
            assert "(0, 0.15]" in str(e), (bad, e)
        else:
            raise AssertionError(f"no ValueError for cap {bad!r}")


def test_construction_accepts_valid_caps_and_default():
    for good in VALID_CAPS:
        assert ak.FleetRiskConfig(
            max_cluster_allocation=good).max_cluster_allocation == good
    assert ak.FleetRiskConfig().max_cluster_allocation == 0.15


def test_construction_other_fields_unaffected():
    assert ak.FleetRiskConfig(max_round_trip_cost=None).max_round_trip_cost is None
    assert ak.FleetRiskConfig(require_on_chain_for_crypto=False) is not None
    assert ak.FleetRiskConfig(kelly_multiplier=0.25).kelly_multiplier == 0.25


def test_helpers_directly():
    for good in VALID_CAPS:
        assert ak._is_valid_max_cluster_allocation(good) is True, good
    for bad in INVALID_CAPS:
        assert ak._is_valid_max_cluster_allocation(bad) is False, bad
    try:
        ak._validate_max_cluster_allocation(True)
    except ValueError as e:
        assert "max_cluster_allocation" in str(e), e
    else:
        raise AssertionError("_validate accepted True")
    assert ak._fleet_risk_config_range_violation(
        ak.FleetRiskConfig()) is None
    v = ak._fleet_risk_config_range_violation(_mutated(0.90))
    assert v[0] == "max_cluster_allocation" and v[1] == 0.90, v
    assert v[2] == "(0, 0.15]", v


# ---- 2. Gate 2.5: mutated invalid caps SKIP, never trade / crash -----------

def test_mutated_invalid_caps_skip_invalid_config_buyyes():
    for bad in INVALID_CAPS:
        used = {}
        r = _eval(_mutated(bad), used=used)  # must not raise (no TypeError)
        _assert_invalid_config(r, used_before={}, used_after=used)


def test_mutated_invalid_caps_skip_on_buyno_side():
    for bad in (True, "0.15", float("nan"), float("inf"), 0.90, None):
        used = {}
        r = _eval(_mutated(bad), p=0.20, used=used)
        _assert_invalid_config(r, used_before={}, used_after=used)


def test_headline_three_cases_closed():
    # True: was BUY_YES $20,858.58585858586 (20.86% in one cluster).
    r = _eval(_mutated(True))
    _assert_invalid_config(r)
    assert r.invested_capital == 0.0, r
    # '0.15': was a bare TypeError in the cap subtraction.
    r = _eval(_mutated("0.15"))
    _assert_invalid_config(r)
    assert "'0.15'" in r.skip_reason, r.skip_reason
    # NaN: was SKIP CLUSTER_CAP_REACHED on a fresh (empty) cluster.
    used = {}
    r = _eval(_mutated(float("nan")), used=used)
    _assert_invalid_config(r, used_before={}, used_after=used)


def test_invalid_cap_with_existing_exposure_still_invalid_config():
    # Even where a cap-reached outcome would be plausible, an invalid
    # cap is reported as a config error and the ledger is untouched.
    for bad in (float("nan"), True, "0.15"):
        used = {"C10847": 0.05}
        r = _eval(_mutated(bad), used=used)
        _assert_invalid_config(r, used_before={"C10847": 0.05},
                               used_after=used)


def test_unpickled_and_raw_configs_skip():
    cfg = pickle.loads(pickle.dumps(ak.FleetRiskConfig()))
    cfg.max_cluster_allocation = float("nan")  # post-load mutation
    r = _eval(cfg)
    _assert_invalid_config(r)
    raw = object.__new__(ak.FleetRiskConfig)
    raw.min_net_edge, raw.min_confidence = 0.05, 0.60
    raw.kelly_multiplier, raw.crypto_kelly_dampener = 0.35, 0.50
    raw.max_cluster_allocation = "0.15"
    raw.max_inter_agent_dispersion = 0.22
    raw.require_on_chain_for_crypto, raw.max_round_trip_cost = True, 0.05
    raw.lock_scaled_hurdle_enabled = False
    raw.budget_sensitivity_veto_enabled = True
    r2 = _eval(raw)
    _assert_invalid_config(r2)


def test_missing_attribute_ducktyped_config_skips_for_cluster():
    # A duck-typed config object with no max_cluster_allocation
    # attribute at all: Gate 2.5's helper reads it via getattr as
    # None and rejects it by name. (A raw FleetRiskConfig instance
    # cannot exercise this path — the dataclass field default lives
    # on the class, so an unset instance attribute reads the valid
    # 0.15; that case is pinned separately below.)
    from types import SimpleNamespace
    cfg = SimpleNamespace(
        require_on_chain_for_crypto=True,
        max_inter_agent_dispersion=0.22,
        kelly_multiplier=0.35, min_net_edge=0.05)
    used = {}
    r = _eval(cfg, used=used)
    _assert_invalid_config(r, used_before={}, used_after=used)
    assert "max_cluster_allocation=None" in r.skip_reason, r.skip_reason


def test_raw_instance_without_cluster_reads_class_default():
    # object.__new__ bypasses __init__/__post_init__, but the field
    # default is a class attribute: an unset instance reads the valid
    # 0.15 and trades exactly as the default config. Not a bypass —
    # the value read is in-band by construction.
    raw = object.__new__(ak.FleetRiskConfig)
    raw.min_net_edge, raw.min_confidence = 0.05, 0.60
    raw.kelly_multiplier, raw.crypto_kelly_dampener = 0.35, 0.50
    raw.max_inter_agent_dispersion = 0.22
    raw.require_on_chain_for_crypto, raw.max_round_trip_cost = True, 0.05
    raw.lock_scaled_hurdle_enabled = False
    raw.budget_sensitivity_veto_enabled = True
    assert getattr(raw, "max_cluster_allocation") == 0.15
    r = _eval(raw)
    assert r.action == ak.TradeAction.BUY_YES, r
    assert r.invested_capital == 15000.0, r


def test_earlier_gates_keep_precedence():
    # Gate 0.0 (invalid posterior) still precedes Gate 2.5.
    r = ak.evaluate_and_size_opportunity(
        _book(), float("nan"), 0.0, 100000.0, {}, config=_mutated(True))
    _assert_zero_skip(r)
    assert r.skip_reason.startswith("INVALID_POSTERIOR"), r.skip_reason
    # Gate 1 (crypto on-chain) still precedes Gate 2.5.
    b = _book(domain=ak.MarketDomain.CRYPTO)
    b.on_chain_verified = False
    r2 = ak.evaluate_and_size_opportunity(
        b, 0.80, 0.0, 100000.0, {}, config=_mutated(True))
    _assert_zero_skip(r2)
    assert r2.skip_reason == "FAILED_CRYPTOBENCH_ONCHAIN_GATE", r2.skip_reason
    # Gate 2 (dispersion) still precedes Gate 2.5 — documented order;
    # on the prereg fixture (dispersion 0.0) Gate 2.5 is the first
    # gate that can fire for an invalid cap.
    r3 = _eval(_mutated(True), disp=0.50)
    assert r3.skip_reason == "UNRESOLVED_AGENT_DISPERSION (0.50 > 0.22)", r3.skip_reason


# ---- 3. Valid caps: byte-identical to pre-fix snapshots ---------------------
# Snapshots recorded from the PRE-FIX module (SF-BL-9397's pristine
# prepatch copy) on this task's fixture — see repro/gen notes in
# DONE-CLAIM.md. Every TradeExecutionResult field is asserted exactly.
# The cluster dict is asserted exactly too, except the seeded-used
# case's stored fraction: pre-fix wrote 0.15000000000000002 where the
# current module writes 0.15 — the SF-BL-10762 dollar-ledger re-base
# (an intervening, unrelated task), documented in DONE-CLAIM.md; the
# current value is pinned here.

_VALID_CASES = [
    # label, cap, p, equity, used_seed, action, invested, shares, exec,
    # liq, kelly_raw, sized, alloc, kl, elg, skip, used_after
    ("cap0.15-default", 0.15, 0.80, 100000.0, None, "BUY_YES", 15000.0,
     29702.970297029704, 0.505, 0.48, 0.595959595959596, 0.15, 0.15,
     0.18679455950992335, 0.07723680133014557, None, {"C10847": 0.15}),
    ("cap0.10", 0.10, 0.80, 100000.0, None, "BUY_YES", 10000.0,
     19801.980198019803, 0.505, 0.48, 0.595959595959596, 0.1, 0.1,
     0.18679455950992335, 0.05373459888048433, None, {"C10847": 0.1}),
    ("cap0.05", 0.05, 0.80, 100000.0, None, "BUY_YES", 5000.0,
     9900.990099009901, 0.505, 0.48, 0.595959595959596, 0.05, 0.05,
     0.18679455950992335, 0.028018755420928822, None, {"C10847": 0.05}),
    ("cap0.01", 0.01, 0.80, 100000.0, None, "BUY_YES", 1000.0,
     1980.19801980198, 0.505, 0.48, 0.595959595959596, 0.01, 0.01,
     0.18679455950992335, 0.005793334766233573, None, {"C10847": 0.01}),
    ("cap0.001-capreached", 0.001, 0.80, 100000.0, None, "SKIP", 0.0,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     "CLUSTER_CAP_REACHED (C10847)", {}),
    ("cap0.15-used0.05", 0.15, 0.80, 100000.0, 0.05, "BUY_YES", 10000.0,
     19801.980198019803, 0.505, 0.48, 0.595959595959596,
     0.09999999999999999, 0.1, 0.18679455950992335, 0.05373459888048433,
     None, {"C10847": 0.15}),
    ("cap0.15-used0.15-capreached", 0.15, 0.80, 100000.0, 0.15, "SKIP",
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     "CLUSTER_CAP_REACHED (C10847)", {"C10847": 0.15}),
    ("cap0.15-buyno", 0.15, 0.20, 100000.0, None, "BUY_NO", 15000.0,
     28560.54836252856, 0.5252, 0.5, 0.5791245791245792, 0.15, 0.15,
     0.16375080895393376, 0.06922899365264411, None, {"C10847": 0.15}),
    ("cap0.10-buyno", 0.10, 0.20, 100000.0, None, "BUY_NO", 10000.0,
     19040.36557501904, 0.5252, 0.5, 0.5791245791245792, 0.1, 0.1,
     0.16375080895393376, 0.04816626011922326, None, {"C10847": 0.1}),
    ("cap0.15-equity10k", 0.15, 0.80, 10000.0, None, "BUY_YES", 1500.0,
     2970.29702970297, 0.505, 0.48, 0.595959595959596, 0.15, 0.15,
     0.18679455950992335, 0.07723680133014557, None, {"C10847": 0.15}),
    ("cap0.149999", 0.149999, 0.80, 100000.0, None, "BUY_YES", 14999.9,
     29702.772277227723, 0.505, 0.48, 0.595959595959596, 0.149999,
     0.149999, 0.18679455950992335, 0.07723635298119148, None,
     {"C10847": 0.149999}),
]


def test_valid_cases_byte_identical():
    for (label, cap, p, eq, seed, action, inv, shares, q, liq, kelly,
         sized, alloc, kl, elg, reason, used_after) in _VALID_CASES:
        used = {} if seed is None else {"C10847": seed}
        r = _eval(ak.FleetRiskConfig(max_cluster_allocation=cap),
                  p=p, equity=eq, used=used)
        assert r.action.value == action, (label, r)
        assert r.invested_capital == inv, (label, r.invested_capital)
        assert r.shares_acquired == shares, (label, r.shares_acquired)
        assert r.exec_price == q, (label, r.exec_price)
        assert r.liquidation_bid_price == liq, (label, r)
        assert r.kelly_fraction_raw == kelly, (label, r.kelly_fraction_raw)
        assert r.kelly_fraction_sized == sized, (label, r)
        assert r.allocated_fraction == alloc, (label, r.allocated_fraction)
        assert r.kl_divergence_nats == kl, (label, r.kl_divergence_nats)
        assert r.expected_log_growth == elg, (label, r.expected_log_growth)
        assert r.market_mid_prob == 0.49, (label, r.market_mid_prob)
        assert r.skip_reason == reason, (label, r.skip_reason)
        assert used == used_after, (label, used)


def test_valid_cluster_cap_reached_is_legitimate():
    # With a VALID cap, CLUSTER_CAP_REACHED remains the correct reason
    # (tiny valid cap; fully-used cluster) — the NaN fix removes only
    # the mislabel for invalid caps, not the genuine signal.
    r = _eval(ak.FleetRiskConfig(max_cluster_allocation=0.001))
    assert r.skip_reason == "CLUSTER_CAP_REACHED (C10847)", r.skip_reason
    used = {"C10847": 0.15}
    r2 = _eval(ak.FleetRiskConfig(), used=used)
    assert r2.skip_reason == "CLUSTER_CAP_REACHED (C10847)", r2.skip_reason
    assert used == {"C10847": 0.15}, used


def test_constructed_valid_matches_mutated_valid():
    a = _eval(ak.FleetRiskConfig(max_cluster_allocation=0.10))
    b = _eval(_mutated(0.10))
    assert (a.action, a.invested_capital, a.shares_acquired,
            a.allocated_fraction, a.skip_reason) == \
           (b.action, b.invested_capital, b.shares_acquired,
            b.allocated_fraction, b.skip_reason), (a, b)


def test_default_config_path_unchanged():
    r = ak.evaluate_and_size_opportunity(_book(), 0.80, 0.0, 100000.0, {})
    assert r.action == ak.TradeAction.BUY_YES, r
    assert r.invested_capital == 15000.0, r


# ---- 4. Benchmark still runs (scores not pinned: intervening fleet --------
# landings after SF-BL-9397 moved the disciplined score; SF-BL-10847
# makes no production change, so it cannot move it — see DONE-CLAIM).

def test_benchmark_runs_with_finite_scores():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    scores = [float(line.split(":")[1].strip().rstrip(","))
              for line in out.splitlines() if "north_star_score" in line]
    assert len(scores) == 2 and all(math.isfinite(s) for s in scores), out[:400]


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
