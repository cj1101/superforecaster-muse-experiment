"""
test_alpha_kelly_max_round_trip_cost_validation.py — SF-BL-10798.

Pre-registered done/kill criterion (TASK.md): "a non-finite or negative
max_round_trip_cost must be rejected (named ValueError at config use, or
Gate D treated as veto-all with a logged flag) so the repro fixture
SKIPs, fixtured at NaN, +inf and -1.0, with finite thresholds (including
None = disabled) bit-identical; kill if FleetRiskConfig gains field
validation covering this field under SF-BL-9397's landed fix."

Kill NOT met — see CALLER_CENSUS.md in the task work dir (SF-BL-9397's
landed block lists max_round_trip_cost as an out-of-scope separate
prereg surface; its helper and __post_init__ never check the field).

Pre-fix reproduction (repro.py / repro_prefix.txt in the task work
dir): prereg fixture (ask 0.50 / bid 0.10, p=0.80, fee 0.01, equity
$10,000, rt_cost 0.406/share) at the clean 0.05 threshold SKIPs
ROUND_TRIP_GATE_D, but at NaN / +inf BUY_YES $1,500 (full cluster cap)
because `rt_cost > NaN` is always False; -1.0 / -inf SKIP only
incidentally as an ultra-tight cap (vetoing a tight rt=0.020 book
too); '0.05' raises a bare TypeError; True masquerades as 1.0.

Fix (defence in depth, mirroring SF-BL-9397):
(a) FleetRiskConfig.__post_init__ raises ValueError naming the field,
    value and SF-BL-10798 for any max_round_trip_cost that is not None
    or a finite real >= 0 (bool excluded).
(b) New Gate 2.6 in evaluate_and_size_opportunity, immediately after
    Gate 2.5 (SF-BL-9397) and before the field's first read at Gate D,
    SKIPs a mutated/unpickled config INVALID_CONFIG /
    INVALID_MAX_ROUND_TRIP_COST at zero size (EC-09), cluster dict
    untouched — never a TypeError, never a trade.
(c) Finite thresholds (including None, 0.0, and the strict-> Gate D
    boundary) are bit-identical: pinned below against pre-fix
    snapshots (234-case snapshot diff in the task work dir was empty
    apart from a concurrent task's new None provenance fields).

Run:  python3 test_alpha_kelly_max_round_trip_cost_validation.py
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

INVALID = [float("nan"), float("inf"), float("-inf"), -1.0, -1e-9,
           -0.05, "0.05", "nan", True, False, [], {}, 10 ** 400]
VALID = [None, 0, 0.0, 0.05, 0.10, 0.405, 0.406, 0.99, 1, 2.0, 1e-9, 1e6]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(ask=0.50, bid=0.10, cluster="C10798"):
    return ak.OrderBookSnapshot(
        market_id="FIX-10798", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster=cluster, est_days_to_resolution=7.0,
        yes_asks=[ak.OrderBookLevel(price=ask, size_shares=100000)],
        yes_bids=[ak.OrderBookLevel(price=bid, size_shares=100000)],
        fee_rate=0.01, on_chain_verified=True)


def _wide():
    return _book(bid=0.10)   # rt_cost 0.406 at q_exec 0.505


def _tight():
    return _book(bid=0.49)   # rt_cost 0.020 at q_exec 0.505


def _mutated(value):
    cfg = ak.FleetRiskConfig()
    cfg.max_round_trip_cost = value
    return cfg


def _eval(cfg, p=0.80, book=None, used=None):
    return ak.evaluate_and_size_opportunity(
        book if book is not None else _wide(), p, 0.0, 10000.0,
        used if used is not None else {}, config=cfg)


def _assert_zero_skip(r):
    assert r.action == ak.TradeAction.SKIP, r
    assert r.invested_capital == 0.0 and r.shares_acquired == 0.0, r
    assert r.exec_price == 0.0 and r.liquidation_bid_price == 0.0, r
    assert r.kelly_fraction_raw == 0.0 and r.allocated_fraction == 0.0, r
    assert r.kelly_fraction_sized == 0.0, r
    assert r.kl_divergence_nats == 0.0 and r.expected_log_growth == 0.0, r


def _assert_invalid_config(r, used, value):
    _assert_zero_skip(r)
    assert r.skip_reason is not None, r
    assert r.skip_reason.startswith("INVALID_CONFIG"), r.skip_reason
    assert "INVALID_MAX_ROUND_TRIP_COST" in r.skip_reason, r.skip_reason
    assert "max_round_trip_cost" in r.skip_reason, r.skip_reason
    assert repr(value) in r.skip_reason, (value, r.skip_reason)
    assert "SF-BL-10798" in r.skip_reason, r.skip_reason
    assert used == {}, used  # cluster untouched


# ---- 1. Construction: named ValueError for invalid, valid accepted ---------

def test_construction_rejects_prereg_fixtures():
    for bad in (float("nan"), float("inf"), -1.0):
        try:
            ak.FleetRiskConfig(max_round_trip_cost=bad)
        except ValueError as e:
            assert "max_round_trip_cost" in str(e), (bad, e)
            assert "SF-BL-10798" in str(e), (bad, e)
        else:
            raise AssertionError(f"no ValueError for {bad!r}")


def test_construction_rejects_all_invalid():
    for bad in INVALID:
        try:
            ak.FleetRiskConfig(max_round_trip_cost=bad)
        except ValueError as e:
            assert "max_round_trip_cost" in str(e), (bad, e)
            assert "SF-BL-10798" in str(e), (bad, e)
        else:
            raise AssertionError(f"no ValueError for {bad!r}")


def test_construction_accepts_valid_and_default():
    for good in VALID:
        cfg = ak.FleetRiskConfig(max_round_trip_cost=good)
        assert cfg.max_round_trip_cost == good or (
            good is None and cfg.max_round_trip_cost is None), good
    assert ak.FleetRiskConfig().max_round_trip_cost == 0.05


def test_construction_other_config_fields_unaffected():
    # SF-BL-9397 first-violation order at construction is preserved:
    # the wild 9397 config still names kelly_multiplier first.
    try:
        ak.FleetRiskConfig(kelly_multiplier=2.0, max_round_trip_cost=float("nan"))
    except ValueError as e:
        assert "kelly_multiplier" in str(e), e
    else:
        raise AssertionError("wild config accepted")
    assert ak.FleetRiskConfig(min_confidence=0.0).min_confidence == 0.0


def test_helpers_directly():
    for good in VALID:
        assert ak._is_valid_max_round_trip_cost(good) is True, good
    for bad in INVALID:
        assert ak._is_valid_max_round_trip_cost(bad) is False, bad
    try:
        ak._validate_max_round_trip_cost(float("nan"))
    except ValueError as e:
        assert "max_round_trip_cost" in str(e) and "SF-BL-10798" in str(e), e
    else:
        raise AssertionError("helper accepted NaN")


# ---- 2. Gate 2.6: mutated / unpickled invalid configs SKIP, never trade ----

def test_mutated_prereg_fixtures_skip_on_wide_book():
    # The headline bug: NaN / +inf traded $1,500 pre-fix; -1.0 must now
    # be rejected as invalid config, not as an incidental Gate D veto.
    for bad in (float("nan"), float("inf"), -1.0):
        used = {}
        r = _eval(_mutated(bad), used=used)
        _assert_invalid_config(r, used, bad)


def test_mutated_all_invalid_skip_on_wide_book():
    for bad in INVALID:
        used = {}
        r = _eval(_mutated(bad), used=used)
        _assert_invalid_config(r, used, bad)


def test_mutated_invalid_skips_on_tight_book_too():
    # Pre-fix -1.0 vetoed the tight book as ROUND_TRIP_GATE_D and NaN
    # traded it; both must now SKIP as invalid config.
    for bad in (float("nan"), float("inf"), -1.0, -1e-9):
        used = {}
        r = _eval(_mutated(bad), book=_tight(), used=used)
        _assert_invalid_config(r, used, bad)


def test_mutated_invalid_skips_on_buy_no_side():
    for bad in (float("nan"), float("inf"), -1.0):
        used = {}
        r = _eval(_mutated(bad), p=0.20, book=_tight(), used=used)
        _assert_invalid_config(r, used, bad)


def test_string_threshold_no_longer_raises_typeerror():
    r = _eval(_mutated("0.05"))
    _assert_zero_skip(r)
    assert "'0.05'" in r.skip_reason, r.skip_reason


def test_unpickled_config_skips():
    cfg = pickle.loads(pickle.dumps(ak.FleetRiskConfig()))
    cfg.max_round_trip_cost = float("nan")
    used = {}
    r = _eval(cfg, used=used)
    _assert_invalid_config(r, used, float("nan"))


def test_raw_config_skips():
    raw = object.__new__(ak.FleetRiskConfig)
    raw.min_net_edge, raw.min_confidence = 0.05, 0.60
    raw.kelly_multiplier, raw.crypto_kelly_dampener = 0.35, 0.50
    raw.max_cluster_allocation = 0.15
    raw.max_inter_agent_dispersion = 0.22
    raw.require_on_chain_for_crypto = True
    raw.max_round_trip_cost = float("inf")
    raw.lock_scaled_hurdle_enabled = False
    raw.dispersion_taper_enabled = False
    raw.budget_sensitivity_veto_enabled = True
    used = {}
    r = _eval(raw, used=used)
    _assert_invalid_config(r, used, float("inf"))


def test_earlier_gates_keep_precedence():
    # Invalid posterior (Gate 0.0) still precedes Gate 2.6.
    r = ak.evaluate_and_size_opportunity(
        _wide(), float("nan"), 0.0, 10000.0, {},
        config=_mutated(float("nan")))
    _assert_zero_skip(r)
    assert r.skip_reason.startswith("INVALID_POSTERIOR"), r.skip_reason
    # A valid config with high dispersion still hits Gate 2 exactly.
    r2 = ak.evaluate_and_size_opportunity(
        _tight(), 0.80, 0.50, 10000.0, {}, config=ak.FleetRiskConfig())
    assert r2.skip_reason == "UNRESOLVED_AGENT_DISPERSION (0.50 > 0.22)", r2.skip_reason


# ---- 3. Valid thresholds: bit-identical to pre-fix snapshots ---------------
# Snapshots recorded from the PRE-PATCH file on this suite's fixtures
# (ask 0.50, fee 0.01, equity $10,000, p=0.80 unless noted) — full
# 234-case pre/post snapshot diff in the task work dir.

_VALID_CASES = [
    # book, p, threshold, action, invested, shares, exec, skip_reason
    ("wide", 0.80, 0.05, "SKIP", 0.0, 0.0, 0.0,
     "ROUND_TRIP_GATE_D (rt_cost=0.406 > 0.050; q_exec=0.505, q_exit_walked=0.099)"),
    ("wide", 0.80, 0.0, "SKIP", 0.0, 0.0, 0.0,
     "ROUND_TRIP_GATE_D (rt_cost=0.406 > 0.000; q_exec=0.505, q_exit_walked=0.099)"),
    ("wide", 0.80, 0.405, "SKIP", 0.0, 0.0, 0.0,
     "ROUND_TRIP_GATE_D (rt_cost=0.406 > 0.405; q_exec=0.505, q_exit_walked=0.099)"),
    ("wide", 0.80, 0.406, "BUY_YES", 1500.0, 2970.29702970297, 0.505, None),
    ("wide", 0.80, None, "BUY_YES", 1500.0, 2970.29702970297, 0.505, None),
    ("wide", 0.80, 0.99, "BUY_YES", 1500.0, 2970.29702970297, 0.505, None),
    ("wide", 0.80, 2.0, "BUY_YES", 1500.0, 2970.29702970297, 0.505, None),
    ("tight", 0.80, 0.05, "BUY_YES", 1500.0, 2970.29702970297, 0.505, None),
    ("tight", 0.80, 0.0, "SKIP", 0.0, 0.0, 0.0,
     "ROUND_TRIP_GATE_D (rt_cost=0.020 > 0.000; q_exec=0.505, q_exit_walked=0.485)"),
    ("tight", 0.20, 0.05, "BUY_NO", 1500.0, 2912.0559114735, 0.5151, None),
    ("tight", 0.20, None, "BUY_NO", 1500.0, 2912.0559114735, 0.5151, None),
]


def test_valid_cases_bit_identical():
    books = {"wide": _wide, "tight": _tight}
    for (bname, p, val, action, inv, shares, q, reason) in _VALID_CASES:
        for mode in ("ctor", "mut"):
            cfg = (ak.FleetRiskConfig(max_round_trip_cost=val)
                   if mode == "ctor" else _mutated(val))
            r = _eval(cfg, p=p, book=books[bname]())
            assert r.action.value == action, (bname, p, val, mode, r)
            assert r.invested_capital == inv, (bname, p, val, mode, r)
            assert r.shares_acquired == shares, (bname, p, val, mode, r)
            assert r.exec_price == q, (bname, p, val, mode, r)
            assert r.skip_reason == reason, (bname, p, val, mode, r)


def test_valid_trade_still_consumes_cluster_room():
    used = {}
    r = _eval(ak.FleetRiskConfig(max_round_trip_cost=None), used=used)
    assert r.action == ak.TradeAction.BUY_YES, r
    assert used == {"C10798": 0.15}, used


def test_gate_d_arms_with_valid_cap_unchanged():
    # Lock-scaled and edge-keyed A/B arms still read the valid field.
    cfg = ak.FleetRiskConfig(max_round_trip_cost=0.05,
                             round_trip_lock_scaled_enabled=True)
    r = _eval(cfg)
    _assert_zero_skip(r)
    assert r.skip_reason.startswith("ROUND_TRIP_GATE_D"), r.skip_reason
    cfg2 = ak.FleetRiskConfig(max_round_trip_cost=0.05,
                              round_trip_edge_keyed_enabled=True)
    r2 = _eval(cfg2)
    _assert_zero_skip(r2)
    assert r2.skip_reason.startswith("ROUND_TRIP_GATE_D_EDGE_KEYED"), r2.skip_reason
    # The lock-scaled helper's own corrupt-input fallback is untouched.
    assert ak.lock_scaled_max_round_trip_cost(None, 7.0) is None
    assert ak.lock_scaled_max_round_trip_cost(0.05, 7.0) == 0.05


def test_default_config_path_unchanged():
    r = ak.evaluate_and_size_opportunity(_tight(), 0.80, 0.0, 10000.0, {})
    assert r.action == ak.TradeAction.BUY_YES and r.invested_capital == 1500.0, r


# ---- 4. No regression on the built-in benchmark -----------------------------

def test_benchmark_nstar_unchanged():
    # Pinned to the values the PRE-PATCH file produces on this tree
    # (verified live this run: alpha_kelly_prefix.py in the task work
    # dir scores -0.1611 / -0.1499; the older 0.0547 pin in the
    # SF-BL-9397 suite already fails on the pre-patch file — a drift
    # predating this task, not caused by it).
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    assert '"north_star_score": -0.1611' in out, out[:400]
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
