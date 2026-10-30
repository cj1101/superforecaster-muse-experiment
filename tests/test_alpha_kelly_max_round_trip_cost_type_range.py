"""
test_alpha_kelly_max_round_trip_cost_type_range.py — SF-BL-10877.

Pre-registered done/kill criterion (TASK.md): "Gate D must reject a
max_round_trip_cost that is not None or a finite real in [0, 1] with
bool excluded (named SKIP or ValueError before the veto comparison),
fixtured at True, False and '0.05', with the 0.05 default and None
(disabled) bit-identical; kill if SF-BL-10798's landed fix is shown
to type-check this field as well as range-check it."

Kill NOT met in full — see KILL_AUDIT.md in the task work dir:
SF-BL-10798 type-checks the field (True / False / '0.05' were already
rejected on the inherited tree) but range-checks to [0, inf), pinning
2.0 and 1e6 as VALID. This task narrows the shared validator to the
preregistered [0, 1].

Fix: `_is_valid_max_round_trip_cost` now requires a finite real in
[0, MAX_ROUND_TRIP_COST_MAX=1.0] (bool excluded via
_is_finite_number); both enforcement layers (FleetRiskConfig
construction ValueError, evaluate Gate 2.6 SKIP INVALID_CONFIG)
inherit it. Messages keep the SF-BL-10798 token and add SF-BL-10877
plus the [0, 1] domain text.

Run:  python3 test_alpha_kelly_max_round_trip_cost_type_range.py
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

# This task's fixtures + the (1, inf) band its [0, 1] domain adds.
INVALID_TYPE = [True, False, "0.05", "1.0", "nan"]
INVALID_RANGE = [1.0000001, 1.5, 2.0, 1e6, 10 ** 400]
INVALID_PRIOR = [float("nan"), float("inf"), float("-inf"), -1.0, -1e-9]
INVALID = INVALID_TYPE + INVALID_RANGE + INVALID_PRIOR + [[], {}]
VALID = [None, 0, 0.0, 1e-9, 0.05, 0.10, 0.405, 0.406, 0.99, 1, 1.0]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(bid=0.10, cluster="C10877", days=7.0):
    return ak.OrderBookSnapshot(
        market_id="FIX-10877", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster=cluster, est_days_to_resolution=days,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000)],
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


def _eval(cfg, p=0.80, book=None, used=None, equity=10000.0):
    return ak.evaluate_and_size_opportunity(
        book if book is not None else _wide(), p, 0.0, equity,
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
    assert "[0, 1]" in r.skip_reason, r.skip_reason
    assert repr(value) in r.skip_reason, (value, r.skip_reason)
    assert "SF-BL-10798" in r.skip_reason, r.skip_reason
    assert "SF-BL-10877" in r.skip_reason, r.skip_reason
    assert used == {}, used  # cluster untouched


# ---- 1. Construction: named ValueError -------------------------------------

def test_construction_rejects_prereg_fixtures():
    for bad in (True, False, "0.05"):
        try:
            ak.FleetRiskConfig(max_round_trip_cost=bad)
        except ValueError as e:
            assert "max_round_trip_cost" in str(e), (bad, e)
            assert "[0, 1]" in str(e), (bad, e)
            assert "SF-BL-10877" in str(e), (bad, e)
        else:
            raise AssertionError(f"no ValueError for {bad!r}")


def test_construction_rejects_all_invalid():
    for bad in INVALID:
        try:
            ak.FleetRiskConfig(max_round_trip_cost=bad)
        except ValueError as e:
            assert "max_round_trip_cost" in str(e), (bad, e)
            assert "SF-BL-10877" in str(e), (bad, e)
        else:
            raise AssertionError(f"no ValueError for {bad!r}")


def test_construction_accepts_valid_and_default():
    for good in VALID:
        cfg = ak.FleetRiskConfig(max_round_trip_cost=good)
        assert cfg.max_round_trip_cost == good or (
            good is None and cfg.max_round_trip_cost is None), good
    assert ak.FleetRiskConfig().max_round_trip_cost == 0.05


def test_bool_rejected_but_boundary_numbers_accepted():
    # True == 1 and False == 0 numerically, yet the bools are invalid
    # BY TYPE while int 1 / float 1.0 / 0 are valid: the [0, 1]
    # boundary is inclusive for reals, exclusive for bools.
    assert ak._is_valid_max_round_trip_cost(1) is True
    assert ak._is_valid_max_round_trip_cost(1.0) is True
    assert ak._is_valid_max_round_trip_cost(0) is True
    assert ak._is_valid_max_round_trip_cost(True) is False
    assert ak._is_valid_max_round_trip_cost(False) is False
    assert ak.MAX_ROUND_TRIP_COST_MAX == 1.0


def test_helpers_directly():
    for good in VALID:
        assert ak._is_valid_max_round_trip_cost(good) is True, good
    for bad in INVALID:
        assert ak._is_valid_max_round_trip_cost(bad) is False, bad
    try:
        ak._validate_max_round_trip_cost(1.5)
    except ValueError as e:
        assert "max_round_trip_cost" in str(e) and "[0, 1]" in str(e), e
        assert "SF-BL-10877" in str(e), e
    else:
        raise AssertionError("helper accepted 1.5")


# ---- 2. Gate 2.6: mutated / unpickled invalid configs SKIP -----------------

def test_mutated_prereg_fixtures_skip_on_wide_book():
    for bad in (True, False, "0.05"):
        used = {}
        r = _eval(_mutated(bad), used=used)
        _assert_invalid_config(r, used, bad)


def test_mutated_above_one_skips_on_wide_book():
    # The residual gap: pre-fix these traded $1,500 (equity $10k).
    for bad in INVALID_RANGE:
        used = {}
        r = _eval(_mutated(bad), used=used)
        _assert_invalid_config(r, used, bad)


def test_title_scale_15000_no_longer_trades():
    # Equity $100,000: the 15% cluster cap is $15,000 — the title
    # figure. 0.05 vetoes it; None trades it BY DESIGN (disabled);
    # True and 1.5 must SKIP as invalid config, never trade it.
    for bad in (True, 1.5, "0.05"):
        used = {}
        r = _eval(_mutated(bad), used=used, equity=100000.0)
        _assert_invalid_config(r, used, bad)
    r = _eval(ak.FleetRiskConfig(max_round_trip_cost=0.05),
              equity=100000.0)
    _assert_zero_skip(r)
    assert r.skip_reason.startswith("ROUND_TRIP_GATE_D"), r.skip_reason
    r = _eval(ak.FleetRiskConfig(max_round_trip_cost=None),
              equity=100000.0)
    assert r.action == ak.TradeAction.BUY_YES, r
    assert r.invested_capital == 15000.0, r


def test_mutated_invalid_skips_on_tight_book_and_buy_no():
    for bad in (True, "0.05", 1.5, 2.0):
        used = {}
        r = _eval(_mutated(bad), book=_tight(), used=used)
        _assert_invalid_config(r, used, bad)
        used = {}
        r = _eval(_mutated(bad), p=0.20, book=_tight(), used=used)
        _assert_invalid_config(r, used, bad)


def test_string_threshold_never_raises_typeerror():
    r = _eval(_mutated("0.05"))
    _assert_zero_skip(r)
    assert "'0.05'" in r.skip_reason, r.skip_reason


def test_unpickled_and_raw_configs_skip():
    cfg = pickle.loads(pickle.dumps(ak.FleetRiskConfig()))
    cfg.max_round_trip_cost = 2.0
    used = {}
    r = _eval(cfg, used=used)
    _assert_invalid_config(r, used, 2.0)
    raw = object.__new__(ak.FleetRiskConfig)
    raw.min_net_edge, raw.min_confidence = 0.05, 0.60
    raw.kelly_multiplier, raw.crypto_kelly_dampener = 0.35, 0.50
    raw.max_cluster_allocation = 0.15
    raw.max_inter_agent_dispersion = 0.22
    raw.require_on_chain_for_crypto = True
    raw.max_round_trip_cost = True
    raw.lock_scaled_hurdle_enabled = False
    raw.dispersion_taper_enabled = False
    raw.budget_sensitivity_veto_enabled = True
    used = {}
    r = _eval(raw, used=used)
    _assert_invalid_config(r, used, True)


def test_earlier_gates_keep_precedence():
    r = ak.evaluate_and_size_opportunity(
        _wide(), float("nan"), 0.0, 10000.0, {},
        config=_mutated(1.5))
    _assert_zero_skip(r)
    assert r.skip_reason.startswith("INVALID_POSTERIOR"), r.skip_reason


# ---- 3. In-domain values: bit-identical pins -------------------------------
# Recorded from the PRE-PATCH (inherited) file on these fixtures; the
# 162-case pre/post snapshot diff in the task work dir is empty.

_VALID_CASES = [
    ("wide", 0.80, 0.05, "SKIP", 0.0, 0.0, 0.0,
     "ROUND_TRIP_GATE_D (rt_cost=0.406 > 0.050; q_exec=0.505, q_exit_walked=0.099)"),
    ("wide", 0.80, 0.0, "SKIP", 0.0, 0.0, 0.0,
     "ROUND_TRIP_GATE_D (rt_cost=0.406 > 0.000; q_exec=0.505, q_exit_walked=0.099)"),
    ("wide", 0.80, 0.405, "SKIP", 0.0, 0.0, 0.0,
     "ROUND_TRIP_GATE_D (rt_cost=0.406 > 0.405; q_exec=0.505, q_exit_walked=0.099)"),
    ("wide", 0.80, 0.406, "BUY_YES", 1500.0, 2970.29702970297, 0.505, None),
    ("wide", 0.80, None, "BUY_YES", 1500.0, 2970.29702970297, 0.505, None),
    ("wide", 0.80, 0.99, "BUY_YES", 1500.0, 2970.29702970297, 0.505, None),
    ("wide", 0.80, 1, "BUY_YES", 1500.0, 2970.29702970297, 0.505, None),
    ("wide", 0.80, 1.0, "BUY_YES", 1500.0, 2970.29702970297, 0.505, None),
    ("tight", 0.80, 0.05, "BUY_YES", 1500.0, 2970.29702970297, 0.505, None),
    ("tight", 0.80, 0.0, "SKIP", 0.0, 0.0, 0.0,
     "ROUND_TRIP_GATE_D (rt_cost=0.020 > 0.000; q_exec=0.505, q_exit_walked=0.485)"),
    ("tight", 0.20, 0.05, "BUY_NO", 1500.0, 2912.0559114735, 0.5151, None),
    ("tight", 0.20, None, "BUY_NO", 1500.0, 2912.0559114735, 0.5151, None),
    ("tight", 0.20, 1.0, "BUY_NO", 1500.0, 2912.0559114735, 0.5151, None),
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


def test_lock_scaled_effective_cap_may_exceed_one():
    # The [0, 1] bound is on the FIELD (base). Base 1.0 on a 28d lock
    # scales to an effective cap of 2.0 and still trades the wide
    # book, exactly as pre-fix (snapshot-pinned).
    assert ak.lock_scaled_max_round_trip_cost(1.0, 28.0) == 2.0
    cfg = ak.FleetRiskConfig(max_round_trip_cost=1.0,
                             round_trip_lock_scaled_enabled=True)
    r = _eval(cfg, book=_book(bid=0.10, days=28.0))
    assert r.action == ak.TradeAction.BUY_YES, r
    assert r.invested_capital == 1500.0, r
    # But an out-of-domain BASE is rejected even with the arm on.
    used = {}
    cfg2 = _mutated(1.5)
    cfg2.round_trip_lock_scaled_enabled = True
    r2 = _eval(cfg2, book=_book(bid=0.10, days=28.0), used=used)
    _assert_invalid_config(r2, used, 1.5)


def test_edge_keyed_arm_and_default_path_unchanged():
    cfg = ak.FleetRiskConfig(max_round_trip_cost=0.05,
                             round_trip_edge_keyed_enabled=True)
    r = _eval(cfg)
    _assert_zero_skip(r)
    assert r.skip_reason.startswith("ROUND_TRIP_GATE_D_EDGE_KEYED"), \
        r.skip_reason
    r = ak.evaluate_and_size_opportunity(_tight(), 0.80, 0.0, 10000.0, {})
    assert r.action == ak.TradeAction.BUY_YES, r
    assert r.invested_capital == 1500.0, r


def test_valid_trade_still_consumes_cluster_room():
    used = {}
    r = _eval(ak.FleetRiskConfig(max_round_trip_cost=None), used=used)
    assert r.action == ak.TradeAction.BUY_YES, r
    assert used == {"C10877": 0.15}, used


# ---- 4. Benchmark -----------------------------------------------------------

def test_benchmark_nstar_unchanged():
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
