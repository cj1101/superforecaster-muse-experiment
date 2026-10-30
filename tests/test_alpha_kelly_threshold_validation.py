"""
test_threshold_validation.py — SF-BL-10844 verification tests.

Pre-registered done/kill criterion: "Done when evaluate SKIPs zero-size
with an explicit invalid-threshold reason (or raises a named ValueError)
for any max_inter_agent_dispersion that is not a finite real >= 0, with
the clean 0.22-threshold fixtures byte-identical; kill if a caller
census shows every FleetRiskConfig is constructed in-repo with a
validated literal threshold." (Kill NOT met — see CALLER_CENSUS.md:
no in-repo construction passes a threshold at all, the class had no
__post_init__, and it is a mutable dataclass, so mutation/unpickling
bypass construction regardless.)

Pre-fix reproduction (verified live 2026-10-22, prereg fixture book
p=0.80, ask 0.50 / bid 0.48, fee 0.01, equity $100,000, dispersion 0.50):
    threshold= 0.22  -> SKIP UNRESOLVED_AGENT_DISPERSION (0.50 > 0.22)
    threshold= nan   -> BUY_YES invested $15,000.00 (gate fails open)
    threshold= +inf  -> BUY_YES invested $15,000.00 (gate fails open)
    threshold= -inf  -> SKIP UNRESOLVED (incidental, wrong reason)
    threshold= '0.22'-> RAISE bare TypeError from the Gate 2 comparison
    threshold= -0.10 -> SKIP UNRESOLVED (incidental, wrong reason)
    threshold= True  -> BUY_YES $15,000.00 (bool masquerades as 1.0)
    construction with any of the above -> succeeded silently (no
                        __post_init__ on FleetRiskConfig at all)

Fix (defence in depth, mirroring SF-BL-9363 fee_rate):
(a) FleetRiskConfig.__post_init__ raises ValueError (via
    _validate_max_inter_agent_dispersion) for any threshold that is
    not a finite real >= 0 (bool excluded). NOTE: landed concurrently
    with SF-BL-10845, whose validator shares the single merged
    __post_init__ — a second def would shadow the first.
(b) New Gate 1.9 in evaluate_and_size_opportunity, immediately before
    Gate 2, SKIPs a mutated/unpickled config's invalid threshold with
    INVALID_MAX_INTER_AGENT_DISPERSION and zero size (EC-09), cluster
    dict untouched — never a TypeError, never a trade. Valid thresholds
    pass untouched, so Gate 2 behaviour is byte-identical.

Run:  python3 test_threshold_validation.py   (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_dispersion.py.
"""
import contextlib
import io
import math
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

INVALID_THRESHOLDS = [
    float("nan"), float("inf"), float("-inf"),
    -0.10, -1, -1e308,
    "0.22", "", "nan", None, True, False, [], {}, object(),
]
VALID_THRESHOLDS = [0.22, 0, 0.0, 1, 1.0, 2, 0.10, 1e308]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(domain=ak.MarketDomain.PREDICTION_MARKET, on_chain=True):
    return ak.OrderBookSnapshot(
        market_id="FIXTURE-10844", title="prereg fixture", domain=domain,
        factor_cluster="C10844", est_days_to_resolution=10,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100000)],
        fee_rate=0.01, on_chain_verified=on_chain)


def _cfg_mutated(thr):
    """Config with threshold set by post-construction mutation (the
    __post_init__ bypass: mutated / unpickled / shared-default configs)."""
    cfg = ak.FleetRiskConfig()
    cfg.max_inter_agent_dispersion = thr
    return cfg


def _eval(disp, thr, p=0.80, used=None, book=None):
    return ak.evaluate_and_size_opportunity(
        book if book is not None else _book(), p, disp, 100000.0,
        used if used is not None else {}, config=_cfg_mutated(thr))


def _assert_zero_skip(r):
    assert r.action == ak.TradeAction.SKIP, r
    assert r.invested_capital == 0.0 and r.shares_acquired == 0.0, r
    assert r.exec_price == 0.0 and r.liquidation_bid_price == 0.0, r
    assert r.kelly_fraction_raw == 0.0 and r.allocated_fraction == 0.0, r  # EC-09
    assert r.kl_divergence_nats == 0.0 and r.expected_log_growth == 0.0, r


# ---- 1. Construction: named ValueError for invalid, valid accepted ---------

def test_construction_rejects_invalid_thresholds():
    for bad in INVALID_THRESHOLDS:
        try:
            ak.FleetRiskConfig(max_inter_agent_dispersion=bad)
        except ValueError as e:
            assert "max_inter_agent_dispersion" in str(e), (bad, e)
            assert "SF-BL-10844" in str(e), (bad, e)
        else:
            raise AssertionError(f"no ValueError for threshold {bad!r}")


def test_construction_accepts_valid_thresholds_and_default():
    for good in VALID_THRESHOLDS:
        cfg = ak.FleetRiskConfig(max_inter_agent_dispersion=good)
        assert cfg.max_inter_agent_dispersion == good, (good, cfg)
    assert ak.FleetRiskConfig().max_inter_agent_dispersion == 0.22


def test_construction_of_other_fields_unaffected():
    # Existing in-repo constructions (other fields only) still work.
    assert ak.FleetRiskConfig(max_round_trip_cost=None).max_round_trip_cost is None
    assert ak.FleetRiskConfig(require_on_chain_for_crypto=False) is not None


# ---- 2. Gate: mutated/unpickled invalid thresholds SKIP, never trade -------

def test_invalid_thresholds_skip_with_invalid_reason():
    for bad in INVALID_THRESHOLDS:
        r = _eval(0.50, bad)
        _assert_zero_skip(r)
        assert r.skip_reason is not None, (bad, r)
        assert r.skip_reason.startswith("INVALID_MAX_INTER_AGENT_DISPERSION"), (bad, r.skip_reason)
        assert "not a finite real >= 0" in r.skip_reason, (bad, r.skip_reason)
        assert "UNRESOLVED_AGENT_DISPERSION" not in r.skip_reason, (bad, r.skip_reason)


def test_headline_fail_open_is_closed():
    # The prereg bug: dispersion 0.50 must NOT trade at NaN/+inf threshold.
    for bad in (float("nan"), float("inf")):
        used = {}
        r = _eval(0.50, bad, used=used)
        _assert_zero_skip(r)
        assert used == {}, (bad, used)  # sizing never reached, cluster untouched


def test_string_threshold_no_longer_raises_typeerror():
    # Pre-fix this raised a bare TypeError; post-fix it is a named SKIP.
    r = _eval(0.50, "0.22")
    _assert_zero_skip(r)
    assert "'0.22'" in r.skip_reason, r.skip_reason


def test_invalid_threshold_skips_on_buy_no_side_too():
    for bad in (float("nan"), float("inf"), "0.22", -0.10):
        r = _eval(0.50, bad, p=0.20)
        _assert_zero_skip(r)
        assert r.skip_reason.startswith("INVALID_MAX_INTER_AGENT_DISPERSION"), (bad, r.skip_reason)


def test_invalid_threshold_gate_precedes_sizing_at_low_dispersion():
    # Even a dispersion that would trade at a valid threshold is blocked.
    for bad in (float("nan"), float("inf"), None):
        r = _eval(0.10, bad)
        _assert_zero_skip(r)
        assert r.skip_reason.startswith("INVALID_MAX_INTER_AGENT_DISPERSION"), (bad, r.skip_reason)


def test_unpickled_config_with_invalid_threshold_skips():
    cfg = pickle.loads(pickle.dumps(ak.FleetRiskConfig()))
    cfg.max_inter_agent_dispersion = float("nan")  # post-load mutation
    r = ak.evaluate_and_size_opportunity(_book(), 0.80, 0.50, 100000.0, {}, config=cfg)
    _assert_zero_skip(r)
    assert r.skip_reason.startswith("INVALID_MAX_INTER_AGENT_DISPERSION"), r.skip_reason
    # And a config materialised without __init__/__post_init__ at all:
    raw = object.__new__(ak.FleetRiskConfig)
    raw.min_net_edge, raw.min_confidence = 0.05, 0.60
    raw.kelly_multiplier, raw.crypto_kelly_dampener = 0.35, 0.50
    raw.max_cluster_allocation = 0.15
    raw.max_inter_agent_dispersion = "0.22"
    raw.require_on_chain_for_crypto, raw.max_round_trip_cost = True, 0.05
    r2 = ak.evaluate_and_size_opportunity(_book(), 0.80, 0.50, 100000.0, {}, config=raw)
    _assert_zero_skip(r2)
    assert r2.skip_reason.startswith("INVALID_MAX_INTER_AGENT_DISPERSION"), r2.skip_reason


def test_helper_directly():
    for good in VALID_THRESHOLDS:
        assert ak._is_valid_max_inter_agent_dispersion(good) is True, good
    for bad in INVALID_THRESHOLDS:
        assert ak._is_valid_max_inter_agent_dispersion(bad) is False, bad
    assert math.isfinite(0.22)  # sanity


# ---- 3. Valid thresholds: byte-identical to pre-fix snapshots --------------
# Expected snapshots recorded from the PRE-FIX code (full field tuples).

_VALID_CASES = [
    # disp, thr, p, action, invested, shares, exec, kelly_raw, alloc, skip_reason
    (0.50, 0.22, 0.80, "SKIP", 0.0, 0.0, 0.0, 0.0, 0.0,
     "UNRESOLVED_AGENT_DISPERSION (0.50 > 0.22)"),
    (0.10, 0.22, 0.80, "BUY_YES", 15000.0, 29702.970297029704, 0.505,
     0.595959595959596, 0.15, None),
    (0.22, 0.22, 0.80, "BUY_YES", 15000.0, 29702.970297029704, 0.505,
     0.595959595959596, 0.15, None),  # boundary: > is strict, 0.22 trades
    (0.00, 0.22, 0.80, "BUY_YES", 15000.0, 29702.970297029704, 0.505,
     0.595959595959596, 0.15, None),
    (0.50, 0, 0.80, "SKIP", 0.0, 0.0, 0.0, 0.0, 0.0,
     "UNRESOLVED_AGENT_DISPERSION (0.50 > 0.00)"),
    (0.00, 0, 0.80, "BUY_YES", 15000.0, 29702.970297029704, 0.505,
     0.595959595959596, 0.15, None),
    (0.50, 1, 0.80, "BUY_YES", 15000.0, 29702.970297029704, 0.505,
     0.595959595959596, 0.15, None),
    (0.50, 0.10, 0.80, "SKIP", 0.0, 0.0, 0.0, 0.0, 0.0,
     "UNRESOLVED_AGENT_DISPERSION (0.50 > 0.10)"),
    (0.05, 2, 0.80, "BUY_YES", 15000.0, 29702.970297029704, 0.505,
     0.595959595959596, 0.15, None),
    (1, 0.5, 0.80, "SKIP", 0.0, 0.0, 0.0, 0.0, 0.0,
     "UNRESOLVED_AGENT_DISPERSION (1.00 > 0.50)"),
    (0, 0.5, 0.80, "BUY_YES", 15000.0, 29702.970297029704, 0.505,
     0.595959595959596, 0.15, None),
    (0.10, 0.22, 0.20, "BUY_NO", 15000.0, 28560.54836252856, 0.5252,
     0.5787700084245999, 0.15, None),
    (0.50, 0.22, 0.20, "SKIP", 0.0, 0.0, 0.0, 0.0, 0.0,
     "UNRESOLVED_AGENT_DISPERSION (0.50 > 0.22)"),
]


def test_valid_cases_byte_identical():
    for disp, thr, p, action, inv, shares, q, kelly, alloc, reason in _VALID_CASES:
        r = _eval(disp, thr, p=p)
        assert r.action.value == action, (disp, thr, r)
        assert r.invested_capital == inv, (disp, thr, r.invested_capital)
        assert r.shares_acquired == shares, (disp, thr, r.shares_acquired)
        assert r.exec_price == q, (disp, thr, r.exec_price)
        assert r.kelly_fraction_raw == kelly, (disp, thr, r.kelly_fraction_raw)
        assert r.allocated_fraction == alloc, (disp, thr, r.allocated_fraction)
        assert r.skip_reason == reason, (disp, thr, r.skip_reason)


def test_constructed_valid_config_matches_mutated():
    # A config CONSTRUCTED with a valid threshold behaves identically to
    # the same threshold set by mutation (both reach Gate 2 untouched).
    cfg = ak.FleetRiskConfig(max_inter_agent_dispersion=0.22)
    r = ak.evaluate_and_size_opportunity(_book(), 0.80, 0.50, 100000.0, {}, config=cfg)
    assert r.skip_reason == "UNRESOLVED_AGENT_DISPERSION (0.50 > 0.22)", r.skip_reason


def test_default_config_path_unchanged():
    r = ak.evaluate_and_size_opportunity(_book(), 0.80, 0.50, 100000.0, {})
    assert r.skip_reason == "UNRESOLVED_AGENT_DISPERSION (0.50 > 0.22)", r.skip_reason
    r2 = ak.evaluate_and_size_opportunity(_book(), 0.80, 0.10, 100000.0, {})
    assert r2.action == ak.TradeAction.BUY_YES and r2.invested_capital == 15000.0, r2


def test_valid_trade_still_consumes_cluster_room():
    used = {}
    r = _eval(0.10, 0.22, used=used)
    assert r.action == ak.TradeAction.BUY_YES, r
    assert used == {"C10844": 0.15}, used


def test_earlier_gates_keep_precedence():
    # An invalid POSTERIOR on a book whose config has an invalid
    # threshold still reports INVALID_POSTERIOR (Gate 0.0 runs first).
    r = ak.evaluate_and_size_opportunity(
        _book(), float("nan"), 0.50, 100000.0, {}, config=_cfg_mutated(float("nan")))
    _assert_zero_skip(r)
    assert r.skip_reason.startswith("INVALID_POSTERIOR"), r.skip_reason


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
        except Exception as e:  # noqa: BLE001 - a raise is a failure here
            check(fn.__name__, False, f"{type(e).__name__}: {e}")
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
