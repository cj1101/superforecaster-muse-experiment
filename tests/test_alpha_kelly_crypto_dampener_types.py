"""
test_crypto_dampener_types.py — SF-BL-10878 verification tests.

Pre-registered done/kill criterion: "evaluate must SKIP with a named
invalid-dampener reason (zero size, cluster dict untouched) for a bool,
string or None crypto_kelly_dampener before sizing, fixtured at True,
False, '0.5' and None, with dampener 0.50 and all non-crypto books
bit-identical; kill if SF-BL-10821's landed fix is shown to type-check
this field as well as range-check it."
Kill NOT met — see KILL_AUDIT.md (SF-BL-10821 has not landed: its work
dir holds only TASK.md, the pre-fix file carried no dampener validator,
and no repo test type-checks the field).

Pre-fix reproduction (verified live, repro_prepatch.txt): fixture
(verified crypto book ask 0.60 / bid 0.58, fee 0.01, p=0.70, equity
$100,000) — dampener 0.50 BUY_YES $4,175.13; True BUY_YES $8,350.25
(exactly double — bool removed the dampener); False SKIP
CLUSTER_CAP_REACHED (mislabel); '0.5' / None bare TypeError;
construction accepted every value silently.

Fix (defence in depth, mirroring SF-BL-9397 / SF-BL-10798):
(a) FleetRiskConfig.__post_init__ raises ValueError naming the field,
    value, type and SF-BL-10878 unless the dampener is a finite real
    in (0, 1] (bool excluded via _is_finite_number).
(b) New Gate 2.7 in evaluate_and_size_opportunity, after Gate 2.6 and
    before Step 4's first read of the field, SKIPs a mutated /
    unpickled CRYPTO config INVALID_CONFIG /
    INVALID_CRYPTO_KELLY_DAMPENER at zero size (EC-09), cluster dict
    untouched — never a TypeError, never a trade, never the
    CLUSTER_CAP_REACHED mislabel. Scoped to CRYPTO (the only path
    that reads the field, Gate 0.9 precedent), so non-crypto books
    are bit-identical for EVERY dampener value.

Run:  python3 test_crypto_dampener_types.py  (exit 0 iff all pass)
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

# The four prereg type fixtures first, then the wider invalid set
# (SF-BL-10821 range failures share the same (0, 1] predicate).
PREREG_TYPES = [True, False, "0.5", None]
INVALID_DAMPENER = PREREG_TYPES + [
    "1.0", "0", 2.0, -1.0, 0.0, 0, -0.5, 1.5, float("nan"),
    float("inf"), float("-inf"), [], {}, 10 ** 400,
]
VALID_DAMPENER = [0.50, 1.0, 1, 0.25, 0.01, 0.999999, 0.75]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(domain=ak.MarketDomain.CRYPTO, ask=0.60, bid=0.58,
          cluster="C10878"):
    return ak.OrderBookSnapshot(
        market_id="FIXTURE-10878", title="dampener type fixture",
        domain=domain, factor_cluster=cluster,
        est_days_to_resolution=10,
        yes_asks=[ak.OrderBookLevel(price=ask, size_shares=1000000)],
        yes_bids=[ak.OrderBookLevel(price=bid, size_shares=1000000)],
        fee_rate=0.01, on_chain_verified=True)


def _mutated(**kw):
    cfg = ak.FleetRiskConfig()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _eval(cfg, p=0.70, book=None, used=None):
    return ak.evaluate_and_size_opportunity(
        book if book is not None else _book(), p, 0.0, 100000.0,
        used if used is not None else {}, config=cfg)


def _assert_zero_skip(r):
    assert r.action == ak.TradeAction.SKIP, r
    assert r.invested_capital == 0.0 and r.shares_acquired == 0.0, r
    assert r.exec_price == 0.0 and r.liquidation_bid_price == 0.0, r
    assert r.kelly_fraction_raw == 0.0 and r.allocated_fraction == 0.0, r
    assert r.kelly_fraction_sized == 0.0, r
    assert r.kl_divergence_nats == 0.0 and r.expected_log_growth == 0.0, r


def _assert_invalid_dampener_skip(r, bad):
    _assert_zero_skip(r)
    assert r.skip_reason is not None, (bad, r)
    assert r.skip_reason.startswith("INVALID_CONFIG"), (bad, r.skip_reason)
    assert "INVALID_CRYPTO_KELLY_DAMPENER" in r.skip_reason, (bad, r.skip_reason)
    assert "crypto_kelly_dampener" in r.skip_reason, (bad, r.skip_reason)
    assert "SF-BL-10878" in r.skip_reason, (bad, r.skip_reason)
    assert "CLUSTER_CAP_REACHED" not in r.skip_reason, (bad, r.skip_reason)


def _tuple(r, used):
    return (r.action.value, r.invested_capital, r.shares_acquired,
            r.exec_price, r.kelly_fraction_raw, r.kelly_fraction_sized,
            r.allocated_fraction, r.liquidation_bid_price, r.skip_reason,
            dict(used))


# ---- 1. Construction: named ValueError for invalid, valid accepted ---------

def test_construction_rejects_invalid_dampener():
    for bad in INVALID_DAMPENER:
        try:
            ak.FleetRiskConfig(crypto_kelly_dampener=bad)
        except ValueError as e:
            assert "crypto_kelly_dampener" in str(e), (bad, e)
            assert "SF-BL-10878" in str(e), (bad, e)
        else:
            raise AssertionError(f"no ValueError for dampener {bad!r}")


def test_construction_accepts_valid_and_default():
    for good in VALID_DAMPENER:
        assert ak.FleetRiskConfig(
            crypto_kelly_dampener=good).crypto_kelly_dampener == good
    assert ak.FleetRiskConfig().crypto_kelly_dampener == 0.50


def test_construction_other_fields_unaffected():
    # Earlier fields keep their own validators / first-violation order.
    try:
        ak.FleetRiskConfig(kelly_multiplier=2.0,
                           crypto_kelly_dampener=True)
    except ValueError as e:
        assert "kelly_multiplier" in str(e), e
    else:
        raise AssertionError("wild kelly accepted")
    assert ak.FleetRiskConfig(max_round_trip_cost=None) is not None
    assert ak.FleetRiskConfig(min_confidence=0.0) is not None


def test_helpers_directly():
    for good in VALID_DAMPENER:
        assert ak._is_valid_crypto_kelly_dampener(good) is True, good
    for bad in INVALID_DAMPENER:
        assert ak._is_valid_crypto_kelly_dampener(bad) is False, bad
    # True == 1.0 numerically, yet the type check must still reject it.
    assert ak._is_valid_crypto_kelly_dampener(1.0) is True
    assert ak._is_valid_crypto_kelly_dampener(True) is False


# ---- 2. Gate 2.7: mutated invalid dampeners SKIP on crypto, never trade ----

def test_prereg_type_fixtures_skip_invalid_dampener():
    for bad in PREREG_TYPES:
        used = {}
        r = _eval(_mutated(crypto_kelly_dampener=bad), used=used)
        _assert_invalid_dampener_skip(r, bad)
        assert used == {}, (bad, used)  # cluster untouched


def test_full_invalid_set_skips_on_crypto():
    for bad in INVALID_DAMPENER:
        used = {}
        r = _eval(_mutated(crypto_kelly_dampener=bad), used=used)
        _assert_invalid_dampener_skip(r, bad)
        assert used == {}, (bad, used)


def test_headline_true_no_longer_doubles_stake():
    # Pre-fix True sized $8,350.25 (2x the $4,175.13 mandated stake).
    r = _eval(_mutated(crypto_kelly_dampener=True))
    _assert_invalid_dampener_skip(r, True)
    clean = _eval(ak.FleetRiskConfig())
    assert clean.invested_capital == 4175.126903553298, clean


def test_false_no_longer_mislabels_cluster_cap():
    r = _eval(_mutated(crypto_kelly_dampener=False))
    _assert_invalid_dampener_skip(r, False)


def test_string_and_none_no_longer_raise_typeerror():
    for bad in ("0.5", None):
        r = _eval(_mutated(crypto_kelly_dampener=bad))  # must not raise
        _assert_invalid_dampener_skip(r, bad)
        assert repr(bad) in r.skip_reason, (bad, r.skip_reason)


def test_invalid_dampener_skips_on_buy_no_side_too():
    for bad in PREREG_TYPES:
        used = {}
        r = _eval(_mutated(crypto_kelly_dampener=bad), p=0.30, used=used)
        _assert_invalid_dampener_skip(r, bad)
        assert used == {}, (bad, used)


def test_unpickled_and_raw_configs_skip():
    cfg = pickle.loads(pickle.dumps(ak.FleetRiskConfig()))
    cfg.crypto_kelly_dampener = True  # post-load mutation
    r = _eval(cfg)
    _assert_invalid_dampener_skip(r, True)
    raw = object.__new__(ak.FleetRiskConfig)
    raw.min_net_edge, raw.min_confidence = 0.05, 0.60
    raw.kelly_multiplier = 0.35
    # __post_init__ never ran on a __new__ instance, so this invalid
    # value reaches evaluate unchecked by construction.
    raw.crypto_kelly_dampener = "0.5"
    raw.max_cluster_allocation = 0.15
    raw.max_inter_agent_dispersion = 0.22
    raw.require_on_chain_for_crypto, raw.max_round_trip_cost = True, 0.05
    raw.lock_scaled_hurdle_enabled = False
    raw.budget_sensitivity_veto_enabled = True
    r2 = _eval(raw)
    _assert_invalid_dampener_skip(r2, "0.5")


def test_earlier_gates_keep_precedence():
    # Invalid posterior (Gate 0.0) still precedes Gate 2.7.
    r = ak.evaluate_and_size_opportunity(
        _book(), float("nan"), 0.0, 100000.0, {},
        config=_mutated(crypto_kelly_dampener=True))
    _assert_zero_skip(r)
    assert r.skip_reason.startswith("INVALID_POSTERIOR"), r.skip_reason
    # Gate 1 (unverified crypto) still precedes Gate 2.7.
    b = _book()
    b.on_chain_verified = False
    r2 = ak.evaluate_and_size_opportunity(
        b, 0.70, 0.0, 100000.0, {},
        config=_mutated(crypto_kelly_dampener=True))
    _assert_zero_skip(r2)
    assert r2.skip_reason == "FAILED_CRYPTOBENCH_ONCHAIN_GATE", r2.skip_reason
    # Gate 2.5 (wild kelly) still precedes Gate 2.7.
    r3 = _eval(_mutated(kelly_multiplier=2.0, crypto_kelly_dampener=True))
    _assert_zero_skip(r3)
    assert "kelly_multiplier" in r3.skip_reason, r3.skip_reason


# ---- 3. Non-crypto books: bit-identical for EVERY dampener value -----------

def test_noncrypto_bit_identical_all_dampeners():
    for domain in (ak.MarketDomain.PREDICTION_MARKET,
                   ak.MarketDomain.EQUITIES_MACRO):
        used0 = {}
        base = _tuple(_eval(ak.FleetRiskConfig(), book=_book(domain),
                            used=used0), used0)
        assert base[0] == "BUY_YES" and base[1] == 8350.253807106596, base
        for damp in VALID_DAMPENER + INVALID_DAMPENER:
            used = {}
            r = _eval(_mutated(crypto_kelly_dampener=damp),
                      book=_book(domain), used=used)
            assert _tuple(r, used) == base, (domain, damp, r, base)


# ---- 4. Valid crypto configs: byte-identical to pre-fix snapshots ----------
# Snapshots recorded from the PRE-PATCH file (snapshots_prepatch.txt)
# on this suite's fixture (ask 0.60 / bid 0.58, size 1e6, fee 0.01,
# equity $100,000).

_VALID_CASES = [
    # label, dampener, p, action, invested, shares, exec, kelly_raw, sized, alloc, liq
    ("damp0.50-mandated", 0.50, 0.70, "BUY_YES", 4175.126903553298,
     6889.648355698511, 0.606, 0.2385786802030456, 0.04175126903553298,
     0.04175126903553298, 0.58),
    ("damp1.0", 1.0, 0.70, "BUY_YES", 8350.253807106596,
     13779.296711397023, 0.606, 0.2385786802030456, 0.08350253807106596,
     0.08350253807106596, 0.58),
    ("damp-int-1", 1, 0.70, "BUY_YES", 8350.253807106596,
     13779.296711397023, 0.606, 0.2385786802030456, 0.08350253807106596,
     0.08350253807106596, 0.58),
    ("damp0.25", 0.25, 0.70, "BUY_YES", 2087.563451776649,
     3444.8241778492556, 0.606, 0.2385786802030456, 0.02087563451776649,
     0.02087563451776649, 0.58),
    ("damp0.50-buy-no", 0.50, 0.30, "BUY_NO", 8356.844305120165,
     19700.245886657627, 0.42419999999999997, 0.47753396029258083,
     0.08356844305120165, 0.08356844305120165, 0.4),
]


def test_valid_crypto_cases_byte_identical():
    for (label, damp, p, action, inv, shares, q, kelly, sized,
         alloc, liq) in _VALID_CASES:
        used = {}
        r = _eval(ak.FleetRiskConfig(crypto_kelly_dampener=damp), p=p,
                  used=used)
        assert r.action.value == action, (label, r)
        assert r.invested_capital == inv, (label, r.invested_capital)
        assert r.shares_acquired == shares, (label, r.shares_acquired)
        assert r.exec_price == q, (label, r.exec_price)
        assert r.kelly_fraction_raw == kelly, (label, r.kelly_fraction_raw)
        assert r.kelly_fraction_sized == sized, (label, r.kelly_fraction_sized)
        assert r.allocated_fraction == alloc, (label, r.allocated_fraction)
        assert r.liquidation_bid_price == liq, (label, r.liquidation_bid_price)
        assert r.skip_reason is None, (label, r.skip_reason)
        assert used == {"C10878": sized}, (label, used)


def test_default_config_path_unchanged():
    used = {}
    r = ak.evaluate_and_size_opportunity(_book(), 0.70, 0.0, 100000.0, used)
    assert r.action == ak.TradeAction.BUY_YES, r
    assert r.invested_capital == 4175.126903553298, r
    assert used == {"C10878": 0.04175126903553298}, used


def test_constructed_valid_matches_mutated():
    a = _eval(ak.FleetRiskConfig(crypto_kelly_dampener=0.25))
    b = _eval(_mutated(crypto_kelly_dampener=0.25))
    assert (a.action, a.invested_capital, a.allocated_fraction) == \
           (b.action, b.invested_capital, b.allocated_fraction), (a, b)


# ---- 5. No regression on the built-in benchmark -----------------------------

def test_benchmark_nstar_unchanged():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    # Values captured from the PRE-PATCH file on this run's baseline
    # (benchmark_prepatch.txt; full output byte-identical pre/post).
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
