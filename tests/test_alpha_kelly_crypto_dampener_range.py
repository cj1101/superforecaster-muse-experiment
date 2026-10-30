"""
test_alpha_kelly_crypto_dampener_range.py — SF-BL-10821 verification.

Pre-registered done/kill criterion: "a crypto_kelly_dampener outside
(0, 1] must be rejected or SKIP with a named invalid-dampener reason
before sizing, fixtured at 2.0, -1.0, 0.0 and NaN, with dampener=0.50
(mandated) and non-crypto books bit-identical; kill if a producer
census shows dampener is only ever the literal 0.50 default."
Kill NOT met — see PRODUCER_CENSUS.md in the task work dir (call
sites pass only 0.50, but the mutable dataclass, unpickling, the
shared mutable default and the provenance-free evaluate signature
make that an observation, not an invariant; the normal construction
path itself accepted 2.0 pre-fix).

Task fixture (verified live repros in TASK.md): CRYPTO book ask
0.60 / bid 0.58, fee 0.01, p=0.70, on_chain_verified=True. At equity
$10,000 the mandated dampener 0.50 sizes $417.51, an unbound 2.0
sizes at the full $1,500 cluster cap, and in-domain 1.0 sizes
$835.03 — exactly double the mandated stake. Pre-fix, -1.0 SKIPped
behind the misleading CLUSTER_CAP_REACHED and NaN behind
ORDER_BOOK_DEPTH_EXHAUSTED; both mislabels are asserted absent.

Fix surface (landed by SF-BL-10878 as a single predicate covering
both preregs; this task adds the SF-BL-10821 citations and pins the
range fixtures): _is_valid_crypto_kelly_dampener = finite real in
(0, 1], bool excluded; FleetRiskConfig.__post_init__ raises a named
ValueError; Gate 2.7 in evaluate_and_size_opportunity SKIPs a
mutated/unpickled CRYPTO config INVALID_CONFIG /
INVALID_CRYPTO_KELLY_DAMPENER at zero size, cluster dict untouched,
before Step 4's domain_mult read. Gate 2.7 is CRYPTO-scoped (the
only path that reads the field), so non-crypto books are
bit-identical for EVERY dampener value.

Run:  python3 test_alpha_kelly_crypto_dampener_range.py
No pytest dependency: test_* functions with assert + __main__ runner.
"""
import contextlib
import io
import math
import os
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
EQUITY = 10000.0

# The four prereg range fixtures first, then the wider invalid set
# (SF-BL-10878 type failures share the same (0, 1] predicate).
PREREG_RANGE = [2.0, -1.0, 0.0, float("nan")]
INVALID_DAMPENER = PREREG_RANGE + [
    1.5, -0.5, 0, float("inf"), float("-inf"), 10 ** 400,
    True, False, "0.5", None,
]
VALID_DAMPENER = [0.50, 1.0, 1, 0.25, 0.01, 0.999999, 0.75]
MISLABELS = ("CLUSTER_CAP_REACHED", "ORDER_BOOK_DEPTH_EXHAUSTED")


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(domain=ak.MarketDomain.CRYPTO):
    return ak.OrderBookSnapshot(
        market_id="FIXTURE-10821", title="dampener range fixture",
        domain=domain, factor_cluster="C10821",
        est_days_to_resolution=10,
        yes_asks=[ak.OrderBookLevel(price=0.60, size_shares=1000000)],
        yes_bids=[ak.OrderBookLevel(price=0.58, size_shares=1000000)],
        fee_rate=0.01, on_chain_verified=True)


def _mutated(**kw):
    cfg = ak.FleetRiskConfig()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _eval(cfg, p=0.70, book=None, used=None):
    return ak.evaluate_and_size_opportunity(
        book if book is not None else _book(), p, 0.0, EQUITY,
        used if used is not None else {}, config=cfg)


def _assert_invalid_dampener_skip(r, bad):
    assert r.action == ak.TradeAction.SKIP, r
    assert r.invested_capital == 0.0 and r.shares_acquired == 0.0, r
    assert r.exec_price == 0.0 and r.liquidation_bid_price == 0.0, r
    assert r.kelly_fraction_raw == 0.0 and r.allocated_fraction == 0.0, r
    assert r.kelly_fraction_sized == 0.0, r
    assert r.skip_reason is not None, (bad, r)
    assert r.skip_reason.startswith("INVALID_CONFIG"), (bad, r.skip_reason)
    assert "INVALID_CRYPTO_KELLY_DAMPENER" in r.skip_reason, (bad, r.skip_reason)
    assert "crypto_kelly_dampener" in r.skip_reason, (bad, r.skip_reason)
    assert "SF-BL-10821" in r.skip_reason, (bad, r.skip_reason)
    for mislabel in MISLABELS:
        assert mislabel not in r.skip_reason, (bad, r.skip_reason)


def _tuple(r, used):
    return (r.action.value, r.invested_capital, r.shares_acquired,
            r.exec_price, r.kelly_fraction_raw, r.kelly_fraction_sized,
            r.allocated_fraction, r.liquidation_bid_price, r.skip_reason,
            dict(used))


# ---- 1. Construction: named ValueError for out-of-range, valid accepted ---

def test_construction_rejects_prereg_range_fixtures():
    for bad in PREREG_RANGE:
        try:
            ak.FleetRiskConfig(crypto_kelly_dampener=bad)
        except ValueError as e:
            assert "crypto_kelly_dampener" in str(e), (bad, e)
            assert "SF-BL-10821" in str(e), (bad, e)
            assert "(0, 1]" in str(e), (bad, e)
        else:
            raise AssertionError(f"no ValueError for dampener {bad!r}")


def test_construction_rejects_full_invalid_set():
    for bad in INVALID_DAMPENER:
        try:
            ak.FleetRiskConfig(crypto_kelly_dampener=bad)
        except ValueError as e:
            assert "crypto_kelly_dampener" in str(e), (bad, e)
        else:
            raise AssertionError(f"no ValueError for dampener {bad!r}")


def test_construction_accepts_valid_and_default():
    for good in VALID_DAMPENER:
        assert ak.FleetRiskConfig(
            crypto_kelly_dampener=good).crypto_kelly_dampener == good
    assert ak.FleetRiskConfig().crypto_kelly_dampener == 0.50


def test_predicate_domain_boundaries():
    assert ak._is_valid_crypto_kelly_dampener(0.50) is True
    assert ak._is_valid_crypto_kelly_dampener(1.0) is True   # inclusive
    assert ak._is_valid_crypto_kelly_dampener(1) is True
    assert ak._is_valid_crypto_kelly_dampener(0.0) is False  # exclusive
    assert ak._is_valid_crypto_kelly_dampener(2.0) is False
    assert ak._is_valid_crypto_kelly_dampener(-1.0) is False
    assert ak._is_valid_crypto_kelly_dampener(float("nan")) is False
    assert ak._is_valid_crypto_kelly_dampener(float("inf")) is False
    # Just outside / just inside the boundaries.
    assert ak._is_valid_crypto_kelly_dampener(1.0000001) is False
    assert ak._is_valid_crypto_kelly_dampener(1e-12) is True
    assert math.isfinite(1e-12)


# ---- 2. Gate 2.7: mutated out-of-range dampeners SKIP before sizing -------

def test_prereg_range_fixtures_skip_invalid_dampener():
    for bad in PREREG_RANGE:
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


def test_headline_2_no_longer_sizes_at_cluster_cap():
    # Pre-fix 2.0 sized BUY_YES at the full $1,500 cluster cap.
    r = _eval(_mutated(crypto_kelly_dampener=2.0))
    _assert_invalid_dampener_skip(r, 2.0)
    assert r.invested_capital != 1500.0


def test_negative_and_nan_no_longer_mislabel():
    # Pre-fix: -1.0 -> CLUSTER_CAP_REACHED, NaN -> ORDER_BOOK_DEPTH_EXHAUSTED.
    for bad in (-1.0, float("nan")):
        r = _eval(_mutated(crypto_kelly_dampener=bad))
        _assert_invalid_dampener_skip(r, bad)


def test_invalid_dampener_skips_on_buy_no_side_too():
    for bad in PREREG_RANGE:
        used = {}
        r = _eval(_mutated(crypto_kelly_dampener=bad), p=0.30, used=used)
        _assert_invalid_dampener_skip(r, bad)
        assert used == {}, (bad, used)


# ---- 3. Mandated / in-domain values: exact title figures ------------------

def test_mandated_dampener_sizes_title_figure():
    used = {}
    r = _eval(ak.FleetRiskConfig(crypto_kelly_dampener=0.50), used=used)
    assert r.action == ak.TradeAction.BUY_YES, r
    assert r.invested_capital == 417.5126903553298, r.invested_capital
    assert r.shares_acquired == 688.9648355698512, r.shares_acquired
    assert r.kelly_fraction_sized == 0.04175126903553298, r
    assert r.allocated_fraction == 0.04175126903553298, r
    assert r.skip_reason is None, r
    assert used == {"C10821": 0.04175126903553298}, used


def test_dampener_1_is_indomain_and_exactly_doubles_mandated():
    mandated = _eval(ak.FleetRiskConfig(crypto_kelly_dampener=0.50))
    for one in (1.0, 1):
        r = _eval(ak.FleetRiskConfig(crypto_kelly_dampener=one))
        assert r.action == ak.TradeAction.BUY_YES, r
        assert r.invested_capital == 835.0253807106596, r.invested_capital
        assert r.invested_capital == 2.0 * mandated.invested_capital, r


def test_valid_dampener_quarter():
    r = _eval(ak.FleetRiskConfig(crypto_kelly_dampener=0.25))
    assert r.action == ak.TradeAction.BUY_YES, r
    assert r.invested_capital == 208.7563451776649, r.invested_capital


def test_default_config_path_unchanged():
    used = {}
    r = ak.evaluate_and_size_opportunity(_book(), 0.70, 0.0, EQUITY, used)
    assert r.action == ak.TradeAction.BUY_YES, r
    assert r.invested_capital == 417.5126903553298, r
    assert used == {"C10821": 0.04175126903553298}, used


# ---- 4. Non-crypto books: bit-identical for EVERY dampener value ----------

def test_noncrypto_bit_identical_all_dampeners():
    for domain in (ak.MarketDomain.PREDICTION_MARKET,
                   ak.MarketDomain.EQUITIES_MACRO):
        used0 = {}
        base = _tuple(_eval(ak.FleetRiskConfig(), book=_book(domain),
                            used=used0), used0)
        assert base[0] == "BUY_YES" and base[1] == 835.0253807106596, base
        for damp in VALID_DAMPENER + INVALID_DAMPENER:
            used = {}
            r = _eval(_mutated(crypto_kelly_dampener=damp),
                      book=_book(domain), used=used)
            assert _tuple(r, used) == base, (domain, damp, r, base)


# ---- 5. No regression on the built-in benchmark ---------------------------

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
