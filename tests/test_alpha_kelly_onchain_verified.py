"""
test_onchain_verified_validation.py — SF-BL-10793 verification tests.

Pre-registered done criterion: "evaluate must SKIP with a named
invalid-verification reason (or raise) for every non-bool
on_chain_verified ('False', 'no', '0', 0.5, None) on a CRYPTO book, with
bool True/False fixtures bit-identical to today; kill if a caller census
shows every producer passes strict bools from a schema-validated source."

Pre-fix reproduction (verified live 2026-10-22, before the patch):
  prereg fixture (CRYPTO, ask 0.50 / bid 0.48, p=0.80, fee 0.01,
  equity $10,000, deep size):
    on_chain_verified=True  (bool)  -> BUY_YES invested $1,042.93
    on_chain_verified=False (bool)  -> SKIP FAILED_CRYPTOBENCH_ONCHAIN_GATE
    on_chain_verified='False'       -> BUY_YES invested $1,042.93  (BUG)
    on_chain_verified='no' / '0' / 'True' / 1 / 0.5 / ['x'] -> BUY_YES $1,042.93
    on_chain_verified=0 / '' / None / [] -> SKIP, but with the genuine
      FAILED_CRYPTOBENCH_ONCHAIN_GATE reason (falsy accident, not a
      type rejection)

Kill condition is NOT met (see CALLER_CENSUS.md): no producer sources
the flag from a schema-validated source — benchmark fixtures and test
helpers pass bare literals, no code builds an OrderBookSnapshot from
JSON/CSV at all, the dataclass is mutable (mutation/unpickle bypass
__post_init__, and repo tests mutate this field) and the `bool`
annotation is unenforced — so the fix, not the kill, applies.

Fix (alpha_kelly.py): _validate_on_chain_verified in
OrderBookSnapshot.__post_init__ raises ValueError on any non-bool
(construction path), and Gate 0.9 in evaluate_and_size_opportunity
SKIPs a mutated/unpickled non-bool on a gated CRYPTO book with
INVALID_ON_CHAIN_VERIFIED (gate path), zero size, cluster untouched.

Run:  python3 test_onchain_verified_validation.py  (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_fee_rate.py.
"""
import os
import pickle
import sys

# Works from both homes of this file: the task work dir and the repo
# copy at tests/test_alpha_kelly_onchain_verified.py.
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

# Every non-bool value that must be rejected: the prereg set first,
# then the wider truthy/falsy non-bool space.
NON_BOOLS = [
    "False", "no", "0", 0.5, None,          # prereg set
    "True", "true", "yes", "",              # other strings
    0, 1, 0.0, 1.0, -1,                     # ints / floats (1/0 are NOT bools)
    [], ["x"], {}, {"v": 1},                # containers
]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(verified=True, domain=ak.MarketDomain.CRYPTO, cluster="C10793"):
    """The prereg fixture book: ask 0.50 / bid 0.48, fee 0.01, deep size."""
    return ak.OrderBookSnapshot(
        market_id="FIXTURE", title="prereg fixture", domain=domain,
        factor_cluster=cluster, est_days_to_resolution=10,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100000)],
        fee_rate=0.01, on_chain_verified=verified)


def _mutated_book(verified, domain=ak.MarketDomain.CRYPTO):
    """Book whose flag bypasses __post_init__ (post-construction mutation)."""
    b = _book(verified=True, domain=domain)
    b.on_chain_verified = verified
    return b


def _eval(book, p=0.80, dispersion=0.0, equity=10000, config=None):
    cluster = {}
    if config is None:
        r = ak.evaluate_and_size_opportunity(book, p, dispersion, equity, cluster)
    else:
        r = ak.evaluate_and_size_opportunity(
            book, p, dispersion, equity, cluster, config=config)
    return r, cluster


def _assert_zero_skip(r, cluster):
    assert r.action == ak.TradeAction.SKIP, r
    assert r.invested_capital == 0.0 and r.shares_acquired == 0.0, r
    assert r.exec_price == 0.0 and r.allocated_fraction == 0.0, r  # EC-09
    assert r.kelly_fraction_raw == 0.0, r
    assert cluster == {}, cluster


# ---- 1. Constructor rejects every non-bool ---------------------------------

def test_constructor_rejects_prereg_set():
    for bad in ("False", "no", "0", 0.5, None):
        try:
            _book(verified=bad)
        except ValueError:
            continue
        raise AssertionError(f"on_chain_verified={bad!r} accepted at construction")


def test_constructor_rejects_all_non_bools_all_domains():
    for bad in NON_BOOLS:
        for domain in (ak.MarketDomain.CRYPTO, ak.MarketDomain.PREDICTION_MARKET,
                       ak.MarketDomain.EQUITIES_MACRO):
            try:
                _book(verified=bad, domain=domain)
            except ValueError:
                continue
            raise AssertionError(
                f"on_chain_verified={bad!r} accepted for domain={domain}")


def test_constructor_accepts_strict_bools_and_default():
    for good in (True, False):
        assert _book(verified=good).on_chain_verified is good
    b = ak.OrderBookSnapshot(
        market_id="T", title="t", domain=ak.MarketDomain.CRYPTO,
        factor_cluster="C", est_days_to_resolution=5,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=1000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=1000)])
    assert b.on_chain_verified is True  # documented default unchanged


def test_helper_strict_bool_semantics():
    assert ak._is_strict_bool(True) and ak._is_strict_bool(False)
    for bad in NON_BOOLS:
        assert not ak._is_strict_bool(bad), bad
    # bool subclasses int, not the reverse: 1/0 must not pass as bools.
    assert not ak._is_strict_bool(1) and not ak._is_strict_bool(0)


# ---- 2. Gate path: mutated / unpickled non-bools SKIP INVALID ---------------

def test_mutated_prereg_set_skips_invalid():
    for bad in ("False", "no", "0", 0.5, None):
        r, cluster = _eval(_mutated_book(bad))
        _assert_zero_skip(r, cluster)
        assert r.skip_reason.startswith("INVALID_ON_CHAIN_VERIFIED"), (bad, r.skip_reason)
        assert "invalid verification" in r.skip_reason, r.skip_reason


def test_mutated_all_non_bools_skip_invalid():
    for bad in NON_BOOLS:
        r, cluster = _eval(_mutated_book(bad))
        _assert_zero_skip(r, cluster)
        assert r.skip_reason.startswith("INVALID_ON_CHAIN_VERIFIED"), (bad, r.skip_reason)
        # Never the genuine-unverified reason: that is reserved for bool False.
        assert "FAILED_CRYPTOBENCH_ONCHAIN_GATE" not in r.skip_reason.split("(")[0], bad


def test_unpickled_string_false_skips_invalid():
    b = _mutated_book("False")
    restored = pickle.loads(pickle.dumps(b))  # unpickling bypasses __post_init__
    assert restored.on_chain_verified == "False"
    r, cluster = _eval(restored)
    _assert_zero_skip(r, cluster)
    assert r.skip_reason.startswith("INVALID_ON_CHAIN_VERIFIED"), r.skip_reason


def test_truthy_string_no_longer_trades_1042():
    # The headline bug: 'False' traded BUY_YES $1,042.93 pre-fix.
    r, _ = _eval(_mutated_book("False"))
    assert r.action == ak.TradeAction.SKIP and r.invested_capital == 0.0, r


# ---- 3. Bool fixtures bit-identical ------------------------------------------

def test_bool_true_trades_identically():
    r, cluster = _eval(_book(verified=True))
    assert r.action == ak.TradeAction.BUY_YES, r
    assert abs(r.invested_capital - 1042.929292929293) < 1e-9, r.invested_capital
    assert abs(r.allocated_fraction - 0.10429292929292931) < 1e-12, r
    assert cluster == {"C10793": 0.10429292929292931}, cluster
    assert r.skip_reason is None


def test_bool_false_skips_failed_gate_identically():
    r, cluster = _eval(_book(verified=False))
    _assert_zero_skip(r, cluster)
    assert r.skip_reason == "FAILED_CRYPTOBENCH_ONCHAIN_GATE", r.skip_reason


def test_bool_false_non_crypto_still_trades():
    # Gate 1 is CRYPTO-only: a bool False on other domains never gated.
    for domain in (ak.MarketDomain.PREDICTION_MARKET, ak.MarketDomain.EQUITIES_MACRO):
        r, _ = _eval(_book(verified=False, domain=domain))
        assert r.action == ak.TradeAction.BUY_YES, (domain, r)


def test_gate_disabled_bool_false_still_trades():
    cfg = ak.FleetRiskConfig(require_on_chain_for_crypto=False)
    r, _ = _eval(_book(verified=False), config=cfg)
    assert r.action == ak.TradeAction.BUY_YES, r


# ---- 4. Precedence vs the earlier input-validity gates -----------------------

def test_invalid_fee_precedes_invalid_verification():
    b = _mutated_book("False")
    b.fee_rate = -0.20  # mutation bypass, as in SF-BL-9363 tests
    r, cluster = _eval(b)
    _assert_zero_skip(r, cluster)
    assert r.skip_reason.startswith("INVALID_FEE_RATE"), r.skip_reason


def test_invalid_posterior_precedes_invalid_verification():
    r, cluster = _eval(_mutated_book("False"), p=1.50)
    _assert_zero_skip(r, cluster)
    assert r.skip_reason.startswith("INVALID_POSTERIOR"), r.skip_reason


def test_invalid_dispersion_precedes_invalid_verification():
    r, cluster = _eval(_mutated_book(None), dispersion=float("nan"))
    _assert_zero_skip(r, cluster)
    assert r.skip_reason.startswith("INVALID_INTER_AGENT_DISPERSION"), r.skip_reason


def test_invalid_verification_precedes_gate2_and_sizing():
    # Valid inputs otherwise, dispersion over Gate 2 threshold: the
    # invalid flag must be reported, not the dispersion business gate.
    r, cluster = _eval(_mutated_book("no"), dispersion=0.50)
    _assert_zero_skip(r, cluster)
    assert r.skip_reason.startswith("INVALID_ON_CHAIN_VERIFIED"), r.skip_reason


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for fn in ALL_TESTS:
        try:
            fn()
            check(fn.__name__, True)
        except AssertionError as e:
            check(fn.__name__, False, str(e))
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
