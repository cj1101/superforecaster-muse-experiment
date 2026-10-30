"""
test_require_on_chain_config_validation.py — SF-BL-10845 verification tests.

Pre-registered done criterion: "a non-bool require_on_chain_for_crypto
is rejected with a named invalid-config SKIP/ValueError at evaluate
entry and strict True/False behaviour is byte-identical; kill if a
caller census shows the flag is only ever set by validated in-repo
literals."

Pre-fix reproduction (verified live 2026-10-22, before the patch):
  prereg fixture (CRYPTO, on_chain_verified=False, ask 0.50 / bid 0.48,
  fee 0.01, p=0.80, equity $100,000, deep size):
    require=True  (bool, also the default) -> SKIP FAILED_CRYPTOBENCH_ONCHAIN_GATE
    require=False (bool)                   -> BUY_YES invested $10,429.29 (intended)
    require='' / 0 / 0.0 / None / []       -> BUY_YES $10,429.29 (BUG: Gate 1 silently off)
    require='False' / 'True' / 1 / 0.5 ... -> SKIP FAILED_CRYPTOBENCH_ONCHAIN_GATE
      (BUG: gate armed by truthiness; 'False' reads as off to a producer)
    book flag mutated to 'False' + require='' -> BUY_YES $10,429.29
      (BUG: Gate 0.9's book-flag check is conditioned on this flag)
  and FleetRiskConfig had NO __post_init__: every non-bool above was
  also accepted verbatim at construction.

Kill condition is NOT met (see CALLER_CENSUS.md): the in-repo setters
are bare unenforced literals (default True; one test literal False),
there is no schema-validation layer anywhere in the repo, the dataclass
is mutable, unpickling bypasses construction, and evaluate's default
config is a shared mutable instance — so the fix, not the kill, applies.

Fix (alpha_kelly.py): _validate_require_on_chain_for_crypto in the new
FleetRiskConfig.__post_init__ raises ValueError on any non-bool
(construction path), and Gate E0 at evaluate_and_size_opportunity ENTRY
SKIPs a mutated/unpickled non-bool config with INVALID_CONFIG (gate
path), zero size, cluster untouched, before Gate 0.9/Gate 1 can read
the flag. The entry check is unconditional (the config object itself is
malformed, whatever the book); strict bools pass untouched.

Run:  python3 test_require_on_chain_config_validation.py  (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_onchain_verified.py.
"""
import os
import pickle
import sys

# Works from both homes of this file: the task work dir and the repo
# copy at tests/test_alpha_kelly_require_on_chain_config.py.
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

# Every non-bool value that must be rejected: the TASK.md set first
# ('' and 0 silently off; 'False' silently on), then the wider space.
NON_BOOLS = [
    "", 0, "False",                          # TASK.md headline set
    "True", "true", "no", "0", "yes",        # other strings
    1, 0.0, 1.0, 0.5, -1,                    # ints / floats (1/0 are NOT bools)
    None,                                    # missing / unset
    [], ["x"], {}, {"v": 1},                 # containers
]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(verified=False, domain=ak.MarketDomain.CRYPTO, cluster="C10845"):
    """The prereg fixture book: ask 0.50 / bid 0.48, fee 0.01, deep size,
    on_chain_verified=False (strict bool) unless overridden."""
    return ak.OrderBookSnapshot(
        market_id="FIXTURE", title="prereg fixture", domain=domain,
        factor_cluster=cluster, est_days_to_resolution=10,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100000)],
        fee_rate=0.01, on_chain_verified=verified)


def _cfg(require):
    """Config whose flag bypasses __post_init__ (post-construction mutation)."""
    c = ak.FleetRiskConfig()
    c.require_on_chain_for_crypto = require
    return c


def _eval(book, p=0.80, dispersion=0.0, equity=100000, config=None):
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


def _assert_invalid_config(r, cluster):
    _assert_zero_skip(r, cluster)
    assert r.skip_reason.startswith("INVALID_CONFIG"), r.skip_reason
    assert "invalid config" in r.skip_reason, r.skip_reason
    assert "require_on_chain_for_crypto" in r.skip_reason, r.skip_reason
    assert "SF-BL-10845" in r.skip_reason, r.skip_reason
    # Never the genuine gate reason: that is reserved for bool True + bool False.
    assert not r.skip_reason.startswith("FAILED_CRYPTOBENCH"), r.skip_reason


# ---- 1. Constructor rejects every non-bool ---------------------------------

def test_constructor_rejects_task_set():
    for bad in ("", 0, "False"):
        try:
            ak.FleetRiskConfig(require_on_chain_for_crypto=bad)
        except ValueError as e:
            assert "require_on_chain_for_crypto" in str(e), e
            assert "SF-BL-10845" in str(e), e
            continue
        raise AssertionError(f"require={bad!r} accepted at construction")


def test_constructor_rejects_all_non_bools():
    for bad in NON_BOOLS:
        try:
            ak.FleetRiskConfig(require_on_chain_for_crypto=bad)
        except ValueError:
            continue
        raise AssertionError(f"require={bad!r} accepted at construction")


def test_constructor_accepts_strict_bools_and_default():
    assert ak.FleetRiskConfig(require_on_chain_for_crypto=True).require_on_chain_for_crypto is True
    assert ak.FleetRiskConfig(require_on_chain_for_crypto=False).require_on_chain_for_crypto is False
    assert ak.FleetRiskConfig().require_on_chain_for_crypto is True  # default unchanged


def test_constructor_other_fields_unaffected():
    # Only this field is validated: existing constructions keep working.
    c = ak.FleetRiskConfig(max_round_trip_cost=None)
    assert c.max_round_trip_cost is None and c.require_on_chain_for_crypto is True
    c = ak.FleetRiskConfig(kelly_multiplier=0.25, min_net_edge=0.03)
    assert c.kelly_multiplier == 0.25 and c.min_net_edge == 0.03


def test_validator_helper_semantics():
    ak._validate_require_on_chain_for_crypto(True)
    ak._validate_require_on_chain_for_crypto(False)
    for bad in NON_BOOLS:
        try:
            ak._validate_require_on_chain_for_crypto(bad)
        except ValueError:
            continue
        raise AssertionError(f"validator accepted {bad!r}")
    # bool subclasses int, not the reverse: 1/0 must not pass as bools.
    assert not ak._is_strict_bool(1) and not ak._is_strict_bool(0)


# ---- 2. Entry gate: mutated / unpickled non-bools SKIP INVALID_CONFIG -------

def test_mutated_task_set_skips_invalid_config():
    for bad in ("", 0, "False"):
        r, cluster = _eval(_book(), config=_cfg(bad))
        _assert_invalid_config(r, cluster)


def test_mutated_all_non_bools_skip_invalid_config():
    for bad in NON_BOOLS:
        r, cluster = _eval(_book(), config=_cfg(bad))
        _assert_invalid_config(r, cluster)


def test_unpickled_config_skips_invalid_config():
    c = _cfg("False")
    restored = pickle.loads(pickle.dumps(c))  # unpickling bypasses __post_init__
    assert restored.require_on_chain_for_crypto == "False"
    r, cluster = _eval(_book(), config=restored)
    _assert_invalid_config(r, cluster)


def test_none_config_skips_invalid_config():
    cluster = {}
    r = ak.evaluate_and_size_opportunity(_book(), 0.80, 0.0, 100000, cluster,
                                         config=None)
    _assert_invalid_config(r, cluster)


def test_falsy_non_bool_no_longer_trades_10429():
    # The headline silent-off bug: '' and 0 traded BUY_YES $10,429.29 pre-fix.
    for bad in ("", 0, 0.0, None, []):
        r, _ = _eval(_book(), config=_cfg(bad))
        assert r.action == ak.TradeAction.SKIP and r.invested_capital == 0.0, (bad, r)


def test_invalid_config_on_verified_book_and_non_crypto():
    # The entry check is unconditional: a malformed config is rejected
    # even where Gate 1 would never fire (verified book / other domains).
    r, cluster = _eval(_book(verified=True), config=_cfg("False"))
    _assert_invalid_config(r, cluster)
    for domain in (ak.MarketDomain.PREDICTION_MARKET, ak.MarketDomain.EQUITIES_MACRO):
        r, cluster = _eval(_book(verified=False, domain=domain), config=_cfg(0))
        _assert_invalid_config(r, cluster)


# ---- 3. Bool fixtures bit-identical ------------------------------------------

def test_bool_true_skips_failed_gate_identically():
    for cfg in (ak.FleetRiskConfig(), ak.FleetRiskConfig(require_on_chain_for_crypto=True)):
        r, cluster = _eval(_book(), config=cfg)
        _assert_zero_skip(r, cluster)
        assert r.skip_reason == "FAILED_CRYPTOBENCH_ONCHAIN_GATE", r.skip_reason


def test_bool_false_trades_10429_identically():
    r, cluster = _eval(_book(), config=ak.FleetRiskConfig(require_on_chain_for_crypto=False))
    assert r.action == ak.TradeAction.BUY_YES, r
    assert abs(r.invested_capital - 10429.29292929293) < 1e-9, r.invested_capital
    assert abs(r.allocated_fraction - 0.10429292929292931) < 1e-12, r
    assert list(cluster) == ["C10845"], cluster
    assert abs(cluster["C10845"] - 0.10429292929292931) < 1e-12, cluster
    assert r.skip_reason is None


def test_bool_fixtures_at_10k_match_10793_amounts():
    # Same book at the SF-BL-10793 equity: $1,042.93 scale, gate on/off.
    r, _ = _eval(_book(), equity=10000,
                 config=ak.FleetRiskConfig(require_on_chain_for_crypto=False))
    assert r.action == ak.TradeAction.BUY_YES, r
    assert abs(r.invested_capital - 1042.929292929293) < 1e-9, r.invested_capital
    r, cluster = _eval(_book(), equity=10000,
                       config=ak.FleetRiskConfig(require_on_chain_for_crypto=True))
    _assert_zero_skip(r, cluster)
    assert r.skip_reason == "FAILED_CRYPTOBENCH_ONCHAIN_GATE", r.skip_reason


def test_default_config_path_unchanged():
    # No config argument: the shared default instance (require=True).
    r, cluster = _eval(_book())
    _assert_zero_skip(r, cluster)
    assert r.skip_reason == "FAILED_CRYPTOBENCH_ONCHAIN_GATE", r.skip_reason


def test_bool_configs_non_crypto_unchanged():
    # Gate 1 is CRYPTO-only: bool False book flag on other domains trades
    # under BOTH bool switch values, exactly as pre-fix.
    for domain in (ak.MarketDomain.PREDICTION_MARKET, ak.MarketDomain.EQUITIES_MACRO):
        for req in (True, False):
            r, _ = _eval(_book(verified=False, domain=domain),
                         config=ak.FleetRiskConfig(require_on_chain_for_crypto=req))
            assert r.action == ak.TradeAction.BUY_YES, (domain, req, r)


def test_verified_book_trades_under_both_bool_switches():
    for req in (True, False):
        r, _ = _eval(_book(verified=True),
                     config=ak.FleetRiskConfig(require_on_chain_for_crypto=req))
        assert r.action == ak.TradeAction.BUY_YES, (req, r)
        assert abs(r.invested_capital - 10429.29292929293) < 1e-9, r.invested_capital


# ---- 4. Interaction with Gate 0.9 (SF-BL-10793) and precedence ---------------

def test_gate09_restored_when_switch_is_bool_true():
    # Book flag mutated to a non-bool, switch a strict True: Gate 0.9
    # must fire again (pre-fix a falsy switch disabled it entirely).
    b = _book(verified=True)
    b.on_chain_verified = "False"
    r, cluster = _eval(b, config=ak.FleetRiskConfig(require_on_chain_for_crypto=True))
    _assert_zero_skip(r, cluster)
    assert r.skip_reason.startswith("INVALID_ON_CHAIN_VERIFIED"), r.skip_reason


def test_invalid_config_precedes_gate09():
    # Both flags malformed: the config error is reported at entry,
    # before Gate 0.9 can be conditioned on the malformed switch.
    b = _book(verified=True)
    b.on_chain_verified = "False"
    r, cluster = _eval(b, config=_cfg(""))
    _assert_invalid_config(r, cluster)


def test_invalid_config_precedes_other_input_gates():
    # Entry precedence: config validity outranks posterior / fee errors.
    r, cluster = _eval(_book(), p=1.50, config=_cfg(0))
    _assert_invalid_config(r, cluster)
    b = _book()
    b.fee_rate = -0.20  # mutation bypass, as in SF-BL-9363 tests
    r, cluster = _eval(b, config=_cfg("False"))
    _assert_invalid_config(r, cluster)


def test_bool_false_switch_with_mutated_book_flag_still_trades():
    # Documented 10793 scope, unchanged: with the gate legitimately OFF
    # (strict bool False) the book flag is never read, so a mutated
    # non-bool book flag does not block the trade in evaluate
    # (construction still rejects it). Bool-switch behaviour identical.
    b = _book(verified=True)
    b.on_chain_verified = "False"
    r, _ = _eval(b, config=ak.FleetRiskConfig(require_on_chain_for_crypto=False))
    assert r.action == ak.TradeAction.BUY_YES, r


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
