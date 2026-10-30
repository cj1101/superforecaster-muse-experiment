"""
test_domain_validation.py — SF-BL-10795 verification tests.

Pre-registered done criterion: "a book whose domain is not a
MarketDomain member must be rejected at construction or SKIP with a
named invalid-domain reason before Gate 1, fixtured at 'CRYPTO',
'Crypto' and 'crypto ' (trailing space), with genuine enum fixtures
bit-identical; kill if a producer census shows domain is only ever
constructed from the enum."

Kill NOT met (PRODUCER_AUDIT.md): bare pass-through producers, no
production JSON/CSV deserialiser, mutable/unpicklable dataclass,
unenforced annotation.

Pre-fix reproduction (repro_prepatch.txt): domain='CRYPTO',
on_chain_verified=False -> BUY_YES $1,500.00 (full cluster cap;
Gate 1 off AND crypto dampener off) instead of SKIP
FAILED_CRYPTOBENCH_ONCHAIN_GATE; genuine verified CRYPTO trades
$1,042.93, genuine unverified CRYPTO SKIPs FAILED_...

Fix (alpha_kelly.py): _validate_market_domain in
OrderBookSnapshot.__post_init__ raises ValueError INVALID_DOMAIN on
any non-member (construction path), and Gate ED in
evaluate_and_size_opportunity SKIPs a mutated/unpickled non-member
with INVALID_DOMAIN before Gate 1 (gate path), zero size, cluster
untouched, unconditionally. isinstance, never coerced.

Run: python3 test_domain_validation.py (exit 0 iff all pass)
"""
import copy
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

PREREG_BAD = ["CRYPTO", "Crypto", "crypto "]  # prereg fixtures
WIDER_BAD = [
    "crypto", "prediction_market", "PREDICTION_MARKET", "equities_macro",
    "Crypto ", "CRYPTO ", " crypto", "", "CRYPTOBENCH", None, 123, 0,
    b"crypto", ["crypto"], ak.TradeAction.BUY_YES,
]
ALL_BAD = PREREG_BAD + WIDER_BAD
GENUINE = [ak.MarketDomain.CRYPTO, ak.MarketDomain.PREDICTION_MARKET,
           ak.MarketDomain.EQUITIES_MACRO]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(domain=ak.MarketDomain.CRYPTO, verified=False, cluster="C10795"):
    return ak.OrderBookSnapshot(
        market_id="FIXTURE", title="prereg fixture", domain=domain,
        factor_cluster=cluster, est_days_to_resolution=10,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100000)],
        fee_rate=0.01, on_chain_verified=verified)


def _mutated(domain, verified=False):
    b = _book(domain=ak.MarketDomain.CRYPTO, verified=verified)
    b.domain = domain
    return b


def _eval(book, p=0.80, config=None):
    cluster = {}
    kw = dict(calibrated_prob_yes=p, inter_agent_dispersion=0.0,
              portfolio_equity=10000, cluster_exposure_used=cluster)
    if config is not None:
        kw["config"] = config
    return ak.evaluate_and_size_opportunity(book, **kw), cluster


def _zero_invalid_skip(r, cluster):
    return (r.action == ak.TradeAction.SKIP
            and r.skip_reason is not None
            and r.skip_reason.startswith("INVALID_DOMAIN")
            and "SF-BL-10795" in r.skip_reason
            and r.invested_capital == 0.0 and r.shares_acquired == 0.0
            and r.allocated_fraction == 0.0 and cluster == {})


# --- construction path ----------------------------------------------------

def test_constructor_rejects_prereg():
    ok = True
    for bad in PREREG_BAD:
        try:
            _book(domain=bad)
            print("  accepted", repr(bad)); ok = False
        except ValueError as e:
            ok &= "INVALID_DOMAIN" in str(e) and "SF-BL-10795" in str(e)
    check("constructor rejects prereg 'CRYPTO'/'Crypto'/'crypto ' with named INVALID_DOMAIN", ok)


def test_constructor_rejects_wider_space():
    ok = True
    for bad in WIDER_BAD:
        try:
            _book(domain=bad)
            print("  accepted", repr(bad)); ok = False
        except ValueError:
            continue
    check("constructor rejects wider non-member space ('crypto', None, 123, bytes, TradeAction, ...)", ok)


def test_constructor_rejection_unconditional_on_verified_and_position():
    # domain validity does not depend on the flag or on require config
    ok = True
    for bad in PREREG_BAD:
        for verified in (True, False):
            try:
                _book(domain=bad, verified=verified); ok = False
            except ValueError:
                continue
    check("constructor rejection unconditional (verified True/False)", ok)


def test_constructor_accepts_genuine_members():
    ok = all(_book(domain=d).domain is d for d in GENUINE)
    check("constructor accepts all 3 genuine MarketDomain members (identity preserved)", ok)


# --- gate path (mutated / bypassed books) ----------------------------------

def test_mutated_prereg_skips_invalid_domain():
    ok = True
    for bad in PREREG_BAD:
        r, cl = _eval(_mutated(bad, verified=False))
        ok &= _zero_invalid_skip(r, cl)
    check("mutated prereg domains SKIP INVALID_DOMAIN, zero size, cluster untouched", ok)


def test_mutated_wider_space_skips():
    ok = True
    for bad in WIDER_BAD:
        r, cl = _eval(_mutated(bad, verified=False))
        ok &= _zero_invalid_skip(r, cl)
    check("mutated wider non-member space SKIPs INVALID_DOMAIN", ok)


def test_object_setattr_and_dict_bypass_skip():
    ok = True
    for mode in ("object.__setattr__", "__dict__"):
        b = _book(domain=ak.MarketDomain.CRYPTO, verified=False)
        if mode == "object.__setattr__":
            object.__setattr__(b, "domain", "CRYPTO")
        else:
            b.__dict__["domain"] = "Crypto"
        r, cl = _eval(b)
        ok &= _zero_invalid_skip(r, cl)
    check("object.__setattr__ / __dict__ bypass books SKIP INVALID_DOMAIN", ok)


def test_unpickled_book_skips():
    b = _book(domain=ak.MarketDomain.CRYPTO, verified=False)
    b.__dict__["domain"] = "crypto "
    raw = pickle.dumps(b)
    b2 = pickle.loads(raw)
    r, cl = _eval(b2)
    check("unpickled 'crypto ' book SKIPs INVALID_DOMAIN", _zero_invalid_skip(r, cl))


def test_invalid_domain_precedes_gate1():
    # unverified mutated CRYPTO-string book must NOT report the genuine
    # gate reason and must NOT trade: INVALID_DOMAIN wins over Gate 1.
    r, _ = _eval(_mutated("CRYPTO", verified=False))
    check("INVALID_DOMAIN precedes Gate 1 (not FAILED_CRYPTOBENCH_ONCHAIN_GATE, not BUY_YES)",
          r.skip_reason.startswith("INVALID_DOMAIN")
          and r.action == ak.TradeAction.SKIP)


def test_invalid_domain_unconditional_gate_disabled_and_verified():
    cfg = ak.FleetRiskConfig(require_on_chain_for_crypto=False)
    r1, cl1 = _eval(_mutated("CRYPTO", verified=False), config=cfg)
    r2, cl2 = _eval(_mutated("Crypto", verified=True))
    check("INVALID_DOMAIN unconditional: gate disabled and verified=True mutated books still SKIP",
          _zero_invalid_skip(r1, cl1) and _zero_invalid_skip(r2, cl2))


def test_invalid_domain_carried_verbatim_on_skip_record():
    r, _ = _eval(_mutated("crypto ", verified=False))
    check("SKIP record carries offending domain verbatim for audit",
          r.domain == "crypto " and type(r.domain) is str)


# --- genuine enum fixtures bit-identical -----------------------------------

def test_genuine_crypto_gate_bit_identical():
    r, cl = _eval(_book(domain=ak.MarketDomain.CRYPTO, verified=False))
    ok1 = (r.action == ak.TradeAction.SKIP
           and r.skip_reason == "FAILED_CRYPTOBENCH_ONCHAIN_GATE" and cl == {})
    r2, _ = _eval(_book(domain=ak.MarketDomain.CRYPTO, verified=True))
    ok2 = (r2.action == ak.TradeAction.BUY_YES
           and abs(r2.invested_capital - 1042.93) < 0.01)
    check("genuine CRYPTO: False -> FAILED_CRYPTOBENCH_ONCHAIN_GATE, True -> BUY_YES $1,042.93",
          ok1 and ok2, f"(true invested={r2.invested_capital:.2f})")


def test_genuine_non_crypto_bit_identical():
    ok = True
    for d in (ak.MarketDomain.PREDICTION_MARKET, ak.MarketDomain.EQUITIES_MACRO):
        r, _ = _eval(_book(domain=d, verified=False))
        ok &= r.action == ak.TradeAction.BUY_YES and r.invested_capital == 1500.0
    check("genuine non-CRYPTO unverified books still trade BUY_YES $1,500.00 (Gate 1 is CRYPTO-only)", ok)


def test_genuine_gate_disabled_trades():
    cfg = ak.FleetRiskConfig(require_on_chain_for_crypto=False)
    r, _ = _eval(_book(domain=ak.MarketDomain.CRYPTO, verified=False), config=cfg)
    check("genuine CRYPTO + gate disabled still trades (dampened $1,042.93)",
          r.action == ak.TradeAction.BUY_YES and abs(r.invested_capital - 1042.93) < 0.01)


def test_genuine_pickle_roundtrip_identical():
    b = _book(domain=ak.MarketDomain.CRYPTO, verified=True)
    r1, _ = _eval(b)
    r2, _ = _eval(pickle.loads(pickle.dumps(b)))
    check("genuine book pickle round-trip trades identically",
          r1.action == r2.action and r1.invested_capital == r2.invested_capital)


def test_genuine_deepcopy_and_side_symmetry():
    b = _book(domain=ak.MarketDomain.CRYPTO, verified=True)
    r1, _ = _eval(copy.deepcopy(b))
    r_no, _ = _eval(_book(domain=ak.MarketDomain.CRYPTO, verified=True), p=0.20)
    check("genuine deepcopy trades; genuine CRYPTO p=0.20 takes BUY_NO side",
          r1.action == ak.TradeAction.BUY_YES and r_no.action == ak.TradeAction.BUY_NO)


def test_precedence_other_gates_unchanged_for_genuine_books():
    # A genuine-domain book with a mutated invalid fee still reports
    # the fee gate, not INVALID_DOMAIN.
    b = _book(domain=ak.MarketDomain.CRYPTO, verified=True)
    b.fee_rate = -0.20
    r, cl = _eval(b)
    check("genuine domain + invalid fee still SKIPs INVALID_FEE_RATE",
          r.skip_reason is not None and r.skip_reason.startswith("INVALID_FEE_RATE") and cl == {})


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
    sys.exit(0 if all(PASS) and PASS else 1)
