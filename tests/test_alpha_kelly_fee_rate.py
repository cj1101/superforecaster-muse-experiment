"""
test_fee_rate_validation.py — SF-BL-9363 verification tests.

Pre-registered done criterion: "OrderBookSnapshot or the gate rejects
fee_rate outside [0, 0.10] (pre-registered bound) and the negative-fee
fixture SKIPs identically to the 0.01 case; kill if every fee_rate
producer is proven to source from a validated venue schedule with a
tested range invariant."

Pre-fix reproduction (verified live 2026-10-21, before the patch):
  fixture book (best ask 0.62, bid 0.60, p=0.65, equity $10,000):
    fee_rate= 0.01 -> SKIP  INSUFFICIENT_NET_EDGE (max_edge=0.024 < 0.050)
    fee_rate=-0.20 -> BUY_YES invested $1,069.44 at q_exec 0.496
    sweep_order_book(fee=-0.20, budget $2,000) -> (2000.0, 4032.26, 0.496)

Kill condition is NOT met (see PRODUCER_AUDIT.md): producers pass bare
literals (benchmark fixtures 0.005-0.015, r63 scan 0.0 / 0.0175, test
helpers arbitrary) — none sources from a validated venue schedule — so
the fix, not the kill, applies.

Run:  python3 test_fee_rate_validation.py   (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_cli_edge.py.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "superforecaster"))

import alpha_kelly as ak  # noqa: E402

PASS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(fee=0.01, ask=0.62, bid=0.60, ask_size=10000):
    return ak.OrderBookSnapshot(
        market_id="FIXTURE", title="prereg fixture", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C9363", est_days_to_resolution=5,
        yes_asks=[ak.OrderBookLevel(price=ask, size_shares=ask_size)],
        yes_bids=[ak.OrderBookLevel(price=bid, size_shares=10000)],
        fee_rate=fee)


def _mutated_book(fee):
    """Book whose fee_rate bypasses __post_init__ (post-construction mutation)."""
    b = _book(fee=0.01)
    b.fee_rate = fee
    return b


# ---- 1. Constructor rejects every out-of-bound fee --------------------------

def test_constructor_rejects_negative_and_outsized_fees():
    for bad in (-0.20, -0.001, -1.0, 0.100001, 0.11, 0.5, 1.0, 2.0):
        try:
            _book(fee=bad)
        except ValueError:
            continue
        raise AssertionError(f"fee_rate={bad} was accepted at construction")


def test_constructor_rejects_nonfinite_and_nonnumeric_fees():
    for bad in (float("nan"), float("inf"), float("-inf"), "0.01", None, True):
        try:
            _book(fee=bad)
        except (ValueError, TypeError):
            continue
        raise AssertionError(f"fee_rate={bad!r} was accepted at construction")


def test_constructor_accepts_boundaries_and_inrepo_fees():
    # Boundaries inclusive; every fee literal used in-repo must still build:
    # benchmark fixtures 0.005/0.01/0.015, r63 scan 0.0/0.0175, SF-BL-0169
    # venue templates up to ~0.07.
    for good in (0.0, 0.005, 0.01, 0.015, 0.0175, 0.04, 0.05, 0.07, 0.10):
        b = _book(fee=good)
        assert b.fee_rate == good


# ---- 2. Prereg fixture: 0.01 case unchanged, negative fee cannot trade -----

def test_fixture_fee_001_skips_as_before():
    r = ak.evaluate_and_size_opportunity(_book(fee=0.01), 0.65, 0.0, 10000, {})
    assert r.action == ak.TradeAction.SKIP, r
    assert r.skip_reason == "INSUFFICIENT_NET_EDGE (max_edge=0.024 < 0.050)", r.skip_reason
    assert r.invested_capital == 0.0 and r.shares_acquired == 0.0
    assert r.kelly_fraction_raw == 0.0 and r.allocated_fraction == 0.0  # EC-09


def test_negative_fee_fixture_never_buys():
    # Construction path: rejected outright (stronger than a SKIP).
    try:
        _book(fee=-0.20)
        raise AssertionError("negative fee constructed")
    except ValueError:
        pass
    # Gate path (mutation bypass): SKIPs, zero size, explicit reason —
    # never the pre-fix BUY_YES $1,069.44 at q_exec 0.496.
    r = ak.evaluate_and_size_opportunity(_mutated_book(-0.20), 0.65, 0.0, 10000, {})
    assert r.action == ak.TradeAction.SKIP, r
    assert r.skip_reason.startswith("INVALID_FEE_RATE"), r.skip_reason
    assert r.invested_capital == 0.0 and r.shares_acquired == 0.0 and r.exec_price == 0.0
    assert r.kelly_fraction_raw == 0.0 and r.allocated_fraction == 0.0


def test_gate_rejects_all_out_of_bound_mutations():
    for bad in (-0.20, 0.11, 1.0, float("nan"), float("inf")):
        r = ak.evaluate_and_size_opportunity(_mutated_book(bad), 0.65, 0.0, 10000, {})
        assert r.action == ak.TradeAction.SKIP and r.invested_capital == 0.0, (bad, r)


def test_gate0_precedes_other_gates():
    # Even a book that would fail Gate 1 (crypto unverified) reports the
    # fee gate first when its fee is invalid: fee validity is Gate 0.
    b = _mutated_book(-0.20)
    b.domain = ak.MarketDomain.CRYPTO
    b.on_chain_verified = False
    r = ak.evaluate_and_size_opportunity(b, 0.65, 0.0, 10000, {})
    assert r.skip_reason.startswith("INVALID_FEE_RATE"), r.skip_reason


# ---- 3. sweep_order_book -----------------------------------------------------

def test_sweep_rejects_mutated_invalid_fee():
    for bad in (-0.20, 0.5, float("nan")):
        try:
            ak.sweep_order_book(_mutated_book(bad), ak.TradeAction.BUY_YES, 2000.0, 0.65)
        except ValueError:
            continue
        raise AssertionError(f"sweep accepted fee_rate={bad}")


def test_sweep_valid_fees_unchanged():
    # fee 0.01: top rung eff 0.6262, edge 0.0238 < 0.05 stop -> no fill.
    assert ak.sweep_order_book(_book(fee=0.01), ak.TradeAction.BUY_YES, 2000.0, 0.65) == (0.0, 0.0, 0.0)
    # fee 0.0 on a deep cheap rung: fills at raw price exactly.
    inv, sh, q = ak.sweep_order_book(_book(fee=0.0, ask=0.50, bid=0.48),
                                     ak.TradeAction.BUY_YES, 500.0, 0.70)
    assert (inv, sh, q) == (500.0, 1000.0, 0.50), (inv, sh, q)
    # fee at the upper boundary 0.10: eff price = price * 1.10 exactly.
    inv, sh, q = ak.sweep_order_book(_book(fee=0.10, ask=0.50, bid=0.48),
                                     ak.TradeAction.BUY_YES, 550.0, 0.70)
    assert inv == 550.0 and abs(q - 0.55) < 1e-12 and abs(sh - 1000.0) < 1e-9, (inv, sh, q)


# ---- 4. No regression on valid-fee paths ------------------------------------

def test_trade_control_unchanged():
    # Same control as tests/test_cli_edge.py::test_ak_trade_unchanged.
    b = ak.OrderBookSnapshot(
        market_id="T", title="t", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C1", est_days_to_resolution=5,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=1000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=1000)], fee_rate=0.0)
    r = ak.evaluate_and_size_opportunity(b, 0.70, 0.0, 10000, {})
    assert r.action == ak.TradeAction.BUY_YES
    assert abs(r.kelly_fraction_raw - 0.40) < 1e-9, r
    assert r.allocated_fraction == 0.05 and r.invested_capital == 500.0, r


def test_default_fee_unchanged():
    b = ak.OrderBookSnapshot(
        market_id="T", title="t", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C1", est_days_to_resolution=5,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=1000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=1000)])
    assert b.fee_rate == 0.01 and ak.FEE_RATE_MIN == 0.0 and ak.FEE_RATE_MAX == 0.10
    assert math.isfinite(b.fee_rate)


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
