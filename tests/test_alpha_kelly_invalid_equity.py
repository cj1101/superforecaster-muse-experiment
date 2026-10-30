"""
test_invalid_equity.py — SF-BL-9438 verification tests.

Pre-registered done criterion: "Fail closed on non-finite portfolio_equity
(and non-finite sweep returns) with an INVALID_EQUITY skip reason before
sizing, fixtured at NaN and inf equity with the benchmark slate unchanged;
kill if a single upstream equity producer is proven finite-only (then
assert finiteness at that producer and reference it in the docstring)."

Pre-fix reproduction (verified live 2026-10-21, before the patch; repro.py):
  fixture book (ask 0.50 x 100k, bid 0.48, fee 0.0, p=0.80):
    equity=10000 -> BUY_YES invested 1500.0 allocated 0.15 shares 3000.0
    equity=NaN   -> BUY_YES invested/allocated/shares/exec ALL NaN,
                    cluster {'C9438': nan} (poisoned)
    equity=+inf  -> BUY_YES invested 50000.0 (whole ladder) allocated 0.0,
                    cluster {'C9438': 0.0} (cap-invisible full-book trade)
    equity=-inf  -> SKIP ORDER_BOOK_DEPTH_EXHAUSTED_AT_REQUIRED_EDGE
    sweep budget=NaN -> (nan, nan, nan); budget=+inf -> (50000, 100000, 0.5)
    deep NaN-price rung -> BUY_YES invested 1500.0 shares nan exec nan

Kill condition is NOT met (see PRODUCER_AUDIT.md): equity has multiple
producers (benchmark PnL-derived equity, r63 literal, test literals, live
account-derived values), none proven finite-only — so the fix, not the
kill, applies.

Run:  python3 test_invalid_equity.py   (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_cli_edge.py / test_alpha_kelly_fee_rate.py.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "superforecaster"))
# Fallback when run from the backlog work dir (not the repo tests/ dir):
if not os.path.exists(os.path.join(sys.path[0], "alpha_kelly.py")):
    sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster/superforecaster"))

import alpha_kelly as ak  # noqa: E402

PASS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(cluster="C9438", asks=None, bids=None):
    return ak.OrderBookSnapshot(
        market_id="FIXTURE", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster=cluster,
        est_days_to_resolution=5,
        yes_asks=asks if asks is not None else [ak.OrderBookLevel(price=0.50, size_shares=100_000)],
        yes_bids=bids if bids is not None else [ak.OrderBookLevel(price=0.48, size_shares=100_000)],
        fee_rate=0.0)


def _assert_invalid_equity_skip(r, used_before, used_after):
    assert r.action == ak.TradeAction.SKIP, r
    assert r.skip_reason is not None and r.skip_reason.startswith("INVALID_EQUITY"), r.skip_reason
    assert r.invested_capital == 0.0 and r.shares_acquired == 0.0 and r.exec_price == 0.0, r
    assert r.kelly_fraction_raw == 0.0 and r.allocated_fraction == 0.0, r  # EC-09
    assert all(math.isfinite(v) for v in (
        r.invested_capital, r.shares_acquired, r.exec_price,
        r.allocated_fraction, r.kl_divergence_nats, r.expected_log_growth)), r
    assert used_after == used_before, (used_before, used_after)  # cluster untouched


# ---- 1. Prereg fixtures: NaN and inf equity ---------------------------------

def test_nan_equity_skips_invalid_equity():
    used = {}
    r = ak.evaluate_and_size_opportunity(_book(), 0.80, 0.0, float("nan"), used)
    _assert_invalid_equity_skip(r, {}, used)


def test_inf_equity_skips_invalid_equity():
    for bad in (float("inf"), float("-inf")):
        used = {}
        r = ak.evaluate_and_size_opportunity(_book(), 0.80, 0.0, bad, used)
        _assert_invalid_equity_skip(r, {}, used)


def test_nonfinite_equity_never_poisons_cluster():
    # Pre-fix the NaN trade wrote nan into the cluster dict and every later
    # cap check inherited it. Post-fix the dict is untouched and a follow-up
    # valid trade on the same cluster sizes normally.
    used = {}
    ak.evaluate_and_size_opportunity(_book(), 0.80, 0.0, float("nan"), used)
    assert used == {}, used
    r = ak.evaluate_and_size_opportunity(_book(), 0.80, 0.0, 10000.0, used)
    assert r.action == ak.TradeAction.BUY_YES and r.invested_capital == 1500.0, r
    assert used == {"C9438": 0.15}, used


def test_other_invalid_equity_values_skip():
    # Non-positive equity is the same caller capital error (SF-BL-9401's
    # depth-mislabel fixture); non-numeric values must not raise TypeError.
    for bad in (0.0, -5000.0, "10000", None, True):
        used = {}
        r = ak.evaluate_and_size_opportunity(_book(), 0.80, 0.0, bad, used)
        _assert_invalid_equity_skip(r, {}, used)


def test_equity_gate_precedes_crypto_and_dispersion_gates():
    # Before sizing means before the other gates too: an unverified crypto
    # book with high dispersion still reports INVALID_EQUITY first.
    b = _book()
    b.domain = ak.MarketDomain.CRYPTO
    b.on_chain_verified = False
    r = ak.evaluate_and_size_opportunity(b, 0.80, 0.99, float("nan"), {})
    assert r.skip_reason.startswith("INVALID_EQUITY"), r.skip_reason


# ---- 2. Non-finite sweep returns ---------------------------------------------

def test_sweep_nonfinite_budget_fails_closed():
    assert ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, float("nan"), 0.80) == (0.0, 0.0, 0.0)
    assert ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, float("inf"), 0.80) == (0.0, 0.0, 0.0)
    assert ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, float("-inf"), 0.80) == (0.0, 0.0, 0.0)


def test_deep_nan_rung_sweep_return_skips_invalid_equity():
    # Finite budget/equity, but a deep rung at a NaN price makes the sweep
    # return (finite invested, NaN shares, NaN q_exec) — pre-fix this
    # EXECUTED. Post-fix: INVALID_EQUITY skip, cluster untouched.
    b = _book(cluster="C9438-DEEP", asks=[
        ak.OrderBookLevel(price=0.50, size_shares=1000),
        ak.OrderBookLevel(price=float("nan"), size_shares=1000)])
    used = {}
    r = ak.evaluate_and_size_opportunity(b, 0.80, 0.0, 10000.0, used)
    _assert_invalid_equity_skip(r, {}, used)


def test_sweep_valid_budget_unchanged():
    inv, sh, q = ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 500.0, 0.80)
    assert (inv, sh, q) == (500.0, 1000.0, 0.50), (inv, sh, q)
    assert ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 0.0, 0.80) == (0.0, 0.0, 0.0)


# ---- 3. No regression on valid-equity paths ----------------------------------

def test_trade_control_unchanged():
    used = {}
    r = ak.evaluate_and_size_opportunity(_book(), 0.80, 0.0, 10000.0, used)
    assert r.action == ak.TradeAction.BUY_YES, r
    assert r.invested_capital == 1500.0 and r.shares_acquired == 3000.0, r
    assert r.allocated_fraction == 0.15 and r.exec_price == 0.50, r
    assert used == {"C9438": 0.15}, used


def test_int_equity_accepted():
    r = ak.evaluate_and_size_opportunity(_book(), 0.80, 0.0, 10000, {})
    assert r.action == ak.TradeAction.BUY_YES and r.invested_capital == 1500.0, r


def test_genuine_depth_exhaustion_keeps_its_reason():
    # A tradeable book with a valid (positive finite) but dust-sized equity
    # produces a sub-micro budget the sweep cannot fill: that still reports
    # depth exhaustion, not INVALID_EQUITY — the equity gate must not
    # swallow genuine sweep outcomes.
    r = ak.evaluate_and_size_opportunity(_book(), 0.80, 0.0, 1e-6, {})
    assert r.action == ak.TradeAction.SKIP, r
    assert r.skip_reason.startswith("ORDER_BOOK_DEPTH_EXHAUSTED"), r.skip_reason


def test_helper_domain():
    assert ak._is_valid_portfolio_equity(10000.0) and ak._is_valid_portfolio_equity(1)
    for bad in (float("nan"), float("inf"), float("-inf"), 0.0, -1.0, "x", None, True):
        assert not ak._is_valid_portfolio_equity(bad), bad


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
