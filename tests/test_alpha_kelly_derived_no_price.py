"""
test_derived_no_price_validation.py — SF-BL-9473 verification tests.

Pre-registered done/kill criteria (TASK.md):
  "Pass if sweep rejects or skips any derived NO rung priced outside
   (0,1) and BUY_NO on the benchmark unlock book still returns its
   current invested/shares/q_exec; kill if sweep is proven unreachable
   with mutated or externally built books (caller census including the
   naive baseline)."

Pre-fix reproduction (verified live 2026-10-21, before the patch;
see repro.py in the task work dir):
  YES bid mutated to 1.50 after construction, 1000 shares, fee 0.01:
    sweep BUY_NO -> invested=-505.0, shares=1000.0, q_exec=-0.505
    (the exact prereg numbers); naive-style (min_net_edge=-1.0) same.
  bid 1.0 -> ZeroDivisionError; bid 0.0 -> q_exec 1.01; bid -0.30 ->
  q_exec 1.313; bid 2.0 -> invested -1010.0; bid NaN -> NaN shares on a
  finite spend; bid +inf -> invested -inf. Mixed ladder (1.50 rung in
  front of the benchmark unlock ladder) executed through evaluate at
  419.15 / 3400 / 0.1233 instead of the clean 924.15 / 2400 / 0.3850625.

Kill criterion: NOT triggered — CALLER_CENSUS.md (task work dir) shows
OrderBookLevel has no price validation, OrderBookSnapshot.__post_init__
checks only fee_rate, both dataclasses are mutable/unpicklable past any
constructor check, and callers include evaluate_and_size_opportunity
and the naive baseline (direct sweep, min_net_edge=-1.0, return booked
unchecked).

Strongest-form checks used: a book with invalid-price rungs must sweep
IDENTICALLY to the same book with those rungs removed (both sides, a
grid of budgets / beliefs / min-edges), and the benchmark unlock book's
BUY_NO values must equal the pre-patch ground truth captured from the
unpatched module.

Run:  python3 test_derived_no_price_validation.py   (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_fee_rate.py.
"""
import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(
    os.path.expanduser("~/workspace/superforecaster"), "superforecaster"))

import alpha_kelly as ak  # noqa: E402

PASS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def unlock_book(bids=None, asks=None, fee=0.01):
    """Benchmark scenario 4 (CRYPTO-UNLOCK-HOLD-SUPPORT), verbatim unless
    bids/asks are overridden."""
    return ak.OrderBookSnapshot(
        market_id="CRYPTO-UNLOCK-HOLD-SUPPORT",
        title="Token ABC holds above $2.00 through 12% cliff unlock week",
        domain=ak.MarketDomain.CRYPTO,
        factor_cluster="TOKEN_UNLOCK_SHORT",
        est_days_to_resolution=5.0,
        yes_asks=asks if asks is not None
        else [ak.OrderBookLevel(price=0.68, size_shares=1200)],
        yes_bids=bids if bids is not None else [
            ak.OrderBookLevel(price=0.65, size_shares=900),
            ak.OrderBookLevel(price=0.60, size_shares=1500),
        ],
        fee_rate=fee,
        on_chain_verified=True,
    )


def lvl(price, size=1000.0):
    return ak.OrderBookLevel(price=price, size_shares=size)


# ---- 1. Prereg fixture -------------------------------------------------------

def test_prereg_fixture_mutated_bid_150():
    book = unlock_book()
    book.yes_bids = [lvl(1.50)]  # post-construction mutation (SF-BL-9363 path)
    for min_edge in (0.05, -1.0):
        got = ak.sweep_order_book(book, ak.TradeAction.BUY_NO, 10_000.0, 0.72,
                                  min_net_edge=min_edge)
        assert got == (0.0, 0.0, 0.0), (min_edge, got)  # pre-fix: (-505.0, 1000.0, -0.505)


def test_prereg_fixture_inplace_level_mutation():
    book = unlock_book()
    book.yes_bids[0].price = 1.50
    book.yes_bids = book.yes_bids[:1]
    book.yes_bids[0].size_shares = 1000.0
    got = ak.sweep_order_book(book, ak.TradeAction.BUY_NO, 10_000.0, 0.72)
    assert got == (0.0, 0.0, 0.0), got


def test_constructed_book_with_bad_bid():
    # OrderBookLevel / snapshot construction does not reject a 1.50 bid;
    # the sweep itself must refuse to price NO from it.
    book = unlock_book(bids=[lvl(1.50)])
    got = ak.sweep_order_book(book, ak.TradeAction.BUY_NO, 10_000.0, 0.72)
    assert got == (0.0, 0.0, 0.0), got


# ---- 2. Full out-of-(0,1) derived-price set ----------------------------------

BAD_BIDS = [1.0, 0.0, -0.30, 2.0, 1.50, -1.0,
            float("nan"), float("inf"), float("-inf"), True, False]


def test_all_bad_bids_sweep_to_zero_without_raising():
    for bid in BAD_BIDS:
        book = unlock_book(bids=[lvl(bid)])
        for min_edge in (0.05, -1.0):
            got = ak.sweep_order_book(book, ak.TradeAction.BUY_NO, 1_200.0,
                                      0.72, min_net_edge=min_edge)
            assert got == (0.0, 0.0, 0.0), (bid, min_edge, got)


def test_helper_domain():
    for good in (0.5, 0.001, 0.999, 0.999999, 1e-9):
        assert ak._is_valid_contract_price(good), good
    for bad in (0.0, 1.0, -0.1, 1.5, float("nan"), float("inf"),
                float("-inf"), "0.5", None, True, False):
        assert not ak._is_valid_contract_price(bad), bad


def test_near_boundary_valid_bids_still_trade():
    # Open interval: bids arbitrarily close to the bounds derive valid
    # NO prices and must still sweep (fee 0.0 keeps the rung price < 1).
    book = unlock_book(bids=[lvl(0.999)], fee=0.0)
    inv, sh, q = ak.sweep_order_book(book, ak.TradeAction.BUY_NO, 100.0, 0.90,
                                     min_net_edge=-1.0)
    assert inv > 0 and sh > 0 and abs(q - 0.001) < 1e-9, (inv, sh, q)
    book = unlock_book(bids=[lvl(0.001)], fee=0.0)
    inv, sh, q = ak.sweep_order_book(book, ak.TradeAction.BUY_NO, 100.0, 0.90,
                                     min_net_edge=-1.0)
    assert inv > 0 and sh > 0 and abs(q - 0.999) < 1e-9, (inv, sh, q)


# ---- 3. Mixed ladders are identical to clean ladders -------------------------

def test_mixed_no_ladder_identical_to_clean():
    clean = unlock_book()
    bad_rungs = [lvl(1.50), lvl(1.0), lvl(0.0), lvl(-0.30), lvl(float("nan")),
                 lvl(2.0, 500.0)]
    for bad in bad_rungs:
        for pos in (0, 1, 2):
            bids = [lvl(0.65, 900.0), lvl(0.60, 1500.0)]
            bids.insert(pos, bad)
            mixed = unlock_book(bids=bids)
            for budget in (100.0, 500.0, 1200.0, 10_000.0):
                for belief in (0.5, 0.72, 0.9):
                    for min_edge in (0.05, -1.0):
                        a = ak.sweep_order_book(mixed, ak.TradeAction.BUY_NO,
                                                budget, belief, min_net_edge=min_edge)
                        b = ak.sweep_order_book(clean, ak.TradeAction.BUY_NO,
                                                budget, belief, min_net_edge=min_edge)
                        assert a == b, (bad.price, pos, budget, belief, min_edge, a, b)


def test_bad_yes_asks_sweep_to_zero():
    # Symmetric raw-ask domain: any ordered price outside (0, 1) is
    # skipped on the YES side too (NaN asks are SF-BL-9438's domain —
    # see test_nan_yes_ask_remains_9438_invalid_equity).
    for bad_price in (1.50, 0.0, 1.0, -0.50, 2.0,
                      float("inf"), float("-inf")):
        book = unlock_book(asks=[lvl(bad_price)])
        got = ak.sweep_order_book(book, ak.TradeAction.BUY_YES, 1_200.0,
                                  0.90, min_net_edge=-1.0)
        assert got == (0.0, 0.0, 0.0), (bad_price, got)


def test_nan_yes_ask_remains_9438_invalid_equity():
    # A deep NaN ask is NOT redefined by SF-BL-9473: evaluate still SKIPs
    # INVALID_EQUITY with the cluster untouched (SF-BL-9438's pinned
    # behaviour, tests/test_alpha_kelly_invalid_equity.py).
    book = unlock_book(asks=[lvl(0.50, 1000.0), lvl(float("nan"), 1000.0)],
                       bids=[lvl(0.48, 1000.0)], fee=0.0)
    cluster = {}
    res = ak.evaluate_and_size_opportunity(
        book=book, calibrated_prob_yes=0.80, inter_agent_dispersion=0.0,
        portfolio_equity=10_000.0, cluster_exposure_used=cluster)
    assert res.action == ak.TradeAction.SKIP, res.action
    assert res.skip_reason and res.skip_reason.startswith("INVALID_EQUITY"), res.skip_reason
    assert cluster == {}, cluster


def test_mixed_yes_ladder_identical_to_clean():
    clean = unlock_book(asks=[lvl(0.50, 2000.0), lvl(0.55, 2000.0)])
    for bad_price in (1.50, 0.0, 1.0, -0.50, float("inf"), float("-inf")):
        asks = [lvl(0.50, 2000.0), lvl(bad_price, 700.0), lvl(0.55, 2000.0)]
        mixed = unlock_book(asks=asks)
        for budget in (100.0, 500.0, 5_000.0):
            for min_edge in (0.05, -1.0):
                a = ak.sweep_order_book(mixed, ak.TradeAction.BUY_YES,
                                        budget, 0.90, min_net_edge=min_edge)
                b = ak.sweep_order_book(clean, ak.TradeAction.BUY_YES,
                                        budget, 0.90, min_net_edge=min_edge)
                assert a == b, (bad_price, budget, min_edge, a, b)


def test_invalid_rung_plus_zero_size_interplay():
    # SF-BL-9398 live filter and SF-BL-9473 price filter compose: a book
    # whose only extra rungs are zero-size or bad-price sweeps as clean.
    bids = [lvl(1.50), lvl(0.65, 0.0), lvl(0.65, 900.0), lvl(0.60, 1500.0)]
    mixed = unlock_book(bids=bids)
    got = ak.sweep_order_book(mixed, ak.TradeAction.BUY_NO, 10_000.0, 0.72)
    assert got == (924.15, 2400.0, 0.3850625), got


# ---- 4. Benchmark unlock book BUY_NO unchanged (pre-patch ground truth) ------

UNLOCK_TRUTH = [
    # (side, budget, belief, min_edge, expected tuple) — captured live from
    # the UNPATCHED module on 2026-10-21 before the fix was applied.
    ("BUY_NO", 100.0, 0.5, 0.05, (100.0, 282.8854314002829, 0.3534999999999999)),
    ("BUY_NO", 100.0, 0.72, -1.0, (100.0, 282.8854314002829, 0.3534999999999999)),
    ("BUY_NO", 500.0, 0.5, 0.05, (500.0, 1350.1237623762377, 0.3703364194701622)),
    ("BUY_NO", 500.0, 0.72, -1.0, (500.0, 1350.1237623762377, 0.3703364194701622)),
    ("BUY_NO", 1200.0, 0.72, 0.05, (924.15, 2400.0, 0.3850625)),
    ("BUY_NO", 10000.0, 0.5, -1.0, (924.15, 2400.0, 0.3850625)),
    ("BUY_NO", 10000.0, 0.72, 0.05, (924.15, 2400.0, 0.3850625)),
    ("BUY_YES", 100.0, 0.72, -1.0, (100.0, 145.602795573675, 0.6868000000000001)),
    ("BUY_YES", 500.0, 0.5, -1.0, (500.0, 728.013977868375, 0.6868000000000001)),
    ("BUY_YES", 1200.0, 0.72, -1.0, (824.1600000000001, 1200.0, 0.6868000000000001)),
    ("BUY_YES", 100.0, 0.72, 0.05, (0.0, 0.0, 0.0)),
]


def test_unlock_book_ground_truth_unchanged():
    for side, budget, belief, min_edge, expected in UNLOCK_TRUTH:
        got = ak.sweep_order_book(unlock_book(), ak.TradeAction[side],
                                  budget, belief, min_net_edge=min_edge)
        assert got == expected, (side, budget, belief, min_edge, got, expected)


# ---- 5. Evaluate path fails closed -------------------------------------------

def _eval(book, p, cluster):
    return ak.evaluate_and_size_opportunity(
        book=book, calibrated_prob_yes=p, inter_agent_dispersion=0.05,
        portfolio_equity=10_000.0, cluster_exposure_used=cluster)


def test_evaluate_all_invalid_ladder_skips_zero_cluster_untouched():
    book = unlock_book(bids=[lvl(1.50)])
    cluster = {}
    res = _eval(book, 0.28, cluster)
    assert res.action == ak.TradeAction.SKIP, res.action
    assert res.invested_capital == 0.0 and res.shares_acquired == 0.0
    assert res.exec_price == 0.0 and cluster == {}, cluster


def test_evaluate_mixed_ladder_identical_to_clean():
    clean_cluster, mixed_cluster = {}, {}
    clean = _eval(unlock_book(), 0.28, clean_cluster)
    bids = [lvl(1.50), lvl(0.65, 900.0), lvl(0.60, 1500.0)]
    mixed = _eval(unlock_book(bids=bids), 0.28, mixed_cluster)
    assert mixed.action == clean.action == ak.TradeAction.BUY_NO
    assert mixed.invested_capital == clean.invested_capital == 924.15, (
        mixed.invested_capital, clean.invested_capital)  # pre-fix mixed: 419.15
    assert mixed.shares_acquired == clean.shares_acquired == 2400.0
    assert mixed.exec_price == clean.exec_price == 0.3850625
    assert mixed_cluster == clean_cluster


# ---- 6. Built-in benchmark unchanged ------------------------------------------

def test_benchmark_simulation_unchanged():
    buf = io.StringIO()
    with redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    assert '"north_star_score": 0.1059' in out, out[:2000]
    assert '"north_star_score": -0.1499' in out, out[:2000]


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    for name, fn in fns:
        try:
            fn()
            check(name, True)
        except Exception as e:  # noqa: BLE001
            check(name, False, f"-> {type(e).__name__}: {e}")
    print(f"\n{sum(PASS)}/{len(PASS)} passed")
    sys.exit(0 if all(PASS) else 1)
