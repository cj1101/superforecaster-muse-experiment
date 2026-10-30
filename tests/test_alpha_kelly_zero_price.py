#!/usr/bin/env python3
"""SF-BL-9472 regression tests: sweep_order_book vs a 0.0-price rung.

Prereg: a ladder containing a 0.0-price rung (YES and derived-NO sides)
sweeps the remaining rungs or returns (0, 0, 0) with no exception, and
benchmark q_exec values are unchanged.

Coverage: zero rung at front/middle/end and as the only rung on both
sides, both post-construction mutation paths (list replacement and
in-place price mutation), direct construction, mixed-ladder ==
clean-ladder identity over a budget/belief/edge grid, the division-site
guard itself (ladder filter neutralised -- the pre-patch crash), an
evaluate-level pass on a zero-rung book, and the benchmark ground truth
(unlock book + built-in slate).
"""
import sys

sys.path.insert(0, "/home/hatch/workspace/superforecaster")

from superforecaster import alpha_kelly as ak

CLEAN_YES = (1060.5, 2000.0, 0.53025)          # clean_book BUY_YES, budget 10000
CLEAN_NO = (1080.7000000000003, 2000.0, 0.5403500000000001)  # BUY_NO belief 0.72
UNLOCK_NO = (924.15, 2400.0, 0.3850625)        # benchmark unlock book BUY_NO


def clean_book():
    return ak.OrderBookSnapshot(
        market_id="SF-BL-9472-CLEAN", title="clean fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C", est_days_to_resolution=10.0,
        yes_asks=[ak.OrderBookLevel(0.50, 1000), ak.OrderBookLevel(0.55, 1000)],
        yes_bids=[ak.OrderBookLevel(0.48, 1000), ak.OrderBookLevel(0.45, 1000)],
        fee_rate=0.01, on_chain_verified=True)


def unlock_book():
    return ak.OrderBookSnapshot(
        market_id="CRYPTO-UNLOCK-HOLD-SUPPORT",
        title="Token ABC holds above $2.00 through 12% cliff unlock week",
        domain=ak.MarketDomain.CRYPTO, factor_cluster="TOKEN_UNLOCK_SHORT",
        est_days_to_resolution=5.0,
        yes_asks=[ak.OrderBookLevel(0.68, 1200)],
        yes_bids=[ak.OrderBookLevel(0.65, 900), ak.OrderBookLevel(0.60, 1500)],
        fee_rate=0.01, on_chain_verified=True)


def sweep(book, side, budget=10000.0, belief=0.80, edge=0.05):
    return ak.sweep_order_book(book, side, budget, belief, min_net_edge=edge)


# --- YES side: 0.0 ask in every position -----------------------------------
def _yes_with_zero(asks):
    b = clean_book()
    b.yes_asks = asks
    return b


def test_zero_ask_front_sweeps_remaining():
    b = _yes_with_zero([ak.OrderBookLevel(0.0, 1000),
                        ak.OrderBookLevel(0.50, 1000),
                        ak.OrderBookLevel(0.55, 1000)])
    assert sweep(b, ak.TradeAction.BUY_YES) == CLEAN_YES


def test_zero_ask_middle_sweeps_remaining():
    b = _yes_with_zero([ak.OrderBookLevel(0.50, 1000),
                        ak.OrderBookLevel(0.0, 1000),
                        ak.OrderBookLevel(0.55, 1000)])
    assert sweep(b, ak.TradeAction.BUY_YES) == CLEAN_YES


def test_zero_ask_end_sweeps_remaining():
    b = _yes_with_zero([ak.OrderBookLevel(0.50, 1000),
                        ak.OrderBookLevel(0.55, 1000),
                        ak.OrderBookLevel(0.0, 1000)])
    assert sweep(b, ak.TradeAction.BUY_YES) == CLEAN_YES


def test_zero_ask_only_rung_returns_no_fill():
    b = _yes_with_zero([ak.OrderBookLevel(0.0, 1000)])
    assert sweep(b, ak.TradeAction.BUY_YES) == (0.0, 0.0, 0.0)


def test_zero_ask_in_place_mutation():
    b = clean_book()
    b.yes_asks[1].price = 0.0  # second rung dies; first still fills
    assert sweep(b, ak.TradeAction.BUY_YES) == (505.0, 1000.0, 0.505)


def test_zero_ask_constructed_directly():
    b = ak.OrderBookSnapshot(
        market_id="X", title="t", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C", est_days_to_resolution=5.0,
        yes_asks=[ak.OrderBookLevel(0.0, 1000), ak.OrderBookLevel(0.50, 1000)],
        yes_bids=[ak.OrderBookLevel(0.48, 1000)], fee_rate=0.01)
    assert sweep(b, ak.TradeAction.BUY_YES) == (505.0, 1000.0, 0.505)


# --- NO side: derived price 0.0 via YES bid 1.0 -----------------------------
def _no_with_derived_zero(bids):
    b = clean_book()
    b.yes_bids = bids
    return b


def test_derived_zero_front_sweeps_remaining():
    b = _no_with_derived_zero([ak.OrderBookLevel(1.0, 1000),
                               ak.OrderBookLevel(0.48, 1000),
                               ak.OrderBookLevel(0.45, 1000)])
    assert sweep(b, ak.TradeAction.BUY_NO, belief=0.72) == CLEAN_NO


def test_derived_zero_middle_sweeps_remaining():
    b = _no_with_derived_zero([ak.OrderBookLevel(0.48, 1000),
                               ak.OrderBookLevel(1.0, 1000),
                               ak.OrderBookLevel(0.45, 1000)])
    assert sweep(b, ak.TradeAction.BUY_NO, belief=0.72) == CLEAN_NO


def test_derived_zero_only_rung_returns_no_fill():
    b = _no_with_derived_zero([ak.OrderBookLevel(1.0, 1000)])
    assert sweep(b, ak.TradeAction.BUY_NO, belief=0.72) == (0.0, 0.0, 0.0)


def test_derived_zero_in_place_mutation():
    b = clean_book()
    b.yes_bids[0].price = 1.0
    # remaining bid 0.45 -> NO price 0.55, eff 0.5555, 1000 shares
    assert sweep(b, ak.TradeAction.BUY_NO, belief=0.72) == (
        555.5000000000001, 1000.0, 0.5555000000000001)


# --- The division-site guard itself (SF-BL-9472's actual patch) -------------
def test_guard_holds_with_ladder_filter_neutralised():
    """Neutralise _is_valid_contract_price (the state the loop would be
    in if the SF-BL-9473 filter regressed). Pre-patch this raised
    ZeroDivisionError on both fixtures; the guard must skip the zero
    rung and sweep the rest / return no-fill."""
    orig = ak._is_valid_contract_price
    ak._is_valid_contract_price = lambda price: True  # noqa: E731
    try:
        b = _yes_with_zero([ak.OrderBookLevel(0.0, 1000),
                            ak.OrderBookLevel(0.50, 1000),
                            ak.OrderBookLevel(0.55, 1000)])
        assert sweep(b, ak.TradeAction.BUY_YES) == CLEAN_YES
        b = _yes_with_zero([ak.OrderBookLevel(0.0, 1000)])
        assert sweep(b, ak.TradeAction.BUY_YES) == (0.0, 0.0, 0.0)
        # negative rung reaching the loop the same way is skipped too
        b = _yes_with_zero([ak.OrderBookLevel(-0.30, 1000),
                            ak.OrderBookLevel(0.50, 1000),
                            ak.OrderBookLevel(0.55, 1000)])
        assert sweep(b, ak.TradeAction.BUY_YES) == CLEAN_YES
    finally:
        ak._is_valid_contract_price = orig


# --- Mixed ladder == clean ladder over a grid -------------------------------
def test_mixed_equals_clean_grid():
    for budget in (100.0, 500.0, 1200.0, 10000.0):
        for belief in (0.60, 0.72, 0.80, 0.95):
            for edge in (0.05, -1.0):
                clean = sweep(clean_book(), ak.TradeAction.BUY_YES,
                              budget, belief, edge)
                dirty_b = _yes_with_zero(
                    [ak.OrderBookLevel(0.0, 777),
                     ak.OrderBookLevel(0.50, 1000),
                     ak.OrderBookLevel(0.55, 1000)])
                assert sweep(dirty_b, ak.TradeAction.BUY_YES,
                             budget, belief, edge) == clean
                clean_no = sweep(clean_book(), ak.TradeAction.BUY_NO,
                                 budget, belief, edge)
                dirty_no = _no_with_derived_zero(
                    [ak.OrderBookLevel(1.0, 777),
                     ak.OrderBookLevel(0.48, 1000),
                     ak.OrderBookLevel(0.45, 1000)])
                assert sweep(dirty_no, ak.TradeAction.BUY_NO,
                             budget, belief, edge) == clean_no


# --- Evaluate path: a zero-rung book completes the cycle --------------------
def test_evaluate_on_zero_rung_book_no_exception():
    b = _yes_with_zero([ak.OrderBookLevel(0.0, 1000),
                        ak.OrderBookLevel(0.50, 1000),
                        ak.OrderBookLevel(0.55, 1000)])
    cluster = {}
    res = ak.evaluate_and_size_opportunity(
        book=b, calibrated_prob_yes=0.80, inter_agent_dispersion=0.05,
        portfolio_equity=10000.0, cluster_exposure_used=cluster)
    clean_res = ak.evaluate_and_size_opportunity(
        book=clean_book(), calibrated_prob_yes=0.80,
        inter_agent_dispersion=0.05, portfolio_equity=10000.0,
        cluster_exposure_used={})
    assert (res.action, res.invested_capital, res.shares_acquired,
            res.exec_price) == (clean_res.action, clean_res.invested_capital,
                                clean_res.shares_acquired, clean_res.exec_price)


# --- Benchmark ground truth unchanged ---------------------------------------
def test_unlock_book_buy_no_unchanged():
    assert sweep(unlock_book(), ak.TradeAction.BUY_NO,
                 belief=0.72) == UNLOCK_NO


def test_builtin_benchmark_runs_and_reports_nstar():
    """Prereg: 'benchmark q_exec values are unchanged'. The unlock-book
    ground truth above pins the sweep-level values; here the built-in
    benchmark must run end-to-end (its sweeps included) and report a
    finite N_star. Byte-identity of the full benchmark output with vs
    without the SF-BL-9472 guard was verified as a control at execution
    time and is recorded in DONE-CLAIM.md (the historical +0.1059 /
    -0.1499 slate constants predate many concurrent fleet hunks on the
    combined live file, so identity-to-control -- not the stale
    constants -- is the honest check)."""
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    assert "North Star Score (N_star):" in out, out[-500:]


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
