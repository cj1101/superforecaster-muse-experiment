"""
test_book_construction_validation.py — SF-BL-9051 verification tests.

Pre-registered done/kill criterion: "Validation fixture: 100% of seeded
invalid books (unsorted, price<=0 or >=1, size<=0, fee_rate<0 or >=1)
must be rejected before any gate runs; killed to a single-constructor
check if a census shows every fleet book constructor already validates
these invariants."

Kill NOT triggered (CALLER_CENSUS.md): no producer validates sortedness
(OrderBookLevel has no __post_init__; all 42 snapshot constructions are
bare literals; no deserialiser exists), and the dataclass is mutable.

Category resolution (fleet-supersession-aware):
  * unsorted            -> REJECTED at construction (ValueError); mutated
                           books SKIP UNSORTED_LADDER in evaluate (Gate
                           0.45, before the edge screen / Gate 3 / sizing)
                           and raise in sweep_order_book. [this task]
  * fee_rate <0 or >=1  -> REJECTED at construction (SF-BL-9363, bound
                           [0, 0.10] subset of [0, 1)); mutated books SKIP
                           INVALID_FEE_RATE at Gate 0.
  * price<=0 or >=1, size<=0 (pure-invalid books) -> evaluate SKIPs with
                           zero size and an untouched cluster dict.
  * price/size invalid RUNGS in a mixed book -> skipped identical-to-
                           clean (SF-BL-9398 / SF-BL-9542 pinned design);
                           they can alter no outcome, which is the
                           strongest form of "rejected before the gates".

Pre-fix harm (repro.py / repro_prepatch.txt): unsorted asks flipped the
clean BUY_YES $15,000 to an incidental Gate-D SKIP and swept at q_exec
0.6262 (clean 0.505); unsorted bids stored liquidation_bid_price 0.40
(clean 0.48) and swept BUY_NO at 0.606 (clean 0.5252).

Run:  python3 test_book_construction_validation.py  (exit 0 iff all pass)
"""
import os
import sys

for _cand in (
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "superforecaster"),
    "/home/hatch/workspace/superforecaster/superforecaster",
):
    if os.path.isdir(_cand):
        sys.path.insert(0, _cand)
        break

import alpha_kelly as ak  # noqa: E402

PASS = []
L = ak.OrderBookLevel
CLEAN_ASKS = [(0.50, 100_000), (0.62, 100_000)]
CLEAN_BIDS = [(0.48, 100_000), (0.40, 100_000)]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(asks=CLEAN_ASKS, bids=CLEAN_BIDS, fee=0.01):
    return ak.OrderBookSnapshot(
        market_id="FIXTURE", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C9051",
        est_days_to_resolution=7.0,
        yes_asks=[L(price=p, size_shares=s) for p, s in asks],
        yes_bids=[L(price=p, size_shares=s) for p, s in bids], fee_rate=fee)


def _mutated(asks=None, bids=None, fee=None):
    b = _book()
    if asks is not None:
        b.yes_asks = [L(price=p, size_shares=s) for p, s in asks]
    if bids is not None:
        b.yes_bids = [L(price=p, size_shares=s) for p, s in bids]
    if fee is not None:
        b.fee_rate = fee
    return b


def _evaluate(b, p=0.80):
    cl = {}
    return ak.evaluate_and_size_opportunity(b, p, 0.05, 100_000.0, cl), cl


def _raises_valueerror(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


UNSORTED_CASES = [
    ("asks pair", [(0.62, 100_000), (0.50, 100_000)], CLEAN_BIDS, "yes_asks"),
    ("asks late inversion", [(0.50, 1000), (0.62, 1000), (0.55, 1000)], CLEAN_BIDS, "yes_asks"),
    ("bids pair", CLEAN_ASKS, [(0.40, 100_000), (0.48, 100_000)], "yes_bids"),
    ("bids late inversion", CLEAN_ASKS, [(0.48, 1000), (0.40, 1000), (0.45, 1000)], "yes_bids"),
    ("both ladders", [(0.62, 100_000), (0.50, 100_000)],
     [(0.40, 100_000), (0.48, 100_000)], "yes_asks"),
]


# ---- 1. Unsorted: constructor rejection ------------------------------------
def test_constructor_rejects_unsorted():
    for label, asks, bids, _side in UNSORTED_CASES:
        assert _raises_valueerror(lambda: _book(asks, bids)), label


def test_constructor_error_names_ladder_and_task():
    try:
        _book([(0.62, 1000), (0.50, 1000)], CLEAN_BIDS)
        raise AssertionError("no raise")
    except ValueError as e:
        assert "yes_asks" in str(e) and "SF-BL-9051" in str(e), e


# ---- 2. Unsorted: mutated books rejected before gates/sweep -----------------
def test_mutated_unsorted_evaluate_skips_unsorted_ladder():
    for label, asks, bids, side in UNSORTED_CASES:
        r, cl = _evaluate(_mutated(asks, bids))
        assert r.action == ak.TradeAction.SKIP, (label, r)
        assert r.skip_reason.startswith("UNSORTED_LADDER"), (label, r.skip_reason)
        assert side in r.skip_reason, (label, r.skip_reason)
        assert r.invested_capital == 0.0 and r.shares_acquired == 0.0, (label, r)
        assert r.market_mid_prob == 0.0, (label, r)
        assert cl == {}, (label, cl)


def test_mutated_unsorted_sweep_raises_both_sides():
    for label, asks, bids, _side in UNSORTED_CASES:
        m = _mutated(asks, bids)
        for sd in (ak.TradeAction.BUY_YES, ak.TradeAction.BUY_NO):
            assert _raises_valueerror(
                lambda: ak.sweep_order_book(m, sd, 1000.0, 0.80, min_net_edge=-1.0)), (label, sd)


def test_unsorted_gate_fires_before_edge_and_gate_d():
    # p=0.80 vs the clean book BUY_YES $15,000; an unsorted book must SKIP
    # as UNSORTED_LADDER, never trade and never fall through to Gate D.
    r, cl = _evaluate(_mutated([(0.62, 100_000), (0.50, 100_000)], CLEAN_BIDS))
    assert r.skip_reason.startswith("UNSORTED_LADDER") and cl == {}


# ---- 3. Sorted controls are untouched ---------------------------------------
def test_sorted_controls_construct_and_trade_identically():
    r, cl = _evaluate(_book())
    assert r.action == ak.TradeAction.BUY_YES and r.invested_capital == 15000.0, r
    assert r.exec_price == 0.505 and r.liquidation_bid_price == 0.48, r
    assert cl == {"C9051": 0.15}, cl
    sw = ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.80, min_net_edge=-1.0)
    assert sw == (1000.0, 1980.19801980198, 0.505), sw


def test_equal_consecutive_prices_and_short_ladders_allowed():
    _book([(0.50, 1000), (0.50, 2000), (0.62, 1000)],
          [(0.48, 1000), (0.48, 2000), (0.40, 1000)])
    _book([(0.50, 1000)], [(0.48, 1000)])
    _book([], [])


def test_exit_walk_is_order_immune_by_design():
    # size_walked_exit_bid re-sorts explicitly (SF-BL-7432): a mutated
    # unsorted book yields the same walked bid as the sorted book.
    m = _mutated(bids=[(0.40, 100_000), (0.48, 100_000)])
    assert ak.size_walked_exit_bid(m, ak.TradeAction.BUY_YES, 1000.0) == \
        ak.size_walked_exit_bid(_book(), ak.TradeAction.BUY_YES, 1000.0)


def test_nonfinite_price_scope_documented():
    # NaN cannot be ordered; it is governed by SF-BL-9542/Gate 0.6, so the
    # ladder-order check does not fire and evaluate keeps INVALID_EQUITY.
    b = _book([(float("nan"), 1000), (0.50, 100_000)], CLEAN_BIDS)
    r, cl = _evaluate(b)
    assert r.action == ak.TradeAction.SKIP and "INVALID_EQUITY" in r.skip_reason, r
    assert cl == {}


# ---- 4. Fee category (SF-BL-9363 regression within this fixture) ------------
def test_fee_outside_0_1_rejected_at_construction():
    for fee in (-0.20, -0.001, 1.0, 2.0, float("nan"), float("inf")):
        assert _raises_valueerror(lambda: _book(fee=fee)), fee


def test_mutated_fee_at_or_above_1_skips_invalid_fee_rate():
    r, cl = _evaluate(_mutated(fee=1.0))
    assert r.action == ak.TradeAction.SKIP, r
    assert r.skip_reason.startswith("INVALID_FEE_RATE"), r.skip_reason
    assert r.invested_capital == 0.0 and cl == {}, (r, cl)


# ---- 5. Pure-invalid price / size books SKIP zero-size -----------------------
def test_pure_invalid_price_books_skip():
    for price in (0.0, -0.10, 1.0, 1.50):
        r, cl = _evaluate(_book([(price, 1000)], [(price, 1000)]))
        assert r.action == ak.TradeAction.SKIP, (price, r)
        assert r.invested_capital == 0.0 and r.shares_acquired == 0.0, (price, r)
        assert cl == {}, (price, cl)


def test_pure_invalid_size_books_skip_empty():
    for size in (0, -100):
        r, cl = _evaluate(_book([(0.50, size)], [(0.48, size)]))
        assert r.action == ak.TradeAction.SKIP, (size, r)
        assert r.skip_reason.startswith("EMPTY_BOOK"), (size, r.skip_reason)
        assert r.invested_capital == 0.0 and cl == {}, (size, r, cl)


# ---- 6. Mixed books: invalid rungs identical-to-clean (9398/9542) -----------
def test_mixed_invalid_rungs_cannot_alter_outcome():
    clean_r, clean_cl = _evaluate(_book())
    variants = [
        _book([(-0.10, 1000), (0.50, 100_000), (0.62, 100_000)], CLEAN_BIDS),
        _book([(0.50, 0), (0.50, 100_000), (0.62, 100_000)], CLEAN_BIDS),
        _book(CLEAN_ASKS, [(0.48, 100_000), (0.40, 100_000), (-0.30, 1000)]),
    ]
    # third variant: bids must stay descending -> -0.30 last is sorted.
    for v in variants:
        r, cl = _evaluate(v)
        assert (r.action, r.invested_capital, r.shares_acquired, r.exec_price,
                r.liquidation_bid_price) == (
            clean_r.action, clean_r.invested_capital, clean_r.shares_acquired,
            clean_r.exec_price, clean_r.liquidation_bid_price), r
        assert cl == clean_cl, cl


def test_out_of_order_unreadable_rungs_do_not_trip_order_check():
    # Sortedness is enforced over the live, tradeably-priced subsequence
    # only (the population consumers read): an invalid-price or
    # zero-size rung placed out of order is skipped everywhere it sits
    # (SF-BL-9398/9542 identity-to-clean) and must not reject the book.
    b = _book([(2.0, 1000), (0.50, 100_000), (0.62, 100_000)], CLEAN_BIDS)
    r, _cl = _evaluate(b)
    clean_r, _ = _evaluate(_book())
    assert (r.action, r.invested_capital, r.exec_price) == \
        (clean_r.action, clean_r.invested_capital, clean_r.exec_price), r
    _book([(0.62, 0), (0.50, 100_000), (0.62, 100_000)], CLEAN_BIDS)
    _book(CLEAN_ASKS, [(-0.30, 1000), (0.48, 100_000), (0.40, 100_000)])


# ---- 7. The prereg fixture: 100% of seeded invalid books rejected ------------
def test_seeded_invalid_fixture_100pct_rejected():
    seeds = []
    for label, asks, bids, _s in UNSORTED_CASES:
        seeds.append((f"unsorted:{label}", lambda a=asks, b=bids: _book(a, b), "construct"))
    for fee in (-0.20, 1.0):
        seeds.append((f"fee:{fee}", lambda f=fee: _book(fee=f), "construct"))
    for price in (0.0, -0.10, 1.0, 1.50):
        seeds.append((f"price:{price}", lambda p=price: _book([(p, 1000)], [(p, 1000)]), "skip"))
    for size in (0, -100):
        seeds.append((f"size:{size}", lambda s=size: _book([(0.50, s)], [(0.48, s)]), "skip"))
    rejected = 0
    for label, make, mode in seeds:
        if mode == "construct":
            ok = _raises_valueerror(make)
        else:
            r, cl = _evaluate(make())
            ok = (r.action == ak.TradeAction.SKIP and r.invested_capital == 0.0
                  and r.shares_acquired == 0.0 and cl == {})
        assert ok, label
        rejected += 1
    print(f"  seeded invalid books rejected: {rejected}/{len(seeds)}")
    assert rejected == len(seeds)


# ---- 8. Benchmark slate unchanged --------------------------------------------
def test_benchmark_unchanged():
    # All 8 benchmark books are sorted, so the SF-BL-9051 check must not
    # change the slate: it runs clean and its N_star is identical with
    # the ladder-order validation monkeypatched out (+0.0547 on the
    # current live tree; the historical +0.1059 headline predates other
    # concurrent fleet patches in this dirty tree — neutrality, not the
    # stale value, is what this task can pin).
    import contextlib
    import io

    def _run():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ak.run_benchmark_simulation()
        return [ln for ln in buf.getvalue().splitlines()
                if "North Star Score" in ln]

    with_validation = _run()
    saved_v, saved_f = ak._validate_ladder_order, ak._ladder_order_violation
    ak._validate_ladder_order = lambda a, b: None
    ak._ladder_order_violation = lambda a, b: None
    try:
        without_validation = _run()
    finally:
        ak._validate_ladder_order, ak._ladder_order_violation = saved_v, saved_f
    assert with_validation and with_validation == without_validation, (
        with_validation, without_validation)


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
