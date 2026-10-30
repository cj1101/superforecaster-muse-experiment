"""
test_alpha_kelly_level_field_types.py — SF-BL-11492 verification tests.

Pre-registered done/kill criterion: "OrderBookLevel construction (or
OrderBookSnapshot post-init) raises a named ValueError for a bool or
non-numeric price/size_shares, fixtured at price='0.50',
size_shares='100000' and size_shares=True, with numeric-level books
evaluating bit-identically; kill if a producer census shows every
ladder is built from validated floats before construction."

Kill NOT met (PRODUCER_AUDIT.md in the task work dir): 263 in-repo
construction sites pass bare literals/variables, the only converting
fleet scripts convert partially and are unpinned, no in-repo CLOB
deserialiser exists, and the dataclasses are mutable.

Scope pinned here: TYPE only. Numeric values — price 0.0/1.0/1.50/
-0.30/NaN/inf/10**400, size 0/negative/NaN — still construct and
keep their SF-BL-9398/9542 filtering behaviour bit-identically.

Run:  python3 test_alpha_kelly_level_field_types.py  (exit 0 iff all pass)
"""
import os
import sys
from decimal import Decimal

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


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(asks, bids, fee=0.01):
    return ak.OrderBookSnapshot(
        market_id="FIXTURE", title="SF-BL-11492 fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C11492",
        est_days_to_resolution=7.0, yes_asks=asks, yes_bids=bids,
        fee_rate=fee)


def _clean_book():
    return _book([L(price=0.50, size_shares=100_000)],
                 [L(price=0.48, size_shares=100_000)])


def _evaluate(b, p=0.80):
    cl = {}
    return ak.evaluate_and_size_opportunity(b, p, 0.05, 100_000.0, cl), cl


def _raises_11492(fn, field):
    try:
        fn()
    except ValueError as e:
        return "SF-BL-11492" in str(e) and field in str(e)
    except Exception:
        return False
    return False


# ---- 1. TASK fixtures -------------------------------------------------------
def test_task_fixtures_raise_named_valueerror():
    assert _raises_11492(lambda: L(price="0.50", size_shares=100000), "price")
    assert _raises_11492(lambda: L(price=0.50, size_shares="100000"), "size_shares")
    assert _raises_11492(lambda: L(price=0.50, size_shares=True), "size_shares")


def test_task_fixtures_raise_valueerror_not_typeerror():
    for fn in (lambda: L(price="0.50", size_shares=100000),
               lambda: L(price=0.50, size_shares="100000"),
               lambda: L(price=0.50, size_shares=True)):
        try:
            fn()
            raise AssertionError("no raise")
        except ValueError:
            pass


# ---- 2. Type grid ------------------------------------------------------------
def test_type_grid_raises_11492():
    bad_prices = [True, False, None, "", "0.50", "foo", [], {}, (0.5,),
                  0.5j, Decimal("0.5"), object()]
    bad_sizes = [True, False, None, "", "100000", "0", [], {}, (1000,),
                 1000j, Decimal("1000"), object()]
    for v in bad_prices:
        assert _raises_11492(lambda v=v: L(price=v, size_shares=1000), "price"), v
    for v in bad_sizes:
        assert _raises_11492(lambda v=v: L(price=0.50, size_shares=v), "size_shares"), v


def test_price_checked_before_size():
    try:
        L(price="0.50", size_shares="100000")
        raise AssertionError("no raise")
    except ValueError as e:
        assert "price" in str(e) and "SF-BL-11492" in str(e), e


def test_no_coercion():
    # '0.50' must never be parsed into a working level.
    try:
        L(price="0.50", size_shares=100000)
        raise AssertionError("no raise")
    except ValueError:
        pass
    assert not ak._level_field_is_numeric("0.50")
    assert not ak._level_field_is_numeric(True)
    assert ak._level_field_is_numeric(0.50)
    assert ak._level_field_is_numeric(1)


# ---- 3. Snapshot surface ------------------------------------------------------
def test_snapshot_with_bypass_level_raises_11492():
    lvl = object.__new__(L)
    object.__setattr__(lvl, "price", "0.50")
    object.__setattr__(lvl, "size_shares", 100000)
    assert _raises_11492(
        lambda: _book([lvl], [L(price=0.48, size_shares=100000)]), "price")
    lvl2 = object.__new__(L)
    object.__setattr__(lvl2, "price", 0.48)
    object.__setattr__(lvl2, "size_shares", True)
    assert _raises_11492(
        lambda: _book([L(price=0.50, size_shares=100000)], [lvl2]), "size_shares")


def test_string_book_no_longer_reads_one_sided():
    # The filing harm: the string book constructed and SKIPped
    # ONE_SIDED_BOOK at best_yes_ask 0.99. Post-fix it cannot be built.
    try:
        _book([L(price="0.50", size_shares=100000)],
              [L(price=0.48, size_shares=100000)])
        raise AssertionError("no raise")
    except ValueError as e:
        assert "SF-BL-11492" in str(e), e


# ---- 4. Assignment surface -----------------------------------------------------
def test_assignment_of_bad_type_raises_and_preserves_value():
    lvl = L(price=0.50, size_shares=1000)
    for field, val in [("price", "0.50"), ("price", True), ("price", None),
                       ("size_shares", "100000"), ("size_shares", True),
                       ("size_shares", None)]:
        try:
            setattr(lvl, field, val)
            raise AssertionError(f"no raise for {field}={val!r}")
        except ValueError as e:
            assert "SF-BL-11492" in str(e) and field in str(e), e
    assert lvl.price == 0.50 and lvl.size_shares == 1000, lvl


def test_numeric_assignment_still_works():
    # The repo's own mutation pattern (derived_no_price / zero_price suites).
    lvl = L(price=0.50, size_shares=1000)
    lvl.price = 1.50
    lvl.size_shares = 1000.0
    assert lvl.price == 1.50 and lvl.size_shares == 1000.0
    lvl.price = 0.0
    assert lvl.price == 0.0


# ---- 5. Numeric books bit-identical -------------------------------------------
def test_clean_book_evaluates_identically():
    b = _clean_book()
    assert b.best_yes_ask == 0.50 and b.best_yes_bid == 0.48, b
    assert b.mid_price == 0.49, b
    r, cl = _evaluate(b)
    assert r.action == ak.TradeAction.BUY_YES, r
    assert r.invested_capital == 15000.0, r
    assert r.shares_acquired == 29702.970297029704, r
    assert r.exec_price == 0.505 and r.liquidation_bid_price == 0.48, r
    assert cl == {"C11492": 0.15}, cl
    sw = ak.sweep_order_book(_clean_book(), ak.TradeAction.BUY_YES,
                             1000.0, 0.80, min_net_edge=-1.0)
    assert sw == (1000.0, 1980.19801980198, 0.505), sw


def test_numeric_edge_grid_constructs():
    for kw in [dict(price=0.0, size_shares=1000),
               dict(price=1.0, size_shares=1000),
               dict(price=1.50, size_shares=1000),
               dict(price=-0.30, size_shares=1000),
               dict(price=float("nan"), size_shares=1000),
               dict(price=float("inf"), size_shares=1000),
               dict(price=float("-inf"), size_shares=1000),
               dict(price=10**400, size_shares=1000),
               dict(price=1, size_shares=1000),
               dict(price=0.50, size_shares=0),
               dict(price=0.50, size_shares=-100),
               dict(price=0.50, size_shares=float("nan"))]:
        L(**kw)  # must not raise


def test_numeric_invalid_books_keep_prefix_outcomes():
    r, cl = _evaluate(_book([L(price=0.50, size_shares=0)],
                            [L(price=0.48, size_shares=0)]))
    assert r.action == ak.TradeAction.SKIP, r
    assert r.skip_reason.startswith("EMPTY_BOOK"), r.skip_reason
    assert r.invested_capital == 0.0 and cl == {}, (r, cl)
    r, cl = _evaluate(_book(
        [L(price=float("nan"), size_shares=1000),
         L(price=0.50, size_shares=100_000)],
        [L(price=0.48, size_shares=100_000)]))
    assert r.action == ak.TradeAction.SKIP, r
    assert "INVALID_EQUITY" in r.skip_reason, r.skip_reason
    assert cl == {}, cl


def test_converted_string_book_trades_identically():
    # The remediation the error names: float() conversion at the producer.
    raw_ask, raw_size = "0.50", "100000"
    b = _book([L(price=float(raw_ask), size_shares=float(raw_size))],
              [L(price=0.48, size_shares=100_000)])
    r, cl = _evaluate(b)
    clean_r, clean_cl = _evaluate(_clean_book())
    assert (r.action, r.invested_capital, r.exec_price) == \
        (clean_r.action, clean_r.invested_capital, clean_r.exec_price), r
    assert cl == clean_cl, cl


# ---- 6. Benchmark slate unchanged ----------------------------------------------
def test_benchmark_unchanged():
    import contextlib
    import io

    def _run():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ak.run_benchmark_simulation()
        return [ln for ln in buf.getvalue().splitlines()
                if "North Star Score" in ln]

    with_validation = _run()
    saved = ak._validate_order_book_level
    saved_f = ak._validate_order_book_level_field
    ak._validate_order_book_level = lambda p, s: None
    ak._validate_order_book_level_field = lambda n, v: None
    try:
        without_validation = _run()
    finally:
        ak._validate_order_book_level = saved
        ak._validate_order_book_level_field = saved_f
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
