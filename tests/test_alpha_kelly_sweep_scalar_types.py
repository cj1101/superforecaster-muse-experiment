"""SF-BL-11045 regression tests: sweep_order_book scalar TYPE dimension.

Prereg: "sweep_order_book raises a named ValueError (or returns an
explicitly flagged invalid marker) for a bool or non-numeric
target_budget, belief_prob_side or min_net_edge, fixtured at '1000',
True, '0.8' and None, with finite-argument sweeps bit-identical; kill
if a caller census proves every sweep caller passes only validated
floats under a tested invariant."

Kill NOT met (CALLER_CENSUS.md in the task work dir).

State found this run (repro.py in the task work dir): on a book that
fills $1,000 clean — (1000.0, 1980.198..., 0.505) — pre-fix
target_budget='1000'/True/None returned (0.0, 0.0, 0.0),
indistinguishable from genuine depth exhaustion, while
belief_prob_side='0.8' and min_net_edge=None ALREADY raised named
ValueErrors via SF-BL-10502 (landed after this task was filed).

Fix (alpha_kelly.py, SF-BL-11045): _validate_target_budget_type runs
in sweep_order_book before the SF-BL-9438 finiteness return; a bool
or non-numeric target_budget raises ValueError
INVALID_TARGET_BUDGET_TYPE. TYPE ONLY: every numeric budget is
bit-identical (finite sweeps unchanged; NaN/+-inf and zero/negative
budgets keep their (0.0, 0.0, 0.0) returns). Belief/hurdle type raises
(SF-BL-10502) are pinned here unchanged.

Run:  python3 test_alpha_kelly_sweep_scalar_types.py  (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner.
"""
import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "superforecaster"))
if not os.path.exists(os.path.join(sys.path[0], "alpha_kelly.py")):
    sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster/superforecaster"))

import alpha_kelly as ak  # noqa: E402

PASS = []
NAN = float("nan")
INF = float("inf")
ZERO = (0.0, 0.0, 0.0)
CLEAN = (1000.0, 1980.19801980198, 0.505)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(fee=0.01):
    return ak.OrderBookSnapshot(
        market_id="FIXTURE-11045", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C11045",
        est_days_to_resolution=5,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100_000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100_000)],
        fee_rate=fee)


def _raises(fn, token, task_id):
    try:
        fn()
    except ValueError as e:
        assert token in str(e), (token, str(e))
        assert task_id in str(e), str(e)
        return
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"wrong exception {type(e).__name__}: {e}")
    raise AssertionError(f"no ValueError raised (expected {token})")


# ---- 1. Prereg fixtures: all three scalar TYPE errors raise named ----------

def test_prereg_budget_string_raises():
    _raises(lambda: ak.sweep_order_book(
        _book(), ak.TradeAction.BUY_YES, "1000", 0.8, 0.05),
        "INVALID_TARGET_BUDGET_TYPE", "SF-BL-11045")


def test_prereg_budget_bool_raises():
    _raises(lambda: ak.sweep_order_book(
        _book(), ak.TradeAction.BUY_YES, True, 0.8, 0.05),
        "INVALID_TARGET_BUDGET_TYPE", "SF-BL-11045")


def test_prereg_belief_string_raises():
    _raises(lambda: ak.sweep_order_book(
        _book(), ak.TradeAction.BUY_YES, 1000.0, "0.8", 0.05),
        "INVALID_BELIEF_PROB_SIDE", "SF-BL-10502")


def test_prereg_hurdle_none_raises():
    _raises(lambda: ak.sweep_order_book(
        _book(), ak.TradeAction.BUY_YES, 1000.0, 0.8, None),
        "INVALID_MIN_NET_EDGE", "SF-BL-10502")


# ---- 2. Budget type classes --------------------------------------------------

def test_budget_bool_and_nonnumeric_raise():
    for bad in (True, False, None, "1000", "0", "", [], {}, object()):
        _raises(lambda b=bad: ak.sweep_order_book(
            _book(), ak.TradeAction.BUY_YES, b, 0.8, 0.05),
            "INVALID_TARGET_BUDGET_TYPE", "SF-BL-11045")
        _raises(lambda b=bad: ak.sweep_order_book(
            _book(), ak.TradeAction.BUY_NO, b, 0.8, 0.05),
            "INVALID_TARGET_BUDGET_TYPE", "SF-BL-11045")


def test_budget_type_error_names_value_and_type():
    try:
        ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, "1000", 0.8, 0.05)
    except ValueError as e:
        assert "'1000'" in str(e) and "str" in str(e), str(e)
        return
    raise AssertionError("no raise")


def test_budget_raises_before_walking_and_on_all_paths():
    # SKIP side: validation is at entry, before the SKIP early return.
    _raises(lambda: ak.sweep_order_book(
        _book(), ak.TradeAction.SKIP, "1000", 0.8, 0.05),
        "INVALID_TARGET_BUDGET_TYPE", "SF-BL-11045")
    # Keyword-argument call path (naive baseline / budget-sensitivity style).
    _raises(lambda: ak.sweep_order_book(
        book=_book(), side=ak.TradeAction.BUY_YES, target_budget=True,
        belief_prob_side=0.8, min_net_edge=-1.0),
        "INVALID_TARGET_BUDGET_TYPE", "SF-BL-11045")
    # LCB path also validates the budget type first.
    _raises(lambda: ak.sweep_order_book(
        _book(), ak.TradeAction.BUY_YES, None, 0.8, 0.05,
        posterior_n_eff=30, lcb_z=1.0),
        "INVALID_TARGET_BUDGET_TYPE", "SF-BL-11045")
    # Budget type is checked before belief/hurdle validation (the
    # budget guard precedes it, as the SF-BL-9438 return always did).
    _raises(lambda: ak.sweep_order_book(
        _book(), ak.TradeAction.BUY_YES, "1000", "0.8", None),
        "INVALID_TARGET_BUDGET_TYPE", "SF-BL-11045")


def test_belief_and_hurdle_type_classes_still_raise():
    for bad in (True, False, None, "0.8"):
        _raises(lambda b=bad: ak.sweep_order_book(
            _book(), ak.TradeAction.BUY_YES, 1000.0, b, 0.05),
            "INVALID_BELIEF_PROB_SIDE", "SF-BL-10502")
    for bad in (True, False, None, "0.05"):
        _raises(lambda h=bad: ak.sweep_order_book(
            _book(), ak.TradeAction.BUY_YES, 1000.0, 0.8, h),
            "INVALID_MIN_NET_EDGE", "SF-BL-10502")


def test_helper_predicate():
    for good in (0, 1, 1000, 1000.0, 0.5, -5.0, NAN, INF):
        assert ak._target_budget_is_numeric_type(good) is True, good
    for bad in (True, False, None, "1000", [], {}):
        assert ak._target_budget_is_numeric_type(bad) is False, bad


# ---- 3. Numeric budgets: bit-identical, incl. SF-BL-9438 contract -----------

def test_clean_control_bit_identical():
    assert ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.80, 0.05) == CLEAN
    assert ak.sweep_order_book(_book(), ak.TradeAction.BUY_NO, 1000.0, 0.80, 0.05) == \
        (1000.0, 1904.036557501904, 0.5252)


def test_int_budget_accepted_bit_identical():
    assert ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000, 0.80, 0.05) == CLEAN


def test_nonfinite_numeric_budget_still_returns_zero():
    # SF-BL-9438 pinned contract: numeric non-finite is a VALUE case.
    for bad in (NAN, INF, -INF, 10 ** 400):
        assert ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, bad, 0.80, 0.05) == ZERO, bad


def test_zero_and_negative_budget_still_return_zero():
    assert ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 0.0, 0.80, 0.05) == ZERO
    assert ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 0, 0.80, 0.05) == ZERO
    assert ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, -250.0, 0.80, 0.05) == ZERO


def test_genuine_no_fill_still_returns_zero_not_raise():
    # Depth exhaustion and a genuine no-edge stop keep the tuple —
    # now distinguishable from a type error, which raises.
    empty = ak.OrderBookSnapshot(
        market_id="FIXTURE-11045-E", title="exhausted",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C11045-E",
        est_days_to_resolution=5,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=0)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100_000)],
        fee_rate=0.01)
    assert ak.sweep_order_book(empty, ak.TradeAction.BUY_YES, 1000.0, 0.80, 0.05) == ZERO
    assert ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.40, 0.05) == ZERO
    assert ak.sweep_order_book(_book(), ak.TradeAction.SKIP, 1000.0, 0.80, 0.05) == ZERO


def test_finite_grid_bit_identical():
    def reference(book, budget, belief, hurdle):
        remaining, cost, shares = budget, 0.0, 0.0
        for lvl in book.yes_asks:
            if remaining <= 1e-6:
                break
            eff = lvl.price * (1.0 + book.fee_rate)
            if eff >= 1.0:
                break
            if (belief - eff) < hurdle:
                break
            spend = min(remaining, lvl.size_shares * eff)
            cost += spend
            shares += spend / eff
            remaining -= spend
        return (cost, shares, cost / shares) if shares > 0 else ZERO

    for budget in (0.0, 100.0, 505.0, 1000.0, 5000.0):
        for belief in (0.40, 0.55, 0.65, 0.80, 0.99):
            for hurdle in (-1.0, 0.0, 0.05, 0.30):
                got = ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES,
                                          budget, belief, hurdle)
                want = reference(_book(), budget, belief, hurdle)
                assert got == want, (budget, belief, hurdle, got, want)


# ---- 4. Precedence with existing guards --------------------------------------

def test_invalid_book_still_raises_first():
    _raises(lambda: ak.sweep_order_book(None, ak.TradeAction.BUY_YES, "1000", 0.8, 0.05),
            "INVALID_BOOK", "SF-BL-10964")


def test_invalid_fee_still_raises_before_budget_type():
    b = _book()
    b.fee_rate = -0.20
    try:
        ak.sweep_order_book(b, ak.TradeAction.BUY_YES, "1000", 0.8, 0.05)
    except ValueError as e:
        assert "fee_rate" in str(e) and "INVALID_TARGET_BUDGET_TYPE" not in str(e), str(e)
        return
    raise AssertionError("invalid fee did not raise")


# ---- 5. No regression: evaluate path + benchmark slate ------------------------

def test_evaluate_trade_control_unchanged():
    b = ak.OrderBookSnapshot(
        market_id="FIXTURE", title="control",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C11045-CTL",
        est_days_to_resolution=5,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100_000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100_000)],
        fee_rate=0.0)
    used = {}
    r = ak.evaluate_and_size_opportunity(b, 0.80, 0.0, 10000.0, used)
    assert r.action == ak.TradeAction.BUY_YES, r
    assert r.invested_capital == 1500.0 and r.shares_acquired == 3000.0, r
    assert used == {"C11045-CTL": 0.15}, used


def test_benchmark_fixture_unchanged():
    buf = io.StringIO()
    with redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    assert '"north_star_score": -0.1611' in out, out[-2000:]
    assert '"north_star_score": -0.1499' in out, out[-2000:]


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
