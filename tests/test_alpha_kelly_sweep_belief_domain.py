"""SF-BL-10502 regression tests: sweep_order_book belief/hurdle domain.

Pre-registered done/kill criteria (TASK.md):
  "Fix passes if sweep_order_book raises a named error on a non-finite
   or out-of-[0,1] belief_prob_side (and min_net_edge) before walking
   and fills on valid beliefs are bit-identical; kill if a percent-unit
   belief still invests any budget."

Pre-fix reproduction (verified live, repro.py in the task work dir):
single 0.90 ask, fee 0, budget $100 — belief 82.0 swept the full budget
(100.0, 111.111..., 0.90) while clean belief 0.6 stopped at (0, 0, 0).

Fix (alpha_kelly.py, SF-BL-10502): sweep_order_book validates both stop
operands at entry, before walking, via _validate_sweep_belief_and_edge:
belief_prob_side must be a finite real in [0, 1] (else ValueError
INVALID_BELIEF_PROB_SIDE) and min_net_edge a finite real in [-1.0, 1.0]
(else ValueError INVALID_MIN_NET_EDGE; the documented -1.0 full-walk
override is the lower bound, so every in-repo finite hurdle — -1.0,
-0.10, 0.0, 0.05, 0.30 — stays in domain and bit-identical). This
supersedes SF-BL-9471's return-(0, 0, 0) convention for non-finite
belief/hurdle operands: they now raise.

Run:  python3 test_sweep_belief_domain.py   (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_sweep_nan_edge_stop.py.
"""
import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "superforecaster"))
# Fallback when run from the backlog work dir (not the repo tests/ dir):
if not os.path.exists(os.path.join(sys.path[0], "alpha_kelly.py")):
    sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster/superforecaster"))

import alpha_kelly as ak  # noqa: E402

PASS = []
NAN = float("nan")
INF = float("inf")
ZERO = (0.0, 0.0, 0.0)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(fee=0.0, cluster="C10502"):
    return ak.OrderBookSnapshot(
        market_id="FIXTURE", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster=cluster,
        est_days_to_resolution=5,
        yes_asks=[ak.OrderBookLevel(price=p, size_shares=800)
                  for p in (0.50, 0.60, 0.70, 0.80)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=10_000)],
        fee_rate=fee)


def _title_book():
    # TASK title fixture: single 0.90 ask, fee_rate=0.
    return ak.OrderBookSnapshot(
        market_id="FIXTURE-10502", title="single 0.90 ask",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C10502-T",
        est_days_to_resolution=5,
        yes_asks=[ak.OrderBookLevel(price=0.90, size_shares=100_000)],
        yes_bids=[ak.OrderBookLevel(price=0.88, size_shares=100_000)],
        fee_rate=0.0)


def _raises(fn, token):
    try:
        fn()
    except ValueError as e:
        assert token in str(e), (token, str(e))
        assert "SF-BL-10502" in str(e), str(e)
        return
    raise AssertionError(f"no ValueError raised (expected {token})")


# ---- 1. Title fixture: percent-unit belief raises, clean belief stops -------

def test_title_fixture_percent_belief_raises():
    _raises(lambda: ak.sweep_order_book(
        _title_book(), ak.TradeAction.BUY_YES, 100.0, 82.0, 0.05),
        "INVALID_BELIEF_PROB_SIDE")


def test_title_fixture_clean_belief_still_stops():
    out = ak.sweep_order_book(_title_book(), ak.TradeAction.BUY_YES, 100.0, 0.6, 0.05)
    assert out == ZERO, out


def test_title_fixture_valid_high_belief_still_fills():
    out = ak.sweep_order_book(_title_book(), ak.TradeAction.BUY_YES, 100.0, 0.98, 0.05)
    assert out == (100.0, 111.11111111111111, 0.9), out


# ---- 2. Out-of-domain belief classes raise the named error ------------------

def test_belief_out_of_range_raises_both_sides():
    for bad in (82.0, 1.5, 1.0000001, 2.0, -0.2, -1.0, -82.0):
        for side in (ak.TradeAction.BUY_YES, ak.TradeAction.BUY_NO):
            _raises(lambda b=bad, s=side: ak.sweep_order_book(
                _book(), s, 1000.0, b, 0.05), "INVALID_BELIEF_PROB_SIDE")


def test_belief_nonfinite_raises():
    for bad in (NAN, INF, -INF):
        _raises(lambda b=bad: ak.sweep_order_book(
            _book(), ak.TradeAction.BUY_YES, 1000.0, b, 0.05),
            "INVALID_BELIEF_PROB_SIDE")
        # default-hurdle 4-arg call path too
        _raises(lambda b=bad: ak.sweep_order_book(
            _book(), ak.TradeAction.BUY_YES, 1000.0, b),
            "INVALID_BELIEF_PROB_SIDE")


def test_belief_bool_and_nonnumeric_raise_valueerror_not_typeerror():
    for bad in (True, False, None, "0.82", "82"):
        _raises(lambda b=bad: ak.sweep_order_book(
            _book(), ak.TradeAction.BUY_YES, 1000.0, b, 0.05),
            "INVALID_BELIEF_PROB_SIDE")


def test_belief_huge_int_raises():
    _raises(lambda: ak.sweep_order_book(
        _book(), ak.TradeAction.BUY_YES, 1000.0, 10 ** 400, 0.05),
        "INVALID_BELIEF_PROB_SIDE")


# ---- 3. min_net_edge classes -------------------------------------------------

def test_hurdle_nonfinite_raises():
    for bad in (NAN, INF, -INF):
        _raises(lambda h=bad: ak.sweep_order_book(
            _book(), ak.TradeAction.BUY_YES, 1000.0, 0.65, h),
            "INVALID_MIN_NET_EDGE")


def test_hurdle_out_of_range_raises():
    for bad in (82.0, 1.5, 1.0000001, -1.5, -2.0, -82.0):
        _raises(lambda h=bad: ak.sweep_order_book(
            _book(), ak.TradeAction.BUY_YES, 1000.0, 0.65, h),
            "INVALID_MIN_NET_EDGE")


def test_hurdle_bool_and_nonnumeric_raise():
    for bad in (True, None, "0.05"):
        _raises(lambda h=bad: ak.sweep_order_book(
            _book(), ak.TradeAction.BUY_YES, 1000.0, 0.65, h),
            "INVALID_MIN_NET_EDGE")


def test_belief_validated_before_hurdle():
    # Both bad: the belief error wins (documented validation order).
    _raises(lambda: ak.sweep_order_book(
        _book(), ak.TradeAction.BUY_YES, 1000.0, 82.0, NAN),
        "INVALID_BELIEF_PROB_SIDE")


def test_raises_before_walking():
    # Validation is at entry: it fires even when no rung would be walked
    # (SKIP side, zero budget on a finite budget value).
    _raises(lambda: ak.sweep_order_book(
        _book(), ak.TradeAction.SKIP, 1000.0, 82.0, 0.05),
        "INVALID_BELIEF_PROB_SIDE")
    _raises(lambda: ak.sweep_order_book(
        _book(), ak.TradeAction.BUY_YES, 0.0, 82.0, 0.05),
        "INVALID_BELIEF_PROB_SIDE")


# ---- 4. Valid beliefs / hurdles are bit-identical ----------------------------

def test_prereg_fixture_bit_identical():
    out = ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.65, 0.05)
    assert out == (880.0, 1600.0, 0.55), out


def test_boundary_beliefs_accepted():
    # 0 and 1 are in [0, 1]: 0 stops at the first rung, 1 sweeps deep.
    assert ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.0, 0.05) == ZERO
    out = ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000.0, 1.0, 0.05)
    assert out[0] == 1000.0 and out[1] > 0.0, out
    out = ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000.0, 1, 0)
    assert out[0] == 1000.0 and out[1] > 0.0, out  # int belief/hurdle


def test_documented_hurdle_range_bit_identical():
    # -1.0 full-walk override (naive baseline / budget sensitivity):
    out = ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.65, -1.0)
    assert out[0] == 1000.0 and abs(out[1] - 1771.4285714285716) < 1e-9, out
    assert abs(out[2] - 0.564516129032258) < 1e-12, out
    # -0.10 grid hurdle and the 1.0 upper bound behave as plain hurdles:
    out = ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.65, -0.10)
    assert out[0] == 1000.0, out
    assert ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.99, 1.0) == ZERO


def test_finite_grid_matches_reference_stop_semantics():
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

    for fee in (0.0, 0.01):
        for belief in (0.0, 0.30, 0.54, 0.55, 0.65, 0.75, 0.90, 0.99, 1.0):
            for hurdle in (-1.0, -0.10, 0.0, 0.05, 0.10, 0.30, 1.0):
                got = ak.sweep_order_book(_book(fee=fee), ak.TradeAction.BUY_YES,
                                          1000.0, belief, hurdle)
                want = reference(_book(fee=fee), 1000.0, belief, hurdle)
                assert got == want, (fee, belief, hurdle, got, want)


def test_buy_no_valid_belief_bit_identical():
    out = ak.sweep_order_book(_book(), ak.TradeAction.BUY_NO, 520.0, 0.65, 0.05)
    assert out == (520.0, 1000.0, 0.52), out
    assert ak.sweep_order_book(_book(), ak.TradeAction.BUY_NO, 520.0, 0.55, 0.05) == ZERO


def test_lcb_path_valid_and_invalid_belief():
    # Opt-in LCB stop with a valid belief is unchanged in shape...
    out = ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.65, 0.05,
                              posterior_n_eff=30, lcb_z=1.0)
    assert out == ZERO or out[1] > 0.0, out
    # ...and an invalid belief raises before the LCB stop is resolved.
    _raises(lambda: ak.sweep_order_book(
        _book(), ak.TradeAction.BUY_YES, 1000.0, 82.0, 0.05,
        posterior_n_eff=30, lcb_z=1.0), "INVALID_BELIEF_PROB_SIDE")


# ---- 5. Interaction with existing guards -------------------------------------

def test_invalid_fee_still_raises_first():
    b = _book()
    b.fee_rate = -0.20
    try:
        ak.sweep_order_book(b, ak.TradeAction.BUY_YES, 1000.0, 82.0, NAN)
    except ValueError as e:
        assert "fee_rate" in str(e), str(e)
        return
    raise AssertionError("invalid fee did not raise")


def test_nonfinite_budget_guard_unchanged():
    # SF-BL-9438: non-finite budget with a VALID belief still returns
    # the no-fill tuple (its guard precedes the new validation).
    assert ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, NAN, 0.80, 0.05) == ZERO
    assert ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, INF, 0.80, 0.05) == ZERO


# ---- 6. No regression: evaluate path + benchmark slate ------------------------

def test_evaluate_trade_control_unchanged():
    b = ak.OrderBookSnapshot(
        market_id="FIXTURE", title="control",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C10502-CTL",
        est_days_to_resolution=5,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100_000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100_000)],
        fee_rate=0.0)
    used = {}
    r = ak.evaluate_and_size_opportunity(b, 0.80, 0.0, 10000.0, used)
    assert r.action == ak.TradeAction.BUY_YES, r
    assert r.invested_capital == 1500.0 and r.shares_acquired == 3000.0, r
    assert r.allocated_fraction == 0.15 and r.exec_price == 0.50, r
    assert used == {"C10502-CTL": 0.15}, used


def test_benchmark_fixture_unchanged():
    buf = io.StringIO()
    with redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    # Slate as of this run's tree (disciplined -0.1611 / naive -0.1499;
    # verified bit-identical between the pristine pre-patch module and
    # the patched module — see DONE-CLAIM.md differential evidence).
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
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
