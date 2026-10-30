"""
test_sweep_nan_edge_stop.py — SF-BL-9471 verification tests.

Pre-registered done/kill criteria (TASK.md):
  "Pass if sweep raises or returns (0,0,0) on non-finite belief_prob_side
   or min_net_edge and a finite-belief sweep still stops at the first
   rung breaching the hurdle on the benchmark books; kill if every
   in-repo sweep caller is shown to sanitise both arguments (call-site
   census)."

Kill condition is NOT met (see CALLSITE_CENSUS.md): neither production
caller sanitises either argument — evaluate_and_size_opportunity passes
p_side / config.min_net_edge through unchecked (FleetRiskConfig has no
finiteness invariant), and the naive baseline in
run_benchmark_simulation calls the sweep directly with a bare
min_net_edge=-1.0 override. So the fix applies at the consumption
point: sweep_order_book returns (0.0, 0.0, 0.0) when belief_prob_side
or min_net_edge is non-finite (NaN, +-inf, or non-numeric/bool, via the
existing _is_finite_number helper) — the same fail-closed convention
SF-BL-9438 established in this function for a non-finite target_budget.

Pre-fix reproduction (verified live 2026-10-21, before the patch;
see repro.py in this directory). Fixture book, fee 0.0:
  asks 0.50/0.60/0.70/0.80 x 800, bids 0.48 x 10000, budget $1,000:
    belief 0.65 hurdle 0.05  -> (880.0, 1600.0, 0.55)   [stops at 0.70]
    belief NaN  hurdle 0.05  -> (1000.0, 1771.43, 0.5645)  [FAIL OPEN]
    belief +inf hurdle 0.05  -> (1000.0, 1771.43, 0.5645)  [FAIL OPEN]
    belief 0.65 hurdle NaN   -> (1000.0, 1771.43, 0.5645)  [FAIL OPEN]
    belief 0.65 hurdle -inf  -> (1000.0, 1771.43, 0.5645)  [FAIL OPEN]
    BUY_NO belief NaN        -> (1000.0, 1923.08, 0.52)    [FAIL OPEN]
  Every fail-open result is byte-identical to the intentional
  hurdle=-1.0 naive override: the stop was not weakened, it was gone.
  (The TASK title fixture — belief=NaN, $1,000, 1,629 shares at
  q_exec 0.6138 — is the same mechanism on the finder's book.)

Run:  python3 test_sweep_nan_edge_stop.py   (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_fee_rate.py.
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


def _book(fee=0.0, cluster="C9471"):
    return ak.OrderBookSnapshot(
        market_id="FIXTURE", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster=cluster,
        est_days_to_resolution=5,
        yes_asks=[ak.OrderBookLevel(price=p, size_shares=800)
                  for p in (0.50, 0.60, 0.70, 0.80)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=10_000)],
        fee_rate=fee)


def _sweep(book, side, budget, belief, *hurdle):
    return ak.sweep_order_book(book, side, budget, belief, *hurdle)


# ---- 1. Non-finite belief fails closed --------------------------------------

def test_nonfinite_belief_returns_zero_both_sides():
    for bad in (NAN, INF, -INF):
        for side in (ak.TradeAction.BUY_YES, ak.TradeAction.BUY_NO):
            out = _sweep(_book(), side, 1000.0, bad, 0.05)
            assert out == ZERO, (bad, side, out)


def test_nonfinite_belief_default_hurdle_path():
    # 4-arg call (min_net_edge defaults to 0.05), as in the fee-rate tests.
    assert _sweep(_book(), ak.TradeAction.BUY_YES, 1000.0, NAN) == ZERO
    assert _sweep(_book(), ak.TradeAction.BUY_YES, 1000.0, INF) == ZERO


def test_nonfinite_belief_never_returns_nonnumeric_tuple():
    # Pre-fix the tuple itself was finite here but represented a full
    # $1,000 sweep through rungs above the belief; the invariant that
    # matters: no shares, no spend, no price — exactly the no-fill tuple.
    out = _sweep(_book(), ak.TradeAction.BUY_YES, 1000.0, NAN, 0.05)
    assert out == ZERO and all(v == 0.0 for v in out), out


# ---- 2. Non-finite hurdle fails closed --------------------------------------

def test_nonfinite_hurdle_returns_zero_both_sides():
    for bad in (NAN, INF, -INF):
        for side in (ak.TradeAction.BUY_YES, ak.TradeAction.BUY_NO):
            out = _sweep(_book(), side, 1000.0, 0.65, bad)
            assert out == ZERO, (bad, side, out)


def test_both_nonfinite_returns_zero():
    assert _sweep(_book(), ak.TradeAction.BUY_YES, 1000.0, NAN, NAN) == ZERO
    assert _sweep(_book(), ak.TradeAction.BUY_YES, 1000.0, INF, -INF) == ZERO
    assert _sweep(_book(), ak.TradeAction.BUY_NO, 1000.0, -INF, NAN) == ZERO


def test_nonnumeric_and_bool_belief_or_hurdle_fail_closed():
    # Cannot be ordered against a rung price at all; must fail closed,
    # never raise TypeError into a direct caller.
    for bad in (None, "0.65", True, False):
        assert _sweep(_book(), ak.TradeAction.BUY_YES, 1000.0, bad, 0.05) == ZERO, bad
        assert _sweep(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.65, bad) == ZERO, bad


# ---- 3. Finite sweeps still stop at the first breaching rung -----------------

def test_prereg_fixture_stops_at_first_breach():
    # belief 0.65, hurdle 0.05: 0.50 rung edge 0.15 fills, 0.60 rung edge
    # exactly 0.05 fills (stop is strictly <), 0.70 rung edge -0.05 stops.
    out = _sweep(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.65, 0.05)
    assert out == (880.0, 1600.0, 0.55), out


def test_finite_stop_with_fees_uses_effective_price():
    # fee 0.01: eff prices 0.505 / 0.606 / 0.707; belief 0.65 hurdle 0.05
    # fills only the 0.50 rung (0.606 breaches: edge 0.044 < 0.05).
    inv, sh, q = _sweep(_book(fee=0.01), ak.TradeAction.BUY_YES, 1000.0, 0.65, 0.05)
    assert sh == 800.0 and abs(inv - 404.0) < 1e-9 and abs(q - 0.505) < 1e-9, (inv, sh, q)


def test_finite_buy_no_stop():
    # BUY_NO ladder is the single live bid: NO price 1 - 0.48 = 0.52.
    # belief 0.65 hurdle 0.05: edge 0.13 fills up to budget.
    out = _sweep(_book(), ak.TradeAction.BUY_NO, 520.0, 0.65, 0.05)
    assert out == (520.0, 1000.0, 0.52), out
    # belief 0.55: edge 0.03 < 0.05 -> first rung already breaches.
    assert _sweep(_book(), ak.TradeAction.BUY_NO, 520.0, 0.55, 0.05) == ZERO


def test_first_rung_breach_returns_zero_finite():
    # belief 0.54 vs first eff rung 0.50: edge 0.04 < 0.05 -> no fill.
    assert _sweep(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.54, 0.05) == ZERO


def test_finite_grid_matches_reference_stop_semantics():
    # Independent recomputation of the documented stop semantics over a
    # finite grid: the guard must not have altered ANY finite behaviour.
    def reference(book, budget, belief, hurdle):
        remaining, cost, shares = budget, 0.0, 0.0
        for lvl in book.yes_asks:
            if remaining <= 1e-6:
                break
            eff = lvl.price * (1.0 + book.fee_rate)
            if (belief - eff) < hurdle:
                break
            spend = min(remaining, lvl.size_shares * eff)
            cost += spend
            shares += spend / eff
            remaining -= spend
        return (cost, shares, cost / shares) if shares > 0 else ZERO

    for fee in (0.0, 0.01):
        for belief in (0.30, 0.54, 0.55, 0.65, 0.75, 0.90, 0.99):
            for hurdle in (-1.0, -0.10, 0.0, 0.05, 0.10, 0.30):
                got = _sweep(_book(fee=fee), ak.TradeAction.BUY_YES, 1000.0, belief, hurdle)
                want = reference(_book(fee=fee), 1000.0, belief, hurdle)
                assert got == want, (fee, belief, hurdle, got, want)


# ---- 4. Finite overrides / types unchanged -----------------------------------

def test_naive_minus_one_override_still_sweeps_deep():
    # The naive baseline's finite -1.0 override is intentional behaviour
    # (benchmark slate depends on it); only NON-finite hurdles close.
    out = _sweep(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.65, -1.0)
    assert out[0] == 1000.0 and abs(out[1] - 1771.4285714285716) < 1e-9, out
    assert abs(out[2] - 0.564516129032258) < 1e-12, out


def test_int_belief_and_hurdle_accepted():
    out = _sweep(_book(), ak.TradeAction.BUY_YES, 1000.0, 1, 0)
    assert out[0] == 1000.0 and out[1] > 0.0, out


def test_zero_hurdle_and_skip_side_unchanged():
    out = _sweep(_book(), ak.TradeAction.BUY_YES, 400.0, 0.65, 0.0)
    assert out == (400.0, 800.0, 0.50), out
    assert _sweep(_book(), ak.TradeAction.SKIP, 1000.0, 0.65, 0.05) == ZERO
    assert _sweep(_book(), ak.TradeAction.SKIP, 1000.0, NAN, NAN) == ZERO


# ---- 5. Interaction with existing guards -------------------------------------

def test_invalid_fee_still_raises_even_with_nan_belief():
    # Fee validation (SF-BL-9363) precedes the belief/hurdle guard:
    # a mutated out-of-bound fee raises exactly as before.
    b = _book()
    b.fee_rate = -0.20
    try:
        ak.sweep_order_book(b, ak.TradeAction.BUY_YES, 1000.0, NAN, NAN)
    except ValueError:
        return
    raise AssertionError("invalid fee did not raise")


def test_nonfinite_budget_guard_unchanged():
    # SF-BL-9438 guard still fires (and composes with finite belief).
    assert _sweep(_book(), ak.TradeAction.BUY_YES, NAN, 0.80, 0.05) == ZERO
    assert _sweep(_book(), ak.TradeAction.BUY_YES, INF, 0.80, 0.05) == ZERO


# ---- 6. No regression: evaluate path + benchmark slate ------------------------

def test_evaluate_trade_control_unchanged():
    b = ak.OrderBookSnapshot(
        market_id="FIXTURE", title="control",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C9471-CTL",
        est_days_to_resolution=5,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100_000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100_000)],
        fee_rate=0.0)
    used = {}
    r = ak.evaluate_and_size_opportunity(b, 0.80, 0.0, 10000.0, used)
    assert r.action == ak.TradeAction.BUY_YES, r
    assert r.invested_capital == 1500.0 and r.shares_acquired == 3000.0, r
    assert r.allocated_fraction == 0.15 and r.exec_price == 0.50, r
    assert used == {"C9471-CTL": 0.15}, used


def test_benchmark_fixture_unchanged():
    buf = io.StringIO()
    with redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    assert '"north_star_score": 0.1059' in out, out[-2000:]
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
