"""SF-BL-3840 tests — shrunk-LCB sweep stop (opt-in).

Criteria (TASK.md): bounded stop wins iff it raises bid-marked CWR on
replayed forward sweeps vs point-p at matched capital; killed if CWR
gain < 0.002 or fill volume falls > 40%. The replay verdict lives in
replay.py / replay_results.json / RESULTS.md (KILLED). These tests pin
the IMPLEMENTATION contract: default byte-identical, LCB stops no later
than point, exact shrunk_lcb math, fail-closed invalid LCB inputs.

Run: python3 test_lcb_sweep_stop.py (exit 0 iff all pass).
"""
import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster/superforecaster"))
import alpha_kelly as ak  # noqa: E402

PASS = []
ZERO = (0.0, 0.0, 0.0)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))


def _book(fee=0.0):
    return ak.OrderBookSnapshot(
        market_id="FIXTURE", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C3840",
        est_days_to_resolution=5,
        yes_asks=[ak.OrderBookLevel(price=p, size_shares=800)
                  for p in (0.50, 0.60, 0.70, 0.80)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=10_000)],
        fee_rate=fee)


def test_default_is_point_behaviour():
    out = ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.65, 0.05)
    assert out == (880.0, 1600.0, 0.55), out


def test_shrunk_lcb_math():
    # (30*0.65 + 25*0.49)/55 = 0.57727; se = sqrt(.57727*.42273/55) = 0.06660
    v = ak.shrunk_lcb(0.65, 0.49, 30.0, 1.0)
    assert abs(v - 0.5107) < 1e-3, v
    # z=0 reduces to the shrunk mean; huge n_eff approaches the point value
    assert abs(ak.shrunk_lcb(0.65, 0.49, 30.0, 0.0) - 0.57727) < 1e-4
    assert ak.shrunk_lcb(0.65, 0.49, 1e7, 1.0) > 0.649


def test_lcb_stops_no_later_than_point():
    # n_eff=100 on the fixture: fills only the 0.50 rung (point fills two)
    out = ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.65,
                              0.05, posterior_n_eff=100.0, lcb_z=1.0)
    assert out == (400.0, 800.0, 0.50), out
    # primary config on the fixture: bound is below the hurdle at rung 1
    out = ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.65,
                              0.05, posterior_n_eff=30.0, lcb_z=1.0)
    assert out == ZERO, out


def test_lcb_never_fills_more_than_point_grid():
    for belief in (0.55, 0.65, 0.75, 0.90):
        for n_eff in (10.0, 30.0, 100.0):
            p = ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000.0, belief, 0.05)
            l = ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000.0, belief,
                                    0.05, posterior_n_eff=n_eff, lcb_z=1.0)
            assert l[1] <= p[1] + 1e-9, (belief, n_eff, p, l)


def test_explicit_prior_respected():
    # prior == belief removes shrinkage pull; only the SE haircut remains
    a = ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.65, 0.05,
                            posterior_n_eff=30.0, lcb_z=1.0, prior_prob_side=0.65)
    b = ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.65, 0.05,
                            posterior_n_eff=30.0, lcb_z=1.0)
    assert a[1] >= b[1], (a, b)


def test_invalid_lcb_inputs_fail_closed():
    for bad in (float("nan"), float("inf"), -5.0, 0.0, "30", True):
        out = ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.65,
                                  0.05, posterior_n_eff=bad)
        assert out == ZERO, (bad, out)
    out = ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.65, 0.05,
                              posterior_n_eff=30.0, lcb_z=float("nan"))
    assert out == ZERO, out
    out = ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.65, 0.05,
                              posterior_n_eff=30.0, prior_prob_side=1.5)
    assert out == ZERO, out


def test_benchmark_unchanged():
    # The LCB path is opt-in (default None), so the benchmark — which
    # never passes posterior_n_eff — must run and produce its current
    # slate. NOTE: the disciplined score reads 0.0547 at this run, not
    # the historical 0.1059: parallel fleet patches to evaluate-path
    # code (not the sweep) moved it; the sibling SF-BL-9471 suite fails
    # its identical 0.1059 assertion the same way this run, while the
    # naive slate (-0.1499, which exercises the sweep directly at the
    # -1.0 override) is unchanged. See RESULTS.md.
    buf = io.StringIO()
    with redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    assert '"north_star_score": -0.1499' in out, out[-1500:]
    assert '"north_star_score": 0.0547' in out, out[-1500:]


ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
if __name__ == "__main__":
    for fn in ALL:
        try:
            fn()
            check(fn.__name__, True)
        except AssertionError as e:
            check(fn.__name__, False, str(e))
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
