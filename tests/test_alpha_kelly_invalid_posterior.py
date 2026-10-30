"""
test_invalid_posterior.py — SF-BL-9470 verification tests.

Prereg done criterion: "Pass if posteriors in {-0.20, 1.50, inf, -inf}
SKIP with an explicit INVALID_POSTERIOR reason and zero size while the
8-market benchmark fixture still reproduces N_star +0.1059 exactly;
kill if a caller census proves aggregate_fleet_forecasts is the sole
producer and its [0.02,0.98] clip is total (including mutated result
objects)."

Kill NOT met — see CALLER_CENSUS.md in the task work dir: aggregate is
only the benchmark producer; evaluate takes a plain float, direct
in-repo callers pass bare literals, and TradeExecutionResult is a
mutable dataclass constructed outside the aggregate path.

Pre-fix reproduction (verified live 2026-10-21, before the patch; repro.py):
  fixture book (ask 0.50 x 100k, bid 0.48, fee 0.0, equity 10000):
    p=0.80  -> BUY_YES invested 1500.0 shares 3000.0 (control)
    p=1.50  -> BUY_YES invested 1500.0 shares 3000.0 (cluster cap)
    p=-0.20 -> BUY_NO  invested 1500.0 shares 2884.62
    p=+inf  -> BUY_YES invested 1500.0
    p=-inf  -> BUY_NO  invested 1500.0
    p=nan   -> SKIP INSUFFICIENT_NET_EDGE (incidental, SF-BL-9440)

Run:  python3 test_invalid_posterior.py   (exit 0 iff all pass)
No pytest dependency, same convention as the other alpha_kelly tests.
"""
import io
import contextlib
import math
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "superforecaster"))
if not os.path.exists(os.path.join(sys.path[0], "alpha_kelly.py")):
    sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster/superforecaster"))

import alpha_kelly as ak  # noqa: E402

PASS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(cluster="C9470"):
    return ak.OrderBookSnapshot(
        market_id="FIXTURE", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster=cluster,
        est_days_to_resolution=5,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100_000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100_000)],
        fee_rate=0.0)


def _assert_invalid_posterior_skip(r, used_before, used_after):
    assert r.action == ak.TradeAction.SKIP, r
    assert r.skip_reason is not None and r.skip_reason.startswith("INVALID_POSTERIOR"), r.skip_reason
    assert r.invested_capital == 0.0 and r.shares_acquired == 0.0 and r.exec_price == 0.0, r
    assert r.kelly_fraction_raw == 0.0 and r.allocated_fraction == 0.0, r  # EC-09
    assert r.liquidation_bid_price == 0.0 and r.kl_divergence_nats == 0.0 and r.expected_log_growth == 0.0, r
    assert used_after == used_before, (used_before, used_after)  # cluster untouched


# ---- 1. Prereg fixtures ------------------------------------------------------

def test_prereg_posteriors_skip_invalid_posterior():
    for bad in (-0.20, 1.50, float("inf"), float("-inf")):
        used = {}
        r = ak.evaluate_and_size_opportunity(_book(), bad, 0.0, 10000, used)
        _assert_invalid_posterior_skip(r, {}, used)


def test_nan_posterior_skips_invalid_posterior_not_incidental_edge():
    # SF-BL-9440's NaN case: pre-fix it SKIPped only incidentally as
    # INSUFFICIENT_NET_EDGE; post-fix the reason is explicitly invalid input.
    used = {}
    r = ak.evaluate_and_size_opportunity(_book(), float("nan"), 0.0, 10000, used)
    _assert_invalid_posterior_skip(r, {}, used)


def test_just_outside_bounds_skips():
    for bad in (-1e-9, 1.0000001, -1.0, 2.0):
        r = ak.evaluate_and_size_opportunity(_book(), bad, 0.0, 10000, {})
        assert r.skip_reason.startswith("INVALID_POSTERIOR"), (bad, r.skip_reason)


def test_non_numeric_posterior_skips_without_raising():
    for bad in ("0.80", None, True, False):
        used = {}
        r = ak.evaluate_and_size_opportunity(_book(), bad, 0.0, 10000, used)
        _assert_invalid_posterior_skip(r, {}, used)


def test_invalid_posterior_never_poisons_cluster():
    used = {}
    ak.evaluate_and_size_opportunity(_book(), 1.50, 0.0, 10000, used)
    assert used == {}, used
    r = ak.evaluate_and_size_opportunity(_book(), 0.80, 0.0, 10000, used)
    assert r.action == ak.TradeAction.BUY_YES and r.invested_capital == 1500.0, r
    assert used == {"C9470": 0.15}, used


def test_posterior_gate_is_first():
    # Invalid posterior wins even over an unverified crypto book, high
    # dispersion, and invalid equity — it is validated before all of them.
    b = _book()
    b.domain = ak.MarketDomain.CRYPTO
    b.on_chain_verified = False
    r = ak.evaluate_and_size_opportunity(b, 1.50, 0.99, float("nan"), {})
    assert r.skip_reason.startswith("INVALID_POSTERIOR"), r.skip_reason


# ---- 2. In-range behaviour unchanged -----------------------------------------

def test_trade_control_unchanged():
    used = {}
    r = ak.evaluate_and_size_opportunity(_book(), 0.80, 0.0, 10000, used)
    assert r.action == ak.TradeAction.BUY_YES, r
    assert r.invested_capital == 1500.0 and r.shares_acquired == 3000.0, r
    assert r.allocated_fraction == 0.15 and r.exec_price == 0.50, r
    assert used == {"C9470": 0.15}, used


def test_endpoints_and_aggregate_range_pass_the_gate():
    # 0.0 / 1.0 are in [0,1]: they keep their pre-fix behaviour (they trade
    # on this fixture); they must NOT report INVALID_POSTERIOR.
    for p, side in ((0.0, ak.TradeAction.BUY_NO), (1.0, ak.TradeAction.BUY_YES)):
        r = ak.evaluate_and_size_opportunity(_book(), p, 0.0, 10000, {})
        assert r.action == side, (p, r)
    # Aggregate-range values pass the gate (they may still skip for ordinary
    # reasons, but never INVALID_POSTERIOR).
    for p in (0.02, 0.98, 0.5):
        r = ak.evaluate_and_size_opportunity(_book(), p, 0.0, 10000, {})
        assert not (r.skip_reason or "").startswith("INVALID_POSTERIOR"), (p, r.skip_reason)
    # Int posterior is a valid real.
    r = ak.evaluate_and_size_opportunity(_book(), 1, 0.0, 10000, {})
    assert r.action == ak.TradeAction.BUY_YES, r


def test_helper_domain():
    assert ak._posterior_in_bounds(0.0) and ak._posterior_in_bounds(1.0)
    assert ak._posterior_in_bounds(0.5) and ak._posterior_in_bounds(1)
    for bad in (-0.20, 1.50, float("inf"), float("-inf"), float("nan"), "x", None, True):
        assert not ak._posterior_in_bounds(bad), bad


# ---- 3. Benchmark slate unchanged ---------------------------------------------

def test_benchmark_nstar_exact():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
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
