"""
test_invalid_forecast.py — SF-BL-9440 verification tests.

Prereg done/kill criterion (TASK.md):
  "Add an explicit non-finite-posterior skip (INVALID_FORECAST) before
   edge computation, fixtured at NaN and +/-inf posteriors with
   skip-reason counts on the benchmark slate unchanged; kill if
   aggregate_fleet_forecasts is proven to be the sole posterior source
   and made NaN-proof first (then assert the invariant at this
   function's entry as its regression test)."

Kill NOT met — see PRODUCER_AUDIT.md in this dir: aggregate IS now
NaN-proof for its own return (SF-BL-9439/9506 drop non-finite members;
all-non-finite panels raise), but it is NOT the sole posterior source
(evaluate takes a plain float; the only in-module production caller is
the benchmark; test suites and out-of-tree fleet scripts pass bare
literals). So the fix lives at evaluate entry.

Naming disposition (documented in DONE-CLAIM.md): SF-BL-9470, filed
later the same day and closed AGREE, covers the finite out-of-range
case and pins the primary skip token INVALID_POSTERIOR with startswith
tests for NaN/+inf/-inf as well. Gate 0.0a therefore keeps
INVALID_POSTERIOR as the primary token and names this task's class
INVALID_FORECAST inside the same reason — both tokens are present,
telemetry can count the INVALID_FORECAST class distinctly, and no
agreed test regresses. These tests assert BOTH tokens, the absence of
INSUFFICIENT_NET_EDGE, EC-09 zero sizes, cluster non-poisoning,
precedence over the edge screen and downstream gates, the distinguish-
ability of a genuinely edgeless finite market (which keeps its
INSUFFICIENT_NET_EDGE reason), and benchmark-slate invariance against
the baseline captured this run immediately before the patch
(disciplined slate: 3 BUY_YES; skips INSUFFICIENT_NET_EDGE x1,
FAILED_CRYPTOBENCH_ONCHAIN_GATE x1, ROUND_TRIP_GATE_D x1,
BUDGET_SENSITIVITY_VETO x1, UNRESOLVED_AGENT_DISPERSION x1;
N_star +0.0547 disciplined / -0.1499 naive on the current combined
file — the +0.1059 figure in older task files predates the Gate D /
budget-sensitivity additions and is not this task's baseline).

Run:  python3 test_invalid_forecast.py   (exit 0 iff all pass)
No pytest dependency, same convention as the other alpha_kelly tests.
"""
import io
import contextlib
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "superforecaster"))
if not os.path.exists(os.path.join(sys.path[0], "alpha_kelly.py")):
    sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster/superforecaster"))

import alpha_kelly as ak  # noqa: E402

PASS = []
NAN = float("nan")
INF = float("inf")


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(cluster="C9440", ask=0.50, bid=0.48, fee=0.0):
    return ak.OrderBookSnapshot(
        market_id="FIXTURE", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster=cluster,
        est_days_to_resolution=5,
        yes_asks=[ak.OrderBookLevel(price=ask, size_shares=100_000)],
        yes_bids=[ak.OrderBookLevel(price=bid, size_shares=100_000)],
        fee_rate=fee)


def _assert_invalid_forecast_skip(r, used_before, used_after):
    assert r.action == ak.TradeAction.SKIP, r
    reason = r.skip_reason or ""
    # This task's class token is named explicitly in the reason...
    assert "INVALID_FORECAST" in reason, reason
    assert "SF-BL-9440" in reason, reason
    # ...under SF-BL-9470's pinned primary token (agreed, startswith).
    assert reason.startswith("INVALID_POSTERIOR"), reason
    # Never the pre-fix incidental no-edge reason, never 'max_edge=nan'.
    assert "INSUFFICIENT_NET_EDGE" not in reason, reason
    assert "max_edge=nan" not in reason, reason
    # EC-09 zero sizes.
    assert r.invested_capital == 0.0 and r.shares_acquired == 0.0 and r.exec_price == 0.0, r
    assert r.kelly_fraction_raw == 0.0 and r.allocated_fraction == 0.0, r
    assert r.liquidation_bid_price == 0.0 and r.kl_divergence_nats == 0.0, r
    assert r.expected_log_growth == 0.0, r
    assert used_after == used_before, (used_before, used_after)


# ---- 1. Prereg fixtures: NaN and +/-inf -------------------------------------

def test_prereg_nonfinite_posteriors_skip_invalid_forecast():
    for bad in (NAN, INF, -INF):
        used = {"PRE": 0.05}
        r = ak.evaluate_and_size_opportunity(_book(), bad, 0.0, 10000, used)
        _assert_invalid_forecast_skip(r, {"PRE": 0.05}, used)


def test_non_numeric_posterior_is_invalid_forecast_without_raising():
    for bad in ("0.80", None, True, False):
        used = {}
        r = ak.evaluate_and_size_opportunity(_book(), bad, 0.0, 10000, used)
        _assert_invalid_forecast_skip(r, {}, used)


def test_helper_classifies_nonfinite_only():
    for bad in (NAN, INF, -INF, "x", None, True):
        assert ak._posterior_not_finite(bad), bad
    for good in (0.0, 0.5, 1.0, 1, -0.20, 1.50):  # finite: 9470's case, not 9440's
        assert not ak._posterior_not_finite(good), good


# ---- 2. Before edge computation / precedence ---------------------------------

def test_invalid_forecast_precedes_edge_screen_and_downstream_gates():
    # A NaN posterior on a book that WOULD trade at p=0.80, combined with
    # high dispersion and NaN equity: the invalid-forecast reason wins
    # over Gate 2 / equity / the edge screen — it is checked at entry.
    r = ak.evaluate_and_size_opportunity(_book(), NAN, 0.99, NAN, {})
    _assert_invalid_forecast_skip(r, {}, {})
    # Mutated invalid fee must not pre-empt it either (Gate 0.0a is
    # ordered before the fee gate).
    b = _book()
    b.fee_rate = -0.20
    r = ak.evaluate_and_size_opportunity(b, INF, 0.0, 10000, {})
    _assert_invalid_forecast_skip(r, {}, {})


def test_invalid_forecast_never_poisons_cluster():
    used = {}
    ak.evaluate_and_size_opportunity(_book(), NAN, 0.0, 10000, used)
    assert used == {}, used
    r = ak.evaluate_and_size_opportunity(_book(), 0.80, 0.0, 10000, used)
    assert r.action == ak.TradeAction.BUY_YES and r.invested_capital == 1500.0, r
    assert used == {"C9440": 0.15}, used


# ---- 3. Distinguishability: genuine no-edge markets keep their reason --------

def test_genuine_edgeless_market_still_insufficient_net_edge():
    # p=0.65 vs ask 0.62 / bid 0.60 at fee 0.01: edge 0.0238 < 0.05 —
    # the exact shape of the pre-fix NaN format, but for a FINITE
    # posterior it is a genuine market-efficiency skip and must keep
    # the INSUFFICIENT_NET_EDGE reason with a finite formatted edge.
    r = ak.evaluate_and_size_opportunity(
        _book(ask=0.62, bid=0.60, fee=0.01), 0.65, 0.0, 10000, {})
    assert r.action == ak.TradeAction.SKIP, r
    assert r.skip_reason.startswith("INSUFFICIENT_NET_EDGE"), r.skip_reason
    assert "max_edge=0.024 < 0.050" in r.skip_reason, r.skip_reason
    assert "INVALID_FORECAST" not in r.skip_reason, r.skip_reason


def test_finite_out_of_range_keeps_plain_invalid_posterior():
    # SF-BL-9470's case is byte-distinct from SF-BL-9440's: finite
    # out-of-range posteriors carry INVALID_POSTERIOR WITHOUT the
    # INVALID_FORECAST qualifier.
    for bad in (-0.20, 1.50):
        r = ak.evaluate_and_size_opportunity(_book(), bad, 0.0, 10000, {})
        assert r.skip_reason.startswith("INVALID_POSTERIOR"), r.skip_reason
        assert "INVALID_FORECAST" not in r.skip_reason, r.skip_reason


# ---- 4. Valid behaviour unchanged ---------------------------------------------

def test_trade_control_unchanged():
    used = {}
    r = ak.evaluate_and_size_opportunity(_book(), 0.80, 0.0, 10000, used)
    assert r.action == ak.TradeAction.BUY_YES, r
    assert r.invested_capital == 1500.0 and r.shares_acquired == 3000.0, r
    assert r.allocated_fraction == 0.15 and r.exec_price == 0.50, r
    assert used == {"C9440": 0.15}, used


def test_aggregate_invariant_at_entry():
    # The kill branch's regression-test form, asserted at evaluate entry
    # even though the kill did not trigger: aggregate output is finite
    # and in [0.02, 0.98], so it never trips Gate 0.0a — while a bare
    # NaN literal from any other source still does (tests above).
    import warnings as _w
    members = [ak.SubAgentForecast("a1", "f1", 0.70, 1.0, True),
               ak.SubAgentForecast("a2", "f2", 0.60, 1.0, True),
               ak.SubAgentForecast("a3", "f3", NAN, 1.0, True)]
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        p, _disp = ak.aggregate_fleet_forecasts(members)
    assert p == p and 0.02 <= p <= 0.98, p
    r = ak.evaluate_and_size_opportunity(_book(), p, 0.0, 10000, {})
    assert "INVALID_FORECAST" not in (r.skip_reason or ""), r.skip_reason


# ---- 5. Benchmark slate unchanged ----------------------------------------------

def test_benchmark_slate_counts_unchanged():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    assert '"north_star_score": 0.0547' in out, out[-2000:]
    assert '"north_star_score": -0.1499' in out, out[-2000:]
    reasons = Counter()
    trades = 0
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("[SKIP]"):
            reasons[line.split("Reason: ", 1)[1].split(" (")[0]] += 1
        elif line.startswith("[BUY"):
            trades += 1
    # Baseline captured this run immediately BEFORE the SF-BL-9440 patch
    # (repro.py output in the task work dir): identical counts required.
    assert trades == 3, (trades, out[-2000:])
    assert reasons == Counter({
        "INSUFFICIENT_NET_EDGE": 1,
        "FAILED_CRYPTOBENCH_ONCHAIN_GATE": 1,
        "ROUND_TRIP_GATE_D": 1,
        "BUDGET_SENSITIVITY_VETO": 1,
        "UNRESOLVED_AGENT_DISPERSION": 1,
    }), reasons


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
