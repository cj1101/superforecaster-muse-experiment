"""
test_posterior_spec_clip.py — SF-BL-9329 verification tests.

Prereg done criterion: "fixtures at p in {0.999, 1.0, -0.05} currently trade
or size differently from their [0.02,0.98]-clipped equivalents on the
CLI/backtest surfaces, and the conformance fix makes them identical; kill
if a caller census proves aggregate_fleet_forecasts is the only reachable
caller and its output is in-band by construction, downgrading this to a
guard test."

Kill NOT met — see CALLER_CENSUS.md in the task work dir: aggregate is
only the benchmark producer; direct callers (repo tests, wave3 r63 scan)
pass bare float literals with no provenance.

Reconciliation of the -0.05 arm: this task was filed before SF-BL-9470
landed Gate 0.0 ([0, 1] validity). Out-of-domain posteriors SKIP
INVALID_POSTERIOR and the clip must never rescue them (clipping -0.05 to
0.02 would also turn 1.50 into a 0.98 trade). The fix therefore validates
first, then clips the validated posterior: 0.999 / 1.0 become identical
to 0.98 in EVERY result field, and -0.05 keeps its INVALID_POSTERIOR skip.

Pre-fix reproduction (repro.py, this dir; verified live 2026-10-23):
  fixture (ask 0.50 x 100k, bid 0.48, fee 0.0, equity 10000):
    p=0.999 -> kelly_raw 0.998, stored_p 0.999   (clip 0.98: 0.96)
    p=1.0   -> kelly_raw 1.0,   stored_p 1.0
    uncapped config: p=0.999 invested 3493.0, p=1.0 invested 3500.0
                     vs p=0.98 invested 3360.0; p=0.01 invested 3427.08
                     vs p=0.02 invested 3354.17
  post-fix every pair above is field-identical.

Benchmark note: the 8-market benchmark is asserted UNCHANGED relative to
the pre-patch baseline captured this run (disciplined +0.0547 / naive
-0.1499). The +0.1059 figure pinned by older test files drifted before
this task (concurrent fleet changes; test_alpha_kelly_invalid_posterior
.py's benchmark check already failed 9/10 pre-patch) — the aggregate path
is in-band, so the clip is the identity there and cannot move it.

Run:  python3 test_posterior_spec_clip.py   (exit 0 iff all pass)
No pytest dependency, same convention as the other alpha_kelly tests.
"""
import io
import contextlib
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


def _book(cluster="C9329", ask=0.50, bid=0.48, fee=0.0):
    return ak.OrderBookSnapshot(
        market_id="FIXTURE", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster=cluster,
        est_days_to_resolution=5,
        yes_asks=[ak.OrderBookLevel(price=ask, size_shares=100_000)],
        yes_bids=[ak.OrderBookLevel(price=bid, size_shares=100_000)],
        fee_rate=fee)


def _fields(r):
    return (r.action, r.calibrated_prob_yes, r.market_mid_prob, r.exec_price,
            r.liquidation_bid_price, r.kl_divergence_nats,
            r.expected_log_growth, r.kelly_fraction_raw, r.allocated_fraction,
            r.invested_capital, r.shares_acquired, r.skip_reason)


def _uncapped_cfg():
    return ak.FleetRiskConfig(max_cluster_allocation=1.0,
                              max_round_trip_cost=None,
                              budget_sensitivity_veto_enabled=False)


# ---- 1. Prereg fixtures: out-of-band == clipped equivalent -------------------

def test_prereg_high_fixtures_identical_to_clip():
    for p in (0.999, 1.0, 0.995, 0.99):
        a = ak.evaluate_and_size_opportunity(_book("A"), p, 0.0, 10000, {})
        b = ak.evaluate_and_size_opportunity(_book("A"), 0.98, 0.0, 10000, {})
        assert _fields(a) == _fields(b), (p, a, b)
        assert a.calibrated_prob_yes == 0.98, (p, a.calibrated_prob_yes)


def test_low_fixtures_identical_to_clip():
    for p in (0.01, 0.0, 0.001):
        a = ak.evaluate_and_size_opportunity(_book("A"), p, 0.0, 10000, {})
        b = ak.evaluate_and_size_opportunity(_book("A"), 0.02, 0.0, 10000, {})
        assert _fields(a) == _fields(b), (p, a, b)
        assert a.calibrated_prob_yes == 0.02, (p, a.calibrated_prob_yes)


def test_uncapped_sizing_identical():
    cfg = _uncapped_cfg()
    a = ak.evaluate_and_size_opportunity(_book("U"), 0.999, 0.0, 10000, {}, cfg)
    b = ak.evaluate_and_size_opportunity(_book("U"), 1.0, 0.0, 10000, {}, cfg)
    c = ak.evaluate_and_size_opportunity(_book("U"), 0.98, 0.0, 10000, {}, cfg)
    assert a.invested_capital == b.invested_capital == c.invested_capital, (a, b, c)
    assert a.invested_capital == 3359.9999999999995, a.invested_capital
    d = ak.evaluate_and_size_opportunity(_book("U"), 0.01, 0.0, 10000, {}, cfg)
    e = ak.evaluate_and_size_opportunity(_book("U"), 0.02, 0.0, 10000, {}, cfg)
    assert d.invested_capital == e.invested_capital == 3354.1666666666665, (d, e)


def test_identity_on_skip_paths_after_clip():
    # Dispersion SKIP record: stored posterior + reason identical.
    a = ak.evaluate_and_size_opportunity(_book("A"), 0.999, 0.50, 10000, {})
    b = ak.evaluate_and_size_opportunity(_book("A"), 0.98, 0.50, 10000, {})
    assert _fields(a) == _fields(b), (a, b)
    # Edge-screen SKIP on a high ask: max_edge in the reason is computed
    # from the clipped posterior, so the records are identical too.
    a = ak.evaluate_and_size_opportunity(_book("A", ask=0.95, bid=0.93), 0.999, 0.0, 10000, {})
    b = ak.evaluate_and_size_opportunity(_book("A", ask=0.95, bid=0.93), 0.98, 0.0, 10000, {})
    assert a.action == ak.TradeAction.SKIP and _fields(a) == _fields(b), (a, b)


def test_no_side_identity():
    # Low posterior trades BUY_NO; the mirrored p_side uses the clip too.
    a = ak.evaluate_and_size_opportunity(_book("A"), 0.0, 0.0, 10000, {})
    b = ak.evaluate_and_size_opportunity(_book("A"), 0.02, 0.0, 10000, {})
    assert a.action == ak.TradeAction.BUY_NO and _fields(a) == _fields(b), (a, b)


def test_int_posterior_clips():
    r = ak.evaluate_and_size_opportunity(_book(), 1, 0.0, 10000, {})
    assert r.action == ak.TradeAction.BUY_YES and r.calibrated_prob_yes == 0.98, r


# ---- 2. SF-BL-9470 ordering preserved: clip never rescues out-of-domain -----

def test_out_of_domain_still_invalid_posterior():
    for bad in (-0.05, -0.20, 1.50, float("inf"), float("-inf"), float("nan")):
        used = {}
        r = ak.evaluate_and_size_opportunity(_book(), bad, 0.0, 10000, used)
        assert r.action == ak.TradeAction.SKIP, (bad, r)
        assert r.skip_reason.startswith("INVALID_POSTERIOR"), (bad, r.skip_reason)
        assert r.calibrated_prob_yes == bad or (
            isinstance(bad, float) and bad != bad and r.calibrated_prob_yes != r.calibrated_prob_yes), (bad, r)
        assert r.invested_capital == 0.0 and used == {}, (bad, r, used)


# ---- 3. In-band behaviour unchanged -------------------------------------------

def test_in_band_control_unchanged():
    used = {}
    r = ak.evaluate_and_size_opportunity(_book(), 0.80, 0.0, 10000, used)
    assert r.action == ak.TradeAction.BUY_YES, r
    assert r.invested_capital == 1500.0 and r.shares_acquired == 3000.0, r
    assert r.calibrated_prob_yes == 0.80 and r.kelly_fraction_raw == 0.6000000000000001, r
    assert used == {"C9329": 0.15}, used
    for p in (0.02, 0.5, 0.98):
        r = ak.evaluate_and_size_opportunity(_book(), p, 0.0, 10000, {})
        assert r.calibrated_prob_yes == p, (p, r.calibrated_prob_yes)


def test_clip_constants_match_spec_surfaces():
    assert ak.POSTERIOR_CLIP_MIN == 0.02 and ak.POSTERIOR_CLIP_MAX == 0.98
    assert ak.clip_probability(0.999) == 0.98 and ak.clip_probability(0.0) == 0.02
    # aggregate output is already in-band: the clip is the identity there.
    cal, _ = ak.aggregate_fleet_forecasts([
        ak.SubAgentForecast("a", "fam1", 0.999),
        ak.SubAgentForecast("b", "fam2", 0.995),
        ak.SubAgentForecast("c", "fam3", 0.99)])
    assert 0.02 <= cal <= 0.98 and ak.clip_probability(cal) == cal, cal


def test_benchmark_unchanged_from_prepatch_baseline():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    assert '"north_star_score": 0.0547' in out, out[-2000:]
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
