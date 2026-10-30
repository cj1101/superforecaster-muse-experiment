#!/usr/bin/env python3
"""SF-BL-35035 tests: cluster_exposure_released assignment guard.

Prereg: "Success if cluster_exposure_released assignments are
strict-bool guard-validated (named ValueError on non-bool) and a
post-release reset cannot produce a second subtraction on a
co-tenant fixture (ledger stays 0.10 after exactly one release),
with first releases bit-identical; kill if a census shows the flag
is only ever written by release_cluster_exposure itself under a
tested invariant."

Kill NOT met (CALLER_CENSUS.md): default + construction kwargs,
mutable dataclass (any caller assignment), and bypass routes all
write the flag outside release_cluster_exposure.

Run: python3 test_alpha_kelly_cluster_released_guard.py
No pytest dependency: test_* functions with assert + __main__ runner.
"""
import contextlib, io, os, pickle, re, sys
sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster"))
from superforecaster import alpha_kelly as ak  # noqa: E402

BAD = ("yes", "False", "True", "false", "", "0", "1", 1, 0, 1.0, 0.0, -1, None, [], [False], {}, [True])

def _trade(ledger=None, alloc=0.10, action=ak.TradeAction.BUY_YES, released=False):
    return ak.TradeExecutionResult(
        market_id="FIX", title="fixture", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C1", action=action, calibrated_prob_yes=0.6,
        market_mid_prob=0.55, exec_price=0.50, liquidation_bid_price=0.40,
        kl_divergence_nats=0.1, expected_log_growth=0.01, kelly_fraction_raw=0.1,
        allocated_fraction=alloc, invested_capital=1000.0, shares_acquired=2000.0,
        est_days_to_resolution=30.0, cluster_exposure_used=ledger,
        cluster_exposure_released=released)

def _invalid(fn):
    try: fn()
    except ValueError as e: return "INVALID_CLUSTER_EXPOSURE_RELEASED" in str(e) and "SF-BL-35035" in str(e)
    except Exception: return False
    return False

def _reset_named(fn):
    try: fn()
    except ValueError as e: return "CLUSTER_EXPOSURE_RELEASED_RESET" in str(e) and "SF-BL-35035" in str(e)
    except Exception: return False
    return False

# 1. Prereg co-tenant fixture -------------------------------------------------
def test_prereg_first_release_exact():
    ledger = {"C1": 0.20}; t = _trade(ledger)
    removed = ak.release_cluster_exposure(t, ledger)
    assert removed == 0.10 and ledger == {"C1": 0.10} and t.cluster_exposure_released is True

def test_prereg_reset_raises_and_second_release_noop():
    ledger = {"C1": 0.20}; t = _trade(ledger)
    ak.release_cluster_exposure(t, ledger)
    assert _reset_named(lambda: setattr(t, "cluster_exposure_released", False))
    assert t.cluster_exposure_released is True  # rejected assignment preserves
    assert ak.release_cluster_exposure(t, ledger) == 0.0 and ledger == {"C1": 0.10}

def test_bypass_reset_cannot_double_release():
    for mode in ("object", "dict"):
        ledger = {"C1": 0.20}; t = _trade(ledger)
        ak.release_cluster_exposure(t, ledger)
        if mode == "object": object.__setattr__(t, "cluster_exposure_released", False)
        else: t.__dict__["cluster_exposure_released"] = False
        assert ak.release_cluster_exposure(t, ledger) == 0.0, mode
        assert ledger == {"C1": 0.10} and t.cluster_exposure_released is True, mode

def test_settle_after_reset_cannot_double_release():
    ledger = {"C1": 0.20}; t = _trade(ledger)
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1, cluster_exposure_used=ledger)
    assert ledger == {"C1": 0.10}
    assert _reset_named(lambda: setattr(t, "cluster_exposure_released", False))
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1, cluster_exposure_used=ledger)
    assert ledger == {"C1": 0.10}

# 2. Strict-bool assignment / construction ------------------------------------
def test_assignment_rejects_nonbool_space_and_preserves():
    for bad in BAD:
        t = _trade()
        assert _invalid(lambda bad=bad, t=t: setattr(t, "cluster_exposure_released", bad)), bad
        assert t.cluster_exposure_released is False, bad

def test_assignment_yes_prereg():
    t = _trade(); assert _invalid(lambda: setattr(t, "cluster_exposure_released", "yes"))

def test_construction_rejects_nonbool_space():
    for bad in BAD:
        assert _invalid(lambda bad=bad: _trade(released=bad)), bad

def test_construction_accepts_genuine_bools_and_default():
    assert _trade(released=True).cluster_exposure_released is True
    assert _trade(released=False).cluster_exposure_released is False
    t = ak.TradeExecutionResult(
        market_id="D", title="D", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C1", action=ak.TradeAction.BUY_YES, calibrated_prob_yes=0.6,
        market_mid_prob=0.55, exec_price=0.5, liquidation_bid_price=0.4,
        kl_divergence_nats=0.1, expected_log_growth=0.01, kelly_fraction_raw=0.1,
        allocated_fraction=0.05, invested_capital=500.0, shares_acquired=1000.0,
        est_days_to_resolution=30.0)
    assert t.cluster_exposure_released is False

def test_genuine_bool_assignments_pre_release():
    t = _trade()
    t.cluster_exposure_released = False; assert t.cluster_exposure_released is False
    t.cluster_exposure_released = True; assert t.cluster_exposure_released is True
    # once True (even manually), resetting is the latch violation
    assert _reset_named(lambda: setattr(t, "cluster_exposure_released", False))
    t.cluster_exposure_released = True; assert t.cluster_exposure_released is True

def test_released_true_assignment_idempotent():
    ledger = {"C1": 0.20}; t = _trade(ledger)
    ak.release_cluster_exposure(t, ledger)
    t.cluster_exposure_released = True; assert t.cluster_exposure_released is True

# 3. Consumption gates (bypass records) ---------------------------------------
def test_release_gate_rejects_bypass_nonbool():
    for bad in ("yes", 1, 0, None, [False]):
        ledger = {"C1": 0.20}; t = _trade(ledger)
        object.__setattr__(t, "cluster_exposure_released", bad)
        assert _invalid(lambda t=t, ledger=ledger: ak.release_cluster_exposure(t, ledger)), bad
        assert ledger == {"C1": 0.20}, bad

def test_settle_gate_rejects_bypass_nonbool_before_writes():
    ledger = {"C1": 0.05}; t = _trade(ledger, alloc=0.05)
    t.is_settled = False; t.outcome_yes = None; t.net_pnl_bid_marked = -123.0
    object.__setattr__(t, "cluster_exposure_released", "yes")
    assert _invalid(lambda: ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1, cluster_exposure_used=ledger))
    assert t.is_settled is False and t.outcome_yes is None and t.net_pnl_bid_marked == -123.0
    assert ledger == {"C1": 0.05}

def test_pickled_released_trade_cannot_double_release():
    ledger = {"C1": 0.20}; t = _trade(ledger)
    ak.release_cluster_exposure(t, ledger)
    # Pickle the (trade, ledger) pair together so the loaded graph
    # preserves ledger identity (SF-BL-35212's entry-ledger snapshot
    # is the loaded ledger itself, as in a genuine resumed batch).
    t2, ledger2 = pickle.loads(pickle.dumps((t, ledger)))
    assert ledger2 == {"C1": 0.10}  # co-tenant remainder after release 1
    assert ak.release_cluster_exposure(t2, ledger2) == 0.0 and ledger2 == {"C1": 0.10}

def test_constructed_released_true_noops():
    ledger = {"C1": 0.20}; t = _trade(ledger, released=True)
    assert ak.release_cluster_exposure(t, ledger) == 0.0 and ledger == {"C1": 0.20}
    # and a bypass reset of this record still cannot re-arm it
    object.__setattr__(t, "cluster_exposure_released", False)
    assert ak.release_cluster_exposure(t, ledger) == 0.0 and ledger == {"C1": 0.20}

# 4. First releases / preserved surfaces bit-identical -------------------------
def test_first_release_via_settle_exact():
    ledger = {"C1": 0.20}; t = _trade(ledger)
    out = ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1, cluster_exposure_used=ledger)
    assert out.net_pnl_bid_marked == 1000.0 and ledger == {"C1": 0.10}
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1, cluster_exposure_used=ledger)
    assert ledger == {"C1": 0.10}  # double settle releases once only

def test_evaluate_settle_lifecycle_exact():
    cfg = ak.FleetRiskConfig(budget_sensitivity_veto_enabled=False, max_round_trip_cost=None)
    book = ak.OrderBookSnapshot(market_id="E1", title="t", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="CE", est_days_to_resolution=5.0,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=1_000_000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=1_000_000)], fee_rate=0.01)
    cluster = {}
    t = ak.evaluate_and_size_opportunity(book=book, calibrated_prob_yes=0.80,
        inter_agent_dispersion=0.05, portfolio_equity=10_000.0, cluster_exposure_used=cluster, config=cfg)
    assert t.action != ak.TradeAction.SKIP and cluster["CE"] > 0
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    assert cluster["CE"] == 0.0 and t.cluster_exposure_released is True

def test_skip_and_no_ledger_noop():
    ledger = {"C1": 0.20}; t = _trade(ledger, action=ak.TradeAction.SKIP)
    assert ak.release_cluster_exposure(t, ledger) == 0.0 and ledger == {"C1": 0.20}
    t2 = _trade(None)
    assert ak.release_cluster_exposure(t2) == 0.0 and t2.cluster_exposure_released is False

def test_mismatch_gate_preserved():
    A = {"C1": 0.15}; B = {"C1": 0.10}; t = _trade(A, alloc=0.15)
    try:
        ak.release_cluster_exposure(t, B); raise AssertionError("mismatch accepted")
    except ValueError as e:
        assert "MISMATCHED_CLUSTER_LEDGER" in str(e) and "SF-BL-34868" in str(e)
    assert A == {"C1": 0.15} and B == {"C1": 0.10} and t.cluster_exposure_released is False
    assert ak.release_cluster_exposure(t, A) == 0.15 and A == {"C1": 0.0}

def test_benchmark_scores_unchanged():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): ak.run_benchmark_simulation()
    scores = [float(x) for x in re.findall(r'"north_star_score": (-?[\d.]+)', buf.getvalue())]
    assert scores == [-0.1611, -0.1499], scores

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try: fn(); print(f"PASS {fn.__name__}")
        except Exception as e: failed += 1; print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} tests PASS")
    raise SystemExit(1 if failed else 0)
