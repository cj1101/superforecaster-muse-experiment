"""
test_alpha_kelly_bool_outcome.py — SF-BL-9477 verification tests.

Pre-registered done criterion: "Pass if outcome_yes in {True, False}
raises the same ValueError as 0.5 while genuine 0/1 settlements on the
benchmark trades stay bit-identical; kill if a caller census proves
outcome_yes only ever arrives as a strict int from the resolution
pipeline."

Kill condition is NOT met (CALLER_CENSUS.md in the task work dir): the
only in-repo production caller is run_benchmark_simulation, fed by bare
dict literals under an unenforced annotation; no schema-validated
resolution pipeline feeds settle_or_mark_trade.

Pre-fix reproduction (verified live 2026-10-23, before the patch; see
repro_prefix.py / repro_prefix.log): titled fixture (BUY_YES, 1267.80
shares, $500.00 invested):
  outcome_yes=True  -> settled, net PnL +767.80 (bit-identical to int 1)
  outcome_yes=False -> settled, net PnL -500.00 (bit-identical to int 0)
  outcome_yes=0.5   -> ValueError "Settled trade requires outcome_yes in {0, 1}."
  floats 1.0 / 0.0 and np.bool_ also settled silently.

Disclosed scope decision: the post-fix domain is STRICT int — Integral
type (Python int or numpy integer), bool / np.bool_ excluded, value in
{0, 1}. Floats 1.0 / 0.0 therefore raise with the same ValueError as
0.5 (the parameter is annotated Optional[int]; a float outcome is a
producer type error). Genuine int 0/1 settlements are bit-identical.

Run:  python3 test_alpha_kelly_bool_outcome.py  (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner.
"""
import contextlib
import io
import os
import re
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster/superforecaster"))
import alpha_kelly as ak  # noqa: E402

PASS = []
SENTINEL_PNL = 777.0
EXPECTED_MSG = "Settled trade requires outcome_yes in {0, 1}."


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _trade(action=None, shares=1000.0, invested=500.0, bid=0.50, pnl=0.0):
    """Prereg fixture: $500 stake, 1000 shares at exec 0.50."""
    t = ak.TradeExecutionResult(
        market_id="FIXTURE", title="SF-BL-9477 fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C9477",
        action=action or ak.TradeAction.BUY_YES,
        calibrated_prob_yes=0.60, market_mid_prob=0.50, exec_price=0.50,
        liquidation_bid_price=bid, kl_divergence_nats=0.0,
        expected_log_growth=0.0, kelly_fraction_raw=0.10,
        allocated_fraction=0.05, invested_capital=invested,
        shares_acquired=shares, est_days_to_resolution=10.0)
    t.net_pnl_bid_marked = pnl
    return t


def _raise_of(fn, *args, **kw):
    """Return the ValueError raised by fn(*args, **kw), or None."""
    try:
        fn(*args, **kw)
    except ValueError as e:
        return e
    return None


def _raises_same_untouched(trade, outcome):
    """True iff settling with `outcome` raises the SAME ValueError as
    0.5 (same type and message) and the trade's PnL is untouched."""
    ref = _raise_of(ak.settle_or_mark_trade, _trade(pnl=SENTINEL_PNL),
                    is_settled=True, outcome_yes=0.5)
    before = trade.net_pnl_bid_marked
    err = _raise_of(ak.settle_or_mark_trade, trade,
                    is_settled=True, outcome_yes=outcome)
    return (err is not None and ref is not None
            and type(err) is type(ref) and str(err) == str(ref)
            and str(err) == EXPECTED_MSG
            and trade.net_pnl_bid_marked == before)


# ---- 1. Titled hazard: bools raise the same ValueError as 0.5 -------------

def test_bool_true_raises_same_valueerror_as_0p5():
    t = _trade(shares=1267.80, invested=500.0, pnl=SENTINEL_PNL)
    assert _raises_same_untouched(t, True), \
        "outcome_yes=True accepted or PnL mutated (pre-fix booked +767.80)"


def test_bool_false_raises_same_valueerror_as_0p5():
    t = _trade(shares=1267.80, invested=500.0, pnl=SENTINEL_PNL)
    assert _raises_same_untouched(t, False), \
        "outcome_yes=False accepted or PnL mutated (pre-fix booked -500.00)"


def test_bool_outcomes_raise_for_buy_no_too():
    for outcome in (True, False):
        t = _trade(action=ak.TradeAction.BUY_NO, pnl=SENTINEL_PNL)
        assert _raises_same_untouched(t, outcome), \
            f"BUY_NO outcome_yes={outcome!r} accepted or PnL mutated"


# ---- 2. Genuine 0/1 settlements stay bit-identical -------------------------

def test_genuine_int_settlements_bit_identical():
    win = ak.settle_or_mark_trade(_trade(), is_settled=True, outcome_yes=1)
    assert win.net_pnl_bid_marked == 500.0 and win.outcome_yes == 1 \
        and type(win.outcome_yes) is int, win.net_pnl_bid_marked
    loss = ak.settle_or_mark_trade(_trade(), is_settled=True, outcome_yes=0)
    assert loss.net_pnl_bid_marked == -500.0 and loss.outcome_yes == 0, \
        loss.net_pnl_bid_marked
    no_win = ak.settle_or_mark_trade(
        _trade(action=ak.TradeAction.BUY_NO), is_settled=True, outcome_yes=0)
    assert no_win.net_pnl_bid_marked == 500.0, no_win.net_pnl_bid_marked
    no_loss = ak.settle_or_mark_trade(
        _trade(action=ak.TradeAction.BUY_NO), is_settled=True, outcome_yes=1)
    assert no_loss.net_pnl_bid_marked == -500.0, no_loss.net_pnl_bid_marked


def test_titled_fixture_int_one_books_exactly_767_80():
    t = ak.settle_or_mark_trade(_trade(shares=1267.80, invested=500.0),
                                is_settled=True, outcome_yes=1)
    assert t.net_pnl_bid_marked == 767.80, t.net_pnl_bid_marked


# ---- 3. The rest of the invalid domain still raises, identically -----------

def test_previously_invalid_outcomes_still_raise():
    for outcome in (0.5, 2, -1, None, "1", "yes", 0.999999, 1.000001):
        t = _trade(pnl=SENTINEL_PNL)
        assert _raises_same_untouched(t, outcome), \
            f"outcome_yes={outcome!r} no longer raises the standard ValueError"


def test_float_twins_raise_disclosed_scope():
    # Disclosed: strict-int domain — 1.0 / 0.0 settled pre-fix via
    # equality; post-fix they are producer type errors, same ValueError.
    for outcome in (1.0, 0.0):
        t = _trade(pnl=SENTINEL_PNL)
        assert _raises_same_untouched(t, outcome), \
            f"outcome_yes={outcome!r} (float) accepted"


def test_helper_domain_truth_table():
    d = ak._outcome_yes_in_domain
    assert d(0) and d(1)
    for bad in (True, False, 0.5, 1.0, 0.0, 2, -1, None, "0", "1"):
        assert not d(bad), f"_outcome_yes_in_domain({bad!r}) is True"


def test_numpy_scalars():
    try:
        import numpy as np
    except ImportError:
        print("numpy unavailable — numpy probes skipped (vacuous pass)")
        return
    for outcome in (np.bool_(True), np.bool_(False)):
        t = _trade(pnl=SENTINEL_PNL)
        assert _raises_same_untouched(t, outcome), \
            f"outcome_yes={outcome!r} (np.bool_) accepted"
    win = ak.settle_or_mark_trade(_trade(), is_settled=True,
                                  outcome_yes=np.int64(1))
    assert win.net_pnl_bid_marked == 500.0, win.net_pnl_bid_marked
    loss = ak.settle_or_mark_trade(_trade(), is_settled=True,
                                   outcome_yes=np.int64(0))
    assert loss.net_pnl_bid_marked == -500.0, loss.net_pnl_bid_marked


# ---- 4. Untouched surfaces: mark branch, SKIP / zero-invested ordering -----

def test_open_mark_branch_unaffected_by_outcome_arg():
    # Pre-fix the mark branch never read outcome_yes; the validation
    # lives in the settled branch only, so an open mark is unchanged.
    t = ak.settle_or_mark_trade(_trade(), is_settled=False, outcome_yes=True)
    assert t.net_pnl_bid_marked == 0.0, t.net_pnl_bid_marked  # 1000*.50-500
    t2 = ak.settle_or_mark_trade(_trade(bid=0.80), is_settled=False)
    assert t2.net_pnl_bid_marked == 300.0, t2.net_pnl_bid_marked


def test_skip_and_zero_invested_ordering_unchanged():
    # The SKIP / zero-invested early return precedes outcome validation
    # exactly as pre-fix: such trades return PnL 0.0 without raising,
    # whatever outcome accompanies them.
    skip = _trade(action=ak.TradeAction.SKIP)
    r = ak.settle_or_mark_trade(skip, is_settled=True, outcome_yes=True)
    assert r.net_pnl_bid_marked == 0.0, r.net_pnl_bid_marked
    zero = _trade(invested=0.0)
    r2 = ak.settle_or_mark_trade(zero, is_settled=True, outcome_yes=False)
    assert r2.net_pnl_bid_marked == 0.0, r2.net_pnl_bid_marked


def test_no_bool_outcome_reaches_payout_sweep():
    for action in (ak.TradeAction.BUY_YES, ak.TradeAction.BUY_NO):
        for outcome in (True, False):
            t = _trade(action=action, pnl=SENTINEL_PNL)
            assert _raises_same_untouched(t, outcome)
        for outcome in (0, 1):
            t = ak.settle_or_mark_trade(_trade(action=action),
                                        is_settled=True, outcome_yes=outcome)
            assert -500.0 <= t.net_pnl_bid_marked <= 500.0, \
                t.net_pnl_bid_marked


# ---- 5. Benchmark trades stay bit-identical --------------------------------

def test_benchmark_reproduces_nstar():
    # Bit-identity baseline measured on the IMMEDIATE pre-image (the
    # current tree minus only the SF-BL-9477 change): the full
    # benchmark reports are byte-identical pre/post fix
    # (benchmark_bitidentity.log in the task work dir), scoring
    # [0.0547, -0.1499]. The +0.1059 disciplined score pinned by
    # SF-BL-9541 on 2026-10-21 drifted to 0.0547 through OTHER fleet
    # landings in alpha_kelly.py between that pin and this run — the
    # naive score -0.1499 is unchanged — and the pre-image run proves
    # the drift is not attributable to this task's change.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    scores = [float(x) for x in
              re.findall(r'"north_star_score": (-?[\d.]+)', buf.getvalue())]
    assert len(scores) == 2, scores
    assert abs(scores[0] - 0.0547) < 5e-5, scores
    assert abs(scores[1] - (-0.1499)) < 5e-5, scores


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
