"""
test_alpha_kelly_float_outcome.py — SF-BL-11009 verification tests.

Pre-registered done criterion: "settle_or_mark_trade must raise the
same named ValueError for a float outcome_yes as for 0.5, fixtured at
1.0 and 0.0, leaving net_pnl_bid_marked untouched, with genuine int 0/1
settlements bit-identical; kill if a producer census proves every
outcome reaching settle is a strict int under a tested invariant."

Titled fixture (from TASK.md): BUY_YES, 1000 shares, $500 stake.
Pre-fix (equality membership `outcome_yes not in (0, 1)`): in Python
`1.0 == 1` and `0.0 == 0` are True, so outcome_yes=1.0 booked the full
+$500.00 YES win and 0.0 booked the -$500.00 loss — byte-identical to
the genuine int settlements — while outcome_yes='1' correctly raised.
Distinct input dimension (float TYPE) from SF-BL-9477 (bool outcomes);
SF-BL-10822 covers the is_settled argument, not this one.

State at this task's run (2026-10-23): the strict-int validator landed
for SF-BL-9477 (_validate_outcome_yes / _outcome_yes_in_domain:
numbers.Integral, bool excluded) already rejects floats — its suite
pins 1.0/0.0 as a disclosed scope decision. This suite is the
task-specific verification for the FLOAT dimension: the exact titled
fixture, atomicity of all three settle fields on an OPEN trade,
BUY_NO mirroring, float-typed numpy scalars, and the JSON-producer
scenario (json.loads parses `1.0` as float, `1` as int).

Kill condition NOT met (PRODUCER_CENSUS.md in this dir): the only
in-repo production caller is run_benchmark_simulation, fed by bare
dict literals under an unenforced `outcome_yes: int` annotation; no
tested invariant guarantees strict ints, and a JSON producer emitting
1.0/0.0 is exactly the titled hazard.

Run:  python3 test_alpha_kelly_float_outcome.py  (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner.
"""
import json
import os
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
    """Titled prereg fixture: BUY_YES, 1000 shares, $500 stake."""
    t = ak.TradeExecutionResult(
        market_id="FIXTURE", title="SF-BL-11009 fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C11009",
        action=action or ak.TradeAction.BUY_YES,
        calibrated_prob_yes=0.60, market_mid_prob=0.50, exec_price=0.50,
        liquidation_bid_price=bid, kl_divergence_nats=0.0,
        expected_log_growth=0.0, kelly_fraction_raw=0.10,
        allocated_fraction=0.05, invested_capital=invested,
        shares_acquired=shares, est_days_to_resolution=10.0)
    t.net_pnl_bid_marked = pnl
    return t


def _open_trade(**kw):
    """An open (unsettled) trade with a sentinel PnL, for atomicity."""
    t = _trade(pnl=SENTINEL_PNL, **kw)
    t.is_settled = False
    t.outcome_yes = None
    return t


def _raise_of(fn, *args, **kw):
    try:
        fn(*args, **kw)
    except ValueError as e:
        return e
    return None


def _raises_same_as_0p5(trade, outcome):
    """True iff settling with `outcome` raises the SAME ValueError
    (type + message) as the 0.5 reference raise."""
    ref = _raise_of(ak.settle_or_mark_trade, _trade(),
                    is_settled=True, outcome_yes=0.5)
    err = _raise_of(ak.settle_or_mark_trade, trade,
                    is_settled=True, outcome_yes=outcome)
    return (err is not None and ref is not None
            and type(err) is type(ref) and str(err) == str(ref)
            and str(err) == EXPECTED_MSG)


# ---- 1. Titled hazard: float 1.0 / 0.0 raise exactly as 0.5 -----------------

def test_float_one_raises_same_valueerror_as_0p5():
    t = _trade(pnl=SENTINEL_PNL)
    assert _raises_same_as_0p5(t, 1.0), \
        "outcome_yes=1.0 accepted (pre-fix booked +500.0)"
    assert t.net_pnl_bid_marked == SENTINEL_PNL, t.net_pnl_bid_marked


def test_float_zero_raises_same_valueerror_as_0p5():
    t = _trade(pnl=SENTINEL_PNL)
    assert _raises_same_as_0p5(t, 0.0), \
        "outcome_yes=0.0 accepted (pre-fix booked -500.0)"
    assert t.net_pnl_bid_marked == SENTINEL_PNL, t.net_pnl_bid_marked


def test_float_outcomes_raise_for_buy_no_too():
    for outcome in (1.0, 0.0):
        t = _trade(action=ak.TradeAction.BUY_NO, pnl=SENTINEL_PNL)
        assert _raises_same_as_0p5(t, outcome), \
            f"BUY_NO outcome_yes={outcome!r} accepted or wrong error"
        assert t.net_pnl_bid_marked == SENTINEL_PNL


def test_raising_float_settle_leaves_open_trade_byte_identical():
    # SF-BL-10724 ordering, exercised for the float dimension: a
    # raising settle on an OPEN trade leaves is_settled, outcome_yes
    # and net_pnl_bid_marked exactly at their pre-call values.
    for outcome in (1.0, 0.0):
        t = _open_trade()
        assert _raises_same_as_0p5(t, outcome)
        assert t.is_settled is False and t.outcome_yes is None \
            and t.net_pnl_bid_marked == SENTINEL_PNL, \
            (t.is_settled, t.outcome_yes, t.net_pnl_bid_marked)


def test_negative_zero_and_other_floats_raise():
    # -0.0 == 0.0 == 0 by value; it must still raise — the domain is
    # the TYPE (strict int), not the value. Same for near-miss floats.
    for outcome in (-0.0, 0.5, 1.0000001, 0.9999999, -1.0, 2.0):
        t = _trade(pnl=SENTINEL_PNL)
        assert _raises_same_as_0p5(t, outcome), \
            f"outcome_yes={outcome!r} (float) accepted"
        assert t.net_pnl_bid_marked == SENTINEL_PNL


# ---- 2. The titled producer scenario: a JSON resolution --------------------

def test_json_producer_float_outcome_raises_int_outcome_settles():
    # The hazard in TASK.md: "a float-typed resolution from a JSON
    # producer silently settles". json.loads parses 1.0 as float and
    # 1 as int — same payload shape, different type.
    payload_float = json.loads('{"outcome_yes": 1.0, "is_settled": true}')
    assert type(payload_float["outcome_yes"]) is float
    t = _trade(pnl=SENTINEL_PNL)
    assert _raises_same_as_0p5(t, payload_float["outcome_yes"])
    assert t.net_pnl_bid_marked == SENTINEL_PNL
    payload_int = json.loads('{"outcome_yes": 1, "is_settled": true}')
    assert type(payload_int["outcome_yes"]) is int
    win = ak.settle_or_mark_trade(_trade(), is_settled=True,
                                  outcome_yes=payload_int["outcome_yes"])
    assert win.net_pnl_bid_marked == 500.0, win.net_pnl_bid_marked


def test_string_one_still_raises_as_task_records():
    # Control from TASK.md: '1' correctly raised pre-fix and still does,
    # with the identical message — the float fix changes nothing here.
    t = _trade(pnl=SENTINEL_PNL)
    assert _raises_same_as_0p5(t, "1")
    assert t.net_pnl_bid_marked == SENTINEL_PNL


# ---- 3. Genuine int 0/1 settlements stay bit-identical ----------------------

def test_genuine_int_settlements_bit_identical():
    win = ak.settle_or_mark_trade(_trade(), is_settled=True, outcome_yes=1)
    assert win.net_pnl_bid_marked == 500.0 and win.outcome_yes == 1 \
        and type(win.outcome_yes) is int, win.net_pnl_bid_marked
    loss = ak.settle_or_mark_trade(_trade(), is_settled=True, outcome_yes=0)
    assert loss.net_pnl_bid_marked == -500.0 and loss.outcome_yes == 0 \
        and type(loss.outcome_yes) is int, loss.net_pnl_bid_marked
    no_win = ak.settle_or_mark_trade(
        _trade(action=ak.TradeAction.BUY_NO), is_settled=True, outcome_yes=0)
    assert no_win.net_pnl_bid_marked == 500.0, no_win.net_pnl_bid_marked
    no_loss = ak.settle_or_mark_trade(
        _trade(action=ak.TradeAction.BUY_NO), is_settled=True, outcome_yes=1)
    assert no_loss.net_pnl_bid_marked == -500.0, no_loss.net_pnl_bid_marked


def test_helper_domain_is_type_based_not_value_based():
    d = ak._outcome_yes_in_domain
    assert d(0) and d(1)
    # Every float whose VALUE equals 0 or 1 is still out of domain.
    for bad in (1.0, 0.0, -0.0, 0.5, 2.0):
        assert not d(bad), f"_outcome_yes_in_domain({bad!r}) is True"


# ---- 4. Float-typed numpy scalars (the JSON-adjacent producer types) --------

def test_numpy_float_scalars_raise_numpy_int_settles():
    try:
        import numpy as np
    except ImportError:
        print("numpy unavailable — numpy probes skipped (vacuous pass)")
        return
    for outcome in (np.float64(1.0), np.float64(0.0), np.float32(1.0)):
        t = _trade(pnl=SENTINEL_PNL)
        assert _raises_same_as_0p5(t, outcome), \
            f"outcome_yes={outcome!r} ({type(outcome).__name__}) accepted"
        assert t.net_pnl_bid_marked == SENTINEL_PNL
    win = ak.settle_or_mark_trade(_trade(), is_settled=True,
                                  outcome_yes=np.int64(1))
    assert win.net_pnl_bid_marked == 500.0, win.net_pnl_bid_marked


# ---- 5. Preserved orderings (disclosed, pinned by SF-BL-9477) ---------------

def test_open_mark_branch_unaffected_by_float_outcome_arg():
    # The mark branch never reads outcome_yes; validation lives in the
    # settled branch only, so an open mark is unchanged.
    t = ak.settle_or_mark_trade(_trade(), is_settled=False, outcome_yes=1.0)
    assert t.net_pnl_bid_marked == 0.0, t.net_pnl_bid_marked  # 1000*.50-500


def test_skip_and_zero_invested_ordering_unchanged():
    # The SKIP / zero-invested early return precedes outcome validation
    # exactly as pre-fix: such trades book PnL 0.0 (they hold no
    # position, so there is no settlement to corrupt) without raising.
    skip = _trade(action=ak.TradeAction.SKIP)
    r = ak.settle_or_mark_trade(skip, is_settled=True, outcome_yes=1.0)
    assert r.net_pnl_bid_marked == 0.0, r.net_pnl_bid_marked
    zero = _trade(invested=0.0)
    r2 = ak.settle_or_mark_trade(zero, is_settled=True, outcome_yes=0.0)
    assert r2.net_pnl_bid_marked == 0.0, r2.net_pnl_bid_marked


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
