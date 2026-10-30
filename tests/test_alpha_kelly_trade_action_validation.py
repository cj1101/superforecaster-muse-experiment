"""
test_alpha_kelly_trade_action_validation.py — SF-BL-10823 verification.

Pre-registered done criterion: "a non-TradeAction action must be
rejected with a named ValueError at TradeExecutionResult
construction/assignment or at settle/scorer entry, fixtured at
'buy_yes', 'MAYBE' and None, so no such record can settle or be scored
as executed, with genuine BUY_YES/BUY_NO/SKIP records bit-identical;
kill if a producer census shows action is only ever assigned from
evaluate_and_size_opportunity's enum returns."

Kill NOT met (PRODUCER_AUDIT.md): direct construction by the naive
fleet + test suites, unrestricted mutation, JSON round-trip degrading
action to a plain string, no __post_init__ at all pre-fix.

Pre-fix reproduction (repro.py / repro_prefix.txt): $500 / 1000-share
fixture, YES resolves — 'buy_yes' / 'MAYBE' / None each settled as
-$500.00 and scored executed=1, CWR -1.0, N_star -2.2226.

This suite pins the fix at ALL layers (the prereg's "or" is satisfied
at each): construction, assignment, settle entry (both branches,
incl. bypass-constructed records), scorer entry — plus genuine-record
bit-identity against settle/score controls and the benchmark.

Run:  python3 test_alpha_kelly_trade_action_validation.py
No pytest dependency: test_* functions with assert + __main__ runner.
"""
import contextlib
import io
import os
import pickle
import re
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster/superforecaster"))
import alpha_kelly as ak  # noqa: E402

PASS = []
SENTINEL_PNL = 777.0

# Prereg fixtures first, then the extended non-member space. Plain
# 'BUY_YES'/'BUY_NO'/'SKIP' strings are included deliberately: they
# compare EQUAL to the enum members (str Enum) yet are not members.
BAD_ACTIONS = ("buy_yes", "MAYBE", None,
               "BUY_YES", "BUY_NO", "SKIP", "buy_no", "skip", "",
               "Buy_Yes", 0, 1, 0.0, 1.0, True, False, [], {})


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _kwargs(action):
    return dict(
        market_id="FIXTURE-10823", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C10823",
        action=action, calibrated_prob_yes=0.60, market_mid_prob=0.50,
        exec_price=0.50, liquidation_bid_price=0.50,
        kl_divergence_nats=0.0, expected_log_growth=0.0,
        kelly_fraction_raw=0.10, allocated_fraction=0.05,
        invested_capital=500.0, shares_acquired=1000.0,
        est_days_to_resolution=10.0)


def _trade(action=ak.TradeAction.BUY_YES, invested=500.0, shares=1000.0,
           pnl=0.0):
    t = ak.TradeExecutionResult(**{**_kwargs(action),
                                   "invested_capital": invested,
                                   "shares_acquired": shares})
    t.net_pnl_bid_marked = pnl
    return t


def _bypass_trade(action, invested=500.0, shares=1000.0, pnl=0.0):
    """Record built bypassing __setattr__/__post_init__ (the
    unpickling-equivalent route the entry gates exist to close)."""
    t = object.__new__(ak.TradeExecutionResult)
    fields = {**_kwargs(action), "invested_capital": invested,
              "shares_acquired": shares, "skip_reason": None,
              "is_settled": False, "outcome_yes": None,
              "net_pnl_bid_marked": pnl, "liquidation_mark_vwap": None,
              "liquidation_fill_fraction": None,
              "liquidation_bid_as_of": None, "liquidation_bid_source": None,
              "kelly_fraction_sized": 0.0, "cluster_exposure_used": None,
              "cluster_exposure_released": False}
    for k, v in fields.items():
        object.__setattr__(t, k, v)
    return t


def _named_value_error(fn):
    """(raised, message) — True iff fn raises ValueError naming the task."""
    try:
        fn()
    except ValueError as e:
        return ("SF-BL-10823" in str(e) and "INVALID_TRADE_ACTION" in str(e)
                and "action" in str(e)), str(e)
    except Exception as e:  # noqa: BLE001
        return False, f"wrong exception {type(e).__name__}: {e}"
    return False, "no raise"


# ---- 1. Construction rejects every non-member -------------------------------

def test_construction_rejects_prereg_fixtures():
    for bad in ("buy_yes", "MAYBE", None):
        ok, msg = _named_value_error(
            lambda bad=bad: ak.TradeExecutionResult(**_kwargs(bad)))
        assert ok, f"action={bad!r}: {msg}"


def test_construction_rejects_extended_space():
    for bad in BAD_ACTIONS:
        ok, msg = _named_value_error(
            lambda bad=bad: ak.TradeExecutionResult(**_kwargs(bad)))
        assert ok, f"action={bad!r}: {msg}"


def test_construction_accepts_genuine_members():
    for good in ak.TradeAction:
        t = ak.TradeExecutionResult(**_kwargs(good))
        assert t.action is good


def test_helper_domain():
    for good in ak.TradeAction:
        assert ak._is_valid_trade_action(good)
    for bad in BAD_ACTIONS:
        assert not ak._is_valid_trade_action(bad), bad
    # The equality trap the helper exists for:
    assert "BUY_YES" == ak.TradeAction.BUY_YES
    assert not ak._is_valid_trade_action("BUY_YES")


# ---- 2. Assignment rejects; the stored value survives -----------------------

def test_assignment_rejects_and_preserves():
    for bad in BAD_ACTIONS:
        t = _trade()
        def _assign(t=t, bad=bad):
            t.action = bad
        ok, msg = _named_value_error(_assign)
        assert ok, f"assign {bad!r}: {msg}"
        assert t.action is ak.TradeAction.BUY_YES, \
            f"failed assignment overwrote action with {t.action!r}"


def test_assignment_accepts_genuine_members():
    t = _trade()
    t.action = ak.TradeAction.BUY_NO
    assert t.action is ak.TradeAction.BUY_NO
    t.action = ak.TradeAction.SKIP
    assert t.action is ak.TradeAction.SKIP


def test_dict_write_bypass_still_caught_at_gates():
    # Direct __dict__ writes bypass __setattr__ by construction; the
    # settle/scorer gates are the line that catches them (sections 3-4
    # exercise the same bypass via object.__setattr__).
    t = _trade()
    t.__dict__["action"] = "MAYBE"
    ok, _ = _named_value_error(
        lambda: ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1))
    assert ok


# ---- 3. Settle entry: bypass records can neither settle nor mark ------------

def test_settle_rejects_prereg_fixtures_win_and_loss():
    for bad in ("buy_yes", "MAYBE", None):
        for outcome in (1, 0):
            t = _bypass_trade(bad, pnl=SENTINEL_PNL)
            ok, msg = _named_value_error(
                lambda t=t, outcome=outcome: ak.settle_or_mark_trade(
                    t, is_settled=True, outcome_yes=outcome))
            assert ok, f"action={bad!r} outcome={outcome}: {msg}"
            assert t.net_pnl_bid_marked == SENTINEL_PNL
            assert t.is_settled is False and t.outcome_yes is None, \
                "raising settle flipped settlement state (SF-BL-10724)"


def test_settle_rejects_extended_space_both_branches():
    for bad in BAD_ACTIONS:
        t = _bypass_trade(bad, pnl=SENTINEL_PNL)
        ok, msg = _named_value_error(
            lambda t=t: ak.settle_or_mark_trade(
                t, is_settled=True, outcome_yes=1))
        assert ok, f"settled action={bad!r}: {msg}"
        assert t.net_pnl_bid_marked == SENTINEL_PNL
        t = _bypass_trade(bad, pnl=SENTINEL_PNL)
        ok, msg = _named_value_error(
            lambda t=t: ak.settle_or_mark_trade(t, is_settled=False))
        assert ok, f"mark action={bad!r}: {msg}"
        assert t.net_pnl_bid_marked == SENTINEL_PNL


def test_settle_rejects_zero_invested_bad_action():
    # Validation precedes the SKIP / zero-invested early return: even a
    # record that could book no PnL may not pass through settle.
    for bad in ("buy_yes", "MAYBE", None, "SKIP"):
        t = _bypass_trade(bad, invested=0.0, shares=0.0, pnl=SENTINEL_PNL)
        ok, msg = _named_value_error(
            lambda t=t: ak.settle_or_mark_trade(
                t, is_settled=True, outcome_yes=1))
        assert ok, f"zero-invested action={bad!r}: {msg}"
        assert t.net_pnl_bid_marked == SENTINEL_PNL


def test_pickled_bad_record_rejected_at_settle():
    t = _bypass_trade("buy_yes")
    t2 = pickle.loads(pickle.dumps(t))
    assert t2.action == "buy_yes"  # pickle preserves the bad value
    ok, _ = _named_value_error(
        lambda: ak.settle_or_mark_trade(t2, is_settled=True, outcome_yes=1))
    assert ok


# ---- 4. Scorer entry: bypass records cannot be scored as executed -----------

def _score(trades):
    return ak.evaluate_fleet_north_star(
        "fixture", trades, [(0.6, 1), (0.6, 0)], starting_capital=10000.0)


def test_scorer_rejects_prereg_fixtures():
    for bad in ("buy_yes", "MAYBE", None):
        t = _bypass_trade(bad, pnl=-500.0)
        object.__setattr__(t, "is_settled", True)
        ok, msg = _named_value_error(lambda t=t: _score([t]))
        assert ok, f"score action={bad!r}: {msg}"


def test_scorer_rejects_extended_space_and_mixed_books():
    good = _trade(pnl=100.0)
    for bad in BAD_ACTIONS:
        t = _bypass_trade(bad, pnl=-500.0)
        ok, msg = _named_value_error(lambda t=t: _score([t]))
        assert ok, f"score action={bad!r}: {msg}"
        t = _bypass_trade(bad, pnl=-500.0)
        ok, msg = _named_value_error(lambda t=t: _score([good, t]))
        assert ok, f"mixed book action={bad!r}: {msg}"


def test_scorer_rejects_plain_skip_string_with_capital():
    # Pre-fix a plain 'SKIP' string partitioned as skipped by equality,
    # silently vanishing a $500 record from the executed set.
    t = _bypass_trade("SKIP", pnl=-500.0)
    ok, _ = _named_value_error(lambda: _score([t]))
    assert ok


# ---- 5. Genuine records bit-identical ----------------------------------------

def test_genuine_settle_exact():
    cases = ((ak.TradeAction.BUY_YES, 1, 500.0),
             (ak.TradeAction.BUY_YES, 0, -500.0),
             (ak.TradeAction.BUY_NO, 0, 500.0),
             (ak.TradeAction.BUY_NO, 1, -500.0))
    for action, outcome, expect in cases:
        t = ak.settle_or_mark_trade(
            _trade(action=action), is_settled=True, outcome_yes=outcome)
        assert t.net_pnl_bid_marked == expect, (action, outcome, t)
    for bid, pnl in ((0.5, 0.0), (0.25, -250.0), (0.75, 250.0),
                     (0.0, -500.0), (1.0, 500.0)):
        t = _trade()
        object.__setattr__  # (no-op reference; bid set via constructor below)
        t = ak.TradeExecutionResult(
            **{**_kwargs(ak.TradeAction.BUY_YES),
               "liquidation_bid_price": bid})
        t = ak.settle_or_mark_trade(t, is_settled=False)
        assert t.net_pnl_bid_marked == pnl, (bid, t.net_pnl_bid_marked)
    skip = _trade(action=ak.TradeAction.SKIP, invested=0.0, shares=0.0)
    assert ak.settle_or_mark_trade(
        skip, is_settled=True, outcome_yes=1).net_pnl_bid_marked == 0.0


def test_genuine_book_scores_unchanged():
    win = ak.settle_or_mark_trade(_trade(), is_settled=True, outcome_yes=1)
    loss = ak.settle_or_mark_trade(
        _trade(action=ak.TradeAction.BUY_NO), is_settled=True, outcome_yes=1)
    skip = _trade(action=ak.TradeAction.SKIP, invested=0.0, shares=0.0)
    sc = _score([win, loss, skip])
    assert sc.executed_trades == 2 and sc.skipped_opportunities == 1, sc
    assert sc.total_net_pnl == 0.0 and sc.cwr_bid_marked == 0.0, sc
    single = _score([ak.settle_or_mark_trade(
        _trade(), is_settled=True, outcome_yes=1)])
    assert single.executed_trades == 1 and single.total_net_pnl == 500.0
    assert single.cwr_bid_marked == 1.0, single


def test_evaluate_produced_records_pass_all_gates():
    # The engine's own producer path: evaluate -> settle -> score.
    book = ak.OrderBookSnapshot(
        market_id="M1", title="t", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C", est_days_to_resolution=10.0,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100000)],
        fee_rate=0.01)
    cluster = {}
    r = ak.evaluate_and_size_opportunity(
        book=book, calibrated_prob_yes=0.80, inter_agent_dispersion=0.05,
        portfolio_equity=10000.0, cluster_exposure_used=cluster)
    assert r.action is ak.TradeAction.BUY_YES, r
    r = ak.settle_or_mark_trade(r, is_settled=True, outcome_yes=1)
    sc = _score([r])
    assert sc.executed_trades == 1, sc


# ---- 6. Benchmark: 8-market fixture output unchanged -------------------------

def test_benchmark_reproduces_nstar():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    scores = [float(x) for x in
              re.findall(r'"north_star_score": (-?[\d.]+)', buf.getvalue())]
    assert len(scores) == 2, scores
    # Current-tree baseline, verified byte-identical pre/post patch
    # (full-output diff in the task work dir notes).
    assert abs(scores[0] - (-0.1611)) < 5e-5, scores
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
