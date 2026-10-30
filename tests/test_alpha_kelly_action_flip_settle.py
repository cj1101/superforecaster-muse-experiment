"""
test_alpha_kelly_action_flip_settle.py — SF-BL-34997 verification.

Pre-registered done criterion: "Success if settle/mark detects an
action that no longer matches the side the position's exec_price /
shares / stored bid were built from (or the record pins its entry
side immutably) on flipped-settle and flipped-mark fixtures, while
unflipped lifecycles stay bit-identical; kill if a caller census
shows action is never reassigned after evaluate under a tested
invariant."

Kill NOT met (CALLER_CENSUS.md): no invariant pins action after
evaluate; the agreed SF-BL-10823 suite itself pins member-to-member
reassignment as accepted behaviour, so the fix pins a separate
entry_action at construction (immutable through the normal
assignment path) and detects the mismatch at the settle gate
(unscoped) and the scorer gate (executed records only).

Pre-fix reproduction (repro.py / repro_prefix.log): the prereg
fixture (BUY_YES, $1,025.25 invested, 2,030.20 shares, stored YES
bid 0.48) flipped to BUY_NO settled at outcome_yes=0 booking
+$1,004.95, marked at the YES bid under a BUY_NO claim, and rewrote
its stored bid to the NO bid 0.50 on the exit-book path.

Run:  python3 test_alpha_kelly_action_flip_settle.py
No pytest dependency: test_* functions with assert + __main__ runner.
"""
import contextlib
import io
import os
import pickle
import re
import sys

sys.path.insert(0, os.path.expanduser(
    "~/workspace/superforecaster/superforecaster"))
import alpha_kelly as ak  # noqa: E402

PASS = []
SENTINEL_PNL = 777.0

BOOK = ak.OrderBookSnapshot(
    market_id="M1", title="prereg fixture",
    domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C1",
    est_days_to_resolution=10.0,
    yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000)],
    yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100000)],
    fee_rate=0.01)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _evaluate(p=0.65, cluster=None):
    return ak.evaluate_and_size_opportunity(
        book=BOOK, calibrated_prob_yes=p, inter_agent_dispersion=0.05,
        portfolio_equity=10000.0,
        cluster_exposure_used={} if cluster is None else cluster)


def _kwargs(action, **over):
    kw = dict(
        market_id="FIXTURE-34997", title="fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C34997",
        action=action, calibrated_prob_yes=0.65, market_mid_prob=0.49,
        exec_price=0.505, liquidation_bid_price=0.48,
        kl_divergence_nats=0.1, expected_log_growth=0.01,
        kelly_fraction_raw=0.29, allocated_fraction=0.102525,
        invested_capital=1025.25, shares_acquired=2030.20,
        est_days_to_resolution=10.0)
    kw.update(over)
    return kw


def _named(fn, token="MISMATCHED_TRADE_ACTION"):
    """(ok, message): True iff fn raises ValueError naming SF-BL-34997
    and the given token."""
    try:
        fn()
    except ValueError as e:
        return ("SF-BL-34997" in str(e) and token in str(e)), str(e)
    except Exception as e:  # noqa: BLE001
        return False, f"wrong exception {type(e).__name__}: {e}"
    return False, "no raise"


def _flip(trade, new_action):
    trade.action = new_action  # genuine member: SF-BL-10823 guard passes
    assert trade.action is new_action
    return trade


# ---- 1. The pin at construction ---------------------------------------------

def test_evaluate_pins_entry_side():
    r = _evaluate()
    assert r.action is ak.TradeAction.BUY_YES
    assert r.entry_action is ak.TradeAction.BUY_YES, r.entry_action
    rn = _evaluate(p=0.20)
    assert rn.action is ak.TradeAction.BUY_NO
    assert rn.entry_action is ak.TradeAction.BUY_NO


def test_manual_construction_pins_entry_side():
    for action in ak.TradeAction:
        t = ak.TradeExecutionResult(**_kwargs(action))
        assert t.entry_action is action, (action, t.entry_action)


def test_explicit_agreeing_pin_kept_mismatched_pin_rejected():
    t = ak.TradeExecutionResult(**_kwargs(
        ak.TradeAction.BUY_YES, entry_action=ak.TradeAction.BUY_YES))
    assert t.entry_action is ak.TradeAction.BUY_YES
    ok, msg = _named(lambda: ak.TradeExecutionResult(**_kwargs(
        ak.TradeAction.BUY_YES, entry_action=ak.TradeAction.BUY_NO)))
    assert ok, msg


def test_pin_immutable_through_assignment():
    t = ak.TradeExecutionResult(**_kwargs(ak.TradeAction.BUY_YES))
    t.entry_action = ak.TradeAction.BUY_YES  # same member: no-op success
    assert t.entry_action is ak.TradeAction.BUY_YES
    ok, msg = _named(lambda: setattr(t, "entry_action",
                                     ak.TradeAction.BUY_NO),
                     token="ENTRY_ACTION_PINNED")
    assert ok, msg
    assert t.entry_action is ak.TradeAction.BUY_YES, "pin was rewritten"
    ok, msg = _named(lambda: setattr(t, "entry_action", None),
                     token="ENTRY_ACTION_PINNED")
    assert ok, msg
    ok, msg = _named(lambda: setattr(t, "entry_action", "BUY_NO"),
                     token="INVALID_ENTRY_ACTION")
    assert ok, msg
    assert t.entry_action is ak.TradeAction.BUY_YES


def test_action_reassignment_still_accepted():
    # The SF-BL-10823 agreed contract this fix must NOT break:
    # assigning another genuine member to `action` itself succeeds —
    # the flip is detected at the gates, not forbidden at assignment.
    t = ak.TradeExecutionResult(**_kwargs(ak.TradeAction.BUY_YES))
    t.action = ak.TradeAction.BUY_NO
    assert t.action is ak.TradeAction.BUY_NO
    t.action = ak.TradeAction.SKIP
    assert t.action is ak.TradeAction.SKIP


def test_pin_excluded_from_repr_and_equality():
    a = ak.TradeExecutionResult(**_kwargs(ak.TradeAction.BUY_YES))
    b = ak.TradeExecutionResult(**_kwargs(ak.TradeAction.BUY_YES))
    assert a == b
    assert "entry_action" not in repr(a)


# ---- 2. Flipped settle: the prereg fixture -----------------------------------

def test_flipped_settle_prereg_fixture_rejected_untouched():
    cluster = {}
    t = _evaluate(cluster=cluster)
    assert abs(t.invested_capital - 1025.25) < 0.01
    assert abs(t.shares_acquired - 2030.20) < 0.01
    assert t.liquidation_bid_price == 0.48
    charged = dict(cluster)
    _flip(t, ak.TradeAction.BUY_NO)
    t.net_pnl_bid_marked = SENTINEL_PNL
    ok, msg = _named(lambda: ak.settle_or_mark_trade(
        t, is_settled=True, outcome_yes=0))
    assert ok, msg
    # SF-BL-10724 ordering: the raising settle wrote nothing.
    assert t.net_pnl_bid_marked == SENTINEL_PNL
    assert t.is_settled is True and t.outcome_yes is None  # evaluate defaults
    assert t.liquidation_bid_price == 0.48
    assert t.cluster_exposure_released is False
    assert cluster == charged, "rejected settle released cluster exposure"
    # The honest path is still available afterwards: flip back and
    # settle — the YES position loses its stake at outcome_yes=0.
    t.action = ak.TradeAction.BUY_YES
    out = ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=0)
    assert abs(out.net_pnl_bid_marked - (-1025.25)) < 0.01, out
    assert cluster.get("C1", 0.0) == 0.0, cluster


def test_flipped_settle_all_flip_pairs_rejected():
    pairs = [(0.65, ak.TradeAction.BUY_NO, 0),
             (0.65, ak.TradeAction.BUY_NO, 1),
             (0.20, ak.TradeAction.BUY_YES, 1),
             (0.20, ak.TradeAction.BUY_YES, 0),
             (0.65, ak.TradeAction.SKIP, 0),
             (0.65, ak.TradeAction.SKIP, 1)]
    for p, new_action, outcome in pairs:
        t = _evaluate(p=p)
        _flip(t, new_action)
        t.net_pnl_bid_marked = SENTINEL_PNL
        ok, msg = _named(lambda t=t, outcome=outcome:
                         ak.settle_or_mark_trade(
                             t, is_settled=True, outcome_yes=outcome))
        assert ok, (p, new_action, outcome, msg)
        assert t.net_pnl_bid_marked == SENTINEL_PNL


def test_skip_record_flipped_to_executed_rejected():
    # Gate precedes the SKIP / zero-invested early return: a SKIP
    # record (entry_action SKIP, zero invested) flipped to BUY_YES
    # must raise, not pass through the early return booking 0.0.
    t = ak.TradeExecutionResult(**_kwargs(
        ak.TradeAction.SKIP, invested_capital=0.0, shares_acquired=0.0,
        exec_price=0.0, liquidation_bid_price=0.0))
    _flip(t, ak.TradeAction.BUY_YES)
    ok, msg = _named(lambda: ak.settle_or_mark_trade(
        t, is_settled=True, outcome_yes=1))
    assert ok, msg


# ---- 3. Flipped marks ---------------------------------------------------------

def test_flipped_legacy_mark_rejected():
    t = _flip(_evaluate(), ak.TradeAction.BUY_NO)
    t.net_pnl_bid_marked = SENTINEL_PNL
    ok, msg = _named(lambda: ak.settle_or_mark_trade(t, is_settled=False))
    assert ok, msg
    assert t.net_pnl_bid_marked == SENTINEL_PNL
    assert t.liquidation_bid_price == 0.48


def test_flipped_exit_book_mark_rejected_bid_untouched():
    t = _flip(_evaluate(), ak.TradeAction.BUY_NO)
    ok, msg = _named(lambda: ak.settle_or_mark_trade(
        t, is_settled=False, exit_book=BOOK))
    assert ok, msg
    # Pre-fix the NO-ladder walk rewrote the stored bid to 0.50.
    assert t.liquidation_bid_price == 0.48
    assert t.liquidation_mark_vwap is None
    assert t.liquidation_bid_source == ak.BID_SOURCE_ENTRY_BOOK


def test_mirror_flipped_mark_rejected():
    t = _flip(_evaluate(p=0.20), ak.TradeAction.BUY_YES)
    ok, msg = _named(lambda: ak.settle_or_mark_trade(t, is_settled=False))
    assert ok, msg


# ---- 4. Bypass flips are caught at the gates ---------------------------------

def test_bypass_action_flip_caught_at_settle():
    for mode in ("object.__setattr__", "__dict__"):
        t = _evaluate()
        if mode == "object.__setattr__":
            object.__setattr__(t, "action", ak.TradeAction.BUY_NO)
        else:
            t.__dict__["action"] = ak.TradeAction.BUY_NO
        ok, msg = _named(lambda t=t: ak.settle_or_mark_trade(
            t, is_settled=True, outcome_yes=0))
        assert ok, (mode, msg)
        t2 = _evaluate()
        if mode == "object.__setattr__":
            object.__setattr__(t2, "action", ak.TradeAction.BUY_NO)
        else:
            t2.__dict__["action"] = ak.TradeAction.BUY_NO
        ok, msg = _named(lambda t2=t2: ak.settle_or_mark_trade(
            t2, is_settled=False))
        assert ok, (mode, msg)


def test_pickled_flipped_trade_caught():
    t = _flip(_evaluate(), ak.TradeAction.BUY_NO)
    t2 = pickle.loads(pickle.dumps(t))
    assert t2.action is ak.TradeAction.BUY_NO
    assert t2.entry_action is ak.TradeAction.BUY_YES
    ok, msg = _named(lambda: ak.settle_or_mark_trade(
        t2, is_settled=True, outcome_yes=0))
    assert ok, msg


def test_bypass_nonmember_pin_caught():
    t = _evaluate()
    object.__setattr__(t, "entry_action", "BUY_YES")  # plain string pin
    ok, msg = _named(lambda: ak.settle_or_mark_trade(
        t, is_settled=True, outcome_yes=1))
    assert ok, msg


def test_unpinned_bypass_record_settles_legacy():
    # A bypass-constructed record that never carried entry_action is
    # unpinned: no entry side was recorded, so it settles exactly as
    # pre-fix (the documented scope limit, as for other gates).
    t = object.__new__(ak.TradeExecutionResult)
    fields = {**_kwargs(ak.TradeAction.BUY_YES),
              "skip_reason": None, "is_settled": False,
              "outcome_yes": None, "net_pnl_bid_marked": 0.0,
              "liquidation_mark_vwap": None,
              "liquidation_fill_fraction": None,
              "liquidation_bid_as_of": None, "liquidation_bid_source": None,
              "kelly_fraction_sized": 0.0, "cluster_exposure_used": None,
              "cluster_exposure_released": False,
              "entry_factor_cluster": "C34997",
              "entry_cluster_ledger": None}
    for k, v in fields.items():
        object.__setattr__(t, k, v)
    assert getattr(t, "entry_action", None) is None
    out = ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    assert abs(out.net_pnl_bid_marked - 1004.95) < 0.01, out


# ---- 5. Scorer gate ------------------------------------------------------------

def _score(trades):
    return ak.evaluate_fleet_north_star(
        "fixture", trades, [(0.6, 1), (0.6, 0)], starting_capital=10000.0)


def test_scorer_rejects_flipped_executed_trade():
    t = ak.settle_or_mark_trade(_evaluate(), is_settled=True, outcome_yes=1)
    assert abs(t.net_pnl_bid_marked - 1004.95) < 0.01
    _flip(t, ak.TradeAction.BUY_NO)  # PnL booked as YES, record now claims NO
    ok, msg = _named(lambda: _score([t]))
    assert ok, msg
    # SKIP -> executed flip is also caught at the scorer.
    s = ak.TradeExecutionResult(**_kwargs(
        ak.TradeAction.SKIP, invested_capital=0.0, shares_acquired=0.0))
    _flip(s, ak.TradeAction.BUY_YES)
    ok, msg = _named(lambda: _score([s]))
    assert ok, msg


def test_scorer_allows_record_reassigned_to_skip():
    # SF-BL-10767 precedent (bid_freshness suite): a BUY_YES record
    # reassigned to SKIP partitions as skipped; its side is never
    # read by the scorer, so the gate (scoped to executed records)
    # does not raise.
    t = _evaluate()
    t.action = ak.TradeAction.SKIP
    t.invested_capital = 0.0
    sc = _score([t])
    assert sc.skipped_opportunities == 1 and sc.executed_trades == 0, sc


def test_scorer_unflipped_book_unchanged():
    win = ak.settle_or_mark_trade(_evaluate(), is_settled=True, outcome_yes=1)
    loss = ak.settle_or_mark_trade(
        _evaluate(p=0.20), is_settled=True, outcome_yes=1)
    sc = _score([win, loss])
    assert sc.executed_trades == 2, sc
    assert abs(sc.total_net_pnl - (1004.95 - 1500.00)) < 0.02, sc


# ---- 6. Unflipped lifecycles bit-identical ------------------------------------

def test_unflipped_settle_exact_values():
    cases = [(0.65, 1, 1004.95), (0.65, 0, -1025.25),
             (0.20, 0, 1356.05), (0.20, 1, -1500.00)]
    for p, outcome, expect in cases:
        t = ak.settle_or_mark_trade(
            _evaluate(p=p), is_settled=True, outcome_yes=outcome)
        assert abs(t.net_pnl_bid_marked - expect) < 0.01, (
            p, outcome, t.net_pnl_bid_marked, expect)


def test_unflipped_marks_exact_values():
    t = ak.settle_or_mark_trade(_evaluate(), is_settled=False)
    assert abs(t.net_pnl_bid_marked - (-50.76)) < 0.01, t.net_pnl_bid_marked
    assert t.liquidation_bid_price == 0.48
    t = ak.settle_or_mark_trade(_evaluate(), is_settled=False, exit_book=BOOK)
    assert abs(t.net_pnl_bid_marked - (-50.76)) < 0.01, t.net_pnl_bid_marked
    assert t.liquidation_bid_price == 0.48  # full-depth walk: VWAP == top bid
    t = ak.settle_or_mark_trade(_evaluate(p=0.20), is_settled=False)
    assert abs(t.net_pnl_bid_marked - (-71.98)) < 0.5, t.net_pnl_bid_marked


def test_evaluate_record_fields_unchanged_apart_from_pin():
    r = _evaluate()
    assert r.action is ak.TradeAction.BUY_YES
    assert abs(r.invested_capital - 1025.25) < 0.01
    assert abs(r.shares_acquired - 2030.20) < 0.01
    assert r.exec_price == 0.505 and r.liquidation_bid_price == 0.48
    assert r.skip_reason is None


# ---- 7. Benchmark ---------------------------------------------------------------

def test_benchmark_reproduces():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    scores = [float(x) for x in
              re.findall(r'"north_star_score": (-?[\d.]+)', buf.getvalue())]
    assert len(scores) == 2, scores
    # Current-tree baseline, verified identical with the SF-BL-34997
    # gates monkeypatched out (benchmark_isolation.log in the task
    # work dir): the pin/gates change no benchmark number.
    assert abs(scores[0] - BENCH_0) < 5e-5, scores
    assert abs(scores[1] - BENCH_1) < 5e-5, scores


BENCH_0 = -0.1611
BENCH_1 = -0.1499

ALL_TESTS = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    for fn in ALL_TESTS:
        try:
            fn()
            check(fn.__name__, True)
        except AssertionError as e:
            check(fn.__name__, False, str(e))
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
