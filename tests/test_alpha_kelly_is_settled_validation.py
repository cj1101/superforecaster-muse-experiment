"""
test_alpha_kelly_is_settled_validation.py — SF-BL-11536 verification.

Pre-registered done criterion: "evaluate_fleet_north_star (or
TradeExecutionResult construction/assignment) must reject any
non-bool is_settled on an executed trade with a named error,
fixtured at is_settled='False' on the two-trade book against the
100.0 win-rate control, with genuine bool books bit-identical; kill
if the string-marked open trade still enters the settled denominator
or a caller census shows is_settled is only ever written by
settle_or_mark_trade."

Kill NOT met (CALLER_CENSUS.md): the dataclass default, direct
construction kwargs in the test suites / wave3 recompute, direct
post-construction assignment in the repo's own tests and fleet
scripts, and mutation/unpickling all write the field outside
settle_or_mark_trade.

Pre-fix reproduction (repro.py / repro_prefix.txt): two-trade book
(settled winner +$100, open trade -$100 mark) — control
is_settled=False scores settlement win rate 100.0; the string
'False' scores 50.0 because bare truthiness counts the open trade
as settled.

This suite pins the fix at every layer the prereg names:
construction, assignment, scorer entry (incl. bypass-built and
pickled records) — plus genuine-bool bit-identity (scorecard
controls and the full benchmark), the documented SKIP scope of the
scorer gate, and the settle write-path side effect (a non-bool
settle argument raises at the is_settled assignment, first field
written, leaving the trade untouched).

Run:  python3 test_alpha_kelly_is_settled_validation.py
No pytest dependency: test_* functions with assert + __main__ runner.
"""
import contextlib
import io
import os
import pickle
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster"))
from superforecaster import alpha_kelly as ak  # noqa: E402

RESOLVED = [(0.6, 1), (0.6, 0)]

# Prereg fixture first, then the extended non-bool space: truthy
# strings/numbers/containers, falsy non-bools (which pre-fix read as
# open — the right partition for the wrong reason), and None.
BAD_VALUES = ("False", "True", "false", "0", "1", "",
              1, 0, 1.0, 0.0, -1, None, [], [False], {}, [True])


def _trade(mid="M1", is_settled=True, pnl=100.0,
           action=ak.TradeAction.BUY_YES, invested=500.0):
    return ak.TradeExecutionResult(
        market_id=mid, title=f"Market {mid}",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C1",
        action=action, calibrated_prob_yes=0.6, market_mid_prob=0.55,
        exec_price=0.50, liquidation_bid_price=0.40,
        kl_divergence_nats=0.1, expected_log_growth=0.01,
        kelly_fraction_raw=0.1, allocated_fraction=0.05,
        invested_capital=invested, shares_acquired=1000.0,
        est_days_to_resolution=30.0,
        is_settled=is_settled, outcome_yes=None,
        net_pnl_bid_marked=pnl)


def _book(open_is_settled):
    """Prereg two-trade book: settled winner + open trade (-100)."""
    return [_trade("WIN", True, 100.0),
            _trade("OPEN", open_is_settled, -100.0)]


def _score(trades):
    return ak.evaluate_fleet_north_star(
        "fixture", trades, RESOLVED, starting_capital=10_000.0)


def _bypass_trade(mid, is_settled, pnl, action=ak.TradeAction.BUY_YES):
    """Record built bypassing __setattr__/__post_init__ (the
    unpickling-equivalent route the scorer gate exists to close)."""
    t = _trade(mid, True, pnl, action=action)
    object.__setattr__(t, "is_settled", is_settled)
    return t


def _named(fn):
    """True iff fn raises ValueError naming SF-BL-11536's token."""
    try:
        fn()
    except ValueError as e:
        return "INVALID_IS_SETTLED" in str(e) and "SF-BL-11536" in str(e)
    except Exception:  # noqa: BLE001
        return False
    return False


# ---- 1. Construction ---------------------------------------------------------

def test_construction_rejects_prereg_fixture():
    assert _named(lambda: _trade("X", "False", -100.0))


def test_construction_rejects_extended_space():
    for bad in BAD_VALUES:
        assert _named(lambda bad=bad: _trade("X", bad, -100.0)), bad


def test_construction_accepts_genuine_bools_and_default():
    assert _trade("A", True, 0.0).is_settled is True
    assert _trade("B", False, 0.0).is_settled is False
    t = ak.TradeExecutionResult(
        market_id="D", title="D", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C1", action=ak.TradeAction.BUY_YES,
        calibrated_prob_yes=0.6, market_mid_prob=0.55, exec_price=0.5,
        liquidation_bid_price=0.4, kl_divergence_nats=0.1,
        expected_log_growth=0.01, kelly_fraction_raw=0.1,
        allocated_fraction=0.05, invested_capital=500.0,
        shares_acquired=1000.0, est_days_to_resolution=30.0)
    assert t.is_settled is True  # pre-existing default, unchanged


# ---- 2. Assignment -----------------------------------------------------------

def test_assignment_rejects_and_preserves():
    for bad in BAD_VALUES:
        t = _trade("MUT", False, -100.0)
        assert _named(lambda bad=bad, t=t: setattr(t, "is_settled", bad)), bad
        assert t.is_settled is False, bad


def test_assignment_accepts_genuine_bools():
    t = _trade("MUT", False, -100.0)
    t.is_settled = True
    assert t.is_settled is True
    t.is_settled = False
    assert t.is_settled is False


# ---- 3. Scorer: prereg fixture ----------------------------------------------

def test_scorer_control_win_rate_100():
    card = _score(_book(False))
    assert card.settlement_win_rate_pct == 100.0, card
    assert card.executed_trades == 2


def test_scorer_genuine_settled_loss_win_rate_50_is_legitimate():
    # A genuine 50.0 exists: settled win + settled loss. Only the
    # string-forged 50.0 (open trade in the denominator) is rejected.
    card = _score([_trade("W", True, 100.0), _trade("L", True, -100.0)])
    assert card.settlement_win_rate_pct == 50.0, card


def test_scorer_rejects_bypass_prereg_fixture():
    book = [_trade("WIN", True, 100.0),
            _bypass_trade("OPEN", "False", -100.0)]
    assert _named(lambda: _score(book))


def test_scorer_rejects_bypass_extended_space():
    for bad in BAD_VALUES:
        book = [_trade("WIN", True, 100.0),
                _bypass_trade("OPEN", bad, -100.0)]
        assert _named(lambda book=book: _score(book)), bad


def test_scorer_rejects_solo_and_mixed_and_buy_no():
    assert _named(lambda: _score([_bypass_trade("S", "False", -100.0)]))
    assert _named(lambda: _score([
        _trade("OK", False, 10.0),
        _bypass_trade("BAD", 1, 10.0, action=ak.TradeAction.BUY_NO)]))


def test_scorer_rejects_pickled_bad_record():
    t = _trade("PKL", False, -100.0)
    t.__dict__["is_settled"] = "False"  # forge past __setattr__
    t2 = pickle.loads(pickle.dumps(t))
    assert t2.is_settled == "False"
    book = [_trade("WIN", True, 100.0), t2]
    assert _named(lambda: _score(book))


def test_scorer_skip_scope_documented():
    # Documented scope (SF-BL-9475 precedent): the scorer gate covers
    # executed (non-SKIP) trades — the only records whose is_settled
    # either scorer partition reads. A bypass SKIP record's field is
    # never read, so the book still scores; construction/assignment
    # of such a record through the normal path raises (tests 1-2).
    skip = _bypass_trade("SKP", "False", 0.0, action=ak.TradeAction.SKIP)
    object.__setattr__(skip, "invested_capital", 0.0)
    card = _score([_trade("WIN", True, 100.0), skip])
    assert card.skipped_opportunities == 1, card
    assert card.settlement_win_rate_pct == 100.0, card


def test_scorer_rejects_scored_at_open_partition_case():
    # The SF-BL-10767 open partition reads the same field: a bypass
    # record must raise even when scored_at is supplied.
    book = [_trade("WIN", True, 100.0),
            _bypass_trade("OPEN", "False", -100.0)]
    assert _named(lambda: ak.evaluate_fleet_north_star(
        "fixture", book, RESOLVED, starting_capital=10_000.0,
        scored_at=1_700_000_000.0))


# ---- 4. Genuine books bit-identical ------------------------------------------

def test_genuine_book_scorecard_exact():
    card = _score(_book(False))
    assert card.settlement_win_rate_pct == 100.0
    assert card.total_net_pnl == 0.0 and card.cwr_bid_marked == 0.0
    assert card.executed_trades == 2 and card.skipped_opportunities == 0
    card_all_settled = _score([_trade("W", True, 100.0),
                               _trade("L", True, -100.0)])
    assert card_all_settled.settlement_win_rate_pct == 50.0
    card_open_only = _score([_trade("O", False, -100.0)])
    assert card_open_only.settlement_win_rate_pct == 0.0


def test_benchmark_reproduces_identically():
    import re
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    scores = [float(x) for x in
              re.findall(r'"north_star_score": (-?[\d.]+)', buf.getvalue())]
    assert len(scores) == 2, scores
    # Exact identity is pinned by the pre/post full-output diff in
    # the work dir (benchmark_prefix.txt == benchmark_postfix.txt).


# ---- 5. Settle write path -----------------------------------------------------

def test_settle_genuine_bools_bit_identical():
    win = ak.settle_or_mark_trade(_trade("S", False, 0.0),
                                  is_settled=True, outcome_yes=1)
    assert win.is_settled is True and win.net_pnl_bid_marked == 500.0
    mark = ak.settle_or_mark_trade(_trade("S", True, 0.0), is_settled=False)
    assert mark.is_settled is False and mark.net_pnl_bid_marked == -100.0


def test_settle_nonbool_argument_raises_untouched():
    # Side effect of the record guard: settle stores its argument
    # through __setattr__, is_settled first on every path, so a
    # non-bool argument that REACHES an assignment raises the named
    # SF-BL-11536 error with the trade's (is_settled, outcome_yes,
    # net_pnl) byte-identical.
    for kwargs in (dict(is_settled="False", outcome_yes=1),
                   dict(is_settled="True", outcome_yes=1),
                   dict(is_settled=1, outcome_yes=1),
                   dict(is_settled=None),
                   dict(is_settled=0)):
        t = _trade("S", False, -100.0)
        before = (t.is_settled, t.outcome_yes, t.net_pnl_bid_marked)
        assert _named(lambda t=t, kwargs=kwargs:
                      ak.settle_or_mark_trade(t, **kwargs)), kwargs
        after = (t.is_settled, t.outcome_yes, t.net_pnl_bid_marked)
        assert before == after, (kwargs, before, after)


def test_settle_string_false_without_outcome_raises_9477_untouched():
    # SUPERSEDED by SF-BL-10822 (landed after this suite): settle now
    # validates its is_settled ARGUMENT at entry, before any branch
    # selection, so 'False' with no outcome raises the SF-BL-10822
    # INVALID_IS_SETTLED error (which also carries this task's
    # SF-BL-11536 token, the shared strict-bool domain) instead of
    # SF-BL-9477's outcome error — still a named ValueError, still
    # leaving the trade byte-identical (SF-BL-10724). The original
    # version of this test pinned the pre-10822 layered ordering and
    # was updated by the SF-BL-10822 executor for exactly this change;
    # every other assertion in this suite is unchanged.
    t = _trade("S", False, -100.0)
    before = (t.is_settled, t.outcome_yes, t.net_pnl_bid_marked)
    try:
        ak.settle_or_mark_trade(t, is_settled="False")
        raise AssertionError("no raise")
    except ValueError as e:
        assert "INVALID_IS_SETTLED" in str(e) and "SF-BL-10822" in str(e), e
    assert (t.is_settled, t.outcome_yes, t.net_pnl_bid_marked) == before


def test_evaluate_produced_records_pass_all_gates():
    book = ak.OrderBookSnapshot(
        market_id="EV", title="EV",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C1",
        est_days_to_resolution=30.0,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100000)],
        fee_rate=0.01)
    rec = ak.evaluate_and_size_opportunity(
        book=book, calibrated_prob_yes=0.80, inter_agent_dispersion=0.05,
        portfolio_equity=10_000.0, cluster_exposure_used={})
    assert rec.is_settled is True or rec.is_settled is False
    settled = ak.settle_or_mark_trade(rec, is_settled=True, outcome_yes=1)
    card = _score([settled])
    assert card.executed_trades == 1


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in fns:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failures}/{len(fns)} PASS")
    sys.exit(1 if failures else 0)
