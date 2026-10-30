"""
test_sf_bl_35210.py — SF-BL-35210 verification tests.

Pre-registered done criterion: "Success if the scorer raises or flags
any settled executed trade whose net_pnl_bid_marked != (shares -
invested) on a win or != -invested on a loss (pre-registered rounding
epsilon), fixtured at +100/-100 above, while settle-produced books
score bit-identically; kill if a producer census shows every scored
record's PnL is written only by settle_or_mark_trade from validated
position fields under a tested invariant."

Kill NOT met (PRODUCER_AUDIT.md): the field is an unvalidated
constructor keyword on a mutable dataclass; settle is not the sole
effective writer. Fix (PREREG.md, as amended): the FLAG branch of
"raises or flags" — evaluate_fleet_north_star reports
settled_pnl_mismatch_count / settled_pnl_mismatch_invested on the
scorecard for executed, settled trades with a strict-int outcome_yes
in {0, 1} whose booked PnL differs from the exact payout by more than
SETTLED_PNL_EPSILON = 0.005 (half a cent). The raise branch was
implemented first and broke at least six ratified neighbour suites
whose control fixtures pin non-payout settled PnLs for unrelated
gates; the flag branch (SF-BL-34866 precedent) keeps every
pre-existing scorecard field bit-identical for every book.

Pre-fix reproduction (repro.py, verified live before the patch;
shares=1000, invested=500, outcome_yes=1, resolved [(0.6,1),(0.6,0)]):
  BUY_YES win carrying +100 -> total +100.0, CWR 0.20, N* 0.4445, NO flag existed
  BUY_NO  loss carrying -100 -> total -100.0, CWR -0.20, NO flag existed
  settle-produced controls   -> +500 N* 2.2226 / -500 N* -2.2226
Post-fix the two fixtures score identically but flag count=1 /
invested=500.0, and the controls flag 0 / 0.0, bit-identical.

Run:  python3 test_sf_bl_35210.py  (exit 0 iff pass)
No pytest dependency: test_* functions with assert + __main__ runner.
"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster/superforecaster"))
import alpha_kelly as ak  # noqa: E402

PASS = []
RESOLVED = [(0.6, 1), (0.6, 0)]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _trade(**over):
    base = dict(
        market_id="FIXTURE-35210", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C35210",
        action=ak.TradeAction.BUY_YES,
        calibrated_prob_yes=0.60, market_mid_prob=0.50, exec_price=0.50,
        liquidation_bid_price=0.50, kl_divergence_nats=0.0,
        expected_log_growth=0.0, kelly_fraction_raw=0.10,
        allocated_fraction=0.05, invested_capital=500.0,
        shares_acquired=1000.0, est_days_to_resolution=10.0,
        is_settled=True, outcome_yes=1, net_pnl_bid_marked=100.0)
    base.update(over)
    return ak.TradeExecutionResult(**base)


def _score(trades, resolved=RESOLVED):
    return ak.evaluate_fleet_north_star("fixture", trades, resolved)


def _raises(trades, resolved=RESOLVED):
    try:
        ak.evaluate_fleet_north_star("fixture", trades, resolved)
    except ValueError as e:
        return str(e)
    return None


def _settled_via_settle(action, outcome, shares, invested):
    t = _trade(action=action, outcome_yes=None, is_settled=False,
               shares_acquired=shares, invested_capital=invested,
               net_pnl_bid_marked=0.0)
    return ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=outcome)


# ---- 1. Prereg headline fixtures: flagged, scores unchanged --------------------

def test_yes_win_carrying_100_flagged():
    sc = _score([_trade()])  # BUY_YES, outcome 1, +100; only possible +500
    assert sc.settled_pnl_mismatch_count == 1, sc
    assert sc.settled_pnl_mismatch_invested == 500.0, sc
    # Flag branch: pre-existing fields bit-identical to pre-fix.
    assert sc.total_net_pnl == 100.0 and sc.cwr_bid_marked == 0.2, sc
    assert sc.north_star_score == 0.4445, sc


def test_no_loss_carrying_minus_100_flagged():
    sc = _score([_trade(action=ak.TradeAction.BUY_NO,
                        net_pnl_bid_marked=-100.0)])
    assert sc.settled_pnl_mismatch_count == 1, sc
    assert sc.settled_pnl_mismatch_invested == 500.0, sc
    assert sc.total_net_pnl == -100.0 and sc.cwr_bid_marked == -0.2, sc


# ---- 2. All four side x outcome combinations ------------------------------------

def test_all_combinations_wrong_flagged_exact_clean():
    combos = [
        (ak.TradeAction.BUY_YES, 1, 500.0),    # YES win
        (ak.TradeAction.BUY_YES, 0, -500.0),   # YES loss
        (ak.TradeAction.BUY_NO, 0, 500.0),     # NO win
        (ak.TradeAction.BUY_NO, 1, -500.0),    # NO loss
    ]
    for action, outcome, expected in combos:
        inward = expected - (1.0 if expected > 0 else -1.0)
        for wrong in (0.0, inward, -expected, expected / 2.0):
            sc = _score([_trade(action=action, outcome_yes=outcome,
                                net_pnl_bid_marked=wrong)])
            assert sc.settled_pnl_mismatch_count == 1, (action, outcome, wrong, sc)
        sc = _score([_trade(action=action, outcome_yes=outcome,
                            net_pnl_bid_marked=expected)])
        assert sc.settled_pnl_mismatch_count == 0, sc
        assert sc.settled_pnl_mismatch_invested == 0.0, sc
        assert sc.total_net_pnl == expected, sc


def test_win_carrying_the_loss_figure_flagged():
    sc = _score([_trade(net_pnl_bid_marked=-500.0)])
    assert sc.settled_pnl_mismatch_count == 1, sc
    sc = _score([_trade(action=ak.TradeAction.BUY_NO, outcome_yes=1,
                        net_pnl_bid_marked=500.0)])
    assert sc.settled_pnl_mismatch_count == 1, sc


# ---- 3. Pre-registered epsilon ----------------------------------------------------

def test_epsilon_boundary():
    assert ak.SETTLED_PNL_EPSILON == 0.005
    # A win's expected PnL IS the envelope maximum (a loss's the
    # minimum), so the live epsilon range is INWARD: within half a
    # cent of the exact payout is clean; further inward flags.
    for ok_pnl in (500.0, 499.996, 499.999):
        sc = _score([_trade(net_pnl_bid_marked=ok_pnl)])
        assert sc.settled_pnl_mismatch_count == 0, (ok_pnl, sc)
    for ok_pnl in (-500.0, -499.996):
        sc = _score([_trade(action=ak.TradeAction.BUY_NO,
                            net_pnl_bid_marked=ok_pnl)])
        assert sc.settled_pnl_mismatch_count == 0, (ok_pnl, sc)
    for bad_pnl in (499.994, 499.99, 499.0):
        sc = _score([_trade(net_pnl_bid_marked=bad_pnl)])
        assert sc.settled_pnl_mismatch_count == 1, (bad_pnl, sc)
    # Outward deviations cross the envelope bound itself and keep
    # SF-BL-34956's raising identity (its dust epsilon fires first).
    msg = _raises([_trade(net_pnl_bid_marked=500.006)])
    assert msg is not None and "SF-BL-34956" in msg, msg


def test_cent_rounded_booked_pnl_clean():
    t = _trade(shares_acquired=1267.80, net_pnl_bid_marked=767.80)
    sc = _score([t])
    assert sc.settled_pnl_mismatch_count == 0, sc
    assert sc.total_net_pnl == 767.80, sc


# ---- 4. Settle-produced books bit-identical, flag 0 --------------------------------

def test_settle_produced_headline_controls_bit_identical():
    yes = _settled_via_settle(ak.TradeAction.BUY_YES, 1, 1000.0, 500.0)
    assert yes.net_pnl_bid_marked == 500.0
    sc = _score([yes])
    assert sc.total_net_pnl == 500.0 and sc.cwr_bid_marked == 1.0, sc
    assert sc.north_star_score == 2.2226, sc  # pre-fix value (repro.py)
    assert sc.settled_pnl_mismatch_count == 0, sc
    assert sc.settled_pnl_mismatch_invested == 0.0, sc
    no = _settled_via_settle(ak.TradeAction.BUY_NO, 1, 1000.0, 500.0)
    assert no.net_pnl_bid_marked == -500.0
    sc = _score([no])
    assert sc.total_net_pnl == -500.0 and sc.cwr_bid_marked == -1.0, sc
    assert sc.north_star_score == -2.2226, sc  # pre-fix value (repro.py)
    assert sc.settled_pnl_mismatch_count == 0, sc


def test_settle_produced_mixed_book_scores_exact_totals():
    book = [
        _settled_via_settle(ak.TradeAction.BUY_YES, 1, 1267.80, 500.0),
        _settled_via_settle(ak.TradeAction.BUY_NO, 0, 800.0, 400.0),
        _settled_via_settle(ak.TradeAction.BUY_YES, 0, 500.0, 250.0),
        _settled_via_settle(ak.TradeAction.BUY_NO, 1, 250.0, 100.0),
    ]
    pnls = [t.net_pnl_bid_marked for t in book]
    assert pnls == [767.80, 400.0, -250.0, -100.0], pnls
    sc = _score(book)
    assert sc.total_net_pnl == 817.80, sc
    assert sc.cwr_bid_marked == round(817.80 / 1250.0, 4), sc
    assert sc.executed_trades == 4, sc
    assert sc.settled_pnl_mismatch_count == 0, sc


def test_post_settlement_mutation_flagged():
    t = _settled_via_settle(ak.TradeAction.BUY_YES, 1, 1000.0, 500.0)
    t.net_pnl_bid_marked = 100.0  # edited apart from the booked payout
    assert _score([t]).settled_pnl_mismatch_count == 1
    t2 = _settled_via_settle(ak.TradeAction.BUY_YES, 1, 1000.0, 500.0)
    t2.shares_acquired = 2000.0  # position edited after settlement
    assert _score([t2]).settled_pnl_mismatch_count == 1


# ---- 5. Scope: open / SKIP / unclassifiable not counted ------------------------------

def test_open_trade_bid_mark_not_counted():
    # An open trade's +100 is a bid mark (bid 0.60 on 1000 shares),
    # not a settlement payout: not counted, scores exactly as pre-fix.
    t = _trade(is_settled=False, outcome_yes=None,
               liquidation_bid_price=0.60, net_pnl_bid_marked=100.0)
    sc = _score([t])
    assert sc.settled_pnl_mismatch_count == 0, sc
    assert sc.total_net_pnl == 100.0 and sc.cwr_bid_marked == 0.2, sc
    assert sc.north_star_score == 0.4445, sc


def test_skip_records_not_counted():
    skip = _trade(action=ak.TradeAction.SKIP, invested_capital=0.0,
                  net_pnl_bid_marked=123.0, skip_reason="FIXTURE")
    sc = _score([skip])
    assert sc.executed_trades == 0 and sc.skipped_opportunities == 1, sc
    assert sc.settled_pnl_mismatch_count == 0, sc
    assert sc.north_star_score == 0.0, sc


def test_settled_without_valid_outcome_out_of_scope():
    # Documented residual (PREREG.md): no valid outcome => the trade
    # cannot be classified win/loss, so this check does not count it.
    t = _trade(outcome_yes=None, net_pnl_bid_marked=100.0)
    sc = _score([t])
    assert sc.settled_pnl_mismatch_count == 0, sc
    assert sc.total_net_pnl == 100.0, sc


def test_mixed_book_counts_only_the_bad_settled():
    good = _settled_via_settle(ak.TradeAction.BUY_YES, 1, 1000.0, 500.0)
    good.market_id = "GOOD"
    open_t = _trade(market_id="OPEN", is_settled=False, outcome_yes=None,
                    liquidation_bid_price=0.60, net_pnl_bid_marked=100.0)
    bad = _trade(market_id="BAD", invested_capital=250.0,
                 shares_acquired=500.0, net_pnl_bid_marked=10.0)
    sc = _score([good, open_t, bad])
    assert sc.settled_pnl_mismatch_count == 1, sc
    assert sc.settled_pnl_mismatch_invested == 250.0, sc
    sc2 = _score([bad, good, open_t])
    assert sc2.settled_pnl_mismatch_count == 1, sc2


# ---- 6. Ordering: earlier gates keep their raising identities -------------------------

def test_nonfinite_pnl_keeps_9475_identity():
    msg = _raises([_trade(net_pnl_bid_marked=float("nan"))])
    assert msg is not None and "SF-BL-9475" in msg, msg


def test_nonpositive_invested_keeps_11304_identity():
    msg = _raises([_trade(invested_capital=-100.0, net_pnl_bid_marked=50.0)])
    assert msg is not None and "SF-BL-11304" in msg, msg


def test_out_of_envelope_keeps_34956_identity():
    msg = _raises([_trade(shares_acquired=2000.0, invested_capital=1000.0,
                          net_pnl_bid_marked=-50000.0)])
    assert msg is not None and "SF-BL-34956" in msg, msg


def test_empty_book_unchanged():
    sc = _score([])
    assert sc.north_star_score == 0.0 and sc.executed_trades == 0, sc
    assert sc.settled_pnl_mismatch_count == 0, sc
    assert sc.settled_pnl_mismatch_invested == 0.0, sc


def test_scorecard_defaults_keep_direct_construction_working():
    import inspect
    fields = inspect.signature(ak.NorthStarScorecard).parameters
    assert fields["settled_pnl_mismatch_count"].default == 0
    assert fields["settled_pnl_mismatch_invested"].default == 0.0


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    for name, fn in fns:
        try:
            fn()
            check(name, True)
        except AssertionError as e:
            check(name, False, str(e))
    print(f"\n{sum(PASS)}/{len(PASS)} passed")
    sys.exit(0 if all(PASS) and PASS else 1)
