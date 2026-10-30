"""
test_alpha_kelly_pnl_envelope.py — SF-BL-34956 verification tests.

Pre-registered done criterion: "Success if the scorer raises or flags
any executed trade whose PnL lies outside [-invested, shares - invested]
(with a pre-registered epsilon) on fixture books while in-envelope books
score bit-identically; kill if a producer census shows every scored
record's PnL is written only by settle_or_mark_trade from validated
position fields."

Kill condition is NOT met (see PRODUCER_AUDIT.md in the task work dir):
net_pnl_bid_marked is a plain dataclass field accepted as a constructor
keyword and mutated freely after construction (only action / is_settled
are guarded on TradeExecutionResult), direct construction is the norm
outside settle, and the scorer never validated shares_acquired at all.
So the fix applies: an envelope gate in evaluate_fleet_north_star
(_validate_scored_trade_pnl_envelopes) raising the named ValueError
PNL_OUTSIDE_PAYOUT_ENVELOPE on any executed (non-SKIP) trade whose PnL
lies outside [-invested, shares - invested] beyond the pre-registered
epsilon eps = 1e-9 + 1e-9 * max(1, |shares|, |invested|), and on any
executed trade whose shares_acquired is not a finite real >= 0 (the
envelope is undefined there).

Pre-fix reproduction (verified live 2026-10-27, before the patch;
fixture: one settled BUY_YES trade, 2000 shares, $1000 invested,
est_days 10, resolved [(0.6,1),(0.6,0)]):
  pnl=-50000 -> RETURNED cwr=-50.0 ending=-40000 north_star=-112.8276
  pnl=+50000 -> RETURNED cwr=+50.0 ending=+60000 north_star=+111.1276
Post-fix both raise; the in-envelope controls (-1000 / +100 / +1000)
return bit-identical scorecards, and the benchmark is bit-identical.

NOTE on benchmark values: the SF-BL-9475-era disciplined card
(+0.1059) has since moved on HEAD (later gates, e.g. SF-BL-10761
panel provenance, changed the disciplined fleet's executed set). The
bit-identity asserted here is against the CURRENT pre-fix HEAD capture
(benchmark_prefix.txt in the task work dir): disciplined
north_star -0.1611 / ending 9170.88 / cwr -1.0; naive -0.1499 /
9326.36 / -0.0784.

CONCURRENT NEIGHBOUR (landed mid-run, after this gate): SF-BL-35210's
exact settled-payout gate runs IMMEDIATELY AFTER this envelope gate,
so settled fixtures in this suite use exact payouts (win = shares -
invested, loss = -invested) and generic in-envelope controls are open
marks (is_settled=False), to which the envelope applies identically
and SF-BL-35210 does not. Out-of-envelope settled fixtures still raise
THIS gate's error first (ordering pinned in
test_headline_* / test_open_and_buy_no_twins_raise).

Run:  python3 test_alpha_kelly_pnl_envelope.py  (exit 0 iff pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_north_star_finiteness.py.
"""
import io
import contextlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster/superforecaster"))
import alpha_kelly as ak  # noqa: E402

PASS = []
NAN = float("nan")
INF = float("inf")
RESOLVED = [(0.6, 1), (0.6, 0)]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _trade(**over):
    base = dict(
        market_id="FIXTURE-34956", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C34956",
        action=ak.TradeAction.BUY_YES,
        calibrated_prob_yes=0.60, market_mid_prob=0.50, exec_price=0.50,
        liquidation_bid_price=0.50, kl_divergence_nats=0.0,
        expected_log_growth=0.0, kelly_fraction_raw=0.10,
        allocated_fraction=0.10, invested_capital=1000.0,
        shares_acquired=2000.0, est_days_to_resolution=10.0,
        is_settled=False, outcome_yes=None, net_pnl_bid_marked=100.0)
    base.update(over)
    return ak.TradeExecutionResult(**base)


def _raises(trades, resolved=RESOLVED):
    try:
        ak.evaluate_fleet_north_star("fixture", trades, resolved)
    except ValueError as e:
        return str(e)
    return None


def _score(trades, resolved=RESOLVED):
    return ak.evaluate_fleet_north_star("fixture", trades, resolved)


# ---- 1. Prereg headline cases -------------------------------------------------

def test_headline_minus_50000_raises():
    # The prereg fixture is a SETTLED trade; the envelope gate runs
    # before SF-BL-35210's exact-payout gate, so its token wins.
    msg = _raises([_trade(net_pnl_bid_marked=-50000.0, is_settled=True,
                          outcome_yes=1)])
    assert msg is not None, "PnL -50000 returned a scorecard (pre-fix CWR -50.0)"
    assert "PNL_OUTSIDE_PAYOUT_ENVELOPE" in msg and "SF-BL-34956" in msg, msg


def test_headline_plus_50000_raises():
    msg = _raises([_trade(net_pnl_bid_marked=50000.0, is_settled=True,
                          outcome_yes=1)])
    assert msg is not None, "PnL +50000 returned a scorecard (pre-fix N_star +111.1276)"
    assert "PNL_OUTSIDE_PAYOUT_ENVELOPE" in msg, msg


def test_error_names_market_pnl_envelope_and_position():
    msg = _raises([_trade(net_pnl_bid_marked=-50000.0, is_settled=True,
                          outcome_yes=1)])
    assert "FIXTURE-34956" in msg and "-50000" in msg, msg
    assert "-1000" in msg and "1000" in msg, msg  # envelope bounds named
    assert "shares_acquired=2000" in msg and "invested_capital=1000" in msg, msg


# ---- 2. Just outside the envelope raises ---------------------------------------

def test_just_outside_raises():
    for pnl in (1000.01, 1001.0, -1000.01, -1001.0):
        msg = _raises([_trade(net_pnl_bid_marked=pnl)])
        assert msg is not None and "PNL_OUTSIDE_PAYOUT_ENVELOPE" in msg, (pnl, msg)


def test_outside_among_good_trades_raises_either_order():
    good = _trade(market_id="GOOD")
    bad = _trade(market_id="BAD", net_pnl_bid_marked=5000.0)
    assert _raises([good, bad]) is not None
    assert _raises([bad, good]) is not None


def test_open_and_buy_no_twins_raise():
    msg = _raises([_trade(net_pnl_bid_marked=50000.0)])
    assert msg is not None and "PNL_OUTSIDE_PAYOUT_ENVELOPE" in msg, msg
    msg = _raises([_trade(action=ak.TradeAction.BUY_NO, is_settled=True,
                          outcome_yes=0, net_pnl_bid_marked=-50000.0)])
    assert msg is not None and "PNL_OUTSIDE_PAYOUT_ENVELOPE" in msg, msg


def test_settled_exact_payouts_score():
    # Settled boundary PnLs are exactly the SF-BL-35210 exact payouts:
    # win = shares - invested = +1000, loss = -invested = -1000.
    sc = _score([_trade(is_settled=True, outcome_yes=1,
                        net_pnl_bid_marked=1000.0)])
    assert sc.north_star_score == 2.2226 and sc.settlement_win_rate_pct == 100.0, sc
    sc = _score([_trade(is_settled=True, outcome_yes=0,
                        net_pnl_bid_marked=-1000.0)])
    assert sc.north_star_score == -2.2226 and sc.settlement_win_rate_pct == 0.0, sc


# ---- 3. Epsilon discipline ------------------------------------------------------

def test_epsilon_boundary():
    eps = ak._pnl_envelope_epsilon(2000.0, 1000.0)
    assert eps == ak.PNL_ENVELOPE_EPSILON_ABS + ak.PNL_ENVELOPE_EPSILON_REL * 2000.0
    # Within dust of either bound: passes.
    _score([_trade(net_pnl_bid_marked=1000.0 + eps / 2.0)])
    _score([_trade(net_pnl_bid_marked=-1000.0 - eps / 2.0)])
    # Meaningfully beyond: raises.
    assert _raises([_trade(net_pnl_bid_marked=1000.0 + 100.0 * eps)]) is not None
    assert _raises([_trade(net_pnl_bid_marked=-1000.0 - 100.0 * eps)]) is not None


def test_epsilon_scales_with_position():
    # A 1e12-scale position: envelope [-1e12, +1e12], eps ~ $1000 of dust.
    big = dict(invested_capital=1e12, shares_acquired=2e12,
               est_days_to_resolution=3650.0)
    _score([_trade(net_pnl_bid_marked=-1e12, **big)])
    _score([_trade(net_pnl_bid_marked=1e12, **big)])
    assert _raises([_trade(net_pnl_bid_marked=2e12, **big)]) is not None


# ---- 4. shares_acquired domain (envelope undefined otherwise) -------------------

def test_bad_shares_raise_envelope_error():
    for bad in (NAN, INF, -INF, -100.0, "2000", None, True):
        msg = _raises([_trade(shares_acquired=bad)])
        assert msg is not None, f"shares={bad!r} accepted"
        assert "PNL_OUTSIDE_PAYOUT_ENVELOPE" in msg and "shares_acquired" in msg, msg


def test_zero_shares_edge():
    # Zero shares: envelope collapses to [-invested, -invested].
    _score([_trade(shares_acquired=0.0, net_pnl_bid_marked=-1000.0)])
    assert _raises([_trade(shares_acquired=0.0, net_pnl_bid_marked=0.0)]) is not None


# ---- 5. In-envelope books bit-identical ------------------------------------------

def test_control_fixtures_exact():
    sc = _score([_trade(net_pnl_bid_marked=100.0)])
    assert sc.north_star_score == 0.2223 and sc.cwr_bid_marked == 0.1, sc
    assert sc.ending_account_value_bid_marked == 10100.0, sc
    sc = _score([_trade(net_pnl_bid_marked=-1000.0)])
    assert sc.north_star_score == -2.2226 and sc.cwr_bid_marked == -1.0, sc
    assert sc.ending_account_value_bid_marked == 9000.0, sc
    sc = _score([_trade(net_pnl_bid_marked=1000.0)])
    assert sc.north_star_score == 2.2226 and sc.cwr_bid_marked == 1.0, sc
    sc = _score([_trade(net_pnl_bid_marked=0.0)])
    assert sc.north_star_score == 0.0 and sc.total_net_pnl == 0.0, sc


def test_settle_produced_pnls_score_fine():
    # Every PnL settle_or_mark_trade itself can produce is in-envelope
    # and scores: settled win/loss on both sides, open marks across bids.
    for action, outcome in ((ak.TradeAction.BUY_YES, 1), (ak.TradeAction.BUY_YES, 0),
                            (ak.TradeAction.BUY_NO, 0), (ak.TradeAction.BUY_NO, 1)):
        t = _trade(action=action, net_pnl_bid_marked=0.0)
        ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=outcome)
        assert -1000.0 <= t.net_pnl_bid_marked <= 1000.0, t.net_pnl_bid_marked
        _score([t])
    for bid in (0.0, 0.25, 0.5, 0.75, 1.0):
        t = _trade(net_pnl_bid_marked=0.0, liquidation_bid_price=bid)
        ak.settle_or_mark_trade(t, is_settled=False)
        _score([t])


def test_post_settlement_mutation_raises():
    t = _trade(net_pnl_bid_marked=0.0)
    ak.settle_or_mark_trade(t, is_settled=True, outcome_yes=1)
    assert t.net_pnl_bid_marked == 1000.0, t.net_pnl_bid_marked
    t.net_pnl_bid_marked = 50000.0  # mutated after settlement
    assert _raises([t]) is not None
    t2 = _trade(net_pnl_bid_marked=0.0)
    ak.settle_or_mark_trade(t2, is_settled=True, outcome_yes=1)
    t2.shares_acquired = 100.0  # shrinking shares strands the honest PnL outside
    assert _raises([t2]) is not None


# ---- 6. Scope & gate ordering ------------------------------------------------------

def test_skip_trades_exempt():
    skip = _trade(action=ak.TradeAction.SKIP, invested_capital=0.0,
                  shares_acquired=0.0, net_pnl_bid_marked=999999.0,
                  skip_reason="FIXTURE")
    sc = _score([skip])
    assert sc.executed_trades == 0 and sc.skipped_opportunities == 1, sc
    assert sc.north_star_score == 0.0, sc


def test_neighbour_gate_identities_preserved():
    # SF-BL-9475 owns non-finite PnL; SF-BL-11304 owns invested <= 0;
    # SF-BL-9364 owns the lockup — each keeps its error identity when a
    # trade trips it together with the envelope.
    msg = _raises([_trade(net_pnl_bid_marked=NAN)])
    assert "NON_FINITE_TRADE_FIELD" in msg, msg
    msg = _raises([_trade(invested_capital=-5.0, net_pnl_bid_marked=50000.0)])
    assert "NON_POSITIVE_INVESTED_CAPITAL" in msg, msg
    msg = _raises([_trade(est_days_to_resolution=-5.0,
                          net_pnl_bid_marked=50000.0)])
    assert "NON_POSITIVE_EST_DAYS" in msg, msg


def test_helper_scope():
    ak._validate_scored_trade_pnl_envelopes([_trade()])  # in-envelope: no raise
    try:
        ak._validate_scored_trade_pnl_envelopes(
            [_trade(net_pnl_bid_marked=50000.0)])
    except ValueError as e:
        assert "PNL_OUTSIDE_PAYOUT_ENVELOPE" in str(e)
        return
    raise AssertionError("helper accepted out-of-envelope PnL")


# ---- 7. Benchmark scorecards bit-identical (current HEAD capture) -----------------

def test_benchmark_scorecards_bit_identical():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    cards = []
    decoder = json.JSONDecoder()
    idx = 0
    while True:
        start = out.find('{\n  "fleet_name"', idx)
        if start < 0:
            break
        obj, end = decoder.raw_decode(out[start:])
        cards.append(obj)
        idx = start + end
    assert len(cards) == 2, f"expected 2 scorecards, got {len(cards)}"
    assert cards[0]["north_star_score"] == -0.1611, cards[0]
    assert cards[1]["north_star_score"] == -0.1499, cards[1]
    assert cards[0]["ending_account_value_bid_marked"] == 9170.88, cards[0]
    assert cards[1]["ending_account_value_bid_marked"] == 9326.36, cards[1]
    assert cards[0]["cwr_bid_marked"] == -1.0, cards[0]
    assert cards[1]["cwr_bid_marked"] == -0.0784, cards[1]
    assert math.isfinite(cards[0]["north_star_score"])


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
