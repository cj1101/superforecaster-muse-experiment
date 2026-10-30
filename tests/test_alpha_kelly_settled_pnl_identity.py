"""
test_alpha_kelly_settled_pnl_identity.py — SF-BL-34957 verification.

SF-BL-34957 prereg: "Success if the scorer flags or raises on settled
records whose stored PnL deviates from the settlement identity by
more than a pre-registered epsilon, with identity-consistent books
bit-identical; kill if a producer census shows settled records
reaching the scorer come only from settle_or_mark_trade on unmutated
evaluate outputs."

The settlement identity (TASK.md): for a settled record the true PnL
is shares x 1{won(action, outcome_yes)} - invested, won being
(BUY_YES and outcome_yes == 1) or (BUY_NO and outcome_yes == 0).

OVERLAP (verified live this run): while this task was executing, the
concurrent executor of SF-BL-35210 — the same defect stated with
different fixtures (a 1,000-share / $500 YES win carrying +$100
instead of the only possible +$500) — landed exactly this check in
evaluate_fleet_north_star. Its first implementation RAISED
(SETTLED_PNL_PAYOUT_MISMATCH); its regression sweep then broke at
least six ratified suites plus SF-BL-34866 / SF-BL-9477, whose
CONTROL fixtures pin settled records carrying non-payout PnL for
unrelated gates, so — per the SF-BL-34866 precedent and the
disjunctive "raises or flags" success text both preregs share — it
landed the FLAG branch: _assess_settled_pnl_mismatches counts settled
executed trades whose |stored - expected| exceeds the pre-registered
SETTLED_PNL_EPSILON = 0.005 (expected = shares - invested on a win,
-invested on a loss — algebraically the identity above), reported as
NorthStarScorecard.settled_pnl_mismatch_count /
settled_pnl_mismatch_invested, with every pre-existing numeric field
bit-identical. SF-BL-34956's payout-envelope gate (also landed
concurrently) runs immediately before it and validates
shares_acquired. This suite therefore VERIFIES the landed flag
against SF-BL-34957's own fixtures and criteria rather than landing
a duplicate check; the pristine snapshot in this task's work dir
(pristine/alpha_kelly.py, taken at run start, containing neither
gate) supplies the pre-fix behaviour for the bit-identity tests —
live-vs-pristine scorecard equality is asserted after removing
exactly the two new flag fields.

Run:  python3 test_alpha_kelly_settled_pnl_identity.py  (exit 0 iff pass)
"""
import importlib.util
import io
import contextlib
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster/superforecaster"))
import alpha_kelly as ak  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_PRISTINE = os.path.join(_HERE, "pristine", "alpha_kelly.py")
if not os.path.exists(_PRISTINE):  # installed copy in tests/: use work dir
    _PRISTINE = os.path.expanduser(
        "~/workspace/superforecaster_out/backlog/work/SF-BL-34957"
        "/pristine/alpha_kelly.py")
_spec = importlib.util.spec_from_file_location("alpha_kelly_pristine", _PRISTINE)
pre = importlib.util.module_from_spec(_spec)
sys.modules["alpha_kelly_pristine"] = pre  # dataclass processing needs it
_spec.loader.exec_module(pre)

PASS = []
RESOLVED = [(0.6, 1), (0.6, 0)]
FLAG_FIELDS = ("settled_pnl_mismatch_count", "settled_pnl_mismatch_invested")


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _trade(mod, mid, action, outcome, shares, invested, pnl, **over):
    base = dict(
        market_id=mid, title="SF-BL-34957 fixture",
        domain=mod.MarketDomain.PREDICTION_MARKET, factor_cluster="C34957",
        action=action, calibrated_prob_yes=0.60, market_mid_prob=0.50,
        exec_price=0.50, liquidation_bid_price=0.50,
        kl_divergence_nats=0.0, expected_log_growth=0.0,
        kelly_fraction_raw=0.10, allocated_fraction=0.10,
        invested_capital=invested, shares_acquired=shares,
        est_days_to_resolution=10.0, is_settled=True,
        outcome_yes=outcome, net_pnl_bid_marked=pnl)
    base.update(over)
    return mod.TradeExecutionResult(**base)


def trade(mid, action, outcome, shares, invested, pnl, **over):
    return _trade(ak, mid, action, outcome, shares, invested, pnl, **over)


def score(mod, trades, **kw):
    return mod.evaluate_fleet_north_star("fixture", trades, RESOLVED,
                                         starting_capital=10000.0, **kw)


def flags(trades):
    sc = score(ak, trades)
    return sc.settled_pnl_mismatch_count, sc.settled_pnl_mismatch_invested


def same_numbers(live_sc, old_sc):
    a, b = asdict(live_sc), asdict(old_sc)
    for f in FLAG_FIELDS:
        a.pop(f)
    return a == b


# ---- 1. The prereg fixtures ---------------------------------------------------

def test_prereg_lost_with_plus300_flagged():
    bad = trade("BAD-LOST", ak.TradeAction.BUY_YES, 0, 2000.0, 1000.0, 300.0)
    cnt, inv = flags([bad])
    assert (cnt, inv) == (1, 1000.0), (cnt, inv)
    # Pre-fix (pristine): the same record scored CWR 0.3 / win 100%
    # with NO flag anywhere on the scorecard.
    old = score(pre, [_trade(pre, "BAD-LOST", pre.TradeAction.BUY_YES,
                              0, 2000.0, 1000.0, 300.0)])
    assert old.cwr_bid_marked == 0.3 and old.settlement_win_rate_pct == 100.0, old
    assert not hasattr(old, "settled_pnl_mismatch_count"), old
    # Flag-only: the landed branch changes no pre-existing number.
    assert same_numbers(score(ak, [bad]), old)


def test_prereg_won_with_minus200_flagged():
    bad = trade("BAD-WON", ak.TradeAction.BUY_YES, 1, 2000.0, 1000.0, -200.0)
    cnt, inv = flags([bad])
    assert (cnt, inv) == (1, 1000.0), (cnt, inv)
    old = score(pre, [_trade(pre, "BAD-WON", pre.TradeAction.BUY_YES,
                              1, 2000.0, 1000.0, -200.0)])
    assert old.settlement_win_rate_pct == 0.0 and old.total_net_pnl == -200.0, old


def test_buy_no_mirrors_flagged():
    # BUY_NO loses when outcome_yes == 1, wins when outcome_yes == 0.
    assert flags([trade("BN-LOST", ak.TradeAction.BUY_NO, 1,
                         2000.0, 1000.0, 300.0)])[0] == 1
    assert flags([trade("BN-WON", ak.TradeAction.BUY_NO, 0,
                         2000.0, 1000.0, -200.0)])[0] == 1


# ---- 2. Exact identity: no flag, all four side x outcome combos ---------------

def test_all_four_combos_exact_pnl_unflagged():
    combos = [
        (ak.TradeAction.BUY_YES, 1, True), (ak.TradeAction.BUY_YES, 0, False),
        (ak.TradeAction.BUY_NO, 0, True), (ak.TradeAction.BUY_NO, 1, False),
    ]
    for action, outcome, won in combos:
        expected = 2000.0 - 1000.0 if won else -1000.0
        sc = score(ak, [trade("C", action, outcome, 2000.0, 1000.0, expected)])
        assert sc.settled_pnl_mismatch_count == 0, (action, outcome, sc)
        assert sc.total_net_pnl == expected, (action, outcome, sc)
        assert sc.settlement_win_rate_pct == (100.0 if won else 0.0), sc
        # A materially different stored figure is flagged. The
        # deviation points INSIDE the payout envelope (a win's
        # expected value IS the envelope ceiling, a loss's the floor;
        # an outward deviation is SF-BL-34956's envelope case).
        inward = expected - 250.0 if won else expected + 250.0
        cnt, _ = flags([trade("C", action, outcome, 2000.0, 1000.0, inward)])
        assert cnt == 1, (action, outcome)


# ---- 3. Pre-registered epsilon (landed gate: 0.005, half a cent) --------------

def test_epsilon_boundary():
    assert ak.SETTLED_PNL_EPSILON == 0.005
    # Loss fixture (expected -1000.0): deviation 0.004 (inside the
    # half-cent epsilon) is unflagged; deviation 0.006 is flagged.
    assert flags([trade("EPS", ak.TradeAction.BUY_YES, 0,
                         2000.0, 1000.0, -999.996)])[0] == 0
    assert flags([trade("EPS", ak.TradeAction.BUY_YES, 0,
                         2000.0, 1000.0, -999.994)])[0] == 1
    # Exact settle figures differ by 0.0 and are never flagged.
    assert flags([trade("EPS2", ak.TradeAction.BUY_YES, 1,
                         2000.0, 1000.0, 1000.0)])[0] == 0


# ---- 4. Identity-consistent books bit-identical -------------------------------

def test_identity_consistent_books_bit_identical_to_pristine():
    # Specs use side NAMES so each module resolves its own TradeAction
    # members (the pristine and live enums are distinct classes).
    specs = [
        ("BUY_YES", 1, 2000.0, 1000.0),
        ("BUY_YES", 0, 2000.0, 1000.0),
        ("BUY_NO", 0, 2000.0, 1000.0),
        ("BUY_NO", 1, 2000.0, 1000.0),
        ("BUY_YES", 1, 1267.80, 500.0),
        ("BUY_YES", 0, 333.33, 111.11),
        ("BUY_NO", 0, 0.0, 250.0),      # zero-share win: -invested
        ("BUY_YES", 1, 1000.0, 1000.0),  # scratch win: 0.0
    ]
    def one(mod, mid, side, o, sh, inv):
        a = getattr(mod.TradeAction, side)
        won = (side == "BUY_YES" and o == 1) or (side == "BUY_NO" and o == 0)
        return _trade(mod, mid, a, o, sh, inv, sh - inv if won else -inv)
    live = score(ak, [one(ak, f"K{i}", *s) for i, s in enumerate(specs)])
    old = score(pre, [one(pre, f"K{i}", *s) for i, s in enumerate(specs)])
    assert live.settled_pnl_mismatch_count == 0, live
    assert live.settled_pnl_mismatch_invested == 0.0, live
    assert same_numbers(live, old), (asdict(live), asdict(old))
    for i, s in enumerate(specs):  # single-trade books too
        assert same_numbers(score(ak, [one(ak, f"S{i}", *s)]),
                            score(pre, [one(pre, f"S{i}", *s)]))


def test_settle_produced_books_bit_identical():
    def build(mod):
        out = []
        for i, (a, o) in enumerate([(mod.TradeAction.BUY_YES, 1),
                                    (mod.TradeAction.BUY_YES, 0),
                                    (mod.TradeAction.BUY_NO, 0),
                                    (mod.TradeAction.BUY_NO, 1)]):
            t = _trade(mod, f"SP{i}", a, None, 2000.0, 1000.0, 0.0,
                       is_settled=False)
            out.append(mod.settle_or_mark_trade(t, is_settled=True,
                                                outcome_yes=o))
        return out
    live_trades, old_trades = build(ak), build(pre)
    assert [t.net_pnl_bid_marked for t in live_trades] == \
        [t.net_pnl_bid_marked for t in old_trades] == \
        [1000.0, -1000.0, 1000.0, -1000.0]
    live_sc = score(ak, live_trades)
    assert live_sc.settled_pnl_mismatch_count == 0, live_sc
    assert same_numbers(live_sc, score(pre, old_trades))


# ---- 5. Exempt surfaces --------------------------------------------------------

def test_open_and_skip_records_untouched():
    # Open trade: PnL is a bid mark (in-envelope, not the payout) —
    # the identity does not determine it, so it is not flagged.
    open_t = trade("OPEN", ak.TradeAction.BUY_YES, None, 2000.0, 1000.0,
                   300.0, is_settled=False)
    sc = score(ak, [open_t])
    assert sc.settled_pnl_mismatch_count == 0 and sc.total_net_pnl == 300.0, sc
    old = score(pre, [_trade(pre, "OPEN", pre.TradeAction.BUY_YES, None,
                             2000.0, 1000.0, 300.0, is_settled=False)])
    assert same_numbers(sc, old)
    # SKIP records contribute nothing and are exempt, as for every
    # neighbouring scorer gate.
    skip = trade("SKIP", ak.TradeAction.SKIP, 0, 2000.0, 1000.0, 300.0)
    sc2 = score(ak, [skip])
    assert sc2.settled_pnl_mismatch_count == 0, sc2
    assert sc2.executed_trades == 0 and sc2.north_star_score == 0.0, sc2


def test_pooled_flags_count_every_mismatch():
    good = trade("GOOD", ak.TradeAction.BUY_YES, 1, 2000.0, 1000.0, 1000.0)
    bad1 = trade("BAD1", ak.TradeAction.BUY_YES, 0, 2000.0, 1000.0, 300.0)
    bad2 = trade("BAD2", ak.TradeAction.BUY_NO, 0, 1500.0, 500.0, -100.0)
    cnt, inv = flags([good, bad1, bad2])
    assert (cnt, inv) == (2, 1500.0), (cnt, inv)
    cnt, inv = flags([good])
    assert (cnt, inv) == (0, 0.0), (cnt, inv)


def test_benchmark_unchanged():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    for needle in ('"north_star_score": -0.1611',
                   '"north_star_score": -0.1499',
                   '"ending_account_value_bid_marked": 9170.88',
                   '"ending_account_value_bid_marked": 9326.36',
                   '"settled_pnl_mismatch_count": 0'):
        assert needle in out, needle


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    for name, fn in fns:
        try:
            fn()
            check(name, True)
        except Exception as e:  # noqa: BLE001
            check(name, False, f"{type(e).__name__}: {e}")
    print(f"\n{sum(PASS)}/{len(PASS)} PASS")
    sys.exit(0 if all(PASS) and PASS else 1)
