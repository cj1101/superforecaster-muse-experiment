"""SF-BL-6060 tests — empty-ladder fallback prices must never be
consumed as a tradeable book.

Prereg: "Census passes when every evaluation path either rejects
empty-side books or logs a fallback flag, and the historical
fallback-consumption rate is reported; killed if the rate is exactly
zero across all corpora (then add a regression assertion only)."
The scan (scan_corpora.py) found 33.57% of archived books
fallback-exposed, so the kill does NOT trigger and the census must
pass on the merits: these tests pin, for every canonical evaluation
path, that a fallback-backed book is rejected with a named reason /
fail-closed return, or valued only via the SF-BL-3549 declared rule —
and that two-sided tradeable books are bit-identical.

Run: python3 test_empty_ladder_fallback.py (exit 0 iff all pass).
"""
import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster"))
sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster/superforecaster"))
from superforecaster import alpha_kelly as ak

PASS = []
ZERO3 = (0.0, 0.0, 0.0)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))


def book(asks, bids, fee=0.01):
    return ak.OrderBookSnapshot(
        market_id="FIX-6060", title="sf-bl-6060 fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C6060",
        est_days_to_resolution=7.0,
        yes_asks=[ak.OrderBookLevel(price=p, size_shares=s) for p, s in asks],
        yes_bids=[ak.OrderBookLevel(price=p, size_shares=s) for p, s in bids],
        fee_rate=fee)


EMPTY = book([], [])
ASK_ONLY = book([(0.50, 100000)], [])
BID_ONLY = book([], [(0.50, 100000)])
ALL_INVALID = book([(-0.10, 1000)], [(1.50, 1000)])
INVALID_ASKS_EMPTY_BIDS = book([(-0.10, 1000)], [])
EMPTY_INVALID_BIDS = book([], [(1.50, 1000)])
ZERO_SIZE_ONLY = book([(0.50, 0)], [(0.48, 0)])
TWO_SIDED = book([(0.50, 100000)], [(0.48, 100000)])
CFG_OFF = ak.FleetRiskConfig(max_round_trip_cost=None)
CFG_ON = ak.FleetRiskConfig(max_round_trip_cost=0.05)


def _eval(b, p=0.80, cfg=CFG_OFF):
    cl = {}
    r = ak.evaluate_and_size_opportunity(b, p, 0.05, 100000.0, cl, cfg)
    return r, cl


# ---- 1. Fallback definition: helper names exactly the fallback sides ----

def test_fallback_sides_helper():
    assert ak._fallback_price_sides(EMPTY) == ("yes_asks", "yes_bids")
    assert ak._fallback_price_sides(ASK_ONLY) == ("yes_bids",)
    assert ak._fallback_price_sides(BID_ONLY) == ("yes_asks",)
    assert ak._fallback_price_sides(ALL_INVALID) == ("yes_asks", "yes_bids")
    assert ak._fallback_price_sides(ZERO_SIZE_ONLY) == ("yes_asks", "yes_bids")
    assert ak._fallback_price_sides(TWO_SIDED) == ()
    # a live tradeable rung behind dead rungs is a measured touch
    mixed = book([(0.50, 0), (-0.10, 5), (0.60, 1000)], [(0.48, 1000)])
    assert ak._fallback_price_sides(mixed) == ()


def test_property_sentinels_pinned_unchanged():
    # SF-BL-9398/9542 property contract is NOT this task's to change.
    assert EMPTY.best_yes_ask == 0.99 and EMPTY.best_yes_bid == 0.01
    assert EMPTY.mid_price == 0.50
    assert ALL_INVALID.best_yes_ask == 0.99 and ALL_INVALID.mid_price == 0.50


# ---- 2. Evaluate: every fallback class gets a NAMED rejection -----------

def test_evaluate_named_rejections():
    cases = [(EMPTY, "EMPTY_BOOK"), (ZERO_SIZE_ONLY, "EMPTY_BOOK"),
             (ASK_ONLY, "ONE_SIDED_BOOK"), (BID_ONLY, "ONE_SIDED_BOOK"),
             (ALL_INVALID, "NO_TRADEABLE_PRICE"),
             (INVALID_ASKS_EMPTY_BIDS, "NO_TRADEABLE_PRICE"),
             (EMPTY_INVALID_BIDS, "NO_TRADEABLE_PRICE")]
    for b, token in cases:
        for cfg in (CFG_OFF, CFG_ON):
            for p in (0.05, 0.20, 0.50, 0.80, 0.95):
                r, cl = _eval(b, p, cfg)
                assert r.action == ak.TradeAction.SKIP, (token, p, cfg, r)
                assert r.skip_reason.startswith(token), (token, p, r.skip_reason)
                assert r.market_mid_prob == 0.0, (token, p, r)
                assert r.liquidation_bid_price == 0.0, (token, p, r)
                assert r.invested_capital == 0.0 and r.shares_acquired == 0.0
                assert cl == {}, (token, p, cl)


def test_no_fallback_book_ever_skips_insufficient_net_edge():
    # The pre-fix failure: fallback books screened against the
    # sentinels and mislabelled INSUFFICIENT_NET_EDGE.
    for b in (EMPTY, ASK_ONLY, BID_ONLY, ALL_INVALID,
              INVALID_ASKS_EMPTY_BIDS, EMPTY_INVALID_BIDS, ZERO_SIZE_ONLY):
        for p in (0.05, 0.35, 0.65, 0.95):
            r, _ = _eval(b, p)
            assert not r.skip_reason.startswith("INSUFFICIENT_NET_EDGE"), (p, r)


def test_two_sided_controls_unchanged():
    r, _ = _eval(TWO_SIDED, 0.80)
    assert r.action == ak.TradeAction.BUY_YES and r.invested_capital == 15000.0, r
    assert r.market_mid_prob == 0.49 and r.liquidation_bid_price == 0.48, r
    r, _ = _eval(TWO_SIDED, 0.50)
    assert r.action == ak.TradeAction.SKIP
    assert r.skip_reason.startswith("INSUFFICIENT_NET_EDGE"), r.skip_reason


# ---- 3. Sweep: LCB default prior never reads a fallback mid --------------

ASK_ONLY_2 = book([(0.50, 1000), (0.70, 100000)], [])
BID_ONLY_2 = book([], [(0.50, 1000), (0.30, 100000)])
TWO_SIDED_2 = book([(0.50, 1000), (0.70, 100000)], [(0.48, 100000)])


def test_sweep_lcb_default_prior_fails_closed_on_fallback():
    for b, side in [(ASK_ONLY_2, ak.TradeAction.BUY_YES),
                    (BID_ONLY_2, ak.TradeAction.BUY_NO),
                    (EMPTY, ak.TradeAction.BUY_YES),
                    (ALL_INVALID, ak.TradeAction.BUY_YES)]:
        out = ak.sweep_order_book(b, side, 5000.0, 0.90, 0.05,
                                  posterior_n_eff=100.0, lcb_z=1.0)
        assert out == ZERO3, (side, out)


def test_sweep_lcb_explicit_prior_untouched_on_fallback_book():
    # An explicit prior is the caller's own value, not a fallback.
    out = ak.sweep_order_book(ASK_ONLY_2, ak.TradeAction.BUY_YES, 5000.0,
                              0.90, 0.05, posterior_n_eff=100.0, lcb_z=1.0,
                              prior_prob_side=0.49)
    assert out == (5000.0, 7357.850070721358, 0.6795463283352556), out


def test_sweep_lcb_two_sided_default_prior_unchanged():
    out = ak.sweep_order_book(TWO_SIDED_2, ak.TradeAction.BUY_YES, 5000.0,
                              0.90, 0.05, posterior_n_eff=100.0, lcb_z=1.0)
    assert out == (5000.0, 7357.850070721358, 0.6795463283352556), out
    # default (no LCB) sweep on a fallback book's entry ladder is the
    # SF-BL-10760-pinned primitive contract: entry walk only, unchanged.
    out = ak.sweep_order_book(ASK_ONLY, ak.TradeAction.BUY_YES, 1000.0,
                              0.90, -1.0)
    assert out[0] > 0.0, out


# ---- 4. Crossed predicate: no fallback-fabricated crossing ---------------

def test_crossed_predicate_fallback_safe():
    assert ak._is_crossed_book(book([(0.005, 1000)], [])) is False  # was True pre-fix
    assert ak._is_crossed_book(EMPTY) is False
    assert ak._is_crossed_book(ASK_ONLY) is False
    assert ak._is_crossed_book(ALL_INVALID) is False
    crossed = book([(0.40, 1000)], [(0.60, 1000)])
    locked = book([(0.50, 1000)], [(0.50, 1000)])
    assert ak._is_crossed_book(crossed) is True
    assert ak._is_crossed_book(locked) is True
    assert ak._is_crossed_book(TWO_SIDED) is False
    r, _ = _eval(crossed, 0.62)
    assert r.skip_reason.startswith("CROSSED_BOOK"), r.skip_reason


# ---- 5. Declared rule / settle / benchmark surfaces stay declared --------

def test_declared_helpers_still_hold():
    assert ak._declared_mid_prob(EMPTY) == 0.0
    assert ak._declared_mid_prob(ASK_ONLY) == 0.0
    assert ak._declared_mid_prob(TWO_SIDED) == 0.49
    assert ak._declared_liquidation_bid(ASK_ONLY, ak.TradeAction.BUY_YES) == 0.0
    assert ak._declared_liquidation_bid(TWO_SIDED, ak.TradeAction.BUY_YES) == 0.48


def test_benchmark_unchanged():
    # All 8 fixture books are two-sided, so this task's gates cannot
    # move the benchmark: the full stdout is byte-identical pre/post
    # patch (verified against the reconstructed pre-fix module — see
    # DONE-CLAIM.md). The naive slate (-0.1499, which exercises the
    # sweep directly) is the stable pin; the disciplined slate reads
    # -0.1611 at this run, drifted from the 0.0547 SF-BL-3549 recorded
    # by parallel fleet patches to evaluate-path code — the pre-fix
    # module reconstructed for this task produces -0.1611 too.
    buf = io.StringIO()
    with redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    assert '"north_star_score": -0.1499' in out, out[-1200:]
    assert '"north_star_score": -0.1611' in out, out[-1200:]


ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
if __name__ == "__main__":
    for fn in ALL:
        try:
            fn()
            check(fn.__name__, True)
        except AssertionError as e:
            check(fn.__name__, False, str(e))
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
