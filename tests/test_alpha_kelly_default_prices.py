"""
test_default_prices.py — SF-BL-3549 verification tests.

Pre-registered done/kill criteria (TASK.md):
  "Unit fixtures with ask-only, bid-only and empty books must show
   zero phantom marks (position valued via a declared rule, never
   the 0.01/0.99 defaults) after the fix; kill if a corpus scan of
   archived books finds zero one-sided books ever reaching the
   engine (defaults declared unreachable, documented)."

Declared rule under test (SF-BL-3549, in alpha_kelly.py): where a
side has no live, tradeably-priced ladder there is NO observable
price on that side — an engine record's mid is the real mid_price
only when BOTH ladders are observable, otherwise 0.0; a side's
liquidation bid is its real best bid only when that side's exit
ladder is observable, otherwise 0.0. The property-level sentinels
(0.99 / 0.01) are pinned by SF-BL-9398/9542 and are NOT changed;
this suite tests that no engine RECORD consumes them as prices.

Pre-fix reproduction (repro.py / repro_prefix.txt in the task dir):
  EMPTY_BOOK record mid 0.50; INVALID_POSTERIOR on ask-only mid
  0.255, on bid-only 0.745; all-invalid book INSUFFICIENT_NET_EDGE
  mid 0.50; naive baseline marked an ask-only YES position at the
  0.01 sentinel (-$1,176.24 on a $1,200 stake).

Kill criterion: scan_books.py / SCAN.md — 426 of 1,131 archived
forward S2 books are one-sided (37.7%), so the defaults ARE reached
and the kill does NOT trigger (asserted from scan_results.json when
present).

Run:  python3 test_default_prices.py   (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner.
"""
import io
import json
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(
    os.path.expanduser("~/workspace/superforecaster"), "superforecaster"))

import alpha_kelly as ak  # noqa: E402

PASS = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(asks, bids, fee=0.01, domain=ak.MarketDomain.PREDICTION_MARKET):
    return ak.OrderBookSnapshot(
        market_id="FIXTURE-3549", title="SF-BL-3549 fixture",
        domain=domain, factor_cluster="C3549", est_days_to_resolution=5,
        yes_asks=[ak.OrderBookLevel(price=p, size_shares=s) for p, s in asks],
        yes_bids=[ak.OrderBookLevel(price=p, size_shares=s) for p, s in bids],
        fee_rate=fee)


ASK_ONLY = [(0.50, 100000)], []
BID_ONLY = [], [(0.50, 100000)]
EMPTY = [], []
ALL_INVALID = [(-0.10, 1000)], [(1.50, 1000)]
ZERO_ONLY = [(0.40, 0)], [(0.88, 0)]
DEAD_EXIT = [(0.50, 100000)], [(0.48, 0)]          # live asks, zero-size bids
INVALID_EXIT = [(0.50, 100000)], [(-0.30, 1000)]    # live asks, invalid bids
TWO_SIDED = [(0.50, 100000)], [(0.48, 100000)]

NON_TWO_SIDED = {
    "ask_only": ASK_ONLY, "bid_only": BID_ONLY, "empty": EMPTY,
    "all_invalid": ALL_INVALID, "zero_only": ZERO_ONLY,
    "dead_exit": DEAD_EXIT, "invalid_exit": INVALID_EXIT,
}


def _eval(book, p, cfg=None):
    kwargs = dict(book=book, calibrated_prob_yes=p,
                  inter_agent_dispersion=0.05, portfolio_equity=100000.0,
                  cluster_exposure_used={})
    if cfg is not None:
        kwargs["config"] = cfg
    return ak.evaluate_and_size_opportunity(**kwargs)


# ---- 1. Declared-rule helpers ------------------------------------------------

def test_declared_mid_zero_when_any_side_unobservable():
    for name, (asks, bids) in NON_TWO_SIDED.items():
        b = _book(asks, bids)
        assert ak._declared_mid_prob(b) == 0.0, (name, ak._declared_mid_prob(b))


def test_declared_liquidation_bid_zero_when_exit_unobservable():
    # BUY_YES exits on yes_bids; BUY_NO exits on the derived NO bid
    # (yes_asks). Each assert pairs a side with its missing exit ladder.
    assert ak._declared_liquidation_bid(_book(*ASK_ONLY), ak.TradeAction.BUY_YES) == 0.0
    assert ak._declared_liquidation_bid(_book(*BID_ONLY), ak.TradeAction.BUY_NO) == 0.0
    assert ak._declared_liquidation_bid(_book(*EMPTY), ak.TradeAction.BUY_YES) == 0.0
    assert ak._declared_liquidation_bid(_book(*EMPTY), ak.TradeAction.BUY_NO) == 0.0
    assert ak._declared_liquidation_bid(_book(*DEAD_EXIT), ak.TradeAction.BUY_YES) == 0.0
    assert ak._declared_liquidation_bid(_book(*INVALID_EXIT), ak.TradeAction.BUY_YES) == 0.0


def test_declared_helpers_return_real_prices_when_observable():
    b = _book(*TWO_SIDED)
    assert ak._declared_mid_prob(b) == b.mid_price == 0.49
    assert ak._declared_liquidation_bid(b, ak.TradeAction.BUY_YES) == 0.48
    assert ak._declared_liquidation_bid(b, ak.TradeAction.BUY_NO) == 0.50  # 1 - 0.50 ask
    # A live rung behind a dead rung is observable at its real price.
    b2 = _book([(0.50, 100000)], [(0.88, 0), (0.48, 100000)])
    assert ak._declared_liquidation_bid(b2, ak.TradeAction.BUY_YES) == 0.48
    assert ak._declared_mid_prob(b2) == 0.49


def test_property_sentinels_unchanged():
    # The SF-BL-9398/9542 property contract is deliberately untouched:
    # sentinels still exist at the property level; they are simply
    # never consumed as record prices (tests below).
    b = _book(*EMPTY)
    assert b.best_yes_ask == 0.99 and b.best_yes_bid == 0.01
    assert b.mid_price == 0.50


# ---- 2. Evaluate records: zero phantom marks on every fixture ----------------

def test_evaluate_grid_no_phantom_marks():
    for name, (asks, bids) in NON_TWO_SIDED.items():
        for p in (0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95):
            for gate_d in (None, 0.05):
                cfg = ak.FleetRiskConfig(max_round_trip_cost=gate_d)
                cl = {}
                r = ak.evaluate_and_size_opportunity(
                    _book(asks, bids), p, 0.05, 100000.0, cl, config=cfg)
                assert r.action == ak.TradeAction.SKIP, (name, p, gate_d, r)
                assert r.market_mid_prob == 0.0, (name, p, gate_d, r.market_mid_prob)
                assert r.market_mid_prob not in (0.50, 0.255, 0.745)
                assert r.liquidation_bid_price == 0.0, (name, p, gate_d, r)
                assert r.invested_capital == 0.0 and r.shares_acquired == 0.0
                assert cl == {}, (name, p, gate_d, cl)
                marked = ak.settle_or_mark_trade(r, is_settled=False)
                assert marked.net_pnl_bid_marked == 0.0, (name, p, gate_d, marked)


def test_empty_book_record_mid_is_declared_zero_not_phantom_half():
    r = _eval(_book(*EMPTY), 0.80)
    assert r.skip_reason.startswith("EMPTY_BOOK"), r.skip_reason
    assert r.market_mid_prob == 0.0 and r.market_mid_prob != 0.50


def test_all_invalid_book_no_phantom_mid():
    r = _eval(_book(*ALL_INVALID), 0.80)
    assert r.action == ak.TradeAction.SKIP, r
    assert r.market_mid_prob == 0.0, r.market_mid_prob


# ---- 3. Early gates (ordered before Gate 0.5/0.7) carry no phantom mid -------

def test_invalid_posterior_on_one_sided_books():
    r = _eval(_book(*ASK_ONLY), 1.50)
    assert r.skip_reason.startswith("INVALID_POSTERIOR") and r.market_mid_prob == 0.0, r
    r = _eval(_book(*BID_ONLY), -0.20)
    assert r.skip_reason.startswith("INVALID_POSTERIOR") and r.market_mid_prob == 0.0, r


def test_invalid_fee_on_empty_book():
    b = _book(*EMPTY)
    b.fee_rate = 2.0  # post-construction mutation bypasses __post_init__
    r = _eval(b, 0.80)
    assert r.skip_reason.startswith("INVALID_FEE_RATE") and r.market_mid_prob == 0.0, r


def test_invalid_dispersion_and_equity_on_one_sided_books():
    r = ak.evaluate_and_size_opportunity(
        _book(*ASK_ONLY), 0.80, float("nan"), 100000.0, {})
    assert r.skip_reason.startswith("INVALID_INTER_AGENT_DISPERSION"), r.skip_reason
    assert r.market_mid_prob == 0.0, r
    r = ak.evaluate_and_size_opportunity(
        _book(*BID_ONLY), 0.80, 0.05, float("nan"), {})
    assert r.skip_reason.startswith("INVALID_EQUITY"), r.skip_reason
    assert r.market_mid_prob == 0.0, r


def test_nan_rung_on_ask_only_book_keeps_invalid_equity_and_no_mid():
    b = _book([(float("nan"), 1000), (0.50, 1000)], [])
    r = _eval(b, 0.80)
    assert r.skip_reason.startswith("INVALID_EQUITY"), r.skip_reason
    assert r.market_mid_prob == 0.0, r


def test_invalid_config_on_empty_book():
    cfg = ak.FleetRiskConfig()
    cfg.require_on_chain_for_crypto = "yes"  # mutated non-bool switch
    r = _eval(_book(*EMPTY), 0.80, cfg=cfg)
    assert r.skip_reason.startswith("INVALID_CONFIG"), r.skip_reason
    assert r.market_mid_prob == 0.0, r


def test_one_sided_gate_regression():
    for gate_d in (None, 0.05):
        cfg = ak.FleetRiskConfig(max_round_trip_cost=gate_d)
        r = _eval(_book(*ASK_ONLY), 0.80, cfg=cfg)
        assert r.skip_reason.startswith("ONE_SIDED_BOOK"), r.skip_reason
        assert r.market_mid_prob == 0.0 and r.liquidation_bid_price == 0.0


# ---- 4. Naive baseline declared rule ------------------------------------------

def test_naive_block_uses_declared_rule_in_source():
    src = open(os.path.join(
        os.path.expanduser("~/workspace/superforecaster"),
        "superforecaster", "alpha_kelly.py")).read()
    assert "_declared_liquidation_bid(book, naive_side)" in src
    assert "NO_OBSERVABLE_PRICE" in src


# ---- 5. Two-sided books unchanged ---------------------------------------------

def test_two_sided_control_unchanged():
    r = _eval(_book(*TWO_SIDED), 0.80)
    assert r.action == ak.TradeAction.BUY_YES, r
    assert r.invested_capital == 15000.0, r
    assert r.market_mid_prob == 0.49 and r.liquidation_bid_price == 0.48, r
    marked = ak.settle_or_mark_trade(r, is_settled=False)
    expect = r.shares_acquired * 0.48 - 15000.0
    assert abs(marked.net_pnl_bid_marked - expect) < 1e-9, marked


def test_two_sided_buy_no_liquidation_is_real_derived_bid():
    b = _book([(0.52, 100000)], [(0.50, 100000)])
    r = _eval(b, 0.20)
    assert r.action == ak.TradeAction.BUY_NO, r
    assert r.liquidation_bid_price == 0.48, r  # 1 - 0.52 real ask
    assert abs(r.market_mid_prob - 0.51) < 1e-12, r


def test_benchmark_still_runs_and_scores():
    buf = io.StringIO()
    with redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    assert "ALPHA-KELLY NORTH STAR BENCHMARK" in out, out[:500]
    assert '"fleet_name": "Alpha-Kelly Disciplined Fleet"' in out
    assert '"fleet_name": "Naive Brier-Only Baseline Fleet"' in out


# ---- 6. Kill criterion ---------------------------------------------------------

def test_kill_criterion_not_triggered():
    path = os.path.join(HERE, "scan_results.json")
    if not os.path.exists(path):
        # Scan not shipped alongside (repo copy): the criterion is
        # documented in SCAN.md; do not fail the unit suite for it.
        print("scan_results.json absent — kill check documented in SCAN.md")
        return
    res = json.load(open(path))
    assert res["one_sided_total"] == 426, res
    assert res["books_total"] == 1131, res
    assert res["kill_triggered"] is False, res


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
