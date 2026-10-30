"""
test_no_fee_mirror.py — SF-BL-8892 verification tests.

Pre-registered done/kill criteria (TASK.md):
  "On a swept grid of bids 0.05-0.95 and fee_rate 0.005-0.015, the
   corrected NO effective ask must match 1 - bid*(1-fee) to within
   1e-6 at every grid point and change at least 5% of borderline
   (edge within 0.005 of the hurdle) NO verdicts on forward books;
   kill if verdict changes are below 1% and price deltas below
   0.0005 everywhere."

Pre-fix reproduction (verified live 2026-10-22, before the patch;
see repro.py in the task work dir):
  bid=0.65 fee=0.01: old eff 0.353500 vs mirror 0.356500 (delta -0.0030)
  bid=0.30 fee=0.01: old eff 0.707000 vs mirror 0.703000 (delta +0.0040)
  Engine q_top (recovered from kelly_fraction_raw) matched the OLD
  formula to ~1e-16 at every probed point, and the borderline NO
  candidate at bid 0.65 (p_no=0.4055, old edge 0.0520) passed the edge
  screen pre-fix.

Kill condition: NOT met (see REPLAY.md) — replay_forward.py on the 705
two-sided archived forward S2 books (2026-10-20T00 + 2026-10-21T00,
deduped): 218 of 452 borderline candidates change screen verdict at
fee 0.01 (48.2% >= 5%), max |price delta| 0.00998, and 698 of 705 books
carry |delta| >= 0.0005.

Run:  python3 test_no_fee_mirror.py   (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_fee_rate.py.
"""
import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(
    os.path.expanduser("~/workspace/superforecaster"), "superforecaster"))

import alpha_kelly as ak  # noqa: E402

PASS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def tight_book(bid, fee=0.01, size=100000.0):
    """Two-sided book, ask = bid + 0.02 (tight spread, deep ladders)."""
    return ak.OrderBookSnapshot(
        market_id="T8892", title="t", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C8892", est_days_to_resolution=5.0,
        yes_asks=[ak.OrderBookLevel(price=round(bid + 0.02, 6), size_shares=size)],
        yes_bids=[ak.OrderBookLevel(price=bid, size_shares=size)],
        fee_rate=fee)


def _forced_config():
    """Gates off everywhere except the edge screen + Kelly sizing, so
    q_top is observable via kelly_fraction_raw across the whole grid."""
    return ak.FleetRiskConfig(
        min_net_edge=-1.0,
        min_confidence=0.0,
        max_round_trip_cost=None,
        budget_sensitivity_veto_enabled=False)


def recovered_q_top(res, p_side):
    """Invert kelly_raw = (p - q) / (1 - q) -> q = (p - k) / (1 - k)."""
    k = res.kelly_fraction_raw
    assert 0.0 < k < 1.0, (res.action, k)
    return (p_side - k) / (1.0 - k)


# ---- 1. Prereg formula points -----------------------------------------------

def test_prereg_formula_points():
    for bid, fee, want_delta in ((0.65, 0.01, -0.0030), (0.30, 0.01, +0.0040)):
        old = round(1.0 - bid, 6) * (1.0 + fee)
        new = 1.0 - bid * (1.0 - fee)
        helper = ak._mirror_no_effective_ask(bid, fee)
        assert abs(helper - new) < 1e-12, (bid, helper, new)
        assert abs((old - new) - want_delta) < 1e-9, (bid, old - new, want_delta)


def test_helper_sign_flip_and_exactness():
    # Delta = fee*(1-2*bid): negative above 0.50, positive below, zero at 0.50.
    for bid in (0.05, 0.25, 0.50, 0.75, 0.95):
        for fee in (0.005, 0.01, 0.015):
            helper = ak._mirror_no_effective_ask(bid, fee)
            assert abs(helper - (1.0 - bid * (1.0 - fee))) < 1e-12, (bid, fee)
            old = round(1.0 - bid, 6) * (1.0 + fee)
            want = fee * (1.0 - 2.0 * bid)
            assert abs((old - helper) - want) < 2e-9, (bid, fee, old - helper, want)


# ---- 2. Grid: engine q_top vs mirror formula ----------------------------------

BIDS = [round(0.05 + 0.05 * i, 2) for i in range(19)]   # 0.05..0.95
FEES = (0.005, 0.01, 0.015)


def test_grid_q_top_matches_mirror_within_1e6():
    cfg = _forced_config()
    worst = 0.0
    for bid in BIDS:
        for fee in FEES:
            want = 1.0 - bid * (1.0 - fee)
            p_no = want + 0.5 * (1.0 - want)  # kelly_raw = 0.5 by construction
            assert 0.0 < 1.0 - p_no < 1.0, (bid, fee, p_no)
            res = ak.evaluate_and_size_opportunity(
                book=tight_book(bid, fee=fee), calibrated_prob_yes=1.0 - p_no,
                inter_agent_dispersion=0.0, portfolio_equity=100_000.0,
                cluster_exposure_used={}, config=cfg)
            assert res.action == ak.TradeAction.BUY_NO, (bid, fee, res.action, res.skip_reason)
            q = recovered_q_top(res, p_no)
            err = abs(q - want)
            worst = max(worst, err)
            assert err <= 1e-6, (bid, fee, q, want, err)
            # and it is NOT the old formula anywhere the two differ
            old = round(1.0 - bid, 6) * (1.0 + fee)
            if abs(old - want) > 1e-6:
                assert abs(q - old) > 1e-6, (bid, fee, q, old)
    print(f"   worst |q_top - mirror| over grid: {worst:.2e}")


# ---- 3. Borderline verdict flips at default config ----------------------------

def test_borderline_flip_bid_065():
    # bid 0.65, fee 0.01: old eff 0.3535 vs mirror 0.3565; p_no=0.4055
    # gives old edge 0.0520 (screen PASS) vs mirror edge 0.0490 (FAIL).
    res = ak.evaluate_and_size_opportunity(
        book=tight_book(0.65), calibrated_prob_yes=1.0 - 0.4055,
        inter_agent_dispersion=0.0, portfolio_equity=100_000.0,
        cluster_exposure_used={})
    assert res.action == ak.TradeAction.SKIP, res.action
    assert res.skip_reason and res.skip_reason.startswith("INSUFFICIENT_NET_EDGE"), res.skip_reason


def test_borderline_pass_just_above_hurdle_bid_065():
    # p_no=0.4075 -> mirror edge 0.0510: screen PASSES (LOW_CONVICTION
    # then, since p_side < 0.60 with ratio < 1.35 — the skip reason
    # proving the screen passed).
    res = ak.evaluate_and_size_opportunity(
        book=tight_book(0.65), calibrated_prob_yes=1.0 - 0.4075,
        inter_agent_dispersion=0.0, portfolio_equity=100_000.0,
        cluster_exposure_used={})
    assert res.action == ak.TradeAction.SKIP, res.action
    assert res.skip_reason and res.skip_reason.startswith("LOW_CONVICTION"), res.skip_reason


def test_borderline_flip_bid_030():
    # bid 0.30, fee 0.01: old eff 0.7070 vs mirror 0.7030; p_no=0.7540
    # gives old edge 0.0470 (FAIL) vs mirror edge 0.0510 (PASS).
    res = ak.evaluate_and_size_opportunity(
        book=tight_book(0.30), calibrated_prob_yes=1.0 - 0.7540,
        inter_agent_dispersion=0.0, portfolio_equity=100_000.0,
        cluster_exposure_used={})
    # Screen passed under the live (corrected) formula — it got past
    # INSUFFICIENT_NET_EDGE and LOW_CONVICTION into the sweep; the
    # sweep's own (uncorrected) per-rung fee stop then fails the fill
    # (SF-BL-8892 scope: the screen price only).
    assert res.action == ak.TradeAction.SKIP, res.action
    assert res.skip_reason and res.skip_reason.startswith(
        "ORDER_BOOK_DEPTH_EXHAUSTED_AT_REQUIRED_EDGE"), res.skip_reason


def test_bid_050_unchanged_by_construction():
    # At bid 0.50 the two formulas coincide; a borderline candidate
    # behaves identically — the correction is a no-op there.
    res = ak.evaluate_and_size_opportunity(
        book=tight_book(0.50), calibrated_prob_yes=1.0 - 0.5560,
        inter_agent_dispersion=0.0, portfolio_equity=100_000.0,
        cluster_exposure_used={})
    # mirror eff 0.505, edge 0.051 -> passes screen, LOW_CONVICTION
    assert res.action == ak.TradeAction.SKIP, res.action
    assert res.skip_reason and res.skip_reason.startswith("LOW_CONVICTION"), res.skip_reason


# ---- 4. YES side untouched; benchmark slate unchanged -------------------------

def test_yes_side_unchanged():
    # Fee is charged on the traded leg for YES: ask * (1 + fee).
    b = ak.OrderBookSnapshot(
        market_id="T", title="t", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C1", est_days_to_resolution=5,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=1000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=1000)], fee_rate=0.0)
    r = ak.evaluate_and_size_opportunity(b, 0.70, 0.0, 10000, {})
    assert r.action == ak.TradeAction.BUY_YES
    assert abs(r.kelly_fraction_raw - 0.40) < 1e-9, r
    assert r.allocated_fraction == 0.05 and r.invested_capital == 500.0, r


def test_benchmark_simulation_unchanged():
    # The historical +0.1059 disciplined pin is stale in the current
    # combined tree: Gate D / Gate BS era landings and the concurrent
    # SF-BL-8890 / SF-BL-8891 fixes moved the disciplined slate to
    # +0.0547 (the unlock scenario now SKIPs at Gate D) BEFORE this
    # task's patch. SF-BL-8892's own neutrality was verified directly:
    # benchmark output is byte-identical between the live module and a
    # reconstruction with only this task's two hunks reverted (diff
    # empty; see DONE-CLAIM.md). This test pins the current combined
    # values so any FUTURE change to them is visible.
    buf = io.StringIO()
    with redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    assert '"north_star_score": 0.0547' in out, out[:2000]
    assert '"north_star_score": -0.1499' in out, out[:2000]


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
