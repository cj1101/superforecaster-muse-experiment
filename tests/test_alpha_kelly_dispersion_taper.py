"""
test_dispersion_taper.py — SF-BL-9639 verification tests.

Pre-registered done/kill criterion (TASK.md): "On forward paper books,
tapered sizing (lambda x max(0, 1 - dispersion/0.22)) beats the binary
gate on N_star with MDD no worse over the pre-registered forward window;
kill if CWR_bid falls or trade count drops >30% vs the binary-gate
control." The forward-window half is replay.py / REPORT.md in the task
work dir; this file tests the shipped mechanism: the helper's exact
arithmetic, the Gate 2 replacement behind
FleetRiskConfig.dispersion_taper_enabled, and byte-identical behaviour
with the flag off (the production default).

Pre-fix cliff (pristine module, fixture p=0.80, ask 0.50 / bid 0.48,
fee 0.01, equity $100,000): dispersion 0.21 -> BUY_YES $15,000 (full
cluster cap); dispersion 0.23 -> SKIP UNRESOLVED_AGENT_DISPERSION.

Run:  python3 test_dispersion_taper.py   (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_dispersion.py.
"""
import contextlib
import io
import math
import os
import sys

_CANDIDATES = [
    os.path.expanduser("~/workspace/superforecaster/superforecaster"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "superforecaster"),
]
for _c in _CANDIDATES:
    if os.path.isfile(os.path.join(_c, "alpha_kelly.py")):
        sys.path.insert(0, _c)
        break

import alpha_kelly as ak  # noqa: E402

PASS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(domain=ak.MarketDomain.PREDICTION_MARKET, on_chain=True,
          ask=0.50, bid=0.48):
    return ak.OrderBookSnapshot(
        market_id="FIXTURE-9639", title="prereg fixture", domain=domain,
        factor_cluster="C9639", est_days_to_resolution=10,
        yes_asks=[ak.OrderBookLevel(price=ask, size_shares=100000)],
        yes_bids=[ak.OrderBookLevel(price=bid, size_shares=100000)],
        fee_rate=0.01, on_chain_verified=on_chain)


def _uncapped_book():
    """Book whose Kelly target sits far below the 15% cluster cap on
    $10,000 equity, so sizing ratios are directly observable:
    p=0.70 vs ask 0.62 (fee-adj 0.6262): kelly_raw 0.1974, binary
    target fraction 0.0691 (~$691)."""
    return _book(ask=0.62, bid=0.60)


def _eval(disp, p=0.80, book=None, used=None, taper=False, equity=100000.0):
    cfg = ak.FleetRiskConfig(dispersion_taper_enabled=taper)
    return ak.evaluate_and_size_opportunity(
        book if book is not None else _book(), p, disp, equity,
        used if used is not None else {}, config=cfg)


def _assert_zero_skip(r):
    assert r.action == ak.TradeAction.SKIP, r
    assert r.invested_capital == 0.0 and r.shares_acquired == 0.0, r
    assert r.exec_price == 0.0 and r.liquidation_bid_price == 0.0, r
    assert r.kelly_fraction_raw == 0.0 and r.allocated_fraction == 0.0, r
    assert r.kl_divergence_nats == 0.0 and r.expected_log_growth == 0.0, r


# ---- 1. Helper: exact pre-registered arithmetic ----------------------------

def test_helper_grid_points():
    f = ak.dispersion_taper_factor
    cases = [(0.0, 1.0), (0.055, 0.75), (0.11, 0.50), (0.165, 0.25),
             (0.198, 0.10), (0.22, 0.0), (0.231, 0.0), (0.50, 0.0),
             (1.0, 0.0)]
    for d, want in cases:
        got = f(d, 0.22)
        assert math.isclose(got, want, rel_tol=0, abs_tol=1e-12), (d, got, want)


def test_helper_reads_threshold_not_literal():
    f = ak.dispersion_taper_factor
    assert math.isclose(f(0.10, 0.40), 0.75, abs_tol=1e-12)
    assert math.isclose(f(0.30, 0.40), 0.25, abs_tol=1e-12)
    assert f(0.40, 0.40) == 0.0


def test_helper_negative_dispersion_clips_to_full_size():
    # Finite negative dispersion trades full-size under the binary gate
    # (SF-BL-9395 pinned); the taper must not amplify lambda above 1x.
    assert ak.dispersion_taper_factor(-0.05, 0.22) == 1.0
    assert ak.dispersion_taper_factor(-1.0, 0.22) == 1.0


def test_helper_zero_threshold_limit():
    f = ak.dispersion_taper_factor
    assert f(0.0, 0.0) == 1.0
    assert f(0.01, 0.0) == 0.0


def test_helper_corrupt_inputs_fail_to_full_size():
    f = ak.dispersion_taper_factor
    for bad_d in (float("nan"), float("inf"), float("-inf"), None, "0.1", True):
        assert f(bad_d, 0.22) == 1.0, bad_d
    for bad_t in (float("nan"), float("inf"), None, "0.22", True, -0.1, 1.5):
        assert f(0.11, bad_t) == 1.0, bad_t


# ---- 2. Flag OFF: binary cliff byte-identical ------------------------------

def test_flag_off_cliff_unchanged():
    r21 = _eval(0.21, taper=False)
    assert r21.action == ak.TradeAction.BUY_YES and r21.invested_capital == 15000.0, r21
    r22 = _eval(0.22, taper=False)
    assert r22.action == ak.TradeAction.BUY_YES and r22.invested_capital == 15000.0, r22
    r23 = _eval(0.23, taper=False)
    _assert_zero_skip(r23)
    assert r23.skip_reason == "UNRESOLVED_AGENT_DISPERSION (0.23 > 0.22)", r23.skip_reason


def test_flag_off_default_config_is_binary():
    # The shared default config (no explicit config argument) is the
    # binary arm: dispersion_taper_enabled defaults to False.
    assert ak.FleetRiskConfig().dispersion_taper_enabled is False
    r = ak.evaluate_and_size_opportunity(_book(), 0.80, 0.50, 100000.0, {})
    assert r.skip_reason == "UNRESOLVED_AGENT_DISPERSION (0.50 > 0.22)", r.skip_reason


# ---- 3. Flag ON: continuous taper replaces the cliff -----------------------

def test_flag_on_zero_dispersion_identical_to_binary():
    rb = _eval(0.0, p=0.70, book=_uncapped_book(), taper=False, equity=10000.0)
    rt = _eval(0.0, p=0.70, book=_uncapped_book(), taper=True, equity=10000.0)
    assert rt.action == ak.TradeAction.BUY_YES, rt
    assert rt.invested_capital == rb.invested_capital, (rt, rb)
    assert rt.kelly_fraction_sized == rb.kelly_fraction_sized, (rt, rb)


def test_flag_on_half_dispersion_half_size():
    rb = _eval(0.0, p=0.70, book=_uncapped_book(), taper=False, equity=10000.0)
    rt = _eval(0.11, p=0.70, book=_uncapped_book(), taper=True, equity=10000.0)
    assert rt.action == ak.TradeAction.BUY_YES, rt
    assert math.isclose(rt.kelly_fraction_sized, rb.kelly_fraction_sized * 0.5,
                        rel_tol=1e-12), (rt.kelly_fraction_sized, rb.kelly_fraction_sized)
    assert math.isclose(rt.invested_capital, rb.invested_capital * 0.5,
                        rel_tol=1e-9), (rt.invested_capital, rb.invested_capital)
    # kelly_fraction_raw is the unscaled diagnostic in both arms.
    assert rt.kelly_fraction_raw == rb.kelly_fraction_raw


def test_flag_on_factor_grid_scales_sized_fraction():
    rb = _eval(0.0, p=0.70, book=_uncapped_book(), taper=False, equity=10000.0)
    for d, factor in ((0.055, 0.75), (0.165, 0.25), (0.198, 0.10)):
        rt = _eval(d, p=0.70, book=_uncapped_book(), taper=True, equity=10000.0)
        assert rt.action == ak.TradeAction.BUY_YES, (d, rt)
        assert math.isclose(rt.kelly_fraction_sized,
                            rb.kelly_fraction_sized * factor, rel_tol=1e-9), \
            (d, rt.kelly_fraction_sized, rb.kelly_fraction_sized)


def test_flag_on_at_threshold_sizes_zero_not_full():
    # Boundary registered in PREREG: binary trades at d == 0.22 (strict
    # >), the task's formula gives factor 0 there.
    used = {}
    rt = _eval(0.22, taper=True, used=used)
    _assert_zero_skip(rt)
    assert rt.skip_reason.startswith("DISPERSION_TAPER_ZERO"), rt.skip_reason
    assert used == {}, used


def test_flag_on_above_threshold_taper_zero_not_unresolved():
    for d in (0.231, 0.50, 1.0):
        used = {}
        rt = _eval(d, taper=True, used=used)
        _assert_zero_skip(rt)
        assert rt.skip_reason.startswith("DISPERSION_TAPER_ZERO"), (d, rt.skip_reason)
        assert "UNRESOLVED_AGENT_DISPERSION" not in rt.skip_reason, rt.skip_reason
        assert used == {}, used


def test_flag_on_negative_dispersion_matches_binary():
    rb = _eval(-0.05, p=0.70, book=_uncapped_book(), taper=False, equity=10000.0)
    rt = _eval(-0.05, p=0.70, book=_uncapped_book(), taper=True, equity=10000.0)
    assert rt.action == rb.action == ak.TradeAction.BUY_YES, (rt, rb)
    assert rt.invested_capital == rb.invested_capital, (rt, rb)
    assert rt.kelly_fraction_sized == rb.kelly_fraction_sized, (rt, rb)


def test_flag_on_buy_no_side_tapers_too():
    # p=0.40: NO side p_side=0.60, kelly_raw 0.329, binary target
    # 0.1151 — below the cluster cap, so the taper ratio is observable
    # (at p=0.30 the binary NO target caps at 0.15 and masks it).
    rb = _eval(0.0, p=0.40, book=_uncapped_book(), taper=False, equity=10000.0)
    rt = _eval(0.11, p=0.40, book=_uncapped_book(), taper=True, equity=10000.0)
    assert rb.action == rt.action == ak.TradeAction.BUY_NO, (rb, rt)
    assert math.isclose(rt.kelly_fraction_sized, rb.kelly_fraction_sized * 0.5,
                        rel_tol=1e-9), (rt, rb)


def test_flag_on_crypto_dampener_still_applies_under_taper():
    book = _book(domain=ak.MarketDomain.CRYPTO, on_chain=True, ask=0.62, bid=0.60)
    rt = _eval(0.11, p=0.70, book=book, taper=True, equity=10000.0)
    rb = _eval(0.0, p=0.70, book=_uncapped_book(), taper=False, equity=10000.0)
    # crypto: raw * 0.35 (lambda) * 0.50 (dampener) * 0.50 (taper)
    assert math.isclose(rt.kelly_fraction_sized,
                        rb.kelly_fraction_sized * 0.25, rel_tol=1e-9), (rt, rb)


def test_flag_on_taper_monotone_in_dispersion():
    sizes = []
    for d in (0.0, 0.055, 0.11, 0.165, 0.198):
        r = _eval(d, p=0.70, book=_uncapped_book(), taper=True, equity=10000.0)
        assert r.action == ak.TradeAction.BUY_YES, (d, r)
        sizes.append(r.invested_capital)
    assert all(a > b for a, b in zip(sizes, sizes[1:])), sizes


# ---- 4. Gate precedence under the taper ------------------------------------

def test_flag_on_nonfinite_dispersion_still_invalid_input():
    # Gate 0.75 (SF-BL-9395) precedes the taper: a non-finite dispersion
    # is a data error in both arms, never a taper computation.
    for bad in (float("nan"), float("inf"), float("-inf")):
        r = _eval(bad, taper=True)
        _assert_zero_skip(r)
        assert r.skip_reason.startswith("INVALID_INTER_AGENT_DISPERSION"), r.skip_reason


def test_flag_on_invalid_threshold_still_invalid_config():
    # Gate 1.9 (SF-BL-10844) precedes the taper: mutate post-construction.
    cfg = ak.FleetRiskConfig(dispersion_taper_enabled=True)
    cfg.max_inter_agent_dispersion = float("nan")
    r = ak.evaluate_and_size_opportunity(_book(), 0.80, 0.11, 100000.0, {}, config=cfg)
    _assert_zero_skip(r)
    assert r.skip_reason.startswith("INVALID_MAX_INTER_AGENT_DISPERSION"), r.skip_reason


def test_flag_on_custom_threshold_tapers_to_it():
    cfg = ak.FleetRiskConfig(dispersion_taper_enabled=True,
                             max_inter_agent_dispersion=0.40)
    r = ak.evaluate_and_size_opportunity(
        _uncapped_book(), 0.70, 0.30, 10000.0, {}, config=cfg)
    rb = _eval(0.0, p=0.70, book=_uncapped_book(), taper=False, equity=10000.0)
    assert r.action == ak.TradeAction.BUY_YES, r
    assert math.isclose(r.kelly_fraction_sized, rb.kelly_fraction_sized * 0.25,
                        rel_tol=1e-9), (r, rb)


# ---- 5. No regression on the built-in benchmark -----------------------------

def test_benchmark_nstar_unchanged():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    # Current-module values, verified byte-identical (sha256
    # 1df34fe8877e738d643c14283a2b900b11b6c0c48cb26f7a2d7083da8254df98)
    # on the pristine pre-task module: the header comment's +0.1059
    # predates later module changes; 0.0547 is what the module prints
    # both before and after this task's change.
    assert '"north_star_score": 0.0547' in out, out[:400]
    assert '"north_star_score": -0.1499' in out, out[:400]


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
