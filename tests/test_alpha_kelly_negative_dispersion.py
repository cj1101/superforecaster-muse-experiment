"""
test_negative_dispersion.py — SF-BL-9714 verification tests.

Pre-registered done/kill criterion: "Negative finite dispersion SKIPs
zero-size with the explicit INVALID_INTER_AGENT_DISPERSION reason at the
existing Gate 0.75 (or raises); success = the -0.5 and -100 repros SKIP
with that reason and every finite dispersion >= 0 fixture is
byte-identical; kill if a caller census proves dispersion only ever
reaches evaluate from aggregate_fleet_forecasts' pstdev under a tested
non-negativity invariant." Kill audit: CALLER_CENSUS.md — NOT met.

Pre-fix repro (repro.py / repro_prefix.txt, standard fixture p=0.80,
ask 0.50 / bid 0.48, fee 0.01, equity $100,000):
    dispersion=-100 / -0.5 / -0.05 -> BUY_YES $15,000 (full cluster cap),
    byte-identical to dispersion=0.10.

Fix: Gate 0.75 now uses _dispersion_in_domain (finite AND >= 0). The
non-finite reason is byte-identical to SF-BL-9395; the negative branch
reads "... is negative: dispersion is a pstdev and cannot be < 0".
Finite >= 0 (incl. -0.0) passes untouched.

Run: python3 test_negative_dispersion.py (exit 0 iff all pass)
"""
import contextlib
import io
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


def _book(domain=ak.MarketDomain.PREDICTION_MARKET, on_chain=True):
    return ak.OrderBookSnapshot(
        market_id="FIXTURE-9714", title="prereg fixture", domain=domain,
        factor_cluster="C9714", est_days_to_resolution=10,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100000)],
        fee_rate=0.01, on_chain_verified=on_chain)


def _eval(disp, p=0.80, book=None, used=None, config=None):
    kwargs = {}
    if config is not None:
        kwargs["config"] = config
    return ak.evaluate_and_size_opportunity(
        book if book is not None else _book(), p, disp, 100000.0,
        used if used is not None else {}, **kwargs)


def _assert_zero_skip(r):
    assert r.action == ak.TradeAction.SKIP, r
    assert r.invested_capital == 0.0 and r.shares_acquired == 0.0, r
    assert r.exec_price == 0.0 and r.liquidation_bid_price == 0.0, r
    assert r.kelly_fraction_raw == 0.0 and r.allocated_fraction == 0.0, r
    assert r.kelly_fraction_sized == 0.0, r
    assert r.kl_divergence_nats == 0.0 and r.expected_log_growth == 0.0, r


# ---- 1. Done criterion: -0.5 and -100 (and other negatives) SKIP ----------

def test_prereg_repros_skip_with_invalid_reason():
    for bad in (-0.5, -100, -100.0, -0.05, -1, -1e-9, -0.000001):
        r = _eval(bad)
        _assert_zero_skip(r)
        assert r.skip_reason is not None, r
        assert r.skip_reason.startswith("INVALID_INTER_AGENT_DISPERSION"), (bad, r.skip_reason)
        assert "negative" in r.skip_reason, (bad, r.skip_reason)
        assert repr(bad) in r.skip_reason or str(bad) in r.skip_reason, (bad, r.skip_reason)


def test_negative_never_reaches_sizing_or_cluster():
    for bad in (-0.5, -100):
        used = {}
        r = _eval(bad, used=used)
        _assert_zero_skip(r)
        assert used == {}, (bad, used)


def test_negative_skips_on_buy_no_side_too():
    for bad in (-0.5, -100):
        r = _eval(bad, p=0.20)
        _assert_zero_skip(r)
        assert r.skip_reason.startswith("INVALID_INTER_AGENT_DISPERSION"), r.skip_reason


def test_negative_gate_precedes_crypto_and_taper():
    b = _book(domain=ak.MarketDomain.CRYPTO, on_chain=False)
    r = _eval(-0.5, book=b)
    _assert_zero_skip(r)
    assert r.skip_reason.startswith("INVALID_INTER_AGENT_DISPERSION"), r.skip_reason
    cfg = ak.FleetRiskConfig(dispersion_taper_enabled=True)
    r2 = _eval(-0.5, config=cfg)
    _assert_zero_skip(r2)
    assert r2.skip_reason.startswith("INVALID_INTER_AGENT_DISPERSION"), r2.skip_reason


def test_domain_helper():
    for good in (0.0, -0.0, 0.10, 0.22, 1, 1e308):
        assert ak._dispersion_in_domain(good) is True, good
    for bad in (-0.5, -100, -1, float("nan"), float("inf"), float("-inf"),
                None, "0.10", True, False):
        assert ak._dispersion_in_domain(bad) is False, bad
    # SF-BL-9395 helper semantics unchanged: negative finite IS finite.
    assert ak._dispersion_is_finite(-0.5) is True
    assert ak._dispersion_is_finite(float("nan")) is False


# ---- 2. Non-finite reason byte-identical to SF-BL-9395 --------------------

def test_nonfinite_reason_unchanged():
    for bad, rep in ((float("nan"), "nan"), (float("inf"), "inf"),
                     (float("-inf"), "-inf")):
        r = _eval(bad)
        _assert_zero_skip(r)
        assert r.skip_reason == (
            f"INVALID_INTER_AGENT_DISPERSION "
            f"(inter_agent_dispersion={rep} is not finite)"), r.skip_reason


# ---- 3. Finite >= 0 byte-identical (snapshots recorded pre-fix) -----------

_FINITE_CASES = [
    # disp, action, invested, shares, exec, kelly_raw, alloc, liq, skip
    (0.0, "BUY_YES", 15000.0, 29702.970297029704, 0.505,
     0.595959595959596, 0.15, 0.48, None),
    (-0.0, "BUY_YES", 15000.0, 29702.970297029704, 0.505,
     0.595959595959596, 0.15, 0.48, None),
    (0.10, "BUY_YES", 15000.0, 29702.970297029704, 0.505,
     0.595959595959596, 0.15, 0.48, None),
    (0.22, "BUY_YES", 15000.0, 29702.970297029704, 0.505,
     0.595959595959596, 0.15, 0.48, None),
    (0, "BUY_YES", 15000.0, 29702.970297029704, 0.505,
     0.595959595959596, 0.15, 0.48, None),
    (0.50, "SKIP", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     "UNRESOLVED_AGENT_DISPERSION (0.50 > 0.22)"),
    (1.00, "SKIP", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     "UNRESOLVED_AGENT_DISPERSION (1.00 > 0.22)"),
    (1, "SKIP", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     "UNRESOLVED_AGENT_DISPERSION (1.00 > 0.22)"),
]


def test_finite_nonnegative_byte_identical():
    for disp, action, inv, shares, ex, kr, alloc, liq, reason in _FINITE_CASES:
        used = {}
        r = _eval(disp, used=used)
        assert r.action.value == action, (disp, r)
        assert r.invested_capital == inv, (disp, r)
        assert r.shares_acquired == shares, (disp, r)
        assert r.exec_price == ex, (disp, r)
        assert r.kelly_fraction_raw == kr, (disp, r)
        assert r.allocated_fraction == alloc, (disp, r)
        assert r.liquidation_bid_price == liq, (disp, r)
        assert r.skip_reason == reason, (disp, r)
        if action == "BUY_YES":
            assert used == {"C9714": 0.15}, (disp, used)
        else:
            assert used == {}, (disp, used)


def test_aggregate_path_unchanged():
    members = [
        ak.SubAgentForecast("a1", "gemini-3-pro", 0.77, 1.2),
        ak.SubAgentForecast("a2", "gpt-5", 0.79, 1.1),
        ak.SubAgentForecast("a3", "claude-4.5-sonnet", 0.75, 1.0),
        ak.SubAgentForecast("a4", "grok-4", 0.80, 1.1),
    ]
    p, disp = ak.aggregate_fleet_forecasts(members)
    assert disp >= 0.0, disp
    r = _eval(disp, p=p)
    assert r.action in (ak.TradeAction.BUY_YES, ak.TradeAction.SKIP), r


def test_benchmark_nstar_unchanged():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    # Recorded from the pre-fix code this run (2026-10-23).
    assert '"north_star_score": 0.0547' in out, out[:400]
    assert '"north_star_score": -0.1499' in out, out[:400]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        try:
            fn()
            check(fn.__name__, True)
        except Exception as e:  # noqa: BLE001
            check(fn.__name__, False, f"{type(e).__name__}: {e}")
    print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
    sys.exit(0 if all(PASS) and PASS else 1)
