"""
test_dispersion_finiteness.py — SF-BL-9395 verification tests.

Pre-registered done/kill criterion: "Fixture: NaN, +inf and -inf dispersion
must all SKIP with an explicit invalid-input reason and zero size, while
finite dispersion behaviour is byte-identical; kill if any non-finite
dispersion reaches the sizing step on the fixture book."

Pre-fix reproduction (verified live 2026-10-21, before the patch), prereg
fixture book (p=0.80, best ask 0.50 / bid 0.48, fee_rate 0.01, equity
$100,000, empty cluster map):
    dispersion= nan  -> BUY_YES invested $15,000.00 (full cluster cap)
    dispersion= -inf -> BUY_YES invested $15,000.00 (full cluster cap)
    dispersion= +inf -> SKIP UNRESOLVED_AGENT_DISPERSION (inf > 0.22)
                        (SKIP only incidentally; no invalid-input reason)
    dispersion= 0.50 -> SKIP UNRESOLVED_AGENT_DISPERSION (0.50 > 0.22)
    dispersion= 0.10 -> BUY_YES invested $15,000.00, q_exec 0.505

Root cause: Gate 2 in evaluate_and_size_opportunity was the bare comparison
``inter_agent_dispersion > config.max_inter_agent_dispersion``; NaN never
satisfies ``>`` and -inf never exceeds a finite threshold.

Fix: Gate 0.75 in evaluate_and_size_opportunity (after the Gate 0 fee check
and Gate 0.5 empty-book check, before Gate 1 / Gate 2 / sizing) rejects any
non-finite (or non-numeric / bool) dispersion with SKIP +
INVALID_INTER_AGENT_DISPERSION and zero size (EC-09). Finite values pass
the check untouched, so every finite path is byte-identical.

Run:  python3 test_dispersion_finiteness.py   (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_cli_edge.py / test_alpha_kelly_fee_rate.py.
"""
import contextlib
import io
import math
import os
import sys

# Locate the repo's inner package dir from either home (work-dir copy) or
# this file's location (repo tests/ copy).
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
    """The prereg fixture book: ask 0.50 / bid 0.48, fee 0.01, deep size."""
    return ak.OrderBookSnapshot(
        market_id="FIXTURE-9395", title="prereg fixture", domain=domain,
        factor_cluster="C9395", est_days_to_resolution=10,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100000)],
        fee_rate=0.01, on_chain_verified=on_chain)


def _eval(disp, p=0.80, book=None, used=None):
    return ak.evaluate_and_size_opportunity(
        book if book is not None else _book(), p, disp, 100000.0,
        used if used is not None else {})


def _assert_zero_skip(r):
    assert r.action == ak.TradeAction.SKIP, r
    assert r.invested_capital == 0.0 and r.shares_acquired == 0.0, r
    assert r.exec_price == 0.0 and r.liquidation_bid_price == 0.0, r
    assert r.kelly_fraction_raw == 0.0 and r.allocated_fraction == 0.0, r  # EC-09
    assert r.kl_divergence_nats == 0.0 and r.expected_log_growth == 0.0, r


# ---- 1. Done criterion: every non-finite dispersion SKIPs, invalid reason --

def test_nonfinite_dispersions_skip_with_invalid_reason():
    for bad in (float("nan"), float("inf"), float("-inf")):
        r = _eval(bad)
        _assert_zero_skip(r)
        assert r.skip_reason is not None, r
        assert r.skip_reason.startswith("INVALID_INTER_AGENT_DISPERSION"), r.skip_reason
        assert "not finite" in r.skip_reason, r.skip_reason


def test_nonfinite_never_reaches_sizing_step():
    # Kill criterion: sizing would consume cluster room (15% cap = $15,000
    # on this fixture pre-fix). After each non-finite call the cluster map
    # must be untouched and no capital invested.
    for bad in (float("nan"), float("inf"), float("-inf")):
        used = {}
        r = _eval(bad, used=used)
        _assert_zero_skip(r)
        assert used == {}, (bad, used)


def test_nonfinite_skips_on_buy_no_side_too():
    # p=0.20 makes BUY_NO the candidate side; pre-fix NaN sized it too.
    for bad in (float("nan"), float("inf"), float("-inf")):
        r = _eval(bad, p=0.20)
        _assert_zero_skip(r)
        assert r.skip_reason.startswith("INVALID_INTER_AGENT_DISPERSION"), r.skip_reason


def test_nonfinite_reason_is_not_unresolved_dispersion():
    # +inf pre-fix SKIPped only incidentally as UNRESOLVED; the reason must
    # now be the explicit invalid-input reason for all three non-finites.
    for bad in (float("nan"), float("inf"), float("-inf")):
        r = _eval(bad)
        assert "UNRESOLVED_AGENT_DISPERSION" not in (r.skip_reason or ""), r.skip_reason


def test_nonfinite_gate_precedes_crypto_gate():
    # Input-validity gates run before business gates (Gate 0 fee precedent):
    # a crypto book failing Gate 1 with NaN dispersion reports INVALID.
    b = _book(domain=ak.MarketDomain.CRYPTO, on_chain=False)
    r = _eval(float("nan"), book=b)
    _assert_zero_skip(r)
    assert r.skip_reason.startswith("INVALID_INTER_AGENT_DISPERSION"), r.skip_reason


def test_nonnumeric_and_bool_dispersion_skip_invalid():
    # Documented extension of the same gate: values that cannot be ordered
    # against the threshold at all are invalid input, never a TypeError and
    # never a trade. (bool True pre-fix masqueraded as dispersion 1.0.)
    for bad in (None, "0.10", True, False):
        r = _eval(bad)
        _assert_zero_skip(r)
        assert r.skip_reason.startswith("INVALID_INTER_AGENT_DISPERSION"), (bad, r.skip_reason)


# ---- 2. Finite behaviour is byte-identical ---------------------------------

# Expected snapshots recorded from the PRE-FIX code on the fixture book
# (full dataclass field tuples, in asdict order is not assumed; fields are
# compared individually below).
_FINITE_CASES = [
    # disp, action, invested, shares, exec, kelly_raw, alloc, skip_reason
    (0.50, "SKIP", 0.0, 0.0, 0.0, 0.0, 0.0,
     "UNRESOLVED_AGENT_DISPERSION (0.50 > 0.22)"),
    (1.00, "SKIP", 0.0, 0.0, 0.0, 0.0, 0.0,
     "UNRESOLVED_AGENT_DISPERSION (1.00 > 0.22)"),
    (0.10, "BUY_YES", 15000.0, 29702.970297029704, 0.505,
     0.595959595959596, 0.15, None),
    (0.22, "BUY_YES", 15000.0, 29702.970297029704, 0.505,
     0.595959595959596, 0.15, None),  # boundary: > is strict, 0.22 trades
    (0.00, "BUY_YES", 15000.0, 29702.970297029704, 0.505,
     0.595959595959596, 0.15, None),
    (-0.05, "BUY_YES", 15000.0, 29702.970297029704, 0.505,
     0.595959595959596, 0.15, None),  # finite negative: pre-fix traded; unchanged
]


def test_finite_cases_byte_identical():
    for disp, action, inv, shares, q, kelly, alloc, reason in _FINITE_CASES:
        r = _eval(disp)
        assert r.action.value == action, (disp, r)
        assert r.invested_capital == inv, (disp, r.invested_capital)
        assert r.shares_acquired == shares, (disp, r.shares_acquired)
        assert r.exec_price == q, (disp, r.exec_price)
        assert r.kelly_fraction_raw == kelly, (disp, r.kelly_fraction_raw)
        assert r.allocated_fraction == alloc, (disp, r.allocated_fraction)
        assert r.skip_reason == reason, (disp, r.skip_reason)


def test_finite_int_dispersions_unchanged():
    # int is a finite real: 0 trades exactly like 0.0, 1 SKIPs unresolved.
    r0 = _eval(0)
    assert r0.action == ak.TradeAction.BUY_YES and r0.invested_capital == 15000.0, r0
    r1 = _eval(1)
    assert r1.action == ak.TradeAction.SKIP, r1
    assert r1.skip_reason == "UNRESOLVED_AGENT_DISPERSION (1.00 > 0.22)", r1.skip_reason


def test_finite_dispersion_still_consumes_cluster_room():
    used = {}
    r = _eval(0.10, used=used)
    assert r.action == ak.TradeAction.BUY_YES, r
    assert used == {"C9395": 0.15}, used


def test_helper_directly():
    for good in (0.0, 0.22, 1, -0.05, 1e308):
        assert ak._dispersion_is_finite(good) is True, good
    for bad in (float("nan"), float("inf"), float("-inf"), None, "0.1", True):
        assert ak._dispersion_is_finite(bad) is False, bad
    assert math.isfinite(0.22)  # sanity: stdlib semantics the helper wraps


# ---- 3. No regression on the built-in benchmark -----------------------------

def test_benchmark_nstar_unchanged():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    assert '"north_star_score": 0.1059' in out, out[:400]
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
