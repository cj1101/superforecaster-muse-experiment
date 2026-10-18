"""
test_cli_edge.py — EC-14 gate tests + EC-09 SKIP-Kelly regression tests.

Run:  python3 tests/test_cli_edge.py     (or: python3 -m pytest tests/test_cli_edge.py)
No pytest dependency: every check is a test_* function using assert, and
__main__ runs them all.

EC-09 regression: a SKIP must size zero. Before the fix, `sf edge` computed
the Kelly fraction BEFORE the gate decisions and emitted it unchanged on
every SKIP branch (e.g. --p 0.55 --q-exec 0.45 -> SKIP by Gate D with
kelly_fraction_of_account 0.0636), and alpha_kelly's CLUSTER_CAP_REACHED /
ORDER_BOOK_DEPTH_EXHAUSTED SKIP returns leaked kelly_fraction_raw. These
tests assert kelly == 0.0 (and expected log growth == 0.0) on EVERY SKIP
branch of both surfaces, and that TRADE-branch numbers are unchanged.
"""
import json
import subprocess
import sys

import os as _os
_sys_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'superforecaster')
sys.path.insert(0, _sys_path)  # repo layout bootstrap

import alpha_kelly as ak  # noqa: E402

CLI = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'superforecaster', 'cli.py')
PASS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def sf_edge(*args):
    out = subprocess.run([sys.executable, CLI, "edge", *args],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def assert_skip_zero(payload, label):
    assert payload["decision"] == "SKIP", f"{label}: not SKIP: {payload}"
    assert payload["side"] is None, f"{label}: side not null on SKIP: {payload}"
    assert payload["kelly_fraction_of_account"] == 0.0, \
        f"{label}: EC-09 leak, kelly={payload['kelly_fraction_of_account']}"
    assert payload["expected_log_growth_at_kelly_size"] == 0.0, \
        f"{label}: EC-09 sibling leak, growth={payload['expected_log_growth_at_kelly_size']}"


# ---- CLI SKIP branches -----------------------------------------------------

def test_cli_skip_insufficient_net_edge():
    # |p - q| = 0.02 < 0.05 hurdle (pre-fix kelly was 0.014).
    assert_skip_zero(sf_edge("--p", "0.52", "--q-exec", "0.50"), "insufficient-edge")


def test_cli_skip_gate_d_long():
    # EC-14 repro: p_side 0.55, LR 1.222 < 1.35 and < 0.60 (pre-fix kelly 0.0636).
    r = sf_edge("--p", "0.55", "--q-exec", "0.45")
    assert r["skip_reason"] == "LOW_CONVICTION (Gate D)"
    assert_skip_zero(r, "gate-d-long")


def test_cli_skip_gate_d_short_side():
    # Same conviction failure on the BUY_NO side (p_side = 1 - p = 0.55).
    r = sf_edge("--p", "0.45", "--q-exec", "0.55")
    assert r["skip_reason"] == "LOW_CONVICTION (Gate D)"
    assert_skip_zero(r, "gate-d-short")


def test_cli_skip_gate_a_crypto_unverified():
    # Passes Gate D (LR 1.381) but crypto without on-chain verification
    # (pre-fix kelly was 0.0483, the dampened TRADE size).
    r = sf_edge("--p", "0.58", "--q-exec", "0.42", "--domain", "crypto")
    assert r["skip_reason"] == "FAILED_CRYPTOBENCH_ONCHAIN_GATE (Gate A)"
    assert_skip_zero(r, "gate-a")


def test_cli_skip_gate_a_and_insufficient_edge_crypto():
    r = sf_edge("--p", "0.52", "--q-exec", "0.50", "--domain", "crypto")
    assert_skip_zero(r, "crypto-insufficient")


# ---- CLI TRADE branches: numbers must be exactly the pre-fix values --------

def test_cli_trade_gate_d_pass_unchanged():
    r = sf_edge("--p", "0.58", "--q-exec", "0.42")
    assert r["decision"] == "TRADE" and r["side"] == "BUY_YES"
    assert r["kelly_fraction_of_account"] == 0.0966, r
    assert r["expected_log_growth_at_kelly_size"] == 0.02995, r


def test_cli_trade_crypto_verified_unchanged():
    r = sf_edge("--p", "0.58", "--q-exec", "0.42", "--domain", "crypto", "--onchain-verified")
    assert r["decision"] == "TRADE" and r["side"] == "BUY_YES"
    assert r["kelly_fraction_of_account"] == 0.0483, r  # 0.5x CryptoBench dampener
    assert r["expected_log_growth_at_kelly_size"] == 0.01665, r


def test_cli_trade_cluster_cap_unchanged():
    r = sf_edge("--p", "0.80", "--q-exec", "0.50")
    assert r["decision"] == "TRADE"
    assert r["kelly_fraction_of_account"] == 0.15, r  # 15% cluster cap binds
    assert r["expected_log_growth_at_kelly_size"] == 0.07931, r


def test_cli_trade_buy_no_unchanged():
    r = sf_edge("--p", "0.20", "--q-exec", "0.50")
    assert r["decision"] == "TRADE" and r["side"] == "BUY_NO"
    assert r["kelly_fraction_of_account"] == 0.15, r


# ---- alpha_kelly SKIP branches ----------------------------------------------

def _book(domain=ak.MarketDomain.PREDICTION_MARKET, ask=0.50, bid=0.48,
          fee=0.0, verified=True, cluster="C1", ask_size=1000):
    return ak.OrderBookSnapshot(
        market_id="T", title="t", domain=domain, factor_cluster=cluster,
        est_days_to_resolution=5,
        yes_asks=[ak.OrderBookLevel(price=ask, size_shares=ask_size)],
        yes_bids=[ak.OrderBookLevel(price=bid, size_shares=1000)],
        fee_rate=fee, on_chain_verified=verified)


def assert_ak_skip_zero(res, label):
    assert res.action == ak.TradeAction.SKIP, f"{label}: not SKIP: {res.action}"
    assert res.kelly_fraction_raw == 0.0, f"{label}: EC-09 leak kelly_raw={res.kelly_fraction_raw}"
    assert res.allocated_fraction == 0.0, f"{label}: allocated={res.allocated_fraction}"
    assert res.invested_capital == 0.0 and res.shares_acquired == 0.0, label
    assert res.expected_log_growth == 0.0, label


def test_ak_skip_gate_a_onchain():
    assert_ak_skip_zero(ak.evaluate_and_size_opportunity(
        _book(domain=ak.MarketDomain.CRYPTO, verified=False, ask=0.42, bid=0.40),
        0.58, 0.0, 10000, {}), "ak-gate-a")


def test_ak_skip_dispersion():
    assert_ak_skip_zero(ak.evaluate_and_size_opportunity(
        _book(), 0.70, 0.30, 10000, {}), "ak-dispersion")


def test_ak_skip_insufficient_edge():
    assert_ak_skip_zero(ak.evaluate_and_size_opportunity(
        _book(ask=0.53, bid=0.50), 0.55, 0.0, 10000, {}), "ak-insufficient")


def test_ak_skip_low_conviction_gate3():
    assert_ak_skip_zero(ak.evaluate_and_size_opportunity(
        _book(ask=0.45, bid=0.43), 0.55, 0.0, 10000, {}), "ak-gate3")


def test_ak_skip_cluster_cap():
    # Pre-fix this branch returned kelly_fraction_raw == 0.40 on a SKIP.
    assert_ak_skip_zero(ak.evaluate_and_size_opportunity(
        _book(), 0.70, 0.0, 10000, {"C1": 0.15}), "ak-cluster-cap")


def test_ak_skip_depth_exhausted():
    # Pre-fix this branch returned kelly_fraction_raw == 0.40 on a SKIP.
    assert_ak_skip_zero(ak.evaluate_and_size_opportunity(
        _book(ask_size=0), 0.70, 0.0, 10000, {}), "ak-depth")


def test_ak_trade_unchanged():
    r = ak.evaluate_and_size_opportunity(_book(), 0.70, 0.0, 10000, {})
    assert r.action == ak.TradeAction.BUY_YES
    assert abs(r.kelly_fraction_raw - 0.40) < 1e-9, r
    assert r.allocated_fraction == 0.05 and r.invested_capital == 500.0, r


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
