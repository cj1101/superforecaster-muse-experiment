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


# ---- SF-BL-9323: --q-exec side convention -----------------------------------
# Before the fix, --q-exec was silently assumed YES-side: a NO-side q_exec
# (e.g. a NO-book VWAP from nstar.vwap_for_dollars) was complemented a
# second time before Gate D / Kelly. The tag --q-exec-side no makes the
# NO price be used as-is; the default yes path is unchanged.

def sf_edge_raw(*args):
    return subprocess.run([sys.executable, CLI, "edge", *args],
                          capture_output=True, text=True)


def test_cli_edge_no_side_gate_d_flip():
    # Fixture A: p=0.42, NO VWAP 0.40. Untagged (buggy) read: SKIP,
    # gate_d.pass False. Tagged: p_side 0.58, LR 1.45 -> TRADE BUY_NO.
    r = sf_edge("--p", "0.42", "--q-exec", "0.40", "--q-exec-side", "no")
    assert r["decision"] == "TRADE" and r["side"] == "BUY_NO", r
    assert r["q_exec_side"] == "no" and r["q_side"] == 0.40, r
    assert r["gate_d"] == {"p_side": 0.58, "likelihood_ratio": 1.45, "pass": True}, r
    assert r["net_edge"] == 0.18 and r["kelly_fraction_of_account"] == 0.105, r


def test_cli_edge_no_side_kelly_uses_uncomplemented_price():
    # Fixture B: p=0.32, NO VWAP 0.45 -> kelly (0.68-0.45)/0.55*0.35 = 0.1464
    # (double-complemented read gave 0.1011).
    r = sf_edge("--p", "0.32", "--q-exec", "0.45", "--q-exec-side", "no")
    assert r["decision"] == "TRADE" and r["side"] == "BUY_NO", r
    assert r["kelly_fraction_of_account"] == 0.1464, r


def test_cli_edge_no_side_overpriced_no_book_skips():
    # Fixture C: p=0.45, NO VWAP 0.60 (overpriced NO). The buggy read
    # manufactured TRADE kelly 0.0875; tagged it must SKIP and size zero.
    r = sf_edge("--p", "0.45", "--q-exec", "0.60", "--q-exec-side", "no")
    assert r["skip_reason"] == "INSUFFICIENT_NET_EDGE", r
    assert_skip_zero(r, "no-side-overpriced")


def test_cli_edge_no_side_never_flips_to_yes():
    # p=0.90 with a cheap-looking NO price: tagged NO edge is 0.10-0.30 < 0,
    # so SKIP — 1 - q_no must not be invented as the YES price.
    r = sf_edge("--p", "0.90", "--q-exec", "0.30", "--q-exec-side", "no")
    assert_skip_zero(r, "no-side-no-flip")


def test_cli_edge_no_side_crypto_gate_a():
    r = sf_edge("--p", "0.42", "--q-exec", "0.40", "--q-exec-side", "no",
                "--domain", "crypto")
    assert r["skip_reason"] == "FAILED_CRYPTOBENCH_ONCHAIN_GATE (Gate A)", r
    assert_skip_zero(r, "no-side-gate-a")
    r2 = sf_edge("--p", "0.42", "--q-exec", "0.40", "--q-exec-side", "no",
                 "--domain", "crypto", "--onchain-verified")
    assert r2["decision"] == "TRADE" and r2["side"] == "BUY_NO", r2
    assert r2["kelly_fraction_of_account"] == 0.0525, r2  # 0.105 x 0.5 dampener


def test_cli_edge_invalid_side_tag_rejected():
    r = sf_edge_raw("--p", "0.42", "--q-exec", "0.40", "--q-exec-side", "maybe")
    assert r.returncode == 2, r


def test_cli_edge_yes_tag_default_and_explicit_identical():
    a = sf_edge("--p", "0.20", "--q-exec", "0.50")
    b = sf_edge("--p", "0.20", "--q-exec", "0.50", "--q-exec-side", "yes")
    assert a == b, (a, b)
    assert a["q_exec_side"] == "yes" and a["q_side"] == 0.50, a


# ---- SF-BL-9580: --domain canonicalisation ---------------------------------
# Before the fix, cmd_edge compared --domain verbatim: any variant of the
# crypto label ("Crypto", "CRYPTO", " crypto", "Cryptocurrency", ...)
# silently behaved as a non-crypto domain — Gate A never fired and the
# x0.5 dampener never applied (repro: --p 0.80 --q-exec 0.50 --domain
# Crypto -> TRADE kelly 0.15; exact "crypto" -> SKIP Gate A). Unknown
# labels ("unknown", "BTC", "") also TRADED as generic non-crypto.
# Now the label is canonicalised first; unknowns are rejected exit 2
# with one JSON error object. Canonical inputs are unchanged.

CRYPTO_VARIANTS = ["Crypto", "CRYPTO", "cRyPtO", " crypto", "crypto ",
                   "Cryptocurrency", "cryptocurrencies", "CryptoBench"]


def test_cli_edge_domain_variant_gate_a():
    for v in CRYPTO_VARIANTS:
        r = sf_edge("--p", "0.80", "--q-exec", "0.50", "--domain", v)
        assert r["skip_reason"] == "FAILED_CRYPTOBENCH_ONCHAIN_GATE (Gate A)", (v, r)
        assert r["gate_a_onchain_verified"] is False, (v, r)
        assert_skip_zero(r, f"variant-gate-a-{v!r}")


def test_cli_edge_domain_variant_dampener_verified():
    for v in CRYPTO_VARIANTS:
        r = sf_edge("--p", "0.58", "--q-exec", "0.42", "--domain", v, "--onchain-verified")
        assert r["decision"] == "TRADE", (v, r)
        assert r["kelly_fraction_of_account"] == 0.0483, (v, r)  # dampened, as exact crypto
        assert r["gate_a_onchain_verified"] is True, (v, r)


def test_cli_edge_domain_variant_matches_exact_crypto():
    ref = sf_edge("--p", "0.58", "--q-exec", "0.42", "--domain", "crypto")
    for v in CRYPTO_VARIANTS:
        assert sf_edge("--p", "0.58", "--q-exec", "0.42", "--domain", v) == ref, v
    ref_v = sf_edge("--p", "0.58", "--q-exec", "0.42", "--domain", "crypto", "--onchain-verified")
    for v in CRYPTO_VARIANTS:
        assert sf_edge("--p", "0.58", "--q-exec", "0.42", "--domain", v,
                       "--onchain-verified") == ref_v, v


def test_cli_edge_domain_variant_no_side():
    r = sf_edge("--p", "0.42", "--q-exec", "0.40", "--q-exec-side", "no", "--domain", "Crypto")
    assert r["skip_reason"] == "FAILED_CRYPTOBENCH_ONCHAIN_GATE (Gate A)", r
    assert_skip_zero(r, "variant-no-side-gate-a")
    r2 = sf_edge("--p", "0.42", "--q-exec", "0.40", "--q-exec-side", "no",
                 "--domain", "Crypto", "--onchain-verified")
    assert r2["decision"] == "TRADE" and r2["kelly_fraction_of_account"] == 0.0525, r2


def test_cli_edge_domain_markets_predmarkets_variants_unchanged():
    ref = sf_edge("--p", "0.58", "--q-exec", "0.42", "--domain", "predmarkets")
    assert ref["kelly_fraction_of_account"] == 0.0966, ref
    for v in ["Predmarkets", "PREDMARKETS", " predmarkets ", "prediction-markets",
              "Markets", "MARKETS", " markets "]:
        assert sf_edge("--p", "0.58", "--q-exec", "0.42", "--domain", v) == ref, v


def test_cli_edge_domain_unknown_rejected():
    for v in ["unknown", "BTC", "", "prediction-markets-xyz", "crypto2"]:
        r = sf_edge_raw("--p", "0.58", "--q-exec", "0.42", "--domain", v)
        assert r.returncode == 2, (v, r)
        body = json.loads(r.stdout)  # exactly one strict-JSON object on stdout
        assert "error" in body and "decision" not in body, (v, body)


def test_cli_forecast_domain_variant_crypto():
    def sf_forecast(*args):
        out = subprocess.run([sys.executable, CLI, "forecast", *args],
                             capture_output=True, text=True, check=True)
        return json.loads(out.stdout)
    ref = sf_forecast("--domain", "crypto", "--asset", "BTC", "--horizon", "7")
    for v in ["Crypto", "CRYPTO", " crypto "]:
        assert sf_forecast("--domain", v, "--asset", "BTC", "--horizon", "7") == ref, v
    bad = subprocess.run([sys.executable, CLI, "forecast", "--domain", "nosuchthing",
                          "--asset", "BTC", "--horizon", "7"],
                         capture_output=True, text=True)
    assert bad.returncode == 2 and "error" in json.loads(bad.stdout), bad


# ---- SF-BL-9717: non-finite --p / --q-exec rejected at the boundary -------
# Before the fix, cmd_edge clipped with np.clip and no finiteness check:
# NaN passed through the clip (filed symptom: exit 0, bare NaN tokens for
# p / net_edge / Gate D — invalid strict JSON; after SF-BL-9507 landed
# downstream: uncaught ValueError traceback, exit 1, no JSON), while
# +/-inf was silently clipped to the boundary and TRADED (e.g. --p inf
# -> TRADE at p 0.98). Now any non-finite --p or --q-exec exits 2 with
# exactly one strict-JSON error object naming the flag, and finite
# fixtures (including finite out-of-range clip controls) are unchanged.

def _reject_const(tok):
    raise ValueError(f"non-strict token {tok}")


def assert_edge_nonfinite_rejected(args, flag):
    r = sf_edge_raw(*args)
    assert r.returncode == 2, (args, r.returncode, r.stdout, r.stderr)
    body = json.loads(r.stdout, parse_constant=_reject_const)  # strict JSON
    assert "error" in body and flag in body["error"], (args, body)
    assert "decision" not in body and "kelly_fraction_of_account" not in body, (args, body)
    assert "Traceback" not in r.stderr, (args, r.stderr)


def test_cli_edge_p_nan_rejected():
    assert_edge_nonfinite_rejected(("--p", "nan", "--q-exec", "0.55"), "--p")


def test_cli_edge_q_nan_rejected():
    assert_edge_nonfinite_rejected(("--p", "0.62", "--q-exec", "nan"), "--q-exec")


def test_cli_edge_inf_rejected():
    assert_edge_nonfinite_rejected(("--p", "inf", "--q-exec", "0.55"), "--p")
    assert_edge_nonfinite_rejected(("--p", "0.62", "--q-exec", "inf"), "--q-exec")
    assert_edge_nonfinite_rejected(("--p=-inf", "--q-exec", "0.55"), "--p")
    assert_edge_nonfinite_rejected(("--p", "0.62", "--q-exec=-inf"), "--q-exec")
    assert_edge_nonfinite_rejected(("--p", "-inf", "--q-exec", "0.55"), "--p")
    assert_edge_nonfinite_rejected(("--p", "0.62", "--q-exec", "-inf"), "--q-exec")


def test_cli_edge_nonfinite_spellings_rejected():
    for tok in ["NaN", "NAN", "Infinity", "-Infinity", "1e999", "-1e999"]:
        assert_edge_nonfinite_rejected(("--p", tok, "--q-exec", "0.55"), "--p")
        assert_edge_nonfinite_rejected(("--p", "0.62", "--q-exec", tok), "--q-exec")


def test_cli_edge_nonfinite_no_side_and_crypto_rejected():
    assert_edge_nonfinite_rejected(
        ("--p", "0.42", "--q-exec", "nan", "--q-exec-side", "no"), "--q-exec")
    assert_edge_nonfinite_rejected(
        ("--p", "nan", "--q-exec", "0.40", "--q-exec-side", "no"), "--p")
    assert_edge_nonfinite_rejected(
        ("--p", "nan", "--q-exec", "0.50", "--domain", "crypto"), "--p")
    assert_edge_nonfinite_rejected(
        ("--p", "0.62", "--q-exec", "inf", "--domain", "Crypto",
         "--onchain-verified"), "--q-exec")
    assert_edge_nonfinite_rejected(("--p", "nan", "--q-exec", "nan"), "--p")


def test_cli_edge_finite_clip_controls_unchanged():
    # Finite out-of-range values keep the pre-existing clip behaviour.
    r = sf_edge("--p", "1.5", "--q-exec", "0.55")
    assert r["decision"] == "TRADE" and r["p"] == 0.98, r
    r = sf_edge("--p=-0.5", "--q-exec", "0.55")
    assert r["decision"] == "TRADE" and r["p"] == 0.02, r
    r = sf_edge("--p", "0.62", "--q-exec", "0.55")
    assert r["decision"] == "TRADE" and r["kelly_fraction_of_account"] == 0.0544, r


# ---- SF-BL-10659: --lam validated at the boundary ---------------------------
# Before the fix, cmd_edge sized inline off the raw --lam and never called
# nstar.kelly_fraction (the function SF-BL-9778 guarded): lam=-1.0 exited 0
# as a TRADE with kelly_fraction_of_account -0.2083 on the fixture
# (p 0.62, q-exec 0.52), lam=2.0 silently sized at the 0.15 cluster cap,
# lam=inf/1e999 TRADED at the cap with a bare Infinity lambda token,
# lam=nan died as an exit-1 traceback on a TRADE path and emitted a bare
# NaN token on a SKIP path, and finite out-of-band values sized silently.
# Now any non-finite or out-of-[0.25, 0.50] --lam exits 2 with exactly one
# strict-JSON error object naming --lam, and in-band fixtures (boundaries
# 0.25/0.50 inclusive, default 0.35) are unchanged.

def assert_edge_lam_rejected(args):
    r = sf_edge_raw(*args)
    assert r.returncode == 2, (args, r.returncode, r.stdout, r.stderr)
    body = json.loads(r.stdout, parse_constant=_reject_const)  # strict JSON
    assert "error" in body and "--lam" in body["error"], (args, body)
    assert "decision" not in body and "kelly_fraction_of_account" not in body, (args, body)
    assert "Traceback" not in r.stderr, (args, r.stderr)


def test_cli_edge_lam_negative_rejected():
    # TASK repro 1: pre-fix TRADE kelly -0.2083, both argument forms.
    assert_edge_lam_rejected(("--p", "0.62", "--q-exec", "0.52", "--lam=-1.0"))
    assert_edge_lam_rejected(("--p", "0.62", "--q-exec", "0.52", "--lam", "-1.0"))


def test_cli_edge_lam_above_band_rejected():
    # TASK repro 2 is 2.0 (pre-fix: silently at the 0.15 cap).
    for lam in ["2.0", "1.0", "0.51", "0.5000001"]:
        assert_edge_lam_rejected(("--p", "0.62", "--q-exec", "0.52", "--lam", lam))


def test_cli_edge_lam_below_band_rejected():
    for lam in ["0", "0.24", "0.2499999", "-0.5"]:
        assert_edge_lam_rejected(("--p", "0.62", "--q-exec", "0.52", f"--lam={lam}"))


def test_cli_edge_lam_nonfinite_rejected():
    for lam in ["nan", "NaN", "inf", "Infinity", "1e999"]:
        assert_edge_lam_rejected(("--p", "0.62", "--q-exec", "0.52", "--lam", lam))
    assert_edge_lam_rejected(("--p", "0.62", "--q-exec", "0.52", "--lam=-inf"))
    assert_edge_lam_rejected(("--p", "0.62", "--q-exec", "0.52", "--lam", "-inf"))


def test_cli_edge_lam_rejected_on_skip_no_side_and_crypto():
    # Validation is unconditional: it precedes the SKIP/Gate decisions,
    # the NO-side branch and the crypto dampener/Gate A.
    assert_edge_lam_rejected(("--p", "0.52", "--q-exec", "0.50", "--lam", "nan"))
    assert_edge_lam_rejected(("--p", "0.52", "--q-exec", "0.50", "--lam=-1.0"))
    assert_edge_lam_rejected(
        ("--p", "0.42", "--q-exec", "0.40", "--q-exec-side", "no", "--lam", "2.0"))
    assert_edge_lam_rejected(
        ("--p", "0.62", "--q-exec", "0.52", "--domain", "crypto", "--lam=-1.0"))
    assert_edge_lam_rejected(
        ("--p", "0.62", "--q-exec", "0.52", "--domain", "Crypto",
         "--onchain-verified", "--lam", "nan"))


def test_cli_edge_lam_inband_unchanged():
    # Frozen CLI fixture (TASK): full Kelly (0.62-0.52)/(1-0.52) = 0.20833.
    r = sf_edge("--p", "0.62", "--q-exec", "0.52", "--lam", "0.25")
    assert r["decision"] == "TRADE" and r["kelly_fraction_of_account"] == 0.0521, r
    r = sf_edge("--p", "0.62", "--q-exec", "0.52", "--lam", "0.50")
    assert r["decision"] == "TRADE" and r["kelly_fraction_of_account"] == 0.1042, r
    a = sf_edge("--p", "0.62", "--q-exec", "0.52")
    b = sf_edge("--p", "0.62", "--q-exec", "0.52", "--lam", "0.35")
    assert a == b and a["kelly_fraction_of_account"] == 0.0729, (a, b)
    # An in-band lam on a SKIP path still SKIPs and sizes zero (EC-09).
    r = sf_edge("--p", "0.52", "--q-exec", "0.50", "--lam", "0.25")
    assert_skip_zero(r, "lam-inband-skip")


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
