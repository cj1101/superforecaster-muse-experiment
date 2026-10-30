"""
test_min_confidence_bool.py — SF-BL-10972 verification tests.

Prereg: "evaluate must SKIP zero-size with a named invalid-min-confidence
reason (cluster dict untouched) for min_confidence=True and False before
Gate 3, with genuine thresholds in [0,1] byte-identical; kill if a caller
census proves min_confidence is only ever set by validated numeric
literals and never mutated or unpickled."
Kill NOT met — see CALLER_CENSUS.md (no validator existed, the dataclass
is mutable and its own tests mutate this field on raw instances,
unpickling bypasses __post_init__, evaluate's default is shared).

Pre-fix repro (repro_prepatch.txt): fixture ask 0.50/bid 0.48, fee 0.01,
p=0.57, equity $100,000 — default 0.60 SKIP LOW_CONVICTION; False BUY_YES
$4,595.96; True SKIP LOW_CONVICTION as `< 1.00`; NaN/-1.0 also traded;
'0.6'/None bare TypeError; construction accepted everything.

Fix: _is_valid_min_confidence = finite real in [0,1] via
_is_finite_number (bool excluded); construction ValueError; new Gate 2.8
in evaluate (after Gate 2.7, before edge screen/Gate 3) SKIPs
INVALID_CONFIG / INVALID_MIN_CONFIDENCE at zero size. The single
predicate also closes SF-BL-10820's NaN/-1.0/2.0/'0.6' surface (that task
had not landed).

Run: python3 test_min_confidence_bool.py (exit 0 iff all pass)
"""
import os, pickle, sys
_CANDIDATES = [
    os.path.expanduser("~/workspace/superforecaster/superforecaster"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "superforecaster"),
]
for _c in _CANDIDATES:
    if os.path.isfile(os.path.join(_c, "alpha_kelly.py")):
        sys.path.insert(0, _c); break
import alpha_kelly as ak

PASS = []
def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond)); return cond

PREREG_BOOLS = [True, False]
INVALID_MC = PREREG_BOOLS + ["0.6", "0.60", None, float("nan"), float("inf"),
    float("-inf"), -1.0, -0.5, 1.5, 2.0, [], {}, b"0.6", 10**400]
VALID_MC = [0, 0.0, 0.25, 0.5, 0.57, 0.60, 0.99, 1, 1.0]

def _book(domain=ak.MarketDomain.PREDICTION_MARKET, cluster="C10972"):
    return ak.OrderBookSnapshot(market_id="FIXTURE-10972", title="mc fixture",
        domain=domain, factor_cluster=cluster, est_days_to_resolution=10,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=1000000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=1000000)],
        fee_rate=0.01, on_chain_verified=True)

def _mutated(val):
    cfg = ak.FleetRiskConfig(); cfg.min_confidence = val; return cfg

def _eval(cfg, p=0.57, book=None, used=None):
    return ak.evaluate_and_size_opportunity(
        book if book is not None else _book(), p, 0.0, 100000.0,
        used if used is not None else {}, config=cfg)

def _zero_skip(r):
    return (r.action == ak.TradeAction.SKIP and r.invested_capital == 0.0
        and r.shares_acquired == 0.0 and r.exec_price == 0.0
        and r.allocated_fraction == 0.0 and r.kelly_fraction_raw == 0.0
        and r.kelly_fraction_sized == 0.0 and r.liquidation_bid_price == 0.0)

def _named(r):
    return (r.skip_reason is not None and "INVALID_MIN_CONFIDENCE" in r.skip_reason
        and "SF-BL-10972" in r.skip_reason)

def test_prereg_bool_fixtures():
    ok = True
    for v in PREREG_BOOLS:
        used = {"PRE": 0.03}
        r = _eval(_mutated(v), used=used)
        ok &= _zero_skip(r) and _named(r) and used == {"PRE": 0.03} \
            and "LOW_CONVICTION" not in (r.skip_reason or "")
    check("prereg True/False SKIP INVALID_MIN_CONFIDENCE, zero size, pre-filled cluster untouched", ok)

def test_headline_false_no_longer_trades():
    r = _eval(_mutated(False))
    check("False closes the $4,595.96 headline trade", _zero_skip(r) and _named(r))
    r = _eval(_mutated(0.0))
    check("genuine 0.0 still trades $4,595.96 (bool False != float 0.0)",
          r.action == ak.TradeAction.BUY_YES and abs(r.invested_capital - 4595.959595959593) < 1e-9)

def test_true_no_longer_masquerades_as_1():
    r = _eval(_mutated(True))
    check("True SKIPs INVALID, not LOW_CONVICTION < 1.00", _named(r) and "LOW_CONVICTION" not in r.skip_reason)
    r = _eval(_mutated(1.0))
    check("genuine 1.0 still SKIPs LOW_CONVICTION < 1.00",
          r.action == ak.TradeAction.SKIP and r.skip_reason == "LOW_CONVICTION (p_side=0.57 < 1.00)")

def test_full_invalid_set():
    ok = True
    for v in INVALID_MC:
        used = {}
        try:
            r = _eval(_mutated(v), used=used)
            ok &= _zero_skip(r) and _named(r) and used == {}
        except Exception as e:
            print("  raised for", repr(v), type(e).__name__, e); ok = False
    check("full invalid set SKIPs named, no crash, cluster untouched", ok)

def test_10820_surface_closed():
    ok = all(_named(_eval(_mutated(v))) for v in (float("nan"), -1.0, 2.0, "0.6"))
    check("SF-BL-10820 fixtures NaN/-1.0/2.0/'0.6' also SKIP INVALID_MIN_CONFIDENCE", ok)

def test_buy_no_side():
    ok = True
    for v in PREREG_BOOLS:
        used = {}
        r = _eval(_mutated(v), p=0.20, used=used)  # BUY_NO candidate (p_side 0.80)
        ok &= _zero_skip(r) and _named(r) and used == {}
    check("BUY_NO side: bools SKIP INVALID_MIN_CONFIDENCE", ok)
    r = _eval(_mutated(0.60), p=0.20)
    check("BUY_NO side valid 0.60 trades at cluster cap $15,000",
          r.action == ak.TradeAction.BUY_NO and r.invested_capital == 15000.0)

def test_all_domains():
    ok = True
    for dom in (ak.MarketDomain.PREDICTION_MARKET, ak.MarketDomain.EQUITIES_MACRO, ak.MarketDomain.CRYPTO):
        r = _eval(_mutated(False), book=_book(domain=dom, cluster="CDOM"))
        ok &= _zero_skip(r) and _named(r)
    check("Gate 2.8 unconditional: bools SKIP on all domains incl. CRYPTO", ok)

def test_construction():
    ok = True
    for v in INVALID_MC:
        try:
            ak.FleetRiskConfig(min_confidence=v); print("  accepted", repr(v)); ok = False
        except ValueError as e:
            ok &= "min_confidence" in str(e) and "SF-BL-10972" in str(e)
    for v in VALID_MC:
        try:
            ok &= ak.FleetRiskConfig(min_confidence=v).min_confidence == v
        except Exception as e:
            print("  valid rejected", repr(v), e); ok = False
    ok &= ak.FleetRiskConfig().min_confidence == 0.60
    check("construction: invalid raises named ValueError, valid + default accepted", ok)

def test_helper_truth_table():
    ok = (ak._is_valid_min_confidence(True) is False and ak._is_valid_min_confidence(False) is False
          and ak._is_valid_min_confidence(1) is True and ak._is_valid_min_confidence(1.0) is True
          and ak._is_valid_min_confidence(0) is True and ak._is_valid_min_confidence(0.0) is True
          and ak._is_valid_min_confidence("0.6") is False and ak._is_valid_min_confidence(float("nan")) is False)
    check("helper: bool rejected despite equalling in-range 0.0/1.0; int endpoints accepted", ok)

def test_unpickled_and_raw_configs():
    cfg = ak.FleetRiskConfig(); cfg.min_confidence = True
    restored = pickle.loads(pickle.dumps(cfg))
    r = _eval(restored)
    ok = _zero_skip(r) and _named(r)
    raw = object.__new__(ak.FleetRiskConfig)
    # raw instance needs every field evaluate/Gate2.5-2.7 read; fill from a default via __dict__
    raw.__dict__.update(ak.FleetRiskConfig().__dict__); raw.min_confidence = False
    r = _eval(raw)
    ok &= _zero_skip(r) and _named(r)
    check("unpickled + raw object.__new__ bool configs SKIP INVALID", ok)

def test_gate_precedence():
    # Earlier gates keep precedence: invalid posterior (Gate 0.0) wins over Gate 2.8
    r = ak.evaluate_and_size_opportunity(_book(), float("nan"), 0.0, 100000.0, {}, config=_mutated(False))
    ok = r.skip_reason is not None and r.skip_reason.startswith("INVALID_POSTERIOR")
    # Gate 2.5 (kelly band) wins over Gate 2.8
    cfg = _mutated(False); cfg.kelly_multiplier = 2.0
    r = _eval(cfg)
    ok &= r.skip_reason is not None and "kelly_multiplier" in r.skip_reason
    # Gate 2.8 wins over the edge screen / Gate 3: a no-edge candidate with a bool still reports INVALID, not INSUFFICIENT/LOW_CONVICTION
    r = _eval(_mutated(True), p=0.52)
    ok &= _named(r)
    check("precedence: Gate 0.0 and Gate 2.5 before Gate 2.8; Gate 2.8 before edge screen/Gate 3", ok)

def test_valid_byte_identical():
    # Pre-fix snapshot values (snapshots_prepatch.txt, asserted field-by-field key numbers)
    cases = [
        (0, 0.57, ak.TradeAction.BUY_YES, 4595.959595959593, 9100.910091009095, None),
        (0.5, 0.57, ak.TradeAction.BUY_YES, 4595.959595959593, 9100.910091009095, None),
        (0.57, 0.57, ak.TradeAction.BUY_YES, 4595.959595959593, 9100.910091009095, None),
        (0.60, 0.57, ak.TradeAction.SKIP, 0.0, 0.0, "LOW_CONVICTION (p_side=0.57 < 0.60)"),
        (0.99, 0.57, ak.TradeAction.SKIP, 0.0, 0.0, "LOW_CONVICTION (p_side=0.57 < 0.99)"),
        (1, 0.57, ak.TradeAction.SKIP, 0.0, 0.0, "LOW_CONVICTION (p_side=0.57 < 1.00)"),
        (1.0, 0.57, ak.TradeAction.SKIP, 0.0, 0.0, "LOW_CONVICTION (p_side=0.57 < 1.00)"),
        (0.0, 0.80, ak.TradeAction.BUY_YES, 15000.0, 29702.970297029704, None),
        (0.60, 0.80, ak.TradeAction.BUY_YES, 15000.0, 29702.970297029704, None),
        (1.0, 0.80, ak.TradeAction.BUY_YES, 15000.0, 29702.970297029704, None),  # ratio arm >=1.35
    ]
    ok = True
    for mc, p, act, inv, sh, reason in cases:
        used = {}
        r = _eval(_mutated(mc), p=p, used=used)
        ok &= (r.action == act and abs(r.invested_capital - inv) < 1e-9
               and abs(r.shares_acquired - sh) < 1e-9 and r.skip_reason == reason)
    check("valid thresholds byte-identical vs pre-fix snapshots (incl. ratio-arm trade at mc=1.0, p=0.80)", ok)

def test_default_config_unchanged():
    used = {}
    r = ak.evaluate_and_size_opportunity(_book(), 0.57, 0.0, 100000.0, used)
    ok = r.action == ak.TradeAction.SKIP and r.skip_reason == "LOW_CONVICTION (p_side=0.57 < 0.60)" and used == {}
    used = {}
    r = ak.evaluate_and_size_opportunity(_book(), 0.80, 0.0, 100000.0, used)
    ok &= r.action == ak.TradeAction.BUY_YES and r.invested_capital == 15000.0
    check("default-config path (no config argument) unchanged", ok)

def test_valid_trade_consumes_cluster():
    used = {}
    r = _eval(_mutated(0.5), used=used)
    check("valid trade still writes exactly its sized fraction to cluster",
          r.action == ak.TradeAction.BUY_YES and used == {"C10972": 0.04595959595959593})

def test_benchmark():
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    check("benchmark runs, N_star -0.1611 (byte-identical output asserted externally)",
          "N_star" in out and "-0.1611" in out)

if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
    sys.exit(0 if all(PASS) and PASS else 1)
