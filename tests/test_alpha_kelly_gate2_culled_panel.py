"""
test_gate2_culled_panel.py — SF-BL-10761 verification tests.

Pre-registered done/kill criterion (PREREG.md in the task work dir):
thread a dropped-member count through the aggregate return and Gate 2
SKIPs whenever drops>0 or kept n<3; kill if no forward panel in the
test window has a drop that changes a Gate 2 verdict (the verdict-change
half is exercised here on the repro panels and restated over the full
registered window by forward_panel_restatement.py).

Pre-fix reproduction (repro.py, verified live 2026-10-23):
    panel 0.60 / NaN / inf-weight -> (0.6078, 0.0) 2-tuple, warnings only;
        evaluate -> BUY_YES $7,265.79 (single survivor, dispersion 0.0
        by construction, passes Gate 2 at maximum apparent agreement)
    panel 0.60 / 0.62 / NaN -> (0.6185, 0.01); evaluate -> BUY_YES
        $8,023.77 (survivorship dispersion over the kept pair)
    panel 0.10 / 0.90 / NaN (3 families in) == clean 2-family panel
        exactly: the pooling-branch flip is invisible in the return.

Fix: aggregate_fleet_forecasts returns FleetAggregation(calibrated,
dispersion, provenance: PanelProvenance); evaluate_and_size_opportunity
gains Gate 2C (panel_n_dropped / panel_n_kept): culled or thin panels
SKIP as CULLED_PANEL_GATE2, malformed provenance SKIPs as
INVALID_PANEL_PROVENANCE, both zero-size with the cluster dict
untouched; no provenance (both None) and certified-clean provenance
pass byte-identically.

Run:  python3 test_gate2_culled_panel.py   (exit 0 iff all pass)
"""
import contextlib
import io
import os
import sys
import warnings

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
NAN, INF = float("nan"), float("inf")


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _f(agent, fam, p, w=1.0):
    return ak.SubAgentForecast(agent, fam, p, w, True)


def _book():
    return ak.OrderBookSnapshot(
        market_id="FIXTURE-10761", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C10761",
        est_days_to_resolution=10,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100000)],
        fee_rate=0.01, on_chain_verified=True)


def _agg(members):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = ak.aggregate_fleet_forecasts(members)
    return result, [str(w.message) for w in caught]


def _eval(p, disp, used=None, **kw):
    return ak.evaluate_and_size_opportunity(
        _book(), p, disp, 100000.0, used if used is not None else {}, **kw)


def _assert_zero_skip(r):
    assert r.action == ak.TradeAction.SKIP, r
    assert r.invested_capital == 0.0 and r.shares_acquired == 0.0, r
    assert r.exec_price == 0.0 and r.liquidation_bid_price == 0.0, r
    assert r.kelly_fraction_raw == 0.0 and r.allocated_fraction == 0.0, r
    assert r.kl_divergence_nats == 0.0 and r.expected_log_growth == 0.0, r


# ---- 1. Provenance is threaded through the aggregate return ----------------

def test_single_survivor_provenance():
    r, msgs = _agg([_f("a1", "fam1", 0.60), _f("a2", "fam2", NAN),
                    _f("a3", "fam3", 0.90, INF)])
    assert isinstance(r, ak.FleetAggregation), type(r)
    assert (r[0], r[1]) == (0.6077590814688054, 0.0), r
    p = r.provenance
    assert (p.n_input, p.n_kept, p.n_dropped) == (3, 1, 2), p
    assert (p.n_prob_dropped, p.n_weight_dropped, p.n_range_dropped) == (1, 1, 0), p
    assert (p.distinct_families_input, p.distinct_families_kept) == (3, 1), p
    assert len(msgs) == 2, msgs  # 9439 + 9506 warnings unchanged


def test_clean_panel_provenance_and_values_unchanged():
    r, msgs = _agg([_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60),
                    _f("a3", "fam3", 0.65)])
    assert (r[0], r[1]) == (0.6989517116705906, 0.04082482904638629), r
    p = r.provenance
    assert (p.n_input, p.n_kept, p.n_dropped) == (3, 3, 0), p
    assert p.distinct_families_input == p.distinct_families_kept == 3, p
    assert p.branch_flipped is False, p
    assert msgs == [], msgs


def test_range_drop_counted():
    r, _ = _agg([_f("a1", "fam1", 0.60), _f("a2", "fam2", 1.50),
                 _f("a3", "fam3", 0.62)])
    p = r.provenance
    assert (p.n_dropped, p.n_range_dropped, p.n_kept) == (1, 1, 2), p
    assert (r[0], r[1]) == (0.6184789651776837, 0.010000000000000009), r


def test_member_bad_on_both_fields_counted_once():
    r, _ = _agg([_f("a1", "fam1", NAN, NAN), _f("a2", "fam2", 0.60),
                 _f("a3", "fam3", 0.65), _f("a4", "fam4", 0.70)])
    p = r.provenance
    assert (p.n_input, p.n_kept, p.n_dropped) == (4, 3, 1), p
    assert (p.n_prob_dropped, p.n_weight_dropped) == (1, 0), p


def test_branch_flip_recorded():
    r, _ = _agg([_f("a1", "fam1", 0.10), _f("a2", "fam2", 0.90),
                 _f("a3", "fam3", NAN)])
    assert (r[0], r[1]) == (0.5, 0.4), r  # median branch, as pre-fix
    assert r.provenance.branch_flipped is True, r.provenance
    assert r.provenance.distinct_families_kept == 2, r.provenance


def test_unpack_and_index_compat():
    # The historical 2-tuple contract is preserved exactly (2-unpack,
    # indexing, tuple equality — the SF-BL-9439/9506 pinned shape), and
    # the provenance rides on the return object as an attribute.
    r, _ = _agg([_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60),
                 _f("a3", "fam3", 0.65)])
    cal, disp = r
    assert cal == r[0] == r.calibrated_probability
    assert disp == r[1] == r.dispersion
    assert tuple(r) == (0.6989517116705906, 0.04082482904638629)
    assert r == (0.6989517116705906, 0.04082482904638629)
    assert len(r) == 2
    assert isinstance(r.provenance, ak.PanelProvenance)
    assert r.provenance.n_kept == 3 and r.provenance.n_dropped == 0


def test_all_dropped_still_raises():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ak.aggregate_fleet_forecasts([_f("a1", "fam1", NAN)])
    except ValueError:
        return
    raise AssertionError("all-dropped panel did not raise ValueError")


# ---- 2. Gate 2C: culled / thin panels SKIP ---------------------------------

def test_single_survivor_now_skips_at_gate2c():
    r, _ = _agg([_f("a1", "fam1", 0.60), _f("a2", "fam2", NAN),
                 _f("a3", "fam3", 0.90, INF)])
    used = {}
    out = _eval(r[0], r[1], used=used,
                panel_n_dropped=r.provenance.n_dropped,
                panel_n_kept=r.provenance.n_kept)
    _assert_zero_skip(out)
    assert out.skip_reason.startswith("CULLED_PANEL_GATE2"), out.skip_reason
    assert "n_dropped=2" in out.skip_reason and "n_kept=1" in out.skip_reason
    assert used == {}, used


def test_tight_pair_after_drop_now_skips():
    r, _ = _agg([_f("a1", "fam1", 0.60), _f("a2", "fam2", 0.62),
                 _f("a3", "fam3", NAN)])
    out = _eval(r[0], r[1], panel_n_dropped=1, panel_n_kept=2)
    _assert_zero_skip(out)
    assert out.skip_reason.startswith("CULLED_PANEL_GATE2"), out.skip_reason


def test_thin_panel_without_drops_skips():
    # A genuinely small clean panel (n=2, no drops) is thin: kept n<3.
    out = _eval(0.80, 0.01, panel_n_dropped=0, panel_n_kept=2)
    _assert_zero_skip(out)
    assert out.skip_reason.startswith("CULLED_PANEL_GATE2"), out.skip_reason


def test_drop_with_high_dispersion_reports_gate2c_not_gate2():
    # Provenance gate precedes Gate 2: the reason names the cull, and the
    # panel would have SKIPped at Gate 2 anyway — precedence is pinned.
    out = _eval(0.80, 0.50, panel_n_dropped=1, panel_n_kept=3)
    _assert_zero_skip(out)
    assert out.skip_reason.startswith("CULLED_PANEL_GATE2"), out.skip_reason


def test_gate2c_precedes_taper_mode():
    cfg = ak.FleetRiskConfig(dispersion_taper_enabled=True)
    out = ak.evaluate_and_size_opportunity(
        _book(), 0.80, 0.01, 100000.0, {}, config=cfg,
        panel_n_dropped=1, panel_n_kept=2)
    _assert_zero_skip(out)
    assert out.skip_reason.startswith("CULLED_PANEL_GATE2"), out.skip_reason


# ---- 3. Gate 2C: malformed provenance SKIPs invalid -------------------------

def test_malformed_provenance_skips_invalid():
    cases = [
        dict(panel_n_dropped=1),                      # kept missing
        dict(panel_n_kept=3),                         # dropped missing
        dict(panel_n_dropped=-1, panel_n_kept=3),     # negative
        dict(panel_n_dropped=0, panel_n_kept=0),      # sums to zero
        dict(panel_n_dropped=True, panel_n_kept=3),   # bool
        dict(panel_n_dropped=1.0, panel_n_kept=3),    # float
        dict(panel_n_dropped="1", panel_n_kept=3),    # string
    ]
    for kw in cases:
        used = {}
        out = _eval(0.80, 0.10, used=used, **kw)
        _assert_zero_skip(out)
        assert out.skip_reason.startswith("INVALID_PANEL_PROVENANCE"), (kw, out.skip_reason)
        assert used == {}, (kw, used)


# ---- 4. Legacy / clean behaviour byte-identical -----------------------------

def test_no_provenance_trades_as_before():
    # The two pre-fix repro trades, called the legacy way (no counts).
    out = _eval(0.6077590814688054, 0.0)
    assert out.action == ak.TradeAction.BUY_YES, out
    assert round(out.invested_capital, 2) == 7265.79, out.invested_capital
    out = _eval(0.6184789651776837, 0.010000000000000009)
    assert out.action == ak.TradeAction.BUY_YES, out
    assert round(out.invested_capital, 2) == 8023.77, out.invested_capital


def test_certified_clean_provenance_trades_as_before():
    out = _eval(0.80, 0.10, panel_n_dropped=0, panel_n_kept=4)
    assert out.action == ak.TradeAction.BUY_YES, out
    assert out.invested_capital == 15000.0, out.invested_capital


def test_clean_provenance_high_dispersion_still_gate2():
    out = _eval(0.80, 0.50, panel_n_dropped=0, panel_n_kept=4)
    _assert_zero_skip(out)
    assert out.skip_reason == "UNRESOLVED_AGENT_DISPERSION (0.50 > 0.22)", out.skip_reason


def test_helper_directly():
    for good in (0, 1, 3, 100):
        assert ak._is_valid_panel_count(good) is True, good
    for bad in (-1, True, False, 1.0, "3", None, float("nan")):
        assert ak._is_valid_panel_count(bad) is False, bad


# ---- 5. No regression on the built-in benchmark -----------------------------

def test_benchmark_nstar_unchanged():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    # Pinned to the PRISTINE snapshot's values this run (-0.1611 /
    # -0.1499; see PREREG.md amendment — older claims' +0.1059 is stale
    # benchmark drift from other fleet work, identical pre/post patch).
    assert '"north_star_score": -0.1611' in out, out[:400]
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
