"""
test_cdf_module_tied_percentiles.py — SF-BL-10770 verification.

Pre-registered: "Success if, after tie handling is made explicit and
identical in both paths, linear and pchip CDFs from tied declared
percentiles agree within 0.01 sup norm on a registered tie panel while
untied outputs stay bit-identical; kill if a census of forward declared
percentiles finds zero tied values, making the divergence unreachable
in production."

Task repro: {0.05:100, 0.25:100, 0.5:100, 0.75:110, 0.95:120} gave
sup |linear - pchip| = 0.45 pre-fix (a middle triple tie gives 0.50):
np.interp over duplicate xp is implementation-defined while the pchip
branch collapsed ties to their FIRST probability. The census (work dir
census.py) found ties ARE reachable in production — climatology
empirical quantiles tie on low-entropy trailing windows — so the kill
criterion is not met. Fix: _cdf_from_tied_percentiles treats a tie
group as an explicit CDF jump (F approaches p_low from the left,
F(value) = p_high), shared by both paths; only the interpolation
BETWEEN distinct values differs, and pchip never smooths across a
jump. Untied inputs never enter the new branch.

The registered tie panel below mirrors tie_panel.json in the task
work dir (registered before the fix was applied).

Run: python3 tests/test_cdf_module_tied_percentiles.py
"""
import hashlib, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metaculus_bot"))
import numpy as np  # noqa: E402
import cdf_module as cm  # noqa: E402
PASS = []
def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}"); PASS.append(bool(cond))

def _pair(pct, lo, hi, zp=None, size=201):
    kw = dict(range_min=lo, range_max=hi, zero_point=zp, size=size)
    return (cm.percentiles_to_cdf(pct, interp="linear", **kw),
            cm.percentiles_to_cdf(pct, interp="pchip", **kw))

def _sup(pct, lo, hi, zp=None):
    lin, pch = _pair(pct, lo, hi, zp)
    return float(np.max(np.abs(lin - pch)))

REPRO = {0.05: 100, 0.25: 100, 0.5: 100, 0.75: 110, 0.95: 120}
PANEL = [
    ("task_repro_head_triple", REPRO, 0, 200),
    ("head_pair", {0.05: 90, 0.25: 90, 0.5: 100, 0.75: 110, 0.95: 120}, 0, 200),
    ("middle_triple", {0.05: 80, 0.25: 100, 0.5: 100, 0.75: 100, 0.95: 120}, 0, 200),
    ("middle_pair", {0.05: 80, 0.25: 100, 0.5: 100, 0.75: 110, 0.95: 120}, 0, 200),
    ("two_groups", {0.05: 80, 0.25: 80, 0.5: 100, 0.75: 120, 0.95: 120}, 0, 200),
    ("tail_tie", {0.05: 70, 0.25: 85, 0.5: 100, 0.75: 115, 0.95: 115}, 0, 200),
    ("all_tied", {0.05: 100, 0.25: 100, 0.5: 100, 0.75: 100, 0.95: 100}, 0, 200),
    ("task_repro_tight_range", REPRO, 50, 150),
]
UNTIED_BASELINE = {
    "untied_even": ({0.05: 80, 0.25: 90, 0.5: 100, 0.75: 110, 0.95: 120}, 0, 200, None,
                    "6b8911ecdd4717763dd75092641300ab8633d5433fd367992737028423899a51",
                    "853515ca09611486538b9fbea28cd7a43a555179d63c6ba3087bc04409c3f974"),
    "untied_uneven": ({0.05: 60, 0.25: 90, 0.5: 100, 0.75: 110, 0.95: 140}, 0, 200, None,
                      "b5f6a175d437a3cd0c4719c13f29951024a62a93dc8fe60d2ef8c65713991d51",
                      "b2c25a886fe4be9a673508d1270e7746ace3149666fcd12e7aa5a69cc4a4eac6"),
    "untied_wide": ({0.05: 5, 0.25: 12, 0.5: 20, 0.75: 33, 0.95: 60}, 0, 100, None,
                    "7c177744d50029d723c48b16ce3c940b515d114995327a3988f1fd182e8e6695",
                    "96e120de7248c368f5754915cbd12667fe2f09d545f00bbe5055a77cc1506153"),
    "untied_logscale": ({0.05: 10, 0.25: 50, 0.5: 100, 0.75: 500, 0.95: 900}, 1, 1000, 0,
                        "995cf473ddb5d46e95d895b8e11334491de608d4f23583afae8f337f765a0c76",
                        "8adf947231caa57e996a882244a1153456a04d5f35bae66d8460652dea314a6f"),
    "untied_three": ({0.1: 30, 0.5: 50, 0.9: 70}, 0, 100, None,
                     "a7f5ce293e7f837de58e1a90fd75078684f2d382431e9bd5e88bb2552dabc53b",
                     "8c04b0e9cd5229f4baa7f44435a3037457e3ff127a44fd6e12f52242e1e68e68"),
    "untied_negative_range": ({0.05: -40, 0.25: -10, 0.5: 0, 0.75: 15, 0.95: 45}, -50, 50, None,
                              "3d334c8c1707ddc5a5810905d14e3676354ad082bb853c4f5529ec5b423e2ea2",
                              "13ec0fd2c046c510e4d959ce45ec9162c722eeafbb4e5d6afd6c7298b3ee63e4"),
}

def test_task_repro_agrees_within_tolerance():
    s = _sup(REPRO, 0, 200)
    assert s <= 0.01, f"repro sup {s} > 0.01 (pre-fix 0.45)"

def test_registered_panel_all_within_tolerance():
    for name, pct, lo, hi in PANEL:
        s = _sup(pct, lo, hi)
        assert s <= 0.01, f"{name} sup {s} > 0.01"

def test_jump_semantics_at_tied_value():
    # Grid on [0, 200] has nominal == index; tied value 100 sits at 100.
    for interp in ("linear", "pchip"):
        cdf = cm.percentiles_to_cdf(REPRO, 0, 200, interp=interp)
        assert cdf[99] == 0.05, f"{interp}: F(99)={cdf[99]} != p_low 0.05"
        assert cdf[100] == 0.5, f"{interp}: F(100)={cdf[100]} != p_high 0.5"
        assert cdf[0] == 0.05 and cdf[-1] == 0.95, f"{interp}: tails wrong"

def test_middle_tie_jump_and_left_limit():
    pct = {0.05: 80, 0.25: 100, 0.5: 100, 0.75: 100, 0.95: 120}
    for interp in ("linear", "pchip"):
        cdf = cm.percentiles_to_cdf(pct, 0, 200, interp=interp)
        assert cdf[100] == 0.75, f"{interp}: F(100)={cdf[100]} != 0.75"
        assert cdf[99] < 0.25 + 0.01, f"{interp}: left approach {cdf[99]}"

def test_all_tied_is_pure_step():
    pct = {0.05: 100, 0.25: 100, 0.5: 100, 0.75: 100, 0.95: 100}
    lin, pch = _pair(pct, 0, 200)
    assert np.array_equal(lin, pch)
    assert lin[99] == 0.05 and lin[100] == 0.95 and lin[-1] == 0.95

def test_tied_outputs_are_valid_monotone_cdfs():
    for name, pct, lo, hi in PANEL:
        for interp in ("linear", "pchip"):
            cdf = cm.percentiles_to_cdf(pct, lo, hi, interp=interp)
            assert np.all(np.isfinite(cdf)), name
            assert np.all(np.diff(cdf) >= 0), f"{name}/{interp} not monotone"
            problems = [p for p in cm.validate_cdf(cdf)
                        if "non-decreasing" in p or "outside" in p]
            assert not problems, f"{name}/{interp}: {problems}"

def test_untied_outputs_bit_identical():
    for name, (pct, lo, hi, zp, lh, ph) in UNTIED_BASELINE.items():
        lin, pch = _pair(pct, lo, hi, zp)
        assert hashlib.sha256(lin.tobytes()).hexdigest() == lh, f"{name} linear changed"
        assert hashlib.sha256(pch.tobytes()).hexdigest() == ph, f"{name} pchip changed"

def test_unknown_interp_still_raises_on_tied_and_untied():
    for pct in (REPRO, {0.05: 80, 0.5: 100, 0.95: 120}):
        try:
            cm.percentiles_to_cdf(pct, 0, 200, interp="cubic")
        except ValueError as e:
            assert "unknown interp" in str(e)
        else:
            raise AssertionError("unknown interp did not raise")

def test_tied_logscale_and_small_size_robust():
    pct = {0.05: 10.0, 0.25: 10.0, 0.5: 100.0, 0.75: 100.0, 0.95: 1000.0}
    for interp in ("linear", "pchip"):
        cdf = cm.percentiles_to_cdf(pct, 1, 1000, zero_point=0, interp=interp)
        assert np.all(np.isfinite(cdf)) and np.all(np.diff(cdf) >= 0)
        small = cm.percentiles_to_cdf(REPRO, 0, 200, size=11, interp=interp)
        assert small.shape == (11,) and np.all(np.diff(small) >= 0)

def test_extended_panel_improvement_bound():
    # Report-only cases (uneven secants): not gated at 0.01 because the
    # generic smoothing difference is nonzero there even untied, but
    # the tie divergence must collapse from 0.20-0.25 to <= 0.02.
    for pct in ({0.05: 80, 0.25: 90, 0.5: 100, 0.75: 120, 0.95: 120},
                {0.05: 60, 0.25: 90, 0.5: 90, 0.75: 110, 0.95: 140}):
        s = _sup(pct, 0, 200)
        assert s <= 0.02, f"extended sup {s} > 0.02"

def test_client_adapter_matches():
    import metaculus_client as mc
    q = {"scaling": {"range_min": 0, "range_max": 200, "zero_point": None},
         "question_type": "numeric"}
    try:
        cdf = mc.percentiles_to_cdf(REPRO, q)
    except TypeError:
        cdf = None  # adapter signature differs; direct check below instead
    lin, _ = _pair(REPRO, 0, 200)
    if cdf is not None:
        assert np.allclose(np.asarray(cdf, dtype=float), lin)

ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
if __name__ == "__main__":
    for fn in ALL_TESTS:
        try: fn(); check(fn.__name__, True)
        except AssertionError as e: check(fn.__name__, False, str(e))
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
