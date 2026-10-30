"""
test_zero_weight_panel.py — SF-BL-8901 verification tests.

Pre-registered done/kill criteria (TASK.md):
  "Fix passes iff a zero-total-weight independent panel pools to the
   equal-weight log-odds result (or raises a named no-weight error) in
   a regression test; kill if a census shows historical_brier_weight is
   floored above 0 at every construction site."

Kill condition is NOT met (see CENSUS.md in the SF-BL-8901 work dir):
SubAgentForecast is a plain dataclass with no __post_init__ — it accepts
0.0 at construction (proven live) — its only in-repo producer passes bare
literals with no clamp or tested invariant, the dataclass is mutable,
and SF-BL-9506's own tests pin 0.0 as a reachable finite weight deferred
to this prereg. So the fix applies at aggregate_fleet_forecasts: in the
weighted log-odds branch (distinct families >= 3, independent search),
a total weight of exactly 0.0 falls back to equal-weight log-odds
pooling, logged via UserWarning "SF-BL-8901: ..." naming the pooled
agent_ids, instead of dividing by zero.

Pre-fix reproduction (verified live 2026-10-22, before the patch;
see repro.py in the work dir):
  probs 0.70/0.60/0.65, 3 distinct families, independent, all weights 0.0
      -> ZeroDivisionError: float division by zero
  equal-weight reference (same probs, weight 1.0 each)
      -> (0.6989517116705906, 0.04082482904638629)

Run:  python3 test_zero_weight_panel.py   (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_nonfinite_weight.py.
"""
import io
import os
import sys
import warnings
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(
    os.path.expanduser("~/workspace/superforecaster"), "superforecaster"))

import alpha_kelly as ak  # noqa: E402

PASS = []
NAN = float("nan")


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _f(agent, fam, p, weight=1.0, independent=True):
    return ak.SubAgentForecast(agent, fam, p, weight, independent)


def _agg(forecasts):
    """Aggregate with warnings captured; returns (result, [messages])."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = ak.aggregate_fleet_forecasts(forecasts)
    return result, [str(w.message) for w in caught]


def _close(a, b, tol=1e-12):
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


# ---- 0. Kill-condition probe: the constructor floors nothing ---------------

def test_constructor_does_not_floor_weight():
    assert _f("x", "fam", 0.5, 0.0).historical_brier_weight == 0.0
    m = _f("x", "fam", 0.5, 1.2)
    m.historical_brier_weight = 0.0  # mutation reaches the aggregator too
    assert m.historical_brier_weight == 0.0


# ---- 1. Prereg fixture: all-zero independent panel --------------------------

def test_prereg_zero_panel_pools_equal_weight():
    members = [_f("a1", "fam1", 0.70, 0.0), _f("a2", "fam2", 0.60, 0.0),
               _f("a3", "fam3", 0.65, 0.0)]
    result, msgs = _agg(members)  # pre-fix: ZeroDivisionError here
    assert _close(result, (0.6989517116705906, 0.04082482904638629)), result
    equal, equal_msgs = _agg([_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60),
                              _f("a3", "fam3", 0.65)])
    assert _close(result, equal), (result, equal)
    assert equal_msgs == [], equal_msgs
    assert any("SF-BL-8901" in m and "equal-weight" in m
               and "a1" in m and "a2" in m and "a3" in m for m in msgs), msgs


def test_prereg_variant_probs_pools_equal_weight():
    members = [_f("a1", "fam1", 0.90, 0.0), _f("a2", "fam2", 0.30, 0.0),
               _f("a3", "fam3", 0.30, 0.0)]
    result, msgs = _agg(members)
    assert _close(result, (0.5563059058407515, 0.282842712474619)), result
    assert any("SF-BL-8901" in m for m in msgs), msgs


def test_zero_panel_equals_unit_panel_for_many_probs():
    for probs in ([0.70, 0.60, 0.65], [0.90, 0.30, 0.30], [0.55, 0.45, 0.50],
                  [0.20, 0.80, 0.50, 0.65], [0.99, 0.01, 0.50]):
        zero = [_f(f"a{i}", f"fam{i}", p, 0.0) for i, p in enumerate(probs)]
        unit = [_f(f"a{i}", f"fam{i}", p, 1.0) for i, p in enumerate(probs)]
        zr, zmsgs = _agg(zero)
        ur, umsgs = _agg(unit)
        assert _close(zr, ur), (probs, zr, ur)
        assert any("SF-BL-8901" in m for m in zmsgs), (probs, zmsgs)
        assert umsgs == [], (probs, umsgs)


def test_four_member_zero_panel():
    members = [_f("a1", "fam1", 0.78, 0.0), _f("a2", "fam2", 0.81, 0.0),
               _f("a3", "fam3", 0.76, 0.0), _f("a4", "fam4", 0.79, 0.0)]
    result, msgs = _agg(members)
    assert _close(result, (0.8522971183903788, 0.018027756377319962)), result
    assert any("SF-BL-8901" in m and "4 independent" in m for m in msgs), msgs


# ---- 2. Zero total by cancellation; non-zero totals untouched ---------------

def test_cancelling_weights_fall_back_to_equal_weight():
    members = [_f("a1", "fam1", 0.70, 1.0), _f("a2", "fam2", 0.60, -1.0),
               _f("a3", "fam3", 0.65, 0.5), _f("a4", "fam4", 0.55, -0.5)]
    result, msgs = _agg(members)
    assert _close(result, (0.6680679464068928, 0.05590169943749472)), result
    assert any("SF-BL-8901" in m for m in msgs), msgs


def test_mixed_zero_and_positive_weights_no_fallback():
    # Total is 1.0, not 0: weighted branch as before — the two zero-weight
    # members contribute nothing, no SF-BL-8901 warning.
    members = [_f("a1", "fam1", 0.70, 0.0), _f("a2", "fam2", 0.60, 0.0),
               _f("a3", "fam3", 0.65, 1.0)]
    result, msgs = _agg(members)
    assert _close(result, (0.6975594326036093, 0.04082482904638629)), result
    assert msgs == [], msgs


def test_negative_weights_nonzero_total_unchanged():
    # SF-BL-9147 pin: finite negative weights with a non-zero total divide
    # exactly as pre-fix (anti-pooling value preserved), no warning.
    members = [_f("a1", "fam1", 0.70, 1.0), _f("a2", "fam2", 0.60, -0.5),
               _f("a3", "fam3", 0.65, 1.0)]
    result, msgs = _agg(members)
    assert _close(result, (0.7571732864368983, 0.04082482904638629)), result
    assert msgs == [], msgs


# ---- 3. Branches that never read weights are untouched ----------------------

def test_median_branch_zero_weights_no_warning():
    members = [_f("a1", "fam1", 0.70, 0.0, False),
               _f("a2", "fam2", 0.60, 0.0, False),
               _f("a3", "fam3", 0.65, 0.0, False)]
    result, msgs = _agg(members)
    assert _close(result, (0.6611811695087533, 0.04082482904638629)), result
    assert msgs == [], msgs


def test_shared_family_zero_weights_no_warning():
    # <3 distinct families -> median branch even with independent search.
    members = [_f("a1", "fam1", 0.70, 0.0), _f("a2", "fam1", 0.60, 0.0),
               _f("a3", "fam2", 0.65, 0.0)]
    result, msgs = _agg(members)
    assert _close(result, (0.6611811695087533, 0.04082482904638629)), result
    assert msgs == [], msgs


def test_trimmed_branch_zero_weights_no_warning():
    base = [0.5, 0.6, 0.7, 0.6, 0.5]
    members = [_f(f"a{i}", "fam1", p, 0.0) for i, p in enumerate(base)]
    result, msgs = _agg(members)
    assert _close(result, (0.5719288328804474, 0.0748331477354788)), result
    assert msgs == [], msgs


# ---- 4. Mutation and composition with SF-BL-9506 -----------------------------

def test_mutated_to_zero_panel_falls_back():
    members = [_f("a1", "fam1", 0.70, 1.2), _f("a2", "fam2", 0.60, 1.1),
               _f("a3", "fam3", 0.65, 1.0)]
    for m in members:
        m.historical_brier_weight = 0.0
    result, msgs = _agg(members)
    assert _close(result, (0.6989517116705906, 0.04082482904638629)), result
    assert any("SF-BL-8901" in m for m in msgs), msgs


def test_nonfinite_drop_then_zero_total_falls_back():
    # SF-BL-9506 drops the NaN-weight member first; the kept 3 are all
    # zero-weight, so SF-BL-8901 falls back over the kept members only.
    members = [_f("bad", "fam1", 0.70, NAN), _f("a2", "fam2", 0.60, 0.0),
               _f("a3", "fam3", 0.65, 0.0), _f("a4", "fam4", 0.55, 0.0)]
    result, msgs = _agg(members)
    assert _close(result, (0.6344438174355361, 0.04082482904638629)), result
    assert any("SF-BL-9506" in m and "bad" in m and "n=3 of 4" in m
               for m in msgs), msgs
    assert any("SF-BL-8901" in m and "a2" in m for m in msgs), msgs


# ---- 5. No regression on clean pools -----------------------------------------

def test_clean_pools_unchanged():
    cases = [
        ([_f("a1", "fam1", 0.90), _f("a2", "fam2", 0.30), _f("a3", "fam3", 0.30)],
         (0.5563059058407515, 0.282842712474619)),
        ([_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60), _f("a3", "fam3", 0.65)],
         (0.6989517116705906, 0.04082482904638629)),
        ([_f("a1", "fam1", 0.70), _f("a2", "fam1", 0.60)],
         (0.6611811695087533, 0.05)),
        ([_f("a1", "fam1", 0.60)], (0.6077590814688054, 0.0)),
    ]
    for members, expected in cases:
        result, msgs = _agg(members)
        assert _close(result, expected), (result, expected)
        assert msgs == [], msgs


def test_benchmark_fixture_unchanged():
    # The benchmark panels use weights 1.2/1.1/1.0 (total > 0), so this
    # patch cannot alter them; the naive score -0.1499 is pinned here and
    # the disciplined score is proven byte-identical with the patch
    # reverse-applied in DONE-CLAIM.md (in the current working tree it
    # reads 0.0547, moved from the committed 0.1059 by other fleet tasks'
    # uncommitted patches — identical 0.0547 with and without SF-BL-8901).
    buf = io.StringIO()
    with redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    assert out.count('"north_star_score":') == 2, out[-2000:]
    assert '"north_star_score": -0.1499' in out, out[-2000:]


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
