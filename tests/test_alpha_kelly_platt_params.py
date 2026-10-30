"""
test_alpha_kelly_platt_params.py — SF-BL-9508 verification tests.

Pre-registered done/kill criteria (TASK.md):
  "Falsifiable: aggregate/platt_recalibrate raise ValueError for
   non-finite platt_alpha/platt_beta (and alpha <= 0) and the repro
   pools the uniform panel to ~0.70; kill if every caller's Platt
   parameters are proven to come only from a fitted,
   finiteness-checked calibrator with a tested invariant."

Kill condition is NOT met (see PRODUCER_AUDIT.md in the SF-BL-9508
work dir): no caller passes either parameter at all (the benchmark
uses the literal defaults), the repo's fitted PlattScaler is wired
nowhere near this surface, and its intercept-only fit returns
A_ = 0.0 by design with no tested output invariant. So the fix
applies at both consumption gates, per PREREG.md in the work dir:
platt_alpha / alpha must be a finite real strictly > 0 and
platt_beta / beta any finite real, validated at
aggregate_fleet_forecasts entry (before member processing, every
branch) and at platt_recalibrate entry, else a ValueError naming
SF-BL-9508 and the parameter at that surface is raised. Validating
via the module's _is_finite_number also rejects bool / non-numeric
parameters with the named ValueError — the deliberate SF-BL-10966
overlap recorded in PREREG.md, which triggers that task's stated
kill criterion.

Pre-fix reproduction (verified live 2026-10-23, before the patch;
see repro.py / repro_prefix.txt in the work dir — uniform 0.70
panel, 3 distinct families, independent search, weight 1.0):
  aggregate default        -> (0.7583861891129468, 0.0)
  platt_alpha=nan / beta=nan / beta=+inf -> (0.98, 0.0)  (0.98 cap)
  platt_alpha=-inf / beta=-inf           -> (0.02, 0.0)
  platt_alpha=0.0          -> (0.5, 0.0)   (panel discarded)
  platt_alpha=-1.08        -> (0.24161381088705333, 0.0) (rank-inverted)
  platt_alpha='1.08' / beta=None -> bare TypeError
  platt_alpha=True         -> (0.7425227848396216, 0.0) (silent alpha=1.0)
  platt_recalibrate(0.60) default -> 0.6116185719584099
  platt alpha=nan -> 0.98; alpha=0.0 -> 0.5; alpha=True -> 0.6

Run:  python3 test_alpha_kelly_platt_params.py  (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_extremizing_a.py.
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
INF = float("inf")


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _f(agent, fam, p, weight=1.0, independent=True):
    return ak.SubAgentForecast(agent, fam, p, weight, independent)


def _uniform(p=0.70):
    return [_f("a1", "fam1", p), _f("a2", "fam2", p), _f("a3", "fam3", p)]


def _raises_9508_agg(forecasts, param, **kwargs):
    """True iff aggregate raises ValueError naming SF-BL-9508 + param."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ak.aggregate_fleet_forecasts(forecasts, **kwargs)
    except ValueError as exc:
        return "SF-BL-9508" in str(exc) and param in str(exc)
    except Exception:  # noqa: BLE001 - wrong exception type is a failure
        return False
    return False


def _raises_9508_platt(param, **kwargs):
    """True iff platt_recalibrate raises ValueError naming SF-BL-9508 + param."""
    try:
        ak.platt_recalibrate(0.60, **kwargs)
    except ValueError as exc:
        return "SF-BL-9508" in str(exc) and param in str(exc)
    except Exception:  # noqa: BLE001
        return False
    return False


def _close(a, b, tol=1e-12):
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


# ---- 1. Aggregate: out-of-domain Platt parameters raise ---------------------

def test_agg_nonfinite_alpha_raises():
    for a in (NAN, INF, -INF):
        assert _raises_9508_agg(_uniform(), "platt_alpha", platt_alpha=a), a


def test_agg_zero_and_negative_alpha_raises():
    for a in (0.0, 0, -0.5, -1.08, -1e-9):
        assert _raises_9508_agg(_uniform(), "platt_alpha", platt_alpha=a), a


def test_agg_nonfinite_beta_raises():
    for b in (NAN, INF, -INF):
        assert _raises_9508_agg(_uniform(), "platt_beta", platt_beta=b), b


def test_agg_bool_and_non_numeric_raise_named_valueerror():
    # SF-BL-10966 overlap (deliberate, see module docstring): pre-fix
    # '1.08'/None died with a bare TypeError and True pooled as alpha=1.0.
    assert _raises_9508_agg(_uniform(), "platt_alpha", platt_alpha="1.08")
    assert _raises_9508_agg(_uniform(), "platt_alpha", platt_alpha=True)
    assert _raises_9508_agg(_uniform(), "platt_beta", platt_beta=None)
    assert _raises_9508_agg(_uniform(), "platt_beta", platt_beta=False)


# ---- 2. Aggregate validation is at entry: every branch raises ---------------

def test_median_branch_bad_platt_raises():
    shared = [_f("a1", "fam1", 0.70, 1.0, False),
              _f("a2", "fam2", 0.60, 1.0, False),
              _f("a3", "fam3", 0.65, 1.0, False)]
    assert _raises_9508_agg(shared, "platt_alpha", platt_alpha=NAN)
    assert _raises_9508_agg(shared, "platt_alpha", platt_alpha=-1.08)
    assert _raises_9508_agg(shared, "platt_beta", platt_beta=INF)


def test_trimmed_branch_bad_platt_raises():
    members = [_f(f"a{i}", "fam1", p)
               for i, p in enumerate([0.5, 0.6, 0.7, 0.6, 0.5])]
    assert _raises_9508_agg(members, "platt_alpha", platt_alpha=NAN)


def test_bad_platt_raises_before_member_processing():
    # Entry validation precedes the SF-BL-9439/9506 member drops: a bad
    # Platt pair raises even on a panel that would otherwise just drop
    # a member and pool.
    members = [_f("bad", "fam1", NAN), _f("a2", "fam2", 0.60),
               _f("a3", "fam3", 0.65), _f("a4", "fam4", 0.55)]
    assert _raises_9508_agg(members, "platt_alpha", platt_alpha=-1.08)
    assert _raises_9508_agg(members, "platt_beta", platt_beta=NAN)


# ---- 3. platt_recalibrate direct: same domains, its own param names ---------

def test_platt_nonfinite_alpha_raises():
    for a in (NAN, INF, -INF):
        assert _raises_9508_platt("alpha", alpha=a), a


def test_platt_zero_and_negative_alpha_raises():
    for a in (0.0, 0, -0.5, -1.12):
        assert _raises_9508_platt("alpha", alpha=a), a


def test_platt_nonfinite_beta_raises():
    for b in (NAN, INF, -INF):
        assert _raises_9508_platt("beta", beta=b), b


def test_platt_bool_and_non_numeric_raise_named_valueerror():
    assert _raises_9508_platt("alpha", alpha="1.12")
    assert _raises_9508_platt("alpha", alpha=True)
    assert _raises_9508_platt("beta", beta=None)
    assert _raises_9508_platt("beta", beta=False)


# ---- 4. In-domain pairs recalibrate / pool bit-identically to pre-fix -------

def test_platt_valid_pairs_unchanged():
    cases = [
        ({}, 0.6116185719584099),
        ({"alpha": 0.5}, 0.5505102572168218),
        ({"alpha": 1.0}, 0.6),
        ({"alpha": 1.12}, 0.6116185719584099),
        ({"alpha": 2.0}, 0.6923076923076922),
        ({"alpha": 2}, 0.6923076923076922),  # int alpha accepted
        ({"beta": -0.3}, 0.5384541427883593),
        ({"beta": 0.0}, 0.6116185719584099),
        ({"beta": 0.25}, 0.6691007993209436),
    ]
    for kwargs, exp in cases:
        result = ak.platt_recalibrate(0.60, **kwargs)
        assert abs(result - exp) <= 1e-12, (kwargs, result, exp)
    assert abs(ak.platt_recalibrate(0.30) - 0.2790912854168121) <= 1e-12
    assert abs(ak.platt_recalibrate(0.90) - 0.9213542876231214) <= 1e-12


def test_agg_valid_pairs_unchanged():
    cases = [
        ({"platt_alpha": 0.5}, (0.6293807543295477, 0.0)),
        ({"platt_alpha": 1.0}, (0.7425227848396216, 0.0)),
        ({"platt_alpha": 1.08}, (0.7583861891129468, 0.0)),
        ({"platt_alpha": 2.0}, (0.8926638564673068, 0.0)),
        ({"platt_beta": -0.3}, (0.6992758944377286, 0.0)),
        ({"platt_beta": 0.25}, (0.801206498796536, 0.0)),
        ({"platt_alpha": 1.0, "platt_beta": 0.0}, (0.7425227848396216, 0.0)),
    ]
    for kwargs, exp in cases:
        result = ak.aggregate_fleet_forecasts(_uniform(), **kwargs)
        assert _close(result, exp), (kwargs, result, exp)


def test_beta_zero_is_valid_not_confused_with_alpha_zero():
    # Domain asymmetry is the point: beta=0 is the ordinary fitted
    # intercept (and both defaults); only alpha <= 0 is invalid.
    result = ak.aggregate_fleet_forecasts(_uniform(), platt_beta=0)
    assert _close(result, (0.7583861891129468, 0.0)), result
    assert abs(ak.platt_recalibrate(0.60, beta=0) - 0.6116185719584099) <= 1e-12


# ---- 5. Defaults unchanged (panels + benchmark) ------------------------------

def test_default_panels_unchanged():
    cases = [
        ([0.70, 0.60, 0.65], (0.6989517116705906, 0.04082482904638629)),
        ([0.90, 0.30, 0.30], (0.5563059058407515, 0.282842712474619)),
        ([0.55, 0.45, 0.50], (0.5, 0.04082482904638632)),
        ([0.20, 0.80, 0.50, 0.65], (0.5520422672030745, 0.2218529918662356)),
        ([0.72, 0.74, 0.71, 0.73], (0.78745414972618, 0.011180339887498959)),
        ([0.30, 0.30, 0.30], (0.24161381088705325, 0.0)),
        ([0.05, 0.95, 0.50], (0.49999999999999994, 0.36742346141747667)),
        ([0.70, 0.70, 0.70], (0.7583861891129468, 0.0)),
    ]
    for probs, exp in cases:
        members = [_f(f"a{i}", f"fam{i}", p) for i, p in enumerate(probs)]
        result = ak.aggregate_fleet_forecasts(members)
        assert _close(result, exp), (probs, result, exp)


def test_median_branch_default_unchanged():
    shared = [_f("a1", "fam1", 0.70, 1.0, False),
              _f("a2", "fam2", 0.60, 1.0, False),
              _f("a3", "fam3", 0.65, 1.0, False)]
    result = ak.aggregate_fleet_forecasts(shared)
    assert _close(result, (0.6611811695087535, 0.04082482904638629)), result


def test_benchmark_fixture_unchanged():
    # Benchmark panels all use the default Platt pair; the patch is
    # pure validation for in-domain pairs, so both fleet scores are
    # pinned at their pre-fix values on the current tree (0.0547 /
    # -0.1499; proven byte-identical with the patch reverse-applied
    # in DONE-CLAIM.md).
    buf = io.StringIO()
    with redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    assert out.count('"north_star_score":') == 2, out[-2000:]
    assert '"north_star_score": 0.0547' in out, out[-2000:]
    assert '"north_star_score": -0.1499' in out, out[-2000:]


def test_empty_list_still_raises_empty_error():
    try:
        ak.aggregate_fleet_forecasts([])
    except ValueError as exc:
        assert "cannot be empty" in str(exc), exc
    else:
        raise AssertionError("empty list did not raise")


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
