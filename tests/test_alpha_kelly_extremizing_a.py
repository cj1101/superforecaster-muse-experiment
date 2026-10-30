"""
test_alpha_kelly_extremizing_a.py — SF-BL-9476 verification tests.

Pre-registered done/kill criteria (TASK.md):
  "Pass if extremizing_a outside a pre-registered positive finite band
   raises and the default a=1.25 pooled outputs on the benchmark panels
   are unchanged; kill if a caller census proves extremizing_a is only
   ever the module default at every call site."

Kill condition is NOT met (see CENSUS.md in the SF-BL-9476 work dir):
the only in-repo production caller (the benchmark) and every test
caller omit the argument today, but extremizing_a is a public
parameter with no default-only enforcement, so an in-repo census
cannot bind out-of-tree fleet callers or config-supplied kwargs.
So the fix applies at aggregate_fleet_forecasts entry: extremizing_a
must be a finite real number strictly > 0 (the pre-registered band in
PREREG.md in the work dir; no upper bound is preregistered, so none is
imposed — cf. SF-BL-9364), else a ValueError naming SF-BL-9476 is
raised, on every branch, before any pooling.

Pre-fix reproduction (verified live 2026-10-23, before the patch;
see repro.py / repro_prefix.txt in the work dir — uniform 0.70 panel,
3 distinct families, independent search, weight 1.0):
  default            -> (0.7583861891129468, 0.0)
  a=nan              -> (0.98, 0.0)          (0.98 cap)
  a=+inf             -> (0.98, 0.0)
  a=-inf             -> (0.02, 0.0)
  a=0.0              -> (0.5, 0.0)           (pool collapsed to centre)
  a=-2.0             -> (0.13821881130243546, 0.0)  (rank-inverted)
  a='1.25' / None    -> bare TypeError
  a=True             -> (0.7140389069736216, 0.0)   (silent a=1.0 pool)
  median branch, a=nan / -2.0 -> silently ignored, pooled as default

Note on types: validating via the module's _is_finite_number also
rejects bool / non-numeric factors with the named ValueError. That
overlaps SF-BL-10965's fixture set deliberately (PREREG.md) and
triggers that task's stated kill criterion.

Run:  python3 test_alpha_kelly_extremizing_a.py  (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_zero_weight_panel.py.
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


def _raises_9476(forecasts, a):
    """True iff aggregate raises ValueError naming SF-BL-9476 for a."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ak.aggregate_fleet_forecasts(forecasts, extremizing_a=a)
    except ValueError as exc:
        return "SF-BL-9476" in str(exc) and "extremizing_a" in str(exc)
    except Exception:  # noqa: BLE001 - wrong exception type is a failure
        return False
    return False


def _close(a, b, tol=1e-12):
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


# ---- 1. Prereg fixtures: out-of-band factors raise --------------------------

def test_nan_raises():
    assert _raises_9476(_uniform(), NAN)


def test_positive_inf_raises():
    assert _raises_9476(_uniform(), INF)


def test_negative_inf_raises():
    assert _raises_9476(_uniform(), -INF)


def test_zero_raises():
    assert _raises_9476(_uniform(), 0.0)
    assert _raises_9476(_uniform(), 0)


def test_negative_factors_raise():
    for a in (-2.0, -0.5, -1.0, -1e-9):
        assert _raises_9476(_uniform(), a), a


def test_bool_and_non_numeric_raise_named_valueerror():
    # SF-BL-10965 overlap (deliberate, see module docstring): pre-fix
    # '1.25'/None died with a bare TypeError and True pooled as a=1.0.
    for a in ("1.25", None, True, False):
        assert _raises_9476(_uniform(), a), a


# ---- 2. Validation is at entry: branches that never read a also raise -------

def test_median_branch_bad_a_raises():
    shared = [_f("a1", "fam1", 0.70, 1.0, False),
              _f("a2", "fam2", 0.60, 1.0, False),
              _f("a3", "fam3", 0.65, 1.0, False)]
    for a in (NAN, -2.0, INF, 0.0):
        assert _raises_9476(shared, a), a


def test_shared_family_branch_bad_a_raises():
    members = [_f("a1", "fam1", 0.70), _f("a2", "fam1", 0.60),
               _f("a3", "fam2", 0.65)]
    assert _raises_9476(members, NAN)
    assert _raises_9476(members, -2.0)


def test_trimmed_branch_bad_a_raises():
    members = [_f(f"a{i}", "fam1", p)
               for i, p in enumerate([0.5, 0.6, 0.7, 0.6, 0.5])]
    assert _raises_9476(members, NAN)


def test_bad_a_raises_before_member_processing():
    # Entry validation precedes the SF-BL-9439/9506 member drops: a bad
    # factor raises even on a panel that would otherwise just drop a
    # member and pool.
    members = [_f("bad", "fam1", NAN), _f("a2", "fam2", 0.60),
               _f("a3", "fam3", 0.65), _f("a4", "fam4", 0.55)]
    assert _raises_9476(members, -2.0)


# ---- 3. In-band factors pool bit-identically to pre-fix ---------------------

def test_in_band_factors_unchanged():
    expected = {
        0.5: (0.6124306342424013, 0.0),
        1.0: (0.7140389069736216, 0.0),
        1.25: (0.7583861891129468, 0.0),
        2.0: (0.8617811886975646, 0.0),
        # No upper bound is preregistered, so none is imposed: a large
        # finite factor pools exactly as pre-fix (here, to the cap).
        5.0: (0.98, 0.0),
    }
    for a, exp in expected.items():
        result = ak.aggregate_fleet_forecasts(_uniform(), extremizing_a=a)
        assert _close(result, exp), (a, result, exp)


def test_int_factor_accepted():
    # int is a finite real for _is_finite_number (module convention).
    result = ak.aggregate_fleet_forecasts(_uniform(), extremizing_a=2)
    assert _close(result, (0.8617811886975646, 0.0)), result


def test_small_positive_factor_shrinks_toward_centre():
    # 0 < a < 1 is legitimate de-extremization: pooled between the
    # a=0 collapse point (0.5 pre-Platt) and the identity pool.
    shrunk = ak.aggregate_fleet_forecasts(_uniform(), extremizing_a=0.5)
    identity = ak.aggregate_fleet_forecasts(_uniform(), extremizing_a=1.0)
    assert 0.5 < shrunk[0] < identity[0], (shrunk, identity)


# ---- 4. Default-a outputs unchanged (panels + benchmark) --------------------

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
        explicit = ak.aggregate_fleet_forecasts(members, extremizing_a=1.25)
        assert _close(explicit, exp), (probs, explicit, exp)


def test_median_branch_default_unchanged():
    shared = [_f("a1", "fam1", 0.70, 1.0, False),
              _f("a2", "fam2", 0.60, 1.0, False),
              _f("a3", "fam3", 0.65, 1.0, False)]
    result = ak.aggregate_fleet_forecasts(shared)
    assert _close(result, (0.6611811695087535, 0.04082482904638629)), result


def test_benchmark_fixture_unchanged():
    # Benchmark panels all use the default a; the patch is pure
    # validation for in-band factors, so both fleet scores are pinned
    # at their pre-fix values on the current tree (0.0547 / -0.1499;
    # proven byte-identical with the patch reverse-applied in
    # DONE-CLAIM.md).
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
