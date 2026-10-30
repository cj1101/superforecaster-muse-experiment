"""
test_alpha_kelly_platt_p_raw_range.py — SF-BL-10904 verification tests.

Pre-registered done/kill criteria (TASK.md):
  "platt_recalibrate must raise a named ValueError for a finite
   p_raw outside [0, 1], fixtured at 82.0, 1.50 and -0.20, with
   finite in-range recalibrations (including the aggregate pooled
   panels) bit-identical; kill if a caller census proves every
   p_raw reaching platt_recalibrate is the output of a
   range-validated producer under a tested invariant."

Kill condition is NOT met (see CALLER_CENSUS.md in the SF-BL-10904
work dir): aggregate's internal call passes a range-validated
pooled_prob by construction, but platt_recalibrate is public —
in-repo tests call it with bare literals and out-of-tree fleet
scripts pass bare floats, with no tested invariant in between.
Per PREREG.md in the work dir: p_raw must be a finite real in the
closed interval [0, 1], validated at platt_recalibrate entry inside
_validate_platt_p_raw — the SF-BL-9579 finiteness check first
(non-finite / bool / non-numeric keep raising naming SF-BL-9579),
then the range check raising naming SF-BL-10904 and p_raw, BEFORE
the SF-BL-9508 parameter and SF-BL-9713 bounds checks. Endpoints
0.0 / 1.0 (and ints 0 / 1) stay valid and recalibrate
bit-identically. logit's own SF-BL-9883 rejection at logit entry
is untouched and still governs direct logit callers.

Pre-fix state on the current tree (verified live 2026-10-23;
repro.py / repro_prefix.txt in the work dir): the TASK fixtures
already raised via SF-BL-9883's landed logit check, but naming
SF-BL-9883 and `p`, after the parameter/bounds checks:
  platt_recalibrate(82.0) / (1.50) / (-0.20)
      -> ValueError naming SF-BL-9883, `p`
  platt_recalibrate(82.0, alpha=nan)
      -> ValueError naming SF-BL-9508, `alpha`
  platt_recalibrate(-0.20, min_p=0.90, max_p=0.10)
      -> ValueError naming SF-BL-9713, clip bounds
At the TASK's filing (before SF-BL-9883 landed) the same fixtures
returned the clips: (82.0) -> 0.98, (1.50) -> 0.98, (-0.20) -> 0.02;
clean platt_recalibrate(0.60) -> 0.6116185719584099.

Run:  python3 test_alpha_kelly_platt_p_raw_range.py  (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_platt_p_raw.py.
"""
import io
import os
import sys
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


def _title_panel():
    return [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60),
            _f("a3", "fam3", 0.65)]


def _raises_10904(p_raw, **kwargs):
    """True iff platt raises ValueError naming SF-BL-10904 + p_raw + [0, 1]."""
    try:
        ak.platt_recalibrate(p_raw, **kwargs)
    except ValueError as exc:
        msg = str(exc)
        return "SF-BL-10904" in msg and "p_raw" in msg and "[0, 1]" in msg
    except Exception:  # noqa: BLE001 - wrong exception type is a failure
        return False
    return False


def _raises_9579(p_raw, **kwargs):
    try:
        ak.platt_recalibrate(p_raw, **kwargs)
    except ValueError as exc:
        return "SF-BL-9579" in str(exc) and "p_raw" in str(exc)
    except Exception:  # noqa: BLE001
        return False
    return False


# ---- 1. Finite out-of-range p_raw raises at platt entry (TASK core) --------

def test_task_fixtures_raise():
    # TASK fixtures: 82.0 (percent-unit), 1.50, -0.20.
    assert _raises_10904(82.0)
    assert _raises_10904(1.50)
    assert _raises_10904(-0.20)


def test_finite_out_of_range_grid_raises():
    for p in (2.0, -1.0, 1.0000001, -1e-9, 100.0, -100.0):
        assert _raises_10904(p), p


def test_out_of_range_raises_with_valid_nondefault_params():
    assert _raises_10904(82.0, alpha=0.5, beta=-0.3)
    assert _raises_10904(1.5, alpha=2.0, beta=0.25)
    assert _raises_10904(-0.2, min_p=0.1, max_p=0.9)


# ---- 2. Precedence: p_raw named before params / bounds ----------------------

def test_p_raw_named_before_bad_params():
    # SF-BL-9579 precedence pattern: an out-of-range p_raw is named
    # (SF-BL-10904) even when alpha / beta are also bad — pre-fix
    # these named SF-BL-9508 via the parameter check.
    assert _raises_10904(82.0, alpha=NAN)
    assert _raises_10904(-0.2, beta=INF)
    assert _raises_10904(1.5, alpha=0.0)


def test_p_raw_named_before_bad_bounds():
    # Pre-fix these named SF-BL-9713 via the bounds check.
    assert _raises_10904(-0.20, min_p=0.90, max_p=0.10)
    assert _raises_10904(82.0, min_p=NAN)


# ---- 3. Non-finite / type rejection unchanged (SF-BL-9579 precedence) ------

def test_nonfinite_still_raises_9579():
    for p in (NAN, INF, -INF):
        assert _raises_9579(p), p
    assert _raises_9579(NAN, alpha=NAN)


def test_bool_and_non_numeric_still_raise_9579():
    # True == 1.0 / False == 0.0 must NOT slip through the range
    # check as in-range endpoints; the finiteness screen rejects
    # them first, naming SF-BL-9579.
    for p in ("0.6", None, True, False, [0.6]):
        assert _raises_9579(p), p


# ---- 4. In-domain fixtures unchanged (bit-identical) ------------------------

def test_in_range_platt_unchanged():
    cases = [
        (0.60, 0.6116185719584099),
        (0.30, 0.2790912854168121),
        (0.90, 0.9213542876231214),
        (0.5, 0.5),
        (0.02, 0.02),
        (0.98, 0.98),
        # Endpoints stay in-domain and recalibrate as before.
        (0.0, 0.02),
        (1.0, 0.98),
        (0, 0.02),   # int endpoints accepted
        (1, 0.98),
    ]
    for p, exp in cases:
        result = ak.platt_recalibrate(p)
        assert abs(result - exp) <= 1e-12, (p, result, exp)


def test_in_range_with_valid_params_unchanged():
    assert abs(ak.platt_recalibrate(0.60, alpha=0.5)
               - 0.5505102572168218) <= 1e-12
    assert abs(ak.platt_recalibrate(0.60, alpha=1.0) - 0.6) <= 1e-12
    assert abs(ak.platt_recalibrate(0.60, alpha=2)
               - 0.6923076923076922) <= 1e-12
    assert abs(ak.platt_recalibrate(0.60, beta=-0.3)
               - 0.5384541427883593) <= 1e-12
    assert abs(ak.platt_recalibrate(0.60, beta=0.25)
               - 0.6691007993209436) <= 1e-12


def test_sibling_validations_unchanged():
    # In-range p_raw with bad params / bounds keeps the sibling names.
    try:
        ak.platt_recalibrate(0.60, alpha=NAN)
        raise AssertionError("alpha=NaN did not raise")
    except ValueError as exc:
        assert "SF-BL-9508" in str(exc), exc
    try:
        ak.platt_recalibrate(0.60, min_p=0.90, max_p=0.10)
        raise AssertionError("inverted bounds did not raise")
    except ValueError as exc:
        assert "SF-BL-9713" in str(exc), exc


def test_logit_direct_surface_unchanged():
    # SF-BL-9883's closed surface: direct logit callers keep the
    # SF-BL-9883 name; only the platt entry gained its own check.
    try:
        ak.logit(82.0)
        raise AssertionError("logit(82.0) did not raise")
    except ValueError as exc:
        assert "SF-BL-9883" in str(exc), exc
    assert ak.logit(0.5) == 0.0


# ---- 5. Aggregate pooled panels unchanged -----------------------------------

def test_aggregate_panels_unchanged():
    result = ak.aggregate_fleet_forecasts(_title_panel())
    assert abs(result[0] - 0.6989517116705906) <= 1e-12, result
    assert abs(result[1] - 0.04082482904638629) <= 1e-12, result
    uniform = [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.70),
               _f("a3", "fam3", 0.70)]
    result = ak.aggregate_fleet_forecasts(uniform)
    assert abs(result[0] - 0.7583861891129468) <= 1e-12, result
    assert result[1] == 0.0, result


def test_benchmark_fixture_unchanged():
    buf = io.StringIO()
    with redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    assert out.count('"north_star_score":') == 2, out[-2000:]
    assert '"north_star_score": 0.0547' in out, out[-2000:]
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
