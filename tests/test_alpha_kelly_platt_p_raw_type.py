"""
test_alpha_kelly_platt_p_raw_type.py — SF-BL-11014 verification tests.

Pre-registered done/kill criteria (TASK.md):
  "platt_recalibrate must raise a named ValueError for a p_raw that
   is not a finite real (bool excluded), fixtured at '0.6' and True,
   with numeric in-range recalibrations bit-identical; kill if a
   caller census proves p_raw only ever arrives type-validated from
   aggregate_fleet_forecasts under a tested invariant."

Kill condition is NOT met (see CALLER_CENSUS.md in the SF-BL-11014
work dir): aggregate's internal call passes a float pooled_prob by
construction, but the public function is called directly with bare
literals by in-repo tests and serialized out-of-tree producers, with
no type-validated producer under a tested invariant. Per PREREG.md
in the work dir: p_raw's TYPE (int / float, bool excluded) is
validated at platt_recalibrate entry as the FIRST step of
_validate_platt_p_raw, else a ValueError naming SF-BL-11014 and
p_raw. The type message dual-names SF-BL-9579 (registered in the
PREREG) so the landed SF-BL-9579 / SF-BL-10904 substring pins on
these fixtures stay true; numeric p_raw keeps its existing
SF-BL-9579 (non-finite) and SF-BL-10904 (out-of-range) rejections,
which do NOT name SF-BL-11014.

Pre-fix state on the current tree (verified live 2026-10-23;
repro.py / repro_prefix.txt in the work dir): the TASK fixtures
already raised, but naming SF-BL-9579 only ("p_raw '0.6' is not
finite") — the TYPE dimension had no named check of its own. At the
TASK's filing (2026-10-22, before SF-BL-9579 landed):
  platt_recalibrate('0.6') -> bare TypeError in the logit chain
  platt_recalibrate(True) -> 0.98 (the max clip, as if certainty)
  platt_recalibrate(0.60) default -> 0.6116185719584099

Run:  python3 test_alpha_kelly_platt_p_raw_type.py
      (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__
runner, same convention as tests/test_alpha_kelly_platt_p_raw.py.
"""
import io
import os
import sys
from contextlib import redirect_stdout
from decimal import Decimal

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


def _raises_11014(p_raw, **kwargs):
    """True iff platt raises ValueError naming SF-BL-11014 + p_raw."""
    try:
        ak.platt_recalibrate(p_raw, **kwargs)
    except ValueError as exc:
        return "SF-BL-11014" in str(exc) and "p_raw" in str(exc)
    except Exception:  # noqa: BLE001 - wrong exception type is a failure
        return False
    return False


def _raises_9579_only(p_raw, **kwargs):
    """ValueError naming SF-BL-9579 + p_raw and NOT SF-BL-11014."""
    try:
        ak.platt_recalibrate(p_raw, **kwargs)
    except ValueError as exc:
        return ("SF-BL-9579" in str(exc) and "p_raw" in str(exc)
                and "SF-BL-11014" not in str(exc))
    except Exception:  # noqa: BLE001
        return False
    return False


def _raises_10904(p_raw, **kwargs):
    try:
        ak.platt_recalibrate(p_raw, **kwargs)
    except ValueError as exc:
        return "SF-BL-10904" in str(exc) and "p_raw" in str(exc)
    except Exception:  # noqa: BLE001
        return False
    return False


TYPE_GRID = ["0.6", True, False, None, "", "foo", "1.0", [0.6],
             {"p": 0.6}, (0.6,), 0.6j, Decimal("0.6"), object]


# ---- 1. TASK fixtures: type-invalid p_raw raises naming SF-BL-11014 ---------

def test_task_fixtures_raise_11014():
    assert _raises_11014("0.6"), "0.6"
    assert _raises_11014(True), True


def test_type_grid_raises_11014():
    for p in TYPE_GRID:
        assert _raises_11014(p), p


def test_type_grid_no_bare_typeerror_no_silent_return():
    # Filing behaviour was a bare TypeError ('0.6') or a silent clip
    # return (True -> 0.98). Every type-invalid probe must raise
    # ValueError specifically — _raises_11014 already returns False
    # on any other exception or on a return; assert it directly too.
    for p in TYPE_GRID:
        try:
            result = ak.platt_recalibrate(p)
        except ValueError:
            continue
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"{p!r} raised {type(exc).__name__}")
        raise AssertionError(f"{p!r} silently returned {result!r}")


def test_type_raises_with_valid_nondefault_params():
    assert _raises_11014("0.6", alpha=0.5, beta=-0.3)
    assert _raises_11014(True, alpha=2.0, beta=0.25)
    assert _raises_11014(None, alpha=1.0, beta=0.0)


def test_type_named_before_bad_params():
    # p_raw's type is named (SF-BL-11014) even when alpha is also bad.
    assert _raises_11014("0.6", alpha=NAN)
    assert _raises_11014(True, beta=NAN)


def test_type_named_before_bad_bounds():
    assert _raises_11014(True, min_p=0.90, max_p=0.10)
    assert _raises_11014("0.6", min_p=0.90, max_p=0.10)


# ---- 2. Dual naming: landed SF-BL-9579 pins on type fixtures stay true ------

def test_type_raises_also_name_9579():
    for p in ("0.6", None, True, False, [0.6]):
        try:
            ak.platt_recalibrate(p)
            raise AssertionError(f"platt({p!r}) did not raise")
        except ValueError as exc:
            assert "SF-BL-9579" in str(exc), (p, exc)
            assert "p_raw" in str(exc), (p, exc)


# ---- 3. Numeric rejections unchanged (9579 / 10904, no 11014) ---------------

def test_nonfinite_still_raises_9579_only():
    for p in (NAN, INF, -INF, 10 ** 400):
        assert _raises_9579_only(p), p
    assert _raises_9579_only(NAN, alpha=NAN)


def test_out_of_range_still_raises_10904():
    for p in (82.0, 1.50, -0.20, 2.0, -1.0):
        assert _raises_10904(p), p


def test_sibling_validations_unchanged():
    try:
        ak.platt_recalibrate(0.60, alpha=NAN)
        raise AssertionError("alpha=NaN did not raise")
    except ValueError as exc:
        assert "SF-BL-9508" in str(exc) and "alpha" in str(exc), exc
    try:
        ak.platt_recalibrate(0.60, min_p=0.90, max_p=0.10)
        raise AssertionError("inverted bounds did not raise")
    except ValueError as exc:
        assert "SF-BL-9713" in str(exc), exc


# ---- 4. Numeric in-range recalibrations bit-identical -----------------------

def test_numeric_in_range_unchanged():
    cases = [
        (0.60, 0.6116185719584099),
        (0.30, 0.2790912854168121),
        (0.90, 0.9213542876231214),
        (0.02, 0.02),
        (0.98, 0.98),
        (0.5, 0.5),
        (0.0, 0.02),
        (1.0, 0.98),
        (0, 0.02),   # int p_raw accepted (a real number, not a bool)
        (1, 0.98),
    ]
    for p, exp in cases:
        result = ak.platt_recalibrate(p)
        assert abs(result - exp) <= 1e-12, (p, result, exp)


def test_numeric_with_valid_params_unchanged():
    assert abs(ak.platt_recalibrate(0.60, alpha=0.5)
               - 0.5505102572168218) <= 1e-12
    assert abs(ak.platt_recalibrate(0.60, alpha=1.0) - 0.6) <= 1e-12
    assert abs(ak.platt_recalibrate(0.60, alpha=2)
               - 0.6923076923076922) <= 1e-12
    assert abs(ak.platt_recalibrate(0.60, beta=-0.3)
               - 0.5384541427883593) <= 1e-12
    assert abs(ak.platt_recalibrate(0.60, beta=0.25)
               - 0.6691007993209436) <= 1e-12


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
