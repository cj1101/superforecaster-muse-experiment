"""
test_alpha_kelly_clip_logit_finite.py — SF-BL-9675 verification tests.

Pre-registered done/kill criteria (TASK.md):
  "Done when clip_probability raises ValueError on NaN/+inf/-inf/
   non-numeric input (finite in-range and clipping behaviour
   byte-identical) and logit inherits the rejection; killed if a
   caller census proves every clip_probability/logit call site
   pre-validates finiteness with a tested invariant."

Kill condition is NOT met (see CALLER_CENSUS.md in the SF-BL-9675
work dir): aggregate's member clips and evaluate's posterior clip are
shielded by sibling filters and platt's internal logit call by
SF-BL-9579's entry check, but both helpers are public and direct
callers reach them with no validation at all. Per PREREG.md in the
work dir: p must be a finite real for BOTH helpers, validated at
entry via _is_finite_number, else a ValueError naming SF-BL-9675 and
p. Finite out-of-range values keep their pre-fix clip/clamp behaviour
bit-identically (logit's finite range is SF-BL-9883's surface; the
clip bounds are SF-BL-9713 / 11122 / 11491's). The bool/non-numeric
rejection via _is_finite_number is the deliberate SF-BL-11079 overlap
recorded in PREREG.md.

Pre-fix reproduction (verified live 2026-10-23; repro.py /
repro_prefix.txt in the work dir):
  clip_probability(nan) -> 0.98; (+inf) -> 0.98; (-inf) -> 0.02
  clip_probability(nan, 0.1, 0.9) -> 0.9 (custom max laundered too)
  logit(nan) -> 13.815509557935018; (+inf) -> same;
  logit(-inf) -> -13.815509557963773
  clip/logit('0.5') / (None) -> bare TypeError; (True) -> 0.98 / 13.8155

Run:  python3 test_alpha_kelly_clip_logit_finite.py  (exit 0 iff all pass)
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


def _raises_9675_clip(p, **kwargs):
    """True iff clip_probability raises ValueError naming SF-BL-9675 + p."""
    try:
        ak.clip_probability(p, **kwargs)
    except ValueError as exc:
        return "SF-BL-9675" in str(exc) and "clip_probability" in str(exc)
    except Exception:  # noqa: BLE001 - wrong exception type is a failure
        return False
    return False


def _raises_9675_logit(p):
    """True iff logit raises ValueError naming SF-BL-9675 + logit."""
    try:
        ak.logit(p)
    except ValueError as exc:
        return "SF-BL-9675" in str(exc) and "logit" in str(exc)
    except Exception:  # noqa: BLE001
        return False
    return False


# ---- 1. Non-finite p raises at clip_probability (TASK core) ----------------

def test_nonfinite_clip_raises():
    # TASK title cases: clip_probability(NaN) -> 0.98, (+inf) -> 0.98 pre-fix.
    for p in (NAN, INF, -INF):
        assert _raises_9675_clip(p), p


def test_nonfinite_clip_raises_with_custom_bounds():
    assert _raises_9675_clip(NAN, min_p=0.1, max_p=0.9)
    assert _raises_9675_clip(INF, min_p=0.005, max_p=0.995)
    assert _raises_9675_clip(-INF, min_p=0.1, max_p=0.9)


def test_bool_and_non_numeric_clip_raise_named_valueerror():
    # SF-BL-11079 overlap (deliberate, see module docstring): pre-fix
    # '0.5'/None died with a bare TypeError and True returned 0.98.
    for p in ("0.5", None, True, False, [0.5]):
        assert _raises_9675_clip(p), p


# ---- 2. logit inherits the rejection ----------------------------------------

def test_nonfinite_logit_raises():
    # TASK title case: logit(NaN) -> 13.8155 pre-fix.
    for p in (NAN, INF, -INF):
        assert _raises_9675_logit(p), p


def test_bool_and_non_numeric_logit_raise_named_valueerror():
    for p in ("0.5", None, True, False, [0.5]):
        assert _raises_9675_logit(p), p


# ---- 3. Finite fixtures unchanged (byte-identical) --------------------------

def test_finite_clip_unchanged():
    cases = [
        (0.5, 0.5), (0.02, 0.02), (0.98, 0.98), (0.25, 0.25),
        (0.0, 0.02), (0.999, 0.98), (1.0, 0.98),
        # Finite out-of-range: pre-fix clipping preserved exactly.
        (-0.2, 0.02), (1.5, 0.98),
        (0, 0.02), (1, 0.98),  # int p accepted
    ]
    for p, exp in cases:
        result = ak.clip_probability(p)
        assert result == exp, (p, result, exp)


def test_finite_clip_custom_bounds_unchanged():
    assert ak.clip_probability(0.5, 0.1, 0.9) == 0.5
    assert ak.clip_probability(0.05, 0.1, 0.9) == 0.1
    assert ak.clip_probability(0.95, 0.1, 0.9) == 0.9
    assert ak.clip_probability(0.001, 0.005, 0.995) == 0.005
    assert ak.clip_probability(0.999, 0.005, 0.995) == 0.995


def test_finite_logit_unchanged():
    cases = [
        (0.5, 0.0),
        (0.02, -3.8918202981106265),
        (0.98, 3.8918202981106256),
        (0.25, -1.0986122886681098),
        (0.75, 1.0986122886681098),
        # Endpoints clamp bit-identically (SF-BL-9883: endpoints may
        # clip). Finite out-of-range (-0.2 / 1.5) was pinned here
        # pre-SF-BL-9883; it now raises naming SF-BL-9883 and is
        # pinned in tests/test_alpha_kelly_logit_range.py.
        (0.0, -13.815509557963773),
        (1.0, 13.815509557935018),
    ]
    for p, exp in cases:
        result = ak.logit(p)
        assert result == exp, (p, result, exp)


# ---- 4. Inherited surfaces unchanged ----------------------------------------

def test_platt_surface_unchanged():
    assert abs(ak.platt_recalibrate(0.60) - 0.6116185719584099) <= 1e-12
    # platt's own non-finite p_raw still raises naming SF-BL-9579 first.
    try:
        ak.platt_recalibrate(NAN)
        raise AssertionError("platt(NaN) did not raise")
    except ValueError as exc:
        assert "SF-BL-9579" in str(exc), exc


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
