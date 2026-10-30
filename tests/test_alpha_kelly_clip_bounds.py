"""
test_alpha_kelly_clip_bounds.py — SF-BL-9713 verification tests.

Pre-registered done/kill criteria (TASK.md):
  "clip_probability raises ValueError unless its bounds are finite
   with 0 <= min_p < max_p <= 1; success = the inverted-bounds and
   NaN-bound repros raise, aggregate_fleet_forecasts inherits the
   rejection, and all default-bound fixtures are byte-identical;
   kill if a call-site census proves the bounds are only ever the
   in-code defaults and the parameters are pinned or removed."

Kill condition is NOT met (see CALLER_CENSUS.md in the SF-BL-9713
work dir): the bounds are public caller parameters on
clip_probability, platt_recalibrate and aggregate_fleet_forecasts —
never pinned or removed — and direct callers reach the chain with
corrupt bounds live. Per PREREG.md in the work dir: bounds must be
finite reals with 0 <= min_p < max_p <= 1, validated at
clip_probability entry (after the SF-BL-9675 p check) and at
platt_recalibrate / aggregate_fleet_forecasts entry via
_is_finite_number, else a ValueError naming SF-BL-9713 and the
offending bound. Bool / non-numeric bounds raise the same named
ValueError (deliberate SF-BL-11122/11491 type overlap recorded in
PREREG.md). Pure validation: every in-domain pair — defaults
0.02/0.98, member 0.005/0.995, boundary 0.0/1.0, int 0/1 — clips
bit-identically to pre-fix.

Pre-fix reproduction (verified live 2026-10-23; repro.py /
repro_prefix.txt in the work dir):
  clip_probability(0.5, 0.90, 0.10) -> 0.90 (every p pinned)
  clip_probability(0.5, nan, 0.98) -> nan; (0.5, 0.02, nan) -> 0.02
  clip_probability(0.5, +inf, 0.98) -> inf; (0.5, 0.5, 0.5) -> 0.5
  platt_recalibrate(0.60, min_p=0.90, max_p=0.10) -> 0.90
  aggregate panel 0.70/0.60/0.65 at min_p=nan -> (nan, 0.04082...)
  clip bounds ('0.02') / (None) -> bare TypeError; (True) -> True

Run:  python3 test_alpha_kelly_clip_bounds.py  (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_clip_logit_finite.py.
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

BAD_BOUNDS = [
    dict(min_p=0.90, max_p=0.10),   # inverted (TASK title case)
    dict(min_p=NAN, max_p=0.98),   # NaN min (TASK title case)
    dict(min_p=0.02, max_p=NAN),   # NaN max
    dict(min_p=INF, max_p=0.98),
    dict(min_p=0.02, max_p=INF),
    dict(min_p=-INF, max_p=0.98),
    dict(min_p=0.02, max_p=-INF),
    dict(min_p=0.5, max_p=0.5),     # equal: collapses to a constant
    dict(min_p=0.0, max_p=0.0),
    dict(min_p=1.0, max_p=1.0),
    dict(min_p=-0.5, max_p=0.98),   # min below the simplex
    dict(min_p=-0.01, max_p=0.98),
    dict(min_p=0.02, max_p=1.5),    # max above the simplex
    dict(min_p=0.02, max_p=1.01),
    dict(min_p=1.5, max_p=1.2),
    dict(min_p=True, max_p=0.98),   # bool bounds (type overlap)
    dict(min_p=0.02, max_p=False),
    dict(min_p="0.02", max_p=0.98),  # non-numeric bounds
    dict(min_p=0.02, max_p=None),
]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _f(agent, fam, p, weight=1.0, independent=True):
    return ak.SubAgentForecast(agent, fam, p, weight, independent)


def _title_panel():
    return [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60),
            _f("a3", "fam3", 0.65)]


def _shared_panel():
    return [_f("a1", "fam1", 0.70), _f("a2", "fam1", 0.60),
            _f("a3", "fam1", 0.65)]


def _raises_9713(fn, *args, **kwargs):
    """True iff fn raises ValueError naming SF-BL-9713 + a bound name."""
    try:
        fn(*args, **kwargs)
    except ValueError as exc:
        return "SF-BL-9713" in str(exc) and (
            "min_p" in str(exc) or "max_p" in str(exc))
    except Exception:  # noqa: BLE001 - wrong exception type is a failure
        return False
    return False


# ---- 1. Bad bounds raise at clip_probability (TASK core) -------------------

def test_inverted_and_nan_bounds_raise():
    # TASK title cases: (0.90, 0.10) -> 0.90 and (NaN, 0.98) -> NaN pre-fix.
    assert _raises_9713(ak.clip_probability, 0.5, min_p=0.90, max_p=0.10)
    assert _raises_9713(ak.clip_probability, 0.5, min_p=NAN, max_p=0.98)
    assert _raises_9713(ak.clip_probability, 0.5, min_p=0.02, max_p=NAN)


def test_full_bad_bound_grid_raises_at_clip():
    for kwargs in BAD_BOUNDS:
        assert _raises_9713(ak.clip_probability, 0.5, **kwargs), kwargs


def test_bad_bounds_raise_for_any_p():
    for p in (0.0, 0.02, 0.5, 0.98, 1.0):
        assert _raises_9713(ak.clip_probability, p,
                            min_p=0.90, max_p=0.10), p


def test_p_check_precedence_preserved():
    # SF-BL-9675: a non-finite p still raises naming SF-BL-9675 first,
    # even when a bound is also bad (PREREG precedence).
    try:
        ak.clip_probability(NAN, min_p=0.90, max_p=0.10)
        raise AssertionError("did not raise")
    except ValueError as exc:
        assert "SF-BL-9675" in str(exc), exc


# ---- 2. platt_recalibrate rejects bad bounds at entry ----------------------

def test_full_bad_bound_grid_raises_at_platt():
    for kwargs in BAD_BOUNDS:
        assert _raises_9713(ak.platt_recalibrate, 0.60, **kwargs), kwargs


def test_platt_own_checks_still_first():
    # SF-BL-9579 p_raw and SF-BL-9508 alpha/beta keep their precedence.
    try:
        ak.platt_recalibrate(NAN, min_p=0.90, max_p=0.10)
        raise AssertionError("did not raise")
    except ValueError as exc:
        assert "SF-BL-9579" in str(exc), exc
    try:
        ak.platt_recalibrate(0.60, alpha=NAN, min_p=0.90, max_p=0.10)
        raise AssertionError("did not raise")
    except ValueError as exc:
        assert "SF-BL-9508" in str(exc), exc


# ---- 3. aggregate_fleet_forecasts inherits the rejection -------------------

def test_full_bad_bound_grid_raises_at_aggregate():
    for kwargs in BAD_BOUNDS:
        assert _raises_9713(ak.aggregate_fleet_forecasts,
                            _title_panel(), **kwargs), kwargs


def test_aggregate_task_fixture_raises():
    # TASK fixture: 0.70/0.60/0.65 panel at min_p=NaN pooled to
    # (NaN, 0.041) pre-fix; inverted bounds pooled to (0.90, ...).
    assert _raises_9713(ak.aggregate_fleet_forecasts,
                        _title_panel(), min_p=NAN)
    assert _raises_9713(ak.aggregate_fleet_forecasts,
                        _title_panel(), min_p=0.90, max_p=0.10)


def test_aggregate_median_branch_and_entry_raise():
    assert _raises_9713(ak.aggregate_fleet_forecasts,
                        _shared_panel(), min_p=0.90, max_p=0.10)
    assert _raises_9713(ak.aggregate_fleet_forecasts,
                        _shared_panel(), min_p=NAN)
    # Entry validation: bad bound beats the empty-list error.
    assert _raises_9713(ak.aggregate_fleet_forecasts, [],
                        min_p=0.90, max_p=0.10)


# ---- 4. In-domain fixtures unchanged (byte-identical) -----------------------

def test_finite_clip_defaults_unchanged():
    cases = [
        (0.5, 0.5), (0.02, 0.02), (0.98, 0.98), (0.25, 0.25),
        (0.0, 0.02), (0.999, 0.98), (1.0, 0.98),
        (-0.2, 0.02), (1.5, 0.98),
        (0, 0.02), (1, 0.98),  # int p accepted (SF-BL-9675)
    ]
    for p, exp in cases:
        result = ak.clip_probability(p)
        assert result == exp, (p, result, exp)


def test_valid_custom_bounds_unchanged():
    assert ak.clip_probability(0.5, 0.1, 0.9) == 0.5
    assert ak.clip_probability(0.05, 0.1, 0.9) == 0.1
    assert ak.clip_probability(0.95, 0.1, 0.9) == 0.9
    assert ak.clip_probability(0.001, 0.005, 0.995) == 0.005
    assert ak.clip_probability(0.999, 0.005, 0.995) == 0.995


def test_boundary_bounds_valid_and_unchanged():
    # 0 <= min_p < max_p <= 1: the boundary pairs are IN domain.
    assert ak.clip_probability(0.5, 0.0, 1.0) == 0.5
    assert ak.clip_probability(-0.2, 0.0, 1.0) == 0.0
    assert ak.clip_probability(1.5, 0.0, 1.0) == 1.0
    assert ak.clip_probability(0.5, 0, 1) == 0.5  # int bounds
    assert ak.clip_probability(0.001, 0.0, 0.98) == 0.001
    assert ak.clip_probability(0.999, 0.02, 1.0) == 0.999


def test_platt_surface_unchanged():
    assert abs(ak.platt_recalibrate(0.60) - 0.6116185719584099) <= 1e-12
    assert abs(ak.platt_recalibrate(0.60, min_p=0.1, max_p=0.9)
               - 0.6116185719584099) <= 1e-12


def test_aggregate_panels_unchanged():
    result = ak.aggregate_fleet_forecasts(_title_panel())
    assert abs(result[0] - 0.6989517116705906) <= 1e-12, result
    assert abs(result[1] - 0.04082482904638629) <= 1e-12, result
    uniform = [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.70),
               _f("a3", "fam3", 0.70)]
    result = ak.aggregate_fleet_forecasts(uniform)
    assert abs(result[0] - 0.7583861891129468) <= 1e-12, result
    assert result[1] == 0.0, result
    result = ak.aggregate_fleet_forecasts(_title_panel(),
                                          min_p=0.1, max_p=0.9)
    assert abs(result[0] - 0.6989517116705906) <= 1e-12, result


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
