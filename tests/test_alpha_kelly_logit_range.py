"""
test_alpha_kelly_logit_range.py — SF-BL-9883 verification tests.

Pre-registered done/kill criteria (TASK.md):
  "logit must raise on non-finite input and on finite input outside
   [0, 1] (endpoints may clip); kill if a caller census proves every
   logit caller pre-validates its argument to a finite in-range
   probability."

Kill condition is NOT met (see CALLER_CENSUS.md in the SF-BL-9883
work dir): aggregate's weighted branch logits clips in
[0.005, 0.995] (in-range by construction), but platt_recalibrate
pre-validates p_raw finite-only (SF-BL-9579) and forwards finite
out-of-range values into logit, and direct callers reach the public
logit with no validation at all. Per PREREG.md in the work dir: p
must be a finite real in the closed interval [0, 1], validated at
logit entry in _validate_logit_p — the SF-BL-9675 finiteness check
first (non-finite / bool / non-numeric keep raising naming
SF-BL-9675), then the range check raising naming SF-BL-9883.
Endpoints 0.0 / 1.0 (and ints 0 / 1) stay valid and clip
bit-identically. Inherited effect: platt_recalibrate's finite
out-of-range p_raw raised naming SF-BL-9883 via logit when this
suite landed; SF-BL-10904 has since landed the SF-BL-10904-named
entry check at platt itself (before logit is reached), and the
platt fixture below now pins that name; in-range platt and both
aggregate panels are unchanged.

Pre-fix reproduction (verified live 2026-10-23; repro.py /
repro_prefix.txt in the work dir):
  logit(2.0) -> 13.815509557935018; logit(-1.0) ->
  -13.815509557963773 (the TASK's 13.8155 / -13.8155)
  logit(1.5) / (82.0) -> 13.815509557935018; (-0.2) ->
  -13.815509557963773; platt(-0.2) -> 0.02; platt(1.5) / (82.0) ->
  0.98; logit(nan/inf) already raised naming SF-BL-9675

Run:  python3 test_alpha_kelly_logit_range.py  (exit 0 iff all pass)
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


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _f(agent, fam, p, weight=1.0, independent=True):
    return ak.SubAgentForecast(agent, fam, p, weight, independent)


def _title_panel():
    return [_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60),
            _f("a3", "fam3", 0.65)]


def _raises_9883_logit(p):
    """True iff logit raises ValueError naming SF-BL-9883 + logit domain."""
    try:
        ak.logit(p)
    except ValueError as exc:
        return "SF-BL-9883" in str(exc) and "[0, 1]" in str(exc)
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


# ---- 1. Finite out-of-range p raises at logit (TASK core) ------------------

def test_task_title_cases_raise():
    # TASK title: logit(2.0) -> 13.8155, logit(-1.0) -> -13.8155 pre-fix.
    assert _raises_9883_logit(2.0)
    assert _raises_9883_logit(-1.0)


def test_finite_out_of_range_grid_raises():
    for p in (1.5, -0.2, 82.0, 1.0000001, -1e-9, 100.0, -100.0):
        assert _raises_9883_logit(p), p


# ---- 2. Non-finite / type rejection unchanged (SF-BL-9675 precedence) ------

def test_nonfinite_still_raises_9675():
    for p in (NAN, INF, -INF):
        assert _raises_9675_logit(p), p


def test_bool_and_non_numeric_still_raise_9675():
    # True == 1.0 / False == 0.0 must NOT slip through the range check
    # as in-range endpoints; the finiteness screen rejects them first.
    for p in ("0.5", None, True, False, [0.5]):
        assert _raises_9675_logit(p), p


# ---- 3. In-domain fixtures unchanged (byte-identical) ----------------------

def test_in_range_logit_unchanged():
    cases = [
        (0.5, 0.0),
        (0.02, -3.8918202981106265),
        (0.98, 3.8918202981106256),
        (0.25, -1.0986122886681098),
        (0.75, 1.0986122886681098),
        # Endpoints may clip (TASK): bit-identical to pre-fix.
        (0.0, -13.815509557963773),
        (1.0, 13.815509557935018),
        (0, -13.815509557963773),  # int endpoints accepted
        (1, 13.815509557935018),
    ]
    for p, exp in cases:
        result = ak.logit(p)
        assert result == exp, (p, result, exp)


def test_clip_probability_untouched():
    # clip's finite out-of-range clipping is correct for a clip
    # function and is SF-BL-9675/9713's closed surface.
    assert ak.clip_probability(-0.2) == 0.02
    assert ak.clip_probability(1.5) == 0.98
    assert ak.clip_probability(0.5) == 0.5


# ---- 4. Inherited surfaces --------------------------------------------------

def test_platt_in_range_unchanged():
    assert abs(ak.platt_recalibrate(0.60) - 0.6116185719584099) <= 1e-12
    assert ak.platt_recalibrate(0.0) == 0.02
    assert ak.platt_recalibrate(1.0) == 0.98
    # platt's own non-finite p_raw still raises naming SF-BL-9579 first.
    try:
        ak.platt_recalibrate(NAN)
        raise AssertionError("platt(NaN) did not raise")
    except ValueError as exc:
        assert "SF-BL-9579" in str(exc), exc


def test_platt_out_of_range_now_raises_via_logit():
    # Was inherited from logit (PREREG.md): the SF-BL-10904 fixtures
    # raised naming SF-BL-9883 via the internal logit call.
    # SF-BL-10904 has since landed platt's own entry check, so these
    # now raise naming SF-BL-10904 and p_raw at platt entry, before
    # logit is reached (pinned in
    # tests/test_alpha_kelly_platt_p_raw_range.py). Direct logit
    # fixtures above keep raising naming SF-BL-9883 unchanged.
    for p in (-0.2, 1.5, 82.0):
        try:
            ak.platt_recalibrate(p)
            raise AssertionError(f"platt({p}) did not raise")
        except ValueError as exc:
            assert "SF-BL-10904" in str(exc), (p, exc)


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
