"""
test_alpha_kelly_platt_p_raw.py — SF-BL-9579 verification tests.

Pre-registered done/kill criteria (TASK.md):
  "Done when platt_recalibrate raises ValueError on non-finite p_raw,
   alpha or beta (and aggregate_fleet_forecasts inherits the
   rejection), with finite-parameter fixtures unchanged; killed if a
   producer audit shows alpha/beta only ever come from a
   fitted-and-validated calibrator object."

Kill condition is NOT met (see PRODUCER_AUDIT.md in the SF-BL-9579
work dir): alpha/beta are bare caller configuration from no calibrator
at all (SF-BL-9508 audit), and p_raw reaches the public
platt_recalibrate from direct callers with no validated producer
invariant. Per PREREG.md in the work dir: p_raw must be a finite real
(any finite value passed bit-identically at this task's landing;
finite out-of-range was SF-BL-10904's separate surface and now
raises naming SF-BL-10904 at platt entry — see the fixture note in
test_finite_p_raw_unchanged), validated at
platt_recalibrate entry BEFORE the SF-BL-9508 alpha/beta check, else a
ValueError naming SF-BL-9579 and p_raw. The alpha/beta half of the
done criterion was already landed by SF-BL-9508 and is re-pinned here
at the platt + aggregate surfaces. Validation via _is_finite_number
also rejects bool / non-numeric p_raw with the named ValueError — the
deliberate SF-BL-11014 overlap recorded in PREREG.md.

Pre-fix reproduction (verified live 2026-10-23; repro.py /
repro_prefix.txt in the work dir):
  platt_recalibrate(nan) -> 0.98; (+inf) -> 0.98; (-inf) -> 0.02
  platt_recalibrate('0.6') / (None) -> bare TypeError
  platt_recalibrate(True) -> 0.98; (False) -> 0.02
  platt_recalibrate(0.60) default -> 0.6116185719584099
  aggregate panel 0.70/0.60/0.65 default
      -> (0.6989517116705906, 0.04082482904638629)

Run:  python3 test_alpha_kelly_platt_p_raw.py  (exit 0 iff all pass)
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_alpha_kelly_platt_params.py.
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


def _raises_9579(p_raw, **kwargs):
    """True iff platt_recalibrate raises ValueError naming SF-BL-9579 + p_raw."""
    try:
        ak.platt_recalibrate(p_raw, **kwargs)
    except ValueError as exc:
        return "SF-BL-9579" in str(exc) and "p_raw" in str(exc)
    except Exception:  # noqa: BLE001 - wrong exception type is a failure
        return False
    return False


def _raises_9508_platt(param, **kwargs):
    try:
        ak.platt_recalibrate(0.60, **kwargs)
    except ValueError as exc:
        return "SF-BL-9508" in str(exc) and param in str(exc)
    except Exception:  # noqa: BLE001
        return False
    return False


def _raises_9508_agg(param, **kwargs):
    try:
        ak.aggregate_fleet_forecasts(_title_panel(), **kwargs)
    except ValueError as exc:
        return "SF-BL-9508" in str(exc) and param in str(exc)
    except Exception:  # noqa: BLE001
        return False
    return False


# ---- 1. Non-finite p_raw raises (the TASK's core criterion) -----------------

def test_nonfinite_p_raw_raises():
    for p in (NAN, INF, -INF):
        assert _raises_9579(p), p


def test_nonfinite_p_raw_raises_with_valid_nondefault_params():
    assert _raises_9579(NAN, alpha=0.5, beta=-0.3)
    assert _raises_9579(INF, alpha=2.0, beta=0.25)
    assert _raises_9579(-INF, alpha=1.0, beta=0.0)


def test_p_raw_named_before_bad_params():
    # Entry order per PREREG: a non-finite p_raw is named (SF-BL-9579)
    # even when alpha is also non-finite.
    assert _raises_9579(NAN, alpha=NAN)
    assert _raises_9579(INF, beta=NAN)


def test_bool_and_non_numeric_p_raw_raise_named_valueerror():
    # SF-BL-11014 overlap (deliberate, see module docstring): pre-fix
    # '0.6'/None died with a bare TypeError and True returned 0.98.
    for p in ("0.6", None, True, False, [0.6]):
        assert _raises_9579(p), p


# ---- 2. Alpha/beta half of the criterion (SF-BL-9508, re-pinned) ------------

def test_platt_nonfinite_alpha_beta_raise():
    # TASK title cases: platt_recalibrate(0.60, alpha=NaN), alpha=+inf.
    for a in (NAN, INF, -INF):
        assert _raises_9508_platt("alpha", alpha=a), a
    for b in (NAN, INF, -INF):
        assert _raises_9508_platt("beta", beta=b), b


def test_aggregate_inherits_rejection():
    # TASK title case: aggregate(platt_alpha=NaN) on the 0.70/0.60/0.65
    # panel raises instead of pooling to 0.98.
    assert _raises_9508_agg("platt_alpha", platt_alpha=NAN)
    assert _raises_9508_agg("platt_alpha", platt_alpha=INF)
    assert _raises_9508_agg("platt_beta", platt_beta=NAN)
    assert _raises_9508_agg("platt_beta", platt_beta=INF)


# ---- 3. Finite fixtures unchanged -------------------------------------------

def test_finite_p_raw_unchanged():
    cases = [
        (0.60, 0.6116185719584099),
        (0.30, 0.2790912854168121),
        (0.90, 0.9213542876231214),
        (0.02, 0.02),
        (0.98, 0.98),
        (0.5, 0.5),
        # Endpoints stay in-domain (SF-BL-9883: endpoints may clip).
        (0.0, 0.02),
        (1.0, 0.98),
        (0, 0.02),   # int p_raw accepted
        (1, 0.98),
    ]
    for p, exp in cases:
        result = ak.platt_recalibrate(p)
        assert abs(result - exp) <= 1e-12, (p, result, exp)
    # Finite out-of-range (-0.2 / 1.5): was pinned here as preserved
    # logit-clamp behaviour under "(SF-BL-10904)", then as raising
    # naming SF-BL-9883 via the internal logit(p_raw) call after
    # SF-BL-9883 landed. SF-BL-10904 has since landed platt's own
    # entry check, so these now raise naming SF-BL-10904 and p_raw
    # at platt_recalibrate entry (pinned in
    # tests/test_alpha_kelly_platt_p_raw_range.py).
    for p in (-0.2, 1.5):
        try:
            ak.platt_recalibrate(p)
            raise AssertionError(f"platt({p}) did not raise")
        except ValueError as exc:
            assert "SF-BL-10904" in str(exc), (p, exc)


def test_finite_p_raw_with_valid_params_unchanged():
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
