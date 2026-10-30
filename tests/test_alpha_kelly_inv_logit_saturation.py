"""
test_alpha_kelly_inv_logit_saturation.py — SF-BL-9919 verification tests.

Pre-registered done/kill criteria (TASK.md):
  "Fix makes inv_logit reject non-finite input and clip its output
   into the open interval (0, 1) (or the [0.02, 0.98] band): repros
   inv_logit(1000.0) must return < 1.0 and inv_logit(NaN) must raise;
   kill if pooled_prob on the e1 fixture panel is bit-identical
   pre/post fix."

Per PREREG.md in the SF-BL-9919 work dir, the fix clips into the open
interval (0, 1) at the nextafter bounds (INV_LOGIT_MIN = 5e-324,
INV_LOGIT_MAX = 0.9999999999999999): the [0.02, 0.98] band is
clip_probability's policy and already applies downstream of both
inv_logit call sites (platt_recalibrate / aggregate_fleet_forecasts),
while embedding it in the sigmoid would break the logit round-trip
for |x| > 3.89. The clip therefore binds only on exact 0.0 / 1.0
outputs; every finite non-saturating value is bit-identical pre/post.

Kill criterion: NOT triggered in its meaningful form (CALLER_CENSUS.md
in the work dir). Ordinary panels cannot saturate — the aggregate
member clip [0.005, 0.995] bounds |weighted_logit| <= 5.293 and at the
default extremizing_a = 1.25 the pooled argument is <= 6.62 — so
their bit-identity is the required regression property. On a panel
that DOES saturate (members at 0.995, extremizing_a = 10, a value
SF-BL-9476 leaves valid: finite, > 0, no upper bound), pooled_prob is
exactly 1.0 pre-fix and 0.9999999999999999 post-fix — not
bit-identical (test_pooled_saturating_panel_changes).

Pre-fix reproduction (verified live 2026-10-23; repro.py /
repro_prefix.txt in the work dir):
  inv_logit(1000.0) -> 1.0; inv_logit(-1000.0) -> 0.0
  inv_logit(nan) -> nan; inv_logit(+inf) -> 1.0; inv_logit(-inf) -> 0.0
  inv_logit(37.0) -> 1.0 (positive saturation begins at x = 37)
  inv_logit(-746.0) -> 0.0 (exp underflow)
  platt_recalibrate(0.99, alpha=1e308) -> 0.98 (product overflows to
  +inf from all-finite validated inputs; inv_logit(+inf) laundered it)
  pooled_prob, 0.995 panel at a=10 -> exactly 1.0

Run:  python3 test_alpha_kelly_inv_logit_saturation.py
(exit 0 iff all pass). No pytest dependency: test_* functions with
assert + __main__ runner, same convention as
tests/test_alpha_kelly_clip_logit_finite.py.
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


def _raises_named(fn, *needles):
    try:
        fn()
    except ValueError as exc:
        msg = str(exc)
        return all(n in msg for n in needles)
    except Exception:
        return False
    return False


def test_task_repros_clip_into_open_interval():
    hi = ak.inv_logit(1000.0)
    assert hi < 1.0, hi                       # TASK done criterion
    assert hi == ak.INV_LOGIT_MAX == 0.9999999999999999, hi
    assert hi > 0.98, hi
    lo = ak.inv_logit(-1000.0)
    assert lo > 0.0, lo                       # open interval, low side
    assert lo == ak.INV_LOGIT_MIN == 5e-324, lo
    assert lo < 0.02, lo
    # saturation boundary: positive side saturates from x = 37,
    # negative-side exact zero from x = -746
    for x in (37.0, 40.0, 50.0, 100.0, 745.0):
        assert ak.inv_logit(x) == ak.INV_LOGIT_MAX, (x, ak.inv_logit(x))
    for x in (-746.0, -1000.0):
        assert ak.inv_logit(x) == ak.INV_LOGIT_MIN, (x, ak.inv_logit(x))


def test_nonfinite_raises_named_valueerror():
    for bad in (NAN, INF, -INF):
        assert _raises_named(lambda b=bad: ak.inv_logit(b),
                             "SF-BL-9919", "not finite", repr(bad)), bad


def test_bool_and_non_numeric_raise_named_valueerror():
    # Overlap with SF-BL-11042 is deliberate (PREREG.md): pre-fix
    # '1.0'/None died with a bare TypeError from the x >= 0 comparison
    # and True silently returned 0.7310585786300049.
    for bad in (True, False, "1.0", None, [1.0]):
        assert _raises_named(lambda b=bad: ak.inv_logit(b),
                             "SF-BL-9919", "not finite"), bad


def test_finite_interior_bit_identical():
    # Every value below is the pre-fix output recorded this run
    # (repro_prefix.txt); none saturates, so none may move.
    pinned = {
        0.0: 0.5,
        1.0: 0.7310585786300049,
        -1.0: 0.2689414213699951,
        2.0: 0.8807970779778823,
        -2.0: 0.11920292202211755,
        3.8918202981106265: 0.98,
        5.0: 0.9933071490757153,
        -5.0: 0.006692850924284856,
        10.0: 0.9999546021312976,
        -10.0: 4.5397868702434395e-05,
        20.0: 0.9999999979388463,
        -20.0: 2.0611536181902033e-09,
        30.0: 0.9999999999999065,
        -30.0: 9.3576229688393e-14,
        35.0: 0.9999999999999993,
        36.0: 0.9999999999999998,
        -36.0: 2.319522830243569e-16,
        -37.0: 8.533047625744066e-17,
        -40.0: 4.248354255291589e-18,
        -100.0: 3.720075976020836e-44,
        -745.0: 5e-324,
    }
    for x, want in pinned.items():
        got = ak.inv_logit(x)
        assert got == want and repr(got) == repr(want), (x, got, want)
    assert ak.inv_logit(5) == 0.9933071490757153  # int input


def test_output_always_in_open_interval():
    x = -1000.0
    while x <= 1000.0:
        r = ak.inv_logit(x)
        assert 0.0 < r < 1.0, (x, r)
        x += 0.5


def test_round_trip_preserved():
    for x in (0.0, 1.5, -2.5, -3.5):
        assert ak.logit(ak.inv_logit(x)) == x, x
    # x = 3.0 round-trips to 3.000000000000003 on the PRE-fix code too
    # (float arithmetic, verified this run) — pinned, not "fixed".
    assert ak.logit(ak.inv_logit(3.0)) == 3.000000000000003
    for p in (0.02, 0.25, 0.5, 0.75, 0.98):
        assert ak.inv_logit(ak.logit(p)) == p, p


def test_platt_surface_unchanged():
    assert ak.platt_recalibrate(0.60) == 0.6116185719584099
    assert ak.platt_recalibrate(0.99) == 0.98
    assert ak.platt_recalibrate(0.01) == 0.02


def test_platt_overflow_alpha_raises():
    # alpha = 1e308 is finite and valid under SF-BL-9508 (no upper
    # bound), but alpha * logit(0.99) overflows to +inf; pre-fix
    # inv_logit(+inf) == 1.0 laundered that to the 0.98 cap.
    assert _raises_named(lambda: ak.platt_recalibrate(0.99, alpha=1e308),
                         "SF-BL-9919", "not finite")


def _panel(probs):
    return [ak.SubAgentForecast(f"a{i + 1}", f"f{i + 1}", p)
            for i, p in enumerate(probs)]


def test_aggregate_panels_unchanged():
    result = ak.aggregate_fleet_forecasts(_panel([0.70, 0.60, 0.65]))
    assert result == (0.6989517116705906, 0.04082482904638629), result
    result = ak.aggregate_fleet_forecasts(_panel([0.70, 0.70, 0.70]))
    assert result == (0.7583861891129468, 0.0), result


def _pooled_prob(panel, a):
    # Replicates aggregate's weighted log-odds branch exactly.
    probs = [ak.clip_probability(f.raw_probability, 0.005, 0.995)
             for f in panel]
    total_w = sum(f.historical_brier_weight for f in panel)
    weighted_logit = sum(
        (f.historical_brier_weight / total_w) * ak.logit(p)
        for f, p in zip(panel, probs))
    return ak.inv_logit(a * weighted_logit)


def test_pooled_saturating_panel_changes():
    extreme = _panel([0.995, 0.995, 0.995])
    # Default factor: cannot saturate, bit-identical to pre-fix.
    assert _pooled_prob(extreme, 1.25) == 0.998663856858122
    assert _pooled_prob(_panel([0.70, 0.60, 0.65]), 1.25) \
        == 0.6856623489134867
    # Saturating factor: pre-fix pooled_prob was exactly 1.0; the fix
    # must change it (kill criterion, meaningful form — PREREG.md).
    pooled = _pooled_prob(extreme, 10.0)
    assert pooled == ak.INV_LOGIT_MAX and pooled < 1.0, pooled
    # The calibrated aggregate return on that panel is 0.98 both
    # pre- and post-fix: platt's own [0.02, 0.98] clip masks the
    # pooled change downstream, as registered in PREREG.md.
    result = ak.aggregate_fleet_forecasts(extreme, extremizing_a=10.0)
    assert result == (0.98, 0.0), result


def test_benchmark_fixture_unchanged():
    # Pins recorded this run on the pre-fix tree (repro_prefix.txt);
    # the disciplined score reflects concurrent fleet landings
    # (SF-BL-9883 et al.), the naive score is the long-standing pin.
    buf = io.StringIO()
    with redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    assert out.count('"north_star_score":') == 2, out[-2000:]
    assert '"north_star_score": -0.1611' in out, out[-2000:]
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
