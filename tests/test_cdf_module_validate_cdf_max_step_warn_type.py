"""
test_cdf_module_validate_cdf_max_step_warn_type.py — SF-BL-11133 verification.

Pre-registered done/kill criteria (TASK.md):
  "validate_cdf raises ValueError naming max_step_warn on any bool,
   string or None max_step_warn, with numeric-threshold problem
   lists bit-identical; kill if max_step_warn=True returns an empty
   problem list for [0.0, 0.9, 1.0] or '0.19'/None raises a bare
   TypeError instead of the named ValueError."

Bug: cdf_module.validate_cdf never type-validated max_step_warn.
True was read as 1.0, so the concentrated CDF [0.0, 0.9, 1.0] was
certified valid ([]); '0.19' died with a bare numpy UFuncTypeError
and None with a bare TypeError, neither naming max_step_warn.
Distinct from SF-BL-10355 (value domain of a numeric threshold:
non-finite / negative), a separate task — this fix is type-only
and leaves every numeric threshold, incl. NaN/inf/negative,
behaving exactly as before.

Fix (metaculus_bot/cdf_module.py, validate_cdf): a type screen at
entry — bool/np.bool_ rejected explicitly (a bool is not a
number), str/bytes/None rejected explicitly, any other
non-numeric rejected via a float() probe re-raised as a named
ValueError — raises "validate_cdf requires max_step_warn to be a
real number; got max_step_warn=... (SF-BL-11133)". Numeric
thresholds keep the original object for comparison/formatting,
so their problem lists are bit-identical.

Run:  python3 tests/test_cdf_module_validate_cdf_max_step_warn_type.py
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metaculus_bot"))

import numpy as np  # noqa: E402

import cdf_module as cm  # noqa: E402

CONCENTRATED = [0.0, 0.9, 1.0]


def _reference(cdf, max_step_warn=0.19):
    """Verbatim pre-11133 validate_cdf logic for numeric panels."""
    cdf = np.asarray(cdf, dtype=float)
    problems = []
    if not np.all(np.isfinite(cdf)):
        problems.append("CDF contains non-finite values")
    if np.any((cdf < 0) | (cdf > 1)):
        problems.append("CDF values outside [0, 1]")
    if np.any(np.diff(cdf) < -1e-12):
        problems.append("CDF is not non-decreasing")
    steps = np.diff(cdf)
    if len(steps) and steps.max() > max_step_warn:
        problems.append(f"largest CDF step {steps.max():.3f} > {max_step_warn}")
    return problems


def _assert_named_valueerror(bad, cdf=CONCENTRATED):
    try:
        cm.validate_cdf(cdf, max_step_warn=bad)
    except ValueError as exc:
        assert type(exc) is ValueError, type(exc)
        assert "max_step_warn" in str(exc), str(exc)
        return
    except Exception as exc:  # bare TypeError / UFuncTypeError = kill
        raise AssertionError(
            f"max_step_warn={bad!r} raised bare {type(exc).__name__}: {exc}")
    raise AssertionError(f"max_step_warn={bad!r} did not raise")


def test_kill_true_does_not_certify_concentrated_cdf():
    # Pre-fix this returned [] (threshold read as 1.0).
    _assert_named_valueerror(True)
    _assert_named_valueerror(False)
    _assert_named_valueerror(np.bool_(True))


def test_kill_string_and_none_raise_named_valueerror():
    _assert_named_valueerror("0.19")
    _assert_named_valueerror(None)


def test_all_bad_types_raise_named():
    for bad in ("", "abc", b"0.19", np.str_("0.19"), [0.19], (0.19,),
                {"x": 0.19}, 0.19j, object(), np.array([0.19, 0.2])):
        _assert_named_valueerror(bad)


def test_bad_threshold_raises_regardless_of_cdf():
    for cdf in ([], [0.5], list(np.linspace(0, 1, 201)), [True] * 201):
        _assert_named_valueerror("0.19", cdf=cdf)
        _assert_named_valueerror(None, cdf=cdf)
        _assert_named_valueerror(True, cdf=cdf)


def test_numeric_thresholds_bit_identical():
    panels = [
        CONCENTRATED, list(np.linspace(0, 1, 201)), [0.0, 0.1, 0.2],
        [0.5, 0.5, 0.5], [0.0, 1.0], [-0.1, 0.5, 1.2], [0.9, 0.5, 0.9],
        [0.0, 0.5, 0.55], [], [0.5], [2.0], [0, 1, 1],
        np.array([0, 0, 1]), (0.0, 0.25, 0.5, 1.0),
        [0.1, float("nan"), 0.9], [0.1, float("inf"), 0.9],
    ]
    thresholds = [0.19, 0.0, 0.5, 0.9, 1, 2, 0, -0.1,
                  np.float64(0.19), np.int64(1), np.float32(0.5),
                  float("nan"), float("inf"), float("-inf")]
    for cdf in panels:
        assert cm.validate_cdf(cdf) == _reference(cdf), cdf
        for th in thresholds:
            assert cm.validate_cdf(cdf, max_step_warn=th) == \
                _reference(cdf, max_step_warn=th), (cdf, th)
    rng = np.random.default_rng(11133)
    for i in range(200):
        cdf = np.sort(rng.random(int(rng.integers(2, 60))))
        th = [0.05, 0.19, 0.3, 0.75, 1.0][i % 5]
        assert cm.validate_cdf(cdf, max_step_warn=th) == \
            _reference(cdf, max_step_warn=th)


def test_default_and_lawful_cdf_unchanged():
    assert cm.validate_cdf(CONCENTRATED) == \
        ["largest CDF step 0.900 > 0.19"]
    assert cm.validate_cdf(CONCENTRATED, max_step_warn=1) == []  # int 1 is lawful
    cdf = cm.percentiles_to_cdf({0.05: 90.0, 0.25: 98.0, 0.5: 100.0,
                                 0.75: 102.0, 0.95: 110.0}, 0.0, 200.0)
    assert cm.validate_cdf(cdf) == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} tests passed")
