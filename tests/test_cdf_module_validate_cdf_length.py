"""
test_cdf_module_validate_cdf_length.py — SF-BL-10426 verification.

Pre-registered done/kill criteria (TASK.md):
  "Fix reports a length problem for len(cdf) < 2 (and for any
   length != expected grid size where known) with tests; kill if
   an empty CDF still returns []."

Bug: cdf_module.validate_cdf iterated only over the CDF's values
and diffs, so validate_cdf([]) == [] and validate_cdf([0.5]) == []
— a zero-length / single-point forecast passed the module's own
validity gate, although the design contract is a 201-point CDF
(discrete: outcome_count+1 points) and downstream consumers need
at least 2 points. Distinct from SF-BL-9901 (NaN VALUES inside a
CDF) and SF-BL-10269 (crps_from_cdf's empty-grid score, a
different function).

Fix (metaculus_bot/cdf_module.py, validate_cdf, both live
copies): after float conversion, a length screen runs BEFORE the
value checks — length 0 reports "CDF length 0 is too short: an
empty CDF is not a valid forecast (a CDF needs at least 2
points)", length 1 the single-point analogue (a bare scalar is
reshaped to a 1-point CDF and reported, not raised), and when
the caller supplies the new expected_size parameter (the grid
size, where known) any other length reports "CDF length N does
not match expected grid size M". A bad expected_size (bool /
string / non-integral / non-finite / < 2) is a caller bug and
raises a ValueError naming expected_size at entry, mirroring
the max_step_warn screen. Scope: this supersedes the length-0/1
panels pinned by SF-BL-9901 / SF-BL-10471 / SF-BL-11133 for
lengths 0 and 1 only — every length >= 2 CDF with
expected_size=None keeps its pre-fix problems list bit-identical
(pinned here against a verbatim copy of the pre-fix logic), as
does any CDF whose length matches a supplied expected_size.

Run:  python3 tests/test_cdf_module_validate_cdf_length.py
"""
import os
import sys
from decimal import Decimal
from fractions import Fraction

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metaculus_bot"))

import numpy as np  # noqa: E402

import cdf_module as cm  # noqa: E402

LEN0 = ("CDF length 0 is too short: an empty CDF is not a "
        "valid forecast (a CDF needs at least 2 points)")
LEN1 = ("CDF length 1 is too short: a single-point CDF is "
        "not a valid forecast (a CDF needs at least 2 points)")
NONFINITE = "CDF contains non-finite values"
NAN, INF = float("nan"), float("inf")


def _reference(cdf, max_step_warn=0.19):
    """Verbatim pre-fix validate_cdf numeric logic (post-9901)."""
    cdf = np.asarray(cdf, dtype=float)
    problems = []
    if not np.all(np.isfinite(cdf)):
        problems.append(NONFINITE)
    if np.any((cdf < 0) | (cdf > 1)):
        problems.append("CDF values outside [0, 1]")
    if np.any(np.diff(cdf) < -1e-12):
        problems.append("CDF is not non-decreasing")
    steps = np.diff(cdf)
    if len(steps) and steps.max() > max_step_warn:
        problems.append(f"largest CDF step {steps.max():.3f} > {max_step_warn}")
    return problems


def test_task_repro_empty_and_single():
    assert cm.validate_cdf([]) == [LEN0]
    assert cm.validate_cdf([0.5]) == [LEN1]


def test_kill_empty_never_returns_empty_list():
    for empty in ([], (), np.array([]), np.array([], dtype=float)):
        assert cm.validate_cdf(empty) != [], empty
        assert cm.validate_cdf(empty) == [LEN0], empty
    for single in ([0.5], (0.5,), np.array([0.5]), [0], [1]):
        assert cm.validate_cdf(single) == [LEN1], single


def test_short_cdf_other_defects_preserved_after_length():
    assert cm.validate_cdf([2.0]) == [LEN1, "CDF values outside [0, 1]"]
    assert cm.validate_cdf([NAN]) == [LEN1, NONFINITE]
    assert cm.validate_cdf([INF]) == \
        [LEN1, NONFINITE, "CDF values outside [0, 1]"]
    assert cm.validate_cdf(0.5) == [LEN1]  # scalar: reported, not raised


def test_length_at_least_two_bit_identical():
    panels = [
        list(np.linspace(0, 1, 201)), [0.0, 0.1, 0.2], [0.5, 0.5, 0.5],
        [0.0, 1.0], [-0.1, 0.5, 1.2], [0.9, 0.5, 0.9],
        [0.5, 0.5 - 5e-13, 0.6], [0.5, 0.5 - 2e-12, 0.6],
        [0.0, 0.5, 0.55], [0, 1, 1], np.array([0, 0, 1]),
        (0.0, 0.25, 0.5, 1.0), [0.1, NAN, 0.9], [0.1, INF, 0.9],
        [0.0, 0.9, 1.0], list(np.linspace(0, 1, 100)),
        list(np.linspace(0, 1, 11)),
    ]
    rng = np.random.default_rng(10426)
    panels += [np.sort(rng.random(n)) for n in (2, 3, 5, 50, 200, 201)]
    panels += [rng.random(n) for n in (2, 7, 50)]
    for cdf in panels:
        assert cm.validate_cdf(cdf) == _reference(cdf), cdf
        for warn in (0.4, 0.6, 1):
            assert cm.validate_cdf(cdf, max_step_warn=warn) == \
                _reference(cdf, max_step_warn=warn), (cdf, warn)


def test_expected_size_match_and_mismatch():
    lin100 = list(np.linspace(0, 1, 100))
    lin201 = list(np.linspace(0, 1, 201))
    assert cm.validate_cdf(lin201, expected_size=201) == []
    assert cm.validate_cdf(lin100, expected_size=100) == []
    assert cm.validate_cdf(lin100) == []  # unknown size: not flagged
    assert cm.validate_cdf(lin100, expected_size=201) == \
        ["CDF length 100 does not match expected grid size 201"]
    assert cm.validate_cdf(lin201, expected_size=100) == \
        ["CDF length 201 does not match expected grid size 100"]
    assert cm.validate_cdf([], expected_size=201) == \
        [LEN0, "CDF length 0 does not match expected grid size 201"]
    assert cm.validate_cdf([0.1, NAN, 0.9], expected_size=3) == [NONFINITE]
    assert cm.validate_cdf([0.1, NAN, 0.9], expected_size=201) == \
        ["CDF length 3 does not match expected grid size 201", NONFINITE]
    for form in (np.int64(201), 201.0, Decimal("201"), Fraction(201, 1),
                 np.array(201)):
        assert cm.validate_cdf(lin201, expected_size=form) == [], form


def test_bad_expected_size_raises_named_valueerror():
    bad_sizes = [True, False, np.bool_(True), "201", b"201", 201.5,
                 NAN, INF, -5, 0, 1, [201], (201,), 201j, object(),
                 np.array([201]), Decimal("201.5"), Fraction(5, 2)]
    for bad in bad_sizes:
        for cdf in (list(np.linspace(0, 1, 201)), [], [0.5]):
            try:
                cm.validate_cdf(cdf, expected_size=bad)
            except ValueError as exc:
                assert type(exc) is ValueError, type(exc)
                assert "expected_size" in str(exc), str(exc)
            except Exception as exc:  # bare TypeError etc. = failure
                raise AssertionError(
                    f"expected_size={bad!r} raised bare "
                    f"{type(exc).__name__}: {exc}")
            else:
                raise AssertionError(
                    f"expected_size={bad!r} did not raise")


def test_prior_screens_preserved():
    got = cm.validate_cdf([True] * 201)
    assert len(got) == 1 and "boolean" in got[0], got
    assert any("boolean" in p for p in cm.validate_cdf([True]))
    try:
        cm.validate_cdf(["0.1", "0.5", "0.9"])
    except TypeError as exc:
        assert "cdf" in str(exc), str(exc)
    else:
        raise AssertionError("string CDF no longer raises TypeError")
    assert cm.validate_cdf([0.1, NAN, 0.9]) == [NONFINITE]
    assert cm.validate_cdf([0.0, 0.9, 1.0]) == \
        ["largest CDF step 0.900 > 0.19"]


def test_lawful_generated_cdf_still_valid():
    cdf = cm.percentiles_to_cdf({0.05: 90.0, 0.25: 98.0, 0.5: 100.0,
                                 0.75: 102.0, 0.95: 110.0}, 0.0, 200.0)
    assert cm.validate_cdf(cdf) == []
    assert cm.validate_cdf(cdf, expected_size=201) == []
    assert cm.validate_cdf(cdf, expected_size=cm.CDF_SIZE) == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} tests passed")
