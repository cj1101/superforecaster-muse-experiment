"""
test_cdf_module_validate_cdf_cdf_string_type.py — SF-BL-11166.

Pre-registered done/kill criteria (TASK.md):
  "Done when validate_cdf raises TypeError naming cdf on any
   string or object-dtype element, with float-CDF problem lists
   bit-identical; kill if validate_cdf(['0.1', '0.5', '0.9'])
   still returns a problems list instead of raising."

Bug: a string-typed CDF was validated as data — originally
(before SF-BL-10471) parsed through np.asarray(dtype=float) and
step-checked (['largest CDF step 0.400 > 0.19']), after 10471
returned as a non-numeric problems list — never rejected as the
type error a JSON-string-serialised CDF is. Distinct from
SF-BL-10471 (bool cdf, returned problem) and SF-BL-11133
(max_step_warn type, ValueError).

Fix (metaculus_bot/cdf_module.py, validate_cdf): the existing
raw-input element scan also collects string/bytes elements
(incl. string-dtype and object-dtype ndarray elements) and
bare object() elements, plus a bare str/bytes cdf; any of them
raises TypeError naming cdf (SF-BL-11166). This SUPERSEDES
SF-BL-10471 only for string/object elements: bool / None /
other non-numeric panels keep their returned problems, and
numeric CDFs are bit-identical (see
tests/test_cdf_module_validate_cdf_bool.py, whose string
fixtures are superseded by this file).

Run:  python3 tests/test_cdf_module_validate_cdf_cdf_string_type.py
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metaculus_bot"))

import numpy as np  # noqa: E402

import cdf_module as cm  # noqa: E402


def _assert_raises_cdf(panel):
    try:
        got = cm.validate_cdf(panel)
    except TypeError as e:
        assert type(e) is TypeError, type(e)
        assert "cdf" in str(e), e
        assert "SF-BL-11166" in str(e), e
        return
    raise AssertionError(f"returned {got!r} instead of raising for {panel!r}")


def test_kill_string_list_raises_not_returns():
    _assert_raises_cdf(["0.1", "0.5", "0.9"])


def test_string_and_object_fixtures_raise():
    panels = [
        ("0.1", "0.5", "0.9"), [0.1, "0.5", 0.9], ["0.5"],
        [0.1, b"0.5", 0.9], [0.1, np.str_("0.5"), 0.9],
        np.array(["0.1", "0.5", "0.9"]),
        np.array([b"0.1", b"0.5"]),
        np.array(["0.1", "0.5", "0.9"], dtype=object),
        np.array([0.1, "0.5", 0.9], dtype=object),
        "0.1,0.5,0.9", b"0.5",
        [0.1, object(), 0.9],
        np.array([object(), object()], dtype=object),
        (x for x in ["0.1", "0.5"]),
        [True, "0.5", 0.9],
    ]
    for panel in panels:
        _assert_raises_cdf(panel)


def test_bool_and_other_nonnumeric_still_return_problems():
    assert cm.validate_cdf([True] * 201) == [
        "CDF contains boolean values at indices "
        f"{list(range(201))} (a bool is not a number)"]
    assert cm.validate_cdf([0.0, True, 1.0]) == [
        "CDF contains boolean values at indices [1] "
        "(a bool is not a number)"]
    assert cm.validate_cdf([0.1, None, 0.9]) == [
        "CDF contains non-numeric values at indices [1] "
        "(a CDF member must be a number)"]
    # Object-dtype arrays carrying only numeric/bool/None keep
    # their SF-BL-10471 elementwise treatment.
    assert cm.validate_cdf(np.array([0.1, 0.5, 0.9], dtype=object)) == \
        ["largest CDF step 0.400 > 0.19"]
    assert any("boolean" in p for p in cm.validate_cdf(
        np.array([0.0, np.bool_(True), 1.0], dtype=object)))
    assert any("non-numeric" in p for p in cm.validate_cdf(
        np.array([0.0, None, 1.0], dtype=object)))


def test_float_cdf_problem_lists_bit_identical():
    cases = [
        ([0.1, 0.5, 0.9], ["largest CDF step 0.400 > 0.19"]),
        ([0.0, 0.9, 1.0], ["largest CDF step 0.900 > 0.19"]),
        ([0.0, 0.1, 0.2], []),
        ([0.5, 0.5, 0.5], []),
        ([-0.1, 0.5, 1.2], ["CDF values outside [0, 1]",
                            "largest CDF step 0.700 > 0.19"]),
        ([0.9, 0.5, 0.9], ["CDF is not non-decreasing",
                           "largest CDF step 0.400 > 0.19"]),
        ([0.1, float("nan"), 0.9], ["CDF contains non-finite values"]),
        ([], []),
        ([0.5], []),
    ]
    for panel, expected in cases:
        assert cm.validate_cdf(panel) == expected, (panel, cm.validate_cdf(panel))
    cdf = cm.percentiles_to_cdf({0.05: 90.0, 0.25: 98.0, 0.5: 100.0,
                                 0.75: 102.0, 0.95: 110.0}, 0.0, 200.0)
    assert cm.validate_cdf(cdf) == []


def test_max_step_warn_gate_unchanged():
    # SF-BL-11133 screen runs at entry, ahead of the cdf screen.
    for bad in (True, "0.19", None):
        try:
            cm.validate_cdf(["0.1", "0.5"], max_step_warn=bad)
        except ValueError as e:
            assert "max_step_warn" in str(e), e
        else:
            raise AssertionError(f"max_step_warn={bad!r} did not raise")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} tests passed")
