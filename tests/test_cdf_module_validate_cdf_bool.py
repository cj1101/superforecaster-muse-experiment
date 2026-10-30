"""
test_cdf_module_validate_cdf_bool.py — SF-BL-10471 verification.

Pre-registered done/kill criteria (TASK.md):
  "Fix reports a named problem for bool-dtype (or non-numeric) CDF
   input before float coercion, with all-True/mixed-bool tests;
   kill if [True]*201 still returns []."

Bug: cdf_module.validate_cdf opened with
np.asarray(cdf, dtype=float), coercing True->1.0 / False->0.0
before every check, so validate_cdf([True]*201) returned [] — an
all-bool panel passed as the constant CDF 1.0. Distinct from
metaculus_client.validate_forecast_payload's all-bool case
(SF-BL-10387), a different function.

Fix (metaculus_bot/cdf_module.py, validate_cdf): a bool /
non-numeric screen runs on the RAW input before coercion — numpy
bool arrays by dtype, other sequences elementwise (np.bool_
included) — and returns "CDF contains boolean values at indices
[...] (a bool is not a number)" and/or "CDF contains non-numeric
values ..." on its own. Numeric input falls through to the
identical pre-existing checks (pinned against a verbatim copy of
the pre-10471, post-SF-BL-9901 logic).

Run:  python3 tests/test_cdf_module_validate_cdf_bool.py
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metaculus_bot"))

import numpy as np  # noqa: E402

import cdf_module as cm  # noqa: E402

BOOL_MARK = "CDF contains boolean values"
NONNUM_MARK = "CDF contains non-numeric values"


def _reference(cdf, max_step_warn=0.19):
    """Verbatim pre-10471 validate_cdf logic (post-SF-BL-9901)."""
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


def test_task_repro_all_true():
    got = cm.validate_cdf([True] * 201)
    assert got != [], "kill criterion: [True]*201 still returns []"
    assert len(got) == 1 and BOOL_MARK in got[0] and "bool" in got[0], got
    assert "200" in got[0], got  # every index named, 0..200


def test_all_bool_and_mixed_bool_flagged():
    panels = [
        [False] * 201, [True], [False],
        [False, True] * 100 + [True],
        [0.0, True, 1.0], [0.0, False, 1.0], (True, False, True),
        np.array([True] * 201), np.array([False, True]),
        np.array([0.0, np.bool_(True), 1.0], dtype=object),
        True, np.bool_(False),
    ]
    for panel in panels:
        got = cm.validate_cdf(panel)
        assert any(BOOL_MARK in p for p in got), (panel, got)


def test_mixed_bool_names_only_bool_indices():
    got = cm.validate_cdf([0.0, True, 1.0])
    assert got == ["CDF contains boolean values at indices [1] "
                   "(a bool is not a number)"], got


def test_non_numeric_flagged_not_coerced_or_raised():
    panels = [
        ["0.0", "0.5", "1.0"], [0.0, "a", 1.0], [0.0, None, 1.0],
        [None], np.array(["0.0", "0.5"]),
        np.array([0.0, None, 1.0], dtype=object), "0.5", None,
    ]
    for panel in panels:
        got = cm.validate_cdf(panel)
        assert any(NONNUM_MARK in p for p in got), (panel, got)
    got = cm.validate_cdf([True, "a", 0.5])
    assert any(BOOL_MARK in p for p in got), got
    assert any(NONNUM_MARK in p for p in got), got


def test_numeric_problems_unchanged():
    panels = [
        list(np.linspace(0, 1, 201)), [0.0, 0.1, 0.2], [0.5, 0.5, 0.5],
        [0.0, 1.0], [-0.1, 0.5, 1.2], [0.9, 0.5, 0.9],
        [0.5, 0.5 - 5e-13, 0.6], [0.5, 0.5 - 2e-12, 0.6],
        [0.0, 0.5, 0.55], [], [0.5], [2.0], [0, 1, 1],
        np.array([0, 0, 1]), (0.0, 0.25, 0.5, 1.0),
        [0.1, float("nan"), 0.9], [0.1, float("inf"), 0.9],
    ]
    rng = np.random.default_rng(10471)
    panels += [np.sort(rng.random(201)), rng.random(50),
               rng.random(50) * 2 - 0.5]
    for cdf in panels:
        assert cm.validate_cdf(cdf) == _reference(cdf), cdf
    for warn in (0.4, 0.6):
        assert cm.validate_cdf([0.0, 0.5, 0.55], max_step_warn=warn) == \
            _reference([0.0, 0.5, 0.55], max_step_warn=warn)


def test_lawful_generated_cdf_still_valid():
    cdf = cm.percentiles_to_cdf({0.05: 90.0, 0.25: 98.0, 0.5: 100.0,
                                 0.75: 102.0, 0.95: 110.0}, 0.0, 200.0)
    assert cm.validate_cdf(cdf) == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} tests passed")
