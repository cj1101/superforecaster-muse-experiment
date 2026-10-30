"""
test_cdf_module_validate_cdf_nonfinite.py — SF-BL-9901 verification.

Pre-registered done/kill criteria (TASK.md):
  "validate_cdf reports a non-finite-values problem for any NaN/inf
   CDF entry; kill if the problems list for any finite CDF changes."

Bug: cdf_module.validate_cdf checked (cdf<0)|(cdf>1),
diff < -1e-12 and max step — all NaN-blind — so
validate_cdf([0.1, NaN, 0.9]) returned [] and a corrupt pooled CDF
was certified valid. +/-inf only tripped the other checks
incidentally, never naming non-finiteness.

Fix (metaculus_bot/cdf_module.py, validate_cdf): a non-finite screen
runs FIRST — `if not np.all(np.isfinite(cdf))` appends
"CDF contains non-finite values" — and the pre-existing checks are
untouched, so every finite CDF's problems list is identical to
pre-fix (pinned here against a verbatim copy of the pre-fix logic).

Note: distribution_validity.validate_cdf_values is a DIFFERENT
function and is out of scope for this task.

Run:  python3 tests/test_cdf_module_validate_cdf_nonfinite.py
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metaculus_bot"))

import numpy as np  # noqa: E402

import cdf_module as cm  # noqa: E402

NAN, INF = float("nan"), float("inf")
NONFINITE = "CDF contains non-finite values"


def _reference(cdf, max_step_warn=0.19):
    """Verbatim pre-fix validate_cdf logic."""
    cdf = np.asarray(cdf, dtype=float)
    problems = []
    if np.any((cdf < 0) | (cdf > 1)):
        problems.append("CDF values outside [0, 1]")
    if np.any(np.diff(cdf) < -1e-12):
        problems.append("CDF is not non-decreasing")
    steps = np.diff(cdf)
    if len(steps) and steps.max() > max_step_warn:
        problems.append(f"largest CDF step {steps.max():.3f} > {max_step_warn}")
    return problems


def test_task_repro():
    assert cm.validate_cdf([0.1, NAN, 0.9]) == [NONFINITE]


def test_any_nonfinite_entry_flagged():
    base = np.linspace(0, 1, 201)
    cases = [[0.1, INF, 0.9], [0.1, -INF, 0.9], [NAN], [INF], [-INF],
             [NAN, NAN, NAN], [0.0, 0.5, NAN], [NAN, 0.5, 0.9],
             [0.0, NAN, INF, 1.0],
             [0.1, np.float64("nan"), 0.9]]
    for idx in (0, 100, 200):
        c = base.copy(); c[idx] = NAN; cases.append(c)
    c = base.copy(); c[57] = INF; cases.append(c)
    for cdf in cases:
        got = cm.validate_cdf(cdf)
        assert got[:1] == [NONFINITE], (cdf, got)
        assert got == [NONFINITE] + _reference(cdf), (cdf, got)


def test_finite_problems_unchanged():
    panels = [
        list(np.linspace(0, 1, 201)), [0.0, 0.1, 0.2], [0.5, 0.5, 0.5],
        [0.0, 1.0], [-0.1, 0.5, 1.2], [0.9, 0.5, 0.9],
        [0.5, 0.5 - 5e-13, 0.6], [0.5, 0.5 - 2e-12, 0.6],
        [0.0, 0.5, 0.55], [], [0.5], [2.0],
        np.array([0, 0, 1]), (0.0, 0.25, 0.5, 1.0),
    ]
    rng = np.random.default_rng(9901)
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
