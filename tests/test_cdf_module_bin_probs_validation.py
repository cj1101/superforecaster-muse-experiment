"""
test_cdf_module_bin_probs_validation.py — SF-BL-10190 verification.

Pre-registered done/kill criteria (TASK.md):
  "Fix passes iff cdf_to_bin_probs raises a named ValueError on a
   non-finite or non-monotone CDF and asserts the returned bin
   probabilities are finite and sum to 1 within 1e-9, with clean CDFs
   byte-identical; kill if every caller is shown to validate the CDF
   immediately before conversion."

Bug: cdf_module.cdf_to_bin_probs screened nothing. np.interp propagated
a NaN CDF entry into every bin value and the epsilon-floor /
renormalisation step never detected the result summed to NaN, not 1:
cdf_to_bin_probs([0.0, NaN, 1.0], [0, 1, 2], [0.5, 1.5]) returned
[NaN, NaN, NaN]. A non-monotone CDF's negative bin masses were silently
clipped and renormalised into a plausible distribution (e.g.
[0.0, 0.8, 0.2, 1.0] -> [0.4, 0.1, 0.1, 0.4]). Kill criterion checked
and NOT met: no caller validates the CDF immediately before conversion
(evaluate.py and test_module.py pass percentiles_to_cdf output straight
in; metaculus_client.py only re-exports the function).

Fix (metaculus_bot/cdf_module.py, cdf_to_bin_probs): reject-only
screens before any interpolation — degenerate (<2-point) or mismatched
grids, non-finite cdf / nominals / bin_edges, non-increasing nominals,
and a CDF decrease > 1e-12 (validate_cdf's monotonicity tolerance in
the same module) each raise a ValueError naming cdf_to_bin_probs and
SF-BL-10190; a post-condition re-checks the returned probs are finite
and sum to 1 within 1e-9. The bin computation is untouched, so clean
CDFs are byte-identical: expected raw bytes below were captured live
from the pre-fix implementation (work dir verify_fix.py covers all 29
panels on both module copies; this file pins the canonical copy).

Run:  python3 tests/test_cdf_module_bin_probs_validation.py
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_cdf_module_cell_density_nonfinite.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metaculus_bot"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import cdf_module as cm  # noqa: E402

NAN, INF = float("nan"), float("inf")
NOM = [0.0, 1.0, 2.0]
EDGES = [0.5, 1.5]


def _raises_named(cdf, noms, edges):
    with pytest.raises(ValueError) as excinfo:
        cm.cdf_to_bin_probs(cdf, noms, edges)
    msg = str(excinfo.value)
    assert "cdf_to_bin_probs" in msg and "SF-BL-10190" in msg, msg


def test_task_repro_nan_cdf_raises_named():
    _raises_named([0.0, NAN, 1.0], [0, 1, 2], [0.5, 1.5])


def test_nonfinite_cdf_any_kind_or_position_raises():
    for bad in ([NAN, 0.5, 1.0], [0.0, NAN, 1.0], [0.0, 0.5, NAN],
                [0.0, INF, 1.0], [0.0, -INF, 1.0], [INF, 0.5, 1.0]):
        _raises_named(bad, NOM, EDGES)


def test_nonmonotone_cdf_raises():
    for bad in ([0.0, 0.8, 0.2, 1.0], [0.0, 0.5, 1.0, 0.9],
                [1.0, 0.5, 0.0], [0.0, 0.5, 0.5 - 1e-6, 1.0]):
        noms = [0, 1, 2, 3] if len(bad) == 4 else NOM
        edges = [0.5, 1.5, 2.5] if len(bad) == 4 else EDGES
        _raises_named(bad, noms, edges)


def test_nonfinite_grid_and_edges_raise():
    _raises_named([0.0, 0.5, 1.0], [0.0, NAN, 2.0], EDGES)
    _raises_named([0.0, 0.5, 1.0], [0.0, 1.0, INF], EDGES)
    _raises_named([0.0, 0.5, 1.0], NOM, [0.5, NAN])
    _raises_named([0.0, 0.5, 1.0], NOM, [0.5, INF])


def test_degenerate_mismatched_and_nonincreasing_grids_raise():
    _raises_named([], [], [0.5])
    _raises_named([1.0], [2.0], [0.5])
    _raises_named([0.0, 0.5, 1.0], [0, 1], [0.5])
    _raises_named([0.0, 0.5, 1.0], [2, 1, 0], EDGES)
    _raises_named([0.0, 0.5, 1.0], [0, 1, 1], EDGES)


def test_no_bad_input_returns_probs():
    bads = [([0.0, NAN, 1.0], NOM, EDGES),
            ([0.0, 0.8, 0.2, 1.0], [0, 1, 2, 3], [0.5, 1.5, 2.5]),
            ([0.0, 0.5, 1.0], [0.0, NAN, 2.0], EDGES),
            ([], [], [0.5])]
    for cdf, noms, edges in bads:
        try:
            out = cm.cdf_to_bin_probs(cdf, noms, edges)
        except ValueError:
            continue
        raise AssertionError(f"bad input returned {out!r} instead of "
                             "raising")


def test_clean_panels_byte_identical():
    expected = {
        "clean_task": "000000000000d03f000000000000e03f"
                      "000000000000d03f",
        "flat_cdf": "172f21e7fdffdf3fec673c87f6c6b03e"
                    "172f21e7fdffdf3f",
        "two_point": "000000000000e03f000000000000e03f",
        "no_edges": "000000000000f03f",
    }
    panels = {
        "clean_task": ([0.0, 0.5, 1.0], [0.0, 1.0, 2.0], [0.5, 1.5]),
        "flat_cdf": ([0.5, 0.5, 0.5], [0.0, 1.0, 2.0], [0.5, 1.5]),
        "two_point": ([0.0, 1.0], [0.0, 10.0], [5.0]),
        "no_edges": ([0.0, 0.5, 1.0], [0.0, 1.0, 2.0], []),
    }
    for name, (cdf, noms, edges) in panels.items():
        p = cm.cdf_to_bin_probs(cdf, noms, edges)
        assert p.tobytes().hex() == expected[name], name


def test_clean_outputs_finite_and_sum_to_one():
    pct = {0.05: 400.0, 0.25: 470.0, 0.5: 500.0, 0.75: 530.0,
           0.95: 600.0}
    noms = cm.grid_nominals(0.0, 1000.0, None, 201)
    cdf = cm.percentiles_to_cdf(pct, 0.0, 1000.0, None, 201)
    p = cm.cdf_to_bin_probs(cdf, noms, [450.0, 500.0, 550.0])
    assert len(p) == 4 and np.all(np.isfinite(p))
    assert abs(float(p.sum()) - 1.0) <= 1e-9
    assert p.tobytes().hex() == ("f88aaff88aafc83f843aa8833aa8d33f"
                                  "843aa8833aa8d33ff88aaff88aafc83f")


def test_flat_and_step_cdfs_still_convert():
    # Non-decreasing includes flat: lawful pre-fix, must not raise.
    p = cm.cdf_to_bin_probs([0.0, 0.0, 1.0, 1.0], [0, 1, 2, 3],
                            [0.5, 1.5, 2.5])
    assert np.all(np.isfinite(p)) and abs(float(p.sum()) - 1.0) <= 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed.append(fn.__name__)
            print(f"FAIL  {fn.__name__}: {exc!r}")
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
