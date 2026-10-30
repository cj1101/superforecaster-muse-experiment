"""
test_cdf_module_bin_edges_nonfinite.py — SF-BL-10276 verification.

Pre-registered done/kill criterion (TASK.md):
  "Fix raises ValueError naming bin_edges if any edge is non-finite,
   with finite-edge outputs bit-identical; kill if a NaN or inf edge
   still returns a probability vector."

Bug class: the non-finite bin_edges input to cdf_to_bin_probs —
distinct from SF-BL-10190 (NaN in the cdf values argument) and from
SF-BL-9926 (unsorted but finite edges). Live pre-fix state (recorded
in work/SF-BL-10276/repro.py + RESULTS.md):
  * wave3 source-of-truth copy (wave3/meta-forecast-types/cdf_module.py)
    screened nothing: the task repro
    cdf_to_bin_probs([0.0, 0.5, 1.0], [0, 1, 2], bin_edges=[NaN])
    returned [nan, nan], and +/-inf edges returned plausible-looking
    distributions ([0.999999, 1e-06] for +inf).
  * Live copies had SF-BL-10190's edges screen, but it ran AFTER the
    cdf/grid screens, so a non-finite edge paired with a non-finite
    CDF or degenerate grid raised an error that did not name
    bin_edges, and no error was tagged SF-BL-10276.

Fix (cdf_to_bin_probs, all three copies): a dedicated screen
immediately after the bin_edges asarray conversion — before any other
validation or interpolation — raises
  ValueError: cdf_to_bin_probs requires finite bin_edges; got
  non-finite bin_edges value(s) [...] (SF-BL-10276; bin_edges
  screening cf. SF-BL-10190; grid validation cf. SF-BL-10109)
The cross-reference tags keep the SF-BL-10190 / SF-BL-10109 suites'
message assertions valid. Finite edges fall through untouched, so
finite-edge outputs are bit-identical (expected raw bytes below were
captured live from the pre-fix implementation).

Run:  python3 tests/test_cdf_module_bin_edges_nonfinite.py
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_cdf_module_bin_probs_validation.py.
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
CDF = [0.0, 0.5, 1.0]
NOM = [0.0, 1.0, 2.0]


def _raises_naming_bin_edges(cdf, noms, edges):
    with pytest.raises(ValueError) as excinfo:
        cm.cdf_to_bin_probs(cdf, noms, edges)
    msg = str(excinfo.value)
    assert "bin_edges" in msg, msg
    assert "finite" in msg, msg
    assert "SF-BL-10276" in msg, msg
    # Cross-refs the earlier suites assert on this function's errors.
    assert "SF-BL-10190" in msg and "SF-BL-10109" in msg, msg


def test_task_repro_nan_edge_raises_naming_bin_edges():
    _raises_naming_bin_edges([0.0, 0.5, 1.0], [0, 1, 2], [NAN])


def test_nonfinite_edge_any_kind_or_position_raises():
    for edges in ([NAN], [INF], [-INF], [0.5, NAN], [NAN, 1.5],
                  [0.5, INF], [0.5, -INF], [NAN, INF], [0.25, 0.75, NAN]):
        _raises_naming_bin_edges(CDF, NOM, edges)


def test_nonfinite_edge_precedence_over_other_bad_inputs():
    # The criterion is unconditional: even when another argument is
    # also corrupt, a non-finite edge must raise the bin_edges error,
    # not the cdf/grid error SF-BL-10190's later screen produced.
    _raises_naming_bin_edges([0.0, NAN, 1.0], NOM, [NAN])      # NaN cdf too
    _raises_naming_bin_edges([1.0], [0], [NAN])                # degenerate
    _raises_naming_bin_edges(CDF, [2, 1, 0], [INF])            # reversed noms
    _raises_naming_bin_edges([0.0, 0.8, 0.2], NOM, [0.5, NAN])  # non-monotone


def test_nonfinite_edge_numpy_and_tuple_forms_raise():
    _raises_naming_bin_edges(CDF, NOM, np.array([0.5, np.nan]))
    _raises_naming_bin_edges(CDF, NOM, (np.inf,))
    _raises_naming_bin_edges(np.array(CDF), np.array(NOM), [NAN])


def test_no_nonfinite_edge_returns_a_vector():
    bad_edges = [[NAN], [INF], [-INF], [0.5, NAN], [NAN, NAN],
                 [1.0, INF, 2.0]]
    for edges in bad_edges:
        try:
            out = cm.cdf_to_bin_probs(CDF, NOM, edges)
        except ValueError:
            continue
        raise AssertionError(
            f"bin_edges={edges} returned {out!r} instead of raising "
            "(kill criterion)")


def test_finite_panels_bit_identical():
    expected = {
        "clean_task": "000000000000d03f000000000000e03f"
                      "000000000000d03f",
        "task_shape_finite_edge": "000000000000e03f000000000000e03f",
        "flat_cdf": "172f21e7fdffdf3fec673c87f6c6b03e"
                    "172f21e7fdffdf3f",
        "two_point": "000000000000e03f000000000000e03f",
        "no_edges": "000000000000f03f",
        "step_cdf": "3007c36df5c6b03e8fa442cefbffdf3f"
                    "8fa442cefbffdf3f3007c36df5c6b03e",
        "outside_edges": "3107c36df5c6b03e91a442cefbffef3f"
                         "3107c36df5c6b03e",
    }
    panels = {
        "clean_task": ([0.0, 0.5, 1.0], [0.0, 1.0, 2.0], [0.5, 1.5]),
        "task_shape_finite_edge": ([0.0, 0.5, 1.0], [0, 1, 2], [1.0]),
        "flat_cdf": ([0.5, 0.5, 0.5], [0.0, 1.0, 2.0], [0.5, 1.5]),
        "two_point": ([0.0, 1.0], [0.0, 10.0], [5.0]),
        "no_edges": ([0.0, 0.5, 1.0], [0.0, 1.0, 2.0], []),
        "step_cdf": ([0.0, 0.0, 1.0, 1.0], [0, 1, 2, 3],
                     [0.5, 1.5, 2.5]),
        "outside_edges": ([0.0, 0.5, 1.0], [0.0, 1.0, 2.0],
                          [-5.0, 99.0]),
    }
    for name, (cdf, noms, edges) in panels.items():
        p = cm.cdf_to_bin_probs(cdf, noms, edges)
        assert p.tobytes().hex() == expected[name], name


def test_scaffold_201pt_panel_bit_identical():
    pct = {0.05: 400.0, 0.25: 470.0, 0.5: 500.0, 0.75: 530.0,
           0.95: 600.0}
    noms = cm.grid_nominals(0.0, 1000.0, None, 201)
    cdf = cm.percentiles_to_cdf(pct, 0.0, 1000.0, None, 201)
    p = cm.cdf_to_bin_probs(cdf, noms, [450.0, 500.0, 550.0])
    assert p.tobytes().hex() == ("f88aaff88aafc83f843aa8833aa8d33f"
                                  "843aa8833aa8d33ff88aaff88aafc83f")
    assert np.all(np.isfinite(p)) and abs(float(p.sum()) - 1.0) <= 1e-9


def test_unsorted_but_finite_edges_still_convert():
    # SF-BL-9926's class (unsorted but finite) is out of scope here:
    # the 10276 screen must not start rejecting finite edges.
    p = cm.cdf_to_bin_probs(CDF, NOM, [1.5, 0.5])
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
