"""
test_cdf_module_bin_edges_type_validation.py — SF-BL-11272 verification.

Pre-registered done/kill criterion (TASK.md):
  "Fix passes iff cdf_to_bin_probs raises ValueError naming bin_edges
   on any bool or string-dtype edge, with numeric-edge bin
   probabilities bit-identical; kill if bin_edges=['50','150'] still
   returns [0.3,0.4,0.3] or bin_edges=[True] still returns a
   probability vector."

Bug class: the bin_edges TYPE input to cdf_to_bin_probs — distinct
from SF-BL-9926 (edge value domain: non-increasing / outside the grid)
and SF-BL-10276 (edge finiteness: NaN/+/-inf). Live pre-fix state
(recorded in work/SF-BL-11272/pre_fix_repro.txt, all three live
copies): on cdf [0.1, 0.5, 0.9] over nominals [0, 100, 200],
np.asarray(bin_edges, dtype=float) coerced with no dtype screen, so
bin_edges=['50','150'] returned [0.3,0.4,0.3] bit-identical to the
numeric control [50,150], and bin_edges=[True] returned [0.104,0.896]
(bool read as an edge at nominal 1.0); a bare string died with a numpy
concatenate ValueError that never named bin_edges.

Fix (cdf_to_bin_probs, all three live copies: repo
superforecaster/metaculus_bot/cdf_module.py, integrated
superforecaster_out/metaculus_bot/cdf_module.py, wave3 original
superforecaster_out/wave3/meta-forecast-types/cdf_module.py): a
dedicated type screen at function entry, BEFORE any float coercion or
other screen — bool or string/bytes edges (list/tuple elements,
np.bool_/np.str_/np.bytes_ scalars, bool- or string-dtype ndarrays,
object-dtype elements, or a bare bool/string bin_edges) raise
  ValueError: cdf_to_bin_probs requires bin_edges to be real numbers
  (a bool is not a number and a string is not a number); got bool or
  string bin_edges value(s) [...] at bin_edges indices [...] (SF-BL-11272; ...)
A bool is not a number (fleet convention, cf. validate_cdf SF-BL-10471
and crps_from_cdf SF-BL-11134 in this module); a string-dtype edge is
a config-typing bug. Numeric edges fall through untouched, so
numeric-edge outputs are bit-identical (expected raw bytes below were
captured live from the pre-fix implementation).

Run:  python3 tests/test_cdf_module_bin_edges_type_validation.py
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_cdf_module_bin_edges_nonfinite.py.
"""
import os
import struct
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metaculus_bot"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import cdf_module as cm  # noqa: E402

CDF = [0.1, 0.5, 0.9]
NOM = [0, 100, 200]

CONTROL_HEX = ("343333333333d33f989999999999d93f"
               "343333333333d33f")


def _raises_naming_bin_edges_type(cdf, noms, edges, *needles):
    with pytest.raises(ValueError) as excinfo:
        cm.cdf_to_bin_probs(cdf, noms, edges)
    msg = str(excinfo.value)
    assert "bin_edges" in msg, msg
    assert "SF-BL-11272" in msg, msg
    for n in needles:
        assert n in msg, f"missing {n!r} in {msg!r}"
    return msg


def test_task_repro_string_edges_raise_not_identical_to_control():
    # Kill case: pre-fix ['50','150'] returned the control vector,
    # bit-identical to [50, 150] -> hex CONTROL_HEX.
    _raises_naming_bin_edges_type(CDF, NOM, ['50', '150'], "string")
    p = cm.cdf_to_bin_probs(CDF, NOM, [50, 150])
    assert p.tobytes().hex() == CONTROL_HEX


def test_task_repro_bool_edge_raises_not_vector():
    # Kill case: pre-fix [True] returned [0.104, 0.896].
    _raises_naming_bin_edges_type(CDF, NOM, [True], "bool")


def test_string_edge_all_forms_raise():
    for edges in (['50', '150'], ['50'], [50, '150'], ['150', 50],
                  ('50', '150'), [np.str_('50')], [b'50'],
                  np.array(['50', '150']), np.array([b'50']),
                  np.array(['50', '150'], dtype=object),
                  np.array('50'), '50', b'50', np.str_('50')):
        _raises_naming_bin_edges_type(CDF, NOM, edges, "string")


def test_bool_edge_all_forms_raise():
    for edges in ([True], [False], [True, 150], [50, True],
                  [True, False], [np.bool_(True)], [np.bool_(False)],
                  np.array([True, False]), np.array([True]),
                  np.array([50, 150], dtype=bool),
                  np.array([50, '150'], dtype=object),
                  np.array(True), True, False, np.bool_(True)):
        _raises_naming_bin_edges_type(CDF, NOM, edges, "bool")


def test_type_screen_precedence_over_other_bad_inputs():
    # The criterion is unconditional: a bad-typed edge raises the
    # bin_edges type error even when another argument is also bad.
    _raises_naming_bin_edges_type([0.0, float('nan'), 1.0], [0, 1, 2],
                                  ['50'])                 # NaN cdf too
    _raises_naming_bin_edges_type([1.0], [0], [True])      # degenerate
    _raises_naming_bin_edges_type(CDF, [200, 100, 0], [False])  # reversed
    _raises_naming_bin_edges_type(CDF, NOM,
                                  [True], )  # 0-d bool still named
    _raises_naming_bin_edges_type(CDF, NOM, ['50', float('nan')],
                                  "string")               # mixed bad


def test_no_bad_type_returns_a_vector():
    bad_edges = [['50', '150'], ['50'], [50, '150'], [True], [False],
                 [True, 150], [50, True], [np.bool_(True)],
                 [np.str_('50')], [b'50'], np.array(['50', '150']),
                 np.array([True, False]), ('50', '150'), '50', True]
    for edges in bad_edges:
        try:
            out = cm.cdf_to_bin_probs(CDF, NOM, edges)
        except ValueError:
            continue
        raise AssertionError(
            f"bin_edges={edges!r} returned {out!r} instead of raising "
            "(kill criterion)")


def test_numeric_edges_bit_identical():
    expected = {
        "task_control": CONTROL_HEX,
        "clean_task": "000000000000d03f000000000000e03f"
                      "000000000000d03f",
        "flat_cdf": "172f21e7fdffdf3fec673c87f6c6b03e"
                    "172f21e7fdffdf3f",
        "two_point": "000000000000e03f000000000000e03f",
        "no_edges": "000000000000f03f",
        "step_cdf": "3007c36df5c6b03e8fa442cefbffdf3f"
                    "8fa442cefbffdf3f3007c36df5c6b03e",
        "outside_edges": "3107c36df5c6b03e91a442cefbffef3f"
                         "3107c36df5c6b03e",
        "int_tuple": CONTROL_HEX,
        "np_int_array": CONTROL_HEX,
        "np_object_int": CONTROL_HEX,
    }
    panels = {
        "task_control": (CDF, NOM, [50, 150]),
        "clean_task": ([0.0, 0.5, 1.0], [0.0, 1.0, 2.0], [0.5, 1.5]),
        "flat_cdf": ([0.5, 0.5, 0.5], [0.0, 1.0, 2.0], [0.5, 1.5]),
        "two_point": ([0.0, 1.0], [0.0, 10.0], [5.0]),
        "no_edges": ([0.0, 0.5, 1.0], [0.0, 1.0, 2.0], []),
        "step_cdf": ([0.0, 0.0, 1.0, 1.0], [0, 1, 2, 3],
                     [0.5, 1.5, 2.5]),
        "outside_edges": ([0.0, 0.5, 1.0], [0.0, 1.0, 2.0],
                          [-5.0, 99.0]),
        "int_tuple": (CDF, NOM, (50, 150)),
        "np_int_array": (CDF, NOM, np.array([50, 150])),
        "np_object_int": (CDF, NOM, np.array([50, 150], dtype=object)),
    }
    for name, (cdf, noms, edges) in panels.items():
        p = cm.cdf_to_bin_probs(cdf, noms, edges)
        assert p.tobytes().hex() == expected[name], name


def test_numeric_edge_forms_bit_identical():
    # int / float / numpy scalar / Decimal / Fraction / range forms of
    # the same edges all produced CONTROL_HEX pre-fix and must still.
    from decimal import Decimal
    from fractions import Fraction
    for edges in ([Decimal('50'), Decimal('150')],
                  [Fraction(50), Fraction(150)],
                  [np.float64(50), np.int64(150)],
                  [np.float32(50), np.float32(150)],
                  range(50, 151, 100)):
        p = cm.cdf_to_bin_probs(CDF, NOM, edges)
        assert p.tobytes().hex() == CONTROL_HEX, edges


def test_scaffold_201pt_panel_bit_identical():
    pct = {0.05: 400.0, 0.25: 470.0, 0.5: 500.0, 0.75: 530.0,
           0.95: 600.0}
    noms = cm.grid_nominals(0.0, 1000.0, None, 201)
    cdf = cm.percentiles_to_cdf(pct, 0.0, 1000.0, None, 201)
    p = cm.cdf_to_bin_probs(cdf, noms, [450.0, 500.0, 550.0])
    assert p.tobytes().hex() == ("f88aaff88aafc83f843aa8833aa8d33f"
                                  "843aa8833aa8d33ff88aaff88aafc83f")
    assert np.all(np.isfinite(p)) and abs(float(p.sum()) - 1.0) <= 1e-9


def test_prior_screens_preserved():
    # SF-BL-10276: NaN edge still raises with its tag (not 11272).
    with pytest.raises(ValueError) as excinfo:
        cm.cdf_to_bin_probs(CDF, NOM, [float('nan')])
    msg = str(excinfo.value)
    assert "SF-BL-10276" in msg and "bin_edges" in msg, msg
    # SF-BL-10190 suites' classes still raise with their tags.
    with pytest.raises(ValueError) as excinfo:
        cm.cdf_to_bin_probs([0.0, float('nan'), 1.0], [0, 1, 2], [0.5, 1.5])
    assert "SF-BL-10190" in str(excinfo.value), str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        cm.cdf_to_bin_probs([0.0, 0.5, 1.0], [2, 1, 0], [0.5, 1.5])
    assert "SF-BL-10190" in str(excinfo.value), str(excinfo.value)
    # SF-BL-9926's class (unsorted but finite edges) is untouched.
    p = cm.cdf_to_bin_probs([0.0, 0.5, 1.0], [0.0, 1.0, 2.0], [1.5, 0.5])
    assert np.all(np.isfinite(p)) and abs(float(p.sum()) - 1.0) <= 1e-9


def test_client_adapter_raises_and_matches():
    import metaculus_client as mc
    assert mc.cdf_to_bin_probs is cm.cdf_to_bin_probs
    _raises_naming_bin_edges_type(CDF, NOM, ['50', '150'])
    _raises_naming_bin_edges_type(CDF, NOM, [True])
    assert mc.cdf_to_bin_probs(CDF, NOM, [50, 150]).tobytes().hex() \
        == CONTROL_HEX


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
