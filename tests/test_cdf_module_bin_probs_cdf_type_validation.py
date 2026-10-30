"""
test_cdf_module_bin_probs_cdf_type_validation.py — SF-BL-11351 verification.

Pre-registered done/kill criterion (TASK.md):
  "cdf_to_bin_probs raises ValueError naming cdf on any bool or
   string cdf element, with numeric-cdf bin probabilities
   bit-identical; kill if string cdf still returns [0.25, 0.5, 0.25]
   or a bool cdf still returns [0.5, 0.5, 1e-06]."

Bug class: the cdf TYPE input to cdf_to_bin_probs — distinct from
SF-BL-11315 (this function's nominals type), SF-BL-11272 (its
bin_edges type), SF-BL-11308 (the cdf type at the cdf_at entry
point, a different function) and SF-BL-10190 (cdf finiteness —
value domain), none of which fixtures the cdf type domain at this
entry point. Live pre-fix state (recorded in
work/SF-BL-11351/pre_fix_repro.txt): on nominals [0, 50, 100],
bin_edges [25, 75], cdf=['0','0.5','1'] returned [0.25, 0.5, 0.25]
bit-identical to the numeric control [0, 0.5, 1], and
cdf=[False, True, True] returned
[0.49999950000050003, 0.49999950000050003, 9.99999000001e-07]
(the task's [0.5, 0.5, 1e-06]).

Fix (cdf_to_bin_probs, all three live copies: repo
superforecaster/metaculus_bot/cdf_module.py, integrated
superforecaster_out/metaculus_bot/cdf_module.py, wave3 original
superforecaster_out/wave3/meta-forecast-types/cdf_module.py): a
dedicated type screen at function entry, BEFORE any float coercion
or other screen — bool or string/bytes cdf elements (list/tuple
elements, np.bool_/np.str_/np.bytes_ scalars, bool- or
string-dtype ndarrays, object-dtype elements, or a bare
bool/string cdf) raise
  ValueError: cdf_to_bin_probs requires cdf to be real numbers
  (a bool is not a number and a string is not a number); got bool
  or string cdf value(s) [...] at cdf indices [...]
  (SF-BL-11351; ...)
A bool is not a number (fleet convention, cf. validate_cdf
SF-BL-10471 and crps_from_cdf SF-BL-11134 in this module); a
string cdf element is a caller type bug. Numeric cdf values fall
through untouched, so numeric-cdf outputs are bit-identical
(expected raw bytes below were captured live from the pre-fix
implementation).

Run:  python3 tests/test_cdf_module_bin_probs_cdf_type_validation.py
No pytest dependency beyond the repo convention: test_* functions
with assert + __main__ runner.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metaculus_bot"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import cdf_module as cm  # noqa: E402

NOM = [0, 50, 100]
EDGES = [25, 75]
CDF = [0, 0.5, 1]

CONTROL_HEX = ("000000000000d03f000000000000e03f"
               "000000000000d03f")


def _raises_naming_cdf_type(cdf, noms, edges, *needles):
    with pytest.raises(ValueError) as excinfo:
        cm.cdf_to_bin_probs(cdf, noms, edges)
    msg = str(excinfo.value)
    assert "cdf" in msg, msg
    assert "SF-BL-11351" in msg, msg
    for n in needles:
        assert n in msg, f"missing {n!r} in {msg!r}"
    return msg


def test_task_repro_string_cdf_raise_not_identical_to_control():
    # Kill case: pre-fix ['0','0.5','1'] returned the control vector.
    _raises_naming_cdf_type(['0', '0.5', '1'], NOM, EDGES, "string")
    p = cm.cdf_to_bin_probs(CDF, NOM, EDGES)
    assert p.tobytes().hex() == CONTROL_HEX
    assert list(p) == [0.25, 0.5, 0.25]


def test_task_repro_bool_cdf_raise_not_vector():
    # Kill case: pre-fix [False, True, True] returned
    # [0.49999950000050003, 0.49999950000050003, 9.99999e-07].
    _raises_naming_cdf_type([False, True, True], NOM, EDGES, "bool")


def test_string_cdf_all_forms_raise():
    for cdf in (['0', '0.5', '1'], ['0'], [0, '0.5', 1],
                [0, 0.5, '1'], ('0', '0.5', '1'), [np.str_('0'), 0.5, 1],
                [b'0', 0.5, 1], np.array(['0', '0.5', '1']),
                np.array([b'0', b'0.5']),
                np.array(['0', '0.5', '1'], dtype=object),
                np.array([0, '0.5', 1], dtype=object),
                np.array('0'), '0', b'0', np.str_('0'),
                [np.array('0.5'), 0.5, 1]):
        _raises_naming_cdf_type(cdf, NOM, EDGES, "string")


def test_bool_cdf_all_forms_raise():
    for cdf in ([False, True, True], [True, 0.5, 1], [0, 0.5, True],
                [False], [np.bool_(True), 0.5, 1],
                np.array([False, True, True]), np.array([True]),
                np.array(True), np.array([0, 0.5, 1], dtype=bool),
                True, False, np.bool_(True),
                [np.array(True), 0.5, 1]):
        _raises_naming_cdf_type(cdf, NOM, EDGES, "bool")


def test_type_screen_precedence_over_other_bad_inputs():
    # The criterion is unconditional: a bad-typed cdf raises the
    # cdf type error even when another argument is also bad.
    _raises_naming_cdf_type(['0', '0.5', '1'],
                            [float('nan'), 50, 100], EDGES)  # NaN nominals too
    _raises_naming_cdf_type([True, 0.5, 1], [0], EDGES)  # degenerate grid
    _raises_naming_cdf_type(['0', '0.5', '1'], NOM,
                            ['25', '75'])  # bad-typed edges too
    _raises_naming_cdf_type([0, '0.5', 1], NOM,
                            [float('nan')])  # NaN edge too
    _raises_naming_cdf_type([False, True, True],
                            ['0', '50', '100'], EDGES)  # bad-typed nominals too


def test_no_bad_type_returns_a_vector():
    bad_cdfs = [['0', '0.5', '1'], ['0'], [0, '0.5', 1],
                [False, True, True], [True, 0.5, 1], [0, 0.5, True],
                [np.bool_(True), 0.5, 1], [np.str_('0'), 0.5, 1],
                [b'0', 0.5, 1], np.array(['0', '0.5', '1']),
                np.array([False, True, True]), ('0', '0.5', '1'),
                '0', True]
    for cdf in bad_cdfs:
        try:
            out = cm.cdf_to_bin_probs(cdf, NOM, EDGES)
        except ValueError:
            continue
        raise AssertionError(
            f"cdf={cdf!r} returned {out!r} instead of raising "
            "(kill criterion)")


def test_numeric_cdf_bit_identical():
    expected = {
        "task_control": CONTROL_HEX,
        "task_panel2": "343333333333d33f989999999999d93f"
                       "343333333333d33f",
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
        "task_control": (CDF, NOM, EDGES),
        "task_panel2": ([0.1, 0.5, 0.9], [0, 100, 200], [50, 150]),
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


def test_numeric_cdf_forms_bit_identical():
    from decimal import Decimal
    from fractions import Fraction
    for cdf in ([0.0, 0.5, 1.0], (0, 0.5, 1),
                np.array([0., 0.5, 1.]),
                np.array([0, 0.5, 1], dtype=object),
                [Decimal('0'), Decimal('0.5'), Decimal('1')],
                [Fraction(0), Fraction(1, 2), Fraction(1)],
                [np.int64(0), np.float64(0.5), np.float32(1)]):
        p = cm.cdf_to_bin_probs(cdf, NOM, EDGES)
        assert p.tobytes().hex() == CONTROL_HEX, cdf
    # An int-dtype cdf array is numeric too (a different panel).
    p = cm.cdf_to_bin_probs(np.array([0, 1, 1]), NOM, EDGES)
    assert p.tobytes().hex() == ("192f21e7fdffdf3f192f21e7fdffdf3f"
                                  "ed673c87f6c6b03e")


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
    # SF-BL-11315: string nominal still raises with its tag (not 11351).
    with pytest.raises(ValueError) as excinfo:
        cm.cdf_to_bin_probs(CDF, ['0', '50', '100'], EDGES)
    msg = str(excinfo.value)
    assert "SF-BL-11315" in msg and "nominals" in msg, msg
    # SF-BL-11272: string edge still raises with its tag (not 11351).
    with pytest.raises(ValueError) as excinfo:
        cm.cdf_to_bin_probs(CDF, NOM, ['50'])
    msg = str(excinfo.value)
    assert "SF-BL-11272" in msg and "bin_edges" in msg, msg
    # SF-BL-10276: NaN edge still raises with its tag.
    with pytest.raises(ValueError) as excinfo:
        cm.cdf_to_bin_probs(CDF, NOM, [float('nan')])
    assert "SF-BL-10276" in str(excinfo.value), str(excinfo.value)
    # SF-BL-10190: NaN cdf (value domain) still raises with its
    # tag, not the type tag; NaN nominal likewise.
    with pytest.raises(ValueError) as excinfo:
        cm.cdf_to_bin_probs([0.0, float('nan'), 1.0], [0, 1, 2], [0.5, 1.5])
    msg = str(excinfo.value)
    assert "SF-BL-10190" in msg and "SF-BL-11351" not in msg, msg
    with pytest.raises(ValueError) as excinfo:
        cm.cdf_to_bin_probs(CDF, [float('nan'), 50, 100], EDGES)
    assert "SF-BL-10190" in str(excinfo.value), str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        cm.cdf_to_bin_probs([0.0, 0.5, 1.0], [2, 1, 0], [0.5, 1.5])
    assert "SF-BL-10190" in str(excinfo.value), str(excinfo.value)


def test_client_adapter_raises_and_matches():
    import metaculus_client as mc
    assert mc.cdf_to_bin_probs is cm.cdf_to_bin_probs
    _raises_naming_cdf_type(['0', '0.5', '1'], NOM, EDGES)
    _raises_naming_cdf_type([False, True, True], NOM, EDGES)
    assert mc.cdf_to_bin_probs(CDF, NOM, EDGES).tobytes().hex() \
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
