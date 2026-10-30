"""
test_cdf_module_nominals_type_validation.py — SF-BL-11315 verification.

Pre-registered done/kill criterion (TASK.md):
  "cdf_to_bin_probs raises ValueError naming nominals on any bool
   or string nominal, with numeric-nominal bin probabilities
   bit-identical; kill if string nominals still return
   [0.25, 0.5, 0.25] or bool nominals still return
   [0.621, 0.253, 0.126]."

Bug class: the nominals TYPE input to cdf_to_bin_probs — distinct
from SF-BL-11272 (this function's bin_edges type only), SF-BL-9926
(bin_edges value domain) and SF-BL-10190 (cdf finiteness — value
domain), none of which fixtures the nominals type domain. Live
pre-fix state (recorded in work/SF-BL-11315/pre_fix_repro.txt, all
three live copies): on cdf [0, 0.5, 1], bin_edges [25, 75], bounds
were built directly from nominals with no type screen, so
nominals=['0','50','100'] returned [0.25, 0.5, 0.25] bit-identical
to the numeric control [0, 50, 100] (string nominals parse through
cdf_at), and nominals=[False, True, 100] returned
[0.621212..., 0.252525..., 0.126262...] (bool nominals become 0/1
grid points).

Fix (cdf_to_bin_probs, all three live copies: repo
superforecaster/metaculus_bot/cdf_module.py, integrated
superforecaster_out/metaculus_bot/cdf_module.py, wave3 original
superforecaster_out/wave3/meta-forecast-types/cdf_module.py): a
dedicated type screen at function entry, BEFORE any float coercion
or other screen — bool or string/bytes nominals (list/tuple
elements, np.bool_/np.str_/np.bytes_ scalars, bool- or
string-dtype ndarrays, object-dtype elements, or a bare
bool/string nominals) raise
  ValueError: cdf_to_bin_probs requires nominals to be real numbers
  (a bool is not a number and a string is not a number); got bool or
  string nominals value(s) [...] at nominals indices [...]
  (SF-BL-11315; ...)
A bool is not a number (fleet convention, cf. validate_cdf
SF-BL-10471 and crps_from_cdf SF-BL-11134 in this module); a
string-dtype nominal is a caller type bug. Numeric nominals fall
through untouched, so numeric-nominal outputs are bit-identical
(expected raw bytes below were captured live from the pre-fix
implementation).

Run:  python3 tests/test_cdf_module_nominals_type_validation.py
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

CDF = [0, 0.5, 1]
NOM = [0, 50, 100]
EDGES = [25, 75]

CONTROL_HEX = ("000000000000d03f000000000000e03f"
               "000000000000d03f")


def _raises_naming_nominals_type(cdf, noms, edges, *needles):
    with pytest.raises(ValueError) as excinfo:
        cm.cdf_to_bin_probs(cdf, noms, edges)
    msg = str(excinfo.value)
    assert "nominals" in msg, msg
    assert "SF-BL-11315" in msg, msg
    for n in needles:
        assert n in msg, f"missing {n!r} in {msg!r}"
    return msg


def test_task_repro_string_nominals_raise_not_identical_to_control():
    # Kill case: pre-fix ['0','50','100'] returned the control vector.
    _raises_naming_nominals_type(CDF, ['0', '50', '100'], EDGES, "string")
    p = cm.cdf_to_bin_probs(CDF, NOM, EDGES)
    assert p.tobytes().hex() == CONTROL_HEX
    assert list(p) == [0.25, 0.5, 0.25]


def test_task_repro_bool_nominals_raise_not_vector():
    # Kill case: pre-fix [False, True, 100] returned
    # [0.621212..., 0.252525..., 0.126262...].
    _raises_naming_nominals_type(CDF, [False, True, 100], EDGES, "bool")


def test_string_nominal_all_forms_raise():
    for noms in (['0', '50', '100'], ['0'], [0, '50', 100],
                 [0, 50, '100'], ('0', '50', '100'), [np.str_('0'), 50, 100],
                 [b'0', 50, 100], np.array(['0', '50', '100']),
                 np.array([b'0', b'50']), np.array(['0', '50', '100'], dtype=object),
                 np.array([0, '50', 100], dtype=object),
                 np.array('0'), '0', b'0', np.str_('0'),
                 [np.array('50'), 50, 100]):
        _raises_naming_nominals_type(CDF, noms, EDGES, "string")


def test_bool_nominal_all_forms_raise():
    for noms in ([False, True, 100], [True, 50, 100], [0, 50, True],
                 [False], [np.bool_(True), 50, 100],
                 np.array([False, True, False]), np.array([True]),
                 np.array(True), np.array([0, 50, 100], dtype=bool),
                 True, False, np.bool_(True),
                 [np.array(True), 50, 100]):
        _raises_naming_nominals_type(CDF, noms, EDGES, "bool")


def test_type_screen_precedence_over_other_bad_inputs():
    # The criterion is unconditional: a bad-typed nominal raises the
    # nominals type error even when another argument is also bad.
    _raises_naming_nominals_type([0.0, float('nan'), 1.0],
                                 ['0', '50', '100'], EDGES)  # NaN cdf too
    _raises_naming_nominals_type([1.0], [True, 50, 100], EDGES)  # degenerate
    _raises_naming_nominals_type(CDF, ['0', '50', '100'],
                                 ['25', '75'])  # bad-typed edges too
    _raises_naming_nominals_type(CDF, [0, '50', 100],
                                 [float('nan')])  # NaN edge too
    _raises_naming_nominals_type(CDF, [False, True, 100], EDGES)


def test_no_bad_type_returns_a_vector():
    bad_noms = [['0', '50', '100'], ['0'], [0, '50', 100],
                [False, True, 100], [True, 50, 100], [0, 50, True],
                [np.bool_(True), 50, 100], [np.str_('0'), 50, 100],
                [b'0', 50, 100], np.array(['0', '50', '100']),
                np.array([False, True, False]), ('0', '50', '100'),
                '0', True]
    for noms in bad_noms:
        try:
            out = cm.cdf_to_bin_probs(CDF, noms, EDGES)
        except ValueError:
            continue
        raise AssertionError(
            f"nominals={noms!r} returned {out!r} instead of raising "
            "(kill criterion)")


def test_numeric_nominals_bit_identical():
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


def test_numeric_nominal_forms_bit_identical():
    from decimal import Decimal
    from fractions import Fraction
    for noms in ([0.0, 50.0, 100.0], (0, 50, 100),
                 np.array([0., 50., 100.]), np.array([0, 50, 100]),
                 np.array([0, 50, 100], dtype=object),
                 [Decimal('0'), Decimal('50'), Decimal('100')],
                 [Fraction(0), Fraction(50), Fraction(100)],
                 [np.int64(0), np.float64(50), np.float32(100)],
                 range(0, 101, 50)):
        p = cm.cdf_to_bin_probs(CDF, noms, EDGES)
        assert p.tobytes().hex() == CONTROL_HEX, noms


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
    # SF-BL-11272: string edge still raises with its tag (not 11315).
    with pytest.raises(ValueError) as excinfo:
        cm.cdf_to_bin_probs(CDF, NOM, ['50'])
    msg = str(excinfo.value)
    assert "SF-BL-11272" in msg and "bin_edges" in msg, msg
    # SF-BL-10276: NaN edge still raises with its tag.
    with pytest.raises(ValueError) as excinfo:
        cm.cdf_to_bin_probs(CDF, NOM, [float('nan')])
    assert "SF-BL-10276" in str(excinfo.value), str(excinfo.value)
    # SF-BL-10190: NaN nominal (value domain) still raises with its
    # tag, not the type tag; NaN cdf likewise.
    with pytest.raises(ValueError) as excinfo:
        cm.cdf_to_bin_probs(CDF, [float('nan'), 50, 100], EDGES)
    msg = str(excinfo.value)
    assert "SF-BL-10190" in msg and "SF-BL-11315" not in msg, msg
    with pytest.raises(ValueError) as excinfo:
        cm.cdf_to_bin_probs([0.0, float('nan'), 1.0], [0, 1, 2], [0.5, 1.5])
    assert "SF-BL-10190" in str(excinfo.value), str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        cm.cdf_to_bin_probs([0.0, 0.5, 1.0], [2, 1, 0], [0.5, 1.5])
    assert "SF-BL-10190" in str(excinfo.value), str(excinfo.value)


def test_client_adapter_raises_and_matches():
    import metaculus_client as mc
    assert mc.cdf_to_bin_probs is cm.cdf_to_bin_probs
    _raises_naming_nominals_type(CDF, ['0', '50', '100'], EDGES)
    _raises_naming_nominals_type(CDF, [False, True, 100], EDGES)
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
