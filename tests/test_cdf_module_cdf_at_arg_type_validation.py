"""
test_cdf_module_cdf_at_arg_type_validation.py — SF-BL-11308.

Pre-registered done/kill criteria (TASK.md):
  "cdf_at raises ValueError naming cdf or nominals on any bool or
   string element of either array, with numeric-grid interpolations
   bit-identical; kill if string nominals still return 0.5 or bool
   cdf/nominals still return 1.0/0.7475."

Bug: cdf_module.cdf_at never type-validated its cdf or nominals
arguments. np.interp / np.asarray(dtype=float) silently parse
strings and read bools as 1.0/0.0 — live pre-fix repros (cdf
[0.0, 0.5, 1.0] over nominals [0, 50, 100], value=50, control 0.5):
nominals=['0', '50', '100'] -> 0.5 bit-identical to the control,
nominals=[False, True, 100] -> 0.7474747474747475, cdf=[False,
True, True] -> 1.0, and a string cdf died in the scaffold copy
with a bare numpy cast TypeError naming nothing (in this repo
copy it was parsed to the control 0.5).

Fix (metaculus_bot/cdf_module.py, cdf_at +
_checked_cdf_at_grid_arg_11308): type-screen the RAW cdf and
nominals arguments at entry, before any float coercion — any
bool / np.bool_ / str / bytes element (and any other non-numeric
element, bool/string-dtype array, or bare bool/string argument)
raises a ValueError naming that argument + SF-BL-11308.
Reject-only: numeric elements fall through to the identical
conversion and np.interp call, so numeric-grid interpolations
stay bit-identical (asserted against a direct np.interp oracle
by float bit pattern). Distinct from SF-BL-11270 (the value
argument's type) and SF-BL-10109 (grid finiteness / ordering —
the value domain, re-verified below).

Run:  python3 tests/test_cdf_module_cdf_at_arg_type_validation.py
      or under pytest.
"""
import os
import struct
import sys
from decimal import Decimal
from fractions import Fraction

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metaculus_bot"))

import numpy as np  # noqa: E402
import cdf_module as cm  # noqa: E402

PASS = []
CDF = [0.0, 0.5, 1.0]
NOM = [0, 50, 100]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _raises_named(fn, argname):
    try:
        fn()
    except ValueError as e:
        msg = str(e)
        assert argname in msg and "SF-BL-11308" in msg, msg
        return msg
    raise AssertionError(f"no ValueError naming {argname}")


def _bits(x):
    return struct.pack("<d", float(x))


BAD_NOMINALS = [
    ['0', '50', '100'], ['0', 50, 100], [0, '50', 100], [0, 50, '100'],
    [False, True, 100], [True, 50, 100], [0, False, 100], [0, 50, True],
    [0, np.bool_(True), 100], [0, np.str_('50'), 100], [0, b'50', 100],
    [0, None, 100], [0, 50 + 0j, 100], [0, [50], 100],
    np.array([False, True, True]), np.array(['0', '50', '100']),
    np.array([0, '50', 100], dtype=object), ('0', '50', '100'),
    '0,50,100',
]
BAD_CDFS = [
    ['0.0', '0.5', '1.0'], ['a', 'b', 'c'], ['0.0', 0.5, 1.0],
    [0.0, '0.5', 1.0], [0.0, 0.5, '1.0'],
    [False, True, True], [True, 0.5, 1.0], [0.0, False, 1.0],
    [0.0, 0.5, True], [0.0, np.bool_(True), 1.0],
    [0.0, np.str_('0.5'), 1.0], [0.0, b'0.5', 1.0], [0.0, None, 1.0],
    np.array([False, True, True]), np.array(['0.0', '0.5', '1.0']),
    np.array([0.0, '0.5', 1.0], dtype=object), (False, True, True),
    '0.0,0.5,1.0',
]


def test_bool_and_string_nominals_each_position():
    for nom in BAD_NOMINALS:
        _raises_named(lambda n=nom: cm.cdf_at(CDF, n, 50), "nominals")
    check("bool/string nominals raise naming nominals", True)


def test_bool_and_string_cdf_each_position():
    for c in BAD_CDFS:
        _raises_named(lambda c=c: cm.cdf_at(c, NOM, 50), "cdf")
    check("bool/string cdf raises naming cdf", True)


def test_kill_panels_do_not_return_prefix_values():
    for c, nom in [(CDF, ['0', '50', '100']),
                   (CDF, [False, True, 100]),
                   ([False, True, True], NOM)]:
        try:
            out = cm.cdf_at(c, nom, 50)
        except ValueError:
            continue
        raise AssertionError(f"kill panel returned {out!r}")
    check("kill panels raise instead of 0.5/0.7475/1.0", True)


def test_string_cdf_is_valueerror_not_bare_typeerror():
    try:
        cm.cdf_at(['0.0', '0.5', '1.0'], NOM, 50)
    except ValueError as e:
        check("string cdf raises ValueError naming cdf",
              "cdf" in str(e))
        return
    except TypeError:
        check("string cdf raises ValueError naming cdf", False,
              "bare TypeError")
        return
    check("string cdf raises ValueError naming cdf", False, "returned")


def test_numeric_panels_bit_identical():
    panels = [
        (CDF, NOM), ([0, 1, 1], [0, 50, 100]),
        (tuple(CDF), tuple(NOM)),
        (np.array(CDF), np.array(NOM, dtype=float)),
        (np.array([0, 1, 1]), np.array(NOM)),
        ([Decimal('0.0'), Fraction(1, 2), Decimal('1.0')],
         [Fraction(0), Decimal('50'), 100]),
        ([np.array(0.0), np.array(0.5), np.array(1.0)],
         [np.array(0), np.array(50), np.array(100)]),
        ([0.0, 0.5, 1.0], range(0, 101, 50)),
    ]
    for c, nom in panels:
        for v in [-10, 0, 1, 25, 49.999, 50, 75, 99.9, 100, 150]:
            want = float(np.interp(
                v, np.asarray(nom, dtype=float),
                np.asarray(c, dtype=float)))
            assert _bits(cm.cdf_at(c, nom, v)) == _bits(want), (c, nom, v)
    cdf201 = cm.percentiles_to_cdf({0.1: 10, 0.5: 50, 0.9: 90}, 0, 100)
    nom201 = cm.grid_nominals(0, 100)
    for v in [-10, 0, 25, 50, 75, 100, 150]:
        assert _bits(cm.cdf_at(cdf201, nom201, v)) == _bits(
            float(np.interp(v, nom201, cdf201)))
    assert cm.cdf_at(CDF, NOM, 50) == 0.5
    assert isinstance(cm.cdf_at(CDF, NOM, 50), float)
    check("numeric panels bit-identical to np.interp oracle", True)


def test_prior_10109_screens_preserved():
    for fn, needle in [
            (lambda: cm.cdf_at(CDF, [100, 50, 0], 50), "SF-BL-10109"),
            (lambda: cm.cdf_at(CDF, NOM, float("nan")), "SF-BL-10109"),
            (lambda: cm.cdf_at([0.0, 0.5], NOM, 50), "SF-BL-10109"),
            (lambda: cm.cdf_at([0.0, float("nan"), 1.0], NOM, 50),
             "SF-BL-10109"),
            (lambda: cm.cdf_at(CDF, [0, float("inf"), 100], 50),
             "SF-BL-10109")]:
        try:
            fn()
        except ValueError as e:
            assert needle in str(e), str(e)
            continue
        raise AssertionError("SF-BL-10109 screen lost")
    check("SF-BL-10109 grid/value-domain screens preserved", True)


def test_metaculus_client_reexport():
    import metaculus_client as mc
    assert mc.cdf_at is cm.cdf_at
    _raises_named(lambda: mc.cdf_at(CDF, ['0', '50', '100'], 50),
                  "nominals")
    assert mc.cdf_at(CDF, NOM, 50) == 0.5
    check("metaculus_client.cdf_at is the same screened function", True)


if __name__ == "__main__":
    test_bool_and_string_nominals_each_position()
    test_bool_and_string_cdf_each_position()
    test_kill_panels_do_not_return_prefix_values()
    test_string_cdf_is_valueerror_not_bare_typeerror()
    test_numeric_panels_bit_identical()
    test_prior_10109_screens_preserved()
    test_metaculus_client_reexport()
    print(f"\n{sum(PASS)}/{len(PASS)} PASS")
    sys.exit(0 if all(PASS) else 1)
