"""SF-BL-11495 regression tests: predmarkets rule_shrink_fixed s TYPE screen.

TASK criterion: rule_shrink_fixed raises a named ValueError for a bool or
non-numeric s, fixtured at '0.5', None and True, with numeric-s outputs
bit-identical ([0.7, 0.3] at s=0.5 on [0.9, 0.1]).

Pre-fix (both copies): s='0.5' -> bare UFuncTypeError, s=None -> bare
TypeError, s=True -> [0.9, 0.1] silently unchanged (no shrinkage),
s=False -> [0.5, 0.5], one-element list/tuple broadcast silently,
complex s -> complex "probabilities", Decimal -> bare TypeError.

Scope: TYPE dimension only. The s VALUE domain (2.0 / -1.0 / NaN) is
SF-BL-11375 and is deliberately not asserted here.

Run: python3 -m pytest test_predmarkets_shrink_fixed_s_types.py -q
"""
import importlib.util
import struct
import sys
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

COPIES = {
    "canonical": "/home/hatch/workspace/superforecaster/superforecaster/predmarkets.py",
    "out": "/home/hatch/workspace/superforecaster_out/predmarkets/predmarkets.py",
}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(f"pm11495_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MODS = {name: _load(name, path) for name, path in COPIES.items()}
P = np.array([0.9, 0.1])

TASK_BAD = ["0.5", None, True]
EXTRA_BAD = [False, np.bool_(True), np.bool_(False), "foo", "", b"0.5",
             [0.5], (0.5,), {"s": 0.5}, {0.5}, 0.5 + 0j, Decimal("0.5"),
             np.array(0.5), np.array([0.5]), np.array(True), object()]


def _prefix(p, s):
    """The pre-fix computation, verbatim."""
    return 0.5 + s * (np.asarray(p, dtype=float) - 0.5)


def _bytes(arr):
    a = np.asarray(arr, dtype=float)
    return struct.pack(f"<{a.size}d", *a.ravel().tolist())


@pytest.mark.parametrize("copy", COPIES)
@pytest.mark.parametrize("bad", TASK_BAD)
def test_task_fixtures_raise_named_valueerror(copy, bad):
    with pytest.raises(ValueError, match=r"rule_shrink_fixed s must be"):
        MODS[copy].rule_shrink_fixed(P, None, None, {"s": bad})


@pytest.mark.parametrize("copy", COPIES)
@pytest.mark.parametrize("bad", EXTRA_BAD)
def test_extra_non_real_types_raise_named_valueerror(copy, bad):
    with pytest.raises(ValueError, match=r"rule_shrink_fixed s must be"):
        MODS[copy].rule_shrink_fixed(P, None, None, {"s": bad})


@pytest.mark.parametrize("copy", COPIES)
@pytest.mark.parametrize("bad", TASK_BAD + EXTRA_BAD)
def test_error_is_valueerror_not_typeerror(copy, bad):
    try:
        MODS[copy].rule_shrink_fixed(P, None, None, {"s": bad})
    except ValueError as e:
        assert "SF-BL-11495" in str(e)
        assert not isinstance(e, TypeError)
    else:
        raise AssertionError(f"s={bad!r} did not raise")


@pytest.mark.parametrize("copy", COPIES)
def test_task_control_bit_identical(copy):
    r = MODS[copy].rule_shrink_fixed(P, None, None, {"s": 0.5})
    assert list(r) == [0.7, 0.3]
    assert _bytes(r) == _bytes(_prefix(P, 0.5))


@pytest.mark.parametrize("copy", COPIES)
@pytest.mark.parametrize("s", [0.0, 0.5, 0.7, 0.925, 1.0, 1.15, 0, 1,
                               np.float64(0.5), np.float32(0.5),
                               np.int64(1), Fraction(1, 2)])
@pytest.mark.parametrize("panel", [[0.9, 0.1], [0.001, 0.999], [0.5],
                                    [0.13, 0.42, 0.77, 0.99]])
def test_numeric_s_bit_identical(copy, s, panel):
    p = np.array(panel)
    r = MODS[copy].rule_shrink_fixed(p, None, None, {"s": s})
    assert _bytes(r) == _bytes(_prefix(p, s))


@pytest.mark.parametrize("copy", COPIES)
def test_fit_grid_s_values_all_lawful(copy):
    for s in np.arange(0.70, 1.151, 0.025):  # fit_rules grid, float(s)
        r = MODS[copy].rule_shrink_fixed(P, None, None, {"s": float(s)})
        assert _bytes(r) == _bytes(_prefix(P, float(s)))


@pytest.mark.parametrize("copy", COPIES)
def test_seeded_random_panel_bit_identical(copy):
    rng = np.random.default_rng(11495)
    p = rng.random(500)
    for s in (0.5, 0.83, 1.0):
        r = MODS[copy].rule_shrink_fixed(p, None, None, {"s": s})
        assert _bytes(r) == _bytes(_prefix(p, s))


@pytest.mark.parametrize("copy", COPIES)
def test_dispatch_and_apply_rule_paths(copy):
    mod = MODS[copy]
    fn = mod.RULE_FUNCS["shrink_fixed"]
    with pytest.raises(ValueError, match=r"rule_shrink_fixed s must be"):
        fn(P, None, None, {"s": "0.5"})
    df = pd.DataFrame({"price": [0.9, 0.1], "days_to_expiry": [7.0, 7.0],
                       "log10_volume": [5.0, 6.0]})
    with pytest.raises(ValueError, match=r"rule_shrink_fixed s must be"):
        mod.apply_rule("shrink_fixed", df, {"s": True}, 5.0, 1.0)
    ok = mod.apply_rule("shrink_fixed", df, {"s": 0.5}, 5.0, 1.0)
    assert list(ok) == [0.7, 0.3]


def test_both_copies_agree():
    for bad in TASK_BAD + EXTRA_BAD:
        for name, mod in MODS.items():
            with pytest.raises(ValueError):
                mod.rule_shrink_fixed(P, None, None, {"s": bad})
    for s in (0.5, 0.925, 1):
        outs = [MODS[n].rule_shrink_fixed(P, None, None, {"s": s})
                for n in COPIES]
        assert _bytes(outs[0]) == _bytes(outs[1])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
