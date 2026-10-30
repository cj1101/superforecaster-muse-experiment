"""SF-BL-11314 regression tests: predmarkets ece bins TYPE screen.

TASK criterion: predmarkets ece raises ValueError naming bins on any
bool, string or non-integer-float bins, with integer-bin ECE values
bit-identical; kill if bins=True still returns 0.0 or bins='10' still
raises a bare TypeError.

Pre-fix (both copies), on p=[0.9, 0.1], y=[1, 0]:
  bins=True  -> RETURNED 0.0 (bool read as bins=1; control bins=10 -> 0.1)
  bins='10'  -> bare TypeError: can only concatenate str (not "int") to str
  bins=2.5   -> bare TypeError: 'float' object cannot be interpreted as
                an integer
  bins=10.0  -> same bare float TypeError; bins=None -> bare TypeError

Scope: TYPE dimension only, per PREREG.md. All floats are rejected,
including integral floats (10.0 is a non-integer TYPE for a bin count),
following _validate_snapshot_timing's horizon_d screen (SF-BL-11530)
and calibration._validate_n_bins (SF-BL-9722). The bins VALUE domain
(bins < 1, SF-BL-9743) keeps its existing error and is asserted only
to confirm it still fires for lawful-type bins.

Run: python3 -m pytest test_predmarkets_ece_bins_types.py -q
"""
import importlib.util
import struct
import sys
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

COPIES = {
    "canonical": "/home/hatch/workspace/superforecaster/superforecaster/predmarkets.py",
    "out": "/home/hatch/workspace/superforecaster_out/predmarkets/predmarkets.py",
}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(f"pm11314_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MODS = {name: _load(name, path) for name, path in COPIES.items()}
P, Y = [0.9, 0.1], [1, 0]

TASK_BAD = [True, "10", 2.5]
EXTRA_BAD = [False, np.bool_(True), np.bool_(False), "foo", "", b"10",
             0.5, 10.0, -1.5, float("nan"), float("inf"), float("-inf"),
             None, 10 + 0j, [10], (10,), {"bins": 10}, {10},
             Decimal("10"), Fraction(5, 2), Fraction(10, 1),
             np.float64(2.5), np.float64(10.0), np.array(10),
             np.array([10]), object()]


def _prefix_ece(p, y, bins):
    """The pre-fix ece binning computation, verbatim (lawful bins)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    out, n = 0.0, len(p)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1
                               else p <= edges[i + 1])
        if m.sum():
            out += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(out)


def _bits(x):
    return struct.pack("<d", float(x))


# ---- 1. Task fixtures + kill criteria -----------------------------------
@pytest.mark.parametrize("copy", COPIES)
@pytest.mark.parametrize("bad", TASK_BAD)
def test_task_fixtures_raise_named_valueerror(copy, bad):
    with pytest.raises(ValueError, match=r"ece bins must be"):
        MODS[copy].ece(P, Y, bins=bad)


@pytest.mark.parametrize("copy", COPIES)
def test_kill_bins_true_no_longer_returns_zero(copy):
    with pytest.raises(ValueError):
        MODS[copy].ece(P, Y, bins=True)


@pytest.mark.parametrize("copy", COPIES)
def test_kill_bins_str_no_longer_bare_typeerror(copy):
    try:
        MODS[copy].ece(P, Y, bins="10")
    except ValueError:
        pass
    except TypeError as e:
        raise AssertionError(f"bare TypeError survived: {e}")
    else:
        raise AssertionError("bins='10' did not raise")


# ---- 2. Extended type domain --------------------------------------------
@pytest.mark.parametrize("copy", COPIES)
@pytest.mark.parametrize("bad", EXTRA_BAD)
def test_extra_non_integer_types_raise_named_valueerror(copy, bad):
    with pytest.raises(ValueError, match=r"ece bins must be"):
        MODS[copy].ece(P, Y, bins=bad)


@pytest.mark.parametrize("copy", COPIES)
@pytest.mark.parametrize("bad", TASK_BAD + EXTRA_BAD)
def test_error_is_valueerror_not_typeerror_and_tagged(copy, bad):
    try:
        MODS[copy].ece(P, Y, bins=bad)
    except ValueError as e:
        assert "SF-BL-11314" in str(e) and "bins" in str(e)
        assert not isinstance(e, TypeError)
    else:
        raise AssertionError(f"bins={bad!r} did not raise")


# ---- 3. Integer bins bit-identical ---------------------------------------
@pytest.mark.parametrize("copy", COPIES)
def test_task_control_bit_identical(copy):
    r = MODS[copy].ece(P, Y, bins=10)
    assert r == 0.09999999999999999
    assert _bits(r) == _bits(_prefix_ece(P, Y, 10))
    assert MODS[copy].ece(P, Y) == r  # default bins=10


@pytest.mark.parametrize("copy", COPIES)
@pytest.mark.parametrize("bins", [1, 2, 3, 5, 10, 20, 100,
                                  np.int64(10), np.int32(5), np.int64(1)])
@pytest.mark.parametrize("panel", [([0.9, 0.1], [1, 0]),
                                   ([0.0, 1.0], [0, 1]),
                                   ([0.5], [1]),
                                   ([0.13, 0.42, 0.77, 0.99], [0, 1, 1, 0])])
def test_integer_bins_bit_identical(copy, bins, panel):
    p, y = panel
    r = MODS[copy].ece(p, y, bins=bins)
    assert _bits(r) == _bits(_prefix_ece(p, y, int(bins)))


@pytest.mark.parametrize("copy", COPIES)
def test_seeded_random_panel_bit_identical(copy):
    rng = np.random.default_rng(11314)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(float)
    for bins in (1, 4, 10, 25):
        r = MODS[copy].ece(p, y, bins=bins)
        assert _bits(r) == _bits(_prefix_ece(p, y, bins))


# ---- 4. Neighbouring screens preserved -----------------------------------
@pytest.mark.parametrize("copy", COPIES)
@pytest.mark.parametrize("bins", [0, -1, -100])
def test_value_domain_bins_below_1_still_raises(copy, bins):
    with pytest.raises(ValueError, match=r"bins must be >= 1"):
        MODS[copy].ece(P, Y, bins=bins)


@pytest.mark.parametrize("copy", COPIES)
def test_panel_screens_unchanged(copy):
    mod = MODS[copy]
    with pytest.raises(ValueError, match=r"len\(p\)"):
        mod.ece([0.5, 0.6], [1])
    with pytest.raises(ValueError, match=r"empty input"):
        mod.ece([], [])
    with pytest.raises(ValueError, match=r"non-finite"):
        mod.ece([float("nan"), 0.7], [0, 1])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        mod.ece([1.5], [1])
    with pytest.raises(ValueError, match=r"binary 0/1"):
        mod.ece([0.5], [2])


@pytest.mark.parametrize("copy", COPIES)
def test_type_screen_precedes_panel_use(copy):
    # A mistyped bins raises the bins error even on an empty/invalid
    # panel — the screen runs first (snapshot-timing precedent).
    with pytest.raises(ValueError, match=r"ece bins must be"):
        MODS[copy].ece([], [], bins="10")
    with pytest.raises(ValueError, match=r"ece bins must be"):
        MODS[copy].ece([float("nan")], [1], bins=True)


def test_both_copies_agree():
    for bad in TASK_BAD + EXTRA_BAD:
        for name, mod in MODS.items():
            with pytest.raises(ValueError):
                mod.ece(P, Y, bins=bad)
    for bins in (1, 2, 10, np.int64(7)):
        outs = [MODS[n].ece(P, Y, bins=bins) for n in COPIES]
        assert _bits(outs[0]) == _bits(outs[1])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
