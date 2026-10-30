"""
test_cdf_module_cdf_at_grid_validation.py — SF-BL-10109 verification tests.

Pre-registered done/kill criteria (TASK.md):
  "Fix raises a named ValueError on non-increasing/mismatched nominals
   or non-finite cdf/value and ordered-grid results stay bit-identical;
   kill if a reversed grid or NaN still returns a number."

Bug: cdf_module.cdf_at was a bare np.interp wrapper. np.interp assumes
xp (nominals) is increasing without checking, so a reversed grid
silently returned the wrong endpoint value — live pre-fix repro:
cdf_at([0.2, 0.5, 0.9], nominals=[2, 1, 0], value=1.0) -> 0.9 — and
non-finite inputs propagated silently (value=NaN -> NaN, value=+/-inf
pinned to an endpoint, NaN/+/-inf cdf or nominal entries returning
NaN/inf/plausible numbers). Mismatched lengths raised only numpy's
bare "fp and xp are not of the same length" ValueError, which names
neither the grid nor the task. cdf_at feeds cdf_to_bin_probs, so the
corruption reached multiple-choice probabilities (reversed nominals
returned [0.8999991, 1e-06, 0.0999999] pre-fix).

Fix (metaculus_bot/cdf_module.py, cdf_at): before interpolating, raise
a ValueError naming the problem and SF-BL-10109 on (in order) a
cdf/nominals size mismatch, an empty grid, non-finite nominals,
non-finite cdf values, a non-finite/non-scalar query value, or
nominals that are not strictly increasing. Lawful grids fall through
to the identical np.interp call: expected float bit patterns below
were recorded from the pre-fix implementation in the live repro
(work/SF-BL-10109/repro.py), so bit-identity is asserted, not assumed.

Scope note: a single-point grid stays lawful — constant interpolation
off one point is well-defined in np.interp, was lawful pre-fix, and
the prereg names no degenerate-grid criterion (contrast SF-BL-10269,
where <2 points scored a perfect CRPS 0.0).

Run:  python3 tests/test_cdf_module_cdf_at_grid_validation.py
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as tests/test_cdf_module_nonfinite_percentiles.py.
"""
import os
import struct
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metaculus_bot"))

import numpy as np  # noqa: E402
import cdf_module as cm  # noqa: E402

PASS = []
NAN = float("nan")
INF = float("inf")
CDF = [0.2, 0.5, 0.9]
NOM = [0, 1, 2]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _raises_named(fn, *needles):
    try:
        fn()
    except ValueError as e:
        msg = str(e)
        assert "SF-BL-10109" in msg, f"message missing task id: {msg!r}"
        for n in needles:
            assert n in msg, f"message {msg!r} missing {n!r}"
        return msg
    raise AssertionError("no ValueError raised — a number may have leaked")


def _bits(v):
    return struct.pack("<d", float(v)).hex()


# ---- 1. Task repros ---------------------------------------------------------

def test_task_repro_reversed_grid_raises():
    # Pre-fix returned 0.9 (the wrong endpoint) with no error.
    _raises_named(lambda: cm.cdf_at(CDF, [2, 1, 0], 1.0),
                  "strictly increasing")


def test_task_repro_nan_value_raises():
    # Pre-fix returned NaN with no error.
    _raises_named(lambda: cm.cdf_at(CDF, NOM, NAN), "finite", "value")


# ---- 2. Grid order ----------------------------------------------------------

def test_unsorted_and_duplicate_nominals_raise():
    for nom in ([0, 2, 1], [2, 1, 0], [0, 1, 1], [1, 1, 2], [2, 0, 1],
                [0.0, -1.0, 2.0]):
        _raises_named(lambda n=nom: cm.cdf_at(CDF, n, 1.0),
                      "strictly increasing")
    _raises_named(lambda: cm.cdf_at([0.2, 0.9], [1, 0], 0.5),
                  "strictly increasing")
    _raises_named(lambda: cm.cdf_at([0.0, 0.5, 1.0], [10, 0, -10], 0.0),
                  "strictly increasing")


def test_no_bad_grid_returns_a_number_kill_sweep():
    # Kill criterion: no reversed/unsorted/duplicate/NaN input, at any
    # query value, may return a number.
    bad_grids = [
        (CDF, [2, 1, 0]), (CDF, [0, 2, 1]), (CDF, [0, 1, 1]),
        ([0.2, NAN, 0.9], NOM), (CDF, [0, NAN, 2]),
        ([INF, 0.5, 0.9], NOM), (CDF, [0, 1, INF]),
    ]
    for c, n in bad_grids:
        for v in (-1.0, 0.0, 0.5, 1.0, 2.0, 1e6, NAN, INF):
            try:
                out = cm.cdf_at(c, n, v)
            except ValueError:
                continue
            raise AssertionError(
                f"cdf_at({c}, {n}, {v}) returned {out!r} (kill criterion)")


# ---- 3. Mismatch / empty -----------------------------------------------------

def test_mismatched_lengths_raise_named():
    for c, n in (([0.2, 0.5], NOM), (CDF, [0, 1]), ([], NOM),
                 (CDF, []), ([0.5], NOM), (CDF, [1.0])):
        msg = _raises_named(lambda c=c, n=n: cm.cdf_at(c, n, 1.0),
                            "mismatch")
        assert "same number of grid points" in msg, msg
        assert "fp and xp" not in msg, msg  # not numpy's bare message


def test_empty_grid_raises_named():
    _raises_named(lambda: cm.cdf_at([], [], 1.0), "empty")


# ---- 4. Finiteness -----------------------------------------------------------

def test_nonfinite_value_raises():
    for bad in (NAN, INF, -INF):
        _raises_named(lambda b=bad: cm.cdf_at(CDF, NOM, b),
                      "finite", "value")


def test_nonfinite_cdf_raises_any_position():
    for bad in (NAN, INF, -INF):
        for c in ([bad, 0.5, 0.9], [0.2, bad, 0.9], [0.2, 0.5, bad]):
            _raises_named(lambda c=c: cm.cdf_at(c, NOM, 1.0),
                          "finite", "cdf")


def test_nonfinite_nominals_raise_any_position():
    for bad in (NAN, INF, -INF):
        for n in ([bad, 1, 2], [0, bad, 2], [0, 1, bad]):
            _raises_named(lambda n=n: cm.cdf_at(CDF, n, 1.0),
                          "finite", "nominal")


# ---- 5. Ordered grids bit-identical (bit patterns from pre-fix repro) -------

# (name, cdf, nominals, value, pre-fix float bits hex)
ORDERED = [
    ("node v=1.0", CDF, NOM, 1.0, "000000000000e03f"),
    ("interp v=0.5", CDF, NOM, 0.5, "666666666666d63f"),
    ("interp v=1.5", CDF, NOM, 1.5, "666666666666e63f"),
    ("below grid v=-5", CDF, NOM, -5.0, "9a9999999999c93f"),
    ("above grid v=5", CDF, NOM, 5.0, "cdccccccccccec3f"),
    ("at left edge v=0", CDF, NOM, 0.0, "9a9999999999c93f"),
    ("at right edge v=2", CDF, NOM, 2.0, "cdccccccccccec3f"),
    ("negative range v=3.3", [0.0, 0.5, 1.0], [-10, 0, 10], 3.3,
     "48e17a14ae47e53f"),
    ("non-uniform grid v=2.0", [0.1, 0.4, 0.95], [0, 0.3, 10], 2.0,
     "b0b7d4e9e1c4df3f"),
    ("duplicate cdf values v=1.37", [0.5, 0.5, 0.5], NOM, 1.37,
     "000000000000e03f"),
    ("single-point grid v=99", [0.5], [1.0], 99.0, "000000000000e03f"),
    ("two-point grid v=2.5", [0.0, 1.0], [0, 10], 2.5,
     "000000000000d03f"),
]


def test_ordered_grids_bit_identical():
    for name, c, n, v, want_bits in ORDERED:
        out = cm.cdf_at(c, n, v)
        assert _bits(out) == want_bits, f"{name}: {_bits(out)} != {want_bits}"
        assert np.isfinite(out), name


def test_ordered_201pt_scaffold_grid_bit_identical():
    nominals = cm.grid_nominals(0, 30)
    cdf = cm.percentiles_to_cdf({0.1: 5, 0.5: 10, 0.9: 20}, 0, 30)
    want = {0.0: "9a9999999999b93f", 7.5: "343333333333d33f",
            15.0: "666666666666e63f", 29.9: "cdccccccccccec3f",
            30.0: "cdccccccccccec3f"}  # pre-fix bits
    for v, want_bits in want.items():
        assert _bits(cm.cdf_at(cdf, nominals, v)) == want_bits, v


def test_array_and_tuple_inputs_bit_identical():
    assert cm.cdf_at(np.array(CDF), np.array(NOM), np.float64(0.5)) == 0.35
    assert cm.cdf_at(tuple(CDF), tuple(NOM), 1) == 0.5
    assert isinstance(cm.cdf_at(CDF, NOM, 1.0), float)


# ---- 6. Downstream: cdf_to_bin_probs -----------------------------------------

def test_bin_probs_reversed_grid_now_raises():
    # Pre-fix returned corrupt [0.8999991, 1e-06, 0.0999999].
    _raises_named(lambda: cm.cdf_to_bin_probs(CDF, [2, 1, 0], [0.5, 1.5]),
                  "strictly increasing")
    _raises_named(lambda: cm.cdf_to_bin_probs([0.2, NAN, 0.9], NOM, [0.5]),
                  "finite")
    _raises_named(lambda: cm.cdf_to_bin_probs(CDF, NOM, [NAN]), "finite")


def test_bin_probs_ordered_bit_identical():
    probs = cm.cdf_to_bin_probs(CDF, NOM, [0.5, 1.5])
    want_bits = ["666666666666d63f", "666666666666d63f",
                 "343333333333d33f"]  # pre-fix bits for [0.35, 0.35, 0.3]
    assert [_bits(p) for p in probs] == want_bits
    assert abs(probs.sum() - 1.0) < 1e-12


def test_client_adapter_raises_and_matches():
    import metaculus_client as mc
    assert mc.cdf_at is cm.cdf_at
    _raises_named(lambda: mc.cdf_at(CDF, [2, 1, 0], 1.0),
                  "strictly increasing")
    _raises_named(lambda: mc.cdf_at(CDF, NOM, NAN), "finite")
    assert mc.cdf_at(CDF, NOM, 0.5) == 0.35


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for fn in ALL_TESTS:
        try:
            fn()
            check(fn.__name__, True)
        except AssertionError as e:
            check(fn.__name__, False, str(e))
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
