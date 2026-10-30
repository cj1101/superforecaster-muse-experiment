"""
test_cdf_module_mc_scores_probs_type_validation.py — SF-BL-11269 verification.

Preregistered criterion (TASK.md): "mc_scores raises ValueError naming
probs on any bool or string-dtype probs element, with numeric probs
(log score, Brier) pairs bit-identical; kill if probs=['0.5','0.5']
still returns (-0.6931, 0.5) or probs=[True, False] still returns
(0.0, 0.0)."

Bug: cdf_module.mc_scores validated probs VALUES only after
np.asarray(probs, dtype=float) coerced every member, so string members
parsed to their values (['0.5','0.5'] scored bit-identical to the
numeric control) and bools read as 1.0/0.0 ([True, False] scored a
perfect (0.0, 0.0)). Distinct from SF-BL-9925 (probs values /
off-simplex), SF-BL-10472 (NaN members) and SF-BL-11202 (true_bin type
only); none fixtured probs member TYPE.

Fix: a reject-only type screen at mc_scores entry, BEFORE any float
coercion, on the raw input — bool / string / bytes members in any form
(Python or numpy scalars, sequence elements at any depth, bool- or
string-dtype ndarrays, object-dtype arrays, bare string/bytes probs)
raise a ValueError naming probs, triple-tagged SF-BL-11269 +
SF-BL-10472 + SF-BL-9925 so all suites' message assertions stay valid.
Genuinely numeric probs fall through untouched; clean panels below are
bit-identical (hex captured pre-fix).

Run:  python3 tests/test_cdf_module_mc_scores_probs_type_validation.py
"""
import os
import sys
from decimal import Decimal
from fractions import Fraction

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metaculus_bot"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import cdf_module as cm  # noqa: E402


def _raises_named(probs, true_bin):
    with pytest.raises(ValueError) as excinfo:
        cm.mc_scores(probs, true_bin)
    msg = str(excinfo.value)
    assert "mc_scores" in msg and "probs" in msg \
        and "SF-BL-11269" in msg, msg


def test_task_repro_string_probs_raise_named():
    _raises_named(['0.5', '0.5'], 1)
    _raises_named(['0.5', '0.5'], 0)


def test_task_repro_bool_probs_raise_named():
    _raises_named([True, False], 0)
    _raises_named([False, True], 1)


def test_kill_string_probs_do_not_return_control_pair():
    try:
        out = cm.mc_scores(['0.5', '0.5'], 1)
    except ValueError:
        return
    raise AssertionError(f"kill: string probs returned {out!r}")


def test_kill_bool_probs_do_not_return_perfect_pair():
    try:
        out = cm.mc_scores([True, False], 0)
    except ValueError:
        return
    raise AssertionError(f"kill: bool probs returned {out!r}")


def test_string_forms_all_raise():
    for bad in (['0.5', '0.5'], ('0.5', '0.5'), ['0.2', '0.8'],
                [0.5, '0.5'], ['0.5', 0.5], [np.str_('0.5'), 0.5],
                [b'0.5', b'0.5'], '0.5', b'0.5',
                np.array(['0.5', '0.5']), np.array(['0.2', '0.8']),
                np.array([b'0.5', b'0.5']),
                np.array(['0.5', '0.5'], dtype=object),
                np.array([0.5, '0.5'], dtype=object),
                [['0.5', '0.5']]):
        _raises_named(bad, 0)


def test_bool_forms_all_raise():
    for bad in ([True, False], [False, True], [True, True],
                [0.5, True], [False, 0.5], [np.bool_(True), False],
                [1, True], np.array([True, False]),
                np.array([True, False], dtype=object),
                np.array([0.5, True], dtype=object),
                True, np.bool_(False)):
        _raises_named(bad, 0)


def test_no_corrupt_type_returns_any_score_kill_sweep():
    corrupt = [
        ['0.5', '0.5'], [True, False], np.array(['0.5', '0.5']),
        np.array([True, False]), [0.5, '0.5'], [0.5, False],
        [np.str_('1.0')], [np.bool_(True)], '1.0', [b'1.0'],
    ]
    for bad in corrupt:
        for tb in (0, 1):
            try:
                cm.mc_scores(bad, tb)
            except ValueError as e:
                assert "probs" in str(e) and "SF-BL-11269" in str(e)
                continue
            raise AssertionError(f"kill sweep: {bad!r} tb={tb} scored")


def test_numeric_controls_bit_identical():
    panels = [
        ([0.5, 0.5], 1, "-0x1.62e42fefa39efp-1", "0x1.0000000000000p-1"),
        ([0.2, 0.8], 1, "-0x1.c8ff7c79a9a20p-3", "0x1.47ae147ae147ap-4"),
        ([0.2, 0.8], 0, "-0x1.9c041f7ed8d33p+0", "0x1.47ae147ae147cp+0"),
        ([0.25, 0.25, 0.25, 0.25], 2, "-0x1.62e42fefa39efp+0", "0x1.8000000000000p-1"),
        ([0.0, 1.0], 0, "-0x1.ba18a998fffa0p+4", "0x1.0000000000000p+1"),
        ([1.0], 0, "0x0.0p+0", "0x0.0p+0"),
    ]
    for probs, tb, log_hex, brier_hex in panels:
        ls, br = cm.mc_scores(probs, tb)
        assert float(ls).hex() == log_hex, (probs, tb, ls)
        assert float(br).hex() == brier_hex, (probs, tb, br)


def test_numeric_forms_still_score_identically():
    ref, refb = cm.mc_scores([0.2, 0.8], 1)
    for form in ((0.2, 0.8), np.array([0.2, 0.8]),
                 [Fraction(1, 5), Fraction(4, 5)],
                 [Decimal('0.2'), Decimal('0.8')],
                 np.array([0.2, 0.8], dtype=object),
                 [np.float64(0.2), np.float64(0.8)],
                 [np.int64(0), np.int64(1)] if False else [0, 1]):
        ls, br = cm.mc_scores(form, 1)
        if isinstance(form, list) and form == [0, 1]:
            assert float(ls) == 0.0 and float(br) == 0.0
        else:
            assert float(ls).hex() == float(ref).hex(), form
            assert float(br).hex() == float(refb).hex(), form
    # int probs and within-tolerance sum still score (pre-fix behaviour)
    ls, _ = cm.mc_scores([0.5, 0.5000005], 0)
    assert ls == float(np.log(0.5))
    ls, _ = cm.mc_scores([0.2, 0.8], np.int64(1))
    assert float(ls).hex() == "-0x1.c8ff7c79a9a20p-3"


def test_prior_screens_preserved():
    # SF-BL-9925 value classes still raise with their tag
    for bad, tb in (([-0.5, 1.5], 0), ([0.5, 0.6], 0), ([0.2, 0.8], -1),
                    ([0.2, 0.8], True), (["a", "b"], 0)):
        with pytest.raises(ValueError) as ei:
            cm.mc_scores(bad, tb)
        assert "SF-BL-9925" in str(ei.value), (bad, tb, ei.value)
    # SF-BL-10472 non-finite members still raise with their tag
    for bad in ([float("nan"), 0.5], [0.5, float("inf")]):
        with pytest.raises(ValueError) as ei:
            cm.mc_scores(bad, 0)
        assert "SF-BL-10472" in str(ei.value), (bad, ei.value)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} passed")
