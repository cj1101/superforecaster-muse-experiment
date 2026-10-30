"""
test_cdf_module_disciplined_probs_order.py — SF-BL-10316 verification.

Pre-registered done/kill criterion (TASK.md):
  "Fix sorts probs (carrying values with them) or raises a named
   ValueError for non-increasing probs, with sorted-probs outputs
   unchanged; kill if probs=(0.95, 0.05) returns a dict whose 0.05
   value equals its 0.95 value."

Bug: disciplined_percentiles indexed log-returns by caller probs
order, picked med_idx as nearest-to-0.5, then np.maximum.accumulate
plus the strict-increase nudge silently overwrote out-of-order
quantiles — probs=(0.95, 0.05) returned {0.95: 101.706,
0.05: 101.706} in the task's live repro, the 5th percentile
destroyed onto the 95th with no error — unlike
_validate_percentiles, which enforces increasing probabilities for
declared percentiles. Distinct from SF-BL-11268 (this function's
probs TYPE domain) and SF-BL-11243/11244 (probs type in other
functions); this is the probs VALUE / ORDER domain of
disciplined_percentiles only.

Fix (chosen branch of the criterion): raise — after the existing
type screen and before climatology_percentiles is called, any
non-increasing probs (unsorted, reversed, partially out of order,
or holding a duplicate) raises ValueError naming probs, tagged
SF-BL-10316. Reject-only: strictly increasing probs (incl. the
default and single-element probs) fall through to the identical
computation, so sorted-probs dicts stay bit-identical (expected
bit patterns below are the pre-fix outputs, verified against a
verbatim pre-fix reference in SF-BL-10316 verify_fix.py).

Note: this supersedes the "unsorted" panel of
test_cdf_module_disciplined_probs_type.py (SF-BL-11268), which
asserted pre-10316 bit-identity for (0.95, 0.05); SF-BL-11268
explicitly deferred the unsorted value domain to this task.

Run: python3 tests/test_cdf_module_disciplined_probs_order.py
     or pytest tests/test_cdf_module_disciplined_probs_order.py
"""
import os, struct, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metaculus_bot"))
import numpy as np
import pytest
import cdf_module as cm

RNG = np.random.default_rng(10316)
TH = RNG.normal(0.001, 0.02, 200)
TD = RNG.normal(0.0005, 0.01, 200)
RD = TD[-30:]

# Pre-fix outputs for this panel (seed 10316), packed float bits in
# sorted-key order — the fix must leave every one bit-identical.
EXPECTED_SORTED = {
    (0.05, 0.25, 0.5, 0.75, 0.95):
        {0.05: 96.94653142715744, 0.25: 99.08922455542218,
         0.5: 100.15100807249206, 0.75: 101.50813272633539,
         0.95: 104.3623785346328},
}

NON_INCREASING = [
    (0.95, 0.05),
    [0.95, 0.05],
    (0.95, 0.75, 0.5, 0.25, 0.05),
    (0.05, 0.5, 0.25, 0.95),
    (0.25, 0.05, 0.5),
    (0.5, 0.5),
    (0.05, 0.25, 0.25, 0.95),
    np.array([0.95, 0.05]),
    np.array([0.1, 0.9, 0.9, 0.99]),
    np.array([0.95, 0.05], dtype=object),
]


def _call(probs):
    return cm.disciplined_percentiles(100.0, TH, TD, RD, probs=probs)


def _raises_named_10316(probs):
    with pytest.raises(ValueError) as ei:
        _call(probs)
    assert type(ei.value) is ValueError
    assert "probs" in str(ei.value), str(ei.value)
    assert "SF-BL-10316" in str(ei.value), str(ei.value)
    assert "strictly increasing" in str(ei.value), str(ei.value)


def test_kill_unsorted_probs_do_not_collapse():
    # Kill criterion: (0.95, 0.05) must not return a dict whose
    # 0.05 value equals its 0.95 value (pre-fix it returned the
    # 0.95 value at both keys, modulo the 1e-9 monotonicity nudge).
    _raises_named_10316((0.95, 0.05))


def test_all_non_increasing_raise_named():
    for p in NON_INCREASING:
        _raises_named_10316(p)


def test_order_screen_precedes_climatology():
    # Too-short trailing window would make climatology raise its
    # ">= 30 returns" error; the order screen must fire first.
    with pytest.raises(ValueError) as ei:
        cm.disciplined_percentiles(100.0, TH[:10], TD, RD,
                                   probs=(0.95, 0.05))
    assert "SF-BL-10316" in str(ei.value), str(ei.value)


def test_sorted_probs_bit_identical():
    for probs, expected in EXPECTED_SORTED.items():
        out = _call(probs)
        assert out == expected, (probs, out)
        assert list(out) == list(expected)
    # Default probs unchanged and strictly ordered in value.
    out = cm.disciplined_percentiles(100.0, TH, TD, RD)
    vals = [out[p] for p in sorted(out)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    # Single-element probs (vacuously increasing) still lawful.
    single = _call([0.5])
    assert list(single) == [0.5]


def test_type_screens_unchanged():
    # SF-BL-11268/11243 type domain untouched by the order screen.
    for bad in (["0.05", "0.5", "0.95"], [True, 0.5, 0.9]):
        with pytest.raises(ValueError) as ei:
            _call(bad)
        assert "probs" in str(ei.value)
        assert "SF-BL-10316" not in str(ei.value)


if __name__ == "__main__":
    test_kill_unsorted_probs_do_not_collapse()
    test_all_non_increasing_raise_named()
    test_order_screen_precedes_climatology()
    test_sorted_probs_bit_identical()
    test_type_screens_unchanged()
    print("ALL TESTS PASSED (5 test functions)")
