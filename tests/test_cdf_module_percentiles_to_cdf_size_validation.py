"""
test_cdf_module_percentiles_to_cdf_size_validation.py — SF-BL-10311.

Pre-registered done/kill criteria (TASK.md):
  "Fix raises a named ValueError naming size for size<2 (and
   non-integer size) before grid construction, with the default
   201-point output byte-identical; kill if size=0 or size=1
   returns any CDF object instead of raising."

Bug: cdf_module.percentiles_to_cdf never validated its size
argument: it forwarded size unchecked into grid_nominals /
np.linspace, so in the live superforecaster_out copy size=0
returned an empty CDF [] and size=1 a 1-point CDF [0.1] (task
live repro panel), True/False were read as 1/0, negative sizes
leaked a bare numpy ValueError naming nothing, and non-integer
sizes ('201', None, 201.5, NaN/inf, containers) leaked bare
TypeErrors in both copies. In the repo copy integer size<2 was
caught only collaterally, and mislabelled as a grid_nominals
problem, by SF-BL-10108. A CDF needs at least 2 points for
downstream scoring (crps_from_cdf, _cell_density).

Fix (metaculus_bot/cdf_module.py, percentiles_to_cdf, mirrored
to the superforecaster_out live copy): a _validate_cdf_size_10311
screen at entry, BEFORE grid construction — any size that is not
an integer >= 2 (bool / string / bytes / None / containers /
complex / non-finite / non-integral) raises a ValueError naming
size and SF-BL-10311. Integral sizes (int / np.integer /
integral float / Decimal / Fraction / 0-d integral arrays) are
normalised to the identical int, so lawful outputs stay
byte-identical — the SHA-256 pins below were captured live
PRE-fix (both copies agreed on every pin).

Run: python3 tests/test_cdf_module_percentiles_to_cdf_size_validation.py
     or pytest tests/test_cdf_module_percentiles_to_cdf_size_validation.py
"""
import hashlib
import os
import sys
from decimal import Decimal
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metaculus_bot"))
import numpy as np
import pytest
import cdf_module as cm

PANEL = {0.1: 5, 0.5: 10, 0.9: 20}
SHA_DEFAULT = "3d1b186d416e50e2feac16b247071f216de7d0f1adc6bf964145e5ebf10c6b19"
SHA_PCHIP = "14ef3fb2846cd8fa2b9244de55134fc8f0ede997a98d85937bbf70f1fd1408fb"
SHA_SIZE11 = "f6c30bd12ed3b1260a753aecc993e599c8d9321bef4d353d348e6e7c137db3fb"
SHA_TIED = "734c4d725d7015916fadcabb4439f8729314a78f6ebf94b51a56b1d82cfdb00e"
SHA_LOG = "d2d83907982c8c11f45b845fd767bd39e35d8cec0c602059fcfcd8bd174a1bcd"


def _sha(cdf):
    return hashlib.sha256(cdf.tobytes()).hexdigest()


def _raises_named(fn):
    with pytest.raises(ValueError) as ei:
        fn()
    msg = str(ei.value)
    assert "size" in msg and "SF-BL-10311" in msg, msg


def _s(sz, **kw):
    return lambda: cm.percentiles_to_cdf(PANEL, 0, 30, size=sz, **kw)


def test_kill_size_zero_and_one_raise_not_return():
    for sz in (0, 1):
        _raises_named(_s(sz))
        _raises_named(_s(sz, interp="pchip"))


def test_size_below_two_raises_named():
    for sz in (0, 1, -1, -5, -201, np.int64(1), False, 0.0, 1.0, -2.0):
        _raises_named(_s(sz))


def test_non_integer_size_raises_named_not_bare_typeerror():
    for sz in (201.5, 2.5, 0.5, np.float64(201.5), "201", "abc", "",
               b"201", np.str_("201"), None, [201], (201,), {201},
               {"s": 201}, np.array([201]), np.array("201"),
               Decimal("201.5"), Fraction(5, 2), 201 + 0j, object(),
               float("nan"), float("inf"), float("-inf")):
        _raises_named(_s(sz))


def test_bool_size_raises_named():
    for sz in (True, False, np.bool_(True), np.array(True)):
        _raises_named(_s(sz))


def test_no_bad_size_returns_a_cdf_kill_sweep():
    for sz in (0, 1, -1, True, False, 201.5, "201", None, [201]):
        try:
            out = cm.percentiles_to_cdf(PANEL, 0, 30, size=sz)
        except ValueError:
            continue
        raise AssertionError(f"size={sz!r} returned {out!r}")


def test_positional_and_tied_size_screened():
    _raises_named(lambda: cm.percentiles_to_cdf(PANEL, 0, 30, None, 1))
    _raises_named(lambda: cm.percentiles_to_cdf(
        {0.1: 10, 0.5: 10, 0.9: 20}, 0, 30, size=0))


def test_default_201_byte_identical():
    assert _sha(cm.percentiles_to_cdf(PANEL, 0, 30)) == SHA_DEFAULT
    assert _sha(cm.percentiles_to_cdf(PANEL, 0, 30, size=201)) == SHA_DEFAULT
    assert len(cm.percentiles_to_cdf(PANEL, 0, 30)) == 201
    assert _sha(cm.percentiles_to_cdf(PANEL, 0, 30, interp="pchip")) == SHA_PCHIP
    assert _sha(cm.percentiles_to_cdf(PANEL, 0, 30, size=11)) == SHA_SIZE11
    assert _sha(cm.percentiles_to_cdf({0.1: 10, 0.5: 10, 0.9: 20}, 0, 30)) == SHA_TIED
    assert _sha(cm.percentiles_to_cdf(PANEL, 1.0, 100.0, zero_point=0.0)) == SHA_LOG


def test_integral_size_forms_bit_identical():
    for sz in (201, np.int64(201), np.int32(201), 201.0, np.float64(201.0),
               Decimal("201"), Fraction(201, 1), np.array(201), np.array(201.0)):
        assert _sha(cm.percentiles_to_cdf(PANEL, 0, 30, size=sz)) == SHA_DEFAULT, repr(sz)


def test_size_two_is_lawful():
    assert cm.percentiles_to_cdf(PANEL, 0, 30, size=2).tolist() == [0.1, 0.9]


def test_prior_screens_preserved():
    _raises_named(_s(1))  # size screen precedes grid construction
    with pytest.raises(ValueError, match="percentiles"):
        cm.percentiles_to_cdf({0.1: True, 0.5: 10, 0.9: 20}, 0, 30)
    with pytest.raises(ValueError, match="percentile"):
        cm.percentiles_to_cdf({0.1: float("nan"), 0.5: 10, 0.9: 20}, 0, 30)
    with pytest.raises(ValueError, match="interp"):
        cm.percentiles_to_cdf(PANEL, 0, 30, interp="cubic")


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
