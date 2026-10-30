"""
test_crypto_horizon_type_validation.py — SF-BL-11240 regression tests.

Prereg: crypto.py make_labels and make_return_label raise ValueError
naming horizon on any bool, string or non-integer-float horizon, with
positive-integer-horizon labels bit-identical.
Kill: horizon=True still returns a label series, or horizon='5' raises
a bare TypeError instead of the named guard.

Pre-fix repro (both live copies): make_labels(df, True) returned the
horizon=1 series bit-identically; make_labels(df, '5') raised
TypeError (bad operand / '<' not supported) naming nothing;
make_labels(df, 2.5) raised TypeError (slice indices).

Fix mirrors calibration._validate_n_bins (SF-BL-9722): only int /
np.integer are lawful (bool excluded first); floats — non-integer
and integral alike — strings, bytes, None and other non-integers
raise the named ValueError before any shift. SF-BL-9848's value
domain (horizon < 1) is preserved under the same message.

Run: python3 tests/test_crypto_horizon_type_validation.py (exit 0 iff all pass)
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "superforecaster"))
from crypto import make_labels, make_return_label  # noqa: E402

SMALL = pd.DataFrame({"close": [100., 101., 102., 103.]})
BAD = [True, False, np.bool_(True), "5", "1", "", b"5",
       2.5, 1.5, 0.5, 2.0, 7.0, float("nan"), float("inf"), None, [1], (1,)]

def _raises(fn, h):
    try:
        fn(SMALL, h)
    except ValueError as e:
        assert "horizon" in str(e), f"does not name horizon: {e}"
        return
    except Exception as e:
        raise AssertionError(f"h={h!r} raised {type(e).__name__}, not ValueError: {e}")
    raise AssertionError(f"h={h!r} did not raise")

def test_bool_string_float_horizons_raise_named():
    for fn in (make_labels, make_return_label):
        for h in BAD: _raises(fn, h)

def test_value_domain_preserved():
    for fn in (make_labels, make_return_label):
        for h in (0, -1, -7): _raises(fn, h)

def test_kill_repros_closed():
    for fn in (make_labels, make_return_label):
        _raises(fn, True); _raises(fn, "5")

def test_positive_int_labels_bit_identical():
    assert make_labels(SMALL, 1).tolist()[:3] == [1.0, 1.0, 1.0]
    assert make_return_label(SMALL, 1).tolist()[0] == 0.009950330853168092
    for fn in (make_labels, make_return_label):
        a, b = fn(SMALL, np.int64(1)), fn(SMALL, 1)
        assert a.to_numpy(dtype=float).tobytes() == b.to_numpy(dtype=float).tobytes()

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests: fn(); print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
