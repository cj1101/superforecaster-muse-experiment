"""SF-BL-11466 regression suite: markets.calibration_table bins type validation.

Run: python3 -m pytest test_markets_bins_type_validation.py -q
     (or plain python3 test_markets_bins_type_validation.py)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "markets"))
# Fallback for repo tests/ location
if not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "..", "markets", "markets.py")):
    sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster_out/markets"))

import numpy as np
import markets
from markets import calibration_table

Y = np.array([0., 1.])
P = np.array([0.25, 0.75])
RNG = np.random.default_rng(42)
P_BIG = RNG.random(200)
Y_BIG = (RNG.random(200) < P_BIG).astype(float)


def _prefix_table(y, p, bins):
    """Exact pre-fix body, as oracle for bit-identical lawful tables."""
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum() > 0:
            rows.append({"bin_lo": round(float(edges[i]), 2), "bin_hi": round(float(edges[i + 1]), 2),
                         "n": int(m.sum()), "mean_pred": round(float(p[m].mean()), 4),
                         "empirical_rate": round(float(y[m].mean()), 4)})
    return rows


def _assert_named_value_error(bad):
    try:
        r = calibration_table(Y, P, bins=bad)
    except ValueError as e:
        assert "bins" in str(e), f"ValueError must name bins for {bad!r}: {e}"
        return
    except Exception as e:
        raise AssertionError(f"bins={bad!r} raised {type(e).__name__} not ValueError: {e}")
    raise AssertionError(f"bins={bad!r} did not raise; returned {r!r}")


def test_prereg_cases_raise_named_value_error():
    for bad in [True, "10", 2.5, None]:
        _assert_named_value_error(bad)


def test_other_bad_types_raise_named_value_error():
    for bad in [False, np.bool_(True), np.bool_(False), "", "ten", 10.0, 1.0, 0.5,
                float("nan"), float("inf"), float("-inf"), [10], (10,), {"bins": 10}, object()]:
        _assert_named_value_error(bad)


def test_nonpositive_integers_raise_named_value_error():
    # SF-BL-9888 value domain: previously [] for 0 / -1
    for bad in [0, -1, -100, np.int64(0)]:
        _assert_named_value_error(bad)


def test_kill_criterion_bins_true_no_longer_single_bin():
    # Pre-fix: [{'bin_lo':0.0,'bin_hi':1.0,'n':2,'mean_pred':0.5,...}]
    _assert_named_value_error(True)


def test_integer_bins_bit_identical():
    for b in [1, 2, 3, 5, 10, 20]:
        assert calibration_table(Y_BIG, P_BIG, bins=b) == _prefix_table(Y_BIG, P_BIG, b), b
        assert calibration_table(Y, P, bins=b) == _prefix_table(Y, P, b), b
    # default
    assert calibration_table(Y, P) == _prefix_table(Y, P, 10)
    assert calibration_table(Y_BIG, P_BIG) == _prefix_table(Y_BIG, P_BIG, 10)


def test_numpy_integers_accepted_bit_identical():
    for b in [np.int64(7), np.int32(4), np.int16(2)]:
        assert calibration_table(Y_BIG, P_BIG, bins=b) == _prefix_table(Y_BIG, P_BIG, int(b))


def test_task_repro_control():
    # bins=10 control from TASK.md on the repro panel
    assert calibration_table(Y, P, bins=10) == [
        {"bin_lo": 0.2, "bin_hi": 0.3, "n": 1, "mean_pred": 0.25, "empirical_rate": 0.0},
        {"bin_lo": 0.7, "bin_hi": 0.8, "n": 1, "mean_pred": 0.75, "empirical_rate": 1.0},
    ]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} tests passed")
