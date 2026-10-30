"""
test_markets_auc_panel_type_validation.py — SF-BL-11432 regression tests.

Pre-registered done/kill criteria (TASK.md): markets.py auc raises
ValueError naming y/p on any string, object-dtype or bool element,
with float panels bit-identical; kill if string forecasts or
outcomes still return AUC 1.0 or the bool panel still returns 1.0
instead of raising.

Pre-fix reproduction (numpy panels, both markets.py copies):
  auc([1, 0], ['0.9', '0.1']) == 1.0   (string forecasts coerced,
                                        bit-identical to control)
  auc([1, 0], [True, False]) == 1.0   (bool flags = perfect AUC)
  auc([True, False], [0.9, 0.1]) == 1.0
  auc(['1', '0'], [0.9, 0.1]) is NaN  (string outcomes never == 1,
                                        silently undefined)
  auc([1, 0], [0.9, 0.1]) == 1.0      (clean control)

Note the argument order: markets.auc is (y, p). Lawful-panel
controls below use the container forms auc could already score
pre-fix (ndarray / pandas Series); the type screen itself runs on
the RAW inputs of any container form, so bad panels raise the named
ValueError even as plain lists.

Run:  python3 tests/test_markets_auc_panel_type_validation.py
No pytest dependency: test_* functions with assert + __main__
runner, same convention as the other fleet tests.
"""
import os
import sys

import numpy as np
import pandas as pd

_PKG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "superforecaster")
if not os.path.exists(os.path.join(_PKG, "markets.py")):
    # Work-dir copy (backlog/work/SF-BL-11432/): use the repo layout.
    _PKG = "/home/hatch/workspace/superforecaster/superforecaster"
sys.path.insert(0, _PKG)

from markets import auc, brier, calibration_table, log_score  # noqa: E402

Y_OK = np.array([1, 0])
P_OK = np.array([0.9, 0.1])

BAD_P_PANELS = [
    # (panel, kind word expected in the error message)
    (np.array(["0.9", "0.1"]), "string"),          # kill case
    (["0.9", "0.1"], "string"),
    ([0.9, "0.1"], "string"),
    (np.array(["abc", "def"]), "string"),
    (np.array([b"0.9", b"0.1"]), "string"),
    (np.array(["0.9", "0.1"], dtype=object), "string"),
    (pd.Series(["0.9", "0.1"]), "string"),
    (pd.Series(["0.9", "0.1"], dtype="string"), "string"),
    (pd.Categorical(["0.9", "0.1"]), "string"),
    (np.array([], dtype="<U3"), "string"),
    ("0.9", "string"),
    (np.array([True, False]), "bool"),             # kill case
    ([True, False], "bool"),
    ([0.9, True], "bool"),
    (np.array([True, False], dtype=object), "bool"),
    (np.array([np.bool_(True), 0.5], dtype=object), "bool"),
    (pd.Series([True, False]), "bool"),
    (pd.Series([True, False], dtype="boolean"), "bool"),
    (np.array([object(), 0.5], dtype=object), "object"),
    (np.array([None, 0.5], dtype=object), "object"),
]

BAD_Y_PANELS = [
    (np.array(["1", "0"]), "string"),              # kill case (was NaN)
    (["1", "0"], "string"),
    ([1, "0"], "string"),
    (np.array(["1", "0"], dtype=object), "string"),
    (pd.Series(["1", "0"]), "string"),
    (pd.Categorical(["1", "0"]), "string"),
    (np.array([True, False]), "bool"),             # was AUC 1.0
    ([True, False], "bool"),
    (np.array([True, False], dtype=object), "bool"),
    (pd.Series([True, False]), "bool"),
    (np.array([object(), 0], dtype=object), "object"),
]


def _assert_named_valueerror(y, p, panel, kind_word):
    try:
        auc(y, p)
    except ValueError as e:
        msg = str(e)
        assert type(e) is ValueError, type(e)
        assert "SF-BL-11432" in msg, f"no task tag: {e}"
        assert "auc" in msg, f"does not name the scorer: {e}"
        assert f"in {panel}" in msg, f"does not name {panel}: {e}"
        assert kind_word in msg, f"does not name {kind_word!r}: {e}"
        return
    except Exception as e:  # noqa: BLE001 — any other type is a failure
        raise AssertionError(f"wrong exception {type(e).__name__}: {e}")
    raise AssertionError(f"auc(y={y!r}, p={p!r}) did not raise")


def test_kill_string_forecast_panel_no_longer_returns_1():
    # Kill criterion: string forecasts must not return AUC 1.0.
    _assert_named_valueerror(Y_OK, np.array(["0.9", "0.1"]), "p", "string")


def test_kill_string_outcome_panel_raises():
    # Kill criterion: string outcomes must not return AUC 1.0 (pre-fix
    # they silently returned NaN — either way, not a named error).
    _assert_named_valueerror(np.array(["1", "0"]), P_OK, "y", "string")


def test_kill_bool_forecast_panel_no_longer_returns_1():
    # Kill criterion: the bool panel must not return AUC 1.0.
    _assert_named_valueerror(Y_OK, np.array([True, False]), "p", "bool")
    _assert_named_valueerror(Y_OK, [True, False], "p", "bool")


def test_auc_rejects_all_bad_forecast_panels():
    for p, kind in BAD_P_PANELS:
        _assert_named_valueerror(Y_OK, p, "p", kind)


def test_auc_rejects_all_bad_outcome_panels():
    for y, kind in BAD_Y_PANELS:
        _assert_named_valueerror(y, P_OK, "y", kind)


def test_both_string_panels_name_y_first():
    _assert_named_valueerror(np.array(["1", "0"]), np.array(["0.9", "0.1"]),
                             "y", "string")


def test_float_panels_bit_identical():
    # Every expected value captured from the pre-fix implementation.
    assert auc(np.array([1, 0]), np.array([0.9, 0.1])) == 1.0
    assert auc(np.array([1, 0, 1, 0]),
               np.array([0.9, 0.1, 0.8, 0.2])) == 1.0
    assert auc(np.array([1, 0, 1, 0]),
               np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert auc(np.array([1, 0, 1, 0, 1]),
               np.array([0.7, 0.6, 0.4, 0.3, 0.9])) == 0.8333333333333334
    assert auc(np.array([1, 0]), np.array([0.1, 0.9])) == 0.0
    # Int panels are lawful (distinct from bool).
    assert auc(np.array([1, 0, 1, 0]), np.array([1, 0, 1, 0])) == 1.0
    # Fleet precedent (SF-BL-11433): an object-dtype container holding
    # only plain numbers stays lawful — elements are scanned, the
    # container is not rejected wholesale.
    assert auc(np.array([1, 0], dtype=object),
               np.array([0.9, 0.1], dtype=object)) == 1.0
    # Pandas float Series stay lawful.
    assert auc(pd.Series([1.0, 0.0, 1.0, 0.0]),
               pd.Series([0.9, 0.1, 0.8, 0.2])) == 1.0
    # Larger deterministic panel, exact pre-fix value.
    rng = np.random.default_rng(11432)
    yr = (rng.random(50) > 0.5).astype(float)
    pr = rng.random(50)
    assert auc(yr, pr) == 0.5942028985507246


def test_computation_path_unchanged_out_of_scope():
    # Single-class panels are SF-BL-10066's scope, not this task's:
    # behaviour must be exactly as pre-fix (NaN, no new error).
    assert np.isnan(auc(np.array([1, 1]), np.array([0.9, 0.1])))


def test_sibling_screens_preserved():
    # The shared helper's earlier call sites keep their own tags.
    for fn, tag in [(brier, "SF-BL-11433"), (log_score, "SF-BL-11433"),
                    (calibration_table, "SF-BL-11465")]:
        try:
            fn([1, 0], [True, False])
        except ValueError as e:
            assert tag in str(e), f"{fn.__name__}: {e}"
        else:
            raise AssertionError(f"{fn.__name__} bool panel did not raise")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
