"""
test_markets_panel_type_validation.py — SF-BL-11433 regression tests.

Pre-registered done/kill criteria (TASK.md): markets.py brier and
log_score raise ValueError naming y/p on any string, object-dtype or
bool element, with float panels bit-identical; kill if
brier([1, 0], [True, False]) still returns 0.0 or a string panel
still raises a bare UFuncTypeError instead of the named ValueError.

Pre-fix reproduction (both markets.py copies):
  brier([1, 0], [True, False]) == 0.0        (bool = perfect forecast)
  log_score([1, 0], [True, False]) == -1.000000500029089e-06
  brier([1, 0], ['0.5', '0.5']) == 0.25      (numeric strings coerced)
  log_score([1, 0], ['0.5', '0.5']) == -0.6931471805599453
  brier([0.0, 1.0], [0.3, 0.7]) == 0.09000000000000001  (clean control)
  log_score([1.0, 0.0], [0.9, 0.1]) == -0.10536051565782628 (control)

Note the argument order: markets.brier / markets.log_score are (y, p).

Run:  python3 tests/test_markets_panel_type_validation.py
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as the other fleet tests.
"""
import os
import sys

import numpy as np
import pandas as pd

_PKG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "superforecaster")
if not os.path.exists(os.path.join(_PKG, "markets.py")):
    # Work-dir copy (backlog/work/SF-BL-11433/): use the repo layout.
    _PKG = "/home/hatch/workspace/superforecaster/superforecaster"
sys.path.insert(0, _PKG)

from markets import brier, log_score  # noqa: E402

SCORERS = [(brier, "brier"), (log_score, "log_score")]

BAD_PANELS = {
    "p": [
        [True, False], [0.5, True], np.array([True, False]),
        np.array([np.bool_(True), 0.5], dtype=object),
        pd.Series([True, False]), pd.Series([True, False], dtype="boolean"),
        ["0.5", "0.5"], [0.5, "0.5"], ("0.5", "0.5"), [["0.5"], ["0.5"]],
        np.array(["0.5", "0.5"]), np.array([b"0.5", b"0.5"]),
        np.array(["0.5", "0.5"], dtype=object), ["abc", "0.5"],
        [b"0.5"], "0.5", pd.Series(["0.9", "0.1"]),
        pd.Series(["0.9", "0.1"], dtype="string"),
        pd.Index(["0.9", "0.1"]), pd.Categorical(["0.9", "0.1"]),
        np.array([], dtype="<U3"),
        np.array([object(), 0.5], dtype=object),
    ],
    "y": [
        [True, False], np.array([True, False]),
        np.array([np.bool_(True), 0.0], dtype=object),
        ["1", "0"], [1, "0"], np.array(["1", "0"]),
        np.array(["1", "0"], dtype=object), [np.str_("1"), 0],
        pd.Series(["1", "0"]), pd.Series(["1", "0"], dtype=object),
        pd.Categorical(["1", "0"]),
        np.array([object(), 0], dtype=object),
    ],
}


def _assert_named_valueerror(fn, y, p, panel, kind_word):
    try:
        fn(y, p)
    except ValueError as e:
        msg = str(e)
        assert type(e) is ValueError, type(e)
        assert "SF-BL-11433" in msg, f"no task tag: {e}"
        assert f"in {panel}" in msg, f"does not name {panel}: {e}"
        assert kind_word in msg, f"does not name {kind_word!r}: {e}"
        return
    except Exception as e:  # noqa: BLE001 — bare UFuncTypeError lands here
        raise AssertionError(f"wrong exception {type(e).__name__}: {e}")
    raise AssertionError(f"{fn.__name__}(y={y!r}, p={p!r}) did not raise")


def test_kill_bool_forecast_panel_no_longer_perfect():
    # Kill criterion: brier([True, False] forecasts) must not return 0.0.
    _assert_named_valueerror(brier, [1, 0], [True, False], "p", "bool")
    _assert_named_valueerror(log_score, [1, 0], [True, False], "p", "bool")


def test_kill_string_panel_raises_named_valueerror_not_ufunc():
    # Kill criterion: a string panel must raise the named ValueError.
    _assert_named_valueerror(brier, [1, 0], ["0.5", "0.5"], "p", "string")
    _assert_named_valueerror(log_score, [1, 0], ["0.5", "0.5"], "p", "string")
    _assert_named_valueerror(brier, [1, 0], ["abc", "0.5"], "p", "string")


def test_brier_rejects_all_bad_panels():
    for p in BAD_PANELS["p"]:
        kind = "bool" if _is_bool_panel(p) else (
            "object" if _is_bare_object_panel(p) else "string")
        _assert_named_valueerror(brier, [1, 0], p, "p", kind)
    for y in BAD_PANELS["y"]:
        kind = "bool" if _is_bool_panel(y) else (
            "object" if _is_bare_object_panel(y) else "string")
        _assert_named_valueerror(brier, y, [0.5, 0.5], "y", kind)


def test_log_score_rejects_all_bad_panels():
    for p in BAD_PANELS["p"]:
        kind = "bool" if _is_bool_panel(p) else (
            "object" if _is_bare_object_panel(p) else "string")
        _assert_named_valueerror(log_score, [1, 0], p, "p", kind)
    for y in BAD_PANELS["y"]:
        kind = "bool" if _is_bool_panel(y) else (
            "object" if _is_bare_object_panel(y) else "string")
        _assert_named_valueerror(log_score, y, [0.5, 0.5], "y", kind)


def _is_bool_panel(panel):
    if isinstance(panel, np.ndarray):
        return panel.dtype.kind == "b" or (
            panel.dtype == object and panel.size
            and isinstance(panel.ravel()[0], (bool, np.bool_)))
    if isinstance(panel, (pd.Series, pd.Index)):
        return panel.dtype.kind == "b" or str(panel.dtype) == "boolean"
    if isinstance(panel, (list, tuple)):
        return any(isinstance(v, (bool, np.bool_)) for v in panel)
    return False


def _is_bare_object_panel(panel):
    return isinstance(panel, np.ndarray) and panel.dtype == object \
        and panel.size and type(panel.ravel()[0]) is object


def test_both_string_panels_name_y_first():
    _assert_named_valueerror(brier, ["1"], ["0.5"], "y", "string")
    _assert_named_valueerror(log_score, ["1"], ["0.5"], "y", "string")


def test_float_panels_bit_identical():
    assert brier([0.0, 1.0], [0.3, 0.7]) == 0.09000000000000001
    assert brier([1, 0], [0.9, 0.1]) == 0.009999999999999998
    assert brier([1, 0], [1, 0]) == 0.0  # int endpoints are lawful
    assert brier(np.array([1.0, 0.0]), np.array([0.5, 0.5])) == 0.25
    assert log_score([1.0, 0.0], [0.9, 0.1]) == -0.10536051565782628
    assert log_score([1, 0], [0.5, 0.5]) == -0.6931471805599453
    # Lawful containers stay lawful (fleet precedent: object arrays
    # holding only plain numbers, numeric categoricals, float Series).
    assert brier([1, 0], np.array([0.5, 0.5], dtype=object)) == 0.25
    assert brier([1, 0], pd.Categorical([0.9, 0.1])) == 0.009999999999999998
    assert brier(pd.Series([1, 0]), pd.Series([0.9, 0.1])) \
        == 0.009999999999999998
    assert log_score((1, 0), (0.9, 0.1)) == -0.10536051565782628


def test_prior_validations_preserved():
    # SF-BL-9886 (brier): shape mismatch / empty / non-finite masking.
    for y, p in [([1, 0], [0.5]), ([], [])]:
        try:
            brier(y, p)
        except ValueError as e:
            assert "SF-BL-9886" in str(e), e
        else:
            raise AssertionError(f"brier({y}, {p}) did not raise")
    assert brier(np.array([np.nan, 1.0]), np.array([0.5, 0.5])) == 0.25
    # SF-BL-9887 (log_score): shape / empty / non-finite / range / binary y.
    for y, p in [([1, 0], [0.5]), ([], []), ([1, 0], [1.5, 0.5]),
                 ([2, 0], [0.5, 0.5]), ([1, 0], [np.nan, 0.5])]:
        try:
            log_score(y, p)
        except ValueError as e:
            assert "SF-BL-9887" in str(e), e
        else:
            raise AssertionError(f"log_score({y}, {p}) did not raise")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
