"""test_backtest_auc_type_validation.py — SF-BL-11391 verification.

Pre-registered done/kill criteria (TASK.md):
  "Guard raises ValueError naming y/p on any string or bool element
   reaching auc_mann_whitney, with float panels bit-identical; kill if
   auc_mann_whitney([1, 0], ['0.9', '0.1']) or the [True, False] panel
   still returns 1.0 instead of raising."

Fix (backtest.py, both live copies): auc_mann_whitney now screens its
RAW panels first, under its own name — _reject_string_panels (string,
either panel), _reject_bool_outcomes (bool outcomes, new helper),
_reject_bool_probabilities (bool forecasts, SF-BL-11017 screen) —
all before _check's float coercion, tagged SF-BL-11391. The bool
detector also screens pandas categorical containers. Lawful panels
are pure-validation bit-identical (lawful_baseline digest
ff87c15c4ac3e326d237921f80d29601166c1f588e59e3fbea9788cb1a915d7d
identical pre/post on both copies).

Run:  BT_PATH=<path-to-backtest.py> python3 test_backtest_auc_type_validation.py
      (default: the canonical package module; also pytest-collectable)
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_bt_path = os.environ.get("BT_PATH") or (sys.argv[1] if len(sys.argv) > 1 and os.path.basename(sys.argv[1]) == "backtest.py" else None)
if _bt_path:
    _spec = importlib.util.spec_from_file_location("bt_under_test_11391", _bt_path)
    bt = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = bt
    _spec.loader.exec_module(bt)
else:
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from superforecaster import backtest as bt  # noqa: E402

RESULTS = []


def check(name, ok):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}: {name}")


def raises_named(fn, panel, kind):
    try:
        fn()
    except ValueError as e:
        m = str(e)
        return (m.startswith("auc_mann_whitney:") and f"in {panel} " in m
                and kind in m and "SF-BL-11391" in m)
    except Exception:  # noqa: BLE001
        return False
    return False


def raises_tag(fn, tag):
    try:
        fn()
    except ValueError as e:
        return tag in str(e)
    except Exception:  # noqa: BLE001
        return False
    return False


STR_PS = [
    ("list", ["0.9", "0.1"]), ("tuple", ("0.9", "0.1")),
    ("mixed", [0.9, "0.1"]), ("non-numeric", ["abc", "0.1"]),
    ("nested", [["0.9"], ["0.1"]]), ("scalar", "0.9"),
    ("np.str_", [np.str_("0.9"), np.str_("0.1")]),
    ("nd <U", np.array(["0.9", "0.1"])), ("nd <S", np.array([b"0.9", b"0.1"])),
    ("nd object", np.array(["0.9", 0.1], dtype=object)),
    ("bytes elem", [b"0.9", b"0.1"]),
    ("Series str", pd.Series(["0.9", "0.1"])),
    ("Series string-dtype", pd.Series(["0.9", "0.1"], dtype="string")),
    ("Series category", pd.Series(["0.9", "0.1"], dtype="category")),
    ("Index str", pd.Index(["0.9", "0.1"])),
    ("Categorical", pd.Categorical(["0.9", "0.1"])),
    ("CategoricalIndex", pd.CategoricalIndex(["0.9", "0.1"])),
    ("empty <U dtype", np.array([], dtype="<U3")),
    ("empty Categorical str cats", pd.Categorical([], categories=["0.9"])),
]
STR_YS = [
    ("list", ["1", "0"]), ("tuple", ("1", "0")),
    ("nd <U", np.array(["1", "0"])),
    ("nd object", np.array(["1", 0], dtype=object)),
    ("Series str", pd.Series(["1", "0"])),
    ("Series category", pd.Series(["1", "0"], dtype="category")),
    ("Categorical", pd.Categorical(["1", "0"])), ("scalar", "1"),
]
BOOL_PS = [
    ("list", [True, False]), ("mixed", [0.9, True]),
    ("tuple", (True, False)),
    ("np.bool_", [np.bool_(True), np.bool_(False)]),
    ("nd bool", np.array([True, False])),
    ("nd object", np.array([0.9, True], dtype=object)),
    ("nested", [[True], [False]]),
    ("Series bool", pd.Series([True, False])),
    ("Index bool", pd.Index([True, False])),
    ("Series category", pd.Series([True, False], dtype="category")),
    ("Categorical", pd.Categorical([True, False])), ("scalar", True),
]
BOOL_YS = [
    ("list", [True, False]), ("nd bool", np.array([True, False])),
    ("Series bool", pd.Series([True, False])), ("tuple", (True, False)),
    ("mixed int+bool", [1, True]), ("Categorical", pd.Categorical([True, False])),
]


def test_kill_criteria():
    check("A1 KILL string p", raises_named(lambda: bt.auc_mann_whitney([1, 0], ["0.9", "0.1"]), "p", "string"))
    check("A2 KILL bool p", raises_named(lambda: bt.auc_mann_whitney([1, 0], [True, False]), "p", "boolean"))
    check("A3 KILL bool y", raises_named(lambda: bt.auc_mann_whitney([True, False], [0.9, 0.1]), "y", "boolean"))
    check("A4 KILL string y", raises_named(lambda: bt.auc_mann_whitney(["1", "0"], [0.9, 0.1]), "y", "string"))


def test_string_forms():
    for name, p in STR_PS:
        check(f"B string p {name}", raises_named(lambda p=p: bt.auc_mann_whitney([1, 0], p), "p", "string"))
    for name, y in STR_YS:
        check(f"B string y {name}", raises_named(lambda y=y: bt.auc_mann_whitney(y, [0.9, 0.1]), "y", "string"))


def test_bool_forms():
    for name, p in BOOL_PS:
        check(f"C bool p {name}", raises_named(lambda p=p: bt.auc_mann_whitney([1, 0], p), "p", "boolean"))
    for name, y in BOOL_YS:
        check(f"C bool y {name}", raises_named(lambda y=y: bt.auc_mann_whitney(y, [0.9, 0.1]), "y", "boolean"))


def test_lawful_exact():
    check("D1 control", bt.auc_mann_whitney([1, 0], [0.9, 0.1]) == 1.0)
    check("D2 reversed", bt.auc_mann_whitney([1, 0], [0.1, 0.9]) == 0.0)
    check("D3 ties", bt.auc_mann_whitney([1, 0], [0.5, 0.5]) == 0.5)
    check("D4 endpoints int", bt.auc_mann_whitney([0, 1], [0, 1]) == 1.0)
    check("D5 multi", bt.auc_mann_whitney([1, 0, 1], [0.9, 0.1, 0.8]) == 1.0)
    check("D6 object-float lawful", bt.auc_mann_whitney([1, 0], np.array([0.9, 0.1], dtype=object)) == 1.0)
    check("D7 numeric Categorical lawful", bt.auc_mann_whitney([1, 0], pd.Categorical([0.9, 0.1])) == 1.0
          and bt.auc_mann_whitney(pd.Categorical([1, 0]), [0.9, 0.1]) == 1.0)
    check("D8 numeric Series lawful", bt.auc_mann_whitney(pd.Series([1, 0]), pd.Series([0.9, 0.1])) == 1.0)
    check("D9 empty NaN", np.isnan(bt.auc_mann_whitney([], [])))
    check("D10 single-class NaN", np.isnan(bt.auc_mann_whitney([1, 1], [0.9, 0.1])))


def test_legacy_screens():
    check("E1 SF-BL-9853 non-binary y", raises_tag(lambda: bt.auc_mann_whitney([2, 0], [0.5, 0.5]), "SF-BL-9853"))
    check("E2 shape mismatch", raises_tag(lambda: bt.auc_mann_whitney([1, 0], [0.5]), "shapes differ"))
    check("E3 SF-BL-10983 brier bool-p tag kept", raises_tag(lambda: bt.brier_score([1], [True]), "SF-BL-10983"))
    check("E4 SF-BL-11059 brier string tag kept", raises_tag(lambda: bt.brier_score([1], ["0.5"]), "SF-BL-11059"))
    check("E5 evaluate lawful scorecard", bt.evaluate_forecasts([1, 0], [0.9, 0.1])["auc"] == 1.0)


def test_scope():
    check("F1 _check still clips", bt._check([1], [1.5])[1][0] == 1 - bt.EPS)
    check("F2 brier bool-y legacy (other scorers untouched)", bt.brier_score([True], [0.5]) == 0.25)
    check("F3 siblings lawful", bt.brier_score([1, 0], [0.9, 0.1]) == 0.009999999999999998
          and bt.log_score([1, 0], [0.9, 0.1]) == 0.10536051565782628)


ALL_TESTS = [test_kill_criteria, test_string_forms, test_bool_forms,
             test_lawful_exact, test_legacy_screens, test_scope]

if __name__ == "__main__":
    for t in ALL_TESTS:
        t()
    n_pass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{n_pass}/{len(RESULTS)} PASS  module={getattr(bt, '__file__', 'package')}")
    sys.exit(0 if n_pass == len(RESULTS) else 1)
