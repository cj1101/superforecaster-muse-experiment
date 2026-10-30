"""
test_backtest_bool_entrypoints.py — SF-BL-11017 verification.

Pre-registered done/kill criteria (TASK.md):
  "Pass if evaluate_forecasts, calibration_table and auc_mann_whitney
   each raise a named ValueError on any bool probability while lawful
   0/1 panels stay bit-identical; kill if [True, False] still returns
   a perfect evaluate_forecasts scorecard or AUC 1.0."

Bug: the shared _check float-converts bools before scoring, so
evaluate_forecasts([1, 0], [True, False]) returned a perfect
scorecard (Brier 9.99e-31, log 9.99e-16, ECE 9.99e-16, AUC 1.0) and
auc_mann_whitney([0, 1], [False, True]) returned 1.0.
calibration_table already raised via SF-BL-10804's screen.

Fix (backtest.py, both live copies): _reject_bool_probabilities
(SF-BL-10983 helper) now runs on the RAW p in evaluate_forecasts
(after the string screen, before _validate_n_bins / range screen /
_check) and as the first statement of auc_mann_whitney, raising
"<scorer>: forecasts p must not contain boolean values; got boolean
value(s) in p (a bool is not a valid probability) (SF-BL-10983)".
Lawful panels bit-identical: pre/post SHA-256 over lawful panels +
legacy error fingerprints is
8754585aae708b81be1e9e81f2918c24b3a586e75485c3264a2b20f3e309214c
both ways (lawful_baseline.py in the task work dir).

Run:  BT_PATH=<path-to-backtest.py> python3 test_backtest_bool_entrypoints.py
      (default: the canonical package module; also pytest-collectable)
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_bt_path = os.environ.get("BT_PATH") or (sys.argv[1] if len(sys.argv) > 1 and os.path.basename(sys.argv[1]) == "backtest.py" else None)
if _bt_path:
    _spec = importlib.util.spec_from_file_location("bt_under_test_11017", _bt_path)
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


def raises_named(fn, scorer):
    try:
        fn()
    except ValueError as e:
        m = str(e)
        return m.startswith(f"{scorer}:") and "boolean" in m and "p" in m
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


ENTRY = [
    ("evaluate_forecasts", lambda p: bt.evaluate_forecasts([1, 0], p)),
    ("calibration_table", lambda p: bt.calibration_table([1, 0], p)),
    ("auc_mann_whitney", lambda p: bt.auc_mann_whitney([0, 1], p)),
]


def test_titled_repros():
    check("A1 evaluate titled", raises_named(lambda: bt.evaluate_forecasts([1, 0], [True, False]), "evaluate_forecasts"))
    check("A2 table titled", raises_named(lambda: bt.calibration_table([1, 0], [True, False]), "calibration_table"))
    check("A3 auc titled", raises_named(lambda: bt.auc_mann_whitney([0, 1], [False, True]), "auc_mann_whitney"))


def test_bool_forms():
    forms = [
        ("list bool", [True, False]),
        ("mixed float+bool", [0.9, True]),
        ("tuple bool", (False, True)),
        ("np bool array", np.array([True, False])),
        ("np.bool_ list", [np.bool_(True), np.bool_(False)]),
        ("object array bool", np.array([0.1, True], dtype=object)),
        ("object list nested", [[True], [False]]),
        ("pd Series bool", pd.Series([True, False])),
        ("pd Index bool", pd.Index([True, False])),
        ("pd Series object bool", pd.Series([0.1, True], dtype=object)),
    ]
    for scorer, call in ENTRY:
        for fname, p in forms:
            check(f"B {scorer} {fname}", raises_named(lambda c=call, q=p: c(q), scorer))
    check("B evaluate scalar True", raises_named(lambda: bt.evaluate_forecasts([1], True), "evaluate_forecasts"))
    check("B auc scalar panel", raises_named(lambda: bt.auc_mann_whitney([0, 1], True), "auc_mann_whitney"))


def test_kill_criterion():
    def returns_value(fn):
        try:
            fn(); return True
        except ValueError:
            return False
    check("C1 evaluate [True,False] returns nothing", not returns_value(lambda: bt.evaluate_forecasts([1, 0], [True, False])))
    check("C2 auc [False,True] returns nothing", not returns_value(lambda: bt.auc_mann_whitney([0, 1], [False, True])))


def test_lawful_exact():
    sc = bt.evaluate_forecasts([1, 0], [1, 0])
    check("D1 evaluate lawful endpoint exact", sc["brier"] == 9.992010415851716e-31 and sc["auc"] == 1.0 and sc["mean_pred"] == 0.5 and sc["n"] == 2)
    sc2 = bt.evaluate_forecasts([1, 0], [0.9, 0.1])
    check("D2 evaluate lawful exact", sc2["brier"] == 0.009999999999999998 and sc2["log_score"] == 0.10536051565782628 and sc2["ece"] == 0.09999999999999999)
    check("D3 auc lawful exact", bt.auc_mann_whitney([0, 1], [0, 1]) == 1.0 and bt.auc_mann_whitney([1, 0, 1], [0.9, 0.1, 0.8]) == 1.0 and bt.auc_mann_whitney([0, 1], [0.8, 0.2]) == 0.0)
    tab = bt.calibration_table([1, 0], [0.9, 0.1])
    nz = tab[tab["n"] > 0]
    check("D4 table lawful exact", tab.shape == (10, 6) and int(tab["n"].sum()) == 2
          and float(nz.iloc[0]["mean_pred"]) == 0.1 and float(nz.iloc[1]["mean_pred"]) == 0.9)
    check("D5 lawful int p accepted", bt.evaluate_forecasts([1, 0], [1, 0])["hit_rate"] == 1.0
          and bt.auc_mann_whitney(np.array([0, 1]), np.array([0, 1])) == 1.0)
    check("D6 empty panel legacy", np.isnan(bt.auc_mann_whitney([], [])))


def test_legacy_screens():
    check("E1 SF-BL-11018 evaluate range", raises_tag(lambda: bt.evaluate_forecasts([1, 0], [1.5, 0.5]), "SF-BL-11018"))
    check("E2 SF-BL-11059 evaluate string", raises_tag(lambda: bt.evaluate_forecasts([1, 0], ["0.9", "0.1"]), "SF-BL-11059"))
    check("E3 SF-BL-9791 n_bins", raises_tag(lambda: bt.evaluate_forecasts([1, 0], [0.5, 0.5], 0), "SF-BL-9791"))
    check("E4 SF-BL-10804 table range", raises_tag(lambda: bt.calibration_table([1], [1.5]), "SF-BL-10804"))
    check("E5 SF-BL-9853 auc non-binary y", raises_tag(lambda: bt.auc_mann_whitney([2, 0], [0.5, 0.5]), "SF-BL-9853"))
    check("E6 SF-BL-10983 brier bool still", raises_named(lambda: bt.brier_score([1], [True]), "brier_score"))
    check("E7 shape mismatch still ValueError", raises_tag(lambda: bt.evaluate_forecasts([1, 0], [0.5]), "shapes differ"))


def test_scope():
    check("F1 bool y legacy via _check path (auc)", bt.auc_mann_whitney([False, True], [0.2, 0.8]) == 1.0)
    y2, p2 = bt._check([1, 0], [1, 0])
    check("F2 _check still clips", abs(float(p2[0]) - (1 - bt.EPS)) < 1e-18 and abs(float(p2[1]) - bt.EPS) < 1e-18)


ALL_TESTS = [test_titled_repros, test_bool_forms, test_kill_criterion,
             test_lawful_exact, test_legacy_screens, test_scope]

if __name__ == "__main__":
    for t in ALL_TESTS:
        t()
    n_pass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{n_pass}/{len(RESULTS)} PASS  module={getattr(bt, '__file__', 'package')}")
    sys.exit(0 if n_pass == len(RESULTS) else 1)
