"""Conformance suite for SF-BL-11467 — string panels in backtest.py's
brier_score / expected_calibration_error / calibration_table.

Run per copy:  python3 test_backtest_string_panels_ece_table.py <path-to-backtest.py>
Or via package: python3 test_backtest_string_panels_ece_table.py
Sections: A brier · B ECE · C calibration_table + kill · D forms x3 ·
E lawful exact · F legacy preserved · G no false positives.
A raise counts only if the ValueError names the SCORER itself, names
the panel ("in y " / "in p "), and contains "string" + "SF-BL-11467"
(the pre-fix `_check`-backstop messages fail the scorer-name check).
"""
import importlib.util
import math
import sys

import numpy as np
import pandas as pd

FAILS = []


def load(path):
    if path:
        spec = importlib.util.spec_from_file_location("bt_suite11467", path)
        m = importlib.util.module_from_spec(spec)
        sys.modules["bt_suite11467"] = m
        spec.loader.exec_module(m)
        return m
    sys.path.insert(0, "/home/hatch/workspace/superforecaster")
    from superforecaster import backtest
    return backtest


def check(label, cond):
    print(("PASS " if cond else "FAIL ") + label)
    if not cond:
        FAILS.append(label)


def raises_named(fn, scorer, panel):
    try:
        fn()
    except ValueError as e:
        msg = str(e)
        return (scorer in msg and f"in {panel} " in msg
                and "string" in msg and "SF-BL-11467" in msg)
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


SCORERS = ("brier_score", "expected_calibration_error", "calibration_table")


def call(bt, scorer, y, p):
    return getattr(bt, scorer)(y, p)


def main():
    bt = load(sys.argv[1] if len(sys.argv) > 1 else None)

    # A — brier_score string panels, incl. the task kill form
    check("A1 KILL brier(['1','0'],['0.9','0.1']) raises naming y",
          raises_named(lambda: bt.brier_score(['1', '0'], ['0.9', '0.1']), "brier_score", "y"))
    try:
        r = bt.brier_score(['1', '0'], ['0.9', '0.1'])
        check("A2 kill: no 0.01 returned", False)
    except ValueError:
        check("A2 kill: no 0.01 returned", True)
    check("A3 brier string p", raises_named(lambda: bt.brier_score([1, 0], ['0.9', '0.1']), "brier_score", "p"))
    check("A4 brier string y", raises_named(lambda: bt.brier_score(['1', '0'], [0.9, 0.1]), "brier_score", "y"))

    # B — expected_calibration_error string panels name ECE itself
    check("B1 ECE string both", raises_named(lambda: bt.expected_calibration_error(['1', '0'], ['0.9', '0.1']), "expected_calibration_error", "y"))
    check("B2 ECE string p", raises_named(lambda: bt.expected_calibration_error([1, 0], ['0.9', '0.1']), "expected_calibration_error", "p"))
    check("B3 ECE string y", raises_named(lambda: bt.expected_calibration_error(['1', '0'], [0.9, 0.1]), "expected_calibration_error", "y"))
    check("B4 ECE message is not the _check backstop",
          not _msg_starts_check(lambda: bt.expected_calibration_error([1, 0], ['0.9', '0.1'])))

    # C — calibration_table string panels name calibration_table itself
    check("C1 table string both", raises_named(lambda: bt.calibration_table(['1', '0'], ['0.9', '0.1']), "calibration_table", "y"))
    check("C2 table string p", raises_named(lambda: bt.calibration_table([1, 0], ['0.9', '0.1']), "calibration_table", "p"))
    check("C3 table string y", raises_named(lambda: bt.calibration_table(['1', '0'], [0.9, 0.1]), "calibration_table", "y"))
    try:
        r = bt.calibration_table(['1', '0'], ['0.9', '0.1'])
        check("C4 kill: no (10, 6) frame returned", False)
    except ValueError:
        check("C4 kill: no (10, 6) frame returned", True)

    # D — string forms x all three scorers
    forms_p = [
        ("list", ['0.9', '0.1']), ("tuple", ('0.9', '0.1')),
        ("mixed", [0.9, '0.1']), ("nested", [['0.9'], ['0.1']]),
        ("<U", np.array(['0.9', '0.1'])), ("<S", np.array([b'0.9', b'0.1'])),
        ("object", np.array(['0.9', '0.1'], dtype=object)),
        ("Series", pd.Series(['0.9', '0.1'])),
        ("Series-string", pd.Series(['0.9', '0.1'], dtype='string')),
        ("Index", pd.Index(['0.9', '0.1'])),
        ("Categorical", pd.Categorical(['0.9', '0.1'])),
        ("Series-category", pd.Series(['0.9', '0.1'], dtype='category')),
        ("CategoricalIndex", pd.CategoricalIndex(['0.9', '0.1'])),
    ]
    for s in SCORERS:
        for label, p in forms_p:
            check(f"D {s} p {label}", raises_named(lambda s=s, p=p: call(bt, s, [1, 0], p), s, "p"))
        check(f"D {s} y <U", raises_named(lambda s=s: call(bt, s, np.array(['1', '0']), [0.9, 0.1]), s, "y"))
        check(f"D {s} p empty <U dtype",
              raises_named(lambda s=s: call(bt, s, [], np.array([], dtype='<U3')), s, "p"))
        check(f"D {s} p scalar str",
              raises_named(lambda s=s: call(bt, s, [1], '0.5'), s, "p"))

    # E — lawful panels: exact values (task controls)
    check("E1 brier exact", bt.brier_score([1, 0], [0.9, 0.1]) == 0.009999999999999998)
    check("E2 ECE exact", bt.expected_calibration_error([1, 0], [0.9, 0.1]) == 0.09999999999999999)
    tab = bt.calibration_table([1, 0], [0.9, 0.1])
    check("E3 table (10, 6) + cells", tab.shape == (10, 6)
          and tab["n"].tolist() == [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
          and tab.loc[9, "mean_pred"] == 0.9 and tab.loc[9, "empirical_rate"] == 1.0
          and tab.loc[1, "mean_pred"] == 0.1 and tab.loc[1, "empirical_rate"] == 0.0)
    check("E4 brier endpoints", bt.brier_score([1, 0], [1, 0]) == 9.992010415851716e-31)
    check("E5 int panels", bt.brier_score([1, 0], [1, 0]) == bt.brier_score([1.0, 0.0], [1.0, 0.0]))
    check("E6 empty lawful brier NaN", math.isnan(bt.brier_score([], [])))
    check("E7 empty lawful ECE NaN", math.isnan(bt.expected_calibration_error([], [])))
    check("E8 empty lawful table (10, 6) zero-n",
          bt.calibration_table([], []).shape == (10, 6) and bt.calibration_table([], [])["n"].sum() == 0)

    # F — legacy behaviour preserved
    check("F1 10940 brier range", raises_tag(lambda: bt.brier_score([1], [1.5]), "SF-BL-10940"))
    check("F2 10983 brier bool", raises_tag(lambda: bt.brier_score([1], [True]), "SF-BL-10983"))
    check("F3 10983 ECE bool", raises_tag(lambda: bt.expected_calibration_error([1], [True]), "SF-BL-10983"))
    check("F4 9853 non-binary y (brier)", raises_tag(lambda: bt.brier_score([2, 0], [0.5, 0.5]), "SF-BL-9853"))
    check("F5 9853 non-binary y (table)", raises_tag(lambda: bt.calibration_table([2, 0], [0.5, 0.5]), "SF-BL-9853"))
    check("F6 9791 ECE n_bins=0", raises_tag(lambda: bt.expected_calibration_error([1, 0], [0.5, 0.5], 0), "SF-BL-9791"))
    check("F7 9791 table n_bins=0", raises_tag(lambda: bt.calibration_table([1, 0], [0.5, 0.5], 0), "SF-BL-9791"))
    check("F8 11059 tag retained (brier)", raises_tag(lambda: bt.brier_score([1], ['0.5']), "SF-BL-11059"))
    check("F9 shape mismatch (ECE)", raises_tag(lambda: bt.expected_calibration_error([1, 0], [0.5]), "shapes differ"))
    check("F10 evaluate scorecard lawful",
          bt.evaluate_forecasts([1, 0], [0.9, 0.1])["brier"] == 0.009999999999999998)

    # G — no false positives on lawful non-float panels
    check("G1 object floats lawful", bt.brier_score([1, 0], np.array([0.9, 0.1], dtype=object)) == 0.009999999999999998)
    check("G2 numeric Categorical lawful", bt.brier_score([1, 0], pd.Categorical([0.9, 0.1])) == 0.009999999999999998)
    check("G3 numeric category Series lawful (ECE)",
          bt.expected_calibration_error([1, 0], pd.Series([0.9, 0.1], dtype='category')) == 0.09999999999999999)
    check("G4 numeric pd Series lawful (table)",
          bt.calibration_table(pd.Series([1, 0]), pd.Series([0.9, 0.1])).shape == (10, 6))

    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + str(FAILS)}")
    return 1 if FAILS else 0


def _msg_starts_check(fn):
    try:
        fn()
    except ValueError as e:
        return str(e).startswith("_check:")
    return False


if __name__ == "__main__":
    sys.exit(main())
