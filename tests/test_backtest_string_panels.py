"""Conformance suite for SF-BL-11059 — string panels in backtest.py.

Run per copy:  python3 test_backtest_string_panels.py <path-to-backtest.py>
Or via package: python3 test_backtest_string_panels.py  (canonical import)
Sections: A string p raises (brier) · B string y raises (brier) ·
C evaluate_forecasts raises + kill criterion · D string forms ·
E lawful panels exact values · F legacy behaviour preserved ·
G categorical string panels raise / numeric categoricals lawful
(round 2, G1).
"""
import importlib.util
import math
import sys

import numpy as np
import pandas as pd

FAILS = []


def load(path):
    if path:
        spec = importlib.util.spec_from_file_location("bt_suite", path)
        m = importlib.util.module_from_spec(spec)
        sys.modules["bt_suite"] = m
        spec.loader.exec_module(m)
        return m
    sys.path.insert(0, "/home/hatch/workspace/superforecaster")
    from superforecaster import backtest
    return backtest


def check(label, cond):
    print(("PASS " if cond else "FAIL ") + label)
    if not cond:
        FAILS.append(label)


def raises_naming(fn, arg):
    """True iff fn raises ValueError naming `arg` ('y' or 'p') + the task tag."""
    try:
        fn()
    except ValueError as e:
        msg = str(e)
        return f"in {arg} " in msg and "SF-BL-11059" in msg and "string" in msg
    except Exception:  # noqa: BLE001
        return False
    return False


def main():
    bt = load(sys.argv[1] if len(sys.argv) > 1 else None)

    # A — string forecasts raise in brier_score, naming p
    check("A1 brier [1],['0.5']", raises_naming(lambda: bt.brier_score([1], ['0.5']), "p"))
    check("A2 brier ['0.9','0.1']", raises_naming(lambda: bt.brier_score([1, 0], ['0.9', '0.1']), "p"))
    check("A3 brier mixed [0.9,'0.1']", raises_naming(lambda: bt.brier_score([1, 0], [0.9, '0.1']), "p"))
    check("A4 brier ['abc'] named (not bare numpy error)",
          raises_naming(lambda: bt.brier_score([1], ['abc']), "p"))
    check("A5 brier tuple ('0.5',)", raises_naming(lambda: bt.brier_score([1], ('0.5',)), "p"))
    check("A6 brier nested [['0.5']]", raises_naming(lambda: bt.brier_score([1], [['0.5']]), "p"))

    # B — string outcomes raise in brier_score, naming y
    check("B1 brier ['1'],[0.5]", raises_naming(lambda: bt.brier_score(['1'], [0.5]), "y"))
    check("B2 brier ['1','0']", raises_naming(lambda: bt.brier_score(['1', '0'], [0.9, 0.1]), "y"))
    check("B3 brier both string -> y named first",
          raises_naming(lambda: bt.brier_score(['1'], ['0.5']), "y"))

    # C — evaluate_forecasts raises; kill criterion
    check("C1 KILL evaluate([1,0],['0.9','0.1']) raises naming p",
          raises_naming(lambda: bt.evaluate_forecasts([1, 0], ['0.9', '0.1']), "p"))
    check("C2 evaluate string y raises naming y",
          raises_naming(lambda: bt.evaluate_forecasts(['1', '0'], [0.9, 0.1]), "y"))
    try:
        r = bt.evaluate_forecasts([1, 0], ['0.9', '0.1'])
        check("C3 kill: no scorecard returned", False)
    except ValueError:
        check("C3 kill: no scorecard returned", True)

    # D — string forms: numpy / pandas / object / bytes / scalar
    check("D1 np <U p", raises_naming(lambda: bt.brier_score([1, 0], np.array(['0.9', '0.1'])), "p"))
    check("D2 np <U y", raises_naming(lambda: bt.brier_score(np.array(['1', '0']), [0.9, 0.1]), "y"))
    check("D3 np <S bytes p", raises_naming(lambda: bt.brier_score([1, 0], np.array([b'0.9', b'0.1'])), "p"))
    check("D4 object array str", raises_naming(lambda: bt.brier_score([1, 0], np.array(['0.9', 0.1], dtype=object)), "p"))
    check("D5 object array floats lawful (no false positive)",
          bt.brier_score([1, 0], np.array([0.9, 0.1], dtype=object)) == bt.brier_score([1, 0], [0.9, 0.1]))
    check("D6 pd Series str p", raises_naming(lambda: bt.brier_score([1, 0], pd.Series(['0.9', '0.1'])), "p"))
    check("D7 pd Series string-dtype p", raises_naming(lambda: bt.brier_score([1, 0], pd.Series(['0.9', '0.1'], dtype='string')), "p"))
    check("D8 pd Series str y", raises_naming(lambda: bt.brier_score(pd.Series(['1', '0']), [0.9, 0.1]), "y"))
    check("D9 pd Index str p", raises_naming(lambda: bt.evaluate_forecasts([1, 0], pd.Index(['0.9', '0.1'])), "p"))
    check("D10 empty <U dtype p raises (string-dtype panel)",
          raises_naming(lambda: bt.brier_score([], np.array([], dtype='<U3')), "p"))
    check("D11 bytes element", raises_naming(lambda: bt.brier_score([1], [b'0.5']), "p"))
    check("D12 scalar str p", raises_naming(lambda: bt.brier_score([1], '0.5'), "p"))
    check("D13 np.str_ element", raises_naming(lambda: bt.brier_score([1], [np.str_('0.5')]), "p"))
    check("D14 evaluate np <U p", raises_naming(lambda: bt.evaluate_forecasts([1, 0], np.array(['0.9', '0.1'])), "p"))
    check("D15 _check funnel backstop names p",
          raises_naming(lambda: bt._check([1], ['0.5']), "p"))

    # E — lawful panels: exact pre-fix values
    check("E1 lawful brier exact", bt.brier_score([1, 0], [0.9, 0.1]) == 0.009999999999999998)
    check("E2 lawful endpoint brier", bt.brier_score([1, 0], [1, 0]) == 9.992010415851716e-31)
    sc = bt.evaluate_forecasts([1, 0], [0.9, 0.1])
    check("E3 lawful scorecard", sc["n"] == 2 and sc["brier"] == 0.009999999999999998
          and sc["log_score"] == 0.10536051565782628 and sc["auc"] == 1.0)
    check("E4 int panels lawful", bt.brier_score([1, 0], [1, 0]) == bt.brier_score([1.0, 0.0], [1.0, 0.0]))
    check("E5 numeric pd Series lawful",
          bt.brier_score(pd.Series([1, 0]), pd.Series([0.9, 0.1])) == 0.009999999999999998)
    check("E6 empty lawful panel NaN", math.isnan(bt.brier_score([], [])))

    # F — legacy behaviour preserved
    def raises_tag(fn, tag):
        try:
            fn()
        except ValueError as e:
            return tag in str(e)
        except Exception:  # noqa: BLE001
            return False
        return False
    check("F1 SF-BL-10940 range still raises", raises_tag(lambda: bt.brier_score([1], [1.5]), "SF-BL-10940"))
    check("F2 SF-BL-10983 bool p still raises", raises_tag(lambda: bt.brier_score([1], [True]), "SF-BL-10983"))
    check("F3 SF-BL-9853 non-binary y still raises", raises_tag(lambda: bt.brier_score([2, 0], [0.5, 0.5]), "SF-BL-9853"))
    check("F4 SF-BL-11018 evaluate range still raises", raises_tag(lambda: bt.evaluate_forecasts([1, 0], [1.5, 0.5]), "SF-BL-11018"))
    check("F5 SF-BL-9791 n_bins still raises", raises_tag(lambda: bt.evaluate_forecasts([1, 0], [0.5, 0.5], 0), "SF-BL-9791"))
    check("F6 shape mismatch still raises",
          raises_tag(lambda: bt.brier_score([1, 0], [0.5]), "shapes differ") or
          _shape_ok(bt))
    check("F7 sibling scorers lawful via _check (log/ECE/AUC unchanged)",
          bt.log_score([1, 0], [0.9, 0.1]) == 0.10536051565782628
          and bt.auc_mann_whitney([1, 0], [0.9, 0.1]) == 1.0)

    # G — round 2 (G1): categorical string panels raise; numeric lawful
    check("G1 brier Categorical str p", raises_naming(lambda: bt.brier_score([1, 0], pd.Categorical(['0.9', '0.1'])), "p"))
    check("G2 brier Categorical str y", raises_naming(lambda: bt.brier_score(pd.Categorical(['1', '0']), [0.9, 0.1]), "y"))
    check("G3 brier Series category str p", raises_naming(lambda: bt.brier_score([1, 0], pd.Series(['0.9', '0.1'], dtype='category')), "p"))
    check("G4 brier Index category str p", raises_naming(lambda: bt.brier_score([1, 0], pd.Index(['0.9', '0.1'], dtype='category')), "p"))
    check("G5 evaluate Series category str p (kill form)", raises_naming(lambda: bt.evaluate_forecasts([1, 0], pd.Series(['0.9', '0.1'], dtype='category')), "p"))
    check("G6 evaluate Categorical str p (kill form)", raises_naming(lambda: bt.evaluate_forecasts([1, 0], pd.Categorical(['0.9', '0.1'])), "p"))
    check("G7 brier CategoricalIndex str p", raises_naming(lambda: bt.brier_score([1, 0], pd.CategoricalIndex(['0.9', '0.1'])), "p"))
    check("G8 brier Categorical bytes p", raises_naming(lambda: bt.brier_score([1, 0], pd.Categorical([b'0.9', b'0.1'])), "p"))
    check("G9 brier Series category str y", raises_naming(lambda: bt.brier_score(pd.Series(['1', '0'], dtype='category'), [0.9, 0.1]), "y"))
    check("G10 evaluate Categorical str y", raises_naming(lambda: bt.evaluate_forecasts(pd.Categorical(['1', '0']), [0.9, 0.1]), "y"))
    check("G11 empty Categorical str categories raises",
          raises_naming(lambda: bt.brier_score([], pd.Categorical([], categories=['0.9', '0.1'])), "p"))
    check("G12 _check backstop Categorical str p", raises_naming(lambda: bt._check([1, 0], pd.Categorical(['0.9', '0.1'])), "p"))
    check("G13 brier list containing Categorical str p", raises_naming(lambda: bt.brier_score([1], [pd.Categorical(['0.5'])]), "p"))
    check("G14 lawful Categorical numeric p exact",
          bt.brier_score([1, 0], pd.Categorical([0.9, 0.1])) == 0.009999999999999998)
    check("G15 lawful Categorical numeric y exact",
          bt.brier_score(pd.Categorical([1, 0]), [0.9, 0.1]) == 0.009999999999999998)
    check("G16 lawful Series category numeric p exact",
          bt.brier_score([1, 0], pd.Series([0.9, 0.1], dtype='category')) == 0.009999999999999998)
    check("G17 lawful Index category numeric p exact",
          bt.brier_score([1, 0], pd.Index([0.9, 0.1], dtype='category')) == 0.009999999999999998)
    check("G18 lawful Categorical int p exact",
          bt.brier_score([1, 0], pd.Categorical([1, 0])) == 9.992010415851716e-31)
    sc_cat = bt.evaluate_forecasts(pd.Categorical([1, 0]), pd.Categorical([0.9, 0.1]))
    check("G19 lawful evaluate Categorical numeric scorecard",
          sc_cat["brier"] == 0.009999999999999998 and sc_cat["log_score"] == 0.10536051565782628 and sc_cat["auc"] == 1.0)
    check("G20 lawful evaluate Series category numeric scorecard",
          bt.evaluate_forecasts([1, 0], pd.Series([0.9, 0.1], dtype='category'))["brier"] == 0.009999999999999998)

    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + str(FAILS)}")
    return 1 if FAILS else 0


def _shape_ok(bt):
    try:
        bt.brier_score([1, 0], [0.5])
    except ValueError as e:
        return "shapes differ" in str(e)
    return False


if __name__ == "__main__":
    sys.exit(main())
