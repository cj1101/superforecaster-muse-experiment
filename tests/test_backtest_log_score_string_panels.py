"""SF-BL-11392 conformance: direct log_score string/object panels.
Run as a script: python3 test_backtest_log_score_string_panels.py [path-to-backtest.py]
Without an argument it imports the package copy. Prints PASS/FAIL per check.
"""
import importlib.util, sys
import numpy as np, pandas as pd

if len(sys.argv) > 1:
    spec = importlib.util.spec_from_file_location("bt_suite", sys.argv[1])
    bt = importlib.util.module_from_spec(spec); sys.modules[spec.name] = bt; spec.loader.exec_module(bt)
    src = open(sys.argv[1]).read()
else:
    sys.path.insert(0, "/home/hatch/workspace/superforecaster")
    from superforecaster import backtest as bt
    src = open(bt.__file__).read()

log_score = bt.log_score
results = []
def check(label, cond):
    results.append((label, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'} {label}")

def raises_named(y, p, panel):
    try:
        v = log_score(y, p)
        return False, f"returned {v!r}"
    except ValueError as e:
        m = str(e)
        return (m.startswith("log_score:") and "string" in m and "SF-BL-11392" in m
                and f"in {panel}" in m), m
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

# A. Kill + naming
ok, m = raises_named([1, 0], ['0.5', '0.5'], 'p'); check(f"A1 kill form raises naming log_score/p/11392 :: {m}", ok)
ok, m = raises_named(['1', '0'], [0.5, 0.5], 'y'); check(f"A2 string-y raises naming y :: {m}", ok)
ok, m = raises_named(['1', '0'], ['0.5', '0.5'], 'y'); check(f"A3 string-both names y first :: {m}", ok)
try:
    log_score([1, 0], ['0.5', '0.5']); check("A4 kill never returns 0.6931", False)
except ValueError:
    check("A4 kill never returns 0.6931", True)

# B. Diverse string / object-dtype forms
forms = [
    ("tuple_p", [1,0], ('0.5','0.5'), 'p'), ("mixed_p", [1,0], [0.5,'0.5'], 'p'),
    ("scalar_p", 1, '0.5', 'p'), ("scalar_y", '1', 0.5, 'y'),
    ("ndarray_U_p", [1,0], np.array(['0.5','0.5']), 'p'),
    ("ndarray_S_p", [1,0], np.array([b'0.5',b'0.5']), 'p'),
    ("object_str_p", [1,0], np.array(['0.5','0.5'],dtype=object), 'p'),
    ("object_str_y", np.array(['1','0'],dtype=object), [0.5,0.5], 'y'),
    ("object_mixed_p", [1,0], np.array([0.5,'0.5'],dtype=object), 'p'),
    ("series_object_p", [1,0], pd.Series(['0.5','0.5']), 'p'),
    ("series_stringdtype_p", [1,0], pd.Series(['0.5','0.5'],dtype='string'), 'p'),
    ("index_p", [1,0], pd.Index(['0.5','0.5']), 'p'),
    ("categorical_p", [1,0], pd.Categorical(['0.5','0.5']), 'p'),
    ("categorical_y", pd.Categorical(['1','0']), [0.5,0.5], 'y'),
    ("category_series_p", [1,0], pd.Series(pd.Categorical(['0.5','0.5'])), 'p'),
    ("empty_U_p", [], np.array([],dtype='<U1'), 'p'),
    ("nonnumeric_p", [1,0], ['high','low'], 'p'),
    ("bytes_list_p", [1,0], [b'0.5',b'0.5'], 'p'),
    ("nested_p", [1,0], [['0.5'],['0.5']], 'p'),
    ("np_str_scalar_p", [1,0], np.str_('0.5'), 'p'),
]
for label, y, p, panel in forms:
    ok, m = raises_named(y, p, panel); check(f"B-{label} :: {m[:80]}", ok)

# C. Not the _check backstop
try:
    log_score([1,0], ['0.5','0.5'])
    check("C1 message does not start with _check:", False)
except ValueError as e:
    check("C1 message does not start with _check:", not str(e).startswith("_check:"))

# D. Numeric controls exact
check("D1 control 0.6931471805599453", log_score([1,0],[0.5,0.5]) == 0.6931471805599453)
check("D2 floats 0.10536051565782628", log_score([1.0,0.0],[0.9,0.1]) == 0.10536051565782628)
check("D3 endpoints", log_score([1,0],[1.0,0.0]) == 9.992007221626415e-16)
check("D4 larger panel", log_score([1,0,1,1,0],[0.8,0.2,0.6,0.9,0.4]) == 0.31465977316364546)

# E. No false positives
check("E1 object-float lawful", log_score(np.array([1.,0.],dtype=object), np.array([0.5,0.5],dtype=object)) == 0.6931471805599453)
check("E2 numeric categorical p lawful", log_score([1,0], pd.Categorical([0.5,0.5])) == 0.6931471805599453)
check("E3 numeric categorical y lawful", log_score(pd.Categorical([1,0]), [0.5,0.5]) == 0.6931471805599453)
check("E4 Series lawful", log_score(pd.Series([1,0]), pd.Series([0.5,0.5])) == 0.6931471805599453)
check("E5 int panels lawful", log_score([1,0],[1,0]) == 9.992007221626415e-16)

# F. Legacy verdicts preserved
def raises_with(y, p, needle):
    try:
        log_score(y, p); return False
    except ValueError as e:
        return needle in str(e)
    except Exception:
        return False
check("F1 bool p still SF-BL-10983", raises_with([1],[True], "SF-BL-10983"))
check("F2 bool p not misnamed as string", raises_with([1],[True], "boolean"))
check("F3 range still SF-BL-10071", raises_with([1],[1.5], "SF-BL-10071"))
check("F4 empty still SF-BL-10071", raises_with([],[], "empty panel"))
check("F5 nonfinite still refused", raises_with([0,1],[float('nan'),0.9], "non-finite"))
check("F6 binary-y still SF-BL-9853", raises_with([2,0],[0.5,0.5], "SF-BL-9853"))
check("F7 shape mismatch still raises", raises_with([1,0],[0.5], "shapes differ"))
check("F8 string+bool precedence: string named", raises_named([1,0], ['0.5', True], 'p')[0])

# G. Source check
i = src.find("def log_score"); j = src.find("def _validate_n_bins", i)
body = src[i:j]
check("G1 log_score calls _reject_string_panels under own name", '_reject_string_panels(y, p, "log_score"' in body)
check("G2 string screen precedes bool screen", body.find('_reject_string_panels(y, p, "log_score"') < body.find('_reject_bool_probabilities(p, "log_score")'))
check("G3 string screen precedes _validate_log_score_panel", body.find("_reject_string_panels") < body.find("_validate_log_score_panel(y, p)"))

npass = sum(1 for _, c in results if c)
print(f"SUMMARY {npass}/{len(results)} PASS")
sys.exit(0 if npass == len(results) else 1)
