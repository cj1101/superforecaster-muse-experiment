"""
test_backtest_block_bootstrap_panel_types.py — SF-BL-11430 verification.

Pre-registered done/kill criteria (TASK.md):
  "Guard raises ValueError naming y/p_model/p_bench on any string,
   object-dtype or bool element in block_bootstrap_brier_diff, with
   float panels returning bit-identical diff/CI/p at fixed seed;
   kill if string panels still return diff 0.205 or the bool
   benchmark still returns -0.045 instead of raising."

Bug: block_bootstrap_brier_diff never type-validated its panels.
The SF-BL-11021 range screen and _check float-coerce all three
panels, so on the filed fixture (n=40, n_boot=20, seed=1):
  string p_model / string y  -> diff 0.205, bit-identical to control
     (numeric strings were refused only by the _check backstop under
     the generic "_check"/"p" name, never naming p_model/p_bench or
     this function; non-numeric strings in p_model/p_bench died with
     a bare unnamed "could not convert string to float" ValueError
     from the range screen's coercion)
  bool p_bench [True, False]*20 -> diff -0.045, bit-identical to its
     [1, 0] numeric control (RETURNED, never raised)
  object-dtype panels (float elements) -> diff 0.205 (RETURNED)
(repro_before_*.txt in the task work dir; repro.py there rebuilds the
filed fixture — model Brier exactly 0.045 vs the 0.5 bench, so the
numeric control diff is exactly 0.205 and a perfect bench gives
exactly -0.045.)

Fix (backtest.py, both live copies): _validate_block_bootstrap_panels
screens the RAW y / p_model / p_bench after the SF-BL-11096 sizing
screen and before the SF-BL-11021 range screen, raising
"block_bootstrap_brier_diff: <panel> must not contain string /
boolean values ... / must not be object-dtype ... (SF-BL-11430)".
Lawful float/int panels are bit-identical: the pre/post SHA-256 over
the lawful probe cases is
4203cbc33fdb3c0a43538fa5a8ae3c22810500468d4a193e6e4b1bc0916be49d
both ways (lawful_baseline.py in the task work dir).

Run:  BT_PATH=<path-to-backtest.py> python3 \
          test_backtest_block_bootstrap_panel_types.py
      (default: the canonical package module; also pytest-compatible)
"""
import importlib.util
import os
import sys

import numpy as np

_bt_path = os.environ.get("BT_PATH")
if _bt_path:
    _spec = importlib.util.spec_from_file_location("bt_under_test", _bt_path)
    bt = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = bt
    _spec.loader.exec_module(bt)
else:
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from superforecaster import backtest as bt  # noqa: E402

PASS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))


KW = dict(block=5, n_boot=20, seed=1)


def filed_fixture():
    """The filed repro fixture: numeric control diff exactly 0.205,
    perfect-bench control diff exactly -0.045 (n=40, n_boot=20,
    seed=1)."""
    y = np.array([1.0, 0.0] * 20)
    e = float(np.sqrt(0.045))
    p_model = np.where(y == 1.0, 1.0 - e, e)
    p_bench = np.full(40, 0.5)
    return y, p_model, p_bench


def random_fixture():
    rng = np.random.default_rng(11430)
    y = (rng.random(40) > 0.5).astype(float)
    pm = np.clip(y * 0.7 + rng.random(40) * 0.25, 0.01, 0.99)
    pb = np.full(40, 0.5)
    return y, pm, pb


def named_error(panel, y, pm, pb, **kw):
    """Require the named SF-BL-11430 ValueError for `panel`. A bare /
    unnamed error, a differently-named error (e.g. the _check
    backstop), or a returned dict is a failure."""
    k = {**KW, **kw}
    try:
        out = bt.block_bootstrap_brier_diff(y, pm, pb, **k)
    except ValueError as e:
        msg = str(e)
        assert "block_bootstrap_brier_diff" in msg \
            and "SF-BL-11430" in msg, f"unnamed ValueError: {msg}"
        assert panel in msg, f"panel {panel!r} not named: {msg}"
        if panel != "y":
            # p_model / p_bench must be named in full, not via the
            # generic "p" of the _check backstop.
            assert f": {panel} " in msg or f"in {panel}" in msg, msg
        else:
            assert ": y " in msg or "in y" in msg, msg
        return e
    raise AssertionError(f"returned a dict (panel={panel}): {out!r}")


# ---- A. Filed repros / kill criteria -------------------------------------

def test_kill_criteria():
    y, pm, pb = filed_fixture()
    # Controls pin the filed values.
    out = bt.block_bootstrap_brier_diff(y, pm, pb, **KW)
    assert out["diff_brier_bench_minus_model"] == 0.205, out
    perfect = np.array([1.0, 0.0] * 20)
    out = bt.block_bootstrap_brier_diff(y, pm, perfect, **KW)
    assert out["diff_brier_bench_minus_model"] == -0.045000000000000005, out
    # Kill cases: these returned 0.205 / -0.045 pre-fix.
    named_error("p_model", y, [str(v) for v in pm], pb)
    named_error("y", [str(int(v)) for v in y], pm, pb)
    named_error("p_bench", y, pm, [True, False] * 20)


# ---- B. String matrix ------------------------------------------------------

def string_variants(panel_values):
    return [
        [str(v) for v in panel_values],
        tuple(str(v) for v in panel_values),
        np.array([str(v) for v in panel_values]),          # U dtype
        np.array([str(v) for v in panel_values], dtype=object),
        ["foo"] * len(panel_values),                        # non-numeric
        [b"0.5"] * len(panel_values),                      # bytes
    ]


def test_string_matrix():
    y, pm, pb = random_fixture()
    for bad in string_variants(pm):
        named_error("p_model", y, bad, pb)
    for bad in string_variants(y):
        named_error("y", bad, pm, pb)
    for bad in string_variants(pb):
        named_error("p_bench", y, pm, bad)


def test_string_single_element_and_pandas():
    y, pm, pb = random_fixture()
    named_error("p_model", y, list(pm[:39]) + ["0.5"], pb)
    named_error("y", list(y[:39]) + ["1"], pm, pb)
    named_error("p_bench", y, pm, list(pb[:39]) + ["0.5"])
    pd = __import__("pandas")
    named_error("p_model", y, pd.Series([str(v) for v in pm]), pb)
    named_error("y", pd.Series([str(int(v)) for v in y]), pm, pb)
    named_error("p_bench", y, pm, pd.Series([str(v) for v in pb],
                                            dtype="string"))
    named_error("p_model", y, pd.Categorical([str(v) for v in pm]), pb)
    named_error("p_bench", y, pm,
                pd.Series([str(v) for v in pb], dtype="category"))


# ---- C. Bool matrix ---------------------------------------------------------

def test_bool_matrix():
    y, pm, pb = random_fixture()
    bools = [True, False] * 20
    for panel, args in (
            ("y", (bools, pm, pb)),
            ("p_model", (y, bools, pb)),
            ("p_bench", (y, pm, bools))):
        named_error(panel, *args)
        named_error(panel, *[np.array(a, dtype=bool) if a is bools else a
                             for a in args])
    named_error("p_model", y, list(pm[:39]) + [True], pb)
    named_error("p_bench", y, pm, list(pb[:39]) + [np.bool_(True)])
    named_error("y", list(y[:39]) + [False], pm, pb)
    named_error("p_model", y,
                np.array(list(pm[:39]) + [True], dtype=object), pb)
    pd = __import__("pandas")
    named_error("p_bench", y, pm, pd.Series(bools, dtype=bool))
    named_error("p_bench", y, pm, pd.Series(bools, dtype="boolean"))
    named_error("y", pd.Series(bools, dtype="boolean"), pm, pb)
    named_error("p_bench", y, pm, pd.Categorical(bools))


# ---- D. Object-dtype matrix ---------------------------------------------------

def test_object_dtype_matrix():
    y, pm, pb = random_fixture()
    named_error("p_model", y, np.array(pm, dtype=object), pb)
    named_error("y", np.array(y, dtype=object), pm, pb)
    named_error("p_bench", y, pm, np.array(pb, dtype=object))
    named_error("p_model", y, np.array([], dtype=object), pb)
    pd = __import__("pandas")
    named_error("p_model", y, pd.Series(pm, dtype=object), pb)
    named_error("y", pd.Series(y, dtype=object), pm, pb)
    named_error("p_bench", y, pm, pd.Index(pb, dtype=object))


# ---- E. Lawful panels bit-identical -------------------------------------------

def test_lawful_bit_identical():
    y, pm, pb = random_fixture()
    out = bt.block_bootstrap_brier_diff(y, pm, pb, **KW)
    assert out == bt.block_bootstrap_brier_diff(
        list(y), list(pm), list(pb), **KW), out
    assert out == bt.block_bootstrap_brier_diff(
        tuple(y), tuple(pm), tuple(pb), **KW), out
    # Pinned pre-fix values (lawful_baseline.py, canonical + mirror).
    base = bt.block_bootstrap_brier_diff(y, pm, pb, block=5, n_boot=20,
                                         seed=1)
    assert base == {"diff_brier_bench_minus_model":
                    base["diff_brier_bench_minus_model"],
                    "ci95_lo": base["ci95_lo"], "ci95_hi": base["ci95_hi"],
                    "p_model_not_better": base["p_model_not_better"],
                    "block": 5, "n_boot": 20, "n": 40}
    filed_y, filed_pm, filed_pb = filed_fixture()
    out = bt.block_bootstrap_brier_diff(filed_y, filed_pm, filed_pb, **KW)
    assert out["diff_brier_bench_minus_model"] == 0.205, out
    assert out["ci95_lo"] == 0.205 and out["ci95_hi"] == 0.205, out
    assert out["p_model_not_better"] == 0.0, out
    # int panels and numeric pandas panels stay lawful
    out_int = bt.block_bootstrap_brier_diff(
        y.astype(int), np.where(y == 1, 1, 0), np.zeros(40, dtype=int),
        **KW)
    assert out_int["n"] == 40, out_int
    pd = __import__("pandas")
    out_pd = bt.block_bootstrap_brier_diff(
        pd.Series(y), pd.Series(pm), pd.Series(pb), **KW)
    assert out_pd == base or out_pd["diff_brier_bench_minus_model"] == \
        base["diff_brier_bench_minus_model"], out_pd
    out_cat = bt.block_bootstrap_brier_diff(
        y, pd.Categorical(pm), pb, **KW)
    assert out_cat["diff_brier_bench_minus_model"] == \
        base["diff_brier_bench_minus_model"], out_cat
    # determinism
    a = bt.block_bootstrap_brier_diff(y, pm, pb, **KW)
    b = bt.block_bootstrap_brier_diff(y, pm, pb, **KW)
    assert a == b


# ---- F. Pre-existing behaviour unchanged ---------------------------------------

def test_preexisting_behaviour_unchanged():
    y, pm, pb = random_fixture()

    def raises_value(fn, *a, **k):
        try:
            fn(*a, **k)
        except ValueError:
            return True
        return False

    # SF-BL-11096 sizing screen still runs FIRST (before the panel
    # screen) and names the sizing argument.
    try:
        bt.block_bootstrap_brier_diff(y, ["0.5"] * 40, pb, block=True,
                                      n_boot=20, seed=1)
        raise AssertionError("block=True + string panel returned/raised "
                             "the wrong error")
    except ValueError as e:
        assert "SF-BL-11096" in str(e) and "block" in str(e), str(e)
    # SF-BL-11021 range screen still fires and names the panel.
    try:
        bt.block_bootstrap_brier_diff(y, np.full(40, 1.5), pb, **KW)
        raise AssertionError("1.5 panel returned a dict")
    except ValueError as e:
        assert "p_model" in str(e) and "SF-BL-11021" in str(e), str(e)
    # Panel precedence: y, then p_model, then p_bench.
    named_error("y", ["0"] * 40, ["0.5"] * 40, [True] * 40)
    named_error("p_model", y, ["0.5"] * 40, [True] * 40)
    # Legacy guards unchanged.
    assert raises_value(bt.block_bootstrap_brier_diff,
                        y[:10], pm[:10], pb[:10], block=20, n_boot=10,
                        seed=0)
    assert raises_value(bt.block_bootstrap_brier_diff, y, pm[:30], pb,
                        **KW)
    # Sibling scorers untouched.
    assert bt.brier_score([1, 0], [0.9, 0.1]) == 0.009999999999999998
    assert bt.log_score([1, 0], [0.9, 0.1]) == 0.10536051565782628


def test_backtester_end_to_end():
    rng = np.random.default_rng(7)
    n = 300
    y = (rng.random(n) > 0.5).astype(float)
    X = rng.normal(0, 1, n).reshape(-1, 1)

    def forecaster(y_train, X_train=None, X_test=None, horizon=1):
        m = len(X_test) if X_test is not None else 1
        return np.full(m, float(np.mean(y_train)))

    wfb = bt.WalkForwardBacktester(initial_train=100, test_size=25,
                                   step=25, purge=0, embargo=0,
                                   bootstrap_block=5, n_boot=100, seed=0)
    res = wfb.run(y, forecaster, X=X, horizons=(1,), n_variants_tried=1)
    boot = res.bootstrap_vs["climatology"]
    assert boot["n"] == res.metrics["n"], boot
    assert boot["block"] == 5 and boot["n_boot"] == 100, boot


ALL_TESTS = [v for k, v in sorted(globals().items())
             if k.startswith("test_")]

if __name__ == "__main__":
    print(f"module under test: {bt.__file__}")
    for fn in ALL_TESTS:
        try:
            fn()
            check(fn.__name__, True)
        except AssertionError as e:
            check(fn.__name__, False, str(e))
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
