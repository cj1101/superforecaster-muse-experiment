"""
test_backtest_block_bootstrap_types.py — SF-BL-11096 verification.

Pre-registered done/kill criteria (TASK.md):
  "guard raises ValueError naming the argument on any bool, string or
   non-integer-float block/n_boot and any bool or string seed, with
   integer sizing and integer-seed results bit-identical; kill if
   block=True returns a bootstrap dict, or n_boot='10'/block=2.5/
   seed='0' raises a bare TypeError instead of the named ValueError."

Bug: block_bootstrap_brier_diff never type-validated block / n_boot /
seed. On the 40-row fixture in probe.py (work dir), pre-fix:
  block=True  RETURNED a dict (silently block=1; diff 0.2268 here,
              0.225 on the filed fixture)
  seed=True / seed=False / seed=None silently RETURNED dicts
  n_boot='10' TypeError: 'str' object cannot be interpreted as an integer
  block=2.5   TypeError: slice indices must be integers ...
  seed='0'    TypeError: SeedSequence expects int or sequence of ints ...
  block=0     ZeroDivisionError;  n_boot=0  IndexError
(repro_before_*.txt in the task work dir.)

Fix (backtest.py, both live copies): _validate_block_bootstrap_args
screens block / n_boot / seed at the top of
block_bootstrap_brier_diff — integer TYPE required (int / np.integer;
bool / np.bool_ and every float rejected), block / n_boot >= 1,
seed >= 0 — raising
"block_bootstrap_brier_diff: <arg> must be a positive integer
(>= 1) / a non-negative integer; got <arg>=<repr> (...) (SF-BL-11096)".
Lawful integer calls are bit-identical: the pre/post SHA-256 over the
lawful probe cases is
bbc9f01a396ba08219e529cfbc1ac82adc6d7b1b909b87122c45ab08796fbbbd
both ways (lawful_baseline.json in the task work dir).

Run:  BT_PATH=<path-to-backtest.py> python3 \
          test_backtest_block_bootstrap_types.py
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


def fixture():
    """Deterministic 40-row fixture (same construction as probe.py)."""
    rng = np.random.default_rng(11096)
    y = (rng.random(40) > 0.5).astype(float)
    p_model = np.clip(y * 0.7 + rng.random(40) * 0.25, 0.01, 0.99)
    p_bench = np.full(40, 0.5)
    return y, p_model, p_bench


BASE = dict(block=5, n_boot=200, seed=0)


def named_error(arg, **override):
    """Call with BASE+override; require the named SF-BL-11096
    ValueError for `arg`. A bare TypeError (or any other exception,
    or a returned dict) is a failure, not a pass."""
    y, pm, pb = fixture()
    kw = {**BASE, **override}
    try:
        out = bt.block_bootstrap_brier_diff(y, pm, pb, **kw)
    except ValueError as e:
        msg = str(e)
        assert arg in msg and "SF-BL-11096" in msg \
            and "block_bootstrap_brier_diff" in msg, \
            f"unnamed ValueError: {msg}"
        return e
    except TypeError as e:
        raise AssertionError(f"bare TypeError for {override}: {e}")
    raise AssertionError(f"returned a dict for {override}: {out!r}")


# ---- A. Titled repros --------------------------------------------------------

def test_titled_repros():
    named_error("block", block=True)
    named_error("n_boot", n_boot="10")
    named_error("block", block=2.5)
    named_error("seed", seed="0")


# ---- B. block / n_boot type matrix -------------------------------------------

BAD_SIZING = [
    True, False, np.bool_(True), "20", "10", "", 2.5, 20.0, 0.5,
    float("nan"), float("inf"), float("-inf"), None, [5], (5,),
    np.float64(5.0), np.float64(2.5),
]

def test_sizing_type_matrix():
    for bad in BAD_SIZING:
        named_error("block", block=bad)
        named_error("n_boot", n_boot=bad)


# ---- C. seed type matrix -------------------------------------------------------

BAD_SEEDS = [
    True, False, np.bool_(True), "0", "42", "", 1.5, 0.0, -0.5,
    float("nan"), float("inf"), None, [0, 1], (0,), {"seed": 0},
    np.float64(3.0),
]

def test_seed_type_matrix():
    for bad in BAD_SEEDS:
        named_error("seed", seed=bad)


# ---- D. Value domain (named, not ZeroDivision/IndexError) ---------------------

def test_value_domain():
    named_error("block", block=0)
    named_error("block", block=-3)
    named_error("n_boot", n_boot=0)
    named_error("n_boot", n_boot=-5)
    named_error("seed", seed=-1)
    named_error("seed", seed=np.int64(-2))


# ---- E. Kill criterion ---------------------------------------------------------

def test_kill_criterion():
    y, pm, pb = fixture()
    for override in (dict(block=True), dict(n_boot=True),
                     dict(seed=True), dict(seed=None)):
        kw = {**BASE, **override}
        try:
            out = bt.block_bootstrap_brier_diff(y, pm, pb, **kw)
        except ValueError:
            continue
        raise AssertionError(f"returned a dict for {override} (kill): "
                             f"{out!r}")


# ---- F. Lawful integer calls bit-identical (values pinned pre-fix) -------------

def test_lawful_bit_identical():
    y, pm, pb = fixture()
    out = bt.block_bootstrap_brier_diff(y, pm, pb, **BASE)
    assert out == {"diff_brier_bench_minus_model": 0.22676354585395914,
                   "ci95_lo": 0.21936464179326481,
                   "ci95_hi": 0.2367072959353776,
                   "p_model_not_better": 0.0,
                   "block": 5, "n_boot": 200, "n": 40}, out
    out = bt.block_bootstrap_brier_diff(y, pm, pb, block=20, n_boot=500,
                                        seed=42)
    assert out["ci95_lo"] == 0.22705670616292303, out
    assert out["ci95_hi"] == 0.23406041568271743, out
    out = bt.block_bootstrap_brier_diff(y, pm, pb)  # signature defaults
    assert out["block"] == 20 and out["n_boot"] == 2000, out
    assert out["ci95_lo"] == 0.22713388090087464, out
    assert out["ci95_hi"] == 0.23406079435979096, out
    out = bt.block_bootstrap_brier_diff(y, pm, pb, block=1, n_boot=100,
                                        seed=7)
    assert out["ci95_lo"] == 0.22107035745974155, out
    # numpy integers are lawful integer types
    out = bt.block_bootstrap_brier_diff(
        y, pm, pb, block=np.int64(4), n_boot=np.int32(150),
        seed=np.int64(3))
    assert out["ci95_lo"] == 0.21926105652782824, out
    assert out["ci95_hi"] == 0.2364708853792678, out
    assert out["block"] == 4 and out["n_boot"] == 150, out
    # determinism: same integer seed twice -> identical dict
    a = bt.block_bootstrap_brier_diff(y, pm, pb, **BASE)
    b = bt.block_bootstrap_brier_diff(y, pm, pb, **BASE)
    assert a == b


# ---- G. Pre-existing behaviour unchanged ---------------------------------------

def test_preexisting_behaviour_unchanged():
    y, pm, pb = fixture()

    def raises_value(fn, *a, **k):
        try:
            fn(*a, **k)
        except ValueError:
            return True
        return False

    # SF-BL-11021 panel range screen still fires and names the panel
    bad_model = np.full(40, 1.5)
    try:
        bt.block_bootstrap_brier_diff(y, bad_model, pb, **BASE)
        raise AssertionError("1.5 panel returned a dict")
    except ValueError as e:
        assert "p_model" in str(e) and "SF-BL-11021" in str(e), str(e)
    # too-short series guard unchanged (lawful sizing)
    assert raises_value(bt.block_bootstrap_brier_diff,
                        y[:10], pm[:10], pb[:10], block=20, n_boot=10,
                        seed=0)
    # shape mismatch still raises
    assert raises_value(bt.block_bootstrap_brier_diff,
                        y, pm[:30], pb, **BASE)
    # sibling scorers untouched (values pinned by SF-BL-10983 suite)
    assert bt.brier_score([1, 0], [0.9, 0.1]) == 0.009999999999999998
    assert bt.log_score([1, 0], [0.9, 0.1]) == 0.10536051565782628
    # deflated p-value path (uses _validate_n_variants_tried) untouched
    assert bt.deflated_brier_pvalue(0.05, 3) > 0.05


def test_backtester_end_to_end():
    # WalkForwardBacktester forwards its (integer) bootstrap args to
    # block_bootstrap_brier_diff per benchmark — must keep working.
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
    assert "diff_brier_bench_minus_model" in boot, boot


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
