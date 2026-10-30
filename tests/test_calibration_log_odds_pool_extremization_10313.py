"""SF-BL-10313: calibration.log_odds_pool extremization finiteness guard.

Done criterion: a named ValueError naming extremization for
non-finite or non-positive values; finite positive outputs unchanged.
Kill iff extremization=NaN or inf returns any pooled probability
instead of raising.

Standalone (python3 test_sf_bl_10313.py) or under pytest.
Pristine reference = committed HEAD calibration.py via `git show`.
"""
import importlib.util
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster"))
from superforecaster.calibration import (  # noqa: E402
    extremized_pool, log_odds_pool, optimize_pool_weights)

REPO = os.path.expanduser("~/workspace/superforecaster")
PANELS = [[[0.7, 0.3]], [[0.7]], [[0.6, 0.8]], [[0.8, 0.8]], [[0.02, 0.98]]]
INVALID = [float("nan"), float("inf"), float("-inf"),
           np.float64("nan"), np.float64("inf"),
           0.0, -0.0, -1.0, -1e-9]
VALID = [0.05, 0.3, 0.5, 1.0, 1.25, 2.0, 4.0, 10.0, 1, 2]


def _load_pristine():
    src = subprocess.run(
        ["git", "-C", REPO, "show", "HEAD:superforecaster/calibration.py"],
        capture_output=True, text=True, check=True).stdout
    fd, p = tempfile.mkstemp(suffix=".py", prefix="cal_pristine_10313_")
    with os.fdopen(fd, "w") as f:
        f.write(src)
    try:
        spec = importlib.util.spec_from_file_location("cal_pristine_10313", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["cal_pristine_10313"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        os.unlink(p)


def _bits(x):
    return [struct.pack("<d", float(v)) for v in np.atleast_1d(np.asarray(x, dtype=float))]


def test_pristine_reproduces_task_bug():
    pre = _load_pristine()
    assert np.isnan(pre.log_odds_pool([[0.7, 0.3]], extremization=float("nan"))[0])
    assert pre.log_odds_pool([[0.7, 0.3]], extremization=float("inf"))[0] == 0.0
    assert pre.log_odds_pool([[0.7]], extremization=float("inf"))[0] == 1.0


def test_kill_criterion_nan_inf_raise_everywhere():
    for panel in PANELS:
        for a in (float("nan"), float("inf"), float("-inf")):
            try:
                out = log_odds_pool(panel, extremization=a)
            except ValueError as e:
                assert "extremization" in str(e), str(e)
            else:
                raise AssertionError(f"KILL: a={a} panel={panel} returned {out}")


def test_invalid_grid_raises_named_valueerror():
    for panel in PANELS:
        for a in INVALID:
            try:
                log_odds_pool(panel, extremization=a)
            except ValueError as e:
                assert "extremization" in str(e), str(e)
                assert "finite" in str(e), str(e)
            except Exception as e:  # noqa: BLE001
                raise AssertionError(f"a={a!r} raised {type(e).__name__}, not ValueError")
            else:
                raise AssertionError(f"a={a!r} panel={panel} did not raise")
    # weighted path too
    try:
        log_odds_pool([[0.7, 0.3]], weights=[0.5, 0.5], extremization=float("nan"))
    except ValueError as e:
        assert "extremization" in str(e)


def test_finite_positive_bit_identical_to_pristine():
    pre = _load_pristine()
    grid = [round(0.02 + 0.02 * i, 10) for i in range(49)]
    for p in grid:
        for panel in ([[p, 1 - p]], [[p]], [[p, 0.5, 1 - p]]):
            for a in VALID:
                got = log_odds_pool(panel, extremization=a)
                exp = pre.log_odds_pool(panel, extremization=a)
                assert _bits(got) == _bits(exp), (panel, a, got, exp)
    # known values
    assert float(log_odds_pool([[0.8, 0.8]], extremization=1.0)[0]) == 0.8
    assert abs(float(log_odds_pool([[0.8, 0.8]], extremization=2.0)[0])
               - 0.9411764705882353) < 1e-15


def test_alias_and_downstream_optimizer():
    assert extremized_pool is log_odds_pool
    try:
        extremized_pool([[0.7, 0.3]], extremization=float("inf"))
    except ValueError as e:
        assert "extremization" in str(e)
    else:
        raise AssertionError("alias did not raise on inf")
    rng = np.random.default_rng(0)
    P = rng.uniform(0.05, 0.95, size=(80, 3))
    y = (rng.uniform(size=80) < P.mean(axis=1)).astype(float)
    res = optimize_pool_weights(y, P, method="extremized")
    assert res["extremization"] is None or np.isfinite(res["extremization"]), res
    if res.get("pool_fn") is not None:
        out = res["pool_fn"](P)
        assert np.all(np.isfinite(out)), "pool_fn produced non-finite output"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
