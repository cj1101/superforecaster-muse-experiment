"""
test_cli_pool.py — SF-BL-9544: `sf pool` CLI-boundary input validation.

Run:  python3 tests/test_cli_pool.py     (or: python3 -m pytest tests/test_cli_pool.py)
No pytest dependency: every check is a test_* function using assert, and
__main__ runs them all.

Before the fix, cmd_pool parsed --probs/--weights as bare floats and
pooled them raw: `--probs 1.5,0.6` exited 0 publishing trimmed_mean /
mean / median of 1.05 (a "probability" above 1 that a downstream edge
command would clip silently into a trade decision), and non-finite /
non-numeric / mis-weighted inputs died as uncaught tracebacks (exit 1,
no JSON), breaking the CLI's one-JSON-object / exit-2 contract. These
tests pin the fixed contract: every prob finite in [0,1], weights
finite / non-negative / length-matched / positive-sum, violations exit
2 with exactly one JSON error object and no pooled figure, and valid
inputs return the pre-fix pools unchanged.
"""
import json
import os as _os
import subprocess
import sys

CLI = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                    'superforecaster', 'cli.py')
PASS = []
POOL_KEYS = {"log_odds_pool", "trimmed_mean", "mean", "median"}


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def sf_pool_raw(*args):
    return subprocess.run([sys.executable, CLI, "pool", *args],
                          capture_output=True, text=True)


def assert_invalid(*args):
    r = sf_pool_raw(*args)
    assert r.returncode == 2, f"{args}: exit {r.returncode}, stdout={r.stdout!r}"
    try:
        payload = json.loads(r.stdout)  # exactly one JSON object
    except ValueError:
        raise AssertionError(f"{args}: stdout is not one JSON object: {r.stdout!r}")
    assert "error" in payload, payload
    assert not (POOL_KEYS & set(payload)), f"{args}: pooled figure leaked: {payload}"
    assert "Traceback" not in r.stderr, r.stderr


def assert_valid(expect, *args):
    r = sf_pool_raw(*args)
    assert r.returncode == 0, f"{args}: exit {r.returncode}, stderr={r.stderr!r}"
    payload = json.loads(r.stdout)
    for k, v in expect.items():
        assert payload[k] == v, f"{args}: {k}={payload[k]} != {v}"


# ---- invalid: probs ---------------------------------------------------------

def test_pool_prob_above_one_rejected():
    assert_invalid("--probs", "1.5,0.6")  # the filed fixture: pre-fix exit 0, pools 1.05


def test_pool_prob_negative_rejected():
    assert_invalid("--probs", "-0.1,0.6")


def test_pool_prob_nan_rejected():
    assert_invalid("--probs", "nan,0.6,0.7")  # the filed fixture


def test_pool_prob_inf_rejected():
    assert_invalid("--probs", "inf,0.6")
    assert_invalid("--probs", "0.6,-inf")


def test_pool_prob_non_numeric_rejected():
    assert_invalid("--probs", "abc,0.6")
    assert_invalid("--probs", "0.6,,0.7")
    assert_invalid("--probs", "")


# ---- invalid: weights --------------------------------------------------------

def test_pool_weights_length_mismatch_rejected():
    assert_invalid("--probs", "0.6,0.7", "--weights", "0.16")
    assert_invalid("--probs", "0.6,0.7", "--weights", "0.3,0.3,0.4")


def test_pool_weights_negative_rejected():
    assert_invalid("--probs", "0.6,0.7", "--weights=-0.5,1.5")
    assert_invalid("--probs", "0.6,0.7", "--weights", "-0.5,1.5")  # space form (argv join)


def test_pool_weights_nonfinite_rejected():
    assert_invalid("--probs", "0.6,0.7", "--weights=nan,0.5")
    assert_invalid("--probs", "0.6,0.7", "--weights=inf,0.5")


def test_pool_weights_all_zero_rejected():
    assert_invalid("--probs", "0.6,0.7", "--weights", "0,0")


# ---- valid: pre-fix pools unchanged ------------------------------------------

def test_pool_schema_example_unchanged():
    assert_valid({"log_odds_pool": 0.6246, "trimmed_mean": 0.6167,
                  "mean": 0.6167, "median": 0.6},
                 "--probs", "0.60,0.70,0.55", "--weights", "0.16,0.14,0.10")


def test_pool_unweighted_unchanged():
    assert_valid({"log_odds_pool": 0.6188, "trimmed_mean": 0.6167,
                  "mean": 0.6167, "median": 0.6},
                 "--probs", "0.60,0.70,0.55")


def test_pool_endpoints_inclusive():
    assert_valid({"log_odds_pool": 0.5, "mean": 0.5, "median": 0.5},
                 "--probs", "0,1")


def test_pool_individual_zero_weight_allowed():
    assert_valid({"mean": 0.5}, "--probs", "0.5,0.5", "--weights", "0,1")


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for fn in ALL_TESTS:
        try:
            fn()
            check(fn.__name__, True)
        except AssertionError as e:
            check(fn.__name__, False, str(e))
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
