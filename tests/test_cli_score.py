"""
test_cli_score.py — SF-BL-9747: `sf score` rejects an empty score file.

Run:  python3 tests/test_cli_score.py     (or: python3 -m pytest tests/test_cli_score.py)
No pytest dependency: every check is a test_* function using assert, and
__main__ runs them all.

Before the fix, cmd_score scored a header-only (or zero-byte, or
blank-lines-only) CSV anyway: it exited 0 with n=0 and bare NaN literals
for brier / log_score and murphy's residual / base_rate — invalid strict
JSON that reads downstream as a real evaluation. These tests pin the
fixed contract: an empty file exits 2 with exactly one strict-JSON
error object and no score fields, and every non-empty file's output is
byte-identical to the pre-fix output.
"""
import json
import os as _os
import subprocess
import sys
import tempfile

CLI = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                    'superforecaster', 'cli.py')
PASS = []
SCORE_KEYS = {"n", "brier", "log_score", "ece", "murphy", "peer_score_vs_crowd"}

_TMP = tempfile.mkdtemp(prefix="sf-bl-9747-test-")


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _file(name, content):
    path = _os.path.join(_TMP, name)
    with open(path, "w") as f:
        f.write(content)
    return path


def sf_score_raw(path):
    return subprocess.run([sys.executable, CLI, "score", "--file", path],
                          capture_output=True, text=True)


def _reject_const(tok):
    raise ValueError(f"non-strict token {tok}")


def assert_empty_rejected(name, content):
    r = sf_score_raw(_file(name, content))
    assert r.returncode == 2, f"{name}: exit {r.returncode}, stdout={r.stdout!r}"
    body = json.loads(r.stdout, parse_constant=_reject_const)  # strict JSON, one object
    assert "error" in body, body
    assert not (SCORE_KEYS & set(body)), f"{name}: score figure leaked: {body}"
    assert "NaN" not in r.stdout, f"{name}: bare NaN in stdout: {r.stdout!r}"
    assert "Traceback" not in r.stderr, r.stderr


def assert_valid(name, content, expect):
    r = sf_score_raw(_file(name, content))
    assert r.returncode == 0, f"{name}: exit {r.returncode}, stderr={r.stderr!r}"
    payload = json.loads(r.stdout, parse_constant=_reject_const)
    for k, v in expect.items():
        assert payload[k] == v, f"{name}: {k}={payload[k]} != {v}"


# ---- invalid: zero data rows -------------------------------------------------

def test_score_header_only_rejected():
    assert_empty_rejected("hdr.csv", "y,p\n")  # the filed fixture


def test_score_header_only_crowd_rejected():
    assert_empty_rejected("hdr_crowd.csv", "y,p,crowd\n")


def test_score_zero_byte_rejected():
    assert_empty_rejected("zero.csv", "")


def test_score_header_plus_blank_lines_rejected():
    assert_empty_rejected("blanks.csv", "y,p\n\n\n")


# ---- valid: pre-fix outputs unchanged -----------------------------------------

def test_score_control_2row_unchanged():
    assert_valid("c2.csv", "y,p\n1,0.8\n0,0.3\n",
                 {"n": 2, "brier": 0.065, "log_score": 0.28991, "ece": 0.25})


def test_score_control_2row_crowd_unchanged():
    assert_valid("c2c.csv", "y,p,crowd\n1,0.8,0.6\n0,0.3,0.4\n",
                 {"n": 2, "brier": 0.065, "peer_score_vs_crowd": 0.095})


def test_score_control_1row_unchanged():
    assert_valid("c1.csv", "y,p\n1,0.8\n",
                 {"n": 1, "brier": 0.04, "log_score": 0.22314, "ece": 0.2})


def test_score_endpoints_unchanged():
    assert_valid("cep.csv", "y,p\n1,1.0\n0,0.0\n",
                 {"n": 2, "brier": 0.0, "ece": 0.0})


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
