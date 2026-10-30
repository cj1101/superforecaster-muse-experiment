"""
test_calibration_string_panels.py — SF-BL-11052 verification.

Pre-registered done/kill criteria (TASK.md):
  "Guard raises ValueError naming y/p on any string-dtype forecast
   or outcome panel in brier_score, log_score and evaluate, with
   numeric panels bit-identical; kill if brier_score([1], ['0.5'])
   or log_score(['1'], [0.5]) returns a float instead of raising."

Bug: _as_y and the p conversions in brier_score / log_score used
np.asarray(..., dtype=float), which silently parses numeric
strings, so a JSON/text pipeline bug was scored as a genuine
forecast:
  brier_score([1], ['0.5'])          == 0.25
  brier_score(['1'], [0.5])          == 0.25
  log_score([1], ['0.5'])            == 0.6931
  evaluate([1, 0], ['0.5', '0.5'])   == full scorecard (Brier 0.25)
(repro_prefix.py / repro_prefix.log in the task work dir.)

Fix (superforecaster/calibration.py): _reject_strings / _has_string
run on the RAW inputs before any float conversion — at the top of
brier_score and log_score for p, inside _as_y for y, and at the top
of evaluate for both — raising
"p|y must not contain string values; got string value(s) in p|y
(a string is not a valid p|y value) (SF-BL-11052)".
Lawful panels are pure-validation bit-identical: the pre/post
SHA-256 over 11 lawful panels (brier, log, evaluate) is
14e5247c69d2a0b725e76dcd3166576e02ecd18571e058acea356d479685c087
both ways (lawful_baseline_prefix.json in the task work dir).

Run:  python3 tests/test_calibration_string_panels.py
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from superforecaster.calibration import (  # noqa: E402
    brier_score, evaluate, log_score)

PASS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))


def _string_error(fn, name, *args, **kw):
    """True iff fn raises the named SF-BL-11052 ValueError for `name`."""
    try:
        fn(*args, **kw)
    except ValueError as e:
        msg = str(e)
        return ("string" in msg and "SF-BL-11052" in msg
                and msg.startswith(f"{name} must not contain")
                and f"in {name}" in msg)
    return False


def _any_value_error(fn, *args, **kw):
    try:
        fn(*args, **kw)
    except ValueError:
        return True
    return False


def main():
    # --- 1. Titled + kill repros raise the named error ---------------
    check("brier p-string (titled)", _string_error(
        brier_score, "p", [1], ["0.5"]))
    check("brier y-string (titled)", _string_error(
        brier_score, "y", ["1"], [0.5]))
    check("log p-string (titled)", _string_error(
        log_score, "p", [1], ["0.5"]))
    check("log y-string (KILL repro)", _string_error(
        log_score, "y", ["1"], [0.5]))
    check("evaluate p-string (titled)", _string_error(
        evaluate, "p", [1, 0], ["0.5", "0.5"]))
    check("evaluate y-string", _string_error(
        evaluate, "y", ["1", "0"], [0.5, 0.5]))
    # Kill condition stated negatively: no float comes back.
    try:
        r = brier_score([1], ["0.5"])
        kill_ok = False
    except ValueError:
        kill_ok = True
    check("kill: brier([1],['0.5']) returns no float", kill_ok)

    # --- 2. String forms: mixed, scalar, ndarray dtypes, nesting -----
    check("brier mixed panel [0.9,'0.1']", _string_error(
        brier_score, "p", [1, 0], [0.9, "0.1"]))
    check("brier scalar string p='0.5'", _string_error(
        brier_score, "p", [1], "0.5"))
    check("brier np.str_ scalar", _string_error(
        brier_score, "p", [1], [np.str_("0.5")]))
    check("brier string-dtype ndarray p", _string_error(
        brier_score, "p", [1, 0], np.array(["0.5", "0.5"])))
    check("brier string-dtype ndarray y", _string_error(
        brier_score, "y", np.array(["1", "0"]), [0.5, 0.5]))
    check("brier bytes-dtype ndarray p", _string_error(
        brier_score, "p", [1, 0], np.array([b"0.5", b"0.5"])))
    check("log object ndarray with string leaf", _string_error(
        log_score, "p", [1, 0],
        np.array([0.9, "0.1"], dtype=object)))
    check("log tuple p", _string_error(
        log_score, "p", [1, 0], ("0.9", "0.1")))
    check("evaluate nested-list string", _string_error(
        evaluate, "p", [1, 0], [["0.5"], ["0.5"]]))
    check("brier non-numeric string still named error", _string_error(
        brier_score, "p", [1], ["abc"]))
    check("evaluate y-string outcome panel", _string_error(
        evaluate, "y", ["1"], [0.5]))

    # --- 3. Lawful panels bit-identical -------------------------------
    base_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..",
        "superforecaster_out", "backlog", "work", "SF-BL-11052",
        "lawful_baseline_prefix.json")
    rng = np.random.default_rng(11052)
    panels = [([1], [1.0]), ([1, 0], [0.9, 0.1]), ([1, 0], [1, 0]),
              ([0, 1, 1, 0], [0.25, 0.75, 0.5, 0.05])]
    for n in (1, 2, 3, 7, 23, 57):
        panels.append((rng.integers(0, 2, size=n).tolist(),
                       rng.random(n).tolist()))
    panels.append((np.array([1, 0, 1]), np.array([0.8, 0.2, 0.6])))
    rows = []
    for y, p in panels:
        ev = evaluate(y, p)
        rows.append({
            "y": np.asarray(y).ravel().tolist(),
            "p": np.asarray(p).ravel().tolist(),
            "brier": brier_score(y, p), "log": log_score(y, p),
            "eval": {k: ev[k] for k in sorted(ev)}})
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True).encode()).hexdigest()
    check("lawful hash matches pre-fix baseline",
          digest == "14e5247c69d2a0b725e76dcd3166576e02ecd1857"
                    "1e058acea356d479685c087", digest[:16])
    if os.path.exists(base_path):
        with open(base_path) as f:
            baseline = json.load(f)
        check("lawful rows exactly equal baseline file", rows == baseline)
    check("pinned lawful brier", brier_score([1, 0], [0.5, 0.5]) == 0.25)
    check("pinned lawful log",
          log_score([1, 0], [0.9, 0.1]) == 0.10536051565782628)
    check("lawful int endpoints still score",
          brier_score([1, 0], [1, 0]) == 0.0)

    # --- 4. Pre-existing errors unchanged ------------------------------
    check("out-of-range p still raises (SF-BL-9484)", _any_value_error(
        brier_score, [1], [1.5]))
    check("non-finite p still raises", _any_value_error(
        log_score, [1], [float("nan")]))
    check("bool p still raises SF-BL-10938 error", _any_value_error(
        brier_score, [1], [True]))
    check("bool y still raises", _any_value_error(
        brier_score, [True], [0.5]))
    check("bad eps still raises (SF-BL-9659)", _any_value_error(
        log_score, [1], [0.5], 0))
    check("bool n_bins still raises (SF-BL-9722)", _any_value_error(
        evaluate, [1], [0.5], True))
    check("non-binary y still raises", _any_value_error(
        brier_score, [2], [0.5]))
    check("length mismatch still raises", _any_value_error(
        brier_score, [0, 1, 0], [0.5]))
    check("empty evaluate still raises (SF-BL-10742)", _any_value_error(
        evaluate, [], []))

    print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
    return 0 if all(PASS) else 1


if __name__ == "__main__":
    sys.exit(main())
