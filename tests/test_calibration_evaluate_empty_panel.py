"""SF-BL-10742: evaluate() must raise a named ValueError on empty or
all-non-finite panels, and be bit-identical on non-empty lawful panels.
Tests both shipped copies against their pristine pre-task snapshots.
Run: python3 test_calibration_evaluate_empty_panel.py (exit 0 = all pass)
"""
import importlib.util, math, struct, sys
import numpy as np

W = "/home/hatch/workspace/superforecaster_out/backlog/work/SF-BL-10742"
PAIRS = [
    ("repo", f"{W}/pristine/calibration_repo.py",
     "/home/hatch/workspace/superforecaster/superforecaster/calibration.py"),
    ("out", f"{W}/pristine/calibration_out.py",
     "/home/hatch/workspace/superforecaster_out/calibration/calibration.py"),
]

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def bits(x):
    if isinstance(x, dict):
        return {k: bits(v) for k, v in x.items()}
    return struct.pack("<d", float(x))

CHECKS = []
def check(name, fn):
    try:
        fn(); CHECKS.append((name, "PASS", ""))
    except Exception as e:
        CHECKS.append((name, "FAIL", f"{type(e).__name__}: {e}"))

def expect_named_valueerror(fn, *needles):
    try:
        fn()
    except ValueError as e:
        msg = str(e)
        for n in needles:
            assert n in msg, f"message {msg!r} missing {n!r}"
        return
    raise AssertionError("no ValueError raised")

for label, pristine_path, live_path in PAIRS:
    pre = load(pristine_path, f"pre_{label}")
    live = load(live_path, f"live_{label}")

    def t_repro_pristine(pre=pre):
        r = pre.evaluate([], [])
        assert r["ece"] == 0.0 and math.isnan(r["brier"]), r
    check(f"{label}: pristine reproduces mixed verdict", t_repro_pristine)

    def t_empty(live=live):
        for y, p in [([], []), (np.array([]), np.array([])), ([], np.array([]))]:
            expect_named_valueerror(lambda y=y, p=p: live.evaluate(y, p),
                                    "evaluate", "empty", "SF-BL-10742")
        # n_bins variants still raise the named empty error
        expect_named_valueerror(lambda: live.evaluate([], [], n_bins=3),
                                "evaluate", "empty", "SF-BL-10742")
    check(f"{label}: empty panels raise named ValueError", t_empty)

    def t_all_nonfinite(live=live):
        cases = [
            ([float("nan"), float("nan")], [0.5, 0.5]),
            ([float("inf"), float("-inf")], [0.5, 0.5]),
            ([0, 1], [float("nan"), float("nan")]),
            ([0, 1], [float("inf"), float("-inf")]),
            ([float("nan")], [float("nan")]),
            ([float("nan"), 0], [0.5, float("nan")]),  # no fully-finite row
        ]
        for y, p in cases:
            expect_named_valueerror(lambda y=y, p=p: live.evaluate(y, p),
                                    "evaluate", "all-non-finite", "SF-BL-10742")
    check(f"{label}: all-non-finite panels raise named ValueError", t_all_nonfinite)

    def t_partial_still_raises(live=live):
        for y, p in [([0, 1], [0.5, float("nan")]), ([0, float("nan")], [0.5, 0.5]),
                     ([0, 1], [0.5, 1.5]), ([0, 2], [0.5, 0.5])]:
            try:
                live.evaluate(y, p)
            except ValueError:
                pass
            else:
                raise AssertionError(f"partial/bad panel accepted: {y} {p}")
    check(f"{label}: partially-nonfinite / bad panels still raise", t_partial_still_raises)

    def t_precedence(live=live):
        # n_bins validated first (pre-existing), length before panel guard
        expect_named_valueerror(lambda: live.evaluate([], [], n_bins=0), "n_bins")
        expect_named_valueerror(lambda: live.evaluate([], [0.5]), "len(y)")
        expect_named_valueerror(lambda: live.evaluate([0, 1], [0.5]), "len(y)")
    check(f"{label}: n_bins / length precedence preserved", t_precedence)

    def t_bit_identical(pre=pre, live=live):
        rng = np.random.default_rng(10742)
        panels = [
            ([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]),
            ([0], [0.0]), ([1], [1.0]), ([0, 0, 1, 1], [0.25, 0.25, 0.75, 0.75]),
            ([1, 1, 1], [0.99, 0.5, 0.01]),
        ]
        for n in (50, 500):
            pt = rng.beta(2, 2, n); yy = (rng.random(n) < pt).astype(float)
            panels.append((yy, pt)); panels.append((yy, np.clip(pt, 0, 1)))
        for y, p in panels:
            for nb in (1, 2, 5, 10, 20):
                a, b = pre.evaluate(y, p, n_bins=nb), live.evaluate(y, p, n_bins=nb)
                assert bits(a) == bits(b), (nb, a, b)
    check(f"{label}: lawful panels bit-identical vs pristine", t_bit_identical)

    def t_doc(live=live):
        assert "SF-BL-10742" in live.evaluate.__doc__
        assert "SF-BL-10742" in live._validate_evaluate_panel.__doc__
    check(f"{label}: raise documented in docstrings", t_doc)

def test_evaluate_empty_panel_all():
    failed = [(n, d) for n, s, d in CHECKS if s != "PASS"]
    assert not failed, failed

if __name__ == "__main__":
    for name, status, detail in CHECKS:
        print(f"{status}  {name}  {detail}")
    n_pass = sum(1 for _, s, _ in CHECKS if s == "PASS")
    print(f"{n_pass}/{len(CHECKS)} PASS")
    sys.exit(0 if n_pass == len(CHECKS) else 1)
