"""
test_cdf_module_naive_probs_validation.py — SF-BL-10474 verification.

Pre-registered done/kill criterion (TASK.md):
  "Fix validates probs lie in (0, 1) finite at entry with a named
   error, regression-tested for 0.0, 1.0, negative and NaN probs;
   kill if any endpoint prob still yields a percentile entry or
   interior-prob outputs change."

Bug: naive_normal_percentiles flowed probs unchecked into
stats.norm.ppf: ppf(0)=-inf / ppf(1)=+inf, so probs=(0.0, 0.5, 1.0)
returned {0.0: 0.0, 0.5: 99.52..., 1.0: inf} live (seed 10474), and
negative / >1 / NaN probs returned NaN entries as if lawful.

Fix: validate probs at entry (finite, in the open interval (0, 1),
non-empty 1-D), raising ValueError naming probs and SF-BL-10474.
Lawful computation uses the asarray-converted probs, bit-identical
to pre-fix (expected bit patterns below captured live pre-fix).

Run: python3 tests/test_cdf_module_naive_probs_validation.py
     or pytest tests/test_cdf_module_naive_probs_validation.py
"""
import os, struct, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metaculus_bot"))
import numpy as np
import pytest
import cdf_module as cm

RNG = np.random.default_rng(10474)
RETURNS = RNG.normal(0.001, 0.02, 60)
EXPECTED_BITS = {
    "default": "6353111b6476564010b0dddedddb5740926613de5ae158409e4238b909f259405d767479f18e5b40",
    "custom": "b9c384e8e9875540608096f8b5f95640926613de5ae15840097c6acf6df15a40759db4552ec05c40",
    "single": "926613de5ae15840",
    "pair": "10b0dddedddb57409e4238b909f25940",
    "near": "52dc75cd58235140926613de5ae15840f236c17e5f0f6240",
}
PANELS = {"default": None, "custom": (0.01, 0.1, 0.5, 0.9, 0.99),
          "single": (0.5,), "pair": [0.25, 0.75], "near": (1e-9, 0.5, 1 - 1e-9)}

def _bits(d): return "".join(struct.pack("<d", d[p]).hex() for p in sorted(d))
_DEFAULT = object()
def _call(probs=_DEFAULT):
    if probs is _DEFAULT:
        return cm.naive_normal_percentiles(100.0, RETURNS, 10)
    return cm.naive_normal_percentiles(100.0, RETURNS, 10, probs=probs)
def _raises_named(probs):
    with pytest.raises(ValueError) as ei:
        _call(probs)
    msg = str(ei.value)
    assert "probs" in msg and "SF-BL-10474" in msg, msg

def test_kill_set_raises_named():
    # task-named regression set: 0.0, 1.0, negative, NaN — alone and mixed
    for p in [(0.0, 0.5, 1.0), (0.0,), (1.0,), (-0.1, 0.5), (-1.0,),
              (0.5, float("nan")), (float("nan"),)]:
        _raises_named(p)

def test_other_bad_probs_raise_named():
    for p in [(1.5,), (2.0,), (float("inf"),), (0.5, float("-inf")),
              (), 0.5, "abc", None, [0.5, "x"]]:
        _raises_named(p)

def test_no_endpoint_prob_yields_entry():
    for p in [(0.0, 0.5, 1.0), (0.0,), (1.0,), (-0.1, 0.5), (0.5, float("nan"))]:
        try:
            out = _call(p)
        except ValueError:
            continue
        raise AssertionError(f"probs={p!r} returned {out!r}")

def test_lawful_bit_identical():
    for name, probs in PANELS.items():
        out = _call() if probs is None else _call(probs)
        assert _bits(out) == EXPECTED_BITS[name], name

def test_lawful_variants():
    assert _call(np.array([0.25, 0.75])) == _call((0.25, 0.75))
    out = _call((1e-9, 0.5, 1 - 1e-9))
    assert all(np.isfinite(v) for v in out.values())

def test_prior_screens_intact():
    with pytest.raises(ValueError, match="SF-BL-10393"):
        cm.naive_normal_percentiles(float("nan"), RETURNS, 10)
    with pytest.raises(ValueError, match="SF-BL-10267"):
        cm.naive_normal_percentiles(100.0, RETURNS, -5)
    with pytest.raises(ValueError):
        cm.naive_normal_percentiles(100.0, RETURNS[:10], 10)

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}")
        except Exception as exc:
            failed.append(fn.__name__); print(f"FAIL  {fn.__name__}: {exc!r}")
    print(f"\n{len(fns)-len(failed)}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
