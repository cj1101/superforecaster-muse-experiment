"""
test_cdf_module_naive_horizon_validation.py — SF-BL-10267 verification.

Pre-registered done/kill criterion (TASK.md):
  "Fix raises ValueError naming horizon_days unless it is finite and
   >0, with lawful-horizon outputs bit-identical; kill if horizon in
   {NaN, -5, 0, inf} still returns a percentile dict."

Bug: naive_normal_percentiles flowed horizon_days unchecked into
mean*horizon and std*sqrt(horizon): NaN/-5 returned all-NaN dicts,
0 returned a degenerate all-spot dict, inf returned a 0.0/NaN mix —
all confirmed live pre-fix with 40 finite daily returns (seed 10267).

Fix: validate horizon_days (float-converted, finite, >0) before any
arithmetic, raising ValueError naming horizon_days and SF-BL-10267.
Lawful computation uses the converted float, bit-identical to pre-fix
(expected bit patterns below captured live pre-fix).

Run: python3 tests/test_cdf_module_naive_horizon_validation.py
     or pytest tests/test_cdf_module_naive_horizon_validation.py
"""
import os, struct, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metaculus_bot"))
import numpy as np
import pytest
import cdf_module as cm

RNG = np.random.default_rng(10267)
RETURNS = RNG.normal(0.001, 0.02, 40)
EXPECTED_BITS = {
    1: "2ea390f12b29584007161a7c2ca45840901a61011cfb5840da44223d3e535940c743f3402cd45940",
    5: "c9d38e900d1d574015149b0c6927584021c83a9695e758406d230c1abbad5940d8d96212a6d55a40",
    7: "009c0ed1c5c356407c4149ba70fb57402b1a701ad8dd58401cb1f8d798c859402fd4e15798295b40",
    30: "ab0ea6b5cc5954408b1fde161fab5640803bcb61e56e58407efc8d64d7555a402953f50ea4555d40",
    0.5: "e862c928536858408b9f944fefbf5840771a10e28dfd5840b2677cdec53b5940619da30159965940",
    2.5: "cc98f6c465ab57406a506f2b066b5840cf929dcec7f35840573d9f5d877f594011fa1eb8f54d5a40",
}
def _bits(d): return "".join(struct.pack("<d", d[p]).hex() for p in sorted(d))
def _raises_named(h):
    with pytest.raises(ValueError) as ei:
        cm.naive_normal_percentiles(100.0, RETURNS, h)
    msg = str(ei.value)
    assert "horizon_days" in msg and "SF-BL-10267" in msg, msg

def test_kill_set_raises_named():
    for h in [float("nan"), -5, 0, float("inf")]:
        _raises_named(h)

def test_other_bad_horizons_raise_named():
    for h in [float("-inf"), -0.5, -1, 0.0, None, "abc", np.nan, np.inf]:
        _raises_named(h)

def test_no_bad_horizon_returns_dict():
    for h in [float("nan"), -5, 0, float("inf")]:
        try:
            out = cm.naive_normal_percentiles(100.0, RETURNS, h)
        except ValueError:
            continue
        raise AssertionError(f"h={h!r} returned {out!r}")

def test_lawful_bit_identical():
    for h, exp in EXPECTED_BITS.items():
        assert _bits(cm.naive_normal_percentiles(100.0, RETURNS, h)) == exp, h

def test_lawful_variants():
    assert cm.naive_normal_percentiles(100.0, RETURNS, 7) == cm.naive_normal_percentiles(100.0, RETURNS, 7.0)
    assert cm.naive_normal_percentiles(100.0, RETURNS, np.int64(7)) == cm.naive_normal_percentiles(100.0, RETURNS, 7)
    for h in [1e-6, 365]:
        out = cm.naive_normal_percentiles(100.0, RETURNS, h)
        assert all(np.isfinite(v) for v in out.values())

def test_short_returns_guard_intact():
    with pytest.raises(ValueError):
        cm.naive_normal_percentiles(100.0, RETURNS[:10], 5)

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
