"""
test_cdf_module_naive_probs_horizon_type_validation.py — SF-BL-11244.

Pre-registered done/kill criteria (TASK.md):
  "naive_normal_percentiles raises ValueError naming the argument on
   any bool or string probs element or horizon_days, with numeric
   probs/horizon percentile dicts bit-identical; kill if
   probs=[True] still returns {1.0: inf} or horizon_days=True
   returns a dict instead of raising."

Bug: naive_normal_percentiles never TYPE-validated probs or
horizon_days. float()/np.asarray(dtype=float) coercion read
horizon_days=True as a 1-day horizon (task live repro on 40 trailing
returns at spot 100: control horizon_days=5 median 99.397,
horizon_days=True median 99.879), horizon_days='5' died with a bare
'can't multiply sequence by non-int' TypeError (or was silently
coerced to 5.0 once the SF-BL-10267 float conversion landed), and
probs=[True] fed norm.ppf(1.0) returning {1.0: inf}; string probs
elements such as ['0.5'] were silently parsed. Distinct from
SF-BL-10267 (horizon VALUE domain in a different signature context)
and SF-BL-10393 (this function's spot).

Fix (metaculus_bot/cdf_module.py, naive_normal_percentiles, mirrored
to the superforecaster_out live copy; type screens only added to the
wave3 original): a probs type screen at entry and a horizon_days
type screen before the value screen — any bool / string / bytes /
None element or horizon raises a ValueError naming the argument and
SF-BL-11244 (dual-tagged with SF-BL-10474 / SF-BL-10267 so those
screens' cases keep naming them). Numeric forms (int / float /
numpy numeric / Decimal / Fraction / 0-d numeric array) fall
through untouched, so numeric dicts stay bit-identical — the
expected bit patterns below were captured live PRE-fix
(seed 11244, 40 returns).

Run: python3 tests/test_cdf_module_naive_probs_horizon_type_validation.py
     or pytest tests/test_cdf_module_naive_probs_horizon_type_validation.py
"""
import os, struct, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metaculus_bot"))
import numpy as np
import pytest
import cdf_module as cm

RNG = np.random.default_rng(11244)
RETURNS = RNG.normal(0.001, 0.02, 40)

def _bits(d): return "".join(struct.pack("<d", d[p]).hex() for p in sorted(d))

BITS_H5 = "4a70d144187e57400d6bb45d57905840fcd8ff064d565940a8e2e90a7e225a403d8c083e95535b40"
BITS_H1 = "1683431ddf3b58404c90fe97d4b85840320aa1f62a115940f7399efbbc6a5940460df127cced5940"
BITS_H7 = "e417e4f33e4b57403a76f4a6528e584081492c6e257959403a7652c6bd6c5a401ef75fe03ddb5b40"
BITS_CUSTOM = "a4e7a999abc456401bb042b657e35740fcd8ff064d56594022199af6c2df5a40eedd3d3e21325c40"
BITS_SINGLE = "fcd8ff064d565940"
BITS_PAIR = "0d6bb45d57905840a8e2e90a7e225a40"

def _raises_named(fn, arg):
    with pytest.raises(ValueError) as ei:
        fn()
    msg = str(ei.value)
    assert arg in msg and "SF-BL-11244" in msg, msg

def _h(h): return lambda: cm.naive_normal_percentiles(100.0, RETURNS, h)
def _p(probs): return lambda: cm.naive_normal_percentiles(100.0, RETURNS, 5, probs=probs)

def test_kill_probs_true_does_not_return_inf_dict():
    _raises_named(_p([True]), "probs")

def test_kill_horizon_true_does_not_return_dict():
    _raises_named(_h(True), "horizon_days")

def test_bool_probs_elements_raise_named():
    for p in ([True], [False], [0.5, True], [False, 0.5], [np.bool_(True)],
              np.array([True]), np.array([True, False]), True, np.bool_(False),
              np.array([True, 0.5], dtype=object)):
        _raises_named(_p(p), "probs")

def test_string_probs_elements_raise_named():
    for p in (["0.5"], ["0.05", "0.5", "0.95"], [0.5, "0.9"], [np.str_("0.5")],
              [b"0.5"], np.array(["0.5"]), np.array(["0.25", "0.75"]),
              "0.5", "abc", [0.5, "x"], [0.5, None], None):
        _raises_named(_p(p), "probs")

def test_bool_horizon_raises_named():
    for h in (True, False, np.bool_(True), np.bool_(False), np.array(True)):
        _raises_named(_h(h), "horizon_days")

def test_string_and_none_horizon_raises_named_not_bare_typeerror():
    for h in ("5", "5.0", "abc", "", np.str_("5"), b"5", None, [5], (5,), {"h": 5}, object()):
        _raises_named(_h(h), "horizon_days")

def test_no_corrupt_type_returns_a_dict_kill_sweep():
    bad_h = [True, False, "5", "abc", None, b"5", np.bool_(True)]
    bad_p = [[True], ["0.5"], [0.5, True], [0.5, "0.9"], np.array([True]), np.array(["0.5"])]
    for h in bad_h:
        try: out = cm.naive_normal_percentiles(100.0, RETURNS, h)
        except ValueError: continue
        raise AssertionError(f"horizon_days={h!r} returned {out!r}")
    for p in bad_p:
        try: out = cm.naive_normal_percentiles(100.0, RETURNS, 5, probs=p)
        except ValueError: continue
        raise AssertionError(f"probs={p!r} returned {out!r}")

def test_numeric_bit_identical():
    assert _bits(cm.naive_normal_percentiles(100.0, RETURNS, 5)) == BITS_H5
    assert _bits(cm.naive_normal_percentiles(100.0, RETURNS, 1)) == BITS_H1
    assert _bits(cm.naive_normal_percentiles(100.0, RETURNS, 7.0)) == BITS_H7
    assert _bits(cm.naive_normal_percentiles(100.0, RETURNS, 5, (0.01, 0.1, 0.5, 0.9, 0.99))) == BITS_CUSTOM
    assert _bits(cm.naive_normal_percentiles(100.0, RETURNS, 5, (0.5,))) == BITS_SINGLE
    assert _bits(cm.naive_normal_percentiles(100.0, RETURNS, 5, [0.25, 0.75])) == BITS_PAIR
    assert _bits(cm.naive_normal_percentiles(100.0, RETURNS, 5, np.array([0.25, 0.75]))) == BITS_PAIR

def test_numeric_forms_bit_identical():
    from decimal import Decimal
    from fractions import Fraction
    for h in (5, 5.0, np.int64(5), np.float64(5.0), Decimal("5"), Fraction(5, 1), np.array(5.0)):
        assert _bits(cm.naive_normal_percentiles(100.0, RETURNS, h)) == BITS_H5, repr(h)
    assert _bits(cm.naive_normal_percentiles(100.0, RETURNS, 5, [Decimal("0.5")])) == BITS_SINGLE
    assert _bits(cm.naive_normal_percentiles(100.0, RETURNS, 5, [Fraction(1, 2)])) == BITS_SINGLE

def test_prior_screens_preserved():
    # SF-BL-10474 value domain: endpoint / out-of-(0,1) numeric probs.
    with pytest.raises(ValueError, match="SF-BL-10474"):
        cm.naive_normal_percentiles(100.0, RETURNS, 5, probs=(0.0, 0.5, 1.0))
    with pytest.raises(ValueError, match="SF-BL-10474"):
        cm.naive_normal_percentiles(100.0, RETURNS, 5, probs=(float("nan"),))
    # SF-BL-10393 spot.
    with pytest.raises(ValueError, match="SF-BL-10393"):
        cm.naive_normal_percentiles(float("nan"), RETURNS, 5)
    # SF-BL-10267 value domain: numeric non-positive / non-finite horizon.
    for h in (-5, 0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="SF-BL-10267"):
            cm.naive_normal_percentiles(100.0, RETURNS, h)
    # Dual-tagging: the old string cases still name their original tags.
    with pytest.raises(ValueError, match="SF-BL-10267"):
        cm.naive_normal_percentiles(100.0, RETURNS, "abc")
    with pytest.raises(ValueError, match="SF-BL-10474"):
        cm.naive_normal_percentiles(100.0, RETURNS, 5, probs=[0.5, "x"])
    with pytest.raises(ValueError):
        cm.naive_normal_percentiles(100.0, RETURNS[:10], 5)

def test_client_adapter_raises_and_matches():
    import metaculus_client as mc
    assert mc.naive_normal_percentiles is cm.naive_normal_percentiles
    _raises_named(lambda: mc.naive_normal_percentiles(100.0, RETURNS, True), "horizon_days")
    _raises_named(lambda: mc.naive_normal_percentiles(100.0, RETURNS, 5, probs=[True]), "probs")
    assert _bits(mc.naive_normal_percentiles(100.0, RETURNS, 5)) == BITS_H5

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
