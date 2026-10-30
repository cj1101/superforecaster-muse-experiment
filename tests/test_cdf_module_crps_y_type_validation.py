"""
test_cdf_module_crps_y_type_validation.py — SF-BL-11134 verification.

Pre-registered: "crps_from_cdf raises ValueError naming y on any bool,
string or None y, with numeric-y CRPS values bit-identical; kill if
y=True returns 0.875 or y='2.0'/None raises a bare TypeError instead of
the named ValueError."

Task repro panel: cdf [0,0.25,0.5,0.75,1.0] on nominals [0,1,2,3,4].
Pre-fix (verified live on all three live copies): y=True returned
0.875, bit-identical to the lawful y=1.0; y=False returned 1.375;
y='2.0' / np.str_('2.0') / b'2.0' died with a bare UFuncTypeError from
the (nominals >= y) integrand comparison that never names y; y=None
was caught only collaterally by SF-BL-10473's float() finiteness catch
(mislabelled "non-finite or non-numeric", no type screen of its own).
Distinct from SF-BL-9924 / SF-BL-10473 (the VALUE domain of y:
NaN/+/-inf) and from SF-BL-11203 (_cell_density's y, a different
function) / SF-BL-11309 (cdf/nominals element types).

Run: python3 tests/test_cdf_module_crps_y_type_validation.py
"""
import os, struct, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metaculus_bot"))
import numpy as np  # noqa: E402
import cdf_module as cm  # noqa: E402
PASS=[]
def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}"); PASS.append(bool(cond))
def _raises_11134(fn, *needles):
    try:
        fn()
    except ValueError as e:
        msg=str(e)
        assert "SF-BL-11134" in msg, f"missing SF-BL-11134 tag: {msg!r}"
        assert "y" in msg, f"error does not name y: {msg!r}"
        for n in needles: assert n in msg, f"missing {n!r} in {msg!r}"
        return msg
    except Exception as e:  # a bare TypeError/UFuncTypeError is the kill case
        raise AssertionError(f"raised {type(e).__name__} instead of the named ValueError: {e}")
    raise AssertionError("no ValueError raised — corrupt y type returned a score (kill)")
def _bits(v): return struct.pack("<d", float(v)).hex()
CDF=[0, 0.25, 0.5, 0.75, 1.0]; NOM=[0, 1, 2, 3, 4]

def test_task_repro_bool_true_raises_not_0875():
    # Kill case: pre-fix y=True returned 0.875, identical to y=1.0.
    _raises_11134(lambda: cm.crps_from_cdf(CDF, NOM, True), "y", "bool")
    assert cm.crps_from_cdf(CDF, NOM, 1.0) == 0.875  # lawful control intact

def test_bool_y_raises_all_forms():
    for y in (True, False, np.bool_(True), np.bool_(False)):
        _raises_11134(lambda y=y: cm.crps_from_cdf(CDF, NOM, y), "boolean")

def test_string_y_raises_all_forms():
    for y in ("2.0", "2", "abc", "", np.str_("2.0"), b"2.0"):
        _raises_11134(lambda y=y: cm.crps_from_cdf(CDF, NOM, y), "y")

def test_none_y_raises_named_valueerror():
    # Kill case: must be the named ValueError, never a bare TypeError.
    _raises_11134(lambda: cm.crps_from_cdf(CDF, NOM, None), "y", "None")

def test_other_nonnumeric_y_raises():
    for y in ([2.0], (2.0,), {"y": 2.0}, object(),
              np.array([2.0]), np.array(["2.0"]), np.array(True)):
        _raises_11134(lambda y=y: cm.crps_from_cdf(CDF, NOM, y), "y")

def test_no_corrupt_type_returns_any_score_kill_sweep():
    bad = [True, False, np.bool_(True), "2.0", "0", "abc", "", None,
           b"2.0", [1.0], {"a": 1}, object(), np.array(["2.0"])]
    grids = [(CDF, NOM), ([0.0, 0.5, 1.0], [0, 1, 2]),
             ([0.0, 0.5, 1.0], [0.0, 5.0, 10.0])]
    for c, n in grids:
        for y in bad:
            _raises_11134(lambda c=c, n=n, y=y: cm.crps_from_cdf(c, n, y))

def test_numeric_y_crps_bit_identical():
    # Pre-fix bit patterns (little-endian double hex), captured pre-patch.
    panels = [
        (CDF, NOM, 1.0, "000000000000ec3f"),
        (CDF, NOM, 0.0, "000000000000f63f"),
        (CDF, NOM, 4.0, "000000000000ec3f"),
        (CDF, NOM, 2.5, "000000000000d83f"),
        ([0.0, 0.5, 1.0], [0, 1, 2], 1.0, "000000000000d03f"),
        ([0.0, 0.5, 1.0], [0.0, 5.0, 10.0], 5.0, "000000000000f43f"),
        ([0.1, 0.4, 0.9], [10.0, 20.0, 30.0], 17.0, "9a99999999990d40"),
    ]
    for c, n, y, bits in panels:
        v = cm.crps_from_cdf(c, n, y)
        assert _bits(v) == bits, f"y={y}: bits {_bits(v)} != pre-fix {bits}"

def test_numeric_y_forms_bit_identical():
    # int / float / numpy scalar / numbers.Number / 0-d array forms of
    # the same outcome all scored 0.375 pre-fix and must still do so.
    from decimal import Decimal
    from fractions import Fraction
    for y in (2, 2.0, np.float64(2.0), np.int64(2), np.float32(2.0),
              Decimal("2.0"), Fraction(2, 1), np.array(2.0)):
        v = cm.crps_from_cdf(CDF, NOM, y)
        assert _bits(v) == "000000000000d83f", f"y={y!r}: {v!r}"

def test_201pt_scaffold_grid_bit_identical():
    nom = cm.grid_nominals(0.0, 100.0)
    cdf = np.clip((nom - 20) / 60.0, 0, 1)
    for y, bits in ((0.0, "c2166cc116004440"), (15.0, "842dd8822d403940"),
                    (50.0, "0cb6600bb6001440"), (100.0, "c2166cc116e04340")):
        assert _bits(cm.crps_from_cdf(cdf, nom, y)) == bits, f"201pt y={y}"

def test_prior_screens_preserved():
    # SF-BL-10473/9924 value domain: NaN y still raises naming y.
    try: cm.crps_from_cdf(CDF, NOM, float("nan"))
    except ValueError as e: assert "SF-BL-10473" in str(e) and "y" in str(e), str(e)
    else: raise AssertionError("NaN y no longer raises")
    # SF-BL-9924 classes still raise: reversed nominals, CDF outside [0,1].
    for c, n, y in (([0.0, 0.5, 1.0], [2, 1, 0], 1.0),
                    ([0.0, 0.5, 1.5], [0, 1, 2], 1.0)):
        try: cm.crps_from_cdf(c, n, y)
        except ValueError as e: assert "SF-BL-9924" in str(e), str(e)
        else: raise AssertionError(f"9924 class no longer raises: {c} {n} {y}")
    # SF-BL-10269 degenerate grid still raises with its own tag.
    try: cm.crps_from_cdf([1.0], [0.0], 1.0)
    except ValueError as e: assert "SF-BL-10269" in str(e), str(e)
    else: raise AssertionError("10269 degenerate class no longer raises")

def test_client_adapter_raises_and_matches():
    import metaculus_client as mc
    assert mc.crps_from_cdf is cm.crps_from_cdf
    _raises_11134(lambda: mc.crps_from_cdf(CDF, NOM, True), "y")
    _raises_11134(lambda: mc.crps_from_cdf(CDF, NOM, "2.0"), "y")
    assert mc.crps_from_cdf(CDF, NOM, 1.0) == 0.875

ALL_TESTS=[v for k,v in sorted(globals().items()) if k.startswith("test_")]
if __name__=="__main__":
    for fn in ALL_TESTS:
        try: fn(); check(fn.__name__,True)
        except AssertionError as e: check(fn.__name__,False,str(e))
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
