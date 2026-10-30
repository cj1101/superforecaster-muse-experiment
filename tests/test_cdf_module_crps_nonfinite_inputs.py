"""
test_cdf_module_crps_nonfinite_inputs.py — SF-BL-10473 verification.

Pre-registered: "Fix raises a named ValueError on any non-finite
cdf/nominals/y input, regression-tested per argument; kill if any
non-finite input still returns a numeric or NaN CRPS, or clean-grid
CRPS values change."

Task repro: crps_from_cdf([0.0, NaN, 1.0], [0, 1, 2], y=1.0) returned
NaN on the unscreened copies (the frozen wave3 patch snapshots still
do), grading a corrupt numeric forecast as an ordinary NaN CRPS.
Distinct from SF-BL-9924 (unsorted nominals / NaN outcome / CDF
outside [0,1]), SF-BL-10269 (empty grid) — this is the finiteness
class itself, screened per argument and tagged SF-BL-10473 (with an
SF-BL-9924 cross-reference so that suite's message assertions stay
valid).

Run: python3 tests/test_cdf_module_crps_nonfinite_inputs.py
"""
import os, struct, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metaculus_bot"))
import numpy as np  # noqa: E402
import cdf_module as cm  # noqa: E402
PASS=[]
def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}"); PASS.append(bool(cond))
def _raises_10473(fn, *needles):
    try:
        fn()
    except ValueError as e:
        msg=str(e)
        assert "SF-BL-10473" in msg, f"missing SF-BL-10473 tag: {msg!r}"
        for n in needles: assert n in msg, f"missing {n!r} in {msg!r}"
        return msg
    raise AssertionError("no ValueError raised — non-finite input returned a score (kill)")
def _bits(v): return struct.pack("<d", float(v)).hex()
NAN, INF = float("nan"), float("inf")

def test_task_repro_nan_cdf_raises():
    _raises_10473(lambda: cm.crps_from_cdf([0.0, NAN, 1.0], [0, 1, 2], 1.0), "cdf", "finite")

def test_nonfinite_cdf_raises_per_argument_any_position():
    for bad in (NAN, INF, float("-inf")):
        for i in range(3):
            c=[0.0,0.5,1.0]; c[i]=bad
            _raises_10473(lambda c=c: cm.crps_from_cdf(c,[0,1,2],1.0), "cdf")

def test_nonfinite_nominals_raises_per_argument_any_position():
    for bad in (NAN, INF, float("-inf")):
        for i in range(3):
            n=[0.0,1.0,2.0]; n[i]=bad
            _raises_10473(lambda n=n: cm.crps_from_cdf([0.0,0.5,1.0],n,1.0), "nominals")

def test_nonfinite_y_raises_per_argument():
    for y in (NAN, INF, float("-inf")):
        _raises_10473(lambda y=y: cm.crps_from_cdf([0.0,0.5,1.0],[0,1,2],y), "y", "finite")
    _raises_10473(lambda: cm.crps_from_cdf([0.0,0.5,1.0],[0,1,2],np.float64(NAN)), "y")

def test_nonfinite_numpy_and_tuple_forms_raise():
    _raises_10473(lambda: cm.crps_from_cdf(np.array([0.0,NAN,1.0]), np.array([0.,1.,2.]), 1.0), "cdf")
    _raises_10473(lambda: cm.crps_from_cdf((0.0,0.5,1.0), (0.0,INF,2.0), 1.0), "nominals")
    _raises_10473(lambda: cm.crps_from_cdf([0.0,0.5,1.0],[0,1,2], np.array(NAN)), "y")

def test_no_nonfinite_input_returns_any_score_kill_sweep():
    bads=[]
    for bad in (NAN, INF, float("-inf")):
        for i in range(3):
            c=[0.0,0.5,1.0]; c[i]=bad; bads.append((c,[0,1,2],1.0))
            n=[0.0,1.0,2.0]; n[i]=bad; bads.append(([0.0,0.5,1.0],n,1.0))
        bads.append(([0.0,0.5,1.0],[0,1,2],bad))
    # combinations: two arguments corrupt at once still raise (never score)
    bads.append(([NAN,0.5,1.0],[0,NAN,2],NAN))
    bads.append(([0.0,INF,1.0],[0,1,2],INF))
    for c,n,y in bads:
        try: v=cm.crps_from_cdf(c,n,y)
        except ValueError: continue
        raise AssertionError(f"returned {v!r} for cdf={c} nominals={n} y={y} (kill)")

def test_clean_grid_crps_bit_identical():
    cases=[([0.0,0.5,1.0],[0,1,2],1.0,"000000000000d03f"),([0.0,0.5,1.0],[0.0,5.0,10.0],5.0,"000000000000f43f"),([0.0,1.0],[0.0,10.0],5.0,"0000000000000000"),([0.1,0.4,0.6,0.9],[0,1,2,3],1.5,"2085eb51b81ed53f"),([0.0,0.5,1.0],[-10.0,0.0,10.0],0.0,"0000000000000440")]
    for c,n,y,b in cases:
        v=cm.crps_from_cdf(c,n,y); assert _bits(v)==b, f"{_bits(v)} != {b}"

def test_201pt_scaffold_grid_bit_identical():
    nom=cm.grid_nominals(0,100); cdf=cm.percentiles_to_cdf({0.05:10,0.25:30,0.5:50,0.75:70,0.95:90},0,100)
    exp={0:"d9cef75363754140",50:"c976be9f1aab1b40",15.0:"e6d022dbf9dd3540"}
    for y,b in exp.items(): assert _bits(cm.crps_from_cdf(cdf,nom,y))==b

def test_prior_screens_preserved():
    # SF-BL-10269 degenerate (finite) still names 10269 — the 10473
    # screen is reject-only on non-finite values and does not pre-empt it.
    try: cm.crps_from_cdf([],[],0.5)
    except ValueError as e: assert "SF-BL-10269" in str(e)
    else: raise AssertionError("degenerate grid did not raise")
    # SF-BL-9924 classes still raise: reversed nominals, CDF outside [0,1]
    for c,n,y in (([0.0,0.5,1.0],[2,1,0],1.0),([0.0,0.5,1.5],[0,1,2],1.0)):
        try: cm.crps_from_cdf(c,n,y)
        except ValueError as e: assert "SF-BL-9924" in str(e), str(e)
        else: raise AssertionError(f"9924 class no longer raises: {c} {n} {y}")

def test_client_adapter_raises_and_matches():
    import metaculus_client as mc
    assert mc.crps_from_cdf is cm.crps_from_cdf
    _raises_10473(lambda: mc.crps_from_cdf([0.0,NAN,1.0],[0,1,2],1.0), "cdf")
    assert mc.crps_from_cdf([0.0,0.5,1.0],[0.0,5.0,10.0],5.0)==1.25

ALL_TESTS=[v for k,v in sorted(globals().items()) if k.startswith("test_")]
if __name__=="__main__":
    for fn in ALL_TESTS:
        try: fn(); check(fn.__name__,True)
        except AssertionError as e: check(fn.__name__,False,str(e))
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
