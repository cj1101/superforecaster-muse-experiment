"""
test_cdf_module_crps_grid_outcome_validation.py — SF-BL-9924 verification.

Pre-registered: "Fix passes when crps_from_cdf() raises a named
ValueError on non-increasing nominals, non-finite cdf/nominals/y, or a
CDF outside [0,1], and reproduces the control CRPS on a sorted valid
grid; kill if any production caller passes deliberately reversed
grids (then sort-and-flag, never integrate silently)."

Kill check (audited in work dir RESULTS.md): no production caller
passes a reversed grid — evaluate.py:96 and run_hc2.py:76,128 build
nominals via grid_nominals(lo, hi) lo<hi; oob_budget.scaled_crps
validates span>0 before delegating. So raising is correct.

Run: python3 tests/test_cdf_module_crps_grid_outcome_validation.py
"""
import os, struct, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metaculus_bot"))
import numpy as np  # noqa: E402
import cdf_module as cm  # noqa: E402
PASS=[]
def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}"); PASS.append(bool(cond))
def _raises_named(fn, *needles):
    try:
        fn()
    except ValueError as e:
        msg=str(e)
        assert "SF-BL-9924" in msg, f"missing task id: {msg!r}"
        for n in needles: assert n in msg, f"missing {n!r} in {msg!r}"
        return msg
    raise AssertionError("no ValueError raised — bad input returned a score")
def _bits(v): return struct.pack("<d", float(v)).hex()

def test_reversed_nominals_raise():
    _raises_named(lambda: cm.crps_from_cdf([0.0,0.5,1.0],[2,1,0],1.0), "nominals", "non-increasing")
    _raises_named(lambda: cm.crps_from_cdf([0.2,0.5,0.9],[2,1,0],1.0), "nominals")
def test_unsorted_and_duplicate_nominals_raise():
    for n in ([0,2,1],[0,1,1],[2,1,0],[0,0,1],[3,2,1]):
        _raises_named(lambda n=n: cm.crps_from_cdf([0.0,0.5,1.0],n,1.0), "nominals")
def test_nonfinite_y_raises():
    for y in (float("nan"), float("inf"), float("-inf")):
        _raises_named(lambda y=y: cm.crps_from_cdf([0.0,0.5,1.0],[0,1,2],y), "y")
def test_nonfinite_cdf_raises_any_position():
    for bad in (float("nan"), float("inf"), float("-inf")):
        for i in range(3):
            c=[0.0,0.5,1.0]; c[i]=bad
            _raises_named(lambda c=c: cm.crps_from_cdf(c,[0,1,2],1.0), "cdf")
def test_nonfinite_nominals_raise_any_position():
    for bad in (float("nan"), float("inf"), float("-inf")):
        for i in range(3):
            n=[0.0,1.0,2.0]; n[i]=bad
            _raises_named(lambda n=n: cm.crps_from_cdf([0.0,0.5,1.0],n,1.0), "nominals")
def test_cdf_outside_01_raises():
    for c in ([0.0,0.5,1.5],[0.0,-0.1,1.0],[-0.5,0.5,1.0],[0.0,0.5,1.000001],[2.0,2.0,2.0]):
        _raises_named(lambda c=c: cm.crps_from_cdf(c,[0,1,2],1.0), "cdf", "[0, 1]")
def test_cdf_boundary_values_lawful():
    # 0 and 1 exactly are inside [0,1] and must not raise
    assert cm.crps_from_cdf([0.0,0.5,1.0],[0,1,2],1.0)==0.25
    assert cm.crps_from_cdf([0.0,1.0],[0.0,10.0],5.0)==0.0
def test_no_bad_input_returns_finite_score_kill_sweep():
    bads=[]
    for n in ([2,1,0],[0,2,1],[0,1,1]): bads.append(([0.0,0.5,1.0],n,1.0))
    for y in (float("nan"),float("inf")): bads.append(([0.0,0.5,1.0],[0,1,2],y))
    bads += [([0.0,float("nan"),1.0],[0,1,2],1.0),([0.0,0.5,1.0],[0,float("nan"),2],1.0),([0.0,0.5,1.5],[0,1,2],1.0)]
    for c,n,y in bads:
        try: v=cm.crps_from_cdf(c,n,y)
        except ValueError: continue
        raise AssertionError(f"returned {v!r} for cdf={c} nominals={n} y={y} (kill)")
def test_control_sorted_valid_grid_bit_identical():
    cases=[([0.0,0.5,1.0],[0,1,2],1.0,"000000000000d03f"),([0.0,0.5,1.0],[0.0,5.0,10.0],5.0,"000000000000f43f"),([0.0,1.0],[0.0,10.0],5.0,"0000000000000000"),([0.1,0.4,0.6,0.9],[0,1,2,3],1.5,"2085eb51b81ed53f"),([0.0,0.5,1.0],[-10.0,0.0,10.0],0.0,"0000000000000440")]
    for c,n,y,b in cases:
        v=cm.crps_from_cdf(c,n,y); assert _bits(v)==b, f"{_bits(v)} != {b}"
def test_201pt_scaffold_grid_bit_identical():
    nom=cm.grid_nominals(0,100); cdf=cm.percentiles_to_cdf({0.05:10,0.25:30,0.5:50,0.75:70,0.95:90},0,100)
    exp={0:"d9cef75363754140",50:"c976be9f1aab1b40",15.0:"e6d022dbf9dd3540"}
    for y,b in exp.items(): assert _bits(cm.crps_from_cdf(cdf,nom,y))==b
def test_degenerate_still_raises_10269():
    try: cm.crps_from_cdf([],[],0.5)
    except ValueError as e: assert "SF-BL-10269" in str(e); return
    raise AssertionError("degenerate did not raise")
def test_client_adapter_raises_and_matches():
    import metaculus_client as mc
    assert mc.crps_from_cdf is cm.crps_from_cdf
    _raises_named(lambda: mc.crps_from_cdf([0.0,0.5,1.0],[2,1,0],1.0))
    _raises_named(lambda: mc.crps_from_cdf([0.0,0.5,1.0],[0,1,2],float("nan")))
    assert mc.crps_from_cdf([0.0,0.5,1.0],[0.0,5.0,10.0],5.0)==1.25

ALL_TESTS=[v for k,v in sorted(globals().items()) if k.startswith("test_")]
if __name__=="__main__":
    for fn in ALL_TESTS:
        try: fn(); check(fn.__name__,True)
        except AssertionError as e: check(fn.__name__,False,str(e))
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
