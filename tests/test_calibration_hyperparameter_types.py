"""
test_calibration_hyperparameter_types.py — SF-BL-11097 verification.

Pre-registered done/kill criteria (TASK.md):
  "constructors raise ValueError naming the hyperparameter on any
   bool or non-numeric max_iter, tol or l2 (PlattScaler) and l2
   (BetaCalibrator) at fit time or construction, with
   numeric-hyperparameter fits bit-identical; kill if max_iter=True,
   l2=True or any string hyperparameter returns fitted parameters or
   raises a bare TypeError instead of the named ValueError."

Bug (live repros, 4-row panel y=[0,1,0,1], p=[0.2,0.8,0.3,0.7]):
  PlattScaler(max_iter=True).fit(...)  -> A_=2.159727364605987,
      n_iter_=1 (bool read as range(1): a single Newton iteration,
      vs A_=13.828830393498272 for the full fit on this panel;
      TASK.md's panel gave 1.8626 vs 10.9247)
  PlattScaler(l2=True).fit(...)        -> A_=1.032992625325645
      (bool read as l2=1.0, 1e6x the 1e-6 default)
  PlattScaler(max_iter='100').fit(...) -> bare TypeError:
      'str' object cannot be interpreted as an integer
  PlattScaler(l2='1e-6').fit(...)      -> bare UFuncTypeError
  BetaCalibrator(max_iter=True).fit(...) -> a_=1.6922885738473261
      (single solver iteration, vs a_=13.077263360344004 clean)
  BetaCalibrator(max_iter='100')       -> bare TypeError
  (Platt tol and Beta l2/tol type cases were already guarded by
   SF-BL-9683 / SF-BL-9682 / SF-BL-9718 and are re-asserted here;
   Platt l2 was guarded only in the superforecaster_out mirror
   under SF-BL-9681, not in this canonical copy.)

Fix (superforecaster/calibration.py): _validate_max_iter (new,
SF-BL-11097) and _validate_platt_l2 (new in this copy) run at
construction and again at the top of fit() — before the
intercept-only early return — for PlattScaler; _validate_max_iter
also runs in BetaCalibrator.__init__/fit and inside the shared
_logistic_fit_newton solver. All raise ValueError naming the
hyperparameter. Pure validation: lawful fits are bit-identical
(pre/post SHA-256 ce7f802fcbdf833e7a05923306... over 4 panels x
7 lawful configurations; lawful_baseline_prefix.json in the task
work dir SF-BL-11097).

Run:  python3 tests/test_calibration_hyperparameter_types.py
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from superforecaster.calibration import BetaCalibrator, PlattScaler  # noqa: E402

PASS = []

Y4 = [0, 1, 0, 1]
P4 = [0.2, 0.8, 0.3, 0.7]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))


def expect_named_valueerror(name, fn, param):
    try:
        fn()
    except ValueError as e:
        check(name, type(e) is ValueError and param in str(e),
              f"-> ValueError: {e}")
    except Exception as e:  # noqa: BLE001
        check(name, False, f"-> WRONG TYPE {type(e).__name__}: {e}")
    else:
        check(name, False, "-> no error; silent construction/fit")


BAD_MAX_ITER = [True, False, "100", 2.5, 100.0, None, [100], 0, -3]
BAD_TOL = [True, "1e-10", None, float("nan"), 0.0, -1.0]
BAD_L2 = [True, False, "1e-6", None, float("nan"), float("inf"), -1.0]

for v in BAD_MAX_ITER:
    expect_named_valueerror(f"Platt max_iter={v!r}",
                            lambda v=v: PlattScaler(max_iter=v), "max_iter")
    expect_named_valueerror(f"Beta max_iter={v!r}",
                            lambda v=v: BetaCalibrator(max_iter=v), "max_iter")
for v in BAD_TOL:
    expect_named_valueerror(f"Platt tol={v!r}",
                            lambda v=v: PlattScaler(tol=v), "tol")
    expect_named_valueerror(f"Beta tol={v!r}",
                            lambda v=v: BetaCalibrator(tol=v), "tol")
for v in BAD_L2:
    expect_named_valueerror(f"Platt l2={v!r}",
                            lambda v=v: PlattScaler(l2=v), "l2")
    expect_named_valueerror(f"Beta l2={v!r}",
                            lambda v=v: BetaCalibrator(l2=v), "l2")

# Fit-time mutation cannot bypass the screens.
for cls, attr, bad in [(PlattScaler, "max_iter", True),
                       (PlattScaler, "l2", "1e-6"),
                       (BetaCalibrator, "max_iter", "100"),
                       (BetaCalibrator, "l2", True)]:
    def mutated(cls=cls, attr=attr, bad=bad):
        obj = cls()
        setattr(obj, attr, bad)
        return obj.fit(Y4, P4)
    expect_named_valueerror(f"{cls.__name__} mutated {attr}={bad!r} at fit",
                            mutated, attr)

# Lawful fits are bit-identical to the pre-fix baselines.
platt = PlattScaler().fit(Y4, P4)
check("Platt default bit-identical",
      repr(platt.A_) == "13.828830393498272"
      and repr(platt.B_) == "5.647169839504838e-12"
      and platt.n_iter_ == 17,
      f"-> A_={platt.A_!r} B_={platt.B_!r} n_iter_={platt.n_iter_}")
beta = BetaCalibrator().fit(Y4, P4)
check("Beta default bit-identical",
      repr(beta.a_) == "13.077263360344004"
      and repr(beta.b_) == "-13.077263360332795"
      and repr(beta.c_) == "3.6097333804048382e-12",
      f"-> a_={beta.a_!r} b_={beta.b_!r} c_={beta.c_!r}")
check("Platt np-int max_iter lawful",
      PlattScaler(max_iter=np.int64(37)).fit(Y4, P4).A_ == platt.A_)
check("Beta custom lawful",
      np.isfinite(BetaCalibrator(l2=0.5, max_iter=25).fit(Y4, P4).a_))

print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
print("RESULT:", "ALL PASS" if all(PASS) else "FAILURES PRESENT")
if __name__ == "__main__":
    sys.exit(0 if all(PASS) else 1)
