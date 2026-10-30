"""
test_floor_mass_validation.py — SF-BL-11242 verification.

Pre-registered done/kill criteria (TASK.md):
  "_cell_density raises ValueError naming floor_mass on any bool, string,
   non-finite or negative floor_mass, with lawful floor densities
   bit-identical; kill if floor_mass=True still returns (0.5, 0.5) or
   floor_mass=nan returns a NaN pair."

Bug: cdf_module._cell_density (metaculus_bot/cdf_module.py) never
validated its floor_mass argument. Live pre-fix repro on the control
panel _cell_density([0.1,0.5,0.9],[0,1,2],y=1.0):
  control (default 1e-4) -> (0.40001, 0.40001)
  floor_mass=nan         -> (nan, nan)
  floor_mass=True        -> (0.5, 0.5)   (bool read as a 1.0 floor)
  floor_mass=False       -> (0.4, 0.4)   (bool read as a 0.0 floor)
  floor_mass=-0.5        -> (0.35, 0.35)
  floor_mass='1e-4'      -> bare TypeError naming nothing
  floor_mass=None        -> bare TypeError naming nothing
  floor_mass=+/-inf      -> (nan, nan)

Fix: an entry guard in _cell_density — bool / string / non-numeric /
non-scalar floor_mass, non-finite floor_mass and negative floor_mass
each raise a ValueError naming floor_mass and SF-BL-11242. The float
copy is used only for the checks; the cdf_mix computation keeps the
original object, so lawful floors are bit-identical — the expected
hex bit patterns below were recorded from the pre-fix implementation
(the current file with the guard stripped; see repro.py / RESULTS.md).

Module location: repo layout (../../metaculus_bot next to tests/) if
present, else the production copy in superforecaster_out/metaculus_bot;
override with CDF_MODULE_DIR.

Run:  python3 test_floor_mass_validation.py
No pytest dependency: test_* functions with assert + __main__ runner,
same convention as the fleet's other cdf_module suites.
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_repo_bot = os.path.join(_here, "..", "..", "metaculus_bot")
_default = (_repo_bot if os.path.exists(os.path.join(_repo_bot, "cdf_module.py"))
            else os.path.expanduser(
                "~/workspace/superforecaster_out/metaculus_bot"))
sys.path.insert(0, os.environ.get("CDF_MODULE_DIR", _default))

import numpy as np  # noqa: E402
import cdf_module as cm  # noqa: E402

PASS = []
CDF = [0.1, 0.5, 0.9]
NOM = [0, 1, 2]
NAN, INF = float("nan"), float("inf")

BAD_FLOORS = [
    # bool (kill criterion: True must not return (0.5, 0.5))
    True, False, np.bool_(True), np.bool_(False),
    # string / bytes (pre-fix: bare TypeError; must be ValueError)
    "1e-4", "0.5", "", b"1e-4",
    # non-finite (kill criterion: nan must not return a NaN pair)
    NAN, INF, -INF, np.float64("nan"), np.float64("inf"),
    # negative
    -0.5, -1e-12, -1, np.float64(-0.25),
    # other non-numeric / non-scalar (caller bugs, same named guard)
    None, [1e-4], (1e-4,), {"floor": 1e-4}, complex(1e-4, 0),
    np.array([1e-4]), np.array([0.1, 0.2]), object(),
]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _raises_named_valueerror(fm):
    """floor_mass=fm must raise ValueError (never TypeError, never a
    return) whose message names floor_mass."""
    try:
        out = cm._cell_density(CDF, NOM, 1.0, floor_mass=fm)
    except ValueError as e:
        assert "floor_mass" in str(e), f"message does not name floor_mass: {e}"
        return
    except TypeError as e:
        raise AssertionError(f"bare TypeError leaked for {fm!r}: {e}")
    raise AssertionError(f"floor_mass={fm!r} returned {out} instead of raising")


def test_bool_floor_raises_named():
    for fm in (True, False, np.bool_(True), np.bool_(False)):
        _raises_named_valueerror(fm)


def test_string_floor_raises_valueerror_not_typeerror():
    for fm in ("1e-4", "0.5", "", b"1e-4", np.str_("1e-4")):
        _raises_named_valueerror(fm)


def test_nonfinite_floor_raises_named():
    for fm in (NAN, INF, -INF, np.float64("nan"), np.float32("inf")):
        _raises_named_valueerror(fm)


def test_negative_floor_raises_named():
    for fm in (-0.5, -1e-12, -1, -1.0, np.float64(-0.25)):
        _raises_named_valueerror(fm)


def test_non_numeric_floor_raises_named():
    for fm in (None, [1e-4], (1e-4,), {"floor": 1e-4}, complex(1e-4, 0),
               np.array([1e-4]), object()):
        _raises_named_valueerror(fm)


def test_kill_criterion_sweep_no_bad_floor_returns():
    # Kill: "floor_mass=True still returns (0.5, 0.5) or floor_mass=nan
    # returns a NaN pair" — generalised: NO bad floor returns anything.
    for fm in BAD_FLOORS:
        _raises_named_valueerror(fm)
    # Explicitly: the two named kill cases cannot reproduce.
    for fm in (True, NAN):
        try:
            cm._cell_density(CDF, NOM, 1.0, floor_mass=fm)
            raise AssertionError(f"kill case floor_mass={fm!r} returned")
        except ValueError:
            pass


def test_guard_fires_at_every_branch_and_grid():
    # The guard is at entry: it fires whatever y / grid branch would
    # have served the density, and names floor_mass even alongside a
    # degenerate grid (entry validation precedes the SF-BL-10110
    # grid screens).
    for y in (-0.5, 0.0, 1.0, 2.0, 2.5):
        try:
            cm._cell_density(CDF, NOM, y, floor_mass=True)
            raise AssertionError(f"y={y}: bool floor returned")
        except ValueError as e:
            assert "floor_mass" in str(e)
    try:
        cm._cell_density([0.5], [0], 0.0, floor_mass=NAN)
        raise AssertionError("degenerate grid + nan floor returned")
    except ValueError as e:
        assert "floor_mass" in str(e), str(e)


# --- Lawful floors: bit-identical to the pre-fix implementation -------
# (floor label -> object, y) -> (density.hex(), mass.hex()), recorded
# pre-fix on the control panel CDF/NOM above (all branches: below /
# at-lo / interior / at-hi / above).
FLOOR_OBJECTS = {
    "default": None, "0": 0, "0.0": 0.0, "1e-5": 1e-5, "1e-4": 1e-4,
    "1e-3": 1e-3, "0.01": 0.01, "0.1": 0.1, "1": 1, "1.0": 1.0, "2.0": 2.0,
    "np.float64(1e-4)": np.float64(1e-4),
    "np.float32(0.001)": np.float32(0.001),
}
PANEL = {
    ('default', -0.5): ('0x1.999c38b04ab60p-2', '0x1.3331e3a7daa50p-2'),
    ('default', 0.0): ('0x1.999c38b04ab60p-2', '0x1.3331e3a7daa50p-2'),
    ('default', 0.5): ('0x1.999c38b04ab60p-2', '0x1.999c38b04ab60p-2'),
    ('default', 1.0): ('0x1.999c38b04ab60p-2', '0x1.999c38b04ab60p-2'),
    ('default', 1.5): ('0x1.999c38b04ab60p-2', '0x1.999c38b04ab60p-2'),
    ('default', 2.0): ('0x1.999c38b04ab60p-2', '0x1.0000000000000p-1'),
    ('default', 2.5): ('0x1.999c38b04ab60p-2', '0x1.0000000000000p-1'),
    ('0', -0.5): ('0x1.999999999999ap-2', '0x1.3333333333334p-2'),
    ('0', 0.0): ('0x1.999999999999ap-2', '0x1.3333333333334p-2'),
    ('0', 0.5): ('0x1.999999999999ap-2', '0x1.999999999999ap-2'),
    ('0', 1.0): ('0x1.999999999999ap-2', '0x1.999999999999ap-2'),
    ('0', 1.5): ('0x1.999999999999ap-2', '0x1.999999999999ap-2'),
    ('0', 2.0): ('0x1.999999999999ap-2', '0x1.0000000000000p-1'),
    ('0', 2.5): ('0x1.999999999999ap-2', '0x1.0000000000000p-1'),
    ('0.0', -0.5): ('0x1.999999999999ap-2', '0x1.3333333333334p-2'),
    ('0.0', 0.0): ('0x1.999999999999ap-2', '0x1.3333333333334p-2'),
    ('0.0', 0.5): ('0x1.999999999999ap-2', '0x1.999999999999ap-2'),
    ('0.0', 1.0): ('0x1.999999999999ap-2', '0x1.999999999999ap-2'),
    ('0.0', 1.5): ('0x1.999999999999ap-2', '0x1.999999999999ap-2'),
    ('0.0', 2.0): ('0x1.999999999999ap-2', '0x1.0000000000000p-1'),
    ('0.0', 2.5): ('0x1.999999999999ap-2', '0x1.0000000000000p-1'),
    ('1e-5', -0.5): ('0x1.9999dcb5781c7p-2', '0x1.333311a543f1cp-2'),
    ('1e-5', 0.0): ('0x1.9999dcb5781c7p-2', '0x1.333311a543f1cp-2'),
    ('1e-5', 0.5): ('0x1.9999dcb5781c7p-2', '0x1.9999dcb5781c7p-2'),
    ('1e-5', 1.0): ('0x1.9999dcb5781c7p-2', '0x1.9999dcb5781c7p-2'),
    ('1e-5', 1.5): ('0x1.9999dcb5781c8p-2', '0x1.9999dcb5781c8p-2'),
    ('1e-5', 2.0): ('0x1.9999dcb5781c8p-2', '0x1.0000000000000p-1'),
    ('1e-5', 2.5): ('0x1.9999dcb5781c8p-2', '0x1.0000000000000p-1'),
    ('1e-4', -0.5): ('0x1.999c38b04ab60p-2', '0x1.3331e3a7daa50p-2'),
    ('1e-4', 0.0): ('0x1.999c38b04ab60p-2', '0x1.3331e3a7daa50p-2'),
    ('1e-4', 0.5): ('0x1.999c38b04ab60p-2', '0x1.999c38b04ab60p-2'),
    ('1e-4', 1.0): ('0x1.999c38b04ab60p-2', '0x1.999c38b04ab60p-2'),
    ('1e-4', 1.5): ('0x1.999c38b04ab60p-2', '0x1.999c38b04ab60p-2'),
    ('1e-4', 2.0): ('0x1.999c38b04ab60p-2', '0x1.0000000000000p-1'),
    ('1e-4', 2.5): ('0x1.999c38b04ab60p-2', '0x1.0000000000000p-1'),
    ('1e-3', -0.5): ('0x1.99b3d07c84b5ep-2', '0x1.332617c1bda51p-2'),
    ('1e-3', 0.0): ('0x1.99b3d07c84b5ep-2', '0x1.332617c1bda51p-2'),
    ('1e-3', 0.5): ('0x1.99b3d07c84b5ep-2', '0x1.99b3d07c84b5ep-2'),
    ('1e-3', 1.0): ('0x1.99b3d07c84b5ep-2', '0x1.99b3d07c84b5ep-2'),
    ('1e-3', 1.5): ('0x1.99b3d07c84b5ep-2', '0x1.99b3d07c84b5ep-2'),
    ('1e-3', 2.0): ('0x1.99b3d07c84b5ep-2', '0x1.0000000000000p-1'),
    ('1e-3', 2.5): ('0x1.99b3d07c84b5ep-2', '0x1.0000000000000p-1'),
    ('0.01', -0.5): ('0x1.9a9fbe76c8b44p-2', '0x1.32b020c49ba5ep-2'),
    ('0.01', 0.0): ('0x1.9a9fbe76c8b44p-2', '0x1.32b020c49ba5ep-2'),
    ('0.01', 0.5): ('0x1.9a9fbe76c8b44p-2', '0x1.9a9fbe76c8b44p-2'),
    ('0.01', 1.0): ('0x1.9a9fbe76c8b44p-2', '0x1.9a9fbe76c8b44p-2'),
    ('0.01', 1.5): ('0x1.9a9fbe76c8b44p-2', '0x1.9a9fbe76c8b44p-2'),
    ('0.01', 2.0): ('0x1.9a9fbe76c8b44p-2', '0x1.0000000000000p-1'),
    ('0.01', 2.5): ('0x1.9a9fbe76c8b44p-2', '0x1.0000000000000p-1'),
    ('0.1', -0.5): ('0x1.a3d70a3d70a3dp-2', '0x1.2e147ae147ae1p-2'),
    ('0.1', 0.0): ('0x1.a3d70a3d70a3dp-2', '0x1.2e147ae147ae1p-2'),
    ('0.1', 0.5): ('0x1.a3d70a3d70a3dp-2', '0x1.a3d70a3d70a3dp-2'),
    ('0.1', 1.0): ('0x1.a3d70a3d70a3dp-2', '0x1.a3d70a3d70a3dp-2'),
    ('0.1', 1.5): ('0x1.a3d70a3d70a3ep-2', '0x1.a3d70a3d70a3ep-2'),
    ('0.1', 2.0): ('0x1.a3d70a3d70a3ep-2', '0x1.0000000000000p-1'),
    ('0.1', 2.5): ('0x1.a3d70a3d70a3ep-2', '0x1.0000000000000p-1'),
    ('1', -0.5): ('0x1.0000000000000p-1', '0x1.0000000000000p-2'),
    ('1', 0.0): ('0x1.0000000000000p-1', '0x1.0000000000000p-2'),
    ('1', 0.5): ('0x1.0000000000000p-1', '0x1.0000000000000p-1'),
    ('1', 1.0): ('0x1.0000000000000p-1', '0x1.0000000000000p-1'),
    ('1', 1.5): ('0x1.0000000000000p-1', '0x1.0000000000000p-1'),
    ('1', 2.0): ('0x1.0000000000000p-1', '0x1.0000000000000p-1'),
    ('1', 2.5): ('0x1.0000000000000p-1', '0x1.0000000000000p-1'),
    ('1.0', -0.5): ('0x1.0000000000000p-1', '0x1.0000000000000p-2'),
    ('1.0', 0.0): ('0x1.0000000000000p-1', '0x1.0000000000000p-2'),
    ('1.0', 0.5): ('0x1.0000000000000p-1', '0x1.0000000000000p-1'),
    ('1.0', 1.0): ('0x1.0000000000000p-1', '0x1.0000000000000p-1'),
    ('1.0', 1.5): ('0x1.0000000000000p-1', '0x1.0000000000000p-1'),
    ('1.0', 2.0): ('0x1.0000000000000p-1', '0x1.0000000000000p-1'),
    ('1.0', 2.5): ('0x1.0000000000000p-1', '0x1.0000000000000p-1'),
    ('2.0', -0.5): ('0x1.3333333333333p-1', '0x1.9999999999999p-3'),
    ('2.0', 0.0): ('0x1.3333333333333p-1', '0x1.9999999999999p-3'),
    ('2.0', 0.5): ('0x1.3333333333333p-1', '0x1.3333333333333p-1'),
    ('2.0', 1.0): ('0x1.3333333333333p-1', '0x1.3333333333333p-1'),
    ('2.0', 1.5): ('0x1.3333333333334p-1', '0x1.3333333333334p-1'),
    ('2.0', 2.0): ('0x1.3333333333334p-1', '0x1.0000000000000p-1'),
    ('2.0', 2.5): ('0x1.3333333333334p-1', '0x1.0000000000000p-1'),
    ('np.float64(1e-4)', -0.5): ('0x1.999c38b04ab60p-2', '0x1.3331e3a7daa50p-2'),
    ('np.float64(1e-4)', 0.0): ('0x1.999c38b04ab60p-2', '0x1.3331e3a7daa50p-2'),
    ('np.float64(1e-4)', 0.5): ('0x1.999c38b04ab60p-2', '0x1.999c38b04ab60p-2'),
    ('np.float64(1e-4)', 1.0): ('0x1.999c38b04ab60p-2', '0x1.999c38b04ab60p-2'),
    ('np.float64(1e-4)', 1.5): ('0x1.999c38b04ab60p-2', '0x1.999c38b04ab60p-2'),
    ('np.float64(1e-4)', 2.0): ('0x1.999c38b04ab60p-2', '0x1.0000000000000p-1'),
    ('np.float64(1e-4)', 2.5): ('0x1.999c38b04ab60p-2', '0x1.0000000000000p-1'),
    ('np.float32(0.001)', -0.5): ('0x1.99b3d07c9999ap-2', '0x1.332617c1b3334p-2'),
    ('np.float32(0.001)', 0.0): ('0x1.99b3d07c9999ap-2', '0x1.332617c1b3334p-2'),
    ('np.float32(0.001)', 0.5): ('0x1.99b3d07c9999ap-2', '0x1.99b3d07c9999ap-2'),
    ('np.float32(0.001)', 1.0): ('0x1.99b3d07c9999ap-2', '0x1.99b3d07c9999ap-2'),
    ('np.float32(0.001)', 1.5): ('0x1.99b3d07c9999ap-2', '0x1.99b3d07c9999ap-2'),
    ('np.float32(0.001)', 2.0): ('0x1.99b3d07c9999ap-2', '0x1.0000000000000p-1'),
    ('np.float32(0.001)', 2.5): ('0x1.99b3d07c9999ap-2', '0x1.0000000000000p-1'),
}


def test_lawful_floors_bit_identical():
    for (label, y), (dhex, mhex) in PANEL.items():
        fm = FLOOR_OBJECTS[label]
        kw = {} if fm is None else {"floor_mass": fm}
        dens, mass = cm._cell_density(CDF, NOM, y, **kw)
        assert float(dens).hex() == dhex, (label, y, float(dens).hex(), dhex)
        assert float(mass).hex() == mhex, (label, y, float(mass).hex(), mhex)
    # Task's control value, exactly:
    assert cm._cell_density(CDF, NOM, 1.0) == (0.40001, 0.40001)


def test_wrappers_default_floor_bit_identical():
    want = {
        -0.5: ('-0x1.d520c8345fd2ep-1', '-0x1.3438a76b3cf40p+0'),
        0.0: ('-0x1.d520c8345fd2ep-1', '-0x1.3438a76b3cf40p+0'),
        0.5: ('-0x1.d520c8345fd2ep-1', '-0x1.d520c8345fd2ep-1'),
        1.0: ('-0x1.d520c8345fd2ep-1', '-0x1.d520c8345fd2ep-1'),
        1.5: ('-0x1.d520c8345fd2ep-1', '-0x1.d520c8345fd2ep-1'),
        2.0: ('-0x1.d520c8345fd2ep-1', '-0x1.62e42fefa39efp-1'),
        2.5: ('-0x1.d520c8345fd2ep-1', '-0x1.62e42fefa39efp-1'),
    }
    for y, (lhex, ihex) in want.items():
        assert cm.log_score_from_cdf(CDF, NOM, y).hex() == lhex, y
        assert cm.interval_log_score_from_cdf(CDF, NOM, y).hex() == ihex, y


def test_audit_floors_from_sf_bl_8742_still_work():
    # SF-BL-8742's sensitivity grid must keep scoring (it calls
    # _cell_density directly with each floor).
    for f, want in ((1e-5, 0.400001), (1e-4, 0.40001), (1e-3, 0.4001)):
        dens, mass = cm._cell_density(CDF, NOM, 1.0, floor_mass=f)
        assert dens == want and mass == want, (f, dens, mass)


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
