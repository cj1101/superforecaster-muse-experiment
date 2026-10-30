"""
test_cdf_module_climatology_probs_type_validation.py — SF-BL-11243.

Pre-registered done/kill criteria (TASK.md):
  "climatology_percentiles raises ValueError naming probs on any bool
   or string-dtype probs element, with numeric probs dicts
   bit-identical; kill if probs=[True] still returns a dict keyed 1.0
   or probs=['0.5'] raises a bare TypeError instead of the named guard."

Bug: climatology_percentiles never type-validated probs. Live repro
on 40 trailing returns (default_rng(0), spot 100): probs=[0.5]
control returned {0.5: 99.742}, probs=[True] returned
{1.0: 103.032} (bool read as the 100th-percentile request, stored as
dict key 1.0), probs=[True, False] returned keys {1.0, 0.0}, and
probs=['0.5'] died with a bare "'<=' not supported" TypeError inside
np.quantile sorting. SF-BL-10266 covers this function's spot value,
not the probs argument's type (its suite must stay green).

Fix (metaculus_bot/cdf_module.py): _screen_probs_types_11243 inspects
the RAW probs (container, ndarray dtype, each element) at entry,
before any float coercion / spot screen, and raises ValueError naming
probs + SF-BL-11243 on any bool or string/bytes element (and any other
non-numeric element / non-sequence). Numeric probs are returned as
the identical float array, so numeric dicts stay bit-identical:
expected raw bytes below were captured live from the pre-fix
implementation.

Run:  python3 tests/test_cdf_module_climatology_probs_type_validation.py
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metaculus_bot"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import cdf_module as cm  # noqa: E402

RNG = np.random.default_rng(0)
RETS40 = RNG.normal(0.0, 0.02, 40)


def _raises_named(probs, spot=100.0, rets=RETS40):
    with pytest.raises(ValueError) as excinfo:
        cm.climatology_percentiles(spot, rets, probs)
    msg = str(excinfo.value)
    assert "probs" in msg and "SF-BL-11243" in msg, msg
    assert "climatology_percentiles" in msg, msg


def test_task_repros_raise_named():
    _raises_named([True])          # pre-fix: {1.0: 103.032}
    _raises_named([True, False])   # pre-fix: keys {1.0, 0.0}
    _raises_named(["0.5"])         # pre-fix: bare TypeError


def test_bool_elements_raise():
    for bad in ([True], [False], [0.5, True], [np.bool_(True)],
                np.array([True, False]), np.array([0.5, True],
                dtype=object)):
        _raises_named(bad)


def test_string_elements_raise():
    for bad in (["0.5"], [0.5, "0.9"], [b"0.5"], [np.str_("0.5")],
                np.array(["0.5", "0.9"]), np.array([b"0.5"]),
                np.array(["0.5"], dtype=object)):
        _raises_named(bad)


def test_container_and_scalar_probs_raise():
    for bad in (True, False, "0.5", b"0.5", 0.5, 5, None):
        _raises_named(bad)


def test_other_nonnumeric_elements_raise_named():
    for bad in ([None], [0.5, None], [[0.5]], [object()]):
        _raises_named(bad)


def test_no_bool_or_string_probs_returns_a_dict():
    for bad in ([True], [True, False], ["0.5"], "0.5", True):
        try:
            out = cm.climatology_percentiles(100.0, RETS40, bad)
        except ValueError:
            continue
        raise AssertionError(f"probs={bad!r} returned {out!r}")


def test_probs_screened_before_spot_screen():
    # Bad probs + bad spot: the error must name probs.
    _raises_named([True], spot=float("nan"))


def test_spot_screen_preserved_for_lawful_probs():
    with pytest.raises(ValueError) as excinfo:
        cm.climatology_percentiles(float("nan"), RETS40, [0.5])
    assert "spot" in str(excinfo.value) and "SF-BL-10266" in str(
        excinfo.value)


def test_returns_screen_preserved_for_lawful_probs():
    with pytest.raises(ValueError) as excinfo:
        cm.climatology_percentiles(100.0, [0.01] * 29, [0.5])
    assert ">= 30" in str(excinfo.value)


def test_clean_panels_bit_identical():
    rng = np.random.default_rng(10266)
    r_norm = rng.normal(0.0, 0.02, 40)
    panels = {
        "task40_default": ((RETS40, None),
            "9a9999999999a93f000000000000d03f000000000000e03f000000000000e83f666666666666ee3f",
            "bf089d445f62584056a21e21c0af58406422ebf37fef584016cfaf9c103059405a5bc49684a95940"),
        "task40_p50": ((RETS40, [0.5]),
            "000000000000e03f", "6422ebf37fef5840"),
        "custom_tuple": ((r_norm, (0.1, 0.5, 0.9)),
            "9a9999999999b93f000000000000e03fcdccccccccccec3f",
            "0f2c5142f82f584011e2b589c9dc5840b1b6f1ba49985940"),
        "list_floats": ((RETS40, [0.05, 0.25, 0.5, 0.75, 0.95]),
            "9a9999999999a93f000000000000d03f000000000000e03f000000000000e83f666666666666ee3f",
            "bf089d445f62584056a21e21c0af58406422ebf37fef584016cfaf9c103059405a5bc49684a95940"),
        "np_array": ((RETS40, np.array([0.05, 0.5, 0.95])),
            "9a9999999999a93f000000000000e03f666666666666ee3f",
            "bf089d445f6258406422ebf37fef58405a5bc49684a95940"),
        "int_probs": ((RETS40, [0, 1]),
            "0000000000000000000000000000f03f",
            "c1f0099d35dd5740951663eb0ac25940"),
    }
    for name, ((rets, probs), keys_hex, vals_hex) in panels.items():
        d = (cm.climatology_percentiles(100.0, rets) if probs is None
             else cm.climatology_percentiles(100.0, rets, probs))
        assert np.array(list(d.keys())).tobytes().hex() == keys_hex, name
        assert np.array(list(d.values())).tobytes().hex() == vals_hex, name


def test_control_value_matches_task_repro():
    d = cm.climatology_percentiles(100.0, RETS40, [0.5])
    assert list(d.keys()) == [0.5]
    assert abs(d[0.5] - 99.74218461954439) == 0.0


def test_disciplined_percentiles_inherits_type_screen():
    daily = RNG.normal(0.0, 0.01, 60)
    recent = RNG.normal(0.0, 0.012, 30)
    for bad in ([True], ["0.5"]):
        with pytest.raises(ValueError) as excinfo:
            cm.disciplined_percentiles(100.0, RETS40, daily, recent, bad)
        assert "probs" in str(excinfo.value)
    disc = cm.disciplined_percentiles(100.0, RETS40, daily, recent)
    vals = np.array(list(disc.values()))
    assert np.all(np.isfinite(vals)) and np.all(np.diff(vals) > 0)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed.append(fn.__name__)
            print(f"FAIL  {fn.__name__}: {exc!r}")
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
