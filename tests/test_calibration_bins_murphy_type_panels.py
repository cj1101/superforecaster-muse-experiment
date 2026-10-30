"""
test_calibration_bins_murphy_type_panels.py — SF-BL-11425 verification.

Pre-registered done/kill criteria (TASK.md):
  "Guard raises ValueError naming y/p on any string, object-dtype or
   bool element in calibration_bins and murphy_decomposition, with
   float panels bit-identical; kill if string panels still return the
   control bin table / Murphy brier 0.01 or the bool panel still
   returns brier 0.0 instead of raising."

Bug: calibration_bins screened only a bool p (SF-BL-10938) and a
non-pandas y via _as_y, and murphy_decomposition float-converted both
panels BEFORE delegating to calibration_bins / brier_score, so no
downstream screen could see its raw types:
  calibration_bins([1,0], ['0.9','0.1']) == numeric-control bin table
  murphy_decomposition([1,0], ['0.9','0.1'])['brier'] == 0.01
  murphy_decomposition([1,0], [True,False])['brier'] == 0.0 (pandas form)

Fix: murphy_decomposition gained its OWN explicit raw screen
_reject_murphy_panel (SF-BL-11425; messages carry the SF-BL-10938 /
SF-BL-11052 tags alongside it), as its first statements in both shipped
copies. calibration_bins' own screen in the canonical copy is the
SF-BL-11424 funnel screen (_reject_binned_panel), landed concurrently,
which raises the same named ValueErrors. Detection covers list/tuple/
nested, U/S/object ndarrays, bare object() leaves, and pandas
Series/Index/Categorical panels. Lawful panels are bit-identical.
Work-dir evidence: superforecaster_out/backlog/work/SF-BL-11425/
(repro_before.txt, repro_after.txt, test_bins_murphy_panels.py 92/92).

Run:  python3 -m pytest tests/test_calibration_bins_murphy_type_panels.py
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from superforecaster.calibration import (  # noqa: E402
    calibration_bins, calibration_table, murphy_decomposition)

pd = pytest.importorskip("pandas")


def _named(fn, name, word, *args, **kw):
    with pytest.raises(ValueError) as ei:
        fn(*args, **kw)
    msg = str(ei.value)
    assert msg.startswith(f"{name} must not contain"), msg
    assert f"in {name}" in msg and word in msg, msg
    return msg


def test_titled_and_kill_repros_raise():
    for fn in (calibration_bins, murphy_decomposition):
        _named(fn, "y", "string", ["1", "0"], [0.9, 0.1])
        _named(fn, "p", "string", [1, 0], ["0.9", "0.1"])
        _named(fn, "y", "boolean", [True, False], [0.9, 0.1])
        _named(fn, "p", "boolean", [1, 0], [True, False])
    # Kill forms explicitly: no bin table / brier comes back.
    msg = _named(murphy_decomposition, "p", "string", [1, 0], ["0.9", "0.1"])
    assert "SF-BL-11425" in msg
    msg = _named(murphy_decomposition, "p", "boolean", [1, 0],
                 pd.Series([True, False]))
    assert "SF-BL-11425" in msg


def test_string_and_bool_forms():
    string_ps = [["0.9", "0.1"], ("0.9", "0.1"), [0.9, "0.1"],
                 np.array(["0.9", "0.1"]),
                 np.array(["0.9", "0.1"], dtype=object),
                 pd.Series(["0.9", "0.1"]),
                 pd.Series(["0.9", "0.1"], dtype="string"),
                 pd.Categorical(["0.9", "0.1"])]
    bool_ps = [[True, False], np.array([True, False]),
               np.array([True, False], dtype=object),
               pd.Series([True, False]),
               pd.Series([True, False], dtype="boolean"),
               pd.Categorical([True, False])]
    bare = [object(), object()]
    for fn in (calibration_bins, murphy_decomposition):
        for p in string_ps:
            _named(fn, "p", "string", [1, 0], p)
        for p in bool_ps:
            _named(fn, "p", "boolean", [1, 0], p)
        _named(fn, "p", "string", [1, 0], bare)
        _named(fn, "y", "string", bare, [0.9, 0.1])
        _named(fn, "y", "string", pd.Series(["1", "0"]), [0.9, 0.1])
        _named(fn, "y", "boolean", pd.Series([True, False]), [0.9, 0.1])
        # quantile / n_bins variants still raise
        _named(fn, "p", "string", [1, 0], ["0.9", "0.1"],
               n_bins=3, strategy="quantile")


def test_alias_inherits():
    _named(calibration_table, "p", "string", [1, 0], ["0.9", "0.1"])


def test_lawful_panels_bit_identical():
    t = calibration_bins([1, 0], [0.9, 0.1])
    assert t["count"].tolist() == [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    assert t["mean_pred"][1] == 0.1 and t["mean_pred"][9] == 0.9
    d = murphy_decomposition([1, 0], [0.9, 0.1])
    assert d == {"reliability": 0.009999999999999998, "resolution": 0.25,
                 "uncertainty": 0.25, "brier_binned": 0.010000000000000009,
                 "brier": 0.009999999999999998,
                 "residual": -1.0408340855860843e-17, "base_rate": 0.5}
    # numeric object-dtype and numeric pandas panels stay lawful
    assert calibration_bins(np.array([1, 0], dtype=object),
                            np.array([0.9, 0.1], dtype=object))["count"].tolist() == t["count"].tolist()
    assert murphy_decomposition(pd.Series([1, 0]),
                                pd.Series([0.9, 0.1])) == d
    assert murphy_decomposition([1, 0, 1, 0], [1, 0, 1, 0])["brier"] == 0.0


def test_preexisting_errors_unchanged():
    for fn in (calibration_bins, murphy_decomposition):
        with pytest.raises(ValueError):
            fn([1, 0], [1.5, 0.1])            # range
        with pytest.raises(ValueError):
            fn([1, 0], [float("nan"), 0.1])   # finiteness
        with pytest.raises(ValueError):
            fn([1, 0, 1], [0.5])              # length mismatch
        with pytest.raises(ValueError):
            fn([1, 0], [0.9, 0.1], n_bins=0)  # n_bins
