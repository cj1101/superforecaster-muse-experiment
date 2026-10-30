"""SF-BL-11087: trimmed_mean_pool must type-validate its trim fraction.

Pre-fix, trim flowed straight into int(np.floor(trim * m)): trim=True
silently pooled the titled 8-member panel to the plain mean 0.4625
(vs 0.45 at the lawful trim=0.25), trim='0.1' died with a bare ufunc
TypeError, and trim=NaN with a bare 'cannot convert float NaN to
integer' ValueError — none naming trim. The guard must raise a
ValueError naming trim on any bool, string or non-finite trim, with
finite numeric trim in [0, 0.5) bit-identical.
"""
import numpy as np
import pytest

from superforecaster.calibration import trimmed_mean_pool

PANEL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]


@pytest.mark.parametrize("bad", [
    True, False, np.bool_(True),
    "0.1", "0", b"0.1", np.str_("0.1"),
    float("nan"), float("inf"), float("-inf"), np.float64("nan"),
    None, [0.1], (0.1,), np.array([0.1, 0.2]), object(),
    -0.1, -1.0, 0.5, 0.6, 1.0, 2,
])
def test_invalid_trim_raises_named_value_error(bad):
    with pytest.raises(ValueError, match="trim"):
        trimmed_mean_pool(PANEL, trim=bad)


def test_kill_repros():
    # trim=True must NOT return the plain mean 0.4625
    with pytest.raises(ValueError, match="SF-BL-11087"):
        trimmed_mean_pool(PANEL, trim=True)
    # trim='0.1' must NOT raise a bare TypeError
    with pytest.raises(ValueError, match="trim"):
        trimmed_mean_pool(PANEL, trim="0.1")
    # trim=NaN must NOT raise the unnamed conversion ValueError
    with pytest.raises(ValueError, match="trim"):
        trimmed_mean_pool(PANEL, trim=float("nan"))


def test_lawful_trim_unchanged():
    assert float(trimmed_mean_pool(PANEL, trim=0.25)[0]) == pytest.approx(0.45)
    assert float(trimmed_mean_pool(PANEL, trim=0.1)[0]) == pytest.approx(0.4625)
    assert float(trimmed_mean_pool(PANEL, trim=0)[0]) == pytest.approx(0.4625)
    assert float(trimmed_mean_pool(PANEL)[0]) == pytest.approx(0.4625)
    assert float(trimmed_mean_pool(PANEL, trim=np.float64(0.25))[0]) == pytest.approx(0.45)
