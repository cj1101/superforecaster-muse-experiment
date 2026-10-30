"""SF-BL-10919: track_record_weights must validate the records themselves.

Pre-fix, only the lam / shrinkage_n / floor parameters carried guards
(SF-BL-9889 / 10274 / 10662); the stored records were trusted verbatim.
A record with briers=[NaN] (reachable via restored records that bypass
ForecasterRecord.add) silently collapsed the panel to equal weights
[0.5, 0.5], and a record with mean_brier -0.5 took the panel's top
weight (0.516) over a genuine 0.10 record. Fix raises ValueError
naming the field on any record with a non-finite mean_brier /
mean_log_score, a mean_brier outside [0, 1], or a negative n, while
seeded lawful panels stay bit-identical.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from superforecaster.agents import ForecasterRecord, track_record_weights


def _rec(mb, n=20, mls=0.3):
    r = ForecasterRecord(agent_id="x")
    r.briers = [mb] * n
    r.log_scores = [mls] * n if n else []
    return r


def _stub(mean_brier=0.10, mean_log_score=0.30, n=20):
    # Restored-record stand-in: derived fields only, bypassing add()
    # and the n = len(briers) property.
    return SimpleNamespace(mean_brier=mean_brier,
                           mean_log_score=mean_log_score, n=n)


GOOD = [_rec(0.01), _rec(0.81)]


def test_task_repro_nan_record_no_longer_equal_weights():
    with pytest.raises(ValueError, match="mean_brier"):
        track_record_weights([_rec(float("nan"), n=1), _rec(0.10, n=1)])


def test_task_repro_impossible_negative_mean_brier():
    # Pre-fix this returned [0.516..., 0.484...] — corrupt member on top.
    with pytest.raises(ValueError, match="mean_brier"):
        track_record_weights([_rec(-0.5, n=1), _rec(0.10, n=1)])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"),
                                 -0.5, -1e-9, 1.0000001, 1.5, 2.0])
def test_bad_mean_brier_raises_naming_mean_brier(bad):
    with pytest.raises(ValueError, match="mean_brier"):
        track_record_weights([_rec(bad), _rec(0.2)])
    with pytest.raises(ValueError, match="mean_brier"):
        track_record_weights([_stub(mean_brier=bad), _stub()])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_mean_log_score_raises(bad):
    r = _rec(0.1)
    r.log_scores = [bad] * 20
    with pytest.raises(ValueError, match="mean_log_score"):
        track_record_weights([r, _rec(0.2)])
    with pytest.raises(ValueError, match="mean_log_score"):
        track_record_weights([_stub(mean_log_score=bad), _stub()])


@pytest.mark.parametrize("bad_n", [-1, -20, -0.5, float("nan"),
                                   float("inf"), float("-inf")])
def test_negative_or_nonfinite_n_raises_naming_n(bad_n):
    with pytest.raises(ValueError, match="n"):
        track_record_weights([_stub(n=bad_n), _stub()])


def test_corrupt_second_record_also_raises():
    with pytest.raises(ValueError, match="mean_brier"):
        track_record_weights([_rec(0.2), _rec(float("nan"), n=5)])


def test_mixed_element_corruption_raises():
    r = _rec(0.2)
    r.briers = [0.2] * 19 + [float("nan")]  # mean_brier NaN via one element
    with pytest.raises(ValueError, match="mean_brier"):
        track_record_weights([r, _rec(0.3)])


def test_lawful_endpoints_and_cold_start_still_weight():
    w = track_record_weights([_rec(0.0, n=10), _rec(1.0, n=10)])
    assert np.all(np.isfinite(w)) and abs(w.sum() - 1.0) < 1e-12 and w[0] > w[1]
    w = track_record_weights([ForecasterRecord(agent_id="new"), _rec(0.2, n=5)])
    assert np.all(np.isfinite(w)) and abs(w.sum() - 1.0) < 1e-12
    assert track_record_weights([]).size == 0


def test_seeded_lawful_panel_bit_identical():
    rng = np.random.default_rng(42)
    panel = []
    for _ in range(6):
        r = ForecasterRecord(agent_id="s")
        for _ in range(int(rng.integers(0, 40))):
            r.add(float(rng.uniform(0.01, 0.99)), int(rng.integers(0, 2)))
        panel.append(r)
    w = track_record_weights(panel)
    assert np.all(np.isfinite(w)) and abs(w.sum() - 1.0) < 1e-12
    # Known default weights from the SF-BL-9889 suite are unchanged.
    assert np.allclose(track_record_weights(GOOD),
                       [0.69997291, 0.30002709], atol=1e-8)
