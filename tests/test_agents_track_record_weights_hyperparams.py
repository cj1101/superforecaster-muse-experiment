"""SF-BL-9889: track_record_weights lam / shrinkage_n / floor-K validation.

Pre-fix, on two 20-record forecasters (mean Briers 0.01 vs 0.81):
lam=-12 returned [0.30, 0.70] (panel inverted), lam=NaN returned
[0.5, 0.5] (track record silently erased), shrinkage_n=-5 returned
[0.983, 0.017] (alpha > 1), shrinkage_n=-20 raised a bare
ZeroDivisionError, and floor >= 1/K was silently accepted (floor=0.99
at K=2 returned [0.5, 0.5]). Fix raises ValueError naming the
offending parameter: lam must be finite and > 0, shrinkage_n finite
and > 0, floor finite and in [0, 1/K) ([0, 1) on the empty panel).
Valid-hyperparameter weights are bit-identical to pre-fix.
"""
import numpy as np
import pytest
from superforecaster.agents import ForecasterRecord, track_record_weights


def _rec(mb, n=20):
    r = ForecasterRecord(agent_id="x")
    r.briers = [mb] * n
    return r


RECS2 = [_rec(0.01), _rec(0.81)]
RECS3 = [_rec(0.01), _rec(0.30), _rec(0.81)]


@pytest.mark.parametrize("bad", [-12.0, -0.001, 0.0, float("nan"),
                                 float("inf"), float("-inf"),
                                 "12", None, True])
def test_bad_lam_raises_naming_lam(bad):
    with pytest.raises(ValueError, match="lam"):
        track_record_weights(RECS2, lam=bad)


@pytest.mark.parametrize("bad", [-5.0, -20.0, 0.0, float("nan"),
                                 float("inf"), float("-inf"),
                                 "30", None, True])
def test_bad_shrinkage_n_raises_naming_shrinkage_n(bad):
    # shrinkage_n=-20 with n=20 records divided by zero pre-fix; the
    # fix must surface ValueError, never ZeroDivisionError.
    with pytest.raises(ValueError, match="shrinkage_n"):
        track_record_weights(RECS2, shrinkage_n=bad)


def test_bad_hyperparams_raise_even_on_empty():
    with pytest.raises(ValueError, match="lam"):
        track_record_weights([], lam=-12.0)
    with pytest.raises(ValueError, match="shrinkage_n"):
        track_record_weights([], shrinkage_n=-5.0)


@pytest.mark.parametrize("recs,bad", [
    (RECS2, 0.5),            # == 1/K at K=2 (endpoint excluded)
    (RECS2, 0.6),
    (RECS2, 0.99),
    (RECS3, 1.0 / 3.0),      # == 1/K at K=3
    (RECS3, 0.34),
])
def test_floor_at_or_above_inverse_k_raises(recs, bad):
    with pytest.raises(ValueError, match="floor"):
        track_record_weights(recs, floor=bad)


def test_task_repro_cases_now_raise():
    # TASK.md live repros: lam=-12 -> [0.30, 0.70]; lam=NaN -> equal
    # weights; shrinkage_n=-5 -> [0.983, 0.017]. All must raise now.
    with pytest.raises(ValueError, match="lam"):
        track_record_weights(RECS2, lam=-12.0)
    with pytest.raises(ValueError, match="lam"):
        track_record_weights(RECS2, lam=float("nan"))
    with pytest.raises(ValueError, match="shrinkage_n"):
        track_record_weights(RECS2, shrinkage_n=-5.0)


def test_defaults_unchanged():
    w = track_record_weights(RECS2)
    assert np.allclose(w, [0.69997291, 0.30002709], atol=1e-8)
    assert w[0] > w[1]  # the Brier-0.01 forecaster keeps the larger weight


@pytest.mark.parametrize("kw", [
    {"lam": 1e-9}, {"lam": 500.0},
    {"shrinkage_n": 1e-9}, {"shrinkage_n": 1e6},
    {"floor": 0.0}, {"floor": 0.49},
])
def test_valid_boundaries_return_normalised_weights(kw):
    w = track_record_weights(RECS2, **kw)
    assert np.all(np.isfinite(w))
    assert abs(w.sum() - 1.0) < 1e-12


def test_valid_k_dependent_floors():
    assert np.all(np.isfinite(track_record_weights(RECS3, floor=0.33)))
    w = track_record_weights([_rec(0.2)], floor=0.9)  # 0.9 < 1/1 at K=1
    assert np.allclose(w, [1.0])
    assert track_record_weights([]).size == 0
