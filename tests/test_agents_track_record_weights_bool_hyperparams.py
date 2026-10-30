"""SF-BL-11056: track_record_weights bool-hyperparameter validation.

TASK panel: two 20-record forecasters, mean Briers 0.01 vs 0.81.
A bool hyperparameter must raise ValueError naming that parameter,
in every form float() would erase: scalar bool, np.bool_, a
bool-dtype array (0-d or 1-element), and an object array holding a
bool. Scalar bools were already rejected by SF-BL-9889/10274; the
array forms slipped pre-fix: lam=np.array(True) returned the lam=1
weights [0.57598979, 0.42401021], shrinkage_n=np.array([True]) the
shrinkage_n=1 weights [0.97612598, 0.02387402], and
floor=np.array(False) the floor=0 (default) weights.
Numeric hyperparameters — including numeric NumPy scalars and
0-d / 1-element numeric arrays, lawfully accepted pre-fix via
float() — must stay bit-identical.
"""
import numpy as np
import pytest

from superforecaster.agents import ForecasterRecord, track_record_weights


def _rec(mb, n=20):
    r = ForecasterRecord(agent_id="x")
    r.briers = [mb] * n
    r.log_scores = [0.5] * n
    return r


RECS2 = [_rec(0.01), _rec(0.81)]

BOOLS = [
    True, False,
    np.bool_(True), np.bool_(False),
    np.array(True), np.array(False),
    np.array([True]), np.array([False]),
    np.array(True, dtype=object), np.array(False, dtype=object),
    np.array([True], dtype=object), np.array([False], dtype=object),
]


@pytest.mark.parametrize("bad", BOOLS)
def test_bool_lam_raises_naming_lam(bad):
    with pytest.raises(ValueError, match="lam"):
        track_record_weights(RECS2, lam=bad)


@pytest.mark.parametrize("bad", BOOLS)
def test_bool_shrinkage_n_raises_naming_shrinkage_n(bad):
    with pytest.raises(ValueError, match="shrinkage_n"):
        track_record_weights(RECS2, shrinkage_n=bad)


@pytest.mark.parametrize("bad", BOOLS)
def test_bool_floor_raises_naming_floor(bad):
    with pytest.raises(ValueError, match="floor"):
        track_record_weights(RECS2, floor=bad)


@pytest.mark.parametrize("param", ["lam", "shrinkage_n", "floor"])
def test_bool_raises_even_on_empty_panel(param):
    with pytest.raises(ValueError, match=param):
        track_record_weights([], **{param: True})


def test_kill_repros():
    # Kill criterion: none of these may return a weight array.
    for kw in ({"lam": True}, {"shrinkage_n": True}, {"floor": True}):
        with pytest.raises(ValueError):
            track_record_weights(RECS2, **kw)


def test_numeric_bit_identical():
    base = track_record_weights(RECS2)
    assert np.array_equal(base, track_record_weights(RECS2, lam=12.0))
    assert np.array_equal(base, track_record_weights(RECS2, lam=12))
    assert np.array_equal(base, track_record_weights(RECS2, lam=np.float64(12.0)))
    assert np.array_equal(base, track_record_weights(RECS2, lam=np.array(12.0)))
    assert np.array_equal(base, track_record_weights(RECS2, shrinkage_n=30.0))
    assert np.array_equal(base, track_record_weights(RECS2, shrinkage_n=np.array(30.0)))
    assert np.array_equal(base, track_record_weights(RECS2, floor=0.02))
    assert np.array_equal(base, track_record_weights(RECS2, floor=np.array(0.02)))
    assert np.allclose(base, [0.69997291, 0.30002709], atol=1e-8)


def test_numeric_values_still_work():
    assert np.allclose(track_record_weights(RECS2, lam=1.0),
                       [0.57598979, 0.42401021], atol=1e-8)
    assert np.allclose(track_record_weights(RECS2, shrinkage_n=1.0),
                       [0.97612598, 0.02387402], atol=1e-8)
    assert np.allclose(track_record_weights(RECS2, floor=0.0),
                       [0.69997291, 0.30002709], atol=1e-8)
