"""SF-BL-10274: track_record_weights floor validation.

Pre-fix, floor=NaN and floor=inf returned [nan, nan]; floor=-inf,
negative floors and floor>=1 were silently accepted. Fix raises
ValueError naming floor unless finite and in [0, 1), with finite-floor
weights bit-identical.

SUPERSEDED IN PART by SF-BL-9889 (2026-10-23): the valid range is now
[0, 1/K) for a panel of K records ([0, 1) only on the empty panel), so
at K=2 floors of 0.5 and above — previously asserted lawful below —
now raise. The non-finite / negative / type screens below are
unchanged; only the lawful-range expectations moved.
"""
import math
import numpy as np
import pytest
from superforecaster.agents import ForecasterRecord, track_record_weights


def _rec(mb, n=20):
    r = ForecasterRecord(agent_id="x")
    r.briers = [mb] * n
    return r


RECS = [_rec(0.01), _rec(0.81)]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"),
                                 -0.1, -1.0, 1.0, 1.5, "0.02", None, True])
def test_bad_floor_raises_naming_floor(bad):
    with pytest.raises(ValueError, match="floor"):
        track_record_weights(RECS, floor=bad)


def test_bad_floor_raises_even_on_empty():
    with pytest.raises(ValueError, match="floor"):
        track_record_weights([], floor=float("nan"))


@pytest.mark.parametrize("good", [0.0, 0.001, 0.02, 0.49])
def test_good_floor_returns_finite_normalised(good):
    # RECS has K=2 members, so lawful floors are [0, 1/K) = [0, 0.5)
    # under SF-BL-9889 (was [0, 1) under SF-BL-10274 alone).
    w = track_record_weights(RECS, floor=good)
    assert np.all(np.isfinite(w))
    assert abs(w.sum() - 1.0) < 1e-12


@pytest.mark.parametrize("bad", [0.5, 0.99, 0.999])
def test_floor_at_or_above_inverse_k_now_raises(bad):
    # SF-BL-9889: floor >= 1/K at K=2 raises (0.5 == 1/K is excluded).
    with pytest.raises(ValueError, match="floor"):
        track_record_weights(RECS, floor=bad)


def test_default_floor_unchanged():
    # Pre-fix value for the TASK repro configuration at default floor=0.02
    w = track_record_weights(RECS)
    assert [float(x).hex() for x in w] == [float(x).hex() for x in
           np.array([0.69997291, 0.30002709], dtype=float)] or np.allclose(
               w, [0.69997291, 0.30002709], atol=1e-8)


def test_floor_zero_and_high_finite():
    assert np.all(np.isfinite(track_record_weights(RECS, floor=0.0)))
    # floor=0.999 is lawful only where 1/K > 0.999, i.e. K=1
    # (SF-BL-9889); at K=2 it raises (see test above).
    w = track_record_weights([_rec(0.01)], floor=0.999)
    assert np.allclose(w, [1.0])
