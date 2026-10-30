"""SF-BL-10662: track_record_weights shrinkage_n validation (canonical agents.py).

TASK repros: shrinkage_n=-1 with a one-forecast record (n=1) raised a
bare ZeroDivisionError pre-fix, and a negative shrinkage_n puts
alpha = n/(n+shrinkage_n) outside [0, 1] (negative when
shrinkage_n < -n, >1 when -n < shrinkage_n < 0), inverting the trust
ordering. Fix passes iff shrinkage_n<=0 or non-finite raises a named
ValueError before any weight is computed and default
(lam=12, shrinkage_n=30, floor=0.02) weights on a seeded panel are
bit-identical. The validator itself landed under SF-BL-9889, whose
docstring in agents.py explicitly satisfies this prereg too; this
suite pins the 10662-specific cases (n=1/-1, n=50/-50) that the
9889 suite (n=20/-20) does not cover.
"""
import math

import numpy as np
import pytest

from superforecaster.agents import ForecasterRecord, track_record_weights


def _rec(mb, n):
    r = ForecasterRecord(agent_id="x")
    r.briers = [mb] * n
    return r


def test_task_repro_shrinkage_minus1_n1_raises_valueerror_not_zerodivision():
    with pytest.raises(ValueError, match="shrinkage_n"):
        track_record_weights([_rec(0.25, 1)], shrinkage_n=-1)


def test_task_repro_shrinkage_minus50_n50_raises():
    recs = [_rec(0.01, 50), _rec(0.81, 50)]
    with pytest.raises(ValueError, match="shrinkage_n"):
        track_record_weights(recs, shrinkage_n=-50)


@pytest.mark.parametrize("bad", [0, 0.0, -0.5, -1, -30, -100,
                                 float("nan"), float("inf"), float("-inf"),
                                 "30", None, True])
def test_nonpositive_or_nonfinite_shrinkage_raises_naming_shrinkage_n(bad):
    with pytest.raises(ValueError, match="shrinkage_n"):
        track_record_weights([_rec(0.2, 5)], shrinkage_n=bad)


def test_negative_alpha_domain_raises():
    # shrinkage_n < -n gives alpha < 0 pre-fix (e.g. -60 with n=50 -> alpha -5,
    # weights inverted to [0.0066, 0.9934] for Briers 0.01 vs 0.81)
    with pytest.raises(ValueError, match="shrinkage_n"):
        track_record_weights([_rec(0.01, 50), _rec(0.81, 50)], shrinkage_n=-60)


def test_invalid_shrinkage_raises_before_any_weight_even_on_empty():
    with pytest.raises(ValueError, match="shrinkage_n"):
        track_record_weights([], shrinkage_n=-1)


def _prefix_weights(records, lam=12.0, shrinkage_n=30.0, floor=0.02):
    k = len(records)
    if k == 0:
        return np.array([])
    equal = 1.0 / k
    raw = np.array([math.exp(-lam * r.mean_brier) for r in records], dtype=float)
    raw = raw / raw.sum() if raw.sum() > 0 else np.full(k, equal)
    alphas = np.array([r.n / (r.n + shrinkage_n) if r.n > 0 else 0.0 for r in records])
    w = alphas * raw + (1.0 - alphas) * equal
    w = np.maximum(w, floor)
    return w / w.sum()


def test_seeded_panel_defaults_bit_identical():
    rng = np.random.default_rng(42)
    seeded = []
    for i in range(6):
        r = ForecasterRecord(agent_id=f"seeded-{i}")
        for _ in range(int(rng.integers(0, 60))):
            r.add(float(rng.uniform(0.05, 0.95)), int(rng.integers(0, 2)))
        seeded.append(r)
    w = track_record_weights(seeded)  # defaults lam=12, shrinkage_n=30, floor=0.02
    assert np.array_equal(w, _prefix_weights(seeded))
    assert w.tobytes() == _prefix_weights(seeded).tobytes()
