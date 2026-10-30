"""
test_duplicate_agent_id.py — SF-BL-10849 verification tests.

Prereg (PREREG.md in the task work dir): aggregate_fleet_forecasts
dedupes by agent_id — first occurrence kept, duplicates
dropped-and-logged like the SF-BL-9439 filter — with unique-agent
panels byte-identical; kill if every panel is built from a set keyed
by agent_id upstream (CALLER_CENSUS.md: NOT met — the only production
caller passes a plain list literal, the API takes a List, and
SubAgentForecast validates nothing).

Pre-fix reproduction (repro.py, verified live):
    single [a1/famA 0.90, a2/famB 0.50, a3/famC 0.50]
        -> (0.7288411627050326, 0.18856180831641267)
    duplicated (a1 submitted twice)
        -> (0.8150443692010382, 0.2000)  # duplicate counted twice

Run:  python3 test_duplicate_agent_id.py   (exit 0 iff all pass)
"""
import contextlib
import io
import os
import sys
import warnings

_CANDIDATES = [
    os.path.expanduser("~/workspace/superforecaster/superforecaster"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "superforecaster"),
]
for _c in _CANDIDATES:
    if os.path.isfile(os.path.join(_c, "alpha_kelly.py")):
        sys.path.insert(0, _c)
        break

import alpha_kelly as ak  # noqa: E402

PASS = []
NAN = float("nan")

SINGLE = (0.7288411627050326, 0.18856180831641267)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _f(agent, fam, p, w=1.0, ind=True):
    return ak.SubAgentForecast(agent, fam, p, w, ind)


def _agg(members):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = ak.aggregate_fleet_forecasts(members)
    return result, [str(w.message) for w in caught]


def _single():
    return [_f("a1", "famA", 0.90), _f("a2", "famB", 0.50), _f("a3", "famC", 0.50)]


# ---- 1. TASK fixture -------------------------------------------------------

def test_task_fixture_duplicate_pools_to_single():
    dup = [_f("a1", "famA", 0.90), _f("a1", "famA", 0.90),
           _f("a2", "famB", 0.50), _f("a3", "famC", 0.50)]
    r, msgs = _agg(dup)
    assert tuple(r) == SINGLE, r
    assert len(msgs) == 1 and "SF-BL-10849" in msgs[0], msgs
    assert "a1=0.9" in msgs[0] and "pooling n=3 of 4" in msgs[0], msgs
    p = r.provenance
    assert (p.n_input, p.n_kept, p.n_dropped, p.n_duplicate_dropped) == (4, 3, 1, 1), p


def test_single_panel_clean_no_warning():
    r, msgs = _agg(_single())
    assert tuple(r) == SINGLE, r
    assert msgs == [], msgs
    p = r.provenance
    assert (p.n_input, p.n_kept, p.n_dropped, p.n_duplicate_dropped) == (3, 3, 0, 0), p


# ---- 2. First-occurrence semantics -----------------------------------------

def test_first_occurrence_kept_conflicting_duplicate():
    # Same agent_id, different prob/family: identity, not value, dedupes.
    conf = [_f("a1", "famA", 0.90), _f("a1", "famX", 0.10),
            _f("a2", "famB", 0.50), _f("a3", "famC", 0.50)]
    r, msgs = _agg(conf)
    assert tuple(r) == SINGLE, r  # the 0.90/famA first copy survived
    assert len(msgs) == 1 and "a1=0.1" in msgs[0], msgs


def test_first_occurrence_reversed_order():
    rev = [_f("a1", "famX", 0.10), _f("a1", "famA", 0.90),
           _f("a2", "famB", 0.50), _f("a3", "famC", 0.50)]
    clean_first = [_f("a1", "famX", 0.10), _f("a2", "famB", 0.50), _f("a3", "famC", 0.50)]
    r, _ = _agg(rev)
    rc, _ = _agg(clean_first)
    assert tuple(r) == tuple(rc), (r, rc)
    assert tuple(r) != SINGLE, r  # now the 0.10 copy is the survivor


def test_triple_submission_and_multiple_ids():
    panel = [_f("a1", "famA", 0.90), _f("a1", "famA", 0.90), _f("a1", "famA", 0.90),
             _f("a2", "famB", 0.50), _f("a2", "famB", 0.50), _f("a3", "famC", 0.50)]
    r, msgs = _agg(panel)
    assert tuple(r) == SINGLE, r
    p = r.provenance
    assert (p.n_input, p.n_kept, p.n_duplicate_dropped) == (6, 3, 3), p
    assert len(msgs) == 1 and "3 duplicate" in msgs[0], msgs
    assert "pooling n=3 of 6" in msgs[0], msgs


def test_all_copies_of_one_agent():
    r, msgs = _agg([_f("a1", "famA", 0.90)] * 3)
    assert tuple(r) == (0.9147453096907046, 0.0), r
    assert r.provenance.n_kept == 1 and r.provenance.n_duplicate_dropped == 2, r.provenance
    assert len(msgs) == 1, msgs


def test_duplicate_position_irrelevant():
    # Duplicate at the end pools identically to duplicate in the middle.
    tail = [_f("a1", "famA", 0.90), _f("a2", "famB", 0.50),
            _f("a3", "famC", 0.50), _f("a1", "famA", 0.90)]
    r, _ = _agg(tail)
    assert tuple(r) == SINGLE, r


# ---- 3. Other pooling branches ----------------------------------------------

def test_median_branch_duplicate():
    clean = [_f("a1", "fam1", 0.70, 1.0, False), _f("a2", "fam1", 0.60),
             _f("a3", "fam1", 0.50)]
    dup = clean + [_f("a1", "fam1", 0.70, 1.0, False)]
    rc, _ = _agg(clean)
    assert tuple(rc) == (0.6077590814688054, 0.08164965809277258), rc
    r, msgs = _agg(dup)
    assert tuple(r) == tuple(rc), (r, rc)
    assert len(msgs) == 1 and "SF-BL-10849" in msgs[0], msgs


def test_trimmed_branch_duplicate():
    probs = [0.40, 0.55, 0.60, 0.65, 0.70, 0.90]
    clean = [_f(f"a{i}", "fam1", p, 1.0, False) for i, p in enumerate(probs, 1)]
    dup = clean[:3] + [clean[2]] + clean[3:]  # a3 (0.60) submitted twice
    rc, _ = _agg(clean)
    assert tuple(rc) == (0.6345279830987509, 0.15184055965240498), rc
    r, _ = _agg(dup)
    assert tuple(r) == tuple(rc), (r, rc)
    assert r.provenance.n_duplicate_dropped == 1, r.provenance


def test_weighted_duplicate_not_double_weighted():
    # A duplicate must not double a1's weight: deduped panel == single
    # panel with a1's true (single) weight.
    weighted_single = [_f("a1", "famA", 0.90, 2.0), _f("a2", "famB", 0.50),
                       _f("a3", "famC", 0.50)]
    dup = [_f("a1", "famA", 0.90, 1.0), _f("a1", "famA", 0.90, 1.0),
           _f("a2", "famB", 0.50), _f("a3", "famC", 0.50)]
    r, _ = _agg(dup)
    assert tuple(r) == SINGLE, r  # single weight 1.0 each, not the w=2 pool
    rw, _ = _agg(weighted_single)
    assert tuple(rw) == (0.8150443692010382, 0.18856180831641267), rw


# ---- 4. Interaction with the other drop filters ------------------------------

def test_duplicate_of_nan_first_copy_does_not_resurrect():
    # Dedupe is on raw order: a1's first copy is NaN (kept for identity,
    # then dropped by SF-BL-9439); the valid second copy is the
    # duplicate drop — a1 contributes nothing.
    panel = [_f("a1", "famA", NAN), _f("a1", "famA", 0.90),
             _f("a2", "famB", 0.50), _f("a3", "famC", 0.50)]
    r, msgs = _agg(panel)
    clean_pair = [_f("a2", "famB", 0.50), _f("a3", "famC", 0.50)]
    rc, _ = _agg(clean_pair)
    assert tuple(r) == tuple(rc), (r, rc)
    p = r.provenance
    assert (p.n_input, p.n_kept, p.n_duplicate_dropped, p.n_prob_dropped) == (4, 2, 1, 1), p
    assert any("SF-BL-10849" in m for m in msgs), msgs
    assert any("SF-BL-9439" in m for m in msgs), msgs


def test_duplicate_and_invalid_other_member():
    panel = [_f("a1", "famA", 0.90), _f("a1", "famA", 0.90),
             _f("a2", "famB", NAN), _f("a3", "famC", 0.50), _f("a4", "famD", 0.60)]
    r, msgs = _agg(panel)
    clean = [_f("a1", "famA", 0.90), _f("a3", "famC", 0.50), _f("a4", "famD", 0.60)]
    rc, _ = _agg(clean)
    assert tuple(r) == tuple(rc), (r, rc)
    p = r.provenance
    assert (p.n_input, p.n_kept, p.n_dropped, p.n_duplicate_dropped,
            p.n_prob_dropped) == (5, 3, 2, 1, 1), p
    assert any("SF-BL-10849" in m and "pooling n=3 of 5" in m for m in msgs), msgs


def test_distinct_family_of_duplicate_counted_in_input():
    # The dropped duplicate carried a family no kept member has: the
    # input-side distinct count records the original panel.
    panel = [_f("a1", "famA", 0.90), _f("a1", "famUnique", 0.10),
             _f("a2", "famB", 0.50), _f("a3", "famC", 0.50)]
    r, _ = _agg(panel)
    assert r.provenance.distinct_families_input == 4, r.provenance
    assert r.provenance.distinct_families_kept == 3, r.provenance


def test_all_members_invalid_with_duplicate_still_raises():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ak.aggregate_fleet_forecasts(
                [_f("a1", "famA", NAN), _f("a1", "famA", NAN)])
    except ValueError as e:
        assert "SF-BL-10849" in str(e), e
        return
    raise AssertionError("all-invalid duplicated panel did not raise")


def test_empty_still_raises():
    try:
        ak.aggregate_fleet_forecasts([])
    except ValueError:
        return
    raise AssertionError("empty panel did not raise")


# ---- 5. Unique panels byte-identical + benchmark -----------------------------

def test_unique_panels_byte_identical():
    cases = [
        ([_f("a1", "fam1", 0.70), _f("a2", "fam2", 0.60), _f("a3", "fam3", 0.65)],
         (0.6989517116705906, 0.04082482904638629)),
        (_single(), SINGLE),
    ]
    for members, expected in cases:
        r, msgs = _agg(members)
        assert tuple(r) == expected, (r, expected)
        assert msgs == [], msgs
        assert r.provenance.n_duplicate_dropped == 0, r.provenance


def test_benchmark_nstar_unchanged():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    # Pinned to the pristine snapshot's values this run (-0.1611 /
    # -0.1499; identical pre/post patch — benchmark panels have unique
    # agent ids, so dedupe never fires there).
    assert '"north_star_score": -0.1611' in out, out[:400]
    assert '"north_star_score": -0.1499' in out, out[:400]


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
