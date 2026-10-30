"""SF-BL-10310 — non-finite group_rank must not manufacture a threshold.

Pre-fix, parse_threshold accepted any float group_rank via isinstance
without a finiteness check, so parse_threshold({'group_rank': NaN})
returned NaN; and infer_group_kind treated NaN as a distinct parseable
threshold (all-not-None plus set-distinctness, and NaN != NaN), so two
binary subquestions with NaN ranks were certified 'threshold_ladder',
whose sort order in validate_group_consistency is undefined.

Pinned contract (task done/kill criterion):
  * parse_threshold returns None for non-finite group_rank (NaN, +/-inf);
  * infer_group_kind returns 'independent' on the NaN-rank control panel;
  * infer_group_kind still returns 'threshold_ladder' for finite
    distinct ranks;
  * kill if any non-finite rank yields a non-None threshold or a ladder.

Plain asserts; runnable with pytest or standalone.
"""
import math
import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "metaculus_bot",
    ),
)

from group_module import (  # noqa: E402
    infer_group_kind,
    parse_threshold,
    validate_group_consistency,
)

NAN = float("nan")
INF = float("inf")


def _panel(ranks):
    return [
        {"id": i + 1, "type": "binary", "group_rank": r}
        for i, r in enumerate(ranks)
    ]


def test_parse_threshold_rejects_nonfinite_group_rank():
    assert parse_threshold({"group_rank": NAN}) is None
    assert parse_threshold({"group_rank": INF}) is None
    assert parse_threshold({"group_rank": -INF}) is None


def test_parse_threshold_finite_controls_unchanged():
    assert parse_threshold({"group_rank": 2.5}) == 2.5
    assert parse_threshold({"group_rank": 0}) == 0.0
    assert parse_threshold({"group_rank": -3}) == -3.0
    assert parse_threshold({"group_rank": "2"}) is None
    assert parse_threshold({}) is None
    assert parse_threshold({"label": "10"}) == 10.0


def test_parse_threshold_rejects_overflowing_numeric_label():
    # A digit string too large for a float parses to inf; it is not a
    # threshold either, and a finite group_rank fallback still applies.
    assert parse_threshold({"label": "9" * 400}) is None
    assert parse_threshold({"label": "9" * 400, "group_rank": 7}) == 7.0


def test_infer_group_kind_nan_panel_is_independent():
    assert infer_group_kind(_panel([NAN, NAN])) == "independent"
    assert infer_group_kind(_panel([NAN, 1.0])) == "independent"


def test_infer_group_kind_nonfinite_panels_are_independent():
    assert infer_group_kind(_panel([INF, 1.0])) == "independent"
    assert infer_group_kind(_panel([INF, -INF])) == "independent"
    assert infer_group_kind(_panel([-INF, 0.0])) == "independent"


def test_infer_group_kind_finite_controls_unchanged():
    assert infer_group_kind(_panel([1.0, 2.0])) == "threshold_ladder"
    assert infer_group_kind(_panel([1.0, 1.0])) == "independent"
    label_panel = [
        {"id": 1, "type": "binary", "label": "10"},
        {"id": 2, "type": "binary", "label": "20"},
    ]
    assert infer_group_kind(label_panel) == "threshold_ladder"


def test_validate_group_consistency_nan_ladder_flagged_not_sorted():
    subs = _panel([NAN, 1.0])
    payloads = [
        {"question": 1, "probability_yes": 0.5},
        {"question": 2, "probability_yes": 0.4},
    ]
    problems = validate_group_consistency(
        subs, payloads, kind="threshold_ladder"
    )
    assert problems and "no parseable threshold" in problems[0], problems
    ok_subs = _panel([1.0, 2.0])
    assert validate_group_consistency(ok_subs, payloads) == []


def test_every_returned_threshold_is_finite():
    for rank in (NAN, INF, -INF, 0.0, 1.5, -2.5):
        got = parse_threshold({"group_rank": rank})
        assert got is None or math.isfinite(got), (rank, got)


if __name__ == "__main__":
    for _name, _fn in sorted(list(globals().items())):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"PASS {_name}")
    print("ALL PASSED")
