"""
self_test.py — offline self-test for the Metaculus bot scaffold.

No network, no token. Feeds synthetic question payloads shaped like real
Metaculus API question objects (as returned inside GET /api/posts/ results)
through the same code paths bot.py uses, and asserts the submission
payloads are valid:

  - binary:           probability in [0.01, 0.99], other fields null
  - multiple_choice:  per-option probabilities, keys == options, sum == 1
  - numeric:          continuous_cdf of 201 points, values in [0, 1],
                      non-decreasing, no step > 0.19
  - numeric (log):    same, with scaling.zero_point set (log-scaled axis)
  - discrete:         continuous_cdf of inbound_outcome_count + 1 points
  - blind view:       aggregations / community prediction stripped

Run:  python3 self_test.py     (exit 0 = pass)
"""

from __future__ import annotations

import sys

from bot import fleet_forecast, judge_and_pool
from group_module import (
    infer_group_kind,
    is_group_post,
    unpack_group_post,
    validate_group_consistency,
)
from metaculus_client import (
    blind_question_view,
    build_comment_payload,
    build_forecast_payload,
    build_group_forecast_payloads,
    cdf_size_for_question,
    validate_forecast_payload,
)

# --- Synthetic payloads, shaped like the real API's question objects ------

BINARY_Q = {
    "id": 39001,
    "title": "Will the synthetic binary question resolve Yes?",
    "type": "binary",
    "description": "Synthetic background text.",
    "resolution_criteria": "Resolves Yes if the synthetic event occurs.",
    "fine_print": "",
    "status": "open",
    "scheduled_close_time": "2026-12-01T00:00:00Z",
    "scheduled_resolve_time": "2026-12-15T00:00:00Z",
    "options": None,
    "scaling": None,
    "open_upper_bound": False,
    "open_lower_bound": False,
    "unit": None,
    "aggregations": {"recency_weighted": {"latest": {"means": [0.63]}}},
    "my_forecasts": None,
}

MC_Q = {
    "id": 39002,
    "title": "Which synthetic option will win?",
    "type": "multiple_choice",
    "description": "Synthetic background text.",
    "resolution_criteria": "Resolves to the winning synthetic option.",
    "fine_print": "",
    "status": "open",
    "scheduled_close_time": "2026-12-01T00:00:00Z",
    "options": ["Alpha", "Beta", "Gamma", "Delta"],
    "scaling": None,
    "open_upper_bound": False,
    "open_lower_bound": False,
    "unit": None,
    "aggregations": {"recency_weighted": {"latest": {"means": [0.4, 0.3, 0.2, 0.1]}}},
    "my_forecasts": None,
}

NUMERIC_Q = {
    "id": 39003,
    "title": "How many synthetic widgets will be counted?",
    "type": "numeric",
    "description": "Synthetic background text.",
    "resolution_criteria": "Resolves to the official synthetic count.",
    "fine_print": "",
    "status": "open",
    "scheduled_close_time": "2026-12-01T00:00:00Z",
    "options": None,
    "scaling": {
        "range_min": 0.0,
        "range_max": 1000.0,
        "zero_point": None,
        "inbound_outcome_count": 200,
    },
    "open_upper_bound": True,
    "open_lower_bound": False,
    "unit": "widgets",
    "aggregations": {"recency_weighted": {"latest": {"means": [512.0]}}},
    "my_forecasts": None,
}

NUMERIC_LOG_Q = dict(
    NUMERIC_Q,
    id=39004,
    title="What will the synthetic log-scaled quantity be?",
    scaling={
        "range_min": 10.0,
        "range_max": 100000.0,
        "zero_point": 0.0,  # log-scaled axis
        "inbound_outcome_count": 200,
    },
)

DISCRETE_Q = dict(
    NUMERIC_Q,
    id=39005,
    title="How many synthetic dice will show sixes?",
    type="discrete",
    scaling={
        "range_min": 0.0,
        "range_max": 10.0,
        "zero_point": None,
        "inbound_outcome_count": 10,
    },
)


def check(question: dict) -> dict:
    view = blind_question_view(question)
    forecast = judge_and_pool(view, fleet_forecast(view))
    payload = build_forecast_payload(question, forecast)
    problems = validate_forecast_payload(question, payload)
    assert not problems, f"q{question['id']}: {problems}"
    assert payload["question"] == question["id"]
    assert payload["source"] == "api"
    return payload


def _binary_sub(qid: int, label: str, title: str | None = None) -> dict:
    q = dict(BINARY_Q)
    q.update(
        id=qid,
        title=title or f"Synthetic group subquestion {label}",
        label=label,
        description="",  # group-level text must be inherited on unpack
        resolution_criteria="",
        fine_print="",
    )
    return q


# Group post, shaped like the real API per forecasting-tools 0.3.4:
# post.group_of_questions.{description,resolution_criteria,fine_print,
# questions[]}; each subquestion has its own id and a `label`.
GROUP_PARTITION_POST = {
    "id": 39100,
    "group_of_questions": {
        "description": "Group-level background text.",
        "resolution_criteria": "Resolves to exactly one synthetic winner.",
        "fine_print": "Group fine print.",
        "questions": [
            _binary_sub(39101, "Alpha"),
            _binary_sub(39102, "Beta"),
            _binary_sub(39103, "Gamma"),
        ],
    },
}

GROUP_LADDER_POST = {
    "id": 39200,
    "group_of_questions": {
        "description": "Group-level background text.",
        "resolution_criteria": "Resolves per threshold.",
        "fine_print": "",
        "questions": [
            _binary_sub(39201, "10", "Will the synthetic count exceed 10?"),
            _binary_sub(39202, "20", "Will the synthetic count exceed 20?"),
            _binary_sub(39203, "30", "Will the synthetic count exceed 30?"),
        ],
    },
}

GROUP_DATE_LADDER_POST = {
    "id": 39300,
    "group_of_questions": {
        "description": "Group-level background text.",
        "resolution_criteria": "Resolves per date.",
        "fine_print": "",
        "questions": [
            _binary_sub(39301, "March 2027", "Will the synthetic event happen by March 2027?"),
            _binary_sub(39302, "June 2027", "Will the synthetic event happen by June 2027?"),
            _binary_sub(39303, "December 2027", "Will the synthetic event happen by December 2027?"),
        ],
    },
}


def group_payloads(subquestions: list[dict], probs: list[float]) -> list[dict]:
    return [
        build_forecast_payload(q, p) for q, p in zip(subquestions, probs)
    ]


def main() -> int:
    # 1. Binary
    payload = check(BINARY_Q)
    p = payload["probability_yes"]
    assert 0.01 <= p <= 0.99, p
    assert payload["probability_yes_per_category"] is None
    assert payload["continuous_cdf"] is None
    print(f"binary          OK  probability_yes={p}")

    # 2. Multiple choice
    payload = check(MC_Q)
    per_cat = payload["probability_yes_per_category"]
    assert set(per_cat) == set(MC_Q["options"]), per_cat
    total = sum(per_cat.values())
    assert abs(total - 1.0) < 1e-9, total
    assert all(0.0 < v < 1.0 for v in per_cat.values())
    assert payload["probability_yes"] is None
    assert payload["continuous_cdf"] is None
    print(f"multiple_choice OK  sum={total:.12f} options={list(per_cat)}")

    # 3. Numeric (linear axis)
    payload = check(NUMERIC_Q)
    cdf = payload["continuous_cdf"]
    assert len(cdf) == 201 == cdf_size_for_question(NUMERIC_Q), len(cdf)
    assert all(0.0 <= v <= 1.0 for v in cdf)
    assert all(b >= a for a, b in zip(cdf, cdf[1:])), "CDF not non-decreasing"
    steps = [b - a for a, b in zip(cdf, cdf[1:])]
    assert max(steps) <= 0.19, max(steps)
    assert cdf[0] <= 0.05 + 1e-9 and cdf[-1] >= 0.95 - 1e-9, (cdf[0], cdf[-1])
    print(f"numeric         OK  cdf[0]={cdf[0]:.3f} cdf[100]={cdf[100]:.3f} "
          f"cdf[-1]={cdf[-1]:.3f} max_step={max(steps):.3f}")

    # 4. Numeric (log-scaled axis via zero_point)
    payload = check(NUMERIC_LOG_Q)
    cdf = payload["continuous_cdf"]
    assert len(cdf) == 201
    assert all(b >= a for a, b in zip(cdf, cdf[1:]))
    print(f"numeric (log)   OK  cdf[0]={cdf[0]:.3f} cdf[-1]={cdf[-1]:.3f}")

    # 5. Discrete (cdf size = inbound_outcome_count + 1)
    payload = check(DISCRETE_Q)
    cdf = payload["continuous_cdf"]
    assert len(cdf) == 11 == cdf_size_for_question(DISCRETE_Q), len(cdf)
    assert all(b >= a for a, b in zip(cdf, cdf[1:]))
    print(f"discrete        OK  cdf points={len(cdf)}")

    # 6. Price-blindness: the forecaster's view must not contain prices
    for q in (BINARY_Q, MC_Q, NUMERIC_Q):
        view = blind_question_view(q)
        assert "aggregations" not in view, q["id"]
        assert "my_forecasts" not in view, q["id"]
        assert view["title"] == q["title"]
    print("blind view      OK  aggregations/my_forecasts stripped")

    # 7. Forecaster percentile monotonicity (input side of the CDF build)
    view = blind_question_view(NUMERIC_Q)
    percentiles = fleet_forecast(view)
    items = sorted(percentiles.items())
    assert all(b[0] > a[0] and b[1] > a[1] for a, b in zip(items, items[1:]))
    print("percentiles     OK  strictly increasing in prob and value")

    # 8. Private comment payload (REQUIRED on every forecast, notebook 38928)
    body = build_comment_payload(12345, "Reasoning trail for this forecast.")
    assert body["on_post"] == 12345 and body["parent"] is None
    assert body["included_forecast"] is True and body["is_private"] is True
    assert body["text"].strip()
    try:
        build_comment_payload(12345, "  ")
        raise AssertionError("empty comment text must be rejected")
    except ValueError:
        pass
    print("comment payload OK  private, included_forecast, non-empty enforced")

    # 9. Group intake: a group post unpacks into ordinary subquestions,
    #    each identified by its own id + label, inheriting group-level
    #    description / resolution criteria / fine print, and the blind
    #    view keeps the label (which subquestion am I?) while still
    #    stripping aggregations.
    assert is_group_post(GROUP_PARTITION_POST)
    subs = unpack_group_post(GROUP_PARTITION_POST)
    assert [q["id"] for q in subs] == [39101, 39102, 39103]
    assert [q["label"] for q in subs] == ["Alpha", "Beta", "Gamma"]
    assert all(q["description"] == "Group-level background text." for q in subs)
    assert all(q["group_post_id"] == 39100 for q in subs)
    assert all(q["group_question_ids"] == [39101, 39102, 39103] for q in subs)
    for q in subs:
        view = blind_question_view(q)
        assert view["label"] == q["label"] and "aggregations" not in view
        payload = check(q)  # same single-question code path as singles
        assert payload["probability_yes"] == 0.5  # stub base rate
    print("group intake    OK  3 subquestions unpacked (ids+labels, "
          "group text inherited, blind view keeps label)")

    # 10. Group partition: mutually exclusive binary buckets must sum to 1.
    #     Payloads are per-subquestion objects (no joint payload exists).
    payloads = group_payloads(subs, [0.50, 0.30, 0.20])
    assert [p["question"] for p in payloads] == [39101, 39102, 39103]
    assert all(p["probability_yes_per_category"] is None for p in payloads)
    problems = validate_group_consistency(subs, payloads, kind="partition")
    assert not problems, problems
    built = build_group_forecast_payloads(
        subs, {39101: 0.50, 39102: 0.30, 39103: 0.20}, kind="partition"
    )
    assert len(built) == 3
    total = sum(p["probability_yes"] for p in payloads)
    print(f"group partition OK  bucket sum={total:.6f} == 1, "
          "3 independent per-subquestion payloads")

    # 11a. Threshold ladder (survival phrasing): P(exceed T) must be
    #      non-increasing in T. Kind + direction inferred from labels.
    ladder_subs = unpack_group_post(GROUP_LADDER_POST)
    assert infer_group_kind(ladder_subs) == "threshold_ladder"
    payloads = group_payloads(ladder_subs, [0.80, 0.50, 0.20])
    problems = validate_group_consistency(ladder_subs, payloads)
    assert not problems, problems
    print("group ladder    OK  P(exceed 10/20/30) = .80/.50/.20 non-increasing")

    # 11b. Date ladder (CDF phrasing): P(by date) must be non-decreasing.
    date_subs = unpack_group_post(GROUP_DATE_LADDER_POST)
    payloads = group_payloads(date_subs, [0.20, 0.50, 0.80])
    problems = validate_group_consistency(
        date_subs, payloads, kind="threshold_ladder"
    )
    assert not problems, problems
    print("group date lad. OK  P(by Mar/Jun/Dec 2027) = .20/.50/.80 non-decreasing")

    # 12. Deliberately inconsistent group forecasts MUST be caught, and
    #     build_group_forecast_payloads must refuse to emit any payload.
    bad_partition = group_payloads(subs, [0.60, 0.50, 0.30])  # sums to 1.40
    problems = validate_group_consistency(subs, bad_partition, kind="partition")
    assert problems and "sum" in problems[0], problems
    try:
        build_group_forecast_payloads(
            subs, {39101: 0.60, 39102: 0.50, 39103: 0.30}, kind="partition"
        )
        raise AssertionError("inconsistent partition must not be emitted")
    except ValueError as exc:
        assert "sum" in str(exc), exc
    bad_ladder = group_payloads(ladder_subs, [0.20, 0.90, 0.10])  # rises at T=20
    problems = validate_group_consistency(ladder_subs, bad_ladder)
    assert problems and "non-increasing" in problems[0], problems
    bad_date = group_payloads(date_subs, [0.80, 0.50, 0.20])  # CDF falls
    problems = validate_group_consistency(
        date_subs, bad_date, kind="threshold_ladder"
    )
    assert problems and "non-decreasing" in problems[0], problems
    print("group invalid   OK  partition sum 1.40, ladder rise, and falling "
          "date CDF all caught; emission refused")

    print("\nALL SELF-TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
