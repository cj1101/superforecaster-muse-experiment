"""
group_module.py — Group-question support for the Metaculus scaffold.

Wave 3 (meta-group-questions). Facts implemented here were established
2026-10-17 from Metaculus's own public sources (archived in
the project research notes):

  * A group post carries the group under the post key `group_of_questions`
    (not `question`). The group object has `questions` (subquestion objects,
    each with its own `id` and a `label`), plus group-level `description`,
    `resolution_criteria` and `fine_print` that apply to every subquestion
    (forecasting-tools 0.3.4, MetaculusClient._unpack_group_question).
  * Forecasts submit PER SUBQUESTION, independently: the same
    POST /api/questions/forecast/ single-question payload shape, keyed by
    the subquestion's own question id (forecasting-tools
    _post_question_prediction; metac-bot-template post_question_prediction).
    There is no joint group forecast object.
  * Each subquestion is scored independently, forecasting every subquestion
    is encouraged but not required, and NO server-side consistency
    constraint (partition sum, ladder monotonicity) is documented anywhere
    (notebook 9861; quick-start docs). Consistency is therefore enforced
    HERE, scaffold-side, before any payload is emitted.

Group kinds understood by the validator:
  partition        — binary subquestions that are mutually exclusive and
                     exhaustive buckets of one outcome (e.g. "who wins?"):
                     probability_yes values must sum to 1 (within tol).
  threshold_ladder — binary subquestions that are points on one cumulative
                     scale, ordered by a threshold parsed from each
                     subquestion's `label` (fallback: `group_rank`, title).
                     Direction: "survival" phrasing (exceeds / at least /
                     more than / >) => probabilities NON-INCREASING in the
                     threshold; "by <date/value>" CDF phrasing =>
                     NON-DECREASING. Ambiguous ladders default to
                     non-increasing, per the task specification.
  independent      — no joint constraint (the documented default for
                     groups: e.g. "top-5 by market cap" k-of-n groups, or
                     unrelated subquestions sharing a page).

`infer_group_kind` is deliberately conservative: it returns "partition"
ONLY when the group payload explicitly declares it (keys
`group_kind`/`kind` == "partition", or a truthy `is_partition` /
`mutually_exclusive` flag — none of which public docs guarantee exist on
read payloads), and "threshold_ladder" only when every subquestion yields
a distinct parseable threshold. Anything else is "independent". Callers
may always pass `kind=` explicitly to override.
"""

from __future__ import annotations

import copy
import math
import re
from datetime import datetime
from typing import Any, Iterable, Optional

GROUP_KINDS = ("partition", "threshold_ladder", "independent")

_SURVIVAL_HINTS = ("exceed", "at least", "more than", "greater", ">=", ">", "above", "over ")
_CDF_HINTS = ("by ", "before ", "by the end")


# ---------------------------------------------------------------------------
# Intake: unpack a group post into ordinary question dicts
# ---------------------------------------------------------------------------

def is_group_post(post: dict) -> bool:
    return isinstance(post.get("group_of_questions"), dict)


def unpack_group_post(post: dict) -> list[dict]:
    """Return the group's subquestions as ordinary question dicts.

    Mirrors forecasting-tools' _unpack_group_question: group-level
    description / resolution_criteria / fine_print are copied onto each
    subquestion when the subquestion does not carry its own, and each
    subquestion is annotated with `group_post_id` and
    `group_question_ids` (all sibling ids) so downstream stages keep the
    grouping without changing the question shape the scaffold already
    handles. The subquestion's own `label` (its group option) is a native
    API field and is preserved as-is.
    """
    group = post["group_of_questions"]
    subquestions = group.get("questions") or []
    ids = [int(q["id"]) for q in subquestions]
    out: list[dict] = []
    for q in subquestions:
        sq = copy.deepcopy(q)
        for field in ("description", "resolution_criteria", "fine_print"):
            if not sq.get(field) and group.get(field):
                sq[field] = group[field]
        sq["group_post_id"] = post.get("id")
        sq["group_question_ids"] = list(ids)
        out.append(sq)
    return out


def post_questions(post: dict) -> list[dict]:
    """All forecastable questions riding on one post (1 for single posts)."""
    if is_group_post(post):
        return unpack_group_post(post)
    question = post.get("question")
    return [question] if question else []


# ---------------------------------------------------------------------------
# Threshold parsing (ladder ordering)
# ---------------------------------------------------------------------------

_MONTH_FORMATS = ("%B %Y", "%b %Y", "%Y-%m-%d", "%Y/%m/%d", "%d %B %Y", "%B %d, %Y")


def _finite_or_none(value: float) -> Optional[float]:
    """Return `value` only if it is finite; NaN/inf are not thresholds.

    NaN has no order (every comparison is False) and inf/-inf would pin a
    subquestion to an end of a ladder regardless of its real rank, so a
    non-finite parsed value must be treated as "nothing parseable" rather
    than as a threshold (SF-BL-10310).
    """
    return value if math.isfinite(value) else None


def parse_threshold(subquestion: dict) -> Optional[float]:
    """Best-effort numeric threshold for ordering a ladder subquestion.

    Tries, in order: the subquestion `label` as a date, the label as a
    number, the title as a date/number, then a numeric `group_rank`.
    Dates map to Unix timestamps so date and numeric ladders share one
    ordering scale. Returns None when nothing parseable is found, and
    also when the only candidate is non-finite (NaN / +/-inf): a corrupt
    `group_rank` must not manufacture a threshold.
    """
    for text in (subquestion.get("label"), subquestion.get("title")):
        if not text:
            continue
        text = str(text).strip()
        for fmt in _MONTH_FORMATS:
            try:
                parsed = _finite_or_none(datetime.strptime(text, fmt).timestamp())
            except ValueError:
                parsed = None
            if parsed is not None:
                return parsed
        # A date embedded in a longer label, e.g. "By March 2027"
        m = re.search(r"(\d{4}-\d{2}-\d{2})", text)
        if m:
            try:
                parsed = _finite_or_none(
                    datetime.strptime(m.group(1), "%Y-%m-%d").timestamp()
                )
            except ValueError:
                parsed = None
            if parsed is not None:
                return parsed
        m = re.search(r"(January|February|March|April|May|June|July|August|"
                      r"September|October|November|December)\s+(\d{4})", text)
        if m:
            try:
                parsed = _finite_or_none(
                    datetime.strptime(f"{m.group(1)} {m.group(2)}", "%B %Y").timestamp()
                )
            except ValueError:
                parsed = None
            if parsed is not None:
                return parsed
        m = re.search(r"-?\d[\d,]*\.?\d*", text)
        if m:
            try:
                parsed = _finite_or_none(float(m.group(0).replace(",", "")))
            except ValueError:
                parsed = None
            if parsed is not None:
                return parsed
    rank = subquestion.get("group_rank")
    if isinstance(rank, (int, float)):
        return _finite_or_none(float(rank))
    return None


def infer_ladder_direction(subquestions: list[dict]) -> str:
    """"non_decreasing" for CDF ('by …') phrasing, else "non_increasing"."""
    text = " ".join(
        str(q.get("title") or "") + " " + str(q.get("label") or "")
        for q in subquestions
    ).lower()
    if any(h in text for h in _SURVIVAL_HINTS):
        return "non_increasing"
    if any(h in text for h in _CDF_HINTS):
        return "non_decreasing"
    return "non_increasing"  # task-specified default for threshold ladders


def infer_group_kind(group_or_subquestions: Any) -> str:
    """Conservative group-kind inference (see module docstring)."""
    if isinstance(group_or_subquestions, dict):
        group = group_or_subquestions
        declared = group.get("group_kind") or group.get("kind")
        if declared in GROUP_KINDS:
            return str(declared)
        if group.get("is_partition") or group.get("mutually_exclusive"):
            return "partition"
        subquestions = group.get("questions") or []
    else:
        subquestions = list(group_or_subquestions)
    if len(subquestions) >= 2 and all(q.get("type") == "binary" for q in subquestions):
        thresholds = [parse_threshold(q) for q in subquestions]
        # Defence in depth (SF-BL-10310): a ladder needs every threshold
        # present, finite and distinct. NaN in particular must never pass
        # this check on its own merits — NaN != NaN, so a set-distinctness
        # test alone would count two NaN ranks as "distinct thresholds".
        if (
            all(t is not None and math.isfinite(t) for t in thresholds)
            and len(set(thresholds)) == len(thresholds)
        ):
            return "threshold_ladder"
    return "independent"


# ---------------------------------------------------------------------------
# Joint-consistency validation
# ---------------------------------------------------------------------------

def validate_group_consistency(
    subquestions: list[dict],
    payloads: Iterable[dict],
    kind: Optional[str] = None,
    direction: Optional[str] = None,
    partition_tolerance: float = 0.01,
    ladder_tolerance: float = 1e-9,
    require_all: bool = False,
) -> list[str]:
    """Validate a set of per-subquestion payloads JOINTLY.

    `payloads` are ordinary single-question payloads (build_forecast_payload
    output), one per forecasted subquestion, keyed by subquestion id in
    their "question" field. Returns a list of problems (empty = valid).
    Per-payload shape/range/CDF checks remain validate_forecast_payload's
    job in metaculus_client; this validator adds only the joint rules:

      partition:        sum of binary probability_yes == 1 (within
                        partition_tolerance); every payload must be binary.
      threshold_ladder: binary probability_yes monotone in the parsed
                        threshold, in `direction` (inferred if omitted).
      independent:      no joint rule (documented Metaculus default).

    Partial coverage is allowed unless require_all=True (public docs:
    forecasting every subquestion is encouraged, not required), but a
    payload for an unknown question id, or two payloads for one
    subquestion, is always a problem.
    """
    problems: list[str] = []
    payloads = list(payloads)
    by_id: dict[int, dict] = {}
    known_ids = {int(q["id"]) for q in subquestions}
    for payload in payloads:
        qid = payload.get("question")
        if qid not in known_ids:
            problems.append(f"group payload for unknown subquestion id {qid!r}")
        elif qid in by_id:
            problems.append(f"duplicate group payload for subquestion {qid}")
        else:
            by_id[qid] = payload
    if require_all and set(by_id) != known_ids:
        missing = sorted(known_ids - set(by_id))
        problems.append(f"group forecast incomplete: missing subquestions {missing}")
    if problems or not by_id:
        return problems

    resolved_kind = kind or infer_group_kind(subquestions)
    if resolved_kind not in GROUP_KINDS:
        return [f"unknown group kind {resolved_kind!r}"]
    subs_by_id = {int(q["id"]): q for q in subquestions}

    if resolved_kind == "partition":
        total = 0.0
        for qid, payload in by_id.items():
            if subs_by_id[qid].get("type") != "binary":
                problems.append(f"partition group subquestion {qid} is not binary")
                continue
            p = payload.get("probability_yes")
            if p is None:
                problems.append(f"partition subquestion {qid} payload has no probability_yes")
                continue
            total += float(p)
        if not problems and abs(total - 1.0) > partition_tolerance:
            problems.append(
                f"partition bucket probabilities sum to {total:.6f}, not 1 "
                f"(tolerance {partition_tolerance})"
            )
    elif resolved_kind == "threshold_ladder":
        points: list[tuple[float, int, float]] = []
        for qid, payload in by_id.items():
            threshold = parse_threshold(subs_by_id[qid])
            if threshold is None or not math.isfinite(threshold):
                problems.append(f"ladder subquestion {qid} has no parseable threshold")
                continue
            p = payload.get("probability_yes")
            if p is None:
                problems.append(
                    f"ladder subquestion {qid} payload has no probability_yes "
                    "(ladder consistency is defined for binary subquestions)"
                )
                continue
            points.append((threshold, qid, float(p)))
        if not problems:
            points.sort()
            resolved_direction = direction or infer_ladder_direction(subquestions)
            for (t0, q0, p0), (t1, q1, p1) in zip(points, points[1:]):
                if resolved_direction == "non_increasing" and p1 > p0 + ladder_tolerance:
                    problems.append(
                        f"ladder not non-increasing in threshold: subquestion {q1} "
                        f"(threshold {t1:g}) has p={p1:.4f} > subquestion {q0} "
                        f"(threshold {t0:g}) p={p0:.4f}"
                    )
                elif resolved_direction == "non_decreasing" and p1 < p0 - ladder_tolerance:
                    problems.append(
                        f"ladder not non-decreasing in threshold: subquestion {q1} "
                        f"(threshold {t1:g}) has p={p1:.4f} < subquestion {q0} "
                        f"(threshold {t0:g}) p={p0:.4f}"
                    )
    return problems
