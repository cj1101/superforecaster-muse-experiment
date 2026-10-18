"""
bot.py — Superforecaster fleet bot for Metaculus (Fall 2026 FutureEval).

Usage:
    python3 bot.py                      # DRY RUN on bot-testing-area (32977)
    python3 bot.py --project fall-2026  # DRY RUN on Fall 2026 (33121)
    python3 bot.py --project bot-testing-area --submit
                                        # LIVE submit to the testing area
    python3 bot.py --project fall-2026 --submit --post-comment
                                        # LIVE submit to Fall 2026 + private
                                        # reasoning comments

DRY-RUN IS THE DEFAULT. Nothing is submitted unless --submit is passed, and
--submit additionally requires METACULUS_API_TOKEN in the environment
(Secure Vault -> env var; see README.md go-live checklist).

Pipeline position (docs/FLEET_NORTH_STAR_PROMPT.md, four-stage topology):
    Stage 1 Discovery        -> Metaculus question intake (this file, run())
    Stage 2 Forecasters      -> price-blind individual forecasts
                                [INTEGRATION POINT A: fleet_forecast()]
    Stage 3 Supervisor /     -> pooling + calibration judge before anything
      Calibration Judge         leaves the building
                                [INTEGRATION POINT B: judge_and_pool()]
    Stage 4 Execution        -> payload formatting + submission
                                (metaculus_client.py; this file, submit path)

Metaculus scoring is log-score based (Peer Scores are differences of log
scores). Scoring technically counts only the last forecast before close,
BUT the Bot Tournament Resources page (notebook 38928, verified
2026-10-17) instructs: "Bot makers should only submit one forecast per
question in these bot-only tournaments", and the Competition Rules add
that a bot maker may not rerun a question because they dislike the
forecast. RULE FOR THIS BOT: forecast ONCE per question in Fall 2026 and
MiniBench; re-forecasting/resubmission is only for the bot-testing-area
(32977) and non-bot-only tournaments. A private comment is REQUIRED on
every forecast. See wave3/meta-rules-reconcile/RULES.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from group_module import validate_group_consistency
from metaculus_client import (
    PROJECTS,
    MetaculusClient,
    already_forecasted,
    blind_question_view,
    build_forecast_payload,
    default_numeric_percentiles,
    dumps_payloads,
    post_questions,
    validate_forecast_payload,
)

DEFAULT_PROJECT = "bot-testing-area"  # 32977 — unscored, one of every type


# ---------------------------------------------------------------------------
# Forecaster — STUB. This is where the fleet pipeline plugs in.
# ---------------------------------------------------------------------------

def fleet_forecast(question_view: dict) -> Any:
    """Return a forecast for one price-blind question view.

    INTEGRATION POINT A (Stage 2 — price-blind Forecasters):
    `question_view` has already been stripped of the community prediction
    and all price/aggregation fields (see blind_question_view). Replace the
    placeholder logic below with calls into the fleet pipeline, e.g.:

        ./sf forecast --domain <...> ...        (production_p ONLY)
        ./sf pool --probs a,b,c                 (log-odds pool of forecasters)

    Expected return shapes (consumed by build_forecast_payload):
      binary:          float, P(yes) — will be clipped to [0.01, 0.99]
      multiple_choice: list[float] aligned with question_view["options"]
      numeric/discrete: dict {percentile_prob: nominal_value},
                        e.g. {0.05: v5, 0.25: v25, 0.5: v50, 0.75: v75,
                              0.95: v95}
      date:            same as numeric, values = Unix timestamps (seconds)

    The placeholder below is deliberately information-free: binary 0.50,
    uniform multiple choice, and a wide percentile spread over the middle
    of the question's scaling range. It exists to exercise the plumbing,
    NOT to score. Do not submit placeholders to a scored tournament.

    NUMERIC PRODUCTION PATH (Wave 3 default, see judge_and_pool): for
    data-backed numeric questions, Stage 2 should elicit percentiles with
    cdf_module.disciplined_percentiles (climatology-anchored; spot +
    trailing return windows) or cdf_module.climatology_percentiles —
    cdf_module.py is the scaffold's single numeric implementation and
    metaculus_client re-exports those helpers. Declared percentiles then
    pass through the default strategy in judge_and_pool before any CDF
    is built.
    """
    qtype = question_view["type"]
    if qtype == "binary":
        return 0.5  # PLACEHOLDER base rate — no information content
    if qtype == "multiple_choice":
        n = len(question_view["options"])
        return [1.0 / n] * n  # PLACEHOLDER uniform
    if qtype in ("numeric", "discrete", "date"):
        scaling = question_view["scaling"]
        lo, hi = float(scaling["range_min"]), float(scaling["range_max"])
        span = hi - lo
        # PLACEHOLDER: broad spread centred on the range midpoint.
        return {
            0.05: lo + 0.10 * span,
            0.25: lo + 0.35 * span,
            0.50: lo + 0.50 * span,
            0.75: lo + 0.65 * span,
            0.95: lo + 0.90 * span,
        }
    raise ValueError(f"unsupported question type: {qtype!r}")


def judge_and_pool(question_view: dict, forecast: Any) -> Any:
    """Stage 3 — Supervisor / Calibration Judge hook.

    INTEGRATION POINT B: in production this is where multiple Stage-2
    forecaster outputs get pooled (track-record-weighted log-odds pool,
    ./sf pool), checked against the fleet's calibration discipline
    (ECE gates; engine cap [0.01, 0.99]; extremization factor 1.00 by
    default), and vetoed or adjusted before submission. Binary and
    multiple-choice forecasts still pass through unchanged here.

    DEFAULT NUMERIC STRATEGY (Wave 3, applied to numeric/discrete
    percentile dicts): elicitation = disciplined percentiles ANCHORED
    to climatology, with naive-normal density smoothing for log-score
    contexts — implemented by metaculus_client.default_numeric_
    percentiles (anchoring when climatology percentiles are supplied
    upstream; naive-normal smoothing always). Evidence
    (wave3/meta-forecast-types/RESULTS.md, pooled crypto n=603):
    disciplined+anchored beats raw climatology on log score
    (+0.137 [+0.009, +0.277]) but does NOT reliably beat a naive normal
    fit (+0.018, CI includes 0, sign flips per asset); all arms are
    near-calibrated. HONEST CAVEAT: this default's log-score edge over
    a plain naive normal fit is unproven — it is adopted because
    anchoring fixes raw climatology and smoothing fixes the lumpy
    density log score punishes, not because it is a validated edge.
    Date forecasts are left unsmoothed (the evaluation studied
    numeric asset closes, not dates). CDF construction itself is
    cdf_module.percentiles_to_cdf via metaculus_client (single
    implementation; PCHIP tails are clipped there, never extrapolated).
    """
    if question_view.get("type") in ("numeric", "discrete") and isinstance(
        forecast, dict
    ):
        return default_numeric_percentiles(forecast, question_view)
    return forecast


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(
    project: str,
    submit: bool = False,
    post_comment: bool = False,
    skip_forecasted: bool = True,
    limit: int | None = None,
) -> int:
    tournament_id: int | str = PROJECTS.get(str(project), project)
    mode = "SUBMIT" if submit else "DRY-RUN"
    print(f"[bot] project={project!r} (tournaments={tournament_id!r}) mode={mode}")

    is_testing_area = str(tournament_id) in ("bot-testing-area", "32977")
    # RULE (notebook 38928): one forecast per question in bot-only
    # tournaments. Refuse a live re-forecast run outside the testing area.
    if submit and not skip_forecasted and not is_testing_area:
        print("[bot] REFUSED: --no-skip-forecasted on a live scored run "
              "would re-forecast questions, violating the one-forecast-per-"
              "question rule for bot-only tournaments. Re-forecasting is "
              "only allowed in bot-testing-area (32977).")
        return 2
    # RULE (notebook 38928): "Bots are required to leave private comments
    # on all questions they forecast." A live submit therefore always
    # posts the private comment; --post-comment is kept for compatibility.
    if submit and not post_comment:
        print("[bot] private comments are REQUIRED on every forecast — "
              "enabling comment posting for this live run.")
        post_comment = True

    client: MetaculusClient | None = None
    if submit:
        client = MetaculusClient()  # raises early if no token — fail fast
        post_groups = list(client.iter_open_posts_with_questions(tournament_id))
    else:
        # Dry-run still lists live questions when a token is available, but
        # must work with NO token at all: fall back to a clear notice. The
        # self-test (self_test.py) exercises payload building offline.
        try:
            client = MetaculusClient()
            post_groups = list(client.iter_open_posts_with_questions(tournament_id))
        except Exception as exc:  # no token / offline — not an error here
            print(f"[bot] cannot list questions ({exc}).")
            print("[bot] dry-run without API access: run self_test.py for "
                  "the offline payload checks, or set METACULUS_API_TOKEN.")
            return 0

    if limit is not None:
        # Limit counts QUESTIONS (group subquestions count individually).
        trimmed: list[tuple[dict, list[dict]]] = []
        remaining = limit
        for post, group_questions in post_groups:
            if remaining <= 0:
                break
            trimmed.append((post, group_questions[:remaining]))
            remaining -= len(group_questions)
        post_groups = trimmed
    n_questions = sum(len(qs) for _p, qs in post_groups)
    print(f"[bot] {n_questions} open question(s) "
          f"across {len(post_groups)} post(s)")

    payloads: list[dict] = []
    for post, group_questions in post_groups:
        is_group = len(group_questions) > 1 or "group_of_questions" in post
        group_payloads: list[dict] = []
        forecasted_subs: list[dict] = []
        for question in group_questions:
            if skip_forecasted and already_forecasted(question):
                print(f"[bot] skip q{question['id']} (already forecasted): "
                      f"{question['title'][:80]}")
                continue
            # Group subquestions go through the SAME integration points
            # (blind view -> fleet_forecast -> judge_and_pool -> payload)
            # as single questions; the blind view carries the subquestion's
            # `label` so the forecaster knows which group option it is.
            view = blind_question_view(question)
            forecast = judge_and_pool(view, fleet_forecast(view))
            payload = build_forecast_payload(question, forecast)
            problems = validate_forecast_payload(question, payload)
            if problems:
                print(f"[bot] INVALID payload for q{question['id']}: {problems}")
                return 2
            group_payloads.append(payload)
            forecasted_subs.append(question)
            print(f"[bot] q{question['id']} [{question['type']}] "
                  f"{question['title'][:80]}")
        if is_group and group_payloads:
            # Joint gate: NO payload from a group is emitted unless the
            # set is jointly consistent (partition sums / ladder
            # monotonicity where the group kind implies them — see
            # group_module). Metaculus enforces nothing server-side.
            joint = validate_group_consistency(forecasted_subs, group_payloads)
            if joint:
                print(f"[bot] INVALID group forecast for post {post.get('id')}: "
                      f"{joint}")
                return 2
            print(f"[bot] group post {post.get('id')}: "
                  f"{len(group_payloads)} subquestion payload(s), "
                  "joint consistency OK")
        payloads.extend(group_payloads)

    if not payloads:
        print("[bot] nothing to submit")
        return 0

    print("[bot] payloads:")
    print(dumps_payloads(payloads))

    if not submit:
        print(f"[bot] DRY-RUN: {len(payloads)} payload(s) built, "
              "nothing submitted. Re-run with --submit to go live.")
        return 0

    assert client is not None
    result = client.submit_forecasts(payloads)
    print(f"[bot] submitted {len(payloads)} forecast(s): "
          f"{json.dumps(result)[:300]}")

    if post_comment:
        # Private reasoning trail on each post. Post IDs are recovered from
        # a fresh listing because payloads carry question IDs, not post IDs.
        print("[bot] posting required private comments: comment text is produced by "
              "the fleet pipeline at INTEGRATION POINT A in production; "
              "the scaffold posts a marker comment.")
        posts = client.list_tournament_posts(tournament_id, limit=50)
        post_by_question = {
            q["id"]: p["id"]
            for p in posts.get("results", [])
            for q in post_questions(p)
        }
        for payload in payloads:
            post_id = post_by_question.get(payload["question"])
            if post_id:
                client.post_comment(
                    post_id,
                    "Superforecaster fleet bot (scaffold run). Reasoning "
                    "trail will be attached by the production pipeline.",
                )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help="tournament slug or id: bot-testing-area (32977, default), "
        "fall-2026 / fall-futureeval-2026 (33121), minibench, or any id/slug",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="actually submit forecasts (default is dry-run)",
    )
    parser.add_argument(
        "--post-comment",
        action="store_true",
        help="post a private reasoning comment per question (with --submit "
        "this is now automatic: comments are REQUIRED on every forecast)",
    )
    parser.add_argument(
        "--no-skip-forecasted",
        action="store_true",
        help="TESTING AREA ONLY: re-forecast already-forecasted questions. "
        "Bot-only tournaments (Fall 2026, MiniBench) allow ONE forecast "
        "per question; live scored runs with this flag are refused.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    return run(
        project=args.project,
        submit=args.submit,
        post_comment=args.post_comment,
        skip_forecasted=not args.no_skip_forecasted,
        limit=args.limit,
    )


if __name__ == "__main__":
    sys.exit(main())
