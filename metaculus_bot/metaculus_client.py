"""
metaculus_client.py — thin Metaculus API client for the Superforecaster fleet bot.

Facts implemented here were verified against Metaculus's own public sources
(2026-10-17):
  - Template bot (no-framework reference implementation):
    https://github.com/Metaculus/metac-bot-template/blob/main/main_with_no_framework.py
  - forecasting-tools package docs (MetaculusClient / MetaculusApi):
    https://pypi.org/project/forecasting-tools/
  - Bot Tournament Resources page (project IDs / slugs, token instructions):
    https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/

API summary (as implemented by Metaculus's template bot):
  - Base URL: https://www.metaculus.com/api
  - Auth: header  Authorization: Token <METACULUS_TOKEN>
    (DRF-style Token auth — NOT Bearer.)
  - List tournament posts:
        GET /api/posts/?tournaments=<id-or-slug>&statuses=open
            &forecast_type=binary,multiple_choice,numeric,discrete
            &include_description=true&limit=<n>&offset=<m>&order_by=-hotness
    Response: {"results": [post, ...]}; each single-question post carries a
    "question" object. Question IDs differ from post IDs.
  - Post detail: GET /api/posts/<post_id>/  -> {"question": {...}, ...}
  - Submit forecasts:
        POST /api/questions/forecast/
        body = JSON LIST of forecast objects:
          [{"question": <question_id>, "source": "api",
            "probability_yes": <float|null>,
            "probability_yes_per_category": <{option_label: prob}|null>,
            "continuous_cdf": <[cdf_size floats]|null>}]
    Exactly one of the three forecast fields is non-null, per question type.
  - Private reasoning comment:
        POST /api/comments/create/
        {"text": ..., "parent": None, "included_forecast": True,
         "is_private": True, "on_post": <post_id>}

Question object fields used here (from GET /api/posts/ payloads):
  id, title, type ("binary" | "multiple_choice" | "numeric" | "discrete"
  | "date"), description, resolution_criteria, fine_print, status,
  scheduled_close_time, options (list[str], multiple_choice only),
  scaling {"range_min", "range_max", "zero_point", "inbound_outcome_count"},
  open_upper_bound, open_lower_bound, unit, my_forecasts.

Numeric CDF format: Metaculus expects `continuous_cdf` with 201 values for
numeric questions (inbound_outcome_count + 1 values for discrete questions).
Point i is the CDF value at scaled location i/(size-1) across
[scaling.range_min, scaling.range_max]. When scaling.zero_point is not null
the axis is log-scaled:
    nominal(l) = range_min + (range_max - range_min)
                 * (deriv_ratio**l - 1) / (deriv_ratio - 1),
    deriv_ratio = (range_max - zero_point) / (range_min - zero_point).
The fleet's forecasters declare *percentiles* ({prob: value}); this module
converts them to a CDF exactly as the template bot does.

CDF implementation (2026-10-17, Wave 3 meta-cdf-integrate): ALL CDF
construction lives in the sibling module `cdf_module.py` — the verified
Wave 3 module, adopted here as the single numeric implementation.
`percentiles_to_cdf` below is a thin adapter that keeps this module's
original public signature ({prob: value}, question) -> list[float] and
delegates to cdf_module.percentiles_to_cdf(..., interp="linear"), which
was proven equivalent to the scaffold's former inline implementation
(max abs diff < 1e-12 on linear, log-scaled and discrete cases; 16/16
tests in wave3/meta-forecast-types/test_module.py). Do NOT re-implement
CDF interpolation in this file. cdf_module also carries the elicitation
helpers (climatology / naive-normal / disciplined) and scoring utilities.
This now makes numpy + scipy hard dependencies of the scaffold.
Tail note: cdf_module's optional interp="pchip" never extrapolates the
cubic outside the declared percentiles — PCHIP tail extrapolation was
measured to explode (F = 7.3 / -10.3 at grid edges); it evaluates on
nominals clipped to the declared range, so tails are constant, exactly
like the linear construction.

Token handling: the token is read from the METACULUS_API_TOKEN environment
variable (METACULUS_TOKEN is accepted as a fallback, matching the template
bot's GitHub secret name). It is never hardcoded, never printed, and never
included in reprs or error messages.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any, Iterable, Iterator, Optional

import numpy as np
import requests
from scipy import stats as _scipy_stats

# Single CDF implementation — see module docstring. Everything numeric
# (grid geometry, percentiles->CDF, elicitation, CDF scoring) is imported
# from cdf_module and re-exported where the scaffold's callers need it.
from cdf_module import (  # noqa: E402
    DEFAULT_PERCENTILES,
    cdf_at,
    cdf_to_bin_probs,
    climatology_percentiles,
    crps_from_cdf,
    disciplined_percentiles,
    grid_nominals,
    interval_log_score_from_cdf,
    log_score_from_cdf,
    mc_scores,
    naive_normal_percentiles,
    nominal_at_location,
)
from cdf_module import percentiles_to_cdf as _cdf_percentiles_to_cdf  # noqa: E402

# Group-question intake + joint-consistency validation (Wave 3,
# meta-group-questions). See group_module.py for the verified API facts:
# group posts carry `group_of_questions.questions`, forecasts submit per
# subquestion independently, and no consistency constraint is enforced
# server-side — the validator in group_module is the scaffold's own gate.
from group_module import (  # noqa: E402
    infer_group_kind,
    is_group_post,
    post_questions,
    unpack_group_post,
    validate_group_consistency,
)

__all_group_reexports__ = (
    "infer_group_kind", "is_group_post", "post_questions",
    "unpack_group_post", "validate_group_consistency",
)

__all_cdf_reexports__ = (
    "DEFAULT_PERCENTILES", "cdf_at", "cdf_to_bin_probs",
    "climatology_percentiles", "crps_from_cdf", "disciplined_percentiles",
    "grid_nominals", "interval_log_score_from_cdf", "log_score_from_cdf",
    "mc_scores", "naive_normal_percentiles", "nominal_at_location",
)

API_BASE_URL = "https://www.metaculus.com/api"

# Well-known projects (verified on the Bot Tournament Resources page and in
# the template bot source, 2026-10-17). Slugs and numeric IDs both work as
# the `tournaments` query parameter.
PROJECTS = {
    "fall-2026": "fall-futureeval-2026",        # numeric id 33121
    "fall-futureeval-2026": "fall-futureeval-2026",
    "33121": "fall-futureeval-2026",
    "bot-testing-area": "bot-testing-area",     # numeric id 32977
    "32977": "bot-testing-area",
    "minibench": "minibench",
}

QUESTION_TYPES = ("binary", "multiple_choice", "numeric", "discrete", "date")

# Post-level forecast_type filter values: single-question types plus the
# group container type (forecasting-tools QuestionFullType, verified
# 2026-10-17 — see group_module.py). Conditional posts are still skipped.
POST_FORECAST_TYPES = QUESTION_TYPES + ("group_of_questions",)

# Fleet engine cap: Metaculus also requires binary probabilities in
# [0.01, 0.99] (forecasting-tools docs), which matches the fleet's cap.
PROB_MIN, PROB_MAX = 0.01, 0.99

DEFAULT_CDF_SIZE = 201
# Metaculus rejects CDFs whose largest single-step probability mass exceeds
# 0.2 (for a 201-point CDF). We warn at 0.19 and let the server be the judge.
MAX_CDF_STEP_WARN = 0.19


class MetaculusError(RuntimeError):
    """Raised on any non-2xx Metaculus response. Never contains the token."""


def _load_token() -> str:
    token = os.environ.get("METACULUS_API_TOKEN") or os.environ.get("METACULUS_TOKEN")
    if not token:
        raise MetaculusError(
            "No Metaculus token found. Set METACULUS_API_TOKEN (see README "
            "go-live checklist: token comes from the Secure Vault, never "
            "from a file committed to a repo)."
        )
    return token


class MetaculusClient:
    """Minimal sync client. One instance per run is fine."""

    def __init__(self, token: Optional[str] = None, timeout: float = 30.0):
        self._token = token if token is not None else _load_token()
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Token {self._token}",
                "Content-Type": "application/json",
            }
        )

    def __repr__(self) -> str:  # never leak the token
        return f"MetaculusClient(base={API_BASE_URL!r}, token=<redacted>)"

    # -- low-level ---------------------------------------------------------
    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{API_BASE_URL}{path}"
        resp = self._session.request(method, url, timeout=self.timeout, **kwargs)
        if not resp.ok:
            raise MetaculusError(
                f"{method} {path} -> HTTP {resp.status_code}: {resp.text[:500]}"
            )
        if not resp.content:
            return None
        return resp.json()

    # -- reads -------------------------------------------------------------
    def list_tournament_posts(
        self,
        tournament_id: int | str,
        statuses: str = "open",
        limit: int = 50,
        offset: int = 0,
        forecast_types: Iterable[str] = POST_FORECAST_TYPES,
    ) -> dict:
        """One page of posts for a tournament (raw API dict)."""
        params = {
            "limit": limit,
            "offset": offset,
            "order_by": "-hotness",
            "forecast_type": ",".join(forecast_types),
            "tournaments": [tournament_id],
            "statuses": statuses,
            "include_description": "true",
        }
        return self._request("GET", "/posts/", params=params)

    def iter_open_posts_with_questions(
        self, tournament_id: int | str, page_size: int = 50
    ) -> Iterator[tuple[dict, list[dict]]]:
        """Yield (post, [question, ...]) for every open post.

        Single-question posts yield a one-element list; GROUP posts yield
        their unpacked subquestions (group_module.unpack_group_post) so
        callers can run joint-consistency validation over the set before
        emitting any payload. Conditional posts (post key `conditional`)
        are still skipped — group support does not cover them.
        """
        offset = 0
        while True:
            page = self.list_tournament_posts(
                tournament_id, limit=page_size, offset=offset
            )
            results = page.get("results", [])
            if not results:
                return
            for post in results:
                questions = [
                    q for q in post_questions(post) if q.get("status") == "open"
                ]
                if questions:
                    yield post, questions
            offset += len(results)
            if page.get("next") is None and len(results) < page_size:
                return

    def iter_open_questions(
        self, tournament_id: int | str, page_size: int = 50
    ) -> Iterator[dict]:
        """Yield every open question, group subquestions unpacked.

        Group subquestions arrive as ordinary question dicts annotated
        with `group_post_id` / `group_question_ids` (see group_module).
        Callers that need joint validation should prefer
        iter_open_posts_with_questions(), which preserves the grouping.
        """
        for _post, questions in self.iter_open_posts_with_questions(
            tournament_id, page_size=page_size
        ):
            yield from questions

    def get_post(self, post_id: int) -> dict:
        return self._request("GET", f"/posts/{post_id}/")

    def get_question_detail(self, post_id: int) -> dict:
        """Full question object for a post (includes my_forecasts)."""
        return self.get_post(post_id)["question"]

    # -- writes ------------------------------------------------------------
    def submit_forecasts(self, payloads: list[dict]) -> Any:
        """POST a list of forecast payloads (see build_forecast_payload)."""
        validate_submission_batch(payloads)
        return self._request("POST", "/questions/forecast/", json=payloads)

    def post_comment(self, post_id: int, text: str, is_private: bool = True) -> Any:
        """Post a comment (private by default — the bot's reasoning trail)."""
        return self._request(
            "POST", "/comments/create/", json=build_comment_payload(post_id, text, is_private)
        )


def build_comment_payload(post_id: int, text: str, is_private: bool = True) -> dict:
    """Pure builder for POST /api/comments/create/ (offline-testable).

    Bot Tournament rules require a private comment on EVERY forecast in
    bot-only tournaments (resources notebook 38928), so live submission
    paths must attach one with included_forecast=True, is_private=True.
    """
    if not text or not text.strip():
        raise ValueError("comment text must be non-empty (required on every forecast)")
    return {
        "text": text,
        "parent": None,
        "included_forecast": True,
        "is_private": is_private,
        "on_post": int(post_id),
    }


# ---------------------------------------------------------------------------
# Payload construction + validation (pure functions — unit-testable offline)
# ---------------------------------------------------------------------------

def clip_prob(p: float) -> float:
    return min(max(float(p), PROB_MIN), PROB_MAX)


def normalize_probabilities(probs: list[float]) -> list[float]:
    """Clamp each prob to [0.01, 0.99], renormalize to sum exactly 1."""
    if not probs:
        raise ValueError("empty probability list")
    clamped = [clip_prob(p) for p in probs]
    total = sum(clamped)
    out = [p / total for p in clamped]
    out[-1] += 1.0 - sum(out)  # absorb float error on the last option
    return out


def cdf_size_for_question(question: dict) -> int:
    """201 for numeric/date; inbound_outcome_count + 1 for discrete."""
    if question.get("type") == "discrete":
        count = (question.get("scaling") or {}).get("inbound_outcome_count")
        if count:
            return int(count) + 1
    return DEFAULT_CDF_SIZE


def _nominal_at_location(
    location: float, range_min: float, range_max: float, zero_point: Optional[float]
) -> float:
    """Map a scaled CDF location in [0, 1] to a nominal (real-world) value.

    Backwards-compatible private wrapper — the implementation is
    cdf_module.nominal_at_location (single numeric implementation).
    """
    return float(
        nominal_at_location(
            location,
            float(range_min),
            float(range_max),
            None if zero_point is None else float(zero_point),
        )
    )


def percentiles_to_cdf(
    percentiles: dict[float, float], question: dict
) -> list[float]:
    """Convert declared percentiles {prob: value} to a Metaculus CDF.

    `percentiles` maps probabilities in (0, 1) to nominal values
    (for date questions, values are Unix timestamps in seconds — the API
    treats dates on the same scaled axis). Mirrors the template bot's
    generate_continuous_cdf: evaluates the implied CDF at cdf_size evenly
    spaced scaled locations, honoring log scaling via scaling.zero_point.

    ADAPTER ONLY: the conversion itself is cdf_module.percentiles_to_cdf
    with interp="linear" (verified scaffold-equivalent, max diff < 1e-12).
    This wrapper's sole jobs are unpacking the question's scaling, picking
    the CDF size for the question type, and returning a plain list[float]
    so the scaffold's public interface is unchanged.
    """
    scaling = question.get("scaling") or {}
    cdf = _cdf_percentiles_to_cdf(
        dict(percentiles),
        float(scaling["range_min"]),
        float(scaling["range_max"]),
        None if scaling.get("zero_point") is None else float(scaling["zero_point"]),
        size=cdf_size_for_question(question),
        interp="linear",
    )
    return [float(v) for v in cdf]


# ---------------------------------------------------------------------------
# Default numeric-forecast strategy (Wave 3 evaluation defaults)
# ---------------------------------------------------------------------------
# Evidence (wave3/meta-forecast-types/RESULTS.md; pooled crypto n=603
# origins, BTC/ETH 7d + GSPC 5d replication, strictly causal):
#   * Disciplined elicitation + climatology anchoring does NOT reliably
#     beat a naive normal fit on log score — the score Metaculus pays:
#     DISC - NAIVE = +0.018, 95% CI [-0.100, +0.138] (includes 0), and the
#     sign flips per asset (BTC +, ETH -, GSPC -).
#   * It DOES beat raw climatology on log score:
#     DISC - CLIM = +0.137, 95% CI [+0.009, +0.277], and earns a small
#     significant CRPS gain over naive (+0.00079 [+0.00030, +0.00126]).
#   * All arms are near-calibrated (PIT means 0.486-0.497; 90% coverage
#     0.88-0.92). No elicitation scheme manufactured an edge; numeric
#     questions will be won on genuine distributional information.
# Default adopted from that evidence — stated honestly:
#   elicitation = disciplined percentiles ANCHORED to climatology, with
#   naive-normal density smoothing for log-score contexts.
# Rationale: anchoring is the one step with a significant log-score win
# (over raw climatology); smoothing toward a fitted normal repairs the
# lumpy implied density that log score punishes in raw empirical
# percentiles. CAVEAT: the combined default's log-score edge over a plain
# naive normal fit is UNPROVEN (see first bullet) — the blend weights
# below (0.5 / 0.5) are transparent heuristic defaults, not tuned or
# validated optima. For data-backed questions, Stage 2 should produce
# the raw percentiles with cdf_module.disciplined_percentiles (spot +
# trailing returns) / cdf_module.climatology_percentiles directly; this
# function is the generic declared-percentiles path.

CLIMATOLOGY_ANCHOR_WEIGHT = 0.5  # declared <-> climatology blend
NAIVE_NORMAL_SMOOTHING_WEIGHT = 0.5  # anchored <-> fitted-normal blend


def anchor_percentiles_to_climatology(
    percentiles: dict[float, float],
    climatology_percentiles: dict[float, float],
    weight: float = CLIMATOLOGY_ANCHOR_WEIGHT,
) -> dict[float, float]:
    """Blend declared percentiles toward climatology percentiles.

    Same probability keys must be present in both dicts. The blend is
    geometric (log-space) when every value on both sides is > 0 — the
    price-like case the evaluation studied — and arithmetic otherwise.
    """
    items = sorted((float(p), float(v)) for p, v in percentiles.items())
    clim = [float(climatology_percentiles[p]) for p, _ in items]
    vals = np.array([v for _, v in items], dtype=float)
    clim_arr = np.array(clim, dtype=float)
    if np.all(vals > 0) and np.all(clim_arr > 0):
        blended = np.exp((1.0 - weight) * np.log(vals) + weight * np.log(clim_arr))
    else:
        blended = (1.0 - weight) * vals + weight * clim_arr
    return {p: float(v) for (p, _), v in zip(items, blended)}


def naive_normal_smoothed_percentiles(
    percentiles: dict[float, float],
    question: dict,
    weight: float = NAIVE_NORMAL_SMOOTHING_WEIGHT,
) -> dict[float, float]:
    """Blend percentiles toward a naive normal fit through them.

    Fits value ~= mu + sigma * Phi^-1(p) by least squares over the
    declared points (in log-value space when the question axis is
    log-scaled, i.e. scaling.zero_point is set, and all values > 0;
    otherwise in value space), evaluates the fit at the same declared
    probabilities, and blends. This smooths the implied density — the
    property that made the naive fit a strong log-score baseline — while
    keeping half of the declared (anchored) shape.
    """
    items = sorted((float(p), float(v)) for p, v in percentiles.items())
    if len(items) < 3:
        return dict(items)  # a normal fit needs >= 3 points to mean anything
    probs = np.array([p for p, _ in items], dtype=float)
    vals = np.array([v for _, v in items], dtype=float)
    scaling = question.get("scaling") or {}
    log_space = scaling.get("zero_point") is not None and np.all(vals > 0)
    x = np.log(vals) if log_space else vals
    z = _scipy_stats.norm.ppf(probs)
    design = np.vstack([np.ones_like(z), z]).T
    mu, sigma = np.linalg.lstsq(design, x, rcond=None)[0]
    if sigma <= 0:
        return dict(items)
    fitted = mu + sigma * z
    if log_space:
        fitted = np.exp(fitted)
    blended = (1.0 - weight) * vals + weight * fitted
    return {p: float(v) for (p, _), v in zip(items, blended)}


def default_numeric_percentiles(
    percentiles: dict[float, float],
    question: dict,
    climatology_percentiles: Optional[dict[float, float]] = None,
) -> dict[float, float]:
    """The scaffold's DEFAULT numeric strategy for declared percentiles.

    Steps: (1) if climatology percentiles are supplied, anchor the
    declared percentiles to them; (2) naive-normal density smoothing
    (the log-score default). Then enforce non-decreasing, strictly
    increasing values and clip into the question's scaling range so the
    result is always a valid input to percentiles_to_cdf.

    When no climatology is supplied the anchoring step is a no-op and
    the declared percentiles themselves are the anchor — production
    Stage-2 callers with trailing data should pass climatology (or call
    cdf_module.disciplined_percentiles upstream). See the evidence block
    above for what this default can and cannot claim.
    """
    out = dict(percentiles)
    if climatology_percentiles is not None:
        out = anchor_percentiles_to_climatology(out, climatology_percentiles)
    out = naive_normal_smoothed_percentiles(out, question)
    items = sorted((float(p), float(v)) for p, v in out.items())
    scaling = question.get("scaling") or {}
    lo = scaling.get("range_min")
    hi = scaling.get("range_max")
    vals = np.array([v for _, v in items], dtype=float)
    if lo is not None and hi is not None:
        vals = np.clip(vals, float(lo), float(hi))
    vals = np.maximum.accumulate(vals)
    for i in range(1, len(vals)):
        if vals[i] <= vals[i - 1]:
            vals[i] = vals[i - 1] * (1.0 + 1e-9) if vals[i - 1] != 0 else vals[i - 1] + 1e-9
    return {p: float(v) for (p, _), v in zip(items, vals)}


def build_forecast_payload(question: dict, forecast: Any) -> dict:
    """Build one forecast object for POST /api/questions/forecast/.

    forecast shapes, by question type:
      binary:          float — P(yes)
      multiple_choice: list[float] aligned with question["options"],
                       or dict {option_label: prob}
      numeric/discrete/date: dict {percentile_prob: value}
                       (converted to continuous_cdf here)
    """
    qtype = question["type"]
    payload: dict[str, Any] = {
        "question": int(question["id"]),
        "source": "api",
        "probability_yes": None,
        "probability_yes_per_category": None,
        "continuous_cdf": None,
    }
    if qtype == "binary":
        payload["probability_yes"] = clip_prob(float(forecast))
    elif qtype == "multiple_choice":
        options = list(question["options"])
        if isinstance(forecast, dict):
            probs = [float(forecast[o]) for o in options]
        else:
            probs = [float(p) for p in forecast]
        if len(probs) != len(options):
            raise ValueError(
                f"multiple_choice: {len(probs)} probabilities for "
                f"{len(options)} options"
            )
        normalized = normalize_probabilities(probs)
        payload["probability_yes_per_category"] = {
            option: p for option, p in zip(options, normalized)
        }
    elif qtype in ("numeric", "discrete", "date"):
        payload["continuous_cdf"] = percentiles_to_cdf(dict(forecast), question)
    else:
        raise ValueError(f"unsupported question type: {qtype!r}")
    return payload


def build_group_forecast_payloads(
    subquestions: list[dict],
    forecasts_by_id: dict,
    kind: str | None = None,
) -> list[dict]:
    """Build + jointly validate payloads for one group of subquestions.

    `forecasts_by_id` maps subquestion id -> a forecast in the shape
    build_forecast_payload expects for that subquestion's type. Each
    subquestion gets its OWN ordinary payload (forecasts submit per
    subquestion independently — there is no joint payload object), but
    NO payload is returned unless the whole set passes both the
    per-payload validator and the joint-consistency validator
    (group_module.validate_group_consistency: partition sums, ladder
    monotonicity). Raises ValueError listing every problem otherwise.
    Partial coverage is allowed (Metaculus encourages, but does not
    require, forecasting every subquestion).
    """
    payloads: list[dict] = []
    problems: list[str] = []
    subs_by_id = {int(q["id"]): q for q in subquestions}
    for qid, forecast in forecasts_by_id.items():
        question = subs_by_id.get(int(qid))
        if question is None:
            problems.append(f"forecast for unknown group subquestion id {qid!r}")
            continue
        payload = build_forecast_payload(question, forecast)
        problems.extend(
            f"subquestion {qid}: {p}"
            for p in validate_forecast_payload(question, payload)
        )
        payloads.append(payload)
    problems.extend(validate_group_consistency(subquestions, payloads, kind=kind))
    if problems:
        raise ValueError("group forecast failed validation: " + "; ".join(problems))
    return payloads


def validate_forecast_payload(question: dict, payload: dict) -> list[str]:
    """Return a list of problems with a payload (empty list = valid)."""
    problems: list[str] = []
    qtype = question["type"]
    if payload.get("question") != int(question["id"]):
        problems.append("payload question id does not match question id")
    set_fields = [
        k
        for k in ("probability_yes", "probability_yes_per_category", "continuous_cdf")
        if payload.get(k) is not None
    ]
    if len(set_fields) != 1:
        problems.append(f"exactly one forecast field must be set, got {set_fields}")
        return problems
    if qtype == "binary":
        p = payload["probability_yes"]
        if not (PROB_MIN <= p <= PROB_MAX):
            problems.append(f"binary probability {p} outside [{PROB_MIN}, {PROB_MAX}]")
    elif qtype == "multiple_choice":
        per_cat = payload["probability_yes_per_category"]
        options = list(question["options"])
        if set(per_cat) != set(options):
            problems.append("probability_yes_per_category keys != question options")
        total = sum(per_cat.values())
        if abs(total - 1.0) > 1e-9:
            problems.append(f"multiple_choice probabilities sum to {total}, not 1")
        for option, p in per_cat.items():
            if not (0.0 < p < 1.0):
                problems.append(f"option {option!r} probability {p} not in (0, 1)")
    elif qtype in ("numeric", "discrete", "date"):
        cdf = payload["continuous_cdf"]
        expected = cdf_size_for_question(question)
        if len(cdf) != expected:
            problems.append(f"CDF has {len(cdf)} points, expected {expected}")
        if any(v < 0.0 or v > 1.0 for v in cdf):
            problems.append("CDF values outside [0, 1]")
        if any(b < a for a, b in zip(cdf, cdf[1:])):
            problems.append("CDF is not non-decreasing")
        steps = [b - a for a, b in zip(cdf, cdf[1:])]
        if steps and max(steps) > MAX_CDF_STEP_WARN:
            # Per the MAX_CDF_STEP_WARN note above, this is a warning, not
            # a validation failure: the server is the judge. (Previously
            # this was returned as a problem, and bot.run() aborted the
            # whole batch on any concentrated-but-legitimate forecast.)
            warnings.warn(
                f"largest CDF step {max(steps):.3f} > {MAX_CDF_STEP_WARN} "
                "(Metaculus may reject overly concentrated CDFs; "
                "consider distributing mass)"
            )
    return problems


def validate_submission_batch(payloads: list[dict]) -> None:
    if not isinstance(payloads, list) or not payloads:
        raise ValueError("submission body must be a non-empty JSON list")
    for p in payloads:
        if "question" not in p:
            raise ValueError(f"payload missing 'question': {p!r}")


def already_forecasted(question: dict) -> bool:
    """True if the bot already has a forecast on this question."""
    try:
        values = question["my_forecasts"]["latest"]["forecast_values"]
    except (KeyError, TypeError):
        return False
    return values is not None


def blind_question_view(question: dict) -> dict:
    """Price-blind view of a question for the fleet's forecaster stage.

    The fleet's playbook (docs/FLEET_NORTH_STAR_PROMPT.md) requires Stage-2
    forecasters to be blind to market/community prices. A raw Metaculus
    question payload contains `aggregations` (the community prediction) and
    `my_forecasts`; this view strips everything except what a forecaster may
    legitimately see: the question text, criteria, options and scaling.
    """
    keep = (
        "id",
        "title",
        "type",
        "description",
        "resolution_criteria",
        "fine_print",
        "options",
        "scaling",
        "open_upper_bound",
        "open_lower_bound",
        "unit",
        "scheduled_close_time",
        "scheduled_resolve_time",
        # Group subquestions only: the subquestion's group option label,
        # its position in the group, its parent post and its siblings —
        # the forecaster needs the label to know WHICH subquestion this
        # is (forecasting-tools does the same via group_question_option).
        "label",
        "group_variable",
        "group_rank",
        "group_post_id",
        "group_question_ids",
    )
    return {k: question[k] for k in keep if k in question}


def dumps_payloads(payloads: list[dict]) -> str:
    return json.dumps(payloads, indent=2)
