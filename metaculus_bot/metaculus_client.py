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
import math
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
    # A bool is not a probability (SF-BL-10349): float(True)=1.0 /
    # float(False)=0.0 would clip to near-certainty (0.99 / 0.01).
    if isinstance(p, bool):
        raise ValueError(
            f"clip_prob: bool probability {p!r} is not a number "
            "(a bool is not a probability)"
        )
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
    # Scaling-bounds screen (SF-BL-10432): float() converts NaN /
    # +/-inf without complaint, so a present-but-non-finite bound
    # (the TASK repro: range_min=NaN) flowed into cdf_module and
    # built a fully NaN CDF that build_forecast_payload would then
    # submit or score as if it were a forecast; an inverted or equal
    # range likewise returned a plausible-looking but wrong CDF
    # (descending / constant grid). Validate every scaling value
    # this wrapper consumes, by name, BEFORE any CDF construction:
    # range_min, range_max and (when not None) zero_point must each
    # be finite, and the bounds must satisfy range_min < range_max,
    # else raise a named ValueError. A bool is not a number (fleet
    # convention, cf. clip_prob SF-BL-10349), so a bool bound -- or a
    # value float() cannot convert at all -- is named as not finite
    # the same way. Missing range_min / range_max keys are NOT
    # screened here: the indexing below raises the legacy KeyError,
    # which is SF-BL-10391's distinct missing-scaling case. Well-
    # formed scaling converts to the identical floats the unscreened
    # float() calls produced, so well-formed CDFs are unchanged.
    _range_min_raw = scaling["range_min"] if "range_min" in scaling else None
    _range_max_raw = scaling["range_max"] if "range_max" in scaling else None
    _zero_point_raw = scaling.get("zero_point")
    _screened_scaling: dict[str, Any] = {}
    for _bound_name, _bound_raw in (
        ("range_min", _range_min_raw),
        ("range_max", _range_max_raw),
        ("zero_point", _zero_point_raw),
    ):
        if _bound_name == "range_min" and "range_min" not in scaling:
            continue
        if _bound_name == "range_max" and "range_max" not in scaling:
            continue
        if _bound_name == "zero_point" and _bound_raw is None:
            _screened_scaling[_bound_name] = None
            continue
        _bound_val = None
        if not isinstance(_bound_raw, bool) and not isinstance(
            _bound_raw, np.bool_
        ):
            try:
                _bound_val = float(_bound_raw)
            except (TypeError, ValueError):
                _bound_val = None
        if _bound_val is None or not math.isfinite(_bound_val):
            raise ValueError(
                "percentiles_to_cdf: scaling "
                f"{_bound_name} {_bound_raw!r} is not finite "
                "(scaling range_min/range_max/zero_point must be "
                "finite with range_min < range_max, got "
                f"range_min={_range_min_raw!r}, "
                f"range_max={_range_max_raw!r}, "
                f"zero_point={_zero_point_raw!r})"
            )
        _screened_scaling[_bound_name] = _bound_val
    if (
        "range_min" in _screened_scaling
        and "range_max" in _screened_scaling
        and not _screened_scaling["range_min"] < _screened_scaling["range_max"]
    ):
        raise ValueError(
            "percentiles_to_cdf: scaling bounds must be finite "
            "with range_min < range_max, got "
            f"range_min={_range_min_raw!r}, "
            f"range_max={_range_max_raw!r}"
        )
    cdf = _cdf_percentiles_to_cdf(
        dict(percentiles),
        _screened_scaling["range_min"]
        if "range_min" in _screened_scaling
        else float(scaling["range_min"]),
        _screened_scaling["range_max"]
        if "range_max" in _screened_scaling
        else float(scaling["range_max"]),
        _screened_scaling.get("zero_point"),
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
    # The blend weight must be a finite interpolation weight in [0, 1]:
    # NaN silently poisoned every output value and weight > 1 (or < 0)
    # extrapolated past both inputs (SF-BL-10345), so reject both by
    # name before any blending arithmetic runs.
    try:
        _weight_is_finite = math.isfinite(weight)
    except TypeError:
        raise ValueError(
            "anchor_percentiles_to_climatology: weight must be finite "
            f"and in [0, 1], got {weight!r}"
        ) from None
    if not _weight_is_finite or not 0.0 <= weight <= 1.0:
        raise ValueError(
            "anchor_percentiles_to_climatology: weight must be finite "
            f"and in [0, 1], got {weight!r}"
        )
    # Every declared AND climatology percentile value must be finite
    # before blending (SF-BL-10390): a single NaN declared value
    # ({0.25:5, 0.5:NaN, 0.75:15}) previously returned a NaN anchored
    # value at that probability with no screen, and a non-finite
    # climatology value poisoned the blend the same way -- in the
    # geometric branch np.log(NaN/inf) propagates silently, in the
    # arithmetic branch the blend itself does. This is the values
    # screen, distinct from the SF-BL-10345 blend-weight screen above
    # and the SF-BL-10346 missing-climatology-key case. All entries of
    # both dicts are screened, before any blending arithmetic runs.
    items = []
    for _p_raw, _v_raw in percentiles.items():
        try:
            _p, _v = float(_p_raw), float(_v_raw)
        except (TypeError, ValueError):
            raise ValueError(
                "anchor_percentiles_to_climatology: declared percentile "
                f"value {_v_raw!r} at probability {_p_raw!r} is not finite"
            ) from None
        if not math.isfinite(_v):
            raise ValueError(
                "anchor_percentiles_to_climatology: declared percentile "
                f"value {_v_raw!r} at probability {_p_raw!r} is not finite"
            )
        items.append((_p, _v))
    items.sort()
    for _cp_raw, _cv_raw in climatology_percentiles.items():
        try:
            _cv = float(_cv_raw)
        except (TypeError, ValueError):
            raise ValueError(
                "anchor_percentiles_to_climatology: climatology percentile "
                f"value {_cv_raw!r} at probability {_cp_raw!r} is not finite"
            ) from None
        if not math.isfinite(_cv):
            raise ValueError(
                "anchor_percentiles_to_climatology: climatology percentile "
                f"value {_cv_raw!r} at probability {_cp_raw!r} is not finite"
            )
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
    # Input screen (SF-BL-10348): norm.ppf(p) is infinite at p in {0, 1}
    # and NaN for p outside [0, 1] or NaN, which made the least-squares
    # fit below die with a bare numpy LinAlgError ("SVD did not
    # converge") that never names the offending percentile; and a
    # non-finite declared value silently poisoned the fit into an
    # all-NaN result dict. Screen both before any fit (and before the
    # <3-point early return, so short inputs are screened too).
    for p, v in items:
        if not math.isfinite(p) or not 0.0 < p < 1.0:
            raise ValueError(
                f"naive_normal_smoothed_percentiles: percentile probability "
                f"{p!r} is not in the open interval (0, 1)"
            )
        if not math.isfinite(v):
            raise ValueError(
                f"naive_normal_smoothed_percentiles: declared value {v!r} "
                f"at percentile {p!r} is not finite"
            )
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
    # Declared-value screen BEFORE any smoothing (SF-BL-10389): one
    # non-finite declared value (NaN or +/-inf) previously propagated
    # through the anchoring blend and the naive-normal fit and spread
    # across every output percentile -- the clip / monotonicity repair
    # below does not detect NaN -- so the default path emitted a fully
    # corrupt percentile set ({0.25:5, 0.5:NaN, 0.75:15} on range 0-20
    # returned all-NaN). Downstream helpers screen their own inputs,
    # but this entry point must reject corrupt declared values itself,
    # by name, before step (1) runs. Only the declared values are
    # screened here; climatology values are anchor_'s screen and
    # probability keys are the smoothing helper's screen.
    for _p_raw, _v_raw in percentiles.items():
        try:
            _v = float(_v_raw)
        except (TypeError, ValueError):
            raise ValueError(
                "default_numeric_percentiles: declared percentile "
                f"value {_v_raw!r} at probability {_p_raw!r} is not finite"
            ) from None
        if not math.isfinite(_v):
            raise ValueError(
                "default_numeric_percentiles: declared percentile "
                f"value {_v_raw!r} at probability {_p_raw!r} is not finite"
            )
    # Scaling-bounds screen BEFORE any anchoring / smoothing / clipping
    # (SF-BL-10476): the final clip below trusts scaling completely, so
    # one corrupt scaling object converted a sound declared panel into
    # a degenerate or NaN percentile dict -- an inverted range
    # (range_min=100, range_max=0) collapsed every value to ~0 with a
    # 1e-9 staircase (np.clip into an inverted interval, then the
    # monotonicity repair), and range_max=NaN returned an all-NaN dict
    # (np.clip and the repair never validate the bounds they clip
    # into; +/-inf bounds silently passed through as no-op clips).
    # Validate the bounds this function clips into, by name, before
    # any step runs: every PRESENT bound must be finite (a bool is
    # not a number -- fleet convention, cf. clip_prob SF-BL-10349),
    # and when both bounds are present they must satisfy
    # range_min < range_max. Missing (absent / None) bounds keep the
    # legacy no-clip behaviour below -- the missing-bounds case is
    # distinct from the present-but-corrupt case fixed here.
    # Well-formed scaling validates and is then used exactly as
    # before, so well-formed outputs are unchanged.
    scaling = question.get("scaling") or {}
    _lo_raw = scaling.get("range_min")
    _hi_raw = scaling.get("range_max")
    _lo = _hi = None
    for _bound_name, _bound_raw in (
        ("range_min", _lo_raw),
        ("range_max", _hi_raw),
    ):
        if _bound_raw is None:
            continue
        _bound = None
        if not isinstance(_bound_raw, bool) and not isinstance(
            _bound_raw, np.bool_
        ):
            try:
                _bound = float(_bound_raw)
            except (TypeError, ValueError):
                _bound = None
        if _bound is None or not math.isfinite(_bound):
            raise ValueError(
                "default_numeric_percentiles: scaling "
                f"{_bound_name} {_bound_raw!r} is not finite "
                "(scaling bounds must be finite with "
                "range_min < range_max, got "
                f"range_min={_lo_raw!r}, range_max={_hi_raw!r})"
            )
        if _bound_name == "range_min":
            _lo = _bound
        else:
            _hi = _bound
    if _lo is not None and _hi is not None and not _lo < _hi:
        raise ValueError(
            "default_numeric_percentiles: scaling bounds must be "
            "finite with range_min < range_max, got "
            f"range_min={_lo_raw!r}, range_max={_hi_raw!r}"
        )
    out = dict(percentiles)
    if climatology_percentiles is not None:
        out = anchor_percentiles_to_climatology(out, climatology_percentiles)
    out = naive_normal_smoothed_percentiles(out, question)
    items = sorted((float(p), float(v)) for p, v in out.items())
    vals = np.array([v for _, v in items], dtype=float)
    if _lo is not None and _hi is not None:
        vals = np.clip(vals, _lo, _hi)
    vals = np.maximum.accumulate(vals)
    # Strict-increase nudge (SF-BL-10771): the step must always move UP.
    # The pre-fix step vals[i-1] * (1 + 1e-9) moves NEGATIVE values
    # DOWN (tied -10 -> -10.00000001, a decreasing output that
    # cdf_module._validate_percentiles then rejects) and moves values
    # tied at range_max ABOVE the range (tied 100 -> 100.0000001).
    # Sign-aware step: a positive predecessor keeps the exact pre-fix
    # step (all-positive interior outputs are bit-identical), a
    # negative predecessor steps toward zero / up by the same relative
    # magnitude (-10 -> -10 * (1 - 1e-9) = -9.99999999), and zero keeps
    # the pre-fix +1e-9 step. nextafter is a last-resort guard for
    # magnitudes so small the relative step rounds back to the same
    # float. Cap overflow created by this step is repaired below --
    # the step itself never clips a valid value out of range.
    _pre_nudge = vals.copy()
    for i in range(1, len(vals)):
        if vals[i] <= vals[i - 1]:
            _prev = vals[i - 1]
            if _prev > 0:
                _cand = _prev * (1.0 + 1e-9)
            elif _prev < 0:
                _cand = _prev * (1.0 - 1e-9)
            else:
                _cand = _prev + 1e-9
            if not _cand > _prev:
                _cand = np.nextafter(_prev, np.inf)
            vals[i] = _cand
    # Range-cap repair (SF-BL-10771): a nudge that pushed a value which
    # was at (or below) range_max to above range_max is pulled back to
    # range_max, and the tied predecessors are stepped DOWN from the
    # cap by the symmetric sign-aware step (positive: * (1 - 1e-9);
    # negative cap, e.g. hi=-10: * (1 + 1e-9), i.e. more negative;
    # zero cap: -1e-9), so the panel stays strictly increasing while
    # its top value remains exactly at range_max. Values that were
    # already above range_max before the nudge (only possible in the
    # legacy single-bound / no-clip case) are left untouched. If the
    # downward staircase would cross range_min (a range too narrow for
    # a 1e-9-relative staircase, e.g. [999999, 1000000] tied at the
    # cap), the panel is redistributed evenly over [range_min,
    # range_max] -- strictly increasing and in range whenever the
    # bounds admit n distinct floats at all.
    if _hi is not None and len(vals):
        _overflow = (vals > _hi) & (_pre_nudge <= _hi)
        if np.any(_overflow):
            vals[_overflow] = _hi
            for i in range(len(vals) - 2, -1, -1):
                if vals[i] >= vals[i + 1]:
                    _nxt = vals[i + 1]
                    if _nxt > 0:
                        _cand = _nxt * (1.0 - 1e-9)
                    elif _nxt < 0:
                        _cand = _nxt * (1.0 + 1e-9)
                    else:
                        _cand = _nxt - 1e-9
                    if not _cand < _nxt:
                        _cand = np.nextafter(_nxt, -np.inf)
                    vals[i] = _cand
            if _lo is not None and np.any(vals < _lo):
                vals = np.linspace(_lo, _hi, len(vals))
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
        # A bool is not a probability (SF-BL-10349): float(True)=1.0
        # flowed into clip_prob unscreened and returned
        # probability_yes 0.99. Reject by name before float().
        if isinstance(forecast, bool):
            raise ValueError(
                f"build_forecast_payload: binary forecast {forecast!r} "
                "is a bool, not a probability (a bool is not a number)"
            )
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
    build_forecast_payload expects for that subquestion's type. Its
    keys must be int subquestion ids exactly: a non-int key (float,
    bool, str, ...) is rejected with a named problem, never coerced
    with int() (SF-BL-10428). Each
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
        # Forecast keys must be int subquestion ids exactly -- never
        # coerce them with int(). int() silently truncates a float
        # key (1.9 -> 1), maps a bool key (True -> 1; bool is an int
        # subclass) and parses a str/bytes key ('1' / b'1' -> 1), so
        # a corrupt caller key attached its forecast to a subquestion
        # it never named and the set could still pass joint validation
        # (SF-BL-10428). Reject every non-int key (float/bool/str/...)
        # with a named problem instead; bool is screened first because
        # isinstance(True, int) is True (fleet convention: a bool is
        # not a number / id).
        if isinstance(qid, bool) or not isinstance(qid, int):
            problems.append(
                f"forecast key {qid!r} is a non-int key "
                f"(type {type(qid).__name__}): forecast keys must be "
                "int subquestion ids, not float/bool/str "
                "(non-integer keys are rejected, not coerced)"
            )
            continue
        question = subs_by_id.get(qid)
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
    # Payload question id must be a plain int exactly -- never compare
    # it with a bare != (SF-BL-10469). In Python 42.0 == 42 and
    # True == 1 / False == 0, so a bare != blessed float and bool ids
    # produced upstream at this last validation stage (the gate that
    # build_group_forecast_payloads and the submit paths rely on).
    # Screen the type first: bool is checked explicitly because
    # isinstance(True, int) is True (fleet convention: a bool is not
    # a number / id, cf. validate_submission_batch SF-BL-10427 and
    # the forecast-key screen SF-BL-10428). Any non-int id (float --
    # integral, non-integral, NaN or inf -- str, bool, None, bytes
    # or a container) is flagged as a named non-int problem, never
    # coerced; only a plain int falls through to the identity check.
    _payload_qid = payload.get("question")
    if isinstance(_payload_qid, bool) or not isinstance(_payload_qid, int):
        problems.append(
            f"payload question id {_payload_qid!r} is a non-int question id "
            f"(type {type(_payload_qid).__name__}): question id must be "
            "a plain int, not float/bool/str "
            "(non-integer ids are rejected, not coerced)"
        )
    elif _payload_qid != int(question["id"]):
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
        # Non-numeric values (str, bytes, containers, ...) crash the
        # range comparison below with a bare TypeError ('<=' between
        # float and str) although this validator's contract is to
        # RETURN a list of problems (SF-BL-10388). A bool is likewise
        # not a number (fleet convention, cf.
        # distribution_validity._is_number and the CDF branch below),
        # so name every non-numeric value instead of comparing it.
        # Numeric values (int/float, incl. NaN/inf) fall through to
        # the original range check, verdicts unchanged.
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            problems.append(
                f"binary probability {p!r} is non-numeric "
                f"(type {type(p).__name__}: probability_yes must be a number)"
            )
        elif not (PROB_MIN <= p <= PROB_MAX):
            problems.append(f"binary probability {p} outside [{PROB_MIN}, {PROB_MAX}]")
    elif qtype == "multiple_choice":
        per_cat = payload["probability_yes_per_category"]
        options = list(question["options"])
        if not isinstance(per_cat, dict):
            # A non-dict container (list, str, ...) has no .values() /
            # .items() and crashed with AttributeError (SF-BL-10388);
            # name it instead. Dict payloads validate exactly as before.
            problems.append(
                "probability_yes_per_category is non-numeric / not a "
                f"mapping (type {type(per_cat).__name__}: expected a dict "
                "of option -> number)"
            )
        else:
            if set(per_cat) != set(options):
                problems.append("probability_yes_per_category keys != question options")
            # Non-numeric members (str, bytes, None, bool, ...) crash
            # sum() / the range comparisons below with a bare TypeError
            # (unsupported operand for +) (SF-BL-10388); a bool is not a
            # number (fleet convention, as in the binary/CDF branches).
            # Name them and skip the numeric checks — a sum over corrupt
            # members would be meaningless. Numeric members (incl.
            # NaN/inf) fall through to the original checks unchanged.
            non_numeric_options = [
                option
                for option, v in per_cat.items()
                if isinstance(v, bool) or not isinstance(v, (int, float))
            ]
            if non_numeric_options:
                problems.append(
                    "probability_yes_per_category contains non-numeric "
                    f"values for options {non_numeric_options} "
                    "(each per-category probability must be a number)"
                )
            else:
                total = sum(per_cat.values())
                if abs(total - 1.0) > 1e-9:
                    problems.append(f"multiple_choice probabilities sum to {total}, not 1")
                for option, p in per_cat.items():
                    if not (0.0 < p < 1.0):
                        problems.append(f"option {option!r} probability {p} not in (0, 1)")
    elif qtype in ("numeric", "discrete", "date"):
        cdf = payload["continuous_cdf"]
        expected = cdf_size_for_question(question)
        if isinstance(cdf, (str, bytes, dict)) or not hasattr(cdf, "__len__") \
                or not hasattr(cdf, "__iter__") or not hasattr(cdf, "__getitem__"):
            # A non-sequence CDF (scalar, str/bytes, dict, ...) has no
            # numeric members to screen: len() on a scalar raised a bare
            # TypeError (SF-BL-10388). Name it and screen an empty CDF
            # below, which adds only the length problem. List/tuple (and
            # other sized, sliceable iterables of numbers — a set is not
            # sliceable and crashed zip(cdf, cdf[1:])) validate as before.
            problems.append(
                "continuous_cdf is non-numeric / not a sequence of "
                f"numbers (type {type(cdf).__name__}: expected a list "
                "of CDF values)"
            )
            cdf = []
        if len(cdf) != expected:
            problems.append(f"CDF has {len(cdf)} points, expected {expected}")
        # Bool members defeat every check below: bool is a subclass of
        # int, so True passes 0<=v<=1 as 1 and False as 0, and the
        # monotonicity comparisons treat them as 1/0 — an all-bool CDF
        # previously passed as valid (SF-BL-10387). A bool is not a
        # number (fleet convention, cf. distribution_validity._is_number
        # and group_module._screen_probability_yes), so reject any bool
        # member explicitly, by name, first.
        bool_indices = [i for i, v in enumerate(cdf) if isinstance(v, bool)]
        if bool_indices:
            problems.append(
                "CDF contains boolean values at indices "
                f"{bool_indices} (a bool is not a number)"
            )
        # Non-numeric members (str, None, ...) are likewise not numbers:
        # name them instead of crashing math.isfinite / the comparisons
        # below (group_module screens them the same way).
        non_numeric = [
            i for i, v in enumerate(cdf)
            if not isinstance(v, bool) and not isinstance(v, (int, float))
        ]
        if non_numeric:
            problems.append(
                "CDF contains non-numeric values at indices "
                f"{non_numeric} (a CDF member must be a number)"
            )
        # Non-finite values defeat every finite check below (all comparisons
        # against NaN are False, so an all-NaN CDF previously passed as
        # valid — SF-BL-10315), so reject them explicitly, by name, first.
        # Bool and non-numeric members are excluded here — they are named
        # above, and math.isfinite would crash on the non-numeric ones.
        non_finite = [
            i for i, v in enumerate(cdf)
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and not math.isfinite(v)
        ]
        if non_finite:
            problems.append(
                "CDF contains non-finite values (NaN or +/-inf) at indices "
                f"{non_finite}"
            )
        if not (bool_indices or non_numeric or non_finite):
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


FORECAST_FIELDS = (
    "probability_yes",
    "probability_yes_per_category",
    "continuous_cdf",
)


def validate_submission_batch(payloads: list[dict]) -> None:
    """Last gate before POST /api/questions/forecast/ (submit_forecasts).

    Batch-level invariants only -- raises a named ValueError on:
      * a body that is not a non-empty list,
      * a payload that is not a dict (SF-BL-10427: a bare string such
        as 'question' previously passed because 'question' in p is a
        substring test on a str payload),
      * a payload missing the 'question' key,
      * a payload whose 'question' id is not an int — None, str,
        float, bool, list/dict and every other non-int type raise
        (SF-BL-10427: {'question': None} previously passed; bool is
        rejected explicitly because bool is a subclass of int but is
        not a question id, matching the fleet convention that a bool
        is not a number),
      * a payload whose 'question' id is NaN, non-finite (+/-inf),
        a bool, or a non-integer float — each named explicitly in
        the error (SF-BL-10547: a NaN id is unidentifiable and, since
        NaN != NaN, two distinct NaN objects previously evaded the
        dict-based duplicate scan entirely),
      * duplicate question ids anywhere in the batch,
      * a payload setting zero forecast fields, or more than one
        forecast field (exactly one of FORECAST_FIELDS must be
        non-None; a field present but None does not count as set,
        matching build_forecast_payload, which nulls the other two).

    Per-type shape checks are deliberately NOT done here: probability
    ranges, multiple-choice sums / option keys, and CDF length /
    range / monotonicity / finiteness need the question object and
    are deferred to validate_forecast_payload(question, payload).
    A pass here therefore does not certify shape, and callers must
    not treat it as a substitute for validate_forecast_payload
    upstream (bot.py / build_group_forecast_payloads already run it).
    """
    if not isinstance(payloads, list) or not payloads:
        raise ValueError("submission body must be a non-empty JSON list")
    # Pass 1 -- structure + duplicate question ids. Duplicates are
    # scanned over the whole batch before any field-count check, so
    # a duplicated id is reported as a duplicate even when one of
    # its payloads also has a field-count problem.
    seen: dict = {}
    for idx, p in enumerate(payloads):
        if not isinstance(p, dict):
            raise ValueError(
                f"validate_submission_batch: payload at index {idx} "
                f"is not a dict: {p!r}"
            )
        if "question" not in p:
            raise ValueError(f"payload missing 'question': {p!r}")
        qid = p["question"]
        # SF-BL-10547: NaN / non-finite ids get their OWN named error,
        # before the generic non-int check and before the duplicate
        # scan. A NaN id is unidentifiable (NaN != NaN — not even to
        # itself), so two payloads carrying two DISTINCT NaN objects
        # previously passed this gate: the float was never rejected
        # specifically as NaN and the dict duplicate scan below could
        # never fire on it. math.isnan/isinf also catch Decimal NaN /
        # Infinity and numpy floating NaN/inf (np.float64 subclasses
        # float; np.float32 does not, hence the explicit np check).
        # Non-numeric types raise TypeError from math.* and fall
        # through to the generic check; signaling Decimals raise
        # ArithmeticError on conversion and likewise fall through.
        if not isinstance(qid, bool):
            _is_nan = _is_nonfinite = False
            try:
                _is_nan = bool(math.isnan(qid))  # type: ignore[arg-type]
                _is_nonfinite = _is_nan or bool(math.isinf(qid))  # type: ignore[arg-type]
            except (TypeError, ValueError, ArithmeticError, OverflowError):
                if isinstance(qid, np.floating):
                    _is_nan = bool(np.isnan(qid))
                    _is_nonfinite = _is_nan or bool(np.isinf(qid))
            if _is_nan:
                raise ValueError(
                    f"validate_submission_batch: payload at index {idx} "
                    f"has NaN question id {qid!r}: a NaN question id is "
                    f"unidentifiable (NaN != NaN) and can never be a "
                    f"valid question id"
                )
            if _is_nonfinite:
                raise ValueError(
                    f"validate_submission_batch: payload at index {idx} "
                    f"has non-finite question id {qid!r}: question id "
                    f"must be a finite int"
                )
            if isinstance(qid, float) and not qid.is_integer():
                raise ValueError(
                    f"validate_submission_batch: payload at index {idx} "
                    f"has non-integer float question id {qid!r}: "
                    f"question id must be an int, not a float"
                )
        # SF-BL-10427: the question id must be a genuine int. None,
        # str (including numeric strings), float (including integral
        # floats), bool, and unhashable containers all previously
        # passed or were misreported; reject every non-int here,
        # before the duplicate scan, with a named error. Bool is
        # checked first: isinstance(True, int) is True in Python.
        if isinstance(qid, bool) or not isinstance(qid, int):
            raise ValueError(
                f"validate_submission_batch: payload at index {idx} "
                f"has non-int question id {qid!r} "
                f"(type {type(qid).__name__}): question id must be "
                f"an int, not None or another type"
            )
        try:
            hash(qid)
        except TypeError:
            raise ValueError(
                f"validate_submission_batch: unhashable question id "
                f"at index {idx}: {qid!r}"
            ) from None
        # SF-BL-10547 (defense in depth): the duplicate scan keys on a
        # canonical key, never the raw id. Any NaN-like id maps to one
        # shared sentinel, so two distinct NaN objects WOULD collide
        # here even if the NaN gate above were ever relaxed — a raw
        # `qid in seen` dict lookup can never match NaN != NaN. (With
        # the gate in force, NaN ids raise above and never reach this
        # scan; genuine ints key as themselves.)
        dup_key: Any = qid
        try:
            if math.isnan(qid):  # ints are never NaN; keeps key honest
                dup_key = ("__nan_question_id__",)
        except (TypeError, ValueError, ArithmeticError, OverflowError):
            pass
        if dup_key in seen:
            raise ValueError(
                f"validate_submission_batch: duplicate question id "
                f"{qid!r} at indices {seen[dup_key]} and {idx}"
            )
        seen[dup_key] = idx
    # Pass 2 -- exactly one forecast field set per payload.
    for idx, p in enumerate(payloads):
        set_fields = [k for k in FORECAST_FIELDS if p.get(k) is not None]
        if len(set_fields) != 1:
            raise ValueError(
                f"validate_submission_batch: payload for question "
                f"{p['question']!r} at index {idx} must set exactly "
                f"one forecast field {FORECAST_FIELDS}, got "
                f"{len(set_fields)}: {set_fields}"
            )


def already_forecasted(question: dict) -> bool:
    """True if the bot already has a forecast on this question.

    SF-BL-10352: True only when the latest forecast carries at least
    one non-None value. The API can return an empty latest-forecast
    shell (``forecast_values == []``, or all-None such as ``[None]``)
    for a question the bot never actually forecast; the old
    ``values is not None`` check marked such questions done, so the
    bot skipped them forever. The test per value is ``is not None``,
    never truthiness: 0 / 0.0 is a real forecast.
    """
    try:
        values = question["my_forecasts"]["latest"]["forecast_values"]
    except (KeyError, TypeError):
        return False
    if values is None:
        return False
    try:
        return any(v is not None for v in values)
    except TypeError:
        # Non-iterable, non-None scalar: still evidences a forecast
        # (preserves the pre-fix behaviour for such payloads).
        return True


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
