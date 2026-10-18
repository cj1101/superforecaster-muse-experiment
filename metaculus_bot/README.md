# Superforecaster Fleet Bot — Metaculus (Fall 2026 FutureEval)

A thin scaffold that connects the fleet's production forecasting pipeline to
Metaculus's bot tournaments. It handles question intake, payload formatting,
and submission; the actual forecasting plugs in at two clearly marked points.

**Status (2026-10-17): scaffold complete, self-test passing, NOT live.**
Account registration and the participation form are handled separately by CJ.
No token exists yet; nothing has been submitted anywhere.

**Update (2026-10-17, Wave 3 meta-cdf-integrate):** the verified Wave 3
`cdf_module.py` is now integrated as the scaffold's **single** numeric CDF
implementation (see "Numeric forecasts" below), and the bot's default
numeric strategy is set from that module's evaluation. Both test suites
pass: `self_test.py` and `wave3/meta-forecast-types/test_module.py`
(16/16, including scaffold-equivalence). No submissions were made.

## Files

| File | Role |
|---|---|
| `cdf_module.py` | **Single numeric implementation:** grid geometry, percentiles→CDF, elicitation helpers (climatology / naive-normal / disciplined), CDF scoring. numpy + scipy. Integrated verbatim from `wave3/meta-forecast-types/cdf_module.py` — do not re-implement CDF logic elsewhere. |
| `metaculus_client.py` | Thin API client + pure payload builders/validators (offline-testable). Its `percentiles_to_cdf(percentiles, question)` is a thin adapter delegating to `cdf_module` (public signature unchanged); it also holds the default numeric strategy (`default_numeric_percentiles`). |
| `bot.py` | Orchestration: intake → forecast → judge → format → (dry-run \| submit). `judge_and_pool()` applies the default numeric strategy to numeric/discrete percentile dicts. |
| `group_module.py` | **Group questions (Wave 3):** unpacks `group_of_questions` posts into subquestions and validates joint consistency (partition sums, threshold-ladder monotonicity) before any payload is emitted. No server-side equivalent exists — see `wave3/meta-group-questions/RESULTS.md`. |
| `self_test.py` | Offline test: synthetic binary / multiple-choice / numeric (+log-scaled, +discrete) payloads, blind-view checks, and group cases (intake, partition, ladders, inconsistent-group rejection). Run: `python3 self_test.py`. |
| `.env.example` | Environment variable template. Never commit a filled `.env`. |

## Architecture

The fleet playbook (`~/workspace/superforecaster/docs/FLEET_NORTH_STAR_PROMPT.md`)
defines a four-stage topology; this bot maps onto it directly:

```
Stage 1  DISCOVERY            bot.py: run()
  Metaculus intake            GET /api/posts/?tournaments=<project>
                              -> open posts; GROUP posts are unpacked into
                              their subquestions (group_module.py; conditional
                              posts are still skipped)

Stage 2  FORECASTERS          bot.py: fleet_forecast()   << INTEGRATION POINT A
  price-blind                 Receives blind_question_view(question): title,
                              description, resolution criteria, fine print,
                              options, scaling — the community prediction
                              (`aggregations`) and `my_forecasts` are STRIPPED.
                              Production wiring: ./sf forecast (consume
                              production_p only), ./sf pool for multi-
                              forecaster log-odds pooling.

Stage 3  SUPERVISOR /         bot.py: judge_and_pool()   << INTEGRATION POINT B
  CALIBRATION JUDGE           Pooling + calibration discipline before
                              anything leaves: engine cap [0.01, 0.99],
                              ECE gates, extremization 1.00 by default.
                              For numeric/discrete percentile dicts it
                              also applies the DEFAULT NUMERIC STRATEGY
                              (climatology anchoring + naive-normal
                              smoothing; see "Numeric forecasts" below).
                              Metaculus scores with LOG scores (Peer Scores
                              are log-score differences), so calibration —
                              not bravado — is what pays.

Stage 4  EXECUTION            metaculus_client.py + cdf_module.py
                              build_forecast_payload() per question type
                              (CDF construction delegated to cdf_module,
                              the single numeric implementation),
                              validate_forecast_payload(), then
                              POST /api/questions/forecast/
                              RULE (resources notebook 38928, verified
                              2026-10-17): submit ONE forecast per question
                              in bot-only tournaments (Fall 2026, MiniBench).
                              Scoring counts the last forecast, but
                              re-forecasting scored questions is against
                              the tournament instruction — test new versions
                              in bot-testing-area (32977) instead. A private
                              comment is REQUIRED on every forecast; live
                              submits now attach one automatically.
                              See ../wave3/meta-rules-reconcile/RULES.md.
```

## API facts (verified 2026-10-17 from Metaculus's own public sources)

- Base URL: `https://www.metaculus.com/api`
- Auth header: `Authorization: Token <token>` (DRF Token auth, **not** Bearer).
- List questions: `GET /api/posts/?tournaments=<id|slug>&statuses=open&forecast_type=binary,multiple_choice,numeric,discrete&include_description=true` → `{"results": [...]}`; question objects ride inside posts (`post.question`); **question ID ≠ post ID**.
- Post detail: `GET /api/posts/<post_id>/`.
- Submit: `POST /api/questions/forecast/` with a JSON **list**:
  `[{"question": <qid>, "source": "api", "probability_yes": <float|null>, "probability_yes_per_category": <{option: prob}|null>, "continuous_cdf": <[floats]|null>}]` — exactly one forecast field set per question.
- Binary probabilities must lie in [0.01, 0.99] (matches the fleet engine cap).
- Multiple choice uses `probability_yes_per_category`, a dict keyed by option label (template bot format), probabilities summing to 1.
- Numeric uses `continuous_cdf`: 201 points (discrete: `scaling.inbound_outcome_count + 1`), evaluated at evenly spaced scaled locations across `[scaling.range_min, scaling.range_max]`; log-scaled when `scaling.zero_point` is set. Forecasters declare percentiles; `cdf_module.py` converts (same construction as the template bot — see the Numeric forecasts section below).
- Private reasoning comment (**REQUIRED on every forecast** — "Bots are required to leave private comments on all questions they forecast", notebook 38928): `POST /api/comments/create/` with `{"included_forecast": true, "is_private": true, "on_post": <post_id>}`. Built by `build_comment_payload()` in `metaculus_client.py`; `bot.py` forces comment posting on every `--submit` run.
- Projects: Fall 2026 FutureEval = slug `fall-futureeval-2026`, id **33121**; bot-testing-area = slug `bot-testing-area`, id **32977** (unscored, one question of every type); MiniBench = slug `minibench`.

Sources:
- Template bot, no-framework implementation (API calls, payload formats, CDF construction): https://github.com/Metaculus/metac-bot-template/blob/main/main_with_no_framework.py
- Template bot README (token name `METACULUS_TOKEN`, test/live modes, participation-form requirement): https://github.com/Metaculus/metac-bot-template
- forecasting-tools docs (MetaculusClient, binary bounds, comments): https://pypi.org/project/forecasting-tools/
- Bot Tournament Resources page (project IDs/slugs, token creation steps): https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/
- Official API root (linked from the resources page): https://www.metaculus.com/api

## Numeric forecasts — CDF construction and default strategy

**Construction (single implementation).** All numeric CDF work lives in
`cdf_module.py`, adopted from `wave3/meta-forecast-types/` after it was
proven equivalent to this scaffold's former inline implementation:
`percentiles_to_cdf(..., interp="linear")` reproduces the old scaffold
CDF with max abs diff < 1e-12 on linear, log-scaled (`zero_point` set),
and discrete questions (16/16 tests in
`wave3/meta-forecast-types/test_module.py`). `metaculus_client.py` keeps
its original public interface — `percentiles_to_cdf(percentiles,
question) -> list[float]`, `build_forecast_payload`, validators — as a
thin adapter over `cdf_module`; `bot.py` is unchanged in interface and
only routes numeric/discrete forecasts through the new default (below).
Running the scaffold now requires **numpy + scipy** (cdf_module's only
dependencies, beyond `requests`).

**Tail rule.** The optional `interp="pchip"` smoother in `cdf_module`
never extrapolates outside the declared percentiles: PCHIP tail
extrapolation was measured to explode (F = 7.3 / −10.3 at the grid edges
on a tame case, pinning the repaired CDF to 1). It evaluates on nominals
clipped to the declared range, so tails are constant — exactly like the
default linear construction, which is what submitted payloads use.

**Default strategy (from the Wave 3 evaluation, `wave3/meta-forecast-types/RESULTS.md`).**
Elicitation = **disciplined percentiles anchored to climatology, with
naive-normal density smoothing for log-score contexts**:

- Stage 2 (data-backed questions): elicit with
  `cdf_module.disciplined_percentiles` (climatology anchor + 50% drift
  shrinkage + volatility-regime rescaling + kurtosis tail fattening) or
  `cdf_module.climatology_percentiles` / `naive_normal_percentiles`.
- Stage 3 (`bot.py::judge_and_pool` → `metaculus_client.default_numeric_percentiles`):
  numeric/discrete percentile dicts are anchored to climatology
  percentiles when supplied (blend weight 0.5), then blended 50/50
  toward a naive normal fit through the declared points (log-value
  space on log-scaled axes). Date forecasts are not smoothed.

**What the evaluation actually found (pooled crypto n=603 origins;
BTC/ETH 7-day, GSPC 5-day replication; strictly causal):**

| Comparison (log score) | Result |
|---|---|
| Disciplined+anchored − naive normal | **+0.018 [−0.100, +0.138] — not significant; sign flips per asset (BTC +, ETH −, GSPC −)** |
| Disciplined+anchored − raw climatology | +0.137 [+0.009, +0.277] — beats raw climatology |
| Disciplined+anchored − naive (CRPS) | small significant gain, ~1.5% relative |
| Calibration | all arms near-calibrated (PIT means 0.486–0.497; 90% coverage 0.88–0.92) |

Honest caveat, also stated in the code: **the default's log-score edge
over a plain naive normal fit is unproven.** Anchoring is adopted
because it beats raw climatology and smoothing because log score
punishes the lumpy density of raw empirical percentiles — not because
the combination is a validated edge. Expect numeric questions to be won
on genuine distributional information, not CDF engineering. The 0.5/0.5
blend weights are transparent heuristic defaults, not tuned optima.

## Usage

```bash
python3 self_test.py                          # offline checks, no token needed
python3 bot.py                                # DRY-RUN, bot-testing-area
python3 bot.py --project fall-2026            # DRY-RUN, Fall 2026 questions
python3 bot.py --project bot-testing-area --submit   # LIVE (testing area)
python3 bot.py --project fall-2026 --submit   # LIVE (scored! private comment auto-attached)
```

Dry-run is the default; `--submit` is required for any live call. If a token
is present, dry-run still lists live questions (read-only GETs) and prints
the exact payloads it *would* send.

## Testing-area (32977) dry-run procedure

1. `python3 self_test.py` — all payload checks pass offline; and
   `python3 ../wave3/meta-forecast-types/test_module.py` — 16/16,
   including cdf_module ↔ scaffold CDF equivalence.
2. With the token in the environment (see checklist), run
   `python3 bot.py --project bot-testing-area` — confirm the intake lists
   one question of each type and the printed payloads validate.
3. `python3 bot.py --project bot-testing-area --submit` — submit to the
   unscored testing area.
4. On Metaculus: human account → Settings → My Forecasting Bots → switch to
   the bot account → open each testing question and confirm the forecasts
   landed (they appear on the bot account, not the human account).
5. Only then run against 33121.

## Go-live checklist

- [ ] CJ creates the human account and bot: metaculus.com → Settings →
      "My Forecasting Bots" → Create Bot → copy the API token.
      First bot created is the primary (prize-eligible) bot.
- [ ] CJ submits the FutureEval participation form (required before
      submitting forecasts; linked from the resources page / template repo).
- [ ] Token enters the Secure Vault (never chat, never a repo file).
- [ ] At run time only, the token is exported into the process environment:
      `export METACULUS_API_TOKEN=<from vault>` (the template's
      `METACULUS_TOKEN` name is also accepted as a fallback).
- [ ] Replace the `fleet_forecast` stub at INTEGRATION POINT A with the
      production pipeline (./sf forecast / ./sf pool) and wire the
      calibration judge at INTEGRATION POINT B. **Do not submit stub
      placeholders to a scored tournament.**
- [ ] Complete the 32977 testing-area procedure above, end to end.
- [ ] First scored run: `python3 bot.py --project fall-2026 --submit`
      during an attended session; verify forecasts on the bot account.
- [ ] Decide cadence: questions are open ~3h each at random times in
      FutureEval seasons — an unattended schedule (the template uses a
      20-minute loop) is a separate, later decision, as is any cron.

## Known unknowns (until the token exists)

- Exact live acceptance of `probability_yes_per_category` as a dict keyed by
  option label — this is the template bot's format, but the first 32977
  submission is the real proof. If the server wants an ordered list instead,
  only `build_forecast_payload` changes.
- Whether `GET /api/posts/` returns `aggregations` for bot tokens on
  tournament questions (irrelevant to forecasting — the blind view strips
  them regardless — but affects nothing else either way).
- Rate limits / pagination behaviour on project 33121 under a bot token;
  the client paginates conservatively (50/page) and surfaces server errors
  verbatim.
- Conditional questions are still skipped by this scaffold. GROUP questions
  are supported since Wave 3 (meta-group-questions): posts carrying
  `group_of_questions` are unpacked into per-subquestion forecasts through
  the same integration points, gated by a joint-consistency validator
  (partition sums / ladder monotonicity) in `group_module.py` — Metaculus
  documents NO server-side consistency enforcement and no joint payload
  object; forecasts submit per subquestion independently. UNVERIFIED until
  the 32977 dry-run: live field set of `group_of_questions` beyond
  questions/description/resolution_criteria/fine_print, and live acceptance
  of a multi-entry forecast list batching several subquestions in one POST
  (forecasting-tools and the template both post one entry per call).
  The Fall 2026 slate is expected to be single questions, but this is
  unverified until intake runs with a token.
