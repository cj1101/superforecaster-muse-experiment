# Workstream F — Agent Ensemble / Superforecaster Protocol

**Scope.** The multi-agent forecasting protocol that sits *on top of* the quantitative
models produced by the other workstreams (calibration math, backtesting, markets,
prediction markets, crypto). Those models are inputs and evidence sources here; this
layer decides **how forecasts are elicited, challenged, revised, aggregated, scored,
and re-weighted over time**. Domain priority follows CJ's order: **(1) markets,
(2) prediction markets, (3) crypto** — and, critically, track records and weights are
kept **separate per domain** (§6), because skill does not transfer cleanly across them.

**Runnable companion.** `agents.py` (same directory) implements the aggregation and
scoring layer in pure numpy/stdlib, plus a simulated tournament that tests the
aggregation choices in this document. `RESULTS.md` reports what that simulation
actually produced. Where simulation and textbook intuition disagreed, this document
follows the simulation and says so (§5.3).

---

## 0. Evidence base — what we rely on, and what each finding buys us

| Finding (source) | Design consequence in this protocol |
|---|---|
| The Good Judgment Project (GJP) won the IARPA ACE forecasting tournament decisively; its top ~2% ("superforecasters") sustained their edge across years (Tetlock & Gardner, *Superforecasting*, 2015; Mellers et al., 2014) | Forecasting skill is real, identifiable, and persistent enough to justify track-record weighting (§6) — but it is a *distribution*, not a priesthood: weights are continuous, never winner-take-all. |
| Brief probability-reasoning **training** (base rates, avoiding over/underconfidence, averaging independent judgments) improved accuracy in GJP (Mellers et al., 2014, *Psychological Science*) | Every persona prompt embeds the training content: start from a base rate, update in log-odds, avoid 0/1, consider the opposite. Personas are *trained roles*, not vibes. |
| **Teaming** improved accuracy beyond solo forecasting (Mellers et al., 2014) | Structured debate round (§4) — but see the herding caveat below. |
| **Tracking / selecting** better forecasters and weighting them more improved the aggregate (Mellers et al., 2014; Tetlock & Gardner, 2015) | Exponential Brier-based pool weights with shrinkage (`track_record_weights` in `agents.py`). |
| **Extremizing** the crowd aggregate — pushing it away from 50% — improved GJP accuracy, because individual forecasters share information and each dilutes the others' private signal (Satopää, Baron, Foster, Mellers, Tetlock & Ungar, 2014, *Management Science*; Mellers et al., 2015) | Extremizing is in the aggregator (`extremize`) — but as a **learned, per-domain parameter, not a fixed constant**. Our simulation (§5.3, `RESULTS.md`) shows a fixed 1.3× factor *hurting* when the ensemble already contains overconfident and decorrelated members. The GJP result is conditional on the crowd's correlation/underconfidence profile; we measure that profile instead of assuming it. |
| Social influence can destroy crowd wisdom: even mild peer visibility narrows diversity and can worsen the aggregate (Lorenz, Rauhut, Schweitzer & Helbing, 2011, *PNAS*) | Round 1 is **blind and independent** (§3). No agent sees another's number before committing its own. Debate happens only after numbers are locked. |
| Considering the opposite / generating counterarguments reduces overconfidence (Lord, Lepper & Preston, 1984; Hirt & Markman, 1995 — "consider an alternative") | The debate round's hard requirement: each agent must state the **strongest evidence against its own number** (§4). Not a token devil's-advocate line — the single best counter-case, with a source. |
| Proper scoring rules (Brier, 1950; Gneiting & Raftery, 2007) make honesty the score-maximising strategy | All scoring, weighting, and leaderboard reporting uses Brier (primary) and log score (secondary, punishes confident error). No agent is ever rewarded for "calling the direction" while mispricing the probability. |
| Logarithmic opinion pools preserve external Bayesianity and treat each forecast as evidence in log-odds (Genest & Zidek, 1986; standard Bayesian aggregation practice in GJP) | Aggregation is done in **log-odds space**, not by averaging probabilities (§5). |
| Diversity of the pool is itself predictive — a mediocre-but-decorrelated forecaster can add more than a good-but-redundant one ("diversity prediction theorem", Page, 2007) | Weight floor (no persona is ever zeroed out), persona methods are *forced to differ* (§2), and we track inter-agent forecast correlation as a health metric (§7). |
| Superforecasters update **frequently and in small increments**, integrating new evidence rather than defending an anchor (Tetlock & Gardner, 2015, ch. on updating) | Update cadence schedule (§7) with small-step revisions and a mandatory changelog per update, rather than rare large flips. |
| "Surprisingly popular" / meta-prediction methods can beat simple pools when the crowd is systematically biased (Prelec, Seung & Shaper, 2017, *Nature*) | Not in v1. Logged as a v2 candidate (§9) — v1 stays with interpretable, auditable pooling. |

A note on transfer: almost all of the above is *human* forecaster research. For
LLM-agent ensembles two things change and two do not. **Unchanged:** the maths of
pooling, scoring, and extremizing; the value of independence before interaction.
**Changed:** (a) LLM agents sharing a base model share failure modes far more than
humans do — the "common shock" in our simulation — so persona diversity must be
enforced through *different tools, data, and methods*, not just different system
prompts; (b) LLM agents herd and defer to each other more readily in debate, which is
why debate here is structured, anonymised, and evidence-first (§4).

---

## 1. Question specification schema

No question enters the pipeline until it is resolvable *without argument*. Most
real-world forecasting failures are specification failures, not probability failures.

```yaml
question_id:        MKT-2026-0142            # domain prefix: MKT | PM | CRY
text:               "Will the S&P 500 close above 6,800 on 2026-12-31?"
resolution_criteria: >
  Resolved YES iff the official S&P 500 closing level published by
  S&P Dow Jones Indices for the last trading day of 2026 is > 6,800.00.
  Index redefinitions after question creation do not retroactively change
  the criterion; if the series is discontinued, resolves on the successor
  series named by the publisher, else N/A (excluded from scoring).
resolution_source:  "S&P DJI official close"   # single named arbiter, in advance
horizon_days:       74
created_at:         2026-10-17T16:00Z
resolves_at:        2026-12-31T21:00Z
base_rate:          0.61
base_rate_source:   "Frequency of >X% calendar-year gains, S&P 500, 1950-2025 (reference class stated explicitly)"
asset_class:        markets                  # markets | prediction_markets | crypto
question_type:      binary                   # v1: binary only; continuous in v2 via quantiles
update_schedule:    weekly                   # derived from horizon, see §7
status:             open                     # open | closed | resolved_yes | resolved_no | void
```

Rules:

1. **Binary, dated, single-arbiter.** One resolution source named at creation.
   "Will X do well?" is rejected. Ambiguity discovered later → question is voided
   and excluded from scoring (never resolved by vibes — voiding protects the
   integrity of the track records everything else depends on).
2. **Base rate is a required field**, with its reference class written out. If no
   defensible reference class exists, `base_rate: null` is allowed but must carry a
   one-line justification, and the Base-Rate Anchorer persona (§2) must say so
   explicitly in round 1 — a missing base rate is itself information about how
   speculative the question is.
3. **Prediction-market questions** additionally record the market, its current price,
   liquidity, and resolution rules *verbatim* — the market price is evidence for the
   News/Signal Hunter, never a substitute for the ensemble's own number, and the
   ensemble's edge (if any) is defined as its divergence from that price.
4. **Crypto questions** additionally record the exact venue/index for the price
   (e.g. a named exchange close vs. an index), because cross-venue discrepancies are
   a classic resolution trap in that domain.
5. Immutability: the spec is frozen at creation. Corrections create a *new*
   question_id; the old one is voided.

The `Question` dataclass in `agents.py` mirrors this schema.

---

## 2. Agent personas — genuinely different methods, not different tones

Five core personas (always on) + three simulation/bench personas. Each persona is
defined by **(a) a mandatory procedure, (b) its permitted evidence sources,
(c) its known failure mode**. Two personas that would produce the same number for
the same reason are merged — diversity is measured by forecast correlation (§7),
not by counting system prompts.

| # | Persona | Mandatory procedure | Primary evidence | Known failure mode (red-teamed in §4) |
|---|---|---|---|---|
| 1 | **Base-Rate Anchorer** | Start from the question's reference-class base rate. Adjust *only* for named, question-specific factors, each with an explicit log-odds increment. Must publish the adjustment ledger. | Historical frequencies, reference classes, long-run statistics | Too dull: misses genuine regime change; anchors on a stale reference class |
| 2 | **Trend / Model Reader** | Read the quantitative models (other workstreams): trend, momentum, vol regime, model-implied probabilities. Report what the models say *before* applying judgement; judgement adjustments are listed separately so model-vs-judgement can be scored apart later. | Workstream model outputs, price/technical structure, term structure, options-implied probabilities | Model worship: extrapolates a fitted regime past its break |
| 3 | **News / Signal Hunter** | Search for *question-specific, dated, checkable* signals: filings, scheduled catalysts, order flow, on-chain data (crypto), market microstructure (prediction markets). Every signal gets a date, a source, and a stated direction+strength. No undated "sentiment". | News, filings, calendars, on-chain / microstructure data | Recency & salience bias; highest variance of the core five — valuable signal, needs the pool to tame it (confirmed in simulation: worst-but-one core persona solo, yet the pool beats every solo agent) |
| 4 | **Contrarian / Red-Team** | Argue the consensus (including the other personas' likely consensus) is wrong: premortem ("it resolved the other way — what happened?"), base-rate neglect hunt, incentive analysis (who profits from the current price/narrative?). Must still output a number, not just objections. | Incentives, positioning, historical analogues of consensus failure | Reflexive opposition: decorrelated but biased; weak solo, useful *only* inside a pool — which is exactly what the simulation shows (Brier 0.234 solo at seed 42, yet removing diversity is never tested as an improvement) |
| 5 | **Bayesian Updater** | Owns the running posterior. Starts from the previous ensemble posterior (or base rate for new questions), applies each new evidence item as an explicit likelihood ratio, publishes the full update chain. In round 1 it works from base rate + models only, blind like everyone else. | Prior posterior, likelihood-ratio discipline, all dated evidence | False precision: a tidy chain built on invented likelihood ratios — mitigated by requiring each LR to cite its evidence item |
| 6 | Overconfident Pundit *(bench)* | Strong narrative, high confidence — included in the simulation roster to prove the weighting layer can contain this archetype. Not a production persona. | Narrative | Overconfidence (β>1 in the sim) |
| 7 | Underconfident Hedger *(bench)* | Shrinks everything toward 50%. Bench persona. | — | Under-reaction (β<1) |
| 8 | Noisy Novice *(bench)* | Untrained baseline. Bench persona; in the sim its final pool weight falls to 0.068 (lowest) — the weighting layer working as intended. | — | Noise |

**Enforcing real diversity.** Personas differ in *tools and inputs*, not just
instructions: the Anchorer does not see today's headlines in round 1; the Signal
Hunter does not see the model stack; the Model Reader does not see narrative news.
Where personas run on the same underlying LLM, they additionally get different
retrieval sets and different worked examples. Forecast correlation between personas
is computed weekly; any pair sustaining correlation > 0.95 over 50+ shared questions
is flagged as collapsed diversity and one of them is re-specified.

---

## 3. Round 1 — independent forecasts (no anchoring on peers)

1. Each persona receives the frozen question spec + its own evidence pack only.
   **No persona sees any other persona's forecast, rationale, or identity-linked
   commentary before submitting.**
2. Submission format (machine-checkable):

```yaml
agent_id: trend_model_reader
question_id: MKT-2026-0142
probability: 0.64            # strictly in (0.01, 0.99) in production; 0/1 banned
round: 1
base_rate_used: 0.61
adjustments:                 # log-odds ledger, items must sum (approx.) to the move from base rate
  - {factor: "options skew / implied vol", log_odds: +0.12, source: "model stack v3, 2026-10-17"}
strongest_counterevidence: ""   # filled in round 2 preparation, see §4
```

3. Forecasts are **locked and timestamped** on submission. The lock is what makes
   round-1 scores meaningful and what makes the wisdom-of-crowds maths applicable
   (Lorenz et al., 2011: independence first, interaction second).
4. The round-1 pool (§5, equal weights — no track record is question-specific yet
   at this stage of the round) is computed and stored, but **not shown to agents**
   before debate statements are written. Showing it would anchor the debate.

---

## 4. Structured debate — argue against yourself, on the record

Debate is where teaming gains come from (Mellers et al., 2014) and where herding
losses come from (Lorenz et al., 2011). The structure below is designed to harvest
the first and block the second:

1. **Anonymised exchange.** Agents receive the other round-1 rationales with
   agent identities and probabilities **stripped of identity** (numbers visible —
   they are the object of debate — but unattributed, so a strong track record
   cannot substitute for an argument).
2. **Mandatory self-refutation.** Each agent must submit, before seeing revisions:
   - the **strongest evidence against its own number** (one best counter-case,
     sourced — "risks exist" does not pass review);
   - the specific observation that would make it move most ("what would change
     my mind"), stated in advance;
   - a steelman of the position furthest from its own.
   This is the `strongest_counterevidence` field on `AgentForecast` in `agents.py`.
3. **Cross-examination by role.** The Contrarian's self-refutation is checked by
   the Anchorer; the Signal Hunter's by the Bayesian Updater; etc. — each agent
   grades whether the assigned peer's counter-case is genuinely its strongest, or
   a strawman. Strawman verdicts are returned for rewrite, once.
4. **No new numbers during debate.** Debate trades *evidence items and likelihood
   ratios only*. Anyone wanting to move does so in round 2, with the evidence item
   that moved them cited. This keeps the revision attributable: we can later score
   *which kinds of evidence actually improve forecasts*.
5. **Facilitator summary** (non-forecasting role): a neutral list of (a) agreed
   facts, (b) contested facts, (c) evidence items introduced in debate that no
   round-1 forecast used. Only (c) is expected to move numbers much; if round 2
   moves a lot on (a)/(b) alone, that flags anchoring/herding for review.

---

## 5. Round 2, revision, and aggregation

### 5.1 Revision rules

- Round-2 submission uses the §3 format with `round: 2`, plus `moved_because:`
  citing the debate evidence item(s) responsible. A move with no cited item is
  accepted but flagged as *unattributed* (herding suspect).
- **Small-step prior:** superforecasters update in small increments (Tetlock &
  Gardner, 2015). Moves > 0.25 in probability require an explicit "regime change"
  justification; they are allowed — sometimes the world does change — but they are
  audited after resolution.
- Agents may also **stand pat**. Standing pat with a stated reason is a complete
  submission; forced movement is banned (it manufactures noise).

### 5.2 Aggregation (implemented in `agents.py`)

Applied to round-2 forecasts (round-1 pool is kept as a diagnostic: if the round-2
pool is systematically *worse* than round-1, debate is hurting and §4 gets revised).

1. **Pool in log-odds space** with track-record weights:
   `logit(p_pool) = Σ wᵢ · logit(pᵢ)`, `wᵢ` from §6.
   (`log_odds_pool` in `agents.py`.)
2. **Extremize** with a *learned* factor: `logit(p_final) = a · logit(p_pool)`
   around the 50% centre (`extremize`), where `a` is selected per domain on
   resolved history (§5.3).
3. **Diagnostics always reported alongside the number:** simple mean, median,
   equal-weight log-odds pool, the pool's dispersion (IQR of agent log-odds), and
   each agent's weight. A forecast shipped without its diagnostics is incomplete —
   dispersion is the ensemble's own uncertainty-about-uncertainty signal and feeds
   position-sizing discussions downstream (outside this workstream).

Baselines kept for comparison forever: the **median** (robust, embarrassingly
strong — it won one of our five simulation seeds outright) and the **best single
agent** (any ensemble that cannot beat its best member is overhead, not a system).

### 5.3 The extremizing factor — measured, not assumed

GJP crowds were systematically underconfident, so extremizing helped them
(Satopää et al., 2014). Our simulated ensemble is *not* a GJP crowd: it contains an
overconfident member, a decorrelated contrarian, and substantial shared-information
noise. In the simulation (`RESULTS.md`), a fixed 1.30× factor **worsened** Brier in
every 1,000-question seed tested, and the burn-in-tuned factor landed at
0.80–1.40 depending on seed (0.95 at the canonical seed) — i.e. the honest answer is
"it depends on the ensemble's measured correlation/underconfidence profile."

Production rule, therefore:

- Default factor **1.00** (no extremizing) until a domain has ≥150 resolved
  questions.
- Then select the factor by grid search (0.80–2.00, step 0.05) minimising Brier
  on resolved history (`select_extremize_factor`), **capped at 1.40**, re-fit
  every 100 new resolutions, and adopted only if it beats 1.00 on a held-out
  slice — never on the same data it was fit on.
- The weighted-extremized strategy stays in the code and in every tournament
  report, so if the live ensemble's profile drifts toward GJP-like
  underconfidence, the data will show it and the factor will rise on evidence.

---

## 6. Forecaster scoring & weighting over time

- **Scores.** Every resolved question scores every agent (both rounds), the pool,
  and every baseline, on **Brier (primary)** and **log score (secondary)**.
  Scores are stored per domain (markets / prediction markets / crypto) and pooled
  only for reporting, never for weighting across domains.
- **Weights.** `wᵢ ∝ exp(−λ · mean_Brierᵢ)` with λ = 12, **shrunk toward equal
  weights** by history length (`n/(n+30)`) and **floored at 0.02**
  (`track_record_weights`). Rationale:
  - exponential-in-Brier is the standard proper-score weighting (a 0.05 Brier
    edge ≈ 1.8× weight before shrinkage — material, not decisive);
  - shrinkage stops a lucky 5-question streak from capturing the pool;
  - the floor protects diversity (Page, 2007): the Contrarian's value is
    decorrelation, which a pure Brier ranking under-prices in quiet regimes.
- **Decay.** Weights use a rolling window of the last 200 resolved questions per
  domain (plus the shrinkage prior), so a persona whose method stops working —
  or whose underlying model is replaced — loses weight within a quarter, not
  never. (The simulation uses an expanding window; the rolling window is the
  production change, noted here so the two are not confused.)
- **Calibration reporting.** Per-agent and per-pool calibration (ECE + reliability
  bins, `calibration_stats`) is published with every stats update. A well-ranked
  but badly calibrated agent is a sizing hazard even if its Brier is fine.
- **Round-1 vs round-2 delta** is scored per agent: personas whose revisions
  systematically *worsen* their score get their debate behaviour reviewed (§4) —
  some agents should learn to stand pat.
- **Retirement/repair.** A persona in the bottom weight decile for two consecutive
  quarters *and* with calibration worse than the pool's is re-specified (new
  procedure/examples) rather than deleted; its history resets under a new
  agent_id version suffix (`news_signal_hunter_v2`) so the record stays honest.

---

## 7. Update cadence as resolution approaches

Superforecasters' edge is substantially an *updating* edge. Cadence is set by
horizon and domain volatility, and **tightens as resolution approaches**, because
information arrival accelerates and stale forecasts are the most common live failure:

| Time to resolution | Markets | Prediction markets | Crypto |
|---|---|---|---|
| > 90 days | Monthly | Monthly | Biweekly |
| 30–90 days | Biweekly | Weekly | Weekly |
| 7–30 days | Weekly | Twice weekly | Twice weekly |
| 1–7 days | Twice weekly | Daily | Daily |
| < 24 hours | Daily + event-driven | Event-driven (price move > 5pp or news catalyst) | Event-driven (price move > 5% or catalyst) |
| Any time | **Event-driven update** on: scheduled catalyst release (earnings, CPI, expiry), a ≥2σ move in a model input, or any persona's pre-registered "what would change my mind" trigger firing (§4.2) | Same | Same |

Each update is a *mini-round*: the Bayesian Updater posts the evidence items and
proposed likelihood ratios; personas confirm or dissent in log-odds increments;
the pool is recomputed and the change from the previous published forecast is
logged with its cause. Full blind round-1 + debate is re-run only on a regime-change
flag (§5.1) or at the <7-day mark, whichever comes first.

**Health metrics reviewed weekly:** inter-persona correlation matrix (diversity,
§2), round-2-minus-round-1 pool Brier delta (is debate helping?), unattributed-move
rate (herding), mean |update| size (small-step discipline), and per-domain weight
drift (is anyone quietly capturing the pool?).

---

## 8. Failure modes and mitigations (summary)

| Failure mode | Mitigation in this design |
|---|---|
| Correlated LLM failure (same base model, same blind spot) | Different tools/data per persona (§2); correlation monitoring; common-shock term in the simulation so we never mistake correlated agreement for independent confirmation |
| Anchoring on peers / herding in debate | Blind round 1, anonymised debate, no numbers traded in debate, unattributed-move flag (§§3–5) |
| Resolution ambiguity corrupting track records | Frozen spec, single arbiter, void-don't-fudge rule (§1) |
| Overconfident extremizing | Learned, capped, held-out-validated factor; default 1.00 (§5.3) |
| Weight capture by a lucky persona | Shrinkage + floor + rolling window (§6) |
| News hallucination (Signal Hunter) | Dated, sourced signals only; facilitator fact list separates agreed vs contested facts (§4) |
| Debate theatre (strawman self-refutation) | Cross-grading of counter-cases (§4.3) |

## 9. Explicit non-goals / v2 candidates

- No trade/bet execution — this layer outputs probabilities and diagnostics only.
- Continuous/distributional questions (quantile elicitation + CRPS) — v2.
- "Surprisingly popular" aggregation (Prelec et al., 2017) — v2 candidate, to be
  A/B-tested in the tournament harness before any production use.
- Market-price blending for prediction markets (pool ⊕ market price) — v2; v1
  keeps the ensemble's number independent so its *standalone* skill stays measurable.

## References

- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*.
- Genest, C. & Zidek, J. V. (1986). Combining probability distributions. *Statistical Science*.
- Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA*.
- Hirt, E. R. & Markman, K. D. (1995). Multiple explanation discounting. *JPSP*.
- Lord, C. G., Lepper, M. R. & Preston, E. (1984). Considering the opposite. *JPSP*.
- Lorenz, J., Rauhut, H., Schweitzer, F. & Helbing, D. (2011). How social influence can undermine the wisdom of crowd effect. *PNAS*.
- Mellers, B. et al. (2014). Psychological strategies for winning a geopolitical forecasting tournament. *Psychological Science*.
- Mellers, B. et al. (2015). The psychology of intelligence analysis / GJP aggregation reports (extremizing & teaming results). 
- Page, S. E. (2007). *The Difference* (diversity prediction theorem).
- Prelec, D., Seung, H. S. & Shaper, J. (2017). A solution to the single-question crowd wisdom problem. *Nature*.
- Satopää, V. A., Baron, J., Foster, D. P., Mellers, B. A., Tetlock, P. E. & Ungar, L. H. (2014). Combining multiple probability predictions using a simple logit model. / Extremizing in GJP. *Management Science* & related GJP working papers.
- Tetlock, P. E. (2005). *Expert Political Judgment*.
- Tetlock, P. E. & Gardner, D. (2015). *Superforecasting: The Art and Science of Prediction*.

---

## Addendum v1.1 — requirements adopted from external advice review (2026-10-17)

Full review: `docs/EXTERNAL_ADVICE_REVIEW.md`. Binding for the live-agent phase:

1. **Anchorer isolation (hardened).** The Base-Rate Anchorer receives the question
   spec and historical/reference-class data ONLY — no news feed, no market chatter,
   no other agent's output — and must name 2–3 reference classes with their empirical
   frequencies. Its anchor enters the pipeline as the mandatory prior.
2. **Contamination control for LLM evaluation.** Any backtest of LLM agents on
   resolved questions must enforce publication-date cutoffs at the retrieval layer
   AND run a foreknowledge filter (drop questions whose resolution, or documents
   describing it, could exist in an agent's pretraining/retrieval window).
3. **Supervisor = search-to-reconcile.** No rationale judging, no prose-quality
   re-weighting. The supervisor activates only when Round-2 forecasts show high
   variance or bimodal disagreement; it identifies the single factual premise
   causing the split, runs a targeted verification search on that crux, and updates
   the aggregate only if the new evidence decisively resolves it. Default
   aggregation remains the track-record-weighted log-odds pool.
4. **Scenario-mixture outputs.** Live agents report M mutually exclusive scenarios
   with weights summing to 1 and P(event | scenario); the agent's probability is
   the mixture sum, never a bare scalar appended to free-form reasoning.
5. **Scoring.** Agents are scored with strictly proper rules (Brier primary, log
   secondary, peer score vs the contemporaneous crowd/market wherever one exists).
