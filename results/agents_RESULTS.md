# Workstream F — Simulation Results

All numbers below are from **actually running** `agents.py` in this directory
(numpy 1.26.4, Python stdlib + numpy only). No numbers are estimated or invented.
Reproduce with the commands in §5.

**Simulation in one paragraph.** 8 synthetic agents with known parameters forecast
binary questions whose true generating probabilities are drawn from a
moderate/skewed/sharp mixture (see `generate_questions` in `agents.py`). Each
agent's forecast is generated in log-odds space as
`logit(pᵢ) = biasᵢ + βᵢ·logit(p_true) + common_shock + εᵢ`, with a shared shock
(sd 0.35 — everyone read the same news) and private noise εᵢ (sd = agent's σ).
Outcomes are Bernoulli draws from `p_true`. Aggregation strategies are evaluated
**causally**: track-record weights for question *t* use only questions *< t*, the
best-single-agent baseline is picked on the burn-in split only, and scores are
reported on the held-out test split only.

Agent roster (β = responsiveness to the true signal, σ = private noise):

| Agent | Persona | β | σ | bias |
|---|---|---:|---:|---:|
| bayesian_updater | Bayesian updater | 0.98 | 0.55 | 0.00 |
| base_rate_anchor | Base-rate anchorer | 0.62 | 0.35 | 0.00 |
| trend_model_reader | Trend / model reader | 0.90 | 0.75 | +0.05 |
| news_signal_hunter | News / signal hunter | 1.05 | 1.05 | +0.10 |
| contrarian | Contrarian / red-team | 0.55 | 0.90 | −0.15 |
| overconfident | Overconfident pundit (bench) | 1.35 | 0.85 | +0.15 |
| underconfident | Underconfident hedger (bench) | 0.70 | 0.60 | 0.00 |
| noisy_novice | Noisy novice (bench) | 0.80 | 1.60 | +0.20 |

---

## 1. Canonical run — 1,000 questions, seed 42 (150 burn-in / 850 test)

### Headline

| Forecaster | Brier ↓ | Log score ↓ | ECE ↓ |
|---|---:|---:|---:|
| Climatology (always the test base rate) | 0.2500 | — | — |
| Oracle (knows the true probability) | 0.1760 | 0.5264 | — |
| **Best single agent** (bayesian_updater, picked on burn-in) | 0.1863 | 0.5574 | — |
| Simple mean | 0.1870 | 0.5543 | 0.0328 |
| Median | 0.1875 | 0.5560 | 0.0440 |
| Log-odds pool (equal weights) | 0.1867 | 0.5542 | 0.0497 |
| Extremized pool (fixed 1.30×) | 0.1899 | 0.5698 | 0.0692 |
| **Track-record-weighted log-odds pool** | **0.1850** | **0.5507** | 0.0426 |
| Track-record-weighted extremized pool (1.30×) | 0.1880 | 0.5658 | 0.0656 |
| Tuned extremized pool (factor 0.95, fit on burn-in only) | 0.1865 | 0.5533 | — |

**Winner: the track-record-weighted log-odds pool**, on both scoring rules.

- vs. best single agent: Brier **0.1850 vs 0.1863** (−0.0013, −0.7% relative);
  log score **0.5507 vs 0.5574** (−0.0067, −1.2%).
- vs. simple mean: Brier −0.0020 (−1.1%); log score −0.0036.
- vs. equal-weight log-odds pool: Brier −0.0017 — the weighting layer, not just
  the pooling geometry, earns its keep.
- It remains 0.0090 Brier above the oracle (0.1760): the ensemble recovers part,
  not all, of the information the agents collectively hold.

At this sample size (850 test questions) the Brier margin over the best single
agent is **small**. §3–§4 show it widens and stabilises with more questions and
holds across most seeds — but it should be read as "consistent, modest edge",
not a rout. That is the normal shape of aggregation gains.

### Per-agent results (test split, seed 42)

| Agent | Brier ↓ | Log score ↓ | Final pool weight |
|---|---:|---:|---:|
| bayesian_updater | 0.1863 | 0.5574 | 0.160 |
| base_rate_anchor | 0.1942 | 0.5698 | 0.144 |
| trend_model_reader | 0.1954 | 0.5823 | 0.143 |
| overconfident | 0.1960 | 0.6012 | 0.138 |
| underconfident | 0.2007 | 0.5870 | 0.135 |
| news_signal_hunter | 0.2137 | 0.6386 | 0.117 |
| contrarian | 0.2344 | 0.6750 | 0.095 |
| noisy_novice | 0.2591 | 0.7662 | 0.068 |

The weighting layer ranks agents in exactly their true-skill order at the
extremes: the Bayesian updater ends with the highest weight (0.160 vs. the
equal-weight 0.125) and the noisy novice the lowest (0.068) — without ever
zeroing anyone out (floor + shrinkage working as designed in `DESIGN.md` §6).
Note the deliberate imperfection: the overconfident pundit (0.138) is weighted
above the signal hunter (0.117) because its *Brier* genuinely is better solo;
its poor log score (0.6012) is the warning the secondary metric exists to surface.

---

## 2. The extremizing result — the simulation's most important finding

A fixed **1.30× extremizing factor made the pool worse in every run**:
canonical Brier 0.1899 vs. 0.1867 un-extremized (log score 0.5698 vs. 0.5542),
and the weighted-extremized pool (0.1880) likewise lost to the plain weighted
pool (0.1850). The burn-in-tuned factor at seed 42 was **0.95** — i.e. the data
asked for essentially *no* extremizing, marginally the opposite.

This does **not** refute Satopää et al. (2014): GJP human crowds were
underconfident individuals sharing a prior, the exact profile extremizing
corrects. This simulated ensemble has a different profile (an overconfident
member with β = 1.35, a low-β contrarian, and a large shared shock), and in
that profile pushing the pool away from 50% mostly amplifies the shared error,
which log score punishes severely (note the log-score damage is consistently
larger than the Brier damage — overconfidence is expensive).

**Consequence (already written into `DESIGN.md` §5.3):** extremizing ships as a
*learned, per-domain, held-out-validated* parameter with default 1.00 — never
as a fixed 1.3× constant.

---

## 3. Robustness — same tournament, five seeds (1,000 questions each)

Brier score on each seed's held-out test split (850 questions). Bold = best pool.

| Strategy | seed 1 | seed 7 | seed 42 | seed 123 | seed 2026 |
|---|---:|---:|---:|---:|---:|
| Simple mean | 0.1794 | 0.1879 | 0.1870 | 0.1861 | 0.1829 |
| Median | 0.1804 | **0.1863** | 0.1875 | 0.1863 | 0.1851 |
| Log-odds pool | 0.1789 | 0.1883 | 0.1867 | 0.1866 | 0.1829 |
| Extremized (1.30×) | 0.1810 | 0.1914 | 0.1899 | 0.1908 | 0.1863 |
| **Weighted log-odds** | **0.1784** | 0.1873 | **0.1850** | **0.1860** | **0.1825** |
| Weighted extremized (1.30×) | 0.1805 | 0.1902 | 0.1880 | 0.1900 | 0.1860 |
| Tuned extremized (burn-in fit) | 0.1796 (×1.15) | 0.1886 (×0.80) | 0.1865 (×0.95) | 0.1866 (×1.00) | 0.1878 (×1.40) |

- Weighted log-odds pool: **best Brier in 4 of 5 seeds**; second in the fifth.
- Median won seed 7 outright (0.1863) — the median is a genuinely strong,
  parameter-free baseline and stays in the production diagnostics for that reason.
- Fixed 1.30× extremizing (plain or weighted) was **never** the best strategy
  in any seed, and was worst or second-worst in most.
- Tuned factors were unstable across seeds (0.80–1.40) and the tuned strategy
  never beat the weighted pool — at 150 burn-in questions there is not enough
  data to fit this parameter reliably, which is precisely why the production
  rule requires held-out improvement before adopting any factor ≠ 1.00.

## 4. Longer run — 2,000 questions, seed 42 (150 burn-in / 1,850 test)

| Forecaster | Brier ↓ | Log score ↓ |
|---|---:|---:|
| Climatology | 0.2498 | — |
| Oracle (knows p_true) | 0.1607 | 0.4814 |
| Best single agent (bayesian_updater) | 0.1756 | 0.5206 |
| Simple mean | 0.1708 | 0.5110 |
| Median | 0.1695 | 0.5057 |
| Log-odds pool | 0.1691 | 0.5045 |
| Extremized (1.30×) | 0.1700 | 0.5062 |
| **Weighted log-odds pool** | **0.1686** | **0.5029** |
| Weighted extremized (1.30×) | 0.1695 | 0.5046 |

With twice the test data the weighted pool's edge over the best single agent
widens to **0.0070 Brier (−4.0% relative, 0.1686 vs 0.1756)** and 0.0177 on log
score (0.5029 vs 0.5206), and it also posts the best calibration of any pool
(ECE 0.0223 vs. 0.0353 for the mean). The longer run also flatters extremizing
less badly (1.30× roughly ties the equal-weight pool) but it still does not win.

Per-agent at 2,000 questions (test Brier): bayesian_updater 0.1756,
base_rate_anchor 0.1816, trend_model_reader 0.1816, overconfident 0.1833,
underconfident 0.1865, news_signal_hunter 0.1948, contrarian 0.2079,
noisy_novice 0.2326. Final weights again put the Bayesian updater first (0.151)
and the noisy novice last (0.076).

---

## 5. Recommended production configuration

```python
# Aggregation (agents.py)
pool      = log_odds_pool(round2_probs, weights=track_record_weights(records))
final     = extremize(pool, factor=1.00)   # learned per domain later; see below

# Weights: track_record_weights(records, lam=12.0, shrinkage_n=30.0, floor=0.02)
#   - exponential in mean Brier, shrunk to equal weights by history, floored
#   - computed separately per domain: markets / prediction markets / crypto
#   - production change vs. sim: rolling last-200 window instead of expanding

# Extremizing: default 1.00. After >=150 resolved questions in a domain,
#   grid-search 0.80-2.00 (select_extremize_factor), cap at 1.40, adopt only
#   if it beats 1.00 held-out. Re-fit every 100 resolutions.

# Ensemble: the 5 core personas in DESIGN.md §2, blind round 1, structured
#   debate with mandatory self-refutation, round-2 revision, pool as above.
#   Always publish alongside: mean, median, equal-weight pool, best-single-
#   agent score, and pool dispersion.
```

**What the simulation does and does not establish.** It establishes that, for an
ensemble with this skill/noise/correlation profile, (a) pooling beats the best
member, (b) Brier-based track-record weighting beats equal weighting, and
(c) blind extremizing is harmful. It does **not** establish that live LLM
personas have these β/σ parameters — those must be measured on resolved live
questions, which is exactly what the scoring layer in `agents.py` is built to
do from the first question onward. Treat the synthetic tournament as a test of
the *aggregation machinery*, not as a forecast of live accuracy.

## 6. Reproduction

```bash
cd ~/workspace/superforecaster_out/agents
python3 agents.py                                  # canonical: 1000 q, seed 42
python3 agents.py --n-questions 2000 --seed 42     # longer run (§4)
for s in 1 7 42 123 2026; do python3 agents.py --seed $s; done   # robustness (§3)
```
