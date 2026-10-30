# Agent Interface — `sf` CLI

Every fleet agent (and any external agent) talks to the Superforecaster through
one JSON contract. Run `./sf schema` for the machine-readable spec — it is the
source of truth and always matches the shipped code.

## Contract
- One JSON object on stdout per call. Exit 0 ok / 2 bad input / 3 data unavailable.
- Consume `production_p` from `sf forecast`, never `raw_p`: production numbers
  are skill-gated and shrinkage-disciplined (superforecaster/engine.py).
- Trading decisions go through `sf edge`: it enforces the N_star rules
  (SKIP unless |p - q_exec| >= 0.05 after costs; fractional Kelly lambda in
  [0.25, 0.5]; 15% cluster cap; crypto x0.5 dampener) and returns the capped
  Kelly fraction plus the KL information edge. **A SKIP sizes zero (EC-09):
  on any SKIP, `kelly_fraction_of_account` is 0.0 and
  `expected_log_growth_at_kelly_size` is 0.0 — never size a position off a
  SKIP payload's other fields.**
- **`--q-exec` side convention (SF-BL-9323): `--q-exec` is the YES-side
  execution price by default.** If your execution price comes from walking
  a NO-token book (e.g. `nstar.vwap_for_dollars` on NO asks), pass
  `--q-exec-side no`: the value is then used as-is as the NO price for
  BUY_NO and is never complemented. Never pass a NO-side VWAP untagged —
  read as a YES price it is complemented a second time, which corrupts
  Gate D and the Kelly fraction (and can manufacture a trade on an
  overpriced NO book). A `--q-exec-side no` call never flips to BUY_YES:
  `1 - q_no` is not the YES VWAP. Every `sf edge` payload echoes
  `q_exec_side` and the `q_side` Gate D actually used.
- **`--domain` is canonicalised (SF-BL-9580):** for both `sf forecast` and
  `sf edge`, the label is stripped, case-folded and stripped of `-`/`_`/space
  separators, then mapped to `markets` / `crypto` / `predmarkets` (crypto
  aliases: `cryptocurrency`, `cryptocurrencies`, `cryptobench`). Every
  crypto label variant — `Crypto`, `CRYPTO`, ` crypto` — triggers Gate A
  and the crypto x0.5 Kelly dampener exactly like `--domain crypto`. Any
  other label is rejected with exit 2 and exactly one JSON
  `{"error": ...}` object: an unknown domain is never silently traded as a
  non-crypto domain. (Before this fix the comparison was verbatim, so
  `--domain Crypto` admitted an unverified, undampened TRADE.)
- **`--p` and `--q-exec` must be finite (SF-BL-9717):** `sf edge`
  validates both at the CLI boundary, before any clipping, gate or
  sizing. A non-finite value — `nan`, `inf`, `-inf`, in any spelling
  Python's float parser accepts (including `1e999`, which overflows to
  inf) — is rejected with exit 2 and exactly one JSON
  `{"error": ...}` object naming the offending flag, and no decision
  is computed. (Before this fix `np.clip` passed NaN through and
  mapped ±inf to the clip boundary, so non-finite inputs produced
  NaN-token bodies, downstream tracebacks, or silent trades at the
  boundary price — never the contract's exit-2 error.) Finite
  out-of-range values keep their pre-existing clip behaviour.
- **`--lam` must be finite and in [0.25, 0.50] (SF-BL-10659):** `sf edge`
  validates `--lam` at the CLI boundary, before any sizing. A non-finite
  value (`nan`, `inf`, `-inf`, any spelling, including `1e999`) or a
  finite value outside [0.25, 0.50] is rejected with exit 2 and exactly
  one JSON `{"error": ...}` object naming `--lam`, and no decision is
  computed — on TRADE and SKIP paths alike; the value is never clamped
  into the band. Boundaries 0.25 and 0.50 are inclusive; the default is
  0.35. (Before this fix `cmd_edge` sized inline off the raw `--lam`
  and never called the guarded `nstar.kelly_fraction`, so `lam=-1.0`
  exited 0 as a TRADE with a negative Kelly fraction of -0.2083 on the
  0.62/0.52 fixture and `lam=2.0` silently sized at the 15% cluster
  cap.) In-band inputs are unchanged.
- Score yourself with `sf score` (Brier / log / ECE / Murphy / peer score).
  The score file must contain at least one data row (SF-BL-9747): an
  empty or header-only file has no defined score (every metric is a
  mean over n=0) and is rejected with exit 2 and exactly one JSON
  `{"error": ...}` object — never an exit-0 body with bare `NaN`
  literals for brier/log_score and the Murphy residual/base_rate.
- Pool colleagues with `sf pool` (production pool = track-record-weighted
  log-odds; extremization ships 1.00 until learned per-domain). Pool inputs
  are validated at the CLI boundary (SF-BL-9544): every `--probs` value
  must be finite and in [0,1], and `--weights` must be finite,
  non-negative, length-matched to `--probs` and have a positive sum. A
  violation exits 2 with exactly one JSON `{"error": ...}` object and no
  pooled figure — a >1 or NaN "probability" can never be published as a
  pool for a downstream command to clip silently into a trade decision.
- The canonical N_star engine is `superforecaster/alpha_kelly.py`
  (CLOB sweep, account simulator, scorecard). `./sf harness-demo` runs its
  synthetic fixture benchmark — a self-test, never performance evidence.

## Quickstart
```
./sf headline
./sf forecast --domain markets --symbol NVDA --horizon 21
./sf forecast --domain predmarkets --slug xi-jinping-out-before-2027
./sf edge --p 0.62 --q-exec 0.55 --domain crypto
./sf edge --p 0.42 --q-exec 0.40 --q-exec-side no   # --q-exec is a NO-book VWAP
./sf pool --probs 0.60,0.70,0.55 --weights 0.16,0.14,0.10
./sf score --file forecasts.csv   # columns: y,p[,crowd]
```
