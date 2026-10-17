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
  Kelly fraction plus the KL information edge.
- Score yourself with `sf score` (Brier / log / ECE / Murphy / peer score).
- Pool colleagues with `sf pool` (production pool = track-record-weighted
  log-odds; extremization ships 1.00 until learned per-domain).
- The canonical N_star engine is `superforecaster/alpha_kelly.py`
  (CLOB sweep, account simulator, scorecard). `./sf harness-demo` runs its
  synthetic fixture benchmark — a self-test, never performance evidence.

## Quickstart
```
./sf headline
./sf forecast --domain markets --symbol NVDA --horizon 21
./sf forecast --domain predmarkets --slug xi-jinping-out-before-2027
./sf edge --p 0.62 --q-exec 0.55 --domain crypto
./sf pool --probs 0.60,0.70,0.55 --weights 0.16,0.14,0.10
./sf score --file forecasts.csv   # columns: y,p[,crowd]
```
