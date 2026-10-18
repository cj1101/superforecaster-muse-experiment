# EC-13 — Eligible-set ECE re-quotes of the headline N_star numbers (Wave 3)

Canonical citations from this file forward. Source workstream:
`superforecaster_out/wave3/ec13-requote/` (`RESULTS.md` §7 memo, `requote_grid.csv`,
`results.json`); the grid and full-precision JSON are vendored here as
`results/ec13_requote_grid.csv` and `results/ec13_requote_results.json` (data, not code).
Method: trades, CWR_bid, MDD and mean lock (7.0 d for every trading cell) are frozen
from the ps1/r8 artefacts; only the ECE term in
N_star = CWR_bid × √(365/mean_lock) × exp(−10·ECE) − 2·max(0, MDD−0.15) is re-quoted.
The headline convention is **per-book eligible-set ECE** (ECE-10U, canonical binning,
each book's own eligible rows, r12 bias floor attached, never subtracted).
Traded-subset headlines are retired to diagnostics. The production pooled ECE
0.0240 remains **telemetry only** — quoted as a candidate book's ECE it flips 6 S2a
cells positive (+0.14 … +0.36) and reverses every grid's ranking; it must never gate a book.

## ps1 — corrected grid (pooled OOS book, n = 1,446 T-7 markets)

| strategy | trades | CWR_bid | ECE eligible | bias floor | MDD | N_star published (traded) | **N_star HEADLINE (eligible)** |
|---|---|---|---|---|---|---|---|
| S0 skip-all | 0 | 0.0000 | — | — | 0.0000 | 0.0000 | **0.0000** |
| S1 naive raw | 0 | 0.0000 | 0.0385 | 0.0230 | 0.0000 | 0.0000 | **0.0000** |
| S2a frozen logit b=0.60 | 305 | +0.2497 | 0.0577 | 0.0248 | 0.7035 | −0.1456 | **−0.0949** |
| S2b frozen shrink s=0.925 | 0 | 0.0000 | 0.0462 | 0.0240 | 0.0000 | 0.0000 | **0.0000** |
| S3 category-conditional | 51 | −0.0433 | 0.0443 | 0.0231 | 0.2376 | −0.2527 | **−0.3759** |
| S4 longshot-bucket | 206 | −0.0545 | 0.0781 | 0.0218 | 0.9467 | −1.7056 | **−1.7736** |

## r8 — corrected grid, R1 FIXED-K mode (other modes in the CSV/JSON)

S0/S1/S2b take 0 trades in every mode and score 0.0000 under every convention.

| mode | strategy | trades | CWR_bid | MDD | N_star published (traded) | **N_star HEADLINE (eligible)** |
|---|---|---|---|---|---|---|
| R1 FIXED-K | S2a | 298 | +0.2141 | 0.6874 | −0.2841 | **−0.2070** |
| R1 | S3 | 51 | −0.0459 | 0.2277 | −0.2221 | **−0.3681** |
| R1 | S4 | 200 | +0.0158 | 0.7616 | −1.1823 | **−1.1710** |

## Verdict

- Traded → eligible: **no sign flips** in any of the 50 ps1/r8 cells; ranking changes
  in 2 of 7 grids, both cosmetic orderings among losers (r8 R1 best-trader, R4 worst).
- Pooled-constant → eligible: 6 sign flips, all S2a — the quantitative reason pooled
  quoting is retired for candidate books.
- **Production headline confirmed from the re-quoted numbers:** max eligible N_star
  over all 50 cells = **0.0000** (zero-trade cells only; best trading cell −0.0532,
  r8 R3-FLAT / S2a). N_star = 0.0000, SKIP stands (0 executed / 1,446 scanned).
- Canonical-scorer divergences measured here (not fixed by EC-13): alpha-binning (D6)
  max |ΔN_star| 0.0107, no verdict effect; velocity weighting (D9) and drawdown basis
  (D10) immaterial on these uniform-7-day-lock grids. The canonical scorer designation
  (capital-weighted, `superforecaster/alpha_kelly.py`) is recorded in the README (v2.0).
