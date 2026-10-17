# Calibration rescue experiment (v2) — RESULTS

Question: the v1 direction models lose to climatology on raw Brier. Can a **causal** calibration
layer rescue them? Every transform is fitted only on out-of-sample forecasts preceding the one
being transformed (expanding window, refit every 63 forecasts, first 200 forecasts are calibration
fuel only and never scored). Script: `experiments/calibrate_oos.py`.

| stream | n scored | raw Brier | climatology | Platt | shrink (fitted w) | Platt+shrink | skill shrink | skill Platt |
|---|---|---|---|---|---|---|---|---|
| crypto/BTC_1d | 2580 | 0.25121 | 0.25098 | 0.24996 | 0.25046 | 0.24996 | +0.0021 | +0.0041 |
| crypto/BTC_30d | 2551 | 0.26377 | 0.24971 | 0.25670 | 0.24971 | 0.25670 | +0.0000 | -0.0280 |
| crypto/BTC_7d | 2574 | 0.25295 | 0.25026 | 0.25242 | 0.25052 | 0.25242 | -0.0010 | -0.0086 |
| crypto/BTC_crash7d | 2574 | 0.08705 | 0.08980 | 0.08729 | 0.08758 | 0.08729 | +0.0247 | +0.0280 |
| crypto/ETH_1d | 2275 | 0.25021 | 0.24983 | 0.24947 | 0.24922 | 0.24945 | +0.0024 | +0.0014 |
| crypto/ETH_30d | 2246 | 0.25900 | 0.25276 | 0.25982 | 0.25471 | 0.25977 | -0.0077 | -0.0279 |
| crypto/ETH_7d | 2269 | 0.25054 | 0.25032 | 0.25094 | 0.25017 | 0.25094 | +0.0006 | -0.0025 |
| crypto/ETH_crash7d | 2269 | 0.14122 | 0.14390 | 0.14019 | 0.14174 | 0.14019 | +0.0150 | +0.0258 |
| markets/^GSPC_1d | 1505 | 0.25291 | 0.24924 | 0.25004 | 0.24954 | 0.25000 | -0.0012 | -0.0032 |
| markets/^GSPC_5d | 1501 | 0.25239 | 0.24333 | 0.24392 | 0.24354 | 0.24388 | -0.0009 | -0.0024 |
| markets/^GSPC_21d | 1485 | 0.24663 | 0.22169 | 0.22609 | 0.22366 | 0.22607 | -0.0089 | -0.0198 |
| markets/QQQ_1d | 1505 | 0.25231 | 0.24819 | 0.25015 | 0.24860 | 0.25013 | -0.0017 | -0.0079 |
| markets/QQQ_5d | 1501 | 0.25170 | 0.24542 | 0.24835 | 0.24604 | 0.24833 | -0.0025 | -0.0119 |
| markets/QQQ_21d | 1485 | 0.24907 | 0.23380 | 0.24153 | 0.23587 | 0.24111 | -0.0089 | -0.0331 |
| markets/NVDA_1d | 1505 | 0.25050 | 0.24901 | 0.24978 | 0.24936 | 0.24978 | -0.0014 | -0.0031 |
| markets/NVDA_5d | 1501 | 0.24993 | 0.24651 | 0.25062 | 0.24717 | 0.25062 | -0.0027 | -0.0167 |
| markets/NVDA_21d | 1485 | 0.23725 | 0.23910 | 0.25119 | 0.23890 | 0.25119 | +0.0008 | -0.0506 |
| markets/TSM_1d | 1505 | 0.25272 | 0.25058 | 0.25111 | 0.25055 | 0.25109 | +0.0001 | -0.0021 |
| markets/TSM_5d | 1501 | 0.25747 | 0.24936 | 0.24888 | 0.24936 | 0.24888 | +0.0000 | +0.0019 |
| markets/TSM_21d | 1485 | 0.26444 | 0.24199 | 0.24814 | 0.24266 | 0.24808 | -0.0028 | -0.0254 |

## Interpretation

- **Fitted shrinkage is the robust layer.** Its skill vs climatology stays inside [-0.0089, +0.0247]
  on all 20 streams: it converts models that lost up to ~-0.10 raw into approximately-climatology
  forecasts (when the evidence says so, the fitted weight goes to ~0), and it preserves the one
  real edge — crypto crash risk (BTC +0.0247, ETH +0.0150 after shrinkage).
- **Platt scaling, applied causally on real streams, is unreliable.** It helped on synthetic data
  (workstream A) and on the stationary-ish crash streams (+0.028), but it badly hurt drifting
  direction streams (NVDA 21d -0.0506, QQQ 21d -0.0331, BTC 30d -0.0280): a recalibration map
  fitted on the past amplifies whatever regime just ended. It is therefore NOT in the production
  path (`superforecaster/engine.py`).
- **Where positive skill survives all layers:** crypto crash risk (both assets), BTC/ETH 1-day
  direction (+0.002 to +0.004 — real but tiny), TSM 5-day (+0.0019). Everything else is, honestly,
  climatology with extra steps — and the production engine treats it exactly that way.
- Caveat: the Platt+shrink variant's shrink weight is fitted on Platt-transformed past forecasts
  using the same Platt map (one-parameter fit; mild optimism in w only). The pure shrink variant
  has no such caveat and is the recommended production transform.
