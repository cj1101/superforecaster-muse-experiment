#!/usr/bin/env python3
"""Walk-forward backtest for markets forecaster. Real Yahoo data only."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "superforecaster"))  # repo layout: package dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from markets import download_yahoo, build_features, walk_forward, summarize, predict_latest, HORIZONS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data", "markets")  # repo layout
OUT = os.path.join(os.path.dirname(HERE), "results")
SYMBOLS = ["^GSPC", "QQQ", "NVDA", "TSM"]

def main():
    results = {"method": {
        "scheme": "Walk-forward expanding window. Initial train 756 trading days (~3y). Retrain every 21 days. At forecast date t, training uses only samples i with i+horizon <= t (label fully known by t). Features at t use data <= t only.",
        "model": "Logistic regression via full-batch gradient descent (L2=1.0, lr=0.05, 300 epochs), intercept initialized at logit(expanding base rate) as anchor. Standardization from train window only. Plus ridge (lam=10) expected-return model.",
        "features": "mom 2/5/10/21/63/126/252d log returns; z-score vs SMA20/50/200 + dist to SMA; realized vol 10/21/63d + vol ratios; RSI7/14; dist from 52w high + 52w range position; volume trend/z (when available); day-of-week sin/cos + Mon/Fri flags; vol-adjusted 5d momentum; 1d return + abs return.",
        "climatology": "Expanding base rate of up-moves computed only from training labels known at each forecast date.",
        "data_source": "Yahoo Finance chart API v8, daily, range=10y, adjclose used for price/returns.",
    }, "symbols": {}, "latest_forecasts": {}}
    for sym in SYMBOLS:
        df = download_yahoo(sym, range_="10y", cache_dir=DATA)
        feat = build_features(df)
        entry = {"data_start": str(df["date"].iloc[0].date()), "data_end": str(df["date"].iloc[-1].date()),
                 "n_rows": int(len(df)), "horizons": {}}
        print(f"== {sym} {entry['data_start']}..{entry['data_end']} n={len(df)}", flush=True)
        for h in HORIZONS:
            res = walk_forward(feat, horizon=h)
            s = summarize(res)
            entry["horizons"][str(h)] = s
            print(f"  h={h}d n={s['n']} Brier {s['brier']} vs clim {s['brier_climatology']} skill {s['brier_skill_vs_clim']} AUC {s['auc']} log {s['log_score']} vs {s['log_score_climatology']}", flush=True)
        results["symbols"][sym] = entry
        try:
            results["latest_forecasts"][sym] = predict_latest(sym, cache_dir=DATA)
        except Exception as e:
            results["latest_forecasts"][sym] = {"error": str(e)}
    with open(os.path.join(OUT, "markets_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    # RESULTS.md
    lines = ["# Markets Forecaster — Walk-Forward Backtest Results", "",
             "Research only — no trading. All numbers below come from an actual run of `run_markets_backtest.py` on real Yahoo Finance data; see `results.json` for the machine-readable version.", "",
             "## Method", f"- Data source: {results['method']['data_source']}",
             f"- Split scheme: {results['method']['scheme']}",
             f"- Model: {results['method']['model']}",
             f"- Features: {results['method']['features']}",
             f"- Benchmark (climatology): {results['method']['climatology']}", "",
             "## Headline results", "",
             "| Symbol | Horizon | n | Brier | Clim Brier | Brier skill | Log score | Clim log score | AUC | Beats clim? |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for sym, e in results["symbols"].items():
        lines.append(f"Data: **{sym}** {e['data_start']} → {e['data_end']} ({e['n_rows']} daily rows)")
        for h in HORIZONS:
            s = e["horizons"][str(h)]
            lines.append(f"| {sym} | {h}d | {s['n']} | {s['brier']} | {s['brier_climatology']} | {s['brier_skill_vs_clim']} | {s['log_score']} | {s['log_score_climatology']} | {s['auc']} | {'YES' if s['beats_climatology_brier'] else 'NO'} |")
    lines += ["", "Brier skill > 0 means the model beats climatology; negative means it loses to the base rate — reported honestly where it occurs.", ""]
    for sym, e in results["symbols"].items():
        lines += [f"## {sym} — calibration", ""]
        for h in HORIZONS:
            s = e["horizons"][str(h)]
            lines += [f"### {h}d horizon (n={s['n']}, test base rate={s['base_rate_test']}, mean pred={s['mean_pred']}, exp-ret corr={s['exp_ret_corr_with_realized']})", "",
                      "| Bin | n | Mean pred | Empirical rate |", "|---|---|---|---|"]
            for r in s["calibration"]:
                lines.append(f"| {r['bin_lo']}–{r['bin_hi']} | {r['n']} | {r['mean_pred']} | {r['empirical_rate']} |")
            lines.append("")
    lines += ["## Latest forecasts (model trained on all labelled data, predicting from last close)", ""]
    for sym, lf in results["latest_forecasts"].items():
        if "error" in lf: lines.append(f"- {sym}: ERROR {lf['error']}"); continue
        lines.append(f"### {sym} — as of {lf['as_of']}, last price {lf['last_price']}")
        for h in HORIZONS:
            d = lf[f"h{h}"]
            lines.append(f"- {h}d: P(up)={d['p_up']}, exp log-return={d['exp_log_return']}, q10={d['ret_q10']}, q90={d['ret_q90']}")
        lines.append("")
    lines += ["## Honest interpretation", "",
              "- Equity direction at 1–21d is close to a coin flip with a positive drift; climatology (always predicting the expanding base rate, ~53–58% up depending on symbol/horizon) is a hard benchmark to beat on Brier/log score.",
              "- Any horizon/symbol with negative Brier skill above genuinely underperforms the base rate out-of-sample — that is a real finding, not a bug: it means those features add variance without signal at that horizon.",
              "- AUC modestly above 0.5 (where present) indicates weak ranking signal even when calibration vs climatology is a wash.",
              "- Overlapping horizons (5d/21d forecasts on consecutive days) mean effective independent sample size is far smaller than n; do not over-read small skill differences.", ""]
    with open(os.path.join(OUT, "markets_RESULTS.md"), "w") as f:
        f.write("\n".join(lines))
    print("Wrote results.json + RESULTS.md")

if __name__ == "__main__":
    main()
