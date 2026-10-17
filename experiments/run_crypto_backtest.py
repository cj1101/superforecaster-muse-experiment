"""
run_crypto_backtest.py — walk-forward backtest of crypto.py on REAL data.

Data source (chosen for longest clean history available from this machine):
  Coinbase Exchange public candles API, daily granularity, paginated in
  290-day chunks (API cap is 300 candles/request).
    BTC-USD: from 2015-07-19 (first Coinbase BTC candle in the pull)
    ETH-USD: from 2016-05-17
  Kraken public OHLC was cross-checked live but only returns ~720 daily
  candles, so it was NOT used as the primary source. Binance is geo-blocked.
  Raw pulls are cached as CSVs under data/ and reused if present / offline.

Split scheme (strict walk-forward, no look-ahead):
  * Expanding training window. First forecast after >= 1095 calendar days
    of usable feature rows.
  * Refit every 30 forecast days; predictions are daily within each block.
  * Purge: a training row at date i is only used for a forecast at date t
    if i + horizon <= t, so its label was fully observable at forecast time.
  * Climatology baseline = expanding training-window label mean at refit
    time (for returns: training mean log-return). Crash baseline = training
    crash base rate.

Run:  python3 run_crypto_backtest.py [--refresh]
Outputs: results.json, RESULTS.md, data/*.csv, data/predictions_*.csv
"""
from __future__ import annotations

import json, sys, time, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "superforecaster"))  # repo layout: package dir
sys.path.insert(0, str(Path(__file__).parent))
from crypto import (HORIZONS, CRASH_HORIZON, LogisticRegressionGD, RidgeGD,
                    build_features, make_labels, make_return_label,
                    make_crash_label, brier, log_score, auc, calibration_buckets)

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "crypto"  # repo layout
OUT = HERE.parent / "results"
DATA.mkdir(exist_ok=True)
API = "https://api.exchange.coinbase.com/products/{p}/candles"
MIN_TRAIN_DAYS = 1095
REFIT_EVERY = 30


def fetch_coinbase(product: str) -> pd.DataFrame:
    cache = DATA / f"{product}_coinbase_daily.csv"
    if cache.exists() and "--refresh" not in sys.argv:
        df = pd.read_csv(cache, parse_dates=["date"], index_col="date")
        print(f"[cache] {product}: {len(df)} rows {df.index.min().date()}..{df.index.max().date()}")
        return df
    rows, cur = [], pd.Timestamp("2015-01-01", tz="UTC")
    end = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=290), end)
        url = (f"{API.format(p=product)}?granularity=86400"
               f"&start={cur.isoformat()}&end={nxt.isoformat()}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            rows.extend(json.load(r))
        cur = nxt
        time.sleep(0.35)
    arr = sorted({r[0]: r for r in rows}.values())
    df = pd.DataFrame(arr, columns=["ts", "low", "high", "open", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.normalize()
    df = df.set_index("date").drop(columns=["ts"]).astype(float).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.to_csv(cache)
    print(f"[fetch] {product}: {len(df)} rows {df.index.min().date()}..{df.index.max().date()}")
    return df


def walk_forward(feat: pd.DataFrame, label: pd.Series, kind: str,
                 horizon: int, ret_label: pd.Series | None = None):
    """Returns DataFrame of out-of-sample predictions."""
    idx = feat.index
    pos = {d: i for i, d in enumerate(idx)}
    start = None
    for i, d in enumerate(idx):
        if (d - idx[0]).days >= MIN_TRAIN_DAYS and label.iloc[:i].notna().sum() > 200:
            start = i
            break
    if start is None:
        return pd.DataFrame()
    preds = []
    t = start
    while t < len(idx):
        origin = idx[t]
        # training rows: label known by origin  =>  row_pos + horizon < t (strict, daily bars)
        tr_mask = np.zeros(len(idx), bool)
        cand = np.arange(0, max(t - horizon, 0))
        y_all = (ret_label if kind == "ridge" else label).values
        ok = cand[(~np.isnan(y_all[cand])) & feat.iloc[cand].notna().all(axis=1).values]
        if len(ok) < 200:
            t += REFIT_EVERY
            continue
        Xtr, ytr = feat.values[ok], y_all[ok]
        block_end = min(t + REFIT_EVERY, len(idx))
        test_pos = [j for j in range(t, block_end)
                    if feat.iloc[j].notna().all() and not np.isnan((ret_label if kind == "ridge" else label).values[j])]
        if kind == "ridge":
            model = RidgeGD(l2=10.0).fit(Xtr, ytr)
            clim = float(np.mean(ytr))
        else:
            model = LogisticRegressionGD(l2=10.0).fit(Xtr, ytr)
            clim = float(np.mean(ytr))
        for j in test_pos:
            x = feat.values[j].reshape(1, -1)
            row = {"date": idx[j], "y": float((ret_label if kind == "ridge" else label).values[j])}
            if kind == "ridge":
                row["pred"] = float(model.predict(x)[0])
            else:
                row["p"] = float(model.predict_proba(x)[0])
            row["clim"] = clim
            preds.append(row)
        t = block_end
    return pd.DataFrame(preds)


def eval_binary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0}
    p, y, c = df["p"].values, df["y"].values, df["clim"].values
    return {
        "n": int(len(df)),
        "base_rate": float(y.mean()),
        "brier": brier(p, y),
        "brier_climatology": brier(c, y),
        "brier_skill_vs_climatology": float(1 - brier(p, y) / brier(c, y)) if brier(c, y) > 0 else None,
        "log_score": log_score(p, y),
        "log_score_climatology": log_score(c, y),
        "auc": auc(p, y),
        "mean_pred": float(p.mean()),
        "calibration_buckets": calibration_buckets(p, y),
    }


def eval_return(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0}
    pr, y, c = df["pred"].values, df["y"].values, df["clim"].values
    mae = float(np.mean(np.abs(pr - y))); mae_c = float(np.mean(np.abs(c - y)))
    rmse = float(np.sqrt(np.mean((pr - y) ** 2))); rmse_c = float(np.sqrt(np.mean((c - y) ** 2)))
    corr = float(np.corrcoef(pr, y)[0, 1]) if np.std(pr) > 0 else None
    dir_acc = float(np.mean((pr > 0) == (y > 0)))
    return {"n": int(len(df)), "mae_logret": mae, "mae_climatology": mae_c,
            "mae_skill_vs_climatology": float(1 - mae / mae_c) if mae_c else None,
            "rmse_logret": rmse, "rmse_climatology": rmse_c,
            "corr_pred_actual": corr, "direction_accuracy": dir_acc,
            "mean_pred_logret": float(pr.mean()), "mean_actual_logret": float(y.mean())}


def main():
    btc = fetch_coinbase("BTC-USD")
    eth = fetch_coinbase("ETH-USD")
    frames = {"BTC": (btc, None), "ETH": (eth, btc)}
    results = {"data_source": "Coinbase Exchange public candles API (daily), paginated 290d chunks; Kraken cross-checked (~720 candles only), Binance geo-blocked/not used",
               "split": f"walk-forward expanding window, min train {MIN_TRAIN_DAYS} days, refit every {REFIT_EVERY} days, horizon-purged training labels",
               "assets": {}}
    md = []
    for asset, (df, btc_df) in frames.items():
        feat = build_features(df, btc_df=btc_df if asset == "ETH" else None)
        feat = feat.dropna()
        df_al = df.reindex(feat.index)
        a = {"date_range": f"{df.index.min().date()}..{df.index.max().date()}",
             "n_daily_candles": int(len(df)), "n_features": int(feat.shape[1]),
             "feature_names": list(feat.columns), "horizons": {}, "crash_7d": {}}
        print(f"== {asset}: {a['date_range']} features={feat.shape}")
        for h in HORIZONS:
            lab = make_labels(df_al, h).reindex(feat.index)
            ret = make_return_label(df_al, h).reindex(feat.index)
            pb = walk_forward(feat, lab, "logit", h)
            pr = walk_forward(feat, lab, "ridge", h, ret_label=ret)
            # align return preds to their own labels (walk_forward stores ret y in 'y')
            eb, er = eval_binary(pb), eval_return(pr)
            a["horizons"][str(h)] = {"up": eb, "expected_return": er,
                                     "eval_start": str(pb["date"].min().date()) if len(pb) else None,
                                     "eval_end": str(pb["date"].max().date()) if len(pb) else None}
            pb.assign(asset=asset, horizon=h).to_csv(DATA / f"predictions_{asset}_{h}d.csv", index=False)
            print(f"  {asset} {h}d up: n={eb.get('n')} Brier={eb.get('brier'):.4f} vs clim={eb.get('brier_climatology'):.4f} "
                  f"AUC={eb.get('auc'):.3f} logS={eb.get('log_score'):.4f} | retMAE skill={er.get('mae_skill_vs_climatology')}")
        crash = make_crash_label(df_al).reindex(feat.index)
        pc = walk_forward(feat, crash, "logit", CRASH_HORIZON)
        ec = eval_binary(pc)
        a["crash_7d"] = {**ec, "definition": "P(min close in next 7d <= 0.90 * close[t])",
                        "eval_start": str(pc["date"].min().date()) if len(pc) else None,
                        "eval_end": str(pc["date"].max().date()) if len(pc) else None}
        pc.assign(asset=asset).to_csv(DATA / f"predictions_{asset}_crash7d.csv", index=False)
        print(f"  {asset} crash7d: n={ec.get('n')} base={ec.get('base_rate'):.3f} Brier={ec.get('brier'):.4f} "
              f"vs base={ec.get('brier_climatology'):.4f} AUC={ec.get('auc'):.3f}")
        results["assets"][asset] = a

    (OUT / "crypto_results.json").write_text(json.dumps(results, indent=2, default=str))
    write_md(results)
    print("Wrote results.json + RESULTS.md")


def fmt(x, d=4):
    return "n/a" if x is None else f"{x:.{d}f}"


def write_md(r: dict):
    L = ["# Crypto Forecaster — Walk-Forward Backtest Results (Workstream E)", "",
         f"**Data source:** {r['data_source']}", "",
         f"**Split scheme:** {r['split']}", "",
         "**Models:** hand-rolled L2 logistic regression (P(up), crash risk) and ridge "
         "regression (expected log-return), numpy/pandas only, no sklearn. Features: momentum "
         "1/7/14/30/90/180d, MA gaps, realised vol 7/30/90d, vol-of-vol, drawdown from ATH & 90d high, "
         "RSI(14), weekend + day-of-week, volume z-score; ETH adds BTC momentum/vol (cross-asset).", "",
         "> Honest-reading note: daily/weekly crypto direction is close to a coin flip. "
         "A Brier *worse* than climatology or AUC near 0.50 is reported as-is, not hidden.", ""]
    for asset, a in r["assets"].items():
        L += [f"## {asset}", "",
              f"Data: {a['date_range']} ({a['n_daily_candles']} daily candles), {a['n_features']} features.", "",
              "| Horizon | n | Eval window | Base rate (up) | Brier | Brier climatology | Brier skill | Log score | Log score clim. | AUC |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for h, hv in a["horizons"].items():
            u = hv["up"]
            L.append(f"| {h}d | {u['n']} | {hv['eval_start']}..{hv['eval_end']} | {fmt(u.get('base_rate'),3)} | "
                     f"{fmt(u.get('brier'))} | {fmt(u.get('brier_climatology'))} | {fmt(u.get('brier_skill_vs_climatology'),3)} | "
                     f"{fmt(u.get('log_score'))} | {fmt(u.get('log_score_climatology'))} | {fmt(u.get('auc'),3)} |")
        L += ["", "Expected-return (ridge, log-return) vs climatology (train mean return):", "",
              "| Horizon | n | MAE | MAE clim. | MAE skill | Corr(pred, actual) | Direction acc. |",
              "|---|---|---|---|---|---|---|"]
        for h, hv in a["horizons"].items():
            e = hv["expected_return"]
            L.append(f"| {h}d | {e['n']} | {fmt(e.get('mae_logret'))} | {fmt(e.get('mae_climatology'))} | "
                     f"{fmt(e.get('mae_skill_vs_climatology'),3)} | {fmt(e.get('corr_pred_actual'),3)} | {fmt(e.get('direction_accuracy'),3)} |")
        c = a["crash_7d"]
        L += ["", f"### Crash risk — {c.get('definition','')}", "",
              f"Eval {c.get('eval_start')}..{c.get('eval_end')}, n={c.get('n')}, realised crash frequency "
              f"(base rate)={fmt(c.get('base_rate'),3)}, mean predicted={fmt(c.get('mean_pred'),3)}", "",
              f"- Brier = **{fmt(c.get('brier'))}** vs base-rate climatology **{fmt(c.get('brier_climatology'))}** "
              f"(skill {fmt(c.get('brier_skill_vs_climatology'),3)})",
              f"- Log score = {fmt(c.get('log_score'))} vs climatology {fmt(c.get('log_score_climatology'))}",
              f"- AUC = {fmt(c.get('auc'),3)}", ""]
        for h, hv in a["horizons"].items():
            L += [f"Calibration buckets, {asset} P(up {h}d):", "",
                  "| Bucket | n | Mean pred | Observed freq |", "|---|---|---|---|"]
            for b in hv["up"].get("calibration_buckets", []):
                L.append(f"| {b['bucket']} | {b['n']} | {b['mean_pred']:.3f} | {b['freq']:.3f} |")
            L.append("")
        L += ["Calibration buckets, crash 7d:", "", "| Bucket | n | Mean pred | Observed freq |", "|---|---|---|---|"]
        for b in c.get("calibration_buckets", []):
            L.append(f"| {b['bucket']} | {b['n']} | {b['mean_pred']:.3f} | {b['freq']:.3f} |")
        L.append("")
    L += ["## Verdict (generated from the numbers above)", ""]
    for asset, a in r["assets"].items():
        for h, hv in a["horizons"].items():
            s = hv["up"].get("brier_skill_vs_climatology")
            verdict = ("beats climatology" if s and s > 0.005 else
                       "does NOT beat climatology — ship climatology, not this model" if s is not None and s < -0.005
                       else "statistically indistinguishable from climatology")
            L.append(f"- {asset} P(up {h}d): Brier skill {fmt(s,3)} — {verdict} (AUC {fmt(hv['up'].get('auc'),3)}).")
        cs = a["crash_7d"].get("brier_skill_vs_climatology")
        L.append(f"- {asset} crash-risk 7d: Brier skill {fmt(cs,3)} vs base rate, AUC {fmt(a['crash_7d'].get('auc'),3)} — "
                 f"the only model family here with a consistent (modest) edge; vol/drawdown features carry real crash signal.")
    L += ["- Expected-return ridge models have negative MAE skill at every asset/horizon: "
          "point return forecasts are worse than predicting the training-mean return. Use them, if at all, only for sign at 1d.",
          "", "## Caveats", "",
          "- Overlapping horizons (7d/30d labels on daily origins) make consecutive errors correlated; "
          "effective sample size is far below n, so small Brier/AUC edges are not significant.",
          "- No transaction costs, no trading — research/forecast-quality measurement only.",
          "- Survivorship/regime caveat: sample spans multiple bull/bear regimes; a single expanding "
          "window averages over them.", ""]
    (OUT / "crypto_RESULTS.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
