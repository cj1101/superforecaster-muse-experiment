"""
calibrate_oos.py — v2 integration experiment.

Question: the v1 direction models (markets + crypto) lose to climatology on
raw Brier. Can a CAUSAL calibration layer rescue them? Every transformation
below is fitted only on out-of-sample forecasts that precede the forecast
being transformed (expanding window, refit every BLOCK forecasts, minimum
MIN_CAL past forecasts before the layer activates).

Variants evaluated on the common post-activation subset:
  raw        the model's untouched probability
  clim       the per-forecast climatology known at the time (benchmark)
  platt      Platt scaling fitted on all past OOS forecasts
  shrink     p' = w*p + (1-w)*clim_t, w fitted on past OOS forecasts
  platt+shrink  Platt, then shrink (w fitted on Platt-transformed past —
             one-parameter fit, mild in-sample optimism in w only, noted)

Data: crypto uses the saved OOS prediction CSVs (data/crypto/predictions_*).
Markets re-runs markets.walk_forward() on the cached Yahoo CSVs in
data/markets/ (no network needed; download_yahoo reads the cache).

Output: results/calibration_rescue.json + printed table. Real numbers only.
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "superforecaster"))
from calibration import PlattScaler, brier_score, log_score          # noqa: E402
from engine import fit_shrink_weight                                  # noqa: E402

BLOCK, MIN_CAL = 63, 200


def evaluate_stream(y, p, clim):
    """Causally transform a stream of OOS forecasts; score post-activation."""
    y = np.asarray(y, float); p = np.asarray(p, float); clim = np.asarray(clim, float)
    n = len(y)
    out = {k: [] for k in ("raw", "clim", "platt", "shrink", "platt_shrink")}
    ys = []
    platt, w_shrink, w_after = None, 0.0, 0.0
    for start in range(0, n, BLOCK):
        end = min(start + BLOCK, n)
        if start >= MIN_CAL:
            past_y, past_p, past_c = y[:start], p[:start], clim[:start]
            platt = PlattScaler().fit(past_y, past_p)
            w_shrink = fit_shrink_weight(past_y, past_p, past_c)
            past_platt = platt.predict(past_p)
            w_after = fit_shrink_weight(past_y, past_platt, past_c)
            for i in range(start, end):
                pp = float(platt.predict(p[i:i + 1])[0])
                out["raw"].append(p[i]); out["clim"].append(clim[i])
                out["platt"].append(pp)
                out["shrink"].append(w_shrink * p[i] + (1 - w_shrink) * clim[i])
                out["platt_shrink"].append(w_after * pp + (1 - w_after) * clim[i])
                ys.append(y[i])
        # blocks before activation are calibration fuel only, never scored
    ys = np.array(ys)
    res = {"n_scored": int(len(ys))}
    for k, v in out.items():
        v = np.array(v)
        res[k] = {"brier": round(float(brier_score(ys, v)), 5),
                  "log_score": round(float(log_score(ys, v)), 5)}
    res["skill_platt_shrink_vs_clim"] = round(1 - res["platt_shrink"]["brier"] / res["clim"]["brier"], 4)
    res["skill_shrink_vs_clim"] = round(1 - res["shrink"]["brier"] / res["clim"]["brier"], 4)
    res["skill_platt_vs_clim"] = round(1 - res["platt"]["brier"] / res["clim"]["brier"], 4)
    return res


def main():
    results = {"method": f"causal expanding calibration, block={BLOCK}, min_cal={MIN_CAL}",
               "streams": {}}

    # ---- crypto: saved OOS predictions ----
    cdir = os.path.join(ROOT, "data", "crypto")
    for f in sorted(os.listdir(cdir)):
        if f.startswith("predictions_") and f.endswith(".csv"):
            df = pd.read_csv(os.path.join(cdir, f))
            name = "crypto/" + f.replace("predictions_", "").replace(".csv", "")
            results["streams"][name] = evaluate_stream(df["y"].values, df["p"].values, df["clim"].values)
            print(f"{name:28s} done", flush=True)

    # ---- markets: re-run walk-forward on cached data ----
    import markets as mk
    mdir = os.path.join(ROOT, "data", "markets")
    symmap = {"GSPC": "^GSPC", "QQQ": "QQQ", "NVDA": "NVDA", "TSM": "TSM"}
    for fname, sym in symmap.items():
        df = pd.read_csv(os.path.join(mdir, fname + ".csv"))
        df["date"] = pd.to_datetime(df["date"])
        feat = mk.build_features(df)
        for h in mk.HORIZONS:
            r = mk.walk_forward(feat, h)
            name = f"markets/{sym}_{h}d"
            results["streams"][name] = evaluate_stream(r["y"], r["p"], r["clim_p"])
            print(f"{name:28s} done", flush=True)

    out_path = os.path.join(ROOT, "results", "calibration_rescue.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'stream':28s} {'n':>6s} {'raw':>8s} {'clim':>8s} {'platt':>8s} {'shrink':>8s} {'p+s':>8s}  skill(p+s)")
    for name, r in results["streams"].items():
        print(f"{name:28s} {r['n_scored']:6d} {r['raw']['brier']:8.5f} {r['clim']['brier']:8.5f} "
              f"{r['platt']['brier']:8.5f} {r['shrink']['brier']:8.5f} {r['platt_shrink']['brier']:8.5f}  "
              f"{r['skill_platt_shrink_vs_clim']:+.4f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
