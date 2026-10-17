"""
Workstream D — Prediction Markets forecaster (Superforecaster project).

Question: how calibrated are prediction-market prices, and can a simple,
fitted-out-of-sample adjustment rule beat the raw market price?

Data (all public, no credentials, research only — no betting):
  * Polymarket Gamma API (resolved markets):
        https://gamma-api.polymarket.com/markets?closed=true
        fetched in monthly end-date windows, ordered by volumeNum desc,
        so the sample is the highest-volume binary Yes/No markets of each
        month (Jan 2024 - Sep 2026), not the flood of tiny sports props
        that dominates an unfiltered "most recently closed" query.
  * Polymarket CLOB price history (pre-resolution price paths):
        https://clob.polymarket.com/prices-history?market=<clobTokenId>
            &interval=max&fidelity=1440
  * Kalshi public API was tried (status=settled works; the task brief's
    status=finalized returns bad_request). Its settled feed is dominated
    by multivariate combo markets with last_price/yes_bid = null, so it
    is NOT used for scoring — see RESULTS.md caveats. The attempt and its
    outcome are recorded in results.json.

PRICE PROXY — read this before trusting any number:
  Gamma's `outcomePrices` for a resolved market is [1,0] or [0,1] (the
  resolution itself), and `lastTradePrice` is ~0/1 at close. Neither is
  a forecast. The forecast price used here is therefore taken from the
  CLOB daily price history at fixed horizons BEFORE the market closed:
  snapshots at T-30d, T-14d, T-7d, T-3d and T-1d (last daily print at or
  before close - horizon, and no older than horizon + 14 days, so the
  print is reasonably fresh). Each (market, horizon) snapshot is one
  forecast sample with known time-to-expiry. The headline comparison
  uses the T-7d snapshot only (one sample per market); pooled snapshots
  are reported as a secondary analysis. Gamma `liquidityNum` is ~0 for
  resolved markets (liquidity is withdrawn at close), so lifetime
  `volumeNum` is used as the liquidity/trust proxy and is named as such.

Split: markets are split fit / test by closing date (earlier 60% of
markets = fit, later 40% = test). All rule parameters are chosen by
grid search on the FIT split only. The "best rule" named in the
calibration table is the rule with the lowest FIT Brier, not the
lowest test Brier — no test-set cherry-picking.

Adjustment rules (each a pure function price -> adjusted probability;
extra inputs are days_to_expiry and log10 volume):
  raw                 identity
  shrink_fixed        p' = 0.5 + s * (p - 0.5)
  logit_shrink        logit(p') = b * logit(p)         (temperature/Platt)
  logit_time          logit(p') = (b0 + b1*log1p(days)) * logit(p)
  logit_time_vol      logit(p') = (b0 + b1*log1p(days) + b2*z_logvol)
                                  * logit(p)
  bucket_recal        per-decile empirical recalibration fitted on fit,
                      shrunk toward identity by bucket sample size

Run:  python3 predmarkets.py            (uses/creates data/ cache)
      python3 predmarkets.py --refresh  (re-fetch everything)
Outputs: results.json and RESULTS.md next to this file, raw API
responses cached under data/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
GAMMA = "https://gamma-api.polymarket.com/markets"
CLOB = "https://clob.polymarket.com/prices-history"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2/markets"

HORIZONS_DAYS = [30, 14, 7, 3, 1]
PRIMARY_HORIZON = 7
TOP_PER_MONTH = 25
MIN_VOLUME = 50_000.0
WINDOWS = [(y, m) for y in (2024, 2025, 2026) for m in range(1, 13)
           if (y, m) <= (2026, 9)]
EPS = 1e-6


# ---------------------------------------------------------------- IO ----
def _get_json(url: str, tries: int = 3):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001 - retry any transient failure
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def fetch_gamma(refresh: bool = False) -> list[dict]:
    DATA.mkdir(parents=True, exist_ok=True)
    markets: dict[str, dict] = {}
    for (y, m) in WINDOWS:
        cache = DATA / f"gamma_{y}-{m:02d}.json"
        if cache.exists() and not refresh:
            rows = json.loads(cache.read_text())
        else:
            start = f"{y}-{m:02d}-01T00:00:00Z"
            ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
            end = f"{ny}-{nm:02d}-01T00:00:00Z"
            url = (f"{GAMMA}?limit={TOP_PER_MONTH}&closed=true"
                   f"&order=volumeNum&ascending=false"
                   f"&end_date_min={start}&end_date_max={end}")
            rows = _get_json(url)
            cache.write_text(json.dumps(rows))
            time.sleep(0.2)
        for r in rows:
            markets[r["id"]] = r
    return list(markets.values())


def fetch_kalshi_probe() -> dict:
    """Try Kalshi settled markets; record whether fields are usable."""
    cache = DATA / "kalshi_probe.json"
    if cache.exists():
        return json.loads(cache.read_text())
    out = {"attempted": True, "usable": False, "note": ""}
    try:
        d = _get_json(f"{KALSHI}?limit=100&status=settled")
        ms = d.get("markets", [])
        with_result = [x for x in ms if x.get("result") in ("yes", "no")]
        with_price = [x for x in with_result if x.get("last_price")]
        out.update(n_returned=len(ms), n_with_result=len(with_result),
                   n_with_last_price=len(with_price))
        out["note"] = ("status=finalized -> bad_request; status=settled works "
                       "but the feed is dominated by multivariate combo "
                       "markets and last_price is null, so Kalshi is not "
                       "scored. Polymarket is the scored source.")
        if with_price:
            out["usable"] = True
    except Exception as e:  # noqa: BLE001
        out["note"] = f"Kalshi fetch failed: {e}"
    cache.write_text(json.dumps(out, indent=2))
    return out


def fetch_clob(token: str, refresh: bool = False) -> list[dict]:
    cdir = DATA / "clob"
    cdir.mkdir(parents=True, exist_ok=True)
    cache = cdir / f"{token}.json"
    if cache.exists() and not refresh:
        try:
            return json.loads(cache.read_text())
        except Exception:  # noqa: BLE001
            pass
    try:
        d = _get_json(f"{CLOB}?market={token}&interval=max&fidelity=1440")
        hist = d.get("history", []) if isinstance(d, dict) else []
    except Exception:  # noqa: BLE001
        hist = []
    cache.write_text(json.dumps(hist))
    return hist


# ------------------------------------------------------------- dataset ----
def clean_markets(raw: list[dict]) -> pd.DataFrame:
    rows = []
    for m in raw:
        try:
            outcomes = json.loads(m.get("outcomes") or "[]")
            prices = json.loads(m.get("outcomePrices") or "[]")
            toks = json.loads(m.get("clobTokenIds") or "[]")
        except Exception:  # noqa: BLE001
            continue
        if outcomes != ["Yes", "No"] or len(toks) < 1:
            continue
        if prices not in (["1", "0"], ["0", "1"]):
            continue  # not a clean binary resolution (e.g. 0/0 legacy)
        vol = float(m.get("volumeNum") or 0.0)
        if vol < MIN_VOLUME:
            continue
        ct = pd.to_datetime(m.get("closedTime"), utc=True, errors="coerce")
        if pd.isna(ct):
            continue
        rows.append({
            "market_id": str(m["id"]),
            "question": m.get("question"),
            "category": m.get("category"),
            "closed_time": ct,
            "outcome": 1 if prices == ["1", "0"] else 0,  # 1 = Yes won
            "volume": vol,
            "liquidity_at_close": float(m.get("liquidityNum") or 0.0),
            "token_yes": toks[0],
        })
    df = pd.DataFrame(rows).sort_values("closed_time").reset_index(drop=True)
    return df


def snapshot_price(hist: list[dict], close_ts: float, horizon_d: int):
    """Last daily print at/before close-horizon, at most horizon+14d old."""
    if not hist:
        return None
    target = close_ts - horizon_d * 86400
    floor = target - 14 * 86400
    cand = [h for h in hist if floor <= h["t"] <= target]
    if not cand:
        return None
    h = max(cand, key=lambda x: x["t"])
    p = float(h["p"])
    return min(max(p, 0.001), 0.999)


def build_snapshots(markets: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    tokens = markets["token_yes"].tolist()
    with ThreadPoolExecutor(max_workers=8) as ex:
        hists = list(ex.map(lambda t: fetch_clob(t, refresh), tokens))
    rows = []
    for (_, m), hist in zip(markets.iterrows(), hists):
        close_ts = m["closed_time"].timestamp()
        for h_days in HORIZONS_DAYS:
            p = snapshot_price(hist, close_ts, h_days)
            if p is None:
                continue
            rows.append({
                "market_id": m["market_id"],
                "question": m["question"],
                "closed_time": m["closed_time"],
                "outcome": int(m["outcome"]),
                "price": p,
                "days_to_expiry": float(h_days),
                "volume": float(m["volume"]),
                "log10_volume": math.log10(max(m["volume"], 1.0)),
                "liquidity_at_close": float(m["liquidity_at_close"]),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------- rules ----
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def rule_raw(p, days, zlogvol, params):
    return np.asarray(p, dtype=float)


def rule_shrink_fixed(p, days, zlogvol, params):
    return 0.5 + params["s"] * (np.asarray(p, dtype=float) - 0.5)


def rule_logit_shrink(p, days, zlogvol, params):
    return _sigmoid(params["b"] * _logit(np.asarray(p, dtype=float)))


def _scale_time(params, days):
    return params["b0"] + params["b1"] * np.log1p(np.asarray(days, dtype=float))


def rule_logit_time(p, days, zlogvol, params):
    return _sigmoid(_scale_time(params, days) * _logit(np.asarray(p, dtype=float)))


def rule_logit_time_vol(p, days, zlogvol, params):
    scale = _scale_time(params, days) + params["b2"] * np.asarray(zlogvol, dtype=float)
    return _sigmoid(scale * _logit(np.asarray(p, dtype=float)))


def rule_bucket_recal(p, days, zlogvol, params):
    p = np.asarray(p, dtype=float)
    edges = np.array(params["edges"])
    vals = np.array(params["values"])  # adjusted prob per bucket
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(vals) - 1)
    return vals[idx]


RULE_FUNCS = {
    "raw": rule_raw,
    "shrink_fixed": rule_shrink_fixed,
    "logit_shrink": rule_logit_shrink,
    "logit_time": rule_logit_time,
    "logit_time_vol": rule_logit_time_vol,
    "bucket_recal": rule_bucket_recal,
}


def apply_rule(name, df, params, z_mean, z_sd):
    z = (df["log10_volume"].to_numpy() - z_mean) / z_sd
    return RULE_FUNCS[name](df["price"].to_numpy(), df["days_to_expiry"].to_numpy(), z, params)


# ------------------------------------------------------------- scoring ----
def brier(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def log_score(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    out, n = 0.0, len(p)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            out += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(out)


def fit_rules(fit: pd.DataFrame):
    """Grid-search every rule's parameters on the fit split (Brier)."""
    y = fit["outcome"].to_numpy()
    p = fit["price"].to_numpy()
    days = fit["days_to_expiry"].to_numpy()
    z_mean = float(fit["log10_volume"].mean())
    z_sd = float(fit["log10_volume"].std() or 1.0)
    z = (fit["log10_volume"].to_numpy() - z_mean) / z_sd

    def score(fn, params):
        return brier(fn(p, days, z, params), y)

    fitted = {"raw": {}}
    best = (None, 1e9)
    for s in np.arange(0.70, 1.151, 0.025):
        v = score(rule_shrink_fixed, {"s": float(s)})
        if v < best[1]:
            best = ({"s": float(s)}, v)
    fitted["shrink_fixed"] = best[0]

    best = (None, 1e9)
    for b in np.arange(0.60, 1.301, 0.02):
        v = score(rule_logit_shrink, {"b": float(b)})
        if v < best[1]:
            best = ({"b": float(b)}, v)
    fitted["logit_shrink"] = best[0]

    best = (None, 1e9)
    for b0 in np.arange(0.60, 1.21, 0.05):
        for b1 in np.arange(-0.10, 0.201, 0.025):
            v = score(rule_logit_time, {"b0": float(b0), "b1": float(b1)})
            if v < best[1]:
                best = ({"b0": float(b0), "b1": float(b1)}, v)
    fitted["logit_time"] = best[0]

    best = (None, 1e9)
    for b0 in np.arange(0.70, 1.16, 0.05):
        for b1 in np.arange(-0.10, 0.151, 0.05):
            for b2 in np.arange(-0.10, 0.151, 0.05):
                v = score(rule_logit_time_vol,
                          {"b0": float(b0), "b1": float(b1), "b2": float(b2)})
                if v < best[1]:
                    best = ({"b0": float(b0), "b1": float(b1), "b2": float(b2)}, v)
    fitted["logit_time_vol"] = best[0]

    # bucket recalibration: decile buckets, empirical rate shrunk to identity
    edges = np.quantile(p, np.linspace(0, 1, 11))
    edges[0], edges[-1] = 0.0, 1.0
    vals = []
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
    for k in range(10):
        m = idx == k
        if m.sum() == 0:
            vals.append(float(np.mean(p[(idx == max(k - 1, 0))])) if k else 0.05)
            continue
        rate, mean_p, n_k = float(y[m].mean()), float(p[m].mean()), int(m.sum())
        w = n_k / (n_k + 25.0)  # shrink toward raw price when bucket is thin
        vals.append(float(np.clip(w * rate + (1 - w) * mean_p, 0.001, 0.999)))
    fitted["bucket_recal"] = {"edges": [float(e) for e in edges],
                              "values": vals}
    return fitted, z_mean, z_sd


def calibration_table(df, best_rule, params, z_mean, z_sd):
    edges = [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1.0001]
    labels = ["0-.05", ".05-.1", ".1-.2", ".2-.3", ".3-.4", ".4-.5", ".5-.6",
              ".6-.7", ".7-.8", ".8-.9", ".9-.95", ".95-1"]
    adj = apply_rule(best_rule, df, params, z_mean, z_sd)
    rows = []
    for i, lab in enumerate(labels):
        m = (df["price"].to_numpy() >= edges[i]) & (df["price"].to_numpy() < edges[i + 1])
        if m.sum() == 0:
            continue
        rows.append({
            "bucket": lab, "n": int(m.sum()),
            "mean_market_price": round(float(df["price"].to_numpy()[m].mean()), 4),
            "empirical_freq": round(float(df["outcome"].to_numpy()[m].mean()), 4),
            "mean_best_rule_price": round(float(adj[m].mean()), 4),
        })
    return rows


# ---------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    kalshi = fetch_kalshi_probe()
    raw = fetch_gamma(refresh=args.refresh)
    markets = clean_markets(raw)
    print(f"clean binary markets: {len(markets)}", file=sys.stderr)
    snaps = build_snapshots(markets, refresh=args.refresh)
    if snaps.empty:
        raise SystemExit("no snapshots built — CLOB history unavailable")
    snaps.to_csv(DATA / "snapshots.csv", index=False)
    markets_out = markets.drop(columns=["token_yes"])
    markets_out.to_csv(DATA / "markets.csv", index=False)

    # split MARKETS by closing date: earliest 60% fit, latest 40% test
    mids = markets.sort_values("closed_time")["market_id"].tolist()
    cut = int(0.6 * len(mids))
    fit_ids, test_ids = set(mids[:cut]), set(mids[cut:])
    cutoff_date = str(markets.sort_values("closed_time")["closed_time"].iloc[cut])

    analyses = {}
    for name, sub in [("primary_T7", snaps[snaps["days_to_expiry"] == PRIMARY_HORIZON]),
                      ("pooled_all_horizons", snaps)]:
        fit = sub[sub["market_id"].isin(fit_ids)]
        test = sub[sub["market_id"].isin(test_ids)]
        if len(fit) < 50 or len(test) < 50:
            analyses[name] = {"n_fit": len(fit), "n_test": len(test),
                              "note": "too few snapshots, skipped"}
            continue
        params, z_mean, z_sd = fit_rules(fit)
        scores = {}
        for rname in RULE_FUNCS:
            pf = apply_rule(rname, fit, params[rname], z_mean, z_sd)
            pt = apply_rule(rname, test, params[rname], z_mean, z_sd)
            scores[rname] = {
                "params": params[rname],
                "fit_brier": round(brier(pf, fit["outcome"]), 5),
                "fit_log": round(log_score(pf, fit["outcome"]), 5),
                "test_brier": round(brier(pt, test["outcome"]), 5),
                "test_log": round(log_score(pt, test["outcome"]), 5),
                "test_ece": round(ece(pt, test["outcome"]), 5),
            }
        best_rule = min((r for r in RULE_FUNCS if r != "raw"),
                        key=lambda r: scores[r]["fit_brier"])
        analyses[name] = {
            "n_fit_snapshots": len(fit), "n_test_snapshots": len(test),
            "n_fit_markets": int(fit["market_id"].nunique()),
            "n_test_markets": int(test["market_id"].nunique()),
            "base_rate_test": round(float(test["outcome"].mean()), 4),
            "best_rule_by_fit": best_rule,
            "best_rule_beats_raw_on_test_brier":
                bool(scores[best_rule]["test_brier"] < scores["raw"]["test_brier"]),
            "scores": scores,
            "calibration_table_test": calibration_table(
                test, best_rule, params[best_rule], z_mean, z_sd),
            "raw_calibration_by_horizon_test": {
                str(int(h)): {"n": int((test["days_to_expiry"] == h).sum()),
                              "brier": round(brier(
                                  test.loc[test["days_to_expiry"] == h, "price"],
                                  test.loc[test["days_to_expiry"] == h, "outcome"]), 5)}
                for h in sorted(test["days_to_expiry"].unique())
            } if name == "pooled_all_horizons" else {},
        }

    results = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source": "Polymarket Gamma + CLOB (public APIs, fetched live)",
        "kalshi_probe": kalshi,
        "price_proxy": ("CLOB daily history: last print at/before close-horizon "
                        "(freshness window horizon+14d); horizons 30/14/7/3/1d; "
                        "headline = T-7d. NOT Gamma outcomePrices/lastTradePrice "
                        "(those are the resolution / ~0-1 at close)."),
        "filters": {"top_per_month_by_volume": TOP_PER_MONTH,
                    "min_volume_usd": MIN_VOLUME,
                    "windows": "2024-01..2026-09 monthly end-date windows",
                    "binary_yes_no_only": True,
                    "clean_resolution_only": "outcomePrices in {[1,0],[0,1]}"},
        "n_gamma_rows_fetched": len(raw),
        "n_clean_markets": len(markets),
        "n_snapshots_total": len(snaps),
        "markets_date_range": [str(markets["closed_time"].min()),
                               str(markets["closed_time"].max())],
        "split": {"method": "markets sorted by closed_time; earliest 60% fit, latest 40% test",
                  "cutoff_closed_time": cutoff_date,
                  "n_fit_markets": len(fit_ids), "n_test_markets": len(test_ids)},
        "total_volume_usd_clean_markets": round(float(markets["volume"].sum()), 2),
        "analyses": analyses,
        "caveats": [
            "Liquidity at close is ~0 for resolved markets; lifetime volume is the trust/liquidity proxy.",
            "Sample is the top-25-by-volume binary markets per month: it over-represents politics/crypto/sports event markets that attracted volume, and under-represents thin markets (survivorship/selection).",
            "Daily CLOB fidelity: the snapshot print can be up to 14 days older than its nominal horizon for illiquid stretches (freshness window); horizons are nominal, not exact.",
            "Snapshots of the same market at different horizons are correlated; pooled analysis counts them as samples, primary T-7 analysis does not.",
            "Markets within an event (e.g. mutually exclusive Fed-decision buckets) are correlated; no event-level clustering adjustment is applied.",
            "Kalshi not scored: settled feed dominated by multivariate combos with null last_price (see kalshi_probe).",
        ],
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    write_md(results)
    print(json.dumps({k: v for k, v in results.items()
                      if k in ("n_clean_markets", "n_snapshots_total")}, indent=2))
    for name, a in analyses.items():
        if "scores" in a:
            print(name, "raw test Brier", a["scores"]["raw"]["test_brier"],
                  "| best-by-fit:", a["best_rule_by_fit"],
                  a["scores"][a["best_rule_by_fit"]]["test_brier"])


def write_md(r: dict):
    a7 = r["analyses"].get("primary_T7", {})
    ap = r["analyses"].get("pooled_all_horizons", {})
    def beats(a):
        if "scores" not in a:
            return "n/a"
        return "YES" if a["best_rule_beats_raw_on_test_brier"] else "NO"
    lines = [
        "# Workstream D — Prediction Markets: calibration & adjustment rules",
        "",
        f"Generated {r['generated_at_utc']} from live public APIs. Research only — no betting.",
        "",
        "## Headline",
        "",
        f"- Clean resolved binary markets analysed: **{r['n_clean_markets']}** "
        f"({r['markets_date_range'][0]} → {r['markets_date_range'][1]}), "
        f"total lifetime volume ${r['total_volume_usd_clean_markets']:,.0f}.",
        f"- Forecast samples (market × horizon snapshots): **{r['n_snapshots_total']}**.",
        f"- Split by closing date: {r['split']['n_fit_markets']} fit markets / "
        f"{r['split']['n_test_markets']} test markets (cutoff {r['split']['cutoff_closed_time']}).",
    ]
    if "scores" in a7:
        s = a7["scores"]
        lines += [
            f"- **Primary (T-7d price): raw market test Brier = {s['raw']['test_brier']}, "
            f"log score = {s['raw']['test_log']}.** Best rule selected on fit "
            f"({a7['best_rule_by_fit']}): test Brier = {s[a7['best_rule_by_fit']]['test_brier']}, "
            f"log score = {s[a7['best_rule_by_fit']]['test_log']}. "
            f"Beats raw out-of-sample: **{beats(a7)}**.",
        ]
    if "scores" in ap:
        s = ap["scores"]
        lines += [
            f"- Pooled horizons: raw test Brier = {s['raw']['test_brier']}; "
            f"best-by-fit rule ({ap['best_rule_by_fit']}) test Brier = "
            f"{s[ap['best_rule_by_fit']]['test_brier']}. Beats raw: **{beats(ap)}**.",
        ]
    lines += ["", "## Price proxy (exactly what was used)", "", r["price_proxy"], "",
              "## Scores", ""]
    for name, a in r["analyses"].items():
        if "scores" not in a:
            lines += [f"### {name}", "", f"Skipped: {a.get('note')} "
                      f"(fit {a.get('n_fit')}, test {a.get('n_test')})", ""]
            continue
        lines += [f"### {name}  (fit n={a['n_fit_snapshots']}, test n={a['n_test_snapshots']}, "
                  f"test base rate={a['base_rate_test']})", "",
                  "| rule | params (fit only) | fit Brier | test Brier | test log score | test ECE |",
                  "|---|---|---|---|---|---|"]
        for rn, sc in a["scores"].items():
            prm = ", ".join(f"{k}={v}" for k, v in sc["params"].items()
                            if k not in ("edges", "values")) or "—"
            if rn == "bucket_recal":
                prm = "decile recal, shrink n/(n+25)"
            lines.append(f"| {rn} | {prm} | {sc['fit_brier']} | {sc['test_brier']} | "
                         f"{sc['test_log']} | {sc['test_ece']} |")
        lines += ["", f"Best rule by FIT Brier: **{a['best_rule_by_fit']}** "
                  f"(beats raw on test: {beats(a)})", ""]
        if a.get("calibration_table_test"):
            lines += ["Calibration on test (market vs best rule):", "",
                      "| price bucket | n | mean market price | empirical freq | mean best-rule price |",
                      "|---|---|---|---|---|"]
            for row in a["calibration_table_test"]:
                lines.append(f"| {row['bucket']} | {row['n']} | {row['mean_market_price']} | "
                             f"{row['empirical_freq']} | {row['mean_best_rule_price']} |")
            lines.append("")
        if a.get("raw_calibration_by_horizon_test"):
            lines += ["Raw market Brier by horizon (test, pooled): "
                      + ", ".join(f"T-{h}d: {v['brier']} (n={v['n']})"
                                  for h, v in a["raw_calibration_by_horizon_test"].items()), ""]
    lines += ["## Kalshi", "", r["kalshi_probe"].get("note", ""), "",
              "## Caveats", ""] + [f"- {c}" for c in r["caveats"]] + [
              "", "## Files", "",
              "- `predmarkets.py` — fetch + dataset + rules + scoring (numpy/pandas only).",
              "- `results.json` — machine-readable version of everything above.",
              "- `data/` — cached raw Gamma JSON per month, per-token CLOB histories, "
              "`markets.csv`, `snapshots.csv`.", ""]
    (HERE / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
