"""
sf — the Superforecaster agent interface (CLI).

Design contract for agent callers:
  * Every command prints exactly one JSON object to stdout (use --human for a
    readable rendering of `headline` only). No other stdout output, ever.
  * Exit code 0 = success, 2 = bad input, 3 = data unavailable.
  * `sf schema` prints the full machine-readable interface description —
    an agent that has never seen this tool can self-onboard from it.
  * Forecasts returned are PRODUCTION forecasts: the raw model number, the
    validated skill behind it, and the engine's shipped probability are all
    separate fields. Agents must consume `production_p`, never `raw_p`.

Examples:
  sf headline
  sf forecast --domain markets --symbol NVDA --horizon 21
  sf forecast --domain crypto --asset BTC --horizon 7
  sf forecast --domain predmarkets --slug xi-jinping-out-before-2027
  sf edge --p 0.62 --q-exec 0.55 --domain crypto
  sf pool --probs 0.60,0.70,0.55 --weights 0.16,0.14,0.10
  sf score --file forecasts.csv            # columns: y,p[,crowd]
  sf harness-demo                          # Argon fixture benchmark (synthetic self-test)
"""
from __future__ import annotations

import argparse, csv, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
from calibration import (brier_score, log_score, expected_calibration_error,  # noqa: E402
                         murphy_decomposition, peer_score, trimmed_mean_pool, log_odds_pool)
from engine import Superforecaster, shrink, skill_weight_from_skill  # noqa: E402
import alpha_kelly as ak  # noqa: E402

NSTAR_CURRENT = 0.00  # production: no validated net edge yet -> ~100% SKIP, capital preserved


def _load(name):
    with open(os.path.join(ROOT, "results", name)) as f:
        return json.load(f)


def _out(obj, code=0):
    print(json.dumps(obj, indent=2, default=str))
    raise SystemExit(code)


def cmd_headline(_a):
    rescue = _load("calibration_rescue.json")["streams"]
    def pooled(streams, key):
        num = sum(r["n_scored"] * r[key]["brier"] for r in streams.values())
        return num / sum(r["n_scored"] for r in streams.values())
    mk = {k: v for k, v in rescue.items() if k.startswith("markets/")}
    cr = {k: v for k, v in rescue.items() if k.startswith("crypto/")}
    _out({
        "n_star": {"score": NSTAR_CURRENT, "status": "no validated net edge; fleet SKIPs, capital preserved",
                   "definition": "CWR_bid * sqrt(365/mean_lock_days) * exp(-10*ECE) - 2*max(0, MDD-0.15)"},
        "brier_headline": {"production": 0.22846, "benchmark": 0.22904, "raw_models": 0.23355, "n": 37589},
        "domains": {
            "markets": {"production_brier": round(pooled(mk, "shrink"), 5), "benchmark_brier": round(pooled(mk, "clim"), 5),
                        "skill": round(1 - pooled(mk, "shrink") / pooled(mk, "clim"), 4)},
            "crypto": {"production_brier": round(pooled(cr, "shrink"), 5), "benchmark_brier": round(pooled(cr, "clim"), 5),
                       "skill": round(1 - pooled(cr, "shrink") / pooled(cr, "clim"), 4)},
            "predmarkets": {"production_brier": 0.09051, "peer_vs_market": 0.0, "n_test": 287},
        },
        "note": "Brier is now a gate/diagnostic. The fleet objective is n_star.",
    })


def _skill_gate(raw_p, prior, skill):
    sf = Superforecaster()
    res = sf.forecast(prior=prior, model_probs=[raw_p], model_skills=[skill])
    return res


def cmd_forecast(a):
    if a.domain == "markets":
        d = _load("markets_results.json")
        sym = a.symbol.upper()
        latest = d["latest_forecasts"].get(sym) or d["latest_forecasts"].get("^" + sym)
        if not latest:
            _out({"error": f"symbol {sym} not in cached results", "available": list(d["latest_forecasts"])}, 3)
        cell = d["symbols"][latest["symbol"]]["horizons"][str(a.horizon)]
        raw = latest[f"h{a.horizon}"]["p_up"]
        skill = cell["brier_skill_vs_clim"]
        prior = cell["base_rate_test"]
        res = _skill_gate(raw, prior, skill)
        _out({"domain": "markets", "symbol": latest["symbol"], "horizon_days": a.horizon,
              "as_of": latest["as_of"], "last_price": latest["last_price"],
              "raw_p": raw, "prior": prior, "validated_skill": skill,
              "production_p": res["p"], "n_components": res["n_components"],
              "note": res.get("note", "model shrunk toward prior by validated skill")})
    if a.domain == "crypto":
        d = _load("crypto_results.json")
        asset = a.asset.upper()
        cell = d["assets"][asset]["horizons"][str(a.horizon)]["up"]
        skill = cell.get("brier_skill_vs_climatology", cell.get("brier_skill_vs_clim", 0.0))
        crash = d["assets"][asset]["crash_7d"]
        _out({"domain": "crypto", "asset": asset, "horizon_days": a.horizon,
              "direction": {"validated_skill": skill,
                            "production": "climatology (direction model has no validated edge)"
                            if skill <= 0 else "shrunk model", "brier": cell["brier"],
                            "brier_climatology": cell["brier_climatology"], "auc": cell["auc"]},
              "crash_risk_7d": {"base_rate": crash["base_rate"], "brier": crash["brier"],
                                "brier_climatology": crash["brier_climatology"],
                                "skill": crash["brier_skill_vs_climatology"], "auc": crash["auc"],
                                "production": "validated — the fleet's one quant edge; use crash model output live via crypto.py"}})
    if a.domain == "predmarkets":
        import urllib.request
        url = f"https://gamma-api.polymarket.com/markets?slug={a.slug}"
        req = urllib.request.Request(url, headers={"User-Agent": "superforecaster-cli"})
        with urllib.request.urlopen(req, timeout=30) as r:
            rows = json.load(r)
        if not rows:
            _out({"error": f"no market found for slug {a.slug}"}, 3)
        m = rows[0]
        prices = json.loads(m["outcomePrices"])
        _out({"domain": "predmarkets", "question": m["question"], "slug": a.slug,
              "market_price_yes": float(prices[0]), "production_p": float(prices[0]),
              "volume": m.get("volume"), "end_date": m.get("endDate"),
              "note": "production = market price (validated: no adjustment rule beats it OOS); "
                      "trade only if your own calibrated p clears q_exec by >= 0.05 (see sf edge)"})
    _out({"error": "unknown domain"}, 2)


def cmd_edge(a):
    p = float(np.clip(a.p, 0.02, 0.98))
    q = float(np.clip(a.q_exec, 0.001, 0.999))
    kl = ak.kl_divergence_binary(p, q)
    lam = a.lam
    f_full = (p - q) / (1 - q) if p > q else (q - p) / q * -1  # signed: >0 long YES edge
    side = "BUY_YES" if p > q else "BUY_NO"
    p_side = p if p > q else 1 - p
    q_side = q if p > q else 1 - q
    f_side = max(0.0, (p_side - q_side) / (1 - q_side)) * lam
    if a.domain == "crypto":
        f_side *= 0.5  # CryptoBench dampener per N_star spec
    f_side = min(f_side, 0.15)  # cluster cap
    net_edge = abs(p - q)
    decision = "TRADE" if net_edge >= 0.05 else "SKIP"
    skip_reason = None if decision == "TRADE" else "INSUFFICIENT_NET_EDGE"
    # Gate D (fleet standard, r51/EC-14; alpha_kelly Gate 3 verbatim): SKIP when
    # p_side < 0.60 AND p_side/q_side < 1.35 (their "LR" is the probability ratio).
    lr = p_side / q_side
    gate_d_pass = (p_side >= 0.60) or (lr >= 1.35)
    if decision == "TRADE" and not gate_d_pass:
        decision, skip_reason = "SKIP", "LOW_CONVICTION (Gate D)"
    # Gate A (fleet standard, r51/EC-14): crypto requires on-chain verification.
    if decision == "TRADE" and a.domain == "crypto" and not a.onchain_verified:
        decision, skip_reason = "SKIP", "FAILED_CRYPTOBENCH_ONCHAIN_GATE (Gate A)"
    _out({"decision": decision, "skip_reason": skip_reason,
          "gate_d": {"p_side": round(p_side, 4), "likelihood_ratio": round(lr, 3), "pass": gate_d_pass},
          "gate_a_onchain_verified": bool(a.onchain_verified) if a.domain == "crypto" else None,
          "side": side if decision == "TRADE" else None,
          "p": p, "q_exec": q, "net_edge": round(net_edge, 4), "hurdle": 0.05,
          "kl_divergence_bits_edge": round(kl / np.log(2), 5), "kl_nats": round(kl, 5),
          "kelly_fraction_of_account": round(f_side, 4), "lambda": lam,
          "expected_log_growth_at_kelly_size": round(ak.expected_log_wealth_growth(p_side, q_side, f_side), 5),
          "note": "SKIP is the default; edge must survive spread/slippage/fees inside q_exec"})


def cmd_pool(a):
    probs = [float(x) for x in a.probs.split(",")]
    weights = [float(x) for x in a.weights.split(",")] if a.weights else None
    _out({"log_odds_pool": round(float(log_odds_pool(np.array([probs]), weights)[0]), 4),
          "trimmed_mean": round(float(trimmed_mean_pool(probs, 0.1)[0]), 4),
          "mean": round(float(np.mean(probs)), 4), "median": round(float(np.median(probs)), 4),
          "production_guidance": "track-record-weighted log-odds is the production pool (agents.py); "
                                 "extremization ships at 1.00 until learned per-domain"})


def cmd_score(a):
    rows = list(csv.DictReader(open(a.file)))
    y = np.array([float(r["y"]) for r in rows]); p = np.array([float(r["p"]) for r in rows])
    out = {"n": len(y), "brier": round(brier_score(y, p), 5), "log_score": round(log_score(y, p), 5),
           "ece": round(expected_calibration_error(y, p), 4), "murphy": murphy_decomposition(y, p)}
    if rows and "crowd" in rows[0]:
        c = np.array([float(r["crowd"]) for r in rows])
        out["peer_score_vs_crowd"] = round(peer_score(y, p, c), 5)
    _out(out)


def cmd_harness_demo(_a):
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ak.run_benchmark_simulation()
    text = buf.getvalue()
    _out({"benchmark": "alpha_kelly fixture suite (SYNTHETIC self-test — not performance evidence)",
          "disciplined_n_star": 0.1059, "naive_n_star": -0.1499,
          "log_tail": text[-1200:]})


def cmd_schema(_a):
    _out({"tool": "sf (Superforecaster agent interface)", "version": "1.4",
          "output_contract": "one JSON object on stdout; exit 0 ok / 2 bad input / 3 data unavailable",
          "commands": {
              "headline": "current N_star + Brier headline + per-domain skills",
              "forecast": {"args": "--domain markets|crypto|predmarkets; markets: --symbol --horizon 1|5|21; crypto: --asset BTC|ETH --horizon 1|7|30; predmarkets: --slug",
                           "returns": "raw_p, prior, validated_skill, production_p (consume production_p only)"},
              "edge": {"args": "--p --q-exec [--lam 0.25..0.5] [--domain crypto applies 0.5x dampener]",
                       "returns": "decision TRADE|SKIP (hurdle |p-q_exec|>=0.05; Gate D conviction p_side>=0.60 or p_side/q_side>=1.35; Gate A crypto on-chain), side, KL edge, capped Kelly fraction"},
              "pool": {"args": "--probs a,b,c [--weights w1,w2,w3]", "returns": "log-odds pool, trimmed mean, mean, median"},
              "score": {"args": "--file CSV with columns y,p[,crowd]", "returns": "brier, log_score, ece, murphy, peer_score"},
              "harness-demo": "runs the alpha_kelly fixture benchmark (synthetic)",
              "schema": "this document"}})


def main():
    ap = argparse.ArgumentParser(prog="sf")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("headline")
    f = sub.add_parser("forecast"); f.add_argument("--domain", required=True)
    f.add_argument("--symbol"); f.add_argument("--asset"); f.add_argument("--slug")
    f.add_argument("--horizon", type=int, default=21)
    e = sub.add_parser("edge"); e.add_argument("--p", type=float, required=True)
    e.add_argument("--q-exec", type=float, required=True); e.add_argument("--lam", type=float, default=0.35)
    e.add_argument("--domain", default="predmarkets")
    e.add_argument("--onchain-verified", action="store_true",
                   help="Gate A: required for crypto trades (CryptoBench on-chain checks passed)")
    pl = sub.add_parser("pool"); pl.add_argument("--probs", required=True); pl.add_argument("--weights")
    sc = sub.add_parser("score"); sc.add_argument("--file", required=True)
    sub.add_parser("harness-demo"); sub.add_parser("schema")
    a = ap.parse_args()
    {"headline": cmd_headline, "forecast": cmd_forecast, "edge": cmd_edge, "pool": cmd_pool,
     "score": cmd_score, "harness-demo": cmd_harness_demo, "schema": cmd_schema}[a.cmd](a)


if __name__ == "__main__":
    main()
