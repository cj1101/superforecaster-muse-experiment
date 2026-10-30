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
  sf edge --p 0.42 --q-exec 0.40 --q-exec-side no   # --q-exec prices NO (NO-book VWAP)
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


# SF-BL-9580: --domain canonicalisation. Pre-fix, --domain was a free-form
# string compared verbatim (`a.domain == "crypto"`), so any variant —
# "Crypto", "CRYPTO", " crypto", "Cryptocurrency" — silently behaved as a
# NON-crypto domain in cmd_edge: Gate A (on-chain verification) never fired
# and the x0.5 CryptoBench Kelly dampener never applied (live repro:
# `edge --p 0.80 --q-exec 0.50 --domain Crypto` -> TRADE kelly 0.15, while
# the identical `--domain crypto` call SKIPped on Gate A). cmd_forecast
# failed safe on variants (exit-2 "unknown domain") but inconsistently.
# Now both commands canonicalise FIRST: strip whitespace, casefold, drop
# "-"/"_"/space separators, map the alias table below; anything unmapped
# is REJECTED with one JSON error object, exit 2 — an unknown label is
# never again silently read as a verified, undampened non-crypto domain.
_DOMAIN_ALIASES = {
    "crypto": "crypto", "cryptocurrency": "crypto",
    "cryptocurrencies": "crypto", "cryptobench": "crypto",
    "markets": "markets", "market": "markets",
    "predmarkets": "predmarkets", "predmarket": "predmarkets",
    "predictionmarkets": "predmarkets", "predictionmarket": "predmarkets",
}


def _canonical_domain(raw):
    """Canonical domain for a --domain value, or None if unknown."""
    if raw is None:
        return None
    key = str(raw).strip().casefold().replace("-", "").replace("_", "").replace(" ", "")
    return _DOMAIN_ALIASES.get(key)


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
    domain = _canonical_domain(a.domain)  # SF-BL-9580: see _canonical_domain
    if domain is None:
        _out({"error": f"unknown domain {a.domain!r}; expected markets, crypto or predmarkets "
                       "(case/whitespace-insensitive; crypto aliases: cryptocurrency, cryptobench)"}, 2)
    if domain == "markets":
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
    if domain == "crypto":
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
    if domain == "predmarkets":
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
    # SF-BL-9580: canonicalise --domain before ANY gate/dampener decision;
    # an unknown label is rejected (exit 2, one JSON error), never traded
    # as a silent non-crypto domain. Canonical inputs are unchanged.
    domain = _canonical_domain(a.domain)
    if domain is None:
        _out({"error": f"unknown domain {a.domain!r}; expected markets, crypto or predmarkets "
                       "(case/whitespace-insensitive; crypto aliases: cryptocurrency, cryptobench)"}, 2)
    # SF-BL-9717: reject non-finite --p / --q-exec at the CLI boundary,
    # BEFORE either clip. np.clip passes NaN through and maps +/-inf to
    # the clip boundary, so pre-fix a non-finite input was never bad
    # input here: as filed it exited 0 with bare NaN tokens for p,
    # net_edge and both Gate D fields (invalid strict JSON), and after
    # SF-BL-9507 landed downstream it died as an uncaught ValueError
    # traceback (exit 1, no JSON), while +/-inf still silently TRADED
    # as the clipped boundary price (e.g. --p inf -> TRADE at p 0.98).
    # On any non-finite value: exactly one JSON error object, exit 2,
    # per this CLI's documented contract — no decision is computed.
    # Finite inputs (including finite out-of-range values, which keep
    # their pre-existing clip behaviour) are unchanged.
    if not np.isfinite(a.p):
        _out({"error": f"--p must be finite; got {a.p!r}"}, 2)
    if not np.isfinite(a.q_exec):
        _out({"error": f"--q-exec must be finite; got {a.q_exec!r}"}, 2)
    # SF-BL-10659: reject a non-finite or out-of-band --lam at the CLI
    # boundary, BEFORE any sizing. cmd_edge reimplements sizing inline
    # (f_side = max(0, (p_side - q_side)/(1 - q_side)) * lam) and never
    # calls nstar.kelly_fraction — the function SF-BL-9778 guarded — so
    # pre-fix lam flowed in raw: lam=-1.0 exited 0 as a TRADE with
    # kelly_fraction_of_account -0.2083 on the fixture (p 0.62,
    # q_exec 0.52; only SKIP zeroes the fraction), lam=2.0 / inf /
    # 1e999 silently sized at the 0.15 cluster cap (inf also emitted a
    # bare Infinity token for lambda — invalid strict JSON), lam=nan
    # died downstream as an uncaught ValueError traceback on a TRADE
    # path (exit 1, no JSON) and emitted a bare NaN token on a SKIP
    # path, and finite out-of-band values (0, 0.24, 0.5000001) sized
    # silently. The schema text said 0.25..0.5 but nothing enforced
    # it. On any violation: exactly one JSON error object, exit 2,
    # per this CLI's documented contract — no decision is computed,
    # on TRADE and SKIP paths alike, and the value is never clamped
    # into the band (clamping would silently size on a lambda the
    # caller never validly held; cf. SF-BL-9778). Boundaries 0.25
    # and 0.50 are inclusive (mandated grid points). In-band inputs
    # (including the 0.35 default) are unchanged.
    if not np.isfinite(a.lam) or not (0.25 <= a.lam <= 0.50):
        _out({"error": f"--lam must be finite and in [0.25, 0.50]; got {a.lam!r}"}, 2)
    p = float(np.clip(a.p, 0.02, 0.98))
    q = float(np.clip(a.q_exec, 0.001, 0.999))
    # SF-BL-9323: --q-exec is SIDE-TAGGED. Default ("yes"): q is the YES-side
    # execution price — the historical convention, previously stated nowhere
    # in cmd_schema or docs/AGENT_INTERFACE.md. With --q-exec-side no, q is
    # the NO-side execution price (e.g. the VWAP from walking a NO-token
    # book with nstar.vwap_for_dollars) and is used AS-IS as q_side for
    # BUY_NO, never complemented. Before this tag, a NO-side q_exec was read
    # as a YES price and complemented a second time: Gate D divided by the
    # double-complemented price and Kelly sized off it (fixture: p=0.45 with
    # a NO VWAP of 0.60 — an overpriced NO book — TRADED at kelly 0.0875).
    # A tagged NO call never flips to BUY_YES: 1 - q_no is NOT the YES VWAP
    # (spread/fees make q_exec_yes + q_exec_no > 1; cf. nstar.exec_prices),
    # so a negative NO edge SKIPs instead of inventing a YES price.
    q_exec_side = getattr(a, "q_exec_side", "yes")
    lam = a.lam
    if q_exec_side == "no":
        side = "BUY_NO"
        p_side = 1 - p
        q_side = q
        kl = ak.kl_divergence_binary(p_side, q_side)
        net_edge = p_side - q_side  # signed NO-side edge
    else:
        kl = ak.kl_divergence_binary(p, q)
        f_full = (p - q) / (1 - q) if p > q else (q - p) / q * -1  # signed: >0 long YES edge
        side = "BUY_YES" if p > q else "BUY_NO"
        p_side = p if p > q else 1 - p
        q_side = q if p > q else 1 - q
        net_edge = abs(p - q)
    f_side = max(0.0, (p_side - q_side) / (1 - q_side)) * lam
    if domain == "crypto":
        f_side *= 0.5  # CryptoBench dampener per N_star spec
    f_side = min(f_side, 0.15)  # cluster cap
    decision = "TRADE" if net_edge >= 0.05 else "SKIP"
    skip_reason = None if decision == "TRADE" else "INSUFFICIENT_NET_EDGE"
    # Gate D (fleet standard, r51/EC-14; alpha_kelly Gate 3 verbatim): SKIP when
    # p_side < 0.60 AND p_side/q_side < 1.35 (their "LR" is the probability ratio).
    lr = p_side / q_side
    gate_d_pass = (p_side >= 0.60) or (lr >= 1.35)
    if decision == "TRADE" and not gate_d_pass:
        decision, skip_reason = "SKIP", "LOW_CONVICTION (Gate D)"
    # Gate A (fleet standard, r51/EC-14): crypto requires on-chain verification.
    if decision == "TRADE" and domain == "crypto" and not a.onchain_verified:
        decision, skip_reason = "SKIP", "FAILED_CRYPTOBENCH_ONCHAIN_GATE (Gate A)"
    # EC-09: a SKIP sizes zero. f_side was computed before the gate decisions
    # above, so without this reset every SKIP branch (insufficient edge,
    # Gate D, Gate A) leaked a nonzero Kelly fraction — and a nonzero
    # expected log growth at that size — into the JSON contract.
    if decision == "SKIP":
        f_side = 0.0
    _out({"decision": decision, "skip_reason": skip_reason,
          "gate_d": {"p_side": round(p_side, 4), "likelihood_ratio": round(lr, 3), "pass": gate_d_pass},
          "gate_a_onchain_verified": bool(a.onchain_verified) if domain == "crypto" else None,
          "side": side if decision == "TRADE" else None,
          "p": p, "q_exec": q, "q_exec_side": q_exec_side, "q_side": round(q_side, 4),
          "net_edge": round(net_edge, 4), "hurdle": 0.05,
          "kl_divergence_bits_edge": round(kl / np.log(2), 5), "kl_nats": round(kl, 5),
          "kelly_fraction_of_account": round(f_side, 4), "lambda": lam,
          "expected_log_growth_at_kelly_size": round(ak.expected_log_wealth_growth(p_side, q_side, f_side), 5),
          "note": "SKIP is the default; edge must survive spread/slippage/fees inside q_exec"})


def _parse_pool_floats(raw):
    """Parse a comma-separated float list for cmd_pool (SF-BL-9544).

    Returns the list of floats, or None if any token is not a number —
    the caller turns that into the contract's exit-2 JSON error instead
    of the pre-fix uncaught ValueError traceback (exit 1, no JSON)."""
    try:
        return [float(x) for x in raw.split(",")]
    except ValueError:
        return None


def cmd_pool(a):
    # SF-BL-9544: validate at the CLI boundary BEFORE any pool is computed.
    # Pre-fix, --probs/--weights were parsed as bare floats and pooled raw:
    # `--probs 1.5,0.6` exited 0 publishing trimmed_mean/mean/median 1.05 —
    # a "probability" above 1 that a downstream edge command would clip
    # silently into a trade decision — while non-finite, non-numeric and
    # mis-weighted inputs died as uncaught tracebacks (exit 1, no JSON),
    # breaking this CLI's one-JSON-object / exit-2 contract. On any
    # violation: exactly one JSON error object, exit 2, no pooled figure.
    probs = _parse_pool_floats(a.probs)
    if not probs:
        _out({"error": f"--probs must be a non-empty comma-separated list of numbers; got {a.probs!r}"}, 2)
    bad = [p for p in probs if not np.isfinite(p) or p < 0.0 or p > 1.0]
    if bad:
        _out({"error": f"each --probs value must be finite and in [0, 1]; got {bad[0]!r}"}, 2)
    weights = None
    if a.weights is not None:
        weights = _parse_pool_floats(a.weights)
        if weights is None or len(weights) != len(probs):
            _out({"error": f"--weights must be numbers matching --probs in length ({len(probs)}); got {a.weights!r}"}, 2)
        bad_w = [w for w in weights if not np.isfinite(w) or w < 0.0]
        if bad_w:
            _out({"error": f"each --weights value must be finite and >= 0; got {bad_w[0]!r}"}, 2)
        if not any(w > 0.0 for w in weights):
            _out({"error": "--weights must have a positive sum; all-zero weights have no defined pool"}, 2)
    _out({"log_odds_pool": round(float(log_odds_pool(np.array([probs]), weights)[0]), 4),
          "trimmed_mean": round(float(trimmed_mean_pool(probs, 0.1)[0]), 4),
          "mean": round(float(np.mean(probs)), 4), "median": round(float(np.median(probs)), 4),
          "production_guidance": "track-record-weighted log-odds is the production pool (agents.py); "
                                 "extremization ships at 1.00 until learned per-domain"})


def cmd_score(a):
    rows = list(csv.DictReader(open(a.file)))
    # SF-BL-9747: an empty score file (header-only, zero-byte, or only
    # blank lines — csv yields zero data rows for all three) has no
    # defined score: brier/log/uncertainty are means over n=0. Pre-fix
    # cmd_score scored it anyway and exited 0 with bare NaN literals
    # for brier/log_score and murphy's residual/base_rate — invalid
    # strict JSON that reads downstream as a real (perfectly empty)
    # evaluation. Reject at the CLI boundary, like cmd_pool
    # (SF-BL-9544) and cmd_edge (SF-BL-9717): exactly one JSON error
    # object, exit 2, no score computed. Non-empty files take the
    # identical code path below, byte-identical.
    if not rows:
        _out({"error": "--file contains no forecast rows; at least one y,p data row "
                       "is required (empty or header-only file)"}, 2)
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
              "forecast": {"args": "--domain markets|crypto|predmarkets (SF-BL-9580: canonicalised — case/whitespace-insensitive, '-'/'_' ignored, crypto aliases cryptocurrency/cryptobench accepted; any other label is rejected, exit 2, one JSON error object); markets: --symbol --horizon 1|5|21; crypto: --asset BTC|ETH --horizon 1|7|30; predmarkets: --slug",
                           "returns": "raw_p, prior, validated_skill, production_p (consume production_p only)"},
              "edge": {"args": "--p --q-exec — both must be finite (SF-BL-9717): a non-finite --p or --q-exec (nan/inf/-inf, any spelling) is rejected exit 2 with one JSON error object and no decision [--q-exec-side yes|no: --q-exec is the YES-side execution price unless tagged no, in which case it is the NO-side price (e.g. a NO-book VWAP) used as-is for BUY_NO, never complemented] [--lam must be finite and in [0.25, 0.50] (SF-BL-10659): a non-finite or out-of-band --lam is rejected exit 2 with one JSON error object and no decision; boundaries inclusive, default 0.35] [--domain crypto applies 0.5x dampener + Gate A; --domain is canonicalised as in forecast (SF-BL-9580): every crypto label variant triggers Gate A and the dampener, unknown labels are rejected exit 2]",
                       "returns": "decision TRADE|SKIP (hurdle: side edge p_side-q_side>=0.05, = |p-q_exec| on the yes path; Gate D conviction p_side>=0.60 or p_side/q_side>=1.35; Gate A crypto on-chain), side, q_exec_side, q_side, KL edge, capped Kelly fraction"},
              "pool": {"args": "--probs a,b,c [--weights w1,w2,w3] — validated at the CLI boundary (SF-BL-9544): each prob finite and in [0,1]; weights finite, >= 0, length-matched to probs, positive sum; any violation exits 2 with one JSON error object and no pooled figure",
                       "returns": "log-odds pool, trimmed mean, mean, median"},
              "score": {"args": "--file CSV with columns y,p[,crowd] — must contain at least one data row (SF-BL-9747): an empty or header-only file is rejected exit 2 with one JSON error object and no score (n=0 has no defined brier/log/ece)",
                       "returns": "brier, log_score, ece, murphy, peer_score"},
              "harness-demo": "runs the alpha_kelly fixture benchmark (synthetic)",
              "schema": "this document"}})


def main():
    ap = argparse.ArgumentParser(prog="sf")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("headline")
    f = sub.add_parser("forecast"); f.add_argument("--domain", required=True,
                   help="markets|crypto|predmarkets — canonicalised case/whitespace-insensitively "
                        "(SF-BL-9580); unknown labels are rejected with a JSON error, exit 2")
    f.add_argument("--symbol"); f.add_argument("--asset"); f.add_argument("--slug")
    f.add_argument("--horizon", type=int, default=21)
    e = sub.add_parser("edge"); e.add_argument("--p", type=float, required=True)
    e.add_argument("--q-exec", type=float, required=True)
    e.add_argument("--q-exec-side", choices=["yes", "no"], default="yes",
                   help="SF-BL-9323: which side --q-exec prices (default yes: YES-side execution price; "
                        "no: NO-side execution price, e.g. a NO-book VWAP, used as-is and never complemented)")
    e.add_argument("--lam", type=float, default=0.35,
                   help="fractional-Kelly lambda — must be finite and in [0.25, 0.50] "
                        "(SF-BL-10659); a non-finite or out-of-band value is rejected "
                        "with a JSON error, exit 2")
    e.add_argument("--domain", default="predmarkets",
                   help="markets|crypto|predmarkets (default predmarkets) — canonicalised "
                        "case/whitespace-insensitively (SF-BL-9580): every crypto label variant "
                        "triggers Gate A + the 0.5x dampener; unknown labels are rejected, exit 2")
    e.add_argument("--onchain-verified", action="store_true",
                   help="Gate A: required for crypto trades (CryptoBench on-chain checks passed)")
    pl = sub.add_parser("pool"); pl.add_argument("--probs", required=True); pl.add_argument("--weights")
    sc = sub.add_parser("score"); sc.add_argument("--file", required=True)
    sub.add_parser("harness-demo"); sub.add_parser("schema")
    # SF-BL-9544: argparse treats a space-separated pool value starting
    # with '-' (e.g. `--probs -0.1,0.6`) as a new option and dies with a
    # usage error on stderr and NO JSON object — the contract's exit-2
    # JSON error would never be emitted for exactly the negative values
    # cmd_pool must reject. Join such values into --flag=value form so
    # they reach cmd_pool's validation. Pool flags only; a following
    # '--...' token is a real flag and is left alone.
    # SF-BL-9717: same join for the edge flags --p / --q-exec — a
    # space-form `-inf` is not matched as a negative number by argparse
    # and died the same way (usage on stderr, no JSON), so the finite
    # validation in cmd_edge could never emit its JSON error for it.
    # SF-BL-10659: same join for --lam — a space-form `-inf` after
    # --lam died the same way, so the band validation in cmd_edge
    # could never emit its JSON error for it.
    argv = sys.argv[1:]
    joined, i = [], 0
    while i < len(argv):
        if (argv[i] in ("--probs", "--weights", "--p", "--q-exec", "--lam") and i + 1 < len(argv)
                and argv[i + 1].startswith("-") and not argv[i + 1].startswith("--")):
            joined.append(f"{argv[i]}={argv[i + 1]}")
            i += 2
        else:
            joined.append(argv[i])
            i += 1
    a = ap.parse_args(joined)
    {"headline": cmd_headline, "forecast": cmd_forecast, "edge": cmd_edge, "pool": cmd_pool,
     "score": cmd_score, "harness-demo": cmd_harness_demo, "schema": cmd_schema}[a.cmd](a)


if __name__ == "__main__":
    main()
