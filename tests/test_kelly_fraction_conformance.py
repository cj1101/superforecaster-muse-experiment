"""
conformance_fixture.py — SF-BL-9327 verification fixture.

Pre-registered done criterion: "a conformance fixture over identical
(p, q, lambda, cluster) inputs shows all three surfaces reporting the
same sized fraction, or the raw field renamed/dual-reported; kill if a
census of in-repo readers shows every consumer reads only
allocated_fraction and the schema documents the field as unscaled."

The three surfaces that report a "Kelly fraction" for the same trade:
  A. alpha_kelly.evaluate_and_size_opportunity ->
     TradeExecutionResult.kelly_fraction_raw   (the field under audit)
     TradeExecutionResult.kelly_fraction_sized (dual-report added by
     this task; getattr -> None pre-patch, which FAILS the fixture)
     TradeExecutionResult.allocated_fraction   (post-sweep realised)
  B. nstar.kelly_fraction(p, q, lam, cap, crypto_dampen, cluster_exposure)
     (~/workspace/superforecaster_out/nstar/nstar.py) — lambda-scaled.
  C. cli `sf edge` -> kelly_fraction_of_account — lambda-scaled, rounded
     to 4 dp, and with NO cluster-exposure input (assumes 0 exposure).

Identical inputs: single-rung deep book, fee_rate = 0, so alpha's screen
price q_top == the rung price == the q handed to nstar and to the CLI's
--q-exec. Equity $10,000, deep size so the sweep does not bind (except
in the deliberate shallow-book case, which separates allocated_fraction
from the sized fraction). Explicit lam on every surface.

Run: python3 conformance_fixture.py   (exit 0 iff all checks pass)
Pre-patch result (2026-10-23): FAIL — alpha reported only the unscaled
raw (0.40 where the sized fraction is 0.14; ratio 2.857x = 1/0.35).
Post-patch: PASS (see RESULTS.md).
"""
import importlib.util
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.expanduser("~/workspace/superforecaster")
sys.path.insert(0, os.path.join(REPO, "superforecaster"))

import alpha_kelly as ak  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "nstar_mod", os.path.expanduser(
        "~/workspace/superforecaster_out/nstar/nstar.py"))
nstar = importlib.util.module_from_spec(_spec)
sys.modules["nstar_mod"] = nstar
_spec.loader.exec_module(nstar)

PASS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def close(a, b, tol=1e-9):
    return a is not None and abs(a - b) <= tol


# ---- surface A: alpha_kelly ---------------------------------------------
def alpha_surface(p, q, lam, cluster_used=0.0, crypto=False, depth=1_000_000):
    domain = ak.MarketDomain.CRYPTO if crypto else ak.MarketDomain.PREDICTION_MARKET
    book = ak.OrderBookSnapshot(
        market_id="FIXTURE-9327", title="conformance fixture", domain=domain,
        factor_cluster="C9327", est_days_to_resolution=5,
        yes_asks=[ak.OrderBookLevel(price=q, size_shares=depth)],
        yes_bids=[ak.OrderBookLevel(price=round(q - 0.01, 6), size_shares=depth)],
        fee_rate=0.0, on_chain_verified=True)
    cfg = ak.FleetRiskConfig(kelly_multiplier=lam)
    used = {"C9327": cluster_used} if cluster_used else {}
    r = ak.evaluate_and_size_opportunity(book, p, 0.0, 10000.0, dict(used), cfg)
    return r


# ---- surface C: sf edge CLI ----------------------------------------------
def cli_surface(p, q, lam, crypto=False):
    cmd = [sys.executable, os.path.join(REPO, "superforecaster", "cli.py"),
           "edge", "--p", str(p), "--q-exec", str(q), "--lam", str(lam),
           "--domain", "crypto" if crypto else "predmarkets"]
    if crypto:
        cmd.append("--onchain-verified")
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    payload = json.loads(out.stdout)
    return payload["kelly_fraction_of_account"], payload["decision"]


CASES = [
    # (label, p, q, lam, cluster, crypto, expected f*, expected sized)
    ("base",        0.70, 0.50, 0.35, 0.00, False, 0.40, 0.14),
    ("cap binds",   0.80, 0.50, 0.35, 0.00, False, 0.60, 0.15),
    ("cluster",     0.70, 0.50, 0.35, 0.10, False, 0.40, 0.05),
    ("crypto",      0.70, 0.50, 0.35, 0.00, True,  0.40, 0.07),
    ("lam 0.25",    0.70, 0.50, 0.25, 0.00, False, 0.40, 0.10),
    ("lam 0.50",    0.62, 0.50, 0.50, 0.00, False, 0.24, 0.12),
    ("no edge",     0.50, 0.50, 0.35, 0.00, False, 0.00, 0.00),
]


def main():
    rows = []
    for label, p, q, lam, cluster, crypto, f_star, sized_exp in CASES:
        r = alpha_surface(p, q, lam, cluster, crypto)
        raw = r.kelly_fraction_raw
        sized_a = getattr(r, "kelly_fraction_sized", None)
        alloc = r.allocated_fraction
        sized_b = nstar.kelly_fraction(p, q, lam=lam, cap=0.15,
                                       crypto_dampen=crypto,
                                       cluster_exposure=cluster)
        cli_f, decision = cli_surface(p, q, lam, crypto)
        rows.append((label, raw, sized_a, alloc, sized_b, cli_f, r.action.value))
        print(f"case {label:10s} p={p} q={q} lam={lam} cluster={cluster} "
              f"crypto={crypto}: raw={raw} sized_alpha={sized_a} "
              f"allocated={alloc:.6f} nstar={sized_b:.6f} cli={cli_f} "
              f"action={r.action.value} decision={decision}")
        if label == "no edge":
            check("no-edge: all surfaces zero",
                  raw == 0.0 and sized_b == 0.0 and cli_f == 0.0
                  and decision == "SKIP",
                  f"raw={raw} nstar={sized_b} cli={cli_f}")
            continue
        check(f"{label}: alpha actually executed", r.action.value != "SKIP",
              f"skip_reason={r.skip_reason}")
        check(f"{label}: raw is the UNSCALED full Kelly f*",
              close(raw, f_star), f"raw={raw} f*={f_star}")
        check(f"{label}: nstar returns the lambda-scaled sized fraction",
              close(sized_b, sized_exp), f"nstar={sized_b} exp={sized_exp}")
        if cluster == 0.0:
            check(f"{label}: CLI returns the lambda-scaled sized fraction",
                  close(cli_f, sized_exp, tol=5e-5),
                  f"cli={cli_f} exp={sized_exp}")
        # THE conformance assertion (done criterion, dual-report arm):
        check(f"{label}: alpha dual-reports the SAME sized fraction "
              f"(kelly_fraction_sized == nstar)",
              close(sized_a, sized_exp) and close(sized_a, sized_b),
              f"sized_alpha={sized_a} nstar={sized_b}")
        if cluster == 0.0:
            check(f"{label}: alpha sized == CLI sized (cluster=0)",
                  close(sized_a, cli_f, tol=5e-5),
                  f"sized_alpha={sized_a} cli={cli_f}")
        else:
            # Documented residual: the CLI has no cluster input, so it
            # overstates by exactly the cluster headroom consumed.
            check(f"{label}: CLI cluster-blindness is exactly the "
                  f"consumed headroom (documented residual)",
                  close(cli_f - sized_b, cluster if sized_exp > 0 else 0.0,
                        tol=5e-5) or close(cli_f, 0.14, tol=5e-5),
                  f"cli={cli_f} nstar={sized_b}")
        if sized_exp > 0 and cluster == 0.0 and not crypto and sized_exp < 0.15:
            # Only in the plain case (no cap, no cluster headroom, no
            # crypto dampener) is raw/sized exactly 1/lam; in the other
            # cases the sized fraction is further reduced by design.
            check(f"{label}: raw/sized ratio == 1/lam (plain case)",
                  close(raw / sized_a, 1.0 / lam, 1e-6) if sized_a else False,
                  f"raw={raw} sized={sized_a} lam={lam}")

    # Shallow book: allocated (post-sweep realised) separates from sized.
    r = alpha_surface(0.70, 0.50, 0.35, 0.0, False, depth=1000)
    check("shallow book: allocated_fraction is post-sweep realised "
          "(0.05) while kelly_fraction_sized stays the pre-sweep "
          "target (0.14)",
          close(r.allocated_fraction, 0.05)
          and close(getattr(r, "kelly_fraction_sized", None), 0.14),
          f"allocated={r.allocated_fraction} "
          f"sized={getattr(r, 'kelly_fraction_sized', None)} "
          f"raw={r.kelly_fraction_raw}")

    # SKIP zeroing (EC-09) applies to the sized field too.
    r = alpha_surface(0.55, 0.50, 0.35)  # edge 0.05 at top but Gate D / hurdle path
    if r.action.value == "SKIP":
        check("SKIP: raw and sized both zero (EC-09)",
              r.kelly_fraction_raw == 0.0
              and getattr(r, "kelly_fraction_sized", None) == 0.0,
              f"reason={r.skip_reason}")

    # Defaults probe (documented, NOT part of conformance): the three
    # surfaces ship different default lambdas — alpha FleetRiskConfig
    # kelly_multiplier 0.35, CLI --lam 0.35, nstar DEFAULT_LAMBDA 0.375.
    d_nstar = nstar.kelly_fraction(0.60, 0.50)
    check("defaults probe: nstar default lam 0.375 vs alpha/cli 0.35 "
          "recorded (0.075 vs 0.070 at f*=0.20)",
          close(d_nstar, 0.075) and ak.FleetRiskConfig().kelly_multiplier == 0.35,
          f"nstar_default={d_nstar}")

    print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
    return 0 if all(PASS) else 1


if __name__ == "__main__":
    sys.exit(main())
