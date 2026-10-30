"""
test_round_trip_q_exec_validation.py — SF-BL-9811 verification tests.

Pre-registered done criterion: "round_trip_cost must raise a named
ValueError on any q_exec outside (0, 1] or non-finite before computing
the exit walk, and Gate D must never be passable by a NaN cost; kill
if an in-repo caller is shown to sanitise q_exec into (0, 1] before
every round_trip_cost call."

Fix under test: round_trip_cost validates q_exec via
_validate_round_trip_q_exec (domain (0, 1], finite real, bool /
non-numeric rejected via _is_finite_number) BEFORE the exit walk,
alongside the SF-BL-9882 fee re-validation. Kill NOT met (see
CALLER_CENSUS.md): only the in-evaluate Gate D call is incidentally
shielded; direct callers pass unsanitised floats / bare literals.

Pre-fix reproduction (verified live 2026-10-23, before the patch),
fixture ask 0.60 / bid 0.58, fee 0.01, shares 1000 (repro.py):
  q_exec= 0.60 -> rt 0.0258, q_exit 0.5742, fill 1.0   (clean)
  q_exec= NaN  -> rt NaN  (Gate D's NaN > 0.05 is False; veto fails open)
  q_exec=-0.20 -> rt -0.7742
  q_exec= 0.0  -> rt -0.5742 ; q_exec=-inf -> rt -inf
  q_exec= 1.5  -> rt 0.9258 ; q_exec=+inf -> rt +inf (above binary max)

Run:  python3 test_round_trip_q_exec_validation.py  (exit 0 iff all pass)
No pytest dependency: plain asserts + __main__ runner, same convention
as tests/test_alpha_kelly_round_trip_fee.py.
"""
import math
import os
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster"))
from superforecaster import alpha_kelly as ak  # noqa: E402

PASS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(fee=0.01, bid_size=100000):
    return ak.OrderBookSnapshot(
        market_id="T-9811", title="fixture", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C9811", est_days_to_resolution=7.0,
        yes_asks=[ak.OrderBookLevel(price=0.60, size_shares=100000)],
        yes_bids=[ak.OrderBookLevel(price=0.58, size_shares=bid_size)],
        fee_rate=fee)


def _raises_named_value_error(q_exec, book=None, side=ak.TradeAction.BUY_YES,
                              shares=1000.0):
    """True iff the call raises a ValueError naming q_exec / SF-BL-9811 / (0, 1]."""
    try:
        ak.round_trip_cost(book if book is not None else _book(), side, shares, q_exec)
    except ValueError as e:
        msg = str(e)
        return "q_exec" in msg and "SF-BL-9811" in msg and "(0, 1]" in msg
    except Exception:
        return False
    return False


# ---- 1. Clean fixture is byte-identical post-fix ---------------------------
rt, qx, ff = ak.round_trip_cost(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.60)
check("clean q_exec 0.60 rt_cost", abs(rt - 0.0258) < 1e-9, f"rt={rt}")
check("clean q_exec 0.60 q_exit", abs(qx - 0.5742) < 1e-9, f"q_exit={qx}")
check("clean q_exec 0.60 fill", ff == 1.0, f"fill={ff}")
# The SF-BL-9882 fixture point (same book, q_exec 0.606) is unchanged:
rt, qx, ff = ak.round_trip_cost(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.606)
check("clean q_exec 0.606 rt/q_exit", abs(rt - 0.0318) < 1e-9 and abs(qx - 0.5742) < 1e-9,
      f"rt={rt} q_exit={qx}")
# BUY_NO side, clean: NO exit bids derive from asks 0.60 -> 0.40,
# q_exit = 0.40 * 1.0 * 0.99 = 0.396, rt = 0.606 - 0.396 = 0.210.
rt, qx, ff = ak.round_trip_cost(_book(), ak.TradeAction.BUY_NO, 1000.0, 0.606)
check("clean BUY_NO q_exit/rt", abs(qx - 0.396) < 1e-9 and abs(rt - 0.210) < 1e-9,
      f"q_exit={qx} rt={rt}")
# No exit depth -> q_exit 0.0, rt == q_exec (unchanged contract).
empty_exit = ak.OrderBookSnapshot(
    market_id="T-9811-E", title="no bids", domain=ak.MarketDomain.PREDICTION_MARKET,
    factor_cluster="C9811", est_days_to_resolution=7.0,
    yes_asks=[ak.OrderBookLevel(price=0.60, size_shares=1000)],
    yes_bids=[], fee_rate=0.01)
rt, qx, ff = ak.round_trip_cost(empty_exit, ak.TradeAction.BUY_YES, 1000.0, 0.606)
check("no exit depth -> (q_exec, 0, 0)", rt == 0.606 and qx == 0.0 and ff == 0.0,
      f"rt={rt} q_exit={qx} fill={ff}")
# Partial fill (SF-BL-9844 interplay): bids absorb 100 of 1000 shares,
# q_exit = 0.58 * 0.10 * 0.99 = 0.05742, rt = 0.60 - 0.05742 = 0.54258.
rt, qx, ff = ak.round_trip_cost(_book(bid_size=100), ak.TradeAction.BUY_YES, 1000.0, 0.60)
check("partial fill rt/q_exit finite & exact",
      abs(qx - 0.05742) < 1e-9 and abs(rt - 0.54258) < 1e-9 and ff == 0.1,
      f"rt={rt} q_exit={qx} fill={ff}")

# ---- 2. Domain boundaries: (0, 1] accepted, computed exactly ---------------
for q, exp_rt in [(1.0, 1.0 - 0.5742), (1, 1.0 - 0.5742),
                  (0.999999, 0.999999 - 0.5742), (1e-9, 1e-9 - 0.5742),
                  (0.5742, 0.0)]:
    rt, qx, ff = ak.round_trip_cost(_book(), ak.TradeAction.BUY_YES, 1000.0, q)
    check(f"accept q_exec {q!r}", abs(rt - exp_rt) < 1e-9 and abs(qx - 0.5742) < 1e-9,
          f"rt={rt}")

# ---- 3. Every q_exec outside (0, 1] or non-finite raises (named) -----------
for bad in [float("nan"), float("inf"), float("-inf"),
            0.0, -0.0, -0.20, -1e-9, -1.0, 1.0000001, 1.5, 2.0]:
    check(f"reject q_exec {bad!r}", _raises_named_value_error(bad))
# Bool / non-numeric / absurd-magnitude operands raise ValueError too
# (True would otherwise masquerade as the in-domain price 1.0; a huge
# int must not leak an OverflowError out of the finiteness check).
for bad in [True, False, "0.60", None, 10 ** 400]:
    check(f"reject q_exec {bad!r}", _raises_named_value_error(bad))

# ---- 4. Validation precedes the exit walk ----------------------------------
# Even a SKIP side / zero shares / no-depth call raises: no caller can
# smuggle an invalid q_exec past on a no-op invocation.
check("reject precedes no-op walk (SKIP, 0 shares)",
      _raises_named_value_error(float("nan"), side=ak.TradeAction.SKIP, shares=0.0))
check("reject precedes walk on no-depth book",
      _raises_named_value_error(-0.20, book=empty_exit))

# ---- 5. Gate D can never be passed by a NaN cost from this function --------
# For every accepted q_exec the returned cost is finite, so the bare
# `rt_cost > max_round_trip_cost` comparison is a real verdict; and the
# two pre-fix fail-open inputs (NaN, -0.20) now raise instead of
# returning a passable cost (covered in §3, restated as the gate claim):
ok = True
for q in [1e-9, 0.01, 0.10, 0.50, 0.5742, 0.60, 0.90, 0.999999, 1.0]:
    rt, qx, ff = ak.round_trip_cost(_book(), ak.TradeAction.BUY_YES, 1000.0, q)
    ok = ok and math.isfinite(rt) and math.isfinite(qx) and (0.0 <= qx <= 0.58)
    ok = ok and isinstance(rt > 0.05, bool)
check("invariant: rt finite & Gate D comparison decidable for all accepted q_exec", ok)

# ---- 6. SF-BL-9882 fee validation still fires (no regression) --------------
mut = _book()
mut.fee_rate = -0.50  # post-construction mutation bypasses __post_init__
try:
    ak.round_trip_cost(mut, ak.TradeAction.BUY_YES, 1000.0, 0.606)
    fee_raises = False
except ValueError:
    fee_raises = True
check("invalid fee still raises with valid q_exec", fee_raises)

# ---- 7. Evaluate / Gate D path unchanged -----------------------------------
tight = ak.OrderBookSnapshot(
    market_id="T-9811-T", title="tight", domain=ak.MarketDomain.PREDICTION_MARKET,
    factor_cluster="C9811T", est_days_to_resolution=7.0,
    yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000)],
    yes_bids=[ak.OrderBookLevel(price=0.49, size_shares=100000)], fee_rate=0.01)
cl = {}
r = ak.evaluate_and_size_opportunity(tight, 0.80, 0.05, 10000.0, cl)
check("evaluate tight still BUY_YES", r.action == ak.TradeAction.BUY_YES,
      f"{r.action} {r.skip_reason}")
wide = ak.OrderBookSnapshot(
    market_id="T-9811-W", title="wide", domain=ak.MarketDomain.PREDICTION_MARKET,
    factor_cluster="C9811W", est_days_to_resolution=7.0,
    yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000)],
    yes_bids=[ak.OrderBookLevel(price=0.40, size_shares=100000)], fee_rate=0.01)
cl = {}
r = ak.evaluate_and_size_opportunity(wide, 0.80, 0.05, 10000.0, cl)
check("evaluate wide still vetoed by Gate D, zero size, cluster untouched",
      r.action == ak.TradeAction.SKIP
      and bool(r.skip_reason) and r.skip_reason.startswith("ROUND_TRIP_GATE_D")
      and r.invested_capital == 0.0 and r.shares_acquired == 0.0 and cl == {},
      f"{r.action} {r.skip_reason} cl={cl}")

print(f"\n{sum(PASS)}/{len(PASS)} passed")
sys.exit(0 if all(PASS) else 1)
