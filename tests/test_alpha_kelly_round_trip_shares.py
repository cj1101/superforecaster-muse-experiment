"""
test_round_trip_shares_validation.py — SF-BL-10466 verification tests.

Pre-registered done criterion: "Fix must reject non-finite or <=0
shares with a named error before the exit walk on a frozen book
fixture, with positive-share round-trip costs bit-identical; kill if
any invalid share count still returns an rt_cost tuple."

Fix under test: round_trip_cost validates shares via
_validate_round_trip_shares (domain (0, finite], finite real > 0,
bool / non-numeric rejected via _is_finite_number) BEFORE the exit
walk, after the SF-BL-9882 fee and SF-BL-9811 q_exec validations.
Kill NOT met (see CALLER_CENSUS.md): only the in-evaluate call was
incidentally shielded; direct/S2 callers pass unsanitised shares,
and the SF-BL-7432/7785 `shares <= 0` guards are False for NaN.

Pre-fix reproduction (verified live 2026-10-23, before the patch),
frozen fixture ask 0.60 / bid 0.58, fee 0.01, q_exec=0.65 (repro.py):
  shares= 1000 -> rt 0.0758, q_exit 0.5742, fill 1.0   (clean)
  shares= NaN  -> rt 0.65, q_exit 0.0, fill 0.0  (fabricated full price)
  shares= -100 -> rt 0.65, q_exit 0.0, fill 0.0  (fabricated full price)
  shares= 0    -> rt 0.65, q_exit 0.0, fill 0.0  (fabricated full price)

Run:  python3 test_round_trip_shares_validation.py  (exit 0 iff all pass)
No pytest dependency: plain asserts + __main__ runner, same convention
as tests/test_alpha_kelly_round_trip_q_exec.py.
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
        market_id="T-10466", title="fixture", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C10466", est_days_to_resolution=7.0,
        yes_asks=[ak.OrderBookLevel(price=0.60, size_shares=100000)],
        yes_bids=[ak.OrderBookLevel(price=0.58, size_shares=bid_size)],
        fee_rate=fee)


def _raises_named_value_error(shares, book=None, side=ak.TradeAction.BUY_YES,
                              q_exec=0.65):
    """True iff the call raises a ValueError naming shares / SF-BL-10466."""
    try:
        ak.round_trip_cost(book if book is not None else _book(), side, shares, q_exec)
    except ValueError as e:
        msg = str(e)
        return "shares" in msg and "SF-BL-10466" in msg and "(0, finite]" in msg
    except Exception:
        return False
    return False


# ---- 1. Clean fixture is bit-identical post-fix ----------------------------
rt, qx, ff = ak.round_trip_cost(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.65)
check("clean shares 1000 q_exec 0.65 rt_cost", abs(rt - 0.0758) < 1e-9, f"rt={rt}")
check("clean shares 1000 q_exit", abs(qx - 0.5742) < 1e-9, f"q_exit={qx}")
check("clean shares 1000 fill", ff == 1.0, f"fill={ff}")
# Sibling fixture points (SF-BL-9811/9882, q_exec 0.60 / 0.606) unchanged:
rt, qx, ff = ak.round_trip_cost(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.60)
check("clean q_exec 0.60 rt/q_exit", abs(rt - 0.0258) < 1e-9 and abs(qx - 0.5742) < 1e-9,
      f"rt={rt} q_exit={qx}")
rt, qx, ff = ak.round_trip_cost(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.606)
check("clean q_exec 0.606 rt/q_exit", abs(rt - 0.0318) < 1e-9 and abs(qx - 0.5742) < 1e-9,
      f"rt={rt} q_exit={qx}")
# BUY_NO side, clean: NO exit bids derive from asks 0.60 -> 0.40,
# q_exit = 0.40 * 1.0 * 0.99 = 0.396, rt = 0.606 - 0.396 = 0.210.
rt, qx, ff = ak.round_trip_cost(_book(), ak.TradeAction.BUY_NO, 1000.0, 0.606)
check("clean BUY_NO q_exit/rt", abs(qx - 0.396) < 1e-9 and abs(rt - 0.210) < 1e-9,
      f"q_exit={qx} rt={rt}")
# No exit depth with POSITIVE shares -> q_exit 0.0, rt == q_exec
# (unchanged contract; distinct from the invalid-shares fabrication —
# here the position exists, there is simply no bid to exit into).
empty_exit = ak.OrderBookSnapshot(
    market_id="T-10466-E", title="no bids", domain=ak.MarketDomain.PREDICTION_MARKET,
    factor_cluster="C10466", est_days_to_resolution=7.0,
    yes_asks=[ak.OrderBookLevel(price=0.60, size_shares=1000)],
    yes_bids=[], fee_rate=0.01)
rt, qx, ff = ak.round_trip_cost(empty_exit, ak.TradeAction.BUY_YES, 1000.0, 0.606)
check("no exit depth, positive shares -> (q_exec, 0, 0)",
      rt == 0.606 and qx == 0.0 and ff == 0.0, f"rt={rt} q_exit={qx} fill={ff}")
# Partial fill (SF-BL-9844 interplay): bids absorb 100 of 1000 shares,
# q_exit = 0.58 * 0.10 * 0.99 = 0.05742, rt = 0.60 - 0.05742 = 0.54258.
rt, qx, ff = ak.round_trip_cost(_book(bid_size=100), ak.TradeAction.BUY_YES, 1000.0, 0.60)
check("partial fill rt/q_exit exact",
      abs(qx - 0.05742) < 1e-9 and abs(rt - 0.54258) < 1e-9 and ff == 0.1,
      f"rt={rt} q_exit={qx} fill={ff}")

# ---- 2. Positive-share boundaries accepted, computed exactly ---------------
for s in [1, 1e-9, 0.5, 1.0, 1000.0, 100000.0]:
    rt, qx, ff = ak.round_trip_cost(_book(), ak.TradeAction.BUY_YES, s, 0.65)
    check(f"accept shares {s!r}", abs(rt - 0.0758) < 1e-9 and abs(qx - 0.5742) < 1e-9
          and ff == 1.0, f"rt={rt} fill={ff}")

# ---- 3. TASK.md named cases + every non-finite / <=0 shares raises ---------
for bad in [float("nan"), -100, 0]:
    check(f"reject shares {bad!r} (TASK named)", _raises_named_value_error(bad))
for bad in [float("inf"), float("-inf"), 0.0, -0.0, -1e-9, -0.20, -1.0, -100.0]:
    check(f"reject shares {bad!r}", _raises_named_value_error(bad))
# Bool / non-numeric / absurd-magnitude operands raise ValueError too
# (True would otherwise masquerade as a 1-share position; a huge int
# must not leak an OverflowError out of the finiteness check).
for bad in [True, False, "1000", None, 10 ** 400]:
    check(f"reject shares {bad!r}", _raises_named_value_error(bad))
# Both sides raise (validation is side-independent, before the walk):
check("reject BUY_NO shares NaN",
      _raises_named_value_error(float("nan"), side=ak.TradeAction.BUY_NO))

# ---- 4. Validation precedes the exit walk ----------------------------------
# Even a SKIP side / no-depth call raises with valid fee+q_exec: no
# caller can smuggle an invalid share count past on a no-op invocation.
check("reject precedes no-op walk (SKIP side)",
      _raises_named_value_error(0.0, side=ak.TradeAction.SKIP))
check("reject precedes walk on no-depth book",
      _raises_named_value_error(-100, book=empty_exit))
# size_walked_exit_bid's own contract is UNCHANGED (SF-BL-3660 relies
# on it): the primitive still returns (0.0, 0.0) for invalid shares —
# only round_trip_cost refuses to fabricate a cost from it.
v, f = ak.size_walked_exit_bid(_book(), ak.TradeAction.BUY_YES, 0.0)
check("size_walked_exit_bid contract unchanged (0,0)", v == 0.0 and f == 0.0,
      f"v={v} f={f}")
m = ak.bid_ladder_liquidation_mark(_book(), ak.TradeAction.BUY_YES, 0.0)
check("bid_ladder_liquidation_mark contract unchanged (0,0,0,0)",
      m == (0.0, 0.0, 0.0, 0.0), f"mark={m}")

# ---- 5. Kill test: NO invalid share count returns an rt_cost tuple ---------
ok = True
for bad in [float("nan"), float("inf"), float("-inf"), 0, 0.0, -0.0,
            -100, -1e-9, True, False, "1000", None, 10 ** 400]:
    try:
        ak.round_trip_cost(_book(), ak.TradeAction.BUY_YES, bad, 0.65)
        ok = False  # returned a tuple — kill criterion triggered
    except ValueError:
        pass
    except Exception:
        ok = False  # wrong error type — not the named rejection
check("kill: no invalid shares returns a tuple / wrong error", ok)
# And every accepted (positive) shares yields a finite, decidable cost:
ok = True
for s in [1e-9, 0.5, 1.0, 100.0, 1000.0, 50000.0, 100000.0]:
    rt, qx, ff = ak.round_trip_cost(_book(), ak.TradeAction.BUY_YES, s, 0.65)
    ok = ok and math.isfinite(rt) and math.isfinite(qx) and isinstance(rt > 0.05, bool)
check("invariant: rt finite & Gate D comparison decidable for all accepted shares", ok)

# ---- 6. Sibling validations still fire (no regression) ---------------------
mut = _book()
mut.fee_rate = -0.50  # post-construction mutation bypasses __post_init__
try:
    ak.round_trip_cost(mut, ak.TradeAction.BUY_YES, 1000.0, 0.606)
    fee_raises = False
except ValueError:
    fee_raises = True
check("invalid fee still raises with valid shares", fee_raises)
try:
    ak.round_trip_cost(_book(), ak.TradeAction.BUY_YES, 1000.0, float("nan"))
    qexec_raises = False
except ValueError:
    qexec_raises = True
check("invalid q_exec still raises with valid shares", qexec_raises)
# Pinned precedence preserved: fee / q_exec named errors still win on
# the no-op calls their agreed tests use (SKIP side, shares=0.0).
try:
    ak.round_trip_cost(mut, ak.TradeAction.SKIP, 0.0, 0.606)
    fee_prec = False
except ValueError as e:
    fee_prec = "SF-BL-9882" in str(e) or "fee_rate" in str(e)
check("fee precedence on SKIP/0-shares no-op preserved", fee_prec)
try:
    ak.round_trip_cost(_book(), ak.TradeAction.SKIP, 0.0, float("nan"))
    qexec_prec = False
except ValueError as e:
    qexec_prec = "SF-BL-9811" in str(e)
check("q_exec precedence on SKIP/0-shares no-op preserved", qexec_prec)

# ---- 7. Evaluate / Gate D path unchanged -----------------------------------
tight = ak.OrderBookSnapshot(
    market_id="T-10466-T", title="tight", domain=ak.MarketDomain.PREDICTION_MARKET,
    factor_cluster="C10466T", est_days_to_resolution=7.0,
    yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000)],
    yes_bids=[ak.OrderBookLevel(price=0.49, size_shares=100000)], fee_rate=0.01)
cl = {}
r = ak.evaluate_and_size_opportunity(tight, 0.80, 0.05, 10000.0, cl)
check("evaluate tight still BUY_YES", r.action == ak.TradeAction.BUY_YES,
      f"{r.action} {r.skip_reason}")
wide = ak.OrderBookSnapshot(
    market_id="T-10466-W", title="wide", domain=ak.MarketDomain.PREDICTION_MARKET,
    factor_cluster="C10466W", est_days_to_resolution=7.0,
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
