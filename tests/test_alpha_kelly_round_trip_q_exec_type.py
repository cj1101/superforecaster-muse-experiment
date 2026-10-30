"""
test_q_exec_type_validation.py — SF-BL-11080 verification tests.

Pre-registered done criterion: "round_trip_cost raises a named
ValueError for a bool or non-numeric q_exec, fixtured at '0.606', None
and True, with finite-q_exec results bit-identical; kill if
SF-BL-9882's landed fee fix is shown to also validate q_exec's type
at this function."

Coverage note: the type rejection itself is SF-BL-9811's landed
_validate_round_trip_q_exec (via _is_finite_number: bool excluded,
int/float only, OverflowError-safe). SF-BL-11080, filed a day earlier
on the type dimension alone, is verified here at ITS OWN fixture
(ask 0.60 / bid 0.58, fee 0.01, shares 100 — SF-BL-9811's suite used
shares 1000), and the validator's message now also cites SF-BL-11080.
Kill audit: see KILL_AUDIT.md (the literal SF-BL-9882 kill is NOT met;
the coverage is SF-BL-9811's, not SF-BL-9882's).

Historical pre-fix behaviour (pre-SF-BL-9811 baseline, reproduced in
historical_repro.py / repro_prefix.txt): '0.606' / None -> bare
TypeError at the rt_cost subtraction; True -> rt_cost 0.4258 vs the
clean 0.606 fixture's 0.0318.

Run: python3 test_q_exec_type_validation.py (exit 0 iff all pass)
"""
import importlib.util
import math
import os
import sys
from decimal import Decimal

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster"))
from superforecaster import alpha_kelly as ak  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PASS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(bid_size=10000):
    return ak.OrderBookSnapshot(
        market_id="T-11080", title="fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C11080", est_days_to_resolution=7.0,
        yes_asks=[ak.OrderBookLevel(price=0.60, size_shares=10000)],
        yes_bids=[ak.OrderBookLevel(price=0.58, size_shares=bid_size)],
        fee_rate=0.01)


def _raises_named(q, book=None, side=ak.TradeAction.BUY_YES, shares=100):
    """True iff ValueError naming q_exec, SF-BL-9811, SF-BL-11080, (0, 1]."""
    try:
        ak.round_trip_cost(book if book is not None else _book(),
                           side, shares, q)
    except ValueError as e:
        msg = str(e)
        return ("q_exec" in msg and "SF-BL-9811" in msg
                and "SF-BL-11080" in msg and "(0, 1]" in msg)
    except Exception:
        return False
    return False


# ---- 1. TASK fixture, clean: exact values ---------------------------------
rt, qx, ff = ak.round_trip_cost(_book(), ak.TradeAction.BUY_YES, 100, 0.606)
check("clean 0.606 at 100 shares exact",
      rt == 0.03180000000000005 and qx == 0.5741999999999999 and ff == 1.0,
      f"rt={rt!r} q_exit={qx!r} fill={ff!r}")
# BUY_NO clean at the same fixture: exit derives from asks -> 0.40 bid,
# q_exit = 0.40 * 1.0 * 0.99 = 0.396, rt = 0.606 - 0.396 = 0.210.
rt, qx, ff = ak.round_trip_cost(_book(), ak.TradeAction.BUY_NO, 100, 0.606)
check("clean BUY_NO exact", abs(rt - 0.210) < 1e-12 and abs(qx - 0.396) < 1e-12,
      f"rt={rt} q_exit={qx}")

# ---- 2. TASK-named type fixtures raise the named ValueError ---------------
for bad in ["0.606", None, True]:
    check(f"TASK fixture q_exec={bad!r} raises named ValueError",
          _raises_named(bad))

# ---- 3. Extended type set: no TypeError / OverflowError leak --------------
for bad in [False, "0.60", "", "abc", [0.606], (0.606,), {"q": 0.606},
            Decimal("0.606"), 10 ** 400, float("nan"), float("inf"),
            float("-inf"), 0.0, -0.20, 1.5]:
    check(f"reject q_exec={bad!r}", _raises_named(bad))

# ---- 4. Validation precedes the exit walk ---------------------------------
check("reject precedes walk (SKIP side, 0 shares)",
      _raises_named(None, side=ak.TradeAction.SKIP, shares=0))
empty_exit = ak.OrderBookSnapshot(
    market_id="T-11080-E", title="no bids",
    domain=ak.MarketDomain.PREDICTION_MARKET,
    factor_cluster="C11080", est_days_to_resolution=7.0,
    yes_asks=[ak.OrderBookLevel(price=0.60, size_shares=1000)],
    yes_bids=[], fee_rate=0.01)
check("reject precedes walk on no-depth book",
      _raises_named("0.606", book=empty_exit))
# Sibling precedence preserved on a genuine book: fee (SF-BL-9882)
# fires before q_exec, q_exec before shares (SF-BL-10466).
mut = _book()
mut.fee_rate = -0.50
try:
    ak.round_trip_cost(mut, ak.TradeAction.BUY_YES, 100, "0.606")
    fee_first = False
except ValueError as e:
    fee_first = "fee_rate" in str(e) and "SF-BL-9363" in str(e)
check("fee validation precedes q_exec validation", fee_first)
try:
    ak.round_trip_cost(_book(), ak.TradeAction.BUY_YES, "100", "0.606")
    qexec_before_shares = False
except ValueError as e:
    qexec_before_shares = "q_exec" in str(e)
check("q_exec validation precedes shares validation", qexec_before_shares)

# ---- 5. Finite results bit-identical to the pre-11080 snapshot ------------
_base_path = os.path.join(HERE, "alpha_kelly_prepatch.py")
if not os.path.exists(_base_path):  # running from repo tests/
    _base_path = os.path.expanduser(
        "~/workspace/superforecaster_out/backlog/work/SF-BL-11080/"
        "alpha_kelly_prepatch.py")
spec = importlib.util.spec_from_file_location("ak_pre11080", _base_path)
pre = importlib.util.module_from_spec(spec)
sys.modules["ak_pre11080"] = pre
spec.loader.exec_module(pre)


def _pre_book(bid_size=10000):
    return pre.OrderBookSnapshot(
        market_id="T-11080", title="fixture",
        domain=pre.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C11080", est_days_to_resolution=7.0,
        yes_asks=[pre.OrderBookLevel(price=0.60, size_shares=10000)],
        yes_bids=[pre.OrderBookLevel(price=0.58, size_shares=bid_size)],
        fee_rate=0.01)


ident = True
for q in [1e-9, 0.01, 0.10, 0.50, 0.5742, 0.606, 0.90, 0.999999, 1.0, 1]:
    for bid_size in [10000, 100, 50]:  # full, partial, partial fills
        a = ak.round_trip_cost(_book(bid_size), ak.TradeAction.BUY_YES, 100, q)
        b = pre.round_trip_cost(_pre_book(bid_size),
                                pre.TradeAction.BUY_YES, 100, q)
        ident = ident and (a == b)
        a = ak.round_trip_cost(_book(bid_size), ak.TradeAction.BUY_NO, 100, q)
        b = pre.round_trip_cost(_pre_book(bid_size),
                                pre.TradeAction.BUY_NO, 100, q)
        ident = ident and (a == b)
check("finite-q_exec grid bit-identical vs pre-11080 snapshot", ident)

# ---- 6. Gate D invariant: accepted costs are finite & decidable -----------
ok = True
for q in [1e-9, 0.01, 0.50, 0.606, 1.0]:
    rt, qx, ff = ak.round_trip_cost(_book(), ak.TradeAction.BUY_YES, 100, q)
    ok = ok and math.isfinite(rt) and math.isfinite(qx) \
        and isinstance(rt > 0.05, bool)
check("accepted q_exec -> finite rt, Gate D comparison decidable", ok)

# ---- 7. Evaluate / Gate D path unchanged ----------------------------------
tight = ak.OrderBookSnapshot(
    market_id="T-11080-T", title="tight",
    domain=ak.MarketDomain.PREDICTION_MARKET,
    factor_cluster="C11080T", est_days_to_resolution=7.0,
    yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000)],
    yes_bids=[ak.OrderBookLevel(price=0.49, size_shares=100000)],
    fee_rate=0.01)
cl = {}
r = ak.evaluate_and_size_opportunity(tight, 0.80, 0.05, 10000.0, cl)
check("evaluate tight still BUY_YES", r.action == ak.TradeAction.BUY_YES,
      f"{r.action} {r.skip_reason}")
wide = ak.OrderBookSnapshot(
    market_id="T-11080-W", title="wide",
    domain=ak.MarketDomain.PREDICTION_MARKET,
    factor_cluster="C11080W", est_days_to_resolution=7.0,
    yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000)],
    yes_bids=[ak.OrderBookLevel(price=0.40, size_shares=100000)],
    fee_rate=0.01)
cl = {}
r = ak.evaluate_and_size_opportunity(wide, 0.80, 0.05, 10000.0, cl)
check("evaluate wide still vetoed by Gate D, zero size, cluster untouched",
      r.action == ak.TradeAction.SKIP
      and bool(r.skip_reason) and r.skip_reason.startswith("ROUND_TRIP_GATE_D")
      and r.invested_capital == 0.0 and r.shares_acquired == 0.0 and cl == {},
      f"{r.action} {r.skip_reason} cl={cl}")

print(f"\n{sum(PASS)}/{len(PASS)} passed")
sys.exit(0 if all(PASS) else 1)
