"""
test_shares_type_validation.py — SF-BL-11005 verification tests.

Pre-registered done criterion: "both functions must raise a named
ValueError for a shares argument that is not a finite real (bool
excluded), fixtured at '100', True and None, with positive numeric
share counts bit-identical; kill if SF-BL-10466's landed fix is
shown to type-check shares as well as range-check it."

Kill audit (CALLER_CENSUS.md): PARTIALLY met — SF-BL-10466 already
type-checks shares inside round_trip_cost (its fixtures raise the
named SF-BL-10466 error, now also citing SF-BL-11005), but it
explicitly left size_walked_exit_bid unchanged. This task's fix is
therefore in size_walked_exit_bid only: a finite-real gate
(_validate_exit_walk_shares, SF-BL-11005) after the SF-BL-11003 book
check. Finite <= 0 keeps the primitive's (0.0, 0.0) contract
(SF-BL-10466 agreed test §4 pin); positivity stays with
round_trip_cost / bid_ladder_liquidation_mark.

Frozen fixture (TASK.md): ask 0.50 / bid 0.48, fee 0.01, q_exec
0.505; clean shares=100 -> size_walked (0.48, 1.0),
round_trip (0.0298, 0.4752, 1.0).

Run:  python3 test_shares_type_validation.py  (exit 0 iff all pass)
"""
import importlib.util
import math
import os
import sys
from decimal import Decimal

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster"))
from superforecaster import alpha_kelly as ak  # noqa: E402

# Pre-patch baseline (live file with only this task's hunks reverted)
# for the bit-identical grid: sibling of this file in the task work
# dir, or the work-dir copy when this file runs from repo tests/.
_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "alpha_kelly_prepatch.py")
if not os.path.exists(_base_path):
    _base_path = os.path.expanduser(
        "~/workspace/superforecaster_out/backlog/work/SF-BL-11005/"
        "alpha_kelly_prepatch.py")
_spec = importlib.util.spec_from_file_location("ak_pre11005", _base_path)
pre = importlib.util.module_from_spec(_spec)
sys.modules["ak_pre11005"] = pre
_spec.loader.exec_module(pre)

PASS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(mod=ak, bid_levels=((0.48, 100000),), ask_levels=((0.50, 100000),),
          fee=0.01):
    return mod.OrderBookSnapshot(
        market_id="T-11005", title="fixture",
        domain=mod.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C11005", est_days_to_resolution=7.0,
        yes_asks=[mod.OrderBookLevel(price=p, size_shares=s)
                  for p, s in ask_levels],
        yes_bids=[mod.OrderBookLevel(price=p, size_shares=s)
                  for p, s in bid_levels],
        fee_rate=fee)


def _walk_raises(shares, side=ak.TradeAction.BUY_YES, book=None):
    """(raised_named_11005, other_exception) for size_walked_exit_bid."""
    try:
        ak.size_walked_exit_bid(book if book is not None else _book(),
                                side, shares)
    except ValueError as e:
        msg = str(e)
        return ("shares" in msg and "SF-BL-11005" in msg
                and "finite real" in msg), False
    except Exception:  # noqa: BLE001
        return False, True
    return False, False


def _rt_raises(shares, side=ak.TradeAction.BUY_YES, q_exec=0.505):
    """True iff round_trip_cost raises a ValueError naming shares and
    SF-BL-11005 (the type dimension) while retaining SF-BL-10466."""
    try:
        ak.round_trip_cost(_book(), side, shares, q_exec)
    except ValueError as e:
        msg = str(e)
        return ("shares" in msg and "SF-BL-11005" in msg
                and "SF-BL-10466" in msg and "(0, finite]" in msg)
    except Exception:  # noqa: BLE001
        return False
    return False


# ---- 1. Clean fixture (TASK.md values) --------------------------------------
v, f = ak.size_walked_exit_bid(_book(), ak.TradeAction.BUY_YES, 100)
check("clean size_walked shares=100 -> (0.48, 1.0)", (v, f) == (0.48, 1.0),
      f"({v}, {f})")
rt, qx, ff = ak.round_trip_cost(_book(), ak.TradeAction.BUY_YES, 100, 0.505)
check("clean round_trip shares=100 -> (0.0298, 0.4752, 1.0)",
      abs(rt - 0.0298) < 1e-9 and abs(qx - 0.4752) < 1e-9 and ff == 1.0,
      f"({rt}, {qx}, {ff})")

# ---- 2. TASK-named fixtures raise in BOTH functions --------------------------
for bad in ["100", True, None]:
    ok, other = _walk_raises(bad)
    check(f"size_walked raises named SF-BL-11005 for {bad!r}", ok and not other)
    check(f"round_trip raises named (10466+11005) for {bad!r}", _rt_raises(bad))
# Both sides raise (validation is side-independent, before the walk):
ok, _ = _walk_raises("100", side=ak.TradeAction.BUY_NO)
check("size_walked BUY_NO raises for '100'", ok)
check("round_trip BUY_NO raises for True",
      _rt_raises(True, side=ak.TradeAction.BUY_NO))

# ---- 3. Extended non-finite-real set raises in size_walked -------------------
for bad in [False, "1000", "", float("nan"), float("inf"), float("-inf"),
            [100], (100,), {"shares": 100}, Decimal("10"), 10 ** 400]:
    ok, other = _walk_raises(bad)
    check(f"size_walked raises ValueError (no leak) for {bad!r:.40}",
          ok and not other)
# Raise precedes even a SKIP-side / no-depth no-op call:
ok, _ = _walk_raises(None, side=ak.TradeAction.SKIP)
check("size_walked raise precedes SKIP-side no-op", ok)
empty = _book(bid_levels=())
ok, _ = _walk_raises("100", book=empty)
check("size_walked raise precedes no-depth return", ok)

# ---- 4. Preserved contracts (negative space) ---------------------------------
for s in [0, 0.0, -0.0, -100, -1e-9]:
    v, f = ak.size_walked_exit_bid(_book(), ak.TradeAction.BUY_YES, s)
    check(f"size_walked finite shares={s!r} keeps (0.0, 0.0) "
          "(SF-BL-10466 pin)", (v, f) == (0.0, 0.0), f"({v}, {f})")
v, f = ak.size_walked_exit_bid(empty, ak.TradeAction.BUY_YES, 100.0)
check("size_walked no-depth, positive shares -> (0.0, 0.0)",
      (v, f) == (0.0, 0.0), f"({v}, {f})")
v, f = ak.size_walked_exit_bid(_book(), ak.TradeAction.SKIP, 100.0)
check("size_walked invalid side, valid shares -> (0.0, 0.0) "
      "(SF-BL-10827 scope untouched)", (v, f) == (0.0, 0.0), f"({v}, {f})")
m = ak.bid_ladder_liquidation_mark(_book(), ak.TradeAction.BUY_YES, "100")
check("bid_ladder contract unchanged for '100' (0,0,0,0)",
      m == (0.0, 0.0, 0.0, 0.0), f"{m}")
m = ak.bid_ladder_liquidation_mark(_book(), ak.TradeAction.BUY_YES, 100)
check("bid_ladder valid shares unchanged (0.48, 0.48, 1.0, 48.0)",
      m == (0.48, 0.48, 1.0, 48.0), f"{m}")

# ---- 5. Kill sweep: no non-finite-real shares returns a tuple ----------------
ok = True
for bad in ["100", True, False, None, float("nan"), float("inf"),
            float("-inf"), "1000", [100], Decimal("10"), 10 ** 400]:
    for fn in (lambda v: ak.size_walked_exit_bid(
                   _book(), ak.TradeAction.BUY_YES, v),
               lambda v: ak.round_trip_cost(
                   _book(), ak.TradeAction.BUY_YES, v, 0.505)):
        try:
            fn(bad)
            ok = False  # returned a tuple
        except ValueError:
            pass
        except Exception:  # noqa: BLE001
            ok = False  # wrong error type / leaked exception
check("kill sweep: no non-finite-real shares returns a tuple "
      "from either function", ok)

# ---- 6. Bit-identical positive-share grid vs pre-patch baseline --------------
books = [
    dict(),  # TASK fixture
    dict(bid_levels=((0.48, 50), (0.45, 1000), (0.40, 100000)),
         ask_levels=((0.50, 100000), (0.55, 100000))),
    dict(bid_levels=((0.48, 30),), ask_levels=((0.50, 100000),)),
    dict(bid_levels=(), ask_levels=((0.50, 1000),)),
]
share_grid = [0.5, 1, 17, 100, 100.0, 100.5, 5000, 200000]
ident = True
for kw in books:
    for side_name in ["BUY_YES", "BUY_NO"]:
        for s in share_grid:
            live_b, pre_b = _book(ak, **kw), _book(pre, **kw)
            lv = ak.size_walked_exit_bid(
                live_b, getattr(ak.TradeAction, side_name), s)
            pv = pre.size_walked_exit_bid(
                pre_b, getattr(pre.TradeAction, side_name), s)
            ident = ident and (lv == pv)
            lr = ak.round_trip_cost(
                live_b, getattr(ak.TradeAction, side_name), s, 0.505)
            pr = pre.round_trip_cost(
                pre_b, getattr(pre.TradeAction, side_name), s, 0.505)
            ident = ident and (lr == pr)
            lm = ak.bid_ladder_liquidation_mark(
                live_b, getattr(ak.TradeAction, side_name), s)
            pm = pre.bid_ladder_liquidation_mark(
                pre_b, getattr(pre.TradeAction, side_name), s)
            ident = ident and (lm == pm)
check("positive-share grid bit-identical vs baseline "
      "(walk + round_trip + bid mark, both sides)", ident)

# ---- 7. Precedence / ordering -------------------------------------------------
# SF-BL-11003 book error still wins over a shares type error:
try:
    ak.size_walked_exit_bid(None, ak.TradeAction.BUY_YES, "100")  # type: ignore[arg-type]
    book_prec = False
except ValueError as e:
    book_prec = "SF-BL-11003" in str(e)
check("book (SF-BL-11003) precedence over shares in size_walked", book_prec)
# round_trip: its own shares validator fires before the walk (spy = 0 calls):
orig = ak.size_walked_exit_bid
calls = []
ak.size_walked_exit_bid = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]  # type: ignore[misc]
try:
    ak.round_trip_cost(_book(), ak.TradeAction.BUY_YES, "100", 0.505)
    spy_ok = False
except ValueError:
    spy_ok = (len(calls) == 0)
finally:
    ak.size_walked_exit_bid = orig  # type: ignore[misc]
check("round_trip shares validation precedes the exit walk (0 walk calls)",
      spy_ok)
# Sibling fee / q_exec raises still fire with valid shares:
mut = _book()
mut.fee_rate = -0.50
try:
    ak.round_trip_cost(mut, ak.TradeAction.BUY_YES, 100.0, 0.505)
    fee_ok = False
except ValueError:
    fee_ok = True
check("invalid fee still raises with valid shares", fee_ok)
try:
    ak.round_trip_cost(_book(), ak.TradeAction.BUY_YES, 100.0, float("nan"))
    qe_ok = False
except ValueError:
    qe_ok = True
check("invalid q_exec still raises with valid shares", qe_ok)

# ---- 8. Evaluate / Gate D path unchanged --------------------------------------
tight = ak.OrderBookSnapshot(
    market_id="T-11005-T", title="tight",
    domain=ak.MarketDomain.PREDICTION_MARKET,
    factor_cluster="C11005T", est_days_to_resolution=7.0,
    yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000)],
    yes_bids=[ak.OrderBookLevel(price=0.49, size_shares=100000)],
    fee_rate=0.01)
cl = {}
r = ak.evaluate_and_size_opportunity(tight, 0.80, 0.05, 10000.0, cl)
check("evaluate tight still BUY_YES", r.action == ak.TradeAction.BUY_YES,
      f"{r.action} {r.skip_reason}")
wide = ak.OrderBookSnapshot(
    market_id="T-11005-W", title="wide",
    domain=ak.MarketDomain.PREDICTION_MARKET,
    factor_cluster="C11005W", est_days_to_resolution=7.0,
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
check("all accepted size_walked results finite",
      all(math.isfinite(x) for x in ak.size_walked_exit_bid(
          _book(), ak.TradeAction.BUY_YES, 250.0)))

print(f"\n{sum(PASS)}/{len(PASS)} passed")
sys.exit(0 if all(PASS) else 1)
