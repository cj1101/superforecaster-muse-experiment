"""SF-BL-11047 — size_walked_exit_bid exit-ladder validation tests.

Prereg: size_walked_exit_bid treats a None/non-iterable ladder or
non-OrderBookLevel entry as no exit depth (0.0, 0.0) with a logged
count, OR raises a named ValueError, fixtured at yes_bids=None and a
dict entry, with well-formed ladders bit-identical; kill if a caller
census proves direct callers only ever pass constructor-validated
books (census NOT met — see CALLER_CENSUS.md). The named-ValueError
branch is the one implemented, matching SF-BL-11003/11005.

Run: python3 test_exit_ladder_validation.py
"""
import importlib.util
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster"))
from superforecaster import alpha_kelly as ak

WORK = os.path.dirname(os.path.abspath(__file__))
_base_path = os.path.join(WORK, "alpha_kelly_prepatch.py")
if not os.path.exists(_base_path):
    _base_path = os.path.expanduser(
        "~/workspace/superforecaster_out/backlog/work/SF-BL-11047/"
        "alpha_kelly_prepatch.py")
_spec = importlib.util.spec_from_file_location(
    "alpha_kelly_prepatch_11047", _base_path)
pre = importlib.util.module_from_spec(_spec)
sys.modules["alpha_kelly_prepatch_11047"] = pre
_spec.loader.exec_module(pre)

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def clean_book():
    # TASK.md frozen fixture: clean book returns (0.48, 1.0) for
    # 100 shares BUY_YES, (0.50, 1.0) for BUY_NO.
    return ak.OrderBookSnapshot(
        market_id="T-11047", title="fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C11047", est_days_to_resolution=7.0,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=1000.0)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=1000.0)],
        fee_rate=0.01)


def mutated(field, value):
    book = clean_book()
    setattr(book, field, value)
    return book


def expect_invalid_ladder(name, book, side=ak.TradeAction.BUY_YES,
                          shares=100.0):
    try:
        out = ak.size_walked_exit_bid(book, side, shares)
    except ValueError as exc:
        msg = str(exc)
        check(name, "INVALID_EXIT_LADDER" in msg and "SF-BL-11047" in msg
              and "OrderBookLevel" in msg, f"msg={msg!r}")
    except Exception as exc:  # noqa: BLE001
        check(name, False, f"wrong exception {type(exc).__name__}: {exc}")
    else:
        check(name, False, f"returned {out!r} instead of raising")


# --- 1. Well-formed ladders bit-identical -------------------------------
v, f = ak.size_walked_exit_bid(clean_book(), ak.TradeAction.BUY_YES, 100.0)
check("clean BUY_YES 100sh == (0.48, 1.0)", v == 0.48 and f == 1.0, f"{v} {f}")
v, f = ak.size_walked_exit_bid(clean_book(), ak.TradeAction.BUY_NO, 100.0)
check("clean BUY_NO 100sh == (0.50, 1.0)", v == 0.50 and f == 1.0, f"{v} {f}")

multi = ak.OrderBookSnapshot(
    market_id="T-11047-M", title="multi",
    domain=ak.MarketDomain.PREDICTION_MARKET,
    factor_cluster="C11047", est_days_to_resolution=7.0,
    yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000.0)],
    yes_bids=[ak.OrderBookLevel(price=0.44, size_shares=100.0),
              ak.OrderBookLevel(price=0.30, size_shares=100000.0)],
    fee_rate=0.01)
v, f = ak.size_walked_exit_bid(multi, ak.TradeAction.BUY_YES, 1100.0)
exp = (100 * 0.44 + 1000 * 0.30) / 1100
check("multi-rung VWAP exact", v == exp and f == 1.0, f"{v} vs {exp}")
v, f = ak.size_walked_exit_bid(multi, ak.TradeAction.BUY_YES, 200000.0)
check("partial fill exact", abs(f - 100100 / 200000) < 1e-12 and v > 0,
      f"{v} {f}")
v, f = ak.size_walked_exit_bid(mutated("yes_bids", []),
                               ak.TradeAction.BUY_YES, 10.0)
check("valid empty ladder -> (0.0, 0.0)", v == 0.0 and f == 0.0, f"{v} {f}")
v, f = ak.size_walked_exit_bid(
    mutated("yes_bids", (ak.OrderBookLevel(price=0.48, size_shares=1000.0),)),
    ak.TradeAction.BUY_YES, 100.0)
check("tuple ladder bit-identical", v == 0.48 and f == 1.0, f"{v} {f}")


class _SubLevel(ak.OrderBookLevel):
    pass


v, f = ak.size_walked_exit_bid(
    mutated("yes_bids", [_SubLevel(price=0.48, size_shares=1000.0)]),
    ak.TradeAction.BUY_YES, 100.0)
check("OrderBookLevel subclass entry passes", v == 0.48 and f == 1.0,
      f"{v} {f}")
# Non-live / non-tradeable genuine levels keep identical-to-clean
# filtering (SF-BL-9398/9542), untouched by the shape gate.
dirty = mutated("yes_bids", [ak.OrderBookLevel(price=1.50, size_shares=1000.0),
                             ak.OrderBookLevel(price=0.48, size_shares=0.0),
                             ak.OrderBookLevel(price=0.48, size_shares=1000.0)])
v, f = ak.size_walked_exit_bid(dirty, ak.TradeAction.BUY_YES, 100.0)
check("invalid-price/zero-size rungs still filtered", v == 0.48 and f == 1.0,
      f"{v} {f}")

# --- 2. TASK-named fixtures raise the named ValueError ------------------
expect_invalid_ladder("yes_bids=None BUY_YES raises named ValueError",
                      mutated("yes_bids", None))
expect_invalid_ladder("yes_bids=[dict] BUY_YES raises named ValueError",
                      mutated("yes_bids", [{"price": 0.48,
                                            "size_shares": 1000.0}]))
expect_invalid_ladder("yes_asks=None BUY_NO raises named ValueError",
                      mutated("yes_asks", None), side=ak.TradeAction.BUY_NO)
expect_invalid_ladder("yes_asks=[dict] BUY_NO raises named ValueError",
                      mutated("yes_asks", [{"price": 0.50,
                                            "size_shares": 1000.0}]),
                      side=ak.TradeAction.BUY_NO)

# --- 3. Extended malformed sweep ----------------------------------------
for label, bad in [("int 42", 42), ("float 0.48", 0.48), ("str 'ladder'",
                    "ladder"), ("dict as ladder", {"price": 0.48}),
                   ("set container", {"0.48x1000"}),
                   ("single OrderBookLevel as ladder",
                    ak.OrderBookLevel(price=0.48, size_shares=1000.0)),
                   ("bool True", True)]:
    expect_invalid_ladder(f"yes_bids={label} raises named ValueError",
                          mutated("yes_bids", bad))
for label, bad in [("None entry", None), ("str entry", "0.48x1000"),
                   ("int entry", 42), ("tuple entry", (0.48, 1000.0)),
                   ("list entry", [0.48, 1000.0]),
                   ("duck-typed SimpleNamespace entry",
                    SimpleNamespace(price=0.48, size_shares=1000.0))]:
    expect_invalid_ladder(f"yes_bids=[{label}] raises named ValueError",
                          mutated("yes_bids", [bad]))

# --- 4. Whole ladder scanned: bad entry beyond fill depth raises ------
expect_invalid_ladder(
    "mixed [valid, dict] with tiny shares still raises",
    mutated("yes_bids", [ak.OrderBookLevel(price=0.48, size_shares=1000.0),
                         {"price": 0.44, "size_shares": 1000.0}]),
    shares=10.0)
expect_invalid_ladder(
    "mixed [dict, valid] BUY_NO still raises",
    mutated("yes_asks", [{"price": 0.50, "size_shares": 1000.0},
                         ak.OrderBookLevel(price=0.52, size_shares=1000.0)]),
    side=ak.TradeAction.BUY_NO, shares=10.0)

# --- 5. Ordering / scope pins -------------------------------------------
v, f = ak.size_walked_exit_bid(mutated("yes_bids", None),
                               ak.TradeAction.BUY_YES, 0.0)
check("malformed ladder + shares=0 -> (0,0) (no walk, no masking of a "
      "walk)", v == 0.0 and f == 0.0, f"{v} {f}")
v, f = ak.size_walked_exit_bid(mutated("yes_bids", None),
                               ak.TradeAction.BUY_YES, -5.0)
check("malformed ladder + shares<0 -> (0,0)", v == 0.0 and f == 0.0,
      f"{v} {f}")
v, f = ak.size_walked_exit_bid(mutated("yes_bids", None),
                               ak.TradeAction.SKIP, 100.0)
check("malformed ladder + SKIP side -> (0,0)", v == 0.0 and f == 0.0,
      f"{v} {f}")
v, f = ak.size_walked_exit_bid(mutated("yes_asks", None),
                               ak.TradeAction.BUY_YES, 100.0)
check("malformed NON-selected ladder (yes_asks) + BUY_YES bit-identical",
      v == 0.48 and f == 1.0, f"{v} {f}")
v, f = ak.size_walked_exit_bid(mutated("yes_bids", None),
                               ak.TradeAction.BUY_NO, 100.0)
check("malformed NON-selected ladder (yes_bids) + BUY_NO bit-identical",
      v == 0.50 and f == 1.0, f"{v} {f}")
# Precedence: SF-BL-11003 book gate and SF-BL-11005 shares gate first.
try:
    ak.size_walked_exit_bid(None, ak.TradeAction.BUY_YES, 100.0)
    check("book=None precedence (11003 first)", False, "no raise")
except ValueError as exc:
    check("book=None precedence (11003 first)", "SF-BL-11003" in str(exc),
          str(exc))
try:
    ak.size_walked_exit_bid(mutated("yes_bids", None),
                            ak.TradeAction.BUY_YES, "100")
    check("shares='100' precedence (11005 before ladder)", False, "no raise")
except ValueError as exc:
    check("shares='100' precedence (11005 before ladder)",
          "SF-BL-11005" in str(exc), str(exc))

# --- 6. No bare TypeError / AttributeError in the reject set ------------
rejects = [mutated("yes_bids", None), mutated("yes_bids", 42),
           mutated("yes_bids", [{"price": 0.48, "size_shares": 1.0}]),
           mutated("yes_bids", [None])]
for i, book in enumerate(rejects):
    try:
        ak.size_walked_exit_bid(book, ak.TradeAction.BUY_YES, 100.0)
        check(f"reject[{i}] raises (no bare crash)", False, "returned")
    except (TypeError, AttributeError) as exc:
        check(f"reject[{i}] raises (no bare crash)", False,
              f"bare {type(exc).__name__}: {exc}")
    except ValueError:
        check(f"reject[{i}] raises (no bare crash)", True)

# --- 7. Downstream callers: clean unchanged, malformed propagates named --
rt, qx, ff = ak.round_trip_cost(clean_book(), ak.TradeAction.BUY_YES,
                                100.0, 0.505)
check("round_trip_cost on clean book exact",
      abs(rt - (0.505 - 0.48 * 1.0 * 0.99)) < 1e-12 and ff == 1.0,
      f"{rt} {qx} {ff}")
try:
    ak.round_trip_cost(mutated("yes_bids", None), ak.TradeAction.BUY_YES,
                       100.0, 0.505)
    check("round_trip_cost malformed ladder propagates named ValueError",
          False, "no raise")
except ValueError as exc:
    check("round_trip_cost malformed ladder propagates named ValueError",
          "INVALID_EXIT_LADDER" in str(exc) and "SF-BL-11047" in str(exc),
          str(exc))
except Exception as exc:  # noqa: BLE001
    check("round_trip_cost malformed ladder propagates named ValueError",
          False, f"wrong {type(exc).__name__}: {exc}")
mark, vwap, fill, proceeds = ak.bid_ladder_liquidation_mark(
    clean_book(), ak.TradeAction.BUY_YES, 100.0)
check("bid_ladder mark on clean book exact",
      vwap == 0.48 and fill == 1.0 and abs(mark - 0.48) < 1e-12
      and abs(proceeds - 48.0) < 1e-9, f"{mark} {vwap} {fill} {proceeds}")
try:
    ak.bid_ladder_liquidation_mark(mutated("yes_bids", [{"price": 0.48}]),
                                   ak.TradeAction.BUY_YES, 100.0)
    check("bid_ladder malformed ladder propagates named ValueError",
          False, "no raise")
except ValueError as exc:
    check("bid_ladder malformed ladder propagates named ValueError",
          "INVALID_EXIT_LADDER" in str(exc), str(exc))
except Exception as exc:  # noqa: BLE001
    check("bid_ladder malformed ladder propagates named ValueError",
          False, f"wrong {type(exc).__name__}: {exc}")

# --- 8. Bit-identical vs pre-patch module on a well-formed grid ---------
def _pre_book(bids, asks):
    return pre.OrderBookSnapshot(
        market_id="G", title="grid", domain=pre.MarketDomain.PREDICTION_MARKET,
        factor_cluster="CG", est_days_to_resolution=7.0,
        yes_asks=[pre.OrderBookLevel(price=p, size_shares=s) for p, s in asks],
        yes_bids=[pre.OrderBookLevel(price=p, size_shares=s) for p, s in bids],
        fee_rate=0.01)


def _live_book(bids, asks):
    return ak.OrderBookSnapshot(
        market_id="G", title="grid", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="CG", est_days_to_resolution=7.0,
        yes_asks=[ak.OrderBookLevel(price=p, size_shares=s) for p, s in asks],
        yes_bids=[ak.OrderBookLevel(price=p, size_shares=s) for p, s in bids],
        fee_rate=0.01)


grid_books = [
    ([(0.48, 1000.0)], [(0.50, 1000.0)]),
    ([(0.44, 100.0), (0.30, 100000.0)], [(0.50, 100000.0)]),
    ([], [(0.50, 100.0)]),
    ([(0.48, 1000.0)], []),
    ([(1.50, 1000.0), (0.48, 0.0), (0.40, 500.0)],
     [(0.50, 1000.0), (0.60, 250.0)]),
]
mismatch = 0
for bids, asks in grid_books:
    for side_live, side_pre in ((ak.TradeAction.BUY_YES, pre.TradeAction.BUY_YES),
                                (ak.TradeAction.BUY_NO, pre.TradeAction.BUY_NO),
                                (ak.TradeAction.SKIP, pre.TradeAction.SKIP)):
        for shares in (0.0, -5.0, 10.0, 100.0, 1100.0, 200000.0):
            a = ak.size_walked_exit_bid(_live_book(bids, asks), side_live, shares)
            b = pre.size_walked_exit_bid(_pre_book(bids, asks), side_pre, shares)
            if a != b:
                mismatch += 1
                print(f"    mismatch bids={bids} side={side_live} "
                      f"shares={shares}: live={a} pre={b}")
check("well-formed grid bit-identical to pre-patch (90 calls)",
      mismatch == 0, f"{mismatch} mismatches")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
