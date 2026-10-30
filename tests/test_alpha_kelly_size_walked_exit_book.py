"""SF-BL-11003 — size_walked_exit_bid book-type validation tests.

Prereg: size_walked_exit_bid must raise a named ValueError for a
non-OrderBookSnapshot book, fixtured at None, dict and string, with
clean-book exit VWAPs bit-identical; kill if a caller census shows
every caller passes a construction-validated OrderBookSnapshot under
a tested invariant (census NOT met — see CALLER_CENSUS.md).

Run: python3 test_size_walked_exit_book_validation.py
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster"))
from superforecaster import alpha_kelly as ak

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
    # TASK.md frozen fixture shape: clean book returns (0.48, 1.0)
    # for 100 shares BUY_YES.
    return ak.OrderBookSnapshot(
        market_id="T-11003", title="fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C11003", est_days_to_resolution=7.0,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=1000.0)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=1000.0)],
        fee_rate=0.01)


def expect_invalid_book(name, book, side=ak.TradeAction.BUY_YES,
                        shares=100.0):
    try:
        out = ak.size_walked_exit_bid(book, side, shares)
    except ValueError as exc:
        msg = str(exc)
        check(name, "INVALID_BOOK" in msg and "SF-BL-11003" in msg
              and "OrderBookSnapshot" in msg, f"msg={msg!r}")
    except Exception as exc:  # noqa: BLE001
        check(name, False, f"wrong exception {type(exc).__name__}: {exc}")
    else:
        check(name, False, f"returned {out!r} instead of raising")


# --- 1. Clean-book results bit-identical --------------------------------
v, f = ak.size_walked_exit_bid(clean_book(), ak.TradeAction.BUY_YES, 100.0)
check("clean BUY_YES 100sh == (0.48, 1.0)", v == 0.48 and f == 1.0, f"{v} {f}")
v, f = ak.size_walked_exit_bid(clean_book(), ak.TradeAction.BUY_NO, 100.0)
check("clean BUY_NO 100sh == (0.50, 1.0)", v == 0.50 and f == 1.0, f"{v} {f}")

multi = ak.OrderBookSnapshot(
    market_id="T-11003-M", title="multi",
    domain=ak.MarketDomain.PREDICTION_MARKET,
    factor_cluster="C11003", est_days_to_resolution=7.0,
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
empty = ak.OrderBookSnapshot(
    market_id="T-11003-E", title="empty",
    domain=ak.MarketDomain.PREDICTION_MARKET,
    factor_cluster="C11003", est_days_to_resolution=7.0,
    yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100.0)],
    yes_bids=[], fee_rate=0.01)
v, f = ak.size_walked_exit_bid(empty, ak.TradeAction.BUY_YES, 10.0)
check("valid book, no bids -> (0.0, 0.0)", v == 0.0 and f == 0.0, f"{v} {f}")

# Valid-book early-return contracts unchanged (finite operands only;
# non-finite shares are SF-BL-11005's separate scope).
v, f = ak.size_walked_exit_bid(clean_book(), ak.TradeAction.BUY_YES, 0.0)
check("valid book shares=0 -> (0,0)", v == 0.0 and f == 0.0, f"{v} {f}")
v, f = ak.size_walked_exit_bid(clean_book(), ak.TradeAction.BUY_YES, -5.0)
check("valid book shares<0 -> (0,0)", v == 0.0 and f == 0.0, f"{v} {f}")
v, f = ak.size_walked_exit_bid(clean_book(), ak.TradeAction.SKIP, 100.0)
check("valid book SKIP side -> (0,0)", v == 0.0 and f == 0.0, f"{v} {f}")

# --- 2. TASK-named fixtures raise the named ValueError ------------------
expect_invalid_book("book=None BUY_YES raises named ValueError", None)
expect_invalid_book("book=None BUY_NO raises named ValueError", None,
                    side=ak.TradeAction.BUY_NO)
expect_invalid_book("book=dict BUY_YES raises named ValueError",
                    {"yes_bids": [], "yes_asks": [], "fee_rate": 0.01})
expect_invalid_book("book=dict BUY_NO raises named ValueError",
                    {"yes_bids": [], "yes_asks": [], "fee_rate": 0.01},
                    side=ak.TradeAction.BUY_NO)
expect_invalid_book("book='book' BUY_YES raises named ValueError", "book")
expect_invalid_book("book='book' BUY_NO raises named ValueError", "book",
                    side=ak.TradeAction.BUY_NO)

# --- 3. Extended non-snapshot sweep (kill sweep) -------------------------
for label, bad in [("int 42", 42), ("float 0.48", 0.48), ("list", []),
                   ("tuple", ()), ("bool True", True), ("object", object()),
                   ("OrderBookLevel as book",
                    ak.OrderBookLevel(price=0.48, size_shares=100.0)),
                   ("duck-typed SimpleNamespace",
                    SimpleNamespace(yes_bids=[], yes_asks=[],
                                    fee_rate=0.01))]:
    expect_invalid_book(f"book={label} raises named ValueError", bad)

# --- 4. Validation precedes the shares / side early returns -------------
expect_invalid_book("None book + shares=0 still raises (no masking)",
                    None, shares=0.0)
expect_invalid_book("None book + SKIP side still raises (no masking)",
                    None, side=ak.TradeAction.SKIP, shares=100.0)
expect_invalid_book("dict book + shares<0 still raises (no masking)",
                    {"yes_bids": []}, shares=-5.0)

# --- 5. No bare AttributeError anywhere in the reject set ---------------
for bad in (None, {}, "book", 42, []):
    try:
        ak.size_walked_exit_bid(bad, ak.TradeAction.BUY_YES, 100.0)
        check(f"no AttributeError for {type(bad).__name__}", False,
              "returned instead of raising")
    except AttributeError as exc:
        check(f"no AttributeError for {type(bad).__name__}", False,
              f"AttributeError: {exc}")
    except ValueError:
        check(f"no AttributeError for {type(bad).__name__}", True)

# --- 6. Downstream valid-book behaviour unchanged ------------------------
rt, qx, ff = ak.round_trip_cost(clean_book(), ak.TradeAction.BUY_YES,
                                100.0, 0.505)
check("round_trip_cost on clean book exact",
      abs(rt - (0.505 - 0.48 * 1.0 * 0.99)) < 1e-12 and ff == 1.0,
      f"{rt} {qx} {ff}")
mark, vwap, fill, proceeds = ak.bid_ladder_liquidation_mark(
    clean_book(), ak.TradeAction.BUY_YES, 100.0)
check("bid_ladder mark on clean book exact",
      vwap == 0.48 and fill == 1.0 and abs(mark - 0.48) < 1e-12
      and abs(proceeds - 48.0) < 1e-9, f"{mark} {vwap} {fill} {proceeds}")
mark = ak.bid_ladder_liquidation_mark(clean_book(), ak.TradeAction.BUY_YES,
                                      0.0)
check("bid_ladder shares=0 contract unchanged (SF-BL-3660)",
      mark == (0.0, 0.0, 0.0, 0.0), f"{mark}")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
