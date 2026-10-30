"""
test_round_trip_book_validation.py — SF-BL-11004 verification tests.

Pre-registered done criterion: "round_trip_cost must raise a named
ValueError for a non-OrderBookSnapshot book, fixtured at None and
dict, before the fee check or exit walk, with clean-book round-trip
costs bit-identical; kill if a caller census shows round_trip_cost
is only ever reached through evaluate_and_size_opportunity's
already-validated book path."

Fix under test: round_trip_cost validates the book's type via
_validate_round_trip_book (strict isinstance(book,
OrderBookSnapshot)) as its FIRST statement — before the SF-BL-9882
fee validation (which dereferences book.fee_rate and was itself the
pre-fix crash site), the SF-BL-9811 q_exec validation, the
SF-BL-10466 shares validation, and the exit walk. Kill NOT met (see
CALLER_CENSUS.md): direct callers bypass evaluate entirely, and
evaluate itself performs no book-type validation (SF-BL-10963).

Pre-fix reproduction (verified live 2026-10-23, before the patch),
TASK fixture ask 0.50 / bid 0.48, fee 0.01, q_exec 0.505
(repro.py / repro_prefix.txt):
  clean book -> rt 0.0298, q_exit 0.4752, fill 1.0
  book=None  -> AttributeError: 'NoneType' object has no attribute 'fee_rate'
  book={}    -> AttributeError: 'dict' object has no attribute 'fee_rate'
  duck-type SimpleNamespace (fee_rate present) -> silently COMPUTED

Run:  python3 test_round_trip_book_validation.py  (exit 0 iff all pass)
No pytest dependency: plain asserts + __main__ runner, same convention
as tests/test_alpha_kelly_round_trip_shares.py.
"""
import os
import pickle
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster"))
from superforecaster import alpha_kelly as ak  # noqa: E402

PASS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(ask=0.50, bid=0.48, fee=0.01, bid_size=100000):
    return ak.OrderBookSnapshot(
        market_id="T-11004", title="fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C11004", est_days_to_resolution=7.0,
        yes_asks=[ak.OrderBookLevel(price=ask, size_shares=100000)],
        yes_bids=[ak.OrderBookLevel(price=bid, size_shares=bid_size)],
        fee_rate=fee)


def _sibling_book():
    return _book(ask=0.60, bid=0.58)


def _raises_named_book_error(book, side=ak.TradeAction.BUY_YES,
                             shares=1000.0, q_exec=0.505):
    """True iff the call raises a ValueError naming book /
    OrderBookSnapshot / SF-BL-11004 (and NOT an AttributeError)."""
    try:
        ak.round_trip_cost(book, side, shares, q_exec)
    except ValueError as e:
        msg = str(e)
        return ("book" in msg and "OrderBookSnapshot" in msg
                and "SF-BL-11004" in msg)
    except Exception:
        return False
    return False


# ---- 1. Clean fixtures are bit-identical post-fix --------------------------
rt, qx, ff = ak.round_trip_cost(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.505)
check("TASK clean rt_cost 0.0298", abs(rt - 0.0298) < 1e-9, f"rt={rt}")
check("TASK clean q_exit 0.4752", abs(qx - 0.4752) < 1e-9, f"q_exit={qx}")
check("TASK clean fill 1.0", ff == 1.0, f"fill={ff}")
# Sibling fixture (SF-BL-9882/9811/10466): ask 0.60 / bid 0.58, q_exec 0.606.
rt, qx, ff = ak.round_trip_cost(_sibling_book(), ak.TradeAction.BUY_YES, 1000.0, 0.606)
check("sibling clean rt/q_exit", abs(rt - 0.0318) < 1e-9 and abs(qx - 0.5742) < 1e-9,
      f"rt={rt} q_exit={qx}")
# BUY_NO clean on TASK book: exit derives from asks 0.50 -> NO bid 0.50,
# q_exit = 0.50 * 0.99 = 0.495, rt = 0.505 - 0.495 = 0.010.
rt, qx, ff = ak.round_trip_cost(_book(), ak.TradeAction.BUY_NO, 1000.0, 0.505)
check("clean BUY_NO q_exit/rt", abs(qx - 0.495) < 1e-9 and abs(rt - 0.010) < 1e-9,
      f"q_exit={qx} rt={rt}")
# No exit depth on a REAL book -> (q_exec, 0, 0) contract unchanged.
empty_exit = ak.OrderBookSnapshot(
    market_id="T-11004-E", title="no bids",
    domain=ak.MarketDomain.PREDICTION_MARKET,
    factor_cluster="C11004", est_days_to_resolution=7.0,
    yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=1000)],
    yes_bids=[], fee_rate=0.01)
rt, qx, ff = ak.round_trip_cost(empty_exit, ak.TradeAction.BUY_YES, 1000.0, 0.505)
check("no exit depth, real book -> (q_exec, 0, 0)",
      rt == 0.505 and qx == 0.0 and ff == 0.0, f"rt={rt} q_exit={qx} fill={ff}")
# Partial fill (SF-BL-9844 interplay): bids absorb 100 of 1000 shares,
# q_exit = 0.48 * 0.10 * 0.99 = 0.04752, rt = 0.505 - 0.04752 = 0.45748.
rt, qx, ff = ak.round_trip_cost(_book(bid_size=100), ak.TradeAction.BUY_YES, 1000.0, 0.505)
check("partial fill rt/q_exit exact",
      abs(qx - 0.04752) < 1e-9 and abs(rt - 0.45748) < 1e-9 and ff == 0.1,
      f"rt={rt} q_exit={qx} fill={ff}")

# ---- 2. TASK.md named cases: None and dict raise the named ValueError ------
check("reject book None (TASK named)", _raises_named_book_error(None))
check("reject book {} (TASK named)", _raises_named_book_error({}))
check("reject book dict-with-fee",
      _raises_named_book_error({"fee_rate": 0.01, "yes_asks": [], "yes_bids": []}))
# Even a COMPLETE field-for-field dict dump of a real book raises:
# the JSON-loaded shape is a dict however complete it is.
import dataclasses
full_dump = dataclasses.asdict(_book())
check("reject book full asdict() dump", _raises_named_book_error(full_dump))

# ---- 3. Extended non-snapshot set ------------------------------------------
for bad in ["book", 123, 0.505, True, False, [], (), ["x"]]:
    check(f"reject book {bad!r}", _raises_named_book_error(bad))
check("reject book OrderBookLevel (wrong dataclass)",
      _raises_named_book_error(ak.OrderBookLevel(price=0.50, size_shares=1000)))
# Duck-typed impostor: pre-fix this silently COMPUTED via attribute
# access; it carries none of the snapshot's construction invariants.
duck = SimpleNamespace(
    fee_rate=0.01,
    yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000)],
    yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100000)])
check("reject duck-type SimpleNamespace", _raises_named_book_error(duck))
# Both sides raise (validation is side-independent, before the walk):
check("reject BUY_NO book None",
      _raises_named_book_error(None, side=ak.TradeAction.BUY_NO))

# ---- 4. Validation precedes the fee check AND the exit walk ----------------
# A no-op call (SKIP side, zero shares) still raises the book error:
check("reject precedes no-op walk (SKIP side, 0 shares)",
      _raises_named_book_error(None, side=ak.TradeAction.SKIP, shares=0.0))
check("reject precedes walk on no-depth-shaped dict",
      _raises_named_book_error({"fee_rate": 0.01}))
# Before the FEE check: a dict carrying an OUT-OF-BOUND fee still
# raises the BOOK error, not the SF-BL-9882 fee error — the fee
# validator dereferences .fee_rate and cannot run on a dict at all.
try:
    ak.round_trip_cost({"fee_rate": -0.50}, ak.TradeAction.BUY_YES, 1000.0, 0.505)
    book_before_fee = False
except ValueError as e:
    # The book error names SF-BL-11004 / the type failure; the fee
    # error's distinguishing text is its bound clause (absent here).
    # (The book message mentions the SF-BL-9882 line historically, so
    # the bare task id is not the discriminator.)
    book_before_fee = ("SF-BL-11004" in str(e)
                       and "is not an OrderBookSnapshot" in str(e)
                       and "outside pre-registered bound" not in str(e))
check("book error precedes fee error (dict, bad fee inside)", book_before_fee)
# Before q_exec / shares validation too: a None book with a NaN
# q_exec AND zero shares still raises the BOOK error (signature
# order — the book precedes every other argument).
try:
    ak.round_trip_cost(None, ak.TradeAction.BUY_YES, 0.0, float("nan"))
    book_first = False
except ValueError as e:
    book_first = "SF-BL-11004" in str(e)
check("book error precedes q_exec/shares errors", book_first)

# ---- 5. Kill test: NO non-snapshot book returns a tuple / wrong error ------
ok = True
for bad in [None, {}, {"fee_rate": 0.01}, full_dump, "book", 123,
            True, [], duck,
            ak.OrderBookLevel(price=0.50, size_shares=1000)]:
    try:
        ak.round_trip_cost(bad, ak.TradeAction.BUY_YES, 1000.0, 0.505)
        ok = False  # returned a tuple — kill criterion triggered
    except ValueError as e:
        ok = ok and ("SF-BL-11004" in str(e))
    except Exception:
        ok = False  # wrong error type (e.g. bare AttributeError)
check("kill: no non-snapshot book returns a tuple / wrong error", ok)

# ---- 6. Genuine snapshots still accepted in every legitimate form ---------
class _SubBook(ak.OrderBookSnapshot):
    pass

sub = _SubBook(
    market_id="T-11004-SUB", title="subclass",
    domain=ak.MarketDomain.PREDICTION_MARKET,
    factor_cluster="C11004", est_days_to_resolution=7.0,
    yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000)],
    yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100000)],
    fee_rate=0.01)
rt, qx, ff = ak.round_trip_cost(sub, ak.TradeAction.BUY_YES, 1000.0, 0.505)
check("subclass instance computes bit-identical",
      abs(rt - 0.0298) < 1e-9 and abs(qx - 0.4752) < 1e-9 and ff == 1.0,
      f"rt={rt} q_exit={qx}")
# An unpickled REAL snapshot is still an instance (the type check is
# not a freshness check): computes bit-identical.
rt, qx, ff = ak.round_trip_cost(pickle.loads(pickle.dumps(_book())),
                                ak.TradeAction.BUY_YES, 1000.0, 0.505)
check("unpickled real snapshot computes bit-identical",
      abs(rt - 0.0298) < 1e-9 and abs(qx - 0.4752) < 1e-9 and ff == 1.0,
      f"rt={rt} q_exit={qx}")
# Helper surface agrees with the call-site behaviour:
check("_is_order_book_snapshot / _validate_round_trip_book surface",
      ak._is_order_book_snapshot(_book()) is True
      and ak._is_order_book_snapshot(None) is False
      and ak._is_order_book_snapshot({}) is False)

# ---- 7. Sibling validations still fire on real books (no regression) -------
mut = _book()
mut.fee_rate = -0.50  # post-construction mutation bypasses __post_init__
try:
    ak.round_trip_cost(mut, ak.TradeAction.BUY_YES, 1000.0, 0.505)
    fee_raises = False
except ValueError as e:
    fee_raises = "fee_rate" in str(e)
check("invalid fee on real book still raises fee error", fee_raises)
try:
    ak.round_trip_cost(_book(), ak.TradeAction.BUY_YES, 1000.0, float("nan"))
    qexec_raises = False
except ValueError as e:
    qexec_raises = "SF-BL-9811" in str(e)
check("invalid q_exec on real book still raises q_exec error", qexec_raises)
try:
    ak.round_trip_cost(_book(), ak.TradeAction.BUY_YES, 0.0, 0.505)
    shares_raises = False
except ValueError as e:
    shares_raises = "SF-BL-10466" in str(e)
check("invalid shares on real book still raises shares error", shares_raises)
# Sibling precedence on a real book preserved: fee > q_exec > shares.
try:
    ak.round_trip_cost(mut, ak.TradeAction.BUY_YES, 0.0, float("nan"))
    fee_prec = False
except ValueError as e:
    fee_prec = "fee_rate" in str(e)
check("fee precedence over q_exec/shares on real book preserved", fee_prec)

# ---- 8. Evaluate / Gate D path unchanged -----------------------------------
tight = ak.OrderBookSnapshot(
    market_id="T-11004-T", title="tight",
    domain=ak.MarketDomain.PREDICTION_MARKET,
    factor_cluster="C11004T", est_days_to_resolution=7.0,
    yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000)],
    yes_bids=[ak.OrderBookLevel(price=0.49, size_shares=100000)], fee_rate=0.01)
cl = {}
r = ak.evaluate_and_size_opportunity(tight, 0.80, 0.05, 10000.0, cl)
check("evaluate tight still BUY_YES", r.action == ak.TradeAction.BUY_YES,
      f"{r.action} {r.skip_reason}")
wide = ak.OrderBookSnapshot(
    market_id="T-11004-W", title="wide",
    domain=ak.MarketDomain.PREDICTION_MARKET,
    factor_cluster="C11004W", est_days_to_resolution=7.0,
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
