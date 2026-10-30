"""
test_round_trip_fee_validation.py — SF-BL-9882 verification tests.

Pre-registered done criterion: "round_trip_cost must reject or
clamp-with-warning any book.fee_rate outside [0, 1) so q_exit never
exceeds the walked bid and rt_cost is never negative-by-construction
or NaN; kill if a caller census shows every book reaching Gate D is
freshly constructed and re-validated after any mutation."

Fix under test: round_trip_cost re-validates book.fee_rate via the
SF-BL-9363 validator (_validate_fee_rate, pre-registered bound
[0, 0.10] — a strict subset of [0, 1)) and raises ValueError before
the exit walk. Kill NOT met (see CALLER_CENSUS.md): books are mutable,
direct callers bypass evaluate Gate 0.

Pre-fix reproduction (verified live 2026-10-21, before the patch),
fixture ask 0.60 / walked bid 0.58, q_exec 0.606, shares 1000:
  fee= 0.01 -> rt 0.0318, q_exit 0.5742, fill 1.0   (clean)
  fee=-0.50 -> rt -0.264, q_exit 0.870 (> 0.58 bid; Gate D veto fails open)
  fee= 2.0  -> rt 1.186,  q_exit -0.58 (negative exit price)
  fee= NaN  -> rt NaN,    q_exit NaN   (NaN > max is False; veto never fires)
  fee=+inf  -> rt +inf,   q_exit -inf

Run:  python3 test_round_trip_fee_validation.py   (exit 0 iff all pass)
No pytest dependency: plain asserts + __main__ runner, same convention
as tests/test_alpha_kelly_fee_rate.py.
"""
import math
import os
import pickle
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster"))
from superforecaster import alpha_kelly as ak  # noqa: E402

PASS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(fee=0.01):
    return ak.OrderBookSnapshot(
        market_id="T-9882", title="fixture", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C9882", est_days_to_resolution=7.0,
        yes_asks=[ak.OrderBookLevel(price=0.60, size_shares=100000)],
        yes_bids=[ak.OrderBookLevel(price=0.58, size_shares=100000)],
        fee_rate=fee)


def _mutated_book(fee):
    """Book whose fee_rate bypasses __post_init__ (post-construction mutation)."""
    b = _book(fee=0.01)
    b.fee_rate = fee
    return b


def _raises_value_error(book, side=ak.TradeAction.BUY_YES, shares=1000.0, q_exec=0.606):
    try:
        ak.round_trip_cost(book, side, shares, q_exec)
    except ValueError:
        return True
    except Exception:
        return False
    return False


# ---- 1. Clean fixture is byte-identical post-fix ---------------------------
rt, qx, ff = ak.round_trip_cost(_book(0.01), ak.TradeAction.BUY_YES, 1000.0, 0.606)
check("clean fee 0.01 rt_cost", abs(rt - 0.0318) < 1e-9, f"rt={rt}")
check("clean fee 0.01 q_exit", abs(qx - 0.5742) < 1e-9, f"q_exit={qx}")
check("clean fee 0.01 fill", ff == 1.0, f"fill={ff}")
# Boundary / in-bound fees still compute, with q_exit in [0, bid]:
for fee, exp_qx in [(0.0, 0.58), (0.10, 0.58 * 0.90), (0.005, 0.58 * 0.995)]:
    rt, qx, ff = ak.round_trip_cost(_book(fee), ak.TradeAction.BUY_YES, 1000.0, 0.606)
    check(f"in-bound fee {fee} q_exit", abs(qx - exp_qx) < 1e-9 and 0.0 <= qx <= 0.58,
          f"q_exit={qx}")
    check(f"in-bound fee {fee} rt finite", math.isfinite(rt), f"rt={rt}")
# BUY_NO side, clean: NO exit derives from asks 0.60 -> NO bid 0.40,
# q_exit = 0.40 * 0.99 = 0.396, rt = 0.606 - 0.396 = 0.210.
rt, qx, ff = ak.round_trip_cost(_book(0.01), ak.TradeAction.BUY_NO, 1000.0, 0.606)
check("clean BUY_NO q_exit/rt", abs(qx - 0.396) < 1e-9 and abs(rt - 0.210) < 1e-9,
      f"q_exit={qx} rt={rt}")
# No exit depth -> q_exit 0.0, rt == q_exec (unchanged contract).
empty_exit = ak.OrderBookSnapshot(
    market_id="T-9882-E", title="no bids", domain=ak.MarketDomain.PREDICTION_MARKET,
    factor_cluster="C9882", est_days_to_resolution=7.0,
    yes_asks=[ak.OrderBookLevel(price=0.60, size_shares=1000)],
    yes_bids=[], fee_rate=0.01)
rt, qx, ff = ak.round_trip_cost(empty_exit, ak.TradeAction.BUY_YES, 1000.0, 0.606)
check("no exit depth -> (q_exec, 0, 0)", rt == 0.606 and qx == 0.0 and ff == 0.0,
      f"rt={rt} q_exit={qx} fill={ff}")

# ---- 2. Every fee outside [0, 1) is rejected (task's named cases first) ----
for bad in [-0.50, 2.0, float("nan"), float("inf"), float("-inf"),
            1.0, 1.5, -0.001, -1e-9, -0.20]:
    check(f"reject fee {bad!r} (mutated)", _raises_value_error(_mutated_book(bad)))
# Non-numeric / bool fees are rejected too (cannot be ordered / not a rate).
for bad in ["0.01", None, True, False]:
    check(f"reject fee {bad!r} (mutated)", _raises_value_error(_mutated_book(bad)))

# ---- 3. Stricter system bound: (0.10, 1) also rejected, consistently -------
# [0, 0.10] is the SF-BL-9363 pre-registered bound used by the constructor,
# Gate 0 and the sweep; round_trip_cost must not accept a fee every other
# consumption site rejects.
for bad in [0.100001, 0.5, 0.99]:
    check(f"reject fee {bad} (system bound)", _raises_value_error(_mutated_book(bad)))

# ---- 4. Bypass paths: pickle round-trip and SKIP-side call -----------------
b = _book(0.01)
b.fee_rate = -0.50
b2 = pickle.loads(pickle.dumps(b))  # unpickling skips __post_init__
check("reject fee -0.50 (unpickled)", _raises_value_error(b2))
# Validation precedes the walk: even a SKIP side / zero shares raises,
# so no caller can smuggle an invalid fee past on a no-op call.
check("reject precedes no-op walk",
      _raises_value_error(_mutated_book(float("nan")), side=ak.TradeAction.SKIP,
                          shares=0.0))

# ---- 5. Invariant sweep: no accepted fee yields q_exit > bid or NaN rt -----
ok = True
for fee in [0.0, 0.001, 0.01, 0.05, 0.10]:
    rt, qx, ff = ak.round_trip_cost(_book(fee), ak.TradeAction.BUY_YES, 1000.0, 0.606)
    ok = ok and (0.0 <= qx <= 0.58) and math.isfinite(rt) and math.isfinite(qx)
check("invariant: q_exit in [0, bid], rt finite for all accepted fees", ok)

# ---- 6. Evaluate / Gate D path unchanged -----------------------------------
# Tight book still trades; wide book still vetoed by Gate D; a mutated
# invalid fee still SKIPs at Gate 0 (INVALID_FEE_RATE) before Gate D —
# the new raise inside round_trip_cost is never reached via evaluate.
tight = ak.OrderBookSnapshot(
    market_id="T-9882-T", title="tight", domain=ak.MarketDomain.PREDICTION_MARKET,
    factor_cluster="C9882T", est_days_to_resolution=7.0,
    yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100000)],
    yes_bids=[ak.OrderBookLevel(price=0.49, size_shares=100000)], fee_rate=0.01)
cl = {}
r = ak.evaluate_and_size_opportunity(tight, 0.80, 0.05, 10000.0, cl)
check("evaluate tight still BUY_YES", r.action == ak.TradeAction.BUY_YES,
      f"{r.action} {r.skip_reason}")
mut = _book(0.01)
mut.yes_asks = [ak.OrderBookLevel(price=0.50, size_shares=100000)]
mut.yes_bids = [ak.OrderBookLevel(price=0.49, size_shares=100000)]
mut.fee_rate = -0.50
cl = {}
r = ak.evaluate_and_size_opportunity(mut, 0.80, 0.05, 10000.0, cl)
check("evaluate mutated fee SKIPs INVALID_FEE_RATE, zero size, cluster untouched",
      r.action == ak.TradeAction.SKIP
      and bool(r.skip_reason) and r.skip_reason.startswith("INVALID_FEE_RATE")
      and r.invested_capital == 0.0 and r.shares_acquired == 0.0 and cl == {},
      f"{r.action} {r.skip_reason} cl={cl}")

print(f"\n{sum(PASS)}/{len(PASS)} passed")
sys.exit(0 if all(PASS) else 1)
