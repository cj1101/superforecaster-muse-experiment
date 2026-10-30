"""SF-BL-10964 regression tests: sweep_order_book book-argument type.

Prereg: "sweep_order_book must raise a named ValueError for a
non-OrderBookSnapshot book, fixtured at None and a dict, with enum-side
fills on constructed books bit-identical and the SKIP (0.0, 0.0, 0.0)
contract unchanged; kill if a caller census shows the sweep is only
ever reached with constructed books under a tested invariant."

Pre-fix (repro.py): book=None -> AttributeError 'NoneType' has no
attribute 'fee_rate'; book=dict -> AttributeError 'dict' ...; a
duck-typed fake swept successfully, bypassing construction validation.

Fix: sweep_order_book calls _validate_sweep_book(book) before ANY
dereference; non-OrderBookSnapshot raises ValueError INVALID_BOOK
(SF-BL-10964). Subclasses pass isinstance and sweep identically.

Run: python3 test_sweep_book_type.py  (exit 0 iff all pass)
"""
import io, os, sys
from contextlib import redirect_stdout
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "superforecaster"))
if not os.path.exists(os.path.join(sys.path[0], "alpha_kelly.py")):
    sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster/superforecaster"))
import alpha_kelly as ak  # noqa: E402

PASS = []
ZERO = (0.0, 0.0, 0.0)

def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond)); return cond

def _book(fee=0.01):
    return ak.OrderBookSnapshot(
        market_id="FIXTURE-10964", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C10964",
        est_days_to_resolution=5,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100_000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100_000)],
        fee_rate=fee)

def _raises(fn, token="INVALID_BOOK"):
    try:
        fn()
    except ValueError as e:
        assert token in str(e), (token, str(e))
        if token == "INVALID_BOOK":
            assert "SF-BL-10964" in str(e), str(e)
            assert "OrderBookSnapshot" in str(e), str(e)
        return
    except Exception as e:
        raise AssertionError(f"wrong exception {type(e).__name__}: {e}")
    raise AssertionError(f"no ValueError raised (expected {token})")

# ---- 1. Prereg fixtures: None and dict ----
def test_none_book_raises():
    _raises(lambda: ak.sweep_order_book(None, ak.TradeAction.BUY_YES, 1000.0, 0.80, 0.05))
    _raises(lambda: ak.sweep_order_book(None, ak.TradeAction.BUY_NO, 1000.0, 0.80, 0.05))
    _raises(lambda: ak.sweep_order_book(None, ak.TradeAction.SKIP, 1000.0, 0.80, 0.05))

def test_dict_book_raises():
    _raises(lambda: ak.sweep_order_book({}, ak.TradeAction.BUY_YES, 1000.0, 0.80, 0.05))
    _raises(lambda: ak.sweep_order_book({"fee_rate": 0.01, "yes_asks": [], "yes_bids": []},
                                        ak.TradeAction.BUY_YES, 1000.0, 0.80, 0.05))
    _raises(lambda: ak.sweep_order_book({"market_id": "X"}, ak.TradeAction.BUY_NO, 1000.0, 0.80))

# ---- 2. Other non-book classes ----
def test_other_types_raise_not_attributeerror():
    for bad in ("book", "", 42, 0, 0.5, True, [], (), object(),
                ak.OrderBookLevel(price=0.5, size_shares=1000)):
        _raises(lambda b=bad: ak.sweep_order_book(b, ak.TradeAction.BUY_YES, 1000.0, 0.80, 0.05))

def test_duck_typed_fake_raises():
    class Fake:
        fee_rate = 0.01
        yes_asks = [ak.OrderBookLevel(price=0.50, size_shares=100_000)]
        yes_bids = [ak.OrderBookLevel(price=0.48, size_shares=100_000)]
        mid_price = 0.49
    _raises(lambda: ak.sweep_order_book(Fake(), ak.TradeAction.BUY_YES, 1000.0, 0.80, 0.05))

def test_raises_before_other_guards():
    # Book validation is first: it fires on SKIP side, zero/non-finite
    # budget, and even with an invalid belief/hurdle (no book to check
    # those against).
    _raises(lambda: ak.sweep_order_book(None, ak.TradeAction.SKIP, 1000.0, 0.80, 0.05))
    _raises(lambda: ak.sweep_order_book(None, ak.TradeAction.BUY_YES, 0.0, 0.80, 0.05))
    _raises(lambda: ak.sweep_order_book(None, ak.TradeAction.BUY_YES, float("nan"), 0.80, 0.05))
    _raises(lambda: ak.sweep_order_book({}, ak.TradeAction.BUY_YES, 1000.0, 82.0, float("nan")))
    # keyword-argument call path (naive baseline / budget_sensitivity style)
    _raises(lambda: ak.sweep_order_book(book=None, side=ak.TradeAction.BUY_YES,
                                        target_budget=1200.0, belief_prob_side=0.8, min_net_edge=-1.0))
    # LCB path also validates first
    _raises(lambda: ak.sweep_order_book(None, ak.TradeAction.BUY_YES, 1000.0, 0.80, 0.05,
                                        posterior_n_eff=30, lcb_z=1.0))

def test_error_names_offending_type():
    for bad, tname in [(None, "NoneType"), ({}, "dict"), ("x", "str"), (42, "int")]:
        try:
            ak.sweep_order_book(bad, ak.TradeAction.BUY_YES, 1000.0, 0.80)
        except ValueError as e:
            assert tname in str(e), str(e)
        else:
            raise AssertionError(f"no raise for {bad!r}")

# ---- 3. Constructed books: bit-identical fills + SKIP contract ----
def test_enum_side_fills_bit_identical():
    assert ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.80, 0.05) == (1000.0, 1980.19801980198, 0.505)
    assert ak.sweep_order_book(_book(), ak.TradeAction.BUY_NO, 1000.0, 0.80, 0.05) == (1000.0, 1904.036557501904, 0.5252)

def test_skip_contract_unchanged():
    assert ak.sweep_order_book(_book(), ak.TradeAction.SKIP, 1000.0, 0.80, 0.05) == ZERO
    assert ak.sweep_order_book(_book(), ak.TradeAction.SKIP, 1000.0, 0.80) == ZERO
    assert ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 0.0, 0.80, 0.05) == ZERO

def test_subclass_book_accepted():
    class SubBook(ak.OrderBookSnapshot):
        pass
    b = SubBook(market_id="SUB", title="sub", domain=ak.MarketDomain.PREDICTION_MARKET,
                factor_cluster="C10964-SUB", est_days_to_resolution=5,
                yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100_000)],
                yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100_000)], fee_rate=0.01)
    assert ak.sweep_order_book(b, ak.TradeAction.BUY_YES, 1000.0, 0.80, 0.05) == (1000.0, 1980.19801980198, 0.505)

def test_valid_guards_still_apply_on_real_books():
    # Invalid fee on a genuine (mutated) book still raises fee error first among book-internal guards
    b = _book(); b.fee_rate = -0.20
    try:
        ak.sweep_order_book(b, ak.TradeAction.BUY_YES, 1000.0, 0.80)
    except ValueError as e:
        assert "fee_rate" in str(e) and "INVALID_BOOK" not in str(e), str(e)
    else:
        raise AssertionError("invalid fee did not raise")
    # Invalid belief on a genuine book still raises INVALID_BELIEF_PROB_SIDE
    _raises(lambda: ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000.0, 82.0, 0.05),
            token="INVALID_BELIEF_PROB_SIDE")
    # Non-finite budget on a genuine book still returns (0,0,0) (SF-BL-9438)
    assert ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, float("nan"), 0.80, 0.05) == ZERO
    assert ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, float("inf"), 0.80, 0.05) == ZERO

def test_no_fill_and_partial_fill_unchanged():
    # Belief below ask: genuine no-edge stop, not an error
    assert ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.40, 0.05) == ZERO
    # -1.0 full-walk override (naive baseline) unchanged
    out = ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000.0, 0.80, -1.0)
    assert out == (1000.0, 1980.19801980198, 0.505), out

# ---- 4. No regression: evaluate + benchmark ----
def test_evaluate_control_unchanged():
    b = ak.OrderBookSnapshot(market_id="FIXTURE", title="control",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C10964-CTL",
        est_days_to_resolution=5,
        yes_asks=[ak.OrderBookLevel(price=0.50, size_shares=100_000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100_000)], fee_rate=0.0)
    used = {}
    r = ak.evaluate_and_size_opportunity(b, 0.80, 0.0, 10000.0, used)
    assert r.action == ak.TradeAction.BUY_YES, r
    assert r.invested_capital == 1500.0 and r.shares_acquired == 3000.0, r

def test_benchmark_runs():
    buf = io.StringIO()
    with redirect_stdout(buf):
        ak.run_benchmark_simulation()
    out = buf.getvalue()
    assert "north_star_score" in out, out[-1000:]

ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
if __name__ == "__main__":
    for fn in ALL_TESTS:
        try:
            fn(); check(fn.__name__, True)
        except AssertionError as e:
            check(fn.__name__, False, str(e))
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
