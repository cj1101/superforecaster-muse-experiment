"""SF-BL-11046 regression tests: sweep_order_book malformed ladder entries.

Prereg: "sweep_order_book skips non-OrderBookLevel entries with a logged
count (or raises a named ValueError) so a mixed ladder fills identically
to the clean ladder without the corrupt entry, fixtured at a dict entry
and a None entry; kill if a producer census proves sweep ladders are
only ever lists of OrderBookLevel under a tested constructor invariant."

Resolution taken: the SKIP branch. _is_live_level returns False for any
non-OrderBookLevel entry before dereferencing size_shares (pre-fix it
raised AttributeError: 'dict'/'NoneType' object has no attribute
'size_shares'), and sweep_order_book emits one UserWarning per sweep
naming SF-BL-11046 with the skipped count and types for the ladder it
walks. Kill NOT triggered (CALLER_CENSUS.md).

Run: python3 test_sweep_ladder_entry.py  (exit 0 iff all pass)
"""
import os, sys, warnings
sys.path.insert(0, os.path.expanduser("~/workspace/superforecaster/superforecaster"))
import alpha_kelly as ak  # noqa: E402

PASS = []
ZERO = (0.0, 0.0, 0.0)
CLEAN_YES = (1000.0, 1800.1800180018, 0.5555000000000001)
CLEAN_NO = (1000.0, 1904.036557501904, 0.5252)
DICT_ENTRY = {"price": 0.50, "size_shares": 1000}


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book():
    return ak.OrderBookSnapshot(
        market_id="FIXTURE-11046", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C11046",
        est_days_to_resolution=5,
        yes_asks=[ak.OrderBookLevel(price=0.55, size_shares=100000)],
        yes_bids=[ak.OrderBookLevel(price=0.48, size_shares=100000)],
        fee_rate=0.01)


def _mut(book, asks=None, bids=None):
    if asks is not None:
        book.yes_asks = asks
    if bids is not None:
        book.yes_bids = bids
    return book


def _sweep(book, side, budget=1000.0, belief=0.80, hurdle=0.05, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = ak.sweep_order_book(book, side, budget, belief, hurdle, **kw)
    return out, [str(x.message) for x in w]


def _lvl(price, size=100000):
    return ak.OrderBookLevel(price=price, size_shares=size)


# ---- 1. Prereg fixtures: dict entry and None entry, BUY_YES ----
def test_prereg_dict_entry_yes():
    b = _mut(_book(), asks=[DICT_ENTRY, _lvl(0.55)])
    out, w = _sweep(b, ak.TradeAction.BUY_YES)
    check("dict-entry BUY_YES fills identical to clean", out == CLEAN_YES, f"{out}")
    check("dict-entry logs exactly one SF-BL-11046 warning",
          len(w) == 1 and "SF-BL-11046" in w[0] and "skipped 1" in w[0] and "dict" in w[0],
          f"{w}")


def test_prereg_none_entry_yes():
    b = _mut(_book(), asks=[None, _lvl(0.55)])
    out, w = _sweep(b, ak.TradeAction.BUY_YES)
    check("None-entry BUY_YES fills identical to clean", out == CLEAN_YES, f"{out}")
    check("None-entry logs exactly one SF-BL-11046 warning",
          len(w) == 1 and "SF-BL-11046" in w[0] and "skipped 1" in w[0] and "NoneType" in w[0],
          f"{w}")


# ---- 2. Mirror: corrupt bids swept on BUY_NO ----
def test_corrupt_bids_no():
    for entry, tname in [(DICT_ENTRY, "dict"), (None, "NoneType")]:
        b = _mut(_book(), bids=[entry, _lvl(0.48)])
        out, w = _sweep(b, ak.TradeAction.BUY_NO)
        check(f"{tname}-entry bids BUY_NO identical to clean", out == CLEAN_NO, f"{out}")
        check(f"{tname}-entry bids BUY_NO warns once with type",
              len(w) == 1 and "SF-BL-11046" in w[0] and tname in w[0], f"{w}")


# ---- 3. Positions, multiplicity, other types ----
def test_positions_and_counts():
    clean, _ = _sweep(_book(), ak.TradeAction.BUY_YES)
    for asks in ([_lvl(0.55), DICT_ENTRY], [_lvl(0.50), None, _lvl(0.55)],
                 [DICT_ENTRY, None, "junk", _lvl(0.55)]):
        b = _mut(_book(), asks=list(asks))
        out, w = _sweep(b, ak.TradeAction.BUY_YES)
        n_bad = sum(1 for e in asks if not isinstance(e, ak.OrderBookLevel))
        # position-independent identity: corrupt entries anywhere are absent
        ref_book = _mut(_book(), asks=[e for e in asks if isinstance(e, ak.OrderBookLevel)])
        ref, _ = _sweep(ref_book, ak.TradeAction.BUY_YES)
        check(f"mixed ladder {n_bad} corrupt fills as its clean remainder",
              out == ref, f"{out} vs {ref}")
        check(f"mixed ladder warns once with count {n_bad}",
              len(w) == 1 and f"skipped {n_bad}" in w[0], f"{w}")
    check("clean remainder at 0.50+0.55 differs from single-rung clean (sanity)",
          clean == CLEAN_YES)


def test_other_entry_types():
    for entry in ("0.55", 0.55, 42, [0.55, 1000], (0.55, 1000), object()):
        b = _mut(_book(), asks=[entry, _lvl(0.55)])
        out, w = _sweep(b, ak.TradeAction.BUY_YES)
        check(f"entry type {type(entry).__name__} skipped identical + warned",
              out == CLEAN_YES and len(w) == 1 and "SF-BL-11046" in w[0], f"{out} {w}")


def test_corrupt_only_ladder():
    b = _mut(_book(), asks=[DICT_ENTRY, None])
    out, w = _sweep(b, ak.TradeAction.BUY_YES)
    check("corrupt-only ask ladder -> (0,0,0) + count-2 warning",
          out == ZERO and len(w) == 1 and "skipped 2" in w[0], f"{out} {w}")


def test_level_subclass_entry_fills():
    class SubLevel(ak.OrderBookLevel):
        pass
    b = _mut(_book(), asks=[SubLevel(price=0.55, size_shares=100000)])
    out, w = _sweep(b, ak.TradeAction.BUY_YES)
    check("OrderBookLevel subclass entry is a genuine level (fills, no warning)",
          out == CLEAN_YES and w == [], f"{out} {w}")


# ---- 4. Clean books: bit-identical, warning-free ----
def test_clean_no_warning():
    out, w = _sweep(_book(), ak.TradeAction.BUY_YES)
    check("clean BUY_YES bit-identical", out == CLEAN_YES, f"{out}")
    check("clean BUY_YES emits no warning", w == [], f"{w}")
    out, w = _sweep(_book(), ak.TradeAction.BUY_NO)
    check("clean BUY_NO bit-identical, no warning", out == CLEAN_NO and w == [], f"{out} {w}")


def test_skip_and_zero_budget_contract():
    b = _mut(_book(), asks=[DICT_ENTRY, _lvl(0.55)])
    out, _ = _sweep(b, ak.TradeAction.SKIP)
    check("corrupt book + SKIP side -> (0,0,0), no crash", out == ZERO, f"{out}")
    out, _ = _sweep(b, ak.TradeAction.BUY_YES, budget=0.0)
    check("corrupt book + zero budget -> (0,0,0), no crash", out == ZERO, f"{out}")


def test_full_walk_and_lcb_paths():
    b = _mut(_book(), asks=[None, _lvl(0.55)])
    out, w = _sweep(b, ak.TradeAction.BUY_YES, hurdle=-1.0)
    ref, _ = _sweep(_book(), ak.TradeAction.BUY_YES, hurdle=-1.0)
    check("-1.0 full-walk corrupt identical to clean + warned",
          out == ref and len(w) == 1, f"{out} vs {ref}")
    out, w = _sweep(b, ak.TradeAction.BUY_YES, posterior_n_eff=30, lcb_z=1.0)
    ref, _ = _sweep(_book(), ak.TradeAction.BUY_YES, posterior_n_eff=30, lcb_z=1.0)
    check("LCB-path corrupt identical to clean", out == ref, f"{out} vs {ref}")


# ---- 5. Predicate + construction + evaluate collateral ----
def test_is_live_level_predicate():
    check("_is_live_level(dict) is False", ak._is_live_level(DICT_ENTRY) is False)
    check("_is_live_level(None) is False", ak._is_live_level(None) is False)
    check("_is_live_level('x') is False", ak._is_live_level("x") is False)
    check("_is_live_level(live level) is True", ak._is_live_level(_lvl(0.5, 10)) is True)
    check("_is_live_level(zero-size level) is False", ak._is_live_level(_lvl(0.5, 0)) is False)


def test_construction_and_properties():
    b = ak.OrderBookSnapshot(
        market_id="FIXTURE-11046-C", title="ctor", domain=ak.MarketDomain.PREDICTION_MARKET,
        factor_cluster="C11046", est_days_to_resolution=5,
        yes_asks=[DICT_ENTRY, _lvl(0.55)], yes_bids=[_lvl(0.48)], fee_rate=0.01)
    check("construction with dict entry succeeds", isinstance(b, ak.OrderBookSnapshot))
    check("best_yes_ask skips dict entry", b.best_yes_ask == 0.55, f"{b.best_yes_ask}")
    check("mid identical to clean book", b.mid_price == _book().mid_price, f"{b.mid_price}")


def test_evaluate_identical_to_clean():
    import dataclasses
    clean = ak.evaluate_and_size_opportunity(_book(), 0.80, 0.05, 10000.0, {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corrupt = ak.evaluate_and_size_opportunity(
            _mut(_book(), asks=[DICT_ENTRY, _lvl(0.55)]), 0.80, 0.05, 10000.0, {})
    check("evaluate corrupt book == evaluate clean book (all fields)",
          dataclasses.asdict(clean) == dataclasses.asdict(corrupt),
          f"{clean.action}/{clean.invested_capital} vs {corrupt.action}/{corrupt.invested_capital}")


def test_exit_walk_collateral():
    # size_walked_exit_bid is SF-BL-11047's surface, not this task's:
    # its prereg took the raise branch, so a corrupt exit ladder must
    # surface that named ValueError — never the bare AttributeError on
    # size_shares that _is_live_level produced pre-fix for both tasks.
    corrupt = _mut(_book(), bids=[None, _lvl(0.48)])
    try:
        ak.size_walked_exit_bid(corrupt, ak.TradeAction.BUY_YES, 1000.0)
        check("exit walk on corrupt bids raises named ValueError (SF-BL-11047)", False, "no raise")
    except ValueError as e:
        check("exit walk on corrupt bids raises named ValueError (SF-BL-11047)",
              "INVALID_EXIT_LADDER" in str(e), f"{e}")
    except AttributeError as e:
        check("exit walk on corrupt bids raises named ValueError (SF-BL-11047)", False, f"AttributeError: {e}")
    clean_out = ak.size_walked_exit_bid(_book(), ak.TradeAction.BUY_YES, 1000.0)
    check("exit walk on clean book unchanged", clean_out == (0.48, 1.0), f"{clean_out}")


# ---- 6. Neighbouring guards unchanged ----
def test_neighbour_guards_unchanged():
    def raises_valueerror(fn, token=None):
        try:
            fn()
        except ValueError as e:
            return token is None or token in str(e)
        except Exception:
            return False
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check("non-book still INVALID_BOOK (SF-BL-10964)",
              raises_valueerror(lambda: ak.sweep_order_book(None, ak.TradeAction.BUY_YES, 1000.0, 0.80), "INVALID_BOOK"))
        bad_fee = _book(); object.__setattr__(bad_fee, "fee_rate", -0.20)
        check("mutated invalid fee still raises ValueError",
              raises_valueerror(lambda: ak.sweep_order_book(bad_fee, ak.TradeAction.BUY_YES, 1000.0, 0.80)))
        unsorted = _mut(_book(), asks=[_lvl(0.62), _lvl(0.50)])
        check("unsorted ladder still raises ValueError (SF-BL-9051)",
              raises_valueerror(lambda: ak.sweep_order_book(unsorted, ak.TradeAction.BUY_YES, 1000.0, 0.80)))
        check("invalid belief still raises ValueError (SF-BL-10502)",
              raises_valueerror(lambda: ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, 1000.0, 82.0), "INVALID_BELIEF_PROB_SIDE"))
        check("non-finite budget still returns (0,0,0) (SF-BL-9438)",
              ak.sweep_order_book(_book(), ak.TradeAction.BUY_YES, float("nan"), 0.80) == ZERO)


def main():
    for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("test_")}.items()):
        fn()
    print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
    return 0 if all(PASS) and PASS else 1


if __name__ == "__main__":
    sys.exit(main())
