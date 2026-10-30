"""
test_alpha_kelly_invalid_book.py — SF-BL-10963 verification tests.

Pre-registered done/kill criterion: "evaluate must SKIP zero-size with
a named invalid-book reason (cluster dict untouched) or raise a named
ValueError at entry when book is not an OrderBookSnapshot, fixtured at
None and a dict, with constructed-book fixtures bit-identical; kill if
a caller census shows book is only ever a constructed OrderBookSnapshot
under a tested invariant."

Kill NOT triggered (CALLER_CENSUS.md in the task work dir): the sole
production caller is the benchmark's literal constructions, the type
annotation is unenforced, no deserialiser materialises snapshots from
JSON under any test, and no test pins the invariant.

Resolution (PREREG.md): the prereg's ValueError branch — a SKIP record
copies its identity fields FROM the book, so for book=None there is no
honest market_id/title/domain/cluster to record, and a dict's domain
is a plain string; fabricating placeholders would write a corrupt
trade record. `_validate_book_argument` therefore raises a ValueError
naming INVALID_BOOK + SF-BL-10963 + the offending type, as the FIRST
statement of evaluate (before Gate E0, whose SKIP construction itself
dereferences book.*), so no gate runs and the cluster dict is
untouched by construction.

Pre-fix harm (repro.py / repro_prefix.txt in the task work dir):
book=None and book=dict (the JSON-loaded shape) both aborted with a
bare AttributeError on book.fee_rate; str/int books likewise.

Run:  python3 test_alpha_kelly_invalid_book.py  (exit 0 iff all pass)
"""
import copy
import os
import sys
from types import SimpleNamespace

for _cand in (
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "superforecaster"),
    "/home/hatch/workspace/superforecaster/superforecaster",
):
    if os.path.isdir(_cand):
        sys.path.insert(0, _cand)
        break

import alpha_kelly as ak  # noqa: E402

PASS = []
L = ak.OrderBookLevel


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    PASS.append(bool(cond))
    return cond


def _book(p_cluster="C10963"):
    return ak.OrderBookSnapshot(
        market_id="FIXTURE", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster=p_cluster,
        est_days_to_resolution=7.0,
        yes_asks=[L(price=0.50, size_shares=100_000),
                  L(price=0.62, size_shares=100_000)],
        yes_bids=[L(price=0.48, size_shares=100_000),
                  L(price=0.40, size_shares=100_000)],
        fee_rate=0.01)


JSON_BOOK = {
    "market_id": "FIXTURE", "title": "prereg fixture",
    "domain": "prediction_market", "factor_cluster": "C10963",
    "est_days_to_resolution": 7.0,
    "yes_asks": [{"price": 0.50, "size_shares": 100000},
                 {"price": 0.62, "size_shares": 100000}],
    "yes_bids": [{"price": 0.48, "size_shares": 100000},
                 {"price": 0.40, "size_shares": 100000}],
    "fee_rate": 0.01, "on_chain_verified": True, "snapshot_ts": None,
}


def _evaluate(book, p=0.80, cluster=None, **kw):
    cl = {} if cluster is None else cluster
    return ak.evaluate_and_size_opportunity(book, p, 0.05, 100_000.0, cl, **kw), cl


def _invalid_book_error(book, **kw):
    """Return the ValueError raised for `book`, or None if none raised."""
    try:
        _evaluate(book, **kw)
    except ValueError as e:
        return e
    return None


# ---- 1. The prereg fixtures: None and dict ---------------------------------
def test_none_book_raises_named_valueerror():
    cl = {"OTHER": 0.10}
    snapshot = dict(cl)
    try:
        ak.evaluate_and_size_opportunity(None, 0.80, 0.05, 100_000.0, cl)
        raise AssertionError("no raise")
    except ValueError as e:
        msg = str(e)
        assert "INVALID_BOOK" in msg and "SF-BL-10963" in msg, msg
        assert "NoneType" in msg, msg
    except AttributeError as e:  # pre-fix failure mode must be gone
        raise AssertionError(f"bare AttributeError: {e}")
    assert cl == snapshot, cl


def test_dict_book_raises_named_valueerror():
    cl = {}
    try:
        ak.evaluate_and_size_opportunity(
            copy.deepcopy(JSON_BOOK), 0.80, 0.05, 100_000.0, cl)
        raise AssertionError("no raise")
    except ValueError as e:
        msg = str(e)
        assert "INVALID_BOOK" in msg and "SF-BL-10963" in msg, msg
        assert "dict" in msg, msg
    except AttributeError as e:
        raise AssertionError(f"bare AttributeError: {e}")
    assert cl == {}, cl


def test_error_is_valueerror_not_attribute_or_typeerror():
    for bad in (None, copy.deepcopy(JSON_BOOK), "FIXTURE", 42, 0.5,
                ["not", "a", "book"], object()):
        e = _invalid_book_error(bad)
        assert e is not None, bad
        assert "INVALID_BOOK" in str(e) and "SF-BL-10963" in str(e), (bad, e)


# ---- 2. Duck-typed lookalikes are rejected ----------------------------------
def test_duck_typed_lookalike_rejected():
    # A SimpleNamespace carrying every attribute evaluate reads (the
    # pre-fix crash was an AttributeError, so a lookalike with the
    # attributes present would have slipped through attribute access):
    # the contract is the dataclass + its __post_init__ invariants.
    real = _book()
    fake = SimpleNamespace(
        market_id=real.market_id, title=real.title, domain=real.domain,
        factor_cluster=real.factor_cluster,
        est_days_to_resolution=real.est_days_to_resolution,
        yes_asks=real.yes_asks, yes_bids=real.yes_bids,
        fee_rate=real.fee_rate, on_chain_verified=real.on_chain_verified,
        snapshot_ts=None)
    e = _invalid_book_error(fake)
    assert e is not None and "SimpleNamespace" in str(e), e


def test_orderbooklevel_is_not_a_book():
    e = _invalid_book_error(L(price=0.50, size_shares=1000))
    assert e is not None and "OrderBookLevel" in str(e), e


# ---- 3. Precedence: the entry check is first --------------------------------
def test_invalid_book_beats_invalid_posterior():
    e = _invalid_book_error(None, p=float("nan"))
    assert e is not None and "INVALID_BOOK" in str(e), e


def test_invalid_book_beats_invalid_config():
    # Gate E0's SKIP construction dereferences book.*, so the book check
    # must precede it: bad book + bad config names the BOOK.
    cl = {}
    try:
        ak.evaluate_and_size_opportunity(
            copy.deepcopy(JSON_BOOK), 0.80, 0.05, 100_000.0, cl, config=None)
        raise AssertionError("no raise")
    except ValueError as e:
        assert "INVALID_BOOK" in str(e), e
    assert cl == {}, cl


# ---- 4. Constructed books are bit-identical ---------------------------------
def test_constructed_book_trades_identically():
    r, cl = _evaluate(_book())
    assert r.action == ak.TradeAction.BUY_YES, r
    assert r.invested_capital == 15000.0 and r.exec_price == 0.505, r
    assert r.liquidation_bid_price == 0.48, r
    assert cl == {"C10963": 0.15}, cl


def test_constructed_book_skip_paths_unchanged():
    # No-edge constructed book still SKIPs INSUFFICIENT_NET_EDGE.
    r, cl = _evaluate(_book(), p=0.52)
    assert r.action == ak.TradeAction.SKIP, r
    assert r.skip_reason.startswith("INSUFFICIENT_NET_EDGE"), r.skip_reason
    assert r.invested_capital == 0.0 and cl == {}, (r, cl)
    # Empty constructed book still SKIPs EMPTY_BOOK.
    empty = ak.OrderBookSnapshot(
        market_id="FIXTURE", title="prereg fixture",
        domain=ak.MarketDomain.PREDICTION_MARKET, factor_cluster="C10963",
        est_days_to_resolution=7.0, yes_asks=[], yes_bids=[], fee_rate=0.01)
    r, cl = _evaluate(empty)
    assert r.action == ak.TradeAction.SKIP, r
    assert r.skip_reason.startswith("EMPTY_BOOK"), r.skip_reason
    assert cl == {}, cl
    # Valid book + invalid config still SKIPs INVALID_CONFIG (Gate E0),
    # i.e. the book check did not swallow the config gate.
    r, cl = _evaluate(_book(), config=None)
    assert r.action == ak.TradeAction.SKIP, r
    assert r.skip_reason.startswith("INVALID_CONFIG"), r.skip_reason
    assert cl == {}, cl


def test_subclass_book_passes():
    class SubBook(ak.OrderBookSnapshot):
        pass

    b = _book()
    sub = SubBook(**{f: getattr(b, f) for f in (
        "market_id", "title", "domain", "factor_cluster",
        "est_days_to_resolution", "yes_asks", "yes_bids", "fee_rate",
        "on_chain_verified", "snapshot_ts")})
    r, cl = _evaluate(sub)
    assert r.action == ak.TradeAction.BUY_YES and r.invested_capital == 15000.0, r
    assert cl == {"C10963": 0.15}, cl


def test_materialised_json_book_trades_identically():
    # The remediation the error message names: materialise the dict as
    # an OrderBookSnapshot (converting domain + ladder entries), and it
    # trades bit-identically to the directly constructed book.
    d = copy.deepcopy(JSON_BOOK)
    mat = ak.OrderBookSnapshot(
        market_id=d["market_id"], title=d["title"],
        domain=ak.MarketDomain(d["domain"]),
        factor_cluster=d["factor_cluster"],
        est_days_to_resolution=d["est_days_to_resolution"],
        yes_asks=[L(**lvl) for lvl in d["yes_asks"]],
        yes_bids=[L(**lvl) for lvl in d["yes_bids"]],
        fee_rate=d["fee_rate"], on_chain_verified=d["on_chain_verified"],
        snapshot_ts=d["snapshot_ts"])
    r_mat, cl_mat = _evaluate(mat)
    r_ref, cl_ref = _evaluate(_book())
    assert (r_mat.action, r_mat.invested_capital, r_mat.exec_price,
            r_mat.shares_acquired, r_mat.liquidation_bid_price) == (
        r_ref.action, r_ref.invested_capital, r_ref.exec_price,
        r_ref.shares_acquired, r_ref.liquidation_bid_price), r_mat
    assert cl_mat == cl_ref, (cl_mat, cl_ref)


# ---- 5. Benchmark slate unchanged --------------------------------------------
def test_benchmark_unchanged():
    import contextlib
    import io

    def _run():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ak.run_benchmark_simulation()
        return [ln for ln in buf.getvalue().splitlines()
                if "North Star Score" in ln]

    with_validation = _run()
    saved = ak._validate_book_argument
    ak._validate_book_argument = lambda book: None
    try:
        without_validation = _run()
    finally:
        ak._validate_book_argument = saved
    assert with_validation and with_validation == without_validation, (
        with_validation, without_validation)


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for fn in ALL_TESTS:
        try:
            fn()
            check(fn.__name__, True)
        except AssertionError as e:
            check(fn.__name__, False, str(e))
    print(f"\n==== {sum(PASS)}/{len(PASS)} checks passed ====")
    raise SystemExit(0 if all(PASS) else 1)
