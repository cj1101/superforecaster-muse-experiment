"""
Workstream D — Prediction Markets forecaster (Superforecaster project).

Question: how calibrated are prediction-market prices, and can a simple,
fitted-out-of-sample adjustment rule beat the raw market price?

Data (all public, no credentials, research only — no betting):
  * Polymarket Gamma API (resolved markets):
        https://gamma-api.polymarket.com/markets?closed=true
        fetched in monthly end-date windows, ordered by volumeNum desc,
        so the sample is the highest-volume binary Yes/No markets of each
        month (Jan 2024 - Sep 2026), not the flood of tiny sports props
        that dominates an unfiltered "most recently closed" query.
  * Polymarket CLOB price history (pre-resolution price paths):
        https://clob.polymarket.com/prices-history?market=<clobTokenId>
            &interval=max&fidelity=1440
  * Kalshi public API was tried (status=settled works; the task brief's
    status=finalized returns bad_request). Its settled feed is dominated
    by multivariate combo markets with last_price/yes_bid = null, so it
    is NOT used for scoring — see RESULTS.md caveats. The attempt and its
    outcome are recorded in results.json.

PRICE PROXY — read this before trusting any number:
  Gamma's `outcomePrices` for a resolved market is [1,0] or [0,1] (the
  resolution itself), and `lastTradePrice` is ~0/1 at close. Neither is
  a forecast. The forecast price used here is therefore taken from the
  CLOB daily price history at fixed horizons BEFORE the market closed:
  snapshots at T-30d, T-14d, T-7d, T-3d and T-1d (last daily print at or
  before close - horizon, and no older than horizon + 14 days, so the
  print is reasonably fresh). Each (market, horizon) snapshot is one
  forecast sample with known time-to-expiry. The headline comparison
  uses the T-7d snapshot only (one sample per market); pooled snapshots
  are reported as a secondary analysis. Gamma `liquidityNum` is ~0 for
  resolved markets (liquidity is withdrawn at close), so lifetime
  `volumeNum` is used as the liquidity/trust proxy and is named as such.

Split: markets are split fit / test by closing date (earlier 60% of
markets = fit, later 40% = test). All rule parameters are chosen by
grid search on the FIT split only. The "best rule" named in the
calibration table is the rule with the lowest FIT Brier, not the
lowest test Brier — no test-set cherry-picking.

Adjustment rules (each a pure function price -> adjusted probability;
extra inputs are days_to_expiry and log10 volume):
  raw                 identity
  shrink_fixed        p' = 0.5 + s * (p - 0.5)
  logit_shrink        logit(p') = b * logit(p)         (temperature/Platt)
  logit_time          logit(p') = (b0 + b1*log1p(days)) * logit(p)
  logit_time_vol      logit(p') = (b0 + b1*log1p(days) + b2*z_logvol)
                                  * logit(p)
  bucket_recal        per-decile empirical recalibration fitted on fit,
                      shrunk toward identity by bucket sample size

Run:  python3 predmarkets.py            (uses/creates data/ cache)
      python3 predmarkets.py --refresh  (re-fetch everything)
Outputs: results.json and RESULTS.md next to this file, raw API
responses cached under data/.
"""

from __future__ import annotations

import argparse
import json
import math
import numbers
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
GAMMA = "https://gamma-api.polymarket.com/markets"
CLOB = "https://clob.polymarket.com/prices-history"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2/markets"

HORIZONS_DAYS = [30, 14, 7, 3, 1]
PRIMARY_HORIZON = 7
TOP_PER_MONTH = 25
MIN_VOLUME = 50_000.0
WINDOWS = [(y, m) for y in (2024, 2025, 2026) for m in range(1, 13)
           if (y, m) <= (2026, 9)]
EPS = 1e-6


# ---------------------------------------------------------------- IO ----
def _get_json(url: str, tries: int = 3):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001 - retry any transient failure
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def fetch_gamma(refresh: bool = False) -> list[dict]:
    DATA.mkdir(parents=True, exist_ok=True)
    markets: dict[str, dict] = {}
    for (y, m) in WINDOWS:
        cache = DATA / f"gamma_{y}-{m:02d}.json"
        if cache.exists() and not refresh:
            rows = json.loads(cache.read_text())
        else:
            start = f"{y}-{m:02d}-01T00:00:00Z"
            ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
            end = f"{ny}-{nm:02d}-01T00:00:00Z"
            url = (f"{GAMMA}?limit={TOP_PER_MONTH}&closed=true"
                   f"&order=volumeNum&ascending=false"
                   f"&end_date_min={start}&end_date_max={end}")
            rows = _get_json(url)
            cache.write_text(json.dumps(rows))
            time.sleep(0.2)
        for r in rows:
            markets[r["id"]] = r
    return list(markets.values())


def fetch_kalshi_probe() -> dict:
    """Try Kalshi settled markets; record whether fields are usable."""
    cache = DATA / "kalshi_probe.json"
    if cache.exists():
        return json.loads(cache.read_text())
    out = {"attempted": True, "usable": False, "note": ""}
    try:
        d = _get_json(f"{KALSHI}?limit=100&status=settled")
        ms = d.get("markets", [])
        with_result = [x for x in ms if x.get("result") in ("yes", "no")]
        with_price = [x for x in with_result if x.get("last_price")]
        out.update(n_returned=len(ms), n_with_result=len(with_result),
                   n_with_last_price=len(with_price))
        out["note"] = ("status=finalized -> bad_request; status=settled works "
                       "but the feed is dominated by multivariate combo "
                       "markets and last_price is null, so Kalshi is not "
                       "scored. Polymarket is the scored source.")
        if with_price:
            out["usable"] = True
    except Exception as e:  # noqa: BLE001
        out["note"] = f"Kalshi fetch failed: {e}"
    cache.write_text(json.dumps(out, indent=2))
    return out


def fetch_clob(token: str, refresh: bool = False) -> list[dict]:
    cdir = DATA / "clob"
    cdir.mkdir(parents=True, exist_ok=True)
    cache = cdir / f"{token}.json"
    if cache.exists() and not refresh:
        try:
            return json.loads(cache.read_text())
        except Exception:  # noqa: BLE001
            pass
    try:
        d = _get_json(f"{CLOB}?market={token}&interval=max&fidelity=1440")
    except Exception:  # noqa: BLE001
        # SF-BL-11231: a failed fetch is NOT an empty history. Return
        # [] transiently for this call (build_snapshots drops the
        # market this run) but write NOTHING to the cache — previously
        # the except branch fell through to cache.write_text('[]'),
        # persisting the outage as data, so every later call served
        # the cached empty history with zero network calls and the
        # market's snapshots were dropped for good. Not writing also
        # preserves a pre-existing good cache file on refresh=True.
        # The next call after recovery therefore retries the network
        # and returns (and caches) the real history. A genuinely
        # empty API history still takes the success path below and
        # caches/returns [] exactly as before.
        return []
    hist = d.get("history", []) if isinstance(d, dict) else []
    cache.write_text(json.dumps(hist))
    return hist


# ------------------------------------------------------------- dataset ----
CLEAN_MARKETS_COLUMNS = [
    "market_id", "question", "category", "closed_time",
    "outcome", "volume", "liquidity_at_close", "token_yes",
]


def _empty_clean_markets() -> pd.DataFrame:
    """Correctly-columned empty clean_markets result (SF-BL-10144)."""
    return pd.DataFrame({
        "market_id": pd.Series([], dtype=object),
        "question": pd.Series([], dtype=object),
        "category": pd.Series([], dtype=object),
        "closed_time": pd.Series([], dtype="datetime64[ns, UTC]"),
        "outcome": pd.Series([], dtype="int64"),
        "volume": pd.Series([], dtype="float64"),
        "liquidity_at_close": pd.Series([], dtype="float64"),
        "token_yes": pd.Series([], dtype=object),
    }, columns=CLEAN_MARKETS_COLUMNS)


def clean_markets(raw: list[dict]) -> pd.DataFrame:
    """Clean raw Gamma rows into the resolved binary market panel.

    SF-BL-10144: a record whose volumeNum or liquidityNum cannot be
    parsed as a float (e.g. 'abc') is SKIPPED and counted — it never
    aborts the whole clean. The counts are surfaced on the returned
    frame as ``df.attrs["n_excluded_non_numeric"]`` (total),
    ``df.attrs["n_excluded_non_numeric_volume"]``,
    ``df.attrs["n_excluded_non_numeric_liquidity"]`` and
    ``df.attrs["excluded_non_numeric_by_field"]``
    (``{"volumeNum": n, "liquidityNum": n}``); they are present (0) on
    a fully clean panel. Missing/None/empty numeric fields keep the
    pre-fix ``or 0.0`` semantics and are not counted.
    SF-BL-9744: a record whose volume parses but is NON-FINITE
    ('NaN'/'nan', a float NaN, 'inf'/'+Infinity'/'1e400', '-inf') is
    likewise SKIPPED and counted — previously NaN and +inf survived
    the ``vol < MIN_VOLUME`` filter (both comparisons are False) and
    poisoned ``log10_volume`` downstream, and -inf was dropped
    silently as if below-minimum. The finiteness screen runs BEFORE
    the MIN_VOLUME filter, so every non-finite volume is counted as
    such, in ``df.attrs["n_excluded_non_finite"]`` (total),
    ``df.attrs["n_excluded_non_finite_volume"]`` and
    ``df.attrs["excluded_non_finite_by_field"]``
    (``{"volumeNum": n}``); the combined "non-finite or unparseable
    volume" count the criterion names is
    ``df.attrs["n_excluded_bad_volume"]``
    (= non-numeric-volume + non-finite-volume). These are present (0)
    on a fully clean panel, and every finite-volume row is identical
    to the pre-fix output. Liquidity finiteness is NOT screened here
    (not this item's criterion; liquidity follows the pre-fix path).
    When no rows survive (empty input or every record filtered), a
    correctly-columned empty DataFrame (CLEAN_MARKETS_COLUMNS, typed
    closed_time/outcome/volume/liquidity_at_close) is returned instead
    of raising KeyError 'closed_time' on the final sort.
    """
    rows = []
    n_excluded_volume = 0
    n_excluded_liquidity = 0
    n_excluded_nonfinite_volume = 0
    for m in raw:
        try:
            outcomes = json.loads(m.get("outcomes") or "[]")
            prices = json.loads(m.get("outcomePrices") or "[]")
            toks = json.loads(m.get("clobTokenIds") or "[]")
        except Exception:  # noqa: BLE001
            continue
        if outcomes != ["Yes", "No"] or len(toks) < 1:
            continue
        if prices not in (["1", "0"], ["0", "1"]):
            continue  # not a clean binary resolution (e.g. 0/0 legacy)
        try:
            vol = float(m.get("volumeNum") or 0.0)
        except (TypeError, ValueError, OverflowError):
            n_excluded_volume += 1
            continue
        if not math.isfinite(vol):
            n_excluded_nonfinite_volume += 1
            continue
        if vol < MIN_VOLUME:
            continue
        ct = pd.to_datetime(m.get("closedTime"), utc=True, errors="coerce")
        if pd.isna(ct):
            continue
        try:
            liquidity = float(m.get("liquidityNum") or 0.0)
        except (TypeError, ValueError, OverflowError):
            n_excluded_liquidity += 1
            continue
        rows.append({
            "market_id": str(m["id"]),
            "question": m.get("question"),
            "category": m.get("category"),
            "closed_time": ct,
            "outcome": 1 if prices == ["1", "0"] else 0,  # 1 = Yes won
            "volume": vol,
            "liquidity_at_close": liquidity,
            "token_yes": toks[0],
        })
    if rows:
        df = (pd.DataFrame(rows, columns=CLEAN_MARKETS_COLUMNS)
              .sort_values("closed_time").reset_index(drop=True))
    else:
        df = _empty_clean_markets()
    df.attrs["n_excluded_non_numeric"] = n_excluded_volume + n_excluded_liquidity
    df.attrs["n_excluded_non_numeric_volume"] = n_excluded_volume
    df.attrs["n_excluded_non_numeric_liquidity"] = n_excluded_liquidity
    df.attrs["excluded_non_numeric_by_field"] = {
        "volumeNum": n_excluded_volume,
        "liquidityNum": n_excluded_liquidity,
    }
    df.attrs["n_excluded_non_finite"] = n_excluded_nonfinite_volume
    df.attrs["n_excluded_non_finite_volume"] = n_excluded_nonfinite_volume
    df.attrs["excluded_non_finite_by_field"] = {
        "volumeNum": n_excluded_nonfinite_volume,
    }
    df.attrs["n_excluded_bad_volume"] = (
        n_excluded_volume + n_excluded_nonfinite_volume)
    return df


def _validate_snapshot_timing(close_ts, horizon_d):
    """Validate the TYPES of snapshot_price's timing arguments
    (SF-BL-11530 type screen).

    Pure validation — lawful values pass through UNCHANGED (never
    float()/int() converted), so outputs for lawful numeric timing
    arguments are bit-identical to the pre-fix computation,
    including Fraction / numpy-scalar close_ts inputs.

    Raises
    ------
    ValueError
        If ``close_ts`` is a bool or any non-numeric value, or if
        ``horizon_d`` is a bool, a string, or any non-integer
        value. Previously neither argument's type was checked —
        SF-BL-9740 guarded only the horizon's VALUE (negative) —
        so a bool horizon from a JSON/config round-trip silently
        retimed the snapshot: bool is an int subclass, so
        ``horizon_d=True`` read as a 1-day horizon and returned
        the 1-day print (0.60) instead of the intended 7-day
        print (0.50) on the day-3/day-9 fixture; ``close_ts=True``
        read as epoch+1s and returned None exactly as if no print
        existed; and string timing arguments died with bare
        TypeErrors that never named the argument (``horizon_d='7'``
        in the SF-BL-9740 ``horizon_d < 0`` comparison itself,
        ``close_ts='864000'`` in the target subtraction). The
        screen is explicit, following this file's own type-screen
        precedent (SF-BL-11495 / SF-BL-11496): booleans (including
        ``np.bool_``) are rejected outright; ``close_ts`` must be
        an instance of ``numbers.Real`` (int, float, ``Fraction``
        and real numpy scalars qualify) and ``horizon_d`` must be
        an instance of ``numbers.Integral`` (int and numpy
        integers qualify — a float such as 7.0 is a non-integer
        TYPE for a day count and is rejected, as are strings,
        None, complex, sequences/mappings, ndarrays and all other
        non-integral objects) — even where ``float()`` could parse
        them, because a timing argument that arrives as text is
        corrupt state, not a number to be silently parsed. The
        VALUE domains are untouched: a negative integer horizon
        still raises under SF-BL-9740, and every other real
        close_ts value keeps its pre-fix behaviour here.
    """
    if isinstance(close_ts, (bool, np.bool_)) or not isinstance(
            close_ts, numbers.Real):
        raise ValueError(
            "snapshot_price close_ts must be a non-bool real number "
            f"(SF-BL-11530); got close_ts={close_ts!r}")
    if isinstance(horizon_d, (bool, np.bool_)) or not isinstance(
            horizon_d, numbers.Integral):
        raise ValueError(
            "snapshot_price horizon_d must be a non-bool integer "
            f"number of days (SF-BL-11530); got horizon_d={horizon_d!r}")


def _validate_snapshot_print_types(hist):
    """Validate the TYPES of snapshot_price's history print prices
    (SF-BL-11567 type screen).

    Pure validation — lawful print prices pass through UNCHANGED
    (never float()-converted here), so outputs for lawful numeric
    prints are bit-identical to the pre-fix computation.

    Raises
    ------
    ValueError
        If any print in ``hist`` has a ``p`` that is a bool or any
        non-numeric value. Previously the selected print was read
        with a bare ``float(h["p"])`` and NO type check — a
        parseable string print ``p='0.50'`` was silently coerced
        and returned 0.50, bit-identical to the numeric control,
        and a bool print ``p=True`` read as 1.0 and was clamped to
        0.999, a fabricated near-certainty snapshot price
        (``p=False`` likewise fabricated 0.001). CLOB histories are
        deserialised JSON, so a print price that arrives as text or
        truth is corrupt state, not a number to be silently parsed.
        The screen is explicit, following this file's own
        type-screen precedent (SF-BL-11530,
        :func:`_validate_snapshot_timing`): booleans (including
        ``np.bool_``) are rejected outright, and every print's
        ``p`` must be an instance of ``numbers.Real`` (int, float,
        ``Fraction`` and real numpy scalars qualify); strings/bytes,
        None, complex, sequences/mappings, ndarrays and all other
        non-real objects raise the same documented ValueError
        naming ``p`` — even where ``float()`` could parse them.
        Every print in ``hist`` is screened, not only the selected
        one: a mistyped print anywhere in the history is corrupt
        input for this snapshot. Prints missing the ``p`` key are
        NOT screened here — entry-shape failures are SF-BL-11232's
        scope and keep their prior behaviour on the selection
        path. The VALUE domain of a numeric print (non-finite /
        outside [0, 1]) is SF-BL-9716's separate screen, not this
        one: numeric out-of-domain prints keep their
        return-None / dropped-count behaviour.
    """
    if not hist:
        return
    for entry in hist:
        if not isinstance(entry, dict) or "p" not in entry:
            continue
        raw_p = entry["p"]
        if isinstance(raw_p, (bool, np.bool_)) or not isinstance(
                raw_p, numbers.Real):
            raise ValueError(
                "snapshot_price history print p must be a non-bool "
                f"real number (SF-BL-11567); got p={raw_p!r}")


def _snapshot_price_detail(hist: list[dict], close_ts: float, horizon_d: int):
    """(price, dropped_invalid) pair behind snapshot_price (SF-BL-9716).

    dropped_invalid is True exactly when a print WAS selected by the
    freshness window but its value failed the domain screen (non-finite
    or outside [0, 1]); "no candidate in window" and an empty history
    return (None, False) and are not drops.

    The timing arguments' TYPES are screened first (SF-BL-11530;
    see :func:`_validate_snapshot_timing`), before the SF-BL-9740
    negative-horizon check and before the empty-history early
    return — so a mistyped argument raises even on an empty
    history, exactly as a negative horizon already did.

    The history print prices' TYPES are screened next
    (SF-BL-11567; see :func:`_validate_snapshot_print_types`),
    before candidate selection — so a bool or string print raises
    the named ValueError wherever it sits in ``hist``, instead of
    being coerced by ``float()`` when selected.
    """
    _validate_snapshot_timing(close_ts, horizon_d)
    if horizon_d < 0:
        raise ValueError(
            f"horizon_d must be >= 0, got {horizon_d!r}: a negative "
            "horizon would select post-resolution prints")
    _validate_snapshot_print_types(hist)
    if not hist:
        return None, False
    target = close_ts - horizon_d * 86400
    floor = target - 14 * 86400
    cand = [h for h in hist if floor <= h["t"] <= target and h["t"] <= close_ts]
    if not cand:
        return None, False
    h = max(cand, key=lambda x: x["t"])
    p = float(h["p"])
    if not math.isfinite(p) or p < 0.0 or p > 1.0:
        return None, True
    return min(max(p, 0.001), 0.999), False


def snapshot_price(hist: list[dict], close_ts: float, horizon_d: int):
    """Last daily print at/before close-horizon, at most horizon+14d old.

    Raises
    ------
    ValueError
        If ``close_ts`` is a bool or any non-numeric value, or if
        ``horizon_d`` is a bool, a string or any non-integer value
        (SF-BL-11530; see :func:`_validate_snapshot_timing`).
        Previously ``horizon_d=True`` silently selected the 1-day
        print instead of the 7-day print, ``close_ts=True``
        silently returned None, and string timing arguments died
        with bare TypeErrors. Numeric-timing outputs are
        bit-identical to the pre-fix computation.
    ValueError
        If any history print's ``p`` is a bool or any non-numeric
        value (SF-BL-11567; see
        :func:`_validate_snapshot_print_types`). Previously the
        selected print was coerced with a bare ``float()``:
        ``p='0.50'`` returned 0.50 bit-identical to the numeric
        control, and ``p=True`` was clamped to a fabricated 0.999.
        Numeric-print outputs are bit-identical to the pre-fix
        computation.

    horizon_d must be >= 0 (SF-BL-9740): a negative horizon moves the
    target past close_ts and would select post-resolution prints —
    resolution leakage into a pre-resolution forecast snapshot.
    The candidate filter also hard-caps prints at close_ts so no
    post-close print can ever be selected.

    The selected print must be a finite probability in [0, 1]
    (SF-BL-9716): a non-finite print (NaN, +/-inf) or a print outside
    [0, 1] (e.g. a percent-unit 82.0) returns None — the snapshot row
    is dropped and counted by build_snapshots — instead of being
    clamped into a plausible in-range price that would pass the
    downstream SF-BL-9359 INVALID_PRICE guard unseen. There is no
    fallback to an older print. In-range prints keep the pre-fix
    [0.001, 0.999] epsilon clamp exactly, so every lawful snapshot
    (including genuine 0.001/0.999 prints) is unchanged.
    """
    price, _dropped_invalid = _snapshot_price_detail(hist, close_ts, horizon_d)
    return price


def build_snapshots(markets: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Build the (market, horizon) snapshot panel.

    SF-BL-9716: a (market, horizon) whose selected CLOB print is
    non-finite or outside [0, 1] is dropped (no row emitted, no
    clamped substitute) and counted on the returned frame as
    ``df.attrs["n_dropped_invalid_price"]`` — present (0) on a
    fully clean build. Horizons with no print in their freshness
    window are skipped as before and are NOT counted as drops.
    """
    tokens = markets["token_yes"].tolist()
    with ThreadPoolExecutor(max_workers=8) as ex:
        hists = list(ex.map(lambda t: fetch_clob(t, refresh), tokens))
    rows = []
    n_dropped_invalid = 0
    for (_, m), hist in zip(markets.iterrows(), hists):
        close_ts = m["closed_time"].timestamp()
        for h_days in HORIZONS_DAYS:
            p, dropped_invalid = _snapshot_price_detail(hist, close_ts, h_days)
            if dropped_invalid:
                n_dropped_invalid += 1
            if p is None:
                continue
            rows.append({
                "market_id": m["market_id"],
                "question": m["question"],
                "closed_time": m["closed_time"],
                "outcome": int(m["outcome"]),
                "price": p,
                "days_to_expiry": float(h_days),
                "volume": float(m["volume"]),
                "log10_volume": math.log10(max(m["volume"], 1.0)),
                "liquidity_at_close": float(m["liquidity_at_close"]),
            })
    df = pd.DataFrame(rows)
    df.attrs["n_dropped_invalid_price"] = n_dropped_invalid
    return df


# --------------------------------------------------------------- rules ----
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _validated_price_array(p):
    """Return ``p`` as a float array, screening its TYPE first
    (SF-BL-11498).

    Raises
    ------
    ValueError
        If ``p`` is not a numeric (int/float) panel. The type screen
        runs BEFORE any float coercion, on ``np.asarray(p)`` with no
        dtype: previously every rule entered through
        ``np.asarray(p, dtype=float)``, which laundered a string
        panel ``['0.9', '0.1']`` into the numeric panel
        ``[0.9, 0.1]`` (outputs bit-identical to the numeric call in
        all six rules) and a bool panel ``[True, False]`` into
        ``[1.0, 0.0]``, so a CSV-round-tripped or otherwise mistyped
        snapshot price panel was indistinguishable from lawful
        floats in every adjusted forecast the rules emit. Neither
        ``apply_rule`` nor ``fit_rules`` validates the dtype of the
        ``price`` column they forward (``.to_numpy()`` preserves an
        object/string or bool column's dtype), and the rules are a
        public surface, so the screen lives here — the same
        ``dtype.kind not in "iuf"`` accept rule as the SF-BL-11499
        ``days`` screen in :func:`_scale_time`. Validation is pure —
        a lawful numeric panel is returned via exactly the pre-fix
        ``np.asarray(p, dtype=float)`` conversion — so numeric-panel
        outputs are bit-identical for every rule. The price VALUE
        domain (1.5 / NaN) is SF-BL-11417's separate screen, not
        this one.
    """
    arr = np.asarray(p)
    kind = arr.dtype.kind
    if kind in "iuf":
        return np.asarray(p, dtype=float)
    if kind == "b":
        raise ValueError(
            "p must be a numeric (int/float) price panel; got a bool "
            f"price panel (dtype {arr.dtype}): booleans are not "
            "prices (True/False would coerce to 1.0/0.0)")
    if kind in "US":
        raise ValueError(
            "p must be a numeric (int/float) price panel; got a "
            f"string price panel (dtype {arr.dtype}): string prices "
            "(e.g. from a CSV round-trip) are not validated floats — "
            "parse and validate them before applying a rule")
    raise ValueError(
        "p must be a numeric (int/float) price panel; got prices "
        f"with dtype {arr.dtype} (kind {kind!r}) — string, bool and "
        "other non-numeric price panels are not accepted")


def rule_raw(p, days, zlogvol, params):
    return _validated_price_array(p)


def _validate_shrink_fixed_s(s) -> float:
    """Validate ``rule_shrink_fixed``'s shrink factor ``s``
    (SF-BL-11495 type screen; SF-BL-11375 value domain).

    Returns ``s`` as a float. Pure validation — a valid ``s`` is
    returned unchanged in value, so outputs for lawful numeric
    ``s`` are bit-identical to the pre-fix computation.

    Raises
    ------
    ValueError
        Unless ``s`` is a non-bool real number. Previously ``s``
        was multiplied raw against the price panel:
        ``s='0.5'`` died with a bare ``UFuncTypeError`` that never
        named ``s``, ``s=None`` with a bare ``TypeError``, and
        ``s=True`` silently applied NO shrinkage at all — bool is
        an int subclass, so True read as the fitted factor 1.0 and
        ``[0.9, 0.1]`` passed through unchanged as the adjusted
        forecast (``s=False`` likewise fabricated ``[0.5, 0.5]``);
        a one-element list/tuple broadcast silently, and a complex
        ``s`` returned complex "probabilities". ``params`` are
        deserialised from results.json, so a serialised fitted
        factor reaches scoring with exactly these failure modes.
        The screen is explicit, following this file's own
        type-screen precedent (SF-BL-11497, ``apply_rule``):
        booleans (including ``np.bool_``) are rejected outright,
        and ``s`` must be an instance of
        ``numbers.Real`` (int, float, ``Fraction`` and real numpy
        scalars all qualify); strings/bytes, None,
        complex, sequences/mappings, ndarrays of any ndim and all
        other non-real objects raise the same documented
        ValueError naming ``s`` — even where ``float()`` could
        parse them, because a fitted factor that arrives as text
        is corrupt state, not a number to be silently parsed.
        The VALUE domain of ``s`` is validated below as well
        (SF-BL-11375): ``s`` must be finite and lie in [0, 1],
        the probability-preserving domain — for ``p`` in [0, 1],
        ``0.5 + s * (p - 0.5)`` stays in [0, 1] for every ``p``
        iff ``0 <= s <= 1``; ``s > 1`` expands away from 0.5 and
        can leave [0, 1] (s=2.0 mapped [0.9, 0.1] to
        [1.3, -0.3]), ``s < 0`` inverts the forecast (s=-1.0
        mapped [0.9, 0.1] to [0.1, 0.9]) and a non-finite ``s``
        (NaN, +/-inf) poisoned every output to NaN/inf. The
        fitted grid in :func:`fit_rules` (originally
        0.70-1.15) is therefore clipped to this domain
        (effective grid 0.70-1.00), and a deserialised fitted
        ``s`` outside [0, 1] — e.g. a corrupt or hand-edited
        results.json — now raises the named ValueError below
        instead of reaching Brier/ECE scoring unchecked.
    """
    msg = ("rule_shrink_fixed s must be a non-bool real number "
           f"(SF-BL-11495); got s={s!r}")
    if isinstance(s, (bool, np.bool_)):
        raise ValueError(msg)
    if not isinstance(s, numbers.Real):
        raise ValueError(msg)
    try:
        s_f = float(s)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(msg) from None
    if not math.isfinite(s_f):
        raise ValueError(
            "rule_shrink_fixed s must be finite (SF-BL-11375); "
            f"got s={s!r}")
    if not 0.0 <= s_f <= 1.0:
        raise ValueError(
            "rule_shrink_fixed s must lie in [0, 1] (SF-BL-11375); "
            f"got s={s!r}: an s outside [0, 1] is not a shrink — "
            "it maps probabilities outside [0, 1] or inverts the "
            "forecast")
    return s_f


def rule_shrink_fixed(p, days, zlogvol, params):
    """Shrink toward 0.5 by the fitted factor ``params["s"]``:
    ``p' = 0.5 + s * (p - 0.5)``.

    Raises
    ------
    ValueError
        If ``s`` is a bool or any non-numeric value (SF-BL-11495;
        see :func:`_validate_shrink_fixed_s`). Previously
        ``s='0.5'`` / ``s=None`` crashed with bare TypeErrors in
        the shrink multiplication and ``s=True`` silently returned
        the raw prices unchanged, as if s=1.0 had been fitted.
        Numeric-``s`` outputs are bit-identical to the pre-fix
        computation ([0.7, 0.3] at s=0.5 on [0.9, 0.1]).
    ValueError
        If ``s`` is non-finite (NaN, +/-inf) or lies outside
        [0, 1] (SF-BL-11375; see :func:`_validate_shrink_fixed_s`).
        Previously ``s=2.0`` returned the impossible
        probabilities [1.3, -0.3] on [0.9, 0.1], ``s=-1.0``
        silently inverted the forecast to [0.1, 0.9], and
        ``s=NaN`` returned [nan, nan] — and ``apply_rule``
        forwards ``params`` deserialised from results.json, so a
        corrupt fitted ``s`` reached Brier/ECE scoring unchecked.
    """
    s = _validate_shrink_fixed_s(params["s"])
    return 0.5 + s * (_validated_price_array(p) - 0.5)


def _validate_logit_shrink_b(b) -> float:
    """Validate ``rule_logit_shrink``'s fitted scale ``b``
    (SF-BL-11532 type screen; SF-BL-11416 value domain).

    Returns ``b`` as a float. Pure validation — a valid ``b`` is
    returned unchanged in value, so outputs for lawful numeric
    ``b`` (float, int, real numpy scalars) are bit-identical to
    the pre-fix computation.

    Raises
    ------
    ValueError
        Unless ``b`` is a non-bool real number. Previously ``b``
        was multiplied raw against the logit panel:
        ``b='0.5'`` died with a bare ``UFuncTypeError`` that never
        named ``b``, ``b=None`` with a bare ``TypeError``, and
        ``b=True`` silently applied NO shrinkage at all — bool is
        an int subclass, so True read as the fitted scale 1.0 and
        ``[0.9, 0.1]`` passed through unchanged as the adjusted
        forecast, against ``[0.75, 0.25]`` at the fitted
        ``b=0.5`` (``b=False`` likewise fabricated ``[0.5, 0.5]``);
        a one-element list/tuple broadcast silently, a 0-d or
        one-element ndarray did the same, and a complex ``b``
        returned complex "probabilities". ``params`` are
        deserialised from results.json, so a serialised fitted
        scale reaches scoring with exactly these failure modes.
        This was the last unscreened fitted parameter in the
        file: ``s`` was fixtured in SF-BL-11495 and ``b0``/``b1``/
        ``b2`` in SF-BL-11496. The screen is explicit, following
        the SF-BL-11495 precedent (``rule_shrink_fixed``'s ``s``,
        the other single scalar multiplied directly against the
        panel): booleans (including ``np.bool_``) are rejected
        outright, and ``b`` must be an instance of
        ``numbers.Real`` (int, float, ``Fraction`` and real numpy
        scalars all qualify); strings/bytes, None, complex,
        sequences/mappings, ndarrays of any ndim and all other
        non-real objects raise the same documented ValueError
        naming ``b`` — even where ``float()`` could parse them,
        because a fitted scale that arrives as text is corrupt
        state, not a number to be silently parsed. A ``Fraction``
        ``b`` is accepted via ``float()``: pre-fix it crashed
        inside ``_sigmoid`` (the object-dtype product has no
        ``np.exp`` loop), so no lawful output is altered. The
        VALUE domain of ``b`` is validated below as well
        (SF-BL-11416): ``b`` must be finite and >= 0 — the
        order-preserving domain of the logit scale, since
        ``logit(p') = b * logit(p)`` keeps the forecast's ordering
        (and stays inside (0, 1) for p in (0, 1)) iff ``b >= 0``,
        ``b=0`` is the lawful full shrink to 0.5, and the fitted
        grid in :func:`fit_rules` (0.60-1.30) lies inside the
        domain. Previously ``b=-1.0`` silently inverted
        ``[0.9, 0.1]`` to ``[0.1, 0.9]``, ``b=inf`` saturated it
        to the certain forecast ``[1.0, 0.0]``, ``b=-inf``
        inverted and saturated it to ``[0.0, 1.0]``, and
        ``b=NaN`` returned ``[nan, nan]`` — a corrupt fitted
        slope manufactured an inverted, certain or NaN forecast
        that Brier/ECE scoring then consumed unchecked.
    """
    msg = ("rule_logit_shrink b must be a non-bool real number "
           f"(SF-BL-11532); got b={b!r}")
    if isinstance(b, (bool, np.bool_)):
        raise ValueError(msg)
    if not isinstance(b, numbers.Real):
        raise ValueError(msg)
    try:
        b_f = float(b)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(msg) from None
    if not math.isfinite(b_f):
        raise ValueError(
            "rule_logit_shrink b must be finite (SF-BL-11416); "
            f"got b={b!r}")
    if b_f < 0.0:
        raise ValueError(
            "rule_logit_shrink b must be >= 0 (SF-BL-11416); "
            f"got b={b!r}: a negative b is not a shrink — it "
            "inverts the forecast")
    return b_f


def rule_logit_shrink(p, days, zlogvol, params):
    """Logit-shrink rule: ``logit(p') = b * logit(p)``
    (temperature / Platt scaling by the fitted scale
    ``params["b"]``).

    Raises
    ------
    ValueError
        If ``b`` is a bool or any non-numeric value (SF-BL-11532;
        see :func:`_validate_logit_shrink_b`). Previously
        ``b='0.5'`` / ``b=None`` crashed with bare
        UFuncTypeError/TypeError in the scale multiplication and
        ``b=True`` silently returned the raw prices unchanged,
        as if b=1.0 had been fitted. Numeric-``b`` outputs are
        bit-identical to the pre-fix computation ([0.75, 0.25]
        at b=0.5 on [0.9, 0.1]).
    ValueError
        If ``b`` is non-finite (NaN, +/-inf) or negative
        (SF-BL-11416; see :func:`_validate_logit_shrink_b`).
        Previously ``b=-1.0`` silently inverted [0.9, 0.1] to
        [0.1, 0.9], ``b=inf`` saturated it to [1.0, 0.0] and
        ``b=NaN`` returned [nan, nan].
    """
    b = _validate_logit_shrink_b(params["b"])
    return _sigmoid(b * _logit(_validated_price_array(p)))


def _validate_logit_scale_param(name, value):
    """Validate a fitted logit-scale parameter (``b0``/``b1``/``b2``)
    for the logit_time rules (SF-BL-11496 type screen).

    Returns the parameter UNCHANGED. Pure validation — the value
    is never transformed, so outputs for lawful numeric parameters
    (float, int, real numpy scalars, ``Fraction``) are bit-identical
    to the pre-fix computation, including the object-dtype
    arithmetic a ``Fraction`` parameter produced pre-fix.

    Raises
    ------
    ValueError
        Unless the parameter is a non-bool real number. Previously
        the parameters were read raw from the params dict:
        ``b0='1'`` and ``b1='0.1'`` died inside ``_scale_time`` with
        bare UFuncTypeErrors that never named the parameter,
        ``b2='0.5'`` died the same way in ``rule_logit_time_vol``'s
        volume product, and a bool silently ran as the fitted value
        1.0 — bool is an int subclass, so ``b0=True`` applied an
        intercept of 1.0 and ``b2=True`` added the FULL volume
        z-score, moving ``[0.9, 0.1]`` to ``[0.9643, 0.25]`` on a
        ``z=[0.5, -0.5]`` panel. Neither ``apply_rule`` nor
        ``fit_rules`` validates params it forwards or produces for
        other callers, and fitted params are serialised into
        results.json, so a deserialised parameter reaches scoring
        with exactly these failure modes. The screen is explicit,
        following this file's own type-screen precedent
        (SF-BL-11495, ``rule_shrink_fixed``'s ``s``): booleans
        (including ``np.bool_``) are rejected outright, and the
        parameter must be an instance of ``numbers.Real`` (int,
        float, ``Fraction`` and real numpy scalars all qualify);
        strings/bytes, None, complex, sequences/mappings, ndarrays
        of any ndim and all other non-real objects raise the same
        documented ValueError naming the parameter — even where
        ``float()`` could parse them, because a fitted parameter
        that arrives as text is corrupt state, not a number to be
        silently parsed. The VALUE domain of the parameters
        (finite / range — NaN, inf, negatives) is not this screen:
        out-of-domain real numbers keep their pre-fix behaviour
        here.
    """
    msg = (f"logit scale parameter {name} must be a non-bool real "
           f"number (SF-BL-11496); got {name}={value!r}")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(msg)
    if not isinstance(value, numbers.Real):
        raise ValueError(msg)
    return value


def _scale_time(params, days):
    """Time scale for the logit_time rules: ``b0 + b1 * log1p(days)``.

    Raises
    ------
    ValueError
        If ``days`` is not a numeric (int/float) panel (SF-BL-11499).
        The type screen runs BEFORE any float coercion: previously
        ``np.asarray(days, dtype=float)`` laundered a string panel
        ``['10', '20']`` into the numeric panel ``[10, 20]`` (outputs
        bit-identical to the numeric call) and a bool panel
        ``[True, False]`` into 1-day / 0-day horizons, so a mistyped
        horizon silently repriced the shrinkage. Validation is pure —
        it never transforms a lawful numeric panel — so numeric-days
        outputs are bit-identical to the pre-fix computation. The
        days VALUE domain (negative / NaN / inf) is SF-BL-11374's
        separate screen, not this one.
    ValueError
        If ``b0`` or ``b1`` is a bool or any non-numeric value
        (SF-BL-11496; see :func:`_validate_logit_scale_param`).
        Previously ``b0='1'`` / ``b1='0.1'`` crashed with bare
        UFuncTypeErrors in the scale arithmetic and ``b0=True``
        silently ran an intercept of 1.0. Numeric-parameter outputs
        are bit-identical to the pre-fix computation.
    """
    arr = np.asarray(days)
    if arr.dtype.kind not in "iuf":
        raise ValueError(
            "days must be a numeric (int/float) panel of day counts; "
            f"got days with dtype {arr.dtype} — string, bool and other "
            "non-numeric horizons are not valid day counts")
    b0 = _validate_logit_scale_param("b0", params["b0"])
    b1 = _validate_logit_scale_param("b1", params["b1"])
    return b0 + b1 * np.log1p(arr.astype(float))


def rule_logit_time(p, days, zlogvol, params):
    return _sigmoid(_scale_time(params, days) * _logit(_validated_price_array(p)))


def _validated_zlogvol_array(zlogvol):
    """Return ``zlogvol`` as a float array, screening its TYPE first
    (SF-BL-11500).

    Raises
    ------
    ValueError
        If ``zlogvol`` is not a numeric (int/float) panel. The type
        screen runs BEFORE any float coercion, on
        ``np.asarray(zlogvol)`` with no dtype: previously
        ``rule_logit_time_vol`` entered through
        ``np.asarray(zlogvol, dtype=float)``, which laundered a
        string panel ``['0.5', '-0.5']`` into the numeric panel
        ``[0.5, -0.5]`` (output bit-identical to the numeric call,
        ``[0.9397, 0.1614]`` at the b0=1.0/b1=0.0/b2=0.5 fixture on
        ``[0.9, 0.1]``) and a bool panel ``[True, False]`` into
        ``[1.0, 0.0]`` (returning ``[0.9643, 0.1]``), so a
        serialised or otherwise mistyped volume z-score panel —
        the fleet's liquidity proxy feeding the time/vol
        adjustment — was indistinguishable from computed floats.
        ``None`` likewise became an all-NaN panel and a complex
        panel died with a bare TypeError. Neither ``apply_rule``
        nor ``fit_rules`` validates the dtype of the z panel a
        direct caller supplies (they compute their own float z
        internally, but the rule is a public surface), so the
        screen lives here — the same ``dtype.kind not in "iuf"``
        accept rule as the SF-BL-11498 price screen in
        :func:`_validated_price_array` and the SF-BL-11499 ``days``
        screen in :func:`_scale_time`. Validation is pure — a
        lawful numeric panel is returned via exactly the pre-fix
        ``np.asarray(zlogvol, dtype=float)`` conversion — so
        numeric-panel outputs are bit-identical. The zlogvol VALUE
        domain (inf / -inf / NaN) is SF-BL-11376's separate screen,
        not this one.
    """
    arr = np.asarray(zlogvol)
    kind = arr.dtype.kind
    if kind in "iuf":
        return np.asarray(zlogvol, dtype=float)
    if kind == "b":
        raise ValueError(
            "zlogvol must be a numeric (int/float) panel of volume "
            f"z-scores (SF-BL-11500); got a bool zlogvol panel "
            f"(dtype {arr.dtype}): booleans are not z-scores "
            "(True/False would coerce to 1.0/0.0)")
    if kind in "US":
        raise ValueError(
            "zlogvol must be a numeric (int/float) panel of volume "
            f"z-scores (SF-BL-11500); got a string zlogvol panel "
            f"(dtype {arr.dtype}): string z-scores (e.g. serialised "
            "z-scores) are not validated floats — parse and "
            "validate them before applying a rule")
    raise ValueError(
        "zlogvol must be a numeric (int/float) panel of volume "
        f"z-scores (SF-BL-11500); got zlogvol with dtype "
        f"{arr.dtype} (kind {kind!r}) — string, bool and other "
        "non-numeric zlogvol panels are not accepted")


def rule_logit_time_vol(p, days, zlogvol, params):
    """Logit-time rule with a volume term:
    ``logit(p') = (b0 + b1*log1p(days) + b2*z_logvol) * logit(p)``.

    Raises
    ------
    ValueError
        If ``b2`` is a bool or any non-numeric value (SF-BL-11496;
        see :func:`_validate_logit_scale_param`), or if ``b0`` /
        ``b1`` / ``days`` fail their screens in :func:`_scale_time`.
        Previously ``b2='0.5'`` crashed with a bare UFuncTypeError
        in the volume product and ``b2=True`` silently added the
        full volume z-score (a weight of 1.0), moving
        ``[0.9, 0.1]`` to ``[0.9643, 0.25]`` on a ``z=[0.5, -0.5]``
        panel at b0=1.0/b1=0.0. Numeric-parameter outputs are
        bit-identical to the pre-fix computation.
    ValueError
        If ``zlogvol`` is not a numeric (int/float) panel
        (SF-BL-11500; see :func:`_validated_zlogvol_array`).
        Previously the z panel was coerced raw through
        ``np.asarray(zlogvol, dtype=float)``: a string panel
        ``['0.5', '-0.5']`` returned ``[0.9397, 0.1614]``
        bit-identical to the numeric panel at b2=0.5, and a bool
        panel ``[True, False]`` silently entered as ``z=[1, 0]``,
        returning ``[0.9643, 0.1]``. Numeric-z outputs are
        bit-identical to the pre-fix computation.
    """
    b2 = _validate_logit_scale_param("b2", params["b2"])
    scale = _scale_time(params, days) + b2 * _validated_zlogvol_array(zlogvol)
    return _sigmoid(scale * _logit(_validated_price_array(p)))


def _validate_bucket_recal_param(name, raw):
    """Return a fitted ``bucket_recal`` parameter (``edges`` or
    ``values``) as an array, screening its TYPE first (SF-BL-11531).

    Returns exactly the pre-fix ``np.array(raw)`` for a lawful
    numeric parameter. Pure validation — the array is never cast or
    transformed — so genuine numeric fits (float/int lists, tuples,
    ndarrays, Series) are bit-identical to the pre-fix computation,
    dtype included.

    Raises
    ------
    ValueError
        If any edge or value is a bool or a string, or the parameter
        is otherwise non-numeric. Previously the rule built both
        parameters with bare ``np.array()`` and no dtype: a
        JSON-round-tripped fit with ``values=['0.2', '0.8']`` was
        returned verbatim as the recalibrated "probabilities"
        ``['0.8', '0.2']`` (string dtype), ``values=[False, True]``
        was returned as ``[True, False]``, ``edges=['0.0', '0.5']``
        bucketed bit-identically to the numeric control
        ``[0.8, 0.2]`` on ``p=[0.9, 0.1]``, and bool edges
        ``[False, True]`` silently bucketed as ``[0.0, 1.0]`` —
        downstream scoring (``calibration_table``'s
        ``adj[m].mean()``, ``brier``, ``ece``) then crashed on or
        coerced those non-probabilities. The screen runs BEFORE any
        bucketing, on ``np.array(raw)`` with no dtype argument — the
        same ``dtype.kind not in "iuf"`` accept rule as the
        SF-BL-11498 price screen in :func:`_validated_price_array` —
        plus an element scan for list/tuple inputs, because NumPy
        coercion hides an offending element inside a mixed list:
        ``values=[0.2, True]`` becomes a float64 panel and pre-fix
        silently ran as the fit ``[0.2, 1.0]``, returning
        ``[1.0, 0.2]``. Booleans (including ``np.bool_``) and
        strings/bytes therefore raise the named ValueError even
        where coercion or ``float()`` could parse them, because a
        fitted parameter that arrives as text or truth is corrupt
        state, not a number to be silently parsed. The VALUE domain
        of the fit (values in [0, 1], finite, monotonic edges) is
        SF-BL-11418's separate screen, enforced by
        :func:`_validate_bucket_recal_domain` in
        :func:`rule_bucket_recal` after this type screen runs:
        out-of-domain real numbers are rejected there.
    """
    if isinstance(raw, (list, tuple)):
        for el in raw:
            if isinstance(el, (bool, np.bool_)):
                raise ValueError(
                    f"bucket_recal {name} must be a numeric "
                    f"(int/float) fitted array; got a bool element "
                    f"{el!r} in {name}={raw!r}: booleans are not "
                    "fitted probabilities or edges (True/False would "
                    "coerce to 1.0/0.0) (SF-BL-11531)")
            if isinstance(el, (str, bytes)):
                raise ValueError(
                    f"bucket_recal {name} must be a numeric "
                    f"(int/float) fitted array; got a string element "
                    f"{el!r} in {name}={raw!r}: string fitted "
                    "parameters (e.g. from a JSON round-trip) are "
                    "not validated numbers — parse and validate "
                    "them before applying the rule (SF-BL-11531)")
    arr = np.array(raw)
    kind = arr.dtype.kind
    if kind in "iuf":
        return arr
    if kind == "b":
        raise ValueError(
            f"bucket_recal {name} must be a numeric (int/float) "
            f"fitted array; got a bool {name} (dtype {arr.dtype}): "
            "booleans are not fitted probabilities or edges "
            "(True/False would coerce to 1.0/0.0) (SF-BL-11531)")
    if kind in "US":
        raise ValueError(
            f"bucket_recal {name} must be a numeric (int/float) "
            f"fitted array; got a string {name} (dtype "
            f"{arr.dtype}): string fitted parameters (e.g. from a "
            "JSON round-trip) are not validated numbers — parse "
            "and validate them before applying the rule "
            "(SF-BL-11531)")
    raise ValueError(
        f"bucket_recal {name} must be a numeric (int/float) fitted "
        f"array; got {name} with dtype {arr.dtype} (kind "
        f"{kind!r}) — string, bool and other non-numeric fitted "
        "parameters are not accepted (SF-BL-11531)")


def _validate_bucket_recal_domain(edges, vals):
    """Validate the fitted VALUE domain of ``bucket_recal``
    (SF-BL-11418): ``values`` in [0, 1] and finite, ``edges``
    finite and monotonic (non-decreasing).

    Pure validation — the arrays handed in (the exact arrays the
    SF-BL-11531 type screen returned) are never cast, sorted or
    clipped by this screen, and the caller applies the rule with
    them unchanged, so every lawful fit is bit-identical to the
    pre-fix computation, dtype included.

    Raises
    ------
    ValueError
        If ``values`` is empty, holds a non-finite entry, or holds
        any entry outside [0, 1]; or if ``edges`` holds a
        non-finite entry or any strict decrease. Previously the
        rule validated only the parameters' TYPE (SF-BL-11531) and
        applied the fit unchecked: ``values=[0.2, 1.5, -0.3]`` at
        ``edges=[0.0, 0.5]`` returned the impossible probability
        ``[1.5, 0.2]`` on ``p=[0.9, 0.1]`` (and ``[-0.3, 0.2]``
        at ``edges=[0.0, 0.5, 0.75]``), NaN/inf values were
        emitted verbatim, NaN edges silently bucketed every price
        into bucket 0, unsorted ``edges=[0.8, 0.2]`` returned
        ``[0.8, 0.2]`` via ``np.searchsorted`` on an array whose
        sortedness it assumes, and empty ``values`` died with a
        bare ``IndexError``. The whole fit is validated BEFORE any
        bucketing — not only the entries a given panel would
        select — because a corrupt fit must not be applied at all:
        an unselected 1.5 is still an impossible fitted
        probability waiting for the panel that selects it. Edges
        are NOT range-checked against [0, 1]: the prereg names
        only monotonicity for edges, and sentinel edges outside
        [0, 1] bucket lawfully. Equal adjacent edges stay lawful
        (non-decreasing): a degenerate quantile fit in
        ``fit_rules`` can produce them, and ``searchsorted``'s
        sortedness assumption is broken only by a decrease.
    """
    if vals.size == 0:
        raise ValueError(
            "bucket_recal values must be a non-empty fitted array; "
            f"got values={vals!r}: a fit with no values cannot "
            "emit a lawful probability (SF-BL-11418)")
    vf = np.asarray(vals, dtype=float)
    nonfinite = vf[~np.isfinite(vf)]
    if nonfinite.size:
        raise ValueError(
            "bucket_recal values must be finite fitted "
            f"probabilities in [0, 1]; got non-finite value(s) "
            f"{nonfinite.tolist()!r} in values={vals!r}: a NaN or "
            "infinite fitted value is not a probability "
            "(SF-BL-11418)")
    outside = vf[(vf < 0.0) | (vf > 1.0)]
    if outside.size:
        raise ValueError(
            "bucket_recal values must be fitted probabilities in "
            f"[0, 1]; got value(s) {outside.tolist()!r} outside "
            f"[0, 1] in values={vals!r}: the bucket rule would "
            "emit an impossible probability from a corrupt fit "
            "(SF-BL-11418)")
    ef = np.asarray(edges, dtype=float)
    nonfinite_e = ef[~np.isfinite(ef)]
    if nonfinite_e.size:
        raise ValueError(
            "bucket_recal edges must be finite fitted bucket "
            f"edges; got non-finite edge(s) "
            f"{nonfinite_e.tolist()!r} in edges={edges!r}: a NaN "
            "or infinite edge silently mis-buckets every price "
            "(SF-BL-11418)")
    if ef.size >= 2:
        drops = np.flatnonzero(np.diff(ef) < 0)
        if drops.size:
            i = int(drops[0])
            raise ValueError(
                "bucket_recal edges must be monotonic "
                "(non-decreasing) fitted bucket edges; got "
                f"edges[{i}]={ef.flat[i]!r} > "
                f"edges[{i + 1}]={ef.flat[i + 1]!r} in "
                f"edges={edges!r}: np.searchsorted assumes sorted "
                "edges, so unsorted edges silently mis-bucket "
                "(SF-BL-11418)")


def rule_bucket_recal(p, days, zlogvol, params):
    """Bucket recalibration: return the fitted ``values`` entry for
    the bucket ``p`` falls into, per the fitted ``edges``.

    Raises
    ------
    ValueError
        If any fitted edge or value is a bool or a string, or
        otherwise non-numeric (SF-BL-11531; see
        :func:`_validate_bucket_recal_param`). Previously string
        values were returned verbatim as the recalibrated
        "probabilities" and string edges bucketed bit-identically
        to the numeric control. Genuine numeric fits are
        bit-identical to the pre-fix computation ([0.8, 0.2] at
        edges=[0.0, 0.5] / values=[0.2, 0.8] on [0.9, 0.1]).
    ValueError
        If the fitted domain is corrupt (SF-BL-11418; see
        :func:`_validate_bucket_recal_domain`): any ``values``
        entry outside [0, 1] or non-finite, an empty ``values``,
        or ``edges`` non-finite or non-monotonic. Previously
        ``values=[0.2, 1.5, -0.3]`` was applied unchecked and
        emitted impossible probabilities, and unsorted
        ``edges=[0.8, 0.2]`` silently mis-bucketed via
        ``np.searchsorted``. Lawful fits are bit-identical to
        the pre-fix computation.
    """
    p = _validated_price_array(p)
    edges = _validate_bucket_recal_param("edges", params["edges"])
    vals = _validate_bucket_recal_param("values", params["values"])
    _validate_bucket_recal_domain(edges, vals)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(vals) - 1)
    return vals[idx]


RULE_FUNCS = {
    "raw": rule_raw,
    "shrink_fixed": rule_shrink_fixed,
    "logit_shrink": rule_logit_shrink,
    "logit_time": rule_logit_time,
    "logit_time_vol": rule_logit_time_vol,
    "bucket_recal": rule_bucket_recal,
}


def apply_rule(name, df, params, z_mean, z_sd):
    """Apply rule ``name`` to a snapshot frame.

    Raises
    ------
    ValueError
        If ``z_sd`` is non-finite or zero, or if ``z_mean`` is
        non-finite (SF-BL-9745). Previously the caller-supplied
        scalars were divided by unchecked:
        ``apply_rule('logit_time_vol', df, params, z_mean=5.0,
        z_sd=0.0)`` emitted divide-by-zero/invalid warnings and
        returned [NaN, 1.0] adjusted probabilities (0/0 -> NaN on
        one row, x/0 -> +inf -> saturated 1.0 on the next), and a
        non-finite z_mean/z_sd poisoned every z the same way.
        ``fit_rules`` guards its own std with ``or 1.0``, but that
        rescues only an exactly-zero std (NaN is truthy, so a NaN
        std from a 1-row or NaN-bearing fit passes through, as does
        an inf mean) and ``apply_rule`` is the public consumption
        surface — callers compute or load (z_mean, z_sd) themselves
        — so the scalars are validated here. Validation is pure:
        for every finite ``z_mean`` and finite non-zero ``z_sd``
        (in particular the pair ``fit_rules`` produces on a lawful
        panel) the output is byte-identical to the pre-fix
        computation.
    ValueError
        If ``z_mean`` or ``z_sd`` is a bool or a non-numeric type
        (SF-BL-11497). The SF-BL-9745 screen above is value-only:
        ``math.isfinite('1.0')`` raises a bare ``TypeError`` (not
        the named ``ValueError``), and a bool passes every range
        check because ``bool`` is an ``int`` subclass —
        ``apply_rule('logit_time_vol', df, params, z_mean=True,
        z_sd=1.0)`` on the TASK fixture (b0=1.0/b1=0.0/b2=0.5,
        price [0.9, 0.1], log10_volume [5, 6]) silently centred
        the volume z-scores at 1 instead of 5 and returned
        [0.9986, 0.0005] against the [0.9, 0.0357] control, and
        ``z_sd=True`` silently divided by 1. Both scalars are
        therefore type-checked BEFORE the range checks: a bool,
        or any value that is not a real number (``numbers.Real``
        — int, float, Fraction and numpy real scalars all
        qualify), raises ``ValueError`` naming the
        argument. The screen is pure validation, so lawful
        scalars — in particular the float pair ``fit_rules``
        produces — are bit-identical to the pre-fix computation.
    """
    if isinstance(z_mean, bool) or not isinstance(z_mean, numbers.Real):
        raise ValueError(
            f"z_mean must be a real number, got {z_mean!r} "
            f"({type(z_mean).__name__}): a bool or non-numeric "
            "centre would silently mis-centre or crash the "
            "volume z-scores")
    if isinstance(z_sd, bool) or not isinstance(z_sd, numbers.Real):
        raise ValueError(
            f"z_sd must be a real number, got {z_sd!r} "
            f"({type(z_sd).__name__}): a bool or non-numeric "
            "scale would silently rescale or crash the "
            "volume z-scores")
    if not math.isfinite(z_sd) or z_sd == 0:
        raise ValueError(
            f"z_sd must be finite and non-zero, got {z_sd!r}: "
            "dividing (log10_volume - z_mean) by it would produce "
            "NaN/inf volume z-scores")
    if not math.isfinite(z_mean):
        raise ValueError(
            f"z_mean must be finite, got {z_mean!r}: "
            "a non-finite centre would produce NaN/inf volume "
            "z-scores")
    z = (df["log10_volume"].to_numpy() - z_mean) / z_sd
    return RULE_FUNCS[name](df["price"].to_numpy(), df["days_to_expiry"].to_numpy(), z, params)


# ------------------------------------------------------------- scoring ----
def _validated_scoring_panel(values, name):
    """Return ``values`` as a float array, screening its TYPE first
    (SF-BL-11354).

    ``name`` is the panel name (``"p"`` or ``"y"``) used in every
    error message, so a corrupt panel is always named.

    Raises
    ------
    ValueError
        If any element of the panel is a bool or a string, if the
        panel itself is a bool/string scalar, or if the panel is a
        non-numeric (object / complex / other) panel. The screen
        runs BEFORE any float coercion: previously :func:`brier`
        and :func:`log_score` entered through
        ``np.asarray(p, float)`` / ``np.asarray(y, float)``, which
        laundered a string panel ``['0.5', '0.5']`` into the numeric
        panel ``[0.5, 0.5]`` (scores bit-identical to the numeric
        controls, 0.25 / 0.6931) and a bool panel
        ``[True, False]`` into ``[1.0, 0.0]`` (``brier`` returned
        a perfect 0.0), while the sibling :func:`ece` in this file
        panel-validates its value domain under SF-BL-10038.
        Python list/tuple elements are scanned element-wise first,
        so a mixed panel such as ``[0.5, True]`` or
        ``[0.5, '0.5']`` -- which NumPy would promote to float64
        and hide -- is still caught, per the prereg wording "any
        bool or string element". The dtype screen then follows
        this file's SF-BL-11498 accept rule (kinds ``i``/``u``/``f``
        pass); object panels are rejected wholesale (the
        CSV-round-trip hazard in the SF-BL-11498 precedent),
        with bool/string object elements named specifically.
        Validation is pure -- a lawful numeric panel is returned
        via exactly the pre-fix ``np.asarray(values, dtype=float)``
        conversion -- so numeric-panel scores are bit-identical.
        The panel VALUE domains (length / empty / finite /
        [0, 1] / binary) are SF-BL-9741/9742/11267's separate
        screens in the callers, not this one.
    """
    # Scalar panel: a bare bool/string is a mistyped panel, not data.
    if isinstance(values, (bool, np.bool_)):
        raise ValueError(
            f"{name} must be a numeric (int/float) panel; got a bool "
            f"panel ({values!r}): booleans are not valid panel values "
            "(True/False would coerce to 1.0/0.0) (SF-BL-11354)")
    if isinstance(values, (str, bytes)):
        raise ValueError(
            f"{name} must be a numeric (int/float) panel; got a string "
            f"panel ({values!r}): string values (e.g. from a CSV "
            "round-trip) are not validated numbers (SF-BL-11354)")
    # Element scan for Python sequences: catches mixed panels whose
    # bool/string elements NumPy promotion to float64 would hide.
    if isinstance(values, (list, tuple)):
        for _el in values:
            _check_scoring_element(_el, name)
    arr = np.asarray(values)
    kind = arr.dtype.kind
    if kind in "iuf":
        return np.asarray(values, dtype=float)
    if kind == "b":
        raise ValueError(
            f"{name} must be a numeric (int/float) panel; got a bool "
            f"panel (dtype {arr.dtype}): booleans are not valid "
            "panel values (True/False would coerce to 1.0/0.0) "
            "(SF-BL-11354)")
    if kind in "US":
        raise ValueError(
            f"{name} must be a numeric (int/float) panel; got a "
            f"string panel (dtype {arr.dtype}): string values "
            "(e.g. from a CSV round-trip) are not validated "
            "numbers (SF-BL-11354)")
    if kind == "O":
        for _el in arr.flat:
            if isinstance(_el, (bool, np.bool_)):
                raise ValueError(
                    f"{name} must be a numeric (int/float) panel; "
                    f"got a bool element ({_el!r}) in an object "
                    "panel: booleans are not valid panel values "
                    "(True/False would coerce to 1.0/0.0) "
                    "(SF-BL-11354)")
            if isinstance(_el, (str, bytes)):
                raise ValueError(
                    f"{name} must be a numeric (int/float) panel; "
                    f"got a string element ({_el!r}) in an object "
                    "panel: string values (e.g. from a CSV "
                    "round-trip) are not validated numbers "
                    "(SF-BL-11354)")
        raise ValueError(
            f"{name} must be a numeric (int/float) panel; got an "
            f"object panel (dtype {arr.dtype}): non-numeric panels "
            "are not accepted (SF-BL-11354)")
    raise ValueError(
        f"{name} must be a numeric (int/float) panel; got a panel "
        f"with dtype {arr.dtype} (kind {kind!r}) -- string, bool "
        "and other non-numeric panels are not accepted "
        "(SF-BL-11354)")


def _check_scoring_element(el, name):
    """Element-wise bool/string scan for list/tuple panels
    (SF-BL-11354); recurses into nested list/tuple/array elements."""
    if isinstance(el, (bool, np.bool_)):
        raise ValueError(
            f"{name} must be a numeric (int/float) panel; got a bool "
            f"element ({el!r}): booleans are not valid panel values "
            "(True/False would coerce to 1.0/0.0) (SF-BL-11354)")
    if isinstance(el, (str, bytes)):
        raise ValueError(
            f"{name} must be a numeric (int/float) panel; got a string "
            f"element ({el!r}): string values (e.g. from a CSV "
            "round-trip) are not validated numbers (SF-BL-11354)")
    if isinstance(el, (list, tuple)):
        for _sub in el:
            _check_scoring_element(_sub, name)
    elif isinstance(el, np.ndarray):
        _validated_scoring_panel(el, name)


def brier(p, y):
    """Mean Brier score: ``mean((p - y) ** 2)``.

    Raises
    ------
    ValueError
        If ``p`` and ``y`` have different lengths, or if the panel
        is empty (SF-BL-9741). Previously the bare subtraction
        broadcast a singleton outcome across every forecast —
        ``brier([0.5, 0.6], [1])`` returned 0.205, grading two
        forecasts against one broadcast outcome — a singleton
        forecast broadcast the other way just as silently, and
        ``brier([], [])`` returned NaN (mean of an empty slice).
        Both now fail loudly, because this score selects the
        Wave-1 fit/test rule in :func:`fit_rules`: a broadcast or
        empty Brier there would silently pick the wrong rule.
        Validation is pure — it never transforms p or y — so
        matched non-empty panels are bit-identical to the
        pre-fix computation.
    ValueError
        If any forecast in ``p`` is non-finite or outside [0, 1],
        or if any outcome in ``y`` is not binary 0/1
        (SF-BL-11267). Previously ``brier([nan], [1])`` returned
        NaN, ``brier([1.5], [1])`` returned a plausible 0.25 for
        an impossible forecast, and ``brier([0.5], [2])`` returned
        an impossible 2.25 — the sibling :func:`ece` in this file
        has carried the equivalent screen since SF-BL-10038, but
        the Brier score that selects the best adjustment rule in
        :func:`fit_rules` had none. The screen runs BEFORE scoring
        and is pure validation (it never transforms p or y), so
        clean panels are bit-identical to the pre-fix computation.
        Kill criterion audited and not triggered: build_snapshots
        cannot guarantee in-range panels on every path, because
        fit_rules/main feed brier RULE-ADJUSTED probabilities and
        rule_shrink_fixed at grid s up to 1.15 maps lawful snapshot
        prices 0.001/0.999 to -0.074/1.074.
    ValueError
        If any element of ``p`` or ``y`` is a bool or a
        string, or either panel is a non-numeric panel
        (SF-BL-11354; see :func:`_validated_scoring_panel`).
        Previously both panels were coerced with a bare
        ``np.asarray(..., float)``: a string panel returned
        the numeric control bit-identically (0.25) and
        ``brier([True, False], [1, 0])`` returned a perfect
        0.0. The type screen runs first, before the length,
        finiteness and range screens above; numeric panels
        are bit-identical to the pre-fix computation.
    """
    p = _validated_scoring_panel(p, "p")
    y = _validated_scoring_panel(y, "y")
    if p.size != y.size:
        raise ValueError(
            f"len(p) ({p.size}) != len(y) ({y.size}): "
            "p and y must have the same length")
    if p.size == 0:
        raise ValueError(
            "brier: empty input: p and y must be non-empty "
            "(got length 0)")
    if not np.all(np.isfinite(p)):
        raise ValueError(
            "p must be finite probabilities in [0, 1]; "
            "got non-finite value(s) in p")
    lo, hi = float(p.min()), float(p.max())
    if lo < 0.0 or hi > 1.0:
        raise ValueError(
            f"p must lie in [0, 1]; got values outside that range "
            f"(min={lo!r}, max={hi!r})")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError(
            "y must be binary 0/1 outcomes; got non-binary value(s) in y")
    return float(np.mean((p - y) ** 2))


def log_score(p, y):
    """Mean negative log-likelihood of a forecast panel.

    Raises
    ------
    ValueError
        If ``p`` and ``y`` have different lengths, or if the panel is
        empty (SF-BL-9742). Previously the outcome array broadcast
        against the clipped forecasts, so ``log_score([0.5, 0.6], [1])``
        returned 0.602 — a truncated outcome column grading every extra
        forecast as if it shared the one outcome — and
        ``log_score([], [])`` returned NaN, a silent no-data value in
        the log-score column published beside Brier in the results.json
        rule comparisons. Validation is pure (it never transforms p or
        y; the EPS clip below is the pre-fix arithmetic), so every
        matched non-empty panel is bit-identical to the pre-fix
        computation.
    ValueError
        If any forecast in ``p`` is non-finite or outside [0, 1],
        or if any outcome in ``y`` is not binary 0/1
        (SF-BL-11267). Previously ``log_score([nan], [1])`` returned
        NaN and ``log_score([1.5], [1])`` returned ~1e-6 — the EPS
        clip below laundered an impossible forecast into a
        near-perfect log score — while the sibling :func:`ece` in
        this file has carried the equivalent screen since
        SF-BL-10038. The screen runs BEFORE the EPS clip and is
        pure validation (it never transforms p or y), so clean
        panels — including lawful endpoints 0/1, which the clip
        still handles exactly as before — are bit-identical to
        the pre-fix computation.
    ValueError
        If any element of ``p`` or ``y`` is a bool or a
        string, or either panel is a non-numeric panel
        (SF-BL-11354; see :func:`_validated_scoring_panel`).
        Previously both panels were coerced with a bare
        ``np.asarray(..., float)``: a string panel returned
        the numeric control bit-identically (0.6931) and
        a bool forecast panel was clipped into a ~1e-6
        near-perfect score. The type screen runs first,
        before the length screen and the EPS clip;
        numeric panels are bit-identical.
    """
    p_arr = _validated_scoring_panel(p, "p")
    y_arr = _validated_scoring_panel(y, "y")
    if p_arr.size != y_arr.size:
        raise ValueError(
            f"len(p) ({p_arr.size}) != len(y) ({y_arr.size}): "
            "p and y must have the same length")
    if p_arr.size == 0:
        raise ValueError(
            "log_score is undefined on an empty panel: "
            "len(p) == len(y) == 0")
    if not np.all(np.isfinite(p_arr)):
        raise ValueError(
            "p must be finite probabilities in [0, 1]; "
            "got non-finite value(s) in p")
    lo, hi = float(p_arr.min()), float(p_arr.max())
    if lo < 0.0 or hi > 1.0:
        raise ValueError(
            f"p must lie in [0, 1]; got values outside that range "
            f"(min={lo!r}, max={hi!r})")
    if not np.all((y_arr == 0) | (y_arr == 1)):
        raise ValueError(
            "y must be binary 0/1 outcomes; got non-binary value(s) in y")
    p = np.clip(p_arr, EPS, 1 - EPS)
    return float(-np.mean(y_arr * np.log(p) + (1 - y_arr) * np.log(1 - p)))


def _validate_ece_bins(bins) -> int:
    """Validate ``ece``'s ``bins`` argument TYPE and return it as a
    plain int (SF-BL-11314 type screen).

    Pure validation — a lawful bin count is returned via ``int()``
    unchanged in value, so integer-bin ECE values are bit-identical
    to the pre-fix computation, numpy-integer bins included.

    Raises
    ------
    ValueError
        Unless ``bins`` is a non-bool integer. Previously ``bins``
        was never type-checked — SF-BL-9743 guarded only its VALUE
        (``bins < 1``) — so a mistyped bin count reached the
        binning arithmetic raw: bool is an int subclass, so
        ``bins=True`` read as a 1-bin ECE and returned 0.0
        (perfect calibration) on the p=[0.9, 0.1] / y=[1, 0]
        fixture where the bins=10 control is 0.1; ``bins='10'``
        died in ``np.linspace(0, 1, bins + 1)`` with a bare
        ``TypeError: can only concatenate str (not "int") to str``
        that never named ``bins``; and ``bins=2.5`` died in
        ``range(bins)`` with a bare ``TypeError: 'float' object
        cannot be interpreted as an integer``. The screen is
        explicit, following this file's own type-screen precedent
        for an integer count (SF-BL-11530,
        :func:`_validate_snapshot_timing`'s ``horizon_d``) and the
        fleet's bin-count precedent
        (``calibration._validate_n_bins``, SF-BL-9722): booleans
        (including ``np.bool_``) are rejected outright, and
        ``bins`` must be an instance of ``numbers.Integral`` (int
        and numpy integers qualify); every float — including an
        integral float such as 10.0, which is a non-integer TYPE
        for a bin count — strings/bytes, None, complex,
        ``Fraction``/``Decimal``, sequences/mappings, ndarrays and
        all other non-integral objects raise the same documented
        ValueError naming ``bins`` — even where a downstream
        ``int()``/``float()`` coercion could have parsed them,
        because a bin count that arrives mistyped is corrupt
        state, not a number to be silently parsed. The VALUE
        domain of ``bins`` (``bins < 1``) is SF-BL-9743's separate
        guard in :func:`ece`, not this screen: a lawful-type
        bins below 1 keeps raising that error.
    """
    if isinstance(bins, (bool, np.bool_)) or not isinstance(
            bins, numbers.Integral):
        raise ValueError(
            "ece bins must be a non-bool integer number of bins "
            f"(SF-BL-11314); got bins={bins!r}")
    return int(bins)


def ece(p, y, bins=10):
    """Expected calibration error (mask-binned, count-weighted).

    Raises
    ------
    ValueError
        If ``p`` and ``y`` have different lengths, if any forecast in
        ``p`` is non-finite or outside [0, 1], or if any outcome in
        ``y`` is not binary 0/1 (SF-BL-10038). The screen runs BEFORE
        binning: previously a NaN forecast counted in the denominator
        ``n`` but fell in no bin (``ece([nan, 0.7], [0, 1])`` returned
        0.15, half the clean panel's 0.30) and ``ece([1.5], [1])``
        returned a perfect 0.0 for an impossible forecast. Validation
        is pure — it never transforms p or y — so clean panels are
        bit-identical to the pre-fix computation. Empty input and
        ``bins < 1`` (SF-BL-9743) also raise ValueError: an empty
        panel has no calibration to measure, and with ``bins < 1``
        the bin loop never executes, so both would otherwise return
        the 0.0 accumulator — a perfect-calibration value for a
        panel / rule table with no data behind it.
    ValueError
        If ``bins`` is a bool, a string or a non-integer float —
        or any other non-integer type (SF-BL-11314; see
        :func:`_validate_ece_bins`). The type screen runs FIRST,
        before the p/y screens: previously ``bins=True`` silently
        returned 0.0 (bool read as bins=1) against the 0.1 control
        at bins=10, and ``bins='10'`` / ``bins=2.5`` died with bare
        TypeErrors that never named ``bins``. Integer-bin outputs
        (int and numpy-integer bins) are bit-identical to the
        pre-fix computation.
    ValueError
        If any element of ``p`` or ``y`` is a bool or a string, if
        either panel is an object-dtype or other non-numeric panel,
        or if either panel is a bool/string scalar (SF-BL-11428;
        see :func:`_validated_scoring_panel`, the SF-BL-11354
        screen ``brier``/``log_score`` already use). Previously
        ``ece`` alone entered through a bare
        ``np.asarray(p, float), np.asarray(y, float)``: a string
        panel ``['0.9', '0.1']`` / ``['1', '0']`` returned 0.1
        bit-identical to the numeric control and
        ``ece([True, False], [1, 0])`` returned 0.0 — perfect
        calibration fabricated from booleans. The type screen runs
        after the ``bins`` screen and before the length, finiteness
        and range screens above; numeric panels are returned via
        exactly the pre-fix float conversion, so float-panel ECE
        values are bit-identical.
    """
    bins = _validate_ece_bins(bins)
    p = _validated_scoring_panel(p, "p")
    y = _validated_scoring_panel(y, "y")
    if len(p) != len(y):
        raise ValueError(
            f"len(p) ({len(p)}) != len(y) ({len(y)}): "
            "p and y must have the same length")
    if len(p) == 0:
        raise ValueError(
            "ece is undefined on empty input: len(p) == len(y) == 0; "
            "an empty panel has no calibration to measure")
    if p.size:
        if not np.all(np.isfinite(p)):
            raise ValueError(
                "p must be finite probabilities in [0, 1]; "
                "got non-finite value(s) in p")
        lo, hi = float(p.min()), float(p.max())
        if lo < 0.0 or hi > 1.0:
            raise ValueError(
                f"p must lie in [0, 1]; got values outside that range "
                f"(min={lo!r}, max={hi!r})")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError(
            "y must be binary 0/1 outcomes; got non-binary value(s) in y")
    # bins is a guaranteed non-bool Integral here (SF-BL-11314
    # screen above), so this SF-BL-9743 value check compares plainly.
    if bins < 1:
        raise ValueError(
            f"bins must be >= 1, got bins={bins!r}: with bins < 1 "
            "the bin loop never executes and ece would return 0.0 "
            "(perfect calibration) for any panel")
    edges = np.linspace(0, 1, bins + 1)
    out, n = 0.0, len(p)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            out += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(out)


def fit_rules(fit: pd.DataFrame):
    """Grid-search every rule's parameters on the fit split (Brier)."""
    y = fit["outcome"].to_numpy()
    p = fit["price"].to_numpy()
    days = fit["days_to_expiry"].to_numpy()
    z_mean = float(fit["log10_volume"].mean())
    z_sd = float(fit["log10_volume"].std() or 1.0)
    z = (fit["log10_volume"].to_numpy() - z_mean) / z_sd

    def score(fn, params):
        return brier(fn(p, days, z, params), y)

    fitted = {"raw": {}}
    best = (None, 1e9)
    # SF-BL-11375: the search grid (originally 0.70-1.15) is clipped
    # to the probability-preserving domain [0, 1] — effective grid
    # 0.70-1.00 — because rule_shrink_fixed now (correctly) raises for
    # s outside [0, 1], whose outputs can leave [0, 1]. np.clip (not a
    # shorter arange) so float accumulation (the nominal 1.00 grid
    # point is 1.0000000000000002) lands on exactly 1.0.
    for s in np.clip(np.arange(0.70, 1.151, 0.025), 0.0, 1.0):
        v = score(rule_shrink_fixed, {"s": float(s)})
        if v < best[1]:
            best = ({"s": float(s)}, v)
    fitted["shrink_fixed"] = best[0]

    best = (None, 1e9)
    for b in np.arange(0.60, 1.301, 0.02):
        v = score(rule_logit_shrink, {"b": float(b)})
        if v < best[1]:
            best = ({"b": float(b)}, v)
    fitted["logit_shrink"] = best[0]

    best = (None, 1e9)
    for b0 in np.arange(0.60, 1.21, 0.05):
        for b1 in np.arange(-0.10, 0.201, 0.025):
            v = score(rule_logit_time, {"b0": float(b0), "b1": float(b1)})
            if v < best[1]:
                best = ({"b0": float(b0), "b1": float(b1)}, v)
    fitted["logit_time"] = best[0]

    best = (None, 1e9)
    for b0 in np.arange(0.70, 1.16, 0.05):
        for b1 in np.arange(-0.10, 0.151, 0.05):
            for b2 in np.arange(-0.10, 0.151, 0.05):
                v = score(rule_logit_time_vol,
                          {"b0": float(b0), "b1": float(b1), "b2": float(b2)})
                if v < best[1]:
                    best = ({"b0": float(b0), "b1": float(b1), "b2": float(b2)}, v)
    fitted["logit_time_vol"] = best[0]

    # bucket recalibration: decile buckets, empirical rate shrunk to identity
    edges = np.quantile(p, np.linspace(0, 1, 11))
    edges[0], edges[-1] = 0.0, 1.0
    vals = []
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
    for k in range(10):
        m = idx == k
        if m.sum() == 0:
            vals.append(float(np.mean(p[(idx == max(k - 1, 0))])) if k else 0.05)
            continue
        rate, mean_p, n_k = float(y[m].mean()), float(p[m].mean()), int(m.sum())
        w = n_k / (n_k + 25.0)  # shrink toward raw price when bucket is thin
        vals.append(float(np.clip(w * rate + (1 - w) * mean_p, 0.001, 0.999)))
    fitted["bucket_recal"] = {"edges": [float(e) for e in edges],
                              "values": vals}
    return fitted, z_mean, z_sd


def calibration_table(df, best_rule, params, z_mean, z_sd):
    edges = [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1.0001]
    labels = ["0-.05", ".05-.1", ".1-.2", ".2-.3", ".3-.4", ".4-.5", ".5-.6",
              ".6-.7", ".7-.8", ".8-.9", ".9-.95", ".95-1"]
    adj = apply_rule(best_rule, df, params, z_mean, z_sd)
    rows = []
    for i, lab in enumerate(labels):
        m = (df["price"].to_numpy() >= edges[i]) & (df["price"].to_numpy() < edges[i + 1])
        if m.sum() == 0:
            continue
        rows.append({
            "bucket": lab, "n": int(m.sum()),
            "mean_market_price": round(float(df["price"].to_numpy()[m].mean()), 4),
            "empirical_freq": round(float(df["outcome"].to_numpy()[m].mean()), 4),
            "mean_best_rule_price": round(float(adj[m].mean()), 4),
        })
    return rows


# ---------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    kalshi = fetch_kalshi_probe()
    raw = fetch_gamma(refresh=args.refresh)
    markets = clean_markets(raw)
    print(f"clean binary markets: {len(markets)}", file=sys.stderr)
    snaps = build_snapshots(markets, refresh=args.refresh)
    n_dropped_invalid = snaps.attrs.get("n_dropped_invalid_price", 0)
    if n_dropped_invalid:
        print(f"dropped invalid-price snapshots: {n_dropped_invalid}",
              file=sys.stderr)
    if snaps.empty:
        raise SystemExit("no snapshots built — CLOB history unavailable")
    snaps.to_csv(DATA / "snapshots.csv", index=False)
    markets_out = markets.drop(columns=["token_yes"])
    markets_out.to_csv(DATA / "markets.csv", index=False)

    # split MARKETS by closing date: earliest 60% fit, latest 40% test
    mids = markets.sort_values("closed_time")["market_id"].tolist()
    cut = int(0.6 * len(mids))
    fit_ids, test_ids = set(mids[:cut]), set(mids[cut:])
    cutoff_date = str(markets.sort_values("closed_time")["closed_time"].iloc[cut])

    analyses = {}
    for name, sub in [("primary_T7", snaps[snaps["days_to_expiry"] == PRIMARY_HORIZON]),
                      ("pooled_all_horizons", snaps)]:
        fit = sub[sub["market_id"].isin(fit_ids)]
        test = sub[sub["market_id"].isin(test_ids)]
        if len(fit) < 50 or len(test) < 50:
            analyses[name] = {"n_fit": len(fit), "n_test": len(test),
                              "note": "too few snapshots, skipped"}
            continue
        params, z_mean, z_sd = fit_rules(fit)
        scores = {}
        for rname in RULE_FUNCS:
            pf = apply_rule(rname, fit, params[rname], z_mean, z_sd)
            pt = apply_rule(rname, test, params[rname], z_mean, z_sd)
            scores[rname] = {
                "params": params[rname],
                "fit_brier": round(brier(pf, fit["outcome"]), 5),
                "fit_log": round(log_score(pf, fit["outcome"]), 5),
                "test_brier": round(brier(pt, test["outcome"]), 5),
                "test_log": round(log_score(pt, test["outcome"]), 5),
                "test_ece": round(ece(pt, test["outcome"]), 5),
            }
        best_rule = min((r for r in RULE_FUNCS if r != "raw"),
                        key=lambda r: scores[r]["fit_brier"])
        analyses[name] = {
            "n_fit_snapshots": len(fit), "n_test_snapshots": len(test),
            "n_fit_markets": int(fit["market_id"].nunique()),
            "n_test_markets": int(test["market_id"].nunique()),
            "base_rate_test": round(float(test["outcome"].mean()), 4),
            "best_rule_by_fit": best_rule,
            "best_rule_beats_raw_on_test_brier":
                bool(scores[best_rule]["test_brier"] < scores["raw"]["test_brier"]),
            "scores": scores,
            "calibration_table_test": calibration_table(
                test, best_rule, params[best_rule], z_mean, z_sd),
            "raw_calibration_by_horizon_test": {
                str(int(h)): {"n": int((test["days_to_expiry"] == h).sum()),
                              "brier": round(brier(
                                  test.loc[test["days_to_expiry"] == h, "price"],
                                  test.loc[test["days_to_expiry"] == h, "outcome"]), 5)}
                for h in sorted(test["days_to_expiry"].unique())
            } if name == "pooled_all_horizons" else {},
        }

    results = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source": "Polymarket Gamma + CLOB (public APIs, fetched live)",
        "kalshi_probe": kalshi,
        "price_proxy": ("CLOB daily history: last print at/before close-horizon "
                        "(freshness window horizon+14d); horizons 30/14/7/3/1d; "
                        "headline = T-7d. NOT Gamma outcomePrices/lastTradePrice "
                        "(those are the resolution / ~0-1 at close)."),
        "filters": {"top_per_month_by_volume": TOP_PER_MONTH,
                    "min_volume_usd": MIN_VOLUME,
                    "windows": "2024-01..2026-09 monthly end-date windows",
                    "binary_yes_no_only": True,
                    "clean_resolution_only": "outcomePrices in {[1,0],[0,1]}"},
        "n_gamma_rows_fetched": len(raw),
        "n_clean_markets": len(markets),
        "n_snapshots_total": len(snaps),
        "markets_date_range": [str(markets["closed_time"].min()),
                               str(markets["closed_time"].max())],
        "split": {"method": "markets sorted by closed_time; earliest 60% fit, latest 40% test",
                  "cutoff_closed_time": cutoff_date,
                  "n_fit_markets": len(fit_ids), "n_test_markets": len(test_ids)},
        "total_volume_usd_clean_markets": round(float(markets["volume"].sum()), 2),
        "analyses": analyses,
        "caveats": [
            "Liquidity at close is ~0 for resolved markets; lifetime volume is the trust/liquidity proxy.",
            "Sample is the top-25-by-volume binary markets per month: it over-represents politics/crypto/sports event markets that attracted volume, and under-represents thin markets (survivorship/selection).",
            "Daily CLOB fidelity: the snapshot print can be up to 14 days older than its nominal horizon for illiquid stretches (freshness window); horizons are nominal, not exact.",
            "Snapshots of the same market at different horizons are correlated; pooled analysis counts them as samples, primary T-7 analysis does not.",
            "Markets within an event (e.g. mutually exclusive Fed-decision buckets) are correlated; no event-level clustering adjustment is applied.",
            "Kalshi not scored: settled feed dominated by multivariate combos with null last_price (see kalshi_probe).",
        ],
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    write_md(results)
    print(json.dumps({k: v for k, v in results.items()
                      if k in ("n_clean_markets", "n_snapshots_total")}, indent=2))
    for name, a in analyses.items():
        if "scores" in a:
            print(name, "raw test Brier", a["scores"]["raw"]["test_brier"],
                  "| best-by-fit:", a["best_rule_by_fit"],
                  a["scores"][a["best_rule_by_fit"]]["test_brier"])


def write_md(r: dict):
    a7 = r["analyses"].get("primary_T7", {})
    ap = r["analyses"].get("pooled_all_horizons", {})
    def beats(a):
        if "scores" not in a:
            return "n/a"
        return "YES" if a["best_rule_beats_raw_on_test_brier"] else "NO"
    lines = [
        "# Workstream D — Prediction Markets: calibration & adjustment rules",
        "",
        f"Generated {r['generated_at_utc']} from live public APIs. Research only — no betting.",
        "",
        "## Headline",
        "",
        f"- Clean resolved binary markets analysed: **{r['n_clean_markets']}** "
        f"({r['markets_date_range'][0]} → {r['markets_date_range'][1]}), "
        f"total lifetime volume ${r['total_volume_usd_clean_markets']:,.0f}.",
        f"- Forecast samples (market × horizon snapshots): **{r['n_snapshots_total']}**.",
        f"- Split by closing date: {r['split']['n_fit_markets']} fit markets / "
        f"{r['split']['n_test_markets']} test markets (cutoff {r['split']['cutoff_closed_time']}).",
    ]
    if "scores" in a7:
        s = a7["scores"]
        lines += [
            f"- **Primary (T-7d price): raw market test Brier = {s['raw']['test_brier']}, "
            f"log score = {s['raw']['test_log']}.** Best rule selected on fit "
            f"({a7['best_rule_by_fit']}): test Brier = {s[a7['best_rule_by_fit']]['test_brier']}, "
            f"log score = {s[a7['best_rule_by_fit']]['test_log']}. "
            f"Beats raw out-of-sample: **{beats(a7)}**.",
        ]
    if "scores" in ap:
        s = ap["scores"]
        lines += [
            f"- Pooled horizons: raw test Brier = {s['raw']['test_brier']}; "
            f"best-by-fit rule ({ap['best_rule_by_fit']}) test Brier = "
            f"{s[ap['best_rule_by_fit']]['test_brier']}. Beats raw: **{beats(ap)}**.",
        ]
    lines += ["", "## Price proxy (exactly what was used)", "", r["price_proxy"], "",
              "## Scores", ""]
    for name, a in r["analyses"].items():
        if "scores" not in a:
            lines += [f"### {name}", "", f"Skipped: {a.get('note')} "
                      f"(fit {a.get('n_fit')}, test {a.get('n_test')})", ""]
            continue
        lines += [f"### {name}  (fit n={a['n_fit_snapshots']}, test n={a['n_test_snapshots']}, "
                  f"test base rate={a['base_rate_test']})", "",
                  "| rule | params (fit only) | fit Brier | test Brier | test log score | test ECE |",
                  "|---|---|---|---|---|---|"]
        for rn, sc in a["scores"].items():
            prm = ", ".join(f"{k}={v}" for k, v in sc["params"].items()
                            if k not in ("edges", "values")) or "—"
            if rn == "bucket_recal":
                prm = "decile recal, shrink n/(n+25)"
            lines.append(f"| {rn} | {prm} | {sc['fit_brier']} | {sc['test_brier']} | "
                         f"{sc['test_log']} | {sc['test_ece']} |")
        lines += ["", f"Best rule by FIT Brier: **{a['best_rule_by_fit']}** "
                  f"(beats raw on test: {beats(a)})", ""]
        if a.get("calibration_table_test"):
            lines += ["Calibration on test (market vs best rule):", "",
                      "| price bucket | n | mean market price | empirical freq | mean best-rule price |",
                      "|---|---|---|---|---|"]
            for row in a["calibration_table_test"]:
                lines.append(f"| {row['bucket']} | {row['n']} | {row['mean_market_price']} | "
                             f"{row['empirical_freq']} | {row['mean_best_rule_price']} |")
            lines.append("")
        if a.get("raw_calibration_by_horizon_test"):
            lines += ["Raw market Brier by horizon (test, pooled): "
                      + ", ".join(f"T-{h}d: {v['brier']} (n={v['n']})"
                                  for h, v in a["raw_calibration_by_horizon_test"].items()), ""]
    lines += ["## Kalshi", "", r["kalshi_probe"].get("note", ""), "",
              "## Caveats", ""] + [f"- {c}" for c in r["caveats"]] + [
              "", "## Files", "",
              "- `predmarkets.py` — fetch + dataset + rules + scoring (numpy/pandas only).",
              "- `results.json` — machine-readable version of everything above.",
              "- `data/` — cached raw Gamma JSON per month, per-token CLOB histories, "
              "`markets.csv`, `snapshots.csv`.", ""]
    (HERE / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
