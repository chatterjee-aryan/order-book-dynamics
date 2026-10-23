"""Download explicit futures OHLCV without instrument-definition history.

Instructions file: ../instructions/getting_futures_data.md
Run: python -m workspace.data_processing.get_futures_ohlcv_data

T labels are calendar-month candidate slots, NOT expiration-aware rolls. ZQ
uses the current month plus six months; other products use the next two eligible
contract months including the current month when eligible. Expired candidates
remain missing. Roll decisions belong in downstream code.

Each asset also has a long-format *_contracts.csv containing ALL returned
contract bars, including observations outside the calendar candidate slots.
Canonical two-digit-year labels, delivery year/month, vendor raw symbols, and
instrument IDs are preserved. CME raw symbols use a one-digit year; requests
are restricted to at most five years to avoid decade collisions in the local
symbol list. Historical API symbol mapping is date-aware.

Requests run concurrently across assets, with a separate client per worker.
Completed nonempty block downloads are cached under data/ohlcv/.cache. Delete
that cache or set USE_CACHE=False to refresh provider revisions. Empty results
and failed requests are not cached. Every requested weekday remains in the
T-labelled CSV; missing values are never filled.

Daily bars use UTC dates. Requests express the complete UTC bar boundaries in
America/New_York time; output labels preserve the provider's UTC dates.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import monotonic

import databento as db
import pandas as pd
from dotenv import load_dotenv

from workspace.utils import futures_ticker as tickers


START_DATE = "2010-06-07"
END_DATE = "2026-10-21"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "ohlcv"
DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1d"
TIMEZONE = "America/New_York"
BLOCK_YEARS = 1
FIELDS = ["open", "high", "low", "close", "volume"]
PRODUCTS = {
    "ZQ": tickers.fed_funds_ticker,
    "ZT": tickers.treasury_2y_ticker,
    "ZN": tickers.treasury_10y_ticker,
    "GC": tickers.gold_ticker,
    "ES": tickers.equity_ticker,
    "BTC": tickers.btc_usd_ticker,
    "6E": tickers.eur_usd_ticker,
}
MAX_WORKERS = 4
USE_CACHE = True
MAP_COLUMNS = ["date", "tenor", "canonical_symbol", "raw_symbol",
               "contract_year", "contract_month"]
BAR_COLUMNS = ["date", "raw_symbol", "instrument_id", *FIELDS]
LOG = logging.getLogger(__name__)


def candidate_dates(start: str | None, end: str | None) -> pd.DatetimeIndex:
    """Validate inclusive date constants and return candidate weekdays."""
    if start is None or end is None:
        raise ValueError("Set START_DATE and END_DATE before running")
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(first) or pd.isna(last):
        raise ValueError("Start and end must be valid dates")
    if first.tzinfo is not None or last.tzinfo is not None:
        raise ValueError("Use date-only START_DATE and END_DATE (YYYY-MM-DD)")
    if first != first.normalize() or last != last.normalize():
        raise ValueError("Use date-only START_DATE and END_DATE (YYYY-MM-DD)")
    if first > last:
        raise ValueError("START_DATE must not follow END_DATE")
    if first.dayofweek >= 5 or last.dayofweek >= 5:
        raise ValueError("START_DATE and END_DATE must both be weekdays")
    return pd.bdate_range(first, last, name="date")


def request_bounds(days: pd.DatetimeIndex) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Cover full provider UTC dates, expressed as New York-aware instants."""
    start = days[0].tz_localize("UTC").tz_convert(TIMEZONE)
    end = (days[-1] + pd.Timedelta(days=1)).tz_localize("UTC").tz_convert(TIMEZONE)
    return start, end


def date_blocks(days: pd.DatetimeIndex) -> list[pd.DatetimeIndex]:
    """Partition weekdays into non-overlapping BLOCK_YEARS windows."""
    if not 1 <= BLOCK_YEARS <= 5:
        raise ValueError("BLOCK_YEARS must be between 1 and 5")
    blocks = []
    start = days[0]
    while start <= days[-1]:
        end = start + pd.DateOffset(years=BLOCK_YEARS)
        block = days[(days >= start) & (days < end)]
        if not block.empty:
            blocks.append(block)
        start = end
    return blocks


def available_bounds(days: pd.DatetimeIndex, schema: str,
                     coverage: dict | None) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Intersect requests with entitled coverage; daily bars must be complete."""
    start, end = request_bounds(days)
    if coverage is not None:
        bounds = coverage.get("schema", {}).get(schema, coverage)
        available_start = pd.to_datetime(bounds["start"], utc=True)
        available_end = pd.to_datetime(bounds["end"], utc=True)
        if schema == SCHEMA:
            available_start = available_start.ceil("D")
            available_end = available_end.floor("D")
        start = max(start, available_start).tz_convert(TIMEZONE)
        end = min(end, available_end).tz_convert(TIMEZONE)
        if (start, end) != request_bounds(days):
            LOG.warning("%s: limiting request to available coverage: %s to %s",
                        schema, start, end)
    if start >= end:
        LOG.warning("%s: requested dates do not overlap available coverage", schema)
        return None
    return start, end


def make_symbol_map(root: str, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate calendar candidates locally, with no API or expiry calculations."""
    count = 7 if root == "ZQ" else 2
    convention = next(value for value in tickers.FUTURES_CONVENTIONS.values()
                      if value["root"] == root)
    rows = []
    for period in days.to_period("M").unique():
        if root == "BTC" and period < pd.Period("2017-12", freq="M"):
            continue
        contracts = []
        month = period
        while len(contracts) < count:
            if month.month in convention["months"]:
                canonical = PRODUCTS[root](month.year, month.month)
                contracts.append({
                    "tenor": f"{root}_T{len(contracts)}",
                    "canonical_symbol": canonical,
                    "raw_symbol": canonical[:-2] + canonical[-1],
                    "contract_year": month.year, "contract_month": month.month,
                })
            month += 1
        for day in days[days.to_period("M") == period]:
            rows.extend({"date": day, **contract} for contract in contracts)
    return pd.DataFrame(rows, columns=MAP_COLUMNS)


def download_bars(client: db.Historical, root: str, days: pd.DatetimeIndex,
                  mapping: pd.DataFrame, coverage: dict | None = None) -> pd.DataFrame:
    """Download deduplicated raw contracts once per block; reuse completed cache."""
    symbols = sorted(set(mapping["raw_symbol"]))
    if not symbols:
        return pd.DataFrame(columns=BAR_COLUMNS)
    if mapping.groupby("raw_symbol")["canonical_symbol"].nunique().gt(1).any():
        raise ValueError(f"{root}: raw symbols span multiple decades; shorten the block")
    if len(symbols) > 2000:
        raise ValueError(f"{root}: more than 2,000 symbols; shorten the block")
    bounds = available_bounds(days, SCHEMA, coverage)
    if bounds is None:
        return pd.DataFrame(columns=BAR_COLUMNS)
    start, end = bounds
    signature = json.dumps([DATASET, SCHEMA, str(start), str(end), symbols])
    digest = hashlib.sha256(signature.encode()).hexdigest()[:20]
    cache = OUTPUT_DIR / ".cache" / f"{root}_{digest}.csv"
    if USE_CACHE and cache.exists():
        LOG.info("%s: using cached block %s through %s", root, days[0].date(), days[-1].date())
        return pd.read_csv(cache, parse_dates=["date"])
    LOG.info("%s: downloading %d distinct contracts, %s through %s",
             root, len(symbols), days[0].date(), days[-1].date())
    began = monotonic()
    raw = client.timeseries.get_range(
        dataset=DATASET, schema=SCHEMA, stype_in="raw_symbol",
        symbols=symbols, start=start, end=end,
    ).to_df().reset_index()
    if raw.empty:
        LOG.warning("%s: no daily bars returned", root)
        return pd.DataFrame(columns=BAR_COLUMNS)
    required = {"ts_event", "symbol", "instrument_id", *FIELDS}
    if missing := required.difference(raw.columns):
        raise ValueError(f"{root}: missing OHLCV fields: {sorted(missing)}")
    raw["date"] = pd.to_datetime(raw["ts_event"], utc=True).dt.tz_localize(None).dt.normalize()
    raw = raw.rename(columns={"symbol": "raw_symbol"})
    bars = raw.loc[raw["date"].isin(days) & raw["raw_symbol"].isin(symbols), BAR_COLUMNS].copy()
    if bars.duplicated(["date", "raw_symbol"]).any():
        raise ValueError(f"{root}: ambiguous raw symbol/day bars; inspect instrument IDs")
    LOG.info("%s: received %s bars in %.1fs", root, f"{len(bars):,}", monotonic() - began)
    if USE_CACHE and not bars.empty:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix(".tmp")
            bars.to_csv(temporary, index=False)
            temporary.replace(cache)
        except OSError:
            LOG.exception("%s: could not cache bars; continuing with downloaded data", root)
    return bars


def make_ohlcv_frame(root: str, days: pd.DatetimeIndex, mapping: pd.DataFrame,
                     bars: pd.DataFrame) -> pd.DataFrame:
    """Build calendar candidate columns; preserve every requested weekday."""
    labels = [f"{root}_T{i}" for i in range(7 if root == "ZQ" else 2)]
    columns = pd.MultiIndex.from_product([labels, FIELDS], names=["ticker", "field"])
    output = pd.DataFrame(index=days, columns=columns, dtype=float)
    if mapping.empty or bars.empty:
        return output
    joined = mapping.merge(bars, on=["date", "raw_symbol"],
                           how="left", validate="many_to_one")
    for label in labels:
        values = joined.loc[joined["tenor"].eq(label)].set_index("date")
        for field in FIELDS:
            output[(label, field)] = values[field].reindex(days)
    return output


def download_asset(root: str, key: str, days: pd.DatetimeIndex,
                   coverage: dict | None) -> dict:
    """Use one client per asset; isolate block failures and keep successful data."""
    client = db.Historical(key)
    frames, mappings, contracts, errors = [], [], [], []
    blocks = date_blocks(days)
    for number, block_days in enumerate(blocks, start=1):
        block_label = f"{block_days[0]:%Y-%m-%d} through {block_days[-1]:%Y-%m-%d}"
        LOG.info("%s: block %d/%d (%s)", root, number, len(blocks), block_label)
        mapping = pd.DataFrame(columns=MAP_COLUMNS)
        frame = make_ohlcv_frame(root, block_days, mapping, pd.DataFrame())
        try:
            mapping = make_symbol_map(root, block_days)
            bars = download_bars(client, root, block_days, mapping, coverage)
            if not bars.empty:
                metadata = mapping.drop(columns=["date", "tenor"]).drop_duplicates()
                # Preserve all returned bars, not only those selected into T slots.
                contracts.append(bars.merge(metadata, on="raw_symbol", validate="many_to_one"))
            frame = make_ohlcv_frame(root, block_days, mapping, bars)
        except Exception as exc:
            errors.append(f"{block_label}: {type(exc).__name__}: {exc}")
            LOG.exception("%s: block %s failed; continuing", root, block_label)
        frames.append(frame)
        if not mapping.empty:
            mappings.append(mapping)
    frame = pd.concat(frames).reindex(days)
    mapping = (pd.concat(mappings, ignore_index=True) if mappings
               else pd.DataFrame(columns=MAP_COLUMNS))
    contract_bars = (pd.concat(contracts, ignore_index=True).sort_values(["date", "canonical_symbol"])
                     if contracts else pd.DataFrame(columns=[*BAR_COLUMNS, "canonical_symbol",
                                                            "contract_year", "contract_month"]))
    if errors:
        status = "error"
    elif not frame.notna().any().any():
        status = "no_data"
    elif frame.isna().any().any():
        status = "partial"
    else:
        status = "ok"
    failed_blocks = len(errors)
    suffix = f"{days[0]:%Y-%m-%d}_{days[-1]:%Y-%m-%d}"
    try:
        frame.to_csv(OUTPUT_DIR / f"{root}_{suffix}.csv", date_format="%Y-%m-%d")
        mapping.to_csv(OUTPUT_DIR / f"{root}_{suffix}_symbols.csv", index=False)
        contract_bars.to_csv(OUTPUT_DIR / f"{root}_{suffix}_contracts.csv", index=False)
        LOG.info("%s: saved %d dates and %d contract bars (status=%s)",
                 root, len(frame), len(contract_bars), status)
    except OSError as exc:
        status = "error"
        errors.append(f"saving CSV: {exc}")
        LOG.exception("%s: could not save CSV files", root)
    return {"root": root, "status": status, "error": "; ".join(errors),
            "blocks": len(blocks), "failed_blocks": failed_blocks,
            "requested_dates": len(days),
            "dates_with_data": int(frame.notna().any(axis=1).sum())}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    days = candidate_dates(START_DATE, END_DATE)
    date_blocks(days)  # Validate block size before any network calls.
    if MAX_WORKERS < 1:
        raise ValueError("MAX_WORKERS must be positive")
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError("DATABENTO_API_KEY missing; check the root .env file")
    try:
        coverage = db.Historical(key).metadata.get_dataset_range(dataset=DATASET)
    except Exception:
        LOG.exception("Could not check dataset coverage; asset requests will still be attempted")
        coverage = None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    statuses = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_asset, root, key, days, coverage): root
                   for root in PRODUCTS}
        for future in as_completed(futures):
            root = futures[future]
            try:
                statuses.append(future.result())
            except Exception as exc:
                LOG.exception("%s: asset download failed", root)
                statuses.append({"root": root, "status": "error",
                                 "error": f"{type(exc).__name__}: {exc}"})
    suffix = f"{days[0]:%Y-%m-%d}_{days[-1]:%Y-%m-%d}"
    status_path = OUTPUT_DIR / f"download_status_{suffix}.csv"
    pd.DataFrame(statuses).to_csv(status_path, index=False)
    failed = [item["root"] for item in statuses if item["status"] == "error"]
    if failed:
        LOG.error("Failed assets: %s. See %s", ", ".join(failed), status_path)


if __name__ == "__main__":
    main()
