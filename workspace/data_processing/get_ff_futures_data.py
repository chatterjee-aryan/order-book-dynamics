"""Daily six-contract Fed funds futures curve range and FOMC surprise.

For each trading day, M0 is the current calendar month's ZQ future,
M1..M5 the following five months. TCM (bp) is the calendar-day-weighted
average of futures high-low ranges expressed in implied EFFR basis points.
M0 is scaled for the remaining calendar days after the observation date,
and omitted on the month's last calendar day.

FOMC surprise = announcement-day TCM / mean TCM of the preceding
90 VALID NON-FOMC trading days. Neither the event day nor earlier
FOMC days enter the baseline. Missing sessions are never filled with zero.

Install: pip install databento pandas numpy python-dotenv
Set DATABENTO_API_KEY in .env or your environment.
"""
from __future__ import annotations

import calendar
import os
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1d"
DATA_START = "2010-09-01"
DATA_END = "2026-10-17"  # Exclusive
MEETING_START = "2011-01-01"
MEETING_END = "2026-10-16"
MEETING_FILE = Path("data/raw/external/fomc_meeting_dates.csv")
OUTPUT_DIR = Path("data/processed/fomc")
DAILY_FILE = OUTPUT_DIR / "fomc_daily_tcm.csv"
SURPRISE_FILE = OUTPUT_DIR / "fomc_surprises.csv"
SYMBOL_FILE = OUTPUT_DIR / "fomc_daily_symbols.csv"
MEETING_SYMBOL_FILE = OUTPUT_DIR / "fomc_meeting_symbols.csv"
LOOKBACK = 90
N_MONTHS = 6
MONTH_CODE = dict(zip(range(1, 13), "FGHJKMNQUVXZ"))


def month_offset(day: pd.Timestamp, offset: int) -> pd.Timestamp:
    return pd.Timestamp(day.year, day.month, 1) + pd.DateOffset(months=offset)


def make_zq_symbol(month_start: pd.Timestamp) -> str:
    return f"ZQ{MONTH_CODE[month_start.month]}{month_start.year % 10}"


def announcement_dates_from_file(path: Path) -> pd.DatetimeIndex:
    raw = pd.read_csv(path)
    dates = pd.to_datetime(raw["meeting_date"], errors="coerce").dropna()
    dates = pd.DatetimeIndex(
        dates.dt.tz_localize(None).dt.normalize().unique()
    ).sort_values()
    return pd.DatetimeIndex([
        d for i, d in enumerate(dates)
        if i == len(dates) - 1 or (dates[i + 1] - d).days > 1
    ])


def make_symbol_table(days: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame([
        {"date": day, **{
            f"M{j}": make_zq_symbol(month_offset(day, j))
            for j in range(N_MONTHS)
        }}
        for day in days
    ])


def download_bars(symbol_table: pd.DataFrame) -> pd.DataFrame:
    """Fetch the complete date range and union of required raw symbols once."""
    key = os.getenv("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError("DATABENTO_API_KEY missing; check your .env file")

    contract_cols = [f"M{j}" for j in range(N_MONTHS)]
    symbols = sorted(set(symbol_table[contract_cols].to_numpy().ravel()))
    print(f"Downloading {len(symbols)} distinct symbols in one request: "
          f"{DATA_START} to {DATA_END} (exclusive)")
    try:
        raw = db.Historical(key).timeseries.get_range(
            dataset=DATASET,
            schema=SCHEMA,
            stype_in="raw_symbol",
            symbols=symbols,
            start=DATA_START,
            end=DATA_END,
        ).to_df().reset_index()
    except Exception as exc:
        raise RuntimeError(f"Databento download failed: {exc}") from exc

    if raw.empty:
        raise RuntimeError("No daily bars returned for the requested period")
    raw["ts_event"] = pd.to_datetime(raw["ts_event"], utc=True)
    # Databento daily bars are labeled by UTC session date.
    raw["date"] = raw["ts_event"].dt.tz_localize(None).dt.normalize()
    bars = raw[["date", "symbol", "high", "low"]].copy()
    if bars.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate symbol/day bars; inspect Databento resolution")
    print(f"Downloaded {len(bars):,} daily contract bars")
    return bars


def compute_daily_tcm(symbol_table: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    # The union of dates with observed contract bars determines the trading
    # calendar. We never assume every weekday is a trading session.
    trading_dates = pd.DatetimeIndex(bars["date"].unique()).sort_values()
    table = symbol_table.loc[symbol_table["date"].isin(trading_dates)].copy()
    table = table.sort_values("date").reset_index(drop=True)
    for j in range(N_MONTHS):
        label = f"M{j}"
        lookup = bars.rename(columns={"symbol": label,
                                      "high": f"{label}_high",
                                      "low": f"{label}_low"})
        table = table.merge(lookup, on=["date", label], how="left", validate="one_to_one")
        table[f"{label}_range_bp"] = (
            table[f"{label}_high"] - table[f"{label}_low"]
        ).abs() * 100.0

    dates = pd.DatetimeIndex(table["date"])
    days_in_month = np.array([
        calendar.monthrange(d.year, d.month)[1] for d in dates
    ], dtype=int)
    remaining = days_in_month - dates.day.to_numpy()
    weights = np.column_stack([
        remaining,
        *[
            np.array([calendar.monthrange(
                (d + pd.DateOffset(months=j)).year,
                (d + pd.DateOffset(months=j)).month
            )[1] for d in dates], dtype=int)
            for j in range(1, N_MONTHS)
        ]
    ])
    ranges = table[[f"M{j}_range_bp" for j in range(N_MONTHS)]].to_numpy(dtype=float)
    active = weights > 0
    valid = np.all(~active | (np.isfinite(ranges) & (ranges >= 0)), axis=1)

    # Algebraically identical to scaling M0 by D/remaining, weighting
    # it by remaining, and then taking the weighted average.
    adjusted = ranges.copy()
    np.multiply(
        adjusted[:, 0],
        np.divide(days_in_month, remaining, out=np.zeros(len(remaining), dtype=float), where=remaining > 0),
        out=adjusted[:, 0],
    )
    adjusted[~active] = 0.0
    tcm = np.sum(adjusted * weights, axis=1) / np.sum(weights, axis=1)
    tcm[~valid] = np.nan
    table["tcm_hl"] = tcm
    table["horizon_days"] = weights.sum(axis=1)
    table["status"] = np.where(valid, "ok", "missing_contract_high_low")
    table["date"] = table["date"].dt.strftime("%Y-%m-%d")
    return table


def compute_meeting_surprises(daily: pd.DataFrame, announcements: pd.DatetimeIndex) -> pd.DataFrame:
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")
    event_set = set(announcements)
    ordinary = daily.loc[
        ~daily["date"].isin(event_set) & daily["tcm_hl"].notna(),
        ["date", "tcm_hl"],
    ].copy().reset_index(drop=True)
    ordinary_dates = ordinary["date"].to_numpy(dtype="datetime64[ns]")
    ordinary_values = ordinary["tcm_hl"].to_numpy(dtype=float)
    indexed = daily.set_index("date")
    rows = []
    for event in announcements:
        if not (pd.Timestamp(MEETING_START) <= event <= pd.Timestamp(MEETING_END)):
            continue
        out = {"meeting_date": event.date().isoformat()}
        for j in range(N_MONTHS):
            out[f"M{j}"] = make_zq_symbol(month_offset(event, j))
        if event not in indexed.index:
            out.update(tcm_hl=np.nan, baseline_hl_90d=np.nan,
                       surprise=np.nan, surprise_hl=np.nan,
                       baseline_n=0, status="missing_meeting_day")
        else:
            event_row = indexed.loc[event]
            event_tcm = float(event_row["tcm_hl"])
            count_before = int(np.searchsorted(ordinary_dates, np.datetime64(event), side="left"))
            previous = ordinary_values[max(0, count_before - LOOKBACK):count_before]
            baseline = float(previous.mean()) if len(previous) == LOOKBACK else np.nan
            if not np.isfinite(event_tcm):
                status = "missing_meeting_contract_range"
            elif len(previous) < LOOKBACK:
                status = "insufficient_90_day_history"
            elif not np.isfinite(baseline) or baseline <= 0:
                status = "invalid_90_day_baseline"
            else:
                status = "ok"
            surprise = event_tcm / baseline if status == "ok" else np.nan
            out.update(tcm_hl=event_tcm, baseline_hl_90d=baseline,
                       surprise=surprise, surprise_hl=surprise,
                       baseline_n=len(previous), status=status)
        rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    announcements = announcement_dates_from_file(MEETING_FILE)
    # Business days are a candidate calendar only; actual session dates
    # are obtained from Databento daily bars.
    candidates = pd.bdate_range(DATA_START, pd.Timestamp(DATA_END) - pd.Timedelta(days=1))
    symbols = make_symbol_table(candidates)
    symbols_out = symbols.copy()
    symbols_out["date"] = symbols_out["date"].dt.strftime("%Y-%m-%d")
    symbols_out.to_csv(SYMBOL_FILE, index=False)
    print(f"Saved daily symbol table: {SYMBOL_FILE} ({len(symbols)} candidate days)")

    bars = download_bars(symbols)
    daily = compute_daily_tcm(symbols, bars)
    daily.to_csv(DAILY_FILE, index=False)
    print(f"Saved daily TCM: {DAILY_FILE} ({len(daily)} sessions)")

    meeting_symbols = make_symbol_table(announcements)
    meeting_symbols = meeting_symbols.rename(columns={"date": "meeting_date"})
    meeting_symbols["meeting_date"] = meeting_symbols["meeting_date"].dt.strftime("%Y-%m-%d")
    meeting_symbols.to_csv(MEETING_SYMBOL_FILE, index=False)

    meetings = compute_meeting_surprises(daily, announcements)
    meetings.to_csv(SURPRISE_FILE, index=False)
    print(f"Saved normalized surprises: {SURPRISE_FILE}")
    print("\nDaily status counts:\n", daily["status"].value_counts(dropna=False).to_string())
    print("\nMeeting status counts:\n", meetings["status"].value_counts(dropna=False).to_string())
    print("\nDaily TCM (bp):\n", daily["tcm_hl"].describe().to_string())
    print("\nNormalized meeting surprises (multiples):\n", meetings["surprise"].describe().to_string())
    print("\nNon-ok meeting rows:\n", meetings.loc[meetings["status"] != "ok", ["meeting_date", "status", "baseline_n"]].to_string(index=False))
    print("\nNote: 2010-09-01 may not provide 90 ordinary sessions before the earliest 2011 meetings.")
    print("Check session-date labeling and historical raw-symbol identity before interpreting results.")


if __name__ == "__main__":
    main()
