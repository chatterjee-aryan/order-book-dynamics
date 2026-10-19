"""Daily raw 4-channel FOMC measures and normalized announcement-day vectors.

Channels (bp):
  tcm_hl_bp          calendar-day weighted high-low ZQ M0..M5, with M0 adjusted
  target_hl_bp       M0 high-low adjusted by D/(D-day); M1 high-low only on month-end
  curve_slope_abs_bp abs daily change in (DGS10 - DGS2), in yield bp
  inflation_abs_bp   abs daily change in T10YIE breakeven inflation, in bp

For each FOMC announcement, normalize each raw channel against its OWN
preceding 90 valid NON-FOMC trading-day observations (no look-ahead).
The daily output remains unnormalized. FRED yields are forward-filled only.

Note: target_hl_bp is an unsigned *range*, not Kuttner's signed unexpected
policy decision. DGS10-DGS2 is a curve slope, not a structural term premium.
T10YIE is inflation compensation, not pure expected inflation.
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
DATA_END = "2026-10-17"  # exclusive
MEETING_START = "2011-01-01"
MEETING_END = "2026-10-16"
MEETING_FILE = Path("data/raw/external/fomc_meeting_dates.csv")
DGS2_FILE = Path("data/raw/external/DGS2.csv")
DGS10_FILE = Path("data/raw/external/DGS10.csv")
T10YIE_FILE = Path("data/raw/external/T10YIE.csv")
OUTPUT_DIR = Path("data/processed/fomc")
DAILY_FILE = OUTPUT_DIR / "fomc_daily_tcm.csv"
SURPRISE_FILE = OUTPUT_DIR / "fomc_surprises.csv"
SYMBOL_FILE = OUTPUT_DIR / "fomc_daily_symbols.csv"
MEETING_SYMBOL_FILE = OUTPUT_DIR / "fomc_meeting_symbols.csv"
LOOKBACK = 90
N_MONTHS = 6
MONTH_CODE = dict(zip(range(1, 13), "FGHJKMNQUVXZ"))
CHANNELS = ("tcm_hl_bp", "target_hl_bp", "curve_slope_abs_bp", "inflation_abs_bp")


def month_offset(day: pd.Timestamp, offset: int) -> pd.Timestamp:
    return pd.Timestamp(day.year, day.month, 1) + pd.DateOffset(months=offset)


def make_zq_symbol(month_start: pd.Timestamp) -> str:
    return f"ZQ{MONTH_CODE[month_start.month]}{month_start.year % 10}"


def announcement_dates_from_file(path: Path) -> pd.DatetimeIndex:
    raw = pd.read_csv(path)
    dates = pd.to_datetime(raw["meeting_date"], errors="coerce").dropna()
    dates = pd.DatetimeIndex(dates.dt.tz_localize(None).dt.normalize().unique()).sort_values()
    return pd.DatetimeIndex([
        d for i, d in enumerate(dates)
        if i == len(dates) - 1 or (dates[i + 1] - d).days > 1
    ])


def make_symbol_table(days: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame([
        {"date": day, **{f"M{j}": make_zq_symbol(month_offset(day, j))
                         for j in range(N_MONTHS)}}
        for day in days
    ])


def download_bars(symbol_table: pd.DataFrame) -> pd.DataFrame:
    key = os.getenv("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError("DATABENTO_API_KEY missing; check .env")
    symbols = sorted(set(symbol_table[[f"M{j}" for j in range(N_MONTHS)]].to_numpy().ravel()))
    print(f"Downloading {len(symbols)} distinct symbols in one request: {DATA_START} to {DATA_END}")
    raw = db.Historical(key).timeseries.get_range(
        dataset=DATASET, schema=SCHEMA, stype_in="raw_symbol",
        symbols=symbols, start=DATA_START, end=DATA_END,
    ).to_df().reset_index()
    if raw.empty:
        raise RuntimeError("No daily bars returned")
    raw["ts_event"] = pd.to_datetime(raw["ts_event"], utc=True)
    # Preserve the original script's UTC session-date convention.
    raw["date"] = raw["ts_event"].dt.tz_localize(None).dt.normalize()
    bars = raw[["date", "symbol", "high", "low"]].copy()
    if bars.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate symbol/date daily bars")
    print(f"Downloaded {len(bars):,} daily contract bars")
    return bars


def compute_daily_tcm(symbol_table: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    trading_dates = pd.DatetimeIndex(bars["date"].unique()).sort_values()
    table = symbol_table.loc[symbol_table["date"].isin(trading_dates)].copy()
    table = table.sort_values("date").reset_index(drop=True)
    for j in range(N_MONTHS):
        label = f"M{j}"
        lookup = bars.rename(columns={
            "symbol": label, "high": f"{label}_high", "low": f"{label}_low"
        })
        table = table.merge(lookup, on=["date", label], how="left", validate="one_to_one")
        table[f"{label}_range_bp"] = (
            table[f"{label}_high"] - table[f"{label}_low"]
        ).abs() * 100.0

    dates = pd.DatetimeIndex(table["date"])
    days_in_month = np.array([calendar.monthrange(d.year, d.month)[1] for d in dates], dtype=int)
    remaining = days_in_month - dates.day.to_numpy()
    weights = np.column_stack([
        remaining,
        *[np.array([calendar.monthrange(
            (d + pd.DateOffset(months=j)).year,
            (d + pd.DateOffset(months=j)).month
        )[1] for d in dates], dtype=int) for j in range(1, N_MONTHS)]
    ])
    ranges = table[[f"M{j}_range_bp" for j in range(N_MONTHS)]].to_numpy(dtype=float)
    active = weights > 0
    valid = np.all(~active | (np.isfinite(ranges) & (ranges >= 0)), axis=1)

    # Kuttner-style remaining-month correction for M0.
    # D/(D-d) is undefined on the last calendar day: omit M0 for TCM.
    # Only on that day, use the unadjusted M1 range for the target channel.
    correction = np.divide(
        days_in_month, remaining,
        out=np.full(len(remaining), np.nan, dtype=float), where=remaining > 0
    )
    target = ranges[:, 0] * correction
    target[~np.isfinite(ranges[:, 0])] = np.nan
    month_end = remaining == 0
    target[month_end] = ranges[month_end, 1]
    table["target_hl_bp"] = target
    table["target_contract_source"] = np.where(month_end, "M1_month_end_fallback", "M0_adjusted")
    table["m0_remaining_days"] = remaining
    table["m0_correction"] = correction

    adjusted = ranges.copy()
    adjusted[:, 0] = np.where(remaining > 0, target, 0.0)
    adjusted[~active] = 0.0
    tcm = np.sum(adjusted * weights, axis=1) / np.sum(weights, axis=1)
    tcm[~valid] = np.nan
    table["tcm_hl_bp"] = tcm
    table["horizon_days"] = weights.sum(axis=1)
    table["futures_status"] = np.where(valid, "ok", "missing_contract_high_low")
    return table


def load_treasury_series(path: Path, column: str) -> pd.DataFrame:
    raw = pd.read_csv(path, na_values=[".", "", "NA", "N/A", "null"])
    if "observation_date" not in raw.columns or column not in raw.columns:
        raise ValueError(f"Expected observation_date and {column} in {path}")
    raw["date"] = pd.to_datetime(raw["observation_date"], errors="coerce").dt.normalize()
    raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("date")
    if raw["date"].duplicated().any():
        raise ValueError(f"Duplicate observation_date in {path}")
    return raw[["date", column]]


def add_treasury_curve(daily: pd.DataFrame) -> pd.DataFrame:
    # Compute yield changes on the FRED observation calendar FIRST, then
    # join to futures sessions. This prevents a missing futures day from
    # incorrectly turning a two-day move into a one-day move.
    d2 = load_treasury_series(DGS2_FILE, "DGS2")
    d10 = load_treasury_series(DGS10_FILE, "DGS10")
    inflation = load_treasury_series(T10YIE_FILE, "T10YIE")
    treasury = d2.merge(d10, on="date", how="outer", validate="one_to_one")
    treasury = treasury.merge(inflation, on="date", how="outer", validate="one_to_one")
    treasury = treasury.sort_values("date").reset_index(drop=True)
    # Forward fill each yield independently; never backfill from the future.
    treasury[["DGS2", "DGS10", "T10YIE"]] = treasury[["DGS2", "DGS10", "T10YIE"]].ffill()
    treasury["curve_slope_pctpt"] = treasury["DGS10"] - treasury["DGS2"]
    # FRED DGS yields are in percent. 1 percentage point = 100 basis points.
    treasury["curve_slope_bp"] = treasury["curve_slope_pctpt"] * 100.0
    treasury["curve_slope_change_bp"] = treasury["curve_slope_bp"].diff()
    treasury["curve_slope_abs_bp"] = treasury["curve_slope_change_bp"].abs()
    # T10YIE is reported in percentage points; multiply by 100 for bp.
    treasury["inflation_change_bp"] = treasury["T10YIE"].diff() * 100.0
    treasury["inflation_abs_bp"] = treasury["inflation_change_bp"].abs()
    # Missing FRED quotes that were ffilled on the observation date can yield
    # a zero change; preserve it as a genuine ffill-based zero per user request.
    daily = daily.merge(treasury, on="date", how="left", validate="one_to_one")
    return daily


def compute_meeting_surprises(daily: pd.DataFrame,
                              announcements: pd.DatetimeIndex) -> pd.DataFrame:
    daily = daily.copy().sort_values("date")
    event_set = set(announcements)
    ordinary = daily.loc[~daily["date"].isin(event_set)].copy()
    indexed = daily.set_index("date")
    reference = {}
    for channel in CHANNELS:
        valid_ordinary = ordinary.loc[np.isfinite(ordinary[channel]), ["date", channel]]
        reference[channel] = (
            valid_ordinary["date"].to_numpy(dtype="datetime64[ns]"),
            valid_ordinary[channel].to_numpy(dtype=float),
        )

    rows = []
    for event in announcements:
        if not (pd.Timestamp(MEETING_START) <= event <= pd.Timestamp(MEETING_END)):
            continue
        out = {"meeting_date": event.date().isoformat()}
        for j in range(N_MONTHS):
            out[f"M{j}"] = make_zq_symbol(month_offset(event, j))
        exists = event in indexed.index
        event_row = indexed.loc[event] if exists else None
        out["futures_status"] = event_row["futures_status"] if exists else "missing_meeting_day"
        out["target_contract_source"] = event_row["target_contract_source"] if exists else "missing_meeting_day"
        for channel in CHANNELS:
            raw_value = float(event_row[channel]) if exists else np.nan
            dates, values = reference[channel]
            idx = np.searchsorted(dates, np.datetime64(event), side="left")
            prior = values[max(0, idx - LOOKBACK):idx]
            baseline = float(prior.mean()) if len(prior) == LOOKBACK else np.nan
            if not np.isfinite(raw_value):
                status = "missing_event_value"
            elif len(prior) < LOOKBACK:
                status = "insufficient_90_day_history"
            elif not np.isfinite(baseline) or baseline <= 0:
                status = "invalid_90_day_baseline"
            else:
                status = "ok"
            out[channel] = raw_value
            out[f"{channel}_baseline_90d"] = baseline
            out[f"{channel}_baseline_n"] = len(prior)
            out[f"{channel}_normalized"] = raw_value / baseline if status == "ok" else np.nan
            out[f"{channel}_status"] = status
        out["vector_complete"] = all(out[f"{c}_status"] == "ok" for c in CHANNELS)
        rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    announcements = announcement_dates_from_file(MEETING_FILE)
    candidates = pd.bdate_range(DATA_START, pd.Timestamp(DATA_END) - pd.Timedelta(days=1))
    symbols = make_symbol_table(candidates)
    symbols_out = symbols.copy()
    symbols_out["date"] = symbols_out["date"].dt.strftime("%Y-%m-%d")
    symbols_out.to_csv(SYMBOL_FILE, index=False)

    bars = download_bars(symbols)
    daily = compute_daily_tcm(symbols, bars)
    daily = add_treasury_curve(daily)
    daily["is_fomc"] = daily["date"].isin(announcements)
    meetings = compute_meeting_surprises(daily, announcements)

    # Daily values are RAW, never normalized.
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    daily.to_csv(DAILY_FILE, index=False)
    meetings.to_csv(SURPRISE_FILE, index=False)
    meeting_symbols = make_symbol_table(announcements).rename(columns={"date": "meeting_date"})
    meeting_symbols["meeting_date"] = meeting_symbols["meeting_date"].dt.strftime("%Y-%m-%d")
    meeting_symbols.to_csv(MEETING_SYMBOL_FILE, index=False)

    print(f"Saved daily raw vectors: {DAILY_FILE} ({len(daily):,} sessions)")
    print(f"Saved normalized FOMC vectors: {SURPRISE_FILE} ({len(meetings):,} meetings)")
    print(f"Complete four-channel vectors: {meetings['vector_complete'].sum()} / {len(meetings)}")
    for channel in CHANNELS:
        print(f"\n{channel} daily raw:\n{daily[channel].describe().to_string()}")
        print(f"{channel} meeting statuses:\n{meetings[f'{channel}_status'].value_counts().to_string()}")
        print(f"{channel} normalized FOMC:\n{meetings[f'{channel}_normalized'].describe().to_string()}")
    print("\nIncomplete FOMC vectors:")
    print(meetings.loc[~meetings["vector_complete"],
                       ["meeting_date"] + [f"{c}_status" for c in CHANNELS]].to_string(index=False))
    print("\nIMPORTANT: Verify Databento daily bar session dates and ZQ symbol year ambiguity.")


if __name__ == "__main__":
    main()
