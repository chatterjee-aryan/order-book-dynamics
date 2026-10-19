"""Compare ETF excess-range deciles on all dates versus FOMC dates.

Run from project root:
    python plot_fomc_vs_all_days_four_channel.py

Requires existing daily raw channels:
    data/processed/fomc/fomc_daily_tcm.csv
and existing meeting data:
    data/processed/fomc/fomc_surprises.csv
Inflation daily raw channel is read from the daily CSV if present, otherwise
computed from data/raw/external/T10YIE.csv.

All-day channel decile boundaries are shared by the all-day and FOMC samples.
An additional FOMC-only decile plot reproduces the original event ranking.
No return-continuation or post-meeting range-continuation analysis.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

N_QUANTILES = 10
BASELINE_DAYS = 90
FOMC_FILE = Path("data/processed/fomc/fomc_surprises.csv")
DAILY_FILE = Path("data/processed/fomc/fomc_daily_tcm.csv")
T10YIE_FILE = Path("data/raw/external/T10YIE.csv")
OUTPUT_DIR = Path("data/processed/fomc/etf_four_channel_all_days_comparison")
PLOT_DIR = OUTPUT_DIR / "plots"
TICKERS = ["SPY", "GLD"]
SHOW_PLOTS = True

CHANNELS = [
    ("tcm_hl_bp", "Policy path (six-contract TCM)"),
    ("target_hl_bp", "Target rate (M0 / M1 fallback)"),
    ("curve_slope_abs_bp", "Yield curve |Δ(10Y−2Y)|"),
    ("inflation_abs_bp", "Inflation |Δ(10Y breakeven)|"),
]

def quantiles(series, n=N_QUANTILES):
    """Equal-count bins; ties split by input order, consistent with prior script."""
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    result = pd.Series(pd.NA, index=s.index, dtype="Int64")
    valid = s.dropna()
    if len(valid) < n:
        return result
    result.loc[valid.index] = pd.qcut(valid.rank(method="first"), q=n, labels=False).astype(int) + 1
    return result

def reference_bins(all_series, event_mask, n=N_QUANTILES):
    """Full-horizon channel cutoffs applied identically to all and FOMC days.

    rank-first quantiles are used for all-day bins, and FOMC observations
    inherit the exact same assigned bins (no independent reranking).
    """
    return quantiles(all_series, n)

def prior_nonfomc_mean(series, excluded_dates, window=BASELINE_DAYS):
    """Mean of strictly preceding valid nonmeeting observations, no lookahead."""
    series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    eligible = series.notna() & ~series.index.isin(excluded_dates)
    ordinary = series.loc[eligible]
    prev = ordinary.shift(1).rolling(window, min_periods=window).mean()
    incl = ordinary.rolling(window, min_periods=window).mean()
    mapped = incl.reindex(series.index).ffill()
    mapped.loc[eligible] = prev.reindex(mapped.index[eligible]).to_numpy()
    return mapped.where(mapped > 0)

def extract_field(raw, field):
    if isinstance(raw.columns, pd.MultiIndex):
        if field in raw.columns.get_level_values(0):
            out = raw.xs(field, axis=1, level=0)
        else:
            out = raw.xs(field, axis=1, level=1)
    else:
        out = raw[[field]].rename(columns={field: TICKERS[0]})
    out = out.reindex(columns=TICKERS)
    idx = pd.to_datetime(out.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    return out.sort_index()

def load_inflation_from_fred(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Daily inflation channel absent, and {path} not found. "
            "Supply the T10YIE FRED CSV or inflation_abs_bp in the daily CSV."
        )
    frame = pd.read_csv(path)
    date_col = next((c for c in frame if c.lower() in ("date", "observation_date")), frame.columns[0])
    val_col = "T10YIE" if "T10YIE" in frame else next(
        (c for c in frame if c != date_col), None
    )
    if val_col is None:
        raise ValueError(f"No breakeven value column in {path}")
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    vals = pd.to_numeric(frame[val_col], errors="coerce")
    # Do not forward fill: use consecutive valid FRED observations.
    s = pd.Series(vals.to_numpy(), index=dates).dropna()
    s = s.loc[~s.index.isna()].sort_index()
    return s.diff().abs() * 100  # FRED percent units -> basis points

def stats_by_bin(df, bin_col, ticker, sample, channel):
    v = df[[bin_col, f"{ticker}_excess_range", f"{ticker}_excess_range_decile"]].dropna()
    out = (v.groupby(bin_col)
           .agg(count=(f"{ticker}_excess_range_decile", "size"),
                mean_etf_decile=(f"{ticker}_excess_range_decile", "mean"),
                median_etf_decile=(f"{ticker}_excess_range_decile", "median"),
                mean_excess_range=(f"{ticker}_excess_range", "mean"))
           .reindex(range(1, N_QUANTILES + 1)).reset_index())
    out["ticker"] = ticker
    out["channel"] = channel
    out["sample"] = sample
    return out

def draw_comparison(ticker, panel_data, suffix, subtitle):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True,
                             constrained_layout=True)
    x = np.arange(1, N_QUANTILES + 1)
    for ax, (label, all_stats, fomc_stats) in zip(axes.flat, panel_data):
        ax.bar(x - .18, all_stats["mean_etf_decile"], width=.35,
               label="All trading days", alpha=.8)
        ax.bar(x + .18, fomc_stats["mean_etf_decile"], width=.35,
               label="FOMC announcement days", alpha=.8)
        ax.axhline((N_QUANTILES + 1) / 2, linestyle="--", color="black",
                   linewidth=.9, alpha=.6)
        n_all = int(all_stats["count"].sum())
        n_fomc = int(fomc_stats["count"].sum())
        ax.set_title(f"{label}\nAll n={n_all:,}; FOMC n={n_fomc:,}", fontsize=10)
        ax.set_xticks(x)
        ax.set_ylim(0, N_QUANTILES + .5)
        ax.grid(axis="y", alpha=.2)
    for ax in axes[-1, :]:
        ax.set_xlabel("Channel decile", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean ETF excess-range decile", fontsize=10)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.015), ncol=2)
    fig.suptitle(f"{ticker}: all dates vs FOMC D+0\n{subtitle}", fontsize=13)
    dest = PLOT_DIR / f"{ticker.lower()}_{suffix}.png"
    fig.savefig(dest, dpi=180, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    print("Saved:", dest)

def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    meetings = pd.read_csv(FOMC_FILE, parse_dates=["meeting_date"])
    if "meeting_date" not in meetings:
        raise ValueError("Missing meeting_date in FOMC file")
    meetings["meeting_date"] = meetings["meeting_date"].dt.normalize()
    if meetings["meeting_date"].duplicated().any():
        raise ValueError("Duplicate meeting dates")
    fomc_dates = pd.DatetimeIndex(meetings["meeting_date"].dropna())
    daily = pd.read_csv(DAILY_FILE, parse_dates=["date"]).set_index("date").sort_index()
    daily.index = daily.index.normalize()
    if daily.index.duplicated().any():
        raise ValueError("Duplicate dates in daily file")

    if "inflation_abs_bp" not in daily:
        daily["inflation_abs_bp"] = load_inflation_from_fred(T10YIE_FILE).reindex(daily.index)
    for channel, _ in CHANNELS:
        if channel not in daily:
            raise ValueError(f"Missing {channel} in {DAILY_FILE}")
        raw = pd.to_numeric(daily[channel], errors="coerce").replace([np.inf, -np.inf], np.nan)
        raw = raw.where(raw >= 0)
        daily[channel] = raw
        daily[f"{channel}_baseline_90d"] = prior_nonfomc_mean(raw, fomc_dates)
        daily[f"{channel}_normalized"] = raw / daily[f"{channel}_baseline_90d"]
        daily[f"{channel}_all_decile"] = reference_bins(daily[f"{channel}_normalized"], daily.index.isin(fomc_dates))
        # Legacy FOMC-only deciles, independently ranked among meetings.
        event_only = daily.loc[daily.index.isin(fomc_dates), f"{channel}_normalized"]
        daily[f"{channel}_fomc_decile"] = pd.Series(pd.NA, index=daily.index, dtype="Int64")
        daily.loc[event_only.index, f"{channel}_fomc_decile"] = quantiles(event_only)

    start = (daily.index.min() - pd.Timedelta(days=260)).strftime("%Y-%m-%d")
    end = (daily.index.max() + pd.Timedelta(days=8)).strftime("%Y-%m-%d")
    raw_etf = yf.download(TICKERS, start=start, end=end, auto_adjust=True,
                          actions=False, progress=False, threads=False)
    if raw_etf.empty:
        raise ValueError("No ETF data returned by yfinance")
    high = extract_field(raw_etf, "High")
    low = extract_field(raw_etf, "Low")
    ranges = (high - low).where((high >= low) & (low > 0))
    for ticker in TICKERS:
        baseline = prior_nonfomc_mean(ranges[ticker], fomc_dates)
        excess = (ranges[ticker] / baseline).replace([np.inf, -np.inf], np.nan)
        # Rank ETF range against the entire ETF trading-day horizon, as before.
        etf_deciles = quantiles(excess)
        daily[f"{ticker}_excess_range"] = excess.reindex(daily.index)
        daily[f"{ticker}_excess_range_decile"] = etf_deciles.reindex(daily.index)

    daily["is_fomc"] = daily.index.isin(fomc_dates)
    daily.to_csv(OUTPUT_DIR / "all_daily_channels_and_etf_ranges.csv", index_label="date")
    stats_rows = []
    for ticker in TICKERS:
        common_panels = []
        legacy_panels = []
        for channel, label in CHANNELS:
            all_stats = stats_by_bin(daily, f"{channel}_all_decile", ticker, "all_days", channel)
            fomc_stats = stats_by_bin(daily.loc[daily.is_fomc], f"{channel}_all_decile",
                                      ticker, "fomc_shared_bins", channel)
            fomc_legacy = stats_by_bin(daily.loc[daily.is_fomc], f"{channel}_fomc_decile",
                                       ticker, "fomc_only_bins", channel)
            stats_rows.extend([all_stats, fomc_stats, fomc_legacy])
            common_panels.append((label, all_stats, fomc_stats))
            legacy_panels.append((label, all_stats, fomc_legacy))
            print(f"{ticker:3} {channel:26} all={int(all_stats['count'].sum()):5} "
                  f"FOMC(shared)={int(fomc_stats['count'].sum()):3} "
                  f"FOMC(own)={int(fomc_legacy['count'].sum()):3}")
        draw_comparison(ticker, common_panels, "shared_deciles",
                        "Identical full-horizon channel deciles (directly comparable)")
        draw_comparison(ticker, legacy_panels, "fomc_only_deciles",
                        "FOMC-only channel deciles vs all-day deciles (different cutoffs)")
    pd.concat(stats_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "all_days_vs_fomc_decile_statistics.csv", index=False
    )
    print("\nDONE:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
