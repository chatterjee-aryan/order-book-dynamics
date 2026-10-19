"""FOMC ETF study: excess intraday ranges D+0..D+5 and cumulative continuation.
Run from project root: python plot_etf_four_channel_event_days_continuation.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

N_QUANTILES = 10
BASELINE_DAYS = 90
MAX_LAG = 5
FOMC_FILE = Path('data/processed/fomc/fomc_surprises.csv')
OUTPUT_DIR = Path('data/processed/fomc/etf_four_channel_event_days_90d')
PLOT_DIR = OUTPUT_DIR / 'plots'
TICKERS = ['SPY', 'GLD']
CHANNELS = [
    ('tcm_hl_bp_normalized', 'Policy path (six-contract TCM)'),
    ('target_hl_bp_normalized', 'Target rate (M0 / M1 fallback)'),
    ('curve_slope_abs_bp_normalized', 'Yield curve |Δ(10Y−2Y)|'),
    ('inflation_abs_bp_normalized', 'Inflation |Δ(10Y breakeven)|'),
]
SHOW_PLOTS = True
BENCHMARK = (N_QUANTILES + 1) / 2


def make_quantiles(series, n=N_QUANTILES):
    """Equal-count quantiles, splitting ties in chronological order."""
    result = pd.Series(pd.NA, index=series.index, dtype='Int64')
    valid = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < n:
        raise ValueError(f'Need at least {n} observations; got {len(valid)}')
    bins = pd.qcut(valid.rank(method='first'), q=n, labels=False)
    result.loc[valid.index] = bins.astype(int) + 1
    return result


def extract_field(raw, field):
    if isinstance(raw.columns, pd.MultiIndex):
        if field in raw.columns.get_level_values(0):
            values = raw.xs(field, axis=1, level=0)
        elif field in raw.columns.get_level_values(1):
            values = raw.xs(field, axis=1, level=1)
        else:
            raise KeyError(f'Missing {field} in yfinance response')
    else:
        if len(TICKERS) != 1:
            raise ValueError('Unexpected non-MultiIndex yfinance response')
        values = raw[[field]].rename(columns={field: TICKERS[0]})
    values = values.reindex(columns=TICKERS)
    idx = pd.to_datetime(values.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    values.index = idx.normalize()
    return values.sort_index()


def previous_nonmeeting_mean(series, excluded_dates, window=BASELINE_DAYS):
    """Mean of strictly previous 90 valid non-FOMC sessions; no lookahead."""
    eligible = series.notna() & ~series.index.isin(excluded_dates)
    ordinary = series.loc[eligible]
    previous = ordinary.shift(1).rolling(window, min_periods=window).mean()
    including = ordinary.rolling(window, min_periods=window).mean()
    mapped = including.reindex(series.index).ffill()
    mapped.loc[eligible] = previous.reindex(mapped.index[eligible]).to_numpy()
    return mapped.where(mapped > 0)


def make_plot(ticker, lag, panels, kind):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.3), sharex=True,
                             sharey=True, constrained_layout=True)
    x = np.arange(1, N_QUANTILES + 1)
    for ax, ((channel, label), (stats, n)) in zip(axes.flat, zip(CHANNELS, panels)):
        if kind == 'range':
            ax.bar(x, stats['mean_etf_decile'], width=.75, alpha=.75,
                   label='Mean ETF decile')
            ax.plot(x, stats['median_etf_decile'], 'o-', markersize=3,
                    linewidth=1, color='tab:orange', label='Median ETF decile')
            ax.axhline(BENCHMARK, linestyle='--', linewidth=.9,
                       color='black', alpha=.65, label='All-day benchmark')
            ax.set_ylim(0, N_QUANTILES + .5)
        else:
            ax.bar(x, 100 * stats['cumulative_prob'], width=.75, alpha=.75,
                   label='Cumulative continuation probability')
            ax.axhline(50, linestyle='--', linewidth=.9,
                       color='black', alpha=.65, label='50% reference')
            ax.set_ylim(0, 100)
        ax.set_title(f'{label} (n={n})', fontsize=9)
        ax.set_xticks(x)
        ax.grid(axis='y', alpha=.2)
        ax.tick_params(labelsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel('FOMC channel quantile', fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel('ETF range quantile' if kind == 'range' else 'Probability (%)', fontsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, -.025),
               ncol=3, fontsize=8)
    if kind == 'range':
        fig.suptitle(f'{ticker}: D+{lag} excess range by FOMC channel quantile', fontsize=12)
        filename = f'{ticker.lower()}_dplus{lag}_four_channel_binning.png'
    else:
        fig.suptitle(f'{ticker}: D+{lag} cumulative since D+0 close direction continuation', fontsize=12)
        filename = f'{ticker.lower()}_dplus{lag}_cumulative_continuation.png'
    dest = PLOT_DIR / filename
    fig.savefig(dest, dpi=180, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    print('Saved:', dest)


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    # Remove obsolete daily continuation images from earlier versions.
    for ticker in TICKERS:
        for lag in range(1, MAX_LAG + 1):
            (PLOT_DIR / f'{ticker.lower()}_dplus{lag}_daily_continuation.png').unlink(missing_ok=True)

    fomc = pd.read_csv(FOMC_FILE, parse_dates=['meeting_date']).sort_values('meeting_date')
    required = ['meeting_date'] + [c for c, _ in CHANNELS]
    missing = [c for c in required if c not in fomc.columns]
    if missing:
        raise ValueError(f'Missing columns in {FOMC_FILE}: {missing}')
    fomc['meeting_date'] = fomc['meeting_date'].dt.normalize()
    if fomc['meeting_date'].duplicated().any():
        raise ValueError('Duplicate FOMC meeting dates')
    excluded_dates = pd.DatetimeIndex(fomc['meeting_date'].dropna())
    for col, _ in CHANNELS:
        fomc[col] = pd.to_numeric(fomc[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        if (fomc[col].dropna() < 0).any():
            raise ValueError(f'Negative normalized channel values in {col}')
        fomc[f'{col}_decile'] = make_quantiles(fomc[col])

    start = (fomc['meeting_date'].min() - pd.Timedelta(days=260)).strftime('%Y-%m-%d')
    end = (fomc['meeting_date'].max() + pd.Timedelta(days=18)).strftime('%Y-%m-%d')
    raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True,
                      actions=False, progress=False, threads=False)
    if raw.empty:
        raise ValueError('No ETF data returned by yfinance')
    opens = extract_field(raw, 'Open')
    closes = extract_field(raw, 'Close')
    high = extract_field(raw, 'High')
    low = extract_field(raw, 'Low')
    ranges = (high - low).where((high >= low) & (low > 0))

    daily = pd.DataFrame(index=ranges.index)
    for ticker in TICKERS:
        baseline = previous_nonmeeting_mean(ranges[ticker], excluded_dates)
        excess = (ranges[ticker] / baseline).replace([np.inf, -np.inf], np.nan)
        daily[f'{ticker}_open'] = opens[ticker]
        daily[f'{ticker}_close'] = closes[ticker]
        daily[f'{ticker}_range_move'] = ranges[ticker]
        daily[f'{ticker}_baseline_90d_move'] = baseline
        daily[f'{ticker}_excess_range'] = excess
        daily[f'{ticker}_excess_range_decile'] = make_quantiles(excess)
        daily[f'{ticker}_open_close_return'] = (closes[ticker] / opens[ticker] - 1).where(
            (opens[ticker] > 0) & closes[ticker].notna())
    daily.to_csv(OUTPUT_DIR / 'etf_all_daily_excess_ranges.csv', index_label='date')

    event_rows = []
    for ticker in TICKERS:
        etf_dates = daily.index[daily[f'{ticker}_open'].notna() & daily[f'{ticker}_close'].notna()]
        date_to_pos = {d: i for i, d in enumerate(etf_dates)}
        for _, meeting in fomc.iterrows():
            meeting_date = meeting['meeting_date']
            if meeting_date not in date_to_pos:
                continue
            pos = date_to_pos[meeting_date]
            d0 = daily.loc[meeting_date]
            r0 = d0[f'{ticker}_open_close_return']
            direction = np.sign(r0) if pd.notna(r0) else np.nan
            valid_direction = pd.notna(direction) and direction != 0
            d0_close = d0[f'{ticker}_close']
            for lag in range(MAX_LAG + 1):
                if pos + lag >= len(etf_dates):
                    continue
                date = etf_dates[pos + lag]
                obs = daily.loc[date]
                close = obs[f'{ticker}_close']
                cum_ret = (close / d0_close - 1) if pd.notna(close) and pd.notna(d0_close) and d0_close > 0 else np.nan
                row = {
                    'ticker': ticker, 'meeting_date': meeting_date,
                    'event_date': date, 'lag': lag,
                    'meeting_open_close_return': r0,
                    'meeting_direction': direction,
                    'cumulative_return_from_meeting_close': cum_ret,
                    'range_move': obs[f'{ticker}_range_move'],
                    'baseline_90d_move': obs[f'{ticker}_baseline_90d_move'],
                    'excess_range': obs[f'{ticker}_excess_range'],
                    'excess_range_decile': obs[f'{ticker}_excess_range_decile'],
                    'cumulative_continuation': (
                        int(direction * cum_ret > 0)
                        if lag > 0 and valid_direction and pd.notna(cum_ret) else np.nan
                    ),
                }
                for col, _ in CHANNELS:
                    row[col] = meeting[col]
                    row[f'{col}_decile'] = meeting[f'{col}_decile']
                event_rows.append(row)

    events = pd.DataFrame(event_rows)
    if events.empty:
        raise ValueError('No matched ETF event dates')
    events.to_csv(OUTPUT_DIR / 'fomc_etf_four_channel_event_days.csv', index=False)

    range_stats = []
    continuation_stats = []
    summary = []
    for ticker in TICKERS:
        for lag in range(MAX_LAG + 1):
            subset = events.loc[(events['ticker'] == ticker) & (events['lag'] == lag)]
            range_panels = []
            continuation_panels = []
            for channel, label in CHANNELS:
                dec_col = f'{channel}_decile'
                pair = subset[[dec_col, 'excess_range', 'excess_range_decile']].dropna()
                stats = (pair.groupby(dec_col)
                         .agg(count=('excess_range_decile', 'size'),
                              mean_etf_decile=('excess_range_decile', 'mean'),
                              median_etf_decile=('excess_range_decile', 'median'),
                              mean_etf_excess_range=('excess_range', 'mean'))
                         .reindex(range(1, N_QUANTILES + 1)).reset_index())
                stats['ticker'], stats['lag'], stats['channel'] = ticker, lag, channel
                range_stats.append(stats)
                range_panels.append((stats, len(pair)))
                if lag > 0:
                    directional = subset[[dec_col, 'cumulative_continuation']].dropna()
                    cs = (directional.groupby(dec_col)
                          .agg(count=('cumulative_continuation', 'size'),
                               cumulative_prob=('cumulative_continuation', 'mean'))
                          .reindex(range(1, N_QUANTILES + 1)).reset_index())
                    cs['ticker'], cs['lag'], cs['channel'] = ticker, lag, channel
                    continuation_stats.append(cs)
                    continuation_panels.append((cs, len(directional)))
            make_plot(ticker, lag, range_panels, 'range')
            if lag > 0:
                make_plot(ticker, lag, continuation_panels, 'continuation')
            valid_cum = subset['cumulative_continuation'].dropna()
            summary.append({
                'ticker': ticker, 'lag': lag,
                'range_n': int(subset['excess_range_decile'].notna().sum()),
                'mean_range_decile': pd.to_numeric(subset['excess_range_decile'], errors='coerce').mean(),
                'mean_excess_range': subset['excess_range'].mean(),
                'directional_n': len(valid_cum),
                'cumulative_continuation_prob': valid_cum.mean() if len(valid_cum) else np.nan,
            })
    pd.concat(range_stats, ignore_index=True).to_csv(
        OUTPUT_DIR / 'fomc_etf_four_channel_event_day_decile_statistics.csv', index=False)
    pd.concat(continuation_stats, ignore_index=True).to_csv(
        OUTPUT_DIR / 'fomc_etf_four_channel_continuation_by_decile.csv', index=False)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUT_DIR / 'fomc_etf_event_day_summary.csv', index=False)
    print('\nOverall event-day summary:')
    print(summary_df.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print('\nDONE:', OUTPUT_DIR)


if __name__ == '__main__':
    main()
