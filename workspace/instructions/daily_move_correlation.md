# Daily open-to-close analysis

Study same-day cross-asset correlations, return distributions, lag-one
within-asset autocorrelation, and next-five-trading-day continuation. Use open-to-close movements, not high-low
ranges. This is descriptive analysis, not a backtest or causal event study.

## Dependencies and inputs

Paths below are relative to the project root:

- `workspace/instructions/futures_ticker_convention.md`
- `workspace/instructions/futures_ticker.md`
- `workspace/instructions/getting_futures_data.md`
- `workspace/instructions/setting_futures_roll.md`
- `workspace/instructions/ff_curve_pca.md`
- Reference code: `workspace/research/ff_curve/plot_curve_factors.py`

Code file: `workspace/research/daily/daily_correlations.py`

Run from project root:

```bash
./bin/python -m workspace.research.daily.daily_correlations
```

Use `data/ohlcv/ZQ_2010-06-07_2026-10-21.csv` and the corresponding ZT, ZN,
GC, ES, BTC, and 6E files for that interval. Non-ZQ files must contain the
`is_primary` fields produced by `set_futures_roll.py`.

Event-date files:

- `data/event/cpi_dates.csv`
- `data/event/fomc_dates.csv`
- `data/event/nfp_dates.csv`
- `data/event/pce_dates.csv`

Read the `date` column, ignore hyphen-only separators, and deduplicate dates.
Use the listed dates without shifting them. Dates label UTC daily bars, not
New York calendar days or isolated announcement windows.

## Construct the series

1. Convert Fed funds futures prices to implied rates using `100 - price`.
   Compute each T0-T6 rate's `close - open` in basis points. These are already
   changes; do not difference or calculate percentage changes of PC1 afterward.
2. For PCA, use dates complete across all seven maturity changes. Winsorize
   each maturity at its sample p5/p95, subtract its sample mean, and fit
   covariance PCA without standardizing the maturities. Keep PC1, orienting its
   largest absolute loading positively as in the reference code.
3. For the other six assets, select the same day's primary contract and compute
   `(close - open) / open`. Preserve missing observations. Require exactly one
   primary whenever any bar is present; do not difference across contracts.
   Do not winsorize or center these asset returns.

## Same-day correlations

1. Analyze all dates, then CPI, FOMC, NFP, and PCE dates separately.
2. Refit Fed funds PCA within each selected sample, matching the existing curve
   PCA analysis. Other series are the primary-contract returns on those dates.
3. Compute and print the seven-series Pearson correlation matrix. Use pairwise
   complete observations, with at least three per pair. A constant series has
   undefined correlation. Keep BTC's shorter history rather than truncating
   every other pair to BTC's history.
4. Display and save a heatmap for each sample, using a common -1 to +1 color
   scale and annotating correlations and pairwise observation counts. Print
   valid observation counts per series.

## Open-to-close distributions and moments

1. Fit one full-sample Fed funds PCA, with one set of full-sample p5/p95 caps
   and maturity means. Use those scores directly for every distribution subset.
   Do not refit or recenter separately by event for this analysis.
2. Show distributions for all dates, each event type, and normal dates. Normal
   dates exclude the union of CPI, FOMC, NFP, and PCE dates. Event samples can
   overlap and are not mutually exclusive.
3. Use FF PC1 in basis-point units and the other returns in percent. Other
   returns are not centered. The full-sample FF score mean is zero by
   construction, but conditional means need not be zero.
4. Plot a seven-panel density histogram for each sample. Use the same bin edges
   for a given asset across samples, and show the valid observation count.
5. Print mean, sample standard deviation (`ddof=1`), skewness, and excess
   kurtosis (normal distribution = 0), plus units and sample size, for each
   asset and sample. Do not print quantile tables.

## Lag-one autocorrelation: continuation or reversal

1. For each of the seven series, compute the Pearson correlation between its
   move on day t and its move on the next observed trading date. This is
   within-asset autocorrelation, not cross-asset lead-lag correlation.
2. Use the common full-sample FF PC1 scores from the distribution analysis on
   BOTH endpoints. Do not refit PCA on event dates for this calculation and do
   not take another change in PC1. Other assets retain their daily primary
   contract's within-day returns, even if the primary changes between dates.
3. Establish a common chronological calendar from dates where at least one of
   the seven series is observed. Remove shared all-missing dates. Friday can
   pair with Monday, and full-closure dates are skipped. Do not compact each
   asset's observations independently: if other assets traded on an intervening
   date, a missing endpoint for this asset invalidates the pair rather than
   jumping across that date. Do not fill missing returns with zero.
4. Construct next-trading-date pairs BEFORE applying any event filter. Report
   all pairs and subsets where the FIRST date is CPI, FOMC, NFP, PCE, or normal.
   The second date may have any label. Do not correlate one event release with
   the next release of that same event. The final sample date has no successor
   and cannot be a valid first endpoint.
5. Use complete endpoint pairs for each asset/sample, with at least three pairs
   and nonconstant values at both endpoints. Otherwise report NaN/N/A.
6. Print the lag-one autocorrelation table and its valid pair counts. Display
   and save a sample-by-asset heatmap, with a -1 to +1 color scale and counts.
   Positive correlation indicates linear continuation, negative indicates
   linear reversal, and near-zero indicates little linear relationship. This
   does not measure overnight returns or the probability of matching signs.

## Next-five-trading-day continuation

1. Keep the one-day results and add a second analysis for the following week.
   For each asset, correlate the day-t open-to-close move with the move from
   the NEXT trading date's open (t+1) to the FIFTH subsequent trading date's
   close (t+5). This spans five future trading observations, not five calendar
   days. It is not day-t versus day-t+5's one-day return.
2. Construct this forward window on the same shared observed trading calendar
   as the one-day analysis, BEFORE filtering by events. The event label belongs
   to day t only. Include all dates, each event type, and normal dates. Later
   days in the window may contain other events.
3. For ZT, ZN, GC, ES, BTC, and 6E, choose the primary contract on t+1 using
   its symbols-file mapping. Obtain BOTH t+1 open and t+5 close for that exact
   contract from the asset's `*_contracts.csv` file. Identify it by full delivery
   year and month within its asset, not the repeating one-digit raw year.
   Calculate `100 * (close[t+5] / open[t+1] - 1)`. Do not divide prices from
   different contracts, dynamically splice primary prices, or sum only the five
   intraday returns. This endpoint move includes the intervening overnight moves.
4. For FF, fix the seven T0-T6 contracts assigned on t+1. Use their contract-level
   opening and closing prices at the same endpoints and compute implied-rate
   changes in basis points: `100 * (price_open[t+1] - price_close[t+5])`.
   Require all seven endpoint changes. Winsorize each horizon using its own
   FULL forward-sample p5/p95 thresholds and subtract its forward-sample mean.
   Project these changes onto the daily full-sample PC1 loadings. Do not estimate
   a separate weekly PCA or difference daily PC1 scores. Day-t FF remains the
   original common full-sample daily PC1 score.
5. Preserve missing endpoints; if a fixed contract expires or has no endpoint
   data, omit that asset's pair. Do not substitute another contract or extend
   the horizon. Intervening bars are not required for an endpoint return. The
   last five calendar observations lack a complete future window and are omitted.
6. Print the continuation correlation table and pair counts. Display and save
   another sample-by-asset heatmap with the same -1 to +1 scale. Require at
   least three pairs and nonconstant endpoints, as in the one-day calculation.
   This is a correlation of daily moves with future five-day moves, rather than
   conventional equal-horizon autocorrelation. Windows overlap in the all-days
   sample, so pair counts do not represent independent observations.

## Outputs

Display all figures and save PNGs to `outputs/daily/`:

- `open_close_correlations_<sample>.png`
- `open_close_distributions_<sample>.png`
- `open_close_lag1_autocorrelations.png`
- `open_close_forward5_continuation.png`

Do not modify input data, create report CSVs, or save new high-low plots. Older
high-low plot files from earlier runs are not regenerated or automatically
removed. No test files or dry run are required.
