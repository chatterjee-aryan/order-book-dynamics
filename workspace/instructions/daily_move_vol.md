# Daily Move Vol

Here, we wish to do exploratory analysis of how much more volatile are these assets on event days compared to normal days

Measure of vol = High - Low / average of past 60 days high - low

Dependency instructions :
    workspace/instructions/daily_move_correlation.md
    workspace/instructions/ff_curve_pca.md
    workspace/instructions/futures_ticker_convention.md
    workspace/instructions/futures_ticker.md
    workspace/instructions/setting_futures_roll.md

## Steps

1. Get the OHLC data for these assets and use PC1 for Fed Funds
2. For each day, compute the high - low / average of past 30 days high - low. For PC1, do not do high low, just compute PC1 using high low
3. Compute the median , 75th pct for normal days. 
4. Do same for each asset, event day type
5. Plot verticle bar charts showing normal day median 75th pct and event day median and 75th pct . Keep asset names on x axis. Make 1 plot for each event type 
6. Show plots one by one
7. Save plots to outputs/daily