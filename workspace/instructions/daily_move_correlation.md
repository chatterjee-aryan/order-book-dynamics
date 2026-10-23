# Daily move correlations

The point here is to study correlations between our assets . We will study open close and high low. 

Dependency list :
    - workspace/instructions/futures_ticker_convention.md
    - workspace/instructions/futures_ticker.md
    - workspace/instructions/getting_futures_data.md
    - workspace/instructions/setting_futures_roll.md
    - workspace/instructions/ff_curve_pca.md

We will study correlations over the entire sample and on these particular meeting days. The dependecy files contain important data file references. 

Additional code reference : workspace/research/ff_curve/plot_curve_factors.py

## Steps

1. Grab the OHLCV data from the list. Focus on open close for now.
2. Work on the FF first, do PCA as provided in the code reference. Keep PC1 only. Make sure you winsorize first as before.
3. For all other futures, when you pick a open close observation, pick the primary one only
4. Now it is simple enough, for the 7 assets, find the corr matrix for the full sample and then for each of the event days as before. Plot a heat map.
5. Save plots to outputs/daily
6. Now we go to high low. The high-low becomes a measure of the day's vol. Do this. For each asset (corresponds to PC1 for FF). Find distribution. For each asset, do distribution on full days and on event days. Make histograms. Show plots. Save plots to outputs/daily. 
7. Print p5,p25,p50,p75,p95 for each asset on normal day and on FOMC day