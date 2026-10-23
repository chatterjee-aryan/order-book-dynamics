# FF Curve PCA

Dependency instruction files :
    - getting_futures_data.md
    - futures_ticker_convention.md
    - futures_ticker.md

## What we want to do

The fed funds curve is the front end of the yield curve. When studying the dynamics, we usually want to look at curve factors like level, slope, curvature. So when finding interactions with fed fund futures, it is often better to look at the factors instead of just one or two of the futures

The data file is at data/ohlcv/ZQ_2010-06-07_2026-10-21.csv

Reference code is at workspace/data_processing/get_futures_ohlcv_data.py and workspace/data_processing/set_futures_roll.py

## Steps

1. Go through the data file and make a dataframe out of it
2. For each futures price, convert to rate using 100 - P. This is the IMM rate and not yield. But that is okay. 
3. At this point you will have Open High Low Close of the rates. We will study change in rates to do PCA
4. First, focus on open close. Find change in rate for each month future using the open and close. Make a time series out of this. Do PCA. Find first three factors and plot the loadings against the month index 0 to 6. This should look like level slope curvature.
5. Do same for high low moves. 
6. Do same thing for meeting dates. Dates are in :
    - data/event/cpi_dates.csv
    - data/event/fomc_dates.csv
    - data/event/nfp_dates.csv
    - data/event/pce_dates.csv
So go through data file, pick only the dates in one of these meeting dates and then do PCA again. Please print out number of observations/dates too. Do this for each meeting type separately.
6. That is it. This is mainly a plotting excercise.

