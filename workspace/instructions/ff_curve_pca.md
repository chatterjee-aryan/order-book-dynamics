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

1. Go through the data file and 

