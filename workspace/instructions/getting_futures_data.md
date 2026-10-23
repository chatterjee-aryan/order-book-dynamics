# Getting Futures data

This file lists out what needs to be done to gather daily OHLC data for the futures

Dependency instructions : 
    - ROOT/workspace/instructions/futures_ticker.md 
    - ROOT/workspace/instructions/futures_ticker_convention.md 

Note, the coding file will be called using the module convention python -m from project root

## What I want to do

For each day in a given timeframe, I want to pick daily data for a bunch of symbols. I only need Open High Low Close Volume data for each day for now

## The key issue to keep in mind

The futures naming convention can be tricky. The symbol corresponding to the notion of "front month" can change as we move across months. 

## Futures needed

1. ZQ (Fed Funds)
2. ZT (2Y Treasury)
3. ZN (10Y Treasury)
4. GC (Gold)
5. ES (E-Mini S&P 500)
6. BTC (Bitcoin USD)
7. 6E (EUR/USD)

## Steps to Run

1. Take the timeframe which is the start date and end date. Assume these are inclusive. If any of these are weekends, just return an error. Produce a list of trading days using these end dates. The start and end dates should be kept as constants at the start of the file

2. For each date thus created, create a mapping from date to futures symbol. The following symbols must be there
    - The current ZQ Month (Front month)
    - The next 6 Months after that (so we have T0 to T6 for fed funds which gives us the entire policy path proxy)
    - The front and next month ZT
    - The front and next month ZN
    - The front and next month GC
    - The front and next month ES
    - The front and next month BTC
    - The front and next month 6E
Note, by front month, I mean the month in which the date falls. Use ROOT/workspace/utils/futures_ticker.py to do this. 

3. After doing this for each date, create a set of symbols. This is important because futures symbols repeat. For instance, Feb 2016 and Fed 2026 will have the same year code

4. Using the final symbol list, hit the databento realtimeapi. Use dotenv to get the API Key. I have a .env file in the root level with the key DATABENTO_API_KEY. Use schema OHLC-1d. The dataset will be GLBX.MDP3 and schema OHLCV-1d. Use `workspace/data_processing/get_ff_futures_data.py` (relative to the project root) as a reference to know how to call databento. For EUR/USD the dataset will be GLBX.MDP3 only and not Eurex so do not worry. Make sure we use new york datetimes. Use the reference file just in case.

5. Run the symbol call for each asset type but for the entire time interval. This keeps it easy for us as well as databento. Keep some progress logs. 

6. Once you have the data, you will need to convert it to a dataframe so that I may use it. I like the yfinance style format. In case you are confused by that, let me know. But the symbol col names should not be the ticker name for the futures contract. They should read FUTURES_ROOT_T0, FUTURES_ROOT_T1 and so on. Fed funds will run till T6 but others will only have T0 and T1. Make sure you use the date to symbol map to correctly label as T0, T1, etc. 

7. At this point, you should have one dataframe for each of the 7 asset types. The dataframe should be containing dates running as index. Then we have the major cols. And secondary cols called open, high, low, close, volume. Just as we have in the yfinance format. Save each dataframe as csv in ROOT/data/ohlcv/ . The filename should be of form FUTURES_ROOT_datestart_dateend.csv. Where futures root is ZQ,ZN, etc. 

8. By the end of this I should have ohlcv1d data in a nice format in the required folder. And futures symbol and roll should be taken care of.
