# Presentation Oct 23rd

Project : P116 - Order Book Dynamics Around Major Events

## What we wish to achieve

This is going to be the first presentation after the introductory meet. We basically need to show what progress we have made and first steps we have taken.

Following is what slides I want. I need you to make a latex presentation for me and also make a bash file to convert latex to pdf presentation using pandoc. 

## Slide 1

First slide. Just Title, nothing else needed

## Slide 2

This is the contents slide. It should contain the following line items as a numbered list. Feel free to improve the wording. It also functions as the meeting agenda and overview

1. What have we done so far
    - Preliminary explorations and getting to know the platform
    - Narrowing down an asset list
    - Data collection for daily data
    - Narrowing down of events we wish to study
    - Exploratory data analysis of daily moves across normal days and event days 
2. Discussion of future line items
    - Analysis of intra-day dynamics using MBO data
    - Dashboard for realtime monitoring of market tone

Again, try to improve the wording

## Slide 3

Title : Asset list 

We have decided to only use futures markets for this study. Futures markets have a unique combination of liquidity, high trading volume and long trading hours. Also, futures prices are a cleaner way to judge changes in market expectations.

Assets chosen :

1. Fed Funds Futures (ZQ) : Probably the most important market, especially for FOMC days. We study the current and next 6 month contracts in order to study how the entire policy path reprices
2. 2Y Treasury Notes (ZT) : Useful because it contains expectations of policy as well as inflation and flight to safety
3. 10Y Treasury Notes (ZN) : Contains information about long term inflation and term premium. Particularly useful with used along with ZT to see curve slope. And flight to safety
4. Gold (GC) : Information about inflation as well as flight to safety. Become a very hot topic over the past year. Extremely rate sensitive.
5. EUR/USD (6E) : Informational regarding the dollar's strength. Again, extremely rate sensitive.
6. S&P 500 (ES) : Informational for equity response to inflation and rates news. Important because it represents investor confidence in corporate earnings and growth.
7. Bitcoin (BTC) : New age asset. We feel there are possibilities that it has not been studied as much as the others on this list. Hot topic. Would be interesting to see how it reacts.

All of these futures are listed on the CME and data is available on Databento's GLBX.MDP3 dataset.

## Slide 4

Title : How we got and cleaned OHLCV-1d data for the futures

1. The main thing to be taken care of is the symbol list and the futures roll
2. For Fed Funds we take the most recent 7 futures contracts for each date. Fed funds futures in this range are all extremely liquid.
3. For all others, we pick the nearest and the next with one of them marked as primary for each date, this is how we take care of the roll.
4. The rolling is done unconventionally. We compute total volume of the 2 futures. The roll is done when the next future's volume is more than 65% of the total for 3 days straight. If the current future expires then the roll is done as usual. Rolling this way avoids the issue of picking the more illiquid future for analysis.

## Slide 5

Title : Studying OHLCV-1d data

This is the first section of results we wish to show. Think of this as the set up before the major intra-day study chunk. Many interesting results have come out already.

We have studied open-close returns and high-low moves for each bar. Open-Close corresponds to mid night to mid night in this case. But because CME is open during these hours, this should not be an issue although it falls outside of the regular trading hours. We can align time to the exact release when styding intra-day dynamics

For Fed Funds, we use the First principal component of open-close returns or high-low moves and not any individual month's contract. There is a lot of information in the curve in the first 6 months which gets lost if we pick any individual month

3 things have been studied on all days and compared specifically to event days :

1. Cross Correlation : Here, we want to see how these assets move together and if the dynamics change or become more intensified given the event.
2. Auto-Correlation : Over here, we are studying correlation between 1 day lagged open to close returns (marked as percentage of open). The idea here is to see if there is any continuation due to the event. For the event, the first day is the event day
3. Volatility of the moves : We have departed from the standard definition in this case. We use high-low as a measure of the day's vol. Volatility is a latent variable, using adjacent days to calculate can mask the event's effect. We normalise by the average of the past 60 days high-low to capture regime effects.

## Slide 6 

Title : Fed Funds PCA

Plot : outputs/ff_curve/ff_curve_pca_all_dates.png
Plot : outputs/ff_curve/ff_curve_pca_all_dates.png

Pretty much what we expect. We get level, slope, curvature. (this is the plot of the PC loadings). 

Key finding : The Level PCA is upward sloping like a ramp. We have not normalised open close returns by their std dev. The front months are probably pinned by the target FF rate while uncertainty increases as we go further out. So this makes sense.

## Slide 7

Title : Cross Asset Correlations

Here we are studying the cross correlations on normal days and event days

Plot : outputs/daily/open_close_correlations_all_dates.png

As expected, the rates group is highly correlated even on normal days

## Slide 8

Title : Correlation on FOMC days

Plot : outputs/daily/open_close_correlations_fomc_dates.png

The heatmap drastically changes with all of these assets showing much higher correlations. The correlations jump by several multiples.

## Slide 9

Title : Correlation on CPI days

Plot : outputs/daily/open_close_correlations_cpi_dates.png

## Slide 10

Title : Correlation on NFP days

Plot : outputs/daily/open_close_correlations_nfp_dates.png


## Slide 11

Title : Auto-Correlation

Auto-Correlation of open-close returns is a measure of continuation in the direction of the day's trend.

Plot : outputs/daily/open_close_lag1_autocorrelations.png

Key finding : Gold shows a tendency for continuation

## Slide 12

Title : Volatility of the moves

As mentioned before, we are studying the high-low, normalised using the average of the past 60 days. We compare the median and 75th percentile for normal and event days

Plot : outputs/daily/daily_vol_fomc_vs_normal.png


## Slide 13

Title : Volatility of the moves on CPI days

Plot : outputs/daily/daily_vol_cpi_vs_normal.png

## Slide 14

Title : Volatility of the moves on NFP days

Plot : outputs/daily/daily_vol_nfp_vs_normal.png

## Slide 15

Title : Proposals for intra-day research

These are proposals only and the team might decide to do something different if needed. But we wish to study the following three things

1. Liquidity
    - Bid ask 
    - Depth at level k
    - Additions and cancellations at level k
2. Order flow
    - Total order flow
    - Signed order flow
3. Price
    - Returns
    - EWMA vol

## Slide 16

Title: Method of study and key metrics

1. Trajectories
    - We can explore trajectories of the above features on the event day before and after the release as compared to the normal day
    - Key metrics can include expected time to peak/bottom and time to stabilisation
    - Some assets might show very fast or slow stabilisation or time to peak which might be interesting
2. Intra-day information flow
    - The point here is to study how information is transmitting through these markets during the event
    - We can study lead lag behavior across assets. Interesting questions pop up such as, does signed order flow in EUR/USD futures predict prices or order flow in Gold ? Does liquidity withdrawal in market predict withdrawal and magnitude in another over the next n-minutes?
    - Studying the flow of information intra-day could be potentially novel and interesting
    - Explore event-specific flow hierarchies: which markets lead, which follow, and whether that ordering differs across announcements.

## Slide 17

Title : About the dashboard

The project as outlined so far might already include too much. But we will do our best to make a dashboard too. The idea here is to provide real-time stats of many of the metrics such as order flow, etc. Including things like expected time to normalisation, etc. The dashboard can be conditioned on the event or not. It should update its forecasts depending on if an important release just took place.

Open to suggestions

## Slide 18

Thank you
    


## Presentation style

Use the standard Madrid Beamer theme, with Databento purple (#6266E5), charcoal (#323348), and white, verified against the official website stylesheet. Keep the standard Beamer typography and layouts.
