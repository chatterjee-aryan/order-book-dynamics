# FOMC Repricing and Cross-Market Liquidity: Project Progress

*Team discussion note — October 18, 2026*

## 1. What we are trying to understand

Our broader question is how major information events affect prices and liquidity across connected financial markets. We began with FOMC announcements and asked whether monetary-policy repricing is associated with unusually large price ranges in equities and gold.

Our approach is deliberately **not a directional forecasting exercise**. Rather than predicting whether the Fed will be hawkish or dovish, we measure **how intensely markets reprice** and how other markets respond. We also want to understand the intraday process: where repricing occurs, how liquidity changes, and how quickly different markets recover.

## 2. Measuring monetary-policy repricing

We constructed four daily, **unsigned** channels:

| Channel | Practical interpretation | Measurement idea |
|---|---|---|
| **Policy path** | How much the market revises the expected path of short-term policy rates | Combine the intraday high–low ranges of six consecutive Fed funds futures contracts (M0–M5), using calendar-aware weights and a remaining-month adjustment for the front contract |
| **Current target** | How much the near-term policy-rate expectation moves | Front-month Fed funds futures range, adjusted for remaining days in the month; next-month fallback near month-end |
| **Treasury curve slope** | Magnitude of repricing in the 10-year versus 2-year Treasury yield spread | Absolute daily change in the 10Y–2Y yield spread |
| **Inflation compensation** | Magnitude of repricing in market-implied inflation compensation | Absolute daily change in the 10-year breakeven inflation rate |

These are **market-based repricing measures**, not economist-survey forecast errors. The policy-path measure is a weighted composite of daily futures ranges, not a signed, pre/post announcement surprise. Likewise, the curve and breakeven channels should not be interpreted as pure structural term-premium or inflation-expectation shocks.

To make observations comparable across different market regimes, each channel is divided by its **average value over the preceding 90 valid non-FOMC trading days**. A normalized value of 2 means the day's movement is twice its recent ordinary-day baseline. This normalization uses only prior observations.

## 3. Initial descriptive comparison

We compared the normalized channels with the daily high–low ranges of **SPY** (equities) and **GLD** (gold), also normalized relative to recent non-FOMC days.

For each repricing channel, we grouped **all trading days** into deciles and compared the typical ETF range for:

- All days in the decile (including FOMC dates); and
- FOMC dates falling into the same decile.

This lets us ask: **For a similar observed amount of rates or macro repricing, does an FOMC date look different from an ordinary trading day?**

The plots suggested that SPY's range is especially elevated on FOMC dates when policy-path repricing is large. GLD showed an additional FOMC-day range premium across much of the distribution, including relatively quiet repricing bins.

These comparisons are descriptive; they do not establish that policy-path changes cause ETF volatility.

## 4. Interaction regressions

We then estimated regressions explaining daily ETF range using:

1. All four normalized repricing channels;
2. An indicator for FOMC announcement days; and
3. An interaction between the FOMC indicator and each repricing channel.

The interaction terms are the key: they test whether a channel's association with ETF range **changes on FOMC days**, after controlling for the other measured channels. We used heteroskedasticity- and autocorrelation-consistent (Newey–West, five-lag) standard errors.

Our preferred dependent variable is **excess ETF range**: the daily ETF high–low range divided by its recent 90-day non-FOMC baseline. We also ran specifications using the raw dollar range.

### Main results: normalized ETF ranges

| Result | SPY | GLD |
|---|---:|---:|
| Observations | 3,916 | 3,916 |
| FOMC observations | 131 | 131 |
| Full-model R² | **10.83%** | **10.19%** |
| R² without FOMC terms | 9.72% | 6.94% |
| FOMC × policy-path coefficient | **+0.2409** | **+0.1764** |
| Policy-path interaction significance | p < 0.001 | p < 0.001 |
| Joint test of four FOMC interactions | p = 2.7 × 10⁻⁶ | p = 2.0 × 10⁻⁵ |
| Baseline FOMC-day indicator | +0.0855 (not significant) | **+0.4796 (significant)** |

**Interpretation:** The policy-path repricing channel has a substantially stronger positive association with ETF price range on FOMC days than on ordinary days, conditional on the other channels. This appears for both SPY and GLD in the normalized-range specifications.

The other channels do matter on ordinary days: current-target repricing, Treasury-curve changes, and inflation-compensation changes generally have positive partial associations with ETF ranges. But **not every channel becomes more influential on FOMC dates**. In the SPY model, for example, the FOMC interactions for the current-target and inflation-compensation channels are negative. The distinctive *positive* FOMC interaction is concentrated in the policy-path measure.

GLD also exhibits a large positive FOMC-day baseline range premium even when the four normalized channels are at their ordinary-day reference values; SPY's corresponding baseline premium is not statistically significant.

### Important qualifications

- These are **contemporaneous explanatory associations**, not return forecasts or identified causal effects.
- The policy-path variable is an unsigned high–low range. It can reflect rapid two-way repricing and liquidity conditions as well as lasting revisions to expectations.
- **Robustness differs by specification:** SPY's positive policy-path interaction is also significant using raw dollar ranges; GLD's corresponding interaction is **not** significant in the raw-dollar specification. The normalized-range model is more comparable across years because ETF price levels change over time.
- We should audit a few small mismatches between the normalized FOMC values reconstructed for the regressions and the original normalized feature file before treating the results as finalized.
- We should also test subsamples, sensitivity to extreme meetings, nonlinear relationships, and common-support differences between FOMC and non-FOMC observations.

## 5. What we want to investigate next: intraday transmission

The daily results motivate a microstructure question:

> **Does liquidity recover first in the Fed funds futures market, while SPY and GLD are still adjusting to the announcement?**

We plan to reconstruct a synchronized **synthetic policy-path bid–ask spread** using the same six Fed funds futures contracts and the same weighting logic as our policy-path price index. For positive weights, this is the weighted sum of the six individual quoted spreads. We will compare it with the bid–ask spreads of SPY and GLD around the FOMC announcement.

The initial event-time study would align multiple meetings around **2:00 p.m. Eastern** (statement release), with a separate marker for the **2:30 p.m. press conference**. For each instrument, we would:

- Normalize spreads by their pre-announcement baseline;
- Construct **median spread trajectories across FOMC meetings**, with interquartile bands;
- Measure when spreads widen, peak, and sustainably return toward baseline; and
- Compare the timing of price repricing with the timing of liquidity recovery.

We have discussed calling the difference in recovery times **Liquidity Transmission Delay (LTD)**:

**LTD for an ETF = ETF spread recovery time − synthetic Fed funds policy-path spread recovery time.**

Positive LTD means the ETF's quoted spread recovers later than the futures composite. This is a descriptive timing statistic, **not proof that one market causally leads another**. We should distinguish liquidity recovery from price-discovery timing; prices could adjust almost simultaneously even if spreads normalize at different speeds.

Practical issues include synchronizing quotes, excluding stale or crossed markets, checking the depth available at each futures leg, and distinguishing the quoted spread of a synthetic index from an actually executable six-leg package.

## 6. Broader extension: CPI and nonfarm payrolls

The same framework could extend beyond FOMC meetings:

| Announcement | Candidate directly exposed markets | Downstream markets to compare |
|---|---|---|
| FOMC | Fed funds futures term structure | SPY, GLD, potentially Treasury ETFs |
| CPI | Rates futures, Treasury futures, inflation-linked markets | SPY, GLD, equity sectors |
| Nonfarm payrolls | Rates futures and Treasury futures | SPY, GLD, equity sectors |

The central idea is to use **observed repricing in instruments directly exposed to an announcement**, rather than relying exclusively on the difference between the published statistic and survey consensus. This allows us to study the **magnitude, path, and liquidity consequences** of market adjustment, including reversals that a single signed pre/post surprise might miss.

The caveat is that market-implied repricing is **not a pure measure of fundamental information**: it also reflects liquidity, positioning, and potentially multiple simultaneous interpretations of a release. CPI and payrolls also transmit information through several channels, so selecting a single primary market may be less straightforward than for FOMC.

## 7. Current takeaway

**We have preliminary evidence that the relationship between policy-path repricing and cross-asset price variation is materially different on FOMC dates.** The next stage is to see whether the *intraday liquidity dynamics* help explain this difference: specifically, whether the synthetic Fed funds policy-path market normalizes sooner than the ETFs responding to the same announcement.

The project is moving from a **daily cross-market association** to a **high-frequency study of price discovery and liquidity recovery**, with a methodology that may generalize across major macroeconomic announcements.
