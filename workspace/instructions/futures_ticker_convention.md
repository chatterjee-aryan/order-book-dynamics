# Instructions for Futures Tickers

Futures symbols change with the contract expiration month and year. The same abbreviated symbol can also refer to contracts from different decades.

When retrieving historical futures data, always determine the exact contract and expiration before requesting data.

## General symbol convention

The standard futures symbol convention is:

`ROOT + MONTH_CODE + YEAR`

For this project, use a two-digit year in the canonical symbol.

Example: `ESZ26` means E-mini S&P 500 futures expiring in December 2026.

### Futures month codes

| Month | Code |

|---|---|

| January | F |

| February | G |

| March | H |

| April | J |

| May | K |

| June | M |

| July | N |

| August | Q |

| September | U |

| October | V |

| November | X |

| December | Z |

Important rules:

- A contract's expiration month is not necessarily the month in which it is most actively traded.

- A listed contract is not necessarily liquid.

- Do not automatically select the nearest expiration for every product.

- For historical data, select the contract that was appropriate on the historical observation date.

- Never infer the century from a one-digit year suffix alone.

- Use exchange metadata to verify that the contract existed and was trading on the requested date.

- Preserve the actual contract identifier alongside any continuous futures series.

- Exchange-native symbols and data-vendor symbols may differ.

Source: [CME futures contract specifications and month-code reference](https://www.cmegroup.com/)

## 1. Fed Funds Futures

****Exchange:**** CBOT / CME Group  

****Product:**** 30-Day Federal Funds Futures  

****Root symbol:**** `ZQ`

### Contract months

Fed Funds futures have contracts for every calendar month.

Valid month codes:

`F, G, H, J, K, M, N, Q, U, V, X, Z`

The expiration month corresponds to the calendar month over which the average effective federal funds rate is determined.

### Selection rule

Given an observation date:

1. Determine the observation's calendar month in the specified research timezone.

2. Advance exactly one calendar month.

3. Construct the ticker using `ZQ`, the corresponding month code, and the expiration year.

4. Handle December-to-January transitions correctly.

5. Verify that the contract is listed and available on the observation date.

### Example

Observation date: `2026-10-22`

- Current month: October 2026

- Next month: November 2026

- November code: `X`

- Required contract: `ZQX26`

Another example:

Observation date: `2026-12-15`

Required contract: `ZQF27` (January 2027).

### Sources

- [CME Group — 30-Day Federal Funds Futures](https://www.cmegroup.com/markets/interest-rates/stirs/30-day-federal-fund.html)

- [CBOT Rulebook Chapter 22 — Federal Funds Futures](https://www.cmegroup.com/rulebook/CBOT/III/22.pdf)

## 2. Gold (GC) Futures

****Exchange:**** COMEX / CME Group  

****Product:**** Gold Futures, 100 troy ounces  

****Root symbol:**** `GC`

### Contract months

The principal active gold contract months are:

| Expiration month | Code |

|---|---|

| February | G |

| April | J |

| June | M |

| August | Q |

| October | V |

| December | Z |

Therefore, the principal expiration cycle is:

`G, J, M, Q, V, Z`

GC also lists nearby serial months. Do not assume that the six principal months are the only contracts that can exist.

### Selection rule

For historical research:

1. Retrieve available GC contracts on the observation date.

2. Prefer the most actively traded contract, using historical trading volume.

3. Record the actual contract selected.

4. Use a consistent roll policy when constructing continuous daily prices.

5. Avoid switching contracts inside an intraday event window.

Do not blindly select the nearest calendar-month contract.

### Example

December 2026 gold futures:

- Root: `GC`

- December code: `Z`

- Year: `26`

- Symbol: `GCZ26`

June 2026 gold futures: `GCM26`.

These examples identify specific expirations, not necessarily the most liquid contract on every historical date.

### Sources

- [CME Group — Gold Futures](https://www.cmegroup.com/markets/metals/precious/gold.html)

- [CME Group — Trading COMEX Gold and Silver](https://www.cmegroup.com/education/articles-and-reports/trading-comex-gold-and-silver)

## 3. 2Y Treasury Note (ZT) Futures

****Exchange:**** CBOT / CME Group  

****Product:**** 2-Year U.S. Treasury Note Futures  

****Root symbol:**** `ZT`

### Contract months

ZT follows the quarterly expiration cycle:

| Expiration month | Code |

|---|---|

| March | H |

| June | M |

| September | U |

| December | Z |

Valid expiration codes:

`H, M, U, Z`

### Selection rule

For historical research:

1. Identify the listed quarterly contracts.

2. Select the active contract based on historical trading volume and the chosen roll methodology.

3. Do not assume that the calendar quarter alone determines the active contract.

4. Record the contract expiration and roll dates.

5. Avoid rolling during an event window.

Treasury futures are physically delivered, so the most liquid contract can change before its final trading date.

### Example

September 2026 2Y Treasury futures:

`ZTU26`

December 2026 2Y Treasury futures:

`ZTZ26`

### Sources

- [CME Group — 2-Year T-Note Futures](https://www.cmegroup.com/markets/interest-rates/us-treasury/2-year-us-treasury-note.html)

- [CME Group — Treasury Futures](https://www.cmegroup.com/markets/interest-rates/us-treasury.html)

## 4. 10Y Treasury Note (ZN) Futures

****Exchange:**** CBOT / CME Group  

****Product:**** 10-Year U.S. Treasury Note Futures  

****Root symbol:**** `ZN`

### Contract months

ZN follows the quarterly expiration cycle:

| Expiration month | Code |

|---|---|

| March | H |

| June | M |

| September | U |

| December | Z |

Valid expiration codes:

`H, M, U, Z`

### Selection rule

Use the same general contract-selection methodology as ZT:

1. Identify the listed quarterly expirations.

2. Determine the historically active contract.

3. Prefer historical volume or a documented roll schedule.

4. Preserve the selected contract's expiration.

5. Avoid mixing expirations within an intraday event window.

ZT and ZN do not necessarily roll on exactly the same date. Select each independently.

### Example

March 2026 10Y Treasury futures:

`ZNH26`

December 2026 10Y Treasury futures:

`ZNZ26`

### Sources

- [CME Group — 10-Year T-Note Futures](https://www.cmegroup.com/markets/interest-rates/us-treasury/10-year-us-treasury-note.html)

- [CME Group — Treasury Futures](https://www.cmegroup.com/markets/interest-rates/us-treasury.html)

## 5. EUR/USD Futures (6E)

**Exchange:** CME / CME Group  
**Product:** Euro FX Futures (EUR/USD)  
**Root symbol:** `6E`

IMPORTANT: This project uses CME Euro FX futures (`6E`), not Eurex EUR/USD futures (`FCEU`). Do not substitute the Eurex product.

### Contract months

The standard Euro FX futures expiration cycle is quarterly:

| Expiration month | Code |
|---|---|
| March | H |
| June | M |
| September | U |
| December | Z |

Standard quarterly expiration codes: `H, M, U, Z`.

CME may also list serial monthly expirations. Do not infer that a listed serial contract is the actively traded benchmark. Query historical exchange/vendor instrument definitions for the complete set of contracts available on the observation date.

### Symbol construction

For the project's canonical two-digit-year convention, use:

`6E + MONTH_CODE + TWO_DIGIT_YEAR`

This is a canonical contract label. Verify the data provider's accepted raw symbol and instrument ID before requesting historical data.

### Selection rule

1. Retrieve CME `6E` futures contracts listed on the historical observation date.
2. Identify the most actively traded contract, generally among the quarterly expirations, using historical volume or a documented roll policy.
3. Resolve the actual CME/data-vendor symbol and instrument ID.
4. Record its expiration date and the contract chosen for each observation date.
5. Keep the same contract throughout each intraday event window; do not splice two contracts inside an event.
6. Do not apply Eurex listing, trading-hour, or expiration rules to CME `6E`.

### Example

June 2026 Euro FX futures:

- Root: `6E`
- June code: `M`
- Year: `26`
- Canonical symbol: `6EM26`

December 2026 Euro FX futures: `6EZ26`.

These identify specific expirations, not necessarily the most liquid contract on every observation date.

### Sources

- [CME Group — Euro FX Futures](https://www.cmegroup.com/markets/fx/g10/euro-fx.html)
- [CME Group — Euro FX Futures Contract Specifications](https://www.cmegroup.com/markets/fx/g10/euro-fx.contractSpecs.html)

## 6. BTC/USD Futures (BTC)

****Exchange:**** CME  

****Product:**** Bitcoin Futures  

****Root symbol:**** `BTC`

This refers to standard CME Bitcoin futures, not Micro Bitcoin futures (`MBT`) or spot BTC/USD.

### Contract months

The current standard CME BTC futures listing schedule includes:

- Six consecutive monthly contracts.

- Four additional quarterly contracts in the March, June, September, December cycle.

- An additional December contract when required by the listing schedule.

Therefore, near-term BTC futures can have expirations in any month.

Valid month codes:

`F, G, H, J, K, M, N, Q, U, V, X, Z`

For historical research, do not assume today's listing schedule applied unchanged in every prior year.

CME Bitcoin futures began trading in December 2017. They cannot be used for dates before their launch.

### Selection rule

1. Retrieve BTC futures contracts listed on the observation date.

2. Identify the active contract using historical trading volume.

3. Resolve its expiration month and year.

4. Use the same active-contract selection methodology throughout the sample.

5. Do not confuse CME BTC futures with spot Bitcoin or other cryptocurrency futures products.

### Example

November 2026 Bitcoin futures:

`BTCX26`

December 2026 Bitcoin futures:

`BTCZ26`

### Sources

- [CME Group — Bitcoin Futures Specifications](https://www.cmegroup.com/markets/cryptocurrencies/bitcoin/bitcoin/specs.html)

- [CME Group — Original Bitcoin Futures Listing Specifications (2017)](https://www.cmegroup.com/content/dam/cmegroup/market-regulation/rule-filings/2017/12/17-417S.pdf)

## 7. Equity (ES) Futures

****Exchange:**** CME  

****Product:**** E-mini S&P 500 Futures  

****Root symbol:**** `ES`

### Contract months

ES follows the quarterly expiration cycle:

| Expiration month | Code |

|---|---|

| March | H |

| June | M |

| September | U |

| December | Z |

Valid expiration codes:

`H, M, U, Z`

### Selection rule

1. Retrieve listed ES quarterly contracts.

2. Identify the most liquid active contract on the historical date.

3. Apply a consistent historical roll policy.

4. Preserve the original contract expiration.

5. Do not switch contracts within an event window.

ES liquidity generally migrates from the expiring quarterly contract to the next quarterly contract before expiration.

### Example

September 2026 E-mini S&P 500 futures:

`ESU26`

December 2026 E-mini S&P 500 futures:

`ESZ26`

### Sources

- [CME Group — E-mini S&P 500 Futures](https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.html)

- [CME Group — Equity Index Futures](https://www.cmegroup.com/markets/equities.html)
