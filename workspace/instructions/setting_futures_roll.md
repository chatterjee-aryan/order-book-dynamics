# Setting the Primary Futures Contract

Dependency instructions:
- `ROOT/workspace/instructions/futures_ticker.md`
- `ROOT/workspace/instructions/futures_ticker_convention.md`

## What we need to achieve

Identify the primary contract using observed daily volume. Add an `is_primary` field under each contract tenor in the existing OHLCV dataframe. This replaces the previous requirement to determine expiration dates and add an `expired` flag.

Use the downloaded data only; no Databento definitions API calls are needed.

Do not apply this process to Fed funds (`ZQ`). Leave all Fed funds files unchanged.

Apply it to ZT, ZN, GC, ES, BTC, and 6E, each of which has T0 and T1.

## Input files

Data file: `PROJECT_ROOT/data/ohlcv/FUTURES_ROOT_START_DATE_END_DATE.csv`

Symbols file: `PROJECT_ROOT/data/ohlcv/FUTURES_ROOT_START_DATE_END_DATE_symbols.csv`

Reference code: `workspace/data_processing/get_futures_ohlcv_data.py`

The data file has a date index and two column levels: contract tenor, then field.

For example, `(6E_T0, volume)` and `(6E_T1, volume)` contain daily contract volumes.

The symbols file identifies the actual contract assigned to each tenor on each observation date. Use the full contract year and month to distinguish contracts whose raw symbols repeat in different decades.

## Primary-contract rules

1. Add `(FUTURES_ROOT_T0, is_primary)` and `(FUTURES_ROOT_T1, is_primary)` as boolean fields. When at least one contract has a bar, exactly one must be True and the other False.

2. Initially, if both contracts have bars, use T0 as primary until the other contract meets the three-day volume-share rule.

3. A non-primary contract becomes primary when its daily volume is **greater than or equal to 65% of the combined daily volume of T0 and T1 for three consecutive trading days**.

   Calculate:

   `total_volume = T0_volume + T1_volume`

   `non_primary_share = non_primary_volume / total_volume`

   A day qualifies when:

   `non_primary_share >= 0.65`

   Keep the current contract primary during the first two qualifying days. Switch on the third qualifying day.

4. A volume share below 65%, missing volume for either contract, or zero combined volume breaks the qualifying streak. A switch requires three new consecutive qualifying trading days. Do not replace missing volume with zero. If a contract has no bar, apply the missing-bar override in Rule 5 instead.

5. If a contract has no bar on a date, it is automatically not primary. If the other contract has a bar, set the other contract's `is_primary` to True immediately, without waiting three days. Reset any qualifying streak.

6. If neither contract has a bar, set both flags to False. Such dates do not provide a volume comparison and do not count toward a streak. Ignore dates with neither bar when counting consecutive trading observations.

7. A bar is absent when all five OHLCV fields are missing. Zero volume is a numeric value, not a missing bar. A partially populated bar should be flagged during validation rather than silently treated as absent.

8. Track the primary contract and qualifying streak by actual contract identity, not just T0/T1: a contract can move from T1 to T0 as the calendar advances. Preserve its primary status if it remains in the pair. Reset the qualifying streak when the compared contract pair changes. If the previous primary is no longer in the pair, initialize the new pair using the rules above.

9. Processing a new year must not itself reset the primary selection or a streak when the same contracts are still being compared.

## Example

Suppose T0 is initially the primary contract.

| Trading day | T0 volume | T1 volume | T1 share | Primary |
|---|---:|---:|---:|---|
| Day 1 | 500,000 | 500,000 | 50% | T0 |
| Day 2 | 330,000 | 670,000 | 67% | T0 |
| Day 3 | 300,000 | 700,000 | 70% | T0 |
| Day 4 | 350,000 | 650,000 | 65% | T1 |
| Day 5 | 250,000 | 750,000 | 75% | T1 |

T1 qualifies on Days 2, 3, and 4. It becomes primary on Day 4.

Once T1 becomes primary, it remains primary unless T0 subsequently satisfies the same three-day rule or a missing-bar override applies.

## Steps

1. Find and pair the OHLCV and symbols files. Skip ZQ entirely.

2. Read each pair, process dates in chronological order, and apply the rules above.

3. Verify that the only dataframe changes are the addition or recomputation of the `is_primary` fields. Preserve all original OHLCV values, dates, tenor assignments, and other existing columns, including any existing `expired` fields. Do not modify the symbols or contract-level files.

4. Verify primary exclusivity, missing-bar overrides, and switching after three consecutive trading days with the non-primary contract holding at least 65% of combined daily volume.

5. Rewrite the original OHLCV file only after these checks succeed, then move to the next asset. Do not create report files.