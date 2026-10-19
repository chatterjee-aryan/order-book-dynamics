import pandas as pd

DATA_PATH = "data/raw/statistics/gld-2026-01-06.statistics.csv"

# Load data
df = pd.read_csv(DATA_PATH)

# Parse and sort by event time
df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
df = df.sort_values("ts_event").reset_index(drop=True)

# Extract cleared volume statistics
daily_volume_rows = df.loc[
    df["stat_type"] == 6,
    ["ts_event", "ts_ref", "quantity", "update_action", "symbol"],
]

print(daily_volume_rows)

# Final reported daily volume
if not daily_volume_rows.empty:
    total_volume = daily_volume_rows.iloc[-1]["quantity"]
    print(f"\nTotal daily volume: {total_volume:,.0f}")
else:
    print("\nNo cleared-volume statistic found.")