import pandas as pd

DATA_PATH = "data/raw/tbbo/gld-2026-01-06.tbbo.csv"

# Load data
df = pd.read_csv(DATA_PATH)

# Parse event time as UTC
df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)

# Convert to New York time
df["ts_event_ny"] = df["ts_event"].dt.tz_convert("America/New_York")

# Sort chronologically
df = df.sort_values("ts_event").reset_index(drop=True)

# Keep regular trading hours only: 09:30 to 16:00 ET
rth = df[
    (df["ts_event_ny"].dt.time >= pd.Timestamp("09:30").time())
    & (df["ts_event_ny"].dt.time < pd.Timestamp("16:00").time())
].copy()

# Keep trades only
trades = rth[rth["action"] == "T"].copy()

# Dollar volume = trade price × trade size
trades["dollar_volume"] = trades["price"] * trades["size"]

# Total regular-hours dollar volume
total_dollar_volume = trades["dollar_volume"].sum()

# US regular trading session = 390 minutes
RTH_MINUTES = 390

# Reference 1-minute dollar volume
dollar_volume_per_minute = total_dollar_volume / RTH_MINUTES

print(f"RTH first event: {rth['ts_event_ny'].iloc[0]}")
print(f"RTH last event:  {rth['ts_event_ny'].iloc[-1]}")

print(f"\nTotal RTH dollar volume: ${total_dollar_volume:,.2f}")
print(
    f"Average RTH dollar volume per minute: "
    f"${dollar_volume_per_minute:,.2f}"
)