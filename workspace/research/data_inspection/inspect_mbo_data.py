import pandas as pd

MBO_PATH = "data/raw/mbo/gld-2026-01-29.mbo.csv"

first_ts_event = None
last_ts_event = None

first_ts_recv = None
last_ts_recv = None

n_rows = 0

for chunk in pd.read_csv(
    MBO_PATH,
    usecols=[
        "ts_event",
        "ts_recv",
    ],
    chunksize=500_000,
):

    chunk["ts_event"] = pd.to_datetime(
        chunk["ts_event"],
        utc=True,
    )

    chunk["ts_recv"] = pd.to_datetime(
        chunk["ts_recv"],
        utc=True,
    )

    if first_ts_event is None:
        first_ts_event = chunk["ts_event"].iloc[0]
        first_ts_recv = chunk["ts_recv"].iloc[0]

    last_ts_event = chunk["ts_event"].iloc[-1]
    last_ts_recv = chunk["ts_recv"].iloc[-1]

    n_rows += len(chunk)


print(f"MBO rows: {n_rows:,}")

print()
print("File-order timestamps:")
print(f"First ts_event: {first_ts_event}")
print(f"Last ts_event:  {last_ts_event}")
print(f"First ts_recv:  {first_ts_recv}")
print(f"Last ts_recv:   {last_ts_recv}")