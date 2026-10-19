from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

NORMAL_DAY_PATH = Path(
    "data/raw/tbbo/gld-2026-01-06.tbbo.csv"
)

EVENT_DAY_PATH = Path(
    "data/raw/tbbo/gld-2026-01-29.tbbo.csv"
)

CHUNK_SIZE = 1_000_000

# Normal-day reference interval
BASE_INTERVAL_MINUTES = 5.0

# Event-day sampler cannot run more than 10x faster
MIN_TIME_MULTIPLIER = 0.05

MIN_SAMPLE_SECONDS = (
    BASE_INTERVAL_MINUTES
    * 60.0
    * MIN_TIME_MULTIPLIER
)

# Clock-time benchmark
TIME_FREQUENCY = "5min"


# ============================================================
# COLUMN HELPERS
# ============================================================

def get_tbbo_columns(path: Path):

    cols = pd.read_csv(
        path,
        nrows=0,
    ).columns.tolist()

    required = [
        "ts_event",
        "price",
        "size",
        "bid_px_00",
        "ask_px_00",
    ]

    missing = [
        col for col in required
        if col not in cols
    ]

    if missing:
        raise ValueError(
            f"Missing required TBBO columns: {missing}"
        )

    usecols = required.copy()

    # Some Databento CSVs contain action.
    # If it exists, use only actual trade rows.
    if "action" in cols:
        usecols.append("action")

    return usecols


# ============================================================
# FILTER / CLEAN A TBBO CHUNK
# ============================================================

def clean_tbbo_chunk(
    chunk: pd.DataFrame,
) -> pd.DataFrame:

    # --------------------------------------------------------
    # Keep actual trades if action exists
    # --------------------------------------------------------

    if "action" in chunk.columns:

        chunk = chunk.loc[
            chunk["action"] == "T"
        ].copy()

    else:

        chunk = chunk.copy()

    if chunk.empty:
        return chunk

    # --------------------------------------------------------
    # Parse timestamp
    # --------------------------------------------------------

    chunk["ts_event"] = pd.to_datetime(
        chunk["ts_event"],
        utc=True,
        format="mixed",
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_cols = [
        "price",
        "size",
        "bid_px_00",
        "ask_px_00",
    ]

    for col in numeric_cols:

        chunk[col] = pd.to_numeric(
            chunk[col],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    chunk = chunk.dropna(
        subset=[
            "ts_event",
            "price",
            "size",
            "bid_px_00",
            "ask_px_00",
        ]
    )

    chunk = chunk.loc[
        (chunk["price"] > 0)
        & (chunk["size"] > 0)
        & (chunk["bid_px_00"] > 0)
        & (chunk["ask_px_00"] > 0)
        & (
            chunk["ask_px_00"]
            >= chunk["bid_px_00"]
        )
    ].copy()

    if chunk.empty:
        return chunk

    # --------------------------------------------------------
    # Trade notional
    # --------------------------------------------------------

    chunk["notional"] = (
        chunk["price"]
        * chunk["size"]
    )

    # --------------------------------------------------------
    # Mid price directly from TBBO
    # --------------------------------------------------------

    chunk["mid"] = (
        chunk["bid_px_00"]
        + chunk["ask_px_00"]
    ) / 2.0

    return chunk


# ============================================================
# NORMAL DAY CALIBRATION
#
# Stream Jan 6.
#
# Find:
#   total traded notional
#   first timestamp
#   last timestamp
#
# Then:
#
# threshold =
# total notional / number of 5-minute intervals
# ============================================================

def calibrate_normal_day(
    path: Path,
    chunk_size: int,
    interval_minutes: float = 5.0,
):

    usecols = get_tbbo_columns(
        path
    )

    total_notional = 0.0

    first_ts = None
    last_ts = None

    total_trades = 0

    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        chunksize=chunk_size,
    ):

        chunk = clean_tbbo_chunk(
            chunk
        )

        if chunk.empty:
            continue

        total_notional += (
            chunk["notional"].sum()
        )

        total_trades += len(
            chunk
        )

        chunk_first = (
            chunk["ts_event"].min()
        )

        chunk_last = (
            chunk["ts_event"].max()
        )

        if (
            first_ts is None
            or chunk_first < first_ts
        ):
            first_ts = chunk_first

        if (
            last_ts is None
            or chunk_last > last_ts
        ):
            last_ts = chunk_last

    if first_ts is None:
        raise ValueError(
            "No valid trades found on normal day."
        )

    # --------------------------------------------------------
    # Number of 5-minute intervals covering the observed day
    # --------------------------------------------------------

    interval_seconds = (
        interval_minutes * 60.0
    )

    span_seconds = (
        last_ts - first_ts
    ).total_seconds()

    n_intervals = max(
        1,
        int(
            np.ceil(
                span_seconds
                / interval_seconds
            )
        ),
    )

    threshold = (
        total_notional
        / n_intervals
    )

    return {
        "total_trades":
            total_trades,

        "total_notional":
            total_notional,

        "first_ts":
            first_ts,

        "last_ts":
            last_ts,

        "n_intervals":
            n_intervals,

        "threshold":
            threshold,
    }


# ============================================================
# EVENT-DAY STREAMING
#
# Produce BOTH:
#
# 1. Dollar-notional samples
# 2. Genuine 5-minute clock samples
#
# without loading the full event-day CSV.
# ============================================================

def sample_event_day(
    path: Path,
    chunk_size: int,
    notional_threshold: float,
    min_sample_seconds: float,
    time_frequency: str = "5min",
):

    usecols = get_tbbo_columns(
        path
    )

    # --------------------------------------------------------
    # Dollar-notional sampler state
    # --------------------------------------------------------

    notional_samples = []

    cumulative_notional = 0.0

    last_notional_sample_ts = None

    # --------------------------------------------------------
    # Time sampler state
    # --------------------------------------------------------

    time_samples = []

    next_time_sample = None

    latest_mid = np.nan
    latest_mid_ts = None

    # ========================================================
    # STREAM FILE
    # ========================================================

    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        chunksize=chunk_size,
    ):

        chunk = clean_tbbo_chunk(
            chunk
        )

        if chunk.empty:
            continue

        chunk = chunk.sort_values(
            "ts_event",
            kind="stable",
        )

        for row in chunk.itertuples(
            index=False
        ):

            ts = row.ts_event
            mid = float(row.mid)
            notional = float(row.notional)

            # =================================================
            # CLOCK-TIME SAMPLING
            #
            # Before replacing latest_mid with current row,
            # fill any scheduled timestamps that occurred
            # between the previous row and this trade.
            # =================================================

            if next_time_sample is None:

                next_time_sample = (
                    ts.ceil(
                        time_frequency
                    )
                )

            while (
                latest_mid_ts is not None
                and next_time_sample <= ts
            ):

                time_samples.append(
                    {
                        "sample_time":
                            next_time_sample,

                        "mid":
                            latest_mid,
                    }
                )

                next_time_sample += (
                    pd.Timedelta(
                        time_frequency
                    )
                )

            # Current TBBO observation becomes latest known mid
            latest_mid = mid
            latest_mid_ts = ts

            # =================================================
            # DOLLAR-NOTIONAL SAMPLING
            # =================================================

            cumulative_notional += (
                notional
            )

            # -------------------------------------------------
            # First bar:
            # only notional threshold matters.
            # -------------------------------------------------

            if last_notional_sample_ts is None:

                if (
                    cumulative_notional
                    >= notional_threshold
                ):

                    notional_samples.append(
                        {
                            "ts_event":
                                ts,

                            "mid":
                                mid,

                            "notional":
                                cumulative_notional,

                            "elapsed_seconds":
                                np.nan,
                        }
                    )

                    cumulative_notional = 0.0

                    last_notional_sample_ts = (
                        ts
                    )

                continue

            # -------------------------------------------------
            # Subsequent bars:
            #
            # Need BOTH:
            #
            # cumulative notional >= threshold
            #
            # AND
            #
            # elapsed time >= 30 seconds
            # -------------------------------------------------

            elapsed_seconds = (
                ts
                - last_notional_sample_ts
            ).total_seconds()

            if (
                cumulative_notional
                >= notional_threshold
                and
                elapsed_seconds
                >= min_sample_seconds
            ):

                notional_samples.append(
                    {
                        "ts_event":
                            ts,

                        "mid":
                            mid,

                        "notional":
                            cumulative_notional,

                        "elapsed_seconds":
                            elapsed_seconds,
                    }
                )

                # ---------------------------------------------
                # Reset bar.
                # Overshoot discarded.
                # ---------------------------------------------

                cumulative_notional = 0.0

                last_notional_sample_ts = (
                    ts
                )

    # --------------------------------------------------------
    # Finish remaining clock timestamps up to last trade
    # --------------------------------------------------------

    if (
        latest_mid_ts is not None
        and next_time_sample is not None
    ):

        end_time = (
            latest_mid_ts.floor(
                time_frequency
            )
        )

        while (
            next_time_sample
            <= end_time
        ):

            time_samples.append(
                {
                    "sample_time":
                        next_time_sample,

                    "mid":
                        latest_mid,
                }
            )

            next_time_sample += (
                pd.Timedelta(
                    time_frequency
                )
            )

    notional_df = pd.DataFrame(
        notional_samples
    )

    time_df = pd.DataFrame(
        time_samples
    )

    if notional_df.empty:
        raise ValueError(
            "No notional samples generated."
        )

    if time_df.empty:
        raise ValueError(
            "No time samples generated."
        )

    return (
        notional_df,
        time_df,
    )


# ============================================================
# MID-PRICE LOG RETURNS
# ============================================================

def add_log_returns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    out["log_return"] = np.log(
        out["mid"]
        / out["mid"].shift(1)
    )

    out["log_return_bps"] = (
        10_000
        * out["log_return"]
    )

    return out


# ============================================================
# DISTRIBUTION STATISTICS
# ============================================================

def distribution_stats(
    returns: pd.Series,
) -> dict:

    x = (
        returns
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    return {
        "N":
            len(x),

        "Mean":
            x.mean(),

        "Median":
            x.median(),

        "Std":
            x.std(),

        "Mean |r|":
            x.abs().mean(),

        "Median |r|":
            x.abs().median(),

        "Min":
            x.min(),

        "1%":
            x.quantile(0.01),

        "5%":
            x.quantile(0.05),

        "25%":
            x.quantile(0.25),

        "75%":
            x.quantile(0.75),

        "95%":
            x.quantile(0.95),

        "99%":
            x.quantile(0.99),

        "Max":
            x.max(),

        "Skew":
            x.skew(),

        "Excess kurtosis":
            x.kurt(),
    }


def print_distribution_results(
    notional_samples: pd.DataFrame,
    time_samples: pd.DataFrame,
):

    table = pd.DataFrame(
        {
            "Dollar-notional":
                distribution_stats(
                    notional_samples[
                        "log_return_bps"
                    ]
                ),

            "5-minute time":
                distribution_stats(
                    time_samples[
                        "log_return_bps"
                    ]
                ),
        }
    )

    print()
    print("=" * 90)

    print(
        "MID-PRICE LOG RETURN DISTRIBUTION — BASIS POINTS"
    )

    print("=" * 90)

    print(
        table.to_string(
            float_format=lambda x:
                f"{x:.4f}"
        )
    )


# ============================================================
# ONLY PLOT REQUESTED:
# MID PRICE AGAINST TIME
# ============================================================

def plot_mid_price(
    notional_samples: pd.DataFrame,
    time_samples: pd.DataFrame,
):

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True,
    )

    # --------------------------------------------------------
    # Dollar-notional sampling
    # --------------------------------------------------------

    axes[0].plot(
        notional_samples[
            "ts_event"
        ],
        notional_samples[
            "mid"
        ],
        linewidth=0.8,
        marker="o",
        markersize=1.5,
    )

    axes[0].set_title(
        "Dollar-Notional Sampling — Mid Price"
    )

    axes[0].set_ylabel(
        "Mid Price"
    )

    axes[0].grid(
        alpha=0.25
    )

    # --------------------------------------------------------
    # 5-minute time sampling
    # --------------------------------------------------------

    axes[1].plot(
        time_samples[
            "sample_time"
        ],
        time_samples[
            "mid"
        ],
        linewidth=0.8,
        marker="o",
        markersize=1.5,
    )

    axes[1].set_title(
        "5-Minute Time Sampling — Mid Price"
    )

    axes[1].set_xlabel(
        "Time"
    )

    axes[1].set_ylabel(
        "Mid Price"
    )

    axes[1].grid(
        alpha=0.25
    )

    fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # 1. CALIBRATE FROM JAN 6
    # ========================================================

    calibration = calibrate_normal_day(
        path=
            NORMAL_DAY_PATH,

        chunk_size=
            CHUNK_SIZE,

        interval_minutes=
            BASE_INTERVAL_MINUTES,
    )

    NOTIONAL_THRESHOLD = (
        calibration["threshold"]
    )

    print()
    print("=" * 70)
    print("JAN 6 NORMAL-DAY CALIBRATION")
    print("=" * 70)

    print(
        f"Trades: "
        f"{calibration['total_trades']:,}"
    )

    print(
        f"Total traded notional: "
        f"${calibration['total_notional']:,.2f}"
    )

    print(
        f"Number of 5-minute intervals: "
        f"{calibration['n_intervals']:,}"
    )

    print(
        f"Dollar-notional threshold: "
        f"${NOTIONAL_THRESHOLD:,.2f}"
    )

    print(
        f"Minimum event-day bar duration: "
        f"{MIN_SAMPLE_SECONDS:.2f} seconds"
    )

    # ========================================================
    # 2. STREAM JAN 29 AND SAMPLE
    # ========================================================

    (
        notional_samples,
        time_samples,
    ) = sample_event_day(
        path=
            EVENT_DAY_PATH,

        chunk_size=
            CHUNK_SIZE,

        notional_threshold=
            NOTIONAL_THRESHOLD,

        min_sample_seconds=
            MIN_SAMPLE_SECONDS,

        time_frequency=
            TIME_FREQUENCY,
    )

    print()

    print(
        f"Dollar-notional samples: "
        f"{len(notional_samples):,}"
    )

    print(
        f"5-minute time samples: "
        f"{len(time_samples):,}"
    )

    # ========================================================
    # 3. MID-PRICE RETURNS
    # ========================================================

    notional_samples = (
        add_log_returns(
            notional_samples
        )
    )

    time_samples = (
        add_log_returns(
            time_samples
        )
    )

    # ========================================================
    # 4. PRINT DISTRIBUTION RESULTS
    # ========================================================

    print_distribution_results(
        notional_samples,
        time_samples,
    )

    # ========================================================
    # 5. ONLY PLOT: MID PRICE
    # ========================================================

    plot_mid_price(
        notional_samples,
        time_samples,
    )