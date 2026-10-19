from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

MBO_PATH = Path("data/raw/mbo/gld-2026-01-29.mbo.csv")

CHUNK_SIZE = 1_000_000

# ------------------------------------------------------------
# Time sampling
# ------------------------------------------------------------

TIME_FREQUENCY = "5min"

# ------------------------------------------------------------
# Adaptive imbalance sampling
# ------------------------------------------------------------

IMBALANCE_THRESHOLD = 3000
ROLLING_MINUTES = 30

# ------------------------------------------------------------
# Dollar-volume sampling
# ------------------------------------------------------------

DOLLAR_VOLUME_THRESHOLD = 5_000_000.0


# ============================================================
# LOAD TRADES
# ============================================================

def load_trades(
    path: Path,
    chunk_size: int = 1_000_000,
) -> pd.DataFrame:

    pieces = []

    usecols = [
        "ts_event",
        "action",
        "side",
        "price",
        "size",
    ]

    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        chunksize=chunk_size,
    ):

        trades = chunk.loc[
            (chunk["action"] == "T")
            & (chunk["side"].isin(["B", "A"]))
        ].copy()

        if trades.empty:
            continue

        # B = aggressive buy
        # A = aggressive sell
        trades["signed_volume"] = np.where(
            trades["side"] == "B",
            trades["size"],
            -trades["size"],
        )

        trades["dollar_volume"] = (
            trades["price"]
            * trades["size"]
        )

        pieces.append(
            trades[
                [
                    "ts_event",
                    "price",
                    "size",
                    "side",
                    "signed_volume",
                    "dollar_volume",
                ]
            ]
        )

    if not pieces:
        raise ValueError(
            "No classified trade rows found."
        )

    df = pd.concat(
        pieces,
        ignore_index=True,
    )

    df["ts_event"] = pd.to_datetime(
        df["ts_event"],
        utc=True,
        format="mixed",
    )

    df = (
        df.sort_values("ts_event")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# 1-MINUTE SIGNED-VOLUME IMBALANCE
# ============================================================

def build_minute_imbalance(
    trades: pd.DataFrame,
) -> pd.Series:

    return (
        trades
        .set_index("ts_event")["signed_volume"]
        .resample("1min")
        .sum()
    )


# ============================================================
# 1. ADAPTIVE IMBALANCE SAMPLING
#
# Rule:
#
# |S_T - mu_hat * T| >= threshold
#
# S_T:
# cumulative signed aggressive volume since last sample
#
# mu_hat:
# rolling mean of PRIOR completed 1-minute signed-volume bins
#
# T:
# elapsed minutes since last sample
#
# Before 30 minutes of history:
# mu_hat = 0
# ============================================================

def adaptive_imbalance_sampling(
    trades: pd.DataFrame,
    minute_imbalance: pd.Series,
    threshold: float = 3_000,
    rolling_minutes: int = 30,
) -> pd.DataFrame:

    rolling_mean = (
        minute_imbalance
        .rolling(
            window=rolling_minutes,
            min_periods=rolling_minutes,
        )
        .mean()
        .shift(1)
    )

    samples = []

    cumulative_imbalance = 0.0

    first_trade_time = (
        trades["ts_event"].iloc[0]
    )

    last_sample_time = None

    for row in trades.itertuples(index=False):

        ts = row.ts_event

        cumulative_imbalance += float(
            row.signed_volume
        )

        # ----------------------------------------------------
        # Elapsed time
        # ----------------------------------------------------

        if last_sample_time is None:
            reference_time = first_trade_time
        else:
            reference_time = last_sample_time

        elapsed_minutes = (
            ts - reference_time
        ).total_seconds() / 60.0

        elapsed_minutes = max(
            elapsed_minutes,
            0.0,
        )

        # ----------------------------------------------------
        # Rolling mean
        # ----------------------------------------------------

        current_minute = ts.floor("1min")

        mu = rolling_mean.get(
            current_minute,
            np.nan,
        )

        if pd.isna(mu):
            expected_imbalance = 0.0
        else:
            expected_imbalance = (
                mu * elapsed_minutes
            )

        # ----------------------------------------------------
        # Residual flow surprise
        # ----------------------------------------------------

        residual = (
            cumulative_imbalance
            - expected_imbalance
        )

        # ----------------------------------------------------
        # Trigger sample
        # ----------------------------------------------------

        if abs(residual) >= threshold:

            samples.append(
                {
                    "ts_event": ts,
                    "price": float(row.price),

                    "elapsed_minutes":
                        elapsed_minutes,

                    "cumulative_imbalance":
                        cumulative_imbalance,

                    "rolling_mean_1m":
                        mu,

                    "expected_imbalance":
                        expected_imbalance,

                    "residual":
                        residual,
                }
            )

            cumulative_imbalance = 0.0
            last_sample_time = ts

    if not samples:
        raise ValueError(
            "No imbalance samples generated."
        )

    return pd.DataFrame(samples)


# ============================================================
# 2. FIXED DOLLAR-VOLUME SAMPLING
#
# Threshold-crossing trade belongs to current bar.
# Overshoot is discarded.
# ============================================================

def dollar_volume_sampling(
    trades: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:

    samples = []

    cumulative_dollar_volume = 0.0

    for row in trades.itertuples(index=False):

        cumulative_dollar_volume += float(
            row.dollar_volume
        )

        if cumulative_dollar_volume >= threshold:

            samples.append(
                {
                    "ts_event": row.ts_event,
                    "price": float(row.price),
                    "dollar_volume_at_cross":
                        cumulative_dollar_volume,
                }
            )

            cumulative_dollar_volume = 0.0

    if not samples:
        raise ValueError(
            "No dollar-volume samples generated."
        )

    return pd.DataFrame(samples)


# ============================================================
# 3. TRUE CLOCK-TIME SAMPLING
#
# Every 1 minute.
#
# At each timestamp use the most recent classified trade price.
# ============================================================

def time_sampling(
    trades: pd.DataFrame,
    frequency: str = "1min",
) -> pd.DataFrame:

    start_time = (
        trades["ts_event"].min().ceil(frequency)
    )

    end_time = (
        trades["ts_event"].max().floor(frequency)
    )

    time_grid = pd.date_range(
        start=start_time,
        end=end_time,
        freq=frequency,
    )

    grid = pd.DataFrame(
        {
            "sample_time": time_grid,
        }
    )

    trade_prices = (
        trades[
            [
                "ts_event",
                "price",
            ]
        ]
        .sort_values("ts_event")
        .copy()
    )

    samples = pd.merge_asof(
        grid,
        trade_prices,
        left_on="sample_time",
        right_on="ts_event",
        direction="backward",
    )

    samples = (
        samples
        .dropna(subset=["price"])
        .reset_index(drop=True)
    )

    return samples


# ============================================================
# ADD LOG RETURNS
# ============================================================

def add_log_returns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    out["log_return"] = np.log(
        out["price"]
        /
        out["price"].shift(1)
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

    x = returns.dropna()

    mu = x.mean()
    sigma = x.std()

    if sigma > 0:

        z = (
            x - mu
        ) / sigma

        outside_2 = (
            np.abs(z) > 2
        ).mean()

        outside_3 = (
            np.abs(z) > 3
        ).mean()

    else:

        outside_2 = np.nan
        outside_3 = np.nan

    return {
        "N": len(x),

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

        "Outside 2 sigma (%)":
            100 * outside_2,

        "Outside 3 sigma (%)":
            100 * outside_3,
    }


def print_distribution_comparison(
    imbalance_samples: pd.DataFrame,
    dollar_samples: pd.DataFrame,
    time_samples: pd.DataFrame,
) -> None:

    imbalance_stats = distribution_stats(
        imbalance_samples["log_return_bps"]
    )

    dollar_stats = distribution_stats(
        dollar_samples["log_return_bps"]
    )

    time_stats = distribution_stats(
        time_samples["log_return_bps"]
    )

    table = pd.DataFrame(
        {
            "Imbalance sampling":
                imbalance_stats,

            "Dollar-volume sampling":
                dollar_stats,

            "Time sampling":
                time_stats,
        }
    )

    print()
    print("=" * 100)

    print(
        "LOG RETURN DISTRIBUTION COMPARISON "
        "— BASIS POINTS"
    )

    print("=" * 100)

    print(
        table.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )


# ============================================================
# PRICE COMPARISON
# ============================================================

def plot_price_comparison(
    imbalance_samples: pd.DataFrame,
    dollar_samples: pd.DataFrame,
    time_samples: pd.DataFrame,
) -> None:

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9, 6.5),
        sharex=True,
    )

    # --------------------------------------------------------
    # Imbalance
    # --------------------------------------------------------

    axes[0].plot(
        imbalance_samples["ts_event"],
        imbalance_samples["price"],
        linewidth=0.8,
        marker="o",
        markersize=1.4,
    )

    axes[0].set_title(
        "Adaptive Imbalance Sampling"
    )

    axes[0].set_ylabel("Price")

    axes[0].grid(alpha=0.25)

    # --------------------------------------------------------
    # Dollar volume
    # --------------------------------------------------------

    axes[1].plot(
        dollar_samples["ts_event"],
        dollar_samples["price"],
        linewidth=0.8,
        marker="o",
        markersize=1.4,
    )

    axes[1].set_title(
        f"Dollar-Volume Sampling "
        f"(${DOLLAR_VOLUME_THRESHOLD / 1e6:.1f}MM)"
    )

    axes[1].set_ylabel("Price")

    axes[1].grid(alpha=0.25)

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    axes[2].plot(
        time_samples["sample_time"],
        time_samples["price"],
        linewidth=0.8,
        marker="o",
        markersize=1.4,
    )

    axes[2].set_title(
        f"Clock-Time Sampling ({TIME_FREQUENCY})"
    )

    axes[2].set_xlabel("Time")

    axes[2].set_ylabel("Price")

    axes[2].grid(alpha=0.25)

    fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()


# ============================================================
# RETURN HISTOGRAM
# ============================================================

def plot_return_distribution_comparison(
    imbalance_samples: pd.DataFrame,
    dollar_samples: pd.DataFrame,
    time_samples: pd.DataFrame,
    bins: int = 70,
) -> None:

    r_imb = (
        imbalance_samples["log_return_bps"]
        .dropna()
    )

    r_dollar = (
        dollar_samples["log_return_bps"]
        .dropna()
    )

    r_time = (
        time_samples["log_return_bps"]
        .dropna()
    )

    combined = np.concatenate(
        [
            r_imb.to_numpy(),
            r_dollar.to_numpy(),
            r_time.to_numpy(),
        ]
    )

    bin_edges = np.histogram_bin_edges(
        combined,
        bins=bins,
    )

    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    ax.hist(
        r_time,
        bins=bin_edges,
        density=True,
        alpha=0.4,
        label="Clock time",
    )

    ax.hist(
        r_dollar,
        bins=bin_edges,
        density=True,
        alpha=0.4,
        label="Dollar volume",
    )

    ax.hist(
        r_imb,
        bins=bin_edges,
        density=True,
        alpha=0.4,
        label="Imbalance",
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_title(
        "Log Return Distribution"
    )

    ax.set_xlabel(
        "Log Return (bps)"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.legend()

    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()


# ============================================================
# HEAVY-TAIL VIEW
# ============================================================

def plot_return_distribution_log_scale(
    imbalance_samples: pd.DataFrame,
    dollar_samples: pd.DataFrame,
    time_samples: pd.DataFrame,
    bins: int = 70,
) -> None:

    r_imb = (
        imbalance_samples["log_return_bps"]
        .dropna()
    )

    r_dollar = (
        dollar_samples["log_return_bps"]
        .dropna()
    )

    r_time = (
        time_samples["log_return_bps"]
        .dropna()
    )

    combined = np.concatenate(
        [
            r_imb.to_numpy(),
            r_dollar.to_numpy(),
            r_time.to_numpy(),
        ]
    )

    bin_edges = np.histogram_bin_edges(
        combined,
        bins=bins,
    )

    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    ax.hist(
        r_time,
        bins=bin_edges,
        density=True,
        histtype="step",
        linewidth=1.2,
        label="Clock time",
    )

    ax.hist(
        r_dollar,
        bins=bin_edges,
        density=True,
        histtype="step",
        linewidth=1.2,
        label="Dollar volume",
    )

    ax.hist(
        r_imb,
        bins=bin_edges,
        density=True,
        histtype="step",
        linewidth=1.2,
        label="Imbalance",
    )

    ax.set_yscale("log")

    ax.set_title(
        "Return Distribution Tails"
    )

    ax.set_xlabel(
        "Log Return (bps)"
    )

    ax.set_ylabel(
        "Log Density"
    )

    ax.legend()

    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load classified trades
    # --------------------------------------------------------

    trades = load_trades(
        MBO_PATH,
        chunk_size=CHUNK_SIZE,
    )

    print(
        f"Classified trades loaded: "
        f"{len(trades):,}"
    )

    # --------------------------------------------------------
    # Minute imbalance process
    # --------------------------------------------------------

    minute_imbalance = (
        build_minute_imbalance(
            trades
        )
    )

    # --------------------------------------------------------
    # Imbalance sampling
    # --------------------------------------------------------

    imbalance_samples = (
        adaptive_imbalance_sampling(
            trades=trades,
            minute_imbalance=minute_imbalance,
            threshold=IMBALANCE_THRESHOLD,
            rolling_minutes=ROLLING_MINUTES,
        )
    )

    print(
        "Imbalance samples:",
        f"{len(imbalance_samples):,}"
    )

    # --------------------------------------------------------
    # Dollar-volume sampling
    # --------------------------------------------------------

    dollar_samples = (
        dollar_volume_sampling(
            trades=trades,
            threshold=DOLLAR_VOLUME_THRESHOLD,
        )
    )

    print(
        "Dollar-volume samples:",
        f"{len(dollar_samples):,}"
    )

    # --------------------------------------------------------
    # True 1-minute clock sampling
    # --------------------------------------------------------

    time_samples = (
        time_sampling(
            trades=trades,
            frequency=TIME_FREQUENCY,
        )
    )

    print(
        "Time samples:",
        f"{len(time_samples):,}"
    )

    # --------------------------------------------------------
    # Add log returns
    # --------------------------------------------------------

    imbalance_samples = add_log_returns(
        imbalance_samples
    )

    dollar_samples = add_log_returns(
        dollar_samples
    )

    time_samples = add_log_returns(
        time_samples
    )

    # --------------------------------------------------------
    # Print comparison
    # --------------------------------------------------------

    print_distribution_comparison(
        imbalance_samples,
        dollar_samples,
        time_samples,
    )

    # --------------------------------------------------------
    # Price paths
    # --------------------------------------------------------

    plot_price_comparison(
        imbalance_samples,
        dollar_samples,
        time_samples,
    )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    plot_return_distribution_comparison(
        imbalance_samples,
        dollar_samples,
        time_samples,
        bins=70,
    )

    # --------------------------------------------------------
    # Tail comparison
    # --------------------------------------------------------

    plot_return_distribution_log_scale(
        imbalance_samples,
        dollar_samples,
        time_samples,
        bins=70,
    )