from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MBO_PATH = Path("data/raw/mbo/gld-2026-01-06.mbo.csv")
CHUNK_SIZE = 1_000_000

N_MINUTES = 5


def load_cumulative_volume_imbalance(
    path: Path,
    chunk_size: int = 1_000_000,
) -> pd.DataFrame:

    pieces = []
    cumulative_imbalance = 0

    usecols = [
        "ts_event",
        "action",
        "side",
        "size",
    ]

    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        chunksize=chunk_size,
    ):
        trades = chunk.loc[chunk["action"] == "T"].copy()

        trades = trades.loc[
            trades["side"].isin(["B", "A"])
        ].copy()

        if trades.empty:
            continue

        trades["signed_volume"] = np.where(
            trades["side"] == "B",
            trades["size"],
            -trades["size"],
        )

        trades["cum_volume_imbalance"] = (
            trades["signed_volume"].cumsum()
            + cumulative_imbalance
        )

        cumulative_imbalance = (
            trades["cum_volume_imbalance"].iloc[-1]
        )

        pieces.append(
            trades[
                [
                    "ts_event",
                    "side",
                    "size",
                    "signed_volume",
                    "cum_volume_imbalance",
                ]
            ]
        )

    if not pieces:
        raise ValueError("No signed trade records found.")

    df = pd.concat(
        pieces,
        ignore_index=True,
    )

    df["ts_event"] = pd.to_datetime(
        df["ts_event"],
        utc=True,
        format="mixed",
    )

    df = df.sort_values("ts_event").reset_index(drop=True)

    return df


def compute_n_minute_imbalance(
    df: pd.DataFrame,
    n_minutes: int,
) -> pd.DataFrame:

    x = (
        df.set_index("ts_event")["signed_volume"]
        .resample(f"{n_minutes}min")
        .sum()
        .rename("imbalance_change")
        .to_frame()
    )

    x["cum_volume_imbalance"] = (
        x["imbalance_change"].cumsum()
    )

    return x


def print_distribution_stats(
    imbalance_df: pd.DataFrame,
    n_minutes: int,
) -> None:

    x = imbalance_df["imbalance_change"].dropna()

    print(f"\n{n_minutes}-minute imbalance distribution")
    print("-" * 50)

    print(f"N intervals:     {len(x)}")
    print(f"Mean:            {x.mean():,.2f}")
    print(f"Median:          {x.median():,.2f}")
    print(f"Std:             {x.std():,.2f}")
    print(f"Min:             {x.min():,.2f}")
    print(f"Max:             {x.max():,.2f}")

    print()
    print("Quantiles:")
    print(
        x.quantile(
            [
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
    )

    print()
    print(
        "P(imbalance > 0):",
        (x > 0).mean(),
    )

    print(
        "P(imbalance < 0):",
        (x < 0).mean(),
    )


def plot_imbalance_distribution(
    imbalance_df: pd.DataFrame,
    n_minutes: int,
) -> None:

    x = imbalance_df["imbalance_change"].dropna()

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.hist(
        x,
        bins=80,
        alpha=0.8,
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_title(
        f"Distribution of {n_minutes}-Minute "
        "Aggressive Volume Imbalance"
    )

    ax.set_xlabel(
        "Buy Volume - Sell Volume"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.grid(
        alpha=0.25
    )

    plt.tight_layout()
    plt.show()


def plot_imbalance_over_time(
    imbalance_df: pd.DataFrame,
    n_minutes: int,
) -> None:

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.plot(
        imbalance_df.index,
        imbalance_df["imbalance_change"],
        linewidth=1,
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_title(
        f"{n_minutes}-Minute Aggressive Volume Imbalance"
    )

    ax.set_xlabel(
        "Time"
    )

    ax.set_ylabel(
        "Buy Volume - Sell Volume"
    )

    ax.grid(
        alpha=0.25
    )

    fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    trades = load_cumulative_volume_imbalance(
        MBO_PATH,
        CHUNK_SIZE,
    )

    imbalance = compute_n_minute_imbalance(
        trades,
        N_MINUTES,
    )

    print_distribution_stats(
        imbalance,
        N_MINUTES,
    )

    plot_imbalance_distribution(
        imbalance,
        N_MINUTES,
    )

    plot_imbalance_over_time(
        imbalance,
        N_MINUTES,
    )