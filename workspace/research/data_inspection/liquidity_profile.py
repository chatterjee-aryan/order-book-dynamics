from __future__ import annotations

import math
from pathlib import Path

import databento as db
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from workspace.research.order_book.orderbook import Book


# ============================================================
# CONFIG
# ============================================================

MBO_PATH = Path("data/raw/mbo/gld-2026-01-29.mbo.csv")

DOLLAR_VOLUME_THRESHOLD = 1_000_000.0
MBO_CHUNK_SIZE = 1_000_000

# Local book window used for the volume-weighted bid/ask.
MAX_DISTANCE_DOLLARS = 1.00
MAX_DISTANCE_FIXED = int(
    round(
        MAX_DISTANCE_DOLLARS
        * db.FIXED_PRICE_SCALE
    )
)

# Existing convention: deciles.
N_SIGNAL_BINS = 10
N_DURATION_BINS = 10

# Future horizons measured in dollar-volume bars.
HORIZONS = [1, 5, 10, 25, 50, 100]


# ============================================================
# LIGHTWEIGHT MBO MESSAGE
# ============================================================

class FastMBOMsg:
    __slots__ = (
        "instrument_id",
        "publisher_id",
        "order_id",
        "price",
        "size",
        "flags",
        "action",
        "side",
        "sequence",
    )

    def __init__(
        self,
        instrument_id: int,
        publisher_id: int,
        order_id: int,
        price: int,
        size: int,
        flags: int,
        action: str,
        side: str,
        sequence: int,
    ) -> None:
        self.instrument_id = instrument_id
        self.publisher_id = publisher_id
        self.order_id = order_id
        self.price = price
        self.size = size
        self.flags = flags
        self.action = action
        self.side = side
        self.sequence = sequence


def to_fixed_price(value) -> int:
    if pd.isna(value):
        return db.UNDEF_PRICE

    return int(
        round(
            float(value)
            * db.FIXED_PRICE_SCALE
        )
    )


def from_fixed_price(
    value: int,
) -> float:
    return (
        value
        / db.FIXED_PRICE_SCALE
    )


def make_mbo(
    row,
) -> FastMBOMsg:
    return FastMBOMsg(
        instrument_id=int(
            row.instrument_id
        ),
        publisher_id=int(
            row.publisher_id
        ),
        order_id=int(
            row.order_id
        ),
        price=to_fixed_price(
            row.price
        ),
        size=int(
            row.size
        ),
        flags=int(
            row.flags
        ),
        action=row.action,
        side=row.side,
        sequence=int(
            row.sequence
        ),
    )


# ============================================================
# BUILD DOLLAR-VOLUME BAR BOUNDARIES FROM MBO
#
# Only action == "T" contributes to traded volume.
# F is ignored so passive-side fills are not double counted.
#
# Existing convention preserved:
# - full threshold-crossing trade belongs to current bar
# - accumulator then resets to zero
# - no overshoot carry
# ============================================================

BOUNDARY_USECOLS = [
    "ts_event",
    "instrument_id",
    "publisher_id",
    "action",
    "price",
    "size",
    "sequence",
]


def build_volume_boundaries_from_mbo(
    path: Path,
) -> tuple[
    list[dict],
    int,
    int,
]:
    boundaries: list[dict] = []

    bar_dollar_volume = 0.0

    target_instrument = None
    target_publisher = None

    for chunk in pd.read_csv(
        path,
        usecols=BOUNDARY_USECOLS,
        chunksize=MBO_CHUNK_SIZE,
        engine="c",
        memory_map=True,
    ):
        trades = chunk.loc[
            chunk["action"].eq("T")
        ]

        if trades.empty:
            continue

        for row in trades.itertuples(
            index=False
        ):
            instrument_id = int(
                row.instrument_id
            )

            publisher_id = int(
                row.publisher_id
            )

            if target_instrument is None:
                target_instrument = (
                    instrument_id
                )
                target_publisher = (
                    publisher_id
                )

            elif (
                instrument_id
                != target_instrument
                or publisher_id
                != target_publisher
            ):
                raise ValueError(
                    "Multiple instrument/publisher "
                    "pairs found in MBO trade data."
                )

            price = float(
                row.price
            )

            size = int(
                row.size
            )

            if (
                not math.isfinite(price)
                or size <= 0
            ):
                continue

            bar_dollar_volume += (
                price
                * size
            )

            if (
                bar_dollar_volume
                >= DOLLAR_VOLUME_THRESHOLD
            ):
                boundaries.append(
                    {
                        "end_sequence":
                            int(
                                row.sequence
                            ),
                        "ts_event":
                            pd.Timestamp(
                                row.ts_event
                            ),
                        "dollar_volume":
                            float(
                                bar_dollar_volume
                            ),
                    }
                )

                bar_dollar_volume = 0.0

    if not boundaries:
        raise RuntimeError(
            "No completed dollar-volume bars "
            "found in MBO trades."
        )

    if (
        target_instrument is None
        or target_publisher is None
    ):
        raise RuntimeError(
            "Could not infer instrument/publisher "
            "from MBO trades."
        )

    return (
        boundaries,
        int(target_instrument),
        int(target_publisher),
    )


# ============================================================
# FAST LOCAL BOOK STATISTICS
# ============================================================

def get_level_size(
    level_orders,
) -> int:
    return sum(
        int(order.size)
        for order
        in level_orders.orders
    )


def get_local_side_stats(
    book: Book,
    side: str,
) -> tuple[
    float,
    int,
]:
    """
    Return the volume-weighted price and total visible
    displayed depth within +/- MAX_DISTANCE_DOLLARS
    of the touch.
    """

    if side == "B":
        levels = book.bids

        if not levels:
            return (
                math.nan,
                0,
            )

        best_price = (
            levels.peekitem(-1)[0]
        )

        minimum_price = (
            best_price
            - MAX_DISTANCE_FIXED
        )

        price_iterator = (
            levels.irange(
                minimum=minimum_price,
                maximum=best_price,
                reverse=True,
            )
        )

    elif side == "A":
        levels = book.offers

        if not levels:
            return (
                math.nan,
                0,
            )

        best_price = (
            levels.peekitem(0)[0]
        )

        maximum_price = (
            best_price
            + MAX_DISTANCE_FIXED
        )

        price_iterator = (
            levels.irange(
                minimum=best_price,
                maximum=maximum_price,
            )
        )

    else:
        raise ValueError(
            "side must be 'B' or 'A'"
        )

    weighted_price_sum = 0
    total_size = 0

    for price in price_iterator:
        size = get_level_size(
            levels[
                price
            ]
        )

        if size <= 0:
            continue

        weighted_price_sum += (
            int(price)
            * size
        )

        total_size += size

    if total_size <= 0:
        return (
            math.nan,
            0,
        )

    vw_price = (
        weighted_price_sum
        / total_size
        / db.FIXED_PRICE_SCALE
    )

    return (
        float(vw_price),
        int(total_size),
    )


# ============================================================
# BOOK METRICS
# ============================================================

def empty_metrics() -> dict:
    return {
        "bid": math.nan,
        "ask": math.nan,
        "bid_size": math.nan,
        "ask_size": math.nan,
        "spread": math.nan,
        "microprice": math.nan,
        "vw_bid": math.nan,
        "vw_ask": math.nan,
        "local_bid_depth": math.nan,
        "local_ask_depth": math.nan,
        "vw_spread": math.nan,
        "vw_microprice": math.nan,
        "microprice_displacement": math.nan,
    }


def capture_metrics(
    book: Book,
) -> dict:
    bid_level, ask_level = (
        book.bbo()
    )

    if (
        bid_level is None
        or ask_level is None
    ):
        return empty_metrics()

    bid = from_fixed_price(
        bid_level.price
    )

    ask = from_fixed_price(
        ask_level.price
    )

    bid_size = int(
        bid_level.size
    )

    ask_size = int(
        ask_level.size
    )

    spread = (
        ask
        - bid
    )

    top_depth = (
        bid_size
        + ask_size
    )

    if top_depth > 0:
        microprice = (
            (
                ask
                * bid_size
            )
            +
            (
                bid
                * ask_size
            )
        ) / top_depth

    else:
        microprice = (
            math.nan
        )

    (
        vw_bid,
        local_bid_depth,
    ) = get_local_side_stats(
        book,
        "B",
    )

    (
        vw_ask,
        local_ask_depth,
    ) = get_local_side_stats(
        book,
        "A",
    )

    if (
        math.isfinite(vw_bid)
        and math.isfinite(vw_ask)
    ):
        vw_spread = (
            vw_ask
            - vw_bid
        )

    else:
        vw_spread = (
            math.nan
        )

    local_total_depth = (
        local_bid_depth
        + local_ask_depth
    )

    if (
        local_total_depth > 0
        and math.isfinite(vw_bid)
        and math.isfinite(vw_ask)
    ):
        vw_microprice = (
            (
                vw_ask
                * local_bid_depth
            )
            +
            (
                vw_bid
                * local_ask_depth
            )
        ) / local_total_depth

    else:
        vw_microprice = (
            math.nan
        )

    if (
        math.isfinite(microprice)
        and math.isfinite(
            vw_microprice
        )
    ):
        microprice_displacement = (
            vw_microprice
            - microprice
        )

    else:
        microprice_displacement = (
            math.nan
        )

    return {
        "bid":
            bid,

        "ask":
            ask,

        "bid_size":
            bid_size,

        "ask_size":
            ask_size,

        "spread":
            spread,

        "microprice":
            microprice,

        "vw_bid":
            vw_bid,

        "vw_ask":
            vw_ask,

        "local_bid_depth":
            local_bid_depth,

        "local_ask_depth":
            local_ask_depth,

        "vw_spread":
            vw_spread,

        "vw_microprice":
            vw_microprice,

        "microprice_displacement":
            microprice_displacement,
    }


# ============================================================
# REPLAY MBO TO DOLLAR-VOLUME BAR ENDPOINTS
#
# The book is sampled immediately BEFORE the
# threshold-crossing sequence, preserving the existing
# project convention.
#
# Only A/C/M/R mutate the actual book.
# T/F/N do not mutate it.
# ============================================================

REPLAY_USECOLS = [
    "instrument_id",
    "publisher_id",
    "order_id",
    "price",
    "size",
    "flags",
    "action",
    "side",
    "sequence",
]


def replay_to_boundaries(
    path: Path,
    boundaries: list[dict],
    target_instrument: int,
    target_publisher: int,
) -> pd.DataFrame:
    book = Book()
    rows: list[dict] = []

    bar_idx = 0
    previous_sequence = None

    print(
        "Streaming MBO..."
    )

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            usecols=REPLAY_USECOLS,
            chunksize=MBO_CHUNK_SIZE,
            engine="c",
            memory_map=True,
        ),
        start=1,
    ):
        chunk = chunk.loc[
            chunk[
                "instrument_id"
            ].eq(
                target_instrument
            )
            &
            chunk[
                "publisher_id"
            ].eq(
                target_publisher
            )
        ]

        if chunk.empty:
            continue

        # Preserve native MBO file order.
        # Never sort.
        for row in chunk.itertuples(
            index=False
        ):
            if (
                bar_idx
                >= len(boundaries)
            ):
                break

            sequence = int(
                row.sequence
            )

            is_new_sequence = (
                previous_sequence
                is None
                or sequence
                != previous_sequence
            )

            if is_new_sequence:
                while (
                    bar_idx
                    < len(boundaries)
                    and sequence
                    >= boundaries[
                        bar_idx
                    ][
                        "end_sequence"
                    ]
                ):
                    expected_sequence = (
                        boundaries[
                            bar_idx
                        ][
                            "end_sequence"
                        ]
                    )

                    if (
                        sequence
                        > expected_sequence
                    ):
                        raise RuntimeError(
                            "Missed MBO boundary "
                            f"sequence {expected_sequence} "
                            f"for bar {bar_idx}; "
                            f"current sequence={sequence}."
                        )

                    metrics = (
                        capture_metrics(
                            book
                        )
                    )

                    metrics.update(
                        {
                            "bar":
                                bar_idx,

                            "ts_event":
                                boundaries[
                                    bar_idx
                                ][
                                    "ts_event"
                                ],

                            "dollar_volume":
                                boundaries[
                                    bar_idx
                                ][
                                    "dollar_volume"
                                ],
                        }
                    )

                    rows.append(
                        metrics
                    )

                    bar_idx += 1

                previous_sequence = (
                    sequence
                )

            action = row.action

            if action in (
                "A",
                "C",
                "M",
                "R",
            ):
                book.apply(
                    make_mbo(
                        row
                    )
                )

        print(
            f"Chunk {chunk_number} | "
            f"bars "
            f"{bar_idx:,}/"
            f"{len(boundaries):,}"
        )

        if (
            bar_idx
            >= len(boundaries)
        ):
            break

    results = pd.DataFrame(
        rows
    )

    if results.empty:
        raise RuntimeError(
            "No matched dollar-volume bars."
        )

    return results


# ============================================================
# DERIVED FEATURES
# ============================================================

def add_derived_features(
    results: pd.DataFrame,
) -> pd.DataFrame:
    results = (
        results.copy()
    )

    results[
        "ts_event"
    ] = pd.to_datetime(
        results[
            "ts_event"
        ],
        utc=True,
    )

    results[
        "cum_dollar_volume_mm"
    ] = (
        results[
            "dollar_volume"
        ]
        .cumsum()
        / 1_000_000
    )

    results[
        "spread_difference"
    ] = (
        results[
            "vw_spread"
        ]
        -
        results[
            "spread"
        ]
    )

    results[
        "microprice_displacement_bps"
    ] = (
        (
            results[
                "vw_microprice"
            ]
            -
            results[
                "microprice"
            ]
        )
        /
        results[
            "microprice"
        ]
        *
        10_000
    )

    results[
        "D_bps"
    ] = (
        results[
            "microprice_displacement_bps"
        ]
    )

    results[
        "delta_D_bps"
    ] = (
        results[
            "D_bps"
        ]
        .diff()
    )

    # Bar duration / trading intensity.
    results[
        "bar_duration_seconds"
    ] = (
        results[
            "ts_event"
        ]
        .diff()
        .dt.total_seconds()
    )

    results.loc[
        results[
            "bar_duration_seconds"
        ]
        <= 0,
        "bar_duration_seconds",
    ] = np.nan

    results[
        "dollar_intensity_mm_per_sec"
    ] = (
        results[
            "dollar_volume"
        ]
        / 1_000_000
        /
        results[
            "bar_duration_seconds"
        ]
    )

    # Position of traditional microprice within BBO.
    positive_spread = (
        results[
            "spread"
        ]
        > 0
    )

    results[
        "microprice_location"
    ] = np.nan

    results.loc[
        positive_spread,
        "microprice_location",
    ] = (
        (
            results.loc[
                positive_spread,
                "microprice",
            ]
            -
            results.loc[
                positive_spread,
                "bid",
            ]
        )
        /
        results.loc[
            positive_spread,
            "spread",
        ]
    )

    local_total_depth = (
        results[
            "local_bid_depth"
        ]
        +
        results[
            "local_ask_depth"
        ]
    )

    results[
        "local_depth_imbalance"
    ] = np.where(
        local_total_depth > 0,
        (
            results[
                "local_bid_depth"
            ]
            -
            results[
                "local_ask_depth"
            ]
        )
        /
        local_total_depth,
        np.nan,
    )

    results[
        "bid_centroid_distance"
    ] = (
        results[
            "bid"
        ]
        -
        results[
            "vw_bid"
        ]
    )

    results[
        "ask_centroid_distance"
    ] = (
        results[
            "vw_ask"
        ]
        -
        results[
            "ask"
        ]
    )

    results[
        "centroid_distance_imbalance"
    ] = (
        results[
            "ask_centroid_distance"
        ]
        -
        results[
            "bid_centroid_distance"
        ]
    )

    # Future traditional microprice returns.
    for h in HORIZONS:
        results[
            f"R_{h}_bps"
        ] = (
            (
                results[
                    "microprice"
                ]
                .shift(
                    -h
                )
                /
                results[
                    "microprice"
                ]
            )
            - 1
        ) * 10_000

    return results


# ============================================================
# QUANTILE BINNING
# ============================================================

def add_quantile_bins(
    results: pd.DataFrame,
) -> pd.DataFrame:
    results = (
        results.copy()
    )

    valid_signal = (
        results[
            "delta_D_bps"
        ]
        .notna()
    )

    results.loc[
        valid_signal,
        "delta_D_bin",
    ] = (
        pd.qcut(
            results.loc[
                valid_signal,
                "delta_D_bps",
            ],
            q=N_SIGNAL_BINS,
            labels=False,
            duplicates="drop",
        )
        + 1
    )

    valid_duration = (
        results[
            "bar_duration_seconds"
        ]
        .notna()
    )

    results.loc[
        valid_duration,
        "duration_bin",
    ] = (
        pd.qcut(
            results.loc[
                valid_duration,
                "bar_duration_seconds",
            ],
            q=N_DURATION_BINS,
            labels=False,
            duplicates="drop",
        )
        + 1
    )

    return results


# ============================================================
# SIGNAL BIN SUMMARY
# ============================================================

def build_signal_bin_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    agg_spec = {
        "delta_D_bps": [
            "count",
            "mean",
            "median",
            "min",
            "max",
        ]
    }

    for h in HORIZONS:
        agg_spec[
            f"R_{h}_bps"
        ] = [
            "count",
            "mean",
            "median",
            "std",
        ]

    summary = (
        results
        .dropna(
            subset=[
                "delta_D_bin"
            ]
        )
        .groupby(
            "delta_D_bin",
            observed=True,
        )
        .agg(
            agg_spec
        )
    )

    summary.columns = [
        (
            f"{column}_{stat}"
        )
        for column, stat
        in summary.columns
    ]

    summary = (
        summary
        .reset_index()
    )

    for h in HORIZONS:
        mean_col = (
            f"R_{h}_bps_mean"
        )

        std_col = (
            f"R_{h}_bps_std"
        )

        summary[
            f"R_{h}_signal_to_noise"
        ] = np.where(
            summary[
                std_col
            ].abs() > 0,
            summary[
                mean_col
            ]
            /
            summary[
                std_col
            ],
            np.nan,
        )

    return summary


# ============================================================
# DURATION SUMMARY
# ============================================================

def build_duration_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    return (
        results
        .dropna(
            subset=[
                "duration_bin",
                "bar_duration_seconds",
            ]
        )
        .groupby(
            "duration_bin",
            observed=True,
        )[
            "bar_duration_seconds"
        ]
        .agg(
            [
                "count",
                "min",
                "mean",
                "median",
                "max",
            ]
        )
        .reset_index()
    )


# ============================================================
# CORE PLOTS
# ============================================================

def plot_core_metrics(
    results: pd.DataFrame,
) -> None:
    x = results[
        "cum_dollar_volume_mm"
    ]

    plot_specs = [
        (
            "ask",
            "vw_ask",
            "Best ask",
            "Volume-weighted ask",
            "Price",
            (
                "Best Ask vs "
                "Volume-Weighted Ask "
                f"(Local Book +/- "
                f"${MAX_DISTANCE_DOLLARS:.2f})"
            ),
        ),
        (
            "bid",
            "vw_bid",
            "Best bid",
            "Volume-weighted bid",
            "Price",
            (
                "Best Bid vs "
                "Volume-Weighted Bid "
                f"(Local Book +/- "
                f"${MAX_DISTANCE_DOLLARS:.2f})"
            ),
        ),
        (
            "spread",
            "vw_spread",
            "Standard spread",
            "Volume-weighted spread",
            "Spread ($)",
            (
                "Standard Spread vs "
                "Volume-Weighted Liquidity Spread"
            ),
        ),
        (
            "microprice",
            "vw_microprice",
            "Traditional microprice",
            "Volume-weighted microprice",
            "Price",
            (
                "Traditional Microprice vs "
                "Volume-Weighted Microprice"
            ),
        ),
        (
            "bid_centroid_distance",
            "ask_centroid_distance",
            "Bid centroid distance",
            "Ask centroid distance",
            "Distance from Touch ($)",
            (
                "Local Liquidity "
                "Centroid Distance"
            ),
        ),
    ]

    for (
        col_1,
        col_2,
        label_1,
        label_2,
        ylabel,
        title,
    ) in plot_specs:
        fig, ax = plt.subplots(
            figsize=(
                14,
                6,
            )
        )

        ax.plot(
            x,
            results[
                col_1
            ],
            linewidth=1.0,
            label=label_1,
        )

        ax.plot(
            x,
            results[
                col_2
            ],
            linewidth=1.2,
            label=label_2,
        )

        ax.set_xlabel(
            "Cumulative Dollar Volume ($MM)"
        )

        ax.set_ylabel(
            ylabel
        )

        ax.set_title(
            title
        )

        ax.grid(
            alpha=0.3
        )

        ax.legend(
            loc="best"
        )

        plt.tight_layout()
        plt.show()

    fig, ax = plt.subplots(
        figsize=(
            14,
            6,
        )
    )

    ax.plot(
        x,
        results[
            "microprice_displacement_bps"
        ],
        linewidth=1.1,
        label=(
            "VW microprice - "
            "traditional microprice"
        ),
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Cumulative Dollar Volume ($MM)"
    )

    ax.set_ylabel(
        "Microprice Displacement (bps)"
    )

    ax.set_title(
        "Volume-Weighted "
        "Microprice Displacement"
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend(
        loc="best"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 1D SIGNAL DECILE PLOTS
# ============================================================

def plot_signal_bins(
    summary: pd.DataFrame,
) -> None:
    x = summary[
        "delta_D_bin"
    ]

    for h in HORIZONS:
        fig, ax = plt.subplots(
            figsize=(
                11,
                6,
            )
        )

        ax.plot(
            x,
            summary[
                f"R_{h}_bps_mean"
            ],
            marker="o",
            linewidth=1.5,
        )

        ax.axhline(
            0.0,
            linestyle="--",
            linewidth=1,
        )

        ax.set_xlabel(
            "Delta D Decile "
            "(1 = Most Negative, "
            f"{N_SIGNAL_BINS} = Most Positive)"
        )

        ax.set_ylabel(
            f"Average Future "
            f"{h}-Bar Microprice Return (bps)"
        )

        ax.set_title(
            "Change in VW Microprice "
            f"Displacement vs Future "
            f"{h}-Bar Return"
        )

        ax.grid(
            alpha=0.3
        )

        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(
            figsize=(
                11,
                6,
            )
        )

        ax.plot(
            x,
            summary[
                f"R_{h}_bps_std"
            ],
            marker="o",
            linewidth=1.5,
        )

        ax.set_xlabel(
            "Delta D Decile "
            "(1 = Most Negative, "
            f"{N_SIGNAL_BINS} = Most Positive)"
        )

        ax.set_ylabel(
            f"Std Dev of Future "
            f"{h}-Bar Microprice Return (bps)"
        )

        ax.set_title(
            "Change in VW Microprice "
            f"Displacement vs Future "
            f"{h}-Bar Return Dispersion"
        )

        ax.grid(
            alpha=0.3
        )

        plt.tight_layout()
        plt.show()


# ============================================================
# HEATMAP HELPERS
#
# Return heatmap cell:
#
# E[
#   R_{t,h}
#   |
#   Delta D decile,
#   duration decile
# ]
# ============================================================

def build_heatmap_table(
    results: pd.DataFrame,
    value_col: str,
    agg: str,
) -> pd.DataFrame:
    grouped = (
        results
        .dropna(
            subset=[
                "delta_D_bin",
                "duration_bin",
                value_col,
            ]
        )
        .groupby(
            [
                "duration_bin",
                "delta_D_bin",
            ],
            observed=True,
        )[
            value_col
        ]
    )

    if agg == "mean":
        table = (
            grouped.mean()
        )

    elif agg == "std":
        table = (
            grouped.std()
        )

    else:
        raise ValueError(
            f"Unknown aggregation: {agg}"
        )

    return (
        table
        .unstack(
            "delta_D_bin"
        )
        .reindex(
            index=range(
                1,
                N_DURATION_BINS + 1,
            ),
            columns=range(
                1,
                N_SIGNAL_BINS + 1,
            ),
        )
    )


def plot_heatmap(
    table: pd.DataFrame,
    title: str,
    colorbar_label: str,
    value_format: str,
) -> None:
    fig, ax = plt.subplots(
        figsize=(
            12,
            8,
        )
    )

    image = ax.imshow(
        table.values,
        aspect="auto",
        origin="lower",
    )

    ax.set_xticks(
        range(
            N_SIGNAL_BINS
        )
    )

    ax.set_xticklabels(
        range(
            1,
            N_SIGNAL_BINS + 1,
        )
    )

    ax.set_yticks(
        range(
            N_DURATION_BINS
        )
    )

    ax.set_yticklabels(
        range(
            1,
            N_DURATION_BINS + 1,
        )
    )

    ax.set_xlabel(
        "Delta D Decile "
        "(1 = Most Negative, "
        f"{N_SIGNAL_BINS} = Most Positive)"
    )

    ax.set_ylabel(
        "Bar Duration Decile "
        "(1 = Fastest / Highest Intensity, "
        f"{N_DURATION_BINS} = Slowest)"
    )

    ax.set_title(
        title
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        colorbar_label
    )

    for row_idx in range(
        table.shape[
            0
        ]
    ):
        for col_idx in range(
            table.shape[
                1
            ]
        ):
            value = (
                table.iloc[
                    row_idx,
                    col_idx,
                ]
            )

            if pd.notna(
                value
            ):
                ax.text(
                    col_idx,
                    row_idx,
                    format(
                        value,
                        value_format,
                    ),
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    plt.tight_layout()
    plt.show()


# ============================================================
# SIGNAL x DURATION HEATMAPS
# ============================================================

def plot_signal_duration_heatmaps(
    results: pd.DataFrame,
) -> None:
    for h in HORIZONS:
        return_col = (
            f"R_{h}_bps"
        )

        mean_table = (
            build_heatmap_table(
                results,
                value_col=return_col,
                agg="mean",
            )
        )

        plot_heatmap(
            mean_table,
            title=(
                "Delta D x Bar Duration\n"
                f"Mean Future "
                f"{h}-Bar Traditional "
                "Microprice Return"
            ),
            colorbar_label=(
                "Mean Future Return (bps)"
            ),
            value_format=".2f",
        )

        std_table = (
            build_heatmap_table(
                results,
                value_col=return_col,
                agg="std",
            )
        )

        plot_heatmap(
            std_table,
            title=(
                "Delta D x Bar Duration\n"
                f"Std Dev of Future "
                f"{h}-Bar Traditional "
                "Microprice Return"
            ),
            colorbar_label=(
                "Return Std Dev (bps)"
            ),
            value_format=".2f",
        )

    count_table = (
        results
        .dropna(
            subset=[
                "delta_D_bin",
                "duration_bin",
            ]
        )
        .groupby(
            [
                "duration_bin",
                "delta_D_bin",
            ],
            observed=True,
        )
        .size()
        .unstack(
            "delta_D_bin"
        )
        .reindex(
            index=range(
                1,
                N_DURATION_BINS + 1,
            ),
            columns=range(
                1,
                N_SIGNAL_BINS + 1,
            ),
        )
    )

    plot_heatmap(
        count_table,
        title=(
            "Observation Count by "
            "Delta D Decile x "
            "Bar Duration Decile"
        ),
        colorbar_label=(
            "Observations"
        ),
        value_format=".0f",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        "Building dollar-volume bars "
        "directly from MBO T records..."
    )

    (
        boundaries,
        target_instrument,
        target_publisher,
    ) = (
        build_volume_boundaries_from_mbo(
            MBO_PATH
        )
    )

    print(
        f"Bars: "
        f"{len(boundaries):,}"
    )

    print(
        f"Instrument: "
        f"{target_instrument}"
    )

    print(
        f"Publisher: "
        f"{target_publisher}"
    )

    results = (
        replay_to_boundaries(
            path=MBO_PATH,
            boundaries=boundaries,
            target_instrument=(
                target_instrument
            ),
            target_publisher=(
                target_publisher
            ),
        )
    )

    results = (
        add_derived_features(
            results
        )
    )

    results = (
        add_quantile_bins(
            results
        )
    )

    # --------------------------------------------------------
    # SAMPLE METRICS
    # --------------------------------------------------------

    print()
    print(
        "=" * 180
    )
    print(
        "VOLUME-WEIGHTED BOOK METRICS"
    )
    print(
        "=" * 180
    )

    sample_columns = [
        "bar",
        "ts_event",
        "cum_dollar_volume_mm",
        "dollar_volume",
        "bar_duration_seconds",
        "dollar_intensity_mm_per_sec",
        "bid",
        "vw_bid",
        "ask",
        "vw_ask",
        "spread",
        "vw_spread",
        "microprice",
        "vw_microprice",
        "D_bps",
        "delta_D_bps",
        "local_bid_depth",
        "local_ask_depth",
        "local_depth_imbalance",
        "bid_centroid_distance",
        "ask_centroid_distance",
        "centroid_distance_imbalance",
    ]

    print(
        results[
            sample_columns
        ]
        .head(
            30
        )
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # SIGNAL DECILES
    # --------------------------------------------------------

    signal_summary = (
        build_signal_bin_summary(
            results
        )
    )

    print()
    print(
        "=" * 200
    )
    print(
        "BINNED DELTA D VS FUTURE "
        "TRADITIONAL MICROPRICE RETURN"
    )
    print(
        "=" * 200
    )

    print(
        signal_summary
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # DURATION DECILES
    # --------------------------------------------------------

    duration_summary = (
        build_duration_summary(
            results
        )
    )

    print()
    print(
        "=" * 100
    )
    print(
        "BAR DURATION DECILES"
    )
    print(
        "=" * 100
    )

    print(
        duration_summary
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    plot_core_metrics(
        results
    )

    plot_signal_bins(
        signal_summary
    )

    plot_signal_duration_heatmaps(
        results
    )


if __name__ == "__main__":
    main()
