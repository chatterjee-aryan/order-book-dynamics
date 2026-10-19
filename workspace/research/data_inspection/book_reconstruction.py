from __future__ import annotations

import math
from pathlib import Path

import databento as db
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from workspace.research.order_book.orderbook import (
    Book,
    LevelOrders,
)


# ============================================================
# CONFIG
# ============================================================

MBO_PATH = Path("data/raw/mbo/gld-2026-01-29.mbo.csv")

DOLLAR_VOLUME_THRESHOLD = 1_000_000.0
MBO_CHUNK_SIZE = 1_000_000

HALF_LIFE = 7
WINDOW_SIZE = 100

PLOT_START_CUM_VOLUME_MM = 0.0


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


def from_fixed_price(value: int) -> float:
    return (
        value
        / db.FIXED_PRICE_SCALE
    )


def make_mbo(row) -> FastMBOMsg:
    return FastMBOMsg(
        instrument_id=int(row.instrument_id),
        publisher_id=int(row.publisher_id),
        order_id=int(row.order_id),
        price=to_fixed_price(row.price),
        size=int(row.size),
        flags=int(row.flags),
        action=row.action,
        side=row.side,
        sequence=int(row.sequence),
    )


# ============================================================
# BOOK HELPERS
# ============================================================

def clone_order(order) -> FastMBOMsg:
    return FastMBOMsg(
        instrument_id=order.instrument_id,
        publisher_id=order.publisher_id,
        order_id=order.order_id,
        price=order.price,
        size=order.size,
        flags=order.flags,
        action=order.action,
        side=order.side,
        sequence=order.sequence,
    )


def clone_book(source: Book) -> Book:
    """
    Purpose-built clone for the frozen-passive counterfactual.
    """
    target = Book()

    for price, level in source.bids.items():
        cloned_orders = []

        for order in level.orders:
            cloned = clone_order(order)
            cloned_orders.append(cloned)

            if (
                order.order_id
                in source.orders_by_id
            ):
                target.orders_by_id[
                    cloned.order_id
                ] = cloned

        target.bids[price] = LevelOrders(
            price=price,
            orders=cloned_orders,
        )

    for price, level in source.offers.items():
        cloned_orders = []

        for order in level.orders:
            cloned = clone_order(order)
            cloned_orders.append(cloned)

            if (
                order.order_id
                in source.orders_by_id
            ):
                target.orders_by_id[
                    cloned.order_id
                ] = cloned

        target.offers[price] = LevelOrders(
            price=price,
            orders=cloned_orders,
        )

    return target


def get_bbo(
    book: Book,
) -> tuple[float, float]:
    bid, ask = book.bbo()

    bid_px = (
        math.nan
        if bid is None
        else from_fixed_price(
            bid.price
        )
    )

    ask_px = (
        math.nan
        if ask is None
        else from_fixed_price(
            ask.price
        )
    )

    return (
        bid_px,
        ask_px,
    )


def get_full_depth(
    book: Book,
) -> tuple[int, int, float]:
    """
    Total visible resting quantity on each side and normalized
    full-book depth imbalance.

        +1 = entirely bid-heavy
         0 = balanced
        -1 = entirely ask-heavy
    """
    bid_depth = (
        book.total_bid_depth()
    )

    ask_depth = (
        book.total_ask_depth()
    )

    total_depth = (
        bid_depth
        + ask_depth
    )

    if total_depth > 0:
        imbalance = (
            bid_depth
            - ask_depth
        ) / total_depth

    else:
        imbalance = math.nan

    return (
        bid_depth,
        ask_depth,
        imbalance,
    )


# ============================================================
# BUILD DOLLAR-VOLUME BOUNDARIES DIRECTLY FROM MBO TRADES
#
# Only action == "T" contributes to traded volume.
# F/C records are NOT counted as trades.
#
# Existing project convention is preserved:
# - entire threshold-crossing trade belongs to current bar
# - no excess dollar volume is carried into the next bar
# ============================================================

MBO_BOUNDARY_USECOLS = [
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
    bar_start_sequence = None

    target_instrument = None
    target_publisher = None

    for chunk in pd.read_csv(
        path,
        usecols=MBO_BOUNDARY_USECOLS,
        chunksize=MBO_CHUNK_SIZE,
        engine="c",
        memory_map=True,
    ):
        # MBO already contains the trades.
        # Count T only -- never F.
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

            sequence = int(
                row.sequence
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

            if (
                bar_start_sequence
                is None
            ):
                bar_start_sequence = (
                    sequence
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
                        "start_sequence":
                            bar_start_sequence,

                        "end_sequence":
                            sequence,

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

                # Preserve the existing convention:
                # no overshoot carry.
                bar_dollar_volume = 0.0
                bar_start_sequence = None

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
            "Could not infer target "
            "instrument/publisher from MBO trades."
        )

    return (
        boundaries,
        int(target_instrument),
        int(target_publisher),
    )


# ============================================================
# MBO REPLAY
# ============================================================

MBO_REPLAY_USECOLS = [
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


def replay_mbo(
    path: Path,
    boundaries: list[dict],
    target_instrument: int,
    target_publisher: int,
) -> pd.DataFrame:
    """
    Replay the actual book and the frozen-passive
    counterfactual using only MBO.

    Endpoint convention:
    - actual and counterfactual BBO are sampled immediately
      BEFORE the threshold-crossing sequence;
    - T records in that crossing sequence are still counted
      in aggressor flow for the completed dollar-volume bar;
    - T/F/N do not mutate the actual book;
    - only A/C/M/R mutate the actual book.
    """

    actual_book = Book()
    counterfactual_book = None

    results: list[dict] = []

    bar_idx = 0
    previous_sequence = None
    boundary_sampled = False

    start_bid = math.nan
    start_ask = math.nan

    pending_result = None

    buy_aggressor_shares = 0
    sell_aggressor_shares = 0

    buy_aggressor_dollars = 0.0
    sell_aggressor_dollars = 0.0
    unknown_aggressor_dollars = 0.0

    def reset_bar_flow() -> None:
        nonlocal buy_aggressor_shares
        nonlocal sell_aggressor_shares
        nonlocal buy_aggressor_dollars
        nonlocal sell_aggressor_dollars
        nonlocal unknown_aggressor_dollars

        buy_aggressor_shares = 0
        sell_aggressor_shares = 0

        buy_aggressor_dollars = 0.0
        sell_aggressor_dollars = 0.0
        unknown_aggressor_dollars = 0.0

    def begin_sequence(
        sequence: int,
    ) -> None:
        nonlocal counterfactual_book
        nonlocal boundary_sampled
        nonlocal start_bid
        nonlocal start_ask
        nonlocal pending_result

        if (
            bar_idx
            >= len(boundaries)
        ):
            return

        bar = boundaries[
            bar_idx
        ]

        # ----------------------------------------------------
        # START OF BAR
        #
        # Sample actual BBO and freeze passive book before
        # the bar's first trade sequence.
        # ----------------------------------------------------

        if (
            counterfactual_book is None
            and sequence
            == bar["start_sequence"]
        ):
            (
                start_bid,
                start_ask,
            ) = get_bbo(
                actual_book
            )

            counterfactual_book = (
                clone_book(
                    actual_book
                )
            )

            reset_bar_flow()

        if (
            counterfactual_book is None
            and sequence
            > bar["start_sequence"]
        ):
            raise RuntimeError(
                f"Missed start of bar {bar_idx}. "
                f"Expected sequence "
                f"{bar['start_sequence']}, "
                f"got {sequence}."
            )

        # ----------------------------------------------------
        # END OF BAR
        #
        # Sample state before applying any message from the
        # threshold-crossing sequence.
        # ----------------------------------------------------

        if (
            sequence
            == bar["end_sequence"]
        ):
            if (
                counterfactual_book
                is None
            ):
                raise RuntimeError(
                    f"Reached end of bar {bar_idx} "
                    "without a counterfactual book."
                )

            (
                actual_bid,
                actual_ask,
            ) = get_bbo(
                actual_book
            )

            (
                cf_bid,
                cf_ask,
            ) = get_bbo(
                counterfactual_book
            )

            (
                total_bid_depth,
                total_ask_depth,
                full_depth_imbalance,
            ) = get_full_depth(
                actual_book
            )

            pending_result = {
                "bar":
                    bar_idx,

                "ts_event":
                    bar["ts_event"],

                "dollar_volume":
                    bar["dollar_volume"],

                "start_bid":
                    start_bid,

                "start_ask":
                    start_ask,

                "actual_bid":
                    actual_bid,

                "actual_ask":
                    actual_ask,

                "cf_bid":
                    cf_bid,

                "cf_ask":
                    cf_ask,

                "total_bid_depth":
                    total_bid_depth,

                "total_ask_depth":
                    total_ask_depth,

                "full_depth_imbalance":
                    full_depth_imbalance,
            }

            boundary_sampled = True

    def finish_sequence(
        sequence: int,
    ) -> None:
        nonlocal bar_idx
        nonlocal counterfactual_book
        nonlocal boundary_sampled
        nonlocal pending_result

        if (
            bar_idx
            >= len(boundaries)
        ):
            return

        bar = boundaries[
            bar_idx
        ]

        if (
            boundary_sampled
            and sequence
            == bar["end_sequence"]
        ):
            directional_dollars = (
                buy_aggressor_dollars
                + sell_aggressor_dollars
            )

            if (
                directional_dollars
                > 0
            ):
                trade_imbalance = (
                    buy_aggressor_dollars
                    - sell_aggressor_dollars
                ) / directional_dollars

            else:
                trade_imbalance = (
                    math.nan
                )

            directional_shares = (
                buy_aggressor_shares
                + sell_aggressor_shares
            )

            pending_result.update(
                {
                    "buy_aggressor_shares":
                        buy_aggressor_shares,

                    "sell_aggressor_shares":
                        sell_aggressor_shares,

                    "directional_aggressor_shares":
                        directional_shares,

                    "buy_aggressor_dollars":
                        buy_aggressor_dollars,

                    "sell_aggressor_dollars":
                        sell_aggressor_dollars,

                    "directional_aggressor_dollars":
                        directional_dollars,

                    "unknown_aggressor_dollars":
                        unknown_aggressor_dollars,

                    "trade_imbalance":
                        trade_imbalance,
                }
            )

            results.append(
                pending_result
            )

            pending_result = None
            counterfactual_book = None
            boundary_sampled = False

            bar_idx += 1

    print(
        "Streaming MBO replay..."
    )

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            usecols=MBO_REPLAY_USECOLS,
            chunksize=MBO_CHUNK_SIZE,
            engine="c",
            memory_map=True,
        ),
        start=1,
    ):
        chunk = chunk.loc[
            chunk["instrument_id"].eq(
                target_instrument
            )
            &
            chunk["publisher_id"].eq(
                target_publisher
            )
        ]

        if chunk.empty:
            continue

        # IMPORTANT:
        # Preserve native MBO file order.
        # Never sort the MBO stream.
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

            action = row.action

            if (
                previous_sequence
                is None
            ):
                previous_sequence = (
                    sequence
                )

                begin_sequence(
                    sequence
                )

            elif (
                sequence
                != previous_sequence
            ):
                finish_sequence(
                    previous_sequence
                )

                if (
                    bar_idx
                    >= len(boundaries)
                ):
                    break

                previous_sequence = (
                    sequence
                )

                begin_sequence(
                    sequence
                )

            # ------------------------------------------------
            # ACTUAL BOOK
            #
            # Only passive/order-state records mutate it.
            # ------------------------------------------------

            if action in (
                "A",
                "C",
                "M",
                "R",
            ):
                actual_book.apply(
                    make_mbo(
                        row
                    )
                )

            # ------------------------------------------------
            # TRADE / AGGRESSOR FLOW
            #
            # Count T only.
            # F is deliberately ignored to avoid counting
            # the passive-side fill as another trade.
            # ------------------------------------------------

            elif (
                action == "T"
                and counterfactual_book
                is not None
            ):
                mbo = make_mbo(
                    row
                )

                px = float(
                    row.price
                )

                qty = int(
                    row.size
                )

                if (
                    math.isfinite(px)
                    and qty > 0
                ):
                    notional = (
                        px
                        * qty
                    )

                    if (
                        mbo.side
                        == "B"
                    ):
                        buy_aggressor_shares += (
                            qty
                        )

                        buy_aggressor_dollars += (
                            notional
                        )

                    elif (
                        mbo.side
                        == "A"
                    ):
                        sell_aggressor_shares += (
                            qty
                        )

                        sell_aggressor_dollars += (
                            notional
                        )

                    else:
                        unknown_aggressor_dollars += (
                            notional
                        )

                # The threshold-crossing sequence is omitted
                # from counterfactual replay because its
                # endpoint was sampled pre-sequence.
                if (
                    not boundary_sampled
                    and mbo.side
                    in ("A", "B")
                ):
                    counterfactual_book.apply_counterfactual(
                        mbo
                    )

        print(
            f"Chunk {chunk_number} | "
            f"bars {bar_idx:,}/"
            f"{len(boundaries):,}"
        )

        if (
            bar_idx
            >= len(boundaries)
        ):
            break

    if (
        previous_sequence
        is not None
        and bar_idx
        < len(boundaries)
    ):
        finish_sequence(
            previous_sequence
        )

    frame = pd.DataFrame(
        results
    )

    if frame.empty:
        raise RuntimeError(
            "No matched bars."
        )

    return frame


# ============================================================
# DERIVED BAR-LEVEL METRICS
# ============================================================

def add_bar_metrics(
    results: pd.DataFrame,
) -> pd.DataFrame:
    results = results.copy()

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
        "mid_price"
    ] = (
        results[
            "actual_bid"
        ]
        +
        results[
            "actual_ask"
        ]
    ) / 2

    # --------------------------------------------------------
    # BID DECOMPOSITION
    # --------------------------------------------------------

    results[
        "bid_aggressor_effect_bps"
    ] = (
        (
            results["cf_bid"]
            -
            results["start_bid"]
        )
        /
        results["start_bid"]
        *
        10_000
    )

    results[
        "bid_supply_effect_bps"
    ] = (
        (
            results["actual_bid"]
            -
            results["cf_bid"]
        )
        /
        results["start_bid"]
        *
        10_000
    )

    # --------------------------------------------------------
    # ASK DECOMPOSITION
    # --------------------------------------------------------

    results[
        "ask_aggressor_effect_bps"
    ] = (
        (
            results["cf_ask"]
            -
            results["start_ask"]
        )
        /
        results["start_ask"]
        *
        10_000
    )

    results[
        "ask_supply_effect_bps"
    ] = (
        (
            results["actual_ask"]
            -
            results["cf_ask"]
        )
        /
        results["start_ask"]
        *
        10_000
    )

    # --------------------------------------------------------
    # ABSORPTION
    # --------------------------------------------------------

    results[
        "bid_absorption"
    ] = (
        -results[
            "bid_supply_effect_bps"
        ]
        /
        results[
            "bid_aggressor_effect_bps"
        ]
    )

    results[
        "ask_absorption"
    ] = (
        -results[
            "ask_supply_effect_bps"
        ]
        /
        results[
            "ask_aggressor_effect_bps"
        ]
    )

    for side in (
        "bid",
        "ask",
    ):
        aggressor_col = (
            f"{side}_aggressor_effect_bps"
        )

        absorption_col = (
            f"{side}_absorption"
        )

        results.loc[
            results[
                aggressor_col
            ].eq(0),
            absorption_col,
        ] = np.nan

        results[
            absorption_col
        ] = (
            results[
                absorption_col
            ]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
        )

    # --------------------------------------------------------
    # EWMA ABSORPTION
    # --------------------------------------------------------

    results[
        "bid_absorption_ewma"
    ] = (
        results[
            "bid_absorption"
        ]
        .ewm(
            halflife=HALF_LIFE,
            adjust=False,
        )
        .mean()
    )

    results[
        "ask_absorption_ewma"
    ] = (
        results[
            "ask_absorption"
        ]
        .ewm(
            halflife=HALF_LIFE,
            adjust=False,
        )
        .mean()
    )

    return results


# ============================================================
# NON-OVERLAPPING WINDOW SUMMARY
# ============================================================

def build_window_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    windowed = results.copy()

    windowed[
        "window_id"
    ] = (
        np.arange(
            len(windowed)
        )
        //
        WINDOW_SIZE
    )

    summary = (
        windowed
        .groupby(
            "window_id",
            observed=True,
        )
        .agg(
            start_bar=(
                "bar",
                "first",
            ),

            end_bar=(
                "bar",
                "last",
            ),

            start_time=(
                "ts_event",
                "first",
            ),

            end_time=(
                "ts_event",
                "last",
            ),

            start_cum_volume_mm=(
                "cum_dollar_volume_mm",
                "first",
            ),

            end_cum_volume_mm=(
                "cum_dollar_volume_mm",
                "last",
            ),

            median_bid_absorption=(
                "bid_absorption",
                "median",
            ),

            median_ask_absorption=(
                "ask_absorption",
                "median",
            ),

            median_full_depth_imbalance=(
                "full_depth_imbalance",
                "median",
            ),

            mean_full_depth_imbalance=(
                "full_depth_imbalance",
                "mean",
            ),

            median_total_bid_depth=(
                "total_bid_depth",
                "median",
            ),

            median_total_ask_depth=(
                "total_ask_depth",
                "median",
            ),

            buy_aggressor_dollars=(
                "buy_aggressor_dollars",
                "sum",
            ),

            sell_aggressor_dollars=(
                "sell_aggressor_dollars",
                "sum",
            ),

            unknown_aggressor_dollars=(
                "unknown_aggressor_dollars",
                "sum",
            ),

            mid_price=(
                "mid_price",
                "last",
            ),

            bars_in_window=(
                "bar",
                "count",
            ),
        )
        .reset_index()
    )

    summary[
        "absorption_imbalance"
    ] = (
        summary[
            "median_bid_absorption"
        ]
        -
        summary[
            "median_ask_absorption"
        ]
    )

    summary[
        "directional_aggressor_dollars"
    ] = (
        summary[
            "buy_aggressor_dollars"
        ]
        +
        summary[
            "sell_aggressor_dollars"
        ]
    )

    directional = (
        summary[
            "directional_aggressor_dollars"
        ]
    )

    summary[
        "trade_imbalance"
    ] = np.where(
        directional > 0,
        (
            summary[
                "buy_aggressor_dollars"
            ]
            -
            summary[
                "sell_aggressor_dollars"
            ]
        )
        /
        directional,
        np.nan,
    )

    summary[
        "buyer_aggressor_share"
    ] = np.where(
        directional > 0,
        summary[
            "buy_aggressor_dollars"
        ]
        /
        directional,
        np.nan,
    )

    return summary


# ============================================================
# OUTPUT
# ============================================================

def print_bar_sample(
    results: pd.DataFrame,
) -> None:
    columns = [
        "bar",
        "ts_event",
        "cum_dollar_volume_mm",
        "dollar_volume",
        "start_bid",
        "start_ask",
        "actual_bid",
        "actual_ask",
        "cf_bid",
        "cf_ask",
        "bid_absorption",
        "ask_absorption",
        "total_bid_depth",
        "total_ask_depth",
        "full_depth_imbalance",
        "buy_aggressor_dollars",
        "sell_aggressor_dollars",
        "unknown_aggressor_dollars",
        "trade_imbalance",
        "mid_price",
    ]

    print()
    print(
        "=" * 180
    )
    print(
        "BAR-LEVEL MICROSTRUCTURE STATE"
    )
    print(
        "=" * 180
    )

    print(
        results[
            columns
        ]
        .head(30)
        .to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )


def print_window_summary(
    summary: pd.DataFrame,
) -> None:
    columns = [
        "window_id",
        "start_bar",
        "end_bar",
        "median_bid_absorption",
        "median_ask_absorption",
        "absorption_imbalance",
        "median_total_bid_depth",
        "median_total_ask_depth",
        "median_full_depth_imbalance",
        "buy_aggressor_dollars",
        "sell_aggressor_dollars",
        "unknown_aggressor_dollars",
        "trade_imbalance",
        "buyer_aggressor_share",
        "mid_price",
        "bars_in_window",
    ]

    print()
    print(
        "=" * 180
    )
    print(
        f"NON-OVERLAPPING "
        f"{WINDOW_SIZE}-BAR WINDOWS"
    )
    print(
        "=" * 180
    )

    print(
        summary[
            columns
        ]
        .to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )


# ============================================================
# PLOTS
# ============================================================

def plot_absorption(
    results: pd.DataFrame,
) -> None:
    plot_results = results.loc[
        results[
            "cum_dollar_volume_mm"
        ]
        >=
        PLOT_START_CUM_VOLUME_MM
    ].copy()

    x = plot_results[
        "cum_dollar_volume_mm"
    ]

    for (
        side,
        column,
        ylabel,
    ) in [
        (
            "Bid",
            "bid_absorption_ewma",
            "Bid Absorption EWMA",
        ),
        (
            "Ask",
            "ask_absorption_ewma",
            "Ask Absorption EWMA",
        ),
    ]:
        fig, ax1 = plt.subplots(
            figsize=(13, 6)
        )

        ax1.plot(
            x,
            plot_results[
                column
            ],
            label=(
                f"{side} absorption EWMA"
            ),
        )

        ax1.axhline(
            1.0,
            linestyle="--",
            linewidth=1,
            label="Full absorption",
        )

        ax1.set_xlabel(
            "Cumulative Dollar Volume ($MM)"
        )

        ax1.set_ylabel(
            ylabel
        )

        ax1.grid(
            alpha=0.3
        )

        ax2 = ax1.twinx()

        ax2.plot(
            x,
            plot_results[
                "mid_price"
            ],
            alpha=0.40,
            linewidth=1.0,
            label="Mid price",
        )

        ax2.set_ylabel(
            "Mid Price"
        )

        lines1, labels1 = (
            ax1.get_legend_handles_labels()
        )

        lines2, labels2 = (
            ax2.get_legend_handles_labels()
        )

        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="best",
        )

        ax1.set_title(
            f"{side} Absorption EWMA "
            f"(Half-Life = "
            f"{HALF_LIFE} bars)"
        )

        plt.tight_layout()
        plt.show()


def plot_window_state(
    summary: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    x = summary[
        "end_cum_volume_mm"
    ]

    ax.plot(
        x,
        summary[
            "trade_imbalance"
        ],
        marker="o",
        linewidth=1.5,
        label=(
            "Aggressor trade imbalance"
        ),
    )

    ax.plot(
        x,
        summary[
            "median_full_depth_imbalance"
        ],
        marker="s",
        linewidth=1.5,
        label=(
            "Full depth imbalance"
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
        "Normalized Imbalance (-1 to +1)"
    )

    ax.set_title(
        "Aggressor Flow vs "
        "Total Liquidity Resistance "
        f"({WINDOW_SIZE}-Bar Windows)"
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
    ) = build_volume_boundaries_from_mbo(
        MBO_PATH
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

    results = replay_mbo(
        path=MBO_PATH,
        boundaries=boundaries,
        target_instrument=target_instrument,
        target_publisher=target_publisher,
    )

    results = add_bar_metrics(
        results
    )

    window_summary = (
        build_window_summary(
            results
        )
    )

    print_bar_sample(
        results
    )

    print_window_summary(
        window_summary
    )

    plot_absorption(
        results
    )

    plot_window_state(
        window_summary
    )


if __name__ == "__main__":
    main()
