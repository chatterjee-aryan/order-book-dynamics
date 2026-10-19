from __future__ import annotations

import math
from pathlib import Path

import databento as db
import matplotlib.pyplot as plt
import pandas as pd

from workspace.research.order_book.orderbook import Book


# ============================================================
# CONFIG
# ============================================================

MBO_PATH = Path(
    "data/raw/mbo/gld-2026-01-29.mbo.csv"
)

DOLLAR_VOLUME_THRESHOLD = 10_000_000.0
MBO_CHUNK_SIZE = 1_000_000

# Skip level 1 and use levels 2 through 11.
BOOK_LEVELS = 10
CHOP = 0.1

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
# BUILD DOLLAR-VOLUME BAR BOUNDARIES
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
            "No completed dollar-volume bars found."
        )

    if (
        target_instrument is None
        or target_publisher is None
    ):

        raise RuntimeError(
            "Could not infer instrument/publisher."
        )

    return (
        boundaries,
        int(target_instrument),
        int(target_publisher),
    )


# ============================================================
# BOOK HELPERS
# ============================================================

def get_level_size(
    level_orders,
) -> int:

    return sum(
        int(order.size)
        for order
        in level_orders.orders
    )


def get_deep_levels(
    book: Book,
    side: str,
    k: int,
) -> list[
    tuple[
        float,
        int,
    ]
]:

    """
    Return [(price, size), ...] for levels 2 through k+1.

    Level 1 is deliberately excluded.
    """

    output = []

    if side == "B":

        levels = book.bids

        if len(levels) <= 1:
            return output

        n_available = min(
            k,
            len(levels) - 1,
        )

        # Best bid is peekitem(-1).
        # Start with second-best bid at -2.
        indices = range(
            -2,
            -2 - n_available,
            -1,
        )

    elif side == "A":

        levels = book.offers

        if len(levels) <= 1:
            return output

        n_available = min(
            k,
            len(levels) - 1,
        )

        # Best ask is peekitem(0).
        # Start with second-best ask at 1.
        indices = range(
            1,
            1 + n_available,
        )

    else:

        raise ValueError(
            "side must be 'B' or 'A'"
        )

    for i in indices:

        price_fixed = (
            levels.peekitem(i)[0]
        )

        level_size = get_level_size(
            levels[
                price_fixed
            ]
        )

        if level_size <= 0:
            continue

        price = from_fixed_price(
            price_fixed
        )

        output.append(
            (
                price,
                level_size,
            )
        )

    return output


def volume_weighted_price(
    levels: list[
        tuple[
            float,
            int,
        ]
    ],
) -> float:

    if not levels:
        return math.nan

    total_size = sum(
        size
        for _, size
        in levels
    )

    if total_size <= 0:
        return math.nan

    weighted_price = sum(
        price * size
        for price, size
        in levels
    )

    return (
        weighted_price
        / total_size
    )


# ============================================================
# PRICE / BOOK FEATURES
#
# 1. Touch mid
# 2. Deep book mid
# 3. Book-mid skew
# 4. Full depth-weighted COM
# ============================================================

def capture_price_metrics(
    book: Book,
) -> dict:

    bid_level, ask_level = (
        book.bbo()
    )

    if (
        bid_level is None
        or ask_level is None
    ):

        return {
            "touch_mid":
                math.nan,

            "book_mid":
                math.nan,

            "book_skew_bps":
                math.nan,

            "book_com_bps":
                math.nan,
        }

    # ========================================================
    # TOUCH MID
    # ========================================================

    best_bid = from_fixed_price(
        bid_level.price
    )

    best_ask = from_fixed_price(
        ask_level.price
    )

    touch_mid = (
        best_bid
        + best_ask
    ) / 2.0

    # ========================================================
    # DEEP LEVELS
    #
    # Levels 2-11 only.
    # ========================================================

    bid_levels = get_deep_levels(
        book=book,
        side="B",
        k=BOOK_LEVELS,
    )

    ask_levels = get_deep_levels(
        book=book,
        side="A",
        k=BOOK_LEVELS,
    )

    # ========================================================
    # DEEP BID / ASK VWAPS
    # ========================================================

    deep_bid_vwap = (
        volume_weighted_price(
            bid_levels
        )
    )

    deep_ask_vwap = (
        volume_weighted_price(
            ask_levels
        )
    )

    # ========================================================
    # BOOK MID
    # ========================================================

    if (
        math.isfinite(
            deep_bid_vwap
        )
        and math.isfinite(
            deep_ask_vwap
        )
    ):

        book_mid = (
            deep_bid_vwap
            + deep_ask_vwap
        ) / 2.0

    else:

        book_mid = math.nan

    # ========================================================
    # EXISTING BOOK-MID SKEW
    #
    # Positive:
    #   deep book midpoint lies above touch midpoint.
    #
    # Negative:
    #   deep book midpoint lies below touch midpoint.
    # ========================================================

    if (
        math.isfinite(
            book_mid
        )
        and touch_mid != 0
    ):

        book_skew_bps = (
            (
                book_mid
                - touch_mid
            )
            /
            touch_mid
            * 10_000
        )

    else:

        book_skew_bps = (
            math.nan
        )

    # ========================================================
    # CENTER OF MASS
    #
    # Use every displayed level individually.
    #
    #              sum q_i (p_i - M)
    # COM_price = --------------------
    #                   sum q_i
    #
    # where M = touch mid.
    #
    # Bids contribute negative distances.
    # Asks contribute positive distances.
    # ========================================================

    all_deep_levels = (
        bid_levels
        + ask_levels
    )

    total_depth = sum(
        size
        for _, size
        in all_deep_levels
    )

    if (
        total_depth > 0
        and touch_mid != 0
    ):

        com_price = (
            sum(
                size
                * (
                    price
                    - touch_mid
                )
                for price, size
                in all_deep_levels
            )
            /
            total_depth
        )

        book_com_bps = (
            com_price
            /
            touch_mid
            * 10_000
        )

    else:

        book_com_bps = (
            math.nan
        )

    return {
        "touch_mid":
            touch_mid,

        "book_mid":
            book_mid,

        "book_skew_bps":
            book_skew_bps,

        "book_com_bps":
            book_com_bps,
    }


# ============================================================
# REPLAY TO DOLLAR-VOLUME BAR ENDPOINTS
#
# Sample the book immediately BEFORE the threshold-crossing
# sequence.
#
# Only A/C/M/R mutate the reconstructed book.
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
                previous_sequence is None
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
                        capture_price_metrics(
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
                        }
                    )

                    rows.append(
                        metrics
                    )

                    bar_idx += 1

                previous_sequence = (
                    sequence
                )

            if row.action in (
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
# PLOT EXISTING BOOK-MID SKEW
# ============================================================

def plot_book_skew(
    results: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(
        figsize=(
            14,
            6,
        )
    )

    ax.plot(
        results[
            "bar"
        ],
        results[
            "book_skew_bps"
        ],
        linewidth=1.0,
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=0.8,
    )

    ax.set_xlabel(
        "Dollar-Volume Bar"
    )

    ax.set_ylabel(
        "Book Skew (bps)"
    )

    ax.set_title(
        f"Deep Book Mid Skew "
        f"(Levels 2-{BOOK_LEVELS + 1})"
    )

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# PLOT CENTER OF MASS
# ============================================================

def plot_book_com(
    results: pd.DataFrame,
) -> None:

    fig, (
        ax_mid,
        ax_com,
    ) = plt.subplots(
        2,
        1,
        figsize=(
            14,
            9,
        ),
        sharex=True,
    )

    # ========================================================
    # TOP: TOUCH MID
    # ========================================================

    ax_mid.plot(
        results[
            "bar"
        ],
        results[
            "touch_mid"
        ],
        linewidth=1.0,
    )

    ax_mid.set_ylabel(
        "Touch Mid Price"
    )

    ax_mid.set_title(
        "Touch Mid Price"
    )

    ax_mid.grid(
        alpha=0.3
    )

    # ========================================================
    # BOTTOM: BOOK CENTER OF MASS
    # ========================================================

    ax_com.plot(
        results[
            "bar"
        ],
        results[
            "book_com_bps"
        ],
        linewidth=1.0,
    )

    ax_com.axhline(
        0.0,
        linestyle="--",
        linewidth=0.8,
    )

    ax_com.set_xlabel(
        "Dollar-Volume Bar"
    )

    ax_com.set_ylabel(
        "Book Center of Mass (bps)"
    )

    ax_com.set_title(
        f"Depth-Weighted Book Center of Mass "
        f"(Levels 2-{BOOK_LEVELS + 1})"
    )

    ax_com.grid(
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()

# ============================================================
# OPTIONAL: PLOT BOTH ON SAME AXES
# ============================================================

def plot_skew_and_com(
    results: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(
        figsize=(
            14,
            6,
        )
    )

    ax.plot(
        results[
            "bar"
        ],
        results[
            "book_skew_bps"
        ],
        linewidth=1.0,
        label="Deep Book Mid Skew",
    )

    ax.plot(
        results[
            "bar"
        ],
        results[
            "book_com_bps"
        ],
        linewidth=1.0,
        label="Depth-Weighted COM",
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=0.8,
    )

    ax.set_xlabel(
        "Dollar-Volume Bar"
    )

    ax.set_ylabel(
        "Book Displacement (bps)"
    )

    ax.set_title(
        f"Book Skew vs Center of Mass "
        f"(Levels 2-{BOOK_LEVELS + 1})"
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

    (
        boundaries,
        target_instrument,
        target_publisher,
    ) = (
        build_volume_boundaries_from_mbo(
            MBO_PATH
        )
    )

    print()

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

    results = replay_to_boundaries(
        path=MBO_PATH,
        boundaries=boundaries,
        target_instrument=(
            target_instrument
        ),
        target_publisher=(
            target_publisher
        ),
    )
    cutoff = int(len(results)*CHOP)
    plotting_results = results.copy()[cutoff:-cutoff]

    # Existing measure
    plot_book_skew(
        plotting_results
    )

    # New COM measure
    plot_book_com(
        plotting_results
    )


if __name__ == "__main__":
    main()