from __future__ import annotations

import math
from pathlib import Path

import databento as db
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from workspace.research.order_book.orderbook import Book


# ============================================================
# CONFIG
# ============================================================

MBO_PATH = Path(
    "data/raw/mbo/gld-2026-01-29.mbo.csv"
)

DOLLAR_VOLUME_THRESHOLD = 1_000_000.0
MBO_CHUNK_SIZE = 1_000_000

EXOTIC_WINDOW = 150
RETURN_HORIZON = 150


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


def to_fixed_price(
    value,
) -> int:

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
# PRICE FEATURES
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

            "bid_ask_spread":
                math.nan,
        }

    bid = from_fixed_price(
        bid_level.price
    )

    ask = from_fixed_price(
        ask_level.price
    )

    touch_mid = (
        bid + ask
    ) / 2.0

    bid_ask_spread = (
        ask - bid
    )

    return {
        "touch_mid":
            touch_mid,

        "bid_ask_spread":
            bid_ask_spread,
    }


# ============================================================
# REPLAY TO DOLLAR-VOLUME BAR ENDPOINTS
#
# Book sampled immediately BEFORE threshold-crossing sequence.
#
# Only A/C/M/R mutate reconstructed book.
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
# 150-BAR INVERSE-SPREAD-WEIGHTED EXOTIC PRICE
#
#                sum(mid_i / spread_i)
# P_t =         -----------------------
#                   sum(1 / spread_i)
#
# over i = t-149, ..., t
#
# No look-ahead.
# ============================================================

def add_exotic_price(
    results: pd.DataFrame,
) -> pd.DataFrame:

    results = results.copy()

    results[
        "cum_dollar_volume_mm"
    ] = (
        results[
            "dollar_volume"
        ]
        .cumsum()
        / 1_000_000
    )

    valid_spread = (
        results[
            "bid_ask_spread"
        ] > 0
    )

    inverse_weight = (
        1.0
        /
        results[
            "bid_ask_spread"
        ].where(
            valid_spread
        )
    )

    weighted_mid = (
        results[
            "touch_mid"
        ]
        *
        inverse_weight
    )

    numerator = (
        weighted_mid
        .rolling(
            window=EXOTIC_WINDOW,
            min_periods=EXOTIC_WINDOW,
        )
        .sum()
    )

    denominator = (
        inverse_weight
        .rolling(
            window=EXOTIC_WINDOW,
            min_periods=EXOTIC_WINDOW,
        )
        .sum()
    )

    results[
        "exotic_price_150"
    ] = (
        numerator
        /
        denominator
    )

    return results


# ============================================================
# PAST + FUTURE 150-BAR EXOTIC RETURNS
#
# Predictor:
#
#              P_t
# X_t = ---------------- - 1
#            P_{t-150}
#
#
# Target:
#
#            P_{t+150}
# Y_t = ---------------- - 1
#               P_t
#
# Returns expressed in basis points.
# ============================================================

def add_exotic_returns(
    results: pd.DataFrame,
) -> pd.DataFrame:

    results = results.copy()

    P = results[
        "exotic_price_150"
    ]

    # --------------------------------------------------------
    # PAST RETURN:
    # t-150 -> t
    #
    # Predictor uses only information known at t.
    # --------------------------------------------------------

    results[
        "past_exotic_return_150_bps"
    ] = (
        (
            P
            /
            P.shift(
                RETURN_HORIZON
            )
            - 1.0
        )
        * 10_000
    )

    # --------------------------------------------------------
    # FUTURE RETURN:
    # t -> t+150
    #
    # Forecast target.
    # --------------------------------------------------------

    results[
        "future_exotic_return_150_bps"
    ] = (
        (
            P.shift(
                -RETURN_HORIZON
            )
            /
            P
            - 1.0
        )
        * 10_000
    )

    return results


# ============================================================
# REGRESSION
#
# future exotic return(t -> t+150)
#
#       =
#
# alpha
# + beta * past exotic return(t-150 -> t)
# + error
# ============================================================

def run_exotic_return_regression(
    results: pd.DataFrame,
) -> dict:

    x_col = (
        "past_exotic_return_150_bps"
    )

    y_col = (
        "future_exotic_return_150_bps"
    )

    df = (
        results[
            [
                x_col,
                y_col,
            ]
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
        .copy()
    )

    if len(df) < 10:

        raise RuntimeError(
            "Not enough valid observations "
            "for regression."
        )

    X = sm.add_constant(
        df[
            x_col
        ]
    )

    y = df[
        y_col
    ]

    # --------------------------------------------------------
    # HAC standard errors
    #
    # Future 150-bar returns overlap heavily.
    # --------------------------------------------------------

    model = sm.OLS(
        y,
        X,
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags":
                RETURN_HORIZON
        },
    )

    return {
        "n":
            len(df),

        "alpha_bps":
            model.params[
                "const"
            ],

        "beta":
            model.params[
                x_col
            ],

        "beta_tstat":
            model.tvalues[
                x_col
            ],

        "beta_pvalue":
            model.pvalues[
                x_col
            ],

        "r_squared":
            model.rsquared,

        "model":
            model,

        "regression_data":
            df,
    }


# ============================================================
# PLOT EXOTIC PRICE
# ============================================================

def plot_exotic_price(
    results: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(
        figsize=(
            14,
            7,
        )
    )

    ax.plot(
        results[
            "cum_dollar_volume_mm"
        ],
        results[
            "exotic_price_150"
        ],
        linewidth=1.5,
        label=(
            "150-Bar Exotic Price"
        ),
    )

    ax.set_xlabel(
        "Cumulative Dollar Volume ($MM)"
    )

    ax.set_ylabel(
        "Price"
    )

    ax.set_title(
        "150-Bar Inverse-Spread-Weighted Exotic Price"
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
# SCATTER PLOT
#
# Past 150-bar exotic return
# vs
# Future 150-bar exotic return
# ============================================================

def plot_return_regression(
    regression_result: dict,
) -> None:

    df = regression_result[
        "regression_data"
    ]

    model = regression_result[
        "model"
    ]

    x_col = (
        "past_exotic_return_150_bps"
    )

    y_col = (
        "future_exotic_return_150_bps"
    )

    x_grid = np.linspace(
        df[
            x_col
        ].min(),
        df[
            x_col
        ].max(),
        300,
    )

    y_grid = (
        model.params[
            "const"
        ]
        +
        model.params[
            x_col
        ]
        * x_grid
    )

    fig, ax = plt.subplots(
        figsize=(
            9,
            7,
        )
    )

    ax.scatter(
        df[
            x_col
        ],
        df[
            y_col
        ],
        alpha=0.3,
        s=15,
    )

    ax.plot(
        x_grid,
        y_grid,
        linewidth=2.0,
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=0.8,
    )

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=0.8,
    )

    ax.set_xlabel(
        "Past 150-Bar Exotic Return "
        "(t-150 -> t, bps)"
    )

    ax.set_ylabel(
        "Future 150-Bar Exotic Return "
        "(t -> t+150, bps)"
    )

    ax.set_title(
        "150-Bar Exotic Price Momentum"
    )

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # BUILD DOLLAR-VOLUME BARS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RECONSTRUCT BOOK
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # BUILD 150-BAR EXOTIC PRICE
    # --------------------------------------------------------

    results = add_exotic_price(
        results
    )

    # --------------------------------------------------------
    # BUILD PAST/FUTURE RETURNS
    # --------------------------------------------------------

    results = add_exotic_returns(
        results
    )

    # --------------------------------------------------------
    # REGRESSION
    # --------------------------------------------------------

    regression_result = (
        run_exotic_return_regression(
            results
        )
    )

    print()
    print(
        "=" * 100
    )

    print(
        "150-BAR EXOTIC PRICE MOMENTUM REGRESSION"
    )

    print(
        "Future Exotic Return "
        "(t -> t+150) "
        "~ "
        "Past Exotic Return "
        "(t-150 -> t)"
    )

    print(
        "=" * 100
    )

    print(
        f"n                   : "
        f"{regression_result['n']}"
    )

    print(
        f"alpha_bps           : "
        f"{regression_result['alpha_bps']:.6f}"
    )

    print(
        f"beta                : "
        f"{regression_result['beta']:.6f}"
    )

    print(
        f"beta_tstat          : "
        f"{regression_result['beta_tstat']:.6f}"
    )

    print(
        f"beta_pvalue         : "
        f"{regression_result['beta_pvalue']:.6f}"
    )

    print(
        f"r_squared           : "
        f"{regression_result['r_squared']:.6f}"
    )

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    plot_exotic_price(
        results
    )

    plot_return_regression(
        regression_result
    )


if __name__ == "__main__":
    main()