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
    "data/raw/mbo/gld-2026-01-06.mbo.csv"
)

DOLLAR_VOLUME_THRESHOLD = 1_000_000.0
MBO_CHUNK_SIZE = 1_000_000

BOOK_LEVELS = 10

EWMA_HALFLIFE_BARS = 10
EWMA_MIN_PERIODS = 10

HORIZONS = [
    1,
    5,
    10,
    25,
    50,
]

HEATMAP_HORIZON = 25
N_DECILES = 10


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
#
# - only T contributes to traded dollar volume
# - F is ignored
# - threshold-crossing trade stays in current bar
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

def get_level_size(
    level_orders,
) -> int:

    return sum(
        int(order.size)
        for order
        in level_orders.orders
    )


def volume_weighted_first_k_levels(
    book: Book,
    side: str,
    k: int,
) -> float:

    if side == "B":

        levels = book.bids

        if not levels:
            return math.nan

        prices = (
            levels.peekitem(i)[0]
            for i
            in range(
                -1,
                -min(
                    k,
                    len(levels)
                ) - 1,
                -1,
            )
        )

    elif side == "A":

        levels = book.offers

        if not levels:
            return math.nan

        prices = (
            levels.peekitem(i)[0]
            for i
            in range(
                min(
                    k,
                    len(levels)
                )
            )
        )

    else:

        raise ValueError(
            "side must be 'B' or 'A'"
        )

    weighted_price_sum = 0
    total_size = 0

    for price in prices:

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

        total_size += (
            size
        )

    if total_size <= 0:
        return math.nan

    return float(
        weighted_price_sum
        / total_size
        / db.FIXED_PRICE_SCALE
    )


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

            "microprice":
                math.nan,

            "book_mid":
                math.nan,
        }

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

    # --------------------------------------------------------
    # TOUCH MID
    # --------------------------------------------------------

    touch_mid = (
        bid + ask
    ) / 2.0

    # --------------------------------------------------------
    # MICROPRICE
    # --------------------------------------------------------

    top_depth = (
        bid_size
        + ask_size
    )

    if top_depth > 0:

        microprice = (
            ask * bid_size
            + bid * ask_size
        ) / top_depth

    else:

        microprice = math.nan

    # --------------------------------------------------------
    # BOOK MID
    # --------------------------------------------------------

    book_bid = (
        volume_weighted_first_k_levels(
            book,
            "B",
            BOOK_LEVELS,
        )
    )

    book_ask = (
        volume_weighted_first_k_levels(
            book,
            "A",
            BOOK_LEVELS,
        )
    )

    if (
        math.isfinite(book_bid)
        and math.isfinite(book_ask)
    ):

        book_mid = (
            book_bid
            + book_ask
        ) / 2.0

    else:

        book_mid = math.nan

    return {
        "touch_mid":
            touch_mid,

        "microprice":
            microprice,

        "book_mid":
            book_mid,
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
# RETURNS + EWMA VOLATILITY
# ============================================================

def add_returns_and_ewma_vol(
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

    price_cols = [
        "touch_mid",
        "microprice",
        "book_mid",
    ]

    for col in price_cols:

        ret_col = (
            f"{col}_return_bps"
        )

        vol_col = (
            f"{col}_ewma_vol_bps"
        )

        results[
            ret_col
        ] = (
            results[
                col
            ]
            .pct_change(
                fill_method=None
            )
            * 10_000
        )

        results[
            vol_col
        ] = np.sqrt(
            results[
                ret_col
            ]
            .pow(2)
            .ewm(
                halflife=EWMA_HALFLIFE_BARS,
                adjust=False,
                min_periods=EWMA_MIN_PERIODS,
            )
            .mean()
        )

    # --------------------------------------------------------
    # BOOK / TOUCH VOL RATIO
    # --------------------------------------------------------

    results[
        "book_to_touch_vol_ratio"
    ] = np.where(
        results[
            "touch_mid_ewma_vol_bps"
        ] > 0,
        (
            results[
                "book_mid_ewma_vol_bps"
            ]
            /
            results[
                "touch_mid_ewma_vol_bps"
            ]
        ),
        np.nan,
    )

    return results


# ============================================================
# FUTURE TOUCH VOLATILITY
# ============================================================

def add_future_vol_targets(
    results: pd.DataFrame,
) -> pd.DataFrame:

    results = results.copy()

    for h in HORIZONS:

        results[
            f"future_touch_vol_{h}"
        ] = (
            results[
                "touch_mid_ewma_vol_bps"
            ]
            .shift(
                -h
            )
        )

    return results


# ============================================================
# SINGLE OLS REGRESSION
# ============================================================

def run_regression(
    results: pd.DataFrame,
    x_col: str,
    y_col: str,
    hac_lags: int,
) -> dict:

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
    )

    X = sm.add_constant(
        df[
            x_col
        ]
    )

    y = df[
        y_col
    ]

    model = sm.OLS(
        y,
        X,
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags":
                hac_lags
        },
    )

    return {
        "n":
            len(df),

        "alpha":
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
    }


# ============================================================
# MULTIPLE OLS REGRESSION
# ============================================================

def run_multi_regression(
    results: pd.DataFrame,
    x_cols: list[str],
    y_col: str,
    hac_lags: int,
) -> dict:

    cols = (
        x_cols
        + [y_col]
    )

    df = (
        results[
            cols
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    X = sm.add_constant(
        df[
            x_cols
        ]
    )

    y = df[
        y_col
    ]

    model = sm.OLS(
        y,
        X,
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags":
                hac_lags
        },
    )

    return {
        "n":
            len(df),

        "alpha":
            model.params[
                "const"
            ],

        "betas": {
            col:
                model.params[col]
            for col
            in x_cols
        },

        "beta_tstats": {
            col:
                model.tvalues[col]
            for col
            in x_cols
        },

        "beta_pvalues": {
            col:
                model.pvalues[col]
            for col
            in x_cols
        },

        "r_squared":
            model.rsquared,

        "adj_r_squared":
            model.rsquared_adj,

        "model":
            model,
    }


# ============================================================
# VOLATILITY REGRESSIONS
# ============================================================

def run_vol_regressions(
    results: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for h in HORIZONS:

        y_col = (
            f"future_touch_vol_{h}"
        )

        hac_lags = max(
            10,
            h,
        )

        # ----------------------------------------------------
        # TOUCH ONLY
        # ----------------------------------------------------

        mid_reg = run_regression(
            results=results,
            x_col=(
                "touch_mid_ewma_vol_bps"
            ),
            y_col=y_col,
            hac_lags=hac_lags,
        )

        rows.append(
            {
                "horizon":
                    h,

                "regression":
                    (
                        "Future Touch Vol "
                        "~ Current Touch Vol"
                    ),

                **mid_reg,
            }
        )

        # ----------------------------------------------------
        # TOUCH + BOOK
        # ----------------------------------------------------

        multi_reg = (
            run_multi_regression(
                results=results,
                x_cols=[
                    "touch_mid_ewma_vol_bps",
                    "book_mid_ewma_vol_bps",
                ],
                y_col=y_col,
                hac_lags=hac_lags,
            )
        )

        rows.append(
            {
                "horizon":
                    h,

                "regression":
                    (
                        "Future Touch Vol "
                        "~ Current Touch Vol "
                        "+ Current Book Vol"
                    ),

                "n":
                    multi_reg["n"],

                "alpha":
                    multi_reg["alpha"],

                "touch_beta":
                    multi_reg[
                        "betas"
                    ][
                        "touch_mid_ewma_vol_bps"
                    ],

                "touch_tstat":
                    multi_reg[
                        "beta_tstats"
                    ][
                        "touch_mid_ewma_vol_bps"
                    ],

                "touch_pvalue":
                    multi_reg[
                        "beta_pvalues"
                    ][
                        "touch_mid_ewma_vol_bps"
                    ],

                "book_beta":
                    multi_reg[
                        "betas"
                    ][
                        "book_mid_ewma_vol_bps"
                    ],

                "book_tstat":
                    multi_reg[
                        "beta_tstats"
                    ][
                        "book_mid_ewma_vol_bps"
                    ],

                "book_pvalue":
                    multi_reg[
                        "beta_pvalues"
                    ][
                        "book_mid_ewma_vol_bps"
                    ],

                "r_squared":
                    multi_reg[
                        "r_squared"
                    ],

                "adj_r_squared":
                    multi_reg[
                        "adj_r_squared"
                    ],
            }
        )

        # ----------------------------------------------------
        # BOOK ONLY
        # ----------------------------------------------------

        book_reg = run_regression(
            results=results,
            x_col=(
                "book_mid_ewma_vol_bps"
            ),
            y_col=y_col,
            hac_lags=hac_lags,
        )

        rows.append(
            {
                "horizon":
                    h,

                "regression":
                    (
                        "Future Touch Vol "
                        "~ Current Book Vol"
                    ),

                **book_reg,
            }
        )

        # ----------------------------------------------------
        # RATIO ONLY
        # ----------------------------------------------------

        ratio_reg = run_regression(
            results=results,
            x_col=(
                "book_to_touch_vol_ratio"
            ),
            y_col=y_col,
            hac_lags=hac_lags,
        )

        rows.append(
            {
                "horizon":
                    h,

                "regression":
                    (
                        "Future Touch Vol "
                        "~ Current Book/Touch Ratio"
                    ),

                **ratio_reg,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# VOLATILITY + MID PRICE PLOT
# ============================================================

def plot_ewma_volatility(
    results: pd.DataFrame,
) -> None:

    x = results[
        "cum_dollar_volume_mm"
    ]

    fig, (
        ax1,
        ax2,
    ) = plt.subplots(
        2,
        1,
        figsize=(
            11,
            6,
        ),
        sharex=True,
        gridspec_kw={
            "height_ratios":
                [
                    2,
                    1,
                ],
        },
    )

    ax1.plot(
        x,
        results[
            "touch_mid_ewma_vol_bps"
        ],
        linewidth=1.1,
        label="Touch mid",
    )

    ax1.plot(
        x,
        results[
            "microprice_ewma_vol_bps"
        ],
        linewidth=1.1,
        label="Microprice",
    )

    ax1.plot(
        x,
        results[
            "book_mid_ewma_vol_bps"
        ],
        linewidth=1.1,
        label=(
            f"Book mid "
            f"({BOOK_LEVELS} levels)"
        ),
    )

    ax1.set_ylabel(
        "EWMA Vol (bps/bar)"
    )

    ax1.set_title(
        "EWMA Volatility of Price Returns "
        f"(halflife = "
        f"{EWMA_HALFLIFE_BARS} "
        "dollar-volume bars)"
    )

    ax1.grid(
        alpha=0.3
    )

    ax1.legend(
        loc="best",
        fontsize=9,
    )

    ax2.plot(
        x,
        results[
            "touch_mid"
        ],
        linewidth=1.0,
    )

    ax2.set_xlabel(
        "Cumulative Dollar Volume ($MM)"
    )

    ax2.set_ylabel(
        "Mid Price"
    )

    ax2.grid(
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# BOOK / TOUCH VOL RATIO + MID PRICE
# ============================================================

def plot_vol_ratio(
    results: pd.DataFrame,
) -> None:

    x = results[
        "cum_dollar_volume_mm"
    ]

    fig, (
        ax1,
        ax2,
    ) = plt.subplots(
        2,
        1,
        figsize=(
            11,
            6,
        ),
        sharex=True,
        gridspec_kw={
            "height_ratios":
                [
                    2,
                    1,
                ],
        },
    )

    ax1.plot(
        x,
        results[
            "book_to_touch_vol_ratio"
        ],
        linewidth=1.1,
        label=(
            "Book to Touch Vol Ratio"
        ),
    )

    ax1.axhline(
        1.0,
        linestyle="--",
        linewidth=0.8,
    )

    ax1.set_ylabel(
        "Book to Touch Vol Ratio"
    )

    ax1.set_title(
        "Ratio of Book Vol EWMA "
        "to Touch Vol EWMA"
    )

    ax1.grid(
        alpha=0.3
    )

    ax1.legend(
        loc="best"
    )

    ax2.plot(
        x,
        results[
            "touch_mid"
        ],
        linewidth=1.0,
    )

    ax2.set_xlabel(
        "Cumulative Dollar Volume ($MM)"
    )

    ax2.set_ylabel(
        "Mid Price"
    )

    ax2.grid(
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# DECILE HEAT MAP
#
# x-axis:
#     current touch volatility decile
#
# y-axis:
#     current book volatility decile
#
# color:
#     mean touch volatility 25 bars ahead
# ============================================================

def plot_future_vol_heatmap(
    results: pd.DataFrame,
    horizon: int = HEATMAP_HORIZON,
    n_deciles: int = N_DECILES,
) -> None:

    target_col = (
        f"future_touch_vol_{horizon}"
    )

    df = (
        results[
            [
                "touch_mid_ewma_vol_bps",
                "book_mid_ewma_vol_bps",
                target_col,
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

    # --------------------------------------------------------
    # DECILES
    # --------------------------------------------------------

    df[
        "touch_vol_decile"
    ] = (
        pd.qcut(
            df[
                "touch_mid_ewma_vol_bps"
            ],
            q=n_deciles,
            labels=False,
            duplicates="drop",
        )
        + 1
    )

    df[
        "book_vol_decile"
    ] = (
        pd.qcut(
            df[
                "book_mid_ewma_vol_bps"
            ],
            q=n_deciles,
            labels=False,
            duplicates="drop",
        )
        + 1
    )

    # --------------------------------------------------------
    # CELL MEAN
    # --------------------------------------------------------

    heatmap = df.pivot_table(
        index="book_vol_decile",
        columns="touch_vol_decile",
        values=target_col,
        aggfunc="mean",
    )

    # --------------------------------------------------------
    # CELL COUNTS
    # --------------------------------------------------------

    counts = df.pivot_table(
        index="book_vol_decile",
        columns="touch_vol_decile",
        values=target_col,
        aggfunc="count",
    )

    # --------------------------------------------------------
    # REINDEX SO 1 -> 10 ALWAYS APPEARS IN ORDER
    # --------------------------------------------------------

    deciles = list(
        range(
            1,
            n_deciles + 1,
        )
    )

    heatmap = (
        heatmap
        .reindex(
            index=deciles,
            columns=deciles,
        )
    )

    counts = (
        counts
        .reindex(
            index=deciles,
            columns=deciles,
        )
    )

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            9,
            8,
        )
    )

    im = ax.imshow(
        heatmap.values,
        origin="lower",
        aspect="auto",
    )

    ax.set_xticks(
        np.arange(
            n_deciles
        )
    )

    ax.set_xticklabels(
        deciles
    )

    ax.set_yticks(
        np.arange(
            n_deciles
        )
    )

    ax.set_yticklabels(
        deciles
    )

    ax.set_xlabel(
        "Current Touch Vol Decile"
    )

    ax.set_ylabel(
        "Current Book Vol Decile"
    )

    ax.set_title(
        f"Mean Touch Vol "
        f"{horizon} Dollar-Volume Bars Ahead"
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
    )

    cbar.set_label(
        "Future Touch Vol (bps/bar)"
    )

    # --------------------------------------------------------
    # CELL LABELS
    #
    # First line = future touch vol
    # Second line = number of observations
    # --------------------------------------------------------

    for i in range(
        n_deciles
    ):

        for j in range(
            n_deciles
        ):

            value = (
                heatmap.iloc[
                    i,
                    j,
                ]
            )

            count = (
                counts.iloc[
                    i,
                    j,
                ]
            )

            if pd.notna(
                value
            ):

                ax.text(
                    j,
                    i,
                    (
                        f"{value:.2f}\n"
                        f"n={int(count)}"
                    ),
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # PRINT TABLE TOO
    # --------------------------------------------------------

    print()

    print(
        "=" * 120
    )

    print(
        f"MEAN TOUCH VOL "
        f"{horizon} BARS AHEAD"
    )

    print(
        "ROWS = CURRENT BOOK VOL DECILE"
    )

    print(
        "COLUMNS = CURRENT TOUCH VOL DECILE"
    )

    print(
        "=" * 120
    )

    print(
        heatmap.to_string(
            float_format=lambda x:
                f"{x:.3f}"
        )
    )

    print()

    print(
        "=" * 120
    )

    print(
        "OBSERVATION COUNTS"
    )

    print(
        "=" * 120
    )

    print(
        counts.fillna(
            0
        )
        .astype(int)
        .to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # BUILD DOLLAR-VOLUME BOUNDARIES
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
    # REPLAY BOOK
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
    # RETURNS + EWMA VOL
    #
    # Calculate these using the whole reconstructed history.
    # --------------------------------------------------------

    results = (
        add_returns_and_ewma_vol(
            results
        )
    )

    # --------------------------------------------------------
    # CHOP
    #
    # Remove first and last 10% because the deeper book is
    # contaminated / too thin there.
    # --------------------------------------------------------

    chop = int(
        len(results)
        / 10
    )

    analysis_results = (
        results
        .iloc[
            chop:-chop
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print(
        f"Raw bars: "
        f"{len(results):,}"
    )

    print(
        f"Chop per side: "
        f"{chop:,}"
    )

    print(
        f"Analysis bars: "
        f"{len(analysis_results):,}"
    )

    # --------------------------------------------------------
    # FUTURE VOL TARGETS
    #
    # Build AFTER chopping so future targets do not reach
    # into the contaminated end region.
    # --------------------------------------------------------

    analysis_results = (
        add_future_vol_targets(
            analysis_results
        )
    )

    # --------------------------------------------------------
    # PRINT PRICE / VOL FEATURES
    # --------------------------------------------------------

    print()

    print(
        "=" * 180
    )

    print(
        "PRICE AND VOLATILITY FEATURES"
    )

    print(
        "=" * 180
    )

    print(
        analysis_results[
            [
                "bar",
                "cum_dollar_volume_mm",

                "touch_mid",
                "microprice",
                "book_mid",

                "touch_mid_return_bps",
                "microprice_return_bps",
                "book_mid_return_bps",

                "touch_mid_ewma_vol_bps",
                "microprice_ewma_vol_bps",
                "book_mid_ewma_vol_bps",

                "book_to_touch_vol_ratio",
            ]
        ]
        .tail(20)
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # ORIGINAL VOLATILITY PLOTS
    # --------------------------------------------------------

    plot_ewma_volatility(
        analysis_results
    )

    plot_vol_ratio(
        analysis_results
    )

    # --------------------------------------------------------
    # REGRESSIONS
    #
    # Print results only.
    # No regression plots.
    # --------------------------------------------------------

    regression_summary = (
        run_vol_regressions(
            analysis_results
        )
    )

    print()

    print(
        "=" * 180
    )

    print(
        "FUTURE TOUCH VOL REGRESSIONS"
    )

    print(
        "=" * 180
    )

    print(
        regression_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # HEAT MAP
    #
    # X = current touch-vol decile
    # Y = current book-vol decile
    # Color = mean touch vol 25 bars ahead
    # --------------------------------------------------------

    plot_future_vol_heatmap(
        analysis_results,
        horizon=HEATMAP_HORIZON,
        n_deciles=N_DECILES,
    )


if __name__ == "__main__":
    main()