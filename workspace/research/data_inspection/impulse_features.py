from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

TBBO_PATH = Path(
    "data/raw/tbbo/gld-2026-01-06.tbbo.csv"
)

DOLLAR_VOLUME_THRESHOLD = 1_000_000.0
CHUNK_SIZE = 500_000

# Smoothing in dollar-volume-bar space.
EWMA_HALFLIFE_BARS = 10

# Remove first and last 10%.
CHOP_FRACTION = 0.10


USECOLS = [
    "ts_event",
    "action",
    "side",
    "price",
    "size",
    "bid_px_00",
    "ask_px_00",
]


# ============================================================
# BUILD DOLLAR-VOLUME BARS
#
# side == "A":
#     seller aggressor -> bid hit
#
# side == "B":
#     buyer aggressor -> offer lifted
#
# Spread-weighted impulse:
#
#     CH = sum(q_i * S_i)
#     CL = sum(q_i * S_i)
#
# Volume:
#
#     VH = sum(q_i) for bids hit
#     VL = sum(q_i) for offers lifted
#
# Raw impulse imbalance:
#
#     I = (CL - CH) / (CL + CH)
#
# Raw volume imbalance:
#
#     VI = (VL - VH) / (VL + VH)
#
# Dollar-volume sampling:
#
# - threshold-crossing trade remains in current bar
# - no overshoot carry
# ============================================================

def build_impulse_bars(
    path: Path,
) -> pd.DataFrame:

    rows = []

    bar_dollar_volume = 0.0

    cost_hit = 0.0
    cost_lift = 0.0

    volume_hit = 0.0
    volume_lifted = 0.0

    bids_hit = 0
    offers_lifted = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            usecols=USECOLS,
            chunksize=CHUNK_SIZE,
            engine="c",
            memory_map=True,
        ),
        start=1,
    ):

        trades = chunk.loc[
            chunk["action"].eq("T")
        ]

        if trades.empty:
            continue

        for row in trades.itertuples(
            index=False
        ):

            trade_price = float(
                row.price
            )

            trade_size = float(
                row.size
            )

            if (
                not np.isfinite(trade_price)
                or not np.isfinite(trade_size)
                or trade_size <= 0
            ):
                continue

            # ------------------------------------------------
            # DOLLAR-VOLUME CLOCK
            # ------------------------------------------------

            trade_notional = (
                trade_price
                * trade_size
            )

            bar_dollar_volume += (
                trade_notional
            )

            # ------------------------------------------------
            # BBO IMMEDIATELY BEFORE TRADE
            # ------------------------------------------------

            bid = float(
                row.bid_px_00
            )

            ask = float(
                row.ask_px_00
            )

            valid_bbo = (
                np.isfinite(bid)
                and np.isfinite(ask)
                and bid > 0
                and ask > bid
            )

            if valid_bbo:

                mid = (
                    bid + ask
                ) / 2.0

                spread_bps = (
                    (ask - bid)
                    / mid
                    * 10_000.0
                )

                # q_i * S_i
                spread_weighted_size = (
                    trade_size
                    * spread_bps
                )

                # --------------------------------------------
                # SELLER AGGRESSOR -> BID HIT
                # --------------------------------------------

                if row.side == "A":

                    cost_hit += (
                        spread_weighted_size
                    )

                    volume_hit += (
                        trade_size
                    )

                    bids_hit += 1

                # --------------------------------------------
                # BUYER AGGRESSOR -> OFFER LIFTED
                # --------------------------------------------

                elif row.side == "B":

                    cost_lift += (
                        spread_weighted_size
                    )

                    volume_lifted += (
                        trade_size
                    )

                    offers_lifted += 1

            else:

                mid = np.nan

            # ------------------------------------------------
            # COMPLETE DOLLAR-VOLUME BAR
            # ------------------------------------------------

            if (
                bar_dollar_volume
                >= DOLLAR_VOLUME_THRESHOLD
            ):

                # --------------------------------------------
                # RAW IMPULSE IMBALANCE
                # --------------------------------------------

                total_cost = (
                    cost_hit
                    + cost_lift
                )

                if total_cost > 0:

                    impulse_imbalance = (
                        cost_lift
                        - cost_hit
                    ) / total_cost

                else:

                    impulse_imbalance = np.nan

                # --------------------------------------------
                # RAW VOLUME IMBALANCE
                # --------------------------------------------

                total_volume = (
                    volume_hit
                    + volume_lifted
                )

                if total_volume > 0:

                    volume_imbalance = (
                        volume_lifted
                        - volume_hit
                    ) / total_volume

                else:

                    volume_imbalance = np.nan

                # --------------------------------------------
                # STORE BAR
                # --------------------------------------------

                rows.append(
                    {
                        "bar":
                            len(rows),

                        "ts_event":
                            pd.Timestamp(
                                row.ts_event
                            ),

                        "dollar_volume":
                            bar_dollar_volume,

                        "mid_price":
                            mid,

                        "bids_hit":
                            bids_hit,

                        "offers_lifted":
                            offers_lifted,

                        "volume_hit":
                            volume_hit,

                        "volume_lifted":
                            volume_lifted,

                        "cost_hit":
                            cost_hit,

                        "cost_lift":
                            cost_lift,

                        "impulse_imbalance":
                            impulse_imbalance,

                        "volume_imbalance":
                            volume_imbalance,
                    }
                )

                # --------------------------------------------
                # RESET BAR
                # --------------------------------------------

                bar_dollar_volume = 0.0

                cost_hit = 0.0
                cost_lift = 0.0

                volume_hit = 0.0
                volume_lifted = 0.0

                bids_hit = 0
                offers_lifted = 0

        print(
            f"Chunk {chunk_number} | "
            f"bars {len(rows):,}"
        )

    if not rows:

        raise RuntimeError(
            "No completed dollar-volume bars found."
        )

    results = pd.DataFrame(
        rows
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
        / 1_000_000.0
    )

    return results


# ============================================================
# EWMA FEATURES
#
# IMPORTANT:
#
# Smooth underlying positive quantities first.
# Then calculate normalized imbalances.
#
# No cumulative features.
# ============================================================

def add_ewma_features(
    results: pd.DataFrame,
) -> pd.DataFrame:

    results = results.copy()

    # ========================================================
    # SPREAD-WEIGHTED IMPULSE
    # ========================================================

    results[
        "cost_hit_ewma"
    ] = (
        results[
            "cost_hit"
        ]
        .ewm(
            halflife=EWMA_HALFLIFE_BARS,
            adjust=False,
        )
        .mean()
    )

    results[
        "cost_lift_ewma"
    ] = (
        results[
            "cost_lift"
        ]
        .ewm(
            halflife=EWMA_HALFLIFE_BARS,
            adjust=False,
        )
        .mean()
    )

    total_impulse = (
        results[
            "cost_hit_ewma"
        ]
        +
        results[
            "cost_lift_ewma"
        ]
    )

    results[
        "impulse_imbalance_ewma"
    ] = np.where(
        total_impulse > 0,
        (
            results[
                "cost_lift_ewma"
            ]
            -
            results[
                "cost_hit_ewma"
            ]
        )
        /
        total_impulse,
        np.nan,
    )

    # ========================================================
    # SIDE-SPECIFIC AGGRESSIVE VOLUME
    # ========================================================

    results[
        "volume_hit_ewma"
    ] = (
        results[
            "volume_hit"
        ]
        .ewm(
            halflife=EWMA_HALFLIFE_BARS,
            adjust=False,
        )
        .mean()
    )

    results[
        "volume_lifted_ewma"
    ] = (
        results[
            "volume_lifted"
        ]
        .ewm(
            halflife=EWMA_HALFLIFE_BARS,
            adjust=False,
        )
        .mean()
    )

    total_volume = (
        results[
            "volume_hit_ewma"
        ]
        +
        results[
            "volume_lifted_ewma"
        ]
    )

    results[
        "volume_imbalance_ewma"
    ] = np.where(
        total_volume > 0,
        (
            results[
                "volume_lifted_ewma"
            ]
            -
            results[
                "volume_hit_ewma"
            ]
        )
        /
        total_volume,
        np.nan,
    )

    return results


# ============================================================
# FIGURE 1
#
# 1. IMPULSE IMBALANCE
# 2. VOLUME IMBALANCE
# 3. MID PRICE
# ============================================================

def plot_flow_features(
    results: pd.DataFrame,
) -> None:

    x = results[
        "cum_dollar_volume_mm"
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(
            11,
            6,
        ),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                1,
                1,
                1.15,
            ],
        },
    )

    # --------------------------------------------------------
    # 1. IMPULSE IMBALANCE
    # --------------------------------------------------------

    axes[0].plot(
        x,
        results[
            "impulse_imbalance_ewma"
        ],
        linewidth=1.0,
    )

    axes[0].axhline(
        0.0,
        linestyle="--",
        linewidth=0.8,
    )

    axes[0].set_ylim(
        -1,
        1,
    )

    axes[0].set_ylabel(
        "Impulse\nImbalance"
    )

    axes[0].set_title(
        "Aggressive Trade Flow "
        f"(EWMA halflife = "
        f"{EWMA_HALFLIFE_BARS} bars)"
    )

    axes[0].grid(
        alpha=0.3
    )

    # --------------------------------------------------------
    # 2. VOLUME IMBALANCE
    # --------------------------------------------------------

    axes[1].plot(
        x,
        results[
            "volume_imbalance_ewma"
        ],
        linewidth=1.0,
    )

    axes[1].axhline(
        0.0,
        linestyle="--",
        linewidth=0.8,
    )

    axes[1].set_ylim(
        -1,
        1,
    )

    axes[1].set_ylabel(
        "Volume\nImbalance"
    )

    axes[1].grid(
        alpha=0.3
    )

    # --------------------------------------------------------
    # 3. MID PRICE
    # --------------------------------------------------------

    axes[2].plot(
        x,
        results[
            "mid_price"
        ],
        linewidth=1.0,
    )

    axes[2].set_ylabel(
        "Mid Price"
    )

    axes[2].set_xlabel(
        "Cumulative Dollar Volume ($MM)"
    )

    axes[2].grid(
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# FIGURE 2
#
# SIDE-SPECIFIC AGGRESSIVE VOLUME
#
# 1. VOLUME HIT
# 2. VOLUME LIFTED
# 3. VOLUME IMBALANCE
# 4. MID PRICE
# ============================================================

def plot_side_specific_volumes(
    results: pd.DataFrame,
) -> None:

    x = results[
        "cum_dollar_volume_mm"
    ]

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(
            11,
            7,
        ),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                1,
                1,
                1,
                1.15,
            ],
        },
    )

    # --------------------------------------------------------
    # 1. VOLUME HIT
    #
    # Aggressive selling.
    # --------------------------------------------------------

    axes[0].plot(
        x,
        results[
            "volume_hit_ewma"
        ],
        linewidth=1.0,
    )

    axes[0].set_ylabel(
        "Volume Hit"
    )

    axes[0].set_title(
        "Side-Specific Aggressive Volume "
        f"(EWMA halflife = "
        f"{EWMA_HALFLIFE_BARS} bars)"
    )

    axes[0].grid(
        alpha=0.3
    )

    # --------------------------------------------------------
    # 2. VOLUME LIFTED
    #
    # Aggressive buying.
    # --------------------------------------------------------

    axes[1].plot(
        x,
        results[
            "volume_lifted_ewma"
        ],
        linewidth=1.0,
    )

    axes[1].set_ylabel(
        "Volume Lifted"
    )

    axes[1].grid(
        alpha=0.3
    )

    # --------------------------------------------------------
    # 3. VOLUME IMBALANCE
    # --------------------------------------------------------

    axes[2].plot(
        x,
        results[
            "volume_imbalance_ewma"
        ],
        linewidth=1.0,
    )

    axes[2].axhline(
        0.0,
        linestyle="--",
        linewidth=0.8,
    )

    axes[2].set_ylim(
        -1,
        1,
    )

    axes[2].set_ylabel(
        "Volume\nImbalance"
    )

    axes[2].grid(
        alpha=0.3
    )

    # --------------------------------------------------------
    # 4. MID PRICE
    # --------------------------------------------------------

    axes[3].plot(
        x,
        results[
            "mid_price"
        ],
        linewidth=1.0,
    )

    axes[3].set_ylabel(
        "Mid Price"
    )

    axes[3].set_xlabel(
        "Cumulative Dollar Volume ($MM)"
    )

    axes[3].grid(
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

    results = build_impulse_bars(
        TBBO_PATH
    )

    print(
        f"Raw bars: "
        f"{len(results):,}"
    )

    # --------------------------------------------------------
    # EWMA FEATURES
    #
    # Calculate BEFORE chop so the first retained values
    # have proper smoothing history.
    # --------------------------------------------------------

    results = add_ewma_features(
        results
    )

    # ========================================================
    # CHOP FIRST / LAST 10%
    # ========================================================

    chop = int(
        len(results)
        * CHOP_FRACTION
    )

    if chop > 0:

        results = (
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
        f"Chop per side: "
        f"{chop:,}"
    )

    print(
        f"Analysis bars: "
        f"{len(results):,}"
    )

    # --------------------------------------------------------
    # PRINT SAMPLE
    # --------------------------------------------------------

    print()

    print(
        "=" * 180
    )

    print(
        "AGGRESSIVE TRADE FLOW FEATURES"
    )

    print(
        "=" * 180
    )

    print(
        results[
            [
                "bar",
                "cum_dollar_volume_mm",
                "mid_price",

                "cost_hit",
                "cost_lift",

                "impulse_imbalance",
                "impulse_imbalance_ewma",

                "volume_hit",
                "volume_lifted",

                "volume_hit_ewma",
                "volume_lifted_ewma",

                "volume_imbalance",
                "volume_imbalance_ewma",
            ]
        ]
        .tail(20)
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # ========================================================
    # PLOTS
    # ========================================================

    # Impulse imbalance + volume imbalance + price.
    plot_flow_features(
        results
    )

    # Individual aggressive volumes + volume imbalance + price.
    plot_side_specific_volumes(
        results
    )


if __name__ == "__main__":
    main()