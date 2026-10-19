import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "data/raw/tbbo/gld-2026-01-29.tbbo.csv"
DOLLAR_VOLUME_THRESHOLD = 2_000_000


# --------------------------------------------------
# Load and prepare data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["ts_event"] = pd.to_datetime(
    df["ts_event"],
    utc=True
)

df = (
    df
    .sort_values("ts_event")
    .reset_index(drop=True)
)


# --------------------------------------------------
# Market-state variables from pre-trade BBO
# --------------------------------------------------

# Midpoint
df["mid"] = (
    df["bid_px_00"]
    + df["ask_px_00"]
) / 2


# Bid-ask spread
df["spread"] = (
    df["ask_px_00"]
    - df["bid_px_00"]
)


# Microprice
depth_sum = (
    df["bid_sz_00"]
    + df["ask_sz_00"]
)

df["microprice"] = (
    (
        df["ask_px_00"] * df["bid_sz_00"]
        + df["bid_px_00"] * df["ask_sz_00"]
    )
    / depth_sum.where(depth_sum > 0)
)


# Microprice deviation from midpoint
df["microprice_minus_mid"] = (
    df["microprice"]
    - df["mid"]
)


# User convention:
#
# positive = more ask liquidity
# negative = more bid liquidity
df["book_imbalance"] = (
    df["ask_sz_00"]
    - df["bid_sz_00"]
)


# --------------------------------------------------
# Trade data
# --------------------------------------------------

trade_mask = (
    df["action"] == "T"
)

trades = df.loc[
    trade_mask
].copy()


# Dollar volume
df["trade_dollar_volume"] = 0.0

df.loc[
    trade_mask,
    "trade_dollar_volume"
] = (
    df.loc[trade_mask, "price"]
    * df.loc[trade_mask, "size"]
)


# --------------------------------------------------
# Global trade-size statistics
# --------------------------------------------------

GLOBAL_MEDIAN_TRADE_SIZE = (
    trades["size"].median()
)

TRADE_SIZE_90TH_PERCENTILE = (
    trades["size"].quantile(0.90)
)


print(
    f"Global median trade size: "
    f"{GLOBAL_MEDIAN_TRADE_SIZE:,.2f} shares"
)

print(
    f"90th percentile trade size: "
    f"{TRADE_SIZE_90TH_PERCENTILE:,.2f} shares"
)


# --------------------------------------------------
# Build dollar-volume bars
# --------------------------------------------------

bars = []


# Total volume clock
total_volume = 0.0


# Original OFI
buy_volume = 0.0
sell_volume = 0.0


# OFI after removing top 10% by trade size
filtered_buy_volume = 0.0
filtered_sell_volume = 0.0


# Trade-size observations within current bar
trade_sizes = []


# Optional diagnostics
unknown_volume = 0.0


for row in df.itertuples():

    # Only trades contribute to volume bars
    if row.action != "T":
        continue


    dollar_volume = (
        row.trade_dollar_volume
    )


    # --------------------------------------------------
    # All-trade statistics
    # --------------------------------------------------

    total_volume += dollar_volume

    trade_sizes.append(
        row.size
    )


    # --------------------------------------------------
    # Original aggressive OFI
    # --------------------------------------------------

    if row.side == "B":

        buy_volume += (
            dollar_volume
        )


    elif row.side == "A":

        sell_volume += (
            dollar_volume
        )


    else:

        unknown_volume += (
            dollar_volume
        )


    # --------------------------------------------------
    # Filtered OFI
    #
    # Remove trades above global 90th percentile
    # --------------------------------------------------

    if row.size <= TRADE_SIZE_90TH_PERCENTILE:

        if row.side == "B":

            filtered_buy_volume += (
                dollar_volume
            )


        elif row.side == "A":

            filtered_sell_volume += (
                dollar_volume
            )


    # --------------------------------------------------
    # Close dollar-volume bar
    # --------------------------------------------------

    if (
        total_volume
        >= DOLLAR_VOLUME_THRESHOLD
    ):

        # ------------------------------
        # Original OFI
        # ------------------------------

        known_side_volume = (
            buy_volume
            + sell_volume
        )

        ofi = (
            (
                buy_volume
                - sell_volume
            )
            / known_side_volume

            if known_side_volume > 0

            else 0.0
        )


        # ------------------------------
        # Filtered OFI
        # ------------------------------

        filtered_known_volume = (
            filtered_buy_volume
            + filtered_sell_volume
        )

        filtered_ofi = (
            (
                filtered_buy_volume
                - filtered_sell_volume
            )
            / filtered_known_volume

            if filtered_known_volume > 0

            else 0.0
        )


        # ------------------------------
        # Median trade size inside bar
        # ------------------------------

        median_trade_size = (
            pd.Series(
                trade_sizes
            ).median()
            if trade_sizes
            else float("nan")
        )


        bars.append({

            # Timing
            "ts_event":
                row.ts_event,


            # Pre-trade market state
            "mid":
                row.mid,

            "microprice":
                row.microprice,

            "microprice_minus_mid":
                row.microprice_minus_mid,

            "spread":
                row.spread,

            "book_imbalance":
                row.book_imbalance,

            "bid_size":
                row.bid_sz_00,

            "ask_size":
                row.ask_sz_00,


            # Bar volume
            "dollar_volume":
                total_volume,


            # Original aggressive flow
            "buy_volume":
                buy_volume,

            "sell_volume":
                sell_volume,

            "unknown_volume":
                unknown_volume,

            "ofi":
                ofi,


            # Filtered aggressive flow
            "filtered_buy_volume":
                filtered_buy_volume,

            "filtered_sell_volume":
                filtered_sell_volume,

            "ofi_ex_top10":
                filtered_ofi,


            # Trade-size statistics
            "median_trade_size":
                median_trade_size,

            "trade_count":
                len(trade_sizes),
        })


        # --------------------------------------------------
        # Reset bar
        #
        # Threshold-crossing trade remains entirely
        # inside the completed bar.
        # --------------------------------------------------

        total_volume = 0.0

        buy_volume = 0.0
        sell_volume = 0.0

        filtered_buy_volume = 0.0
        filtered_sell_volume = 0.0

        unknown_volume = 0.0

        trade_sizes = []


# --------------------------------------------------
# Convert to DataFrame
# --------------------------------------------------

bars = pd.DataFrame(
    bars
)


if bars.empty:

    raise ValueError(
        "No completed dollar-volume bars were created."
    )


# --------------------------------------------------
# Cumulative activity-time axis
# --------------------------------------------------

bars["cum_dollar_volume"] = (
    bars["dollar_volume"]
    .cumsum()
)

bars["cum_dollar_volume_mm"] = (
    bars["cum_dollar_volume"]
    / 1_000_000
)


threshold_mm = (
    DOLLAR_VOLUME_THRESHOLD
    / 1_000_000
)


# ==================================================
# TIME-AXIS PLOTS
# ==================================================


# --------------------------------------------------
# 1. Midpoint
# --------------------------------------------------

plt.figure(figsize=(9, 4))

plt.plot(
    bars["ts_event"],
    bars["mid"],
    linewidth=1,
)

plt.title(
    f"GLD Mid Price — "
    f"${threshold_mm:g}MM Dollar Volume Bars"
)

plt.xlabel("Time")
plt.ylabel("Mid Price")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 2. Microprice
# --------------------------------------------------

plt.figure(figsize=(9, 4))

plt.plot(
    bars["ts_event"],
    bars["microprice"],
    linewidth=1,
)

plt.title(
    f"GLD Microprice — "
    f"${threshold_mm:g}MM Dollar Volume Bars"
)

plt.xlabel("Time")
plt.ylabel("Microprice")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 3. Midpoint vs microprice
# --------------------------------------------------

plt.figure(figsize=(9, 4))

plt.plot(
    bars["ts_event"],
    bars["mid"],
    linewidth=1,
    label="Mid",
)

plt.plot(
    bars["ts_event"],
    bars["microprice"],
    linewidth=1,
    label="Microprice",
)

plt.title(
    f"GLD Mid vs Microprice — "
    f"${threshold_mm:g}MM Bars"
)

plt.xlabel("Time")
plt.ylabel("Price")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 4. Microprice - midpoint
# --------------------------------------------------

plt.figure(figsize=(9, 4))

plt.plot(
    bars["ts_event"],
    bars["microprice_minus_mid"],
    linewidth=1,
)

plt.axhline(
    0,
    linewidth=0.8,
)

plt.title(
    "GLD Microprice - Midpoint"
)

plt.xlabel("Time")
plt.ylabel("Microprice - Mid ($)")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 5. Original OFI
# --------------------------------------------------

plt.figure(figsize=(9, 4))

plt.plot(
    bars["ts_event"],
    bars["ofi"],
    linewidth=1,
)

plt.axhline(
    0,
    linewidth=0.8,
)

plt.title(
    "GLD Aggressive Dollar OFI"
)

plt.xlabel("Time")
plt.ylabel("OFI")

plt.ylim(-1, 1)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 6. OFI excluding top 10% of trades by size
# --------------------------------------------------

plt.figure(figsize=(9, 4))

plt.plot(
    bars["ts_event"],
    bars["ofi_ex_top10"],
    linewidth=1,
)

plt.axhline(
    0,
    linewidth=0.8,
)

plt.title(
    "GLD Aggressive Dollar OFI "
    "Excluding Top 10% Trade Sizes"
)

plt.xlabel("Time")
plt.ylabel("Filtered OFI")

plt.ylim(-1, 1)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 7. Original OFI vs filtered OFI
# --------------------------------------------------

plt.figure(figsize=(9, 4))

plt.plot(
    bars["ts_event"],
    bars["ofi"],
    linewidth=1,
    label="All Trades",
)

plt.plot(
    bars["ts_event"],
    bars["ofi_ex_top10"],
    linewidth=1,
    label="Ex Top 10% Trade Size",
)

plt.axhline(
    0,
    linewidth=0.8,
)

plt.title(
    "GLD OFI — All Trades vs Ex-Large Trades"
)

plt.xlabel("Time")
plt.ylabel("OFI")

plt.ylim(-1, 1)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 8. Bid-ask spread
# --------------------------------------------------

plt.figure(figsize=(9, 4))

plt.plot(
    bars["ts_event"],
    bars["spread"],
    linewidth=1,
)

plt.title(
    "GLD Bid-Ask Spread"
)

plt.xlabel("Time")
plt.ylabel("Spread ($)")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 9. Book imbalance
# --------------------------------------------------

plt.figure(figsize=(9, 4))

plt.plot(
    bars["ts_event"],
    bars["book_imbalance"],
    linewidth=1,
)

plt.axhline(
    0,
    linewidth=0.8,
)

plt.title(
    "GLD Order Book Imbalance — "
    "Ask Size - Bid Size"
)

plt.xlabel("Time")
plt.ylabel("Ask Size - Bid Size")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 10. Median trade size per volume bar
# --------------------------------------------------

plt.figure(figsize=(9, 4))

plt.plot(
    bars["ts_event"],
    bars["median_trade_size"],
    linewidth=1,
)

plt.axhline(
    GLOBAL_MEDIAN_TRADE_SIZE,
    linewidth=0.8,
    label="Global Median",
)

plt.title(
    "GLD Median Trade Size per Dollar-Volume Bar"
)

plt.xlabel("Time")
plt.ylabel("Median Trade Size (shares)")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ==================================================
# CUMULATIVE DOLLAR-VOLUME PLOTS
# ==================================================


# --------------------------------------------------
# 11. Midpoint
# --------------------------------------------------

plt.figure(figsize=(9, 4))

plt.plot(
    bars["cum_dollar_volume_mm"],
    bars["mid"],
    linewidth=1,
)

plt.title(
    "GLD Mid Price vs Cumulative Dollar Volume"
)

plt.xlabel(
    "Cumulative Dollar Volume ($MM)"
)

plt.ylabel("Mid Price")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 12. Microprice
# --------------------------------------------------

plt.figure(figsize=(9, 4))

plt.plot(
    bars["cum_dollar_volume_mm"],
    bars["microprice"],
    linewidth=1,
)

plt.title(
    "GLD Microprice vs Cumulative Dollar Volume"
)

plt.xlabel(
    "Cumulative Dollar Volume ($MM)"
)

plt.ylabel("Microprice")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 13. Microprice - midpoint
# --------------------------------------------------

plt.figure(figsize=(9, 4))

plt.plot(
    bars["cum_dollar_volume_mm"],
    bars["microprice_minus_mid"],
    linewidth=1,
)

plt.axhline(
    0,
    linewidth=0.8,
)

plt.title(
    "GLD Microprice - Midpoint "
    "vs Cumulative Dollar Volume"
)

plt.xlabel(
    "Cumulative Dollar Volume ($MM)"
)

plt.ylabel(
    "Microprice - Mid ($)"
)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 14. Original OFI
# --------------------------------------------------

plt.figure(figsize=(9, 4))

plt.plot(
    bars["cum_dollar_volume_mm"],
    bars["ofi"],
    linewidth=1,
)

plt.axhline(
    0,
    linewidth=0.8,
)

plt.title(
    "GLD Aggressive Dollar OFI "
    "vs Cumulative Dollar Volume"
)

plt.xlabel(
    "Cumulative Dollar Volume ($MM)"
)

plt.ylabel("OFI")

plt.ylim(-1, 1)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 15. Filtered OFI
# --------------------------------------------------

plt.figure(figsize=(9, 4))

plt.plot(
    bars["cum_dollar_volume_mm"],
    bars["ofi_ex_top10"],
    linewidth=1,
)

plt.axhline(
    0,
    linewidth=0.8,
)

plt.title(
    "GLD OFI Excluding Top 10% Trade Sizes "
    "vs Cumulative Dollar Volume"
)

plt.xlabel(
    "Cumulative Dollar Volume ($MM)"
)

plt.ylabel("Filtered OFI")

plt.ylim(-1, 1)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 16. Original vs filtered OFI
# --------------------------------------------------

plt.figure(figsize=(9, 4))

plt.plot(
    bars["cum_dollar_volume_mm"],
    bars["ofi"],
    linewidth=1,
    label="All Trades",
)

plt.plot(
    bars["cum_dollar_volume_mm"],
    bars["ofi_ex_top10"],
    linewidth=1,
    label="Ex Top 10% Trade Size",
)

plt.axhline(
    0,
    linewidth=0.8,
)

plt.title(
    "GLD OFI — All Trades vs Ex-Large Trades"
)

plt.xlabel(
    "Cumulative Dollar Volume ($MM)"
)

plt.ylabel("OFI")

plt.ylim(-1, 1)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 17. Spread
# --------------------------------------------------

plt.figure(figsize=(9, 4))

plt.plot(
    bars["cum_dollar_volume_mm"],
    bars["spread"],
    linewidth=1,
)

plt.title(
    "GLD Bid-Ask Spread "
    "vs Cumulative Dollar Volume"
)

plt.xlabel(
    "Cumulative Dollar Volume ($MM)"
)

plt.ylabel("Spread ($)")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 18. Book imbalance
# --------------------------------------------------

plt.figure(figsize=(9, 4))

plt.plot(
    bars["cum_dollar_volume_mm"],
    bars["book_imbalance"],
    linewidth=1,
)

plt.axhline(
    0,
    linewidth=0.8,
)

plt.title(
    "GLD Order Book Imbalance "
    "vs Cumulative Dollar Volume"
)

plt.xlabel(
    "Cumulative Dollar Volume ($MM)"
)

plt.ylabel(
    "Ask Size - Bid Size"
)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 19. Median trade size per bar
# --------------------------------------------------

plt.figure(figsize=(9, 4))

plt.plot(
    bars["cum_dollar_volume_mm"],
    bars["median_trade_size"],
    linewidth=1,
)

plt.axhline(
    GLOBAL_MEDIAN_TRADE_SIZE,
    linewidth=0.8,
    label="Global Median",
)

plt.title(
    "GLD Median Trade Size "
    "vs Cumulative Dollar Volume"
)

plt.xlabel(
    "Cumulative Dollar Volume ($MM)"
)

plt.ylabel(
    "Median Trade Size (shares)"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ==================================================
# DIAGNOSTICS
# ==================================================

print()
print(
    f"Number of completed bars: "
    f"{len(bars)}"
)

print(
    f"Residual unfinished volume: "
    f"${total_volume:,.2f}"
)

print(
    f"Global median trade size: "
    f"{GLOBAL_MEDIAN_TRADE_SIZE:,.2f} shares"
)

print(
    f"90th percentile trade size: "
    f"{TRADE_SIZE_90TH_PERCENTILE:,.2f} shares"
)

print(
    f"Average median trade size across bars: "
    f"{bars['median_trade_size'].mean():,.2f} shares"
)

print(
    f"Average realized bar size: "
    f"${bars['dollar_volume'].mean():,.2f}"
)

print(
    f"Total completed-bar dollar volume: "
    f"${bars['dollar_volume'].sum():,.2f}"
)

print(
    f"Average spread: "
    f"${bars['spread'].mean():.4f}"
)

print(
    f"Average absolute microprice deviation: "
    f"${bars['microprice_minus_mid'].abs().mean():.4f}"
)

print(
    f"Average absolute book imbalance: "
    f"{bars['book_imbalance'].abs().mean():,.2f} shares"
)