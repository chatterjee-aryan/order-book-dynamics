from pathlib import Path
import math

import databento as db
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from workspace.research.order_book.orderbook import Book, LevelOrders


# ============================================================
# CONFIG
# ============================================================

MBO_PATH = Path("data/raw/mbo/gld-2026-01-06.mbo.csv")

DOLLAR_VOLUME_THRESHOLD = 1_000_000.0
MBO_CHUNK_SIZE = 1_000_000
EWMA_HALFLIFE_BARS = 1
CHOP_FRACTION = 0.10
TOP_N_LEVELS = 10


# ============================================================
# LIGHTWEIGHT MBO MESSAGE
# ============================================================

class FastMBOMsg:
    __slots__ = (
        "instrument_id", "publisher_id", "order_id",
        "price", "size", "flags", "action", "side", "sequence",
    )

    def __init__(
        self,
        instrument_id,
        publisher_id,
        order_id,
        price,
        size,
        flags,
        action,
        side,
        sequence,
    ):
        self.instrument_id = instrument_id
        self.publisher_id = publisher_id
        self.order_id = order_id
        self.price = price
        self.size = size
        self.flags = flags
        self.action = action
        self.side = side
        self.sequence = sequence


def to_fixed_price(x):
    if pd.isna(x):
        return db.UNDEF_PRICE
    return int(round(float(x) * db.FIXED_PRICE_SCALE))


def from_fixed_price(x):
    return x / db.FIXED_PRICE_SCALE


def make_mbo(row):
    return FastMBOMsg(
        int(row.instrument_id),
        int(row.publisher_id),
        int(row.order_id),
        to_fixed_price(row.price),
        int(row.size),
        int(row.flags),
        row.action,
        row.side,
        int(row.sequence),
    )


# ============================================================
# BOOK CLONE
# ============================================================

def clone_order(order):
    return FastMBOMsg(
        order.instrument_id,
        order.publisher_id,
        order.order_id,
        order.price,
        order.size,
        order.flags,
        order.action,
        order.side,
        order.sequence,
    )


def clone_book(source):
    target = Book()

    for side_name in ("bids", "offers"):
        source_levels = getattr(source, side_name)
        target_levels = getattr(target, side_name)

        for price, level in source_levels.items():
            orders = []

            for order in level.orders:
                cloned = clone_order(order)
                orders.append(cloned)

                if order.order_id in source.orders_by_id:
                    target.orders_by_id[cloned.order_id] = cloned

            target_levels[price] = LevelOrders(
                price=price,
                orders=orders,
            )

    return target


# ============================================================
# BOOK HELPERS
# ============================================================

def bbo_fixed(book):
    bid, ask = book.bbo()
    return (
        None if bid is None else int(bid.price),
        None if ask is None else int(ask.price),
    )


def top_n_snapshot(book, side, n=TOP_N_LEVELS):
    """
    Returns [(price, size), ...] from touch outward.
    This is the INITIAL ladder used for the response measure.
    """
    if side == "B":
        items = list(reversed(book.bids.items()))[:n]
    else:
        items = list(book.offers.items())[:n]

    return [
        (int(px), int(level.level.size))
        for px, level in items
    ]


def total_snapshot_depth(snapshot):
    return sum(size for _, size in snapshot)


def implied_depletion(snapshot, final_px, side):
    """
    Cumulative INITIAL depth implied to have disappeared
    for the touch to reach final_px.

    Endpoint level itself is NOT counted.

    Bid:
        sum initial depth strictly ABOVE final bid.

    Ask:
        sum initial depth strictly BELOW final ask.

    If price moves through all top-N initial levels,
    depletion saturates at total top-N depth.
    """
    if not snapshot:
        return np.nan

    if side == "B":
        return sum(
            size
            for px, size in snapshot
            if px > final_px
        )

    return sum(
        size
        for px, size in snapshot
        if px < final_px
    )


# ============================================================
# FAST COUNTERFACTUAL
# ============================================================

def consume_quantity(book, aggressor_side, quantity):
    """
    Mechanical counterfactual depletion by PRICE-LEVEL quantity.

    No FIFO/order-by-order traversal.

    buyer aggressor ("B") -> consume asks
    seller aggressor ("A") -> consume bids

    Returns unconsumed quantity if frozen side is exhausted.
    """
    remaining = int(quantity)

    if remaining <= 0:
        return 0

    if aggressor_side == "B":
        levels = book.offers
        take_best = lambda: levels.peekitem(0)

    elif aggressor_side == "A":
        levels = book.bids
        take_best = lambda: levels.peekitem(-1)

    else:
        return remaining

    while remaining > 0 and levels:
        price, level = take_best()
        level_size = int(level.level.size)

        if level_size <= 0:
            levels.pop(price)
            continue

        if remaining >= level_size:
            remaining -= level_size
            levels.pop(price)

        else:
            # We only care about aggregate level quantity / BBO,
            # so resize one representative order and remove the rest.
            first = level.orders[0]
            first.size = level_size - remaining
            level.orders[:] = [first]
            remaining = 0

    return remaining


# ============================================================
# DOLLAR-VOLUME BOUNDARIES
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


def build_boundaries(path):
    boundaries = []
    dv = 0.0
    start_sequence = None
    instrument = None
    publisher = None

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

        for row in trades.itertuples(index=False):
            iid = int(row.instrument_id)
            pid = int(row.publisher_id)

            if instrument is None:
                instrument = iid
                publisher = pid

            elif (
                iid != instrument
                or pid != publisher
            ):
                raise ValueError(
                    "Multiple instrument/publisher pairs found."
                )

            price = float(row.price)
            size = int(row.size)

            if (
                not math.isfinite(price)
                or size <= 0
            ):
                continue

            if start_sequence is None:
                start_sequence = int(row.sequence)

            dv += price * size

            if dv >= DOLLAR_VOLUME_THRESHOLD:
                boundaries.append(
                    {
                        "start_sequence": start_sequence,
                        "end_sequence": int(row.sequence),
                        "ts_event": pd.Timestamp(row.ts_event),
                        "dollar_volume": dv,
                    }
                )

                dv = 0.0
                start_sequence = None

    if not boundaries:
        raise RuntimeError(
            "No completed dollar-volume bars."
        )

    return boundaries, instrument, publisher


# ============================================================
# RESPONSE MEASURE
# ============================================================

def side_response(
    initial_snapshot,
    initial_touch,
    actual_touch,
    cf_touch,
    side,
):
    """
    New response measure.

    Bid:
        if B > B0: response = 1
        else:
            R_B = [D0(B_cf) - D0(B)] / total initial top-N depth

    Ask:
        if A < A0: response = 1
        else:
            R_A = [D0(A_cf) - D0(A)] / total initial top-N depth

    Positive = absorption / replenishment.
    Zero     = actual followed frozen-book depletion.
    Negative = actual retreated farther than counterfactual.
    """
    if (
        initial_touch is None
        or actual_touch is None
        or cf_touch is None
        or not initial_snapshot
    ):
        return np.nan

    total_depth = total_snapshot_depth(
        initial_snapshot
    )

    if total_depth <= 0:
        return np.nan

    # Full absorption + favorable repricing beyond start.
    if side == "B" and actual_touch > initial_touch:
        return 1.0

    if side == "A" and actual_touch < initial_touch:
        return 1.0

    d_cf = implied_depletion(
        initial_snapshot,
        cf_touch,
        side,
    )

    d_actual = implied_depletion(
        initial_snapshot,
        actual_touch,
        side,
    )

    return (
        d_cf - d_actual
    ) / total_depth


# ============================================================
# REPLAY
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


def replay_response(
    path,
    boundaries,
    instrument,
    publisher,
):
    actual = Book()
    cf = None

    rows = []
    bar_idx = 0
    prev_sequence = None
    sampled = False

    start_bid = start_ask = None
    start_bid_snapshot = None
    start_ask_snapshot = None

    bid_cf_exhausted = False
    ask_cf_exhausted = False

    pending = None

    def begin_sequence(sequence):
        nonlocal cf
        nonlocal sampled
        nonlocal pending
        nonlocal start_bid
        nonlocal start_ask
        nonlocal start_bid_snapshot
        nonlocal start_ask_snapshot
        nonlocal bid_cf_exhausted
        nonlocal ask_cf_exhausted

        if bar_idx >= len(boundaries):
            return

        bar = boundaries[bar_idx]

        if (
            cf is None
            and sequence == bar["start_sequence"]
        ):
            start_bid, start_ask = bbo_fixed(actual)

            start_bid_snapshot = top_n_snapshot(
                actual,
                "B",
            )

            start_ask_snapshot = top_n_snapshot(
                actual,
                "A",
            )

            cf = clone_book(actual)

            bid_cf_exhausted = False
            ask_cf_exhausted = False
            sampled = False
            pending = None

        if (
            cf is None
            and sequence > bar["start_sequence"]
        ):
            raise RuntimeError(
                f"Missed start sequence for bar {bar_idx}"
            )

        # Sample immediately BEFORE threshold-crossing sequence.
        if sequence == bar["end_sequence"]:
            actual_bid, actual_ask = bbo_fixed(actual)
            cf_bid, cf_ask = bbo_fixed(cf)

            bid_response = side_response(
                start_bid_snapshot,
                start_bid,
                actual_bid,
                cf_bid,
                "B",
            )

            ask_response = side_response(
                start_ask_snapshot,
                start_ask,
                actual_ask,
                cf_ask,
                "A",
            )

            if bid_cf_exhausted:
                bid_response = np.nan

            if ask_cf_exhausted:
                ask_response = np.nan

            pending = {
                "bar": bar_idx,
                "ts_event": bar["ts_event"],
                "dollar_volume": bar["dollar_volume"],

                "start_bid": (
                    np.nan
                    if start_bid is None
                    else from_fixed_price(start_bid)
                ),
                "start_ask": (
                    np.nan
                    if start_ask is None
                    else from_fixed_price(start_ask)
                ),

                "actual_bid": (
                    np.nan
                    if actual_bid is None
                    else from_fixed_price(actual_bid)
                ),
                "actual_ask": (
                    np.nan
                    if actual_ask is None
                    else from_fixed_price(actual_ask)
                ),

                "cf_bid": (
                    np.nan
                    if cf_bid is None
                    else from_fixed_price(cf_bid)
                ),
                "cf_ask": (
                    np.nan
                    if cf_ask is None
                    else from_fixed_price(cf_ask)
                ),

                "bid_response": bid_response,
                "ask_response": ask_response,

                "bid_cf_exhausted": bid_cf_exhausted,
                "ask_cf_exhausted": ask_cf_exhausted,
            }

            sampled = True

    def finish_sequence(sequence):
        nonlocal bar_idx
        nonlocal cf
        nonlocal sampled
        nonlocal pending

        if bar_idx >= len(boundaries):
            return

        bar = boundaries[bar_idx]

        if (
            sampled
            and sequence == bar["end_sequence"]
        ):
            rows.append(pending)
            pending = None
            cf = None
            sampled = False
            bar_idx += 1

    print("Streaming MBO replay...")

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
            chunk["instrument_id"].eq(instrument)
            &
            chunk["publisher_id"].eq(publisher)
        ]

        for row in chunk.itertuples(index=False):
            if bar_idx >= len(boundaries):
                break

            sequence = int(row.sequence)

            if prev_sequence is None:
                prev_sequence = sequence
                begin_sequence(sequence)

            elif sequence != prev_sequence:
                finish_sequence(prev_sequence)

                if bar_idx >= len(boundaries):
                    break

                prev_sequence = sequence
                begin_sequence(sequence)

            action = row.action

            # Actual book: passive actions only.
            if action in ("A", "C", "M", "R"):
                actual.apply(
                    make_mbo(row)
                )

            # Frozen counterfactual: trades only.
            elif (
                action == "T"
                and cf is not None
                and not sampled
            ):
                side = row.side

                if side in ("A", "B"):
                    remaining = consume_quantity(
                        cf,
                        side,
                        int(row.size),
                    )

                    if remaining > 0:
                        if side == "A":
                            bid_cf_exhausted = True
                        else:
                            ask_cf_exhausted = True

        print(
            f"Chunk {chunk_number} | "
            f"bars {bar_idx:,}/{len(boundaries):,}"
        )

        if bar_idx >= len(boundaries):
            break

    if (
        prev_sequence is not None
        and bar_idx < len(boundaries)
    ):
        finish_sequence(prev_sequence)

    results = pd.DataFrame(rows)

    if results.empty:
        raise RuntimeError(
            "No response bars constructed."
        )

    return results


# ============================================================
# FEATURES + PLOT
# ============================================================

def add_features(results):
    results = results.copy()

    results["ts_event"] = pd.to_datetime(
        results["ts_event"],
        utc=True,
    )

    results["cum_dollar_volume_mm"] = (
        results["dollar_volume"].cumsum()
        / 1_000_000.0
    )

    results["mid_price"] = (
        results["actual_bid"]
        + results["actual_ask"]
    ) / 2.0

    results["bid_response_ewma"] = (
        results["bid_response"]
        .ewm(
            halflife=EWMA_HALFLIFE_BARS,
            adjust=False,
        )
        .mean()
    )

    results["ask_response_ewma"] = (
        results["ask_response"]
        .ewm(
            halflife=EWMA_HALFLIFE_BARS,
            adjust=False,
        )
        .mean()
    )

    results["response_imbalance_ewma"] = (
        results["bid_response_ewma"]
        - results["ask_response_ewma"]
    )

    return results


def plot_response(results):
    x = results[
        "cum_dollar_volume_mm"
    ]

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11, 7),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                1,
                1,
                1,
                1.15,
            ]
        },
    )

    series = [
        (
            "bid_response_ewma",
            "Bid Response",
        ),
        (
            "ask_response_ewma",
            "Ask Response",
        ),
        (
            "response_imbalance_ewma",
            "Response\nImbalance",
        ),
    ]

    for ax, (col, label) in zip(
        axes[:3],
        series,
    ):
        ax.plot(
            x,
            results[col],
            linewidth=1.0,
        )

        ax.axhline(
            0.0,
            linestyle="--",
            linewidth=0.8,
        )

        ax.set_ylabel(label)
        ax.grid(alpha=0.3)

    axes[0].set_title(
        "Initial-Book Depth Response "
        f"(Top {TOP_N_LEVELS}, "
        f"EWMA halflife = "
        f"{EWMA_HALFLIFE_BARS} bars)"
    )

    axes[3].plot(
        x,
        results["mid_price"],
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

def main():
    boundaries, instrument, publisher = build_boundaries(
        MBO_PATH
    )

    results = replay_response(
        MBO_PATH,
        boundaries,
        instrument,
        publisher,
    )

    results = add_features(
        results
    )

    chop = int(
        len(results)
        * CHOP_FRACTION
    )

    if chop > 0:
        results = (
            results
            .iloc[chop:-chop]
            .copy()
            .reset_index(drop=True)
        )

    print(
        f"Analysis observations after chop: "
        f"{len(results):,}"
    )

    # ========================================================
    # NEGATIVE RESPONSE DIAGNOSTICS
    # ========================================================

    for side in ["bid", "ask"]:
        r = results[
            f"{side}_response"
        ].dropna()

        print(
            side,
            "P(R < 0):",
            (r < 0).mean(),
            "mean negative:",
            r[r < 0].mean(),
            "P(R > 0):",
            (r > 0).mean(),
        )

        print(
            side,
            "P(R < 0.5):",
            (r < 0.5).mean(),
            "mean negative:",
            r[r < 0.5].mean(),
            "P(R > 0):",
            (r > 0.5).mean(),
        )

    neg_any = (
        (results["bid_response"] < 0)
        |
        (results["ask_response"] < 0)
    )

    neg_both = (
        (results["bid_response"] < 0)
        &
        (results["ask_response"] < 0)
    )

    print(
        "Any-side retreat:",
        neg_any.mean()
    )

    print(
        "Both-side retreat:",
        neg_both.mean()
    )

    plot_response(
        results
    )


if __name__ == "__main__":
    main()