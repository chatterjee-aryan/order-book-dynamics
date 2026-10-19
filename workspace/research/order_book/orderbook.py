from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import takewhile

import databento as db
from sortedcontainers import SortedDict


@dataclass(slots=True)
class PriceLevel:
    price: int
    size: int = 0
    count: int = 0

    def __str__(self) -> str:
        price = self.price / db.FIXED_PRICE_SCALE
        return f"{self.size:4} @ {price:6.2f} | {self.count:2} order(s)"


@dataclass(slots=True)
class DepthProfileLevel:
    level: int
    price: int
    size: int
    count: int
    distance_from_touch: int


@dataclass(slots=True)
class LevelOrders:
    price: int
    orders: list[db.MBOMsg] = field(
        default_factory=list,
        compare=False,
    )

    def __bool__(self) -> bool:
        return bool(self.orders)

    @property
    def level(self) -> PriceLevel:
        return PriceLevel(
            price=self.price,
            count=sum(
                1
                for order in self.orders
                if not order.flags & db.RecordFlags.F_TOB
            ),
            size=sum(
                order.size
                for order in self.orders
            ),
        )


@dataclass(slots=True)
class Book:
    orders_by_id: dict[int, db.MBOMsg] = field(
        default_factory=dict
    )

    offers: SortedDict[int, LevelOrders] = field(
        default_factory=SortedDict
    )

    bids: SortedDict[int, LevelOrders] = field(
        default_factory=SortedDict
    )

    # ========================================================
    # BASIC BOOK ACCESS
    # ========================================================

    def bbo(
        self,
    ) -> tuple[
        PriceLevel | None,
        PriceLevel | None,
    ]:
        return (
            self.get_bid_level(),
            self.get_ask_level(),
        )

    def get_bid_level(
        self,
        idx: int = 0,
    ) -> PriceLevel | None:
        if (
            self.bids
            and
            len(self.bids) > idx
        ):
            return (
                self.bids
                .peekitem(
                    -(idx + 1)
                )[1]
                .level
            )

        return None

    def get_ask_level(
        self,
        idx: int = 0,
    ) -> PriceLevel | None:
        if (
            self.offers
            and
            len(self.offers) > idx
        ):
            return (
                self.offers
                .peekitem(
                    idx
                )[1]
                .level
            )

        return None

    def get_bid_level_by_px(
        self,
        px: int,
    ) -> PriceLevel | None:
        try:
            return (
                self
                ._get_level(
                    px,
                    "B",
                )
                .level
            )

        except KeyError:
            return None

    def get_ask_level_by_px(
        self,
        px: int,
    ) -> PriceLevel | None:
        try:
            return (
                self
                ._get_level(
                    px,
                    "A",
                )
                .level
            )

        except KeyError:
            return None

    def get_order(
        self,
        id: int,
    ) -> db.MBOMsg | None:
        return (
            self.orders_by_id
            .get(
                id
            )
        )

    def get_queue_pos(
        self,
        id: int,
    ) -> int | None:
        order = self.get_order(
            id
        )

        if not order:
            return None

        level = self._get_level(
            order.price,
            order.side,
        )

        return sum(
            queued_order.size

            for queued_order
            in takewhile(
                lambda queued_order:
                    queued_order.order_id
                    != id,

                level.orders,
            )
        )

    def get_snapshot(
        self,
        level_count: int = 1,
    ) -> list[db.BidAskPair]:
        snapshots = []

        for level_idx in range(
            level_count
        ):
            ba_pair = (
                db.BidAskPair()
            )

            bid = (
                self.get_bid_level(
                    level_idx
                )
            )

            if bid:
                ba_pair.bid_px = (
                    bid.price
                )

                ba_pair.bid_sz = (
                    bid.size
                )

                ba_pair.bid_ct = (
                    bid.count
                )

            ask = (
                self.get_ask_level(
                    level_idx
                )
            )

            if ask:
                ba_pair.ask_px = (
                    ask.price
                )

                ba_pair.ask_sz = (
                    ask.size
                )

                ba_pair.ask_ct = (
                    ask.count
                )

            snapshots.append(
                ba_pair
            )

        return snapshots

    # ========================================================
    # COMPLETE DEPTH PROFILE
    # ========================================================

    def get_depth_profile(
        self,
        side: str,
    ) -> list[DepthProfileLevel]:
        """
        Return the complete visible depth profile from
        the touch outward.

        side:
            "B" or "bid" -> bid side
            "A" or "ask" -> ask side

        Returned ordering:
            bid:
                best bid -> deeper bids

            ask:
                best ask -> deeper asks

        distance_from_touch:
            stored in Databento fixed-price units.

            bid:
                best_bid - price

            ask:
                price - best_ask
        """

        side = side.upper()

        if side in (
            "B",
            "BID",
        ):
            raw_levels = list(
                reversed(
                    self.bids.items()
                )
            )

            normalized_side = "B"

        elif side in (
            "A",
            "ASK",
        ):
            raw_levels = list(
                self.offers.items()
            )

            normalized_side = "A"

        else:
            raise ValueError(
                "side must be one of: "
                "'B', 'bid', 'A', 'ask'"
            )

        if not raw_levels:
            return []

        touch_price = (
            raw_levels[0][0]
        )

        profile = []

        for (
            level_number,
            (
                price,
                level_orders,
            ),
        ) in enumerate(
            raw_levels,
            start=1,
        ):
            level = (
                level_orders.level
            )

            if (
                normalized_side
                == "B"
            ):
                distance = (
                    touch_price
                    - price
                )

            else:
                distance = (
                    price
                    - touch_price
                )

            profile.append(
                DepthProfileLevel(
                    level=level_number,
                    price=price,
                    size=level.size,
                    count=level.count,
                    distance_from_touch=distance,
                )
            )

        return profile

    # ========================================================
    # FULL DEPTH
    # ========================================================

    def total_depth(
        self,
        side: str,
    ) -> int:
        return sum(
            profile_level.size

            for profile_level
            in self.get_depth_profile(
                side
            )
        )

    def total_bid_depth(
        self,
    ) -> int:
        return (
            self.total_depth(
                "B"
            )
        )

    def total_ask_depth(
        self,
    ) -> int:
        return (
            self.total_depth(
                "A"
            )
        )

    def full_depth_difference(
        self,
    ) -> int:
        return (
            self.total_bid_depth()
            -
            self.total_ask_depth()
        )

    def full_depth_imbalance(
        self,
    ) -> float:
        """
        +1:
            all visible depth is bid-side

         0:
            equal total bid and ask depth

        -1:
            all visible depth is ask-side
        """

        bid_depth = (
            self.total_bid_depth()
        )

        ask_depth = (
            self.total_ask_depth()
        )

        total = (
            bid_depth
            +
            ask_depth
        )

        if total <= 0:
            return float(
                "nan"
            )

        return (
            bid_depth
            -
            ask_depth
        ) / total

    # ========================================================
    # NORMAL MBO REPLAY
    # ========================================================

    def apply(
        self,
        mbo: db.MBOMsg,
    ) -> None:
        # Trade, Fill, or None:
        # no passive-state mutation.
        if mbo.action in (
            "T",
            "F",
            "N",
        ):
            return

        # Clear book.
        if mbo.action == "R":
            self._clear()
            return

        assert mbo.side in (
            "A",
            "B",
        )

        # UNDEF_PRICE + F_TOB:
        # clear displayed side.
        if (
            mbo.price
            ==
            db.UNDEF_PRICE

            and

            mbo.flags
            &
            db.RecordFlags.F_TOB
        ):
            (
                self
                ._side_levels(
                    mbo.side
                )
                .clear()
            )

            return

        if mbo.action == "A":
            self._add(
                mbo
            )

        elif mbo.action == "C":
            self._cancel(
                mbo
            )

        elif mbo.action == "M":
            self._modify(
                mbo
            )

        else:
            raise ValueError(
                f"Unknown "
                f"action={mbo.action}"
            )

    # ========================================================
    # COUNTERFACTUAL REPLAY
    # ========================================================

    def apply_counterfactual(
        self,
        mbo: db.MBOMsg,
    ) -> int:
        """
        Frozen-passive counterfactual.

        B = buyer aggressor
            -> consumes asks

        A = seller aggressor
            -> consumes bids

        Returns remaining quantity if the
        frozen book is exhausted.
        """

        if mbo.action != "T":
            return 0

        if mbo.side == "B":
            return (
                self._consume_fifo(
                    side="A",
                    quantity=int(
                        mbo.size
                    ),
                )
            )

        if mbo.side == "A":
            return (
                self._consume_fifo(
                    side="B",
                    quantity=int(
                        mbo.size
                    ),
                )
            )

        return int(
            mbo.size
        )

    def _consume_fifo(
        self,
        side: str,
        quantity: int,
    ) -> int:
        remaining = int(
            quantity
        )

        if remaining <= 0:
            return 0

        levels = (
            self._side_levels(
                side
            )
        )

        while (
            remaining > 0
            and
            levels
        ):
            if side == "A":
                (
                    price,
                    level,
                ) = levels.peekitem(
                    0
                )

            else:
                (
                    price,
                    level,
                ) = levels.peekitem(
                    -1
                )

            while (
                remaining > 0
                and
                level.orders
            ):
                resting = (
                    level.orders[0]
                )

                consumed = min(
                    int(
                        resting.size
                    ),
                    remaining,
                )

                resting.size -= (
                    consumed
                )

                remaining -= (
                    consumed
                )

                if (
                    resting.size
                    == 0
                ):
                    level.orders.pop(
                        0
                    )

                    (
                        self
                        .orders_by_id
                        .pop(
                            resting.order_id,
                            None,
                        )
                    )

            if not level.orders:
                levels.pop(
                    price
                )

        return remaining

    # ========================================================
    # INTERNAL NORMAL BOOK MUTATORS
    # ========================================================

    def _clear(
        self,
    ) -> None:
        self.orders_by_id.clear()
        self.offers.clear()
        self.bids.clear()

    def _add(
        self,
        mbo: db.MBOMsg,
    ) -> None:
        if (
            mbo.flags
            &
            db.RecordFlags.F_TOB
        ):
            levels = (
                self._side_levels(
                    mbo.side
                )
            )

            levels.clear()

            levels[
                mbo.price
            ] = LevelOrders(
                price=mbo.price,
                orders=[
                    mbo
                ],
            )

        else:
            level = (
                self
                ._get_or_insert_level(
                    mbo.price,
                    mbo.side,
                )
            )

            assert (
                mbo.order_id
                not in
                self.orders_by_id
            )

            self.orders_by_id[
                mbo.order_id
            ] = mbo

            level.orders.append(
                mbo
            )

    def _cancel(
        self,
        mbo: db.MBOMsg,
    ) -> None:
        order = (
            self.orders_by_id[
                mbo.order_id
            ]
        )

        level = (
            self._get_level(
                mbo.price,
                mbo.side,
            )
        )

        assert (
            order.size
            >=
            mbo.size
        )

        order.size -= (
            mbo.size
        )

        if order.size == 0:
            self.orders_by_id.pop(
                mbo.order_id
            )

            level.orders.remove(
                order
            )

            if not level:
                self._remove_level(
                    mbo.price,
                    mbo.side,
                )

    def _modify(
        self,
        mbo: db.MBOMsg,
    ) -> None:
        order = (
            self.orders_by_id
            .get(
                mbo.order_id
            )
        )

        if order is None:
            self._add(
                mbo
            )

            return

        assert (
            order.side
            ==
            mbo.side
        ), (
            f"Order {order} "
            f"changed side "
            f"to {mbo.side}"
        )

        level = (
            self._get_level(
                order.price,
                order.side,
            )
        )

        if (
            order.price
            !=
            mbo.price
        ):
            level.orders.remove(
                order
            )

            if not level:
                self._remove_level(
                    order.price,
                    mbo.side,
                )

            level = (
                self
                ._get_or_insert_level(
                    mbo.price,
                    mbo.side,
                )
            )

            level.orders.append(
                mbo
            )

        elif (
            order.size
            <
            mbo.size
        ):
            level.orders.remove(
                order
            )

            level.orders.append(
                mbo
            )

        else:
            level.orders[
                level.orders.index(
                    order
                )
            ] = mbo

        self.orders_by_id[
            mbo.order_id
        ] = mbo

    # ========================================================
    # INTERNAL LEVEL HELPERS
    # ========================================================

    def _side_levels(
        self,
        side: str,
    ) -> SortedDict:
        if side == "A":
            return self.offers

        if side == "B":
            return self.bids

        raise ValueError(
            f"Invalid "
            f"{side =}"
        )

    def _get_level(
        self,
        price: int,
        side: str,
    ) -> LevelOrders:
        levels = (
            self._side_levels(
                side
            )
        )

        if (
            price
            not in
            levels
        ):
            raise KeyError(
                "No price level "
                "found for "
                f"{price =} "
                f"and {side =}"
            )

        return (
            levels[
                price
            ]
        )

    def _get_or_insert_level(
        self,
        price: int,
        side: str,
    ) -> LevelOrders:
        levels = (
            self._side_levels(
                side
            )
        )

        if (
            price
            in
            levels
        ):
            return (
                levels[
                    price
                ]
            )

        level = (
            LevelOrders(
                price=price
            )
        )

        levels[
            price
        ] = level

        return level

    def _remove_level(
        self,
        price: int,
        side: str,
    ) -> None:
        levels = (
            self._side_levels(
                side
            )
        )

        levels.pop(
            price
        )


@dataclass(slots=True)
class Market:
    books: defaultdict[
        int,
        defaultdict[
            int,
            Book,
        ],
    ] = field(
        default_factory=lambda:
            defaultdict(
                lambda:
                    defaultdict(
                        Book
                    )
            )
    )

    def get_books_by_pub(
        self,
        instrument_id: int,
    ) -> defaultdict[
        int,
        Book,
    ]:
        return (
            self.books[
                instrument_id
            ]
        )

    def get_book(
        self,
        instrument_id: int,
        publisher_id: int,
    ) -> Book:
        return (
            self.books[
                instrument_id
            ][
                publisher_id
            ]
        )

    def bbo(
        self,
        instrument_id: int,
        publisher_id: int,
    ) -> tuple[
        PriceLevel | None,
        PriceLevel | None,
    ]:
        return (
            self.books[
                instrument_id
            ][
                publisher_id
            ]
            .bbo()
        )

    def aggregated_bbo(
        self,
        instrument_id: int,
    ) -> tuple[
        PriceLevel | None,
        PriceLevel | None,
    ]:
        agg_bbo: list[
            PriceLevel | None
        ] = [
            None,
            None,
        ]

        for (
            idx,
            reducer,
        ) in [
            (
                0,
                max,
            ),
            (
                1,
                min,
            ),
        ]:
            all_best = [
                book.bbo()[
                    idx
                ]

                for book
                in self.books[
                    instrument_id
                ].values()
            ]

            all_best = [
                level

                for level
                in all_best

                if level
            ]

            if not all_best:
                continue

            best_price = reducer(
                level.price

                for level
                in all_best
            )

            best = [
                level

                for level
                in all_best

                if (
                    level.price
                    ==
                    best_price
                )
            ]

            agg_bbo[
                idx
            ] = PriceLevel(
                price=best_price,

                size=sum(
                    level.size

                    for level
                    in best
                ),

                count=sum(
                    level.count

                    for level
                    in best
                ),
            )

        return tuple(
            agg_bbo
        )

    def apply(
        self,
        mbo: db.MBOMsg,
    ) -> None:
        book = (
            self.books[
                mbo.instrument_id
            ][
                mbo.publisher_id
            ]
        )

        book.apply(
            mbo
        )