"""Construct canonical futures symbols.

Instructions file: ../instructions/futures_ticker.md
Dependency instructions: ../instructions/futures_ticker_convention.md

Inputs identify the exact expiration year and month.
These labels do not verify historical listing, liquidity, or vendor identifiers.
Gold uses the six principal expiration months and CME Euro FX uses quarterly
expirations; nearby serial contracts are outside this utility's supported cycles.
"""

from numbers import Integral


MONTH_CODES = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}

FUTURES_CONVENTIONS = {
    "fed_funds": {"root": "ZQ", "months": tuple(range(1, 13))},
    "gold": {"root": "GC", "months": (2, 4, 6, 8, 10, 12)},
    "treasury_2y": {"root": "ZT", "months": (3, 6, 9, 12)},
    "treasury_10y": {"root": "ZN", "months": (3, 6, 9, 12)},
    "eur_usd": {"root": "6E", "months": (3, 6, 9, 12)},
    "btc_usd": {"root": "BTC", "months": tuple(range(1, 13))},
    "equity": {"root": "ES", "months": (3, 6, 9, 12)},
}


def _validate_date(year: int, month: int) -> None:
    if isinstance(year, bool) or not isinstance(year, Integral):
        raise TypeError("year must be a four-digit integer")
    if not 1000 <= year <= 9999:
        raise ValueError("year must be between 1000 and 9999")
    if isinstance(month, bool) or not isinstance(month, Integral):
        raise TypeError("month must be an integer from 1 to 12")
    if not 1 <= month <= 12:
        raise ValueError("month must be between 1 and 12")


def _ticker(product: str, year: int, month: int) -> str:
    _validate_date(year, month)
    convention = FUTURES_CONVENTIONS[product]
    if month not in convention["months"]:
        raise ValueError(
            f"{convention['root']} month must be one of {convention['months']}; "
            f"received {month}"
        )
    return f"{convention['root']}{MONTH_CODES[month]}{year % 100:02d}"


def fed_funds_ticker(year: int, month: int) -> str:
    """Return ZQ for the supplied expiration year and month."""
    return _ticker("fed_funds", year, month)


def gold_ticker(year: int, month: int) -> str:
    """Return GC for a principal expiration month (2, 4, 6, 8, 10, 12)."""
    return _ticker("gold", year, month)


def treasury_2y_ticker(year: int, month: int) -> str:
    """Return ZT for a quarterly expiration month (3, 6, 9, 12)."""
    return _ticker("treasury_2y", year, month)


def treasury_10y_ticker(year: int, month: int) -> str:
    """Return ZN for a quarterly expiration month (3, 6, 9, 12)."""
    return _ticker("treasury_10y", year, month)


def eur_usd_ticker(year: int, month: int) -> str:
    """Return CME 6E for a quarterly expiration month (3, 6, 9, 12)."""
    return _ticker("eur_usd", year, month)


def btc_usd_ticker(year: int, month: int) -> str:
    """Return standard CME BTC for an expiration from its December 2017 launch."""
    _validate_date(year, month)
    if (year, month) < (2017, 12):
        raise ValueError("CME BTC futures expirations begin in December 2017")
    return _ticker("btc_usd", year, month)


def equity_ticker(year: int, month: int) -> str:
    """Return ES for a quarterly expiration month (3, 6, 9, 12)."""
    return _ticker("equity", year, month)
