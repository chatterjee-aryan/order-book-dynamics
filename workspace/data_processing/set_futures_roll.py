"""Set primary-contract flags using local daily volumes.

Instruction file: ../instructions/setting_futures_roll.md
Run: python -m workspace.data_processing.set_futures_roll

The challenger must have at least 65% of combined volume for three consecutive
trading observations. A sole available bar wins immediately. Dates with neither
bar have two False flags and do not count toward the streak. Existing columns,
including expired flags, are preserved. No API calls or report files are used.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "ohlcv"
FIELDS = ["open", "high", "low", "close", "volume"]
CONFIRMATION_DAYS = 3
VOLUME_SHARE_PERCENT = 65
FILE_PATTERN = re.compile(r"^(ZT|ZN|GC|ES|BTC|6E)_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}\.csv$")
LOG = logging.getLogger(__name__)


def read_ohlcv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=[0, 1], index_col=0, float_precision="round_trip")


def set_primary(frame: pd.DataFrame, symbols: pd.DataFrame, root: str) -> pd.DataFrame:
    """Apply the stateful share rule in date order, retaining original row order."""
    if root == "ZQ":
        return frame.copy()
    labels = [f"{root}_T0", f"{root}_T1"]
    dates = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise"))
    if dates.hasnans or not dates.is_unique or not frame.columns.is_unique:
        raise ValueError("OHLCV dates and columns must be valid and unique")
    if dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError("OHLCV index must contain date-only labels")
    if set(frame.columns.get_level_values(0)) != set(labels):
        raise ValueError(f"Expected exactly {labels}")
    volumes, present = [], []
    for label in labels:
        if not set(FIELDS).issubset(frame[label].columns):
            raise ValueError(f"{label}: missing OHLCV columns")
        values = frame[label][FIELDS]
        numeric = values.apply(pd.to_numeric, errors="raise")
        if np.isinf(numeric.to_numpy(dtype=float)).any():
            raise ValueError(f"{label}: infinite OHLCV values")
        volume = numeric["volume"]
        if (volume.notna() & ((volume < 0) | (volume % 1 != 0))).any():
            raise ValueError(f"{label}: negative or noninteger volume")
        has_bar = values.notna().any(axis=1).to_numpy()
        partial = has_bar & values.isna().any(axis=1).to_numpy()
        if partial.any():
            LOG.warning("%s: %d partially populated bars; missing volume breaks the streak",
                        label, partial.sum())
        volumes.append(volume.to_numpy(dtype=float))
        present.append(has_bar)
    required = {"date", "tenor", "canonical_symbol", "raw_symbol", "contract_year", "contract_month"}
    if missing := required.difference(symbols.columns):
        raise ValueError(f"Missing symbol columns: {sorted(missing)}")
    symbols = symbols.copy()
    symbols["date"] = pd.to_datetime(symbols["date"], errors="raise")
    if symbols["date"].isna().any() or symbols.duplicated(["date", "tenor"]).any():
        raise ValueError("Invalid or duplicate symbol date/tenor mappings")
    lookup = symbols.set_index(["date", "tenor"])

    def identity(day, slot, has_bar):
        if (day, labels[slot]) not in lookup.index:
            if has_bar:
                raise ValueError(f"Missing symbol mapping for {day.date()} {labels[slot]}")
            return None
        row = lookup.loc[(day, labels[slot])]
        if row[["canonical_symbol", "raw_symbol", "contract_year", "contract_month"]].isna().any():
            raise ValueError(f"Missing contract identity for {day.date()} {labels[slot]}")
        year, month = float(row.contract_year), float(row.contract_month)
        if not (year.is_integer() and month.is_integer() and 1000 <= year <= 9999 and 1 <= month <= 12):
            raise ValueError("Invalid contract delivery year/month")
        # Full year and month distinguish raw symbols reused across decades.
        return root, int(year), int(month)

    flags = np.zeros((len(frame), 2), dtype=bool)
    primary, previous_pair, challenger = None, None, None
    streak = 0
    for position in np.argsort(dates.asi8):
        day = dates[position]
        pair = tuple(identity(day, slot, present[slot][position]) for slot in range(2))
        if pair[0] is not None and pair[0] == pair[1]:
            raise ValueError(f"Both tenors map to the same contract on {day.date()}")
        if pair != previous_pair:
            streak, challenger = 0, None
            previous_pair = pair
        available = [slot for slot in range(2) if present[slot][position]]
        if not available:
            # A shared no-bar date does not interrupt an otherwise unchanged pair.
            continue
        if len(available) == 1:
            chosen = available[0]
            primary = pair[chosen]
            streak, challenger = 0, None
        else:
            if primary not in pair:
                primary = pair[0]
                streak, challenger = 0, None
            chosen = pair.index(primary)
            other = 1 - chosen
            first, second = volumes[0][position], volumes[1][position]
            total = first + second
            qualifies = (np.isfinite(first) and np.isfinite(second) and total > 0
                         and volumes[other][position] * 100 >= total * VOLUME_SHARE_PERCENT)
            if qualifies:
                streak = streak + 1 if challenger == pair[other] else 1
                challenger = pair[other]
                if streak >= CONFIRMATION_DAYS:
                    chosen = other
                    primary = pair[chosen]
                    streak, challenger = 0, None
            else:
                streak, challenger = 0, None
        flags[position, chosen] = True
    available_any = present[0] | present[1]
    if not np.array_equal(flags.sum(axis=1), available_any.astype(int)):
        raise AssertionError("Primary flags are not exclusive")
    if any((flags[:, slot] & ~present[slot]).any() for slot in range(2)):
        raise AssertionError("A missing bar was marked primary")
    result = frame.copy()
    for slot, label in enumerate(labels):
        result[(label, "is_primary")] = flags[:, slot]
    base = frame.loc[:, frame.columns.get_level_values(1) != "is_primary"]
    pd.testing.assert_frame_equal(result[base.columns], base, check_exact=True)
    return result


def write_verified(path: Path, original: pd.DataFrame, updated: pd.DataFrame) -> None:
    """Replace the original only after verifying every other column survives."""
    base = original.loc[:, original.columns.get_level_values(1) != "is_primary"]
    pd.testing.assert_frame_equal(updated[base.columns], base, check_exact=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        updated.to_csv(temporary)
        saved = read_ohlcv(temporary)
        pd.testing.assert_frame_equal(saved[base.columns], base, check_exact=True)
        for column in updated.columns:
            if column[1] == "is_primary":
                pd.testing.assert_series_equal(saved[column], updated[column], check_exact=True)
        pd.testing.assert_frame_equal(read_ohlcv(path), original, check_exact=True)
        temporary.chmod(path.stat().st_mode & 0o777)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def process_pair(path: Path, symbols_path: Path, root: str) -> None:
    original = read_ohlcv(path)
    updated = set_primary(original, pd.read_csv(symbols_path), root)
    write_verified(path, original, updated)
    LOG.info("%s: updated is_primary for %d dates; existing data verified unchanged",
             path.name, len(updated))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pairs = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        if match := FILE_PATTERN.fullmatch(path.name):
            symbols_path = path.with_name(f"{path.stem}_symbols.csv")
            if not symbols_path.exists():
                raise FileNotFoundError(f"Missing paired symbols file: {symbols_path}")
            pairs.append((path, symbols_path, match[1]))
    if not pairs:
        raise SystemExit(f"No non-Fed-funds OHLCV/symbol file pairs found in {DATA_DIR}")
    failures = []
    for path, symbols_path, root in pairs:
        try:
            process_pair(path, symbols_path, root)
        except Exception:
            failures.append(path.name)
            LOG.exception("%s: update failed; file not replaced", path.name)
    if failures:
        raise SystemExit(f"Could not update: {', '.join(failures)}")


if __name__ == "__main__":
    main()
