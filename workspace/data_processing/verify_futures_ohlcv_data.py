"""Simple local verification of downloaded futures OHLCV CSVs.

Instruction file: ../instructions/verifying_futures_data.md
Run: python -m workspace.data_processing.verify_futures_ohlcv_data

Reads T-labelled and contract-level outputs from get_futures_ohlcv_data.py.
Missing history and empty candidate slots are reported, not assumed to be data
errors. This checks the saved files; it cannot prove provider completeness.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "ohlcv"
ROOTS = ("ZQ", "ZT", "ZN", "GC", "ES", "BTC", "6E")
FIELDS = ["open", "high", "low", "close", "volume"]
FILE_PATTERN = re.compile(r"^(ZQ|ZT|ZN|GC|ES|BTC|6E)_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.csv$")


def bar_checks(values: pd.DataFrame) -> dict:
    """Count complete/missing bars and invalid numbers or price relationships."""
    numeric = values.apply(pd.to_numeric, errors="coerce")
    finite = pd.DataFrame(np.isfinite(numeric), index=numeric.index, columns=numeric.columns)
    prices = numeric[["open", "high", "low", "close"]]
    complete_prices = finite[["open", "high", "low", "close"]].all(axis=1)
    bad_prices = complete_prices & (
        (prices["low"] > prices["high"])
        | (prices["open"] < prices["low"]) | (prices["open"] > prices["high"])
        | (prices["close"] < prices["low"]) | (prices["close"] > prices["high"])
    )
    bad_volume = finite["volume"] & (
        (numeric["volume"] < 0) | (numeric["volume"] % 1 != 0)
    )
    present = numeric.notna().any(axis=1)
    return {
        "bars_with_data": int(present.sum()),
        "complete_bars": int(finite.all(axis=1).sum()),
        "empty_bars": int(numeric.isna().all(axis=1).sum()),
        "partial_bars": int((numeric.notna().any(axis=1) & numeric.isna().any(axis=1)).sum()),
        "nan_cells": int(numeric.isna().sum().sum()),
        "nonnumeric_cells": int((values.notna() & numeric.isna()).sum().sum()),
        "infinite_cells": int((numeric.notna() & ~finite).sum().sum()),
        "invalid_ohlc_bars": int(bad_prices.sum()),
        "invalid_volume_bars": int(bad_volume.sum()),
    }


def date_text(values) -> str:
    dates = pd.DatetimeIndex(values).dropna()
    return "" if dates.empty else dates.min().strftime("%Y-%m-%d")


def summarize(path: Path, root: str, start: str, end: str,
              kind: str) -> tuple[dict, list[dict]]:
    row = {"asset": root, "kind": kind, "file": path.name,
           "requested_start": start, "requested_end": end}
    if not path.exists():
        return {**row, "status": "error", "error": "file missing"}, []
    try:
        if kind == "tenors":
            frame = pd.read_csv(path, header=[0, 1], index_col=0)
            labels = [f"{root}_T{i}" for i in range(7 if root == "ZQ" else 2)]
            expected = pd.MultiIndex.from_product([labels, FIELDS])
            optional = {(label, flag) for label in labels for flag in ("expired", "is_primary")}
            if (not set(expected).issubset(frame.columns)
                    or not set(frame.columns).issubset(set(expected) | optional)
                    or not frame.columns.is_unique):
                raise ValueError("Unexpected or duplicate tenor/OHLCV columns")
            dates = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce"))
            blocks = [(label, frame[label].reset_index(drop=True)[FIELDS]) for label in labels]
        else:
            frame = pd.read_csv(path)
            required = {"date", "canonical_symbol", "raw_symbol", "instrument_id", *FIELDS}
            if missing := required.difference(frame.columns):
                raise ValueError(f"Missing columns: {sorted(missing)}")
            dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="coerce"))
            labels = []
            blocks = [("contracts", frame[FIELDS])]
            row["distinct_contracts"] = frame["canonical_symbol"].nunique()
            row["missing_identifiers"] = int(frame[["canonical_symbol", "raw_symbol", "instrument_id"]].isna().any(axis=1).sum())
            row["duplicate_contract_dates"] = int(frame.duplicated(["date", "canonical_symbol"]).sum())
        row.update(rows=len(frame), first_date=date_text(dates),
                   last_date=dates.max().strftime("%Y-%m-%d") if dates.notna().any() else "",
                   invalid_dates=int(dates.isna().sum()),
                   unsorted_dates=not dates.dropna().is_monotonic_increasing,
                   dates_outside_range=int(((dates < pd.Timestamp(start)) | (dates > pd.Timestamp(end))).sum()))
        if kind == "tenors":
            row["duplicate_dates"] = int(dates.duplicated().sum())
            expected_dates = pd.bdate_range(start, end)
            row["missing_weekday_rows"] = len(expected_dates.difference(dates))
        summaries = []
        observed = np.zeros(len(frame), dtype=bool)
        for label, values in blocks:
            stats = bar_checks(values)
            present = values.apply(pd.to_numeric, errors="coerce").notna().any(axis=1).to_numpy()
            observed |= present
            seen = dates[present].dropna()
            summaries.append({"asset": root, "tenor": label, "file": path.name,
                              "first_data": date_text(seen),
                              "last_data": seen.max().strftime("%Y-%m-%d") if len(seen) else "",
                              **stats})
            for key, value in stats.items():
                row[key] = row.get(key, 0) + value
        seen = dates[observed].dropna()
        row["first_data"] = date_text(seen)
        row["last_data"] = seen.max().strftime("%Y-%m-%d") if len(seen) else ""
        row["dates_with_data"] = seen.nunique()
        error_keys = ["invalid_dates", "dates_outside_range", "duplicate_dates",
                      "duplicate_contract_dates", "missing_identifiers", "missing_weekday_rows",
                      "partial_bars", "nonnumeric_cells", "infinite_cells",
                      "invalid_ohlc_bars", "invalid_volume_bars"]
        issues = [key for key in error_keys if row.get(key, 0)]
        if row["unsorted_dates"]:
            issues.append("unsorted_dates")
        row["status"] = "error" if issues else ("no_data" if not observed.any() else "ok")
        row["error"] = ", ".join(issues)
        return row, summaries if kind == "tenors" else []
    except Exception as exc:
        return {**row, "status": "error", "error": f"{type(exc).__name__}: {exc}"}, []


def main() -> None:
    periods = set()
    for path in DATA_DIR.glob("*.csv"):
        if match := FILE_PATTERN.fullmatch(path.name):
            periods.add((match[2], match[3]))
    if not periods:
        raise SystemExit(f"No futures OHLCV files found in {DATA_DIR}")
    rows, tenors = [], []
    for start, end in sorted(periods):
        for root in ROOTS:
            for kind, suffix in [("tenors", ""), ("contracts", "_contracts")]:
                path = DATA_DIR / f"{root}_{start}_{end}{suffix}.csv"
                row, detail = summarize(path, root, start, end, kind)
                rows.append(row)
                tenors.extend(detail)
        status_path = DATA_DIR / f"download_status_{start}_{end}.csv"
        if status_path.exists():
            print(f"\nDownload status ({start} through {end}):")
            print(pd.read_csv(status_path).to_string(index=False))
    summary = pd.DataFrame(rows)
    detail = pd.DataFrame(tenors)
    columns = ["asset", "kind", "rows", "first_data", "last_data", "bars_with_data",
               "nan_cells", "partial_bars", "invalid_ohlc_bars", "status", "error"]
    print("\nFile checks:")
    print(summary.reindex(columns=columns).to_string(index=False))
    print("\nPer-tenor coverage:")
    print(detail.to_string(index=False))
    print("Missing observations alone do not distinguish absent history, no trades, holidays, or provider gaps.")


if __name__ == "__main__":
    main()
