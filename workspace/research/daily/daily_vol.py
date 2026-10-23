"""Compare normalized daily high-low ranges on normal and event dates.

Instruction file: ../../instructions/daily_move_vol.md
Run: python -m workspace.research.daily.daily_vol

For each asset, divide today's range by the mean range on the preceding 60
shared observed trading dates, excluding today. Require all 60 past values;
missing or nonpositive baselines yield NaN. Baselines include event dates.
Normal dates exclude the union of CPI, FOMC, NFP, and PCE dates.

For FF, estimate PC1 from centered, per-maturity p5/p95-winsorized rate ranges
over the full sample. Apply its loadings to the uncentered winsorized ranges
to obtain a range level before rolling normalization. A centered PC1 score
has no meaningful positive rolling-mean denominator. Negative projected levels
are excluded. The same weights and caps are used on normal and event dates.
The six other assets use the primary contract's raw price high minus low.

Show one grouped bar chart at a time and save PNGs under outputs/daily.
No input data is modified and no reports or data CSVs are saved.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from workspace.research.daily.daily_correlations import ASSETS, load_primary_prices
from workspace.research.ff_curve.plot_curve_factors import (
    DATA_FILE as FF_FILE,
    EVENT_FILES,
    WINSOR_LOWER,
    WINSOR_UPPER,
    fit_pca,
    load_event_dates,
    load_rates,
)


LOOKBACK = 60
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "daily"


def fed_funds_range_level(rates: dict[str, pd.DataFrame]) -> pd.Series:
    """Use full-sample range-PC1 weights without subtracting the projected mean."""
    moves = (rates["high"] - rates["low"]) * 100.0
    loadings, explained, dates = fit_pca(moves)
    complete = moves.loc[dates]
    clipped = complete.clip(lower=complete.quantile(WINSOR_LOWER),
                            upper=complete.quantile(WINSOR_UPPER), axis="columns")
    level = pd.Series(clipped.to_numpy() @ loadings["PC1"].to_numpy(),
                      index=dates, name="ZQ_PC1")
    negative = level.lt(0)
    if negative.any():
        print(f"FF range PC1: excluding {int(negative.sum())} negative projected levels")
    print(f"FF range PCA: {len(dates):,} complete dates; PC1 explains {explained[0]:.1%}. "
          "Using uncentered projected range levels for the rolling ratio.")
    return level.mask(negative)


def normalize_ranges(ranges: pd.DataFrame) -> pd.DataFrame:
    """Normalize on a shared trading calendar without filling missing values."""
    if not ranges.index.is_unique:
        raise ValueError("Range dates must be unique")
    ranges = ranges.replace([np.inf, -np.inf], np.nan).sort_index().dropna(how="all")
    if ranges.lt(0).any().any():
        raise ValueError("Ranges cannot be negative")
    baseline = ranges.shift(1).rolling(LOOKBACK, min_periods=LOOKBACK).mean()
    normalized = ranges / baseline.where(baseline > 0)
    return normalized.replace([np.inf, -np.inf], np.nan)


def sample_summary(normalized: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    values = normalized.loc[normalized.index.isin(dates)]
    return pd.DataFrame({"n": values.count(), "median": values.median(),
                         "p75": values.quantile(0.75)}).reindex(ASSETS)


def plot_comparison(event: str, normal: pd.DataFrame, event_stats: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    fig.canvas.manager.set_window_title(f"Normalized daily ranges — {event} vs normal")
    x = np.arange(len(ASSETS))
    width = 0.19
    series = [
        ("Normal median", normal["median"], "#9ecae1"),
        ("Normal 75th percentile", normal["p75"], "#3182bd"),
        (f"{event} median", event_stats["median"], "#fdae6b"),
        (f"{event} 75th percentile", event_stats["p75"], "#e6550d"),
    ]
    for position, (label, heights, color) in enumerate(series):
        ax.bar(x + (position - 1.5) * width, heights.to_numpy(), width=width,
               label=label, color=color)
    ax.axhline(1, color="gray", linestyle="--", linewidth=1, label="Prior-60-date mean")
    ax.set_xticks(x, ASSETS)
    ax.set_xlabel("Asset (FF represented by range PC1 level)")
    ax.set_ylabel("Daily range / preceding 60-trading-date mean range")
    ax.set_title(f"Daily range intensity: {event} versus normal dates\n"
                 "Normal excludes all four event categories; today excluded from baseline")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(ncol=2)
    path = OUTPUT_DIR / f"daily_vol_{event.lower()}_vs_normal.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"Saved {path}")
    plt.show(block=True)
    plt.close(fig)


def main() -> None:
    rates = load_rates(FF_FILE)
    index = rates["open"].index
    ranges = pd.DataFrame(index=index, columns=ASSETS, dtype=float)
    ranges["ZQ_PC1"] = fed_funds_range_level(rates).reindex(index)
    for root in ASSETS[1:]:
        path = FF_FILE.with_name(FF_FILE.name.replace("ZQ_", f"{root}_", 1))
        primary = load_primary_prices(path, root)
        ranges[root] = (primary["high"] - primary["low"]).reindex(index)
    events = {name: load_event_dates(path) for name, path in EVENT_FILES.items()}
    normalized = normalize_ranges(ranges)
    event_union = pd.DatetimeIndex([])
    for dates in events.values():
        event_union = event_union.union(dates)
    normal_dates = normalized.index[~normalized.index.isin(event_union)]
    normal = sample_summary(normalized, normal_dates)
    print(f"\nNormalized ranges require {LOOKBACK} complete preceding trading observations. "
          "Missing ranges and invalid baselines remain unavailable.")
    print("\nNormal-date normalized ranges:")
    print(normal.to_string(float_format=lambda value: f"{value:.4f}"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for event, dates in events.items():
        event_stats = sample_summary(normalized, dates)
        print(f"\n{event}-date normalized ranges:")
        print(event_stats.to_string(float_format=lambda value: f"{value:.4f}"))
        plot_comparison(event, normal, event_stats)


if __name__ == "__main__":
    main()
