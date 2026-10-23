"""Plot the first three PCA loadings of Fed funds curve movements.

Instruction file: ../../instructions/ff_curve_pca.md
Run: python -m workspace.research.ff_curve.plot_curve_factors

Rates are 100 minus futures prices. Open-to-close changes are signed; high-low
ranges are unsigned and describe variation in movement magnitudes. PCA uses
centered basis-point observations without standardizing individual maturities.
Each analysis independently excludes dates missing any of the seven inputs.
Before centering, each maturity's moves are winsorized to its sample p5/p95
computed on those complete dates. Extreme moves are capped, not dropped.
Fits the full sample and CPI, FOMC, NFP, and PCE event-date subsets separately.
Event dates are used as supplied, including both days of multi-day FOMC meetings
if present in the source; dates are not shifted to another trading day.
Level, slope, and curvature are possible loading shapes, not imposed factors.
Displays plots only; does not write data, figures, or reports.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = PROJECT_ROOT / "data" / "ohlcv" / "ZQ_2010-06-07_2026-10-21.csv"
TENORS = [f"ZQ_T{month}" for month in range(7)]
FIELDS = ["open", "high", "low", "close"]
N_COMPONENTS = 3
WINSOR_LOWER = 0.05
WINSOR_UPPER = 0.95
EVENT_FILES = {
    "CPI": PROJECT_ROOT / "data" / "event" / "cpi_dates.csv",
    "FOMC": PROJECT_ROOT / "data" / "event" / "fomc_dates.csv",
    "NFP": PROJECT_ROOT / "data" / "event" / "nfp_dates.csv",
    "PCE": PROJECT_ROOT / "data" / "event" / "pce_dates.csv",
}


def load_rates(path: Path) -> dict[str, pd.DataFrame]:
    """Convert price OHLC to implied-rate OHLC in percentage points."""
    prices = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=[0])
    if prices.index.hasnans or not prices.index.is_unique or not prices.columns.is_unique:
        raise ValueError("Input must have valid, unique dates and columns")
    required = {(tenor, field) for tenor in TENORS for field in FIELDS}
    if missing := required.difference(prices.columns):
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")
    prices = prices.sort_index()
    fields = {}
    for field in FIELDS:
        values = prices.xs(field, axis=1, level=1)[TENORS]
        fields[field] = values.apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    invalid = (fields["high"] < fields["low"])
    for field in ("open", "close"):
        invalid |= (fields[field] > fields["high"]) | (fields[field] < fields["low"])
    if invalid.any().any():
        raise ValueError("Input contains inconsistent OHLC price relationships")
    # The transformation is decreasing: a price low is a rate high.
    return {
        "open": 100.0 - fields["open"],
        "close": 100.0 - fields["close"],
        "high": 100.0 - fields["low"],
        "low": 100.0 - fields["high"],
    }


def fit_pca(moves: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex]:
    """Winsorize each maturity, then fit centered covariance PCA using SVD."""
    complete = moves.dropna(how="any")
    if len(complete) < N_COMPONENTS + 1:
        raise ValueError("PCA requires at least four complete seven-maturity observations")
    lower = complete.quantile(WINSOR_LOWER)
    upper = complete.quantile(WINSOR_UPPER)
    winsorized = complete.clip(lower=lower, upper=upper, axis="columns")
    values = winsorized.to_numpy(dtype=float)
    centered = values - values.mean(axis=0)
    _, singular_values, directions = np.linalg.svd(centered, full_matrices=False)
    total = np.square(singular_values).sum()
    if total <= 0:
        raise ValueError("PCA is undefined because all observations are constant")
    loadings = directions[:N_COMPONENTS].T.copy()
    # Eigenvector signs are arbitrary. Orient the largest absolute loading
    # positively for reproducible plots without changing the PCA solution.
    for component in range(N_COMPONENTS):
        largest = np.abs(loadings[:, component]).argmax()
        if loadings[largest, component] < 0:
            loadings[:, component] *= -1
    explained = np.square(singular_values[:N_COMPONENTS]) / total
    return (
        pd.DataFrame(loadings, index=pd.Index(range(7), name="month"),
                     columns=[f"PC{i + 1}" for i in range(N_COMPONENTS)]),
        explained,
        complete.index,
    )


def load_event_dates(path: Path) -> pd.DatetimeIndex:
    """Read unique calendar dates without changing their event-day assignment."""
    events = pd.read_csv(path)
    if "date" not in events:
        raise ValueError(f"{path}: missing date column")
    values = events["date"].astype("string").str.strip()
    separators = values.str.fullmatch(r"-+", na=False)
    if separators.any():
        print(f"{path.name}: skipped {int(separators.sum())} separator lines")
    values = values.loc[~separators]
    dates = pd.DatetimeIndex(pd.to_datetime(values, format="%Y-%m-%d", errors="raise"))
    if dates.hasnans:
        raise ValueError(f"{path}: missing event dates")
    if dates.tz is not None:
        raise ValueError(f"{path}: expected timezone-naive calendar dates")
    return dates.normalize().unique().sort_values()


def plot_sample(sample: str, movements: dict[str, pd.DataFrame]) -> None:
    """Plot both types of moves, fitting each PCA on this sample only."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, constrained_layout=True)
    fig.canvas.manager.set_window_title(f"Fed funds PCA — {sample}")
    for ax, (name, moves) in zip(axes, movements.items()):
        complete_count = len(moves.dropna(how="any"))
        print(f"{sample} — {name}: {len(moves):,} selected dates; "
              f"{complete_count:,} complete, {len(moves) - complete_count:,} excluded")
        if complete_count < N_COMPONENTS + 1:
            ax.set_title(name)
            ax.text(0.5, 0.5, "Fewer than four complete dates; PCA skipped",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        loadings, explained, dates = fit_pca(moves)
        for component, column in enumerate(loadings.columns):
            ax.plot(loadings.index, loadings[column], marker="o", linewidth=2,
                    label=f"{column} ({explained[component]:.1%})")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xticks(range(7))
        ax.set_xlabel("Calendar-month horizon (T0–T6)")
        ax.set_title(f"{name}\n{len(dates):,} complete dates; first 3 PCs: {explained.sum():.1%}")
        ax.grid(alpha=0.25)
        ax.legend(title="Explained variance")
        print(f"{sample} — {name}: complete-date coverage "
              f"{dates.min():%Y-%m-%d} through {dates.max():%Y-%m-%d}")
    axes[0].set_ylabel("PCA loading (unit eigenvector)")
    fig.suptitle(f"Fed funds futures curve — {sample}\n"
                 "PCA of daily rate movements (sample p5–p95 winsorized)")


def main() -> None:
    rates = load_rates(DATA_FILE)
    movements = {
        "Open-to-close rate change": (rates["close"] - rates["open"]) * 100.0,
        "High–low rate range (unsigned)": (rates["high"] - rates["low"]) * 100.0,
    }
    # Read all event inputs before opening figures so missing files fail clearly.
    events = {name: load_event_dates(path) for name, path in EVENT_FILES.items()}
    available = rates["open"].index
    plot_sample("All dates", movements)
    for name, dates in events.items():
        in_range = dates[(dates >= available.min()) & (dates <= available.max())]
        matched = available[available.isin(in_range)]
        print(f"\n{name}: {len(dates):,} unique source dates, {len(in_range):,} within data range; "
              f"{len(matched):,} matched dates, {len(in_range.difference(available)):,} absent from data index")
        subset = {measure: values.loc[matched] for measure, values in movements.items()}
        plot_sample(f"{name} dates", subset)
    plt.show()


if __name__ == "__main__":
    main()
