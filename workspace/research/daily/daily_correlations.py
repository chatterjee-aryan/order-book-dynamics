"""Plot daily open-to-close correlations, distributions, and lag-one autocorrelations.

Instruction file: ../../instructions/daily_move_correlation.md
Run: python -m workspace.research.daily.daily_correlations

Fed funds: PC1 of T0–T6 open-to-close rate changes in basis points, with per-maturity
p5–p95 winsorization and centering fitted separately within each date sample,
matching plot_curve_factors.py. Positive signed PC1 means higher implied rates
at the maturity with the largest absolute loading.

Other assets: primary-contract (close-open)/open. Their returns are not
winsorized. Select using the same day's is_primary flag;
there is no cross-contract price differencing. Daily bars retain UTC dates.
Pearson correlations use pairwise complete observations (minimum three), with
the pairwise count displayed in each heatmap cell.

Open-to-close distributions use primary-contract returns in percent, and
FF rate-change PC1 scores in basis-point units. The distribution PCA is fitted on the
full sample with full-sample winsorization/centering; event subsets reuse these
scores so their distributions have a common basis. Other asset returns are
not centered. Normal days exclude CPI, FOMC, NFP, and PCE dates. Print mean,
sample standard deviation (ddof=1), skewness, and excess kurtosis (normal=0)
for each distribution. Save figures under outputs/daily; no data/report CSVs.

Lag-one autocorrelations pair each asset's move on t with its move on the next
observed trading date. Build pairs BEFORE filtering the first date by event.
Use the common full-sample FF PC1 scores directly, without differencing. Omit
dates where all series are missing from the trading calendar, but do not bridge
an individual asset's missing observation when other assets traded that day.
These pairs exclude overnight returns: both endpoints are open-to-close moves.

The five-day continuation analysis correlates day t's move with the move from
t+1 open to t+5 close, including intervening overnight moves. Hold the contract
selected at t+1 fixed across endpoints. FF uses the t+1 T0–T6 contracts, their
endpoint rate changes, and the daily full-sample PC1 loadings (not a new PCA).
Winsorize the forward FF inputs using their own full-sample p5/p95 thresholds.
Unavailable endpoints are excluded; do not roll, fill, or extend the horizon.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from workspace.research.ff_curve.plot_curve_factors import (
    DATA_FILE as FF_FILE,
    EVENT_FILES,
    WINSOR_LOWER,
    WINSOR_UPPER,
    fit_pca,
    load_event_dates,
    load_rates,
)


ASSETS = ["ZQ_PC1", "ZT", "ZN", "GC", "ES", "BTC", "6E"]
PRICE_FIELDS = ["open", "high", "low", "close"]
MIN_OBSERVATIONS = 3
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "daily"


def load_primary_prices(path: Path, root: str) -> pd.DataFrame:
    """Read primary-contract OHLC and retain its selected tenor per date."""
    data = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=[0]).sort_index()
    if data.index.hasnans or not data.index.is_unique or not data.columns.is_unique:
        raise ValueError(f"{path}: invalid or duplicate dates/columns")
    labels = [f"{root}_T0", f"{root}_T1"]
    primary = pd.DataFrame(index=data.index)
    selected = pd.DataFrame(np.nan, index=data.index, columns=PRICE_FIELDS)
    selected["tenor"] = pd.Series(index=data.index, dtype="object")
    has_bars = pd.DataFrame(index=data.index)
    values_by_label = {}
    for label in labels:
        required = {(label, field) for field in [*PRICE_FIELDS, "volume", "is_primary"]}
        if missing := required.difference(data.columns):
            raise ValueError(f"{path}: missing {sorted(missing)}; run set_futures_roll first")
        raw_flags = data[(label, "is_primary")].astype("string").str.strip().str.lower()
        if not raw_flags.dropna().isin(["true", "false"]).all():
            raise ValueError(f"{path}: invalid is_primary values in {label}")
        primary[label] = raw_flags.eq("true").fillna(False).astype(bool)
        values = data[label][[*PRICE_FIELDS, "volume"]].apply(pd.to_numeric, errors="raise")
        values = values.replace([np.inf, -np.inf], np.nan)
        has_bars[label] = values.notna().any(axis=1)
        if (primary[label] & ~has_bars[label]).any():
            raise ValueError(f"{path}: {label} is primary without a bar")
        values_by_label[label] = values[PRICE_FIELDS]
    counts = primary.sum(axis=1)
    if (counts > 1).any() or (has_bars.any(axis=1) & counts.ne(1)).any():
        raise ValueError(f"{path}: expected exactly one primary wherever a bar exists")
    for label in labels:
        mask = primary[label]
        selected.loc[mask, PRICE_FIELDS] = values_by_label[label].loc[mask, PRICE_FIELDS]
        selected.loc[mask, "tenor"] = label
    invalid = selected["high"] < selected["low"]
    for field in ["open", "close"]:
        invalid |= (selected[field] < selected["low"]) | (selected[field] > selected["high"])
    if invalid.any():
        raise ValueError(f"{path}: inconsistent primary OHLC prices")
    return selected


def fed_funds_pc1(moves: pd.DataFrame) -> tuple[pd.Series, float]:
    """Project the exact winsorized/centered sample used to estimate PC1."""
    loadings, explained, dates = fit_pca(moves)
    complete = moves.loc[dates]
    clipped = complete.clip(lower=complete.quantile(WINSOR_LOWER),
                            upper=complete.quantile(WINSOR_UPPER), axis="columns")
    centered = clipped - clipped.mean(axis=0)
    scores = centered.to_numpy() @ loadings["PC1"].to_numpy()
    return pd.Series(scores, index=dates, name="ZQ_PC1"), float(explained[0])


def draw_heatmap(ax, correlation: pd.DataFrame, counts: pd.DataFrame, title: str):
    """Annotate both Pearson correlations and their actual pairwise counts."""
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    picture = ax.imshow(np.ma.masked_invalid(correlation.to_numpy()), cmap=cmap,
                        vmin=-1, vmax=1)
    for i in range(len(ASSETS)):
        for j in range(len(ASSETS)):
            value = correlation.iloc[i, j]
            label = "N/A" if pd.isna(value) else f"{value:.2f}"
            ax.text(j, i, f"{label}\nn={counts.iloc[i, j]:,}", ha="center", va="center",
                    fontsize=8, color="white" if pd.notna(value) and abs(value) > 0.55 else "black")
    ax.set_xticks(range(len(ASSETS)), ASSETS, rotation=45, ha="right")
    ax.set_yticks(range(len(ASSETS)), ASSETS)
    ax.set_title(title)
    return picture


def save_figure(fig, name: str) -> None:
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"Saved {path}")


def plot_return_distributions(returns: pd.DataFrame,
                             samples: dict[str, pd.DatetimeIndex], explained: float) -> None:
    """Use identical bins per asset across all sample-specific histograms."""
    bins = {}
    for asset in ASSETS:
        values = returns[asset].dropna().to_numpy()
        if len(values):
            edges = np.histogram_bin_edges(values, bins="fd")
            bins[asset] = np.linspace(edges[0], edges[-1], min(len(edges) - 1, 80) + 1)
    for sample, dates in samples.items():
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
        fig.canvas.manager.set_window_title(f"Open-to-close distributions — {sample}")
        for ax, asset in zip(axes.flat, ASSETS):
            values = returns[asset].reindex(dates).dropna()
            ax.set_title(f"{asset} — n={len(values):,}")
            ax.set_xlabel("Centered FF rate-change PC1 (bp)" if asset == "ZQ_PC1" else "(Close − open) / open (%)")
            ax.set_ylabel("Density")
            if len(values):
                ax.hist(values, bins=bins[asset], density=True, color="steelblue",
                        edgecolor="white", alpha=0.85)
            else:
                ax.text(0.5, 0.5, "No observations", ha="center", va="center", transform=ax.transAxes)
            ax.grid(axis="y", alpha=0.2)
        axes.flat[-1].set_visible(False)
        fig.suptitle(f"Daily open-to-close distributions — {sample}\n"
                     f"FF: common full-sample PC1 ({explained:.1%} variance), p5–p95 winsorized inputs")
        save_figure(fig, f"open_close_distributions_{sample.lower().replace(' ', '_')}")


def print_return_moments(returns: pd.DataFrame, samples: dict[str, pd.DatetimeIndex]) -> None:
    rows = []
    for sample, dates in samples.items():
        for asset in ASSETS:
            values = returns[asset].reindex(dates).dropna()
            rows.append({"sample": sample, "asset": asset,
                         "units": "PC1 bp" if asset == "ZQ_PC1" else "% of open",
                         "n": len(values),
                         "mean": values.mean(), "std_dev": values.std(ddof=1),
                         "skew": values.skew(), "excess_kurtosis": values.kurt()})
    print("\nOpen-to-close moments: sample standard deviation; excess kurtosis (normal=0).")
    print("Normal dates exclude all four event categories. FF scores share one full-sample basis.")
    print(pd.DataFrame(rows).set_index(["sample", "asset"]).to_string(
        float_format=lambda value: f"{value:.5f}"))


def lag_one_autocorrelations(returns: pd.DataFrame,
                             samples: dict[str, pd.DatetimeIndex]) -> pd.DataFrame:
    """Condition on the first date; the second is the next trading observation."""
    calendar = returns.sort_index().dropna(how="all")
    if not calendar.index.is_unique:
        raise ValueError("Lag-one autocorrelations require unique dates")
    # Shared all-missing dates (e.g., full closures) are not trading observations.
    # Keep individual missing values so we cannot jump over another trading day.
    next_moves = calendar.shift(-1)
    rows = []
    for sample, first_dates in samples.items():
        mask = calendar.index.isin(first_dates)
        for asset in ASSETS:
            pairs = pd.DataFrame({"first": calendar[asset], "next": next_moves[asset]})
            pairs = pairs.loc[mask].dropna()
            correlation = np.nan
            if (len(pairs) >= MIN_OBSERVATIONS
                    and pairs["first"].nunique() > 1 and pairs["next"].nunique() > 1):
                correlation = pairs["first"].corr(pairs["next"])
            rows.append({"sample": sample, "asset": asset, "n_pairs": len(pairs),
                         "lag1_autocorr": correlation})
    return pd.DataFrame(rows)


def plot_lag_one_autocorrelations(summary: pd.DataFrame, *, forward_week: bool = False) -> None:
    """Print and plot a sample-by-asset table of continuation or reversal."""
    order = summary["sample"].drop_duplicates().tolist()
    value_column = "forward5_corr" if forward_week else "lag1_autocorr"
    correlation = summary.pivot(index="sample", columns="asset", values=value_column)
    correlation = correlation.reindex(index=order, columns=ASSETS)
    counts = summary.pivot(index="sample", columns="asset", values="n_pairs")
    counts = counts.reindex(index=order, columns=ASSETS)
    if forward_week:
        print("\nFive-trading-day continuation: corr(move[t], move[t+1 open to t+5 close]).")
        print("Future price endpoints use the SAME contract chosen at t+1; missing endpoints exclude the pair.")
        print("FF forward rate changes use daily full-sample PC1 loadings, with full-forward-sample p5/p95 caps.")
    else:
        print("\nLag-one autocorrelation: corr(move[t], move[next trading date]).")
    print("Sample labels apply to the FIRST day only; the second day may have any event label.")
    print("Day-t FF uses common full-sample PC1 scores directly; no additional difference is taken.")
    print(correlation.to_string(float_format=lambda value: f"{value:.4f}"))
    print("\nValid observation pairs:")
    print(counts.to_string())
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    title = ("Day-t move versus next-five-trading-day move" if forward_week
             else "Lag-one autocorrelation of open-to-close moves")
    fig.canvas.manager.set_window_title(title)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    picture = ax.imshow(np.ma.masked_invalid(correlation.to_numpy()), cmap=cmap,
                        vmin=-1, vmax=1, aspect="auto")
    for i in range(len(order)):
        for j in range(len(ASSETS)):
            value = correlation.iloc[i, j]
            label = "N/A" if pd.isna(value) else f"{value:.2f}"
            ax.text(j, i, f"{label}\nn={int(counts.iloc[i, j]):,}", ha="center", va="center",
                    color="white" if pd.notna(value) and abs(value) > 0.55 else "black")
    ax.set_xticks(range(len(ASSETS)), ASSETS, rotation=45, ha="right")
    ax.set_yticks(range(len(order)), order)
    ax.set_ylabel("Sample selected by first day")
    ax.set_title(f"{title}\n"
                 "Positive: linear continuation; negative: linear reversal")
    fig.colorbar(picture, ax=ax, shrink=0.8, label="Pearson autocorrelation")
    filename = "open_close_forward5_continuation" if forward_week else "open_close_lag1_autocorrelations"
    save_figure(fig, filename)


def forward_contract_moves(root: str, calendar: pd.DatetimeIndex,
                           primary: pd.DataFrame | None = None) -> pd.DataFrame:
    """Look up t+1 open and t+5 close by the SAME full delivery year/month."""
    path = FF_FILE.with_name(FF_FILE.name.replace("ZQ_", f"{root}_", 1))
    symbols = pd.read_csv(path.with_name(f"{path.stem}_symbols.csv"), parse_dates=["date"])
    bars = pd.read_csv(path.with_name(f"{path.stem}_contracts.csv"), parse_dates=["date"])
    keys = ["contract_year", "contract_month"]
    for label, table, required in [
        ("symbols", symbols, {"date", "tenor", *keys}),
        ("contracts", bars, {"date", "open", "close", *keys}),
    ]:
        if missing := required.difference(table.columns):
            raise ValueError(f"{root} {label}: missing fields {sorted(missing)}")
        if table["date"].isna().any() or not table["date"].eq(table["date"].dt.normalize()).all():
            raise ValueError(f"{root} {label}: invalid calendar dates")
    if symbols.duplicated(["date", "tenor"]).any() or bars.duplicated(["date", *keys]).any():
        raise ValueError(f"{root}: duplicate symbol or contract-date records")
    endpoints = pd.DataFrame({"anchor": calendar, "date": pd.Series(calendar).shift(-1),
                              "end_date": pd.Series(calendar).shift(-5)})
    if root == "ZQ":
        requests = pd.concat([endpoints.assign(tenor=f"ZQ_T{i}") for i in range(7)], ignore_index=True)
        columns = [f"ZQ_T{i}" for i in range(7)]
    else:
        if primary is None:
            raise ValueError(f"{root}: primary selections required")
        requests = endpoints.assign(tenor=primary.reindex(calendar)["tenor"].shift(-1).to_numpy())
        columns = [root]
    requests = requests.dropna(subset=["date", "end_date", "tenor"])
    requests = requests.merge(symbols[["date", "tenor", *keys]], on=["date", "tenor"],
                              how="left", validate="many_to_one")
    requests = requests.dropna(subset=keys)
    requests = requests.merge(bars[["date", *keys, "open"]], on=["date", *keys],
                              how="left", validate="many_to_one")
    closing = bars[["date", *keys, "close"]].rename(columns={"date": "end_date"})
    requests = requests.merge(closing, on=["end_date", *keys], how="left", validate="many_to_one")
    opening = pd.to_numeric(requests["open"], errors="raise").replace([np.inf, -np.inf], np.nan)
    closing = pd.to_numeric(requests["close"], errors="raise").replace([np.inf, -np.inf], np.nan)
    if root == "ZQ":
        # (100 - closing price) - (100 - opening price), in basis points.
        requests["move"] = (opening - closing) * 100.0
        output = requests.pivot(index="anchor", columns="tenor", values="move")
    else:
        requests["move"] = (closing / opening.where(opening > 0) - 1.0) * 100.0
        output = requests.set_index("anchor")[["move"]].rename(columns={"move": root})
    output = output.reindex(index=calendar, columns=columns)
    return output.replace([np.inf, -np.inf], np.nan)


def five_day_continuation(returns: pd.DataFrame, primary_prices: dict[str, pd.DataFrame],
                          ff_movements: pd.DataFrame,
                          samples: dict[str, pd.DatetimeIndex]) -> pd.DataFrame:
    """Correlate the first day's move with the next five-day endpoint move."""
    calendar = returns.sort_index().dropna(how="all").index
    following = pd.DataFrame(index=calendar, columns=ASSETS, dtype=float)
    for root in ASSETS[1:]:
        following[root] = forward_contract_moves(root, calendar, primary_prices[root])[root]
    forward_ff = forward_contract_moves("ZQ", calendar)
    try:
        loadings, _, _ = fit_pca(ff_movements)
        complete = forward_ff.dropna(how="any")
        if len(complete) < MIN_OBSERVATIONS:
            raise ValueError("Fewer than three complete forward FF observations")
        clipped = complete.clip(lower=complete.quantile(WINSOR_LOWER),
                                upper=complete.quantile(WINSOR_UPPER), axis="columns")
        scores = (clipped - clipped.mean()).to_numpy() @ loadings["PC1"].to_numpy()
        following.loc[complete.index, "ZQ_PC1"] = scores
    except ValueError as exc:
        print(f"Five-day FF continuation unavailable: {exc}")
    rows = []
    for sample, dates in samples.items():
        for asset in ASSETS:
            pairs = pd.DataFrame({"first": returns[asset].reindex(calendar), "future": following[asset]})
            pairs = pairs.loc[pairs.index.isin(dates)].dropna()
            correlation = np.nan
            if (len(pairs) >= MIN_OBSERVATIONS
                    and pairs["first"].nunique() > 1 and pairs["future"].nunique() > 1):
                correlation = pairs["first"].corr(pairs["future"])
            rows.append({"sample": sample, "asset": asset, "n_pairs": len(pairs),
                         "forward5_corr": correlation})
    return pd.DataFrame(rows)


def main() -> None:
    rates = load_rates(FF_FILE)
    ff_movements = (rates["close"] - rates["open"]) * 100.0
    primary_prices = {
        root: load_primary_prices(FF_FILE.with_name(FF_FILE.name.replace("ZQ_", f"{root}_", 1)), root)
        for root in ASSETS[1:]
    }
    other_returns = {
        root: ((prices["close"] - prices["open"]) / prices["open"].where(prices["open"] > 0)).rename(root)
        for root, prices in primary_prices.items()
    }
    events = {name: load_event_dates(path) for name, path in EVENT_FILES.items()}
    index = rates["open"].index
    samples = {"All dates": index}
    samples.update({f"{name} dates": index[index.isin(dates)] for name, dates in events.items()})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for sample, dates in samples.items():
        fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
        fig.canvas.manager.set_window_title(f"Daily correlations — {sample}")
        print(f"\n{sample}: {len(dates):,} selected dates")
        ff_sample = ff_movements.loc[dates]
        try:
            scores, explained = fed_funds_pc1(ff_sample)
        except ValueError as exc:
            print(f"{sample} — Open-to-close: FF PCA unavailable: {exc}")
            scores = pd.Series(index=dates, dtype=float, name="ZQ_PC1")
            explained = np.nan
        columns = {"ZQ_PC1": scores}
        columns.update({root: other_returns[root].reindex(dates) for root in ASSETS[1:]})
        values = pd.DataFrame(columns, index=dates).reindex(columns=ASSETS)
        values = values.replace([np.inf, -np.inf], np.nan)
        correlation = values.corr(method="pearson", min_periods=MIN_OBSERVATIONS)
        available = values.notna().astype("int64")
        counts = available.T @ available
        print("Open-to-close correlation matrix:")
        print(correlation.to_string(float_format=lambda value: f"{value:.4f}"))
        print("Valid observations per series: "
              + ", ".join(f"{asset}={count:,}" for asset, count in values.notna().sum().items()))
        picture = draw_heatmap(ax, correlation, counts, f"Open-to-close\nFF PC1 variance: {explained:.1%}")
        fig.colorbar(picture, ax=ax, shrink=0.8, label="Pearson correlation")
        fig.suptitle(f"Daily cross-asset correlations — {sample}\n"
                     "FF PC1: sample p5–p95 winsorized; other assets: primary-contract returns")
        save_figure(fig, f"open_close_correlations_{sample.lower().replace(' ', '_')}")
    try:
        distribution_scores, distribution_explained = fed_funds_pc1(ff_movements)
    except ValueError as exc:
        print(f"Open-to-close FF distribution PCA unavailable: {exc}")
        distribution_scores = pd.Series(index=index, dtype=float, name="ZQ_PC1")
        distribution_explained = np.nan
    returns = pd.DataFrame({"ZQ_PC1": distribution_scores,
                           **{root: other_returns[root].reindex(index) * 100.0 for root in ASSETS[1:]}},
                          index=index).reindex(columns=ASSETS).replace([np.inf, -np.inf], np.nan)
    event_union = pd.DatetimeIndex([])
    for dates in events.values():
        event_union = event_union.union(dates)
    distribution_samples = {**samples, "Normal dates": index[~index.isin(event_union)]}
    print(f"\nOpen-to-close distribution FF PC1: fitted on {distribution_scores.notna().sum():,} complete dates; "
          "same basis and winsorization thresholds used for every distribution.")
    plot_return_distributions(returns, distribution_samples, distribution_explained)
    print_return_moments(returns, distribution_samples)
    lag_summary = lag_one_autocorrelations(returns, distribution_samples)
    plot_lag_one_autocorrelations(lag_summary)
    weekly_summary = five_day_continuation(returns, primary_prices, ff_movements, distribution_samples)
    plot_lag_one_autocorrelations(weekly_summary, forward_week=True)
    plt.show()


if __name__ == "__main__":
    main()
