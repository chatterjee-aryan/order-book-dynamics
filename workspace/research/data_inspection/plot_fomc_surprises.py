"""Plot four normalized FOMC channels as meeting deciles and their divergence.

Run from project root:
    python plot_fomc_vector_deciles_4channel.py

Input: data/processed/fomc/fomc_surprises.csv
Outputs:
    data/processed/fomc/fomc_event_vector_deciles.png
    data/processed/fomc/fomc_event_divergence.png
    data/processed/fomc/fomc_event_divergence.csv

Deciles are computed independently for each channel over valid FOMC meetings.
Divergence = max decile - min decile across all four channels.
Extreme = at least one decile >= 9 and another <= 3.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT = Path('data/processed/fomc/fomc_surprises.csv')
OUTPUT_DIR = Path('data/processed/fomc')
DECILE_PLOT = OUTPUT_DIR / 'fomc_event_vector_deciles.png'
DIVERGENCE_PLOT = OUTPUT_DIR / 'fomc_event_divergence.png'
DIVERGENCE_CSV = OUTPUT_DIR / 'fomc_event_divergence.csv'

CHANNELS = [
    ('tcm_hl_bp_normalized', 'Policy path: six-contract TCM'),
    ('target_hl_bp_normalized', 'Target rate: current-month / fallback'),
    ('curve_slope_abs_bp_normalized', 'Yield curve: |Δ(10Y − 2Y)|'),
    ('inflation_abs_bp_normalized', 'Inflation compensation: |Δ(10Y breakeven)|'),
]


def meeting_deciles(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan)
    ranks = numeric.rank(method='average', pct=True)
    return np.ceil(10 * ranks).clip(1, 10).astype('Int64')


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT, parse_dates=['meeting_date']).sort_values('meeting_date').reset_index(drop=True)
    required = ['meeting_date'] + [c for c, _ in CHANNELS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns in {INPUT}: {missing}')

    decile_cols = []
    for column, title in CHANNELS:
        decile_col = column.replace('_normalized', '_decile')
        df[decile_col] = meeting_deciles(df[column])
        decile_cols.append(decile_col)
        print(f'{title}: {df[decile_col].notna().sum()} valid, {df[decile_col].isna().sum()} missing')

    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(15, 7), sharex=True, constrained_layout=True)
    for ax, (_, title), decile_col in zip(axes, CHANNELS, decile_cols):
        valid = df[decile_col].notna()
        ax.plot(df.loc[valid, 'meeting_date'], df.loc[valid, decile_col].astype(float),
                marker='o', markersize=3.8, linewidth=0.8, alpha=0.8)
        ax.set_title(f'{title}  (n={valid.sum()})', loc='left', fontsize=11)
        ax.set_ylabel('Decile')
        ax.set_yticks(range(1, 11))
        ax.set_ylim(0.5, 10.5)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel('FOMC meeting date')
    fig.suptitle('FOMC event vector: normalized channel deciles', fontsize=15)
    fig.savefig(DECILE_PLOT, dpi=200, bbox_inches='tight')
    print(f'\nSaved: {DECILE_PLOT}')

    complete = df.dropna(subset=decile_cols).copy()
    complete['divergence'] = (
        complete[decile_cols].max(axis=1) - complete[decile_cols].min(axis=1)
    ).astype(int)
    complete['extreme_divergence'] = (
        (complete[decile_cols].max(axis=1) >= 9)
        & (complete[decile_cols].min(axis=1) <= 3)
    )

    print('\n' + '=' * 60)
    print('FOUR-CHANNEL DIVERGENCE SUMMARY')
    print('=' * 60)
    print(f'Complete FOMC vectors: {len(complete)}')

    if not complete.empty:
        print(f"Mean divergence: {complete['divergence'].mean():.2f}")
        print(f"Median divergence: {complete['divergence'].median():.2f}")
        n_extreme = int(complete['extreme_divergence'].sum())
        print(f'Extreme divergence meetings: {n_extreme}/{len(complete)} ({n_extreme / len(complete):.1%})')
        print('\nDivergence distribution:')
        print(complete['divergence'].value_counts().reindex(range(10), fill_value=0).sort_index().to_string())
        complete['highest_channel'] = complete[decile_cols].astype(float).idxmax(axis=1)
        complete['lowest_channel'] = complete[decile_cols].astype(float).idxmin(axis=1)
        print('\nTop 20 most divergent meetings:')
        top = complete.sort_values(['divergence', 'meeting_date'], ascending=[False, True]).head(20)
        print(top[['meeting_date'] + decile_cols + ['divergence', 'extreme_divergence']].to_string(index=False))
        print('\nExtreme divergence meetings:')
        print(complete.loc[complete['extreme_divergence'],
              ['meeting_date'] + decile_cols + ['divergence', 'highest_channel', 'lowest_channel']]
              .sort_values('meeting_date').to_string(index=False))

    output_cols = (['meeting_date'] + [c for c, _ in CHANNELS] + decile_cols
                   + ['divergence', 'extreme_divergence'])
    if not complete.empty:
        output_cols += ['highest_channel', 'lowest_channel']
    complete[output_cols].to_csv(DIVERGENCE_CSV, index=False)
    print(f'\nSaved: {DIVERGENCE_CSV}')

    fig_div, ax_div = plt.subplots(figsize=(15, 4.5))
    if not complete.empty:
        ax_div.plot(complete['meeting_date'], complete['divergence'], marker='o',
                    markersize=4, linewidth=0.9, alpha=0.8, label='Decile divergence')
        extreme = complete['extreme_divergence']
        ax_div.scatter(complete.loc[extreme, 'meeting_date'],
                       complete.loc[extreme, 'divergence'], color='red', s=45,
                       zorder=3, label='Extreme divergence')
    ax_div.set_title('Cross-channel divergence across FOMC meetings (four channels)', fontsize=14)
    ax_div.set_xlabel('FOMC meeting date')
    ax_div.set_ylabel('Max decile − Min decile')
    ax_div.set_ylim(-0.5, 9.5)
    ax_div.set_yticks(range(10))
    ax_div.grid(alpha=0.25)
    ax_div.legend()
    fig_div.tight_layout()
    fig_div.savefig(DIVERGENCE_PLOT, dpi=200, bbox_inches='tight')
    print(f'Saved: {DIVERGENCE_PLOT}')
    plt.show()


if __name__ == '__main__':
    main()
