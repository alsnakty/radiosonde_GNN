# -*- coding: utf-8 -*-
"""Vertical profiles of imputation error (mae_real) by pressure level.

Two panels -- relative humidity and wind speed -- comparing VHT-GNN, LSTM and
IDW on the test set. Multi-seed models are drawn as the mean over 5 seeds with a
+/- standard-deviation band; IDW is deterministic (no band). Relative humidity is
restricted to pressure >= 200 hPa, matching the stratospheric-RH exclusion applied
throughout the evaluation (the aggregate RH metric drops pressure < 200 hPa).

Reads results_canonical/<model>/[seed_<n>/]metrics_by_pressure.csv.
Writes comparison_output/figures/vertical_profile_mae.{png,pdf}.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

RESULTS_DIR = Path("results_canonical")
OUT_DIR = Path("comparison_output/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRIC = 'mae_real'

# Standard pressure levels, surface -> top of atmosphere.
ALL_LEVELS = [1000, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 10]
# Stratospheric RH (pressure < 200 hPa) is excluded from RH metrics study-wide.
RH_LEVELS = [p for p in ALL_LEVELS if p >= 200]

# (model directory, legend label, colour, linestyle)
MODELS = [
    ('vht_gnn', 'VHT-GNN (ours)', '#c0392b', '-'),
    ('lstm',    'LSTM',           '#2c7fb8', '-'),
    ('idw',     'IDW',            '#636363', '--'),
]
# (feature key, panel title, x-axis label, pressure levels to draw)
PANELS = [
    ('relative_humidity', 'Relative humidity', 'MAE (%)',          RH_LEVELS),
    ('wind_speed',        'Wind speed',        'MAE (m s$^{-1}$)',  ALL_LEVELS),
]


def profile_for(model, feature, levels):
    """Return (mean, std) of METRIC aligned to `levels`. Multi-seed models
    (seed_* dirs) -> mean/std over seeds; flat models -> mean only (std=None)."""
    mdir = RESULTS_DIR / model
    seed_dirs = sorted(mdir.glob("seed_*"))
    paths = ([sd / "metrics_by_pressure.csv" for sd in seed_dirs]
             if seed_dirs else [mdir / "metrics_by_pressure.csv"])
    cols = []
    for p in paths:
        df = pd.read_csv(p)
        df = df[df['feature'] == feature].copy()
        df['pressure'] = df['pressure'].round().astype(int)
        cols.append(df.set_index('pressure')[METRIC])
    mat = pd.concat(cols, axis=1).reindex(levels)
    mean = mat.mean(axis=1).values
    std = mat.std(axis=1, ddof=1).values if mat.shape[1] > 1 else None
    return mean, std


def main():
    fig, axes = plt.subplots(1, len(PANELS), figsize=(10.5, 6))
    for ax, (feat, title, xlabel, levels) in zip(axes, PANELS):
        y = np.array(levels, dtype=float)
        for model, label, colour, ls in MODELS:
            mean, std = profile_for(model, feat, levels)
            ax.plot(mean, y, color=colour, linestyle=ls, marker='o', ms=4,
                    lw=1.9, label=label, zorder=3)
            if std is not None:
                ax.fill_betweenx(y, mean - std, mean + std, color=colour,
                                 alpha=0.18, linewidth=0, zorder=2)
        ax.set_yscale('log')
        ax.set_ylim(max(levels), min(levels))   # surface at bottom, TOA at top
        ax.set_yticks(levels)
        ax.set_yticklabels([str(p) for p in levels])
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlim(left=0)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(True, ls=':', alpha=0.45, zorder=0)

    axes[0].set_ylabel('Pressure (hPa)')
    axes[0].legend(loc='best', frameon=True, fontsize=9)
    fig.tight_layout()
    # 600 dpi: meets journals that require >= 600 dpi raster figures.
    for ext in ('png', 'pdf'):
        out = OUT_DIR / f"vertical_profile_mae.{ext}"
        fig.savefig(out, dpi=600, bbox_inches='tight')
        print("Saved:", out)


if __name__ == "__main__":
    main()
