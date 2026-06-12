# -*- coding: utf-8 -*-
"""Build the canonical Table 1 (R^2 by variable) from results_canonical.

Reads results_canonical/<model>/[seed_<n>/]metrics.csv, computes the 5-seed
mean +/- std of R^2 per variable for the reframed model set (flat_graphsage is
the GraphSAGE baseline; the multi-relational variant is an ablation, not a Table-1
baseline; SAITS is included). Linear (Spatial) is flagged: it is evaluated only on
the ~2590 convex-hull-interior points, so it is never bolded as best.

Writes comparison_output/table1_r2_canonical.tex and prints a plain summary.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
import numpy as np
import pandas as pd

RESULTS_DIR = Path("results_canonical")
OUT = Path("comparison_output")
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = ['temperature', 'relative_humidity', 'wind_speed',
            'sin_wd', 'cos_wd', 'geopotential']
FEAT_HEAD = ['Temp', 'RH', 'WS', 'Sin WD', 'Cos WD', 'Geopot']

# Reframed Table-1 model set, grouped. (model_dir, display name)
GROUPS = [
    ('GNN Models', [
        ('vht_gnn', 'VHT-GNN'),
        ('flat_graphsage', 'GraphSAGE'),
        ('mpnn', 'MPNN'),
        ('multiscale_graphsage', 'Multiscale-GraphSAGE'),
        ('gat', 'GAT'),
    ]),
    ('Deep Learning and Transformer Baselines', [
        ('lstm', 'LSTM'),
        ('saits', 'SAITS'),
        ('cnn', 'CNN'),
        ('mlp', 'MLP'),
    ]),
    ('Statistical Baselines', [
        ('linear_spatial', r'Linear (Spatial)$^{\dagger}$'),
        ('linear_combined', 'Linear (Combined)'),
        ('idw', 'IDW'),
        ('linear_temporal', 'Linear (Temporal)'),
        ('linear_vertical', 'Linear (Vertical)'),
    ]),
]
# Not directly comparable (restricted evaluation domain) -> excluded from bolding.
NO_BOLD = {'linear_spatial'}


def r2_mean_std(model):
    """Return {feature: (mean, std_or_None)} of R^2 for a model."""
    mdir = RESULTS_DIR / model
    seed_dirs = sorted(mdir.glob("seed_*"))
    paths = ([sd / "metrics.csv" for sd in seed_dirs]
             if seed_dirs else [mdir / "metrics.csv"])
    cols = []
    for p in paths:
        s = pd.read_csv(p).set_index('feature')['r2']
        cols.append(s)
    mat = pd.concat(cols, axis=1)
    out = {}
    for f in FEATURES:
        if f in mat.index:
            vals = mat.loc[f].values.astype(float)
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
            out[f] = (mean, std)
        else:
            out[f] = (None, None)
    return out


def _num(x):
    # LaTeX-friendly: render a negative sign as a math minus.
    return f'$-${abs(x):.3f}' if x < 0 else f'{x:.3f}'


def fmt(mean, std, bold):
    if mean is None:
        return '--'
    m = f'\\textbf{{{_num(mean)}}}' if bold else _num(mean)
    if std is None:
        return m
    return f'{m}\\,$\\pm$\\,{std:.3f}'


def main():
    # Collect data for every listed model.
    data = {}
    for _, members in GROUPS:
        for model, _disp in members:
            data[model] = r2_mean_std(model)

    # Per-feature best mean among bold-eligible models.
    best = {}
    for f in FEATURES:
        cand = [(data[m][f][0], m) for _, members in GROUPS for m, _ in members
                if m not in NO_BOLD and data[m][f][0] is not None]
        best[f] = max(v for v, _ in cand) if cand else None

    ncol = len(FEATURES)
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{$R^2$ scores by variable (mean $\pm$ std over five seeds; '
        r'statistical baselines are deterministic). Best value per variable in '
        r'bold. $^{\dagger}$Linear (Spatial) is evaluated only on the '
        r'$\sim$2590 convex-hull-interior points (vs. $\sim$19{,}568 for the '
        r'other methods) and is therefore not directly comparable.}',
        r'\label{tab:R2_comparison}',
        r'\setlength{\tabcolsep}{4pt}',
        r'\footnotesize',
        r'\begin{tabular}{l' + 'c' * ncol + '}',
        r'\toprule',
        r'\textbf{Model} & ' + ' & '.join(f'\\textbf{{{h}}}' for h in FEAT_HEAD) + r' \\',
        r'\midrule',
    ]
    for gi, (gname, members) in enumerate(GROUPS):
        if gi > 0:
            lines.append(r'\midrule')
        lines.append(r'\multicolumn{%d}{l}{\textit{%s}} \\' % (ncol + 1, gname))
        for model, disp in members:
            cells = []
            for f in FEATURES:
                mean, std = data[model][f]
                is_best = (model not in NO_BOLD and best[f] is not None
                           and mean is not None and abs(mean - best[f]) < 1e-9)
                cells.append(fmt(mean, std, is_best))
            lines.append(f'{disp} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    tex = '\n'.join(lines)

    out_path = OUT / 'table1_r2_canonical.tex'
    out_path.write_text(tex, encoding='utf-8')
    print('Saved:', out_path)
    print()
    # Plain summary (ascii-safe)
    print(f"{'Model':<24}" + ''.join(f'{h:>9}' for h in FEAT_HEAD))
    for _, members in GROUPS:
        for model, disp in members:
            row = ''
            for f in FEATURES:
                mean, std = data[model][f]
                row += f'{mean:>9.3f}' if mean is not None else f'{"--":>9}'
            star = ' (dagger: n=2590)' if model in NO_BOLD else ''
            print(f"{disp[:24]:<24}{row}{star}")


if __name__ == "__main__":
    main()
