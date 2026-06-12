# -*- coding: utf-8 -*-
"""Rebuild the secondary Results tables from results_canonical:

  tab:pressure_performance -- temperature R^2 by pressure level (mean over seeds)
                              for the GNN family plus IDW.
  tab:mae_real             -- mean absolute error in physical units (mean +/- std)
                              for the GNN family plus the LSTM baseline.

Writes comparison_output/table_pressure_temp_canonical.tex and
comparison_output/table_mae_real_canonical.tex, and prints a plain summary.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
import numpy as np
import pandas as pd

RESULTS_DIR = Path("results_canonical")
OUT = Path("comparison_output")
OUT.mkdir(parents=True, exist_ok=True)

LEVELS = [1000, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 10]

PRESS_MODELS = [('vht_gnn', 'VHT-GNN'), ('flat_graphsage', 'GraphSAGE'),
                ('mpnn', 'MPNN'), ('gat', 'GAT'), ('idw', 'IDW')]

MAE_MODELS = [('vht_gnn', 'VHT-GNN'), ('flat_graphsage', 'GraphSAGE'),
              ('mpnn', 'MPNN'), ('multiscale_graphsage', 'Multiscale-GraphSAGE'),
              ('gat', 'GAT'), ('lstm', 'LSTM')]
MAE_VARS = [('temperature', 'Temperature (°C)'),
            ('relative_humidity', 'Rel. Humidity (\\%)'),
            ('wind_speed', 'Wind Speed (m/s)'),
            ('geopotential', 'Geopotential (m)')]


def _seed_paths(model, fname):
    base = RESULTS_DIR / model
    seed_dirs = sorted(base.glob("seed_*"))
    return ([sd / fname for sd in seed_dirs] if seed_dirs
            else [base / fname])


def temp_r2_by_level(model):
    """Series of temperature R^2 indexed by pressure level (mean over seeds)."""
    cols = []
    for p in _seed_paths(model, "metrics_by_pressure.csv"):
        df = pd.read_csv(p)
        df = df[df['feature'] == 'temperature'].copy()
        df['pressure'] = df['pressure'].round().astype(int)
        cols.append(df.set_index('pressure')['r2'])
    return pd.concat(cols, axis=1).reindex(LEVELS).mean(axis=1)


def mae_real(model, feature):
    """(mean, std) of mae_real for a feature; std=0.0 for flat models."""
    vals = [pd.read_csv(p).set_index('feature').loc[feature, 'mae_real']
            for p in _seed_paths(model, "metrics.csv")]
    vals = np.array(vals, dtype=float)
    return float(vals.mean()), (float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)


def build_pressure_table():
    data = {m: temp_r2_by_level(m) for m, _ in PRESS_MODELS}
    disp = [d for _, d in PRESS_MODELS]
    ncol = len(PRESS_MODELS)
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        r'\caption{Temperature $R^2$ by pressure level for the GNN models and the '
        r'classical IDW baseline (mean over five seeds). Best value per level in bold.}',
        r'\label{tab:pressure_performance}', r'\small',
        r'\begin{tabular}{r' + 'c' * ncol + '}', r'\toprule',
        r'\textbf{hPa} & ' + ' & '.join(f'\\textbf{{{d}}}' for d in disp) + r' \\',
        r'\midrule',
    ]
    for p in LEVELS:
        vals = [data[m][p] for m, _ in PRESS_MODELS]
        best = max(vals)
        cells = [f'\\textbf{{{v:.3f}}}' if abs(v - best) < 1e-9 else f'{v:.3f}'
                 for v in vals]
        lines.append(f'{p} & ' + ' & '.join(cells) + r' \\')
    lines.append(r'\midrule')
    avgs = [float(data[m].mean()) for m, _ in PRESS_MODELS]
    best = max(avgs)
    cells = [f'\\textbf{{{v:.3f}}}' if abs(v - best) < 1e-9 else f'{v:.3f}'
             for v in avgs]
    lines.append(r'\textbf{Avg} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines), data


def build_mae_table():
    data = {m: {f: mae_real(m, f) for f, _ in MAE_VARS} for m, _ in MAE_MODELS}
    cap = (r'\caption{Mean absolute error in physical units (mean $\pm$ std over five '
           r'seeds). Lowest error per variable in bold.}')
    tex = _pivot_table(cap, r'\label{tab:mae_real}', PIVOT_VARS_U,
                       lambda m, f: data[m][f], min)
    return tex, data


ABL_VARS = ['temperature', 'relative_humidity', 'wind_speed', 'geopotential']
ABL_ROWS = [
    ('vht_gnn', r'VHT-GNN (full)', False),
    ('vht_gnn_no_vertical', r'\quad w/o vertical edges', False),
    ('vht_gnn_no_horizontal', r'\quad w/o horizontal edges', False),
    ('vht_gnn_no_temporal', r'\quad w/o temporal attention', False),
    ('vht_gnn_no_gating', r'\quad w/o edge gating', False),
    ('vht_gnn_fixed_fusion', r'\quad fixed fusion ($\alpha{=}1/3$)', False),
    ('vht_gnn_global_norm', r'\quad w/o level-aware norm.$^{\dagger}$', True),
]


def r2_means(model):
    vals = [pd.read_csv(p).set_index('feature') for p in _seed_paths(model, "metrics.csv")]
    return {f: float(np.mean([v.loc[f, 'r2'] for v in vals])) for f in ABL_VARS}


def build_ablation_table():
    data = {m: r2_means(m) for m, _, _ in ABL_ROWS}
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        r'\caption{Ablation of VHT-GNN components: mean $R^2$ over five seeds for the '
        r'four primary variables and their mean (relative humidity at $\geq$~200~hPa, as '
        r'in Table~\ref{tab:R2_comparison}). $^{\dagger}$Removing level-aware normalization '
        r'inflates $R^2$---global normalization lets the large inter-level variance '
        r'dominate the $R^2$ denominator---while severely degrading physical accuracy: '
        r'geopotential MAE rises from 29~m to 272~m (see text).}',
        r'\label{tab:ablation}', r'\small',
        r'\begin{tabular}{lccccc}', r'\toprule',
        r'\textbf{Configuration} & \textbf{Temp} & \textbf{RH} & \textbf{WS} & '
        r'\textbf{Geopot} & \textbf{Mean} \\',
        r'\midrule',
    ]
    for i, (m, label, _dag) in enumerate(ABL_ROWS):
        if i == 1:
            lines.append(r'\midrule')
        r = data[m]
        mean4 = np.mean([r[v] for v in ABL_VARS])
        cells = [f'{r[v]:.3f}' for v in ABL_VARS] + [f'{mean4:.3f}']
        lines.append(f'{label} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


HYDRO_ROWS = [
    ('ground_truth', 'Ground-truth observations'),
    ('vht_gnn', 'VHT-GNN'),
    ('flat_graphsage', 'GraphSAGE'),
    ('mpnn', 'MPNN'),
    ('multiscale_graphsage', 'Multiscale-GraphSAGE'),
    ('gat', 'GAT'),
]


def build_hydrostatic_table():
    """Reads the M12 output comparison_output/hydrostatic_residual_summary.csv."""
    df = pd.read_csv(OUT / 'hydrostatic_residual_summary.csv').set_index('model')
    means = {m: df.loc[m, 'mean_residual_m'] for m, _ in HYDRO_ROWS
             if m in df.index and m != 'ground_truth'}
    best = min(means.values())
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        r'\caption{Hydrostatic residual of the reconstructed profiles (mean $\pm$ std '
        r'over five seeds) and its ratio to the ground-truth observational baseline. '
        r'Lower is more physically consistent; the baseline is non-zero because the '
        r'hydrostatic relation assumes dry air. Best (lowest) value in bold.}',
        r'\label{tab:hydrostatic}', r'\small',
        r'\begin{tabular}{lcc}', r'\toprule',
        r'\textbf{Method} & \textbf{Residual (m)} & \textbf{Ratio to baseline} \\',
        r'\midrule',
    ]
    for i, (m, label) in enumerate(HYDRO_ROWS):
        if m not in df.index:
            continue
        mean = df.loc[m, 'mean_residual_m']; std = df.loc[m, 'std_residual_m']
        ratio = df.loc[m, 'baseline_ratio']
        if m == 'ground_truth':
            res = f'{mean:.2f}'
        else:
            num = f'\\textbf{{{mean:.2f}}}' if abs(mean - best) < 1e-9 else f'{mean:.2f}'
            res = f'{num}\\,$\\pm$\\,{std:.2f}'
        lines.append(f'{label} & {res} & {ratio:.2f}' + r' \\')
        if i == 0:
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


MULTI_RATIO_MODELS = [('vht_gnn', 'VHT-GNN'), ('flat_graphsage', 'GraphSAGE')]
MULTI_RATIOS = [0.15, 0.30, 0.40, 0.50, 0.60]


def build_multi_ratio_table():
    """Reads comparison_output/multi_ratio_canonical.csv (M11 output).
    LSTM is intentionally excluded: its M11 re-evaluation does not match the
    canonical Table-1 value, so it is not cross-consistent here."""
    df = pd.read_csv(OUT / "multi_ratio_canonical.csv")
    cell = {}  # (model, ratio) -> (mean, std)
    for _, r in df.iterrows():
        cell[(r['model'], round(r['mask_ratio'], 2))] = (r['mean_r2'], r['std_r2'])
    best = {}
    for ratio in MULTI_RATIOS:
        vals = [cell[(m, ratio)][0] for m, _ in MULTI_RATIO_MODELS if (m, ratio) in cell]
        best[ratio] = max(vals) if vals else None
    head = ' & '.join(f'\\textbf{{{int(r*100)}\\%}}' for r in MULTI_RATIOS)
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        r'\caption{Average $R^2$ across all six variables at increasing '
        r'artificial-missingness ratios (mean $\pm$ std over five seeds). The 15\% '
        r'column is the training mask ratio; higher ratios probe robustness to more '
        r'severe gaps. Best value per ratio in bold.}',
        r'\label{tab:multi_ratio}', r'\small',
        r'\begin{tabular}{l' + 'c' * len(MULTI_RATIOS) + '}', r'\toprule',
        r'\textbf{Model} & ' + head + r' \\', r'\midrule',
    ]
    for m, disp in MULTI_RATIO_MODELS:
        cells = []
        for ratio in MULTI_RATIOS:
            mean, std = cell[(m, ratio)]
            num = (f'\\textbf{{{mean:.3f}}}'
                   if best[ratio] is not None and abs(mean - best[ratio]) < 1e-9
                   else f'{mean:.3f}')
            cells.append(f'{num}\\,$\\pm$\\,{std:.3f}')
        lines.append(f'{disp} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


EXT_MODELS = MAE_MODELS  # same six-model set as the real-unit error table
EXT_VARS = MAE_VARS      # variable labels with physical units (for bias)
PLAIN_VARS = [('temperature', 'Temperature'),
              ('relative_humidity', 'Rel. Humidity'),
              ('wind_speed', 'Wind Speed'),
              ('geopotential', 'Geopotential')]
N_EXTREME = {'temperature': 62, 'relative_humidity': 32,
             'wind_speed': 256, 'geopotential': 28}


def seed_metric(model, feature, col):
    """(mean, std) of any metrics.csv column over seeds; std=0.0 if single run."""
    vals = [pd.read_csv(p).set_index('feature').loc[feature, col]
            for p in _seed_paths(model, "metrics.csv")]
    vals = np.array(vals, dtype=float)
    return float(vals.mean()), (float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)


def seed_profile(model, feature, col):
    """(mean, std) of any profile_consistency.csv column over seeds."""
    vals = [pd.read_csv(p).set_index('feature').loc[feature, col]
            for p in _seed_paths(model, "profile_consistency.csv")]
    vals = np.array(vals, dtype=float)
    return float(vals.mean()), (float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)


# Pivoted layout: models are rows, variables are columns. This keeps wide
# six-model tables within the narrow journal text width without scaling boxes
# (\resizebox conflicts with the sn-jnl threeparttable table environment).
PIVOT_VARS_U = [('temperature', 'Temp (°C)'), ('relative_humidity', 'RH (\\%)'),
                ('wind_speed', 'WS (m/s)'), ('geopotential', 'Geopot (m)')]
PIVOT_VARS_P = [('temperature', 'Temp'), ('relative_humidity', 'RH'),
                ('wind_speed', 'WS'), ('geopotential', 'Geopot')]


def _pivot_table(caption, label, vars_, cell, best_of):
    """Render a models-row by variables-column table of mean +/- std cells.
    cell(model, feature) -> (mean, std); best_of(list of means) -> value to bold."""
    ncol = len(vars_)
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        caption, label, r'\footnotesize',
        r'\begin{tabular}{l' + 'c' * ncol + '}', r'\toprule',
        r'\textbf{Model} & ' + ' & '.join(f'\\textbf{{{h}}}' for _, h in vars_) + r' \\',
        r'\midrule',
    ]
    best = {fk: best_of([cell(m, fk)[0] for m, _ in EXT_MODELS]) for fk, _ in vars_}
    for m, mdisp in EXT_MODELS:
        cells = []
        for fk, _ in vars_:
            mean, std = cell(m, fk)
            num = (f'\\textbf{{{mean:.2f}}}' if abs(mean - best[fk]) < 1e-9
                   else f'{mean:.2f}')
            cells.append(f'{num}\\,$\\pm$\\,{std:.2f}')
        lines.append(f'{mdisp} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def build_bias_table():
    """Signed bias in physical units (bias_real); smallest |bias| per variable bold."""
    cap = (r'\caption{Systematic bias in physical units, $\mu(\hat{y}-y)$ after '
           r'level-aware denormalization (mean over five seeds). The value closest to '
           r'zero per variable is in bold; a negative sign denotes under-prediction.}')
    cell = lambda m, f: seed_metric(m, f, 'bias_real')
    best_of = lambda means: min(means, key=abs)
    return _pivot_table(cap, r'\label{tab:bias}', PIVOT_VARS_U, cell, best_of)


def build_profile_table():
    """Per-profile vertical Pearson correlation (mean_r); highest per variable bold."""
    cap = (r'\caption{Vertical-profile shape fidelity: mean Pearson correlation '
           r'between predicted and observed profiles over all (station, timestamp) '
           r'pairs with at least three available levels (mean over five seeds). '
           r'Higher is better; best per variable in bold.}')
    cell = lambda m, f: seed_profile(m, f, 'mean_r')
    return _pivot_table(cap, r'\label{tab:profile_consistency}', PIVOT_VARS_P, cell, max)


def build_extreme_table():
    """MAE restricted to extreme observations (|z|>3, normalized); smallest bold."""
    n = N_EXTREME
    cap = (r'\caption{Extreme-event reconstruction error: MAE on the tail subset of '
           r'observations whose level-normalized value satisfies $|z|>3$ '
           r'(normalized units, mean over five seeds). The subset is identical for '
           r'all models; its size is $n=%d$ (temperature), $%d$ (relative humidity), '
           r'$%d$ (wind speed), and $%d$ (geopotential). Smallest error per variable '
           r'in bold.}' % (n['temperature'], n['relative_humidity'],
                           n['wind_speed'], n['geopotential']))
    cell = lambda m, f: seed_metric(m, f, 'mae_extreme')
    return _pivot_table(cap, r'\label{tab:extreme}', PIVOT_VARS_P, cell, min)


SIG_FEATURES = ['temperature', 'relative_humidity', 'wind_speed',
                'sin_wd', 'cos_wd', 'geopotential']
SIG_HEAD = ['Temp', 'RH', 'WS', 'Sin WD', 'Cos WD', 'Geopot']
SIG_EXTERNAL = [
    ('flat_graphsage', 'GraphSAGE'),
    ('lstm', 'LSTM'),
    ('saits', 'SAITS'),
    ('mpnn', 'MPNN'),
    ('multiscale_graphsage', 'Multiscale-GraphSAGE'),
    ('gat', 'GAT'),
]
SIG_ABLATION = [
    ('vht_gnn_no_vertical', r'\quad w/o vertical edges'),
    ('vht_gnn_no_horizontal', r'\quad w/o horizontal edges'),
    ('vht_gnn_no_temporal', r'\quad w/o temporal attention'),
    ('vht_gnn_no_gating', r'\quad w/o edge gating'),
    ('vht_gnn_fixed_fusion', r'\quad fixed fusion ($\alpha{=}1/3$)'),
]


def _load_sig(fname):
    """{(model_b, feature): (ttest_p, diff)} for rows with model_a == vht_gnn."""
    df = pd.read_csv(OUT / fname)
    df = df[df['model_a'] == 'vht_gnn']
    return {(r['model_b'], r['feature']): (float(r['ttest_p']), float(r['diff']))
            for _, r in df.iterrows()}


def _sig_cell(p, vht_better):
    s = r'$<$0.001' if p < 0.001 else f'{p:.3f}'
    if p < 0.05:
        s = r'\textbf{' + s + '}'
    if not vht_better:
        s = s + r'$^{\ddagger}$'
    return s


def build_significance_table(fname, label, metric_phrase, better_when_positive):
    """Paired t-test p-value matrix, VHT-GNN vs each method, per variable.
    better_when_positive: True for R^2 (VHT better when diff>0), False for MAE."""
    d = _load_sig(fname)
    ncol = len(SIG_FEATURES)
    cap = (r'\caption{Paired two-sided $t$-test $p$-values comparing VHT-GNN against '
           r'each alternative on the per-seed ' + metric_phrase + r' (five matched '
           r'seeds). Bold marks $p<0.05$; $^{\ddagger}$ marks the cases where the '
           r'alternative attains the better mean, so a significant difference there '
           r'favours the alternative rather than VHT-GNN. The Wilcoxon signed-rank '
           r'test is omitted: with five seeds its two-sided $p$ is floored at 0.0625 '
           r'and cannot reach 0.05, so the $t$-test together with the effect sizes in '
           r'Tables~\ref{tab:R2_comparison} and \ref{tab:mae_real} carries the '
           r'inference.}')
    lines = [
        r'\begin{table}[htbp]', r'\centering', cap, label,
        r'\setlength{\tabcolsep}{4pt}', r'\footnotesize',
        r'\begin{tabular}{l' + 'c' * ncol + '}', r'\toprule',
        r'\textbf{Comparison (vs.\ VHT-GNN)} & '
        + ' & '.join(f'\\textbf{{{h}}}' for h in SIG_HEAD) + r' \\',
        r'\midrule',
    ]
    groups = [('External methods', SIG_EXTERNAL),
              ('Ablated VHT-GNN variants', SIG_ABLATION)]
    for gi, (gname, members) in enumerate(groups):
        if gi > 0:
            lines.append(r'\midrule')
        lines.append(r'\multicolumn{%d}{l}{\textit{%s}} \\' % (ncol + 1, gname))
        for model, disp in members:
            cells = []
            for f in SIG_FEATURES:
                if (model, f) not in d:
                    cells.append('--')
                    continue
                p, diff = d[(model, f)]
                vht_better = (diff > 0) if better_when_positive else (diff < 0)
                cells.append(_sig_cell(p, vht_better))
            lines.append(f'{disp} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def main():
    ptex, pdata = build_pressure_table()
    mtex, mdata = build_mae_table()
    atex = build_ablation_table()
    htex = build_hydrostatic_table()
    rtex = build_multi_ratio_table()
    btex = build_bias_table()
    pctex = build_profile_table()
    etex = build_extreme_table()
    sr2tex = build_significance_table(
        'significance_matrix_r2.csv', r'\label{tab:significance_r2}',
        r'$R^2$', better_when_positive=True)
    smaetex = build_significance_table(
        'significance_matrix_mae_real.csv', r'\label{tab:significance_mae}',
        r'real-unit MAE', better_when_positive=False)
    (OUT / 'table_pressure_temp_canonical.tex').write_text(ptex, encoding='utf-8')
    (OUT / 'table_mae_real_canonical.tex').write_text(mtex, encoding='utf-8')
    (OUT / 'table_ablation_canonical.tex').write_text(atex, encoding='utf-8')
    (OUT / 'table_hydrostatic_canonical.tex').write_text(htex, encoding='utf-8')
    (OUT / 'table_multi_ratio_canonical.tex').write_text(rtex, encoding='utf-8')
    (OUT / 'table_bias_canonical.tex').write_text(btex, encoding='utf-8')
    (OUT / 'table_profile_consistency_canonical.tex').write_text(pctex, encoding='utf-8')
    (OUT / 'table_extreme_canonical.tex').write_text(etex, encoding='utf-8')
    (OUT / 'table_significance_r2_canonical.tex').write_text(sr2tex, encoding='utf-8')
    (OUT / 'table_significance_mae_canonical.tex').write_text(smaetex, encoding='utf-8')
    print('Saved 10 tables: pressure, mae_real, ablation, hydrostatic, multi_ratio, '
          'bias, profile_consistency, extreme, significance_r2, significance_mae')

    print('\n[A] Temperature R2 by level (avg row):')
    for m, d in PRESS_MODELS:
        print(f'  {d:<12} avg={float(pdata[m].mean()):.3f}  '
              f'[1000={pdata[m][1000]:.3f}, 200={pdata[m][200]:.3f}, 10={pdata[m][10]:.3f}]')
    print('\n[B] mae_real (mean) per variable:')
    hdr = ''.join(f'{d[:12]:>14}' for _, d in MAE_MODELS)
    print(f'{"var":<12}{hdr}')
    for f, fl in MAE_VARS:
        row = ''.join(f'{mdata[m][f][0]:>14.2f}' for m, _ in MAE_MODELS)
        print(f'{f[:12]:<12}{row}')


if __name__ == "__main__":
    main()
