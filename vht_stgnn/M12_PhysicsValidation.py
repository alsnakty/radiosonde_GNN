# -*- coding: utf-8 -*-
"""
Hydrostatic residual evaluation for trained GNN models.

For each adjacent pressure-level pair (p1 > p2) in an imputed profile,
the hydrostatic equation gives:
    Delta_Z_hydrostatic = (R * T_mean_K / g) * ln(p1 / p2)
The residual measures how far the model's predicted Delta_Z departs
from this physical expectation:
    residual = |Delta_Z_predicted - Delta_Z_hydrostatic|   [meters]

Constants: R = 287.05 J/(kg.K), g = 9.80665 m/s^2.

For each GNN model with a saved checkpoint, this script:
  1. Loads the model
  2. Runs inference on the test graph (denormalizing per-level)
  3. Computes the per-pair residual
  4. Writes results/<model>/hydrostatic_residual.csv
The same metric is computed on the ground-truth test observations as
a baseline reference (residual is non-zero on real profiles because
the formula above neglects virtual temperature corrections).
"""

import os
import json

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from M00_Config import cfg
from M02_DataLoading import (
    RadiosondeLoader,
    RadiosondeGraphBuilder,
    RadiosondeSlidingWindowDataset,
    collate_graph_windows,
)
from M03_Model import RadiosondeSpatioTemporalGNN


# Constants
R_GAS = 287.05    # gas constant for dry air, J/(kg.K)
G_ACC = 9.80665   # gravitational acceleration, m/s^2

# Feature indices (M02 feature_columns ordering)
IDX_TEMPERATURE  = 0
IDX_GEOPOTENTIAL = 5

# Pressure-level adjacency: (p1 lower altitude > p2 higher altitude)
PRESSURE_LEVELS = list(cfg.pressure_levels)  # [1000, 850, ..., 10]
ADJACENT_PAIRS = list(zip(PRESSURE_LEVELS[:-1], PRESSURE_LEVELS[1:]))

# Level-aware GNN models with multi-seed checkpoints (results_canonical/<model>/seed_<n>/).
# global_norm variants are excluded: they were trained with global (not level-aware)
# normalization, so denormalize_level_aware would not invert their outputs correctly.
# DL/SAITS/statistical models have no RadiosondeSpatioTemporalGNN checkpoint.
GNN_MODELS = [
    'vht_gnn', 'vanilla_graphsage', 'flat_graphsage', 'multiscale_graphsage',
    'gat', 'mpnn',
    'vht_gnn_no_temporal', 'vht_gnn_no_gating', 'vht_gnn_fixed_fusion',
    'vht_gnn_no_vertical', 'vht_gnn_no_horizontal',
]

# stations.json has 8 active stations; cfg.DataQuality min default is 9, but the
# skeleton emits 8 unique stations per timestamp, so 9 would reject all.
OVERRIDE_MIN_STATIONS = 8

# All GNN sub-configs share dataset_window_size=3 (M00); inference window must
# match training. Read the shared value from VHT_GNN.
WINDOW_SIZE = cfg.VHT_GNN.dataset_window_size

RESULTS_DIR = Path("results_canonical")
COMPARISON_DIR = Path("comparison_output")
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)


# Data loading (mirrors M99 pipeline + 70/10/20 split)
def load_and_filter_observations():
    loader = RadiosondeLoader(
        stations_json=cfg.stations_path,
        filter_active=cfg.filter_active_stations,
        exclude_stations=cfg.exclude_stations,
    )
    observations, stations_meta = loader.load_dataset_from_csv(
        csv_path=cfg.data_path,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
    )

    feature_cols = ['temperature', 'relative_humidity', 'wind_speed',
                    'wind_direction', 'geopotential']
    feature_cols = [c for c in feature_cols if c in observations.columns]
    min_quality = cfg.DataQuality.min_time_step_quality

    remove_times = set()
    for dt, group in observations.groupby('datetime'):
        group_nan = group[feature_cols].isna().sum().sum()
        group_total = len(group) * len(feature_cols)
        quality = 1 - (group_nan / group_total) if group_total > 0 else 0
        n_stations = group['station_id'].nunique()
        if quality < min_quality:
            remove_times.add(dt)
        elif n_stations < OVERRIDE_MIN_STATIONS:
            remove_times.add(dt)
    observations = observations[~observations['datetime'].isin(remove_times)].copy()
    return observations, stations_meta


def split_and_build_graphs(observations, stations_meta):
    import pandas as pd
    gap = pd.Timedelta(days=cfg.chronological_gap_days)
    dates = observations['datetime'].unique()
    train_end_idx = int(len(dates) * 0.70)
    val_end_idx   = int(len(dates) * 0.80)
    train_split_date = dates[train_end_idx]
    val_split_date   = dates[val_end_idx]
    test_start_date  = val_split_date + gap

    train_obs = observations[observations['datetime'] <  train_split_date]
    test_obs  = observations[observations['datetime'] >= test_start_date]

    builder = RadiosondeGraphBuilder(
        station_metadata=stations_meta,
        temporal_window=cfg.graph_temporal_window,
        include_surface=False,
    )
    train_graph = builder.build_graph_from_observations(train_obs)
    train_stats = train_graph['scaling_stats']
    test_graph = builder.build_graph_from_observations(
        test_obs,
        external_stats=train_stats,
    )
    return train_graph, test_graph


# Model loading
def load_gnn_model(seed_dir, model_type, device):
    """Load one checkpoint from a seed_<n> directory. Returns (model, params)
    or (None, None) if the checkpoint or config is missing."""
    cfg_path = seed_dir / "config.json"
    pt_path  = seed_dir / "model.pt"

    if not pt_path.exists() or not cfg_path.exists():
        print(f"  [skip] {model_type}/{seed_dir.name}: model.pt or config.json missing")
        return None, None

    with open(cfg_path, 'r') as f:
        saved_cfg = json.load(f)
    params = saved_cfg.get('parameters', {})

    kwargs = {
        'input_dim':       cfg.input_dim,
        'hidden_dim':      params.get('hidden_dim', 64),
        'num_gnn_layers':  params.get('num_gnn_layers', 1),
        'dropout':         params.get('dropout', 0.1),
        'model_type':      model_type,
    }
    if model_type == 'gat':
        # heads not stored in config.json; fall back to cfg
        kwargs['heads'] = params.get('heads', cfg.Gat.heads)

    model = RadiosondeSpatioTemporalGNN(**kwargs)
    state_dict = torch.load(pt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, params


# Inference + denormalization
def run_inference(model, test_graph, window_size, batch_size, device, desc='inference'):
    """Runs sliding-window inference and returns aligned arrays for
    predictions, targets, and per-node metadata."""
    test_ds = RadiosondeSlidingWindowDataset(
        test_graph, window_size, cfg.use_realistic_masking,
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_graph_windows,
    )

    all_preds, all_targets = [], []
    all_pressures, all_stations, all_datetimes = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_dl, desc=desc):
            xw     = batch['x'].to(device)
            target = batch['target']
            pred, _ = model(
                xw, batch['pos_info'], batch['edge_indices'],
                batch['edge_attrs'], batch['node_metadata'],
            )
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.numpy())

            md = batch['node_metadata']
            all_pressures.extend(list(md['pressure']))
            all_stations.extend(list(md['station_id']))
            all_datetimes.extend(list(md['datetime']))

    pred_arr   = np.concatenate(all_preds,   axis=0)
    target_arr = np.concatenate(all_targets, axis=0)
    return pred_arr, target_arr, np.array(all_pressures), np.array(all_stations), np.array(all_datetimes)


def denormalize_level_aware(values, pressures, level_stats):
    """Level-aware inverse transform: x_real = x_norm * stds + means
    per the matching p_<level> entry in level_stats. NaNs propagate."""
    out = np.full_like(values, np.nan, dtype=np.float64)
    for p_level in PRESSURE_LEVELS:
        level_key = f"p_{int(p_level)}"
        if level_key not in level_stats:
            continue
        means = np.asarray(level_stats[level_key]['means'], dtype=np.float64)
        stds  = np.asarray(level_stats[level_key]['stds'],  dtype=np.float64)
        mask = np.abs(pressures - p_level) < 1.0
        if mask.sum() == 0:
            continue
        out[mask] = values[mask].astype(np.float64) * stds + means
    return out


# Residual computation
def compute_pair_residuals(values_real, pressures, stations, datetimes):
    """For each (station, time) trajectory, compute the absolute
    hydrostatic residual at every adjacent pressure-level pair.

    Overlapping sliding windows produce duplicate (station, time, p)
    rows; we average them first."""
    df = pd.DataFrame({
        'station':  stations,
        'time':     datetimes,
        'pressure': pressures.round().astype(int),
        'T_C':      values_real[:, IDX_TEMPERATURE],
        'Z_m':      values_real[:, IDX_GEOPOTENTIAL],
    })
    df = df.groupby(['station', 'time', 'pressure'], as_index=False).mean()

    rows = []
    for (station, time), grp in df.groupby(['station', 'time']):
        p_to_vals = {int(row['pressure']): (row['T_C'], row['Z_m'])
                     for _, row in grp.iterrows()}
        for (p1, p2) in ADJACENT_PAIRS:
            if p1 not in p_to_vals or p2 not in p_to_vals:
                continue
            T1_C, Z1 = p_to_vals[p1]
            T2_C, Z2 = p_to_vals[p2]
            if not np.isfinite([T1_C, T2_C, Z1, Z2]).all():
                continue
            T_mean_K = ((T1_C + T2_C) / 2.0) + 273.15
            dZ_pred  = Z2 - Z1
            dZ_hydro = (R_GAS * T_mean_K / G_ACC) * np.log(p1 / p2)
            rows.append({
                'station':       station,
                'time':          time,
                'pair':          f"{p1}-{p2}",
                'p1':            p1,
                'p2':            p2,
                'T_mean_K':      T_mean_K,
                'dZ_predicted':  dZ_pred,
                'dZ_hydrostatic': dZ_hydro,
                'residual_m':    abs(dZ_pred - dZ_hydro),
            })
    return pd.DataFrame(rows)


def summarize_residuals(residual_df, label):
    """Per-pair and overall summary."""
    if residual_df.empty:
        return pd.DataFrame()
    per_pair = (residual_df.groupby('pair', as_index=False)
                .agg(n=('residual_m', 'size'),
                     mean_m=('residual_m', 'mean'),
                     median_m=('residual_m', 'median'),
                     p95_m=('residual_m', lambda x: np.percentile(x, 95)),
                     max_m=('residual_m', 'max')))
    per_pair['label'] = label
    return per_pair


# Main
def main():
    device = cfg.device
    print("=" * 70)
    print(" HYDROSTATIC RESIDUAL EVALUATION")
    print("=" * 70)
    print(f"Device: {device}")

    print("\n[1] Loading data...")
    observations, stations_meta = load_and_filter_observations()
    print(f"  Filtered timestamps: {observations['datetime'].nunique():,}")

    print("\n[2] Building train/test graphs (70/10/20)...")
    train_graph, test_graph = split_and_build_graphs(observations, stations_meta)
    test_scaling = test_graph['scaling_stats']
    level_stats  = test_scaling.get('level_stats', {})

    # Ground truth residual (baseline)
    print("\n[3] Computing ground-truth residual (baseline)...")
    target_norm = test_graph['x'].numpy()
    md = test_graph['node_metadata']
    gt_pressures  = np.array(md['pressure'])
    gt_stations   = np.array(md['station_id'])
    gt_datetimes  = np.array(md['datetime'])
    target_real = denormalize_level_aware(target_norm, gt_pressures, level_stats)
    gt_residuals = compute_pair_residuals(target_real, gt_pressures, gt_stations, gt_datetimes)
    gt_summary = summarize_residuals(gt_residuals, label='ground_truth')
    print(f"  Ground-truth residual pairs: {len(gt_residuals):,}")
    if not gt_summary.empty:
        print(f"  Ground-truth mean residual (all pairs): "
              f"{gt_residuals['residual_m'].mean():.2f} m")

    gt_mean = gt_residuals['residual_m'].mean() if not gt_residuals.empty else float('nan')

    # Per-model residual, aggregated over seeds
    overall_rows = []      # one row per model: seed mean +/- std of the overall residual
    perseed_rows = []      # one row per (model, seed): enables paired significance tests
    is_cuda = isinstance(device, str) and device.startswith('cuda')

    for model_name in GNN_MODELS:
        print(f"\n[4] Evaluating {model_name} ...")
        model_dir = RESULTS_DIR / model_name
        seed_dirs = sorted(model_dir.glob("seed_*"))
        if not seed_dirs:
            print(f"  [skip] {model_name}: no seed_* directories")
            continue

        seed_means = []      # overall mean residual per seed
        seed_perpair = []    # per-pair summary per seed
        for sd in seed_dirs:
            model, params = load_gnn_model(sd, model_name, device)
            if model is None:
                continue
            batch_size = params.get('batch_size', 16)

            pred_arr, target_arr, pressures, stations, datetimes = run_inference(
                model, test_graph, WINDOW_SIZE, batch_size, device,
                desc=f'{model_name}/{sd.name}',
            )
            del model
            if is_cuda:
                torch.cuda.empty_cache()

            pred_real = denormalize_level_aware(pred_arr, pressures, level_stats)
            residuals = compute_pair_residuals(pred_real, pressures, stations, datetimes)
            if residuals.empty:
                print(f"  [warn] {model_name}/{sd.name}: no residual pairs")
                continue
            seed_mean = residuals['residual_m'].mean()
            seed_means.append(seed_mean)
            seed_perpair.append(summarize_residuals(residuals, label=model_name))
            perseed_rows.append({'model': model_name, 'seed': sd.name,
                                 'residual_m': seed_mean})

        if not seed_means:
            print(f"  [skip] {model_name}: no usable seeds")
            continue

        arr = np.array(seed_means, dtype=np.float64)
        mean_m = arr.mean()
        std_m  = arr.std(ddof=1) if len(arr) > 1 else 0.0
        overall_rows.append({
            'model':           model_name,
            'n_seeds':         len(arr),
            'mean_residual_m': mean_m,
            'std_residual_m':  std_m,
            'baseline_ratio':  mean_m / gt_mean if gt_mean and np.isfinite(gt_mean) else float('nan'),
        })

        # Per-model CSV: per-pair residual averaged across seeds
        pm = pd.concat(seed_perpair, ignore_index=True)
        pm_agg = pm.groupby('pair', as_index=False).agg(
            n=('n', 'mean'), mean_m=('mean_m', 'mean'), median_m=('median_m', 'mean'),
            p95_m=('p95_m', 'mean'), max_m=('max_m', 'mean'))
        pm_agg['label'] = model_name
        pm_agg.to_csv(model_dir / "hydrostatic_residual.csv", index=False)

        ratio = mean_m / gt_mean if gt_mean and np.isfinite(gt_mean) else float('nan')
        print(f"  {model_name}: {mean_m:.2f} +/- {std_m:.2f} m  "
              f"({ratio:.2f}x ground-truth baseline), {len(arr)} seeds")

    # Combined summary (per-model mean +/- std + ground-truth baseline)
    gt_row = {
        'model': 'ground_truth', 'n_seeds': 0,
        'mean_residual_m': gt_mean, 'std_residual_m': 0.0, 'baseline_ratio': 1.0,
    }
    overall_df = pd.DataFrame([gt_row] + sorted(overall_rows, key=lambda r: r['mean_residual_m']))
    out_path = COMPARISON_DIR / "hydrostatic_residual_summary.csv"
    overall_df.to_csv(out_path, index=False)
    print(f"\n[5] Combined summary saved: {out_path}")

    # Per-seed residuals for paired significance testing (one row per model x seed)
    if perseed_rows:
        perseed_path = COMPARISON_DIR / "hydrostatic_residual_perseed.csv"
        pd.DataFrame(perseed_rows).to_csv(perseed_path, index=False)
        print(f"    Per-seed residuals saved: {perseed_path}")

    print("\n--- Mean hydrostatic residual (m), mean +/- std over seeds ---")
    for _, r in overall_df.iterrows():
        tag = ' (baseline)' if r['model'] == 'ground_truth' else ''
        print(f"  {r['model']:<24} {r['mean_residual_m']:6.2f} +/- {r['std_residual_m']:.2f} m{tag}")

    print("\n" + "=" * 70)
    print(" DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
