# -*- coding: utf-8 -*-
"""
Computational complexity profiling.

For every model with a saved checkpoint in results/, this script collects:
  - total parameter count (from config.json)
  - training time per run (from config.json)
  - inference time as previously logged (from config.json)
  - measured per-sample inference latency (fresh inference run)
  - peak inference memory in MB (torch.cuda.max_memory_allocated; CUDA only)

GNN models additionally get a theoretical scaling row that quantifies
edge counts as a function of station count N, pressure-level count L,
and time-step count T.

Output: comparison_output/complexity_extended.csv (does not overwrite
the simpler complexity.csv produced by M98_CompareResults).
"""

import os
import json
import time
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import torch
from torch.utils.data import DataLoader

from M00_Config import cfg
from M02_DataLoading import (
    RadiosondeLoader,
    RadiosondeGraphBuilder,
    RadiosondeSlidingWindowDataset,
    collate_graph_windows,
)
from M03_Model import RadiosondeSpatioTemporalGNN


RESULTS_DIR    = Path(cfg.results_dir)
COMPARISON_DIR = Path(cfg.data_dir) / "comparison_output"
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

GNN_MODELS = [
    'vht_gnn', 'vanilla_graphsage', 'multiscale_graphsage', 'gat', 'mpnn',
    # ablation variants (only if their results dir exists)
    'vht_gnn_no_temporal', 'vht_gnn_no_gating', 'vht_gnn_fixed_fusion',
    'vht_gnn_global_norm', 'vanilla_graphsage_global_norm',
]
DL_MODELS  = ['lstm', 'cnn', 'mlp']
ALL_MODELS_FOR_STATIC = GNN_MODELS + DL_MODELS + ['saits']

OVERRIDE_MIN_STATIONS = 8   # see verify_split_seasonality.py


# Static info from saved config.json (no inference required)
def _find_config_paths(model_name):
    """Return list of (config_path, label) tuples. Supports both legacy
    layout (results/<model>/config.json) and seeded layout
    (results/<model>/seed_<n>/config.json)."""
    model_dir = RESULTS_DIR / model_name
    if not model_dir.exists():
        return []
    paths = []
    legacy = model_dir / "config.json"
    if legacy.exists():
        paths.append((legacy, ''))
    for seed_dir in sorted(model_dir.glob('seed_*')):
        cfg_path = seed_dir / "config.json"
        if cfg_path.exists():
            paths.append((cfg_path, seed_dir.name))
    return paths


def collect_static_info():
    rows = []
    for model_name in ALL_MODELS_FOR_STATIC:
        for cfg_path, seed_label in _find_config_paths(model_name):
            try:
                with open(cfg_path, 'r') as f:
                    cdata = json.load(f)
            except Exception as e:
                print(f"  [skip] {cfg_path}: {e}")
                continue
            complexity = cdata.get('complexity', {})
            params     = cdata.get('parameters', {})
            rows.append({
                'model':         model_name,
                'seed':          seed_label or '-',
                'total_params':  complexity.get('total_parameters'),
                'train_time_s':  complexity.get('training_time_seconds'),
                'infer_time_s':  complexity.get('inference_time_seconds'),
                'hidden_dim':    params.get('hidden_dim'),
                'num_gnn_layers': params.get('num_gnn_layers'),
                'batch_size':    params.get('batch_size'),
                'num_epochs':    params.get('num_epochs'),
            })
    return pd.DataFrame(rows)


# Fresh inference profiling (GNN models only)
def _load_filtered_observations():
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


def _build_train_and_test_graphs():
    import pandas as pd
    observations, stations_meta = _load_filtered_observations()
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
    test_graph  = builder.build_graph_from_observations(
        test_obs, external_stats=train_graph['scaling_stats'],
    )
    return train_graph, test_graph


def _load_gnn_model(model_name, device):
    """Use the first available checkpoint (legacy or seed_*) to load."""
    paths = _find_config_paths(model_name)
    if not paths:
        return None, None
    cfg_path, seed_label = paths[0]
    model_dir = cfg_path.parent
    pt_path = model_dir / "model.pt"
    if not pt_path.exists():
        return None, None
    with open(cfg_path, 'r') as f:
        saved = json.load(f)
    params = saved.get('parameters', {})

    # The ablation variants use specific model_type strings; map model_name
    # to model_type. For most models the name is the model_type directly.
    model_type = {
        'vht_gnn_global_norm':            'vht_gnn',
        'vanilla_graphsage_global_norm':  'vanilla_graphsage',
    }.get(model_name, model_name)

    kwargs = {
        'input_dim':       cfg.input_dim,
        'hidden_dim':      params.get('hidden_dim', 64),
        'num_gnn_layers':  params.get('num_gnn_layers', 1),
        'dropout':         params.get('dropout', 0.1),
        'model_type':      model_type,
    }
    if model_name == 'gat':
        kwargs['heads'] = params.get('heads', cfg.Gat.heads)

    try:
        model = RadiosondeSpatioTemporalGNN(**kwargs)
        state_dict = torch.load(pt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"  [skip] {model_name}: failed to load — {e}")
        return None, None
    model.to(device)
    model.eval()
    return model, params


def profile_gnn_inference(test_graph, device, n_warmup=2, n_measure=10):
    """For each GNN model with a checkpoint, run inference on a single
    batch and measure: per-sample latency and peak VRAM (CUDA only).
    Returns a DataFrame."""
    rows = []
    for model_name in GNN_MODELS:
        model, params = _load_gnn_model(model_name, device)
        if model is None:
            continue

        # Window size: use saved seq_length proxy via cfg sub-config
        sub_cfg = {
            'vht_gnn':                        cfg.VHT_GNN,
            'vanilla_graphsage':              cfg.VanillaGraphSAGE,
            'multiscale_graphsage':           cfg.MultiscaleGraphSAGE,
            'gat':                            cfg.Gat,
            'mpnn':                           cfg.Mpnn,
            'vht_gnn_no_temporal':            cfg.VHT_GNN_NoTemporal,
            'vht_gnn_no_gating':              cfg.VHT_GNN_NoGating,
            'vht_gnn_fixed_fusion':           cfg.VHT_GNN_FixedFusion,
            'vht_gnn_global_norm':            cfg.VHT_GNN_GlobalNorm,
            'vanilla_graphsage_global_norm':  cfg.VanillaGraphSAGE_GlobalNorm,
        }.get(model_name)
        if sub_cfg is None:
            continue
        window_size = sub_cfg.dataset_window_size
        batch_size  = sub_cfg.batch_size

        test_ds = RadiosondeSlidingWindowDataset(
            test_graph, window_size, cfg.use_realistic_masking,
        )
        test_dl = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_graph_windows,
        )
        batch = next(iter(test_dl))
        if batch is None or 'x' not in batch:
            continue

        # Move once
        xw = batch['x'].to(device)
        target = batch['target'].to(device)

        # Reset VRAM stat
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()

        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(xw, batch['pos_info'], batch['edge_indices'],
                          batch['edge_attrs'], batch['node_metadata'])
        if device.startswith('cuda'):
            torch.cuda.synchronize()

        # Measure
        start = time.time()
        with torch.no_grad():
            for _ in range(n_measure):
                _ = model(xw, batch['pos_info'], batch['edge_indices'],
                          batch['edge_attrs'], batch['node_metadata'])
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        elapsed = time.time() - start

        per_batch_ms = (elapsed / n_measure) * 1000.0
        per_sample_ms = per_batch_ms / batch['x'].shape[0]

        peak_mb = None
        if device.startswith('cuda'):
            peak_bytes = torch.cuda.max_memory_allocated(device)
            peak_mb = peak_bytes / (1024 ** 2)

        n_params = sum(p.numel() for p in model.parameters())

        rows.append({
            'model':              model_name,
            'total_params':       n_params,
            'batch_size':         batch_size,
            'window_size':        window_size,
            'per_batch_ms':       round(per_batch_ms, 3),
            'per_sample_ms':      round(per_sample_ms, 4),
            'peak_inference_mb':  round(peak_mb, 1) if peak_mb is not None else None,
            'device':             device,
        })

        del model
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


# Theoretical scaling note
def theoretical_scaling_note(n_stations, n_levels, n_timesteps):
    """Returns a list-of-dict rows describing the edge counts for each
    edge type as functions of (N stations, L levels, T timesteps).
    Used to discuss scaling to global radiosonde networks (M1.4)."""
    return [
        {'edge_type': 'vertical',
         'formula':   'O(N * (L-1) * T)',
         'value_here': n_stations * max(0, n_levels - 1) * n_timesteps,
         'scaling':    'linear in stations and time, levels-bounded'},
        {'edge_type': 'horizontal',
         'formula':   'O(N^2 * L * T)',
         'value_here': n_stations * n_stations * n_levels * n_timesteps,
         'scaling':    'quadratic in stations — bottleneck for global networks'},
        {'edge_type': 'temporal',
         'formula':   'O(N * L * (T-1))',
         'value_here': n_stations * n_levels * max(0, n_timesteps - 1),
         'scaling':    'linear in stations and time'},
    ]


# Main
def main():
    device = cfg.device
    print("=" * 70)
    print(" COMPLEXITY PROFILING")
    print("=" * 70)
    print(f"Device: {device}")

    print("\n[1] Collecting static info from saved configs...")
    static_df = collect_static_info()
    if static_df.empty:
        print("  No config.json files found in results/. Skipping static info.")
    else:
        static_out = COMPARISON_DIR / "complexity_static.csv"
        static_df.to_csv(static_out, index=False)
        print(f"  Saved: {static_out}  ({len(static_df)} rows)")
        print(static_df.to_string(index=False))

    print("\n[2] Fresh inference profiling (GNN models with checkpoints)...")
    try:
        train_graph, test_graph = _build_train_and_test_graphs()
    except Exception as e:
        print(f"  [skip] graph build failed: {e}")
        train_graph = test_graph = None

    profile_df = pd.DataFrame()
    if test_graph is not None:
        profile_df = profile_gnn_inference(test_graph, device)
        if not profile_df.empty:
            profile_out = COMPARISON_DIR / "complexity_inference_profile.csv"
            profile_df.to_csv(profile_out, index=False)
            print(f"  Saved: {profile_out}  ({len(profile_df)} rows)")
            print(profile_df.to_string(index=False))
        else:
            print("  No GNN checkpoints available for profiling.")

    print("\n[3] Theoretical scaling for current Türkiye setup...")
    if train_graph is not None:
        md = train_graph['node_metadata']
        n_stations = len(set(md['station_id']))
        n_levels   = len(set([int(round(p)) for p in md['pressure']]))
        n_timesteps = len(set(md['datetime']))
        scaling_rows = theoretical_scaling_note(n_stations, n_levels, n_timesteps)
        scaling_df = pd.DataFrame(scaling_rows)
        scaling_out = COMPARISON_DIR / "complexity_scaling_theory.csv"
        scaling_df.to_csv(scaling_out, index=False)
        print(f"  Saved: {scaling_out}")
        print(f"  Setup: N={n_stations} stations, L={n_levels} levels, T={n_timesteps} timesteps")
        print(scaling_df.to_string(index=False))

    print("\n[4] Combined extended summary...")
    combined = COMPARISON_DIR / "complexity_extended.csv"
    if not static_df.empty and not profile_df.empty:
        # Join on model name where possible; static may have multiple seed rows
        agg_static = (static_df
                      .groupby('model', as_index=False)
                      .agg(mean_train_time_s=('train_time_s', 'mean'),
                           mean_infer_time_s=('infer_time_s', 'mean'),
                           total_params=('total_params', 'first'),
                           n_runs=('seed', 'size')))
        merged = agg_static.merge(profile_df, on='model', how='outer',
                                  suffixes=('_static', '_profile'))
        merged.to_csv(combined, index=False)
        print(f"  Saved: {combined}")
        print(merged.to_string(index=False))
    else:
        print("  [skip] not enough data for combined summary.")

    print("\n" + "=" * 70)
    print(" DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
