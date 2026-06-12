# -*- coding: utf-8 -*-
"""
SAITS (Self-Attention-based Imputation for Time Series) baseline wrapper.

Wraps the PyPOTS implementation of SAITS so it slots into the same
training/evaluation pipeline as the LSTM/CNN/MLP baselines in M07.
Each (station, pressure-level) is treated as an independent time
series of 6 features; sliding windows of seq_length time steps form
training samples. SAITS treats NaN as missing.

Reference: Du et al., "SAITS: Self-Attention-based Imputation for
Time Series", Expert Systems with Applications, 2023.
PyPOTS: https://pypots.com
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch

from M00_Config import cfg
from M07_DeepLearningBaselines import calculate_real_mae

try:
    from pypots.imputation.saits import SAITS as _PyPOTSSAITS
    HAS_PYPOTS = True
except ImportError:
    _PyPOTSSAITS = None
    HAS_PYPOTS = False


# Mirrors M07 conventions
MASK_SEED  = cfg.Masking.random_seed
MASK_RATIO = cfg.Masking.mask_ratio

FEATURE_NAMES = ['temperature', 'relative_humidity', 'wind_speed',
                 'sin_wd', 'cos_wd', 'geopotential']
FEATURE_UNITS = {
    'temperature':       '°C',
    'relative_humidity': '%',
    'wind_speed':        'm/s',
    'sin_wd':            '-',
    'cos_wd':            '-',
    'geopotential':      'm',
}


def _collect_station_level_indices(graph_data, seq_length):
    """Return list of dicts: {'indices': np.ndarray, 'station': str, 'pressure': float}.
    indices are chronologically sorted node indices for that
    (station, pressure) pair; pairs with fewer than seq_length samples
    are skipped (no sequence possible)."""
    metadata    = graph_data['node_metadata']
    station_ids = np.array(metadata['station_id'])
    pressures   = np.array(metadata['pressure'])
    datetimes   = metadata['datetime']

    unique_times = sorted(set(datetimes))
    time_to_idx  = {t: i for i, t in enumerate(unique_times)}
    unique_stations  = np.unique(station_ids)
    unique_pressures = np.unique(pressures)

    out = []
    for station in unique_stations:
        for pressure in unique_pressures:
            mask = (station_ids == station) & (np.abs(pressures - pressure) < 1.0)
            indices = np.where(mask)[0]
            if len(indices) < seq_length:
                continue
            t_idx = np.array([time_to_idx[datetimes[i]] for i in indices])
            indices = indices[np.argsort(t_idx)]
            out.append({'indices': indices,
                        'station': str(station),
                        'pressure': float(pressure)})
    return out


def _build_training_windows(graph_data, seq_length, mask_ratio, mode, seed=None):
    """Build (n_windows, seq_length, 6) tensors of inputs (X with NaN at
    masked positions) and targets (original). For mode='train', mask is
    re-randomised every call; for mode='test'/'val', a deterministic
    per-window seed is used."""
    x_full = graph_data['x']
    if isinstance(x_full, torch.Tensor):
        x_full = x_full.numpy()
    x_full = np.asarray(x_full, dtype=np.float32)

    pairs = _collect_station_level_indices(graph_data, seq_length)

    X_list, Y_list = [], []
    base_seed = seed if seed is not None else MASK_SEED
    win_counter = 0

    for pair in pairs:
        indices = pair['indices']
        for i in range(len(indices) - seq_length + 1):
            win_indices = indices[i:i + seq_length]
            window = x_full[win_indices].copy()
            target = window.copy()

            if mode != 'train':
                np.random.seed(base_seed + win_counter)
            rand_mask = np.random.rand(*window.shape) < mask_ratio
            full_mask = rand_mask | np.isnan(window)
            window[full_mask] = np.nan          # SAITS: NaN = missing

            X_list.append(window)
            Y_list.append(target)
            win_counter += 1

    X = np.asarray(X_list, dtype=np.float32)
    Y = np.asarray(Y_list, dtype=np.float32)
    return X, Y


def _evaluate_predictions_on_test(predictions, test_graph, mask_applied):
    """Per-feature R^2 / MAE / MAE_real / RMSE on the locations selected
    by mask_applied. Mirrors M07.evaluate_baseline_model's metric block."""
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    test_x  = test_graph['x']
    if isinstance(test_x, torch.Tensor):
        test_x = test_x.numpy()
    original = np.asarray(test_x, dtype=np.float32).copy()
    metadata = test_graph['node_metadata']
    pressures = np.array(metadata['pressure'])

    scaling_stats = test_graph.get('scaling_stats', {})
    level_stats   = scaling_stats.get('level_stats', {})

    results = []
    print(f"\n{'Feature':<20} | {'R²':>8} | {'MAE':>8} | {'MAE_real':>10} | {'Unit':>6}")
    print("-" * 65)

    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        feat_mask = mask_applied[:, feat_idx]
        if feat_mask.sum() < 10:
            continue
        orig = original[feat_mask, feat_idx]
        pred = predictions[feat_mask, feat_idx]
        p_masked = pressures[feat_mask]
        valid = ~(np.isnan(orig) | np.isnan(pred))
        if valid.sum() < 10:
            continue
        orig_v = orig[valid]
        pred_v = pred[valid]
        p_v    = p_masked[valid]

        r2   = r2_score(orig_v, pred_v)
        mae  = mean_absolute_error(orig_v, pred_v)
        rmse = float(np.sqrt(mean_squared_error(orig_v, pred_v)))

        if len(level_stats) > 0:
            mae_real = calculate_real_mae(orig_v, pred_v, p_v, level_stats, feat_idx)
        else:
            global_stds = scaling_stats.get('stds', np.ones(6))
            scale = global_stds[feat_idx] if isinstance(global_stds, np.ndarray) else 1.0
            mae_real = mae * scale

        results.append({
            'method':   'SAITS',
            'feature':  feat_name,
            'r2':       r2,
            'mae':      mae,
            'mae_real': mae_real,
            'unit':     FEATURE_UNITS.get(feat_name, '-'),
            'rmse':     rmse,
            'n':        int(valid.sum()),
        })
        print(f"{feat_name:<20} | {r2:>8.4f} | {mae:>8.4f} | {mae_real:>10.4f} | "
              f"{FEATURE_UNITS.get(feat_name, '-'):>6}")
    return pd.DataFrame(results)


def _sliding_window_impute(saits_model, test_graph, seq_length):
    """Run SAITS in inference mode over all (station, pressure) test
    sequences using sliding windows with step = seq_length // 2.
    Predictions for overlapping windows are averaged."""
    test_x = test_graph['x']
    if isinstance(test_x, torch.Tensor):
        test_x = test_x.numpy()
    test_x = np.asarray(test_x, dtype=np.float32)
    metadata = test_graph['node_metadata']
    station_ids = np.array(metadata['station_id'])
    pressures   = np.array(metadata['pressure'])

    # Shared mask (same RNG seed and ratio as M07 evaluate_baseline_model)
    np.random.seed(MASK_SEED)
    mask = np.random.rand(*test_x.shape) < MASK_RATIO
    mask = mask & ~np.isnan(test_x)

    test_x_nan = test_x.copy()
    test_x_nan[mask] = np.nan

    predictions = np.zeros_like(test_x, dtype=np.float32)
    pred_count  = np.zeros(len(test_x), dtype=np.int32)

    unique_stations  = np.unique(station_ids)
    unique_pressures = np.unique(pressures)
    step_size = max(1, seq_length // 2)

    all_windows         = []
    all_window_indices  = []

    for station in unique_stations:
        for pressure in unique_pressures:
            idx_mask = (station_ids == station) & (np.abs(pressures - pressure) < 1.0)
            indices = np.where(idx_mask)[0]
            if len(indices) < seq_length:
                continue
            seq_data_nan = test_x_nan[indices]
            for i in range(0, len(indices), step_size):
                end_i = min(i + seq_length, len(indices))
                if end_i - i < seq_length:
                    start_i = max(0, end_i - seq_length)
                else:
                    start_i = i
                window_indices = indices[start_i:start_i + seq_length]
                window_data    = seq_data_nan[start_i:start_i + seq_length]
                if len(window_data) < seq_length:
                    continue
                all_windows.append(window_data)
                all_window_indices.append(window_indices)

    if not all_windows:
        return predictions, mask

    all_X = np.asarray(all_windows, dtype=np.float32)
    print(f"  SAITS imputing {len(all_X)} sliding windows...")
    result = saits_model.impute(test_set={'X': all_X})
    if isinstance(result, dict):
        imputed = result.get('imputation', None)
        if imputed is None:
            raise RuntimeError("SAITS.impute() returned no 'imputation' key.")
    else:
        imputed = result  # older API: returns ndarray directly

    for w_idx, w_indices in enumerate(all_window_indices):
        for j, idx in enumerate(w_indices):
            if pred_count[idx] == 0:
                predictions[idx] = imputed[w_idx, j]
            else:
                predictions[idx] = (predictions[idx] * pred_count[idx] + imputed[w_idx, j]) \
                                   / (pred_count[idx] + 1)
            pred_count[idx] += 1

    # Fill any node never covered by a window (small / disjoint pairs)
    untouched = (pred_count == 0)
    if untouched.any():
        predictions[untouched] = np.nanmean(test_x, axis=0)

    return predictions, mask


def train_and_evaluate_saits(train_graph, test_graph, val_graph,
                             device='cuda',
                             num_epochs=None, batch_size=None,
                             seq_length=None, patience=None,
                             seed=None):
    """End-to-end: build SAITS, fit on train + val, evaluate on test."""
    if not HAS_PYPOTS:
        raise ImportError(
            "PyPOTS not installed. Install with: pip install pypots"
        )

    num_epochs = num_epochs if num_epochs is not None else cfg.Baseline.num_epochs
    batch_size = batch_size if batch_size is not None else cfg.Baseline.batch_size
    seq_length = seq_length if seq_length is not None else cfg.Baseline.seq_length
    patience   = patience   if patience   is not None else cfg.Baseline.patience
    # PyPOTS asserts patience < num_epochs. When num_epochs is too small for the
    # configured patience (e.g. --epochs smoke runs), disable early stopping
    # entirely rather than clamping to an artificial value. PyPOTS treats
    # patience=None as "no early stopping" (saits/model.py docstring, line 84).
    if patience is not None and patience >= num_epochs:
        print(f"  [info] SAITS: patience ({patience}) >= num_epochs ({num_epochs}); "
              f"disabling early stopping for this run.")
        patience = None

    # Build training and validation tensors. Y = original (pre-mask) targets;
    # PyPOTS uses val 'X_ori' as ground truth for validation metrics.
    train_X, _ = _build_training_windows(
        train_graph, seq_length, MASK_RATIO, mode='train', seed=seed,
    )
    val_X, val_Y = _build_training_windows(
        val_graph,   seq_length, MASK_RATIO, mode='val',   seed=MASK_SEED,
    )
    print(f"\nSAITS training data: {train_X.shape}  (windows x seq_length x features)")
    print(f"SAITS validation data: {val_X.shape}")

    # Model hyperparameters: d_model must be divisible by n_heads
    n_features = train_X.shape[-1]
    d_model    = 64
    n_heads    = 4
    saits = _PyPOTSSAITS(
        n_steps     = seq_length,
        n_features  = n_features,
        n_layers    = 2,
        d_model     = d_model,
        n_heads     = n_heads,
        d_k         = d_model // n_heads,
        d_v         = d_model // n_heads,
        d_ffn       = 128,
        dropout     = 0.1,
        batch_size  = batch_size,
        epochs      = num_epochs,
        patience    = patience,
        device      = device,
        verbose     = True,
    )

    saits.fit(
        train_set={'X': train_X},
        val_set={'X': val_X, 'X_ori': val_Y},
    )

    # Test-set inference + metrics
    predictions, mask_applied = _sliding_window_impute(saits, test_graph, seq_length)
    results_df = _evaluate_predictions_on_test(predictions, test_graph, mask_applied)

    # Arrays for M99 parity metrics (extreme / by_pressure / profile_consistency)
    metadata = test_graph['node_metadata']
    test_x_full = test_graph['x']
    if isinstance(test_x_full, torch.Tensor):
        test_x_full = test_x_full.numpy()
    original = np.asarray(test_x_full, dtype=np.float32).copy()
    eval_arrays = {
        'predictions': predictions,
        'targets':     original,
        'mask':        mask_applied,
        'pressures':   np.array(metadata['pressure']),
        'stations':    np.array(metadata['station_id']),
        'datetimes':   np.array(metadata['datetime']),
    }
    return saits, results_df, eval_arrays
