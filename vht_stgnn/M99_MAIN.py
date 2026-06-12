# -*- coding: utf-8 -*-
"""
M99_MAIN.py - Radiosonde VHT-GNN
Manuel seçimle modelleri çalıştır, sonuçları kaydet
Sonra M98_CompareResults.py ile devam et
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path

# CUDA Optimizasyonları
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  
    torch.backends.cuda.matmul.allow_tf32 = True  
    torch.backends.cudnn.allow_tf32 = True

# Modüller
from M00_Config import cfg
from M01_Utils import load_station_metadata
from M02_DataLoading import (RadiosondeLoader, RadiosondeGraphBuilder, 
                             RadiosondeSlidingWindowDataset, collate_graph_windows)
from M03_Model import RadiosondeSpatioTemporalGNN, TemporalAttention
from M04_Training import Trainer
# M05_Visualization import edilmez: visualize_predictions M99'da cagrilmiyor,
# ama M05 import'u seaborn'u tetikliyor. Seaborn bazi calistirma ortamlarinda
# kurulu olmayabilir -> ilk import satirinda fail eder, o yuzden import yok.


# Feature birimleri (Real MAE için)
FEATURE_UNITS = {
    'temperature': '°C',
    'relative_humidity': '%',
    'wind_speed': 'm/s',
    'sin_wd': '-',
    'cos_wd': '-',
    'geopotential': 'm'
}

# 
# WHICH MODELS TO RUN
# 


# GNN Modelleri
# 50 epoch 
RUN_VHT_GNN = False  # Evde Carşamba 11.02.2026
RUN_VANILLA_GRAPHSAGE = False #   # Evde Carşamba 11.02.2026
RUN_MULTISCALE_GRAPHSAGE = False  #  Okulda Perşembe 12.02.2026
RUN_GAT = False #  Okulda Perşembe 12.02.2026
RUN_MPNN = False  #  Okulda Perşembe 12.02.2026
RUN_FLAT_GRAPHSAGE = False  # true flat GNN baseline (V/H/T edges merged into one graph)

# Ablation models (component-drop variants of VHT-GNN)
RUN_VHT_GNN_NO_TEMPORAL  = False
RUN_VHT_GNN_NO_GATING    = False
RUN_VHT_GNN_FIXED_FUSION = False
RUN_VHT_GNN_NO_VERTICAL   = False        # drops vertical edge type
RUN_VHT_GNN_NO_HORIZONTAL = False        # drops horizontal edge type
RUN_VHT_GNN_GLOBAL_NORM  = False        # graph rebuilt with use_level_normalization=False
RUN_VANILLA_GRAPHSAGE_GLOBAL_NORM = False  # same: global-norm graph required

# Deep Learning Baselines
RUN_LSTM = False #  Okulda Perşembe 12.02.2026
RUN_CNN = False #  Okulda Perşembe 12.02.2026
RUN_MLP = False # Evde Carşamba 11.02.2026

# Transformer Baseline (PyPOTS SAITS)
RUN_SAITS = False

# İstatistiksel Baselines
RUN_STATISTICAL = True   # Evde Carşamba 11.02.2026


# sonuc klasoru
RESULTS_DIR = Path(cfg.results_dir)


# Multi-seed training: each seed produces an independent run saved under
# results/<model>/seed_<n>/. Statistical significance (paired Wilcoxon, std,
# confidence intervals) is computed across these runs in M14_SeedAggregation.
# Set to [42] for single-seed (backward-compatible behaviour).
SEEDS = [42, 123, 456, 789, 2024]


# LR scheduler info — M04 Trainer ve M07 BaselineTrainer ile koordineli.
# config.json'a yazilarak runtime'da kullanilan gercek scheduler ayarlari
# kayit altina alinir (reproducibility icin).
SCHEDULER_INFO = {
    "type":     "ReduceLROnPlateau",
    "mode":     "min",
    "factor":   0.5,
    "patience": 3,
    "min_lr":   1e-6,
}


def set_seed(seed: int):
    """Make a run reproducible: seeds numpy + torch + cuDNN."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 
# DATA QUALITY FILTERING
# 

def filter_data_by_time_step(df: pd.DataFrame, 
                              target_nan_ratio: float = None,
                              min_quality: float = 0.25,
                              min_stations: int = 3,
                              verbose: bool = True) -> pd.DataFrame:
    """
    Zaman adımı bazlı veri kalitesi filtreleme.
    
    Bu fonksiyon:
    - GNN için graf yapısını korur (tüm istasyonlar aynı zamanda)
    - LSTM/CNN için zaman serisi sürekliliğini korur
    - Tüm modeller için ADİL karşılaştırma sağlar
    
    Parameters:
    -----------
    df : pd.DataFrame
        Observations DataFrame
    target_nan_ratio : float
        Hedef NaN oranı. None ise sadece eşik bazlı filtreleme yapılır.
    min_quality : float
        Minimum zaman adımı kalitesi (0-1 arası)
    min_stations : int
        Minimum istasyon sayısı
    verbose : bool
        Detaylı çıktı
    
    Returns:
    --------
    pd.DataFrame : Filtrelenmiş veri
    """
    
    feature_cols = ['temperature', 'relative_humidity', 'wind_speed', 
                    'wind_direction', 'geopotential']
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    if verbose:
        print("\n" + "="*70)
        print(" DATA QUALITY FILTERING (Time Step Based)")
        print("="*70)
    
    # Mevcut durum
    initial_nan_ratio = df[feature_cols].isna().sum().sum() / (len(df) * len(feature_cols))
    initial_time_steps = df['datetime'].nunique()
    initial_rows = len(df)
    
    if verbose:
        print(f"\n BAŞLANGIÇ DURUMU:")
        print(f"   Satır: {initial_rows:,}")
        print(f"   Zaman adımı: {initial_time_steps:,}")
        print(f"   NaN oranı: {initial_nan_ratio:.1%}")
    
    # Zaman adımı kalitelerini hesapla
    time_quality = []
    for dt, group in df.groupby('datetime'):
        group_nan = group[feature_cols].isna().sum().sum()
        group_total = len(group) * len(feature_cols)
        quality = 1 - (group_nan / group_total) if group_total > 0 else 0
        n_stations = group['station_id'].nunique()
        
        time_quality.append({
            'datetime': dt,
            'quality': quality,
            'n_stations': n_stations,
            'n_rows': len(group)
        })
    
    time_df = pd.DataFrame(time_quality).sort_values('quality')
    
    # Hangi zaman adımlarını çıkaracağız?
    remove_times = set()
    
    if target_nan_ratio is not None:
        # Hedef bazlı filtreleme
        if verbose:
            print(f"\n HEDEF: NaN oranını {initial_nan_ratio:.1%} → {target_nan_ratio:.1%}")
        
        if initial_nan_ratio <= target_nan_ratio:
            if verbose:
                print(" Mevcut NaN oranı zaten hedefin altında!")
        else:
            # En kötüden başlayarak çıkar
            remaining_df = df.copy()
            
            for _, row in tqdm(time_df.iterrows(), 
                               total=len(time_df),
                               desc="Zaman adımları filtreleniyor",
                               disable=not verbose):
                
                current_nan = remaining_df[feature_cols].isna().sum().sum()
                current_total = len(remaining_df) * len(feature_cols)
                current_ratio = current_nan / current_total if current_total > 0 else 0
                
                if current_ratio <= target_nan_ratio:
                    break
                
                # Bu zaman adımını çıkar
                remove_times.add(row['datetime'])
                remaining_df = remaining_df[remaining_df['datetime'] != row['datetime']]
                
                if len(remaining_df) == 0:
                    break
    
    else:
        # Eşik bazlı filtreleme
        if verbose:
            print(f"\nEŞİK: Kalite < {min_quality:.0%} olan zaman adımlarını çıkar")
            print(f"     Minimum istasyon sayısı: {min_stations}")
        
        for _, row in time_df.iterrows():
            if row['quality'] < min_quality:
                remove_times.add(row['datetime'])
            elif row['n_stations'] < min_stations:
                remove_times.add(row['datetime'])
    
    # Filtreleme uygula
    filtered_df = df[~df['datetime'].isin(remove_times)].copy()
    
    # Sonuç istatistikleri
    final_nan_ratio = filtered_df[feature_cols].isna().sum().sum() / (len(filtered_df) * len(feature_cols)) if len(filtered_df) > 0 else 0
    final_time_steps = filtered_df['datetime'].nunique()
    final_rows = len(filtered_df)
    
    if verbose:
        print(f"\n FİLTRELEME SONUCU:")
        print(f"   Çıkarılan zaman adımı: {len(remove_times):,}")
        print(f"   Çıkarılan satır: {initial_rows - final_rows:,} ({100*(initial_rows - final_rows)/initial_rows:.1f}%)")
        print(f"   Kalan satır: {final_rows:,}")
        print(f"   NaN oranı: {initial_nan_ratio:.1%} -> {final_nan_ratio:.1%}")
        print("="*70)
    
    return filtered_df


# 
# HELPER FUNCTIONS
# 

def ensure_results_dir(model_name: str, seed=None) -> Path:
    """Returns the output directory for one model run.
    If seed is given, output goes under results/<model>/seed_<n>/ so that
    multi-seed runs do not overwrite each other. If seed is None, the
    legacy results/<model>/ path is used (backward compatible)."""
    if seed is None:
        model_dir = RESULTS_DIR / model_name
    else:
        model_dir = RESULTS_DIR / model_name / f"seed_{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def save_metrics(metrics_list: list, save_dir: Path, filename: str = "metrics.csv"):
    """Metrikleri CSV olarak kaydet."""
    df = pd.DataFrame(metrics_list)
    df.to_csv(save_dir / filename, index=False)
    print(f"Saved: {save_dir / filename}")


def save_config(model_name: str, model_cfg, model, training_time: float, 
                train_samples: int, test_samples: int, save_dir: Path):
    """Model config ve meta bilgileri JSON olarak kaydet."""
    
    # Parametre sayısını hesapla
    if model is not None:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        total_params = 0
        trainable_params = 0
    
    config_dict = {
        "model_name": model_name,
        "parameters": {
            "hidden_dim": getattr(model_cfg, 'hidden_dim', None),
            "num_gnn_layers": getattr(model_cfg, 'num_gnn_layers', None),
            "dropout": getattr(model_cfg, 'dropout', None),
            "batch_size": getattr(model_cfg, 'batch_size', None),
            "learning_rate": getattr(model_cfg, 'learning_rate', None),
            "num_epochs": getattr(model_cfg, 'num_epochs', None),
            "patience": getattr(model_cfg, 'patience', None),
        },
        "scheduler": SCHEDULER_INFO,
        "complexity": {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "training_time_seconds": round(training_time, 2),
        },
        "data": {
            "train_samples": train_samples,
            "test_samples": test_samples,
            "mask_ratio": cfg.Masking.mask_ratio,
            "mask_seed": cfg.Masking.random_seed,
            "data_quality_filtering": cfg.DataQuality.enable_filtering,
            "target_nan_ratio": cfg.DataQuality.target_nan_ratio,
        },
        "timestamp": datetime.now().isoformat()
    }

    with open(save_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"   Saved: {save_dir / 'config.json'}")


def save_training_history(history: dict, save_dir: Path):
    """Eğitim geçmişini CSV olarak kaydet."""
    if not history or 'train_losses' not in history:
        return
    
    df = pd.DataFrame({
        'epoch': range(1, len(history['train_losses']) + 1),
        'train_loss': history['train_losses'],
        'val_loss': history.get('val_losses', [None] * len(history['train_losses']))
    })
    df.to_csv(save_dir / "training_history.csv", index=False)
    print(f"Saved: {save_dir / 'training_history.csv'}")


def get_level_std(pressure: float, level_stats: dict, feature_idx: int) -> float:
    """
    Belirli bir basınç seviyesi için feature'ın std değerini döndürür.
    """
    level_key = f"p_{int(pressure)}"
    
    if level_key in level_stats:
        stds = level_stats[level_key].get('stds', None)
        if stds is not None:
            if isinstance(stds, np.ndarray):
                return float(stds[feature_idx])
            elif isinstance(stds, list):
                return float(stds[feature_idx])
    
    # Tam eşleşme yoksa en yakın seviyeyi bul
    available_levels = []
    for key in level_stats.keys():
        if key.startswith('p_'):
            try:
                p = int(key.split('_')[1])
                available_levels.append(p)
            except (ValueError, IndexError):
                pass
    
    if not available_levels:
        return 1.0
    
    nearest = min(available_levels, key=lambda x: abs(x - pressure))
    nearest_key = f"p_{nearest}"
    
    if nearest_key in level_stats:
        stds = level_stats[nearest_key].get('stds', None)
        if stds is not None:
            if isinstance(stds, np.ndarray):
                return float(stds[feature_idx])
            elif isinstance(stds, list):
                return float(stds[feature_idx])
    
    return 1.0


def calculate_real_mae(y_true: np.ndarray, y_pred: np.ndarray, 
                       pressures: np.ndarray, level_stats: dict, 
                       feature_idx: int) -> float:
    """Level-aware Real MAE hesapla."""
    if len(y_true) == 0:
        return np.nan
    
    real_errors = []
    
    for i in range(len(y_true)):
        pressure = pressures[i]
        std = get_level_std(pressure, level_stats, feature_idx)
        error_norm = abs(y_true[i] - y_pred[i])
        error_real = error_norm * std
        real_errors.append(error_real)
    
    return float(np.mean(real_errors))


def compute_vertical_profile_consistency(pred, target, pressures, stations, datetimes):
    """Per-profile Pearson correlation between predicted and observed
    vertical profiles. A profile is the values at all pressure levels
    of one (station, datetime) pair.

    Returns one row per (feature, profile-aggregation): mean, median, p5, p95.
    Surface-only features (sin_wd/cos_wd) excluded since direction has
    no meaningful vertical correlation structure.
    """
    if stations is None or datetimes is None:
        return []
    if len(stations) != len(pred) or len(datetimes) != len(pred):
        return []

    profile_features = [
        ('temperature',       0),
        ('relative_humidity', 1),
        ('wind_speed',        2),
        ('geopotential',      5),
    ]

    df_base = pd.DataFrame({
        'station':  stations,
        'time':     datetimes,
        'pressure': np.round(pressures).astype(int),
    })

    rows = []
    for fname, fi in profile_features:
        df = df_base.copy()
        df['y_true'] = target[:, fi]
        df['y_pred'] = pred[:, fi]
        df = df.dropna(subset=['y_true', 'y_pred'])
        # Aggregate duplicate (station, time, pressure) from sliding windows
        df = (df.groupby(['station', 'time', 'pressure'], as_index=False)
                .mean(numeric_only=True))

        corrs = []
        for (station, time), grp in df.groupby(['station', 'time']):
            if len(grp) < 3:
                continue
            grp = grp.sort_values('pressure', ascending=False)
            true_p = grp['y_true'].values
            pred_p = grp['y_pred'].values
            if np.std(true_p) < 1e-9 or np.std(pred_p) < 1e-9:
                continue
            r = np.corrcoef(true_p, pred_p)[0, 1]
            if np.isfinite(r):
                corrs.append(r)

        if not corrs:
            continue
        corrs = np.array(corrs)
        rows.append({
            'feature':  fname,
            'n_profiles': int(corrs.size),
            'mean_r':   round(float(np.mean(corrs)),   4),
            'median_r': round(float(np.median(corrs)), 4),
            'p5_r':     round(float(np.percentile(corrs, 5)),  4),
            'p95_r':    round(float(np.percentile(corrs, 95)), 4),
        })
    return rows


def _build_metric_row(y_true_norm, y_pred_norm, pressures_valid,
                      feature_name, feature_idx, scaling_stats):
    """Tek feature icin 12-sutunlu metric dict uretir.
    GNN'in evaluate_and_save_gnn icindeki metric block'unun aynisi - DL/SAITS/Statistical
    parite icin generic hale getirildi."""
    level_stats = scaling_stats.get('level_stats', {})

    r2   = r2_score(y_true_norm, y_pred_norm)
    mae  = mean_absolute_error(y_true_norm, y_pred_norm)
    rmse = float(np.sqrt(mean_squared_error(y_true_norm, y_pred_norm)))
    bias = float(np.mean(y_pred_norm - y_true_norm))

    if pressures_valid is not None and len(level_stats) > 0:
        mae_real = calculate_real_mae(
            y_true_norm, y_pred_norm, pressures_valid, level_stats, feature_idx,
        )
        bias_real = float(np.mean([
            (y_pred_norm[k] - y_true_norm[k]) * get_level_std(
                pressures_valid[k], level_stats, feature_idx,
            )
            for k in range(len(y_true_norm))
        ]))
    else:
        global_stds = scaling_stats.get('stds', np.ones(6))
        scale = global_stds[feature_idx] if isinstance(global_stds, np.ndarray) else 1.0
        mae_real  = mae * scale
        bias_real = bias * scale

    extreme_mask = np.abs(y_true_norm) > 3.0
    n_extreme = int(extreme_mask.sum())
    if n_extreme >= 10:
        y_true_ex = y_true_norm[extreme_mask]
        y_pred_ex = y_pred_norm[extreme_mask]
        r2_extreme  = float(r2_score(y_true_ex, y_pred_ex))
        mae_extreme = float(mean_absolute_error(y_true_ex, y_pred_ex))
    else:
        r2_extreme  = np.nan
        mae_extreme = np.nan

    unit = FEATURE_UNITS.get(feature_name, '-')

    return {
        'feature':     feature_name,
        'r2':          round(r2, 4),
        'mae':         round(mae, 4),
        'mae_real':    round(mae_real, 4),
        'unit':        unit,
        'rmse':        round(rmse, 4),
        'bias':        round(bias, 4),
        'bias_real':   round(bias_real, 4),
        'r2_extreme':  round(r2_extreme, 4) if not np.isnan(r2_extreme) else None,
        'mae_extreme': round(mae_extreme, 4) if not np.isnan(mae_extreme) else None,
        'n_extreme':   n_extreme,
        'n':           int(len(y_true_norm)),
    }


def build_metrics_list_from_arrays(predictions, targets, mask_applied,
                                   pressures, scaling_stats,
                                   feature_names=None,
                                   drop_humidity_above_200hpa=True):
    """4 model arasinda metric paritesi icin tek giris noktasi.
    predictions/targets normalized uzayda; mask_applied bool (N,F) - sadece bu
    noktalar metriklerde sayilir. Returns: list[dict] (12 sutunlu metrics_list)."""
    if feature_names is None:
        feature_names = ['temperature', 'relative_humidity', 'wind_speed',
                         'sin_wd', 'cos_wd', 'geopotential']

    metrics_list = []
    for i, fname in enumerate(feature_names):
        m = mask_applied[:, i].copy()
        if fname == 'relative_humidity' and drop_humidity_above_200hpa and pressures is not None:
            m = m & (pressures >= 200.0)
        if m.sum() < 10:
            continue
        y_true = targets[m, i]
        y_pred = predictions[m, i]
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        if valid.sum() < 10:
            continue
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        p_valid = pressures[m][valid] if pressures is not None else None
        metrics_list.append(
            _build_metric_row(y_true, y_pred, p_valid, fname, i, scaling_stats)
        )
    return metrics_list


def write_per_pressure_csv(predictions, targets, pressures, test_graph, save_dir,
                           eval_mask=None):
    """Calls evaluate_by_pressure and saves metrics_by_pressure.csv. No-op if empty.

    eval_mask: (N, F) bool — imputation hedeflerini gosterir. Verilmezse tum
    noktalarda olcer (reconstruction skoru, metrics.csv ile tutarsiz)."""
    scaling_stats = test_graph.get('scaling_stats', {})
    level_stats   = scaling_stats.get('level_stats', {})
    rows = evaluate_by_pressure(predictions, targets, test_graph,
                                batch_pressures=pressures, level_stats=level_stats,
                                eval_mask=eval_mask)
    if rows:
        save_metrics(rows, save_dir, "metrics_by_pressure.csv")


def dedup_sliding_window_predictions(input_cat, pred_cat, target_cat,
                                     pressures, stations, datetimes,
                                     n_features=6):
    """GNN sliding window overlap dedup. Ayni (station, pressure, datetime) icin:
    - pred: mean (overlap-averaging, LSTM/SAITS'in pred_count pattern'inin karsiligi)
    - input/target: first (deterministic; global mask + ayni target)
    LSTM/CNN/SAITS zaten internal pred_count ile dedup yapar; GNN sliding window'da
    bu yapilmadigi icin ayni node ~window_size kez gozukur, n inflate olur ve
    ornekler arasi korelasyon olur. Bu helper o farki kapatir."""
    df = pd.DataFrame({
        'station':  stations,
        'pressure': pressures,
        'datetime': datetimes,
        **{f'pred_{i}':   pred_cat[:, i]   for i in range(n_features)},
        **{f'input_{i}':  input_cat[:, i]  for i in range(n_features)},
        **{f'target_{i}': target_cat[:, i] for i in range(n_features)},
    })
    agg = {f'pred_{i}': 'mean' for i in range(n_features)}
    agg.update({f'input_{i}':  'first' for i in range(n_features)})
    agg.update({f'target_{i}': 'first' for i in range(n_features)})
    grouped = df.groupby(['station', 'pressure', 'datetime'], as_index=False).agg(agg)

    pred_d   = np.column_stack([grouped[f'pred_{i}'].values   for i in range(n_features)])
    input_d  = np.column_stack([grouped[f'input_{i}'].values  for i in range(n_features)])
    target_d = np.column_stack([grouped[f'target_{i}'].values for i in range(n_features)])
    pressures_d = grouped['pressure'].values
    stations_d  = grouped['station'].values
    datetimes_d = grouped['datetime'].values
    return input_d, pred_d, target_d, pressures_d, stations_d, datetimes_d


def write_profile_consistency_csv(predictions, targets, pressures, stations,
                                  datetimes, save_dir):
    """Calls compute_vertical_profile_consistency and saves profile_consistency.csv.
    No-op if stations/datetimes missing or no profile has 3+ levels."""
    rows = compute_vertical_profile_consistency(
        predictions, targets, pressures, stations, datetimes,
    )
    if rows:
        save_metrics(rows, save_dir, "profile_consistency.csv")


def evaluate_and_save_gnn(model, test_graph, model_cfg, device, model_name,
                          training_time, train_samples, test_samples, trainer=None,
                          seed=None):

    print(f"\n{model_name.upper()} Değerlendiriliyor...")

    save_dir = ensure_results_dir(model_name, seed=seed)
    model.eval()
    
    # Dataset oluştur
    window_size = model_cfg.dataset_window_size
    batch_size = model_cfg.batch_size
    
    # Test set'inde gerçek imputation: girdiyi maskele, model maskelenen noktaları tahmin etsin.
    # Pozisyonel arg sırası (graph_data, window_size, mask_ratio, use_realistic_masking, seed) —
    # keyword zorunlu, aksi halde mask_ratio slot'una use_realistic_masking düşer (False=0).
    test_ds = RadiosondeSlidingWindowDataset(
        test_graph, window_size,
        mask_ratio=cfg.Masking.mask_ratio,
        use_realistic_masking=cfg.use_realistic_masking,
        seed=cfg.Masking.random_seed,
    )
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_windows)

    # Tahmin ve metadata toplama
    all_inputs, all_preds, all_targets = [], [], []
    all_pressures, all_stations, all_datetimes = [], [], []

    # Inference time ölç
    inference_start = time.time()

    with torch.no_grad():
        for batch in tqdm(test_dl, desc=f"Evaluating {model_name}"):
            xw = batch['x'].to(device)
            target = batch['target'].to(device)
            pred, _ = model(xw, batch['pos_info'], batch['edge_indices'],
                           batch['edge_attrs'], batch['node_metadata'])
            all_inputs.append(xw.cpu())
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

            md = batch.get('node_metadata', {})
            if 'pressure' in md:
                bp = md['pressure']
                all_pressures.extend(bp if isinstance(bp, list) else bp.tolist())
            if 'station_id' in md:
                bs = md['station_id']
                all_stations.extend(bs if isinstance(bs, list) else list(bs))
            if 'datetime' in md:
                bd = md['datetime']
                all_datetimes.extend(bd if isinstance(bd, list) else list(bd))

    inference_time = time.time() - inference_start

    input_cat  = torch.cat(all_inputs,  0).numpy()
    pred_cat   = torch.cat(all_preds,   0).numpy()
    target_cat = torch.cat(all_targets, 0).numpy()
    pressures_cat = np.array(all_pressures) if all_pressures else None
    stations_cat = np.array(all_stations) if all_stations else None
    datetimes_cat = np.array(all_datetimes) if all_datetimes else None

    # Sliding window overlap dedup: ayni fiziksel node W kez tekrarlandi (overlap).
    # LSTM/SAITS internal pred_count ile dedup yapar; GNN'de yapilmiyordu.
    # Dedup olmadan n ~W kat inflate, ornekler arasi korelasyon var (Bug #5).
    if (pressures_cat is not None and stations_cat is not None and datetimes_cat is not None
            and len(pressures_cat) == len(pred_cat)):
        input_cat, pred_cat, target_cat, pressures_cat, stations_cat, datetimes_cat = (
            dedup_sliding_window_predictions(
                input_cat, pred_cat, target_cat,
                pressures_cat, stations_cat, datetimes_cat,
            )
        )

    scaling_stats = test_graph.get('scaling_stats', {})
    level_stats = scaling_stats.get('level_stats', {})

    feature_names = ['temperature', 'relative_humidity', 'wind_speed',
                     'sin_wd', 'cos_wd', 'geopotential']

    metrics_list = []

    print(f"\n{'Feature':<20} | {'R²':>8} | {'MAE':>8} | {'MAE_real':>10} | {'Unit':>6}")
    print("-" * 65)

    # Imputation hedefleri = modele NaN olarak verilen ama target'ta dolu olan noktalar.
    # Bu mask test_ds'in mask_ratio + seed parametrelerinden geliyor (dataset uyguladı);
    # burada artificial_mask yeniden uretmek YANLIS olurdu cunku model gercekten o
    # noktalari gormeden tahmin etti, baska bir mask uzerinde olcmek baseline'lar ile
    # parite'yi bozar.
    artificial_mask = np.isnan(input_cat) & ~np.isnan(target_cat)

    for i, fname in enumerate(feature_names):
        mask = artificial_mask[:, i]  #      DEĞİŞTİ: Sadece yapay maskelenen noktalar
        if fname == 'relative_humidity' and pressures_cat is not None:
            # Basıncı 200 hPa'dan küçük olanları (yani daha yüksek irtifaları) maskeden çıkar
            # pressures_cat numpy array olduğu için direkt karşılaştırma yapabiliriz
            valid_pressure_mask = (pressures_cat >= 200.0)
            mask = mask & valid_pressure_mask
        if mask.sum() < 10:
            continue
        
        y_true = target_cat[mask, i]
        y_pred = pred_cat[mask, i] 
        
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        if valid.sum() < 10:
            continue
        
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        
        # Normalized metrikler
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        bias = float(np.mean(y_pred - y_true))

        # Real-unit MAE (level-aware denormalize)
        if pressures_cat is not None and len(level_stats) > 0:
            p_valid = pressures_cat[mask][valid]
            mae_real = calculate_real_mae(y_true, y_pred, p_valid, level_stats, i)
            bias_real = float(np.mean(
                [(y_pred[k] - y_true[k]) * get_level_std(p_valid[k], level_stats, i)
                 for k in range(len(y_true))]
            ))
        else:
            global_stds = scaling_stats.get('stds', np.ones(6))
            scale = global_stds[i] if isinstance(global_stds, np.ndarray) else 1.0
            mae_real  = mae * scale
            bias_real = bias * scale

        # Extreme-event accuracy: |z| > 3 in normalized space
        # (level-aware norm makes y_true ~ N(0,1), so |y_true| > 3 = extreme)
        extreme_mask = np.abs(y_true) > 3.0
        n_extreme = int(extreme_mask.sum())
        if n_extreme >= 10:
            y_true_ex = y_true[extreme_mask]
            y_pred_ex = y_pred[extreme_mask]
            r2_extreme  = float(r2_score(y_true_ex, y_pred_ex))
            mae_extreme = float(mean_absolute_error(y_true_ex, y_pred_ex))
        else:
            r2_extreme  = np.nan
            mae_extreme = np.nan

        unit = FEATURE_UNITS.get(fname, '-')

        metrics_list.append({
            'feature':     fname,
            'r2':          round(r2, 4),
            'mae':         round(mae, 4),
            'mae_real':    round(mae_real, 4),
            'unit':        unit,
            'rmse':        round(rmse, 4),
            'bias':        round(bias, 4),
            'bias_real':   round(bias_real, 4),
            'r2_extreme':  round(r2_extreme, 4) if not np.isnan(r2_extreme) else None,
            'mae_extreme': round(mae_extreme, 4) if not np.isnan(mae_extreme) else None,
            'n_extreme':   n_extreme,
            'n':           int(valid.sum()),
        })

        print(f"{fname:<20} | {r2:>8.4f} | {mae:>8.4f} | {mae_real:>10.4f} | {unit:>6}")
    
    # Basınç seviyesine göre metrikler — sadece imputation hedeflerinde ölç,
    # metrics.csv ile tutarlı (Bug #4 fix). Aksi halde reconstruction skoru çıkar.
    pressure_metrics = evaluate_by_pressure(pred_cat, target_cat, test_graph,
                                            batch_pressures=pressures_cat,
                                            level_stats=level_stats,
                                            eval_mask=artificial_mask)

    # Vertical profile consistency (per (station, time) Pearson r)
    profile_consistency = compute_vertical_profile_consistency(
        pred_cat, target_cat, pressures_cat, stations_cat, datetimes_cat,
    )

    # Kaydet
    save_metrics(metrics_list, save_dir, "metrics.csv")

    if pressure_metrics:
        save_metrics(pressure_metrics, save_dir, "metrics_by_pressure.csv")

    if profile_consistency:
        save_metrics(profile_consistency, save_dir, "profile_consistency.csv")
    
    # Config kaydet
    save_config(model_name, model_cfg, model, training_time, train_samples, test_samples, save_dir)
    
    # Inference time'ı config'e ekle
    config_path = save_dir / "config.json"
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    config_data['complexity']['inference_time_seconds'] = round(inference_time, 2)
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    # Training history kaydet
    if trainer is not None:
        history = trainer.get_training_history()
        save_training_history(history, save_dir)
    
    # Model ağırlıkları kaydet
    torch.save(model.state_dict(), save_dir / "model.pt")
    print(f"Saved: {save_dir / 'model.pt'}")
    
    return metrics_list


def evaluate_by_pressure(predictions, targets, test_graph,
                         batch_pressures=None, level_stats=None,
                         eval_mask=None):
    """Basınç seviyesine göre metrikleri hesapla.

    eval_mask: (N, F) bool — sadece bu noktalarda ölç (imputation hedefleri).
    None verilirse tum noktalarda ölcer (reconstruction skoru, eski davranis).
    Imputation degerlendirmesi icin eval_mask verilmesi sart, aksi halde
    metrics.csv (imputation) ile metrics_by_pressure.csv (reconstruction)
    farkli seyleri olcer."""

    feature_names = ['temperature', 'relative_humidity', 'wind_speed',
                     'sin_wd', 'cos_wd', 'geopotential']

    if batch_pressures is not None:
        pressures = np.array(batch_pressures) if not isinstance(batch_pressures, np.ndarray) else batch_pressures
    else:
        metadata = test_graph['node_metadata']
        pressures = np.array(metadata['pressure'])

    if level_stats is None:
        scaling_stats = test_graph.get('scaling_stats', {})
        level_stats = scaling_stats.get('level_stats', {})

    if len(pressures) != len(predictions):
        print(f"   Pressure metrics: size mismatch ({len(pressures)} vs {len(predictions)})")
        return []

    # 14 standard pressure levels. Each data point is assigned to EXACTLY ONE
    # nearest level (no overlap). 70/50/30/10 hPa are 20 hPa apart, so the old
    # < 25 hPa tolerance caused 2-3x double-counting in upper-atmosphere buckets.
    standard_levels_arr = np.array(
        [1000, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 10],
        dtype=np.float64,
    )
    dist_matrix       = np.abs(pressures[:, None] - standard_levels_arr[None, :])
    nearest_level_idx = np.argmin(dist_matrix, axis=1)
    nearest_dist      = dist_matrix[np.arange(len(pressures)), nearest_level_idx]
    # Drop points that aren't really near any standard level (half the smallest gap = 10 hPa)
    nearest_level_idx[nearest_dist > 10.0] = -1

    unique_pressures = [
        standard_levels_arr[i] for i in range(len(standard_levels_arr))
        if (nearest_level_idx == i).any()
    ]

    results = []

    for pressure in unique_pressures:
        level_i = int(np.argmin(np.abs(standard_levels_arr - pressure)))
        level_mask = (nearest_level_idx == level_i)

        if level_mask.sum() < 50:
            continue

        for feat_idx, feat_name in enumerate(feature_names):
            # Imputation hedefleri ile kesisim — verilen mask'in feature-spesifik
            # kolonu ile level_mask'i kombine et.
            if eval_mask is not None:
                point_mask = level_mask & eval_mask[:, feat_idx]
            else:
                point_mask = level_mask

            y_true = targets[point_mask, feat_idx]
            y_pred = predictions[point_mask, feat_idx]
            
            valid = ~(np.isnan(y_true) | np.isnan(y_pred))
            if valid.sum() < 10:
                continue
            
            y_true_valid = y_true[valid]
            y_pred_valid = y_pred[valid]
            
            r2 = r2_score(y_true_valid, y_pred_valid)
            mae = mean_absolute_error(y_true_valid, y_pred_valid)
            rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
            bias = float(np.mean(y_pred_valid - y_true_valid))

            std = get_level_std(pressure, level_stats, feat_idx)
            mae_real  = mae * std
            bias_real = bias * std

            unit = FEATURE_UNITS.get(feat_name, '-')

            results.append({
                'feature':   feat_name,
                'pressure':  int(pressure),
                'r2':        round(r2, 4),
                'mae':       round(mae, 4),
                'mae_real':  round(mae_real, 4),
                'unit':      unit,
                'rmse':      round(rmse, 4),
                'bias':      round(bias, 4),
                'bias_real': round(bias_real, 4),
                'n':         int(valid.sum()),
            })
    
    return results


def cleanup_after_training():
    """Eğitim sonrası bellek temizliği."""
    torch.cuda.empty_cache()
    if hasattr(TemporalAttention, 'reset_counters'):
        TemporalAttention.reset_counters()


def print_model_config(model_name, model_cfg):
    """Model konfigürasyonunu yazdırır."""
    print(f"\n {model_name.upper()} Konfigürasyonu:")
    print(f"   hidden_dim      = {getattr(model_cfg, 'hidden_dim', 'N/A')}")
    print(f"   num_gnn_layers  = {getattr(model_cfg, 'num_gnn_layers', 'N/A')}")
    print(f"   dropout         = {getattr(model_cfg, 'dropout', 'N/A')}")
    print(f"   batch_size      = {getattr(model_cfg, 'batch_size', 'N/A')}")
    print(f"   learning_rate   = {getattr(model_cfg, 'learning_rate', 'N/A')}")
    print(f"   num_epochs      = {getattr(model_cfg, 'num_epochs', 'N/A')}")
    print(f"   patience        = {getattr(model_cfg, 'patience', 'N/A')}")


# 
# GNN MODEL FUNCTIONS
# 

def run_vht_gnn(train_graph, test_graph, val_graph, train_samples, test_samples, seed=None):
    """vth_gnn modelini eğit ve değerlendir."""

    print("\n" + "="*60)
    print(f"VHT_GNN" + (f"  (seed={seed})" if seed is not None else ""))
    print("="*60)

    cleanup_after_training()
    if seed is not None:
        set_seed(seed)

    gs_cfg = cfg.VHT_GNN
    print_model_config('vht_gnn', gs_cfg)

    model = RadiosondeSpatioTemporalGNN(
        input_dim=cfg.input_dim,
        hidden_dim=gs_cfg.hidden_dim,
        num_gnn_layers=gs_cfg.num_gnn_layers,
        dropout=gs_cfg.dropout,
        model_type='vht_gnn'
    )

    trainer = Trainer(
        model=model,
        train_graph=train_graph,
        test_graph=test_graph,
        val_graph=val_graph,
        window_size=gs_cfg.dataset_window_size,
        batch_size=gs_cfg.batch_size,
        learning_rate=gs_cfg.learning_rate,
        patience=gs_cfg.patience,
        device=cfg.device,
        save_dir=cfg.save_dir,
        use_realistic_masking=cfg.use_realistic_masking,
        use_physics_informed=cfg.use_physics_informed_loss
    )

    start_time = time.time()
    trainer.fit(num_epochs=gs_cfg.num_epochs)
    training_time = time.time() - start_time

    metrics = evaluate_and_save_gnn(
        model, test_graph, gs_cfg, cfg.device, 'vht_gnn',
        training_time, train_samples, test_samples, trainer, seed=seed,
    )

    del model, trainer
    torch.cuda.empty_cache()

    return metrics

def run_vanilla_graphsage(train_graph, test_graph, val_graph, train_samples, test_samples, seed=None):
    """Multi-Relational GraphSAGE (edge attribute'suz) modelini eğit ve değerlendir."""

    print("\n" + "="*60)
    print("     VANILLA GRAPHSAGE (Baseline)" + (f"  (seed={seed})" if seed is not None else ""))
    print("="*60)

    cleanup_after_training()
    if seed is not None:
        set_seed(seed)

    vs_cfg = cfg.VanillaGraphSAGE
    print_model_config('vanilla_graphsage', vs_cfg)
    
    model = RadiosondeSpatioTemporalGNN(
        input_dim=cfg.input_dim,
        hidden_dim=vs_cfg.hidden_dim,
        num_gnn_layers=vs_cfg.num_gnn_layers,
        dropout=vs_cfg.dropout,
        model_type='vanilla_graphsage'
    )
    
    trainer = Trainer(
        model=model,
        train_graph=train_graph,
        test_graph=test_graph,
        val_graph=val_graph,
        window_size=vs_cfg.dataset_window_size,
        batch_size=vs_cfg.batch_size,
        learning_rate=vs_cfg.learning_rate,
        patience=vs_cfg.patience,
        device=cfg.device,
        save_dir=cfg.save_dir,
        use_realistic_masking=cfg.use_realistic_masking,
        use_physics_informed=cfg.use_physics_informed_loss
    )
    
    start_time = time.time()
    trainer.fit(num_epochs=vs_cfg.num_epochs)
    training_time = time.time() - start_time
    
    metrics = evaluate_and_save_gnn(
        model, test_graph, vs_cfg, cfg.device, 'vanilla_graphsage',
        training_time, train_samples, test_samples, trainer, seed=seed,
    )
    
    del model, trainer
    torch.cuda.empty_cache()
    
    return metrics

def run_multiscale_graphsage(train_graph, test_graph, val_graph, train_samples, test_samples, seed=None):
    """MultiscaleGraphSAGE modelini eğit ve değerlendir."""

    print("\n" + "="*60)
    print(" MULTISCALE GRAPHSAGE" + (f"  (seed={seed})" if seed is not None else ""))
    print("="*60)

    cleanup_after_training()
    if seed is not None:
        set_seed(seed)

    ms_cfg = cfg.MultiscaleGraphSAGE
    print_model_config('MultiscaleGraphSAGE', ms_cfg)
    print(f"   scales          = {ms_cfg.scales}")
    
    model = RadiosondeSpatioTemporalGNN(
        input_dim=cfg.input_dim,
        hidden_dim=ms_cfg.hidden_dim,
        num_gnn_layers=ms_cfg.num_gnn_layers,
        dropout=ms_cfg.dropout,
        model_type='multiscale_graphsage'
    )
    
    trainer = Trainer(
        model=model,
        train_graph=train_graph,
        test_graph=test_graph,
        val_graph=val_graph,
        window_size=ms_cfg.dataset_window_size,
        batch_size=ms_cfg.batch_size,
        learning_rate=ms_cfg.learning_rate,
        patience=ms_cfg.patience,
        device=cfg.device,
        save_dir=cfg.save_dir,
        use_realistic_masking=cfg.use_realistic_masking,
        use_physics_informed=cfg.use_physics_informed_loss
    )
    
    start_time = time.time()
    trainer.fit(num_epochs=ms_cfg.num_epochs)
    training_time = time.time() - start_time
    
    metrics = evaluate_and_save_gnn(
        model, test_graph, ms_cfg, cfg.device, 'multiscale_graphsage',
        training_time, train_samples, test_samples, trainer, seed=seed,
    )
    
    del model, trainer
    torch.cuda.empty_cache()
    
    return metrics


def run_gat(train_graph, test_graph, val_graph, train_samples, test_samples, seed=None):
    """GAT modelini eğit ve değerlendir."""

    print("\n" + "="*60)
    print(" GAT (Graph Attention Network)" + (f"  (seed={seed})" if seed is not None else ""))
    print("="*60)

    cleanup_after_training()
    if seed is not None:
        set_seed(seed)

    gat_cfg = cfg.Gat
    print_model_config('GAT', gat_cfg)
    print(f"   heads           = {gat_cfg.heads}")
    
    # Hidden dim kontrolü
    if gat_cfg.hidden_dim % gat_cfg.heads != 0:
        corrected_hidden_dim = gat_cfg.heads * (gat_cfg.hidden_dim // gat_cfg.heads)
        print(f"  hidden_dim düzeltildi: {corrected_hidden_dim}")
    else:
        corrected_hidden_dim = gat_cfg.hidden_dim
    
    model = RadiosondeSpatioTemporalGNN(
        input_dim=cfg.input_dim,
        hidden_dim=corrected_hidden_dim,
        num_gnn_layers=gat_cfg.num_gnn_layers,
        dropout=gat_cfg.dropout,
        model_type='gat',
        heads=gat_cfg.heads
    )
    
    trainer = Trainer(
        model=model,
        train_graph=train_graph,
        test_graph=test_graph,
        val_graph=val_graph,
        window_size=gat_cfg.dataset_window_size,
        batch_size=gat_cfg.batch_size,
        learning_rate=gat_cfg.learning_rate,
        patience=gat_cfg.patience,
        device=cfg.device,
        save_dir=cfg.save_dir,
        use_realistic_masking=cfg.use_realistic_masking,
        use_physics_informed=cfg.use_physics_informed_loss
    )
    
    start_time = time.time()
    trainer.fit(num_epochs=gat_cfg.num_epochs)
    training_time = time.time() - start_time
    
    metrics = evaluate_and_save_gnn(
        model, test_graph, gat_cfg, cfg.device, 'gat',
        training_time, train_samples, test_samples, trainer, seed=seed,
    )
    
    del model, trainer
    torch.cuda.empty_cache()
    
    return metrics


def run_mpnn(train_graph, test_graph, val_graph, train_samples, test_samples, seed=None):
    """MPNN modelini eğit ve değerlendir."""

    print("\n" + "="*60)
    print(" MPNN (Message Passing Neural Network)" + (f"  (seed={seed})" if seed is not None else ""))
    print("="*60)

    cleanup_after_training()
    if seed is not None:
        set_seed(seed)

    mpnn_cfg = cfg.Mpnn
    print_model_config('MPNN', mpnn_cfg)
    
    model = RadiosondeSpatioTemporalGNN(
        input_dim=cfg.input_dim,
        hidden_dim=mpnn_cfg.hidden_dim,
        num_gnn_layers=mpnn_cfg.num_gnn_layers,
        dropout=mpnn_cfg.dropout,
        model_type='mpnn'
    )
    
    trainer = Trainer(
        model=model,
        train_graph=train_graph,
        test_graph=test_graph,
        val_graph=val_graph,
        window_size=mpnn_cfg.dataset_window_size,
        batch_size=mpnn_cfg.batch_size,
        learning_rate=mpnn_cfg.learning_rate,
        patience=mpnn_cfg.patience,
        device=cfg.device,
        save_dir=cfg.save_dir,
        use_realistic_masking=cfg.use_realistic_masking,
        use_physics_informed=cfg.use_physics_informed_loss
    )
    
    start_time = time.time()
    trainer.fit(num_epochs=mpnn_cfg.num_epochs)
    training_time = time.time() - start_time
    
    metrics = evaluate_and_save_gnn(
        model, test_graph, mpnn_cfg, cfg.device, 'mpnn',
        training_time, train_samples, test_samples, trainer, seed=seed,
    )
    
    del model, trainer
    torch.cuda.empty_cache()

    return metrics


#
# ABLATION MODELS  (M2.3 — component-drop variants of VHT-GNN)
#

def _run_gnn_ablation(name, model_type, sub_cfg, train_graph, test_graph, val_graph,
                      train_samples, test_samples, seed=None):
    """Generic GNN ablation runner. Wraps the same fit + evaluate pipeline
    used by run_vht_gnn et al. Output goes to results/<name>/(seed_<n>/)."""
    print("\n" + "="*60)
    print(f" {name.upper()}" + (f"  (seed={seed})" if seed is not None else ""))
    print("="*60)

    cleanup_after_training()
    if seed is not None:
        set_seed(seed)
    print_model_config(name, sub_cfg)

    model = RadiosondeSpatioTemporalGNN(
        input_dim=cfg.input_dim,
        hidden_dim=sub_cfg.hidden_dim,
        num_gnn_layers=sub_cfg.num_gnn_layers,
        dropout=sub_cfg.dropout,
        model_type=model_type,
    )

    trainer = Trainer(
        model=model,
        train_graph=train_graph,
        test_graph=test_graph,
        val_graph=val_graph,
        window_size=sub_cfg.dataset_window_size,
        batch_size=sub_cfg.batch_size,
        learning_rate=sub_cfg.learning_rate,
        patience=sub_cfg.patience,
        device=cfg.device,
        save_dir=cfg.save_dir,
        use_realistic_masking=cfg.use_realistic_masking,
        use_physics_informed=cfg.use_physics_informed_loss,
    )

    start_time = time.time()
    trainer.fit(num_epochs=sub_cfg.num_epochs)
    training_time = time.time() - start_time

    metrics = evaluate_and_save_gnn(
        model, test_graph, sub_cfg, cfg.device, name,
        training_time, train_samples, test_samples, trainer, seed=seed,
    )

    del model, trainer
    torch.cuda.empty_cache()
    return metrics


def run_vht_gnn_no_temporal(train_graph, test_graph, val_graph, train_samples, test_samples, seed=None):
    return _run_gnn_ablation(
        'vht_gnn_no_temporal', 'vht_gnn_no_temporal', cfg.VHT_GNN_NoTemporal,
        train_graph, test_graph, val_graph, train_samples, test_samples, seed=seed,
    )


def run_vht_gnn_no_gating(train_graph, test_graph, val_graph, train_samples, test_samples, seed=None):
    return _run_gnn_ablation(
        'vht_gnn_no_gating', 'vht_gnn_no_gating', cfg.VHT_GNN_NoGating,
        train_graph, test_graph, val_graph, train_samples, test_samples, seed=seed,
    )


def run_vht_gnn_fixed_fusion(train_graph, test_graph, val_graph, train_samples, test_samples, seed=None):
    return _run_gnn_ablation(
        'vht_gnn_fixed_fusion', 'vht_gnn_fixed_fusion', cfg.VHT_GNN_FixedFusion,
        train_graph, test_graph, val_graph, train_samples, test_samples, seed=seed,
    )


def run_vht_gnn_no_vertical(train_graph, test_graph, val_graph, train_samples, test_samples, seed=None):
    return _run_gnn_ablation(
        'vht_gnn_no_vertical', 'vht_gnn_no_vertical', cfg.VHT_GNN_NoVertical,
        train_graph, test_graph, val_graph, train_samples, test_samples, seed=seed,
    )


def run_vht_gnn_no_horizontal(train_graph, test_graph, val_graph, train_samples, test_samples, seed=None):
    return _run_gnn_ablation(
        'vht_gnn_no_horizontal', 'vht_gnn_no_horizontal', cfg.VHT_GNN_NoHorizontal,
        train_graph, test_graph, val_graph, train_samples, test_samples, seed=seed,
    )


def run_flat_graphsage(train_graph, test_graph, val_graph, train_samples, test_samples, seed=None):
    """True flat GNN baseline: V/H/T edges merged into a single homogeneous graph.
    Shares the fit + evaluate pipeline with the other GNN models."""
    return _run_gnn_ablation(
        'flat_graphsage', 'flat_graphsage', cfg.FlatGraphSAGE,
        train_graph, test_graph, val_graph, train_samples, test_samples, seed=seed,
    )


def run_vht_gnn_global_norm(train_graph_gn, test_graph_gn, val_graph_gn,
                            train_samples_gn, test_samples_gn, seed=None):
    """Uses GLOBAL-NORM graphs (built with use_level_normalization=False).
    Same architecture as full VHT-GNN; isolates the contribution of
    level-aware normalization."""
    return _run_gnn_ablation(
        'vht_gnn_global_norm', 'vht_gnn', cfg.VHT_GNN_GlobalNorm,
        train_graph_gn, test_graph_gn, val_graph_gn,
        train_samples_gn, test_samples_gn, seed=seed,
    )


def run_vanilla_graphsage_global_norm(train_graph_gn, test_graph_gn, val_graph_gn,
                                      train_samples_gn, test_samples_gn, seed=None):
    """Uses GLOBAL-NORM graphs. Same architecture as Multi-Relational GraphSAGE;
    paired with run_vht_gnn_global_norm to complete the 2x2 norm x architecture
    ablation matrix."""
    return _run_gnn_ablation(
        'vanilla_graphsage_global_norm', 'vanilla_graphsage', cfg.VanillaGraphSAGE_GlobalNorm,
        train_graph_gn, test_graph_gn, val_graph_gn,
        train_samples_gn, test_samples_gn, seed=seed,
    )


#
# DEEP LEARNING BASELINES
#


def run_deep_learning_baselines(train_graph, test_graph, val_graph, run_lstm, run_cnn, run_mlp, seed=None):
    """LSTM, CNN, MLP baseline modellerini çalıştır."""

    try:
        from M07_DeepLearningBaselines import train_and_evaluate_baseline
    except ImportError:
        print(" M07_DeepLearningBaselines import edilemedi!")
        return

    baseline_cfg = cfg.Baseline

    models_to_run = []
    if run_lstm:
        models_to_run.append('lstm')
    if run_cnn:
        models_to_run.append('cnn')
    if run_mlp:
        models_to_run.append('mlp')

    for model_type in models_to_run:
        if seed is not None:
            set_seed(seed)

        print("\n" + "="*60)
        print(f" {model_type.upper()} (Deep Learning Baseline) | seed={seed}")
        print("="*60)

        start_time = time.time()

        model, results_df, training_history, eval_arrays = train_and_evaluate_baseline(
            model_type=model_type,
            train_graph=train_graph,
            test_graph=test_graph,
            val_graph=val_graph,
            device=cfg.device,
            num_epochs=baseline_cfg.num_epochs,
            batch_size=baseline_cfg.batch_size,
            seq_length=baseline_cfg.seq_length,
            patience=baseline_cfg.patience
        )

        training_time = time.time() - start_time

        # Sonuçları kaydet
        save_dir = ensure_results_dir(model_type, seed=seed)

        # Metrics — GNN ile tam parite (12 sutun: + extreme)
        if eval_arrays is not None:
            scaling_stats = test_graph.get('scaling_stats', {})
            metrics_list = build_metrics_list_from_arrays(
                predictions=eval_arrays['predictions'],
                targets=eval_arrays['targets'],
                mask_applied=eval_arrays['mask'],
                pressures=eval_arrays['pressures'],
                scaling_stats=scaling_stats,
            )
            if metrics_list:
                save_metrics(metrics_list, save_dir, "metrics.csv")

            # by_pressure + profile_consistency (GNN parite). eval_mask geçilmesi
            # şart: aksi halde by_pressure reconstruction skoru üretir, metrics.csv
            # ile farklı şeyler ölçer (Bug #4).
            write_per_pressure_csv(
                predictions=eval_arrays['predictions'],
                targets=eval_arrays['targets'],
                pressures=eval_arrays['pressures'],
                test_graph=test_graph,
                save_dir=save_dir,
                eval_mask=eval_arrays['mask'],
            )
            write_profile_consistency_csv(
                predictions=eval_arrays['predictions'],
                targets=eval_arrays['targets'],
                pressures=eval_arrays['pressures'],
                stations=eval_arrays['stations'],
                datetimes=eval_arrays['datetimes'],
                save_dir=save_dir,
            )

        # Training history CSV (GNN ile parite)
        if training_history and len(training_history.get('train_loss', [])) > 0:
            train_losses = training_history['train_loss']
            val_losses = training_history.get('val_loss', [None] * len(train_losses))
            hist_rows = [
                {'epoch': i + 1, 'train_loss': tl, 'val_loss': vl}
                for i, (tl, vl) in enumerate(zip(train_losses, val_losses))
            ]
            save_metrics(hist_rows, save_dir, "training_history.csv")
        
        # Config
        total_params = sum(p.numel() for p in model.parameters()) if model else 0
        config_dict = {
            "model_name": model_type.upper(),
            "parameters": {
                "hidden_dim": getattr(baseline_cfg, f'{model_type}_hidden_dim', None),
                "seq_length": baseline_cfg.seq_length,
                "batch_size": baseline_cfg.batch_size,
                "learning_rate": baseline_cfg.learning_rate,
                "num_epochs": baseline_cfg.num_epochs,
                "patience": baseline_cfg.patience,
            },
            "scheduler": SCHEDULER_INFO,
            "complexity": {
                "total_parameters": total_params,
                "training_time_seconds": round(training_time, 2),
            },
            "data": {
                "mask_ratio": cfg.Masking.mask_ratio,
                "mask_seed": cfg.Masking.random_seed,
                "data_quality_filtering": cfg.DataQuality.enable_filtering,
                "target_nan_ratio": cfg.DataQuality.target_nan_ratio,
            },
            "timestamp": datetime.now().isoformat()
        }
        
        with open(save_dir / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Saved: {save_dir / 'config.json'}")
        
        # Model ağırlıkları
        if model is not None:
            torch.save(model.state_dict(), save_dir / "model.pt")
            print(f"Saved: {save_dir / 'model.pt'}")
        
        del model
        torch.cuda.empty_cache()


#
# TRANSFORMER BASELINE (SAITS via PyPOTS)
#

def run_saits(train_graph, test_graph, val_graph, train_samples, test_samples, seed=None):
    """SAITS Transformer baseline modelini eğit ve değerlendir."""

    try:
        from M08_TransformerBaselines import train_and_evaluate_saits, HAS_PYPOTS
    except ImportError as e:
        print(f"  [skip] M08_TransformerBaselines import error: {e}")
        return
    if not HAS_PYPOTS:
        print("  [skip] PyPOTS not installed. Run: pip install pypots")
        return

    print("\n" + "="*60)
    print(" SAITS (Transformer Baseline)" + (f"  (seed={seed})" if seed is not None else ""))
    print("="*60)

    cleanup_after_training()
    if seed is not None:
        set_seed(seed)

    baseline_cfg = cfg.Baseline

    start_time = time.time()
    saits_model, results_df, eval_arrays = train_and_evaluate_saits(
        train_graph=train_graph,
        test_graph=test_graph,
        val_graph=val_graph,
        device=cfg.device,
        num_epochs=baseline_cfg.num_epochs,
        batch_size=baseline_cfg.batch_size,
        seq_length=baseline_cfg.seq_length,
        patience=baseline_cfg.patience,
        seed=seed,
    )
    training_time = time.time() - start_time

    save_dir = ensure_results_dir('saits', seed=seed)

    # Metrics — GNN ile tam parite (12 sutun: + bias + extreme)
    if eval_arrays is not None:
        scaling_stats = test_graph.get('scaling_stats', {})
        metrics_list = build_metrics_list_from_arrays(
            predictions=eval_arrays['predictions'],
            targets=eval_arrays['targets'],
            mask_applied=eval_arrays['mask'],
            pressures=eval_arrays['pressures'],
            scaling_stats=scaling_stats,
        )
        if metrics_list:
            save_metrics(metrics_list, save_dir, "metrics.csv")

        # by_pressure + profile_consistency (GNN parite). eval_mask sart (Bug #4 fix).
        write_per_pressure_csv(
            predictions=eval_arrays['predictions'],
            targets=eval_arrays['targets'],
            pressures=eval_arrays['pressures'],
            test_graph=test_graph,
            save_dir=save_dir,
            eval_mask=eval_arrays['mask'],
        )
        write_profile_consistency_csv(
            predictions=eval_arrays['predictions'],
            targets=eval_arrays['targets'],
            pressures=eval_arrays['pressures'],
            stations=eval_arrays['stations'],
            datetimes=eval_arrays['datetimes'],
            save_dir=save_dir,
        )

    # Config JSON
    config_dict = {
        "model_name": "SAITS",
        "parameters": {
            "n_layers":      2,
            "d_model":       64,
            "n_heads":       4,
            "d_ffn":         128,
            "dropout":       0.1,
            "seq_length":    baseline_cfg.seq_length,
            "batch_size":    baseline_cfg.batch_size,
            "learning_rate": baseline_cfg.learning_rate,
            "num_epochs":    baseline_cfg.num_epochs,
            "patience":      baseline_cfg.patience,
        },
        "scheduler": {"managed_by": "pypots.imputation.SAITS"},
        "complexity": {
            "training_time_seconds": round(training_time, 2),
        },
        "data": {
            "mask_ratio":            cfg.Masking.mask_ratio,
            "mask_seed":             cfg.Masking.random_seed,
            "data_quality_filtering": cfg.DataQuality.enable_filtering,
            "target_nan_ratio":      cfg.DataQuality.target_nan_ratio,
            "seed":                  seed,
        },
        "timestamp": datetime.now().isoformat(),
    }
    with open(save_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"   Saved: {save_dir / 'config.json'}")

    del saits_model
    if cfg.device == 'cuda':
        torch.cuda.empty_cache()

    return results_df


#
# STATISTICAL BASELINES
#

def run_statistical_baselines(test_graph):
    """IDW,  Linear interpolasyon baseline'larını çalıştır."""
    
    try:
        from M06_BaselineComparison import run_baseline_comparison
    except ImportError:
        print("   M06_BaselineComparison import edilemedi!")
        return
    
    print("\n" + "="*60)
    print(" İSTATİSTİKSEL BASELINE'LAR")
    print("="*60)
    
    start_time = time.time()

    global_df, level_df, predictions_per_method, shared_arrays = run_baseline_comparison(test_graph)

    total_time = time.time() - start_time

    scaling_stats = test_graph.get('scaling_stats', {})

    # Her method için ayrı kaydet — GNN ile tam parite (12 sutun + by_pressure + profile_consistency)
    if predictions_per_method:
        n_methods = len(predictions_per_method)
        for method, predictions in predictions_per_method.items():
            method_name = method.lower().replace(' ', '_')
            save_dir = ensure_results_dir(method_name)

            # Metrics — build_metrics_list_from_arrays uses same code path as GNN/DL/SAITS
            metrics_list = build_metrics_list_from_arrays(
                predictions=predictions,
                targets=shared_arrays['targets'],
                mask_applied=shared_arrays['mask'],
                pressures=shared_arrays['pressures'],
                scaling_stats=scaling_stats,
            )
            if metrics_list:
                save_metrics(metrics_list, save_dir, "metrics.csv")

            # by_pressure (M99 generic, GNN format). eval_mask sart (Bug #4 fix) —
            # statistical methodlari'nin mask disi noktalarda zaten ground truth
            # ile birebir esit olmasi by_pressure R²'sini yapay olarak sisirir.
            write_per_pressure_csv(
                predictions=predictions,
                targets=shared_arrays['targets'],
                pressures=shared_arrays['pressures'],
                test_graph=test_graph,
                save_dir=save_dir,
                eval_mask=shared_arrays['mask'],
            )

            # profile_consistency (yeni)
            write_profile_consistency_csv(
                predictions=predictions,
                targets=shared_arrays['targets'],
                pressures=shared_arrays['pressures'],
                stations=shared_arrays['stations'],
                datetimes=shared_arrays['datetimes'],
                save_dir=save_dir,
            )

            # Config
            config_dict = {
                "model_name": method,
                "parameters": {
                    "idw_power": cfg.Baseline.idw_power if 'idw' in method.lower() else None,
                },
                "complexity": {
                    "total_parameters": 0,
                    "training_time_seconds": 0,
                    "inference_time_seconds": round(total_time / n_methods, 2),
                },
                "data": {
                    "mask_ratio": cfg.Masking.mask_ratio,
                    "mask_seed": cfg.Masking.random_seed,
                    "data_quality_filtering": cfg.DataQuality.enable_filtering,
                    "target_nan_ratio": cfg.DataQuality.target_nan_ratio,
                },
                "timestamp": datetime.now().isoformat()
            }

            with open(save_dir / "config.json", 'w') as f:
                json.dump(config_dict, f, indent=2)
            print(f"Saved: {save_dir / 'config.json'}")


# 
# MAIN  
# 


def main():
    print("="*70)
    print("   RADIOSONDE VHT-GNN BENCHMARK")
    print("="*70)
    print(f" Device: {cfg.device}")
    print(f"Data: {os.path.basename(cfg.data_path)}")
    print(f" Results: {RESULTS_DIR.absolute()}")
    
    print(f"\nVeri Kalitesi Filtreleme: ...")
    print(f"\n    Veri Kalitesi Filtreleme: {'AÇIK     ' if cfg.DataQuality.enable_filtering else 'KAPALI'}")
    if cfg.DataQuality.target_nan_ratio is not None:
        print(f"   Hedef NaN oranı: {cfg.DataQuality.target_nan_ratio:.0%}")
    else:
        print(f"Hedef NaN oranı: Yok (sadece kalite eşiği kullanılacak)")
    
    # Hangi modeller çalışacak?
    print("\n Çalıştırılacak Modeller:")
    print(f"   GNN: vht_gnn={RUN_VHT_GNN},Vanilla={RUN_VANILLA_GRAPHSAGE}, Multiscale={RUN_MULTISCALE_GRAPHSAGE}, GAT={RUN_GAT}, MPNN={RUN_MPNN}")
    print(f"   DL:  LSTM={RUN_LSTM}, CNN={RUN_CNN}, MLP={RUN_MLP}")
    print(f"   Stat: {RUN_STATISTICAL}")
    
    # Hiçbir model seçilmediyse uyar
    any_model_selected = (
        RUN_VHT_GNN or RUN_VANILLA_GRAPHSAGE or RUN_MULTISCALE_GRAPHSAGE or RUN_GAT or RUN_MPNN
        or RUN_FLAT_GRAPHSAGE
        or RUN_LSTM or RUN_CNN or RUN_MLP or RUN_SAITS or RUN_STATISTICAL
        or RUN_VHT_GNN_NO_TEMPORAL or RUN_VHT_GNN_NO_GATING or RUN_VHT_GNN_FIXED_FUSION
        or RUN_VHT_GNN_NO_VERTICAL or RUN_VHT_GNN_NO_HORIZONTAL
        or RUN_VHT_GNN_GLOBAL_NORM or RUN_VANILLA_GRAPHSAGE_GLOBAL_NORM
    )
    
    if not any_model_selected:
        print("\n   Hiçbir model seçilmedi!")
        print("   Dosyanın başındaki RUN_XXX değişkenlerinden en az birini True yapın.")
        return
    
    # 
    # 1. VERİ YÜKLEME
    # 
    print("\n" + "-"*60)
    print("    1. VERİ YÜKLEME")
    print("-"*60)
    
    loader = RadiosondeLoader(
        stations_json=cfg.stations_path,
        filter_active=cfg.filter_active_stations,
        exclude_stations=cfg.exclude_stations
    )

    try:
        observations, stations = loader.load_dataset_from_csv(
            csv_path=cfg.data_path,
            start_date=cfg.start_date,
            end_date=cfg.end_date
        )
    except FileNotFoundError:
        print(" Veri dosyası bulunamadı!")
        return

    print(f"  {len(observations):,} gözlem yüklendi")
    print(f"  {len(stations)} istasyon")

    # 
    #      1.5 VERİ KALİTESİ FİLTRELEME (YENİ ADIM)
    # 
    if cfg.DataQuality.enable_filtering:
        print("\n" + "-"*60)
        print("    1.5 VERİ KALİTESİ FİLTRELEME")
        print("-"*60)
        
        observations = filter_data_by_time_step(
            df=observations,
            target_nan_ratio=cfg.DataQuality.target_nan_ratio,
            min_quality=cfg.DataQuality.min_time_step_quality,
            min_stations=cfg.DataQuality.min_stations_per_time,
            verbose=cfg.DataQuality.verbose
        )
        
        print(f"  Filtreleme sonrası: {len(observations):,} gözlem")

    # 
    # 2. TRAIN / TEST AYRIMI
    # 
    print("\n" + "-"*60)
    print("     2. TRAIN/TEST AYRIMI")
    print("-"*60)
    
    # Chronological 70/10/20 split with buffer gaps to mitigate autocorrelation
    gap = pd.Timedelta(days=cfg.chronological_gap_days)
    dates = observations['datetime'].unique()
    train_end_idx = int(len(dates) * 0.70)
    val_end_idx   = int(len(dates) * 0.80)
    train_split_date = dates[train_end_idx]
    val_split_date   = dates[val_end_idx]
    val_start_date   = train_split_date + gap
    test_start_date  = val_split_date + gap

    train_obs = observations[observations['datetime'] < train_split_date]
    val_obs   = observations[(observations['datetime'] >= val_start_date) &
                             (observations['datetime'] <  val_split_date)]
    test_obs  = observations[observations['datetime'] >= test_start_date]

    print(f"  Train: {len(train_obs):,} satır ({train_obs['datetime'].nunique():,} timestamp)")
    print(f"  Val:   {len(val_obs):,} satır ({val_obs['datetime'].nunique():,} timestamp)")
    print(f"  Test:  {len(test_obs):,} satır ({test_obs['datetime'].nunique():,} timestamp)")
    
    # 
    # 3. GRAFİK OLUŞTURMA
    # 
    print("\n" + "-"*60)
    print("     3. GRAFİK OLUŞTURMA")
    print("-"*60)
    
    builder = RadiosondeGraphBuilder(
        station_metadata=stations,
        temporal_window=cfg.graph_temporal_window,
        include_surface=False 
    )

    print("Building train graph...")
    train_graph = builder.build_graph_from_observations(train_obs)
    train_stats = train_graph['scaling_stats']

    print("Building val graph...")
    val_graph = builder.build_graph_from_observations(
        val_obs,
        external_stats=train_stats
    )

    print("Building test graph...")
    test_graph = builder.build_graph_from_observations(
        test_obs,
        external_stats=train_stats
    )

    train_samples = train_graph['x'].shape[0]
    val_samples   = val_graph['x'].shape[0]
    test_samples  = test_graph['x'].shape[0]

    print(f"  Train nodes: {train_samples:,}")
    print(f"  Val nodes:   {val_samples:,}")
    print(f"  Test nodes:  {test_samples:,}")

    # Global-norm graphs (only built if a *_GLOBAL_NORM ablation is requested).
    # These are separate graphs because level-aware normalization is applied
    # at graph-construction time, not at model time.
    needs_global_norm = RUN_VHT_GNN_GLOBAL_NORM or RUN_VANILLA_GRAPHSAGE_GLOBAL_NORM
    train_graph_gn = val_graph_gn = test_graph_gn = None
    train_samples_gn = val_samples_gn = test_samples_gn = 0
    if needs_global_norm:
        print("\nBuilding global-norm graphs (ablation: level-aware norm OFF)...")
        builder_gn = RadiosondeGraphBuilder(
            station_metadata=stations,
            temporal_window=cfg.graph_temporal_window,
            use_level_normalization=False,
            include_surface=False,
        )
        train_graph_gn = builder_gn.build_graph_from_observations(train_obs)
        train_stats_gn = train_graph_gn['scaling_stats']
        val_graph_gn   = builder_gn.build_graph_from_observations(val_obs,  external_stats=train_stats_gn)
        test_graph_gn  = builder_gn.build_graph_from_observations(test_obs, external_stats=train_stats_gn)
        train_samples_gn = train_graph_gn['x'].shape[0]
        val_samples_gn   = val_graph_gn['x'].shape[0]
        test_samples_gn  = test_graph_gn['x'].shape[0]
        print(f"  GN Train nodes: {train_samples_gn:,}")
        print(f"  GN Val nodes:   {val_samples_gn:,}")
        print(f"  GN Test nodes:  {test_samples_gn:,}")

    #
    # 4. MODELLERİ ÇALIŞTIR
    #
    seed_aware_models = (
        RUN_VHT_GNN or RUN_VANILLA_GRAPHSAGE or RUN_MULTISCALE_GRAPHSAGE
        or RUN_GAT or RUN_MPNN or RUN_FLAT_GRAPHSAGE
        or RUN_VHT_GNN_NO_TEMPORAL or RUN_VHT_GNN_NO_GATING
        or RUN_VHT_GNN_FIXED_FUSION
        or RUN_VHT_GNN_NO_VERTICAL or RUN_VHT_GNN_NO_HORIZONTAL
        or RUN_VHT_GNN_GLOBAL_NORM or RUN_VANILLA_GRAPHSAGE_GLOBAL_NORM
        or RUN_SAITS or RUN_LSTM or RUN_CNN or RUN_MLP
    )

    if seed_aware_models:
        print("\n" + "="*70)
        print(f" MODELLERİ ÇALIŞTIRMA  (seeds: {SEEDS})")
        print("="*70)

        for seed in SEEDS:
            if len(SEEDS) > 1:
                print("\n" + "#"*70)
                print(f"# SEED = {seed}")
                print("#"*70)

            # GNN Modelleri
            if RUN_VHT_GNN:
                run_vht_gnn(train_graph, test_graph, val_graph, train_samples, test_samples, seed=seed)
            if RUN_VANILLA_GRAPHSAGE:
                run_vanilla_graphsage(train_graph, test_graph, val_graph, train_samples, test_samples, seed=seed)
            if RUN_MULTISCALE_GRAPHSAGE:
                run_multiscale_graphsage(train_graph, test_graph, val_graph, train_samples, test_samples, seed=seed)
            if RUN_GAT:
                run_gat(train_graph, test_graph, val_graph, train_samples, test_samples, seed=seed)
            if RUN_MPNN:
                run_mpnn(train_graph, test_graph, val_graph, train_samples, test_samples, seed=seed)
            if RUN_FLAT_GRAPHSAGE:
                run_flat_graphsage(train_graph, test_graph, val_graph, train_samples, test_samples, seed=seed)

            # Ablation models (component-drop variants of VHT-GNN) — level-aware-norm graphs
            if RUN_VHT_GNN_NO_TEMPORAL:
                run_vht_gnn_no_temporal(train_graph, test_graph, val_graph, train_samples, test_samples, seed=seed)
            if RUN_VHT_GNN_NO_GATING:
                run_vht_gnn_no_gating(train_graph, test_graph, val_graph, train_samples, test_samples, seed=seed)
            if RUN_VHT_GNN_FIXED_FUSION:
                run_vht_gnn_fixed_fusion(train_graph, test_graph, val_graph, train_samples, test_samples, seed=seed)
            if RUN_VHT_GNN_NO_VERTICAL:
                run_vht_gnn_no_vertical(train_graph, test_graph, val_graph, train_samples, test_samples, seed=seed)
            if RUN_VHT_GNN_NO_HORIZONTAL:
                run_vht_gnn_no_horizontal(train_graph, test_graph, val_graph, train_samples, test_samples, seed=seed)

            # Ablation models on GLOBAL-NORM graphs (level-aware norm OFF)
            if RUN_VHT_GNN_GLOBAL_NORM:
                if train_graph_gn is None:
                    print("  [skip] vht_gnn_global_norm: global-norm graphs not built (set RUN_VHT_GNN_GLOBAL_NORM=True before main).")
                else:
                    run_vht_gnn_global_norm(train_graph_gn, test_graph_gn, val_graph_gn,
                                            train_samples_gn, test_samples_gn, seed=seed)
            if RUN_VANILLA_GRAPHSAGE_GLOBAL_NORM:
                if train_graph_gn is None:
                    print("  [skip] vanilla_graphsage_global_norm: global-norm graphs not built.")
                else:
                    run_vanilla_graphsage_global_norm(train_graph_gn, test_graph_gn, val_graph_gn,
                                                      train_samples_gn, test_samples_gn, seed=seed)

            # SAITS Transformer baseline (multi-seed destekli, GNN'lerle aynı pattern)
            if RUN_SAITS:
                run_saits(train_graph, test_graph, val_graph, train_samples, test_samples, seed=seed)

            if (RUN_LSTM or RUN_CNN or RUN_MLP):
                run_deep_learning_baselines(train_graph, test_graph, val_graph,
                                            RUN_LSTM, RUN_CNN, RUN_MLP, seed=seed)
    
    # İstatistiksel Baselines
    if RUN_STATISTICAL:
        run_statistical_baselines(test_graph)
    
    # 
    # 5. ÖZET
    # 
    print("\n" + "="*70)
    print("TAMAMLANDI")
    print("="*70)
    print(f"Sonuçlar kaydedildi: {RESULTS_DIR.absolute()}")
    print("\nKarşılaştırma tabloları için:")
    print("   python M98_CompareResults.py")


if __name__ == "__main__":
    import argparse

    MODEL_FLAG_MAP = {
        'vht_gnn':                       'RUN_VHT_GNN',
        'vanilla_graphsage':             'RUN_VANILLA_GRAPHSAGE',
        'multiscale_graphsage':          'RUN_MULTISCALE_GRAPHSAGE',
        'gat':                           'RUN_GAT',
        'mpnn':                          'RUN_MPNN',
        'flat_graphsage':                'RUN_FLAT_GRAPHSAGE',
        'vht_gnn_no_temporal':           'RUN_VHT_GNN_NO_TEMPORAL',
        'vht_gnn_no_gating':             'RUN_VHT_GNN_NO_GATING',
        'vht_gnn_fixed_fusion':          'RUN_VHT_GNN_FIXED_FUSION',
        'vht_gnn_no_vertical':           'RUN_VHT_GNN_NO_VERTICAL',
        'vht_gnn_no_horizontal':         'RUN_VHT_GNN_NO_HORIZONTAL',
        'vht_gnn_global_norm':           'RUN_VHT_GNN_GLOBAL_NORM',
        'vanilla_graphsage_global_norm': 'RUN_VANILLA_GRAPHSAGE_GLOBAL_NORM',
        'lstm':                          'RUN_LSTM',
        'cnn':                           'RUN_CNN',
        'mlp':                           'RUN_MLP',
        'saits':                         'RUN_SAITS',
        'statistical':                   'RUN_STATISTICAL',
    }

    CFG_ATTR_MAP = {
        'vht_gnn':                       'VHT_GNN',
        'vanilla_graphsage':             'VanillaGraphSAGE',
        'multiscale_graphsage':          'MultiscaleGraphSAGE',
        'gat':                           'Gat',
        'mpnn':                          'Mpnn',
        'flat_graphsage':                'FlatGraphSAGE',
        'vht_gnn_no_temporal':           'VHT_GNN_NoTemporal',
        'vht_gnn_no_gating':             'VHT_GNN_NoGating',
        'vht_gnn_fixed_fusion':          'VHT_GNN_FixedFusion',
        'vht_gnn_no_vertical':           'VHT_GNN_NoVertical',
        'vht_gnn_no_horizontal':         'VHT_GNN_NoHorizontal',
        'vht_gnn_global_norm':           'VHT_GNN_GlobalNorm',
        'vanilla_graphsage_global_norm': 'VanillaGraphSAGE_GlobalNorm',
        'lstm':                          'Baseline',
        'cnn':                           'Baseline',
        'mlp':                           'Baseline',
        'saits':                         'Baseline',
    }

    parser = argparse.ArgumentParser(
        description='Radiosonde GNN Benchmark - model/seed/epoch override',
        epilog='Örnek: python M99_MAIN.py --model vht_gnn --seed 42 --epochs 3'
    )
    parser.add_argument('--model', type=str, default=None,
                        choices=list(MODEL_FLAG_MAP.keys()),
                        help='Tek model çalıştır. Verilmezse file başındaki RUN_* bayrakları kullanılır.')
    parser.add_argument('--seed', type=int, nargs='+', default=None,
                        help='Seed override. Örn: --seed 42 veya --seed 42 123 456')
    parser.add_argument('--epochs', type=int, default=None,
                        help='num_epochs override (smoke test için kullanışlı)')
    args = parser.parse_args()

    if args.model is not None:
        for _flag in list(globals().keys()):
            if _flag.startswith('RUN_') and _flag.isupper() and isinstance(globals()[_flag], bool):
                globals()[_flag] = False
        globals()[MODEL_FLAG_MAP[args.model]] = True
        print(f"[CLI] Sadece '{args.model}' çalıştırılacak ({MODEL_FLAG_MAP[args.model]}=True)")

    if args.seed is not None:
        globals()['SEEDS'] = list(args.seed)
        print(f"[CLI] SEEDS override: {args.seed}")

    if args.epochs is not None:
        if args.model is not None and args.model in CFG_ATTR_MAP:
            getattr(cfg, CFG_ATTR_MAP[args.model]).num_epochs = args.epochs
        else:
            for _attr in set(CFG_ATTR_MAP.values()):
                if hasattr(cfg, _attr):
                    getattr(cfg, _attr).num_epochs = args.epochs
        print(f"[CLI] num_epochs override: {args.epochs}")

    main()
