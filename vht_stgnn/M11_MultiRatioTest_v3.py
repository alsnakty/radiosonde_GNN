# -*- coding: utf-8 -*-
"""
M11_MultiRatioTest_v3.py - Multi-Ratio Test 

M99_MAIN.py'deki AYNI evaluation  
   

Random veya Realistic masking seçilebilir
Seed desteği ile ADİL karşılaştırma
Tüm modeller için AYNI mask kullanılır
Ocak 2026
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm

# Proje modülleri
from M00_Config import cfg
from M02_DataLoading import (RadiosondeLoader, RadiosondeGraphBuilder,
                             RadiosondeSlidingWindowDataset, collate_graph_windows)
from M03_Model import RadiosondeSpatioTemporalGNN

# Mevcut importlara şunları ekle:
from M06_BaselineComparison import BaselineInterpolator
from M07_DeepLearningBaselines import LSTMImputer

# 
# AYARLAR
# 

MASK_RATIOS = [0.15, 0.30, 0.40, 0.50, 0.60]
MASK_SEED = cfg.Masking.random_seed  # 42

FEATURE_NAMES = ['temperature', 'relative_humidity', 'wind_speed', 
                 'sin_wd', 'cos_wd', 'geopotential']
FEATURE_UNITS = {
    'temperature': '°C',
    'relative_humidity': '%',
    'wind_speed': 'm/s',
    'sin_wd': '-',
    'cos_wd': '-',
    'geopotential': 'm'
}

#  
# M99'DAN KOPYALANAN FONKSİYONLAR (BİREBİR AYNI)
#  

def get_level_std(pressure: float, level_stats: dict, feature_idx: int) -> float:
    """Belirli bir basınç seviyesi için feature'ın std değerini döndürür."""
    level_key = f"p_{int(pressure)}"
    
    if level_key in level_stats:
        stds = level_stats[level_key].get('stds', None)
        if stds is not None:
            if isinstance(stds, np.ndarray):
                return float(stds[feature_idx])
            elif isinstance(stds, list):
                return float(stds[feature_idx])
    
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


def evaluate_gnn_at_ratio(model, test_graph, model_cfg, device, mask_ratio, 
                          use_realistic_masking=False, seed=None):
    """
    GNN modelini belirli mask ratio ile değerlendir.
    
    Parameters:
    
    """
    
    model.eval()
    
    window_size = model_cfg.dataset_window_size
    batch_size = model_cfg.batch_size
    
    #  GÜNCELLEME: seed ve use_realistic_masking parametreleri
    test_ds = RadiosondeSlidingWindowDataset(
        test_graph, 
        window_size, 
        mask_ratio=mask_ratio,
        use_realistic_masking=use_realistic_masking,
        seed=seed  # ← ADİL KARŞILAŞTIRMA
    )
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                         collate_fn=collate_graph_windows, num_workers=0)
    
    all_inputs = []
    all_preds = []
    all_targets = []
    all_pressures = []
    all_stations = []
    all_datetimes = []

    with torch.no_grad():
        for batch in tqdm(test_dl, desc=f"Eval {int(mask_ratio*100)}%", leave=False):
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

    input_cat  = torch.cat(all_inputs,  0).numpy()
    pred_cat   = torch.cat(all_preds,   0).numpy()
    target_cat = torch.cat(all_targets, 0).numpy()
    pressures_cat = np.array(all_pressures) if all_pressures else None
    stations_cat  = np.array(all_stations) if all_stations else None
    datetimes_cat = np.array(all_datetimes) if all_datetimes else None

    # Sliding window overlap dedup (Bug #5 fix) — LSTM/SAITS pariate
    from M99_MAIN import dedup_sliding_window_predictions
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

    # Imputation hedefleri = modele NaN olarak verilen ama target'ta dolu noktalar.
    # LSTM tarafindaki random %15 mask ile parite.
    eval_mask_all = np.isnan(input_cat) & ~np.isnan(target_cat)

    results = {}

    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        mask = eval_mask_all[:, feat_idx]
        if mask.sum() < 10:
            results[feat_name] = {'r2': np.nan, 'mae': np.nan, 'mae_real': np.nan, 'n': 0}
            continue
        
        y_true = target_cat[mask, feat_idx]
        y_pred = pred_cat[mask, feat_idx]
        
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        if valid.sum() < 10:
            results[feat_name] = {'r2': np.nan, 'mae': np.nan, 'mae_real': np.nan, 'n': 0}
            continue
        
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        
        # RH: stratosferik seviyeleri çıkar (< 200 hPa)
        if feat_name == 'relative_humidity' and pressures_cat is not None:
            p_valid = pressures_cat[mask][valid]
            rh_mask = (p_valid >= 200.0)
            y_true = y_true[rh_mask]
            y_pred = y_pred[rh_mask]
            if len(y_true) < 10:
                results[feat_name] = {'r2': np.nan, 'mae': np.nan, 'mae_real': np.nan, 'n': 0}
                continue
        
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        if pressures_cat is not None and len(level_stats) > 0:
            p_valid = pressures_cat[mask][valid]
            # RH için filtrelenmiş pressure kullan
            if feat_name == 'relative_humidity':
                rh_p_mask = (p_valid >= 200.0)
                p_valid = p_valid[rh_p_mask]
            mae_real = calculate_real_mae(y_true, y_pred, p_valid, level_stats, feat_idx)
        else:
            global_stds = scaling_stats.get('stds', np.ones(6))
            if isinstance(global_stds, np.ndarray):
                mae_real = mae * global_stds[feat_idx]
            else:
                mae_real = mae
        
        results[feat_name] = {
            'r2': round(r2, 4),
            'mae': round(mae, 4),
            'mae_real': round(mae_real, 4),
            'n': len(y_true)
        }
    
    return results



# 
 

def evaluate_lstm_at_ratio(model, test_graph, device, mask_ratio, seed=42):
    """
    LSTM modelini belirli mask ratio ile değerlendir.
    Model dışarıdan yüklenmiş olarak gelir.
    """
    
    model.eval()
    
    # 1. Test verisini al
    test_x = test_graph['x'].numpy() if isinstance(test_graph['x'], torch.Tensor) else np.array(test_graph['x'])
    original = test_x.copy()
    metadata = test_graph['node_metadata']
    pressures = np.array(metadata['pressure'])
    station_ids = np.array(metadata['station_id'])
    
    # 2. Mask oluştur (seed ile)
    np.random.seed(seed)
    mask = (np.random.rand(*test_x.shape) < mask_ratio).astype(bool)
    mask = mask & ~np.isnan(test_x)
    
    # 3. Masked input
    test_x_masked = test_x.copy()
    test_x_masked[mask] = 0.0
    
    # 4. LSTM prediction (station-level sequence bazlı)
    predictions = np.zeros_like(test_x)
    pred_count = np.zeros(len(test_x))
    
    unique_stations = np.unique(station_ids)
    unique_pressures = np.unique(pressures)
    seq_length = cfg.Baseline.seq_length  # 5
    step_size = max(1, seq_length // 2)
    
    for station in unique_stations:
        for pressure in unique_pressures:
            idx_mask = (station_ids == station) & (np.abs(pressures - pressure) < 1)
            indices = np.where(idx_mask)[0]
            
            if len(indices) < seq_length:
                # train-normalized mean is 0 in the level-aware z-scored space
                predictions[indices] = 0.0
                pred_count[indices] = 1
                continue
            
            seq_data_masked = test_x_masked[indices].copy()
            
            for i in range(0, len(indices), step_size):
                end_i = min(i + seq_length, len(indices))
                if end_i - i < seq_length:
                    start_i = max(0, end_i - seq_length)
                else:
                    start_i = i
                
                window_indices = indices[start_i:start_i + seq_length]
                window_data = seq_data_masked[start_i:start_i + seq_length]
                
                if len(window_data) < seq_length:
                    continue
                
                seq_tensor = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred = model(seq_tensor).squeeze(0).cpu().numpy()
                
                for j, idx in enumerate(window_indices):
                    if pred_count[idx] == 0:
                        predictions[idx] = pred[j]
                    else:
                        predictions[idx] = (predictions[idx] * pred_count[idx] + pred[j]) / (pred_count[idx] + 1)
                    pred_count[idx] += 1
    
    # train-normalized mean is 0 in the level-aware z-scored space
    zero_count = pred_count == 0
    if zero_count.any():
        predictions[zero_count] = 0.0
    
    # 5. Metrik hesapla
    scaling_stats = test_graph.get('scaling_stats', {})
    level_stats = scaling_stats.get('level_stats', {})
    results = {}
    
    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        feat_mask = mask[:, feat_idx]
        
        # RH: stratosferik seviyeleri çıkar
        if feat_name == 'relative_humidity':
            feat_mask = feat_mask & (pressures >= 200.0)
        
        if feat_mask.sum() < 10:
            results[feat_name] = {'r2': np.nan, 'mae': np.nan, 'mae_real': np.nan, 'n': 0}
            continue
        
        y_true = original[feat_mask, feat_idx]
        y_pred = predictions[feat_mask, feat_idx]
        
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        if valid.sum() < 10:
            results[feat_name] = {'r2': np.nan, 'mae': np.nan, 'mae_real': np.nan, 'n': 0}
            continue
        
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        p_valid = pressures[feat_mask][valid]
        
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        if len(level_stats) > 0:
            mae_real = calculate_real_mae(y_true, y_pred, p_valid, level_stats, feat_idx)
        else:
            global_stds = scaling_stats.get('stds', np.ones(6))
            mae_real = mae * (global_stds[feat_idx] if isinstance(global_stds, np.ndarray) else 1.0)
        
        results[feat_name] = {
            'r2': round(r2, 4),
            'mae': round(mae, 4),
            'mae_real': round(mae_real, 4),
            'n': len(y_true)
        }
    
    return results


def evaluate_linear_spatial_at_ratio(test_graph, mask_ratio, seed=42):
    """
    Linear (Spatial) interpolasyonu belirli mask ratio ile değerlendir.
    """
    
    interpolator = BaselineInterpolator(test_graph, test_graph.get('scaling_stats'))
    original = interpolator.x.copy()
    
    # Mask oluştur (seed ile)
    np.random.seed(seed)
    mask = (np.random.rand(*interpolator.x.shape) < mask_ratio).astype(bool)
    mask = mask & ~np.isnan(interpolator.x)
    
    # Linear Spatial imputation
    imputed = interpolator.linear_spatial_impute(mask)
    
    # Metrik hesapla (M06'nın kendi evaluate'i RH filtresini zaten uyguluyor)
    results = {}
    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        metrics = interpolator.evaluate(original, imputed, mask, feat_idx)
        results[feat_name] = {
            'r2': round(metrics['r2'], 4) if not np.isnan(metrics['r2']) else np.nan,
            'mae': round(metrics['mae'], 4) if not np.isnan(metrics['mae']) else np.nan,
            'mae_real': round(metrics['mae_real'], 4) if not np.isnan(metrics['mae_real']) else np.nan,
            'n': metrics['n']
        }
    
    return results

# 
# ANA FONKSİYON
# 

def _find_model_pt(model_dir, checkpoint_seed):
    """Multi-seed layout first (results/<model>/seed_<N>/model.pt),
    flat path fallback (results/<model>/model.pt). Geri uyumlu."""
    candidates = [
        model_dir / f"seed_{checkpoint_seed}" / "model.pt",
        model_dir / "model.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"model.pt bulunamadi. Denenen yollar: {[str(c) for c in candidates]}"
    )


def run_multi_ratio_test(gnn_models=None,
                         mask_type='random', seed=None,
                         include_lstm=True, include_linear_spatial=True,
                         checkpoint_seed=42):
    """
    Multi-ratio test - GNN + LSTM + Linear (Spatial).

    Ornek: run_multi_ratio_test(gnn_models=['vht_gnn', 'vanilla_graphsage'])
     4 model: VHT-GNN, Multi-Relational GraphSAGE, LSTM, Linear (Spatial)

    checkpoint_seed: M99 multi-seed layout'unda hangi seed klasorunden model.pt
    yukleneceği. Default 42. Flat path (results/<model>/model.pt) hala varsa
    o da kullanilabilir (fallback)."""

    if gnn_models is None:
        gnn_models = ['vht_gnn']

    if seed is None:
        seed = MASK_SEED
    
    print("="*70)
    print(f" MULTI-RATIO TEST ({len(gnn_models)} GNN + LSTM + Linear Spatial)")
    print("="*70)
    print(f"GNN Models: {[m.upper() for m in gnn_models]}")
    print(f"Seed: {seed}")
    print(f"Test oranları: {[f'{r*100:.0f}%' for r in MASK_RATIOS]}")
    
    device = cfg.device
    
    # 1. VERİ YÜKLEME
    print("\n" + "-"*60)
    print(" 1. VERİ YÜKLEME")
    print("-"*60)
    
    loader = RadiosondeLoader(
        stations_json=cfg.stations_path,
        filter_active=cfg.filter_active_stations,
        exclude_stations=cfg.exclude_stations
    )
    
    observations, stations = loader.load_dataset_from_csv(
        csv_path=cfg.data_path,
        start_date=cfg.start_date,
        end_date=cfg.end_date
    )
    
    # 70/10/20 split with buffer gap (parity with M99); val skipped, only train + test needed
    gap = pd.Timedelta(days=cfg.chronological_gap_days)
    dates = observations['datetime'].unique()
    train_end_idx = int(len(dates) * 0.70)
    val_end_idx   = int(len(dates) * 0.80)
    train_split_date = dates[train_end_idx]
    val_split_date   = dates[val_end_idx]
    test_start_date  = val_split_date + gap

    train_obs = observations[observations['datetime'] <  train_split_date]
    test_obs  = observations[observations['datetime'] >= test_start_date]
    
    # 2. GRAPH OLUŞTURMA
    print("\n" + "-"*60)
    print(" 2. GRAPH OLUŞTURMA")
    print("-"*60)
    
    builder = RadiosondeGraphBuilder(
        station_metadata=stations,
        temporal_window=cfg.graph_temporal_window,
        include_surface=False
    )
    
    train_graph = builder.build_graph_from_observations(train_obs)
    test_graph = builder.build_graph_from_observations(
        test_obs,
        external_stats=train_graph['scaling_stats']
    )
    
    # 3. MODELLERİ YÜKLEME (hepsini bir kez yükle)
    print("\n" + "-"*60)
    print(" 3. MODEL YÜKLEME")
    print("-"*60)
    
    # --- GNN modelleri ---
    loaded_gnns = {}
    for gnn_name in gnn_models:
        if gnn_name == 'vht_gnn':
            mcfg = cfg.VHT_GNN
        elif gnn_name == 'vanilla_graphsage':
            mcfg = cfg.VanillaGraphSAGE
        elif gnn_name == 'gat':
            mcfg = cfg.Gat
        elif gnn_name == 'mpnn':
            mcfg = cfg.Mpnn
        elif gnn_name == 'multiscale_graphsage':
            mcfg = cfg.MultiscaleGraphSAGE
        else:
            mcfg = cfg.VHT_GNN
        
        m = RadiosondeSpatioTemporalGNN(
            input_dim=cfg.input_dim,
            hidden_dim=mcfg.hidden_dim,
            num_gnn_layers=mcfg.num_gnn_layers,
            dropout=mcfg.dropout,
            model_type=gnn_name
        )
        p = _find_model_pt(Path("results") / gnn_name, checkpoint_seed)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        m.to(device)
        m.eval()
        print(f"  [OK] {gnn_name.upper()} yüklendi: {p}")

        loaded_gnns[gnn_name] = (m, mcfg)

    # --- LSTM modeli (bir kez yukle) ---
    lstm_model = None
    if include_lstm:
        try:
            lstm_model_path = _find_model_pt(Path("results") / "lstm", checkpoint_seed)
        except FileNotFoundError:
            lstm_model_path = Path("results") / "lstm" / "model.pt"  # raise edecek asagida
        if lstm_model_path.exists():
            lstm_model = LSTMImputer(
                input_dim=6,
                hidden_dim=cfg.Baseline.lstm_hidden_dim,
                num_layers=2
            )
            lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=device, weights_only=True))
            lstm_model.to(device)
            lstm_model.eval()
            print(f"  [OK] LSTM yüklendi: {lstm_model_path}")
        else:
            print(f"  [WARNING] LSTM model bulunamadı: {lstm_model_path}")
            include_lstm = False
    
    # 4. MULTI-RATIO TEST
    print("\n" + "-"*60)
    print(" 4. MULTI-RATIO TEST")
    print("-"*60)
    
    all_results = []
    
    for mask_ratio in MASK_RATIOS:
        print(f"\n{'='*50}")
        print(f" Mask Ratio: {mask_ratio*100:.0f}%")
        print(f"{'='*50}")
        
        # --- GNN'ler ---
        for gnn_name, (gnn_model, gnn_cfg) in loaded_gnns.items():
            print(f"\n  [{gnn_name.upper()}]")
            use_realistic = (mask_type == 'realistic')
            gnn_results = evaluate_gnn_at_ratio(
                gnn_model, test_graph, gnn_cfg, device, mask_ratio,
                use_realistic_masking=use_realistic, seed=seed
            )
            for feat_name in FEATURE_NAMES:
                r = gnn_results[feat_name]
                all_results.append({
                    'model': gnn_name.upper(),
                    'mask_ratio': mask_ratio,
                    'feature': feat_name,
                    'r2': r['r2'], 'mae': r['mae'], 'mae_real': r['mae_real'], 'n': r['n']
                })
            avg_r2 = np.mean([r['r2'] for r in gnn_results.values() if not np.isnan(r['r2'])])
            print(f"    Avg R²: {avg_r2:.4f}")
        
        # --- LSTM ---
        if include_lstm and lstm_model is not None:
            print(f"\n  [LSTM]")
            lstm_results = evaluate_lstm_at_ratio(
                lstm_model, test_graph, device, mask_ratio, seed=seed
            )
            if lstm_results:
                for feat_name in FEATURE_NAMES:
                    r = lstm_results[feat_name]
                    all_results.append({
                        'model': 'LSTM',
                        'mask_ratio': mask_ratio,
                        'feature': feat_name,
                        'r2': r['r2'], 'mae': r['mae'], 'mae_real': r['mae_real'], 'n': r['n']
                    })
                avg_r2 = np.mean([r['r2'] for r in lstm_results.values() if not np.isnan(r['r2'])])
                print(f"    Avg R²: {avg_r2:.4f}")
        
        # --- Linear (Spatial) ---
        if include_linear_spatial:
            print(f"\n  [Linear (Spatial)]")
            lin_results = evaluate_linear_spatial_at_ratio(
                test_graph, mask_ratio, seed=seed
            )
            if lin_results:
                for feat_name in FEATURE_NAMES:
                    r = lin_results[feat_name]
                    all_results.append({
                        'model': 'Linear_Spatial',
                        'mask_ratio': mask_ratio,
                        'feature': feat_name,
                        'r2': r['r2'], 'mae': r['mae'], 'mae_real': r['mae_real'], 'n': r['n']
                    })
                avg_r2 = np.mean([r['r2'] for r in lin_results.values() if not np.isnan(r['r2'])])
                print(f"    Avg R²: {avg_r2:.4f}")
    
    # 5. ÖZET TABLO
    print("\n" + "="*70)
    print(" ÖZET: Avg R² by Model and Mask Ratio")
    print("="*70)
    
    results_df = pd.DataFrame(all_results)
    
    pivot = results_df.pivot_table(
        index='model',
        columns='mask_ratio',
        values='r2',
        aggfunc='mean'
    )
    pivot.columns = [f"{int(c*100)}%" for c in pivot.columns]
    
    print(pivot.round(4).to_string())
    
    # Kaydet
    output_dir = Path("results") / "multi_ratio"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_dir / "multi_ratio_all_models.csv", index=False)
    pivot.to_csv(output_dir / "multi_ratio_summary.csv")
    
    print(f"\n  Saved: {output_dir / 'multi_ratio_all_models.csv'}")
    print(f"  Saved: {output_dir / 'multi_ratio_summary.csv'}")
    
    return results_df, pivot

def run_both_mask_types(gnn_models=None, seed=None):
    """
    Hem random hem realistic mask ile test yap ve karşılaştır.
    """
    
    print("\n" + "="*70)
    print(" RANDOM vs REALISTIC MASK KARŞILAŞTIRMASI")
    print("="*70)
    
    # Random mask testi
    print("\n" + " "*20)
    random_df, random_pivot = run_multi_ratio_test(
        gnn_models=gnn_models, mask_type='random', seed=seed
    )
    print("\n" + " "*20)
    realistic_df, realistic_pivot = run_multi_ratio_test(
        gnn_models=gnn_models, mask_type='realistic', seed=seed
    )
    
    
    
    # Karşılaştırma
    if random_pivot is not None and realistic_pivot is not None:
        print("\n" + "="*70)
        print(" KARŞILAŞTIRMA: Random vs Realistic")
        print("="*70)
        
        comparison = pd.DataFrame({
            'Random': random_pivot.mean(),
            'Realistic': realistic_pivot.mean(),
            'Difference': random_pivot.mean() - realistic_pivot.mean()
        })
        
        print("\nOrtalama R² (her mask ratio için):")
        print(comparison.round(4).to_string())
        
        # Kaydet
        output_dir = Path("results") / "multi_ratio"
        comparison.to_csv(output_dir / "mask_type_comparison.csv")
        print(f"\n Saved: {output_dir / 'mask_type_comparison.csv'}")
    
    return random_df, realistic_df


#
# MULTI-SEED CANONICAL RUNNER (results_canonical, mean+-std over seeds)
#
import json as _json
import types as _types

SEEDS_CANON = [42, 123, 456, 789, 2024]
RESULTS_DIR_CANON = Path("results_canonical")


def _load_gnn_canon(model_name, seed_dir, device):
    """Build a GNN from its saved config.json and load the seed checkpoint."""
    params = _json.load(open(seed_dir / "config.json")).get('parameters', {})
    kwargs = dict(input_dim=cfg.input_dim, hidden_dim=params.get('hidden_dim', 64),
                  num_gnn_layers=params.get('num_gnn_layers', 1),
                  dropout=params.get('dropout', 0.1), model_type=model_name)
    if model_name == 'gat':
        kwargs['heads'] = params.get('heads', cfg.Gat.heads)
    m = RadiosondeSpatioTemporalGNN(**kwargs)
    m.load_state_dict(torch.load(seed_dir / "model.pt", map_location=device, weights_only=True))
    m.to(device); m.eval()
    return m, params.get('batch_size', 16)


def _avg_r2(results):
    vals = [r['r2'] for r in results.values()
            if r['r2'] is not None and not np.isnan(r['r2'])]
    return float(np.mean(vals)) if vals else np.nan


def run_multiseed_ratio_test(gnn_models, include_lstm=True):
    """Sensitivity to mask ratio over five seeds (results_canonical).
    Fair comparison: the artificial mask is fixed (MASK_SEED) across models and
    ratios; only the trained checkpoint (seed) varies. Saves a tidy CSV with the
    seed mean/std of the across-variable average R^2 per (model, ratio)."""
    device = cfg.device
    is_cuda = isinstance(device, str) and device.startswith('cuda')

    print("Loading data and building graph (once)...")
    loader = RadiosondeLoader(stations_json=cfg.stations_path,
                              filter_active=cfg.filter_active_stations,
                              exclude_stations=cfg.exclude_stations)
    observations, stations = loader.load_dataset_from_csv(
        csv_path=cfg.data_path, start_date=cfg.start_date, end_date=cfg.end_date)
    gap = pd.Timedelta(days=cfg.chronological_gap_days)
    dates = observations['datetime'].unique()
    train_split = dates[int(len(dates) * 0.70)]
    val_split = dates[int(len(dates) * 0.80)]
    test_start = val_split + gap
    train_obs = observations[observations['datetime'] < train_split]
    test_obs = observations[observations['datetime'] >= test_start]
    builder = RadiosondeGraphBuilder(station_metadata=stations,
                                     temporal_window=cfg.graph_temporal_window,
                                     include_surface=False)
    train_graph = builder.build_graph_from_observations(train_obs)
    test_graph = builder.build_graph_from_observations(
        test_obs, external_stats=train_graph['scaling_stats'])

    window = cfg.VHT_GNN.dataset_window_size
    rows = []
    for ratio in MASK_RATIOS:
        for gnn in gnn_models:
            avgs = []
            for s in SEEDS_CANON:
                sd = RESULTS_DIR_CANON / gnn / f"seed_{s}"
                if not (sd / "model.pt").exists():
                    continue
                m, bs = _load_gnn_canon(gnn, sd, device)
                mcfg = _types.SimpleNamespace(dataset_window_size=window, batch_size=bs)
                res = evaluate_gnn_at_ratio(m, test_graph, mcfg, device, ratio,
                                            use_realistic_masking=False, seed=MASK_SEED)
                avgs.append(_avg_r2(res))
                del m
                if is_cuda:
                    torch.cuda.empty_cache()
            arr = np.array([v for v in avgs if not np.isnan(v)])
            mean = float(arr.mean()); std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            rows.append({'model': gnn, 'mask_ratio': ratio, 'n_seeds': len(arr),
                         'mean_r2': mean, 'std_r2': std})
            print(f"  {int(ratio*100):>2}%  {gnn:<16} {mean:.4f} +/- {std:.4f}  ({len(arr)} seeds)")
        if include_lstm:
            avgs = []
            for s in SEEDS_CANON:
                sd = RESULTS_DIR_CANON / "lstm" / f"seed_{s}"
                if not (sd / "model.pt").exists():
                    continue
                lstm = LSTMImputer(input_dim=6, hidden_dim=cfg.Baseline.lstm_hidden_dim,
                                   num_layers=2)
                lstm.load_state_dict(torch.load(sd / "model.pt", map_location=device,
                                                weights_only=True))
                lstm.to(device); lstm.eval()
                res = evaluate_lstm_at_ratio(lstm, test_graph, device, ratio, seed=MASK_SEED)
                avgs.append(_avg_r2(res))
                del lstm
                if is_cuda:
                    torch.cuda.empty_cache()
            arr = np.array([v for v in avgs if not np.isnan(v)])
            mean = float(arr.mean()); std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            rows.append({'model': 'lstm', 'mask_ratio': ratio, 'n_seeds': len(arr),
                         'mean_r2': mean, 'std_r2': std})
            print(f"  {int(ratio*100):>2}%  {'lstm':<16} {mean:.4f} +/- {std:.4f}  ({len(arr)} seeds)")

    out_dir = Path("comparison_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "multi_ratio_canonical.csv", index=False)
    print(f"\nSaved: {out_dir / 'multi_ratio_canonical.csv'}")
    return df


if __name__ == "__main__":
    run_multiseed_ratio_test(
        gnn_models=['vht_gnn', 'flat_graphsage'],
        include_lstm=True,
    )
