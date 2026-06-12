# -*- coding: utf-8 -*-
"""
M07_DeepLearningBaselines.py
- LSTM:  
- CNN:  
- MLP: 

"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
from pathlib import Path

# Config import
from M00_Config import cfg
from M01_Utils import load_station_metadata
from M02_DataLoading import RadiosondeLoader, RadiosondeGraphBuilder

# Masking sabitleri (karşılaştırma için)
MASK_SEED = cfg.Masking.random_seed    # 42
MASK_RATIO = cfg.Masking.mask_ratio    # 0.15

# Feature birimleri (Real MAE için)
FEATURE_UNITS = {
    'temperature': '°C',
    'relative_humidity': '%',
    'wind_speed': 'm/s',
    'sin_wd': '-',
    'cos_wd': '-',
    'geopotential': 'm'
}

# DATASET FOR DEEP LEARNING BASELINES

class RadiosondeSequenceDataset(Dataset):
    """
    LSTM/CNN/MLP için sequence dataset.
    Her istasyon-seviye kombinasyonu için zaman serisi oluşturur.
    """
    
    def __init__(self, graph_data, seq_length=5, mask_ratio=0.15, mode='train'):
        """
        Args:
            graph_data: RadiosondeGraphBuilder çıktısı
            seq_length: LSTM/CNN için sequence uzunluğu
            mask_ratio: Eksik veri oranı
            mode: 'train' veya 'test'
        """
        self.seq_length = seq_length
        self.mask_ratio = mask_ratio
        self.mode = mode
        
        # Graph verisini çıkar
        self.x = graph_data['x'].numpy() if isinstance(graph_data['x'], torch.Tensor) else graph_data['x']
        self.metadata = graph_data['node_metadata']
        self.scaling_stats = graph_data.get('scaling_stats', {})
        
        # Metadata arrays
        self.station_ids = np.array(self.metadata['station_id'])
        self.pressures = np.array(self.metadata['pressure'])
        self.datetimes = self.metadata['datetime']
        
        # Unique değerler
        self.unique_stations = np.unique(self.station_ids)
        self.unique_pressures = np.unique(self.pressures)
        self.unique_times = sorted(set(self.datetimes))
        
        # Time index mapping
        self.time_to_idx = {t: i for i, t in enumerate(self.unique_times)}
        
        # Sequences oluştur
        self.sequences = self._build_sequences()
        
        print(f"{mode.upper()} Dataset: {len(self.sequences)} sequences")
        print(f"Stations: {len(self.unique_stations)}, Levels: {len(self.unique_pressures)}")
        print(f"Time steps: {len(self.unique_times)}, Seq length: {seq_length}")
    
    def _build_sequences(self):
        """Her istasyon-seviye için zaman serisi sequences oluştur."""
        sequences = []
        
        num_times = len(self.unique_times)
        
        for station in self.unique_stations:
            for pressure in self.unique_pressures:
                # Bu station-pressure kombinasyonu için tüm zamanları bul
                mask = (self.station_ids == station) & (np.abs(self.pressures - pressure) < 1)
                indices = np.where(mask)[0]
                
                if len(indices) < self.seq_length:
                    continue
                
                # Zamana göre sırala
                time_indices = np.array([self.time_to_idx[self.datetimes[i]] for i in indices])
                sort_order = np.argsort(time_indices)
                indices = indices[sort_order]
                
                # Sliding window sequences
                for i in range(len(indices) - self.seq_length + 1):
                    seq_indices = indices[i:i + self.seq_length]
                    sequences.append({
                        'indices': seq_indices,
                        'station': station,
                        'pressure': pressure
                    })
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    


    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        indices = seq_info['indices']
        
        seq_data = self.x[indices].copy()
        target = seq_data.copy()
        
        # DÜZELTME: Hem train hem testte maskeleme yapıyoruz. 
        # Testte her zaman aynı maskeyi üretmek için idx'i seed olarak kullanıyoruz.
        if self.mode == 'test':
            np.random.seed(idx)
            
        mask = np.random.rand(*seq_data.shape) < self.mask_ratio
        mask = mask | np.isnan(seq_data) # Zaten NaN olanları da maskeye ekle
        
        seq_data[mask] = 0.0  # Modelin görmemesi gereken yerleri sıfırladık
        
        return {
            'x': torch.tensor(seq_data, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.bool),   
            'station': seq_info['station'],
            'pressure': seq_info['pressure']
        }

class RadiosondeMLPDataset(Dataset):
    """
    MLP için node-level dataset.
    Her node bağımsız olarak işlenir (spatial/temporal bilgi yok).
    """
    
    def __init__(self, graph_data, mask_ratio=0.15, mode='train'):
        self.mask_ratio = mask_ratio
        self.mode = mode
        
        self.x = graph_data['x'].numpy() if isinstance(graph_data['x'], torch.Tensor) else graph_data['x']
        self.metadata = graph_data['node_metadata']
        
        # Positional features ekle
        self.pos_features = self._build_positional_features()
        
        print(f"{mode.upper()} MLP Dataset: {len(self.x)} samples")
    
    def _build_positional_features(self):
        """Pozisyonel özellikler: pressure, lat, lon, time."""
        pressure = np.array(self.metadata['pressure']) / 1000.0
        lat = np.array(self.metadata['lat']) / 90.0
        lon = np.array(self.metadata['lon']) / 180.0
        
        # Time features
        datetimes = self.metadata['datetime']
        hours = np.array([dt.hour for dt in datetimes])
        time_sin = np.sin(2 * np.pi * hours / 24)
        time_cos = np.cos(2 * np.pi * hours / 24)
        
        return np.column_stack([pressure, lat, lon, time_sin, time_cos])
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x_orig = self.x[idx].copy()
        target = x_orig.copy()
        pos = self.pos_features[idx]
        
        # Test modunda seed sabitleme ve her durumda maskeleme
        if self.mode == 'test':
            np.random.seed(idx)
            
        mask = np.random.rand(len(x_orig)) < self.mask_ratio
        mask = mask | np.isnan(x_orig)
        
        x_masked = x_orig.copy()
        x_masked[mask] = 0.0 # Giriş verisini maskeledik
        
        x_with_pos = np.concatenate([x_masked, pos]) # Maskeli veri + pozisyon
        
        return {
            'x': torch.tensor(x_with_pos, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.bool)
        }   

# MODEL DEFINITIONS 

class LSTMImputer(nn.Module):
    """
    LSTM-based imputation model.
    
    Makaledeki argüman: "Standard LSTMs typically model each station 
    independently, ignoring spatial correlations"
    
    Bu model her istasyon-seviye kombinasyonunu bağımsız işler.
    """
    
    def __init__(self, 
                 input_dim=6, 
                 hidden_dim=128, 
                 num_layers=2, 
                 dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_length, input_dim)
            mask: (batch, seq_length, input_dim) - True = missing
        """
        batch_size, seq_len, _ = x.shape
        
        # Handle NaN
        x = torch.nan_to_num(x, nan=0.0)
        
        # Project input
        h = self.input_proj(x)  # (batch, seq, hidden)
        
        # LSTM
        lstm_out, _ = self.lstm(h)  # (batch, seq, hidden*2)
        
        # Output
        output = self.output_proj(lstm_out)  # (batch, seq, input_dim)
        
        return output


class CNNImputer(nn.Module):
    """
    1D CNN-based imputation model.
    
    Makaledeki argüman: "CNN-based methods require regular grid structures 
    that are incompatible with the sparse and irregular distribution of 
    radiosonde stations"
    
    Bu model sadece temporal 1D convolution yapar, spatial yapı yok.
    """
    
    def __init__(self, 
                 input_dim=6, 
                 hidden_dim=128, 
                 num_layers=4, 
                 kernel_size=3, 
                 dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        
        # Dilated causal convolutions
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size, 
                             padding=padding, dilation=dilation),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        
        # Output projection
        self.output_proj = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_length, input_dim)
        """
        # Handle NaN
        x = torch.nan_to_num(x, nan=0.0)
        
        # (batch, seq, features) -> (batch, features, seq)
        x = x.permute(0, 2, 1)
        
        # Input projection
        h = self.input_proj(x)
        
        # Conv layers with residual
        for conv in self.conv_layers:
            h = h + conv(h)
        
        # Output
        output = self.output_proj(h)
        
        # (batch, features, seq) -> (batch, seq, features)
        output = output.permute(0, 2, 1)
        
        return output


class MLPImputer(nn.Module):
    """
    MLP-based imputation model.
    
    En basit neural baseline - her node bağımsız, sadece positional 
    features ile zenginleştirilmiş.
    """
    
    def __init__(self, 
                 input_dim=6, 
                 pos_dim=5, 
                 hidden_dim=256, 
                 num_layers=4, 
                 dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        total_input = input_dim + pos_dim
        
        layers = []
        layers.append(nn.Linear(total_input, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, input_dim + pos_dim)
        """
        x = torch.nan_to_num(x, nan=0.0)
        return self.mlp(x)


# TRAINING


class BaselineTrainer:
    """Unified trainer for all baseline models."""
    
    def __init__(self, model, model_type, train_loader, val_loader, 
                 device='cuda', learning_rate=0.001, save_dir='./checkpoints'):
        
        self.model = model.to(device)
        self.model_type = model_type
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )
        self.criterion = nn.MSELoss()
        
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in self.train_loader:            
            x = batch['x'].to(self.device)
            target = batch['target'].to(self.device)                               
            pred = self.model(x)            
            
            # Loss on all values (not just masked)
            valid_mask = ~torch.isnan(target)
            if valid_mask.sum() == 0:
                continue
            
            loss = self.criterion(pred[valid_mask], target[valid_mask])
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                x = batch['x'].to(self.device)
                target = batch['target'].to(self.device)
                
                pred = self.model(x)
                
                valid_mask = ~torch.isnan(target)
                if valid_mask.sum() == 0:
                    continue
                
                loss = self.criterion(pred[valid_mask], target[valid_mask])
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def fit(self, num_epochs=50, patience=10):
        print(f"\n{'='*60}")
        print(f"{self.model_type.upper()} EĞİTİM BAŞLIYOR")
        print(f"{'='*60}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        pbar = tqdm(range(num_epochs), desc="Eğitim")
        
        for epoch in pbar:
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{num_epochs}: Train={train_loss:.4f}, Val={val_loss:.4f}, Best={best_val_loss:.4f}, LR={current_lr:.2e}")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        print(f"{self.model_type.upper()} Eğitim Bitti. Best Val Loss: {best_val_loss:.4f}")
        
        return best_val_loss


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
    """
    Level-aware Real MAE hesapla.
    """
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


def evaluate_baseline_model(model, model_type, test_graph, device='cuda', seq_length=5,
                            return_arrays=False):
    """Evaluate a baseline model on test data with dynamic sequence length.

    If return_arrays=True, returns tuple (results_df, predictions, original, mask,
    pressures, station_ids, datetimes) so callers (M99) can compute extra metrics
    (extreme / by_pressure / profile_consistency) with parity to GNN."""

    print(f"\n{'='*60}")
    print(f"{model_type.upper()} DEĞERLENDİRME (Seq Length: {seq_length})")
    print(f"{'='*60}")

    model.eval()

    # Get test data
    test_x = test_graph['x'].numpy() if isinstance(test_graph['x'], torch.Tensor) else test_graph['x']
    original = test_x.copy()

    feature_names = ['temperature', 'relative_humidity', 'wind_speed',
                     'sin_wd', 'cos_wd', 'geopotential']

    #  Tüm modellerde AYNI mask kullanılacak (Adil karşılaştırma)
    np.random.seed(MASK_SEED)
    mask = np.random.rand(*test_x.shape) < MASK_RATIO
    mask = mask & ~np.isnan(test_x)
    
    #  Masked input oluştur (model bunu görecek)
    test_x_masked = test_x.copy()
    test_x_masked[mask] = 0.0
    
    # Boş tahmin dizisi oluştur
    predictions = np.zeros_like(test_x)
    pred_count = np.zeros(len(test_x))
    
    if model_type == 'mlp':
        # MLP: node-level prediction
        metadata = test_graph['node_metadata']
        
        # Pozisyonel features oluştur
        pressure = np.array(metadata['pressure']) / 1000.0
        lat = np.array(metadata['lat']) / 90.0
        lon = np.array(metadata['lon']) / 180.0
        datetimes = metadata['datetime']
        hours = np.array([dt.hour for dt in datetimes])
        time_sin = np.sin(2 * np.pi * hours / 24)
        time_cos = np.cos(2 * np.pi * hours / 24)
        pos_features = np.column_stack([pressure, lat, lon, time_sin, time_cos])
        
        # Masked veri + pozisyon
        x_with_pos = np.concatenate([test_x_masked, pos_features], axis=1)
        
        # Batch halinde tahmin
        batch_size = 512
        all_preds = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(x_with_pos), batch_size), desc=f"{model_type.upper()} Eval"):
                batch = torch.tensor(x_with_pos[i:i+batch_size], dtype=torch.float32).to(device)
                pred = model(batch)
                all_preds.append(pred.cpu().numpy())
        
        predictions = np.concatenate(all_preds, axis=0)
        
    else:
        # LSTM/CNN: sequence-level prediction
        metadata = test_graph['node_metadata']
        station_ids = np.array(metadata['station_id'])
        pressures = np.array(metadata['pressure'])
        
        unique_stations = np.unique(station_ids)
        unique_pressures = np.unique(pressures)
        
        step_size = max(1, seq_length // 2)
        print(f"Sliding Window: Size={seq_length}, Step={step_size}")
        
        for station in tqdm(unique_stations, desc="Station Loop"):
            for pressure in unique_pressures:
                idx_mask = (station_ids == station) & (np.abs(pressures - pressure) < 1)
                indices = np.where(idx_mask)[0]
                
                if len(indices) < seq_length:
                    # train-normalized mean is 0 in the level-aware z-scored space
                    predictions[indices] = 0.0
                    pred_count[indices] = 1
                    continue
                
                #  Global masked veriyi kullan
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
                            # Overlap averaging
                            predictions[idx] = (predictions[idx] * pred_count[idx] + pred[j]) / (pred_count[idx] + 1)
                        pred_count[idx] += 1
        
        # train-normalized mean is 0 in the level-aware z-scored space
        zero_count = pred_count == 0
        if zero_count.any():
            predictions[zero_count] = 0.0
    
    # METRİK HESAPLAMA (Real MAE dahil)
    
    # Scaling stats ve pressure bilgisi al
    scaling_stats = test_graph.get('scaling_stats', {})
    level_stats = scaling_stats.get('level_stats', {})
    metadata = test_graph['node_metadata']
    pressures = np.array(metadata['pressure'])
    
    results = []
    print(f"\n{'Feature':<20} | {'R²':>8} | {'MAE':>8} | {'MAE_real':>10} | {'Unit':>6}")
    print("-" * 65)
    
    for feat_idx, feat_name in enumerate(feature_names):
        feat_mask = mask[:, feat_idx]
        if feat_mask.sum() < 10: 
            continue
        
        orig = original[feat_mask, feat_idx]
        pred = predictions[feat_mask, feat_idx]
        p_masked = pressures[feat_mask]  # ← YENİ: Bu noktaların basınçları
        
        valid = ~(np.isnan(orig) | np.isnan(pred))
        if valid.sum() < 10: 
            continue
        
        orig_valid = orig[valid]
        pred_valid = pred[valid]
        p_valid = p_masked[valid]  # ← YENİ
        
        # Normalized metrikler
        r2 = r2_score(orig_valid, pred_valid)
        mae = mean_absolute_error(orig_valid, pred_valid)
        bias = float(np.mean(pred_valid - orig_valid))
        mse = mean_squared_error(orig_valid, pred_valid)
        rmse = float(np.sqrt(mse))

        # Real MAE + Real bias (level-aware denormalize)
        if len(level_stats) > 0:
            mae_real = calculate_real_mae(orig_valid, pred_valid, p_valid, level_stats, feat_idx)
            real_biases = np.empty(len(orig_valid))
            for i in range(len(orig_valid)):
                std = get_level_std(p_valid[i], level_stats, feat_idx)
                real_biases[i] = (pred_valid[i] - orig_valid[i]) * std
            bias_real = float(np.mean(real_biases))
        else:
            global_stds = scaling_stats.get('stds', np.ones(6))
            if isinstance(global_stds, np.ndarray):
                mae_real = mae * global_stds[feat_idx]
                bias_real = bias * global_stds[feat_idx]
            else:
                mae_real = mae
                bias_real = bias

        unit = FEATURE_UNITS.get(feat_name, '-')

        results.append({
            'method':    model_type.upper(),
            'feature':   feat_name,
            'r2':        r2,
            'mae':       mae,
            'mae_real':  mae_real,
            'unit':      unit,
            'rmse':      rmse,
            'bias':      bias,
            'bias_real': bias_real,
            'n':         int(valid.sum()),
        })
        print(f"{feat_name:<20} | {r2:>8.4f} | {mae:>8.4f} | {mae_real:>10.4f} | {unit:>6}")

    results_df = pd.DataFrame(results)
    if return_arrays:
        metadata = test_graph['node_metadata']
        station_ids = np.array(metadata['station_id'])
        datetimes   = np.array(metadata['datetime'])
        pressures_arr = np.array(metadata['pressure'])
        return results_df, predictions, original, mask, pressures_arr, station_ids, datetimes
    return results_df


def train_and_evaluate_baseline(model_type, train_graph, test_graph, val_graph,
                                device='cuda',
                                num_epochs=None, batch_size=None,
                                seq_length=None, patience=None):
    """Train and evaluate a single baseline model."""
    
    
    
    # Config'den varsayılan değerleri al
    num_epochs = num_epochs or cfg.Baseline.num_epochs
    batch_size = batch_size or cfg.Baseline.batch_size
    seq_length = seq_length or cfg.Baseline.seq_length
    patience = patience or cfg.Baseline.patience
    
    # Model tipine göre hidden_dim seç
    if model_type == 'mlp':
        hidden_dim = cfg.Baseline.mlp_hidden_dim      # 256
    elif model_type == 'lstm':
        hidden_dim = cfg.Baseline.lstm_hidden_dim     # 128
    elif model_type == 'cnn':
        hidden_dim = cfg.Baseline.cnn_hidden_dim      # 64
    else:
        raise ValueError(f"Unknown model type: {model_type}")
     
    print(f"\n{'='*70}")
    print(f"{model_type.upper()} MODEL (Seq: {seq_length}, Hidden: {hidden_dim}, Patience: {patience})")
    print(f"{'='*70}")
    
    # Create datasets
    if model_type == 'mlp':
        train_ds = RadiosondeMLPDataset(train_graph, mask_ratio=cfg.Masking.mask_ratio, mode='train')
        val_ds   = RadiosondeMLPDataset(val_graph,   mask_ratio=cfg.Masking.mask_ratio, mode='test')
    else:
        train_ds = RadiosondeSequenceDataset(train_graph, seq_length=seq_length, mask_ratio=cfg.Masking.mask_ratio, mode='train')
        val_ds   = RadiosondeSequenceDataset(val_graph,   seq_length=seq_length, mask_ratio=cfg.Masking.mask_ratio, mode='test')

        
    train_batch = batch_size  # Hepsi için aynı
        
    train_dl = DataLoader(train_ds, batch_size=train_batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=train_batch, shuffle=False, num_workers=0)
    
    if model_type == 'lstm':
        model = LSTMImputer(input_dim=6, hidden_dim=hidden_dim, num_layers=2)
    elif model_type == 'cnn':
        model = CNNImputer(input_dim=6, hidden_dim=hidden_dim, num_layers=4)
    elif model_type == 'mlp':
        model = MLPImputer(input_dim=6, pos_dim=5, hidden_dim=hidden_dim, num_layers=4)
    else: 
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = BaselineTrainer(
        model=model,
        model_type=model_type,
        train_loader=train_dl,
        val_loader=val_dl,
        device=device,
        learning_rate=cfg.Baseline.learning_rate,  # 0.001
        save_dir=cfg.save_dir
    )
    
    trainer.fit(num_epochs=num_epochs, patience=patience)

    # Evaluate (return_arrays=True so M99 can compute extra parity metrics)
    eval_out = evaluate_baseline_model(
        model, model_type, test_graph, device,
        seq_length=seq_length, return_arrays=True,
    )
    results, predictions, original, mask, pressures_arr, station_ids, datetimes = eval_out
    eval_arrays = {
        'predictions': predictions,
        'targets':     original,
        'mask':        mask,
        'pressures':   pressures_arr,
        'stations':    station_ids,
        'datetimes':   datetimes,
    }

    return model, results, trainer.history, eval_arrays



def run_all_baselines(train_graph, test_graph, val_graph, device='cuda',
                      seq_length=5, patience=10,
                      num_epochs=50, batch_size=64):

    """Run all baseline models and combine results."""

    all_results = []

    for model_type in ['mlp', 'lstm', 'cnn']:
        try:

            _, results, _, _ = train_and_evaluate_baseline(
                model_type,
                train_graph,
                test_graph,
                val_graph=val_graph,
                device=device,
                seq_length=seq_length,
                patience=patience,
                num_epochs=num_epochs,
                batch_size=batch_size
            )
            all_results.append(results)
        except Exception as e:
            # Hata detayını görmek için traceback eklemek iyi olur
            import traceback
            traceback.print_exc()
            print(f"{model_type.upper()} failed: {str(e)[:100]}")
    
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        
        # Save results
        # Klasör yoksa hata vermemesi için kontrol ekleyelim
        import os
        os.makedirs('HTML_Export', exist_ok=True)
        combined.to_csv('HTML_Export/deep_learning_baselines.csv', index=False)
        print(f"\n Results saved: HTML_Export/deep_learning_baselines.csv")
        # Print comparison table
        print("\n" + "="*70)
        print("DEEP LEARNING BASELINE KARŞILAŞTIRMA (R² Scores)")
        print("="*70)
        
        pivot = combined.pivot(index='feature', columns='method', values='r2')
        print(pivot.round(4).to_string())
        
        return combined
    
    return None

# ENTRY POINT (MAIN EXECUTION)


if __name__ == "__main__":
    # Parametreleri Config'den veya argümanlardan alabiliriz
    parser = argparse.ArgumentParser(description='Deep Learning Baselines')
    parser.add_argument('--model', type=str, default='all', choices=['lstm', 'cnn', 'mlp', 'all'])
    
    # Varsayılan değerleri Config dosyasından alalım
    parser.add_argument('--epochs', type=int, default=cfg.Baseline.num_epochs, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=cfg.Baseline.batch_size, help='Batch size') 
    parser.add_argument('--patience', type=int, default=cfg.Baseline.patience, help='Early stopping patience')
    
        

    

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_known_args()[0]
    
    print(f"""
    
      DEEP LEARNING BASELINES FOR RADIOSONDE IMPUTATION                   
      Config:  Epochs={args.epochs}, Pat={args.patience}                  

    """)
    
    device = cfg.device
    print(f"Device: {device}")
    
    # 1. ADIM: Veri yükleme
    print("\n Veri yükleniyor...")
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
    
    # 2. ADIM: Train/Val/Test split (70/10/20, chronological with buffer gaps)
    import pandas as pd
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

    # 3. ADIM: Grafik Yapılarının Oluşturulması
    builder = RadiosondeGraphBuilder(
        station_metadata=stations,
        temporal_window=cfg.graph_temporal_window
    )

    print("\n Grafikler oluşturuluyor...")
    train_graph = builder.build_graph_from_observations(train_obs)
    val_graph = builder.build_graph_from_observations(
        val_obs,
        external_stats=train_graph['scaling_stats']
    )
    test_graph = builder.build_graph_from_observations(
        test_obs,
        external_stats=train_graph['scaling_stats']
    )

    # 4. ADIM: Modellerin Çalıştırılması
    if args.model == 'all':
        # run_all_baselines parametrelerini Config'den alarak gönderelim
        results = run_all_baselines(
            train_graph,
            test_graph,
            val_graph=val_graph,
            device=device,
            seq_length=cfg.Baseline.seq_length,  # Config
            patience=args.patience,              # Arg (default=Config)
            num_epochs=args.epochs,              # Arg (default=Config)
            batch_size=args.batch_size           # Arg (default=Config)
        )
    else:
        model, results, _, _ = train_and_evaluate_baseline(
                args.model,
                train_graph,
                test_graph,
                val_graph=val_graph,
                device=device,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                seq_length=cfg.Baseline.seq_length,
                patience=args.patience
            )
    
    print("\n Tüm işlemler başarıyla tamamlandı!")
    


