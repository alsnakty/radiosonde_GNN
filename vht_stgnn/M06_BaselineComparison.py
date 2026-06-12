"""
M06_BaselineComparison.py - Baseline Comparison Module
"""

import os
import gc
import torch
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, griddata
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Config import
from M00_Config import cfg

# Masking constants (for comparison)
MASK_SEED = cfg.Masking.random_seed    # 42
MASK_RATIO = cfg.Masking.mask_ratio    # 0.15

FEATURE_UNITS = {
    'temperature': '°C',
    'relative_humidity': '%',
    'wind_speed': 'm/s',
    'sin_wd': '-',
    'cos_wd': '-',
    'geopotential': 'm'
}


class BaselineInterpolator:
    """
    Baseline interpolation methods compatible with the VHT-GNN framework.
    
    Memory Optimized:
    - IDW with batch processing
    - float32 precision
    """
    
    FEATURE_NAMES = ['temperature', 'relative_humidity', 'wind_speed', 
                     'sin_wd', 'cos_wd', 'geopotential']
    FEATURE_UNITS = ['°C', '%', 'm/s', '', '', 'm']
    
    def __init__(self, graph_data: Dict, scaling_stats: Dict = None):
        """
        Initialize with graph data from RadiosondeGraphBuilder.
        """
        # Convert to float32 (memory saving)
        if isinstance(graph_data['x'], torch.Tensor):
            self.x = graph_data['x'].numpy().astype(np.float32)
        else:
            self.x = np.array(graph_data['x'], dtype=np.float32)
        
        self.metadata = graph_data['node_metadata']
        self.time_steps = graph_data.get('time_steps', self.metadata.get('time_steps', []))
        self.scaling_stats = scaling_stats or graph_data.get('scaling_stats', {})
        self.level_stats = self.scaling_stats.get('level_stats', {})
        
        # Convert metadata to numpy arrays (float32)
        self.pressures = np.array(self.metadata['pressure'], dtype=np.float32)
        self.lats = np.array(self.metadata['lat'], dtype=np.float32)
        self.lons = np.array(self.metadata['lon'], dtype=np.float32)
        self.station_ids = np.array(self.metadata['station_id'])
        self.level_types = np.array(self.metadata['level_type'])
        
        # Convert datetimes to numeric
        datetimes = self.metadata['datetime']
        self.datetime_numeric = np.array([
            (dt - datetimes[0]).total_seconds() / 3600 for dt in datetimes
        ], dtype=np.float32)
        
        self.num_nodes = self.x.shape[0]
        self.num_features = self.x.shape[1]
        
        # Coordinate precomputation (for IDW)
        self._precompute_coordinates()
        
        print(f" BaselineInterpolator initialized (Memory Optimized)")
        print(f"Nodes: {self.num_nodes:,}")
        print(f"Time steps: {len(self.time_steps)}")
        print(f"Unique stations: {len(np.unique(self.station_ids))}")
        print(f"Memory: ~{self.x.nbytes / 1024 / 1024:.1f} MB (features only)")
    
    def _precompute_coordinates(self):
        """
        Precompute 3D coordinates (for IDW).
        """
        mean_lat = np.mean(self.lats)
        cos_lat = np.cos(np.radians(mean_lat))
        
        # Vertical scale factor
        vertical_scale = 100
        
        # 3D coordinates: lat_km, lon_km, pseudo_altitude_km
        self.coords_3d = np.column_stack([
            self.lats * 111,
            self.lons * 111 * cos_lat,
            -vertical_scale * np.log(np.maximum(self.pressures, 1.0) / 1013.25)
        ]).astype(np.float32)
        
  
    # IDW - BATCH PROCESSING  
    
    def _idw_batch(self, known_coords: np.ndarray, known_values: np.ndarray,
                   target_coords: np.ndarray, power: float = 2.0,
                   batch_size: int = 1000) -> np.ndarray:
        """
        Calculate IDW in batches  
        
        This method does not store large distance matrices in memory.
        """
        n_targets = len(target_coords)
        n_known = len(known_coords)

        # Result array
        result = np.zeros(n_targets, dtype=np.float32)
        
        # Process in batches
        for start_idx in range(0, n_targets, batch_size):
            end_idx = min(start_idx + batch_size, n_targets)
            batch_targets = target_coords[start_idx:end_idx]
            
            # Calculate distances for this batch
            # Broadcasting: (batch, 1, 3) - (1, known, 3) -> (batch, known, 3)
            diff = batch_targets[:, np.newaxis, :] - known_coords[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff ** 2, axis=2))  # (batch, known)
            
            # IDW weights
            with np.errstate(divide='ignore', invalid='ignore'):
                weights = 1.0 / (distances ** power + 1e-10)
            
            # Normalize
            weight_sums = weights.sum(axis=1, keepdims=True)
            weights_normalized = weights / weight_sums
            
            # Interpolate
            result[start_idx:end_idx] = weights_normalized @ known_values
            
            # Clear memory
            del diff, distances, weights, weights_normalized
        
        return result
    
    def idw_interpolate(self, 
                        known_indices: np.ndarray,
                        target_indices: np.ndarray,
                        feature_idx: int,
                        power: float = 2.0) -> np.ndarray:
        """
         3D IDW interpolation.
        """
        known_values = self.x[known_indices, feature_idx]
        
        # Remove NaN from known
        valid = ~np.isnan(known_values)
        if valid.sum() == 0:
            return np.full(len(target_indices), np.nan, dtype=np.float32)
        
        known_indices_valid = known_indices[valid]
        known_values_valid = known_values[valid]
        
        # Use precomputed coordinates
        known_coords = self.coords_3d[known_indices_valid]
        target_coords = self.coords_3d[target_indices]
        
        # Batch IDW
        interpolated = self._idw_batch(known_coords, known_values_valid,
                                        target_coords, power=power,
                                        batch_size=2000)
        
        return interpolated
    
    def idw_impute(self, mask: np.ndarray, power: float = 2.0) -> np.ndarray:
        
        imputed = self.x.copy()
        
        unique_times = np.unique(self.datetime_numeric)
        
        print(f"  IDW processing over {len(unique_times)} time steps...")
        
        for time_step in tqdm(unique_times, desc="IDW Spatial"):
            time_indices = np.where(self.datetime_numeric == time_step)[0]
            
            if len(time_indices) == 0:
                continue
                
            for feat_idx in range(self.num_features):
                current_feat_mask = mask[time_indices, feat_idx]
                
                # Targets (Missing values - at this time step)
                target_local_idx = np.where(current_feat_mask)[0]
                if len(target_local_idx) == 0:
                    continue
                
                # Knowns (Observed values - at this time step)
                # Neither masked nor NaN
                current_values = self.x[time_indices, feat_idx]
                known_local_idx = np.where(~current_feat_mask & ~np.isnan(current_values))[0]
                
                # Convert to global indices
                target_global_idx = time_indices[target_local_idx]
                
                # If not enough neighbors, fill with mean (Spatial insufficiency)
                if len(known_local_idx) < 3:
                    imputed[target_global_idx, feat_idx] = np.nanmean(self.x[:, feat_idx])
                    continue
                
                known_global_idx = time_indices[known_local_idx]
                
                # 3. Interpolate using only neighbors at this time
                # Distance > 0 guaranteed
                imputed_values = self.idw_interpolate(known_global_idx, target_global_idx, 
                                                       feat_idx, power=power)
                imputed[target_global_idx, feat_idx] = imputed_values
        
        gc.collect()
        return imputed
    
    def linear_temporal_impute(self, mask: np.ndarray) -> np.ndarray:
        """
        Linear interpolation in time dimension.
        For each station-pressure combination.
        """
        imputed = self.x.copy()
        
        unique_stations = np.unique(self.station_ids)
        unique_pressures = np.unique(self.pressures)
        
        for feat_idx in range(self.num_features):
            for station in unique_stations:
                for pressure in unique_pressures:
                    group_mask = (self.station_ids == station) & (np.abs(self.pressures - pressure) < 1)
                    group_indices = np.where(group_mask)[0]
                    
                    if len(group_indices) < 2:
                        continue
                    
                    feat_mask = mask[group_indices, feat_idx]
                    if not feat_mask.any() or feat_mask.all():
                        continue
                    
                    times = self.datetime_numeric[group_indices]
                    values = self.x[group_indices, feat_idx]
                    
                    known_mask = ~feat_mask & ~np.isnan(values)
                    if known_mask.sum() < 2:
                        continue
                    
                    
                    t_known = times[known_mask]
                    v_known = values[known_mask]
                    
                    sort_idx = np.argsort(t_known)
                    
                    try:
                        f = interp1d(t_known[sort_idx], v_known[sort_idx],
                                    kind='linear', bounds_error=False,
                                    fill_value='extrapolate')
                        
                        target_mask = feat_mask
                        # Target times don't have to be sorted
                        imputed[group_indices[target_mask], feat_idx] = f(times[target_mask])
                    except:
                        pass
        
        gc.collect()
        return imputed
    
    def linear_vertical_impute(self, mask: np.ndarray) -> np.ndarray:
        """
        Linear interpolation in vertical (pressure) dimension.
        For each station-datetime combination.
        """
        imputed = self.x.copy()
        
        unique_stations = np.unique(self.station_ids)
        unique_times = np.unique(self.datetime_numeric)
        
        for feat_idx in range(self.num_features):
            for station in unique_stations:
                for time in unique_times:
                    group_mask = ((self.station_ids == station) & 
                                  (self.datetime_numeric == time))
                    group_indices = np.where(group_mask)[0]
                    
                    if len(group_indices) < 2:
                        continue
                    
                    feat_mask = mask[group_indices, feat_idx]
                    if not feat_mask.any() or feat_mask.all():
                        continue
                    
                    
                    log_p = np.log(np.maximum(self.pressures[group_indices], 1.0))
                    values = self.x[group_indices, feat_idx]
                    
                    known_mask = ~feat_mask & ~np.isnan(values)
                    if known_mask.sum() < 2:
                        continue
                    
                    
                    sort_idx = np.argsort(log_p[known_mask])
                    
                    try:
                        f = interp1d(log_p[known_mask][sort_idx], 
                                    values[known_mask][sort_idx],
                                    kind='linear', bounds_error=False,
                                    fill_value='extrapolate')
                        
                        target_mask = feat_mask
                        imputed[group_indices[target_mask], feat_idx] = f(log_p[target_mask])
                    except:
                        pass
        
        gc.collect()
        return imputed
    
    def linear_spatial_impute(self, mask: np.ndarray) -> np.ndarray:
        """
        Linear (triangulation-based) spatial interpolation.
        For each datetime-pressure combination.
        """
        imputed = self.x.copy()
        
        unique_times = np.unique(self.datetime_numeric)
        unique_pressures = np.unique(self.pressures)
        
        for feat_idx in range(self.num_features):
            for time in unique_times:
                for pressure in unique_pressures:
                    group_mask = ((self.datetime_numeric == time) & 
                                  (np.abs(self.pressures - pressure) < 1))
                    group_indices = np.where(group_mask)[0]
                    
                    if len(group_indices) < 3:  # Need at least 3 for triangulation
                        continue
                    
                    feat_mask = mask[group_indices, feat_idx]
                    if not feat_mask.any() or feat_mask.all():
                        continue
                    
                    lats = self.lats[group_indices]
                    lons = self.lons[group_indices]
                    values = self.x[group_indices, feat_idx]
                    
                    known_mask = ~feat_mask & ~np.isnan(values)
                    if known_mask.sum() < 3:
                        continue
                    
                    try:
                        known_coords = np.column_stack([lats[known_mask], lons[known_mask]])
                        target_coords = np.column_stack([lats[feat_mask], lons[feat_mask]])
                        
                        interpolated = griddata(known_coords, values[known_mask],
                                               target_coords, method='linear')
                        
                        imputed[group_indices[feat_mask], feat_idx] = interpolated
                    except:
                        pass
        
        gc.collect()
        return imputed
    
    def linear_combined_impute(self, mask: np.ndarray) -> np.ndarray:

            print("    → Temporal...")
            temporal = self.linear_temporal_impute(mask)
            
            print("    → Vertical...")
            vertical = self.linear_vertical_impute(mask)
            
            print("    → Spatial...")
            spatial = self.linear_spatial_impute(mask)
            
            print("    → Averaging...")
            
            stack = np.stack([temporal, vertical, spatial], axis=0)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                combined = np.nanmean(stack, axis=0)
                      
            # Memory cleanup
            del temporal, vertical, spatial, stack
            gc.collect()
            
            return combined.astype(np.float32)
    
    # HELPER: Level-aware std
    
    def get_level_std(self, pressure: float, feature_idx: int) -> float:
        """Returns std value of feature for a specific pressure level."""
        level_key = f"p_{int(pressure)}"
        
        if level_key in self.level_stats:
            stds = self.level_stats[level_key].get('stds', None)
            if stds is not None:
                if isinstance(stds, np.ndarray):
                    return float(stds[feature_idx])
                elif isinstance(stds, list):
                    return float(stds[feature_idx])
        
        # Find nearest level
        available_levels = []
        for key in self.level_stats.keys():
            if key.startswith('p_'):
                try:
                    p = int(key.split('_')[1])
                    available_levels.append(p)
                except:
                    pass
        
        if not available_levels:
            return 1.0
        
        nearest = min(available_levels, key=lambda x: abs(x - pressure))
        nearest_key = f"p_{nearest}"
        
        if nearest_key in self.level_stats:
            stds = self.level_stats[nearest_key].get('stds', None)
            if stds is not None:
                if isinstance(stds, np.ndarray):
                    return float(stds[feature_idx])
                elif isinstance(stds, list):
                    return float(stds[feature_idx])
        
        return 1.0
    
    # EVALUATION
    
    def evaluate(self, original: np.ndarray, imputed: np.ndarray, 
                 mask: np.ndarray, feature_idx: int = None) -> Dict:
        """
        Calculate metrics for imputation - Including Real MAE.
        """
        if feature_idx is not None:
            feat_mask = mask[:, feature_idx]
            
            # NEW ADDITION: 200 hPa filtering for Humidty
            feature_name = self.FEATURE_NAMES[feature_idx]
            if feature_name == 'relative_humidity':
                # Only take levels >= 200 hPa (higher pressure / lower altitude)
                valid_p_mask = self.pressures >= 200.0
                feat_mask = feat_mask & valid_p_mask
            
            orig = original[feat_mask, feature_idx]
            pred = imputed[feat_mask, feature_idx]
            pressures_masked = self.pressures[feat_mask]
        else: 
            orig = original[mask]
            pred = imputed[mask]
            pressures_masked = None
        
        # Remove NaN
        valid = ~(np.isnan(orig) | np.isnan(pred))
        if valid.sum() == 0:
            return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan,
                    'mae_real': np.nan, 'bias': np.nan, 'bias_real': np.nan,
                    'r2': np.nan, 'n': 0}

        orig = orig[valid]
        pred = pred[valid]

        mse = mean_squared_error(orig, pred)
        mae = mean_absolute_error(orig, pred)
        bias = float(np.mean(pred - orig))

        # Real MAE + Real bias (level-aware denormalize)
        mae_real = mae
        bias_real = bias

        if feature_idx is not None and pressures_masked is not None and len(self.level_stats) > 0:
            p_valid = pressures_masked[valid]
            real_errors = np.empty(len(orig))
            real_biases = np.empty(len(orig))
            for i in range(len(orig)):
                std = self.get_level_std(p_valid[i], feature_idx)
                real_errors[i] = abs(orig[i] - pred[i]) * std
                real_biases[i] = (pred[i] - orig[i]) * std
            mae_real = float(np.mean(real_errors))
            bias_real = float(np.mean(real_biases))

        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'mae_real': mae_real,
            'bias': bias,
            'bias_real': bias_real,
            'r2': r2_score(orig, pred),
            'n': len(orig)
        }
    
    def evaluate_by_level(self, original: np.ndarray, imputed: np.ndarray,
                          mask: np.ndarray, feature_idx: int) -> pd.DataFrame:
        """
        Evaluate by pressure level - Real MAE.
        """
        results = []
        
        unique_pressures = sorted(np.unique(self.pressures), reverse=True)
        
        for pressure in unique_pressures:
            level_mask = np.abs(self.pressures - pressure) < 1
            combined_mask = mask[:, feature_idx] & level_mask
            
            if combined_mask.sum() == 0:
                continue
            
            orig = original[combined_mask, feature_idx]
            pred = imputed[combined_mask, feature_idx]
            
            valid = ~(np.isnan(orig) | np.isnan(pred))
            if valid.sum() < 10:
                continue
            
            orig_valid = orig[valid]
            pred_valid = pred[valid]
            
            r2 = r2_score(orig_valid, pred_valid)
            mae = mean_absolute_error(orig_valid, pred_valid)
            bias = float(np.mean(pred_valid - orig_valid))

            # Real MAE + Real bias (level-aware denormalize)
            std = self.get_level_std(pressure, feature_idx)
            mae_real = mae * std
            bias_real = bias * std

            results.append({
                'pressure': int(pressure),
                'feature': self.FEATURE_NAMES[feature_idx],
                'r2': r2,
                'mae': mae,
                'mae_real': mae_real,
                'bias': bias,
                'bias_real': bias_real,
                'unit': FEATURE_UNITS.get(self.FEATURE_NAMES[feature_idx], '-'),
                'n': valid.sum()
            })
        
        return pd.DataFrame(results)


def run_baseline_comparison(test_graph: Dict, 
                            mask: np.ndarray = None,
                            mask_ratio: float = MASK_RATIO,
                            stgnn_predictions: np.ndarray = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all baseline methods and compare with VHT-GNN.
    
    Methods: IDW, Linear (Temporal, Vertical, Spatial, Combined)
    """
    print("\n" + "="*80)
    print("BASELINE COMPARISON STARTING (Memory Optimized)")
    print("="*80)
    
    interpolator = BaselineInterpolator(test_graph, test_graph.get('scaling_stats'))
    
    original = interpolator.x.copy()
    
    # Create mask if not provided
    if mask is None:
        print(f" Creating mask (ratio={mask_ratio})...")
        np.random.seed(MASK_SEED)   
        mask = (np.random.rand(*interpolator.x.shape) < mask_ratio).astype(bool)
        mask = mask & ~np.isnan(interpolator.x)
    
    print(f"Total masked: {mask.sum():,} / {mask.size:,} ({100*mask.mean():.1f}%)")
    
    # Run all methods
    methods = {}
    
    # 1. IDW
    print("\n→ Running IDW (batch mode)...")
    methods['IDW'] = interpolator.idw_impute(mask, power=cfg.Baseline.idw_power)
    gc.collect()
    
    # 2. Linear methods
    print("→ Running Linear Temporal...")
    methods['Linear_Temporal'] = interpolator.linear_temporal_impute(mask)
    gc.collect()
    
    print("→ Running Linear Vertical...")
    methods['Linear_Vertical'] = interpolator.linear_vertical_impute(mask)
    gc.collect()
    
    print("→ Running Linear Spatial...")
    methods['Linear_Spatial'] = interpolator.linear_spatial_impute(mask)
    gc.collect()
    
    print("→ Running Linear Combined...")
    methods['Linear_Combined'] = interpolator.linear_combined_impute(mask)
    gc.collect()
    
    # Add STGNN if provided
    if stgnn_predictions is not None:
        methods['VHT-GNN'] = stgnn_predictions
    
    # Evaluate all methods
    global_results = []
    level_results = []
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    for feat_idx, feat_name in enumerate(BaselineInterpolator.FEATURE_NAMES):
        print(f"\n--- {feat_name.upper()} ---")
        unit = FEATURE_UNITS.get(feat_name, '-')
        
        for method_name, imputed in methods.items():
            metrics = interpolator.evaluate(original, imputed, mask, feat_idx)
            metrics['method'] = method_name
            metrics['feature'] = feat_name
            metrics['unit'] = FEATURE_UNITS.get(feat_name, '-')
            global_results.append(metrics)
            
            print(f"{method_name:<18}: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}, MAE_real={metrics['mae_real']:.4f} {unit}")
            
            # Level-based evaluation
            level_df = interpolator.evaluate_by_level(original, imputed, mask, feat_idx)
            if not level_df.empty:
                level_df['method'] = method_name
                level_df['feature'] = feat_name
                level_results.append(level_df)
    
    global_df = pd.DataFrame(global_results)
    level_df = pd.concat(level_results, ignore_index=True) if level_results else pd.DataFrame()

    # Arrays for M99 parity metrics (extreme / by_pressure / profile_consistency)
    predictions_per_method = dict(methods)  # IDW + 4 Linear variants
    shared_arrays = {
        'targets':   original,
        'mask':      mask,
        'pressures': interpolator.pressures,
        'stations':  interpolator.station_ids,
        'datetimes': np.array(interpolator.metadata['datetime']),
    }

    # Final cleanup
    gc.collect()

    return global_df, level_df, predictions_per_method, shared_arrays


def generate_comparison_table(global_results: pd.DataFrame, 
                              metric: str = 'r2') -> pd.DataFrame:
    """
    Generate comparison table for paper.
    """
    pivot = global_results.pivot(index='feature', columns='method', values=metric)
    
 
    method_order = ['Linear_Temporal', 'Linear_Vertical', 'Linear_Spatial',
                    'Linear_Combined', 'IDW', 'VHT-GNN']
    cols = [c for c in method_order if c in pivot.columns]
    
    return pivot[cols]


def generate_latex_table(global_results: pd.DataFrame, 
                         metric: str = 'r2',
                         caption: str = 'Comparison of imputation methods') -> str:
    """
    Generate LaTeX table.
    """
    pivot = generate_comparison_table(global_results, metric)
    
    feature_display = {
        'temperature': 'Temperature',
        'relative_humidity': 'Relative Humidity',
        'wind_speed': 'Wind Speed',
        'sin_wd': 'Sin(Wind Dir)',
        'cos_wd': 'Cos(Wind Dir)',
        'geopotential': 'Geopotential'
    }
    
    metric_label = '$R^2$' if metric == 'r2' else 'MAE'
    
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        f'\\caption{{{caption} ({metric_label})}}',
        r'\label{tab:baseline_comparison}',
        r'\begin{tabular}{l' + 'c' * len(pivot.columns) + '}',
        r'\hline',
        r'\textbf{Variable} & ' + ' & '.join([f'\\textbf{{{m.replace("_", " ")}}}' 
                                               for m in pivot.columns]) + r' \\',
        r'\hline'
    ]
    
    for feat in pivot.index:
        display_name = feature_display.get(feat, feat)
        values = pivot.loc[feat].values
        
        # Bold the best value
        valid_values = [v for v in values if not np.isnan(v)]
        best_val = max(valid_values) if metric == 'r2' and valid_values else (
                   min(valid_values) if valid_values else np.nan)
        
        formatted = []
        for val in values:
            if np.isnan(val):
                formatted.append('--')
            elif abs(val - best_val) < 0.0001:
                formatted.append(f'\\textbf{{{val:.3f}}}')
            else:
                formatted.append(f'{val:.3f}')
        
        lines.append(f'{display_name} & ' + ' & '.join(formatted) + r' \\')
    
    lines.extend([
        r'\hline',
        r'\end{tabular}',
        r'\end{table}'
    ])
    
    return '\n'.join(lines)


class RealisticBaselineMasking:
    """
    Same realistic masking pattern as VHT-GNN.
    Baselines must use the same mask for fair comparison.
    """
    
    def __init__(self, mask_ratio=0.15):
        self.mask_ratio = mask_ratio
        self.probabilities = {
            'burst': 0.35,
            'sensor': 0.25,
            'comm_loss': 0.20,
            'station_offline': 0.10,
            'partial': 0.10
        }
    
    def create_mask(self, num_nodes, num_features, metadata):
        """
        Create realistic missing pattern mask.
        """
        np.random.seed(MASK_SEED)
        mask = np.zeros((num_nodes, num_features), dtype=bool)
        
        pressures = np.array(metadata['pressure'])
        station_ids = np.array(metadata['station_id'])
        
        # Unique time steps
        datetimes = metadata['datetime']
        unique_times = sorted(set(datetimes))
        time_to_idx = {t: i for i, t in enumerate(unique_times)}
        time_indices = np.array([time_to_idx[dt] for dt in datetimes])
        
        num_time_steps = len(unique_times)
        unique_stations = np.unique(station_ids)
        num_stations = len(unique_stations)
        
        # Apply multiple realistic patterns
        target_masked = int(num_nodes * num_features * self.mask_ratio)
        
        # 1. Balloon Burst Pattern (35%)
        n_bursts = max(1, int(num_time_steps * num_stations * 0.05))
        for _ in range(n_bursts):
            station = np.random.choice(unique_stations)
            time_idx = np.random.randint(0, num_time_steps)
            burst_pressure = np.random.choice([300, 250, 200, 150, 100])
            
            burst_mask = ((station_ids == station) & 
                         (time_indices == time_idx) & 
                         (pressures <= burst_pressure))
            mask[burst_mask, :] = True
        
        # 2. Sensor Failure Pattern (25%)
        n_sensor_failures = max(1, int(num_time_steps * 0.03))
        for _ in range(n_sensor_failures):
            feature_idx = np.random.choice(num_features, p=[0.20, 0.35, 0.20, 0.10, 0.10, 0.05])
            duration = np.random.randint(1, min(5, num_time_steps))
            start_time = np.random.randint(0, num_time_steps - duration + 1)
            station = np.random.choice(unique_stations)
            
            for t in range(start_time, start_time + duration):
                sensor_mask = (station_ids == station) & (time_indices == t)
                mask[sensor_mask, feature_idx] = True
        
        # 3. Communication Loss (20%)
        n_comm_loss = max(1, int(num_time_steps * 0.02))
        for _ in range(n_comm_loss):
            station = np.random.choice(unique_stations)
            duration = np.random.randint(1, min(3, num_time_steps))
            start_time = np.random.randint(0, num_time_steps - duration + 1)
            
            # Middle pressure levels affected
            for t in range(start_time, start_time + duration):
                comm_mask = ((station_ids == station) & 
                            (time_indices == t) & 
                            (pressures >= 300) & (pressures <= 700))
                mask[comm_mask, :] = True
        
        # 4. Level-dependent random missing
        level_rates = {
            'surface': 0.15, 1000: 0.20, 925: 0.12, 850: 0.08,
            700: 0.10, 500: 0.05, 300: 0.05, 200: 0.08, 100: 0.10
        }
        
        for pressure_level, rate in level_rates.items():
            if pressure_level == 'surface':
                level_mask = pressures >= 950
            else:
                level_mask = np.abs(pressures - pressure_level) < 25
            
            level_indices = np.where(level_mask & ~mask.any(axis=1))[0]
            n_to_mask = int(len(level_indices) * rate * 0.5)
            
            if n_to_mask > 0 and len(level_indices) > 0:
                selected = np.random.choice(level_indices, 
                                           min(n_to_mask, len(level_indices)), 
                                           replace=False)
                for idx in selected:
                    feat_mask = np.random.rand(num_features) < 0.5
                    mask[idx, feat_mask] = True
        
        # 5. Adjust to target ratio
        current_ratio = mask.mean()
        
        if current_ratio < self.mask_ratio:
            remaining = ~mask
            remaining_indices = np.where(remaining)
            n_additional = int((self.mask_ratio - current_ratio) * num_nodes * num_features)
            
            if n_additional > 0 and len(remaining_indices[0]) > 0:
                selected = np.random.choice(len(remaining_indices[0]), 
                                           min(n_additional, len(remaining_indices[0])),
                                           replace=False)
                mask[remaining_indices[0][selected], remaining_indices[1][selected]] = True
        
        elif current_ratio > self.mask_ratio * 1.5:
            masked_indices = np.where(mask)
            n_remove = int((current_ratio - self.mask_ratio) * num_nodes * num_features)
            
            if n_remove > 0 and len(masked_indices[0]) > 0:
                selected = np.random.choice(len(masked_indices[0]),
                                           min(n_remove, len(masked_indices[0])),
                                           replace=False)
                mask[masked_indices[0][selected], masked_indices[1][selected]] = False
        
        return mask
