"""
M02_DataLoading.py - Radiosonde Data Loading and Graph Building
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset
from M01_Utils import load_station_metadata


from M00_Config import cfg

class RealisticRadiosondeMasking:
    """
    Gerçekçi radiosonde veri kaybı simülasyonu.
    
     GÜNCELLEME: Seed desteği eklendi!
    - seed=None: Her çağrıda farklı mask (eğitim için)
    - seed=42: Her çağrıda AYNI mask (adil karşılaştırma için)
    
    Pattern Dağılımı:
    - Balloon Burst (35%): Üst seviyelerde tam kayıp
    - Sensor Failure (25%): Tek sensör arızası
    - Communication Loss (20%): Bölgesel iletişim kaybı
    - Station Offline (10%): İstasyon çevrimdışı
    - Partial Loss (10%): Rastgele kısmi kayıp
    """
    
    def __init__(self, mask_ratio=0.15,
                 burst_probability=0.35, 
                 sensor_probability=0.25,
                 comm_loss_probability=0.20,
                 station_offline_probability=0.10, 
                 partial_loss_probability=0.10,
                 seed=None):  # ← YENİ: Seed parametresi
        
        self.mask_ratio = mask_ratio
        self.seed = seed  # ← YENİ

        total = (burst_probability + sensor_probability + comm_loss_probability +
                station_offline_probability + partial_loss_probability)

        self.probabilities = {
            'burst': burst_probability / total,
            'sensor': sensor_probability / total,
            'comm_loss': comm_loss_probability / total,
            'station_offline': station_offline_probability / total,
            'partial': partial_loss_probability / total
        }

    def apply_masking(self, x_window, metadata, nodes_per_time=None, window_idx=None):
        """Apply masking."""
        # Set seed for reproducibility
        if self.seed is not None:
            if window_idx is not None:
                np.random.seed(self.seed + window_idx)
                torch.manual_seed(self.seed + window_idx)
            else:
                np.random.seed(self.seed)
                torch.manual_seed(self.seed)
        
        num_nodes, num_features = x_window.shape
        num_time_steps = len(metadata['time_steps'])
        total_elements = num_nodes * num_features

        if nodes_per_time is None:
            nodes_per_time = num_nodes // num_time_steps

        # 1. Calc natural NaN ratio
        natural_nan_count = torch.isnan(x_window).sum().item()
        natural_nan_ratio = natural_nan_count / total_elements
        
        # 2. Calc additional mask ratio
        remaining_ratio = max(0, self.mask_ratio - natural_nan_ratio)
        
        if remaining_ratio <= 0:
            return torch.zeros_like(x_window, dtype=torch.bool)
        
        # 3. Target additional masks
        target_additional_masks = int(total_elements * remaining_ratio)
        
        # 4. Select strategy
        mask = torch.zeros_like(x_window, dtype=torch.bool)
        
        strategy = np.random.choice(
            list(self.probabilities.keys()),
            p=list(self.probabilities.values())
        )

        if strategy == 'burst':
            mask = self._balloon_burst(mask, num_nodes, num_features, nodes_per_time, num_time_steps)
        elif strategy == 'sensor':
            mask = self._sensor_failure(mask, num_nodes, num_features, nodes_per_time, num_time_steps)
        elif strategy == 'comm_loss':
            mask = self._communication_loss(mask, num_nodes, num_features, nodes_per_time, num_time_steps)
        elif strategy == 'station_offline':
            mask = self._station_offline(mask, num_nodes, num_features, nodes_per_time, num_time_steps)
        else:
            mask = self._partial_loss(mask, num_nodes, num_features)
        
        # 5. Remove mask from existing NaNs
        existing_nan = torch.isnan(x_window)
        mask = mask & ~existing_nan
        
        # 6. Adjust mask ratio
        current_mask_count = mask.sum().item()
        
        if current_mask_count < target_additional_masks:
            valid_positions = ~existing_nan & ~mask
            valid_indices = torch.where(valid_positions.flatten())[0]
            need_more = min(target_additional_masks - current_mask_count, len(valid_indices))
            
            if need_more > 0:
                extra_indices = valid_indices[torch.randperm(len(valid_indices))[:need_more]]
                flat_mask = mask.flatten()
                flat_mask[extra_indices] = True
                mask = flat_mask.reshape(mask.shape)
                
        elif current_mask_count > target_additional_masks * 1.5:
            masked_indices = torch.where(mask.flatten())[0]
            remove_count = current_mask_count - target_additional_masks
            if remove_count > 0 and len(masked_indices) > 0:
                remove_indices = masked_indices[torch.randperm(len(masked_indices))[:remove_count]]
                flat_mask = mask.flatten()
                flat_mask[remove_indices] = False
                mask = flat_mask.reshape(mask.shape)

        return mask

    def _balloon_burst(self, mask, num_nodes, num_features, nodes_per_time, num_time_steps):
        time_idx = np.random.randint(0, num_time_steps)
        time_start = time_idx * nodes_per_time

        avg_levels = max(13, nodes_per_time // 8)
        num_stations = max(1, nodes_per_time // avg_levels)
        station_idx = np.random.randint(0, num_stations)
        burst_level = np.random.randint(5, min(13, avg_levels))

        station_start = time_start + (station_idx * avg_levels)
        burst_node_idx = station_start + burst_level
        station_end = min(station_start + avg_levels, (time_idx + 1) * nodes_per_time)

        if burst_node_idx < mask.shape[0] and station_end <= mask.shape[0]:
            mask[burst_node_idx:station_end, :] = True

        return mask

    def _sensor_failure(self, mask, num_nodes, num_features, nodes_per_time, num_time_steps):
        feature_probs = [0.15, 0.30, 0.25, 0.10, 0.10, 0.10]
        feature_idx = np.random.choice(num_features, p=feature_probs)

        failure_duration = np.random.randint(1, min(6, num_time_steps))
        start_time = np.random.randint(0, num_time_steps - failure_duration + 1)

        start_node = start_time * nodes_per_time
        end_node = (start_time + failure_duration) * nodes_per_time

        mask[start_node:end_node, feature_idx] = True
        return mask

    def _communication_loss(self, mask, num_nodes, num_features, nodes_per_time, num_time_steps):
        loss_duration = np.random.randint(1, min(4, num_time_steps))
        start_time = np.random.randint(0, num_time_steps - loss_duration + 1)
        
        avg_levels = max(13, nodes_per_time // 8)
        num_stations = max(1, nodes_per_time // avg_levels)
        affected_stations = np.random.randint(1, min(3, num_stations + 1))
        
        station_indices = np.random.choice(num_stations, size=affected_stations, replace=False)
        
        for t in range(start_time, start_time + loss_duration):
            time_start = t * nodes_per_time
            for station_idx in station_indices:
                station_start = time_start + (station_idx * avg_levels)
                station_end = min(station_start + avg_levels, (t + 1) * nodes_per_time)
                
                if station_start < mask.shape[0] and station_end <= mask.shape[0]:
                    mask[station_start:station_end, :] = True

        return mask

    def _station_offline(self, mask, num_nodes, num_features, nodes_per_time, num_time_steps):
        avg_levels = max(13, nodes_per_time // 8)
        num_stations = max(1, nodes_per_time // avg_levels)
        offline_station = np.random.randint(0, num_stations)

        offline_duration = np.random.randint(1, min(4, num_time_steps))
        start_time = np.random.randint(0, num_time_steps - offline_duration + 1)

        for t in range(start_time, start_time + offline_duration):
            time_start = t * nodes_per_time
            station_start = time_start + (offline_station * avg_levels)
            station_end = min(station_start + avg_levels, (t + 1) * nodes_per_time)

            if station_start < mask.shape[0] and station_end <= mask.shape[0]:
                mask[station_start:station_end, :] = True

        return mask

    def _partial_loss(self, mask, num_nodes, num_features):
        num_hotspots = np.random.randint(2, 6)

        for _ in range(num_hotspots):
            center_node = np.random.randint(0, num_nodes)
            center_feature = np.random.randint(0, num_features)
            radius = np.random.randint(5, 21)

            for i in range(max(0, center_node - radius), min(num_nodes, center_node + radius)):
                for j in range(num_features):
                    distance = abs(i - center_node) + abs(j - center_feature)
                    prob = np.exp(-distance / (radius / 2))
                    if np.random.random() < prob * self.mask_ratio * 3:
                        mask[i, j] = True

        return mask


class RadiosondeLoader:
    def __init__(self, stations_json: str = "stations.json", 
                 filter_active: bool = True, 
                 exclude_stations: Optional[List[str]] = None):
        
        self.station_metadata = load_station_metadata(stations_json, filter_active=filter_active)
        
        if exclude_stations:
            initial_count = len(self.station_metadata)
            self.station_metadata = self.station_metadata[
                ~self.station_metadata['station_id'].isin(exclude_stations)
            ]
            excluded_count = initial_count - len(self.station_metadata)
            print(f"Excluded {excluded_count} stations: {exclude_stations}")

        self.UPPER_LEVELS = cfg.pressure_levels
        print(f"Station Info ({len(self.station_metadata)} stations)")
    
    

    def load_dataset_from_csv(self, csv_path: str, start_date: str = None, end_date: str = None) -> tuple:
        """Reads ERA5 CSV and prepares for model."""
        print(f"Reading: {csv_path}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dosya bulunamadı: {csv_path}")

        df = pd.read_csv(csv_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['station_id'] = df['station_id'].astype(str)

        if start_date or end_date:
            original_count = len(df)

            if start_date:
                start_dt = pd.to_datetime(start_date)
                df = df[df['datetime'] >= start_dt]
                print(f"Start date: {start_date}")

            if end_date:
                end_dt = pd.to_datetime(end_date)
                df = df[df['datetime'] <= end_dt]
                print(f"End date: {end_date}")

            filtered_count = original_count - len(df)
            print(f"Filtered {filtered_count} rows. Remaning: {len(df)}")

        active_station_ids = self.station_metadata['station_id'].tolist()
        df = df[df['station_id'].isin(active_station_ids)].copy()

        if 'level_type' in df.columns:
            mask = (df['pressure'].isin(self.UPPER_LEVELS)) | (df['level_type'] == 'surface')
            df = df[mask].copy()
        else:
            print("Warning: 'level_type' column missing.")
            df = df[df['pressure'].isin(self.UPPER_LEVELS)].copy()

        df = self._add_station_metadata(df)
        df = self._create_grid_skeleton(df)

        print(f"Data loaded: {len(df)} rows")
        print(f"Stations: {df['station_id'].nunique()}")
        print(f"Date range: {df['datetime'].min()} - {df['datetime'].max()}")

        return df, self.station_metadata

    def _add_station_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [col for col in ['lat', 'lon', 'elevation', 'name'] if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        df = df.merge(
            self.station_metadata[['station_id', 'name', 'lat', 'lon', 'elevation']],
            on='station_id',
            how='left'
        )
        return df

    def _create_grid_skeleton(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Aligning data to grid...")

        dates = df['datetime'].unique()
        stations = df['station_id'].unique()

        index = pd.MultiIndex.from_product(
            [stations, dates, self.UPPER_LEVELS],
            names=['station_id', 'datetime', 'pressure']
        )
        skeleton = pd.DataFrame(index=index).reset_index()
        skeleton['level_type'] = 'standard'

        if 'level_type' in df.columns:
            surface_df = df[df['level_type'] == 'surface'].copy()
            standard_df = df[df['level_type'] != 'surface'].copy()
        else:
            surface_df = pd.DataFrame()
            standard_df = df.copy()

        merged_standard = pd.merge(
            skeleton, standard_df,
            on=['station_id', 'datetime', 'pressure'],
            how='left', suffixes=('', '_y')
        )

        for col in merged_standard.columns:
            if col.endswith('_y'):
                base_col = col[:-2]
                if base_col in merged_standard.columns:
                    merged_standard[base_col] = merged_standard[base_col].fillna(merged_standard[col])
                merged_standard.drop(columns=[col], inplace=True)

        merged_standard['level_type'] = 'standard'

        if not surface_df.empty:
            final_df = pd.concat([merged_standard, surface_df], ignore_index=True)
        else:
            print("No surface data found.")
            surface_skeleton = pd.DataFrame(
                list(pd.MultiIndex.from_product([stations, dates])),
                columns=['station_id', 'datetime']
            )
            surface_skeleton['level_type'] = 'surface'
            surface_skeleton['pressure'] = np.nan
            final_df = pd.concat([merged_standard, surface_skeleton], ignore_index=True)

        for sid in stations:
            mask = final_df['station_id'] == sid
            meta = self.station_metadata[self.station_metadata['station_id'] == sid]
            if len(meta) > 0:
                final_df.loc[mask, 'lat'] = meta.iloc[0]['lat']
                final_df.loc[mask, 'lon'] = meta.iloc[0]['lon']
                final_df.loc[mask, 'elevation'] = meta.iloc[0]['elevation']
                final_df.loc[mask, 'name'] = meta.iloc[0]['name']

        return final_df.sort_values(by=['datetime', 'station_id', 'pressure'], ascending=[True, True, False])


class RadiosondeGraphBuilder:
    """
    Constructs 3D Spatio-Temporal Graph from Radiosonde data.
    """

    def __init__(
        self,
        station_metadata: pd.DataFrame,
        pressure_levels: List[float] = None,
        max_horizontal_distance: float = 10000,
        temporal_window: int = 1,
        use_level_normalization: bool = True,
        include_surface: bool = False  
    ):
        self.station_metadata = station_metadata
        self.use_level_normalization = use_level_normalization
        self.include_surface = include_surface 

        if pressure_levels is None:
            self.pressure_levels = cfg.pressure_levels
        else:
            self.pressure_levels = sorted(pressure_levels, reverse=True)

        self.max_horizontal_distance = max_horizontal_distance
        self.temporal_window = temporal_window

        # Station-Aware Seviye Hesaplama
        self.STATION_VALID_LEVELS = {}

        for idx, row in self.station_metadata.iterrows():
            sid = str(row['station_id'])
            elevation = float(row['elevation'])
            levels = self._calculate_levels_by_elevation(elevation)
            self.STATION_VALID_LEVELS[sid] = levels

        self.feature_means = None
        self.feature_stds = None
        self.level_stats = {}

        self.feature_columns = [
            'temperature', 'relative_humidity', 'wind_speed',
            'wd_sin', 'wd_cos', 'geopotential'
        ]

        self._build_station_distance_matrix()
        
        surface_status = "INCLUDED" if include_surface else "EXCLUDED"
        print(f"Graph Builder initialized (Norm: {use_level_normalization}, Surface: {surface_status})")

    def _calculate_levels_by_elevation(self, elevation: float) -> List[float]:
        levels = list(self.pressure_levels)
        if elevation > 100 and 1000 in levels:
            levels.remove(1000)
        if elevation > 1450 and 850 in levels:
            levels.remove(850)
        if elevation > 2900 and 700 in levels:
            levels.remove(700)
        return sorted(levels, reverse=True)

    def _build_station_distance_matrix(self):
        n_stations = len(self.station_metadata)
        self.distance_matrix = np.zeros((n_stations, n_stations))
        self.bearing_matrix = np.zeros((n_stations, n_stations))

        for i in range(n_stations):
            for j in range(n_stations):
                if i != j:
                    lat1, lon1 = self.station_metadata.iloc[i][['lat', 'lon']]
                    lat2, lon2 = self.station_metadata.iloc[j][['lat', 'lon']]
                    self.distance_matrix[i, j] = self._haversine_distance(lat1, lon1, lat2, lon2)
                    self.bearing_matrix[i, j] = self._calculate_bearing(lat1, lon1, lat2, lon2)

    @staticmethod
    def _haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))

    @staticmethod
    def _calculate_bearing(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        return np.arctan2(x, y)

    def _normalize_by_level(self, X_df, pressure_array, level_type_array, external_stats=None):
        """Level-based normalization"""
        normalized = X_df.copy()
        level_stats = {}
        
        # Surface dahilse listeye ekle
        if self.include_surface:
            unique_levels = ['surface'] + list(self.pressure_levels)
        else:
            unique_levels = list(self.pressure_levels)

        for level in unique_levels:
            if level == 'surface':
                mask = (level_type_array == 'surface')
                level_key = 'surface'
            else:
                mask = (np.abs(pressure_array - level) < 1.0) & (level_type_array != 'surface')
                level_key = f"p_{int(level)}"

            if mask.sum() == 0:
                continue

            level_data = X_df[mask]

            if external_stats is None:
                means = level_data.mean(numeric_only=True).values.astype(np.float32)
                stds = level_data.std(numeric_only=True).values.astype(np.float32)
                stds[stds == 0] = 1.0
                stds[np.isnan(stds)] = 1.0
                means[np.isnan(means)] = 0.0
                level_stats[level_key] = {'means': means, 'stds': stds, 'count': mask.sum()}
            else:
                if level_key in external_stats:
                    means = external_stats[level_key]['means']
                    stds = external_stats[level_key]['stds']
                else:
                    means = np.zeros(6, dtype=np.float32)
                    stds = np.ones(6, dtype=np.float32)
                level_stats[level_key] = external_stats.get(level_key, {'means': means, 'stds': stds})

            normalized.iloc[mask] = (level_data - means) / stds

        return normalized, level_stats

    def build_graph_from_observations(self, observations, temporal_grouping='12h', external_stats=None):
        """Creates PyG-compatible graph data from DataFrame."""
        print(f"Building graph...")

        observations = observations.copy()
        observations['time_group'] = pd.to_datetime(observations['datetime']).dt.floor(temporal_grouping)
        time_steps = sorted(observations['time_group'].unique())

        print(f"Total Time Steps: {len(time_steps)}")

        node_features = []
        node_metadata = {
            'station_id': [], 'pressure': [], 'lat': [], 'lon': [],
            'datetime': [], 'time_idx': [], 'level_type': []
        }
        node_mapping = {}
        node_idx = 0

        for time_idx, time in enumerate(time_steps):
            time_data = observations[observations['time_group'] == time]

            for station_idx, station_row in self.station_metadata.iterrows():
                sid = str(station_row['station_id'])
                elevation = station_row['elevation']
                st_data = time_data[time_data['station_id'] == sid]

                if self.include_surface:
                    s_data = st_data[st_data['level_type'] == 'surface']
                    estimated_surface_p = 1013.25 * (1 - 2.25577e-5 * elevation)**5.25588

                    if len(s_data) > 0:
                        row = s_data.iloc[0]
                        wd = row.get('wind_direction', np.nan)
                        wd_sin = np.sin(np.deg2rad(wd)) if pd.notna(wd) else np.nan
                        wd_cos = np.cos(np.deg2rad(wd)) if pd.notna(wd) else np.nan
                        features = [
                            row.get('temperature', np.nan),
                            row.get('relative_humidity', np.nan),
                            row.get('wind_speed', np.nan),
                            wd_sin, wd_cos,
                            row.get('geopotential', elevation)
                        ]
                        actual_p = row.get('pressure', estimated_surface_p)
                        if pd.isna(actual_p): actual_p = estimated_surface_p
                    else:
                        features = [np.nan] * 6
                        actual_p = estimated_surface_p

                    node_features.append(features)
                    node_metadata['station_id'].append(sid)
                    node_metadata['pressure'].append(actual_p)
                    node_metadata['lat'].append(station_row['lat'])
                    node_metadata['lon'].append(station_row['lon'])
                    node_metadata['datetime'].append(time)
                    node_metadata['time_idx'].append(time_idx)
                    node_metadata['level_type'].append('surface')
                    node_mapping[(sid, 'SURFACE', time_idx)] = node_idx
                    node_idx += 1

                # Upper Air Levels (her zaman dahil)
                valid_levels = self.STATION_VALID_LEVELS.get(sid, self.pressure_levels)

                for pressure in valid_levels:
                    p_data = st_data[(st_data['pressure'] - pressure).abs() < 1.0]

                    if len(p_data) > 0:
                        row = p_data.iloc[0]
                        wd = row.get('wind_direction', np.nan)
                        wd_sin = np.sin(np.deg2rad(wd)) if pd.notna(wd) else np.nan
                        wd_cos = np.cos(np.deg2rad(wd)) if pd.notna(wd) else np.nan
                        features = [
                            row.get('temperature', np.nan),
                            row.get('relative_humidity', np.nan),
                            row.get('wind_speed', np.nan),
                            wd_sin, wd_cos,
                            row.get('geopotential', np.nan)
                        ]
                    else:
                        features = [np.nan] * 6

                    node_features.append(features)
                    node_metadata['station_id'].append(sid)
                    node_metadata['pressure'].append(pressure)
                    node_metadata['lat'].append(station_row['lat'])
                    node_metadata['lon'].append(station_row['lon'])
                    node_metadata['datetime'].append(time)
                    node_metadata['time_idx'].append(time_idx)
                    node_metadata['level_type'].append('air')
                    node_mapping[(sid, pressure, time_idx)] = node_idx
                    node_idx += 1

        print(f"Created {node_idx} nodes.")
        node_features = np.array(node_features, dtype=np.float32)
        node_metadata['time_steps'] = time_steps

        # Normalizasyon
        X_df = pd.DataFrame(node_features, columns=self.feature_columns)
        pressure_array = np.array(node_metadata['pressure'])
        level_type_array = np.array(node_metadata['level_type'])
        station_id_array = np.array(node_metadata['station_id'])

        if self.use_level_normalization:
            if external_stats is None:
                print("Calculating stats (TRAIN)...")
                X_norm_df, self.level_stats = self._normalize_by_level(
                    X_df, pressure_array, level_type_array, external_stats=None
                )
                self.feature_means = X_df.mean(numeric_only=True).values.astype(np.float32)
                self.feature_stds = X_df.std(numeric_only=True).values.astype(np.float32)
                self.feature_stds[self.feature_stds == 0] = 1.0
            else:
                print("Using external stats (TEST)...")
                level_external = external_stats.get('level_stats', {})
                X_norm_df, self.level_stats = self._normalize_by_level(
                    X_df, pressure_array, level_type_array, external_stats=level_external
                )
                self.feature_means = external_stats.get('means', np.zeros(6))
                self.feature_stds = external_stats.get('stds', np.ones(6))
        else:
            if external_stats is None:
                self.feature_means = X_df.mean(numeric_only=True).values.astype(np.float32)
                self.feature_stds = X_df.std(numeric_only=True).values.astype(np.float32)
                self.feature_stds[self.feature_stds == 0] = 1.0
            else:
                self.feature_means = external_stats['means']
                self.feature_stds = external_stats['stds']
            X_norm_df = (X_df - self.feature_means) / self.feature_stds

        X = torch.tensor(X_norm_df.values, dtype=torch.float32)

        nan_count = torch.isnan(X).sum().item()
        total_values = X.numel()
        print(f"NaN Ratio: {nan_count}/{total_values} ({nan_count/total_values*100:.2f}%)")

        # Kenar oluşturma
        edges = self._build_all_edges(node_mapping, time_steps)
        pos_info = self._build_positional_info(node_metadata)

        scaling_stats = {
            'means': self.feature_means,
            'stds': self.feature_stds,
            'level_stats': self.level_stats,
            'use_level_normalization': self.use_level_normalization
        }

        return {
            'x': X,
            'edge_indices': edges['indices'],
            'edge_attrs': edges['attrs'],
            'pos_info': pos_info,
            'node_metadata': node_metadata,
            'node_mapping': node_mapping,
            'time_steps': time_steps,
            'scaling_stats': scaling_stats,
            'station_metadata': self.station_metadata
        }

    def _build_all_edges(self, node_mapping, time_steps):
        """Constructs all edge types"""
        print("Creating edges...")
        v_edges, v_attrs = [], []
        h_edges, h_attrs = [], []
        t_edges, t_attrs = [], []

        station_ids = self.station_metadata['station_id'].astype(str).tolist()

        for t in range(len(time_steps)):
            for sid in station_ids:
                valid_levels = self.STATION_VALID_LEVELS.get(sid, self.pressure_levels)

                # Surface dahilse edge listesine ekle
                if self.include_surface:
                    all_levels = ['SURFACE'] + valid_levels
                else:
                    all_levels = valid_levels

                # Vertical edges
                for i in range(len(all_levels) - 1):
                    level1, level2 = all_levels[i], all_levels[i + 1]
                    key1, key2 = (sid, level1, t), (sid, level2, t)

                    if key1 in node_mapping and key2 in node_mapping:
                        u, v = node_mapping[key1], node_mapping[key2]
                        
                        # Basınç farkı hesapla
                        if level1 == 'SURFACE':
                            p1 = 1013.25  # Yaklaşık surface pressure
                        else:
                            p1 = level1
                        if level2 == 'SURFACE':
                            p2 = 1013.25
                        else:
                            p2 = level2
                        
                        p_diff = abs(np.log(p1) - np.log(p2))
                        v_edges.extend([[u, v], [v, u]])
                        v_attrs.extend([[p_diff], [p_diff]])

            # Horizontal edges (aynı seviye, farklı istasyonlar)
            for i, sid1 in enumerate(station_ids):
                for j, sid2 in enumerate(station_ids):
                    if i >= j:
                        continue
                    dist = self.distance_matrix[i, j]
                    if dist > self.max_horizontal_distance:
                        continue

                    levels1 = self.STATION_VALID_LEVELS.get(sid1, self.pressure_levels)
                    levels2 = self.STATION_VALID_LEVELS.get(sid2, self.pressure_levels)
                    common_levels = set(levels1) & set(levels2)

                    for pressure in common_levels:
                        key1, key2 = (sid1, pressure, t), (sid2, pressure, t)
                        if key1 in node_mapping and key2 in node_mapping:
                            u, v = node_mapping[key1], node_mapping[key2]
                            bearing = self.bearing_matrix[i, j]
                            # B->A bearing = A->B bearing + pi (mod 2pi).
                            # Eski kod -bearing veriyordu (yansima, ters yon degil) —
                            # yatay edge asymmetric spatial info'yu yanlis temsil ediyordu.
                            bearing_rev = bearing + np.pi
                            if bearing_rev > np.pi:
                                bearing_rev -= 2 * np.pi
                            h_edges.extend([[u, v], [v, u]])
                            h_attrs.extend([[dist / 1000, bearing], [dist / 1000, bearing_rev]])

        # Temporal edges
        for sid in station_ids:
            valid_levels = self.STATION_VALID_LEVELS.get(sid, self.pressure_levels)
            
            if self.include_surface:
                all_levels = ['SURFACE'] + valid_levels
            else:
                all_levels = valid_levels

            for pressure in all_levels:
                for t in range(len(time_steps)):
                    for step in range(1, self.temporal_window + 1):
                        if t + step < len(time_steps):
                            key1 = (sid, pressure, t)
                            key2 = (sid, pressure, t + step)
                            if key1 in node_mapping and key2 in node_mapping:
                                u, v = node_mapping[key1], node_mapping[key2]
                                # SONRA uzaktakini daha az dinle
                                diff = (time_steps[t + step] - time_steps[t]).total_seconds() / 3600
                                diff_normalized = diff / 12.0  # 12 saat = 1.0  24 saat = 2.0 ...
                                t_edges.extend([[u, v], [v, u]])
                                t_attrs.extend([[diff_normalized], [diff_normalized]])

        indices, attrs = {}, {}
        if v_edges:
            indices['vertical'] = torch.tensor(v_edges, dtype=torch.long).t()
            attrs['vertical'] = torch.tensor(v_attrs, dtype=torch.float)
            print(f" Vertical: {len(v_edges)} edges")
        if h_edges:
            indices['horizontal'] = torch.tensor(h_edges, dtype=torch.long).t()
            attrs['horizontal'] = torch.tensor(h_attrs, dtype=torch.float)
            print(f" Horizontal: {len(h_edges)} edges")
        if t_edges:
            indices['temporal'] = torch.tensor(t_edges, dtype=torch.long).t()
            attrs['temporal'] = torch.tensor(t_attrs, dtype=torch.float)
            print(f" Temporal: {len(t_edges)} edges")

        return {'indices': indices, 'attrs': attrs}

    def _build_positional_info(self, node_metadata):
        pres = torch.tensor(node_metadata['pressure'], dtype=torch.float).unsqueeze(1) / 1000.0
        lat = torch.tensor(node_metadata['lat'], dtype=torch.float).unsqueeze(1) / 90.0
        lon = torch.tensor(node_metadata['lon'], dtype=torch.float).unsqueeze(1) / 180.0

        time_feats = []
        for dt in node_metadata['datetime']:
            hour = dt.hour
            time_feats.append([np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)])

        return {
            'pressure': pres, 'lat': lat, 'lon': lon,
            'time': torch.tensor(time_feats, dtype=torch.float)
        }


def collate_graph_windows(batch):
    if len(batch) == 0:
        return {}

    keys = batch[0].keys()
    collated = {}

    for key in keys:
        if key in ['x', 'target']:
            collated[key] = torch.cat([item[key] for item in batch], dim=0)
        elif key == 'edge_indices':
            edge_dict = {}
            for edge_type in batch[0]['edge_indices'].keys():
                offset = 0
                edge_list = []
                for item in batch:
                    edges = item['edge_indices'][edge_type]
                    if edges.shape[1] > 0:
                        edge_list.append(edges + offset)
                    offset += item['x'].shape[0]
                if edge_list:
                    edge_dict[edge_type] = torch.cat(edge_list, dim=1)
                else:
                    edge_dict[edge_type] = torch.empty((2, 0), dtype=torch.long)
            collated['edge_indices'] = edge_dict
        elif key == 'edge_attrs':
            edge_attr_dict = {}
            for edge_type in batch[0]['edge_attrs'].keys():
                attr_list = [item['edge_attrs'][edge_type] for item in batch if item['edge_attrs'][edge_type].shape[0] > 0]
                if attr_list:
                    edge_attr_dict[edge_type] = torch.cat(attr_list, dim=0)
                else:
                    edge_attr_dict[edge_type] = torch.empty((0, 1), dtype=torch.float)
            collated['edge_attrs'] = edge_attr_dict
        elif key == 'pos_info':
            pos_dict = {pos_key: torch.cat([item['pos_info'][pos_key] for item in batch], dim=0)
                        for pos_key in batch[0]['pos_info'].keys()}
            collated['pos_info'] = pos_dict
        elif key == 'time_steps':
            collated[key] = batch[0][key]
        elif key == 'node_metadata':
            # batch_size: TemporalAttention'in dogru reshape yapabilmesi icin.
            # x shape (B*W*S, h) -> (B, W, S, h); B bilgisi olmadan W ekseni
            # boyunca attention yapan reshape batch'leri birbirine karistirir.
            meta_dict = {'batch_size': len(batch)}
            if batch[0]['node_metadata']:
                for meta_key in batch[0]['node_metadata'].keys():
                    if meta_key == 'time_steps':
                        meta_dict[meta_key] = batch[0]['node_metadata'][meta_key]
                    else:
                        meta_dict[meta_key] = []
                        for item in batch:
                            val = item['node_metadata'][meta_key]
                            if isinstance(val, list):
                                meta_dict[meta_key].extend(val)
                            else:
                                meta_dict[meta_key].append(val)
            collated['node_metadata'] = meta_dict
        else:
            collated[key] = [item[key] for item in batch]

    return collated


class RadiosondeSlidingWindowDataset(Dataset):
    """
    Sliding window dataset for radiosonde graph data.
    
     GÜNCELLEME: Seed desteği eklendi!
    - seed=None: Her çağrıda farklı mask (eğitim)
    - seed=42: Tekrarlanabilir mask (test/karşılaştırma)
    """
    
    def __init__(self, graph_data, window_size=10, mask_ratio=0.15,
                 use_realistic_masking=True, seed=None):
        self.full_x = graph_data['x']
        self.full_edge_indices = graph_data['edge_indices']
        self.full_edge_attrs = graph_data['edge_attrs']
        self.full_pos_info = graph_data['pos_info']
        self.full_node_metadata = graph_data['node_metadata']

        self.all_time_steps = graph_data.get('time_steps', self.full_node_metadata.get('time_steps', []))

        self.window_size = window_size
        self.mask_ratio = mask_ratio
        self.use_realistic_masking = use_realistic_masking
        self.seed = seed

        if use_realistic_masking:
            self.masker = RealisticRadiosondeMasking(mask_ratio=mask_ratio, seed=seed)

        self.total_time_steps = len(self.all_time_steps)
        self.total_nodes = self.full_x.shape[0]
        self.nodes_per_step = self.total_nodes // self.total_time_steps if self.total_time_steps > 0 else 0
        self.num_windows = max(0, self.total_time_steps - self.window_size + 1)

        # Global deterministic mask: seed verildiyse __init__'te bir kez hesapla,
        # her __getitem__ window slice'ini ayni mask'tan alir. Bu sayede overlap'lerdeki
        # ayni fiziksel node ayni NaN pattern'e sahip olur — post-processing dedup icin sart.
        # seed=None ise her __getitem__ farkli mask uygular (training augmentation).
        #
        # RNG secimi: numpy legacy Mersenne Twister (np.random.RandomState). DL (M07),
        # SAITS (M08) ve Statistical (M06) hepsi `np.random.seed(seed); np.random.rand()`
        # kullaniyor — RandomState(seed).rand() ile birebir AYNI mask pozisyonlari uretir.
        # Boylece 4 model ailesi de birebir ayni noktalari degerlendirir.
        if not use_realistic_masking and seed is not None:
            rng = np.random.RandomState(int(seed))
            np_mask = rng.rand(*self.full_x.shape) < mask_ratio
            self.global_random_mask = torch.from_numpy(np_mask)
        else:
            self.global_random_mask = None

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start_time_idx = idx
        end_time_idx = idx + self.window_size

        start_node = start_time_idx * self.nodes_per_step
        end_node = end_time_idx * self.nodes_per_step

        x_window = self.full_x[start_node:end_node].clone()

        pos_info_window = {k: v[start_node:end_node] for k, v in self.full_pos_info.items()}

        edge_indices_window, edge_attrs_window = {}, {}
        for edge_type, edge_index in self.full_edge_indices.items():
            src, dst = edge_index
            mask = (src >= start_node) & (src < end_node) & (dst >= start_node) & (dst < end_node)
            if mask.sum() > 0:
                edge_indices_window[edge_type] = edge_index[:, mask] - start_node
                edge_attrs_window[edge_type] = self.full_edge_attrs[edge_type][mask]
            else:
                edge_indices_window[edge_type] = torch.empty((2, 0), dtype=torch.long)
                edge_attrs_window[edge_type] = torch.empty((0, 1), dtype=torch.float)

        window_time_steps = self.all_time_steps[start_time_idx:end_time_idx]

        node_metadata_window = {'time_steps': window_time_steps}
        for k, v in self.full_node_metadata.items():
            if k == 'time_steps':
                continue
            elif isinstance(v, list) and len(v) == self.total_nodes:
                node_metadata_window[k] = v[start_node:end_node]
            else:
                node_metadata_window[k] = v

        target = x_window.clone()

        if self.use_realistic_masking:
            try:
                #  YENİ: window_idx'i geçir (her pencere için farklı ama tekrarlanabilir mask)
                mask = self.masker.apply_masking(x_window, node_metadata_window, 
                                                  self.nodes_per_step, window_idx=idx)
                x_window = x_window.clone()
                x_window[mask] = float('nan')
            except:
                # Fallback: random mask
                if self.seed is not None:
                    np.random.seed(self.seed + idx)
                    torch.manual_seed(self.seed + idx)  # ← PyTorch seed
                rand_mask = torch.rand_like(x_window) < self.mask_ratio
                x_window[rand_mask] = float('nan')
        else:
            # Random masking
            if self.global_random_mask is not None:
                # Deterministic global mask: ayni fiziksel node, ayni mask.
                # Bu overlap-eden window'larda post-processing dedup icin sart.
                rand_mask = self.global_random_mask[start_node:end_node]
            else:
                # seed=None (training): her epoch farkli mask (augmentation)
                rand_mask = torch.rand_like(x_window) < self.mask_ratio
            x_window[rand_mask] = float('nan')

        return {
            'x': x_window, 'target': target,
            'edge_indices': edge_indices_window, 'edge_attrs': edge_attrs_window,
            'pos_info': pos_info_window, 'time_steps': window_time_steps,
            'node_metadata': node_metadata_window
        }
