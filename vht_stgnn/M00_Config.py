import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from dataclasses import dataclass, field
from typing import List, Literal

# Environment detection. On Kaggle, /kaggle/input is read-only and the
# dataset must be added separately; outputs go under /kaggle/working.
# Locally everything lives next to this file.
IS_KAGGLE = os.path.exists('/kaggle/input')
if IS_KAGGLE:
    _DATASET_SLUG = os.environ.get('RADIOSONDE_DATASET', 'radiosonde-gnn-data')
    DATA_DIR    = f'/kaggle/input/{_DATASET_SLUG}'
    OUTPUT_BASE = '/kaggle/working'
else:
    DATA_DIR    = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_BASE = DATA_DIR

STATIONS_FILE = "stations.json"
ERA5_CSV_FILE = "DATA/ERA5_Land_plus_Level_Data.csv"
IGRA_CSV_FILE = "DATA/Turkiye_IGRA_Dataset_2006-11-30_2021-05-18.csv"


# Model Settings


@dataclass
class VHTGNNConfig:

    # Model Architecture
    hidden_dim: int = 64
    num_gnn_layers: int = 1
    dropout: float = 0.1
    aggregator: str = "gcn"  # 'mean', 'gcn', 'pool', 'lstm'
    
    # Data Window
    dataset_window_size: int = 3
    
    # Training Settings
    batch_size: int = 16
    learning_rate: float = 0.001
    num_epochs: int = 200  
    patience: int = 15
    

@dataclass
class VanillaGraphSage:

    # Model Architecture
    hidden_dim: int = 64
    num_gnn_layers: int = 1
    dropout: float = 0.1
    aggregator: str = "gcn"  # 'mean', 'gcn', 'pool', 'lstm'
    
    # Data Window
    dataset_window_size: int = 3
    
    # Training Settings
    batch_size: int = 16
    learning_rate: float = 0.001
    num_epochs: int = 200
    patience: int = 15


@dataclass
class FlatGraphSage:
    # True flat baseline: V/H/T edges merged into one homogeneous graph,
    # standard GraphSAGE conv, no multi-relational separation or fusion.
    # Hyperparameters mirror VanillaGraphSage for a fair comparison.
    hidden_dim: int = 64
    num_gnn_layers: int = 1
    dropout: float = 0.1
    dataset_window_size: int = 3
    batch_size: int = 16
    learning_rate: float = 0.001
    num_epochs: int = 200
    patience: int = 15

@dataclass
class MultiscaleGraphSageConfig:

    # Model Architecture
    hidden_dim: int = 64
    num_gnn_layers: int = 1
    dropout: float = 0.5
    scales: List[int] = field(default_factory=lambda: [1, 2, 4])
    aggregator_types: List[str] = field(default_factory=lambda: ['mean', 'max', 'min'])
    
    # Data Window
    dataset_window_size: int = 3
    
    # Training Settings
    batch_size: int = 16
    learning_rate: float = 0.001
    num_epochs: int = 200
    patience: int = 15


@dataclass
class GATConfig:

    # Model Architecture
    hidden_dim: int = 64        # Must be divisible by heads
    num_gnn_layers: int = 1
    dropout: float = 0.6
    heads: int = 4              # Number of attention heads
    concat: bool = True         # Concatenate head outputs (True) or average them (False)
    
    # Data Window
    dataset_window_size: int = 3
    
    # Training Settings
    batch_size: int = 16
    learning_rate: float = 0.001
    num_epochs: int = 200
    patience: int = 15


@dataclass
class MPNNConfig:

    # Model Architecture
    hidden_dim: int = 64
    num_gnn_layers: int = 3    # 3-hop receptive field; other GNN variants use 1 layer
    dropout: float = 0.1
    edge_dim: int = 1          # Edge feature dimension
    aggr: str = "add"          # Message aggregation: 'add', 'mean', 'max'
    
    # Data Window
    dataset_window_size: int = 3

    # Training
    batch_size: int = 16
    learning_rate: float = 0.001
    num_epochs: int = 200
    patience: int = 15


# Ablation configs: each mirrors the full VHT-GNN settings; the only
# difference is which component is removed (controlled by model_type
# in M03_Model or by use_level_normalization in graph construction).

@dataclass
class VHTGNNNoTemporalConfig:
    hidden_dim: int = 64
    num_gnn_layers: int = 1
    dropout: float = 0.1
    dataset_window_size: int = 3
    batch_size: int = 16
    learning_rate: float = 0.001
    num_epochs: int = 200
    patience: int = 15


@dataclass
class VHTGNNNoGatingConfig:
    hidden_dim: int = 64
    num_gnn_layers: int = 1
    dropout: float = 0.1
    dataset_window_size: int = 3
    batch_size: int = 16
    learning_rate: float = 0.001
    num_epochs: int = 200
    patience: int = 15


@dataclass
class VHTGNNFixedFusionConfig:
    hidden_dim: int = 64
    num_gnn_layers: int = 1
    dropout: float = 0.1
    dataset_window_size: int = 3
    batch_size: int = 16
    learning_rate: float = 0.001
    num_epochs: int = 200
    patience: int = 15


@dataclass
class VHTGNNNoVerticalConfig:
    hidden_dim: int = 64
    num_gnn_layers: int = 1
    dropout: float = 0.1
    dataset_window_size: int = 3
    batch_size: int = 16
    learning_rate: float = 0.001
    num_epochs: int = 200
    patience: int = 15


@dataclass
class VHTGNNNoHorizontalConfig:
    hidden_dim: int = 64
    num_gnn_layers: int = 1
    dropout: float = 0.1
    dataset_window_size: int = 3
    batch_size: int = 16
    learning_rate: float = 0.001
    num_epochs: int = 200
    patience: int = 15


@dataclass
class VHTGNNGlobalNormConfig:
    # Full VHT-GNN, but graph built with use_level_normalization=False
    hidden_dim: int = 64
    num_gnn_layers: int = 1
    dropout: float = 0.1
    dataset_window_size: int = 3
    batch_size: int = 16
    learning_rate: float = 0.001
    num_epochs: int = 200
    patience: int = 15


@dataclass
class VanillaGraphSAGEGlobalNormConfig:
    # Multi-Relational GraphSAGE, but graph built with use_level_normalization=False
    hidden_dim: int = 64
    num_gnn_layers: int = 1
    dropout: float = 0.1
    dataset_window_size: int = 3
    batch_size: int = 16
    learning_rate: float = 0.001
    num_epochs: int = 200
    patience: int = 15


#  Common Settings


@dataclass
class MaskingConfig:

    random_seed: int = 42
    mask_ratio: float = 0.15
    mask_type: str = "random"  # 'random' or 'realistic'
    
    # For multiple ratio testing (optional)
    test_ratios: List[float] = field(default_factory=lambda: [0.15, 0.30, 0.40, 0.50, 0.60])


@dataclass
class DataQualityConfig:
    # Is filtering enabled?
    enable_filtering: bool = True
    
    # If None, only threshold-based filtering is performed
    target_nan_ratio: float = None      
    
    # Minimum time step quality (0-1)
    # Time steps below this value are removed
    min_time_step_quality: float = 0.85
    
    # Minimum number of stations per time step
    # stations.json'da 8 active istasyon var. Grid skeleton her timestamp'te
    # 8'in tamamini yaratir, dolayisiyla n_stations daima 8. 9 verilirse hicbir timestamp
    # esigi gecmez ve filter tum veriyi atar.
    min_stations_per_time: int = 8
    
    # Show detailed output
    verbose: bool = True


@dataclass  
class BaselineConfig:

    # Common Training Settings
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    patience: int = 15
    
    # Model-Specific Hidden Dims
    mlp_hidden_dim: int = 256
    lstm_hidden_dim: int = 128
    cnn_hidden_dim: int = 64
    
    # For sequence models (LSTM, CNN)
    seq_length: int = 5
    
    # Traditional baselines
    idw_power: float = 2.0 
    


@dataclass
class Config:
    # General Paths and Device
    data_dir: str = DATA_DIR
    stations_file: str = STATIONS_FILE    
    save_dir: str = os.path.join(OUTPUT_BASE, "checkpoints")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data Source  
    data_source: str = "IGRA"
    csv_file: str = field(init=False) 

    # Data Filtering and Time
    # start_date: str = "2006-11-29"
    # end_date: str = "2021-05-17"
    
    start_date: str = "2006-11-29"
    end_date: str = "2021-05-17"
    filter_active_stations: bool = True
    exclude_stations: List[str] = field(default_factory=list)
    
    # Standard Pressure Levels (WMO)
    pressure_levels: List[int] = field(default_factory=lambda: [1000, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 10])

    # GNN Data Window (for Graph creation)
    graph_temporal_window: int = 1

    # Buffer between chronological splits (train->val, val->test). 10 days
    # spans roughly two synoptic-system lifetimes, so residual autocorrelation
    # between adjacent partitions decays substantially.
    chronological_gap_days: int = 10

    # Fixed Parameters (Same across all models)
    input_dim: int = 6
    use_physics_informed_loss: bool = False
    use_realistic_masking: bool = False
    
    # Masking, Data Quality, and Baseline Settings
    Masking: MaskingConfig = field(default_factory=MaskingConfig)
    DataQuality: DataQualityConfig = field(default_factory=DataQualityConfig)  
    Baseline: BaselineConfig = field(default_factory=BaselineConfig)

    # Sub-Model Settings
    VHT_GNN: VHTGNNConfig = field(default_factory=VHTGNNConfig)
    VanillaGraphSAGE: VanillaGraphSage = field(default_factory=VanillaGraphSage)
    FlatGraphSAGE: FlatGraphSage = field(default_factory=FlatGraphSage)
    MultiscaleGraphSAGE: MultiscaleGraphSageConfig = field(default_factory=MultiscaleGraphSageConfig)
    Gat: GATConfig = field(default_factory=GATConfig)
    Mpnn: MPNNConfig = field(default_factory=MPNNConfig)

    # Ablation Settings (M2.3)
    VHT_GNN_NoTemporal:      VHTGNNNoTemporalConfig      = field(default_factory=VHTGNNNoTemporalConfig)
    VHT_GNN_NoGating:        VHTGNNNoGatingConfig        = field(default_factory=VHTGNNNoGatingConfig)
    VHT_GNN_FixedFusion:     VHTGNNFixedFusionConfig     = field(default_factory=VHTGNNFixedFusionConfig)
    VHT_GNN_NoVertical:      VHTGNNNoVerticalConfig      = field(default_factory=VHTGNNNoVerticalConfig)
    VHT_GNN_NoHorizontal:    VHTGNNNoHorizontalConfig    = field(default_factory=VHTGNNNoHorizontalConfig)
    VHT_GNN_GlobalNorm:      VHTGNNGlobalNormConfig      = field(default_factory=VHTGNNGlobalNormConfig)
    VanillaGraphSAGE_GlobalNorm: VanillaGraphSAGEGlobalNormConfig = field(default_factory=VanillaGraphSAGEGlobalNormConfig)

    def __post_init__(self):
        # File Path Selection
        if self.data_source == "IGRA":
            self.csv_file = IGRA_CSV_FILE
        elif self.data_source == "ERA5":
            self.csv_file = ERA5_CSV_FILE
        else:
            raise ValueError(f"Invalid data_source: {self.data_source}")

        # Flat-layout fallback: the default csv_file paths begin with "DATA/",
        # which matches the local repo. If the CSV does not exist at that
        # location (e.g. on Kaggle when the dataset was uploaded without a
        # DATA/ subfolder), look for the file directly under data_dir.
        expected_csv = os.path.join(self.data_dir, self.csv_file)
        if not os.path.exists(expected_csv):
            flat_name = os.path.basename(self.csv_file)
            flat_csv  = os.path.join(self.data_dir, flat_name)
            if os.path.exists(flat_csv):
                print(f"Config: CSV not at {self.csv_file}; using flat layout {flat_name}")
                self.csv_file = flat_name

        os.makedirs(self.save_dir, exist_ok=True)
            
    @property
    def data_path(self):
        return os.path.join(self.data_dir, self.csv_file)

    @property
    def stations_path(self):
        return os.path.join(self.data_dir, self.stations_file)

    @property
    def results_dir(self):
        return os.path.join(OUTPUT_BASE, "results")

# Global Config Instance
cfg = Config()

