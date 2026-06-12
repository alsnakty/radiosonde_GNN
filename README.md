# Level-Aware Spatio-Temporal Graph Neural Networks for Radiosonde Data Imputation

Code for the paper "Level-Aware Spatio-Temporal Graph Neural Networks for Radiosonde Data Imputation".

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## About

This repo contains code for filling missing values in radiosonde (weather balloon) observations. The model uses three types of graph edges:

- **Vertical**: Adjacent pressure levels within the same sounding
- **Horizontal**: Stations at the same pressure level  
- **Temporal**: Same station-level pairs across consecutive time steps

Key idea: representing the observations as a multi-relational graph (vertical, horizontal, temporal) combined with level-aware normalization. Edge-conditioned gating and learnable edge-type fusion are lightweight refinements.

## Installation

```bash
git clone https://github.com/alsnakty/radiosonde_GNN.git
cd radiosonde_GNN
pip install -r requirements.txt
```

Requires Python 3.8+, PyTorch 2.0+, PyTorch Geometric 2.4+.

## Project Structure

```
radiosonde_GNN/
├── data_collection/                  # IGRA raw data download & preprocessing
│   ├── 01_convert_igra_to_csv.py
│   ├── 02_merge_stations_data.py
│   ├── 03_analyze_dataset.py
│   ├── README.md
│   └── stations.json
├── vht_stgnn/                         # modeling pipeline
│   ├── M00_Config.py                 # Configuration
│   ├── M01_Utils.py                  # Station metadata loader
│   ├── M02_DataLoading.py            # Data loading and graph construction
│   ├── M03_Model.py                  # VHT-GNN and GNN baselines
│   ├── M04_Training.py               # Training loop and physics-informed loss
│   ├── M05_Visualization.py          # Prediction plots
│   ├── M06_BaselineComparison.py     # Statistical baselines (IDW, Linear)
│   ├── M07_DeepLearningBaselines.py  # LSTM, CNN, MLP
│   ├── M08_TransformerBaselines.py   # SAITS (requires PyPOTS)
│   ├── M11_MultiRatioTest_v3.py      # Robustness vs missing ratio
│   ├── M12_PhysicsValidation.py      # Hydrostatic-consistency check
│   ├── M13_ComplexityProfiling.py    # Parameter count and timing
│   ├── M14_SeedAggregation.py        # Multi-seed mean/std and significance tests
│   ├── M15_VerticalProfileFigure.py  # Vertical-profile MAE figure
│   ├── M16_Table1.py                 # Main R-squared table generator
│   ├── M17_ResultsTables.py          # Secondary and significance table generators
│   ├── M98_CompareResults.py         # Results aggregation and comparison
│   ├── M99_MAIN.py                   # Orchestrator / entry point
│   └── stations.json
├── LICENSE
├── README.md
└── requirements.txt
```

## Usage

The pipeline uses module-relative imports, so run it from inside `vht_stgnn/`:

```bash
cd vht_stgnn

# Train and evaluate. Either edit the RUN_* flags at the top of M99_MAIN.py
# and run with no arguments, or use the CLI:
python M99_MAIN.py --model vht_gnn --seed 42 123 456 789 2024
python M99_MAIN.py --model saits             # SAITS (requires PyPOTS)
python M99_MAIN.py --model statistical       # IDW + Linear baselines

# Aggregate results; multi-seed mean/std and significance tests
python M98_CompareResults.py
python M14_SeedAggregation.py

# Robustness across missing ratios
python M11_MultiRatioTest_v3.py
```

**Result directories.** Training writes per-model outputs to `results/<model>/seed_<n>/`.
The aggregation, table, and figure scripts (`M98`, `M14`–`M17`) read from
`results_canonical/`, the directory into which all per-model results are gathered.
For a single-machine run, copy or rename `results/` to `results_canonical/` before
running them:

```bash
cp -r results results_canonical   # or: mv results results_canonical
```

CLI `--model` options include: `vht_gnn`, `flat_graphsage`, `multiscale_graphsage`,
`gat`, `mpnn`, the ablations (`vht_gnn_no_vertical`, `vht_gnn_no_gating`, ...),
`lstm`, `cnn`, `mlp`, `saits`, and `statistical`. Without arguments, the
`RUN_*` flags at the top of `M99_MAIN.py` control which models run.

## Data Format

Input CSV columns:

| Column | Description |
|--------|-------------|
| `datetime` | Timestamp |
| `station_id` | Station ID |
| `pressure` | Pressure level (hPa) |
| `temperature` | Temperature (°C) |
| `relative_humidity` | RH (%) |
| `wind_speed` | Wind speed (m/s) |
| `wind_direction` | Wind direction (°) |
| `geopotential` | Geopotential height (m) |

Pressure levels: 1000, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 10 hPa

See [`data_collection/`](data_collection/) for IGRA raw data download and preprocessing scripts.

## Architecture

```
Input (X) → Positional Encoding & Input Projection
                           ↓
         ┌─────────────────┼─────────────────┐
         ↓                 ↓                 ↓
    σ(Edge_V)         σ(Edge_H)         σ(Edge_T)
         ↓                 ↓                 ↓
  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
  │   Vertical   │ │  Horizontal  │ │   Temporal   │
  │  Gated Conv  │ │  Gated Conv  │ │  Gated Conv  │
  └──────────────┘ └──────────────┘ └──────────────┘
         └─────────────────┼─────────────────┘
                           ↓
                  Adaptive Fusion (Σ αk)
                           ↓
               Temporal Attention (Masked)
                           ↓
                    Output Projection
                           ↓
                    Imputed Data (X̂)
```

## Results

Output structure:
```
results/
├── vht_gnn/
│   ├── config.json
│   ├── metrics.csv
│   └── metrics_by_pressure.csv
├── lstm/
└── ...
```

## Citation

This work is currently under review. Until it is published, please cite it as:

```bibtex
@unpublished{aaktay2026vhtgnn,
  title  = {Level-Aware Spatio-Temporal Graph Neural Networks
            for Radiosonde Data Imputation},
  ...
}
```

## License

MIT
