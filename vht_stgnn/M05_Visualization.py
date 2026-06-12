import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class MissingDataConfig:
    """Configuration for missing data simulation."""

    # Whether to add synthetic missing data?
    ADD_SYNTHETIC_MISSING = True

    # Missing rates (0.0 - 1.0)
    MISSING_RATES = {
        'surface': 0.15,      # 15% missing at Surface
        '1000': 0.20,         # 20% missing at 1000 hPa
        '850': 0.10,          # 10% missing at 850 hPa
        '700': 0.08,
        '500': 0.05,
        '400': 0.05,
        '300': 0.05,
        '250': 0.05,
        '200': 0.05,
        '150': 0.08,
        '100': 0.10,
        'default': 0.05       # For other levels
    }

    # Variables  
    MISSING_VARIABLES = {
        'temperature': True,
        'humidity': True,
        'wind_speed': True,
        'wind_direction': True,
        'geopotential': False   
    }

    # Type of missing pattern
    MISSING_PATTERN = 'realistic'  # 'random', 'burst', 'realistic'
    BURST_LENGTH = 3  # Number of consecutive missing values for burst pattern

def analyze_missing_data(df, config=MissingDataConfig):
    """Analyzes original and synthetic missing data patterns."""

    print("="*80)
    print("MISSING DATA ANALİZİ")
    print("="*80)

    # Variable names
    var_columns = ['temperature', 'relative_humidity', 'wind_speed',
                   'wind_direction', 'u_component', 'v_component', 'geopotential_height']
    available_cols = [col for col in var_columns if col in df.columns]

    # 1. ORIGINAL MISSING ANALYSIS
    print("\nORİJİNAL VERİDEKİ MISSING ORANLARI:")
    print("-"*50)

    # Overall missing
    total_missing = df[available_cols].isna().sum().sum()
    total_cells = len(df) * len(available_cols)
    print(f"Toplam Missing: {total_missing:,} / {total_cells:,} ({100*total_missing/total_cells:.2f}%)")

    # Variable-based
    print("\nDeğişken Bazlı:")
    for col in available_cols:
        missing_count = df[col].isna().sum()
        missing_pct = 100 * missing_count / len(df)
        print(f"  {col:<25}: {missing_pct:6.2f}% ({missing_count:,} değer)")

    # Level-based
    print("\nSEVİYE BAZLI MISSING ORANLARI:")
    print("-"*50)

    if 'level_type' in df.columns and 'pressure' in df.columns:
        # Surface
        surface_df = df[df['level_type'] == 'surface']
        if len(surface_df) > 0:
            surface_missing = surface_df[available_cols].isna().mean().mean()
            print(f"{'Surface':<15}: {100*surface_missing:6.2f}% (n={len(surface_df):,})")

        # Pressure levels
        pressure_levels = [1000, 850, 700, 500, 400, 300, 250, 200, 150, 100]
        for pressure in pressure_levels:
            level_df = df[df['pressure'] == pressure]
            if len(level_df) > 0:
                level_missing = level_df[available_cols].isna().mean().mean()
                print(f"{pressure:>4} hPa{'':<10}: {100*level_missing:6.2f}% (n={len(level_df):,})")

    # Station-based
    print("\nİSTASYON BAZLI MISSING ORANLARI:")
    print("-"*50)

    station_missing = df.groupby('station_id')[available_cols].apply(lambda x: x.isna().mean().mean())
    station_counts = df.groupby('station_id').size()

    # Merge with station metadata
    station_info = []
    for station_id in station_missing.index:
        missing_pct = 100 * station_missing[station_id]
        count = station_counts[station_id]

        # Find station name (if metadata exists)
        if 'station_name' in df.columns:
            name = df[df['station_id'] == station_id]['station_name'].iloc[0]
        else:
            name = f"Station_{station_id}"

        station_info.append({
            'station_id': station_id,
            'name': name,
            'missing_pct': missing_pct,
            'n_obs': count
        })

    station_df = pd.DataFrame(station_info).sort_values('missing_pct', ascending=False)

    print("En Çok Missing Olan İstasyonlar:")
    for _, row in station_df.head(5).iterrows():
        print(f"  {row['name']:<20} ({row['station_id']}): {row['missing_pct']:6.2f}% (n={row['n_obs']:,})")

    print("\nEn Az Missing Olan İstasyonlar:")
    for _, row in station_df.tail(5).iterrows():
        print(f"  {row['name']:<20} ({row['station_id']}): {row['missing_pct']:6.2f}% (n={row['n_obs']:,})")

    # Temporal analysis
    if 'datetime' in df.columns:
        print("\nZAMANSAL MISSING PATTERNİ:")
        print("-"*50)

        df['year_month'] = pd.to_datetime(df['datetime']).dt.to_period('M')
        temporal_missing = df.groupby('year_month')[available_cols].apply(lambda x: x.isna().mean().mean())

        # Highest/lowest missing months
        highest_months = temporal_missing.nlargest(3)
        lowest_months = temporal_missing.nsmallest(3)

        print("En Çok Missing Olan Aylar:")
        for month, pct in highest_months.items():
            print(f"  {month}: {100*pct:.2f}%")

        print("\nEn Az Missing Olan Aylar:")
        for month, pct in lowest_months.items():
            print(f"  {month}: {100*pct:.2f}%")

    # 2. SYNTHETIC MISSING SIMULATION
    if config.ADD_SYNTHETIC_MISSING:
        print("\n" + "="*80)
        print("YAPAY MISSING SİMÜLASYONU PLANI")
        print("="*80)

        print(f"\nKonfigürasyon:")
        print(f"  Pattern: {config.MISSING_PATTERN}")
        if config.MISSING_PATTERN == 'burst':
            print(f"  Burst Length: {config.BURST_LENGTH}")

        print(f"\nEklenecek Missing Oranları (Seviye Bazlı):")
        for level, rate in config.MISSING_RATES.items():
            if level != 'default':
                print(f"  {level:<10}: %{100*rate:.0f}")

        print(f"\nEtkilenecek Değişkenler:")
        for var, will_add in config.MISSING_VARIABLES.items():
            if will_add:
                print(f"  + {var}")
            else:
                print(f"  - {var} (korunacak)")

        # Estimated final missing rates
        print(f"\nTAHMİNİ FİNAL MISSING ORANLARI:")
        print("-"*50)

        pressure_levels = [1000, 850, 700, 500, 400, 300, 250, 200, 150, 100]
        for pressure in ['surface'] + pressure_levels:
            if pressure == 'surface':
                original = surface_missing if 'surface_missing' in locals() else 0
                synthetic = config.MISSING_RATES.get('surface', config.MISSING_RATES['default'])
            else:
                level_df = df[df['pressure'] == pressure]
                if len(level_df) > 0:
                    original = level_df[available_cols].isna().mean().mean()
                    synthetic = config.MISSING_RATES.get(str(pressure), config.MISSING_RATES['default'])
                else:
                    continue

            # Final = original + synthetic * (1 - original)
            final = original + synthetic * (1 - original)
            print(f"{str(pressure):>8}: {100*original:5.1f}% -> {100*final:5.1f}% (+{100*synthetic*(1-original):4.1f}%)")

    print("\n" + "="*80)

    return station_df, temporal_missing if 'datetime' in df.columns else None

def add_synthetic_missing(data, metadata, config=MissingDataConfig):
    """Adds synthetic missing data."""

    if not config.ADD_SYNTHETIC_MISSING:
        return data

    print("\nYapay missing ekleniyor...")

    data_with_missing = data.clone()

    # Add missing data for each level
    for i in range(len(data)):
        # Determine level
        if 'level_type' in metadata:
            level_type = metadata['level_type'][i]
            pressure = metadata.get('pressure', [None])[i]

            if level_type == 'surface':
                missing_rate = config.MISSING_RATES.get('surface', config.MISSING_RATES['default'])
            elif pressure:
                missing_rate = config.MISSING_RATES.get(str(int(pressure)), config.MISSING_RATES['default'])
            else:
                missing_rate = config.MISSING_RATES['default']
        else:
            missing_rate = config.MISSING_RATES['default']

        # Add missing based on pattern
        if config.MISSING_PATTERN == 'random':
            # Random missing
            mask = torch.rand(6) < missing_rate

            # Variable-specific control
            if not config.MISSING_VARIABLES.get('temperature', True):
                mask[0] = False
            if not config.MISSING_VARIABLES.get('humidity', True):
                mask[1] = False
            if not config.MISSING_VARIABLES.get('wind_speed', True):
                mask[2] = False
            if not config.MISSING_VARIABLES.get('wind_direction', True):
                mask[3:5] = False
            if not config.MISSING_VARIABLES.get('geopotential', True):
                mask[5] = False

            data_with_missing[i][mask] = float('nan')

        elif config.MISSING_PATTERN == 'burst':
            # Burst pattern - consecutive missing values
            if torch.rand(1) < missing_rate:
                start_idx = torch.randint(0, 6-config.BURST_LENGTH+1, (1,)).item()
                data_with_missing[i, start_idx:start_idx+config.BURST_LENGTH] = float('nan')

        elif config.MISSING_PATTERN == 'realistic':
            # Realistic pattern - certain variables tend to be missing together
            if torch.rand(1) < missing_rate:
                # Temperature and humidity are usually missing together
                if torch.rand(1) < 0.7:
                    data_with_missing[i, 0:2] = float('nan')  # Temp + Humidity
                # Wind components are likely missing together
                if torch.rand(1) < 0.5:
                    data_with_missing[i, 2:5] = float('nan')  # Wind speed + direction

    # Show statistics
    original_missing = torch.isnan(data).float().mean()
    final_missing = torch.isnan(data_with_missing).float().mean()
    print(f"Missing eklendi: {100*original_missing:.1f}% -> {100*final_missing:.1f}%")

    return data_with_missing

def visualize_predictions(predictions, targets, feature_names, save_dir='HTML_Export'):
    """
    Visualizes and saves prediction results.
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    num_features = len(feature_names)
    
    # 1. Scatter Plots (Ground Truth vs Prediction)
    print("Scatter Plotlar oluşturuluyor...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i in range(num_features):
        if i >= len(axes): break
        
        ax = axes[i]
        y_true = targets[:, i]
        y_pred = predictions[:, i]
        
        # NaN cleanup
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0: continue
        
        # Scatter plot (use hexbin or sample scatter for density)
        # If too many points, take a sample
        if len(y_true) > 10000:
            idx = np.random.choice(len(y_true), 10000, replace=False)
            ax.scatter(y_true[idx], y_pred[idx], alpha=0.1, s=1)
        else:
            ax.scatter(y_true, y_pred, alpha=0.3, s=5)
            
        # Ideal line (y=x)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax.set_title(f"{feature_names[i]}")
        ax.set_xlabel("Gerçek")
        ax.set_ylabel("Tahmin")
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scatter_plots.png'), dpi=300)
    plt.close()
    print(f"Scatter plotlar kaydedildi: {os.path.join(save_dir, 'scatter_plots.png')}")

    # 2. Error Distribution (Histogram)
    print("Hata Histogramları oluşturuluyor...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i in range(num_features):
        if i >= len(axes): break
        
        ax = axes[i]
        y_true = targets[:, i]
        y_pred = predictions[:, i]
        
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        errors = y_pred[mask] - y_true[mask]
        
        if len(errors) == 0: continue
        
        sns.histplot(errors, bins=50, kde=True, ax=ax, color='skyblue')
        
        ax.set_title(f"{feature_names[i]} Hata Dağılımı")
        ax.set_xlabel("Hata (Tahmin - Gerçek)")
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_histograms.png'), dpi=300)
    plt.close()
    print(f"Hata histogramları kaydedildi: {os.path.join(save_dir, 'error_histograms.png')}")
