# -*- coding: utf-8 -*-
"""
M98_CompareResults.py - Model Karşılaştırma ve Rapor Oluşturma

save
- csv latex
- png
- Özet raporlar

 
2026
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# 
# AYARLAR
 
RESULTS_DIR = Path("results_canonical")
OUTPUT_DIR = Path("comparison_output")

MODEL_CATEGORIES = {
    'GNN': ['vht_gnn', 'vanilla_graphsage', 'flat_graphsage', 'multiscale_graphsage', 'gat', 'mpnn'],
    'Deep Learning': ['lstm', 'cnn', 'mlp'],
    'Statistical': ['idw', 'linear_temporal', 'linear_vertical', 
                    'linear_spatial', 'linear_combined']
}

# Model görüntü adları (tablolar için)
MODEL_DISPLAY_NAMES = {
    'vht_gnn': 'VHT-GNN',
    'vanilla_graphsage': 'Multi-Relational GraphSAGE',
    'flat_graphsage': 'GraphSAGE',
    'multiscale_graphsage': 'Multiscale-GraphSAGE',
    'gat': 'GAT',
    'mpnn': 'MPNN',
    'lstm': 'LSTM',
    'cnn': 'CNN',
    'mlp': 'MLP',
    'idw': 'IDW',
    'linear_temporal': 'Linear (Temporal)',
    'linear_vertical': 'Linear (Vertical)',
    'linear_spatial': 'Linear (Spatial)',
    'linear_combined': 'Linear (Combined)'
}

# Feature görüntü adları
FEATURE_DISPLAY_NAMES = {
    'temperature': 'Temperature',
    'relative_humidity': 'Relative Humidity',
    'wind_speed': 'Wind Speed',
    'sin_wd': 'Sin(Wind Dir)',
    'cos_wd': 'Cos(Wind Dir)',
    'geopotential': 'Geopotential'
}

FEATURE_UNITS = {
    'temperature': '°C',
    'relative_humidity': '%',
    'wind_speed': 'm/s',
    'sin_wd': '-',
    'cos_wd': '-',
    'geopotential': 'm'
}

# 
# VERİ YÜKLEME
# 

def _load_single_dir(d) -> dict:
    """Tek klasörden (flat ya da tek seed) sonuçları oku. metrics.csv yoksa None."""
    metrics_file = d / "metrics.csv"
    if not metrics_file.exists():
        return None
    metrics_df = pd.read_csv(metrics_file)

    config = {}
    if (d / "config.json").exists():
        with open(d / "config.json", 'r') as f:
            config = json.load(f)

    pressure_df = None
    if (d / "metrics_by_pressure.csv").exists():
        pressure_df = pd.read_csv(d / "metrics_by_pressure.csv")

    history_df = None
    if (d / "training_history.csv").exists():
        history_df = pd.read_csv(d / "training_history.csv")

    return {
        'metrics': metrics_df,
        'metrics_std': None,
        'config': config,
        'pressure_metrics': pressure_df,
        'training_history': history_df,
        'seeds': [],
    }


def _aggregate_seed_results(seed_dirs) -> dict:
    """Multi-seed klasorlerden seedler arasi mean + std hesapla.
    metrics: feature bazinda mean. metrics_std: feature bazinda std.
    pressure_metrics: (feature, pressure) bazinda mean.
    training_history: ilk seedin'ki (epoch sayilari farkli olabilir, agregasyon karmasik).
    config: ilk seedin'ki."""
    metrics_dfs   = []
    pressure_dfs  = []
    history_dfs   = []
    config        = {}
    seeds         = []

    for sd in seed_dirs:
        single = _load_single_dir(sd)
        if single is None:
            continue
        seeds.append(sd.name.replace("seed_", ""))
        metrics_dfs.append(single['metrics'])
        if single['pressure_metrics'] is not None:
            pressure_dfs.append(single['pressure_metrics'])
        if single['training_history'] is not None:
            history_dfs.append(single['training_history'])
        if not config:
            config = single['config']

    if not metrics_dfs:
        return None

    # Aggregate metrics.csv: seedler arasi (feature, unit) bazinda mean/std
    combined = pd.concat(metrics_dfs, ignore_index=True)
    numeric_cols = combined.select_dtypes(include='number').columns.tolist()
    mean_df = combined.groupby('feature', as_index=False)[numeric_cols].mean()
    std_df  = combined.groupby('feature', as_index=False)[numeric_cols].std()
    if 'unit' in combined.columns:
        unit_map = combined.groupby('feature')['unit'].first().to_dict()
        mean_df['unit'] = mean_df['feature'].map(unit_map)
        std_df['unit']  = std_df['feature'].map(unit_map)

    # Aggregate metrics_by_pressure.csv: (feature, pressure) bazinda mean
    pressure_df = None
    if pressure_dfs:
        p_combined = pd.concat(pressure_dfs, ignore_index=True)
        if {'feature', 'pressure'}.issubset(p_combined.columns):
            p_numeric = [c for c in p_combined.select_dtypes(include='number').columns if c != 'pressure']
            pressure_df = p_combined.groupby(['feature', 'pressure'], as_index=False)[p_numeric].mean()
            if 'unit' in p_combined.columns:
                p_unit_map = p_combined.groupby('feature')['unit'].first().to_dict()
                pressure_df['unit'] = pressure_df['feature'].map(p_unit_map)

    history_df = history_dfs[0] if history_dfs else None

    return {
        'metrics': mean_df,
        'metrics_std': std_df,
        'config': config,
        'pressure_metrics': pressure_df,
        'training_history': history_df,
        'seeds': seeds,
    }


def load_all_results() -> dict:
    """Tum model sonuclarini yukle. seed_N alt klasorleri varsa seedler arasi
    mean+std hesaplar; yoksa flat klasoru okur (statistical baseline gibi)."""

    results = {}

    if not RESULTS_DIR.exists():
        print(f" Results klasörü bulunamadı: {RESULTS_DIR}")
        return results

    for model_dir in RESULTS_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        seed_dirs = sorted(model_dir.glob("seed_*"))

        if seed_dirs:
            model_data = _aggregate_seed_results(seed_dirs)
            if model_data is None:
                continue
            print(f" Loaded (multi-seed, {len(model_data['seeds'])}): {model_name}")
        else:
            model_data = _load_single_dir(model_dir)
            if model_data is None:
                continue
            print(f" Loaded: {model_name}")

        results[model_name] = model_data

    return results


# 
# ADİL KARŞILAŞTIRMA KONTROLÜ
# 

def check_fairness(results: dict) -> tuple:
    """
    Adil karşılaştırma kontrolü yap.
    
    Returns:
        (errors, warnings) - Kritik hatalar ve uyarılar
    """
    
    errors = []
    warnings = []
    
    if not results:
        return errors, warnings
    
    # Config'leri topla
    configs = {}
    for model_name, data in results.items():
        configs[model_name] = data.get('config', {})
    
    # 
    # 1. MASK SEED KONTROLÜ (KRİTİK)
    # 
    mask_seeds = {}
    for model, cfg in configs.items():
        data_cfg = cfg.get('data', {})
        if 'mask_seed' in data_cfg:
            mask_seeds[model] = data_cfg['mask_seed']
    
    unique_seeds = set(mask_seeds.values())
    if len(unique_seeds) > 1:
        errors.append(f" KRİTİK: Farklı mask seed kullanılmış!")
        for model, seed in mask_seeds.items():
            errors.append(f"   - {model}: seed={seed}")
        errors.append("   → Karşılaştırma GEÇERSİZ! Tüm modeller aynı seed ile çalıştırılmalı.")
    
    # 
    # 2. MASK RATIO KONTROLÜ (KRİTİK)
    # 
    mask_ratios = {}
    for model, cfg in configs.items():
        data_cfg = cfg.get('data', {})
        if 'mask_ratio' in data_cfg:
            mask_ratios[model] = data_cfg['mask_ratio']
    
    unique_ratios = set(mask_ratios.values())
    if len(unique_ratios) > 1:
        errors.append(f" KRİTİK: Farklı mask ratio kullanılmış!")
        for model, ratio in mask_ratios.items():
            errors.append(f"   - {model}: ratio={ratio}")
        errors.append("   → Karşılaştırma GEÇERSİZ! Tüm modeller aynı mask ratio ile çalıştırılmalı.")
    
    # 
    # 3. EKSİK MODEL KONTROLÜ (UYARI)
    # 
    expected_models = 13  # 5 GNN + 3 DL + 5 Statistical (Kriging hariç)
    actual_models = len(results)
    
    if actual_models < expected_models:
        warnings.append(f" Eksik modeller: {expected_models} modelden {actual_models}'i mevcut")
        
        # Hangileri eksik?
        all_expected = (
            ['vht_gnn', 'vanilla_graphsage', 'multiscale_graphsage', 'gat', 'mpnn'] +  # GNN
            ['lstm', 'cnn', 'mlp'] +  # DL
            ['idw', 'linear_temporal', 'linear_vertical', 'linear_spatial', 'linear_combined']  # Stat
        )
        missing = [m for m in all_expected if m not in results]
        if missing:
            warnings.append(f"   Eksik: {', '.join(missing)}")
    
    #  
    # 4. GNN EPOCH FARKI KONTROLÜ (UYARI)
    #  
    gnn_models = ['vht_gnn','vanilla_graphsage', 'multiscale_graphsage', 'gat', 'mpnn']
    gnn_epochs = {}
    
    for model in gnn_models:
        if model in configs:
            params = configs[model].get('parameters', {})
            if 'num_epochs' in params and params['num_epochs'] is not None:
                gnn_epochs[model] = params['num_epochs']
    
    if len(gnn_epochs) > 1:
        unique_gnn_epochs = set(gnn_epochs.values())
        if len(unique_gnn_epochs) > 1:
            warnings.append(f" GNN modelleri farklı epoch sayılarıyla eğitilmiş:")
            for model, epoch in gnn_epochs.items():
                warnings.append(f"   - {model}: {epoch} epoch")
    
    return errors, warnings


def print_fairness_report(errors: list, warnings: list):
    """Adil karşılaştırma raporunu yazdır."""
    
    print("\n" + "="*70)
    print("FAIRNESS CHECK")
    print("="*70)
    
    if not errors and not warnings:
        print(" Tüm kontroller başarılı! Karşılaştırma ADİL.")
        return True
    
    if errors:
        print("\n KRİTİK HATALAR:")
        for err in errors:
            print(f"  {err}")
    
    if warnings:
        print("\n UYARILAR:")
        for warn in warnings:
            print(f"  {warn}")
    
    if errors:
        print("\n" + "-"*70)
        print("SONUÇ: Karşılaştırma GEÇERSİZ!")
        print("   Kritik hataları düzeltip modelleri yeniden çalıştırın.")
        print("-"*70)
        return False
    else:
        print("\n" + "-"*70)
        print(" SONUÇ: Karşılaştırma GEÇERLİ (uyarıları dikkate alın)")
        print("-"*70)
        return True


# 
# KARŞILAŞTIRMA TABLOLARI
# 

def create_comparison_table(results: dict, metric: str = 'r2') -> pd.DataFrame:
    """
    Tüm modeller için karşılaştırma tablosu oluştur.
    
    Returns:
        DataFrame: rows=features, columns=models
    """
    
    all_data = []
    
    for model_name, data in results.items():
        metrics_df = data['metrics']
        
        for _, row in metrics_df.iterrows():
            all_data.append({
                'model': model_name,
                'feature': row['feature'],
                metric: row[metric] if metric in row else None
            })
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    
    # Pivot table oluştur
    pivot = df.pivot(index='feature', columns='model', values=metric)
    
    # Sütun sırasını düzenle (GNN -> DL -> Statistical)
    ordered_cols = []
    for category in ['GNN', 'Deep Learning', 'Statistical']:
        for model in MODEL_CATEGORIES.get(category, []):
            if model in pivot.columns:
                ordered_cols.append(model)
    
    # Eksik modelleri de ekle
    for col in pivot.columns:
        if col not in ordered_cols:
            ordered_cols.append(col)
    
    pivot = pivot[[c for c in ordered_cols if c in pivot.columns]]
    
    return pivot


def create_complexity_table(results: dict) -> pd.DataFrame:
    """Model karmaşıklık karşılaştırma tablosu."""
    
    data = []
    
    for model_name, model_data in results.items():
        config = model_data.get('config', {})
        complexity = config.get('complexity', {})
        
        data.append({
            'Model': MODEL_DISPLAY_NAMES.get(model_name, model_name),
            'Parameters': complexity.get('total_parameters', 0),
            'Training Time (s)': complexity.get('training_time_seconds', 0),
            'Inference Time (s)': complexity.get('inference_time_seconds', 0),
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('Parameters', ascending=False)
    
    return df


def find_best_models(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Her feature için en iyi modeli bul."""
    
    results = []
    
    for feature in comparison_df.index:
        row = comparison_df.loc[feature]
        valid_values = row.dropna()
        
        if len(valid_values) == 0:
            continue
        
        best_model = valid_values.idxmax()
        best_value = valid_values.max()
        
        results.append({
            'Feature': FEATURE_DISPLAY_NAMES.get(feature, feature),
            'Best Model': MODEL_DISPLAY_NAMES.get(best_model, best_model),
            'R²': round(best_value, 4)
        })
    
    return pd.DataFrame(results)


def calculate_average_scores(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Model başına ortalama skorları hesapla."""
    
    avg_scores = comparison_df.mean().sort_values(ascending=False)
    
    results = []
    for model, score in avg_scores.items():
        results.append({
            'Model': MODEL_DISPLAY_NAMES.get(model, model),
            'Average R²': round(score, 4)
        })
    
    return pd.DataFrame(results)


# 
# LATEX TABLO ÜRETİMİ
# 

def generate_latex_comparison_table(comparison_df: pd.DataFrame, 
                                    caption: str = "Comparison of imputation methods (R² scores)",
                                    label: str = "tab:comparison") -> str:
    """LaTeX formatında karşılaştırma tablosu üret."""
    
    # Sütun başlıkları
    model_names = [MODEL_DISPLAY_NAMES.get(m, m) for m in comparison_df.columns]
    
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\small',
        f'\\caption{{{caption}}}',
        f'\\label{{{label}}}',
        r'\begin{tabular}{l' + 'c' * len(model_names) + '}',
        r'\toprule',
        r'\textbf{Variable} & ' + ' & '.join([f'\\textbf{{{m}}}' for m in model_names]) + r' \\',
        r'\midrule'
    ]
    
    # Her feature için satır
    for feature in comparison_df.index:
        row_values = comparison_df.loc[feature]
        display_name = FEATURE_DISPLAY_NAMES.get(feature, feature)
        
        # En iyi değeri bul (bold için)
        valid_values = row_values.dropna()
        best_val = valid_values.max() if len(valid_values) > 0 else None
        
        formatted_values = []
        for val in row_values:
            if pd.isna(val):
                formatted_values.append('--')
            elif best_val is not None and abs(val - best_val) < 0.0001:
                formatted_values.append(f'\\textbf{{{val:.4f}}}')
            else:
                formatted_values.append(f'{val:.4f}')
        
        lines.append(f'{display_name} & ' + ' & '.join(formatted_values) + r' \\')
    
    # Ortalama satırı
    lines.append(r'\midrule')
    avg_values = comparison_df.mean()
    best_avg = avg_values.max()
    
    formatted_avgs = []
    for val in avg_values:
        if pd.isna(val):
            formatted_avgs.append('--')
        elif abs(val - best_avg) < 0.0001:
            formatted_avgs.append(f'\\textbf{{{val:.4f}}}')
        else:
            formatted_avgs.append(f'{val:.4f}')
    
    lines.append(r'\textbf{Average} & ' + ' & '.join(formatted_avgs) + r' \\')
    
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}'
    ])
    
    return '\n'.join(lines)


def generate_latex_complexity_table(complexity_df: pd.DataFrame) -> str:
    """LaTeX formatında karmaşıklık tablosu üret."""
    
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Computational complexity comparison}',
        r'\label{tab:complexity}',
        r'\begin{tabular}{lccc}',
        r'\toprule',
        r'\textbf{Model} & \textbf{Parameters} & \textbf{Training (s)} & \textbf{Inference (s)} \\',
        r'\midrule'
    ]
    
    for _, row in complexity_df.iterrows():
        params = f"{row['Parameters']:,}" if row['Parameters'] > 0 else '--'
        train_time = f"{row['Training Time (s)']:.1f}" if row['Training Time (s)'] > 0 else '--'
        inf_time = f"{row['Inference Time (s)']:.1f}" if row['Inference Time (s)'] > 0 else '--'
        
        lines.append(f"{row['Model']} & {params} & {train_time} & {inf_time}" + r' \\')
    
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}'
    ])
    
    return '\n'.join(lines)


# 
# GÖRSELLEŞTIRME
# 

def create_visualizations(results: dict, comparison_df: pd.DataFrame, output_dir: Path):
    """Görselleştirmeler oluştur."""
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # GUI olmadan çalış
    except ImportError:
        print(" matplotlib yüklü değil, görselleştirmeler atlanıyor")
        return
    
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # 1. R² Bar Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    avg_scores = comparison_df.mean().sort_values(ascending=True)
    colors = []
    for model in avg_scores.index:
        if model in MODEL_CATEGORIES['GNN']:
            colors.append('#2ecc71')  # Yeşil
        elif model in MODEL_CATEGORIES['Deep Learning']:
            colors.append('#3498db')  # Mavi
        else:
            colors.append('#95a5a6')  # Gri
    
    bars = ax.barh(range(len(avg_scores)), avg_scores.values, color=colors)
    ax.set_yticks(range(len(avg_scores)))
    ax.set_yticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in avg_scores.index])
    ax.set_xlabel('Average R² Score')
    ax.set_title('Model Comparison: Average R² Scores')
    ax.set_xlim(0, 1)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='GNN'),
        Patch(facecolor='#3498db', label='Deep Learning'),
        Patch(facecolor='#95a5a6', label='Statistical')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(figures_dir / "r2_comparison_barplot.png", dpi=300)
    plt.savefig(figures_dir / "r2_comparison_barplot.pdf")  # Vektör format
    plt.close()
    print(f"   Saved: {figures_dir / 'r2_comparison_barplot.png'} (300 dpi)")
    print(f"   Saved: {figures_dir / 'r2_comparison_barplot.pdf'}")
    
    # 2. Feature-wise Heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Sadece mevcut modeller
    plot_df = comparison_df.copy()
    plot_df.columns = [MODEL_DISPLAY_NAMES.get(m, m) for m in plot_df.columns]
    plot_df.index = [FEATURE_DISPLAY_NAMES.get(f, f) for f in plot_df.index]
    
    im = ax.imshow(plot_df.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(plot_df.columns)))
    ax.set_xticklabels(plot_df.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)
    
    # Değerleri hücrelere yaz
    for i in range(len(plot_df.index)):
        for j in range(len(plot_df.columns)):
            val = plot_df.iloc[i, j]
            if not pd.isna(val):
                text_color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=8)
    
    plt.colorbar(im, ax=ax, label='R² Score')
    ax.set_title('R² Scores by Feature and Model')
    
    plt.tight_layout()
    plt.savefig(figures_dir / "r2_heatmap.png", dpi=300)
    plt.savefig(figures_dir / "r2_heatmap.pdf")  # Vektör format
    plt.close()
    print(f"   Saved: {figures_dir / 'r2_heatmap.png'} (300 dpi)")
    print(f"   Saved: {figures_dir / 'r2_heatmap.pdf'}")
    
    # 3. Training Loss Curves (eğer varsa)
    models_with_history = [m for m, d in results.items() if d.get('training_history') is not None]
    
    if models_with_history:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model_name in models_with_history:
            history = results[model_name]['training_history']
            if 'val_loss' in history.columns:
                ax.plot(history['epoch'], history['val_loss'], 
                       label=MODEL_DISPLAY_NAMES.get(model_name, model_name))
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Training Curves')
        ax.legend()
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(figures_dir / "training_curves.png", dpi=300)
        plt.savefig(figures_dir / "training_curves.pdf")  # Vektör format
        plt.close()
        print(f"   Saved: {figures_dir / 'training_curves.png'} (300 dpi)")
        print(f"   Saved: {figures_dir / 'training_curves.pdf'}")


# 
# RAPOR OLUŞTURMA
# 

def generate_summary_report(results: dict, comparison_df: pd.DataFrame, output_dir: Path):
    """Özet rapor oluştur (Markdown)."""
    
    report_lines = [
        "# Model Comparison Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\nTotal Models: {len(results)}",
        "\n---\n",
        "## Summary",
        "\n### Best Model by Feature\n"
    ]
    
    # En iyi modeller
    best_models = find_best_models(comparison_df)
    if not best_models.empty:
        report_lines.append(best_models.to_markdown(index=False))
    
    # Ortalama skorlar
    report_lines.extend([
        "\n### Average R² Scores\n"
    ])
    
    avg_scores = calculate_average_scores(comparison_df)
    if not avg_scores.empty:
        report_lines.append(avg_scores.to_markdown(index=False))
    
    # En iyi model
    if not avg_scores.empty:
        best_overall = avg_scores.iloc[0]
        report_lines.extend([
            f"\n###  Best Overall Model: **{best_overall['Model']}** (R² = {best_overall['Average R²']:.4f})",
        ])
    
    # Karmaşıklık
    complexity_df = create_complexity_table(results)
    if not complexity_df.empty:
        report_lines.extend([
            "\n---\n",
            "## Computational Complexity\n",
            complexity_df.to_markdown(index=False)
        ])
    
    # Full comparison table
    report_lines.extend([
        "\n---\n",
        "## Full Comparison Table (R²)\n"
    ])
    
    # DataFrame'i markdown'a çevir
    display_df = comparison_df.copy()
    display_df.columns = [MODEL_DISPLAY_NAMES.get(m, m) for m in display_df.columns]
    display_df.index = [FEATURE_DISPLAY_NAMES.get(f, f) for f in display_df.index]
    report_lines.append(display_df.round(4).to_markdown())
    
    # Dosyaya yaz
    report_path = output_dir / "comparison_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"   Saved: {report_path}")


# 
# MAIN
# 

def main():
    print("="*70)
    print(" MODEL KARŞILAŞTIRMA RAPORU")
    print("="*70)
    
    # Output klasörü oluştur
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Sonuçları yükle
    print("\n Sonuçlar yükleniyor...")
    results = load_all_results()
    
    if not results:
        print(" Hiç sonuç bulunamadı!")
        print(f"   Önce M99_MAIN.py ile modelleri çalıştırın.")
        return
    
    print(f"\n {len(results)} model yüklendi")
    
    # Adil karşılaştırma kontrolü
    errors, warnings = check_fairness(results)
    is_fair = print_fairness_report(errors, warnings)
    
    if not is_fair:
        print("\n Karşılaştırma tabloları yine de oluşturulacak, ancak sonuçlar GEÇERSİZ!")
        user_input = input("Devam etmek istiyor musunuz? (e/h): ")
        if user_input.lower() != 'e':
            print("İptal edildi.")
            return
    
    # Karşılaştırma tablosu oluştur
    print("\n Karşılaştırma tabloları oluşturuluyor...")
    
    comparison_r2 = create_comparison_table(results, 'r2')
    comparison_mae = create_comparison_table(results, 'mae')
    
    if comparison_r2.empty:
        print(" Karşılaştırma tablosu oluşturulamadı!")
        return
    
    # CSV kaydet
    comparison_r2.to_csv(OUTPUT_DIR / "comparison_r2.csv")
    print(f"   Saved: {OUTPUT_DIR / 'comparison_r2.csv'}")
    
    comparison_mae.to_csv(OUTPUT_DIR / "comparison_mae.csv")
    print(f"   Saved: {OUTPUT_DIR / 'comparison_mae.csv'}")
    
    # 
    # ADDED SECTION
    # 
    
    # Real MAE tablosu (eğer varsa)
    comparison_mae_real = create_comparison_table(results, 'mae_real')
    if not comparison_mae_real.empty:
        comparison_mae_real.to_csv(OUTPUT_DIR / "comparison_mae_real.csv")
        print(f"   Saved: {OUTPUT_DIR / 'comparison_mae_real.csv'}")
    
    # 
    # END ADDED SECTION
    # 
    
    # Complexity table
    complexity_df = create_complexity_table(results)
    complexity_df.to_csv(OUTPUT_DIR / "complexity.csv", index=False)
    print(f"   Saved: {OUTPUT_DIR / 'complexity.csv'}")
    
    # Pressure metrics birleştir
    print("\n Basınç seviyesi metrikleri birleştiriliyor...")
    all_pressure_metrics = []
    for model_name, data in results.items():
        if data.get('pressure_metrics') is not None and not data['pressure_metrics'].empty:
            pm = data['pressure_metrics'].copy()
            pm['model'] = model_name
            all_pressure_metrics.append(pm)
    
    if all_pressure_metrics:
        combined_pressure = pd.concat(all_pressure_metrics, ignore_index=True)
        combined_pressure.to_csv(OUTPUT_DIR / "metrics_by_pressure.csv", index=False)
        print(f"   Saved: {OUTPUT_DIR / 'metrics_by_pressure.csv'}")
        
        # Pressure level özet tablosu (temperature için)
        temp_pressure = combined_pressure[combined_pressure['feature'] == 'temperature']
        if not temp_pressure.empty:
            pressure_pivot = temp_pressure.pivot_table(
                index='pressure', 
                columns='model', 
                values='r2', 
                aggfunc='mean'
            ).round(4)
            pressure_pivot.to_csv(OUTPUT_DIR / "pressure_comparison.csv")
            print(f"   Saved: {OUTPUT_DIR / 'pressure_comparison.csv'}")
    else:
        print("   No pressure metrics found")

    # LaTeX tabloları
    print("\n LaTeX tabloları oluşturuluyor...")
    
    latex_r2 = generate_latex_comparison_table(comparison_r2)
    with open(OUTPUT_DIR / "table_comparison_r2.tex", 'w') as f:
        f.write(latex_r2)
    print(f"   Saved: {OUTPUT_DIR / 'table_comparison_r2.tex'}")
    
    latex_complexity = generate_latex_complexity_table(complexity_df)
    with open(OUTPUT_DIR / "table_complexity.tex", 'w') as f:
        f.write(latex_complexity)
    print(f"   Saved: {OUTPUT_DIR / 'table_complexity.tex'}")
    
    # Görselleştirmeler
    print("\n Görselleştirmeler oluşturuluyor...")
    create_visualizations(results, comparison_r2, OUTPUT_DIR)
    
    # Özet rapor
    print("\n Özet rapor oluşturuluyor...")
    generate_summary_report(results, comparison_r2, OUTPUT_DIR)
    
    # Konsola özet yazdır
    print("\n" + "="*70)
    print(" SONUÇ ÖZETİ")
    print("="*70)
    
    print("\n EN İYİ MODELLER (Feature bazında):")
    best_models = find_best_models(comparison_r2)
    for _, row in best_models.iterrows():
        print(f"   {row['Feature']:<20} → {row['Best Model']} (R²={row['R²']:.4f})")
    
    print("\n ORTALAMA R² SKORLARI:")
    avg_scores = calculate_average_scores(comparison_r2)
    for _, row in avg_scores.iterrows():
        print(f"   {row['Model']:<25} → {row['Average R²']:.4f}")

    # Real MAE özeti
    if not comparison_mae_real.empty:
        print("\n REAL MAE DEĞERLERİ (En İyi Model):")
        for feature in comparison_mae_real.index:
            row = comparison_mae_real.loc[feature]
            valid_values = row.dropna()
            if len(valid_values) > 0:
                best_model = valid_values.idxmin()  # MAE için minimum en iyi
                best_value = valid_values.min()
                unit = FEATURE_UNITS.get(feature, '')
                display_feature = FEATURE_DISPLAY_NAMES.get(feature, feature)
                display_model = MODEL_DISPLAY_NAMES.get(best_model, best_model)
                print(f"   {display_feature:<20} → {display_model}: {best_value:.2f} {unit}")
    
    
    if not avg_scores.empty:
        best = avg_scores.iloc[0]
        print(f"\n EN İYİ MODEL: {best['Model']} (Ortalama R²: {best['Average R²']:.4f})")
    
    print("\n" + "="*70)
    print(f" Tüm çıktılar kaydedildi: {OUTPUT_DIR.absolute()}")
    print("="*70)


if __name__ == "__main__":
    main()