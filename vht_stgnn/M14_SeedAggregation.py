# -*- coding: utf-8 -*-
"""
M14_SeedAggregation.py - Multi-seed sonuc agregasyonu + istatistiksel anlamlilik

Okur:
  results/<model>/seed_<n>/metrics.csv   (multi-seed GNN/DL/SAITS)
  results/<model>/metrics.csv            (flat, deterministic statistical baseline)

Uretir:
  comparison_output/seed_aggregated.csv      - model x feature bazinda mean ve std
  comparison_output/significance_matrix.csv  - secili model ciftleri icin paired test

Istatistiksel notlar:
  - Maskeleme seed'i sabittir (cfg.Masking.random_seed=42); tum egitim seedleri
    ayni test noktalarini ayni yapay deliklerle skorlar. Seedler arasi varyans
    yalnizca model init/egitim stokastikliginden gelir, veriden degil.
  - Iki multi-seed model: ortak seedler uzerinden eslestirilmis (paired) test.
    ttest_rel (paired t) + Wilcoxon signed-rank.
  - Multi-seed model vs deterministic baseline (tek deger): one-sample ttest_1samp
    ve (model_degerleri - sabit) farklarinin Wilcoxon signed-rank testi.
  - Guc siniri: 5 seed ile Wilcoxon signed-rank'in iki-yonlu minimum p-degeri
    0.0625'tir; tek basina alpha=0.05'e ulasamaz. ttest_rel df=4 ile calisir ama
    farklarin normalligini varsayar. p-degerleri mean +/- std ve etki buyuklugu
    (fark / std) ile birlikte yorumlanmalidir.

RESULTS_DIR: tum modellerin tum seedlerinin metrics.csv'lerini iceren sonuc
klasoru (model bazli alt-klasorler, her birinde seed_<n>/ veya duz klasor).
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = Path("results_canonical")
OUTPUT_DIR = Path("comparison_output")

# Seedler arasi mean/std hesaplanacak metrikler (metrics.csv sutunlari)
AGG_METRICS = ['r2', 'mae', 'mae_real', 'rmse', 'bias', 'bias_real']

# Paired test icin oncelikli model ciftleri. Eksik model olan ciftler atlanir.
FOCUS_PAIRS = [
    # Ana baseline kiyaslari
    ('vht_gnn', 'flat_graphsage'),
    ('vht_gnn', 'lstm'),
    ('vht_gnn', 'gat'),
    ('vht_gnn', 'mpnn'),
    ('vht_gnn', 'multiscale_graphsage'),
    ('vht_gnn', 'saits'),
    # Ablasyon (bilesen gerekliligi)
    ('vht_gnn', 'vht_gnn_no_gating'),
    ('vht_gnn', 'vht_gnn_fixed_fusion'),
    ('vht_gnn', 'vht_gnn_no_temporal'),
    ('vht_gnn', 'vht_gnn_no_vertical'),
    ('vht_gnn', 'vht_gnn_no_horizontal'),
    # Baseline yardimci
    ('saits', 'linear_temporal'),
]

# Anlamlilik testi metrigi (her feature icin)
SIG_METRIC = 'r2'

FLAT = '__flat__'  # deterministic (seedsiz) model etiketi


def load_model_seeds(model_dir: Path) -> dict:
    """model_dir altindaki seed_<n>/metrics.csv'leri oku. seed klasoru yoksa flat
    metrics.csv. Donen: {seed_label: metrics_df}. metrics.csv yoksa bos dict."""
    out = {}
    seed_dirs = sorted(model_dir.glob("seed_*"))
    if seed_dirs:
        for sd in seed_dirs:
            mf = sd / "metrics.csv"
            if mf.exists():
                out[sd.name.replace("seed_", "")] = pd.read_csv(mf)
    else:
        mf = model_dir / "metrics.csv"
        if mf.exists():
            out[FLAT] = pd.read_csv(mf)
    return out


def load_all(results_dir: Path) -> dict:
    """Tum modelleri yukle. Donen: {model_name: {seed_label: metrics_df}}."""
    all_models = {}
    if not results_dir.exists():
        print(f"Results klasoru bulunamadi: {results_dir}")
        return all_models
    for md in sorted(results_dir.iterdir()):
        if not md.is_dir():
            continue
        seeds = load_model_seeds(md)
        if seeds:
            all_models[md.name] = seeds
    return all_models


def aggregate(all_models: dict) -> pd.DataFrame:
    """Model x feature bazinda her metrigin seedler arasi mean ve std'si.
    Ornek std icin ddof=1; n_seeds<2 ise std NaN (tek ornekten std kestirilemez)."""
    rows = []
    for model, seeds in all_models.items():
        n_seeds = len(seeds)
        first = next(iter(seeds.values()))
        for feat in first['feature']:
            row = {'model': model, 'feature': feat, 'n_seeds': n_seeds}
            for metric in AGG_METRICS:
                vals = []
                for df in seeds.values():
                    sub = df.loc[df['feature'] == feat]
                    if sub.empty or metric not in sub.columns:
                        continue
                    v = sub.iloc[0][metric]
                    if pd.notna(v):
                        vals.append(float(v))
                if vals:
                    row[f'{metric}_mean'] = float(np.mean(vals))
                    row[f'{metric}_std'] = float(np.std(vals, ddof=1)) if len(vals) >= 2 else np.nan
                else:
                    row[f'{metric}_mean'] = np.nan
                    row[f'{metric}_std'] = np.nan
            if 'unit' in first.columns:
                u = first.loc[first['feature'] == feat, 'unit']
                if not u.empty:
                    row['unit'] = u.iloc[0]
            if 'n' in first.columns:
                nval = first.loc[first['feature'] == feat, 'n']
                if not nval.empty:
                    row['n'] = int(nval.iloc[0])
            rows.append(row)
    return pd.DataFrame(rows)


def _metric_by_seed(seeds: dict, feat: str, metric: str) -> dict:
    """Bir model icin {seed_label: metric_value} (feature sabit, NaN'lar atlanir)."""
    out = {}
    for sl, df in seeds.items():
        sub = df.loc[df['feature'] == feat]
        if not sub.empty and metric in sub.columns:
            v = sub.iloc[0][metric]
            if pd.notna(v):
                out[sl] = float(v)
    return out


def paired_tests(all_models: dict, pairs, metric: str) -> pd.DataFrame:
    """Secili model ciftleri icin per-feature anlamlilik testi.
    test_type: 'paired' (iki multi-seed), 'one_sample_vs_const' (multi-seed vs
    deterministic), 'deterministic_no_test' (iki deterministic, test yapilamaz)."""
    rows = []
    any_seeds = next(iter(all_models.values()))
    features = list(next(iter(any_seeds.values()))['feature'])

    for a, b in pairs:
        if a not in all_models or b not in all_models:
            continue
        for feat in features:
            va = _metric_by_seed(all_models[a], feat, metric)
            vb = _metric_by_seed(all_models[b], feat, metric)
            a_flat = FLAT in va
            b_flat = FLAT in vb

            rec = {'model_a': a, 'model_b': b, 'feature': feat, 'metric': metric,
                   'mean_a': np.nan, 'mean_b': np.nan, 'diff': np.nan,
                   'n_pair': 0, 'test_type': None,
                   'ttest_p': np.nan, 'wilcoxon_p': np.nan}

            if not a_flat and not b_flat:
                common = sorted(set(va) & set(vb))
                rec['test_type'] = 'paired'
                rec['n_pair'] = len(common)
                if common:
                    xa = np.array([va[s] for s in common])
                    xb = np.array([vb[s] for s in common])
                    rec['mean_a'] = float(np.mean(xa))
                    rec['mean_b'] = float(np.mean(xb))
                    rec['diff'] = rec['mean_a'] - rec['mean_b']
                    if len(common) >= 2:
                        try:
                            rec['ttest_p'] = float(stats.ttest_rel(xa, xb).pvalue)
                        except Exception:
                            pass
                        try:
                            if not np.allclose(xa, xb):
                                rec['wilcoxon_p'] = float(stats.wilcoxon(xa, xb).pvalue)
                        except Exception:
                            pass

            elif a_flat ^ b_flat:
                rec['test_type'] = 'one_sample_vs_const'
                if a_flat:
                    const = va[FLAT]
                    multi = np.array(list(vb.values()))
                    rec['mean_a'] = const
                    rec['mean_b'] = float(np.mean(multi)) if multi.size else np.nan
                else:
                    const = vb[FLAT]
                    multi = np.array(list(va.values()))
                    rec['mean_a'] = float(np.mean(multi)) if multi.size else np.nan
                    rec['mean_b'] = const
                rec['n_pair'] = int(multi.size)
                if multi.size:
                    rec['diff'] = rec['mean_a'] - rec['mean_b']
                if multi.size >= 2:
                    try:
                        rec['ttest_p'] = float(stats.ttest_1samp(multi, const).pvalue)
                    except Exception:
                        pass
                    try:
                        d = multi - const
                        if not np.allclose(d, 0):
                            rec['wilcoxon_p'] = float(stats.wilcoxon(d).pvalue)
                    except Exception:
                        pass

            else:
                rec['test_type'] = 'deterministic_no_test'
                rec['mean_a'] = va[FLAT]
                rec['mean_b'] = vb[FLAT]
                rec['diff'] = rec['mean_a'] - rec['mean_b']
                rec['n_pair'] = 1

            rows.append(rec)
    return pd.DataFrame(rows)


def main():
    print("M14 - Multi-seed agregasyon + anlamlilik testi")
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_models = load_all(RESULTS_DIR)
    if not all_models:
        print("Hic model bulunamadi. Once M99_MAIN.py ile sonuc uretin.")
        return

    print(f"Yuklenen modeller ({len(all_models)}):")
    for m, seeds in all_models.items():
        labels = [s for s in seeds if s != FLAT]
        if labels:
            print(f"  {m}: {len(labels)} seed ({', '.join(sorted(labels))})")
        else:
            print(f"  {m}: flat (deterministic, seedsiz)")

    agg = aggregate(all_models)
    agg_path = OUTPUT_DIR / "seed_aggregated.csv"
    agg.to_csv(agg_path, index=False)
    print(f"Yazildi: {agg_path} ({len(agg)} satir)")

    sig = paired_tests(all_models, FOCUS_PAIRS, SIG_METRIC)
    if not sig.empty:
        sig_path = OUTPUT_DIR / "significance_matrix.csv"
        sig.to_csv(sig_path, index=False)
        print(f"Yazildi: {sig_path} ({len(sig)} satir, metric={SIG_METRIC})")
        print("Not: 5 seed ile Wilcoxon iki-yonlu min p=0.0625; p-degerleri "
              "mean+/-std ve fark/std ile birlikte yorumlanmali.")
    else:
        print("Anlamlilik testi: FOCUS_PAIRS'teki modeller henuz mevcut degil, atlandi.")


if __name__ == "__main__":
    main()
