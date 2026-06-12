# -*- coding: utf-8 -*-
"""
Türkiye Soundings Dataset - Comprehensive Analysis

Analyzes the merged IGRA dataset for data quality, coverage,
and temporal characteristics relevant to VHT-GNN model training.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# CONFIG

INPUT_FILE = 'Turkiye_IGRA_Dataset_2006-11-30_2021-05-18.csv'

# LOAD DATA

print("Loading data...")
if not os.path.exists(INPUT_FILE):
    print(f"ERROR: File not found: {INPUT_FILE}")
    sys.exit()

df = pd.read_csv(INPUT_FILE)
df['datetime'] = pd.to_datetime(df['datetime'])

print("\n" + "="*70)
print("1. BASIC STATISTICS")
print("="*70)

print(f"\nTotal rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print(f"\nColumns: {list(df.columns)}")

print(f"\nDate range: {df['datetime'].min()} -> {df['datetime'].max()}")
total_days = (df['datetime'].max() - df['datetime'].min()).days
print(f"Total days: {total_days:,}")
print(f"Total years: {total_days/365.25:.1f}")

sondaj_keys = df.groupby(['datetime', 'station_id']).ngroups
print(f"\nTotal soundings: {sondaj_keys:,}")
print(f"Number of stations: {df['station_id'].nunique()}")

# 2. STATION SUMMARY

print("\n" + "="*70)
print("2. STATION SUMMARY")
print("="*70)

# 2 soundings per day
theoretical_soundings = (total_days + 1) * 2

print(f"\n{'ID':<8} {'Name':<12} {'Elev':>6} {'First Date':<12} {'Last Date':<12} {'Soundings':>10} {'Coverage':>8}")
print("-"*75)

for station_id in sorted(df['station_id'].unique()):
    sdf = df[df['station_id'] == station_id]
    name = sdf['name'].iloc[0]
    elev = sdf['elevation'].iloc[0]
    first_date = sdf['datetime'].min()
    last_date = sdf['datetime'].max()
    sounding_count = sdf['datetime'].nunique()
    coverage = (sounding_count / theoretical_soundings) * 100
    
    print(f"{station_id:<8} {name:<12} {elev:>5}m {str(first_date)[:10]:<12} {str(last_date)[:10]:<12} {sounding_count:>10,} {coverage:>7.1f}%")

print(f"\nTheoretical max soundings ({total_days+1} days x 2): {theoretical_soundings:,}")

# 3. DATA QUALITY ANALYSIS

print("\n" + "="*70)
print("3. DATA QUALITY ANALYSIS (Missing Data Rates)")
print("="*70)

numeric_cols = ['geopotential', 'temperature', 'relative_humidity', 'wind_speed', 'wind_direction']

print(f"\n{'Variable':<20} {'Total':>12} {'Missing':>12} {'Missing %':>10} {'Filled %':>10}")
print("-"*65)

for col in numeric_cols:
    if col in df.columns:
        total = len(df)
        missing = df[col].isna().sum()
        missing_pct = (missing / total) * 100
        filled_pct = 100 - missing_pct
        print(f"{col:<20} {total:>12,} {missing:>12,} {missing_pct:>9.1f}% {filled_pct:>9.1f}%")

# RH for levels >= 200 hPa  
rh_valid_levels = df[df['pressure'] >= 200]
rh_missing = rh_valid_levels['relative_humidity'].isna().sum()
rh_total = len(rh_valid_levels)
print(f"\n* relative_humidity (>=200 hPa): {rh_total-rh_missing:,}/{rh_total:,} filled ({(rh_total-rh_missing)/rh_total*100:.1f}%)")
print(f"  (Humidity data above 200 hPa is unreliable, excluded from model)")

# Missing data by pressure level
print("\n\nMissing data by pressure level (temperature):")
print(f"{'Pressure (hPa)':<15} {'Rows':>10} {'T missing':>10} {'T missing %':>12}")
print("-"*50)

standard_levels = [1000, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 10]
for pres in standard_levels:
    subset = df[df['pressure'] == pres]
    if len(subset) > 0:
        missing = subset['temperature'].isna().sum()
        pct = (missing / len(subset)) * 100
        print(f"{pres:<15} {len(subset):>10,} {missing:>10,} {pct:>11.1f}%")

# 4. TIME SERIES ANALYSIS

print("\n" + "="*70)
print("4. TIME SERIES ANALYSIS")
print("="*70)

# Yearly distribution
df['year'] = df['datetime'].dt.year
yearly = df.groupby(['year', 'station_id'])['datetime'].nunique().reset_index()
yearly_avg = yearly.groupby('year')['datetime'].mean()

print(f"\nYearly average soundings/station:")
print(f"{'Year':<6} {'Avg Soundings':>14} {'Coverage':>10}")
print("-"*32)
for year, count in yearly_avg.items():
    coverage = (count / 730) * 100   
    print(f"{year:<6} {count:>14.0f} {coverage:>9.1f}%")


df['month'] = df['datetime'].dt.month
monthly_avg = df.groupby(['year', 'month', 'station_id'])['datetime'].nunique().groupby('month').mean()

print(f"\nMonthly average soundings/station (all years):")
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for month, count in monthly_avg.items():
    print(f"  {month_names[month-1]}: {count:.1f}")

# 5. DATA WINDOW ANALYSIS (Single + Cross-Station)

print("\n" + "="*70)
print("5. DATA WINDOW ANALYSIS")
print("="*70)

def count_valid_windows(station_df, window_size):
    """
    Count consecutive valid windows.
    
    Parameters:
        station_df: DataFrame for single station
        window_size: Number of consecutive soundings
    
    Returns:
        Number of valid windows (sounding exists = valid, missing data allowed)
    """
    # Get unique datetimes and sort
    datetimes = station_df['datetime'].drop_duplicates().sort_values().values
    
    if len(datetimes) < window_size:
        return 0
    
    valid_windows = 0
    
    # Sliding window
    for i in range(len(datetimes) - window_size + 1):
        window = datetimes[i:i+window_size]
        
        # Check if consecutive (12 hours apart)
        is_consecutive = True
        for j in range(1, len(window)):
            diff = (window[j] - window[j-1]) / np.timedelta64(1, 'h')
            if diff != 12:
                is_consecutive = False
                break
        
        if is_consecutive:
            valid_windows += 1
    
    return valid_windows

def count_cross_station_windows(df, window_size):
    """
    Count windows where ALL stations have data simultaneously.
    
    For GNN: All stations (graph nodes) must have data at the same time.
    
    Parameters:
        df: Full DataFrame with all stations
        window_size: Number of consecutive soundings
    
    Returns:
        Number of valid cross-station windows
    """
    stations = sorted(df['station_id'].unique())
    
    # Get datetimes for each station as sets
    station_datetimes = {}
    for sid in stations:
        sdf = df[df['station_id'] == sid]
        station_datetimes[sid] = set(sdf['datetime'].unique())
    
    # Get all datetimes
    all_datetimes = np.array(sorted(df['datetime'].unique()))
    
    if len(all_datetimes) < window_size:
        return 0
    
    valid_windows = 0
    
    for i in range(len(all_datetimes) - window_size + 1):
        window = all_datetimes[i:i+window_size]
        
        # Check if consecutive
        is_consecutive = True
        for j in range(1, len(window)):
            diff = (window[j] - window[j-1]) / np.timedelta64(1, 'h')
            if diff != 12:
                is_consecutive = False
                break
        
        if not is_consecutive:
            continue
        
        # Check if ALL stations have data at each datetime
        all_present = True
        for dt in window:
            for sid in stations:
                if dt not in station_datetimes[sid]:
                    all_present = False
                    break
            if not all_present:
                break
        
        if all_present:
            valid_windows += 1
    
    return valid_windows

# Window sizes to test
window_sizes = [3, 6, 12, 24, 48]

print(f"\nWindow descriptions:")
print(f"  W=3:  36 hours (1.5 days) - 3 consecutive soundings")
print(f"  W=6:  3 days              - 6 consecutive soundings")
print(f"  W=12: 6 days              - 12 consecutive soundings")
print(f"  W=24: 12 days             - 24 consecutive soundings")
print(f"  W=48: 24 days             - 48 consecutive soundings")
print(f"\nCondition: Consecutive 12-hour soundings (sounding exists = valid)")

print(f"\nCalculating... (may take a few minutes for large datasets)")

# Calculate per station
results = {}
station_names = df.groupby('station_id')['name'].first().to_dict()

for station_id in sorted(df['station_id'].unique()):
    print(f"  {station_id} {station_names[station_id]:<12}...", end=" ", flush=True)
    station_df = df[df['station_id'] == station_id]
    
    results[station_id] = {}
    for ws in window_sizes:
        valid = count_valid_windows(station_df, ws)
        results[station_id][ws] = valid
    
    print(f"OK")

# Calculate cross-station
print(f"\n  Cross-station (all {len(station_names)} stations together)...", end=" ", flush=True)
cross_results = {}
for ws in window_sizes:
    cross_results[ws] = count_cross_station_windows(df, ws)
print("OK")

# Results table
print(f"\n\n{'='*85}")
print("RESULTS TABLE: Consecutive Sounding Set Counts")
print(f"{'='*85}")

print(f"\n{'Station':<20}", end="")
for ws in window_sizes:
    print(f"{'W='+str(ws):>12}", end="")
print()
print("-" * (20 + 12*len(window_sizes)))

totals = {ws: 0 for ws in window_sizes}

for station_id in sorted(results.keys()):
    name = station_names[station_id]
    print(f"{station_id} {name:<12}", end="")
    for ws in window_sizes:
        valid = results[station_id][ws]
        totals[ws] += valid
        print(f"{valid:>12,}", end="")
    print()

print("-" * (20 + 12*len(window_sizes)))
print(f"{'SINGLE TOTAL':<20}", end="")
for ws in window_sizes:
    print(f"{totals[ws]:>12,}", end="")
print()

print(f"\n{'-'*85}")
print(f"{'ALL STATIONS':<20}", end="")
for ws in window_sizes:
    print(f"{cross_results[ws]:>12,}", end="")
print()
print(f"{'(Available for GNN)':<20}")
print(f"{'-'*85}")

print(f"\n* 'Single Total': Sum of valid sets for each station individually")
print(f"* 'All Stations': Sets where ALL {len(station_names)} stations have data simultaneously")
print(f"  -> This row is important for GNN model (all graph nodes must be present)")

# SUMMARY

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"File: {INPUT_FILE}")
print(f"Total rows: {len(df):,}")
print(f"Total soundings: {sondaj_keys:,}")
print(f"Number of stations: {df['station_id'].nunique()}")
print(f"Date range: {df['datetime'].min().date()} -> {df['datetime'].max().date()}")
print(f"Total years: {total_days/365.25:.1f}")
print(f"\nAvailable sets for GNN (all stations simultaneously):")
for ws in window_sizes:
    days = ws * 12 / 24
    print(f"  W={ws} ({days:.1f} days): {cross_results[ws]:,} sets")

print("\n" + "="*70)
print("ANALYSIS COMPLETED!")
print("="*70)


# RUN

if __name__ == '__main__':
    pass  # Already runs on import
