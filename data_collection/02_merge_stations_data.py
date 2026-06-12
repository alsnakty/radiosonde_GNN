# -*- coding: utf-8 -*-
"""
Merge All Turkish Stations into Single Dataset

Combines individual station CSV files into a unified dataset containing
only surface and standard pressure levels for VHT-GNN model input.
"""

import glob
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# CONFIG

WYOMING_PATH = '../wyoming_format'        
STATIONS_FILE = '../stations.json'        

# Date range (based on optimal station coverage analysis)
START_DATE = '2006-11-30'
END_DATE = '2021-05-18'

OUTPUT_FILE = f'Turkiye_IGRA_Dataset_{START_DATE}_{END_DATE}.csv'

# Standard pressure levels (hPa)
STANDARD_LEVELS = [1000, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 10]


# LOAD STATIONS

def load_stations(filepath):
    """Load station metadata from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    stations = {}
    for s in data['stations']:
        stations[s['station_id']] = {
            'name': s['name'],
            'lat': s['lat'],
            'lon': s['lon'],
            'elevation': s['elevation']
        }
    return stations


# MAIN

def merge_stations(wyoming_path, stations, start_date, end_date, output_file):
    """
    Merge all station CSV files into single dataset.
    
    Filters for surface + standard pressure levels only,
    applies date range filtering, and adds station metadata.
    """
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    print(f"Date range: {start_date} -> {end_date}")
    print(f"Stations: {len(stations)}")
    print(f"Standard levels: {STANDARD_LEVELS}")
    print("\nScanning files...")
    
    all_data = []
    file_count = 0
    skipped_count = 0
    
    for station_id in stations.keys():
        station_info = stations[station_id]
        pattern = f'{wyoming_path}/{station_id}/**/*.csv'
        files = glob.glob(pattern, recursive=True)
        
        station_count = 0
        for f in files:
            fname = os.path.basename(f)
            parts = fname.replace('.csv', '').split('_')
            
            if len(parts) >= 3:
                date_str = parts[1]
                hour = parts[2].replace('Z', '')
                
                try:
                    file_date = datetime.strptime(date_str, '%Y%m%d')
                    
                    # Date range check
                    if file_date < start_dt or file_date > end_dt:
                        skipped_count += 1
                        continue
                    
                    # Read CSV file
                    df = pd.read_csv(f)
                    
                    if df.empty or 'pressure_hPa' not in df.columns:
                        continue
                    
                    # Surface = max pressure (first row)
                    surface_pressure = df['pressure_hPa'].max()
                    
                    # Add level_type column
                    df['level_type'] = df['pressure_hPa'].apply(
                        lambda x: 'surface' if x == surface_pressure else 'standard'
                    )
                    
                    # Pressure levels to get: surface + standard levels
                    levels_to_get = [surface_pressure] + STANDARD_LEVELS
                    
                    # Filter pressure levels
                    selected = df[df['pressure_hPa'].isin(levels_to_get)].copy()

                    # 1. Remove duplicates and set pressure as index (required for reindex)
                    selected = selected.drop_duplicates(subset=['pressure_hPa'], keep='first')
                    selected = selected.set_index('pressure_hPa')

                    # 2. Create index with all required levels
                    full_index = pd.Index(levels_to_get, name='pressure_hPa')

                    # 3. Reindex to add missing levels as NaN
                    selected = selected.reindex(full_index)

                    # 4. Reset index and fill level_type for missing rows
                    selected = selected.reset_index()
                    selected['level_type'] = selected['level_type'].fillna('standard')

                    # Rename columns
                    selected = selected.rename(columns={
                        'pressure_hPa': 'pressure',
                        'geopotential height_m': 'geopotential',
                        'temperature_C': 'temperature',
                        'relative humidity_%': 'relative_humidity',
                        'wind speed_m/s': 'wind_speed',
                        'wind direction_degree': 'wind_direction',
                    })
                    
                    selected['station_id'] = station_id
                    selected['name'] = station_info['name']
                    selected['lat'] = station_info['lat']
                    selected['lon'] = station_info['lon']
                    selected['elevation'] = station_info['elevation']
                    
                    # If level_type is 'surface' and geopotential is empty, use elevation
                    selected.loc[selected['level_type'] == 'surface', 'geopotential'] = selected['elevation']
                    
                    selected['datetime'] = f"{file_date.strftime('%Y-%m-%d')} {hour}:00"
                    selected['hour'] = int(hour)
                    
                    all_data.append(selected)
                    station_count += 1
                    file_count += 1
                    
                except Exception as e:
                    print(f"ERROR: Could not process {fname}: {e}")
                    continue
        
        print(f"  {station_id} ({station_info['name']:<10} elev:{station_info['elevation']:>4}m): {station_count:,} files")
    
    print(f"\nTotal: {file_count:,} files read, {skipped_count:,} files skipped (out of date range)")
    
    if all_data:
        print("\nMerging...")
        final_df = pd.concat(all_data, ignore_index=True)
        
        column_order = [
            'datetime', 'hour', 'station_id', 'name', 'lat', 'lon', 'elevation',
            'pressure', 'level_type', 'geopotential',
            'temperature', 'relative_humidity',
            'wind_speed', 'wind_direction'
        ]
        
        final_cols = [c for c in column_order if c in final_df.columns]
        final_df = final_df[final_cols]
        
        final_df = final_df.sort_values(['station_id', 'datetime', 'pressure'], 
                                          ascending=[True, True, False])
        
        final_df = final_df.reset_index(drop=True)
        
        # Statistics
        print(f"\nSUMMARY:")
        print(f"  Total rows: {len(final_df):,}")
        print(f"  Total soundings: {file_count:,}")
        print(f"  Unique days: {final_df['datetime'].str[:10].nunique():,}")
        print(f"  Average levels/sounding: {len(final_df) / file_count:.1f}")
        
        print(f"\nLevel type distribution:")
        print(final_df['level_type'].value_counts().to_string())
        
        print(f"\nLevels per station:")
        sondaj_counts = final_df.groupby('station_id').apply(
            lambda x: len(x) / x['datetime'].nunique(), include_groups=False
        )
        for sid, avg in sondaj_counts.sort_values(ascending=False).items():
            name = stations[sid]['name']
            elev = stations[sid]['elevation']
            print(f"  {sid} {name:<10} ({elev:>4}m): avg {avg:.1f} levels/sounding")
        
        print(f"\nPressure level distribution (top 15):")
        pressure_counts = final_df['pressure'].value_counts().sort_index(ascending=False)
        for p, count in list(pressure_counts.items())[:15]:
            pct = count / file_count * 100
            print(f"  {p:>7.1f} hPa: {count:>8,} ({pct:>5.1f}%)")
        
        # Save
        print(f"\nSaving: {output_file}")
        final_df.to_csv(output_file, index=False)
        
        file_size = os.path.getsize(output_file) / 1024 / 1024
        print(f"File size: {file_size:.1f} MB")
        
        # Sample output
        print(f"\nSample output (first 3 rows):")
        print(final_df.head(3).to_string(index=False))
        
        print("\nDONE!")
    else:
        print("No data found!")


# RUN

if __name__ == '__main__':
    STATIONS = load_stations(STATIONS_FILE)
    merge_stations(WYOMING_PATH, STATIONS, START_DATE, END_DATE, OUTPUT_FILE)
