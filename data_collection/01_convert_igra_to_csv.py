# -*- coding: utf-8 -*-
"""
IGRA Raw Data to CSV Converter

Converts NOAA IGRA (Integrated Global Radiosonde Archive) raw data files
to individual CSV files in Wyoming format for further processing.
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import metpy.calc as mpcalc
from metpy.units import units

# CONFIG

DATA_FOLDER = '../data'                # IGRA raw files folder
OUTPUT_FOLDER = '../wyoming_format'    # Output folder
MIN_YEAR = 2000                        # Skip data before this year


def parse_igra_file(filepath):
    """
    Parse IGRA data file.
    
    IGRA format: Header lines start with '#', followed by data lines.
    Returns list of DataFrames, one per sounding.
    """
    soundings = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if line.startswith('#'):
            # Header line: Starts with '#'
            header = {
                'station_id': line[1:12].strip(),
                'year': int(line[13:17]),
                'month': int(line[18:20]),
                'day': int(line[21:23]),
                'hour': int(line[24:26]),
                'numlev': int(line[32:36].strip()),
            }
            
            # Extract last 5 digits of station code
            header['short_id'] = header['station_id'][-5:]
            
            levels = []
            i += 1
            
            while i < len(lines) and not lines[i].startswith('#'):
                data_line = lines[i]
                if len(data_line) >= 50:
                    level = {
                        'lvltyp1': int(data_line[0:1]) if data_line[0:1].strip() else np.nan,
                        'pressure_Pa': int(data_line[9:15]) if data_line[9:15].strip().lstrip('-').isdigit() else np.nan,
                        'gph_m': int(data_line[16:21]) if data_line[16:21].strip().lstrip('-').isdigit() else np.nan,
                        'temp_x10': int(data_line[22:27]) if data_line[22:27].strip().lstrip('-').isdigit() else np.nan,
                        'rh_x10': int(data_line[28:33]) if data_line[28:33].strip().lstrip('-').isdigit() else np.nan,
                        'dpd_x10': int(data_line[34:39]) if data_line[34:39].strip().lstrip('-').isdigit() else np.nan,
                        'wdir': int(data_line[40:45]) if data_line[40:45].strip().lstrip('-').isdigit() else np.nan,
                        'wspd_x10': int(data_line[46:51]) if data_line[46:51].strip().lstrip('-').isdigit() else np.nan,
                    }
                    levels.append(level)
                i += 1
            
            if levels:
                df = pd.DataFrame(levels)
                df = df.replace([-9999, -8888], np.nan)
                
                # Add header info to dataframe
                for key, val in header.items():
                    df[key] = val
                
                soundings.append(df)
        else:
            i += 1
    
    return soundings


def convert_to_wyoming_format(sounding_df):
    """
    Convert single sounding to Wyoming format.
    
    Performs unit conversions and calculates derived variables
    (relative humidity, mixing ratio) using MetPy.
    """
    df = sounding_df.copy()
    
    # Basic unit conversions
    df['pressure_hPa'] = df['pressure_Pa'] / 100
    df['temperature_C'] = df['temp_x10'] / 10
    df['dpd_C'] = df['dpd_x10'] / 10
    df['dewpoint_C'] = df['temperature_C'] - df['dpd_C']
    df['wspd_ms'] = df['wspd_x10'] / 10
    
    # Calculate relative humidity and mixing ratio using MetPy
    output_rows = []
    
    # Note: iterrows() is slow for large datasets but readable for row-wise calculations
    for idx, row in df.iterrows():
        out = {
            'pressure_hPa': round(row['pressure_hPa'], 1) if pd.notna(row['pressure_hPa']) else np.nan,
            'geopotential height_m': row['gph_m'] if pd.notna(row['gph_m']) else np.nan,
            'temperature_C': round(row['temperature_C'], 2) if pd.notna(row['temperature_C']) else np.nan,
            'dew point temperature_C': round(row['dewpoint_C'], 2) if pd.notna(row['dewpoint_C']) else np.nan,
            'ice point temperature_C': round(row['dewpoint_C'], 2) if pd.notna(row['dewpoint_C']) else np.nan,
            'wind direction_degree': round(row['wdir'], 0) if pd.notna(row['wdir']) else np.nan,
            'wind speed_m/s': round(row['wspd_ms'], 1) if pd.notna(row['wspd_ms']) else np.nan,
        }
        
        # Calculate RH and mixing ratio
        if pd.notna(row['pressure_hPa']) and pd.notna(row['temperature_C']) and pd.notna(row['dewpoint_C']):
            try:
                p = row['pressure_hPa'] * units.hPa
                T = row['temperature_C'] * units.degC
                Td = row['dewpoint_C'] * units.degC
                
                rh = mpcalc.relative_humidity_from_dewpoint(T, Td) * 100
                mr = mpcalc.mixing_ratio_from_relative_humidity(p, T, rh/100) * 1000
                
                out['relative humidity_%'] = round(rh.magnitude, 1)
                out['humidity wrt ice_%'] = round(rh.magnitude, 1)
                out['mixing ratio_g/kg'] = round(mr.magnitude, 2)
            except:
                out['relative humidity_%'] = np.nan
                out['humidity wrt ice_%'] = np.nan
                out['mixing ratio_g/kg'] = np.nan
        else:
            out['relative humidity_%'] = np.nan
            out['humidity wrt ice_%'] = np.nan
            out['mixing ratio_g/kg'] = np.nan
        
        output_rows.append(out)
    
    # Column order same as Wyoming format
    columns = [
        'pressure_hPa', 'geopotential height_m', 'temperature_C',
        'dew point temperature_C', 'ice point temperature_C',
        'relative humidity_%', 'humidity wrt ice_%', 'mixing ratio_g/kg',
        'wind direction_degree', 'wind speed_m/s'
    ]
    
    return pd.DataFrame(output_rows)[columns]


def process_all_igra_files(data_folder, output_folder):
    """
    Process all IGRA files and save as CSV.
    
    Iterates through all TUM*.txt files, parses soundings,
    converts to Wyoming format, and saves individual CSV files.
    """
    
    all_files = glob.glob(f'{data_folder}/TUM*.txt')
    print(f"Found {len(all_files)} files\n")
    
    total_soundings = 0
    total_saved = 0
    
    for file_idx, filepath in enumerate(all_files):
        filename = os.path.basename(filepath)
        print(f"[{file_idx+1}/{len(all_files)}] Processing {filename}...")
        
        soundings = parse_igra_file(filepath)
        print(f"  {len(soundings)} soundings found")
        
        saved_count = 0
        for sounding in soundings:
            short_id = sounding['short_id'].iloc[0]
            year = sounding['year'].iloc[0]
            month = sounding['month'].iloc[0]
            day = sounding['day'].iloc[0]
            hour = sounding['hour'].iloc[0]
            
            # Skip data before MIN_YEAR
            if year < MIN_YEAR:
                continue
            
            # Output directory and filename
            out_dir = os.path.join(output_folder, short_id, str(year), f"{month:02d}")
            out_filename = f"{short_id}_{year}{month:02d}{day:02d}_{hour:02d}Z.csv"
            out_path = os.path.join(out_dir, out_filename)
            
            # Skip if file already exists (for faster re-runs)
            if os.path.exists(out_path):
                continue
            
            # Convert and save
            try:
                wyoming_df = convert_to_wyoming_format(sounding)
                
                if len(wyoming_df) > 0:
                    os.makedirs(out_dir, exist_ok=True)
                    wyoming_df.to_csv(out_path, index=False)
                    saved_count += 1
            except Exception as e:
                continue
        
        total_soundings += len(soundings)
        total_saved += saved_count
        print(f"  {saved_count} files saved\n")
    
    print(f"\n{'='*50}")
    print(f"COMPLETED!")
    print(f"Total soundings: {total_soundings}")
    print(f"Saved: {total_saved}")


# RUN

if __name__ == '__main__':
    process_all_igra_files(DATA_FOLDER, OUTPUT_FOLDER)
