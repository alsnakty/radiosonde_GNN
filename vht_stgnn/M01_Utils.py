import json
import pandas as pd

def load_station_metadata(json_path: str = "stations.json", filter_active: bool = True) -> pd.DataFrame:
    """Loads station metadata from the local JSON config."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        stations_df = pd.DataFrame(data['stations'])
        # Convert IDs to string
        stations_df['station_id'] = stations_df['station_id'].astype(str)

        if filter_active:
            if 'active' in stations_df.columns:
                original_count = len(stations_df)
                stations_df = stations_df[stations_df['active'] == True].copy()
                filtered_count = original_count - len(stations_df)
                
                if filtered_count > 0:
                    print(f"Dropped {filtered_count} inactive stations.")

        print(f"Loaded {len(stations_df)} stations.")
        return stations_df

    except FileNotFoundError:
        print(f"Config '{json_path}' not found. Using fallback.")
        
        # Fallback data
        stations_df = pd.DataFrame({
            'station_id': ['17030', '17064', '17095', '17130', 
                           '17220', '17240', '17281', '17351'],
            'name': ['Samsun', 'Istanbul', 'Erzurum', 'Ankara',
                     'Izmir', 'Isparta', 'Diyarbakir', 'Adana'],
            'lat': [41.28, 40.91, 39.91, 39.95,
                    38.39, 37.78, 37.54, 37.00],
            'lon': [36.30, 29.16, 41.25, 32.88,
                    27.08, 30.57, 40.12, 35.34],
            'elevation': [4, 19, 1861, 891,
                          31, 998, 675, 28],
            'region': ['turkey'] * 8,
            'active': [True] * 8
        })
        return stations_df