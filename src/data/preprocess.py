import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helpers import load_config, setup_logger

logger = setup_logger("preprocess")

def load_data(raw_dir: Path):
    logger.info("Loading raw datasets...")
    # Load Dengue Data
    d_path = raw_dir / "Dengue_Data (2010-2020).xlsx"
    d_df = pd.read_excel(d_path)
    
    # Load Location Metadata
    l_path = raw_dir / "locationData.csv"
    l_df = pd.read_csv(l_path)
    
    # Load Weather Data
    w_path = raw_dir / "weatherData.csv"
    w_df = pd.read_csv(w_path)
    
    return d_df, l_df, w_df

def clean_dengue_cities(d_df: pd.DataFrame, l_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning city names in dengue dataset...")
    
    # Text Normalization: Title Case and strip whitespaces
    d_df['City'] = d_df['City'].str.title().str.strip()
    l_df['city_name'] = l_df['city_name'].str.title().str.strip()
    
    city_mapping = {
        'Nuwaraeliya': 'Nuwara Eliya',
        'N Eliya': 'Nuwara Eliya',
        'N Elliya': 'Nuwara Eliya',
        'Monaragala': 'Moneragala',
        'Rathnapura': 'Ratnapura',
        'Anuradapura': 'Anuradhapura',
        'Battocolo': 'Batticaloa',
        'Puttalama': 'Puttalam',
        'Killinnochchi': 'Kilinochchi[1]',
        'Killinochchi': 'Kilinochchi[1]',
        'Kegalla': 'Kegalle',
        'Mulative': 'Mullaitivu',
        'Vauniya': 'Vavuniya',
        'Kalminai': 'Ampara'
    }
    
    d_df['City'] = d_df['City'].replace(city_mapping)

    # Some regions like Kalmunai are collapsed into Ampara district. 
    # Group by City and Date to sum cases appropriately
    if 'Date' in d_df.columns:
        d_df = d_df.groupby(['City', 'Date'], as_index=False)['Value'].sum()
    
    valid_cities = set(l_df['city_name'].unique())
    d_cities = set(d_df['City'].unique())
    
    # Find cities in Dengue data that are NOT in the Location data
    unmapped_cities = d_cities - valid_cities
    
    if unmapped_cities:
        logger.warning(f"Unmapped districts found: {unmapped_cities}")
        
    return d_df

def aggregate_weather(w_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Aggregating weather data to monthly level...")
    
    # Ensure time is datetime
    w_df['date'] = pd.to_datetime(w_df['date'])
    
    # Extract Year and Month for grouping
    w_df['Year'] = w_df['date'].dt.year
    w_df['Month'] = w_df['date'].dt.month
    
    # Define aggregation rules per feature
    agg_funcs = {
        'temperature_2m_mean (°C)': 'mean',
        'temperature_2m_max (°C)': 'mean',  # Mean of highest daily temps
        'temperature_2m_min (°C)': 'mean',  # Mean of lowest daily temps
        'apparent_temperature_mean (°C)': 'mean',
        'apparent_temperature_max (°C)': 'mean',
        'apparent_temperature_min (°C)': 'mean',
        
        'precipitation_sum (mm)': 'sum',
        'rain_sum (mm)': 'sum',
        'precipitation_hours (h)': 'sum',
        
        'shortwave_radiation_sum (MJ/m²)': 'sum',
        'et0_fao_evapotranspiration (mm)': 'sum',
        
        'wind_speed_10m_max (km/h)': 'max',
        'wind_gusts_10m_max (km/h)': 'max',
        'wind_direction_10m_dominant (°)': 'mean', # Technically a circular mean is better, but mean serves as a rough approximation
        
        'weather_code (wmo code)': lambda x: pd.Series.mode(x)[0] if not x.empty else np.nan, # Most frequent code
    }
    
    # Only aggregate columns that actually exist in the dataframe
    actual_agg_funcs = {k: v for k, v in agg_funcs.items() if k in w_df.columns}
    
    w_monthly = w_df.groupby(['location_id', 'Year', 'Month']).agg(actual_agg_funcs).reset_index()
    return w_monthly

def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creating 1-month lag features to prevent data leakage...")
    
    # Sort chronologically
    df.sort_values(by=['City', 'Year', 'Month'], inplace=True)
    
    # Identify variables to lag (Everything except identifiers and the Target 'Value')
    identifiers = ['Date', 'City', 'location_id', 'latitude', 'longitude', 'elevation', 'city_name', 'Year', 'Month']
    target = 'Value'
    
    weather_cols = [c for c in df.columns if c not in identifiers and c != target]
    
    # Lag the target variable (Cases from last month)
    df['Value_lag1'] = df.groupby('City')[target].shift(1)
    
    # Lag all weather features by 1 month
    for col in weather_cols:
        df[f'{col}_lag1'] = df.groupby('City')[col].shift(1)
        
    # CRITICAL: Drop concurrent (Month T) weather features to prevent leakage
    df.drop(columns=weather_cols, inplace=True)
    
    # Drop rows with NaN resulting from the lag (the very first month for each city)
    df.dropna(subset=['Value_lag1'], inplace=True)
    
    return df

def run_preprocessing():
    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    d_df, l_df, w_df = load_data(raw_dir)
    
    # 1. Clean Cities
    d_df = clean_dengue_cities(d_df, l_df)
    
    # 2. Add Year Month to Dengue
    d_df['Date'] = pd.to_datetime(d_df['Date'])
    d_df['Year'] = d_df['Date'].dt.year
    d_df['Month'] = d_df['Date'].dt.month
    
    # 3. Aggregate Weather Monthly
    w_monthly = aggregate_weather(w_df)
    
    # 4. Merge Location into Weather to get City mapping
    w_loc_monthly = pd.merge(w_monthly, l_df, on='location_id', how='left')
    
    # 5. Merge Weather+Location with Dengue data
    logger.info("Merging datasets...")
    # Map 'city_name' to 'City' for merging
    w_loc_monthly.rename(columns={'city_name': 'City'}, inplace=True)
    master_df = pd.merge(d_df, w_loc_monthly, on=['City', 'Year', 'Month'], how='inner')
    
    # 6. Engineer Features & Drop Leakage
    final_df = create_lag_features(master_df)
    
    # Drop all extraneous location metadata to keep only the District Name (City)
    cols_to_drop = [
        'location_id', 'latitude', 'longitude', 'elevation', 
        'city_name', 'country', 'country_id', 
        'timezone', 'timezone_abbreviation'
    ]
    # Also drop 1-month lagged versions of metadata if they were created accidentally
    lag_meta_drops = [f"{c}_lag1" for c in cols_to_drop]
    
    to_drop = [c for c in cols_to_drop + lag_meta_drops if c in final_df.columns]
    final_df.drop(columns=to_drop, inplace=True, errors='ignore')
    
    # 7. Save
    out_path = processed_dir / "integrated_data.csv"
    final_df.to_csv(out_path, index=False)
    logger.info(f"Preprocessing complete. Saved shape {final_df.shape} to {out_path}")

if __name__ == "__main__":
    run_preprocessing()
