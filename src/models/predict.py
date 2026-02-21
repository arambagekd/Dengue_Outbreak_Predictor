# =============================================================================
# src/models/predict.py
# Inference module for the Streamlit Front-End
# =============================================================================

import pandas as pd
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helpers import get_path

def load_pkl(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def predict(input_data: dict, config: dict) -> float:
    """
    Receives input data from Streamlit app.
    Loads the trained model and preprocessor.
    """
    try:
        model = load_pkl(get_path(config["artifacts"]["model"]))
        preprocessor = load_pkl(get_path(config["artifacts"]["preprocessor"]))
    except FileNotFoundError as e:
        raise FileNotFoundError("Model or preprocessor not found. Please train the model first.") from e

    # Convert the requested single sample into a DataFrame
    df = pd.DataFrame([input_data])
    
    # Identify which columns were numerical in the preprocessor vs categorical
    # Must match train.py exactly
    cat_cols = ["City", "Month"]
    num_cols = [
        'Value_lag1', 
        'precipitation_hours (h)_lag1', 
        'precipitation_sum (mm)_lag1', 
        'temperature_2m_mean (°C)_lag1'
    ]
    
    # Ensure all columns exist, if not, fill with 0
    for col in num_cols + cat_cols:
        if col not in df.columns:
            df[col] = 0
            
    # The preprocessor expects columns in the exact order it was fit: 
    # [all num_cols] + [cat_cols].
    df_ordered = df[num_cols + cat_cols]
    
    # Preprocess
    X_p = preprocessor.transform(df_ordered)
    
    # Predict
    preds = model.predict(X_p)
    
    # Cases can't be negative biologically
    prediction = max(0.0, preds[0])
    
    return float(prediction)
