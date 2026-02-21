# =============================================================================
# src/utils/helpers.py
# =============================================================================

import yaml
from pathlib import Path
import logging
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_config(config_path: str = "config.yaml") -> dict:
    full_path = PROJECT_ROOT / config_path
    if not full_path.exists():
        # Minimum default config to prevent crashes
        return {
            "data": {
                "raw": "data/raw",
                "processed": {"integrated": "data/processed/integrated_data.csv"}
            },
            "split": {"test_size": 0.2, "random_state": 42},
            "model": {"params": {"random_state": 42, "max_iter": 200}},
            "artifacts": {
                "model": "models/model.pkl",
                "preprocessor": "models/preprocessor.pkl",
                "top_features": "models/top_features.json"
            },
            "figures": {
                "feature_importance": "reports/figures/feature_importance.png",
                "pdp_plot": "reports/figures/pdp.png",
                "actual_vs_predicted": "reports/figures/actual_vs_predicted.png"
            }
        }
    with open(full_path, "r") as f:
        return yaml.safe_load(f)

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def get_path(rel_path: str) -> Path:
    return PROJECT_ROOT / rel_path

def ensure_dirs(config: dict):
    dirs_to_create = [
        get_path(config["data"]["raw"]),
        get_path(config["data"]["processed"]["integrated"]).parent,
        get_path(config["artifacts"]["model"]).parent,
        get_path(config["figures"]["feature_importance"]).parent
    ]
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)

import pickle
import json

def save_artifact(obj, rel_path: str):
    path = get_path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".json"):
        with open(path, 'w') as f:
            json.dump(obj, f, indent=4)
    else:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
