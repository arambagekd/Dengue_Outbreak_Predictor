# =============================================================================
# src/models/train.py
# Machine Learning Assignment: Dengue Forecasting
# Focus: Chronological Splitting, HistGradientBoostingRegressor, Hyperparameter Tuning
# Feature Selection & Explainability: Feature Importance & PDP (No SHAP)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

from src.utils.helpers import load_config, get_path, setup_logger, save_artifact, ensure_dirs

logger = setup_logger("train")

TARGET = "Value"
CATEGORICAL_COLS = ["City", "Month"]
# Forced set of features as requested by the user
FORCED_NUM_COLS = [
    'Value_lag1', 
    'precipitation_hours (h)_lag1', 
    'precipitation_sum (mm)_lag1', 
    'temperature_2m_mean (°C)_lag1'
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_preprocessor(numerical_cols, categorical_cols):
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ])

def get_feature_names(preprocessor, numerical_cols, categorical_cols):
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)
    return numerical_cols + list(cat_names)

def evaluate(model, X_test, y_test, dataset_name="Test"):
    preds = model.predict(X_test)
    r2   = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    logger.info(f"[{dataset_name} Metrics] R2={r2:.4f}  |  RMSE={rmse:.2f}  |  MAE={mae:.2f}")
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "preds": preds}

# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(y_test, preds, config):
    r2_val = r2_score(y_test, preds)
    plt.figure(figsize=(8, 7))
    plt.scatter(y_test, preds, alpha=0.55, color="#4A90D9", edgecolors="white", s=55)
    lim = [min(y_test.min(), preds.min()) - 20, max(y_test.max(), preds.max()) + 20]
    plt.plot(lim, lim, "r--", lw=2, label="Perfect Prediction Line")
    plt.xlabel("Actual Dengue Cases")
    plt.ylabel("Predicted Dengue Cases")
    plt.title(f"Actual vs. Predicted Dengue Cases (R2 = {r2_val:.4f})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    save_path = get_path(config["figures"].get("actual_vs_predicted", "reports/figures/actual_vs_predicted.png"))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    logger.info(f"Saved Actual vs Predicted plot -> {save_path}")

def plot_explainability(model, X_test_df, y_test, feature_names, top_num_cols, config):
    """Generates Feature Importance Bar Plot & Partial Dependence Plot (PDP)."""
    logger.info("Generating Explainability Plots (Feature Importance & PDP)...")
    
    # 1. Feature Importance Plot (Permutation Importance on Test Set)
    result = permutation_importance(model, X_test_df, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()[-10:] # Top 10 features overall
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        tick_labels=np.array(feature_names)[sorted_idx]
    )
    plt.title("Top 10 Feature Importances (Permutation Decrease in Validation R2)")
    plt.xlabel("Decrease in accuracy when feature is shuffled")
    plt.tight_layout()
    fi_path = get_path(config["figures"].get("feature_importance", "reports/figures/feature_importance.png"))
    plt.savefig(fi_path, dpi=150)
    plt.clf()
    logger.info(f"Saved Feature Importance Plot -> {fi_path}")

    # 2. Partial Dependence Plot (PDP)
    # Pick top 2 numerical features excluding Value_lag1 (to show weather impact)
    weather_only = [f for f in top_num_cols if f != 'Value_lag1'][:2]
    
    if weather_only:
        plt.figure(figsize=(12, 5))
        disp = PartialDependenceDisplay.from_estimator(
            model, X_test_df, features=weather_only,
            feature_names=feature_names, kind="average", grid_resolution=50,
        )
        plt.subplots_adjust(top=0.9)
        disp.figure_.suptitle("Partial Dependence Plots (How Weather Impacts Output)")
        plt.tight_layout()
        
        pdp_path = get_path(config["figures"].get("pdp_plot", "reports/figures/pdp.png"))
        plt.savefig(pdp_path, dpi=150)
        plt.clf()
        logger.info(f"Saved Partial Dependence Plot -> {pdp_path}")

    # Return importances
    importances_dict = {}
    for i in sorted_idx[::-1]: # descending order
        importances_dict[feature_names[i]] = float(result.importances_mean[i])
    return importances_dict


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_training_pipeline(config=None):
    if config is None:
        config = load_config()

    ensure_dirs(config)

    # 1. Load data
    path = get_path(config["data"]["processed"]["integrated"])
    df = pd.read_csv(path)
    
    # Filter only available requested features
    available_num_cols = [c for c in FORCED_NUM_COLS if c in df.columns]
    df.dropna(subset=available_num_cols + [TARGET], inplace=True)
    
    logger.info(f"Loaded integrated data. Shape: {df.shape}")

    # 2. Chronological Split
    df_train = df[df['Year'] <= 2017].copy()
    df_val = df[df['Year'] == 2018].copy()
    df_test = df[df['Year'] >= 2019].copy()

    # --- SINGLE PHASE: TRAINING ON SPECIFIC FEATURES ---
    logger.info("--- Training Model on Forced Feature Set ---")
    preprocessor = build_preprocessor(available_num_cols, CATEGORICAL_COLS)
    
    X_train_p = preprocessor.fit_transform(df_train[available_num_cols + CATEGORICAL_COLS])
    X_val_p = preprocessor.transform(df_val[available_num_cols + CATEGORICAL_COLS])
    X_test_p = preprocessor.transform(df_test[available_num_cols + CATEGORICAL_COLS])
    
    feature_names = get_feature_names(preprocessor, available_num_cols, CATEGORICAL_COLS)
    X_train_df = pd.DataFrame(X_train_p, columns=feature_names)
    X_val_df = pd.DataFrame(X_val_p, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_p, columns=feature_names)

    # Combine Train and Val for RandomizedSearchCV
    X_train_val = pd.concat([X_train_df, X_val_df], axis=0).reset_index(drop=True)
    y_train_val = np.concatenate([df_train[TARGET].values, df_val[TARGET].values])
    y_test = df_test[TARGET].values
    
    # Create TimeSeriesSplit for more robust Cross-Validation
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    logger.info("Tuning Model (TimeSeries Cross-Validation)...")
    base_model = HistGradientBoostingRegressor(loss='squared_error', random_state=42)
    
    # Wrap with TransformedTargetRegressor to naturally handle right-skewed extreme outliers
    wrapped_model = TransformedTargetRegressor(
        regressor=base_model,
        func=np.log1p,
        inverse_func=np.expm1
    )
    
    param_distributions = {
        'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'regressor__max_iter': [100, 200, 300, 500],
        'regressor__max_leaf_nodes': [15, 31, 63, 127],
        'regressor__max_depth': [None, 3, 5, 10],
        'regressor__min_samples_leaf': [10, 20, 50],
        'regressor__l2_regularization': [0.0, 0.1, 1.0, 10.0]
    }
    
    # We use all data up to 2018 for CV
    search = RandomizedSearchCV(
        wrapped_model, 
        param_distributions=param_distributions, 
        n_iter=20, 
        cv=tscv, 
        scoring='neg_root_mean_squared_error', 
        random_state=42, 
        n_jobs=-1
    )
    search.fit(X_train_val, y_train_val)
    best_model = search.best_estimator_
    
    logger.info(f"Best parameters: {search.best_params_}")

    # Evaluate
    eval_metrics = evaluate(best_model, X_test_df, y_test, dataset_name="Test")

    # Plots
    plot_actual_vs_predicted(y_test, eval_metrics["preds"], config)
    feature_importances = plot_explainability(best_model, X_test_df, y_test, feature_names, available_num_cols, config)

    # Save
    save_artifact(best_model, config["artifacts"]["model"])
    save_artifact(preprocessor, config["artifacts"]["preprocessor"])
    
    metrics_dict = {
        "R2": round(float(eval_metrics["R2"]), 4),
        "RMSE": round(float(eval_metrics["RMSE"]), 2),
        "MAE": round(float(eval_metrics["MAE"]), 2),
    }
    
    model_metadata = {
        "feature_names": feature_names,
        "numerical_cols": available_num_cols,
        "categorical_cols": CATEGORICAL_COLS,
        "metrics": metrics_dict,
        "feature_importances": feature_importances,
        "model_info": {
            "training_rows": len(df_train),
            "total_rows": len(df),
            "datasets": [
                {
                    "name_en": "Sri Lanka Dengue Cases (2010-2020)",
                    "name_si": "ශ්‍රී ලංකාවේ ඩෙංගු රෝගීන්ගේ දත්ත (2010-2020)",
                    "url": "https://www.kaggle.com/datasets/sadaruwan/sri-lanka-dengue-cases-2010-2020"
                },
                {
                    "name_en": "Sri Lanka Weather Dataset",
                    "name_si": "ශ්‍රී ලංකාවේ කාලගුණ දත්ත",
                    "url": "https://www.kaggle.com/datasets/rasulmah/sri-lanka-weather-dataset"
                }
            ]
        }
    }
    save_artifact(model_metadata, config["artifacts"]["top_features"])
    
    # Save accuracy details to a readable text file
    acc_path = get_path("reports/accuracy.txt")
    acc_path.parent.mkdir(parents=True, exist_ok=True)
    with open(acc_path, "w") as f:
        f.write("=== Model Evaluation Metrics (Test Set 2019-2020) ===\n")
        f.write(f"R-squared (R2) Score   : {metrics_dict['R2']} ({(metrics_dict['R2']*100):.2f}%)\n")
        f.write(f"Root Mean Squared Error: {metrics_dict['RMSE']} cases\n")
        f.write(f"Mean Absolute Error    : {metrics_dict['MAE']} cases\n")
        f.write("\n=== Selected Features ===\n")
        for idx, feat in enumerate(available_num_cols + CATEGORICAL_COLS, 1):
            f.write(f"{idx}. {feat}\n")
    logger.info(f"Saved Accuracy details to -> {acc_path}")
    
    logger.info("Pipeline complete! All artifacts and plots saved successfully.")
    
if __name__ == "__main__":
    run_training_pipeline()
