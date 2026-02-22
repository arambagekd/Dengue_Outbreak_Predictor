# Dengue Outbreak Predictor

A machine learning application designed to forecast monthly dengue cases across Sri Lankan districts using historical dengue case records and weather data (temperature, precipitation, wind).

**Live Application:** [Dengue Outbreak Predictor (Live App)](https://dengueoutbreakpredictor.streamlit.app/)

## Project Structure

```text
ML/
├── app/                  # Streamlit web application frontend
│   └── app.py
├── data/
│   ├── raw/              # Original raw datasets (dengue, weather, location)
│   └── processed/        # Cleaned and integrated datasets
├── models/               # Saved trained model artifacts and configuration files
├── reports/              # Generated visualizations and analysis
│   └── figures/
├── src/                  # Source code for data pipelines and modeling
│   ├── data/             # Scripts for preprocessing and feature engineering
│   ├── models/           # Scripts for model training and prediction
│   └── utils/            # Helper functions
pyproject.toml            # Modern Python dependencies and project configuration
```

## Quick Start (Local Setup)

### 1. Install Dependencies

Ensure you have Python 3.8+ installed. Since this project uses a standard `pyproject.toml`, you can install all dependencies cleanly by running:

```bash
pip install .
```

### 2. Preprocess the Data

Clean the raw datasets, aggregate weather metrics to a monthly level, and engineer the 1-month lag features to prevent data leakage:

```bash
python src/data/preprocess.py
```

### 3. Train the Model

Train the `HistGradientBoostingRegressor` model, perform hyperparameter tuning, and generate explainability plots (stored in `reports/figures/`):

```bash
python src/models/train.py
```

### 4. Run the Web Application

Launch the Streamlit dashboard locally:

```bash
streamlit run app/app.py
```

_The app will automatically open in your default browser at `http://localhost:8501`._

## Methodology

- **Target Transformation:** Outbreak spikes are heavily right-skewed. The model uses a `Log1p` transformation to stabilize the variance.
- **Explainable AI (XAI):** Permutation Feature Importance and Partial Dependence Plots (PDP) are used to interpret the model, confirming that previous case counts (`lag1`) and rainfall are the strongest predictors.
- **Algorithm:** `HistGradientBoostingRegressor` (from `scikit-learn`) was chosen for its capability to model complex non-linear weather interactions and natively handle missing values.
