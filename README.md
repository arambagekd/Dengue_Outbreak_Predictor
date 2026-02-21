# Dengue Outbreak Prediction — ML Assignment

## MSc in Artificial Intelligence

A machine learning project to predict monthly dengue case counts across Sri Lanka using historical weather and location data.

---

## Project Structure

```
ML/
├── data/
│   ├── raw/            ← Original datasets (dengue, weather, location)
│   ├── processed/      ← Cleaned & merged data (auto-generated)
│   └── external/       ← Any additional data
├── notebooks/          ← Step-by-step Jupyter notebooks
├── src/
│   ├── data/           ← preprocessing.py
│   ├── models/         ← train.py, predict.py
│   ├── evaluation/     ← metrics.py
│   ├── explainability/ ← shap_analysis.py
│   └── utils/          ← helpers.py
├── models/saved/       ← Trained model artifacts (.pkl)
├── reports/figures/    ← Generated plots and graphs
├── app/                ← Streamlit web application
├── tests/              ← Unit tests
├── .streamlit/         ← Cloud configuration
├── requirements.txt    ← Dependencies
└── README.md           ← This file
```

---

## Deployment (Free)

This application is optimized for deployment on **Streamlit Community Cloud**.

1. **GitHub**: Push this repository to GitHub.
2. **Deploy**: Link your repository to [Streamlit Cloud](https://share.streamlit.io/).
3. **Guide**: See the [Deployment Guide](file:///C:/Users/Arambage%20K%20D/.gemini/antigravity/brain/bd3cadc3-f5eb-4b76-920e-d0cec9db9f62/deployment_guide.md) for step-by-step instructions.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
# Step 1: Preprocess data
python src/data/preprocessing.py

# Step 2: Train the model
python src/models/train.py

# Step 3: Launch the web app
streamlit run app/app.py
```

### 3. Or use Jupyter Notebooks

Open notebooks in order: `01` → `02` → `03` → `04` → `05`

---

## Algorithm

**HistGradientBoostingRegressor** (scikit-learn)  
A modern histogram-based gradient boosting regressor — faster than standard GBM, handles missing values natively, and outperforms Decision Trees, KNN, and Logistic Regression on tabular data.

---

## XAI Methods Used

- **Permutation Feature Importance** — which features reduce accuracy most when shuffled
- **Partial Dependence Plots (PDP)** — marginal effect of each weather feature on predictions

---

## Dataset

- `Dengue_Data (2010-2020).xlsx` — Monthly dengue case counts per district
- `weatherData.csv` — Daily weather observations per location (2010–2020)
- `locationData.csv` — City/district metadata with location IDs
