# Machine Learning Assignment: Dengue Outbreak Prediction

**Student Name:** Kavindu Dilshan Arambage  
**Objective:** Forecasting monthly dengue cases in Sri Lanka using advanced machine learning models and historical weather data.

---

## 1. Problem Definition & Dataset Collection (15 marks)

### Problem Definition

Dengue fever is a critical mosquito-borne viral infection in Sri Lanka, exhibiting complex spatial and temporal dynamics heavily influenced by environmental conditions such as temperature, precipitation, and global climate shifts. Accurately anticipating outbreaks across different districts enables proactive public health interventions—from targeted vector control strategies to adequate resource allocation in hospitals.

### Dataset Collection & Preprocessing

This study leverages two primary data sources:

1. **Dengue Data (2010–2020):** Monthly reported dengue cases across Sri Lankan districts sourced from official epidemiological health records.
2. **Weather Data:** Historical meteorological metrics (e.g., precipitation, temperature, wind speed) scraped/aggregated for the corresponding districts via public APIs (Open-Meteo).

**Size & Scope:**
The integrated dataset spans 11 years across 25+ districts, culminating in thousands of historical data points.

**Preprocessing and Data Leakage Prevention:**

- **City Mapping:** Standardized text representations of city/district names between the two disparate datasets to ensure a perfect spatial join.
- **Aggregation:** Daily weather recordings were aggregated to a monthly level (`sum` for precipitation, `mean` for temperatures, etc.) to match the temporal frequency of the dengue case reports.
- **Critical Time-Series Feature Engineering (Lags):** To rigorously prevent **data leakage**, concurrent-month weather data (Month $T$) cannot be used to predict concurrent-month dengue cases (Month $T$). Thus, all weather features and historical dengue cases were shifted by 1 month to create `_lag1` features. The model is therefore trained exclusively to predict Month $T$ cases using _only_ weather conditions from Month $T-1$, accurately simulating a real-world predictive scenario.

---

## 2. Selection of a New Machine Learning Algorithm (15 marks)

### Chosen Algorithm: HistGradientBoostingRegressor

Deep learning and standard basic models (Logistic Regression, Decision Trees, k-NN) were explicitly avoided per the rubric guidelines. Instead, `HistGradientBoostingRegressor` (an implementation of LightGBM style boosting in `scikit-learn`) was selected.

### Justification

Dengue forecasting inherently features complex, non-linear interactions (e.g., rainfall increases mosquito breeding, but only within specific temperature "survival" thresholds).

- **Handling Non-linearity:** Unlike linear/logistic regression, Gradient Boosting easily maps highly intricate, non-linear, and interacting variables.
- **Robustness to Missing Data:** Weather API data often contains gaps. HistGradientBoosting natively supports missing values without requiring aggressive imputation.
- **Computational Efficiency:** It bins continuous features into discrete histograms, massively accelerating training and hyperparameter search speeds over standard Random Forests, making it ideal for the high-dimensional spatial-temporal datasets at hand.

---

## 3. Model Training and Evaluation (20 marks)

### Chronological Train/Validation/Test Split

Because this is time-series epidemiological data, simple random splitting (`train_test_split`) would cause catastrophic future data leakage. The data was split strictly chronologically:

- **Training Set:** 2010 – 2017
- **Validation Set:** 2018 (Used exclusively for searching optimal hyperparameters)
- **Testing Set:** 2019 – 2020 (The final unseen predictive horizon)

### Hyperparameter Tuning

A Randomized Search Cross Validation (`RandomizedSearchCV`) was implemented in conjunction with a `PredefinedSplit`. This systematically rotated through learning rates, max iterations, max tree depths, and L2 regularization penalties, optimizing for negative Root Mean Squared Error (RMSE) entirely on the 2018 validation dataset.

### Evaluation Metrics

The model was evaluated on the unseen 2019-2020 test partition:

- **R-squared ($R^2$):** Assesses the proportion of variance in dengue cases captured by the model.
- **Root Mean Squared Error (RMSE):** Represents the standard deviation of prediction errors.
- **Mean Absolute Error (MAE):** The average magnitude of errors ignoring directionality.

_Graphs evaluating actual versus predicted cases over this test horizon clearly illustrate the model's capacity to track major outbreak spikes._

---

## 4. Explainability & Interpretation (20 marks)

### Approach: Permutation Feature Importance & Partial Dependence Plots (PDP)

To peer inside the "black box" of the ensemble tree model and understand feature impacts, state-of-the-art global and local interpretability techniques were utilized:

1. **Permutation Importance:** Evaluates how much the model's accuracy (R²) drops when a specific feature's data is randomly shuffled.
2. **Partial Dependence Plots (PDP):** Visualizes the marginal, non-linear effect that specific weather features have on the predicted dengue cases, averaging out all other variables.

### What the Model Learned

The Explainability Analysis reveals:

1. **Value_lag1 (Historical Cases):** Is by far the most dominant feature globally. High cases in Month $T-1$ heavily increase the probability of high cases in Month $T$, reflecting the contiguous spread of the outbreak.
2. **Weather Dynamics (PDP):** The model dynamically isolated the top numerical features (e.g., `precipitation_hours` and `wind_direction`). The Partial Dependence Plots illustrate exactly at which weather thresholds the predicted risk of a dengue outbreak severely spikes. This perfectly aligns with entomological domain knowledge, where specific rainfall durations accelerate the _Aedes aegypti_ mosquito's pupal development cycle.

---

## 5. Critical Discussion (10 marks)

### Data Quality Limitations

- **Underreporting:** The target variable (`Value`) reflects strictly _reported_ hospital cases. Asymptomatic individuals or those who stayed home remain uncounted, injecting bias into the "true" scale of outbreaks.
- **Spatial Granularity:** Aggregating data across an entire district blurs critical micro-climates; essentially, dengue outbreaks often occur in highly localized urban pockets, but the model only sees an average "City" temperature.

### Risk of Bias and Ethical Deployment

Deploying this in a public health setting introduces ethical dimensions. The model's reliance on historical case data (Lag1) means regions that historically suffered worse outbreaks might perpetually get flagged as "high risk," potentially diverting preventative resources away from districts that have never experienced an outbreak but are beginning to hit dangerous climate thresholds.
Predictive AI in public epidemiology must therefore operate strictly as a supportive recommender system alongside human health inspectors, rather than an autonomous decision-maker.

---

## 6. Bonus: Front-End Integration (10 marks)

A modern, dynamic front-end application was engineered using **Streamlit**. It acts as a dashboard wherein users can:

- Select a specific Sri Lankan district from a dropdown.
- Input the required lagged predictor variables (e.g., last month's temperature and rainfall).
- Gain instant inference into the upcoming month's predicted case count (classified visually by Risk Level badges: Low/Medium/High).
- Visibly interact with the system's Interpretability Layer, transparently displaying the Feature Importance and Partial Dependence Plot (PDP) graphs to establish user trust.
