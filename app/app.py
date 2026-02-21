import streamlit as st
import pandas as pd
import numpy as np
import sys
import textwrap
from pathlib import Path

# Add project root to sys path to import from src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Attempt to load the predict tool and config
try:
    from src.models.predict import predict
    from src.utils.helpers import load_config
    config = load_config()
except Exception as e:
    st.error(f"Error loading backend modules: {e}")
    st.stop()

import json
features_path = PROJECT_ROOT / "models" / "top_features.json"
feature_importances = {}
try:
    if features_path.exists():
        with open(features_path, 'r') as f:
            features_metadata = json.load(f)
            feature_importances = features_metadata.get("feature_importances", {})
except:
    pass

total_imp = sum(feature_importances.values()) if feature_importances else 1.0
def get_pct(keys):
    if not feature_importances: return "0%"
    val = sum(feature_importances.get(k, 0.0) for k in keys)
    return f"{(val / total_imp) * 100:.1f}%"

lag_pct = get_pct(["Value_lag1"])
precip_pct = get_pct(["precipitation_sum (mm)_lag1", "precipitation_hours (h)_lag1"])
temp_pct = get_pct(["temperature_2m_mean (°C)_lag1"])
loc_keys = [k for k in feature_importances if k.startswith("City_") or k.startswith("Month_")]
loc_pct = get_pct(loc_keys)

# Load dynamic model information (datasets, rows)
training_rows = "2,375"
ds1_en, ds1_si, ds1_url = "Sri Lanka Dengue Cases (2010-2020)", "ශ්‍රී ලංකාවේ ඩෙංගු රෝගීන්ගේ දත්ත (2010-2020)", "https://www.kaggle.com/datasets/sadaruwan/sri-lanka-dengue-cases-2010-2020"
ds2_en, ds2_si, ds2_url = "Sri Lanka Weather Dataset", "ශ්‍රී ලංකාවේ කාලගුණ දත්ත", "https://www.kaggle.com/datasets/rasulmah/sri-lanka-weather-dataset"

try:
    if "model_info" in features_metadata:
        mi = features_metadata["model_info"]
        training_rows = f"{mi.get('training_rows', 2375):,}"
        if len(mi.get("datasets", [])) >= 2:
            ds1, ds2 = mi["datasets"][0], mi["datasets"][1]
            ds1_en, ds1_si, ds1_url = ds1["name_en"], ds1["name_si"], ds1["url"]
            ds2_en, ds2_si, ds2_url = ds2["name_en"], ds2["name_si"], ds2["url"]
except:
    pass

# --- CONFIGURATION & TRANSLATIONS ---
CITIES = [
    "Colombo", "Gampaha", "Kalutara", "Kandy", "Matale", "Nuwara Eliya", 
    "Galle", "Matara", "Hambantota", "Jaffna", "Kilinochchi[1]", "Mannar", 
    "Vavuniya", "Mullaitivu", "Batticaloa", "Ampara", "Trincomalee", 
    "Kurunegala", "Puttalam", "Anuradhapura", "Polonnaruwa", "Badulla", 
    "Moneragala", "Ratnapura", "Kegalle", "Welimada", "Bandarawela"
]

TRANSLATIONS = {
    "en": {
        "title": "Dengue Outbreak Prediction",
        "subtitle": "Overview",
        "welcome_desc": "Welcome to the Sri Lanka Dengue AI Forecaster! This machine learning model uses a decade of historical dengue case counts combined with regional weather patterns to predict the number of potentially upcoming cases for any given month and district. Our goal is to provide proactive insights for epidemic preparedness.",
        "btn_go_model": "Launch Forecast Model",
        "btn_model_info": "System & Analytics Info",
        "lang_selector": "Select Language / භාෂාව තෝරන්න",
        "model_title": "Dengue Outbreak Predictor",
        "model_desc": "Enter the required details below to forecast dengue outbreaks for a specific district.",
        "city_lbl": "Select District",
        "month_lbl": "Select Month",
        "lag1_lbl": "Cases Last Month (Value_lag1)",
        "precip_h_lbl": "Precipitation Hours (h)",
        "weather_code_lbl": "Weather Code (WMO code)",
        "precip_sum_lbl": "Precipitation Sum (mm)",
        "temp_lbl": "Mean Temperature (°C)",
        "predict_btn": "Forecast Cases",
        "back_btn": "← Back",
        "result_lbl": "Predicted Dengue Cases:",
        "info_title": "About the Model & Analytics",
        "info_desc": "This predictive engine leverages advanced machine learning to forecast dengue outbreaks across Sri Lanka. By capturing complex, non-linear relationships between climatic factors (such as rainfall and temperature) and historical case data, the system helps public health officials anticipate and prepare for potential epidemics.",
        "info_algorithm": "<h3 style='color: #00f2fe; margin-top: 0;'>Algorithm</h3><p>At the core of this system is a <b>Histogram-Based Gradient Boosting Regressor (HistGradientBoostingRegressor)</b> from scikit-learn. Chosen for its state-of-the-art performance on tabular data, this ensemble algorithm builds decision trees sequentially to correct past errors. It is highly optimized for large datasets, natively handles missing values, and utilizes a Log1p target transformation to accurately model extreme outbreak spikes.</p>",
        "info_table": """
<h3 style='color: #00f2fe; margin-top: 20px;'>Training & Architecture Breakdown</h3>
<table style='width: 100%; border-collapse: collapse; margin-top: 10px; color: #cbd5e1; font-size: 0.95rem; text-align: left;'>
    <tr style='border-bottom: 1px solid rgba(255,255,255,0.1); background: rgba(0,242,254,0.05);'>
        <th style='padding: 12px;'>Detail</th>
        <th style='padding: 12px;'>Specification</th>
    </tr>
    <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
        <td style='padding: 10px 12px; font-weight: 500; color: white;'>Base Regressor</td>
        <td style='padding: 10px 12px;'>HistGradientBoostingRegressor (squared_error loss)</td>
    </tr>
    <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
        <td style='padding: 10px 12px; font-weight: 500; color: white;'>Target Transform</td>
        <td style='padding: 10px 12px;'>Log1p (np.log1p) mapping for extreme outbreak handling</td>
    </tr>
    <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
        <td style='padding: 10px 12px; font-weight: 500; color: white;'>Chronological Split</td>
        <td style='padding: 10px 12px;'>Train (≤ 2017) | Validation (2018) | Test (≥ 2019)</td>
    </tr>
    <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
        <td style='padding: 10px 12px; font-weight: 500; color: white;'>Model Tuning</td>
        <td style='padding: 10px 12px;'>RandomizedSearchCV (max_iter, learning_rate, max_leaf_nodes)</td>
    </tr>
    <tr>
        <td style='padding: 10px 12px; font-weight: 500; color: white;'>Cross-Validation</td>
        <td style='padding: 10px 12px;'>TimeSeriesSplit (5 Splits) over combined Train/Val sets</td>
    </tr>
</table>
        """,
        "info_dataset": f"<h3 style='color: #00f2fe; margin-top: 0;'>Datasets Used</h3><p>The model was trained on the following official public datasets, which were rigorously preprocessed and geographically integrated on a monthly, district-wise basis (yielding exactly <b>{training_rows} training records</b>):</p><ul style='color: #e2e8f0; font-size: 1.05rem; line-height: 1.5;'><li><a href='{ds1_url}' target='_blank' style='color: #4facfe; text-decoration: none;'>{ds1_en}</a></li><li><a href='{ds2_url}' target='_blank' style='color: #4facfe; text-decoration: none;'>{ds2_en}</a></li></ul>",
        "info_xai": f"<h3 style='color: #00f2fe; margin-top: 0;'>Explainable AI (XAI)</h3><p>We analyze the model's decisions using <b>Permutation Importance (Decrease in Validation R²)</b> to ensure they align with epidemiological domain knowledge instead of purely acting as a 'black box'. The top influential factors identified are:</p><ul style='color: #e2e8f0; font-weight: 500; font-size: 1.05rem;'><li><span style='color: #00f2fe;'>🦟</span> <b>Historical Cases ({lag_pct}):</b> The number of cases in the previous month is the strongest predictor of the current month's cases.</li><li><span style='color: #00f2fe;'>🌧️</span> <b>Precipitation Duration & Amount ({precip_pct}):</b> The amount and duration of rain significantly impact mosquito breeding.</li><li><span style='color: #00f2fe;'>🌡️</span> <b>Average Temperature ({temp_pct}):</b> Warmer temperatures generally accelerate the mosquito life cycle and virus replication.</li><li><span style='color: #00f2fe;'>📍</span> <b>Location & Seasonal Constraints ({loc_pct}):</b> The specific geographical profile of the city and seasonal trends.</li></ul>",
        "info_metrics_title": "<h3 style='color: #00f2fe; margin-top: 0;'>Model Performance metrics</h3><p>The model was rigorously evaluated. Key metrics achieved on the test set:</p>",
        "metric_r2": "Accuracy (R² Score)",
        "metric_rmse": "RMSE",
        "metric_mae": "Mean Absolute Error",
        "month_names": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
        "help_city": "Select the district for which you want to forecast dengue cases.",
        "help_month": "Select the month of the year. Dengue cases typically vary by season.",
        "help_lag1": "The number of dengue cases reported in this district during the previous month. This is a strong predictor of current cases.",
        "help_precip_h": "Total duration of precipitation (rain) in hours over the month.",
        "help_weather_code": "Most frequent WMO weather code for the month (e.g., 51 for light drizzle, 61 for rain).",
        "help_precip_sum": "Total amount of precipitation (rain) in millimeters over the month.",
        "help_temp": "Average temperature at 2 meters above ground in Celsius.",
        "result_desc": "Based on the historical data and weather patterns provided, the model estimates approximately <b>{cases}</b> dengue cases for <b>{city}</b> in the selected month.",
        "risk_low": "🟢 Low Risk",
        "risk_med": "🟡 Medium Risk",
        "risk_high": "🔴 High Risk",
        "feature_imp_title": "Key Influencing Factors",
        "feature_imp_desc": "How much each of your inputs contributed to the current forecast:",
        "feat_lag1": "Historical Cases (Value_lag1)",
        "feat_precip": "Precipitation (mm & h)",
        "feat_temp": "Temperature (°C)",
        "feat_other": "Location & Month factor",
        "caption_lag1": "Valid range: 0 - 2,000 cases",
        "caption_precip_h": "Valid range: 0.0 - 24.0 h",
        "caption_weather_code": "Valid range: 0 - 99",
        "caption_precip_sum": "Valid range: 0.0 - 35.0 mm",
        "caption_temp": "Valid range: 15.0 - 45.0 °C",
        "footer_text": "Developed by <span>Kavindu Dilshan Arambage</span>.",
        "why_title": "Why this prediction?",
        "why_desc": "The AI used several factors to arrive at this forecast. Here is a breakdown of how your inputs influenced the prediction based on our global feature analysis:",
        "why_lag": "🦟 Previous Cases Impact",
        "why_lag_desc": "The <b>{lag_value}</b> cases reported last month strongly suggested the baseline transmission rate. A higher number typically carries over to the next month due to the continuous biological cycle of mosquitoes.",
        "why_precip": "🌧️ Rainfall & Precipitation",
        "why_precip_desc": "With <b>{precip_sum} mm</b> of rain over <b>{precip_h} hours</b>, the conditions created varying degrees of stagnant water, which act as primary breeding grounds for dengue vector mosquitoes.",
        "why_temp": "🌡️ Temperature Effect",
        "why_temp_desc": "The average temperature of <b>{temp_mean} °C</b> heavily influenced mosquito maturation and virus replication within them. Temperatures around 28-30 °C are highly optimal for rapid dengue transmission.",
        "why_loc": "📍 Location & Seasonal Constraints",
        "why_loc_desc": "The model also considered the specific geographical profile of <b>{city}</b> and seasonal trends for <b>Month {month}</b> to finalize the estimated {pred_val} cases.",
        "influence": "Influence"
    },
    "si": {
        "title": "ඩෙංගු රෝග ව්‍යාප්තිය පුරෝකථනය",
        "subtitle": "දළ විශ්ලේෂණය",
        "welcome_desc": "ශ්‍රී ලංකා ඩෙංගු AI පුරෝකථන පද්ධතියට සාදරයෙන් පිළිගනිමු! මෙම යන්ත්‍ර ඉගෙනුම් ආකෘතිය මඟින් දශකයක ඩෙංගු රෝගීන්ගේ ඓතිහාසික දත්ත සහ සවිස්තරාත්මක ප්‍රාදේශීය කාලගුණික දත්ත භාවිතා කර ඕනෑම දිස්ත්‍රික්කයක ඉදිරි මාසයේ වාර්තා විය හැකි රෝගීන් සංඛ්‍යාව බුද්ධිමත්ව පුරෝකථනය කරයි. අපගේ අරමුණ වසංගත සඳහා කල්තියා සූදානම් වීමට අවශ්‍ය අවබෝධය ලබා දීමයි.",
        "btn_go_model": "ආකෘතිය වෙත යන්න",
        "btn_model_info": "පද්ධතිය ගැන තොරතුරු",
        "lang_selector": "භාෂාව තෝරන්න / Select Language",
        "model_title": "ඩෙංගු පුරෝකථන ආකෘතිය",
        "model_desc": "විශේෂිත දිස්ත්‍රික්කයක් සඳහා ඩෙංගු පුරෝකථනය ලබා ගැනීමට පහත තොරතුරු ඇතුළත් කරන්න.",
        "city_lbl": "දිස්ත්‍රික්කය තෝරන්න",
        "month_lbl": "මාසය තෝරන්න",
        "lag1_lbl": "පසුගිය මාසයේ රෝගීන් ගණන (Value_lag1)",
        "precip_h_lbl": "වර්ෂාපතන පැය ගණන (h)",
        "weather_code_lbl": "කාලගුණ කේතය (WMO code)",
        "precip_sum_lbl": "මුළු වර්ෂාපතනය (මි.මී.)",
        "temp_lbl": "මධ්‍යම උෂ්ණත්වය (°C)",
        "predict_btn": "රෝගීන් පුරෝකථනය කරන්න",
        "back_btn": "← ආපසු",
        "result_lbl": "පුරෝකථනය කළ ඩෙංගු රෝගීන් ගණන:",
        "info_title": "ආකෘතිය සහ විශ්ලේෂණය ගැන",
        "info_desc": "ශ්‍රී ලංකාව පුරා ඩෙංගු ව්‍යාප්තිය පුරෝකථනය කිරීම සඳහා මෙම පද්ධතිය දියුණු යන්ත්‍ර ඉගෙනුම් (Machine Learning) තාක්ෂණය භාවිතා කරයි. වර්ෂාපතනය, උෂ්ණත්වය වැනි දේශගුණික සාධක සහ අතීත රෝගීන් සංඛ්‍යාව අතර ඇති සංකීර්ණ සම්බන්ධතා හඳුනාගැනීම හරහා, ඉදිරි වසංගත තත්ත්වයන් සඳහා කල්තියා සූදානම් වීමට මහජන සෞඛ්‍ය නිලධාරීන්ට මෙය සහාය වේ.",
        "info_algorithm": "<h3 style='color: #00f2fe; margin-top: 0;'>ඇල්ගොරිතම (Algorithm)</h3><p>මෙම පද්ධතියේ ප්‍රධානතම තාක්ෂණය වන්නේ scikit-learn හි <b>Histogram-Based Gradient Boosting Regressor</b> ය. විශාල දත්ත කට්ටල සඳහා ඉතා වේගවත් හා නිවැරදි ප්‍රතිඵල ලබාදෙන මෙම ක්‍රමය, අස්ථානගත වූ දත්ත (missing values) ස්වයංක්‍රීයව හසුරුවයි. එසේම, ඩෙංගු රෝගීන් අනපේක්ෂිත ලෙස ඉහළ යන අවස්ථා නිවැරදිව හඳුනාගැනීමට මෙහිදී Log1p ඉලක්ක පරිවර්තනය (target transformation) භාවිතා කර ඇත.</p>",
        "info_table": """
<h3 style='color: #00f2fe; margin-top: 20px;'>පුහුණු කිරීම සහ ආකෘති සැලසුම</h3>
<table style='width: 100%; border-collapse: collapse; margin-top: 10px; color: #cbd5e1; font-size: 0.95rem; text-align: left;'>
    <tr style='border-bottom: 1px solid rgba(255,255,255,0.1); background: rgba(0,242,254,0.05);'>
        <th style='padding: 12px;'>විස්තරය (Detail)</th>
        <th style='padding: 12px;'>පිරිවිතර (Specification)</th>
    </tr>
    <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
        <td style='padding: 10px 12px; font-weight: 500; color: white;'>මූලික ආකෘතිය</td>
        <td style='padding: 10px 12px;'>HistGradientBoostingRegressor (squared_error loss)</td>
    </tr>
    <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
        <td style='padding: 10px 12px; font-weight: 500; color: white;'>ඉලක්ක පරිවර්තනය</td>
        <td style='padding: 10px 12px;'>අනපේක්ෂිත වැඩිවීම් පාලනයට Log1p (np.log1p) භාවිතය</td>
    </tr>
    <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
        <td style='padding: 10px 12px; font-weight: 500; color: white;'>කාලානුක්‍රමික බෙදීම</td>
        <td style='padding: 10px 12px;'>තරු කිරීම (≤ 2017) | තහවුරු කිරීම (2018) | පරීක්ෂණ (≥ 2019)</td>
    </tr>
    <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
        <td style='padding: 10px 12px; font-weight: 500; color: white;'>ආකෘතිය සුසර කිරීම</td>
        <td style='padding: 10px 12px;'>RandomizedSearchCV (max_iter, learning_rate, max_leaf_nodes)</td>
    </tr>
    <tr>
        <td style='padding: 10px 12px; font-weight: 500; color: white;'>හරස් වලංගුකරණය</td>
        <td style='padding: 10px 12px;'>කාල ශ්‍රේණි බෙදීම (TimeSeriesSplit) - කාණ්ඩ 5ක්</td>
    </tr>
</table>
        """,
        "info_dataset": f"<h3 style='color: #00f2fe; margin-top: 0;'>දත්ත කට්ටල (Datasets)</h3><p>මෙම ආකෘතියේ පුරෝකථන හැකියාව ලබාගෙන ඇත්තේ දශකයකට අධික කාලයක පහත දැක්වෙන නිල ලබා ගත හැකි දත්ත කට්ටල, මාසික හා දිස්ත්‍රික්ක වශයෙන් පූර්ව සැකසුම් (preprocessed) කර භූගෝලීයව ඒකාබද්ධ කිරීමෙනි (මෙය නිවැරදිව ලබා දෙන <b>පුහුණු දත්ත වාර්තා {training_rows} කින්</b> සමන්විත වේ):</p><ul style='color: #e2e8f0; font-size: 1.05rem; line-height: 1.5;'><li><a href='{ds1_url}' target='_blank' style='color: #4facfe; text-decoration: none;'>{ds1_si}</a></li><li><a href='{ds2_url}' target='_blank' style='color: #4facfe; text-decoration: none;'>{ds2_si}</a></li></ul>",
        "info_xai": f"<h3 style='color: #00f2fe; margin-top: 0;'>පැහැදිලි කළ හැකි AI (XAI)</h3><p><b>Permutation Importance (සාධක වෙනස් වීම අනුව පරීක්ෂණ දෝෂය විශ්ලේෂණය)</b> තාක්ෂණය භාවිතා කරමින් ආකෘතියේ තීරණ වෛද්‍ය විද්‍යාත්මක දැනුම හා ගැලපෙන බව තහවුරු කිරීමට අපි ඒවා විශ්ලේෂණය කරමු. මෙහිදී හඳුනාගත් ප්‍රධාන සාධක වන්නේ:</p><ul style='color: #e2e8f0; font-weight: 500; font-size: 1.05rem;'><li><span style='color: #00f2fe;'>🦟</span> <b>අතීත රෝගීන් ගණන ({lag_pct}):</b> පසුගිය මාසයේ ඇති වූ රෝගීන් සංඛ්‍යාව වත්මන් මාසයේ රෝගීන් පුරෝකථනය කිරීමට ප්‍රධානතම සාධකය වේ.</li><li><span style='color: #00f2fe;'>🌧️</span> <b>වර්ෂාපතන කාලය සහ ප්‍රමාණය ({precip_pct}):</b> වැසි ලැබෙන ප්‍රමාණය සහ කාලසීමාව මදුරුවන් බෝවීමට සැලකිය යුතු බලපෑමක් කරයි.</li><li><span style='color: #00f2fe;'>🌡️</span> <b>සාමාන්‍ය උෂ්ණත්වය ({temp_pct}):</b> ඉහළ උෂ්ණත්වය මදුරුවන්ගේ ජීවන චක්‍රය සහ වෛරසය ව්‍යාප්තිය වේගවත් කරයි.</li><li><span style='color: #00f2fe;'>📍</span> <b>ස්ථානය සහ සෘතුමය බලපෑම් ({loc_pct}):</b> දිස්ත්‍රික්කයේ භූගෝලීය ස්වභාවය සහ මාසය අනුව පවතින සෘතුමය වෙනස්කම්.</li></ul>",
        "info_metrics_title": "<h3 style='color: #00f2fe; margin-top: 0;'>ආකෘතියේ කාර්ය සාධනය</h3><p>ආකෘතිය පරීක්ෂණ දත්ත මත ඇගයීමට ලක් කර ඇත. ලබාගත් ප්‍රධාන ප්‍රතිඵල:</p>",
        "metric_r2": "නිරවද්‍යතාව (R² Score)",
        "metric_rmse": "RMSE අගය",
        "metric_mae": "මධ්‍යම නිරපේක්ෂ දෝෂය",
        "month_names": ["ජනවාරි", "පෙබරවාරි", "මාර්තු", "අප්‍රේල්", "මැයි", "ජූනි", "ජූලි", "අගෝස්තු", "සැප්තැම්බර්", "ඔක්තෝබර්", "නොවැම්බර්", "දෙසැම්බර්"],
        "help_city": "ඔබට ඩෙංගු රෝගීන් පුරෝකථනය කිරීමට අවශ්‍ය දිස්ත්‍රික්කය තෝරන්න.",
        "help_month": "වසරේ මාසය තෝරන්න. ඩෙංගු රෝගීන් ගණන කාලගුණය අනුව වෙනස් වේ.",
        "help_lag1": "පසුගිය මාසය තුළ මෙම දිස්ත්‍රික්කයේ වාර්තා වූ ඩෙංගු රෝගීන් ගණන. මෙය ප්‍රබල පුරෝකථන සාධකයකි.",
        "help_precip_h": "මාසය තුළ වර්ෂාපතනය (වැසි) ලැබුණු මුළු පැය ගණන.",
        "help_weather_code": "මාසය සඳහා බහුලවම පැවති WMO කාලගුණ කේතය (උදා: සුළු වැස්ස සඳහා 51, වර්ෂාව සඳහා 61).",
        "help_precip_sum": "මාසය තුළ මුළු වර්ෂාපතනය (මිලිමීටර් වලින්).",
        "help_temp": "පොළොව මට්ටමේ සිට මීටර් 2 ක් ඉහළින් පවතින සාමාන්‍ය උෂ්ණත්වය (සෙල්සියස් වලින්).",
        "result_desc": "ලබා දී ඇති ඓතිහාසික දත්ත සහ කාලගුණික රටා මත පදනම්ව, තෝරාගත් මාසයේ <b>{city}</b> දිස්ත්‍රික්කය සඳහා ආසන්න වශයෙන් ඩෙංගු රෝගීන් <b>{cases}</b> ක් ඇතිවිය හැකි බවට ආකෘතිය ඇස්තමේන්තු කරයි.",
        "risk_low": "🟢 අඩු අවදානමක්",
        "risk_med": "🟡 මධ්‍යම අවදානමක්",
        "risk_high": "🔴 ඉහළ අවදානමක්",
        "feature_imp_title": "ප්‍රධාන බලපාන සාධක",
        "feature_imp_desc": "වත්මන් පුරෝකථනය සඳහා ඔබේ එක් එක් ආදාන කොපමණ දායකත්වයක් ලබා දුන්නේද:",
        "feat_lag1": "පෙර රෝගීන් ගණන (Value_lag1)",
        "feat_precip": "වර්ෂාපතනය (මි.මී. සහ පැය)",
        "feat_temp": "උෂ්ණත්වය (°C)",
        "feat_other": "පිහිටීම සහ මාසය",
        "caption_lag1": "වලංගු පරාසය: 0 - 2,000 රෝගීන්",
        "caption_precip_h": "වලංගු පරාසය: 0.0 - 24.0 පැය",
        "caption_weather_code": "වලංගු පරාසය: 0 - 99",
        "caption_precip_sum": "වලංගු පරාසය: 0.0 - 35.0 මි.මී.",
        "caption_temp": "වලංගු පරාසය: 15.0 - 45.0 °C",
        "footer_text": "සංවර්ධනය කළේ <span>කවිඳු දිල්ශාන් අරඹගේ</span> විසිනි.",
        "why_title": "මෙම පුරෝකථනයට හේතුව?",
        "why_desc": "මෙම පුරෝකථනය සඳහා AI සාධක කිහිපයක් භාවිතා කර ඇත. ගෝලීය විශ්ලේෂණ වලට අනුව ඔබේ ආදානයන් පුරෝකථනයට බලපෑවේ කෙසේද යන්න මෙහි දැක්වේ:",
        "why_lag": "🦟 අතීත රෝගීන්ගේ බලපෑම",
        "why_lag_desc": "පසුගිය මාසයේ වාර්තා වූ රෝගීන් <b>{lag_value}</b> ක් මූලික සම්ප්‍රේෂණ අනුපාතය දැඩි ලෙස යෝජනා කළේය. මදුරුවන්ගේ අඛණ්ඩ ජීව විද්‍යාත්මක චක්‍රය හේතුවෙන් ඉහළ සංඛ්‍යාවක් සාමාන්‍යයෙන් ඊළඟ මාසයට ගෙන යයි.",
        "why_precip": "🌧️ වර්ෂාපතනය සහ වර්ෂාව",
        "why_precip_desc": "<b>පැය {precip_h}</b> ක කාලයක් තුළ මි.මී. <b>{precip_sum}</b> ක වර්ෂාපතනයක් සමඟින්, ඩෙංගු වාහක මදුරුවන්ගේ ප්‍රධාන අභිජනන ස්ථාන ලෙස ක්‍රියා කරන විවිධ මට්ටම්වල එකතැන පල්වෙන ජල තත්වයන් නිර්මාණය විය.",
        "why_temp": "🌡️ උෂ්ණත්වයේ බලපෑම",
        "why_temp_desc": "සාමාන්‍ය උෂ්ණත්වය <b>{temp_mean} °C</b> ක් වීම මදුරුවන් පරිණත වීමට සහ ඔවුන් තුළ වෛරස් ප්‍රතිවර්තනය වීමට දැඩි ලෙස බලපෑවේය. අපේක්ෂිත වේගවත් ඩෙංගු සම්ප්‍රේෂණය සඳහා 28-30 °C පමණ උෂ්ණත්වය ඉතා ප්‍රශස්ත වේ.",
        "why_loc": "📍 ස්ථානය සහ සෘතුමය බලපෑම්",
        "why_loc_desc": "ඇස්තමේන්තුගත රෝගීන් {pred_val} ක් සඳහා, ආකෘතිය විසින් <b>{city}</b> හි නිශ්චිත භූගෝලීය ස්වභාවය සහ <b>{month} මාසය</b> සඳහා සෘතුමය වෙනස්කම් ද සලකා බලන ලදී.",
        "influence": "බලපෑම"
    }
}

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Dengue Outbreak Predictor",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
/* Base UI text and layout improvements */
body, .stApp {
    background: radial-gradient(circle at 10% 20%, rgb(0, 52, 89) 0%, rgb(0, 0, 0) 90%);
    color: #e2e8f0;
}
[data-testid="stAppViewContainer"] {
    background: transparent;
}
[data-testid="stHeader"] {
    background-color: transparent;
}
h1, h2, h3, p, span, div, label {
    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
}
.hero-title {
    font-size: 4rem;
    font-weight: 900;
    text-align: center;
    margin-bottom: 0px;
    padding-top: 50px;
    background: -webkit-linear-gradient(45deg, #00f2fe, #4facfe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0px 4px 10px rgba(0,242,254,0.3);
}
.hero-subtitle {
    font-size: 1.4rem;
    color: #cbd5e1;
    text-align: center;
    margin-top: 15px;
    margin-bottom: 50px;
    font-weight: 300;
}
/* Modern Button Styling */
div.stButton > button {
    background: rgba(255, 255, 255, 0.05);
    color: #ffffff;
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 12px;
    padding: 15px 30px;
    font-size: 1.2rem;
    font-weight: 600;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
div.stButton > button:hover {
    background: rgba(255, 255, 255, 0.15);
    border-color: #00f2fe;
    box-shadow: 0 0 20px rgba(0, 242, 254, 0.4);
    transform: translateY(-2px);
    color: #ffffff;
}
div.stButton > button:active {
    transform: translateY(1px);
}
.primary-btn > div > button {
    background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
    border: none;
    color: white;
}
.primary-btn > div > button:hover {
    background: linear-gradient(135deg, #0072ff 0%, #00c6ff 100%);
    box-shadow: 0 8px 25px rgba(0, 114, 255, 0.5);
    border: none;
}
/* Input boxes */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
    background-color: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: white !important;
    border-radius: 8px;
    transition: all 0.3s ease;
}
/* Modern Glassmorphism Cards for Containers */
[data-testid="stVerticalBlockBorderWrapper"] > div {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 20px !important;
    padding: 10px 10px !important;
    box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.3) !important;
    backdrop-filter: blur(10px) !important;
}
div[data-baseweb="select"] > div:hover, div[data-baseweb="input"] > div:hover {
    border-color: rgba(255,255,255,0.3) !important;
}
.result-box {
    text-align: center;
    padding: 30px;
    background: linear-gradient(135deg, rgba(0,242,254,0.1), rgba(79,172,254,0.1));
    border: 1px solid rgba(0,242,254,0.3);
    border-radius: 20px;
    margin-top: 30px;
    color: white;
    backdrop-filter: blur(10px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    animation: fadeIn 0.5s ease-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.result-val {
    font-size: 2.5rem;
    font-weight: 900;
    color: #00f2fe;
    margin: 15px 0 0 0;
    line-height: 1.1;
    text-shadow: 0 0 30px rgba(0,242,254,0.6);
}
.risk-badge {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 1.2rem;
    font-weight: bold;
    margin-top: 15px;
    background: rgba(255,255,255,0.1);
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
}
.info-section {
    background: rgba(255,255,255,0.03);
    border-radius: 15px;
    padding: 25px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.05);
    transition: transform 0.3s ease;
}
.info-section:hover {
    transform: translateY(-5px);
    background: rgba(255,255,255,0.05);
    border-color: rgba(0,242,254,0.3);
}
.feature-bar-wrapper {
    display: flex;
    align-items: center;
    margin-top: 10px;
    margin-bottom: 10px;
}
.feature-label {
    flex: 1;
    font-size: 0.95rem;
    color: #e2e8f0;
    text-align: left;
}
.feature-bar-container {
    flex: 2;
    background: rgba(255,255,255,0.1);
    height: 10px;
    border-radius: 5px;
    margin: 0 15px;
    overflow: hidden;
}
.feature-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    border-radius: 5px;
}
.feature-val {
    width: 40px;
    font-size: 0.9rem;
    font-weight: bold;
    color: #00f2fe;
    text-align: right;
}
.metric-box {
    background: linear-gradient(135deg, rgba(0,198,255,0.1) 0%, rgba(0,114,255,0.1) 100%);
    border: 1px solid rgba(0, 198, 255, 0.3);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: transform 0.3s ease;
}
.metric-box:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0,0,0,0.2);
}
.metric-value {
    font-size: 2.5rem;
    font-weight: bold;
    background: -webkit-linear-gradient(45deg, #00f2fe, #4facfe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}
.metric-label {
    font-size: 0.9rem;
    color: #cbd5e1;
    font-weight: 500;
}

/* Fix Streamlit column wrapping on strictly mobile screens */
@media (max-width: 600px) {
    .stButton > button {
        padding: 5px 10px !important;
        font-size: 0.8rem !important;
        min-height: 32px !important;
    }
    div[data-baseweb="select"] > div {
        padding: 0px 5px !important;
        font-size: 0.85rem !important;
        min-height: 32px !important;
    }
    /* Provide gap for squished container boxes */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        margin-bottom: 20px !important;
    }
    .hero-title {
        font-size: 2.2rem;
    }
    .hero-subtitle {
        font-size: 1rem;
    }
}

/* Footer Styling */
.footer {
    text-align: center;
    padding: 30px 0 15px 0;
    margin-top: 50px;
    font-size: 0.9rem;
    color: rgba(255, 255, 255, 0.6);
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}
.footer span {
    color: #00f2fe;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'

lang = st.session_state.lang
t = TRANSLATIONS[lang]

# Top-level language toggle
col1, col2 = st.columns([8, 2])
with col2:
    selected_lang = st.pills(
        label="Language Toggle",
        options=['EN', 'SI'],
        selection_mode="single",
        default='EN' if lang == 'en' else 'SI',
        label_visibility="collapsed"
    )
    if selected_lang:
        new_lang = 'en' if selected_lang == 'EN' else 'si'
        if new_lang != st.session_state.lang:
            st.session_state.lang = new_lang
            st.rerun()

# --- ROUTING ---

if st.session_state.page == 'home':
    st.markdown(f"<h1 class='hero-title' style='margin-bottom: 20px;'>{t['title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background: rgba(0,0,0,0.3); padding: 30px; border-radius: 20px; border: 1px solid rgba(0,242,254,0.3); text-align: center; max-width: 800px; margin: 0 auto; box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.4); backdrop-filter: blur(10px);'>
        <p style='color: #cbd5e1; font-size: 1.15rem; line-height: 1.6; margin: 0;'>{t['welcome_desc']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    st.write("")
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<div class='primary-btn'>", unsafe_allow_html=True)
        if st.button(t['btn_go_model'], use_container_width=True):
            st.session_state.page = 'model'
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.write("")
        st.write("")
        
        if st.button(t['btn_model_info'], use_container_width=True):
            st.session_state.page = 'info'
            st.rerun()

elif st.session_state.page == 'model':
    # Navigation header
    b_col1, b_col2 = st.columns([1, 6])
    with b_col1:
        if st.button(t['back_btn']):
            st.session_state.page = 'home'
            st.rerun()
            
    st.markdown(f"<h2 style='text-align: center; color: white;'>{t['model_title']}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #cbd5e1; margin-bottom: 40px;'>{t['model_desc']}</p>", unsafe_allow_html=True)
    
    # Input Form
    with st.container():
        col_a, empty_col, col_b = st.columns([5, 1, 5])
        
        with col_a:
            with st.container(border=True):
                st.markdown(f"<h4 style='color: #00f2fe; margin-bottom: 25px; font-weight: 600; font-size: 1.5rem;'>📍 Location & History</h4>", unsafe_allow_html=True)
                city = st.selectbox(t['city_lbl'], options=CITIES, help=t['help_city'])
                month_name = st.selectbox(t['month_lbl'], options=t['month_names'], help=t['help_month'])
                month = t['month_names'].index(month_name) + 1
                lag_value = st.slider(t['lag1_lbl'], min_value=0, max_value=2000, value=50, step=1, help=t['help_lag1'])
            
        with col_b:
            with st.container(border=True):
                st.markdown(f"<h4 style='color: #00f2fe; margin-bottom: 25px; font-weight: 600; font-size: 1.5rem;'>🌧️ 🌡️ Weather Conditions</h4>", unsafe_allow_html=True)
                precip_h = st.slider(t['precip_h_lbl'], min_value=0.0, max_value=720.0, value=50.0, step=1.0, help=t['help_precip_h'])
                precip_sum = st.slider(t['precip_sum_lbl'], min_value=0.0, max_value=1000.0, value=150.0, step=5.0, help=t['help_precip_sum'])
                temp_mean = st.slider(t['temp_lbl'], min_value=15.0, max_value=45.0, value=28.0, step=0.1, help=t['help_temp'])
            
    st.write("")
    st.write("")
    
    # Predict Action
    p_c1, p_c2, p_c3 = st.columns([1,2,1])
    with p_c2:
        st.markdown("<div class='primary-btn'>", unsafe_allow_html=True)
        is_pred_clicked = st.button(t['predict_btn'], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if is_pred_clicked:
            input_data = {
                "City": city,
                "Month": month,
                "Value_lag1": lag_value,
                "precipitation_hours (h)_lag1": precip_h,
                "precipitation_sum (mm)_lag1": precip_sum,
                "temperature_2m_mean (°C)_lag1": temp_mean
            }
            with st.spinner("Analyzing data..."):
                try:
                    prediction = predict(input_data, config)
                    pred_val = int(prediction)
                    
                    if pred_val < 50:
                        risk_level = t['risk_low']
                        color = "#4ade80" # Green
                    elif pred_val < 150:
                        risk_level = t['risk_med']
                        color = "#facc15" # Yellow
                    else:
                        risk_level = t['risk_high']
                        color = "#ef4444" # Red
                        
                    html_content = f"""
<div class='result-box' id='prediction-result'>
<p style='font-size: 1.3rem; margin: 0; font-weight: 300; color: #e2e8f0;'>{t['result_lbl']}</p>
<div style='margin: 15px 0;'><span class='result-val'>{pred_val}</span></div>
<div class='risk-badge' style='color: {color};'>{risk_level}</div>
<p style='font-size: 1.1rem; margin-top: 20px; margin-bottom: 30px; color: #cbd5e1; line-height: 1.5;'>{t['result_desc'].format(cases=pred_val, city=city)}</p>
</div>
<div style='background: rgba(0,0,0,0.2); padding: 25px; border-radius: 20px; text-align: left; margin-top: 30px; border: 1px solid rgba(0,242,254,0.2); backdrop-filter: blur(10px);'>
<h3 style="color: #00f2fe; margin-bottom: 15px; font-weight: 800;">{t['why_title']}</h3>
<p style="color: #cbd5e1; font-size: 1.05rem; margin-bottom: 15px; line-height: 1.6;">{t['why_desc']}</p>
<div style="margin-bottom: 20px; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 12px; border-left: 4px solid #00f2fe;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <h4 style="color: white; font-size: 1.1rem; margin: 0;">{t['why_lag']}</h4>
        <span style="background: rgba(0,242,254,0.2); color: #00f2fe; padding: 4px 10px; border-radius: 12px; font-size: 0.9rem; font-weight: bold;">{lag_pct} {t['influence']}</span>
    </div>
    <p style="color: #cbd5e1; font-size: 0.95rem; margin: 0; line-height: 1.5;">{t['why_lag_desc'].format(lag_value=lag_value)}</p>
</div>
<div style="margin-bottom: 20px; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 12px; border-left: 4px solid #4ade80;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <h4 style="color: white; font-size: 1.1rem; margin: 0;">{t['why_precip']}</h4>
        <span style="background: rgba(74,222,128,0.2); color: #4ade80; padding: 4px 10px; border-radius: 12px; font-size: 0.9rem; font-weight: bold;">{precip_pct} {t['influence']}</span>
    </div>
    <p style="color: #cbd5e1; font-size: 0.95rem; margin: 0; line-height: 1.5;">{t['why_precip_desc'].format(precip_sum=precip_sum, precip_h=precip_h)}</p>
</div>
<div style="margin-bottom: 20px; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 12px; border-left: 4px solid #facc15;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <h4 style="color: white; font-size: 1.1rem; margin: 0;">{t['why_temp']}</h4>
        <span style="background: rgba(250,204,21,0.2); color: #facc15; padding: 4px 10px; border-radius: 12px; font-size: 0.9rem; font-weight: bold;">{temp_pct} {t['influence']}</span>
    </div>
    <p style="color: #cbd5e1; font-size: 0.95rem; margin: 0; line-height: 1.5;">{t['why_temp_desc'].format(temp_mean=temp_mean)}</p>
</div>
<div style="padding: 15px; background: rgba(255,255,255,0.05); border-radius: 12px; border-left: 4px solid #ef4444;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <h4 style="color: white; font-size: 1.1rem; margin: 0;">{t['why_loc']}</h4>
        <span style="background: rgba(239,68,68,0.2); color: #ef4444; padding: 4px 10px; border-radius: 12px; font-size: 0.9rem; font-weight: bold;">{loc_pct} {t['influence']}</span>
    </div>
    <p style="color: #cbd5e1; font-size: 0.95rem; margin: 0; line-height: 1.5;">{t['why_loc_desc'].format(city=city, month=t['month_names'][month-1], pred_val=pred_val)}</p>
</div>
</div>
"""
                    
                    st.markdown(html_content, unsafe_allow_html=True)
                    
                    # Auto-scroll to the result via a small HTML component with a slight delay
                    import streamlit.components.v1 as components
                    components.html(
                        '''
                        <script>
                            setTimeout(function() {
                                const elements = window.parent.document.getElementsByClassName('result-box');
                                if (elements.length > 0) {
                                    elements[0].scrollIntoView({behavior: 'smooth', block: 'start'});
                                }
                            }, 150);
                        </script>
                        ''', 
                        height=0
                    )
                except Exception as e:
                    st.error(f"Error making prediction: {e}")

elif st.session_state.page == 'info':
    # Navigation header
    b_col1, b_col2 = st.columns([1, 6])
    with b_col1:
        if st.button(t['back_btn']):
            st.session_state.page = 'home'
            st.rerun()
            
    st.markdown(f"<h2 style='text-align: center; color: white; margin-bottom: 40px;'>{t['info_title']}</h2>", unsafe_allow_html=True)
    
    # Detailed Info
    i_col1, i_col2, i_col3 = st.columns([1, 8, 1])
    with i_col2:
        st.markdown(f"<div class='info-section'><h3 style='color: #00f2fe; margin-top: 0;'>Overview</h3><p>{t['info_desc']}</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-section'>{t['info_dataset']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-section'>{t['info_algorithm']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-section' style='padding-top: 5px;'>{t['info_table']}</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-section'>
            {t['info_xai']}
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics Section
        st.markdown(f"<div class='info-section'>{t['info_metrics_title']}</div>", unsafe_allow_html=True)
        
        try:
            if features_path.exists():
                r2_val = f"{features_metadata['metrics']['R2'] * 100:.2f}%"
                rmse_val = f"{features_metadata['metrics']['RMSE']:.2f}"
                mae_val = f"{features_metadata['metrics']['MAE']:.2f}"
            else:
                r2_val, rmse_val, mae_val = "N/A", "N/A", "N/A"
        except Exception as e:
            r2_val, rmse_val, mae_val = "N/A", "N/A", "N/A"
            
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-value'>{r2_val}</div>
                <div class='metric-label'>{t['metric_r2']}</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-value'>{rmse_val}</div>
                <div class='metric-label'>{t['metric_rmse']}</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-value'>{mae_val}</div>
                <div class='metric-label'>{t['metric_mae']}</div>
            </div>
            """, unsafe_allow_html=True)
        st.write("")
        st.write("")
        st.write("")

# --- FOOTER ---
def render_footer():
    st.markdown(f"""
    <div class="footer">
        {t['footer_text']}
    </div>
    """, unsafe_allow_html=True)

render_footer()
