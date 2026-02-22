import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
ROOT = Path("C:/ML")
RAW_DENGUE = ROOT / "data/raw/Dengue_Data (2010-2020).xlsx"
PROCESSED_DATA = ROOT / "data/processed/integrated_data.csv"
OUT_DIR = ROOT / "reports/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

def plot_target_distribution():
    print("Plotting Target Distribution (Value/Cases)...")
    # Load raw data
    raw_df = pd.read_excel(RAW_DENGUE)
    # Load processed data
    proc_df = pd.read_csv(PROCESSED_DATA)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw target (Monthly Cases)
    sns.histplot(raw_df['Value'].dropna(), bins=50, kde=True, ax=axes[0], color='salmon')
    axes[0].set_title('Raw Distribution of Dengue Cases (Highly Skewed)')
    axes[0].set_xlabel('Monthly Dengue Cases')
    axes[0].set_ylabel('Frequency')
    
    # Transformed target (Log1p transformation used in Model)
    # The Log1p transform happens inside TransformedTargetRegressor, but we simulate it here for visualization
    transformed_target = np.log1p(proc_df['Value'].dropna())
    sns.histplot(transformed_target, bins=50, kde=True, ax=axes[1], color='skyblue')
    axes[1].set_title('Target Transformation (Log1p)')
    axes[1].set_xlabel('Log1p(Monthly Dengue Cases)')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'target_distribution_transformation.png', dpi=150)
    plt.close()

def plot_feature_distributions():
    print("Plotting Feature Distributions...")
    proc_df = pd.read_csv(PROCESSED_DATA)
    
    features_to_plot = [
        ('temperature_2m_mean (°C)_lag1', 'Mean Temperature (°C) Lag 1', 'darkorange'),
        ('precipitation_sum (mm)_lag1', 'Precipitation Sum (mm) Lag 1', 'steelblue')
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for i, (col, title, color) in enumerate(features_to_plot):
        if col in proc_df.columns:
            sns.histplot(proc_df[col].dropna(), bins=40, kde=True, ax=axes[i], color=color)
            axes[i].set_title(f'Distribution: {title}')
            axes[i].set_xlabel(title)
            axes[i].set_ylabel('Frequency')
            
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'feature_distributions.png', dpi=150)
    plt.close()

def plot_correlation_matrix():
    print("Plotting Correlation Matrix...")
    proc_df = pd.read_csv(PROCESSED_DATA)
    
    cols = ['Value', 'Value_lag1', 'precipitation_hours (h)_lag1', 
            'precipitation_sum (mm)_lag1', 'temperature_2m_mean (°C)_lag1']
            
    existing_cols = [c for c in cols if c in proc_df.columns]
    
    if len(existing_cols) > 1:
        corr = proc_df[existing_cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'correlation_matrix.png', dpi=150)
        plt.close()

def plot_time_series():
    print("Plotting Time Series trend...")
    raw_df = pd.read_excel(RAW_DENGUE)
    
    raw_df['Date'] = pd.to_datetime(raw_df['Date'])
    monthly_total = raw_df.groupby('Date')['Value'].sum().reset_index()
    
    plt.figure(figsize=(14, 5))
    plt.plot(monthly_total['Date'], monthly_total['Value'], color='#e74c3c', linewidth=2)
    plt.title('Total Monthly Dengue Cases in Sri Lanka (2010 - 2020)')
    plt.xlabel('Year')
    plt.ylabel('Total Cases')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'time_series_trend.png', dpi=150)
    plt.close()

if __name__ == "__main__":
    plot_target_distribution()
    plot_feature_distributions()
    plot_correlation_matrix()
    plot_time_series()
    print("Plots generated successfully in reports/figures/")
