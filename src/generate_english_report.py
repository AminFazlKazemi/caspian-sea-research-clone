#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate English Report with Full Tables
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# ============================================
# Settings
# ============================================
BASE_DIR = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503"
OUTPUT_PDF = os.path.join(BASE_DIR, "lake_border", "Caspian_Lake_Analysis_Report_En.pdf")
os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)

LAKE_ANALYSIS_DIR = os.path.join(BASE_DIR, "caspian_lake_complete_analysis")
COMPARISON_DIR = os.path.join(BASE_DIR, "basin_lake_comparison")

print("="*70)
print("Generating English Report with Full Tables")
print("="*70)

# ============================================
# 1. Load Data
# ============================================
print("\n📊 Loading data...")

def load_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame()

df_mk = load_csv(os.path.join(LAKE_ANALYSIS_DIR, "mk_annual.csv"))
df_bootstrap = load_csv(os.path.join(LAKE_ANALYSIS_DIR, "bootstrap_trend.csv"))
df_pettitt = load_csv(os.path.join(LAKE_ANALYSIS_DIR, "pettitt.csv"))
df_quantile = load_csv(os.path.join(LAKE_ANALYSIS_DIR, "quantile_regression.csv"))
df_enso = load_csv(os.path.join(LAKE_ANALYSIS_DIR, "enso_correlation.csv"))
df_composite = load_csv(os.path.join(LAKE_ANALYSIS_DIR, "composite_results.csv"))
df_rf = load_csv(os.path.join(LAKE_ANALYSIS_DIR, "rf_importance.csv"))
df_comparison = load_csv(os.path.join(COMPARISON_DIR, "comparison_statistics.csv"))

print(f"   ✅ mk_annual: {len(df_mk)} records")
print(f"   ✅ bootstrap: {len(df_bootstrap)} records")
print(f"   ✅ pettitt: {len(df_pettitt)} records")
print(f"   ✅ quantile: {len(df_quantile)} records")
print(f"   ✅ enso: {len(df_enso)} records")
print(f"   ✅ composite: {len(df_composite)} records")
print(f"   ✅ rf: {len(df_rf)} records")
print(f"   ✅ comparison: {len(df_comparison)} records")

# ============================================
# 2. Collect Images
# ============================================
print("\n📸 Collecting images...")

all_images = []
patterns = [
    (LAKE_ANALYSIS_DIR, "bootstrap_*.png"),
    (LAKE_ANALYSIS_DIR, "pettitt_*.png"),
    (LAKE_ANALYSIS_DIR, "quantile_*.png"),
    (LAKE_ANALYSIS_DIR, "arima_*.png"),
    (LAKE_ANALYSIS_DIR, "wavelet_*.png"),
    (LAKE_ANALYSIS_DIR, "extreme_*.png"),
    (LAKE_ANALYSIS_DIR, "seasonal_boxplot_*.png"),
    (LAKE_ANALYSIS_DIR, "seasonal_ts_*.png"),
    (LAKE_ANALYSIS_DIR, "decadal_*.png"),
    (LAKE_ANALYSIS_DIR, "stl_*.png"),
    (LAKE_ANALYSIS_DIR, "spectrum_*.png"),
    (COMPARISON_DIR, "timeseries_comparison.png"),
    (COMPARISON_DIR, "scatter_comparison.png"),
    (COMPARISON_DIR, "trend_comparison_bar.png"),
    (COMPARISON_DIR, "correlation_heatmap.png"),
]

for base_dir, pattern in patterns:
    for f in glob.glob(os.path.join(base_dir, pattern)):
        if os.path.exists(f):
            try:
                Image.open(f).verify()
                all_images.append(f)
            except:
                pass

print(f"   ✅ {len(all_images)} valid images found.")

# ============================================
# 3. Generate PDF
# ============================================
print("\n📄 Generating PDF report...")

def add_table(pdf, df, title, description, columns, col_labels):
    """Add table to PDF"""
    if df.empty:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')
        ax.text(0.05, 0.95, title, fontsize=16, fontweight='bold', va='top')
        ax.text(0.05, 0.88, description, fontsize=11, va='top', wrap=True)
        ax.text(0.5, 0.5, '⚠️ No data available for this table.', fontsize=14, ha='center', va='center')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        return
    
    # Select available columns
    available = [c for c in columns if c in df.columns]
    if not available:
        df_display = df.copy()
        cols = list(df.columns)
        labels = cols
    else:
        df_display = df[available].copy()
        cols = available
        labels = col_labels[:len(available)]
    
    # Convert data to strings
    data = df_display.values.tolist()
    for i in range(len(data)):
        for j in range(len(data[i])):
            if isinstance(data[i][j], (int, float)):
                data[i][j] = f"{data[i][j]:.4f}"
            else:
                data[i][j] = str(data[i][j])
    
    n_cols = len(cols)
    
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis('off')
    ax.text(0.05, 0.95, title, fontsize=16, fontweight='bold', va='top')
    ax.text(0.05, 0.88, description, fontsize=11, va='top', wrap=True)
    
    col_widths = [0.3] * n_cols
    tbl = ax.table(cellText=data, colLabels=labels, loc='center',
                   cellLoc='center', colWidths=col_widths,
                   colColours=['#4472C4']*n_cols)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

with PdfPages(OUTPUT_PDF) as pdf:

    # ---- Title Page ----
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis('off')
    ax.text(0.5, 0.8, 'Caspian Sea Data Analysis Report', fontsize=28, ha='center', va='center', fontweight='bold')
    ax.text(0.5, 0.65, 'Based on ERA5 Data and Lake Shapefile Mask', fontsize=18, ha='center', va='center')
    ax.text(0.5, 0.52, f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}', fontsize=14, ha='center', va='center')
    ax.text(0.5, 0.42, 'Period: 1965 – 2025 (61 years)', fontsize=14, ha='center', va='center')
    ax.text(0.5, 0.32, 'Variables: PWAT, Precipitation, IVT Divergence, Net Flux', fontsize=12, ha='center', va='center')
    ax.text(0.5, 0.22, 'Analyses: Trend, Change Points, Correlation, Forecasting, Wavelet, Extreme Events, etc.', fontsize=12, ha='center', va='center')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ---- 1. Mann-Kendall ----
    add_table(pdf, df_mk,
              '1. Mann-Kendall Trend Results (Annual)',
              'Slope and significance of trends for each variable.',
              ['variable', 'slope', 'p_value', 'trend_sign'],
              ['Variable', 'Slope', 'p-value', 'Trend'])

    # ---- 2. Bootstrap ----
    add_table(pdf, df_bootstrap,
              '2. Bootstrap Trend (95% Confidence Interval)',
              'Main slope and confidence intervals from 1000 bootstrap iterations.',
              ['variable', 'main_slope', 'ci_lower', 'ci_upper'],
              ['Variable', 'Main Slope', 'Lower CI', 'Upper CI'])

    # ---- 3. Pettitt ----
    add_table(pdf, df_pettitt,
              '3. Pettitt Change Points',
              'Sudden change year in each variable time series.',
              ['variable', 'change_year', 'p_value'],
              ['Variable', 'Change Year', 'p-value'])

    # ---- 4. Quantile Regression ----
    add_table(pdf, df_quantile,
              '4. Quantile Regression (Slopes at Different Quantiles)',
              'Trend slopes at 0.1 (low), 0.5 (median), and 0.9 (high) quantiles.',
              ['variable', 'quantile', 'slope', 'p_value'],
              ['Variable', 'Quantile', 'Slope', 'p-value'])

    # ---- 5. ENSO Correlation ----
    add_table(pdf, df_enso,
              '5. Correlation with Nino3.4 (ENSO)',
              'Pearson correlation of each variable with ENSO index.',
              ['variable', 'correlation', 'p_value'],
              ['Variable', 'Correlation', 'p-value'])

    # ---- 6. Composite Analysis ----
    add_table(pdf, df_composite,
              '6. Composite Analysis (Low vs High Water Level Years)',
              'Comparison of variable means between low (1976,77,88) and high (1994,95,96) water level periods.',
              ['variable', 'low_mean', 'high_mean', 'diff', 'p_value'],
              ['Variable', 'Low Mean', 'High Mean', 'Difference', 'p-value'])

    # ---- 7. Random Forest Feature Importance ----
    add_table(pdf, df_rf,
              '7. Random Forest Feature Importance',
              'Relative importance of each variable in predicting net flux.',
              ['feature', 'importance'],
              ['Feature', 'Importance'])

    # ---- 8. Basin vs Lake Comparison ----
    add_table(pdf, df_comparison,
              '8. Basin vs Lake Comparison',
              'Comparison of means, trends, and correlations between basin and lake.',
              ['Variable', 'Basin_Mean', 'Lake_Mean', 'Basin_Trend', 'Lake_Trend', 'Correlation'],
              ['Variable', 'Basin Mean', 'Lake Mean', 'Basin Trend', 'Lake Trend', 'Correlation'])

    # ---- 9. Summary of Additional Analyses ----
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis('off')
    ax.text(0.05, 0.95, '9. Summary of Additional Analyses', fontsize=16, fontweight='bold', va='top')
    ax.text(0.05, 0.85, '''
    • STL Decomposition: Separating trend, seasonal, and residual components
    • Wavelet Analysis (Morlet): Identifying 2-4 year and 8-12 year periodicities
    • ARIMA Forecasting: Model (1,0,1) for 2026-2030
    • Extreme Events: Years above 90th percentile and below 10th percentile
    • Granger Causality: Causal relationships between variables
    • Random Forest: Feature importance for net flux prediction
    • Power Spectrum: Dominant frequencies with red noise test
    ''', fontsize=12, va='top', linespacing=1.8)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ---- 10. Images ----
    for idx, img_path in enumerate(all_images):
        try:
            fig = plt.figure(figsize=(11.69, 8.27))
            img = Image.open(img_path)
            plt.imshow(img)
            plt.axis('off')
            plt.title(os.path.basename(img_path), fontsize=10, pad=10)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        except:
            continue

print(f"\n✅ English report generated successfully.")
print(f"📂 Path: {OUTPUT_PDF}")
print(f"📸 Total images: {len(all_images)}")