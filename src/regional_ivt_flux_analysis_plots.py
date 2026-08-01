#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate English charts from regional IVT flux CSV files.
Inputs: CSV files for North, Center, South (monthly and annual) and comparison table.
Outputs: PNG charts with English labels.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# Settings
# ============================================
BASE_DIR = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\regional_ivt_analysis"
OUTPUT_DIR = os.path.join(BASE_DIR, "english_charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Years for low and high water periods
LOW_YEARS = [1976, 1977, 1978]
HIGH_YEARS = [1994, 1995, 1996]
REGIONS = ['North', 'Center', 'South']

# Column names
INFLOW = 'inflow_kg_s'
OUTFLOW = 'outflow_kg_s'
NET = 'net_flux_kg_s'

# Load data
monthly_dfs = {}
annual_dfs = {}
for region in REGIONS:
    monthly_file = os.path.join(BASE_DIR, f"{region}_ivt_flux_monthly.csv")
    annual_file = os.path.join(BASE_DIR, f"{region}_ivt_flux_annual.csv")
    if os.path.exists(monthly_file):
        monthly_dfs[region] = pd.read_csv(monthly_file)
    if os.path.exists(annual_file):
        annual_dfs[region] = pd.read_csv(annual_file)

# Load comparison table
comp_file = os.path.join(BASE_DIR, "comparison_table.csv")
df_comp = pd.read_csv(comp_file) if os.path.exists(comp_file) else None

print(f"Loaded monthly data for: {list(monthly_dfs.keys())}")
print(f"Loaded annual data for: {list(annual_dfs.keys())}")
print(f"Loaded comparison table: {df_comp is not None}")

# ============================================
# Plotting functions
# ============================================
def plot_annual_timeseries(region, df_annual, low_years, high_years, output_dir):
    """Plot annual inflow, outflow, net flux for one region."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_annual['year'], df_annual[INFLOW], 'g-', label='Inflow', linewidth=2)
    ax.plot(df_annual['year'], df_annual[OUTFLOW], 'r-', label='Outflow', linewidth=2)
    ax.plot(df_annual['year'], df_annual[NET], 'b-', label='Net Flux', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    for yr in low_years:
        ax.axvline(x=yr, color='orange', linestyle=':', alpha=0.6, linewidth=2)
    for yr in high_years:
        ax.axvline(x=yr, color='green', linestyle=':', alpha=0.6, linewidth=2)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('IVT Flux (kg/s)', fontsize=12)
    ax.set_title(f'{region} - Annual IVT Flux', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{region}_annual_timeseries_en.png'), dpi=150)
    plt.close()
    print(f"   ✅ {region} annual timeseries saved.")

def plot_combined_net_flux(annual_dfs, low_years, high_years, output_dir):
    """Plot annual net flux for all regions on one figure."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {'North': 'blue', 'Center': 'green', 'South': 'red'}
    for region in REGIONS:
        if region in annual_dfs:
            df = annual_dfs[region]
            ax.plot(df['year'], df[NET], '-', color=colors[region], label=f'{region} Net Flux', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    for yr in low_years:
        ax.axvline(x=yr, color='orange', linestyle=':', alpha=0.6, linewidth=2)
    for yr in high_years:
        ax.axvline(x=yr, color='green', linestyle=':', alpha=0.6, linewidth=2)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Net IVT Flux (kg/s)', fontsize=12)
    ax.set_title('Regional Net IVT Flux Comparison', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_net_flux_en.png'), dpi=150)
    plt.close()
    print("   ✅ Combined net flux plot saved.")

def plot_monthly_climatology(region, df_monthly, output_dir):
    """Plot mean monthly inflow, outflow, net flux."""
    df_mean = df_monthly.groupby('month_num')[ [INFLOW, OUTFLOW, NET] ].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_mean.index, df_mean[INFLOW], 'g-', label='Inflow', marker='o')
    ax.plot(df_mean.index, df_mean[OUTFLOW], 'r-', label='Outflow', marker='s')
    ax.plot(df_mean.index, df_mean[NET], 'b-', label='Net Flux', marker='d')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Mean IVT Flux (kg/s)', fontsize=12)
    ax.set_title(f'{region} - Monthly Climatology', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{region}_monthly_climatology_en.png'), dpi=150)
    plt.close()
    print(f"   ✅ {region} monthly climatology saved.")

def plot_comparison_bar(df_comp, output_dir):
    """Plot bar chart comparing mean values across regions for low and high periods."""
    if df_comp is None:
        return
    # Prepare data: we want for each region and variable (inflow, outflow, net) the low and high means.
    # We'll create two separate bar charts: one for inflow, one for outflow, one for net.
    variables = ['ورودی', 'خروجی', 'خالص']  # Persian column names; we'll map to English
    var_map = {'ورودی': 'Inflow', 'خروجی': 'Outflow', 'خالص': 'Net Flux'}
    for var_persian, var_en in var_map.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        df_sub = df_comp[df_comp['متغیر'] == var_persian]
        if df_sub.empty:
            continue
        regions = df_sub['بخش'].values
        low_means = df_sub['میانگین کمینه'].values
        high_means = df_sub['میانگین بیشینه'].values
        x = np.arange(len(regions))
        width = 0.35
        ax.bar(x - width/2, low_means, width, label='Low Period', color='coral')
        ax.bar(x + width/2, high_means, width, label='High Period', color='skyblue')
        ax.set_xlabel('Region', fontsize=12)
        ax.set_ylabel(f'{var_en} (kg/s)', fontsize=12)
        ax.set_title(f'Comparison of {var_en} between Low and High Periods', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(regions)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_{var_en.lower().replace(" ", "_")}_en.png'), dpi=150)
        plt.close()
        print(f"   ✅ Comparison bar chart for {var_en} saved.")

def plot_comparison_table_image(df_comp, output_dir):
    """Generate a table image from the comparison data."""
    if df_comp is None:
        return
    # Clean data for display
    df_display = df_comp.pivot(index='بخش', columns='متغیر', values=['میانگین کمینه', 'میانگین بیشینه', 'تفاوت', 'درصد تغییر', 'p-value (t-test)'])
    # Flatten multi-index columns
    df_display.columns = ['_'.join(col).strip() for col in df_display.columns.values]
    df_display = df_display.round(2)
    # Create figure and table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    table = ax.table(cellText=df_display.values,
                     rowLabels=df_display.index,
                     colLabels=df_display.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    ax.set_title('Comparison of IVT Flux between Low and High Periods', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_table_en.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Comparison table image saved.")

# ============================================
# Run all plots
# ============================================
print("\n🔄 Generating English charts...")

# 1. Annual timeseries for each region
for region in REGIONS:
    if region in annual_dfs:
        plot_annual_timeseries(region, annual_dfs[region], LOW_YEARS, HIGH_YEARS, OUTPUT_DIR)

# 2. Combined net flux
plot_combined_net_flux(annual_dfs, LOW_YEARS, HIGH_YEARS, OUTPUT_DIR)

# 3. Monthly climatology for each region
for region in REGIONS:
    if region in monthly_dfs:
        plot_monthly_climatology(region, monthly_dfs[region], OUTPUT_DIR)

# 4. Comparison bar charts
if df_comp is not None:
    plot_comparison_bar(df_comp, OUTPUT_DIR)
    plot_comparison_table_image(df_comp, OUTPUT_DIR)

print(f"\n✅ All English charts saved in: {OUTPUT_DIR}")