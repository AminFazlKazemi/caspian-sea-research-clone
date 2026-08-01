#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate English outputs for regional IVT flux analysis
Reads existing CSVs, renames columns, creates English plots and summary table
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu
import warnings
warnings.filterwarnings("ignore")

# ============================================
# Settings
# ============================================
BASE_DIR = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503"
INPUT_DIR = os.path.join(BASE_DIR, "regional_ivt_analysis")
OUTPUT_DIR = os.path.join(INPUT_DIR, "english_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOW_YEARS = [1976, 1977, 1978]
HIGH_YEARS = [1994, 1995, 1996]
REGIONS = ['North', 'Center', 'South']

# Column name mapping (Persian -> English)
COL_MAP = {
    'month': 'month',
    'year': 'year',
    'month_num': 'month_num',
    'inflow_kg_s': 'inflow_kg_s',
    'outflow_kg_s': 'outflow_kg_s',
    'net_flux_kg_s': 'net_flux_kg_s'
}

VAR_NAMES_EN = {
    'inflow_kg_s': 'Inflow',
    'outflow_kg_s': 'Outflow',
    'net_flux_kg_s': 'Net Flux'
}

# ============================================
# 1. Load existing data
# ============================================
print("Loading existing CSV files...")
dfs = {}
for region in REGIONS:
    filepath = os.path.join(INPUT_DIR, f"{region}_ivt_flux_monthly.csv")
    if not os.path.exists(filepath):
        print(f"   ⚠️ {filepath} not found. Skipping.")
        continue
    df = pd.read_csv(filepath)
    dfs[region] = df
    print(f"   ✅ {region}: {len(df)} records")

if not dfs:
    raise FileNotFoundError("No CSV files found. Please run the main analysis first.")

# ============================================
# 2. Save English versions of CSVs (with English column names if needed)
# ============================================
print("\nSaving English CSV files...")
for region, df in dfs.items():
    # Columns are already in English (inflow_kg_s, etc.) but we keep as is
    df_en = df.copy()
    # Add a column for region if needed
    df_en['region'] = region
    # Save with English name
    out_file = os.path.join(OUTPUT_DIR, f"{region}_ivt_flux_monthly_en.csv")
    df_en.to_csv(out_file, index=False)
    print(f"   ✅ {region} -> {out_file}")

# ============================================
# 3. Annual summary and comparison table (English)
# ============================================
print("\nCreating English comparison table...")
comparison_rows = []
for region, df in dfs.items():
    low_df = df[df['year'].isin(LOW_YEARS)]
    high_df = df[df['year'].isin(HIGH_YEARS)]
    for var in ['inflow_kg_s', 'outflow_kg_s', 'net_flux_kg_s']:
        low_vals = low_df[var].dropna()
        high_vals = high_df[var].dropna()
        if len(low_vals) < 2 or len(high_vals) < 2:
            continue
        t_stat, t_p = ttest_ind(low_vals, high_vals, equal_var=False)
        mw_stat, mw_p = mannwhitneyu(low_vals, high_vals, alternative='two-sided')
        comparison_rows.append({
            'Region': region,
            'Variable': VAR_NAMES_EN[var],
            'Low_Mean': low_vals.mean(),
            'High_Mean': high_vals.mean(),
            'Difference': high_vals.mean() - low_vals.mean(),
            'Change_%': ((high_vals.mean() - low_vals.mean()) / low_vals.mean()) * 100 if low_vals.mean() != 0 else np.nan,
            't_pvalue': t_p,
            'MW_pvalue': mw_p
        })

df_comp = pd.DataFrame(comparison_rows)
comp_file = os.path.join(OUTPUT_DIR, "comparison_table_en.csv")
df_comp.to_csv(comp_file, index=False)
print(f"   ✅ Comparison table saved: {comp_file}")

# ============================================
# 4. Plot annual time series (English labels)
# ============================================
print("\nPlotting annual time series (English)...")
sns.set_style('whitegrid')
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
for idx, region in enumerate(REGIONS):
    if region not in dfs:
        continue
    ax = axes[idx]
    df = dfs[region]
    df_annual = df.groupby('year')[['inflow_kg_s', 'outflow_kg_s', 'net_flux_kg_s']].mean().reset_index()
    ax.plot(df_annual['year'], df_annual['inflow_kg_s'], 'g-', label='Inflow')
    ax.plot(df_annual['year'], df_annual['outflow_kg_s'], 'r-', label='Outflow')
    ax.plot(df_annual['year'], df_annual['net_flux_kg_s'], 'b-', label='Net Flux', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    for yr in LOW_YEARS:
        ax.axvline(x=yr, color='orange', linestyle=':', alpha=0.6, linewidth=2, label='Low' if yr == LOW_YEARS[0] else "")
    for yr in HIGH_YEARS:
        ax.axvline(x=yr, color='green', linestyle=':', alpha=0.6, linewidth=2, label='High' if yr == HIGH_YEARS[0] else "")
    ax.set_title(f'{region} - IVT Flux')
    ax.set_ylabel('kg/s')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if idx == 2:
        ax.set_xlabel('Year')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'annual_timeseries_all_regions_en.png'), dpi=150)
plt.close()

# ============================================
# 5. Bar plot comparing regions (English)
# ============================================
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(3)
width = 0.25
for i, var in enumerate(['inflow_kg_s', 'outflow_kg_s', 'net_flux_kg_s']):
    means = [dfs[region][var].mean() for region in REGIONS if region in dfs]
    ax.bar(x + i*width, means, width, label=VAR_NAMES_EN[var])
ax.set_xticks(x + width)
ax.set_xticklabels(REGIONS)
ax.set_ylabel('kg/s')
ax.set_title('Annual average IVT flux by region')
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'regional_comparison_bar_en.png'), dpi=150)
plt.close()

# ============================================
# 6. Monthly climatology (English)
# ============================================
for region in REGIONS:
    if region not in dfs:
        continue
    df = dfs[region]
    monthly_mean = df.groupby('month_num')[['inflow_kg_s', 'outflow_kg_s', 'net_flux_kg_s']].mean()
    plt.figure(figsize=(10, 6))
    monthly_mean.plot(kind='bar', ax=plt.gca())
    plt.title(f'{region} - Monthly mean IVT flux')
    plt.xlabel('Month')
    plt.ylabel('kg/s')
    plt.legend(['Inflow', 'Outflow', 'Net Flux'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{region}_monthly_climatology_en.png'), dpi=150)
    plt.close()

# ============================================
# 7. Summary text report (English)
# ============================================
report_lines = []
report_lines.append("="*70)
report_lines.append("Regional IVT Flux Analysis - Summary (English)")
report_lines.append("="*70)
report_lines.append("")
report_lines.append(f"Period: {min(dfs[region]['year'].min() for region in REGIONS if region in dfs)} - {max(dfs[region]['year'].max() for region in REGIONS if region in dfs)}")
report_lines.append(f"Low water years: {LOW_YEARS}")
report_lines.append(f"High water years: {HIGH_YEARS}")
report_lines.append("")
report_lines.append("Comparison Table (Low vs High periods):")
report_lines.append(df_comp.round(3).to_string(index=False))
report_lines.append("")
report_lines.append("Notes:")
report_lines.append("  - Positive Net Flux means outflow > inflow (moisture leaving the region)")
report_lines.append("  - Negative Net Flux means inflow > outflow (moisture entering the region)")
report_lines.append("  - p-value < 0.05 indicates significant difference between low and high periods")
report_lines.append("")
report_lines.append("="*70)

report_file = os.path.join(OUTPUT_DIR, "analysis_summary_en.txt")
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("\n".join(report_lines))
print(f"   ✅ Summary report saved: {report_file}")

print(f"\n✅ All English outputs have been saved to: {OUTPUT_DIR}")
print("Files generated:")
print("  - *_ivt_flux_monthly_en.csv (monthly data for each region)")
print("  - comparison_table_en.csv (statistical comparison)")
print("  - annual_timeseries_all_regions_en.png (time series plots)")
print("  - regional_comparison_bar_en.png (bar chart)")
print("  - *_monthly_climatology_en.png (monthly averages)")
print("  - analysis_summary_en.txt (text summary)")