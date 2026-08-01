#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
تحلیل اختصاصی سال‌های کمینه و بیشینه سطح آب خزر (اصلاح‌شده)
- Basin (حوضه آبریز) و Lake (خود دریاچه)
- دوره کمینه: 1976, 1977, 1978 (تصحیح شده)
- دوره بیشینه: 1994, 1995, 1996
- خروجی: جداول عددی، نمودارهای مقایسه‌ای، سری‌های زمانی ماهانه
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu
import warnings
warnings.filterwarnings("ignore")

# ============================================
# تنظیمات مسیرها و پارامترها
# ============================================
BASE_DIR = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503"
OUTPUT_DIR = os.path.join(BASE_DIR, "low_high_years_corrected")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# سال‌های مورد نظر (اصلاح‌شده)
LOW_YEARS = [1976, 1977, 1978]   # ← سال 1988 به 1978 تغییر کرد
HIGH_YEARS = [1994, 1995, 1996]

# مسیر فایل‌های داده
BASIN_CSV = os.path.join(BASE_DIR, "basin_border", "stats", "border_flux_summary.csv")
LAKE_CSV = os.path.join(BASE_DIR, "caspian_lake_analysis", "caspian_lake_flux_summary.csv")

# متغیرهای مورد بررسی
BASIN_VARS = ['pwat_mm', 'precip_mm', 'net_flux_kg_s', 'div_ivt_kg_m2_s']
LAKE_VARS = ['pwat_mm', 'precip_mm', 'border_net_flux', 'div_ivt']

# نام‌های فارسی برای گزارش
VAR_NAMES_FA = {
    'pwat_mm': 'آب قابل بارش (PWAT)',
    'precip_mm': 'بارش',
    'net_flux_kg_s': 'شار خالص مرزی',
    'border_net_flux': 'شار خالص مرزی',
    'div_ivt_kg_m2_s': 'واگرایی IVT',
    'div_ivt': 'واگرایی IVT'
}

print("="*70)
print("تحلیل اختصاصی سال‌های کمینه و بیشینه سطح آب خزر (اصلاح‌شده)")
print("="*70)
print(f"دوره کمینه: {LOW_YEARS}")
print(f"دوره بیشینه: {HIGH_YEARS}")
print("="*70)

# ============================================
# ۱. بارگذاری داده‌ها
# ============================================
def load_data(csv_path, region_name):
    if not os.path.exists(csv_path):
        print(f"⚠️ فایل {csv_path} یافت نشد!")
        return None
    df = pd.read_csv(csv_path)
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    print(f"✅ {region_name}: {df['year'].min()}-{df['year'].max()}, {len(df)} رکورد")
    return df

df_basin = load_data(BASIN_CSV, "حوضه آبریز (Basin)")
df_lake = load_data(LAKE_CSV, "دریای خزر (Lake)")

if df_basin is None or df_lake is None:
    raise FileNotFoundError("فایل‌های داده یافت نشدند!")

# ============================================
# ۲. فیلتر کردن سال‌های مورد نظر
# ============================================
def filter_years(df, years, vars_list):
    df_filtered = df[df['year'].isin(years)].copy()
    df_annual = df_filtered.groupby('year')[vars_list].mean().reset_index()
    df_period_mean = df_filtered[vars_list].mean()
    df_period_std = df_filtered[vars_list].std()
    return df_filtered, df_annual, df_period_mean, df_period_std

# Basin
basin_low_df, basin_low_annual, basin_low_mean, basin_low_std = filter_years(df_basin, LOW_YEARS, BASIN_VARS)
basin_high_df, basin_high_annual, basin_high_mean, basin_high_std = filter_years(df_basin, HIGH_YEARS, BASIN_VARS)

# Lake
lake_low_df, lake_low_annual, lake_low_mean, lake_low_std = filter_years(df_lake, LOW_YEARS, LAKE_VARS)
lake_high_df, lake_high_annual, lake_high_mean, lake_high_std = filter_years(df_lake, HIGH_YEARS, LAKE_VARS)

# ============================================
# ۳. محاسبه آمارهای توصیفی و آزمون‌های آماری
# ============================================
def compute_stats(region_name, vars_list, low_df, high_df):
    results = []
    for var in vars_list:
        low_vals = low_df[var].dropna()
        high_vals = high_df[var].dropna()
        
        # آمار توصیفی
        low_mean = low_vals.mean()
        low_std = low_vals.std()
        low_min = low_vals.min()
        low_max = low_vals.max()
        high_mean = high_vals.mean()
        high_std = high_vals.std()
        high_min = high_vals.min()
        high_max = high_vals.max()
        
        # آزمون t-test
        t_stat, t_p = ttest_ind(low_vals, high_vals, equal_var=False)
        # آزمون من-ویتنی
        u_stat, mw_p = mannwhitneyu(low_vals, high_vals, alternative='two-sided')
        
        results.append({
            'region': region_name,
            'variable': var,
            'low_mean': low_mean,
            'low_std': low_std,
            'low_min': low_min,
            'low_max': low_max,
            'high_mean': high_mean,
            'high_std': high_std,
            'high_min': high_min,
            'high_max': high_max,
            'diff_mean': high_mean - low_mean,
            'diff_percent': ((high_mean - low_mean) / low_mean) * 100 if low_mean != 0 else np.nan,
            't_pvalue': t_p,
            'mw_pvalue': mw_p
        })
    return pd.DataFrame(results)

basin_stats = compute_stats('Basin', BASIN_VARS, basin_low_df, basin_high_df)
lake_stats = compute_stats('Lake', LAKE_VARS, lake_low_df, lake_high_df)

# ============================================
# ۴. تولید جداول خلاصه
# ============================================
def format_table(df_stats, region_name):
    df = df_stats.copy()
    df['variable'] = df['variable'].map(VAR_NAMES_FA).fillna(df['variable'])
    df['low_mean'] = df['low_mean'].round(3)
    df['high_mean'] = df['high_mean'].round(3)
    df['diff_mean'] = df['diff_mean'].round(3)
    df['diff_percent'] = df['diff_percent'].round(1)
    df['t_pvalue'] = df['t_pvalue'].round(4)
    df['mw_pvalue'] = df['mw_pvalue'].round(4)
    df['significant'] = df['t_pvalue'] < 0.05
    return df

basin_stats_formatted = format_table(basin_stats, 'Basin')
lake_stats_formatted = format_table(lake_stats, 'Lake')

# ذخیره جداول
basin_stats_formatted.to_csv(os.path.join(OUTPUT_DIR, 'basin_stats_low_high_corrected.csv'), index=False)
lake_stats_formatted.to_csv(os.path.join(OUTPUT_DIR, 'lake_stats_low_high_corrected.csv'), index=False)

print("\n📊 جداول آماری ذخیره شدند.")

# ============================================
# ۵. نمودارهای مقایسه‌ای
# ============================================
sns.set_style('whitegrid')
sns.set_palette('Set2')

def plot_comparison(region_name, stats_df, low_df, high_df, vars_list, output_dir):
    # ۵-۱. نمودار ستونی مقایسه میانگین‌ها
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, var in enumerate(vars_list):
        if idx >= 4:
            break
        ax = axes[idx]
        var_fa = VAR_NAMES_FA.get(var, var)
        low_mean = low_df[var].mean()
        high_mean = high_df[var].mean()
        low_std = low_df[var].std()
        high_std = high_df[var].std()
        ax.bar(['کمینه\n(76,77,78)', 'بیشینه\n(94,95,96)'], 
               [low_mean, high_mean], 
               yerr=[low_std, high_std], 
               capsize=8, color=['coral', 'skyblue'], edgecolor='black')
        ax.set_ylabel(var_fa)
        ax.set_title(f'{region_name} - {var_fa}')
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate([low_mean, high_mean]):
            ax.text(i, v + 0.05*abs(v), f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{region_name}_bar_comparison_corrected.png'), dpi=150)
    plt.close()

    # ۵-۲. باکس‌پلات ماهانه
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, var in enumerate(vars_list):
        if idx >= 4:
            break
        ax = axes[idx]
        var_fa = VAR_NAMES_FA.get(var, var)
        low_vals = low_df[var].dropna()
        high_vals = high_df[var].dropna()
        data = pd.DataFrame({
            'دوره': ['کمینه']*len(low_vals) + ['بیشینه']*len(high_vals),
            'مقدار': pd.concat([low_vals, high_vals], ignore_index=True)
        })
        sns.boxplot(data=data, x='دوره', y='مقدار', ax=ax, palette=['coral', 'skyblue'])
        ax.set_ylabel(var_fa)
        ax.set_title(f'{region_name} - {var_fa}')
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{region_name}_boxplot_comparison_corrected.png'), dpi=150)
    plt.close()

    # ۵-۳. سری زمانی ماهانه برای سال‌های خاص
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, var in enumerate(vars_list):
        if idx >= 4:
            break
        ax = axes[idx]
        var_fa = VAR_NAMES_FA.get(var, var)
        for year in LOW_YEARS:
            data_year = low_df[low_df['year'] == year]
            ax.plot(data_year['month'], data_year[var], 'o-', label=str(year), color='red', alpha=0.7)
        for year in HIGH_YEARS:
            data_year = high_df[high_df['year'] == year]
            ax.plot(data_year['month'], data_year[var], 's-', label=str(year), color='blue', alpha=0.7)
        ax.set_xlabel('ماه')
        ax.set_ylabel(var_fa)
        ax.set_title(f'{region_name} - {var_fa}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{region_name}_monthly_timeseries_corrected.png'), dpi=150)
    plt.close()

# اجرای نمودارها برای Basin و Lake
plot_comparison('Basin', basin_stats, basin_low_df, basin_high_df, BASIN_VARS, OUTPUT_DIR)
plot_comparison('Lake', lake_stats, lake_low_df, lake_high_df, LAKE_VARS, OUTPUT_DIR)

print("📈 نمودارهای مقایسه‌ای ذخیره شدند.")

# ============================================
# ۶. نمودار مقایسه Basin و Lake با هم
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
common_vars = ['pwat_mm', 'precip_mm']
for idx, var in enumerate(common_vars):
    if idx >= 2:
        break
    ax = axes[idx]
    var_fa = VAR_NAMES_FA.get(var, var)
    basin_low_mean = basin_low_df[var].mean()
    basin_high_mean = basin_high_df[var].mean()
    basin_low_std = basin_low_df[var].std()
    basin_high_std = basin_high_df[var].std()
    lake_low_mean = lake_low_df[var].mean()
    lake_high_mean = lake_high_df[var].mean()
    lake_low_std = lake_low_df[var].std()
    lake_high_std = lake_high_df[var].std()
    
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width/2, [basin_low_mean, basin_high_mean], width, 
           yerr=[basin_low_std, basin_high_std], capsize=5, label='Basin', color=['coral', 'skyblue'])
    ax.bar(x + width/2, [lake_low_mean, lake_high_mean], width,
           yerr=[lake_low_std, lake_high_std], capsize=5, label='Lake', color=['darkred', 'darkblue'])
    ax.set_xticks(x)
    ax.set_xticklabels(['کمینه', 'بیشینه'])
    ax.set_ylabel(var_fa)
    ax.set_title(f'مقایسه Basin و Lake - {var_fa}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'basin_lake_comparison_bar_corrected.png'), dpi=150)
plt.close()

print("📈 نمودار مقایسه Basin و Lake ذخیره شد.")

# ============================================
# ۷. تولید گزارش متنی
# ============================================
def generate_report():
    lines = []
    lines.append("="*70)
    lines.append("گزارش تحلیل اختصاصی سال‌های کمینه و بیشینه سطح آب خزر (اصلاح‌شده)")
    lines.append("="*70)
    lines.append("")
    lines.append(f"دوره کمینه (تصحیح‌شده): {', '.join(map(str, LOW_YEARS))}")
    lines.append(f"دوره بیشینه: {', '.join(map(str, HIGH_YEARS))}")
    lines.append("")
    
    lines.append("## ۱. حوضه آبریز (Basin)")
    lines.append("")
    lines.append("### جدول مقایسه آماری")
    lines.append(basin_stats_formatted.to_string(index=False))
    lines.append("")
    
    lines.append("## ۲. خود دریای خزر (Lake)")
    lines.append("")
    lines.append("### جدول مقایسه آماری")
    lines.append(lake_stats_formatted.to_string(index=False))
    lines.append("")
    
    lines.append("## ۳. تفسیر نتایج")
    lines.append("")
    for idx, row in basin_stats.iterrows():
        var = VAR_NAMES_FA.get(row['variable'], row['variable'])
        diff = row['diff_mean']
        p = row['t_pvalue']
        sig = "معنی‌دار" if p < 0.05 else "غیرمعنی‌دار"
        direction = "بیشتر" if diff > 0 else "کمتر"
        lines.append(f"   • {var}: در دوره بیشینه {diff:.3f} واحد {direction} از دوره کمینه است ({sig}، p={p:.4f})")
    lines.append("")
    lines.append("   نکته: مقادیر p-value کمتر از ۰٫۰۵ نشان‌دهنده تفاوت معنی‌دار بین دو دوره است.")
    lines.append("")
    lines.append("="*70)
    return "\n".join(lines)

report_text = generate_report()
with open(os.path.join(OUTPUT_DIR, 'analysis_report_corrected.txt'), 'w', encoding='utf-8') as f:
    f.write(report_text)

print("\n📄 گزارش متنی ذخیره شد.")

# ============================================
# ۸. جمع‌بندی نهایی
# ============================================
print("\n" + "="*70)
print("✅ تحلیل اختصاصی سال‌های کمینه و بیشینه با سال‌های اصلاح‌شده انجام شد.")
print(f"📂 خروجی‌ها در: {OUTPUT_DIR}")
print("")
print("📊 فایل‌های تولید شده:")
print("   - basin_stats_low_high_corrected.csv (آمار حوضه آبریز)")
print("   - lake_stats_low_high_corrected.csv (آمار خود دریاچه)")
print("   - Basin_bar_comparison_corrected.png (نمودار ستونی حوضه)")
print("   - Basin_boxplot_comparison_corrected.png (باکس‌پلات حوضه)")
print("   - Basin_monthly_timeseries_corrected.png (سری زمانی ماهانه حوضه)")
print("   - Lake_bar_comparison_corrected.png (نمودار ستونی دریاچه)")
print("   - Lake_boxplot_comparison_corrected.png (باکس‌پلات دریاچه)")
print("   - Lake_monthly_timeseries_corrected.png (سری زمانی ماهانه دریاچه)")
print("   - basin_lake_comparison_bar_corrected.png (مقایسه حوضه و دریاچه)")
print("   - analysis_report_corrected.txt (گزارش متنی)")
print("="*70)