#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
تحلیل و مصورسازی کامل داده‌های شار رطوبتی (ورودی/خروجی)
برای سه بخش شمالی، میانی، جنوبی و چهار ضلع
شامل:
- سری زمانی ماهانه + خط روند (من-کندال)
- میانگین فصلی
- آزمون تی‌استودنت بین دوره‌های کم‌آبی و پرآبی
- نقشه‌های حرارتی (Heatmap) ماهانه در طول سال‌ها
- خروجی‌های عددی (جداول روند و آزمون‌ها)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, linregress
import warnings
warnings.filterwarnings("ignore")

# ============================================
# ۱. تنظیمات مسیرها
# ============================================
INPUT_BASE = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\sector_side_flux"
OUTPUT_BASE = os.path.join(INPUT_BASE, "analysis_results")
os.makedirs(OUTPUT_BASE, exist_ok=True)

SECTORS = ['South', 'Center', 'North']
SIDES = ['North', 'South', 'East', 'West']
FLOW_TYPES = ['inflow', 'outflow']
SPECIAL_PERIODS = {
    'Low_Water': list(range(1976, 1979)),   # 1976-1978
    'High_Water': list(range(1994, 1997))   # 1994-1996
}
MONTHS = list(range(1, 13))
SEASONS = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Autumn': [9, 10, 11]
}

# ============================================
# ۲. توابع کمکی
# ============================================
def mann_kendall_test(x):
    """آزمون من-کندال برای تشخیص روند (ساده و سریع)"""
    try:
        from pymannkendall import original_test
        result = original_test(x)
        return result.slope, result.p, result.trend
    except ImportError:
        # اگر pymannkendall نصب نیست، از رگرسیون خطی ساده استفاده می‌کنیم
        from scipy.stats import linregress
        n = len(x)
        if n < 3:
            return np.nan, np.nan, 'insufficient'
        res = linregress(range(n), x)
        trend = 'increasing' if res.slope > 0 else 'decreasing' if res.slope < 0 else 'no trend'
        return res.slope, res.pvalue, trend

def season_mean(df, sector, side, flow_type):
    """محاسبه میانگین فصلی برای یک سری خاص"""
    col = f'{sector}_{side}_{flow_type}'
    if col not in df.columns:
        return None
    season_means = {}
    for season, months in SEASONS.items():
        mask = df['month'].isin(months)
        season_means[season] = df.loc[mask, col].mean()
    return season_means

# ============================================
# ۳. بارگذاری داده‌ها
# ============================================
print("📂 بارگذاری داده‌های ماهانه...")
df_monthly_list = []
for sector in SECTORS:
    path = os.path.join(INPUT_BASE, f'monthly_{sector}.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['sector'] = sector
        df_monthly_list.append(df)
    else:
        print(f"⚠️ فایل {path} یافت نشد.")

df_all = pd.concat(df_monthly_list, ignore_index=True)
print(f"✅ داده‌های ماهانه بارگذاری شد: {len(df_all)} رکورد")

# بارگذاری داده‌های سالانه
annual_path = os.path.join(INPUT_BASE, 'annual_all_sectors.csv')
if os.path.exists(annual_path):
    df_annual = pd.read_csv(annual_path)
    print(f"✅ داده‌های سالانه بارگذاری شد: {len(df_annual)} رکورد")
else:
    print("⚠️ فایل سالانه یافت نشد، از داده‌های ماهانه برای محاسبه سالانه استفاده می‌شود.")
    df_annual = df_all.groupby('year').mean(numeric_only=True).reset_index()

# ============================================
# ۴. سری زمانی ماهانه با خط روند (برای هر بخش)
# ============================================
print("\n📊 رسم سری‌های زمانی ماهانه با خط روند...")
for sector in SECTORS:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Monthly Time Series - {sector} Sector', fontsize=16, fontweight='bold')
    
    df_sec = df_all[df_all['sector'] == sector]
    if df_sec.empty:
        continue
    
    for idx, side in enumerate(SIDES):
        ax = axes[idx // 2, idx % 2]
        for flow in FLOW_TYPES:
            col = f'{sector}_{side}_{flow}'
            if col not in df_sec.columns:
                continue
            # ایجاد سری زمانی
            df_sec['date'] = pd.to_datetime(df_sec['year'].astype(str) + '-' + df_sec['month'].astype(str).str.zfill(2))
            df_sec_sorted = df_sec.sort_values('date')
            ax.plot(df_sec_sorted['date'], df_sec_sorted[col], label=flow, alpha=0.6, linewidth=1)
            
            # خط روند (من-کندال)
            slope, p_val, trend = mann_kendall_test(df_sec_sorted[col].dropna().values)
            if not np.isnan(slope):
                x = np.arange(len(df_sec_sorted))
                trend_line = slope * x + np.mean(df_sec_sorted[col].dropna().values) - slope * len(df_sec_sorted)/2
                ax.plot(df_sec_sorted['date'], trend_line, '--', linewidth=2,
                        label=f'{flow} trend (slope={slope:.2e}, p={p_val:.3f})')
        
        ax.set_title(f'{side} Side')
        ax.set_xlabel('Date')
        ax.set_ylabel('Flux (kg/s)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_BASE, f'time_series_{sector}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ time_series_{sector}.png")

# ============================================
# ۵. میانگین فصلی (نمودار میلهای)
# ============================================
print("\n📊 رسم میانگین فصلی...")
for sector in SECTORS:
    df_sec = df_all[df_all['sector'] == sector]
    if df_sec.empty:
        continue
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Seasonal Means - {sector} Sector', fontsize=16, fontweight='bold')
    
    for idx, side in enumerate(SIDES):
        ax = axes[idx // 2, idx % 2]
        data = []
        for flow in FLOW_TYPES:
            col = f'{sector}_{side}_{flow}'
            if col not in df_sec.columns:
                continue
            season_vals = season_mean(df_sec, sector, side, flow)
            if season_vals:
                data.append({
                    'flow': flow,
                    **season_vals
                })
        
        if not data:
            continue
        df_plot = pd.DataFrame(data)
        df_plot_melt = df_plot.melt(id_vars=['flow'], var_name='season', value_name='mean_flux')
        
        sns.barplot(data=df_plot_melt, x='season', y='mean_flux', hue='flow', ax=ax)
        ax.set_title(f'{side} Side')
        ax.set_xlabel('Season')
        ax.set_ylabel('Mean Flux (kg/s)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_BASE, f'seasonal_means_{sector}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ seasonal_means_{sector}.png")

# ============================================
# ۶. آزمون تی‌استودنت بین دوره‌های کم‌آبی و پرآبی
# ============================================
print("\n📊 انجام آزمون تی‌استودنت بین دوره‌های کم‌آبی و پرآبی...")
ttest_results = []
for sector in SECTORS:
    df_sec = df_all[df_all['sector'] == sector]
    if df_sec.empty:
        continue
    
    for side in SIDES:
        for flow in FLOW_TYPES:
            col = f'{sector}_{side}_{flow}'
            if col not in df_sec.columns:
                continue
            
            low_vals = df_sec[df_sec['year'].isin(SPECIAL_PERIODS['Low_Water'])][col].dropna()
            high_vals = df_sec[df_sec['year'].isin(SPECIAL_PERIODS['High_Water'])][col].dropna()
            
            if len(low_vals) < 2 or len(high_vals) < 2:
                continue
            
            t_stat, p_val = ttest_ind(low_vals, high_vals, equal_var=False)
            mean_low = low_vals.mean()
            mean_high = high_vals.mean()
            diff = mean_high - mean_low
            
            ttest_results.append({
                'sector': sector,
                'side': side,
                'flow': flow,
                'mean_low': mean_low,
                'mean_high': mean_high,
                'diff': diff,
                't_stat': t_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            })

if ttest_results:
    df_ttest = pd.DataFrame(ttest_results)
    df_ttest.to_csv(os.path.join(OUTPUT_BASE, 'ttest_periods_comparison.csv'), index=False)
    print(f"   ✅ ttest_periods_comparison.csv")
    
    # رسم نمودار مقایسه‌ای
    for sector in SECTORS:
        df_sec = df_ttest[df_ttest['sector'] == sector]
        if df_sec.empty:
            continue
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Low vs High Water Periods - {sector} Sector', fontsize=16, fontweight='bold')
        
        for idx, side in enumerate(SIDES):
            ax = axes[idx // 2, idx % 2]
            df_side = df_sec[df_sec['side'] == side]
            if df_side.empty:
                continue
            
            x = np.arange(len(df_side))
            width = 0.35
            ax.bar(x - width/2, df_side['mean_low'], width, label='Low (76-78)', color='coral', alpha=0.7)
            ax.bar(x + width/2, df_side['mean_high'], width, label='High (94-96)', color='skyblue', alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(df_side['flow'])
            ax.set_ylabel('Mean Flux (kg/s)')
            ax.set_title(f'{side} Side')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # افزودن p-value به نمودار
            for i, row in df_side.iterrows():
                sig = '*' if row['p_value'] < 0.05 else 'ns'
                ax.text(i, max(row['mean_low'], row['mean_high']) * 1.05,
                        f'p={row["p_value"]:.3f} {sig}', ha='center', fontsize=8)
        
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_BASE, f'ttest_periods_{sector}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ ttest_periods_{sector}.png")

# ============================================
# ۷. نقشه‌های حرارتی (Heatmap) ماهانه
# ============================================
print("\n📊 رسم نقشه‌های حرارتی ماهانه...")
for sector in SECTORS:
    df_sec = df_all[df_all['sector'] == sector]
    if df_sec.empty:
        continue
    
    for flow in FLOW_TYPES:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{sector} Sector - {flow.capitalize()} Heatmap (Monthly Mean by Year)',
                     fontsize=16, fontweight='bold')
        
        for idx, side in enumerate(SIDES):
            ax = axes[idx // 2, idx % 2]
            col = f'{sector}_{side}_{flow}'
            if col not in df_sec.columns:
                continue
            
            # ایجاد جدول محوری (سال × ماه)
            pivot = df_sec.pivot_table(index='year', columns='month', values=col, aggfunc='mean')
            if pivot.empty:
                continue
            
            # رسم هیت‌مپ
            sns.heatmap(pivot, ax=ax, cmap='RdBu_r', center=0,
                        cbar=True, cbar_kws={'label': 'Flux (kg/s)'},
                        xticklabels=1, yticklabels=5)
            ax.set_title(f'{side} Side')
            ax.set_xlabel('Month')
            ax.set_ylabel('Year')
        
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_BASE, f'heatmap_{sector}_{flow}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ heatmap_{sector}_{flow}.png")

# ============================================
# ۸. خروجی‌های عددی (روند و آماره‌ها)
# ============================================
print("\n📊 تولید جداول عددی...")

# ۸-۱. جدول روند (من-کندال) برای سری‌های سالانه
trend_results = []
for sector in SECTORS:
    df_sec = df_annual if 'sector' in df_annual.columns else df_annual.copy()
    if 'sector' not in df_sec.columns:
        # اگر df_annual فاقد ستون sector است، از داده‌های ماهانه استفاده می‌کنیم
        df_sec = df_all[df_all['sector'] == sector].groupby('year').mean(numeric_only=True).reset_index()
    
    for side in SIDES:
        for flow in FLOW_TYPES:
            col = f'{sector}_{side}_{flow}'
            if col not in df_sec.columns:
                continue
            series = df_sec[col].dropna().values
            if len(series) < 5:
                continue
            slope, p_val, trend = mann_kendall_test(series)
            trend_results.append({
                'sector': sector,
                'side': side,
                'flow': flow,
                'slope': slope,
                'p_value': p_val,
                'trend': trend
            })

if trend_results:
    df_trend = pd.DataFrame(trend_results)
    df_trend.to_csv(os.path.join(OUTPUT_BASE, 'mann_kendall_trends.csv'), index=False)
    print(f"   ✅ mann_kendall_trends.csv")

# ۸-۲. آماره‌های پایه (میانگین، انحراف معیار، کمینه، بیشینه)
stats_results = []
for sector in SECTORS:
    df_sec = df_all[df_all['sector'] == sector]
    if df_sec.empty:
        continue
    for side in SIDES:
        for flow in FLOW_TYPES:
            col = f'{sector}_{side}_{flow}'
            if col not in df_sec.columns:
                continue
            series = df_sec[col].dropna()
            if len(series) == 0:
                continue
            stats_results.append({
                'sector': sector,
                'side': side,
                'flow': flow,
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'count': len(series)
            })

if stats_results:
    df_stats = pd.DataFrame(stats_results)
    df_stats.to_csv(os.path.join(OUTPUT_BASE, 'basic_statistics.csv'), index=False)
    print(f"   ✅ basic_statistics.csv")

# ============================================
# ۹. جمع‌بندی نهایی
# ============================================
print("\n" + "="*60)
print("✅ تمام تحلیل‌ها و مصورسازی‌ها با موفقیت انجام شد.")
print(f"📂 خروجی‌ها در: {OUTPUT_BASE}")
print("\n📁 فایل‌های تولید شده:")
print("   📊 تصاویر:")
print("      - time_series_*.png  (سری زمانی ماهانه با خط روند)")
print("      - seasonal_means_*.png  (میانگین فصلی)")
print("      - ttest_periods_*.png  (مقایسه دوره‌های کم‌آبی و پرآبی)")
print("      - heatmap_*.png  (نقشه‌های حرارتی ماهانه)")
print("   📄 جداول عددی:")
print("      - ttest_periods_comparison.csv  (نتایج آزمون تی)")
print("      - mann_kendall_trends.csv  (روند من-کندال)")
print("      - basic_statistics.csv  (آماره‌های پایه)")
print("="*60)