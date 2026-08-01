#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================================================
یافتن تأخیر بهینه دورپیوندها با همبستگی متقاطع و مدل‌سازی
================================================================================
- محاسبه همبستگی متقاطع بین هر شاخص و تراز آب
- انتخاب تأخیر با بیشترین همبستگی (مثبت یا منفی)
- ساخت مدل رگرسیون با تأخیرهای بهینه
- محاسبه ارزش افزوده‌ی واقعی
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# ۱. تنظیمات
# ============================================================
BASE_DIR = r"K:\\gozareshha\\Dr Farjami\\Dr Farjami\\140503"
OUTPUT_DIR = os.path.join(BASE_DIR, "final_analysis", "cross_correlation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEA_LEVEL_FILE = os.path.join(
    BASE_DIR,
    "Final_Analysis_Archive_20260702_060114",
    "basin_border",
    "caspian_unified_analysis",
    "caspian_sea_level_raw.csv"
)
INDICES_FILE = os.path.join(BASE_DIR, "indices_complete.xlsx")

MAX_LAG = 24  # حداکثر تأخیر (ماه)
MIN_LAG = 1   # حداقل تأخیر (ماه)

print("="*80)
print("🌊 یافتن تأخیر بهینه دورپیوندها با همبستگی متقاطع")
print("="*80)

# ============================================================
# ۲. بارگذاری داده‌ها
# ============================================================
print("\n📂 بارگذاری داده‌های ماهانه...")

df_sea = pd.read_csv(SEA_LEVEL_FILE, sep=';', parse_dates=['datetime'])
df_sea['year'] = df_sea['datetime'].dt.year
df_sea['month'] = df_sea['datetime'].dt.month
df_sea['date'] = pd.to_datetime(df_sea[['year', 'month']].assign(day=1))
df_sea = df_sea.sort_values('date').reset_index(drop=True)
df_sea = df_sea[['date', 'year', 'month', 'wse']].copy()
df_sea.rename(columns={'wse': 'sea_level'}, inplace=True)  # ✅ اصلاح نام ستون

if os.path.exists(INDICES_FILE):
    df_indices = pd.read_excel(INDICES_FILE, sheet_name="Sheet1", parse_dates=['date'])
    df_indices = df_indices.sort_values('date').reset_index(drop=True)
    df_indices['year'] = df_indices['date'].dt.year
    df_indices['month'] = df_indices['date'].dt.month
else:
    print("⚠️ فایل indices_complete.xlsx یافت نشد.")
    sys.exit(1)

# ادغام
df_all = df_sea.merge(df_indices, on=['date', 'year', 'month'], how='inner')
df_all = df_all.dropna()
df_all = df_all.sort_values('date').reset_index(drop=True)

# دوره مشترک (۲۰۰۰–۲۰۲۵)
df_use = df_all[(df_all['year'] >= 2000) & (df_all['year'] <= 2025)].copy()
print(f"✅ داده‌های نهایی: {len(df_use)} رکورد ماهانه")

# متغیر هدف
y_original = df_use['sea_level'].values
tele_cols = [c for c in df_use.columns if c not in ['date', 'year', 'month', 'sea_level', 'wse']]
print(f"📊 تعداد شاخص‌های دورپیوند: {len(tele_cols)}")

# ============================================================
# ۳. حذف روند و فصلی (برای همبستگی خالص)
# ============================================================
print("\n🔄 حذف روند و فصلی...")

t = np.arange(len(y_original))
trend = np.polyval(np.polyfit(t, y_original, 1), t)
y_detrend = y_original - trend

# فصلی (میانگین ماهانه)
monthly_means = np.array([y_detrend[df_use['month'] == m].mean() for m in range(1, 13)])
seasonal = np.array([monthly_means[m-1] for m in df_use['month']])
y_residual = y_detrend - seasonal

print("✅ روند و فصلی حذف شدند.")

# ============================================================
# ۴. محاسبه همبستگی متقاطع برای هر شاخص
# ============================================================
print("\n📊 محاسبه همبستگی متقاطع...")

lag_results = []
lag_data = {}

for col in tele_cols:
    x_original = df_use[col].values
    
    # حذف روند و فصلی از شاخص
    trend_x = np.polyval(np.polyfit(t, x_original, 1), t)
    x_detrend = x_original - trend_x
    monthly_means_x = np.array([x_detrend[df_use['month'] == m].mean() for m in range(1, 13)])
    seasonal_x = np.array([monthly_means_x[m-1] for m in df_use['month']])
    x_residual = x_detrend - seasonal_x
    
    # همبستگی متقاطع برای تأخیرهای مختلف
    corrs = []
    for lag in range(MIN_LAG, MAX_LAG+1):
        if lag > 0:
            x_lagged = x_residual[:-lag]
            y_trimmed = y_residual[lag:]
            if len(x_lagged) > 0 and len(y_trimmed) > 0:
                corr = np.corrcoef(x_lagged, y_trimmed)[0, 1]
            else:
                corr = np.nan
        else:
            corr = np.nan
        corrs.append(corr)
    
    # بهترین تأخیر (بیشترین همبستگی مطلق)
    best_lag_idx = np.argmax(np.abs(corrs))
    best_lag = MIN_LAG + best_lag_idx
    best_corr = corrs[best_lag_idx]
    
    lag_results.append({
        'index': col,
        'best_lag': best_lag,
        'best_corr': best_corr,
        'correlations': corrs
    })
    
    if best_lag > 0:
        lag_data[col] = x_residual[:-best_lag]
    else:
        lag_data[col] = x_residual

df_lags = pd.DataFrame(lag_results)
df_lags = df_lags.sort_values('best_corr', key=abs, ascending=False)
print("\n✅ شاخص‌های با بیشترین همبستگی (با تأخیر بهینه):")
print(df_lags.head(10).to_string(index=False))

# ============================================================
# ۵. مدل‌سازی با تأخیرهای بهینه
# ============================================================
print("\n🧠 مدل‌سازی با تأخیرهای بهینه...")

# مدل پایه: روند + فصلی
X_base = np.column_stack([t, np.sin(2*np.pi*df_use['month']/12), np.cos(2*np.pi*df_use['month']/12)])
model_base = LinearRegression()
model_base.fit(X_base, y_original)
y_pred_base = model_base.predict(X_base)
r2_base = r2_score(y_original, y_pred_base)
print(f"✅ R² مدل پایه (روند + فصلی): {r2_base:.4f}")

# انتخاب ۵ شاخص برتر
top_indices = df_lags.head(5)['index'].tolist()
print(f"📌 شاخص‌های انتخاب‌شده: {top_indices}")

# ساخت ماتریس ویژگی‌ها با تأخیرهای بهینه
X_tele = []
for idx in top_indices:
    row = df_lags[df_lags['index'] == idx].iloc[0]
    best_lag = row['best_lag']
    x_original = df_use[idx].values
    if best_lag > 0:
        x_lagged = x_original[:-best_lag]
        y_trimmed = y_original[best_lag:]
        X_tele.append(x_lagged)
    else:
        X_tele.append(x_original)
        y_trimmed = y_original

# اطمینان از هم‌طول بودن
min_len = min([len(x) for x in X_tele] + [len(y_trimmed)])
X_tele = np.column_stack([x[:min_len] for x in X_tele])
y_trimmed = y_trimmed[:min_len]

# مدل با دورپیوندها
X_full = np.column_stack([X_base[:min_len], X_tele])
model_full = LinearRegression()
model_full.fit(X_full, y_trimmed)
y_pred_full = model_full.predict(X_full)
r2_full = r2_score(y_trimmed, y_pred_full)

print(f"✅ R² مدل با دورپیوندها: {r2_full:.4f}")
print(f"✅ ارزش افزوده‌ی دورپیوندها: {(r2_full - r2_base)*100:.2f}%")

# ============================================================
# ۶. نمودارها
# ============================================================
print("\n📈 تولید نمودارها...")

# ۶.۱ همبستگی متقاطع برای ۵ شاخص برتر
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
axes = axes.flatten()
for i, row in enumerate(df_lags.head(5).iterrows()):
    ax = axes[i]
    idx = row[1]['index']
    corrs = row[1]['correlations']
    lags = np.arange(MIN_LAG, MAX_LAG+1)
    ax.bar(lags, corrs, color='steelblue', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    best_lag = row[1]['best_lag']
    best_corr = row[1]['best_corr']
    ax.axvline(x=best_lag, color='red', linestyle='--', label=f'Best Lag = {best_lag}')
    ax.set_xlabel('Lag (months)')
    ax.set_ylabel('Correlation')
    ax.set_title(f'{idx} (r = {best_corr:.3f})')
    ax.grid(True, alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'cross_correlation_plots.png'), dpi=150)
plt.close()

# ۶.۲ مقایسه مدل‌ها
fig, ax = plt.subplots(figsize=(14, 6))
dates = df_use['date'].values[:min_len]
ax.plot(dates, y_trimmed, 'k-', alpha=0.5, label='Actual')
ax.plot(dates, y_pred_base[:min_len], 'b-', alpha=0.7, label='Base Model')
ax.plot(dates, y_pred_full, 'r-', alpha=0.7, label='With Teleconnections')
ax.set_xlabel('Date')
ax.set_ylabel('Sea Level (m)')
ax.set_title('Comparison: Base vs Teleconnection Model (Optimal Lags)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=150)
plt.close()

# ============================================================
# ۷. گزارش نهایی
# ============================================================
with open(os.path.join(OUTPUT_DIR, 'cross_correlation_report.txt'), 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("🌊 گزارش تأخیر بهینه دورپیوندها\n")
    f.write("="*80 + "\n\n")
    f.write(f"دوره: ۲۰۰۰–۲۰۲۵ ({len(df_use)} ماه)\n")
    f.write(f"حداکثر تأخیر بررسی‌شده: {MAX_LAG} ماه\n\n")
    
    f.write("📊 ۵ شاخص با بیشترین همبستگی:\n")
    f.write(df_lags.head(5).to_string(index=False) + "\n\n")
    
    f.write(f"📊 R² مدل پایه (روند + فصلی): {r2_base:.4f}\n")
    f.write(f"📊 R² مدل با دورپیوندها: {r2_full:.4f}\n")
    f.write(f"✅ ارزش افزوده‌ی دورپیوندها: {(r2_full - r2_base)*100:.2f}%\n\n")
    
    f.write("📌 تأخیرهای بهینه:\n")
    for idx in top_indices:
        row = df_lags[df_lags['index'] == idx].iloc[0]
        f.write(f"  {idx}: {row['best_lag']} ماه (r = {row['best_corr']:.3f})\n")
    
    f.write(f"\n✅ خروجی‌ها در: {OUTPUT_DIR}\n")

print(f"\n✅ همه خروجی‌ها در {OUTPUT_DIR} ذخیره شدند.")
print("📄 فایل‌های تولید شده:")
print("   - cross_correlation_plots.png")
print("   - model_comparison.png")
print("   - cross_correlation_report.txt")
print("="*80)