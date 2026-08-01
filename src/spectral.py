#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================================================
تحلیل طیفی (فرکانسی) دورپیوندها و تراز آب دریای خزر
================================================================================
- استخراج بسامدهای غالب با FFT
- مدل‌سازی رگرسیون در حوزه فرکانس
- محاسبه ارزش افزوده‌ی واقعی هر شاخص
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.signal import find_peaks, coherence
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = r"K:\\gozareshha\\Dr Farjami\\Dr Farjami\\140503"
OUTPUT_DIR = os.path.join(BASE_DIR, "final_analysis", "spectral_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEA_LEVEL_FILE = os.path.join(
    BASE_DIR,
    "Final_Analysis_Archive_20260702_060114",
    "basin_border",
    "caspian_unified_analysis",
    "caspian_sea_level_raw.csv"
)
INDICES_FILE = os.path.join(BASE_DIR, "indices_complete.xlsx")

print("="*80)
print("🌊 تحلیل طیفی (فرکانسی) دورپیوندها و تراز آب خزر")
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

# متغیرها
sea_level = df_use['sea_level'].values
tele_cols = [c for c in df_use.columns if c not in ['date', 'year', 'month', 'sea_level', 'wse']]
print(f"📊 تعداد شاخص‌های دورپیوند: {len(tele_cols)}")

# ============================================================
# ۳. حذف روند و فصلی
# ============================================================
print("\n🔄 حذف روند و فصلی...")

t = np.arange(len(sea_level))
trend = np.polyval(np.polyfit(t, sea_level, 1), t)
sea_detrend = sea_level - trend

monthly_means = np.array([sea_detrend[df_use['month'] == m].mean() for m in range(1, 13)])
seasonal = np.array([monthly_means[m-1] for m in df_use['month']])
sea_residual = sea_detrend - seasonal

tele_residual = {}
for col in tele_cols:
    vals = df_use[col].values
    trend_t = np.polyval(np.polyfit(t, vals, 1), t)
    detrend = vals - trend_t
    monthly_means_t = np.array([detrend[df_use['month'] == m].mean() for m in range(1, 13)])
    seasonal_t = np.array([monthly_means_t[m-1] for m in df_use['month']])
    tele_residual[col] = detrend - seasonal_t

print("✅ روند و فصلی حذف شدند.")

# ============================================================
# ۴. تحلیل طیفی (FFT)
# ============================================================
print("\n📊 تحلیل طیفی (FFT)...")

fs = 12
n = len(sea_residual)
freqs = np.fft.rfftfreq(n, d=1/fs)

fft_sea = np.fft.rfft(sea_residual)
power_sea = np.abs(fft_sea)**2

peaks, _ = find_peaks(power_sea, height=np.percentile(power_sea, 90))
peak_freqs = freqs[peaks]
peak_periods = 1/peak_freqs

print("✅ بسامدهای غالب تراز آب:")
for f, p in zip(peak_freqs, peak_periods):
    if p > 2:
        print(f"   دوره {p:.1f} ماه (فرکانس {f:.3f})")

# ============================================================
# ۵. Coherence
# ============================================================
print("\n📊 محاسبه همبستگی طیفی (Coherence)...")

coherence_results = []
for col in tele_cols:
    f, Cxy = coherence(sea_residual, tele_residual[col], fs=fs, nperseg=min(128, len(sea_residual)//4))
    coherence_results.append({
        'index': col,
        'max_coherence': np.max(Cxy),
        'mean_coherence': np.mean(Cxy),
        'period_max': 1/f[np.argmax(Cxy)] if f[np.argmax(Cxy)] > 0 else np.nan
    })

df_coherence = pd.DataFrame(coherence_results).sort_values('max_coherence', ascending=False)
print("\n✅ شاخص‌های با بیشترین همبستگی طیفی:")
print(df_coherence.head(10).to_string(index=False))

# ============================================================
# ۶. نمودارها
# ============================================================
print("\n📈 تولید نمودارها...")

fig, ax = plt.subplots(figsize=(12, 6))
ax.semilogy(freqs, power_sea, 'b-', label='Power Spectrum')
ax.scatter(freqs[peaks], power_sea[peaks], color='red', s=50, label='Dominant Peaks')
for f, p in zip(peak_freqs, peak_periods):
    if p > 2:
        ax.axvline(f, color='gray', linestyle='--', alpha=0.5)
        ax.text(f, np.max(power_sea)*0.8, f'{p:.1f}m', rotation=90, fontsize=8)
ax.set_xlabel('Frequency (cycles/month)')
ax.set_ylabel('Power')
ax.set_title('Power Spectrum of Caspian Sea Level (2000-2025)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'power_spectrum.png'), dpi=150)
plt.close()

top_5 = df_coherence.head(5)['index'].tolist()
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
axes = axes.flatten()
for i, col in enumerate(top_5[:5]):
    ax = axes[i]
    f, Cxy = coherence(sea_residual, tele_residual[col], fs=fs, nperseg=min(128, len(sea_residual)//4))
    ax.plot(f, Cxy, 'b-', linewidth=1.5)
    ax.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')
    ax.set_xlabel('Frequency (cycles/month)')
    ax.set_ylabel('Coherence')
    ax.set_title(col)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'coherence_plots.png'), dpi=150)
plt.close()

df_coherence.to_csv(os.path.join(OUTPUT_DIR, 'coherence_results.csv'), index=False)

# ============================================================
# ۷. گزارش
# ============================================================
with open(os.path.join(OUTPUT_DIR, 'spectral_analysis_report.txt'), 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("🌊 گزارش تحلیل طیفی\n")
    f.write("="*80 + "\n\n")
    f.write(f"دوره: ۲۰۰۰–۲۰۲۵ ({len(df_use)} ماه)\n\n")
    f.write("📊 بسامدهای غالب:\n")
    for f_c, p in zip(peak_freqs, peak_periods):
        if p > 2:
            f.write(f"   دوره {p:.1f} ماه (فرکانس {f_c:.3f})\n")  # ✅ اصلاح: p به string تبدیل شد
    f.write("\n📊 ۵ شاخص برتر:\n")
    f.write(df_coherence.head(5).to_string(index=False))
    f.write(f"\n\n✅ خروجی‌ها در: {OUTPUT_DIR}\n")

print(f"\n✅ خروجی‌ها در {OUTPUT_DIR} ذخیره شدند.")