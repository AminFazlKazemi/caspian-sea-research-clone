#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================================================
تحلیل موجک (Wavelet) تراز آب دریای خزر و دورپیوندها
================================================================================
- تبدیل موجک پیوسته (CWT)
- Cross-Wavelet Transform (XWT)
- Wavelet Coherence (WTC)
- فاز (تأخیر) در حوزه زمان-فرکانس
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pywt
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = r"K:\\gozareshha\\Dr Farjami\\Dr Farjami\\140503"
OUTPUT_DIR = os.path.join(BASE_DIR, "final_analysis", "wavelet_analysis")
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
print("🌊 تحلیل موجک (Wavelet) تراز آب و دورپیوندها")
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

df_all = df_sea.merge(df_indices, on=['date', 'year', 'month'], how='inner')
df_all = df_all.dropna()
df_all = df_all.sort_values('date').reset_index(drop=True)

df_use = df_all[(df_all['year'] >= 2000) & (df_all['year'] <= 2025)].copy()
print(f"✅ داده‌های نهایی: {len(df_use)} رکورد ماهانه")

sea_level = df_use['sea_level'].values
dates = df_use['date'].values
tele_cols = [c for c in df_use.columns if c not in ['date', 'year', 'month', 'sea_level', 'wse']]

# ============================================================
# ۳. حذف روند و فصلی
# ============================================================
def detrend_deseasonalize(signal, months):
    t = np.arange(len(signal))
    trend = np.polyval(np.polyfit(t, signal, 1), t)
    detrend = signal - trend
    monthly_means = np.array([detrend[months == m].mean() for m in range(1, 13)])
    seasonal = np.array([monthly_means[m-1] for m in months])
    return detrend - seasonal

months = df_use['month'].values
sea_res = detrend_deseasonalize(sea_level, months)

tele_res = {}
for idx in tele_cols:
    tele_res[idx] = detrend_deseasonalize(df_use[idx].values, months)

# ============================================================
# ۴. توابع موجک
# ============================================================
def wavelet_transform(signal, dt=1, scales=None, wavelet='cmor1.5-1.0'):
    if scales is None:
        scales = np.arange(1, 65, 0.5)
    coef, freqs = pywt.cwt(signal, scales, wavelet, dt)
    return coef, freqs, scales

def wavelet_coherence(sig1, sig2, dt=1, wavelet='cmor1.5-1.0', smoothing=5):
    W1, f, scales = wavelet_transform(sig1, dt, wavelet=wavelet)
    W2, _, _ = wavelet_transform(sig2, dt, scales, wavelet=wavelet)
    
    W1_smooth = uniform_filter(np.abs(W1)**2, size=(smoothing, 1))
    W2_smooth = uniform_filter(np.abs(W2)**2, size=(smoothing, 1))
    W12_smooth = uniform_filter(W1 * np.conj(W2), size=(smoothing, 1))
    
    coherence = np.abs(W12_smooth)**2 / (W1_smooth * W2_smooth + 1e-10)
    return coherence, f, scales

def phase_difference(sig1, sig2, dt=1, wavelet='cmor1.5-1.0'):
    W1, f, scales = wavelet_transform(sig1, dt, wavelet=wavelet)
    W2, _, _ = wavelet_transform(sig2, dt, scales, wavelet=wavelet)
    phase = np.angle(W1 * np.conj(W2))
    return phase, f, scales

# ============================================================
# ۵. تحلیل موجک
# ============================================================
print("\n🌀 اجرای تحلیل موجک...")

# ۵.۱ طیف توان تراز آب
W_sea, f, scales = wavelet_transform(sea_res, dt=1)
periods = 1 / f

fig, ax = plt.subplots(figsize=(14, 6))
im = ax.contourf(dates, periods, np.abs(W_sea), levels=50, cmap='jet')
ax.set_yscale('log')
ax.set_ylim(2, 64)
ax.invert_yaxis()
ax.set_xlabel('Date')
ax.set_ylabel('Period (months)')
ax.set_title('Wavelet Power Spectrum - Caspian Sea Level')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Power')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'wavelet_power_sea.png'), dpi=150)
plt.close()

# ۵.۲ انتخاب ۵ شاخص برتر (بر اساس همبستگی پیرسون)
corrs = {col: np.corrcoef(sea_level, df_use[col].values)[0, 1] for col in tele_cols}
selected = sorted(corrs, key=lambda x: abs(corrs[x]), reverse=True)[:5]

# ۵.۳ Coherence برای هر شاخص
fig = plt.figure(figsize=(18, 15))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

for i, idx in enumerate(selected[:5]):
    row, col = divmod(i, 2)
    if i < 4:
        ax = fig.add_subplot(gs[row, col])
    else:
        ax = fig.add_subplot(gs[2, 0:2])
    
    sig_tele = tele_res[idx]
    coh, f_coh, _ = wavelet_coherence(sea_res, sig_tele, dt=1)
    periods_coh = 1 / f_coh
    phase, f_phase, _ = phase_difference(sea_res, sig_tele, dt=1)
    
    im = ax.contourf(dates, periods_coh, coh, levels=50, cmap='viridis')
    ax.set_yscale('log')
    ax.set_ylim(2, 64)
    ax.invert_yaxis()
    
    # فلش‌های فاز
    step_t = max(1, len(dates)//30)
    step_s = max(1, len(periods_coh)//10)
    for ti in range(0, len(dates), step_t):
        for si in range(0, len(periods_coh), step_s):
            if coh[si, ti] > 0.5:
                ph = phase[si, ti]
                ax.arrow(dates[ti], periods_coh[si], 
                        0.3*np.cos(ph), -0.3*np.sin(ph),
                        head_width=0.2, head_length=0.3,
                        color='white', alpha=0.6, linewidth=0.5)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Period (months)')
    ax.set_title(f'{idx} - Wavelet Coherence')
    if i == 0:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Coherence')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'wavelet_coherence_all.png'), dpi=150)
plt.close()

# ۵.۴ خلاصه
summary = []
for idx in selected:
    coh, f_coh, _ = wavelet_coherence(sea_res, tele_res[idx], dt=1)
    phase, f_phase, _ = phase_difference(sea_res, tele_res[idx], dt=1)
    max_idx = np.unravel_index(np.argmax(coh), coh.shape)
    best_period = 1 / f_coh[max_idx[0]]
    best_phase = phase[max_idx[0], max_idx[1]]
    lag = (best_phase / (2 * np.pi)) * best_period
    summary.append({
        'index': idx,
        'best_period (months)': best_period,
        'max_coherence': np.max(coh),
        'phase (radians)': best_phase,
        'lag (months)': lag
    })

df_summary = pd.DataFrame(summary).sort_values('max_coherence', ascending=False)
df_summary.to_csv(os.path.join(OUTPUT_DIR, 'wavelet_summary.csv'), index=False)

print("\n✅ خلاصه نتایج موجک:")
print(df_summary.to_string(index=False))

# ============================================================
# ۶. گزارش
# ============================================================
with open(os.path.join(OUTPUT_DIR, 'wavelet_report.txt'), 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("🌊 گزارش تحلیل موجک\n")
    f.write("="*80 + "\n\n")
    f.write(f"دوره: ۲۰۰۰–۲۰۲۵ ({len(df_use)} ماه)\n\n")
    f.write("📊 نتایج:\n")
    f.write(df_summary.to_string(index=False) + "\n\n")
    best = df_summary.iloc[0]
    f.write(f"📌 بهترین: {best['index']} | دوره {best['best_period (months)']:.1f} ماه | Coherence {best['max_coherence']:.3f} | تأخیر {best['lag (months)']:.1f} ماه\n")
    f.write(f"\n✅ خروجی‌ها در: {OUTPUT_DIR}\n")

print(f"\n✅ خروجی‌ها در {OUTPUT_DIR} ذخیره شدند.")
