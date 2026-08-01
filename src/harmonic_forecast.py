#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================================================
پیش‌بینی هارمونیک (فوریه) تراز آب دریای خزر
================================================================================
- حذف روند (خطی) و فصلی (میانگین ماهانه)
- استخراج بسامدهای غالب با FFT (از spectral.py)
- برازش رگرسیون هارمونیک (سینوس/کسینوس) به باقیمانده
- پیش‌بینی هر مؤلفه به‌طور جداگانه تا ۲۰۳۰
- بازسازی با جمع مؤلفه‌ها + روند + فصلی
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# ۱. تنظیمات
# ============================================================
BASE_DIR = r"K:\\gozareshha\\Dr Farjami\\Dr Farjami\\140503"
OUTPUT_DIR = os.path.join(BASE_DIR, "final_analysis", "harmonic_forecast")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEA_LEVEL_FILE = os.path.join(
    BASE_DIR,
    "Final_Analysis_Archive_20260702_060114",
    "basin_border",
    "caspian_unified_analysis",
    "caspian_sea_level_raw.csv"
)

# بسامدهای غالب (از خروجی spectral.py)
# دوره‌ها: 52.3, 8.7, 5.2, 3.7, 3.3, 2.6 ماه
DOMINANT_PERIODS = [52.3, 8.7, 5.2, 3.7, 3.3, 2.6]  # ماه
DOMINANT_FREQS = [1/p for p in DOMINANT_PERIODS]    # سیکل بر ماه

# تعداد بسامدهایی که استفاده می‌شود (۵ بسامد اول کافی است)
N_HARMONICS = 5

print("="*80)
print("🌊 پیش‌بینی هارمونیک (فوریه) تراز آب خزر")
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
df_sea.rename(columns={'wse': 'sea_level'}, inplace=True)

# دوره ۲۰۰۰–۲۰۲۵
df_use = df_sea[(df_sea['year'] >= 2000) & (df_sea['year'] <= 2025)].copy()
print(f"✅ داده‌های نهایی: {len(df_use)} رکورد ماهانه")

y = df_use['sea_level'].values
t = np.arange(len(y))  # اندیس زمانی ۰ تا n-1
months = df_use['month'].values
years = df_use['year'].values
dates = df_use['date'].values

# ============================================================
# ۳. غیرایستا زدایی: حذف روند خطی و فصلی (میانگین ماهانه)
# ============================================================
print("\n🔄 غیرایستا زدایی (حذف روند و فصلی)...")

# ۳.۱ روند خطی
trend_coef = np.polyfit(t, y, 1)
trend = np.polyval(trend_coef, t)
y_detrend = y - trend

# ۳.۲ فصلی (میانگین ماهانه)
monthly_means = np.array([y_detrend[months == m].mean() for m in range(1, 13)])
seasonal = np.array([monthly_means[m-1] for m in months])
y_residual = y_detrend - seasonal

# میانگین باقیمانده (باید نزدیک صفر باشد)
residual_mean = np.mean(y_residual)
print(f"✅ میانگین باقیمانده: {residual_mean:.6f} متر (نزدیک صفر)")

# ============================================================
# ۴. مدل هارمونیک: برازش سینوس/کسینوس به باقیمانده
# ============================================================
print("\n📐 برازش مدل هارمونیک به باقیمانده...")

# تابع مدل هارمونیک: sum of sin/cos برای هر بسامد
def harmonic_model(t, *params):
    """
    params: [A1, B1, A2, B2, ...] برای هر بسامد
    """
    n_freqs = len(params) // 2
    result = np.zeros_like(t, dtype=float)
    for i in range(n_freqs):
        A = params[2*i]
        B = params[2*i+1]
        omega = 2 * np.pi * DOMINANT_FREQS[i]
        result += A * np.cos(omega * t) + B * np.sin(omega * t)
    return result

# تعداد پارامترها
n_params = 2 * N_HARMONICS
initial_guess = np.zeros(n_params)

# برازش
try:
    popt, pcov = curve_fit(harmonic_model, t, y_residual, p0=initial_guess, maxfev=5000)
    y_residual_fitted = harmonic_model(t, *popt)
    r2_residual = r2_score(y_residual, y_residual_fitted)
    print(f"✅ R² برازش هارمونیک روی باقیمانده: {r2_residual:.4f}")
except Exception as e:
    print(f"⚠️ خطا در برازش: {e}")
    sys.exit(1)

# استخراج دامنه و فاز برای هر مؤلفه (برای گزارش)
amplitudes = []
phases = []
for i in range(N_HARMONICS):
    A = popt[2*i]
    B = popt[2*i+1]
    amp = np.sqrt(A**2 + B**2)
    phase = np.arctan2(-B, A)  # فاز بر حسب رادیان
    amplitudes.append(amp)
    phases.append(phase)
    print(f"   بسامد {DOMINANT_FREQS[i]:.4f} (دوره {DOMINANT_PERIODS[i]:.1f} ماه): دامنه = {amp:.4f} متر, فاز = {phase:.2f} رادیان")

# ============================================================
# ۵. پیش‌بینی آینده (۲۰۲۶–۲۰۳۰)
# ============================================================
print("\n🔮 پیش‌بینی تا ۲۰۳۰...")

# اندیس‌های آینده
t_future = np.arange(len(y), len(y) + 60)  # ۵ سال = ۶۰ ماه
n_future = len(t_future)
future_dates = pd.date_range('2026-01-01', '2030-12-01', freq='MS')

# ۵.۱ پیش‌بینی روند
trend_future = np.polyval(trend_coef, t_future)

# ۵.۲ پیش‌بینی فصلی (با تکرار میانگین ماهانه)
months_future = future_dates.month
seasonal_future = np.array([monthly_means[m-1] for m in months_future])

# ۵.۳ پیش‌بینی مؤلفه‌های هارمونیک (تک‌تک)
harmonic_components_future = []
for i in range(N_HARMONICS):
    A = popt[2*i]
    B = popt[2*i+1]
    omega = 2 * np.pi * DOMINANT_FREQS[i]
    comp = A * np.cos(omega * t_future) + B * np.sin(omega * t_future)
    harmonic_components_future.append(comp)

# مجموع مؤلفه‌های هارمونیک
residual_future_sum = np.sum(harmonic_components_future, axis=0)

# ۵.۴ بازسازی کامل
y_future = trend_future + seasonal_future + residual_future_sum

# ============================================================
# ۶. محاسبه فاصله اطمینان (با Bootstrap روی باقیمانده‌ها)
# ============================================================
print("\n📊 محاسبه فاصله اطمینان ۹۵٪ با Bootstrap...")

n_bootstrap = 500
residuals_train = y_residual - y_residual_fitted
all_forecasts = []

for _ in range(n_bootstrap):
    # نمونه‌گیری با جایگزین از باقیمانده‌های آموزشی
    idx = np.random.choice(len(residuals_train), len(residuals_train), replace=True)
    resampled_resid = residuals_train[idx]
    
    # اضافه کردن نویز تصادفی به داده‌های آموزشی (برای شبیه‌سازی عدم‌قطعیت)
    # روش ساده‌تر: نمونه‌گیری از باقیمانده‌ها و بازسازی
    # برای هر بار، مدل را دوباره برازش نمی‌دهیم (برای سرعت)، بلکه خطاها را به پیش‌بینی اضافه می‌کنیم
    # این یک Bootstrap ساده برای فاصله اطمینان است
    pred_noise = np.random.choice(residuals_train, size=n_future, replace=True)
    y_boot = y_future + pred_noise
    all_forecasts.append(y_boot)

all_forecasts = np.array(all_forecasts)
ci_lower = np.percentile(all_forecasts, 2.5, axis=0)
ci_upper = np.percentile(all_forecasts, 97.5, axis=0)

# ============================================================
# ۷. ذخیره نتایج
# ============================================================
print("\n💾 ذخیره نتایج...")

# ۷.۱ جدول پیش‌بینی
df_forecast = pd.DataFrame({
    'date': future_dates,
    'year': future_dates.year,
    'month': future_dates.month,
    'trend': trend_future,
    'seasonal': seasonal_future,
    'harmonic_residual': residual_future_sum,
    'forecast': y_future,
    'ci_lower': ci_lower,
    'ci_upper': ci_upper
})
df_forecast.to_csv(os.path.join(OUTPUT_DIR, 'harmonic_forecast.csv'), index=False)

# ۷.۲ پیش‌بینی سالانه (میانگین)
df_annual_forecast = df_forecast.groupby('year').agg({
    'forecast': 'mean',
    'ci_lower': 'mean',
    'ci_upper': 'mean'
}).reset_index()
df_annual_forecast.to_csv(os.path.join(OUTPUT_DIR, 'harmonic_annual_forecast.csv'), index=False)

# ============================================================
# ۸. نمودارها
# ============================================================
print("\n📈 تولید نمودارها...")

# ۸.۱ نمودار کامل (تاریخی + پیش‌بینی)
fig, ax = plt.subplots(figsize=(16, 7))

# داده‌های تاریخی
ax.plot(dates, y, 'k-', linewidth=1.5, label='Historical (2000-2025)')

# مؤلفه‌های برازش‌شده روی داده‌های تاریخی
y_fitted = trend + seasonal + y_residual_fitted
ax.plot(dates, y_fitted, 'b-', linewidth=1, alpha=0.6, label='Harmonic Fit (2000-2025)')

# پیش‌بینی
ax.plot(future_dates, y_future, 'r-', linewidth=2, label='Forecast (2026-2030)')
ax.fill_between(future_dates, ci_lower, ci_upper, color='red', alpha=0.2, label='95% CI (Bootstrap)')

# خط شروع پیش‌بینی
ax.axvline(x=pd.Timestamp('2026-01-01'), color='gray', linestyle='--', linewidth=1.5, label='Forecast Start')

ax.set_xlabel('Date')
ax.set_ylabel('Sea Level (m)')
ax.set_title('Harmonic (Fourier) Forecast of Caspian Sea Level (2000-2030)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'harmonic_forecast_full.png'), dpi=150)
plt.close()

# ۸.۲ نمودار مؤلفه‌ها (برای نمایش تک‌تک)
fig, axes = plt.subplots(N_HARMONICS, 1, figsize=(14, 12), sharex=True)
for i, ax in enumerate(axes):
    comp = harmonic_model(t, *popt)  # کل مؤلفه‌ها
    # فقط مؤلفه‌ی iام را نمایش بده
    A = popt[2*i]
    B = popt[2*i+1]
    omega = 2 * np.pi * DOMINANT_FREQS[i]
    comp_i_train = A * np.cos(omega * t) + B * np.sin(omega * t)
    comp_i_future = A * np.cos(omega * t_future) + B * np.sin(omega * t_future)
    
    ax.plot(dates, comp_i_train, 'b-', linewidth=1.5, label=f'Period {DOMINANT_PERIODS[i]:.1f}m')
    ax.plot(future_dates, comp_i_future, 'r--', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Amplitude (m)')
    ax.set_title(f'Harmonic {i+1}: Period = {DOMINANT_PERIODS[i]:.1f} months, Amplitude = {amplitudes[i]:.4f} m')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    if i == N_HARMONICS-1:
        ax.set_xlabel('Date')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'harmonic_components.png'), dpi=150)
plt.close()

# ============================================================
# ۹. گزارش نهایی
# ============================================================
with open(os.path.join(OUTPUT_DIR, 'harmonic_report.txt'), 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("🌊 گزارش پیش‌بینی هارمونیک (فوریه) تراز آب خزر\n")
    f.write("="*80 + "\n\n")
    f.write(f"دوره آموزش: ۲۰۰۰–۲۰۲۵ ({len(y)} ماه)\n")
    f.write(f"دوره پیش‌بینی: ۲۰۲۶–۲۰۳۰ ({n_future} ماه)\n\n")
    
    f.write("📐 پارامترهای مدل:\n")
    f.write(f"  شیب روند: {trend_coef[0]:.6f} متر/ماه (={trend_coef[0]*12:.4f} متر/سال)\n")
    f.write(f"  R² برازش هارمونیک روی باقیمانده: {r2_residual:.4f}\n\n")
    
    f.write("📊 مؤلفه‌های هارمونیک:\n")
    for i in range(N_HARMONICS):
        f.write(f"  {i+1}. دوره {DOMINANT_PERIODS[i]:.1f} ماه, ")
        f.write(f"دامنه {amplitudes[i]:.4f} متر, ")
        f.write(f"فاز {phases[i]:.2f} رادیان\n")
    
    f.write("\n📈 پیش‌بینی سالانه:\n")
    f.write(df_annual_forecast.to_string(index=False) + "\n\n")
    
    f.write(f"✅ خروجی‌ها در: {OUTPUT_DIR}\n")

print(f"\n✅ همه خروجی‌ها در {OUTPUT_DIR} ذخیره شدند.")
print("📄 فایل‌های تولید شده:")
print("   - harmonic_forecast.csv (ماهانه)")
print("   - harmonic_annual_forecast.csv (سالانه)")
print("   - harmonic_forecast_full.png (نمودار کامل)")
print("   - harmonic_components.png (نمودار مؤلفه‌ها)")
print("   - harmonic_report.txt (گزارش)")
print("="*80)