#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
پیش‌بینی تراز آب دریای خزر - روش ساده و علمی
با استفاده از میانگین متحرک وزنی و روند خطی
(برای داده‌های با تعداد کم و تغییرات شدید)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# ۱. تنظیمات مسیرها
# ============================================================
BASE_DIR = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503"
SEA_LEVEL_FILE = os.path.join(
    BASE_DIR,
    "Final_Analysis_Archive_20260702_060114",
    "basin_border",
    "caspian_unified_analysis",
    "caspian_sea_level_raw.csv"
)
OUTPUT_DIR = os.path.join(BASE_DIR, "final_analysis", "caspian_flux_output", "reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("📊 پیش‌بینی تراز آب دریای خزر (روش روند و میانگین متحرک)")
print("="*80)

# ============================================================
# ۲. بارگذاری داده تراز آب
# ============================================================
print("\n📂 بارگذاری داده تراز آب...")
df_sea = pd.read_csv(SEA_LEVEL_FILE, sep=';', parse_dates=['datetime'])
df_sea['year'] = df_sea['datetime'].dt.year
df_sea['month'] = df_sea['datetime'].dt.month
df_sea_monthly = df_sea.groupby(['year', 'month'])['wse'].mean().reset_index()
df_sea_monthly.rename(columns={'wse': 'sea_level'}, inplace=True)
print(f"✅ {len(df_sea_monthly)} رکورد ({df_sea_monthly['year'].min()}-{df_sea_monthly['year'].max()})")

# تبدیل به سری سالانه
df_annual = df_sea_monthly.groupby('year')['sea_level'].mean().reset_index()
print(f"✅ {len(df_annual)} رکورد سالانه")

# ============================================================
# ۳. محاسبه روند با وزن بیشتر به سال‌های اخیر
# ============================================================
print("\n📈 محاسبه روند با وزن‌دهی به سال‌های اخیر...")

# استفاده از ۱۰ سال اخیر (۲۰۱۶-۲۰۲۵)
recent_years = df_annual[df_annual['year'] >= 2016].copy()
print(f"   استفاده از {len(recent_years)} سال اخیر (۲۰۱۶-۲۰۲۵)")

# رگرسیون خطی با وزن‌دهی (سال‌های جدیدتر وزن بیشتر)
weights = np.arange(1, len(recent_years) + 1)  # وزن افزایشی
X = recent_years['year'].values.reshape(-1, 1)
y = recent_years['sea_level'].values

model = LinearRegression()
model.fit(X, y, sample_weight=weights)
slope = model.coef_[0]
intercept = model.intercept_
r2 = model.score(X, y)

print(f"   شیب روند: {slope:.4f} متر/سال")
print(f"   R² روند: {r2:.4f}")

# ============================================================
# ۴. پیش‌بینی ۲۰۲۶-۲۰۳۰ (با روند خطی)
# ============================================================
print("\n🔮 پیش‌بینی ۲۰۲۶-۲۰۳۰...")
future_years = np.array([2026, 2027, 2028, 2029, 2030]).reshape(-1, 1)
future_preds = model.predict(future_years)

# محاسبه فاصله اطمینان (بر اساس انحراف معیار باقیمانده‌های ۵ سال اخیر)
residuals = y - model.predict(X)
std_residual = np.std(residuals)
ci_lower = future_preds - 1.96 * std_residual
ci_upper = future_preds + 1.96 * std_residual

print("\n📊 پیش‌بینی‌ها:")
for y, p, l, u in zip([2026, 2027, 2028, 2029, 2030], future_preds, ci_lower, ci_upper):
    print(f"   {y}: {p:.3f} متر (فاصله اطمینان ۹۵٪: {l:.3f} تا {u:.3f})")

# ============================================================
# ۵. رسم نمودارها
# ============================================================
print("\n📊 رسم نمودارها...")

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

fig, ax = plt.subplots(figsize=(14, 7))

# داده واقعی
ax.plot(df_annual['year'], df_annual['sea_level'], 'b-', linewidth=2, label='تراز آب واقعی (سالانه)')

# روند خطی روی ۱۰ سال اخیر
years_fit = np.linspace(2016, 2030, 100)
fit_line = model.predict(years_fit.reshape(-1, 1))
ax.plot(years_fit, fit_line, 'g--', linewidth=2, label=f'روند خطی (شیب = {slope:.3f} متر/سال)')

# پیش‌بینی
ax.plot(future_years, future_preds, 'ro-', markersize=10, linewidth=2, label='پیش‌بینی ۲۰۲۶-۲۰۳۰')
ax.fill_between(future_years.flatten(), ci_lower, ci_upper, alpha=0.2, color='red', label='فاصله اطمینان ۹۵٪')

ax.axvline(x=2026, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('سال', fontsize=13)
ax.set_ylabel('تراز آب (متر)', fontsize=13)
ax.set_title('پیش‌بینی تراز آب دریای خزر (روند ۱۰ سال اخیر + وزن‌دهی)', fontsize=15, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'caspian_sea_level_forecast_trend.png'), dpi=200, bbox_inches='tight')
plt.close()
print("✅ caspian_sea_level_forecast_trend.png")

# ============================================================
# ۶. گزارش نهایی
# ============================================================
print("\n" + "="*80)
print("📋 گزارش نهایی")
print("="*80)
print(f"روش: رگرسیون خطی با وزن‌دهی به سال‌های اخیر (۲۰۱۶-۲۰۲۵)")
print(f"شیب روند: {slope:.4f} متر/سال")
print(f"R² روند: {r2:.4f}")
print(f"انحراف معیار باقیمانده: {std_residual:.3f} متر")
print("\nپیش‌بینی تراز آب برای ۲۰۲۶-۲۰۳۰:")
for y, p, l, u in zip([2026, 2027, 2028, 2029, 2030], future_preds, ci_lower, ci_upper):
    print(f"   {y}: {p:.3f} متر (۹۵% CI: {l:.3f} تا {u:.3f})")
print(f"\n📂 خروجی‌ها در: {OUTPUT_DIR}")
print("="*80)