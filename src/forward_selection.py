#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================================================
انتخاب افزایشی (Forward Selection) دورپیوندها برای پیش‌بینی تراز آب خزر
================================================================================
- مدل پایه: روند خطی + فصلی (سینوسی/کسینوسی)
- اضافه کردن تدریجی شاخص‌ها بر اساس بیشترین افزایش R²
- توقف زمانی که افزودن متغیر جدید بهبود معنی‌دار نداشته باشد
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = r"K:\\gozareshha\\Dr Farjami\\Dr Farjami\\140503"
OUTPUT_DIR = os.path.join(BASE_DIR, "final_analysis", "forward_selection")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEA_LEVEL_FILE = os.path.join(
    BASE_DIR,
    "Final_Analysis_Archive_20260702_060114",
    "basin_border",
    "caspian_unified_analysis",
    "caspian_sea_level_raw.csv"
)
INDICES_FILE = os.path.join(BASE_DIR, "indices_complete.xlsx")

STOP_THRESHOLD = 0.005
MAX_VARS = 10

print("="*80)
print("🌊 انتخاب افزایشی دورپیوندها برای پیش‌بینی تراز آب خزر")
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

y = df_use['sea_level'].values

# ============================================================
# ۳. مدل پایه
# ============================================================
print("\n📐 ساخت مدل پایه (روند + فصلی)...")

years_norm = (df_use['year'] - df_use['year'].min()) / 10
month_rad = 2 * np.pi * df_use['month'] / 12
sin_month = np.sin(month_rad)
cos_month = np.cos(month_rad)
sin_6month = np.sin(2 * month_rad)
cos_6month = np.cos(2 * month_rad)

df_base = pd.DataFrame({
    'year_norm': years_norm,
    'sin_month': sin_month,
    'cos_month': cos_month,
    'sin_6month': sin_6month,
    'cos_6month': cos_6month
})

model_base = LinearRegression()
model_base.fit(df_base, y)
r2_base = r2_score(y, model_base.predict(df_base))
print(f"✅ R² پایه: {r2_base:.4f}")

# ============================================================
# ۴. Forward Selection
# ============================================================
print("\n🔄 انتخاب افزایشی...")

tele_indices = [c for c in df_use.columns if c not in ['date', 'year', 'month', 'sea_level', 'wse'] + list(df_base.columns)]
print(f"📊 تعداد شاخص‌ها: {len(tele_indices)}")

selected = []
remaining = tele_indices.copy()
current_features = df_base.copy()
current_r2 = r2_base
history = []
step = 0

while remaining and len(selected) < MAX_VARS:
    step += 1
    improvements = {}
    for idx in remaining:
        X_test = current_features.copy()
        X_test[idx] = df_use[idx].values
        model_test = LinearRegression()
        model_test.fit(X_test, y)
        improvements[idx] = r2_score(y, model_test.predict(X_test))
    
    best_idx = max(improvements, key=improvements.get)
    best_r2 = improvements[best_idx]
    improvement = best_r2 - current_r2
    
    print(f"\nگام {step}: {best_idx} | R² = {best_r2:.4f} | افزایش = {improvement:.4f}")
    
    if improvement < STOP_THRESHOLD:
        print(f"⏹️ توقف (افزایش < {STOP_THRESHOLD})")
        break
    
    selected.append(best_idx)
    remaining.remove(best_idx)
    current_features[best_idx] = df_use[best_idx].values
    current_r2 = best_r2
    history.append({'step': step, 'added': best_idx, 'r2': current_r2})

print(f"\n✅ انتخاب کامل. {len(selected)} شاخص: {selected}")
print(f"📊 R² نهایی: {current_r2:.4f} | ارزش افزوده: {(current_r2 - r2_base)*100:.2f}%")

# ============================================================
# ۵. Cross-Validation
# ============================================================
print("\n📊 Cross-Validation...")
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []
for train_idx, val_idx in tscv.split(current_features):
    X_tr, X_val = current_features.iloc[train_idx], current_features.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    model_cv = LinearRegression()
    model_cv.fit(X_tr, y_tr)
    cv_scores.append(r2_score(y_val, model_cv.predict(X_val)))
print(f"✅ CV R²: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# ============================================================
# ۶. ذخیره
# ============================================================
model_final = LinearRegression()
model_final.fit(current_features, y)
coef_df = pd.DataFrame({'feature': current_features.columns, 'coefficient': model_final.coef_})
coef_df = coef_df.sort_values('coefficient', key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_use['date'], y, 'k-', alpha=0.5, label='Actual')
ax.plot(df_use['date'], model_base.predict(df_base), 'b-', alpha=0.7, label='Base')
ax.plot(df_use['date'], model_final.predict(current_features), 'r-', alpha=0.7, label='Final')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=150)
plt.close()

df_history = pd.DataFrame(history)
df_history.to_csv(os.path.join(OUTPUT_DIR, 'history.csv'), index=False)
coef_df.to_csv(os.path.join(OUTPUT_DIR, 'coefficients.csv'), index=False)

with open(os.path.join(OUTPUT_DIR, 'forward_selection_report.txt'), 'w', encoding='utf-8') as f:
    f.write("🌊 گزارش انتخاب افزایشی\n")
    f.write(f"دوره: ۲۰۰۰–۲۰۲۵\n")
    f.write(f"R² پایه: {r2_base:.4f}\n")
    f.write(f"R² نهایی: {current_r2:.4f}\n")
    f.write(f"ارزش افزوده: {(current_r2 - r2_base)*100:.2f}%\n")
    f.write(f"CV R²: {np.mean(cv_scores):.4f}\n")
    f.write(f"شاخص‌ها: {selected}\n")
    f.write(f"\nخروجی در: {OUTPUT_DIR}\n")

print(f"\n✅ خروجی‌ها در {OUTPUT_DIR}")