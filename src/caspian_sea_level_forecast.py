#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
پیش‌بینی تراز آب دریای خزر - نسخه نهایی (با وزن‌دهی و Random Forest)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample

# ============================================================
# ۱. تنظیمات مسیرها
# ============================================================
BASE_DIR = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503"
FLUX_DIR = os.path.join(BASE_DIR, "final_analysis", "caspian_flux_output")
SEA_LEVEL_FILE = os.path.join(
    BASE_DIR,
    "Final_Analysis_Archive_20260702_060114",
    "basin_border",
    "caspian_unified_analysis",
    "caspian_sea_level_raw.csv"
)
OUTPUT_DIR = os.path.join(FLUX_DIR, "reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_SIZE = 0.2
FORECAST_STEPS = 5
RANDOM_STATE = 42

print("="*80)
print("🚀 پیش‌بینی تراز آب دریای خزر - نسخه نهایی (Random Forest + وزن‌دهی)")
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

# ============================================================
# ۳. بارگذاری داده‌های شار ماهانه
# ============================================================
print("\n📂 بارگذاری داده‌های شار ماهانه...")

all_flux_dfs = []
flux_files = [f for f in os.listdir(FLUX_DIR) if f.startswith('monthly_') and f.endswith('.csv')]
print(f"   تعداد فایل‌های شار: {len(flux_files)}")

for f in flux_files:
    boundary_side = f.replace('monthly_', '').replace('.csv', '')
    df = pd.read_csv(os.path.join(FLUX_DIR, f))
    
    inflow_col = None
    outflow_col = None
    for col in df.columns:
        if 'inflow' in col.lower():
            inflow_col = col
        if 'outflow' in col.lower():
            outflow_col = col
    
    if inflow_col is None or outflow_col is None:
        print(f"⚠️ فایل {f} فاقد ستون‌های inflow/outflow. رد می‌شود.")
        continue
    
    df_subset = df[['year', 'month', inflow_col, outflow_col]].copy()
    df_subset.rename(columns={inflow_col: 'inflow', outflow_col: 'outflow'}, inplace=True)
    df_subset['boundary_side'] = boundary_side
    all_flux_dfs.append(df_subset)

if not all_flux_dfs:
    print("❌ هیچ داده شار معتبری یافت نشد!")
    sys.exit(1)

df_flux = pd.concat(all_flux_dfs, ignore_index=True)
print(f"✅ {len(df_flux)} رکورد شار بارگذاری شد.")

# تبدیل به فرمت گسترده
df_flux_wide = df_flux.pivot_table(
    index=['year', 'month'],
    columns='boundary_side',
    values=['inflow', 'outflow'],
    aggfunc='first'
).reset_index()
df_flux_wide.columns = ['year', 'month'] + [f"{col[1]}_{col[0]}" for col in df_flux_wide.columns[2:]]
print(f"✅ {len(df_flux_wide)} رکورد شار در فرمت گسترده")

# ============================================================
# ۴. ادغام داده‌ها (فقط شارها، بدون Lags/Rolling)
# ============================================================
print("\n🔗 ادغام داده‌ها (فقط شارها)...")
df_all = pd.merge(df_sea_monthly, df_flux_wide, on=['year', 'month'], how='inner')
print(f"✅ {len(df_all)} رکورد پس از ادغام")

# حذف ردیف‌های NaN
df_all = df_all.dropna()
print(f"✅ {len(df_all)} رکورد نهایی")

# ============================================================
# ۵. آماده‌سازی داده
# ============================================================
feature_cols = [c for c in df_all.columns if c not in ['year', 'month', 'sea_level']]
X = df_all[feature_cols].values
y = df_all['sea_level'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# تقسیم داده (حفظ ترتیب زمانی)
train_size = int((1 - TEST_SIZE) * len(X_scaled))
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"\n📊 Train: {len(X_train)} ({df_all['year'].iloc[0]}–{df_all['year'].iloc[train_size-1]})")
print(f"   Test: {len(X_test)} ({df_all['year'].iloc[train_size]}–{df_all['year'].iloc[-1]})")

# ============================================================
# ۶. وزن‌دهی به داده‌های جدیدتر
# ============================================================
print("\n⚖️ اعمال وزن به داده‌های جدیدتر...")
weights = np.ones(len(y_train))
years_train = df_all['year'].iloc[:train_size].values
weights[years_train >= 2010] = 2.0
weights[years_train >= 2015] = 4.0
weights[years_train >= 2018] = 8.0
print(f"   وزن‌ها: {np.unique(weights)}")

# ============================================================
# ۷. آموزش Random Forest
# ============================================================
print("\n🌲 آموزش Random Forest با وزن‌دهی...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
model.fit(X_train, y_train, sample_weight=weights)

# ============================================================
# ۸. ارزیابی مدل
# ============================================================
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"\n📈 عملکرد روی تست:")
print(f"   R²  = {r2:.4f}")
print(f"   RMSE = {rmse:.4f} متر")
print(f"   MAE  = {mae:.4f} متر")

# ============================================================
# ۹. اهمیت ویژگی‌ها (Gini Importance)
# ============================================================
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 ۵ ویژگی مهم (Gini):")
print(importance_df.head(5).to_string(index=False))

# ============================================================
# ۱۰. پیش‌بینی ۲۰۲۶-۲۰۳۰ (با میانگین متحرک)
# ============================================================
print("\n🔮 پیش‌بینی ۲۰۲۶-۲۰۳۰ (بر اساس روند اخیر)...")

# روش ۱: مدل
last_X = X_scaled[-1:].copy()
future_preds_model = []
for _ in range(FORECAST_STEPS):
    pred = model.predict(last_X)[0]
    future_preds_model.append(pred)
    last_X = last_X * 0.99 + 0.01 * np.mean(X_train, axis=0).reshape(1, -1)

# روش ۲: روند اخیر (میانگین متحرک)
recent = df_all[df_all['year'] >= 2015]
slope = (recent['sea_level'].iloc[-1] - recent['sea_level'].iloc[0]) / (len(recent) - 1)
future_preds_trend = [df_all['sea_level'].iloc[-1] + slope * i for i in range(1, 6)]

future_years = list(range(2026, 2026 + FORECAST_STEPS))

# ============================================================
# ۱۱. رسم نمودارها
# ============================================================
print("\n📊 رسم نمودارها...")

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df_all['year'] + df_all['month']/12, df_all['sea_level'], 'b-', linewidth=2, label='داده واقعی')
ax.plot(future_years, future_preds_trend, 'ro-', markersize=8, linewidth=2, label='پیش‌بینی (روند اخیر)')
ax.plot(future_years, future_preds_model, 'gs-', markersize=8, linewidth=2, label='پیش‌بینی (مدل)')
ax.axvline(x=2026, color='gray', linestyle='--', alpha=0.5, label='شروع پیش‌بینی')
ax.set_xlabel('سال', fontsize=13)
ax.set_ylabel('تراز آب (متر)', fontsize=13)
ax.set_title('پیش‌بینی تراز آب دریای خزر ۲۰۲۶–۲۰۳۰', fontsize=15, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sea_level_forecast_final_rf.png'), dpi=200, bbox_inches='tight')
plt.close()
print("✅ sea_level_forecast_final_rf.png")

# ============================================================
# ۱۲. گزارش نهایی
# ============================================================
print("\n" + "="*80)
print("📋 گزارش نهایی")
print("="*80)
print(f"R² روی تست: {r2:.4f}")
print(f"RMSE: {rmse:.4f} متر")
print(f"MAE: {mae:.4f} متر")
print("\n۵ ویژگی با بیشترین اهمیت (Gini):")
for i, row in importance_df.head(5).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")
print(f"\nپیش‌بینی (روند اخیر) برای {future_years}:")
for y, p in zip(future_years, future_preds_trend):
    print(f"   {y}: {p:.3f} متر")
print(f"\nپیش‌بینی (مدل) برای {future_years}:")
for y, p in zip(future_years, future_preds_model):
    print(f"   {y}: {p:.3f} متر")
print(f"\n📂 خروجی‌ها در: {OUTPUT_DIR}")
print("="*80)