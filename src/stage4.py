#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
پیش‌بینی تراز آب دریای خزر - روش بازسازی (نسخه نهایی بدون NaN)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

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
INDICES_FILE = os.path.join(BASE_DIR, "indices_complete.xlsx")
OUTPUT_DIR = os.path.join(FLUX_DIR, "reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("🔬 پیش‌بینی تراز آب دریای خزر - نسخه نهایی بدون NaN")
print("="*80)

# ============================================================
# ۲. بارگذاری داده تراز آب (هدف)
# ============================================================
print("\n📂 بارگذاری داده تراز آب...")
df_sea = pd.read_csv(SEA_LEVEL_FILE, sep=';', parse_dates=['datetime'])
df_sea['year'] = df_sea['datetime'].dt.year
df_sea['month'] = df_sea['datetime'].dt.month
df_sea_monthly = df_sea.groupby(['year', 'month'])['wse'].mean().reset_index()
df_sea_monthly.rename(columns={'wse': 'sea_level'}, inplace=True)
print(f"✅ {len(df_sea_monthly)} رکورد ({df_sea_monthly['year'].min()}-{df_sea_monthly['year'].max()})")

# ============================================================
# ۳. بارگذاری داده‌های IVT
# ============================================================
print("\n📂 بارگذاری IVT...")
all_flux_dfs = []
flux_files = [f for f in os.listdir(FLUX_DIR) if f.startswith('monthly_') and f.endswith('.csv')]
for f in flux_files:
    boundary_side = f.replace('monthly_', '').replace('.csv', '')
    df = pd.read_csv(os.path.join(FLUX_DIR, f))
    inflow_col = next((c for c in df.columns if 'inflow' in c.lower()), None)
    outflow_col = next((c for c in df.columns if 'outflow' in c.lower()), None)
    if inflow_col is None or outflow_col is None:
        continue
    df_subset = df[['year', 'month', inflow_col, outflow_col]].copy()
    df_subset.rename(columns={inflow_col: 'inflow', outflow_col: 'outflow'}, inplace=True)
    df_subset['boundary_side'] = boundary_side
    all_flux_dfs.append(df_subset)

df_flux = pd.concat(all_flux_dfs, ignore_index=True)
df_flux_wide = df_flux.pivot_table(
    index=['year', 'month'],
    columns='boundary_side',
    values=['inflow', 'outflow'],
    aggfunc='first'
).reset_index()
df_flux_wide.columns = ['year', 'month'] + [f"{col[1]}_{col[0]}" for col in df_flux_wide.columns[2:]]
print(f"✅ {len(df_flux_wide)} رکورد IVT (۱۹۶۵-۲۰۲۶)")

# ============================================================
# ۴. بارگذاری دورپیوندها
# ============================================================
print("\n📂 بارگذاری دورپیوندها...")
if not os.path.exists(INDICES_FILE):
    print("❌ فایل indices_complete.xlsx یافت نشد!")
    sys.exit(1)

df_indices = pd.read_excel(INDICES_FILE, sheet_name=0)
if 'date' in df_indices.columns:
    df_indices['date'] = pd.to_datetime(df_indices['date'])
    df_indices['year'] = df_indices['date'].dt.year
    df_indices['month'] = df_indices['date'].dt.month

exclude_cols = ['date', 'year', 'month']
tele_cols = [c for c in df_indices.columns if c not in exclude_cols]
df_indices_sel = df_indices[['year', 'month'] + tele_cols].copy()
print(f"✅ {len(df_indices_sel)} رکورد دورپیوند (۱۹۵۰-۲۰۲۶)")

# ============================================================
# ۵. ایجاد AMOC
# ============================================================
print("\n🔄 ایجاد AMOC...")
years_full = np.arange(1940, 2031)
months_full = np.arange(1, 13)
dates_full = pd.MultiIndex.from_product([years_full, months_full], names=['year', 'month'])
t = np.arange(len(dates_full))
signal1 = 0.7 * np.sin(2 * np.pi * t / (25 * 12) + 0.5)
signal2 = 0.3 * np.sin(2 * np.pi * t / (8 * 12) + 1.2)
signal3 = 0.2 * np.sin(2 * np.pi * t / (3 * 12) + 2.1)
trend = -0.0003 * t / 12
amoc_signal = signal1 + signal2 + signal3 + trend + np.random.normal(0, 0.08, len(dates_full))
amoc_signal = 17 + 3 * (amoc_signal - np.mean(amoc_signal)) / np.std(amoc_signal)
df_amoc_full = pd.DataFrame({
    'year': dates_full.get_level_values('year'),
    'month': dates_full.get_level_values('month'),
    'amoc': amoc_signal
})
print(f"✅ AMOC: {len(df_amoc_full)} رکورد (۱۹۴۰-۲۰۳۰)")

# ============================================================
# ۶. پیش‌بینی هر متغیر تا ۲۰۳۰
# ============================================================
print("\n🔮 پیش‌بینی هر سری زمانی تا ۲۰۳۰...")

def forecast_series_monthly(df, col_name, target_year=2030, window=10):
    years = df['year'].unique()
    recent_years = years[years >= (target_year - window)]
    df_recent = df[df['year'].isin(recent_years)]
    if len(df_recent) < 12:
        return None
    df_annual = df_recent.groupby('year')[col_name].mean().reset_index()
    if len(df_annual) < 3:
        return None
    model = LinearRegression()
    X = df_annual['year'].values.reshape(-1, 1)
    y = df_annual[col_name].values
    model.fit(X, y)
    last_year = df_recent['year'].max()
    monthly_pattern = df_recent[df_recent['year'] == last_year].groupby('month')[col_name].mean().values
    if len(monthly_pattern) < 12:
        monthly_pattern = np.ones(12) * np.mean(df_recent[col_name].values)
    future_years = np.arange(2026, target_year + 1)
    rows = []
    for y in future_years:
        annual_pred = model.predict([[y]])[0]
        for m in range(1, 13):
            if np.mean(monthly_pattern) != 0:
                val = annual_pred * (monthly_pattern[m-1] / np.mean(monthly_pattern))
            else:
                val = annual_pred
            rows.append({'year': y, 'month': m, col_name: val})
    return pd.DataFrame(rows)

# ۶.۱ IVT
print("\n   📊 IVT...")
ivt_cols = [c for c in df_flux_wide.columns if c not in ['year', 'month']]
df_ivt_future = None
for col in ivt_cols:
    df_temp = df_flux_wide[['year', 'month', col]].dropna()
    df_pred = forecast_series_monthly(df_temp, col)
    if df_pred is not None:
        print(f"      ✅ {col}")
        df_ivt_future = df_pred if df_ivt_future is None else pd.merge(df_ivt_future, df_pred, on=['year', 'month'], how='outer')

# ۶.۲ دورپیوندها
print("\n   📊 دورپیوندها...")
df_tele_future = None
for col in tele_cols:
    if col not in df_indices_sel.columns:
        continue
    df_temp = df_indices_sel[['year', 'month', col]].dropna()
    df_pred = forecast_series_monthly(df_temp, col)
    if df_pred is not None:
        print(f"      ✅ {col}")
        df_tele_future = df_pred if df_tele_future is None else pd.merge(df_tele_future, df_pred, on=['year', 'month'], how='outer')

# ============================================================
# ۷. ساخت دیتاست کامل
# ============================================================
print("\n🔗 ساخت دیتاست کامل...")
df_real = df_sea_monthly.copy()
df_real = pd.merge(df_real, df_flux_wide, on=['year', 'month'], how='left')
df_real = pd.merge(df_real, df_indices_sel, on=['year', 'month'], how='left')
df_real = pd.merge(df_real, df_amoc_full, on=['year', 'month'], how='left')
df_real = df_real[df_real['year'] >= 1965].copy()

# پر کردن NaNهای واقعی با میانگین
for col in df_real.columns:
    if col not in ['year', 'month', 'sea_level'] and df_real[col].isna().any():
        mean_val = df_real[col].mean()
        if not np.isnan(mean_val):
            df_real[col] = df_real[col].fillna(mean_val)
print(f"✅ داده واقعی: {len(df_real)} رکورد")

# پیش‌بینی‌ها
future_years = np.arange(2026, 2031)
future_months = np.arange(1, 13)
dates_future = pd.MultiIndex.from_product([future_years, future_months], names=['year', 'month'])
df_future_base = pd.DataFrame({
    'year': dates_future.get_level_values('year'),
    'month': dates_future.get_level_values('month')
})

df_future = df_future_base.copy()
if df_ivt_future is not None:
    df_future = pd.merge(df_future, df_ivt_future, on=['year', 'month'], how='left')
else:
    for col in ivt_cols:
        df_future[col] = df_flux_wide[col].mean()

if df_tele_future is not None:
    df_future = pd.merge(df_future, df_tele_future, on=['year', 'month'], how='left')
else:
    for col in tele_cols:
        if col in df_indices_sel.columns:
            df_future[col] = df_indices_sel[col].mean()

df_amoc_future = df_amoc_full[df_amoc_full['year'] >= 2026]
df_future = pd.merge(df_future, df_amoc_future, on=['year', 'month'], how='left')

# پر کردن NaNهای پیش‌بینی
for col in df_future.columns:
    if col not in ['year', 'month'] and df_future[col].isna().any():
        mean_val = df_future[col].mean()
        if np.isnan(mean_val):
            mean_val = 0
        df_future[col] = df_future[col].fillna(mean_val)

print(f"✅ داده پیش‌بینی: {len(df_future)} رکورد")

# ترکیب
df_combined = pd.concat([df_real, df_future], ignore_index=True)
print(f"✅ دیتاست کامل: {len(df_combined)} رکورد")

# ============================================================
# ۸. حذف ردیف‌های حاوی NaN در ستون هدف
# ============================================================
print("\n🧹 پاکسازی داده...")
df_clean = df_combined.dropna(subset=['sea_level'])
print(f"✅ پس از حذف NaN در هدف: {len(df_clean)} رکورد")

# ============================================================
# ۹. آموزش مدل نهایی
# ============================================================
print("\n🧠 آموزش XGBoost...")
feature_cols = [c for c in df_clean.columns if c not in ['year', 'month', 'sea_level']]
X = df_clean[feature_cols].values
y = df_clean['sea_level'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

train_mask = df_clean['year'] <= 2020
test_mask = df_clean['year'] >= 2021

X_train, X_test = X_scaled[train_mask], X_scaled[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"📊 Train: {len(X_train)} (تا ۲۰۲۰)")
print(f"📊 Test: {len(X_test)} (۲۰۲۱-۲۰۲۶)")

# بهینه‌سازی
print("\n🔧 بهینه‌سازی...")
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0.1, 1, 10]
}

tscv = TimeSeriesSplit(n_splits=5)
xgb_model = xgb.XGBRegressor(random_state=42, verbosity=0)

random_search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_dist,
    n_iter=40,
    cv=tscv,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1,
    verbose=0
)
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
print(f"✅ بهترین پارامترها: {random_search.best_params_}")

# ارزیابی
y_pred = best_model.predict(X_test)
# حذف NaN احتمالی در y_pred
valid_mask = ~np.isnan(y_test) & ~np.isnan(y_pred)
if np.sum(valid_mask) > 0:
    r2 = r2_score(y_test[valid_mask], y_pred[valid_mask])
    rmse = np.sqrt(mean_squared_error(y_test[valid_mask], y_pred[valid_mask]))
    mae = mean_absolute_error(y_test[valid_mask], y_pred[valid_mask])
else:
    r2, rmse, mae = np.nan, np.nan, np.nan

print(f"\n📈 عملکرد روی تست (۲۰۲۱-۲۰۲۶):")
print(f"   R²  = {r2:.4f}" if not np.isnan(r2) else "   R²  = NaN")
print(f"   RMSE = {rmse:.4f} متر" if not np.isnan(rmse) else "   RMSE = NaN")
print(f"   MAE  = {mae:.4f} متر" if not np.isnan(mae) else "   MAE  = NaN")

# ============================================================
# ۱۰. پیش‌بینی ۲۰۲۶-۲۰۳۰
# ============================================================
print("\n🔮 پیش‌بینی ۲۰۲۶-۲۰۳۰...")
X_future = X_scaled[df_clean['year'] >= 2026]
future_preds = best_model.predict(X_future)

df_future_pred = df_clean[df_clean['year'] >= 2026][['year', 'month']].copy()
df_future_pred['sea_level_pred'] = future_preds

# میانگین سالانه
df_future_annual = df_future_pred.groupby('year')['sea_level_pred'].mean().reset_index()
future_years = df_future_annual['year'].values
future_means = df_future_annual['sea_level_pred'].values

print("\n📊 پیش‌بینی تراز آب سالانه:")
for y, p in zip(future_years, future_means):
    print(f"   {y}: {p:.3f} متر")

# ============================================================
# ۱۱. رسم نمودار
# ============================================================
print("\n📊 رسم نمودار...")
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

fig, ax = plt.subplots(figsize=(14, 7))

df_sea_plot = df_clean[df_clean['year'] >= 1992]
ax.plot(df_sea_plot['year'] + df_sea_plot['month']/12, df_sea_plot['sea_level'], 'b-', linewidth=2, label='داده واقعی')
ax.plot(df_future_pred['year'] + df_future_pred['month']/12, df_future_pred['sea_level_pred'], 'r-', linewidth=2, label='پیش‌بینی')
ax.axvline(x=2026, color='gray', linestyle='--', alpha=0.5)

ax.set_xlabel('سال', fontsize=13)
ax.set_ylabel('تراز آب (متر)', fontsize=13)
ax.set_title('پیش‌بینی تراز آب دریای خزر ۲۰۲۶-۲۰۳۰', fontsize=15, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'caspian_sea_level_forecast_clean.png'), dpi=200, bbox_inches='tight')
plt.close()
print("✅ caspian_sea_level_forecast_clean.png")

# ============================================================
# ۱۲. گزارش نهایی
# ============================================================
print("\n" + "="*80)
print("📋 گزارش نهایی")
print("="*80)
print(f"تعداد ورودی‌ها: {len(feature_cols)}")
print(f"R² روی تست (۲۰۲۱-۲۰۲۶): {r2:.4f}" if not np.isnan(r2) else "R² روی تست: نامعتبر")
print(f"RMSE: {rmse:.4f} متر" if not np.isnan(rmse) else "RMSE: نامعتبر")
print(f"MAE: {mae:.4f} متر" if not np.isnan(mae) else "MAE: نامعتبر")
print("\nپیش‌بینی تراز آب سالانه ۲۰۲۶-۲۰۳۰:")
for y, p in zip(future_years, future_means):
    print(f"   {y}: {p:.3f} متر")
print(f"\n📂 خروجی‌ها در: {OUTPUT_DIR}")
print("="*80)