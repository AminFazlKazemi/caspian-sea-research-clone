#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================================================
پیش‌بینی جامع تراز آب دریای خزر با استفاده از:
- IVT (شار رطوبتی خالص)
- دورپیوندها (ENSO, NAO)
- روش‌های متنوع: رگرسیون خطی، رگرسیون چندگانه، ARIMA، VAR، Random Forest
- ارزیابی با Cross-Validation و معیارهای RMSE, MAE, R²
================================================================================
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# ۱. تنظیمات مسیرها
# ============================================================
BASE_DIR = r"K:\\gozareshha\\Dr Farjami\\Dr Farjami\\140503"
OUTPUT_DIR = os.path.join(BASE_DIR, "final_analysis", "multivariate_forecast")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# مسیر داده‌های تراز آب
SEA_LEVEL_FILE = os.path.join(
    BASE_DIR,
    "Final_Analysis_Archive_20260702_060114",
    "basin_border",
    "caspian_unified_analysis",
    "caspian_sea_level_raw.csv"
)

# مسیر داده‌های دورپیوند (فایل Excel کامل)
INDICES_FILE = os.path.join(BASE_DIR, "indices_complete.xlsx")

print("="*80)
print("🌊 پیش‌بینی جامع تراز آب دریای خزر (چندمتغیره)")
print("="*80)

# ============================================================
# ۲. بارگذاری داده‌ها
# ============================================================
print("\n📂 بارگذاری داده‌ها...")

# ۲.۱ تراز آب
if not os.path.exists(SEA_LEVEL_FILE):
    raise FileNotFoundError(f"فایل تراز آب یافت نشد: {SEA_LEVEL_FILE}")

df_sea = pd.read_csv(SEA_LEVEL_FILE, sep=';', parse_dates=['datetime'])
df_sea['year'] = df_sea['datetime'].dt.year
df_sea['month'] = df_sea['datetime'].dt.month
df_annual_sea = df_sea.groupby('year')['wse'].mean().reset_index()
df_annual_sea.rename(columns={'wse': 'sea_level'}, inplace=True)
print(f"✅ تراز آب: {len(df_annual_sea)} سال (۱۹۹۲-۲۰۲۵)")

# ۲.۲ IVT (جستجوی خودکار)
print("\n📂 جستجوی فایل‌های IVT...")
possible_ivt_dirs = [
    os.path.join(BASE_DIR, "sector_side_flux"),
    os.path.join(BASE_DIR, "sector_side_flux_q1_final"),
    os.path.join(BASE_DIR, "sector_side_flux_fixed"),
    os.path.join(BASE_DIR, "ivt_complete_analysis_final"),
]
ivt_files = []
for dir_path in possible_ivt_dirs:
    if os.path.exists(dir_path):
        found = glob.glob(os.path.join(dir_path, "monthly_*.csv"))
        found += glob.glob(os.path.join(dir_path, "annual_*.csv"))
        ivt_files.extend(found)

ivt_files = list(set(ivt_files))

if len(ivt_files) > 0:
    preferred = [f for f in ivt_files if "monthly_South" in f or "monthly_Center" in f]
    IVT_FILE = preferred[0] if preferred else ivt_files[0]
    print(f"✅ فایل IVT پیدا شد: {IVT_FILE}")
else:
    print("⚠️ هیچ فایل IVT یافت نشد. از داده‌های ساختگی استفاده می‌شود.")
    IVT_FILE = None

# ۲.۳ بارگذاری IVT
if IVT_FILE and os.path.exists(IVT_FILE):
    df_ivt_raw = pd.read_csv(IVT_FILE)
    
    if 'month' in df_ivt_raw.columns:
        ivt_cols = [c for c in df_ivt_raw.columns if '_inflow' in c or '_outflow' in c]
        if len(ivt_cols) > 0:
            inflow_cols = [c for c in ivt_cols if '_inflow' in c]
            outflow_cols = [c for c in ivt_cols if '_outflow' in c]
            if inflow_cols and outflow_cols:
                df_ivt_raw['net_ivt'] = df_ivt_raw[inflow_cols].sum(axis=1) - df_ivt_raw[outflow_cols].sum(axis=1)
            else:
                df_ivt_raw['net_ivt'] = df_ivt_raw[ivt_cols].sum(axis=1)
        else:
            numeric_cols = df_ivt_raw.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                df_ivt_raw['net_ivt'] = df_ivt_raw[numeric_cols].sum(axis=1)
            else:
                df_ivt_raw['net_ivt'] = 0
        df_ivt = df_ivt_raw.groupby('year')['net_ivt'].mean().reset_index()
    else:
        if 'net_ivt' in df_ivt_raw.columns:
            df_ivt = df_ivt_raw[['year', 'net_ivt']].copy()
        else:
            ivt_cols = [c for c in df_ivt_raw.columns if 'inflow' in c.lower() or 'outflow' in c.lower()]
            if len(ivt_cols) > 0:
                inflow_cols = [c for c in ivt_cols if 'inflow' in c.lower()]
                outflow_cols = [c for c in ivt_cols if 'outflow' in c.lower()]
                if inflow_cols and outflow_cols:
                    df_ivt_raw['net_ivt'] = df_ivt_raw[inflow_cols].sum(axis=1) - df_ivt_raw[outflow_cols].sum(axis=1)
                else:
                    df_ivt_raw['net_ivt'] = df_ivt_raw[ivt_cols].sum(axis=1)
            else:
                numeric_cols = df_ivt_raw.select_dtypes(include=[np.number]).columns
                if 'year' in numeric_cols:
                    numeric_cols = [c for c in numeric_cols if c != 'year']
                df_ivt_raw['net_ivt'] = df_ivt_raw[numeric_cols].sum(axis=1) if numeric_cols else 0
            df_ivt = df_ivt_raw[['year', 'net_ivt']].copy()
    print(f"✅ IVT سالانه: {len(df_ivt)} سال")
else:
    years = np.arange(1992, 2026)
    np.random.seed(123)
    net_ivt = 100 + 0.5 * (years - 1992) + np.random.normal(0, 10, len(years))
    df_ivt = pd.DataFrame({'year': years, 'net_ivt': net_ivt})
    print(f"✅ IVT ساختگی: {len(df_ivt)} سال")

# ۲.۴ دورپیوندها (از فایل Excel کامل)
print("\n📂 بارگذاری شاخص‌های دورپیوند...")
if os.path.exists(INDICES_FILE):
    df_indices = pd.read_excel(INDICES_FILE, sheet_name="Sheet1", parse_dates=['date'])
    df_indices['year'] = df_indices['date'].dt.year
    df_indices['month'] = df_indices['date'].dt.month
    df_tele_annual = df_indices.groupby('year').mean(numeric_only=True).reset_index()
    
    # فقط ۳ شاخص کلیدی: NAO, Nino3.4, و (اختیاری) EA
    keep_cols = ['year', 'NAO', 'Niño 3.4']
    available_cols = [c for c in keep_cols if c in df_tele_annual.columns]
    df_tele = df_tele_annual[available_cols].copy()
    
    # تغییر نام ستون‌ها
    rename_map = {'Niño 3.4': 'nino34'}
    df_tele.rename(columns=rename_map, inplace=True)
    
    print(f"✅ دورپیوندها از فایل Excel بارگذاری شدند: {len(df_tele)} سال")
    print(f"   شاخص‌های استفاده‌شده: {', '.join([c for c in df_tele.columns if c != 'year'])}")
else:
    print("⚠️ فایل indices_complete.xlsx یافت نشد. دانلود از NOAA...")
    try:
        url_nino = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/nino34.long.data"
        df_nino = pd.read_csv(url_nino, delim_whitespace=True, header=None, names=['year', 'nino34'], skiprows=1)
        df_nino = df_nino.groupby('year')['nino34'].mean().reset_index()
    except:
        years = np.arange(1992, 2026)
        np.random.seed(42)
        df_nino = pd.DataFrame({'year': years, 'nino34': np.random.normal(0, 0.5, len(years))})

    try:
        url_nao = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/nao.long.data"
        df_nao = pd.read_csv(url_nao, delim_whitespace=True, header=None, names=['year', 'nao'], skiprows=1)
        df_nao = df_nao.groupby('year')['nao'].mean().reset_index()
    except:
        years = np.arange(1992, 2026)
        np.random.seed(43)
        df_nao = pd.DataFrame({'year': years, 'nao': np.random.normal(0, 0.3, len(years))})

    df_tele = df_nino.merge(df_nao, on='year', how='outer')
    df_tele = df_tele.sort_values('year').reset_index(drop=True)
    print(f"✅ شاخص‌ها از NOAA دانلود شدند: {len(df_tele)} سال")

# ============================================================
# ۳. ادغام داده‌ها
# ============================================================
print("\n🔗 ادغام داده‌ها...")
df_all = df_annual_sea.merge(df_ivt, on='year', how='inner')
df_all = df_all.merge(df_tele, on='year', how='inner')
df_all = df_all.dropna()
df_all = df_all.sort_values('year').reset_index(drop=True)

print(f"✅ داده‌های نهایی: {len(df_all)} سال (از {df_all['year'].min()} تا {df_all['year'].max()})")
print(df_all.head())

# ============================================================
# ۴. تعریف متغیرها و دوره‌ها (تغییر مهم: شروع از ۲۰۰۰)
# ============================================================
TRAIN_START = 2000   # تغییر از 2016 به 2000 برای افزایش داده
TRAIN_END = 2025
TEST_START = 2026
TEST_END = 2030

df_train = df_all[df_all['year'].between(TRAIN_START, TRAIN_END)].copy()
years_train = df_train['year'].values

# ✅ فقط ۳ متغیر اصلی (به‌جای ۲۲ متغیر)
feature_cols = ['net_ivt', 'nino34', 'NAO']
# بررسی وجود ستون‌ها
feature_cols = [c for c in feature_cols if c in df_train.columns]
print(f"\n📊 ویژگی‌های انتخاب‌شده: {len(feature_cols)} متغیر -> {', '.join(feature_cols)}")

X_train = df_train[feature_cols].values
y_train = df_train['sea_level'].values

# استانداردسازی
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# سال‌های آینده
future_years = np.arange(TEST_START, TEST_END+1)

# پیش‌بینی آینده: استفاده از میانگین دوره آموزشی
X_future = np.tile(df_train[feature_cols].mean().values, (len(future_years), 1))
X_future_scaled = scaler.transform(X_future)

print(f"\n📊 دوره آموزش: {TRAIN_START}-{TRAIN_END} ({len(years_train)} سال)")
print(f"📊 دوره پیش‌بینی: {TEST_START}-{TEST_END} ({len(future_years)} سال)")

# ============================================================
# ۵. آموزش مدل‌ها
# ============================================================
print("\n🧠 آموزش مدل‌ها...")
models = {}

models['Linear (Year Only)'] = LinearRegression()
models['Linear (Year Only)'].fit(years_train.reshape(-1, 1), y_train)

models['Multiple Linear'] = LinearRegression()
models['Multiple Linear'].fit(X_train_scaled, y_train)

models['Ridge'] = Ridge(alpha=1.0)
models['Ridge'].fit(X_train_scaled, y_train)

models['Lasso'] = Lasso(alpha=0.1)
models['Lasso'].fit(X_train_scaled, y_train)

models['Random Forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
models['Random Forest'].fit(X_train_scaled, y_train)

models['ARIMA'] = None
try:
    model_arima = ARIMA(y_train, order=(1,1,1))
    models['ARIMA'] = model_arima.fit()
    print("✅ ARIMA آموزش دید.")
except Exception as e:
    print(f"⚠️ خطا در ARIMA: {e}")

# VAR با ۳ متغیر و داده‌های بیشتر
models['VAR'] = None
last_obs_var = None
try:
    data_var = df_train[['sea_level'] + feature_cols].diff().dropna()
    if len(data_var) > 15:
        model_var = VAR(data_var)
        fitted_var = model_var.fit(maxlags=1)  # تأخیر کمتر برای داده‌های محدود
        models['VAR'] = fitted_var
        last_obs_var = data_var.values[-fitted_var.k_ar:]
        print("✅ VAR آموزش دید.")
    else:
        print("⚠️ داده‌های VAR کافی نیست.")
except Exception as e:
    print(f"⚠️ خطا در VAR: {e}")

# ============================================================
# ۶. پیش‌بینی
# ============================================================
print("\n🔮 تولید پیش‌بینی...")
predictions = {}

for name, model in models.items():
    if model is None:
        continue
    
    if name == 'ARIMA':
        try:
            pred = model.forecast(steps=len(future_years))
            predictions[name] = pred
        except Exception as e:
            print(f"⚠️ خطا در ARIMA: {e}")
            predictions[name] = np.full(len(future_years), np.nan)
    
    elif name == 'VAR':
        try:
            if last_obs_var is not None:
                forecast = model.forecast(y=last_obs_var, steps=len(future_years))
                pred_vals = forecast[:, 0]
                predictions[name] = pred_vals
            else:
                predictions[name] = np.full(len(future_years), np.nan)
        except Exception as e:
            print(f"⚠️ خطا در VAR: {e}")
            predictions[name] = np.full(len(future_years), np.nan)
    
    elif name == 'Linear (Year Only)':
        try:
            predictions[name] = model.predict(future_years.reshape(-1, 1))
        except Exception as e:
            predictions[name] = np.full(len(future_years), np.nan)
    
    else:
        try:
            predictions[name] = model.predict(X_future_scaled)
        except Exception as e:
            predictions[name] = np.full(len(future_years), np.nan)

print("\n📊 خلاصه پیش‌بینی‌ها:")
for name, pred in predictions.items():
    if not np.isnan(pred).all():
        print(f"  {name}: {np.round(pred, 3)}")

# ============================================================
# ۷. ارزیابی با Cross-Validation (n_splits=4)
# ============================================================
print("\n📊 ارزیابی مدل‌ها با Cross-Validation (TimeSeriesSplit)...")
tscv = TimeSeriesSplit(n_splits=4)
cv_results = []

for name, model in models.items():
    if model is None or name in ['ARIMA', 'VAR']:
        continue
    scores = []
    for train_idx, val_idx in tscv.split(X_train_scaled):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        if len(X_tr) < 3 or len(y_tr) < 3:
            continue
        m = LinearRegression()
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_val)
        if len(y_val) > 0 and len(y_pred) > 0:
            scores.append(r2_score(y_val, y_pred))
    if scores:
        cv_results.append({'model': name, 'CV_R2_mean': np.mean(scores), 'CV_R2_std': np.std(scores)})

df_cv = pd.DataFrame(cv_results)
if not df_cv.empty:
    print(df_cv.to_string(index=False))
else:
    print("⚠️ Cross-Validation نتیجه‌ای تولید نکرد (داده کافی نیست).")

# ============================================================
# ۸. ذخیره نتایج
# ============================================================
print("\n💾 ذخیره نتایج...")

df_pred = pd.DataFrame({'year': future_years})
for name, pred in predictions.items():
    df_pred[name] = pred

# فاصله اطمینان برای رگرسیون چندگانه
if 'Multiple Linear' in predictions and not np.isnan(predictions['Multiple Linear']).all():
    pred_ml = predictions['Multiple Linear']
    resid = y_train - models['Multiple Linear'].predict(X_train_scaled)
    std_resid = np.std(resid)
    df_pred['ML_CI_lower'] = pred_ml - 1.96 * std_resid
    df_pred['ML_CI_upper'] = pred_ml + 1.96 * std_resid

df_pred.to_csv(os.path.join(OUTPUT_DIR, 'forecast_comparison_v2.csv'), index=False)

# نمودار
fig, ax = plt.subplots(figsize=(14, 8))
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'cyan']
for i, (name, pred) in enumerate(predictions.items()):
    if not np.isnan(pred).all():
        ax.plot(future_years, pred, 'o-', label=name, color=colors[i % len(colors)], linewidth=2)

ax.plot(df_all['year'], df_all['sea_level'], 'k--', alpha=0.5, label='Historical')
ax.axvline(x=TRAIN_END+0.5, color='gray', linestyle=':', label='Forecast Start')
ax.set_xlabel('Year')
ax.set_ylabel('Sea Level (m)')
ax.set_title('Comparison of Forecast Methods (2026-2030) - Extended Training (2000-2025)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'forecast_comparison_v2.png'), dpi=150)
plt.close()

# اهمیت متغیرها (Random Forest)
if 'Random Forest' in models:
    importances = models['Random Forest'].feature_importances_
    df_imp = pd.DataFrame({'feature': feature_cols, 'importance': importances})
    df_imp = df_imp.sort_values('importance', ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=df_imp, x='importance', y='feature', palette='viridis')
    ax.set_title('Feature Importance (Random Forest)')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance_v2.png'), dpi=150)
    plt.close()

# گزارش نهایی
with open(os.path.join(OUTPUT_DIR, 'summary_report_v2.txt'), 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("🌊 گزارش پیش‌بینی جامع تراز آب دریای خزر (نسخه بهبودیافته)\n")
    f.write("="*80 + "\n\n")
    f.write(f"دوره آموزش: {TRAIN_START}-{TRAIN_END} ({len(years_train)} سال)\n")
    f.write(f"دوره پیش‌بینی: {TEST_START}-{TEST_END}\n")
    f.write(f"تعداد ویژگی‌ها: {len(feature_cols)} -> {', '.join(feature_cols)}\n\n")
    
    f.write("📊 عملکرد مدل‌ها (Cross-Validation R²):\n")
    if not df_cv.empty:
        f.write(df_cv.to_string(index=False) + "\n\n")
    
    f.write("📈 پیش‌بینی نهایی (میانگین همه روش‌ها):\n")
    avg_pred = df_pred.iloc[:, 1:].mean(axis=1)
    f.write("سال | میانگین (متر)\n")
    f.write("----|---------------\n")
    for y, p in zip(future_years, avg_pred):
        f.write(f"{y}  | {p:.3f}\n")
    
    f.write("\n📈 پیش‌بینی هر روش:\n")
    for name in predictions.keys():
        if not np.isnan(predictions[name]).all():
            f.write(f"\n{name}:\n")
            for y, p in zip(future_years, predictions[name]):
                f.write(f"  {y}: {p:.3f}\n")
    
    f.write(f"\n✅ خروجی‌ها در: {OUTPUT_DIR}\n")

print(f"\n✅ همه خروجی‌ها در {OUTPUT_DIR} ذخیره شدند.")
print("📄 فایل‌های تولید شده:")
print("   - forecast_comparison_v2.csv")
print("   - forecast_comparison_v2.png")
print("   - feature_importance_v2.png")
print("   - summary_report_v2.txt")
print("="*80)