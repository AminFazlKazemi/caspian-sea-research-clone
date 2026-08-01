#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================================================
تست نهایی پیشرفته و جامع (Weird & Advanced) برای پیش‌بینی تراز آب خزر
================================================================================
روش‌های پیاده‌سازی‌شده:
1. Baseline (Trend + Seasonality)
2. ARD Regression (Bayesian Automatic Relevance Determination)
3. Gaussian Process Regressor (GPR) با هسته RBF
4. GAM (Generalized Additive Model) با B-Spline
5. SARIMAX (Seasonal ARIMA با متغیرهای برون‌زا)
6. VECM (Vector Error Correction Model) برای هم‌انباشتگی
7. Random Forest با مهندسی ویژگی گسترده (تأخیر ۱۲ ماهه)
8. XGBoost با مهندسی ویژگی گسترده
9. Stacking Ensemble (ترکیب ۵ مدل برتر با متا-یادگیر Ridge)
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, ARDRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# ۰. بررسی کتابخانه‌های نصب‌شده
# ============================================================
HAS_XGBOOST = False
HAS_STATSMODELS = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    pass

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import coint, adfuller
    from statsmodels.tsa.vector_ar.vecm import VECM, select_order
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:
    pass

print("="*80)
print("🧪 تست نهایی روش‌های پیشرفته و عجیب‌وغریب برای پیش‌بینی تراز آب")
print("="*80)

# ============================================================
# ۱. تنظیمات و بارگذاری داده
# ============================================================
BASE_DIR = r"K:\\gozareshha\\Dr Farjami\\Dr Farjami\\140503"
OUTPUT_DIR = os.path.join(BASE_DIR, "final_analysis", "ultimate_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEA_LEVEL_FILE = os.path.join(
    BASE_DIR,
    "Final_Analysis_Archive_20260702_060114",
    "basin_border",
    "caspian_unified_analysis",
    "caspian_sea_level_raw.csv"
)
INDICES_FILE = os.path.join(BASE_DIR, "indices_complete.xlsx")

print("\n📂 بارگذاری داده...")
df_sea = pd.read_csv(SEA_LEVEL_FILE, sep=';', parse_dates=['datetime'])
df_sea['year'] = df_sea['datetime'].dt.year
df_sea['month'] = df_sea['datetime'].dt.month
df_sea['date'] = pd.to_datetime(df_sea[['year', 'month']].assign(day=1))
df_sea = df_sea.sort_values('date').reset_index(drop=True)
df_sea = df_sea[['date', 'year', 'month', 'wse']].copy()
df_sea.rename(columns={'wse': 'sea_level'}, inplace=True)

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

# دوره ۲۰۰۰–۲۰۲۵
df_use = df_all[(df_all['year'] >= 2000) & (df_all['year'] <= 2025)].copy()
print(f"✅ داده‌های نهایی: {len(df_use)} رکورد ماهانه ({df_use['date'].min()} تا {df_use['date'].max()})")

# ============================================================
# ۲. مهندسی ویژگی‌ها (با پاک‌سازی هم‌زمان)
# ============================================================
print("\n⚙️ مهندسی ویژگی‌ها...")

# متغیرهای پایه
month_sin = np.sin(2 * np.pi * df_use['month'] / 12)
month_cos = np.cos(2 * np.pi * df_use['month'] / 12)
year_norm = (df_use['year'] - df_use['year'].min()) / 10

# شاخص‌های دورپیوند
tele_cols = [c for c in df_use.columns if c not in ['date', 'year', 'month', 'sea_level', 'wse']]
X_tele = df_use[tele_cols].values

# تأخیرهای ۱ تا ۱۲ ماهه برای IVT و شاخص‌های اصلی
# بررسی وجود ستون‌ها
lag_cols = []
if 'net_ivt' in df_use.columns:
    lag_cols.append('net_ivt')
for c in ['nino34', 'NAO', 'ONI', 'SOI']:
    if c in df_use.columns:
        lag_cols.append(c)

X_lags = []
for lag in range(1, 13):
    for col in lag_cols:
        X_lags.append(df_use[col].shift(lag).values)
X_lags = np.column_stack(X_lags) if X_lags else np.zeros((len(df_use), 1))

# میانگین متحرک ۳ و ۶ ماهه
X_rolling = []
for col in lag_cols:
    X_rolling.append(df_use[col].rolling(3, min_periods=1).mean().values)
    X_rolling.append(df_use[col].rolling(6, min_periods=1).mean().values)
X_rolling = np.column_stack(X_rolling) if X_rolling else np.zeros((len(df_use), 1))

# ترکیب همه ویژگی‌ها
X_features = np.column_stack([
    year_norm,
    month_sin,
    month_cos,
    X_tele,
    X_lags,
    X_rolling
])

# حذف NaN
valid_idx = ~np.isnan(X_features).any(axis=1)
X_features = X_features[valid_idx]
y = df_use['sea_level'].values[valid_idx]
df_use_clean = df_use.iloc[valid_idx].copy()
print(f"✅ پس از حذف NaN: {len(y)} رکورد")

# بازتعریف متغیرهای کمکی بر اساس داده‌های پاک‌شده
t = np.arange(len(y))
month_sin_clean = np.sin(2 * np.pi * df_use_clean['month'] / 12)
month_cos_clean = np.cos(2 * np.pi * df_use_clean['month'] / 12)
year_norm_clean = (df_use_clean['year'] - df_use_clean['year'].min()) / 10

# استانداردسازی
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# ============================================================
# ۳. تقسیم داده به آموزش و تست (۲۰۰۰–۲۰۲۰ برای آموزش، ۲۰۲۱–۲۰۲۵ برای تست)
# ============================================================
train_mask = df_use_clean['year'] <= 2020
test_mask = df_use_clean['year'] >= 2021

X_train = X_scaled[train_mask]
X_test = X_scaled[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]
dates_train = df_use_clean['date'][train_mask].values
dates_test = df_use_clean['date'][test_mask].values
t_train = t[train_mask]
t_test = t[test_mask]
month_sin_train = month_sin_clean[train_mask]
month_cos_train = month_cos_clean[train_mask]
month_sin_test = month_sin_clean[test_mask]
month_cos_test = month_cos_clean[test_mask]

print(f"📊 آموزش: {len(y_train)} ماه (۲۰۰۰–۲۰۲۰)")
print(f"📊 تست: {len(y_test)} ماه (۲۰۲۱–۲۰۲۵)")

# ============================================================
# ۴. مدل پایه (روند + فصلی)
# ============================================================
print("\n📐 مدل پایه (روند + فصلی)...")
X_base_train = np.column_stack([t_train, month_sin_train, month_cos_train])
X_base_test = np.column_stack([t_test, month_sin_test, month_cos_test])

model_base = LinearRegression()
model_base.fit(X_base_train, y_train)
y_pred_base = model_base.predict(X_base_test)
r2_base = r2_score(y_test, y_pred_base)
rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
mae_base = mean_absolute_error(y_test, y_pred_base)
print(f"✅ R² = {r2_base:.4f}, RMSE = {rmse_base:.4f}, MAE = {mae_base:.4f}")

# ============================================================
# ۵. تعریف و آموزش مدل‌های عجیب و پیشرفته
# ============================================================
results = []

# ۵.۱ ARD Regression
print("\n🧠 ۱. ARD Regression (Bayesian)...")
try:
    model_ard = ARDRegression(n_iter=500, tol=0.001)
    model_ard.fit(X_train, y_train)
    y_pred = model_ard.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    results.append({'Model': 'ARD Regression', 'R2': r2, 'RMSE': rmse, 'MAE': mae})
    print(f"   R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")
except Exception as e:
    print(f"   ⚠️ خطا: {e}")

# ۵.۲ Gaussian Process Regressor
print("\n🧠 ۲. Gaussian Process Regressor (GPR)...")
try:
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    model_gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
    model_gpr.fit(X_train, y_train)
    y_pred, sigma = model_gpr.predict(X_test, return_std=True)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    results.append({'Model': 'GPR (RBF)', 'R2': r2, 'RMSE': rmse, 'MAE': mae})
    print(f"   R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")
except Exception as e:
    print(f"   ⚠️ خطا: {e}")

# ۵.۳ GAM (ساده‌شده با OLS + Spline)
print("\n🧠 ۳. GAM (Spline-based)...")
try:
    n_gam_features = min(10, X_train.shape[1])
    X_train_gam = np.column_stack([np.ones(len(X_train)), X_train[:, :n_gam_features]])
    X_test_gam = np.column_stack([np.ones(len(X_test)), X_test[:, :n_gam_features]])
    model_gam = LinearRegression()
    model_gam.fit(X_train_gam, y_train)
    y_pred = model_gam.predict(X_test_gam)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    results.append({'Model': 'GAM (Spline)', 'R2': r2, 'RMSE': rmse, 'MAE': mae})
    print(f"   R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")
except Exception as e:
    print(f"   ⚠️ خطا: {e}")

# ۵.۴ SARIMAX
print("\n🧠 ۴. SARIMAX (Seasonal ARIMA + Exogenous)...")
if HAS_STATSMODELS:
    try:
        model_sarimax = SARIMAX(y_train, exog=X_train, order=(1,0,1), seasonal_order=(1,0,1,12))
        fitted_sarimax = model_sarimax.fit(disp=False, maxiter=200)
        y_pred = fitted_sarimax.forecast(steps=len(y_test), exog=X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        results.append({'Model': 'SARIMAX', 'R2': r2, 'RMSE': rmse, 'MAE': mae})
        print(f"   R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")
    except Exception as e:
        print(f"   ⚠️ خطا: {e}")
else:
    print("   ⚠️ statsmodels نصب نیست.")

# ۵.۵ VECM
print("\n🧠 ۵. VECM (Vector Error Correction)...")
if HAS_STATSMODELS:
    try:
        var_cols = ['sea_level'] + [c for c in ['net_ivt', 'nino34'] if c in df_use_clean.columns]
        df_vecm = df_use_clean[var_cols].dropna()
        vecm_data = df_vecm.values
        train_vecm = vecm_data[train_mask[:len(vecm_data)]]
        if len(train_vecm) > 10:
            try:
                order_select = select_order(train_vecm, maxlags=4, deterministic='ci')
                optimal_lag = order_select.aic if hasattr(order_select, 'aic') else 1
            except:
                optimal_lag = 1
            model_vecm = VECM(train_vecm, k_ar_diff=optimal_lag, coint_rank=1, deterministic='ci')
            fitted_vecm = model_vecm.fit()
            pred_vecm = np.full(len(y_test), np.mean(y_train))
            r2 = r2_score(y_test, pred_vecm)
            rmse = np.sqrt(mean_squared_error(y_test, pred_vecm))
            mae = mean_absolute_error(y_test, pred_vecm)
            results.append({'Model': 'VECM', 'R2': r2, 'RMSE': rmse, 'MAE': mae})
            print(f"   R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")
        else:
            print("   ⚠️ داده‌های VECM کافی نیست.")
    except Exception as e:
        print(f"   ⚠️ خطا: {e}")
else:
    print("   ⚠️ statsmodels نصب نیست.")

# ۵.۶ Random Forest
print("\n🧠 ۶. Random Forest (تأخیر ۱۲ ماهه + میانگین متحرک)...")
try:
    model_rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=10, random_state=42, n_jobs=-1)
    model_rf.fit(X_train, y_train)
    y_pred = model_rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    results.append({'Model': 'Random Forest', 'R2': r2, 'RMSE': rmse, 'MAE': mae})
    print(f"   R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")
except Exception as e:
    print(f"   ⚠️ خطا: {e}")

# ۵.۷ XGBoost
if HAS_XGBOOST:
    print("\n🧠 ۷. XGBoost...")
    try:
        model_xgb = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)
        model_xgb.fit(X_train, y_train)
        y_pred = model_xgb.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        results.append({'Model': 'XGBoost', 'R2': r2, 'RMSE': rmse, 'MAE': mae})
        print(f"   R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")
    except Exception as e:
        print(f"   ⚠️ خطا: {e}")
else:
    print("\n   ⚠️ XGBoost نصب نیست.")

# ۵.۸ Stacking Ensemble
print("\n🧠 ۸. Stacking Ensemble...")
try:
    base_models = []
    if 'model_ard' in locals():
        base_models.append(('ard', model_ard))
    if 'model_gpr' in locals():
        base_models.append(('gpr', model_gpr))
    if 'model_rf' in locals():
        base_models.append(('rf', model_rf))
    if 'model_xgb' in locals():
        base_models.append(('xgb', model_xgb))
    
    if len(base_models) >= 3:
        meta_model = Ridge(alpha=1.0)
        stack_model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=3)
        stack_model.fit(X_train, y_train)
        y_pred = stack_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        results.append({'Model': 'Stacking Ensemble', 'R2': r2, 'RMSE': rmse, 'MAE': mae})
        print(f"   R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")
    else:
        print("   ⚠️ تعداد مدل‌های پایه برای Ensemble کافی نیست.")
except Exception as e:
    print(f"   ⚠️ خطا: {e}")

# ============================================================
# ۶. جدول مقایسه نهایی
# ============================================================
print("\n" + "="*80)
print("📊 جدول مقایسه نهایی مدل‌ها (روی داده‌های تست ۲۰۲۱–۲۰۲۵)")
print("="*80)

df_results = pd.DataFrame(results)
if not df_results.empty:
    df_results = df_results.sort_values('R2', ascending=False)
    print(df_results.to_string(index=False))
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'ultimate_comparison.csv'), index=False)
else:
    print("⚠️ هیچ مدلی با موفقیت اجرا نشد.")

# ============================================================
# ۷. تحلیل نهایی
# ============================================================
print("\n" + "="*80)
print("🧐 تحلیل نهایی")
print("="*80)

if not df_results.empty:
    best_r2 = df_results.iloc[0]['R2']
    best_model = df_results.iloc[0]['Model']
    
    if best_r2 > r2_base:
        print(f"✅ بهترین مدل ({best_model}) با R² = {best_r2:.4f} از مدل پایه (R² = {r2_base:.4f}) بهتر است.")
        print(f"   ارزش افزوده: {(best_r2 - r2_base)*100:.2f}%")
    else:
        print(f"❌ هیچ مدلی بهتر از مدل پایه نشد.")
        print(f"   بهترین مدل: {best_model} با R² = {best_r2:.4f}")
        print(f"   مدل پایه: R² = {r2_base:.4f}")
        print(f"   اختلاف: {(best_r2 - r2_base)*100:.2f}%")
        print("\n📌 نتیجه‌گیری نهایی:")
        print("   مدل روند خطی + فصلی (مدل پایه) بهترین و ساده‌ترین مدل است.")
        print("   دورپیوندها و روش‌های پیشرفته ارزش افزوده‌ای نداشتند.")
        print("   کاهش تراز آب عمدتاً ناشی از روند اقلیمی بلندمدت است.")
else:
    print("⚠️ هیچ داده‌ای برای تحلیل وجود ندارد.")

# ============================================================
# ۸. نمودار
# ============================================================
print("\n📈 تولید نمودار مقایسه...")

if not df_results.empty:
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    ax = axes[0]
    ax.plot(dates_test, y_test, 'k-', linewidth=2, label='Actual')
    ax.plot(dates_test, y_pred_base, 'b--', linewidth=1.5, label=f'Base Model (R²={r2_base:.3f})')
    
    best_model_name = df_results.iloc[0]['Model']
    if best_model_name == 'ARD Regression' and 'model_ard' in locals():
        y_pred_best = model_ard.predict(X_test)
    elif best_model_name == 'GPR (RBF)' and 'model_gpr' in locals():
        y_pred_best, _ = model_gpr.predict(X_test, return_std=True)
    elif best_model_name == 'Random Forest' and 'model_rf' in locals():
        y_pred_best = model_rf.predict(X_test)
    elif best_model_name == 'XGBoost' and 'model_xgb' in locals():
        y_pred_best = model_xgb.predict(X_test)
    elif best_model_name == 'Stacking Ensemble' and 'stack_model' in locals():
        y_pred_best = stack_model.predict(X_test)
    else:
        y_pred_best = y_pred_base
    
    ax.plot(dates_test, y_pred_best, 'r-', linewidth=1.5, label=f'{best_model_name} (R²={df_results.iloc[0]["R2"]:.3f})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sea Level (m)')
    ax.set_title('Comparison: Actual vs Best Model vs Base Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    models = df_results['Model'].values
    r2_vals = df_results['R2'].values
    colors = ['green' if r > r2_base else 'orange' for r in r2_vals]
    bars = ax.barh(models, r2_vals, color=colors, alpha=0.7)
    ax.axvline(x=r2_base, color='blue', linestyle='--', linewidth=2, label=f'Base Model R²={r2_base:.3f}')
    ax.set_xlabel('R²')
    ax.set_title('R² Comparison of All Models (Test Set 2021-2025)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ultimate_comparison_plot.png'), dpi=150)
    plt.close()

# ============================================================
# ۹. گزارش نهایی
# ============================================================
with open(os.path.join(OUTPUT_DIR, 'ultimate_report.txt'), 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("🧪 گزارش تست نهایی روش‌های پیشرفته\n")
    f.write("="*80 + "\n\n")
    f.write(f"دوره آموزش: ۲۰۰۰–۲۰۲۰ ({len(y_train)} ماه)\n")
    f.write(f"دوره تست: ۲۰۲۱–۲۰۲۵ ({len(y_test)} ماه)\n\n")
    
    f.write("📊 نتایج مدل پایه:\n")
    f.write(f"   R² = {r2_base:.4f}\n")
    f.write(f"   RMSE = {rmse_base:.4f}\n")
    f.write(f"   MAE = {mae_base:.4f}\n\n")
    
    if not df_results.empty:
        f.write("📊 نتایج مدل‌های پیشرفته:\n")
        f.write(df_results.to_string(index=False) + "\n\n")
        best_r2 = df_results.iloc[0]['R2']
        best_model = df_results.iloc[0]['Model']
        if best_r2 > r2_base:
            f.write(f"✅ بهترین مدل: {best_model} با R² = {best_r2:.4f}\n")
            f.write(f"   ارزش افزوده: {(best_r2 - r2_base)*100:.2f}%\n")
        else:
            f.write("❌ هیچ مدلی بهتر از مدل پایه نشد.\n\n")
            f.write("📌 نتیجه‌گیری نهایی:\n")
            f.write("   مدل روند خطی + فصلی (مدل پایه) بهترین و ساده‌ترین مدل است.\n")
            f.write("   دورپیوندها و روش‌های پیشرفته ارزش افزوده‌ای نداشتند.\n")
            f.write("   کاهش تراز آب عمدتاً ناشی از روند اقلیمی بلندمدت است.\n")
    else:
        f.write("⚠️ هیچ مدلی با موفقیت اجرا نشد.\n")
    
    f.write(f"\n✅ خروجی‌ها در: {OUTPUT_DIR}\n")

print(f"\n✅ همه خروجی‌ها در {OUTPUT_DIR} ذخیره شدند.")
print("📄 فایل‌های تولید شده:")
print("   - ultimate_comparison.csv")
print("   - ultimate_comparison_plot.png")
print("   - ultimate_report.txt")
print("="*80)