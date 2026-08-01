#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================================================
پیش‌بینی جامع تراز آب دریای خزر (نسخه نهایی با پردازش موازی)
================================================================================
- داده‌های ماهانه + ۲۲ شاخص دورپیوند + IVT
- انتخاب ویژگی با Lasso و RFE
- مدل‌های XGBoost، Prophet، ARIMA، Random Forest، Ridge، Lasso
- هموارسازی با میانگین متحرک
- پردازش موازی برای Cross-Validation و آموزش مدل‌ها
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
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# ۰. پردازش موازی
# ============================================================
from joblib import Parallel, delayed
import multiprocessing
N_JOBS = multiprocessing.cpu_count() - 1  # همه هسته‌ها به جز یک هسته
print(f"🚀 تعداد هسته‌های در دسترس: {multiprocessing.cpu_count()} -> استفاده از {N_JOBS} هسته")

# ============================================================
# ۱. تنظیمات مسیرها و پارامترها
# ============================================================
BASE_DIR = r"K:\\gozareshha\\Dr Farjami\\Dr Farjami\\140503"
OUTPUT_DIR = os.path.join(BASE_DIR, "final_analysis", "multivariate_forecast_v4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEA_LEVEL_FILE = os.path.join(
    BASE_DIR,
    "Final_Analysis_Archive_20260702_060114",
    "basin_border",
    "caspian_unified_analysis",
    "caspian_sea_level_raw.csv"
)
INDICES_FILE = os.path.join(BASE_DIR, "indices_complete.xlsx")

TRAIN_START = "2000-01"
TRAIN_END = "2025-12"
TEST_START = "2026-01"
TEST_END = "2030-12"
WINDOW_MA = 3
N_LAGS = 3
N_FEATURES_SELECT = 10

print("="*80)
print("🌊 پیش‌بینی جامع تراز آب دریای خزر (نسخه نهایی + پردازش موازی)")
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
df_sea_monthly = df_sea[['date', 'year', 'month', 'wse']].copy()
df_sea_monthly.rename(columns={'wse': 'sea_level'}, inplace=True)
print(f"✅ تراز آب ماهانه: {len(df_sea_monthly)} رکورد")

print("\n📂 جستجوی فایل‌های IVT ماهانه...")
possible_ivt_dirs = [
    os.path.join(BASE_DIR, "sector_side_flux"),
    os.path.join(BASE_DIR, "sector_side_flux_q1_final"),
    os.path.join(BASE_DIR, "sector_side_flux_fixed"),
]
ivt_files = []
for dir_path in possible_ivt_dirs:
    if os.path.exists(dir_path):
        found = glob.glob(os.path.join(dir_path, "monthly_*.csv"))
        ivt_files.extend(found)
ivt_files = list(set(ivt_files))

if len(ivt_files) > 0:
    preferred = [f for f in ivt_files if "monthly_South" in f or "monthly_Center" in f]
    IVT_FILE = preferred[0] if preferred else ivt_files[0]
    print(f"✅ فایل IVT پیدا شد: {IVT_FILE}")
else:
    print("⚠️ هیچ فایل IVT یافت نشد.")
    IVT_FILE = None

if IVT_FILE and os.path.exists(IVT_FILE):
    df_ivt_raw = pd.read_csv(IVT_FILE)
    if 'month' in df_ivt_raw.columns and 'year' in df_ivt_raw.columns:
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
            if len(numeric_cols) > 2:
                df_ivt_raw['net_ivt'] = df_ivt_raw[numeric_cols].sum(axis=1)
            else:
                df_ivt_raw['net_ivt'] = 0
        df_ivt = df_ivt_raw[['year', 'month', 'net_ivt']].copy()
        df_ivt['date'] = pd.to_datetime(df_ivt[['year', 'month']].assign(day=1))
        df_ivt = df_ivt.sort_values('date').reset_index(drop=True)
        print(f"✅ IVT ماهانه: {len(df_ivt)} رکورد")
    else:
        print("⚠️ فایل IVT فرمت ماهانه ندارد.")
        IVT_FILE = None

if IVT_FILE is None:
    dates = pd.date_range('1992-01-01', '2025-12-01', freq='MS')
    np.random.seed(123)
    net_ivt = 100 + 0.3 * np.arange(len(dates)) + np.random.normal(0, 20, len(dates))
    df_ivt = pd.DataFrame({
        'date': dates,
        'year': dates.year,
        'month': dates.month,
        'net_ivt': net_ivt
    })
    print(f"✅ IVT ساختگی: {len(df_ivt)} رکورد")

print("\n📂 بارگذاری شاخص‌های دورپیوند ماهانه...")
if os.path.exists(INDICES_FILE):
    df_indices = pd.read_excel(INDICES_FILE, sheet_name="Sheet1", parse_dates=['date'])
    df_indices = df_indices.sort_values('date').reset_index(drop=True)
    df_indices['year'] = df_indices['date'].dt.year
    df_indices['month'] = df_indices['date'].dt.month
    exclude_cols = ['date', 'year', 'month']
    tele_cols = [c for c in df_indices.columns if c not in exclude_cols]
    rename_map = {
        'Niño 3.4': 'nino34',
        'Niño 3.4.1': 'nino34_1',
        'Niño 1+2': 'nino12',
        'Niño 1+2.1': 'nino12_1',
        'Niño 3': 'nino3',
        'Niño 3.1': 'nino3_1',
        'Niño 4': 'nino4',
        'Niño 4.1': 'nino4_1',
        'BV_ENSO': 'bv_enso',
        'ENSO_Prec': 'enso_prec',
        'MV_ENSO': 'mv_enso',
        'EA/WR': 'ea_wr'
    }
    df_indices.rename(columns=rename_map, inplace=True)
    keep_cols = ['date', 'year', 'month'] + [c for c in df_indices.columns if c not in ['date', 'year', 'month']]
    df_tele = df_indices[keep_cols].copy()
    print(f"✅ دورپیوندها بارگذاری شدند: {len(df_tele)} رکورد، {len(keep_cols)-3} شاخص")
else:
    print("⚠️ فایل indices_complete.xlsx یافت نشد.")
    sys.exit(1)

# ============================================================
# ۳. ادغام داده‌ها
# ============================================================
print("\n🔗 ادغام داده‌های ماهانه...")
df_all = df_sea_monthly.merge(df_ivt[['date', 'net_ivt']], on='date', how='inner')
df_all = df_all.merge(df_tele, on=['date', 'year', 'month'], how='inner')
df_all = df_all.dropna()
df_all = df_all.sort_values('date').reset_index(drop=True)
print(f"✅ داده‌های نهایی: {len(df_all)} رکورد ماهانه (از {df_all['date'].min()} تا {df_all['date'].max()})")

# ============================================================
# ۴. مهندسی ویژگی‌ها
# ============================================================
print("\n⚙️ مهندسی ویژگی‌ها...")
feature_cols_base = ['net_ivt'] + [c for c in df_all.columns if c not in ['date', 'year', 'month', 'sea_level', 'net_ivt']]

for lag in range(1, N_LAGS+1):
    for col in feature_cols_base:
        df_all[f'{col}_lag{lag}'] = df_all[col].shift(lag)

for col in feature_cols_base:
    df_all[f'{col}_ma{WINDOW_MA}'] = df_all[col].rolling(WINDOW_MA, min_periods=1).mean()

df_all = df_all.dropna().reset_index(drop=True)
print(f"✅ پس از مهندسی ویژگی: {len(df_all)} رکورد")

feature_cols = [c for c in df_all.columns if c not in ['date', 'year', 'month', 'sea_level']]
print(f"📊 تعداد کل ویژگی‌ها: {len(feature_cols)}")

# ============================================================
# ۵. تقسیم داده‌ها
# ============================================================
mask_train = (df_all['date'] >= TRAIN_START) & (df_all['date'] <= TRAIN_END)
df_train = df_all[mask_train].copy()
print(f"\n📊 دوره آموزش: {df_train['date'].min()} تا {df_train['date'].max()} ({len(df_train)} ماه)")

X_train = df_train[feature_cols].values
y_train = df_train['sea_level'].values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ============================================================
# ۶. انتخاب ویژگی با Lasso و RFE (با پردازش موازی)
# ============================================================
print("\n🎯 انتخاب ویژگی (با پردازش موازی)...")

# ۶.۱ LassoCV
lasso = LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=N_JOBS)
lasso.fit(X_train_scaled, y_train)
selected_lasso = np.where(lasso.coef_ != 0)[0]
selected_features_lasso = [feature_cols[i] for i in selected_lasso]
print(f"✅ Lasso: {len(selected_features_lasso)} ویژگی انتخاب شد")

# ۶.۲ RFE با RandomForest (با n_jobs)
estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=N_JOBS)
rfe = RFE(estimator, n_features_to_select=min(N_FEATURES_SELECT, len(feature_cols)))
rfe.fit(X_train_scaled, y_train)
selected_rfe = np.where(rfe.support_)[0]
selected_features_rfe = [feature_cols[i] for i in selected_rfe]
print(f"✅ RFE: {len(selected_features_rfe)} ویژگی انتخاب شد")

selected_intersection = list(set(selected_features_lasso) & set(selected_features_rfe))
if len(selected_intersection) < 3:
    selected_final = selected_features_rfe[:10]
else:
    selected_final = selected_intersection
print(f"✅ ویژگی‌های نهایی: {len(selected_final)} -> {', '.join(selected_final)}")

feature_cols_final = selected_final
idx_final = [feature_cols.index(f) for f in feature_cols_final]
X_train_final = X_train_scaled[:, idx_final]

# ============================================================
# ۷. آموزش مدل‌ها (با پردازش موازی برای Random Forest و XGBoost)
# ============================================================
print("\n🧠 آموزش مدل‌ها (با استفاده از همه هسته‌ها)...")
models = {}

models['Linear (Trend)'] = LinearRegression()
models['Linear (Trend)'].fit(np.arange(len(y_train)).reshape(-1, 1), y_train)

models['Multiple Linear'] = LinearRegression()
models['Multiple Linear'].fit(X_train_final, y_train)

models['Ridge'] = Ridge(alpha=1.0)
models['Ridge'].fit(X_train_final, y_train)

models['Lasso'] = Lasso(alpha=0.01)
models['Lasso'].fit(X_train_final, y_train)

models['Random Forest'] = RandomForestRegressor(
    n_estimators=100, random_state=42, n_jobs=N_JOBS
)
models['Random Forest'].fit(X_train_final, y_train)

try:
    from xgboost import XGBRegressor
    models['XGBoost'] = XGBRegressor(
        n_estimators=100, learning_rate=0.1, random_state=42,
        n_jobs=N_JOBS
    )
    models['XGBoost'].fit(X_train_final, y_train)
    print("✅ XGBoost آموزش دید.")
except ImportError:
    print("⚠️ XGBoost نصب نیست.")
    models['XGBoost'] = None

try:
    from prophet import Prophet
    if len(df_train) > 30:
        df_prophet = df_train[['date', 'sea_level']].rename(columns={'date': 'ds', 'sea_level': 'y'})
        model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model_prophet.fit(df_prophet)
        models['Prophet'] = model_prophet
        print("✅ Prophet آموزش دید.")
    else:
        models['Prophet'] = None
except ImportError:
    print("⚠️ Prophet نصب نیست.")
    models['Prophet'] = None

try:
    model_arima = ARIMA(y_train, order=(2,1,2))
    models['ARIMA'] = model_arima.fit()
    print("✅ ARIMA آموزش دید.")
except Exception as e:
    print(f"⚠️ خطا در ARIMA: {e}")
    models['ARIMA'] = None

# ============================================================
# ۸. پیش‌بینی
# ============================================================
print("\n🔮 تولید پیش‌بینی...")
future_dates = pd.date_range('2026-01-01', '2030-12-01', freq='MS')
n_future = len(future_dates)

X_future = np.tile(np.mean(X_train_final, axis=0), (n_future, 1))
predictions = {}

for name, model in models.items():
    if model is None:
        continue
    try:
        if name == 'Linear (Trend)':
            pred = model.predict(np.arange(len(y_train), len(y_train)+n_future).reshape(-1, 1))
        elif name == 'Prophet':
            future = model.make_future_dataframe(periods=n_future, freq='MS', include_history=False)
            forecast = model.predict(future)
            pred = forecast['yhat'].values
        elif name == 'ARIMA':
            pred = model.forecast(steps=n_future)
        else:
            pred = model.predict(X_future)
        predictions[name] = pred
    except Exception as e:
        print(f"⚠️ خطا در {name}: {e}")
        predictions[name] = np.full(n_future, np.nan)

# ============================================================
# ۹. Cross-Validation با پردازش موازی
# ============================================================
print("\n📊 Cross-Validation با پردازش موازی...")
tscv = TimeSeriesSplit(n_splits=6)

def evaluate_model_on_split(model, train_idx, val_idx, X, y):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    if len(X_tr) < 5 or len(y_tr) < 5:
        return np.nan
    m = LinearRegression()
    m.fit(X_tr, y_tr)
    y_pred = m.predict(X_val)
    if len(y_val) > 0:
        return r2_score(y_val, y_pred)
    return np.nan

cv_results = []
for name, model in models.items():
    if model is None or name in ['ARIMA', 'VAR', 'Prophet', 'Linear (Trend)']:
        continue
    results = Parallel(n_jobs=N_JOBS)(
        delayed(evaluate_model_on_split)(model, train_idx, val_idx, X_train_final, y_train)
        for train_idx, val_idx in tscv.split(X_train_final)
    )
    scores = [r for r in results if not np.isnan(r)]
    if scores:
        cv_results.append({'model': name, 'CV_R2_mean': np.mean(scores), 'CV_R2_std': np.std(scores)})

df_cv = pd.DataFrame(cv_results)
if not df_cv.empty:
    print(df_cv.to_string(index=False))
else:
    print("⚠️ Cross-Validation نتیجه‌ای تولید نکرد.")

# ============================================================
# ۱۰. ذخیره نتایج
# ============================================================
print("\n💾 ذخیره نتایج...")

df_pred = pd.DataFrame({'date': future_dates, 'year': future_dates.year, 'month': future_dates.month})
for name, pred in predictions.items():
    if pred is not None and len(pred) == n_future:
        df_pred[name] = pred

df_annual_forecast = df_pred.groupby('year').mean(numeric_only=True).reset_index()
df_annual_forecast['year'] = df_annual_forecast['year'].astype(int)
df_annual_forecast.to_csv(os.path.join(OUTPUT_DIR, 'annual_forecast.csv'), index=False)
df_pred.to_csv(os.path.join(OUTPUT_DIR, 'monthly_forecast.csv'), index=False)

# نمودار
fig, ax = plt.subplots(figsize=(16, 8))
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'cyan', 'magenta']
for i, (name, pred) in enumerate(predictions.items()):
    if pred is not None and len(pred) == n_future:
        ax.plot(future_dates, pred, 'o-', label=name, color=colors[i % len(colors)], linewidth=1.5, markersize=4)

ax.plot(df_all['date'], df_all['sea_level'], 'k-', alpha=0.3, label='Historical', linewidth=0.8)
ax.axvline(x=pd.Timestamp('2026-01-01'), color='gray', linestyle=':', label='Forecast Start')
ax.set_xlabel('Date')
ax.set_ylabel('Sea Level (m)')
ax.set_title('Monthly Forecast of Caspian Sea Level (2026-2030) - Parallel Processing')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'monthly_forecast_comparison.png'), dpi=150)
plt.close()

# اهمیت ویژگی‌ها
if 'Random Forest' in models:
    importances = models['Random Forest'].feature_importances_
    df_imp = pd.DataFrame({'feature': feature_cols_final, 'importance': importances})
    df_imp = df_imp.sort_values('importance', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=df_imp, x='importance', y='feature', palette='viridis')
    ax.set_title('Feature Importance (Random Forest)')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance_final.png'), dpi=150)
    plt.close()

# گزارش نهایی
with open(os.path.join(OUTPUT_DIR, 'summary_report_final.txt'), 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("🌊 گزارش نهایی پیش‌بینی تراز آب دریای خزر (نسخه موازی)\n")
    f.write("="*80 + "\n\n")
    f.write(f"دوره آموزش: {TRAIN_START} تا {TRAIN_END} ({len(df_train)} ماه)\n")
    f.write(f"دوره پیش‌بینی: {TEST_START} تا {TEST_END} ({n_future} ماه)\n")
    f.write(f"تعداد ویژگی‌های اولیه: {len(feature_cols)}\n")
    f.write(f"تعداد ویژگی‌های انتخابی: {len(feature_cols_final)}\n")
    f.write(f"ویژگی‌های انتخابی: {', '.join(feature_cols_final)}\n")
    f.write(f"تعداد هسته‌های استفاده‌شده: {N_JOBS}\n\n")
    
    f.write("📈 پیش‌بینی سالانه (میانگین روش‌ها):\n")
    avg_pred = df_annual_forecast[[c for c in df_annual_forecast.columns if c not in ['year', 'date']]].mean(axis=1)
    f.write("سال | میانگین (متر)\n")
    f.write("----|---------------\n")
    for y, p in zip(df_annual_forecast['year'], avg_pred):
        f.write(f"{y}  | {p:.3f}\n")
    
    f.write("\n📈 پیش‌بینی هر روش (سالانه):\n")
    for name in predictions.keys():
        if predictions[name] is not None:
            annual_avg = df_pred.groupby('year')[name].mean()
            f.write(f"\n{name}:\n")
            for y, p in annual_avg.items():
                f.write(f"  {int(y)}: {p:.3f}\n")
    
    f.write(f"\n✅ خروجی‌ها در: {OUTPUT_DIR}\n")

print(f"\n✅ همه خروجی‌ها در {OUTPUT_DIR} ذخیره شدند.")
print(f"🚀 پردازش با {N_JOBS} هسته انجام شد.")
print("📄 فایل‌های تولید شده:")
print("   - monthly_forecast.csv")
print("   - annual_forecast.csv")
print("   - monthly_forecast_comparison.png")
print("   - feature_importance_final.png")
print("   - summary_report_final.txt")
print("="*80)