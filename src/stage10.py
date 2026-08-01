#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================================================
پیش‌بینی جامع تراز آب دریای خزر (نسخه نهایی با تمام بهبودها)
================================================================================
- داده‌های ماهانه (به‌جای سالانه)
- ۲۲ شاخص دورپیوند + IVT
- انتخاب ویژگی با Lasso و RFE
- مدل‌های XGBoost و Prophet
- هموارسازی با میانگین متحرک
- پیش‌بینی ماهانه تا ۲۰۳۰ و تبدیل به سالانه
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
# ۱. تنظیمات مسیرها و پارامترها
# ============================================================
BASE_DIR = r"K:\\gozareshha\\Dr Farjami\\Dr Farjami\\140503"
OUTPUT_DIR = os.path.join(BASE_DIR, "final_analysis", "multivariate_forecast_v3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# مسیرها
SEA_LEVEL_FILE = os.path.join(
    BASE_DIR,
    "Final_Analysis_Archive_20260702_060114",
    "basin_border",
    "caspian_unified_analysis",
    "caspian_sea_level_raw.csv"
)
INDICES_FILE = os.path.join(BASE_DIR, "indices_complete.xlsx")

# پارامترها
TRAIN_START = "2000-01"
TRAIN_END = "2025-12"
TEST_START = "2026-01"
TEST_END = "2030-12"
WINDOW_MA = 3          # میانگین متحرک ۳ ماهه
N_LAGS = 3             # تعداد تأخیرها
N_FEATURES_SELECT = 10 # تعداد ویژگی‌های انتخابی

print("="*80)
print("🌊 پیش‌بینی جامع تراز آب دریای خزر (نسخه نهایی)")
print("="*80)

# ============================================================
# ۲. بارگذاری داده‌های ماهانه
# ============================================================
print("\n📂 بارگذاری داده‌های ماهانه...")

# ۲.۱ تراز آب ماهانه
if not os.path.exists(SEA_LEVEL_FILE):
    raise FileNotFoundError(f"فایل تراز آب یافت نشد: {SEA_LEVEL_FILE}")

df_sea = pd.read_csv(SEA_LEVEL_FILE, sep=';', parse_dates=['datetime'])
df_sea['year'] = df_sea['datetime'].dt.year
df_sea['month'] = df_sea['datetime'].dt.month
df_sea['date'] = pd.to_datetime(df_sea[['year', 'month']].assign(day=1))
df_sea = df_sea.sort_values('date').reset_index(drop=True)
df_sea_monthly = df_sea[['date', 'year', 'month', 'wse']].copy()
df_sea_monthly.rename(columns={'wse': 'sea_level'}, inplace=True)
print(f"✅ تراز آب ماهانه: {len(df_sea_monthly)} رکورد (۱۹۹۲-۲۰۲۵)")

# ۲.۲ IVT ماهانه (جستجوی خودکار)
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
    print("⚠️ هیچ فایل IVT یافت نشد. از داده‌های ساختگی استفاده می‌شود.")
    IVT_FILE = None

# بارگذاری IVT ماهانه
if IVT_FILE and os.path.exists(IVT_FILE):
    df_ivt_raw = pd.read_csv(IVT_FILE)
    if 'month' in df_ivt_raw.columns and 'year' in df_ivt_raw.columns:
        # محاسبه net_ivt از ستون‌های inflow/outflow
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
        print("⚠️ فایل IVT فرمت ماهانه ندارد. از داده‌های ساختگی استفاده می‌شود.")
        IVT_FILE = None

if IVT_FILE is None:
    # داده‌های ساختگی
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

# ۲.۳ دورپیوندها ماهانه (از فایل Excel)
print("\n📂 بارگذاری شاخص‌های دورپیوند ماهانه...")
if os.path.exists(INDICES_FILE):
    df_indices = pd.read_excel(INDICES_FILE, sheet_name="Sheet1", parse_dates=['date'])
    df_indices = df_indices.sort_values('date').reset_index(drop=True)
    
    # استخراج سال و ماه
    df_indices['year'] = df_indices['date'].dt.year
    df_indices['month'] = df_indices['date'].dt.month
    
    # انتخاب همه شاخص‌ها به جز ستون‌های تاریخ و شناسه
    exclude_cols = ['date', 'year', 'month']
    tele_cols = [c for c in df_indices.columns if c not in exclude_cols]
    
    # تغییر نام برای سادگی
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
    
    # نگه‌داری ستون‌های اصلی
    keep_cols = ['date', 'year', 'month'] + [c for c in df_indices.columns if c not in ['date', 'year', 'month']]
    df_tele = df_indices[keep_cols].copy()
    print(f"✅ دورپیوندها بارگذاری شدند: {len(df_tele)} رکورد، {len(keep_cols)-3} شاخص")
else:
    print("⚠️ فایل indices_complete.xlsx یافت نشد. اجرا متوقف می‌شود.")
    sys.exit(1)

# ============================================================
# ۳. ادغام داده‌ها در سطح ماهانه
# ============================================================
print("\n🔗 ادغام داده‌های ماهانه...")
df_all = df_sea_monthly.merge(df_ivt[['date', 'net_ivt']], on='date', how='inner')
df_all = df_all.merge(df_tele, on=['date', 'year', 'month'], how='inner')
df_all = df_all.dropna()
df_all = df_all.sort_values('date').reset_index(drop=True)

print(f"✅ داده‌های نهایی: {len(df_all)} رکورد ماهانه (از {df_all['date'].min()} تا {df_all['date'].max()})")

# ============================================================
# ۴. مهندسی ویژگی‌ها (تأخیرها و میانگین متحرک)
# ============================================================
print("\n⚙️ مهندسی ویژگی‌ها...")

# ۴.۱ ایجاد ویژگی‌های تأخیری
feature_cols_base = ['net_ivt'] + [c for c in df_all.columns if c not in ['date', 'year', 'month', 'sea_level', 'net_ivt']]

for lag in range(1, N_LAGS+1):
    for col in feature_cols_base:
        df_all[f'{col}_lag{lag}'] = df_all[col].shift(lag)

# ۴.۲ میانگین متحرک
for col in feature_cols_base:
    df_all[f'{col}_ma{WINDOW_MA}'] = df_all[col].rolling(WINDOW_MA, min_periods=1).mean()

# ۴.۳ حذف ردیف‌های NaN (ناشی از تأخیرها)
df_all = df_all.dropna().reset_index(drop=True)
print(f"✅ پس از مهندسی ویژگی: {len(df_all)} رکورد")

# لیست نهایی ویژگی‌ها
feature_cols = [c for c in df_all.columns if c not in ['date', 'year', 'month', 'sea_level']]
print(f"📊 تعداد کل ویژگی‌ها: {len(feature_cols)}")

# ============================================================
# ۵. تقسیم داده‌ها به آموزش و آزمون
# ============================================================
mask_train = (df_all['date'] >= TRAIN_START) & (df_all['date'] <= TRAIN_END)
mask_test = (df_all['date'] >= TEST_START) & (df_all['date'] <= TEST_END)

df_train = df_all[mask_train].copy()
df_test = df_all[mask_test].copy()

print(f"\n📊 دوره آموزش: {df_train['date'].min()} تا {df_train['date'].max()} ({len(df_train)} ماه)")
print(f"📊 دوره آزمون: {df_test['date'].min()} تا {df_test['date'].max()} ({len(df_test)} ماه)")

X_train = df_train[feature_cols].values
y_train = df_train['sea_level'].values
X_test = df_test[feature_cols].values
y_test = df_test['sea_level'].values if len(df_test) > 0 else None

# استانداردسازی
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) if len(df_test) > 0 else None

# ============================================================
# ۶. انتخاب ویژگی با Lasso و RFE
# ============================================================
print("\n🎯 انتخاب ویژگی...")

# ۶.۱ LassoCV برای انتخاب خودکار
lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso.fit(X_train_scaled, y_train)
selected_lasso = np.where(lasso.coef_ != 0)[0]
selected_features_lasso = [feature_cols[i] for i in selected_lasso]
print(f"✅ Lasso: {len(selected_features_lasso)} ویژگی انتخاب شد")

# ۶.۲ RFE با RandomForest
estimator = RandomForestRegressor(n_estimators=50, random_state=42)
rfe = RFE(estimator, n_features_to_select=min(N_FEATURES_SELECT, len(feature_cols)))
rfe.fit(X_train_scaled, y_train)
selected_rfe = np.where(rfe.support_)[0]
selected_features_rfe = [feature_cols[i] for i in selected_rfe]
print(f"✅ RFE: {len(selected_features_rfe)} ویژگی انتخاب شد")

# ۶.۳ ترکیب دو روش (اشتراک)
selected_intersection = list(set(selected_features_lasso) & set(selected_features_rfe))
if len(selected_intersection) < 3:
    # اگر اشتراک کم بود، از RFE استفاده کن
    selected_final = selected_features_rfe[:10]
else:
    selected_final = selected_intersection

print(f"✅ ویژگی‌های نهایی: {len(selected_final)} -> {', '.join(selected_final)}")

# به‌روزرسانی داده‌های آموزشی با ویژگی‌های انتخاب‌شده
feature_cols_final = selected_final
idx_final = [feature_cols.index(f) for f in feature_cols_final]
X_train_final = X_train_scaled[:, idx_final]
X_test_final = X_test_scaled[:, idx_final] if len(df_test) > 0 else None

# ============================================================
# ۷. آموزش مدل‌ها (با ویژگی‌های انتخاب‌شده)
# ============================================================
print("\n🧠 آموزش مدل‌ها...")
models = {}

# ۷.۱ رگرسیون خطی (فقط زمان)
models['Linear (Trend)'] = LinearRegression()
models['Linear (Trend)'].fit(np.arange(len(y_train)).reshape(-1, 1), y_train)

# ۷.۲ رگرسیون چندگانه
models['Multiple Linear'] = LinearRegression()
models['Multiple Linear'].fit(X_train_final, y_train)

# ۷.۳ Ridge
models['Ridge'] = Ridge(alpha=1.0)
models['Ridge'].fit(X_train_final, y_train)

# ۷.۴ Lasso
models['Lasso'] = Lasso(alpha=0.01)
models['Lasso'].fit(X_train_final, y_train)

# ۷.۵ Random Forest
models['Random Forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
models['Random Forest'].fit(X_train_final, y_train)

# ۷.۶ XGBoost (اگر نصب باشد)
try:
    from xgboost import XGBRegressor
    models['XGBoost'] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    models['XGBoost'].fit(X_train_final, y_train)
    print("✅ XGBoost آموزش دید.")
except ImportError:
    print("⚠️ XGBoost نصب نیست. رد می‌شود.")
    models['XGBoost'] = None

# ۷.۷ Prophet (اگر نصب باشد و داده کافی داشته باشد)
try:
    from prophet import Prophet
    if len(df_train) > 30:
        df_prophet = df_train[['date', 'sea_level']].rename(columns={'date': 'ds', 'sea_level': 'y'})
        model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model_prophet.fit(df_prophet)
        models['Prophet'] = model_prophet
        print("✅ Prophet آموزش دید.")
    else:
        print("⚠️ داده‌های Prophet کافی نیست.")
        models['Prophet'] = None
except ImportError:
    print("⚠️ Prophet نصب نیست. رد می‌شود.")
    models['Prophet'] = None

# ۷.۸ ARIMA
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
predictions = {}

# برای پیش‌بینی آینده (۲۰۲۶-۲۰۳۰)
future_dates = pd.date_range('2026-01-01', '2030-12-01', freq='MS')
n_future = len(future_dates)

# برای مدل‌های مبتنی بر ویژگی، از میانگین ویژگی‌ها استفاده می‌کنیم
X_future = np.tile(np.mean(X_train_final, axis=0), (n_future, 1))

for name, model in models.items():
    if model is None:
        continue
    
    if name == 'Linear (Trend)':
        pred = model.predict(np.arange(len(y_train), len(y_train)+n_future).reshape(-1, 1))
        predictions[name] = pred
    
    elif name == 'Prophet':
        future = model.make_future_dataframe(periods=n_future, freq='MS', include_history=False)
        forecast = model.predict(future)
        pred = forecast['yhat'].values
        predictions[name] = pred
    
    elif name == 'ARIMA':
        try:
            pred = model.forecast(steps=n_future)
            predictions[name] = pred
        except Exception as e:
            print(f"⚠️ خطا در ARIMA: {e}")
            predictions[name] = np.full(n_future, np.nan)
    
    else:
        try:
            pred = model.predict(X_future)
            predictions[name] = pred
        except Exception as e:
            print(f"⚠️ خطا در {name}: {e}")
            predictions[name] = np.full(n_future, np.nan)

# ============================================================
# ۹. ارزیابی روی داده‌های تست (اگر موجود باشد)
# ============================================================
if len(df_test) > 0 and X_test_final is not None:
    print("\n📊 ارزیابی روی داده‌های تست...")
    test_results = []
    for name, model in models.items():
        if model is None or name in ['Linear (Trend)', 'Prophet', 'ARIMA']:
            continue
        try:
            y_pred = model.predict(X_test_final)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            test_results.append({'model': name, 'RMSE': rmse, 'R2': r2})
        except:
            pass
    if test_results:
        df_test_results = pd.DataFrame(test_results)
        print(df_test_results.to_string(index=False))

# ============================================================
# ۱۰. ذخیره نتایج
# ============================================================
print("\n💾 ذخیره نتایج...")

# ۱۰.۱ جدول پیش‌بینی ماهانه
df_pred = pd.DataFrame({'date': future_dates, 'year': future_dates.year, 'month': future_dates.month})
for name, pred in predictions.items():
    if pred is not None and len(pred) == n_future:
        df_pred[name] = pred

# ۱۰.۲ پیش‌بینی سالانه (میانگین)
df_annual_forecast = df_pred.groupby('year').mean(numeric_only=True).reset_index()
df_annual_forecast['year'] = df_annual_forecast['year'].astype(int)
df_annual_forecast.to_csv(os.path.join(OUTPUT_DIR, 'annual_forecast.csv'), index=False)

# ۱۰.۳ ذخیره ماهانه
df_pred.to_csv(os.path.join(OUTPUT_DIR, 'monthly_forecast.csv'), index=False)

# ۱۰.۴ نمودار مقایسه
fig, ax = plt.subplots(figsize=(16, 8))
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'cyan', 'magenta']
for i, (name, pred) in enumerate(predictions.items()):
    if pred is not None and len(pred) == n_future:
        ax.plot(future_dates, pred, 'o-', label=name, color=colors[i % len(colors)], linewidth=1.5, markersize=4)

# داده‌های تاریخی
ax.plot(df_all['date'], df_all['sea_level'], 'k-', alpha=0.4, label='Historical', linewidth=0.8)
ax.axvline(x=pd.Timestamp('2026-01-01'), color='gray', linestyle=':', label='Forecast Start')
ax.set_xlabel('Date')
ax.set_ylabel('Sea Level (m)')
ax.set_title('Monthly Forecast of Caspian Sea Level (2026-2030)')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'monthly_forecast_comparison.png'), dpi=150)
plt.close()

# ۱۰.۵ اهمیت ویژگی‌ها (Random Forest)
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

# ۱۰.۶ گزارش نهایی
with open(os.path.join(OUTPUT_DIR, 'summary_report_final.txt'), 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("🌊 گزارش نهایی پیش‌بینی تراز آب دریای خزر\n")
    f.write("="*80 + "\n\n")
    f.write(f"دوره آموزش: {TRAIN_START} تا {TRAIN_END} ({len(df_train)} ماه)\n")
    f.write(f"دوره پیش‌بینی: {TEST_START} تا {TEST_END} ({n_future} ماه)\n")
    f.write(f"تعداد ویژگی‌های اولیه: {len(feature_cols)}\n")
    f.write(f"تعداد ویژگی‌های انتخابی: {len(feature_cols_final)}\n")
    f.write(f"ویژگی‌های انتخابی: {', '.join(feature_cols_final)}\n\n")
    
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
print("📄 فایل‌های تولید شده:")
print("   - monthly_forecast.csv")
print("   - annual_forecast.csv")
print("   - monthly_forecast_comparison.png")
print("   - feature_importance_final.png")
print("   - summary_report_final.txt")
print("="*80)