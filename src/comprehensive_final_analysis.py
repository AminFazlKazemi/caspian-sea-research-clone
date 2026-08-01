#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
کد جامع نهایی (نسخه ۶.۲) – تحلیل یکپارچه با اصلاحات نهایی
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.utils import resample
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# ۱. تنظیم مسیرها (قابل ویرایش)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "final_comprehensive_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# مسیرهای فایل‌های ورودی
IVT_PATH = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\basin_border\caspian_unified_analysis\merged_annual.csv"
EVAP_PATH = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\basin_border\evaporation_analysis\annual_series.csv"
TEMP_PATH = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\temperature_analysis\temperature_annual_series.csv"
SEA_PATH = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\basin_border\caspian_unified_analysis\caspian_sea_level_raw.csv"
VOLGA_PATH = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\volga_discharge.csv"
INDICES_PATH = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\indices_complete.xlsx"

# ============================================================
# ۲. توابع بارگذاری امن
# ============================================================
def safe_read_csv(path, sep=None):
    """خواندن CSV با تشخیص خودکار جداکننده و اصلاح ستون‌ها"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ فایل {path} یافت نشد!")
    if sep is not None:
        df = pd.read_csv(path, sep=sep)
    else:
        try:
            df = pd.read_csv(path)
        except:
            df = pd.read_csv(path, sep=';')
    if len(df.columns) == 1 and ';' in df.columns[0]:
        df = df[df.columns[0]].str.split(';', expand=True)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def find_column(df, candidates):
    """پیدا کردن اولین ستون از لیست کاندیداها"""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_sea_level_direct(path):
    """بارگذاری مستقیم فایل تراز با جداکننده ; و شناسایی ستون wse"""
    try:
        df = pd.read_csv(path, sep=';')
    except:
        df = pd.read_csv(path)
    if len(df.columns) == 3 and all(c in ['0','1','2'] for c in df.columns):
        df.columns = ['datetime', 'wse', 'wse_u']
    time_col = find_column(df, ['datetime', 'date', 'time'])
    level_col = find_column(df, ['wse', 'level', 'height', 'sea', 'sl', 'level_m', 'wse_avg'])
    if time_col is None or level_col is None:
        raise ValueError("❌ ستون زمان یا سطح آب در فایل تراز یافت نشد!")
    df['year'] = pd.to_datetime(df[time_col]).dt.year
    df[level_col] = pd.to_numeric(df[level_col], errors='coerce')
    yearly = df.groupby('year')[level_col].mean().reset_index()
    yearly.rename(columns={level_col: 'level_m'}, inplace=True)
    yearly['year'] = yearly['year'].astype(int)
    yearly = yearly.dropna(subset=['level_m'])
    print(f"   ✅ تراز خزر مستقیماً بارگذاری شد: {len(yearly)} سال ({yearly['year'].min()}-{yearly['year'].max()})")
    return yearly

def load_amoc_extended():
    """بارگذاری AMOC گسترش‌یافته (در صورت موجود بودن)"""
    print("   🔄 بارگذاری AMOC (اختیاری)...")
    caesar_path = os.path.join(BASE_DIR, "Caesar.csv")
    rapid_path = os.path.join(BASE_DIR, "amoc_rapid_RAPID (12 month mean).csv")
    if not os.path.exists(caesar_path) or not os.path.exists(rapid_path):
        print("   ⚠️ فایل‌های AMOC یافت نشد، از این متغیر صرف‌نظر می‌شود.")
        return None
    try:
        caesar = safe_read_csv(caesar_path)
        caesar.columns = ['year', 'caesar_index', 'lower', 'upper']
        caesar = caesar[['year', 'caesar_index']].dropna()
        caesar['year'] = caesar['year'].astype(int)
        caesar = caesar[caesar['year'] >= 1870]

        rapid = safe_read_csv(rapid_path)
        print("      📊 ستون‌های RAPID:", rapid.columns.tolist())
        rapid_cols = rapid.columns.tolist()
        year_col = None
        rapid_col = None
        for c in rapid_cols:
            if 'year' in c.lower() or 'time' in c.lower():
                year_col = c
            if 'rapid' in c.lower() or 'sv' in c.lower():
                rapid_col = c
        if year_col is None or rapid_col is None:
            if len(rapid_cols) >= 2:
                year_col = rapid_cols[0]
                rapid_col = rapid_cols[1]
                print(f"      ⚠️ ستون‌ها به‌صورت پیش‌فرض: سال='{year_col}', RAPID='{rapid_col}'")
            else:
                raise ValueError("ستون‌های سال یا RAPID در فایل یافت نشدند!")
        rapid['year'] = rapid[year_col].astype(int)
        rapid_annual = rapid.groupby('year')[rapid_col].mean().reset_index()
        rapid_annual.columns = ['year', 'rapid_sv']
        rapid_annual = rapid_annual.dropna()
        rapid_annual = rapid_annual[rapid_annual['year'] >= 2004]

        merged_cal = pd.merge(caesar, rapid_annual, on='year', how='inner')
        merged_cal = merged_cal[merged_cal['year'] <= 2016]
        if len(merged_cal) < 5:
            print("   ⚠️ داده‌های کافی برای کالیبراسیون وجود ندارد!")
            return None

        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(merged_cal[['caesar_index']].values, merged_cal['rapid_sv'].values)
        beta0, beta1 = model.intercept_, model.coef_[0]
        r2_cal = model.score(merged_cal[['caesar_index']].values, merged_cal['rapid_sv'].values)
        print(f"      ✅ کالیبراسیون: R² = {r2_cal:.3f}")

        caesar_pre = caesar[caesar['year'] < 2004].copy()
        caesar_pre['amoc_reconstructed'] = beta0 + beta1 * caesar_pre['caesar_index']
        rapid_final = rapid_annual.copy()
        rapid_final['amoc_reconstructed'] = rapid_final['rapid_sv']

        amoc_final = pd.concat([caesar_pre[['year', 'amoc_reconstructed']],
                               rapid_final[['year', 'amoc_reconstructed']]], ignore_index=True)
        amoc_final = amoc_final.sort_values('year').reset_index(drop=True)
        amoc_final.to_csv(os.path.join(OUTPUT_DIR, 'AMOC_extended.csv'), index=False)
        print(f"      ✅ AMOC: {len(amoc_final)} سال (۱۸۷۰–{amoc_final['year'].max()})")
        return amoc_final
    except Exception as e:
        print(f"   ⚠️ خطا در بارگذاری AMOC: {e}")
        return None

# ============================================================
# ۳. بارگذاری همه داده‌ها (با مدیریت داده‌های مفقود)
# ============================================================
def load_all_data():
    print("\n📂 بارگذاری داده‌ها...")

    # ---- IVT ----
    ivt = safe_read_csv(IVT_PATH)
    ivt_col = find_column(ivt, ['net_flux_kg_s', 'ivt_net', 'ivt', 'flux', 'net_flux'])
    if ivt_col is None:
        raise ValueError(f"❌ ستون IVT پیدا نشد! ستون‌ها: {ivt.columns.tolist()}")
    ivt = ivt[['year', ivt_col]].dropna()
    ivt.rename(columns={ivt_col: 'ivt_net'}, inplace=True)
    ivt['year'] = ivt['year'].astype(int)
    print(f"   ✅ IVT: {len(ivt)} سال")

    # ---- EVAP ----
    evap = safe_read_csv(EVAP_PATH)
    evap_col = find_column(evap, ['e', 'evap', 'evaporation', 'et'])
    if evap_col is None:
        raise ValueError(f"❌ ستون تبخیر پیدا نشد! ستون‌ها: {evap.columns.tolist()}")
    evap = evap[['year', evap_col]].dropna()
    evap.rename(columns={evap_col: 'evaporation'}, inplace=True)
    evap['year'] = evap['year'].astype(int)
    print(f"   ✅ تبخیر: {len(evap)} سال")

    # ---- TEMP ----
    temp = safe_read_csv(TEMP_PATH)
    temp_col = find_column(temp, ['mean_tmean', 'tmean', 'temperature', 'temp'])
    if temp_col is None:
        raise ValueError(f"❌ ستون دما پیدا نشد! ستون‌ها: {temp.columns.tolist()}")
    temp = temp[['year', temp_col]].dropna()
    temp.rename(columns={temp_col: 'temperature'}, inplace=True)
    temp['year'] = temp['year'].astype(int)
    print(f"   ✅ دما: {len(temp)} سال")

    # ---- SEA LEVEL ----
    sea_yearly = load_sea_level_direct(SEA_PATH)

    # ---- VOLGA ----
    volga = None
    if os.path.exists(VOLGA_PATH):
        volga = safe_read_csv(VOLGA_PATH)
        volga_col = find_column(volga, ['discharge_km3', 'volume_km3', 'discharge', 'q'])
        if volga_col is not None:
            volga = volga[['year', volga_col]].dropna()
            volga.rename(columns={volga_col: 'discharge_km3'}, inplace=True)
            volga['year'] = volga['year'].astype(int)
            print(f"   ✅ رواناب ولگا: {len(volga)} سال")
        else:
            print("   ⚠️ ستون رواناب ولگا پیدا نشد!")
            volga = None

    # ---- INDICES ----
    indices = None
    if os.path.exists(INDICES_PATH):
        try:
            indices = pd.read_excel(INDICES_PATH, sheet_name='Sheet1')
            indices['date'] = pd.to_datetime(indices['date'])
            indices['year'] = indices['date'].dt.year
            index_cols = ['year', 'NAO', 'SOI', 'PNA', 'WP', 'EA', 'SCA', 'POL', 'EA/WR', 'ONI']
            existing = [col for col in index_cols if col in indices.columns]
            if existing:
                indices = indices[existing]
                indices = indices.groupby('year').mean().reset_index()
                print(f"   ✅ شاخص‌ها: {len(indices)} سال")
            else:
                indices = None
        except Exception as e:
            print(f"   ⚠️ خطا در خواندن شاخص‌ها: {e}")
            indices = None

    # ---- AMOC ----
    amoc = load_amoc_extended()

    # ---- MERGE ----
    df = ivt.merge(evap, on='year', how='outer')
    df = df.merge(temp, on='year', how='outer')
    df = df.merge(sea_yearly, on='year', how='outer')
    if volga is not None:
        df = df.merge(volga, on='year', how='outer')
    if indices is not None:
        df = df.merge(indices, on='year', how='outer')
    if amoc is not None:
        df = df.merge(amoc, on='year', how='outer')

    df = df.sort_values('year').reset_index(drop=True)
    df = df[df['year'] >= 1940]

    # ---- درون‌یابی ----
    print("\n   🔄 درون‌یابی داده‌های مفقود...")
    numeric_cols = ['ivt_net', 'evaporation', 'temperature', 'level_m', 'discharge_km3']
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
            if df[col].isna().any():
                df[col].fillna(df[col].mean(), inplace=True)
            print(f"      ✅ {col}: درون‌یابی انجام شد")

    # ---- حذف نهایی NaN ----
    df = df.dropna()
    print(f"   ✅ پس از حذف NaN: {len(df)} ردیف")
    return df

# ============================================================
# ۴. تحلیل‌های پیشرفته (بدون تغییر)
# ============================================================
def advanced_analysis(df):
    print("\n" + "=" * 80)
    print("📊 تحلیل‌های پیشرفته")
    print("=" * 80)

    if 'level_m' not in df.columns or len(df) < 10:
        print("⚠️ داده‌های کافی برای تحلیل وجود ندارد!")
        return

    predictors = ['ivt_net', 'evaporation', 'temperature']
    if 'discharge_km3' in df.columns:
        predictors.append('discharge_km3')
    if 'amoc_reconstructed' in df.columns:
        predictors.append('amoc_reconstructed')
    if 'NAO' in df.columns:
        predictors.append('NAO')
    target = 'level_m'

    print(f"\n🎯 متغیر هدف: {target}")
    print(f"📊 متغیرهای پیش‌بینی‌کننده: {predictors}")

    # ---- همبستگی‌ها ----
    print("\n📊 همبستگی‌های پیشرفته...")
    corr_vars = [target] + predictors
    corr_df = df[corr_vars].dropna()
    pearson_corr = corr_df.corr(method='pearson')
    pearson_corr.to_csv(os.path.join(OUTPUT_DIR, 'pearson_correlation.csv'))
    spearman_corr = corr_df.corr(method='spearman')
    spearman_corr.to_csv(os.path.join(OUTPUT_DIR, 'spearman_correlation.csv'))
    kendall_corr = corr_df.corr(method='kendall')
    kendall_corr.to_csv(os.path.join(OUTPUT_DIR, 'kendall_correlation.csv'))

    # ---- Partial Correlation ----
    try:
        from pingouin import partial_corr
        partial_results = []
        for var in predictors:
            if var in df.columns:
                # covar باید شامل همه متغیرهای پیش‌بینی‌کننده به جز خود var باشد
                covar_list = [p for p in predictors if p != var]
                pc = partial_corr(data=df, x=var, y=target, covar=covar_list)
                partial_results.append({
                    'variable': var,
                    'partial_r': pc['r'].values[0],
                    'p_value': pc['p-val'].values[0]
                })
        pd.DataFrame(partial_results).to_csv(os.path.join(OUTPUT_DIR, 'partial_correlation.csv'))
        print("   ✅ همبستگی جزئی محاسبه شد.")
    except ImportError:
        print("   ⚠️ pingouin نصب نیست، همبستگی جزئی انجام نشد.")
    except Exception as e:
        print(f"   ⚠️ خطا در همبستگی جزئی: {e}")
    # ---- تحلیل تأخیری ----
    print("\n⏳ تحلیل تأخیری...")
    max_lag = 10
    lag_results = []
    for lag in range(1, max_lag+1):
        row = {'lag': lag}
        for var in predictors:
            if var in df.columns:
                col = f'{var}_lag_{lag}'
                df[col] = df[var].shift(lag)
                corr_val = df[[target, col]].dropna().corr().iloc[0,1]
                row[f'{var}_corr'] = corr_val
        lag_results.append(row)
    pd.DataFrame(lag_results).to_csv(os.path.join(OUTPUT_DIR, 'lag_analysis.csv'), index=False)
    print("   ✅ تحلیل تأخیری ذخیره شد.")

    # ---- علّیت گرنجر ----
    print("\n🔗 آزمون علّیت گرنجر...")
    granger_vars = predictors
    granger_results = []
    for var in granger_vars:
        if var not in df.columns:
            continue
        data = df[[target, var]].dropna()
        if len(data) < 20:
            continue
        try:
            print(f"\n   Testing {var} → {target}")
            test = grangercausalitytests(data, maxlag=4, verbose=True)
            p_vals = [test[i][0]['ssr_ftest'][1] for i in range(1,5)]
            best_lag = np.argmin(p_vals) + 1
            best_p = p_vals[best_lag-1]
            granger_results.append({
                'cause': var,
                'effect': target,
                'best_lag': best_lag,
                'p_value': best_p,
                'significant': best_p < 0.05
            })
        except Exception as e:
            print(f"   Error: {e}")
    pd.DataFrame(granger_results).to_csv(os.path.join(OUTPUT_DIR, 'granger_causality_full.csv'), index=False)

    # ---- VAR ----
    print("\n📈 مدل VAR...")
    try:
        var_data = df[[target] + [p for p in predictors if p in df.columns]].dropna()
        if len(var_data) > 10:
            model_var = VAR(var_data)
            results_var = model_var.fit(maxlags=2)
            print(results_var.summary())
            with open(os.path.join(OUTPUT_DIR, 'var_summary.txt'), 'w', encoding='utf-8') as f:
                f.write(str(results_var.summary()))
        else:
            print("   ⚠️ داده‌های کافی برای VAR وجود ندارد.")
    except Exception as e:
        print(f"   ⚠️ خطا در VAR: {e}")

    # ---- روند ----
    print("\n📈 تحلیل روند...")
    try:
        import pymannkendall as mk
        trend_vars = [target] + predictors
        trend_results = []
        for var in trend_vars:
            if var not in df.columns:
                continue
            series = df[var].dropna().values
            if len(series) < 10:
                continue
            mk_test = mk.original_test(series)
            sen_slope = mk.sens_slope(series)
            trend_results.append({
                'variable': var,
                'trend': mk_test.trend,
                'p_value': mk_test.p,
                'sen_slope': sen_slope.slope
            })
            print(f"   {var}: {mk_test.trend} (p={mk_test.p:.4f})")
        pd.DataFrame(trend_results).to_csv(os.path.join(OUTPUT_DIR, 'mann_kendall_trends_full.csv'), index=False)
    except ImportError:
        print("   ⚠️ pymannkendall نصب نیست!")

    # ---- طیف ----
    print("\n🎵 تحلیل طیفی...")
    spectral_vars = [target] + predictors
    spectral_results = []
    for var in spectral_vars:
        if var not in df.columns:
            continue
        series = df[var].dropna().values
        if len(series) < 10:
            continue
        x = np.arange(len(series))
        slope, intercept = np.polyfit(x, series, 1)
        detrended = series - (slope*x + intercept)
        n = len(detrended)
        freq = fftfreq(n)[1:n//2]
        fft_vals = fft(detrended)[1:n//2]
        power = np.abs(fft_vals)**2
        peaks, _ = find_peaks(power, height=np.percentile(power, 75))
        periods = 1/freq
        dominant_periods = periods[peaks] if len(peaks)>0 else []
        spectral_results.append({
            'variable': var,
            'dominant_periods': dominant_periods[:5],
            'n_peaks': len(peaks)
        })
        print(f"   {var}: دوره‌های غالب = {[f'{p:.1f}' for p in dominant_periods[:3]]}")
        plt.figure(figsize=(10,5))
        plt.semilogx(periods, power, 'b-', linewidth=1.5)
        plt.scatter(periods[peaks], power[peaks], color='red', s=50, label='قله‌ها')
        plt.xlabel('دوره (سال)')
        plt.ylabel('قدرت طیفی')
        plt.title(f'طیف‌قدرت {var}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, f'spectrum_{var}.png'), dpi=150)
        plt.close()
    pd.DataFrame(spectral_results).to_csv(os.path.join(OUTPUT_DIR, 'spectral_analysis.csv'), index=False)

    # ---- موجک ----
    print("\n🌊 تحلیل موجک...")
    try:
        wavelet_vars = [target] + ['ivt_net', 'evaporation', 'temperature']
        wavelet_results = []
        for var in wavelet_vars:
            if var not in df.columns:
                continue
            series = df[var].dropna().values
            if len(series) < 10:
                continue
            x = np.arange(len(series))
            slope, intercept = np.polyfit(x, series, 1)
            detrended = series - (slope*x + intercept)
            scales = np.arange(1, 30)
            coeffs, freqs = pywt.cwt(detrended, scales, 'morl')
            power = np.abs(coeffs)
            global_power = np.mean(power, axis=1)
            dominant_scale = scales[np.argmax(global_power)]
            dominant_period = 1 / freqs[np.argmax(global_power)] if freqs[np.argmax(global_power)] > 0 else np.nan
            wavelet_results.append({
                'variable': var,
                'dominant_scale': dominant_scale,
                'dominant_period_years': dominant_period,
                'max_power': np.max(global_power)
            })
            print(f"   {var}: دوره غالب = {dominant_period:.1f} سال")
            plt.figure(figsize=(12,6))
            plt.contourf(np.arange(len(series)), scales, power, levels=50, cmap='jet')
            plt.colorbar(label='Intensity')
            plt.xlabel('سال (ایندکس)')
            plt.ylabel('Scale')
            plt.title(f'تبدیل موجک {var}')
            plt.savefig(os.path.join(OUTPUT_DIR, f'wavelet_{var}.png'), dpi=150)
            plt.close()
        pd.DataFrame(wavelet_results).to_csv(os.path.join(OUTPUT_DIR, 'wavelet_analysis.csv'), index=False)
    except ImportError:
        print("   ⚠️ pywt نصب نیست!")

# ============================================================
# ۵. مدل‌سازی و اعتبارسنجی (با مدیریت خطا و ذخیره‌سازی مطمئن)
# ============================================================
def modeling_and_validation(df):
    print("\n" + "=" * 80)
    print("📈 مدل‌سازی و اعتبارسنجی")
    print("=" * 80)

    target = 'level_m'
    if target not in df.columns or len(df) < 10:
        print("⚠️ داده‌های کافی برای مدل‌سازی وجود ندارد!")
        return

    predictors = ['ivt_net', 'evaporation', 'temperature']
    if 'discharge_km3' in df.columns:
        predictors.append('discharge_km3')
    if 'amoc_reconstructed' in df.columns:
        predictors.append('amoc_reconstructed')
    if 'NAO' in df.columns:
        predictors.append('NAO')
    predictors = [p for p in predictors if p in df.columns]

    # ---- حذف ردیف‌های حاوی NaN (نگه‌داری year) ----
    df_clean = df[['year'] + predictors + [target]].dropna()
    if len(df_clean) < 10:
        print(f"⚠️ پس از حذف NaN، فقط {len(df_clean)} ردیف باقی مانده است!")
        return
    print(f"   ✅ تعداد ردیف‌های پاک: {len(df_clean)}")

    X = df_clean[predictors].values
    y = df_clean[target].values
    years = df_clean['year'].values

    # استانداردسازی
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---- اعتبارسنجی K-Fold (با مدیریت خطا) ----
    print("\n🔁 اعتبارسنجی K-Fold (۵ بخش)...")
    try:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_r2 = cross_val_score(LinearRegression(), X_scaled, y, cv=kf, scoring='r2')
        cv_rmse = np.sqrt(-cross_val_score(LinearRegression(), X_scaled, y, cv=kf,
                                           scoring='neg_mean_squared_error'))
        print(f"   R² (میانگین ± انحراف): {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
        print(f"   RMSE (میانگین ± انحراف): {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
    except Exception as e:
        print(f"   ⚠️ خطا در اعتبارسنجی K-Fold: {e}")
        cv_r2 = np.array([np.nan])
        cv_rmse = np.array([np.nan])

    # ---- مدل خطی ----
    print("\n📐 مدل رگرسیون خطی...")
    lr = LinearRegression()
    lr.fit(X_scaled, y)
    y_pred_lr = lr.predict(X_scaled)
    r2_lr = r2_score(y, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y, y_pred_lr))
    mae_lr = mean_absolute_error(y, y_pred_lr)
    print(f"   R² = {r2_lr:.4f}, RMSE = {rmse_lr:.4f}, MAE = {mae_lr:.4f}")

    coef_df = pd.DataFrame({
        'Variable': ['const'] + predictors,
        'Coefficient': [lr.intercept_] + list(lr.coef_),
    })
    coef_df.to_csv(os.path.join(OUTPUT_DIR, 'linear_regression_coefficients.csv'), index=False)

    # ---- بوت‌استرپ ----
    print("\n🎲 بوت‌استرپ (۲۰۰۰ تکرار)...")
    n_bootstrap = 2000
    coefs_boot = []
    preds_boot = []
    for _ in range(n_bootstrap):
        idx = resample(range(len(X_scaled)), replace=True, n_samples=len(X_scaled))
        X_boot = X_scaled[idx]
        y_boot = y[idx]
        model = LinearRegression()
        model.fit(X_boot, y_boot)
        coefs_boot.append(model.coef_)
        preds_boot.append(model.predict(X_scaled))
    coefs_boot = np.array(coefs_boot)
    preds_boot = np.array(preds_boot)

    ci_levels = [90, 95, 99]
    bootstrap_ci = []
    for level in ci_levels:
        lower = (100 - level) / 2
        upper = 100 - lower
        ci_lower = np.percentile(coefs_boot, lower, axis=0)
        ci_upper = np.percentile(coefs_boot, upper, axis=0)
        for i, p in enumerate(predictors):
            bootstrap_ci.append({
                'variable': p,
                'ci_%': level,
                'ci_lower': ci_lower[i],
                'ci_upper': ci_upper[i]
            })
    pd.DataFrame(bootstrap_ci).to_csv(os.path.join(OUTPUT_DIR, 'bootstrap_confidence_intervals.csv'), index=False)
    print("   ✅ عدم‌قطعیت با سطوح مختلف CI محاسبه شد.")

    # ---- Random Forest ----
    print("\n🌳 مدل Random Forest (با بهینه‌سازی)...")
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_rf = GridSearchCV(rf, rf_params, cv=5, scoring='r2', n_jobs=-1)
    grid_rf.fit(X_scaled, y)
    best_rf = grid_rf.best_estimator_
    y_pred_rf = best_rf.predict(X_scaled)
    rf_r2 = r2_score(y, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y, y_pred_rf))
    rf_mae = mean_absolute_error(y, y_pred_rf)
    print(f"   Random Forest: R² = {rf_r2:.4f}, RMSE = {rf_rmse:.4f}, MAE = {rf_mae:.4f}")
    print(f"   بهترین پارامترها: {grid_rf.best_params_}")

    imp_rf = best_rf.feature_importances_
    imp_df = pd.DataFrame({'Variable': predictors, 'Importance': imp_rf})
    imp_df.to_csv(os.path.join(OUTPUT_DIR, 'rf_feature_importance.csv'), index=False)

    # ---- XGBoost ----
    try:
        from xgboost import XGBRegressor
        print("\n🚀 مدل XGBoost...")
        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.05, 0.1]
        }
        xgb = XGBRegressor(random_state=42, n_jobs=-1)
        grid_xgb = GridSearchCV(xgb, xgb_params, cv=5, scoring='r2', n_jobs=-1)
        grid_xgb.fit(X_scaled, y)
        best_xgb = grid_xgb.best_estimator_
        y_pred_xgb = best_xgb.predict(X_scaled)
        xgb_r2 = r2_score(y, y_pred_xgb)
        xgb_rmse = np.sqrt(mean_squared_error(y, y_pred_xgb))
        xgb_mae = mean_absolute_error(y, y_pred_xgb)
        print(f"   XGBoost: R² = {xgb_r2:.4f}, RMSE = {xgb_rmse:.4f}, MAE = {xgb_mae:.4f}")
        print(f"   بهترین پارامترها: {grid_xgb.best_params_}")
        imp_xgb = best_xgb.feature_importances_
        imp_xgb_df = pd.DataFrame({'Variable': predictors, 'Importance': imp_xgb})
        imp_xgb_df.to_csv(os.path.join(OUTPUT_DIR, 'xgb_feature_importance.csv'), index=False)
    except ImportError:
        print("   ⚠️ XGBoost نصب نیست، این مدل اجرا نشد.")

    # ---- مقایسه مدل‌ها ----
    model_comparison = pd.DataFrame({
        'Model': ['Linear Regression', 'Random Forest'],
        'R²': [r2_lr, rf_r2],
        'RMSE': [rmse_lr, rf_rmse],
        'MAE': [mae_lr, rf_mae]
    })
    if 'xgb_r2' in locals():
        model_comparison = pd.concat([model_comparison, pd.DataFrame({
            'Model': ['XGBoost'],
            'R²': [xgb_r2],
            'RMSE': [xgb_rmse],
            'MAE': [xgb_mae]
        })], ignore_index=True)
    model_comparison.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison_advanced.csv'), index=False)
    print("\n📊 مقایسه مدل‌ها:")
    print(model_comparison.round(4).to_string())

    # ---- تحلیل حساسیت ----
    print("\n🔬 تحلیل حساسیت...")
    sensitivity_results = []
    base_pred = best_rf.predict(X_scaled)
    for i, var in enumerate(predictors):
        for pct in [-20, -10, +10, +20]:
            X_perturbed = X_scaled.copy()
            X_perturbed[:, i] = X_perturbed[:, i] * (1 + pct/100)
            y_perturbed = best_rf.predict(X_perturbed)
            change = np.mean(np.abs(y_perturbed - base_pred)) / np.std(y)
            sensitivity_results.append({
                'variable': var,
                'change_%': pct,
                'sensitivity': change
            })
    pd.DataFrame(sensitivity_results).to_csv(os.path.join(OUTPUT_DIR, 'sensitivity_analysis.csv'), index=False)

    # ---- سناریوسازی ----
    print("\n📊 سناریوسازی...")
    scenarios = {
        'Baseline': {},
        'IVT_+10%': {'ivt_net': 1.1},
        'IVT_-10%': {'ivt_net': 0.9},
        'Evap_+20%': {'evaporation': 1.2},
        'Evap_-20%': {'evaporation': 0.8},
        'Temp_+2C': {'temperature': lambda x: x + 2},
        'Temp_-2C': {'temperature': lambda x: x - 2},
        'Discharge_+20%': {'discharge_km3': 1.2},
        'Discharge_-20%': {'discharge_km3': 0.8}
    }
    scenario_results = []
    for name, changes in scenarios.items():
        X_scenario = X_scaled.copy()
        for i, var in enumerate(predictors):
            if var in changes:
                if callable(changes[var]):
                    X_scenario[:, i] = changes[var](X_scenario[:, i])
                else:
                    X_scenario[:, i] = X_scenario[:, i] * changes[var]
        y_scenario = best_rf.predict(X_scenario)
        scenario_results.append({
            'scenario': name,
            'mean_level': np.mean(y_scenario),
            'std_level': np.std(y_scenario),
            'change_from_baseline_%': 100 * (np.mean(y_scenario) - np.mean(y_pred_rf)) / np.mean(y_pred_rf)
        })
    pd.DataFrame(scenario_results).to_csv(os.path.join(OUTPUT_DIR, 'scenario_analysis.csv'), index=False)

    # ---- رویدادهای حدی ----
    print("\n🌊 تحلیل رویدادهای حدی...")
    threshold_high = df[target].quantile(0.9)
    threshold_low = df[target].quantile(0.1)
    high_events = df[df[target] > threshold_high]
    low_events = df[df[target] < threshold_low]
    print(f"   سال‌های تراز بالا ({len(high_events)} سال): {list(high_events['year'].values)}")
    print(f"   سال‌های تراز پایین ({len(low_events)} سال): {list(low_events['year'].values)}")

    extreme_comparison = pd.DataFrame({
        'Variable': predictors,
        'High_Level_Mean': [high_events[var].mean() if var in high_events else np.nan for var in predictors],
        'Low_Level_Mean': [low_events[var].mean() if var in low_events else np.nan for var in predictors]
    })
    extreme_comparison.to_csv(os.path.join(OUTPUT_DIR, 'extreme_events_comparison.csv'), index=False)

    # ---- شاخص ترکیبی ----
    print("\n📊 ساخت شاخص ترکیبی اقلیمی...")
    ivt_std = (df['ivt_net'] - df['ivt_net'].mean()) / df['ivt_net'].std()
    evap_std = (df['evaporation'] - df['evaporation'].mean()) / df['evaporation'].std()
    temp_std = (df['temperature'] - df['temperature'].mean()) / df['temperature'].std()
    if 'discharge_km3' in df.columns:
        dis_std = (df['discharge_km3'] - df['discharge_km3'].mean()) / df['discharge_km3'].std()
        df['climate_composite_index'] = (ivt_std - evap_std + temp_std + dis_std) / 4
    else:
        df['climate_composite_index'] = (ivt_std - evap_std + temp_std) / 3
    composite_corr = df[['climate_composite_index', target]].corr().iloc[0,1]
    print(f"   همبستگی شاخص ترکیبی با تراز: {composite_corr:.4f}")
    df[['year', 'climate_composite_index']].to_csv(os.path.join(OUTPUT_DIR, 'composite_index.csv'), index=False)

    # ---- TimeSeriesSplit ----
    print("\n🔄 اعتبارسنجی با TimeSeriesSplit...")
    tscv = TimeSeriesSplit(n_splits=5)
    ts_scores_r2 = []
    ts_scores_rmse = []
    for train_idx, test_idx in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model_ts = LinearRegression()
        model_ts.fit(X_train, y_train)
        y_pred_ts = model_ts.predict(X_test)
        ts_scores_r2.append(r2_score(y_test, y_pred_ts))
        ts_scores_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred_ts)))
    print(f"   TimeSeriesSplit - R²: {np.mean(ts_scores_r2):.4f} ± {np.std(ts_scores_r2):.4f}")
    print(f"   TimeSeriesSplit - RMSE: {np.mean(ts_scores_rmse):.4f} ± {np.std(ts_scores_rmse):.4f}")
    pd.DataFrame({
        'Fold': range(1, 6),
        'R2': ts_scores_r2,
        'RMSE': ts_scores_rmse
    }).to_csv(os.path.join(OUTPUT_DIR, 'timeseries_cv_results.csv'), index=False)

    # ---- ذخیره پیش‌بینی‌ها (با اطمینان از وجود فایل) ----
    print("\n💾 ذخیره پیش‌بینی‌ها...")
    pred_df = pd.DataFrame({
        'year': years,
        'actual': y,
        'predicted_lr': y_pred_lr,
        'predicted_rf': y_pred_rf,
        'pred_ci_lower': np.percentile(preds_boot, 2.5, axis=0),
        'pred_ci_upper': np.percentile(preds_boot, 97.5, axis=0)
    })
    pred_path = os.path.join(OUTPUT_DIR, 'predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    if os.path.exists(pred_path):
        print(f"   ✅ پیش‌بینی‌ها در {pred_path} ذخیره شدند.")
    else:
        print(f"   ⚠️ خطا در ذخیره‌سازی {pred_path}!")

    # ---- معادله نهایی ----
    eq = f"{target} = {lr.intercept_:.4f}"
    for i, p in enumerate(predictors):
        sign = '+' if lr.coef_[i] >= 0 else '-'
        eq += f" {sign} {abs(lr.coef_[i]):.4f} × {p}"
    with open(os.path.join(OUTPUT_DIR, 'final_equation.txt'), 'w', encoding='utf-8') as f:
        f.write("معادله نهایی مدل ترکیبی\n")
        f.write("=" * 60 + "\n")
        f.write(eq + "\n\n")
        f.write(f"R² = {r2_lr:.4f}, RMSE = {rmse_lr:.4f}\n")
        if not np.isnan(cv_r2).all():
            f.write(f"CV R² (5-Fold) = {cv_r2.mean():.4f} ± {cv_r2.std():.4f}\n")
        f.write(f"Random Forest R² = {rf_r2:.4f}\n")
        if 'xgb_r2' in locals():
            f.write(f"XGBoost R² = {xgb_r2:.4f}\n")
    print(f"\n📝 معادله نهایی ذخیره شد.")

# ============================================================
# ۶. رسم نمودارهای نهایی (با مدیریت عدم وجود فایل)
# ============================================================
def final_plots(df):
    print("\n📈 رسم نمودارهای نهایی...")
    target = 'level_m'
    if target not in df.columns or len(df) < 5:
        print("⚠️ داده‌های کافی برای رسم نمودار وجود ندارد!")
        return

    predictors = ['ivt_net', 'evaporation', 'temperature']
    if 'discharge_km3' in df.columns:
        predictors.append('discharge_km3')
    if 'amoc_reconstructed' in df.columns:
        predictors.append('amoc_reconstructed')
    if 'NAO' in df.columns:
        predictors.append('NAO')
    predictors = [p for p in predictors if p in df.columns]

    years = df['year'].values

    # ---- ۱. سری زمانی ----
    fig, axes = plt.subplots(len(predictors)+2, 1, figsize=(14, 3*(len(predictors)+2)), sharex=True)
    for i, var in enumerate([predictors[0]] + predictors + [target]):
        ax = axes[i]
        ax.plot(years, df[var], linewidth=2)
        ax.set_ylabel(var)
        ax.grid(True, alpha=0.3)
        ax.axhline(df[var].mean(), color='r', linestyle='--', alpha=0.5)
    axes[-1].set_xlabel('سال')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'time_series_all_variables.png'), dpi=300)
    plt.close()
    print("   ✅ time_series_all_variables.png")

    # ---- ۲. ماتریس همبستگی ----
    plt.figure(figsize=(12,10))
    corr = df[[target] + predictors].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('ماتریس همبستگی پیرسون')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'), dpi=300)
    plt.close()
    print("   ✅ correlation_matrix.png")

    # ---- ۳. پیش‌بینی با فاصله اطمینان ----
    pred_path = os.path.join(OUTPUT_DIR, 'predictions.csv')
    if os.path.exists(pred_path):
        pred_df = pd.read_csv(pred_path)
        plt.figure(figsize=(14,6))
        plt.plot(pred_df['year'], pred_df['actual'], 'b-', linewidth=2, label='واقعی')
        plt.plot(pred_df['year'], pred_df['predicted_lr'], 'r--', linewidth=2, label='پیش‌بینی خطی')
        plt.fill_between(pred_df['year'], pred_df['pred_ci_lower'], pred_df['pred_ci_upper'],
                         color='red', alpha=0.15, label='CI 95%')
        plt.xlabel('سال')
        plt.ylabel(target)
        plt.title('پیش‌بینی تراز خزر با فاصله اطمینان')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_with_ci.png'), dpi=300)
        plt.close()
        print("   ✅ prediction_with_ci.png")
    else:
        print("   ⚠️ فایل predictions.csv وجود ندارد، نمودار پیش‌بینی رسم نشد.")

    # ---- ۴. Pairplot ----
    try:
        plot_vars = [target] + predictors
        plot_data = df[plot_vars].dropna()
        sns.pairplot(plot_data, diag_kind='kde', plot_kws={'alpha': 0.6})
        plt.savefig(os.path.join(OUTPUT_DIR, 'pairplot_analysis.png'), dpi=300)
        plt.close()
        print("   ✅ pairplot_analysis.png")
    except:
        print("   ⚠️ خطا در رسم Pairplot")

    # ---- ۵. Heatmap Spearman/Kendall ----
    corr_methods = ['spearman', 'kendall']
    for method in corr_methods:
        plt.figure(figsize=(12,10))
        corr = df[[target] + predictors].corr(method=method)
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
        plt.title(f'همبستگی {method.capitalize()}')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'correlation_heatmap_{method}.png'), dpi=300)
        plt.close()
        print(f"   ✅ correlation_heatmap_{method}.png")

# ============================================================
# ۷. اجرای اصلی
# ============================================================
if __name__ == "__main__":
    print("=" * 80)
    print("🚀 شروع تحلیل جامع نهایی (نسخه ۶.۲)")
    print("=" * 80)

    try:
        df = load_all_data()
        if df is None or len(df) < 5:
            raise ValueError(f"❌ داده‌های کافی برای تحلیل وجود ندارد! ({len(df) if df is not None else 0} سال)")
        advanced_analysis(df)
        modeling_and_validation(df)
        final_plots(df)
        print("\n" + "=" * 80)
        print("✅ تمام تحلیل‌ها با موفقیت انجام شد!")
        print(f"📂 خروجی‌ها در: {OUTPUT_DIR}")
        print("=" * 80)
    except Exception as e:
        print(f"\n❌ خطای کلی: {e}")
        import traceback
        traceback.print_exc()