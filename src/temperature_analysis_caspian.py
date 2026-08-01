#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
تحلیل جامع دمای سطحی (T2m) برای دریای خزر
شامل: روند، نقاط شکست، رویدادهای حدی، تحلیل فصلی، موجک، ARIMA و ...
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, theilslopes, ttest_ind, norm
from scipy.signal import savgol_filter, periodogram
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.formula.api import quantreg
from dask.diagnostics import ProgressBar
import warnings
warnings.filterwarnings("ignore")

# ============================================
# تنظیمات مسیرها
# ============================================
DATA_DIR = r"M:\ERA5\monthly_average_ERA5_Land"
SHAPEFILE_PATH = r"K:\kazemi\python codes\Caspian_Sea.shp"
OUTPUT_BASE = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\temperature_analysis"
os.makedirs(OUTPUT_BASE, exist_ok=True)

# بازه زمانی
START_YEAR = 1965
END_YEAR = 2025

# متغیرهای دمایی
TEMP_VARS = ['mean_tmin', 'mean_tmax', 'mean_tmean', 'Absolute_tmax', 'Absolute_tmin']
VAR_NAMES_FA = {
    'mean_tmin': 'میانگین دمای حداقل',
    'mean_tmax': 'میانگین دمای حداکثر',
    'mean_tmean': 'میانگین دمای متوسط',
    'Absolute_tmax': 'دمای حداکثر مطلق',
    'Absolute_tmin': 'دمای حداقل مطلق'
}
VAR_NAMES_EN = {
    'mean_tmin': 'Mean Tmin',
    'mean_tmax': 'Mean Tmax',
    'mean_tmean': 'Mean Tmean',
    'Absolute_tmax': 'Absolute Tmax',
    'Absolute_tmin': 'Absolute Tmin'
}

LOW_YEARS = [1976, 1977, 1978]
HIGH_YEARS = [1994, 1995, 1996]

print("="*70)
print("تحلیل جامع دمای سطحی دریای خزر")
print("="*70)

# ============================================
# ۱. بارگذاری و ترکیب فایل‌های NetCDF
# ============================================
print("\n📂 بارگذاری فایل‌های NetCDF...")

# یافتن همه فایل‌های NetCDF
nc_files = sorted(glob.glob(os.path.join(DATA_DIR, "daily_statistics_T2m*.nc")))
if not nc_files:
    raise FileNotFoundError(f"هیچ فایل NetCDF در {DATA_DIR} یافت نشد!")

print(f"   تعداد فایل‌ها: {len(nc_files)}")

# بارگذاری و ترکیب
datasets = []
dates = []
for f in nc_files:
    # استخراج سال و ماه از نام فایل
    match = re.search(r'T2m(\d{4})(\d{2})\.nc', os.path.basename(f))
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        date = pd.Timestamp(year=year, month=month, day=1)
        dates.append(date)
        ds = xr.open_dataset(f)
        datasets.append(ds)
    else:
        print(f"   ⚠️ نام فایل نامعتبر: {os.path.basename(f)}")

if not datasets:
    raise ValueError("هیچ فایل معتبری یافت نشد!")

# ترکیب در طول زمان
ds_combined = xr.concat(datasets, dim='time')
ds_combined = ds_combined.assign_coords(time=dates)
ds_combined = ds_combined.sortby('time')

print(f"   بازه زمانی: {ds_combined.time.min().values} تا {ds_combined.time.max().values}")
print(f"   تعداد گام‌های زمانی: {len(ds_combined.time)}")

# ============================================
# ۲. برش منطقه خزر با Shapefile
# ============================================
print("\n🗺️ برش منطقه خزر...")

import geopandas as gpd
from shapely.geometry import Point

# خواندن Shapefile
gdf = gpd.read_file(SHAPEFILE_PATH)
if len(gdf) > 1:
    gdf['area'] = gdf.geometry.area
    gdf = gdf.sort_values('area', ascending=False).head(1)
main_poly = gdf.geometry.iloc[0]

# ساخت ماسک
def create_mask(da, polygon):
    lon_grid = da.longitude.values
    lat_grid = da.latitude.values
    lon2d, lat2d = np.meshgrid(lon_grid, lat_grid, indexing='ij')
    points = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    mask = np.array([polygon.contains(Point(x, y)) for x, y in points]).reshape(lon2d.shape)
    mask_da = xr.DataArray(mask, dims=['longitude', 'latitude'],
                           coords={'longitude': lon_grid, 'latitude': lat_grid}).transpose('latitude', 'longitude')
    return mask_da

# ساخت ماسک از یک نمونه
sample_da = ds_combined['mean_tmean'].isel(time=0)
mask = create_mask(sample_da, main_poly)
print(f"   تعداد پیکسل‌های داخل دریاچه: {mask.sum().values}")

# اعمال ماسک و محاسبه میانگین فضایی
def apply_mask_and_mean(ds, mask):
    results = {}
    weights = np.cos(np.radians(ds.latitude))
    weights_da = xr.DataArray(weights, dims=['latitude'], coords={'latitude': ds.latitude})
    
    for var in TEMP_VARS:
        if var not in ds.data_vars:
            continue
        da_masked = ds[var].where(mask)
        mean_ts = da_masked.weighted(weights_da).mean(dim=['latitude', 'longitude'])
        results[var] = mean_ts
    return results

with ProgressBar():
    temp_series = apply_mask_and_mean(ds_combined, mask)

# ============================================
# ۳. ساخت سری‌های زمانی
# ============================================
print("\n📊 ساخت سری‌های زمانی...")

# تبدیل به دیتافریم
df_monthly = pd.DataFrame()
df_monthly['time'] = ds_combined.time.values
for var in TEMP_VARS:
    if var in temp_series:
        df_monthly[var] = temp_series[var].values

df_monthly['year'] = df_monthly['time'].dt.year
df_monthly['month'] = df_monthly['time'].dt.month
df_monthly = df_monthly.dropna()

print(f"   سری ماهانه: {df_monthly['year'].min()}-{df_monthly['year'].max()}, {len(df_monthly)} رکورد")

# سری سالانه
df_annual = df_monthly.groupby('year')[TEMP_VARS].mean().reset_index()
df_annual['year'] = df_annual['year'].astype(int)

# ذخیره سری‌ها
df_monthly.to_csv(os.path.join(OUTPUT_BASE, 'temperature_monthly_series.csv'), index=False)
df_annual.to_csv(os.path.join(OUTPUT_BASE, 'temperature_annual_series.csv'), index=False)

print(f"   سری سالانه: {df_annual['year'].min()}-{df_annual['year'].max()}, {len(df_annual)} سال")

# ============================================
# ۴. توابع تحلیلی
# ============================================
def pettitt_test(x):
    n = len(x)
    U = np.zeros(n)
    for t in range(1, n):
        U[t] = U[t-1] + np.sum(np.sign(x[t] - x[:t]))
    k = np.argmax(np.abs(U))
    K = np.max(np.abs(U))
    p_value = 2 * np.exp(-6 * K**2 / (n**3 + n**2))
    return k, p_value

def mann_kendall_test(data):
    n = len(data)
    s = 0
    for i in range(n-1):
        for j in range(i+1, n):
            s += np.sign(data[j] - data[i])
    unique, counts = np.unique(data, return_counts=True)
    tie_sum = sum([c*(c-1)*(2*c+5) for c in counts])
    var_s = (n*(n-1)*(2*n+5) - tie_sum) / 18
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    p_value = 2 * (1 - norm.cdf(abs(z)))
    slopes = []
    for i in range(n-1):
        for j in range(i+1, n):
            if data[j] != data[i]:
                slopes.append((data[j] - data[i]) / (j - i))
    slope = np.median(slopes) if slopes else 0
    return {'slope': slope, 'p_value': p_value, 'trend_sign': 'Increasing' if slope > 0 else 'Decreasing',
            'significant': p_value < 0.05}

def extreme_events(series, threshold_percentile=90):
    threshold = np.percentile(series, threshold_percentile)
    extreme_years = series[series >= threshold].index.tolist()
    extreme_values = series[series >= threshold].values.tolist()
    return {'threshold': threshold, 'extreme_years': extreme_years, 'extreme_values': extreme_values,
            'count': len(extreme_years), 'frequency': len(extreme_years) / len(series) * 100}

def composite_analysis(df_annual, var, low_years, high_years, out_dir):
    filename = f'composite_{var}.png'
    out_path = os.path.join(out_dir, filename)
    if os.path.exists(out_path):
        return None
    df_reg = df_annual[df_annual['year'].isin(low_years + high_years)]
    low_vals = df_reg[df_reg['year'].isin(low_years)][var].dropna()
    high_vals = df_reg[df_reg['year'].isin(high_years)][var].dropna()
    if len(low_vals) < 2 or len(high_vals) < 2:
        return None
    mean_low, std_low = low_vals.mean(), low_vals.std()
    mean_high, std_high = high_vals.mean(), high_vals.std()
    t_stat, p_val = ttest_ind(low_vals, high_vals, equal_var=False)
    res = {'variable': var, 'low_mean': mean_low, 'low_std': std_low,
           'high_mean': mean_high, 'high_std': std_high,
           'diff': mean_high - mean_low, 'p_value': p_val}
    plt.figure(figsize=(8, 6))
    labels = ['Low Level\n(76,77,88)', 'High Level\n(94,95,96)']
    means = [mean_low, mean_high]
    errors = [std_low, std_high]
    bars = plt.bar(labels, means, yerr=errors, capsize=8, color=['coral', 'skyblue'], edgecolor='black')
    plt.ylabel(VAR_NAMES_EN.get(var, var))
    plt.title(f'Composite Comparison - {VAR_NAMES_EN.get(var, var)}\np-value = {p_val:.4f}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, m in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05*abs(bar.get_height()),
                 f'{m:.2f}', ha='center', va='bottom', fontweight='bold')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return res

# ============================================
# ۵. اجرای تحلیل‌ها
# ============================================
print("\n🔍 اجرای تحلیل‌ها...")

results_dir = os.path.join(OUTPUT_BASE, 'results')
os.makedirs(results_dir, exist_ok=True)

# ۵-۱. من-کندال
print("\n   ۵-۱. آزمون من-کندال...")
mk_results = []
for var in TEMP_VARS:
    if var not in df_annual.columns:
        continue
    series = df_annual[var].dropna()
    if len(series) >= 10:
        result = mann_kendall_test(series.values)
        result['variable'] = var
        mk_results.append(result)
        print(f"      {var}: slope={result['slope']:.4f}, p={result['p_value']:.4f}, {result['trend_sign']}")
pd.DataFrame(mk_results).to_csv(os.path.join(results_dir, 'mann_kendall.csv'), index=False)

# ۵-۲. بوت‌استرپ
print("\n   ۵-۲. روند با بوت‌استرپ...")
n_iter = 1000
bootstrap_results = []
for var in TEMP_VARS:
    if var not in df_annual.columns:
        continue
    data = df_annual[['year', var]].dropna()
    if len(data) < 5:
        continue
    years = data['year'].values
    values = data[var].values
    slopes = []
    for _ in range(n_iter):
        idx = resample(range(len(years)), replace=True)
        y_resamp = years[idx]
        v_resamp = values[idx]
        slope = theilslopes(v_resamp, y_resamp)[0]
        slopes.append(slope)
    ci_lower = np.percentile(slopes, 2.5)
    ci_upper = np.percentile(slopes, 97.5)
    main_slope = theilslopes(values, years)[0]
    bootstrap_results.append({'variable': var, 'main_slope': main_slope, 'ci_lower': ci_lower, 'ci_upper': ci_upper})
    plt.figure(figsize=(8,5))
    plt.hist(slopes, bins=30, alpha=0.7, color='skyblue')
    plt.axvline(main_slope, color='r', linestyle='-', label=f'Main slope = {main_slope:.3f}')
    plt.axvline(ci_lower, color='k', linestyle='--', label=f'95% CI: {ci_lower:.3f} to {ci_upper:.3f}')
    plt.axvline(ci_upper, color='k', linestyle='--')
    plt.xlabel('Slope (per year)')
    plt.ylabel('Frequency')
    plt.title(f'Bootstrap trend: {VAR_NAMES_EN.get(var, var)}')
    plt.legend()
    plt.savefig(os.path.join(results_dir, f'bootstrap_{var}.png'), dpi=150)
    plt.close()
pd.DataFrame(bootstrap_results).to_csv(os.path.join(results_dir, 'bootstrap_trend.csv'), index=False)

# ۵-۳. پتیت
print("\n   ۵-۳. آزمون پتیت...")
pettitt_results = []
for var in TEMP_VARS:
    if var not in df_annual.columns:
        continue
    series = df_annual[var].dropna().values
    years = df_annual['year'].values
    if len(series) < 5:
        continue
    k, p = pettitt_test(series)
    cp_year = years[k] if k < len(years) else np.nan
    pettitt_results.append({'variable': var, 'change_year': cp_year, 'p_value': p})
    plt.figure(figsize=(12,5))
    plt.plot(years, series, 'o-', label=var, alpha=0.7)
    plt.axvline(x=cp_year, color='r', linestyle='--', label=f'Change point {cp_year} (p={p:.3f})')
    for yr in LOW_YEARS:
        if yr in years:
            plt.axvline(x=yr, color='orange', linestyle=':', alpha=0.8, linewidth=2)
    for yr in HIGH_YEARS:
        if yr in years:
            plt.axvline(x=yr, color='green', linestyle=':', alpha=0.8, linewidth=2)
    plt.xlabel('Year')
    plt.ylabel(VAR_NAMES_EN.get(var, var))
    plt.title(f'Pettitt test: {VAR_NAMES_EN.get(var, var)}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f'pettitt_{var}.png'), dpi=150)
    plt.close()
pd.DataFrame(pettitt_results).to_csv(os.path.join(results_dir, 'pettitt.csv'), index=False)

# ۵-۴. رگرسیون چندک
print("\n   ۵-۴. رگرسیون چندک...")
try:
    qr_results = []
    for var in TEMP_VARS:
        if var not in df_annual.columns:
            continue
        df_q = df_annual[['year', var]].dropna()
        if len(df_q) < 5:
            continue
        for q in [0.1, 0.5, 0.9]:
            model = quantreg(f"{var} ~ year", data=df_q)
            result = model.fit(q=q)
            qr_results.append({'variable': var, 'quantile': q, 'slope': result.params['year'],
                              'p_value': result.pvalues['year']})
        plt.figure(figsize=(12,5))
        plt.scatter(df_q['year'], df_q[var], alpha=0.5, label='Data')
        x_vals = np.linspace(df_q['year'].min(), df_q['year'].max(), 100)
        for q in [0.1, 0.5, 0.9]:
            model = quantreg(f"{var} ~ year", data=df_q).fit(q=q)
            y_vals = model.params['Intercept'] + model.params['year'] * x_vals
            plt.plot(x_vals, y_vals, label=f'q={q} slope={model.params["year"]:.3f}')
        for yr in LOW_YEARS:
            if yr in df_q['year'].values:
                plt.axvline(x=yr, color='orange', linestyle=':', alpha=0.6)
        for yr in HIGH_YEARS:
            if yr in df_q['year'].values:
                plt.axvline(x=yr, color='green', linestyle=':', alpha=0.6)
        plt.xlabel('Year')
        plt.ylabel(VAR_NAMES_EN.get(var, var))
        plt.title(f'Quantile regression: {VAR_NAMES_EN.get(var, var)}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f'quantile_{var}.png'), dpi=150)
        plt.close()
    pd.DataFrame(qr_results).to_csv(os.path.join(results_dir, 'quantile_regression.csv'), index=False)
except Exception as e:
    print(f"      خطا: {e}")

# ۵-۵. تحلیل مرکب
print("\n   ۵-۵. تحلیل مرکب...")
comp_dir = os.path.join(results_dir, 'composite')
os.makedirs(comp_dir, exist_ok=True)
comp_results = []
for var in TEMP_VARS:
    if var not in df_annual.columns:
        continue
    res = composite_analysis(df_annual, var, LOW_YEARS, HIGH_YEARS, comp_dir)
    if res:
        comp_results.append(res)
if comp_results:
    pd.DataFrame(comp_results).to_csv(os.path.join(results_dir, 'composite_results.csv'), index=False)

# ۵-۶. رویدادهای حدی
print("\n   ۵-۶. رویدادهای حدی...")
for var in TEMP_VARS:
    if var not in df_annual.columns:
        continue
    series = df_annual.set_index('year')[var].dropna()
    if len(series) < 10:
        continue
    high = extreme_events(series, 90)
    low = extreme_events(series, 10)
    low['threshold'] = np.percentile(series, 10)
    low['extreme_years'] = series[series <= low['threshold']].index.tolist()
    low['extreme_values'] = series[series <= low['threshold']].values.tolist()
    low['count'] = len(low['extreme_years'])
    low['frequency'] = low['count'] / len(series) * 100
    with pd.ExcelWriter(os.path.join(results_dir, f'extreme_{var}.xlsx')) as writer:
        pd.DataFrame({'year': high['extreme_years'], 'value': high['extreme_values']}).to_excel(writer, sheet_name='High', index=False)
        pd.DataFrame({'year': low['extreme_years'], 'value': low['extreme_values']}).to_excel(writer, sheet_name='Low', index=False)
        pd.DataFrame([{'variable': var, 'high_count': high['count'], 'low_count': low['count']}]).to_excel(writer, sheet_name='Summary', index=False)
    plt.figure(figsize=(14,5))
    plt.plot(series.index, series.values, 'b-', alpha=0.7, label='Observed')
    plt.scatter(high['extreme_years'], high['extreme_values'], color='red', s=80, label='High', zorder=5)
    plt.scatter(low['extreme_years'], low['extreme_values'], color='blue', s=80, label='Low', zorder=5)
    plt.axhline(high['threshold'], color='red', linestyle='--', alpha=0.5)
    plt.axhline(low['threshold'], color='blue', linestyle='--', alpha=0.5)
    plt.xlabel('Year')
    plt.ylabel(VAR_NAMES_EN.get(var, var))
    plt.title(f'Extreme Events: {VAR_NAMES_EN.get(var, var)}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, f'extreme_{var}.png'), dpi=150)
    plt.close()

# ۵-۷. ARIMA
print("\n   ۵-۷. مدل ARIMA...")
try:
    for var in TEMP_VARS:
        if var not in df_annual.columns:
            continue
        series = df_annual[var].dropna()
        if len(series) < 10:
            continue
        model = ARIMA(series, order=(1,0,1))
        fitted = model.fit()
        forecast = fitted.forecast(steps=5)
        plt.figure(figsize=(12,5))
        plt.plot(series.index, series.values, label='Observed')
        plt.plot(range(len(series), len(series)+5), forecast, 'ro-', label='Forecast')
        for yr in LOW_YEARS:
            if yr in df_annual['year'].values:
                idx = df_annual[df_annual['year'] == yr].index[0]
                plt.axvline(x=idx, color='orange', linestyle=':', alpha=0.6)
        for yr in HIGH_YEARS:
            if yr in df_annual['year'].values:
                idx = df_annual[df_annual['year'] == yr].index[0]
                plt.axvline(x=idx, color='green', linestyle=':', alpha=0.6)
        plt.xlabel('Year index')
        plt.ylabel(VAR_NAMES_EN.get(var, var))
        plt.title(f'ARIMA forecast: {VAR_NAMES_EN.get(var, var)}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f'arima_{var}.png'), dpi=150)
        plt.close()
        pd.DataFrame({'year': range(2026, 2031), 'forecast': forecast}).to_csv(os.path.join(results_dir, f'arima_forecast_{var}.csv'), index=False)
except Exception as e:
    print(f"      خطا: {e}")

# ۵-۸. موجک
print("\n   ۵-۸. تبدیل موجک...")
try:
    import pywt
    for var in TEMP_VARS:
        if var not in df_annual.columns:
            continue
        series = df_annual[var].dropna().values
        if len(series) < 5:
            continue
        scales = np.arange(1, 20)
        coeffs, freqs = pywt.cwt(series, scales, 'morl')
        plt.figure(figsize=(12,5))
        plt.contourf(np.arange(len(series)), scales, np.abs(coeffs), levels=50, cmap='jet')
        plt.colorbar(label='Intensity')
        plt.xlabel('Year index')
        plt.ylabel('Scale')
        plt.title(f'Wavelet power: {VAR_NAMES_EN.get(var, var)}')
        plt.savefig(os.path.join(results_dir, f'wavelet_{var}.png'), dpi=150)
        plt.close()
except Exception as e:
    print(f"      خطا: {e}")

# ۵-۹. STL
print("\n   ۵-۹. تحلیل STL...")
for var in TEMP_VARS:
    if var not in df_monthly.columns:
        continue
    try:
        series = df_monthly.set_index('time')[var].dropna()
        if len(series) < 24:
            continue
        stl = STL(series, period=12, robust=True)
        res = stl.fit()
        fig, axes = plt.subplots(4,1,figsize=(14,10),sharex=True)
        axes[0].plot(res.observed, label='Observed'); axes[0].legend()
        axes[1].plot(res.trend, label='Trend', color='red'); axes[1].legend()
        axes[2].plot(res.seasonal, label='Seasonal', color='green'); axes[2].legend()
        axes[3].plot(res.resid, label='Residual', color='orange'); axes[3].legend()
        axes[3].set_xlabel('Date')
        plt.suptitle(f'STL Decomposition: {VAR_NAMES_EN.get(var, var)}')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'stl_{var}.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f"      خطا برای {var}: {e}")

# ۵-۱۰. تحلیل فصلی
print("\n   ۵-۱۰. تحلیل فصلی...")
df_monthly['season'] = df_monthly['month'].map({
    12:'Winter',1:'Winter',2:'Winter',
    3:'Spring',4:'Spring',5:'Spring',
    6:'Summer',7:'Summer',8:'Summer',
    9:'Autumn',10:'Autumn',11:'Autumn'
})
for var in TEMP_VARS:
    if var not in df_monthly.columns:
        continue
    seasonal_mean = df_monthly.groupby(['year','season'])[var].mean().unstack()
    if seasonal_mean.empty:
        continue
    seasonal_mean.to_csv(os.path.join(results_dir, f'seasonal_mean_{var}.csv'))
    plt.figure(figsize=(10,6))
    df_monthly.boxplot(column=var, by='season')
    plt.title(f'Seasonal Distribution: {VAR_NAMES_EN.get(var, var)}')
    plt.suptitle('')
    plt.xlabel('Season')
    plt.ylabel(VAR_NAMES_EN.get(var, var))
    plt.savefig(os.path.join(results_dir, f'seasonal_boxplot_{var}.png'), dpi=150)
    plt.close()
    plt.figure(figsize=(14,6))
    for season in ['Winter','Spring','Summer','Autumn']:
        if season in seasonal_mean.columns:
            plt.plot(seasonal_mean.index, seasonal_mean[season], 'o-', label=season, alpha=0.7)
    plt.xlabel('Year')
    plt.ylabel(VAR_NAMES_EN.get(var, var))
    plt.title(f'Seasonal Time Series: {VAR_NAMES_EN.get(var, var)}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, f'seasonal_ts_{var}.png'), dpi=150)
    plt.close()

# ۵-۱۱. نقشه حرارتی دهه‌ای
print("\n   ۵-۱۱. نقشه حرارتی دهه‌ای...")
df_monthly['decade'] = (df_monthly['year'] // 10) * 10
decadal_mean = df_monthly.groupby(['decade','month'])[TEMP_VARS].mean().reset_index()
for var in TEMP_VARS:
    if var not in decadal_mean.columns:
        continue
    pivot = decadal_mean.pivot(index='decade', columns='month', values=var)
    if pivot.empty:
        continue
    plt.figure(figsize=(12,6))
    sns.heatmap(pivot, cmap='RdBu_r', center=0, annot=True, fmt='.1f')
    plt.title(f'Decadal mean: {VAR_NAMES_EN.get(var, var)}')
    plt.xlabel('Month')
    plt.ylabel('Decade')
    plt.savefig(os.path.join(results_dir, f'decadal_{var}.png'), dpi=150)
    plt.close()

# ============================================
# ۶. تولید گزارش خلاصه
# ============================================
print("\n📄 تولید گزارش خلاصه...")
with open(os.path.join(OUTPUT_BASE, 'analysis_summary.txt'), 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("گزارش تحلیل دمای سطحی دریای خزر\n")
    f.write("="*70 + "\n\n")
    f.write(f"بازه زمانی: {df_annual['year'].min()}-{df_annual['year'].max()}\n")
    f.write(f"متغیرها: {', '.join(TEMP_VARS)}\n\n")
    f.write("نتایج من-کندال:\n")
    for _, row in pd.DataFrame(mk_results).iterrows():
        var = VAR_NAMES_EN.get(row['variable'], row['variable'])
        sig = '*' if row['significant'] else ''
        f.write(f"  {var}: slope={row['slope']:.4f}{sig}, p={row['p_value']:.4f}\n")
    f.write("\nنکته: * نشان‌دهنده روند معنی‌دار (p<0.05) است.\n")

print("\n✅ تمام تحلیل‌های دمای سطحی با موفقیت انجام شد.")
print(f"📂 خروجی‌ها در: {OUTPUT_BASE}")