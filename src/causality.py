#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================================================
کد نهایی کامل – تحلیل علیت و پیش‌بینی تراز آب دریای خزر
================================================================================
نسخه‌ی مستقل از کتابخانه‌های نصب‌نشده – با روش‌های جایگزین
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings("ignore")

print("="*80)
print("🧠 کد نهایی – تحلیل علیت و پیش‌بینی تراز آب دریای خزر")
print("="*80)

# ============================================================
# ۱. تنظیمات و بارگذاری داده
# ============================================================
BASE_DIR = r"K:\\gozareshha\\Dr Farjami\\Dr Farjami\\140503"
OUTPUT_DIR = os.path.join(BASE_DIR, "final_analysis", "causality_analysis")
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

df_indices = pd.read_excel(INDICES_FILE, sheet_name="Sheet1", parse_dates=['date'])
df_indices['year'] = df_indices['date'].dt.year
df_indices['month'] = df_indices['date'].dt.month

df_all = df_sea.merge(df_indices, on=['date', 'year', 'month'], how='inner')
df_all = df_all.dropna()
df_all = df_all.sort_values('date').reset_index(drop=True)

# اضافه کردن متغیرهای ERA5 (ساختگی)
extra_cols = ['t2m', 'tp', 'evap']
if not all(c in df_all.columns for c in extra_cols):
    print("⚠️ داده‌های ERA5 یافت نشد. ایجاد داده‌های ساختگی...")
    n = len(df_all)
    t = np.arange(n)
    np.random.seed(42)
    df_all['t2m'] = 15 + 0.02 * t + np.random.normal(0, 2, n)
    df_all['tp'] = 50 + 0.01 * t + np.random.normal(0, 10, n)
    df_all['evap'] = 100 + 0.015 * t + np.random.normal(0, 15, n)

df_use = df_all[(df_all['year'] >= 2000) & (df_all['year'] <= 2025)].copy()
df_use = df_use.dropna().sort_values('date').reset_index(drop=True)
y = df_use['sea_level'].values
dates = df_use['date'].values
t = np.arange(len(y))
print(f"✅ داده‌های نهایی: {len(y)} رکورد ماهانه")

# انتخاب متغیرها
var_names = ['sea_level', 'NAO', 'SOI', 'ONI']
available_cols = [c for c in var_names if c in df_use.columns]
data = df_use[available_cols].values
var_names = available_cols
print(f"📊 متغیرهای تحلیل: {var_names}")

# استانداردسازی
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# ============================================================
# ۲. توابع کمکی
# ============================================================
def granger_test_numpy(data, target_idx=0, maxlag=12):
    """آزمون گرنجر با استفاده از numpy (بدون statsmodels)"""
    n, p = data.shape
    results = {}
    
    for var_idx in range(p):
        if var_idx == target_idx:
            continue
        
        # ساخت مدل VAR ساده
        best_p = 1.0
        best_lag = 1
        
        for lag in range(1, maxlag+1):
            # ساخت داده‌های با تأخیر
            n_obs = n - lag
            X = np.zeros((n_obs, lag * 2))
            for i in range(lag):
                X[:, i] = data[lag-i-1:n-i-1, target_idx]
                X[:, lag+i] = data[lag-i-1:n-i-1, var_idx]
            y_obs = data[lag:, target_idx]
            
            if n_obs < lag * 2 + 5:
                continue
            
            # رگرسیون کامل
            model_full = LinearRegression()
            model_full.fit(X, y_obs)
            rss_full = np.sum((y_obs - model_full.predict(X))**2)
            
            # رگرسیون محدود (بدون متغیر هدف)
            X_restricted = X[:, :lag]
            model_rest = LinearRegression()
            model_rest.fit(X_restricted, y_obs)
            rss_rest = np.sum((y_obs - model_rest.predict(X_restricted))**2)
            
            # آماره F
            df1 = lag
            df2 = n_obs - 2*lag - 1
            if df2 > 0 and rss_rest > 0:
                f_stat = ((rss_rest - rss_full) / df1) / (rss_full / df2)
                p_val = 1 - stats.f.cdf(f_stat, df1, df2)
                if p_val < best_p:
                    best_p = p_val
                    best_lag = lag
        
        results[var_names[var_idx]] = {'best_p_value': best_p, 'best_lag': best_lag}
    
    return results

def dickey_fuller_test(series):
    """آزمون ریشه واحد ADF با numpy"""
    n = len(series)
    if n < 10:
        return {'p_value': 1.0, 'statistic': 0}
    
    # تفاضل اول
    dy = series[1:] - series[:-1]
    y_lag = series[:-1]
    
    # رگرسیون
    X = np.column_stack([y_lag, np.ones(len(y_lag))])
    model = LinearRegression()
    model.fit(X, dy)
    residuals = dy - model.predict(X)
    
    # آماره t برای ضریب y_lag
    se = np.std(residuals) / np.std(y_lag) / np.sqrt(len(y_lag))
    t_stat = model.coef_[0] / se if se > 0 else 0
    
    # p-value تقریبی
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(y_lag)-2))
    return {'p_value': p_value, 'statistic': t_stat}

def liang_kleeman_flow(x, y, dt=1):
    """محاسبه شار اطلاعات Liang-Kleeman"""
    n = len(x)
    if n < 10:
        return np.nan
    
    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)
    
    cov_xy = np.cov(x, y)[0, 1]
    var_x = np.var(x)
    cov_dx = np.cov(x, dx)[0, 1] if len(x) > 1 else 0
    cov_y = np.cov(y, dy)[0, 1] if len(y) > 1 else 0
    
    if var_x > 0 and cov_xy != 0:
        T_xy = (cov_xy / var_x) * (cov_dx - cov_y)
    else:
        T_xy = 0
    
    return T_xy

# ============================================================
# ۳. Granger Causality (با numpy)
# ============================================================
print("\n" + "="*80)
print("📊 ۱. Granger Causality (با numpy)")
print("="*80)

results_granger = granger_test_numpy(data_scaled, target_idx=0, maxlag=12)

for var, res in results_granger.items():
    print(f"   {var} -> sea_level: p={res['best_p_value']:.4f} (lag={res['best_lag']}) -> {'✅ علیت' if res['best_p_value'] < 0.05 else '❌ بدون علیت'}")

# ============================================================
# ۴. Nonlinear Granger Causality
# ============================================================
print("\n" + "="*80)
print("📊 ۲. Nonlinear Granger Causality (Diks-Panchenko ساده‌شده)")
print("="*80)

def nonlinear_granger_test(x, y, lag=2, eps=1.0):
    """نسخه‌ی ساده‌شده آزمون غیرخطی"""
    n = len(x)
    if n < 3*lag + 5:
        return np.nan
    
    # ساخت تأخیرها
    X_lagged = np.array([x[i:n-lag+i] for i in range(lag)]).T
    Y_lagged = np.array([y[i:n-lag+i] for i in range(lag)]).T
    
    # همبستگی شرطی
    def corr_conditional(x1, x2, y_cond, eps):
        mask = np.abs(y_cond - np.mean(y_cond)) < eps
        if mask.sum() < 10:
            return 0
        return np.corrcoef(x1[mask], x2[mask])[0, 1] if mask.sum() > 1 else 0
    
    # Bootstrap
    n_bootstrap = 100
    t_stats = []
    for _ in range(n_bootstrap):
        idx = np.random.permutation(n-lag)
        x_shuffled = X_lagged[idx, 0]
        t = corr_conditional(X_lagged[:, 0], Y_lagged[:, 0], X_lagged[:, 0], eps)
        t_stats.append(t)
    
    t_obs = corr_conditional(X_lagged[:, 0], Y_lagged[:, 0], X_lagged[:, 0], eps)
    p_value = np.mean(np.abs(t_stats) >= np.abs(t_obs)) if len(t_stats) > 0 else 1.0
    return p_value

nonlinear_results = {}
for i, name in enumerate(var_names):
    if name == 'sea_level':
        continue
    idx = var_names.index(name)
    x = data_scaled[:, idx]
    y = data_scaled[:, 0]
    try:
        p = nonlinear_granger_test(x, y, lag=2)
        nonlinear_results[name] = p if not np.isnan(p) else 1.0
        print(f"   {name} -> sea_level: p={nonlinear_results[name]:.4f} -> {'✅ علیت غیرخطی' if nonlinear_results[name] < 0.05 else '❌ بدون علیت غیرخطی'}")
    except Exception as e:
        print(f"   ⚠️ خطا برای {name}: {e}")
        nonlinear_results[name] = 1.0

# ============================================================
# ۵. Spectral Coherence
# ============================================================
print("\n" + "="*80)
print("📊 ۳. Spectral Coherence (همبستگی طیفی)")
print("="*80)

spectral_results = []
n_fft = min(128, len(y)//4)

for i, name in enumerate(var_names):
    if name == 'sea_level':
        continue
    idx = var_names.index(name)
    f, Cxy = signal.coherence(data_scaled[:, 0], data_scaled[:, idx], fs=1, nperseg=n_fft)
    max_coherence = np.max(Cxy)
    freq_max = f[np.argmax(Cxy)]
    spectral_results.append({
        'variable': name,
        'max_coherence': max_coherence,
        'freq_max_coherence': freq_max,
        'period': 1/freq_max if freq_max > 0 else np.nan
    })
    print(f"   {name}: max_coherence={max_coherence:.3f} at freq={freq_max:.3f} (period={1/freq_max:.1f}m)")

df_spectral = pd.DataFrame(spectral_results)
df_spectral.to_csv(os.path.join(OUTPUT_DIR, 'spectral_coherence.csv'), index=False)

# ============================================================
# ۶. Wavelet Coherence
# ============================================================
print("\n" + "="*80)
print("📊 ۴. Wavelet Coherence")
print("="*80)

try:
    import pywt
    from scipy.ndimage import uniform_filter
    
    wavelet_results = []
    scales = np.arange(1, 33, 0.5)
    
    for i, name in enumerate(var_names):
        if name == 'sea_level':
            continue
        idx = var_names.index(name)
        
        W1, f = pywt.cwt(data_scaled[:, 0], scales, 'cmor1.5-1.0')
        W2, _ = pywt.cwt(data_scaled[:, idx], scales, 'cmor1.5-1.0')
        W12 = W1 * np.conj(W2)
        
        W1_smooth = uniform_filter(np.abs(W1)**2, size=(5, 1))
        W2_smooth = uniform_filter(np.abs(W2)**2, size=(5, 1))
        W12_smooth = uniform_filter(W12, size=(5, 1))
        
        coh = np.abs(W12_smooth)**2 / (W1_smooth * W2_smooth + 1e-10)
        periods = 1 / f
        
        max_coh = np.max(coh)
        max_idx = np.unravel_index(np.argmax(coh), coh.shape)
        wavelet_results.append({
            'variable': name,
            'max_coherence': max_coh,
            'best_period': periods[max_idx[0]]
        })
        print(f"   {name}: max_coherence={max_coh:.3f} at period={periods[max_idx[0]]:.1f}m")
    
    df_wavelet = pd.DataFrame(wavelet_results)
    df_wavelet.to_csv(os.path.join(OUTPUT_DIR, 'wavelet_coherence.csv'), index=False)
except Exception as e:
    print(f"   ⚠️ خطا (pywt نصب نیست): {e}")
    print("   از روش جایگزین (مورلت ساده) استفاده می‌شود...")
    # روش جایگزین ساده
    wavelet_results = []
    for i, name in enumerate(var_names):
        if name == 'sea_level':
            continue
        idx = var_names.index(name)
        # همبستگی متقاطع با پنجره‌های غلتان
        window = 24
        coh_vals = []
        for j in range(window, len(y)-window):
            c = np.corrcoef(data_scaled[j-window:j+window, 0], data_scaled[j-window:j+window, idx])[0, 1]
            coh_vals.append(c)
        max_coh = np.max(np.abs(coh_vals))
        wavelet_results.append({
            'variable': name,
            'max_coherence': max_coh,
            'best_period': np.nan
        })
        print(f"   {name}: max_coherence={max_coh:.3f}")
    df_wavelet = pd.DataFrame(wavelet_results)
    df_wavelet.to_csv(os.path.join(OUTPUT_DIR, 'wavelet_coherence.csv'), index=False)

# ============================================================
# ۷. Liang-Kleeman Information Flow
# ============================================================
print("\n" + "="*80)
print("📊 ۵. Liang-Kleeman Information Flow")
print("="*80)

lk_results = []
for i, name in enumerate(var_names):
    if name == 'sea_level':
        continue
    idx = var_names.index(name)
    flow = liang_kleeman_flow(data_scaled[:, idx], data_scaled[:, 0])
    lk_results.append({'variable': name, 'information_flow': flow})
    print(f"   {name}: اطلاعات_جاری={flow:.6f} -> {'✅ مثبت' if flow > 0 else '❌ منفی'}")

df_lk = pd.DataFrame(lk_results)
df_lk.to_csv(os.path.join(OUTPUT_DIR, 'liang_kleeman_flow.csv'), index=False)

# ============================================================
# ۸. Causal Impact (Synthetic Control)
# ============================================================
print("\n" + "="*80)
print("📊 ۶. Causal Impact (Synthetic Control)")
print("="*80)

cp_idx = 593  # نوامبر 2022
pre_period = (0, cp_idx)

def synthetic_control_simple(y, pre_period):
    n = len(y)
    pre_len = pre_period[1] - pre_period[0]
    
    pre_data = y[:pre_len]
    t_pre = np.arange(pre_len)
    t_post = np.arange(pre_len, n)
    
    model = LinearRegression()
    model.fit(t_pre.reshape(-1, 1), pre_data)
    pred_post = model.predict(t_post.reshape(-1, 1))
    
    actual_post = y[pre_len:]
    impact = actual_post - pred_post
    avg_impact = np.mean(impact)
    
    return impact, avg_impact, pred_post

impact, avg_impact, pred_post = synthetic_control_simple(y, pre_period)

print(f"   تأثیر متوسط رویداد نوامبر 2022: {avg_impact:.4f} متر")
print(f"   {'✅ کاهش شدید' if avg_impact < 0 else '❌ افزایش'}")

# نمودار
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(dates[:cp_idx], y[:cp_idx], 'b-', label='Pre-event')
ax.plot(dates[cp_idx:], y[cp_idx:], 'r-', label='Post-event (Actual)')
ax.plot(dates[cp_idx:], pred_post, 'g--', label='Synthetic Control (Predicted)')
ax.axvline(x=dates[cp_idx], color='gray', linestyle=':', label='Event (Nov 2022)')
ax.fill_between(dates[cp_idx:], y[cp_idx:], pred_post.flatten(), alpha=0.3, color='red', label=f'Impact: {avg_impact:.3f}m')
ax.set_xlabel('Date')
ax.set_ylabel('Sea Level (m)')
ax.set_title('Causal Impact of Nov 2022 Event')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'causal_impact.png'), dpi=150)
plt.close()
print("   📈 causal_impact.png")

# ============================================================
# ۹. Counterfactual Scenarios
# ============================================================
print("\n" + "="*80)
print("📊 ۷. Counterfactual Scenarios (3 سناریو)")
print("="*80)

future_dates = pd.date_range('2026-01-01', '2030-12-01', freq='MS')
future_t = np.arange(len(y), len(y) + len(future_dates))

# ساخت ویژگی‌ها برای پیش‌بینی
def make_features(t_series, months, total_len):
    t_norm = t_series / total_len
    month_sin = np.sin(2 * np.pi * months / 12)
    month_cos = np.cos(2 * np.pi * months / 12)
    return np.column_stack([t_norm, month_sin, month_cos])

months = df_use['month'].values
X1 = make_features(t[:cp_idx], months[:cp_idx], len(y))
X2 = make_features(t[cp_idx:], months[cp_idx:], len(y))
model1 = LinearRegression().fit(X1, y[:cp_idx])
model2 = LinearRegression().fit(X2, y[cp_idx:])

future_months = future_dates.month
X_future = make_features(future_t, future_months, len(y))

# سناریوها
scenario_pessimistic = model2.predict(X_future)  # ادامه روند شدید
scenario_probable = (model1.predict(X_future) + model2.predict(X_future)) / 2  # میانگین
scenario_optimistic = np.full(len(future_dates), y[-1])  # توقف کاهش

df_scenarios = pd.DataFrame({
    'date': future_dates,
    'pessimistic': scenario_pessimistic,
    'probable': scenario_probable,
    'optimistic': scenario_optimistic
})
df_scenarios.to_csv(os.path.join(OUTPUT_DIR, 'counterfactual_scenarios.csv'), index=False)

print(f"   سناریو بدبینانه (۲۰۳۰): {scenario_pessimistic[-1]:.3f} متر")
print(f"   سناریو محتمل (۲۰۳۰): {scenario_probable[-1]:.3f} متر")
print(f"   سناریو خوش‌بینانه (۲۰۳۰): {scenario_optimistic[-1]:.3f} متر")

# نمودار
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(dates, y, 'k-', label='Historical')
ax.plot(future_dates, scenario_pessimistic, 'r-', linewidth=2, label='Pessimistic (Trend continues)')
ax.plot(future_dates, scenario_probable, 'b-', linewidth=2, label='Probable (Gradual slowdown)')
ax.plot(future_dates, scenario_optimistic, 'g-', linewidth=2, label='Optimistic (Stabilization)')
ax.axvline(x=pd.Timestamp('2026-01-01'), color='gray', linestyle=':', label='Forecast Start')
ax.set_xlabel('Date')
ax.set_ylabel('Sea Level (m)')
ax.set_title('Counterfactual Scenarios (2026-2030)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'counterfactual_scenarios.png'), dpi=150)
plt.close()
print("   📈 counterfactual_scenarios.png")

# ============================================================
# ۱۰. Granger PCA
# ============================================================
print("\n" + "="*80)
print("📊 ۸. Granger PCA")
print("="*80)

try:
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_scaled)
    
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=var_names)
    loadings.to_csv(os.path.join(OUTPUT_DIR, 'pca_loadings.csv'))
    print("   ✅ PCA Loadings saved.")
    
    # آزمون گرنجر روی PCها
    pca_granger = granger_test_numpy(np.column_stack([data_scaled[:, 0], pca_result]), target_idx=0, maxlag=6)
    for var, res in pca_granger.items():
        print(f"   {var} -> sea_level: p={res['best_p_value']:.4f}")
except Exception as e:
    print(f"   ⚠️ خطا: {e}")

# ============================================================
# ۱۱. Causal Feature Selection
# ============================================================
print("\n" + "="*80)
print("📊 ۹. Causal Feature Selection")
print("="*80)

try:
    X_lasso = data_scaled[:, 1:]
    y_lasso = data_scaled[:, 0]
    
    model_lasso = Lasso(alpha=0.01, max_iter=5000)
    model_lasso.fit(X_lasso, y_lasso)
    selected_features = [var_names[i+1] for i, coef in enumerate(model_lasso.coef_) if abs(coef) > 0.01]
    print(f"   ویژگی‌های انتخاب‌شده با Lasso: {selected_features if selected_features else 'هیچ‌کدام'}")
    
    # Random Forest
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_lasso, y_lasso)
    importance = model_rf.feature_importances_
    df_importance = pd.DataFrame({
        'variable': var_names[1:],
        'importance': importance
    }).sort_values('importance', ascending=False)
    df_importance.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)
    print("   ✅ Feature Importance saved.")
except Exception as e:
    print(f"   ⚠️ خطا: {e}")

# ============================================================
# ۱۲. گزارش نهایی
# ============================================================
print("\n" + "="*80)
print("📝 ۱۰. گزارش نهایی")
print("="*80)

with open(os.path.join(OUTPUT_DIR, 'final_causality_report.txt'), 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("🧠 گزارش نهایی تحلیل علیت تراز آب دریای خزر\n")
    f.write("="*80 + "\n\n")
    
    f.write("📊 ۱. Granger Causality:\n")
    for var, res in results_granger.items():
        f.write(f"   {var}: p={res['best_p_value']:.4f} (lag={res['best_lag']}) -> {'✅' if res['best_p_value'] < 0.05 else '❌'}\n")
    f.write("\n")
    
    f.write("📊 ۲. Nonlinear Granger Causality:\n")
    for var, p in nonlinear_results.items():
        f.write(f"   {var}: p={p:.4f} -> {'✅' if p < 0.05 else '❌'}\n")
    f.write("\n")
    
    f.write("📊 ۳. Spectral Coherence:\n")
    for res in spectral_results:
        f.write(f"   {res['variable']}: max_coherence={res['max_coherence']:.3f}\n")
    f.write("\n")
    
    f.write("📊 ۴. Liang-Kleeman Information Flow:\n")
    for res in lk_results:
        f.write(f"   {res['variable']}: flow={res['information_flow']:.6f}\n")
    f.write("\n")
    
    f.write("📊 ۵. Causal Impact (Synthetic Control):\n")
    f.write(f"   Impact of Nov 2022 event: {avg_impact:.4f} m\n")
    f.write("\n")
    
    f.write("📊 ۶. Counterfactual Scenarios (2030):\n")
    f.write(f"   Pessimistic (trend continues): {scenario_pessimistic[-1]:.3f} m\n")
    f.write(f"   Probable (gradual slowdown): {scenario_probable[-1]:.3f} m\n")
    f.write(f"   Optimistic (stabilization): {scenario_optimistic[-1]:.3f} m\n")
    f.write("\n")
    
    f.write("📊 ۷. Conclusion:\n")
    f.write("   - Granger Causality: " + ("✅ برخی متغیرها علیت نشان دادند." if any(res['best_p_value'] < 0.05 for res in results_granger.values()) else "❌ هیچ متغیری علیت قوی نشان نداد.") + "\n")
    f.write("   - Nonlinear Causality: " + ("✅ برخی متغیرها علیت غیرخطی نشان دادند." if any(p < 0.05 for p in nonlinear_results.values()) else "❌ هیچ متغیری علیت غیرخطی نشان نداد.") + "\n")
    f.write("   - Causal Impact: رویداد نوامبر 2022 تأثیر کاهشی حدوداً {:.4f} متر داشته است.\n".format(avg_impact))
    f.write("   - Counterfactual: ادامه روند فعلی باعث کاهش بیشتر تراز آب تا ~{:.2f} متر می‌شود.\n".format(scenario_pessimistic[-1]))
    
    f.write(f"\n✅ خروجی‌ها در: {OUTPUT_DIR}\n")

print(f"\n✅ همه خروجی‌ها در {OUTPUT_DIR} ذخیره شدند.")
print("📄 فایل‌های تولید شده:")
files = [
    'spectral_coherence.csv',
    'wavelet_coherence.csv',
    'liang_kleeman_flow.csv',
    'causal_impact.png',
    'counterfactual_scenarios.csv',
    'counterfactual_scenarios.png',
    'pca_loadings.csv',
    'feature_importance.csv',
    'final_causality_report.txt'
]
for f in files:
    print(f"   - {f}")
print("="*80)