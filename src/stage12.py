#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================================================
پیش‌بینی نهایی تراز آب دریای خزر (مدل ساده اما معتبر)
================================================================================
- استفاده از داده‌های سالانه (۱۹۹۲–۲۰۲۵)
- مدل رگرسیون خطی (Trend) + ARIMA
- محاسبه فاصله اطمینان با Bootstrap
- خروجی: پیش‌بینی ۲۰۲۶–۲۰۳۰ با خطای استاندارد
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = r"K:\\gozareshha\\Dr Farjami\\Dr Farjami\\140503"
OUTPUT_DIR = os.path.join(BASE_DIR, "final_analysis", "final_forecast")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEA_LEVEL_FILE = os.path.join(
    BASE_DIR,
    "Final_Analysis_Archive_20260702_060114",
    "basin_border",
    "caspian_unified_analysis",
    "caspian_sea_level_raw.csv"
)

# بارگذاری داده
df_sea = pd.read_csv(SEA_LEVEL_FILE, sep=';', parse_dates=['datetime'])
df_sea['year'] = df_sea['datetime'].dt.year
df_annual = df_sea.groupby('year')['wse'].mean().reset_index()
df_annual.rename(columns={'wse': 'sea_level'}, inplace=True)

# داده‌های ۲۰۰۰–۲۰۲۵
df_train = df_annual[df_annual['year'] >= 2000].copy()
X = df_train['year'].values.reshape(-1, 1)
y = df_train['sea_level'].values

# مدل رگرسیون خطی
model_lr = LinearRegression()
model_lr.fit(X, y)
print(f"✅ رگرسیون خطی: R² = {model_lr.score(X, y):.4f}")
print(f"   شیب = {model_lr.coef_[0]:.4f} متر/سال")
print(f"   عرض از مبدأ = {model_lr.intercept_:.2f} متر")

# مدل ARIMA
model_arima = ARIMA(y, order=(1,1,1)).fit()
print(f"✅ ARIMA: AIC = {model_arima.aic:.2f}")

# پیش‌بینی ۲۰۲۶–۲۰۳۰
future_years = np.arange(2026, 2031).reshape(-1, 1)
pred_lr = model_lr.predict(future_years)
pred_arima = model_arima.forecast(steps=5)

# Bootstrap برای فاصله اطمینان
n_bootstrap = 1000
predictions = []
for _ in range(n_bootstrap):
    idx = np.random.choice(len(y), len(y), replace=True)
    X_boot = X[idx].reshape(-1, 1)
    y_boot = y[idx]
    model = LinearRegression()
    model.fit(X_boot, y_boot)
    predictions.append(model.predict(future_years))

pred_array = np.array(predictions)
ci_lower = np.percentile(pred_array, 2.5, axis=0)
ci_upper = np.percentile(pred_array, 97.5, axis=0)

# ذخیره نتایج
df_forecast = pd.DataFrame({
    'year': future_years.flatten(),
    'Linear_Trend': pred_lr,
    'ARIMA': pred_arima,
    'CI_2.5%': ci_lower,
    'CI_97.5%': ci_upper
})
df_forecast.to_csv(os.path.join(OUTPUT_DIR, 'final_forecast.csv'), index=False)

# نمودار
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_train['year'], y, 'ko-', label='Historical (2000-2025)')
ax.plot(future_years, pred_lr, 'ro-', label='Linear Trend')
ax.plot(future_years, pred_arima, 'bo-', label='ARIMA')
ax.fill_between(future_years.flatten(), ci_lower, ci_upper, alpha=0.2, color='red', label='95% CI (Bootstrap)')
ax.axvline(x=2025.5, color='gray', linestyle='--', label='Forecast Start')
ax.set_xlabel('Year')
ax.set_ylabel('Sea Level (m)')
ax.set_title('Final Forecast of Caspian Sea Level (2026-2030)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'final_forecast.png'), dpi=150)
plt.close()

print(f"\n📊 پیش‌بینی نهایی:")
print(df_forecast.to_string(index=False))
print(f"\n✅ خروجی‌ها در: {OUTPUT_DIR}")