# -*- coding: utf-8 -*-
"""
آموزش روی Extended + رگرسیون ریج + اعتبارسنجی روی RAPID/OSNAP
نسخه اصلاح‌شده - رفع خطای KeyError: 'date'
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import os
import warnings
warnings.filterwarnings('ignore')

# تنظیمات
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

# ایجاد پوشه خروجی
output_dir = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\Final_Results_Ridge_Extended"
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("📊 آموزش روی Extended + رگرسیون ریج + اعتبارسنجی روی RAPID/OSNAP")
print("="*70)

# ==============================================
# 1. تعریف مسیرها
# ==============================================

paths = {
    'rapid': r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\amoc_rapid_RAPID (12 month mean).csv",
    'osnap': r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\amoc_osnap_OSNAP (12 month mean).csv",
    'extended': r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\final_comprehensive_output\AMOC_extended_1870_2023.csv",
    'enso': r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\indices_complete.xlsx"
}

# ==============================================
# 2. توابع بارگذاری
# ==============================================

def load_amoc_rapid(path):
    df = pd.read_csv(path, comment='#')
    df = df.dropna(subset=['RAPID (12 month mean) (Sv)'])
    df['date'] = pd.to_datetime(df['Year'].astype(int).astype(str) + '-' + 
                                df['Month'].astype(int).astype(str) + '-01')
    df = df.rename(columns={'RAPID (12 month mean) (Sv)': 'AMOC'})
    df['year'] = df['date'].dt.year
    return df[['date', 'year', 'AMOC']]

def load_amoc_osnap(path):
    df = pd.read_csv(path, comment='#')
    df = df.dropna(subset=['OSNAP (12 month mean) (Sv)'])
    df['date'] = pd.to_datetime(df['Year'].astype(int).astype(str) + '-' + 
                                df['Month'].astype(int).astype(str) + '-01')
    df = df.rename(columns={'OSNAP (12 month mean) (Sv)': 'AMOC'})
    df['year'] = df['date'].dt.year
    return df[['date', 'year', 'AMOC']]

def load_amoc_extended(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['year'].astype(int).astype(str) + '-07-01')
    df = df.rename(columns={'amoc_reconstructed': 'AMOC'})
    df['year'] = df['date'].dt.year
    return df[['date', 'year', 'AMOC']]

# ==============================================
# 3. بارگذاری داده‌ها
# ==============================================

print("\n🔄 بارگذاری داده‌ها...")
amoc_rapid = load_amoc_rapid(paths['rapid'])
amoc_osnap = load_amoc_osnap(paths['osnap'])
amoc_extended = load_amoc_extended(paths['extended'])
print(f"✅ RAPID: {len(amoc_rapid)} ماه")
print(f"✅ OSNAP: {len(amoc_osnap)} ماه")
print(f"✅ Extended: {len(amoc_extended)} سال")

# بارگذاری شاخص‌ها
enso_df = pd.read_excel(paths['enso'], sheet_name='Sheet1')
enso_df['date'] = pd.to_datetime(enso_df['date'])
enso_df['year'] = enso_df['date'].dt.year
print(f"✅ شاخص‌ها: {len(enso_df)} ماه")

# ==============================================
# 4. تبدیل شاخص‌ها به سالانه
# ==============================================

print("\n🔄 تبدیل شاخص‌ها به سالانه...")
all_features = [col for col in enso_df.columns if col not in ['date', 'year']]
enso_yearly = enso_df.groupby('year')[all_features].mean().reset_index()
print(f"✅ شاخص‌های سالانه: {len(enso_yearly)} سال")

# انتخاب شاخص‌های کلیدی (همه شاخص‌ها)
features_for_model = all_features
print(f"تعداد شاخص‌های مورد استفاده: {len(features_for_model)}")

# ==============================================
# 5. ایجاد داده‌های آموزش (Extended) و آزمون (RAPID/OSNAP)
# ==============================================

print("\n🔄 ایجاد داده‌های آموزش و آزمون...")

# داده‌های آموزش (Extended)
extended_merged = pd.merge(amoc_extended, enso_yearly, on='year', how='inner')
extended_merged = extended_merged.dropna()
print(f"✅ Extended: {len(extended_merged)} سال ({extended_merged['year'].min()}-{extended_merged['year'].max()})")

X_train = extended_merged[features_for_model]
y_train = extended_merged['AMOC']
dates_train = extended_merged['date']

# داده‌های آزمون (RAPID به صورت سالانه)
rapid_yearly = amoc_rapid.groupby('year')['AMOC'].mean().reset_index()
rapid_merged = pd.merge(rapid_yearly, enso_yearly, on='year', how='inner')
rapid_merged = rapid_merged.dropna()
# ایجاد ستون تاریخ (۱ ژوئیه هر سال)
rapid_merged['date'] = pd.to_datetime(rapid_merged['year'].astype(str) + '-07-01')
print(f"✅ RAPID (سالانه): {len(rapid_merged)} سال ({rapid_merged['year'].min()}-{rapid_merged['year'].max()})")

X_test_rapid = rapid_merged[features_for_model]
y_test_rapid = rapid_merged['AMOC']
dates_test_rapid = rapid_merged['date']

# داده‌های آزمون (OSNAP به صورت سالانه)
osnap_yearly = amoc_osnap.groupby('year')['AMOC'].mean().reset_index()
osnap_merged = pd.merge(osnap_yearly, enso_yearly, on='year', how='inner')
osnap_merged = osnap_merged.dropna()
# ایجاد ستون تاریخ (۱ ژوئیه هر سال)
osnap_merged['date'] = pd.to_datetime(osnap_merged['year'].astype(str) + '-07-01')
print(f"✅ OSNAP (سالانه): {len(osnap_merged)} سال ({osnap_merged['year'].min()}-{osnap_merged['year'].max()})")

X_test_osnap = osnap_merged[features_for_model]
y_test_osnap = osnap_merged['AMOC']
dates_test_osnap = osnap_merged['date']

# ==============================================
# 6. استانداردسازی
# ==============================================

print("\n🔄 استانداردسازی داده‌ها...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=features_for_model, index=X_train.index)

X_test_rapid_scaled = scaler.transform(X_test_rapid)
X_test_rapid_scaled = pd.DataFrame(X_test_rapid_scaled, columns=features_for_model, index=X_test_rapid.index)

X_test_osnap_scaled = scaler.transform(X_test_osnap)
X_test_osnap_scaled = pd.DataFrame(X_test_osnap_scaled, columns=features_for_model, index=X_test_osnap.index)

# ==============================================
# 7. رگرسیون ریج با اعتبارسنجی متقاطع (TimeSeriesSplit)
# ==============================================

print("\n🧠 آموزش رگرسیون ریج با اعتبارسنجی متقاطع (TimeSeriesSplit)...")

# اعتبارسنجی متقاطع سری زمانی (۵ تقسیم)
tscv = TimeSeriesSplit(n_splits=5)
ridge_cv = RidgeCV(
    alphas=np.logspace(-3, 3, 50),
    scoring='neg_mean_squared_error',
    cv=tscv
)
ridge_cv.fit(X_train_scaled, y_train)

best_alpha = ridge_cv.alpha_
print(f"✅ بهترین مقدار alpha: {best_alpha:.4f}")

# مدل نهایی با بهترین alpha
model_ridge = Ridge(alpha=best_alpha)
model_ridge.fit(X_train_scaled, y_train)

# ==============================================
# 8. ارزیابی روی Extended (آموزش)
# ==============================================

y_pred_train = model_ridge.predict(X_train_scaled)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_train = r2_score(y_train, y_pred_train)
mape_train = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100

print(f"\n📊 عملکرد روی Extended (آموزش):")
print(f"   RMSE: {rmse_train:.3f} Sv")
print(f"   R²: {r2_train:.3f}")
print(f"   MAPE: {mape_train:.2f}%")

# ==============================================
# 9. ارزیابی روی RAPID (سالانه)
# ==============================================

y_pred_rapid = model_ridge.predict(X_test_rapid_scaled)
rmse_rapid = np.sqrt(mean_squared_error(y_test_rapid, y_pred_rapid))
r2_rapid = r2_score(y_test_rapid, y_pred_rapid)
mape_rapid = np.mean(np.abs((y_test_rapid - y_pred_rapid) / y_test_rapid)) * 100

print(f"\n📊 عملکرد روی RAPID (سالانه):")
print(f"   RMSE: {rmse_rapid:.3f} Sv")
print(f"   R²: {r2_rapid:.3f}")
print(f"   MAPE: {mape_rapid:.2f}%")

# ==============================================
# 10. ارزیابی روی OSNAP (سالانه)
# ==============================================

if len(X_test_osnap) > 0:
    y_pred_osnap = model_ridge.predict(X_test_osnap_scaled)
    rmse_osnap = np.sqrt(mean_squared_error(y_test_osnap, y_pred_osnap))
    r2_osnap = r2_score(y_test_osnap, y_pred_osnap)
    mape_osnap = np.mean(np.abs((y_test_osnap - y_pred_osnap) / y_test_osnap)) * 100
    
    print(f"\n📊 عملکرد روی OSNAP (سالانه):")
    print(f"   RMSE: {rmse_osnap:.3f} Sv")
    print(f"   R²: {r2_osnap:.3f}")
    print(f"   MAPE: {mape_osnap:.2f}%")
else:
    print("\n⚠️ داده‌های OSNAP کافی برای ارزیابی وجود ندارد.")
    rmse_osnap = np.nan
    r2_osnap = np.nan

# ==============================================
# 11. اهمیت ویژگی‌ها (ضرایب رگرسیون ریج)
# ==============================================

coefficients = pd.DataFrame({
    'feature': features_for_model,
    'coefficient': model_ridge.coef_
}).sort_values('coefficient', ascending=False)

# اهمیت (قدر مطلق)
coefficients['importance'] = np.abs(coefficients['coefficient'])
importance = coefficients.sort_values('importance', ascending=False)

print("\n🔟 ویژگی برتر (بر اساس ضرایب رگرسیون ریج):")
print(importance.head(10)[['feature', 'coefficient', 'importance']].to_string(index=False))

# ذخیره
importance.to_csv(os.path.join(output_dir, 'Ridge_Coefficients.csv'), index=False)

# ==============================================
# 12. نمودارها
# ==============================================

print("\n📈 رسم نمودارها...")

# نمودار ۱: پیش‌بینی روی Extended (آموزش)
plt.figure(figsize=(14, 6))
plt.plot(dates_train, y_train, 'o-', label='مشاهده شده (Extended)', linewidth=2)
plt.plot(dates_train, y_pred_train, 's--', label='پیش‌بینی (Ridge)', alpha=0.7)
plt.xlabel('سال')
plt.ylabel('AMOC (Sv)')
plt.title(f'پیش‌بینی روی Extended (آموزش)\nR²={r2_train:.3f}, RMSE={rmse_train:.3f} Sv')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(output_dir, 'Extended_Prediction.png'), dpi=300)
plt.show()

# نمودار ۲: پیش‌بینی روی RAPID (آزمون)
plt.figure(figsize=(14, 6))
plt.plot(dates_test_rapid, y_test_rapid, 'o-', label='مشاهده شده (RAPID)', linewidth=2)
plt.plot(dates_test_rapid, y_pred_rapid, 's--', label='پیش‌بینی (Ridge)', alpha=0.7)
plt.xlabel('سال')
plt.ylabel('AMOC (Sv)')
plt.title(f'پیش‌بینی روی RAPID (آزمون)\nR²={r2_rapid:.3f}, RMSE={rmse_rapid:.3f} Sv')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(output_dir, 'RAPID_Prediction.png'), dpi=300)
plt.show()

# نمودار ۳: پیش‌بینی روی OSNAP (آزمون)
if len(X_test_osnap) > 0:
    plt.figure(figsize=(14, 6))
    plt.plot(dates_test_osnap, y_test_osnap, 'o-', label='مشاهده شده (OSNAP)', linewidth=2)
    plt.plot(dates_test_osnap, y_pred_osnap, 's--', label='پیش‌بینی (Ridge)', alpha=0.7)
    plt.xlabel('سال')
    plt.ylabel('AMOC (Sv)')
    plt.title(f'پیش‌بینی روی OSNAP (آزمون)\nR²={r2_osnap:.3f}, RMSE={rmse_osnap:.3f} Sv')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'OSNAP_Prediction.png'), dpi=300)
    plt.show()

# نمودار ۴: اهمیت ویژگی‌ها
plt.figure(figsize=(12, 8))
top_15 = importance.head(15)
plt.barh(top_15['feature'], top_15['importance'], color='steelblue')
plt.xlabel('اهمیت (قدر مطلق ضریب)')
plt.ylabel('ویژگی')
plt.title('۱۵ ویژگی برتر (رگرسیون ریج)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Feature_Importance.png'), dpi=300)
plt.show()

# ==============================================
# 13. ذخیره نتایج پیش‌بینی
# ==============================================

# نتایج Extended
extended_results = pd.DataFrame({
    'year': extended_merged['year'],
    'date': dates_train,
    'AMOC_observed': y_train,
    'AMOC_predicted': y_pred_train
})
extended_results.to_csv(os.path.join(output_dir, 'Extended_Predictions.csv'), index=False)

# نتایج RAPID
rapid_results = pd.DataFrame({
    'year': rapid_merged['year'],
    'date': dates_test_rapid,
    'AMOC_observed': y_test_rapid,
    'AMOC_predicted': y_pred_rapid
})
rapid_results.to_csv(os.path.join(output_dir, 'RAPID_Predictions.csv'), index=False)

# نتایج OSNAP
if len(X_test_osnap) > 0:
    osnap_results = pd.DataFrame({
        'year': osnap_merged['year'],
        'date': dates_test_osnap,
        'AMOC_observed': y_test_osnap,
        'AMOC_predicted': y_pred_osnap
    })
    osnap_results.to_csv(os.path.join(output_dir, 'OSNAP_Predictions.csv'), index=False)

# ==============================================
# 14. جمع‌بندی نهایی
# ==============================================

print("\n" + "="*70)
print("📊 خلاصه عملکرد مدل‌ها (رگرسیون ریج)")
print("="*70)

summary = pd.DataFrame({
    'داده': ['Extended (آموزش)', 'RAPID (آزمون)', 'OSNAP (آزمون)'],
    'R²': [r2_train, r2_rapid, r2_osnap if not np.isnan(r2_osnap) else np.nan],
    'RMSE (Sv)': [rmse_train, rmse_rapid, rmse_osnap if not np.isnan(rmse_osnap) else np.nan],
    'MAPE (%)': [mape_train, mape_rapid, mape_osnap if not np.isnan(mape_osnap) else np.nan]
})
print(summary.to_string(index=False))
summary.to_csv(os.path.join(output_dir, 'Model_Summary.csv'), index=False)

print("\n" + "="*70)
print("✅ تحلیل کامل شد!")
print(f"📁 خروجی‌ها در: {output_dir}")
print("\n📄 فایل‌های خروجی:")
print("   • Extended_Predictions.csv - پیش‌بینی روی Extended")
print("   • RAPID_Predictions.csv - پیش‌بینی روی RAPID")
print("   • OSNAP_Predictions.csv - پیش‌بینی روی OSNAP")
print("   • Ridge_Coefficients.csv - ضرایب رگرسیون ریج")
print("   • Model_Summary.csv - خلاصه عملکرد")
print("   • Extended_Prediction.png - نمودار پیش‌بینی Extended")
print("   • RAPID_Prediction.png - نمودار پیش‌بینی RAPID")
print("   • OSNAP_Prediction.png - نمودار پیش‌بینی OSNAP")
print("   • Feature_Importance.png - نمودار اهمیت ویژگی‌ها")
print("="*70)