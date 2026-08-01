# -*- coding: utf-8 -*-
"""
انتخاب ویژگی با SHAP و اعتبارسنجی روی OSNAP
نسخه نهایی - برای مقاله
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# تنظیمات
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

# ایجاد پوشه خروجی
output_dir = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\Final_Results_SHAP_OSNAP"
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("📊 انتخاب ویژگی با SHAP و اعتبارسنجی روی OSNAP")
print("="*70)

# ==============================================
# 1. تعریف مسیرها
# ==============================================

paths = {
    'rapid': r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\amoc_rapid_RAPID (12 month mean).csv",
    'osnap': r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\amoc_osnap_OSNAP (12 month mean).csv",
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
    return df[['date', 'AMOC']]

def load_amoc_osnap(path):
    df = pd.read_csv(path, comment='#')
    df = df.dropna(subset=['OSNAP (12 month mean) (Sv)'])
    df['date'] = pd.to_datetime(df['Year'].astype(int).astype(str) + '-' + 
                                df['Month'].astype(int).astype(str) + '-01')
    df = df.rename(columns={'OSNAP (12 month mean) (Sv)': 'AMOC'})
    return df[['date', 'AMOC']]

# ==============================================
# 3. بارگذاری داده‌ها
# ==============================================

print("\n🔄 بارگذاری داده‌ها...")
amoc_rapid = load_amoc_rapid(paths['rapid'])
amoc_osnap = load_amoc_osnap(paths['osnap'])
print(f"✅ RAPID: {len(amoc_rapid)} ماه")
print(f"✅ OSNAP: {len(amoc_osnap)} ماه")

# بارگذاری شاخص‌ها
enso_df = pd.read_excel(paths['enso'], sheet_name='Sheet1')
enso_df['date'] = pd.to_datetime(enso_df['date'])
print(f"✅ شاخص‌ها: {len(enso_df)} ماه")

# شاخص‌های انتخابی (همه شاخص‌های موجود به جز تاریخ)
all_features = [col for col in enso_df.columns if col not in ['date']]
print(f"تعداد کل شاخص‌ها: {len(all_features)}")

# ==============================================
# 4. توابع مهندسی ویژگی (تاخیر و میانگین متحرک)
# ==============================================

def create_features(df, feature_cols, target_col='AMOC', max_lag=24, apply_smoothing=True):
    """
    ایجاد ویژگی‌های با تاخیر و میانگین متحرک
    اگر apply_smoothing=True، روی AMOC میانگین متحرک ۱۲ ماهه اعمال می‌شود.
    """
    df_feat = df.copy()
    
    # اعمال فیلتر پایین‌گذر روی AMOC (میانگین متحرک ۱۲ ماهه)
    if apply_smoothing:
        df_feat['AMOC_smoothed'] = df_feat['AMOC'].rolling(12).mean()
        target = 'AMOC_smoothed'
        df_feat = df_feat.dropna(subset=[target])
    else:
        target = target_col
    
    # تاخیرها
    lags = [1, 3, 6, 9, 12, 18, 24]
    for lag in lags:
        for col in feature_cols:
            df_feat[f'{col}_lag{lag}'] = df_feat[col].shift(lag)
    
    # میانگین متحرک
    for col in feature_cols:
        df_feat[f'{col}_MA3'] = df_feat[col].rolling(3).mean()
        df_feat[f'{col}_MA6'] = df_feat[col].rolling(6).mean()
        df_feat[f'{col}_MA12'] = df_feat[col].rolling(12).mean()
    
    df_feat = df_feat.dropna()
    X_cols = [col for col in df_feat.columns if col not in ['date', 'AMOC', 'AMOC_smoothed']]
    X = df_feat[X_cols]
    y = df_feat[target]
    dates = df_feat['date']
    return X, y, dates, X_cols

# ==============================================
# 5. ایجاد ویژگی‌ها برای RAPID
# ==============================================

print("\n🔄 ایجاد ویژگی‌ها برای RAPID (با فیلتر پایین‌گذر)...")

# ادغام RAPID با شاخص‌ها
merged_rapid = pd.merge(amoc_rapid, enso_df, on='date', how='inner')
merged_rapid = merged_rapid.sort_values('date').reset_index(drop=True)

X_rapid, y_rapid, dates_rapid, feature_names = create_features(
    merged_rapid, all_features, target_col='AMOC', max_lag=24, apply_smoothing=True
)

print(f"تعداد نمونه‌های RAPID: {len(X_rapid)}")
print(f"تعداد ویژگی‌ها: {len(feature_names)}")

# ==============================================
# 6. آموزش XGBoost و محاسبه SHAP
# ==============================================

print("\n🧠 آموزش XGBoost و محاسبه SHAP...")

# تقسیم داده‌های RAPID به آموزش/آزمون (سری زمانی)
split = int(0.8 * len(X_rapid))
X_train, X_test = X_rapid[:split], X_rapid[split:]
y_train, y_test = y_rapid[:split], y_rapid[split:]
dates_train, dates_test = dates_rapid[:split], dates_rapid[split:]

# استانداردسازی
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

# مدل XGBoost
model_xgb = xgb.XGBRegressor(
    n_estimators=200, max_depth=8, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
)
model_xgb.fit(X_train_scaled, y_train)

# ارزیابی اولیه
y_pred_train = model_xgb.predict(X_train_scaled)
y_pred_test = model_xgb.predict(X_test_scaled)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"\n📊 عملکرد XGBoost روی RAPID (با فیلتر پایین‌گذر):")
print(f"   آموزش: RMSE={rmse_train:.3f}, R²={r2_train:.3f}")
print(f"   آزمون: RMSE={rmse_test:.3f}, R²={r2_test:.3f}")

# محاسبه SHAP
explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X_test_scaled)
shap_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

print("\n🔟 ویژگی برتر (SHAP):")
print(shap_importance.head(10).to_string(index=False))

# ==============================================
# 7. انتخاب ۲۰ ویژگی برتر
# ==============================================

top_n = 20
selected_features = shap_importance.head(top_n)['feature'].tolist()
print(f"\n✅ {top_n} ویژگی برتر انتخاب شدند.")

# ذخیره لیست ویژگی‌های منتخب
pd.DataFrame({'selected_features': selected_features}).to_csv(
    os.path.join(output_dir, 'selected_features.csv'), index=False
)

# ==============================================
# 8. آموزش مدل نهایی (Random Forest) روی ویژگی‌های منتخب
# ==============================================

print("\n🧠 آموزش مدل نهایی (Random Forest) روی ویژگی‌های منتخب...")

X_train_sel = X_train_scaled[selected_features]
X_test_sel = X_test_scaled[selected_features]

model_rf = RandomForestRegressor(n_estimators=200, random_state=42)
model_rf.fit(X_train_sel, y_train)

y_pred_train_rf = model_rf.predict(X_train_sel)
y_pred_test_rf = model_rf.predict(X_test_sel)
rmse_train_rf = np.sqrt(mean_squared_error(y_train, y_pred_train_rf))
rmse_test_rf = np.sqrt(mean_squared_error(y_test, y_pred_test_rf))
r2_train_rf = r2_score(y_train, y_pred_train_rf)
r2_test_rf = r2_score(y_test, y_pred_test_rf)

print(f"\n📊 عملکرد Random Forest روی RAPID (ویژگی‌های منتخب):")
print(f"   آموزش: RMSE={rmse_train_rf:.3f}, R²={r2_train_rf:.3f}")
print(f"   آزمون: RMSE={rmse_test_rf:.3f}, R²={r2_test_rf:.3f}")

# ==============================================
# 9. اعتبارسنجی روی OSNAP
# ==============================================

print("\n🔄 اعتبارسنجی روی OSNAP...")

# ادغام OSNAP با شاخص‌ها
merged_osnap = pd.merge(amoc_osnap, enso_df, on='date', how='inner')
merged_osnap = merged_osnap.sort_values('date').reset_index(drop=True)

# ایجاد ویژگی‌ها برای OSNAP (با همان تنظیمات)
X_osnap, y_osnap, dates_osnap, _ = create_features(
    merged_osnap, all_features, target_col='AMOC', max_lag=24, apply_smoothing=True
)

print(f"تعداد نمونه‌های OSNAP: {len(X_osnap)}")
if len(X_osnap) == 0:
    print("⚠️ داده‌های OSNAP پس از اعمال فیلتر خالی هستند. بررسی کنید.")
else:
    # استانداردسازی با استفاده از scaler آموزش‌دیده روی RAPID
    X_osnap_scaled = scaler.transform(X_osnap)
    X_osnap_scaled = pd.DataFrame(X_osnap_scaled, columns=feature_names)
    X_osnap_sel = X_osnap_scaled[selected_features]
    
    # پیش‌بینی
    y_pred_osnap = model_rf.predict(X_osnap_sel)
    
    # ارزیابی
    rmse_osnap = np.sqrt(mean_squared_error(y_osnap, y_pred_osnap))
    r2_osnap = r2_score(y_osnap, y_pred_osnap)
    mape_osnap = np.mean(np.abs((y_osnap - y_pred_osnap) / y_osnap)) * 100
    
    print(f"\n📊 عملکرد مدل روی OSNAP:")
    print(f"   RMSE: {rmse_osnap:.3f} Sv")
    print(f"   R²: {r2_osnap:.3f}")
    print(f"   MAPE: {mape_osnap:.2f}%")
    
    # ذخیره نتایج OSNAP
    osnap_results = pd.DataFrame({
        'date': dates_osnap,
        'AMOC_observed': y_osnap,
        'AMOC_predicted': y_pred_osnap
    })
    osnap_results.to_csv(os.path.join(output_dir, 'OSNAP_validation_results.csv'), index=False)

# ==============================================
# 10. نمودارها
# ==============================================

print("\n📈 رسم نمودارها...")

# نمودار مقایسه پیش‌بینی روی OSNAP
if len(X_osnap) > 0:
    plt.figure(figsize=(14, 6))
    plt.plot(dates_osnap, y_osnap, 'o-', label='مشاهده شده (OSNAP)', linewidth=2)
    plt.plot(dates_osnap, y_pred_osnap, 's--', label='پیش‌بینی مدل (آموزش‌دیده روی RAPID)', alpha=0.7)
    plt.xlabel('تاریخ')
    plt.ylabel('AMOC (Sv)')
    plt.title(f'اعتبارسنجی روی OSNAP\nRMSE={rmse_osnap:.3f} Sv, R²={r2_osnap:.3f}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'OSNAP_validation.png'), dpi=300)
    plt.show()

# نمودار اهمیت SHAP
plt.figure(figsize=(12, 8))
top_15 = shap_importance.head(15)
plt.barh(top_15['feature'], top_15['importance'], color='steelblue')
plt.xlabel('اهمیت SHAP')
plt.ylabel('ویژگی')
plt.title('۱۵ ویژگی برتر (SHAP)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'SHAP_importance_top15.png'), dpi=300)
plt.show()

# ==============================================
# 11. جمع‌بندی نهایی
# ==============================================

print("\n" + "="*70)
print("✅ تحلیل کامل شد!")
print(f"📁 خروجی‌ها در: {output_dir}")
print("\n📄 فایل‌های خروجی:")
print("   • selected_features.csv - لیست ویژگی‌های منتخب")
print("   • OSNAP_validation_results.csv - نتایج اعتبارسنجی روی OSNAP")
print("   • OSNAP_validation.png - نمودار اعتبارسنجی")
print("   • SHAP_importance_top15.png - نمودار اهمیت SHAP")
print("="*70)

# جمع‌بندی عملکرد
print("\n📊 خلاصه عملکرد مدل‌ها:")
summary = pd.DataFrame({
    'مدل': ['XGBoost (همه ویژگی‌ها) - RAPID', 
            'Random Forest (ویژگی‌های منتخب) - RAPID',
            'Random Forest (ویژگی‌های منتخب) - OSNAP'],
    'R²': [r2_test, r2_test_rf, r2_osnap if len(X_osnap)>0 else np.nan],
    'RMSE (Sv)': [rmse_test, rmse_test_rf, rmse_osnap if len(X_osnap)>0 else np.nan]
})
print(summary.to_string(index=False))
summary.to_csv(os.path.join(output_dir, 'Model_Summary.csv'), index=False)