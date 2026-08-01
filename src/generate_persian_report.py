#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
تولید گزارش فارسی با اعداد فارسی و اصلاح علامت اعشار
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from PIL import Image
from bidi.algorithm import get_display
from arabic_reshaper import reshape
import warnings
warnings.filterwarnings("ignore")

# ============================================
# توابع تبدیل به فارسی
# ============================================
def persian_text(text):
    if isinstance(text, str):
        return get_display(reshape(text))
    return text

def persian_number(num):
    """تبدیل عدد به فارسی با نقطه اعشار معمولی"""
    if num is None or (isinstance(num, float) and np.isnan(num)):
        return ''
    # برای اعداد صحیح و اعشاری
    if isinstance(num, float):
        # محدود کردن به ۴ رقم اعشار
        s = f"{num:.4f}"
    else:
        s = str(num)
    # جایگزینی ارقام با فارسی
    mapping = {'0': '۰', '1': '۱', '2': '۲', '3': '۳', '4': '۴',
               '5': '۵', '6': '۶', '7': '۷', '8': '۸', '9': '۹'}
    # نقطه را نگه می‌داریم (نه تبدیل به علامت دیگر)
    # منفی را هم به‌صورت '-' معمولی نگه می‌داریم
    result = ''
    for c in s:
        if c in mapping:
            result += mapping[c]
        else:
            result += c  # نقطه و منفی و سایر کاراکترها بدون تغییر
    return result

def persian_number_simple(num):
    """تبدیل عدد صحیح به فارسی (بدون اعشار)"""
    if num is None or (isinstance(num, float) and np.isnan(num)):
        return ''
    s = f"{int(num)}" if isinstance(num, float) else str(num)
    mapping = {'0': '۰', '1': '۱', '2': '۲', '3': '۳', '4': '۴',
               '5': '۵', '6': '۶', '7': '۷', '8': '۸', '9': '۹'}
    result = ''
    for c in s:
        if c in mapping:
            result += mapping[c]
        else:
            result += c
    return result

# تنظیم فونت فارسی
try:
    import matplotlib.font_manager as fm
    fonts = [f.name for f in fm.fontManager.ttflist if 'B Nazanin' in f.name or 'B Mitra' in f.name or 'Tahoma' in f.name]
    if fonts:
        plt.rcParams['font.family'] = fonts[0]
    else:
        plt.rcParams['font.family'] = 'sans-serif'
except:
    plt.rcParams['font.family'] = 'sans-serif'

# ============================================
# تنظیمات مسیرها
# ============================================
BASE_DIR = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503"
OUTPUT_PDF = os.path.join(BASE_DIR, "lake_border", "Caspian_Lake_Analysis_Report_Fa.pdf")
os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)

LAKE_ANALYSIS_DIR = os.path.join(BASE_DIR, "caspian_lake_complete_analysis")
COMPARISON_DIR = os.path.join(BASE_DIR, "basin_lake_comparison")

print("="*70)
print("تولید گزارش فارسی با اعداد صحیح و علامت اعشار درست")
print("="*70)

# ============================================
# ۱. بارگذاری داده‌ها
# ============================================
print("\n📊 بارگذاری داده‌ها...")

def load_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame()

df_mk = load_csv(os.path.join(LAKE_ANALYSIS_DIR, "mk_annual.csv"))
df_bootstrap = load_csv(os.path.join(LAKE_ANALYSIS_DIR, "bootstrap_trend.csv"))
df_pettitt = load_csv(os.path.join(LAKE_ANALYSIS_DIR, "pettitt.csv"))
df_quantile = load_csv(os.path.join(LAKE_ANALYSIS_DIR, "quantile_regression.csv"))
df_enso = load_csv(os.path.join(LAKE_ANALYSIS_DIR, "enso_correlation.csv"))
df_composite = load_csv(os.path.join(LAKE_ANALYSIS_DIR, "composite_results.csv"))
df_rf = load_csv(os.path.join(LAKE_ANALYSIS_DIR, "rf_importance.csv"))
df_comparison = load_csv(os.path.join(COMPARISON_DIR, "comparison_statistics.csv"))

print(f"   ✅ mk_annual: {len(df_mk)} رکورد")
print(f"   ✅ bootstrap: {len(df_bootstrap)} رکورد")
print(f"   ✅ pettitt: {len(df_pettitt)} رکورد")
print(f"   ✅ quantile: {len(df_quantile)} رکورد")
print(f"   ✅ enso: {len(df_enso)} رکورد")
print(f"   ✅ composite: {len(df_composite)} رکورد")
print(f"   ✅ rf: {len(df_rf)} رکورد")
print(f"   ✅ comparison: {len(df_comparison)} رکورد")

# ============================================
# ۲. جمع‌آوری تصاویر
# ============================================
print("\n📸 جمع‌آوری تصاویر...")

all_images = []
patterns = [
    (LAKE_ANALYSIS_DIR, "bootstrap_*.png"),
    (LAKE_ANALYSIS_DIR, "pettitt_*.png"),
    (LAKE_ANALYSIS_DIR, "quantile_*.png"),
    (LAKE_ANALYSIS_DIR, "arima_*.png"),
    (LAKE_ANALYSIS_DIR, "wavelet_*.png"),
    (LAKE_ANALYSIS_DIR, "extreme_*.png"),
    (LAKE_ANALYSIS_DIR, "seasonal_boxplot_*.png"),
    (LAKE_ANALYSIS_DIR, "seasonal_ts_*.png"),
    (LAKE_ANALYSIS_DIR, "decadal_*.png"),
    (LAKE_ANALYSIS_DIR, "stl_*.png"),
    (LAKE_ANALYSIS_DIR, "spectrum_*.png"),
    (COMPARISON_DIR, "timeseries_comparison.png"),
    (COMPARISON_DIR, "scatter_comparison.png"),
    (COMPARISON_DIR, "trend_comparison_bar.png"),
    (COMPARISON_DIR, "correlation_heatmap.png"),
]

for base_dir, pattern in patterns:
    for f in glob.glob(os.path.join(base_dir, pattern)):
        if os.path.exists(f):
            try:
                Image.open(f).verify()
                all_images.append(f)
            except:
                pass

print(f"   ✅ {len(all_images)} تصویر معتبر یافت شد.")

# ============================================
# ۳. تولید PDF
# ============================================
print("\n📄 تولید گزارش PDF...")

def add_table_fa(pdf, df, title, description, columns, col_labels):
    if df.empty:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')
        ax.text(0.05, 0.95, persian_text(title), fontsize=16, fontweight='bold', va='top')
        ax.text(0.05, 0.88, persian_text(description), fontsize=11, va='top', wrap=True)
        ax.text(0.5, 0.5, persian_text('⚠️ داده‌ای برای این جدول وجود ندارد.'), fontsize=14, ha='center', va='center')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        return
    
    available = [c for c in columns if c in df.columns]
    if not available:
        df_display = df.copy()
        cols = list(df.columns)
        labels = col_labels[:len(cols)]
    else:
        df_display = df[available].copy()
        cols = available
        labels = col_labels[:len(available)]
    
    data = df_display.values.tolist()
    for i in range(len(data)):
        for j in range(len(data[i])):
            val = data[i][j]
            if isinstance(val, (int, float)):
                if 'year' in cols[j].lower() or 'change' in cols[j].lower():
                    data[i][j] = persian_number_simple(val)
                else:
                    data[i][j] = persian_number(val)
            else:
                data[i][j] = persian_text(str(val))
    
    n_cols = len(cols)
    
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis('off')
    ax.text(0.05, 0.95, persian_text(title), fontsize=16, fontweight='bold', va='top')
    # توضیحات با خطوط جداگانه
    desc_lines = description.split('\n')
    y_pos = 0.88
    for line in desc_lines:
        ax.text(0.05, y_pos, persian_text(line), fontsize=11, va='top', wrap=True)
        y_pos -= 0.035
    
    col_widths = [0.3] * n_cols
    tbl = ax.table(cellText=data, colLabels=[persian_text(l) for l in labels], loc='center',
                   cellLoc='center', colWidths=col_widths,
                   colColours=['#4472C4']*n_cols)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

with PdfPages(OUTPUT_PDF) as pdf:

    # صفحه عنوان
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis('off')
    ax.text(0.5, 0.8, persian_text('گزارش جامع تحلیل داده‌های دریای خزر'), fontsize=28, ha='center', va='center', fontweight='bold')
    ax.text(0.5, 0.65, persian_text('بر اساس داده‌های ERA5 و ماسک شیپ‌فایل دریاچه'), fontsize=18, ha='center', va='center')
    ax.text(0.5, 0.52, persian_text(f'تاریخ تهیه: {datetime.now().strftime("%Y-%m-%d %H:%M")}'), fontsize=14, ha='center', va='center')
    ax.text(0.5, 0.42, persian_text('دوره: ۱۹۶۵ – ۲۰۲۵ (۶۱ سال)'), fontsize=14, ha='center', va='center')
    ax.text(0.5, 0.32, persian_text('متغیرها: PWAT، بارش، واگرایی IVT، شار خالص مرزی'), fontsize=12, ha='center', va='center')
    ax.text(0.5, 0.22, persian_text('تحلیل‌ها: روند، نقاط شکست، همبستگی، پیش‌بینی، موجک، رویدادهای حدی و ...'), fontsize=12, ha='center', va='center')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # جداول
    add_table_fa(pdf, df_mk,
                 '۱. نتایج آزمون من-کندال (روند سالانه)',
                 'شیب و معنی‌داری روند هر متغیر در دریای خزر.\nمقادیر p < 0.05 نشان‌دهنده روند معنی‌دار است.',
                 ['variable', 'slope', 'p_value', 'trend_sign'],
                 ['متغیر', 'شیب', 'p-value', 'روند'])

    add_table_fa(pdf, df_bootstrap,
                 '۲. روند با بوت‌استرپ (بازه اطمینان ۹۵٪)',
                 'شیب اصلی و بازه اطمینان از ۱۰۰۰ تکرار بوت‌استرپ.\nبازه اطمینان ۹۵٪ نشان‌دهنده دقت تخمین شیب است.',
                 ['variable', 'main_slope', 'ci_lower', 'ci_upper'],
                 ['متغیر', 'شیب اصلی', 'حد پایین', 'حد بالا'])

    add_table_fa(pdf, df_pettitt,
                 '۳. نقاط شکست (آزمون پتیت)',
                 'سال تغییر ناگهانی در سری زمانی هر متغیر.\np-value کوچک (معمولاً < 0.05) نشان‌دهنده تغییر معنی‌دار است.',
                 ['variable', 'change_year', 'p_value'],
                 ['متغیر', 'سال تغییر', 'p-value'])

    add_table_fa(pdf, df_quantile,
                 '۴. رگرسیون چندک (شیب در صدک‌های مختلف)',
                 'شیب روند در سه صدک ۰٫۱ (پایین), ۰٫۵ (میانگین) و ۰٫۹ (بالا).\nتفاوت در شیب‌ها نشان‌دهنده تغییرات در توزیع است.',
                 ['variable', 'quantile', 'slope', 'p_value'],
                 ['متغیر', 'چندک', 'شیب', 'p-value'])

    add_table_fa(pdf, df_enso,
                 '۵. همبستگی با شاخص Nino3.4 (ENSO)',
                 'همبستگی پیرسون هر متغیر با شاخص النینو.\nمقادیر معنی‌دار (p < 0.05) نشان‌دهنده ارتباط با ENSO است.',
                 ['variable', 'correlation', 'p_value'],
                 ['متغیر', 'همبستگی', 'p-value'])

    add_table_fa(pdf, df_composite,
                 '۶. تحلیل مرکب (مقایسه سال‌های کمینه و بیشینه سطح آب)',
                 'مقایسه میانگین متغیرها در دوره‌های کمینه (۱۹۷۶,۷۷,۸۸)\nو بیشینه (۱۹۹۴,۹۵,۹۶) سطح آب دریای خزر.',
                 ['variable', 'low_mean', 'high_mean', 'diff', 'p_value'],
                 ['متغیر', 'میانگین پایین', 'میانگین بالا', 'تفاوت', 'p-value'])

    add_table_fa(pdf, df_rf,
                 '۷. اهمیت ویژگی‌ها (رندوم فارست)',
                 'اهمیت نسبی هر متغیر در پیش‌بینی شار خالص مرزی.\nمجموع اهمیت‌ها برابر ۱ است.',
                 ['feature', 'importance'],
                 ['ویژگی', 'اهمیت'])

    add_table_fa(pdf, df_comparison,
                 '۸. مقایسه حوضه آبریز و خود دریاچه خزر',
                 'مقایسه میانگین، روند و همبستگی بین دو منطقه.\nمقادیر مثبت نشان‌دهنده بیشتر بودن در دریاچه است.',
                 ['Variable', 'Basin_Mean', 'Lake_Mean', 'Basin_Trend', 'Lake_Trend', 'Correlation'],
                 ['متغیر', 'میانگین حوضه', 'میانگین دریاچه', 'روند حوضه', 'روند دریاچه', 'همبستگی'])

    # خلاصه تحلیل‌های تکمیلی
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis('off')
    ax.text(0.05, 0.95, persian_text('۹. خلاصه تحلیل‌های تکمیلی'), fontsize=16, fontweight='bold', va='top')
    ax.text(0.05, 0.85, persian_text('''
    • تحلیل سری‌های زمانی با روش STL: جداسازی روند، الگوی فصلی و باقیمانده‌ها
    • تبدیل موجک (Morlet): شناسایی دوره‌های تناوب ۲ تا ۴ سال و ۸ تا ۱۲ سال
    • پیش‌بینی ARIMA: مدل (1,0,1) برای سال‌های ۲۰۲۶ تا ۲۰۳۰
    • رویدادهای حدی: شناسایی سال‌های با مقادیر فراتر از صدک ۹۰ و پایین‌تر از صدک ۱۰
    • علیت گرنجر: بررسی روابط علی بین متغیرها
    • رندوم فارست: تعیین اهمیت نسبی متغیرها در پیش‌بینی شار خالص
    • تحلیل طیف توان: شناسایی فرکانس‌های غالب با آزمون نویز قرمز
    '''), fontsize=12, va='top', linespacing=1.8)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # تصاویر
    for idx, img_path in enumerate(all_images):
        try:
            fig = plt.figure(figsize=(11.69, 8.27))
            img = Image.open(img_path)
            plt.imshow(img)
            plt.axis('off')
            plt.title(persian_text(os.path.basename(img_path)), fontsize=10, pad=10)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        except:
            continue

print(f"\n✅ گزارش فارسی با موفقیت ایجاد شد.")
print(f"📂 مسیر: {OUTPUT_PDF}")
print(f"📸 تعداد تصاویر: {len(all_images)}")