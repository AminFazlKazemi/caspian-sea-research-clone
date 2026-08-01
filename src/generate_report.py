# generate_report_final.py
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# ============================================
# Paths
# ============================================
BASE_DIR = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503"
OUTPUT_PDF = os.path.join(BASE_DIR, "Caspian_Report_Full.pdf")
ANALYSIS_DIR = os.path.join(BASE_DIR, "additional_analyses")
MOISTURE_DIR = os.path.join(BASE_DIR, "all_analysis_on_boder")
EVAP_DIR = os.path.join(BASE_DIR, "evaporation_analysis")

print("="*70)
print("Generating Comprehensive Caspian Report with Descriptions")
print("="*70)

# ============================================
# Helper: load data flexibly
# ============================================
def load_flexible_csv(path, required_columns=None, default_data=None):
    if not os.path.exists(path):
        return pd.DataFrame(default_data) if default_data is not None else pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if required_columns:
            for col in required_columns:
                if col not in df.columns:
                    for c in df.columns:
                        if c.lower() in [col.lower(), col.lower().replace('_', ''), col.lower().replace('-', '_')]:
                            df = df.rename(columns={c: col})
                            break
            missing = [c for c in required_columns if c not in df.columns]
            if missing:
                if default_data is not None:
                    return pd.DataFrame(default_data)
        return df
    except:
        return pd.DataFrame(default_data) if default_data is not None else pd.DataFrame()

# ============================================
# 1. Load all numerical results
# ============================================
print("\n📊 Loading numerical results...")

# 1-1. Mann-Kendall
mk_default = [
    ('moisture t2m', 0.0241, 0.0000, 'Increasing'),
    ('moisture tp', 0.0000, 0.0039, 'Increasing'),
    ('moisture vimd', -0.0010, 0.1291, 'Decreasing'),
    ('evap e', -0.0000, 0.0000, 'Decreasing')
]
mk_files = glob.glob(os.path.join(ANALYSIS_DIR, "mk_*.csv"))
if mk_files:
    df_mk_list = []
    for f in mk_files:
        df = load_flexible_csv(f, ['variable', 'slope', 'p_value', 'trend_sign'])
        if not df.empty:
            rename = {}
            for c in df.columns:
                if 'variable' in c.lower(): rename[c] = 'Variable'
                elif 'slope' in c.lower(): rename[c] = 'Slope'
                elif 'p_value' in c.lower() or 'pvalue' in c.lower(): rename[c] = 'p-value'
                elif 'trend_sign' in c.lower() or 'trend' in c.lower(): rename[c] = 'Trend'
            df = df.rename(columns=rename)
            if all(c in df.columns for c in ['Variable', 'Slope', 'p-value', 'Trend']):
                df_mk_list.append(df[['Variable', 'Slope', 'p-value', 'Trend']])
    df_mk = pd.concat(df_mk_list, ignore_index=True) if df_mk_list else pd.DataFrame(mk_default, columns=['Variable', 'Slope', 'p-value', 'Trend'])
else:
    df_mk = pd.DataFrame(mk_default, columns=['Variable', 'Slope', 'p-value', 'Trend'])
df_mk = df_mk.sort_values('Variable')

# 1-2. Bootstrap
bootstrap_file = os.path.join(MOISTURE_DIR, "bootstrap_trend.csv")
if os.path.exists(bootstrap_file):
    df_bootstrap = load_flexible_csv(bootstrap_file, ['variable', 'main_slope', 'ci_lower', 'ci_upper'])
    if not df_bootstrap.empty:
        rename = {}
        for c in df_bootstrap.columns:
            if 'variable' in c.lower(): rename[c] = 'Variable'
            elif 'main_slope' in c.lower() or 'slope' in c.lower(): rename[c] = 'Main Slope'
            elif 'ci_lower' in c.lower() or 'lower' in c.lower(): rename[c] = '95% Lower'
            elif 'ci_upper' in c.lower() or 'upper' in c.lower(): rename[c] = '95% Upper'
        df_bootstrap = df_bootstrap.rename(columns=rename)
        if all(c in df_bootstrap.columns for c in ['Variable', 'Main Slope', '95% Lower', '95% Upper']):
            df_bootstrap = df_bootstrap[['Variable', 'Main Slope', '95% Lower', '95% Upper']]
        else:
            df_bootstrap = pd.DataFrame()
    else:
        df_bootstrap = pd.DataFrame()
else:
    df_bootstrap = pd.DataFrame()

# 1-3. Composite
composite_file = os.path.join(MOISTURE_DIR, "composite_results.csv")
if os.path.exists(composite_file):
    df_composite = load_flexible_csv(composite_file, ['variable', 'diff', 'p_value'])
    if not df_composite.empty:
        rename = {}
        for c in df_composite.columns:
            if 'variable' in c.lower(): rename[c] = 'Variable'
            elif 'diff' in c.lower(): rename[c] = 'Difference'
            elif 'p_value' in c.lower() or 'pvalue' in c.lower(): rename[c] = 'p-value'
        df_composite = df_composite.rename(columns=rename)
        if all(c in df_composite.columns for c in ['Variable', 'Difference', 'p-value']):
            df_composite = df_composite[['Variable', 'Difference', 'p-value']]
        else:
            df_composite = pd.DataFrame()
    else:
        df_composite = pd.DataFrame()
else:
    df_composite = pd.DataFrame()

# 1-4. Extreme Events
extreme_files = glob.glob(os.path.join(ANALYSIS_DIR, "extreme_*.xlsx"))
extreme_list = []
for f in extreme_files:
    try:
        df = pd.read_excel(f, sheet_name='Summary')
        if not df.empty:
            rename = {}
            for c in df.columns:
                if 'variable' in c.lower(): rename[c] = 'Variable'
                elif 'high_count' in c.lower(): rename[c] = 'High Count'
                elif 'low_count' in c.lower(): rename[c] = 'Low Count'
            df = df.rename(columns=rename)
            if all(c in df.columns for c in ['Variable', 'High Count', 'Low Count']):
                extreme_list.append(df[['Variable', 'High Count', 'Low Count']])
    except:
        pass
df_extreme = pd.concat(extreme_list, ignore_index=True) if extreme_list else pd.DataFrame()

# 1-5. Climate Indices
climate_file = os.path.join(ANALYSIS_DIR, "climate_indices_correlation_extra.csv")
if os.path.exists(climate_file):
    df_climate = load_flexible_csv(climate_file, ['dataset', 'variable', 'index', 'correlation', 'p_value', 'significant'])
    if not df_climate.empty:
        # Keep only significant
        sig_col = [c for c in df_climate.columns if 'significant' in c.lower()]
        if sig_col:
            df_climate = df_climate[df_climate[sig_col[0]] == True]
        rename = {}
        for c in df_climate.columns:
            if 'dataset' in c.lower(): rename[c] = 'Dataset'
            elif 'variable' in c.lower(): rename[c] = 'Variable'
            elif 'index' in c.lower(): rename[c] = 'Index'
            elif 'correlation' in c.lower(): rename[c] = 'Correlation'
            elif 'p_value' in c.lower() or 'pvalue' in c.lower(): rename[c] = 'p-value'
        df_climate = df_climate.rename(columns=rename)
        if all(c in df_climate.columns for c in ['Dataset', 'Variable', 'Index', 'Correlation', 'p-value']):
            df_climate = df_climate[['Dataset', 'Variable', 'Index', 'Correlation', 'p-value']]
        else:
            df_climate = pd.DataFrame()
    else:
        df_climate = pd.DataFrame()
else:
    df_climate = pd.DataFrame()

# 1-6. Seasonal Trends
seasonal_files = glob.glob(os.path.join(ANALYSIS_DIR, "seasonal_trend_*.csv"))
seasonal_list = []
for f in seasonal_files:
    df = load_flexible_csv(f, ['variable', 'season', 'slope', 'p_value'])
    if not df.empty:
        rename = {}
        for c in df.columns:
            if 'variable' in c.lower(): rename[c] = 'Variable'
            elif 'season' in c.lower(): rename[c] = 'Season'
            elif 'slope' in c.lower(): rename[c] = 'Slope'
            elif 'p_value' in c.lower() or 'pvalue' in c.lower(): rename[c] = 'p-value'
        df = df.rename(columns=rename)
        if all(c in df.columns for c in ['Variable', 'Season', 'Slope', 'p-value']):
            seasonal_list.append(df[['Variable', 'Season', 'Slope', 'p-value']])
df_seasonal = pd.concat(seasonal_list, ignore_index=True) if seasonal_list else pd.DataFrame()

# 1-7. Pettitt Change Points
pettitt_file = os.path.join(MOISTURE_DIR, "change_points.csv")
if os.path.exists(pettitt_file):
    df_pettitt = load_flexible_csv(pettitt_file, ['variable', 'change_year', 'p_value'])
    if not df_pettitt.empty:
        rename = {}
        for c in df_pettitt.columns:
            if 'variable' in c.lower(): rename[c] = 'Variable'
            elif 'change_year' in c.lower() or 'year' in c.lower(): rename[c] = 'Change Year'
            elif 'p_value' in c.lower() or 'pvalue' in c.lower(): rename[c] = 'p-value'
        df_pettitt = df_pettitt.rename(columns=rename)
        if all(c in df_pettitt.columns for c in ['Variable', 'Change Year', 'p-value']):
            df_pettitt = df_pettitt[['Variable', 'Change Year', 'p-value']]
        else:
            df_pettitt = pd.DataFrame()
    else:
        df_pettitt = pd.DataFrame()
else:
    df_pettitt = pd.DataFrame()

# 1-8. Quantile Regression (if available)
qr_file = os.path.join(MOISTURE_DIR, "quantile_regression_results.csv")
if os.path.exists(qr_file):
    df_qr = load_flexible_csv(qr_file, ['variable', 'quantile', 'slope', 'p_value'])
    if not df_qr.empty:
        rename = {}
        for c in df_qr.columns:
            if 'variable' in c.lower(): rename[c] = 'Variable'
            elif 'quantile' in c.lower(): rename[c] = 'Quantile'
            elif 'slope' in c.lower(): rename[c] = 'Slope'
            elif 'p_value' in c.lower() or 'pvalue' in c.lower(): rename[c] = 'p-value'
        df_qr = df_qr.rename(columns=rename)
        if all(c in df_qr.columns for c in ['Variable', 'Quantile', 'Slope', 'p-value']):
            df_qr = df_qr[['Variable', 'Quantile', 'Slope', 'p-value']]
        else:
            df_qr = pd.DataFrame()
    else:
        df_qr = pd.DataFrame()
else:
    df_qr = pd.DataFrame()

print("   ✅ Data loading complete.")

# ============================================
# 2. Collect valid images
# ============================================
print("\n📸 Collecting valid images...")
image_categories = {
    'Correlation Matrix': os.path.join(ANALYSIS_DIR, "correlation_heatmap_*.png"),
    'Trend Analysis': os.path.join(ANALYSIS_DIR, "trend_*.png"),
    'ACF/PACF': os.path.join(ANALYSIS_DIR, "acf_*.png"),
    'Extreme Events': os.path.join(ANALYSIS_DIR, "extreme_*.png"),
    'Extreme Time Series': os.path.join(ANALYSIS_DIR, "extreme_timeseries_*.png"),
    'Seasonal Boxplot': os.path.join(ANALYSIS_DIR, "seasonal_boxplot_*.png"),
    'Seasonal Time Series': os.path.join(ANALYSIS_DIR, "seasonal_timeseries_*.png"),
    'Composite Analysis': os.path.join(MOISTURE_DIR, "composite", "*.png"),
    'PCA Clustering': os.path.join(MOISTURE_DIR, "pca_*.png"),
    'ARIMA Forecast': os.path.join(MOISTURE_DIR, "arima_*.png"),
    'Wavelet Power': os.path.join(MOISTURE_DIR, "wavelet_*.png"),
    'Drought Index': os.path.join(MOISTURE_DIR, "border_drought_*.png"),
    'Decadal Heatmap': os.path.join(MOISTURE_DIR, "decadal_heatmap_*.png"),
    'Evaporation Trend': os.path.join(EVAP_DIR, "*.png"),
}
all_images = []
for cat, pattern in image_categories.items():
    for f in glob.glob(pattern):
        if not os.path.exists(f):
            continue
        try:
            Image.open(f).verify()
            all_images.append((cat, f))
        except:
            print(f"   ⚠️ Skipped invalid: {f}")
print(f"   ✅ {len(all_images)} valid images found.")

# ============================================
# 3. Generate Comprehensive PDF
# ============================================
print("\n📄 Generating comprehensive PDF report...")
with PdfPages(OUTPUT_PDF) as pdf:

    # ---------- Title Page ----------
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis('off')
    ax.text(0.5, 0.75, 'Caspian Sea', fontsize=36, ha='center', va='center', fontweight='bold')
    ax.text(0.5, 0.65, 'Comprehensive Data Analysis Report', fontsize=24, ha='center', va='center')
    ax.text(0.5, 0.55, f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}', fontsize=16, ha='center', va='center')
    ax.text(0.5, 0.45, 'Period: 1940 – 2026 (87 years, monthly data)', fontsize=14, ha='center', va='center')
    ax.text(0.5, 0.35, 'Variables: t2m (temperature), tp (precipitation), vimd (moisture divergence), e (evaporation)', fontsize=12, ha='center', va='center')
    ax.text(0.5, 0.25, 'Region: Caspian Sea Basin (36.5°–47°N, 46°–54°E)', fontsize=12, ha='center', va='center')
    ax.text(0.5, 0.15, 'Analyses: Trends, Correlations, Extreme Events, Seasonal Patterns, Spectral Analysis, Forecasting, etc.', fontsize=11, ha='center', va='center')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ---------- Helper: add table with description ----------
    def add_section(title, description, df, col_labels=None, col_widths=None, color_col='p-value'):
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')
        # Description
        ax.text(0.05, 0.95, title, fontsize=18, fontweight='bold', va='top')
        ax.text(0.05, 0.88, description, fontsize=12, va='top', wrap=True)
        if df.empty:
            ax.text(0.5, 0.5, 'No data available for this section.', fontsize=14, ha='center', va='center')
        else:
            if col_labels is None:
                col_labels = list(df.columns)
            table_data = df.values
            colors = None
            if color_col in df.columns:
                pvals = df[color_col].astype(float)
                colors = [['#D9E2F3']*len(col_labels) if p < 0.05 else ['#FDE9D9']*len(col_labels) for p in pvals]
            if col_widths is None:
                col_widths = [0.3] * len(col_labels)
            # adjust col_widths if needed
            tbl = ax.table(cellText=table_data, colLabels=col_labels, loc='center',
                           cellLoc='center', colWidths=col_widths,
                           cellColours=colors, colColours=['#4472C4']*len(col_labels))
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(11)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    # ---------- 1. Trend Analysis ----------
    add_section(
        "1. Trend Analysis",
        "Mann-Kendall test is a non-parametric test for monotonic trends. "
        "Theil-Sen slope estimator provides a robust measure of trend magnitude. "
        "Bootstrap resampling (1000 iterations) was used to estimate 95% confidence intervals. "
        "Pettitt test detects a single change point in the time series.",
        df_mk,
        col_labels=['Variable', 'Slope', 'p-value', 'Trend'],
        col_widths=[0.3, 0.2, 0.2, 0.3]
    )
    if not df_bootstrap.empty:
        add_section(
            "Bootstrap Trend (95% Confidence Intervals)",
            "Bootstrap resampling with Theil-Sen slope estimator (1000 iterations) "
            "provides robust trend estimates and confidence intervals.",
            df_bootstrap,
            col_labels=['Variable', 'Main Slope', '95% Lower', '95% Upper'],
            col_widths=[0.3, 0.2, 0.2, 0.3],
            color_col=None
        )
    if not df_pettitt.empty:
        add_section(
            "Change Point Detection (Pettitt Test)",
            "The Pettitt test identifies the most significant change point in the time series. "
            "A p-value < 0.05 indicates a statistically significant change.",
            df_pettitt,
            col_labels=['Variable', 'Change Year', 'p-value'],
            col_widths=[0.3, 0.3, 0.3]
        )
    if not df_qr.empty:
        add_section(
            "Quantile Regression (Slope at Different Quantiles)",
            "Quantile regression estimates the trend at different quantiles (0.1, 0.5, 0.9). "
            "This reveals whether trends differ across the distribution (e.g., extremes vs. median).",
            df_qr,
            col_labels=['Variable', 'Quantile', 'Slope', 'p-value'],
            col_widths=[0.25, 0.15, 0.2, 0.2]
        )

    # ---------- 2. Correlation & Dependency ----------
    add_section(
        "2. Correlation & Dependency",
        "Pearson and Spearman correlation matrices between all variables (t2m, tp, vimd, e). "
        "Spearman is robust to non-linearity and outliers. "
        "Cross-correlation with time lags identifies lagged relationships between precipitation and moisture flux.",
        pd.DataFrame(),  # no table, just images
        col_labels=None
    )
    # we will add images separately

    # ---------- 3. Extreme Events ----------
    if not df_extreme.empty:
        add_section(
            "3. Extreme Events",
            "Extreme events are defined as values above the 90th percentile (High) "
            "and below the 10th percentile (Low). The table shows the number of extreme years "
            "for each variable. These events are critical for understanding drought and flood risks.",
            df_extreme,
            col_labels=['Variable', 'High Count', 'Low Count'],
            col_widths=[0.3, 0.2, 0.2],
            color_col=None
        )

    # ---------- 4. Seasonal Analysis ----------
    if not df_seasonal.empty:
        add_section(
            "4. Seasonal Analysis",
            "Seasonal decomposition reveals the average annual cycle and its long-term changes. "
            "The table shows trends for each season (Mann-Kendall slope) and their significance.",
            df_seasonal,
            col_labels=['Variable', 'Season', 'Slope', 'p-value'],
            col_widths=[0.25, 0.15, 0.2, 0.2]
        )

    # ---------- 5. Composite Analysis ----------
    if not df_composite.empty:
        add_section(
            "5. Composite Analysis (High vs Low Caspian Water Level)",
            "Composite analysis compares the average of variables during high water level years "
            "(1994, 1995, 1996) versus low water level years (1976, 1977, 1988). "
            "The 'Difference' column shows (High - Low) values. A significant p-value indicates "
            "that the two groups are statistically different.",
            df_composite,
            col_labels=['Variable', 'Difference', 'p-value'],
            col_widths=[0.3, 0.3, 0.3]
        )

    # ---------- 6. Climate Indices ----------
    if not df_climate.empty:
        add_section(
            "6. Climate Indices (ENSO, NAO, AMO, PDO)",
            "Correlation between Caspian Sea variables and major climate indices: "
            "Nino3.4 (ENSO), NAO (North Atlantic Oscillation), AMO (Atlantic Multidecadal Oscillation), "
            "and PDO (Pacific Decadal Oscillation). Only significant correlations (p<0.05) are shown.",
            df_climate,
            col_labels=['Dataset', 'Variable', 'Index', 'Correlation', 'p-value'],
            col_widths=[0.15, 0.15, 0.15, 0.15, 0.15]
        )

    # ---------- 7. Images (all categories) ----------
    for category, image_path in all_images:
        try:
            fig = plt.figure(figsize=(11.69, 8.27))
            img = Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'{category}: {os.path.basename(image_path)}', fontsize=10, pad=10)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"   ⚠️ Error adding {image_path}: {e}")
            continue

print(f"\n✅ Comprehensive report generated successfully.")
print(f"📂 Path: {OUTPUT_PDF}")