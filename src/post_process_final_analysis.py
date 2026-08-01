#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
تولید گزارش کامل LaTeX با فرمول‌ها، روش‌شناسی و تمام جداول آماری
- خواندن فایل statistical_summary.csv برای آمار روند
- خواندن فایل‌های ماهانه (monthly_*.csv) برای محاسبه میانگین دوره‌ها
- تولید یک فایل مستقل LaTeX با تمام جزئیات
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================
# ۱. تنظیمات مسیرها
# ============================================================
BASE_DIR = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\final_analysis\caspian_flux_output"
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
CSV_SUMMARY = os.path.join(REPORTS_DIR, "statistical_summary.csv")
OUTPUT_TEX = os.path.join(REPORTS_DIR, "detailed_methods_report.tex")

SPECIAL_PERIODS = {
    'Low_Water': [1976, 1977, 1978],
    'High_Water': [1994, 1995, 1996]
}
PERIOD_30_1 = list(range(1965, 1995))
PERIOD_30_2 = list(range(1995, 2025))

# ============================================================
# ۲. توابع کمکی
# ============================================================
def load_summary():
    return pd.read_csv(CSV_SUMMARY)

def load_monthly_data(boundary, side):
    """بارگذاری فایل ماهانه برای یک مرز و ضلع مشخص"""
    fname = os.path.join(BASE_DIR, f"{boundary}_{side}_monthly.csv")
    if os.path.exists(fname):
        return pd.read_csv(fname)
    return None

def compute_composite_means(df_monthly):
    """محاسبه میانگین دوره‌های خاص و ۳۰ ساله"""
    comp = {}
    for pname, years in SPECIAL_PERIODS.items():
        mask = df_monthly['year'].isin(years)
        if mask.sum() > 0:
            comp[pname] = df_monthly[mask].mean(numeric_only=True).to_dict()
    mask1 = df_monthly['year'].isin(PERIOD_30_1)
    mask2 = df_monthly['year'].isin(PERIOD_30_2)
    if mask1.sum() > 0:
        comp['1965-1994'] = df_monthly[mask1].mean(numeric_only=True).to_dict()
    if mask2.sum() > 0:
        comp['1995-2024'] = df_monthly[mask2].mean(numeric_only=True).to_dict()
    return comp

def format_number(x):
    if pd.isna(x):
        return '--'
    if abs(x) > 1e6:
        return f"{x:.2e}"
    return f"{x:.4f}"

def format_float(x, decimals=4):
    if pd.isna(x):
        return '--'
    return f"{x:.{decimals}f}"

# ============================================================
# ۳. تولید محتوای LaTeX
# ============================================================
def generate_latex():
    df = load_summary()
    today = datetime.now().strftime("%B %d, %Y")

    # جمع‌آوری مرزها و اضلاع
    boundaries = df['boundary'].unique()
    sides_dict = {}
    for b in boundaries:
        sides_dict[b] = df[df['boundary'] == b]['side'].unique().tolist()

    lines = []
    lines.append("\\documentclass[12pt,a4paper]{article}")
    lines.append("\\usepackage[utf8]{inputenc}")
    lines.append("\\usepackage{amsmath,amssymb,amsfonts}")
    lines.append("\\usepackage{geometry}")
    lines.append("\\geometry{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}")
    lines.append("\\usepackage{booktabs}")
    lines.append("\\usepackage{graphicx}")
    lines.append("\\usepackage{hyperref}")
    lines.append("\\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue}")
    lines.append("\\usepackage{longtable}")
    lines.append("\\usepackage{array}")
    lines.append("\\begin{document}")
    lines.append("\\title{Comprehensive Methodology and Statistical Results for Moisture Flux Analysis over the Caspian Sea}")
    lines.append("\\author{Generated from Final Analysis Results}")
    lines.append(f"\\date{{{today}}}")
    lines.append("\\maketitle")

    # ============================================================
    # مقدمه
    # ============================================================
    lines.append("\\section{Introduction}")
    lines.append("This report presents a detailed methodological framework and comprehensive statistical results "
                 "for the analysis of moisture flux over the Caspian Sea. The dataset is derived from ERA5 reanalysis "
                 "(1965–2025) and covers multiple boundary configurations: the Iranian border (four sectors: North, South, East, West), "
                 "the Caspian Sea lake boundary, and three sectors (North, Center, South) based on a 55-point discretization. "
                 "The variables analyzed are inflow and outflow fluxes (kg/s). "
                 "The following sections describe the mathematical formulations of each statistical method and provide "
                 "complete tables of results for all boundaries and sides.")

    # ============================================================
    # روش‌شناسی
    # ============================================================
    lines.append("\\section{Methodology}")
    lines.append("A suite of advanced statistical and time-series techniques was applied to the monthly flux time series. "
                 "Each method is described below with its mathematical formulation and key references.")

    # ۱. Mann-Kendall
    lines.append("\\subsection{Mann-Kendall Trend Test}")
    lines.append("The Mann-Kendall test \\citep{mann1945,kendall1975} is a non-parametric test for monotonic trends. "
                 "The test statistic $S$ is computed as:")
    lines.append("\\begin{equation}")
    lines.append("S = \\sum_{i=1}^{n-1} \\sum_{j=i+1}^{n} \\text{sgn}(x_j - x_i)")
    lines.append("\\end{equation}")
    lines.append("where $\\text{sgn}(u) = 1$ if $u>0$, $0$ if $u=0$, and $-1$ if $u<0$. "
                 "The variance of $S$ is given by:")
    lines.append("\\begin{equation}")
    lines.append("\\text{Var}(S) = \\frac{n(n-1)(2n+5) - \\sum_{k} t_k(t_k-1)(2t_k+5)}{18}")
    lines.append("\\end{equation}")
    lines.append("where $t_k$ are the ties in the data. The standardized test statistic $Z$ is computed and the $p$-value "
                 "is obtained from the standard normal distribution. Kendall's $\\tau$ is also reported.")

    # ۲. Theil-Sen
    lines.append("\\subsection{Theil-Sen Slope Estimator}")
    lines.append("The Theil-Sen estimator \\citep{theil1950,sen1968} provides a robust estimate of the slope $\\beta$ "
                 "by taking the median of all pairwise slopes:")
    lines.append("\\begin{equation}")
    lines.append("\\beta = \\text{median}\\left(\\frac{x_j - x_i}{j - i}\\right), \\quad \\forall i < j")
    lines.append("\\end{equation}")

    # ۳. Pettitt
    lines.append("\\subsection{Pettitt Change Point Test}")
    lines.append("The Pettitt test \\citep{pettitt1979} detects a single change point in the time series. "
                 "The test statistic $U_{t,n}$ is defined as:")
    lines.append("\\begin{equation}")
    lines.append("U_{t,n} = \\sum_{i=1}^{t} \\sum_{j=t+1}^{n} \\text{sgn}(x_i - x_j)")
    lines.append("\\end{equation}")
    lines.append("The most likely change point is at $t$ where $|U_{t,n}|$ is maximized. "
                 "The significance is assessed via an asymptotic $p$-value: $p \\approx 2 \\exp(-6K^2/(n^3+n^2))$, "
                 "where $K = \\max_t |U_{t,n}|$.")

    # ۴. Quantile Regression
    lines.append("\\subsection{Quantile Regression}")
    lines.append("Quantile regression \\citep{koenker2001} estimates the conditional quantile function. "
                 "For a given quantile $\\tau \\in (0,1)$, the model is:")
    lines.append("\\begin{equation}")
    lines.append("Q_{y}(\\tau | x) = \\alpha(\\tau) + \\beta(\\tau) x")
    lines.append("\\end{equation}")
    lines.append("The coefficients are estimated by minimizing the weighted absolute residuals:")
    lines.append("\\begin{equation}")
    lines.append("\\min_{\\alpha,\\beta} \\sum_{i=1}^{n} \\rho_\\tau (y_i - \\alpha - \\beta x_i), \\quad "
                 "\\rho_\\tau(u) = u(\\tau - \\mathbf{1}_{u<0})")
    lines.append("\\end{equation}")
    lines.append("We applied quantile regression for $\\tau = 0.1, 0.5, 0.9$ to examine trends across the distribution.")

    # ۵. Wavelet
    lines.append("\\subsection{Wavelet Analysis (Morlet)}")
    lines.append("The continuous wavelet transform (CWT) \\citep{torrence1998} using the Morlet wavelet is applied to identify dominant periodicities. "
                 "The wavelet transform $W(\\tau, s)$ is defined as:")
    lines.append("\\begin{equation}")
    lines.append("W(\\tau, s) = \\int_{-\\infty}^{\\infty} x(t) \\, \\psi^*\\left(\\frac{t-\\tau}{s}\\right) dt")
    lines.append("\\end{equation}")
    lines.append("where $\\psi$ is the Morlet mother wavelet, $\\tau$ is the translation parameter, and $s$ is the scale. "
                 "The wavelet power spectrum is $|W(\\tau, s)|^2$. Scales from 1 to 20 years were used.")

    # ۶. ARIMA
    lines.append("\\subsection{ARIMA Forecasting}")
    lines.append("The ARIMA model \\citep{box1976} is defined as $ARIMA(p,d,q)$, where $p$ is the autoregressive order, "
                 "$d$ is the differencing order, and $q$ is the moving average order. The model is expressed as:")
    lines.append("\\begin{equation}")
    lines.append("\\phi(B)(1-B)^d y_t = \\theta(B) \\varepsilon_t")
    lines.append("\\end{equation}")
    lines.append("where $\\phi(B)$ and $\\theta(B)$ are polynomials in the backshift operator $B$, and $\\varepsilon_t$ "
                 "is white noise. We used $p=1, d=0, q=1$ and forecasted 5 steps ahead.")

    # ۷. STL
    lines.append("\\subsection{STL Decomposition}")
    lines.append("Seasonal-Trend decomposition using LOESS (STL) \\citep{cleveland1990} decomposes the time series into "
                 "trend, seasonal, and residual components:")
    lines.append("\\begin{equation}")
    lines.append("y_t = T_t + S_t + R_t")
    lines.append("\\end{equation}")
    lines.append("where $T_t$ is the trend, $S_t$ is the seasonal component, and $R_t$ is the remainder. "
                 "A seasonal period of 12 months was used.")

    # ۸. Bootstrap
    lines.append("\\subsection{Bootstrap Resampling for Trend Uncertainty}")
    lines.append("Bootstrap resampling \\citep{politis1994} is used to estimate the uncertainty of the Theil-Sen slope. "
                 "For $B=1000$ iterations, we resample the data with replacement and compute the slope $\\beta^*_b$. "
                 "The 95\\% confidence interval is given by the 2.5th and 97.5th percentiles of the bootstrap distribution.")

    # ۹. Benjamini-Hochberg
    lines.append("\\subsection{Benjamini-Hochberg Correction}")
    lines.append("To control the False Discovery Rate (FDR) in multiple testing, the Benjamini-Hochberg procedure \\citep{benjamini1995} is applied. "
                 "Given $m$ p-values $p_{(1)} \\leq \\cdots \\leq p_{(m)}$, the adjusted p-values are:")
    lines.append("\\begin{equation}")
    lines.append("p_{(i)}^{\\text{adj}} = \\min\\left(1, \\min_{j \\geq i} \\left\\{ \\frac{m}{j} p_{(j)} \\right\\}\\right)")
    lines.append("\\end{equation}")

    # ۱۰. Composite Analysis
    lines.append("\\subsection{Composite Analysis}")
    lines.append("Composite means are computed for two special periods: Low Water (1976–1978) and High Water (1994–1996), "
                 "as well as for two 30-year periods (1965–1994 and 1995–2024). "
                 "A two-sample Welch's $t$-test is used to compare the means of the two 30-year periods.")

    # ============================================================
    # نتایج: جداول روند
    # ============================================================
    lines.append("\\section{Statistical Results}")
    lines.append("The following tables present the trend statistics and composite period means for each boundary and side. "
                 "For variables with suffix \\texttt{_stl}, \\texttt{_wavelet}, \\texttt{_arima}, the values are not applicable (NA) "
                 "as they represent intermediate outputs of the corresponding methods.")

    # جدول روند (فقط ردیف‌های اصلی inflow/outflow را بگیریم)
    lines.append("\\subsection{Trend Analysis}")
    lines.append("\\begin{longtable}{l l l c c c c c c}")
    lines.append("\\caption{Trend statistics (Mann-Kendall, Theil-Sen, Pettitt, Quantile regression) for all boundaries and sides.}")
    lines.append("\\label{tab:trend} \\\\")
    lines.append("\\toprule")
    lines.append("Boundary & Side & Variable & MK-$\\tau$ & MK $p$-value & Theil-Sen slope & Pettitt $p$-value & Mean & Std \\\\")
    lines.append("\\midrule")
    lines.append("\\endfirsthead")
    lines.append("\\multicolumn{8}{c}{{Continued from previous page}} \\\\")
    lines.append("\\toprule")
    lines.append("Boundary & Side & Variable & MK-$\\tau$ & MK $p$-value & Theil-Sen slope & Pettitt $p$-value & Mean & Std \\\\")
    lines.append("\\midrule")
    lines.append("\\endhead")
    lines.append("\\bottomrule")
    lines.append("\\endfoot")

    # فیلتر کردن ردیف‌های اصلی (نام متغیر بدون پسوند)
    main_vars = ['inflow', 'outflow']
    df_main = df[df['variable'].isin(main_vars)]
    for _, row in df_main.iterrows():
        b = row['boundary']
        s = row['side']
        v = row['variable']
        mk_tau = format_float(row['mk_tau'], 3)
        mk_p = format_float(row['mk_p'], 4)
        slope = format_float(row['theil_slope'], 2)
        pettitt_p = format_float(row['pettitt_p'], 4)
        mean = format_float(row['mean'], 2)
        std = format_float(row['std'], 2)
        line = f"{b} & {s} & {v} & {mk_tau} & {mk_p} & {slope} & {pettitt_p} & {mean} & {std} \\\\"
        lines.append(line)
    lines.append("\\end{longtable}")

    # ============================================================
    # جداول Composite
    # ============================================================
    lines.append("\\subsection{Composite Period Means}")
    lines.append("\\begin{longtable}{l l l c c}")
    lines.append("\\caption{Mean inflow and outflow for special periods and 30-year periods.}")
    lines.append("\\label{tab:composite} \\\\")
    lines.append("\\toprule")
    lines.append("Boundary & Side & Period & Inflow Mean & Outflow Mean \\\\")
    lines.append("\\midrule")
    lines.append("\\endfirsthead")
    lines.append("\\multicolumn{5}{c}{{Continued from previous page}} \\\\")
    lines.append("\\toprule")
    lines.append("Boundary & Side & Period & Inflow Mean & Outflow Mean \\\\")
    lines.append("\\midrule")
    lines.append("\\endhead")
    lines.append("\\bottomrule")
    lines.append("\\endfoot")

    # برای هر مرز و ضلع، داده ماهانه را بخوان و composite را محاسبه کن
    for b in boundaries:
        for side in sides_dict[b]:
            df_monthly = load_monthly_data(b, side)
            if df_monthly is None:
                continue
            comp = compute_composite_means(df_monthly)
            for pname, comp_dict in comp.items():
                if pname == 'ttest_30yr':
                    continue
                inflow = comp_dict.get('inflow', np.nan)
                outflow = comp_dict.get('outflow', np.nan)
                if not np.isnan(inflow) and not np.isnan(outflow):
                    line = f"{b} & {side} & {pname} & {inflow:.2f} & {outflow:.2f} \\\\"
                    lines.append(line)
    lines.append("\\end{longtable}")

    # ============================================================
    # شکل‌ها
    # ============================================================
    lines.append("\\section{Figures}")
    lines.append("Figures \\ref{fig:annual} and \\ref{fig:heatmap} show examples of the annual time series and decadal heatmaps. "
                 "All figures are available in the \\texttt{reports} directory.")
    lines.append("\\begin{figure}[h]")
    lines.append("\\centering")
    lines.append("\\includegraphics[width=0.8\\textwidth]{Iran_North_annual.png}")
    lines.append("\\caption{Annual time series for Iran-North boundary.}")
    lines.append("\\label{fig:annual}")
    lines.append("\\end{figure}")
    lines.append("\\begin{figure}[h]")
    lines.append("\\centering")
    lines.append("\\includegraphics[width=0.8\\textwidth]{heatmap_Iran_North_inflow.png}")
    lines.append("\\caption{Decadal monthly heatmap for Iran-North inflow.}")
    lines.append("\\label{fig:heatmap}")
    lines.append("\\end{figure}")

    # ============================================================
    # مراجع
    # ============================================================
    lines.append("\\begin{thebibliography}{99}")
    refs = [
        ("mann1945", "Mann, H. B. (1945). Nonparametric tests against trend. Econometrica, 13(3), 245-259."),
        ("kendall1975", "Kendall, M. G. (1975). Rank Correlation Methods. Griffin."),
        ("theil1950", "Theil, H. (1950). A rank-invariant method of linear and polynomial regression analysis. Proceedings of the Koninklijke Nederlandse Akademie van Wetenschappen, 53, 386-392."),
        ("sen1968", "Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau. Journal of the American Statistical Association, 63(324), 1379-1389."),
        ("pettitt1979", "Pettitt, A. N. (1979). A non-parametric approach to the change-point problem. Applied Statistics, 28(2), 126-135."),
        ("koenker2001", "Koenker, R. (2001). Quantile Regression. Cambridge University Press."),
        ("torrence1998", "Torrence, C., and Compo, G. P. (1998). A practical guide to wavelet analysis. Bulletin of the American Meteorological Society, 79(1), 61-78."),
        ("box1976", "Box, G. E. P., and Jenkins, G. M. (1976). Time Series Analysis: Forecasting and Control. Holden-Day."),
        ("cleveland1990", "Cleveland, R. B., Cleveland, W. S., McRae, J. E., and Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on loess. Journal of Official Statistics, 6(1), 3-73."),
        ("politis1994", "Politis, D. N., and Romano, J. P. (1994). The stationary bootstrap. Journal of the American Statistical Association, 89(428), 1303-1313."),
        ("benjamini1995", "Benjamini, Y., and Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal Statistical Society: Series B, 57(1), 289-300.")
    ]
    for key, citation in refs:
        lines.append(f"\\bibitem{{{key}}} {citation}")
    lines.append("\\end{thebibliography}")
    lines.append("\\end{document}")

    return "\n".join(lines)

# ============================================================
# ۴. اجرا
# ============================================================
def main():
    print("🚀 Generating detailed LaTeX report...")
    tex_content = generate_latex()
    with open(OUTPUT_TEX, 'w', encoding='utf-8') as f:
        f.write(tex_content)
    print(f"✅ Detailed LaTeX report saved to: {OUTPUT_TEX}")
    print("📄 Compile with: pdflatex detailed_methods_report.tex")

if __name__ == "__main__":
    main()