

import numpy as np
from scipy.stats import pearsonr

# === AUTO DETECT LEVEL COLUMN ===
def detect_level_column(df):
    for c in df.columns:
        if 'level' in c.lower():
            print(f"🎯 Level column detected: {c}")
            return c
    raise ValueError("❌ No Caspian level column found")

# === SAFE IVT ===
def detect_ivt_column(df):
    for c in df.columns:
        if any(k in c.lower() for k in ['net', 'ivt', 'flux']):
            print(f"🌊 IVT column detected: {c}")
            return c
    raise ValueError("❌ No IVT column found")

import re
from pathlib import Path

file_path = Path(r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\comprehensive_final_analysis.py")
text = file_path.read_text(encoding="utf-8")

# =====================================
# 1. اضافه کردن توابع هوشمند
# =====================================
helpers = """

import numpy as np
from scipy.stats import pearsonr

# === AUTO DETECT LEVEL COLUMN ===
def detect_level_column(df):
    for c in df.columns:
        if 'level' in c.lower():
            print(f"🎯 Level column detected: {c}")
            return c
    raise ValueError("❌ No Caspian level column found")

# === SAFE IVT ===
def detect_ivt_column(df):
    for c in df.columns:
        if any(k in c.lower() for k in ['net', 'ivt', 'flux']):
            print(f"🌊 IVT column detected: {c}")
            return c
    raise ValueError("❌ No IVT column found")
"""

text = helpers + "\n" + text

# =====================================
# 2. اصلاح target
# =====================================
text = re.sub(
    r"target\s*=\s*['\"]level_m['\"]",
    "target = detect_level_column(df)",
    text
)

# =====================================
# 3. اضافه کردن تحلیل پیشرفته واقعی
# =====================================
advanced = """

# =====================================
# 🔬 SCIENTIFIC ANALYSIS (ADDED)
# =====================================
def scientific_analysis(df):

    print("\\n🔬 SCIENTIFIC ANALYSIS STARTED")

    level_col = detect_level_column(df)

    # -----------------------------
    # 1. Correlation
    # -----------------------------
    print("\\n📊 Correlation with Level:")
    for col in df.columns:
        if col != level_col and df[col].dtype != 'O':
            try:
                r, p = pearsonr(df[col].dropna(), df[level_col].dropna())
                print(f"{col}: r={r:.3f}, p={p:.3f}")
            except:
                pass

    # -----------------------------
    # 2. Lag Analysis (خیلی مهم)
    # -----------------------------
    print("\\n⏳ Lag Analysis (AMOC → Level)")
    if 'amoc_reconstructed' in df.columns:

        results = []
        for lag in range(0, 15):
            df['lag'] = df['amoc_reconstructed'].shift(lag)
            tmp = df[['lag', level_col]].dropna()

            if len(tmp) > 10:
                r, _ = pearsonr(tmp['lag'], tmp[level_col])
                results.append((lag, r))

        results = sorted(results, key=lambda x: abs(x[1]), reverse=True)

        print("Top Lags:")
        for lag, r in results[:5]:
            print(f"lag {lag}: r={r:.3f}")

    # -----------------------------
    # 3. Simple Trend
    # -----------------------------
    print("\\n📈 Trend Check:")
    if 'year' in df.columns:
        slope = np.polyfit(df['year'], df[level_col], 1)[0]
        print(f"Level trend slope: {slope:.4f} per year")

"""

text += "\n" + advanced

# =====================================
# 4. اضافه کردن call
# =====================================
text = re.sub(
    r"advanced_analysis\(df\)",
    "advanced_analysis(df)
    scientific_analysis(df)\n    scientific_analysis(df)",
    text
)

# =====================================
# 5. ذخیره
# =====================================
new_path = file_path.with_name("FINAL_ANALYSIS_READY.py")
new_path.write_text(text, encoding="utf-8")

print("✅ FINAL FILE CREATED:")
print(new_path)


# =====================================
# 🔬 SCIENTIFIC ANALYSIS (ADDED)
# =====================================
def scientific_analysis(df):

    print("\n🔬 SCIENTIFIC ANALYSIS STARTED")

    level_col = detect_level_column(df)

    # -----------------------------
    # 1. Correlation
    # -----------------------------
    print("\n📊 Correlation with Level:")
    for col in df.columns:
        if col != level_col and df[col].dtype != 'O':
            try:
                r, p = pearsonr(df[col].dropna(), df[level_col].dropna())
                print(f"{col}: r={r:.3f}, p={p:.3f}")
            except:
                pass

    # -----------------------------
    # 2. Lag Analysis (خیلی مهم)
    # -----------------------------
    print("\n⏳ Lag Analysis (AMOC → Level)")
    if 'amoc_reconstructed' in df.columns:

        results = []
        for lag in range(0, 15):
            df['lag'] = df['amoc_reconstructed'].shift(lag)
            tmp = df[['lag', level_col]].dropna()

            if len(tmp) > 10:
                r, _ = pearsonr(tmp['lag'], tmp[level_col])
                results.append((lag, r))

        results = sorted(results, key=lambda x: abs(x[1]), reverse=True)

        print("Top Lags:")
        for lag, r in results[:5]:
            print(f"lag {lag}: r={r:.3f}")

    # -----------------------------
    # 3. Simple Trend
    # -----------------------------
    print("\n📈 Trend Check:")
    if 'year' in df.columns:
        slope = np.polyfit(df['year'], df[level_col], 1)[0]
        print(f"Level trend slope: {slope:.4f} per year")

