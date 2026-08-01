#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================================================
Collect Final Files – Caspian Sea Analysis Project
================================================================================
This script copies all essential files (PDFs, LaTeX sources, images, 
analysis codes, and result CSVs) to a single output folder for easy 
archiving or transfer.

Usage: python collect_final_files.py
================================================================================
"""

import os
import shutil
import datetime
from pathlib import Path

# ============================================================
# 1. Configuration – Source and Destination
# ============================================================
SOURCE_BASE = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\predict_caspian"
SOURCE_FINAL_ANALYSIS = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\final_analysis"

# Destination folder (on Desktop or current directory)
DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")
DEST_BASE = os.path.join(DESKTOP, "Caspian_Paper_Final")
# Alternatively, you can set a custom path:
# DEST_BASE = r"C:\Caspian_Paper_Final"

# ============================================================
# 2. Files to Copy (from predict_caspian folder)
# ============================================================
FILES_TO_COPY = [
    # PDFs
    "caspian_analysis_fa.pdf",
    "caspian_analysis_en.pdf",
    # LaTeX sources
    "caspian_analysis_fa.tex",
    "caspian_analysis_en.tex",
    # Images
    "final_analysis_plots.png",
    "spectral_coherence.png",
    "wavelet_coherence.png",
    "causal_impact.png",
    "counterfactual_scenarios.png",
    # Essential Python scripts
    "analysis_final_code.py",
    "causality.py",
    "Compiler.py",
    # Optional but useful
    "Change Point.py",
    "ultimate_test.py",
    "spectral.py",
    "Forward Selection.py",
    "wavelet.py",
    "Cross-Correlation.py",
]

# ============================================================
# 3. Helper Functions
# ============================================================
def create_directory(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    print(f"📁 Created: {path}")

def copy_file(src, dst):
    """Copy a single file, creating parent directories if needed."""
    if not os.path.exists(src):
        print(f"⚠️  Source file not found: {src}")
        return False
    try:
        shutil.copy2(src, dst)
        print(f"   ✅ Copied: {os.path.basename(src)}")
        return True
    except Exception as e:
        print(f"   ❌ Error copying {src}: {e}")
        return False

def copy_directory(src, dst):
    """Copy an entire directory recursively."""
    if not os.path.exists(src):
        print(f"⚠️  Source directory not found: {src}")
        return False
    try:
        if os.path.exists(dst):
            shutil.rmtree(dst)  # Remove existing to avoid conflicts
        shutil.copytree(src, dst)
        print(f"   ✅ Copied directory: {os.path.basename(src)}")
        return True
    except Exception as e:
        print(f"   ❌ Error copying directory {src}: {e}")
        return False

def create_readme(dest_path):
    """Create a README file with project summary."""
    readme_content = f"""
================================================================================
Caspian Sea Level Analysis – Final Project Files
================================================================================
Date collected: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This folder contains all final deliverables for the Caspian Sea level 
decline analysis project.

📁 Folder Structure:
├── caspian_analysis_fa.pdf       – Full paper (Persian)
├── caspian_analysis_en.pdf       – Full paper (English)
├── caspian_analysis_fa.tex       – LaTeX source (Persian)
├── caspian_analysis_en.tex       – LaTeX source (English)
├── *.png                          – All figures used in the paper
├── *.py                           – All analysis Python scripts
└── final_analysis/                – All numerical results (CSV, reports)

📊 Key Results Summary:
- Change Point: November 2022
- Best Model: TVP (Rolling Window) with R² = 0.9755
- Granger Causality: NAO, SOI, ONI (all p < 0.05)
- Causal Impact of Nov 2022 event: -1.33 m
- Forecast 2030 (Pessimistic): -4.76 m
- Forecast 2030 (Probable): -3.29 m
- Forecast 2030 (Optimistic): -3.08 m

🔧 How to Recompile the Paper:
   Run: python Compiler.py

🔬 How to Re-run Analysis:
   Run: python analysis_final_code.py  (for comprehensive analysis)
   Run: python causality.py             (for causality tests)

📖 For more details, see the PDF papers or the reports in final_analysis/.

================================================================================
"""
    readme_path = os.path.join(dest_path, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"   ✅ Created: README.txt")

# ============================================================
# 4. Main Function
# ============================================================
def main():
    print("=" * 80)
    print("📦 Collecting Final Files – Caspian Sea Project")
    print("=" * 80)
    
    # Create destination folder
    create_directory(DEST_BASE)
    
    # Copy individual files
    print("\n📄 Copying files from predict_caspian...")
    for filename in FILES_TO_COPY:
        src = os.path.join(SOURCE_BASE, filename)
        dst = os.path.join(DEST_BASE, filename)
        copy_file(src, dst)
    
    # Copy entire final_analysis folder
    print("\n📂 Copying final_analysis folder (all numerical results)...")
    src_analysis = SOURCE_FINAL_ANALYSIS
    dst_analysis = os.path.join(DEST_BASE, "final_analysis")
    copy_directory(src_analysis, dst_analysis)
    
    # Create README
    print("\n📝 Creating README file...")
    create_readme(DEST_BASE)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ ALL FILES COLLECTED SUCCESSFULLY!")
    print("=" * 80)
    print(f"📂 Destination folder: {DEST_BASE}")
    print("\nYou can now archive or transfer this folder.")
    print("=" * 80)

# ============================================================
# 5. Entry Point
# ============================================================
if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...")