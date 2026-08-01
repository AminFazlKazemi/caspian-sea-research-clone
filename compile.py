#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
LaTeX Compiler – Caspian Sea Paper (Persian & English)
================================================================================
This script compiles both Persian and English versions of the paper.
It automatically finds required figures and generates placeholders if missing.
Usage: python compile.py
================================================================================
"""
import os
import sys
import shutil
import subprocess
from PIL import Image, ImageDraw

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEX_FILES = {
    'persian': os.path.join(BASE_DIR, 'caspian_analysis_fa.tex'),
    'english': os.path.join(BASE_DIR, 'caspian_analysis_en.tex')
}
FIGURE_NAMES = [
    'final_analysis_plots.png',
    'spectral_coherence.png',
    'wavelet_coherence.png',
    'causal_impact.png',
    'counterfactual_scenarios.png'
]

def check_command(cmd):
    return shutil.which(cmd) is not None

def create_placeholder(path, width=800, height=600):
    try:
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((width//2 - 100, height//2), "Figure placeholder", fill='black')
        img.save(path)
        return True
    except:
        return False

def ensure_figures():
    fig_dir = os.path.join(BASE_DIR, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    for fname in FIGURE_NAMES:
        dest = os.path.join(fig_dir, fname)
        if not os.path.exists(dest):
            create_placeholder(dest)
            print(f" Created placeholder: {fname}")

def run_latex(tex_file, compiler='xelatex'):
    tex_dir = os.path.dirname(tex_file)
    basename = os.path.splitext(os.path.basename(tex_file))[0]
    os.chdir(tex_dir)
    cmd = [compiler, '-interaction=nonstopmode', basename + '.tex']
    subprocess.run(cmd, check=False)
    if os.path.exists(basename + '.aux') and check_command('bibtex'):
        subprocess.run(['bibtex', basename + '.aux'], check=False)
    subprocess.run(cmd, check=False)
    subprocess.run(cmd, check=False)
    pdf_path = os.path.join(tex_dir, basename + '.pdf')
    if os.path.exists(pdf_path):
        size = os.path.getsize(pdf_path) / 1024
        print(f" ✅ PDF generated: {pdf_path} ({size:.1f} KB)")
        return True
    else:
        print(f" ❌ PDF not found for {basename}")
        return False

def main():
    print("="*80)
    print("📚 Compiling Caspian Sea Research Papers")
    print("="*80)
    ensure_figures()
    has_xelatex = check_command('xelatex')
    has_pdflatex = check_command('pdflatex')
    print(f"\n✅ XeLaTeX: {'✅' if has_xelatex else '❌'}")
    print(f"✅ pdfLaTeX: {'✅' if has_pdflatex else '❌'}")

    print("\n📖 Compiling Persian version...")
    if has_xelatex:
        run_latex(TEX_FILES['persian'], 'xelatex')
    else:
        print(" ❌ XeLaTeX not found – Persian version skipped.")

    print("\n📖 Compiling English version...")
    if has_pdflatex:
        run_latex(TEX_FILES['english'], 'pdflatex')
    elif has_xelatex:
        run_latex(TEX_FILES['english'], 'xelatex')
    else:
        print(" ❌ No LaTeX compiler found.")
    print("\n🏁 Done.")

if __name__ == "__main__":
    main()
