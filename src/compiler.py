#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================================================
LaTeX Compiler – Caspian Sea Analysis Papers (Improved)
================================================================================
"""

import os
import sys
import subprocess
import shutil

BASE_DIR = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\predict_caspian"

FILES = {
    'persian': {
        'tex': os.path.join(BASE_DIR, 'caspian_analysis_fa.tex'),
        'compiler': 'xelatex',
        'output': 'caspian_analysis_fa.pdf'
    },
    'english': {
        'tex': os.path.join(BASE_DIR, 'caspian_analysis_en.tex'),
        'compiler': 'pdflatex',
        'output': 'caspian_analysis_en.pdf'
    }
}

def check_command(cmd):
    return shutil.which(cmd) is not None

def run_latex(command, cwd, timeout=180):
    """Run LaTeX command and return (success, output)."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        # Check for real errors (not warnings)
        stderr_lower = result.stderr.lower()
        if "! fatal" in stderr_lower or "emergency stop" in stderr_lower:
            return False, result.stderr
        # If PDF exists, consider it success
        return True, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)

def compile_tex(tex_path, compiler):
    tex_dir = os.path.dirname(tex_path)
    basename = os.path.splitext(os.path.basename(tex_path))[0]

    if not os.path.exists(tex_path):
        print(f"❌ File not found: {tex_path}")
        return False

    print(f"\n📄 Compiling: {basename}.tex with {compiler}")

    # First pass
    print("   [1/3] First pass...")
    ok, _ = run_latex([compiler, '-interaction=nonstopmode', basename + '.tex'], tex_dir)
    if not ok:
        print("   ❌ Fatal error detected.")
        return False

    # BibTeX
    if check_command('bibtex') and os.path.exists(os.path.join(tex_dir, basename + '.aux')):
        print("   [2/3] BibTeX...")
        run_latex(['bibtex', basename + '.aux'], tex_dir)

    # Second pass
    print("   [3/3] Final pass...")
    ok, _ = run_latex([compiler, '-interaction=nonstopmode', basename + '.tex'], tex_dir)
    if not ok:
        return False

    pdf_path = os.path.join(tex_dir, basename + '.pdf')
    if os.path.exists(pdf_path):
        size = os.path.getsize(pdf_path) / 1024
        print(f"   ✅ PDF generated: {pdf_path} ({size:.1f} KB)")
        return True
    else:
        print("   ❌ PDF not found.")
        return False

def main():
    print("="*80)
    print("📚 LaTeX Compiler – Caspian Sea Analysis Papers (Improved)")
    print("="*80)

    # Check compilers
    has_xelatex = check_command('xelatex')
    has_pdflatex = check_command('pdflatex')
    print(f"\n✅ XeLaTeX: {'✅' if has_xelatex else '❌'}")
    print(f"✅ pdfLaTeX: {'✅' if has_pdflatex else '❌'}")

    # Compile Persian
    print("\n" + "="*80)
    print("📖 Compiling Persian version...")
    success_fa = compile_tex(FILES['persian']['tex'], FILES['persian']['compiler'])

    # Compile English
    print("\n" + "="*80)
    print("📖 Compiling English version...")
    success_en = compile_tex(FILES['english']['tex'], FILES['english']['compiler'])

    # Summary
    print("\n" + "="*80)
    print("🏁 SUMMARY")
    print("="*80)
    if os.path.exists(os.path.join(BASE_DIR, 'caspian_analysis_fa.pdf')):
        print("✅ Persian PDF: caspian_analysis_fa.pdf")
    if os.path.exists(os.path.join(BASE_DIR, 'caspian_analysis_en.pdf')):
        print("✅ English PDF: caspian_analysis_en.pdf")
    print("="*80)

if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...")