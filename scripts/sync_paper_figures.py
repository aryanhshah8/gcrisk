#!/usr/bin/env python3
"""
scripts/sync_paper_figures.py — Copy generated figures into paper/figures/.

The LaTeX build compiles against paper/figures/, a separate copy from the
repo-root figures/ directory that scripts/plot_*.py and
scripts/compare_iss_solar_cycle.py write to. There is no automatic link
between the two: regenerating a figure in figures/ does NOT update
paper/figures/ until this script (or an equivalent manual copy) is run.
Forgetting this step silently compiles the PDF against stale figures.

Run this after regenerating any figure and before `latexmk`/`pdflatex`.

Usage:
    python scripts/sync_paper_figures.py
"""

import os
import shutil

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
SRC_DIR = os.path.join(REPO_ROOT, 'figures')
DST_DIR = os.path.join(REPO_ROOT, 'paper', 'figures')

# Figures actually referenced by paper/paper.tex (\includegraphics).
PAPER_FIGURES = [
    'shielding_scan.pdf',
    'species_fractions.pdf',
    'let_spectrum.pdf',
    'sep_shielding.pdf',
    'reid_shielding.pdf',
    'iss_solar_cycle.pdf',
    'launch_window_sweep.pdf',
]


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    for name in PAPER_FIGURES:
        src = os.path.join(SRC_DIR, name)
        dst = os.path.join(DST_DIR, name)
        if not os.path.exists(src):
            print(f"  MISSING (not regenerated yet): {name}")
            continue
        shutil.copyfile(src, dst)
        print(f"  Synced {name}")
    print("\nDone. Recompile the paper now (e.g. `latexmk -pdf paper.tex` from paper/).")


if __name__ == '__main__':
    main()
