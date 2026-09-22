#!/usr/bin/env python3
"""Paper 2 — Item 6: LET-overlap comparison figure.

Produces ``figures/paper2/let_overlap.pdf`` with three stacked panels
sharing the LET axis:

  1. **GCR field H-fraction vs LET.** How the mission's dose-equivalent
     budget is distributed across LET, unshielded and behind 16 g/cm² Al.
  2. **Paper 2 biological-attribution sum vs LET bin.** The
     ``Q_factor + DDREF + ERR_scale + EAR_scale`` row-sum of
     ``let_attribution.csv`` — the concentration of REID biological
     uncertainty in LET space.
  3. **Clinical proton-therapy LET coverage.** Points at measured LET
     values from ``data/clinical_rbe/clinical_rbe_normalized.csv`` if
     that file exists; otherwise a shaded 5–25 keV/µm band annotated as
     the expected clinical regime. The section is deliberately
     qualitative (plan §3, third contribution).

This is a **framing observation**, not a quantitative constraint. The
manuscript states explicitly that clinical proton-therapy data is
acute/fractionated while GCR is chronic/mixed-field and does not
propagate numerically into the REID calculation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
))

from gcrisk.paper2 import (
    DEFAULT_LET_BINS_KEV_UM,
    _format_bin_labels,
    let_binned_dose_contribution,
)

from scripts.paper2._common import (
    FIGURES_DIR,
    baseline_shielded_flux_mean,
    ensure_dirs,
    load_baseline,
)


CLINICAL_REGIME_KEV_UM = (5.0, 25.0)
BIOLOGICAL_PARAMS = ('Q_factor', 'DDREF', 'ERR_scale', 'EAR_scale')
FINE_BIN_EDGES = np.array(
    [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
)


def _load_clinical_points() -> pd.DataFrame | None:
    path = Path(FIGURES_DIR).parents[1] / 'data' / 'clinical_rbe' / (
        'clinical_rbe_normalized.csv'
    )
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df


def _load_attribution_table() -> pd.DataFrame:
    path = os.path.join(FIGURES_DIR, 'let_attribution.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected {path} — run scripts/paper2/02_let_attribution.py first"
        )
    return pd.read_csv(path, index_col=0)


def run() -> dict:
    ensure_dirs()
    baseline = load_baseline()

    flux_unshielded = baseline['flux_unshielded_mean']
    flux_shielded = baseline_shielded_flux_mean(baseline)

    fine_un = let_binned_dose_contribution(
        flux_unshielded, baseline['E_grid'], material='tissue',
        bin_edges_keV_um=tuple(FINE_BIN_EDGES),
    )
    fine_sh = let_binned_dose_contribution(
        flux_shielded, baseline['E_grid'], material='tissue',
        bin_edges_keV_um=tuple(FINE_BIN_EDGES),
    )

    attribution = _load_attribution_table()
    bio_sum = attribution.loc[list(BIOLOGICAL_PARAMS)].sum(axis=0)

    clinical_df = _load_clinical_points()

    out_path = os.path.join(FIGURES_DIR, 'let_overlap.pdf')
    _plot(fine_un, fine_sh, bio_sum, clinical_df, out_path)

    return {
        'fine_unshielded': fine_un,
        'fine_shielded': fine_sh,
        'biological_attribution_sum': bio_sum,
        'clinical_points': (
            None if clinical_df is None else len(clinical_df)
        ),
        'overlap_path': out_path,
    }


def _bar_centers(edges: np.ndarray) -> np.ndarray:
    return np.sqrt(edges[:-1] * edges[1:])


def _plot(
    fine_un,
    fine_sh,
    bio_sum: pd.Series,
    clinical_df: pd.DataFrame | None,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.0), sharex=True)
    ax_gcr, ax_bio, ax_clin = axes

    centers = _bar_centers(FINE_BIN_EDGES)
    widths = np.diff(FINE_BIN_EDGES)

    ax_gcr.bar(
        centers, fine_un.H_fraction_by_bin, width=widths * 0.9,
        color='#b8c9de', edgecolor='black', linewidth=0.5,
        align='center', label='Unshielded',
    )
    ax_gcr.bar(
        centers, fine_sh.H_fraction_by_bin, width=widths * 0.55,
        color='#3058a0', edgecolor='black', linewidth=0.5,
        align='center', label='16 g/cm² Al',
    )
    ax_gcr.set_ylabel('Fraction of H by LET bin')
    ax_gcr.set_title('(a) GCR dose-equivalent distribution in LET space')
    ax_gcr.set_xscale('log')
    ax_gcr.legend(loc='upper right', fontsize=9)

    coarse_edges = np.array(DEFAULT_LET_BINS_KEV_UM, dtype=float)
    coarse_edges[-1] = 300.0
    coarse_centers = _bar_centers(coarse_edges)
    coarse_widths = np.diff(coarse_edges)

    ax_bio.bar(
        coarse_centers, bio_sum.values, width=coarse_widths * 0.9,
        color='#c65d5d', edgecolor='black', linewidth=0.5,
        align='center',
    )
    bin_labels = _format_bin_labels(np.array(DEFAULT_LET_BINS_KEV_UM))
    for x, lab, v in zip(coarse_centers, bin_labels, bio_sum.values):
        ax_bio.text(x, v, lab, ha='center', va='bottom', fontsize=8)
    ax_bio.set_ylabel(
        'Biological-parameter\nattribution sum  [Σ ρ² × H-frac]'
    )
    ax_bio.set_title(
        '(b) Where REID biological uncertainty concentrates '
        '(Q + DDREF + ERR + EAR)'
    )
    ax_bio.set_ylim(0, max(bio_sum.values.max() * 1.25, 1e-3))

    lo, hi = CLINICAL_REGIME_KEV_UM
    for ax in (ax_gcr, ax_bio, ax_clin):
        ax.axvspan(lo, hi, color='#f1c96a', alpha=0.25, zorder=0)

    if clinical_df is None:
        ax_clin.text(
            np.sqrt(lo * hi), 0.5,
            'Clinical proton-therapy LET regime\n'
            '(5–25 keV/µm; data curation pending —\n'
            'see data/clinical_rbe/README.md)',
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='#fff4d0', edgecolor='#b08a2a'),
        )
        ax_clin.set_yticks([])
    else:
        source_colors = {
            'pide': '#2d6cdf',
            'paganetti2014': '#d95f02',
            'wedenberg2013': '#1b9e77',
        }
        for src, sub in clinical_df.groupby('source'):
            ax_clin.scatter(
                sub['LET_keV_um'], sub['RBE'],
                color=source_colors.get(src, '#555555'),
                edgecolor='black', linewidth=0.3,
                alpha=0.8, label=f'{src}  (n={len(sub)})',
            )
        ax_clin.set_ylabel('Measured RBE')
        ax_clin.legend(loc='upper left', fontsize=9)

    ax_clin.set_xlabel('LET  [keV/µm]')
    ax_clin.set_title(
        '(c) Clinical proton-therapy LET coverage '
        '(contextual anchor, not a quantitative constraint)'
    )

    ax_clin.set_xlim(FINE_BIN_EDGES.min(), FINE_BIN_EDGES.max())

    fig.suptitle(
        'Paper 2 §3  —  LET overlap between Mars-REID biological '
        'uncertainty and clinical proton-therapy evidence',
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[paper2.04] Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.parse_args()
    run()


if __name__ == '__main__':
    main()
