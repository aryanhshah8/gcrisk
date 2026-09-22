#!/usr/bin/env python3
"""Paper 2 — Items 2 and 3: LET-binned attribution of REID sensitivity.

Produces:

  * ``figures/paper2/let_binned_dose.csv`` — per-bin H and D fractions
    for the baseline shielded flux (16 g/cm² Al, 2011-11-26 transit).
  * ``figures/paper2/let_attribution.csv`` — conditioned sensitivity
    attribution: rows are LHS parameters, columns are LET bins, values
    are Spearman ρ² × H_fraction(bin).
  * ``figures/paper2/let_variance.pdf`` — stacked bar showing where the
    REID sensitivity budget is attributed across LET bins.

The attribution is first-order and framed explicitly as such. See
``gcr/paper2.py:conditioned_let_sensitivity`` for the mathematical
definition and caveats.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
))

from gcr.paper2 import (
    DEFAULT_LET_BINS_KEV_UM,
    _format_bin_labels,
    conditioned_let_sensitivity,
    let_binned_dose_contribution,
    run_ensemble_with_priors,
)

from scripts.paper2._common import (
    FIGURES_DIR,
    baseline_shielded_flux_mean,
    ensure_dirs,
    load_baseline,
)


def run(n_samples: int, seed: int) -> dict:
    ensure_dirs()
    baseline = load_baseline()

    shielded_flux = baseline_shielded_flux_mean(baseline)

    binned = let_binned_dose_contribution(
        shielded_flux, baseline['E_grid'],
        material='tissue',
        bin_edges_keV_um=DEFAULT_LET_BINS_KEV_UM,
    )

    bin_labels = _format_bin_labels(binned.bin_edges_keV_um)
    let_df = pd.DataFrame({
        'LET_bin_keV_um': bin_labels,
        'D_fraction': binned.dose_fraction_by_bin,
        'H_fraction': binned.H_fraction_by_bin,
        'D_rate_Gy_s': binned.dose_by_bin,
        'H_rate_Sv_s': binned.H_by_bin,
    })
    let_path = os.path.join(FIGURES_DIR, 'let_binned_dose.csv')
    let_df.to_csv(let_path, index=False)
    print(f"[paper2.02] Wrote {let_path}")

    print(f"[paper2.02] Running ensemble for attribution  n={n_samples}")
    ens = run_ensemble_with_priors(
        baseline['trajectory_df'],
        shielding_x=baseline['shielding_x_gcm2'],
        material=baseline['material'],
        age=baseline['age'], sex=baseline['sex'],
        prior_overrides=None,
        n_samples=n_samples,
        phi_df=baseline['phi_df'],
        seed=seed,
    )

    attribution = conditioned_let_sensitivity(ens, binned, output='REID')
    attr_path = os.path.join(FIGURES_DIR, 'let_attribution.csv')
    attribution.to_csv(attr_path)
    print(f"[paper2.02] Wrote {attr_path}")

    _plot_let_variance(attribution, binned, out_path=os.path.join(
        FIGURES_DIR, 'let_variance.pdf',
    ))

    return {
        'binned': binned,
        'attribution': attribution,
        'ensemble': ens,
    }


def _plot_let_variance(
    attribution: pd.DataFrame,
    binned,
    out_path: str,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5))

    # Panel 1 — baseline H fraction by LET bin.
    labels = list(attribution.columns)
    H_frac = binned.H_fraction_by_bin
    ax1.bar(labels, H_frac, color='#4a7ab8', edgecolor='black', linewidth=0.6)
    ax1.set_ylabel('Fraction of baseline H (shielded)')
    ax1.set_xlabel('LET bin  [keV/µm]')
    ax1.set_title('Where the baseline mission risk comes from')
    ax1.set_ylim(0, max(H_frac.max() * 1.2, 0.1))

    # Panel 2 — stacked attribution: rows = params, bars = bins.
    # Stack parameters so we can see which parameter contributes most at
    # each LET bin.
    params = list(attribution.index)
    bottom = np.zeros(len(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(params)))
    for p, color in zip(params, colors):
        vals = attribution.loc[p].values.astype(float)
        ax2.bar(labels, vals, bottom=bottom, label=p,
                color=color, edgecolor='black', linewidth=0.4)
        bottom = bottom + vals

    ax2.set_ylabel('Conditioned sensitivity attribution\n(Spearman ρ² × H-fraction)')
    ax2.set_xlabel('LET bin  [keV/µm]')
    ax2.set_title('Attribution of REID sensitivity across LET bins')
    ax2.legend(loc='upper right', fontsize=8, ncol=1)

    fig.suptitle(
        'Paper 2 §3  —  LET-resolved attribution of Mars REID uncertainty',
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[paper2.02] Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n-samples', type=int, default=40)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    run(n_samples=args.n_samples, seed=args.seed)


if __name__ == '__main__':
    main()
