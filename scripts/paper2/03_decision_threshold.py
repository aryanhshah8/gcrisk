#!/usr/bin/env python3
"""Paper 2 — Item 4: decision-threshold inversion.

Takes the Paper 1 baseline total dose equivalent H_total_Sv (from
``integrate_mission_dose``) and sweeps a single biological precision
scale factor across the four biological priors (Q_factor, DDREF,
ERR_scale, EAR_scale). Reports REID percentiles and the fraction of MC
samples exceeding the pre-2023 NASA 3% REID threshold at each precision
level.

Output:
  * ``figures/paper2/decision_threshold.csv``
  * ``figures/paper2/decision_threshold.pdf``
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
))

from gcr.paper2 import (
    HISTORICAL_NASA_REID_THRESHOLD,
    decision_threshold_sweep,
)

from scripts.paper2._common import (
    FIGURES_DIR,
    baseline_total_dose,
    ensure_dirs,
    load_baseline,
)


def run(n_samples: int) -> dict:
    ensure_dirs()
    baseline = load_baseline()

    print("[paper2.03] Running Paper 1 baseline dose integration")
    dose = baseline_total_dose(baseline)
    H_total_Sv = float(dose['H_total_mSv']) / 1000.0
    print(f"[paper2.03] H_total_mSv = {dose['H_total_mSv']:.2f}   "
          f"(H_total_Sv = {H_total_Sv:.4f})")

    precision_scales = np.array(
        [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0], dtype=float,
    )

    print(f"[paper2.03] Sweeping precision scales {list(precision_scales)}"
          f" with n_samples={n_samples}")
    sweep = decision_threshold_sweep(
        H_total_Sv=H_total_Sv,
        age_at_exposure=baseline['age'], sex=baseline['sex'],
        precision_scales=precision_scales,
        n_samples=n_samples,
    )

    out_csv = os.path.join(FIGURES_DIR, 'decision_threshold.csv')
    sweep.to_csv(out_csv, index=False)
    print(f"[paper2.03] Wrote {out_csv}")

    _plot_threshold(sweep, out_path=os.path.join(
        FIGURES_DIR, 'decision_threshold.pdf',
    ))

    return {'sweep': sweep, 'H_total_Sv': H_total_Sv}


def _plot_threshold(sweep, out_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    thr = HISTORICAL_NASA_REID_THRESHOLD

    # Panel 1: REID median with p5/p95 band vs precision.
    ax1.fill_between(
        sweep['precision_scale'], sweep['REID_p5'] * 100.0,
        sweep['REID_p95'] * 100.0,
        color='#4a7ab8', alpha=0.25, label='p5–p95 band',
    )
    ax1.plot(
        sweep['precision_scale'], sweep['REID_median'] * 100.0,
        color='#1e3a6b', marker='o', label='REID median',
    )
    ax1.axhline(thr * 100.0, color='red', linestyle='--',
                label=f'pre-2023 NASA 3% REID threshold')
    ax1.set_xlabel('Biological precision scale  (1.0 = Paper 1 priors)')
    ax1.set_ylabel('REID  [%]')
    ax1.set_title('Mars-transit REID vs. biological prior width')
    ax1.legend(fontsize=8)

    # Panel 2: fraction of samples exceeding the 3% threshold.
    ax2.plot(
        sweep['precision_scale'],
        sweep['exceeds_limit_fraction'],
        color='#9e2a2b', marker='s',
    )
    ax2.axhline(0.05, color='gray', linestyle=':', label='5% exceedance')
    ax2.set_xlabel('Biological precision scale')
    ax2.set_ylabel('Fraction of MC samples above 3% REID')
    ax2.set_title('Decision-threshold inversion')
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, max(sweep['exceeds_limit_fraction'].max() * 1.1, 0.05))

    fig.suptitle(
        'Paper 2 §3  —  How much biological precision is needed to keep '
        'REID below the historical 3% benchmark?',
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[paper2.03] Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n-samples', type=int, default=4000,
                   help='MC samples per sweep point.')
    args = p.parse_args()
    run(n_samples=args.n_samples)


if __name__ == '__main__':
    main()
