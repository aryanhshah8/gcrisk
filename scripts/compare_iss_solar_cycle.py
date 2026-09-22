#!/usr/bin/env python3
"""
scripts/compare_iss_solar_cycle.py — Independent multi-year solar-cycle
modulation cross-check against real ISS radiation telemetry.

Compares the pipeline's phi-driven (Usoskin) predicted GCR dose-equivalent
rate at 1 AU against real ISS TEPC dose-equivalent telemetry
(data/iss/iss_tepc_monthly.csv, fetched by scripts/fetch_iss_tepc_data.py)
over 2010-2018 — a solar minimum-to-maximum-to-declining span roughly 12x
longer than the MSL cruise window used elsewhere in this paper.

SCOPE AND CAVEAT: the ISS orbits inside Earth's magnetosphere, which
attenuates GCR flux by an amount this pipeline does not model (no
geomagnetic cutoff / rigidity transport). Absolute dose levels are
therefore NOT expected to agree and are not compared. What IS a fair,
informative test: does the pipeline's phi-driven modulation *trend* (the
same force-field physics used throughout this paper) correctly track the
*shape* of a real, independent radiation environment's solar-cycle
variation over a much longer baseline than the single MSL cruise window?
Pearson correlation is scale/offset-invariant, so it isolates exactly this
question.

Usage:
    python scripts/compare_iss_solar_cycle.py

Writes figures/iss_solar_cycle.pdf and prints the correlation statistics
used in the paper.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gcrisk.spectrum import gcr_total_flux, load_usoskin_phi
from gcrisk.transport import transport_flux_through_slab
from gcrisk.dose import dose_equivalent_rate
from gcrisk.neutron_table import h_neutron_mSv_day
from gcrisk.utils import DEFAULT_E_GRID

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

SHIELDING_X_GCM2 = 16.0  # same nominal depth used for the MSL calibration point
SHIELDING_MATERIAL = 'aluminum'


def modeled_H_rate_mSv_day(phi_MV: float) -> float:
    """Predicted dose-equivalent rate [mSv/day] at 1 AU for a given phi, at
    the nominal shielding depth — no Mars-specific trajectory machinery.
    Includes both the ion (ICRP-60 Q(L)) and HZETRN-table secondary-neutron
    contributions, consistent with H's definition elsewhere in this paper."""
    flux = gcr_total_flux(DEFAULT_E_GRID, phi_MV)
    transported = transport_flux_through_slab(
        flux, SHIELDING_X_GCM2, SHIELDING_MATERIAL, DEFAULT_E_GRID,
    )
    H_ion = dose_equivalent_rate(transported, DEFAULT_E_GRID)['H_rate_mSv_day']
    H_neutron = h_neutron_mSv_day(SHIELDING_X_GCM2, phi_MV, material=SHIELDING_MATERIAL)
    return H_ion + H_neutron


def main():
    iss = pd.read_csv(os.path.join(DATA_DIR, 'iss', 'iss_tepc_monthly.csv'))
    # Drop data-quality outliers: a median of ~0 uSv/min (vs. ~0.2-0.3 in
    # every other month) indicates an instrument/telemetry gap dominating
    # that month's 5-day sample window, not a real dose measurement.
    n_before = len(iss)
    iss = iss[iss['dose_eq60_median_uSv_min'] > 0.05].reset_index(drop=True)
    if len(iss) < n_before:
        print(f"Dropped {n_before - len(iss)} month(s) with a data-quality "
              f"outlier (near-zero median dose, likely an instrument gap).")
    iss['date'] = pd.to_datetime(iss['date'])

    phi_df = load_usoskin_phi(os.path.join(DATA_DIR, 'usoskin', 'phi_monthly.csv'))
    phi_df = phi_df.reset_index().drop(columns=['date']).rename(columns={'datetime': 'date'})
    phi_df['date'] = phi_df['date'].dt.to_period('M').dt.to_timestamp()
    iss['date'] = iss['date'].dt.to_period('M').dt.to_timestamp()

    merged = pd.merge(iss, phi_df[['date', 'phi_MV']], on='date', how='inner')
    print(f"Merged {len(merged)} months of overlapping data "
          f"({merged['date'].min().date()} to {merged['date'].max().date()})")

    print("Computing modeled H rate for each month's phi...")
    merged['H_model_mSv_day'] = merged['phi_MV'].apply(modeled_H_rate_mSv_day)

    # ISS DOSE_EQ60 is uSv/min -> mSv/day for readability (still not
    # absolute-comparable to H_model_mSv_day; used only for the trend/shape test)
    merged['H_iss_mSv_day'] = merged['dose_eq60_median_uSv_min'] * 1440.0 / 1000.0

    r, p_value = pearsonr(merged['H_iss_mSv_day'], merged['H_model_mSv_day'])
    print(f"\nPearson r (ISS DOSE_EQ60 vs. modeled H, {len(merged)} months) = "
          f"{r:.3f}  (p={p_value:.2e})")
    print("Note: r is expected to be POSITIVE here — both series are dose "
          "quantities that respond the same way to phi (both rise when "
          "solar modulation weakens), unlike the dose-vs-phi correlation "
          "reported elsewhere in this paper, which is negative by construction.")

    merged.to_csv(os.path.join(DATA_DIR, 'iss', 'iss_solar_cycle_comparison.csv'), index=False)

    # --- Figure: twin-axis time series over 2010-2018 ---
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(merged['date'], merged['H_iss_mSv_day'], color='tab:blue',
              marker='o', markersize=3, linewidth=1, label='ISS TEPC (real telemetry)')
    ax1.set_ylabel('ISS dose-equivalent rate [mSv/day]', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xlabel('Date')

    ax2 = ax1.twinx()
    ax2.plot(merged['date'], merged['H_model_mSv_day'], color='tab:red',
              marker='s', markersize=3, linewidth=1, label='Pipeline model (1 AU, 16 g/cm² Al)')
    ax2.set_ylabel('Modeled dose-equivalent rate [mSv/day]', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    # Not inverted: both series are dose quantities and are positively
    # correlated (both rise/fall together with solar modulation).

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    ax1.set_title(f'Solar-cycle modulation: real ISS telemetry vs. pipeline model '
                  f'(r={r:.2f}, n={len(merged)} months, 2010–2018)')
    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, 'iss_solar_cycle.pdf'))
    fig.savefig(os.path.join(FIG_DIR, 'iss_solar_cycle.png'), dpi=150)
    print(f"\nSaved figures/iss_solar_cycle.pdf")


if __name__ == '__main__':
    main()
