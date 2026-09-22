#!/usr/bin/env python3
"""
scripts/plot_shielding_uncertainty.py

Publication figure: GCR absorbed dose rate vs spacecraft shielding thickness,
with physics uncertainty bands from the LHS ensemble.

Produces two panels:
  Left:  D_rate [mGy/day] vs x [g/cm²] for Al and PE
         - Pipeline central values (solid lines)
         - LHS p5/p95 uncertainty band (shaded)
         - MSL RAD point
         - HZETRN envelope
  Right: per-species dose fraction vs shielding (bar chart) at 3 key depths

Output: figures/shielding_uncertainty.png  (also .pdf for paper submission)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gcrisk.trajectory import generate_trajectory
from gcrisk.spectrum import load_usoskin_phi, gcr_total_flux
from gcrisk.dose import integrate_mission_dose, _precompute_transport_factors, _apply_transport_factors, dose_rate_from_flux
from gcrisk.utils import DEFAULT_E_GRID

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIGURES_DIR = os.path.join(REPO_ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Published reference values
# ---------------------------------------------------------------------------
MSL_RAD_X    = 16.0    # g/cm²  (Zeitlin 2013 cruise stage equivalent)
MSL_RAD_D    = 1.84    # mGy/day
MSL_RAD_UNCERT = 0.25  # ±25% total uncertainty band

# HZETRN envelope [lo, hi] mGy/day
HZETRN_TABLE = {
    5:  (2.20, 3.60),
    10: (1.90, 2.60),
    16: (1.45, 2.20),
    30: (1.10, 1.80),
}


def compute_shielding_scan(traj, phi_df, thicknesses, material, n_unc_samples=50):
    """Compute D_rate central + p5/p95 uncertainty band across shielding thicknesses."""
    from gcrisk.uncertainty import lhs_samples, PARAM_NAMES
    import numpy as np

    # LHS samples for physics uncertainty
    samples = lhs_samples(n_unc_samples, seed=42)

    central = []
    p5_arr  = []
    p95_arr = []

    for x in thicknesses:
        # Central value (nominal)
        res = integrate_mission_dose(
            traj, shielding_x_gcm2=x, shielding_material=material,
            phi_df=phi_df, E_grid_MeV=DEFAULT_E_GRID,
        )
        D_central = float(np.mean(res['D_rate_daily']))
        central.append(D_central)

        # Uncertainty samples — only vary phi_scale, LIS_norm, neutron_H (the physics ones)
        D_samp = []
        for i in range(n_unc_samples):
            phi_scale   = float(samples['phi_scale'][i])
            lis_scale   = float(samples['LIS_norm'][i])
            n_scale     = float(samples['neutron_H'][i])
            traj_p = traj.copy()
            traj_p['phi_MV'] = traj_p['phi_MV'] * phi_scale
            try:
                r = integrate_mission_dose(
                    traj_p, shielding_x_gcm2=x, shielding_material=material,
                    phi_df=phi_df, E_grid_MeV=DEFAULT_E_GRID,
                    lis_norm_scale=lis_scale,
                    neutron_yield_scale=n_scale,
                )
                D_samp.append(float(np.mean(r['D_rate_daily'])))
            except Exception:
                D_samp.append(D_central)

        p5_arr.append(float(np.percentile(D_samp, 5)))
        p95_arr.append(float(np.percentile(D_samp, 95)))
        print(f"  {material:12s} x={x:5.0f} g/cm²  "
              f"D={D_central:.3f} [{p5_arr[-1]:.3f}–{p95_arr[-1]:.3f}] mGy/day")

    return np.array(central), np.array(p5_arr), np.array(p95_arr)


def compute_species_fractions(phi_mid, thicknesses, material):
    """Per-species dose fraction at each shielding depth."""
    species_keys = ['H', 'He', 'C', 'O', 'Si', 'Fe']
    fractions = {sp: [] for sp in species_keys}

    flux_bare = gcr_total_flux(DEFAULT_E_GRID, phi_mid)
    for x in thicknesses:
        if x > 0:
            tf = _precompute_transport_factors(x, material, DEFAULT_E_GRID)
            flux = _apply_transport_factors(flux_bare, DEFAULT_E_GRID, tf)
        else:
            flux = flux_bare
        dr = dose_rate_from_flux(flux, DEFAULT_E_GRID)
        total = dr['dose_rate_total']
        for sp in species_keys:
            d = dr['absorbed_dose_rate'].get(sp, 0.0)
            fractions[sp].append(d / total if total > 0 else 0.0)
    return fractions


def make_figure(n_samples=50):
    data_dir = os.path.join(REPO_ROOT, 'data', 'usoskin')
    phi_df = load_usoskin_phi(os.path.join(data_dir, 'phi_transit_frozen.csv'))
    traj = generate_trajectory('2011-11-26', phi_df=phi_df)
    phi_mid = float(traj['phi_MV'].iloc[len(traj)//2])

    thicknesses = [0, 2, 5, 10, 16, 20, 30, 40]
    thick_arr = np.array(thicknesses, dtype=float)

    print("Computing shielding scan (Al)...")
    D_al, p5_al, p95_al = compute_shielding_scan(
        traj, phi_df, thicknesses, 'aluminum', n_samples)
    print("Computing shielding scan (PE)...")
    D_pe, p5_pe, p95_pe = compute_shielding_scan(
        traj, phi_df, thicknesses, 'polyethylene', n_samples)

    bar_depths = [5, 16, 30]
    frac_al = compute_species_fractions(phi_mid, bar_depths, 'aluminum')

    # -----------------------------------------------------------------------
    # Plot -- two SEPARATE figures (previously one combined two-panel figure;
    # split so each gets its own caption/results discussion in the paper).
    # -----------------------------------------------------------------------
    color_al = '#1f77b4'
    color_pe = '#d62728'

    # --- Figure A: shielding scan with uncertainty bands ---
    figA, axA = plt.subplots(figsize=(6.5, 5.5))

    axA.fill_between(thick_arr, p5_al, p95_al, color=color_al, alpha=0.20,
                      label='Al physics uncertainty (p5–p95)')
    axA.fill_between(thick_arr, p5_pe, p95_pe, color=color_pe, alpha=0.20,
                      label='PE physics uncertainty (p5–p95)')
    axA.plot(thick_arr, D_al, '-o', color=color_al, lw=2, ms=5,
              label='Aluminum (pipeline)')
    axA.plot(thick_arr, D_pe, '-s', color=color_pe, lw=2, ms=5,
              label='Polyethylene (pipeline)')

    hzetrn_x  = sorted(HZETRN_TABLE.keys())
    hzetrn_lo = [HZETRN_TABLE[x][0] for x in hzetrn_x]
    hzetrn_hi = [HZETRN_TABLE[x][1] for x in hzetrn_x]
    axA.fill_between(hzetrn_x, hzetrn_lo, hzetrn_hi, color='gray', alpha=0.25,
                      label='HZETRN envelope')

    axA.errorbar([MSL_RAD_X], [MSL_RAD_D],
                  yerr=[[MSL_RAD_D * MSL_RAD_UNCERT], [MSL_RAD_D * MSL_RAD_UNCERT]],
                  fmt='k*', ms=12, capsize=5, lw=2, label='MSL RAD')

    axA.set_xlabel('Shielding thickness [g/cm²]', fontsize=11)
    axA.set_ylabel('Absorbed dose rate [mGy/day]', fontsize=11)
    axA.set_title('GCR Absorbed Dose vs Shielding Thickness — MSL Transit (2011-11-26)',
                   fontsize=10.5, fontweight='bold')
    axA.legend(fontsize=8.5, loc='upper right')
    axA.set_xlim(-1, 42)
    axA.set_ylim(0, None)
    axA.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        out = os.path.join(FIGURES_DIR, f'shielding_scan.{ext}')
        figA.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close(figA)

    # --- Figure B: species dose fractions ---
    species_colors = {
        'H':  '#1f77b4',
        'He': '#2ca02c',
        'C':  '#ff7f0e',
        'O':  '#9467bd',
        'Si': '#8c564b',
        'Fe': '#d62728',
    }
    species_labels = {'H': 'Proton', 'He': 'He', 'C': 'C', 'O': 'O', 'Si': 'Si', 'Fe': 'Fe'}
    species_order = ['H', 'He', 'C', 'O', 'Si', 'Fe']

    figB, axB = plt.subplots(figsize=(6.5, 5.5))
    x_pos = np.arange(len(bar_depths))
    bar_width = 0.12
    offsets = np.linspace(-0.3, 0.3, len(species_order))

    for i, sp in enumerate(species_order):
        vals = frac_al[sp]
        axB.bar(x_pos + offsets[i], vals, bar_width,
                color=species_colors[sp], label=species_labels[sp], alpha=0.85)

    axB.set_xticks(x_pos)
    axB.set_xticklabels([f'{d} g/cm²\nAl' for d in bar_depths], fontsize=10)
    axB.set_ylabel('Dose fraction', fontsize=11)
    axB.set_title('Species Dose Breakdown at 3 Shielding Depths',
                   fontsize=10.5, fontweight='bold')
    axB.legend(fontsize=9, ncol=3, loc='upper right')
    axB.set_ylim(0, 0.7)
    axB.grid(True, alpha=0.3, axis='y')

    axB.text(0.02, 0.97,
              'HZETRN ref. (16 g/cm²):\nH~49%  He~17%  HZE~34%',
              transform=axB.transAxes, fontsize=8,
              verticalalignment='top',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        out = os.path.join(FIGURES_DIR, f'species_fractions.{ext}')
        figB.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close(figB)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--samples', type=int, default=50,
                   help='Number of LHS samples for uncertainty bands (default 50)')
    args = p.parse_args()
    make_figure(n_samples=args.samples)
