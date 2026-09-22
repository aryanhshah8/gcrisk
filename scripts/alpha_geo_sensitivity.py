#!/usr/bin/env python3
"""
scripts/alpha_geo_sensitivity.py

Sensitivity scan over the geometric blending parameter alpha_geo in
_precompute_transport_factors (dose.py line ~277).

alpha_geo controls the blend between direct 1D exponential survival (s1d)
and hemisphere-averaged exponential integral survival (shemi):

    survival = (1 - alpha_geo) * s1d + alpha_geo * shemi

The default value of 0.5 is an assumption (equal weight to direct vs.
diffuse paths). This script shows how absorbed dose and dose equivalent
vary with alpha_geo at 16 g/cm² aluminum, fixing phi = 481 MV (MSL epoch).

Outputs:
  - Table to stdout
  - Fraction change in D and H relative to alpha_geo = 0.5 baseline
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gcr.dose import integrate_mission_dose
from gcr.spectrum import load_usoskin_phi
from gcr.trajectory import generate_trajectory
from gcr.utils import DEFAULT_E_GRID

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'usoskin')


def run_scan(alpha_geo_values, shielding_x=16.0, material='aluminum'):
    phi_df = load_usoskin_phi(os.path.join(DATA_DIR, 'phi_transit_frozen.csv'))
    traj = generate_trajectory('2011-11-26', phi_df=phi_df)

    results = []
    for ag in alpha_geo_values:
        res = integrate_mission_dose(
            traj,
            shielding_x_gcm2=shielding_x,
            shielding_material=material,
            phi_df=phi_df,
            E_grid_MeV=DEFAULT_E_GRID,
            alpha_geo=ag,
        )
        results.append({
            'alpha_geo': ag,
            'D_total_mGy': res['D_total_mGy'],
            'H_total_mSv': res['H_total_mSv'],
            'D_rate_mGy_day': res['D_total_mGy'] / res['mission_duration_days'],
            'H_rate_mSv_day': res['H_total_mSv'] / res['mission_duration_days'],
        })

    # Baseline at alpha_geo = 0.5
    baseline = next(r for r in results if abs(r['alpha_geo'] - 0.5) < 1e-9)
    D_base = baseline['D_rate_mGy_day']
    H_base = baseline['H_rate_mSv_day']

    print(f"\nalpha_geo sensitivity — {shielding_x:.0f} g/cm² {material}, "
          f"259-day MSL transit (phi=481 MV)\n")
    print(f"{'alpha_geo':>10s}  {'D [mGy/d]':>12s}  {'ΔD %':>8s}  "
          f"{'H [mSv/d]':>12s}  {'ΔH %':>8s}")
    print("-" * 60)
    for r in results:
        dD = (r['D_rate_mGy_day'] - D_base) / D_base * 100.0
        dH = (r['H_rate_mSv_day'] - H_base) / H_base * 100.0
        marker = " ← default" if abs(r['alpha_geo'] - 0.5) < 1e-9 else ""
        print(f"{r['alpha_geo']:>10.2f}  {r['D_rate_mGy_day']:>12.4f}  "
              f"{dD:>+7.1f}%  {r['H_rate_mSv_day']:>12.4f}  {dH:>+7.1f}%{marker}")

    D_vals = [r['D_rate_mGy_day'] for r in results]
    H_vals = [r['H_rate_mSv_day'] for r in results]
    print(f"\nD range: {min(D_vals):.4f} – {max(D_vals):.4f} mGy/day "
          f"({(max(D_vals)/min(D_vals) - 1)*100:.1f}% spread)")
    print(f"H range: {min(H_vals):.4f} – {max(H_vals):.4f} mSv/day "
          f"({(max(H_vals)/min(H_vals) - 1)*100:.1f}% spread)")
    return results


if __name__ == '__main__':
    alpha_values = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    run_scan(alpha_values)
