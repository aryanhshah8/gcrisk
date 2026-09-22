#!/usr/bin/env python3
"""
scripts/build_al_slab_reference.py

Build the Python CSDA transport prediction reference for the
GCR proton/aluminum-slab benchmark.

This computes what the Python pipeline predicts for:
  - A 230 MeV proton beam through 20 g/cm² aluminum
  - Exit energy after the slab
  - Residual Bragg-peak depth in water

The benchmark validates that the Python CSDA range model
agrees with Geant4/TOPAS to ±5% on Bragg-peak depth.

Outputs:
    data/topas/gcr_proton_al_slab_python_prediction.csv
    Prints comparison metrics.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gcrisk.transport import proton_range, energy_after_slab

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def main():
    # Test matrix: (E_in_MeV, x_al_gcm2) pairs
    cases = [
        (100.0, 5.0),
        (200.0, 10.0),
        (200.0, 16.0),
        (230.0, 20.0),   # Main benchmark case (matches TOPAS geometry.txt)
        (400.0, 20.0),
        (500.0, 30.0),
    ]

    rows = []
    for E_in, x_al in cases:
        E_out = energy_after_slab(E_in, x_al, 'aluminum', Z_ion=1, A_ion=1)

        # Residual range in water after Al slab
        R_in_al = float(proton_range(np.array([E_in]), 'aluminum')[0])
        R_out_water = float(proton_range(np.array([E_out]), 'water')[0]) if E_out > 0 else 0.0

        # Reference range in water without slab
        R_free_water = float(proton_range(np.array([E_in]), 'water')[0])

        # Water-equivalent thickness of the Al slab
        # R_water_equiv = R_free_water - R_out_water  (should ≈ x_al * (rho_water/rho_al) * stopping_ratio)
        wet = R_free_water - R_out_water

        rows.append({
            'E_in_MeV': E_in,
            'x_al_gcm2': x_al,
            'E_out_MeV': round(E_out, 2),
            'R_in_al_gcm2': round(R_in_al, 3),
            'R_out_water_gcm2': round(R_out_water, 3),
            'R_free_water_gcm2': round(R_free_water, 3),
            'WET_gcm2': round(wet, 3),
            'stopped': E_out <= 0,
        })

        status = "STOPPED" if E_out <= 0 else f"E_out={E_out:.1f} MeV, R_water={R_out_water:.1f} g/cm²"
        print(f"  {E_in:6.0f} MeV → {x_al:4.0f} g/cm² Al → {status}")

    df = pd.DataFrame(rows)
    out_path = os.path.join(REPO_ROOT, 'data', 'topas', 'gcr_proton_al_slab_python_prediction.csv')
    df.to_csv(out_path, index=False)
    print(f"\nPython CSDA prediction saved → {out_path}")

    # Highlight the main benchmark case
    main_case = df[df['E_in_MeV'] == 230.0].iloc[0]
    print(f"\nMain benchmark case (230 MeV → 20 g/cm² Al):")
    print(f"  E_out predicted:       {main_case['E_out_MeV']:.1f} MeV")
    print(f"  Residual range water:  {main_case['R_out_water_gcm2']:.1f} g/cm²  "
          f"(≈ {main_case['R_out_water_gcm2']/1.0:.1f} cm in water)")
    print(f"  Water-equiv thickness: {main_case['WET_gcm2']:.1f} g/cm²")
    print(f"\nValidation target: TOPAS Bragg-peak depth should agree within ±5%.")


if __name__ == '__main__':
    main()
