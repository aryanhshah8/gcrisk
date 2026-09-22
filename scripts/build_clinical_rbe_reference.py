#!/usr/bin/env python3
"""
scripts/build_clinical_rbe_reference.py

Build reference CSVs for the clinical α/β benchmark cases
(Prostate3: α/β = 3 Gy; HN10: α/β = 10 Gy) from the checked-in V79 LETd
spectrum without running TOPAS.

Justification: the V79 benchmark established that the Python Wedenberg and
McNamara formulas match OpenTOPAS-RBE to < 5×10⁻⁸ absolute error.  The LETd
spectrum from a proton-water phantom is cell-line-independent (it depends only
on the physics, not the biology), so the V79 LETd column can be directly reused
for any cell line by applying a different α/β ratio.

Usage:
    python scripts/build_clinical_rbe_reference.py

Outputs:
    data/topas/proton_water_prostate3_reference.csv
    data/topas/proton_water_hn10_reference.csv
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gcrisk.rbe import wedenberg_rbe, mcnamara_rbe

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def build_reference(v79_df: pd.DataFrame, alpha_beta_x_Gy: float, prescribed_dose_Gy: float) -> pd.DataFrame:
    """Build a reference DataFrame for a new cell line from V79 LETd spectrum."""
    df = v79_df.copy()
    df['alpha_beta_x_Gy'] = alpha_beta_x_Gy
    df['dose_Gy'] = prescribed_dose_Gy

    rbe_w = np.array([
        wedenberg_rbe(prescribed_dose_Gy, float(let), alpha_beta_x_Gy)
        for let in df['LETd_keV_um']
    ])
    rbe_mc = np.array([
        mcnamara_rbe(prescribed_dose_Gy, float(let), alpha_beta_x_Gy)
        for let in df['LETd_keV_um']
    ])

    df['RBE_wedenberg_ref'] = rbe_w
    df['RBE_mcnamara_ref'] = rbe_mc
    df['dose_RBE_wedenberg_ref'] = rbe_w * prescribed_dose_Gy
    df['dose_RBE_mcnamara_ref'] = rbe_mc * prescribed_dose_Gy

    return df[[
        'x_bin', 'y_bin', 'z_bin',
        'physical_dose_Gy', 'z_depth_cm', 'LETd_keV_um',
        'RBE_wedenberg_ref', 'RBE_mcnamara_ref',
        'alpha_beta_x_Gy', 'dose_Gy',
        'dose_RBE_wedenberg_ref', 'dose_RBE_mcnamara_ref',
    ]]


def main():
    v79_path = os.path.join(REPO_ROOT, 'data', 'topas', 'proton_water_v79_reference.csv')
    v79_df = pd.read_csv(v79_path)

    # Prescribed dose for clinical benchmark (2 Gy per fraction — standard hypofractionation)
    prescribed_dose_Gy = 2.0

    clinical_cases = [
        ('prostate3', 3.0,  'proton_water_prostate3_reference.csv'),
        ('hn10',      10.0, 'proton_water_hn10_reference.csv'),
    ]

    for name, ab, out_filename in clinical_cases:
        df_ref = build_reference(v79_df, ab, prescribed_dose_Gy)
        out_path = os.path.join(REPO_ROOT, 'data', 'topas', out_filename)
        df_ref.to_csv(out_path, index=False)
        rbe_w_mean = float(df_ref['RBE_wedenberg_ref'][df_ref['LETd_keV_um'] > 0.1].mean())
        rbe_mc_mean = float(df_ref['RBE_mcnamara_ref'][df_ref['LETd_keV_um'] > 0.1].mean())
        print(f"[{name:12s}] α/β={ab:.0f} Gy  "
              f"mean RBE (Wedenberg)={rbe_w_mean:.4f}  "
              f"mean RBE (McNamara)={rbe_mc_mean:.4f}  "
              f"→ {out_path}")

    print("Done.")


if __name__ == '__main__':
    main()
