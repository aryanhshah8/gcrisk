#!/usr/bin/env python3
"""
scripts/calibrate_global_scale.py

Compute the _GLOBAL_DOSE_SCALE for the composition-constrained calibration.

With _GLOBAL_DOSE_SCALE = 1.0, the ACE/CRIS-constrained relative factors are
applied but the absolute normalization is not yet matched to MSL/RAD.
This script runs the pipeline at the calibration conditions and finds the
scale factor that produces D = 1.84 mGy/day.

After running, paste the printed value into gcrisk/spectrum.py as _GLOBAL_DOSE_SCALE.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ---- Temporarily override _GLOBAL_DOSE_SCALE = 1.0 before any imports ----
import gcrisk.spectrum as _spec
_spec._GLOBAL_DOSE_SCALE = 1.0
_spec._SPECIES_FLUX_CALIBRATION = _spec._build_species_calibration()

from gcrisk.trajectory import generate_trajectory
from gcrisk.dose import integrate_mission_dose
from gcrisk.utils import DEFAULT_E_GRID

TARGET_D_MGYPERDAY = 1.84   # MSL/RAD; Zeitlin et al. 2013
CALIB_X_GCM2      = 16.0   # calibration shielding [g/cm² Al]
CALIB_PHI_MV      = 481.0  # MSL mean modulation potential [MV]
CALIB_DAYS        = 259

print("Computing composition-constrained calibration factors at scale = 1.0 ...")

# Build a flat trajectory at the calibration phi
import pandas as pd
from datetime import date, timedelta
dates = [date(2011, 11, 26) + timedelta(days=i) for i in range(CALIB_DAYS)]
traj = pd.DataFrame({
    'date': dates,
    'r_AU': [1.262] * CALIB_DAYS,
    'phi_MV': [CALIB_PHI_MV] * CALIB_DAYS,
    'v_AU_day': [0.01] * CALIB_DAYS,
})

result = integrate_mission_dose(
    traj,
    shielding_x_gcm2=CALIB_X_GCM2,
    shielding_material='aluminum',
    E_grid_MeV=DEFAULT_E_GRID,
)

D_total_mGy = result['D_total_mGy']
D_per_day   = D_total_mGy / CALIB_DAYS
H_per_day   = result['H_total_mSv'] / CALIB_DAYS
Q_eff       = (result['H_total_mSv'] / result['D_total_mGy']) if result['D_total_mGy'] > 0 else 0.0

scale = TARGET_D_MGYPERDAY / D_per_day

print(f"\n=== Calibration result (scale = 1.0) ===")
print(f"  D_per_day : {D_per_day:.4f} mGy/day  (target {TARGET_D_MGYPERDAY:.2f})")
print(f"  H_per_day : {H_per_day:.4f} mSv/day")
print(f"  Q_eff     : {Q_eff:.4f}")
print(f"\n  Required _GLOBAL_DOSE_SCALE: {scale:.6f}")
print(f"\nPaste this into gcrisk/spectrum.py:")
print(f"  _GLOBAL_DOSE_SCALE = {scale:.6f}")

# Show what species fractions look like at scale=1.0 and predicted scale
from gcrisk.spectrum import gcr_total_flux
from gcrisk.dose import dose_rate_from_flux, _precompute_transport_factors, _apply_transport_factors

E_grid = DEFAULT_E_GRID
flux0 = gcr_total_flux(E_grid, CALIB_PHI_MV)
tfactors = _precompute_transport_factors(CALIB_X_GCM2, 'aluminum', E_grid)
flux_t = _apply_transport_factors(flux0, E_grid, tfactors)
dr = dose_rate_from_flux(flux_t, E_grid)
d_by_sp = dr['absorbed_dose_rate']
d_tot = dr['dose_rate_total']
if d_tot > 0:
    print(f"\n=== Species dose fractions at 16 g/cm² Al, phi={CALIB_PHI_MV} MV ===")
    for k in ['H', 'He', 'C', 'O', 'Si', 'Fe']:
        if k in d_by_sp:
            print(f"  {k:3s}: {d_by_sp[k]/d_tot*100:5.1f}%")
    hze = sum(v for k, v in d_by_sp.items() if k not in ('H', 'He', 'neutron'))
    print(f"  HZE total: {hze/d_tot*100:5.1f}%")
    print(f"  H:         {d_by_sp.get('H',0)/d_tot*100:5.1f}%")
    print(f"  He:        {d_by_sp.get('He',0)/d_tot*100:5.1f}%")
