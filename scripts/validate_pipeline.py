#!/usr/bin/env python3
"""
scripts/validate_pipeline.py — Run validation checks and print pass/fail.

Checks:
  1. Proton LIS spectrum normalization vs published directional flux bounds
  2. Proton CSDA range in water vs NIST tabulated value
  3. Full pipeline absorbed dose rate vs MSL RAD measurement
  4. HZETRN benchmark comparison
  5. Dose equivalent rate and Q_eff vs MSL RAD
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gcrisk.spectrum import lis_proton, force_field_modulation, load_usoskin_phi
from gcrisk.transport import proton_range
from gcrisk.dose import integrate_mission_dose
from gcrisk.trajectory import generate_trajectory
from gcrisk.utils import DEFAULT_E_GRID


def main():
    print("=" * 60)
    print("GCR DOSIMETRY PIPELINE — VALIDATION CHECKS")
    print("=" * 60)
    all_pass = True

    # ---------------------------------------------------------------
    # VALIDATION 1: Proton LIS spectrum at phi=550 MV
    # ---------------------------------------------------------------
    print("\nVALIDATION 1: Proton flux at phi=550 MV, E=1000 MeV/n")
    E_test = np.array([1000.0])
    flux_val = force_field_modulation(E_test, 550.0, lis_proton)
    flux_1GeV = float(flux_val[0])

    # The modulated directional flux at 1 GeV (per steradian)
    expected_lo, expected_hi = 4e-4, 4e-3
    status = expected_lo < flux_1GeV < expected_hi
    print(f"  Expected: {expected_lo:.1e}–{expected_hi:.1e} cm^-2 s^-1 MeV^-1 sr^-1")
    print(f"  Result:   {flux_1GeV:.3e}")
    print(f"  Status:   {'PASS' if status else 'FAIL'}")
    if not status:
        all_pass = False

    # ---------------------------------------------------------------
    # VALIDATION 2: Proton range in water (CSDA)
    # ---------------------------------------------------------------
    print("\nVALIDATION 2: Proton range in water (CSDA)")
    R_200 = float(proton_range(np.array([200.0]), 'water')[0])
    expected_R = 25.9  # NIST value at 200 MeV
    tol = 0.05  # 5%
    status = abs(R_200 - expected_R) / expected_R < tol
    print(f"  Expected: {expected_R} g/cm² at 200 MeV (NIST value)")
    print(f"  Result:   {R_200:.1f} g/cm²")
    print(f"  Status:   {'PASS' if status else 'FAIL'} (within {tol*100:.0f}%)")
    if not status:
        all_pass = False

    # ---------------------------------------------------------------
    # VALIDATION 3: Full pipeline vs MSL RAD
    # ---------------------------------------------------------------
    print("\nVALIDATION 3: Full pipeline vs MSL RAD")
    print("  Computing... (this may take a minute)")

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'usoskin')
    phi_df = load_usoskin_phi(os.path.join(data_dir, 'phi_transit_frozen.csv'))

    traj = generate_trajectory('2011-11-26', phi_df=phi_df)

    # Use the shared default grid so validation and production runs exercise
    # the same energy support.
    E_grid = DEFAULT_E_GRID

    result = integrate_mission_dose(
        traj,
        shielding_x_gcm2=16.0,
        shielding_material='aluminum',
        phi_df=phi_df,
        E_grid_MeV=E_grid,
    )

    D_rate_mean = np.mean(result['D_rate_daily'])
    expected_D = 1.84
    margin = 0.25  # 25%
    D_lo = expected_D * (1.0 - margin)
    D_hi = expected_D * (1.0 + margin)
    status = D_lo < D_rate_mean < D_hi

    print(f"  Expected: {D_lo:.2f}–{D_hi:.2f} mGy/day (RAD: 1.84 ± 25%)")
    print(f"  Result:   {D_rate_mean:.2f} mGy/day")
    print(f"  Shielding: 16 g/cm² aluminum (MSL cruise stage equivalent)")
    print(f"  Launch date: 2011-11-26 (actual MSL launch)")
    print(f"  Mission duration: {result['mission_duration_days']} days")
    print(f"  Total dose: {result['D_total_mGy']:.1f} mGy")
    print(f"  Status:   {'PASS' if status else 'FAIL'}")
    if not status:
        all_pass = False

    # ---------------------------------------------------------------
    # VALIDATION 4: HZETRN benchmark comparison
    # ---------------------------------------------------------------
    print("\nVALIDATION 4: HZETRN benchmark")
    print("  Re-using trajectory and result from Validation 3...")

    hzetrn_lo = 1.45
    hzetrn_hi = 2.20
    status_hzetrn = hzetrn_lo < D_rate_mean < hzetrn_hi

    print(f"  HZETRN range: {hzetrn_lo:.2f}–{hzetrn_hi:.2f} mGy/day")
    print(f"  Pipeline result: {D_rate_mean:.2f} mGy/day")

    # Compute pipeline species-resolved dose breakdown (single representative day)
    from gcrisk.spectrum import gcr_total_flux
    from gcrisk.dose import _precompute_transport_factors, _apply_transport_factors, dose_rate_from_flux
    phi_mid = float(traj['phi_MV'].iloc[len(traj) // 2])
    flux_mid = gcr_total_flux(E_grid, phi_mid)
    tfactors = _precompute_transport_factors(16.0, 'aluminum', E_grid)
    flux_transported = _apply_transport_factors(flux_mid, E_grid, tfactors)
    dose_species = dose_rate_from_flux(flux_transported, E_grid)
    D_total_species = dose_species['dose_rate_total']
    D_by_species = dose_species['absorbed_dose_rate']
    D_H = D_by_species.get('H', 0.0)
    D_He = D_by_species.get('He', 0.0)
    D_HZE = sum(v for k, v in D_by_species.items() if k not in ('H', 'He', 'neutron'))
    D_n = D_by_species.get('neutron', 0.0)
    pipe_H_frac = D_H / D_total_species if D_total_species > 0 else 0
    pipe_He_frac = D_He / D_total_species if D_total_species > 0 else 0
    pipe_HZE_frac = D_HZE / D_total_species if D_total_species > 0 else 0
    pipe_n_frac = D_n / D_total_species if D_total_species > 0 else 0

    print(f"  Species breakdown comparison (absorbed dose fraction):")
    print(f"    {'Species':<10} {'HZETRN ref.':<22} {'Pipeline':<15}")
    print(f"    {'H':<10} {'~49%':<22} {pipe_H_frac*100:>5.1f}%")
    print(f"    {'He':<10} {'~17%':<22} {pipe_He_frac*100:>5.1f}%")
    print(f"    {'HZE':<10} {'~34%':<22} {pipe_HZE_frac*100:>5.1f}%")
    if pipe_n_frac > 0.001:
        print(f"    {'neutron':<10} {'(in H only)':<22} {pipe_n_frac*100:>5.1f}%")
    print(f"  Remaining discrepancy sources:")
    print(f"    - Hemisphere-averaged slab vs HZETRN ray-by-ray body model")
    print(f"    - Charged-fragment surrogate instead of full fragmentation transport")
    print(f"  Status:   {'PASS' if status_hzetrn else 'FAIL'}")
    if not status_hzetrn:
        all_pass = False

    # ---------------------------------------------------------------
    # VALIDATION 5: Dose equivalent rate and Q_eff vs MSL RAD
    # ---------------------------------------------------------------
    print("\nVALIDATION 5: Dose equivalent rate vs MSL RAD")

    H_result_val = result['H_total_mSv']
    H_rate_val = H_result_val / result['mission_duration_days']  # mSv/day

    H_rad_published = 4.81
    Q_rad_published = 2.62
    H_rad_lo = H_rad_published * 0.65
    H_rad_hi = H_rad_published * 1.35

    Q_pipeline = H_rate_val / D_rate_mean if D_rate_mean > 0 else 0.0

    status_H = H_rad_lo < H_rate_val < H_rad_hi
    print(f"  Expected H rate: {H_rad_lo:.2f}–{H_rad_hi:.2f} mSv/day (RAD: 4.81 ± 35%)")
    print(f"  Pipeline H rate: {H_rate_val:.2f} mSv/day")
    print(f"  Pipeline Q_eff:  {Q_pipeline:.2f} (RAD measured: {Q_rad_published:.2f})")
    print(f"  Status:   {'PASS' if status_H else 'FAIL'}")
    if not status_H:
        all_pass = False

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    if all_pass:
        print("ALL VALIDATIONS PASSED")
    else:
        print("SOME VALIDATIONS FAILED")
    print("=" * 60)
    print("\nKey pipeline vs published comparison:")
    print(f"  Absorbed dose rate:  {D_rate_mean:.2f} mGy/day (RAD: 1.84, HZETRN: 1.70–2.10)")
    print(f"  Dose equiv. rate:    {H_rate_val:.2f} mSv/day (RAD: 4.81)")
    print(f"  Q_effective:         {Q_pipeline:.2f} (RAD: {Q_rad_published:.2f})")

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
