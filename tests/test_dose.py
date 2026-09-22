"""Tests for gcrisk/dose.py"""

import numpy as np
import pytest
import pandas as pd
from datetime import date, timedelta

from gcrisk.dose import (
    _apply_transport_factors,
    _precompute_transport_factors,
    let_from_energy,
    quality_factor_icrp60,
    dose_rate_from_flux,
    dose_equivalent_rate,
    integrate_mission_dose,
    neutron_h10,
)
from gcrisk.spectrum import gcr_total_flux


class TestDoseModule:

    def test_let_from_energy_proton(self):
        """LET for 200 MeV proton in tissue: expect 0.3–1.0 keV/μm (high energy, low LET)."""
        LET = let_from_energy(np.array([200.0]), Z=1, material='tissue')
        assert LET.shape == (1,)
        assert 0.3 < LET[0] < 1.0, f"200 MeV proton LET in tissue: {LET[0]:.3f} keV/μm"

    def test_let_from_energy_iron(self):
        """LET for 500 MeV/n Fe-56 in tissue: must be >> proton LET (Z²=676 scaling), > 100 keV/μm."""
        LET_Fe = let_from_energy(np.array([500.0]), Z=26, material='tissue')
        LET_p = let_from_energy(np.array([500.0]), Z=1, material='tissue')
        assert LET_Fe[0] > 100.0, f"Fe-56 LET at 500 MeV/n: {LET_Fe[0]:.1f} keV/μm (expect >100)"
        assert LET_Fe[0] > LET_p[0], "Fe LET must exceed proton LET at same energy/nucleon"

    def test_quality_factor_low_let(self):
        """Q(L) = 1 for L < 10 keV/μm (proton regime)."""
        L = np.array([0.5, 1.0, 5.0, 9.9])
        Q = quality_factor_icrp60(L)
        np.testing.assert_array_equal(Q, np.ones(4),
                                      err_msg=f"Q should be 1 for L < 10 keV/μm, got {Q}")

    def test_quality_factor_mid_let(self):
        """Q(L) = 0.32×L - 2.2 for L = 50 keV/μm. Expected value: 13.8."""
        L = np.array([50.0])
        Q = quality_factor_icrp60(L)
        expected = 0.32 * 50.0 - 2.2  # = 13.8
        assert abs(Q[0] - expected) < 1e-6, f"Q(50 keV/μm) = {Q[0]:.4f}, expected {expected}"

    def test_quality_factor_high_let(self):
        """Q(L) = 300/√L for L = 900 keV/μm. Expected value: 10.0."""
        L = np.array([900.0])
        Q = quality_factor_icrp60(L)
        expected = 300.0 / np.sqrt(900.0)  # = 10.0
        assert abs(Q[0] - expected) < 1e-6, f"Q(900 keV/μm) = {Q[0]:.4f}, expected {expected}"

    def test_dose_rate_from_flux_proton_only(self):
        """Proton-only flux on a 10-point grid: dose_rate_mGy_day must be > 0."""
        E_grid = np.logspace(2, 3, 10)  # 100 – 1000 MeV, 10 points
        flux = gcr_total_flux(E_grid, 550.0)
        # Keep only protons
        proton_flux = {'H': flux['H']}
        result = dose_rate_from_flux(proton_flux, E_grid)
        assert result['dose_rate_mGy_day'] > 0, (
            f"Proton-only dose rate should be > 0, got {result['dose_rate_mGy_day']}")

    def test_dose_rate_from_flux_returns_expected_keys(self):
        """dose_rate_from_flux must return the required keys."""
        E_grid = np.logspace(2, 3, 10)
        flux = gcr_total_flux(E_grid, 550.0)
        result = dose_rate_from_flux({'H': flux['H']}, E_grid)
        for key in ('absorbed_dose_rate', 'dose_rate_total', 'dose_rate_mGy_day', 'LET_spectrum'):
            assert key in result, f"Missing key '{key}' in dose_rate_from_flux result"

    def test_dose_equivalent_rate_proton_only_Q1(self):
        """For proton-only flux (low LET → Q≈1), Q_effective should be within 5% of 1.0."""
        E_grid = np.logspace(2, 3, 10)  # 100 – 1000 MeV
        flux = gcr_total_flux(E_grid, 550.0)
        proton_flux = {'H': flux['H']}
        result = dose_equivalent_rate(proton_flux, E_grid)
        Q_eff = result['Q_effective']
        assert abs(Q_eff - 1.0) < 0.05, (
            f"Proton-only Q_effective = {Q_eff:.3f}, expected close to 1.0 (within 5%)")

    def test_dose_equivalent_rate_proton_only_keys(self):
        """dose_equivalent_rate must return the required keys."""
        E_grid = np.logspace(2, 3, 10)
        flux = gcr_total_flux(E_grid, 550.0)
        result = dose_equivalent_rate({'H': flux['H']}, E_grid)
        for key in ('H_rate_Sv_s', 'H_rate_mSv_day', 'Q_effective'):
            assert key in result, f"Missing key '{key}' in dose_equivalent_rate result"

    def test_dose_equivalent_H_geq_D(self):
        """For mixed GCR flux, H_rate_Sv_s >= dose_rate_total (since Q >= 1 everywhere)."""
        E_grid = np.logspace(1, 5, 50)
        flux = gcr_total_flux(E_grid, 550.0)
        dose_result = dose_rate_from_flux(flux, E_grid)
        H_result = dose_equivalent_rate(flux, E_grid)
        assert H_result['H_rate_Sv_s'] >= dose_result['dose_rate_total'], (
            f"H_rate ({H_result['H_rate_Sv_s']:.3e} Sv/s) should be >= "
            f"D_rate ({dose_result['dose_rate_total']:.3e} Gy/s) since Q >= 1")

    def test_neutron_h10_increases_with_energy_low(self):
        """neutron_h10 should be monotonically increasing from 10 → 100 MeV."""
        E = np.array([10.0, 20.0, 50.0, 100.0])
        h10 = neutron_h10(E)
        for i in range(len(E) - 1):
            assert h10[i] < h10[i + 1], (
                f"neutron_h10 should increase: h10({E[i]}) = {h10[i]:.1f} >= "
                f"h10({E[i+1]}) = {h10[i+1]:.1f}")

    def _make_trajectory(self, n_days=5):
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n_days)]
        return pd.DataFrame({
            'date': dates,
            'r_AU': [1.5] * n_days,
            'phi_MV': [550.0] * n_days,
            'v_AU_day': [0.01] * n_days,
        })

    def test_integrate_mission_dose_returns_positive(self):
        """5-day trajectory at 5 g/cm² Al: D_total_mGy > 0 and H_total_mSv > 0."""
        traj = self._make_trajectory(5)
        E_grid = np.logspace(1, 5, 30)
        result = integrate_mission_dose(
            traj,
            shielding_x_gcm2=5.0,
            shielding_material='aluminum',
            E_grid_MeV=E_grid,
        )
        assert result['D_total_mGy'] > 0, f"D_total_mGy should be > 0, got {result['D_total_mGy']}"
        assert result['H_total_mSv'] > 0, f"H_total_mSv should be > 0, got {result['H_total_mSv']}"

    def test_integrate_mission_dose_keys(self):
        """integrate_mission_dose must return all required keys."""
        traj = self._make_trajectory(5)
        E_grid = np.logspace(1, 5, 30)
        result = integrate_mission_dose(
            traj,
            shielding_x_gcm2=5.0,
            shielding_material='aluminum',
            E_grid_MeV=E_grid,
        )
        required_keys = (
            'D_total_mGy', 'H_total_mSv', 'E_effective_mSv',
            'D_rate_daily', 'H_rate_daily', 'mission_duration_days',
        )
        for key in required_keys:
            assert key in result, f"Missing key '{key}' in integrate_mission_dose result"

    def test_integrate_mission_dose_more_shielding_less_dose(self):
        """5 g/cm² Al should give higher D_total_mGy than 30 g/cm² Al."""
        traj = self._make_trajectory(5)
        E_grid = np.logspace(1, 5, 30)
        result_thin = integrate_mission_dose(
            traj,
            shielding_x_gcm2=5.0,
            shielding_material='aluminum',
            E_grid_MeV=E_grid,
        )
        result_thick = integrate_mission_dose(
            traj,
            shielding_x_gcm2=30.0,
            shielding_material='aluminum',
            E_grid_MeV=E_grid,
        )
        assert result_thin['D_total_mGy'] > result_thick['D_total_mGy'], (
            f"Thinner shielding (5 g/cm²) should give more dose: "
            f"D_thin={result_thin['D_total_mGy']:.3f}, D_thick={result_thick['D_total_mGy']:.3f} mGy")

    def test_msl_species_mix_is_reasonably_hzetrn_like(self):
        """
        Composition-constrained transit field should be broadly consistent with
        ACE/CRIS and NSRL GCR reference data (NSRL: H~73%, He~19%, HZE~8%).

        Windows are wider than NSRL to account for: CSDA transport approximations,
        trajectory phi variation, and the ~20% uncertainty in ACE/CRIS relative
        abundances propagated through the single-scale calibration.
        """
        from gcrisk.trajectory import generate_trajectory

        E_grid = np.logspace(1, 5, 200)
        traj = generate_trajectory('2011-11-26')
        phi_mid = float(traj['phi_MV'].iloc[len(traj) // 2])
        flux_mid = gcr_total_flux(E_grid, phi_mid)
        tfactors = _precompute_transport_factors(16.0, 'aluminum', E_grid)
        flux_transported = _apply_transport_factors(flux_mid, E_grid, tfactors)
        dose_species = dose_rate_from_flux(flux_transported, E_grid)
        d_by_species = dose_species['absorbed_dose_rate']
        d_total = dose_species['dose_rate_total']

        frac_h = d_by_species.get('H', 0.0) / d_total
        frac_he = d_by_species.get('He', 0.0) / d_total
        frac_hze = sum(
            v for k, v in d_by_species.items() if k not in ('H', 'He', 'neutron')
        ) / d_total

        # Composition-constrained bounds: H dominant (NSRL 73%), HZE subdominant (NSRL 8%)
        assert 0.55 < frac_h < 0.85, f"H fraction {frac_h:.1%} outside [55%, 85%]"
        assert 0.05 < frac_he < 0.25, f"He fraction {frac_he:.1%} outside [5%, 25%]"
        assert 0.05 < frac_hze < 0.20, f"HZE fraction {frac_hze:.1%} outside [5%, 20%]"
