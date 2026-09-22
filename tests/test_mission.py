"""Tests for gcrisk/mission.py"""

import numpy as np
import pytest
from gcrisk.mission import mars_surface_dose, run_full_mission


# Coarse energy grid to keep transit integration fast during testing
_E_GRID_COARSE = np.logspace(1, 5, 20)


class TestMissionModule:

    def test_mars_surface_dose_returns_positive(self):
        """mars_surface_dose for 10 days: D_total_mGy > 0 and H_total_mSv > 0."""
        result = mars_surface_dose(n_days=10, start_date='2025-06-01')
        assert result['D_total_mGy'] > 0, (
            f"D_total_mGy should be > 0, got {result['D_total_mGy']}")
        assert result['H_total_mSv'] > 0, (
            f"H_total_mSv should be > 0, got {result['H_total_mSv']}")

    def test_mars_surface_dose_keys(self):
        """mars_surface_dose must return the required keys."""
        result = mars_surface_dose(n_days=10, start_date='2025-06-01')
        required_keys = ('D_total_mGy', 'H_total_mSv', 'Q_effective', 'mission_duration_days')
        for key in required_keys:
            assert key in result, f"Missing key '{key}' in mars_surface_dose result"

    def test_mars_surface_dose_hassler_baseline(self):
        """365-day open surface: D/365 should be in range 0.15–0.28 mGy/day (Hassler 2014 baseline)."""
        result = mars_surface_dose(
            n_days=365,
            start_date='2013-11-01',  # near RAD measurement epoch (solar min)
            spacecraft_shielding_x_gcm2=0.0,
        )
        daily_rate = result['D_total_mGy'] / 365.0
        assert 0.15 < daily_rate < 0.28, (
            f"Daily dose rate {daily_rate:.3f} mGy/day outside Hassler baseline range 0.15–0.28")

    def test_mars_surface_more_shielding_less_dose(self):
        """More habitat shielding → lower H_mSv."""
        result_0 = mars_surface_dose(
            n_days=30, start_date='2025-06-01',
            spacecraft_shielding_x_gcm2=0.0,
        )
        result_20 = mars_surface_dose(
            n_days=30, start_date='2025-06-01',
            spacecraft_shielding_x_gcm2=20.0,
        )
        assert result_0['H_total_mSv'] > result_20['H_total_mSv'], (
            f"No shielding H={result_0['H_total_mSv']:.2f} mSv should exceed "
            f"20 g/cm² shielding H={result_20['H_total_mSv']:.2f} mSv")

    def test_run_full_mission_returns_dict(self):
        """run_full_mission returns a dict with phases and total sub-dicts."""
        result = run_full_mission(
            launch_date='2020-01-01',
            surface_days=30,
            E_grid_MeV=_E_GRID_COARSE,
        )
        assert isinstance(result, dict), "run_full_mission must return a dict"
        assert 'total' in result, "Result must have 'total' key"
        assert 'phases' in result, "Result must have 'phases' key"
        total = result['total']
        phases = result['phases']
        assert 'H_total_mSv' in total, "total must have 'H_total_mSv'"
        assert 'D_total_mGy' in total, "total must have 'D_total_mGy'"
        assert 'transit_out' in phases, "phases must have 'transit_out'"
        assert 'surface' in phases, "phases must have 'surface'"
        assert 'transit_back' in phases, "phases must have 'transit_back'"

    def test_run_full_mission_total_greater_than_phase(self):
        """Total H_mSv must be greater than any single phase H_mSv."""
        result = run_full_mission(
            launch_date='2020-01-01',
            surface_days=30,
            E_grid_MeV=_E_GRID_COARSE,
        )
        H_total = result['total']['H_total_mSv']
        H_out = result['phases']['transit_out']['H_total_mSv']
        H_surface = result['phases']['surface']['H_total_mSv']
        H_back = result['phases']['transit_back']['H_total_mSv']
        assert H_total > H_out, (
            f"Total H ({H_total:.1f}) should exceed transit_out H ({H_out:.1f})")
        assert H_total > H_surface, (
            f"Total H ({H_total:.1f}) should exceed surface H ({H_surface:.1f})")
        assert H_total > H_back, (
            f"Total H ({H_total:.1f}) should exceed transit_back H ({H_back:.1f})")

    def test_run_full_mission_surface_scales_with_duration(self):
        """Surface H should scale proportionally with surface days (longer stay → more dose)."""
        result_100 = run_full_mission(
            launch_date='2020-01-01',
            surface_days=100,
            E_grid_MeV=_E_GRID_COARSE,
        )
        result_500 = run_full_mission(
            launch_date='2020-01-01',
            surface_days=500,
            E_grid_MeV=_E_GRID_COARSE,
        )
        H_surface_100 = result_100['phases']['surface']['H_total_mSv']
        H_surface_500 = result_500['phases']['surface']['H_total_mSv']
        # 500-day surface must give more dose than 100-day surface
        assert H_surface_500 > H_surface_100, (
            f"500-day surface H ({H_surface_500:.1f} mSv) should exceed "
            f"100-day surface H ({H_surface_100:.1f} mSv)")
        # Scaling should be roughly linear (within factor 2): expect ~5× more
        ratio = H_surface_500 / H_surface_100 if H_surface_100 > 0 else 0
        assert 3.0 < ratio < 8.0, (
            f"H_surface ratio (500d/100d) = {ratio:.2f}, expected ~5× (linear with time)")
