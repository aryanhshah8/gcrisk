"""Tests for organ-specific dose routing."""

import numpy as np

from gcrisk.organ_dose import (
    ORGAN_DEPTHS_GCMS2,
    ORGAN_ICRP60_WEIGHTS,
    integrate_organ_dose,
)
from gcrisk.reid import reid_from_organ_doses
from gcrisk.trajectory import generate_trajectory


def test_organ_depth_ordering_constant():
    """Sanity check: shallow thyroid should sit above deep bladder."""
    assert ORGAN_DEPTHS_GCMS2['thyroid'] < ORGAN_DEPTHS_GCMS2['bladder']


def test_shallow_organs_receive_more_dose():
    """Shallow organs should receive more dose equivalent than deeper ones."""
    traj = generate_trajectory('2011-11-26')
    result = integrate_organ_dose(
        traj.iloc[:10],
        shielding_x=16.0,
        material='aluminum',
        E_grid=np.logspace(1, 5, 40),
    )
    assert result['organ_H_mSv']['thyroid'] > result['organ_H_mSv']['bladder']


def test_weighted_organ_sum_tracks_effective_dose():
    """ICRP-weighted organ sum should track the reported effective-dose proxy."""
    traj = generate_trajectory('2011-11-26')
    result = integrate_organ_dose(
        traj.iloc[:10],
        shielding_x=16.0,
        material='aluminum',
        E_grid=np.logspace(1, 5, 40),
    )
    weighted = sum(
        ORGAN_ICRP60_WEIGHTS[organ] * result['organ_H_mSv'][organ]
        for organ in ORGAN_ICRP60_WEIGHTS
    )
    denom = max(result['E_effective_mSv'], 1e-12)
    assert abs(weighted - result['E_effective_mSv']) / denom < 0.20


def test_reid_from_organ_doses_returns_positive_summary():
    """Organ-routed REID should return non-negative totals and organ breakdown."""
    traj = generate_trajectory('2011-11-26')
    organ_result = integrate_organ_dose(
        traj.iloc[:10],
        shielding_x=16.0,
        material='aluminum',
        E_grid=np.logspace(1, 5, 30),
    )
    reid = reid_from_organ_doses(
        organ_result['organ_H_mSv'],
        age_at_exposure=35,
        sex='male',
        n_samples=250,
    )
    assert reid['REID_total'] >= 0.0
    assert set(reid['REID_by_organ']) == set(ORGAN_ICRP60_WEIGHTS)
