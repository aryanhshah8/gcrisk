"""Shared helpers for Paper 2 driver scripts.

Nothing here is imported by Paper 1. Paths and baseline case match Paper 1's
canonical setup (35-yr-old male, 259-day cruise, 16 g/cm² Al, frozen transit
phi series).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from gcrisk.dose import integrate_mission_dose
from gcrisk.spectrum import gcr_total_flux, load_usoskin_phi
from gcrisk.trajectory import generate_trajectory
from gcrisk.utils import DEFAULT_E_GRID


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
FIGURES_DIR = os.path.join(REPO_ROOT, 'figures', 'paper2')
DATA_DIR = os.path.join(REPO_ROOT, 'data')
USOSKIN_DIR = os.path.join(DATA_DIR, 'usoskin')

# Paper 1 canonical case.
BASELINE_CASE: dict = {
    'launch_date': '2011-11-26',
    'shielding_x_gcm2': 16.0,
    'material': 'aluminum',
    'age': 35,
    'sex': 'male',
}


def ensure_dirs() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)


def load_baseline() -> dict:
    """Load the trajectory, phi series and canonical shielded flux mean.

    Returns a dict with ``trajectory_df``, ``phi_df``, ``shielded_flux_mean``
    (dict keyed by species, evaluated at mission-averaged phi), and the
    case parameters.
    """
    phi_df = load_usoskin_phi(
        os.path.join(USOSKIN_DIR, 'phi_transit_frozen.csv')
    )
    traj_df = generate_trajectory(BASELINE_CASE['launch_date'], phi_df=phi_df)
    phi_mean = float(np.mean(traj_df['phi_MV']))
    flux_unshielded_mean = gcr_total_flux(DEFAULT_E_GRID, phi_mean)

    return {
        'trajectory_df': traj_df,
        'phi_df': phi_df,
        'phi_mean_MV': phi_mean,
        'flux_unshielded_mean': flux_unshielded_mean,
        'E_grid': DEFAULT_E_GRID,
        **BASELINE_CASE,
    }


def baseline_shielded_flux_mean(baseline: dict) -> dict:
    """Return a mission-mean shielded flux using Paper 1's transport layer."""
    from gcrisk.dose import _apply_transport_factors, _precompute_transport_factors

    factors = _precompute_transport_factors(
        baseline['shielding_x_gcm2'],
        baseline['material'],
        baseline['E_grid'],
    )
    return _apply_transport_factors(
        baseline['flux_unshielded_mean'], baseline['E_grid'], factors,
    )


def baseline_total_dose(baseline: dict) -> dict:
    """Run Paper 1's ``integrate_mission_dose`` once for the baseline case."""
    return integrate_mission_dose(
        baseline['trajectory_df'],
        shielding_x_gcm2=baseline['shielding_x_gcm2'],
        shielding_material=baseline['material'],
        phi_df=baseline['phi_df'],
        E_grid_MeV=baseline['E_grid'],
    )
