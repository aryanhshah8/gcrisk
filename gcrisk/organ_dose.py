"""
gcrisk/organ_dose.py — Organ-specific self-shielding and dose routing.

Bridges the transport physics to the organ-specific risk model by adding
representative tissue-equivalent depth inside the body to the external
spacecraft shielding. The resulting organ doses can be routed into the
organ-specific EAR terms in gcrisk.reid.
"""

from __future__ import annotations

import os

import numpy as np

from .dose import (
    _apply_transport_factors,
    _precompute_transport_factors,
    dose_equivalent_rate,
    dose_rate_from_flux,
    integrate_mission_dose,
)
from .spectrum import gcr_total_flux, load_usoskin_phi
from .utils import DEFAULT_E_GRID


ORGAN_DEPTHS_GCMS2: dict[str, float] = {
    'stomach': 8.0,
    'colon': 9.5,
    'lung': 5.0,
    'liver': 6.0,
    'bladder': 10.0,
    'thyroid': 2.0,
    'esophagus': 4.0,
    'breast': 1.5,
    'ovary': 12.0,
    'leukemia': 4.0,
    'other': 7.0,
}

# Mapping from organ_dose organ keys to ICRP-60 tissue weighting factors.
# The modeled organs sum to 0.95 because skin and bone surface are omitted.
ORGAN_ICRP60_WEIGHTS: dict[str, float] = {
    'stomach': 0.12,
    'colon': 0.12,
    'lung': 0.12,
    'liver': 0.04,
    'bladder': 0.04,
    'thyroid': 0.04,
    'esophagus': 0.05,
    'breast': 0.05,
    'ovary': 0.20,
    'leukemia': 0.12,
    'other': 0.05,
}


def organ_dose_rates(
    flux: dict,
    E_grid: np.ndarray,
    transport_cache: dict[float, dict],
    shielding_x: float,
    material: str,
) -> dict[str, dict[str, float]]:
    """
    Compute dose rates for each organ using organ-specific self-shielding depth.

    Parameters
    ----------
    flux : dict
        Primary modulated GCR flux from gcrisk.spectrum.gcr_total_flux().
    E_grid : ndarray
        Per-nucleon energy grid [MeV/n].
    transport_cache : dict
        Cache keyed by total shielding depth [g/cm^2] containing precomputed
        transport factors from dose._precompute_transport_factors().
    shielding_x : float
        External spacecraft shielding [g/cm^2].
    material : str
        Shielding material key.
    """
    E_grid = np.asarray(E_grid, dtype=float)
    results: dict[str, dict[str, float]] = {}

    for organ, depth in ORGAN_DEPTHS_GCMS2.items():
        x_total = shielding_x + depth
        if x_total not in transport_cache:
            transport_cache[x_total] = _precompute_transport_factors(
                x_total, material, E_grid
            )

        transported_flux = _apply_transport_factors(
            flux, E_grid, transport_cache[x_total]
        )
        dose_result = dose_rate_from_flux(transported_flux, E_grid)
        H_result = dose_equivalent_rate(transported_flux, E_grid)

        results[organ] = {
            'D_rate_Gy_s': dose_result['dose_rate_total'],
            'H_rate_Sv_s': H_result['H_rate_Sv_s'],
            'Q_eff': H_result['Q_effective'],
        }

    return results


def integrate_organ_dose(
    trajectory_df,
    shielding_x: float,
    material: str,
    phi_df=None,
    E_grid=None,
) -> dict[str, object]:
    """
    Integrate organ-specific dose across a trajectory.

    Returns organ-integrated absorbed dose, dose equivalent, and a tissue-weighted
    effective-dose estimate built from the routed organ doses.
    """
    if E_grid is None:
        E_grid = DEFAULT_E_GRID
    E_grid = np.asarray(E_grid, dtype=float)

    if phi_df is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'usoskin')
        phi_df = load_usoskin_phi(os.path.join(data_dir, 'phi_monthly.csv'))

    transport_cache = {
        shielding_x + depth: _precompute_transport_factors(
            shielding_x + depth, material, E_grid
        )
        for depth in ORGAN_DEPTHS_GCMS2.values()
    }

    dt_s = 86400.0
    organ_D_Gy = {organ: 0.0 for organ in ORGAN_DEPTHS_GCMS2}
    organ_H_Sv = {organ: 0.0 for organ in ORGAN_DEPTHS_GCMS2}

    for phi in trajectory_df['phi_MV']:
        flux = gcr_total_flux(E_grid, float(phi))
        organ_rates = organ_dose_rates(
            flux, E_grid, transport_cache, shielding_x, material
        )
        for organ, rates in organ_rates.items():
            organ_D_Gy[organ] += rates['D_rate_Gy_s'] * dt_s
            organ_H_Sv[organ] += rates['H_rate_Sv_s'] * dt_s

    organ_D_mGy = {organ: dose * 1000.0 for organ, dose in organ_D_Gy.items()}
    organ_H_mSv = {organ: dose * 1000.0 for organ, dose in organ_H_Sv.items()}
    organ_Q_eff = {
        organ: (
            organ_H_Sv[organ] / organ_D_Gy[organ]
            if organ_D_Gy[organ] > 0
            else 1.0
        )
        for organ in ORGAN_DEPTHS_GCMS2
    }
    E_effective_mSv = sum(
        ORGAN_ICRP60_WEIGHTS[organ] * organ_H_mSv[organ]
        for organ in ORGAN_DEPTHS_GCMS2
    )

    whole_body = integrate_mission_dose(
        trajectory_df,
        shielding_x_gcm2=shielding_x,
        shielding_material=material,
        phi_df=phi_df,
        E_grid_MeV=E_grid,
    )

    return {
        'organ_D_mGy': organ_D_mGy,
        'organ_H_mSv': organ_H_mSv,
        'organ_Q_eff': organ_Q_eff,
        'H_total_mSv': whole_body['H_total_mSv'],
        'E_effective_mSv': E_effective_mSv,
        'weight_sum': sum(ORGAN_ICRP60_WEIGHTS.values()),
    }
