"""
gcrisk/rbe.py — LET-based RBE modeling.

Dose-averaged LET summaries and two proton RBE parameterizations
(Wedenberg and McNamara) for proton-therapy-style LET/RBE exploration.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.integrate import trapezoid

from .dose import _apply_transport_factors, _precompute_transport_factors, let_from_energy
from .spectrum import gcr_total_flux, load_usoskin_phi
from .utils import DEFAULT_E_GRID, ION_SPECIES, MATERIALS


_WEDENBERG_Q = 0.434
_MCNAMARA_RBEMAX_OFFSET = 0.99064
_MCNAMARA_RBEMAX_SLOPE = 0.35605
_MCNAMARA_RBEMIN_OFFSET = 1.1012
_MCNAMARA_RBEMIN_SLOPE = -0.0038703


def _charged_species(species: list[str] | None = None) -> list[str]:
    """Return valid charged-particle species keys present in the registry."""
    if species is None:
        return list(ION_SPECIES.keys())
    return [sp for sp in species if sp in ION_SPECIES]


def let_moments_from_flux(
    flux: dict,
    E_grid_MeV: np.ndarray,
    material: str = 'tissue',
    species: list[str] | None = None,
) -> dict[str, object]:
    """
    Compute dose-weighted LET moments for a transported flux field.

    Returns LETd = integral(LET * dD) / integral(dD).
    """
    E = np.asarray(E_grid_MeV, dtype=float)
    density = MATERIALS[material]['density']
    selected = _charged_species(species)

    numerator = 0.0
    denominator = 0.0
    dose_by_species: dict[str, float] = {}

    for sp_key in selected:
        if sp_key not in flux:
            continue

        Z = ION_SPECIES[sp_key]['Z']
        flux_arr = np.asarray(flux[sp_key], dtype=float)
        LET = let_from_energy(E, Z, material=material)
        S = LET / (density * 0.1)

        d_dose = trapezoid(flux_arr * S, E)
        d_let_dose = trapezoid(flux_arr * S * LET, E)

        dose_by_species[sp_key] = float(d_dose)
        denominator += float(d_dose)
        numerator += float(d_let_dose)

    letd = numerator / denominator if denominator > 0 else 0.0
    return {
        'LETd_keV_um': float(letd),
        'numerator': float(numerator),
        'denominator': float(denominator),
        'dose_by_species': dose_by_species,
    }


def dose_averaged_let(
    flux: dict,
    E_grid_MeV: np.ndarray,
    material: str = 'tissue',
    species: list[str] | None = None,
) -> float:
    """Dose-averaged LETd in keV/um for a transported charged-particle field."""
    return float(
        let_moments_from_flux(
            flux,
            E_grid_MeV=E_grid_MeV,
            material=material,
            species=species,
        )['LETd_keV_um']
    )


def integrate_mission_let(
    trajectory_df,
    shielding_x_gcm2: float,
    shielding_material: str,
    phi_df=None,
    E_grid_MeV: np.ndarray | None = None,
    species: list[str] | None = None,
    material: str = 'tissue',
) -> dict[str, object]:
    """Integrate mission-average LETd across a trajectory."""
    if E_grid_MeV is None:
        E_grid_MeV = DEFAULT_E_GRID
    E_grid = np.asarray(E_grid_MeV, dtype=float)

    if phi_df is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'usoskin')
        phi_df = load_usoskin_phi(os.path.join(data_dir, 'phi_monthly.csv'))

    if shielding_x_gcm2 > 0:
        transport_factors = _precompute_transport_factors(
            shielding_x_gcm2, shielding_material, E_grid
        )
    else:
        transport_factors = {}

    numerator = 0.0
    denominator = 0.0
    for phi in trajectory_df['phi_MV']:
        flux = gcr_total_flux(E_grid, float(phi), species=species)
        if shielding_x_gcm2 > 0:
            flux = _apply_transport_factors(flux, E_grid, transport_factors)
        moments = let_moments_from_flux(
            flux,
            E_grid_MeV=E_grid,
            material=material,
            species=species,
        )
        numerator += float(moments['numerator'])
        denominator += float(moments['denominator'])

    return {
        'LETd_keV_um': float(numerator / denominator) if denominator > 0 else 0.0,
        'species': _charged_species(species),
        'material': material,
        'shielding_material': shielding_material,
        'shielding_x_gcm2': float(shielding_x_gcm2),
    }


def wedenberg_rbe(dose_Gy: float, letd_keV_um: float, alpha_beta_x_Gy: float) -> float:
    """Wedenberg proton RBE model."""
    dose = float(dose_Gy)
    letd = float(letd_keV_um)
    alpha_beta = float(alpha_beta_x_Gy)
    if dose <= 0:
        return 0.0
    alpha_ratio = 1.0 + _WEDENBERG_Q * letd / alpha_beta
    return float(
        np.sqrt(0.25 * alpha_beta * alpha_beta + alpha_ratio * alpha_beta * dose + dose * dose)
        / dose
        - alpha_beta / (2.0 * dose)
    )


def mcnamara_rbe(dose_Gy: float, letd_keV_um: float, alpha_beta_x_Gy: float) -> float:
    """McNamara proton RBE model."""
    dose = float(dose_Gy)
    letd = float(letd_keV_um)
    alpha_beta = float(alpha_beta_x_Gy)
    if dose <= 0:
        return 0.0

    rbe_max = _MCNAMARA_RBEMAX_OFFSET + (_MCNAMARA_RBEMAX_SLOPE / alpha_beta) * letd
    rbe_min = _MCNAMARA_RBEMIN_OFFSET + _MCNAMARA_RBEMIN_SLOPE * np.sqrt(alpha_beta) * letd
    return float(
        (
            np.sqrt(
                alpha_beta * alpha_beta
                + 4.0 * alpha_beta * rbe_max * dose
                + 4.0 * (rbe_min ** 2) * dose * dose
            )
            - alpha_beta
        )
        / (2.0 * dose)
    )


def rbe_weighted_dose(
    dose_Gy: float,
    letd_keV_um: float,
    alpha_beta_x_Gy: float,
    model: str = 'wedenberg',
) -> dict[str, float]:
    """Return RBE and RBE-weighted dose for a selected proton RBE model."""
    model_key = model.lower()
    if model_key == 'wedenberg':
        rbe = wedenberg_rbe(dose_Gy, letd_keV_um, alpha_beta_x_Gy)
    elif model_key == 'mcnamara':
        rbe = mcnamara_rbe(dose_Gy, letd_keV_um, alpha_beta_x_Gy)
    else:
        raise ValueError(f"Unknown RBE model '{model}'.")

    return {
        'model': model_key,
        'dose_Gy': float(dose_Gy),
        'LETd_keV_um': float(letd_keV_um),
        'alpha_beta_x_Gy': float(alpha_beta_x_Gy),
        'RBE': float(rbe),
        'dose_RBE_Gy': float(rbe * dose_Gy),
    }


def let_spectrum_by_species(
    flux: dict,
    E_grid_MeV: np.ndarray,
    material: str = 'tissue',
    species: list[str] | None = None,
) -> dict[str, object]:
    """
    Per-species LET distributions and dose-averaged LETd values.

    Parameters
    ----------
    flux       : transported flux dict
    E_grid_MeV : energy grid [MeV/n]
    material   : target material
    species    : list of species to include; defaults to all charged species

    Returns
    -------
    dict with species_let, species_letd, species_dose_fraction, field_letd,
    E_grid_MeV, material.
    """
    E = np.asarray(E_grid_MeV, dtype=float)
    density = MATERIALS[material]['density']
    selected = _charged_species(species)

    species_let: dict[str, np.ndarray] = {}
    species_letd: dict[str, float] = {}
    dose_by_species: dict[str, float] = {}

    total_dose = 0.0

    for sp_key in selected:
        if sp_key not in flux:
            continue

        Z = ION_SPECIES[sp_key]['Z']
        flux_arr = np.asarray(flux[sp_key], dtype=float)
        LET = let_from_energy(E, Z, material=material)
        S = LET / (density * 0.1)

        d_dose = float(trapezoid(flux_arr * S, E))
        d_let_dose = float(trapezoid(flux_arr * S * LET, E))
        letd_sp = d_let_dose / d_dose if d_dose > 0 else 0.0

        species_let[sp_key] = LET
        species_letd[sp_key] = letd_sp
        dose_by_species[sp_key] = d_dose
        total_dose += d_dose

    # Dose fractions
    species_dose_fraction = {
        sp: (d / total_dose if total_dose > 0 else 0.0)
        for sp, d in dose_by_species.items()
    }

    field_letd = sum(
        species_letd[sp] * species_dose_fraction[sp]
        for sp in species_letd
    )

    return {
        'species_let': species_let,
        'species_letd': species_letd,
        'species_dose_fraction': species_dose_fraction,
        'field_letd': field_letd,
        'E_grid_MeV': E,
        'material': material,
    }
