"""
gcr/spectrum.py — GCR flux generation.

Computes modulated GCR flux for all ion species using the force-field approximation.
Species normalizations are constrained by ACE/CRIS and PAMELA composition measurements
at the calibration conditions (200 MeV/n, phi = 481 MV), with a single global scale
fitted to the MSL/RAD absorbed dose measurement of 1.84 mGy/day at 16 g/cm² Al.
"""

import os
import numpy as np
import pandas as pd

from .utils import CONSTANTS, ION_SPECIES


# ---------------------------------------------------------------------------
# ACE/CRIS + PAMELA composition reference
# Measured particle flux ratios relative to H = 1.0 at ~200 MeV/n,
# solar modulation phi ~ 481 MV (2011–2012 MSL cruise epoch).
# Sources:
#   Protons/He: Adriani et al. 2014, ApJL 791, L14 (PAMELA 2011–2012)
#   C, O, Si, Fe: George et al. 2009, ApJ 698, 1666 (ACE/CRIS 2003–2008,
#     interpolated to phi = 481 MV using the force-field ratio)
_ACE_CRIS_COMPOSITION = {
    'H':  1.0000,
    'He': 0.0660,   # PAMELA; Adriani et al. 2014
    'C':  1.80e-3,  # ACE/CRIS; George et al. 2009
    'O':  2.20e-3,  # ACE/CRIS; George et al. 2009  (C/O = 0.82, well-established)
    'Si': 1.80e-4,  # ACE/CRIS; George et al. 2009
    'Fe': 1.40e-4,  # ACE/CRIS; George et al. 2009
}

# Reference conditions for the composition constraint
_CALIB_E_REF_MEV  = 200.0   # kinetic energy per nucleon [MeV/n]
_CALIB_PHI_REF_MV = 481.0   # MSL cruise mean modulation potential [MV]

# Global dose-matching scale: fitted so the composition-constrained model reproduces
# the MSL/RAD absorbed dose rate of 1.84 mGy/day at 16 g/cm² Al, phi = 481 MV.
# Recompute with: python scripts/calibrate_global_scale.py
# This single free parameter replaces the previous 6-factor unconstrained calibration.
_GLOBAL_DOSE_SCALE = 0.894438  # fitted: D = 1.84 mGy/day at 16 g/cm² Al, phi = 481 MV


# ---------------------------------------------------------------------------
# LIS functions

def lis_proton(E_MeV_per_n: np.ndarray) -> np.ndarray:
    """
    Local Interstellar Spectrum for protons.

    Parameters
    ----------
    E_MeV_per_n : array-like
        Kinetic energy in MeV/nucleon.

    Returns
    -------
    flux : np.ndarray
        Differential flux in cm^-2 s^-1 MeV^-1 sr^-1
    """
    E = np.asarray(E_MeV_per_n, dtype=float)
    mp_c2 = CONSTANTS['mp_c2']

    E_GeV = E / 1000.0
    gamma = (E + mp_c2) / mp_c2
    b2 = 1.0 - 1.0 / gamma**2

    C = 0.75 / (4.0 * np.pi)
    flux = C * (E_GeV + 0.67)**(-3.93) / b2
    return flux


def lis_helium(E_MeV_per_n: np.ndarray) -> np.ndarray:
    """
    Helium-4 Local Interstellar Spectrum.

    Parameters
    ----------
    E_MeV_per_n : array-like
        Kinetic energy per nucleon in MeV/n.

    Returns
    -------
    flux : np.ndarray
        Differential directional flux in cm^-2 s^-1 MeV^-1 sr^-1
    """
    E = np.atleast_1d(np.asarray(E_MeV_per_n, dtype=float))
    mp_c2 = CONSTANTS['mp_c2']

    alpha_He = 3.84
    E0_He_GeV = 0.50

    E_GeV = E / 1000.0
    E_total_GeV = E_GeV + mp_c2 / 1000.0
    beta2 = 1.0 - (mp_c2 / 1000.0 / E_total_GeV) ** 2
    beta2 = np.maximum(beta2, 0.0)

    shape = 1.0 / (beta2 * (E_GeV + E0_He_GeV) ** alpha_He)

    # Normalize: j_He(1 GeV/n) = 0.096 * j_H(1 GeV/n)
    E_ref_MeV = np.array([1000.0])
    j_H_ref = float(lis_proton(E_ref_MeV)[0])

    E_ref_GeV = 1.0
    E_total_ref_GeV = E_ref_GeV + mp_c2 / 1000.0
    beta2_ref = 1.0 - (mp_c2 / 1000.0 / E_total_ref_GeV) ** 2
    shape_ref = 1.0 / (beta2_ref * (E_ref_GeV + E0_He_GeV) ** alpha_He)

    C_He = 0.096 * j_H_ref / shape_ref

    return C_He * shape


# HZE LIS parameters (C, O, Si, Fe)
_HZE_LIS_PARAMS = {
    'C':  {'alpha': 2.77, 'offset_GeV': 0.48},
    'O':  {'alpha': 2.80, 'offset_GeV': 0.48},
    'Si': {'alpha': 2.78, 'offset_GeV': 0.43},
    'Fe': {'alpha': 2.65, 'offset_GeV': 0.40},
}


def _hze_normalization(species_key: str) -> float:
    """Compute LIS normalization constant C_s for a heavy ion species."""
    params = _HZE_LIS_PARAMS[species_key]
    abundance = ION_SPECIES[species_key]['abundance']
    mp_c2 = CONSTANTS['mp_c2']

    E_ref_MeV = 1000.0
    E_ref_GeV = 1.0

    gamma_ref = (E_ref_MeV + mp_c2) / mp_c2
    beta2_ref = 1.0 - 1.0 / gamma_ref**2

    j_H_ref = lis_proton(np.array([E_ref_MeV]))[0]
    j_target = abundance * j_H_ref

    C_s = j_target * beta2_ref * (E_ref_GeV + params['offset_GeV'])**params['alpha']
    return C_s


def lis_heavy(E_MeV_per_n: np.ndarray, Z: int, A: int) -> np.ndarray:
    """LIS for heavy ions (C, O, Si, Fe) using per-species parameterizations."""
    species_key = None
    for key, sp in ION_SPECIES.items():
        if sp['Z'] == Z and sp['A'] == A:
            species_key = key
            break
    if species_key is None:
        raise ValueError(f"Ion with Z={Z}, A={A} not found in ION_SPECIES")

    if species_key not in _HZE_LIS_PARAMS:
        # Fallback: abundance-scaled proton LIS
        abundance = ION_SPECIES[species_key]['abundance']
        return abundance * lis_proton(E_MeV_per_n)

    params = _HZE_LIS_PARAMS[species_key]
    C_s = _hze_normalization(species_key)

    E = np.asarray(E_MeV_per_n, dtype=float)
    mp_c2 = CONSTANTS['mp_c2']

    E_GeV = E / 1000.0
    gamma = (E + mp_c2) / mp_c2
    beta2 = 1.0 - 1.0 / gamma**2

    flux = C_s * (E_GeV + params['offset_GeV'])**(-params['alpha']) / beta2
    return np.maximum(flux, 0.0)


def _get_lis_func(Z: int, A: int):
    """Return the appropriate LIS function for an ion species."""
    if Z == 1 and A == 1:
        return lis_proton
    elif Z == 2 and A == 4:
        return lis_helium
    else:
        return lambda E: lis_heavy(E, Z, A)


def force_field_modulation(
    E_MeV_per_n: np.ndarray,
    phi_MV: float,
    lis_func: callable,
    Z: int = 1,
    A: int = 1,
) -> np.ndarray:
    """
    Force-field solar modulation.

    Parameters
    ----------
    phi_MV : float
        Modulation potential in megavolts. Solar min ~400 MV, solar max ~1100 MV.
    """
    E = np.asarray(E_MeV_per_n, dtype=float)
    mp_c2 = CONSTANTS['mp_c2']

    phi_shift = (Z / A) * phi_MV
    E_shifted = E + phi_shift

    numerator = E**2 + 2.0 * E * mp_c2
    denominator = E_shifted**2 + 2.0 * E_shifted * mp_c2

    j_lis = lis_func(E_shifted)
    j_mod = j_lis * numerator / denominator

    return np.maximum(j_mod, 0.0)


# ---------------------------------------------------------------------------
# Composition-constrained calibration

def _build_species_calibration() -> dict:
    """
    Build per-species calibration factors constrained by ACE/CRIS measurements.

    For each species s, the factor corrects the LIS model's predicted modulated
    flux ratio (species/H) to match the ACE/CRIS measured ratio at the
    reference conditions (200 MeV/n, phi = 481 MV).  A single global scale
    _GLOBAL_DOSE_SCALE then shifts all species together to match the MSL/RAD
    absorbed dose of 1.84 mGy/day.

    This replaces the previous six-free-parameter calibration (fit to total
    dose only) with a one-parameter calibration that preserves the physical
    GCR inter-species composition.
    """
    E_ref = np.array([_CALIB_E_REF_MEV])
    phi = _CALIB_PHI_REF_MV

    # Modulated LIS flux at reference conditions
    lis_flux = {}
    for sp_key, sp in ION_SPECIES.items():
        Z, A = sp['Z'], sp['A']
        lis_func = _get_lis_func(Z, A)
        j = force_field_modulation(E_ref, phi, lis_func, Z, A)
        lis_flux[sp_key] = float(j[0])

    j_H_lis = lis_flux['H']
    factors = {}
    for sp_key in ION_SPECIES:
        ace_ratio = _ACE_CRIS_COMPOSITION.get(sp_key, None)
        if ace_ratio is None or j_H_lis <= 0:
            factors[sp_key] = _GLOBAL_DOSE_SCALE
            continue
        lis_ratio = lis_flux[sp_key] / j_H_lis
        if lis_ratio > 0:
            factors[sp_key] = (ace_ratio / lis_ratio) * _GLOBAL_DOSE_SCALE
        else:
            factors[sp_key] = _GLOBAL_DOSE_SCALE

    return factors


# Calibration factors: ACE/CRIS-constrained composition × global dose scale.
# The relative inter-species ratios are fixed by measurements; the overall
# magnitude is the single free parameter matched to MSL/RAD.
_SPECIES_FLUX_CALIBRATION = _build_species_calibration()


# ---------------------------------------------------------------------------
# Flux generation

def gcr_total_flux(
    E_MeV_per_n: np.ndarray,
    phi_MV: float,
    species: list = None,
    lis_norm_scale: float = 1.0,
    hze_norm_scale: float = 1.0,
) -> dict:
    """
    Compute modulated GCR flux for all (or specified) ion species.

    Parameters
    ----------
    E_MeV_per_n : energy grid [MeV/nucleon]
    phi_MV : solar modulation potential [MV]
    species : list of species keys to include; defaults to all
    lis_norm_scale : multiplicative scale on all LIS fluxes (uncertainty hook)
    hze_norm_scale : additional multiplicative scale on Z > 2 species only
                     (captures HZE composition uncertainty; uncertainty hook)

    Returns
    -------
    dict with keys = ion species symbols, values = flux arrays in cm^-2 s^-1 MeV^-1 sr^-1.
    Also includes key 'total' = sum over all species.
    """
    if species is None:
        species = list(ION_SPECIES.keys())

    result = {}
    total = np.zeros_like(np.asarray(E_MeV_per_n, dtype=float))

    for sp_key in species:
        sp = ION_SPECIES[sp_key]
        Z, A = sp['Z'], sp['A']
        lis_func = _get_lis_func(Z, A)
        flux = force_field_modulation(E_MeV_per_n, phi_MV, lis_func, Z, A)
        hze_factor = hze_norm_scale if Z > 2 else 1.0
        flux = flux * _SPECIES_FLUX_CALIBRATION.get(sp_key, 1.0) * lis_norm_scale * hze_factor
        result[sp_key] = flux
        total += flux

    result['total'] = total
    return result


def load_usoskin_phi(filepath: str) -> pd.DataFrame:
    """
    Load the Usoskin modulation potential database.

    Expected CSV columns: year, month, phi_MV.
    Returns DataFrame indexed by datetime with phi_MV column.
    Falls back to a synthetic 11-year solar cycle if file not found.
    """
    if not os.path.exists(filepath):
        # Generate synthetic
        rows = []
        for year in range(2000, 2031):
            for month in range(1, 13):
                t = year + (month - 0.5) / 12.0
                phi = 700.0 - 300.0 * np.cos(2.0 * np.pi * (t - 2009.0) / 11.0)
                rows.append({'year': year, 'month': month, 'phi_MV': round(phi, 1),
                             'date': f"{year}-{month:02d}-15"})
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
    else:
        df = pd.read_csv(filepath)

    df['datetime'] = pd.to_datetime(df['date'])
    df = df.set_index('datetime')
    return df


def phi_at_date(date: str, phi_df: pd.DataFrame) -> float:
    """Interpolate modulation potential at a given date string 'YYYY-MM-DD'."""
    target = pd.Timestamp(date)

    idx = phi_df.index
    before = idx[idx <= target]
    after = idx[idx >= target]

    if len(before) == 0:
        return float(phi_df['phi_MV'].iloc[0])
    if len(after) == 0:
        return float(phi_df['phi_MV'].iloc[-1])

    t0 = before[-1]
    t1 = after[0]

    if t0 == t1:
        return float(phi_df.loc[t0, 'phi_MV'])

    phi0 = float(phi_df.loc[t0, 'phi_MV'])
    phi1 = float(phi_df.loc[t1, 'phi_MV'])

    frac = (target - t0).total_seconds() / (t1 - t0).total_seconds()
    return phi0 + frac * (phi1 - phi0)
