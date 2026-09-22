"""
gcrisk/transport.py — Particle transport through shielding.

CSDA energy-loss primitives and Bradt-Peters nuclear cross sections.
"""

import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid

from .utils import (
    CONSTANTS, MATERIALS, MATERIAL_COMPOSITION, ELEMENT_DATA, ION_SPECIES,
    beta, lorentz_gamma,
)


# Cache
_stopping_cache = {}
_range_cache = {}
_ion_range_cache = {}


def load_stopping_power(material: str) -> callable:
    """
    Load NIST stopping power table for protons in given material.

    Returns a scipy interpolation function S(E) [MeV*cm^2/g].
    Falls back to Bethe-Bloch if file not found.
    """
    if material in _stopping_cache:
        return _stopping_cache[material]

    import pandas as pd

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'nist')
    filepath = os.path.join(data_dir, f'proton_{material}.csv')

    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        E = df['KE_MeV'].values
        S = df['stopping_MeV_cm2_g'].values
    else:
        # Fallback: Bethe-Bloch
        E = np.logspace(0, 4, 100)
        S = _bethe_bloch_proton(E, material)

    log_E = np.log10(E)
    log_S = np.log10(np.maximum(S, 1e-10))
    interp = interp1d(log_E, log_S, kind='linear', fill_value='extrapolate')

    def S_func(E_MeV):
        E_arr = np.atleast_1d(np.asarray(E_MeV, dtype=float))
        E_arr = np.maximum(E_arr, 1.0)
        result = 10.0**interp(np.log10(E_arr))
        if result.ndim == 0:
            return float(result)
        return result if result.size > 1 else float(result.item())

    _stopping_cache[material] = S_func
    return S_func


def _bethe_bloch_proton(E_MeV, material):
    """Bethe-Bloch stopping power for protons (fallback)."""
    mp_c2 = CONSTANTS['mp_c2']
    me_c2 = CONSTANTS['me_c2']
    mat = MATERIALS[material]
    composition = MATERIAL_COMPOSITION[material]

    gamma = (E_MeV + mp_c2) / mp_c2
    b2 = 1.0 - 1.0 / gamma**2
    I_MeV = mat['I_eV'] * 1e-6
    K = 0.3071

    S_total = np.zeros_like(E_MeV)
    for elem_sym, w_frac in composition.items():
        elem = ELEMENT_DATA[elem_sym]
        Z_t, A_t = elem['Z'], elem['A']
        T_max = 2.0 * me_c2 * b2 * gamma**2 / (
            1.0 + 2.0 * gamma * me_c2 / mp_c2 + (me_c2 / mp_c2)**2)
        log_arg = np.maximum(2.0 * me_c2 * b2 * gamma**2 * T_max / I_MeV**2, 1.0)
        prefactor = K * (Z_t / A_t) / b2
        S_elem = prefactor * (0.5 * np.log(log_arg) - b2)
        S_total += w_frac * np.maximum(S_elem, 0.01)

    return S_total


def effective_charge(E_MeV_per_n: np.ndarray, Z_ion: int) -> np.ndarray:
    """Energy-dependent effective ion charge (Barkas parameterization)."""
    E = np.atleast_1d(np.asarray(E_MeV_per_n, dtype=float))
    if Z_ion <= 1:
        return np.ones_like(E)

    mp_c2 = CONSTANTS['mp_c2']
    gamma = (E + mp_c2) / mp_c2
    beta_val = np.sqrt(np.maximum(1.0 - 1.0 / gamma**2, 0.0))
    z_eff = Z_ion * (1.0 - np.exp(-125.0 * beta_val * Z_ion ** (-2.0 / 3.0)))
    return np.clip(z_eff, 1.0, float(Z_ion))


def bethe_bloch(E_MeV: np.ndarray, Z_ion: int, material: str) -> np.ndarray:
    """
    Bethe-Bloch stopping power for a heavy ion of charge Z.

    Scales from proton stopping power using effective charge:
        S_ion(E/A, Z) = z_eff(E/A, Z)^2 * S_proton(E/A)
    """
    E = np.atleast_1d(np.asarray(E_MeV, dtype=float))
    S_proton = load_stopping_power(material)
    S_p = np.atleast_1d(S_proton(E))
    z_eff = effective_charge(E, Z_ion)
    return z_eff**2 * S_p


def ion_range(
    E_MeV_per_n: np.ndarray,
    Z_ion: int,
    A_ion: int,
    material: str,
) -> np.ndarray:
    """CSDA range for an ion in g/cm^2 as a function of per-nucleon energy."""
    E = np.atleast_1d(np.asarray(E_MeV_per_n, dtype=float))

    if Z_ion == 1 and A_ion == 1:
        return proton_range(E, material)

    cache_key = (material, int(Z_ion), int(A_ion))
    if cache_key not in _ion_range_cache:
        E_table = np.logspace(0, 5, 500)
        S_vals = np.atleast_1d(bethe_bloch(E_table, Z_ion, material))
        inv_S = 1.0 / np.maximum(S_vals, 0.01)
        R_table = np.zeros_like(E_table)
        R_table[1:] = cumulative_trapezoid(inv_S, E_table)

        _ion_range_cache[cache_key] = (
            E_table,
            R_table,
            interp1d(
                np.log10(E_table),
                np.log10(np.maximum(R_table, 1e-20)),
                kind='cubic',
                fill_value='extrapolate',
            ),
            interp1d(
                np.log10(np.maximum(R_table[1:], 1e-20)),
                np.log10(E_table[1:]),
                kind='cubic',
                fill_value='extrapolate',
            ),
        )

    _, _, R_interp, _ = _ion_range_cache[cache_key]

    result = np.zeros_like(E)
    for i, ei in enumerate(E):
        if ei <= 1.0:
            result[i] = 0.0
        else:
            result[i] = 10.0 ** R_interp(np.log10(ei))

    return result


def energy_from_range(
    R_gcm2: np.ndarray,
    Z_ion: int,
    A_ion: int,
    material: str,
) -> np.ndarray:
    """Invert the cached ion range table and return per-nucleon energy [MeV/n]."""
    R = np.atleast_1d(np.asarray(R_gcm2, dtype=float))

    if Z_ion == 1 and A_ion == 1:
        if material not in _range_cache:
            proton_range(np.array([100.0]), material)
        _, _, _, E_from_R_interp = _range_cache[material]
    else:
        cache_key = (material, int(Z_ion), int(A_ion))
        if cache_key not in _ion_range_cache:
            ion_range(np.array([100.0]), Z_ion, A_ion, material)
        _, _, _, E_from_R_interp = _ion_range_cache[cache_key]

    result = np.zeros_like(R)
    valid = R > 0
    if np.any(valid):
        result[valid] = 10.0 ** E_from_R_interp(np.log10(np.maximum(R[valid], 1e-20)))
    return result


def proton_range(E_MeV: np.ndarray, material: str) -> np.ndarray:
    """
    CSDA range for a proton at energy E in given material.
    Returns range in g/cm^2.
    """
    E = np.atleast_1d(np.asarray(E_MeV, dtype=float))

    cache_key = material
    if cache_key not in _range_cache:
        E_table = np.logspace(0, 5, 500)
        S_func = load_stopping_power(material)
        S_vals = np.atleast_1d(S_func(E_table))

        inv_S = 1.0 / np.maximum(S_vals, 0.01)
        R_table = np.zeros_like(E_table)
        R_table[1:] = cumulative_trapezoid(inv_S, E_table)

        _range_cache[cache_key] = (
            E_table, R_table,
            interp1d(np.log10(E_table), np.log10(np.maximum(R_table, 1e-20)),
                     kind='cubic', fill_value='extrapolate'),
            interp1d(np.log10(np.maximum(R_table[1:], 1e-20)), np.log10(E_table[1:]),
                     kind='cubic', fill_value='extrapolate'),
        )

    E_table, R_table, R_interp, E_from_R_interp = _range_cache[cache_key]

    result = np.zeros_like(E)
    for i, ei in enumerate(E):
        if ei <= 1.0:
            result[i] = 0.0
        else:
            result[i] = 10.0**R_interp(np.log10(ei))

    return result


def energy_after_slab(
    E_in_MeV: float,
    x_gcm2: float,
    material: str,
    Z_ion: int = 1,
    A_ion: int = 1,
) -> float:
    """
    Exit energy of an ion after traversing a slab of given thickness.

    Uses CSDA: R(E_out) = R(E_in) - x. Returns 0.0 if the particle stops.
    """
    if x_gcm2 <= 0:
        return E_in_MeV

    x_eff = x_gcm2 * Z_ion**2 / A_ion

    R_in = float(proton_range(np.array([E_in_MeV / A_ion if A_ion > 1 else E_in_MeV]),
                              material)[0])

    if R_in <= x_eff:
        return 0.0

    R_out = R_in - x_eff

    if R_out <= 0:
        return 0.0

    cache_key = material
    if cache_key not in _range_cache:
        proton_range(np.array([100.0]), material)

    _, _, _, E_from_R_interp = _range_cache[cache_key]
    try:
        E_out = 10.0 ** E_from_R_interp(np.log10(R_out))
    except Exception:
        return 0.0

    if A_ion > 1:
        E_out *= A_ion

    return max(float(E_out), 0.0)


def bradt_peters_cross_section(Z_p: int, A_p: int, Z_t: int, A_t: int) -> float:
    """
    Bradt-Peters total nuclear reaction cross section.
    Returns cross section in cm^2.
    """
    r0 = CONSTANTS['r0_fm'] * 1e-13
    delta = 0.83
    geometric = A_p**(1.0/3.0) + A_t**(1.0/3.0) - delta
    if geometric <= 0:
        return 0.0
    return np.pi * r0**2 * geometric**2


def nuclear_mean_free_path(Z_p: int, A_p: int, material: str) -> float:
    """
    Mean free path for nuclear interaction of ion (Z_p, A_p) in material.
    Returns mean free path in g/cm^2.
    """
    NA = CONSTANTS['NA']
    composition = MATERIAL_COMPOSITION[material]

    inv_lambda_total = 0.0
    for elem_sym, w_frac in composition.items():
        elem = ELEMENT_DATA[elem_sym]
        Z_t, A_t = elem['Z'], elem['A']
        sigma = bradt_peters_cross_section(Z_p, A_p, Z_t, int(round(A_t)))
        if sigma > 0:
            inv_lambda_total += w_frac * NA * sigma / A_t

    if inv_lambda_total <= 0:
        return 1e10

    return 1.0 / inv_lambda_total


def transport_flux_through_slab(
    flux_in: dict,
    x_gcm2: float,
    material: str,
    E_grid_MeV: np.ndarray,
    include_secondaries: bool = True,
) -> dict:
    """
    Transport GCR flux through a shielding slab.

    Parameters
    ----------
    flux_in : dict
        Input flux dict from gcr_total_flux().
    x_gcm2 : float
        Shielding areal density in g/cm².
    material : str
        Shielding material key.
    E_grid_MeV : np.ndarray
        Energy grid in MeV.

    Returns
    -------
    dict : Transported flux with same structure as flux_in plus 'neutron' key.
    """
    if x_gcm2 <= 0:
        return dict(flux_in)

    from .dose import _precompute_transport_factors, _apply_transport_factors

    E_grid = np.asarray(E_grid_MeV, dtype=float)
    factors = _precompute_transport_factors(x_gcm2, material, E_grid)
    return _apply_transport_factors(flux_in, E_grid, factors)
