"""
gcrisk/neutron_table.py — Tabulated secondary neutron dose equivalent.

2D interpolation table for H_neutron(x_gcm2, phi_MV) derived from HZETRN calculations.
Calibration anchor: H_neutron(16 g/cm² Al, 485 MV) = 2.15 mSv/day.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Shielding depth grid [g/cm²]
_X_GRID = np.array([0.0, 5.0, 10.0, 16.0, 20.0, 30.0, 40.0, 60.0, 100.0])

# Solar modulation potential grid [MV]
_PHI_GRID = np.array([
    300., 400., 500., 550., 600., 700., 800., 900., 1000., 1100., 1200.
])

# Solar modulation scaling: phi_scale(phi) = (phi_ref / phi)^1.1
_PHI_REF = 485.0
_PHI_SCALE = (_PHI_REF / _PHI_GRID) ** 1.1

# Depth profile at phi_ref [mSv/day]
_H_DEPTH_PROFILE = np.array([
    0.05,   # x =   0 g/cm²
    0.90,   # x =   5 g/cm²
    1.58,   # x =  10 g/cm²
    2.15,   # x =  16 g/cm²  ← MSL calibration anchor
    2.40,   # x =  20 g/cm²
    2.70,   # x =  30 g/cm²
    2.85,   # x =  40 g/cm²
    2.80,   # x =  60 g/cm²
    2.55,   # x = 100 g/cm²
])

# H_neutron table [mSv/day]: shape (n_x=9, n_phi=11)
_H_NEUTRON_TABLE = np.outer(_H_DEPTH_PROFILE, _PHI_SCALE)

# Material yield corrections (relative to aluminum)
_MATERIAL_PARAMS = {
    'aluminum':     {'yield': 1.00, 'softness_offset': 0.00},
    'polyethylene': {'yield': 1.55, 'softness_offset': +0.25},
    'water':        {'yield': 1.45, 'softness_offset': +0.20},
    'tissue':       {'yield': 1.35, 'softness_offset': +0.15},
    'iron':         {'yield': 0.85, 'softness_offset': -0.15},
}
_MATERIAL_PARAMS_DEFAULT = {'yield': 1.00, 'softness_offset': 0.00}

_interpolator = RegularGridInterpolator(
    (_X_GRID, _PHI_GRID),
    _H_NEUTRON_TABLE,
    method='linear',
    bounds_error=False,
    fill_value=None,
)


def h_neutron_mSv_day(
    x_gcm2: float,
    phi_MV: float,
    material: str = 'aluminum',
    yield_scale: float = 1.0,
) -> float:
    """
    Interpolated secondary neutron dose equivalent rate [mSv/day].

    Parameters
    ----------
    x_gcm2      : shielding areal density [g/cm²]
    phi_MV      : solar modulation potential [MV]
    material    : shielding material
    yield_scale : multiplicative scale on neutron yield
    """
    x_clamp = float(np.clip(x_gcm2, _X_GRID[0], _X_GRID[-1]))
    phi_clamp = float(np.clip(phi_MV, _PHI_GRID[0], _PHI_GRID[-1]))
    H_al = float(_interpolator([[x_clamp, phi_clamp]])[0])

    mat = _MATERIAL_PARAMS.get(material, _MATERIAL_PARAMS_DEFAULT)
    return H_al * mat['yield'] * yield_scale


def neutron_flux_from_table(
    E_grid_MeV: np.ndarray,
    x_gcm2: float,
    phi_MV: float,
    material: str = 'aluminum',
    neutron_yield_scale: float = 1.0,
) -> np.ndarray:
    """
    Secondary neutron directional flux [cm^{-2} s^{-1} MeV^{-1} sr^{-1}].

    Amplitude set by h_neutron_mSv_day(). Spectral shape is depth-dependent:
    shallow shielding is evaporation-dominated; deep shielding is cascade-dominated.
    """
    from .dose import neutron_h10

    E = np.asarray(E_grid_MeV, dtype=float)

    # Spectral shape
    mat = _MATERIAL_PARAMS.get(material, _MATERIAL_PARAMS_DEFAULT)

    softness_al = 0.55 * np.exp(-x_gcm2 / 35.0) + 0.20
    softness = float(np.clip(softness_al + mat['softness_offset'], 0.05, 0.95))

    E_evap = 5.0
    evap = (E / E_evap**2) * np.exp(-E / E_evap)

    E_lo, E_hi = 20.0, 2000.0
    cascade = np.where(
        E < E_lo,
        E / E_lo,
        np.where(E <= E_hi, E_lo / E, E_lo / E_hi * np.exp(-(E - E_hi) / 500.0)),
    )

    shape = softness * evap + (1.0 - softness) * cascade
    shape = np.maximum(shape, 0.0)

    # Normalize to H_neutron target
    H_target_mSv_day = h_neutron_mSv_day(x_gcm2, phi_MV, material,
                                          yield_scale=neutron_yield_scale)
    H_target_Sv_s = H_target_mSv_day * 1e-3 / 86400.0

    E_ref = np.logspace(0, 5, 400)
    evap_ref = (E_ref / E_evap**2) * np.exp(-E_ref / E_evap)
    cascade_ref = np.where(
        E_ref < E_lo,
        E_ref / E_lo,
        np.where(E_ref <= E_hi, E_lo / E_ref, E_lo / E_hi * np.exp(-(E_ref - E_hi) / 500.0)),
    )
    shape_ref = np.maximum(softness * evap_ref + (1.0 - softness) * cascade_ref, 0.0)

    h10_ref = neutron_h10(E_ref)
    four_pi = 4.0 * np.pi
    trial_H = four_pi * np.trapezoid(shape_ref * h10_ref, E_ref) * 1e-12

    A_0 = H_target_Sv_s / trial_H if trial_H > 0 else 0.0
    return A_0 * shape
