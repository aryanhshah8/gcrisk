"""
gcrisk/dose.py — Dose equivalent and effective dose.

Converts transported particle flux into absorbed dose and dose equivalent
using ICRP-60 quality factors. Includes neutron contribution via ICRP-74 coefficients.
"""

import os
import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d

from .utils import (
    CONSTANTS, DEFAULT_E_GRID, ION_SPECIES, MATERIALS, TISSUE_WEIGHTS,
)
from .transport import load_stopping_power, bethe_bloch
from .spectrum import gcr_total_flux, load_usoskin_phi


# Angle quadrature for hemisphere-averaged slab transport
_N_ANGLE = 8
_THETA_MAX_DEG = 80.0
_THETAS = np.linspace(0.0, np.radians(_THETA_MAX_DEG), _N_ANGLE)
_COS_THETAS = np.cos(_THETAS)
_SIN_THETAS = np.sin(_THETAS)
_ANGLE_WEIGHTS = _COS_THETAS * _SIN_THETAS
_ANGLE_WEIGHTS = _ANGLE_WEIGHTS / _ANGLE_WEIGHTS.sum()

# Charged fragment surrogate (tuned against MSL/HZETRN transit benchmark)
_CHARGED_FRAGMENT_CHANNELS = {
    'C': [('He', 0.8)],
    'O': [('He', 1.0), ('C', 0.4)],
    'Si': [('O', 1.0), ('He', 1.2)],
    'Fe': [('O', 2.5), ('C', 1.2), ('He', 2.0)],
}


# ICRP-74 neutron fluence-to-dose conversion table
_NEUTRON_H10_TABLE_MeV = np.array([
    10.0, 20.0, 50.0, 100.0, 200.0, 500.0,
    1000.0, 2000.0, 5000.0, 10000.0, 50000.0, 100000.0,
])
_NEUTRON_H10_TABLE_pSv_cm2 = np.array([
    250.0, 300.0, 350.0, 385.0, 400.0, 400.0,
     385.0, 370.0, 350.0, 340.0, 330.0, 325.0,
])
_NEUTRON_KERMA_TABLE_pGy_cm2 = np.array([
    30.0, 40.0, 55.0, 65.0, 70.0, 72.0,
    70.0, 68.0, 65.0, 62.0, 58.0, 55.0,
])

_neutron_h10_interp = interp1d(
    np.log10(_NEUTRON_H10_TABLE_MeV),
    np.log10(_NEUTRON_H10_TABLE_pSv_cm2),
    kind='linear', fill_value='extrapolate',
)
_neutron_kerma_interp = interp1d(
    np.log10(_NEUTRON_H10_TABLE_MeV),
    np.log10(_NEUTRON_KERMA_TABLE_pGy_cm2),
    kind='linear', fill_value='extrapolate',
)


def neutron_h10(E_MeV: np.ndarray) -> np.ndarray:
    """ICRP-74 neutron h*(10) in pSv·cm² as a function of energy."""
    E = np.atleast_1d(np.asarray(E_MeV, dtype=float))
    return 10.0 ** _neutron_h10_interp(np.log10(np.maximum(E, 1.0)))


def neutron_kerma(E_MeV: np.ndarray) -> np.ndarray:
    """Neutron kerma factor in pGy·cm² as a function of energy."""
    E = np.atleast_1d(np.asarray(E_MeV, dtype=float))
    return 10.0 ** _neutron_kerma_interp(np.log10(np.maximum(E, 1.0)))


def let_from_energy(E_MeV_per_n: np.ndarray, Z: int,
                    material: str = 'tissue') -> np.ndarray:
    """
    Compute LET in tissue from ion energy using Bethe-Bloch.
    Returns LET in keV/um.
    """
    E = np.atleast_1d(np.asarray(E_MeV_per_n, dtype=float))
    density = MATERIALS[material]['density']

    if Z == 1:
        S_func = load_stopping_power(material)
        S = np.atleast_1d(S_func(E))
    else:
        S = bethe_bloch(E, Z, material)

    LET = S * density * 0.1
    return LET


def quality_factor_icrp60(LET_keV_um: np.ndarray) -> np.ndarray:
    """
    ICRP-60 quality factor Q(L).

    Q(L) = 1                 for L < 10 keV/um
    Q(L) = 0.32*L - 2.2     for 10 <= L <= 100 keV/um
    Q(L) = 300 / sqrt(L)    for L > 100 keV/um
    """
    L = np.atleast_1d(np.asarray(LET_keV_um, dtype=float))
    Q = np.ones_like(L)

    mask_mid = (L >= 10.0) & (L <= 100.0)
    mask_high = L > 100.0

    Q[mask_mid] = 0.32 * L[mask_mid] - 2.2
    Q[mask_high] = 300.0 / np.sqrt(L[mask_high])

    return np.maximum(Q, 1.0)


def dose_rate_from_flux(
    flux: dict,
    E_grid_MeV: np.ndarray,
    material: str = 'tissue',
) -> dict:
    """
    Compute absorbed dose rate from particle flux.

    dD_i/dt = 4π * integral over E of [flux_i(E) * S_i(E)] dE
    """
    E = np.asarray(E_grid_MeV, dtype=float)
    density = MATERIALS[material]['density']
    four_pi = 4.0 * np.pi

    dose_rates = {}
    LET_spectrum = {}
    total_dose_rate = 0.0

    for sp_key in flux:
        if sp_key in ('total', 'neutron'):
            continue
        sp = ION_SPECIES.get(sp_key)
        if sp is None:
            continue

        Z = sp['Z']
        flux_arr = np.asarray(flux[sp_key], dtype=float)

        LET = let_from_energy(E, Z, material)
        LET_spectrum[sp_key] = LET

        S = LET / (density * 0.1)
        integrand = flux_arr * S
        dD_dt = four_pi * trapezoid(integrand, E)
        dD_dt_Gy = dD_dt * 1.602e-10

        dose_rates[sp_key] = dD_dt_Gy
        total_dose_rate += dD_dt_Gy

    # Neutron absorbed dose
    if 'neutron' in flux:
        n_flux = np.asarray(flux['neutron'], dtype=float)
        kerma = neutron_kerma(E)
        integrand_n = n_flux * kerma
        dD_n = four_pi * trapezoid(integrand_n, E)
        dD_n_Gy = dD_n * 1e-12
        dose_rates['neutron'] = dD_n_Gy
        total_dose_rate += dD_n_Gy

    return {
        'absorbed_dose_rate': dose_rates,
        'dose_rate_total': total_dose_rate,
        'dose_rate_mGy_day': total_dose_rate * 86400.0 * 1000.0,
        'LET_spectrum': LET_spectrum,
    }


def dose_equivalent_rate(
    flux: dict,
    E_grid_MeV: np.ndarray,
) -> dict:
    """Compute dose equivalent rate using ICRP-60 Q(L)."""
    E = np.asarray(E_grid_MeV, dtype=float)
    density = MATERIALS['tissue']['density']
    four_pi = 4.0 * np.pi

    H_total = 0.0
    D_total = 0.0

    for sp_key in flux:
        if sp_key in ('total', 'neutron'):
            continue
        sp = ION_SPECIES.get(sp_key)
        if sp is None:
            continue

        Z = sp['Z']
        flux_arr = np.asarray(flux[sp_key], dtype=float)

        LET = let_from_energy(E, Z, 'tissue')
        Q = quality_factor_icrp60(LET)

        S = LET / (density * 0.1)
        integrand_D = flux_arr * S
        integrand_H = flux_arr * S * Q

        dD_dt = four_pi * trapezoid(integrand_D, E) * 1.602e-10
        dH_dt = four_pi * trapezoid(integrand_H, E) * 1.602e-10

        D_total += dD_dt
        H_total += dH_dt

    # Neutron dose equivalent via ICRP-74 h*(10)
    if 'neutron' in flux:
        n_flux = np.asarray(flux['neutron'], dtype=float)
        h10 = neutron_h10(E)
        kerma = neutron_kerma(E)

        dH_n = four_pi * trapezoid(n_flux * h10, E) * 1e-12
        dD_n = four_pi * trapezoid(n_flux * kerma, E) * 1e-12

        H_total += dH_n
        D_total += dD_n

    Q_eff = H_total / D_total if D_total > 0 else 1.0

    return {
        'H_rate_Sv_s': H_total,
        'H_rate_mSv_day': H_total * 86400.0 * 1000.0,
        'Q_effective': Q_eff,
    }


def effective_dose_rate(
    H_organ: dict,
    tissue_weights: dict = None,
) -> float:
    """Compute effective dose rate from organ dose equivalents."""
    if tissue_weights is None:
        return sum(H_organ.values())

    total = 0.0
    for organ, H in H_organ.items():
        w = tissue_weights.get(organ, 0.0)
        total += w * H
    return total


def _precompute_transport_factors(
    x_gcm2: float,
    material: str,
    E_grid_MeV: np.ndarray,
    alpha_geo: float = 0.5,
) -> dict:
    """
    Precompute transport energy-mapping and Jacobian factors for constant shielding.

    Since shielding is fixed for a mission, these factors depend only on shielding
    geometry, not on the daily GCR flux. Precomputing once gives ~100x speedup.
    """
    from .transport import (
        nuclear_mean_free_path, load_stopping_power, proton_range,
        _range_cache,
    )
    from .transport import proton_range as _pr
    from scipy.special import expn as _expn
    from .utils import ION_SPECIES as _ION_SPECIES

    E_grid = np.asarray(E_grid_MeV, dtype=float)
    factors = {}

    if x_gcm2 <= 0:
        return factors

    mfp_H = nuclear_mean_free_path(1, 1, material)
    frag_prob_H = 1.0 - float(np.exp(-x_gcm2 / mfp_H))
    n_mult_H = 2.0

    proton_range(np.array([100.0]), material)
    _, _, _, E_from_R_interp = _range_cache[material]

    S_func = load_stopping_power(material)

    for sp_key, sp in _ION_SPECIES.items():
        Z, A = sp['Z'], sp['A']

        mfp = nuclear_mean_free_path(Z, A, material)

        # Per-angle CSDA transport
        angle_factors_list = []
        for i_angle in range(_N_ANGLE):
            x_eff_slab = x_gcm2 / _COS_THETAS[i_angle]
            x_eff_csda = x_eff_slab * Z**2 / A

            z_angle = x_eff_slab / mfp
            _s1d = float(np.exp(-z_angle))
            _shemi = float(2.0 * _expn(3, z_angle)) if z_angle > 0 else 1.0
            survival_angle = (1.0 - alpha_geo) * _s1d + alpha_geo * _shemi

            E_in_grid_angle = np.zeros_like(E_grid)
            jacobian_angle = np.ones_like(E_grid)
            valid_mask_angle = np.zeros(len(E_grid), dtype=bool)

            for j, E_out in enumerate(E_grid):
                E_out_per_n = E_out
                R_out = float(_pr(np.array([E_out_per_n]), material)[0])
                if R_out < x_eff_csda:
                    continue
                R_in = R_out + x_eff_csda

                try:
                    E_in_per_n = 10.0**float(
                        E_from_R_interp(np.log10(max(R_in, 1e-20)))
                    )
                except Exception:
                    continue

                if E_in_per_n > E_grid[-1] * 1.5 or E_in_per_n < E_grid[0]:
                    continue

                S_in = float(np.atleast_1d(S_func(E_in_per_n))[0])
                S_out = float(np.atleast_1d(S_func(E_out_per_n))[0])
                jac = S_in / S_out if S_out > 0 else 1.0

                E_in_grid_angle[j] = E_in_per_n
                jacobian_angle[j] = jac
                valid_mask_angle[j] = True

            angle_factors_list.append({
                'weight': float(_ANGLE_WEIGHTS[i_angle]),
                'survival': survival_angle,
                'E_in_grid': E_in_grid_angle,
                'jacobian': jacobian_angle,
                'valid_mask': valid_mask_angle,
            })

        # Neutron production (mean-x thickness)
        mean_survival = sum(
            af['weight'] * af['survival'] for af in angle_factors_list
        )
        n_mult = 2.0 * (A ** 0.6)
        frag_prob = 1.0 - mean_survival
        neutron_E = E_grid / A * 0.5
        neutron_indices = np.searchsorted(E_grid, neutron_E)
        neutron_valid = (neutron_indices > 0) & (neutron_indices < len(E_grid))

        sp_factors = {
            'angle_factors': angle_factors_list,
            'neutron_scale': n_mult * frag_prob,
            'neutron_indices': neutron_indices,
            'neutron_valid': neutron_valid,
        }

        # Secondary protons + cascade neutrons (Z > 1 only)
        if Z > 1:
            sec_p_E = E_grid * 0.7
            sec_p_indices = np.searchsorted(E_grid, sec_p_E)
            sec_p_valid = (sec_p_indices > 0) & (sec_p_indices < len(E_grid))

            cascade_neutron_E = E_grid * 0.35
            cascade_neutron_indices = np.searchsorted(E_grid, cascade_neutron_E)
            cascade_neutron_valid = (
                (cascade_neutron_indices > 0) & (cascade_neutron_indices < len(E_grid))
            )
            cascade_neutron_scale = 0.5 * frag_prob * n_mult_H * frag_prob_H

            sp_factors['sec_proton_scale'] = 0.5 * frag_prob
            sp_factors['sec_proton_indices'] = sec_p_indices
            sp_factors['sec_proton_valid'] = sec_p_valid
            sp_factors['cascade_neutron_scale'] = cascade_neutron_scale
            sp_factors['cascade_neutron_indices'] = cascade_neutron_indices
            sp_factors['cascade_neutron_valid'] = cascade_neutron_valid

        if sp_key in _CHARGED_FRAGMENT_CHANNELS:
            sp_factors['charged_fragment_channels'] = [
                {'species': child, 'scale': scale * frag_prob}
                for child, scale in _CHARGED_FRAGMENT_CHANNELS[sp_key]
            ]

        factors[sp_key] = sp_factors

    return factors


def _apply_transport_factors(
    flux_in: dict,
    E_grid_MeV: np.ndarray,
    factors: dict,
) -> dict:
    """Apply precomputed CSDA transport factors to an input flux dict."""
    from scipy.interpolate import interp1d as _interp1d

    if not factors:
        return dict(flux_in)

    E_grid = np.asarray(E_grid_MeV, dtype=float)
    log_E_grid = np.log10(E_grid)
    flux_out = {}
    total = np.zeros_like(E_grid)
    neutrons = np.zeros_like(E_grid)
    sec_protons = np.zeros_like(E_grid)
    charged_fragments = {}

    for sp_key, sp_factors in factors.items():
        if sp_key not in flux_in:
            continue

        flux_arr = np.asarray(flux_in[sp_key], dtype=float)

        # Angle-averaged transport
        angle_factors = sp_factors.get('angle_factors', [])
        if not angle_factors:
            # Fallback: single-angle structure
            valid = sp_factors.get('valid_mask', np.zeros(len(E_grid), dtype=bool))
            if not np.any(valid):
                flux_out[sp_key] = np.zeros_like(E_grid)
                continue
            survival = sp_factors['survival']
            E_in_grid = sp_factors['E_in_grid']
            jacobian = sp_factors['jacobian']
            log_flux = np.log10(np.maximum(flux_arr, 1e-30))
            flux_interp = _interp1d(log_E_grid, log_flux, kind='linear',
                                    fill_value=-30.0, bounds_error=False)
            transported = np.zeros_like(E_grid)
            E_in_valid = E_in_grid[valid]
            j_in = 10.0**flux_interp(np.log10(np.maximum(E_in_valid, E_grid[0])))
            transported[valid] = j_in * survival * jacobian[valid]
            flux_out[sp_key] = transported
            total += transported
        else:
            log_flux = np.log10(np.maximum(flux_arr, 1e-30))
            flux_interp = _interp1d(log_E_grid, log_flux, kind='linear',
                                    fill_value=-30.0, bounds_error=False)

            transported = np.zeros_like(E_grid)
            for af in angle_factors:
                valid = af['valid_mask']
                if not np.any(valid):
                    continue
                survival = af['survival']
                E_in_grid = af['E_in_grid']
                jacobian = af['jacobian']
                weight = af['weight']

                E_in_valid = E_in_grid[valid]
                j_in = 10.0**flux_interp(
                    np.log10(np.maximum(E_in_valid, E_grid[0]))
                )
                transported[valid] += weight * j_in * survival * jacobian[valid]

            flux_out[sp_key] = transported
            total += transported

        # Secondary neutrons
        n_scale = sp_factors.get('neutron_scale', 0.0)
        n_idx = sp_factors.get('neutron_indices')
        n_valid = sp_factors.get('neutron_valid')
        if n_scale > 0 and n_idx is not None and n_valid is not None:
            producing = n_valid & (flux_arr > 0)
            if np.any(producing):
                np.add.at(neutrons, n_idx[producing], n_scale * flux_arr[producing])

        # Secondary protons + cascade neutrons
        sp_scale = sp_factors.get('sec_proton_scale', 0.0)
        sp_idx = sp_factors.get('sec_proton_indices')
        sp_valid_mask = sp_factors.get('sec_proton_valid')
        cn_scale = sp_factors.get('cascade_neutron_scale', 0.0)
        cn_idx = sp_factors.get('cascade_neutron_indices')
        cn_valid_mask = sp_factors.get('cascade_neutron_valid')

        if sp_scale > 0 and sp_idx is not None and sp_valid_mask is not None:
            producing_sp = sp_valid_mask & (flux_arr > 0)
            if np.any(producing_sp):
                np.add.at(sec_protons, sp_idx[producing_sp],
                           sp_scale * flux_arr[producing_sp])

        if cn_scale > 0 and cn_idx is not None and cn_valid_mask is not None:
            producing_cn = cn_valid_mask & (flux_arr > 0)
            if np.any(producing_cn):
                np.add.at(neutrons, cn_idx[producing_cn],
                           cn_scale * flux_arr[producing_cn])

        fragment_channels = sp_factors.get('charged_fragment_channels', [])
        for channel in fragment_channels:
            child = channel['species']
            scale = channel['scale']
            if scale <= 0:
                continue
            if child not in charged_fragments:
                charged_fragments[child] = np.zeros_like(E_grid)
            charged_fragments[child] += scale * flux_arr

    # Add secondary protons to transported proton flux
    if 'H' in flux_out and np.any(sec_protons > 0):
        flux_out['H'] = flux_out['H'] + sec_protons
        total += sec_protons

    for child, frag_flux in charged_fragments.items():
        if not np.any(frag_flux > 0):
            continue
        flux_out[child] = flux_out.get(child, np.zeros_like(E_grid)) + frag_flux
        total += frag_flux

    if np.any(neutrons > 0):
        flux_out['neutron'] = neutrons

    flux_out['total'] = total
    return flux_out


from .neutron_table import neutron_flux_from_table as _neutron_flux_from_table


def _equilibrium_neutron_flux(
    E_grid_MeV: np.ndarray,
    x_gcm2: float,
    phi_MV: float,
    primary_D_rate_mGy_day: float = 0.0,
    material: str = 'aluminum',
    ref_x_gcm2: float = 16.0,
    ref_D_rate_mGy_day: float = 1.90,
    neutron_yield_scale: float = 1.0,
) -> np.ndarray:
    """Equilibrium secondary neutron flux [cm^{-2} s^{-1} MeV^{-1} sr^{-1}]."""
    return _neutron_flux_from_table(
        E_grid_MeV, x_gcm2, phi_MV, material=material,
        neutron_yield_scale=neutron_yield_scale,
    )


def integrate_mission_dose(
    trajectory_df,
    shielding_x_gcm2: float,
    shielding_material: str,
    phi_df=None,
    E_grid_MeV: np.ndarray = None,
    lis_norm_scale: float = 1.0,
    neutron_yield_scale: float = 1.0,
    hze_norm_scale: float = 1.0,
    alpha_geo: float = 0.5,
) -> dict:
    """
    Integrate dose over a full mission trajectory.

    Chains all modules: spectrum -> transport -> dose for each day.
    Transport factors are precomputed once for the constant shielding geometry.

    Returns
    -------
    dict with D_total_mGy, H_total_mSv, E_effective_mSv, D_rate_daily,
    H_rate_daily, mission_duration_days.
    """
    if E_grid_MeV is None:
        E_grid_MeV = DEFAULT_E_GRID

    n_days = len(trajectory_df)
    D_rate_daily = np.zeros(n_days)
    H_rate_daily = np.zeros(n_days)

    dt_s = 86400.0

    D_total = 0.0
    H_total = 0.0

    # Precompute transport factors once (shielding is constant across the mission)
    if shielding_x_gcm2 > 0:
        transport_factors = _precompute_transport_factors(
            shielding_x_gcm2, shielding_material, E_grid_MeV,
            alpha_geo=alpha_geo)
    else:
        transport_factors = {}

    # Cache dose rates by phi, quantised to 5 MV bins.
    # Phi is daily-interpolated from monthly data so a ~260-day trajectory
    # spans ~26 unique 5 MV bins rather than 260 unique values.  The maximum
    # rounding error is ±2.5 MV (≤0.5% of phi), producing <0.2% flux change —
    # well below the LHS phi_scale uncertainty of ±15%.
    _PHI_BIN_MV = 5.0
    phi_array = trajectory_df['phi_MV'].values.astype(float)
    phi_binned = np.round(phi_array / _PHI_BIN_MV) * _PHI_BIN_MV
    unique_phis = np.unique(phi_binned)

    _phi_cache: dict[float, tuple[float, float]] = {}
    for phi in unique_phis:
        raw_flux = gcr_total_flux(E_grid_MeV, float(phi),
                                   lis_norm_scale=lis_norm_scale,
                                   hze_norm_scale=hze_norm_scale)
        if shielding_x_gcm2 > 0:
            raw_flux = _apply_transport_factors(raw_flux, E_grid_MeV,
                                                transport_factors)
        D_r = dose_rate_from_flux(raw_flux, E_grid_MeV)['dose_rate_total']
        H_r = dose_equivalent_rate(raw_flux, E_grid_MeV)['H_rate_Sv_s']
        eq_n = _equilibrium_neutron_flux(
            E_grid_MeV, shielding_x_gcm2, float(phi),
            material=shielding_material,
            neutron_yield_scale=neutron_yield_scale)
        H_r += dose_equivalent_rate({'neutron': eq_n}, E_grid_MeV)['H_rate_Sv_s']
        _phi_cache[float(phi)] = (D_r, H_r)

    for i in range(n_days):
        D_rate, H_rate = _phi_cache[phi_binned[i]]
        D_rate_daily[i] = D_rate * dt_s * 1000.0
        H_rate_daily[i] = H_rate * dt_s * 1000.0
        D_total += D_rate * dt_s
        H_total += H_rate * dt_s

    tissue_weight_sum = sum(TISSUE_WEIGHTS.values())
    E_effective = H_total * tissue_weight_sum

    return {
        'D_total_mGy': D_total * 1000.0,
        'H_total_mSv': H_total * 1000.0,
        'E_effective_mSv': E_effective * 1000.0,
        'tissue_weight_sum': tissue_weight_sum,
        'D_rate_daily': D_rate_daily,
        'H_rate_daily': H_rate_daily,
        'mission_duration_days': n_days,
        'launch_date': str(trajectory_df['date'].iloc[0]),
        'shielding': {
            'material': shielding_material,
            'thickness_gcm2': shielding_x_gcm2,
        },
    }
