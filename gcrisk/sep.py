"""
gcrisk/sep.py — Solar Energetic Particle (SEP) event module.

Implements Band-function proton spectra for canonical historical SEP events
(Tylka & Lee 2006, ApJ 646, 1319) and provides dose/dose-equivalent estimates
behind shielding.

The SEP module is the "acute clinical risk" complement to the chronic GCR dose
computed by dose.py.  Together they span the full radiation environment a
mission crew faces — the combination being the "whole" physics-to-clinical
chain described in the mission statement.

Canonical events implemented
-----------------------------
aug1972   : August 4, 1972 — worst-case historical proton event.
             Band params: Tylka & Lee (2006) Table 1, Event 4.
oct2003   : October 28, 2003 — "Halloween" storm.
             Band params: Tylka & Lee (2006) Table 1, Event 19.
jan2005   : January 20, 2005 — hard-spectrum event (large high-E tail).
             Band params: Tylka & Lee (2006) Table 1, Event 22.

Units
-----
* Flux : protons cm^{-2} s^{-1} MeV^{-1} sr^{-1}
* Fluence : protons cm^{-2} MeV^{-1}  (flux × event_duration_s)
* Dose : mGy
* Dose equivalent : mSv
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
import pandas as pd

from gcrisk.utils import DEFAULT_E_GRID, MATERIALS, ION_SPECIES

# ---------------------------------------------------------------------------
# Band-function SEP event parameters (Tylka & Lee 2006, Table 1)
# ---------------------------------------------------------------------------
# Format: {event_key: {C_cm2_s_MeV_sr, gamma_low, gamma_high, E_break_MeV, duration_s}}
#
# C        : normalization [cm^{-2} s^{-1} MeV^{-1} sr^{-1}] at 1 MeV
# gamma_low  : low-energy spectral index (< E_break)
# gamma_high : high-energy spectral index (> E_break)
# E_break_MeV: Band-function break energy [MeV]
# duration_s : canonical event duration used for fluence calculation
#
# Band-function form (Band et al. 1993 — originally for gamma-ray bursts,
# adopted by Tylka & Lee 2006 for SEP spectra):
#   J(E) = C × E^{gamma_low} × exp(-E/E_break)     for E < (gamma_low - gamma_high) × E_break
#   J(E) = C × [(γ_l - γ_h)·E_b]^{γ_l - γ_h} × exp(γ_h - γ_l) × E^{gamma_high}   otherwise
#
# NASA 30-day BFO limit: 250 mGy (Cucinotta 2013)

SEP_EVENTS: dict[str, dict] = {
    'aug1972': {
        # Tylka & Lee (2006) Table 1, Event 4 (August 4–5, 1972)
        # Large, soft-spectrum event; worst-case for BFO dose behind thin shielding.
        'C': 4.0e4,
        'gamma_low': -1.78,
        'gamma_high': -3.98,
        'E_break_MeV': 32.0,
        'duration_s': 2 * 86400.0,   # ~2-day integrated fluence
        'description': 'August 1972 — worst-case historical event (Tylka & Lee 2006, Table 1, Event 4)',
    },
    'oct2003': {
        # Tylka & Lee (2006) Table 1, Event 19 (October 28, 2003 Halloween storm)
        # Large fluence but harder spectrum than Aug 1972.
        'C': 1.5e4,
        'gamma_low': -1.40,
        'gamma_high': -3.20,
        'E_break_MeV': 50.0,
        'duration_s': 3 * 86400.0,   # ~3-day event
        'description': 'October 2003 Halloween storm (Tylka & Lee 2006, Table 1, Event 19)',
    },
    'jan2005': {
        # Tylka & Lee (2006) Table 1, Event 22 (January 20, 2005)
        # Hard spectrum — large high-energy tail, significant behind heavy shielding.
        'C': 3.0e2,
        'gamma_low': -0.90,
        'gamma_high': -2.80,
        'E_break_MeV': 110.0,
        'duration_s': 1 * 86400.0,   # ~1-day impulsive event
        'description': 'January 2005 hard-spectrum event (Tylka & Lee 2006, Table 1, Event 22)',
    },
}

# ---------------------------------------------------------------------------
# NASA acute radiation limits
# ---------------------------------------------------------------------------
# 30-day BFO (blood-forming organs) dose limit: 250 mGy  (Cucinotta 2013)
# Annual BFO limit: 500 mGy
# Career BFO limit: 1000–1500 mGy (age/sex dependent)
BFO_30DAY_LIMIT_mGy: float = 250.0
BFO_ANNUAL_LIMIT_mGy: float = 500.0


# ---------------------------------------------------------------------------
# Band function
# ---------------------------------------------------------------------------

def sep_band_spectrum(
    E_MeV: np.ndarray,
    C: float,
    gamma_low: float,
    gamma_high: float,
    E_break_MeV: float,
) -> np.ndarray:
    """
    Tylka-Lee Band-function differential proton flux [cm^{-2} s^{-1} MeV^{-1} sr^{-1}].

    J(E) = C × E^{γ_l} × exp(-E/E_b)                                if E < E_transition
    J(E) = C × [(γ_l - γ_h) × E_b]^{γ_l - γ_h} × exp(γ_h - γ_l) × E^{γ_h}   otherwise

    The transition energy is E_trans = (γ_l - γ_h) × E_b.

    Parameters
    ----------
    E_MeV      : energy array [MeV]
    C          : normalization [cm^{-2} s^{-1} MeV^{-1} sr^{-1}]
    gamma_low  : low-energy index (typically negative, e.g. -1.78)
    gamma_high : high-energy index (more negative, e.g. -3.98)
    E_break_MeV: Band break energy [MeV]

    Returns
    -------
    np.ndarray : differential flux [cm^{-2} s^{-1} MeV^{-1} sr^{-1}]

    References
    ----------
    Band et al. (1993) ApJ 413, 281
    Tylka & Lee (2006) ApJ 646, 1319, Table 1
    """
    E = np.asarray(E_MeV, dtype=float)
    gl, gh, Eb = gamma_low, gamma_high, E_break_MeV

    E_trans = (gl - gh) * Eb  # transition energy [MeV]

    flux = np.empty_like(E)

    low_mask = E < E_trans
    high_mask = ~low_mask

    # Low-energy branch
    flux[low_mask] = C * E[low_mask] ** gl * np.exp(-E[low_mask] / Eb)

    # High-energy branch — continuity guaranteed by construction
    # Prefactor: C × [(γ_l - γ_h) × E_b]^{γ_l - γ_h} × exp(γ_h - γ_l)
    if high_mask.any():
        prefactor = C * ((gl - gh) * Eb) ** (gl - gh) * np.exp(gh - gl)
        flux[high_mask] = prefactor * E[high_mask] ** gh

    return np.maximum(flux, 0.0)


# ---------------------------------------------------------------------------
# Shielded dose from SEP event
# ---------------------------------------------------------------------------

def sep_event_dose(
    event_key: str,
    shielding_x_gcm2: float,
    material: str = 'aluminum',
    E_grid_MeV: np.ndarray | None = None,
) -> dict:
    """
    Compute absorbed dose and dose equivalent behind shielding for a canonical SEP event.

    Only protons are modeled (SEP heavy-ion contribution to BFO dose is <5%,
    Tylka 2006).  Transport is CSDA via transport.energy_after_slab.

    Parameters
    ----------
    event_key       : one of 'aug1972', 'oct2003', 'jan2005'
    shielding_x_gcm2: spacecraft shielding areal density [g/cm²]
    material        : shielding material ('aluminum', 'polyethylene')
    E_grid_MeV      : proton energy grid [MeV]; defaults to DEFAULT_E_GRID

    Returns
    -------
    dict with keys:
        D_mGy             : absorbed dose [mGy]
        H_mSv             : dose equivalent [mSv]
        D_BFO_mGy         : BFO absorbed dose (= D_mGy for protons, tissue target)
        fluence_cm2       : total fluence above 10 MeV [cm^{-2}]
        peak_flux_cm2_s   : peak differential flux at 10 MeV [cm^{-2} s^{-1} MeV^{-1} sr^{-1}]
        exceeds_acute_limit: bool — True if D_BFO_mGy > BFO_30DAY_LIMIT_mGy (250 mGy)
        event             : event metadata dict

    References
    ----------
    Tylka & Lee (2006) ApJ 646, 1319
    Cucinotta et al. (2013) — NASA REID and limits
    """
    # Lazy import to avoid circular dependency
    from gcrisk.transport import energy_after_slab, proton_range
    from gcrisk.dose import quality_factor_icrp60, let_from_energy

    if event_key not in SEP_EVENTS:
        raise ValueError(f"Unknown event '{event_key}'. Choose from {list(SEP_EVENTS)}")

    ev = SEP_EVENTS[event_key]
    if E_grid_MeV is None:
        E_grid_MeV = DEFAULT_E_GRID   # per-nucleon = per-proton

    E = np.asarray(E_grid_MeV, dtype=float)
    C, gl, gh, Eb = ev['C'], ev['gamma_low'], ev['gamma_high'], ev['E_break_MeV']
    duration_s = ev['duration_s']

    # Incident spectrum [cm^{-2} s^{-1} MeV^{-1} sr^{-1}]
    flux_in = sep_band_spectrum(E, C, gl, gh, Eb)

    # CSDA transport: map each exit energy E_out back to E_in
    # For protons, Z=1, A=1, so per-nucleon = total energy.
    # We work in a flux-conservative manner: for each E_out in E_grid,
    # find E_in, then flux_out(E_out) = flux_in(E_in) × dE_in/dE_out
    if shielding_x_gcm2 > 0:
        S_func = _proton_stopping_power(material)
        flux_out = _transport_sep_flux(E, flux_in, shielding_x_gcm2, material, S_func)
    else:
        flux_out = flux_in.copy()

    # Fluence = flux × duration  [cm^{-2} MeV^{-1} sr^{-1}]
    fluence_spectrum = flux_out * duration_s

    # Total fluence above 10 MeV
    mask_10 = E >= 10.0
    if mask_10.any():
        fluence_total = 4.0 * np.pi * trapezoid(fluence_spectrum[mask_10], E[mask_10])
    else:
        fluence_total = 0.0

    # Absorbed dose in tissue [Gy]
    density_tissue = MATERIALS.get('tissue', MATERIALS['water'])['density']
    LET_tissue = let_from_energy(E, Z=1, material='tissue')  # keV/μm
    S_tissue = LET_tissue / (density_tissue * 0.1)           # MeV·cm²/g

    # D = 4π × ∫ fluence_spectrum × S(E) dE  × unit_conversion
    # fluence_spectrum: cm^{-2} MeV^{-1} sr^{-1} ; S: MeV·cm²/g
    # result: MeV/g → × 1.602e-10 → Gy
    D_Gy = 4.0 * np.pi * trapezoid(fluence_spectrum * S_tissue, E) * 1.602e-10
    D_mGy = D_Gy * 1000.0

    # Dose equivalent H = ∫ Q(L) × dD  [Sv]
    Q_arr = np.array([quality_factor_icrp60(L) for L in LET_tissue])
    H_Sv = 4.0 * np.pi * trapezoid(
        fluence_spectrum * S_tissue * Q_arr, E
    ) * 1.602e-10
    H_mSv = H_Sv * 1000.0

    return {
        'D_mGy': float(np.asarray(D_mGy).ravel()[0]),
        'H_mSv': float(np.asarray(H_mSv).ravel()[0]),
        'D_BFO_mGy': float(np.asarray(D_mGy).ravel()[0]),
        'fluence_cm2': fluence_total,
        'peak_flux_cm2_s': float(flux_in[E >= 10.0][0]) if mask_10.any() else 0.0,
        'exceeds_acute_limit': D_mGy > BFO_30DAY_LIMIT_mGy,
        'event': ev,
        'shielding_x_gcm2': shielding_x_gcm2,
        'material': material,
    }


def sep_mission_probability(
    mission_duration_days: float,
    large_event_rate_per_year: float = 0.8,
) -> dict:
    """
    Poisson probability of SEP encounter during a mission.

    A "large event" is one with peak proton flux > 10^4 pfu at >10 MeV
    (NOAA threshold for S3+ class).  Historical rate ≈ 0.8/year near solar max
    (Xapsos et al. 2000, IEEE Trans Nucl Sci 47, 2218).

    Parameters
    ----------
    mission_duration_days    : total mission duration [days]
    large_event_rate_per_year: mean annual large-event rate [yr^{-1}]; default 0.8

    Returns
    -------
    dict:
        lambda_mission   : expected number of large events during mission
        P_zero           : probability of zero events (Poisson)
        P_one_or_more    : 1 - P_zero
        P_two_or_more    : 1 - P_zero - P_one
        duration_days    : mission_duration_days (echo)
        rate_per_year    : large_event_rate_per_year (echo)
    """
    lam = large_event_rate_per_year * mission_duration_days / 365.25
    P0 = np.exp(-lam)
    P1 = lam * np.exp(-lam)

    return {
        'lambda_mission': lam,
        'P_zero': P0,
        'P_one_or_more': 1.0 - P0,
        'P_two_or_more': max(0.0, 1.0 - P0 - P1),
        'duration_days': mission_duration_days,
        'rate_per_year': large_event_rate_per_year,
    }


def sep_dose_with_shielding_scan(
    event_key: str,
    thicknesses_gcm2: list | np.ndarray | None = None,
    material: str = 'aluminum',
    E_grid_MeV: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Shielding effectiveness table — key figure for publication.

    Computes D_BFO_mGy and H_mSv vs shielding thickness for one SEP event.
    Shows that modest Al shielding (5→20 g/cm²) dramatically reduces acute dose.

    Parameters
    ----------
    event_key       : 'aug1972', 'oct2003', or 'jan2005'
    thicknesses_gcm2: list of shielding thicknesses [g/cm²]; default 0–40 g/cm²
    material        : shielding material
    E_grid_MeV      : energy grid; defaults to DEFAULT_E_GRID

    Returns
    -------
    pd.DataFrame with columns: x_gcm2, D_BFO_mGy, H_mSv, exceeds_acute_limit
    """
    if thicknesses_gcm2 is None:
        thicknesses_gcm2 = [0, 1, 2, 5, 10, 16, 20, 30, 40]

    rows = []
    for x in thicknesses_gcm2:
        r = sep_event_dose(event_key, x, material, E_grid_MeV)
        rows.append({
            'x_gcm2': x,
            'D_BFO_mGy': r['D_BFO_mGy'],
            'H_mSv': r['H_mSv'],
            'exceeds_acute_limit': r['exceeds_acute_limit'],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _proton_stopping_power(material: str):
    """Return proton stopping power callable S(E_MeV) [MeV·cm²/g]."""
    from gcrisk.transport import load_stopping_power
    return load_stopping_power(material)


def _transport_sep_flux(
    E_grid: np.ndarray,
    flux_in: np.ndarray,
    x_gcm2: float,
    material: str,
    S_func,
) -> np.ndarray:
    """
    CSDA flux-conservative transport of a proton SEP spectrum through a slab.

    For each exit energy E_out, finds E_in via range inversion, then:
        flux_out(E_out) = flux_in(E_in) × dE_in/dE_out
    where dE_in/dE_out = S(E_in)/S(E_out)  (Jacobian from flux conservation).

    Particles that stop in the slab (R(E_in) < x) contribute zero flux.
    """
    from gcrisk.transport import proton_range

    # Build log-log range→energy inversion on the fly
    from scipy.interpolate import interp1d

    E_dense = np.logspace(np.log10(E_grid[0] * 0.5), np.log10(E_grid[-1] * 2.0), 3000)
    R_dense = proton_range(E_dense, material)  # g/cm²

    # Build E_from_R interpolator in log-log space
    valid = R_dense > 0
    E_from_R = interp1d(
        np.log10(R_dense[valid]),
        np.log10(E_dense[valid]),
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate',
    )

    flux_out = np.zeros_like(flux_in)
    S_in_interp = interp1d(E_dense, S_func(E_dense), kind='linear',
                           bounds_error=False, fill_value=0.0)

    for j, E_out in enumerate(E_grid):
        R_out = float(proton_range(np.array([E_out]), material)[0])
        R_in = R_out + x_gcm2
        if R_in <= 0 or R_out < 0:
            continue
        if R_in > R_dense[-1] * 10:
            # Particle could not have entered the shield with a meaningful energy
            continue

        E_in = 10.0 ** float(E_from_R(np.log10(R_in)))
        if E_in < E_grid[0] or E_in > E_grid[-1] * 2.0:
            # Outside grid — skip
            continue

        S_in = float(S_in_interp(E_in))
        S_out_val = float(np.atleast_1d(S_func(E_out))[0])
        jac = S_in / S_out_val if S_out_val > 0 else 1.0

        # Interpolate incident flux at E_in
        flux_at_Ein = float(np.interp(E_in, E_grid, flux_in, left=0.0, right=0.0))
        flux_out[j] = flux_at_Ein * jac

    return flux_out
