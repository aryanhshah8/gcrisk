"""
gcrisk/trajectory.py — Orbital mechanics and solar modulation.

Computes heliocentric distance r(t) and solar modulation potential phi(t)
along a Mars transfer trajectory.
"""

import numpy as np
import pandas as pd

from .utils import CONSTANTS
from .spectrum import load_usoskin_phi, phi_at_date


def phi_parker(phi_1au: float, r_AU: float, r_min_AU: float = 0.3) -> float:
    """
    Parker spiral solar modulation potential at heliocentric distance r.

    phi(r) = phi_1AU * (1 AU / r)
    """
    r_safe = max(float(r_AU), r_min_AU)
    return phi_1au * (1.0 / r_safe)


def hohmann_transfer_params(r1_AU: float = 1.0, r2_AU: float = 1.524) -> dict:
    """
    Compute Hohmann transfer orbit parameters.

    Returns dict with a_AU, e, T_days, delta_v1_km_s, delta_v2_km_s.
    """
    GM = CONSTANTS['GM_sun']
    AU = CONSTANTS['AU_km']

    r1 = r1_AU * AU
    r2 = r2_AU * AU

    a_km = (r1 + r2) / 2.0
    a_AU = (r1_AU + r2_AU) / 2.0
    e = (r2_AU - r1_AU) / (r2_AU + r1_AU)

    T_seconds = np.pi * np.sqrt(a_km**3 / GM)
    T_days = T_seconds / 86400.0

    v_circ1 = np.sqrt(GM / r1)
    v_circ2 = np.sqrt(GM / r2)
    v_transfer_periapsis = np.sqrt(GM * (2.0 / r1 - 1.0 / a_km))
    v_transfer_apoapsis = np.sqrt(GM * (2.0 / r2 - 1.0 / a_km))

    dv1 = abs(v_transfer_periapsis - v_circ1)
    dv2 = abs(v_circ2 - v_transfer_apoapsis)

    return {
        'a_AU': a_AU,
        'e': e,
        'T_days': T_days,
        'delta_v1_km_s': dv1,
        'delta_v2_km_s': dv2,
    }


def solve_kepler(M: float, e: float, tol: float = 1e-10) -> float:
    """Solve Kepler's equation M = E - e*sin(E) via Newton-Raphson."""
    E = M
    for _ in range(50):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = f / fp
        E -= dE
        if abs(dE) < tol:
            return E
    return E


def heliocentric_distance(t_days: np.ndarray, params: dict) -> np.ndarray:
    """
    Compute heliocentric distance r(t) along Hohmann transfer.

    Parameters
    ----------
    t_days : array, time since departure (days)
    params : dict from hohmann_transfer_params()

    Returns
    -------
    r_AU : np.ndarray
    """
    t = np.asarray(t_days, dtype=float)
    a = params['a_AU']
    e = params['e']
    T = params['T_days']

    n = np.pi / T

    r_AU = np.zeros_like(t)
    for i, ti in enumerate(t):
        M = n * ti
        E = solve_kepler(float(M), e)
        r_AU[i] = a * (1.0 - e * np.cos(E))

    return r_AU


def phi_along_trajectory(
    t_days: np.ndarray,
    r_AU: np.ndarray,
    launch_date: str,
    phi_df: pd.DataFrame,
) -> np.ndarray:
    """
    Compute solar modulation potential along trajectory.

    Combines time variation (solar cycle) and heliocentric distance scaling
    using a blended Parker spiral: phi(r) = phi_earth * (0.5/r + 0.5/r^0.5).
    """
    t = np.asarray(t_days, dtype=float)
    r = np.asarray(r_AU, dtype=float)

    launch = pd.Timestamp(launch_date)
    phi_traj = np.zeros_like(t)

    for i, ti in enumerate(t):
        current_date = launch + pd.Timedelta(days=float(ti))
        date_str = current_date.strftime('%Y-%m-%d')
        phi_earth = phi_at_date(date_str, phi_df)
        phi_traj[i] = phi_earth * (0.5 / r[i] + 0.5 / r[i]**0.5)

    return phi_traj


def generate_trajectory(
    launch_date: str,
    r1_AU: float = 1.0,
    r2_AU: float = 1.524,
    phi_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Generate complete trajectory with time, distance, and phi.

    Returns DataFrame with columns: t_days, r_AU, phi_MV, date.
    """
    import os

    params = hohmann_transfer_params(r1_AU, r2_AU)
    T = int(np.ceil(params['T_days']))

    t_days = np.arange(T + 1, dtype=float)

    r_AU = heliocentric_distance(t_days, params)

    if phi_df is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'usoskin')
        phi_df = load_usoskin_phi(os.path.join(data_dir, 'phi_monthly.csv'))

    phi_MV = phi_along_trajectory(t_days, r_AU, launch_date, phi_df)

    launch = pd.Timestamp(launch_date)
    dates = [(launch + pd.Timedelta(days=float(d))).strftime('%Y-%m-%d')
             for d in t_days]

    return pd.DataFrame({
        't_days': t_days,
        'r_AU': r_AU,
        'phi_MV': phi_MV,
        'date': dates,
    })
