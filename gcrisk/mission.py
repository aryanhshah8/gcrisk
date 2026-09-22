"""
gcrisk/mission.py — Full Mars mission radiation model.

Models a complete crewed Mars mission:
  Phase 1: Earth→Mars transit (Hohmann transfer, ~259 days)
  Phase 2: Mars surface stay (configurable duration, default 500 days)
  Phase 3: Mars→Earth return transit (~259 days)

Uses published RAD surface data (Hassler et al. 2014) for the surface phase,
and the transit pipeline (spectrum + transport + dose) for both transits.

Reference for surface environment: Hassler et al. (2014), Science 343, 1244797.
"""

import logging

import numpy as np
import pandas as pd

from .trajectory import generate_trajectory, hohmann_transfer_params
from .dose import integrate_mission_dose
from .spectrum import load_usoskin_phi, phi_at_date


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mars surface environment (Hassler et al. 2014, RAD instrument)
# ---------------------------------------------------------------------------

# RAD measured surface dose rates (Table 1, Hassler et al. 2014)
_SURFACE_DOSE_RATE_mGy_day = 0.210      # absorbed dose, mGy/day
_SURFACE_H_RATE_mSv_day = 0.64          # dose equivalent, mSv/day
_SURFACE_Q_EFF = _SURFACE_H_RATE_mSv_day / _SURFACE_DOSE_RATE_mGy_day  # ≈ 3.05

# CO2 atmosphere areal density: ~22 g/cm² at zenith, ~16 g/cm² hemisphere-averaged
_MARS_ATMO_GCMS2 = 16.0


def mars_surface_dose(
    n_days: int,
    start_date: str,
    spacecraft_shielding_x_gcm2: float = 0.0,
    shielding_material: str = 'aluminum',
    phi_df: pd.DataFrame = None,
    surface_rate_scale: float = 1.0,
) -> dict:
    """
    Compute radiation dose during Mars surface stay using RAD surface measurements.

    Uses Hassler et al. (2014) published dose rates as the baseline. Additional
    spacecraft/habitat shielding beyond the CO2 atmosphere can be specified.
    GCR flux varies with solar cycle during the surface stay.

    Parameters
    ----------
    n_days : int
        Duration of surface stay in days.
    start_date : str
        Date of Mars arrival ('YYYY-MM-DD').
    spacecraft_shielding_x_gcm2 : float
        Additional shielding (habitat walls) in g/cm². Default 0 = open surface.
    shielding_material : str
        Material of additional shielding.
    phi_df : pd.DataFrame, optional
        Solar modulation potential database. Loaded from default path if None.
    surface_rate_scale : float
        Multiplicative scale on the RAD baseline dose rates (default 1.0).
        Use this to propagate uncertainty in the Hassler et al. (2014) measurement
        (±20% from RAD instrument calibration and spatial variability).

    Returns
    -------
    dict with:
        D_total_mGy: total absorbed dose (mGy)
        H_total_mSv: total dose equivalent (mSv)
        E_effective_mSv: effective dose (tissue-weighted, mSv)
        D_rate_daily: array of daily absorbed dose rates (mGy/day)
        H_rate_daily: array of daily dose equivalent rates (mSv/day)
        mission_duration_days: n_days
        phase: 'surface'
        location: 'Mars surface'

    Notes
    -----
    RAD baseline rates represent the open surface. Additional shielding reduces
    dose by ~10-20% for typical habitat wall thicknesses (5-20 g/cm²).
    Shielding reduction estimated from HZETRN calculations (Slaba et al. 2014).
    Solar modulation variation: ±15% across the solar cycle at Mars.
    """
    import os

    if phi_df is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'usoskin')
        phi_df = load_usoskin_phi(os.path.join(data_dir, 'phi_monthly.csv'))

    start = pd.Timestamp(start_date)
    days = np.arange(n_days)

    # Solar modulation varies during surface stay — modulates GCR by ±15%
    phi_surface = np.zeros(n_days)
    for i, d in enumerate(days):
        current_date = start + pd.Timedelta(days=int(d))
        date_str = current_date.strftime('%Y-%m-%d')
        phi_earth = phi_at_date(date_str, phi_df)
        # At 1.52 AU, phi is lower than at Earth: phi(r) ∝ r^(-0.5)
        phi_surface[i] = phi_earth * (1.0 / 1.524)**0.5

    # Solar modulation effect: dose ∝ 1/(phi/phi_ref)^0.4 approximately
    phi_ref = 500.0  # MV reference (near solar min when RAD measured)
    phi_factor = (phi_ref / np.maximum(phi_surface, 100.0))**0.35

    # Base RAD surface rates (Hassler et al. 2014)
    D_rate_base = _SURFACE_DOSE_RATE_mGy_day
    H_rate_base = _SURFACE_H_RATE_mSv_day

    # Additional shielding reduction (empirical from HZETRN, Slaba et al. 2014)
    # Each 10 g/cm² reduces dose by ~8% for GCR on Mars surface
    shielding_reduction = np.exp(-spacecraft_shielding_x_gcm2 * 0.008)

    D_rate_daily = D_rate_base * phi_factor * shielding_reduction * surface_rate_scale
    H_rate_daily = H_rate_base * phi_factor * shielding_reduction * surface_rate_scale

    D_total = float(np.sum(D_rate_daily))
    H_total = float(np.sum(H_rate_daily))

    # Effective dose: tissue-weighted (same Q_eff as surface measurement)
    from .utils import TISSUE_WEIGHTS
    tissue_weight_sum = sum(TISSUE_WEIGHTS.values())
    E_effective = H_total * tissue_weight_sum

    return {
        'D_total_mGy': D_total,
        'H_total_mSv': H_total,
        'E_effective_mSv': E_effective,
        'D_rate_daily': D_rate_daily,
        'H_rate_daily': H_rate_daily,
        'mission_duration_days': n_days,
        'Q_effective': _SURFACE_Q_EFF,
        'atmo_shielding_gcm2': _MARS_ATMO_GCMS2,
        'extra_shielding_gcm2': spacecraft_shielding_x_gcm2,
        'phase': 'surface',
        'location': 'Mars surface',
        'reference': 'Hassler et al. (2014), Science 343, 1244797',
    }


def run_full_mission(
    launch_date: str,
    surface_days: int = 500,
    shielding_x_gcm2: float = 20.0,
    shielding_material: str = 'aluminum',
    habitat_shielding_x_gcm2: float = 10.0,
    age: int = 35,
    sex: str = 'male',
    phi_df: pd.DataFrame = None,
    E_grid_MeV: np.ndarray = None,
    surface_rate_scale: float = 1.0,
) -> dict:
    """
    Compute total radiation dose for a complete Mars mission.

    Mission profile:
      Phase 1: Earth→Mars transit (Hohmann transfer, ~259 days)
      Phase 2: Mars surface stay (configurable, default 500 days)
      Phase 3: Mars→Earth return (Hohmann transfer, ~259 days)

    Parameters
    ----------
    launch_date : str
        Earth departure date ('YYYY-MM-DD').
    surface_days : int
        Days on Mars surface. Default 500 (nominal long-stay mission).
    shielding_x_gcm2 : float
        Spacecraft shielding during transit (g/cm²). Default 20 g/cm².
    shielding_material : str
        Transit shielding material.
    habitat_shielding_x_gcm2 : float
        Habitat wall shielding on Mars surface (g/cm²). Default 10 g/cm².
    age : int
        Age at exposure for organ-routed REID summary.
    sex : str
        Biological sex used in the REID model.
    phi_df : pd.DataFrame, optional
        Solar modulation database. Loaded from default path if None.
    E_grid_MeV : np.ndarray, optional
        Energy grid for transport calculations.

    Returns
    -------
    dict with:
        phases: dict of per-phase results (transit_out, surface, transit_back)
        total: combined mission totals
        mission_profile: metadata about the mission

    Notes
    -----
    Return transit launch date is estimated as:
      Mars arrival + surface_days + 26-month synodic wait if needed.
    For a nominal 500-day surface stay, the return opportunity is ~26 months
    after Earth departure (Mars-Earth synodic period = 26 months).
    """
    import os
    from datetime import timedelta

    if phi_df is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'usoskin')
        phi_df = load_usoskin_phi(os.path.join(data_dir, 'phi_monthly.csv'))

    # Phase 1: Earth → Mars transit
    params = hohmann_transfer_params(1.0, 1.524)
    transit_days = int(np.ceil(params['T_days']))

    logger.info("Phase 1: Earth→Mars transit (%s days)", transit_days)
    traj_out = generate_trajectory(launch_date, phi_df=phi_df)
    dose_out = integrate_mission_dose(
        traj_out, shielding_x_gcm2, shielding_material, phi_df, E_grid_MeV)

    # Phase 2: Mars surface stay
    arrival_date = (pd.Timestamp(launch_date) +
                    pd.Timedelta(days=transit_days)).strftime('%Y-%m-%d')
    logger.info(
        "Phase 2: Mars surface stay (%s days, arrival %s)",
        surface_days,
        arrival_date,
    )
    dose_surface = mars_surface_dose(
        surface_days, arrival_date,
        habitat_shielding_x_gcm2, shielding_material, phi_df,
        surface_rate_scale=surface_rate_scale)

    # Phase 3: Mars → Earth return transit
    # Return launch: after surface stay. Mars-Earth synodic period ≈ 779 days.
    # A Hohmann return opportunity opens every ~26 months after Mars arrival.
    return_launch = (pd.Timestamp(arrival_date) +
                     pd.Timedelta(days=surface_days)).strftime('%Y-%m-%d')
    logger.info(
        "Phase 3: Mars→Earth return (%s days, departure %s)",
        transit_days,
        return_launch,
    )
    # Return trajectory: Mars at 1.524 AU → Earth at 1.0 AU
    traj_back = generate_trajectory(return_launch, r1_AU=1.524, r2_AU=1.0,
                                    phi_df=phi_df)
    dose_back = integrate_mission_dose(
        traj_back, shielding_x_gcm2, shielding_material, phi_df, E_grid_MeV)

    # Aggregate totals
    D_total = (dose_out['D_total_mGy'] + dose_surface['D_total_mGy'] +
               dose_back['D_total_mGy'])
    H_total = (dose_out['H_total_mSv'] + dose_surface['H_total_mSv'] +
               dose_back['H_total_mSv'])
    E_total = (dose_out['E_effective_mSv'] + dose_surface['E_effective_mSv'] +
               dose_back['E_effective_mSv'])
    total_days = transit_days + surface_days + transit_days

    from .organ_dose import ORGAN_ICRP60_WEIGHTS, integrate_organ_dose
    from .reid import reid_from_organ_doses
    from .utils import TISSUE_WEIGHTS
    tissue_weight_sum = sum(TISSUE_WEIGHTS.values())

    organ_out = integrate_organ_dose(
        traj_out,
        shielding_x=shielding_x_gcm2,
        material=shielding_material,
        phi_df=phi_df,
        E_grid=E_grid_MeV,
    )
    organ_back = integrate_organ_dose(
        traj_back,
        shielding_x=shielding_x_gcm2,
        material=shielding_material,
        phi_df=phi_df,
        E_grid=E_grid_MeV,
    )
    organ_weight_sum = sum(ORGAN_ICRP60_WEIGHTS.values())
    surface_organ_H = {
        organ: dose_surface['E_effective_mSv'] * weight / organ_weight_sum
        for organ, weight in ORGAN_ICRP60_WEIGHTS.items()
    }
    total_organ_H = {
        organ: (
            organ_out['organ_H_mSv'][organ]
            + organ_back['organ_H_mSv'][organ]
            + surface_organ_H[organ]
        )
        for organ in ORGAN_ICRP60_WEIGHTS
    }
    total_organ_D = {
        organ: (
            organ_out['organ_D_mGy'][organ]
            + organ_back['organ_D_mGy'][organ]
        )
        for organ in ORGAN_ICRP60_WEIGHTS
    }
    organ_risk = reid_from_organ_doses(total_organ_H, age_at_exposure=age, sex=sex)
    organ_risk['organ_H_mSv'] = total_organ_H
    organ_risk['organ_D_mGy'] = total_organ_D
    organ_risk['surface_organ_H_mSv'] = surface_organ_H

    logger.info(
        "Mission complete: %s days, D=%.0f mGy, H=%.0f mSv",
        total_days,
        D_total,
        H_total,
    )

    return {
        'phases': {
            'transit_out': dose_out,
            'surface': dose_surface,
            'transit_back': dose_back,
        },
        'total': {
            'D_total_mGy': D_total,
            'H_total_mSv': H_total,
            'E_effective_mSv': E_total,
            'mission_days': total_days,
            'transit_days': 2 * transit_days,
            'surface_days': surface_days,
        },
        'mission_profile': {
            'launch_date': launch_date,
            'mars_arrival': arrival_date,
            'return_launch': return_launch,
            'shielding_transit_gcm2': shielding_x_gcm2,
            'shielding_material': shielding_material,
            'habitat_shielding_gcm2': habitat_shielding_x_gcm2,
            'age': age,
            'sex': sex,
            'surface_rate_scale': surface_rate_scale,
            'surface_dose_reference': 'Hassler et al. (2014)',
        },
        'organ_risk': organ_risk,
    }


def mission_dose_summary(mission_result: dict) -> str:
    """
    Format a readable summary of full mission radiation results.

    Parameters
    ----------
    mission_result : dict
        Output from run_full_mission().

    Returns
    -------
    str
        Multi-line summary string.
    """
    total = mission_result['total']
    profile = mission_result['mission_profile']
    phases = mission_result['phases']

    lines = [
        "=" * 60,
        "FULL MARS MISSION — RADIATION DOSE SUMMARY",
        "=" * 60,
        f"Launch date:          {profile['launch_date']}",
        f"Mars arrival:         {profile['mars_arrival']}",
        f"Return launch:        {profile['return_launch']}",
        f"Total mission:        {total['mission_days']} days",
        f"  Transit (2×):       {total['transit_days']} days",
        f"  Surface:            {total['surface_days']} days",
        "",
        f"Shielding (transit):  {profile['shielding_transit_gcm2']:.1f} g/cm² "
        f"{profile['shielding_material']}",
        f"Shielding (habitat):  {profile['habitat_shielding_gcm2']:.1f} g/cm²",
        "",
        "ABSORBED DOSE (mGy):",
        f"  Transit out:        {phases['transit_out']['D_total_mGy']:.1f}",
        f"  Mars surface:       {phases['surface']['D_total_mGy']:.1f}",
        f"  Transit back:       {phases['transit_back']['D_total_mGy']:.1f}",
        f"  TOTAL:              {total['D_total_mGy']:.1f} mGy",
        "",
        "DOSE EQUIVALENT (mSv):",
        f"  Transit out:        {phases['transit_out']['H_total_mSv']:.1f}",
        f"  Mars surface:       {phases['surface']['H_total_mSv']:.1f}",
        f"  Transit back:       {phases['transit_back']['H_total_mSv']:.1f}",
        f"  TOTAL:              {total['H_total_mSv']:.1f} mSv",
        "",
        "EFFECTIVE DOSE (mSv, tissue-weighted):",
        f"  TOTAL:              {total['E_effective_mSv']:.1f} mSv",
        "=" * 60,
    ]
    return "\n".join(lines)
