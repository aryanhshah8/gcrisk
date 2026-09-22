"""
gcr/reid.py — Cancer risk (REID) calculation.

Converts effective dose into cancer risk using the NASA REID model,
combining ERR and EAR terms with Monte Carlo uncertainty propagation.
"""

import numpy as np
import pandas as pd

from .dose import integrate_mission_dose
from .organ_dose import ORGAN_ICRP60_WEIGHTS
from .trajectory import generate_trajectory
from .spectrum import load_usoskin_phi


# US life tables — conditional survival S(a | a0)
_SURVIVAL_TABLE = {
    'male': {
        20: 0.9883, 25: 0.9849, 30: 0.9808, 35: 0.9756, 40: 0.9688,
        45: 0.9591, 50: 0.9450, 55: 0.9238, 60: 0.8927, 65: 0.8468,
        70: 0.7820, 75: 0.6888, 80: 0.5636, 85: 0.3980, 90: 0.2141,
    },
    'female': {
        20: 0.9924, 25: 0.9906, 30: 0.9883, 35: 0.9856, 40: 0.9818,
        45: 0.9762, 50: 0.9674, 55: 0.9539, 60: 0.9336, 65: 0.9031,
        70: 0.8574, 75: 0.7881, 80: 0.6824, 85: 0.5231, 90: 0.3248,
    },
}


def _survival_weighted_years(age_start: int, age_end: int, sex: str) -> float:
    """Survival-weighted person-years from age_start to age_end."""
    table = _SURVIVAL_TABLE[sex.lower()]
    s_ref = np.interp(age_start, sorted(table.keys()), [table[k] for k in sorted(table.keys())])

    ages = np.arange(age_start, min(age_end + 1, 91))
    s_vals = np.interp(ages, sorted(table.keys()), [table[k] for k in sorted(table.keys())])
    conditional = s_vals / s_ref

    return float(np.trapezoid(conditional, ages))


def baseline_cancer_mortality_rate(age: int, sex: str) -> float:
    """US background cancer mortality rate [deaths/person/year] from SEER data."""
    rates_male = {
        20: 0.00015, 25: 0.00015, 29: 0.00015,
        30: 0.00034, 35: 0.00034, 39: 0.00034,
        40: 0.00095, 45: 0.00095, 49: 0.00095,
        50: 0.00240, 55: 0.00240, 59: 0.00240,
        60: 0.00540, 65: 0.00540, 69: 0.00540,
        70: 0.01050, 75: 0.01050, 79: 0.01050,
    }
    rates_female = {
        20: 0.00014, 25: 0.00014, 29: 0.00014,
        30: 0.00039, 35: 0.00039, 39: 0.00039,
        40: 0.00115, 45: 0.00115, 49: 0.00115,
        50: 0.00235, 55: 0.00235, 59: 0.00235,
        60: 0.00420, 65: 0.00420, 69: 0.00420,
        70: 0.00820, 75: 0.00820, 79: 0.00820,
    }

    rates = rates_male if sex.lower() == 'male' else rates_female
    ages = sorted(rates.keys())

    if age <= ages[0]:
        return rates[ages[0]]
    if age >= ages[-1]:
        return rates[ages[-1]]

    for i in range(len(ages) - 1):
        if ages[i] <= age <= ages[i + 1]:
            a0, a1 = ages[i], ages[i + 1]
            r0, r1 = rates[a0], rates[a1]
            frac = (age - a0) / (a1 - a0)
            return r0 + frac * (r1 - r0)

    return rates[ages[-1]]


# REID model constants
_DDREF = 1.75
_FC = 0.50


def excess_relative_risk(H_Sv: float, sex: str = 'male') -> float:
    """
    Excess Relative Risk per Sv.

    beta_s = 0.35 Sv^-1 for males, 0.47 Sv^-1 for females.
    """
    beta_s = 0.35 if sex.lower() == 'male' else 0.47
    return beta_s * H_Sv


# Organ EAR parameters
_ORGAN_EAR_PARAMS = {
    'stomach':   {'male': 4.9e-4, 'female': 4.9e-4, 'eta': 0.022},
    'colon':     {'male': 4.2e-4, 'female': 2.4e-4, 'eta': 0.022},
    'lung':      {'male': 6.0e-4, 'female': 9.4e-4, 'eta': 0.018},
    'liver':     {'male': 1.4e-4, 'female': 0.5e-4, 'eta': 0.026},
    'bladder':   {'male': 2.0e-4, 'female': 1.2e-4, 'eta': 0.022},
    'thyroid':   {'male': 0.1e-4, 'female': 1.0e-4, 'eta': 0.024},
    'esophagus': {'male': 0.5e-4, 'female': 0.4e-4, 'eta': 0.022},
    'breast':    {'male': 0.0,    'female': 2.9e-4, 'eta': 0.006},
    'ovary':     {'male': 0.0,    'female': 0.5e-4, 'eta': 0.022},
    'leukemia':  {'male': 2.2e-4, 'female': 1.8e-4, 'eta': 0.018},
    'other':     {'male': 3.0e-4, 'female': 3.0e-4, 'eta': 0.022},
}


def excess_absolute_risk_per_year(H_Sv, age_at_exposure, sex):
    """Total EAR rate summed over all organs."""
    sex_key = sex.lower()
    total_ear = 0.0
    for organ, params in _ORGAN_EAR_PARAMS.items():
        beta = params[sex_key]
        eta = params['eta']
        age_factor = np.exp(eta * (age_at_exposure - 35))
        total_ear += beta * H_Sv * age_factor
    return total_ear


def organ_specific_ear(H_Sv, age_at_exposure, sex):
    """EAR rate per organ. Returns dict mapping organ name → EAR rate (yr^-1)."""
    sex_key = sex.lower()
    result = {}
    for organ, params in _ORGAN_EAR_PARAMS.items():
        beta = params[sex_key]
        eta = params['eta']
        age_factor = np.exp(eta * (age_at_exposure - 35))
        result[organ] = beta * H_Sv * age_factor
    return result


def reid_point_estimate(
    H_total_Sv: float,
    age_at_exposure: int,
    sex: str,
    latency_years: int = 10,
    mission_duration_days: int = 540,
) -> float:
    """
    NASA REID point estimate combining ERR and EAR.

    REID = integral over remaining life of [ERR * lambda_c(a) + EAR(a_e)] * S(a|a_e) da

    This is a simplified deterministic estimate that applies a single fixed
    DDREF (`_DDREF = 1.75`, a commonly cited central value). It is provided
    as a quick scalar API and is independent of the Monte Carlo pathways
    (`reid_with_uncertainty`, `reid_from_organ_doses`) used to generate every
    REID value reported in the manuscript, which instead sample
    DDREF ~ Uniform(1.0, 2.0) per BEIR VII / ICRP-103. No number reported in
    the paper is computed via this function.
    """
    sex_key = sex.lower()
    life_expectancy = 80 if sex_key == 'female' else 78
    age_risk_start = age_at_exposure + latency_years

    if age_risk_start >= life_expectancy:
        return 0.0

    table = _SURVIVAL_TABLE[sex_key]
    s_ref = np.interp(age_at_exposure, sorted(table.keys()),
                      [table[k] for k in sorted(table.keys())])

    ages = np.arange(age_risk_start, min(life_expectancy + 1, 91))
    s_vals = np.interp(ages, sorted(table.keys()),
                       [table[k] for k in sorted(table.keys())])
    conditional = s_vals / s_ref

    H_eff = H_total_Sv / _DDREF
    ERR = excess_relative_risk(H_eff, sex)
    EAR_rate = excess_absolute_risk_per_year(H_eff, age_at_exposure, sex)

    lambda_vals = np.array([baseline_cancer_mortality_rate(int(a), sex) for a in ages])
    reid_err = _FC * float(np.trapezoid(ERR * lambda_vals * conditional, ages))
    reid_ear = _FC * float(np.trapezoid(EAR_rate * conditional, ages))

    return max(reid_err + reid_ear, 0.0)


def reid_with_uncertainty(
    H_total_Sv: float,
    age_at_exposure: int,
    sex: str,
    n_samples: int = 10000,
) -> dict:
    """
    Monte Carlo REID with uncertainty propagation.

    Uncertain parameters:
    1. Quality factor scale: log-normal, geometric_std = 2.0
    2. ERR scale: normal(1.0, 0.30)
    3. DDREF: uniform(1.0, 2.0)
    4. EAR scale: normal(1.0, 0.35)
    """
    sex_key = sex.lower()
    life_expectancy = 80 if sex_key == 'female' else 78
    latency = 10
    age_risk_start = age_at_exposure + latency

    rng = np.random.default_rng(42)

    Q_scale = rng.lognormal(mean=0.0, sigma=np.log(2.0), size=n_samples)
    ERR_scale = rng.normal(1.0, 0.30, size=n_samples)
    DDREF = rng.uniform(1.0, 2.0, size=n_samples)
    EAR_scale = rng.normal(1.0, 0.35, size=n_samples)

    if age_risk_start >= life_expectancy:
        return {'median': 0.0, 'p5': 0.0, 'p95': 0.0, 'mean': 0.0, 'std': 0.0,
                'samples': np.zeros(n_samples), 'nasa_limit': 0.03,
                'exceeds_limit_fraction': 0.0}

    table = _SURVIVAL_TABLE[sex_key]
    _sorted_keys = sorted(table.keys())
    _sorted_vals = [table[k] for k in _sorted_keys]
    ages = np.arange(age_risk_start, min(life_expectancy + 1, 91))
    s_ref = float(np.interp(age_at_exposure, _sorted_keys, _sorted_vals))
    s_vals = np.interp(ages, _sorted_keys, _sorted_vals)
    conditional = s_vals / s_ref
    lambda_vals = np.array([baseline_cancer_mortality_rate(int(a), sex) for a in ages])

    samples = np.zeros(n_samples)
    for i in range(n_samples):
        H_eff = H_total_Sv * Q_scale[i] / DDREF[i]

        ERR = excess_relative_risk(H_eff * ERR_scale[i], sex)
        EAR_rate = excess_absolute_risk_per_year(H_eff, age_at_exposure, sex) * EAR_scale[i]
        reid_err = _FC * float(np.trapezoid(ERR * lambda_vals * conditional, ages))
        reid_ear = _FC * float(np.trapezoid(EAR_rate * conditional, ages))

        samples[i] = reid_err + reid_ear

    samples = np.maximum(samples, 0.0)

    return {
        'median': float(np.median(samples)),
        'p5': float(np.percentile(samples, 5)),
        'p95': float(np.percentile(samples, 95)),
        'mean': float(np.mean(samples)),
        'std': float(np.std(samples)),
        'samples': samples,
        'nasa_limit': 0.03,
        'exceeds_limit_fraction': float(np.mean(samples > 0.03)),
    }


def _organ_ear_rate(H_Sv: float, organ: str, age_at_exposure: int, sex: str) -> float:
    """Single-organ EAR rate."""
    params = _ORGAN_EAR_PARAMS[organ]
    sex_key = sex.lower()
    beta = params[sex_key]
    eta = params['eta']
    age_factor = np.exp(eta * (age_at_exposure - 35))
    return beta * H_Sv * age_factor


def reid_from_organ_doses(
    organ_H_mSv: dict[str, float],
    age_at_exposure: int,
    sex: str,
    n_samples: int = 10000,
) -> dict:
    """Monte Carlo REID using organ-specific dose equivalents for EAR routing."""
    sex_key = sex.lower()
    life_expectancy = 80 if sex_key == 'female' else 78
    latency = 10
    age_risk_start = age_at_exposure + latency

    if age_risk_start >= life_expectancy:
        zeros = {organ: 0.0 for organ in organ_H_mSv}
        return {
            'REID_total': 0.0,
            'REID_by_organ': zeros,
            'uncertainty': {
                'median': 0.0, 'p5': 0.0, 'p95': 0.0,
                'mean': 0.0, 'std': 0.0, 'samples': np.zeros(n_samples),
            },
            'exceeds_limit_fraction': 0.0,
            'effective_H_mSv': 0.0,
        }

    rng = np.random.default_rng(42)
    Q_scale = rng.lognormal(mean=0.0, sigma=np.log(2.0), size=n_samples)
    ERR_scale = rng.normal(1.0, 0.30, size=n_samples)
    DDREF = rng.uniform(1.0, 2.0, size=n_samples)
    EAR_scale = rng.normal(1.0, 0.35, size=n_samples)

    table = _SURVIVAL_TABLE[sex_key]
    sorted_keys = sorted(table.keys())
    sorted_vals = [table[k] for k in sorted_keys]
    ages = np.arange(age_risk_start, min(life_expectancy + 1, 91))
    s_ref = float(np.interp(age_at_exposure, sorted_keys, sorted_vals))
    s_vals = np.interp(ages, sorted_keys, sorted_vals)
    conditional = s_vals / s_ref
    lambda_vals = np.array([baseline_cancer_mortality_rate(int(a), sex) for a in ages])

    effective_H_mSv = sum(
        ORGAN_ICRP60_WEIGHTS.get(organ, 0.0) * dose
        for organ, dose in organ_H_mSv.items()
    )

    samples = np.zeros(n_samples)
    organ_samples = {organ: np.zeros(n_samples) for organ in organ_H_mSv}

    for i in range(n_samples):
        H_eff_Sv = (effective_H_mSv / 1000.0) * Q_scale[i] / DDREF[i]
        ERR = excess_relative_risk(H_eff_Sv * ERR_scale[i], sex)
        reid_err = _FC * float(np.trapezoid(ERR * lambda_vals * conditional, ages))

        reid_total = reid_err
        for organ, H_mSv in organ_H_mSv.items():
            H_organ_Sv = (H_mSv / 1000.0) * Q_scale[i] / DDREF[i]
            EAR_rate = _organ_ear_rate(H_organ_Sv, organ, age_at_exposure, sex)
            reid_organ = _FC * float(
                np.trapezoid(EAR_rate * EAR_scale[i] * conditional, ages)
            )
            organ_samples[organ][i] = max(reid_organ, 0.0)
            reid_total += organ_samples[organ][i]

        samples[i] = max(reid_total, 0.0)

    reid_by_organ = {
        organ: float(np.median(values)) for organ, values in organ_samples.items()
    }

    return {
        'REID_total': float(np.median(samples)),
        'REID_by_organ': reid_by_organ,
        'uncertainty': {
            'median': float(np.median(samples)),
            'p5': float(np.percentile(samples, 5)),
            'p95': float(np.percentile(samples, 95)),
            'mean': float(np.mean(samples)),
            'std': float(np.std(samples)),
            'samples': samples,
        },
        'exceeds_limit_fraction': float(np.mean(samples > 0.03)),
        'effective_H_mSv': effective_H_mSv,
    }


def reid_vs_shielding(
    trajectory_df: pd.DataFrame,
    thicknesses_gcm2: list,
    material: str,
    age: int,
    sex: str,
    phi_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute REID for a range of shielding thicknesses."""
    results = []
    for x in thicknesses_gcm2:
        dose = integrate_mission_dose(trajectory_df, x, material, phi_df)
        H_Sv = dose['H_total_mSv'] / 1000.0
        reid = reid_with_uncertainty(H_Sv, age, sex)
        results.append({
            'thickness_gcm2': x,
            'D_mGy': dose['D_total_mGy'],
            'H_mSv': dose['H_total_mSv'],
            'REID_median': reid['median'],
            'REID_p5': reid['p5'],
            'REID_p95': reid['p95'],
        })
    return pd.DataFrame(results)


def reid_vs_launch_date(
    launch_dates: list,
    shielding_x_gcm2: float,
    material: str,
    age: int,
    sex: str,
    phi_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute REID for a range of launch dates (captures solar cycle variation)."""
    results = []
    for date in launch_dates:
        traj = generate_trajectory(date, phi_df=phi_df)
        dose = integrate_mission_dose(traj, shielding_x_gcm2, material, phi_df)
        H_Sv = dose['H_total_mSv'] / 1000.0
        reid = reid_with_uncertainty(H_Sv, age, sex)
        from .spectrum import phi_at_date
        phi_launch = phi_at_date(date, phi_df)
        results.append({
            'launch_date': date,
            'phi_at_launch': phi_launch,
            'D_mGy': dose['D_total_mGy'],
            'H_mSv': dose['H_total_mSv'],
            'REID_median': reid['median'],
            'REID_p5': reid['p5'],
            'REID_p95': reid['p95'],
        })
    return pd.DataFrame(results)
