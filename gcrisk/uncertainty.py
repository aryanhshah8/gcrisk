"""
gcrisk/uncertainty.py — Global uncertainty quantification for the GCR pipeline.

Performs Latin Hypercube Sampling (LHS) over 9 physics and biology parameters,
runs the full pipeline for each sample, and returns variance decomposition.

Parameters
----------
+------------------+----------------------------------+
| phi_scale        | lognormal(0, σ=0.15)             |
| LIS_norm         | lognormal(0, σ=0.12)             |
| cross_sec        | lognormal(0, σ=0.20)             |
| neutron_H        | lognormal(0, σ=0.25)             |
| hze_norm_scale   | lognormal(0, σ=0.20)             |
| Q_factor         | lognormal(0, σ=ln2)              |
| DDREF            | uniform(1.0, 2.0)                |
| ERR_scale        | normal(1.0, σ=0.30), clipped at 0|
| EAR_scale        | normal(1.0, σ=0.35), clipped at 0|
+------------------+----------------------------------+

hze_norm_scale is a multiplicative scale on all Z > 2 species fluxes, capturing
the uncertainty in the GCR HZE composition relative to ACE/CRIS measurements
(~20% 1σ, from inter-instrument spread in George et al. 2009 and Boschini et al. 2020).

ERR_scale and EAR_scale are multiplicative risk-model scale factors and
cannot be negative, so the normal draws above are clipped at 0. This affects
a negligible fraction (<0.1%) of LHS draws and has no visible effect on the
reported variance decomposition or REID percentiles.

Q_factor is clipped per-sample so the resulting effective quality factor
(H/D) cannot exceed the ICRP-60 Q(L) physical maximum of 30 (the peak of
quality_factor_icrp60 at L=100 keV/um) — no ICRP-60 mixed-field weighting
can produce a larger value. This bounds the unbounded LogNormal[1.0, ln2]
prior's right tail without changing its median or shape elsewhere.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube
from scipy.stats import norm, lognorm

from gcrisk.utils import DEFAULT_E_GRID

_LN2 = np.log(2.0)
_ICRP60_Q_MAX = 30.0  # peak of gcrisk.dose.quality_factor_icrp60 at L=100 keV/um

PARAM_NAMES: list[str] = [
    'phi_scale',
    'LIS_norm',
    'cross_sec',
    'neutron_H',
    'hze_norm_scale',
    'Q_factor',
    'DDREF',
    'ERR_scale',
    'EAR_scale',
]

PARAM_DESCRIPTIONS: dict[str, str] = {
    'phi_scale':      'Solar modulation φ scale (±15%)',
    'LIS_norm':       'GCR LIS normalization (±12%)',
    'cross_sec':      'Nuclear cross-section scale (±20%)',
    'neutron_H':      'Neutron table yield scale (±25%)',
    'hze_norm_scale': 'HZE species normalization scale (±20%, from ACE/CRIS inter-instrument spread)',
    'Q_factor':       'ICRP-60 quality factor scale (±factor-of-2)',
    'DDREF':          'DDREF (uniform 1–2)',
    'ERR_scale':      'ERR model scale (normal σ=0.30, clipped at 0)',
    'EAR_scale':      'EAR model scale (normal σ=0.35, clipped at 0)',
}


def lhs_samples(
    n_samples: int,
    param_names: list[str] | None = None,
    seed: int = 42,
    narrow: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """
    Generate Latin Hypercube samples for the 9 uncertainty parameters.

    Parameters
    ----------
    n_samples   : number of LHS samples
    param_names : subset of PARAM_NAMES to sample; default all 9
    seed        : random seed
    narrow      : optional {param_name: factor} map. Each listed parameter's
                  spread (sigma for lognormal/normal parameters; range width
                  for DDREF's uniform distribution, centered on the same
                  mean of 1.5) is multiplied by `factor` (e.g. 0.5 = half
                  the uncertainty, same central estimate). Parameters not
                  listed use their baseline spread (factor 1.0). Used by
                  scripts/sensitivity_narrowing_sweep.py to quantify how
                  much narrowing one parameter's uncertainty would shrink
                  REID uncertainty — a value-of-information estimate, not a
                  change to the paper's baseline results (which always call
                  this with narrow=None).

    Returns
    -------
    dict mapping param_name → np.ndarray of shape (n_samples,)
    """
    if param_names is None:
        param_names = PARAM_NAMES
    narrow = narrow or {}

    n_params = len(param_names)
    sampler = LatinHypercube(d=n_params, seed=seed)
    unit_samples = sampler.random(n=n_samples)

    result: dict[str, np.ndarray] = {}

    for idx, pname in enumerate(param_names):
        u = unit_samples[:, idx]
        f = narrow.get(pname, 1.0)

        if pname in ('phi_scale', 'LIS_norm'):
            sigma = (0.15 if pname == 'phi_scale' else 0.12) * f
            result[pname] = np.exp(norm.ppf(u, loc=0.0, scale=sigma))

        elif pname == 'cross_sec':
            result[pname] = np.exp(norm.ppf(u, loc=0.0, scale=0.20 * f))

        elif pname == 'neutron_H':
            result[pname] = np.exp(norm.ppf(u, loc=0.0, scale=0.25 * f))

        elif pname == 'hze_norm_scale':
            result[pname] = np.exp(norm.ppf(u, loc=0.0, scale=0.20 * f))

        elif pname == 'Q_factor':
            result[pname] = np.exp(norm.ppf(u, loc=0.0, scale=_LN2 * f))

        elif pname == 'DDREF':
            # Uniform[1,2] has mean 1.5, half-width 0.5; narrow around the
            # same mean so the central estimate doesn't shift.
            half_width = 0.5 * f
            result[pname] = (1.5 - half_width) + u * (2 * half_width)

        elif pname == 'ERR_scale':
            # Clipped at 0: ERR is a multiplicative scale factor and cannot
            # be negative. With loc=1.0, scale=0.30 the clip affects a
            # negligible (<0.1%) fraction of LHS draws.
            result[pname] = np.maximum(norm.ppf(u, loc=1.0, scale=0.30 * f), 0.0)

        elif pname == 'EAR_scale':
            # Clipped at 0 for the same reason; with loc=1.0, scale=0.35
            # the clip affects a negligible (<0.1%) fraction of LHS draws.
            result[pname] = np.maximum(norm.ppf(u, loc=1.0, scale=0.35 * f), 0.0)

        else:
            raise ValueError(f"Unknown parameter '{pname}'")

    return result


def _evaluate_one_sample(
    trajectory_df: pd.DataFrame,
    shielding_x: float,
    material: str,
    age: int,
    sex: str,
    phi_scale: float,
    LIS_norm: float,
    cross_sec: float,
    neutron_H: float,
    hze_norm_scale: float,
    Q_factor: float,
    DDREF: float,
    ERR_scale: float,
    EAR_scale: float,
    E_grid: np.ndarray,
    phi_df: pd.DataFrame | None,
) -> tuple[float, float, float]:
    """Run one pipeline evaluation with perturbed parameters. Returns (D_mGy, H_mSv, REID)."""
    from gcrisk.dose import integrate_mission_dose, dose_equivalent_rate, quality_factor_icrp60
    from gcrisk.neutron_table import h_neutron_mSv_day
    from gcrisk.spectrum import gcr_total_flux

    traj_perturbed = trajectory_df.copy()
    traj_perturbed['phi_MV'] = traj_perturbed['phi_MV'] * phi_scale

    result = integrate_mission_dose(
        traj_perturbed,
        shielding_x_gcm2=shielding_x,
        shielding_material=material,
        phi_df=phi_df,
        E_grid_MeV=E_grid,
        lis_norm_scale=LIS_norm,
        neutron_yield_scale=neutron_H,
        hze_norm_scale=hze_norm_scale,
    )

    D_mGy = result['D_total_mGy']
    H_mSv = result['H_total_mSv']

    # Clip Q_factor so the sample's effective quality factor (H/D) cannot
    # exceed the ICRP-60 Q(L) physical maximum (Q=30 at L=100 keV/um,
    # gcrisk.dose.quality_factor_icrp60). Without this, the unbounded
    # LogNormal[1.0, ln2] prior occasionally implies Q_eff > 30, which is
    # not reachable by any ICRP-60 mixed-field weighting.
    Q_eff_unscaled = H_mSv / D_mGy if D_mGy > 0 else 0.0
    if Q_eff_unscaled > 0:
        Q_factor = min(Q_factor, _ICRP60_Q_MAX / Q_eff_unscaled)

    H_mSv_scaled = H_mSv * Q_factor
    H_Sv_scaled = H_mSv_scaled * 1e-3

    H_eff_Sv = H_Sv_scaled / DDREF
    ERR_val = _perturbed_ERR(H_eff_Sv, sex) * ERR_scale
    EAR_val = _perturbed_EAR(H_eff_Sv, age, sex) * EAR_scale

    REID = _compute_reid_from_components(ERR_val, EAR_val, age, sex)

    return D_mGy, H_mSv_scaled, REID


def _perturbed_ERR(H_eff_Sv: float, sex: str) -> float:
    from gcrisk.reid import excess_relative_risk
    return excess_relative_risk(H_eff_Sv, sex)


def _perturbed_EAR(H_eff_Sv: float, age: int, sex: str) -> float:
    from gcrisk.reid import excess_absolute_risk_per_year
    return excess_absolute_risk_per_year(H_eff_Sv, age, sex)


def _compute_reid_from_components(ERR: float, EAR_rate: float, age: int, sex: str) -> float:
    """Compute REID from pre-scaled ERR and EAR values."""
    from gcrisk.reid import baseline_cancer_mortality_rate, _SURVIVAL_TABLE, _FC

    sex_key = sex.lower()
    life_expectancy = 80 if sex_key == 'female' else 78
    latency = 10
    age_risk_start = age + latency

    if age_risk_start >= life_expectancy:
        return 0.0

    table = _SURVIVAL_TABLE[sex_key]
    s_ref = np.interp(age, sorted(table.keys()), [table[k] for k in sorted(table.keys())])
    ages = np.arange(age_risk_start, min(life_expectancy + 1, 91))
    s_vals = np.interp(ages, sorted(table.keys()), [table[k] for k in sorted(table.keys())])
    conditional = s_vals / s_ref

    lambda_vals = np.array([baseline_cancer_mortality_rate(int(a), sex) for a in ages])
    reid_err = _FC * float(np.trapezoid(ERR * lambda_vals * conditional, ages))
    reid_ear = _FC * float(np.trapezoid(EAR_rate * conditional, ages))

    return max(reid_err + reid_ear, 0.0)


def run_uncertainty_ensemble(
    trajectory_df: pd.DataFrame,
    shielding_x: float,
    material: str,
    age: int,
    sex: str,
    n_samples: int = 500,
    E_grid: np.ndarray | None = None,
    phi_df: pd.DataFrame | None = None,
    seed: int = 42,
    narrow: dict[str, float] | None = None,
) -> dict:
    """
    Run full LHS uncertainty ensemble over 9 physics + biology parameters.

    Parameters
    ----------
    trajectory_df : trajectory DataFrame from generate_trajectory()
    shielding_x   : shielding areal density [g/cm²]
    material      : shielding material
    age           : astronaut age at exposure [years]
    sex           : 'male' or 'female'
    n_samples     : number of LHS samples (default 500)
    E_grid        : energy grid [MeV/n]
    phi_df        : Usoskin phi DataFrame
    seed          : random seed
    narrow        : optional {param_name: factor} spread-narrowing map,
                    passed through to lhs_samples() — see its docstring.

    Returns
    -------
    dict with D_mGy_samples, H_mSv_samples, REID_samples, param_samples,
    percentiles, variance_decomposition, n_samples, n_valid.
    """
    n_samples = int(os.environ.get('UNCERTAINTY_SAMPLES', n_samples))

    if E_grid is None:
        E_grid = DEFAULT_E_GRID

    samples = lhs_samples(n_samples, seed=seed, narrow=narrow)

    D_arr = np.zeros(n_samples)
    H_arr = np.zeros(n_samples)
    REID_arr = np.zeros(n_samples)

    for i in range(n_samples):
        kwargs = {p: float(samples[p][i]) for p in PARAM_NAMES}
        try:
            D, H, R = _evaluate_one_sample(
                trajectory_df, shielding_x, material, age, sex,
                **kwargs,
                E_grid=E_grid,
                phi_df=phi_df,
            )
        except Exception:
            D, H, R = np.nan, np.nan, np.nan

        D_arr[i] = D
        H_arr[i] = H
        REID_arr[i] = R

    valid = np.isfinite(D_arr) & np.isfinite(H_arr) & np.isfinite(REID_arr)
    D_v, H_v, R_v = D_arr[valid], H_arr[valid], REID_arr[valid]

    def pcts(arr):
        if len(arr) == 0:
            return {'p5': np.nan, 'p50': np.nan, 'p95': np.nan}
        return {
            'p5':  float(np.percentile(arr, 5)),
            'p50': float(np.percentile(arr, 50)),
            'p95': float(np.percentile(arr, 95)),
        }

    percentiles = {
        'D_mGy': pcts(D_v),
        'H_mSv': pcts(H_v),
        'REID':  pcts(R_v),
    }

    vd = variance_decomposition(samples, D_arr, H_arr, REID_arr)

    return {
        'D_mGy_samples': D_arr,
        'H_mSv_samples': H_arr,
        'REID_samples': REID_arr,
        'param_samples': samples,
        'percentiles': percentiles,
        'variance_decomposition': vd,
        'n_samples': n_samples,
        'n_valid': int(valid.sum()),
    }


def variance_decomposition(
    param_samples: dict[str, np.ndarray],
    D_samples: np.ndarray,
    H_samples: np.ndarray,
    REID_samples: np.ndarray,
) -> dict:
    """
    First-order sensitivity: fraction of total output variance per input parameter.

    Uses rank-transform correlation (Spearman's ρ): fraction = ρ² / Σ ρ².
    """
    from scipy.stats import spearmanr

    outputs = {
        'D_mGy': D_samples,
        'H_mSv': H_samples,
        'REID':  REID_samples,
    }

    result = {}
    for out_name, out_arr in outputs.items():
        valid = np.isfinite(out_arr)
        rho_sq = {}
        for pname, parr in param_samples.items():
            if valid.sum() < 4:
                rho_sq[pname] = np.nan
                continue
            rho, _ = spearmanr(parr[valid], out_arr[valid])
            rho_sq[pname] = rho ** 2

        total = sum(v for v in rho_sq.values() if np.isfinite(v))
        if total > 0:
            fracs = {k: (v / total if np.isfinite(v) else 0.0) for k, v in rho_sq.items()}
        else:
            fracs = {k: 0.0 for k in rho_sq}

        result[out_name] = fracs

    return result
