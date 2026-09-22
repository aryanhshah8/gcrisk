"""
gcr/paper2.py — Analysis layer for the Paper 2 follow-up.

Paper 2 builds on the Paper 1 pipeline without modifying any Paper 1 code
path. This module is strictly additive: every function here imports from
the frozen Paper 1 modules (gcr.uncertainty, gcr.reid, gcr.rbe, gcr.dose)
and composes new analyses on top of them.

Three contributions, matching the Paper 2 plan:

1. ``lhs_samples_with_priors``  — LHS sampler that accepts per-parameter
   distribution overrides, so Paper 1's baseline priors stay untouched
   while Paper 2 can run robustness checks under widened priors (e.g. a
   BEIR VII-style DDREF window) as a *sensitivity analysis*.

2. ``let_binned_dose_contribution`` + ``conditioned_let_sensitivity``  —
   LET-binned attribution of REID sensitivity. This is framed as a
   conditioned sensitivity attribution, NOT an orthogonal variance
   decomposition: REID is nonlinear in total weighted dose, so bin
   contributions are not guaranteed to sum to the full variance. The
   attribution weight per bin is the baseline fraction of dose-equivalent
   H that comes from that LET bin.

3. ``reid_with_custom_biological_priors`` + ``decision_threshold_sweep``  —
   Decision-threshold inversion against the pre-2023 NASA 3% REID
   benchmark. Sweeps the width of the biological priors (Q_factor, DDREF,
   ERR_scale, EAR_scale) as a single "biological precision" scale factor
   and reports ``exceeds_limit_fraction`` vs. precision. The 2023 NASA
   career-limit framework is the current operational standard; the 3%
   threshold is retained here as the historical benchmark cited in
   Cucinotta 2013 and Paper 1.

Nothing in this module is imported by Paper 1 code. Removing this file
leaves Paper 1's outputs identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.stats import norm, spearmanr
from scipy.stats.qmc import LatinHypercube

from .dose import let_from_energy, neutron_h10, quality_factor_icrp60
from .reid import (
    _FC,
    _SURVIVAL_TABLE,
    baseline_cancer_mortality_rate,
    excess_absolute_risk_per_year,
    excess_relative_risk,
)
from .uncertainty import (
    PARAM_NAMES,
    _evaluate_one_sample,
    variance_decomposition,
)
from .utils import DEFAULT_E_GRID, ION_SPECIES, MATERIALS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAPER1_PRIORS: dict[str, dict] = {
    'phi_scale':     {'kind': 'lognormal', 'sigma': 0.15},
    'LIS_norm':      {'kind': 'lognormal', 'sigma': 0.12},
    'cross_sec':     {'kind': 'lognormal', 'sigma': 0.20},
    'neutron_H':     {'kind': 'lognormal', 'sigma': 0.25},
    'hze_norm_scale':{'kind': 'lognormal', 'sigma': 0.20},
    'Q_factor':      {'kind': 'lognormal', 'sigma': float(np.log(2.0))},
    'DDREF':         {'kind': 'uniform',   'low': 1.0, 'high': 2.0},
    'ERR_scale':     {'kind': 'normal',    'loc': 1.0, 'scale': 0.30, 'clip_min': 0.0},
    'EAR_scale':     {'kind': 'normal',    'loc': 1.0, 'scale': 0.35, 'clip_min': 0.0},
}

# BEIR VII-style widened DDREF prior used as a sensitivity check only.
BEIR_VII_DDREF_PRIOR: dict = {'kind': 'uniform', 'low': 1.0, 'high': 3.0}

BIOLOGICAL_PARAM_NAMES: tuple[str, ...] = (
    'Q_factor', 'DDREF', 'ERR_scale', 'EAR_scale',
)

PHYSICS_PARAM_NAMES: tuple[str, ...] = (
    'phi_scale', 'LIS_norm', 'cross_sec', 'neutron_H', 'hze_norm_scale',
)

# Plan §2 bins: below 3 keV/µm = proton plateau regime; 3–10 = proton/α;
# 10–30 = proton Bragg peak and light HZE; 30–100 = heavy HZE (C, O, Si);
# >100 = Fe-like track cores.
DEFAULT_LET_BINS_KEV_UM: tuple[float, ...] = (0.0, 3.0, 10.0, 30.0, 100.0, np.inf)

# Historical (pre-2023) NASA REID career limit used as the benchmark
# comparison point in Cucinotta 2013 and Paper 1.
HISTORICAL_NASA_REID_THRESHOLD: float = 0.03


# ---------------------------------------------------------------------------
# 1. Widened-prior LHS sampler  (Paper 2 sensitivity analysis, item 1)
# ---------------------------------------------------------------------------

def _draw_from_spec(u: np.ndarray, spec: dict) -> np.ndarray:
    """Map a uniform(0,1) quantile array through a named distribution spec."""
    kind = spec['kind']
    if kind == 'lognormal':
        result = np.exp(norm.ppf(u, loc=0.0, scale=spec['sigma']))
    elif kind == 'uniform':
        low, high = spec['low'], spec['high']
        result = low + u * (high - low)
    elif kind == 'normal':
        result = norm.ppf(u, loc=spec['loc'], scale=spec['scale'])
    else:
        raise ValueError(f"Unknown distribution kind '{kind}'")

    if 'clip_min' in spec:
        result = np.maximum(result, spec['clip_min'])
    return result


def lhs_samples_with_priors(
    n_samples: int,
    prior_overrides: dict[str, dict] | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """LHS sampler that matches Paper 1 exactly unless priors are overridden.

    Parameters
    ----------
    n_samples
        Number of LHS samples to draw.
    prior_overrides
        Mapping of parameter name → distribution spec. Accepted specs:

        - ``{'kind': 'lognormal', 'sigma': float}``  (median 1.0)
        - ``{'kind': 'uniform',   'low': float, 'high': float}``
        - ``{'kind': 'normal',    'loc': float, 'scale': float}``

        Any parameter not listed falls back to Paper 1's prior in
        :data:`PAPER1_PRIORS`. Passing ``None`` exactly reproduces
        :func:`gcr.uncertainty.lhs_samples` for identical seeds and sample
        counts (aside from LHS stratification over all 9 dimensions, which
        is identical by construction).
    seed
        Random seed for the Latin Hypercube design.
    """
    if prior_overrides is None:
        prior_overrides = {}

    priors = {name: prior_overrides.get(name, PAPER1_PRIORS[name])
              for name in PARAM_NAMES}

    sampler = LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    unit_samples = sampler.random(n=n_samples)

    return {
        name: _draw_from_spec(unit_samples[:, idx], priors[name])
        for idx, name in enumerate(PARAM_NAMES)
    }


def run_ensemble_with_priors(
    trajectory_df: pd.DataFrame,
    shielding_x: float,
    material: str,
    age: int,
    sex: str,
    prior_overrides: dict[str, dict] | None = None,
    n_samples: int = 500,
    E_grid: np.ndarray | None = None,
    phi_df: pd.DataFrame | None = None,
    seed: int = 42,
) -> dict:
    """Run the full Paper 1 pipeline ensemble under custom priors.

    Mirrors :func:`gcr.uncertainty.run_uncertainty_ensemble` but replaces
    the sampler with :func:`lhs_samples_with_priors`. The per-sample
    evaluation itself is Paper 1's :func:`_evaluate_one_sample` — no
    Paper 1 code is modified.

    ``prior_overrides=None`` reproduces Paper 1 exactly for identical
    ``seed`` and ``n_samples`` (up to LHS stratification equivalence).
    """
    if E_grid is None:
        E_grid = DEFAULT_E_GRID

    samples = lhs_samples_with_priors(
        n_samples, prior_overrides=prior_overrides, seed=seed,
    )

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

    def _pcts(arr: np.ndarray) -> dict[str, float]:
        if arr.size == 0:
            return {'p5': np.nan, 'p50': np.nan, 'p95': np.nan}
        return {
            'p5':  float(np.percentile(arr, 5)),
            'p50': float(np.percentile(arr, 50)),
            'p95': float(np.percentile(arr, 95)),
        }

    percentiles = {
        'D_mGy': _pcts(D_arr[valid]),
        'H_mSv': _pcts(H_arr[valid]),
        'REID':  _pcts(REID_arr[valid]),
    }

    return {
        'D_mGy_samples': D_arr,
        'H_mSv_samples': H_arr,
        'REID_samples': REID_arr,
        'param_samples': samples,
        'percentiles': percentiles,
        'variance_decomposition': variance_decomposition(
            samples, D_arr, H_arr, REID_arr,
        ),
        'n_samples': n_samples,
        'n_valid': int(valid.sum()),
        'prior_overrides': prior_overrides or {},
    }


# ---------------------------------------------------------------------------
# 2. LET-binned dose attribution  (Paper 2 items 2 and 3)
# ---------------------------------------------------------------------------

@dataclass
class LetBinnedDose:
    """Per-(species, LET-bin) decomposition of absorbed dose and dose equivalent.

    ``bin_edges_keV_um`` has length ``n_bins + 1``. Arrays indexed by species
    hold length-``n_bins`` values. The ``dose_fraction_by_bin`` and
    ``H_fraction_by_bin`` arrays sum across species and are normalized to
    sum to 1 over bins (up to floating-point tolerance) when the total is
    nonzero.
    """
    bin_edges_keV_um: np.ndarray
    dose_by_species_bin: dict[str, np.ndarray]
    H_by_species_bin: dict[str, np.ndarray]
    dose_by_bin: np.ndarray
    H_by_bin: np.ndarray
    dose_fraction_by_bin: np.ndarray
    H_fraction_by_bin: np.ndarray
    total_dose_Gy_s: float
    total_H_Sv_s: float
    material: str


def let_binned_dose_contribution(
    flux: dict,
    E_grid_MeV: np.ndarray,
    material: str = 'tissue',
    bin_edges_keV_um: tuple[float, ...] | None = None,
    species: list[str] | None = None,
) -> LetBinnedDose:
    """Partition absorbed dose and dose-equivalent H by (species, LET bin).

    Uses the same Bethe-Bloch LET tables as Paper 1 (:func:`let_from_energy`)
    and the same ICRP-60 Q(L) quality factor
    (:func:`quality_factor_icrp60`), so numbers are consistent with
    :func:`gcr.dose.dose_rate_from_flux`. Neutrons are folded in via the
    ICRP-74 h*(10) conversion applied at their own kinetic energies; their
    contribution is placed in the low-LET bin under the convention used by
    Paper 1 for tissue-equivalent neutron dose.
    """
    if bin_edges_keV_um is None:
        bin_edges_keV_um = DEFAULT_LET_BINS_KEV_UM

    edges = np.asarray(bin_edges_keV_um, dtype=float)
    n_bins = len(edges) - 1
    if n_bins < 1:
        raise ValueError("bin_edges_keV_um must define at least one bin")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("bin_edges_keV_um must be strictly increasing")

    E = np.asarray(E_grid_MeV, dtype=float)
    density = MATERIALS[material]['density']
    four_pi = 4.0 * np.pi
    MeV_per_g_to_Gy = 1.602e-10

    if species is None:
        species = list(ION_SPECIES.keys())

    dose_by_species_bin: dict[str, np.ndarray] = {}
    H_by_species_bin: dict[str, np.ndarray] = {}
    dose_by_bin = np.zeros(n_bins)
    H_by_bin = np.zeros(n_bins)

    for sp_key in species:
        if sp_key not in flux or sp_key not in ION_SPECIES:
            continue
        Z = ION_SPECIES[sp_key]['Z']
        flux_arr = np.asarray(flux[sp_key], dtype=float)

        LET = let_from_energy(E, Z, material)            # keV/µm
        Q = quality_factor_icrp60(LET)                   # dimensionless
        S = LET / (density * 0.1)                        # MeV cm^2 / g

        dose_integrand = four_pi * flux_arr * S * MeV_per_g_to_Gy
        H_integrand = dose_integrand * Q

        dose_bins_sp = np.zeros(n_bins)
        H_bins_sp = np.zeros(n_bins)
        for b in range(n_bins):
            mask = (LET >= edges[b]) & (LET < edges[b + 1])
            if not mask.any():
                continue
            dose_bins_sp[b] = trapezoid(np.where(mask, dose_integrand, 0.0), E)
            H_bins_sp[b] = trapezoid(np.where(mask, H_integrand, 0.0), E)

        dose_by_species_bin[sp_key] = dose_bins_sp
        H_by_species_bin[sp_key] = H_bins_sp
        dose_by_bin += dose_bins_sp
        H_by_bin += H_bins_sp

    if 'neutron' in flux:
        n_flux = np.asarray(flux['neutron'], dtype=float)
        h10 = neutron_h10(E)
        # Neutron dose and dose-equivalent (ICRP-74 fluence-to-dose).
        dH_n_arr = four_pi * n_flux * h10 * 1e-12
        H_n = float(trapezoid(dH_n_arr, E))
        # Neutron absorbed dose sits in the low-LET bin by Paper 1 convention.
        if n_bins > 0:
            H_by_bin[0] += H_n
            H_by_species_bin['neutron'] = np.zeros(n_bins)
            H_by_species_bin['neutron'][0] = H_n

    total_dose = float(dose_by_bin.sum())
    total_H = float(H_by_bin.sum())

    def _safe_fractions(arr: np.ndarray, total: float) -> np.ndarray:
        return arr / total if total > 0 else np.zeros_like(arr)

    return LetBinnedDose(
        bin_edges_keV_um=edges,
        dose_by_species_bin=dose_by_species_bin,
        H_by_species_bin=H_by_species_bin,
        dose_by_bin=dose_by_bin,
        H_by_bin=H_by_bin,
        dose_fraction_by_bin=_safe_fractions(dose_by_bin, total_dose),
        H_fraction_by_bin=_safe_fractions(H_by_bin, total_H),
        total_dose_Gy_s=total_dose,
        total_H_Sv_s=total_H,
        material=material,
    )


def conditioned_let_sensitivity(
    ensemble_result: dict,
    let_binned: LetBinnedDose,
    output: str = 'REID',
) -> pd.DataFrame:
    """First-order attribution of REID sensitivity onto LET bins.

    The attribution is defined as

        attribution(param, bin)  =  Spearman ρ²(param, output)  ×
                                   H_fraction(bin)

    where the LET-bin weight is the baseline dose-equivalent fraction from
    that bin. This treats the biological parameters (Q_factor, DDREF,
    ERR_scale, EAR_scale) as uniform scalar multipliers on H — which is
    how Paper 1's pipeline applies them — so the rank-correlation signal
    of each parameter is redistributed across bins proportionally to
    where the risk originates in the LET spectrum.

    This is a conditioned sensitivity attribution, not a variance
    decomposition: the parameters themselves do not act selectively per
    LET bin in the current pipeline, and cross-bin correlations induced
    by shielding mean the column sums over bins are constrained to equal
    the full Spearman ρ² for that parameter only by construction of the
    weighting, not as an independent result.

    Parameters
    ----------
    ensemble_result
        Output of :func:`gcr.uncertainty.run_uncertainty_ensemble`, i.e. a
        dict with ``param_samples`` and ``REID_samples`` (or another
        output key named by ``output``).
    let_binned
        Output of :func:`let_binned_dose_contribution` for the baseline
        (unperturbed) shielded flux.
    output
        Which output to attribute. One of ``'REID'``, ``'H_mSv'``,
        ``'D_mGy'``.

    Returns
    -------
    pandas.DataFrame
        Rows: parameter names. Columns: LET-bin labels (e.g. ``"3–10"``,
        ``"≥100"``). Values: attribution weights ρ² × H_fraction.
    """
    sample_key = f'{output}_samples'
    if sample_key not in ensemble_result:
        raise KeyError(
            f"ensemble_result missing '{sample_key}'; got keys "
            f"{sorted(ensemble_result.keys())}"
        )

    out_arr = np.asarray(ensemble_result[sample_key], dtype=float)
    param_samples = ensemble_result['param_samples']
    valid = np.isfinite(out_arr)

    rho_sq: dict[str, float] = {}
    for pname in PARAM_NAMES:
        parr = np.asarray(param_samples[pname], dtype=float)
        if valid.sum() < 4:
            rho_sq[pname] = float('nan')
            continue
        rho, _ = spearmanr(parr[valid], out_arr[valid])
        rho_sq[pname] = float(rho) ** 2 if np.isfinite(rho) else float('nan')

    edges = let_binned.bin_edges_keV_um
    bin_labels = _format_bin_labels(edges)
    weights = let_binned.H_fraction_by_bin

    data = {label: [] for label in bin_labels}
    for pname in PARAM_NAMES:
        r = rho_sq[pname]
        for b, label in enumerate(bin_labels):
            data[label].append(r * weights[b] if np.isfinite(r) else np.nan)

    return pd.DataFrame(data, index=list(PARAM_NAMES))


def _format_bin_labels(edges: np.ndarray) -> list[str]:
    labels: list[str] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if np.isinf(hi):
            labels.append(f'≥{lo:g}')
        elif lo == 0.0:
            labels.append(f'<{hi:g}')
        else:
            labels.append(f'{lo:g}–{hi:g}')
    return labels


# ---------------------------------------------------------------------------
# 3. Decision-threshold inversion  (Paper 2 item 4)
# ---------------------------------------------------------------------------

def reid_with_custom_biological_priors(
    H_total_Sv: float,
    age_at_exposure: int,
    sex: str,
    n_samples: int = 10_000,
    Q_sigma: float | None = None,
    DDREF_low: float = 1.0,
    DDREF_high: float = 2.0,
    ERR_sigma: float = 0.30,
    EAR_sigma: float = 0.35,
    seed: int = 42,
) -> dict:
    """Monte Carlo REID with configurable biological prior widths.

    Mirrors :func:`gcr.reid.reid_with_uncertainty` exactly for default
    arguments; exposes ``Q_sigma``, ``DDREF_low``, ``DDREF_high``,
    ``ERR_sigma``, ``EAR_sigma`` so the caller can sweep biological
    precision without touching Paper 1 code.

    Parameters
    ----------
    H_total_Sv
        Total dose equivalent in Sv (before biological rescaling).
    age_at_exposure, sex
        Crew demographics.
    n_samples
        MC sample count (matches Paper 1's 10 000 default).
    Q_sigma
        Log-normal σ for the Q_factor multiplier. If ``None``, uses
        Paper 1's ``ln(2)``.
    DDREF_low, DDREF_high
        Uniform bounds for DDREF. Paper 1 uses (1.0, 2.0). BEIR VII-style
        widening uses (1.0, 3.0).
    ERR_sigma, EAR_sigma
        Normal σ on the ERR and EAR scale factors.
    seed
        RNG seed. Paper 1's internal call uses seed 42.

    Returns
    -------
    dict
        ``samples``, ``median``, ``p5``, ``p95``, ``mean``, ``std``,
        ``exceeds_limit_fraction`` (vs :data:`HISTORICAL_NASA_REID_THRESHOLD`).
    """
    if Q_sigma is None:
        Q_sigma = float(np.log(2.0))

    sex_key = sex.lower()
    life_expectancy = 80 if sex_key == 'female' else 78
    latency = 10
    age_risk_start = age_at_exposure + latency

    if age_risk_start >= life_expectancy:
        zeros = np.zeros(n_samples)
        return {
            'median': 0.0, 'p5': 0.0, 'p95': 0.0, 'mean': 0.0, 'std': 0.0,
            'samples': zeros,
            'nasa_limit': HISTORICAL_NASA_REID_THRESHOLD,
            'exceeds_limit_fraction': 0.0,
        }

    rng = np.random.default_rng(seed)
    Q_scale = rng.lognormal(mean=0.0, sigma=Q_sigma, size=n_samples)
    ERR_scale = rng.normal(1.0, ERR_sigma, size=n_samples)
    DDREF = rng.uniform(DDREF_low, DDREF_high, size=n_samples)
    EAR_scale = rng.normal(1.0, EAR_sigma, size=n_samples)

    table = _SURVIVAL_TABLE[sex_key]
    sorted_keys = sorted(table.keys())
    sorted_vals = [table[k] for k in sorted_keys]
    ages = np.arange(age_risk_start, min(life_expectancy + 1, 91))
    s_ref = float(np.interp(age_at_exposure, sorted_keys, sorted_vals))
    s_vals = np.interp(ages, sorted_keys, sorted_vals)
    conditional = s_vals / s_ref
    lambda_vals = np.array([
        baseline_cancer_mortality_rate(int(a), sex) for a in ages
    ])

    samples = np.zeros(n_samples)
    for i in range(n_samples):
        H_eff = H_total_Sv * Q_scale[i] / DDREF[i]
        ERR = excess_relative_risk(H_eff * ERR_scale[i], sex)
        EAR_rate = excess_absolute_risk_per_year(
            H_eff, age_at_exposure, sex
        ) * EAR_scale[i]
        reid_err = _FC * float(np.trapezoid(
            ERR * lambda_vals * conditional, ages
        ))
        reid_ear = _FC * float(np.trapezoid(EAR_rate * conditional, ages))
        samples[i] = reid_err + reid_ear

    samples = np.maximum(samples, 0.0)
    thr = HISTORICAL_NASA_REID_THRESHOLD
    return {
        'median': float(np.median(samples)),
        'p5': float(np.percentile(samples, 5)),
        'p95': float(np.percentile(samples, 95)),
        'mean': float(np.mean(samples)),
        'std': float(np.std(samples)),
        'samples': samples,
        'nasa_limit': thr,
        'exceeds_limit_fraction': float(np.mean(samples > thr)),
    }


def decision_threshold_sweep(
    H_total_Sv: float,
    age_at_exposure: int,
    sex: str,
    precision_scales: np.ndarray | list[float] | None = None,
    n_samples: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Sweep biological prior width and report REID CI vs. the 3% threshold.

    All four biological priors are scaled by a single ``precision_scale``
    factor s:

    - ``Q_sigma`` = ln(2) × s
    - ``DDREF_high - 1`` = 1.0 × s   (so s=1 gives Paper 1's (1, 2), s=2
      gives a widened (1, 3) BEIR VII-style window)
    - ``ERR_sigma`` = 0.30 × s
    - ``EAR_sigma`` = 0.35 × s

    s < 1 corresponds to hypothetically tighter biological knowledge
    (e.g. s = 0.5 halves all four widths). s = 0 is a deterministic
    biology, retained as the point-estimate limit.

    The precision-vs-CI curve produced here answers the plan's question:
    *"how much tighter must biology be before REID CI reliably lies
    below the historical 3% REID benchmark?"*

    Parameters
    ----------
    H_total_Sv
        Baseline total dose equivalent in Sv (from Paper 1's
        ``integrate_mission_dose``).
    age_at_exposure, sex
        Crew demographics.
    precision_scales
        Iterable of positive scale factors. Default:
        ``[0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]``.
    n_samples
        MC samples per sweep point.
    seed
        Base RNG seed; each sweep point uses ``seed + i``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``precision_scale``, ``Q_sigma``, ``DDREF_high``,
        ``ERR_sigma``, ``EAR_sigma``, ``REID_median``, ``REID_p5``,
        ``REID_p95``, ``exceeds_limit_fraction``.
    """
    if precision_scales is None:
        precision_scales = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    rows = []
    for i, s in enumerate(precision_scales):
        s = float(s)
        if s < 0:
            raise ValueError(f"precision_scales must be non-negative; got {s}")
        Q_sigma = float(np.log(2.0)) * s
        DDREF_high = 1.0 + 1.0 * s
        ERR_sigma = 0.30 * s
        EAR_sigma = 0.35 * s

        result = reid_with_custom_biological_priors(
            H_total_Sv, age_at_exposure, sex,
            n_samples=n_samples,
            Q_sigma=Q_sigma,
            DDREF_low=1.0, DDREF_high=DDREF_high,
            ERR_sigma=ERR_sigma, EAR_sigma=EAR_sigma,
            seed=seed + i,
        )
        rows.append({
            'precision_scale': s,
            'Q_sigma': Q_sigma,
            'DDREF_high': DDREF_high,
            'ERR_sigma': ERR_sigma,
            'EAR_sigma': EAR_sigma,
            'REID_median': result['median'],
            'REID_p5': result['p5'],
            'REID_p95': result['p95'],
            'exceeds_limit_fraction': result['exceeds_limit_fraction'],
        })

    return pd.DataFrame(rows)
