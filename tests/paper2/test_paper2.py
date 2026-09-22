"""Tests for gcrisk/paper2.py — all additive layer, no Paper 1 changes.

Fast tests only (no full-pipeline ensemble runs) unless marked ``slow``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from gcrisk.paper2 import (
    BEIR_VII_DDREF_PRIOR,
    BIOLOGICAL_PARAM_NAMES,
    DEFAULT_LET_BINS_KEV_UM,
    HISTORICAL_NASA_REID_THRESHOLD,
    PAPER1_PRIORS,
    conditioned_let_sensitivity,
    decision_threshold_sweep,
    let_binned_dose_contribution,
    lhs_samples_with_priors,
    reid_with_custom_biological_priors,
    run_ensemble_with_priors,
)
from gcrisk.spectrum import gcr_total_flux
from gcrisk.uncertainty import PARAM_NAMES, lhs_samples
from gcrisk.utils import DEFAULT_E_GRID


# ---------------------------------------------------------------------------
# lhs_samples_with_priors
# ---------------------------------------------------------------------------

def test_sampler_no_overrides_matches_paper1_distributions():
    """No overrides → all params within Paper 1's prior support."""
    s = lhs_samples_with_priors(500, prior_overrides=None, seed=3)
    assert set(s.keys()) == set(PARAM_NAMES)
    assert len(s['DDREF']) == 500
    # Paper 1 DDREF is uniform(1, 2).
    assert s['DDREF'].min() >= 1.0
    assert s['DDREF'].max() <= 2.0
    # Lognormal params are strictly positive.
    for p in ('phi_scale', 'LIS_norm', 'cross_sec', 'neutron_H', 'Q_factor'):
        assert np.all(s[p] > 0)


def test_sampler_widened_ddref_respects_bounds():
    s = lhs_samples_with_priors(
        500, prior_overrides={'DDREF': BEIR_VII_DDREF_PRIOR}, seed=4,
    )
    assert s['DDREF'].min() >= 1.0
    assert s['DDREF'].max() <= 3.0
    # Widened prior has higher mean than Paper 1's (1, 2).
    assert np.mean(s['DDREF']) > 1.7


def test_sampler_override_only_affects_named_param():
    """Overriding DDREF leaves the other seven params' distributions alone."""
    s_base = lhs_samples_with_priors(500, prior_overrides=None, seed=5)
    s_wide = lhs_samples_with_priors(
        500, prior_overrides={'DDREF': BEIR_VII_DDREF_PRIOR}, seed=5,
    )
    # Same seed + same LHS dimensionality ⇒ identical quantiles for unchanged
    # priors; so the marginal samples for non-DDREF params should be
    # pairwise identical.
    for p in PARAM_NAMES:
        if p == 'DDREF':
            continue
        np.testing.assert_array_equal(s_base[p], s_wide[p])


def test_sampler_paper1_priors_constants_match_uncertainty_module():
    """PAPER1_PRIORS constants match the distributions hardcoded in lhs_samples."""
    s_ref = lhs_samples(2000, seed=9)
    s_new = lhs_samples_with_priors(2000, prior_overrides=None, seed=9)
    # Same LHS quantile grid ⇒ exact match (same seed, same mapping).
    for p in PARAM_NAMES:
        np.testing.assert_allclose(s_ref[p], s_new[p], rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# let_binned_dose_contribution
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def unshielded_flux():
    """Mission-mean unshielded GCR flux at a representative phi."""
    return gcr_total_flux(DEFAULT_E_GRID, phi_MV=500.0)


def test_let_binning_fractions_sum_to_one(unshielded_flux):
    binned = let_binned_dose_contribution(
        unshielded_flux, DEFAULT_E_GRID,
        material='tissue',
        bin_edges_keV_um=DEFAULT_LET_BINS_KEV_UM,
    )
    assert abs(binned.dose_fraction_by_bin.sum() - 1.0) < 1e-6
    assert abs(binned.H_fraction_by_bin.sum() - 1.0) < 1e-6


def test_let_binning_edges_are_monotonic(unshielded_flux):
    with pytest.raises(ValueError):
        let_binned_dose_contribution(
            unshielded_flux, DEFAULT_E_GRID,
            bin_edges_keV_um=(10.0, 5.0, 20.0),   # non-monotone
        )
    with pytest.raises(ValueError):
        let_binned_dose_contribution(
            unshielded_flux, DEFAULT_E_GRID,
            bin_edges_keV_um=(0.0,),               # 0 bins
        )


def test_let_binning_H_tracks_quality_factor(unshielded_flux):
    """Low-LET bin has Q(L)=1, so for that bin H/D ratio should equal 1 (tissue).

    Equivalently, total H = total D in the low-LET bin if no neutrons present
    in the flux.
    """
    flux_no_n = {k: v for k, v in unshielded_flux.items() if k != 'neutron'}
    binned = let_binned_dose_contribution(
        flux_no_n, DEFAULT_E_GRID, material='tissue',
        bin_edges_keV_um=DEFAULT_LET_BINS_KEV_UM,
    )
    # First bin is <3 keV/µm → Q=1 exactly.
    D0 = binned.dose_by_bin[0]
    H0 = binned.H_by_bin[0]
    if D0 > 0:
        np.testing.assert_allclose(H0, D0, rtol=1e-6)


# ---------------------------------------------------------------------------
# conditioned_let_sensitivity
# ---------------------------------------------------------------------------

def _fake_ensemble(n: int, seed: int = 0) -> dict:
    """Build a synthetic ensemble result for attribution tests."""
    rng = np.random.default_rng(seed)
    param_samples = {
        name: rng.uniform(0.5, 1.5, size=n) for name in PARAM_NAMES
    }
    # REID depends strongly on Q_factor (slope 1) + a little noise.
    REID = (
        0.03 * param_samples['Q_factor']
        + 0.002 * rng.standard_normal(n)
    )
    return {
        'param_samples': param_samples,
        'REID_samples': REID,
        'H_mSv_samples': REID * 1000.0,
        'D_mGy_samples': np.full(n, 500.0),
    }


def test_attribution_shape_and_weights(unshielded_flux):
    ens = _fake_ensemble(500)
    binned = let_binned_dose_contribution(
        unshielded_flux, DEFAULT_E_GRID,
        bin_edges_keV_um=DEFAULT_LET_BINS_KEV_UM,
    )
    df = conditioned_let_sensitivity(ens, binned, output='REID')
    # One row per parameter, one column per LET bin.
    assert df.shape[0] == len(PARAM_NAMES)
    assert df.shape[1] == len(DEFAULT_LET_BINS_KEV_UM) - 1
    # Row sums equal Spearman ρ² × sum(H_fraction) == ρ² (weights sum to 1).
    row_sums = df.sum(axis=1).values
    # Q_factor strongly drives REID → attribution row should sum to ~1.
    assert row_sums[PARAM_NAMES.index('Q_factor')] > 0.5


def test_attribution_missing_output_key_raises(unshielded_flux):
    ens = _fake_ensemble(100)
    del ens['REID_samples']
    binned = let_binned_dose_contribution(
        unshielded_flux, DEFAULT_E_GRID,
    )
    with pytest.raises(KeyError):
        conditioned_let_sensitivity(ens, binned, output='REID')


# ---------------------------------------------------------------------------
# reid_with_custom_biological_priors / decision_threshold_sweep
# ---------------------------------------------------------------------------

def test_custom_priors_reproduce_paper1_defaults():
    """Default args must match Paper 1's reid_with_uncertainty distributions.

    Cannot assert sample-by-sample equality because the NumPy RNG call
    ordering differs, but the aggregate statistics should be extremely close.
    """
    from gcrisk.reid import reid_with_uncertainty
    out_paper1 = reid_with_uncertainty(
        H_total_Sv=1.0, age_at_exposure=35, sex='male', n_samples=5000,
    )
    out_new = reid_with_custom_biological_priors(
        H_total_Sv=1.0, age_at_exposure=35, sex='male', n_samples=5000,
        seed=42,
    )
    # Medians should agree to within a few percent at N=5000.
    assert abs(out_paper1['median'] - out_new['median']) < 0.003


def test_decision_threshold_sweep_shape_and_monotonicity():
    sweep = decision_threshold_sweep(
        H_total_Sv=1.0, age_at_exposure=35, sex='male',
        precision_scales=[0.1, 0.5, 1.0, 2.0],
        n_samples=1500,
    )
    assert list(sweep.columns) == [
        'precision_scale', 'Q_sigma', 'DDREF_high',
        'ERR_sigma', 'EAR_sigma',
        'REID_median', 'REID_p5', 'REID_p95',
        'exceeds_limit_fraction',
    ]
    assert len(sweep) == 4
    # CI width is monotonically non-decreasing with precision scale.
    widths = (sweep['REID_p95'] - sweep['REID_p5']).values
    # Allow mild MC jitter at N=1500 but trend must be increasing overall.
    assert widths[-1] > widths[0]


def test_decision_threshold_reports_3pct_benchmark():
    sweep = decision_threshold_sweep(
        H_total_Sv=0.5, age_at_exposure=35, sex='male',
        precision_scales=[1.0], n_samples=500,
    )
    frac = sweep['exceeds_limit_fraction'].iloc[0]
    assert 0.0 <= frac <= 1.0
    # The threshold constant is the documented historical benchmark.
    assert HISTORICAL_NASA_REID_THRESHOLD == 0.03


# ---------------------------------------------------------------------------
# End-to-end (slow): one pipeline evaluation via run_ensemble_with_priors
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_run_ensemble_with_priors_smoke():
    """One-sample smoke test — confirms the ensemble wrapper wires correctly."""
    from gcrisk.spectrum import load_usoskin_phi
    from gcrisk.trajectory import generate_trajectory

    data_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'usoskin',
    )
    phi_df = load_usoskin_phi(os.path.join(data_dir, 'phi_transit_frozen.csv'))
    traj = generate_trajectory('2011-11-26', phi_df=phi_df)

    res = run_ensemble_with_priors(
        traj, shielding_x=16.0, material='aluminum',
        age=35, sex='male',
        prior_overrides={'DDREF': BEIR_VII_DDREF_PRIOR},
        n_samples=3,
        phi_df=phi_df,
        seed=0,
    )
    assert res['n_samples'] == 3
    assert 'variance_decomposition' in res
    assert res['prior_overrides'] == {'DDREF': BEIR_VII_DDREF_PRIOR}
