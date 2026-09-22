"""Tests for gcrisk/uncertainty.py — LHS ensemble and variance decomposition.

All tests in this file are marked @pytest.mark.slow because they run the
full trajectory integration.  They are excluded from CI by default (see pytest.ini).

Run locally:
    pytest -m slow tests/test_uncertainty.py -v
"""

import os
import numpy as np
import pytest

from gcrisk.uncertainty import lhs_samples, run_uncertainty_ensemble, PARAM_NAMES
from gcrisk.trajectory import generate_trajectory
from gcrisk.spectrum import load_usoskin_phi


_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'usoskin')
_N_SAMPLES = int(os.environ.get('UNCERTAINTY_SAMPLES', 20))


@pytest.fixture(scope='module')
def phi_df():
    return load_usoskin_phi(os.path.join(_DATA_DIR, 'phi_transit_frozen.csv'))


@pytest.fixture(scope='module')
def traj_df(phi_df):
    return generate_trajectory('2011-11-26', phi_df=phi_df)


# ---------------------------------------------------------------------------
# LHS sampling tests (fast — no pipeline integration)
# ---------------------------------------------------------------------------

def test_lhs_samples_shape():
    """LHS returns correct shape and number of parameters."""
    s = lhs_samples(50, seed=0)
    assert set(s.keys()) == set(PARAM_NAMES)
    for v in s.values():
        assert len(v) == 50


def test_lhs_samples_distributions():
    """Check approximate distribution properties of LHS samples."""
    s = lhs_samples(2000, seed=1)

    # phi_scale and LIS_norm are lognormal → median should be near 1.0
    assert abs(np.median(s['phi_scale']) - 1.0) < 0.05
    assert abs(np.median(s['LIS_norm']) - 1.0) < 0.05

    # DDREF is uniform(1, 2) → mean ~1.5
    assert abs(np.mean(s['DDREF']) - 1.5) < 0.1

    # ERR_scale is normal(1, 0.30) → mean ~1.0
    assert abs(np.mean(s['ERR_scale']) - 1.0) < 0.05

    # All samples are positive where physically required
    for pname in ('phi_scale', 'LIS_norm', 'cross_sec', 'neutron_H', 'Q_factor'):
        assert np.all(s[pname] > 0), f"{pname} has non-positive samples"

    # DDREF bounded
    assert np.all(s['DDREF'] >= 1.0)
    assert np.all(s['DDREF'] <= 2.0)


# ---------------------------------------------------------------------------
# Ensemble run tests (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_ensemble_returns_correct_structure(traj_df, phi_df):
    """Ensemble returns dict with all required keys."""
    result = run_uncertainty_ensemble(
        traj_df, shielding_x=16.0, material='aluminum',
        age=35, sex='male', n_samples=_N_SAMPLES, phi_df=phi_df, seed=42,
    )

    assert 'D_mGy_samples' in result
    assert 'H_mSv_samples' in result
    assert 'REID_samples' in result
    assert 'percentiles' in result
    assert 'variance_decomposition' in result
    assert result['n_samples'] == _N_SAMPLES

    assert len(result['D_mGy_samples']) == _N_SAMPLES
    assert len(result['REID_samples']) == _N_SAMPLES


@pytest.mark.slow
def test_ensemble_percentile_ordering(traj_df, phi_df):
    """p5 < p50 < p95 for D, H, REID outputs."""
    result = run_uncertainty_ensemble(
        traj_df, shielding_x=16.0, material='aluminum',
        age=35, sex='male', n_samples=_N_SAMPLES, phi_df=phi_df, seed=7,
    )

    for key in ('D_mGy', 'H_mSv', 'REID'):
        p = result['percentiles'][key]
        assert p['p5'] <= p['p50'] <= p['p95'], (
            f"Percentile ordering violated for {key}: "
            f"p5={p['p5']:.4f}, p50={p['p50']:.4f}, p95={p['p95']:.4f}"
        )
        assert p['p5'] > 0, f"p5 for {key} should be positive"


@pytest.mark.slow
def test_ensemble_reid_range_plausible(traj_df, phi_df):
    """REID samples should be positive and plausibly in [0.1%, 20%]."""
    result = run_uncertainty_ensemble(
        traj_df, shielding_x=16.0, material='aluminum',
        age=35, sex='male', n_samples=_N_SAMPLES, phi_df=phi_df, seed=99,
    )
    R = result['REID_samples']
    valid = R[np.isfinite(R)]
    assert len(valid) > 0
    assert np.all(valid >= 0), "REID should be non-negative"
    # Median REID for a 259-day transit at 16 g/cm² Al should be in a plausible range
    # (the full mission including surface is much higher)
    assert np.median(valid) < 0.20, f"Median REID seems too high: {np.median(valid):.3%}"


@pytest.mark.slow
def test_variance_decomposition_sums_to_one(traj_df, phi_df):
    """Variance decomposition fractions sum to ~1.0 for each output."""
    result = run_uncertainty_ensemble(
        traj_df, shielding_x=16.0, material='aluminum',
        age=35, sex='male', n_samples=_N_SAMPLES, phi_df=phi_df, seed=11,
    )
    vd = result['variance_decomposition']
    for out_name in ('D_mGy', 'H_mSv', 'REID'):
        total = sum(vd[out_name].values())
        assert abs(total - 1.0) < 0.05, (
            f"Variance fractions for {out_name} sum to {total:.3f}, expected ~1.0"
        )
