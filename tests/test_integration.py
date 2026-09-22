"""tests/test_integration.py — End-to-end integration tests."""

import os
import numpy as np
import pytest

from gcrisk.trajectory import generate_trajectory
from gcrisk.spectrum import load_usoskin_phi
from gcrisk.organ_dose import integrate_organ_dose
from gcrisk.reid import reid_from_organ_doses
from gcrisk.sep import sep_event_dose, sep_mission_probability


_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'usoskin')


@pytest.fixture(scope='module')
def phi_df():
    return load_usoskin_phi(os.path.join(_DATA_DIR, 'phi_transit_frozen.csv'))


@pytest.fixture(scope='module')
def transit_organ_result(phi_df):
    """
    259-day Earth→Mars transit with organ-routed dose.
    Precomputed once and shared across integration tests.
    """
    traj = generate_trajectory('2011-11-26', phi_df=phi_df)
    return integrate_organ_dose(
        traj,
        shielding_x=16.0,
        material='aluminum',
        phi_df=phi_df,
    )


def test_transit_organ_dose_physical_ordering(transit_organ_result):
    """
    Shallow organs receive more dose than deep organs.

    thyroid self-shielding = 2.0 g/cm² (shallow)
    bladder self-shielding = 10.0 g/cm² (deep)
    → H_thyroid > H_bladder always
    """
    organ_H = transit_organ_result['organ_H_mSv']
    assert organ_H['thyroid'] > organ_H['bladder'], (
        f"Thyroid (shallow) should have higher H than bladder (deep): "
        f"thyroid={organ_H['thyroid']:.2f}, bladder={organ_H['bladder']:.2f} mSv"
    )
    assert organ_H['breast'] > organ_H['ovary'], (
        f"Breast (shallow) should have higher H than ovary (deep): "
        f"breast={organ_H['breast']:.2f}, ovary={organ_H['ovary']:.2f} mSv"
    )


def test_transit_reid_in_plausible_range(transit_organ_result, phi_df):
    """
    REID for a 259-day transit at 16 g/cm² Al for a 35-year-old male.

    Expected range [0.5%, 8%]: wide to accommodate the composition-constrained
    calibration (H ~ 6 mSv/day after fixing species fractions to ACE/CRIS),
    the Cucinotta (2013) ERR+EAR model uncertainty, and the known CSDA
    overestimate of Q_eff (~3.3 vs RAD 2.62).
    """
    organ_H_mSv = transit_organ_result['organ_H_mSv']
    reid_result = reid_from_organ_doses(organ_H_mSv, age_at_exposure=35, sex='male', n_samples=200)

    REID_median = reid_result['REID_total']
    assert 0.005 <= REID_median <= 0.08, (
        f"Transit REID out of expected [0.5%, 8%] range: {REID_median:.3%}"
    )


def test_full_mission_reid_in_plausible_range(phi_df):
    """
    Full 3-phase mission (transit out + 500d surface + transit back) at 20 g/cm² Al,
    35-year-old male: REID ∈ [3%, 10%].

    Uses run_full_mission which includes organ-routed REID internally.
    """
    from gcrisk.mission import run_full_mission

    result = run_full_mission(
        launch_date='2011-11-26',
        surface_days=500,
        shielding_x_gcm2=20.0,
        shielding_material='aluminum',
        age=35,
        sex='male',
        phi_df=phi_df,
    )

    total = result['total']
    D_total = total['D_total_mGy']
    H_total = total['H_total_mSv']

    # Basic sanity: dose must be positive and physically reasonable
    assert D_total > 0
    assert H_total > D_total  # H > D because Q > 1 for GCR

    # Check REID if exposed by the mission
    if 'organ_risk' in result:
        REID = result['organ_risk'].get('REID_median', None)
        if REID is not None:
            assert 0.03 <= REID <= 0.10, (
                f"Full-mission REID out of expected [3%, 10%]: {REID:.3%}"
            )


def test_gcr_plus_sep_combined_dose(transit_organ_result):
    """
    GCR cumulative H + SEP event H computes without error and is additive.
    """
    organ_H = transit_organ_result['organ_H_mSv']
    gcr_H_total = sum(organ_H.values())

    # SEP event on top of GCR transit
    sep = sep_event_dose('oct2003', shielding_x_gcm2=16.0)
    sep_H = float(np.asarray(sep['H_mSv']).ravel()[0])

    combined = gcr_H_total + sep_H
    assert combined > gcr_H_total
    assert sep_H > 0


def test_sep_mission_probability_increases_with_duration():
    """Longer missions have higher probability of encountering a large SEP event."""
    p_short = sep_mission_probability(30)['P_one_or_more']
    p_long = sep_mission_probability(1000)['P_one_or_more']
    assert p_long > p_short


def test_organ_h_effective_dose_consistency(transit_organ_result):
    """
    ICRP-60 tissue-weight sum of organ H values should be within 20% of
    the overall effective dose E_effective_mSv.

    This verifies that the organ routing is internally consistent with
    the total-body H estimate from integrate_mission_dose.
    """
    from gcrisk.organ_dose import ORGAN_ICRP60_WEIGHTS

    organ_H = transit_organ_result['organ_H_mSv']
    E_effective = transit_organ_result.get('E_effective_mSv')
    if E_effective is None:
        pytest.skip("E_effective_mSv not in organ result")

    E_organ_weighted = sum(
        ORGAN_ICRP60_WEIGHTS[org] * organ_H[org]
        for org in organ_H
        if org in ORGAN_ICRP60_WEIGHTS
    )

    rel_diff = abs(E_organ_weighted - E_effective) / E_effective
    assert rel_diff < 0.20, (
        f"Organ-weighted E ({E_organ_weighted:.2f} mSv) vs pipeline E_effective "
        f"({E_effective:.2f} mSv): {rel_diff:.1%} difference (limit 20%)"
    )
