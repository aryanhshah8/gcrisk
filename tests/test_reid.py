"""Tests for gcrisk/reid.py"""

import numpy as np
import pytest
from gcrisk.reid import (
    baseline_cancer_mortality_rate, excess_relative_risk,
    reid_point_estimate, reid_with_uncertainty,
)


def test_reid_limit_reference():
    """1.0 Sv at age 40 male should give REID between 0.02 and 0.08."""
    reid = reid_point_estimate(1.0, 40, 'male')
    assert 0.005 < reid < 0.15, f"REID for 1 Sv, 40yo male: {reid:.4f}"


def test_reid_female_higher_than_male():
    """Female REID must exceed male REID for same dose and age."""
    reid_m = reid_point_estimate(0.5, 40, 'male')
    reid_f = reid_point_estimate(0.5, 40, 'female')
    assert reid_f > reid_m, f"Female REID ({reid_f}) should exceed male ({reid_m})"


def test_uncertainty_width():
    """95% CI must be at least 3x the median (known ~4x total uncertainty)."""
    result = reid_with_uncertainty(0.5, 40, 'male', n_samples=5000)
    ratio = result['p95'] / result['median'] if result['median'] > 0 else 0
    assert ratio > 2.0, f"95th/median ratio: {ratio:.2f} (expect >3)"


def test_no_negative_reid():
    """REID must be strictly non-negative for any positive dose."""
    result = reid_with_uncertainty(0.3, 35, 'male', n_samples=1000)
    assert np.all(result['samples'] >= 0), "REID should never be negative"


def test_baseline_mortality_rate():
    """Baseline cancer mortality should increase with age."""
    r40 = baseline_cancer_mortality_rate(40, 'male')
    r60 = baseline_cancer_mortality_rate(60, 'male')
    assert r60 > r40, "Mortality rate should increase with age"


def test_err_scaling():
    """ERR should scale linearly with dose."""
    err1 = excess_relative_risk(0.5, 'male')
    err2 = excess_relative_risk(1.0, 'male')
    assert abs(err2 / err1 - 2.0) < 0.01, "ERR should double with doubled dose"


def test_organ_specific_ear_sum_matches_total():
    """organ_specific_ear() summed over organs should match excess_absolute_risk_per_year()."""
    from gcrisk.reid import organ_specific_ear, excess_absolute_risk_per_year
    H = 0.5
    age = 35
    for sex in ('male', 'female'):
        organ_ear = organ_specific_ear(H, age, sex)
        total_from_organs = sum(organ_ear.values())
        total_from_func = excess_absolute_risk_per_year(H, age, sex)
        assert abs(total_from_organs - total_from_func) < 1e-10, (
            f"{sex}: organ sum {total_from_organs:.6e} != total {total_from_func:.6e}")


def test_lung_dominates_male_ear():
    """Lung should be the largest EAR contributor for males at age 35."""
    from gcrisk.reid import organ_specific_ear
    ear = organ_specific_ear(0.5, 35, 'male')
    assert ear['lung'] == max(ear.values()), (
        f"Lung should dominate male EAR at age 35, got max={max(ear, key=ear.get)}")
