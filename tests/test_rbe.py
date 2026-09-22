"""Tests for LET/RBE helpers."""

import numpy as np

from gcrisk.rbe import (
    dose_averaged_let,
    integrate_mission_let,
    mcnamara_rbe,
    rbe_weighted_dose,
    wedenberg_rbe,
)
from gcrisk.spectrum import gcr_total_flux
from gcrisk.trajectory import generate_trajectory


def test_wedenberg_is_unity_at_zero_let():
    """At LETd=0 the Wedenberg model should reduce to RBE=1."""
    assert abs(wedenberg_rbe(2.0, 0.0, 3.0) - 1.0) < 1e-8


def test_mcnamara_increases_with_let():
    """McNamara RBE should increase with LETd at fixed dose and alpha/beta."""
    low = mcnamara_rbe(2.0, 1.0, 3.0)
    high = mcnamara_rbe(2.0, 10.0, 3.0)
    assert high > low > 0.0


def test_wedenberg_lower_alpha_beta_gives_higher_rbe():
    """Lower alpha/beta tissues should show higher LET sensitivity."""
    low_ab = wedenberg_rbe(2.0, 5.0, 2.0)
    high_ab = wedenberg_rbe(2.0, 5.0, 10.0)
    assert low_ab > high_ab > 1.0


def test_dose_averaged_let_positive():
    """Mixed-field LETd from a modulated flux should be positive."""
    E = np.logspace(1, 5, 50)
    flux = gcr_total_flux(E, 550.0)
    letd = dose_averaged_let(flux, E)
    assert letd > 0.0


def test_mission_let_positive():
    """Mission-integrated proton LETd should be positive behind shielding."""
    traj = generate_trajectory('2011-11-26')
    result = integrate_mission_let(
        traj.iloc[:10],
        shielding_x_gcm2=16.0,
        shielding_material='aluminum',
        E_grid_MeV=np.logspace(1, 5, 40),
        species=['H'],
    )
    assert result['LETd_keV_um'] > 0.0


def test_rbe_weighted_dose_returns_consistent_payload():
    """RBE helper should return both RBE and weighted dose."""
    result = rbe_weighted_dose(2.0, 5.0, 3.0, model='wedenberg')
    assert result['RBE'] > 1.0
    assert abs(result['dose_RBE_Gy'] - result['RBE'] * result['dose_Gy']) < 1e-10


def test_wedenberg_matches_reference_case():
    """Reference case cross-checked against the OpenTOPAS-RBE equation set."""
    result = wedenberg_rbe(2.0, 5.0, 3.0)
    assert abs(result - 1.2865411854416302) < 1e-12


def test_mcnamara_matches_reference_case():
    """Reference case cross-checked against the OpenTOPAS-RBE equation set."""
    result = mcnamara_rbe(2.0, 5.0, 3.0)
    assert abs(result - 1.269537154977841) < 1e-12
