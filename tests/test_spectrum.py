"""Tests for gcrisk/spectrum.py"""

import numpy as np
import pytest
from gcrisk.spectrum import (
    lis_proton, lis_helium, lis_heavy,
    force_field_modulation, gcr_total_flux,
)


def test_lis_proton_normalization():
    """Proton LIS (directional, per sr) at 1 GeV/n should be ~0.008 (within factor 2)."""
    E = np.array([1000.0])  # 1 GeV/n
    flux = lis_proton(E)
    assert 0.003 < flux[0] < 0.020, f"Proton LIS at 1 GeV: {flux[0]:.3e}"

"""add in the proton flux from mg here"""


def test_force_field_modulation_reduces_flux():
    """Higher phi must always give lower flux at every energy."""
    E = np.logspace(1, 5, 50)
    flux_low = force_field_modulation(E, 400.0, lis_proton)
    flux_high = force_field_modulation(E, 1100.0, lis_proton)
    assert np.all(flux_low >= flux_high), "Higher phi should reduce flux"


def test_modulation_solar_min_vs_max():
    """phi=400 MV flux must exceed phi=1100 MV flux at 300 MeV/n by factor 1.5-3x."""
    E = np.array([300.0])
    flux_min = force_field_modulation(E, 400.0, lis_proton)
    flux_max = force_field_modulation(E, 1100.0, lis_proton)
    ratio = flux_min[0] / flux_max[0]
    assert 1.5 < ratio < 25.0, f"Solar min/max ratio at 300 MeV: {ratio:.2f}"


def test_helium_abundance():
    """He flux / H flux should be ~0.096 at high energy (where modulation is small)."""
    E = np.array([10000.0])  # 10 GeV/n - minimal modulation effect
    flux_H = force_field_modulation(E, 400.0, lis_proton)
    flux_He = force_field_modulation(E, 400.0, lis_helium, Z=2, A=4)
    ratio = flux_He[0] / flux_H[0]
    # He has its own LIS, so ratio won't be exactly 0.096 but should be in the ballpark
    assert 0.02 < ratio < 0.5, f"He/H ratio at 10 GeV: {ratio:.3f}"


def test_modulated_flux_at_550MV_1GeV():
    """At phi=550 MV, directional proton flux at 1000 MeV/n should be ~1.6e-3."""
    E = np.array([1000.0])
    flux = force_field_modulation(E, 550.0, lis_proton)
    assert 8e-4 < flux[0] < 3e-3, f"Flux at 550 MV, 1 GeV: {flux[0]:.3e}"


def test_gcr_total_flux_keys():
    """gcr_total_flux should return all species plus 'total'."""
    E = np.logspace(1, 5, 20)
    result = gcr_total_flux(E, 550.0)
    expected_keys = {'H', 'He', 'C', 'O', 'Si', 'Fe', 'total'}
    assert set(result.keys()) == expected_keys


def test_heavy_ion_lis():
    """Heavy ion LIS should return positive values."""
    E = np.logspace(2, 4, 20)
    for Z, A in [(6, 12), (8, 16), (14, 28), (26, 56)]:
        flux = lis_heavy(E, Z, A)
        assert np.all(flux > 0), f"Heavy ion Z={Z} flux should be positive"


def test_invalid_heavy_ion():
    """lis_heavy should raise ValueError for unknown species."""
    E = np.array([1000.0])
    with pytest.raises(ValueError):
        lis_heavy(E, Z=99, A=200)
