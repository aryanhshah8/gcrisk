"""Tests for gcrisk/transport.py"""

import numpy as np
import pytest
from gcrisk.transport import (
    effective_charge, load_stopping_power, bethe_bloch, proton_range,
    energy_after_slab, bradt_peters_cross_section,
)


def test_proton_range_water():
    """200 MeV proton range in water: ~26 g/cm^2 (NIST), accept within 10%."""
    R = proton_range(np.array([200.0]), 'water')
    assert 23.3 < R[0] < 28.6, f"200 MeV proton range in water: {R[0]:.1f} g/cm²"


def test_stopping_power_uses_effective_charge():
    """Heavy-ion stopping should be below bare Z^2 scaling at finite beta."""
    E = np.array([500.0])  # MeV/nucleon
    S_p = np.atleast_1d(load_stopping_power('water')(E))
    S_Fe = np.atleast_1d(bethe_bloch(E, Z_ion=26, material='water'))
    ratio = float(S_Fe[0] / S_p[0])
    assert 100.0 < ratio < 676.0, f"Fe/proton stopping ratio: {ratio:.1f}"


def test_effective_charge_increases_with_energy():
    """Heavy-ion effective charge should rise toward the bare charge with energy."""
    z_low = float(effective_charge(np.array([10.0]), 26)[0])
    z_mid = float(effective_charge(np.array([500.0]), 26)[0])
    z_high = float(effective_charge(np.array([5000.0]), 26)[0])
    assert 1.0 <= z_low < z_mid < z_high <= 26.0


def test_slab_stops_low_energy():
    """10 MeV proton through 5 g/cm^2 aluminum: must return E_out = 0."""
    E_out = energy_after_slab(10.0, 5.0, 'aluminum')
    assert E_out == 0.0, f"10 MeV proton should stop in 5 g/cm² Al, got E={E_out}"


def test_no_energy_gain():
    """E_out must never exceed E_in."""
    E_in = 500.0
    for material in ['water', 'aluminum', 'polyethylene']:
        for x in [1.0, 5.0, 10.0, 20.0]:
            E_out = energy_after_slab(E_in, x, material)
            assert E_out <= E_in, (
                f"Energy gain! E_in={E_in}, E_out={E_out}, "
                f"material={material}, x={x}")


def test_bradt_peters_cross_section():
    """Cross section should be positive and in reasonable range."""
    sigma = bradt_peters_cross_section(1, 1, 13, 27)  # p + Al
    assert 1e-25 < sigma < 1e-23, f"p+Al cross section: {sigma:.3e} cm²"


def test_thicker_slab_more_energy_loss():
    """More shielding should give lower exit energy."""
    E_in = 500.0
    E_out_thin = energy_after_slab(E_in, 5.0, 'aluminum')
    E_out_thick = energy_after_slab(E_in, 20.0, 'aluminum')
    assert E_out_thin > E_out_thick, "Thicker slab should give lower exit energy"
