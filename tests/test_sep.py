"""Tests for gcrisk/sep.py — SEP event module."""

import numpy as np
import pytest

from gcrisk.sep import (
    SEP_EVENTS,
    sep_band_spectrum,
    sep_event_dose,
    sep_mission_probability,
    sep_dose_with_shielding_scan,
    BFO_30DAY_LIMIT_mGy,
)


def test_band_spectrum_positive():
    """Flux is non-negative at all energies."""
    E = np.logspace(0.5, 4, 100)
    ev = SEP_EVENTS['aug1972']
    flux = sep_band_spectrum(E, ev['C'], ev['gamma_low'], ev['gamma_high'], ev['E_break_MeV'])
    assert np.all(flux >= 0)


def test_band_spectrum_decreasing():
    """Flux decreases monotonically above the break energy."""
    ev = SEP_EVENTS['aug1972']
    E = np.logspace(np.log10(ev['E_break_MeV'] * 2), 4, 50)
    flux = sep_band_spectrum(E, ev['C'], ev['gamma_low'], ev['gamma_high'], ev['E_break_MeV'])
    assert np.all(np.diff(flux) <= 0), "Flux should decrease at energies well above break"


def test_sep_event_dose_unshielded_aug1972_exceeds_limit():
    """Aug 1972 event exceeds the 250 mGy BFO limit with no shielding."""
    result = sep_event_dose('aug1972', shielding_x_gcm2=0.0, material='aluminum')
    assert result['D_BFO_mGy'] > 0
    assert result['exceeds_acute_limit'], (
        f"Aug 1972 (unshielded) should exceed {BFO_30DAY_LIMIT_mGy} mGy BFO limit; "
        f"got {result['D_BFO_mGy']:.1f} mGy"
    )


def test_shielding_reduces_dose():
    """Heavier shielding reduces absorbed dose monotonically."""
    thicknesses = [0, 5, 10, 20]
    doses = [
        sep_event_dose('aug1972', x, 'aluminum')['D_BFO_mGy']
        for x in thicknesses
    ]
    for i in range(len(doses) - 1):
        assert doses[i] >= doses[i + 1], (
            f"Dose should decrease with shielding: x={thicknesses[i]} → {doses[i]:.1f}, "
            f"x={thicknesses[i+1]} → {doses[i+1]:.1f}"
        )


def test_mission_probability_sanity():
    """Poisson probabilities sum correctly; more time = more events."""
    r = sep_mission_probability(500.0, large_event_rate_per_year=0.8)
    assert 0.0 <= r['P_zero'] <= 1.0
    assert abs(r['P_zero'] + r['P_one_or_more'] - 1.0) < 1e-12
    assert r['lambda_mission'] == pytest.approx(0.8 * 500 / 365.25, rel=1e-6)

    r_short = sep_mission_probability(30.0)
    r_long = sep_mission_probability(500.0)
    assert r_long['P_one_or_more'] > r_short['P_one_or_more']


def test_shielding_scan_dataframe():
    """sep_dose_with_shielding_scan returns correctly shaped DataFrame."""
    df = sep_dose_with_shielding_scan('jan2005', thicknesses_gcm2=[0, 5, 16])
    assert list(df.columns) == ['x_gcm2', 'D_BFO_mGy', 'H_mSv', 'exceeds_acute_limit']
    assert len(df) == 3
    assert df['D_BFO_mGy'].iloc[0] >= df['D_BFO_mGy'].iloc[-1]


def test_all_events_defined():
    """All three canonical events are present and have required keys."""
    required = {'C', 'gamma_low', 'gamma_high', 'E_break_MeV', 'duration_s', 'description'}
    for key in ('aug1972', 'oct2003', 'jan2005'):
        assert key in SEP_EVENTS
        assert required.issubset(SEP_EVENTS[key].keys())
