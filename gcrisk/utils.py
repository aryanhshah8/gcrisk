"""
gcrisk/utils.py — Shared constants, unit converters, and I/O helpers.

Physical constants, ion species data, tissue weighting factors,
and shielding material properties used across the pipeline.
"""

import numpy as np

DEFAULT_E_GRID = np.logspace(1, 5, 200)

# Physical constants
CONSTANTS = {
    'mp_c2': 938.272,        # proton rest mass energy, MeV
    'me_c2': 0.511,          # electron rest mass energy, MeV
    'e_charge': 1.602e-19,   # elementary charge, C
    'NA': 6.022e23,          # Avogadro's number
    'r0_fm': 1.35,           # nuclear radius constant, fm
    'barn_cm2': 1e-24,       # 1 barn in cm²
    'AU_km': 1.496e8,        # 1 AU in km
    'GM_sun': 1.327e11,      # solar gravitational parameter, km³/s²
    'c_km_s': 2.998e5,       # speed of light, km/s
}

# Ion species registry
ION_SPECIES = {
    'H':  {'Z': 1,  'A': 1,  'abundance': 1.0000, 'name': 'Proton'},
    'He': {'Z': 2,  'A': 4,  'abundance': 0.0960, 'name': 'Helium-4'},
    'C':  {'Z': 6,  'A': 12, 'abundance': 0.0046, 'name': 'Carbon-12'},
    'O':  {'Z': 8,  'A': 16, 'abundance': 0.0038, 'name': 'Oxygen-16'},
    'Si': {'Z': 14, 'A': 28, 'abundance': 0.0008, 'name': 'Silicon-28'},
    'Fe': {'Z': 26, 'A': 56, 'abundance': 0.0018, 'name': 'Iron-56'},
}

# ICRP-60 tissue weighting factors
TISSUE_WEIGHTS = {
    'bone_marrow': 0.12,
    'colon':       0.12,
    'lung':        0.12,
    'stomach':     0.12,
    'breast':      0.12,
    'gonads':      0.08,
    'bladder':     0.04,
    'liver':       0.04,
    'thyroid':     0.04,
    'skin':        0.01,
    'bone_surface': 0.01,
    'brain':       0.01,
    'remainder':   0.12,
}

# Shielding material properties
MATERIALS = {
    'aluminum':     {'density': 2.699, 'I_eV': 166.0, 'formula': 'Al'},
    'water':        {'density': 1.000, 'I_eV':  75.0, 'formula': 'H2O'},
    'polyethylene': {'density': 0.970, 'I_eV':  57.4, 'formula': 'CH2'},
    'tissue':       {'density': 1.000, 'I_eV':  75.0, 'formula': 'tissue'},
}

# Element data for compound materials
ELEMENT_DATA = {
    'H':  {'Z': 1,  'A': 1.008},
    'C':  {'Z': 6,  'A': 12.011},
    'N':  {'Z': 7,  'A': 14.007},
    'O':  {'Z': 8,  'A': 15.999},
    'Al': {'Z': 13, 'A': 26.982},
    'Si': {'Z': 14, 'A': 28.086},
    'Fe': {'Z': 26, 'A': 55.845},
}

# Material composition: element -> mass fraction
MATERIAL_COMPOSITION = {
    'aluminum':     {'Al': 1.0},
    'water':        {'H': 2 * 1.008 / 18.015, 'O': 15.999 / 18.015},
    'polyethylene': {'H': 2 * 1.008 / 14.027, 'C': 12.011 / 14.027},
    'tissue':       {'H': 2 * 1.008 / 18.015, 'O': 15.999 / 18.015},
}


# Utility functions

def lorentz_gamma(KE_MeV: float, m_c2_MeV: float) -> float:
    """Lorentz factor gamma = (KE + m*c^2) / (m*c^2)."""
    return (KE_MeV + m_c2_MeV) / m_c2_MeV


def beta(KE_MeV: float, m_c2_MeV: float) -> float:
    """Velocity beta = v/c from kinetic energy and rest mass."""
    gamma = lorentz_gamma(KE_MeV, m_c2_MeV)
    return np.sqrt(1.0 - 1.0 / gamma**2)


def rigidity_GV(KE_MeV_per_n: float, Z: int, A: int) -> float:
    """Magnetic rigidity R = pc / (Ze) in GV."""
    mp_c2 = CONSTANTS['mp_c2']
    KE_total = KE_MeV_per_n * A
    m_c2 = A * mp_c2
    E_total = KE_total + m_c2
    pc = np.sqrt(E_total**2 - m_c2**2)
    return pc / (Z * 1000.0)


def MeV_to_Joule(MeV: float) -> float:
    """Convert energy from MeV to Joules."""
    return MeV * 1.602e-13


def gcm2_to_cm(areal_density: float, material_key: str) -> float:
    """Convert areal density (g/cm²) to linear thickness (cm)."""
    rho = MATERIALS[material_key]['density']
    return areal_density / rho
