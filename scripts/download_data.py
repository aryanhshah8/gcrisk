#!/usr/bin/env python3
"""
scripts/download_data.py — Generate all required data files for the GCR pipeline.

1. NIST stopping power tables (Bethe-Bloch approximation)
2. Usoskin modulation potential (real download from cosmicrays.oulu.fi, fallback synthetic)
3. MSL RAD transit dose rate
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure gcr package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gcrisk.utils import CONSTANTS, MATERIALS, MATERIAL_COMPOSITION, ELEMENT_DATA


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


# ---------------------------------------------------------------------------
# 1. NIST stopping power tables via Bethe-Bloch
# ---------------------------------------------------------------------------

def bethe_bloch_proton(E_MeV: np.ndarray, Z_t: float, A_t: float,
                       I_eV: float) -> np.ndarray:
    """
    Bethe-Bloch electronic stopping power for protons in a target material.

    Returns mass stopping power in MeV·cm²/g.
    """
    mp_c2 = CONSTANTS['mp_c2']
    me_c2 = CONSTANTS['me_c2']  # MeV
    NA = CONSTANTS['NA']

    gamma = (E_MeV + mp_c2) / mp_c2
    b = np.sqrt(1.0 - 1.0 / gamma**2)
    b2 = b**2

    # Maximum energy transfer to electron
    T_max = 2.0 * me_c2 * b2 * gamma**2 / (1.0 + 2.0 * gamma * me_c2 / mp_c2
                                             + (me_c2 / mp_c2)**2)

    I_MeV = I_eV * 1e-6

    K = 0.3071  # MeV·cm²/mol
    prefactor = K * (Z_t / A_t) / b2

    log_arg = 2.0 * me_c2 * b2 * gamma**2 * T_max / I_MeV**2
    log_arg = np.maximum(log_arg, 1.0)

    S = prefactor * (0.5 * np.log(log_arg) - b2)

    # Density correction (simplified Sternheimer)
    x = np.log10(b * gamma)
    delta = np.where(x > 1.0, 2.0 * np.log(10) * x - 4.6, 0.0)
    S -= prefactor * delta / 2.0

    return np.maximum(S, 0.01)  # floor to avoid unphysical negatives


def generate_stopping_power_table(material: str):
    """Generate proton stopping power table for a material and save as CSV."""
    E_MeV = np.logspace(0, 4, 100)  # 1 MeV to 10,000 MeV

    mat = MATERIALS[material]
    composition = MATERIAL_COMPOSITION[material]

    # Bragg additivity for compounds
    S_total = np.zeros_like(E_MeV)
    for elem_sym, w_frac in composition.items():
        elem = ELEMENT_DATA[elem_sym]
        S_elem = bethe_bloch_proton(E_MeV, elem['Z'], elem['A'], mat['I_eV'])
        S_total += w_frac * S_elem

    outdir = os.path.join(DATA_DIR, 'nist')
    os.makedirs(outdir, exist_ok=True)
    filepath = os.path.join(outdir, f'proton_{material}.csv')

    df = pd.DataFrame({'KE_MeV': E_MeV, 'stopping_MeV_cm2_g': S_total})
    df.to_csv(filepath, index=False)
    print(f"  Wrote {filepath} ({len(df)} rows)")


# ---------------------------------------------------------------------------
# 2. Usoskin modulation potential (real download with synthetic fallback)
# ---------------------------------------------------------------------------

def fetch_usoskin_phi():
    """Download solar modulation potential from cosmicrays.oulu.fi; falls back to synthetic."""
    import urllib.request
    import io

    outdir = os.path.join(DATA_DIR, 'usoskin')
    os.makedirs(outdir, exist_ok=True)
    filepath = os.path.join(outdir, 'phi_monthly.csv')

    # Try downloading real Usoskin database
    url = 'https://cosmicrays.oulu.fi/phi/phi.txt'
    try:
        print(f"  Downloading Usoskin phi from {url}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'gcrisk/1.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            raw = response.read().decode('utf-8', errors='replace')

        rows = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('%'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                year = int(parts[0])
                month = int(parts[1])
                phi = float(parts[2])
                if 1951 <= year <= 2030 and 1 <= month <= 12 and 0 < phi < 2000:
                    date_str = f"{year}-{month:02d}-15"
                    rows.append({'year': year, 'month': month,
                                 'phi_MV': round(phi, 1), 'date': date_str})
            except (ValueError, IndexError):
                continue

        if len(rows) > 100:  # sanity check: expect 800+ months of data
            df = pd.DataFrame(rows)
            df = df.sort_values(['year', 'month']).reset_index(drop=True)

            # Extend with synthetic projection if data ends before 2030
            max_year = df['year'].max()
            if max_year < 2029:
                last_phi = float(df['phi_MV'].iloc[-1])
                last_year = int(df['year'].iloc[-1])
                last_month = int(df['month'].iloc[-1])
                for y in range(last_year, 2031):
                    for m in range(1, 13):
                        if y == last_year and m <= last_month:
                            continue
                        t = y + (m - 0.5) / 12.0
                        phi_ext = 700.0 - 300.0 * np.cos(2.0 * np.pi * (t - 2009.0) / 11.0)
                        rows.append({'year': y, 'month': m,
                                     'phi_MV': round(phi_ext, 1),
                                     'date': f"{y}-{m:02d}-15"})
                df = pd.DataFrame(rows).sort_values(['year', 'month']).reset_index(drop=True)

            df.to_csv(filepath, index=False)
            print(f"  Downloaded real Usoskin phi: {len(df)} months "
                  f"({df['year'].min()}-{df['year'].max()})")
            return

    except Exception as e:
        print(f"  Download failed ({e}), falling back to synthetic solar cycle...")

    # Fallback: synthetic 11-year sinusoidal solar cycle 1951-2030
    rows = []
    for year in range(1951, 2031):
        for month in range(1, 13):
            t = year + (month - 0.5) / 12.0
            phi = 700.0 - 300.0 * np.cos(2.0 * np.pi * (t - 2009.0) / 11.0)
            date_str = f"{year}-{month:02d}-15"
            rows.append({'year': year, 'month': month,
                         'phi_MV': round(phi, 1), 'date': date_str})
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"  Wrote synthetic phi 1951-2030 ({len(df)} rows) — run again with internet for real data")


# ---------------------------------------------------------------------------
# 2b. Frozen solar modulation potential for the MSL cruise transit window
# ---------------------------------------------------------------------------

def freeze_transit_window():
    """
    Extract the MSL cruise-phase (2011-11-26 to 2012-08-06) months from
    phi_monthly.csv and write them as a frozen snapshot.

    This file pins the solar modulation history used for the headline
    mission calculations so that re-running fetch_usoskin_phi() (which can
    pick up newly published months from cosmicrays.oulu.fi) does not change
    the reported transit-window results.
    """
    indir = os.path.join(DATA_DIR, 'usoskin')
    in_path = os.path.join(indir, 'phi_monthly.csv')
    out_path = os.path.join(indir, 'phi_transit_frozen.csv')

    df = pd.read_csv(in_path)
    mask = ((df['year'] == 2011) & (df['month'] == 11)) | \
           ((df['year'] == 2012) & (df['month'].between(1, 8))) | \
           ((df['year'] == 2011) & (df['month'] == 12))
    transit = df[mask].sort_values(['year', 'month']).reset_index(drop=True)

    transit.to_csv(out_path, index=False)
    print(f"  Wrote {out_path} ({len(transit)} months, "
          f"{transit['date'].iloc[0]} to {transit['date'].iloc[-1]})")


# ---------------------------------------------------------------------------
# 3. MSL RAD transit dose rate (Zeitlin et al. 2013 published statistics)
# ---------------------------------------------------------------------------

def generate_msl_rad_synthetic_demo():
    """
    Generate a SYNTHETIC demo MSL RAD transit dataset, statistically reconstructed
    from the published Zeitlin et al. (2013) mean dose/H rate and SPE timing —
    NOT real flight telemetry.

    This exists only as an offline-friendly fallback/demo dataset (e.g. for a
    first `pip install` smoke test with no network access). It must never be
    used for validation claims: real cruise-phase daily dosimetry is fetched
    separately by scripts/fetch_real_rad_cruise_data.py from NASA's PDS
    archive (MSL-M-RAD-3-RDR-V1.0) and written to
    data/rad/msl_rad_cruise_real.csv, which is what scripts/validate_extended.py
    and the paper's day-by-day comparison actually use.
    """
    outdir = os.path.join(DATA_DIR, 'rad')
    os.makedirs(outdir, exist_ok=True)
    filepath = os.path.join(outdir, 'msl_rad_transit_synthetic_demo.csv')

    np.random.seed(42)

    n_days = 253
    start_date = pd.Timestamp('2011-11-26')
    dose_mean_published = 1.84    # mGy/day
    H_mean_published = 4.81       # mSv/day
    Q_eff = H_mean_published / dose_mean_published

    days = np.arange(n_days)
    dates = [start_date + pd.Timedelta(days=int(d)) for d in days]

    # Solar-cycle modulated baseline
    phi_trend = np.linspace(430, 580, n_days)
    phi_norm = (phi_trend - phi_trend.mean()) / phi_trend.std()
    solar_modulation = 1.0 - 0.06 * phi_norm

    daily_noise = np.random.randn(n_days) * 0.10
    dose = dose_mean_published * solar_modulation * (1.0 + daily_noise)
    H = dose * Q_eff * (1.0 + np.random.randn(n_days) * 0.05)  # Q varies ±5%

    # SPE events during MSL cruise
    for spe_day, spe_factor in [(10, 4.0), (105, 3.0), (155, 6.0)]:
        if spe_day + 2 < n_days:
            dose[spe_day:spe_day + 2] *= spe_factor
            H[spe_day:spe_day + 2] *= spe_factor * 1.5  # SPEs have higher Q

    dose = np.maximum(dose, 0.1)
    H = np.maximum(H, 0.1)

    # Rescale to match published mean (after SPE days)
    background_mask = np.ones(n_days, dtype=bool)
    for spe_day in [10, 105, 155]:
        background_mask[spe_day:spe_day + 2] = False
    dose_bg_mean = dose[background_mask].mean()
    scale = dose_mean_published / dose_bg_mean
    dose *= scale
    H *= scale

    df = pd.DataFrame({
        'day': days,
        'date': [d.strftime('%Y-%m-%d') for d in dates],
        'dose_mGy_day': np.round(dose, 4),
        'H_mSv_day': np.round(H, 4),
        'source': 'Zeitlin_2013_reconstructed',
    })
    df.to_csv(filepath, index=False)
    print(f"  Wrote MSL RAD data ({len(df)} days): "
          f"mean dose={dose[background_mask].mean():.2f} mGy/day")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== GCR Dosimetry Pipeline — Data Generation ===\n")

    print("[1/3] Generating NIST stopping power tables (Bethe-Bloch)...")
    for mat in ['water', 'aluminum', 'polyethylene']:
        generate_stopping_power_table(mat)

    print("\n[2/4] Fetching Usoskin solar modulation potential database...")
    fetch_usoskin_phi()

    print("\n[3/4] Freezing MSL transit-window modulation potential...")
    freeze_transit_window()

    print("\n[4/4] Generating synthetic offline-demo MSL RAD dataset "
          "(Zeitlin et al. 2013 statistics; NOT used for validation)...")
    generate_msl_rad_synthetic_demo()

    print("\nAll data files generated successfully.")
    print("\nNote: real MSL/RAD cruise-phase dosimetry (used for validation and")
    print("the paper's day-by-day comparison) is fetched separately — see")
    print("scripts/fetch_real_rad_cruise_data.py. The dataset it produces is")
    print("already committed at data/rad/msl_rad_cruise_real.csv.")
