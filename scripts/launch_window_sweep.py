#!/usr/bin/env python3
"""
scripts/launch_window_sweep.py — REID vs. launch date across the solar cycle.

Uses gcr.reid.reid_vs_launch_date to sweep mission REID over monthly launch
dates, showing how much a mission's cancer-risk estimate depends on WHEN in
the solar cycle it launches (higher phi / solar maximum -> lower GCR flux
-> lower REID; solar minimum -> the opposite).

Two clearly separated panels, so this is never presented as a forecast it
isn't:

  1. REAL, historical (2003-01 through 2022-12): phi values are read
     directly from data/usoskin/phi_monthly.csv, which is real Usoskin
     reconstruction data over this range. This window comfortably covers
     all of Solar Cycle 23's decline, all of Cycle 24, and the start of
     Cycle 25 -- three real min/max transitions -- with no ambiguity about
     data provenance.

  2. ILLUSTRATIVE, 2023-01 through 2035-12: phi is NOT read from the
     database (which itself begins departing from real reconstruction
     data at an ambiguous point in this range -- we did not want to rely
     on a boundary we could not cleanly verify). Instead this script
     generates its own explicit idealized ~11-year cosine solar-cycle
     phi(t), the same textbook approximation used elsewhere in this
     repository as a documented fallback (see
     gcr.spectrum.load_usoskin_phi's synthetic branch), so its synthetic
     origin is unambiguous. This panel is illustrative only -- it assumes
     a perfectly repeating, idealized solar cycle, NOT a real space-weather
     forecast (real cycles vary in amplitude and length by several years)
     -- and is styled and labeled distinctly in the figure and never used
     to state a "best future launch date" as fact.

Usage:
    python scripts/launch_window_sweep.py

Writes data/launch_window_sweep.csv and figures/launch_window_sweep.pdf.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gcr.spectrum import load_usoskin_phi
from gcr.reid import reid_vs_launch_date

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

REAL_START = '2003-01-01'
REAL_END = '2022-12-01'
ILLUSTRATIVE_START = '2023-01-01'
ILLUSTRATIVE_END = '2035-12-01'

SHIELDING_X_GCM2 = 16.0
MATERIAL = 'aluminum'
AGE = 35
SEX = 'male'
NASA_LIMIT_PCT = 3.0


def _monthly_dates(start: str, end: str) -> list:
    return [d.strftime('%Y-%m-%d') for d in pd.date_range(start, end, freq='MS')]


def _illustrative_phi_df(start: str, end: str) -> pd.DataFrame:
    """Self-generated idealized cosine phi(t) — see module docstring.
    Same functional form as gcr.spectrum.load_usoskin_phi's synthetic
    fallback (700 - 300*cos(2*pi*(t-2009)/11)), generated fresh here so its
    provenance is explicit and never mixed with the real database file."""
    dates = pd.date_range(start, end, freq='MS')
    rows = []
    for d in dates:
        t = d.year + (d.month - 0.5) / 12.0
        phi = 700.0 - 300.0 * np.cos(2.0 * np.pi * (t - 2009.0) / 11.0)
        rows.append({'year': d.year, 'month': d.month, 'phi_MV': round(phi, 1),
                      'date': d.strftime('%Y-%m-15')})
    df = pd.DataFrame(rows)
    df['datetime'] = pd.to_datetime(df['date'])
    return df.set_index('datetime')


def main():
    real_phi_df = load_usoskin_phi(os.path.join(DATA_DIR, 'usoskin', 'phi_monthly.csv'))
    illustrative_phi_df = _illustrative_phi_df(ILLUSTRATIVE_START, ILLUSTRATIVE_END)

    print(f"Sweeping REAL historical launch dates {REAL_START} to {REAL_END} ...")
    real_dates = _monthly_dates(REAL_START, REAL_END)
    real_df = reid_vs_launch_date(real_dates, SHIELDING_X_GCM2, MATERIAL, AGE, SEX, real_phi_df)
    real_df['panel'] = 'real'
    real_df['launch_date'] = pd.to_datetime(real_df['launch_date'])
    print(f"  {len(real_df)} launch dates computed.")

    print(f"Sweeping ILLUSTRATIVE launch dates {ILLUSTRATIVE_START} to {ILLUSTRATIVE_END} ...")
    illustrative_dates = _monthly_dates(ILLUSTRATIVE_START, ILLUSTRATIVE_END)
    illustrative_df = reid_vs_launch_date(illustrative_dates, SHIELDING_X_GCM2, MATERIAL, AGE, SEX,
                                           illustrative_phi_df)
    illustrative_df['panel'] = 'illustrative'
    illustrative_df['launch_date'] = pd.to_datetime(illustrative_df['launch_date'])
    print(f"  {len(illustrative_df)} launch dates computed.")

    combined = pd.concat([real_df, illustrative_df], ignore_index=True)
    combined.to_csv(os.path.join(DATA_DIR, 'launch_window_sweep.csv'), index=False)

    best_real = real_df.loc[real_df['REID_median'].idxmin()]
    worst_real = real_df.loc[real_df['REID_median'].idxmax()]
    print(f"\nWithin the REAL 2003-2022 window:")
    print(f"  Lowest-REID launch month:  {best_real['launch_date']}  "
          f"(median REID = {best_real['REID_median']:.2%}, phi = {best_real['phi_at_launch']:.0f} MV)")
    print(f"  Highest-REID launch month: {worst_real['launch_date']}  "
          f"(median REID = {worst_real['REID_median']:.2%}, phi = {worst_real['phi_at_launch']:.0f} MV)")
    ratio = worst_real['REID_median'] / best_real['REID_median']
    print(f"  Ratio (worst/best): {ratio:.2f}x")

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(real_df['launch_date'], real_df['REID_p5'] * 100, real_df['REID_p95'] * 100,
                     color='tab:blue', alpha=0.15)
    ax.plot(real_df['launch_date'], real_df['REID_median'] * 100, color='tab:blue',
            linewidth=1.5, label='Median REID (real historical solar cycle, 2003-2022)')

    ax.fill_between(illustrative_df['launch_date'], illustrative_df['REID_p5'] * 100,
                     illustrative_df['REID_p95'] * 100, color='tab:gray', alpha=0.12, hatch='//')
    ax.plot(illustrative_df['launch_date'], illustrative_df['REID_median'] * 100, color='tab:gray',
            linewidth=1.5, linestyle='--',
            label='Median REID (ILLUSTRATIVE — idealized repeating cycle, NOT a forecast)')

    ax.axvline(pd.Timestamp(REAL_END), color='black', linewidth=0.8, linestyle=':')
    ax.axhline(NASA_LIMIT_PCT, color='red', linewidth=0.8, linestyle='-.',
               label=f'NASA historical {NASA_LIMIT_PCT:.0f}% career limit')

    ax.set_xlabel('Launch date')
    ax.set_ylabel('REID [%]')
    ax.set_title('Mission cancer risk (REID) vs. launch date across the solar cycle\n'
                  f'(35-year-old male, 259-day transit, {SHIELDING_X_GCM2:.0f} g/cm² Al; '
                  f'shaded = p5–p95)')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, 'launch_window_sweep.pdf'))
    fig.savefig(os.path.join(FIG_DIR, 'launch_window_sweep.png'), dpi=150)
    print(f"\nSaved figures/launch_window_sweep.pdf")


if __name__ == '__main__':
    main()
