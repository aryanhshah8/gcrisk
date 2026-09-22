#!/usr/bin/env python3
"""
scripts/daily_phi_from_nm.py — Attempt to fix the weak daily Pearson r
(model vs. real MSL/RAD dose rate) by giving the model daily-resolution
solar modulation instead of the monthly-interpolated Usoskin phi.

THE HYPOTHESIS: the documented reason for the weak daily correlation
(r=-0.127, p=0.06; scripts/validate_extended.py::check_daily_dose_timeseries)
is that phi(t) only has monthly resolution, so the model cannot see
Forbush decreases or other sub-monthly GCR flux structure that real RAD
telemetry does. Ground-based neutron monitor (NM) count rate is a real,
independently-measured DAILY proxy for the same physical quantity
(cosmic-ray-induced NM count rate is literally what the Usoskin phi
reconstruction is built from, just at monthly resolution there). If this
hypothesis is right, adding a real daily NM-derived perturbation on top of
the existing monthly phi baseline should measurably improve the daily r.

METHOD (calibrate-then-apply, not fit-on-the-test-set):
  1. Fit phi ~= a + b * oulu_count_rate on an INDEPENDENT window
     (data/nmdb/oulu_monthly_calibration_1995_2011.csv vs. Usoskin monthly
     phi for the same months) that ends before the MSL cruise starts. The
     slope b is never touched by anything in the cruise window.
  2. For each cruise day, compute that day's Oulu count-rate anomaly
     relative to its own calendar month's mean (data/nmdb/oulu_daily_msl_cruise.csv),
     and convert to a phi perturbation via the externally-calibrated slope b.
     This perturbation is zero-mean within each month by construction, so it
     adds daily structure without changing the existing monthly calibration.
  3. Add that perturbation to the existing smooth (linearly-interpolated)
     monthly phi baseline to get a genuine daily-resolution phi(t), feed it
     through the SAME trajectory/dose pipeline used everywhere else, and
     recompute the real Pearson r against RAD telemetry exactly as
     validate_extended.py does.

If this does not improve r, that is reported as-is -- this script's job is
to test the hypothesis honestly, not to manufacture a better number.

Run: python scripts/daily_phi_from_nm.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, linregress

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gcr.spectrum import load_usoskin_phi
from gcr.trajectory import generate_trajectory
from gcr.dose import integrate_mission_dose
from gcr.utils import DEFAULT_E_GRID

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAD_D_mGy_day = 1.84


def _load_oulu(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['date'])
    return df


def fit_calibration() -> tuple:
    """Step 1: phi ~= a + b*count_rate, fit ONLY on 1995-2011 data (never
    touches the MSL cruise window this is later applied to)."""
    usoskin = load_usoskin_phi(os.path.join(DATA_DIR, 'usoskin', 'phi_monthly.csv'))
    oulu_m = _load_oulu(os.path.join(DATA_DIR, 'nmdb', 'oulu_monthly_calibration_1995_2011.csv'))
    oulu_m['year'] = oulu_m['date'].dt.year
    oulu_m['month'] = oulu_m['date'].dt.month

    usoskin_reset = usoskin.reset_index()
    usoskin_reset['year'] = usoskin_reset['datetime'].dt.year
    usoskin_reset['month'] = usoskin_reset['datetime'].dt.month

    merged = pd.merge(oulu_m, usoskin_reset[['year', 'month', 'phi_MV']],
                       on=['year', 'month'], how='inner')
    n = len(merged)

    fit = linregress(merged['count_rate'], merged['phi_MV'])
    print(f"  Calibration window: {n} months, 1995-01 to 2011-10 "
          f"(ends {(pd.Timestamp('2011-11-26') - merged['date'].max()).days} days "
          "before the MSL cruise starts)")
    print(f"  phi = {fit.intercept:.1f} + ({fit.slope:.3f}) * count_rate   "
          f"r={fit.rvalue:.3f}, p={fit.pvalue:.2e}")
    return fit.slope, fit.intercept, n


def build_daily_phi_df(slope: float) -> pd.DataFrame:
    """Steps 2-3: monthly Usoskin baseline (linearly interpolated, same as
    the existing pipeline) + NM-derived daily perturbation, one row per day
    so phi_at_date's interpolation returns each day's own value exactly."""
    from gcr.spectrum import phi_at_date

    monthly_baseline = load_usoskin_phi(
        os.path.join(DATA_DIR, 'usoskin', 'phi_transit_frozen.csv')
    )
    oulu_d = _load_oulu(os.path.join(DATA_DIR, 'nmdb', 'oulu_daily_msl_cruise.csv'))
    oulu_d['year'] = oulu_d['date'].dt.year
    oulu_d['month'] = oulu_d['date'].dt.month
    monthly_mean_nm = oulu_d.groupby(['year', 'month'])['count_rate'].transform('mean')
    oulu_d['nm_anomaly'] = oulu_d['count_rate'] - monthly_mean_nm
    oulu_d['phi_perturbation'] = slope * oulu_d['nm_anomaly']
    oulu_d = oulu_d.set_index('date')

    launch = pd.Timestamp('2011-11-26')
    trajectory_dates = pd.date_range(launch, periods=260, freq='D')

    rows = []
    n_with_nm = 0
    for d in trajectory_dates:
        base = phi_at_date(d.strftime('%Y-%m-%d'), monthly_baseline)
        if d in oulu_d.index:
            phi_val = base + float(oulu_d.loc[d, 'phi_perturbation'])
            n_with_nm += 1
        else:
            phi_val = base  # no NM coverage that day -> fall back to smooth baseline
        rows.append({'phi_MV': phi_val, 'date': d.strftime('%Y-%m-%d')})

    print(f"  Daily phi built for {len(rows)} days; {n_with_nm} carry a real "
          f"NM-derived perturbation, {len(rows) - n_with_nm} fall back to the "
          "smooth monthly baseline (no NM coverage that day)")

    df = pd.DataFrame(rows)
    df['datetime'] = pd.to_datetime(df['date'])
    df = df.set_index('datetime')
    return df


def rerun_validation(phi_df: pd.DataFrame, label: str) -> tuple:
    """Re-run the trajectory + dose integration + real Pearson r check,
    exactly mirroring validate_extended.py::check_daily_dose_timeseries."""
    traj = generate_trajectory('2011-11-26', phi_df=phi_df)
    result = integrate_mission_dose(
        traj, shielding_x_gcm2=16.0, shielding_material='aluminum',
        phi_df=phi_df, E_grid_MeV=DEFAULT_E_GRID,
    )
    traj = traj.copy()
    traj['D_rate_daily'] = result['D_rate_daily']
    traj['date'] = pd.to_datetime(traj['date']).dt.date.astype(str)

    rad = pd.read_csv(os.path.join(DATA_DIR, 'rad', 'msl_rad_cruise_real.csv'))
    merged = pd.merge(rad, traj[['date', 'D_rate_daily']], on='date', how='inner')
    n = len(merged)

    rad_raw = merged['dose_rate_B_uGyhr'].values * 24.0 / 1000.0
    pipe_D = merged['D_rate_daily'].values
    scale = RAD_D_mGy_day / np.median(rad_raw)
    rad_D = rad_raw * scale

    r, p_value = pearsonr(rad_D, pipe_D)
    print(f"  [{label}] n={n}  Pearson r={r:.3f}  p={p_value:.3f}")
    return r, p_value, n


def main() -> None:
    print("[1/3] fitting count-rate -> phi calibration on an independent "
          "1995-2011 window")
    slope, intercept, n_calib = fit_calibration()

    print("\n[2/3] building daily-resolution phi(t) for the MSL cruise "
          "(monthly baseline + NM-derived daily perturbation)")
    daily_phi_df = build_daily_phi_df(slope)

    print("\n[3/3] re-running the real validation check")
    monthly_phi_df = load_usoskin_phi(
        os.path.join(DATA_DIR, 'usoskin', 'phi_transit_frozen.csv')
    )
    r_before, p_before, n_before = rerun_validation(monthly_phi_df, "monthly baseline (current pipeline)")
    r_after, p_after, n_after = rerun_validation(daily_phi_df, "monthly + daily NM perturbation")

    print("\n=== RESULT ===")
    print(f"  Before (monthly phi only):        r={r_before:.3f}  p={p_before:.3f}  n={n_before}")
    print(f"  After  (+ daily NM perturbation):  r={r_after:.3f}  p={p_after:.3f}  n={n_after}")
    delta = r_after - r_before
    print(f"  Delta r: {delta:+.3f}")
    if abs(r_after) > abs(r_before) and p_after < p_before:
        print("  VERDICT: improved -- both |r| larger and p smaller.")
    elif abs(r_after) > abs(r_before):
        print("  VERDICT: r moved in the expected direction but not "
              "decisively (check p-value and n above before trusting this).")
    else:
        print("  VERDICT: did NOT improve. The NM-derived daily perturbation "
              "does not rescue the daily correlation on this dataset -- "
              "report this as-is, do not keep tuning until it looks better.")


if __name__ == "__main__":
    main()
