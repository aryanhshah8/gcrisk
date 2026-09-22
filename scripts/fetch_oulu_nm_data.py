#!/usr/bin/env python3
"""
scripts/fetch_oulu_nm_data.py — Fetch real Oulu neutron monitor count-rate
data from NMDB.eu (the Neutron Monitor Database, nmdb.eu), for use as a
daily-resolution proxy for GCR flux modulation.

Motivation: the pipeline's solar modulation potential phi(t) comes from the
Usoskin et al. database at MONTHLY resolution, linearly interpolated. Real
MSL/RAD telemetry shows sub-monthly structure (Forbush decreases, minor SEP
events) that a monthly-resolution phi cannot reproduce, which is the
documented reason the daily Pearson r between model and real RAD dose rate
is weak (r=-0.127, p=0.06; see scripts/validate_extended.py). Ground-based
neutron monitor count rates are a real, independently-measured, DAILY (in
fact sub-hourly) proxy for GCR flux, including Forbush decreases -- this is
the same physical quantity (cosmic-ray-induced NM count rate) that the
Usoskin phi reconstruction is itself built from, just at monthly resolution
there and daily/hourly here.

Fetches two datasets from NMDB's NEST tool (public HTTP API, no key
required):

  1. Daily OULU count rate for the MSL cruise window (2011-11-20 to
     2012-07-20, padded a few days around the trajectory).
  2. Monthly OULU count rate for 1995-01 to 2011-10 -- an INDEPENDENT
     calibration window ending before the cruise starts, used to fit the
     count-rate -> phi relationship without touching the data being
     validated against (scripts/daily_phi_from_nm.py fits on this file
     only, never on the cruise-window file).

Data are free for non-commercial use (nmdb.eu); Oulu station is operated by
the University of Oulu, Finland (see http://cosmicrays.oulu.fi/).

Usage:
    python scripts/fetch_oulu_nm_data.py

Writes:
    data/nmdb/oulu_daily_msl_cruise.csv
    data/nmdb/oulu_monthly_calibration_1995_2011.csv
"""

import os
import re
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'nmdb')
NEST_URL = "https://www.nmdb.eu/nest/draw_graph.php"
ROW_RE = re.compile(r'^(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}:\d{2};([\d.]+)$', re.M)


def _fetch_nest(start_ymd: tuple, end_ymd: tuple, tresolution_min: int) -> list:
    """Query NEST's draw_graph.php (ASCII output embeds a semicolon-separated
    date;count_rate table inside the returned HTML page; grabbed via regex
    rather than a proper parser since NEST has no dedicated JSON endpoint)."""
    sy, sm, sd = start_ymd
    ey, em, ed = end_ymd
    params = (
        f"formchk=1&stations[]=OULU&tabchoice=1h&dtype=corr_for_efficiency"
        f"&tresolution={tresolution_min}&yunits=0&date_choice=bydate"
        f"&start_day={sd:02d}&start_month={sm:02d}&start_year={sy}"
        f"&start_hour=00&start_min=00"
        f"&end_day={ed:02d}&end_month={em:02d}&end_year={ey}"
        f"&end_hour=23&end_min=59&output=ascii"
    )
    url = f"{NEST_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    rows = [(d, float(v)) for d, v in ROW_RE.findall(html)]
    if not rows:
        raise RuntimeError(f"no data rows parsed from NEST response for {url}")
    return rows


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    print("[1/2] fetching daily OULU count rate, MSL cruise window...")
    daily = _fetch_nest((2011, 11, 20), (2012, 7, 20), tresolution_min=1440)
    daily_path = os.path.join(DATA_DIR, 'oulu_daily_msl_cruise.csv')
    with open(daily_path, 'w') as f:
        f.write('date,count_rate\n')
        for d, v in daily:
            f.write(f'{d},{v}\n')
    print(f"      {len(daily)} days -> {daily_path}")

    print("[2/2] fetching monthly OULU count rate, 1995-2011 (calibration, "
          "independent of the cruise window)...")
    monthly = _fetch_nest((1995, 1, 1), (2011, 10, 1), tresolution_min=43200)
    monthly_path = os.path.join(DATA_DIR, 'oulu_monthly_calibration_1995_2011.csv')
    with open(monthly_path, 'w') as f:
        f.write('date,count_rate\n')
        for d, v in monthly:
            f.write(f'{d},{v}\n')
    print(f"      {len(monthly)} months -> {monthly_path}")


if __name__ == "__main__":
    main()
