#!/usr/bin/env python3
"""
scripts/fetch_iss_tepc_data.py — Fetch real ISS radiation dosimetry from
NASA's CDAWeb HAPI service, for an independent, multi-year (2010-2018)
solar-cycle modulation cross-check (separate from the MSL/RAD cruise-phase
comparison, which tests absolute shielded dose at one epoch).

Dataset: ISS_DOSANL_TEPC (Position-Sensitive Tissue-Equivalent Proportional
Counter, dose analyzer). Reports DOSE_EQ60, the ICRP-60 dose-equivalent
rate, at ~10 s resolution. Contact: NASA Space Radiation Analysis Group
(JSC). Source: https://cdaweb.gsfc.nasa.gov/hapi/

A full 8-year pull at native ~10 s resolution would be tens of millions of
rows — far more than needed for a monthly-resolution modulation-trend
check (which is all the pipeline's own monthly-resolution phi input can
be compared against anyway). Instead, for each calendar month this script
pulls the first 5 days only and takes the MEDIAN (robust to transient
South-Atlantic-Anomaly passages and SEP events, which are not the GCR
solar-modulation signal this check is testing) as that month's
representative background dose-equivalent rate.

IMPORTANT CAVEAT (see also scripts/compare_iss_solar_cycle.py and the
paper's Discussion): the ISS orbits inside Earth's magnetosphere, which
attenuates GCR flux by an amount this pipeline's deep-space transport
model does not compute. This dataset is therefore usable only for a
relative/shape (modulation-trend) comparison, never as an absolute-dose
validation point.

Usage:
    python scripts/fetch_iss_tepc_data.py

Writes data/iss/iss_tepc_monthly.csv (already committed to the repo, so
this script does not need to be re-run to reproduce the paper's results).
"""

import csv
import datetime as dt
import os
import statistics
import sys
import time
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'iss')
OUT_FILE = os.path.join(DATA_DIR, 'iss_tepc_monthly.csv')

HAPI_URL = "https://cdaweb.gsfc.nasa.gov/hapi/data"
DATASET_ID = "ISS_DOSANL_TEPC"

START = dt.date(2010, 5, 1)   # dataset starts 2010-04-29; first full month is May 2010
END = dt.date(2018, 4, 1)     # dataset ends 2018-04-11
WINDOW_DAYS = 5                # subsample window per month


def _month_range(start: dt.date, end: dt.date):
    d = dt.date(start.year, start.month, 1)
    while d <= end:
        yield d
        d = dt.date(d.year + (d.month // 12), (d.month % 12) + 1, 1)


def _fetch_month(month_start: dt.date, retries: int = 3) -> list[float]:
    window_end = month_start + dt.timedelta(days=WINDOW_DAYS)
    url = (f"{HAPI_URL}?id={DATASET_ID}&parameters=Time,DOSE_EQ60"
           f"&time.min={month_start.isoformat()}&time.max={window_end.isoformat()}"
           f"&format=csv")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "gcr-dosimetry-pipeline"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            vals = []
            for line in text.strip().splitlines():
                parts = line.split(",")
                if len(parts) != 2:
                    continue
                try:
                    v = float(parts[1])
                except ValueError:
                    continue
                if v > -1e30 and v >= 0:  # HAPI fill value is -1e31
                    vals.append(v)
            return vals
        except Exception as e:
            if attempt == retries - 1:
                print(f"  FAILED {month_start}: {e}", file=sys.stderr)
                return []
            time.sleep(2.0)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    rows = []
    months = list(_month_range(START, END))
    for i, month_start in enumerate(months):
        vals = _fetch_month(month_start)
        if not vals:
            print(f"[{i+1}/{len(months)}] {month_start}: no data")
            continue
        med = statistics.median(vals)
        rows.append({
            "date": month_start.isoformat(),
            "n_samples": len(vals),
            "dose_eq60_median_uSv_min": round(med, 5),
        })
        print(f"[{i+1}/{len(months)}] {month_start}: n={len(vals)}  "
              f"median={med:.4f} uSv/min")
        time.sleep(0.2)

    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "n_samples", "dose_eq60_median_uSv_min"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nWrote {len(rows)} monthly rows to {OUT_FILE}")


if __name__ == "__main__":
    main()
