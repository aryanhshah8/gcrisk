#!/usr/bin/env python3
"""
scripts/fetch_real_rad_cruise_data.py — Fetch real MSL/RAD cruise-phase
dosimetry from NASA's Planetary Data System.

Downloads and parses the Reduced Data Record (RDR) daily files for the
Radiation Assessment Detector (RAD) instrument during the MSL Earth-Mars
cruise phase, archived at:

    Dataset:  MSL-M-RAD-3-RDR-V1.0 (PDS PPI Node, UCLA)
    URL:      https://pds-ppi.igpp.ucla.edu/data/MSL-M-RAD-3-RDR-V1.0/DATA/CRUISE/
    Producer: Scot Rafkin (RAD team)

Each daily .TXT/.LBL pair contains, per observation (roughly one every
20-40 minutes), a "Total Dose B" and "Total Dose E" scalar dosimetry
element (ASCII_REAL, units microGray/hour) at a byte offset given in the
.LBL file. This script extracts those values, averages them per sol, and
writes a daily CSV.

This is real flight telemetry, not a statistical reconstruction — it
replaces the earlier synthetic demo dataset
(scripts/download_data.py:generate_msl_rad_synthetic_demo) as the basis
for the pipeline's day-by-day dose-rate validation
(scripts/validate_extended.py). The PDS cruise-phase archive covers
2011-340 through 2012-196 (218 of the mission's ~253 cruise days; RAD's
early commissioning period and the final weeks before landing are not
included in this archive release).

Note: detector B/E dose rates are individual RAD sub-detector elements,
not independently calibrated to the total absorbed dose Zeitlin et al.
(2013) report (1.84 mGy/day) — do not compare these raw values to the
pipeline's D_rate_daily in absolute terms without the same median-anchored
rescaling that scripts/validate_extended.py applies. Pearson correlation
(the primary use of this dataset) is scale/offset-invariant, so this
rescaling only matters for the RMS-error/bias checks, not the correlation
itself.

Usage:
    python scripts/fetch_real_rad_cruise_data.py

Network use: ~220 files, ~2 requests each, politely rate-limited
(0.15 s between requests). Takes on the order of 10-20+ minutes depending
on network conditions. Writes data/rad/msl_rad_cruise_real.csv, which is
committed to the repository so this script does not need to be re-run to
reproduce the paper's results.
"""

import csv
import datetime as dt
import os
import re
import sys
import time
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'rad')
OUT_FILE = os.path.join(DATA_DIR, 'msl_rad_cruise_real.csv')

BASE_URL = "https://pds-ppi.igpp.ucla.edu/data/MSL-M-RAD-3-RDR-V1.0/DATA/CRUISE/"
HEADERS = {"User-Agent": "Mozilla/5.0 (research data fetch; gcrisk)"}

PTR_RE = re.compile(r'\^(OBS\d+_TOT_DOSE_([BE])_ELEMENT)\s*=\s*\n?\s*\("[^"]+",\s*(\d+)\s*<BYTES>\)')
START_TIME_RE = re.compile(r'START_TIME\s*=\s*([\d\-T:]+)')


def _fetch(url: str, retries: int = 3, timeout: float = 60.0) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2.0)


def _list_cruise_files() -> list[str]:
    """Directory-listing the CRUISE/ folder and extract .TXT filenames."""
    html = _fetch(BASE_URL).decode("utf-8", errors="replace")
    names = sorted(set(re.findall(r'href="(RAD_RDR[^"]*\.TXT)"', html, re.IGNORECASE)))
    return names


def _doy_to_date(start_time: str) -> str:
    """Convert 'YYYY-DDDTHH:MM:SS' to an ISO calendar date string."""
    year, rest = start_time.split('-')
    doy = rest.split('T')[0]
    return (dt.datetime(int(year), 1, 1) + dt.timedelta(days=int(doy) - 1)).date().isoformat()


def fetch_all() -> list[dict]:
    txt_names = _list_cruise_files()
    print(f"Found {len(txt_names)} cruise-phase daily files.")

    rows = []
    for i, txt_name in enumerate(txt_names):
        lbl_name = txt_name[:-4] + ".LBL"
        try:
            lbl = _fetch(BASE_URL + lbl_name).decode("ascii", errors="replace")
        except Exception as e:
            print(f"  [{i+1}/{len(txt_names)}] LBL fetch failed for {lbl_name}: {e}", file=sys.stderr)
            continue

        m = START_TIME_RE.search(lbl)
        if not m:
            continue
        date = _doy_to_date(m.group(1))

        offsets = PTR_RE.findall(lbl)
        if not offsets:
            continue

        try:
            txt_bytes = _fetch(BASE_URL + txt_name)
        except Exception as e:
            print(f"  [{i+1}/{len(txt_names)}] TXT fetch failed for {txt_name}: {e}", file=sys.stderr)
            continue

        vals_b, vals_e = [], []
        for _, kind, off_str in offsets:
            off = int(off_str) - 1
            chunk = txt_bytes[off:off + 8].decode("ascii", errors="replace").strip()
            try:
                v = float(chunk)
            except ValueError:
                continue
            (vals_b if kind == "B" else vals_e).append(v)

        if vals_b or vals_e:
            mean_b = sum(vals_b) / len(vals_b) if vals_b else float("nan")
            mean_e = sum(vals_e) / len(vals_e) if vals_e else float("nan")
            rows.append({
                "date": date,
                "n_obs_B": len(vals_b),
                "n_obs_E": len(vals_e),
                "dose_rate_B_uGyhr": round(mean_b, 5),
                "dose_rate_E_uGyhr": round(mean_e, 5),
            })
            print(f"  [{i+1}/{len(txt_names)}] {date}: "
                  f"nB={len(vals_b)} B={mean_b:.3f} uGy/hr, nE={len(vals_e)} E={mean_e:.3f} uGy/hr")

        time.sleep(0.15)

    rows.sort(key=lambda r: r["date"])
    for i, r in enumerate(rows):
        r["day"] = i
    return rows


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    rows = fetch_all()

    fieldnames = ["day", "date", "n_obs_B", "n_obs_E", "dose_rate_B_uGyhr", "dose_rate_E_uGyhr", "source"]
    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            r["source"] = "NASA_PDS_MSL-M-RAD-3-RDR-V1.0"
            writer.writerow(r)

    print(f"\nWrote {len(rows)} daily rows to {OUT_FILE}")


if __name__ == "__main__":
    main()
