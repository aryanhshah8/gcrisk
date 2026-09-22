"""Validate and concatenate external clinical-RBE CSV files.

This script implements item 5 of the Paper 2 plan. It does NOT curate data;
curation is a human literature-review task. The script's job is to:

- enforce the schema documented in ``data/clinical_rbe/README.md``,
- reject rows without a DOI (provenance is mandatory),
- reject any attempt to ingest ``data/topas/*.csv`` (those are model outputs,
  not independent biological evidence), and
- write a single normalized ``clinical_rbe_normalized.csv`` that downstream
  figures can read.

Usage:
    python scripts/paper2/ingest_clinical_rbe.py

The script exits 0 with an empty output if no source CSVs are present yet —
the rest of the Paper 2 pipeline tolerates that state.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CLINICAL_DIR = REPO_ROOT / 'data' / 'clinical_rbe'
TOPAS_DIR = REPO_ROOT / 'data' / 'topas'
OUTPUT_CSV = CLINICAL_DIR / 'clinical_rbe_normalized.csv'
MANIFEST_JSON = CLINICAL_DIR / 'clinical_rbe_manifest.json'

REQUIRED_COLUMNS = (
    'doi',
    'source',
    'cell_line',
    'alpha_beta_Gy',
    'LET_keV_um',
    'endpoint',
    'RBE',
)
OPTIONAL_COLUMNS = ('RBE_unc',)
ALLOWED_SOURCES = {'pide', 'paganetti2014', 'wedenberg2013'}


class IngestError(ValueError):
    """Raised when a source CSV fails validation."""


def _assert_not_model_output(path: Path) -> None:
    try:
        rel = path.resolve().relative_to(TOPAS_DIR.resolve())
    except ValueError:
        return
    raise IngestError(
        f"Refusing to ingest {rel}: data/topas/ holds Wedenberg/McNamara "
        "model outputs, not independent biological evidence. Place external "
        "RBE measurements under data/clinical_rbe/ instead."
    )


def validate_frame(df: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise IngestError(
            f"{label}: missing required columns {missing}"
        )
    if df['doi'].isna().any() or (df['doi'].astype(str).str.strip() == '').any():
        bad = df[df['doi'].isna() | (df['doi'].astype(str).str.strip() == '')]
        raise IngestError(
            f"{label}: {len(bad)} row(s) missing DOI — provenance required"
        )
    bad_source = set(df['source'].unique()) - ALLOWED_SOURCES
    if bad_source:
        raise IngestError(
            f"{label}: unknown source tag(s) {bad_source}; "
            f"allowed: {sorted(ALLOWED_SOURCES)}"
        )
    for num_col in ('alpha_beta_Gy', 'LET_keV_um', 'RBE'):
        if not pd.api.types.is_numeric_dtype(df[num_col]):
            raise IngestError(f"{label}: column '{num_col}' must be numeric")
    if (df['LET_keV_um'] < 0).any() or (df['RBE'] <= 0).any():
        raise IngestError(f"{label}: nonphysical LET<0 or RBE<=0 present")

    keep = list(REQUIRED_COLUMNS) + [
        c for c in OPTIONAL_COLUMNS if c in df.columns
    ]
    out = df[keep].copy()
    if 'RBE_unc' not in out.columns:
        out['RBE_unc'] = pd.NA
    return out


def discover_source_files() -> list[Path]:
    if not CLINICAL_DIR.is_dir():
        return []
    ignore = {OUTPUT_CSV.name, 'README.md'}
    files = []
    for p in sorted(CLINICAL_DIR.glob('*.csv')):
        if p.name in ignore:
            continue
        files.append(p)
    return files


def ingest(verbose: bool = True) -> dict:
    files = discover_source_files()

    if not files:
        if verbose:
            print(
                "[ingest] No source CSVs found in data/clinical_rbe/. "
                "See README.md for required schema and sources."
            )
        if OUTPUT_CSV.exists():
            OUTPUT_CSV.unlink()
        if MANIFEST_JSON.exists():
            MANIFEST_JSON.unlink()
        return {'n_files': 0, 'n_rows': 0, 'files': []}

    frames: list[pd.DataFrame] = []
    manifest_entries: list[dict] = []
    for p in files:
        _assert_not_model_output(p)
        df = pd.read_csv(p)
        validated = validate_frame(df, label=p.name)
        frames.append(validated)
        manifest_entries.append({
            'file': p.name,
            'rows': len(validated),
            'sources': sorted(validated['source'].unique().tolist()),
            'cell_lines': sorted(validated['cell_line'].unique().tolist()),
            'LET_min_keV_um': float(validated['LET_keV_um'].min()),
            'LET_max_keV_um': float(validated['LET_keV_um'].max()),
        })
        if verbose:
            print(
                f"[ingest] {p.name}: {len(validated)} rows, "
                f"LET {validated['LET_keV_um'].min():.1f}"
                f"–{validated['LET_keV_um'].max():.1f} keV/µm"
            )

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUTPUT_CSV, index=False)

    manifest = {
        'n_files': len(files),
        'n_rows': int(len(combined)),
        'files': manifest_entries,
    }
    with open(MANIFEST_JSON, 'w') as f:
        json.dump(manifest, f, indent=2)

    if verbose:
        print(
            f"[ingest] Wrote {len(combined)} rows to "
            f"{OUTPUT_CSV.relative_to(REPO_ROOT)}"
        )
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()
    try:
        ingest(verbose=not args.quiet)
    except IngestError as e:
        print(f"[ingest] ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
