#!/usr/bin/env python3
"""Paper 2 end-to-end orchestrator.

Runs items 1, 2, 3 (widened-prior sensitivity, LET-binned attribution,
decision-threshold sweep) and emits a single ``results.json`` summarizing
every number the manuscript will cite.

Does not run item 5 (external clinical RBE data ingestion) — that is
literature curation, handled separately in ``data/clinical_rbe/``.

Usage:
    python scripts/paper2/run_paper2_analysis.py \
        --ensemble-samples 40 --mc-samples 2000

For paper-resolution numbers, bump ``--ensemble-samples`` to 500 and
``--mc-samples`` to 10000; expect tens of minutes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
))

# Script modules have numeric prefixes so cannot be imported by dotted
# name — load them via importlib by file path.
_HERE = Path(__file__).resolve().parent


def _load(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        module_name, _HERE / filename,
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


from scripts.paper2._common import FIGURES_DIR, ensure_dirs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ensemble-samples', type=int, default=40,
                   help='LHS samples for the widened-prior ensembles (item 1 and 2).')
    p.add_argument('--mc-samples', type=int, default=2000,
                   help='MC samples per decision-threshold sweep point (item 4).')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    ensure_dirs()

    mod01 = _load('paper2_widened', '01_widened_prior_sensitivity.py')
    mod02 = _load('paper2_let_attribution', '02_let_attribution.py')
    mod03 = _load('paper2_decision', '03_decision_threshold.py')
    mod04 = _load('paper2_overlap', '04_let_overlap_figure.py')
    mod_ingest = _load('paper2_ingest_rbe', 'ingest_clinical_rbe.py')

    print("=" * 60)
    print("[paper2] Item 1 — widened-prior sensitivity analysis")
    print("=" * 60)
    r1 = mod01.run(n_samples=args.ensemble_samples, seed=args.seed)

    print("=" * 60)
    print("[paper2] Items 2 & 3 — LET-binned attribution")
    print("=" * 60)
    r2 = mod02.run(n_samples=args.ensemble_samples, seed=args.seed)

    print("=" * 60)
    print("[paper2] Item 4 — decision-threshold inversion")
    print("=" * 60)
    r3 = mod03.run(n_samples=args.mc_samples)

    print("=" * 60)
    print("[paper2] Item 5 — external clinical-RBE ingestion (optional)")
    print("=" * 60)
    r_ingest = mod_ingest.ingest(verbose=True)

    print("=" * 60)
    print("[paper2] Item 6 — LET-overlap figure")
    print("=" * 60)
    r4 = mod04.run()

    results = {
        'run_config': {
            'ensemble_samples': args.ensemble_samples,
            'mc_samples': args.mc_samples,
            'seed': args.seed,
        },
        'item1_widened_prior': {
            'paper1_baseline_REID_percentiles': (
                r1['baseline']['percentiles']['REID']
            ),
            'widened_DDREF_REID_percentiles': (
                r1['widened']['percentiles']['REID']
            ),
            'paper1_baseline_variance_REID': (
                r1['baseline']['variance_decomposition']['REID']
            ),
            'widened_DDREF_variance_REID': (
                r1['widened']['variance_decomposition']['REID']
            ),
        },
        'item2_let_attribution': {
            'bin_edges_keV_um': list(r2['binned'].bin_edges_keV_um),
            'H_fraction_by_bin': list(r2['binned'].H_fraction_by_bin),
            'D_fraction_by_bin': list(r2['binned'].dose_fraction_by_bin),
            'attribution_table': (
                r2['attribution'].to_dict(orient='index')
            ),
        },
        'item4_decision_threshold': {
            'H_total_Sv': r3['H_total_Sv'],
            'sweep': r3['sweep'].to_dict(orient='records'),
        },
        'item5_clinical_rbe_ingest': r_ingest,
        'item6_let_overlap': {
            'overlap_pdf': r4['overlap_path'],
            'biological_attribution_sum_by_bin': (
                r4['biological_attribution_sum'].to_dict()
            ),
            'clinical_points_ingested': r4['clinical_points'],
        },
    }

    out_path = os.path.join(FIGURES_DIR, 'results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print()
    print(f"[paper2] Wrote consolidated {out_path}")


if __name__ == '__main__':
    main()
