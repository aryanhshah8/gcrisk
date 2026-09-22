#!/usr/bin/env python3
"""Paper 2 — Item 1: sensitivity analysis on biological priors.

Runs the Paper 1 LHS ensemble twice:

  (a) Paper 1 baseline priors (DDREF uniform(1, 2)).
  (b) BEIR VII-style widened DDREF prior (uniform(1, 3)).

Framing: this is a robustness check on the Paper 1 variance ranking, NOT
a change of baseline. Output tables land in ``figures/paper2/`` and are
consumed by the LET-attribution script.

Default sample count is small so the script can run in minutes; set
``UNCERTAINTY_SAMPLES`` to reproduce at paper-resolution.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
))

from gcr.paper2 import BEIR_VII_DDREF_PRIOR, run_ensemble_with_priors

from scripts.paper2._common import FIGURES_DIR, ensure_dirs, load_baseline


def _format_variance(vd: dict) -> pd.DataFrame:
    """Flatten the variance_decomposition dict into a tidy DataFrame."""
    rows = []
    for out_name, frac_map in vd.items():
        for pname, frac in frac_map.items():
            rows.append({
                'output': out_name,
                'param': pname,
                'variance_fraction': frac,
            })
    return pd.DataFrame(rows)


def run(n_samples: int, seed: int) -> dict:
    ensure_dirs()
    baseline = load_baseline()

    print(f"[paper2.01] Running baseline ensemble  n={n_samples}, seed={seed}")
    res_base = run_ensemble_with_priors(
        baseline['trajectory_df'],
        shielding_x=baseline['shielding_x_gcm2'],
        material=baseline['material'],
        age=baseline['age'], sex=baseline['sex'],
        prior_overrides=None,
        n_samples=n_samples,
        phi_df=baseline['phi_df'],
        seed=seed,
    )

    print(f"[paper2.01] Running widened-DDREF ensemble  n={n_samples}, seed={seed}")
    res_wide = run_ensemble_with_priors(
        baseline['trajectory_df'],
        shielding_x=baseline['shielding_x_gcm2'],
        material=baseline['material'],
        age=baseline['age'], sex=baseline['sex'],
        prior_overrides={'DDREF': BEIR_VII_DDREF_PRIOR},
        n_samples=n_samples,
        phi_df=baseline['phi_df'],
        seed=seed,
    )

    vd_base = _format_variance(res_base['variance_decomposition'])
    vd_base['scenario'] = 'paper1_baseline'
    vd_wide = _format_variance(res_wide['variance_decomposition'])
    vd_wide['scenario'] = 'widened_DDREF_uniform_1_3'
    vd = pd.concat([vd_base, vd_wide], ignore_index=True)

    variance_path = os.path.join(FIGURES_DIR, 'variance_widened.csv')
    vd.to_csv(variance_path, index=False)
    print(f"[paper2.01] Wrote {variance_path}")

    summary = {
        'n_samples': n_samples,
        'seed': seed,
        'paper1_baseline': {
            'percentiles': res_base['percentiles'],
            'n_valid': res_base['n_valid'],
            'variance_fractions_REID': res_base['variance_decomposition']['REID'],
        },
        'widened_DDREF': {
            'percentiles': res_wide['percentiles'],
            'n_valid': res_wide['n_valid'],
            'variance_fractions_REID': res_wide['variance_decomposition']['REID'],
        },
    }
    summary_path = os.path.join(FIGURES_DIR, 'widened_prior_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[paper2.01] Wrote {summary_path}")

    return {
        'baseline': res_base,
        'widened': res_wide,
        'variance_long': vd,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n-samples', type=int, default=40,
                   help='LHS samples per ensemble (default 40 for fast smoke).')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    run(n_samples=args.n_samples, seed=args.seed)


if __name__ == '__main__':
    main()
