#!/usr/bin/env python3
"""
scripts/sensitivity_narrowing_sweep.py — Value-of-information sweep.

For each of the 9 LHS uncertainty parameters, reruns the N=500 ensemble
with that single parameter's spread halved (same central estimate, half
the uncertainty; see gcr.uncertainty.lhs_samples's `narrow` argument),
holding every other parameter at its baseline distribution, and reports
how much the REID p95 (and the p5-p95 width) shrinks relative to the
all-baseline ensemble.

This directly answers: "if we could only narrow ONE source of uncertainty
by half, which one would most reduce mission risk uncertainty?" — turning
the variance-decomposition finding (Section 'Sensitivity Decomposition')
into a concrete, ranked research-prioritization table.

Baseline scenario matches scripts/run_ensemble_paper_numbers.py: 35-year-old
male, 259-day Earth-Mars transit, 16 g/cm² aluminum.

Usage:
    python scripts/sensitivity_narrowing_sweep.py [--samples 500]

Saves results to data/sensitivity_narrowing_sweep.json.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gcr.spectrum import load_usoskin_phi
from gcr.trajectory import generate_trajectory
from gcr.uncertainty import PARAM_NAMES, run_uncertainty_ensemble

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'usoskin')
OUT_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'sensitivity_narrowing_sweep.json')

NARROW_FACTOR = 0.5  # halve the uncertainty


def run(n_samples: int = 500, seed: int = 42) -> dict:
    phi_df = load_usoskin_phi(os.path.join(DATA_DIR, 'phi_transit_frozen.csv'))
    traj = generate_trajectory('2011-11-26', phi_df=phi_df)

    common = dict(
        trajectory_df=traj, shielding_x=16.0, material='aluminum',
        age=35, sex='male', n_samples=n_samples, phi_df=phi_df, seed=seed,
    )

    print(f"Baseline (all parameters at full uncertainty), N={n_samples} ...")
    t0 = time.time()
    baseline = run_uncertainty_ensemble(**common)
    print(f"  Done in {time.time()-t0:.1f}s")
    base_p95 = baseline['percentiles']['REID']['p95']
    base_p50 = baseline['percentiles']['REID']['p50']
    base_p5 = baseline['percentiles']['REID']['p5']
    base_width = base_p95 - base_p5
    print(f"  REID p5={base_p5:.3%}  p50={base_p50:.3%}  p95={base_p95:.3%}  "
          f"(p5-p95 width={base_width:.3%})")

    results = {
        'baseline': {
            'p5': base_p5, 'p50': base_p50, 'p95': base_p95, 'width': base_width,
            'n_samples': n_samples,
        },
        'narrowed': {},
    }

    for pname in PARAM_NAMES:
        print(f"\nNarrowing {pname} (sigma/range x{NARROW_FACTOR}), N={n_samples} ...")
        t0 = time.time()
        res = run_uncertainty_ensemble(narrow={pname: NARROW_FACTOR}, **common)
        elapsed = time.time() - t0
        p95 = res['percentiles']['REID']['p95']
        p50 = res['percentiles']['REID']['p50']
        p5 = res['percentiles']['REID']['p5']
        width = p95 - p5
        d_p95 = base_p95 - p95
        pct_reduction = d_p95 / base_p95 * 100 if base_p95 > 0 else float('nan')
        print(f"  Done in {elapsed:.1f}s")
        print(f"  REID p5={p5:.3%}  p50={p50:.3%}  p95={p95:.3%}  "
              f"(width={width:.3%})")
        print(f"  Delta p95: {d_p95:+.3%}  ({pct_reduction:+.1f}% of baseline p95)")

        results['narrowed'][pname] = {
            'p5': p5, 'p50': p50, 'p95': p95, 'width': width,
            'delta_p95': d_p95, 'pct_reduction_p95': pct_reduction,
        }

    ranked = sorted(results['narrowed'].items(), key=lambda kv: -kv[1]['delta_p95'])
    print("\n" + "=" * 70)
    print("RANKED BY p95 REID REDUCTION (halving that parameter's uncertainty)")
    print("=" * 70)
    print(f"{'Parameter':20s}  {'Delta p95':>12s}  {'% of baseline p95':>18s}")
    for pname, r in ranked:
        print(f"{pname:20s}  {r['delta_p95']:+11.3%}  {r['pct_reduction_p95']:+17.1f}%")

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_FILE}")
    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--samples', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    run(args.samples, args.seed)
