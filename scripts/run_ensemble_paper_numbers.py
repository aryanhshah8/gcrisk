#!/usr/bin/env python3
"""
scripts/run_ensemble_paper_numbers.py

Run the N=500 LHS ensemble and print the numbers needed to update the paper:
  - REID p5/p50/p95 (male and female, 35 y.o., 259-day transit at 16 g/cm² Al)
  - D and H p5/p50/p95
  - Normalized Spearman ρ² sensitivity table (all 9 parameters)
  - Physics vs biology fraction of REID variance

Saves results to data/ensemble_paper_numbers.json for reference.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gcr.spectrum import load_usoskin_phi
from gcr.trajectory import generate_trajectory
from gcr.uncertainty import PARAM_NAMES, run_uncertainty_ensemble

PHYSICS_PARAMS = {'phi_scale', 'LIS_norm', 'cross_sec', 'neutron_H', 'hze_norm_scale'}
BIO_PARAMS     = {'Q_factor', 'DDREF', 'ERR_scale', 'EAR_scale'}

DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'usoskin')
OUT_FILE  = os.path.join(os.path.dirname(__file__), '..', 'data', 'ensemble_paper_numbers.json')


def run(n_samples=500, seed=42):
    phi_df = load_usoskin_phi(os.path.join(DATA_DIR, 'phi_transit_frozen.csv'))
    traj   = generate_trajectory('2011-11-26', phi_df=phi_df)

    results = {}
    for sex, label in [('male', 'male'), ('female', 'female')]:
        print(f"\nRunning N={n_samples} ensemble — {label}, 35 y.o., 259-day, 16 g/cm² Al ...")
        t0 = time.time()
        res = run_uncertainty_ensemble(
            traj,
            shielding_x=16.0,
            material='aluminum',
            age=35,
            sex=sex,
            n_samples=n_samples,
            phi_df=phi_df,
            seed=seed,
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s  ({res['n_valid']}/{n_samples} valid)")

        pct = res['percentiles']
        vd  = res['variance_decomposition']

        print(f"\n  REID  p5={pct['REID']['p5']:.3%}  p50={pct['REID']['p50']:.3%}  p95={pct['REID']['p95']:.3%}")
        print(f"  D     p5={pct['D_mGy']['p5']:.1f}  p50={pct['D_mGy']['p50']:.1f}  p95={pct['D_mGy']['p95']:.1f} mGy")
        print(f"  H     p5={pct['H_mSv']['p5']:.1f}  p50={pct['H_mSv']['p50']:.1f}  p95={pct['H_mSv']['p95']:.1f} mSv")

        print(f"\n  Sensitivity indices (normalized ρ²) for {label}:")
        print(f"  {'Parameter':20s}  {'REID':>8s}  {'D':>8s}  {'H':>8s}")
        for p in PARAM_NAMES:
            r  = vd['REID'].get(p, 0.0)
            d  = vd['D_mGy'].get(p, 0.0)
            h  = vd['H_mSv'].get(p, 0.0)
            print(f"  {p:20s}  {r:8.3f}  {d:8.3f}  {h:8.3f}")

        phys_reid = sum(vd['REID'].get(p, 0.0) for p in PHYSICS_PARAMS)
        bio_reid  = sum(vd['REID'].get(p, 0.0) for p in BIO_PARAMS)
        print(f"\n  Physics fraction of REID variance: {phys_reid:.1%}")
        print(f"  Biology fraction of REID variance: {bio_reid:.1%}")

        results[sex] = {
            'percentiles': pct,
            'variance_decomposition': {k: dict(v) for k, v in vd.items()},
            'physics_reid_fraction': phys_reid,
            'biology_reid_fraction': bio_reid,
            'n_valid': res['n_valid'],
            'n_samples': n_samples,
        }

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_FILE}")
    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--samples', type=int, default=500)
    p.add_argument('--seed',    type=int, default=42)
    args = p.parse_args()
    run(args.samples, args.seed)
