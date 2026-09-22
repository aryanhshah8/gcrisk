"""Compare compare-ready TOPAS/OpenTOPAS-RBE reference CSVs against gcrisk.rbe."""

from __future__ import annotations

import argparse

import pandas as pd

from gcrisk.topas_benchmark import compare_topas_rbe_reference, enrich_topas_rbe_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Compare TOPAS/OpenTOPAS-RBE reference CSV exports against gcrisk.rbe outputs.'
    )
    parser.add_argument(
        'reference_csv',
        help=(
            'Compare-ready CSV containing dose_Gy, LETd_keV_um, alpha_beta_x_Gy, '
            'and TOPAS reference columns. This can be generated with '
            'scripts/run_topas_rbe_benchmark.py.'
        ),
    )
    parser.add_argument(
        '--write-enriched',
        help='Optional output CSV containing both reference and gcrisk-computed columns.',
    )
    parser.add_argument('--tolerance-rbe', type=float, default=0.02)
    parser.add_argument('--tolerance-dose-rbe', type=float, default=0.02)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    df = pd.read_csv(args.reference_csv)
    enriched = enrich_topas_rbe_frame(df)
    summaries = compare_topas_rbe_reference(
        enriched,
        tolerance_rbe=args.tolerance_rbe,
        tolerance_dose_rbe=args.tolerance_dose_rbe,
    )

    if args.write_enriched:
        enriched.to_csv(args.write_enriched, index=False)

    print('=' * 60)
    print('TOPAS / GCR RBE COMPARISON')
    print('=' * 60)
    all_pass = True
    for model, summary in summaries.items():
        all_pass &= summary.passed
        print(f'{model}')
        print(f'  n_cases:           {summary.n_cases}')
        print(f'  mae_rbe:           {summary.mae_rbe:.6f}')
        print(f'  max_abs_rbe:       {summary.max_abs_rbe:.6f}')
        print(f'  mae_dose_rbe:      {summary.mae_dose_rbe:.6f}')
        print(f'  max_abs_dose_rbe:  {summary.max_abs_dose_rbe:.6f}')
        print(f"  status:            {'PASS' if summary.passed else 'FAIL'}")
    print('=' * 60)
    print('ALL COMPARISONS PASSED' if all_pass else 'SOME COMPARISONS FAILED')
    print('=' * 60)
    return 0 if all_pass else 1


if __name__ == '__main__':
    raise SystemExit(main())
