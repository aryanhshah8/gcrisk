"""Run the repo-owned TOPAS proton-water benchmark and build a reference CSV."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from gcrisk.topas_benchmark import (
    build_topas_rbe_reference_frame,
    compare_topas_rbe_reference,
    load_alpha_beta_ratio,
    load_prescribed_dose,
)


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    benchmark_dir = repo_root / 'topas' / 'rbe_v79_proton_water'
    parser = argparse.ArgumentParser(
        description='Run the TOPAS/OpenTOPAS-RBE proton-water benchmark.'
    )
    parser.add_argument(
        '--topas-bin',
        default=shutil.which('topas'),
        help='Path to the TOPAS binary. Defaults to the first `topas` on PATH.',
    )
    parser.add_argument(
        '--benchmark-dir',
        default=str(benchmark_dir),
        help='Directory containing run.txt and related TOPAS parameter files.',
    )
    parser.add_argument(
        '--results-dir',
        default=None,
        help='Directory where TOPAS scorer CSVs are written. Defaults to <benchmark-dir>/results.',
    )
    parser.add_argument(
        '--reference-csv',
        default=str(repo_root / 'data' / 'topas' / 'proton_water_v79_reference.csv'),
        help='Output CSV path for the merged reference table.',
    )
    parser.add_argument(
        '--skip-run',
        action='store_true',
        help='Skip the TOPAS execution step and only rebuild the merged CSV from existing results.',
    )
    parser.add_argument('--tolerance-rbe', type=float, default=5e-6)
    parser.add_argument('--tolerance-dose-rbe', type=float, default=5e-6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    benchmark_dir = Path(args.benchmark_dir).resolve()
    results_dir = (
        Path(args.results_dir).resolve()
        if args.results_dir
        else benchmark_dir / 'results'
    )
    reference_csv = Path(args.reference_csv).resolve()

    if not benchmark_dir.exists():
        raise FileNotFoundError(f'Benchmark directory not found: {benchmark_dir}')
    if not args.skip_run and not args.topas_bin:
        raise FileNotFoundError('Unable to locate `topas`. Pass --topas-bin explicitly.')

    results_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_run:
        for csv_file in results_dir.glob('*.csv'):
            csv_file.unlink()
        subprocess.run(
            [args.topas_bin, 'run.txt'],
            cwd=benchmark_dir,
            check=True,
        )

    alpha_beta_x_Gy = load_alpha_beta_ratio(benchmark_dir / 'CellLineV79.txt')
    prescribed_dose_Gy = load_prescribed_dose(benchmark_dir / 'run.txt')

    reference_frame = build_topas_rbe_reference_frame(
        results_dir,
        alpha_beta_x_Gy=alpha_beta_x_Gy,
        prescribed_dose_Gy=prescribed_dose_Gy,
    )
    reference_csv.parent.mkdir(parents=True, exist_ok=True)
    reference_frame.to_csv(reference_csv, index=False)

    summaries = compare_topas_rbe_reference(
        reference_frame,
        tolerance_rbe=args.tolerance_rbe,
        tolerance_dose_rbe=args.tolerance_dose_rbe,
    )

    print('=' * 60)
    print('TOPAS RBE BENCHMARK')
    print('=' * 60)
    print(f'benchmark_dir:      {benchmark_dir}')
    print(f'results_dir:        {results_dir}')
    print(f'reference_csv:      {reference_csv}')
    print(f'prescribed_dose_Gy: {prescribed_dose_Gy:.6f}')
    print(f'alpha_beta_x_Gy:    {alpha_beta_x_Gy:.6f}')
    print('-' * 60)

    all_pass = True
    for model, summary in summaries.items():
        all_pass &= summary.passed
        print(f'{model}')
        print(f'  n_cases:           {summary.n_cases}')
        print(f'  mae_rbe:           {summary.mae_rbe:.8f}')
        print(f'  max_abs_rbe:       {summary.max_abs_rbe:.8f}')
        print(f'  mae_dose_rbe:      {summary.mae_dose_rbe:.8f}')
        print(f'  max_abs_dose_rbe:  {summary.max_abs_dose_rbe:.8f}')
        print(f"  status:            {'PASS' if summary.passed else 'FAIL'}")

    print('=' * 60)
    print('TOPAS BENCHMARK PASSED' if all_pass else 'TOPAS BENCHMARK FAILED')
    print('=' * 60)
    return 0 if all_pass else 1


if __name__ == '__main__':
    raise SystemExit(main())
