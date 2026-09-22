"""Command-line entrypoints for the GCR dosimetry pipeline."""

from __future__ import annotations

import argparse
import json

from .mission import mission_dose_summary, run_full_mission
from .rbe import rbe_weighted_dose


def _mission_report_payload(result: dict) -> dict:
    """Build a compact serializable mission report."""
    organ_h = result['organ_risk']['organ_H_mSv']
    top_organs = sorted(
        organ_h.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:5]

    return {
        'mission_profile': {
            'launch_date': result['mission_profile']['launch_date'],
            'surface_days': int(result['total']['surface_days']),
            'transit_days': int(result['total']['transit_days']),
            'shielding_transit_gcm2': float(result['mission_profile']['shielding_transit_gcm2']),
            'shielding_material': result['mission_profile']['shielding_material'],
            'habitat_shielding_gcm2': float(result['mission_profile']['habitat_shielding_gcm2']),
            'age': int(result['mission_profile']['age']),
            'sex': result['mission_profile']['sex'],
        },
        'totals': {
            'D_total_mGy': float(result['total']['D_total_mGy']),
            'H_total_mSv': float(result['total']['H_total_mSv']),
            'E_effective_mSv': float(result['total']['E_effective_mSv']),
        },
        'organ_risk': {
            'REID_total': float(result['organ_risk']['REID_total']),
            'exceeds_limit_fraction': float(result['organ_risk']['exceeds_limit_fraction']),
            'effective_H_mSv': float(result['organ_risk']['effective_H_mSv']),
            'top_organs_by_H_mSv': [
                {'organ': organ, 'H_mSv': float(h_msv)}
                for organ, h_msv in top_organs
            ],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog='gcr-dose',
        description=(
            'Mission-scale GCR dosimetry and organ-risk reporting for Mars '
            'mission scenarios.'
        ),
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    mission = subparsers.add_parser(
        'mission',
        help='Run a full mission assessment and print a report.',
    )
    mission.add_argument('launch_date', help='Earth departure date in YYYY-MM-DD format.')
    mission.add_argument('--surface-days', type=int, default=500)
    mission.add_argument('--shielding-x-gcm2', type=float, default=20.0)
    mission.add_argument('--habitat-shielding-x-gcm2', type=float, default=10.0)
    mission.add_argument('--shielding-material', default='aluminum')
    mission.add_argument('--age', type=int, default=35)
    mission.add_argument('--sex', default='male', choices=['male', 'female'])
    mission.add_argument(
        '--json',
        action='store_true',
        help='Print a compact JSON report instead of plain text.',
    )

    rbe = subparsers.add_parser(
        'rbe',
        help='Evaluate a proton-style LET/RBE model for a given dose and LETd.',
    )
    rbe.add_argument('--dose-gy', type=float, required=True)
    rbe.add_argument('--letd-kev-um', type=float, required=True)
    rbe.add_argument('--alpha-beta-gy', type=float, required=True)
    rbe.add_argument('--model', default='wedenberg', choices=['wedenberg', 'mcnamara'])
    rbe.add_argument(
        '--json',
        action='store_true',
        help='Print JSON instead of plain text.',
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == 'mission':
        result = run_full_mission(
            launch_date=args.launch_date,
            surface_days=args.surface_days,
            shielding_x_gcm2=args.shielding_x_gcm2,
            shielding_material=args.shielding_material,
            habitat_shielding_x_gcm2=args.habitat_shielding_x_gcm2,
            age=args.age,
            sex=args.sex,
        )
        if args.json:
            print(json.dumps(_mission_report_payload(result), indent=2))
        else:
            print(mission_dose_summary(result))
            print('')
            print('ORGAN RISK SUMMARY')
            print(f"  REID_total:          {result['organ_risk']['REID_total'] * 100:.2f}%")
            print(
                '  exceeds_limit_frac:  '
                f"{result['organ_risk']['exceeds_limit_fraction'] * 100:.1f}%"
            )
            print('  top organs by H:')
            for item in _mission_report_payload(result)['organ_risk']['top_organs_by_H_mSv']:
                print(f"    {item['organ']:<10} {item['H_mSv']:.1f} mSv")
        return 0

    if args.command == 'rbe':
        result = rbe_weighted_dose(
            dose_Gy=args.dose_gy,
            letd_keV_um=args.letd_kev_um,
            alpha_beta_x_Gy=args.alpha_beta_gy,
            model=args.model,
        )
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print('RBE REPORT')
            print(f"  model:            {result['model']}")
            print(f"  dose:             {result['dose_Gy']:.3f} Gy")
            print(f"  LETd:             {result['LETd_keV_um']:.3f} keV/um")
            print(f"  alpha/beta_x:     {result['alpha_beta_x_Gy']:.3f} Gy")
            print(f"  RBE:              {result['RBE']:.3f}")
            print(f"  RBE-weighted dose:{result['dose_RBE_Gy']:.3f} Gy(RBE)")
        return 0

    parser.error(f'Unknown command: {args.command}')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
