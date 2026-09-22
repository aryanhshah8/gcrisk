"""Regression checks for Wedenberg and McNamara RBE models against fixed reference values."""

from __future__ import annotations

from gcrisk.rbe import mcnamara_rbe, wedenberg_rbe


REFERENCE_CASES = [
    {
        'model': 'wedenberg',
        'dose_Gy': 2.0,
        'letd_keV_um': 5.0,
        'alpha_beta_x_Gy': 3.0,
        'expected_rbe': 1.2865411854416302,
    },
    {
        'model': 'wedenberg',
        'dose_Gy': 1.0,
        'letd_keV_um': 2.0,
        'alpha_beta_x_Gy': 2.0,
        'expected_rbe': 1.206354459283458,
    },
    {
        'model': 'mcnamara',
        'dose_Gy': 2.0,
        'letd_keV_um': 5.0,
        'alpha_beta_x_Gy': 3.0,
        'expected_rbe': 1.269537154977841,
    },
    {
        'model': 'mcnamara',
        'dose_Gy': 2.0,
        'letd_keV_um': 10.0,
        'alpha_beta_x_Gy': 3.0,
        'expected_rbe': 1.4631891207001457,
    },
]


def main() -> int:
    print("=" * 60)
    print("RBE MODEL VALIDATION")
    print("=" * 60)

    all_pass = True
    for case in REFERENCE_CASES:
        if case['model'] == 'wedenberg':
            result = wedenberg_rbe(
                case['dose_Gy'],
                case['letd_keV_um'],
                case['alpha_beta_x_Gy'],
            )
        else:
            result = mcnamara_rbe(
                case['dose_Gy'],
                case['letd_keV_um'],
                case['alpha_beta_x_Gy'],
            )

        error = abs(result - case['expected_rbe'])
        status = error < 1e-12
        all_pass &= status

        print(
            f"{case['model']:<10} dose={case['dose_Gy']:.1f} Gy, "
            f"LETd={case['letd_keV_um']:.1f} keV/um, "
            f"alpha/beta={case['alpha_beta_x_Gy']:.1f} Gy"
        )
        print(f"  expected: {case['expected_rbe']:.12f}")
        print(f"  result:   {result:.12f}")
        print(f"  status:   {'PASS' if status else 'FAIL'}")

    print("=" * 60)
    print("ALL RBE CHECKS PASSED" if all_pass else "SOME RBE CHECKS FAILED")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == '__main__':
    raise SystemExit(main())
