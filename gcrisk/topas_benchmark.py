"""Helpers for building and comparing TOPAS/OpenTOPAS-RBE benchmark outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

from .rbe import mcnamara_rbe, wedenberg_rbe


REQUIRED_INPUT_COLUMNS = ('dose_Gy', 'LETd_keV_um', 'alpha_beta_x_Gy')
REFERENCE_COLUMNS = {
    'wedenberg': ('RBE_wedenberg_ref', 'dose_RBE_wedenberg_ref'),
    'mcnamara': ('RBE_mcnamara_ref', 'dose_RBE_mcnamara_ref'),
}
TOPAS_SCORER_FILES = {
    'physical_dose_Gy': 'PhysicalDose.csv',
    'LETd_keV_um': 'LET.csv',
    'RBE_wedenberg_ref': 'Wedenberg_RBE.csv',
    'RBE_mcnamara_ref': 'McNamara_RBE.csv',
}
_ALPHA_BETA_PATTERN = re.compile(
    r'AlphaBetaRatiox\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*Gy'
)
_PRESCRIBED_DOSE_PATTERN = re.compile(
    r'PrescribedDose\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*Gy'
)
_BIN_SIZE_PATTERN = re.compile(
    r'#\s+([XYZ])\s+in\s+\d+\s+bins\s+of\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+(\w+)'
)


@dataclass(frozen=True)
class BenchmarkSummary:
    """Compact summary for a single TOPAS/RBE model comparison."""

    model: str
    n_cases: int
    mae_rbe: float
    max_abs_rbe: float
    mae_dose_rbe: float
    max_abs_dose_rbe: float
    tolerance_rbe: float
    tolerance_dose_rbe: float

    @property
    def passed(self) -> bool:
        return (
            self.max_abs_rbe <= self.tolerance_rbe
            and self.max_abs_dose_rbe <= self.tolerance_dose_rbe
        )


def _validate_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')


def _extract_first_float(path: str | Path, pattern: re.Pattern[str]) -> float:
    text = Path(path).read_text()
    match = pattern.search(text)
    if not match:
        raise ValueError(f'Unable to find matching value in {path}.')
    return float(match.group(1))


def load_alpha_beta_ratio(path: str | Path) -> float:
    """Load `AlphaBetaRatiox` from a TOPAS/OpenTOPAS-RBE cell-line file."""
    return _extract_first_float(path, _ALPHA_BETA_PATTERN)


def load_prescribed_dose(path: str | Path) -> float:
    """Load `PrescribedDose` in Gy from a TOPAS/OpenTOPAS-RBE run file."""
    return _extract_first_float(path, _PRESCRIBED_DOSE_PATTERN)


def _parse_bin_sizes(header_lines: list[str]) -> dict[str, float]:
    bin_sizes: dict[str, float] = {}
    for line in header_lines:
        match = _BIN_SIZE_PATTERN.match(line.strip())
        if not match:
            continue
        axis = match.group(1).lower()
        size = float(match.group(2))
        unit = match.group(3)
        if unit == 'cm':
            bin_sizes[f'{axis}_bin_width_cm'] = size
        elif unit == 'mm':
            bin_sizes[f'{axis}_bin_width_cm'] = size / 10.0
    return bin_sizes


def read_topas_scorer_csv(path: str | Path, value_name: str) -> pd.DataFrame:
    """
    Read a TOPAS scorer CSV into a tidy frame.

    Returned columns:
    - `x_bin`
    - `y_bin`
    - `z_bin`
    - `<value_name>`
    - optional `x_depth_cm`, `y_depth_cm`, `z_depth_cm` when bin sizes are present
    """
    path = Path(path)
    header_lines: list[str] = []
    with path.open() as handle:
        for line in handle:
            if line.startswith('#'):
                header_lines.append(line.rstrip())
            else:
                break

    frame = pd.read_csv(
        path,
        comment='#',
        header=None,
        names=('x_bin', 'y_bin', 'z_bin', value_name),
    )

    bin_sizes = _parse_bin_sizes(header_lines)
    for axis in ('x', 'y', 'z'):
        key = f'{axis}_bin_width_cm'
        if key in bin_sizes:
            frame[f'{axis}_depth_cm'] = (frame[f'{axis}_bin'] + 0.5) * bin_sizes[key]

    return frame


def build_topas_rbe_reference_frame(
    results_dir: str | Path,
    *,
    alpha_beta_x_Gy: float,
    prescribed_dose_Gy: float,
    scorer_files: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Build a compare-ready reference frame from TOPAS/OpenTOPAS-RBE scorer outputs.

    Important:
    - `dose_Gy` is the prescribed dose used by the RBE model, not the scored
      physical dose profile in the water phantom.
    - The scored physical dose profile is preserved as `physical_dose_Gy`.
    """
    results_path = Path(results_dir)
    scorer_map = dict(TOPAS_SCORER_FILES if scorer_files is None else scorer_files)

    physical = read_topas_scorer_csv(
        results_path / scorer_map['physical_dose_Gy'],
        'physical_dose_Gy',
    )
    letd = read_topas_scorer_csv(
        results_path / scorer_map['LETd_keV_um'],
        'LETd_keV_um',
    )
    wedenberg = read_topas_scorer_csv(
        results_path / scorer_map['RBE_wedenberg_ref'],
        'RBE_wedenberg_ref',
    )
    mcnamara = read_topas_scorer_csv(
        results_path / scorer_map['RBE_mcnamara_ref'],
        'RBE_mcnamara_ref',
    )

    merge_keys = ['x_bin', 'y_bin', 'z_bin']
    for depth_col in ('x_depth_cm', 'y_depth_cm', 'z_depth_cm'):
        if depth_col in physical.columns:
            merge_keys.append(depth_col)

    frame = physical.merge(letd, on=merge_keys, how='inner')
    frame = frame.merge(wedenberg, on=merge_keys, how='inner')
    frame = frame.merge(mcnamara, on=merge_keys, how='inner')

    frame['alpha_beta_x_Gy'] = float(alpha_beta_x_Gy)
    frame['dose_Gy'] = float(prescribed_dose_Gy)
    frame['dose_RBE_wedenberg_ref'] = frame['dose_Gy'] * frame['RBE_wedenberg_ref']
    frame['dose_RBE_mcnamara_ref'] = frame['dose_Gy'] * frame['RBE_mcnamara_ref']
    return frame


def enrich_topas_rbe_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add package-computed Wedenberg and McNamara outputs to a TOPAS export frame.

    Expected input columns:
    - dose_Gy
    - LETd_keV_um
    - alpha_beta_x_Gy
    """
    _validate_columns(df, REQUIRED_INPUT_COLUMNS)
    enriched = df.copy()

    enriched['RBE_wedenberg_gcr'] = [
        wedenberg_rbe(dose, letd, alpha_beta)
        for dose, letd, alpha_beta in zip(
            enriched['dose_Gy'],
            enriched['LETd_keV_um'],
            enriched['alpha_beta_x_Gy'],
        )
    ]
    enriched['dose_RBE_wedenberg_gcr'] = (
        enriched['dose_Gy'] * enriched['RBE_wedenberg_gcr']
    )

    enriched['RBE_mcnamara_gcr'] = [
        mcnamara_rbe(dose, letd, alpha_beta)
        for dose, letd, alpha_beta in zip(
            enriched['dose_Gy'],
            enriched['LETd_keV_um'],
            enriched['alpha_beta_x_Gy'],
        )
    ]
    enriched['dose_RBE_mcnamara_gcr'] = (
        enriched['dose_Gy'] * enriched['RBE_mcnamara_gcr']
    )
    return enriched


def compare_topas_rbe_reference(
    df: pd.DataFrame,
    tolerance_rbe: float = 0.02,
    tolerance_dose_rbe: float = 0.02,
) -> dict[str, BenchmarkSummary]:
    """
    Compare exported TOPAS/OpenTOPAS-RBE reference columns against this package.

    The input frame must contain the base columns used by `enrich_topas_rbe_frame`.
    Here `dose_Gy` should be the prescribed dose used by the TOPAS/OpenTOPAS-RBE
    scorer when the reference outputs were generated.
    To compare a given model it must also include:

    - Wedenberg:
      - RBE_wedenberg_ref
      - dose_RBE_wedenberg_ref
    - McNamara:
      - RBE_mcnamara_ref
      - dose_RBE_mcnamara_ref
    """
    enriched = enrich_topas_rbe_frame(df)
    summaries: dict[str, BenchmarkSummary] = {}

    for model, (rbe_ref_col, dose_ref_col) in REFERENCE_COLUMNS.items():
        _validate_columns(enriched, (rbe_ref_col, dose_ref_col))
        rbe_model_col = f'RBE_{model}_gcr'
        dose_model_col = f'dose_RBE_{model}_gcr'

        rbe_err = (enriched[rbe_model_col] - enriched[rbe_ref_col]).abs()
        dose_err = (enriched[dose_model_col] - enriched[dose_ref_col]).abs()

        summaries[model] = BenchmarkSummary(
            model=model,
            n_cases=int(len(enriched)),
            mae_rbe=float(rbe_err.mean()),
            max_abs_rbe=float(rbe_err.max()),
            mae_dose_rbe=float(dose_err.mean()),
            max_abs_dose_rbe=float(dose_err.max()),
            tolerance_rbe=float(tolerance_rbe),
            tolerance_dose_rbe=float(tolerance_dose_rbe),
        )

    return summaries
