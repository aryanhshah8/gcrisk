"""Tests for TOPAS/OpenTOPAS-RBE comparison helpers."""

import pandas as pd
import pytest
from pathlib import Path

from gcrisk.topas_benchmark import (
    build_topas_rbe_reference_frame,
    compare_topas_rbe_reference,
    enrich_topas_rbe_frame,
    load_alpha_beta_ratio,
    load_prescribed_dose,
    read_topas_scorer_csv,
)


def test_enrich_topas_rbe_frame_adds_expected_columns():
    df = pd.DataFrame(
        {
            'dose_Gy': [2.0, 1.0],
            'LETd_keV_um': [5.0, 2.0],
            'alpha_beta_x_Gy': [3.0, 2.0],
        }
    )
    enriched = enrich_topas_rbe_frame(df)
    for column in (
        'RBE_wedenberg_gcr',
        'dose_RBE_wedenberg_gcr',
        'RBE_mcnamara_gcr',
        'dose_RBE_mcnamara_gcr',
    ):
        assert column in enriched.columns


def test_compare_topas_rbe_reference_passes_for_matching_reference():
    df = pd.DataFrame(
        {
            'dose_Gy': [2.0, 1.0],
            'LETd_keV_um': [5.0, 2.0],
            'alpha_beta_x_Gy': [3.0, 2.0],
        }
    )
    enriched = enrich_topas_rbe_frame(df)
    enriched['RBE_wedenberg_ref'] = enriched['RBE_wedenberg_gcr']
    enriched['dose_RBE_wedenberg_ref'] = enriched['dose_RBE_wedenberg_gcr']
    enriched['RBE_mcnamara_ref'] = enriched['RBE_mcnamara_gcr']
    enriched['dose_RBE_mcnamara_ref'] = enriched['dose_RBE_mcnamara_gcr']

    summary = compare_topas_rbe_reference(enriched, tolerance_rbe=1e-12, tolerance_dose_rbe=1e-12)
    assert summary['wedenberg'].passed
    assert summary['mcnamara'].passed


def test_compare_topas_rbe_reference_requires_reference_columns():
    df = pd.DataFrame(
        {
            'dose_Gy': [2.0],
            'LETd_keV_um': [5.0],
            'alpha_beta_x_Gy': [3.0],
        }
    )
    with pytest.raises(ValueError):
        compare_topas_rbe_reference(df)


def test_load_alpha_beta_and_prescribed_dose(tmp_path):
    cell_line = tmp_path / 'CellLine.txt'
    cell_line.write_text('d:Sc/Test/AlphaBetaRatiox = 3.0 Gy\n')
    run_file = tmp_path / 'run.txt'
    run_file.write_text('d:Sc/PrescribedDose = 2.0 Gy\n')

    assert load_alpha_beta_ratio(cell_line) == 3.0
    assert load_prescribed_dose(run_file) == 2.0


def test_build_topas_rbe_reference_frame_from_scorer_csvs(tmp_path):
    header = (
        '# TOPAS Version: 4.2.p3\n'
        '# Z in 2 bins of 0.1 cm\n'
    )
    scorer_rows = '0, 0, 0, {a}\n0, 0, 1, {b}\n'

    (tmp_path / 'PhysicalDose.csv').write_text(header + scorer_rows.format(a=1.0e-5, b=2.0e-5))
    (tmp_path / 'LET.csv').write_text(header + scorer_rows.format(a=1.0, b=2.0))
    (tmp_path / 'Wedenberg_RBE.csv').write_text(header + scorer_rows.format(a=1.1, b=1.2))
    (tmp_path / 'McNamara_RBE.csv').write_text(header + scorer_rows.format(a=1.3, b=1.4))

    frame = build_topas_rbe_reference_frame(
        tmp_path,
        alpha_beta_x_Gy=3.0,
        prescribed_dose_Gy=2.0,
    )

    assert list(frame['dose_Gy']) == [2.0, 2.0]
    assert list(frame['physical_dose_Gy']) == [1.0e-5, 2.0e-5]
    assert list(frame['LETd_keV_um']) == [1.0, 2.0]
    assert list(frame['dose_RBE_wedenberg_ref']) == [2.2, 2.4]
    assert 'z_depth_cm' in frame.columns


def test_read_topas_scorer_csv_exposes_depth_bins(tmp_path):
    scorer = tmp_path / 'LET.csv'
    scorer.write_text(
        '# TOPAS Version: 4.2.p3\n'
        '# Z in 3 bins of 0.2 cm\n'
        '0, 0, 0, 1.0\n'
        '0, 0, 1, 2.0\n'
        '0, 0, 2, 3.0\n'
    )

    frame = read_topas_scorer_csv(scorer, 'LETd_keV_um')
    assert list(frame['z_depth_cm']) == pytest.approx([0.1, 0.3, 0.5])


def test_compare_topas_rbe_reference_passes_for_real_topas_reference():
    reference_csv = (
        Path(__file__).resolve().parents[1]
        / 'data'
        / 'topas'
        / 'proton_water_v79_reference.csv'
    )
    if not reference_csv.exists():
        pytest.skip('TOPAS reference CSV has not been generated yet.')

    df = pd.read_csv(reference_csv)
    summary = compare_topas_rbe_reference(df, tolerance_rbe=5e-6, tolerance_dose_rbe=5e-6)
    assert summary['wedenberg'].passed
    assert summary['mcnamara'].passed
