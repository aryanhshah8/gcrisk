"""Tests for scripts/paper2/ingest_clinical_rbe.py — schema enforcement only."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    'paper2_ingest_rbe',
    _REPO / 'scripts' / 'paper2' / 'ingest_clinical_rbe.py',
)
ingest_mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(ingest_mod)


def _good_frame() -> pd.DataFrame:
    return pd.DataFrame({
        'doi': ['10.1000/fake.1', '10.1000/fake.2'],
        'source': ['pide', 'paganetti2014'],
        'cell_line': ['V79', 'CHO'],
        'alpha_beta_Gy': [2.8, 3.0],
        'LET_keV_um': [7.5, 12.1],
        'endpoint': ['10pct_survival', 'clonogenic'],
        'RBE': [1.15, 1.24],
        'RBE_unc': [0.05, 0.08],
    })


def test_validate_frame_accepts_good_input():
    out = ingest_mod.validate_frame(_good_frame(), label='good.csv')
    assert list(out.columns)[:7] == list(ingest_mod.REQUIRED_COLUMNS)
    assert 'RBE_unc' in out.columns
    assert len(out) == 2


def test_validate_frame_rejects_missing_doi():
    bad = _good_frame()
    bad.loc[0, 'doi'] = ''
    with pytest.raises(ingest_mod.IngestError, match='DOI'):
        ingest_mod.validate_frame(bad, label='bad.csv')


def test_validate_frame_rejects_unknown_source():
    bad = _good_frame()
    bad.loc[0, 'source'] = 'blog_post'
    with pytest.raises(ingest_mod.IngestError, match='source'):
        ingest_mod.validate_frame(bad, label='bad.csv')


def test_validate_frame_rejects_nonphysical_rbe():
    bad = _good_frame()
    bad.loc[1, 'RBE'] = -0.5
    with pytest.raises(ingest_mod.IngestError, match='nonphysical'):
        ingest_mod.validate_frame(bad, label='bad.csv')


def test_validate_frame_rejects_missing_required_column():
    bad = _good_frame().drop(columns=['endpoint'])
    with pytest.raises(ingest_mod.IngestError, match='missing required'):
        ingest_mod.validate_frame(bad, label='bad.csv')


def test_validate_frame_rejects_non_numeric_let():
    bad = _good_frame()
    bad['LET_keV_um'] = ['low', 'mid']
    with pytest.raises(ingest_mod.IngestError, match='numeric'):
        ingest_mod.validate_frame(bad, label='bad.csv')


def test_assert_not_model_output_blocks_topas():
    topas_path = _REPO / 'data' / 'topas'
    if not topas_path.is_dir():
        pytest.skip('data/topas not present in this checkout')
    # Pick any file in data/topas; a valid directory with files.
    candidates = list(topas_path.glob('*.csv'))
    if not candidates:
        pytest.skip('no files in data/topas to test against')
    with pytest.raises(ingest_mod.IngestError, match='model outputs'):
        ingest_mod._assert_not_model_output(candidates[0])


def test_ingest_is_noop_when_directory_empty(tmp_path, monkeypatch):
    # Point the ingestor at an empty temp dir.
    monkeypatch.setattr(ingest_mod, 'CLINICAL_DIR', tmp_path)
    monkeypatch.setattr(ingest_mod, 'OUTPUT_CSV',
                        tmp_path / 'clinical_rbe_normalized.csv')
    monkeypatch.setattr(ingest_mod, 'MANIFEST_JSON',
                        tmp_path / 'clinical_rbe_manifest.json')
    res = ingest_mod.ingest(verbose=False)
    assert res['n_files'] == 0
    assert res['n_rows'] == 0
    assert not (tmp_path / 'clinical_rbe_normalized.csv').exists()


def test_ingest_happy_path(tmp_path, monkeypatch):
    (tmp_path / 'sample_a.csv').write_text(
        _good_frame().to_csv(index=False)
    )
    monkeypatch.setattr(ingest_mod, 'CLINICAL_DIR', tmp_path)
    monkeypatch.setattr(ingest_mod, 'OUTPUT_CSV',
                        tmp_path / 'clinical_rbe_normalized.csv')
    monkeypatch.setattr(ingest_mod, 'MANIFEST_JSON',
                        tmp_path / 'clinical_rbe_manifest.json')
    res = ingest_mod.ingest(verbose=False)
    assert res['n_files'] == 1
    assert res['n_rows'] == 2
    assert (tmp_path / 'clinical_rbe_normalized.csv').exists()
    assert (tmp_path / 'clinical_rbe_manifest.json').exists()
    normalized = pd.read_csv(tmp_path / 'clinical_rbe_normalized.csv')
    assert len(normalized) == 2
    assert set(ingest_mod.REQUIRED_COLUMNS).issubset(normalized.columns)
