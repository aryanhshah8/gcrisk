"""Tests for the command-line interface."""

import json

from gcrisk.cli import main


def test_cli_mission_text_output(capsys):
    """Plain-text mission CLI should print the mission and organ-risk summary."""
    exit_code = main([
        'mission',
        '2011-11-26',
        '--surface-days', '5',
        '--shielding-x-gcm2', '16',
        '--habitat-shielding-x-gcm2', '5',
    ])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert 'FULL MARS MISSION' in captured.out
    assert 'ORGAN RISK SUMMARY' in captured.out


def test_cli_mission_json_output(capsys):
    """JSON mode should emit a compact machine-readable payload."""
    exit_code = main([
        'mission',
        '2011-11-26',
        '--surface-days', '5',
        '--json',
    ])
    captured = capsys.readouterr()

    payload = json.loads(captured.out)
    assert exit_code == 0
    assert 'mission_profile' in payload
    assert 'totals' in payload
    assert 'organ_risk' in payload


def test_cli_rbe_json_output(capsys):
    """RBE CLI should emit a machine-readable result payload."""
    exit_code = main([
        'rbe',
        '--dose-gy', '2.0',
        '--letd-kev-um', '5.0',
        '--alpha-beta-gy', '3.0',
        '--model', 'mcnamara',
        '--json',
    ])
    captured = capsys.readouterr()

    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload['model'] == 'mcnamara'
    assert payload['RBE'] > 1.0
