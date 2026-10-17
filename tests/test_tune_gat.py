"""Prevent repeat held-out evaluation even for the legacy GAT report."""
import json

import pytest

from train import tune_gat


def test_existing_final_refused_before_data_load_even_with_new_output(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_gat, 'REPO_ROOT', tmp_path)
    report = tmp_path / 'results/week08/gat_tuning.json'
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({'final_test_evaluation': {'test_accuracy': 0.822}}))
    monkeypatch.setattr(tune_gat, 'get_cora_data', lambda **kw: pytest.fail('held-out run restarted'))
    with pytest.raises(RuntimeError, match='already exists'):
        tune_gat.run_grid_search(output_dir=tmp_path / 'new-output')
    assert not (tmp_path / 'new-output').exists()


def test_failed_attempt_cannot_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_gat, 'REPO_ROOT', tmp_path)

    def fail(**kwargs):
        raise OSError('cache unavailable')

    monkeypatch.setattr(tune_gat, 'get_cora_data', fail)
    with pytest.raises(OSError, match='cache unavailable'):
        tune_gat.run_grid_search(output_dir=tmp_path / 'run')
    with pytest.raises(FileExistsError):
        tune_gat.run_grid_search(output_dir=tmp_path / 'run')
