"""Authorized repeat preserves original selection and exports auditable artifacts."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import recover_gat_final as recovery


@pytest.fixture
def setup(tmp_path, monkeypatch):
    data = SimpleNamespace(x=torch.randn(9, 5), y=torch.arange(9) % 7,
                           edge_index=torch.tensor([[0, 1], [1, 0]]),
                           train_mask=torch.arange(9) < 4,
                           val_mask=(torch.arange(9) >= 4) & (torch.arange(9) < 7),
                           test_mask=torch.arange(9) >= 7)
    best = dict(config_id=7, hidden_dim=4, heads=2, dropout=0.4,
                learning_rate=0.005, weight_decay=0.0005, best_epoch=1,
                stopped_epoch=1, best_val_loss=0.7, val_accuracy=0.8, val_macro_f1=0.79)
    report = tmp_path / 'original.json'
    report.write_text(json.dumps(dict(seed=42, configurations=[best], best_configuration=best,
                                     final_test_evaluation=dict(test_accuracy=0.8, test_macro_f1=0.79))))

    def fit(model, x, edges, labels, train, val, *args, **kwargs):
        assert torch.all(labels[~(train | val)] == -999)
        return {key: best[key] for key in ('best_epoch', 'stopped_epoch', 'best_val_loss',
                                           'val_accuracy', 'val_macro_f1')}, model.state_dict()

    monkeypatch.setattr(recovery, 'fit_gat', fit)
    return data, report, tmp_path / 'out', tmp_path / 'claim.json'


def test_artifacts_and_cross_output_repeat_guard(setup):
    data, report, output, claim = setup
    result = recovery.recover(data, report, output, {}, claim)
    saved = np.load(output / 'predictions.npz')
    assert result['protocol_exception'] is True
    assert result['test_score']['accuracy'] == (saved['y_true'] == saved['y_pred']).mean()
    assert saved['node_id'].tolist() == [7, 8]
    assert result['original_report_sha256'] == recovery.digest(report)
    with pytest.raises(FileExistsError):
        recovery.recover(data, report, output.with_name('other'), {}, claim)


def test_changed_validation_refuses_test(setup, monkeypatch):
    data, report, output, claim = setup
    original_fit = recovery.fit_gat

    def different(*args, **kwargs):
        summary, state = original_fit(*args, **kwargs)
        summary['best_val_loss'] += 0.1
        return summary, state

    monkeypatch.setattr(recovery, 'fit_gat', different)
    monkeypatch.setattr(recovery, 'evaluate_node_classification',
                        lambda *args: pytest.fail('Test accessed before validation matched'))
    with pytest.raises(ValueError, match='no test evaluated'):
        recovery.recover(data, report, output, {}, claim)
    assert not (output / 'predictions.npz').exists()
