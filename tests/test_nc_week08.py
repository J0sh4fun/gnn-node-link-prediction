"""Synthetic held-out isolation and durable final-use protection."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models import GCN
from scripts.nc_week08 import tune, final_gcn


def graph():
    torch.manual_seed(7)
    return SimpleNamespace(x=torch.randn(9, 5),
                           edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
                           y=torch.arange(9) % 7,
                           train_mask=torch.arange(9) < 4,
                           val_mask=(torch.arange(9) >= 4) & (torch.arange(9) < 7),
                           test_mask=torch.arange(9) >= 7)


def protocol():
    return dict(seed=42, num_threads=1, max_epochs=3, patience=2,
                base=dict(hidden_dim=4, num_layers=2, dropout=0.2,
                          learning_rate=0.01, weight_decay=0.0005),
                candidates=[{}, {'num_layers': 3}])


@pytest.mark.parametrize('depth', [2, 3, 4])
def test_depth_gradients(depth):
    data = graph()
    model = GCN(5, 4, 7, num_layers=depth)
    model(data.x, data.edge_index).square().sum().backward()
    assert len(model.hidden_convs) == depth - 2
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_search_isolation_and_final_once(tmp_path):
    data = graph()
    first = tune(data, protocol(), tmp_path / 'one', {})
    data.y[data.test_mask] = -999
    second = tune(data, protocol(), tmp_path / 'two', {})
    assert first == second
    assert first['test_score'] is None
    data.y[data.test_mask] = torch.tensor([0, 1])
    result = final_gcn(data, tmp_path / 'one', tmp_path / 'final', {})
    saved = np.load(tmp_path / 'final/predictions.npz')
    assert result['test_score']['accuracy'] == float((saved['y_true'] == saved['y_pred']).mean())
    assert saved['node_id'].tolist() == [7, 8]
    with pytest.raises(FileExistsError):
        final_gcn(data, tmp_path / 'one', tmp_path / 'different_output', {})


def test_tampered_checkpoint_and_graph_refused_before_claim(tmp_path):
    data = graph()
    tune(data, protocol(), tmp_path / 'run', {})
    data.x[0, 0] += 1
    with pytest.raises(ValueError, match='changed'):
        final_gcn(data, tmp_path / 'run', tmp_path / 'final', {})
    assert not (tmp_path / 'run/final_claim.json').exists()
    selected = json.loads((tmp_path / 'run/selection.json').read_text())
    (tmp_path / 'run' / selected['checkpoint']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        final_gcn(graph(), tmp_path / 'run', tmp_path / 'final', {})
