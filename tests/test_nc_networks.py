"""Network gradients, trainer integration, and held-out isolation."""
import json
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from models import GAT, GCN
from scripts.node_classification import fit_validation


@pytest.fixture
def graph():
    torch.manual_seed(42)
    return SimpleNamespace(
        x=torch.randn(8, 5), edge_index=torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]]),
        y=torch.tensor([0, 1, 2, 3, 4, 5, 6, 0]),
        train_mask=torch.tensor([True] * 4 + [False] * 4),
        val_mask=torch.tensor([False] * 4 + [True] * 2 + [False] * 2))


@pytest.mark.parametrize("kind", ["gcn", "gat"])
def test_network_wiring_and_backward(kind, graph):
    model = GCN(5, 4, 7, dropout=0) if kind == "gcn" else GAT(
        5, 4, 7, heads=3, output_heads=2, dropout=0, attention_dropout=0)
    model.eval()
    raw_edges = graph.edge_index.clone()
    logits = model(graph.x, graph.edge_index)
    hidden = model.conv1(graph.x, graph.edge_index)
    assert hidden.shape == (8, 4 if kind == "gcn" else 12)
    expected = model.conv2(F.relu(hidden) if kind == "gcn" else F.elu(hidden), raw_edges)
    torch.testing.assert_close(logits, expected)
    assert logits.shape == (8, 7) and torch.isfinite(logits).all()
    F.cross_entropy(logits, graph.y).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.equal(graph.edge_index, raw_edges)


@pytest.mark.parametrize("kind", ["gcn", "gat"])
def test_runner_ignores_heldout_labels_and_restores_checkpoint(kind, graph, tmp_path):
    config = {"seed": 42, "num_threads": 1, "max_epochs": 4, "patience": 2,
              "models": {kind: {"hidden_dim": 4, "dropout": 0.2, "learning_rate": 0.01,
                                 "weight_decay": 0.0005, "num_heads": 2, "output_heads": 2,
                                 "attention_dropout": 0.2}}}
    first = fit_validation(graph, kind, config, tmp_path / "first", {})
    graph.y[6:] = -999  # No test_mask exists; poison unused labels.
    second = fit_validation(graph, kind, config, tmp_path / "second", {})
    for key in ("history", "best_epoch", "validation_score", "split_hash", "input_hash"):
        assert first[key] == second[key]
    assert first["checkpoint_roundtrip"] and first["test_score"] is None
    log = json.loads((tmp_path / "first/training.json").read_text())
    assert log["status"] == "fitted" and "test_accuracy" not in log
    with pytest.raises(FileExistsError):
        fit_validation(graph, kind, config, tmp_path / "first", {})
