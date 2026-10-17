"""Tests for the complete GAT node classification model."""

import pytest
import torch
import torch.nn as nn

from models.gat import GAT


def test_legacy_dropout_alias_preserves_attention_setting():
    model = GAT(hidden_dim=16, heads=8, dropout=0.4, feature_dropout=0.2)
    assert model.conv1.dropout == model.conv2.dropout == 0.4
    assert model.dropout.p == 0.2
    explicit = GAT(hidden_channels=16, dropout=0.2, attention_dropout=0.6)
    assert explicit.conv1.dropout == explicit.conv2.dropout == 0.6
    assert explicit.dropout.p == 0.2


def test_gat_forward_shape():
    """Verify output logits shape matches [num_nodes, num_classes]."""
    num_nodes = 10
    in_channels = 16
    hidden_dim = 4
    num_classes = 3
    heads = 2

    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]],
        dtype=torch.long,
    )

    model = GAT(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        heads=heads,
        dropout=0.0,
        feature_dropout=0.0,
    )
    model.eval()
    out = model(x, edge_index)

    assert out.shape == (num_nodes, num_classes)
    assert torch.isfinite(out).all()


def test_gat_backward_and_gradients():
    """Verify backward pass populates non-None, finite gradients for all parameters."""
    num_nodes = 6
    x = torch.randn(num_nodes, 8, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)

    model = GAT(
        in_channels=8,
        hidden_dim=4,
        num_classes=2,
        heads=2,
        dropout=0.0,
        feature_dropout=0.0,
    )

    out = model(x, edge_index)
    torch.manual_seed(0)
    target_weights = torch.randn_like(out)
    loss = (out * target_weights).sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has None gradient"
        assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient"
        assert param.grad.abs().sum() > 0, f"Parameter {name} has all-zero gradient"


def test_gat_overfit_small_graph():
    """Verify that GAT model can overfit and minimize Cross-Entropy loss on a toy graph."""
    torch.manual_seed(42)
    num_nodes = 12
    in_channels = 8
    num_classes = 3

    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]],
        dtype=torch.long,
    )
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.long)

    model = GAT(
        in_channels=in_channels,
        hidden_dim=8,
        num_classes=num_classes,
        heads=2,
        dropout=0.0,
        feature_dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    criterion = nn.CrossEntropyLoss()

    initial_loss = None
    final_loss = None

    for epoch in range(80):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert final_loss < initial_loss * 0.65, f"Loss failed to decrease sufficiently: {initial_loss} -> {final_loss}"


def test_gat_return_attention():
    """Verify return_attention flag returns second layer attention weights."""
    num_nodes = 5
    x = torch.randn(num_nodes, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

    model = GAT(
        in_channels=8,
        hidden_dim=4,
        num_classes=2,
        heads=2,
        dropout=0.0,
        feature_dropout=0.0,
    )
    model.eval()

    out, (edge_index_out, alpha) = model(x, edge_index, return_attention=True)
    assert out.shape == (num_nodes, 2)
    assert alpha.ndim == 2
    assert alpha.shape[0] == edge_index_out.shape[1]
