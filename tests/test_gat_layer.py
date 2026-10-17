"""Unit and integration tests for custom GATLayer implementation.

Tests follow the Acceptance Criteria in docs/plans/project_plan.md (Section 5)
and the verification checklist in report/week01/member_b.md (Section 11).
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from models.gat_layer import GATLayer


def test_gat_layer_forward_shape_concat():
    """Verify output shape when concat=True: [N, heads * out_channels]."""
    num_nodes = 6
    in_channels = 16
    out_channels = 8
    heads = 4
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 1, 2, 3], [1, 2, 3, 4, 0, 0, 1, 2]], dtype=torch.long
    )

    layer = GATLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        heads=heads,
        concat=True,
        dropout=0.0,
    )
    out = layer(x, edge_index)

    assert out.shape == (num_nodes, heads * out_channels)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_gat_layer_forward_shape_average():
    """Verify output shape when concat=False (averaging heads): [N, out_channels]."""
    num_nodes = 6
    in_channels = 16
    out_channels = 7
    heads = 4
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 1, 2, 3], [1, 2, 3, 4, 0, 0, 1, 2]], dtype=torch.long
    )

    layer = GATLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        heads=heads,
        concat=False,
        dropout=0.0,
    )
    out = layer(x, edge_index)

    assert out.shape == (num_nodes, out_channels)
    assert torch.isfinite(out).all()


def test_gat_layer_backward_and_finite_gradients():
    """Verify all parameters receive non-None, finite, non-zero gradients."""
    num_nodes = 5
    in_channels = 8
    out_channels = 4
    heads = 2
    x = torch.randn(num_nodes, in_channels, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    layer = GATLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        heads=heads,
        concat=True,
        bias=True,
        dropout=0.0,
    )
    out = layer(x, edge_index)
    loss = out.sum()
    loss.backward()

    # Feature input gradient
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    # Parameter gradients
    assert layer.lin.weight.grad is not None
    assert torch.isfinite(layer.lin.weight.grad).all()
    assert layer.lin.weight.grad.abs().sum() > 0

    assert layer.att_src.grad is not None
    assert torch.isfinite(layer.att_src.grad).all()
    assert layer.att_src.grad.abs().sum() > 0

    assert layer.att_dst.grad is not None
    assert torch.isfinite(layer.att_dst.grad).all()
    assert layer.att_dst.grad.abs().sum() > 0

    assert layer.bias is not None
    assert layer.bias.grad is not None
    assert torch.isfinite(layer.bias.grad).all()
    assert layer.bias.grad.abs().sum() > 0


def test_gat_layer_no_nan_or_inf():
    """Test robustness against large inputs, ensuring no NaN or Inf."""
    num_nodes = 4
    x = torch.randn(num_nodes, 8) * 10.0
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    layer = GATLayer(in_channels=8, out_channels=4, heads=2)
    out = layer(x, edge_index)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_gat_layer_attention_weights_properties():
    """Test that attention weights per target node and per head sum to 1.0."""
    num_nodes = 4
    x = torch.randn(num_nodes, 8)
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=torch.long)

    layer = GATLayer(
        in_channels=8,
        out_channels=4,
        heads=3,
        add_self_loops=True,
        dropout=0.0,
    )
    layer.eval()

    out, (edge_index_out, alpha) = layer(x, edge_index, return_attention_weights=True)

    # Check alpha shape: [E_with_loops, heads]
    assert alpha.ndim == 2
    assert alpha.shape[0] == edge_index_out.shape[1]
    assert alpha.shape[1] == 3
    assert (alpha >= 0.0).all()

    # Sum of attention weights into each target node must equal 1.0 for each head
    targets = edge_index_out[1]
    for node in range(num_nodes):
        mask = targets == node
        assert mask.any(), f"Node {node} must have incoming edges (at least self-loop)"
        sum_per_head = alpha[mask].sum(dim=0)
        torch.testing.assert_close(
            sum_per_head,
            torch.ones(3, dtype=alpha.dtype),
            atol=1e-6,
            rtol=1e-5,
        )


def test_gat_layer_analytical_three_node_example():
    """Verify analytical calculation from report/week01/member_b.md (Section 6).

    Graph: 3 nodes (0, 1, 2), undirected edges {0,1}, {0,2} + self-loops.
    h0 = [1, 0], h1 = [0, 1], h2 = [1, 1].
    W = I_2, K=1, out_channels=2, bias=False, dropout=0.0.
    a_dst = [1, -1], a_src = [-1, 2].
    Negative slope = 0.2.

    Analytical expectation for target node 0:
    e_00 = 0, e_01 = 3, e_02 = 2.
    alpha_00 ≈ 0.035119, alpha_01 ≈ 0.705385, alpha_02 ≈ 0.259496.
    h'_0 ≈ [0.294615, 0.964881].
    """
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    # Undirected edges {0,1}, {0,2} (4 directed edges: 0->1, 1->0, 0->2, 2->0)
    edge_index = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=torch.long)

    layer = GATLayer(
        in_channels=2,
        out_channels=2,
        heads=1,
        concat=True,
        negative_slope=0.2,
        dropout=0.0,
        bias=False,
        add_self_loops=True,
    )
    layer.eval()

    # Force manual weights to match analytical setup
    with torch.no_grad():
        layer.lin.weight.copy_(torch.eye(2))
        layer.att_dst.copy_(torch.tensor([[[1.0, -1.0]]]))
        layer.att_src.copy_(torch.tensor([[[-1.0, 2.0]]]))

    out, (edge_index_out, alpha) = layer(x, edge_index, return_attention_weights=True)

    # Locate incoming edges to target node 0
    targets = edge_index_out[1]
    sources = edge_index_out[0]

    mask_00 = (targets == 0) & (sources == 0)
    mask_01 = (targets == 0) & (sources == 1)
    mask_02 = (targets == 0) & (sources == 2)

    alpha_00 = alpha[mask_00, 0].item()
    alpha_01 = alpha[mask_01, 0].item()
    alpha_02 = alpha[mask_02, 0].item()

    # Verify attention coefficients
    assert pytest.approx(alpha_00, abs=1e-5) == 0.035119
    assert pytest.approx(alpha_01, abs=1e-5) == 0.705385
    assert pytest.approx(alpha_02, abs=1e-5) == 0.259496

    # Verify updated node representation h'_0
    h_prime_0 = out[0].tolist()
    assert pytest.approx(h_prime_0[0], abs=1e-5) == 0.294615
    assert pytest.approx(h_prime_0[1], abs=1e-5) == 0.964881


def test_gat_layer_edge_order_permutation_invariance():
    """Verify that permuting edges in edge_index produces identical output."""
    torch.manual_seed(42)
    num_nodes = 8
    x = torch.randn(num_nodes, 10)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 1, 3, 5], [1, 2, 3, 4, 5, 6, 7, 0, 0, 1, 2]],
        dtype=torch.long,
    )

    layer = GATLayer(in_channels=10, out_channels=6, heads=3, dropout=0.0)
    layer.eval()

    out_orig = layer(x, edge_index)

    # Permute edge columns
    perm = torch.randperm(edge_index.size(1))
    edge_index_perm = edge_index[:, perm]

    out_perm = layer(x, edge_index_perm)

    torch.testing.assert_close(out_orig, out_perm, atol=1e-6, rtol=1e-5)


def test_gat_layer_isolated_node_handling():
    """Verify that isolated nodes are handled cleanly without error or NaN."""
    num_nodes = 4
    x = torch.randn(num_nodes, 8)
    # Only connect nodes 0 and 1; nodes 2 and 3 are isolated
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    layer = GATLayer(
        in_channels=8,
        out_channels=4,
        heads=2,
        add_self_loops=True,
    )
    layer.eval()

    out = layer(x, edge_index)
    assert out.shape == (num_nodes, 8)
    assert torch.isfinite(out).all()


def test_gat_layer_comparable_to_pyg_gatconv():
    """Verify numerical agreement with PyG's official GATConv on equivalent weights."""
    torch.manual_seed(123)
    num_nodes = 5
    in_channels = 6
    out_channels = 4
    heads = 2
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)

    our_layer = GATLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        heads=heads,
        concat=True,
        negative_slope=0.2,
        dropout=0.0,
        bias=True,
        add_self_loops=True,
    )

    pyg_layer = GATConv(
        in_channels=in_channels,
        out_channels=out_channels,
        heads=heads,
        concat=True,
        negative_slope=0.2,
        dropout=0.0,
        bias=True,
        add_self_loops=True,
    )

    our_layer.eval()
    pyg_layer.eval()

    # Align weights between our implementation and PyG GATConv
    with torch.no_grad():
        pyg_layer.lin.weight.copy_(our_layer.lin.weight)
        if pyg_layer.bias is not None and our_layer.bias is not None:
            pyg_layer.bias.copy_(our_layer.bias)
        # In PyG GATConv: att_src is [1, heads, out_channels], att_dst is [1, heads, out_channels]
        pyg_layer.att_src.copy_(our_layer.att_src)
        pyg_layer.att_dst.copy_(our_layer.att_dst)

    our_out = our_layer(x, edge_index)
    pyg_out = pyg_layer(x, edge_index)

    torch.testing.assert_close(our_out, pyg_out, atol=1e-5, rtol=1e-5)


def test_gat_layer_sanity_overfit_toy_graph():
    """Verify that a small GAT model can overfit and decrease loss on a toy graph."""
    torch.manual_seed(42)
    num_nodes = 8
    in_channels = 12
    num_classes = 3
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.long
    )
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.long)

    class SmallGAT(nn.Module):
        def __init__(self):
            super().__init__()
            self.gat1 = GATLayer(in_channels, 8, heads=2, concat=True)
            self.elu = nn.ELU()
            self.gat2 = GATLayer(8 * 2, num_classes, heads=1, concat=False)

        def forward(self, x, edge_index):
            x = self.elu(self.gat1(x, edge_index))
            x = self.gat2(x, edge_index)
            return x

    model = SmallGAT()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    initial_loss = None
    final_loss = None

    for epoch in range(60):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert final_loss < initial_loss * 0.3, (
        f"Expected final loss < 0.3 * initial_loss ({0.3 * initial_loss:.4f}), got {final_loss:.4f}"
    )


def test_gat_layer_input_validation():
    """Test validation and clear error messages for invalid inputs."""
    layer = GATLayer(in_channels=8, out_channels=4, heads=2)

    # 1. x must be 2D floating-point tensor
    with pytest.raises(ValueError, match="x must be a 2D floating-point tensor"):
        layer(torch.zeros(4, dtype=torch.float32), torch.zeros((2, 2), dtype=torch.long))

    with pytest.raises(ValueError, match="x must be a 2D floating-point tensor"):
        layer(torch.zeros((4, 8), dtype=torch.long), torch.zeros((2, 2), dtype=torch.long))

    # 2. edge_index must be 2D long tensor with shape [2, E]
    with pytest.raises(ValueError, match="edge_index must be a torch.long tensor of shape \\[2, num_edges\\]"):
        layer(torch.zeros((4, 8)), torch.zeros((3, 2), dtype=torch.long))

    # 3. out of bounds node indices
    with pytest.raises(ValueError, match="Edge indices must lie in \\[0, num_nodes\\)"):
        layer(torch.zeros((4, 8)), torch.tensor([[0, 5], [1, 2]], dtype=torch.long))
