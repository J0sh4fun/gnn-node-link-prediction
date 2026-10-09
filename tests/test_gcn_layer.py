"""Week 5 GCN acceptance checks on small, entirely synthetic graphs."""

import pytest
import torch
from torch import nn
from torch_geometric.nn import GCNConv

from models import GCNLayer


@pytest.fixture(autouse=True)
def deterministic_cpu():
    """Keep tiny graph tests fast and restore caller RNG/thread settings."""
    previous_threads = torch.get_num_threads()
    with torch.random.fork_rng():
        torch.manual_seed(42)
        torch.set_num_threads(1)
        try:
            yield
        finally:
            torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_matches_pyg_output_and_gradients(dtype, bias, weighted):
    # Irregular degrees, an existing loop, and node 4 absent from the edge list.
    edges = torch.tensor([[0, 1, 1, 2, 1, 3, 2], [1, 0, 2, 1, 3, 1, 2]])
    raw_weights = torch.tensor([2., 2., 0.5, 0.5, 3., 3., 1.5], dtype=dtype)
    raw_weights = raw_weights if weighted else None
    layer = GCNLayer(3, 2, bias=bias).to(dtype)
    reference = GCNConv(
        3, 2, bias=bias, cached=False, improved=False,
        add_self_loops=True, normalize=True, flow="source_to_target",
    ).to(dtype)
    with torch.no_grad():
        reference.lin.weight.copy_(layer.weight.t())
        if bias:
            layer.bias.copy_(torch.tensor([0.3, -0.2], dtype=dtype))
            reference.bias.copy_(layer.bias)
    x = torch.randn(5, 3, dtype=dtype, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    before = edges.clone()
    output = layer(x, edges, raw_weights)
    expected = reference(ref_x, edges, raw_weights)
    tolerance = 1e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(output, expected, rtol=tolerance, atol=tolerance)
    assert output.shape == (5, 2)
    assert output.dtype == dtype
    assert torch.equal(edges, before)
    upstream = torch.randn_like(output)
    (output * upstream).sum().backward()
    (expected * upstream).sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(
        layer.weight.grad, reference.lin.weight.grad.t(), rtol=tolerance, atol=tolerance
    )
    if bias:
        torch.testing.assert_close(layer.bias.grad, reference.bias.grad)
    assert torch.isfinite(output).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())


def test_hand_calculated_normalization_and_post_aggregation_bias():
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    layer = GCNLayer(1, 1).double()
    with torch.no_grad():
        layer.weight.fill_(2.)
        layer.bias.fill_(0.7)
    x = torch.tensor([[1.], [2.], [4.], [8.]], dtype=torch.float64)
    # Degrees after adding loops: [2, 3, 2, 1].
    cross = 1 / (6 ** 0.5)
    expected = torch.tensor([
        [2 * (0.5 + 2 * cross) + 0.7],
        [2 * (cross + 2 / 3 + 4 * cross) + 0.7],
        [2 * (2 * cross + 2) + 0.7],
        [16.7],
    ], dtype=x.dtype)
    torch.testing.assert_close(layer(x, edges), expected)


@pytest.mark.parametrize("num_nodes", [0, 4])
def test_edgeless_graph_is_linear_transform(num_nodes):
    layer = GCNLayer(3, 2)
    x = torch.randn(num_nodes, 3, requires_grad=True)
    output = layer(x, torch.empty((2, 0), dtype=torch.long))
    torch.testing.assert_close(output, x @ layer.weight + layer.bias)
    output.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())


def test_duplicate_unweighted_loops_and_changing_graph():
    layer = GCNLayer(1, 1, bias=False)
    with torch.no_grad():
        layer.weight.fill_(1.)
    x = torch.tensor([[2.], [6.], [9.]])
    edges = torch.tensor([[0, 0, 0, 1], [0, 0, 1, 0]])
    torch.testing.assert_close(layer(x, edges), torch.tensor([[4.], [4.], [9.]]))
    # Same edge count but different topology must not reuse normalization.
    changed = torch.tensor([[2, 2, 1, 2], [2, 2, 2, 1]])
    torch.testing.assert_close(layer(x, changed), torch.tensor([[2.], [7.5], [7.5]]))


def test_weighted_duplicate_loops_and_zero_degree():
    layer = GCNLayer(1, 1, bias=False).double()
    with torch.no_grad():
        layer.weight.fill_(1.)
    edges = torch.tensor([[0, 1, 0, 0, 2], [1, 0, 0, 0, 2]])
    weights = torch.tensor([2., 2., 1., 2., 0.], dtype=torch.float64)
    x = torch.tensor([[2.], [6.], [9.]], dtype=weights.dtype)
    expected = torch.tensor([
        [2 * 3 / 5 + 6 * 2 / (15 ** 0.5)],
        [2 * 2 / (15 ** 0.5) + 6 / 3],
        [0.],
    ], dtype=x.dtype)
    torch.testing.assert_close(layer(x, edges, weights), expected)
    assert torch.equal(weights, torch.tensor([2., 2., 1., 2., 0.], dtype=weights.dtype))


def test_reset_and_state_dict_round_trip():
    layer = GCNLayer(4, 2)
    original_weight = layer.weight.detach().clone()
    with torch.no_grad():
        layer.bias.fill_(2.)
    layer.reset_parameters()
    assert not torch.equal(layer.weight, original_weight)
    assert torch.equal(layer.bias, torch.zeros(2))
    restored = GCNLayer(4, 2)
    restored.load_state_dict(layer.state_dict())
    x, edges = torch.randn(3, 4), torch.tensor([[0, 1], [1, 0]])
    torch.testing.assert_close(layer(x, edges), restored(x, edges))


@pytest.mark.parametrize("case", ["shape", "dtype", "index", "negative_weight", "weight_dtype"])
def test_invalid_inputs_fail_clearly(case):
    layer = GCNLayer(2, 2)
    x, edges, weights = torch.ones(2, 2), torch.tensor([[0, 1], [1, 0]]), None
    if case == "shape":
        x = torch.ones(2, 3)
    elif case == "dtype":
        x = x.double()
    elif case == "index":
        edges = torch.tensor([[0, 2], [2, 0]])
    elif case == "negative_weight":
        weights = torch.tensor([-1., -1.])
    else:
        weights = torch.ones(2, dtype=torch.float64)
    with pytest.raises(ValueError):
        layer(x, edges, weights)


def test_gradients_match_finite_differences():
    """Check input and learned-parameter derivatives without a PyG oracle."""
    layer = GCNLayer(2, 2).double()
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    x = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)

    def forward(features, weight, bias):
        return torch.func.functional_call(
            layer, {"weight": weight, "bias": bias}, (features, edges)
        )

    assert torch.autograd.gradcheck(
        forward, (x, layer.weight, layer.bias), eps=1e-6, atol=1e-5, rtol=1e-3
    )


def test_node_relabeling_and_edge_order_preserve_results():
    """Node IDs and storage order must not change the represented graph operator."""
    layer = GCNLayer(3, 2).double()
    x = torch.randn(5, 3, dtype=torch.float64)
    edges = torch.tensor([[0, 1, 1, 2, 1, 3], [1, 0, 2, 1, 3, 1]])
    weights = torch.tensor([2., 2., 0.5, 0.5, 3., 3.], dtype=x.dtype)
    expected = layer(x, edges, weights)
    # Mapping from old node ID to new node ID, including the isolated node.
    relabel = torch.tensor([2, 4, 0, 3, 1])
    reordered_x = torch.empty_like(x)
    reordered_x[relabel] = x
    edge_order = torch.tensor([5, 2, 0, 4, 1, 3])
    actual = layer(
        reordered_x, relabel[edges[:, edge_order]], weights[edge_order]
    )
    torch.testing.assert_close(actual[relabel], expected, rtol=1e-12, atol=1e-12)


def test_changed_raw_weights_recompute_normalization():
    """Reusing an edge/weight tensor after an update must not reuse old degrees."""
    layer = GCNLayer(1, 1, bias=False).double()
    with torch.no_grad():
        layer.weight.fill_(1.)
    x = torch.tensor([[2.], [6.]], dtype=torch.float64)
    edges = torch.tensor([[0, 1], [1, 0]])
    weights = torch.ones(2, dtype=x.dtype)
    torch.testing.assert_close(layer(x, edges, weights), x.new_tensor([[4.], [4.]]))
    weights.fill_(3.)
    # New degree four: a unit self-loop and an edge with weight three.
    torch.testing.assert_close(layer(x, edges, weights), x.new_tensor([[5.], [3.]]))
    assert torch.equal(weights, x.new_tensor([3., 3.]))


@pytest.mark.parametrize("weighted", [False, True])
def test_parallel_non_loop_edges_count_additively(weighted):
    layer = GCNLayer(1, 1, bias=False).double()
    with torch.no_grad():
        layer.weight.fill_(1.)
    x = torch.tensor([[2.], [6.], [9.]], dtype=torch.float64)
    edges = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])
    weights = x.new_tensor([0.5, 1.5, 0.5, 1.5]) if weighted else None
    # Two units of total weight in each direction, plus unit self-loops.
    expected = x.new_tensor([[14 / 3], [10 / 3], [9.]])
    torch.testing.assert_close(layer(x, edges, weights), expected)


def test_message_passing_uses_source_to_target_flow():
    """Probe the message hook directly; symmetric graphs alone cannot detect reversal.

    This is a directed diagnostic of propagate(), not a directed-graph contract
    for forward(). The supplied coefficients are fixed for this hook test.
    """
    layer = GCNLayer(1, 1, bias=False)
    support = torch.tensor([[2.], [5.], [11.]])
    edges = torch.tensor([[0, 2], [1, 1]])
    actual = layer.propagate(
        edges, x=support, norm=torch.tensor([0.5, 0.25]), size=(3, 3)
    )
    torch.testing.assert_close(actual, torch.tensor([[0.], [3.75], [0.]]))


@pytest.mark.parametrize("bad_weight", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_raw_weights_are_rejected(bad_weight):
    layer = GCNLayer(2, 2)
    with pytest.raises(ValueError, match="finite"):
        layer(
            torch.ones(2, 2), torch.tensor([[0, 1], [1, 0]]),
            torch.tensor([bad_weight, bad_weight]),
        )


def test_overfits_thirty_connected_nodes():
    """Acceptance sanity check; this never loads Cora or evaluates held-out labels."""
    labels = torch.arange(3).repeat_interleave(10)
    x = nn.functional.one_hot(labels, num_classes=3).float()
    x = x + 0.05 * torch.randn_like(x)
    source = torch.cat([torch.arange(start, start + 9) for start in (0, 10, 20)])
    edges = torch.stack((torch.cat((source, source + 1)), torch.cat((source + 1, source))))
    layer = GCNLayer(3, 3)
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.05)
    losses = []
    for _ in range(100):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(layer(x, edges), labels)
        assert torch.isfinite(loss)
        losses.append(loss.item())
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
        optimizer.step()
    with torch.no_grad():
        accuracy = (layer(x, edges).argmax(dim=-1) == labels).float().mean().item()
    assert losses[-1] < 0.1 * losses[0]
    assert all(later < earlier for earlier, later in zip(losses[::10], losses[10::10]))
    assert accuracy == 1.0
    print(f"GCN sanity: loss {losses[0]:.6f} -> {losses[-1]:.6f}; accuracy {accuracy:.1%}")
