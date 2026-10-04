"""Analytical sparse-graph tests without allocating dense adjacency matrices."""

import math

import pytest
import torch

from utils.graph_ops import (
    add_remaining_self_loops,
    compute_symmetric_norm,
    sparse_spmm,
)


def test_self_loops_are_unique_and_weights_initialized() -> None:
    """Duplicate input loops collapse while every node receives one loop."""
    edges = torch.tensor([[0, 0, 0, 1], [0, 0, 1, 0]])
    original = edges.clone()
    updated, weights = add_remaining_self_loops(edges, 3, fill_value=2.5)
    assert torch.equal(edges, original)
    assert updated.shape == (2, 5)
    loops = updated[0] == updated[1]
    assert torch.equal(updated[0, loops], torch.arange(3))
    assert torch.equal(updated[:, ~loops], torch.tensor([[0, 1], [1, 0]]))
    assert weights.dtype == torch.get_default_dtype()
    assert weights.device == edges.device
    assert torch.equal(weights, torch.full((5,), 2.5))
    repeated, repeated_weights = add_remaining_self_loops(updated, 3, 2.5)
    assert torch.equal(repeated, updated)
    assert torch.equal(repeated_weights, weights)


def test_normalization_includes_loops_before_degree() -> None:
    """A linked pair has degree two after loops, and an isolated node has degree one."""
    edges, weights = compute_symmetric_norm(torch.tensor([[0, 1], [1, 0]]), 3)
    expected = {(0, 0): 0.5, (0, 1): 0.5, (1, 0): 0.5, (1, 1): 0.5, (2, 2): 1.0}
    assert edges.size(1) == len(expected)
    for pair, weight in zip(edges.t().tolist(), weights.tolist()):
        assert weight == pytest.approx(expected[tuple(pair)])
    x = torch.tensor([[2., 4.], [6., 8.], [3., 9.]])
    assert torch.allclose(
        sparse_spmm(edges, weights, 3, x),
        torch.tensor([[4., 6.], [4., 6.], [3., 9.]]),
    )


def test_weighted_normalization_preserves_existing_loops() -> None:
    """Weighted duplicate loops sum, existing diagonals persist, and missing loops use one."""
    edges = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0]])
    weights = torch.tensor([2., 2., 1., 2.], dtype=torch.float64)
    edges_before, weights_before = edges.clone(), weights.clone()
    normalized_edges, normalized = compute_symmetric_norm(edges, 2, weights)
    expected = {(0, 0): 3 / 5, (0, 1): 2 / math.sqrt(15),
                (1, 0): 2 / math.sqrt(15), (1, 1): 1 / 3}
    assert normalized.dtype == torch.float64
    assert normalized_edges.size(1) == 4
    for pair, weight in zip(normalized_edges.t().tolist(), normalized.tolist()):
        assert weight == pytest.approx(expected[tuple(pair)])
    assert torch.equal(edges, edges_before)
    assert torch.equal(weights, weights_before)


def test_unweighted_duplicate_loops_do_not_increase_degree() -> None:
    """Repeated unweighted loops represent one self-connection, not extra degree."""
    edges = torch.tensor([[0, 0, 0, 1], [0, 0, 1, 0]])
    updated, weights = compute_symmetric_norm(edges, 2)
    assert updated.size(1) == 4
    assert torch.allclose(weights, torch.full((4,), 0.5))


def test_zero_degree_is_finite() -> None:
    """An explicitly zero-weight loop has zero normalization, not infinity or NaN."""
    edges, weights = compute_symmetric_norm(
        torch.tensor([[0], [0]]), 2, torch.tensor([0.])
    )
    assert torch.equal(edges, torch.tensor([[0, 1], [0, 1]]))
    assert torch.equal(weights, torch.tensor([0., 1.]))
    assert torch.isfinite(weights).all()


@pytest.mark.parametrize("num_nodes", [0, 3])
def test_edgeless_graph_is_identity(num_nodes: int) -> None:
    """Missing edges become unit self-connections, including the empty-graph case."""
    edges, weights = compute_symmetric_norm(torch.empty((2, 0), dtype=torch.long), num_nodes)
    x = torch.randn(num_nodes, 2)
    assert edges.shape == (2, num_nodes)
    assert torch.allclose(sparse_spmm(edges, weights, num_nodes, x), x)


def test_spmm_orientation_coalescing_and_input_gradient() -> None:
    """Duplicate source-to-target entries sum and the exact input gradient is preserved."""
    edges = torch.tensor([[0, 0, 1], [1, 1, 2]])
    weights = torch.tensor([0.5, 1.5, 3.0], dtype=torch.float64)
    x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.float64, requires_grad=True)
    output = sparse_spmm(edges, weights, 3, x)
    assert torch.equal(output, torch.tensor([[0., 0.], [2., 4.], [9., 12.]], dtype=x.dtype))
    output.sum().backward()
    assert torch.equal(x.grad, torch.tensor([[2., 2.], [3., 3.], [0., 0.]], dtype=x.dtype))


@pytest.mark.parametrize("bad_weight", [-1.0, float("nan"), float("inf")])
def test_invalid_normalization_weights_raise(bad_weight: float) -> None:
    """Reject weights outside the nonnegative finite spectral-normalization domain."""
    with pytest.raises(ValueError):
        compute_symmetric_norm(torch.tensor([[0], [0]]), 1, torch.tensor([bad_weight]))


def test_invalid_indices_and_mismatched_dtypes_raise() -> None:
    """Fail clearly on out-of-range node IDs or incompatible sparse/dense dtypes."""
    with pytest.raises(ValueError, match="indices"):
        add_remaining_self_loops(torch.tensor([[0], [2]]), 2)
    with pytest.raises(ValueError, match="dtype"):
        sparse_spmm(torch.tensor([[0], [0]]), torch.ones(1), 1, torch.ones(1, 2).double())
