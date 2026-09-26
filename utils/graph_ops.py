"""Sparse graph operations for GCN normalization on undirected weighted graphs.

Edges follow PyG's source-to-target convention: ``edge_index[:, e] = [j, i]``
stores an edge from node j to node i, so the matrix entry is A[i, j]. An
undirected input must contain both directions with equal aggregate weights.
These utilities do not symmetrize input graphs; do that in the data pipeline.
Only edge lists, sparse COO tensors, and node-sized vectors are constructed.
"""

import math

import torch


def _validate_edges(edge_index: torch.Tensor, num_nodes: int) -> None:
    """Validate a COO edge list and its explicitly supplied graph size."""
    if not isinstance(num_nodes, int) or isinstance(num_nodes, bool) or num_nodes < 0:
        raise ValueError("num_nodes must be a nonnegative integer")
    if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must be a torch.long tensor of shape [2, num_edges]")
    if edge_index.numel() and (
        edge_index.min().item() < 0 or edge_index.max().item() >= num_nodes
    ):
        raise ValueError("Edge indices must lie in [0, num_nodes)")


def _validate_weights(edge_index: torch.Tensor, edge_weight: torch.Tensor) -> None:
    """Validate finite floating-point weights aligned with the edge list."""
    if edge_weight.ndim != 1 or edge_weight.numel() != edge_index.size(1):
        raise ValueError("edge_weight must have shape [num_edges]")
    if not edge_weight.is_floating_point():
        raise ValueError("edge_weight must be floating point")
    if edge_weight.device != edge_index.device:
        raise ValueError("edge_index and edge_weight must be on the same device")
    if not torch.isfinite(edge_weight).all().item():
        raise ValueError("edge_weight must contain only finite values")


def add_remaining_self_loops(
    edge_index: torch.Tensor, num_nodes: int, fill_value: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensure exactly one self-loop per node and initialize every edge weight.

    Args:
        edge_index: Source/target indices of shape ``[2, num_edges]``.
        num_nodes: Number of nodes, including isolated nodes.
        fill_value: Finite, nonnegative initial weight for ALL returned edges,
            including existing non-loop edges. Defaults to one.

    Returns:
        An edge list followed by one loop per node in node-ID order, and a
        weight vector using ``torch.get_default_dtype()`` on the input device.
        Existing self-loops are canonicalized, so even duplicate input loops
        produce only one loop per node. Non-loop edges retain their order and
        multiplicity. The input tensor is not modified.

    Notes:
        This unweighted-input helper cannot preserve externally supplied
        weights. Use ``compute_symmetric_norm`` for weighted inputs.
        Storage is O(E + N); no dense adjacency matrix is allocated.
    """
    _validate_edges(edge_index, num_nodes)
    if not math.isfinite(fill_value) or fill_value < 0:
        raise ValueError("fill_value must be finite and nonnegative")
    non_loop = edge_index[0] != edge_index[1]
    nodes = torch.arange(num_nodes, device=edge_index.device, dtype=torch.long)
    loops = torch.stack((nodes, nodes))
    updated_edges = torch.cat((edge_index[:, non_loop], loops), dim=1)
    weights = torch.full(
        (updated_edges.size(1),), fill_value,
        dtype=torch.get_default_dtype(), device=edge_index.device,
    )
    return updated_edges, weights


def compute_symmetric_norm(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute edge-wise coefficients of D_tilde^-1/2 A_tilde D_tilde^-1/2.

    For a loop-free adjacency A, A_tilde = A + I. For already loop-augmented
    inputs, existing diagonal entries are retained instead of adding I again:
    this is the add-remaining-self-loops convention, not literal A + I for
    an adjacency that already contains loops. Normalized output should not
    be passed back as raw input to this function.

    Args:
        edge_index: Undirected source/target edge list of shape ``[2, E]``.
            Reverse directions must have equal total weight.
        num_nodes: Number of nodes, including nodes absent from the edge list.
        edge_weight: Optional finite, nonnegative floating weights of shape
            ``[E]``. Its dtype/device are preserved. Without weights, edges
            receive unit weights and duplicate self-loops are collapsed.

    Returns:
        Coalesced edge indices, including all self-loops, and their normalized
        weights. Weighted duplicate entries (including loops) are summed;
        non-loop duplicates in an unweighted edge list also count additively.
        Missing loops receive weight one, while existing weighted loops retain
        their aggregate value, including zero.

    Notes:
        Degrees are row sums of A_tilde, computed AFTER adding missing loops.
        Under source-to-target storage, this is a scatter sum on target IDs.
        Each coefficient is w_(j,i) / sqrt(d_i * d_j). Infinite inverse square
        roots from zero degrees are set to zero. An isolated node without an
        existing loop receives a unit loop and hence a normalized weight of one.
        Sparse storage uses O(E + N) memory; coalescing may sort the edges.
    """
    _validate_edges(edge_index, num_nodes)
    if edge_weight is None:
        edge_index, edge_weight = add_remaining_self_loops(edge_index, num_nodes)
    else:
        _validate_weights(edge_index, edge_weight)
        if (edge_weight < 0).any().item():
            raise ValueError("Symmetric degree normalization requires nonnegative weights")
        source, target = edge_index
        has_loop = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
        has_loop[source[source == target]] = True
        missing = torch.arange(num_nodes, device=edge_index.device)[~has_loop]
        edge_index = torch.cat((edge_index, torch.stack((missing, missing))), dim=1)
        edge_weight = torch.cat((edge_weight, edge_weight.new_ones(missing.numel())))

    # Coalescing combines parallel entries without constructing an N x N dense tensor.
    adjacency = torch.sparse_coo_tensor(
        edge_index, edge_weight, size=(num_nodes, num_nodes)
    ).coalesce()
    edge_index, edge_weight = adjacency.indices(), adjacency.values()
    source, target = edge_index
    degree = edge_weight.new_zeros(num_nodes)
    degree.index_add_(0, target, edge_weight)
    if not torch.isfinite(degree).all().item():
        raise ValueError("Degree overflow: rescale weights or use a wider floating dtype")
    inv_deg = degree.pow(-0.5)
    inv_deg = inv_deg.masked_fill(torch.isinf(inv_deg), 0.0)
    norm_weight = inv_deg[target] * edge_weight * inv_deg[source]
    return edge_index, norm_weight


def sparse_spmm(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
    x: torch.Tensor,
) -> torch.Tensor:
    """Multiply the sparse adjacency represented by edges and weights by X.

    Args:
        edge_index: Source/target indices of shape ``[2, E]``.
        edge_weight: Floating matrix values of shape ``[E]``; normally the
            output of ``compute_symmetric_norm``. Duplicates are summed.
        num_nodes: Number of rows and columns in the adjacency matrix.
        x: Dense floating node features of shape ``[num_nodes, num_features]``.
            Must share the weights' dtype/device and indices' device.

    Returns:
        Features of shape ``[num_nodes, num_features]``, with
        ``out[i] = sum_j weight[j -> i] * x[j]``. No loops or normalization
        are added here. Autograd remains connected to ``x``; the implementation
        does not detach features or convert them to NumPy.
    """
    _validate_edges(edge_index, num_nodes)
    _validate_weights(edge_index, edge_weight)
    if x.ndim != 2 or x.size(0) != num_nodes or not x.is_floating_point():
        raise ValueError("x must be floating point with shape [num_nodes, num_features]")
    if x.device != edge_weight.device or x.dtype != edge_weight.dtype:
        raise ValueError("x and edge_weight must share dtype and device")
    # PyG stores (source, target); matrix multiplication needs (row=target, col=source).
    adjacency = torch.sparse_coo_tensor(
        edge_index.flip(0), edge_weight, size=(num_nodes, num_nodes)
    ).coalesce()
    return torch.sparse.mm(adjacency, x)


if __name__ == "__main__":
    torch.manual_seed(42)
    num_nodes, num_features = 12, 5
    sampled_edges = torch.randint(num_nodes, (2, 24))
    sampled_edges = sampled_edges[:, sampled_edges[0] != sampled_edges[1]]
    undirected_edges = torch.cat((sampled_edges, sampled_edges.flip(0)), dim=1)
    undirected_edges = torch.unique(undirected_edges, dim=1)
    edges, weights = compute_symmetric_norm(undirected_edges, num_nodes)
    x = torch.randn(num_nodes, num_features, requires_grad=True)
    output = sparse_spmm(edges, weights, num_nodes, x)
    output.square().sum().backward()

    assert edges.shape == (2, weights.numel())
    assert int((edges[0] == edges[1]).sum()) == num_nodes
    assert output.shape == (num_nodes, num_features)
    assert torch.isfinite(output).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print("Normalized edge_index shape:", list(edges.shape))
    print("Normalized edge_weight shape:", list(weights.shape))
    print("Output shape:", list(output.shape))
    print("Input gradient shape:", list(x.grad.shape))
