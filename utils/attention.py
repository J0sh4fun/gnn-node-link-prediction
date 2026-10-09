"""Sparse attention primitives shared by graph-attention layers."""

import torch

def receiver_softmax(scores: torch.Tensor, targets: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Apply a numerically stable softmax independently for each receiver node.

    ``scores`` has shape ``[num_edges, num_heads]`` and ``targets`` contains
    the receiver node for each edge. Empty receiver groups remain empty and
    do not affect the result.
    """
    if not isinstance(num_nodes, int) or isinstance(num_nodes, bool) or num_nodes < 0:
        raise ValueError("num_nodes must be a non-negative integer")
    if scores.ndim != 2 or targets.ndim != 1 or scores.shape[0] != targets.shape[0]:
        raise ValueError("scores must have shape [E, K] and targets shape [E]")
    if not scores.is_floating_point() or targets.dtype != torch.long:
        raise ValueError("scores must be floating point and targets must be torch.long")
    if scores.device != targets.device:
        raise ValueError("scores and targets must be on the same device")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must contain only finite values")
    if targets.numel() and (targets.min() < 0 or targets.max() >= num_nodes):
        raise ValueError("targets contain an index outside [0, num_nodes)")
    if scores.shape[0] == 0:
        return torch.empty_like(scores)

    groups = targets[:, None].expand(-1, scores.shape[1])
    maxima = scores.new_full((num_nodes, scores.shape[1]), -torch.inf)
    maxima.scatter_reduce_(0, groups, scores, reduce="amax", include_self=True)
    exponentials = torch.exp(scores - maxima[targets])
    denominators = scores.new_zeros((num_nodes, scores.shape[1]))
    denominators.index_add_(0, targets, exponentials)
    return exponentials / denominators[targets]

