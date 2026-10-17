"""GCN message passing with sparse, symmetric edge normalization."""

import torch
from torch import nn
from torch_geometric.nn import MessagePassing

from utils.graph_ops import add_remaining_self_loops, compute_symmetric_norm


class GCNLayer(MessagePassing):
    r"""Compute ``A_hat @ (x @ weight) + bias`` on an undirected graph.

    ``edge_index`` uses PyG's source-to-target convention, with both directions
    supplied by the data pipeline. Missing self-loops are added before degrees
    are computed. Raw, nonnegative edge weights are optional; already normalized
    weights must not be passed here. Normalization is recomputed on every call,
    so changing the graph (e.g. a link-prediction split) cannot reuse stale data.

    Unweighted duplicate loops collapse to one; weighted duplicates sum, matching
    ``utils.graph_ops``. Existing weighted loops retain their aggregate weight.
    Bias is applied after aggregation. Activation and dropout belong to the
    enclosing network, so this layer can also produce raw logits or embeddings.
    No dense adjacency is allocated.
    """

    def __init__(
        self, in_channels: int, out_channels: int, bias: bool = True
    ) -> None:
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)
        if any(
            not isinstance(size, int) or isinstance(size, bool) or size <= 0
            for size in (in_channels, out_channels)
        ):
            raise ValueError("in_channels and out_channels must be positive integers")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset learned parameters using Glorot uniform and zero bias."""
        super().reset_parameters()
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map floating features ``[N, in_channels]`` to ``[N, out_channels]``.

        Features, parameters and optional raw weights must share dtype/device;
        edge indices must be on the same device. All N nodes, including isolated
        nodes absent from the edge list, are retained in the output.
        """
        if x.ndim != 2 or x.size(1) != self.in_channels or not x.is_floating_point():
            raise ValueError("x must be floating point with shape [num_nodes, in_channels]")
        if x.dtype != self.weight.dtype or x.device != self.weight.device:
            raise ValueError("x and layer parameters must share dtype and device")
        if edge_index.device != x.device:
            raise ValueError("x and edge_index must share device")
        if edge_weight is not None and (
            edge_weight.dtype != x.dtype or edge_weight.device != x.device
        ):
            raise ValueError("x and edge_weight must share dtype and device")

        num_nodes = x.size(0)
        if edge_weight is None:
            # Canonicalize unweighted loops first, then compute coefficients in
            # the feature dtype (the utility otherwise uses the default dtype).
            edge_index, _ = add_remaining_self_loops(edge_index, num_nodes)
            edge_weight = x.new_ones(edge_index.size(1))
        edge_index, norm = compute_symmetric_norm(edge_index, num_nodes, edge_weight)
        support = x @ self.weight
        output = self.propagate(
            edge_index, x=support, norm=norm, size=(num_nodes, num_nodes)
        )
        if self.bias is not None:
            output = output + self.bias
        return output

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        """Scale each source feature by its edge's symmetric coefficient."""
        return norm.unsqueeze(-1) * x_j
