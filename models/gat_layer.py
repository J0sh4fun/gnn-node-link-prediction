"""Sparse, multi-head graph attention layer implemented with PyTorch ops."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from utils.attention import receiver_softmax


class GATLayer(nn.Module):
    """A single GAT layer using source-to-target train adjacency.

    Attention scores and aggregation are computed only for listed edges; this
    layer never constructs a dense num_nodes by num_nodes attention matrix.

    Dropout is applied to normalized attention weights; feature_dropout is
    applied to input node features. No activation is applied inside the layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        add_self_loops: bool = True,
        bias: bool = True,
        feature_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("in_channels", in_channels),
            ("out_channels", out_channels),
            ("heads", heads),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("dropout", dropout), ("feature_dropout", feature_dropout)):
            if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if (
            not isinstance(negative_slope, (int, float))
            or not math.isfinite(negative_slope)
            or negative_slope < 0
        ):
            raise ValueError("negative_slope must be a finite non-negative number")
        if not isinstance(concat, bool) or not isinstance(add_self_loops, bool):
            raise ValueError("concat and add_self_loops must be bool values")
        if not isinstance(bias, bool):
            raise ValueError("bias must be a bool value")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = float(dropout)
        self.feature_dropout = float(feature_dropout)
        self.negative_slope = float(negative_slope)
        self.add_self_loops = add_self_loops

        self.proj = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))
        output_channels = heads * out_channels if concat else out_channels
        self.bias = nn.Parameter(torch.empty(output_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset projection, attention vectors, and optional bias."""
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _prepare_edges(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if not isinstance(edge_index, torch.Tensor):
            raise TypeError("train_adjacency must be a torch.Tensor")
        if edge_index.dtype != torch.long:
            raise ValueError("train_adjacency must have dtype torch.long")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("train_adjacency must have shape [2, num_edges]")
        if edge_index.device != self.proj.weight.device:
            raise ValueError(
                "train_adjacency and layer parameters must be on the same device"
            )
        if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
            raise ValueError("train_adjacency contains an index outside [0, num_nodes)")

        if not self.add_self_loops:
            return edge_index

        non_loops = edge_index[:, edge_index[0] != edge_index[1]]
        loops = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
        loops = torch.stack((loops, loops))
        return torch.cat((non_loops, loops), dim=1)

    def forward(
        self,
        x: torch.Tensor,
        train_adjacency: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Apply sparse multi-head attention over the supplied graph edges."""
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError("x must be a tensor with shape [num_nodes, in_channels]")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"x has {x.shape[1]} features; expected {self.in_channels}")
        if not x.is_floating_point():
            raise ValueError("x must have a floating-point dtype")
        if x.device != self.proj.weight.device:
            raise ValueError("x and layer parameters must be on the same device")
        if x.dtype != self.proj.weight.dtype:
            raise ValueError("x and layer parameters must have the same dtype")
        if not isinstance(return_attention, bool):
            raise ValueError("return_attention must be a bool value")

        num_nodes = x.shape[0]
        processed_edges = self._prepare_edges(train_adjacency, num_nodes)
        source, target = processed_edges

        features = F.dropout(x, p=self.feature_dropout, training=self.training)
        projected = self.proj(features).view(num_nodes, self.heads, self.out_channels)
        source_term = (projected * self.att_src).sum(dim=-1)
        target_term = (projected * self.att_dst).sum(dim=-1)
        scores = F.leaky_relu(
            source_term[source] + target_term[target],
            negative_slope=self.negative_slope,
        )
        alpha = receiver_softmax(scores, target, num_nodes)
        dropped_alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        messages = projected[source] * dropped_alpha.unsqueeze(-1)
        output = projected.new_zeros((num_nodes, self.heads, self.out_channels))
        output.index_add_(0, target, messages)
        if self.concat:
            output = output.reshape(num_nodes, self.heads * self.out_channels)
        else:
            output = output.mean(dim=1)
        if self.bias is not None:
            output = output + self.bias

        if return_attention:
            return output, (processed_edges, alpha)
        return output


__all__ = ["GATLayer"]
