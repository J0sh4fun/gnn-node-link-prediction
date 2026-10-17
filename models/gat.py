"""Multi-layer Graph Attention Network (GAT) for node classification."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.gat_layer import GATLayer


class GAT(nn.Module):
    """Concatenate hidden heads, apply ELU, then average output heads.

    Supports both Member A and Member B conventions (hidden_channels / hidden_dim).
    """

    def __init__(
        self,
        in_channels: int = 1433,
        hidden_channels: int | None = None,
        out_channels: int | None = None,
        heads: int = 8,
        output_heads: int = 1,
        dropout: float = 0.6,
        attention_dropout: float = 0.6,
        *,
        hidden_dim: int | None = None,
        num_classes: int | None = None,
        out_heads: int | None = None,
        feature_dropout: float | None = None,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        # Resolve aliases
        if hidden_channels is None:
            hidden_channels = hidden_dim if hidden_dim is not None else 8
        if out_channels is None:
            out_channels = num_classes if num_classes is not None else 7
        if out_heads is not None:
            output_heads = out_heads
        if feature_dropout is not None:
            dropout = feature_dropout

        if not 0 <= attention_dropout <= 1:
            raise ValueError("attention_dropout must be between 0 and 1")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.hidden_dim = hidden_channels
        self.out_channels = out_channels
        self.num_classes = out_channels
        self.heads = heads
        self.output_heads = output_heads
        self.out_heads = output_heads

        self.conv1 = GATLayer(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=attention_dropout,
            negative_slope=negative_slope,
        )
        self.conv2 = GATLayer(
            hidden_channels * heads,
            out_channels,
            heads=output_heads,
            concat=False,
            dropout=attention_dropout,
            negative_slope=negative_slope,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through 2 GAT layers with ELU activation.

        Args:
            x (torch.Tensor): Node feature matrix [N, in_channels].
            edge_index (torch.Tensor): Graph edge indices [2, E].
            return_attention (bool): Whether to return second-layer attention weights.

        Returns:
            torch.Tensor: Unnormalized class logits [N, out_channels].
        """
        x = self.conv1(self.dropout(x), edge_index)
        x = F.elu(x)
        if return_attention:
            x, att = self.conv2(self.dropout(x), edge_index, return_attention_weights=True)
            return x, att
        return self.conv2(self.dropout(x), edge_index)


__all__ = ["GAT"]
