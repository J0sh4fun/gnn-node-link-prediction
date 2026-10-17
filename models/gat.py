"""Multi-layer Graph Attention Network (GAT) for node classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gat_layer import GATLayer


class GAT(nn.Module):
    """Two-layer GAT for node classification following Veličković et al. (2018).

    Args:
        in_channels (int): Input feature dimensionality. Default: 1433.
        hidden_dim (int): Dimensionality of each attention head in the hidden layer. Default: 8.
        num_classes (int): Number of target classes. Default: 7.
        heads (int): Number of attention heads in the hidden layer. Default: 8.
        out_heads (int): Number of attention heads in the output layer. Default: 1.
        dropout (float): Attention dropout probability. Default: 0.6.
        feature_dropout (float): Node feature dropout probability. Default: 0.6.
        negative_slope (float): LeakyReLU negative slope angle. Default: 0.2.
    """

    def __init__(
        self,
        in_channels: int = 1433,
        hidden_dim: int = 8,
        num_classes: int = 7,
        heads: int = 8,
        out_heads: int = 1,
        dropout: float = 0.6,
        feature_dropout: float = 0.6,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.heads = heads
        self.out_heads = out_heads

        self.conv1 = GATLayer(
            in_channels=in_channels,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
            feature_dropout=feature_dropout,
            negative_slope=negative_slope,
        )
        self.conv2 = GATLayer(
            in_channels=hidden_dim * heads,
            out_channels=num_classes,
            heads=out_heads,
            concat=False,
            dropout=dropout,
            feature_dropout=feature_dropout,
            negative_slope=negative_slope,
        )

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
            torch.Tensor: Unnormalized class logits [N, num_classes].
        """
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        if return_attention:
            x, att = self.conv2(x, edge_index, return_attention=True)
            return x, att
        return self.conv2(x, edge_index)


__all__ = ["GAT"]
