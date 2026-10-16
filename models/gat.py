"""Independent two-layer multi-head GAT node classifier."""

import torch
from torch import nn
from torch.nn import functional as F

from models.gat_layer import GATLayer


class GAT(nn.Module):
    """Concatenate hidden heads, apply ELU, then average output heads.

    ``hidden_channels`` is the width per hidden head. Output has exactly
    ``out_channels`` columns. Feature and attention dropout are explicit and
    disabled in evaluation mode. Both layers receive the same raw edges.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 heads: int = 8, output_heads: int = 1, dropout: float = 0.6,
                 attention_dropout: float = 0.6) -> None:
        super().__init__()
        if not 0 <= attention_dropout <= 1:
            raise ValueError("attention_dropout must be between 0 and 1")
        self.conv1 = GATLayer(in_channels, hidden_channels, heads=heads,
                              concat=True, dropout=attention_dropout)
        self.conv2 = GATLayer(hidden_channels * heads, out_channels,
                              heads=output_heads, concat=False, dropout=attention_dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.conv1(self.dropout(x), edge_index))
        return self.conv2(self.dropout(x), edge_index)
