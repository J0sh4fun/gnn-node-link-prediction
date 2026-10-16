"""Independent two-layer GCN node classifier over raw graph edges."""

import torch
from torch import nn
from torch.nn import functional as F

from models.gcn_layer import GCNLayer


class GCN(nn.Module):
    """Apply GCN → ReLU → dropout → GCN, returning unnormalized logits."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 dropout: float = 0.5) -> None:
        super().__init__()
        self.conv1 = GCNLayer(in_channels, hidden_channels)
        self.conv2 = GCNLayer(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.conv1(x, edge_index)))
        return self.conv2(x, edge_index)
