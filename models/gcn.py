"""Independent two-layer GCN node classifier over raw graph edges."""

import torch
from torch import nn
from torch.nn import functional as F

from models.gcn_layer import GCNLayer


class GCN(nn.Module):
    """Apply GCN → ReLU → dropout → GCN, returning unnormalized logits."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 dropout: float = 0.5, num_layers: int = 2) -> None:
        super().__init__()
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 2:
            raise ValueError("num_layers must be an integer >= 2")
        self.conv1 = GCNLayer(in_channels, hidden_channels)
        self.hidden_convs = nn.ModuleList(
            GCNLayer(hidden_channels, hidden_channels) for _ in range(num_layers - 2))
        self.conv2 = GCNLayer(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.conv1(x, edge_index)))
        for layer in self.hidden_convs:
            x = self.dropout(F.relu(layer(x, edge_index)))
        return self.conv2(x, edge_index)
