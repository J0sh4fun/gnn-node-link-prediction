"""Feature-only MLP baseline for semi-supervised node classification."""

import torch
from torch import nn


class MLP(nn.Module):
    """Classify nodes using their features without graph connectivity.

    Args:
        in_channels: Number of input features per node (1433 for Cora).
        hidden_channels: Number of hidden units. Defaults to 16.
        out_channels: Number of output classes. Defaults to 7 for Cora.
        dropout: Probability of dropping a hidden activation during training.

    Notes:
        Returns raw logits suitable for ``torch.nn.CrossEntropyLoss``.
        The training loop applies the train mask when computing the loss.
        Dropout is disabled by calling ``model.eval()`` for evaluation.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 16,
        out_channels: int = 7,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize both layers with Glorot uniform weights and zero biases."""
        for layer in (self.lin1, self.lin2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map features of shape [num_nodes, in_channels] to class logits.

        Args:
            x: Floating-point node features; no edge information is used.

        Returns:
            Raw logits of shape ``[num_nodes, out_channels]``.
        """
        x = self.lin1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.lin2(x)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = MLP(in_channels=1433, hidden_channels=16, out_channels=7)
    model.eval()
    with torch.no_grad():
        dummy_x = torch.randn(2708, 1433)
        logits = model(dummy_x)
    assert logits.shape == (2708, 7)
    print("Output shape:", list(logits.shape))
