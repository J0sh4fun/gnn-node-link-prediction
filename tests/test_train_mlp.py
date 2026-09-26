"""Protocol tests for MLP training without downloading Cora or evaluating it."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from models.mlp import MLP
from train import train_mlp


@pytest.mark.parametrize(
    ("losses", "max_epochs", "patience", "best_epoch", "stopped_epoch"),
    [
        ([1.0, 0.5, 0.5, 0.7], 10, 2, 2, 4),
        ([1.0, 0.5, 0.8], 3, 20, 2, 3),
    ],
)
def test_fit_restores_best_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    losses: list[float],
    max_epochs: int,
    patience: int,
    best_epoch: int,
    stopped_epoch: int,
) -> None:
    """Ties consume patience; the selected weights and metrics survive later updates."""
    torch.manual_seed(7)
    model = MLP(3, hidden_channels=4, out_channels=2, dropout=0.0)
    x = torch.randn(6, 3)
    # Out-of-range labels on unused nodes expose accidental unmasked loss calls.
    targets = torch.tensor([0, 1, 0, 1, 99, 99])
    train_mask = torch.tensor([True, True, False, False, False, False])
    val_mask = torch.tensor([False, False, True, True, False, False])
    scheduled_losses = iter(losses)
    states: list[dict[str, torch.Tensor]] = []

    class ScheduledValidationLoss(nn.CrossEntropyLoss):
        """Use real training gradients with a controlled validation loss schedule."""

        def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """Assert mask isolation and substitute only validation losses."""
            assert input.shape == (2, 2)
            assert torch.equal(target, torch.tensor([0, 1]))
            if torch.is_grad_enabled():
                assert model.training
                return super().forward(input, target)
            assert not model.training
            return input.new_tensor(next(scheduled_losses))

    def record_validation(
        logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, float]:
        """Record each checkpoint and return epoch-specific validation metrics."""
        assert not torch.is_grad_enabled()
        assert torch.equal(mask, val_mask)
        states.append(deepcopy(model.state_dict()))
        return {"accuracy": len(states) / 10, "macro_f1": len(states) / 20}

    monkeypatch.setattr(train_mlp, "evaluate_node_classification", record_validation)
    result = train_mlp.fit_mlp(
        model, x, targets, train_mask, val_mask,
        torch.optim.Adam(model.parameters(), lr=0.01), ScheduledValidationLoss(),
        max_epochs=max_epochs, patience=patience,
    )

    assert result.stopped_epoch == stopped_epoch
    assert result.best_epoch == best_epoch
    assert result.best_val_loss == pytest.approx(losses[best_epoch - 1])
    assert result.val_accuracy == pytest.approx(best_epoch / 10)
    assert result.val_macro_f1 == pytest.approx(best_epoch / 20)
    assert len(states) == stopped_epoch
    assert not model.training
    for name, weight in model.state_dict().items():
        assert torch.equal(weight, states[best_epoch - 1][name])
    assert any(
        not torch.equal(states[-1][name], states[best_epoch - 1][name])
        for name in states[-1]
    )


def test_main_evaluates_test_once_after_fitting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Final evaluation follows fitting, writes the requested schema, and ignores cwd."""
    data = Data(
        x=torch.zeros(6, 1433),
        y=torch.tensor([0, 1, 0, 1, 0, 1]),
        train_mask=torch.tensor([True, True, False, False, False, False]),
        val_mask=torch.tensor([False, False, True, True, False, False]),
        test_mask=torch.tensor([False, False, False, False, True, True]),
    )
    events: list[str] = []

    def fake_loader(root: str, normalize_features: bool) -> Data:
        """Require the repository cache path even when cwd is elsewhere."""
        assert Path(root) == tmp_path / "data"
        assert normalize_features
        return data

    def fake_fit(
        model: MLP, x: torch.Tensor, targets: torch.Tensor,
        train_mask: torch.Tensor, val_mask: torch.Tensor,
        optimizer: torch.optim.Optimizer, criterion: nn.CrossEntropyLoss,
        max_epochs: int, patience: int,
    ) -> train_mlp.TrainingResult:
        """Simulate fitting and restoration of known selected weights."""
        assert max_epochs == 200 and patience == 20
        assert torch.equal(train_mask, data.train_mask)
        assert torch.equal(val_mask, data.val_mask)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        model.eval()
        events.append("restored")
        return train_mlp.TrainingResult(25, 5, 0.5, 0.75, 0.7)

    def final_evaluation(
        logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, float]:
        """Verify the only final metric call uses the restored model and test mask."""
        assert events == ["restored"]
        assert not torch.is_grad_enabled()
        assert torch.equal(mask, data.test_mask)
        assert torch.count_nonzero(logits).item() == 0
        events.append("test")
        return {"accuracy": 0.6, "macro_f1": 0.55}

    monkeypatch.setattr(train_mlp, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(train_mlp, "set_seed", lambda seed: None)
    monkeypatch.setattr(train_mlp, "get_cora_data", fake_loader)
    monkeypatch.setattr(train_mlp, "fit_mlp", fake_fit)
    monkeypatch.setattr(train_mlp, "evaluate_node_classification", final_evaluation)
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    train_mlp.main()

    assert events == ["restored", "test"]
    log = json.loads((tmp_path / "results/baseline_mlp_log.json").read_text())
    assert log == {
        "model": "MLP", "dataset": "Cora", "seed": 42,
        "learning_rate": 0.01, "weight_decay": 5e-4, "hidden_dim": 16,
        "num_layers": 2, "dropout": 0.5, "stopped_epoch": 25,
        "best_val_loss": 0.5, "val_accuracy": 0.75, "val_macro_f1": 0.7,
        "test_accuracy": 0.6, "test_macro_f1": 0.55,
    }
    checkpoint = torch.load(
        tmp_path / "results/checkpoints/best_mlp_val.pt", weights_only=True
    )
    assert all(torch.count_nonzero(value).item() == 0 for value in checkpoint.values())
