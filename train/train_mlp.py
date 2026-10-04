"""Train the Cora MLP baseline with validation-only checkpoint selection.

Run ``python train/train_mlp.py`` from the repository root, or execute this
file by its absolute path from any directory. All artifacts are resolved
relative to the repository. Each invocation performs one final test evaluation;
do not use its test metrics to select hyperparameters or seeds.
"""

from __future__ import annotations

import json
import math
import random
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch import nn

from models.mlp import MLP
from utils.data_loader import get_cora_data
from utils.evaluate import evaluate_node_classification


@dataclass(frozen=True)
class TrainingResult:
    """Validation results from the selected checkpoint and stopping epoch."""

    stopped_epoch: int
    best_epoch: int
    best_val_loss: float
    val_accuracy: float
    val_macro_f1: float


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch and enable deterministic CPU execution.

    A single CPU thread avoids variability from parallel reductions. Identical
    results require the same software versions and hardware; reproducibility
    across different platforms or PyTorch releases is not guaranteed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fit_mlp(
    model: MLP,
    x: torch.Tensor,
    targets: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    max_epochs: int = 200,
    patience: int = 20,
) -> TrainingResult:
    """Fit on training nodes, select by validation loss, and restore the best state.

    Test masks and test metrics are deliberately absent from this interface.
    A strict decrease in validation loss resets patience; ties count as a
    non-improving epoch. The model is returned in evaluation mode with the
    selected parameters restored. Epoch numbers are one-based.

    Raises:
        ValueError: If epoch limits are invalid or train/validation masks are
            empty or overlapping.
        FloatingPointError: If a training or validation loss is not finite.
    """
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive")
    if not train_mask.any().item() or not val_mask.any().item():
        raise ValueError("Training and validation masks must both select nodes")
    if (train_mask & val_mask).any().item():
        raise ValueError("Training and validation masks must be disjoint")

    best_val_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] = {}
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        train_loss = criterion(logits[train_mask], targets[train_mask])
        if not torch.isfinite(train_loss).item():
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x)
            val_loss = float(criterion(val_logits[val_mask], targets[val_mask]).item())
            if not math.isfinite(val_loss):
                raise FloatingPointError(f"Non-finite validation loss at epoch {epoch}")
            val_metrics = evaluate_node_classification(val_logits, targets, val_mask)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            # state_dict() alone shares storage with live model parameters.
            best_state = deepcopy(model.state_dict())
            best_metrics = val_metrics.copy()
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        print(
            f"Epoch {epoch:03d}/{max_epochs} | train_loss={train_loss.item():.6f} | "
            f"val_loss={val_loss:.6f} | val_accuracy={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
            + (" | best" if improved else "")
        )
        if stale_epochs >= patience:
            print(f"Early stopping: no validation loss improvement for {patience} epochs.")
            break

    if best_state is None:
        raise RuntimeError("Training completed without a valid validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return TrainingResult(
        stopped_epoch=epoch,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        val_accuracy=best_metrics["accuracy"],
        val_macro_f1=best_metrics["macro_f1"],
    )


def write_run_log(path: Path, results: dict[str, str | int | float]) -> None:
    """Atomically write a JSON run record without allowing NaN or infinity."""
    payload = json.dumps(results, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    """Run the fixed CPU baseline, evaluate test once, and save the final record."""
    seed = 42
    learning_rate = 0.01
    weight_decay = 5e-4
    hidden_dim = 16
    dropout = 0.5
    set_seed(seed)

    # Absolute paths target repository data/ even when the working directory differs.
    data = get_cora_data(root=str(REPO_ROOT / "data"), normalize_features=True).cpu()
    model = MLP(
        in_channels=1433, hidden_channels=hidden_dim, out_channels=7, dropout=dropout
    ).cpu()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    training = fit_mlp(
        model, data.x, data.y, data.train_mask, data.val_mask,
        optimizer, criterion, max_epochs=200, patience=20,
    )

    # fit_mlp has already restored the best validation checkpoint.
    checkpoint_path = REPO_ROOT / "results" / "checkpoints" / "best_mlp_val.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

    # The only test-mask access and test evaluation in this script.
    model.eval()
    with torch.no_grad():
        final_logits = model(data.x)
        test_metrics = evaluate_node_classification(final_logits, data.y, data.test_mask)

    results: dict[str, str | int | float] = {
        "model": "MLP",
        "dataset": "Cora",
        "seed": seed,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "hidden_dim": hidden_dim,
        "num_layers": 2,
        "dropout": dropout,
        "stopped_epoch": training.stopped_epoch,
        "best_val_loss": training.best_val_loss,
        "val_accuracy": training.val_accuracy,
        "val_macro_f1": training.val_macro_f1,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
    }
    log_path = REPO_ROOT / "results" / "baseline_mlp_log.json"
    write_run_log(log_path, results)

    print(f"\nBest epoch: {training.best_epoch} | Stopped epoch: {training.stopped_epoch}")
    print(f"Best validation loss: {training.best_val_loss:.6f}")
    print(f"{'Split':<20} {'Accuracy':>10} {'Macro-F1':>10}")
    print("-" * 42)
    print(f"{'Validation (best)':<20} {training.val_accuracy:>10.4f} {training.val_macro_f1:>10.4f}")
    print(f"{'Test (final)':<20} {test_metrics['accuracy']:>10.4f} {test_metrics['macro_f1']:>10.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Run log: {log_path}")


if __name__ == "__main__":
    main()
