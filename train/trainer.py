"""Reusable node-classification training with validation-only model selection."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, BinaryIO, Callable

import torch
from torch import nn

from utils.evaluate import evaluate_node_classification


@dataclass(frozen=True)
class EpochResult:
    """Training and validation measurements from one completed epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float


@dataclass(frozen=True)
class TrainingResult:
    """Selected validation metrics, stopping information, and epoch history."""

    stopped_epoch: int
    best_epoch: int
    best_val_loss: float
    val_accuracy: float
    val_macro_f1: float
    stopped_early: bool
    history: tuple[EpochResult, ...]


def _atomic_write(path: Path, writer: Callable[[BinaryIO], object]) -> None:
    """Replace a file only after its temporary sibling has been written and closed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(mode="wb", dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            writer(handle)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class NodeClassificationTrainer:
    """Train a feature-only or graph model using strictly validation-based selection.

    Pass ``edge_index=None`` for ``model(x)`` or a tensor for
    ``model(x, edge_index)``. The caller sets seeds and places the model, loss
    weights, and tensors on the desired device before fitting.

    Output paths are resolved at construction, relative to the current working
    directory when necessary. Use separate paths for concurrent experiments.
    ``run_config`` must be JSON-serializable and cannot override result fields.

    Every fit starts a new run record and selection history; it does not reset
    model weights or optimizer state. Use fresh instances for independent runs.
    Final evaluation is permitted once per successful fit. The in-memory best
    state is authoritative; ``best_model.pt`` exports the same state for reuse.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.CrossEntropyLoss,
        max_epochs: int = 200,
        patience: int = 20,
        *,
        checkpoint_dir: Path,
        log_path: Path,
        run_config: dict[str, Any],
    ) -> None:
        if max_epochs < 1 or patience < 1:
            raise ValueError("max_epochs and patience must be positive")
        reserved = {field.name for field in fields(TrainingResult)} | {
            "status", "test_accuracy", "test_macro_f1"
        }
        if reserved.intersection(run_config):
            raise ValueError("run_config cannot override training or evaluation result fields")
        json.dumps(run_config, allow_nan=False)
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        self.checkpoint_path = self.checkpoint_dir / "best_model.pt"
        self.log_path = Path(log_path).expanduser().resolve()
        if self.log_path == self.checkpoint_path:
            raise ValueError("log_path must differ from the checkpoint path")
        self.run_config = deepcopy(run_config)
        self.result: TrainingResult | None = None
        self._best_state: dict[str, Any] | None = None
        self._selection_mask: torch.Tensor | None = None
        self._final_metrics: dict[str, float] | None = None
        self._final_logged = False

    def _forward(self, x: torch.Tensor, edge_index: torch.Tensor | None) -> torch.Tensor:
        """Dispatch explicitly without swallowing exceptions raised by the model."""
        if edge_index is None:
            return self.model(x)
        return self.model(x, edge_index)

    @staticmethod
    def _validate_inputs(x: torch.Tensor, y: torch.Tensor) -> None:
        """Require aligned node features and labels on the same device."""
        if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
            raise ValueError("Expected x [num_nodes, features] and y [num_nodes]")
        if x.device != y.device:
            raise ValueError("Features and labels must be on the same device")

    @staticmethod
    def _validate_mask(mask: torch.Tensor, y: torch.Tensor, name: str) -> None:
        """Require a nonempty boolean node mask aligned with the label tensor."""
        if mask.dtype != torch.bool or mask.shape != y.shape:
            raise ValueError(f"{name} must be boolean with shape [num_nodes]")
        if mask.device != y.device:
            raise ValueError(f"{name} and labels must be on the same device")
        if not mask.any().item():
            raise ValueError(f"{name} must select at least one node")

    def _write_log(self, status: str) -> None:
        """Atomically publish configuration and only the metrics already available."""
        record = {**self.run_config, "status": status}
        if self.result is not None:
            record.update(asdict(self.result))
        if self._final_metrics is not None:
            record.update({f"test_{key}": value for key, value in self._final_metrics.items()})
        payload = (json.dumps(record, indent=2, allow_nan=False) + "\n").encode("utf-8")
        _atomic_write(self.log_path, lambda handle: handle.write(payload))

    def fit(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        y: torch.Tensor,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
    ) -> TrainingResult:
        """Optimize training loss and restore the state with lowest validation loss.

        Ties count as non-improvements. Patience is the number of consecutive
        non-improving epochs tolerated before stopping. Epochs are one-based.
        Validation runs in evaluation mode with gradients disabled. Checkpoints
        include parameters and buffers, with an independent copy kept in memory.

        Raises:
            ValueError: If masks are empty, malformed, or overlap.
            FloatingPointError: If training or validation loss is non-finite.
        """
        self._validate_inputs(x, y)
        self._validate_mask(train_mask, y, "train_mask")
        self._validate_mask(val_mask, y, "val_mask")
        if (train_mask & val_mask).any().item():
            raise ValueError("Training and validation masks must not overlap")

        self.result = None
        self._best_state = None
        self._final_metrics = None
        self._final_logged = False
        self._selection_mask = (train_mask | val_mask).detach().clone()
        self._write_log("running")
        best_loss = math.inf
        best_epoch = 0
        best_metrics: dict[str, float] = {}
        stale_epochs = 0
        history: list[EpochResult] = []

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            logits = self._forward(x, edge_index)
            loss = self.criterion(logits[train_mask], y[train_mask])
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_logits = self._forward(x, edge_index)
                val_loss = float(self.criterion(val_logits[val_mask], y[val_mask]).item())
                if not math.isfinite(val_loss):
                    raise FloatingPointError(f"Non-finite validation loss at epoch {epoch}")
                metrics = evaluate_node_classification(val_logits, y, val_mask)

            history.append(EpochResult(
                epoch, float(loss.item()), val_loss, metrics["accuracy"], metrics["macro_f1"]
            ))
            if val_loss < best_loss:
                state = deepcopy(self.model.state_dict())
                _atomic_write(self.checkpoint_path, lambda handle: torch.save(state, handle))
                self._best_state = state
                best_loss, best_epoch, best_metrics = val_loss, epoch, metrics.copy()
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= self.patience:
                break

        if self._best_state is None:
            raise RuntimeError("No valid validation checkpoint was created")
        self.model.load_state_dict(self._best_state)
        self.model.eval()
        result = TrainingResult(
            stopped_epoch=epoch,
            best_epoch=best_epoch,
            best_val_loss=best_loss,
            val_accuracy=best_metrics["accuracy"],
            val_macro_f1=best_metrics["macro_f1"],
            stopped_early=epoch < self.max_epochs,
            history=tuple(history),
        )
        self.result = result
        try:
            self._write_log("fitted")
        except Exception:
            self.result = None
            raise
        return result

    def evaluate_test(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        y: torch.Tensor,
        test_mask: torch.Tensor,
    ) -> dict[str, float]:
        """Restore the selected state, evaluate once, and atomically publish final metrics.

        Inputs must use the same node ordering and devices as during fitting.
        Raises ValueError for empty, malformed, or overlapping evaluation masks,
        and RuntimeError before a successful fit or after a completed evaluation.
        If final logging fails, a retry writes cached metrics without reevaluation.
        """
        if self.result is None or self._best_state is None or self._selection_mask is None:
            raise RuntimeError("fit() must complete successfully before evaluation")
        if self._final_metrics is not None:
            if self._final_logged:
                raise RuntimeError("Final evaluation has already been performed for this fit")
            # Retry a failed disk write without looking at held-out labels again.
            self._write_log("evaluated")
            self._final_logged = True
            return self._final_metrics.copy()
        self._validate_inputs(x, y)
        self._validate_mask(test_mask, y, "test_mask")
        if self._selection_mask.shape != y.shape or self._selection_mask.device != y.device:
            raise ValueError("Evaluation nodes and device must match the fitted graph")
        if (test_mask & self._selection_mask).any().item():
            raise ValueError("Evaluation mask must not overlap training or validation nodes")

        self.model.load_state_dict(self._best_state)
        self.model.eval()
        with torch.no_grad():
            logits = self._forward(x, edge_index)
            metrics = evaluate_node_classification(logits, y, test_mask)
        self._final_metrics = metrics.copy()
        self._write_log("evaluated")
        self._final_logged = True
        return metrics
