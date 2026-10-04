"""Synthetic-data tests for reusable training, checkpointing, and split isolation."""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
import textwrap
from typing import Any

import pytest
import torch
from torch import nn

from train import trainer as trainer_module
from train.trainer import NodeClassificationTrainer, TrainingResult


class FeatureModel(nn.Module):
    """Small feature-only model with a buffer to exercise full state restoration."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.register_buffer("training_calls", torch.tensor(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits and count training forwards in a persistent buffer."""
        if self.training:
            self.training_calls.add_(1)
        return self.linear(x)


class GraphModel(FeatureModel):
    """Lightweight graph interface that records the supplied connectivity."""

    def __init__(self) -> None:
        super().__init__()
        self.received_edges: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Require connectivity as a second argument and record each dispatch."""
        self.received_edges.append(edge_index)
        return super().forward(x)


class ScheduledLoss(nn.CrossEntropyLoss):
    """Use real gradients for training and a controlled validation loss sequence."""

    def __init__(self, validation_losses: list[float]) -> None:
        super().__init__()
        self.validation_losses = iter(validation_losses)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Assert that loss only sees the two selected nodes for each split."""
        assert input.shape == (2, 2)
        assert torch.equal(target, torch.tensor([0, 1]))
        if torch.is_grad_enabled():
            return super().forward(input, target)
        return input.new_tensor(next(self.validation_losses))


@dataclass
class ToyGraph:
    """Six nodes with two disjoint nodes in each evaluation split."""

    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor


@pytest.fixture
def graph() -> ToyGraph:
    """Build deterministic input tensors without downloads or GPU requirements."""
    return ToyGraph(
        x=torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 1.], [.2, .7], [.8, .1]]),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
        y=torch.tensor([0, 1, 0, 1, 0, 1]),
        train_mask=torch.tensor([True, True, False, False, False, False]),
        val_mask=torch.tensor([False, False, True, True, False, False]),
        test_mask=torch.tensor([False, False, False, False, True, True]),
    )


def make_trainer(
    tmp_path: Path,
    losses: list[float],
    *,
    use_graph: bool = False,
    max_epochs: int = 10,
    patience: int = 2,
) -> NodeClassificationTrainer:
    """Construct an isolated trainer with deterministic parameters and output paths."""
    torch.manual_seed(7)
    model = GraphModel() if use_graph else FeatureModel()
    return NodeClassificationTrainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        ScheduledLoss(losses),
        max_epochs=max_epochs,
        patience=patience,
        checkpoint_dir=tmp_path / "nested/checkpoints",
        log_path=tmp_path / "nested/logs/run.json",
        run_config={"model": type(model).__name__, "dataset": "synthetic", "seed": 7},
    )


@pytest.mark.parametrize("use_graph", [False, True], ids=["features", "graph"])
@pytest.mark.parametrize(
    ("losses", "max_epochs", "expected_best", "expected_stop"),
    [
        ([1.0, 1.0, 1.0], 10, 1, 3),
        ([1.0, 0.5, 0.5, 0.7], 10, 2, 4),
        ([1.0, 0.5, 0.8, 0.4, 0.4, 0.4], 10, 4, 6),
        ([1.0, 0.5, 0.8], 3, 2, 3),
    ],
)
def test_early_stopping_and_best_state_restoration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    graph: ToyGraph,
    use_graph: bool,
    losses: list[float],
    max_epochs: int,
    expected_best: int,
    expected_stop: int,
) -> None:
    """Plateaus consume patience; improvements reset it; parameters and buffers restore."""
    trainer = make_trainer(tmp_path, losses, use_graph=use_graph, max_epochs=max_epochs)
    snapshots: list[dict[str, Any]] = []
    metric_history: list[dict[str, float]] = []
    original_evaluate = trainer_module.evaluate_node_classification

    def record_validation(
        logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, float]:
        """Record validation states while forbidding any other mask during fitting."""
        assert torch.equal(mask, graph.val_mask)
        assert not trainer.model.training
        assert not torch.is_grad_enabled()
        snapshots.append(deepcopy(trainer.model.state_dict()))
        metrics = original_evaluate(logits, targets, mask)
        metric_history.append(metrics)
        return metrics

    monkeypatch.setattr(trainer_module, "evaluate_node_classification", record_validation)
    edges = graph.edge_index if use_graph else None
    result = trainer.fit(graph.x, edges, graph.y, graph.train_mask, graph.val_mask)

    assert isinstance(result, TrainingResult)
    assert result.best_epoch == expected_best
    assert result.stopped_epoch == expected_stop
    assert result.stopped_early == (expected_stop < max_epochs)
    assert result.best_val_loss == pytest.approx(losses[expected_best - 1])
    assert result.val_accuracy == metric_history[expected_best - 1]["accuracy"]
    assert result.val_macro_f1 == metric_history[expected_best - 1]["macro_f1"]
    assert len(result.history) == expected_stop
    assert not trainer.model.training
    assert any(
        not torch.equal(snapshots[-1][key], snapshots[expected_best - 1][key])
        for key in snapshots[-1]
    )

    saved = torch.load(trainer.checkpoint_path, weights_only=True)
    for key, expected in snapshots[expected_best - 1].items():
        assert torch.equal(trainer.model.state_dict()[key], expected)
        assert torch.equal(saved[key], expected)
    if use_graph:
        assert len(trainer.model.received_edges) == 2 * expected_stop
        assert all(edge is edges for edge in trainer.model.received_edges)

    record = json.loads(trainer.log_path.read_text())
    assert record["status"] == "fitted"
    assert record["seed"] == 7
    assert record["best_epoch"] == expected_best
    assert not any(key.startswith("test_") for key in record)
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("invalid", ["empty_train", "empty_val", "overlap", "dtype", "shape"])
def test_invalid_fit_masks_raise(
    tmp_path: Path, graph: ToyGraph, invalid: str
) -> None:
    """Empty, overlapping, or malformed masks fail before optimization and file writes."""
    trainer = make_trainer(tmp_path, [1.0])
    train_mask, val_mask = graph.train_mask.clone(), graph.val_mask.clone()
    if invalid == "empty_train":
        train_mask.zero_()
    elif invalid == "empty_val":
        val_mask.zero_()
    elif invalid == "overlap":
        val_mask[0] = True
    elif invalid == "dtype":
        train_mask = train_mask.long()
    else:
        val_mask = val_mask[:-1]

    with pytest.raises(ValueError):
        trainer.fit(graph.x, None, graph.y, train_mask, val_mask)
    assert trainer.model.training_calls.item() == 0
    assert not trainer.checkpoint_path.exists()
    assert not trainer.log_path.exists()


@pytest.mark.parametrize("use_graph", [False, True])
def test_final_evaluation_restores_state_and_updates_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, graph: ToyGraph, use_graph: bool
) -> None:
    """Final evaluation restores even externally modified weights and runs only once."""
    trainer = make_trainer(tmp_path, [1.0], use_graph=use_graph, max_epochs=1)
    edges = graph.edge_index if use_graph else None
    trainer.fit(graph.x, edges, graph.y, graph.train_mask, graph.val_mask)
    fitted_log = json.loads(trainer.log_path.read_text())
    selected = deepcopy(trainer.model.state_dict())
    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.add_(100)
        trainer.model.training_calls.add_(100)
    trainer.model.train()
    calls: list[torch.Tensor] = []
    original_evaluate = trainer_module.evaluate_node_classification

    def check_restored(
        logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, float]:
        """Assert restoration, evaluation mode, and mask isolation at metric time."""
        assert not trainer.model.training
        assert not torch.is_grad_enabled()
        assert torch.equal(mask, graph.test_mask)
        for key, expected in selected.items():
            assert torch.equal(trainer.model.state_dict()[key], expected)
        calls.append(mask)
        return original_evaluate(logits, targets, mask)

    monkeypatch.setattr(trainer_module, "evaluate_node_classification", check_restored)
    result = trainer.evaluate_test(graph.x, edges, graph.y, graph.test_mask)
    assert set(result) == {"accuracy", "macro_f1"}
    assert all(isinstance(value, float) for value in result.values())
    final_log = json.loads(trainer.log_path.read_text())
    assert final_log == {
        **fitted_log, "status": "evaluated",
        "test_accuracy": result["accuracy"], "test_macro_f1": result["macro_f1"],
    }
    with pytest.raises(RuntimeError, match="already been performed"):
        trainer.evaluate_test(graph.x, edges, graph.y, graph.test_mask)
    assert len(calls) == 1
    assert not list(tmp_path.rglob("*.tmp"))


def test_evaluation_requires_successful_fit(tmp_path: Path, graph: ToyGraph) -> None:
    """A fresh trainer cannot accidentally reuse a stale checkpoint from another run."""
    trainer = make_trainer(tmp_path, [1.0])
    with pytest.raises(RuntimeError, match="fit"):
        trainer.evaluate_test(graph.x, None, graph.y, graph.test_mask)


@pytest.mark.parametrize("mask_name", ["empty", "train", "validation"])
def test_invalid_evaluation_masks_raise(
    tmp_path: Path, graph: ToyGraph, mask_name: str
) -> None:
    """Final evaluation rejects empty masks and previously used nodes."""
    trainer = make_trainer(tmp_path, [1.0], max_epochs=1)
    trainer.fit(graph.x, None, graph.y, graph.train_mask, graph.val_mask)
    mask = {"empty": torch.zeros_like(graph.test_mask),
            "train": graph.train_mask, "validation": graph.val_mask}[mask_name]
    with pytest.raises(ValueError):
        trainer.evaluate_test(graph.x, None, graph.y, mask)


def test_nonfinite_validation_blocks_final_evaluation(tmp_path: Path, graph: ToyGraph) -> None:
    """A partially completed fit cannot authorize final evaluation."""
    trainer = make_trainer(tmp_path, [1.0, float("nan")])
    with pytest.raises(FloatingPointError, match="validation"):
        trainer.fit(graph.x, None, graph.y, graph.train_mask, graph.val_mask)
    with pytest.raises(RuntimeError, match="fit"):
        trainer.evaluate_test(graph.x, None, graph.y, graph.test_mask)


def test_final_log_retry_does_not_reevaluate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, graph: ToyGraph
) -> None:
    """A transient logging failure can be recovered without recomputing held-out metrics."""
    trainer = make_trainer(tmp_path, [1.0], max_epochs=1)
    trainer.fit(graph.x, None, graph.y, graph.train_mask, graph.val_mask)
    original_write = trainer._write_log
    original_evaluate = trainer_module.evaluate_node_classification
    calls: list[str] = []

    def fail_log(status: str) -> None:
        """Simulate unavailable storage during final publication."""
        raise OSError("storage unavailable")

    def count_evaluations(
        logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, float]:
        """Count actual metric computations across the failed call and its retry."""
        calls.append("evaluation")
        return original_evaluate(logits, targets, mask)

    monkeypatch.setattr(trainer_module, "evaluate_node_classification", count_evaluations)
    monkeypatch.setattr(trainer, "_write_log", fail_log)
    with pytest.raises(OSError, match="storage unavailable"):
        trainer.evaluate_test(graph.x, None, graph.y, graph.test_mask)
    monkeypatch.setattr(trainer, "_write_log", original_write)
    metrics = trainer.evaluate_test(graph.x, None, graph.y, graph.test_mask)
    assert calls == ["evaluation"]
    assert json.loads(trainer.log_path.read_text())["test_accuracy"] == metrics["accuracy"]


def test_new_fit_resets_selection_and_final_metrics(tmp_path: Path, graph: ToyGraph) -> None:
    """A later fit selects its own best state even when its loss exceeds the old best."""
    trainer = make_trainer(tmp_path, [0.1], max_epochs=1)
    trainer.fit(graph.x, None, graph.y, graph.train_mask, graph.val_mask)
    trainer.evaluate_test(graph.x, None, graph.y, graph.test_mask)
    trainer.criterion = ScheduledLoss([2.0])
    result = trainer.fit(graph.x, None, graph.y, graph.train_mask, graph.val_mask)
    assert result.best_val_loss == 2.0
    assert len(result.history) == 1
    record = json.loads(trainer.log_path.read_text())
    assert record["status"] == "fitted"
    assert "test_accuracy" not in record and "test_macro_f1" not in record
    trainer.evaluate_test(graph.x, None, graph.y, graph.test_mask)


@pytest.mark.parametrize("artifact", ["checkpoint", "log"])
def test_atomic_failure_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: str
) -> None:
    """Failed replacement leaves the old checkpoint/JSON intact and removes temp files."""
    trainer = make_trainer(tmp_path, [1.0])
    target = trainer.checkpoint_path if artifact == "checkpoint" else trainer.log_path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"previous artifact")
    original_replace = Path.replace

    def fail_replace(source: Path, destination: Path) -> Path:
        """Inject a failure after serialization but before publication."""
        if destination == target:
            raise OSError("replacement failed")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failed"):
        if artifact == "checkpoint":
            trainer_module._atomic_write(
                target, lambda handle: torch.save(trainer.model.state_dict(), handle)
            )
        else:
            trainer._write_log("running")
    assert target.read_bytes() == b"previous artifact"
    assert not list(tmp_path.rglob("*.tmp"))


def test_fit_has_no_test_mask_interface_or_reference() -> None:
    """Guard the structural boundary between fitting and final evaluation."""
    assert "test_mask" not in inspect.signature(NodeClassificationTrainer.fit).parameters
    tree = ast.parse(textwrap.dedent(inspect.getsource(NodeClassificationTrainer.fit)))
    assert not any(
        (isinstance(node, ast.Name) and node.id == "test_mask")
        or (isinstance(node, ast.Attribute) and node.attr == "test_mask")
        for node in ast.walk(tree)
    )
