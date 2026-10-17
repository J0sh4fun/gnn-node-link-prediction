"""Week 8: Hyperparameter tuning for GAT on Cora node classification.

Following the project plan (docs/plans/project_plan.md, Section 7, Week 8):
- Grid search over learning rate, hidden dim, number of heads, weight decay, and dropout.
- STRICT PROTOCOL: Validation set is used exclusively for early stopping and configuration selection.
- Test set is evaluated exactly once on the single best chosen configuration.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from models.gat import GAT
from utils.data_loader import get_cora_data
from utils.evaluate import evaluate_node_classification
from utils.experiment import environment, mapping_hash, write_json
from scripts.nc_week08 import digest

REPO_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


@dataclass(frozen=True)
class ConfigResult:
    """Evaluation result for one hyperparameter configuration."""

    config_id: int
    hidden_dim: int
    heads: int
    learning_rate: float
    weight_decay: float
    dropout: float
    stopped_epoch: int
    best_epoch: int
    best_val_loss: float
    val_accuracy: float
    val_macro_f1: float


def fit_gat(
    model: GAT,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    targets: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    max_epochs: int = 200,
    patience: int = 20,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Fit GAT on training nodes with validation early stopping.

    Returns the validation metrics and the deepcopied best model state_dict.
    """
    best_val_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] = {}
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, edge_index)
        loss = criterion(logits[train_mask], targets[train_mask])
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x, edge_index)
            val_loss = float(criterion(val_logits[val_mask], targets[val_mask]).item())
            metrics = evaluate_node_classification(val_logits, targets, val_mask)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_metrics = metrics.copy()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("No checkpoint was saved during training")

    summary = {
        "stopped_epoch": epoch,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "val_accuracy": best_metrics["accuracy"],
        "val_macro_f1": best_metrics["macro_f1"],
    }
    return summary, best_state


def run_grid_search(
    data_root: Path | None = None,
    output_dir: Path | None = None,
    max_epochs: int = 200,
    patience: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute hyperparameter grid search over representative GAT configurations."""
    if data_root is None:
        data_root = REPO_ROOT / "data"
    if output_dir is None:
        output_dir = REPO_ROOT / "results" / "week08"
    output_dir = Path(output_dir)
    # A legacy final result is already a completed test evaluation. Never
    # overwrite it or evade it by choosing another output directory.
    for path in (REPO_ROOT / 'results/week08/gat_tuning.json', output_dir / 'gat_tuning.json'):
        if path.exists() and 'final_test_evaluation' in json.loads(path.read_text(encoding='utf-8')):
            raise RuntimeError('GAT final already exists; reuse saved results, do not rerun test')
    output_dir.mkdir(parents=True, exist_ok=True)
    # Claim before loading data, including after failures or concurrent calls.
    with (output_dir / 'gat_run_claim.json').open('x', encoding='utf-8') as handle:
        json.dump({'seed': seed, 'max_epochs': max_epochs, 'patience': patience}, handle)

    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Load Cora dataset
    data = get_cora_data(root=str(data_root), normalize_features=True).cpu()
    x = data.x
    edge_index = data.edge_index
    y = data.y
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    # Representative grid exploring hidden dimension, heads, lr, weight decay, and dropout
    grid = [
        # (hidden_dim, heads, lr, weight_decay, dropout)
        {"hidden_dim": 8, "heads": 8, "lr": 0.005, "weight_decay": 5e-4, "dropout": 0.6},
        {"hidden_dim": 8, "heads": 8, "lr": 0.01, "weight_decay": 5e-4, "dropout": 0.6},
        {"hidden_dim": 8, "heads": 4, "lr": 0.005, "weight_decay": 5e-4, "dropout": 0.6},
        {"hidden_dim": 8, "heads": 4, "lr": 0.01, "weight_decay": 1e-3, "dropout": 0.4},
        {"hidden_dim": 16, "heads": 4, "lr": 0.005, "weight_decay": 5e-4, "dropout": 0.6},
        {"hidden_dim": 16, "heads": 4, "lr": 0.01, "weight_decay": 1e-3, "dropout": 0.4},
        {"hidden_dim": 16, "heads": 8, "lr": 0.005, "weight_decay": 5e-4, "dropout": 0.4},
        {"hidden_dim": 16, "heads": 8, "lr": 0.01, "weight_decay": 5e-4, "dropout": 0.6},
    ]
    write_json(output_dir / 'gat_protocol.json', {
        'grid': grid, 'seed': seed, 'max_epochs': max_epochs, 'patience': patience,
        'selection': 'minimum best_val_loss; ties use grid order',
        'attention_dropout': 'same as candidate dropout', 'output_heads': 1, 'num_layers': 2,
        'split_id': 'cora_planetoid_public', 'normalize_features': True,
        'input_split_hash': mapping_hash({key: getattr(data, key).numpy() for key in
                                          ('x', 'edge_index', 'train_mask', 'val_mask')}),
    })
    write_json(output_dir / 'gat_environment.json', environment(REPO_ROOT))

    results: list[dict[str, Any]] = []
    best_config: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_overall_val_loss = math.inf

    criterion = nn.CrossEntropyLoss()

    for idx, params in enumerate(grid, start=1):
        set_seed(seed)
        model = GAT(
            in_channels=1433,
            hidden_dim=params["hidden_dim"],
            num_classes=7,
            heads=params["heads"],
            out_heads=1,
            dropout=params["dropout"],
            feature_dropout=params["dropout"],
            negative_slope=0.2,
            attention_dropout=params['dropout'],
        ).cpu()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )

        train_res, state = fit_gat(
            model=model,
            x=x,
            edge_index=edge_index,
            targets=y,
            train_mask=train_mask,
            val_mask=val_mask,
            optimizer=optimizer,
            criterion=criterion,
            max_epochs=max_epochs,
            patience=patience,
        )

        record = {
            "config_id": idx,
            "hidden_dim": params["hidden_dim"],
            "heads": params["heads"],
            "learning_rate": params["lr"],
            "weight_decay": params["weight_decay"],
            "dropout": params["dropout"],
            **train_res,
        }
        results.append(record)

        # Select by lowest validation loss
        if train_res["best_val_loss"] < best_overall_val_loss:
            best_overall_val_loss = train_res["best_val_loss"]
            best_config = record.copy()
            best_state = state

    if best_config is None or best_state is None:
        raise RuntimeError("Grid search completed with no valid configurations")

    # Save best checkpoint
    best_checkpoint_path = checkpoints_dir / "best_gat_val.pt"
    torch.save(best_state, best_checkpoint_path)

    # FINAL TEST EVALUATION (performed strictly ONCE on the selected best model)
    set_seed(seed)
    best_model = GAT(
        in_channels=1433,
        hidden_dim=best_config["hidden_dim"],
        num_classes=7,
        heads=best_config["heads"],
        out_heads=1,
        dropout=best_config["dropout"],
        feature_dropout=best_config["dropout"],
        attention_dropout=best_config['dropout'],
    ).cpu()
    best_model.load_state_dict(best_state)
    best_model.eval()

    with torch.no_grad():
        test_logits = best_model(x, edge_index)
        test_metrics = evaluate_node_classification(test_logits, y, test_mask)
    np.savez_compressed(output_dir / 'gat_predictions.npz',
                        node_id=test_mask.nonzero().flatten().numpy(),
                        y_true=y[test_mask].numpy(),
                        y_pred=test_logits[test_mask].argmax(-1).numpy(),
                        logits=test_logits[test_mask].numpy())
    np.savez_compressed(output_dir / 'gat_masks.npz', **{key: getattr(data, key).numpy()
                        for key in ('train_mask', 'val_mask', 'test_mask')})

    final_report = {
        "dataset": "Cora",
        "model": "GAT",
        "search_strategy": "grid_search",
        "seed": seed,
        "num_configurations": len(results),
        "configurations": results,
        "best_configuration": best_config,
        "checkpoint_sha256": digest(best_checkpoint_path),
        "predictions_sha256": digest(output_dir / 'gat_predictions.npz'),
        "masks_sha256": digest(output_dir / 'gat_masks.npz'),
        "final_test_evaluation": {
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
        },
    }

    # Write results atomically
    report_path = output_dir / "gat_tuning.json"
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_dir, delete=False) as tmp:
        tmp.write(json.dumps(final_report, indent=2) + "\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(report_path)

    return final_report


def main() -> None:
    """CLI entrypoint to run the hyperparameter grid search."""
    print("Running Week 8 GAT hyperparameter tuning on Cora...")
    report = run_grid_search()
    best = report["best_configuration"]
    test_eval = report["final_test_evaluation"]
    print(f"Tuning complete over {report['num_configurations']} configurations.")
    print(f"Best Config (ID {best['config_id']}): hidden_dim={best['hidden_dim']}, heads={best['heads']}, "
          f"lr={best['learning_rate']}, wd={best['weight_decay']}, dropout={best['dropout']}")
    print(f"Validation Loss: {best['best_val_loss']:.4f}, Val Accuracy: {best['val_accuracy']:.4f}, Val Macro-F1: {best['val_macro_f1']:.4f}")
    print(f"Final Test Evaluation: Accuracy = {test_eval['test_accuracy']:.4f}, Macro-F1 = {test_eval['test_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
