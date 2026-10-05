"""Short validation-only checks of the existing node-classification pipeline."""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import torch

from models.mlp import MLP
from train.train_mlp import fit_mlp, set_seed, write_run_log
from utils.data_loader import get_cora_data
from utils.paths import require_cora_cache


def smoke_node_validation(data_root: Path, output_dir: Path, max_epochs: int = 2) -> dict[str, Any]:
    """Check cached Cora, masked fitting, validation, and checkpoint round-trip.

    This is a smoke check, not a replacement baseline run. No held-out labels
    are evaluated. Masks are inspected only for their structural invariants.
    """
    require_cora_cache(data_root)
    set_seed(42)
    data = get_cora_data(root=str(data_root), normalize_features=True).cpu()
    assert data.x.shape == (2708, 1433) and data.y.shape == (2708,)
    masks = [data.train_mask, data.val_mask, data.test_mask]
    assert [int(mask.sum()) for mask in masks] == [140, 500, 1000]
    assert not any((masks[i] & masks[j]).any().item() for i, j in ((0, 1), (0, 2), (1, 2)))
    assert torch.allclose(data.x.sum(dim=1), torch.ones(data.num_nodes))
    assert data.is_undirected()
    model = MLP(in_channels=1433, hidden_channels=16, out_channels=7, dropout=0.5).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    training = fit_mlp(model, data.x, data.y, data.train_mask, data.val_mask,
                       optimizer, torch.nn.CrossEntropyLoss(), max_epochs=max_epochs, patience=20)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best_mlp_val.pt"
    torch.save(model.state_dict(), checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert all(torch.equal(value, saved[key]) for key, value in model.state_dict().items())
    model.load_state_dict(saved)
    result = {"model": "MLP", "dataset": "Cora", "seed": 42,
              "evaluation_split": "validation", **asdict(training)}
    log_path = output_dir / "validation_smoke.json"
    write_run_log(log_path, result)
    assert json.loads(log_path.read_text(encoding="utf-8")) == result
    return result
