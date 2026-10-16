"""Week 7 Cora validation runs; no final-test evaluation entry point."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from models import GAT, GCN
from train.trainer import NodeClassificationTrainer
from utils.data_loader import get_cora_data
from utils.experiment import environment, mapping_hash, seed_everything, write_json
from utils.paths import isolated_output_root, repository_root, require_cora_cache


def fit_validation(data, model_name: str, config: dict, output: Path,
                   provenance: dict) -> dict:
    """Fit one fresh model using train/validation labels only.

    The test mask is never accessed. Refuse existing output directories so a
    later run cannot silently replace a selected checkpoint or its evidence.
    """
    if model_name not in ("gcn", "gat"):
        raise ValueError("model must be gcn or gat")
    seed_everything(config["seed"])
    torch.set_num_threads(config["num_threads"])
    params = config["models"][model_name]
    common = dict(in_channels=data.x.size(1), hidden_channels=params["hidden_dim"],
                  out_channels=7, dropout=params["dropout"])
    model = GCN(**common) if model_name == "gcn" else GAT(
        **common, heads=params["num_heads"], output_heads=params["output_heads"],
        attention_dropout=params["attention_dropout"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    split_hash = mapping_hash({name: getattr(data, name).cpu().numpy()
                               for name in ("train_mask", "val_mask")})
    input_hash = mapping_hash({name: getattr(data, name).cpu().numpy()
                               for name in ("x", "edge_index")})
    record = dict(model=model_name.upper(), dataset="Cora", seed=config["seed"],
                  num_layers=2, decoder=None,
                  **{"num_heads": None, **params}, max_epochs=config["max_epochs"], patience=config["patience"],
                  num_threads=config["num_threads"], device="cpu",
                  evaluation_split="validation", test_score=None,
                  checkpoint_metric="val_loss", checkpoint_mode="min",
                  split_id="cora_planetoid_public", split_hash=split_hash,
                  input_hash=input_hash, normalize_features=True,
                  graph="full undirected raw edges; self-loops inside layers")
    write_json(output / "config.json", record)
    write_json(output / "environment.json", provenance)
    trainer = NodeClassificationTrainer(
        model, torch.optim.Adam(model.parameters(), lr=params["learning_rate"],
                                weight_decay=params["weight_decay"]),
        torch.nn.CrossEntropyLoss(), max_epochs=config["max_epochs"],
        patience=config["patience"], checkpoint_dir=output,
        log_path=output / "training.json", run_config=record)
    result = trainer.fit(data.x, data.edge_index, data.y, data.train_mask, data.val_mask)
    saved = torch.load(trainer.checkpoint_path, map_location="cpu", weights_only=True)
    if not all(torch.equal(value, saved[key]) for key, value in model.state_dict().items()):
        raise RuntimeError("Restored model differs from exported best checkpoint")
    summary = {**record, **asdict(result), "checkpoint_path": str(trainer.checkpoint_path),
               "checkpoint_sha256": hashlib.sha256(trainer.checkpoint_path.read_bytes()).hexdigest(),
               "checkpoint_roundtrip": True,
               "validation_score": {"accuracy": result.val_accuracy, "macro_f1": result.val_macro_f1}}
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("gcn", "gat", "all"), default="all")
    parser.add_argument("--config", type=Path, default=repository_root() / "configs/node_classification_week07.json")
    parser.add_argument("--data-root", type=Path, default=repository_root() / "data")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["dataset"] != "Cora":
        parser.error("Week 7 requires Cora")
    output = isolated_output_root(args.output_dir, repository_root())
    require_cora_cache(args.data_root)
    data = get_cora_data(str(args.data_root), normalize_features=True).cpu()
    provenance = environment(repository_root())
    for name in (("gcn", "gat") if args.model == "all" else (args.model,)):
        result = fit_validation(data, name, config, output / name, provenance)
        print(json.dumps({key: result[key] for key in
                          ("model", "best_epoch", "val_accuracy", "val_macro_f1", "test_score")}))


if __name__ == "__main__":
    main()
