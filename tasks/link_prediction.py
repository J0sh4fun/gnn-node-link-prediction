"""Saved-split cosine validation; no split regeneration or held-out scoring."""

from pathlib import Path
import time
from typing import Any

import numpy as np
from torch_geometric.datasets import Planetoid

from utils.experiment import array_hash, mapping_hash, new_run, publish_result, read_json
from utils.link_graph import audit_split, bidirectional, canonicalize, load_cora
from utils.link_metrics import cosine_scores, lp_metrics
from utils.paths import require_cora_cache


def load_saved_split(root: Path, data: Any, config: dict[str, Any]) -> tuple[dict, dict, dict]:
    """Load and verify saved arrays against raw Cora and their original manifest.

    Held-out pairs are read only for partition/exclusion audits, never scored.
    Assertion-based checks retain the notebook contract; do not run with ``-O``.
    """
    full_pairs = canonicalize(data.edge_index.numpy(), data.num_nodes)
    fingerprints = {"features": array_hash(data.x.numpy()), "full_pairs": array_hash(full_pairs)}
    split_dir = Path(root) / "artifacts/splits" / config["split_id"]
    manifest = read_json(split_dir / "manifest.json")
    with np.load(split_dir / "edges.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    assert manifest["input_fingerprints"] == fingerprints, "Dataset fingerprint changed"
    assert manifest["config"] == config, "Protocol changed: use a new split ID"
    assert mapping_hash(arrays) == manifest["split_hash"], "Split hash mismatch"
    assert {key: array_hash(value) for key, value in arrays.items()} == manifest["array_hashes"]
    assert manifest["counts"] == {key: value.shape[1] for key, value in arrays.items()}
    assert manifest["num_nodes"] == data.num_nodes
    adjacency = bidirectional(arrays["train_pos"], data.num_nodes)
    audit_split(arrays, full_pairs, data.num_nodes, adjacency)
    return arrays, manifest, fingerprints


def run_cosine_validation(
    root: Path, *, output_root: Path | None = None, data_root: Path | None = None,
    require_cached: bool = True, checks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the existing raw-float64 cosine baseline on fixed validation pairs.

    ``output_root=None`` preserves the historical notebook publication locations.
    CLIs require a separate output root. ``data_root`` is the direct Planetoid
    cache root; its default remains ``<repo>/data/cora`` for this task.
    Fingerprint checks protect node ordering/features when reusing another cache.
    """
    root = Path(root).resolve()
    destination = root if output_root is None else Path(output_root).resolve()
    config = read_json(root / "configs/cosine_baseline.json")
    data_config = read_json(root / "configs/data_protocol.json")
    cache_root = root / "data/cora" if data_root is None else Path(data_root).resolve()
    if require_cached:
        require_cora_cache(cache_root)
    if data_root is None:
        data = load_cora(root, data_config)
    else:
        assert data_config["dataset"] == "Cora" and data_config["public_node_split"] == "public"
        data = Planetoid(root=str(cache_root), name="Cora", split="public")[0]
    arrays, manifest, fingerprints = load_saved_split(root, data, data_config)
    audit = read_json(root / "results/week02/leakage_audit.json")
    assert audit["status"] == "passed" and audit["split_hash"] == manifest["split_hash"]
    assert audit["input_fingerprints"] == fingerprints
    assert config["split_id"] == manifest["split_id"] and config["evaluation_split"] == "validation"
    assert config["feature_preprocessing"] == "raw_float64"
    pairs = np.concatenate([arrays["val_pos"], arrays["val_neg"]], axis=1)
    labels = np.concatenate([np.ones(arrays["val_pos"].shape[1]), np.zeros(arrays["val_neg"].shape[1])])
    checks = {} if checks is None else dict(checks)
    run = new_run(root, "w3_cosine", config, output_root=destination)
    start = time.perf_counter()
    scores = cosine_scores(data.x.numpy(), pairs, config["epsilon"], config["chunk_size"])
    metrics = lp_metrics(labels, scores)
    elapsed = time.perf_counter() - start
    np.savez_compressed(run / "validation_predictions.npz", pairs=pairs, labels=labels, scores=scores)
    previous_path = root / "results/week03/cosine_validation.json"
    if previous_path.exists():
        previous = read_json(previous_path)
        if previous["config"] == config and previous["split_hash"] == manifest["split_hash"]:
            previous_file = root / "runs" / previous["run_id"] / "validation_predictions.npz"
            if previous_file.exists():
                with np.load(previous_file, allow_pickle=False) as old:
                    np.testing.assert_allclose(scores, old["scores"], atol=config["atol"], rtol=config["rtol"])
                    assert np.array_equal(labels, old["labels"])
                np.testing.assert_allclose(
                    [metrics[key] for key in ("roc_auc", "average_precision")],
                    [previous["validation"][key] for key in ("roc_auc", "average_precision")],
                    atol=config["atol"], rtol=config["rtol"],
                )
                checks["independent_run_reproduction"] = {
                    "status": "passed", "previous_run_id": previous["run_id"],
                    "atol": config["atol"], "rtol": config["rtol"],
                }
    # publish_result expects results/ to exist before opening experiment_index.csv.
    (destination / "results").mkdir(parents=True, exist_ok=True)
    return publish_result(
        destination, run, "results/week03/cosine_validation.json",
        {"validation": metrics, "validation_metric": "roc_auc", "validation_value": metrics["roc_auc"],
         "elapsed_seconds": elapsed, "input_fingerprints": fingerprints, "checks": checks,
         "evaluation_split": "validation"}, config, 3, "W3.1", "cosine", "Cora", manifest,
    )
