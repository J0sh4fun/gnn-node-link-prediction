"""Regression tests for code extracted from Phase 1 notebooks; no downloads."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from layers.projected_sum import ProjectedSum
from train.callback_trainer import EarlyStopping, fit
from utils.attention import receiver_softmax
from utils.experiment import array_hash, mapping_hash
from utils.link_graph import audit_split, bidirectional, canonicalize, make_split, with_self_loops
from utils.link_metrics import cosine_scores, lp_metrics

ROOT = Path(__file__).resolve().parents[1]


def test_saved_split_hashes_and_leakage_contract() -> None:
    """The committed split is audited in memory and never regenerated or rewritten."""
    directory = ROOT / "artifacts/splits/cora_lp_v1"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    with np.load(directory / "edges.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    assert mapping_hash(arrays) == manifest["split_hash"]
    assert {key: array_hash(value) for key, value in arrays.items()} == manifest["array_hashes"]
    assert {key: value.shape[1] for key, value in arrays.items()} == manifest["counts"]
    full = np.concatenate([arrays[f"{part}_pos"] for part in ("train", "val", "test")], axis=1)
    assert array_hash(canonicalize(full, manifest["num_nodes"])) == manifest["input_fingerprints"]["full_pairs"]
    adjacency = bidirectional(arrays["train_pos"], manifest["num_nodes"])
    audit_split(arrays, full, manifest["num_nodes"], adjacency)
    leaked = np.concatenate([adjacency, arrays["val_pos"][:, :1]], axis=1)
    with pytest.raises(AssertionError):
        audit_split(arrays, full, manifest["num_nodes"], leaked)


def test_numpy_edge_policy_and_toy_split() -> None:
    """NumPy utilities deduplicate all edges; toy splits retain deterministic order."""
    edges = np.array([[1, 0, 1, 2, 2], [0, 1, 0, 1, 2]], dtype=np.int64)
    canonical = canonicalize(edges, 4)
    np.testing.assert_array_equal(canonical, [[0, 1], [1, 2]])
    loops = with_self_loops(edges, 4)
    np.testing.assert_array_equal(with_self_loops(loops, 4), loops)
    assert loops.shape[1] == 7
    config = json.loads((ROOT / "configs/data_protocol.json").read_text(encoding="utf-8"))
    # Synthetic input only: canonical Cora splits are immutable.
    pairs = np.array([np.arange(20), np.arange(1, 21)], dtype=np.int64)
    first = make_split(pairs, 30, config)
    second = make_split(pairs[:, ::-1], 30, config)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
        assert first[key].dtype == np.int64
    assert [first[f"{part}_pos"].shape[1] for part in ("train", "val", "test")] == [17, 1, 2]
    audit_split(first, pairs, 30, bidirectional(first["train_pos"], 30))


def test_cosine_and_metrics_preserve_notebook_semantics() -> None:
    """Float64 chunked cosine includes zero vectors; ranking ties use sklearn AP."""
    features = np.array([[1, 0], [1, 0], [0, 1], [0, 0]], dtype=np.float32)
    pairs = np.array([[0, 0, 0, 1], [1, 2, 3, 0]])
    scores = cosine_scores(features, pairs, chunk_size=1)
    assert scores.dtype == np.float64
    np.testing.assert_array_equal(scores, [1, 0, 0, 1])
    np.testing.assert_array_equal(scores, cosine_scores(features, pairs))
    assert lp_metrics([0, 1], [1, 1]) == {
        "roc_auc": 0.5, "average_precision": 0.5, "n_positive": 1, "n_negative": 1,
    }
    with pytest.raises(ValueError):
        lp_metrics([1, 1], [0, 1])
    with pytest.raises(ValueError):
        cosine_scores(features, np.empty((2, 0), dtype=int))


def test_existing_message_passing_and_attention_gradients() -> None:
    """Moving primitives must retain source-to-target aggregation and autograd."""
    from torch_geometric.utils import softmax

    layer = ProjectedSum(2, 2)
    with torch.no_grad():
        layer.projection.weight.copy_(torch.eye(2))
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [2., 2.]], requires_grad=True)
    edge = torch.tensor([[0, 2, 1], [1, 1, 2]])
    out = layer(x, edge)
    torch.testing.assert_close(out, torch.tensor([[0., 0.], [2., 1.], [0., 1.], [0., 0.]]))
    out.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.projection.weight.grad is not None
    scores = torch.tensor([[10000., -10000.], [10001., -9998.], [7., 9.]],
                          dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([1, 1, 2])
    alpha = receiver_softmax(scores, targets, 4)
    torch.testing.assert_close(alpha, softmax(scores, targets, num_nodes=4))
    alpha.square().sum().backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


def test_callback_stopper_and_checkpoint(tmp_path: Path) -> None:
    """The callback trainer keeps zero-based epochs, min_delta, and its own schema."""
    stopper = EarlyStopping(mode="max", patience=2, min_delta=0.1)
    assert stopper.update(0.5, 0) == (True, False)
    assert stopper.update(0.55, 1) == (False, False)
    assert stopper.update(0.5, 2) == (False, True)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.ones(2, 1)
    losses = iter([1., 0.5, 0.5, 0.6])
    states: list[dict[str, torch.Tensor]] = []

    def validation(current: torch.nn.Module, value: torch.Tensor) -> dict[str, float]:
        assert not torch.is_grad_enabled() and not current.training
        states.append({key: tensor.clone() for key, tensor in current.state_dict().items()})
        return {"validation_mse": next(losses)}

    result = fit(model, optimizer, x, x, lambda current, value: current(value).square().mean(),
                 validation, {"max_epochs": 10, "mode": "min", "patience": 2,
                              "min_delta": 0., "monitor": "validation_mse"}, tmp_path)
    assert result["best_epoch"] == 1 and len(result["history"]) == 4
    checkpoint = torch.load(result["checkpoint"], weights_only=True)
    assert set(checkpoint) == {"model_state", "epoch", "score"}
    assert (tmp_path / "history.csv").is_file()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, states[1][key])


def test_cosine_task_scores_validation_only_in_isolated_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extracted orchestration must not score held-out pairs or rewrite input assets."""
    import tasks.link_prediction as pipeline
    import utils.experiment as experiment

    root, output = tmp_path / "fixture_repo", tmp_path / "output"
    (root / "configs").mkdir(parents=True)
    for filename in ("data_protocol.json", "cosine_baseline.json"):
        (root / "configs" / filename).write_bytes((ROOT / "configs" / filename).read_bytes())
    config = json.loads((root / "configs/data_protocol.json").read_text())
    full = np.array([np.arange(20), np.arange(1, 21)], dtype=np.int64)
    # Generate a tiny synthetic fixture, never the repository's canonical split.
    arrays = make_split(full, 30, config)
    data = SimpleNamespace(x=torch.eye(30), edge_index=torch.from_numpy(bidirectional(full, 30)),
                           num_nodes=30)
    fingerprints = {"features": array_hash(data.x.numpy()), "full_pairs": array_hash(full)}
    manifest = {"split_id": config["split_id"], "config": config, "num_nodes": 30,
                "input_fingerprints": fingerprints, "split_hash": mapping_hash(arrays),
                "array_hashes": {key: array_hash(value) for key, value in arrays.items()},
                "counts": {key: value.shape[1] for key, value in arrays.items()}}
    split_dir = root / "artifacts/splits" / config["split_id"]
    split_dir.mkdir(parents=True)
    np.savez_compressed(split_dir / "edges.npz", **arrays)
    (split_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "results/week02").mkdir(parents=True)
    (root / "results/week02/leakage_audit.json").write_text(json.dumps({
        "status": "passed", "split_hash": manifest["split_hash"],
        "input_fingerprints": fingerprints,
    }), encoding="utf-8")
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    expected = np.concatenate([arrays["val_pos"], arrays["val_neg"]], axis=1)
    calls: list[np.ndarray] = []

    def score_validation(features: np.ndarray, pairs: np.ndarray, epsilon: float, chunk: int) -> np.ndarray:
        np.testing.assert_array_equal(pairs, expected)
        calls.append(pairs.copy())
        return cosine_scores(features, pairs, epsilon, chunk)

    monkeypatch.setattr(pipeline, "load_cora", lambda *args: data)
    monkeypatch.setattr(pipeline, "cosine_scores", score_validation)
    monkeypatch.setattr(experiment, "environment", lambda root: {
        "code_revision": "synthetic", "dirty_worktree": False,
    })
    result = pipeline.run_cosine_validation(root, output_root=output, require_cached=False)
    assert len(calls) == 1 and result["test_value"] is None
    assert result["evaluation_split"] == "validation"
    assert result["validation"]["roc_auc"] == 0.5
    assert result["validation"]["average_precision"] == 0.5
    log_path = output / "results/week03/cosine_validation.json"
    assert json.loads(log_path.read_text(encoding="utf-8")) == result
    assert (output / "results/experiment_index.csv").is_file()
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
