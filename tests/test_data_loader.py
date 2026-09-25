"""Integration tests for Cora preprocessing and the public Planetoid split.

Run from the repository root with:
    python -m pytest tests/test_data_loader.py -v

The session fixture reuses the repository's data cache. The first run needs
network access if Cora has not been downloaded yet. Tests do not mutate the
shared Data object.
"""

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected

from utils.data_loader import get_cora_data


@pytest.fixture(scope="session")
def cora_data() -> Data:
    """Load normalized Cora once, using a working-directory-independent cache."""
    root = Path(__file__).resolve().parents[1] / "data"
    return get_cora_data(root=str(root), normalize_features=True)


def test_feature_and_label_shapes(cora_data: Data) -> None:
    """Cora must retain its standard node, feature, and label dimensions."""
    assert cora_data.num_nodes == 2708
    assert cora_data.x.shape == (2708, 1433)
    assert cora_data.y.shape == (2708,)
    assert cora_data.x.is_floating_point()
    assert cora_data.y.dtype == torch.long
    assert torch.equal(
        torch.unique(cora_data.y),
        torch.arange(7, device=cora_data.y.device),
    )


def test_feature_normalization(cora_data: Data) -> None:
    """Every Cora feature row must be finite, nonnegative, and sum to one."""
    assert torch.isfinite(cora_data.x).all().item()
    assert (cora_data.x >= 0).all().item()
    row_sums = cora_data.x.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums)), (
        "Expected every Cora feature row to have unit L1 norm; "
        f"maximum deviation: {(row_sums - 1).abs().max().item():.3e}"
    )


def test_mask_counts_and_isolation(cora_data: Data) -> None:
    """Public split masks must have the expected counts and be pairwise disjoint."""
    for name, expected_count in (
        ("train_mask", 140),
        ("val_mask", 500),
        ("test_mask", 1000),
    ):
        mask = getattr(cora_data, name)
        assert mask.dtype == torch.bool, f"{name} must be boolean"
        assert mask.shape == (2708,), f"{name} must contain one entry per node"
        assert mask.sum().item() == expected_count, f"Unexpected {name} count"

    train_mask = cora_data.train_mask
    val_mask = cora_data.val_mask
    test_mask = cora_data.test_mask
    assert not (train_mask & val_mask).any().item(), "Train/validation overlap"
    assert not (train_mask & test_mask).any().item(), "Train/test overlap"
    assert not (val_mask & test_mask).any().item(), "Validation/test overlap"

    train_class_counts = torch.bincount(cora_data.y[train_mask], minlength=7)
    assert torch.equal(train_class_counts, torch.full_like(train_class_counts, 20))
    # The public split intentionally leaves some nodes outside all three masks.
    assert (~(train_mask | val_mask | test_mask)).sum().item() == 1068


def test_undirected_graph(cora_data: Data) -> None:
    """Every edge must have a reverse edge after the transform pipeline."""
    assert cora_data.edge_index.dtype == torch.long
    assert cora_data.edge_index.ndim == 2
    assert cora_data.edge_index.shape[0] == 2
    assert cora_data.edge_index.shape[1] > 0
    assert cora_data.edge_index.min().item() >= 0
    assert cora_data.edge_index.max().item() < cora_data.num_nodes
    assert is_undirected(cora_data.edge_index, num_nodes=cora_data.num_nodes)


def test_raw_features_unnormalized() -> None:
    """Disabling normalization must preserve nonnegative bag-of-words features."""
    root = Path(__file__).resolve().parents[1] / "data"
    data = get_cora_data(root=str(root), normalize_features=False)

    assert data.x.shape == (2708, 1433)
    assert (data.x >= 0).all().item()
    row_sums = data.x.sum(dim=1)
    assert (row_sums > 1.0).any().item(), (
        "Raw bag-of-words features must include rows with L1 norm greater than one"
    )
