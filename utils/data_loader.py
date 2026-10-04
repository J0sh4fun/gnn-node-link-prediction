"""Load Cora for transductive, semi-supervised node classification."""

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid


def get_cora_data(
    root: str = "data/", normalize_features: bool = True
) -> Data:
    """Return undirected Cora with its fixed public Planetoid masks.

    Args:
        root: Dataset cache directory, relative to the current working
            directory unless an absolute path is supplied. Missing data
            is downloaded automatically.
        normalize_features: If True, L1-normalize each nonzero row of Cora's
            nonnegative bag-of-words feature matrix to sum to one, following
            the feature preprocessing used by Kipf and Welling (2017).
            All-zero rows remain zero. If False, preserve the raw features.

    Returns:
        A PyG Data object with node features ``x``, labels ``y``, symmetric
        ``edge_index``, and boolean ``train_mask``, ``val_mask``, and
        ``test_mask``. The standard Cora split has 20 training nodes per
        class (140 total), 500 validation nodes, and 1,000 test nodes.

    Notes:
        PyG calls the standard Planetoid split ``"public"``; it does not
        accept ``split="planetoid"``. The original masks are preserved.

        Transforms run on access, not during cache preprocessing, so toggling
        normalization does not overwrite the cached raw feature values.
        ``NormalizeFeatures`` implements row normalization for Cora's binary
        features. No self-loops are added here; GCN/GAT layers handle them.

        The full graph is used for transductive message passing, but training
        loss must use only ``train_mask``. This graph is not an edge-split
        input for link prediction.
    """
    transforms: list[T.BaseTransform] = [T.ToUndirected()]
    if normalize_features:
        transforms.append(T.NormalizeFeatures(attrs=["x"]))

    dataset = Planetoid(
        root=root,
        name="Cora",
        split="public",
        transform=T.Compose(transforms),
    )
    return dataset[0]
