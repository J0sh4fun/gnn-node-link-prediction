"""Evaluation metrics for node classification and future link prediction."""

import torch
from sklearn.metrics import accuracy_score, f1_score


def evaluate_node_classification(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    """Compute accuracy and macro-F1 for the nodes selected by a boolean mask.

    Args:
        logits: Class scores of shape ``[num_nodes, num_classes]``. Softmax
            is unnecessary because predictions are obtained with argmax.
        targets: Integer class labels of shape ``[num_nodes]``.
        mask: Boolean selection tensor of shape ``[num_nodes]``. All inputs
            must reside on the same device.

    Returns:
        ``accuracy`` and ``macro_f1`` as Python floats. Macro-F1 averages over
        classes present in the masked targets or predictions, following
        scikit-learn's default label selection. Classes absent from both
        are excluded; undefined scores use ``zero_division=0``.

    Raises:
        ValueError: If the mask selects no nodes.

    Notes:
        Selected tensors are detached and moved to CPU for scikit-learn.
        Callers should put their model in evaluation mode before producing
        logits; this function does not change model state.
    """
    predictions = logits.argmax(dim=-1)
    y_pred = predictions[mask].detach().cpu().numpy()
    y_true = targets[mask].detach().cpu().numpy()
    if y_true.size == 0:
        raise ValueError("mask must select at least one node")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def evaluate_link_prediction(
    pos_pred: torch.Tensor, neg_pred: torch.Tensor
) -> dict[str, float]:
    """Evaluate positive and negative edge scores (Member B implementation).

    Args:
        pos_pred: One-dimensional scores for positive edges. Larger scores
            must indicate greater confidence that an edge exists.
        neg_pred: One-dimensional scores for negative edges, on the same
            scale as ``pos_pred``. Both groups must be nonempty.

    Returns:
        Once implemented, a dictionary with ``roc_auc`` and
        ``average_precision`` as Python floats.

    Raises:
        NotImplementedError: Until Member B implements this function.

    Notes:
        TODO (Member B): Detach scores and move them to CPU, concatenate
        positive/negative scores, construct matching labels of one/zero,
        and compute scikit-learn's ``roc_auc_score`` and
        ``average_precision_score``. Use continuous scores without
        thresholding. The caller is responsible for preventing edge leakage.
    """
    raise NotImplementedError(
        "Member B must implement ROC-AUC and Average Precision evaluation."
    )
