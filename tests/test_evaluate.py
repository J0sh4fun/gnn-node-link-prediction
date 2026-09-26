"""Unit tests for evaluation metrics using small, deterministic tensors."""

import warnings

import pytest
import torch
from sklearn.exceptions import UndefinedMetricWarning

from utils.evaluate import evaluate_link_prediction, evaluate_node_classification


def test_perfect_predictions() -> None:
    """Perfect masked predictions must score one and ignore unselected errors."""
    logits = torch.tensor(
        [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0], [5.0, 0.0, 0.0]],
        requires_grad=True,
    )
    targets = torch.tensor([0, 1, 2, 2])
    mask = torch.tensor([True, True, True, False])

    result = evaluate_node_classification(logits, targets, mask)

    assert result == {"accuracy": 1.0, "macro_f1": 1.0}
    assert all(isinstance(value, float) for value in result.values())
    assert logits.grad is None


def test_completely_inverted_predictions() -> None:
    """Swapping every binary label must yield zero accuracy and macro-F1."""
    logits = torch.tensor([[0.0, 4.0], [4.0, 0.0], [0.0, 4.0], [4.0, 0.0]])
    targets = torch.tensor([0, 1, 0, 1])
    mask = torch.ones(4, dtype=torch.bool)

    result = evaluate_node_classification(logits, targets, mask)

    assert result == {"accuracy": 0.0, "macro_f1": 0.0}


@pytest.mark.parametrize(
    ("selected_targets", "selected_predictions", "expected_accuracy", "expected_f1"),
    [
        pytest.param([0, 0], [0, 1], 0.5, 1.0 / 3.0, id="missing-in-targets"),
        pytest.param([0, 1], [0, 0], 0.5, 1.0 / 3.0, id="never-predicted"),
        pytest.param([0, 0], [0, 0], 1.0, 1.0, id="missing-in-both"),
    ],
)
def test_missing_class_in_masked_subset(
    selected_targets: list[int],
    selected_predictions: list[int],
    expected_accuracy: float,
    expected_f1: float,
) -> None:
    """Missing classes must produce finite metrics without undefined warnings."""
    # An unselected node contains class 2; it must not affect macro averaging.
    logits = torch.eye(3)[selected_predictions + [2]] * 5.0
    targets = torch.tensor(selected_targets + [2])
    mask = torch.tensor([True, True, False])

    with warnings.catch_warnings():
        warnings.simplefilter("error", UndefinedMetricWarning)
        result = evaluate_node_classification(logits, targets, mask)

    # In the imperfect cases, class F1 scores are 2/3 and 0, averaging to 1/3.
    assert result["accuracy"] == pytest.approx(expected_accuracy)
    assert result["macro_f1"] == pytest.approx(expected_f1)


def test_empty_mask_raises() -> None:
    """An empty evaluation subset must fail clearly instead of returning NaN."""
    logits = torch.tensor([[2.0, 0.0]])
    targets = torch.tensor([0])
    mask = torch.tensor([False])

    with pytest.raises(ValueError, match="mask must select at least one node"):
        evaluate_node_classification(logits, targets, mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_inputs() -> None:
    """GPU tensors must be transferred to CPU before scikit-learn evaluation."""
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]], device="cuda")
    targets = torch.tensor([0, 1], device="cuda")
    mask = torch.tensor([True, True], device="cuda")

    assert evaluate_node_classification(logits, targets, mask) == {
        "accuracy": 1.0,
        "macro_f1": 1.0,
    }


def test_link_prediction_is_not_implemented() -> None:
    """The Member B stub must fail explicitly rather than return fake metrics."""
    with pytest.raises(NotImplementedError, match="Member B"):
        evaluate_link_prediction(torch.tensor([0.9]), torch.tensor([0.1]))
