"""Self-checks for the sparse, hand-written GAT layer."""

import pytest
import torch
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.utils import softmax as pyg_softmax

from models import GATLayer
from utils.attention import receiver_softmax


def test_receiver_softmax_matches_reference_and_has_finite_gradients() -> None:
    scores = torch.tensor(
        [[10000.0, -10000.0], [10001.0, -9998.0], [7.0, 9.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    targets = torch.tensor([1, 1, 2], dtype=torch.long)

    alpha = receiver_softmax(scores, targets, num_nodes=4)
    reference = pyg_softmax(scores, targets, num_nodes=4)

    torch.testing.assert_close(alpha, reference)
    torch.testing.assert_close(alpha[:2].sum(0), torch.ones(2, dtype=alpha.dtype))
    torch.testing.assert_close(alpha[2], torch.ones(2, dtype=alpha.dtype))
    assert receiver_softmax(
        torch.empty((0, 2)), torch.empty((0,), dtype=torch.long), 4
    ).shape == (0, 2)
    alpha.square().sum().backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


@pytest.mark.parametrize(
    ("heads", "concat", "expected_width"),
    [(1, True, 3), (4, True, 12), (4, False, 3)],
)
def test_output_shapes_and_attention_normalization(
    heads: int, concat: bool, expected_width: int
) -> None:
    x = torch.randn(4, 5)
    edges = torch.tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 0, 2]])
    layer = GATLayer(5, 3, heads=heads, concat=concat)

    output, (processed_edges, alpha) = layer(x, edges, return_attention=True)

    assert output.shape == (4, expected_width)
    assert processed_edges.shape == (2, 8)
    assert torch.equal(
        processed_edges[0] == processed_edges[1],
        torch.tensor([False, False, False, False, True, True, True, True]),
    )
    for receiver in range(4):
        incoming = processed_edges[1] == receiver
        torch.testing.assert_close(alpha[incoming].sum(0), torch.ones(heads))
    assert torch.isfinite(output).all()


def test_duplicate_input_self_loops_are_replaced_by_one_per_node() -> None:
    edges = torch.tensor([[0, 0, 0, 1], [0, 0, 1, 1]])
    original = edges.clone()
    layer = GATLayer(2, 2)
    _, (processed, alpha) = layer(torch.randn(3, 2), edges, return_attention=True)

    assert torch.equal(edges, original)
    loops = processed[0] == processed[1]
    assert processed[:, loops].shape[1] == 3
    assert set(processed[0, loops].tolist()) == {0, 1, 2}
    for receiver in range(3):
        incoming = processed[1] == receiver
        torch.testing.assert_close(alpha[incoming].sum(0), torch.ones(1))


def test_backward_reaches_input_and_every_trainable_parameter() -> None:
    x = torch.randn(5, 4, requires_grad=True)
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    layer = GATLayer(4, 3, heads=2)

    layer(x, edges).square().sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, parameter in layer.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_edge_order_permutation_does_not_change_evaluation_output() -> None:
    torch.manual_seed(3)
    x = torch.randn(6, 4)
    edges = torch.tensor([[0, 1, 2, 3, 4, 1, 5], [1, 2, 3, 4, 5, 4, 0]])
    layer = GATLayer(4, 3, heads=3).eval()
    permutation = torch.tensor([6, 2, 0, 5, 3, 1, 4])

    torch.testing.assert_close(layer(x, edges), layer(x, edges[:, permutation]))


def test_attention_weights_are_pre_dropout_and_dropout_only_changes_training() -> None:
    torch.manual_seed(13)
    x = torch.randn(8, 4)
    edges = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [1, 1, 3, 3, 5, 5, 7, 7]])
    layer = GATLayer(4, 3, heads=4, dropout=0.8, feature_dropout=0.0)

    layer.eval()
    eval_first, (_, alpha_eval) = layer(x, edges, return_attention=True)
    eval_second = layer(x, edges)
    torch.testing.assert_close(eval_first, eval_second)

    layer.train()
    train_first, (_, alpha_train) = layer(x, edges, return_attention=True)
    train_second = layer(x, edges)
    torch.testing.assert_close(alpha_train, alpha_eval)
    assert not torch.equal(train_first, train_second)


def test_feature_dropout_is_separate_and_disabled_in_evaluation() -> None:
    torch.manual_seed(17)
    x = torch.randn(12, 4)
    edges = torch.empty((2, 0), dtype=torch.long)
    layer = GATLayer(4, 3, feature_dropout=0.75, dropout=0.0)

    layer.eval()
    eval_first = layer(x, edges)
    torch.testing.assert_close(eval_first, layer(x, edges))
    layer.train()
    assert not torch.equal(layer(x, edges), layer(x, edges))


def test_edgeless_graph_and_empty_graph_are_supported() -> None:
    layer = GATLayer(3, 2, heads=2, concat=False)
    x = torch.randn(4, 3)
    edges = torch.empty((2, 0), dtype=torch.long)

    output, (processed, alpha) = layer(x, edges, return_attention=True)
    expected = layer.proj(x).view(4, 2, 2).mean(dim=1) + layer.bias
    torch.testing.assert_close(output, expected)
    assert processed.shape == (2, 4)
    torch.testing.assert_close(alpha, torch.ones((4, 2)))

    empty_output = layer(torch.empty((0, 3)), edges)
    assert empty_output.shape == (0, 2)
    assert torch.isfinite(empty_output).all()


def test_no_self_loop_mode_preserves_edges_and_zero_in_degree_outputs_zero() -> None:
    layer = GATLayer(2, 2, add_self_loops=False, bias=False)
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]])
    edges = torch.tensor([[0], [1]])
    output, (processed, alpha) = layer(x, edges, return_attention=True)

    assert torch.equal(processed, edges)
    torch.testing.assert_close(alpha, torch.ones((1, 1)))
    torch.testing.assert_close(output[0], torch.zeros(2))
    torch.testing.assert_close(output[2], torch.zeros(2))


def test_matches_pyg_gatconv_when_weights_and_conventions_match() -> None:
    torch.manual_seed(23)
    x = torch.randn(4, 5)
    edges = torch.tensor([[0, 1, 1, 2, 0, 0], [1, 0, 2, 1, 0, 0]])
    layer = GATLayer(5, 3, heads=2, concat=True, add_self_loops=True).eval()
    reference = GATConv(
        5, 3, heads=2, concat=True, dropout=0.0, add_self_loops=True, bias=True
    ).eval()
    with torch.no_grad():
        reference.lin.weight.copy_(layer.proj.weight)
        reference.att_src.copy_(layer.att_src)
        reference.att_dst.copy_(layer.att_dst)
        reference.bias.copy_(layer.bias)

    output, (processed, alpha) = layer(x, edges, return_attention=True)
    expected, (reference_edges, reference_alpha) = reference(
        x, edges, return_attention_weights=True
    )

    torch.testing.assert_close(processed, reference_edges)
    torch.testing.assert_close(alpha, reference_alpha)
    torch.testing.assert_close(output, expected)


def test_small_toy_graph_can_be_overfit() -> None:
    torch.manual_seed(41)
    x = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 0, 1])
    edges = torch.empty((2, 0), dtype=torch.long)
    layer = GATLayer(2, 2, heads=2, concat=False)
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        initial_loss = loss_fn(layer(x, edges), labels).item()
    for _ in range(80):
        optimizer.zero_grad()
        loss = loss_fn(layer(x, edges), labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = loss_fn(layer(x, edges), labels).item()

    assert final_loss < initial_loss * 0.1


@pytest.mark.parametrize(
    ("x", "edges", "message"),
    [
        (torch.ones(2, 3), torch.tensor([[0], [2]]), "outside"),
        (torch.ones(2, 3), torch.tensor([[0.0], [1.0]]), "torch.long"),
        (torch.ones(2, 3), torch.tensor([0, 1]), "shape"),
        (torch.ones(2, 4), torch.empty((2, 0), dtype=torch.long), "features"),
        (torch.ones(2, 3, dtype=torch.long), torch.empty((2, 0), dtype=torch.long), "floating"),
    ],
)
def test_invalid_inputs_raise_clear_errors(
    x: torch.Tensor, edges: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        GATLayer(3, 2)(x, edges)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"in_channels": 0, "out_channels": 2}, "in_channels"),
        ({"in_channels": 2, "out_channels": 2, "heads": 0}, "heads"),
        ({"in_channels": 2, "out_channels": 2, "dropout": 1.1}, "dropout"),
        (
            {"in_channels": 2, "out_channels": 2, "negative_slope": float("nan")},
            "negative_slope",
        ),
    ],
)
def test_invalid_constructor_arguments_raise(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        GATLayer(**kwargs)
