"""Custom Graph Attention Network (GAT) Layer.

Implements the multi-head graph attentional layer according to:
"Graph Attention Networks" (Veličković et al., ICLR 2018).

This implementation inherits from ``torch_geometric.nn.MessagePassing``,
manually implementing attention coefficient projection, neighborhood
softmax normalization, multi-head concat/average, and edge validation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, softmax


class GATLayer(MessagePassing):
    r"""Graph Attention Network (GAT) layer.

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_j,

    where the attention coefficients :math:`\alpha_{i,j}` are computed as:

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \,\|\, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \,\|\, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output head sample.
        heads (int, optional): Number of multi-head-attentions. (default: ``8``)
        concat (bool, optional): If set to ``False``, the multi-head
            attentions are averaged instead of concatenated. (default: ``True``)
        negative_slope (float, optional): LeakyReLU angle of the negative slope.
            (default: ``0.2``)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients. (default: ``0.0``)
        add_self_loops (bool, optional): If set to ``True``, will add a
            single self-loop per node to the input graph. (default: ``True``)
        bias (bool, optional): If set to ``False``, the layer will not learn
            an additive bias. (default: ``True``)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(aggr="add", node_dim=0, **kwargs)

        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError("in_channels must be a positive integer")
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise ValueError("out_channels must be a positive integer")
        if not isinstance(heads, int) or heads <= 0:
            raise ValueError("heads must be a positive integer")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        # Linear projection: W * x
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)

        # Decoupled attention parameters:
        # a^T [Wh_i || Wh_j] = a_dst^T Wh_i + a_src^T Wh_j
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))

        if bias:
            out_dim = heads * out_channels if concat else out_channels
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.register_parameter("bias", None)

        self._current_alpha: torch.Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset layer parameters with Glorot/Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _validate_inputs(self, x: torch.Tensor, edge_index: torch.Tensor) -> int:
        """Validate input tensor types, shapes, and index bounds."""
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or not x.is_floating_point():
            raise ValueError("x must be a 2D floating-point tensor of shape [num_nodes, in_channels]")
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected input feature dim {self.in_channels}, got {x.size(1)}")

        num_nodes = x.size(0)

        if (
            not isinstance(edge_index, torch.Tensor)
            or edge_index.ndim != 2
            or edge_index.size(0) != 2
            or edge_index.dtype != torch.long
        ):
            raise ValueError("edge_index must be a torch.long tensor of shape [2, num_edges]")

        if edge_index.numel() > 0:
            min_idx = edge_index.min().item()
            max_idx = edge_index.max().item()
            if min_idx < 0 or max_idx >= num_nodes:
                raise ValueError(
                    f"Edge indices must lie in [0, num_nodes). Got min {min_idx}, max {max_idx} with num_nodes={num_nodes}"
                )

        return num_nodes

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the GAT layer.

        Args:
            x (torch.Tensor): Node feature matrix [N, in_channels].
            edge_index (torch.Tensor): Graph edge indices [2, E].
            return_attention_weights (bool, optional): If True, returns a tuple
                (out, (edge_index_out, alpha)) containing normalized attention weights.

        Returns:
            torch.Tensor or (torch.Tensor, (torch.Tensor, torch.Tensor)): Output node
            representations of shape [N, heads * out_channels] (if concat=True) or
            [N, out_channels] (if concat=False).
        """
        num_nodes = self._validate_inputs(x, edge_index)

        # Handle self-loops
        if self.add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            loop_nodes = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
            loop_edges = torch.stack((loop_nodes, loop_nodes), dim=0)
            edge_index = torch.cat((edge_index, loop_edges), dim=1)

        # Linear projection: [N, heads, out_channels]
        h = self.lin(x).view(-1, self.heads, self.out_channels)

        # Precompute attention projections per node:
        # alpha_src = a_src^T (Wh) [N, heads]
        # alpha_dst = a_dst^T (Wh) [N, heads]
        alpha_src = (h * self.att_src).sum(dim=-1)
        alpha_dst = (h * self.att_dst).sum(dim=-1)

        self._current_alpha = None

        # Message passing:
        # In PyG source-to-target flow: edge_index[0] is source j, edge_index[1] is target i.
        # propagate passes alpha=(alpha_src, alpha_dst) where:
        # alpha_j corresponds to source node, alpha_i corresponds to target node.
        out = self.propagate(
            edge_index,
            h=h,
            alpha=(alpha_src, alpha_dst),
            size=(num_nodes, num_nodes),
        )

        alpha = self._current_alpha
        self._current_alpha = None

        # Multi-head combination
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            assert alpha is not None
            return out, (edge_index, alpha)

        return out

    def message(
        self,
        h_j: torch.Tensor,
        alpha_j: torch.Tensor,
        alpha_i: torch.Tensor,
        index: torch.Tensor,
        ptr: torch.Tensor | None,
        size_i: int | None,
    ) -> torch.Tensor:
        """Constructs attention-weighted messages from neighboring nodes.

        Args:
            h_j: Projected source node features [E, heads, out_channels].
            alpha_j: Source attention projection [E, heads].
            alpha_i: Target attention projection [E, heads].
            index: Target node indices for incoming messages [E].
            ptr: Optional index pointer tensor.
            size_i: Total number of target nodes.

        Returns:
            Attention-weighted messages [E, heads, out_channels].
        """
        # Unnormalized attention score: e_ij = LeakyReLU(alpha_dst_i + alpha_src_j)
        e = F.leaky_relu(alpha_i + alpha_j, self.negative_slope)

        # Grouped neighborhood softmax over incoming edges for each target node
        alpha = softmax(e, index, ptr=ptr, num_nodes=size_i)

        # Cache un-dropped attention coefficients for inspection / verification
        self._current_alpha = alpha

        # Apply attention dropout during training
        alpha_drop = F.dropout(alpha, p=self.dropout, training=self.training)

        # Weighted message: alpha_ij * Wh_j -> [E, heads, out_channels]
        return h_j * alpha_drop.unsqueeze(-1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, "
            f"heads={self.heads}, concat={self.concat})"
        )
