# Week 4 — Training framework and sparse GCN architecture

**Owner:** Member A. **Scope:** the verified Week 4 node-classification infrastructure and the Week 5 custom GCN implementation plan.

This document describes the current [trainer](../../train/trainer.py), [graph operations](../../utils/graph_ops.py), [graph-operation tests](../../tests/test_graph_ops.py), and [evaluation API](../../utils/evaluate.py). The Week 4 handoff reports successful verification, 100% graph-operations test coverage, and verified gradients. That status is recorded as supplied by the team; it is not a new coverage measurement, nor does an unspecified coverage percentage establish both statement and branch coverage. The roadmap below is proposed work, not an already implemented GCN.

## 1. Modular training architecture

### Separation of responsibilities

`NodeClassificationTrainer` accepts a model, optimizer, cross-entropy criterion, epoch limit, patience, output paths, and JSON-serializable `run_config`. Model architecture determines how logits are computed; the trainer owns optimization, validation selection, checkpoint restoration, and final reporting.

| Component | Responsibility | Boundary |
|---|---|---|
| `nn.Module` | Transform node inputs into `[N, C]` raw logits | No split selection or metric computation inside the model |
| `get_cora_data` | Features, labels, public masks, undirected connectivity | Supplies data; does not choose a checkpoint |
| `NodeClassificationTrainer` | Training lifecycle and validation-loss selection | Receives the chosen model and optimizer |
| `utils/graph_ops.py` | Self-loops, symmetric coefficients, sparse aggregation | No labels, masks, optimizer, or trainable parameters |
| `utils/evaluate.py` | Task-specific metrics | Does not create splits or repair a leaking adjacency |

`_forward` dispatches to `model(x)` when `edge_index=None` and to `model(x, edge_index)` otherwise. It does not catch a model's internal `TypeError` and silently retry another signature. The same optimization lifecycle can therefore serve the MLP baseline, a custom GCN, and a future GAT.

`EpochResult` records training loss, validation loss, accuracy, and macro-F1 for each epoch. Immutable `TrainingResult` records the selected epoch, stopping epoch, best validation loss, corresponding validation metrics, stopping status, and complete epoch history. A late stopping epoch and the selected checkpoint are distinct concepts.

### Protocol enforcement

The optimization loss is

$$
\mathcal L_{\mathrm{train}}
=\frac{1}{|\mathcal V_{\mathrm{train}}|}
\sum_{i\in\mathcal V_{\mathrm{train}}}
\operatorname{CE}(z_i,y_i).
$$

`fit(x, edge_index, y, train_mask, val_mask)` accepts no `test_mask` and contains no reference to it. It rejects empty, malformed, or overlapping training/validation masks. Every epoch runs optimization in training mode, then validation in `eval()` under `torch.no_grad()`. Only a **strict reduction in validation loss** replaces the selected checkpoint; ties consume patience. Defaults are 200 epochs and patience 20.

A selected state is captured by `deepcopy(model.state_dict())`, including parameters and registered buffers. The trainer restores that state at completion. `evaluate_test(...)` restores it again before computing final metrics, rejects overlap with previously used training/validation nodes, and permits only one successful final evaluation per fit. If final logging fails, a retry publishes cached metrics rather than recomputing them.

These controls enforce the intended no-held-out-label-leakage protocol within the trainer. They cannot inspect how features or graph edges were constructed, or prevent an experimenter from starting fresh runs selected by test performance. Full-graph transductive access is allowed for node classification; held-out labels remain excluded from supervised loss. A second `fit` resets selection history but **does not reinitialize model weights or optimizer state**; use fresh model/optimizer instances for independent runs.

### Deterministic CPU execution and state persistence

**Determinism is a caller responsibility, not an automatic trainer feature.** Before constructing the model, the experiment entry point must seed Python `random`, NumPy, and PyTorch, place the model and tensors on CPU, set a fixed CPU thread count, and enable `torch.use_deterministic_algorithms(True)`. The existing `set_seed` in [`train/train_mlp.py`](../../train/train_mlp.py) implements this setup. The same seed, split, preprocessing, package versions, and hardware configuration must be recorded. PyTorch does not promise identical results across all platforms or releases. [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.7/notes/randomness.html)

Both `checkpoint_dir / "best_model.pt"` and the JSON log use `NamedTemporaryFile` in the destination directory, followed by close and `Path.replace`. Parent directories are created as needed and temporary files are cleaned up on failure. Paths are resolved at construction, and configuration cannot overwrite reserved result fields.

The log progresses through `running`, `fitted`, and `evaluated`; test metrics appear only in the final stage. Each file replacement is atomic under the filesystem's replacement semantics. The checkpoint and JSON are **not a joint transaction**, and the implementation does not provide an `fsync`-based power-loss durability guarantee. The in-memory selected state is authoritative for final evaluation; the exported state dictionary is not a full optimizer/RNG checkpoint for resuming training after process restart.

## 2. Mathematical foundations and sparse complexity

### From a first-order spectral filter to the implemented operator

For symmetric, nonnegative adjacency $A$, write the normalized Laplacian on the positive-degree subspace as

$$
L=I-D^{-1/2}AD^{-1/2}=U\Lambda U^\top.
$$

A spectral filter acts as $g_\theta(L)h=Ug_\theta(\Lambda)U^\top h$. The first-order construction and renormalization follow [Kipf and Welling, Sections 2.1–2.2](https://arxiv.org/pdf/1609.02907#page=2). Explicitly,

$$
g_\theta(L)h
\approx \theta_0h+\theta_1
\left(\frac{2}{\lambda_{\max}}L-I\right)h
\approx \theta_0h-\theta_1D^{-1/2}AD^{-1/2}h.
$$

The second approximation uses $\lambda_{\max}\approx2$. Tying $\theta_0=-\theta_1=\theta$ gives $\theta(I+D^{-1/2}AD^{-1/2})h$. The subsequent renormalization is an **operator replacement**, not an algebraic equality:

$$
I+D^{-1/2}AD^{-1/2}
\ \longmapsto\
\hat A=\tilde D^{-1/2}\tilde A\tilde D^{-1/2},
\qquad
\tilde A=A+I,\quad
\tilde d_i=\sum_j\tilde A_{ij}.
$$

After introducing feature-channel mixing,

$$
H^{(l+1)}=\sigma\!\left(\hat A H^{(l)}W^{(l)}\right).
$$

For `edge_index[:, e] = [j, i]`, an edge carries information from source $j$ to target $i$. Its matrix position is $(i,j)$:

$$
\hat a_e=\frac{\tilde w_{j\to i}}{\sqrt{\tilde d_i\tilde d_j}},
\qquad
Z_i=\sum_{j\to i}\hat a_e(H_jW).
$$

| Mathematical step | Implementation |
|---|---|
| Add self-connections | `add_remaining_self_loops` creates one loop per node for unweighted input and initializes all returned weights |
| Preserve supplied weights | `compute_symmetric_norm` retains existing weighted loops, adds missing loops of weight one, and sums weighted duplicates |
| Compute augmented degrees | `degree.index_add_(0, target, edge_weight)` after loop insertion and coalescing |
| Compute coefficients | `inv_deg[target] * edge_weight * inv_deg[source]` |
| Apply $\hat A$ | `sparse_spmm` flips source/target to COO row/column, coalesces, and calls `torch.sparse.mm` |

For loop-free Cora this is exactly $\tilde A=A+I$. For already augmented weighted input, the helper **adds only missing loops** and preserves existing diagonal values, including zero. Applying normalization to already normalized weights is not an idempotent operation. Undirectedness and equal aggregate reverse-edge weights are caller preconditions; the helper does not symmetrize or verify that symmetry.

For $Y=\hat AX$, reverse-mode differentiation gives

$$
\frac{\partial\mathcal L}{\partial X}
=\hat A^\top\frac{\partial\mathcal L}{\partial Y}.
$$

On an undirected graph $\hat A^\top=\hat A$. The implementation keeps this path attached to PyTorch autograd, enabling gradients through $X=HW$ into $W$. The verified input-gradient behavior should not be generalized into an untested guarantee for learning edge weights at zero-degree singularities.

### Sparse computational and memory costs

Let $m$ be the number of distinct undirected non-loop links and let $M=2m+N$ be the stored augmented entries. Equivalently, $M=|\tilde{\mathcal E}|$ when the edge set counts both directions and loops.

| Operation or storage | Sparse representation | Dense adjacency |
|---|---:|---:|
| Aggregation of $F$ channels | $\mathcal O(MF)$ | $\mathcal O(N^2F)$ |
| Adjacency storage | $\mathcal O(M)$ | $\mathcal O(N^2)$ |
| Node feature/output storage | $\mathcal O(NF)$ | $\mathcal O(NF)$ |
| Degree accumulation | $\mathcal O(M+N)$ | $\mathcal O(N^2)$ if scanning all entries |

Thus the usual $\mathcal O(|\mathcal E|F)$ claim refers to stored sparse entries, including self-loops; for an edgeless graph the $\mathcal O(NF)$ loop contribution must not be omitted. Coalescing can introduce sorting overhead, typically up to $\mathcal O(M\log M)$. The current helpers build/coalesce COO tensors on calls, so their total execution time is not just the SpMM arithmetic count.

Projecting features first gives the layer cost

$$
\mathcal O(NF_{\mathrm{in}}F_{\mathrm{out}})
+\mathcal O(MF_{\mathrm{out}}),
$$

in addition to normalization/coalescing. Parameter and activation storage remain necessary even when adjacency is sparse.

[The PyG Cora specification](https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.datasets.Planetoid.html) gives $N=2708$ and 10,556 stored edge entries. Unit loops yield $M=13,264$. One float32 dense adjacency contains 7,333,264 entries, approximately **27.97 MiB**. A minimal COO representation with two int64 indices and one float32 value per augmented entry requires approximately **0.253 MiB**, excluding temporary tensors and framework overhead. Dense aggregation addresses about 553 times as many matrix positions; this is not a promised wall-clock speedup.

Dense Cora adjacency can fit in memory on ordinary hardware. Its prohibition is an **architectural scalability constraint** from the project plan: avoid quadratic work, unnecessary adjacency copies, and an implementation that fails to generalize to larger sparse graphs. Dense $N\times F$ node features and trainable feature projections are allowed.

### Positive semidefiniteness and the strict spectral bound

**The positive-semidefinite object is the normalized Laplacian**
$\tilde L=I-\hat A$, **not necessarily $\hat A$**. Assume a finite nonempty graph, $\tilde A=\tilde A^\top$ with nonnegative entries, and a strictly positive self-loop $s_i=\tilde A_{ii}>0$ at every node. Define degrees from this same augmented adjacency, so $\tilde d_i>0$.

For any $z\in\mathbb R^N$, set $u=\tilde D^{-1/2}z$. Then

$$
z^\top\tilde Lz
=u^\top(\tilde D-\tilde A)u
=\frac12\sum_{i,j}\tilde A_{ij}(u_i-u_j)^2
\ge0.
$$

Hence $\tilde L\succeq0$. If $S$ is the diagonal matrix of added loop weights, then $(D+S)-(A+S)=D-A$: adding loops preserves the unnormalized Laplacian's positive semidefiniteness, and normalization is a congruence by an invertible positive diagonal matrix.

For the upper bound,

$$
z^\top(2I-\tilde L)z
=u^\top(\tilde D+\tilde A)u
=\frac12\sum_{i,j}\tilde A_{ij}(u_i+u_j)^2
\ge2\sum_i s_i u_i^2.
$$

Let $\alpha=\min_i(s_i/\tilde d_i)>0$. Therefore

$$
z^\top(2I-\tilde L)z\ge2\alpha\|z\|_2^2,
\qquad
0\le\lambda(\tilde L)\le2-2\alpha<2.
$$

This proves $\operatorname{spec}(\tilde L)\subset[0,2)$, without requiring connectedness. The corresponding spectrum of $\hat A$ lies in $(-1,1]$. It can contain negative eigenvalues: two nodes joined with weight two and unit loops have

$$
\hat A=\frac13\begin{bmatrix}1&2\\2&1\end{bmatrix},
\qquad
\operatorname{spec}(\hat A)=\{1,-1/3\}.
$$

**Why loop insertion must precede degree computation:** the quadratic identities require $\tilde D_{ii}=\sum_j\tilde A_{ij}$. With two nodes connected by a unit edge, using the old $D=I$ after adding unit loops produces $I-(A+I)=-A$, whose eigenvalues include $-1$. The PSD guarantee is lost.

**Boundary of the theorem:** the code permits explicitly zero weighted loops. Such input does not satisfy the strict-positive-loop assumption; a bipartite graph with zero loops may retain Laplacian eigenvalue 2. The inverse-degree infinity-to-zero safeguard ensures finite forward coefficients at zero degrees, but does not establish the strict spectral bound. Standard loop-free Cora augmented with unit loops satisfies the theorem; an originally isolated node becomes a unit self-connection with normalized Laplacian value zero. These spectral guarantees concern linear graph propagation, not global stability of arbitrary learned weights, nonlinearities, or deep stacks.

## 3. Interface contract and handoff with Member B

### Shared evaluation ownership

| Owner | Function and inputs | Output and obligations |
|---|---|---|
| Member A | `evaluate_node_classification(logits, targets, mask)`; `[N,C]`, `[N]`, boolean `[N]` | `{"accuracy": float, "macro_f1": float}`; argmax, aligned masking, detached CPU values; macro averaging with `zero_division=0` |
| Member B | `evaluate_link_prediction(pos_pred, neg_pred)`; nonempty one-dimensional continuous edge scores | `{"roc_auc": float, "average_precision": float}`; construct binary labels and compute ROC-AUC/AP without thresholding |

The node evaluator excludes classes absent from both masked targets and predictions under the current scikit-learn defaults. Both members must preserve that convention when reporting comparisons. The link evaluator remains a stub in the inspected checkout while Member B completes Week 4 Sessions 3–4. Replace its `NotImplementedError` test with metric tests when implementing it, while retaining all node-classification regression tests.

### Two transductive protocols with different held-out objects

| Aspect | Member A: node classification | Member B: link prediction |
|---|---|---|
| Held-out object | Node labels on fixed public masks | Positive and negative node pairs |
| Known nodes/features | Full Cora node set and features | Full node set and features |
| Tuning message-passing adjacency | Full undirected citation graph | Symmetrized **train positive edges only** |
| Validation use | Select checkpoint/hyperparameters using validation node labels | Select using held-out validation edges, never included in tuning adjacency |
| Final evaluation | Restore best validation checkpoint, evaluate test node mask once | Follow Section 3.3: retrain from scratch on train + validation positives, then evaluate test edges |
| Shared metrics | Accuracy and Macro-F1 | ROC-AUC and Average Precision |

The full graph is legitimate in the node-label transductive benchmark because connectivity is observed and labels are held out. In link prediction, the edge itself is the target; including a held-out edge in message passing leaks the answer. `NodeClassificationTrainer` does not implement edge losses or edge-split safeguards and must not be reused unchanged as a link-prediction trainer.

For Member B, split unordered pairs so reverse directions cannot cross partitions. Negative pairs must be absent from the full original positive graph, disjoint across splits, and sampled at the planned 1:1 ratio. Full-graph knowledge is used only to exclude positives from negative sampling, not as the encoder adjacency. Add self-loops and compute degrees **from the permitted adjacency for each phase**; never compute degrees on the full graph and then merely remove held-out coefficients. Exclude self-pairs from citation-edge targets. Invalidate normalization caches whenever the permitted edge set changes.

## 4. Implementation roadmap for Week 5: custom GCN

### Session 1 — Implement and verify `GCNLayer`

Create `models/gcn.py` with `GCNLayer(nn.Module)`. It owns a trainable matrix $W\in\mathbb R^{F_{\mathrm{in}}\times F_{\mathrm{out}}}$, optionally a bias vector, and Xavier/Glorot uniform initialization with zero bias.

Proposed interface:

~~~python
GCNLayer(in_channels: int, out_channels: int, bias: bool = True)
forward(x: torch.Tensor, edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None) -> torch.Tensor
~~~

The layer accepts raw adjacency weights. Compute normalized edges with `compute_symmetric_norm`, project `support = x @ weight`, then call `sparse_spmm`. If enabled, add bias **after aggregation**: $\hat A(XW)+b$. A biased linear projection before aggregation would instead produce $\hat A(XW+b)$, generally a different operation because $\hat A$ is not row-stochastic. Leave activation and dropout to the enclosing network.

Verify hand-computable graphs, isolated nodes, duplicate-loop handling, feature/weight dtype alignment, output shape, finite gradients through $x$ and $W$, and explicit reset behavior. No production or test dependency on `torch_geometric.nn.GCNConv` is required.

**Architecture decision:** this handoff requests `nn.Module` plus sparse SpMM. It implements the same local weighted message-passing sum mathematically, but does not implement the `MessagePassing.propagate/message` hooks described in the earlier plan. Record this implementation-route change in the project's design trace; do not claim that those hooks were implemented.

### Session 2 — Compose the two-layer `GCN`

Use `GCN(nn.Module)` with the trainer-compatible signature `forward(x, edge_index)`. Start with dimensions **1433 → 16 → 7**, and

$$
H=\operatorname{Dropout}\!\left(
\operatorname{ReLU}(\hat AXW^{(0)}+b^{(0)})
\right),
\qquad
Z=\hat AHW^{(1)}+b^{(1)}.
$$

Use dropout 0.5 and return raw logits; cross-entropy handles its own log-softmax. For the first correct implementation, pass the **same raw edges** to both layers, each computing its own normalization. Do not pass one layer's normalized coefficients to a second call expecting raw weights.

This recomputation has a measurable setup cost. A later optimization may normalize once and provide an explicitly named pre-normalized internal path; keep it separate from the raw-input contract. Disable cross-run caching initially. Any later cache must account for topology, weights, node count, dtype, device, and split phase, not merely tensor shape or edge count.

Verify `[2708,7]` output, deterministic evaluation mode, active training dropout, gradient propagation through both layers, and overfitting on a small training-only subset. GCN and GAT remain separate networks rather than mixing layer types.

### Session 3 — Integrate with the trainer and freeze the comparison protocol

Construct the custom GCN, Adam (learning rate 0.01, weight decay $5\times10^{-4}$), and cross-entropy after deterministic seed setup. Load normalized Cora with the unchanged public masks. Supply raw undirected `edge_index` to the trainer, with separate checkpoint and log paths such as `results/checkpoints/gcn_seed42/` and `results/gcn_seed42_log.json`.

Use 200 maximum epochs and patience 20 as the initial baseline configuration. Log validation histories, verify restoration and mask isolation on synthetic data, and tune only on validation. A Week 5 integration run is not permission for repeated test checks; retain the planned Week 7 comparison and evaluate test only once the relevant model configuration is frozen. Any extensions must be predeclared.

The reference target is approximately **81.5% test accuracy**, as reported for GCN in [Kipf and Welling, Table 3](https://arxiv.org/pdf/1609.02907#page=7). It is a literature reference, not a guaranteed score, performance ceiling, or acceptance threshold.

Against the finalized [MLP log](../../results/baseline_mlp_log.json):

| Metric | Finalized MLP | Custom GCN |
|---|---:|---:|
| Test Accuracy | 56.80% | Pending final evaluation; reference target approximately 81.5% |
| Test Macro-F1 | 53.87% | Pending final evaluation |

If the target is reached, the absolute accuracy difference is **24.70 percentage points**, not a 24.70% relative increase. Report the actual measured differences as $100(a_{\mathrm{GCN}}-0.568)$ and $100(f_{\mathrm{GCN}}-0.5386504117428538)$. RQ1 remains empirically unresolved until the custom GCN is evaluated under the matched split and selection protocol. Interpret a gain alongside architectural differences and the limitations of one seed; stronger attribution to structure requires a preplanned matched adjacency ablation, not test-driven experimentation.
