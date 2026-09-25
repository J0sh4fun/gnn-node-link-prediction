# From Spectral Graph Convolution to the GCN Formula

**A summary and explanation of _Semi-Supervised Classification with Graph Convolutional Networks_ — Thomas N. Kipf and Max Welling, ICLR 2017.**

Reference version: arXiv:1609.02907v4, dated February 22, 2017. The focus is Section 2, Equations (3)–(8), which motivate the propagation rule in Equation (2). Additional mathematical explanations clarify steps that the paper presents briefly.

> Key idea: the GCN formula is constructed through a sequence of approximations, restrictions on the filter family, and architectural choices. Not every step is an algebraically equivalent transformation; in particular, renormalization changes the propagation operator.

## 1. Problem Setup and Notation

### 1.1. Graphs and Node Features

Consider an undirected graph with nonnegative edge weights and no initial self-loops. The simplest case is an unweighted graph.

| Symbol | Meaning | Dimensions |
|---|---|---|
| $N$ | Number of nodes | Scalar |
| $A$ | Symmetric adjacency matrix | $N\times N$ |
| $D$ | Degree matrix, $D_{ii}=d_i=\sum_j A_{ij}$ | $N\times N$ |
| $X$ | Node feature matrix, with one row per node | $N\times C$ |
| $x$ | A single-channel graph signal | $N$ |
| $I_N$ | Identity matrix | $N\times N$ |

For an unweighted graph, $A_{ij}=1$ if an edge connects nodes $i$ and $j$, and $0$ otherwise.

The unnormalized Laplacian and symmetric normalized Laplacian are:

$$
\Delta=D-A,\qquad L=I_N-D^{-1/2}AD^{-1/2}.
$$

Assume $d_i>0$ throughout the derivation. If isolated nodes exist, a convention is needed, such as setting their corresponding entries in $D^{-1/2}$ to $0$ when using the expression above. After adding self-loops during renormalization, all updated degrees are positive.

Because $L$ is real and symmetric, it admits the decomposition:

$$
L=U\Lambda U^\top,\qquad U^\top U=I_N,
$$

where $\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_N)$ and $0\le\lambda_i\le2$. The eigenvectors form a graph Fourier basis; smaller eigenvalues correspond to signals that vary less across the graph structure.

### 1.2. Semi-Supervised Node Classification

Only a small subset of nodes has labels available for training. The goal is to use both features $X$ and structure $A$ to predict the labels of the remaining nodes.

A traditional approach adds Laplacian regularization to encourage connected nodes to have similar outputs. For a vector of outputs $f$:

$$
\mathcal L_{\mathrm{reg}}=f^\top\Delta f
=\frac12\sum_{i,j}A_{ij}(f_i-f_j)^2.
$$

For an output matrix $Q$:

$$
\mathcal L_{\mathrm{reg}}=\operatorname{tr}(Q^\top\Delta Q)
=\frac12\sum_{i,j}A_{ij}\|Q_i-Q_j\|^2.
$$

The factor $1/2$ appears because summing over all pairs $(i,j)$ counts each undirected edge twice. Equation (1) in the paper omits this factor; in the loss function, it can be absorbed into the regularization coefficient.

GCN incorporates graph structure directly into the model $f(X,A)$ instead of requiring a separate Laplacian regularization term in the loss. This does not mean that the model removes every smoothing bias or avoids weight regularization.

## 2. Spectral Convolution: The Starting Point

In the spectral approach, the Fourier transform of a signal $x$ is:

$$
\hat x=U^\top x.
$$

A diagonal spectral filter acts on the Fourier coefficients, followed by the inverse transform:

$$
g_\theta\star x=U g_\theta U^\top x.
$$

This is Equation (3) in the paper. When parameterizing the filter as a function of the eigenvalues, we write $g_\theta(\Lambda)$.

Direct evaluation presents two challenges:

- Multiplication by the dense eigenvector matrix $U$ costs $O(N^2)$ for one signal.
- Computing a full eigendecomposition of the Laplacian is expensive for large graphs.

The paper uses the spectral approach to motivate GCN; defining graph convolution does not always require a Fourier-domain formulation.

## 3. Chebyshev Approximation: Avoiding a Full Eigendecomposition

### 3.1. Rescaling the Spectrum

Define:

$$
\tilde\Lambda=\frac{2}{\lambda_{\max}}\Lambda-I_N,
\qquad
\tilde L=\frac{2}{\lambda_{\max}}L-I_N.
$$

For $\lambda_{\max}>0$, the largest eigenvalue of $L$, this maps the spectrum into $[-1,1]$.

The Chebyshev polynomials satisfy:

$$
T_0(t)=1,\qquad T_1(t)=t,
$$

$$
T_k(t)=2tT_{k-1}(t)-T_{k-2}(t),\qquad k\ge2.
$$

For a matrix argument, $T_0(\tilde L)=I_N$.

### 3.2. Parameterizing the Filter with a Polynomial

Approximate the filter using a polynomial of degree at most $K$:

$$
g_{\theta'}(\Lambda)\approx\sum_{k=0}^{K}\theta'_kT_k(\tilde\Lambda).
$$

There are $K+1$ coefficients: $\theta'_0,\ldots,\theta'_K$. By the properties of matrix polynomials:

$$
U T_k(\tilde\Lambda)U^\top=T_k(\tilde L),
$$

which gives:

$$
g_{\theta'}\star x\approx\sum_{k=0}^{K}\theta'_kT_k(\tilde L)x.
$$

These are Equations (4)–(5). The expression requires neither storing nor multiplying by $U$. In a learning model, the Chebyshev coefficients are learned directly; there is no need to construct a full spectral filter first and then approximate it.

### 3.3. Locality and Computational Cost

A degree-$K$ polynomial in the Laplacian combines information only from nodes at most $K$ edges away from the node of interest. This is the **$K$-localized** property.

The following vectors can be computed recursively:

$$
v_0=x,\qquad v_1=\tilde Lx,\qquad
v_k=2\tilde Lv_{k-1}-v_{k-2},
$$

and then combined as $\sum_{k=0}^{K}\theta'_kv_k$. No dense matrix powers need to be formed.

With a sparse representation, the cost for one signal is $O(K(|E|+N))$. The paper's notation $O(|E|)$ emphasizes linear scaling in the number of edges when $K$ is fixed and node-related costs are suppressed. Using this exact rescaling still requires knowing or estimating $\lambda_{\max}$, but not computing a full eigendecomposition.

## 4. The First-Order Restriction: Setting $K=1$

When $K=1$:

$$
g_{\theta'}\star x\approx\theta'_0x+\theta'_1\tilde Lx.
$$

The paper then uses $\lambda_{\max}\approx2$. Consequently:

$$
\tilde L\approx L-I_N=-D^{-1/2}AD^{-1/2},
$$

and:

$$
g_{\theta'}\star x
\approx\theta'_0x-\theta'_1D^{-1/2}AD^{-1/2}x.
$$

This is Equation (6), with two free parameters. The paper expects the learned parameters to adapt to the change in spectral scaling. The bound $\lambda_{\max}\le2$ does not imply that the largest eigenvalue is always equal to $2$.

**Meaning of “first-order”:** the filter is a first-degree polynomial in the Laplacian. This is not a Taylor expansion, and it does not mean that the entire layer remains linear after an activation function is added.

One layer propagates information across at most one edge. Stacking $k$ layers gives a receptive field of at most $k$ hops, but a multilayer network with nonlinearities is generally not equivalent to a degree-$k$ Chebyshev filter.

The authors' motivation is to reduce the complexity of each layer and potentially limit overfitting to local neighborhood structures. This does not guarantee that increasing depth will always improve performance.

## 5. Constraining Two Parameters to One

Set:

$$
\theta=\theta'_0=-\theta'_1.
$$

Substitution gives:

$$
g_\theta\star x
\approx\theta\left(I_N+D^{-1/2}AD^{-1/2}\right)x.
$$

This is Equation (7). The node's own signal and the aggregated neighbor signal share the same learned coefficient.

This is a **model constraint**, reducing both the parameter count and flexibility compared with the two-parameter filter. For multichannel signals, it leads to a shared transformation matrix for the self and neighbor components.

## 6. Renormalization: Replacing the Propagation Operator

### 6.1. Why Make This Change?

Let $S=D^{-1/2}AD^{-1/2}$. Since $L=I_N-S$, the operator before renormalization is:

$$
P=I_N+S=2I_N-L.
$$

The eigenvalues of $P$ lie in $[0,2]$. Under repeated multiplication by $P$, components associated with eigenvalues greater than $1$ can grow substantially, while those associated with eigenvalues below $1$ can decay.

In deep networks, this can contribute to numerical instability and gradient-related difficulties.

### 6.2. Adding Self-Loops and Normalizing Again

The authors replace $P$ using:

$$
\tilde A=A+I_N,\qquad
\tilde D_{ii}=\sum_j\tilde A_{ij}=d_i+1,
$$

$$
\hat A=\tilde D^{-1/2}\tilde A\tilde D^{-1/2}.
$$

The distinction is important:

$$
I_N+D^{-1/2}AD^{-1/2}
\;\longrightarrow\;
\tilde D^{-1/2}(A+I_N)\tilde D^{-1/2}.
$$

The arrow denotes **a choice to replace the operator**, not an equality. For a graph without initial self-loops:

$$
\hat A_{ii}=\frac{1}{d_i+1},\qquad
\hat A_{ij}=\frac{A_{ij}}{\sqrt{(d_i+1)(d_j+1)}}\quad(i\ne j).
$$

Renormalization changes both self and neighbor weights; in general, it is not equivalent to multiplying the original operator by a single constant.

### 6.3. Interpretation and Limitations

For an undirected graph with nonnegative weights, the spectrum of $\hat A$ lies in $[-1,1]$ and $\|\hat A\|_2\le1$. Therefore, the propagation operation alone does not increase the Euclidean norm of a signal.

However, this does not guarantee that the entire network avoids exploding or vanishing gradients: weight matrices, activation functions, and the effects of stacking many layers also matter.

Symmetric normalization is not ordinary arithmetic averaging: each row of $\hat A$ does not necessarily sum to $1$.

## 7. From a Single Channel to the GCN Formula

For $X\in\mathbb R^{N\times C}$ and $F$ output channels, introduce a parameter matrix $\Theta\in\mathbb R^{C\times F}$:

$$
Z=\hat A X\Theta
=\tilde D^{-1/2}\tilde A\tilde D^{-1/2}X\Theta,
\qquad Z\in\mathbb R^{N\times F}.
$$

This is Equation (8). The matrix $\hat A$ combines information across nodes, while $\Theta$ mixes feature channels using parameters shared across the graph.

Adding an activation function and separate weights for each layer gives:

$$
\boxed{
H^{(l+1)}=\sigma\!\left(\hat A H^{(l)}W^{(l)}\right),
\qquad H^{(0)}=X.
}
$$

This is the central propagation rule, Equation (2) in the paper. If $H^{(l)}$ has $C_l$ channels, then $W^{(l)}\in\mathbb R^{C_l\times C_{l+1}}$.

### Cost of a Single Layer

For dense $X$, computing $B=X\Theta$ first and then $Z=\hat A B$ costs:

$$
O(NCF)+O((|E|+N)F).
$$

Aggregating first and transforming features afterward instead costs $O((|E|+N)C+NCF)$. Sparse features can further reduce the cost.

The paper states $O(|E|FC)$ to emphasize scaling with the number of edges. An implementation-level analysis should account for matrix multiplication order and channel dimensions.

Storing the sparse graph structure requires $O(|E|+N)$ memory. Training also requires memory for features, intermediate activations, parameters, gradients, and optimizer states.

## 8. The Spatial View and the Connection to WL-1

### 8.1. The Node-Wise Formula

For an unweighted graph, let $\tilde d_i=d_i+1$. The GCN rule is equivalent to:

$$
h_i^{(l+1)}=\sigma\!\left(
\sum_{j\in\mathcal N(i)\cup\{i\}}
\frac{h_j^{(l)}W^{(l)}}{\sqrt{\tilde d_i\tilde d_j}}
\right).
$$

Each node receives information from itself and its neighbors, combines it using normalization coefficients, and applies a nonlinearity. For a weighted graph, the aggregation coefficient additionally includes $\tilde A_{ij}$ in the numerator.

Self-loops retain the node's own features in the aggregation. The degrees in the denominator must be measured **after adding self-loops**.

### 8.2. Connection to Weisfeiler–Lehman

Appendix A offers an intuitive interpretation: replace a discrete label-update operation with a differentiable, parameterized transformation.

A precise description of WL-1 can be written as:

$$
c_i^{(t+1)}=\operatorname{HASH}\!\left(
 c_i^{(t)},\{\!\{c_j^{(t)}:j\in\mathcal N(i)\}\!\}
\right).
$$

The notation $\{\!\{\cdot\}\!\}$ denotes a multiset, preserving both values and their multiplicities. The encoding function must distinguish different inputs. Appendix A's description as hashing a sum of colors is simplified; directly adding numeric color codes can lose information, as illustrated by $1+3=2+2$.

Both GCN and WL-1 update representations through neighborhoods, but this does not imply that GCN has the same discriminative power as WL-1. The later work of Xu et al. (2019) analyzes the expressive limitations of GCN and neighborhood aggregators; those results are not part of the 2017 paper.

Appendix A.1 illustrates a three-layer GCN with random weights and $X=I_N$ on the karate club network. This example shows that the propagation structure can produce useful embeddings before training; it does not establish a guarantee for every graph or initialization.

## 9. A Two-Layer GCN for Semi-Supervised Classification

Precompute $\hat A$ from the graph. The two-layer model is:

$$
Z=\operatorname{softmax}\!\left(
\hat A\operatorname{ReLU}(\hat AXW^{(0)})W^{(1)}
\right).
$$

Here:

- $W^{(0)}\in\mathbb R^{C\times H}$ maps input features to $H$ hidden channels.
- $W^{(1)}\in\mathbb R^{H\times F}$ maps hidden features to $F$ output classes.
- Softmax is applied row-wise, producing a class probability distribution for each node.

For the set of labeled training nodes $\mathcal Y_L$ and one-hot labels $Y$:

$$
\mathcal L_{\mathrm{sup}}
=-\sum_{i\in\mathcal Y_L}\sum_{f=1}^{F}Y_{if}\log Z_{if}.
$$

**Unlabeled nodes still participate in feature propagation.** Only nodes in $\mathcal Y_L$ contribute directly to the supervised loss. In the transductive setting, the structure and features of nodes whose labels are to be predicted are already present during training, but test labels are not used to optimize the model.

The paper uses full-batch training with Adam, dropout, L2 regularization on the first layer's weights, and validation-based early stopping. Thus, omitting a separate Laplacian regularizer does not mean omitting regularization altogether.

## 10. Experimental Results and the Scope of the Conclusions

### 10.1. Accuracy

The following values are classification accuracies (%), taken from Table 2 of the paper for the standard dataset splits used in the experiments:

| Method | Citeseer | Cora | Pubmed | NELL |
|---|---:|---:|---:|---:|
| ManiReg | 60.1 | 59.5 | 70.7 | 21.8 |
| SemiEmb | 59.6 | 59.0 | 71.1 | 26.7 |
| LP | 45.3 | 68.0 | 63.0 | 26.5 |
| DeepWalk | 43.2 | 67.2 | 65.3 | 58.1 |
| ICA | 69.1 | 75.1 | 73.9 | 23.1 |
| Planetoid* | 64.7 | 75.7 | 77.2 | 61.9 |
| **GCN** | **70.3** | **81.5** | **79.0** | **66.0** |

Planetoid* denotes the best-performing Planetoid variant selected for each dataset. GCN achieves the highest accuracy in this table. These conclusions are specific to the experimental setup; the paper also reports results on random splits and shows that performance varies with the split.

### 10.2. Training Time

Table 2 compares the time to convergence of GCN and Planetoid on the same hardware:

| Dataset | GCN | Planetoid |
|---|---:|---:|
| Citeseer | 7 seconds | 26 seconds |
| Cora | 4 seconds | 13 seconds |
| Pubmed | 38 seconds | 25 seconds |
| NELL | 48 seconds | 185 seconds |

GCN is faster on three datasets but slower on Pubmed. This table does not support a claim that GCN is faster than every baseline on every dataset.

### 10.3. Renormalization and Depth

In Table 3, the renormalization variant achieves the highest accuracy among the compared propagation models on Citeseer, Cora, and Pubmed. This provides empirical support for the architectural choice within the tested configurations.

Appendix B reports the best results with two or three layers in its depth experiments. Expanding the receptive field by adding layers does not guarantee higher accuracy.

## 11. Overview of the Derivation

| Step | Expression or Operation | Nature of the Step |
|---|---|---|
| 1 | $Ug_\theta(\Lambda)U^\top x$ | Definition of spectral convolution |
| 2 | $\sum_{k=0}^{K}\theta'_kT_k(\tilde L)x$ | Localized polynomial approximation/parameterization |
| 3 | Set $K=1$ | Restrict the filter to first order |
| 4 | Set $\lambda_{\max}\approx2$ | Simplify spectral scaling |
| 5 | $\theta'_0=\theta$, $\theta'_1=-\theta$ | Constrain the parameters |
| 6 | $I_N+D^{-1/2}AD^{-1/2}\to\hat A$ | Replace the operator through renormalization |
| 7 | $Z=\hat AX\Theta$ | Extend to multiple channels |
| 8 | $H^{(l+1)}=\sigma(\hat AH^{(l)}W^{(l)})$ | Construct a GCN layer |

The final formula combines three operations: propagation over the normalized graph with added self-loops, channel transformation using learned weights, and application of a nonlinearity.

## References

1. Kipf, T. N., & Welling, M. (2017). _Semi-Supervised Classification with Graph Convolutional Networks_. ICLR 2017. [arXiv:1609.02907v4](https://arxiv.org/abs/1609.02907v4). Main source: Sections 2–3, Tables 2–3, and Appendices A–B.
2. Hammond, D. K., Vandergheynst, P., & Gribonval, R. (2011). _Wavelets on graphs via spectral graph theory_. Applied and Computational Harmonic Analysis, 30(2), 129–150. Foundational source cited by the paper for the spectral and Chebyshev approaches.
3. Defferrard, M., Bresson, X., & Vandergheynst, P. (2016). _Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering_. NeurIPS 2016. Foundational source cited by the paper for ChebNet.
4. Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). _How Powerful are Graph Neural Networks?_ ICLR 2019. [arXiv:1810.00826](https://arxiv.org/abs/1810.00826). Supplementary source for the discussion of expressive power and WL-1; not part of Kipf and Welling's original derivation.
