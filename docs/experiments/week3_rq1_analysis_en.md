# Week 3, Session 3 — MLP baseline and RQ1 analysis

**Experiment:** Cora semi-supervised node classification; feature-only, two-layer MLP; seed 42.  
**Evidence:** [final run log](../../results/baseline_mlp_log.json), [training script](../../train/train_mlp.py), [model](../../models/mlp.py), and the experiment summary supplied by Member A. The run is reported as reproducible under its fixed environment; this document does not constitute an independent rerun.

## 1. Section 4 progress table update

The following table updates the Node Classification results in [Section 4 of the project plan](../../Plan_GNN.md). Values are **final test metrics**, expressed as percentages. Validation metrics are reserved for checkpoint selection and are not substituted into this table.

| Model | Test Accuracy (%) | Test Macro-F1 (%) | Status |
|---|---:|---:|---|
| MLP — feature-only baseline | **56.80** | **53.87** | Finalized, Week 3, seed 42 |
| GCN — self-implemented | — | — | Layer implementation: Week 5; comparison: Week 7 |
| GAT — self-implemented | — | — | Layer implementation: Week 6; comparison: Week 7 |

The stored metrics are accuracy `0.568` and macro-F1 `0.5386504117428538`; table rounding does not alter the archived results. A dash denotes an experiment not yet completed, not a zero score or an estimated result.

The baseline uses 1,433 input features, 16 hidden units, seven output classes, ReLU, dropout 0.5, and Adam with learning rate 0.01 and weight decay 0.0005. The public Planetoid split provides 140 training nodes, 500 validation nodes, and 1,000 test nodes. Input features are row-normalized; “feature-only” means that adjacency is not an input to the classifier, rather than that preprocessing is absent.

## 2. Rigorous analysis of RQ1

**RQ1: Does graph structure improve node classification performance over raw bag-of-words features alone, and by how much?**

**Current answer:** the experiment establishes the feature-only reference point, but the project's empirical answer remains open until its own GCN/GAT results are available. A literature comparison motivates an expected benefit; it does not measure a graph-induced improvement in this repository.

### Limitations of lexical features

Cora's raw representation is a sparse binary word-presence vector, subsequently L1-normalized by the loader. It is therefore more precise to describe this input as normalized bag-of-words presence features than as raw word-frequency counts. The following mechanisms are theoretical explanations for possible errors, not conclusions from a completed error analysis:

- **Lexical ambiguity:** terms such as “network” or “learning” may occur in several research areas. A bag-of-words representation discards word order and contextual meaning, leaving an MLP to infer distinctions from sparse co-occurrences.
- **Synonymy:** papers addressing similar questions can use different vocabulary. Separate vocabulary coordinates do not encode semantic equivalence directly; learning such relationships from only 20 labeled nodes per class is difficult.
- **Interdisciplinary overlap:** articles from different classes can share methods and terminology. Conversely, articles in the same class can emphasize different subtopics, producing substantial within-class lexical variation.

The two-layer MLP can learn nonlinear combinations of words, so these limitations should not be interpreted as an inability to learn any semantic associations. Its defining limitation is that a node's prediction depends only on its own feature vector and shared model parameters. It cannot directly inspect citation neighbors or exploit their features at inference time.

### Citation-network inductive bias

The working hypothesis is that citation neighbors often share research topics, methods, or intellectual context. Under this form of homophily, neighborhood aggregation can supply contextual evidence when a paper's own vocabulary is ambiguous or sparse. GCN uses normalized aggregation, whereas GAT learns neighbor-dependent attention weights; both can combine node content with relational information.

This relational signal is **complementary**, not proven statistically orthogonal, to the word vectors. Two nodes with identical word vectors receive identical deterministic MLP predictions, but a graph model may distinguish them through different neighborhoods. Citation links may also connect papers with little lexical overlap. Neither effect requires access to held-out labels.

Homophily is an inductive assumption, not a guarantee for every edge. Cross-topic citations and noisy links can introduce misleading messages. The magnitude of any benefit must therefore be measured under the shared undirected-graph convention rather than presumed from the existence of edges.

### Literature comparison and expected improvement

[Kipf and Welling (ICLR 2017), Table 3](https://arxiv.org/pdf/1609.02907#page=7) report Cora accuracy of **55.1% for an MLP** and **81.5% for their renormalized GCN**. These are averages over repeated initializations, whereas our **56.80%** is one fixed-seed result. The numerical difference from their MLP is **+1.70 percentage points**, without establishing statistically significant superiority or an exact replication.

Using 81.5% as a reference target gives:

```text
Expected accuracy difference = 81.50% − 56.80%
                             = 24.70 percentage points.
```

Thus, a GCN reaching that reference would yield approximately **24–25 percentage points** of absolute improvement. This is not a 24–25% relative increase; the corresponding relative increase in accuracy would be approximately 43.5%.

**81.5% is a reference benchmark, not a performance ceiling for GCN/GAT.** [Veličković et al. (ICLR 2018), Table 2](https://arxiv.org/pdf/1710.10903#page=8) report **83.0 ± 0.7%** for GAT on Cora. Its central value is 26.20 percentage points above our baseline. Neither literature value should be entered as a result for the self-implemented models or used as a pass/fail threshold.

For Weeks 5–7, report the measured differences as:

```text
ΔAccuracy_pp = 100 × (Accuracy_GNN − 0.568)
ΔMacroF1_pp  = 100 × (MacroF1_GNN − 0.5386504117428538)
```

Retain the same split, feature normalization, metric implementation, and validation-only selection protocol. Record architecture and regularization differences when interpreting the comparison. An otherwise matched adjacency ablation would provide stronger attribution to graph structure than a comparison that changes several architectural choices simultaneously. Any additional seeds or ablations must be planned before their final test evaluations, not selected retrospectively from test performance.

## 3. Training dynamics and protocol audit

### Checkpoint selection and stopping

| Quantity | Observed value | Interpretation |
|---|---:|---|
| Best validation epoch | 183 / 200 | Reported in Member A's experiment summary |
| Best validation loss | 1.330289 | JSON value: 1.3302887678146362 |
| Actual stopping epoch | 200 | Maximum epoch budget reached |
| Patience | 20 | Configured, but not exhausted after epoch 183 |
| Validation accuracy at selected checkpoint | 58.80% | Not necessarily the maximum accuracy over all epochs |
| Validation macro-F1 at selected checkpoint | 56.82% | Selected by loss, not independently by F1 |
| Final test accuracy | 56.80% | Evaluated after restoring the selected checkpoint |
| Final test macro-F1 | 53.87% | Same restored checkpoint |

The run found its minimum validation loss late, at epoch 183. This is consistent with useful optimization continuing into the later training phase, but a minimum alone cannot establish a smooth or monotonic validation curve. The current JSON does not store per-epoch losses or `best_epoch`; the latter comes from the supplied run summary. Preserve the console trace or epoch history for a subsequent curve-based analysis.

**Stopping correction:** there are only **17** epochs from 184 through 200. With patience 20, the recorded run ended at the **200-epoch cap**, not because patience was exhausted. Under continued non-improvement, a patience-triggered stop would occur at epoch 203, beyond the configured budget. No extra training or test evaluation is needed to document the completed run.

### Generalization and overfitting

Validation accuracy exceeds test accuracy by **2.00 percentage points**. The macro-F1 difference is approximately **2.95 percentage points**. These moderate discrepancies are consistent with reasonably similar held-out performance and do not, by themselves, indicate a severe validation-to-test collapse.

They do **not prove the absence of severe overfitting**. Overfitting is primarily assessed through training-versus-held-out behavior, and neither a training curve nor a complete validation curve is archived in the supplied JSON. The validation set also participates in checkpoint selection, so some optimism relative to the untouched test set is expected. A single seed and distinct graph-dependent evaluation subsets do not establish statistical significance. The defensible conclusion is that the reported held-out metrics provide no clear evidence of a severe collapse, while a stronger diagnosis requires additional training diagnostics.

### Protocol audit and evidence boundaries

The intended **zero-leakage protocol** is supported by the following properties of the reviewed code:

| Control | Implementation evidence | Scope |
|---|---|---|
| Training-label isolation | Cross-entropy uses only `train_mask` | Validation/test labels do not enter the supervised loss |
| Validation-only selection | Every epoch uses `eval()`, `no_grad()`, and `val_mask`; minimum validation loss selects weights | No test metric participates in early stopping |
| Checkpoint integrity | `deepcopy(state_dict())` captures independent weights; `load_state_dict()` restores them | Later updates cannot overwrite the in-memory selected state |
| Strict test isolation | `fit_mlp()` accepts no test mask; `main()` evaluates `test_mask` once after restoration | One final evaluation per invocation |
| Feature-only baseline | `MLP.forward(x)` accepts no adjacency | Loading an undirected graph does not make the MLP a graph model |
| Label-independent preprocessing | Feature normalization is per node; public split masks are retained | No normalization statistics are fitted to test labels |
| Atomic JSON logging | A temporary file in the destination directory is closed and replaced into the final path; non-finite JSON values are rejected | Prevents publishing a partially written JSON file under the normal replacement semantics |

[Training-protocol tests](../../tests/test_train_mlp.py) cover checkpoint restoration, patience handling, final evaluation order, and logging on synthetic data. This is an implementation audit, not proof of the absence of leakage in all manual actions or every historical run. Atomic logging is a file-integrity mechanism; it does not itself demonstrate statistical validity or leakage prevention.

The finalized MLP test result must remain fixed during later development. The script permits one test evaluation per invocation, so team procedure must also prevent repeated test-driven tuning across invocations. Archive the JSON, selected checkpoint, configuration, and environment versions before further runs overwrite the same output paths.

## 4. Handoff and synchronization notes for Member B

### Node-classification evaluator status

The shared implementation is [`utils/evaluate.py`](../../utils/evaluate.py), with tests in [`tests/test_evaluate.py`](../../tests/test_evaluate.py). It applies argmax, selects targets and predictions with the same mask, and returns Python floats under `accuracy` and `macro_f1`. Macro-F1 uses `average="macro", zero_division=0`; a class absent from both masked targets and predictions is excluded under scikit-learn's default label selection.

**Verified handoff status: 100% pass rate among executed tests**, based on the pytest transcript supplied by Member A: **7 passed, 1 skipped in 126.42 seconds**. All **six CPU node-classification cases passed**. The seventh passing case verifies the unimplemented link-prediction stub; the CUDA case was skipped because CUDA was unavailable. This documentation records the supplied execution evidence; no independent rerun was completed in the documentation environment.

The suite contains six CPU node-classification cases, one CUDA case that is skipped on CPU-only systems, and one test asserting that the link-prediction stub raises `NotImplementedError`. Record passed, failed, and skipped counts separately. “100% passed” means all executed applicable cases passed; it does not mean 100% code coverage or successful execution of a skipped CUDA case.

Run and archive the verification without retraining or reevaluating the Cora test set:

```sh
python -m pytest tests/test_evaluate.py -v -ra --junitxml=results/test_evaluate_junit.xml
```

### Week 4 implementation contract

Member B should implement `evaluate_link_prediction(pos_pred, neg_pred)` inside the existing module for Week 4 validation of the link-splitting/evaluation infrastructure. This preparation does not bring forward the project's final link-prediction test evaluation.

1. Accept finite, one-dimensional, nonempty positive and negative edge scores on the same scale, with larger values indicating stronger evidence for an edge. Define and test the error behavior for empty inputs and non-finite values.
2. Detach tensors, transfer them to CPU, concatenate scores, and create aligned binary labels: ones for positives and zeros for negatives.
3. Compute `sklearn.metrics.roc_auc_score` and `sklearn.metrics.average_precision_score` on **continuous scores**, without argmax or thresholding. Average Precision is not a trapezoidal approximation to PR-AUC.
4. Return exactly `{"roc_auc": float, "average_precision": float}`. Preserve the node-classification API and behavior.
5. Replace the existing stub-exception test with numerical metric tests: perfect ranking, reversed ranking, tied scores, unequal positive/negative counts, and invalid inputs. Retain every node-classification regression test.
6. Keep data-splitting responsibilities outside the metric function: reverse directions of an undirected pair must remain in the same split; validation/test positive edges must not appear in tuning adjacency. Negative pairs must be absent from the full original graph and disjoint across splits, with the project's fixed 1:1 positive/negative ratio.
7. During tuning, evaluate validation edges with embeddings derived only from train adjacency. Reserve test edges for the final protocol; the planned final retraining on train + validation edges is a separate stage under Section 3.3.

Synchronize metric names, score orientation, negative-sampling ratio, and split identifiers before combining Member A's and Member B's result tables. Changing these conventions after viewing test performance would undermine comparability.
