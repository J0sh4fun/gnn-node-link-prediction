# Cora: Node Classification & Link Prediction

Phase 1 (Weeks 1–4) provides the feature-only MLP baseline, fixed link splits,
cosine validation baseline, sparse graph operations and training utilities.
Member A owns node classification/GCN; Member B owns link prediction/GAT.
Complete GCN/GAT models remain Phase 2 work.

Requirements: [Plan_GNN.md](Plan_GNN.md) links to the unchanged integrated plan.
See [cleanup and handoff](docs/phase1_cleanup.md) for semantic comparisons,
compatibility decisions, unresolved issues and verification limits.

## Structure

```text
.
├── README.md / Plan_GNN.md
├── pyproject.toml                 # Editable packages; requirements.txt dependencies
├── requirements.txt / environment.yml / pytest.ini
├── configs/                      # Existing LP protocol, cosine and toy-training config
├── models/mlp.py                 # Complete feature-only baseline
├── layers/projected_sum.py       # Existing Week 4 MessagePassing prototype
├── train/
│   ├── train_mlp.py              # Fixed MLP fitting and legacy entry point
│   ├── trainer.py                # NodeClassificationTrainer
│   └── callback_trainer.py       # Callback/toy trainer; separate semantics
├── tasks/
│   ├── node_classification.py    # Validation-only MLP/cache smoke workflow
│   └── link_prediction.py        # Saved-split audit and cosine validation
├── utils/
│   ├── data_loader.py / evaluate.py  # NC loader and masked metrics
│   ├── graph_ops.py              # Torch sparse symmetric normalization / SpMM
│   ├── link_graph.py             # NumPy edges, split/sampling, leakage audit
│   ├── link_metrics.py           # Raw-float64 cosine and ROC-AUC/AP
│   ├── attention.py              # Existing receiver-wise toy softmax
│   ├── experiment.py             # Run hashes, seed and publication helpers
│   └── paths.py                  # Checkout, cache and output preflight
├── scripts/                     # Thin CLIs: mlp_baseline, cosine_baseline, smoke
├── tests/                       # Existing suite plus extraction regression tests
├── notebooks/                   # Purpose-based walkthroughs importing package code
│   ├── 01_data_exploration/     # Complementary Cora EDA notebooks
│   ├── 03_link_prediction/      # Split, audit, cosine baseline and analysis
│   ├── 04_framework_validation/ # Import references and toy validation
│   └── 05_model_design/         # GAT design and check-in
├── docs/ / report/               # Plans, protocols, architecture and reports
├── artifacts/splits/cora_lp_v1/  # Tracked immutable edges.npz + manifest.json
├── data/                        # Ignored caches; README tracked
├── results/                     # Tracked selected historical records
│   └── checkpoints/             # Ignored local weights
└── runs/                        # Ignored runs, predictions and checkpoints
```

Existing package names are retained instead of adding a second source hierarchy.
Notebooks are grouped by purpose; the [notebook index](notebooks/README.md)
records original contributors/weeks and the old-to-new path mapping. There is
no empty node-classification notebook folder: its baseline remains in Python
modules. Report links use current paths; manifests/results retain historical
provenance paths. Reusable source is organized by responsibility.

## Setup

Use a working **Python 3.11** installation. From the repository, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -c "import torch, torch_geometric, models, tasks; print(torch.__version__, torch_geometric.__version__)"
```

On Linux/macOS activate with `source .venv/bin/activate`. Alternatively:

```text
conda env create -f environment.yml
conda activate gnn-cora
python -m pip install --no-deps -e .
```

Use that interpreter for commands and notebook kernels. Editable installation
supports imports outside the checkout without `PYTHONPATH` or `sys.path` changes.
Repository defaults resolve from the editable checkout; explicit relative CLI
paths resolve from the current directory. A standalone wheel does not include
the repository datasets/configs.

Existing dependency ranges are unchanged and are not a lockfile. Exact
floating-point reproduction may require the versions/hardware recorded in
historical run metadata.

**Cleanup verification:** editable installation succeeded; the full suite passed
with **63 passed, 1 skipped** (CUDA unavailable). Both temporary validation-only
smokes passed, and all 13 notebook schemas validated. Python execution required
access outside the agent sandbox; the existing virtual environment itself works.
See [verification details](docs/phase1_cleanup.md#verification-record-2026-10-04).

## Baseline commands

### Node classification: full MLP

```text
python -m scripts.mlp_baseline
```

This preserves seed 42, L1-normalized features, 1433→16→7 MLP, dropout 0.5,
Adam lr 0.01 / weight decay 5e-4, maximum 200 epochs and patience 20.
Validation loss selects the checkpoint; one final test evaluation follows restore.

This full experiment replaces the original
`results/checkpoints/best_mlp_val.pt` and `results/baseline_mlp_log.json`.
Use it for an intended baseline experiment, **not cleanup verification**.
`python -m train.train_mlp` and legacy `python train/train_mlp.py` remain
available after editable installation.

### Link prediction: cosine validation

From the checkout, explicitly reusing the existing NC cache:

```text
python -m scripts.cosine_baseline --data-root data --output-root runs/cosine-validation
```

The command verifies the saved split/config/features/edges, audits leakage, and
scores only validation pairs. It never regenerates the split. Raw features use
float64, epsilon `1e-12` and chunk size 256 from the existing JSON config.

Original run/result/index formats are written beneath the chosen output root:
`runs/<id>/`, `results/week03/cosine_validation.json` and
`results/experiment_index.csv`. Repository reference results remain untouched.
Without `--data-root`, LP retains its original direct Planetoid root
`data/cora/`, giving cache `data/cora/Cora/`. These CLIs require a complete
existing cache and fail before downloading if absent. Explicit reuse of `data/`
must pass manifest fingerprint checks. Use absolute CLI paths outside the checkout.

Cosine itself does not use message passing; its split audit enforces the
train-only adjacency required by the future encoder. Historical
[notebooks](notebooks/README.md) remain available, but can download data and
publish reference summaries at their original paths.

## Tests and validation-only checks

After editable installation, run unit/regression tests without loading Cora:

```text
python -m pytest --ignore=tests/test_data_loader.py -q
```

For all tests, first preflight the existing cache:

```text
python -c "from utils.paths import repository_root, require_cora_cache; require_cora_cache(repository_root() / 'data')"
python -m pytest tests/ -q
```

If downloads are prohibited, run pytest only after preflight succeeds: the
legacy loader fixture can otherwise download Cora. It checks masks/features
without scoring held-out predictions. Trainer tests use synthetic data.

Preferred cleanup smoke command:

```text
python -m scripts.smoke
```

This uses the existing `data/` cache, checks Cora dimensions/masks/normalization,
runs two MLP epochs with validation metrics, round-trips a checkpoint and JSON
log, audits saved LP splits, and computes cosine validation metrics. Outputs
live in an automatically removed temporary directory. There are no downloads,
canonical split regeneration, full training runs or held-out scores. Test pairs
are read solely for partition/exclusion audits.

To retain a short validation trace separately:

```text
python -m scripts.mlp_baseline --validation-only --max-epochs 2 --output-dir runs/mlp-smoke
```

Help works without importing scientific dependencies:

```text
python -m scripts.mlp_baseline --help
python -m scripts.cosine_baseline --help
python -m scripts.smoke --help
```

## Graph Convention

Cora is **undirected**. NC applies `ToUndirected()` to the full graph and uses
fixed public masks: 140 training, 500 validation and 1,000 test nodes. Features
are L1-normalized by default. Training loss sees only training labels;
validation selects checkpoints. Transductive access to features/topology is
distinct from access to held-out labels.

LP stores unique unordered pairs `u < v`, excluding self-loops. Reverse edges
belong to the same split. Seed 42 and floor rounding give 4,488 / 263 / 527
positive training/validation/test pairs with equally many unique negatives.
Negatives exclude every original positive and do not overlap across partitions.
Training/validation encoder adjacency contains only bidirectional `train_pos`;
held-out edges must not enter message passing.

Spectral propagation adds missing loops before computing degrees:

$$
\tilde A = A + I,\qquad
\hat A = \tilde D^{-1/2}\tilde A\tilde D^{-1/2}.
$$

`utils/graph_ops.py` uses edge weights and sparse SpMM, never dense N×N
adjacency. Torch and NumPy edge helpers have different duplicate-handling
contracts and are not interchangeable. See [protocol](docs/protocol.md) and
[semantic comparison](docs/phase1_cleanup.md#semantic-comparison-and-decisions).

## Artifacts and Phase 2 handoff

Datasets, environments, caches, checkpoints and generated `runs/`/`outputs/`
are ignored. Saved splits, manifests, configs, reports and selected `results/`
are intentional tracked assets. Ignore rules do not untrack files. Do not delete
or regenerate saved splits to make a run succeed; protocol changes require a
separately reviewed split ID. Historical source hashes describe the original
run and must not be rewritten to match refactored code.

Historical MLP test accuracy remains **56.80%**. Cosine validation ROC-AUC/AP
remain **0.812170 / 0.824801**. These are different tasks/metrics, not directly
comparable. The temporary cosine validation smoke reproduced those values;
the full MLP experiment and its held-out test evaluation were not rerun.

Use these existing interfaces to implement Phase 2 GCN/GAT separately. Resolve
the missing planned A `MessagePassingLayer` versus the documented sparse-SpMM
design first. Working LP metrics are in `utils.link_metrics.lp_metrics`;
the old tensor `evaluate_link_prediction` stub remains intentionally
unimplemented. See [handoff notes](docs/phase1_cleanup.md) for untouched issues.
