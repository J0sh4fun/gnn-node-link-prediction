# Phase 1 workspace cleanup and Phase 2 handoff

Cleanup branch: `chore/phase1-workspace-cleanup`, created from integrated `main`
at `0e39a9fd`. No commit, push, merge, branch deletion, cache removal, dataset
download, canonical split regeneration, or held-out Cora evaluation was performed.
The initial tracked/untracked working tree was clean; ignored datasets, local
environment, caches, and checkpoint files were left in place. The environment
later received the editable package and its missing declared notebook dependencies;
scientific packages and the base interpreter were not upgraded.

## Structure and compatibility

The existing top-level `models`, `train`, and `utils` convention is retained.
`pyproject.toml` makes these and the new populated `layers`, `tasks`, and `scripts`
packages editable-installable. CLI modules are thin; task orchestration lives in
`tasks`. Existing `train.train_mlp`, `train.trainer`, `utils.data_loader`,
`utils.evaluate`, `utils.graph_ops`, and `models.mlp` imports remain valid.
The old direct MLP entry point works after editable installation; its `sys.path`
mutation was removed. Dependencies and environment version constraints were not
changed. Installation is intended for an editable research checkout, not a
standalone wheel containing datasets/configs.

`Plan_GNN.md` was absent from integrated main. The existing full plan remains
byte-for-byte at `docs/plans/project_plan.md`; a root `Plan_GNN.md` now links to it.
No `AGENTS.md` was found in the repository or inspected parent directories.

| Original location | Canonical location / cleanup action | Compatibility |
|---|---|---|
| `notebooks/shared/graph_ops.ipynb` | `utils/link_graph.py`: NumPy edges, split, sampling, audit, raw loader | Original notebook is an import adapter |
| `notebooks/shared/metrics.ipynb` | `utils/link_metrics.py`: cosine scores and ROC-AUC/AP | Original names and defaults retained |
| `notebooks/shared/runtime.ipynb` | `utils/experiment.py`: seed, hashes, run provenance, publication | Adapter retained; optional output root added |
| `notebooks/shared/training.ipynb` | `train/callback_trainer.py`: toy/callback training | Original callback contract retained |
| W4 graph notebook `ProjectedSum` | `layers/projected_sum.py` | Notebook imports class; no new layer algorithm |
| W4 attention notebook `receiver_softmax` | `utils/attention.py` | Notebook imports existing toy helper |
| W3 cosine notebook data/scoring/publication cells | `tasks/link_prediction.py` | Notebook calls task; default historical publication paths retained |
| Existing MLP runner | `train/train_mlp.py`, thin `scripts/mlp_baseline.py` CLI | Fixed full-baseline defaults and JSON schema retained |
| Missing `runs/run_all.py` README command | Actual `scripts.cosine_baseline` / `scripts.smoke` modules | No fabricated replacement historical script |

The table above records original extraction locations. A subsequent focused
notebook reorganization moves the walkthroughs into purpose-based directories;
see the [current index and migration table](../notebooks/README.md). Report links
were updated; saved manifests/results keep their original provenance paths.
Reusable implementations remain in Python packages. Notebook outputs are
historical, not evidence of execution after either refactor. The four optional
import references now live in `notebooks/04_framework_validation/`; active
notebooks use explicit package imports, with no notebook-to-notebook imports.

## Semantic comparison and decisions

| Area | Node classification implementation | Link/toy implementation | Decision |
|---|---|---|---|
| Features/cache | `get_cora_data`: optional L1 normalization, `ToUndirected`; direct root `data/` | `load_cora`: raw features, direct root `data/cora/` | Keep both; LP optional cache override requires matching fingerprints |
| Edge representation | Torch `[2,E]`, device-aware, preserves non-loop multiplicity | NumPy int64 `[2,M]`, canonical `u<v`, sorted unique pairs | Keep separate APIs |
| Self-loops | Torch helper keeps non-loop entries/order and supplies weights | NumPy helper unique-sorts all entries including non-loops | Do not substitute helpers |
| Normalization | Edge-based symmetric coefficients and coalesced sparse SpMM | No GCN normalization in LP split helpers | Retain `utils/graph_ops.py` unchanged |
| Seeding | `set_seed`: CPU single thread plus deterministic/cuDNN flags | `seed_everything`: NumPy/Python/Torch, optional CUDA seed, deterministic algorithms | Preserve profiles; do not change thread/RNG behavior |
| Metrics | Masked argmax accuracy/macro-F1; labels inferred by sklearn | Continuous NumPy scores; both labels required; AUC/AP plus counts | Separate contracts; no thresholding or label-policy changes |
| Selection | MLP and NC trainer: strict validation-loss decrease, one-based epochs | Callback trainer: min/max, `min_delta`, zero-based epochs | No trainer replacement |
| Checkpoints | MLP state dict; NC trainer in-memory copy + atomic exported state dict | Callback checkpoint has `model_state`, `epoch`, `score`; ordinary save | Preserve formats and restore behavior |
| Logging | MLP JSON atomic; NC trainer status/history JSON atomic | B JSON/CSV ordinary writes and task-based index replacement | No silent persistence-policy change |

Extraction preserves algorithm bodies, defaults, shapes, dtypes and device
semantics. `new_run` now accepts an optional output root; its default remains
`runs/<id>` within the repository. Provenance now also hashes package `.py` files,
because notebooks no longer contain all implementation source. Existing log
fields, reference records, manifest source hashes and split fingerprints are not
rewritten. This is an intentional extension of **new-run** provenance metadata.

The cosine notebook's existing toy checks are passed into its task call. The
CLI's `checks` contains only checks actually recorded by that invocation; shared
input/manifest/leakage assertions execute in both paths. The original optional
comparison against an earlier run's saved predictions is retained. Missing local
prediction files do not become a claimed independent reproduction.

## Protocol boundaries and open issues

- Node classification uses the complete undirected graph and fixed public node
  masks. Training loss sees only training labels; validation controls selection.
  Full baseline evaluation remains a single final held-out call after restore.
- LP canonicalization, RNG sequence, floor rounding, 1:1 unique negatives and
  exclusion of **all** positives remain unchanged. The saved split has 4,488 / 263 /
  527 positives, with equally many negatives. Train-edge bidirectional adjacency
  is the only adjacency accepted for tuning. Test pairs may be read for leakage
  audits; smoke/validation commands never compute test scores.
- `utils.evaluate.evaluate_link_prediction` remains a pre-existing
  `NotImplementedError` stub. B's working implementation is
  `utils.link_metrics.lp_metrics(labels, scores)`, including count fields. A
  future tensor adapter requires an explicit interface decision; this cleanup
  does not fill the stub or break its existing test.
- The planned A `MessagePassingLayer` base class is absent. The existing concrete
  prototype is B's `ProjectedSum`. A's Week 4 architecture documents choose
  sparse SpMM instead of `MessagePassing` hooks; reconcile that design with the
  plan before Phase 2. No missing GCN/GAT implementation was invented here.
- The old decision log describes A/B cache locations as a case-only difference.
  Their **Planetoid roots differ by a directory level**: A resolves to
  `data/Cora/...`; B to `data/cora/Cora/...`. Defaults remain unchanged. Smoke can
  explicitly reuse A's cache for LP and verifies raw features/graph fingerprints.
- Existing split/audit helpers use Python `assert`; running Python with `-O`
  disables those checks. This pre-existing validation limitation is documented,
  not silently changed to a new exception contract. Use ordinary Python.
- B's callback trainer and publication helpers are not atomic. Do not infer the
  NC trainer's atomicity guarantees apply to them. A harmonized persistence
  policy is separate work.
- Historical raw predictions under ignored `runs/` may be absent in a new clone.
  Reference summary JSON alone is not equivalent to replaying a run. Historical
  environment versions are evidence of the original runs, not versions verified
  on this cleanup machine; dependency ranges are not a lockfile.

## Verification record (2026-10-04)

The existing `.venv` works outside the agent sandbox. Initial launcher errors
resembled a missing Python executable; a subsequent path check returned access
denied, and approved execution outside the sandbox resolved the restriction.
System Python is a separate interpreter without the scientific dependencies.

Editable installation with `--no-deps --no-build-isolation` first failed because
the old local build tooling lacked `bdist_wheel`. The documented
`python -m pip install --no-deps -e .` succeeded using isolated build tooling.
An initial test run therefore failed only the three outside-checkout import
checks; all passed after installation. The first combined smoke completed NC
validation but failed LP provenance collection because `nbformat` was absent.
Installing the already declared `nbformat>=5.10.0` and `nbclient>=0.10.0` (and
their small dependencies) resolved this. No PyTorch upgrade or dataset download
was performed. These failed attempts did not modify historical artifacts.

Verification environment: Python 3.11.9, torch 2.7.1+cpu, torch-geometric 2.7.0,
NumPy 2.2.6, scikit-learn 1.7.2, pytest 8.4.2, nbformat 5.11.1, nbclient 0.11.0.
Commands below use the existing `.venv` interpreter where runtime is needed.

| Command/check | Outcome |
|---|---|
| `.venv/Scripts/python.exe -m pip install --no-deps -e .` | Passed; installed editable checkout |
| `.venv/Scripts/python.exe -m pytest tests/ -q` | **63 passed, 1 skipped in 6.92s**; skipped CUDA test on CPU; includes Cora cache/mask tests and outside-checkout CLI imports |
| `.venv/Scripts/python.exe -m scripts.smoke` | Passed; two-epoch NC validation and cosine validation; temporary outputs removed |
| `nbformat.validate` over all notebooks | All 13 schemas passed; notebooks were not re-executed |
| `python -m scripts.mlp_baseline --help`, cosine/smoke equivalents | All three passed from the checkout, without scientific imports |
| Python AST/JSON/TOML parsing and local Markdown links | Passed: 31 Python files, 13 notebooks, package TOML and updated local links |
| Extraction comparisons | 22 extracted function/class ASTs unchanged; two deliberate provenance/path extensions reviewed separately; all five MLP function/class ASTs unchanged |
| Before/after SHA-256 comparison of protected files | Passed: 42 files unchanged, including datasets, saved splits, checkpoints, reference results/reports, full plan and scientific modules |
| Notebook metadata/outputs comparison | Passed for all notebooks; no fabricated execution results |
| Cache and output-path preflight | Passed with standard-library Python: existing cache detected, missing cache caused no writes, protected destinations rejected |
| `git diff --check`, deletion/conflict scans | Passed; no tracked deletions or conflict markers (Git reports only expected LF/CRLF conversion notices) |

The two-epoch MLP smoke produced validation loss 1.942383050918579, accuracy
0.2520 and macro-F1 0.19630648927698244. These are smoke diagnostics, not a new
200-epoch baseline. Its checkpoint and JSON log round-tripped successfully.
Cosine validation reproduced ROC-AUC 0.8121701918489497 and AP
0.8248011097161196 on 263 positive and 263 negative pairs, matching the original
selected record. This confirms metric reproduction, not byte-for-byte equality
with unavailable historical prediction files.

Use the [README](../README.md) commands to repeat verification. Outside-checkout
imports require editable installation. Full notebook execution, CUDA behavior,
full MLP training and held-out scoring were intentionally not performed. Cora
integration tests inspect masks/features; synthetic tests verify final-evaluation
contracts without accessing held-out Cora labels. Download-free smoke checks
reuse existing data and write only temporary outputs.

Phase 2 starts from these preserved interfaces. Implement custom GCN/GAT in
separate work, decide the shared metric adapter and message-passing contract,
and retain strict validation-only model selection. The historical MLP test
accuracy remains 56.80%; cleanup creates no new benchmark result.
