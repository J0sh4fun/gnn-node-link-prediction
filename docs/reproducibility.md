# Reproducing Phase 1 after integration

Use Python 3.11 and the unchanged dependency/environment files. After installing
dependencies, install the checkout with `python -m pip install --no-deps -e .`;
see the [README](../README.md). Select that environment as the notebook kernel.
Historical environment versions are provenance, not versions verified on the
current cleanup machine; the verification environment is recorded separately.

Reusable implementations now live in `utils/`, `train/`, `layers/` and
`tasks/`. Notebooks import packages directly. The four optional import references
now live in `notebooks/04_framework_validation/`; no notebook imports another
notebook. See the [current index](../notebooks/README.md) for the purpose-based
layout and original contributor/week mapping. Historical source hashes and
creator paths in saved manifests/results are not rewritten when files move.

For verification, use `python -m scripts.smoke`: it requires an existing cache,
uses temporary outputs and performs validation only. It neither downloads Cora,
regenerates canonical splits, nor evaluates held-out nodes/edges. Numerical
verification passed in the existing Python 3.11 virtual environment after
editable installation and adding its missing declared notebook dependencies.
The full suite returned 63 passed / 1 skipped (CUDA unavailable); both temporary
validation smokes passed. See [the verification record](phase1_cleanup.md#verification-record-2026-10-04).

The exploratory order remains W2.1 → W2.2 → W2.3 → W3.1 → W3.2, then W4.1/W4.2
and W4.3. Use **Restart Kernel and Run All**. W2.1 may download Cora; W2.2
computes and compares a split against the saved version. These are intentional
experiment workflows, not cleanup verification commands. W3.2 needs local saved
predictions from W3.1; selected JSON alone is insufficient. Notebook publication
still updates selected `results/` and its index. For isolated cosine validation,
prefer the CLI's required `--output-root`.

NC's direct Planetoid root is `data/` (cache `data/Cora/`); the original LP root
is `data/cora/` (cache `data/cora/Cora/`). They differ by a directory level,
even on a case-insensitive filesystem. The LP CLI can explicitly reuse NC's
cache via `--data-root data`, with raw feature/full-pair fingerprints checked
against the manifest. Do not migrate or delete caches during cleanup.

Saved split/config hashes, RNG order, preprocessing, metric definitions and
checkpoint policies are unchanged. Do not use `python -O`: legacy audits use
assertions. The trainers and seeding profiles remain separate because their
epoch numbering, improvement thresholds, device/thread settings and persistence
contracts differ. New provenance includes Python source hashes alongside
notebook/config hashes. Original manifests/results retain historical metadata.
