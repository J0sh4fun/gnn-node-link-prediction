"""CLI for the unchanged full MLP baseline or a validation-only smoke check."""

import argparse
from pathlib import Path

from utils.paths import isolated_output_root, repository_root


def main() -> None:
    """Parse CLI options before importing optional scientific dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-only", action="store_true", help="No held-out scoring; requires --output-dir")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-root", type=Path, help="Direct Planetoid cache root, default: repository data/")
    parser.add_argument("--max-epochs", type=int, default=2, help="Smoke only; full baseline remains 200 epochs")
    args = parser.parse_args()
    root = repository_root()
    if args.validation_only:
        if args.output_dir is None or not 1 <= args.max_epochs <= 5:
            parser.error("Validation smoke requires --output-dir and 1 <= --max-epochs <= 5")
        from tasks.node_classification import smoke_node_validation
        output = isolated_output_root(args.output_dir, root)
        print(smoke_node_validation(args.data_root or root / "data", output, args.max_epochs))
    else:
        if args.output_dir is not None or args.data_root is not None or args.max_epochs != 2:
            parser.error("Path/epoch overrides are smoke-only; add --validation-only")
        from train.train_mlp import main as run_baseline
        run_baseline()


if __name__ == "__main__":
    main()
