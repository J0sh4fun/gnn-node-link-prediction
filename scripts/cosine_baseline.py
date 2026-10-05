"""CLI for validation-only cosine scoring on the immutable saved edge split."""

import argparse
from pathlib import Path

from utils.paths import isolated_output_root, repository_root


def main() -> None:
    """Require an isolated output destination and a complete existing cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, help="Direct Planetoid cache root; default: repository data/cora/")
    args = parser.parse_args()
    root = repository_root()
    output = isolated_output_root(args.output_root, root)
    from tasks.link_prediction import run_cosine_validation
    result = run_cosine_validation(root, output_root=output, data_root=args.data_root)
    print(result["validation"])
    print("Run:", result["run_id"], "Output root:", output)


if __name__ == "__main__":
    main()
