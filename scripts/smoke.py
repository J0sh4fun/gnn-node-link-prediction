"""Run both validation-only checks in a temporary directory, without downloads."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from utils.paths import repository_root


def main() -> None:
    """Reuse one explicitly selected cache; LP fingerprints must still match."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Direct Planetoid cache root; default: repository data/")
    args = parser.parse_args()
    root = repository_root()
    data_root = (args.data_root or root / "data").resolve()
    from tasks.node_classification import smoke_node_validation
    from tasks.link_prediction import run_cosine_validation
    with TemporaryDirectory(prefix="cora-phase1-smoke-") as directory:
        output = Path(directory)
        print("Node validation:", smoke_node_validation(data_root, output / "node"))
        result = run_cosine_validation(root, output_root=output / "link", data_root=data_root)
        print("Link validation:", result["validation"])
        print("Smoke checks passed; temporary outputs will be removed.")


if __name__ == "__main__":
    main()
