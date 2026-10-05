"""Repository and cache paths for editable research checkouts."""

from pathlib import Path


def repository_root() -> Path:
    """Locate the checkout independently of the current working directory."""
    root = Path(__file__).resolve().parents[1]
    if not (root / "configs/data_protocol.json").is_file():
        raise FileNotFoundError("Use an editable checkout containing configs/data_protocol.json")
    return root


def require_cora_cache(data_root: Path) -> None:
    """Fail before Planetoid can download or process an incomplete local cache."""
    cache = Path(data_root) / "Cora"
    raw_names = [f"ind.cora.{name}" for name in
                 ("x", "tx", "allx", "y", "ty", "ally", "graph", "test.index")]
    required = [cache / "raw" / name for name in raw_names]
    required.append(cache / "processed/data.pt")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("A complete existing Cora cache is required; no download attempted. "
                                "Missing: " + ", ".join(missing))


def isolated_output_root(output_root: Path, root: Path) -> Path:
    """Reject output roots that would overwrite the checkout's reference artifacts."""
    output = Path(output_root).expanduser().resolve()
    root = Path(root).resolve()
    protected = [root / name for name in ("data", "artifacts", "results", "report")]
    if output == root or output in root.parents or any(
        output == path or path in output.parents for path in protected
    ):
        raise ValueError("Choose a separate output root, e.g. runs/validation-check or a temp directory")
    return output
