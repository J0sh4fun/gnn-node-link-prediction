"""Check editable-package entry points and artifact destination guardrails."""

from pathlib import Path
import subprocess
import sys

import pytest

from utils.paths import isolated_output_root, require_cora_cache


@pytest.mark.parametrize("module", ["scripts.mlp_baseline", "scripts.cosine_baseline", "scripts.smoke"])
def test_cli_help_outside_checkout(module: str, tmp_path: Path) -> None:
    """Editable installation makes module CLIs independent of cwd."""
    result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_missing_cache_is_rejected_without_creating_files(tmp_path: Path) -> None:
    """Smoke preflight never triggers a download or creates a cache."""
    with pytest.raises(FileNotFoundError, match="no download attempted"):
        require_cora_cache(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_reference_outputs_are_protected(tmp_path: Path) -> None:
    """CLI output roots cannot shadow reference records, datasets, or saved splits."""
    root = tmp_path / "repo"
    for path in (root, tmp_path, root / "results", root / "artifacts/splits/new"):
        with pytest.raises(ValueError):
            isolated_output_root(path, root)
    assert isolated_output_root(root / "runs/check", root) == (root / "runs/check").resolve()
