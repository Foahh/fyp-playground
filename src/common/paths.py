"""Centralized path resolution for datasets and tools."""

import os
import platform
from pathlib import Path


def get_repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]


def get_datasets_dir() -> Path:
    """Return the datasets directory, respecting DATASETS_DIR env var."""
    root = get_repo_root()
    return Path(os.environ.get("DATASETS_DIR", str(root / "datasets"))).expanduser()


def get_results_dir() -> Path:
    """Return the results root directory, respecting RESULTS_DIR env var."""
    root = get_repo_root()
    return Path(os.environ.get("RESULTS_DIR", str(root / "results"))).expanduser()


def resolve_repo_relative_path(rel: str | Path) -> Path:
    """Resolve paths from configs (e.g. model registry).

    Relative paths under ``results/`` are rooted at ``get_results_dir()`` so outputs
    can live on fast local storage (e.g. on HPC) while the repo stays on shared FS.
    Other relative paths are rooted at the repository. Absolute paths are returned
    expanded only.
    """
    p = Path(rel)
    if p.is_absolute():
        return p.expanduser()
    norm = str(p).replace("\\", "/")
    if norm == "results" or norm.startswith("results/"):
        suffix = norm[8:] if norm.startswith("results/") else ""
        base = get_results_dir()
        return (base / suffix) if suffix else base
    return get_repo_root() / p


def get_stedgeai_path() -> str:
    """Return platform-specific stedgeai executable path."""
    base = os.environ.get("STEDGEAI_CORE_DIR", "")
    system = platform.system()
    if system == "Windows":
        return os.path.join(base, "Utilities", "windows", "stedgeai.exe")
    elif system == "Linux":
        return os.path.join(base, "Utilities", "linux", "stedgeai")
    elif system == "Darwin":
        return os.path.join(base, "Utilities", "mac", "stedgeai")
    else:
        raise OSError(f"Unsupported platform: {system}")


def resolve_coco_root() -> Path:
    """Resolve COCO dataset root directory.

    Searches in order:
    1. $DATASETS_DIR/coco
    2. <repo>/datasets/coco
    3. ~/datasets/coco

    Returns the first path containing annotations/instances_val2017.json.
    """
    candidates = [
        get_datasets_dir() / "coco",
        get_repo_root() / "datasets" / "coco",
        Path.home() / "datasets" / "coco",
    ]
    for root in candidates:
        if (root / "annotations" / "instances_val2017.json").is_file():
            return root
    raise FileNotFoundError(
        "Unable to locate COCO root with annotations. Checked: "
        + ", ".join(str(c / "annotations" / "instances_val2017.json") for c in candidates)
    )
