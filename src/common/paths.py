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
